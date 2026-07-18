"""E2E 真實行為驗收:M20「🩺 資料體檢報告」(第 11 工具)— 設計 25_audit_pipeline.md §4。

真實 DINOv2。合成植入(答案已知):可讀 25 張、壞檔 1、exact 額外複本 5、
洩漏對 1、同內容異標 1 組、val 缺 [classB,classC]、CSV bogus 5 列、2 張無列、
純色離群圖 1 張(AC-G2 斷言其進 outliers 前 10)。
"""
from __future__ import annotations

import csv as _csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from .conftest import load_app, wait_idle

pytestmark = pytest.mark.e2e

DONE_TXT = "體檢報告完成"


def _mk_img(path, seed, solid=None):
    if solid is None:
        rng = np.random.default_rng(seed)
        arr = rng.integers(60, 200, (224, 224, 3)).astype("uint8")
    else:
        arr = np.full((224, 224, 3), solid, dtype="uint8")
    Image.fromarray(arr).save(path)


@pytest.fixture(scope="module")
def audit_tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("audit_ds") / "ds"
    a = root / "train" / "classA"
    b = root / "train" / "classB"
    c = root / "train" / "classC"
    va = root / "val" / "classA"
    for d in (a, b, c, va):
        d.mkdir(parents=True)
    for i in range(8):
        _mk_img(a / f"A_{i:02d}.png", 10 + i)
    for i in range(3):
        shutil.copy(a / "A_00.png", a / f"dup_{i}.png")
    _mk_img(a / "outlier.png", 0, solid=140)
    for i in range(6):
        _mk_img(b / f"B_{i:02d}.png", 30 + i)
    shutil.copy(a / "A_01.png", b / "conflictcopy.png")
    for i in range(2):
        _mk_img(c / f"C_{i:02d}.png", 50 + i)
    for i in range(3):
        _mk_img(va / f"V_{i:02d}.png", 70 + i)
    shutil.copy(a / "A_02.png", va / "leakcopy.png")
    (va / "corrupt.png").write_text("not an image")

    csvf = root.parent / "meta.csv"
    rows = []
    for p in sorted((root / "train").rglob("*.png")):
        if p.parent.name != "classC":
            rows.append((p.name, "T1"))
    for p in sorted((root / "val").rglob("*.png")):
        if p.name != "corrupt.png":
            rows.append((p.name, "T2"))
    rows += [(f"bogus_{i}.png", "T1") for i in range(5)]
    with open(csvf, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["filename", "tool"])
        w.writerows(rows)
    return {"root": root, "csv": csvf}


def _snapshot(root):
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(Path(root).rglob("*"))}


def _fill(page, key, value):
    inp = page.locator(f'.st-key-{key} input').first
    inp.wait_for(state="visible", timeout=20000)
    inp.click()
    inp.fill(str(value))
    inp.press("Enter")
    wait_idle(page)
    page.wait_for_timeout(300)


def _sec(data, key):
    for s in data["sections"]:
        if s["key"] == key:
            return s
    raise AssertionError(f"report.json 缺節:{key}")


def test_audit_g1_g2_g3_end_to_end(app_server, browser, audit_tree, tmp_path):
    """# AC-G1 植入數字全對(檔案級)+ AC-G2 真實 DINOv2 離群命中 + AC-G3 零寫入。"""
    root, csvf = audit_tree["root"], audit_tree["csv"]
    before = _snapshot(root)
    ctx = browser.new_context(viewport={"width": 1800, "height": 1300})
    page = ctx.new_page()
    try:
        page.set_default_timeout(30000)
        load_app(page, app_server)
        page.locator('.st-key-tool_switch').get_by_text("資料體檢", exact=False).first.click()
        wait_idle(page)
        page.wait_for_timeout(400)

        _fill(page, "adt_root", root)
        _fill(page, "adt_csv", csvf)
        btn = page.locator('.st-key-adt_run button')
        btn.wait_for(state="visible", timeout=30000)
        page.wait_for_function(
            """() => { const b = document.querySelector('.st-key-adt_run button');
                       return b && !b.disabled; }""", timeout=30000)
        btn.click()
        page.wait_for_function(
            """(t) => { const m = document.querySelector('[data-testid="stMain"]');
                        return m && (m.innerText || '').includes(t); }""",
            arg=DONE_TXT, timeout=600000)
        wait_idle(page, timeout=120000)
        page.wait_for_timeout(800)
        assert DONE_TXT in page.locator('[data-testid="stMain"]').inner_text()

        out = tmp_path / "report"
        _fill(page, "adt_out_dir", out)
        page.locator('.st-key-adt_export button').click()
        jpath = out / "report.json"
        deadline = time.time() + 120
        while time.time() < deadline and not jpath.exists():
            time.sleep(0.5)
        assert jpath.exists(), "匯出後 report.json 應存在"
        wait_idle(page, timeout=60000)

        data = json.loads(jpath.read_text(encoding="utf-8"))
        # AC-G1:植入數字逐項全對
        assert data["totals"]["n_images"] == 25
        assert _sec(data, "unreadable")["value"] == 1
        assert _sec(data, "exact_dup")["value"] == 5
        assert _sec(data, "leakage")["value"] == 1
        assert _sec(data, "label_conflict")["value"] == 1
        cov = _sec(data, "class_coverage")
        assert cov["value"] == 2
        val_item = next(it for it in cov["items"] if it["split"] == "val")
        assert val_item["missing"] == ["classB", "classC"]
        meta_sec = _sec(data, "metadata")
        assert "5" in meta_sec["text"] and "2" in meta_sec["text"]
        # AC-G2:真實 DINOv2 — 植入純色圖進離群前 10;emb 節存在含 method
        out_items = _sec(data, "outliers")["items"]
        assert any(it["path"].endswith("outlier.png") for it in out_items), \
            f"純色離群圖應入前 10;實際:{[it['path'] for it in out_items]}"
        assert _sec(data, "emb_near_dup")["method"]
        assert (out / "report.html").stat().st_size > 0
        assert (out / "issues.csv").stat().st_size > 0

        from playwright.sync_api import expect
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()
    assert _snapshot(root) == before, "AC-G3:體檢+匯出全程不得寫來源資料夾"
