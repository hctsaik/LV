"""E2E 真實行為驗收:M19「🧪 差異探索」(Good/Bad 群組 patch 差異,第 10 工具)。

設計:3_Architect_Design/22_groupdiff_pipeline.md §5(AC-G1/G2/G3)。真實 DINOv2。

合成影像(答案已知才可客觀斷言):Good=灰底+細格線+每張獨立噪點;Bad=同底+**固定區域**
(56,56)-(112,112) 暗色瑕疵塊(位置固定、強度逐張抖動)。
- AC-G1:good vs bad → 「有穩定差異」banner + 匯出 regions.csv 的 rank1 區域中心落在植入區(±1 patch)。
- AC-G2:null(同源兩夾)→ 「未發現穩定差異」警示,成功文案不得出現。
- AC-G3:全程來源資料夾零寫入(檔案清單+大小+mtime 前後不變)。
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from .conftest import load_app, wait_idle

pytestmark = pytest.mark.e2e

BOX = (56, 56, 112, 112)          # 植入區域(像素;224 基準)
REL = (0.25, 0.25, 0.5, 0.5)      # 對應相對座標
TOL = 1 / 16 + 1e-9               # ±1 patch(224/14 → grid 16×16)

SUCCESS_TXT = "兩群存在穩定的局部差異"
NULL_TXT = "未發現穩定差異"


def _mk_img(rng, defect: bool) -> Image.Image:
    arr = rng.integers(110, 150, (224, 224, 3)).astype("uint8")
    arr[::28, :, :] = 165          # 細格線,給 DINOv2 一點空間結構
    arr[:, ::28, :] = 165
    if defect:
        x0, y0, x1, y1 = BOX
        block = rng.integers(20, 45, (y1 - y0, x1 - x0, 3)).astype("uint8")
        block[..., 0] = np.clip(block[..., 0].astype(int) + 30, 0, 255).astype("uint8")
        arr[y0:y1, x0:x1] = block
    return Image.fromarray(arr)


@pytest.fixture(scope="module")
def gpd_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("gpd")
    dirs = {}
    for name, n, defect, seed0 in (("good", 16, False, 0), ("bad", 12, True, 100),
                                   ("null_a", 8, False, 300), ("null_b", 8, False, 400)):
        d = root / name
        d.mkdir()
        for i in range(n):
            _mk_img(np.random.default_rng(seed0 + i), defect).save(d / f"{name}_{i:02d}.png")
        dirs[name] = d
    return dirs


def _snapshot(*dirs):
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
            for d in dirs for p in sorted(Path(d).rglob("*"))}


def _enter_tool(page, app_server):
    page.set_default_timeout(30000)
    load_app(page, app_server)
    # ⚠ 帶 emoji 的工具 label 用 exact=True 抓不到(Streamlit 拆 emoji 節點)——
    # 沿 test_fewshot_scenarios_e2e 的既證模式:去 emoji + exact=False + .first。
    # 「差異探索」字串在 .st-key-tool_switch 範圍內唯一(caption/popover 是外部兄弟節點)。
    page.locator('.st-key-tool_switch').get_by_text("差異探索", exact=False).first.click()
    wait_idle(page)
    page.wait_for_timeout(400)


def _fill(page, key: str, value):
    inp = page.locator(f'.st-key-{key} input').first
    inp.wait_for(state="visible", timeout=20000)
    inp.click()
    inp.fill(str(value))
    inp.press("Enter")
    wait_idle(page)
    page.wait_for_timeout(300)


def _analyze(page, good_dir, bad_dir) -> str:
    """填兩夾 → 按「🔬 分析差異」→ 等 verdict banner(成功或 null 警示,唯一訊號)。"""
    _fill(page, "gpd_good_dirs_add", good_dir)
    _fill(page, "gpd_bad_dirs_add", bad_dir)
    btn = page.locator('.st-key-gpd_run button')
    btn.wait_for(state="visible", timeout=30000)
    page.wait_for_function(
        """() => { const b = document.querySelector('.st-key-gpd_run button');
                   return b && !b.disabled; }""", timeout=30000)
    btn.click()
    page.wait_for_function(
        """([ok, no]) => { const m = document.querySelector('[data-testid="stMain"]');
                   if (!m) return false;
                   const t = m.innerText || '';
                   return t.includes(ok) || t.includes(no); }""",
        arg=[SUCCESS_TXT, NULL_TXT], timeout=600000)
    wait_idle(page, timeout=120000)
    page.wait_for_timeout(800)
    return page.locator('[data-testid="stMain"]').inner_text()


def test_g1_planted_difference_found_and_localized(app_server, browser, gpd_dirs,
                                                   tmp_path):
    """# AC-G1 + AC-G3:植入命中(檔案級驗證,防 DOM 假綠)+ 來源零寫入。"""
    before = _snapshot(gpd_dirs["good"], gpd_dirs["bad"])
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        _enter_tool(page, app_server)
        main = _analyze(page, gpd_dirs["good"], gpd_dirs["bad"])
        assert SUCCESS_TXT in main, f"應判定有穩定差異;實際:\n{main[:1500]}"
        assert NULL_TXT not in main

        # 匯出 → 用 regions.csv 驗 rank1 區域真的落在植入處
        out = tmp_path / "report"
        _fill(page, "gpd_out_dir", out)
        page.locator('.st-key-gpd_export button').click()
        csv_path = out / "regions.csv"
        deadline = time.time() + 120
        while time.time() < deadline and not csv_path.exists():
            time.sleep(0.5)
        assert csv_path.exists(), "按匯出後 regions.csv 應出現"
        wait_idle(page, timeout=60000)
        with open(csv_path, encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        assert rows, "有穩定差異時 regions.csv 至少一列"
        r1 = rows[0]
        cx = (float(r1["bbox_px_x0"]) + float(r1["bbox_px_x1"])) / 2
        cy = (float(r1["bbox_px_y0"]) + float(r1["bbox_px_y1"])) / 2
        assert REL[0] - TOL <= cx <= REL[2] + TOL, f"rank1 中心 x={cx:.3f} 不在植入區"
        assert REL[1] - TOL <= cy <= REL[3] + TOL, f"rank1 中心 y={cy:.3f} 不在植入區"
        assert (out / "region_01.png").stat().st_size > 0
        assert (out / "heatmap_good.png").stat().st_size > 0

        from playwright.sync_api import expect
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()
    assert _snapshot(gpd_dirs["good"], gpd_dirs["bad"]) == before, \
        "AC-G3:分析+匯出全程不得寫來源資料夾"


def test_g2_null_split_is_honest(app_server, browser, gpd_dirs):
    """# AC-G2 + AC-G3:同源隨機兩夾 → 明講「未發現穩定差異」,不給假線索。"""
    before = _snapshot(gpd_dirs["null_a"], gpd_dirs["null_b"])
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        _enter_tool(page, app_server)
        main = _analyze(page, gpd_dirs["null_a"], gpd_dirs["null_b"])
        assert NULL_TXT in main, f"同源兩夾應誠實判無差異;實際:\n{main[:1500]}"
        assert SUCCESS_TXT not in main, "null 資料不得宣稱找到穩定差異"

        from playwright.sync_api import expect
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()
    assert _snapshot(gpd_dirs["null_a"], gpd_dirs["null_b"]) == before
