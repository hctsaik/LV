"""M22 GUI 接線真實 E2E:🧭 晶圓地圖(第 12 工具)。

設計:3_Architect_Design/M22_gui_wiring.md §5。真實 Streamlit+Playwright+真 DINOv2。
Fixture:50 張(42 normal+8 defect)兩片晶圓;W1 的 8 張 defect 在遠圈座標
(edge 植入,MC p 觸底)、W2 全 normal(k=0 → 看不出模式,deterministic);
CSV 故意缺 2 列+1 列缺 y(G4);G6 降級鏈逐層驗語彙。
"""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import pytest
from playwright.sync_api import expect

from ._anomaly_wizard import TAB_APPLY, apply_model, build_model, click_tab, enter_anomaly
from .conftest import wait_idle

pytestmark = pytest.mark.e2e


def _no_exception(page) -> None:
    assert page.locator('[data-testid="stException"]').count() == 0, \
        page.locator('[data-testid="stException"]').first.inner_text()[:400]


def _folder_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def wafer_dataset(tmp_path_factory):
    """42 normal(綠底)+ 8 defect(紅塊)YOLO 資料 + 兩份 CSV(主/無座標)。

    W1 = n00..n16(中心 ±2 網格)+ d00..d07(半徑 10 圓環,45° 間隔)。
    W2 = n17..n41;其中 n17/n18 不寫進 CSV(對不到列 2)、n19 的 die_y 留空
    (缺座標 1)→ W2 有座標 die = 22(≥ n_min 20)。lot 全 L1。
    """
    from PIL import Image
    root = tmp_path_factory.mktemp("wafer_ds")
    (root / "images").mkdir()
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("good\nbad\n", encoding="utf-8")
    W = H = 128
    box = (0.5, 0.5, 0.92, 0.92)

    def _write(stem, cls, defect):
        img = Image.new("RGB", (W, H), (0, 170, 0))
        if defect:
            for x in range(10, 74):
                for y in range(10, 74):
                    img.putpixel((x, y), (220, 0, 0))
        img.save(root / "images" / f"{stem}.jpg", quality=92)
        (root / "labels" / f"{stem}.txt").write_text(
            f"{cls} {box[0]} {box[1]} {box[2]} {box[3]}\n", encoding="utf-8")

    normals = [f"n{i:02d}" for i in range(42)]
    defects = [f"d{i:02d}" for i in range(8)]
    for s in normals:
        _write(s, 0, False)
    for s in defects:
        _write(s, 1, True)

    # 座標:W1 normal=中心 5×4 網格(x∈-2..2, y∈-2..1 取 17 點);defect=半徑 10 圓環
    w1_norm_coords = [(x, y) for y in range(-2, 2) for x in range(-2, 3)][:17]
    w1_def_coords = [(round(10 * math.cos(math.radians(a)), 3),
                      round(10 * math.sin(math.radians(a)), 3))
                     for a in range(0, 360, 45)]
    w2_coords = [(x, y) for y in range(-2, 3) for x in range(-2, 3)]  # 25 點

    rows = []
    for s, (x, y) in zip(normals[:17], w1_norm_coords):
        rows.append({"filename": f"{s}.jpg", "wafer": "W1",
                     "die_x": x, "die_y": y, "lot": "L1"})
    for s, (x, y) in zip(defects, w1_def_coords):
        rows.append({"filename": f"{s}.jpg", "wafer": "W1",
                     "die_x": x, "die_y": y, "lot": "L1"})
    for s, (x, y) in zip(normals[17:42], w2_coords):
        if s in ("n17", "n18"):
            continue                       # 對不到列 ×2(G4)
        y_val = "" if s == "n19" else y    # 缺座標 ×1(G4)
        rows.append({"filename": f"{s}.jpg", "wafer": "W2",
                     "die_x": x, "die_y": y_val, "lot": "L1"})
    main_csv = root.parent / "wafer_meta.csv"
    with open(main_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "wafer", "die_x", "die_y", "lot"])
        w.writeheader()
        w.writerows(rows)

    nocoord_csv = root.parent / "wafer_meta_nocoord.csv"
    with open(nocoord_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "note"])
        w.writeheader()
        w.writerows([{"filename": f"{s}.jpg", "note": "x"} for s in normals + defects])

    return {"root": root, "main_csv": main_csv, "nocoord_csv": nocoord_csv}


def _fill_input(page, key: str, value: str) -> None:
    inp = page.locator(f'.st-key-{key} input').first
    inp.fill(value)
    inp.press("Enter")
    page.wait_for_timeout(500)
    wait_idle(page, timeout=60000)


def _select_option(page, key: str, label: str) -> None:
    page.locator(f'.st-key-{key} [data-baseweb="select"]').click()
    page.get_by_role("option", name=label, exact=True).click()
    wait_idle(page, timeout=60000)


def _enter_wafer_tool(page) -> None:
    # pitfalls #8:emoji 拆節點 → 去 emoji + exact=False + 限 tool_switch 容器
    page.locator('.st-key-tool_switch').get_by_text("晶圓地圖", exact=False).first.click()
    wait_idle(page)


_GO_LIVE_JS = """() => {
    const bs = Array.from(document.querySelectorAll('.st-key-wmap_go button'))
        .filter(b => b.offsetParent !== null);      // 排除隱藏 ghost(pitfalls #11)
    return bs.find(b => !b.disabled) || null;       // ghost 順序不可靠 → 找 enabled 那顆
}"""


def _wait_go_enabled(page, timeout=15000) -> None:
    try:
        page.wait_for_function(
            f"() => !!({_GO_LIVE_JS})()", timeout=timeout)
    except Exception:
        diag = page.evaluate(
            """() => ({
                go: Array.from(document.querySelectorAll('.st-key-wmap_go button'))
                    .map(b => ({vis: !!b.offsetParent, dis: b.disabled})),
                chk: Array.from(document.querySelectorAll(
                        '.st-key-wmap_single_wafer input'))
                    .map(i => ({vis: !!i.offsetParent, on: i.checked})),
                warns: Array.from(document.querySelectorAll(
                        '[data-testid="stAlert"]'))
                    .filter(a => a.offsetParent)
                    .map(a => a.innerText.slice(0, 60)),
            })""")
        print("DIAG-GO:", diag)
        raise


def _click_go(page) -> None:
    page.evaluate(f"() => ({_GO_LIVE_JS})().click()")
    page.wait_for_timeout(1000)
    wait_idle(page, timeout=120000)


def test_m22_wafer_map_flow(app_server, browser, wafer_dataset):
    ds = wafer_dataset
    digest_before = _folder_digest(ds["root"])
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    try:
        # ── 前置:①建模 → ②套用 → autoseed+就地重評(defect 全 flag,沿 M21 先例)──
        enter_anomaly(page, app_server)
        build_model(page, ds["root"])
        apply_model(page, ds["root"])
        seed = page.locator('.st-key-anomaly_autoseed_normal button')
        expect(seed).to_be_visible()
        seed.click()
        wait_idle(page)
        click_tab(page, TAB_APPLY)
        page.wait_for_function(
            """() => { const bs = Array.from(document.querySelectorAll(
                          '.st-key-anomaly_reapply_btn button'))
                          .filter(b => b.offsetParent !== null);
                       return bs.length === 1; }""", timeout=30000)
        page.locator('.st-key-anomaly_reapply_btn button').locator(
            'visible=true').first.click()
        page.wait_for_timeout(1500)
        wait_idle(page, timeout=240000)
        click_tab(page, TAB_APPLY)
        page.locator('[class*="st-key-anomaly_scatter_apply"]').first.wait_for(
            state="visible", timeout=180000)
        wait_idle(page, timeout=60000)
        _no_exception(page)

        # ── 切「🧭 晶圓地圖」(AC-E5 可達性)──
        _enter_wafer_tool(page)

        # ── AC-E3(G6 降級鏈,由淺入深)──
        # 1) 無 CSV
        expect(page.get_by_text("無 metadata CSV", exact=False).first).to_be_visible()
        # 2) 只有 filename 欄 → 缺座標欄位 + 產生鈕 disabled
        _fill_input(page, "wmap_csv", str(ds["nocoord_csv"]))
        expect(page.get_by_text("缺座標欄位", exact=False).first).to_be_visible()
        assert page.locator('.st-key-wmap_go button').is_disabled()
        # 3) 真 CSV + wafer 選 (無) → 無法分片 + disabled;勾「同屬一片」→ enabled
        _fill_input(page, "wmap_csv", str(ds["main_csv"]))
        # 猜中驗證用 X 欄的「值」當錨字("die_x" 不在 label 裡);wafer 欄的
        # label 是「wafer 欄」→ contain_text("wafer") 恆真=假錨字(pitfalls #1)。
        expect(page.locator('.st-key-wmap_x_col')).to_contain_text("die_x")
        _select_option(page, "wmap_wafer_col", "(無)")
        expect(page.get_by_text("無法分片", exact=False).first).to_be_visible()
        assert page.locator('.st-key-wmap_go button').is_disabled()
        page.locator('.st-key-wmap_single_wafer label').first.click()
        page.wait_for_function(       # 先等勾選真的註冊(input.checked),再等鈕
            """() => { const els = Array.from(document.querySelectorAll(
                          '.st-key-wmap_single_wafer input'))
                          .filter(i => i.offsetParent !== null);
                       const inp = els[els.length - 1];
                       return inp && inp.checked; }""", timeout=10000)
        wait_idle(page)
        _wait_go_enabled(page)
        _click_go(page)
        expect(page.get_by_text("(單片)", exact=False).first).to_be_visible()
        _no_exception(page)

        # ── AC-E1(G3+G1 主流程):wafer 欄選回 → 產生 → 簽名正確 ──
        _select_option(page, "wmap_wafer_col", "wafer")
        _wait_go_enabled(page)
        _click_go(page)
        w1 = page.locator('.st-key-wmap_wafer_0')
        w2 = page.locator('.st-key-wmap_wafer_1')
        expect(w1).to_contain_text("偏邊緣")
        expect(w2).to_contain_text("看不出模式")
        # 紅點=第一 trace(設計契約),數量與②判定一致
        w1.locator("svg").first.wait_for(state="visible", timeout=30000)
        assert w1.locator("g.points").first.locator("path").count() == 8
        _no_exception(page)

        # ── AC-E2(G4 對不上明講)──
        gaps = page.locator('.st-key-wmap_gaps')
        expect(gaps).to_contain_text("對不到列 2")
        expect(gaps).to_contain_text("缺座標 1")

        # ── AC-E4(G5 lot 並排+匯出+零寫入)──
        expect(page.get_by_text("L1", exact=False).first).to_be_visible()
        out_dir = ds["root"].parent / "wmap_out"
        out_dir.mkdir(exist_ok=True)
        _fill_input(page, "wmap_export_dir", str(out_dir))
        page.locator('.st-key-wmap_export_btn button').click()
        page.wait_for_function(
            """() => document.body.innerText.includes('已匯出')""", timeout=60000)
        wait_idle(page)
        names = sorted(p.name for p in out_dir.iterdir())
        assert names == ["wafer_signatures.csv", "wmap_W1.png", "wmap_W2.png"], names
        with open(out_dir / "wafer_signatures.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        byid = {r["wafer_id"]: r for r in rows}
        assert byid["W1"]["verdict"] == "edge"
        assert byid["W2"]["verdict"] == "none"
        assert _folder_digest(ds["root"]) == digest_before, "來源資料夾必須零寫入"
        _no_exception(page)
    finally:
        ctx.close()
