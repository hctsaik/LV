"""M14b 以樣搜樣「合併步驟 + 匯出移頂 + 緊湊 checkbox」真實 E2E —— 10 情境。

設計:3_Architect_Design/M14_extractor_decouple.md 增補(M14b)。
全程**零 anomaly 模型**(預設 DINOv2 特徵器);取代舊 test_fewshot_gui_e2e / test_fewshot_decouple_e2e 的過時步驟導覽。
標記:①=「建立樣本集」、②=「相似度門檻」、③監看=「工作區目錄」;海掃完成=「以樣搜樣掃描完成」;
匯出區=「輸出資料夾」;佇列容器 `.st-key-fewshot_queue` 內 `get_by_role("checkbox")`(預設 checked)。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from PIL import Image
from playwright.sync_api import expect

from .conftest import _add_folder, load_app, wait_idle

pytestmark = pytest.mark.e2e


# ── 共用 helpers(自足,不依賴其他 E2E 檔)──────────────────────────────────
def _switch_fewshot(page):
    page.locator('.st-key-tool_switch').get_by_text("以樣搜樣", exact=False).first.click()
    wait_idle(page)
    page.wait_for_timeout(800)
    assert "以樣搜樣" in page.locator('[data-testid="stMain"]').inner_text()


def _step(page, name, marker):
    """切到某步:反覆點該步選項直到內文 marker 出現(retry 處理 segment 誤 deselect)。"""
    seg = page.locator('.st-key-fewshot_step')
    seg.first.wait_for(state="visible", timeout=15000)
    for _ in range(4):
        if marker in page.locator('[data-testid="stMain"]').inner_text():
            page.wait_for_timeout(300)
            return
        seg.get_by_text(name, exact=False).first.click()
        wait_idle(page)
        page.wait_for_timeout(400)
    raise AssertionError(f"切不到步驟「{name}」(找不到 {marker})")


def _click(page, key, timeout=30000):
    b = page.locator(f'.st-key-{key} button').first
    b.wait_for(state="visible", timeout=timeout)
    page.wait_for_function(
        f"""() => {{ const b=document.querySelector('.st-key-{key} button'); return b && !b.disabled; }}""",
        timeout=timeout)
    b.click()


def _bank_count(page) -> int:
    txt = page.locator('.st-key-fewshot_bank_info').inner_text()
    m = re.search(r"共\s*(\d+)\s*顆", txt)
    return int(m.group(1)) if m else -1


def _snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*") if p.is_file()}


def _label_lines(out_dir: Path) -> int:
    return sum(len([ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()])
               for f in (out_dir / "labels").glob("*.txt"))


def _build_bank(page, root, *, whole_image=False):
    """① 建樣本集(預設特徵器,零 anomaly 模型)。"""
    _step(page, "樣本集", "建立樣本集")
    _add_folder(page, "fewshot_sample_folder", str(root))
    wait_idle(page)
    if whole_image:
        page.locator('.st-key-fewshot_object_source_sel').get_by_text("整張影像", exact=False).first.click()
        wait_idle(page)
    _step(page, "樣本集", "建立樣本集")
    _click(page, "fewshot_build_bank_btn")
    page.wait_for_function(
        """() => { const t=(document.querySelector('[data-testid="stMain"]')||{}).innerText||'';
                   return t.includes('樣本集已建立') || t.includes('建立樣本集失敗'); }""", timeout=240000)
    wait_idle(page)


def _scan(page, root):
    """② 海掃(合併步)。"""
    _step(page, "海掃", "相似度門檻")
    _add_folder(page, "fewshot_target_folder", str(root))
    wait_idle(page)
    _step(page, "海掃", "相似度門檻")
    _click(page, "fewshot_scan_btn")
    page.wait_for_function(
        """() => { const t=(document.querySelector('[data-testid=\"stMain\"]')||{}).innerText||'';
                   return t.includes('以樣搜樣掃描完成'); }""", timeout=300000)
    wait_idle(page)


def _new_page(browser):
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    return ctx, ctx.new_page()


def _export_to(page, out_dir: Path):
    inp = page.locator('.st-key-fewshot_out_dir input').first
    inp.wait_for(state="visible", timeout=30000)
    inp.fill(str(out_dir))
    inp.press("Enter")
    wait_idle(page)
    _click(page, "fewshot_export_btn")
    page.wait_for_function(
        """() => { const t=(document.querySelector('[data-testid=\"stMain\"]')||{}).innerText||'';
                   return t.includes('以樣搜樣匯出完成'); }""", timeout=60000)
    wait_idle(page)


# ══════════════════════════════════════════════════════════════════════
# S1 無模型全流程(旗艦):零 anomaly 模型 → 建樣本集 → 海掃 → 佇列含建議類別
# ══════════════════════════════════════════════════════════════════════
def test_s01_no_model_full_flow(app_server, browser, yolo_defect_at_nmin):
    ds = yolo_defect_at_nmin
    ctx, page = _new_page(browser)
    try:
        load_app(page, app_server)
        _switch_fewshot(page)
        _build_bank(page, ds["root"])
        assert "樣本集已建立" in page.locator('.st-key-fewshot_bank_info').inner_text()
        _scan(page, ds["root"])
        q = page.locator('.st-key-fewshot_queue')
        expect(q.locator('[data-testid="stImage"]').first).to_be_visible(timeout=30000)
        assert any(c in q.inner_text() for c in ("scratch", "stain"))
        _exc = page.locator('[data-testid="stException"]')
        if _exc.count():
            raise AssertionError(f"頁面有例外:{_exc.first.inner_text()[:900]}")
    finally:
        ctx.close()


# S2 匯出雙出 + C6 來源零寫入
def test_s02_export_dual_and_c6(app_server, browser, yolo_defect_at_nmin, tmp_path):
    ds = yolo_defect_at_nmin
    ctx, page = _new_page(browser)
    try:
        load_app(page, app_server)
        _switch_fewshot(page)
        _build_bank(page, ds["root"])
        _scan(page, ds["root"])
        out = tmp_path / "out"
        src_before = _snapshot(ds["root"])
        _export_to(page, out)
        assert list((out / "labels").glob("*.txt")), "應匯出 labels/*.txt"
        assert (out / "retrieval_report.csv").exists() and (out / "classes.txt").exists()
        head = (out / "retrieval_report.csv").read_text(encoding="utf-8").splitlines()[0]
        assert head.startswith("image_path,obj_index,cx,cy,w,h")
        # M14c:預設一併複製影像 → images/ 存在且與 labels 對應(可直接訓練的 YOLO 資料集)
        assert (out / "images").is_dir(), "應含 images/(可直接訓練)"
        img_stems = {f.stem for f in (out / "images").glob("*") if f.is_file()}
        lbl_stems = {f.stem for f in (out / "labels").glob("*.txt")}
        assert img_stems and lbl_stems <= img_stems, f"每個標註影像都應被複製:labels={lbl_stems} images={img_stems}"
        assert _snapshot(ds["root"]) == src_before, "來源零寫入(C6)"
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()


# S3 匯出區在佇列之上(版面:使用者不會漏看匯出)
def test_s03_export_above_queue(app_server, browser, yolo_defect_at_nmin):
    ds = yolo_defect_at_nmin
    ctx, page = _new_page(browser)
    try:
        load_app(page, app_server)
        _switch_fewshot(page)
        _build_bank(page, ds["root"])
        _scan(page, ds["root"])
        exp_box = page.locator('.st-key-fewshot_export_btn button').first.bounding_box()
        img_box = page.locator('.st-key-fewshot_queue [data-testid="stImage"]').first.bounding_box()
        assert exp_box and img_box and exp_box["y"] < img_box["y"], \
            f"匯出區應在佇列之上:export.y={exp_box and exp_box['y']} img.y={img_box and img_box['y']}"
    finally:
        ctx.close()


# S4 緊湊 checkbox 預設打勾;無舊的下拉+略過按鈕
def test_s04_compact_checkbox_default_checked(app_server, browser, yolo_defect_at_nmin):
    ds = yolo_defect_at_nmin
    ctx, page = _new_page(browser)
    try:
        load_app(page, app_server)
        _switch_fewshot(page)
        _build_bank(page, ds["root"])
        _scan(page, ds["root"])
        # Streamlit 把真 input 設 hidden(自訂 span 顯示)→ 不驗 visible,驗存在 + is_checked(讀屬性)
        expect(page.locator('.st-key-fewshot_queue [data-testid="stCheckbox"]').first).to_be_visible(timeout=30000)
        boxes = page.locator('.st-key-fewshot_queue').get_by_role("checkbox")
        assert boxes.count() >= 1, "佇列每張應有 checkbox"
        assert boxes.first.is_checked(), "checkbox 應預設打勾(採納)"
        assert page.locator('.st-key-fewshot_skip_0').count() == 0, "不應再有舊的略過按鈕"
    finally:
        ctx.close()


# S5 取消勾選 → 該顆不匯出(標註行數變少)
def test_s05_uncheck_excludes_from_export(app_server, browser, yolo_defect_at_nmin, tmp_path):
    ds = yolo_defect_at_nmin
    ctx, page = _new_page(browser)
    try:
        load_app(page, app_server)
        _switch_fewshot(page)
        _build_bank(page, ds["root"])
        _scan(page, ds["root"])
        out_all = tmp_path / "all"
        _export_to(page, out_all)
        n_all = _label_lines(out_all)
        assert n_all >= 2, f"需有多顆採納才能驗排除:{n_all}"
        # 取消第一個勾(input 隱藏 → 點可見的 stCheckbox 切換)
        page.locator('.st-key-fewshot_queue [data-testid="stCheckbox"]').first.click()
        wait_idle(page)
        assert not page.locator('.st-key-fewshot_queue').get_by_role("checkbox").first.is_checked(), \
            "點擊後首個 checkbox 應變未勾"
        out_less = tmp_path / "less"
        _export_to(page, out_less)
        n_less = _label_lines(out_less)
        assert n_less < n_all, f"取消一勾後匯出行數應變少:{n_all}→{n_less}"
    finally:
        ctx.close()


# S6 整張影像模式(無 YOLO 標也能建 + 海掃)
def test_s06_whole_image_mode(app_server, browser, yolo_defect_at_nmin):
    ds = yolo_defect_at_nmin
    ctx, page = _new_page(browser)
    try:
        load_app(page, app_server)
        _switch_fewshot(page)
        _build_bank(page, ds["root"], whole_image=True)   # 切「整張影像」
        assert "樣本集已建立" in page.locator('.st-key-fewshot_bank_info').inner_text()
        _scan(page, ds["root"])
        expect(page.locator('.st-key-fewshot_queue [data-testid="stImage"]').first).to_be_visible(timeout=30000)
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()


# S7 加入樣本集迴圈:採納 → 加入 → 樣本集長大
def test_s07_add_to_bank_grows(app_server, browser, yolo_defect_at_nmin):
    ds = yolo_defect_at_nmin
    ctx, page = _new_page(browser)
    try:
        load_app(page, app_server)
        _switch_fewshot(page)
        _build_bank(page, ds["root"])
        n0 = _bank_count(page)
        assert n0 > 0
        _scan(page, ds["root"])
        _click(page, "fewshot_add_to_bank_btn")
        page.wait_for_function(
            """() => { const t=(document.querySelector('[data-testid=\"stMain\"]')||{}).innerText||'';
                       return t.includes('已加入'); }""", timeout=120000)
        wait_idle(page)
        _step(page, "樣本集", "建立樣本集")
        n1 = _bank_count(page)
        assert n1 > n0, f"加入樣本集後應長大:{n0}→{n1}"
    finally:
        ctx.close()


# S8 監看免 anomaly 模型:③ 監看 → 初始化 + 立即掃描一次
def test_s08_monitor_no_model(app_server, browser, yolo_defect_at_nmin):
    ds = yolo_defect_at_nmin
    ctx, page = _new_page(browser)
    try:
        load_app(page, app_server)
        _switch_fewshot(page)
        _build_bank(page, ds["root"])
        _scan(page, ds["root"])            # 設定目標夾(監看沿用)
        _step(page, "監看", "工作區目錄")
        _click(page, "fewshot_watch_scan_btn")   # profile 缺 → 內部先 init 再 run_once
        page.wait_for_function(
            """() => { const t=(document.querySelector('[data-testid=\"stMain\"]')||{}).innerText||'';
                       return t.includes('監看掃描完成') || t.includes('上次掃描') || t.includes('監看掃描失敗'); }""",
            timeout=300000)
        wait_idle(page)
        body = page.locator('[data-testid="stMain"]').inner_text()
        assert "監看掃描失敗" not in body, f"監看免模型應成功:{body[-300:]}"
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()


# S9 相似度門檻 θ 一致套用:佇列 checkbox 數 == 命中數,標頭顯示當前 θ
def test_s09_theta_applied_to_queue(app_server, browser, yolo_defect_at_nmin):
    # 註:此 fixture 命中相似度皆 1.00,θ≤1.0 都保留(≥ 為包含),故驗「θ 一致套用」而非「調高清空」
    ds = yolo_defect_at_nmin
    ctx, page = _new_page(browser)
    try:
        load_app(page, app_server)
        _switch_fewshot(page)
        _build_bank(page, ds["root"])
        _scan(page, ds["root"])
        expect(page.locator('.st-key-fewshot_queue [data-testid="stCheckbox"]').first).to_be_visible(timeout=30000)
        main = page.locator('[data-testid="stMain"]').inner_text()
        m = re.search(r"命中\s*(\d+)", main)
        assert m, f"佇列標頭應顯示命中數:{main[:200]}"
        hit_n = int(m.group(1))
        n_box = page.locator('.st-key-fewshot_queue [data-testid="stCheckbox"]').count()   # 數可見的 checkbox 容器
        assert n_box == hit_n, f"佇列 checkbox 數應=命中數(θ 過濾一致):box={n_box} hit={hit_n}"
        assert "相似度 ≥ 0.30" in main, "佇列標頭應顯示當前 θ(=0.30)"
    finally:
        ctx.close()


# S10 導覽只剩 3 步(② 已併入 確認/匯出;無獨立 ③ 確認、無 ④)
def test_s10_three_steps_only(app_server, browser):
    ctx, page = _new_page(browser)
    try:
        load_app(page, app_server)
        _switch_fewshot(page)
        txt = page.locator('.st-key-fewshot_step').inner_text()
        assert "① 樣本集" in txt and "② 海掃 · 確認/匯出" in txt and "③ 監看" in txt, txt
        assert "④" not in txt, f"不應再有第 4 步:{txt}"
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()
