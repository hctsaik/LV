"""M9-GUI 接線真實 E2E(設計 3_Architect_Design/M9_gui_wiring.md)。

驗「大資料分批掃描」接線行為(非 literal 50k 效能):用小合成資料,驗接線正確——
分批掃描產出**可渲染的 Top-K 標註佇列**(縮圖 + 人話理由 reason)、**不渲染全量散點**(scale-safe)、
未存模型**自動存**再掃、object_source **從磁碟**讀(whole_image 不被默默當 yolo)、續跑無例外。

> v1 範圍(反向閘門 /pg→/architect 修正):大資料分批掃描只出「標註佇列」;探索/完整散點模式延後
> (需 al_batch 未來回傳抽樣 obj_emb 才可行)。故本 E2E 不驗散點三模式切換,改直驗佇列 + 「無散點」。

── PG 必須實作的 widget key 契約 ──
② 分頁內(_anomaly_tab_apply):
  anomaly_batch_scan_btn      「▶ 大資料分批掃描」按鈕
  anomaly_batch_objective     選樣目標(novelty/balanced/confusion/pure;無 head 反灰後兩者)
  anomaly_batch_resume_btn    「▶ 繼續上次」按鈕
  anomaly_batch_queue         標註佇列容器(內含縮圖卡 stImage + 每卡 reason 文字)
  anomaly_batch_scatter       (v1 不渲染;此 class 存在數必為 0 = 大資料 scale-safe)
DOM 穩定標記(避開自續/阻塞 rerun 讓 wait_idle 不穩):
  掃描完成主畫面出現「✅ 分批掃描完成」(含「掃描完成」)。
"""
from __future__ import annotations

import pytest
from playwright.sync_api import expect

from ._anomaly_wizard import (TAB_APPLY, build_model, click_tab, enter_anomaly,
                              set_model_dir)
from .conftest import _add_folder, wait_idle

pytestmark = pytest.mark.e2e


def _wait_scan_done(page, timeout=300000):
    """等分批掃描跑完 —— 用穩定 DOM 標記(阻塞/自續 rerun 下 wait_idle 不可靠)。"""
    page.wait_for_function(
        """() => {
            const m = document.querySelector('[data-testid="stMain"]');
            return m && (m.innerText || '').includes('掃描完成');
        }""", timeout=timeout)
    wait_idle(page, timeout=60000)
    page.wait_for_timeout(600)


def _batch_scan(page, target_root):
    """② 切 tab → 加目標資料夾 → 按「大資料分批掃描」→ 等掃描完成。"""
    click_tab(page, TAB_APPLY)
    _add_folder(page, "anomaly_target_folder", str(target_root))
    wait_idle(page)
    click_tab(page, TAB_APPLY)   # _add_folder rerun 可能重置步驟
    btn = page.locator('.st-key-anomaly_batch_scan_btn button')
    btn.wait_for(state="visible", timeout=30000)
    page.wait_for_function(
        """() => { const b = document.querySelector('.st-key-anomaly_batch_scan_btn button');
                   return b && !b.disabled; }""", timeout=30000)
    btn.click()
    _wait_scan_done(page)


def _assert_queue_and_no_scatter(page):
    queue = page.locator('.st-key-anomaly_batch_queue')
    expect(queue.locator('[data-testid="stImage"]').first).to_be_visible(timeout=30000)
    qtext = queue.inner_text()
    assert any(w in qtext for w in ("很不像正常", "拿不準", "邊界")), \
        f"佇列卡缺人話理由(reason):{qtext[:200]}"
    # scale-safe:大資料分批掃描結果**不渲染全量散點**
    assert page.locator('[class*="st-key-anomaly_batch_scatter"]').count() == 0, \
        "大資料標註佇列不應渲染散點(scale-safe 關鍵)"


# ── 小合成資料集 fixtures ────────────────────────────────────────────────
def _paint(img, defect):
    if defect:
        for x in range(10, 74):
            for y in range(10, 74):
                img.putpixel((x, y), (220, 0, 0))
    return img


@pytest.fixture
def yolo_ds(tmp_path):
    """YOLO 偵測小集:images/+labels/+classes.txt,綠底 + 少數紅塊,每圖 1 框。"""
    from PIL import Image
    root = tmp_path / "yolo_ds"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("obj\n", encoding="utf-8")
    for i in range(8):
        img = _paint(Image.new("RGB", (128, 128), (0, 170, 0)), i % 3 == 0)
        img.save(root / "images" / f"o{i:02d}.jpg", quality=92)
        (root / "labels" / f"o{i:02d}.txt").write_text(
            "0 0.5 0.5 0.9 0.9\n", encoding="utf-8")
    return root


@pytest.fixture
def whole_ds(tmp_path):
    """無 labels 圖片資料夾(整張影像模式):8 張,少數紅塊。"""
    from PIL import Image
    root = tmp_path / "whole_ds"
    root.mkdir(parents=True)
    for i in range(8):
        img = _paint(Image.new("RGB", (128, 128), (0, 170, 0)), i % 3 == 0)
        img.save(root / f"w{i:02d}.jpg", quality=92)
    return root


# ── AC-G1 / AC-G2:分批掃描產出佇列(含 reason);不渲染散點 ──────────────────
def test_g1_g2_batch_scan_queue_and_no_scatter(app_server, browser, yolo_ds, tmp_path):
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)
        build_model(page, yolo_ds, semantic_text="物件類別")   # 先建模(存模型鈕才會 enabled)
        set_model_dir(page, tmp_path / "mdl")
        page.locator('.st-key-anomaly_save_model_btn button').click()
        wait_idle(page)
        _batch_scan(page, yolo_ds)
        _assert_queue_and_no_scatter(page)
    finally:
        ctx.close()


# ── AC-G5:未存模型 → 自動存再掃(使用者拍板:auto-save)─────────────────────
def test_g5_autosave_before_scan(app_server, browser, yolo_ds, tmp_path):
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)
        model_dir = tmp_path / "mdl_auto"
        build_model(page, yolo_ds, semantic_text="物件類別")   # 先建模
        set_model_dir(page, model_dir)                        # 設暫存目錄(但刻意不按存)
        # 刻意「不」按存模型 → 直接分批掃描應自動存後才跑(不出現原始 exception)
        _batch_scan(page, yolo_ds)
        # 佇列渲染 = 掃描成功 = al_batch 讀到了磁碟凍結模型(load_frozen_model 缺 meta.json 會 raise)
        expect(page.locator('.st-key-anomaly_batch_queue [data-testid="stImage"]').first
               ).to_be_visible(timeout=30000)
        # 真實行為訊號:畫面明示「已自動存模型」(auto-save 路徑確實走了,非預先存的)
        main = page.locator('[data-testid="stMain"]').inner_text()
        assert "已自動存模型" in main, f"未存模型時分批掃描應自動存並提示:{main[:300]}"
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()


# ── AC-G6:object_source 從磁碟正確(whole_image 不被默默當 yolo)──────────────
def test_g6_object_source_from_disk(app_server, browser, whole_ds, tmp_path):
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)
        # ① 切「整張影像」物件來源 → 建模 → 設暫存目錄 → 存
        page.locator('.st-key-anomaly_object_source').get_by_text(
            "整張影像", exact=False).first.click()
        wait_idle(page)
        # 整張影像模式**無語義 radio**(恆 1 類、不訓 head)→ build_model 不設語義
        build_model(page, whole_ds)
        set_model_dir(page, tmp_path / "mdl_whole")
        page.locator('.st-key-anomaly_save_model_btn button').click()
        wait_idle(page)
        # 📂 載入(in-memory 曾會掉 object_source)→ 分批掃描仍以整張影像掃
        page.locator('.st-key-anomaly_load_model_btn button').click()
        wait_idle(page)
        _batch_scan(page, whole_ds)
        _assert_queue_and_no_scatter(page)
        # 整張影像:物件數 == 影像數 8(主畫面掃描摘要含物件數)
        main = page.locator('[data-testid="stMain"]').inner_text()
        assert "8" in main, f"整張影像模式物件數應 == 影像數 8(避免被默默當 yolo):{main[:300]}"
    finally:
        ctx.close()


# ── AC-G4:繼續上次 → done、佇列、無例外 ────────────────────────────────────
def test_g4_resume(app_server, browser, yolo_ds, tmp_path):
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)
        build_model(page, yolo_ds, semantic_text="物件類別")   # 先建模
        set_model_dir(page, tmp_path / "mdl_resume")
        page.locator('.st-key-anomaly_save_model_btn button').click()
        wait_idle(page)
        _batch_scan(page, yolo_ds)           # 先跑一輪到完成
        # 「繼續上次」(小資料已 done → 續跑為冪等 no-op,仍應無例外、佇列在)
        click_tab(page, TAB_APPLY)
        resume = page.locator('.st-key-anomaly_batch_resume_btn button')
        resume.wait_for(state="visible", timeout=30000)
        resume.click()
        _wait_scan_done(page)
        _assert_queue_and_no_scatter(page)
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()
