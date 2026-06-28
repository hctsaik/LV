"""E2E 真實行為驗收:M6 統一主動學習畫面的「label 語義 gate」。

重構後第 8 工具瑕疵偵測併成單一畫面,結果區有個「⚙ 進階」popover,內含一個
label 語義 radio(物件類別 / 瑕疵類別)。**分類頭只在「語義=defect AND ≥2 類各達 N_min」
才出現**;語義=object → head 整段不出現(語義安全鎖)。

兩條真實行為斷言(都不是查 element 存在,而是查「行為差異」):
  1. test_semantics_gate_blocks_head:同一份 door/window(各 10≥N_min=8)資料,
     語義切 object → 訓 head 鈕消失;切回 defect → 訓 head 鈕出現。資料量沒變、只有
     語義變 → 證明是「語義守門」非「資料量不足」。
  2. test_below_nmin_no_head:scratch=7<8 的瑕疵資料、預設語義 defect,run 後訓 head 鈕
     不出現,且狀態列列出「待補樣本 / 暫不訓 head」(scratch 不足)。
"""
from __future__ import annotations

import pytest

from .conftest import _add_folder, load_app, wait_idle

pytestmark = pytest.mark.e2e


def _enter_anomaly(page, app_server, root):
    """載入 app → 切到「瑕疵偵測」工具 → 加資料夾。"""
    page.set_default_timeout(30000)
    load_app(page, app_server)
    page.locator('.st-key-tool_switch').get_by_text("瑕疵偵測", exact=True).click()
    wait_idle(page)
    _add_folder(page, "anomaly_folder", str(root))
    wait_idle(page)


def _run(page):
    """按「執行偵測」並等散點圖出現(首跑含模型載入,給足時間)。run 成功會收側欄,
    這裡先(若有)重開側欄再按 run，與既有 anomaly E2E 慣例一致。"""
    rb = page.locator('.st-key-reopen_sidebar button')
    if rb.count():
        try:
            rb.first.click(); wait_idle(page); page.wait_for_timeout(500)
        except Exception:
            pass
    page.locator('.st-key-anomaly_run button').click()
    page.wait_for_timeout(1500)
    page.locator('[class*="st-key-anomaly_scatter"]').first.wait_for(
        state="visible", timeout=180000)
    wait_idle(page, timeout=60000); page.wait_for_timeout(1500)


def _open_advanced_popover(page):
    """打開結果區「⚙ 進階(label 語義 / 模式覆寫 / N_min)」popover。
    Streamlit popover 是 stPopoverButton;rerun 後會收起 → 每次要互動前重開。
    用 JS 找 innerText 含「進階」的 button 點開(最穩,避免被別的 overlay 攔 pointer)。"""
    page.evaluate(
        """() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const b = btns.find(x => (x.innerText || '').includes('進階'));
            if (b) b.click();
        }""")
    page.wait_for_timeout(500)


def _set_semantic(page, option_text):
    """在已開啟的 popover 內,點 label 語義 radio 的某個選項(以可見文字片段匹配)。
    Streamlit radio 選項是 label;rerun 會重畫 → 設完務必 wait_idle。
    option_text 例:'物件類別'(=object) / '瑕疵類別'(=defect)。"""
    clicked = page.evaluate(
        """(t) => {
            const labels = Array.from(document.querySelectorAll('label'));
            const lab = labels.find(l => (l.innerText || '').includes(t));
            if (!lab) return false;
            lab.click();
            return true;
        }""", option_text)
    assert clicked, f"找不到 label 語義 radio 選項:{option_text}"
    page.wait_for_timeout(400)
    wait_idle(page)
    page.wait_for_timeout(800)


def _train_head_btn(page):
    return page.locator('.st-key-anomaly_train_head_btn button')


def test_semantics_gate_blocks_head(app_server, browser, yolo_object_2class):
    """door/window 各 10(≥N_min=8)。語義=object → 不訓 head;切 defect → head 鈕出現。
    資料量不變、只有語義變 → 證明是『語義守門』非『資料量不足』。"""
    ds = yolo_object_2class
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        _enter_anomaly(page, app_server, ds["root"])
        _run(page)

        # (1) 語義=object → 訓 head 鈕不存在(物件類別不訓瑕疵 head)
        _open_advanced_popover(page)
        _set_semantic(page, "物件類別")          # → "object"
        assert _train_head_btn(page).count() == 0, \
            "語義=物件類別時不該出現『訓練分類頭』鈕(語義安全鎖)"

        # (2) 切回 defect → 訓 head 鈕出現(同資料、door/window 各 10≥8)
        _open_advanced_popover(page)
        _set_semantic(page, "瑕疵類別")          # → "defect"
        assert _train_head_btn(page).count() >= 1, \
            "語義=瑕疵類別且每類≥N_min 時應出現『訓練分類頭』鈕 —— 證明 gate 是語義非資料量"
    finally:
        ctx.close()


def test_below_nmin_no_head(app_server, browser, yolo_defect_below_nmin):
    """scratch=7<N_min=8、stain=9。預設語義 defect。run 後訓 head 鈕不出現,
    狀態列列出『待補樣本』/『暫不訓 head』(scratch 不足)。"""
    ds = yolo_defect_below_nmin
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        _enter_anomaly(page, app_server, ds["root"])
        _run(page)

        assert _train_head_btn(page).count() == 0, \
            "某瑕疵類別樣本不足 N_min 時不該出現『訓練分類頭』鈕"
        main = page.locator('[data-testid="stMain"]').inner_text()
        assert ("待補樣本" in main) or ("暫不訓 head" in main), \
            "狀態列應提示樣本不足(待補樣本 / 暫不訓 head),實際 main:\n" + main[:1200]
    finally:
        ctx.close()
