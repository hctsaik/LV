"""E2E 真實行為驗收:整張影像級瑕疵偵測(無 YOLO 標籤)— 設計 07_whole_image_source.md。

對應 AC-G1/G2:① 物件來源選『整張影像』+ 一個**無 labels/** 的圖片資料夾 → 建模成功
(『模型已建立 · 來源=整張影像』),不出現『建模失敗 / object_source / 找不到 YOLO』。

用真實資料夾 C:/code/dataset/MVTec/pill/train/good(267 張、無 labels/);本機沒有則 skip。
分數依據設『物件級』純為 E2E 加速 —— object_source 路由與 patch 模式同一條碼,不影響本驗收。
"""
from __future__ import annotations

import os

import pytest

from ._anomaly_wizard import TAB_BUILD, click_tab, enter_anomaly
from .conftest import _add_folder, wait_idle

pytestmark = pytest.mark.e2e

MVTEC_GOOD = "C:/code/dataset/MVTec/pill/train/good"


def _set_radio(page, key: str, text: str):
    """點 keyed radio(key=.st-key-<key>)中文字含 text 的選項(排除群組問題 label)。"""
    ok = page.evaluate(
        """([k, t]) => {
            const root = document.querySelector('.st-key-' + k) || document;
            const opts = Array.from(root.querySelectorAll('label'))
                .filter(l => l.querySelector('input[type="radio"]'));
            const lab = opts.find(l => (l.innerText || '').includes(t));
            if (!lab) return false;
            const inp = lab.querySelector('input[type="radio"]');
            if (!inp.checked) lab.click();
            return true;
        }""", [key, text])
    assert ok, f"找不到 radio『{key}』的選項:{text}"
    page.wait_for_timeout(300)
    wait_idle(page)
    page.wait_for_timeout(400)
    checked = page.evaluate(
        """([k, t]) => {
            const root = document.querySelector('.st-key-' + k) || document;
            return Array.from(root.querySelectorAll('label'))
                .filter(l => l.querySelector('input[type="radio"]:checked'))
                .some(l => (l.innerText || '').includes(t));
        }""", [key, text])
    assert checked, f"radio『{key}』未切到『{text}』(實際選項未 checked)"


@pytest.mark.skipif(not os.path.isdir(MVTEC_GOOD), reason="需本機 MVTec pill/train/good 資料夾")
def test_whole_image_build_mvtec_pill(app_server, browser):
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)

        # 側欄:分數依據 → 物件級(E2E 加速;與整張影像路徑無關)
        _set_radio(page, "anomaly_score_mode", "物件級")
        click_tab(page, TAB_BUILD)

        # ① 物件來源 → 整張影像(無需 labels/)
        _set_radio(page, "anomaly_object_source", "整張影像")
        # 切到整張影像後,語義 radio 應隱藏、改顯示整張影像提示(行為差異,非僅 element 存在)
        body0 = page.locator('[data-testid="stMain"]').inner_text()
        assert "整張影像模式" in body0, \
            f"選整張影像後 ① 應顯示『整張影像模式…』提示;實際:\n{body0[:1200]}"

        # ① 訓練/參考資料夾 = MVTec good(無 labels/)
        _add_folder(page, "anomaly_train_folder", MVTEC_GOOD)
        wait_idle(page)
        click_tab(page, TAB_BUILD)

        # 建模:等到建模鈕 enabled → 點 → 等結果(含模型載入,給足 timeout)
        btn = page.locator('.st-key-anomaly_build_btn button')
        btn.wait_for(state="visible", timeout=30000)
        page.wait_for_function(
            """() => { const b = document.querySelector('.st-key-anomaly_build_btn button');
                       return b && !b.disabled; }""", timeout=30000)
        btn.click()
        page.wait_for_function(
            """() => { const m = document.querySelector('[data-testid="stMain"]');
                       if (!m) return false;
                       const t = m.innerText || '';
                       return t.includes('模型已建立') || t.includes('建模失敗')
                              || t.includes('找不到 YOLO'); }""",
            timeout=300000)
        wait_idle(page, timeout=60000)
        page.wait_for_timeout(800)

        main = page.locator('[data-testid="stMain"]').inner_text()
        # 真實行為斷言
        assert "模型已建立" in main, f"整張影像建模應成功;實際:\n{main[:1500]}"
        assert "來源=整張影像" in main, f"成功列應標示『來源=整張影像』;實際:\n{main[:1500]}"
        assert "建模失敗" not in main, f"不應建模失敗;實際:\n{main[:1500]}"
        assert "object_source" not in main, \
            "不應再出現 run_pipeline object_source 參數錯(那是殘留舊 server 的症狀)"
        assert "找不到 YOLO" not in main, "整張影像不需 labels,不應報『找不到 YOLO 物件』"

        from playwright.sync_api import expect
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()
