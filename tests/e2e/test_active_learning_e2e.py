"""E2E 真實行為驗收:主動學習取樣佇列(Priority Score → 挑最值得標註的物件)。

重構後(M7 wizard):取樣佇列移到 ③ 挑樣 tab 的 2×2 取樣矩陣 master-detail。
真實行為:① 建模 → ② 套用 → ③ 點一個 2×2 模式格 → 出現優先序的物件縮圖
→ 把該佇列加入購物車(送標註)使計數變非零。
"""
import re

import pytest

from ._anomaly_wizard import (TAB_SAMPLE, apply_model, build_model, click_tab,
                              enter_anomaly)
from .conftest import wait_idle

pytestmark = pytest.mark.e2e


def test_active_learning_queue(app_server, browser, synthetic_yolo_dataset):
    ds = synthetic_yolo_dataset
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    try:
        enter_anomaly(page, app_server)
        build_model(page, ds["root"])
        apply_model(page, ds["root"])

        # ③ 挑樣:點一個 2×2 模式格(均衡)→ 真的算出優先序佇列(有縮圖)
        click_tab(page, TAB_SAMPLE)
        main = page.locator('[data-testid="stMain"]').inner_text()
        assert "取樣佇列" in main, "③ 應出現取樣矩陣(取樣佇列)"
        balanced = page.locator('.st-key-anomaly_qmode_balanced button')
        balanced.wait_for(state="visible", timeout=30000)
        balanced.click()
        wait_idle(page)
        # ⚠ 點模式格的 on_click 觸發整頁 rerun → st.tabs 重置回① → 重切回③ 才看得到大圖牆/購物車鈕
        click_tab(page, TAB_SAMPLE)
        page.wait_for_timeout(1200)
        page.wait_for_function(
            """() => document.querySelectorAll('[data-testid="stMain"] img').length > 0""",
            timeout=30000)
        queue_imgs = page.locator('[data-testid="stMain"] img').count()
        assert queue_imgs > 0, "③ 點模式格後佇列應列出最值得標註的物件縮圖"

        # 把佇列加入購物車(送標註)→ 計數變非零(真實行為)
        cart_btn = page.locator('.st-key-anomaly_q_cart button')
        cart_btn.wait_for(state="visible", timeout=30000)
        cart_btn.click()
        wait_idle(page)
        page.wait_for_timeout(800)
        # 計數膠囊在 ② tab(anomaly_cart_count);切回 ② 讀計數
        from ._anomaly_wizard import TAB_APPLY
        click_tab(page, TAB_APPLY)
        page.wait_for_function(
            """() => { const e = document.querySelectorAll('.st-key-anomaly_cart_count');
                       return e.length >= 1 && /[1-9]/.test(e[0].innerText); }""",
            timeout=30000)
        cart = page.locator('.st-key-anomaly_cart_count').first.inner_text()
        assert re.search(r"[1-9]", cart), "取樣佇列加入購物車後計數應非零"
    finally:
        ctx.close()
