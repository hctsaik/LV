"""E2E 真實行為驗收:主動學習取樣佇列(Priority Score → 挑最值得標註的物件)。

真實行為:偵測→展開取樣佇列→出現優先序的物件縮圖→把佇列加入購物車(送標註)使計數變非零。
"""
import re

import pytest

from .conftest import _add_folder, load_app, wait_idle

pytestmark = pytest.mark.e2e


def _js_open(page, text):
    page.evaluate(
        "(t)=>{for(const d of document.querySelectorAll('details')){if(d.textContent.includes(t)) d.open=true;}}", text)
    page.wait_for_timeout(400)


def _js_click(page, sel):
    page.evaluate("(s)=>{const b=document.querySelector(s); if(b) b.click();}", sel)


def _run(page):
    rb = page.locator('.st-key-reopen_sidebar button')
    if rb.count():
        try:
            rb.first.click(); wait_idle(page); page.wait_for_timeout(500)
        except Exception:
            pass
    page.locator('.st-key-anomaly_run button').click()
    page.wait_for_timeout(1500)
    page.locator('[class*="st-key-anomaly_scatter"]').first.wait_for(state="visible", timeout=180000)
    wait_idle(page, timeout=60000); page.wait_for_timeout(1500)


def test_active_learning_queue(app_server, browser, synthetic_yolo_dataset):
    ds = synthetic_yolo_dataset
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    load_app(page, app_server)
    page.locator('.st-key-tool_switch').get_by_text("瑕疵偵測", exact=True).click()
    wait_idle(page)
    _add_folder(page, "anomaly_folder", str(ds["root"]))
    wait_idle(page)
    _run(page)

    # 展開取樣佇列 → 真的算出優先序佇列(有縮圖)
    _js_open(page, "主動學習取樣佇列")
    page.wait_for_timeout(600)
    main = page.locator('[data-testid="stMain"]').inner_text()
    assert "取樣佇列(優先序" in main, "應出現主動學習取樣佇列"
    queue_imgs = page.locator('.st-key-anomaly_al_queue img').count()
    assert queue_imgs > 0, "佇列應列出最值得標註的物件縮圖"

    # 把佇列加入購物車(送標註)→ 計數變非零(真實行為)
    _js_click(page, '.st-key-anomaly_al_cart button')
    page.wait_for_function(
        """() => { const e = document.querySelectorAll('.st-key-anomaly_cart_count');
                   return e.length === 1 && /[1-9]/.test(e[0].innerText); }""",
        timeout=30000)
    cart = page.locator('.st-key-anomaly_cart_count').inner_text()
    assert re.search(r"[1-9]", cart), "取樣佇列加入購物車後計數應非零"
    ctx.close()
