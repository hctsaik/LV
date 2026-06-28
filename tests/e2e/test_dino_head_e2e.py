"""E2E 真實行為驗收:DINO Classification Head(closed-set 已知類別分類 + 閘控級聯)。

真實行為:偵測→訓練 linear 分類頭→閘控分類結果出現(Normal Bank 異常分數守門:正常/已知類別/Unknown)
→存頭寫出 .joblib(部署用)。
"""
from pathlib import Path

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


def test_train_head_gated_classify(app_server, browser, synthetic_yolo_dataset):
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

    # 訓練分類頭(凍結 DINOv2 特徵 + linear head)
    _js_open(page, "瑕疵分類頭")
    _js_click(page, '.st-key-anomaly_train_head_btn button')
    wait_idle(page); page.wait_for_timeout(1500)
    _js_open(page, "瑕疵分類頭")
    main = page.locator('[data-testid="stMain"]').inner_text()
    assert "閘控分類結果" in main, "訓練後應出現閘控分類結果(Normal Bank 守門的 closed-set 分類)"
    assert any(k in main for k in ("正常", "Unknown", "good", "bad")), "應有閘控後的類別標籤"

    # 存分類頭 → 真的寫出 .joblib(部署用)
    head_path = page.locator('.st-key-anomaly_head_path input').first.input_value()
    assert head_path.endswith(".joblib")
    _js_click(page, '.st-key-anomaly_head_save_btn button')
    wait_idle(page); page.wait_for_timeout(1000)
    assert Path(head_path).exists(), "存頭應寫出 .joblib 檔"
    ctx.close()
