"""E2E 真實行為驗收:主動學習迴圈(M5)—— 學習曲線(主動選樣 vs 隨機)真的算出並呈現。

真實行為:偵測 → 展開迴圈面板 → 跑標註效益曲線 → 出現「主動 vs 隨機」最終結果 + 折線圖。
"""
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


def test_active_loop_curve(app_server, browser, synthetic_yolo_dataset):
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

    # 展開迴圈面板 → 真的在
    _js_open(page, "主動學習迴圈")
    page.wait_for_timeout(500)
    assert "主動選樣 vs 隨機" in page.locator('[data-testid="stMain"]').inner_text(), "迴圈面板應出現"

    # 跑標註效益曲線(active vs random)→ 算出最終結果(依按鈕文字點,最穩)
    page.evaluate(
        """() => { const b=[...document.querySelectorAll('button')]
                     .find(x=>x.innerText.includes('跑標註效益曲線')); if(b) b.click(); }""")
    page.wait_for_timeout(800)
    wait_idle(page, timeout=90000)
    # 重跑後 expander 會回收合 → 結果文字被收起;用 textContent(含收合)且重開面板
    page.wait_for_function(
        """() => { const t = document.querySelector('[data-testid="stMain"]').textContent;
                   return t.includes('最終(標') || t.includes('資料不足'); }""",
        timeout=90000)
    _js_open(page, "主動學習迴圈")
    page.wait_for_timeout(400)
    main = page.locator('[data-testid="stMain"]').text_content()
    assert "最終(標" in main, f"學習曲線應算出主動 vs 隨機結果,得到:{main[-300:]}"
    assert "領先" in main, "應顯示主動相對隨機的領先幅度"
    ctx.close()
