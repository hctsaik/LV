"""E2E 真實行為驗收:瑕疵偵測 Memory Bank 持久化 + 跨資料夾投影(設計見 ANOMALY_BANK_DESIGN_NOTES.md)。

真實行為(非 element 存在):偵測→存 bank profile(寫出 meta/projection)→主畫面載入掛載→
新資料夾用「對照已掛載 bank」重跑、散點疊『舊分佈』灰底 trace(把新資料映射回舊分佈)。
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
    rb = page.locator('.st-key-reopen_sidebar button')   # run 後側欄收合 → 先重開
    if rb.count():
        try:
            rb.first.click(); wait_idle(page); page.wait_for_timeout(500)
        except Exception:
            pass
    page.locator('.st-key-anomaly_run button').click()
    page.wait_for_timeout(1500)
    page.locator('[class*="st-key-anomaly_scatter"]').first.wait_for(state="visible", timeout=180000)
    wait_idle(page, timeout=60000); page.wait_for_timeout(1500)


def test_bank_save_load_project(app_server, browser, synthetic_yolo_dataset):
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

    # 1) 存 bank profile(主畫面 expander)→ 真的寫出檔(守 no-dataset-writes:落 .lv_cache)
    _js_open(page, "把本次結果存成")
    bank_path = page.locator('.st-key-anomaly_bank_save_path input').first.input_value()
    assert ".lv_cache" in bank_path, "預設落點應在 .lv_cache(不寫使用者資料集)"
    _js_click(page, '.st-key-anomaly_save_btn button')
    wait_idle(page); page.wait_for_timeout(1200)
    assert (Path(bank_path) / "meta.json").exists(), "存 bank 應寫出 meta.json(完整性哨兵)"
    assert (Path(bank_path) / "projection.npz").exists(), "存 bank 應寫出投影基底"

    # 2) 主畫面載入 → 真的掛載(膠囊出現)
    li = page.locator('.st-key-anomaly_bank_load_path input').first
    li.fill(bank_path); li.press("Enter")
    wait_idle(page); page.wait_for_timeout(800)
    _js_click(page, '.st-key-anomaly_load_btn button')
    wait_idle(page); page.wait_for_timeout(1200)
    assert "已掛載 bank" in page.locator('[data-testid="stMain"]').inner_text()

    # 3) 掛載後重跑 → 用外部 bank 評分 + 把新資料投影回舊分佈(散點疊『舊分佈』灰底 trace)
    _run(page)
    main = page.locator('[data-testid="stMain"]').inner_text()
    assert "對照已掛載 bank" in main and "投影回舊分佈" in main
    has_bg = page.evaluate(
        """() => { const g = document.querySelector('[class*="st-key-anomaly_scatter"] .js-plotly-plot');
                   return !!(g && g.data && g.data.some(t => (t.name||'') === '舊分佈')); }""")
    assert has_bg, "掛載後散點應疊『舊分佈』灰底 trace(新資料映射回舊分佈)"
    ctx.close()
