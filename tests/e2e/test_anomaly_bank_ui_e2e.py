"""E2E 真實行為驗收:瑕疵偵測模型(memory bank + 投影器)持久化 + 跨資料夾投影。

重構後(M7 wizard):存/載從舊「主畫面 expander 存 bank」改成 ① 模型暫存目錄一鍵存 / 一鍵載。
真實行為(非 element 存在):
  ① 建模 → 💾 存模型(真的寫出 meta.json/projection.npz 到 .lv_cache)
  → 📂 載入模型(模型狀態變『已載入』)→ ② 套用 → 散點疊『參考分佈』灰底 trace
    (把新資料投影回模型分佈)。
"""
from pathlib import Path

import pytest

from ._anomaly_wizard import (apply_model, build_model, enter_anomaly,
                              set_model_dir)
from .conftest import wait_idle

pytestmark = pytest.mark.e2e


def test_model_save_load_project(app_server, browser, synthetic_yolo_dataset):
    ds = synthetic_yolo_dataset
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    try:
        enter_anomaly(page, app_server)

        # ① 建模(留在 tab①)
        build_main = build_model(page, ds["root"])
        assert "模型已建立" in build_main, f"① 應建模成功;實際:\n{build_main[:1000]}"

        # ① 模型暫存目錄:真實 app 行為下這欄位首次 render 在「無資料夾」時建好(value=""),
        # 之後 value= 被忽略 → 欄位空、存鈕 disabled。像真實使用者一樣填一個 .lv_cache 路徑
        # (守 no-dataset-writes);用 app 同款 dataset_cache_dir 算出預設落點。
        from object_eval import dataset_cache_dir
        want_dir = str(dataset_cache_dir(ds["root"], "anomaly_model"))
        model_dir = set_model_dir(page, want_dir)
        assert ".lv_cache" in model_dir, "模型暫存目錄應在 .lv_cache(不寫使用者資料集)"

        # 💾 存模型 → 真的寫出 meta.json(完整性哨兵)+ projection.npz(投影基底)
        save_btn = page.locator('.st-key-anomaly_save_model_btn button')
        save_btn.wait_for(state="visible", timeout=30000)
        save_btn.click()
        wait_idle(page)
        page.wait_for_timeout(1200)
        assert (Path(model_dir) / "meta.json").exists(), "存模型應寫出 meta.json(完整性哨兵)"
        assert (Path(model_dir) / "projection.npz").exists(), "存模型應寫出投影基底"
        assert (Path(model_dir) / "manifest.json").exists(), "存模型應寫出 manifest.json"

        # 📂 載入模型 → 模型狀態變『已載入』(取代 in-session built 模型)
        load_btn = page.locator('.st-key-anomaly_load_model_btn button')
        load_btn.wait_for(state="visible", timeout=30000)
        load_btn.click()
        wait_idle(page)
        page.wait_for_timeout(1000)
        main = page.locator('[data-testid="stMain"]').inner_text()
        assert "模型已載入" in main, f"📂 載入後模型狀態應顯示『已載入』;實際:\n{main[:1000]}"

        # ② 用載入的模型套用偵測 → 把新資料投影回模型分佈,散點疊『參考分佈』灰底 trace
        apply_model(page, ds["root"])
        main = page.locator('[data-testid="stMain"]').inner_text()
        assert "投影回模型分佈" in main, f"② 應說明已投影回模型分佈;實際:\n{main[:1200]}"
        has_bg = page.evaluate(
            """() => { const g = document.querySelector('[class*="st-key-anomaly_scatter"] .js-plotly-plot');
                       return !!(g && g.data && g.data.some(t => (t.name||'') === '參考分佈')); }""")
        assert has_bg, "載入模型後散點應疊『參考分佈』灰底 trace(新資料映射回模型分佈)"
    finally:
        ctx.close()
