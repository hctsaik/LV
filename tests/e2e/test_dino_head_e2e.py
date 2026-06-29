"""E2E 真實行為驗收:DINO Classification Head(closed-set 已知類別分類 + 閘控級聯)。

重構後(M7 wizard):分類頭不再是結果區 expander 內單獨訓練,而是 ① 建模時自動一起訓
(語義=defect AND ≥2 類各達 N_min → 解鎖);存頭併入 ① 的「💾 存模型」(head.joblib 落同目錄)。
真實行為:① 建 defect 模型(含分類頭)→ ② 套用 → ③ 出現閘控分類(正常/已知類別/Unknown 三分)
→ 💾 存模型寫出 head.joblib(部署用)。
"""
from pathlib import Path

import pytest

from ._anomaly_wizard import (TAB_SAMPLE, apply_model, build_model, click_tab,
                              enter_anomaly, set_model_dir)
from .conftest import wait_idle

pytestmark = pytest.mark.e2e


def test_train_head_gated_classify(app_server, browser, yolo_defect_at_nmin):
    # head 解鎖門檻 = 語義=defect AND ≥2 類各達 N_min(預設8)。
    # yolo_defect_at_nmin:scratch×8 + stain×8(語義 defect、兩類各 8=N_min)→ ① 建模自動訓 head。
    ds = yolo_defect_at_nmin
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    try:
        enter_anomaly(page, app_server)

        # ① 建模(語義=瑕疵類別 → 自動訓分類頭)
        build_main = build_model(page, ds["root"], semantic_text="瑕疵類別")
        assert "模型已建立" in build_main, f"① 應建模成功;實際:\n{build_main[:1200]}"
        assert "含分類頭" in build_main, \
            f"defect 語義且每類達 N_min 應訓出分類頭;實際:\n{build_main[:1200]}"

        # 💾 存模型 → head.joblib 併入模型暫存目錄(部署用)。模型暫存目錄欄位在真實 app 下
        # 預設空(value= 被忽略;見 set_model_dir 說明)→ 先像使用者一樣填 .lv_cache 路徑。
        from object_eval import dataset_cache_dir
        model_dir = set_model_dir(page, str(dataset_cache_dir(ds["root"], "anomaly_model")))
        save_btn = page.locator('.st-key-anomaly_save_model_btn button')
        save_btn.wait_for(state="visible", timeout=30000)
        save_btn.click()
        wait_idle(page)
        page.wait_for_timeout(1200)
        assert (Path(model_dir) / "head.joblib").exists(), \
            "存模型應把分類頭寫出 head.joblib(部署用)"

        # ② 套用偵測(同 root 當目標)
        apply_model(page, ds["root"])

        # ③ 挑樣:有分類頭 → 2×2 徽章/標籤出現閘控分類(Normal Bank 守門的 closed-set 三分)
        click_tab(page, TAB_SAMPLE)
        page.wait_for_timeout(800)
        main = page.locator('[data-testid="stMain"]').inner_text()
        # 徽章三分(正常 N·類別 M·未知 U)或標籤閘控判定 → 出現閘控相關字樣
        assert any(k in main for k in ("未知", "類別", "正常")), \
            f"有分類頭時 ③ 應出現閘控分類(正常/類別/未知 三分);實際:\n{main[:1500]}"
        # 點一格出大圖牆 → 標籤含閘控『判定:…』行 + 類別名
        novelty = page.locator('.st-key-anomaly_qmode_novelty button')
        novelty.wait_for(state="visible", timeout=30000)
        novelty.click()
        wait_idle(page)
        # ⚠ 點模式格 on_click 觸發整頁 rerun → st.tabs 重置回① → 重切回③ 才看得到大圖牆/標籤
        click_tab(page, TAB_SAMPLE)
        page.wait_for_timeout(1000)
        body = page.locator('[data-testid="stMain"]').inner_text()
        assert "判定" in body, "③ 影像標籤應含閘控『判定:…』行"
        assert ("scratch" in body or "stain" in body), "③ 標籤應含已知類別名(scratch/stain)"
    finally:
        ctx.close()
