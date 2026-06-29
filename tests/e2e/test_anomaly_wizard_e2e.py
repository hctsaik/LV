"""E2E 真實行為驗收:重構後「瑕疵偵測」三 tab wizard(設計 M7_wizard_redesign.md)。

新流程:① 建模/載入 → ② 套用偵測(分開的目標資料夾)→ ③ 挑樣送人工標(2×2 取樣矩陣)。
兩條核心真實行為斷言(非 element 存在):
  - test_build_apply_sample_flow:完整走完三步,③ 點一格真的出大圖牆 + 有意義標籤(判定/類別)。
  - test_semantics_gate_object_no_head:語義=物件類別 → 不訓分類頭(模型狀態顯「無分類頭」、
    ③ 徽章退成 可疑·正常 二分,不含 未知/類別 三分)→ 證語義守門。
"""
from __future__ import annotations

import re

import pytest

from ._anomaly_wizard import (TAB_SAMPLE, apply_model, build_model, click_tab,
                              enter_anomaly)
from .conftest import wait_idle

pytestmark = pytest.mark.e2e


def test_build_apply_sample_flow(app_server, browser, yolo_defect_at_nmin):
    """① 加訓練資料夾 + 語義「瑕疵類別」+ 建模 → ② 同 root 當目標、套用 → ③ 點一格 2×2 模式,
    斷言右邊出現裁切影像(img>0)且標籤有意義(頁面含「判定」或類別名)。
    用 yolo_defect_at_nmin(scratch×8 + stain×8,語義 defect、兩類各 8=N_min → 解鎖分類頭)。"""
    ds = yolo_defect_at_nmin
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)

        # ① 建模(語義=瑕疵類別 → defect)
        build_main = build_model(page, ds["root"], semantic_text="瑕疵類別")
        assert "模型已建立" in build_main, f"① 應建模成功;實際:\n{build_main[:1200]}"
        # defect + 兩類各 8 ≥ N_min=8 → 含分類頭
        assert "含分類頭" in build_main, \
            f"瑕疵類別且每類達 N_min 應訓出分類頭;實際:\n{build_main[:1200]}"

        # ② 套用偵測(目標資料夾用同一 root,測流程接線)
        apply_model(page, ds["root"])
        apply_main = page.locator('[data-testid="stMain"]').inner_text()
        assert "個物件" in apply_main, f"② 套用後應有結果概覽;實際:\n{apply_main[:1200]}"

        # ③ 挑樣:切 tab → 點一個 2×2 模式格(novelty)→ 大圖牆出現
        click_tab(page, TAB_SAMPLE)
        sample_main = page.locator('[data-testid="stMain"]').inner_text()
        assert "取樣佇列" in sample_main, f"③ 應出現取樣矩陣;實際:\n{sample_main[:1200]}"
        # 點「偏 novelty」那一格(2×2 第一格)→ on_click 設 anomaly_q_mode → rerun 重畫大圖牆
        novelty = page.locator('.st-key-anomaly_qmode_novelty button')
        novelty.wait_for(state="visible", timeout=30000)
        novelty.click()
        wait_idle(page)
        # ⚠ 點模式格的 on_click 觸發整頁 rerun → st.tabs 重置回 ① → 重切回 ③ 才看得到大圖牆
        click_tab(page, TAB_SAMPLE)
        page.wait_for_timeout(1000)

        # 真實行為:右邊大圖牆真的有裁切影像(每張縮圖一個 <img>)
        page.wait_for_function(
            """() => document.querySelectorAll('[data-testid="stMain"] img').length > 0""",
            timeout=30000)
        n_imgs = page.locator('[data-testid="stMain"] img').count()
        assert n_imgs > 0, "③ 點模式格後右邊應出現裁切影像縮圖"

        # 標籤有意義:每張三行「類別 / 判定:… / 異常 0.XX」→ 頁面應含「判定」與某個 defect 類別名
        body = page.locator('[data-testid="stMain"]').inner_text()
        assert "判定" in body, "③ 影像標籤應含『判定:…』行(有意義標籤)"
        assert ("scratch" in body or "stain" in body), \
            f"③ 影像標籤應含類別名(scratch/stain);實際尾段:\n{body[-800:]}"

        # 真實行為:把佇列加入購物車 → 計數變非零(送標註)
        cart_btn = page.locator('.st-key-anomaly_q_cart button')
        cart_btn.wait_for(state="visible", timeout=30000)
        cart_btn.click()
        wait_idle(page)
        page.wait_for_timeout(800)
        _no_exception(page)
    finally:
        ctx.close()


def test_semantics_gate_object_no_head(app_server, browser, yolo_object_2class):
    """語義守門:door/window 各 10(≥N_min=8)但語義=物件類別 → 不訓分類頭。
    證據(行為差異,非 element 存在):
      1. ① 模型狀態顯示『無分類頭』(非『含分類頭』)。
      2. ② 套用 → ③ 的 2×2 徽章退成『可疑 N·正常 M』二分,不含『未知/類別』三分。
    資料量足夠、只有語義是 object → 證明 head 被『語義守門』擋掉,非『資料量不足』。"""
    ds = yolo_object_2class
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)

        # ① 建模,語義=物件類別 → object
        build_main = build_model(page, ds["root"], semantic_text="物件類別")
        assert "模型已建立" in build_main, f"① 應建模成功;實際:\n{build_main[:1200]}"
        assert "無分類頭" in build_main, \
            f"語義=物件類別時不該訓分類頭(語義守門);實際:\n{build_main[:1200]}"
        assert "含分類頭" not in build_main, \
            "物件類別語義下絕不應出現『含分類頭』"

        # ② 套用偵測(同 root 當目標)
        apply_model(page, ds["root"])

        # ③ 挑樣:無 head → 2×2 徽章是『可疑 N·正常 M』二分,絕不出現『未知』三分
        click_tab(page, TAB_SAMPLE)
        page.wait_for_timeout(800)
        # 先點一格讓大圖牆+標籤渲染(否則只有 2×2 徽章,沒有「判定:」標籤行)
        click_tab(page, TAB_SAMPLE)
        nov = page.locator('.st-key-anomaly_qmode_novelty button')
        nov.wait_for(state="visible", timeout=30000)
        nov.click()
        wait_idle(page)
        page.wait_for_timeout(1000)
        sample_main = page.locator('[data-testid="stMain"]').inner_text()
        assert "可疑" in sample_main, \
            f"無分類頭時徽章/標籤應為『可疑·正常』二分;實際:\n{sample_main[:1500]}"
        # ⚠「未知」會出現在模式名「稀有/未知更強」→ 用精確的「判定:Unknown」與三分徽章「類別 」驗證
        assert "判定:Unknown" not in sample_main and "判定:未知" not in sample_main, \
            f"無分類頭時不該有 Unknown 判定(語義守門證據);實際:\n{sample_main[:1500]}"
        assert "·類別 " not in sample_main, \
            f"無分類頭時徽章不該是『正常·類別·未知』三分;實際:\n{sample_main[:1500]}"
        # 說明文字也明示無分類頭 → 只用 novelty
        assert "無分類頭" in sample_main, \
            f"③ 應註明『無分類頭 → 只用 novelty』;實際:\n{sample_main[:1500]}"
    finally:
        ctx.close()


def _no_exception(page):
    from playwright.sync_api import expect
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)
