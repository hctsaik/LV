"""E2E 真實行為驗收:label 語義 gate(重構成 M7 wizard 後)。

重構後語義 radio 從結果區「⚙ 進階」popover 提升到 ① 建模 tab(anomaly_train_semantic);
分類頭在 ① 建模時自動訓(語義=defect AND ≥2 類各達 N_min 才解鎖),建模成功訊息會標
『含分類頭 / 無分類頭』。本檔以「建模後 head 有無」作為 gate 的真實行為信號。

兩條真實行為斷言(都是查『行為差異』,非 element 存在):
  1. test_semantics_gate_blocks_head:同一份 door/window(各 10≥N_min=8)資料,
     語義=物件類別 → 建模『無分類頭』;語義=瑕疵類別 → 建模『含分類頭』。
     資料量沒變、只有語義變 → 證明是『語義守門』非『資料量不足』。
  2. test_below_nmin_no_head:scratch=7<8 的瑕疵資料、語義=瑕疵類別,建模後仍『無分類頭』
     (某瑕疵類樣本不足 N_min → 達標類 <2 → head 不解鎖)。
"""
from __future__ import annotations

import pytest

from ._anomaly_wizard import build_model, enter_anomaly

pytestmark = pytest.mark.e2e


def test_semantics_gate_blocks_head(app_server, browser, yolo_object_2class):
    """door/window 各 10(≥N_min=8)。語義=物件類別 → 建模無分類頭;語義=瑕疵類別 → 含分類頭。
    資料量不變、只有語義變 → 證明是『語義守門』非『資料量不足』。
    用兩個獨立 context(各自乾淨 session)各建一次,避免狀態互相污染。"""
    ds = yolo_object_2class

    # (1) 語義=物件類別 → 建模『無分類頭』(物件類別不訓瑕疵 head)
    ctx1 = browser.new_context(viewport={"width": 1920, "height": 1080})
    page1 = ctx1.new_page()
    try:
        enter_anomaly(page1, app_server)
        main_obj = build_model(page1, ds["root"], semantic_text="物件類別")
        assert "模型已建立" in main_obj, f"① 應建模成功;實際:\n{main_obj[:1000]}"
        assert "無分類頭" in main_obj, \
            "語義=物件類別時不該訓出分類頭(語義安全鎖)"
        assert "含分類頭" not in main_obj, "物件類別語義下絕不應出現『含分類頭』"
    finally:
        ctx1.close()

    # (2) 同資料、語義=瑕疵類別 → 建模『含分類頭』(door/window 各 10≥8)
    ctx2 = browser.new_context(viewport={"width": 1920, "height": 1080})
    page2 = ctx2.new_page()
    try:
        enter_anomaly(page2, app_server)
        main_def = build_model(page2, ds["root"], semantic_text="瑕疵類別")
        assert "模型已建立" in main_def, f"① 應建模成功;實際:\n{main_def[:1000]}"
        assert "含分類頭" in main_def, \
            "語義=瑕疵類別且每類≥N_min 時應訓出分類頭 —— 證明 gate 是語義非資料量"
    finally:
        ctx2.close()


def test_below_nmin_no_head(app_server, browser, yolo_defect_below_nmin):
    """scratch=7<N_min=8、stain=9。語義=瑕疵類別。建模後仍『無分類頭』
    (達標類 <2 → head 不解鎖)。證明 head 解鎖確實看每類樣本數,非總數。"""
    ds = yolo_defect_below_nmin
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)
        main = build_model(page, ds["root"], semantic_text="瑕疵類別")
        assert "模型已建立" in main, f"① 應建模成功;實際:\n{main[:1000]}"
        assert "無分類頭" in main, \
            "某瑕疵類別樣本不足 N_min 時不該訓出分類頭(達標類 <2 → head 不解鎖)"
    finally:
        ctx.close()
