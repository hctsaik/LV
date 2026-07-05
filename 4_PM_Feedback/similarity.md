# AC ↔ 測試對應表:12 similarity

設計:[3_Architect_Design/12_similarity.md](../3_Architect_Design/12_similarity.md)。單元:`tests/test_similarity.py`。

| AC | 說明 | 測試 |
|----|------|------|
| AC1 | cosine 值(同向=1、正交=0) | `test_ac1_cosine_values` |
| AC2 | 相似降冪排序 [0,2,1] + 範圍 [0,1] | `test_ac2_similarity_order_range` |
| AC3 | 參考全零 → ValueError | `test_ac3_ref_zero_raises` |
| AC4 | 空輸入 → 形狀 (0,) | `test_ac4_empty` |
| AC5 | 維度不符 → ValueError | `test_ac5_dim_mismatch` |
| AC6 | 單一物件 → (1,) 值 0.0 | `test_ac6_single` |
| AC7(衍生) | cosine 對正尺度不變 | `test_ac7_positive_scale_invariance` |
| AC8(衍生) | 與參考同向者優先分數最大 | `test_ac8_ref_aligned_is_max` |

雙向覆蓋:AC1–AC6 皆有對應;AC7/8 為 PM 自加的不變量(正尺度不變、同向最大),
逼 cosine 的幾何性質而非只轉抄。GUI 接線 E2E 見 [M12a_gui_wiring.md](M12a_gui_wiring.md)。

## 增補(UX 精修):class_centroid
| AC | 說明 | 測試 |
|----|------|------|
| AC9 | 類別平均向量 | `test_ac9_class_centroid` |
| AC10 | 不存在的類別 → ValueError | `test_ac10_class_centroid_missing` |
| AC11(衍生) | 同類物件對該類 centroid 最像 | `test_ac11_centroid_similarity_same_class` |

## 增補(M13 Task1):multi_ref_similarity
| AC | 說明 | 測試 |
|----|------|------|
| AC-M1 | 基本:最像的類 + max-cosine | `test_ac_m1_multi_ref_basic` |
| AC-M2 | 同類多樣本取 **max**(非平均) | `test_ac_m2_max_not_mean` |
| AC-M3 | 歸最像的類 | `test_ac_m3_picks_closer_class` |
| AC-M4 | 空物件 → 空 | `test_ac_m4_empty_objects` |
| AC-M5 | 無樣本 → ValueError | `test_ac_m5_empty_refs` |
| AC-M6 | 維度不符 → ValueError | `test_ac_m6_dim_mismatch` |
| AC-M7 | 全零樣本列 → ValueError | `test_ac_m7_zero_ref` |
| AC-M8(衍生) | 單類單顆 == 單參考版 | `test_ac_m8_matches_single_ref` |
