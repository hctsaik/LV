# 21 diff_regions — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/21_diff_regions.md](../3_Architect_Design/21_diff_regions.md) · 里程碑 M19 · Tier A(純陣列)
單元測試:`tests/test_diff_regions.py`(gate:`python verify/gate.py diff_regions`)

## 單元 AC ↔ 測試

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | 雙熱區釘死:patch_idx {7,8,13,14} / bbox (1,1,3,3) / 峰 7=6.0 / stability=peak patch 值 / rank 序 | `test_ac1_two_regions_pinned` | 🔴 |
| AC2 | 全低於門檻 → `[]`(誠實空結果) | `test_ac2_all_below_threshold_empty` | 🔴 |
| AC3 | 斜角相鄰 → 8 連通同區(patch_idx {5,10}) | `test_ac3_diagonal_8_connectivity` | 🔴 |
| AC4 | top_k 截斷(rank 連續)+ min_patches 濾單點 + top_k=0 → [] | `test_ac4_top_k_and_min_patches` | 🔴 |
| AC5 | 代表樣本手算:bad 平均距離降冪 [1,2]、good 升冪 [1,2];n_each 超量回全部順序正確 | `test_ac5_representatives_handpicked` | 🔴 |
| AC6 | 白話句:0.82 → pct 82、句型全等;0.005 → pct 0 | `test_ac6_summary_text` | 🔴 |
| AC7 | 3D z_map / 形狀不一致 / 空 patch_idx → ValueError | `test_ac7_validation_errors` | 🔴 |

## 推導測試(設計未明列,PM 自加施壓)

| 測試 | 施壓的不變量 |
|------|-------------|
| `test_derived_peak_inside_region` | 每區 `peak_idx ∈ patch_idx` 且峰的 (r,c) 落在 `bbox_grid` 半開區間內 → 逼出 row/col 對調、bbox 邊界寫錯 |

## done 判準

- `python verify/gate.py diff_regions` 印 `GREEN`(Tier A,無 E2E 要件)。
