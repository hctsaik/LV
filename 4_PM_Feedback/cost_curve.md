# AC ↔ 測試對應:27 cost_curve

設計:[3_Architect_Design/27_cost_curve.md](../3_Architect_Design/27_cost_curve.md)
測試:[tests/test_cost_curve.py](../tests/test_cost_curve.py)

| AC | 測試 | 說明 |
|----|------|------|
| AC1 | `test_ac1_exact_full_arrays` | 固定小例全陣列手算 ==(G3 exact) |
| AC2 | `test_ac2_ratio_moves_workpoint` | 成本比 5:1→1:10,best_idx 3→5 |
| AC3 | `test_ac3_current_not_snapped` / `test_ac3b_current_none` | current 直接算不吸附;未給=None |
| AC4 | `test_ac4_tie_takes_loosest` | 平手取最小 index;K=1 兩點曲線 |
| AC5 | `test_ac5_error_paths` | 單類/成本≤0/label 出集合/長度/空/空 thresholds → ValueError |
| AC6 | `test_ac6_method_line` | method 含判定規則+漏檢:誤報+僅試算 |
| (推導) | `test_derived_cost_identity_and_flag_identity` | 逐點恆等式 cost/flagged |
| (推導) | `test_derived_cost_scale_invariance` | 成本同乘 → best_idx 不變 |
| (推導) | `test_derived_score_shift_invariance` | 分數平移 → 計數不變 |
| (推導) | `test_derived_explicit_thresholds_dedup_sorted` | 顯式閾值去重排序 |
