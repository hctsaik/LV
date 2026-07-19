# AC ↔ 測試對應:26 paired_compare

設計:[3_Architect_Design/26_paired_compare.md](../3_Architect_Design/26_paired_compare.md)
測試:[tests/test_paired_compare.py](../tests/test_paired_compare.py)

| AC | 測試 | 說明 |
|----|------|------|
| AC1 | `test_ac1_effect_detected_exact` | 12 項全 +0.5 → a_better,p=2/4096(窮舉 exact) |
| AC2 | `test_ac2_zero_mean_honest` | mean=0 → inconclusive+無差異 |
| AC3 | `test_ac3_batch_confound_naive_vs_blocked` | 批次混淆:天真 p=128/4096 誤判 vs blocked p=1.0+群組不足 |
| AC4 | `test_ac4_blocked_can_pass` | 6 群×2 → blocked exact p=2/64 可過門檻 |
| AC5 | `test_ac5_small_n_conservative` | N=6 小 p 仍被樣本不足蓋過 |
| AC6 | `test_ac6_direction_flip` | higher_is_better=False 方向正確 |
| AC7 | `test_ac7_error_paths` | 長度/NaN/groups/alpha → ValueError |
| AC8 | `test_ac8_method_line_fragments` | method line 固定片段(含 exploratory) |
| AC9 | `test_ac9_mc_deterministic` | N>12 → MC,同 seed p 相等 |
| AC10 | `test_ac10_null_false_positive_rate` | 200 組 null → FP 率 ≤ 0.075(統計驗收) |
| (推導) | `test_derived_swap_symmetry` | a/b 交換 → 鏡像 verdict、p 不變 |
| (推導) | `test_derived_scale_invariance` | d 正尺度縮放 → p/verdict 不變 |
| (推導) | `test_derived_group_relabel_invariance` | 群組改名不變(只看分割) |
