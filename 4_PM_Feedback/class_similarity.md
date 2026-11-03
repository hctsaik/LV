# AC ↔ 測試對應表:class_similarity

設計:[3_Architect_Design/31_class_similarity.md](../3_Architect_Design/31_class_similarity.md)。單元:`tests/test_class_similarity.py`。

| AC | 說明 | 測試 |
|----|------|------|
| AC01 | 手算 fixture、K=2 逐類、margin=2/0、有效 K | `test_ac01_hand_fixture_k1` `test_ac01_k2_per_class_topk` `test_ac01_margin_all_two` `test_ac01_margin_all_zero` `test_ac01_effective_k_when_k_exceeds_pool` `test_ac01_singleton_c1` |
| AC02 | 獨立 oracle 與分塊／多 block size | `test_ac02_oracle_matches_blocked` `test_ac02_block_sizes_agree` |
| AC03 | 排除自己、同向量不同 ID、同分穩定、列序 | `test_ac03_exclude_self` `test_ac03_duplicate_vectors_kept` `test_ac03_tie_stable_and_row_order` |
| AC04 | 無效輸入與邊界 | `test_ac04_zero_nan_inf` `test_ac04_length_and_k` `test_ac04_empty_all_invalid` `test_ac04_singleton_and_single_class` |
| AC05 | box 全量、null 不當 0、quartile/whisker | `test_ac05_box_full_population` `test_ac05_null_not_zero` `test_ac05_quartile_whisker_fixture` |
| AC12 | 5000 cap、block shape | `test_ac12_reject_over_5000` `test_ac12_block_shape_never_exceeds` |
| AC32 | overflow cosine、微小向量、tied rival cap | `test_ac32_scale_safe_large` `test_ac32_tiny_vector_excluded` `test_ac32_tied_rivals_capped` |

oracle 在測試檔內獨立實作，禁止 import 正式 `analyze` 的內部函式來當期望值。
