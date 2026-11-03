# AC ↔ 測試對應表:class_similarity_threshold

設計:[3_Architect_Design/34_class_similarity_threshold.md](../3_Architect_Design/34_class_similarity_threshold.md)。單元:`tests/test_class_similarity_threshold.py`、`tests/test_class_similarity_evaluation.py`。

| AC | 說明 | 測試 |
|----|------|------|
| AC15 | `<`、相等、重複、±2、none/all、null | `test_ac15_candidate_membership` |
| AC16 | key 不含 t; t 不改 box／鄰居 | `test_ac16_threshold_not_in_score_key` |
| AC17 | 全量 counts、每類加總、filter 不改母體 | `test_ac17_counts_full_population` |
| AC19 | manifest 驗證 | `test_ac19_manifest_validation` |
| AC20 | 手算 TP/FP/FN/TN/P/R、零分母、無真值 | `test_ac20_pr_hand_fixture` `test_ac20_zero_denominator` `test_ac20_no_truth_unavailable` |
| AC21 | 曲線與 predicate、同分整組 | `test_ac21_workload_and_pr_curves` |
| AC22 | label 過期真值; K 重算不造 review | `test_ac22_label_change_stales_truth` `test_ac22_k_rescores_without_forging_reviews` |
| AC24 | heldout 隔離 | `test_ac24_heldout_isolation` |
| AC25 | draft/commit、舊 action 拒絕 | `test_ac25_draft_commit_and_stale_action` |
| AC27 | 模板 ROI、TP 下鑽 ID | `test_ac27_template_and_confusion_ids` |
| AC28 | digest／非法 manifest／空 E0 | `test_ac28_digest_and_manifest_rejects` |
| AC33 | 軸縮放不改統計 | `test_ac33_axis_zoom_does_not_change_stats` |
| AC34 | 操作點精度、tuning/heldout、source group | `test_ac34_operating_point_and_heldout` |
