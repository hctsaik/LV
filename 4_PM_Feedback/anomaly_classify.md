# 對應表:anomaly_classify

設計:[3_Architect_Design/04_anomaly_classify.md](../3_Architect_Design/04_anomaly_classify.md)
測試:`tests/test_anomaly_classify.py`(gate:`python verify/gate.py anomaly_classify`)

| AC | 測試 | 說明 |
|----|------|------|
| AC1 | test_separable_with_both_labels | 可分+雙標籤→門檻介於、全分對、AUROC=1 |
| AC2 | test_unlabeled_percentile_exact_count | 無標籤百分位、精確計數 n_bad=10 |
| AC3 | test_ranking_desc_stable | 排序由高到低、同分索引升冪 |
| AC4 | test_only_bad_confirmed | 只給 bad→門檻≤該分數 |
| AC5 | test_auroc_matches_sklearn | AUROC 對齊 sklearn(=0.75) |
| AC6 | test_empty_and_all_invalid_confirmed | 推導:空/全越界 confirmed 退化不崩 |
