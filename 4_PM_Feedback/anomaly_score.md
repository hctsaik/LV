# 對應表:anomaly_score

設計:[3_Architect_Design/02_anomaly_score.md](../3_Architect_Design/02_anomaly_score.md)
測試:`tests/test_anomaly_score.py`(gate:`python verify/gate.py anomaly_score`)

| AC | 測試 | 說明 |
|----|------|------|
| AC1 | test_identical_query_scores_zero | A==bank → 距離 0、物件分數 0 |
| AC2 | test_single_orthogonal_patch_scores_one | 單一正交壞 patch → 物件分數 1.0 |
| AC3 | test_single_orthogonal_patch_scores_one | patch 分數圖形狀=grid、壞點位置正確 |
| AC4 | test_coreset_budget_respected | coreset 預算被遵守(50 / 全用) |
| AC5 | test_object_score_monotone_in_defect_count | 推導:壞 patch 數↑→分數單調、飽和=1 |
| AC6 | test_defect_strictly_separable_from_normal | 可分性 min(defect)>max(normal)(餵 G1) |
| AC7 | test_empty_bank_and_empty_query | 空 bank→ValueError;空查詢→(0, zeros) |
