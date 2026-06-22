# 對應表:bootstrap_cluster

設計:[3_Architect_Design/03_bootstrap_cluster.md](../3_Architect_Design/03_bootstrap_cluster.md)
測試:`tests/test_bootstrap_cluster.py`(gate:`python verify/gate.py bootstrap_cluster`)

| AC | 測試 | 說明 |
|----|------|------|
| AC1 | test_three_clean_blobs | 乾淨三群→n_clusters=3、無候選 |
| AC2 | test_far_outliers_flagged | 遠離點→label=-1、候選 |
| AC3 | test_small_cluster_is_candidate | 小群即候選(少數壞的) |
| AC4 | test_shapes_and_types | 形狀/型別、normal_mask==~candidate_mask |
| AC5 | test_degenerate_small_n | N=0/1 不崩 |
| AC6 | test_deterministic | 推導:決定性(partition+candidate 一致) |
