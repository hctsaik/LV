# AC ↔ 測試對應表:class_similarity_export

設計:[3_Architect_Design/33_class_similarity_export.md](../3_Architect_Design/33_class_similarity_export.md)。單元:`tests/test_class_similarity_export.py`。

| AC | 說明 | 測試 |
|----|------|------|
| AC09 | 兩 ROI、缺圖 | `test_ac09_two_roi_same_image` `test_ac09_missing_image_reported` |
| AC11 | 往返、中文、null、鄰居重算 mean | `test_ac11_csv_json_roundtrip` `test_ac11_recompute_mean_from_neighbors` |
| AC23 | 重建候選與 PR、精度、未畫出資料 | `test_ac23_export_rebuilds_candidates_and_pr` |
| AC31 | selected 子集 + 全 U index | `test_ac31_selected_scope_full_index` |
