# AC ↔ 測試對應表:13c retrieval_export
設計:[3_Architect_Design/13c_retrieval_export.md](../3_Architect_Design/13c_retrieval_export.md)。單元:`tests/test_retrieval_export.py`。

| AC | 說明 | 測試 |
|----|------|------|
| AC-CE1 | CSV 表頭與值(accepted/skipped/None conf) | `test_ac_ce1_csv_header_values` |
| AC-CE2 | YOLO 只含 accept + 沿用粗框幾何 | `test_ac_ce2_yolo_accept_only_proposal_geom` |
| AC-CE3 | relabel → cls_id 換 | `test_ac_ce3_relabel` |
| AC-CE4 | 雙出(labels+csv+classes)+ 讀回 | `test_ac_ce4_dual_export_readback` |
| AC-CE5 | C6:輸出在來源內 → raise | `test_ac_ce5_c6` |
| AC-CE6 | final_class 不在 class_names → raise | `test_ac_ce6_final_class_unknown` |
