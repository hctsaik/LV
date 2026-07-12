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

## 增補(M14c):copy_images(匯出成 standalone YOLO 資料集)
| AC | 說明 | 測試 |
|----|------|------|
| AC-RIMG-1 | copy_images=True → out_dir/images/ 含有標影像(對應 labels)+ images_copied 計數 | `test_ac_rimg1_copy_images` |
| AC-RIMG-2 | 來源零寫入(C6)+ 同 out_dir 再匯出不 raise(allow_images) | `test_ac_rimg2_c6_and_reexport` |
| AC-RIMG-3(不變量) | copy_images=False 預設 → 無 images/、images_copied==0、既有無回歸 | `test_ac_rimg3_default_no_images` |
| AC-RIMG-4 | 只複製 accepted/relabeled 的影像(全 skip 的圖不複製) | `test_ac_rimg4_only_accepted_images` |
