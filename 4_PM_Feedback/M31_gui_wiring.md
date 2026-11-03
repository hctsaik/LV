# AC ↔ 測試對應表:M31 GUI

設計:[3_Architect_Design/M31_gui_wiring.md](../3_Architect_Design/M31_gui_wiring.md)。
單元:`tests/test_class_similarity_ui.py`。E2E:`tests/e2e/test_class_similarity_e2e.py`、`tests/e2e/test_class_similarity_threshold_e2e.py`。

| AC | 說明 | 測試 |
|----|------|------|
| AC07 | 點與表同 ID; box 不選圖; 取消不復活 | `test_ac07_select_ids` `tests/e2e/test_class_similarity_e2e.py::test_ac07_click_table_and_clear` |
| AC10 | 購物車新增／重加 | `tests/e2e/test_class_similarity_e2e.py::test_ac10_cart_add_skip` |
| AC12 | 拖 threshold 不增加 analyze/confusion calls | `test_ac12_preview_does_not_recall_analyze` + E2E |
| AC13 | 舊混淆／搜尋／購物車／送標仍在 | `tests/e2e/test_class_similarity_e2e.py::test_ac13_regression_old_tools` |
| AC14 | 真實 DINO image 與 object; 來源不變 | `tests/e2e/test_class_similarity_e2e.py::test_ac14_image_and_object_source_readonly` |
| AC16 | slider/數字單一 state、精確值 | E2E `test_ac16_canonical_threshold_precision` |
| AC18 | pointer-held 預覽 | E2E `test_ac18_pointer_held_preview` |
| AC25 | 延遲逆序 request | `test_ac25_out_of_order_requests` + E2E |
| AC26 | 跨頁勾選 | `test_ac26_cross_page_local_selection` + E2E |
| AC33 | 窄分布縮放 | E2E `test_ac33_axis_zoom` |
| AC34 | Run 後改圖 stale | E2E `test_ac34_source_change_stale` |
