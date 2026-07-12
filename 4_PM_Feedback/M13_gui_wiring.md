# AC ↔ 測試對應表:M13 GUI(第 9 工具「以樣搜樣」)
設計:[3_Architect_Design/M13_gui_wiring.md](../3_Architect_Design/M13_gui_wiring.md)。E2E:`tests/e2e/test_fewshot_gui_e2e.py`。

| AC | 說明 | 測試 |
|----|------|------|
| AC-F1 | 樣本集 + 海掃 + 佇列(建議類別+相似度) | `test_f1_f2_sample_scan_export` |
| AC-F2 | 匯出雙出(labels+csv+classes)讀回 + 來源零寫入 | 同上 |
| AC-F3 | 無模型友善導引 | `test_f3_no_model_graceful` |
| AC-F4 | ③加入樣本集 → 樣本集真的長大(N1>N0)+ 來源零寫入(C6) | `test_f4_add_to_sample_bank_grows` |
