# 對應表:anomaly_tool

設計:[3_Architect_Design/06_anomaly_tool.md](../3_Architect_Design/06_anomaly_tool.md)
單元:`tests/test_anomaly_tool.py`(gate:`python verify/gate.py anomaly_tool`,-m "not e2e")
E2E:`tests/e2e/test_anomaly_tool_e2e.py`(`@pytest.mark.e2e`,/ux-test 或人觸發,**不進 gate**)

## 單元 AC
| AC | 測試 | 說明 |
|----|------|------|
| AC1 | test_schema_and_original_path | records 全鍵、path==image_path(原圖) |
| AC2 | test_subtle_defects_top_and_auroc | 細微瑕疵+乾淨 bank→全在前段 |
| AC3 | test_subtle_defects_top_and_auroc | AUROC≥0.85(餵 G1) |
| AC4 | test_bank_pollution_drops_score | 反向驗證:污染 bank→分數<半 |
| AC5 | test_object_mode_and_one_stage_gross | object 模式 schema;1-stage 明顯缺陷排前段 |
| AC6 | test_no_dataset_pollution | 資料集目錄零新增 |

## E2E AC(真實行為;UI 完成後由 /ux-test 跑)
| AC | 測試 | 說明 |
|----|------|------|
| E2E-AC1 | test_defects_rank_in_top | 缺陷檔名集中在排序前段(讀真實文字) |
| E2E-AC2 | test_select_and_add_to_cart | 框選→購物車計數增加 |
| E2E-AC3 | test_export_is_original_image | 匯出 zip 內為原圖尺寸 128×128 + manifest |
| E2E-AC6 | test_error_path_no_labels | 無 labels/→中文指引、無 stException |

注:E2E-AC4/AC5(heatmap 峰值、toggle 切換)列於設計,UI 細節由 PG 對齊選擇器後於 ux-test 補齊。
誠實定位:單元用注入 fake 防實作 bug;真「使用者能用」靠 E2E + 人審。
