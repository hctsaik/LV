# AC ↔ 測試對應表:M14 以樣搜樣特徵器解耦(B1 GUI)+ M14b UX 迭代

設計:[3_Architect_Design/M14_extractor_decouple.md](../3_Architect_Design/M14_extractor_decouple.md)(含 M14b 增補)。
E2E:`tests/e2e/test_fewshot_scenarios_e2e.py`(10 情境;取代舊 test_fewshot_gui_e2e / test_fewshot_decouple_e2e 過時導覽)。

## M14 解耦核心(併入 10 情境)
| AC | 說明 | 情境 |
|----|------|------|
| AC-D1 | 乾淨 session、從未建 anomaly 模型 → ①可用、預設特徵器建成樣本集(死路拆掉) | `test_s01_no_model_full_flow` |
| AC-D2 | 解耦後海掃佇列仍含建議類別+相似度 + 來源零寫入(C6) | `test_s01` + `test_s02_export_dual_and_c6` |
| AC-D3(可選) | 換 object_source 重掃不吃 stale shard | **由單元 AC-RL-4 覆蓋**(引擎層確定性) |

## M14b UX(合併 ②③ + 匯出移頂 + 緊湊 checkbox)
| S | 情境 | 測試 |
|---|------|------|
| S1 | 無模型全流程(建樣本集→海掃→佇列含建議類別) | `test_s01_no_model_full_flow` |
| S2 | 匯出雙出(labels+csv+classes)+ C6 零寫入 | `test_s02_export_dual_and_c6` |
| S3 | 匯出區在佇列之上(bounding_box.y) | `test_s03_export_above_queue` |
| S4 | 緊湊 checkbox 預設打勾;無舊下拉+略過鈕 | `test_s04_compact_checkbox_default_checked` |
| S5 | 取消勾選→該顆不匯出(標註行數變少) | `test_s05_uncheck_excludes_from_export` |
| S6 | 整張影像模式(無 YOLO 標也能建+掃) | `test_s06_whole_image_mode` |
| S7 | 加入樣本集迴圈:採納→加入→N1>N0 | `test_s07_add_to_bank_grows` |
| S8 | 監看免 anomaly 模型:初始化+立即掃描一次 | `test_s08_monitor_no_model` |
| S9 | θ 門檻一致套用:佇列 checkbox 數=命中數、標頭顯示 θ | `test_s09_theta_applied_to_queue` |
| S10 | 導覽只剩 3 步(②併入確認/匯出、無④) | `test_s10_three_steps_only` |

> 後端核心(retrieve 免 bank、model_version 綁特徵器身分)由單元 AC-RL/AC-OS/AC-SVC-DEC 覆蓋;
> E2E 驗 GUI 真實行為,交 /ux-test 或人觸發,不進 PG 自主修綠迴圈。
