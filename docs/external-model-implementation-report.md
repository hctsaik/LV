# 外部模型完整流程：實作與驗收

日期：2026-09-06

OpenSpec：`add-external-model-investigation`。

## 已實作

| 模組 | 交付 |
|---|---|
| `fp_external_runtime.py` | ONNX CPU 單圖推論、嚴格模型契約、分類與三種 YOLO 輸出、前後處理、NMS、tensor／模型身分、完整歷史 parity |
| `fp_investigation.py` | 案件 run／stale 檢查、遮擋與亮度控制實驗、固定目標匹配、獨立 manifest 驗證、雙模型評估及驗收 |
| `fp_training_export.py` | 人工完整覆核的 YOLO 訓練 ZIP、類別對應、完整負樣本與混合框防錯、provenance |
| `fp_external_page.py` | 模型設定、推論／歷史比對、實驗、資料評估、訓練交接與下載 |
| `fp_onnx_demo.py` | 可重現的可執行 ONNX 測試模型產生器，非生產瑕疵模型 |

既有案件頁增加 **外部模型、訓練交接、改善評估**。Markdown 報告會依實際 run／experiment／evaluation／training export 顯示證據，不再固定宣稱外部模型未執行。覆核回傳按鈕會等檔案上傳完成才啟用，避免提早提交空檔案。

## 驗證命令

```powershell
.venv/Scripts/python.exe -m pytest tests/test_fp_external_runtime.py tests/test_fp_investigation.py tests/test_fp_cases.py tests/test_fp_case_search.py tests/test_fp_case_page.py tests/test_similarity.py tests/test_round_ledger.py tests/test_probe_eval.py tests/test_gt_pred_diff.py tests/test_readback_store.py tests/test_retrieval_export.py tests/test_explain_page.py -q

.venv/Scripts/python.exe -m pytest tests/e2e/test_fp_external_investigation_e2e.py tests/e2e/test_fp_case_workspace_e2e.py -q --browser chromium

openspec validate add-external-model-investigation --strict
```

單元／整合與相關回歸 **125 passed**；覆核上傳介面修正後另跑頁面測試 **2 passed**。OpenSpec strict 驗證通過。機器驗證包含以下情境：

- 實際 ONNX inference 的數值與實際 input/output NPZ，非 mock inference。
- letterbox 回原座標、RGB/BGR、normalization、輸出布局錯誤拒絕、YOLO objectness×class 及 NMS。
- 未知歷史來源、本次輸出自我驗證、截圖皆不能標成當時判定已重現。
- 分數相同框不同、少框／多框會失配；接近框的比對不受輸出排序影響。
- 模型內容或設定改變後契約不同；來源／ROI 變更阻擋過期實驗。
- 原亮度控制、遮擋目標消失分數不可判讀、控制組不足明示、不自動提升根因狀態。
- 固定正常／瑕疵資料上的雙模型誤逮與漏檢；誤逮下降但漏檢增加會失敗。
- 相同圖有警報但位置錯誤時，框層級 FP/FN 可揭露圖片警報率掩蓋的錯誤。
- 獨立資料與查案內容重疊、重複影像、缺真值皆拒絕；報案回歸不升格成獨立驗收。
- 混合正常／真瑕疵影像保留真框；局部正常不轉整圖負樣本；截圖來源不經候選洗成訓練原圖。

## 真實瀏覽器範圍

外部模型流程最終驗證 **1 passed（52.61 秒）**。基準／候選模型使用各自的儲存提示，測試等待對應提示才繼續，避免誤用上一個模型的成功訊息。連同下述舊工作台，兩條瀏覽器 E2E 均通過；頁面截圖已人工檢視。

新增瀏覽器流程在真正 Streamlit app 中設定兩個可執行 ONNX：建案、覆核、儲存模型、執行當次推論、提供控制測試的歷史紀錄驗證 parity、下載含 input/output 的完整 ZIP、執行亮度控制實驗、匯出人工確認負樣本訓練 ZIP、比較獨立資料、下載評估結果並重啟恢復。

舊案件工作台瀏覽器測試通過（1 passed，142.28 秒），使用真正 DINOv2 進行相似搜尋、報告與覆核包下載／讀回、來源 hash 不變及服務重啟。

## 邊界

測試以合成影像與有可計算期望值的 ONNX 圖執行，證明接入、數值及流程；未取得使用者實際部署 checkpoint／歷史輸出，不宣稱已驗證其生產模型。支援格式及設定方式見 [接入指南](external-model-investigation.md)。

ONNX 接口不包含任意 `.pt`／TensorRT／REST／segmentation 模型。訓練包交給原本的外部訓練流程，沒有替未知架構啟動重訓或部署。提供的 ONNX 模型與前後處理須符合接入契約。

獨立性包含使用者聲明與工作區內的內容重疊檢查，不能推定未知外部訓練資料完全無洩漏。驗收結果限定該模型、該資料集及該門檻，不是統計因果證明、根因自動確認或部署許可。
