# 誤逮案件工作台：實作與驗收紀錄

日期：2026-09-06

OpenSpec：`add-false-positive-case-workspace`，範圍為已定案的第一版查案工作台。

本檔保留第一輪交付紀錄。外部模型重播、對照實驗、訓練交接及改善評估已接續實作，見 [完整流程驗收紀錄](external-model-implementation-report.md)。

## 交付內容

- 新增工具列入口「誤逮案件」，可用上傳／本機圖片建案、補入來源、恢復既有案件。
- SQLite 交易與版本檢查保存案件及歷史快照；圖片副本、搜尋快取僅落案件工作區。
- 支援人工指定多個 normalized ROI、六欄 YOLO 預測匯入、逐區域人員／依據／裁決。
- 本機 DINOv2 224 CLS＋cosine 對照正常／瑕疵資料夾與已覆核歷史案件；記錄 checkpoint、recipe、來源 hash、搜尋範圍、截斷與壞檔。
- 報案區域預覽與對照卡、正常／瑕疵標籤衝突提示、內容與 ROI 去重、候選保存。
- 引用式說明可編輯並匯出 Markdown／JSON；保留事實、假設、下一步與独立回覆狀態。
- Review-only JSON／圖片 ZIP、既有 round linkage、部分覆核回傳及原子驗證；不建立訓練標籤。
- 手動計時、暫停入帳、補登／修正與持久工時日誌。

## 實作檔案

| 檔案 | 責任 |
|---|---|
| `scripts/fp_cases.py` | 案件／來源／區域／裁決／引用／工時／覆核 manifest 與 round 連結 |
| `scripts/fp_case_search.py` | 明確觸發的真實 LV 檢索、crop／model provenance 與內容快取 |
| `scripts/fp_case_page.py` | 中文案件操作流程、圖片預覽、下載與回傳 |
| `scripts/app.py` | 加入工具列、功能地圖及 dispatch；保留原工作區其他修改 |

## 驗證

Focused + related regression command:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_fp_cases.py tests/test_fp_case_search.py tests/test_fp_case_page.py tests/test_similarity.py tests/test_round_ledger.py tests/test_probe_eval.py tests/test_gt_pred_diff.py tests/test_readback_store.py tests/test_retrieval_export.py tests/test_explain_page.py -q
```

結果：103 passed（其中新功能 19 項）。涵蓋來源不變、持久恢復、並行舊版拒絕、多來源、混合正常／真瑕疵 ROI、缺資訊、嚴格且原子的 YOLO 匯入、引用驗證、工時修正、100 候選只覆核 3 個、錯誤／過期回傳、round 重試去重、檢索標籤衝突、快取及 UI 逐項覆核。

Browser command:

```powershell
.venv/Scripts/python.exe -m pytest tests/e2e/test_fp_case_workspace_e2e.py -q --browser chromium
```

結果：1 passed。使用實際 Streamlit app、Chromium 及本機 DINOv2 權重（CPU），測試圖片為可重現的合成反光／裂紋外觀：建案 → 人工覆核 → 正常／瑕疵真實檢索 → 候選 → 引用說明 → 實際 Markdown 下載 → 含圖片 ZIP → 修改 JSON 並上傳 → 覆核讀回 → 服務重啟 → 恢復已覆核狀態。原始對照資料夾逐檔 hash 不變；AL ledger 只有 start／batch，沒有偽造訓練標籤或改善指標。

修正了實際下載曾被頁面 rerun 取消的問題：本功能下載按鈕使用 `on_click="ignore"`，瀏覽器驗證取得的檔案內容及 ZIP 圖片。

OpenSpec：`openspec validate add-false-positive-case-workspace --strict` 通過。新模組 Python 編譯檢查通過。

## 驗證邊界

這些第一輪測試驗證工作流與資料／證據語意，不證明真實生產模型的根因或檢索準確率，也未量出實際節省工時比例。接續交付已涵蓋外部 ONNX 推論／歷史比對、控制實驗、YOLO 訓練交接及固定獨立評估；實際訓練由使用者原有外部訓練流程執行。

首版覆核包採本功能的 review-only 格式；沒有自動向外部標註服務送件。回傳檔必須對應目前案件版本，期間案件有更新需重新下載並合併覆核結果。圖片顯示與覆核範圍支援手動百分比座標；本版不提供拖曳畫框元件。

未暫停的 session 計時不會自動持久化，需手動暫停或補登。搜尋結果引用原對照路徑並驗證 hash；加入候選後才保存可攜的對照圖片副本。完整工作區備份／還原由檔案備份處理，JSON 匯出不提供案件還原 UI。

操作與儲存說明：[使用指南](false-positive-case-workspace.md)。
