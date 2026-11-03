# 外部模型重播、對照實驗與改善評估

入口：LV → **誤逮案件**。查案後可直接使用 **外部模型**、**訓練交接**、**改善評估**，結果與引用一起保存在同一案件。

## 安裝與支援範圍

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-external-models.txt
```

本機已完成安裝及驗證。使用 ONNX Runtime CPU，不下載任何模型權重或呼叫外部服務。模型須為單檔內嵌權重、單一影像輸入的 ONNX，輸入支援 float32、float16 或符合設定的 uint8；尺寸、NCHW/NHWC 與類別名稱順序須由模型的部署設定指定。

| 輸出格式 | 接受布局 | 解讀方式 |
|---|---|---|
| 分類 | `[1,C]` 或 `[C]` | 單張 top-1 分類；明確指定正常類別、機率／logits activation，低於門檻為拒判 |
| YOLO 類別分數 | `[1,4+C,N]` | cxcywh 加各類分數，對應常見 YOLOv8／v11 非 end-to-end 匯出 |
| YOLO objectness | `[1,N,5+C]` | cxcywh、objectness、各類分數，最終分數為 objectness × class score |
| 已後處理框 | `[1,N,6]` 或 `[N,6]` | xyxy、score、class ID；不再次執行 NMS |

不猜測 raw-head／anchor 解碼、不自動載入 `.pt`、TensorRT engine、自訂 ONNX operator 或 segmentation mask；不支援的輸出會明確拒絕，而不將它錯解為框。可由原模型工具匯出成上述 ONNX 契約後接入。外部資料分離儲存的 ONNX 權重需先合併成單一模型。

实现参考：[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)、[Ultralytics 匯出說明](https://docs.ultralytics.com/modes/export/)。框布局由設定明確指定，不能仅依副檔名判斷。

## 設定與本次推論

在 **外部模型 → 模型設定** 設定基準及可選候選模型。

- 指定路徑、版本名稱、輸出格式、類別順序與正常類別。
- 指定輸入寬高、stretch 或 letterbox，以及判定門檻。
- 進階 JSON 可設定 RGB/BGR、NCHW/NHWC、scale、mean/std、pad、插值、score activation、座標單位、NMS IoU／類別處理、最大框數，以及明確 input/output 名稱。
- 前處理公式：`(pixel * scale - mean) / std`；letterbox 保留實際 resize 寬高與左上 padding，輸出框轉回來源影像的 normalized xyxy。
- 多個輸出時必須指定 output_name；其他非影像輸入需由匯出端固定。
- 儲存設定不執行推論。只有按 **執行外部模型並比對** 才載入模型。

本次結果保存模型 bytes SHA-256、完整前後處理契約及 fingerprint、runtime 版本、實際輸入／原始輸出 tensor hash、模型輸入預覽、原尺寸輸出框及完整 NPZ。可下載 JSON 與完整推論 ZIP。

來源圖片、區域或報案資訊改變後，舊 run 標為歷史，不可拿來直接執行新實驗。模型檔內容變更也會產生不同契約。

## 歷史判定重播

「本次推論成功」與「重現當時完整輸出」分開。

若有當時部署紀錄，在執行頁匯入下列結構：

```json
{
  "origin": "deployment",
  "recorded_at": "2026-09-01T00:00:00Z",
  "image_sha256": "當時原圖內容 hash",
  "contract_id": "當時模型及前後處理契約 hash",
  "prediction": {
    "task": "detection",
    "alarm": true,
    "detections": [
      {"class": "crack", "score": 0.91, "box": [0.1, 0.2, 0.4, 0.6]}
    ]
  }
}
```

分類 prediction 則包含 `task="classification"`、`alarm`、`class`、`decision`（類別名或 `abstain`）及完整 `scores` 類別字典。

`contract_id` 是完整契約 canonical JSON（排序 key、無多餘空白、UTF-8）的 SHA-256，包含實際模型 hash、input/output 名稱、dtype 與設定；以 `fp_external_runtime.Runtime.identity` 為準。路徑不進身分，模型內容與設定進身分。

只有原圖、歷史來源聲明、時間、契約與影像身分足夠，才比較完整結果。類別、框、分數、拒判和最終警報都要相符；分數接近但框不同仍為 mismatch。缺資料／截圖可執行本次推論，但歷史 parity 是 unavailable。

本次 JSON 的 `origin` 是 `lv_local_inference`，不能自我驗證為當時紀錄。不要把本次輸出改名成 deployment 來宣稱已重現；歷史紀錄的真實性由提供者負責。

## 對照實驗

選擇已保存且仍有效的基準 run，固定觀察的模型類別與案件目標位置，指定要修改的區域。

- **亮度**：指定區域亮度變化，另執行倍率 1.0 的控制組。
- **遮擋**：指定區域固定灰階遮擋，另抽取不同位置、等像素面積的控制組，記錄 seed。無法建立不同控制組時明示不足。
- 實驗前先確認原圖輸入與輸出仍可重現；模型內容或脈絡變更時要求重跑基準。
- 需要人員聲明修改保留目標有效外觀，仍須檢查實際修改圖。結果永遠只是单圖敏感性證據，不自動確認根因。
- Detection 固定類別與 ROI，以 IoU ≥0.1 找同位置候選並區分最終輸出框。只有 post-NMS 輸出時，無法取得 NMS 前分數。
- 目標消失／未匹配時 `score_drop=null`，不能當作分數降為 0；未确认目標保留時，保留觀察但標不可判讀。

完整修改輸入與 raw tensor 保存在各 run artifact，實驗 JSON 連到各 run，證據可直接引用進案件說明。

## 訓練資料交接

**覆核批次** 仍是 review-only；**訓練交接** 是另一個明確操作。

1. 逐項覆核所有要交接影像上的案件／候選區域。
2. 明確選取已檢查完整標註的影像，填寫確認者、依據及真瑕疵人工類別對應。
3. 輸出 YOLO 偵測 ZIP：原圖副本 `images/`、normalized cxcywh `labels/`、`classes.txt`、完整 provenance。

整圖負樣本需要明確 full-frame normal 覆核。局部正常不夠；存在真瑕疵或其他案件矛盾時拒絕。混合正常／真瑕疵影像會保留已確認真框，不能整張清成正常。待覆核、爭議、資訊不足和已知截圖不可輸出。訓練類別須由人員指定，不能自動沿用預測名稱。

此包交給原本外部訓練流程，不自動重訓未知架構；也不自動切分 train/val/test。完成訓練後，將新版 ONNX 設為候選模型，直接執行下節比較。

## 前後模型改善評估

使用固定 JSON manifest，兩模型逐張使用同一份 bytes 與真值。

```json
{
  "schema_version": 1,
  "purpose": "independent",
  "independent_declared": true,
  "split_basis": "按批次分開，未參與訓練、選樣或門檻調整",
  "items": [
    {"id": "n1", "path": "normal.png", "truth": "normal", "verified": true, "group_id": "lot-A"},
    {"id": "d1", "path": "defect.png", "truth": "defect", "verified": true, "group_id": "lot-B"}
  ]
}
```

相對路徑需填基準資料夾。選 `purpose="case_regression"` 可以重跑已知報案，但不能宣稱獨立改善。

若需框層級 precision／recall，每一張都要有 `annotations_complete=true` 及 `boxes`，正常為空陣列，瑕疵為 `{class, box:[x1,y1,x2,y2]}`。只有圖片真值時，報告只說明圖片警報／漏檢率，不能證明瑕疵定位正確。

驗證規則：

- 圖片須有明確 verified normal／defect；拒絕重複內容、矛盾真值、執行中檔案變更。
- 獨立資料與本工作區報案、候選、已搜尋對照及訓練匯出內容不得重疊。外部訓練全集未提供時，仍需使用者独立性聲明，無法自動證明完全無洩漏。
- 類別順序、正常定義及任務種類須一致；模型內容或前後處理可以不同，但完整記錄兩者契約。
- 同時提供正常誤逮、瑕疵漏檢、支持數、群組結果及拒判數。
- 預先設定最低支持數、要求誤逮下降、容許漏檢增加。缺支持、拒判或漏檢增加超標不能通過。若有完整框標註，也限制框層級召回下降。
- 通過只代表該模型組合在該資料集符合設定門檻，不是自動部署或因果根因確認。門檻為零改善時，只表示未劣於指定限制，應閱讀實際率差。

## 從熱圖整理成可核對的說明

在 Diagnostics 或回覆頁先選擇「調查問題」：指定類別／框分數、框的位置、門檻／NMS，或機台報警。目前 CAM 的對象是指定候選分數；其他題型會列出定位真值、後處理紀錄或機台規則等缺少的證據，不把熱圖當作整段報警的原因。

啟用遮擋驗證後，可選標準比例或較小的 2%、5%、10%。系統以同形狀、同面積的平移區域作控制，分別使用常數與模糊填補。若有限搜尋找不到足夠低重疊位置，畫面顯示實際可用／要求數量；這不是通過，可嘗試較小比例或其他原圖。不同填補或比例觀察方向不一致時，摘要保留「不穩定」。

已保存的 CAM 可在回覆頁直接查看原有修改圖，核對全部圖片後填寫覆核者、依據及目標外觀是否保留。確認紀錄綁定本次修改圖指紋與目標身分；修改圖或目標變動不能沿用確認。確認外觀保留也不等於證明根因或改善。未確認時仍可回覆，但摘要明列待查項目。

受控摘要優先列人工真值、歷史部署重現範圍、原候選分數及最多六組熱區與控制差異；逐圖數值留在詳細紀錄與工程附錄。差值為正僅表示熱區修改後分數相對更低或上升較少，不代表反光等語意成因。重新匹配到的替代框不能取代原候選固定分數。

調查題型、證據選取和確認紀錄的變更不會覆寫人工回覆。只有明確按「依選取證據重新產生事實草稿」才更新草稿。保存版本會記錄人工文字與當時受控摘要；HTML 把自動摘要與人工文字分開，並列出引用的證據編號。

批次核對需先完成區域真值覆核，再填人工群組、樣本角色、覆核者與依據。核實角色必須與區域正常／瑕疵判讀一致；同圖多框不算獨立影像，不同模型、head 與控制協定分組呈現。沒有框的正常圖使用資料庫改善評估，不當成 CAM 支持。完成批次後可明確保存歷史摘要作回覆證據；群組由人工宣告，樣本數本身不能證明現場效果或節省工時。

## 儲存與可重現測試

案件根目錄新增 `external/<case_id>/<run_id>/` 與 `training/<case_id>/`；案件 SQLite 快照保存 run／experiment／evaluation／training_export 引用。外部 raw arrays 不写入來源資料夾。

測試模型可用以下命令產生，僅供驗證接入與流程，**不是訓練過的瑕疵模型**：

```powershell
.venv/Scripts/python.exe scripts/fp_onnx_demo.py output/demo-classifier.onnx
.venv/Scripts/python.exe scripts/fp_onnx_demo.py output/demo-detector.onnx --detection
```

Demo 輸入為 32×32，分類類別 `good,crack`、normal `good`、activation `none`；detector 用 xyxy 格式與單類 `crack`。其分數依影像亮度計算，用來驗證數值和前後比較，不代表實際生產能力。
