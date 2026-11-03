## Why

LV 已有 DINO 分類器 Grad-CAM 與外部 ONNX 誤逮調查，但二者沒有接線：工程師不能由報案中的 YOLO 框直接取得該模型的歸因、驗證並引用為案件證據。使用者明確將 YOLO／Grad-CAM 列為重要需求，本變更將它列為核心交付，補足上一版只完成重播與擾動實驗的缺口。

## What Changes

- 保留既有 DINO Grad-CAM、兩個入口與 V1 契約；抽出執行分流，擴充同一 Model Diagnostics 工作台。
- 新增有版本與能力檢查的原版 YOLOv5、YOLOv8、YOLO11 軸對齊 Detect 可微 adapter；每族須有真實 forward/backward 測試才可宣稱支援。
- 從案件選取實際偵測框，追溯模型原始候選與警報分數，以完整原圖及相同前處理計算 Grad-CAM。
- 同一可重用、不可變版本模型 profile 描述部署 runtime 與對應梯度 backend；分開顯示跨 backend 對齊、歷史重播、熱圖忠實度及定位驗證。
- 新增以模型上傳為主要入口的「我的模型」流程：匯入、能力檢查、必要設定核對、圖片試跑、儲存回案件；支援後補對應 `.pt`，保留大型模型本機路徑入口。
- 標準正向 Grad-CAM 與 signed 診斷各有明確語意；LV 輔助熱圖不能替代 YOLO 模型歸因。
- 將熱圖原始數值、目標／方法／模型身分與實驗保存為案件證據，接上說明引用、區域對照搜尋與 AL 待覆核批次。
- 增加固定 scalar 忠實度檢查、獨立的最終偵測結果變化、可用 mask 的定位評估及端到端回歸門檻。
- 補齊整體查案操作：圖片先行的推論、批次素材／資料集匯入與配對、可修正的圖上選框、零框／部分紀錄處理、支援／反駁證據、圖文回覆、無必填 JSON 的訓練／評估與持久工作恢復。驗收以產品檢視 U01–U16 使用情境為準。

## Capabilities

### New Capabilities

- `case-linked-detector-attribution`: 既有診斷工作台的 YOLO 歸因、模型對應、案件證據、驗證與覆核整合。

### Modified Capabilities

無已歸檔基準 spec。既有 `add-explainability-model-diagnostics`、`add-external-model-investigation` 的功能保留；本 change 新增橋接能力，不將舊 change 的未完成研究項目改標完成。

## Impact

- 既有：`scripts/app.py`、`explain_page.py`、`explain_ui.py`、`explanation_render.py`、`explain_contracts.py`、`explain_context.py`、`fp_case_page.py`、`fp_external_page.py`、`fp_cases.py`、`fp_investigation.py`。
- 新增規劃：provider 執行服務、detector V2 契約、YOLO adapters、模型 profile registry、跨 backend 對齊與案件歸因服務／artifact。
- 相依套件：分離的可選 PyTorch YOLO 環境與固定版本；原版 v5 與 Ultralytics v8/11 不能以同一未驗證 loader 混用。
- 不修改來源圖片／標籤、不自動訓練或部署外部模型。只具推論能力的 ONNX 路徑仍提供對照實驗，不能標示為 Grad-CAM。
- 本文件是三 agent 三輪規劃產物；新增 YOLO 歸因尚未實作或驗收。
