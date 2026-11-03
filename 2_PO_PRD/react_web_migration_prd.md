# PRD：VisualLatent React Web 遷移

- 需求來源：[react_web_migration.md](../1_user_needs/react_web_migration.md)
- 立案：2026-11-01；狀態：分期開發中

## 使用者與問題

LV 是成熟的本機影像分析 App。現有 Streamlit 畫面把資料選擇、模型執行、結果視覺化、標註交接與持久狀態放在 Streamlit session。目標是以 React 提供等價的使用流程，Python worker 保留模型和領域運算，NativeApp 管理本機生命週期。

## 產品要求

1. 凍結來源工作樹、Python profile、模型權重、設定與輸入資料，建立可重跑的原行為基準。
2. 以目前 LV 工具列和使用指南為清單，為每個可用工具及巢狀流程分配 case ID；遇到原本失敗、缺素材或外部依賴時分開記錄。
3. 將模型、投影、資料集讀取、持久化及輸出留在 Python；前端只呼叫版本化本機 API。
4. React 必須呈現完整操作結果，包括模型／投影切換、選圖、縮圖、警告、進度、取消、下載及 NativeApp 資料夾橋接。
5. 以可見 React UI 的端到端操作核對同一 case ID；持久狀態案例需停止 worker，以新 PID 重新開啟並讀回。
6. 不改變資料內容或領域判斷。建立 React 專用存檔格式前，先將原有可用保存語義列入 mapping 並驗證轉換。

## 覆蓋清單

下表是追蹤索引，尚不代表巢狀流程已盤點或案例已通過。

| Case ID | 現有工具 | React 驗收範圍 |
|---|---|---|
| LV-WEB-01 | Visualize Embeddings | Classifier／Detector、整圖／物件粒度、模型、投影、互動散點、選圖與後續操作 |
| LV-WEB-02 | Compare Distributions | 整圖與物件級比較、距離指標、類別、投影、選取及結果解讀 |
| LV-WEB-03 | 完整度熱力圖 | 軸與分桶、覆蓋判定、熱圖、候選池補格與匯出 |
| LV-WEB-04 | Model Diagnostics | 模型與單圖選取、預測／解釋、錯誤與證據呈現 |
| LV-WEB-05 | 誤逮案件 | 案件載入、逐區域覆核、證據、續查與待覆核交接 |
| LV-WEB-06 | 瑕疵偵測 | Normal bank、建模、推論、風險排序、熱圖、預標及保存 |
| LV-WEB-07 | 以樣搜樣 | 樣本選取、全庫檢索、候選覆核、類別／框及匯出 |
| LV-WEB-08 | 差異探索 | Good／Bad 群組、穩定差異、熱圖、Top-K 對照與匯出 |
| LV-WEB-09 | 匯出 | 跨工具清單、子集、CSV／ZIP、目的地選擇及來源不變 |
| LV-WEB-10 | 標註回饋 | 送出帳本、匯入／讀回、套用修正及跨程序持久性 |

每個工具的巢狀使用旅程、既有行為錨點與 React 驗收狀態，以 [React case inventory](../4_PM_Feedback/react_web_shell.md) 維護；PRD 這張表只定義 10 個頂層範圍，不代表巢狀覆蓋已完成。

## LV-WEB-01 已實作垂直流程與後續切片

首個垂直流程涵蓋 Image Classifier 整圖模式：選取資料夾、模型與投影，執行 embedding、切換 2D／3D 與類別／split 篩選，並選點讀回對應影像。其 4 張影像 fixture 避免原 Streamlit 小樣本規則把 PCA 投影降成一維。

後續實作加入 Object Detector 整圖與物件級模式：整圖按 YOLO `labels/` class ID 標記影像；物件級則每個 YOLO bbox 各自裁切、抽 embedding 並保留原圖索引，兩種模式都以資料集旁 `classes.txt` 優先於手動類別名稱。物件級可調整 crop padding、解析度與 embedding head，快取依來源影像內容、框、模型與 policy 隔離。固定 DINOv2-small 的可見 React 真模型流程在 Windows ARM64 通過整圖 4×384 及物件級 5×384；UI 可按類別篩選並讀回原圖框選。object candidate handoff 已有 fixture 與真實 DINOv2 可見 React 流程，會核對 Curation 中的物件索引、bbox 與不含本機路徑的回應；它尚未證明與 Streamlit 完整數值逐項比較、完整 2D／3D 物件流程、NativeApp host bridge，也不代表整個 LV 遷移完成。固定 UMAP 參考座標、其他 Visualize 巢狀操作及其餘 LV-WEB case 仍各自保持未完成。

## 完成門檻

- 原基準中每個可用巢狀流程都有 case ID、固定 fixture／模型、API／UI 對照與通過證據。
- 任何數值容差、模型／profile 差異和失敗路徑皆有事前約定及證據。
- 原來可用的流程沒有退化；來源資料零非預期寫入；保存／重開案例在新程序讀回相同語義。
- 沒有未映射流程、mock-only 結果或只通過啟動／health 的案例。
