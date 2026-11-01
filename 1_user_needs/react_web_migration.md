# 需求：將 VisualLatent 從 Streamlit 遷移為 React Web

## 使用者原始目標

使用者決定不再支援 Streamlit，要求把現有 App 改寫為 React Web；React 介面由 NativeApp 啟動，透過本機 API 呼叫 App 的 Python 與本機模型。這份需求聚焦 LV，不改變 LV 的模型、演算法、資料語義或既有可用操作。

## 使用者眼中的成功

- 既有可用的 LV 工具和巢狀流程都能在 React UI 操作。
- 相同模型、設定和輸入資料得到相同領域語義；數值差異只能使用預先明訂的容差。
- Python 模型留在本機 worker；React 透過版本化 API 呼叫，不把模型載入瀏覽器。
- 資料夾選擇、進度、取消、錯誤、選圖、匯出、跨工具交接和保存／重開，依原操作語義提供。
- 每個原有可用流程都有穩定案例 ID、Streamlit 基準、React 操作、API 對照和自動驗收；未對應案例為零後，才可宣告遷移完成。
- 開發版本在 LV GitHub repo 以可審閱的分支發布；來源工作樹的既有未提交內容維持原樣，遷移提交只納入經核對的遷移變更。

## 非目標

- 不改寫模型權重、演算法或資料格式來迎合前端。
- 不以 mock、HTTP 200、截圖或另一個 backend 代替真實模型與可見 UI 驗收。
- 不在本需求內取消 NativeApp 管理本機 worker／模型資源的方向。
