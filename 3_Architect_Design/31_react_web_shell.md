# 31 `react_web_shell` — LV React UI 與本機 API 架構

> M24 的第一個 UI/API 垂直切片。需求：[react_web_migration.md](../1_user_needs/react_web_migration.md)；PRD：[react_web_migration_prd.md](../2_PO_PRD/react_web_migration_prd.md)。
> 本設計目前只涵蓋 LV-WEB-01 Image Classifier 整圖模式，不表示其他 LV 工具已遷移。

## 1. 邊界

- `web/` 是 React/Vite UI；`scripts/react_web_shell.py` 是 App-owned FastAPI adapter，`scripts/react_visualize.py` 執行純領域流程。
- NativeApp 啟動單一本機 worker，向 UI 提供同源 API 與版本化啟動資訊。API 僅綁 loopback，worker 不啟動 Streamlit。開發模式在 `web/dist` 存在時，FastAPI 也提供同源靜態頁面作獨立本機預覽。
- 資料夾由 NativeApp 公開 host bridge 選取。開發模式可輸入本機資料夾路徑；API 只對本次 run 已登錄的影像 ID 提供內容，不接受任意檔案讀取路徑。
- `LV_CACHE_DIR` 指向 worker 的可寫資料區。manifest、縮圖、embedding 與 projection cache 不落入來源資料集。
- 既有 Streamlit 原流程作為凍結基準；Python 運算語義以 `_utils.py`、`visualize_embeddings.py`、`manifest.py` 和 `interaction.py` 為準，不 import Streamlit UI/session state。

## 2. HTTP 契約

所有路徑使用 `/api/v1`；錯誤為 `{code, message, request_id}`，不得回傳 traceback 或任意本機檔案路徑。

| Method / route | 用途 | 回應重點 |
|---|---|---|
| `GET /api/v1/health` | worker readiness | API 版本、模型目錄狀態、可用模型數 |
| `GET /api/v1/models` | 讀取模型清單 | model ID、名稱、可用狀態；不載入所有權重 |
| `POST /api/v1/visualize/jobs` | 建立 Image Classifier run | job ID、queued 狀態 |
| `GET /api/v1/jobs/{job_id}` | 讀進度／完成／錯誤 | phase、done、total、可取消狀態、結果 ID |
| `DELETE /api/v1/jobs/{job_id}` | 要求取消 | cancellation state；運算於目前影像完成後停止 |
| `GET /api/v1/runs/{run_id}` | 讀 run metadata 與投影索引 | 順序穩定的 records、model/method、類別、split、點座標 |
| `GET /api/v1/runs/{run_id}/images/{index}` | 載入單一縮圖／影像 | 只依 run record index 解析已登錄檔案 |

Job 輸入欄位：資料夾路徑陣列、`model_id`、`method=pca`。此切片的路徑由 React host picker 或開發模式文字欄提供；重複路徑依原流程規則去重。空資料、不可讀影像、缺模型和不支援方法回可辨識錯誤或警告，不得靜默替換模型。

## 3. 計算語義

1. 以 `discover_images_classifier(..., allow_flat=True)` 依原排序取得 `{path, split, label}`；用 `partition_readable` 同步剔除壞檔及標籤。
2. 以 `update_manifest` 及相同的 SHA-256 cache key 更新 App-side cache；呼叫 `load_model` 與 `extract_embeddings` 產生真實向量。第一切片不預建縮圖，選取時只回讀 run 登錄的原圖。
3. 第一切片支援 PCA，沿用原流程固定 seed 和小樣本規則；t-SNE／UMAP／監督投影留待後續 case，不可將監督投影的分群視為真實可分性。
4. 點座標與記錄同序；run ID 變更時清空舊選取。跨工具暫存由各後續 case 定義。
5. 使用者點選／框選只傳 record index 和 run ID；後端驗證索引，UI 顯示相同 label、split、檔名和原圖。

## 4. 實作狀態與整合限制

- API 目前以本機路徑陣列建立 run；React 的資料夾按鈕會呼叫公開 `window.cimHost.chooseFile()`，但 LV 尚未接入 NativeApp 發布包，因此真實 OS picker、grant 驗證與殼管理 worker 的端到端仍是 AC6 待驗項。
- 手動路徑欄只供本機開發及測試。React build 與 API 可由單一本機 FastAPI 預覽；NativeApp 交付仍須使用平台 host bridge 和 worker 管理流程。

## 5. React 驗收契約

**Case ID: LV-WEB-01-IC-01**。前置：凍結 LV source revision/profile、DINOv2-small 權重 SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`、4 張固定 SHA-256 的 Image Classifier fixture、`LV_CACHE_DIR` 指向隔離位置。

- 選 fixture 資料夾、DINOv2-small、PCA，按 Run；UI 顯示 4 個點，API 向量 shape 為 4×384 且值有限、非零。PCA 維度遵循原 Streamlit 主流程 `min(3, max(1, n-2))`，小於 2 維時補零軸。
- 類別／split 和每個點的原始影像對應與 Streamlit 基準一致；點選結果顯示相同檔案，不得僅以圖表存在或 API 200 判定。
- 同一輸入重跑使用 LV manifest SHA-256 作為 embedding cache key，結果順序和語義一致；來源圖片與 fixture hash 不變，manifest／embedding 寫入只發生在隔離 `LV_CACHE_DIR`。
- 模型缺失、空資料夾和取消須呈現可恢復狀態；不回報成功或改用假的 backend。worker 關閉與 NativeApp lifecycle 由 AC6 驗收。
- 新 worker PID 讀取相同 run 若有持久結果時必須一致；尚未列入保存語義的暫存 run 不宣稱跨重啟持久。

先以小型 deterministic core/API case 驗證契約，再以真實 NativeApp host bridge、可見 React UI 和真實 DINOv2 完成 case。mock 僅供錯誤／邊界單元測試，不能作遷移通過證據。

## 6. 未涵蓋

Object Detector 整圖／物件粒度、多模型比較、多投影切換、資料集變更與撤銷、散點框選完整交互、其它九個工具及巢狀工作流，均留在 M24 後續 task；本設計不縮小全產品 PRD 範圍。
