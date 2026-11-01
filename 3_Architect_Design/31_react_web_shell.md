# 31 `react_web_shell` — LV React UI 與本機 API 架構

> M24 的本機 React UI/API 垂直切片。需求：[react_web_migration.md](../1_user_needs/react_web_migration.md)；PRD：[react_web_migration_prd.md](../2_PO_PRD/react_web_migration_prd.md)。
> 目前有 LV-WEB-01 Image Classifier、LV-WEB-01-CART-01 持久策展清單、LV-WEB-02 Compare Distributions 及 LV-WEB-03-IMG-01 image-level coverage 的局部流程；fixture case 不代表完整真模型、NativeApp 或產品 parity 已驗收。

## 1. 邊界

- `web/` 是 React/Vite UI；`scripts/react_web_shell.py` 是 App-owned FastAPI adapter，`scripts/react_visualize.py` 與 `scripts/react_compare.py` 執行純領域流程。
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
| `POST /api/v1/visualize/jobs` | 建立 Image Classifier 整圖或 Object Detector 整圖／物件級 run | job ID、queued 狀態 |
| `POST /api/v1/coverage/jobs` | 獨立建立 image-level embedding coverage run | folders、model、初始投影／維度／k、job ID |
| `GET /api/v1/coverage/jobs/{job_id}` / `DELETE ...` | 讀取或取消 coverage job | 共用單一 active-job gate、進度及結果 ID |
| `GET /api/v1/coverage/runs/{run_id}` | 讀取 image-level coverage 結果 | records、投影點、稀疏分數與 percentile 門檻；不回傳路徑或 raw embedding |
| `POST /api/v1/coverage/runs/{run_id}/projections` | 切換 coverage 投影 | PCA、t-SNE、UMAP、LDA、Supervised UMAP；2D／3D |
| `POST /api/v1/coverage/runs/{run_id}/sparsity` | 變更稀疏度近鄰數 k | 原始高維 cosine 稀疏分數及前 10% 門檻 |
| `GET /api/v1/coverage/runs/{run_id}/images/{index}/thumbnail` | 讀取選中資料集影像縮圖 | 僅依 run record index 解析已登錄檔案 |
| `POST /api/v1/compare/jobs` | 建立 A/B 比較 run | folders A/B、模型、粒度、可選 object policy、job ID |
| `GET /api/v1/jobs/{job_id}` | 讀進度／完成／錯誤 | phase、done、total、可取消狀態、結果 ID |
| `DELETE /api/v1/jobs/{job_id}` | 要求取消 | cancellation state；運算於目前影像完成後停止 |
| `GET /api/v1/compare/jobs/{job_id}` / `DELETE ...` | 讀取或取消 A/B 比較 | 與 visualize 共用單一 active-job gate |
| `GET /api/v1/runs/{run_id}` | 讀 run metadata 與投影索引 | 順序穩定的 records、model/method、類別、split、點座標 |
| `GET /api/v1/runs/{run_id}/images/{index}` | 載入單一縮圖／影像 | 只依 run record index 解析已登錄檔案 |
| `POST /api/v1/runs/{run_id}/curation/items` | 將選定 run records 加入持久策展清單 | 只收 record indices、source、reason；以 run 內已登錄路徑取 SHA-256；重複項略過且不覆蓋初次 provenance |
| `GET /api/v1/curation/items` | 讀取策展清單 | opaque item ID、檔名、類別、split、SHA、source、score、reason；不回傳本機路徑 |
| `PATCH /api/v1/curation/items/reason` | 為指定清單項目加註理由 | reason 與更新後的清單摘要 |
| `DELETE /api/v1/curation/items/{item_id}` / `DELETE /api/v1/curation/items` | 移除單項／清空清單 | 更新後清單與移除數 |
| `GET /api/v1/curation/items/{item_id}/thumbnail` | 讀取已在清單中的影像 | 僅依 opaque item ID 解析來源；不接受任意路徑 |
| `GET /api/v1/curation.csv` / `.json` / `.zip` | 明確匯出策展清單 | 下載時提供路徑、身分與理由；ZIP 含影像與 `manifest.csv` |
| `POST /api/v1/runs/{run_id}/text-search` | Chinese-CLIP 文字查詢影像 | query、model ID、依 cosine distance 排序的 run record index；相同距離以 index 穩定排序 |
| `POST /api/v1/runs/{run_id}/text-search.csv` | 匯出文字搜尋結果 | 使用者明確匯出時才包含影像路徑 |
| `POST /api/v1/runs/{run_id}/duplicates` | 掃描 dHash 或 embedding 重複候選 | 成對 run record index、距離、跨 split 篩選；不回傳本機路徑 |
| `POST /api/v1/runs/{run_id}/duplicates.csv` | 匯出重複候選對 | 成對檔名、split、距離及來源路徑 |
| `GET /api/v1/comparisons/{run_id}` | 讀取比較結果 | 群組記錄、類別漂移、policy 與警告；不回傳本機路徑 |
| `POST /api/v1/comparisons/{run_id}/projections` | 計算選定群組的投影 | PCA、t-SNE、UMAP、LDA、Supervised UMAP、維度與物件篩選 |
| `GET /api/v1/comparisons/{run_id}/images/{side}/{index}` | 讀取對應來源影像 | 只依本次比較登錄的 group/index 解析，不接受檔案路徑 |

Job 輸入欄位：資料夾路徑陣列、`mode`（`classifier` 或 `detector`，預設 `classifier`）、`unit`（`image` 或 `object`，預設 `image`）、可選的逗號類別名稱 fallback、`model_ids`（最多 8 個）、`methods`（最多 5 個）和可選 `object_policy`（pad、target resolution、head）。Classifier 只接受整圖；Detector 讀取 YOLO `images/` 與 `labels/`，整圖時一張影像一筆，物件級時一個支援的 bbox 一筆。資料集旁 `classes.txt` 優先於手動 fallback，選取的類別表不一致時明確失敗。舊版單數 `model_id`／`method` 欄位仍可供相容呼叫。此切片的路徑由 React host picker 或開發模式文字欄提供；重複路徑依原流程規則去重。空資料、無 bbox、不可讀影像、缺模型和無法計算的投影回可辨識錯誤或警告，不得靜默替換模型。

## 3. 計算語義

1. 以 `discover_images_classifier(..., allow_flat=True)` 依原排序取得 `{path, split, label}`；用 `partition_readable` 同步剔除壞檔及標籤。
2. 以 `update_manifest` 及相同的 SHA-256 cache key 更新 App-side cache；整圖呼叫 `load_model` 與 `extract_embeddings` 產生真實向量。第一切片不預建縮圖，選取時只回讀 run 登錄的原圖。
3. Detector 物件級沿用 `discover_yolo_objects`、`object_eval.embed_objects`、`crop_bbox` 與 `_adaptive_pad_px`；每個 bbox 產生一筆記錄，保留 class ID、object index、正規化 bbox 及原圖路徑。pad／解析度／head 可由 React 設定；物件 embedding cache key 包含原圖 SHA-256、bbox、object index、model ID 和 policy，寫入 App cache，不寫來源資料夾。
4. 每個模型各自抽取 embedding 並使用獨立 cache；投影保留同一模型的 embedding 與記錄順序。PCA、t-SNE、UMAP、LDA 和 Supervised UMAP 沿用 legacy 固定參數及小樣本邊界；無法計算的選項明示警告，只有所選投影全部不可用時才以 PCA 維持可檢視結果。分類器整圖可選擇固定 UMAP reference：以圖片 SHA-256 對應座標，首次參考 reducer 和座標存於 LV app cache，後續新圖片轉入原座標維度，既有圖片保持原位；移除圖片只影響當次顯示，明確要求 rebuild 才重擬合。監督投影的分群不得解讀為真實可分性。此切片的 deterministic API／可見 UI 驗證及 `umap-learn` 真實 reducer integration 已通過；Windows ARM64 runtime 與 Streamlit 座標數值對照仍待驗。
5. 2D 使用 SVG 散點；3D 使用隨使用者切換才載入的本機 Plotly bundle，點選與鍵盤輔助操作都能讀回同一 run record。投影維度不足時缺少的座標軸補零。點座標與記錄同序；run ID 變更時清空舊選取。跨工具暫存由各後續 case 定義。
6. 使用者點選／框選只傳 record index 和 run ID；後端驗證索引，UI 顯示相同 label、split、檔名和原圖；物件級另外顯示所選 bbox overlay 與 object index。
7. Compare Distributions 對 A/B 資料夾各自遞迴找圖、略過 App cache 並跳過不可讀影像；來源路徑只留在 worker 記憶體，公開記錄使用 index、檔名和來源識別碼。
8. 整圖比較沿用目前 Streamlit 的 centroid cosine drift：用所選模型抽取 embedding，分別正規化兩群的平均向量，再計算 `1 - cosine_similarity`。快取依模型、模式、來源路徑和檔案內容雜湊隔離。
9. 物件比較讀取支援的 detection boxes，按較小資料群執行 object policy autotune，再以同一 policy 對 A/B crop embedding；只對共同類別計算 centroid drift，並明示只出現在單側的類別。操作者可改 pad、解析度與 head，明確套用後重跑；API 會驗證手動 policy，結果同時保留自動候選的整體及物件大小分桶純度。
10. React 可切換比較類別／投影、2D／3D、confidence 與物件短邊門檻；點選及框選只改變 run record 選取，雙欄縮圖仍讀回原圖並可顯示 bbox。投影距離只供檢視，不作為 drift 指標。
11. Image Classifier 整圖 run 可使用已載入的 Chinese-CLIP image embedding 進行文字搜尋；文字塔沿用本機 Chinese-CLIP encoder 的查詢正規化，使用 exact cosine distance 遞增排序，同距離依 record index 排序。dHash 重複掃描重用 run manifest 指紋，embedding 掃描重用 run 向量；兩者依距離排序、最多回傳 200 對，可只保留跨 split 候選。搜尋結果和配對都以 run ID／record index 操作；本機路徑只在使用者明確下載 CSV 時輸出。配對只是人工覆核線索，不自動刪除或判定洩漏。固定 fixture 的 API 和可見 React 流程已驗；真實 Chinese-CLIP 權重、Windows ARM64 模型相容性與數值逐項對照尚未驗；React 尚未實作跨工具 Labeling handoff。
12. Visualize 手選把 record index 和 run ID 傳給 worker；worker 在 `LV_CACHE_DIR/curation_log.jsonl` 寫唯一的 append-only cart store。Store 以正規化來源路徑去重，物件級再以 `obj_index` 區分框；同一路徑再次加入不覆蓋初次 source／score／reason。React 清單提供來源篩選、批次理由、移除、確認清空與 CSV／JSON／ZIP；一般 API 不回傳路徑，只有明確下載會包含路徑。Visible React fixture case 驗證 rerun 清空暫存 selection、cart 不清空，並停止 worker 後由不同 PID 讀回資料。舊 Streamlit `viz_export_list` 只存在 session state；`output/curation_log.jsonl` 是獨立的選取日誌，不是 cart store，因此自動匯入尚未實作；NativeApp 實際 cache-root handoff 也未驗證。

## 4. 實作狀態與整合限制

- API 目前以本機路徑陣列建立 run；React 的資料夾按鈕會呼叫公開 `window.cimHost.chooseFile()`，但 LV 尚未接入 NativeApp 發布包，因此真實 OS picker、grant 驗證與殼管理 worker 的端到端仍待驗。
- 手動路徑欄只供本機開發及測試。React build 與 API 可由單一本機 FastAPI 預覽；NativeApp 交付仍須使用平台 host bridge 和 worker 管理流程。
- Visualize classifier、detector 整圖與物件級的固定 DINOv2-small 可見 UI 路徑已在 Windows ARM64 通過；Compare 的 deterministic API 與整圖／物件 UI E2E 使用 fixture embedder。來源 Streamlit 數值逐項對照、完整 object candidate workflow、NativeApp folder grant／lifecycle 及 Compare 真實模型仍未驗證。詳細 case 狀態見 [React case inventory](../4_PM_Feedback/react_web_shell.md)。
- Curation cart 的 fixture API 與可見 Chromium E2E（含不同 worker PID 重開）已通過；這只完成 Visualize 到 cart 的 case slice，不代表跨工具統一收集器、legacy state import、subset folder export 或 NativeApp cache-root 接線完成。
- Image-level coverage 現以獨立 coverage job 保留舊工具「自己選資料夾／模型並開始分析」流程。高維 cosine 稀疏度與顯示投影分開計算；React 可切換 5 種投影、2D／3D、類別／稀疏上色，2D 框選與 3D 點選可讀回縮圖並明確清除。Fixture API 與可見 Chromium E2E 已通過；真實模型數值對照、NativeApp picker／grant／package 與 OBJ／REF／ATTR case 仍未完成。

## 5. React 驗收契約

**Case ID: LV-WEB-01-IC-01 / LV-WEB-01-IC-02**。前置：凍結 LV source revision/profile、DINOv2-small 權重 SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`、4 張固定 SHA-256 的 Image Classifier fixture、`LV_CACHE_DIR` 指向隔離位置。

- 選 fixture 資料夾、DINOv2-small、PCA，按 Run；UI 顯示 4 個點，API 向量 shape 為 4×384 且值有限、非零。PCA 維度遵循原 Streamlit 主流程 `min(3, max(1, n-2))`，小於 2 維時補零軸。
- IC-02 對每個選取模型分別驗證 embedding cache、shape 和點索引一致；切換已產生的投影及 2D／3D 後，類別／split 與每個點的原始影像對應不變。UMAP 跨 run 固定參考座標另由 IC-03 驗收。
- 類別／split 和每個點的原始影像對應與 Streamlit 基準一致；點選結果顯示相同檔案，不得僅以圖表存在或 API 200 判定。
- 同一輸入重跑使用 LV manifest SHA-256 作為 embedding cache key，結果順序和語義一致；來源圖片與 fixture hash 不變，manifest／embedding 寫入只發生在隔離 `LV_CACHE_DIR`。
- 模型缺失、空資料夾和取消須呈現可恢復狀態；不回報成功或改用假的 backend。worker 關閉與 NativeApp lifecycle 由 AC6 驗收。
- 新 worker PID 讀取相同 run 若有持久結果時必須一致；尚未列入保存語義的暫存 run 不宣稱跨重啟持久。

先以小型 deterministic core/API case 驗證契約，再以真實 NativeApp host bridge、可見 React UI 和真實 DINOv2 完成 case。mock 僅供錯誤／邊界單元測試，不能作遷移通過證據。

## 6. 未涵蓋

Object Detector 整圖及物件級已有 React/API 可見流程與固定 DINOv2-small 的 ARM64 UI run；物件級 candidate handoff、policy autotune、與 Streamlit 數值逐項比較及 NativeApp host bridge 尚待驗。Image-level coverage fixture API／可見 UI 已涵蓋獨立啟動、投影／維度／顏色切換、框選、縮圖讀回及清除；真實模型數值 parity、NativeApp host bridge、OBJ／REF／ATTR 和其他 nested workflows 尚待完成。固定 UMAP reference 的 React 控制、快取座標沿用、增刪圖片與重建有 deterministic API／UI 覆蓋，實際 `umap-learn` 0.5.12 integration 在隔離 CPython 3.11 AMD64 驗證通過；Windows ARM64 runtime 和 legacy 座標數值 parity 尚未驗；本設計不縮小全產品 PRD 範圍。
