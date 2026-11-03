# 31 `react_web_shell` — LV React UI 與本機 API 架構

> M24 的 React UI/API 邊界與分期切片。需求：[react_web_migration.md](../1_user_needs/react_web_migration.md)；PRD：[react_web_migration_prd.md](../2_PO_PRD/react_web_migration_prd.md)。
> 目前有 LV-WEB-01 Image Classifier 整圖、固定 UMAP reference、3D KMeans 區塊導覽、Object Detector 整圖／物件級切片、selection viewer、整圖相似／文字／重複查詢、離群／分歧排名複查與持久 curation cart；LV-WEB-02 Compare Distributions 整圖 A/B 也有部分實作。各 journey 是否完成，依 case inventory 的證據狀態判定。

## 1. 邊界

- `web/` 是 React/Vite UI；`scripts/react_web_shell.py` 是 App-owned FastAPI adapter，`scripts/react_visualize.py` 與 `scripts/react_compare.py` 執行各自的純領域流程。
- NativeApp 啟動單一本機 worker，向 UI 提供同源 API 與版本化啟動資訊。API 僅綁 loopback，worker 不啟動 Streamlit。開發模式在 `web/dist` 存在時，FastAPI 也提供同源靜態頁面作獨立本機預覽。
- 資料夾由 NativeApp 公開 host bridge 選取。開發模式可輸入本機資料夾路徑；API 只對本次 run 已登錄的影像 ID 提供內容，不接受任意檔案讀取路徑。
- `LV_CACHE_DIR` 指向 worker 的可寫資料區。manifest、縮圖、embedding 與 projection cache 不落入來源資料集；Compare 的 embedding cache 也使用此 app-owned cache。
- 既有 Streamlit 原流程作為凍結基準；Python 運算語義以 `_utils.py`、`visualize_embeddings.py`、`manifest.py` 和 `interaction.py` 為準，不 import Streamlit UI/session state。

## 2. HTTP 契約

所有路徑使用 `/api/v1`；錯誤為 `{code, message, request_id}`，不得回傳 traceback 或任意本機檔案路徑。

| Method / route | 用途 | 回應重點 |
|---|---|---|
| `GET /api/v1/health` | worker readiness | API 版本、模型目錄狀態、可用模型數 |
| `GET /api/v1/models` | 讀取模型清單 | model ID、名稱、可用狀態；不載入所有權重 |
| `POST /api/v1/visualize/jobs` | 建立 Image Classifier／Object Detector 的整圖或物件級 run | job ID、queued 狀態；detector 可帶粒度、object policy 與類別來源 |
| `GET /api/v1/jobs/{job_id}` | 讀進度／完成／錯誤 | phase、done、total、可取消狀態、結果 ID |
| `DELETE /api/v1/jobs/{job_id}` | 要求取消 | cancellation state；運算於目前影像完成後停止 |
| `GET /api/v1/runs/{run_id}` | 讀 run metadata 與投影索引 | 順序穩定的 records、model/method、類別、split、點座標 |
| `POST /api/v1/runs/{run_id}/rankings` | 依 run/model 對未選取資料或選取 indices 排序 | 回 record index 與可用分數；可按離群度、標籤分歧、檔名或目前投影的空間順序排列，不回 embedding、cache path 或來源路徑 |
| `POST /api/v1/runs/{run_id}/blocks` | 對目前投影的可見 record indices 執行 KMeans | K（4–30）、非空 block membership 與 3D centers；不足樣本時降低 effective K |
| `POST /api/v1/runs/{run_id}/block-projection` | 將選定 block 的原始 embedding 投影到 2D | 只回 record index 與 PCA 座標；cache 路徑及 embedding 不回給 UI |
| `POST /api/v1/runs/{run_id}/similar` | 以 run record 搜尋相似整圖 | 指定模型、query index、K；回傳距離升冪的候選 index／metadata／cosine distance，不回 embedding 或來源路徑 |
| `GET /api/v1/runs/{run_id}/similar.csv` | 匯出查詢圖與目前相似候選 | 下載 CSV；欄位與順序沿用 Streamlit `records_to_csv`，query 在首列，後續依相似距離排列 |
| `POST /api/v1/runs/{run_id}/text-search` | Chinese-CLIP 文字向量搜尋整圖 | 指定文字、模型與 K；回傳距離升冪候選，不排除和文字向量最相符的影像 |
| `GET /api/v1/runs/{run_id}/text-search.csv` | 匯出以文搜圖候選 | 下載與 Streamlit `records_to_csv` 同欄位的候選 CSV |
| `POST /api/v1/runs/{run_id}/duplicates` | 掃描 phash 或 embedding 重複候選 | 支援 Hamming／cosine threshold 與跨 split 篩選；回傳有序成對 record metadata，不刪除來源資料 |
| `GET /api/v1/runs/{run_id}/images/{index}` | 載入單一縮圖／影像 | 只依 run record index 解析已登錄檔案；detector 可用 `overlay=true` 讀取 YOLO overlay，object run 可用 `view=source` 讀取來源整圖 |
| `GET /api/v1/curation-cart` | 讀取已保存的策展清單 | 回 item identity、來源、理由與 hash，不回本機路徑 |
| `POST /api/v1/curation-cart/items` | 依 run ID／record index 加入清單 | 接受 manual/search/similar/duplicate/outlier/disagreement 來源；同一來源路徑重複加入保留第一次 metadata，object rows 另用 object index 區分 |
| `PATCH /api/v1/curation-cart/reasons` | 更新指定清單項目的理由 | 可對 UI 目前篩選的項目批次套用文字理由 |
| `DELETE /api/v1/curation-cart/items/{item_id}`／`DELETE /api/v1/curation-cart` | 移除項目／清空整份清單 | 清空請求必須帶 `confirm: true` |
| `GET /api/v1/curation-cart.csv`／`.json`／`.zip` | 匯出清單 | CSV／JSON 保留來源身分和理由；ZIP 含原圖及 manifest；缺檔記錄在 manifest |
| `POST /api/v1/compare/jobs` | 建立整圖 A/B comparison job | Folder A／B 各一至 32 個資料夾、LV feature model；同側影像合併並去重 |
| `GET /api/v1/compare/jobs/{job_id}` / `DELETE /api/v1/compare/jobs/{job_id}` | 讀進度／要求取消 | 共用單一 worker 槽，回報結果 ID 或可恢復錯誤 |
| `GET /api/v1/compare/runs/{run_id}` | 讀 A/B 形心漂移、計數、警告與順序穩定的影像記錄 | 不回傳私有本機圖片路徑或 embedding |
| `POST /api/v1/compare/runs/{run_id}/projections` | 建立／重用指定投影 | `method`（PCA、t-SNE、UMAP、LDA、Supervised UMAP）與 `dimension`（2／3）；點以 group/index 對應影像 |
| `GET /api/v1/compare/runs/{run_id}/images/{group}/{index}` | 讀取 A 或 B 的選取圖片 | group/index 只在該 run 的圖片清單內解析 |

Job 輸入欄位：資料夾路徑陣列、`model_ids`（最多 8 個）、`methods`（PCA、t-SNE、UMAP、LDA、Supervised UMAP，最多 5 個）、`mode`（`image_classifier` 或 `object_detector`）、可選 `classes_file_text` 與 `class_names`。舊版單數 `model_id`／`method` 欄位仍可供相容呼叫。此切片的路徑由 React host picker 或開發模式文字欄提供；重複路徑依原流程規則去重。空資料、不可讀影像、缺模型和無法計算的投影回可辨識錯誤或警告，不得靜默替換模型。

Image Classifier job 可附 `umap_reference` 與 `rebuild_umap_reference` 布林欄位。手動重建只在固定參考系已啟用且投影包含 `umap` 時接受，否則回 `422 INVALID_UMAP_REFERENCE`。

## 3. 計算語義

1. 以 `discover_images_classifier(..., allow_flat=True)` 依原排序取得 `{path, split, label}`；用 `partition_readable` 同步剔除壞檔及標籤。
2. 以 `update_manifest` 及相同的 SHA-256 cache key 更新 App-side cache；呼叫 `load_model` 與 `extract_embeddings` 產生真實向量。第一切片不預建縮圖，選取時只回讀 run 登錄的原圖。
3. 每個模型各自抽取 embedding 並使用獨立 cache；投影保留同一模型的 embedding 與記錄順序。PCA、t-SNE、UMAP、LDA 和 Supervised UMAP 沿用 legacy 固定參數及小樣本邊界；無法計算的選項明示警告，只有所選投影全部不可用時才以 PCA 維持可檢視結果。監督投影的分群不得解讀為真實可分性。
   固定 UMAP 參考系沿用 `umap_ref.stable_umap`：以圖片 SHA-256 識別已知影像，已知點讀取保存座標，新影像由保存的 reducer `transform`；首次使用、手動重建、embedding 維度／輸出維度不相容或 reference 損壞時重新 fit。Reference 放在 `LV_CACHE_DIR` 的第一個來源資料夾／模型 cache 內，不寫入來源資料集；UI 顯示首次重建或重用與新增影像數。
4. 2D 使用 SVG 散點；3D 使用隨使用者切換才載入的本機 Plotly bundle，點選與鍵盤輔助操作都能讀回同一 run record。投影維度不足時缺少的座標軸補零。點座標與記錄同序；run ID 變更時清空舊選取。跨工具暫存由各後續 case 定義。
5. 使用者點選／框選只傳 record index 和 run ID；後端驗證索引，UI 顯示相同 label、split、檔名和原圖。

### LV-WEB-01-IC-04 — 3D KMeans block navigation

- 3D 檢視依目前 split／class 篩選後的投影點，以 `KMeans(n_init=4, random_state=42)` 分成最多 K 塊；中心以 `#block_id` 標在 Plotly 3D scene，區塊選單只列非空群及點數。
- 選一塊後，worker 從 app-side embedding cache 讀取該模型的原始向量，沿用 legacy `PCA(n_components=2, random_state=42)` 展開成 2D。小於 3 點時沿用投影的 x/y。
- 2D 區塊圖支援 Plotly 點選、box 與 lasso。新選取 record indices 併入既有選取；切 2D／3D 不清掉選取，只有明確 Clear selection 會清除。run state 對 UI 只回 record IDs 與座標，不回本機 cache path 或原始 embeddings。

### LV-WEB-01-OD-01 — Object Detector whole-image analysis

- React 的 dataset mode 選擇 Object Detector 後，每張影像產生一個 embedding point；來源沿用 legacy `discover_images` 的 `images/`、`JPEGImages/` 與支援標註的平鋪目錄 discovery，不會裁切 YOLO bounding box。
- 類別來源順序為使用者挑選的文字檔、第一個資料夾可偵測到的 `classes.txt`／`data.yaml`、手動逗號分隔類別名。YOLO image-level labels 依 class ID 映射成名稱；缺少所有類別來源時回 `CLASSES_REQUIRED`，所選但空白的類別檔回 `CLASSES_FILE_EMPTY`。
- 多模型、投影方法、split/class filter、點選與來源影像回讀沿用 Image Classifier run contract。API 對外回傳 resolved class names/source 類型，不回傳所選類別檔路徑或本機影像路徑。
- `test_case_lv_web_01_od_01_detector_whole_image_class_sources_and_models` 與 `test_lv_web_01_od_01_detector_whole_image_and_class_priority` 使用 deterministic features，證明類別優先序、多模型 API/UI、可見錯誤及來源檔案不變；尚未證明真實模型數值 parity 或 NativeApp host 流程。

### LV-WEB-01-OD-02 — Object Detector object-level embeddings

- Object Detector 可選 whole image 或 object 粒度。Object 粒度沿用 `discover_yolo_objects` 的 image-then-box 順序，每個有效標註框成為一筆 record；class ID 依已解析的類別名稱映射。未發現標註物件時回 `NO_OBJECTS`。
- Object embedding 沿用 `object_eval.embed_objects`，在原圖上以 bbox 裁切；policy 欄位為 `pad`（0–0.5，預設 0.12）、`target_res`（168／196／224／280／336／392／448，預設 224）與 `head`（`cls`／`meanpool`，預設 `cls`）。每邊 padding 依物件長邊換算並限制在 8–64 原圖像素。
- embedding cache 寫入 `LV_CACHE_DIR` 下按來源資料夾、policy 與 model 分開的 app-owned 路徑。key 包含來源圖片 SHA-256、物件順序、class ID 與 bbox；不寫回來源資料夾。run 對外只提供穩定 record index、檔名、split、label、object index、bbox 與 score，不提供原圖路徑或 cache path。
- class filter 與 2D／3D 檢視沿用 run record index，切換時選取對應同一候選物件。選取物件的圖片 route 依 run index 從來源圖即時計算相同 padding crop；3D KMeans block explorer 僅用於 Image Classifier，避免對不同的 object embedding contract 暗示支援。
- `test_case_lv_web_01_od_02_object_embeddings_policy_order_and_crop_readback` 與 `test_lv_web_01_od_02_visible_object_embeddings_policy_filters_and_3d` 使用 deterministic features，覆蓋 policy 傳遞、bbox crop、候選順序、類別 filter、2D／3D、選取讀回及來源 bytes 不變；不構成真實模型數值 parity 或 NativeApp host 證據。

### LV-WEB-01-VIEW-01 — Selection gallery and image viewer

- Scatter point click replaces selection; Shift-click adds points. The selected-thumbnail wall follows the legacy spatial order: high y first in twelve horizontal bands, then ascending x within each band. Clicking a thumbnail changes the active viewer record without changing the selected set.
- Previous／next follows the same displayed thumbnail order. Closing the viewer clears only the active image. In Object Detector mode, the YOLO overlay toggle defaults on and remains in the same state while navigating images; requests use only run ID and record index. The worker draws overlays in memory and never writes to source images.
- Object-level runs retain the OD-02 crop preview and also allow viewing the source image with its YOLO overlay. Switching Image Classifier／Object Detector or object granularity clears the current run and selection; returning to the earlier mode does not restore stale results.
- `test_case_lv_web_01_view_01_detector_image_route_draws_yolo_overlay` checks distinct raw/overlay responses and source bytes; `test_lv_web_01_view_01_selection_gallery_yolo_navigation_and_mode_isolation` visibly checks selection order, thumbnail review, overlay persistence, close behavior, and mode isolation using deterministic features. These checks do not establish NativeApp picker/grant integration.

### LV-WEB-01-SEARCH-01 — Whole-image similarity search

- Start from the active image (or the first selected image); rank other whole-image records by exact cosine distance in ascending order, excluding the query record itself. Stable ties use record index. The result count defaults to at most nine and can be changed up to the run size.
- A candidate can become the next query. Query history chips restore earlier steps. Changing the displayed feature model reruns the same current query against that model's run-scoped embedding cache; unavailable or corrupt cache returns a visible error instead of showing stale results.
- CSV export places the current query first, then the ranked candidates, and reuses the Streamlit `records_to_csv` columns (`index, filename, path, label, split`). Export is generated by the worker only after the user requests it. This case covers whole-image runs; object-level search remains outside this case.
- `test_case_lv_web_01_search_01_similarity_reorders_by_model_and_exports` and `test_lv_web_01_search_01_visible_chain_cross_model_and_csv` use deterministic vectors to check distance order, candidate pivot, model-specific recomputation, CSV contents and unchanged source files. They do not establish real-model ranking parity or NativeApp integration.

### LV-WEB-01-SEARCH-02 — Text search and duplicate/leakage review

- Chinese-CLIP text search uses the model's local text tower in the same space as its image embeddings. Traditional Chinese normalization remains owned by `ChineseClipTextEncoder`; exact cosine distance orders the top K images, including the closest image because a text query has no self record. A text result can pivot into the SEARCH-01 image-query chain. Text CSV contains only the ranked image records, matching Streamlit `records_to_csv`.
- Duplicate review uses the existing 64-bit dHash and Hamming threshold (default 4), or the selected model's image embedding cosine distance (default 0.05). Optional cross-split filtering returns only pairs whose record splits differ. Results keep the existing 200-pair cap, sort by distance then indices, and show at most 50 pairs in a side-by-side human review list. This is a candidate list; it never deletes or decides whether an image is bad.
- Run-scoped phashes and embedding caches remain private to the worker. The visible UI can select either side of a pair to inspect it in the image viewer. Persistent curation/cart handoff remains in `LV-WEB-01-CART-01`.
- `test_case_lv_web_01_search_02_text_search_and_duplicate_leakage_review` and `test_lv_web_01_search_02_visible_text_and_duplicate_leakage_review` use injected deterministic vectors and a train/val byte-copy fixture. They cover text order/pivot/export, phash and embedding pair detection, split filtering and unchanged source files. The existing Streamlit `test_q_text_to_image_search` is conditional on Chinese-CLIP weights; this machine has no local weights, so real-model text-ranking parity remains unverified.

### LV-WEB-01-RANK-01 — Outlier and label-disagreement review

- Whole-image and object-level runs compute both scores per model with the legacy formulas: `k=min(5, N-1)`, mean cosine distance to self-excluded nearest neighbors for outlier score, and the fraction of differently labeled neighbors for label disagreement. Scores stay in worker-private run state; the ranking endpoint returns only run record indices and the requested score.
- With no selected points, the gallery shows the top 50 by outlier score by default; choosing label disagreement changes that order. Spatial and filename choices with no selection retain the legacy outlier fallback. With selected points, the selected records are limited by the active split and sorted by the selected criterion; spatial order uses the current projection's 12 horizontal bands. The separate selected-thumbnail wall and selected set remain unchanged.
- The ranked gallery opens the image viewer in rank order so previous/next follows the displayed rank list. Both scores are explicitly described as manual review cues, never as quality or mislabel verdicts. Runs with fewer than three records do not invent score rankings.
- `test_case_lv_web_01_rank_01_metrics_and_review_orders_match_streamlit` and `test_lv_web_01_rank_01_visible_ranked_gallery_and_viewer_navigation` use deterministic vectors to compare both score orders, filename/spatial order, visible criterion switching, viewer navigation, and unchanged source files. Real-model score parity and NativeApp grants／ARM64 remain unverified.

### LV-WEB-02-IMG-01 — Image-level Compare Distributions

- 兩側各接受一至 32 個資料夾；同一側依選取順序合併，重複路徑／圖片去重。每個根資料夾先使用 `compare_distributions.get_image_paths` 的直接 JPG／JPEG／PNG 全域排序；若該根沒有直接圖片，才遞迴搜尋子資料夾並略過 embedding、crop、thumbnail cache。壞檔略過並顯示警告。
- 用選定 LV feature model 產生 A、B whole-image embeddings。A 與 B 分別取向量平均並正規化，漂移為 `round(max(0, 1 - dot(normalized_mean_A, normalized_mean_B)), 4)`；這與 `scripts/app.py::_compute_compare_whole` 相同。SHA-256 cache 寫入 `LV_CACHE_DIR`，不寫回來源資料夾。
- `POST /compare/runs/{run_id}/projections` 以同一組 embeddings 計算 PCA、t-SNE、UMAP、LDA 或 Supervised UMAP；沿用 Streamlit 固定參數、小樣本零座標及 LDA 補軸語義。切換投影或維度時點的 group/index 身分不變。小樣本之外的投影失敗／依賴缺失會明確回報，不靜默替換方法。
- UI 顯示整體漂移與 A/B 張數，支援投影切換及 2D／3D 檢視。2D 點選／矩形框選後，兩側畫廊顯示選取圖片；沒有選取時顯示前 12 張。3D KMeans 區塊 picker 與 2D 區塊 drilldown、套索選取和加入策展清單仍屬後續 case。
- `GET /compare/runs/{run_id}` 對外只提供漂移、檔名／index、計數與警告；embedding 和本機檔案路徑留在 worker。圖片只依 run ID、group 與 index 回讀。
- 目前 Streamlit 整圖路徑不包含 FID/KID/LPIPS/SSIM、Coverage Gap 或 metric export。`tests/e2e/test_scenarios_r1.py::test_s05_compare_distributions_metrics` 已 skip，原因是 Compare 改為 YOLO 物件級按類別流程；不得將其當作本 case 的正向行為錨點。現行整圖基準是 `_compute_compare_whole` 與 `_render_compare_by_class`。
- `tests/e2e/test_react_web_shell_e2e.py::test_lv_web_02_img_01_visible_compare_contract` 和 API tests 使用 deterministic feature function，證明 React／API 接線、資料回讀與取消；它們不構成真實模型或 Streamlit 數值 parity 收據。

## 4. 實作狀態與整合限制

- API 目前以本機路徑陣列建立 run；React 的資料夾按鈕會呼叫公開 `window.cimHost.chooseFile()`，但 LV 尚未接入 NativeApp 發布包，因此真實 OS picker、grant 驗證與殼管理 worker 的端到端仍是 AC6 待驗項。
- 手動路徑欄只供本機開發及測試。React build 與 API 可由單一本機 FastAPI 預覽；NativeApp 交付仍須使用平台 host bridge 和 worker 管理流程。

## 5. React 驗收契約

**Case ID: LV-WEB-01-IC-01 / LV-WEB-01-IC-02 / LV-WEB-01-IC-03 / LV-WEB-01-IC-04 / LV-WEB-01-OD-01 / LV-WEB-01-OD-02 / LV-WEB-01-VIEW-01 / LV-WEB-01-SEARCH-01 / LV-WEB-01-SEARCH-02 / LV-WEB-01-RANK-01 / LV-WEB-01-CART-01**。前置：凍結 LV source revision/profile、固定 fixture 與模型權重身分、`LV_CACHE_DIR` 指向隔離位置。

- 選 fixture 資料夾、DINOv2-small、PCA，按 Run；UI 顯示 4 個點，API 向量 shape 為 4×384 且值有限、非零。PCA 維度遵循原 Streamlit 主流程 `min(3, max(1, n-2))`，小於 2 維時補零軸。
- IC-02 對每個選取模型分別驗證 embedding cache、shape 和點索引一致；切換已產生的投影及 2D／3D 後，類別／split 與每個點的原始影像對應不變。UMAP 跨 run 固定參考座標另由 IC-03 驗收。
- 類別／split 和每個點的原始影像對應與 Streamlit 基準一致；點選結果顯示相同檔案，不得僅以圖表存在或 API 200 判定。
- 同一輸入重跑使用 LV manifest SHA-256 作為 embedding cache key，結果順序和語義一致；來源圖片與 fixture hash 不變，manifest／embedding 寫入只發生在隔離 `LV_CACHE_DIR`。
- IC-03：開啟固定 UMAP 參考系後，首次執行建立 reference；相同資料重跑時舊點座標不變；加入新圖片時舊點仍不變且新點由既有 reducer 轉換；手動重建會在下一次執行重 fit。驗證 reference 僅寫入隔離 `LV_CACHE_DIR`，來源 fixture 不含 `umap_ref.pkl`。目前 deterministic feature 與可見 React E2E 證明 UI/API/UMAP reducer 流程，尚未證明真實模型的數值 parity 或 NativeApp host 流程。
- IC-04：在可見 3D chart 檢查 KMeans center labels；選定非空 block 後，2D box/lasso 把命中的 record indices 加入既有多選；確認切 2D／3D 後選取數和 record 身分一致，Clear selection 才清除；block API 不回原始向量或 cache path。`test_case_lv_web_01_ic_04_kmeans_blocks_and_embedding_drilldown` 與 `test_lv_web_01_ic_04_kmeans_block_selection_round_trips_across_dimensions` 使用 deterministic features，證明 React/API 接線與既有 PCA/KMeans 行為，尚未證明真實模型或 NativeApp parity。
- OD-01：固定 YOLO fixture 中，所選類別檔優先於資料集 `classes.txt`；清除所選檔後，自動類別優先於手動輸入；無自動或手動類別時顯示明確錯誤。以兩個模型驗證各 model/method view 的 record order、label/split 和點選讀回一致，並核對來源 image、labels、classes bytes 不變。`test_case_lv_web_01_od_01_detector_whole_image_class_sources_and_models` 和可見 `test_lv_web_01_od_01_detector_whole_image_and_class_priority` 使用 deterministic features；真實模型數值 parity 與 NativeApp OS picker/worker lifecycle 仍待驗。
- OD-02：固定 YOLO fixture 中逐筆比對 image-then-box record 順序、class ID/name、bbox、score 與 split；驗證指定 padding／解析度／head 進入 embedder，預覽 crop 與相同 bbox policy 一致。class filter 與 2D／3D 切換後保持候選 index 和選取 identity；核對 source image、labels、classes bytes 不變。`test_case_lv_web_01_od_02_object_embeddings_policy_order_and_crop_readback` 與可見 `test_lv_web_01_od_02_visible_object_embeddings_policy_filters_and_3d` 使用 deterministic features；真實模型數值 parity 與 NativeApp OS picker/worker lifecycle 仍待驗。
- VIEW-01：Shift-click 後 selected set 與 thumbnail-wall record index 一致；縮圖順序沿用 Streamlit spatial order；點縮圖後前／後導覽仍依同一 index 序列。YOLO overlay 切換須更新影像且跨圖保留設定；來源 bytes 不變。切換 Image Classifier／Object Detector 再切回時不重現舊 run、selection 或 viewer。API overlay bytes test 與可見 E2E 使用 deterministic features。
- SEARCH-01：從選取影像啟動查詢，結果不含 query 且 cosine distance 升冪；使用候選續查後可回到前一個 query；切換已在 run 中計算的 model 後，保留 query chain 並按新模型重新排序。CSV 第一筆資料是目前 query，後續資料和 React 結果順序一致；圖片來源 bytes 不變。API 與可見 E2E 使用 deterministic vectors，尚未證明真實模型排序或 NativeApp parity。
- SEARCH-02：Chinese-CLIP 文字查詢回傳與固定文字向量一致的前 K 張，繁體 query 由文字塔正規化；文字結果可以啟動 SEARCH-01 image chain，CSV 只包含候選。phash／embedding duplicate pair 須依 distance 和 index 穩定排序；cross-split 只回不同 split 的候選對；兩側可回讀，來源不被改寫。deterministic API/UI 通過；本機無 Chinese-CLIP 權重，真實模型 parity 與 NativeApp grants 待驗；cart 加入流程由 CART-01 覆蓋，跨工具 handoff 仍待驗。
- RANK-01：以 legacy `compute_outlier_scores` 與 `compute_label_disagreement` 的同一 k 和 self-exclusion 規則對照數值；驗證無選取時 Top-50 預設／分歧切換、選取集 split filter、檔名與 12-band spatial 順序，並確認排名卡與上一張／下一張共用顯示順序。確認排名不改變 selected set、來源 bytes 不變，分數提示只供人工複查。deterministic API 與可見 UI E2E 通過；真模型數值 parity、NativeApp grants／ARM64 待驗。
- CART-01：從手動多選、排名候選、文字搜尋、相似圖與重複圖加入 app-owned cart；同一路徑重複加入保留首次 metadata，物件列用路徑加 object index 識別。編輯可見列理由、來源篩選、逐筆移除與確認清空；CSV 欄位沿用既有 index／filename／path／label／split／sha256／source／score／reason，JSON 保留同一身分，ZIP 含原始影像及 manifest。cart 存在 `LV_CACHE_DIR/curation_cart.json`，worker 重建後讀回；畫面 GET 不暴露本機路徑，來源檔案只讀。API 與可見 React E2E 分別由 `test_case_lv_web_01_cart_01_persists_deduplicates_and_exports`、`test_case_lv_web_01_cart_01_keeps_object_identity_for_boxes_on_one_image` 及 `test_lv_web_01_cart_01_visible_selection_reason_and_exports` 驗收；NativeApp cache/grant 與跨工具 handoff 仍待驗。
- 模型缺失、空資料夾和取消須呈現可恢復狀態；不回報成功或改用假的 backend。worker 關閉與 NativeApp lifecycle 由 AC6 驗收。
- 新 worker PID 讀取相同 run 若有持久結果時必須一致；尚未列入保存語義的暫存 run 不宣稱跨重啟持久。

先以小型 deterministic core/API case 驗證契約，再以真實 NativeApp host bridge、可見 React UI 和真實 DINOv2 完成 case。mock 僅供錯誤／邊界單元測試，不能作遷移通過證據。

## 6. 未涵蓋

Compare 物件級旅程、資料集變更與撤銷、其它工具各自的散點／縮圖／handoff 交互、另外八個頂層工具及巢狀工作流，均留在後續 case；本設計不縮小全產品 PRD 範圍。
