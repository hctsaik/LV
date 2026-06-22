# 設計 06:anomaly_tool(Tier B,GUI 整合 + 真實 E2E)

## Purpose
新增第 8 個工具頁籤,串起「載入 YOLO 資料夾 → 選流程/分數模式 → 跑 01–05 → 散點圖框選/購物車 → 排序清單/熱力圖 → 匯出原圖」。

## 復用(務必,不另造輪子)
- 載入/物件:`object_eval.list_images`/`classes_for`/`dataset_fingerprint`、`interaction.discover_yolo_objects`。
- 物件級 embedding:`object_eval.embed_objects`(供分群與「物件級」分數模式)。
- 散點圖 + 框選 + 購物車 + 匯出:`app.py` 既有 scatter / `interaction.selection_points_to_indices` / `zip_selected_images`。
- 快取目錄:`object_eval.dataset_cache_dir`(`.lv_cache`)。
- 抽樣:`interaction.farthest_point_sampling`。

## 兩層拆法(關鍵:核心無 streamlit,可單元測試)
- **`scripts/anomaly_tool.py`**:純協調核心 `run_pipeline(...)`(無 streamlit,可注入 fake 模型/extractor)。
- GUI render 函式(在 `anomaly_tool.py` 提供、由 `app.py` 呼叫)只做 Streamlit widget 與 session_state,邏輯一律委派核心。沿用 `interaction.py` 的「框架無關核心」慣例。

## I/O 契約(純核心)
```
ObjRecord = {
  "path": str,           # == image_path(原圖!匯出用,符合 [[object-export-crop-bug]])
  "image_path": str, "bbox": (cx,cy,w,h), "obj_index": int,
  "label": str, "split": str,
  "score": float, "verdict": "good"|"bad",
}

def run_pipeline(
    image_paths, class_names, *,
    mode: str = "two_stage",         # "one_stage" | "two_stage"
    score_mode: str = "patch",       # "patch" | "object"
    sample_n: int = 64,              # 2-stage bootstrap 抽樣數
    confirmed: dict[int,str] | None = None,  # 使用者對候選的好/壞確認(全域物件索引)
    contamination: float = 0.05,
    cache_dir: Path | None = None,
    model: str = "dinov2_vits14",
    extractor=None, embed_fn=None,   # 注入點:測試用 fake,免真模型
    progress=None,
) -> dict
#   回 {records: list[ObjRecord], ranking: list[int], threshold: float,
#       auroc: float|None, n_bad: int, candidates: list[int], cluster: ClusterResult,
#       coords2d: np.ndarray|None}  # coords2d 供散點圖(可延後,GUI 端也可自算)
```

## Data Flow
1. `discover_yolo_objects` → 全體物件 meta(N 個)。空 → 友善訊息(見錯誤路徑)。
2. 物件級 embedding(`embed_objects`,快取)→ `bootstrap_cluster.cluster_objects`。
3. **正常集(memory bank 來源)優先序 —— 關鍵,防 false-green**:
   - **(a) 有「確認為 good」時**:正常集 = **僅確認為 good 的物件**(乾淨參考,few-shot AnomalyDINO)。
     **不**併入未經確認的正常密群——因為**細微瑕疵會藏在正常密群裡**(物件級幾乎等於正常),
     一旦併入,其壞 patch 會污染 bank、使該瑕疵在評分時找到壞鄰居而距離≈0 → 漏抓。
   - **(b) 完全沒有確認(純 1-stage)時**:正常集 = 全體 `normal_mask`(大/密群)。
     **誠實上限**:此模式下,藏在正常密群、未被分群標為候選的細微瑕疵會污染 bank 而被漏抓;
     1-stage 適合「瑕疵夠明顯、物件級就會被分群挑成候選」的情況。要抓細微瑕疵請走 (a)/2-stage 確認少數正常。
   - **2-stage** 流程:`farthest_point_sampling` 抽 `sample_n` → 分群 → `candidates` 交 GUI 確認;
     使用者點頭的正常進「確認 good」(走 (a)),點頭的壞排除。
4. 正常集 → `patch_features.embed_objects_patch`(僅正常集物件)→ 攤平成正常 patch 矩陣 → `anomaly_score.build_memory_bank`。
   全體物件(含正常集)仍各自算分數;正常集自身分數會偏低(它就是參考),屬預期。
5. 全體物件 → patch 特徵 → `score_object` 得 `score` 與 patch 分數圖(分數圖供 heatmap,按需算/快取)。`score_mode=="object"` 時改用 `interaction.compute_outlier_scores(物件 emb, 正常物件 emb, k)`。
6. `anomaly_classify.classify(scores, confirmed, contamination)` → `verdict`/`ranking`/`threshold`/`auroc`。
7. 組 `records`(每物件,**path=原圖**),回傳。

## GUI 行為(委派核心)
- 載入路徑沿用既有資料夾輸入元件;`radio` 選 1/2-stage;`radio`/`toggle` 選 物件級/patch 級(預設 patch);`slider` 設 sample_n。
- 散點圖:位置用物件級 embedding 投影(沿用既有 PCA/UMAP scatter),**顏色用 anomaly score**;每點 `customdata=[全域物件索引]`。
- 框選 → `selection_points_to_indices` → 「加入購物車」沿用既有 cart;2-stage 的「確認候選」也走框選/點選 → 寫 `confirmed`。
- 排序清單:依 `ranking` 顯示;點一列 →(Should)`anomaly_heatmap.render_heatmap` 疊圖。
- 匯出:`zip_selected_images(records, indices)` → **原圖 + manifest**(非裁切圖)。

## 邊界與錯誤處理(GUI/整合 false-green 重災區,強制 AC)
- 資料夾無 `labels/` 或無物件 → 顯示**中文友善訊息**(「找不到 YOLO 標註(images/+labels/)…」),不是原始 traceback。
- 模型權重缺失 → 沿用 `model_manifest.explain` 的中文指引。
- 正常集為空(分群把全部判成候選)→ 友善提示「請先確認一些正常樣本」,不崩。
- 一律不寫使用者資料集(快取進 `.lv_cache`)。

## Acceptance Criteria
### 單元(核心 `run_pipeline`,注入 fake、無瀏覽器)
- **AC1**:餵合成 meta + fake extractor/embed_fn → `records` 每筆含 `path/image_path/bbox/obj_index/score/verdict` 全鍵;`path==image_path`(原圖,守 [[object-export-crop-bug]])。
- **AC2**(細微瑕疵 + 乾淨 bank,真正的價值):fake extractor 讓 defect 物件只有「1 個 B patch」(物件級幾乎等於正常);`mode="two_stage"`、`confirmed` 把真實正常標 good、`score_mode="patch"` → bank 只由 confirmed-good 建(乾淨)→ K 個 defect **全部**落在 `ranking` 前 K 名。
- **AC3**(可分性 / 餵 G1):AC2 設定下 `run_pipeline(...)["auroc"]`(以注入 ground-truth 算)`>= 0.85`。
- **AC4**(bank 污染會漏抓 —— 反向驗證 step 3 的設計理由):同 AC2 資料,但把一個 defect 物件也放進 `confirmed` good(污染 bank)→ 該 defect 的 `score` **顯著下降**(< 其在乾淨 bank 時分數的一半),示警「細微瑕疵混進正常集就抓不到」。
- **AC5**:`score_mode="object"` 路徑可跑通且回相同 schema;`mode="one_stage"`(無 confirmed)對「明顯缺陷」(defect 物件多數 patch=B)仍能把 defect 排前段。
- **AC6**(零汙染):跑完資料集目錄無新增檔。

### 真實 E2E(`tests/e2e/test_anomaly_tool_e2e.py`,`@pytest.mark.e2e`,重用既有 harness;不進 PG 修綠迴圈)
> 需 PM 準備合成 **YOLO** 夾具(images/+labels/,正常多數 + 注入缺陷的少數,缺陷位置已知)。
- **E2E-AC1**(真行為,非 element 存在):載入該夾具、跑 1-stage patch → 排序清單**前段**出現被注入缺陷的物件(K 個缺陷全在前 2K 名);讀清單實際文字/順序斷言,不只查元素存在。
- **E2E-AC2**:散點圖渲染後,框選離群區 → 選取數 ≥ 缺陷數 → 「加入購物車」→ 購物車計數增加。
- **E2E-AC3**(匯出原圖):匯出選取 → 下載的 zip 內影像為**原圖尺寸**(等於來源圖,非裁切小圖),且含 manifest.csv。
- **E2E-AC4**(Should):點一個被判 bad 的物件 → 顯示熱力圖覆蓋圖;峰值落在注入缺陷的 bbox 區域內。
- **E2E-AC5**:切換 物件/patch toggle 與 1/2-stage radio 後重跑不報錯(頁面回到可用、無 Python exception 溢到 UI)。
- **E2E-AC6**(錯誤路徑):指向沒有 `labels/` 的資料夾 → 畫面顯示中文指引訊息,無原始 traceback。
- **版面**:工具頁籤在目標視窗寬度下無非預期水平捲動 / 不破版。
