# 07 — whole_image_source 技術設計

對應 PRD:[2_PO_PRD/anomaly_whole_image_prd.md](../2_PO_PRD/anomaly_whole_image_prd.md) · 里程碑 M8 · Tier B

## 1. 目的(Purpose）

把每張可讀影像展開成一筆「全幅物件」記錄,並讓 `run_pipeline` 依「物件來源」選用它,使**無 YOLO 標籤**的圖片資料夾也能跑完整套異常偵測。

## 2. I/O 契約

### 2.1 新函式 `discover_whole_images`(放 `scripts/interaction.py`,與 `discover_yolo_objects` 同檔)

```python
def discover_whole_images(
    image_paths: Sequence[Path],
    *,
    label: str = "image",
) -> list[dict]:
    """每張影像 → 一筆「全幅物件」記錄(不讀 label 檔、不要求 labels/)。
    記錄欄位與 discover_yolo_objects 完全同形,故下游一律不變。"""
```

- 對 `image_paths` 內**每一個** path,**依序**產出恰好一筆記錄:
  ```python
  {"image_path": Path(p), "label": label, "class_id": 0,
   "bbox": (0.5, 0.5, 1.0, 1.0), "obj_index": 0, "score": 1.0}
  ```
- **欄位集合**與 `discover_yolo_objects` 一致:`{"image_path","label","class_id","bbox","obj_index","score"}`。
- **純函式**:不讀檔、不判斷可讀性(可讀性由上游 `run_pipeline` 的 `partition_readable` 處理,與 yolo 路徑相同)。輸入 N 個路徑 → 輸出 N 筆,順序保留。
- `bbox` 用 YOLO 正規化全幅 `(cx,cy,w,h)=(0.5,0.5,1.0,1.0)`;經 `crop_bbox`/`bbox_to_pixels`(clamped)= **整張影像**(見 §4 AC10)。
- `score=1.0`:整張影像無偵測器,合成信賴度固定 1.0(對齊 yolo 記錄的偵測信賴度欄位)。
- `class_id=0` 且 `label` 單一 → **永遠只有 1 類** → `head_unlock_state` 回 `lt_2_classes` → **不解鎖分類頭**(與既有語義鎖一致,無需特例,見 §3.4)。

### 2.2 `run_pipeline` 新增參數(`scripts/anomaly_tool.py`)

```python
def run_pipeline(image_paths, class_names, *, mode="two_stage", score_mode="patch",
                 ..., object_source: str = "yolo") -> dict:   # 新增 object_source
```

- `object_source` 合法值:`"yolo"`(預設,**現況行為**)、`"whole_image"`。
- routing(在既有 `partition_readable` 之後、取代現行第 60 行 `meta = discover_yolo_objects(...)`):
  ```python
  if object_source == "whole_image":
      meta = discover_whole_images(list(image_paths))
  elif object_source == "yolo":
      meta = discover_yolo_objects(list(image_paths), class_names)
  else:
      raise ValueError("object_source must be 'yolo' or 'whole_image'")
  ```
- 其餘 `run_pipeline` 內容**完全不動**(N、空集短路、embedding、分群、normal_set、bank、評分、ranking、cluster coords、skipped 全沿用)。

### 2.3 GUI 接線(`scripts/app.py` `_anomaly_ui`,Tier B/E2E)

- **物件來源切換**(① 分頁本體,放在「(1) 選訓練/參考資料夾」之後、label 語義之前):
  `st.radio` 或 `segmented_control`,key `anomaly_object_source`,選項 format:
  - `yolo` → 「YOLO 物件(讀 labels/ 逐物件)」
  - `whole_image` → 「整張影像(無需 labels/,每張一個對象)」
- `_anomaly_build_model`:把 `object_source=st.session_state["anomaly_object_source"]` 傳入 `run_pipeline`,並寫進 `model["meta"]["object_source"]`(與 model/target_res/score_mode 一起被鎖)。
- `_anomaly_apply_model`:**從 `model.meta["object_source"]` 讀**(鎖定值,非側欄現值)傳入 `run_pipeline`。
- **模式標示**:① 建模成功列與 ② 鎖定 caption 各加目前物件來源文字(「整張影像」/「YOLO 物件」),使用者一眼可辨。
- **整張影像時不要求 labels/**:`whole_image` 永遠有記錄(只要資料夾有可讀影像);既有「找不到 YOLO 物件」守衛只在 `yolo` 模式才語意成立(真空資料夾仍會因 records 空而正確報「資料夾無可讀影像/物件」)。
- **Should — fallback 提示**:`yolo` 模式跑出 `records==[]`(沒讀到任何框)時,除原錯誤訊息外,顯示一鍵「改用整張影像重跑」按鈕(把 `anomaly_object_source` 設 `whole_image` 並重觸發建模)。Could 之外、Won't 之內不做自動切。

## 3. 資料流(Data Flow)

1. `run_pipeline` 收 `image_paths` → `partition_readable` → `readable`, `skipped`。
2. 依 `object_source`:`whole_image` → `discover_whole_images(readable)`(N=len(readable));`yolo` → `discover_yolo_objects(readable, class_names)`(N=框數)。
3. 之後**與現況逐字相同**:`_object_embeddings(meta)` 對每筆 `crop_bbox(img, *bbox, pad)` → 全幅 bbox = 整張影像 → DINOv2 embedding;`cluster_objects`;`normal_set`;Normal Bank;patch/object 評分;`ranking`;`cluster` 2D coords;回傳 dict(含 `skipped`)。
4. GUI:① 建模把 `object_source` 凍進 `model.meta`;② 讀回鎖定值套用;散點每點 = 一筆記錄(whole_image 時 = 一張圖)。

## 4. 邊界條件與錯誤處理

| 情況 | 預期行為 |
|---|---|
| `image_paths=[]` | `discover_whole_images` 回 `[]`;`run_pipeline` 命中既有 N==0 短路、回空結果 dict(不崩潰) |
| 單張影像 | 回 1 筆記錄 |
| 含不可讀影像 | 由 `partition_readable` 前置剔除 → whole_image 的 N == 可讀影像數,`skipped` 列出被剔者 |
| 無 labels/(whole_image) | 照常 N==影像數(`discover_whole_images` 不讀 label 檔) |
| 無 labels/(yolo,現況) | `discover_yolo_objects` 回 `[]` → records 空(行為不變) |
| `object_source` 非法值 | `run_pipeline` 拋 `ValueError`(明確契約,不靜默退化) |
| `label` 參數自訂 | 所有記錄 `label` = 該值;預設 `"image"` |

## 5. Acceptance Criteria(釘死數值 / 行為;給 /pm)

> 單元 AC 用假路徑 / stub `embed_fn`,不依賴真實模型或磁碟影像;GUI AC 走真實 E2E。

**單元(`discover_whole_images` + `run_pipeline` routing)**

- **AC1（記錄內容釘死)**:`discover_whole_images([Path("a.jpg"), Path("b.png"), Path("c.jpg")])[1] ==
  {"image_path": Path("b.png"), "label": "image", "class_id": 0, "bbox": (0.5,0.5,1.0,1.0), "obj_index": 0, "score": 1.0}`。
- **AC2(欄位同形)**:對每筆記錄 `set(rec.keys()) == {"image_path","label","class_id","bbox","obj_index","score"}`(與 `discover_yolo_objects` 輸出的鍵集合相等)。
- **AC3(數量+順序)**:輸入 5 個相異路徑 → 輸出長度 5,且 `[r["image_path"] for r in out] == [Path(p) for p in inputs]`。
- **AC4(label 參數)**:`discover_whole_images([p], label="frame")[0]["label"] == "frame"`;不傳時 `== "image"`。
- **AC5(空輸入)**:`discover_whole_images([]) == []`。
- **AC6(run_pipeline 路由 whole_image,無 labels 也跑得出)**:對 4 張可讀影像、**完全無 label 檔**,
  `len(run_pipeline(paths, None, object_source="whole_image", score_mode="object", embed_fn=stub, ...)["records"]) == 4`;
  且同資料夾 `object_source="yolo"` 時 `["records"] == []`(對照組,證明差異來自路由)。
- **AC7(預設=yolo / 回歸)**:對含已知 K 個框的 fixture,`run_pipeline(paths, names, embed_fn=stub)`(不傳 object_source)
  的 `len(records) == K`,且與顯式 `object_source="yolo"` 結果**逐筆相等**(records 等值)。
- **AC8(非法值)**:`run_pipeline(paths, names, object_source="bogus", embed_fn=stub)` 拋 `ValueError`。
- **AC10(全幅 bbox = 整張影像)**:對 64×48 測試影像 `img`,`crop_bbox(img, 0.5, 0.5, 1.0, 1.0, pad=0.1).size == (64, 48)`
  且像素與 `img` 相同(`list(crop.getdata()) == list(img.getdata())`)—— 證明 whole_image 記錄確實 embed 整張圖(防 silent-wrong)。

**GUI / 整合(真實 E2E,`@pytest.mark.e2e`)**

- **AC-G1(無 labels 能跑)**:① 物件來源選「整張影像」、訓練資料夾指向**只有影像、無 `labels/`** 的資料夾 → 建模**不出現「找不到 YOLO 物件」**;② 套用同型目標 → 散點圖**渲染出資料點**,點數 == 影像數。
- **AC-G2(模式標示)**:整張影像模式下,① 成功列或 ② 鎖定 caption **可見「整張影像」字樣**(真實文字存在,非僅元素存在)。
- **AC-G3(YOLO 回歸)**:物件來源選「YOLO 物件」+ 有 labels 的資料夾 → 建模/套用行為與現況一致(物件數 == 框數、能套用)。
- **AC-G4(鎖進 model.meta)**:以 whole_image 建模後,即使側欄/UI 之後改動,② 仍以 whole_image 套用(`model.meta["object_source"]=="whole_image"`)。

> Should 的 fallback 提示(AC-G5:yolo 模式 0 物件 → 顯示「改用整張影像」按鈕)列為 Should;appetite 緊時可進候選,PM 標記。

## 6. 給 /pm 的提示

- 單元測試 `tests/test_whole_image_source.py`:AC1–AC8、AC10。`run_pipeline` 用 stub `embed_fn`(回確定性向量,如 `np.eye`/雜湊),`score_mode="object"` 走最短路徑避免 patch/HDBSCAN 噪音。
- AC7 的 K 框 fixture 可重用既有 anomaly 測試 fixture(若有);否則造一個 2 圖各 1 框 = K=2。
- E2E 重用既有 `tests/e2e/conftest.py` harness;需要「有影像無 labels」與「有 labels」兩個臨時資料夾 fixture。
- 契約鎖點:`discover_whole_images` 記錄**欄位集合**與 `discover_yolo_objects` 相等(AC2)是防下游漂移的關鍵,務必保留。
