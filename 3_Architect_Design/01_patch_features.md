# 設計 01:patch_features(Tier B)

## Purpose
把每個 YOLO 物件裁切框轉成 DINOv2 **patch-grid** 特徵(含 grid 形狀)並快取到 `.lv_cache/`。

## 背景 / 復用(務必)
- 物件來源:`interaction.discover_yolo_objects(image_paths, class_names)` → `[{image_path,label,class_id,bbox=(cx,cy,w,h),obj_index,score}]`。
- 裁切:`interaction.crop_bbox(img, cx,cy,w,h, pad_px=...)`;pad 用 `object_eval._adaptive_pad_px`。
- 前處理:`models.ImagePreprocessor(size=target_res, keep_aspect=True)` → 長邊縮到 target_res、兩邊吸附成 14 的倍數。
- patch tokens 來源(已存在):`Dinov2Extractor` 內 `model.forward_features(t)["x_norm_patchtokens"]` → `(1, N, D=384)`。
- 快取目錄:`object_eval.dataset_cache_dir(folder, kind)` → `.lv_cache/<name>_<hash>/<kind>`(可 `LV_CACHE_DIR` 覆寫)。**絕不寫使用者資料集**([[no-dataset-writes]])。

## I/O 契約
```
PatchFeat = {
  "feats": np.ndarray,   # (P, 384) float32，已 per-token L2 正規化(每列範數=1)
  "grid":  tuple[int,int]  # (gh, gw)，且 gh*gw == P
}

# 純包裝(可注入 extractor → PM 不需真模型即可測形狀/正規化/快取)
def extract_patch_grid(
    crop: PIL.Image,
    extractor: Callable[[PIL.Image], tuple[np.ndarray, tuple[int,int]]] | None = None,
    *, target_res: int = 224,
) -> PatchFeat
#   extractor(preprocessed_crop) -> (patchtokens (N,384), (gh,gw))
#   預設用真實 DINOv2(forward_features);grid 由前處理後尺寸算:gh=H//14, gw=W//14

def embed_objects_patch(
    meta: list[dict], model: str = "dinov2_vits14", *,
    target_res: int = 224, pad: float = 0.12,
    cache_dir: Path | None = None,
    extractor=None, progress: Callable[[int,int],None] | None = None,
) -> list[PatchFeat]
#   逐物件回傳 PatchFeat(順序同 meta)。cache_dir 給定時逐物件存
#   <cache_dir>/<stem>__<obj_index>.npz(feats float16 + grid),命中即重用、不載 torch。
```

## Data Flow
1. 對每個 `meta[i]`:開圖(快取同一張圖的解碼)→ `crop_bbox`(pad_px adaptive)→ `ImagePreprocessor.preprocess`(keep_aspect, mult-14)。
2. `extractor(crop)` → patchtokens `(N,384)`,grid `(H//14, W//14)`,且 `N==gh*gw`。
3. per-token L2 正規化(`x/||x||`,eps 1e-12),存 float32 回傳。
4. cache_dir:命中載 float16→float32;未命中算完存 float16(`.part.npz`→atomic replace,沿用 `embed_objects` 寫法),每 200 物件 checkpoint。

## 邊界與錯誤處理
- 壞圖 / 解碼失敗:該物件回 `feats` 形狀 `(1,384)` 全 0、`grid=(1,1)`(不中斷整批,沿用 `embed_objects` 對壞圖回零向量的慣例)。
- 退化 bbox:`bbox_to_pixels` 已保證 ≥1px;前處理保證每邊 ≥14px → grid 至少 `(1,1)`。
- cache_dir 必須在 `.lv_cache` 底下;函式不得在資料集目錄建立任何檔。

## Acceptance Criteria
- **AC1**:餵一個會被前處理成 224×224 的方形 crop(注入 extractor 回 `(256,384)` tokens、grid `(16,16)`)→ 回傳 `feats.shape==(256,384)`、`grid==(16,16)`。
- **AC2**:`feats` 每一列 L2 範數 == 1.0(atol 1e-4)。
- **AC3**:`grid[0]*grid[1] == feats.shape[0]` 恆成立(再測一個 28×42→grid `(2,3)`、P==6 的非方形案例)。
- **AC4**(快取往返):同一 meta 跑兩次,第二次命中快取,回傳與第一次 atol 1e-3 相等(float16 容差),且第二次**不呼叫** extractor(用計數 fake 驗證呼叫數==0)。
- **AC5**(零汙染):`embed_objects_patch(..., cache_dir=<.lv_cache 下>)` 跑完後,資料集目錄(meta 的 image_path 上層)內**沒有新增任何檔案**;所有新檔都在 cache_dir 底下。
- **AC6**:壞圖路徑 → 該物件 `grid==(1,1)`、`feats.shape==(1,384)`、全 0,且不丟例外。
