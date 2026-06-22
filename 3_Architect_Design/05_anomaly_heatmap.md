# 設計 05:anomaly_heatmap(Tier A,純函式)

## Purpose
把 patch 分數圖上採樣疊回物件裁切圖,產生「哪裡怪」的熱區覆蓋影像。

## 復用
- colormap 可用 `matplotlib.cm`(專案已有 matplotlib,見 tests/conftest 的 Agg 後端);上採樣用 PIL `resize(BILINEAR)` 或 numpy。不引進新依賴。

## I/O 契約
```
def render_heatmap(
    patch_scores: np.ndarray,          # (gh, gw) float,任意實數範圍
    crop: PIL.Image | np.ndarray,      # 物件裁切圖(RGB 或灰階皆可)
    *, alpha: float = 0.5,
    vmin: float | None = None, vmax: float | None = None,
    cmap: str = "jet",
) -> PIL.Image                          # RGB,大小 == crop 大小
```

## Data Flow
1. `crop` → RGB PIL,記其 `(W, H)`。
2. 正規化:`vmin/vmax` 預設 = `patch_scores.min()/max()`;`norm = clip((s-vmin)/(vmax-vmin), 0, 1)`;若 `vmax==vmin` → `norm` 全 0.5(避免除零)。
3. 上採樣 `norm` 到 `(W,H)`(BILINEAR)→ 套 `cmap` 得 `heat`(RGB)。
4. **per-pixel 淡入疊加**(熱度隨分數淡入,低分處幾乎保留原圖):
   `w = alpha * norm_up`(逐像素權重,(H,W));`out = (1-w)*crop + w*heat`,clip→uint8→PIL RGB。
   (不用固定 alpha 全圖混合:那樣低分區也被色階蓋掉,使「峰值=與原圖差最大」不成立。)

## 邊界與錯誤處理
- 常數分數圖 → norm 全 0.5(無 NaN)。
- 灰階 crop → 轉 RGB。
- `alpha` 夾到 [0,1]。
- `patch_scores` 退化 `(1,1)` → 上採樣為整片均勻熱度,合法。

## Acceptance Criteria(釘死)
- **AC1**:輸出 `.size == crop.size`、`.mode=="RGB"`。
- **AC2**(峰值定位):`patch_scores` 形狀 (gh,gw)=(4,8),僅 cell (r,c)=(2,5) 設高值、其餘 0;crop 大小 (W,H)=(160,80)。對輸出取「熱度通道」(與 crop 差最大的像素)→ 其 argmax 像素座標 `(px,py)` 落在 cell (2,5) 的空間範圍內:`py ∈ [2*80/4,3*80/4)=[40,60)`、`px ∈ [5*160/8,6*160/8)=[100,120)`。
- **AC3**(常數圖無 NaN):`patch_scores` 全相同值 → 輸出無 NaN/inf,且每像素的「crop↔輸出」混合權重一致(疊加均勻)。
- **AC4**(alpha=0 + 單調淡入):`alpha=0.0` → 輸出逐像素 == crop(atol 1);且在 alpha>0 時,
  **分數高的區域偏離原圖比分數低的區域多**(單一高分 cell 的平均 |out−crop| > 其餘區域)——熱度隨分數淡入。
- **AC5**(決定性):同輸入兩次 → 位元組相同。
