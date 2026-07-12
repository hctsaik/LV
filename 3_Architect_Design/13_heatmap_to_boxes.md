# 13 — heatmap_to_boxes 技術設計(anomaly 熱力圖 → YOLO 預標框)

對應 PRD:[2_PO_PRD/al_loop_infrastructure_prd.md](../2_PO_PRD/al_loop_infrastructure_prd.md) · 里程碑 M10 · Tier A(核心)+ B(按鈕)

## 1. 目的(Purpose)

回答「很強的 embedding / anomaly DINO 能不能像 YOLO 一樣做預標?」——**能,但要經過定位橋接**:
DINOv2 embedding 本身只會分類/檢索,不會畫框;可是 anomaly DINO 的 **patch 分數圖有空間資訊**。
本模組把 patch score map → 閾值 → 連通區域 → **YOLO 6 欄預標框(class cx cy w h conf)**,
輸出成與模型預測完全同形的「預測資料夾」,直接餵 10 號(gt_pred_diff / 預標 seed)管線,零特例。

誠實界線(寫進 UI 文案):框粒度受 patch 網格(14px)限制、單一類別(`defect`;分類頭可另行
per-region 給類名,列 Could)、只適用瑕疵型資料;一般多類物件偵測請用真的偵測器。

## 2. I/O 契約

### 2.1 新模組 `scripts/heatmap_to_boxes.py`(依賴 numpy + scipy.ndimage,皆已在環境)

```python
def boxes_from_scoremap(
    pmap: "np.ndarray",              # HxW float,patch 分數圖(不要求正規化)
    *,
    thr: float | None = None,        # 絕對閾值;None → 用 quantile
    quantile: float = 0.98,          # thr=None 時:閾值 = np.quantile(pmap, quantile)
    min_area_frac: float = 0.001,    # 區域面積 < H*W*frac → 濾掉(雜點)
) -> list[tuple[float, float, float, float, float]]
    # 回 [(cx, cy, w, h, conf)],全部 0-1 正規化(相對 pmap 尺寸);conf = 該區域 pmap 最大值。
    # 步驟:mask = pmap > 閾值 → scipy.ndimage.label(8 連通) → 每個 component 的外接框
    # → 面積過濾 → 依 conf 降冪。全零/全低圖回 []。純函式、決定論。

def write_pred_labels(
    items: list[tuple[Path, "np.ndarray"]],   # [(原圖路徑, 該圖 pmap)]
    out_root: Path,                            # 目的根;寫 out_root/labels/<stem>.txt
    *,
    class_id: int = 0,
    thr: float | None = None, quantile: float = 0.98, min_area_frac: float = 0.001,
) -> dict
    # 對每張圖:boxes_from_scoremap → 寫「class cx cy w h conf」一行一框(6 欄,%.6g);
    # 無框圖寫空檔(語義=「有預測來源、此圖無框」,對齊 YOLO 慣例)。
    # 同時寫 out_root/classes.txt(單行 "defect")。
    # 回 {"n_images", "n_boxes", "out_root"}。out_root 一律在 .lv_cache 下(呼叫端保證,見 2.2)。
```

> **⚠ 反向閘門 /pg → /architect(2026-07-12):裁切座標 → 原圖座標的合成缺口(加法式補設計)。**
>
> 初版把 `write_pred_labels` 定成「一圖一 pmap」。但物件模式下 pmap 是**單一物件裁切圖**的分數圖
> (`embed_objects_patch` 以 `pad=0.12` 的 `_adaptive_pad_px` 裁切後才抽特徵),於是:
> ① 框落在**裁切座標系**,直接寫出去會與原圖 GT 對不上(下游 10 號比 IoU 必錯);
> ② 一張圖有**多個物件**時是多張 pmap,舊簽名容不下。整張影像模式(M8)因裁切==全圖,
> 兩個問題都退化不見 —— 這正是初版沒踩到的原因。
>
> **設計裁決(純加法,既有 AC1–AC7 不動)**:新增兩個純函式,`write_pred_labels` 改為薄包裝:
> ```python
> def box_to_image_space(box, region, iw, ih) -> tuple
>     # 裁切座標框 →原圖正規化。region=(x0,y0,x1,y1) 該 pmap 對應的裁切矩形(像素)。
>     # region == 全圖時退化為 identity(整張影像模式)。
>
> def boxes_by_image(objects, *, thr=None, quantile=0.98, min_area_frac=0.001) -> dict[str, list]
>     # objects 每筆 = {"image_path", "pmap", "image_size": (iw,ih), "region": (x0,y0,x1,y1)|None}
>     # 同一張圖的多物件框合併到同一份預標;每個物件的閾值**各自**依其 pmap 分佈算
>     # (與熱力圖所見一致),不把「物件外」當 0 分混進分位數——那會把「沒分數」誤當「正常」。
>
> def write_boxes(boxes_by_img: dict[str, list], out_root, *, class_id=0) -> dict
>     # 原圖座標框 → labels/<stem>.txt(6 欄)。無框寫空檔;冪等(先清 labels/*.txt)。
> ```
> **呼叫端契約(GUI)**:`region` 必須與抽特徵時的裁切**完全一致**(同一個 `pad=0.12`),
> 否則框會系統性偏移。對應 AC8。

### 2.2 GUI 接線(Tier B,`scripts/app.py` 瑕疵偵測 ② 套用結果區)

- 前置條件:模型 `score_mode=="patch"`(pmap 才存在;object 模式按鈕反灰 + tooltip 說明)。
- 新按鈕「⚡ 由熱力圖產生預標(YOLO 格式)」:
  - `out_root = dataset_cache_dir(目標資料夾, f"pred_from_anomaly_{model簽章}")`(**進 .lv_cache**,
    絕不寫使用者資料集);pmap 來源=②結果既有的 per-record patch 分數圖(熱力圖渲染同源)。
  - 完成後顯示:`產出 N 張 / M 框 → <out_root>`,並提供兩個一鍵:
    「填入 Visualize 的模型預測資料夾」(寫 `viz_pred_root`)與「用它當送標預標來源」。
  - 參數 popover:quantile(0.90–0.999,預設 0.98)、min_area_frac;調參即重產(冪等覆寫)。
- 整張影像(M8 whole_image)模式:pmap 為整圖 patch 圖,行為相同(框=圖內異常區)。

## 3. 資料流

瑕疵偵測 patch 評分(既有)→ pmap → boxes_from_scoremap → `.lv_cache/...pred_from_anomaly.../labels/*.txt`
→(a)10 號 scan_dataset 當「預測」比對 GT;(b)送標 seed_source="pred" 當預標;
→ 標註者修框回讀 → 09 入帳。

## 4. 邊界條件與錯誤處理

| 情況 | 預期行為 |
|---|---|
| pmap 全零 / 全同值 | quantile 閾值=該值 → mask 全 False → 回 [](寫空檔) |
| 兩個相鄰但分離的熱區 | 8 連通判定;不相鄰 → 兩框 |
| 貼邊熱區 | 外接框 clamp 在 [0,1],w/h>0 |
| 雜訊單點 | min_area_frac 濾除 |
| thr 與 quantile 同給 | thr 優先(明確契約) |
| pmap 非 2D | ValueError(不靜默壓維) |
| out_root 已存在舊檔 | 整目錄重寫(先清 labels/*.txt 再寫;冪等) |
| score_mode=="object"(無 pmap) | GUI 按鈕反灰,核心函式不會被呼叫 |

## 5. Acceptance Criteria

**單元(`tests/test_heatmap_to_boxes.py`;合成 pmap,不需模型)**

- **AC1(雙熱區釘死)**:20×20 全 0,`pmap[2:6, 2:6]=1.0`、`pmap[12:18, 10:16]=0.8`,thr=0.5 →
  恰 2 框(conf 降冪)。第一框 == `(0.2, 0.2, 0.2, 0.2, 1.0)`(rows/cols 2..5 → cx=cy=(2+6)/2/20=0.2,
  w=h=4/20=0.2);第二框 == `(0.65, 0.75, 0.3, 0.3, 0.8)`(cols 10..15 → cx=(10+16)/2/20=0.65、w=6/20=0.3;
  rows 12..17 → cy=(12+18)/2/20=0.75、h=6/20=0.3)。注意 numpy 索引是 [row, col]=[y, x]。
- **AC2(quantile 預設)**:thr=None 時閾值==np.quantile(pmap,0.98)(以單熱區 fixture 驗框存在/消失的臨界)。
- **AC3(min_area 濾雜點)**:100×100 圖上 1 個單像素熱點 + 1 個 10×10 熱區,min_area_frac=0.001 →
  只回 10×10 那框。
- **AC4(空圖)**:全零 pmap → []。
- **AC5(非 2D)**:3D 輸入拋 ValueError。
- **AC6(寫檔往返)**:write_pred_labels 兩圖(一有框一無框)→ labels/ 兩個 txt;有框者被
  `parse_yolo_boxes_conf` 讀回框數/conf 一致;無框者為空檔;classes.txt == "defect";回傳統計正確。
- **AC7(冪等覆寫)**:改參數重呼 write_pred_labels → 舊框不殘留(檔案內容只反映新參數)。
- **AC8(裁切→原圖座標合成;見 §2.1 反向閘門)**:
  ① `box_to_image_space((0.5,0.5,1.0,1.0,0.9), region=(0,0,W,H), W, H)` == `(0.5,0.5,1.0,1.0,0.9)`
     (region==全圖 → identity,整張影像模式不得被這層改動);
  ② 一個 400×200 原圖、region=(200,0,400,200)(右半)、裁切內框 `(0.5,0.5,0.5,0.5)` →
     原圖框 == `(0.75, 0.5, 0.25, 0.5)`(手算:cx=(200+0.5*200)/400=0.75、w=0.5*200/400=0.25);
  ③ `boxes_by_image` 對**同一張圖的兩個物件**(兩個 region、各有熱區)→ 該圖回**兩個框**、
     conf 降冪(多物件合併到同一份預標,不互相覆蓋)。

**GUI / 整合(真實 E2E,`tests/e2e/test_heatmap_to_boxes_e2e.py`)**

- **AC-G1(端到端定位)**:MVTec-風合成資料(良品純色、壞品貼一塊異色方塊,重用 anomaly E2E fixture 手法)
  patch 模式建模+套用 → 按「⚡ 產生預標」→ 產出的 txt 中,壞品圖**至少一框與植入瑕疵區 IoU ≥ 0.3**
  (真實行為斷言,防 false-green);良品圖 0 框或 conf 顯著較低。
- **AC-G2(餵回管線)**:按「填入 Visualize 預測資料夾」→ `viz_pred_root` 值==out_root;
  10 號的錯誤挖掘面板能以此路徑跑出佇列(至少渲染不報錯)。
- **AC-G3(object 模式反灰)**:score_mode=object 的模型 → 按鈕 disabled 且 tooltip 含「patch」。

## 6. 給 /pm 的提示

- AC1 請先手算再寫(上面已算好);連通用 `scipy.ndimage.label(mask, structure=np.ones((3,3)))`(8 連通)。
- AC-G1 的 IoU 斷言用 10 號的 `iou_cxcywh`(跨模組重用,但 13 不 import 10 —— 測試層 import 即可,
  避免產品程式碼循環)。
- 誠實界線文案(單類/patch 粒度/瑕疵場景限定)要真的出現在 UI(popover 或 caption),列一條 AC-G 附驗即可。
