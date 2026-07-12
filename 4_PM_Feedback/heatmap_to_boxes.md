# 13 heatmap_to_boxes — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/13_heatmap_to_boxes.md](../3_Architect_Design/13_heatmap_to_boxes.md) · 里程碑 M10 · Tier A(核心)+ B(按鈕)
單元測試:`tests/test_heatmap_to_boxes.py`(gate:`python verify/gate.py heatmap_to_boxes`)
E2E:`tests/e2e/test_heatmap_to_boxes_e2e.py`(`@pytest.mark.e2e`,**不進 gate**)

> 設計反向閘門紀錄 **/pg → /architect(2026-07-12)**:初版契約假設「一圖一 pmap」。但物件模式下
> pmap 是**單一物件裁切圖**的分數圖(`embed_objects_patch` 以 `pad=0.12` 裁切後才抽特徵),因此
> ① 框落在裁切座標系(直接寫出去會與原圖 GT 對不上,下游 10 號比 IoU 必錯)、② 一圖多物件時
> 舊簽名容不下。整張影像模式(裁切==全圖)兩問題都退化不見,故初版沒踩到。
> architect 裁決:**純加法**新增 `box_to_image_space` / `boxes_by_image` / `write_boxes`,
> `write_pred_labels` 改為薄包裝(AC1–AC7 全部不動)。新增 **AC8**。

## 單元 AC ↔ 測試

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | 雙熱區釘死:2 框、conf 降冪、`(0.2,0.2,0.2,0.2,1.0)` / `(0.65,0.75,0.3,0.3,0.8)`(鎖 [row,col]=[y,x]) | `test_ac1_two_hotspots_pinned` | 🔴 |
| AC2 | quantile 預設:thr=None ≡ `np.quantile(pmap, 0.98)`;分位頂到熱點值 → 框消失 | `test_ac2_quantile_default` | 🔴 |
| AC3 | min_area_frac 濾單像素雜點,保留 10×10 熱區 | `test_ac3_min_area_filters_speckle` | 🔴 |
| AC4 | 全零 pmap → `[]` | `test_ac4_empty_map` | 🔴 |
| AC5 | 非 2D 輸入 → ValueError(不靜默壓維) | `test_ac5_non_2d_raises` | 🔴 |
| AC6 | 寫檔往返:labels/ 兩 txt、conf 讀回一致、無框者空檔、classes.txt=="defect"、統計正確 | `test_ac6_write_round_trip` | 🔴 |
| AC7 | 冪等覆寫:改參數重呼 → 舊框不殘留 | `test_ac7_idempotent_overwrite` | 🔴 |
| AC8 | **裁切→原圖座標合成**:region==全圖 → identity;右半 region 手算 `(0.75,0.5,0.25,0.5)`;一圖兩物件 → 兩框合併、conf 降冪 | `test_ac8_crop_to_image_space` | 🔴 |

## 推導測試(設計未明列,PM 自加施壓)

| 測試 | 施壓的不變量 |
|------|-------------|
| `test_derived_translation_equivariance` | 熱區純平移 → 框中心等量平移、w/h/conf/框數不變 → 逼出「row/col 混用」「用絕對像素而非正規化中心」 |
| `test_derived_conf_sorted_descending` | 任意數量熱區,輸出恆按 conf 降冪(不只 AC1 的兩框) |
| `test_derived_constant_map_empty` | **全同值圖 → `[]`**(把「全零」一般化:沒有相對突起就沒有異常區) |

## GUI / 整合 E2E AC(Tier B done 要件)

| AC | 內容 | 測試 |
|----|------|------|
| AC-G1 | **端到端定位**:合成瑕疵資料 patch 建模+套用 → 產生預標 → 壞品圖至少一框與植入瑕疵區 **IoU ≥ 0.3**(真實行為斷言,防 false-green);良品圖 0 框或 conf 顯著較低 | `test_pred_localises_defect_e2e` |
| AC-G2 | 餵回管線:按「填入 Visualize 預測資料夾」→ `viz_pred_root` == out_root;10 號挖錯面板能以此跑出佇列 | `test_pred_feeds_pipeline_e2e` |
| AC-G3 | object 模式反灰:score_mode=object → 按鈕 disabled 且 tooltip 含「patch」;**誠實界線文案**(單類 / patch 粒度 / 瑕疵限定)真的出現在 UI | `test_object_mode_disabled_and_honest_caption_e2e` |

## done 判準

- **單元綠**:`python verify/gate.py heatmap_to_boxes` 印 `GREEN`。
- **AND 真實 E2E 綠**:AC-G1 是**防 false-green 的核心**——「有產出 txt」不等於「框真的框到瑕疵」,
  故用 IoU ≥ 0.3 對植入的已知瑕疵區做真實行為斷言(IoU 用 10 號的 `iou_cxcywh`,測試層 import,
  產品碼不循環)。
