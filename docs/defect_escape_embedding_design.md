# 漏檢嵌入挖掘 — 設計文件（B：嵌入工具吃預測 + 信心過濾 + escape 群聚視圖）

> 2026-06-21 · 設計層級，**尚未開發**（A「裁切可靠度過濾」已實作並驗收；本文為 B）
> 承接：[瑕疵問題重定義](defect_problem_redefinition.md)（四桶分類）、
> [需求建議](defect_requirements_recommendation.md)（R0/R0a/R4/R5）、
> [瑕疵漏逮分析](defect_escape_analysis.md)。
> 重用：`scripts/evaluation.py`、`scripts/signal_strength.py`、`_crop_and_embed_objects`、
> `interaction.load_predictions_csv`、完整度熱力圖 coverage 散點基建。

---

## 0. 立場與張力（先講，不可略過）

使用者要的是一份**多訊號「漏檢候選清單」**——自動找出 YOLO「完全沒看見」的 case，
輸出 `Case | YOLO | Frame Diff | DINO Similarity | 建議`。這在精神上對，但**與本 repo
已簽署的 [需求建議](defect_requirements_recommendation.md) 直接相牴觸**：

> 「在你量到『四桶各佔多少』之前，任何功能都是在賭某個桶比較大……**在此之前不建任何
> 自動判定機制。**」並且：**R0a signal_strength 只當桶①篩選器、不當漏逮成因**；
> **R4（逐型態覆蓋）/R5（邊界帶候選）為條件性解鎖**，須先量到該桶佔比可觀才解凍。

把使用者的 5 個 case 對到四桶，會看清「這份清單本質是**自動歸因 escape**」：

| 使用者的 case | 偵測軸 | 四桶歸屬（不確定，需量測） |
|---|---|---|
| ① GT 有 defect、YOLO 無 proposal | escape vs GT | 桶②覆蓋 / 桶④模型 / 桶①訊號 — **無法只憑此判斷** |
| ② YOLO 有 proposal、conf 太低被濾 | 門檻 | 桶④模型不確定性 |
| ③ YOLO 框偏太遠、IoU 不足 | 定位 | 桶④定位 |
| ④ Frame Diff 有訊號、YOLO 沒抓 | 獨立偵測器（需影片） | 非桶①（有訊號）→ 桶②/④ |
| ⑤ DINO 像 defect cluster、YOLO 沒框 | 獨立偵測器（嵌入） | 桶②/④ |

**化解（分階段，本文採用）**：

- **Phase 0 — A（已完成）**：裁切可靠度過濾。純資料衛生，無爭議，已實作驗收。
- **Phase 1 — B（本文）**：把**預測**帶進嵌入空間 + 信心過濾 + escape/低信心**群聚視圖**。
  界線：**呈現訊號為「描述性註記」，建議欄是「人工分流提示」，永不自動判定**。這守住
  R0「列出 ≠ 歸因」「不建自動判定」的底線——我們做的是**讓 escape 在嵌入空間裡被看見**，
  不是替它蓋章歸因。
- **Phase 2 — 完整融合（gated，本文只給規格不解鎖）**：frame-diff/DINO 獨立掃描提案 +
  風險融合 + 自動建議。**解鎖條件＝R4/R5：先由四桶量測證明桶②/桶④佔比可觀**。

> 反向閘門（給 PO）：若你已完成四桶量測、或明確要推翻需求文件而直上 Phase 2，請在審查時
> 註明；否則本文只承諾 Phase 1。

---

## 1. 目的（Purpose）

一句話：**把模型的「漏檢（escape）」與「低信心預測」放進既有的 DINOv2 嵌入空間，
讓使用者「看見」哪些 GT 被漏、它們是否群聚、是否靠近已知 defect cluster——而不是只盯 mAP。**

---

## 2. 範圍（Scope）

B 由三塊組成，全部重用既有基建：

1. **預測物件入嵌入**：`predictions.csv` → 預測框 → 走**同一條** `_crop_and_embed_objects`
   裁切/嵌入管線（預測框就是框，只是多帶 `score`）。
2. **信心過濾**：對預測 `score` 的滑桿（語義同評估 tab 的「信心門檻」），與 A 的「裁切
   可靠度」滑桿**正交共存**（一個濾框太小、一個濾信心太低）。
3. **escape 群聚視圖**：在同一嵌入空間同時畫 GT、預測（TP/FP）、**escape（FN）**，並對每個
   escape 算「到最近 defect cluster 形心的 DINO 餘弦相似度」→ 看 escape 是否群聚、是否像
   已知 defect。

---

## 3. 輸入 / 輸出介面（I/O Contract）

### 3.1 輸入
- **GT**：資料夾 `images/ + labels/`（既有 `_eval_gt_by_image` / `parse_yolo_boxes`）。
- **預測**：`predictions.csv`，欄位 `filename,class,cx,cy,w,h[,score]`（既有
  `interaction.load_predictions_csv`，回傳 `{filename: [{cls,cx,cy,w,h,score}]}`）。
  - **現實限制**：要看見 case②「conf 太低被濾」，CSV 必須是**低門檻/NMS 前**的 proposals
    （含低分框）；只匯出最終檢測會讓 case② 不可見。文件須在 UI 明示此前提。
- **defect 類別**：哪些類別算「defect」（單缺陷檢測常只有 1 類；室內 demo 為 10 類物件）。

### 3.2 新增資料結構（概念簽名，實作階段定案）
```
EscapeRow = {
  "filename": str, "cls": str, "box": {cx,cy,w,h},   # 來自 GT
  "yolo": float | None,        # 最佳重疊預測的 score；None = 完全沒 proposal（case①）
  "yolo_reason": "no_proposal" | "low_conf" | "loc_miss",
  "frame_diff": float | None,  # 影片來源才有；靜態圖 = None（"N/A"）
  "dino_sim": float,           # 到最近 defect-cluster 形心的 cosine ∈ [-1,1]
  "signal": "明顯|疑似|確無|未知" | None,  # signal_strength（描述性註記）
  "suggest": "高風險漏檢" | "建議人工覆判" | "可能FA",
}

DefectReference = {
  "centroids": {cls: vec[384]},  # 各 defect 類別 GT crop 嵌入的形心（L2 對 cosine）
  "global": vec[384],
}
```

### 3.3 概念函式（framework-free 核心，可單元測試）
- `build_predicted_object_records(pred_by_image, image_root) -> list[record]`
  把預測框轉成 `_crop_and_embed_objects` 吃得下的 record（帶 `score`、`bbox`、`image_path`）。
- `build_defect_reference(gt_records, gt_emb, defect_classes) -> DefectReference`
- `dino_similarity(emb_row, ref: DefectReference) -> float`（cosine 到最近形心）
- `score_escapes(gt_by_image, pred_by_image, gt_emb, ref, *, iou_thresh, conf_thresh,
  signal_fn=None, frame_diff=None) -> list[EscapeRow]`
  內部**直接呼叫** `evaluation.match_image` 取得 (tp, fn, fp)，再把 fn 標註成 EscapeRow。
- `recommend(row) -> str`（§6 決策規則）。

### 3.4 輸出
- **散點**：GT / 預測TP / 預測FP / escape 多 trace（重用 coverage 的 `_build_cov_scatter`
  風格與框選看圖）。
- **漏檢候選表**：`EscapeRow` 攤平成表（§5 落點）。
- **匯出**：`escape_candidates.csv`（遵守 **no-dataset-writes**：寫 `.lv_cache` + 下載鈕，
  不回寫資料集）。

---

## 4. 資料流（Data Flow）

```
GT(images+labels) ──parse_yolo_boxes──┐
                                       ├─ _crop_and_embed_objects(GT) ─→ gt_emb（已含 A 可靠度過濾）
predictions.csv ──load_predictions_csv─┘                                   │
        │                                                                  ├─ build_defect_reference(defect_classes)
        └─ build_predicted_object_records ─ _crop_and_embed_objects(pred) ─→ pred_emb（信心過濾 + A 可靠度過濾）
                                                                           │
evaluation.match_image(GT, pred, iou,conf) ─→ (tp, fn, fp)                 │
        │                                                                  ▼
        └─ fn ─→ score_escapes：每個 escape 標 yolo_reason + dino_sim(到 ref) + signal(可選) + suggest
                                                                           │
            ┌──────────────────────────────────────────────────────────────┘
            ▼
   多 trace 散點（GT/TP/FP/escape）＋ 漏檢候選表 ＋ escape_candidates.csv（.lv_cache + 下載）
```

關鍵重用點（file:line 為現況）：
- IoU 配對 / conf 門檻 / FN 清單：`scripts/evaluation.py`（`iou_xywh`、`match_image`、
  `evaluate_detections`）—— **不重寫，直接呼叫**。
- 裁切/嵌入（含 A 的 `short_px` 與 `obj_min_short_px` 過濾）：`_crop_and_embed_objects`。
- 散點 + 框選看圖 + 候選排序：完整度熱力圖 `_render_coverage_view` / `_build_cov_scatter`。
- 訊號強度註記：`scripts/signal_strength.py`（`classify_signal`，**僅描述性**）。

---

## 5. 輸出落點與融合 / 建議規則

### 5.1 落點：延伸『評估』tab（不開新 tab）
- 評估 tab 已吃 `predictions.csv`、已有「信心門檻」、已產 FN 漏抓畫廊——**B 是它的嵌入升級版**：
  在漏抓畫廊旁加「**escape 嵌入視圖**」（多 trace 散點）＋「**漏檢候選表**」。
- 信心過濾沿用評估既有 `eval_conf`，避免兩套信心語義。

### 5.2 風險融合（透明、規則式，非學習模型）
每個 escape 的 `dino_sim` 與 `signal`（可選）與 `yolo_reason` 進**決策表**（門檻可校準）：

| 條件 | 建議 |
|---|---|
| escape（FN）且（`dino_sim` 高 **或** `signal=明顯`）且非低訊號 | **高風險漏檢** |
| 有低分 proposal（case②）或 `dino_sim` 中等或證據邊界 | **建議人工覆判** |
| `dino_sim` 低 且（`signal=確無` 或無其他佐證） | **可能 FA**（GT 標註本身可疑） |

> 原則：建議是**人工分流提示**，每列恰好一個標籤；不自動改 GT、不自動判桶。

### 5.3 Frame Diff 欄
- 僅當輸入是**影片/連續幀**時才有值；對靜態圖資料夾（如 `…/[Small]/yolo`）一律 `N/A`，
  其餘欄照常產出。frame-diff 提案器屬 **Phase 2 gated**，本文不實作、只保留欄位與介面。

---

## 6. 邊界條件與錯誤處理（Edge Cases）

| 情境 | 處理 |
|---|---|
| 沒上傳 predictions.csv | 整個 B 不可用（無法知道 YOLO 看見什麼）；提示去評估 tab 上傳。退回純 GT 嵌入。 |
| CSV 只有最終檢測（無低分框） | case② 不可見；UI 明示「要看低信心漏檢請匯出低門檻/NMS 前 proposals」。 |
| class 命名空間不符（CSV id vs GT 名） | 沿用評估既有偵測 + 提示（或關 class-aware）。 |
| 無 defect-cluster 參照（單類/crop 太少） | `dino_sim` 退化為「到全體 GT 形心」或標 `未知`，不臆造。 |
| 靜態圖（無連續幀） | `frame_diff = N/A`；其餘欄照常。 |
| GT crop 或預測 crop 短邊過小 | 由 **A 的 `obj_min_short_px`** 一併過濾（兩條濾鏡正交）。 |
| 空 GT / 空預測 | 對應表為空 + 友善說明，不崩潰（同 `evaluate_detections` 既有契約）。 |
| 大量點（>數千） | 沿用 coverage 既有 WebGL 切換與投影快取。 |

---

## 7. 驗收標準（Acceptance Criteria，給 /pm 抓契約）

1. **預測入嵌入**：給 `predictions.csv`，預測框走 `_crop_and_embed_objects` 得到與 GT 同維
   （384）的嵌入；每個預測 record 帶 `score` 與 A 的 `short_px`。
2. **信心過濾**：調高信心門檻 → 散點上的預測點數**單調不增**；門檻=0 時等於全顯示。
3. **與評估數字一致**：escape（FN）集合 == `evaluation.evaluate_detections` 在同 `iou/conf`
   下的 `false_negatives`（同一套 `match_image`，數字必須一致）。
4. **escape 三因標註**：每個 escape 的 `yolo_reason ∈ {no_proposal, low_conf, loc_miss}`
   依 §3 規則正確（no_proposal=完全無重疊預測；low_conf=有重疊但 score<門檻；loc_miss=有
   重疊但 IoU<門檻）。
5. **DINO 相似度**：`dino_sim` = 到最近 defect-cluster 形心的 cosine ∈ [-1,1]，可重現。
6. **建議唯一**：§5.2 決策表對每列產生**恰好一個**建議標籤。
7. **Frame Diff 缺幀即 N/A**：靜態圖輸入時 `frame_diff` 欄全 `N/A`，其餘欄照常產出、不崩潰。
8. **no-dataset-writes**：所有產物只進 `.lv_cache` 或下載 CSV，**不寫回**資料夾的
   `images/`、`labels/`。
9. **A 正交共存**：`obj_min_short_px` 與信心門檻同時生效，互不覆蓋。
10. **Phase 2 gate**：frame-diff/DINO 獨立提案掃描與「自動風險判定」預設**關閉**，需顯式
    解鎖旗標＋四桶量測前提，否則不顯示（守 R4/R5）。

---

## 8. 與 A 的關係（已完成部分）

A =『裁切可靠度過濾』已實作：`_crop_and_embed_objects` 內每個物件帶 `short_px`，共用
session key `obj_min_short_px` 後過濾，門檻併入 token；四個嵌入工具（Visualize/Compare/
coverage/objcov）皆露出滑桿。B 帶進來的**預測 crop 自動繼承**這條過濾——這就是先前
「A 與 B 正交、且 B 複用 A 模式」的落實點。

---

## 9. 開放問題（審查時請 PO 裁示）

1. **是否解鎖 Phase 2**（frame-diff/DINO 獨立提案 + 自動風險融合）？需先確認四桶量測狀態
   （見 §0 反向閘門）。
2. **defect 類別定義**：單缺陷 vs 多類；室內 demo（10 類物件）僅供打通流程，非真 defect。
3. **預測來源**：是否要求使用者匯出低門檻/NMS 前 proposals（否則 case② 不可見）？
4. **frame-diff 資料合約**：影片/連續幀如何進站（檔名排序？時間戳？），Phase 2 再定。
