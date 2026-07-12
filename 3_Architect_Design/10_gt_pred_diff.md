# 10 — gt_pred_diff 技術設計(GT×預測比對挖錯 + 預測框預標 seed)

對應 PRD:[2_PO_PRD/al_loop_infrastructure_prd.md](../2_PO_PRD/al_loop_infrastructure_prd.md) · 里程碑 M10 · Tier B(核心純函式為 A 級品質)

## 1. 目的(Purpose)

讓「模型自己」參與迴圈:同時載入 GT 與模型預測(YOLO 6 欄 txt,或 13 號模組由熱力圖產生的同格式檔),
產出四種可送標佇列 —— **FN(漏抓)/ FP(誤抓)/ 類別混淆 / 低信心** —— 並讓送標 seed 改用
**預測框當預標**(標註者改框而非畫框)。

## 2. I/O 契約

### 2.1 新模組 `scripts/gt_pred_diff.py`(依賴僅 stdlib + label_formats/interaction 的既有解析)

```python
def iou_cxcywh(a: tuple, b: tuple) -> float
    # a,b = (cx,cy,w,h) 正規化;轉角點算 IoU;無交集回 0.0;任一 w/h<=0 回 0.0。

def match_boxes(gt: list[tuple], pred: list[tuple], iou_thr: float = 0.5) -> dict
    # gt   = [(cid, cx, cy, w, h), ...]                (parse_yolo_boxes 輸出形)
    # pred = [(cid, cx, cy, w, h, conf|None), ...]     (parse_yolo_boxes_conf 輸出形)
    # 貪婪匹配:pred 依 conf 由高到低(conf None 視為 1.0、同 conf 依原順序),每個 pred 取
    # IoU 最高且 >= iou_thr 的「未被占用」gt;決定論(不依賴 set 順序)。
    # 回 {"tp": [(gi, pi, iou)], "fn": [gi...], "fp": [pi...],
    #     "confused": [(gi, pi, iou)]}   # IoU>=thr 但 class id 不同 → 從 tp 移到 confused
    #     tp/confused 依 pi 升冪;fn/fp 升冪。

def image_error_report(gt, pred, *, iou_thr=0.5, low_conf=0.5) -> dict
    # 疊在 match_boxes 上:加 "low_conf": [pi...](conf 非 None 且 < low_conf 的 pred,含已匹配者)
    # 與 "score": float = 2*len(fn) + 1*len(fp) + 2*len(confused) + 0.5*len(low_conf)(排序用權重,釘死)。

def pred_label_path_for(image_path: Path, pred_root: Path) -> Path
    # 預測資料夾支援兩種佈局:pred_root/labels/<stem>.txt(YOLO 慣例)或 pred_root/<stem>.txt(平鋪)。
    # labels/ 子目錄存在 → 用它;否則平鋪。只回路徑,不檢查存在。

def scan_dataset(records: list[dict], pred_root: Path, *, iou_thr=0.5, low_conf=0.5) -> list[dict]
    # records = viz 現行整圖記錄({"path",...});對每張圖:
    #   gt   = parse_yolo_boxes(yolo_label_path_for(path));若 gt txt 缺 → 走 annotation_boxes_for_image
    #          (COCO/VOC/LabelMe/NDJSON,重用 label_formats;取其 (cid,cx,cy,w,h) 前五欄)
    #   pred = parse_yolo_boxes_conf(pred_label_path_for(path, pred_root))
    # 回 [{"path", "report": image_error_report 結果, "n_gt", "n_pred"}...],
    # 依 report["score"] 降冪。兩邊皆空的圖不進列表(無事可看)。
```

### 2.2 預標 seed(`scripts/labeling_handoff.py` 擴充)

```python
def send_to_labeling(..., seed_source: str = "gt", pred_root: Path | None = None)
    # "gt"(預設,現況行為不變)| "pred":seed 的框改讀 pred_label_path_for(img, pred_root);
    # pred 檔缺→該圖 seed 空(fresh 畫框),不報錯;seed json 內每框帶 conf(供 Labeling 顯示)。
    # "pred" 而 pred_root=None → ValueError(明確契約)。
```

> **⚠ 反向閘門 /pg → /architect(2026-07-12):未經人手的預標不得被當成人工標註讀回。**
>
> PG 落 seed 時發現本設計與既有回讀端**語義衝突**:xAnyLabeling 是靠 `images/<sha>.json`
> 這個路徑自動載入預標的,所以 seed **必須**寫在該路徑;但 `_count_annotated()` 與
> `read_labeling_results()` 正是「該檔存在且有 label ⇒ 已標註」。天真實作的後果是
> **送標當下就顯示全部已標註**,而且回讀會把**模型自己的預測**當成人工答案套回 records
> —— 模型把自己的錯誤當真值餵回自己,正是主動學習最該避免的失敗模式(silent-wrong)。
>
> **設計裁決**:寫 seed 時把該檔的內容 sha256 記進 `_handoff.json` 的 item(`seed_sha`);
> 回讀的兩個唯一入口(`_count_annotated` / `read_labeling_results`)**一律跳過內容仍與
> `seed_sha` 逐位元組相同的 sidecar**。人只要在 Labeling 裡存過檔(即使只挪一個框),
> 內容就變了 → 正常計入。無 seed 的 handoff(`seed_source="gt"`)不帶 `seed_sha` → 行為完全不變。
>
> 誠實界線:標註者「開啟但原封不動存檔」時,若 Labeling 端序列化位元組完全一致,仍會被
> 判為未處理。此側的誤判方向是**保守的**(寧可少算已標,不可把模型預測當人標),故接受。
> 對應 AC9。

### 2.3 GUI 接線(Tier B,`scripts/app.py`)

- **輸入**:Visualize(Object Detector 模式)側欄加選填欄位「模型預測資料夾」
  (`[輸入框|📁]` 同列樣式,key=`viz_pred_root`;沿用單一取代語義)。
- **視圖**:填了預測資料夾且 Run 後,結果區新增「🧪 模型錯誤挖掘」panel:
  - 四個佇列 tab/segmented:FN / FP / 混淆 / 低信心;各佇列 = 影像卡牆(重用畫框
    `draw_yolo_boxes`:GT 紅框 + 預測藍框疊畫 —— 需要 `draw_yolo_boxes` 增加
    `color` 參數與可疊叫用,或新小函式 `draw_two_sources(path, gt, pred)`,設計定案:**新小函式**,不動既有簽名)。
  - 每佇列有「🛒 加入購物車」「📤 送標」(重用 `_batch_add` / `_send_to_labeling_ui`,
    策略名帶 `"pred_diff:fn"` 等 → 直通 09 回合帳本)。
  - `iou_thr` / `low_conf` 兩個 slider(預設 0.5/0.5),改動即重算(結果快取 key 含兩參數)。
- **送標 seed 切換**:`_send_to_labeling_ui` 加 radio「預標來源:GT / 模型預測」
  (只在 `viz_pred_root` 已填時出現;預設 GT)。

## 3. 資料流

GT(labels/ 或四格式標註)+ 預測資料夾(YOLO 6 欄;或 13 產出)→ scan_dataset →
四佇列(依 score 排)→ 人挑 → 送標(seed=pred)→ 標註者改框 → 回讀(09 入帳)。

## 4. 邊界條件與錯誤處理

| 情況 | 預期行為 |
|---|---|
| 圖無 GT、有 pred | 全部 pred 進 fp(誠實:沒有 GT 能對) |
| 圖有 GT、無 pred txt | 全部 gt 進 fn |
| 兩邊皆空 | 不進 scan 結果 |
| conf 全 None(GT 檔被當預測) | 匹配照跑(conf 視為 1.0);low_conf 佇列恆空 |
| pred 檔含 seg 行(>6 欄) | 由既有 parse 防呆跳過(不誤讀) |
| iou_thr=1.0 | 只有完全重合才算 tp;其餘照語義落 fn/fp |
| 同一 gt 被多個 pred 蓋到 | 只配最高 conf 那個,其餘 pred 落 fp(貪婪、決定論) |
| pred_root 不存在 | UI 即時 st.error,不進 scan |

## 5. Acceptance Criteria

**單元(`tests/test_gt_pred_diff.py`;純合成 box,不需影像檔)**

- **AC1(IoU 釘死)**:`iou_cxcywh((.5,.5,.4,.4),(.5,.5,.4,.4))==1.0`;
  `iou_cxcywh((.25,.5,.5,1.),(.75,.5,.5,1.))==0.0`;
  `abs(iou_cxcywh((.5,.5,.5,.5),(.625,.5,.5,.5)) - 0.6) < 1e-9`
  (A 角 x∈[.25,.75]、B 角 x∈[.375,.875],y 同 [.25,.75];交=0.375×0.5=0.1875,
  聯=0.25+0.25−0.1875=0.3125,IoU=0.1875/0.3125=**0.6**)。
- **AC2(tp/fn/fp)**:gt=2 框、pred=命中其一(IoU>thr、同類)+1 個遠框 →
  tp==[(0,0,iou)]、fn==[1]、fp==[1]。
- **AC3(類別混淆)**:同位置(IoU≥thr)但 pred 類 id 不同 → confused==[(0,0,iou)] 且 tp 為空。
- **AC4(貪婪決定論)**:兩個 pred(conf 0.9/0.6)蓋同一 gt → 0.9 者 tp、0.6 者 fp;重跑 100 次結果相同。
- **AC5(low_conf 與 score)**:report 對 fn=1,fp=1,confused=0,low_conf=2 的 fixture,
  `score == 2*1 + 1*1 + 0.5*2 == 4.0`。
- **AC6(pred 佈局雙形)**:`pred_label_path_for` 對「有 labels/ 子目錄」與「平鋪」各回正確路徑。
- **AC7(scan 排序)**:三張圖 score=4/0(不進列表)/7 → 回傳順序 [7 者, 4 者],len==2。
- **AC8(seed_source)**:`send_to_labeling(seed_source="pred", pred_root=fx)` 產出的 per-image json
  框數==pred txt 框數且含 conf;`seed_source="pred", pred_root=None` 拋 ValueError;
  預設(不傳)行為與現況 byte-level 一致(回歸)。
- **AC9(未動過的預標不是標註;見 §2.2 反向閘門)**:以 `seed_source="pred"` 送一批(每圖都有
  pred 框)→ ① `handoff_status()["n_annotated"] == 0`(送標當下不得顯示「已標註」);
  ② `read_labeling_results()` 每筆 `label is None`、`status == "pending"`;
  ③ `apply_readback(handoff, records)` 回 `[]` 且 **records 的 label 一個都沒被改**
  (模型的預測絕不可自動變成真值);④ 把某圖的 sidecar 改寫(模擬人動過)→ 該圖
  `n_annotated` 變 1、`read_labeling_results` 讀得到它的 label。

**GUI / 整合(真實 E2E,`tests/e2e/test_gt_pred_diff_e2e.py`)**

- **AC-G1(佇列真實性)**:合成資料集(2 類 4 圖:1 圖漏標、1 圖多框、1 圖類錯、1 圖全對)+
  對應 pred 資料夾 → FN 佇列恰含漏標那張(以檔名斷言),FP/混淆同理;全對那張不在任何佇列。
- **AC-G2(疊框渲染)**:佇列卡上的影像為「畫了兩色框」的版本(斷言 render 函式被走到:
  對照原圖 bytes 不同即可,不做像素級)。
- **AC-G3(送標接回合)**:從 FN 佇列送標 → 09 的回合卡策略名含 "pred_diff:fn"。
- **AC-G4(seed=pred 落盤)**:選「模型預測」為預標來源送標 → handoff `images/<sha>.json`
  的 shapes 數==該圖 pred 框數。

## 6. 給 /pm 的提示

- AC1 的第三個值:兩框各 0.5×0.5,水平位移 0.125 → 交集 0.375×0.5=0.1875,聯集 0.25+0.25−0.1875=**0.3125**,
  IoU=0.1875/0.3125=**0.6**。別讓實作反推測試——先寫死 0.6。(初版誤算聯集為 0.4375→3/7,已修正。)
- E2E 合成資料集造法同 `test_lv_to_labeling` 的 tiny YOLO fixture;pred 資料夾就是「改壞的 labels 副本」。
- `_send_to_labeling_ui` 的 strategy 參數是 09 的接點,PM 在對應表標注跨模組 AC(AC-G3)依賴 09 已落。
