# PRD:物件級瑕疵偵測(AnomalyDINO 風格)

對應需求:[1_user_needs/anomaly_defect_detection.md](../1_user_needs/anomaly_defect_detection.md)
參考方法:AnomalyDINO (WACV 2025) — training-free / few-shot,凍結 DINOv2 patch 特徵 + memory bank + 最近鄰距離。
模式:規格疊加(實作進 `scripts/` + `scripts/app.py`、測試進 `tests/`)。

## 三行自問(放行前缺一不可)
1. **誰會用**:做視覺品檢 / 資料集檢視的人,手上有「已用 YOLO 框好物件、絕大多數良品、極少瑕疵、且瑕疵樣本不足以訓練」的資料夾。
2. **不做會怎樣**:只能人工翻幾千~幾萬個物件、會漏;想用 AI 又因「瑕疵樣本太少 / 種類無法窮舉」無法走監督式;不平衡資料讓多數方法失去鑑別力。
3. **成功的可觀察判準**(直接餵 `/pm` 當 AC 來源):
   - 載入 YOLO 資料夾後,不手動裁切即列出物件。
   - 只用「少數正常 / 系統自動找出的正常群」即可運作,**不需任何瑕疵標註**即可產生分數。
   - 全資料集每物件得到 **好/壞標記 + 異常分數**,且**真瑕疵集中在分數最高的前段**(用使用者已有的少量壞標籤客觀量:壞件的分數排名顯著高於良品 / 分離度可量)。
   - 2D/3D 散點圖上良品聚團、壞件離群;**框選離群點 → 加入購物車 → 匯出原圖**全可用(沿用既有元件,行為一致)。
   - 點開可疑物件可見熱區,**熱區峰值落在缺陷處**(以合成注入缺陷驗證)。
   - 重跑 / 換正常認定時**不重算 DINOv2 特徵**(有快取),只重建 memory bank。

## 目標與成功指標(可衡量)
- **G1 分離度**:在帶標籤的驗證集上,壞件異常分數的排名分離達標(建議 image-level **AUROC ≥ 0.85** 作為驗收門檻;確切數字交 architect 在 AC 給可量化值)。
- **G2 標註成本**:使用者在 2-stage 流程中,**確認動作 ≤ 數十次**即可跑完整批(不需逐一標註全集)。
- **G3 互動延遲**:換正常範例 / 重跑分類,在已快取特徵下 **數秒內**重出結果(不重算特徵)。
- **G4 零資料集汙染**:所有快取進 `.lv_cache/`,不寫使用者資料夾(沿用既有鐵則 [[no-dataset-writes]])。

## 範圍(MoSCoW)
**Must(v1,端到端可用)**
- 讀 YOLO 物件(復用既有 `discover_yolo_objects`)。
- patch 級 DINOv2 特徵 + 快取。
- 正常 memory bank(含 coreset 下採樣)+ patch 最近鄰異常分數聚合成物件分數。
- 物件級 embedding 分群,標出「正常密群」與「離群候選」。
- 由分數(+ 少量已確認標籤)定門檻,將整批分好/壞並排序。
- GUI 頁籤:載入資料夾、選 **1-stage / 2-stage**、跑流程、接**散點圖 + 框選 + 購物車**、顯示排序清單、匯出原圖。

**Should(同輪盡量做,真的超時可移下一輪)**
- 熱力圖(把 patch 分數疊回物件圖,看「哪裡怪」)。理由:patch 流程本就算出分數圖,渲染成本低;且使用者明言「要看哪裡怪才敢信」。

**Could**
- GUI 切換「物件級 / patch 級」分數(預設 patch 級;物件級為較快的替代)。
- 2-stage 第二輪「邊界模糊樣本」主動學習再確認。

**Won't(本輪)**
- 瑕疵**種類**分類(刮傷/凹陷/污漬…)——只分好/壞。
- 完全不相干的多類別大雜燴資料夾(只支援「同一種物件、可含數種良品長相」)。
- 即時 / 串流。

## 模組分解(交給 `/architect` 逐一設計)
> 同名對齊:`3_Architect_Design/NN_<module>.md` → `tests/test_<module>.py` → `scripts/<module>.py`。
> 純模組(Tier A)以「特徵/分數陣列進、結果出」定義,**不依賴別的模組即可寫紅測試**(餵合成資料)。

| # | Module | Tier | 一句話職責(無「以及/還有」) | 對外相依 | MoSCoW |
|---|--------|------|------------------------------|----------|--------|
| 01 | `patch_features` | B | 把每個 YOLO 物件裁切框轉成 DINOv2 patch-grid 特徵並快取 | 既有 `discover_yolo_objects` + 模型載入器 | Must |
| 02 | `anomaly_score` | A | 用正常 patch 建 memory bank(coreset)並算查詢物件每個 patch 的最近鄰距離後聚合成物件異常分數與 patch 分數圖 | 吃 patch 特徵陣列 | Must |
| 03 | `bootstrap_cluster` | A | 把物件級 embedding 分群並標出正常密群與離群候選 | 吃物件 embedding 陣列 | Must |
| 04 | `anomaly_classify` | A | 由物件異常分數與少量已確認標籤定門檻並將整批分好/壞且排序 | 吃分數 + 少量確認標籤 | Must |
| 05 | `anomaly_heatmap` | A | 把 patch 分數圖疊到物件裁切圖上產生熱區覆蓋影像 | 吃 patch 分數圖 + 裁切影像 | Should |
| 06 | `anomaly_tool` | B | 新增 GUI 頁籤串起載入/流程選擇/散點圖框選購物車/排序清單/熱力圖/匯出 | 01–05 + 既有 viz/cart/export | Must |

**相依無環檢查**:06 → {01,02,03,04,05};01 → 既有函式;02/03/04/05 只吃陣列、彼此不互相 import。可獨立驗收。✅

## 使用者驗收(銜接 User 文件)
- 載入一個真實 YOLO 資料夾(images/ + labels/),不寫任何程式即跑完並看到好/壞排序。
- 2-stage:系統抽樣分群、只丟少數離群候選請使用者確認;確認後整批分類。
- 1-stage:直接對全集分群+分類。
- 散點圖框選 → 購物車 → 匯出原圖(非裁切圖)成功。
- (Should)點開可疑物件看到熱區,位置對得上注入的缺陷。

## 給架構師的開放問題(需技術判斷)
1. patch 特徵取哪一層 / 哪種輸出?是否沿用既有 `dinov2_vits14`(D=384)與 `embed_objects` 的快取機制與 key 格式?
2. coreset 方法與預算(greedy vs 隨機;memory 大小上限),以撐幾萬物件 × 上百 patch?
3. 最近鄰搜尋後端(sklearn vs faiss)在「幾千~幾萬物件」規模的取捨?
4. patch→物件分數的聚合函式(max / top-1% 平均),以及「物件級」替代分數(meanpool 距離)如何定義以支援 Could 的切換?
5. 分群演算法與參數(如 HDBSCAN min_cluster_size);如何客觀定義「正常密群 vs 離群候選」?
6. 門檻校準:有少量確認壞標籤時用什麼(如 Youden's J);**完全沒有確認壞**時的退路(百分位啟發式)?
7. 散點圖「位置用物件級 embedding、顏色用 patch 聚合分數」的 record 介面如何接既有 scatter / `selection_points_to_indices` / 購物車?

## 本輪 appetite
- **上限:此單一 feature 一輪做完,≤6 模組。** Must 五個(01–04, 06)不可砍;Should(05 heatmap)同輪盡量做,真的超時移下一輪。
- 本輪屬「**新增能力**」→ 受 appetite 約束。若後續發現「分數根本分不開好壞 / GUI 不可用」屬 **修不可用 / false-green**,該類**豁免** appetite、修到真能用為止。
