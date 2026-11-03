# 可解釋性與模型診斷工作區：完整設計、驗證與實作路線

> 狀態：三輪 Multi-agent 設計審查完成；P1 guided workspace、handoff、head reliability warning 與 E2E 已實作，後續證據階段仍延後
>
> 日期：2026-08-31
>
> 範圍：VisualLatent 的分類 head、Grad-CAM、DINO Attention、Anomaly Heatmap、資料診斷與多解析度驗證
>
> 最新的單一 CTA 與頁面簡化提案見 [model-diagnostics-ui-ux-redesign.md](model-diagnostics-ui-ux-redesign.md)。該文件提出 selector 無副作用、一次點擊完成 head 與熱圖；採用前需同步修改 OpenSpec。

## 1. Executive decision

建議新增一個頂層工具：

```text
模型診斷
```

頁面完整名稱：

```text
可解釋性與模型診斷
```

但這個 Page 不是把 Visualize 下方現有的所有控制項搬到新頁，也不是再做一個更大的熱圖檢視器。它的首要任務是依序回答：

1. 這個分類 head 能不能信？
2. 目前證據能不能判斷是資料不足？
3. 這張歸因圖是否忠實反映分類器的決策？
4. 歸因位置是否與真實缺陷一致？
5. 224、336、448 中，哪個解析度在相同驗證條件下真正較好？

產品採用雙層結構：

- Visualize inline panel：保留單張、快速、低負擔的檢視與診斷入口。
- 模型診斷 Page：負責可靠性、資料量、faithfulness、mask localization、真正多解析度與報告。

第一版 P1 必須刻意縮小：它是一個「研究診斷預覽」，不是完整模型驗證平台。P1 可顯示單張 signed Grad-CAM 與 selected-sample faithfulness，但在 grouped validation、batch mask benchmark 尚未完成前，不得產生「模型已可靠泛化」「已 focus 在 crack」或「定位可信」等結論。

## 2. 為什麼目前 `crack/001.png` 沒有穩定 focus 在裂痕

### 2.1 已知分類證據

以下是先前以本機完整 capsule cache（六類、132 張）得到的探索性診斷結果。它不是目前 committed E2E fixture 的輸出：現有 E2E 只抽取 `good/scratch/crack` 各 6 張，且主要 target 是 `scratch`，因此不會重現以下六類數字。P0/P1 必須補一份固定 dataset fingerprint、run spec 與 machine-readable report，之後這些數值才能成為 regression baseline；在此以前應視為 local exploratory evidence。

| 指標 | 結果 | 解讀 |
|---|---:|---|
| 5-fold sample-level OOF accuracy | 49.24% | Sample-level OOF predictive performance weak；cross-group generalization unknown |
| OOF balanced accuracy | 50.12% | 六類平均辨識能力弱 |
| OOF macro-F1 | 49.08% | 類別間表現不均且整體偏低 |
| `crack` precision | 50.0% | 被預測成 crack 的樣本仍有大量錯誤 |
| `crack` recall | 21.7% | 23 張 crack 只判對 5 張，漏檢嚴重 |
| `crack` F1 | 30.3% | 此類別尚未被 head 穩定學會 |
| `crack/001.png` OOF | `squeeze` 21.56% uncalibrated probability | held-out 判斷不是 crack |
| `crack/001.png` OOF crack uncalibrated probability | 20.41% | 未做 held-out calibration；crack 只略低於錯誤 top-1 |
| 448 spatial probe crack uncalibrated probability | 約 23.63% | 未校準，也不是經過 448 重訓與驗證的正式 head |
| 448 spatial probe runner-up | `poke` 約 18.60% uncalibrated probability | 決策仍相當模糊 |

`crack` 的 OOF confusion 也顯示它不是只與一個類別混淆：23 張中只有 5 張正確，另被判成 `good` 3 張、`poke` 6 張、`scratch` 7 張、`squeeze` 2 張。

這些證據指出：主要問題不是 overlay 畫錯，而是 whole-image embedding 加上目前分類 head 尚未建立可靠的 crack 決策邊界。

### 2.2 已知空間證據

`crack/001.png` 原圖與 ground-truth mask 約為 1000×1000；mask 約 23,268 pixels，占全圖約 2.3%。裂痕是小而細的區域。

以目前 224 head 作不同輸入尺寸的 spatial probe，曾得到以下單張離線探索結果：

| Probe | Native grid | 有 mask 的 patches | Top-K precision | 正向歸因 mass 在 mask patches | Patch AUC |
|---:|---:|---:|---:|---:|---:|
| 224 | 16×16 | 17 / 256 | 0% | 0.8% | 0.199 |
| 336 | 24×24 | 31 / 576 | 48.39% | 42.82% | 0.942 |
| 448 | 32×32 | 46 / 1024 | 43.48% | 39.37% | 0.888 |

這些數字尚未保存對應的 machine-readable artifact/run id，原始執行也沒有在本 repo 文件中完整記錄 attribution method、target/reference、Top-K 定義、ROC-AUC/PR-AUC 類型、mass denominator、occlusion fraction 與 random repeats。因此它們只能列為 `single-image offline exploratory evidence`，不是可重現的 batch localization benchmark，也不能成為 P1 的產品判決。P4 必須以完整 run metadata 重跑後才能引用為正式比較。

就這次探索結果而言，336 的數值高於 448；它只足以反駁「解析度越大一定越好」的預設，不能證明 336 已正式優於 448。

同一次未封存的探索中，448 top-patch occlusion 對 target/reference margin 造成約 0.457 的下降，隨機 patches 約為 -0.002。由於完整 baseline、K/fraction 與 random repeats 未封存，它只能形成「值得重跑的 faithfulness 假說」。即使重跑後成立，也只代表熱圖可能忠實呈現了目前分類器使用哪些區域，不代表分類器使用的是正確 crack 特徵。分類器可能依賴膠囊輪廓、黑橘邊界、文字、背景、破洞附近的大面積外觀或其他 shortcut。

### 2.3 目前最準確的結論

可以說：

> 目前六類 capsule 分類 head 的可靠性不足，尤其 crack recall 只有 21.7%。`crack/001.png` 的 held-out OOF 被判為 squeeze。Grad-CAM 可用來研究此分類器為何出錯，但目前不足以證明模型已找到裂痕。

不能說：

- 模型已 focus 在 crack。
- 紅色區域就是 crack。
- Grad-CAM 已完成 crack segmentation。
- 448 一定比 224 或 336 好。
- 目前已證明主要原因只是資料量不足。
- selected-sample occlusion 通過就代表分類器判斷正確。

## 3. 三輪 Multi-agent 審查紀錄

本設計由三個獨立視角進行三輪審查：

- 演算法／科學驗證：檢查 Grad-CAM 定義、分類可靠性、learning curve、faithfulness、mask localization 與 shortcut tests。
- 工程／pipeline：檢查現有 Streamlit state、資料流、context handoff、cache、stale、模組邊界與測試策略。
- 產品／UX：檢查新 Page 是否真正回答使用者問題、是否誤導，以及 inline 與診斷工作區如何分工。

### Round 1：獨立提案

三方各自得到相同核心結論：

- 要建立獨立模型診斷 Page。
- 目前 head 弱，不能把熱圖標為「建議聚焦」。
- signed 藍→白→紅語意必須保留。
- 高解析度輸入只是 probe；正式多解析度必須各自產生 embeddings 並重訓 head。
- Grad-CAM 是分類證據，不是 segmentation。

### Round 2：交叉挑錯

主要爭點：

| 爭點 | 衝突 | 最終方向 |
|---|---|---|
| 預設歸因方法 | 現有程式預設 token-wise；科學定義要求 canonical channel Grad-CAM | 先凍結、測試與重新命名，再把 canonical signed Grad-CAM 設為產品主路徑 |
| P1 要做多少驗證 | 完整科學 gate 很重要；一次全部實作風險過高 | P1 做 head 摘要、single-sample faithfulness、mask availability；完整統計分期 |
| 336／448 是否保留 | 有研究價值；UI 很容易誤導 | 從 inline 主流程移除，僅可作 advanced spatial probe，不進正式排行榜 |
| grouped OOF | 是可靠泛化的必要條件；目前缺正式 group contract | P1 顯示 sample-level 限制；P2 建立正式 group contract 與 grouped audit |
| artifact | 可重現性重要；自動落檔會破壞現有 no-cache-write 行為 | P1 session-only 加 metadata JSON；持久 evidence package 後續明確保存才寫入 |

### Round 3：範圍收斂

三方條件式同意以下分期：

```text
P0  正確性與資料契約
P1  單張研究診斷預覽
P2  Grouped OOF 與 Learning Curve
P3  True Multi-Resolution Heads
P4  Batch Localization、Full Faithfulness、Shortcut
P5  新定位模型與實驗平台（依證據決定）
```

最重要的共同裁決：

> P1 可以讓使用者研究弱 head 的錯誤，但不能因為圖成功產生就把狀態變成綠色；P1 最高只能記錄 `single-sample research smoke evidence`，不能支援 production reliability 或 crack localization。

## 4. 產品目標與非目標

### 4.1 Goals

- 把「能算 Grad-CAM」與「模型可信」明確分開。
- 把 actual label、current prediction、OOF prediction、手選 target 分開。
- 使用固定 signed 色盤解釋 target 相對 reference 的正負證據。
- 讓使用者知道結果是 exact head 還是 spatial probe。
- 用 learning curve 回答資料是否可能不足，而不是靠猜測。
- 用 mask 與 perturbation 客觀檢查定位與 faithfulness。
- 保留完整 provenance，避免舊結果配上新設定。
- 不複製 dataset、images、embeddings 或 Normal Bank。
- 把完整驗證與重運算移出 inline viewer。

### 4.2 Non-goals

- 不把 Grad-CAM 轉成生產用 crack mask 或 bounding box。
- 不用單張漂亮熱圖決定模型是否可上線。
- 不把 224 head 套在 448 input 後稱為 448 模型。
- 不在缺 group metadata 時宣稱跨 lot、device 或 acquisition 泛化。
- 不在缺 mask benchmark 時宣稱 defect localization。
- 不把 DINO Attention、Grad-CAM、Anomaly Heatmap 混成同一種證據。
- 不在第一版同時建立 MIL、segmentation、background worker 與實驗平台。

## 5. 三種熱圖必須維持不同語意

| 方法 | 回答的問題 | 必要前提 | 不能宣稱 |
|---|---|---|---|
| DINO Attention | Backbone token 注意哪些視覺區域 | 可載入相容 DINO 模型 | 分類理由、異常或 crack 位置 |
| Signed Grad-CAM | 哪些 patches 支持 target 相對 reference 的分類 margin | 相容且可微分的 classifier head | 通用 anomaly map、pixel segmentation |
| Anomaly Heatmap | 哪些 patches 與 Normal Bank 不相似 | 相容且可追溯的 Normal Bank | 某分類 head 為何選 crack |

三者可以並排研究，但不得合併成一個沒有明確定義的「可信度」。

## 6. Page 資訊架構

### 6.1 雙層介面

#### Visualize inline quick panel

保留：

- Attention／分類理由／Anomaly 的簡化入口。
- Reliability badge。
- Head 可用時的 canonical signed overlay；可靠度不足以 warning 呈現，不增加確認按鈕。
- Opacity 與 patch grid 顯示設定。
- 一句限制聲明。
- 「在模型診斷開啟目前樣本」CTA。

Inline 與 Page 必須共用相同 reliability semantics。Weak head 時，兩者都持續顯示可靠性摘要與研究限制；可用的分類證據動作不需要重複的「仍顯示研究用歸因」確認，且結果產生後仍保留警示。不能讓 inline 成為繞過可靠度語意的旁路。

移出第一層：

- 不再把解析度、方法與色彩語意分散在 Visualize 首屏；它們集中到模型診斷的 guided flow。
- token-wise／global-channel selector 仍放在模型診斷的進階研究選項。
- target-positive 與 signed 平行 selector 仍放在模型診斷的進階研究選項。
- 手選非 top-1 target。
- Session head 訓練作為主 CTA。
- Learning curve、mask metrics、shortcut 與正式 benchmark。

#### 模型診斷 Page

頁面以使用者問題組織，不以演算法名稱堆疊 tabs。實際操作順序固定為：

1. 從 Visualize 選取單張影像並開啟模型診斷。
2. 先選擇分析精細度：224、336、448；三個選項一開始就可見，預設為 224。
3. 準備對應解析度的分類 head：224 以明確按鈕建立；第一次選 336／448 直接建立該解析度的完整資料 head，不要求先建立 224，也不靜默沿用 224。
4. 產生分類證據熱圖：保留 signed 藍—白—紅語意；弱 head 以簡短限制提示持續標示，不再要求重複確認。
5. 解讀分類可靠度、OOF、faithfulness 與 localization 限制；技術 provenance 放在 progressive disclosure。

這個流程回答「這個分類 head 能不能信、是不是資料不夠、模型的分類證據在哪裡、哪個解析度值得後續驗證」，但不把單張熱圖宣稱為 crack segmentation。

目前 UI 的主畫面只保留三層必要資訊：

1. `分析結論`：先說明 head 是否可用、是否只能做研究解讀，以及目前的下一步。
2. 三個關鍵判斷：真實標籤、模型目前判斷、OOF（另附 margin 與校準狀態）。
3. `分類證據熱圖`：head 可用就直接顯示產生動作或結果；弱 head 的研究限制持續可見。

Context identity、三組 evidence badge、fold-level OOF 與 mask 限制集中在
`查看分析 context、可靠度與 OOF 詳細資料`，避免技術欄位與主要決策混在同一層。

### 6.2 固定 Context Header

頁面頂部始終顯示：

- Dataset 名稱、樣本數、類別數、fingerprint。
- Model、checkpoint、embedding recipe。
- Head identity、scope、pooling、訓練解析度、建立時間。
- Selected image 的縮圖、label、split、filename、SHA 短碼。
- Mask：available／unavailable／incompatible。
- Normal Bank：available／unavailable／incompatible。
- Result：current／cached／stale／historical。

### 6.3 P1 線框

```text
可解釋性與模型診斷

[Dataset: MVTec capsule · 132 images · fp:a81c…]
[Model: dinov2_vits14] [Head: session · 224 · CLS]
[Image: crack/001.png] [Mask: resolved by contract / unavailable] [Bank: unavailable]

目前結論：分類 head 不可靠
Balanced accuracy 50.1%；crack recall 21.7%。
此圖 OOF 判為 squeeze。以下歸因僅供研究錯誤決策。

[分析此圖（若缺 head，於同一次任務內先建立）]

分類證據
[Actual label] [Current top-1] [OOF top-1] [Target/reference margin]

Signed Grad-CAM
[原圖] [模型實際輸入] [藍—白—紅 overlay]
Blue = lowers target-vs-reference margin · White = neutral · Red = raises target-vs-reference margin

Selected-sample faithfulness
[Top-K margin drop] [Random control] [Supported/Unsupported/Inconclusive]

定位狀態
Mask available · localization not evaluated

[技術細節] [下載 metadata JSON]
```

### 6.4 Progressive disclosure

每個問題頁使用一致結構：

1. 結論。
2. 三至五個核心證據。
3. 下一步建議。
4. 技術細節 expander。
5. 加入報告或下載 metadata。

不要預設同時顯示全部 folds、全部 heatmaps、全部 resolution 與全部 JSON。

## 7. Context handoff 與 identity

### 7.1 原則

完整架構可接受 Visualize 或 Anomaly inspector，但兩者只傳 reference 與 provenance，不複製大型資料。P1 先交付 Visualize classifier handoff；Anomaly inspector handoff 排在 P1.1，避免在沒有分類 head 時暗中擴大 P1 驗收範圍。

建議建立 versioned contract：

```text
ExplainContextV1
```

欄位至少包含：

```text
context_id
context_version
origin_tool                  # visualize | anomaly
origin_view                  # viewer | inspector
origin_generation            # viz_data_token 或 anomaly generation
record_key
source_path
source_relative_path
source_sha256
source_dimensions
dataset_root_identity
dataset_fingerprint
manifest_ref
scope                        # whole_image | object_crop
source_granularity           # classifier_image | yolo_object | anomaly_object
object_index
bbox_normalized
bbox_pixel
crop_policy
label
split
model_id
checkpoint_id
embedding_recipe
embedding_ref
head_ref
head_provenance
mask_ref
normal_bank_ref
created_at
```

### 7.2 Stable record key

不可只用 array index。建議：

```text
whole image:
  source_sha256

object crop:
  source_sha256 + object_index + normalized_bbox + crop_policy
```

對 object-level record，模型來源必須是 full-resolution image + bbox + crop policy；viewer 的 cached JPEG crop 只可作顯示，不可取代模型輸入來源。

Anomaly 的 full-frame fallback 即使帶有 bbox，也不能被誤判為 object crop。`scope` 與 `source_granularity` 必須由 handoff 明確指定。

### 7.3 Handoff flow

Visualize fragment 中的 CTA：

```text
set explain_page_context
set tool_switch = 模型診斷
set app-level rerun flag
rerun at app scope
```

Page 開啟後：

1. 解析 context。
2. 驗證 origin generation。
3. 驗證 source content SHA。
4. 驗證 dataset fingerprint。
5. 驗證 model、checkpoint、scope、pooling、preprocess、resolution 與 head provenance。
6. 驗證 mask 與 Normal Bank 是否對應目前 recipe。
7. 不相容時顯示差異，不靜默選另一筆資料或另一個 head。

## 8. Provenance 與 stale contract

### 8.1 必須使 compute result stale

- Dataset generation 或 fingerprint 改變。
- Source image 內容 SHA 改變、刪除或不可讀。
- bbox、object index、scope、crop、padding、keep-aspect 改變。
- Model 或 checkpoint 改變。
- Head weights、classes、temperature、L2 normalization 或 provenance 改變。
- Pooling、target、reference、target scalar 改變。
- Attribution method、analysis resolution 改變。
- Occlusion baseline、faithfulness profile 或 seed 改變。
- Normal Bank identity 改變。
- Engine 或 schema version 改變。

### 8.2 只需重新 render

- Opacity。
- Palette。
- Patch grid 顯示。
- Overlay 尺寸。
- Signed map 與已存在的 target-positive display 派生切換。

Render-only 設定不得進 compute key，也不得重新執行模型。

### 8.3 Late result protection

每個 pending job 保存：

```text
requested_context_id
requested_generation
requested_source_sha256
requested_run_key
```

結果發佈前重新比對。若 context 已切換，結果只能成為 historical/stale，不可覆蓋目前結果。

## 9. 分類 head 可靠性

### 9.1 P1 必顯示

- Validation type：目前是 sample-stratified OOF。
- 明示不是 grouped holdout。
- Sample count、fold count、class support。
- Accuracy、balanced accuracy、macro-F1。
- Target class precision／recall／F1。
- Actual label。
- Current inference top-1、runner-up、margin。
- Calibration status；未經 held-out temperature/calibration 評估時，只能顯示 `uncalibrated probability`，不可稱為 calibrated confidence。
- 若該圖屬 OOF，顯示真正 OOF prediction。
- 若不屬 OOF，顯示「沒有 OOF prediction」，不可拿一般 inference 冒充。
- Chance／majority baseline，若可正確計算。

### 9.2 正交狀態與總結

不能把不同證據維度塞進單一先後順序，否則 P1 必然為 `localization_unverified`，卻又可能具有 selected-sample faithfulness 支持，狀態會互相覆蓋。Page 必須同時顯示三組正交 badge：

```text
Head validation
  incompatible_or_stale
  validation_insufficient
  sample_level_audit_available
  grouped_supported

Attribution faithfulness
  unavailable
  unsupported
  inconclusive
  single_sample_smoke_supported
  batch_supported

Localization
  unavailable
  incompatible
  unverified
  mask_agreement_supported
```

另以 `decision_ambiguous` 描述當次 prediction margin，不取代上述三個 badge。總結文案由三者合成，例如：

> Head validation insufficient；single-sample faithfulness smoke supported；localization unverified。此圖可研究分類器的當次錯誤依據，不能作為 crack 定位證據。

P1 不得產生整體 `diagnostic_supported` 結論。即使 attribution badge 為 `single_sample_smoke_supported`，head 與 localization 仍必須各自呈現限制，且不能移除 `validation_insufficient` banner。

### 9.3 暫定 gate

以下門檻只作初始 policy，正式產品門檻需依風險與真實 audit set 校準：

```text
macro-F1 或 balanced accuracy < 60%  => validation_insufficient
critical class recall < 80%          => 不得標為可靠類別解釋
ECE > 0.10                           => confidence 不可直接作決策
CI 過寬                              => validation_insufficient
```

目前 `crack recall = 21.7%` 必須明確 fail。

## 10. Learning curve：怎麼判斷是不是資料不夠

Learning curve 是回答「資料是否不足」的必要實驗，排在 P2。

### 10.1 實驗設計

- 固定一組 audit test set。
- 有 group metadata 時依 source／batch／lot／device／object group 切分。
- 不得由資料夾名稱、檔名或 parent path 默認猜 group。
- 訓練比例：10%、25%、50%、75%、100%。產品 UI 可先顯示 25%、50%、75%、100%。
- 每個比例 5 個 seeds 或 grouped bootstrap。
- 固定 embedding recipe、head family、hyperparameters 與 split policy。
- 先以 frozen embedding + linear head 執行；若 plateau 再測 fine-tuning 或其他模型。

### 10.2 指標

- Train 與 validation macro-F1。
- Balanced accuracy。
- 每類 recall，特別是 crack。
- Defect-vs-good sensitivity／specificity。
- AUROC、AUPRC。
- ECE、Brier score、NLL。
- 95% confidence interval。
- Seed variance。
- 每增加 10% 有效資料的邊際增益。

### 10.3 判讀規則

| 曲線現象 | 支持的解讀 | 不能直接推出 |
|---|---|---|
| 資料增加時 validation 持續上升 | 更多有效資料可能有幫助 | 只要加近重複圖片就會改善 |
| Train 高、validation 低 | 過擬合、資料多樣性不足或 leakage | embedding 一定沒問題 |
| Train 與 validation 都低且早停 | Feature、label、scope 或 head bottleneck | 資料量完全無關 |
| 整體提升但 crack recall 不升 | crack 多樣性或 confusion-pair 資料不足 | 所有類別都要同量補資料 |
| Seed variance > 5 percentage points | Split 或資料量不穩定 | 可用單次最好結果上線 |
| 最後兩段增益 < 2pp 且 CI 重疊 | 暫時支持 plateau | 永遠不用再收資料 |

### 10.4 補資料方向

若 curve 支持 data-limited，補資料應優先：

- 不同裂痕位置、大小、方向、深度與遮蔽。
- 不同照明、背景、拍攝設備、lot 與 acquisition batch。
- 容易與 crack 混淆的 `squeeze`、`poke`、`scratch` 困難反例。
- 少量高品質 mask 或 polygon，支援定位評估。
- 避免大量近重複或同一物件連拍只增加表面數量。

## 11. Grad-CAM 與歸因方法契約

### 11.1 Canonical signed Grad-CAM

產品主路徑定義為：

```text
target scalar = logit(target) - logit(reference)

alpha_k = mean_over_patches(gradient[k, p])
CAM[p] = sum_over_channels(alpha_k * activation[k, p])
```

不套 ReLU，保留正負證據：

```text
藍色 = 降低 target-reference margin／反對 target
白色 = 接近中性
紅色 = 提高 target-reference margin／支持 target
```

在六類問題中，藍色不必然表示某 patch 單獨支持指定 reference；第三類也可能參與負向 margin。只有嚴格 binary target/reference 情境才可額外解讀為 reference-directed evidence。

UI 名稱應為：

```text
Canonical signed Grad-CAM
```

### 11.2 Token gradient × activation

現有另一種方法：

```text
score[p] = sum_over_channels(activation[k, p] * gradient[k, p])
```

正確名稱：

```text
Experimental token-wise attribution
```

它可能較集中，但不是 canonical Grad-CAM，不得再標示「建議聚焦」。它只放在 advanced research comparison，且必須固定相同 image、target scalar、hook layer 與 normalization 後才可比較。

### 11.3 遷移順序

目前程式預設是 token-wise。為避免同時更動 Page、聚合方式與 cache identity，分成：

1. P0 先凍結兩種方法的 formula、名稱、raw output 與 deterministic tests。
2. 讓 metadata 明確記錄 attribution method。
3. 把 canonical signed Grad-CAM 設為產品主路徑。
4. Token-wise 保留於 advanced，不刪除研究能力。
5. 再建立 P1 Page，避免回歸時無法判斷來源。

### 11.4 Raw 與 display 分離

必須保留：

```text
raw_patch_scores
display_patch_scores
```

建議 display scale：

```text
scale = q99(abs(raw_patch_scores))
display = clip(raw / scale, -1, 1)
```

限制：

- 不對正值與負值分開 normalize。
- 不因圖近乎平坦就把噪聲拉滿。
- Flat map 顯示「無有效 attribution」。
- Target-positive/copper 只可由 raw signed map 派生，不可覆蓋 raw evidence。
- 舊結果若只有 copper map，不能偽造 signed negative evidence。

## 12. Selected-sample faithfulness 與完整 faithfulness

### 12.1 P1 smoke test

P1 對 selected sample 做：

1. 固定原圖選出的 target 與 reference。
2. 選 Top-K positive attribution patches。
3. 遮蔽或替換相同面積 patches。
4. 與多次 random patches 比較；`N_random` 必須記錄，初始 smoke-test policy 至少 5 次，預設值由效能 benchmark 校準。
5. 量測 target-reference margin drop。

顯示：

```text
K / fraction
occlusion baseline
top-k margin drop
random mean and range
supported / unsupported / inconclusive
```

這只能稱為 `sampled faithfulness smoke test`，不能推廣成整個模型通過 faithfulness。

### 12.2 P4 full faithfulness

- Exhaustive patch occlusion 或可控制的 blur／inpaint。
- Attribution ranking 與 causal margin-drop ranking 的 Spearman correlation。
- Top-K overlap。
- Deletion curve 與 AOPC。
- Top-K vs random bootstrap CI。
- 多種 perturbation baseline。
- 合理 augmentation 下的 attribution stability。
- 依 class、defect size、resolution 分層。

初始研究 gate 可考慮：

```text
top-k deletion 優於 random，且 95% CI lower bound > 0
AOPC > 0
augmentation Spearman >= 0.6
top-k Jaccard >= 0.4
```

最終門檻需以 validation set 校準，不能在 test set 上調到好看。

## 13. Mask localization

### 13.1 P1

只顯示：

- Mask available。
- Mask unavailable。
- Mask mapping incompatible。
- Mask available but localization not evaluated。

P1 不顯示實際 mask overlay 或與 Grad-CAM 並排，以免在尚未建立 geometry gate、baseline 與量化前誘導目測判定。實際 mask 比較延至 P4；P1 只顯示 availability 與 `localization not evaluated`。

### 13.2 幾何契約

每次量化必須可追溯：

```text
source image dimensions
mask dimensions
bbox / crop
padding
keep_aspect
resize target
model input dimensions
native patch grid
source-to-input transform
input-to-patch transform
```

尺寸或 mapping 不相容時禁止量化，不能自動 resize 後假裝有效。

### 13.3 P4 batch metrics

- Pointing game。
- Patch AUROC／AUPRC。
- Top-K patch precision／recall。
- Attribution mass precision／recall。
- Thresholded IoU、Dice，必須標示 research threshold。
- Mask 外 false-positive area。
- 依 class 與 defect size 分層。
- Bootstrap confidence interval。
- Random map、center bias、DINO Attention、Normal Bank baseline。

Grad-CAM 不是 segmentation，因此 thresholded IoU 不得稱為裂痕分割準確率。

## 14. True Multi-Resolution 與 Spatial Probe

### 14.1 Legacy spatial probe（僅保留作對照）

在尚未建立 exact head 的舊版流程中，224 head 套用 336／448 input 的能力只能回答：

- Patch grid 是否更細。
- Attribution 是否對輸入尺寸敏感。
- 同一個 head 在 distribution mismatch 下產生何種空間變化。

必須顯示：

```text
Backbone input: 448
Native patch grid: 32×32
Classifier head trained at: 224
Status: spatial probe / research only
Not a validated 448 classifier
```

此 legacy probe 不得：

- 出現在正式 resolution leaderboard。
- 產生推薦解析度。
- 比較未校準 confidence 後決定優勝。
- 被命名為 336／448 head。
- 留在 inline 一般主流程。

### 14.2 True multi-resolution

224、336、448 每個解析度必須各自具備：

```text
full-dataset embedding extraction
preprocessing recipe
pooling and normalization
recipe-aware embedding namespace
independent classifier head
head digest and provenance
same dataset snapshot
same group/fold policy
same seed policy
OOF and audit metrics
```

近似 native grids：

```text
224×224 → 16×16
336×336 → 24×24
448×448 → 32×32
```

正式比較表：

| Resolution | Exact head | OOF BA | Macro-F1 | Crack recall | Calibration | Faithfulness | Mask agreement | Cost |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 224 | native-224 | ... | ... | ... | ... | ... | ... | 1.0× |
| 336 | native-336 | ... | ... | ... | ... | ... | ... | 約 2.25× attention cost |
| 448 | native-448 | ... | ... | ... | ... | ... | ... | 約 4× attention cost |

先顯示 dataset-level table，再顯示 selected-image overlays。不能先讓使用者目測三張圖決定「哪張比較像 crack」。

跨解析度單張比較必須固定：

- Dataset snapshot。
- Target/reference policy。
- Signed color semantics。
- Normalization policy。
- Opacity。
- Geometry metadata。

若各模型的 actual top-1／runner-up 不同，分成：

- 解釋各模型自己的實際決策。
- 固定 target/reference 的研究比較。

兩者不可混在同一排而不標示。

### 14.3 Recipe-aware cache migration

目前類似 `embeddings_<model>/embeddings.npz` 的 namespace 不足以容納多解析度 recipe。P3 必須一起完成：

```text
model + checkpoint + resolution + pooling + scope
+ crop/pad + keep_aspect + l2norm
→ recipe key
→ embedding namespace
→ exact head registry
```

缺 recipe-aware cache、head registry 或 exact-head selection 任一項，就不能稱為 true multi-resolution。

## 15. Shortcut 與 counterfactual diagnostics

排在 P4，顯式執行，不自動跑。

建議測試：

- Background-only。
- Object-only。
- 遮蔽物件本體。
- 移除或遮蔽 `500` 文字。
- 更換背景。
- 改變膠囊左右方向。
- 控制殼體顏色。
- Brightness、blur、translation、rotation。
- 只保留疑似 defect 區域或移除 defect 區域。

記錄：

- Class flip rate。
- Target/reference margin 變化。
- Confidence 變化。
- Attribution mass 是否移到背景或文字。
- Background-only／text-only 是否仍保留高 target score。

可能 shortcut 的證據：

- Background-only 仍有高 target margin。
- 移除 defect 後 target score 幾乎不變。
- 只移除文字就造成大量 class flip。
- 小幅 label-preserving transform 造成不合理 flip。
- Attribution 大部分落在物件外。

這些測試揭露依賴，不會單獨證明唯一因果來源。

## 16. 何時改用 Normal Bank、MIL 或 Segmentation

### Layer 0：Whole-image classification + attribution

適合：分類 head 已通過可靠性、只需解釋分類決策。

目前 capsule 不符合，因為 OOF 與 crack recall 過低。

### Layer 1：Patch Normal Bank anomaly

適合：

- 有大量 good／normal images。
- 新缺陷類型可能出現。
- 目標是回答「哪裡不像正常」，不一定要立即命名 crack。

需評估 image AUROC／AUPRC、pixel AUROC、PRO、false alarm rate 與跨批次 holdout。

### Layer 2：MIL／weakly supervised patch classification

適合：只有 image-level label、缺少大量 pixel masks，但需要比 whole-image Grad-CAM 更直接的 patch-level reasoning。

MIL attention 仍不天然等於 faithful explanation，仍需 occlusion 與少量 mask validation。

### Layer 3：Segmentation

適合：產品真正要求輸出 crack 區域、邊界、面積或後續尺寸量測。

需要獨立 mask dataset、grouped split、Dice、IoU、pixel AP、boundary F-score、thin-structure recall 或 clDice。

只有 segmentation 通過獨立 mask 評估後，才能產品化「裂痕區域」。

## 17. 執行、job 與 cache UX

### 17.1 不自動啟動重運算

進 Page、切 tab、調 opacity、選圖都不能自動執行：

- 全資料 embeddings。
- OOF／grouped OOF。
- Learning curve。
- 三解析度 head 訓練。
- Batch mask evaluation。
- Shortcut suite。

### 17.2 執行前摘要

執行摘要必須依 phase 分開，不能讓 P1 看起來會自動啟動 P2／P3 工作：

```text
P1：1 次 selected-sample inference + Top-K/random smoke test
P2：約 5 ratios × 5 seeds = 25 次 learning-curve head fits，
    實際數量依 grouped split 與重複策略調整
P3：3 個 resolution embedding jobs + 各自 head/OOF jobs
P4：batch mask + full faithfulness + shortcut jobs，執行前另估
```

提供個別動作，不把「全部執行」當主要 CTA：

- 執行 head audit。
- 執行 learning curve。
- 建立 exact resolution head。
- 評估 selected-sample faithfulness。
- 執行 batch localization。

### 17.3 Phase 內進度

1. Validate context and provenance。
2. Resolve or build embeddings。
3. Build splits。
4. Train heads。
5. Compute OOF。
6. Compute attribution／faithfulness／mask metrics。
7. Store session result or explicit artifact。

多解析度時每一列獨立顯示 pending／running／cached／complete／failed／stale。取消後保留已完成結果。

### 17.4 Concurrency

P1 使用前景顯式執行。若後續加入 background jobs：

- 每個 model/device concurrency = 1。
- Worker 不直接呼叫 Streamlit API。
- Cancellation 採 cooperative cancellation。
- Model hook、parameter gradients 與 GPU memory 在 exception 後清理。
- OOM 時將 faithfulness 標為 unavailable，不得把 Grad-CAM 本身標為 supported。

## 18. Run 與 artifact contract

### 18.1 ExplainRunSpecV1

```text
kind                         # attention | gradcam | anomaly
scope
target_spec                  # auto_top1 | explicit class/index
reference_spec
target_scalar
attribution_method
analysis_target_res
pooling
keep_aspect
crop_policy
faithfulness_profile
occlusion_baseline
seed
engine_schema_version
```

### 18.2 RenderSpecV1

```text
display_mode                 # signed | target_positive
palette
opacity
show_grid
overlay_size
```

### 18.3 Deterministic compute key

以 canonical JSON 的 SHA-256 產生，至少包含：

```text
source_sha256
source dimensions
scope / bbox / object index
crop / padding / keep_aspect
model id
checkpoint content hash
head digest and provenance
embedding recipe
target/reference
attribution method
analysis resolution
faithfulness profile
baseline and seed
engine/schema version
```

不要使用 path + mtime、`id(result)`、`id(bank)` 或目前 widget 狀態作穩定 identity。

目前 `checkpoint_identity()` 仍以 filename／size／mtime 建立 identity；P0 必須新增 content digest，並以 schema version 區分舊 identity。舊 cache 不可靜默升級成 content-verified，只能標 legacy/stale 或在重新驗證後產生新 key。

### 18.4 Artifact policy

P1 預設 session-only，不改寫來源 dataset，也不改現有 embedding 或 anomaly cache。

P1 應提供 machine-readable metadata JSON，並另提供 raw patch array 的 JSON/NPZ 下載；至少包含 provenance、validation state、faithfulness summary 與可重建 display map 的 raw scores。現有 `metadata_dict` 只保留 raw min/max/sum，尚不足以重現 map，這是 P1 明確要補的 implementation delta，不是既有能力。

後續使用者明確按「保存診斷結果」才寫入：

```text
.lv_cache/explainability/runs/<cache_key>/
  manifest.json
  summary.json
  raw_maps.npz
  overlay_signed.png
  optional_target_positive.png
  occlusion.json
  stability.json
```

不複製來源影像；預設只保存 source SHA、相對路徑與 crop recipe。

## 19. 建議模組邊界

保留低階模組：

```text
dino_explain.py
  DINO model loading, attention, preprocess

gradcam_explain.py
  attribution primitives, signed normalization, occlusion primitive

head_adapter.py
  sklearn-to-torch adapter, provenance validation

dino_head.py
  head training, OOF, save/load

explanation_render.py
  pure colorize, overlay, grid, metadata
```

新增：

```text
explain_contracts.py
  ExplainContextV1, ExplainRunSpecV1, RenderSpecV1,
  result/status enums, canonical serialization

explain_context.py
  handoff, source identity, record resolution,
  stale validation, bbox/crop recipe

explain_engine.py
  Attention/Grad-CAM/Anomaly orchestration,
  no Streamlit, no implicit disk writes

explain_diagnostics.py
  reliability metrics, learning curve, mask metrics,
  faithfulness, shortcut probes

explain_artifact.py
  content-addressed save/load/list, atomic writes,
  schema migration and corruption checks

explain_page.py
  Page UI, run controls, session history, evidence layout
```

`app.py` 只保留 route、handoff callback 與 Page invocation。

`explain_ui.py` 最終縮為 inline adapter + compact controls。Page 與 inline 必須共用：

```text
UI controls
  → ExplainContext
  → ExplainRunSpec
  → explain_engine.run(...)
  → ExplainResult
  → explanation_render
```

Page 不得 import `explain_ui` 的 private functions，也不得重新實作 preprocess、target selection、signature 或 stale rules。

## 20. 空狀態與錯誤語意

### 無 Visualize context

> 尚未載入診斷情境。請先到 Visualize 執行資料分析，再選擇一張圖片開啟模型診斷。

### 有 dataset，沒有選圖

Head audit 與 learning curve 可用；空間分析要求先選圖。

### 沒有 head

- Attention 仍可用。
- Grad-CAM disabled。
- 顯示如何建立研究用 session head 或正式 validated head。
- Session head 標示暫時、未保存、sample-level audit。

### Weak head

> 分類 head 不可靠。熱圖只能用來研究分類器為何出錯，不能判斷裂痕位置。

主要 CTA：`分析此圖`；若缺對應 head，採用最新 UX 提案時於同一次任務內先建立。

不提供研究用途確認 CTA；限制提示持續可見即可。

### Provenance 不相容

完全阻止計算，列出 model、scope、pooling、resolution 或 crop 的差異。這與 weak head 的「可計算但不可靠」不同。

### 無 mask

> 沒有對應 ground-truth mask，因此無法客觀判斷歸因是否落在真實破損區。

### 無 Normal Bank

只停用 Anomaly Heatmap，不影響 head audit、learning curve、Attention 或 Grad-CAM。

### Context stale

保留歷史結果但加 stale banner，不得配上目前新設定；提供「以目前 context 重新執行」。

## 21. 報告與匯出

### 報告結構

1. 結論摘要。
2. Dataset／model／head provenance。
3. Validation split 與限制。
4. Head reliability 與 per-class metrics。
5. Learning curve 判讀。
6. Selected sample prediction、target/reference、signed attribution。
7. Faithfulness。
8. 有 mask 時的 localization benchmark。
9. True multi-resolution comparison。
10. Limitations 與不可宣稱事項。
11. 下一步資料或模型實驗。

### 格式

- Self-contained HTML：人員閱讀與交接。
- JSON：完整 provenance、policy、metrics、cache keys。
- CSV：fold、class、curve、resolution tables。
- PNG：signed overlay 或比較圖。
- Markdown summary：研究紀錄與 issue handoff。

原始影像是否嵌入報告由使用者明確決定；預設不複製 dataset。

## 22. 測試策略

### 22.1 Unit tests

- Context JSON round-trip。
- Context 不包含 PIL image、model object、整批 embeddings 或 bank vectors。
- Source hash、bbox、scope、object identity 穩定。
- Full-frame anomaly record 不被誤判為 object crop。
- Compute key 對 source/head/checkpoint/recipe/target/method 敏感。
- Opacity/grid/palette 不改 compute key。
- Stale/invalidation matrix。
- Head digest deterministic。
- Checkpoint content hash。
- Signed negative／zero／positive color mapping。
- Flat map 不放大噪聲。
- Raw/display scores 分離。
- Target/reference 在 occlusion 中固定。
- Artifact atomic write 與 corrupt artifact rejection。
- Artifact 不寫入來源 dataset。

### 22.2 Engine integration

- TinyViT CLS Grad-CAM。
- Mean-pool Grad-CAM。
- Canonical channel vs token-wise attribution 正確命名。
- Signed negative evidence。
- All-negative／flat map。
- Target/reference margin。
- 224／336／448 native grid shape。
- Whole-image non-square preprocess。
- Object crop bbox geometry。
- Hook cleanup。
- Exception／OOM degradation。
- Model lock and concurrency。

### 22.3 Head diagnostics

- Binary／multiclass adapter parity。
- Temperature、L2 normalization parity。
- Missing provenance blocks Grad-CAM。
- Scope／pad／model／checkpoint mismatch blocks。
- Resolution override 只可標 spatial probe。
- Normal／Unknown 不可作 Grad-CAM target。
- OOF current prediction 與 held-out prediction 不混淆。
- Target class support 不足時顯示 unavailable。
- Group metadata 缺失時不可偽造 grouped split。

### 22.4 UI component tests

- Inline CTA 建立 context 並切換 top-level tool。
- Fragment handoff 成功 app-level rerun。
- Page 進入不自動 inference。
- Weak head 不自動產生 heatmap。
- 使用者可明確選擇研究用歸因。
- Result 綁定正確 context。
- Context 過期顯示 stale。
- Opacity/grid 只重新 render。
- Signed 為預設。
- Target-positive 是 advanced。
- P1 selector migration 完成後，inline 不顯示 resolution／method selector；P0 compatibility 階段仍保留既有 controls。
- Page 不複製 source image 或 embedding。

### 22.5 E2E

1. P1：Visualize classifier → viewer → 模型診斷 handoff。
2. P1.1：Anomaly inspector → 模型診斷 handoff；沒有相容 head 時只顯示 Attention、既有 anomaly evidence 與 Normal Bank provenance，Grad-CAM unavailable。
3. `crack/001.png` 顯示 same image、label、split、model、dataset fingerprint。
4. Whole-image 1000×1000 → input 224×224 → grid 16×16。
5. Object crop 使用原圖 + bbox + crop policy。
6. Signed blue-white-red default 與文字 legend。
7. Current prediction、actual label、OOF prediction 正確分離。
8. Weak head warning 在 heatmap 產生後仍保留。
9. 新 Visualize Run 後舊 Page result stale。
10. 換 model/head/checkpoint 後舊結果不可沿用。
11. Opacity 不觸發模型重算。
12. Spatial probe 不出現在正式 benchmark。
13. 明確 save 前不寫 `.lv_cache/explainability`。
14. Source dataset 與既有 embedding/manifest cache fingerprint 不變。
15. 未被產品規格改動的既有 inline explainability E2E 維持通過；resolution/method controls 的既有 E2E 在 P1 deprecation 時遷移到 Page 或 engine coverage，不能同時要求「移除控制項」與「原 selector 測試原封不動通過」。

## 23. 分期 Backlog

### P0 — Correctness and contracts

目標：先讓方法名稱、色彩、identity 與 stale 行為可測試，再建 Page。

- [ ] 定義 `ExplainContextV1`、`ExplainRunSpecV1`、`RenderSpecV1`。
- [ ] 建立 deterministic canonical serialization 與 compute key。
- [ ] 建立 source/checkpoint/head/bank content digest。
- [ ] 凍結 canonical channel Grad-CAM 與 token-wise attribution 的公式和名稱。
- [ ] 加入 negative／zero／positive signed color deterministic tests。
- [ ] 加入 raw/display separation 與 flat-map tests。
- [ ] 把 canonical signed Grad-CAM 設為產品主路徑。
- [ ] 保留 token-wise 於 advanced research。
- [ ] 建立 stale/incompatible 狀態與 precedence。
- [ ] 抽出 framework-free context/engine boundary。
- [ ] P0 保留現有 inline selectors 與 E2E，先建立 engine/context compatibility。
- [ ] 定義 `explain_page_*` namespace，不受一般 tool navigation 的 `_MODE_CLEAR_KEYS` 誤清除；新的 dataset generation 仍必須使其 stale。
- [ ] 為 checkpoint content digest 與 legacy path/mtime cache 建立 migration policy。
- [ ] 固定完整 capsule diagnostic evidence：dataset fingerprint、六類 counts、run spec、`crack/001.png` identity、1000→224 geometry、OOF metrics 與 machine-readable report；外部 dataset 不複製進 repo。
- [ ] 確保 no-cache-write 與未受規格改動的 inline E2E 維持通過。

完成定義：方法、色彩、context、run key 與 stale matrix 全部有測試，且未建立新 Page 也能重現現有結果。

### P1 — Single-sample research diagnostic preview

目標：一張圖、一個 head reliability summary、一張 canonical signed map、一個 faithfulness smoke test。

- [x] 新增頂層「模型診斷」route。
- [x] Visualize classifier handoff。
- [ ] 以固定 capsule evidence 驗證 `crack/001.png` 的 image identity、label、OOF record 與 geometry；fixture/report 缺失時不得把探索數字當 regression pass。
- [x] 固定 Context Header。
- [x] 顯示 current／actual／OOF prediction 與 target/reference margin。
- [x] 顯示 sample-level OOF、balanced accuracy、macro-F1、target recall 與限制。
- [x] Weak head 先顯示判決，不自動產生 heatmap。
- [x] 移除研究用途確認，直接提供正常分析動作並持續顯示限制。
- [x] 顯示 canonical signed Grad-CAM 與 legend。
- [ ] Selected-sample Top-K vs random faithfulness。
- [ ] 新增 mask resolver／mapping status contract；resolver 未完成或無 fixture 時固定顯示 unavailable，不得由檔名猜測 available。
- [ ] 顯示 geometry 與 provenance。
- [ ] Session-only run history。
- [ ] Metadata JSON download。
- [ ] Raw patch scores JSON/NPZ download；不可只輸出 min/max/sum。
- [x] 將解析度放到模型診斷 guided flow，方法與 display-mode 收進進階研究選項；inline compatibility controls 仍保留。
- [x] 把解析度 E2E 遷移成模型診斷 workspace coverage。
- [x] 完成目前範圍的 P1 unit、UI、E2E gates；faithfulness、mask、下載與完整 evidence export 仍延後。

完成定義：使用者不會再因熱圖成功產生就看到「建議聚焦」；`crack/001.png` 會先顯示 head 不可靠與 OOF 錯判，再允許研究用 attribution。

### P1.1 — Anomaly inspector context expansion

目標：讓只有 Normal Bank、沒有分類 head 的 inspector context 也有明確而不誤導的工作區流程。

- [ ] Anomaly inspector handoff 與 stable bank identity。
- [ ] 顯示 Normal Bank provenance、scope、score 與既有 anomaly heatmap。
- [ ] 沒有相容 head 時明示 Grad-CAM unavailable。
- [ ] 允許 DINO Attention，但不把它稱為 anomaly 或分類理由。
- [ ] 加入 inspector-specific stale、no-copy、no-cache-write E2E。

完成定義：Anomaly context 不會因缺分類 head 落入空白頁，也不會自動借用其他 dataset 的 head 或 Normal Bank。

### P2 — Grouped validation and learning curve

目標：以實驗而非直覺回答「是不是資料不夠」。

- [ ] 定義正式 `group_id` metadata contract。
- [ ] Grouped OOF／grouped holdout。
- [ ] 固定 audit test set。
- [ ] Per-class metrics、confusion matrix、CI、calibration。
- [ ] Duplicate／near-duplicate leakage audit。
- [ ] 10／25／50／75／100% learning curve。
- [ ] Multiple seeds／grouped bootstrap。
- [ ] Train vs validation curve。
- [ ] Data-limited／plateau／high variance 的保守判讀。
- [ ] Confusion-pair 導向的資料收集建議。
- [ ] 執行前工作量與 cache 命中摘要。
- [ ] P2 regression 與真實資料報告。

完成定義：Page 可以用曲線、CI 與 grouped evidence 說「更多有效資料可能有幫助」或「目前更像 feature/model bottleneck」，但不自動宣稱唯一原因。

### P3 — True multi-resolution heads

目標：真正比較 224／336／448，而不是比較 probes。

- [ ] 定義 extraction recipe key。
- [ ] Recipe-aware embedding cache namespace。
- [ ] Manifest embedding refs migration。
- [x] Exact head registry 與 resolution-specific digest（目前為 session/on-demand；正式 benchmark 仍待 P3）。
- [ ] 224 全資料 embeddings + head + OOF。
- [ ] 336 全資料 embeddings + head + OOF。
- [ ] 448 全資料 embeddings + head + OOF。
- [ ] 相同 dataset snapshot、groups、folds、seeds。
- [ ] Dataset-level resolution table。
- [ ] Prediction flip、calibration、latency、memory 比較。
- [x] Selected-image exact-head selection（目前為單張研究流程）。
- [x] Spatial probe 與 exact mode 完全分離。
- [x] 沒有 exact head 時禁止靜默 fallback。
- [ ] P3 cache migration 與 backward compatibility tests。

完成定義：正式 leaderboard 中每一列都有該解析度自己的 embeddings、head、OOF 與 provenance；448 表現較差時 UI 不會推薦 448。

### P4 — Batch localization, faithfulness and shortcuts

目標：判斷歸因是否普遍忠實，以及是否真的對準 defect。

- [ ] Mask discovery／mapping contract。
- [ ] Source-to-input-to-patch geometry validation。
- [ ] Batch pointing、patch AUROC、Top-K、mass、false-positive metrics。
- [ ] Random／center／attention／Normal Bank baselines。
- [ ] Deletion curve、AOPC、bootstrap confidence interval。
- [ ] Cross-augmentation stability。
- [ ] Background／object／text shortcut suite。
- [ ] 依 class、defect size、resolution 分層報告。
- [ ] OOM degradation 與可取消 jobs。
- [ ] HTML／JSON／CSV／PNG evidence package。
- [ ] 真實 MVTec E2E 與 regression gate。

完成定義：只有 batch-level faithfulness 與 mask localization 通過，才可把狀態升級為「支持分類決策解釋」或「與標註區域一致」；仍不得稱為 segmentation。

### P5 — Evidence-driven localization model

只有 P2–P4 結果指出 whole-image attribution 無法滿足目標時才啟動：

- [ ] Normal Bank patch anomaly baseline。
- [ ] Patch classifier baseline。
- [ ] MIL prototype。
- [ ] Segmentation label budget 與 annotation workflow。
- [ ] U-Net／DeepLab／Mask2Former／ViT decoder 等 baseline。
- [ ] Thin-crack metrics：boundary F-score、thin-structure recall、clDice。
- [ ] 與分類 + Grad-CAM 的成本／效果比較。
- [ ] 根據產品目標決定正式方案。

## 24. Must／Should／Defer 摘要

### Must

- Context/provenance/stale 正確。
- Actual、current prediction、OOF prediction 分開。
- Head reliability 先於熱圖。
- Canonical signed Grad-CAM 與藍—白—紅。
- Token-wise 正確命名，不能再稱「建議聚焦」。
- Weak head 明確標為研究用途。
- Selected-sample faithfulness smoke test。
- Mask availability 與未驗證狀態。
- Probe 與 exact head 完全分開。
- P1 不自動重運算、不寫來源資料。
- Unit、integration、UI、E2E regression gates。

### Should

- Chance／majority baseline。
- Target support 與 confidence interval。
- Metadata JSON download。
- Opacity 預設約 0.45–0.60。
- Session history。
- 正式 grouped validation 與 learning curve。
- Recipe-aware true multi-resolution。
- Batch mask／faithfulness／shortcut。
- Evidence report。

### Defer until evidence requires it

- MIL。
- Segmentation。
- Attribution ensemble。
- 自動方法推薦。
- Background distributed jobs。
- Persistent experiment platform。
- Team sharing／remote artifact registry。
- 由 Grad-CAM 自動產出標註或生產 bounding box。

## 25. Release gates

### P1 release gate

- [ ] Context/run key deterministic。
- [ ] Incompatible 與 weak head 是兩種不同狀態。
- [ ] Stale result 不會覆蓋目前 context。
- [ ] Sample-level OOF 限制醒目可見。
- [ ] Current／actual／OOF prediction 語意正確。
- [ ] Canonical signed method 與 color tests 全綠。
- [ ] Raw attribution 可下載／重現。
- [ ] Top-K vs random smoke test 可重現。
- [ ] Mask 未量化時不顯示 localization pass。
- [ ] Weak head 不自動產生熱圖。
- [ ] 不出現「建議聚焦」「已找到 crack」文案。
- [ ] Inline 與 Page 共用 engine，沒有 preprocessing fork。
- [ ] 未被規格取代的 explainability E2E 全綠；被移除的 inline selectors 已有等價 Page/engine migration tests。
- [ ] Source dataset 與既有 caches 沒有非預期修改。

### 能標示 `single_sample_smoke_supported` 的 gate

- Context 與 provenance 相容。
- Sample-level OOF audit 可用且限制已揭露；這不等同 head supported。
- Target class support 足夠。
- Selected sample decision 可定義。
- Faithfulness smoke test 支持當次 margin，且只對這一張圖有效。
- 仍明示 grouped generalization 與 localization 尚未驗證。

### 能宣稱 `diagnostic_supported` 的 gate

至少需要：

- Grouped validation 或獨立 audit set 通過。
- Critical class recall 達產品門檻。
- Calibration 可接受。
- Batch faithfulness 顯著優於 random。
- 有 mask 時 localization 顯著優於 baseline。
- Dataset/model/head/recipe 完整可追溯。

即使通過，也只能說「支持分類決策解釋」；若要宣稱 crack segmentation，必須有獨立 segmentation model 與 mask evaluation。

## 26. 風險與控制

| 風險 | 後果 | 控制 |
|---|---|---|
| Page 與 inline 各自實作 preprocess | 同圖產生不同結果 | 共用 framework-free engine |
| 同輪改 Page、聚合、normalization、cache | 回歸來源不可判斷 | P0 先凍結契約與 tests |
| 高解析 probe 被視為高解析模型 | 錯誤產品結論 | Advanced-only、永久標 probe |
| Sample OOF 被視為 production validation | 過度高估泛化 | 明示 split，P2 grouped contract |
| Heatmap 成功後警示消失 | 使用者誤以為可信 | Reliability banner 持續存在 |
| Old result 配上 new controls | 無法重現或錯誤比較 | Deterministic key + stale guard |
| Opacity 0.95 蓋住原始 crack | 視覺上誤判 focus | 預設 0.45–0.60，opacity 只 render |
| Background job late result | 污染目前 sample | Generation/key check，P1 前景執行 |
| 只補近重複圖片 | Learning curve 虛假改善 | Group-aware split、duplicate audit |
| 單張 mask 看起來重疊 | 誤判定位有效 | Batch metrics + baseline + CI |

## 27. 與既有文件的關係

- 本文件延伸 [single-image-explainability-proposal.md](single-image-explainability-proposal.md)：保留 inline 單張檢視，但增加獨立診斷工作區與可靠性 gate。
- 本文件吸收並重新分期 [gradcam-fast-robust-design.md](gradcam-fast-robust-design.md)：selected-sample smoke test 在 P1，完整 robust/stability/faithfulness 在 P4。
- [gradcam-resolution.md](gradcam-resolution.md) 中 336／448 的描述只適用於 spatial probe。本文件明確取代任何可能讓人誤解「probe 等同已驗證 multi-resolution model」的措辭。
- 現有真實 UI 測試基線見 [e2e-mvtec-explainability-playwright-report.md](e2e-mvtec-explainability-playwright-report.md)，後續 P1–P4 必須維持既有能力並新增相應回歸證據。

## 28. 尚待產品決策

以下不阻擋文件完成，但進入實作前需逐項裁決：

1. P1 頂層工具短名稱使用「模型診斷」或「可解釋性診斷」。建議「模型診斷」。
2. P1 是否允許 advanced counterfactual target。建議允許，但預設收起並加橘色反事實警示。
3. P1 metadata JSON 是純下載還是可選擇寫入 `.lv_cache`。建議純下載，持久化延後。
4. P2 正式 `group_id` 從何種 metadata 取得；沒有可靠欄位時不得猜測。
5. P3 先做 224+336 還是一次 224+336+448。建議先建立可擴展 recipe registry，再依資源決定一次跑哪些解析度。
6. `diagnostic_supported` 的 critical class recall、calibration 與 CI 正式門檻，由產品風險與 audit set 校準。
7. 最終產品目標是解釋分類、發現未知 anomaly，還是輸出 crack mask／尺寸；這會決定 P5 選 Normal Bank、MIL 或 segmentation。

## 29. 後續實作切片

P0 contracts 與 P1 guided workspace 已完成；後續切片應沿著 evidence 邊界接續，而不是把未驗證的結果混入目前頁面：

```text
ExplainContextV1
+ ExplainRunSpecV1
+ canonical method names
+ signed color deterministic tests
+ stale matrix tests
+ existing inline adapter compatibility
```

1. 先完成 P2 grouped OOF／learning curve，回答資料量與 feature bottleneck 的可驗證問題。
2. 再把目前 session/on-demand 的 224／336／448 exact heads 提升為 P3 的可重現 benchmark（完整 embeddings、OOF、cache namespace 與 dataset-level table）。
3. 最後才接 P4 batch faithfulness、mask localization 與 shortcut diagnostics；若目標是 crack 邊界或尺寸，再依證據評估 patch classifier、MIL 或 segmentation。

每個 checkbox 只有在程式、必要測試與文件證據都完成後才勾選；目前頁面不得把單張研究熱圖升格成定位或泛化結論。
