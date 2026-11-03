# Model Diagnostics UI/UX 重新規畫

日期：2026-08-31  
狀態：UX proposal，尚未全部實作  
範圍：Visualize → Model Diagnostics → 分類 head → Signed Grad-CAM

## 1. 結論

使用者的主要意圖只有一個：

> 用選定解析度分析這張圖，並告訴我結果能不能信。

目前介面卻可能把這個意圖拆成多次操作：

1. 建立分類模型。
2. 確認研究用途。
3. 產生分類證據熱圖。

這三步是系統內部流程，不應成為三個同等重要的使用者決策。建議改成：

- 頁面只保留一個 primary CTA。
- 缺少對應解析度 head 時，同一次 CTA 依序完成 embeddings、head 與熱圖。
- CTA 在執行前明確說明會做什麼，以及是否需要較長時間。
- 弱 head 顯示限制提示，但不要求再確認一次。
- 解析度、target 或演算法改變後，只把結果標成待更新，不自動啟動昂貴運算。

推薦主流程：

```text
選取影像
  → 選解析度
  → 按一次「分析此圖」
  → 系統視需要準備 head
  → 產生熱圖
  → 顯示結果與可靠度
```

## 2. 現況的 UI/UX 問題

### 2.1 同一意圖被拆成多個按鈕

「建立模型」、「我知道這是研究用途」與「產生熱圖」都只是完成同一件事的中間步驟。使用者不知道：

- 哪一顆才是主要按鈕。
- 為什麼前一顆成功後還要再按一次。
- 第二次點擊會不會重新運算。
- 研究用途確認到底改變了什麼。

研究限制是結果語意，不是需要使用者承擔的授權流程。它應該持續顯示，但不應阻斷操作。

### 2.2 224、336、448 的操作規則不一致

目前 224 可能需要按「建立模型」，336／448 則可能在選取解析度後直接開始建立。這會造成：

- 相同元件在不同選項下有不同副作用。
- 使用者只是比較選項，卻意外啟動昂貴運算。
- 很難回答「我按下的是選擇，還是執行」。

解析度 selector 應只改設定，不應直接執行。真正的運算一律由同一顆 primary CTA 啟動。

### 2.3 技術狀態與使用者任務混在同一層

Context ID、head digest、OOF fold、mask、Normal Bank、compute key 都有價值，但不是使用者第一眼要回答的問題。第一層只需要：

- 現在分析哪張圖。
- 選了哪個解析度。
- 接下來按哪裡。
- 模型判斷是什麼。
- 這個結果是否可靠。

其餘內容應收進「可靠度與技術詳情」。

### 2.4 警告看起來像操作元件

大面積紅色按鈕會被理解為主要動作，即使它只是研究限制確認。限制訊息應使用 warning/caption/badge，不應使用按鈕外觀。

建議語句：

> 分類模型目前證據不足；熱圖可用來檢查模型的判斷依據，但不能證明找到 crack。

### 2.5 Visualize inline 與 Model Diagnostics 職責重疊

兩邊都放 head 訓練、解析度、target、Grad-CAM 與顯示控制，會造成：

- 同一功能有兩套狀態與文案。
- 一邊修正後，另一邊仍保留舊流程。
- 使用者不知道應留在 Visualize 還是前往 Model Diagnostics。

建議 Visualize 只負責選圖與交接；完整分析集中在 Model Diagnostics。

## 3. 建議的資訊架構

主畫面只保留四個區塊：

1. 分析對象。
2. 分析設定。
3. 唯一主動作。
4. 結果與可信度。

```text
模型診斷

[縮圖] crack | 001.png
真實標籤：crack                  [更換圖片]

分析精細度
(•) 224 / 16×16   ( ) 336 / 24×24   ( ) 448 / 32×32
448：細節較多，首次使用需建立對應模型，預估約 N 分鐘

[ 建立 448 模型並分析此圖 ]

執行時：
1/3 產生 448 embeddings
2/3 訓練與驗證 448 head
3/3 產生 Signed Grad-CAM

分析結果
[原圖] [模型輸入] [藍—白—紅熱圖]

模型判斷：poke 27.1%
真實標籤：crack
OOF：crack 21.9%

⚠ 分類模型證據不足；這是分類決策證據，不是 crack mask。

[可靠度與技術詳情 ▸]
[進階研究設定 ▸]
```

## 4. 單一 Primary CTA

頁面在任何時間最多只能有一個 primary CTA。按鈕文字必須根據狀態說清楚會發生什麼。

| 狀態 | Primary CTA | 點擊後行為 |
|---|---|---|
| 沒有選圖 | 無 | 引導回 Visualize 選圖 |
| 解析度 head 不存在 | `建立 448 模型並分析此圖` | embeddings → head → heatmap，一次完成 |
| head 已存在、尚無結果 | `分析此圖` | 直接產生 heatmap |
| 設定未變、已有結果 | `重新分析` | 明確重新計算 |
| 解析度或 target 已改變 | `使用新設定重新分析` | 產生新結果；舊結果標為 previous |
| 正在執行 | disabled：`正在準備模型…` 或 `正在分析…` | 顯示階段與進度 |
| 執行失敗 | `重試分析` | 沿用相同設定重試 |
| Context stale | 無 | `回到 Visualize 重新選取` 作為導覽動作 |

不再出現：

- `我知道這是研究用途，仍要查看熱圖`
- head 建立成功後還要再按一次熱圖按鈕
- 選 336／448 就立即開始運算

## 5. 一次點擊的運算生命週期

同一次 primary action 可以包含多個內部階段，但必須被呈現為一個任務：

```text
queued
  → preparing_embeddings（需要時）
  → training_head（需要時）
  → validating_head
  → computing_attribution
  → rendering
  → complete / failed / stale
```

UX 規則：

- 執行前顯示預估工作量；448 不應假裝與 224 一樣快。
- 執行中只顯示一個進度區，不要在頁面不同位置同時出現 spinner、status 與 toast。
- head 建立成功後自動繼續產生熱圖，不要求第二次點擊。
- 若 head 已存在，跳過前面階段。
- 任一階段失敗時保留已完成資產；重試不應無條件重做 embeddings。
- 使用者切換圖片或設定時，晚完成的舊工作不可覆蓋目前結果。
- opacity、patch grid、色盤與圖片大小只重畫，不重新跑模型。

## 6. 解析度選擇

解析度 selector 應使用容易比較的卡片或 segmented control，而不是只顯示技術字串的下拉選單。

| 選項 | 顯示名稱 | 輔助說明 |
|---|---|---|
| 224 | 快速 · 16×16 patch | 適合先看整體分類證據 |
| 336 | 平衡 · 24×24 patch | 更多空間細節，需對應 head |
| 448 | 精細 · 32×32 patch | 最多空間取樣，成本最高 |

必要文案：

> 更高解析度只增加空間取樣，不保證模型更準，也不保證熱圖會聚焦 crack。

若對應 head 已存在，顯示 `已準備`；不存在則顯示 `首次分析時建立`。不應在使用者選取選項時就開始建模。

## 7. 結果區的優先順序

結果出現後，順序固定為：

1. 原圖、模型輸入、Signed overlay。
2. 藍／白／紅圖例。
3. 模型目前判斷、真實標籤、OOF 判斷。
4. 一句結論。
5. 可展開的詳細證據。

一句結論應由證據狀態生成：

| 證據狀態 | 建議結論 |
|---|---|
| head 尚未驗證 | `分類模型尚未完成可靠度驗證；熱圖只能作探索。` |
| head 很弱 | `模型對此類別辨識不足；熱圖是在解釋可能錯誤的分類決策。` |
| head 可用、無 mask | `分類證據可檢視，但尚未驗證 defect localization。` |
| attribution 不穩定 | `本次熱圖對擾動不穩定，不建議據此調整模型。` |
| localization 已量化支持 | `在目前資料與 mask 評估下，定位證據獲得支持。` |

避免使用：

- `建議聚焦`
- `模型找到 crack`
- `紅色就是瑕疵`
- 把分類機率寫成已校準 confidence

## 8. Warning 的設計

Warning 是狀態，不是 gate。

規則：

- 弱 head warning 在分析前後都可見。
- 同一限制在主畫面只出現一次。
- 詳細限制放入 expander，不重複堆疊三個 warning box。
- 不用整頁寬紅色按鈕呈現警告。
- warning 不改變按鈕數量。

視覺層級建議：

- 紅色：執行被阻擋，例如 context incompatible。
- 黃色：結果可看但可靠度不足。
- 藍色：一般說明或尚未開始。
- 綠色：技術工作完成；不等於模型可靠或定位正確。

## 9. Visualize 與 Model Diagnostics 的分工

### Visualize 保留

- 選取圖片。
- 顯示圖片與 label。
- 顯示最近一次診斷摘要（若存在）。
- 一個 CTA：`在模型診斷分析此圖`。

### Visualize 移除或降為相容模式

- head 訓練。
- 224／336／448 selector。
- Grad-CAM 方法 selector。
- target/reference selector。
- 完整 triptych 與可靠度表。

### Model Diagnostics 擁有

- 解析度與 head recipe。
- 單一分析 CTA。
- Signed Grad-CAM。
- prediction／OOF／reliability。
- target/reference 與研究方法。
- result history、stale 與 provenance。

這樣可以避免兩套 UI 在不同時間保留不同按鈕或狀態。

## 10. Advanced 區域

以下內容預設收起：

- 反事實 target。
- target/reference 定義。
- canonical 與 experimental attribution method。
- target-positive display。
- opacity、patch grid。
- Context ID、head digest、compute key。
- fold-level OOF 與完整 class metrics。

使用者展開 Advanced 並不啟動運算；改動運算設定後，primary CTA 更新為 `使用新設定重新分析`。

## 11. 必要功能與延後功能

### 現階段必要

- 單張影像識別。
- 224／336／448 選擇。
- 單一 primary CTA。
- 對應解析度 head 的準備與重用。
- Signed Grad-CAM triptych 與固定圖例。
- prediction、OOF、head reliability。
- weak／stale／failed 狀態。
- render-only controls 不重算。

### 可以延後

- Learning curve UI。
- 多解析度完整 benchmark leaderboard。
- batch faithfulness。
- mask localization metrics。
- shortcut audit。
- MIL／patch classifier／segmentation。
- 正式 artifact history 與跨 session report。

延後項目不應先以 disabled 控制或空白 panel 佔據主畫面。

## 12. 驗收標準

### 核心流程

- [x] 從 Model Diagnostics 選定解析度後，畫面只有一個 primary CTA。
- [x] 缺 head 時按一次 CTA，可一路完成 head 與 heatmap。
- [x] head 已存在時按一次 CTA，直接完成 heatmap。
- [x] 不存在研究用途確認按鈕。
- [x] 選取 224／336／448 本身不啟動運算。
- [x] 336／448 不沿用 224 head。

### 結果與語意

- [x] Signed legend 永遠是藍＝降低、白＝中性、紅＝提高 target-reference margin。
- [x] weak head warning 在結果前後保持可見，但不增加點擊次數。
- [x] 綠色完成狀態不會被描述成 localization supported。
- [x] 沒有 mask metrics 時不宣稱找到或分割 crack。

### 狀態與效能

- [x] opacity、grid 與顯示大小不觸發模型 forward。
- [x] 重用相同 recipe 的 embeddings/head，不重做已完成階段。
- [x] 設定改變後舊結果標為 previous/stale，不冒充目前結果。
- [ ] late result 不覆蓋新的圖片或設定。
- [x] 來源資料與既有 cache 不因單張診斷被改寫。

### E2E

- [ ] 224 缺 head：一次點擊完成準備與熱圖。
- [ ] 336 缺 head：一次點擊完成 exact head 與熱圖。
- [ ] 448 缺 head：一次點擊完成 exact head 與熱圖。
- [ ] 已有 head：一次點擊完成熱圖。
- [ ] weak head：只有 warning 與單一 CTA，沒有確認按鈕。
- [ ] 切換 render control 後 compute key 不變。

## 13. 建議實作順序

### Phase A：先解決操作斷裂

1. 建立單一 `diagnostics_run_requested` action。
2. 把「準備 head → 產生熱圖」串成同一 orchestration pipeline。
3. 移除所有研究用途 acknowledgement state、session key 與相容分支。
4. 解析度 selector 改成純設定，不再自動建模。
5. 加入 224／336／448 的一次點擊 E2E。

### Phase B：整理資訊層級

1. 主畫面改成「樣本 → 設定 → CTA → 結果」。
2. provenance、OOF details、target 與方法移入 expander。
3. 統一 warning，只保留一份主要限制。
4. 將 Visualize inline 降為 handoff 與摘要。

### Phase C：補強長任務體驗

1. 顯示 embeddings／head／heatmap 三階段進度。
2. 顯示快取命中與預估成本。
3. 支援安全重試與取消。
4. 保存 previous result，避免設定切換後整頁突然空白。

## 14. 與目前 OpenSpec 的差異

目前 OpenSpec 允許使用者選取非 baseline 解析度時立即啟動 foreground head build。這與本提案的「selector 無副作用、統一由 primary CTA 執行」不同。

若採用本提案，實作前應修改 OpenSpec：

- 非 baseline 解析度 selection 不再視為執行命令。
- primary CTA 是唯一昂貴運算入口。
- 缺 head 時，同一次 run action 負責 exact-resolution head 與 attribution。
- weak-head warning 永遠不形成第二次確認。

## 15. 最終產品原則

> 一個使用者意圖，只需要一個主要動作。

模型內部需要三個階段，不代表使用者要按三次。頁面應讓使用者清楚知道系統正在做什麼、結果能不能信，以及下一步是什麼；不應要求使用者理解 session head、research authorization 或 compute lifecycle 才能完成一次分析。
