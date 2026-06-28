# M5 — Active Learning Loop(主動學習標註迴圈)技術設計

## 動機(來自雙-split 完整測試)
M2 Normal Bank + M3 分類頭 + M4 取樣佇列是「組件」,目前**開迴圈**(一次性評分)。
完整測試的 S7 學習曲線實證:**uncertainty sampling 比隨機省 ~60-75% 標註**(valid 480 標註
active 0.76 vs random 0.60;test 0.61 vs 0.50,跨兩 held-out 都成立;幅度視不均衡/可分性而定)。
M5 工具:佇列選樣(弱類定向)→ 人工 confirm 回流 Normal Bank → 標註效益學習曲線(主動 vs 隨機,
回顧模擬展示省標註)→ 曲線走平就停。詳見文末「誠實界定」對「真閉環 vs 模擬」的劃分。

## 模組:`scripts/active_loop.py`(純邏輯,Tier A 偏;但被 GUI 依賴 → 升 Tier B,需 E2E)

### API 與 AC
- `entropy_score(head_proba) -> np.ndarray`
  - **AC1**:歸一化預測熵 ∈ [0,1];one-hot → 0、均勻分布 → 1;形狀 (N,)。
- `confusion_targeted_priority(anomaly_scores, head_proba, *, anomaly_threshold=None, w_novelty=1.0, w_entropy=1.0) -> np.ndarray`
  - **AC2**:= w_novelty·Novelty(離正常,minmax)+ w_entropy·Entropy(分類頭最混淆/弱類)。
    熵越高 → 優先越高(對齊「最該標分類頭最不確定者」)。
  - **AC3**:`w_entropy=0` 退化為純 novelty;`head_proba=None` 時熵項為 0。
- `label_efficiency_curve(pool_emb, pool_labels, eval_emb, eval_labels, *, strategy="active", seed_n=60, batch=60, rounds=8, seed=0) -> list`
  - **AC4**:模擬標註迴圈,每輪訓 head→在 eval 量 balanced_acc,回 `[[n_labels, acc], ...]`;
    n_labels 嚴格遞增、acc ∈ [0,1]、長度 ≤ rounds。strategy ∈ {active(uncertainty), random}。
  - **AC5**:在「可分但需挑樣本」的合成資料上,active 最終 acc ≥ random 最終 acc(不劣於隨機)。
- `should_stop_labeling(curve, *, patience=2, min_delta=0.01) -> bool`
  - **AC6**:連續 patience 輪 acc 提升 < min_delta(曲線走平)→ True;持續上升 → False。
- `round_summary(confirmed_labels) -> dict`
  - **AC7**:回每類已標數 + 總數(給迴圈面板);接受 list 或 dict(value 為 label)。
- `stratified_pool_eval_split(labels, *, eval_frac=0.3, seed=0) -> (pool_idx, eval_idx)`
  - **AC8**:分層切分,每類(樣本 ≥2)在 pool 與 eval 兩側各至少 1 個 → 防少數類全進一側使 eval 退化
    成單類(那會讓 balanced_accuracy 恆 1.0、學習曲線變假平圖)。不重不漏。
- `confusion_targeted_priority` 防呆
  - **AC9**:空 anomaly_scores → 回 shape (0,);scores 與 head_proba 列數不一致 → 明確 ValueError。
- `label_efficiency_curve` 單類 eval 韌性
  - **AC10**:eval 退化成單一類別時不洩漏 sklearn 警告(內部 catch_warnings)、回傳值仍合法(acc∈[0,1])。

> **誠實界定(名實相符)**:本迴圈對 **Normal Bank 是真閉環**(人工 confirm 的 good/bad 回流擴 bank,
> 走既有 confirm/rerun);**分類頭重訓走「訓練分類頭」按鈕**(在物件 label 上)。**學習曲線是回顧模擬**
> (以資料集既有 label 當 oracle,展示主動選樣省標註的價值),**不消費**人工 confirm 標籤 —— 面板已明示。

## GUI 整合(`scripts/app.py` 的瑕疵偵測工具)
「🔁 主動學習迴圈」expander(在分類頭 + 取樣佇列之後):
- 顯示**學習曲線**(active vs random,呼叫 label_efficiency_curve;池=現有物件、eval=同池held-out切分)。
- 顯示**本輪標註狀態**(round_summary)+ **停止建議**(should_stop_labeling)。
- 取樣佇列改用 **confusion_targeted_priority**(弱類定向)當一個可選優先模式。
- **誠實邊界**:label_efficiency_curve 用 ground-truth 當 oracle 模擬(GUI 內以現有 records 的 label
  當代理);真實人工標仍走既有 confirm UI。曲線是「決策輔助/效益展示」,非自動標註。

## 驗收
- 單元:`python verify/gate.py active_loop` 印 GREEN。
- E2E:`tests/e2e/test_active_loop_e2e.py`(真實 Streamlit + Playwright)斷言迴圈面板出現、
  學習曲線(active/random 兩條)真的畫出、停止建議文字存在。done = 單元綠 AND E2E 綠。

## 不做(誠實界定)
不做全自動標註(仍需人工)、不做線上增量學習(每輪 retrain 即可)、不碰使用者資料集
(快取進 .lv_cache)。
