# M6 — 統一主動學習畫面(修正版 Option 2)架構決策

> 來源:多 agent 架構選型討論 3 輪(發散 5 視角 → 批判 3 反方 → 收斂),wf5fi8595。

## 決策:修正版 Option 2(單畫面,非三工具)
對外只做**一個**「瑕疵偵測 / 主動學習」畫面(維持 segmented_control 第 8 項 `_anomaly_ui`,不拆三工具)。
砍掉字面版「依總筆數自動訓 classifier」。Option1(三獨立工具)**否決**:在 UX 暗示 Bank/Head 可替換,
違反架構文件「並行 + 閘控級聯」結構防線,且對高頻小批量使用者每天還切換稅。hybrid 是假選項
(底層模組已 streamlit-free、可獨立複用是現在就免費擁有的事實,不構成第三條路)。

## 核心原則
**路由只決定「要不要額外解鎖 head」,絕不決定「Bank 閘有沒有」—— Bank + gated_predict 永遠在。**
所有自動判斷在頂部**狀態列**顯式化、可一鍵手動覆寫。run 路由/解鎖判定下沉到 `anomaly_tool`(純函式可單測),app.py 只做呈現。

## 畫面四區塊(由上而下)
1. **輸入 + label 語義宣告**(新增,必答一次):選 YOLO 資料夾後,radio 宣告 label 語義 ——
   「物件類別(YOLO 原生:door/window…)」vs「瑕疵類別(刮傷/污漬…)」。**修 silent-wrong bug**:
   現 `_anomaly_train_head` 無條件把 `record['label']`(來自 interaction.py 的物件類名)當瑕疵類訓 head。
   宣告物件類別 → head 區塊整段不出現;宣告瑕疵類別 → 才允許 head 流程。
2. **Normal Bank 守門**(恆在、骨幹):永遠跑 run_pipeline 算異常分數 + 排序,高異常排前。
   冷啟走無監督 LOO;有 confirmed good 走 few-shot bank;掛 external_bank 走跨資料夾投影。
   狀態列:Phase0 冷啟全人工 / Phase1 對照 few-shot bank。
3. **瑕疵分類頭**(additive、顯式解鎖):僅當(語義=瑕疵類)AND(≥2 類)AND(每類 ≥ N_min)達標,
   狀態列才顯示「達訓練門檻 → [訓練分類頭]」按鈕(漸進揭露,按一下才訓,不偷切換)。
   訓完走 `dino_head.gated_predict` 級聯(Bank 異常分數先守門 → 離正常遠才信 head,沒把握標 Unknown)。
   未達門檻 → 列「哪些類樣本不足、暫不納入」,維持 Bank-only。
4. **主動學習取樣佇列**(恆在):`priority_score`(novelty 為主)+ `select_for_labeling` 排前送購物車。
   學習曲線(active_loop)降為最底層、預設收合 expander。

## 資料量路由規則
head 解鎖 = (label 語義=瑕疵類) AND (distinct 瑕疵類 ≥ 2) AND (每納入類別樣本 ≥ N_min)。
**用「最小類別樣本數」非總數/平均**(500 良品 + 某缺陷 3 張 → 仍不解鎖)。進階:只對達 N_min 的類別訓 head,
不足的類維持 Bank→Unknown 佇列(沿用 active_loop.stratified_pool_eval_split 慣例)。
N_min 預設 8(**未驗證假設**,須在 train/test/valid 三 split 做敏感度掃描定案)。
邊界:剛跨 N_min → 不自動訓、出現一顆按鈕;只良品/單類/語義=物件類 → head 不出現,純 Bank 新穎性。

## 模組對映(皆 streamlit-free、不需新拆)
狀態列+語義宣告=app.py 薄層 + 路由判定下沉 anomaly_tool(新純函式);Normal Bank=anomaly_tool.run_pipeline
→ anomaly_score + anomaly_classify + anomaly_coreset;存讀/投影=anomaly_bank_store + anomaly_project;
分類頭+閘控=dino_head;取樣佇列=active_learning;迴圈/曲線=active_loop;購物車/匯出=既有 _export_ui。

## 既有工具處置:就地重構(不重寫、不拆、不另起)
1. 補語義開關 + 修 silent-wrong(最高優先)。
2. 換路由 predicate(len≥2 → 語義+≥2類+每類≥N_min),判定上提狀態列 + 可覆寫,head 改漸進揭露。
3. 結構降債:run 路由/解鎖下沉 anomaly_tool,app.py 只呈現,狀態列取代隱性觸發。
不刪任何底層模組;不拆三並列工具。

## 未解問題(交第二場討論「頁面怎麼改」解決)
- N_min 實證值(預設 8 未驗)→ 三 split 敏感度掃描。
- 閘門分位數漂移(獨立議題):`_gthr=quantile(scores,1-contam)` 相對門檻,真實瑕疵率>contamination 會 escape;
  何時改用 confirmed good/bad 的絕對距離/校準門檻?
- 「Defect 量常不多」現實:head/two-gate/曲線分支可能長期空殼 → 是否把重心壓在「Bank 守未知 + AL 撈候選」?
- label 語義可能被亂填 → 是否加啟發式提示(同圖多個同類框 → 較像物件偵測)。
- E2E 真實行為斷言:(a) per-class 門檻邊界;(b) label 語義防呆;(c) 分位數恆等下的 escape;
  + 共享 .lv_cache/殘留 server flaky(見 visuallatent-e2e skill)隔離策略。
- 「Bank 閘恆在」是宣稱不變量非型別強制 → 須路由層測試鎖住達門檻分支不繞過 Bank 計算。
