# 半導體分析與 Active Learning 下一階段功能盤點

整理日期：2026-07-19（同日合併第二份獨立盤點，見文末「2026-07-19 合併」節；
原 root `SEMICON_AL_FEATURE_GAPS_2026-07-19.md` 已降為指標，本檔為唯一正典）  
範圍：依目前 repository、ROADMAP、既有產品檢視與已完成模組重新盤點。  
狀態：只做產品分析與優先序建議，不批准開發、不修改產品功能。

## 結論

VisualLatent 現在已經不只是 embedding 展示工具。它已有資料體檢、異常與相似搜尋、批次選樣、送標、read-back、回合帳本、代理評估、群組差異分析與最小 metadata CSV 關聯。下一階段最需要的，不是再增加一個視覺化頁面或再加一種 acquisition score，而是補上四種「決策可信度」：

1. **半導體實體可信**：知道 lot、wafer、die、unit、view、object 之間的關係，而不是只把 CSV 欄位掛在圖片上。
2. **評估可信**：避免同一 wafer 洩漏，也避免只拿 Active Learning 挑中的 hard case 評估自己。
3. **風險可信**：把 critical defect、escape、標註成本與產線 domain 當成明確限制，不只是一個可任意加權的總分。
4. **營運可信**：Unknown、taxonomy、normal bank、threshold、drift 與策略升級都有狀態、版本、簽核和回退路徑。

最重要的新缺口是：

> **group／time-aware 資料切分、selection-bias 控制、每個 domain 的信心校準與拒判、風險約束選樣，以及 Unknown／taxonomy 的完整生命週期。**

這些能力會決定分析結果能不能拿去做半導體驗收，而不只是能不能產生一張好看的圖。

## 現有基線：不要重做

| 現有能力 | 已有基礎 | 本次不再視為從零缺口 |
|---|---|---|
| Active Learning 閉環 | `pool_registry`、`round_ledger`、`hybrid_sampler`、`probe_eval`、`readback_store`；送標→讀回→評估已閉合 | 不重造第二套 pool、round 或 read-back |
| Dataset Audit | M20 `meta_join`、`audit_report`、`audit_pipeline`；已有壞檔、完全／近重複、跨 split 洩漏、同內容異標、類別覆蓋、離群、HTML／JSON／CSV | 下一步是半導體語義與統計可信度，不是再做一張相同報告 |
| 最小 metadata 關聯 | CSV 以 sha256 優先、檔名次之掛到影像，能列匹配率並依任意欄位分組計數 | 已有通用 join；缺的是 typed schema、階層、驗證與跨功能消費 |
| 異常與 Unknown | Normal Bank、異常分數、closed-set head、`gated_predict` 的正常／已知／Unknown 分流 | 不重造基本 Unknown 分數；要補的是 Unknown 後續處置 |
| 信心校準核心 | `dino_head.fit_temperature` 已有 temperature scaling 純邏輯 | 不重寫溫度縮放；目前缺產品接線、per-domain 評估與 threshold 治理 |
| 相似搜尋 | 單／多參考 cosine search、sample bank、retrieval export 已有 | 不重做向量搜尋；補 case context 與工程處置資訊 |
| 群組差異探索 | M19 可找 Good／Bad 的穩定局部差異，含 permutation null 防 false-green | 不把顯著差異誤稱為 root cause；補製程事件關聯與驗證設計 |
| 標註一致性分析 | quiz、Cohen／Fleiss kappa、consensus／gray set 已有 | 不重造 kappa；補 taxonomy 版本與正式裁決流程 |

## 功能缺口總表

| 功能缺口 | 現況 | 為何半導體／AL 需要 | 建議優先度 |
|---|---|---|---|
| Typed 半導體 entity model | metadata 只是逐影像任意欄位 dict | lot→wafer→die／unit→view→object 的階層與唯一性不清，無法做可靠聚合、選樣與追溯 | **P0** |
| Group／time-aware split 與 leakage audit | 目前依資料夾 split，洩漏主要看跨 split 影像近重複 | 同 wafer 鄰近 die、同 unit 多視角或同 lot 時序資料會讓 validation 虛高，即使圖片不重複 | **P0** |
| Selection-bias 控制與隨機哨兵 | 會記錄選樣策略與理由，但沒有 selection probability／獨立隨機 audit stream | AL 挑的是 hard case；只用被挑中的資料評估，不能代表 production distribution | **P0** |
| Risk／cost-constrained acquisition | `risk_weights` 是未接線掛鉤；現有 hybrid sampler 以 uncertainty、diversity、class quota 為主 | critical defect 不能被平均效益稀釋；不同 task、view、expert 的標註成本也不同 | **P0** |
| Per-domain calibration、abstention 與 threshold governance | 有 temperature scaling 核心，但未接 GUI／round；無 ECE、Brier、risk-coverage | 同一 confidence 在不同 tool／recipe 上可能意義不同；需要知道何時自動判、何時拒判 | **P0.5** |
| 多策略 replay、shadow、promotion／rollback | 有 active vs random 的兩策略模擬與 probe plateau，但不是可插拔策略評測產品 | 新策略不應直接替換正式 queue；需固定資料、budget、seed 與 guardrail 比較 | **P0.5** |
| Unknown discovery lifecycle | 能分流 Unknown、找離群與相似群，但沒有正式後續狀態機 | Unknown 必須被合併、拆分、命名、歸入既有類、判 nuisance 或繼續監控 | **P1** |
| Taxonomy／instruction migration | 有 consensus 與變更日誌，但無類別 merge／split／deprecated mapping | taxonomy 改名或拆類後，歷史標註、bank、模型與報告不能失去可比性 | **P1** |
| Golden／normal bank change control | bank 有版本與相容性檢查，但缺 add／remove 提案、污染檢查、簽核與 rollback 報告 | 一筆錯誤正常樣本可能系統性壓低異常分數 | **P1** |
| Drift 分解與製程事件關聯 | Compare 能看 A／B 分布，metadata audit 能分組計數 | 需區分成像漂移、class-prior drift、coverage drift 與 concept drift，並關聯保養／recipe 變更 | **P1** |
| Wafer spatial／multi-view unit 分析 | 文件有需求，程式無正式 die 座標／unit-view 模型 | 空間 pattern、缺 view、跨視角 disagreement 與 unit-level escape 無法只靠獨立圖片分析 | **P1** |
| 影像量測品質與 acquisition health | Audit 目前能找壞檔，但不量 focus、曝光、飽和、照明、配準與視野漂移 | 很多 embedding drift 其實是量測系統變化，不是製程或 defect 變化 | **P1** |
| Rare-event 統計與放行證據 | 有 per-class recall、bootstrap 評測素材與 probe，但產品報告未統一最小樣本與區間規則 | 零漏抓不等於安全；少樣本下 point estimate 很容易誤導 | **P1** |
| Expert routing 與 queue capacity | 已有 task／reason 與 handoff，但缺按技能、成本、期限與 backlog 的派工策略 | classification、geometry、taxonomy 裁決與製程判讀需要不同專家 | P2 |
| Similar Case 的事件／處置層 | 能找相似影像，缺 maintenance、root cause、disposition 與後續效果 | 工程師需要知道「過去怎麼處理」，不只是「哪張最像」 | P2 |

## P0：先讓資料與評估可信

### 1. Typed 半導體 entity model

M20 的 CSV join 是正確的最小起點，但它把每列視為平面欄位，尚未定義：

```text
product
└─ lot
   └─ wafer / panel
      └─ die / unit
         └─ view / inspection event
            └─ image
               └─ ROI / object / patch
```

下一階段需要的不是直接接 MES，而是建立可驗證的 sidecar schema profile：

- 必填與選填欄位、型別、允許值與 schema version。
- `lot_id + wafer_id + die_x + die_y`、`unit_id + view_id` 等 scenario-specific 唯一鍵。
- 同一 unit 的多視角、重拍、不同 station 與不同時間點如何表示。
- 座標範圍、時間順序、tool／chamber／recipe 組合是否合法。
- metadata 缺值、衝突、孤兒影像與一對多匹配的明確錯誤報告。

第一版仍可只讀 CSV／JSONL，不必做資料庫或 MES connector。

### 2. Group／time-aware split 與 leakage audit

目前跨 split leakage 主要回答「相同或近似影像有沒有跨資料夾」。半導體還需要回答：

- 同一 lot／wafer／unit 是否同時出現在 train 與 validation。
- 相鄰 die 或同一視野切出的多個 ROI 是否被拆到兩側。
- 同一缺陷的不同倍率、不同 view、重拍影像是否跨 split。
- validation 是否真的晚於 training，還是未來資料洩漏到過去。
- 各 tool／recipe／product 是否在 holdout 中有足夠覆蓋。

建議至少支援三種 split policy：

1. **Group holdout**：以 lot、wafer 或 unit 整組切分。
2. **Time holdout**：以時間窗做 past→future 驗證。
3. **Leave-one-domain-out**：保留一個 tool／recipe 做外部泛化驗證。

這應同時進 Dataset Audit 和 probe／promotion gate，不能只做一次性資料切分工具。

### 3. Selection-bias 控制

Active Learning 的標註資料不是隨機樣本，而是被策略刻意挑出的 hard case。如果 round report 只看這些資料，會發生兩種錯誤：

- 新策略挑到更難的資料，表面 accuracy 下降，卻可能其實更有價值。
- 策略只挑容易改善的區域，表面曲線上升，卻沒有改善 production risk。

最小可信設計應包含：

- 每輪固定保留一小部分**隨機哨兵樣本**，不受 acquisition score 影響。
- 固定且不可被訓練消耗的 production-like audit set。
- 記錄候選池版本、每筆分數、排名、排除原因、策略參數與是否來自 random quota。
- round report 分開呈現 selected-set、random-sentinel、fixed-audit-set 指標。
- 若未來使用機率式選樣，再記錄 inclusion probability，支援偏差修正；確定性 Top-K 不應假裝能直接做 inverse-propensity correction。

### 4. Risk／cost-constrained acquisition

目前 `risk_weights` 只有底層參數，正式選樣流程沒有資料來源與擁有者。建議不要只改成另一個加權總分，而應支援「先守限制，再最佳化效益」：

```text
先滿足：
- critical class 最低配額
- 各 tool／recipe／lot 的 coverage floor
- random audit quota
- 單一 wafer／相似群上限
- expert queue 容量

再最佳化：
- uncertainty
- novelty
- diversity
- 預期標註效益
- 標註時間／成本
```

Business Risk 必須由 Scenario／Quality Owner 定義並版本化；CV 工程師只負責把規則轉成可稽核的 sampling policy。

## P0.5：讓模型信心與策略升級可信

### 5. Per-domain calibration 與可拒判曲線

現有 temperature scaling 是好地基，但還缺完整產品證據：

- calibration set 必須與 training set 分離，並記錄 dataset／taxonomy／model version。
- 顯示 reliability diagram、ECE、Brier score，而不只顯示 accuracy。
- 依 tool／recipe／product 報 calibration；樣本不足時明講不可判。
- 顯示 risk-coverage curve：自動處理 30%、50%、80% 資料時，underkill／overkill 如何變化。
- threshold 與 min-confidence 必須版本化，可比較、可 rollback。
- 新 domain 先預設 abstain／送人工，不沿用舊 domain confidence 當成已校準。

Conformal prediction 可以作後續研究選項，但不是第一步；先把 calibration set、domain 分層與 risk-coverage 做對。

### 6. Acquisition Strategy Promotion Gate

應把現有 active-vs-random 模擬擴充成固定契約：

- 相同 pool snapshot、oracle label、audit set、budget 與 random seed。
- Random、confidence、novelty、diversity、hybrid、risk-constrained 同場比較。
- 每 100／500／1,000 個 label 比較 critical recall、duplicate rate、domain coverage、unknown discovery 與專家時間。
- Offline replay 通過後進 shadow queue；比較兩策略候選重疊、獨有案例與漏掉的高風險群。
- promotion 必須有 guardrail，並保留舊策略與 rollback 條件。

策略報告必須揭露 replay 的 oracle 假設；歷史標註池不能代表從未被看到的真正未知缺陷。

## P1：補齊半導體營運生命週期

### 7. Unknown discovery lifecycle

Unknown 不是終點標籤。建議狀態至少包含：

```text
new
→ clustered
→ needs-domain-review
→ existing-class / new-class / nuisance / acquisition-artifact
→ taxonomy-approved / monitor / rejected
```

每個群組應記錄代表樣本、群內一致性、跨 lot／tool 的 recurrence、首次／最近出現時間、專家裁決與臨時 containment。新類建立前需先排除成像失焦、照明改變與重複 nuisance。

### 8. Taxonomy 與 instruction migration

需要把 taxonomy 當成版本化資產，而不是只有 `classes.txt`：

- 類別定義、正反例、排除條件與 instruction version。
- rename、merge、split、deprecated 與 historical mapping。
- 舊 dataset／bank／model 使用哪一版 taxonomy。
- prelabel 接受率與人工分歧是否因 instruction 版本改變。
- 裁決前後的 label change，不能靜默覆蓋歷史。

另外應保留少量**盲標 audit**，不要讓所有樣本都先顯示 model prelabel；否則容易量到 annotator 對模型的服從，而不是獨立判斷品質。

### 9. Golden／normal bank change control

現有 bank 相容性與保存機制不足以防污染。建議補：

- add／remove proposal 與理由。
- 與上一版的 domain coverage、score distribution 與 Top-K retrieval 差異。
- critical／canary set 上的回歸檢查。
- Process Owner 確認正常、QA 簽核、CV 發布的責任分離。
- 可立即 rollback 到上一個 bank／threshold 組合。
- 高漂移 domain 不自動吸收進 normal bank，先進 quarantine。

### 10. Drift 分解與製程事件關聯

不要把所有 embedding distance 都叫製程漂移。至少區分：

- **Image acquisition drift**：focus、照明、sensor、magnification、alignment 改變。
- **Covariate drift**：影像外觀分布改變，但 defect 定義未變。
- **Class-prior drift**：各 defect／nuisance 比率改變。
- **Concept drift**：同一外觀與標籤／風險的關係改變。
- **Coverage drift**：新 tool／recipe／product 沒有被 training／validation 覆蓋。

報告應能對齊 maintenance、recipe change、material lot 與 inspection setup 事件，提供 before／after 窗口和候選關聯；必須明講「相關不等於 root cause」，最終因果判定仍由製程工程師負責。

### 11. Wafer spatial 與 multi-view unit 分析

半導體樣本通常不是獨立圖片。建議未來支援：

- wafer map 上的 edge／ring／scratch line／local cluster pattern。
- 空間相鄰 die 的相關性與抽樣上限，避免一次送標同一事件的數十張近鄰。
- `unit_id` 下預期 view 清單、缺 view、重拍與跨視角 disagreement。
- unit-level prediction／escape，而不只 image-level 指標。
- image embedding 與座標 pattern 分開呈現，避免把空間事件硬塞進 DINO 特徵。

### 12. Image acquisition health／量測系統分析

Dataset Audit 除了壞檔，還應能選配檢查：

- blur／focus、曝光、飽和、亮度與對比範圍。
- 視野位置、倍率、方向、crop／registration 偏移。
- dead pixel、條紋、壓縮與邊框等 sensor／pipeline artifact。
- 同一 golden sample 在不同 tool／班別／時間的 repeatability。

這個模組的目的不是判 defect，而是先回答「量測系統是否穩定」。否則 AL 會把取像問題當成高價值 Unknown，一直浪費專家時間。

### 13. Rare-event 統計與放行證據

對 critical defect，報告至少需要：

- 每類 support、recall／underkill 的 confidence interval。
- 零漏抓時仍報樣本量與可支持的風險上限，不寫成「100% 安全」。
- lot／wafer cluster correlation 下的有效樣本數說明。
- critical-class recall floor、overkill ceiling 與 review-capacity ceiling。
- 新舊模型在相同 audit set 的 paired comparison。

`probe_eval` 仍只能當方向性代理，不應成為 production release 的唯一證據。

## P2：提升工程協作與 ROI

### 14. Expert routing 與 queue capacity

不同任務應送不同角色：一般 real／nuisance 複判、geometry 修正、taxonomy 裁決、製程 root-cause 判讀不能混在同一 queue。未來可加入 estimated minutes、required expertise、due time、backlog age 與 escalation reason，讓選樣不超過專家實際容量。

### 15. Similar Case 的事件與處置層

相似搜尋結果應能選配顯示：

- tool／recipe／layer／material／maintenance 時間。
- 當時的 root-cause hypothesis、final disposition、containment 與改善後結果。
- 相似的是影像、製程條件，還是兩者皆相似。

ROI 應量 case review time、找到可重用處置的比例與 root-cause lead time；不能只量 cosine precision。

## 建議的實作順序

```text
Typed entity schema
→ group/time-aware split 與 audit
→ random sentinel + fixed audit set
→ risk/cost/domain-constrained sampling
→ per-domain calibration + risk-coverage
→ multi-strategy replay/shadow/promotion
→ Unknown/taxonomy/bank lifecycle
→ drift event、wafer spatial、multi-view 與 acquisition health
```

這個順序的理由是：前四項先決定「資料與評估是否可信」，之後才有資格比較哪個策略更好；再往後才是營運規模化與進階半導體分析。

## 建議的下一個產品切片

若只能選一個切片，建議不是新做一頁 wafer map，而是：

### Lot／wafer-aware ADC 評估與選樣

```text
匯入影像 + typed metadata sidecar
→ 驗證 lot/wafer/die/unit/view 身分
→ 以 wafer 為 group 建 train/audit split
→ 每輪保留 random sentinel
→ 依 critical risk + domain floor + uncertainty/diversity 選樣
→ ANnoTation 複判／裁決
→ 分別報 selected、sentinel、fixed audit 指標
→ 依 tool/recipe 顯示 calibration 與 risk-coverage
→ 人工決定繼續、promotion 或 rollback
```

它能一次驗證：metadata 是否真有用、評估是否被 AL 選樣偏差污染、critical risk 是否受到保護，以及專家時間是否真的下降。

## 現在不建議做

- 不直接接 MES／FDC／OCAP 或自動停機；先把 sidecar contract 與離線證據做對。
- 不只因 embedding drift 就宣稱找到 root cause。
- 不先做複雜 end-to-end backbone fine-tuning；現有資料切分與 audit 若不可信，模型比較也不可信。
- 不把所有平台治理一次搬進單機工具；Governance Board、RBAC、多人 SLA 等到真的跨團隊共用再立案。
- 不把 conformal prediction、因果推論或持續學習當成第一個答案；它們都依賴更基本的 group split、校準集與版本資料。
- 不用 overall accuracy 或被 AL 挑中的樣本表現作為唯一放行依據。

## 開發前需要 User／PO／Domain Owner 回答

1. 第一個正式 entity 是 image、ROI、die、unit，還是多視角 unit？
2. 哪個欄位可以可靠形成 group holdout：lot、wafer、unit，或 inspection event？
3. production-like fixed audit set 由誰建立、多久更新、誰可解封？
4. critical defect、escape cost、overkill cost 與 review capacity 的最低規則是什麼？
5. 每輪願意保留多少 random sentinel，接受多少短期標註效率損失來換取無偏估計？
6. 哪些 tool／recipe 必須各自校準，最少樣本不足時如何保守退回人工？
7. Unknown 的正式裁決者是誰；何時能建立新 taxonomy class？
8. normal／golden bank、threshold 與 acquisition strategy 的 promotion／rollback 由誰簽核？

## 2026-07-19 合併:第二份獨立盤點的補充

> 同日另有一份獨立完成的盤點(原 root `SEMICON_AL_FEATURE_GAPS_2026-07-19.md`,工作台
> 視角)。兩份對主要缺口**獨立收斂**(wafer 空間、per-domain 差異、時間軸/drift、
> promotion gate、Similar Case 處置層、風險受限選樣、以及「不做清單」幾乎一致)——
> 缺口是真的。本節收錄該份的獨有內容;其中它漏掉、本文抓到的(group/time-aware split、
> selection bias/隨機哨兵、量測系統健康、盲標 audit)以本文正文為準。

### A. 補充缺口:群組批次標註(cluster-level confirm)——標註效率的最大剩餘槓桿

現有迴圈把「選哪些」做得很好(混合選樣/FPS 去重/佇列),但**確認動作仍是逐張點**。
實務正解是「這 40 顆是同一群,整組標 A,我抽 3 顆確認+剔除例外」——一次動作標一群,
標註量再降一個量級(M1 的 2-stage 分群確認就是這個思想,但只活在建模階段、沒進 AL 佇列)。

- 最小版:佇列加「依相似度成組顯示」→ 組卡片(代表圖+成員數)→「整組標 X」+
  點開剔除例外;**帳本記 group_confirm 來源與抽查率**(誠實紀律:批次標註要留痕,
  評估時才能區分逐張 vs 批次品質——與本文 §8 的盲標 audit 同一精神)。
- 驗證:合成同質群+植入 2 顆異類 → 例外剔除流程能揪出;帳本可算批次標註佔比。
- 建議位階:**P0.5**(它與 selection-bias 控制互補:sentinel 保評估無偏,
  批次標註降成本——一省一保)。

### B. 早期切片補充:兩個低成本 quick win(可與 P0 並行,不衝突)

本文的 P0(typed entity/group split/sentinel)是對的根本順序;但有兩個
**工程量極小、直接復用 M19/M20 現成件**的切片,可在 P0 進行中先交付價值:

1. **Tool/Recipe 差異顯著性檢定**:M20 分組現在只有描述性計數;加一節
   「分組率 + permutation p 值 + 未達顯著明講」(M19 檢定框架直接複用,
   多欄位用 max-統計量控多重比較)。回答「B 機台瑕疵率高,是真的還是雜訊?」。
   驗證:同分佈分組判無差異;植入 tool=B 瑕疵率 ×3 → 判顯著且方向正確。
2. **成本敏感工作點面板**:輸入 escape:overkill 相對成本 → 在既有分數分佈上
   畫期望成本曲線、建議門檻(**只建議不自動套**)。它是本文 §5 risk-coverage 的
   最小前哨,也是 Business Risk 治理的第一塊可觸摸拼圖。
   驗證:合成已知分佈 → 建議點 == 解析解;成本 1:1 退化為現行對稱行為。

### C. 補充最小版設計(對應本文 P1/P2 項)

| 本文項 | 補充的單機最小版 |
|---|---|
| §11 wafer spatial | metadata CSV 帶 die_x/die_y → 每 wafer 散點(色=類別/分數/分群)+ **空間聚集度 permutation 檢定**(隨機打散座標當 null,聚集不顯著就明講——M19 同款誠實手法)。驗證:植入 edge-ring/scratch pattern → 聚集顯著判定正確。**前提待使用者確認:資料是否有座標欄位。** |
| §15 Similar Case 處置層 | 策展日誌條目升級為「案例」(附結案筆記+根因標籤);以樣搜樣結果旁顯示「命中歷史案例→看筆記」。純 `.lv_cache` JSON。驗證:建案例→海掃同型樣本連回;刪案例不再連。 |
| 預算規劃器(小) | M5/probe 曲線外推:「輸入 500 張預算 → 預估 recall 增益區間+誠實免責」。 |

### D. 驗證設計原則(此份補充的方法論貢獻)

每個新功能在提案時就定義**植入式客觀驗收**:合成資料植入已知答案(已知數量的
問題/已知位置的差異/已知比例的分組率),驗收=工具輸出逐項等於植入值;
配 null 對照(同分佈/隨機打散)證明不硬擠結論。M19(植入 Top-1+null 誠實)與
M20(六訊號植入數字全對)已示範此模式,**後續所有缺口項沿用**——
無法設計植入驗收的功能,先不做。

### E. 合併後的排序共識

1. 本文 P0(typed entity → group/time split → sentinel/audit set → 風險受限選樣)
   為主軸——「資料與評估可信」先於一切新功能。
2. B 節兩個 quick win(Tool/Recipe 顯著性、成本工作點)可**並行先行**(小、無衝突、
   直接可交付價值)。
3. 群組批次標註(A 節)排 P0.5,與 calibration/promotion gate 同層。
4. wafer spatial 維持 P1,**但先向使用者確認座標欄位是否存在**——有,它的價值
   會顯著前移;沒有,則此項不成立。

## 附錄:兩份盤點的差異分析與裁決(2026-07-19)

> 本檔正文(決策可信度視角)與被併入的第二份盤點(工作台視角)對缺口清單**高度收斂**、
> 且本輪**雙方零事實錯誤**(抽驗:`quiz.py:156/173` 確有 Cohen/Fleiss kappa、
> `dino_head.py:54` 確有 `fit_temperature`——本檔的「已有基線」宣稱屬實)。
> 真正的差異在世界觀與四個具體點,記錄如下供後續裁決引用。

### 世界觀差異(所有分歧的根源)

| | 本檔正文的鏡頭 | 第二份的鏡頭 |
|---|---|---|
| 出發點 | 決策可信度:資料與評估先經得起放行審查 | 工作台實用:單人工程師,每輪可開工可驗收 |
| 優化目標 | 估計不被污染(sentinel/group split/校準) | 人力投報比(quick win/批次標註/工程量) |
| 系統性盲點 | 低估操作者人力(漏:群組批次標註) | 漏方法論根本問題(selection bias/group split) |

兩鏡頭盲點互補——雙盤點的價值即在此;後續重大規劃建議維持雙視角互檢。

### 互相補上的(已併,不再爭)

- 正文抓到、對方漏:group/time-aware split 洩漏、selection bias/隨機哨兵、
  量測系統健康、盲標 audit。
- 對方抓到、正文漏:群組批次標註(P0.5)、植入式驗證設計原則(無法設計植入驗收
  =還不能開工)、Tool/Recipe 顯著性與成本工作點兩個 quick win。

### 保留的兩個真分歧(裁決記錄)

1. **Wafer map 時機**:正文排 P1 且明言「第一切片不該是 wafer map」;對方排第 1
   (附座標前提)。**裁決=折衷**:正文反對的其實是「拿 wafer map 取代信任層當
   第一優先」——這點正確;但**探索性視圖不需要信任層**(不做放行宣稱,只需座標
   欄位正確,M20 匹配率明講已保證),wafer 級**指標與裁決**才需要 P0。
   結論:視圖可早做、裁決晚做,兩者不衝突;前提仍是使用者確認座標欄位存在。
2. **切片大小**:正文的「Lot/wafer-aware ADC」一片橫跨五個能力,依本 repo appetite
   紀律(每輪 ≤3-4 模組)實作時**必然拆 2-3 輪**;方向正確、切片過大。
   另:正文未談 **sentinel 的成本面**——單機小標註預算下,哨兵配額有真實代價,
   最小版應為比例制(如 10%),而非一步到位的完整 audit set 治理。

## 本次判斷依據

- `ROADMAP.md`：M15–M18、M19、M20 的完成狀態與已驗證範圍。
- `scripts/meta_join.py`、`audit_pipeline.py`、`audit_report.py`：目前 metadata join 與 Dataset Audit 的實際邊界。
- `scripts/hybrid_sampler.py`、`active_learning.py`、`al_batch.py`：現有選樣、配額、risk hook 與大資料批次能力。
- `scripts/dino_head.py`、`probe_eval.py`：temperature scaling、Unknown gate 與代理評估的現有能力及限制。
- `scripts/pool_registry.py`、`round_ledger.py`、`readback_store.py`：現有 AL 身分、回合與讀回證據鏈。
- `scripts/quiz.py`、`evaluation.py`：一致性、共識子集與評估基礎。
- `03_半導體工業檢測情境.md`、`08_平台概念差距分析.md`：既有半導體情境與平台層缺口裁決。

