# User × CV 工程師的 AL 協作工作流(角色與動線建議書)

> **2026-07-11 multi-agent 討論產出,尚未開發。**
> 本文件為多代理討論收斂後的建議,非開發指令;所有涉及 M9–M11(設計文件 08–13)的開工
> 仍須 User 放行、PO 記入 ROADMAP 決策日誌後走五階段流程。文中 file:line 均經實際 Read/Grep 複核。

## 本文件回答什麼

1. 一輪 Active Learning 在 LV + ANnoTation 兩套 GUI 上的**目標頁面級動線**該長什麼樣。
2. AL 迴圈中每個決策點的**權責歸屬(RACI)**——User(標註者/領域專家)、CV 工程師、系統各管什麼。
3. LV↔ANnoTation 介面上**必須顯性化的最小資訊集**,讓兩個角色對「一批做完了沒」不再各自解讀。
4. 已規劃的 M9–M11 六個模組(08–13)**各自解掉動線的哪一段**,以及哪些缺口不在其範圍、須另行立案。

優先序標記:**P0**(不做則迴圈斷裂)/ **P1**(做了權責才明確)/ **P2**(體驗與收斂)。
定位標記:`[重排]`=既有元件重排、`[M9-M11]`=已規劃規格涵蓋、`[立案]`=新需求須立案。

---

## 1. 目標 AL 迴圈動線(一輪的頁面級走法)

落點原則(主席裁決):**不新增第七個工具列項**,以「📥 標註回饋」分頁擴建為「🔁 標註迴圈」中樞
(工具正規化耦合見 app.py:8094-8099 區塊);挑樣 UI 原地保留在 Visualize 與瑕疵偵測,後端共用純函式。

### 常駐狀態列(迴圈中樞頂部)

`Round N|已送 X|已回 Y|已套用 Z|pool 未標 W` —— 依賴 08 pool_registry + 09 round_ledger;
「作用中回合常駐顯示」是 09 規格自承缺口(09:57-58 只講 session 記住最近 round),須補規格。**P0 [M9-M11+補規格]**

### ① 挑樣(CV 工程師;頁面:Visualize / 瑕疵偵測③)

- Visualize 選樣 panel:策略選單加「混合(不確定×多樣)」(M11 hybrid_sampler,11:41-48),
  修「top-K 不確定全是近重複」陷阱;弱類定向已有既有函式可接(app.py:1541-1542 已 import
  `confusion_targeted_priority`)。**P1 [M9-M11]**
- Object Detector 模式:側欄「模型預測資料夾」→「🧪 模型錯誤挖掘」四佇列 FN/FP/混淆/低信心
  (M10 gt_pred_diff,10:56-66)。**P1 [M9-M11]**
- 瑕疵偵測③ 2×2 取樣矩陣維持原 UI,後端與 Visualize 共用 M11 純函式;**不合併兩頁 UI**
  (M7 索引空間污染 + st.tabs rerun flaky 實證)。**P1 [重排+M9-M11]**
- 瑕疵③目前最該送標的頁**只進購物車無直送出口**(app.py:1968-1970)→ 補直送鈕,
  復用 `_send_to_labeling_ui`(app.py:7190),送原圖路徑非 crop、key 進 anomaly 命名空間。**P1 [重排]**

### ② 送標(CV 工程師按、系統攔截;頁面:8 個既有站點)

共用元件 `_send_to_labeling_ui`(app.py:7190)現有 8 個呼叫站點:viewer(:2324)、selection(:2406)、
duplicate(:2705)、diversity(:2783)、cart(:3283)、完整度缺口(:6329)、灰帶(:7641、:7790)。此步驟疊加:

- **防重複攔截**:送出前查 pool_registry,「仍包含已送標」須顯式勾選,攔截理由攤在 UI(08:57-66)。**P0 [M9-M11]**
- **回合戳記**:「記為新回合」checkbox,round_id 寫入 `_handoff.json.payload` 與 `_pending.json`(09:55-68)。**P0 [M9-M11]**
- **task 顯式宣告**:目前 verify/relabel 由排序選單隱式推導(app.py:2403-2404
  `_sel_task = LH.TASK_VERIFY if sort == "離群度" else LH.TASK_RELABEL`),連工程師自己都無感知
  → 改顯式 selectbox,預設值=現行推導結果(行為不變)。**P1 [重排]**
- **預標 seed 寫入**:dino_head/瑕疵分數與 M13 熱力圖框,以 ANnoTation module_016 相容的
  X-AnyLabeling sidecar 格式寫進 handoff `images/`,讓 User 從「改框」而非「畫框」開始
  ——現有磁碟契約下零新依賴的最短增益。**P1 [立案(小案)]**

### ③ 標註(User;頁面:ANnoTation module_026 → 012)

- module_026 已自動偵測 LV 批次並帶入路徑(ANnoTation modules/module_026/026_input.py:64-103),
  但 module_012 完全不呈現 `_handoff.json` 的 task/instructions/payload——User 收到的只是一個資料夾,
  adjudicate 與 fresh 長得一樣。→ module_012 頂部呈現 **round·task·選樣理由**(ANT 端零新依賴)。**P1 [立案(ANT 端)]**
- **annotator 歸屬**:sidecar 的 `annotator` 欄位 LV 已會讀(labeling_handoff.py:294)但無人寫
  → ANT 端由 claim_task 歸屬自動填入。**P1 [立案(ANT 端)]**

### ④ 回讀套用(CV 工程師;頁面:📥 標註回饋)

- 既有:批次清單+已標 X/Y(app.py:7264-7273)、「📥 套用讀回結果」→ `apply_readback`
  (app.py:7280-7281 → labeling_handoff.py:378-413)。
- **致命缺口:套用只改記憶體 records,重 Run 即蒸發**(apply_readback 無任何落盤)。
  → `.lv_cache` 標籤覆蓋層 + append-only 變更日誌(舊標/新標/annotator/時間戳),匯出可選「含修正」。
  此為全新需求,08–13 均未涵蓋。**P0 [立案]**
- 四態徽章、未標清單(而非僅計數)、sha 不匹配顯式報錯、結案鈕(接上目前**零呼叫**的
  `delete_pending`,labeling_handoff.py:97)。**P1 [重排+立案]**
- **框級讀回分段**:目前只取第一個 shape 的 label、幾何全丟(labeling_handoff.py:288
  `label = shapes[0].get("label")`)。分類字串迴圈先落地(P0 覆蓋層);shapes 幾何讀回配 M13
  預標框再做。**P1 [立案,配 M13]**
- ANT 端配套:module_014 匯出時把**全部** open 批次一次標 `read_back`
  (ANnoTation modules/module_014/014_process.py:63-67)須拆掉——「ANT 匯出完成」與「LV 已套用」
  是兩件事。**P0 [立案(ANT 端)]**

### ⑤ 評估與停止(CV 工程師看、品保簽核;頁面:📥「🔁 回合」子區)

- M12 probe_eval:「🧪 跑代理評估」按鈕、plotly 學習曲線、附數字理由的停止**建議**
  (明文禁止自動停,12:42-49)。既有 `should_stop_labeling`(active_loop.py:107)目前**沒接 GUI**。**P0 [M9-M11]**
- 回合時間軸卡+「➕ 掛指標」(09:55-68)。**P0 [M9-M11]**

### ⑥ 下一輪(CV 工程師;回到①)

漏抓類→一鍵弱類配額餵回 `confusion_targeted_priority`(app.py:1541 已接);
Compare 漂移表僅作回合參考指標,不做自動排程。**P1 [重排]**

---

## 2. RACI 對應表

主席裁決:「User」細分為三種真實角色——**複判員**(秒級判定,只進 ANT,不碰 LV)、
**製程/領域工程師**(defect 判準、正常樣本確認)、**品保/主管**(門檻簽核、預算與停止)。
「裁量權整包移交籠統 User」已駁回(落地必淪為外包背鍋)。
(R=執行 A=負責 C=諮詢 I=知情)

| 決策/動作 | 複判員 | 製程工程師 | 品保 | CV 工程師 | 系統 |
|---|---|---|---|---|---|
| 挑樣策略、送哪批 | I(可見理由) | C | I | **A/R** | 08 防重複攔截 |
| task 語義宣告 | I(顯性呈現) | I | — | **A**(顯式選,勿隱式推導) | 寫入 `_handoff.json` |
| 標註/改框執行 | **R** | C | — | I | 預標 seed(10/13) |
| few-shot 正常樣本確認 | C | **A**(經 verify 批覆核;流程約定,不寫死) | I | R | I |
| 爭議標籤裁決 | R | **A** | C(重大走 MRB) | R(決定送裁決、接受結果) | 記入 ledger |
| 門檻/何時停、再標多少 | — | C | **A** | R(供 probe 曲線) | 建議+禁自動停(12) |
| bank/模型重建、reset/排除 | I | I | 簽核 golden | **A/R** | append-only 記帳 |
| 讀回套用、批次結案 | I(看得到「已套用」) | I | I | **A/R** | exported/applied 分態 |

現狀對照(為何需要這張表):

- few-shot「確定正常」目前由操作 LV 的工程師在①進階 expander 框選(app.py:1728-1740),
  但「什麼算正常」是領域判斷,現無任何送 User 覆核的出口。**P1 [重排:改走 verify 批]**
- 唯一裁決管線 `TASK_ADJUDICATE` 只存活於**已隱藏**的灰帶覆核(app.py:7641、7790),入口死路;
  ANT 有完整 review 狀態機(ANnoTation domain/core/states.py:20-33 `transition_annotation_set` +
  `apply_review_decision`)但 LV 批次不進佇列。裁決入口掛回 📥。**P2 [重排,PO 先記決策日誌]**
- 停止準則、reset/mark_excluded 有 API 無 UI 歸屬(active_loop.py:107;08:57-66)——本表把
  「何時停」歸品保簽核、reset 歸 CV 工程師且 append-only 留痕。**P0 [M9-M11]**

---

## 3. LV↔ANnoTation 介面必須顯性化的最小資訊集

四個欄位,**兩端同步顯示**;全部是「已存在但不可見」的契約欄位,不需要新工具:

| # | 資訊 | 現狀 | 顯性化做法 | 優先/定位 |
|---|---|---|---|---|
| 1 | **round_id 輪次號** | 兩邊都答不出「第幾輪」;09 規格自承作用中回合不可見 | 寫入 `_handoff.json.payload` 與 `_pending.json`;LV 送標鈕旁常駐「目前回合」,📥 與 ANT module_026/015 都顯示 | **P0 [M9-M11+補規格]** |
| 2 | **批次來源+選樣理由** | `_handoff.json` 帶 task/instructions/payload(含分數、錨例)但 module_012 完全不呈現;User 全盲 | module_012 頂部呈現 source·task·一句理由(離群/漂移類別/FN 佇列)——ANT 端零新依賴的最短增益 | **P1 [立案(ANT 端)]** |
| 3 | **佇列狀態四態** | 三態有定義不顯示(labeling_handoff.py:53-55 `sent/annotating/read_back`);`read_back` 語義重載——module_014 匯出時全批標 read_back(014_process.py:63-67),LV 可能從未套用就被標「已讀回」 | 拆為 `sent → annotating → exported → applied`(**新增狀態值,不改名**,向後相容;LV/ANT/nativeApp vendored 三處同步);每批附**未標清單**而非僅計數(現算邏輯在 labeling_handoff.py:297-310 一帶);結案入口接 `delete_pending`(labeling_handoff.py:97,現零呼叫) | **P0 [立案(跨兩端)]** |
| 4 | **標註者身分 annotator** | LV 已讀(labeling_handoff.py:294)但無人寫;裁決與一致性無法歸因到人 | ANT 端 claim_task 歸屬自動填入 sidecar;LV 讀回明細與變更日誌顯示 | **P1 [立案(ANT 端)]** |

配套文案修正:「不用回 LV」與 📥 讀回自相矛盾——LV 端 8 處文案(如 app.py:7256-7257
「不用回 Labeling,也不會自動輪詢」)、ANT 端 module_026 亦說「不用回 VisualLatent」
(026_input.py:103)、module_014 收尾訊息同調(014_process.py:50)。統一為
「**送出 → ANT 標 → 回 LV 套用**」,同 PR 掃 tests/e2e 可見字串。**P0 [重排]**

---

## 4. 與 M9–M11 模組(設計文件 08–13)的動線對應

| 模組 | 解掉動線哪一段 | 對協作權責的貢獻 |
|---|---|---|
| **08 pool_registry** | ②送標的防重複攔截、④讀回後 `mark_labeled`、常駐 pool 計數 | 保護 User 時間(不重標)、CV 不再靠記憶;攔截理由攤在 UI=不靜默決策(08:6-8、57-66) |
| **09 round_ledger** | 常駐 Round N、②回合戳記、⑤回合時間軸+學習曲線資料底座 | 「這批憑什麼選、回來多少、值不值」可追溯;CV 能對 User/老闆回答「再標 200 張值不值」(09:6-9) |
| **10 gt_pred_diff** | ①的 FN/FP/混淆/低信心四佇列、②的預測框當預標 seed | User 改框而非畫框,直接降標註成本;錯誤加權排序=選樣理由可言說(10:6-9) |
| **11 hybrid_sampler** | ①挑樣策略(不確定×FPS 多樣),兩頁 UI 後端共用 | 近重複對 caption=向 User 交代「為何不是那張」(11:6-10) |
| **12 probe_eval** | ⑤代理評估、報酬遞減視圖、停止「建議」 | 「何時收手」有共同客觀參照;明文禁自動停=最終裁量留給人(12:6-9) |
| **13 heatmap_to_boxes** | ②瑕疵路線的預標框來源(直通 10 號管線,零特例) | 同 10:改框不畫框(13:8-10) |

### 六模組**沒蓋到**、須另行立案的缺口(本文件的立案清單)

| 缺口 | 動線斷點 | 優先/定位 |
|---|---|---|
| 讀回落地層+變更日誌(apply_readback 只改記憶體,labeling_handoff.py:378-413) | ④ | **P0 [立案]** |
| `read_back` 語義拆分 exported/applied + ANT module_014 全批收尾拆掉(014_process.py:63-67) | ④ | **P0 [立案(跨兩端)]** |
| module_012 呈現 round/task/理由;annotator 由 claim_task 寫入 | ③ | **P1 [立案(ANT 端)]** |
| LV 預測以 sidecar 預標寫入 handoff(閉環 module_016 格式) | ②→③ | **P1 [立案(小案)]** |
| 框級 shapes 幾何讀回(labeling_handoff.py:288 只取 `shapes[0]` 的 label) | ④,M10/M13 需要框級 GT | **P1 [立案,配 M13]** |
| handoff 升級為 ANT 一等任務(進 claim_task 佇列,而非資料夾旁路) | ③ 多標註者指派 | **P2 [立案;PRD Won't 明列多標註者仲裁,範圍須 PO 重議]** |
| few-shot 正常確認送 verify 批覆核的流程約定 | ①判準 | **P1 [重排(流程約定,不寫死)]** |
| 灰帶裁決入口掛回 📥、隱藏三工具處置 | ③裁決 | **P2 [重排;PO 先記 ROADMAP 決策日誌]** |

### 明確不做(主席裁決,避免重議)

- 不合併 Visualize/瑕疵③挑樣 UI(M7 索引污染+st.tabs flaky 實證);不新增第七工具列項。
- 不做自動停止、不做漂移自動排程(歸 SPC/FDC/MES)、不做標註 UI 內命名→ANT schema 自動同步(MRB 決策)。
- 緩做大規模更名(E2E 文字選擇器 8 處/10 檔),只統一稱謂。

---

## 附:優先序總覽

| 優先 | 項目 |
|---|---|
| **P0** | 讀回落地層+變更日誌 [立案];四態拆分+ANT 全批收尾拆掉 [立案];「不用回 LV」文案統一 [重排];08/09 依規格開工+round_id 常駐 [M9-M11,待 User 放行];12 接 GUI [M9-M11] |
| **P1** | task 顯式宣告 [重排];瑕疵③直送鈕 [重排];module_012 呈現理由+annotator 寫入 [立案 ANT];預標 sidecar [立案小案];shapes 幾何讀回 [立案,配 M13];10/11 挑樣 [M9-M11];正常樣本 verify 覆核 [重排] |
| **P2** | 裁決入口掛回 📥+隱藏工具處置 [重排,PO 決策];handoff 升 ANT 一等任務 [立案,範圍重議];名詞收斂 [重排] |
