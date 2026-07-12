# 04 半導體工業檢測真實情境 × LV 功能映射 × 缺口分析

> **本文件為 multi-agent 討論產出,尚未開發。** 內容是討論結論與建議,不是開發指令;
> 任何模組開工(尤其 M9–M11 / 設計文件 08–13)仍須 User 放行、PO 記入 ROADMAP 決策日誌後,依 CLAUDE.md 五階段流程進行。

## 本文件回答什麼

1. 半導體產線(AOI/ADC/SEM 檢測)有哪些**真實情境**能用上 LV 現有工具,每個情境現在能走到哪、卡在哪。
2. 每個情境裡 **active learning 如何進場**,以及 **User(複判員/製程工程師/品保)與 CV 工程師的權責分工**。
3. 哪些情境是**真痛點**、哪些是**簡報想像**(產線導入顧問的批判與主席裁決後的修正)。
4. 缺口清單的**優先序(P0/P1/P2)與定位**(既有元件重排/M9–M11 已規劃/新需求須立案)。
5. 主席裁決的 **Top 5 情境排序**與**產線導入最小三件事**。

---

## 一、九個產線情境逐一檢視

每個情境依「情境描述/現有哪頁能用/差在哪/AL 怎麼進場/User vs CV 分工」展開;
文末附產線導入顧問(反方二)對該情境「真痛點 vs 簡報想像」的判定。

### 情境 1:AOI 初篩後 ADC 分類複判 —— 【真痛點,Top 5 之首】

- **描述**:光學初篩(AOI)丟出大量 defect 候選,人工逐張分類複判(real defect / defect type)。
- **現有哪頁**:瑕疵偵測②「套用偵測」評分排序(`scripts/app.py:1743`)+ ③「挑樣送人工標」2×2 取樣矩陣(`scripts/app.py:1831`)。
- **差在哪**:標註讀回只套用 `viz_records`、不回流 anomaly 工具(`scripts/app.py:7275-7281`);且 `apply_readback` 只改記憶體、重 Run 白標蒸發(`scripts/labeling_handoff.py:378-413`,變更套用在 400-413 的 in-place mutation);異常分數無校準門檻。
- **AL 進場**:hybrid_sampler(設計 `3_Architect_Design/11_hybrid_sampler.md`)排出「最值得人看」的複判佇列,人力只花在資訊量最大的樣本。
- **分工**:複判員只看排序後的 top 批;CV 工程師建 bank、鎖模型、設抽樣率。
- **顧問判定**:真痛點。與情境 3、9 是**同一條 ADC 主線**,採購理由就是 overkill/underkill 率;confirmed 與讀回全活在 session,**三班制產線換班即失憶**,這是致命傷。

### 情境 2:極端不平衡(10⁵ die 出 1 缺陷)—— 【真痛點,Top 5 第 2】

- **描述**:rare-event mining;絕大多數影像正常,缺陷樣本極稀少。
- **現有哪頁**:Normal Bank one-class 建模正好對路(`scripts/anomaly_tool.py:106-116`:無外部 bank 時整批參考視為正常建 Normal Bank)。
- **差在哪**:無 pool_registry,同 die 影像跨輪重複送標無防呆;head 缺樣本時 2×2 矩陣鎖三格(`scripts/app.py:1888-1894`:`_HEAD_DEP` 三模式反灰),長期只能停在純 novelty 排序。
- **AL 進場**:novelty 排序即 rare-event mining;probe_eval(`3_Architect_Design/12_probe_eval.md`)量「再標下去還有沒有用」。
- **分工**:複判員標極少數真缺陷;CV 顧 oversample 與去重。
- **顧問判定**:真痛點。缺口=08 去重 + 2×2 鎖格時的 novelty 長跑支撐。

### 情境 3:Nuisance/過殺過濾(AOI 過殺常 >90%)—— 【真痛點,併入 ADC 主線】

- **描述**:AOI 候選中大量是 nuisance(過殺),需快速二分 real/nuisance 並讓系統學會。
- **現有哪頁**:②框選標正常+就地重評(`scripts/app.py:1227-1232`);gt_pred_diff 的 FP 佇列(設計 `3_Architect_Design/10_gt_pred_diff.md`)。
- **差在哪**:confirmed 只活在 session,nuisance 知識不持久回 bank/head——每次重開從零學。
- **AL 進場**:FP 佇列定向送標,讓分類頭學會 nuisance 類;回饋調閾值=overkill rate 治理。
- **分工**:複判員快速二分 real/nuisance;CV 用回饋調閾值、對品保報 overkill 率。
- **顧問判定**:真痛點(ADC 主線的一環)。

### 情境 4:NPI 新層冷啟動 —— 【簡報想像成分高,價值有限】

- **描述**:新產品/新層導入,無標註、無已知缺陷型錄。
- **現有哪頁**:Visualize 多樣性選樣 farthest-point(`scripts/app.py:2754`)+ 完整度覆蓋散點。
- **差在哪**:novelty 失效(全部都新);監督 UMAP 預設在零標籤池無意義;學習曲線(`scripts/active_loop.py:64` `label_efficiency_curve`)存在但未接 GUI。
- **AL 進場**:diversity-first 起跑 → 有標後切 hybrid;學習曲線判收斂。
- **分工**:製程專家命名新 defect mode;CV 維護 classes.txt(但無 schema 同步到 ANT)。
- **顧問判定**:**簡報想像**——NPI 期每張影像本來就全檢,diversity sampling 價值有限;新 mode 命名是 MRB 跨部門會議決定,不是標註 UI 裡「User 命名」。

### 情境 5:Recipe/製程漂移偵測(每 lot vs golden baseline)—— 【需求真、落點錯,降級為報表】

- **描述**:每 lot 影像分布 vs golden baseline,監控製程漂移。
- **現有哪頁**:Compare 逐類漂移表(`scripts/app.py:4862-4871`,ProgressColumn 顯示 A/B 形心餘弦距離)。
- **差在哪**:漂移數字無行動門檻;無跨輪趨勢(round_ledger 未做);A/B 無 baseline/pool 語義;需手動每 lot 跑。
- **AL 進場**:漂移最大類 → 定向補標 → 重建 bank。
- **分工**:User 不進此頁;CV 產出可匯出給製程工程師的報表。
- **顧問判定**:lot 級漂移監控歸 **SPC/FDC(MES/CIM 既有系統)**,離線 Streamlit 手動頁無排程、無 OCAP,不會被排進生產流程。主席裁決:**只做「可匯出報表」**,排程/OCAP 不做。

### 情境 6:多機台/多 chamber 成像差異 —— 【量測題,LV 只做配角】

- **描述**:同一 recipe 在不同機台/chamber 成像亮暗、對比不同,模型可轉移性存疑。
- **現有哪頁**:Compare 兩夾對比;屬性棋盤亮度/對比軸近似(`scripts/app.py:6916`)。
- **差在哪**:無 per-tool metadata 分組軸,>2 機台要 C(n,2) 次兩兩比。
- **AL 進場**:bank 跨資料夾投影(external_bank,`scripts/anomaly_tool.py:110-116`)可測可轉移性,每機台配額取樣。
- **分工**:User 無感;CV 決定「一 bank 通用 vs per-tool bank」。
- **顧問判定**:tool matching 本質是設備工程的量測題,LV 頂多做 per-tool bank。

### 情境 7:稀有缺陷模式發現與命名 —— 【縮小後保留,Top 5 第 4】

- **描述**:發現前所未見的 defect mode,聚攏同 mode 樣本、命名建檔。
- **現有哪頁**:Visualize 離群 top-N(`scripts/app.py:2438`)+ 以圖搜相似鏈式擴散(`scripts/app.py:2563`)撈同 mode 聚落。
- **差在哪**:命名後新類別不回寫 ANT schema(module_017 無同步);discovered mode 無登記處。
- **AL 進場**:離群 → 相似擴散 → 整簇送 fresh 標註 = mode 建檔。
- **分工**:製程工程師命名並定判準(其核心價值);CV 撈簇、建類。
- **顧問判定與裁決**:流程保留(離群→擴散→整簇送標);**命名/schema 同步歸 MRB 會後人工回填,不做自動同步**。

### 情境 8:Golden 樣本庫維護 —— 【顧問升格,Top 5 第 3】

- **描述**:Normal Bank 即 golden 樣本庫,需長期演化、可簽核、可追溯。
- **現有哪頁**:①一鍵存/載模型目錄(`scripts/app.py:1618-1704`)。
- **差在哪**:bank 無版本/血統/差異比對;「哪一輪加了哪些良品」不可追;讀回不回流 → 確認良品併回 bank 這條路是斷的。
- **AL 進場**:每輪把確認良品併回 bank = 庫演化。
- **分工**:品保簽核 golden;CV 凍結版本、跨層複用。
- **顧問判定**:**被低估**——半導體變更管理要求 bank 改版**可簽核**,不只「血統」。主席採納升格至 Top 5 第 3。

### 情境 9:降低複判負荷(auto-binning + 抽驗)—— 【真痛點,採購級問題】

- **描述**:自動分掉大宗(正常/已知類別),人只複判 Unknown 與低信心;對管理層報自動化率/漏檢率。
- **現有哪頁**:dino_head 閘控級聯三分流(正常/已知類/Unknown)。
- **差在哪**:無信心門檻校準 UI;`should_stop_labeling` 存在(`scripts/active_loop.py:107`)但 app.py 僅 import `confusion_targeted_priority`(`scripts/app.py:1541-1542`),整個 active_loop 其餘未接 GUI——「還要標多少」答不出;「自動率/漏檢率」報表不存在。
- **AL 進場**:人只看 Unknown+低信心;probe_eval 給停止建議。
- **分工**:複判員只複判 Unknown;品保拍板門檻與停止(標註預算);CV 供 probe 曲線並產報表。
- **顧問判定**:真痛點(ADC 主線)。「沒這張報表,廠內立項過不了」。

---

## 二、噪音 vs 需重新包裝(半導體語境)

**噪音(對產線近零價值)**:
- train/val 洩漏檢查與「僅跨 split」重複(`scripts/app.py:2705` 一帶)——產線資料軸是 lot/日期/機台,不是 split。
- 以文搜圖(chinese-clip,`scripts/app.py:2569`)對 SEM/AOI 灰階影像近零效。
- 組考卷的分類夾結構與幾何換皮不合 wafer 影像。
- 監督 UMAP 預設在無標籤池反而誤導(NPI 冷啟動)。

**重新包裝(改語彙即增值)**:
- 「桶①佔比」→「光學可偵測性」:對 AOI 極有價值但現名不可解。
- Compare → 「lot 漂移監控站」(僅報表,不做排程)。
- 離群度 → novelty score。
- 三 tab wizard 天然對應「recipe setup → production scoring → review sampling」,改用產線語彙。
- 隱藏的「評估」工具(逐型態 recall + 漏抓畫廊)正是 ADC 驗收報表雛形,應以 gt_pred_diff/probe_eval(M10/M12)形式回歸,而非重造。

---

## 三、缺口與建議清單(優先序 × 定位)

定位標記:【重排】=既有元件重排/文案;【M9–M11】=已規劃(設計 08–13 已寫,待 User 放行);【立案】=新需求須立案。

| # | 建議 | 優先 | 定位 | 依據 |
|---|------|------|------|------|
| R1 | **讀回落地 + append-only 變更日誌**:`apply_readback` 寫入 `.lv_cache` 標籤覆蓋層,記舊標/新標/annotator/時間戳,匯出可選「含修正」 | **P0** | 立案(08–13 均未涵蓋) | `scripts/labeling_handoff.py:378-413` 純記憶體;`scripts/app.py:7283-7292` 變更僅一次性 dataframe 顯示不落盤;`annotator` LV 會讀(`scripts/labeling_handoff.py:294`)但無人寫入 |
| R2 | **pool_registry + round_ledger 開工**:防重複送標、回合可追、換班可恢復;順手拆 ANT module_014 全批標 `read_back` 的語義重載 | **P0** | M9–M11(08/09 規格已備,**仍待 User 放行**) | `3_Architect_Design/08_pool_registry.md`、`09_round_ledger.md`;`C:\code\claude\ANnoTation\modules\module_014\014_process.py:63-67` |
| R3 | **可簽核的 overkill/underkill 報表**:probe_eval 接 GUI + per-class recall + 門檻校準,輸出 CSV | **P0** | M9–M11(12)+ GUI 接線 | `scripts/active_loop.py:107` 未接 GUI;`scripts/app.py:1541` 僅 import 一函式 |
| R4 | 「不用回 LV」文案統一為「送出→ANT 標→回 LV 套用」 | P0 | 重排 | `scripts/app.py:7257` 等 8 處 vs 📥 分頁實際流程矛盾 |
| R5 | 框級 shapes 幾何讀回(現只取第一 shape 的 label) | P1(**分段**:分類字串迴圈先行,幾何配 M13 預標框再做) | M9–M11(13)+ 立案 | `scripts/labeling_handoff.py:287-288` 只取 `shapes[0]`;ADC 大宗是預裁切 crop 分類複判,分類讀回足以跑通第一階段 |
| R6 | 瑕疵③補「直送標註」鈕(現只能進購物車),復用 `_send_to_labeling_ui`;送原圖路徑非 crop、key 進 anomaly 命名空間 | P1 | 重排 | `scripts/app.py:1968-1970` 僅購物車;`scripts/app.py:7190` |
| R7 | task 語義顯式化:selectbox 取代隱式推導,預設=現行推導值 | P1 | 重排 | `scripts/app.py:2403-2404` 依排序語境推導 verify/relabel |
| R8 | 四態(`sent→annotating→exported→applied`,新增狀態值不改名)/round_id/annotator/選樣理由兩端顯性化 | P1 | M9–M11(09)+ 小案(含 nativeApp vendored 三處同步) | `scripts/labeling_handoff.py:294` 已讀 annotator;ANT module_012 呈現端 |
| R9 | Visualize 選樣與瑕疵③後端共用 M11 純函式(**UI 不合併**) | P1 | M9–M11(11) | M7 索引空間污染 + st.tabs rerun flaky 實證 |
| R10 | **lot/wafer/die metadata schema**(klarf/機台 FTP 對接的前置) | P1(前置評估) | 立案(全新) | 三份提案共同盲區:全部假設影像已躺在資料夾;LV 無任何產線 metadata schema——先於任何 UI 重構立案評估 |
| R11 | Golden bank 版本/血統/差異比對/簽核 | P1–P2 | 立案 | `scripts/app.py:1618-1704` 僅存/載,無變更管理 |
| R12 | Compare 漂移表可匯出報表(排程/OCAP 不做,歸 SPC/FDC) | P2 | 重排 | `scripts/app.py:4862-4871` |
| R13 | 行話改產線語彙:只改 caption/help,widget label 緩(E2E 文字選擇器命中) | P2 | 重排 | 「桶①佔比」等 |
| R14 | 隱藏「評估」工具以 M10/M12 形式回歸(漏抓畫廊併入回合報表);灰帶裁決(`TASK_ADJUDICATE`)掛回 📥 | P2 | M9–M11 + PO 決策日誌先行 | `scripts/app.py:8094-8099` 隱藏工具正規化;`scripts/app.py:7642,7791` TASK_ADJUDICATE 僅存活於隱藏頁;`scripts/app.py:4596` `_render_cmp_panel` 零呼叫死碼 |

---

## 四、User × CV 工程師分工(主席裁決版)

籠統的「User」在真實組織站不住,**拆三人**(整包移交駁回——落地必淪為標註外包背鍋):

- **複判員**:秒級判 real/nuisance,只進 ANT,不碰 LV 六頁工具與購物車。
- **製程/領域工程師**:defect 判準、正常樣本確認(「什麼算正常」是製程 spec)、爭議裁決 A(重大走 MRB)。
- **品保/主管**:門檻簽核、標註預算與「何時停」拍板(CV 只供 probe 曲線)。

| 決策 | 複判員 | 製程工程師 | 品保 | CV 工程師 | 系統 |
|---|---|---|---|---|---|
| 挑樣/送批 | I | C | I | **A/R** | 08 防重複攔截 |
| task 宣告 | I | I | — | **A** | 寫入 handoff |
| 標註執行 | **R** | C | — | I | 預標 seed |
| 正常樣本/判準 | C | **A** | I | R | I |
| 爭議裁決 | R | **A** | C(重大走 MRB) | R(送裁決) | ledger 記錄 |
| 門檻/何時停 | — | C | **A** | R(供 probe 曲線) | 建議、禁自動停 |
| bank/重訓/reset | I | I | 簽核 golden | **A/R** | append-only |

**最小顯性化資訊集**(兩端同步):round_id(寫入 `_handoff.json`/`_pending.json`)、source·task·一句選樣理由(ANT module_012 呈現)、四態附未標清單、annotator(ANT claim_task 寫入,LV `scripts/labeling_handoff.py:294` 已讀)。

---

## 五、主席裁決:半導體場景 Top 5 排序

1. **ADC 複判主線**(情境 1+3+9 合併):排序複判 + nuisance 回饋 + auto-binning;缺口=讀回落地(R1)、confirmed 持久化、M12 報表(R3)。
2. **極端不平衡 rare-event**(情境 2):Normal Bank 對路;缺口=08 去重(R2)、2×2 鎖格時的 novelty 長跑支撐。
3. **Golden 樣本庫**(情境 8,顧問升格):缺口=bank 版本/血統/簽核(R11)——變更管理需求,需立案。
4. **稀有模式發現**(情境 7 縮小):離群→相似擴散→整簇送標保留;命名/schema 同步歸 MRB 會後人工回填。
5. **漂移監控**(情境 5 降級):只做可匯出報表(R12);排程/OCAP 歸 SPC/FDC。

**前置盲區**(顧問提出、主席採納):lot/wafer/die metadata schema(R10)為全新需求,**先於任何 UI 重構**立案評估。

## 六、產線導入最小三件事(反方二提案,主席背書)

1. **讀回落地 + 留痕**(R1):`.lv_cache` 標籤覆蓋層 + append-only 變更日誌——一步同時解「白標蒸發」與審計否決項,沒有它三班制產線不能用。
2. **pool_registry + round_ledger 開工**(R2,規格已備、待 User 放行):防重複送標、回合可追、開機即恢復——連續流與班別交接的採購級門檻。
3. **可簽核的 overkill/underkill 報表**(R3):probe_eval 接 GUI + per-class recall + 門檻校準 + CSV——沒這張報表,廠內立項過不了。

## 七、不做/緩做(與本主題相關者)

- **不做**:lot 漂移自動排程/OCAP(歸 MES/CIM);標註 UI 內命名→ANT schema 自動同步(mode 命名是 MRB 決策);挑樣 UI 元件合併(M7 索引污染+st.tabs flaky 實證);裁量權整包移交籠統「User」。
- **緩做**:框級 shapes 幾何讀回(分類迴圈先行,配 M13);widget label 更名(E2E 文字選擇器 8 處/10 檔);隱藏三工具結構性移除(待 PO 決策日誌 + M10/M12 落地)。

---

*追溯註記:本文所有 file:line 已於 2026-07-11 以 Read/Grep 對 repo 現狀複核(含 `scripts/labeling_handoff.py:287-288/378-413`、`scripts/app.py:1541/1743/1831/1888-1894/1968-1970/2403-2404/2754/4596/4862-4871/7190/7255-7292/8094-8099`、`scripts/active_loop.py:64/107`、`scripts/anomaly_tool.py:106-116`、ANT `014_process.py:63-67`)。設計文件 08–13 均已存在於 `3_Architect_Design/`,依 MEMORY「AL Loop 規格待放行」仍未開工。*
