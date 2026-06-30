# ROADMAP(由 PO 維護)

> 狀態圖例:⬜ 待設計 · 📝 設計中 · 🧪 驗收定義中 · 🔨 開發中 · ✅ 完成
> 提醒:狀態是人的判斷,不可由「檔案存在」自動標 ✅。模組表/測試數請定期用
> `python verify/unet_status.py` 核對是否漂移。

> **🏁 Feature 狀態(2026-06-30):瑕疵偵測(AnomalyDINO 風格物件級瑕疵偵測)已宣告收斂。**
> M1–M7 涵蓋 PRD 全部 Must/Should/Could,並在真實瑕疵資料(`C:\code\dataset\fruit`)端到端驗證;
> 無待辦 user_needs。後續只走**維護迴圈**(只動 `scripts/app.py` + E2E、不受 appetite 約束);
> 要再加新能力須從新的 `/user` 需求重新起輪。未蓋棺尾巴(M6 `N_min=8` 三 split 敏感度掃描)列維護候選、非阻擋。
>
> **(2026-06-30 晚續)** 使用者提新需求「整張影像級(無 YOLO 標籤)」→ 依「新能力起新輪」規矩**重新起輪 = M8(進行中)**;
> 維護收斂結論不變,M8 是受 appetite 約束的新能力增量。

## 里程碑
- **M8 — 整張影像級瑕疵偵測(無 YOLO 標籤)** — 📝 **進行中(PRD→設計)**(2026-06-30) — feature 收斂後第一個
  新能力增量(走精簡 U-Net 新輪)。需求:手上只有「無標註圖片資料夾」的人,要把**整張影像當對象**做異常偵測、不必先標框
  (現況無 `labels/` 直接跳「找不到 YOLO 物件」=完全不能用)。範圍(MoSCoW):**Must** 新「物件來源=整張影像」路徑
  (每張可讀圖→一筆**全幅記錄**,欄位對齊 `discover_yolo_objects`、下游 embedding/分群/bank/patch/評分/散點/排序/看圖/匯出
  **不變**)+ `run_pipeline` 依 `object_source` 選用 + GUI ① 切換「YOLO 物件 / 整張影像」+ 模式標示 + 鎖進 `model.meta`;
  **Should** 偵測 0 物件時提示一鍵切整張影像;**Could** 自訂標籤名;**Won't** 整張內多物件分評 / 分類頭 / 背景遮罩。
  模組:**07 `whole_image_source`(Tier B)**。需求 [1_user_needs/anomaly_whole_image.md](1_user_needs/anomaly_whole_image.md);
  PRD [2_PO_PRD/anomaly_whole_image_prd.md](2_PO_PRD/anomaly_whole_image_prd.md)。appetite ≤1 模組 + GUI 接線,一輪做完。
  狀態:PRD 完成,放行 `/architect`。
- **M7 — 瑕疵偵測頁面重設計成引導式 wizard** — ✅ **完成**(2026-06-29~30) — 依使用者 7 項 UX 回饋 +
  多 agent 兩場設計鎖定(架構+頁面)→ `_anomaly_ui` 整段重寫,**引擎模組公開函式幾乎零改**(僅
  anomaly_bank_store.save_bank meta 多帶 `label_semantic`、bootstrap_cluster mcs 夾 ≤N、run_pipeline patch
  全參考 fallback)。結構:第 8 工具改**三步 wizard**(① 建模/載入 → ② 套用偵測 → ③ 挑樣送人工標)、
  **分兩資料夾**(訓練/參考 vs 異常目標)、**唯一模型槽 `anomaly_model`**(①寫②③讀,schema=meta+bank_vectors+
  projection+fewshot+head+label_semantic)、統一模型暫存目錄(一鍵存/載 bank+classifier+manifest)、label 語義
  白話 radio、**③ 2×2 取樣矩陣 master-detail**(點格出大圖牆,欄數/高度可調)。移除 M5 學習曲線 GUI 面板
  (active_loop 函式保留)+ 拆進階 popover。**步驟列最終用 `st.segmented_control`(非 `st.tabs`)**:tabs 的
  active tab 是純前端狀態、任何整頁 rerun 都彈回 Step-1 → 改 keyed segmented_control 跨 rerun 保留(8443933)。
  設計 [3_Architect_Design/M7_wizard_redesign.md](3_Architect_Design/M7_wizard_redesign.md)。驗證:全 gate 綠 +
  全套 anomaly E2E 14 綠(滿載偶有 1 既有 flaky,單跑綠)。多 agent 對抗測試揪 4 個真 bug,皆已修(見決策日誌)。
- **M6 — 統一主動學習畫面(修正版 Option2)** — ✅ **完成**(2026-06-29) — 多 agent 兩場討論(架構選型
  3 輪 + 頁面改造 3 輪)達共識:單畫面、Normal Bank+閘控**恆在**當骨幹,分類頭改成「label 語義=瑕疵類
  AND ≥2 類各達 N_min」才解鎖的 additive 第二段。路由判定下沉 `anomaly_tool` 純函式(per_class_counts /
  head_unlock_state / gate_phase / label_semantic_hint / gate_threshold)。**修 silent-wrong**:把 YOLO 物件
  類別當瑕疵類訓 head(加 label 語義宣告)。app.py 加成熟度狀態列 + 進階 popover + 閘門良品校準。
  設計 [3_Architect_Design/M6_unified_al_screen.md](3_Architect_Design/M6_unified_al_screen.md)。驗證:
  gate anomaly_tool 23 + 全 anomaly gate/E2E 綠;真實 indoor 雙 split S8 路由(object 語義不解鎖=修復坐實、
  bank 恆在)+ S9 閘門校準(test 瑕疵率15.6%>contam 時 escape 87%→57%)。**未定**:N_min=8 待三 split 敏感度掃描。
- **M5 — Active Learning Loop(主動學習標註迴圈)** — ✅ **完成**(2026-06-28) — 把 M2 bank + M3 head +
  M4 佇列串成主動學習工具:佇列選樣(弱類定向)→ 人工 confirm 回流擴 Normal Bank(真閉環)→ 標註效益
  學習曲線(主動 vs 隨機,**回顧模擬**展示省標註,不消費 confirm;分類頭重訓走訓練按鈕)→ 曲線走平就停。
  模組 `active_loop`(entropy_score / confusion_targeted_priority 弱類定向 / label_efficiency_curve 學習
  曲線 / should_stop_labeling 停止準則 / round_summary);GUI「🔁 主動學習迴圈」面板(active vs random 學習
  曲線 + 停止建議)+ 取樣佇列加「弱類定向」模式。實證:雙-split S7 主動選樣省 ~60-75% 標註。設計:
  [3_Architect_Design/M5_active_loop.md](3_Architect_Design/M5_active_loop.md)。驗證:gate active_loop 7 綠 + E2E 1 綠。
- **M4 — 主動學習取樣(Active Learning Priority Score)** — ✅ **完成**(2026-06-28) — 對應架構文件第九節:
  挑「最值得送人工標註」的物件。Priority = **Novelty**(Normal Bank 異常)+ **近決策邊界** + **Model
  Disagreement**(異常高但分類沒把握=Unknown)× Production Risk;選樣加 **Cluster Diversity**(每群代表)。
  距離訊號比 softmax entropy 可靠 → novelty 必納入。模組 `active_learning`;GUI 結果區「🎯 主動學習取樣佇列」。
  驗證:gate 7 綠 + E2E 1 綠。
- **M3 — DINO Classification Head + 閘控級聯** — ✅ **完成**(2026-06-28) — 對應使用者「DINO 工業瑕疵落地架構」
  的 Classification Head + Gated Cascade。frozen DINOv2 + linear head(closed-set 已知瑕疵分類);Normal Bank
  異常分數**先守門**(正常 / 已知類別 / Unknown),從結構上防未知瑕疵被閉集 head 自信誤分。模組 `dino_head`;
  GUI 結果區「🏷 瑕疵分類頭」。驗證:gate 6 綠 + E2E 1 綠。設計:[DINO_HEAD_DESIGN_NOTES.md](DINO_HEAD_DESIGN_NOTES.md)。
- **M2 — Memory Bank 持久化 + 跨資料夾投影** — ✅ **完成**(2026-06-28) — 經多 agent 三輪設計討論 →
  五模組 TDD 落地。目標:第一次建 bank 存硬碟(含 few-shot)→ 第二次選新資料夾讀回 bank 評分 + 把新資料
  投影回舊分佈(灰底=舊、彩色=新)。設計/實作對照:[ANOMALY_BANK_DESIGN_NOTES.md](ANOMALY_BANK_DESIGN_NOTES.md);
  討論記錄:[ANOMALY_BANK_DISCUSSION_RAW.md](ANOMALY_BANK_DISCUSSION_RAW.md)。
  模組:`anomaly_coreset`(greedy,A)、`anomaly_project`(投影器,A→B)、`anomaly_bank_store`(序列化,B)、
  `run_pipeline` external_bank/ref 注入(B)、`_anomaly_ui` GUI(B)。驗證:單元 gate 35 綠 + E2E 8/8 +
  真實 DINOv2 端到端鏈 ALL-OK。
- **M1 — AnomalyDINO 物件級瑕疵偵測(第 8 工具)** — ✅ **完成**(2026-06-23) — appetite:**本輪 ≤6 模組,此 feature 一輪做完**
  - 目標:載入 YOLO 資料夾 → patch 級 DINOv2 異常分數 → 抽樣分群 + 使用者確認(2-stage)或直接整批(1-stage)
    → 全集分好/壞 + 排序 + 散點圖框選/購物車/匯出 +(Should)熱力圖。
  - PRD:[2_PO_PRD/anomaly_defect_detection_prd.md](2_PO_PRD/anomaly_defect_detection_prd.md)
  - 成功門檻:帶標籤驗證集 image-level AUROC ≥ 0.85(待 architect 在 AC 給定可量化值);確認動作 ≤ 數十次;
    換正常認定重跑數秒內(特徵快取);快取進 `.lv_cache/` 不汙染資料集。

## 模組進度
| # | Module | Tier | MoSCoW | 狀態 | 設計 | 對應表 | 實作(scripts/) | 相依 | 備註 |
|---|--------|------|--------|------|------|--------|------|------|------|
| 01 | patch_features | B | Must | ✅ | v | v | v | 既有 discover_yolo_objects + 模型 | 單元 gate 綠;真實 DINOv2 路徑經 E2E 實證 |
| 02 | anomaly_score | A | Must | ✅ | v | v | v | 吃 patch 特徵 | gate 綠(6 測);精確/近似混合 NN |
| 03 | bootstrap_cluster | A | Must | ✅ | v | v | v | 吃物件 embedding | gate 綠(6 測) |
| 04 | anomaly_classify | A | Must | ✅ | v | v | v | 吃分數+少量確認標籤 | gate 綠(6 測) |
| 05 | anomaly_heatmap | A | Should | ✅ | v | v | v | 吃 patch 分數圖+裁切圖 | gate 綠(5 測);零依賴 jet |
| 06 | anomaly_tool | B | Must | ✅ | v | v | v | 01–05 + 既有 viz/cart/export | 單元綠;GUI 第8工具接好;真實 E2E 4/4 綠 |

相依無環:06→{01..05};02/03/04/05 只吃陣列、彼此不 import。✅
進度:**M1 完成** — 6 模組單元 gate 全綠 + 第 8 工具 GUI 接線 + 真實 Playwright E2E 4/4 綠
(done 定義「整合模組=單元綠 AND E2E 綠」已滿足)。E2E 由 multi-agent workflow 跑+對抗驗證:
真實 DINOv2 確實把缺陷排到前段(real_pass,非 false-green)。

## 決策日誌(append-only)
> 每次反向閘門退回、模組增刪、需求被砍,追加一行:日期 / 從哪層退到哪層(或 決策) / 一句為什麼。
- (2026-06-22) 框架初始化(規格疊加模式):實作進 scripts/、測試進 tests/、E2E 重用既有 harness;
  不寫會撞壞既有測試的根 conftest。原因:這是成熟 GUI repo,非 greenfield。
- (2026-06-22) 需求定案:壞=同物件的好/壞(A);patch 級為主、可切物件級(Could);要熱力圖;
  抽樣→分群→確認→整批(2-stage)或直接整批(1-stage)由 User 選;規模幾千~幾萬物件。
- (2026-06-22) PO 拆解:6 模組(4 純 Tier A + 2 整合 Tier B)。Won't:瑕疵種類分類、多類別大雜燴、即時。
  Should:heatmap(同輪盡量)。Could:物件/patch 切換、stage-2 主動學習。
- (2026-06-22) 反向閘門 /pm →/architect:PM 在落 06 驗收時發現「正常集=確認good∪正常密群」對細微瑕疵會
  污染 memory bank → 假綠。架構修為:有確認good時 bank 僅用 confirmed-good(乾淨,few-shot);純 1-stage 才用
  正常密群並標明「細微瑕疵會漏」的上限。新增 06 AC4 反向驗證(污染 bank→分數下降)。
- (2026-06-22) 反向閘門 /pg →/architect:PG 發現 bootstrap_cluster 候選門檻(frac 公式)對「少數但成群」的缺陷
  失效(5 壞群剛好≥min_cluster_size → 不標候選 → 1-stage 漏抓)。設計修:候選門檻加入 `0.5*最大群` 項
  (少數群一律候選,與規模無關)。誠實上限:1-stage 會把「少數良品類」也誤標候選,靠 2-stage 確認校正。
- (2026-06-22) 反向閘門 /pg →/architect+/pm:heatmap AC2(峰值=與原圖差最大)與 AC4(α=1→純色階、與原圖無關)在固定α混合下互斥(jet 低分也離綠遠)。改為標準淡入疊加 w=α·score;AC4 改測『α=0→原圖 + 熱度隨分數單調淡入』。
- (2026-06-28) **M4 主動學習取樣**:新增 `active_learning`(priority_score:novelty+boundary+disagreement×risk;
  select_for_labeling:cluster diversity 每群上限)。GUI 結果區「🎯 主動學習取樣佇列」(挑前 K 送標註 + 加購物車)。
  距離訊號(Normal Bank)必納入、不單靠 softmax(對未知過度自信)。gate 7 + E2E 1 綠 + 全套 anomaly E2E 10/10。
  教訓:跑 E2E 前清殘留 streamlit server(已寫進 .claude/skills/visuallatent-e2e)。
- (2026-06-28) **M3 DINO Classification Head + 閘控級聯**:依使用者提供的工業落地架構,新增 `dino_head`
  (frozen DINOv2 + LogisticRegression closed-set 分類)+ `gated_predict`(Normal Bank 異常分數守門:正常/已知
  類別/Unknown,防未知被自信誤分)。GUI 結果區加「🏷 瑕疵分類頭」(訓練/閘控分類/存讀頭)。先評估既有 memory
  bank=Normal Bank 已符合需求(bank 污染紀律已落實 confirmed-good only),再往分類頭。gate 6 + E2E 1 綠。
- (2026-06-28) **M2 Memory Bank 持久化 + 跨資料夾投影**:多 agent 三輪設計討論收斂 → 五模組 TDD 落地
  (anomaly_coreset/anomaly_project/anomaly_bank_store + run_pipeline external 注入 + _anomaly_ui GUI)。
  決策:fp32 評分、投影 transform-into-fixed-basis、few-shot 內容定址、greedy coreset 接進 build_memory_bank、
  寫硬碟白名單防呆、bank 面板移主畫面(避側欄收合)。單元 35 綠 + E2E 8/8 + 真實 DINOv2 端到端鏈 ALL-OK。
  設計/討論存 ANOMALY_BANK_DESIGN_NOTES.md / ANOMALY_BANK_DISCUSSION_RAW.md。
- (2026-06-27) 維護(非 U-Net feature):瑕疵偵測 GUI 一輪打磨(用戶連續回饋,只動 scripts/app.py + E2E)。
  ① 預覽縮圖標籤改寫「分數·**類別名稱**」,字色依判定 good=黑/bad=紅(先前的 Turbo 色框/底色全移除,
  讓深色字在白底清楚;_score_rgb/pcolors 一併刪)。② 已套用篩選時,框到的「變淡(灰)點」不納入框選
  (sel_idx 以 _passes 過濾)。③ 移除右欄可見排序清單(anomaly_ranked,與 selectbox 重複);E2E 連帶把
  4 處對 anomaly_ranked 的引用改讀 selectbox(新增 _ranked_stems;corrupt 改驗「共 11 個物件」)。
  ④ 三行說明改 manual:標題「🔧 瑕疵偵測」旁放 ❓ st.popover,平常不佔版面。⑤ 框選預覽區高度可由
  User 用 slider 調(預設 400、200–1000)。真實 app 驗證(popover/高度/字色/灰點排除全綠)+ gate 綠 + E2E 7/7。
- (2026-06-26) 維護(非 U-Net feature):瑕疵偵測「✕ 取消框選」比照其它工具,從散點圖下方(與購物車並排)
  移到**散點圖正上方**(placeholder 佔位、sel_idx 定案後填入),購物車鈕改全幅;與 Visualize/Compare/完整度
  一致。真實 app 驗證(鈕在散點上方、框選後啟用顯示張數、按下清空成 0、無例外)+ gate 綠 + E2E 6/6 綠。
- (2026-06-26) 維護(非 U-Net feature):瑕疵偵測再加兩項(同次對話,只動 `scripts/app.py`)。
  ① 框選預覽縮圖加「對應色框」:用 `plotly.colors.sample_colorscale("Turbo")` 取與散點圖同款分數色,
  `ImageOps.expand` 加邊框 → 縮圖一眼對上散點上的點(回應「下面沒標對應的顏色」)。
  ② 分數 slider 改雙邊範圍 `(lo, hi)`,`shown` 改 `lo<=score<=hi`;副作用:拉低高界可排除最極端、看圖大圖
  也能隨分數連動跳。E2E 連帶改 range slider 取 `.first`(低界)thumb。真實 app 驗證(2 thumb、拉低高界
  11/13→6/13 且看圖跳、預覽 9 張色框涵蓋全 Turbo)+ gate 綠 + E2E 6/6 綠。
- (2026-06-26) 維護(非 U-Net feature):瑕疵偵測「篩選連動」修復(用戶回報「篩選時左邊大圖不連動」)。
  原本右欄分數/類別篩選只改清單(`shown`),左欄散點圖固定渲染全件、看圖大圖黏在最可疑那張。修(只動
  `scripts/app.py` `_anomaly_ui`):① 散點圖拆兩 trace,未達門檻/非選類別的點變淡(灰半透明、仍可框選)、
  符合的實色(色階固定全距);② 篩選簽章變更→看圖自動跳到符合清單最可疑那張(篩選沒變保留手選)。
  關鍵:左欄散點圖讀 `session_state` 的篩選值來連動(上游圖回應下游 widget)。真實 app 驗證(slider
  12/13→5/13 散點圖確實變淡、類別 bowl→看圖跳圖)+ anomaly 單元 gate 綠 + E2E 6/6 綠。
- (2026-06-26) 維護(非 U-Net feature):Visualize 大重構 E2E 收尾。`test_scenarios_r1` + `test_gui_flows`
  12 紅 → 全綠(35 passed / 5 skip,verify4)。4 輪 multi-agent 診斷+修(對抗驗證 + 逐輪實跑)。
  **3 處 app 真實行為修復**:s07 切模式不再被「切粒度清除」偷走 viz_records(banner/復原鈕恢復)、
  j 補回 3D 高亮環 caption、w 候選池補回 headless 文字輸入。其餘為測試對齊(預設模型 vits14→vitb14、
  移除圖內標題)+ E2E harness 強化(`_ensure_sidebar` 等側欄真展開、面板切換沉澱+`.first`、互動 retry、
  multiselect 改 Clear-all)。無 skip/xfail/放寬斷言假綠。詳見 VISUALIZE_REFACTOR_E2E_PUNCHLIST.md。
- (2026-06-29) **M5 主動學習標註迴圈**(里程碑詳列):多 agent 複驗揪出分層切分假平圖 bug + E2E 假綠 +
  ROADMAP 名實不符殘留,fix-then-ship(commit 9fe0ba8 / b12de0a)。
- (2026-06-29) **M6 統一主動學習畫面**:修 silent-wrong — 把 YOLO 物件類別當瑕疵類訓 head(加 label 語義
  宣告守衛);S9 驗證 harness 樂觀洩漏誠實化(commit 950359c)。
- (2026-06-29) **M7 瑕疵偵測 wizard 重設計**:三步 wizard + 雙資料夾 + 唯一模型槽 + 2×2 取樣;移除 M5 學習
  曲線 GUI(active_loop 函式保留)。多 agent 對抗測試揪 [high] **confirmed 跨①②索引空間污染(silent-wrong)**:
  單一 key 被①訓練索引/②目標索引共用 → ①框選殘留滲②被當不同物件 → classify 門檻釘死、AUROC 假成 0 →
  命名空間化 `anomaly_confirmed_{build,apply}` 並隨 sig 變清殘留;[med] heat/class filter 共用滲漏、2×2 徽章
  head-exists 誤判 → 改依實際 verdict。**這是雙資料夾設計引入、舊單資料夾沒有的 bug**(commit cd0665f)。
- (2026-06-30) **M7 收尾**(三項,皆 commit 8443933 / a048ce4 / b345ead):① 步驟列 `st.tabs`→`st.segmented_control`
  修「每次互動跳回 Step-1」(tabs 的 active 是純前端狀態、rerun 必彈回 ①;keyed widget 跨 rerun 保留)。
  ② patch 建 Normal Bank 少樣本/diffuse **silent-wrong**(紅測 AC-F4a/F4b 先行):N<5 HDBSCAN min_samples>N 崩潰、
  5≤N<15 全 noise→空 normal_set→bank 靜默 None→假「✅已建立」悄退化物件級;修=mcs 夾 ≤N + patch 無 bank 時
  整批參考視為正常 fallback + app bank=None 明確報錯。real fruit 逐水果 patch 修後建出 bank(AUROC 蘋 0.844/
  香蕉 1.0/橘 1.0,見 [[fruit-anomaly-dataset]])。③ 語義 radio 文案 + 原生目錄 picker + 2×2 無訊號反灰
  (反灰判據用 `_proba is None` 而非 `head is None`,head 在但 predict 失敗也正確反灰)。
- (2026-06-30) **PO 宣告 feature 收斂 → 轉維護**:瑕疵偵測 M1–M7 涵蓋 PRD 全部 MoSCoW、fruit 真實瑕疵
  資料端到端驗證、無待辦 user_needs。後續走維護迴圈(只動 app+E2E、不受 appetite),新能力(瑕疵種類多類
  分類 / 報表 / 批次 CLI 等當初 Won't/Could)須由新 `/user` 需求重新起輪、不在本 feature 內擴張。未蓋棺
  尾巴 `N_min=8` 三 split 敏感度掃描列維護候選、非阻擋。(模組進度表仍為 M1 範圍;M2–M7 模組改以
  `*_DESIGN_NOTES.md`+里程碑追蹤,屬已知可接受分歧,補表為獨立 doc 整理、未排程。)
- (2026-06-30) **M8 起輪(新能力)**:收斂後使用者提「整張影像級瑕疵偵測(無 YOLO 標籤)」→ 依「新能力起新輪」
  開 `/user`→`/po`。需求:無 `labels/` 圖片資料夾要把整張圖當對象做異常偵測(現況跳「找不到 YOLO 物件」不能用)。
  PO 拆 1 模組 `whole_image_source`(Tier B):`discover_whole_images` 全幅記錄 + `run_pipeline` `object_source` routing +
  GUI ① 物件來源切換(鎖進 model.meta)。Won't:整張多物件分評 / 分類頭 / 背景遮罩(整張=單一對象,User 明說不在乎)。
  appetite ≤1 模組 + GUI 接線,一輪做完。模組 07 走里程碑追蹤(同 M2–M7),不動 M1 模組表。
