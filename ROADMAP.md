# ROADMAP(由 PO 維護)

> 狀態圖例:⬜ 待設計 · 📝 設計中 · 🧪 驗收定義中 · 🔨 開發中 · ✅ 完成
> 提醒:狀態是人的判斷,不可由「檔案存在」自動標 ✅。模組表/測試數請定期用
> `python verify/unet_status.py` 核對是否漂移。

## 里程碑
- **M5 — Active Learning Loop(主動學習標註迴圈)** — ✅ **完成**(2026-06-28) — 把 M2 bank + M3 head +
  M4 佇列從**開迴圈串成閉迴圈**:佇列選樣→人工標→回流(擴 Normal Bank + 重訓 head)→量測學習曲線→重複。
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
