# ROADMAP(由 PO 維護)

> 狀態圖例:⬜ 待設計 · 📝 設計中 · 🧪 驗收定義中 · 🔨 開發中 · ✅ 完成
> 提醒:狀態是人的判斷,不可由「檔案存在」自動標 ✅。模組表/測試數請定期用
> `python verify/unet_status.py` 核對是否漂移。

> **🏁 Feature 狀態(2026-06-30):瑕疵偵測(AnomalyDINO 風格物件級瑕疵偵測)已宣告收斂。**
> M1–M7 涵蓋 PRD 全部 Must/Should/Could,並在真實瑕疵資料(`C:\code\dataset\fruit`)端到端驗證;
> 無待辦 user_needs。後續只走**維護迴圈**(只動 `scripts/app.py` + E2E、不受 appetite 約束);
> 要再加新能力須從新的 `/user` 需求重新起輪。未蓋棺尾巴(M6 `N_min=8` 三 split 敏感度掃描)列維護候選、非阻擋。
>
> **(2026-06-30 晚續)** 使用者提新需求「整張影像級(無 YOLO 標籤)」→ 依「新能力起新輪」規矩**重新起輪 = M8(已完成)**;
> 維護收斂結論不變,M8 是受 appetite 約束的新能力增量。
>
> **(2026-07-05)** 使用者提「大資料 AL 可用性 + 離線自動選樣服務」→ 再依「新能力起新輪」開兩個里程碑:
> **M9(大資料 GUI 可用性,設計中)** + **M10(離線監看服務,M9 綠後開)**。皆受 appetite 約束的新能力增量。

## 里程碑
- **M13 — 以樣搜樣(小樣本海撈 + YOLO 預標 + 人工確認,新功能分類=第 9 工具)** — ⬜ **規格已定案,待開發**
  (2026-07-05 使用者拍板 + 二輪問答**全部開放問題已定案**;待使用者下開發指令 → `/po` 收斂 PRD)—
  需求:小量樣本(**4 類×每類 5~10 張**)→ 海掃大資料(**帶低信心六欄 YOLO 粗框;框位置可用、類別不可信**,
  類別一律由樣本比對決定)→ YOLO 預標(沿用粗框幾何)+ **影像清單 CSV** → 人確認;確認回饋樣本集,
  累積夠導流瑕疵偵測①訓分種類(bootstrapping 閉環)。**監看版 v1 一起做**(復用 M10)。
  範圍:三步 wizard + 監看;multi-exemplar **max**-cosine;復用 al_batch(多參考加法擴充,21 測無回歸)/
  prelabel 匯出/M10 架構;**Won't** 粗框自動修框、相似度校準、全自動標註。模組:13a similarity 擴充(A)、
  13b `sample_bank`(B)、08 擴(B)、13c `retrieval_export`(A)、09/10 擴(B)、GUI(B)。
  需求 [1_user_needs/fewshot_search_prelabel.md](1_user_needs/fewshot_search_prelabel.md);規格
  [FEWSHOT_SEARCH_AND_PRELABEL_PLAN.md](FEWSHOT_SEARCH_AND_PRELABEL_PLAN.md)(§0 拍板決策表 + Task 1~7 順序)。
- **M12 — 找相似選樣目標(長得像指定物件)** — ✅ **完成**(2026-07-05;A1+A3 全綠、既有無回歸)—
  A1:`gate.py similarity` 8 + ③ E2E 1/1;A3:`gate.py al_batch` **21**(17 既有+4 similar)、`al_service` **9**
  (8+svc_sim)、`al_workspace` 14、M9 similar E2E `test_g8` 綠、M9/M10 GUI E2E 6/6+3/3 無回歸。—
  PRD [2_PO_PRD/similar_objective_prd.md](2_PO_PRD/similar_objective_prd.md)。挑一顆參考物件 → 整批依 cosine
  相似度降冪排到佇列前面。範圍:**Must** A1(③ 互動:從②結果挑參考→排序)+ A3(al_batch 加 objective=similar
  +ref_vector;M9 選單+參考挑選;M10 profile reference_vector_file)+ 共用純函式 `similarity_priority`;
  **Won't(v1)** 跨模型比對、A2 建模物件 provenance(進候選,參考一律來自當前②結果)。模組:**12 `similarity`
  (Tier A)** + **08 `al_batch` 改(加 similar,既有 17 測須無回歸)** + M9/M10 GUI + 09/10 profile 小改。
  appetite:**A1 先做完做綠 commit → 再 A3**。
- **M11 — 看過類別 AnomalyDINO 預標(分類頭代填,人工最終確認)** — ✅ **完成**(2026-07-05,
  `gate.py prelabel` **18 綠** + 真實 E2E **2/2 綠** + M9/M10 GUI E2E 無回歸;前置 **Task 0 已修+E2E 綠**)—
  PRD [2_PO_PRD/prelabel_prd.md](2_PO_PRD/prelabel_prd.md);設計 [11_prelabel.md](3_Architect_Design/11_prelabel.md)
  + [M11_gui_wiring.md](3_Architect_Design/M11_gui_wiring.md);對應表 [4_PM_Feedback/prelabel.md](4_PM_Feedback/prelabel.md);
  實作 `scripts/prelabel.py` + `app.py::_anomaly_prelabel_section`(③ 挑樣內)。**Should AC-E6(M10 佇列採納建議)
  未做,進候選。** 需求:已教過的瑕疵類別,新資料進來時模型先填好類別與信心,人只做確認/修正,
  重複標註工大減;沒把握留白(Unknown)不亂猜。範圍建議(待 PO 定 MoSCoW):**v1 只做「已有 bbox 的分類預標」**
  (包既有 `gated_predict`:正常/類別/Unknown + min_conf 門檻 + 指定類別 multiselect)→ 預覽表先看再收 →
  匯出 YOLO labels 到使用者**另選**資料夾(C6 絕不寫來源)+ M10 監看佇列旁「採納建議」;
  **Won't(v1)** 無標籤影像的偵測預標(bbox 生成,pmap→連通域是 v2 備忘)。模組建議:11 `prelabel`
  (Tier A 純邏輯)+ GUI 接線(Tier B)。**前置**:Task 0 = M9/M10 選樣目標名稱不匹配 bug(GUI `balanced`/`pure`
  傳 al_batch 會 ValueError,先修)。需求 [1_user_needs/prelabel_seen_class.md](1_user_needs/prelabel_seen_class.md);
  設計素材 [SIMILAR_OBJECTIVE_AND_PRELABEL_PLAN.md](SIMILAR_OBJECTIVE_AND_PRELABEL_PLAN.md)(Feature B + Task 0;
  同文件 Feature A「找相似選樣目標」為候選、未拍板)。
- **M9 — 大資料主動學習 GUI 可用性(分批可續跑引擎 + 標註佇列)** — ✅ **完成**(2026-07-05,引擎 08 gate GREEN + GUI 接線 4 E2E 綠) —
  瑕疵/主動學習 feature 收斂後的**新能力**輪次(起新輪)。需求:資料量大(數萬~十萬)時,現行
  「整條 pipeline 綁在 Streamlit 互動 session 同步跑完」→ 畫面卡死/等數小時/關掉重來,實質不能用。
  範圍(MoSCoW):**Must** 分批+checkpoint+**續跑**選樣引擎(凍結模型下逐批評分,任何時刻給得出
  「已處理併集」正確 Top-K)+ GUI **顯示三模式**(標註免投影只佇列+直方圖 / 探索抽樣散點 / 完整=現況,
  依量自動建議 >5000 走標註模式,可改)+ **暫定 Top-K**(明示「暫定·已處理 x%」)+ 續跑鈕;
  **Should** 佇列每項人話理由;**Won't** 離線服務(→M10)/跨機。模組:**08 `al_batch`(Tier B,
  無 GUI→單元+真實檔案系統整合測試驗收)**。需求 [1_user_needs/al_scale_offline.md](1_user_needs/al_scale_offline.md);
  PRD [2_PO_PRD/al_scale_offline_prd.md](2_PO_PRD/al_scale_offline_prd.md);設計素材
  [AL_SCALE_AND_OFFLINE_SERVICE_PLAN.md](AL_SCALE_AND_OFFLINE_SERVICE_PLAN.md)。appetite ≤1 模組 + GUI 接線,一輪做完。
  狀態:PRD 完成,放行 `/architect`。
- **M10 — 離線監看選樣服務(設定可攜 + 增量 + 佇列消費閉環)** — ✅ **完成**(2026-07-05,09+10 gate 綠 + GUI 3 E2E 綠) —
  需求:資料夾持續進新圖,使用者不想一直開互動畫面等算,要有背景服務按 AL 目標自動挑出「最該標註/釐清
  的前 100 個」。範圍(MoSCoW):**Must** 服務工作區持久化(設定/狀態/佇列/標註/摘要/鎖,單寫者)+
  CLI **run-once**(增量掃描→08 引擎→選樣→合併標註→物化佇列→摘要)+ GUI ④ 監看設定區&**匯出設定檔**
  給服務吃&服務狀態卡&「立即掃描一次」+ Step③ 佇列消費(縮圖+理由+三鍵 append 標註)+ **版本過期重評分**+
  **增量只算新圖**+**只讀來源**;**Should** 確認正常→擴 bank→重存模型引導、複製排程指令;**Won't** 常駐 daemon/
  服務自動重訓/多人/推播/跨機部署本身/CIM 平台整合(留鉤子)。模組:**09 `al_workspace`、10 `al_service`
  (皆 Tier B)**。appetite ≤2 模組 + GUI 接線,一輪做完。狀態:待 M9 完成後放行 `/architect`。
- **M8 — 整張影像級瑕疵偵測(無 YOLO 標籤)** — ✅ **完成**(2026-07-01,ROADMAP 於 2026-07-05 補回填,
  詳見決策日誌「ROADMAP 漂移修正」) — feature 收斂後第一個新能力增量(走精簡 U-Net 新輪)。需求:手上只有
  「無標註圖片資料夾」的人,要把**整張影像當對象**做異常偵測、不必先標框(現況無 `labels/` 直接跳
  「找不到 YOLO 物件」=完全不能用)。範圍(MoSCoW):**Must** 新「物件來源=整張影像」路徑(每張可讀圖→
  一筆**全幅記錄**,欄位對齊 `discover_yolo_objects`、下游 embedding/分群/bank/patch/評分/散點/排序/看圖/匯出
  **不變**)+ `run_pipeline` 依 `object_source` 選用 + GUI ① 切換「YOLO 物件 / 整張影像」+ 模式標示 + 鎖進 `model.meta`;
  **Should** 偵測 0 物件時提示一鍵切整張影像;**Could** 自訂標籤名;**Won't** 整張內多物件分評 / 分類頭 / 背景遮罩。
  模組:**07 `whole_image_source`(Tier B,走里程碑追蹤,同 M2–M7 不進 M1 模組表)**。設計
  [3_Architect_Design/07_whole_image_source.md](3_Architect_Design/07_whole_image_source.md);對應表
  [4_PM_Feedback/whole_image_source.md](4_PM_Feedback/whole_image_source.md);需求
  [1_user_needs/anomaly_whole_image.md](1_user_needs/anomaly_whole_image.md);PRD
  [2_PO_PRD/anomaly_whole_image_prd.md](2_PO_PRD/anomaly_whole_image_prd.md)。實作:
  `discover_whole_images`(interaction.py)+ `run_pipeline` object_source 路由 + GUI 接線(app.py _anomaly_ui)+
  預覽縮圖彩色徽章(commit 9bf0ae1)。驗證:單元 `test_whole_image_source.py` 11 綠 + 真實 E2E
  `test_whole_image_source_e2e.py::test_whole_image_build_mvtec_pill`(MVTec pill/train/good 無 labels 建模,
  commit 9bad2ba)綠;快速上手手冊見 docs(commit 86f59d2)。
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
- (2026-07-01) **M8 完成**:`discover_whole_images` + `run_pipeline object_source` 路由 + GUI 接線 + 預覽縮圖
  彩色徽章(commit 9bf0ae1);快速上手手冊(commit 86f59d2);單元 `test_whole_image_source.py` 11 綠 +
  真實 E2E(MVTec pill/train/good 無 labels 建模,commit 9bad2ba)綠。M8 完成後 ROADMAP 里程碑狀態未同步
  回填(仍停在「PRD→設計」),屬文件漂移,於 2026-07-05 維護輪查核時發現並在此補回填(見下一則)。
- (2026-07-05) **ROADMAP 漂移修正 + 一輪維護(單一資料夾語義 / 類別｜檔名徽章 / 監督UMAP 唯一預設)**:
  例行接手時用 `verify/unet_status.py` + git log 核對,發現 M8 早已完成(commit 9bf0ae1/9bad2ba)但
  ROADMAP 里程碑狀態未回填,已於本則更新前補正(見 M8 里程碑)。同時發現工作樹有一批未提交的維護
  變更(非本次新增,由使用者先前對話產出、待驗證提交),經完整驗證後提交(commit f062cb6):
  ① 資料夾選取改「單一資料夾」語義(選新的取代舊的,UI 改 [輸入路徑|📁] 同列);② 新 `_rec_fname()`
  統一縮圖徽章為「類別｜檔名」,鋪到 Visualize/Compare/Coverage/ObjCov/Anomaly 共 30 處;③ 預設投影方法
  由 4 種減為只算監督UMAP(極小樣本/無標籤仍由既有 PCA 保底邏輯接手,不空結果);④ grid 移除「第n」
  排名標籤。驗證:非 e2e 全套 490 passed(2 個 `test_scenarios_cov.py` 失敗經與 HEAD 逐一對照證實為既有
  跨測試 `.lv_cache` 假 embedding 污染,與本輪改動無關,**未修復**,列維護候選待辦);完整 Playwright E2E
  (55 測)跑 3 輪,過程中對抗式複查(19-agent workflow)+ 揪出並修復 1 個真回歸
  (`test_t_umap_reference_frame` 依賴「UMAP 曾是預設方法」才能點到「固定 UMAP 參考系」開關,改為手動
  加選 UMAP),其餘失敗逐一獨立重跑皆乾淨通過(長 session 資源衰退已知假紅模式,非本輪引入),第 3 輪
  49 passed/6 skipped/0 failed。**已知限制(接受、記錄不迴避,皆有逃生路徑)**:(a) 跨 split(train+val
  同載)洩漏掃描經 UI 已不可達,`test_p_duplicate_leakage_scan` 明確標 skip 並附理由,demo 按鈕程式化雙
  資料夾路徑不受影響;(b) 一般 Classifier 分析下標準 train/val/test 三分割資料集現在同樣只能載一個
  split(非僅洩漏掃描受影響),無額外警告,逃生路徑=分次載入分析;(c) 監督UMAP 唯一預設下,有標籤資料
  只算監督投影,PCA(無監督誠實視圖)保底只在「完全沒算出任何投影」時觸發、正常情況不觸發,需手動加選
  PCA 才有交叉校驗視圖,散點圖本身保證不空白。**維護候選待辦(非阻擋)**:`test_scenarios_cov.py` s07/s10
  的既有跨測試 16-vs-384 維 embedding 快取污染(dim=16 假 embedding 測試與真實 384 維 DINOv2 測試共用
  `.lv_cache` 隔離不足),需要時走 reverse gate 交 `/pm` 補快取隔離。
- (2026-07-05) **M9 + M10 起輪(新能力)**:使用者提「大資料 AL 可用性 + 離線自動選樣服務」→ 依「新能力
  起新輪」開 `/user`→`/po`。根因診斷:兩個痛點(GUI 跑不動大資料、想要離線自動選樣)同一根因=重運算綁死
  在 Streamlit 互動 session。解法一體:抽出「分批可續跑選樣引擎」,GUI 與離線服務為其兩個前端,「已存模型」
  為兩者唯一契約,「標註佇列」為核心產出物。PO 拆:M9=**08 `al_batch`**(分批+checkpoint+續跑+暫定 Top-K,
  Tier B);M10=**09 `al_workspace`**(工作區持久化+增量掃描)+**10 `al_service`**(CLI run-once 編排),皆 Tier B。
  **使用者四項拍板**:K 預設=100;版本過期項目=**重評分**(非標 stale);服務/GUI=v1 同機、**長期跨機**
  (架構不可寫死同機,路徑走可攜設定檔、服務為設定檔純消費者);GUI 有 config 設定區**可匯出設定檔**給服務吃。
  **其餘裁決**:多 profile v1 就做;標註模式自動切門檻 5000(可調);監看==建模資料夾警告不阻擋;已刪檔佇列項標
  stale 隱藏不刪。**Won't**:常駐 daemon(用輪詢)/服務自動重訓/多人/推播/跨機部署本身/CIM 平台整合(留鉤子,
  佇列 schema 參考 `labeling_handoff.py`)。appetite:M9 ≤1 模組、M10 ≤2 模組,各一輪;**M9 綠了才開 M10**。
  設計素材預聚合於 `AL_SCALE_AND_OFFLINE_SERVICE_PLAN.md`(非契約)。模組 08/09/10 走里程碑追蹤(同 M2–M8),不動 M1 表。
  狀態:PRD 完成,放行 `/architect` 設計 08(M9)。
- (2026-07-05) **M9 核心引擎 08 `al_batch` 設計→開發 GREEN**:走完整 /architect→/pm→/pg。
  **/architect** 用 workflow(4 agent)對既有程式碼逐行紮根 + 對抗式挑批次正確性漏洞(7 high+5 med 缺口+11 C8 風險),
  避開一堆 silent-wrong(最關鍵:**不呼叫 run_pipeline**——它每批無條件跑全域 HDBSCAN/門檻/ranking,違反 C8;
  改直接組 per-item 子函式)。**經使用者審查核准 3 個與 plan 的偏離**:① objective `confusion` 更正為
  「entropy 目標(吃 head 不吃 labels)」——既有 `confusion_targeted_priority` 本就不吃 labels,如此 al_batch 零 M10 相依;
  ② M9 丟 disagreement 項(凍結 meta 無門檻,單批算門檻違反 C8),uncertain 改用 boundary+entropy;
  ③ bank-only 不做 HDBSCAN 多樣性(有 head 用預測類別;純 bank `diversity_applied=False`)。
  **/pm** 15 條釘死 AC + 2 推導(多次續跑收斂、AST 反向稽查禁用全域運算),先紅正確。
  **/pg** 實作 `scripts/al_batch.py`(load_frozen_model/run_batched/read_checkpoint;item_id 用影像內容 sha256 → rescore 原地更新;
  Top-K 依 item_id canonical 排序消 tie-break 不確定;shard 原子寫+state sentinel-last 續跑冪等)。
  **反向閘門 /pg →(測試錯)→ /pm**:AC7 的 `dataset_dirs` 宣告過寬使 checkpoint 被 `assert_safe` 誤擋 + 反向稽查測試
  grep 字面字串誤中實作註解 → PM 修為指真實資料夾 + 改用 AST 檢實際呼叫,重 snapshot baseline,再放行 PG。
  驗證:`python verify/gate.py al_batch` = **GREEN(17 測全過、3_/4_ 契約未竄改)**。
  al_batch 無 GUI → done=單元/整合綠(真實檔案系統續跑/原子性/不寫來源資料夾皆有整合斷言)。
  **M9 里程碑其餘**:GUI 三模式顯示(標註/探索/完整)+ 暫定 Top-K + 續跑鈕接線,需真實 Playwright E2E,待做。
- (2026-07-05) **M9-GUI 接線 設計 + E2E 契約完成(PG 實作待做)**:走 /architect→/pm。**/architect** 用 workflow
  (2 agent)紮根 `_anomaly_ui`:好消息=`_anomaly_save_model` 寫的凍結目錄與 `al_batch.load_frozen_model` 消費格式
  100% 相容、引擎公開函式零改;設計 [3_Architect_Design/M9_gui_wiring.md](3_Architect_Design/M9_gui_wiring.md)——進入點在
  ② `_anomaly_tab_apply`(重用 model slot + 目標資料夾選取器 + `dataset_cache_dir` 產 .lv_cache checkpoint),三顯示模式
  (標註免投影只佇列+直方圖 / 探索抽樣散點 / 完整=現況)、跨 rerun「flag+st.rerun 自續」(非 sleep 輪詢)、暫定 Top-K、
  停止/繼續上次。**使用者審查核准**:未存模型→**自動存再掃**;順手修 `_anomaly_load_model` 掉 `object_source` 的既有
  latent bug。**/pm** 4 條真實行為 E2E(`tests/e2e/test_al_batch_gui_e2e.py`:佇列渲染+reason、標註無散點、自動存、
  object_source 從磁碟、續跑)+ 對應表 [4_PM_Feedback/M9_gui_wiring.md](4_PM_Feedback/M9_gui_wiring.md);snapshot 21 契約檔。
  **狀態:設計+契約鎖定,`/pg` 接線待做**(大改動 + 需真實 Playwright E2E 驗收,不進 PG 自主修綠迴圈,交 /ux-test 或人觸發)。
- (2026-07-05) **M9-GUI 接線 /pg 完成 → M9 里程碑達成**:在 `_anomaly_ui` ② 加「⚡ 大資料分批掃描」區塊
  (`_anomaly_batch_run`/`_anomaly_batch_section`/`_anomaly_batch_render_queue`):選樣目標(head 閘控)+ K(預設100)+
  掃描/繼續上次 → **未存模型自動存**(使用者拍板)→ **object_source 從磁碟 meta.json 讀**(不信 in-memory)→
  `al_batch.run_batched`(.lv_cache checkpoint,阻塞+即時進度)→ Top-K **標註佇列**(縮圖+al_batch 自帶 reason+
  分數分佈+分頁+購物車注入 path)。順手修 `_anomaly_load_model` 掉 `object_source` 的既有 latent bug。
  **反向閘門 /pg→/architect**:探索/完整散點模式需 al_batch 未回傳的 obj_emb/投影 → v1 修正為**標註佇列 only**
  (大資料本就不該畫全量散點;散點三模式延後,需 al_batch 未來回傳抽樣 obj_emb)。
  **反向閘門 /pg→/pm**(2 次):E2E `set_model_dir` 需在 build 後呼叫(存模型鈕 `disabled=not(model and dir)`)→ 修測試順序;
  whole_image 模式無語義 radio(恆 1 類)→ g6 不設語義;auto-save 目錄改用真實行為訊號驗證(非脆弱路徑)。
  驗證:**真實 Playwright E2E `tests/e2e/test_al_batch_gui_e2e.py` 4/4 綠**(佇列+reason 渲染、標註無散點 scale-safe、
  未存自動存、object_source 從磁碟=whole_image 物件數==影像數、續跑無例外)。**M9 里程碑(引擎+GUI)完成;M10 解除閘門可開。**
- (2026-07-05) **M10 完成(離線監看選樣服務)—— 瑕疵/主動學習 feature 線收斂**:M9 綠後開 M10,走完整
  /architect→/pm→/pg ×(09、10、GUI)。**09 `al_workspace`**(工作區持久化 + 增量掃描 + 執行鎖;設計
  [3_Architect_Design/09_al_workspace.md](3_Architect_Design/09_al_workspace.md)):單寫者(服務寫 queue/state、GUI 只
  append labels)、可攜 profile.yaml、mtime+size 雙鍵增量、`id`=al_batch item_id(重評分原地)、原子寫/安全目錄/鎖
  重用既有;`verify/gate.py al_workspace` **14 綠**(一次過)。**10 `al_service`**(CLI run-once 編排;設計
  [3_Architect_Design/10_al_service.md](3_Architect_Design/10_al_service.md)):load_profile→鎖→驗模型→增量掃描→
  al_batch 全量評分(.lv_cache 讓只有新圖真 embed=增量)→合併標註(移出已標/deleted)→物化 Top-K 佇列→摘要,
  try/finally 放鎖;模型換版→重評分(item_id 內容不變原地更新)=拍板;`gate.py al_service` **8 綠**(修 1 個
  item_id→id 欄位映射 PG bug 後)。**M10-GUI**(② 加「🛰 持續監看服務」;設計
  [3_Architect_Design/M10_gui_wiring.md](3_Architect_Design/M10_gui_wiring.md)):工作區設定 + 🆕初始化 + 📤匯出設定 +
  ▶立即掃描(in-process run_once,阻塞+進度)+ 服務狀態卡 + 監看佇列消費(縮圖+reason+✅正常/🏷瑕疵/⏭略過→
  append_label→下輪移出)。**反向閘門 /pg→/pm**:探索/完整散點延後(同 M9,al_batch 無 embedding 回傳);
  E2E 讀「實際生效 ws」(text_input 非空預設 fill 不覆蓋)、profile 讀 model_dir 驗自動存。**PG 修 2 個 impl bug**:
  ① watch_ws 預設用 setdefault 在目標未加時鎖住空字串→init 反灰(改「空且有目標→補預設」);② 缺「掃描完成」DOM 標記。
  al_service 加 `progress` 附加參數轉發 al_batch(gate 仍 8 綠)。驗證:**真實 Playwright E2E
  `tests/e2e/test_al_service_gui_e2e.py` 3/3 綠**(初始化+profile.yaml+匯出、掃描+佇列+reason、標註閉環下輪移出、
  未存自動存)。**M10 完成 = 瑕疵偵測→大資料→離線服務整條 feature 線收斂;要再加新能力須新 `/user` 起輪。**
  (2026-07-05 後續:已 rebase+push 上 remote(e24b85b);平台 nativeApp 已釘 submodule 指標(b8521bf),
  平台 repo 本身 push 待觸發。)
- (2026-07-05) **/user 起新輪:M11「看過類別預標」拍板**:使用者確認要「透過 AnomalyDINO/分類頭對看過的
  類別做預標」。與 M5「不做全自動標註」不衝突——M11 定位是**預**標+人工最終確認,人仍是最終裁決。
  需求落 [1_user_needs/prelabel_seen_class.md](1_user_needs/prelabel_seen_class.md);設計素材
  [SIMILAR_OBJECTIVE_AND_PRELABEL_PLAN.md](SIMILAR_OBJECTIVE_AND_PRELABEL_PLAN.md)(Feature B)。盤點時**順手挖到
  真 bug(=Task 0,建議 M11 前置先修)**:M9/M10 GUI 選樣目標 `balanced`/`pure` 原樣傳 `al_batch.run_batched`,
  但引擎只收 `novelty/uncertain/confusion` → ValueError「分批掃描失敗」;現有 E2E 只測 novelty 故未攔到。
  同文件 Feature A「長得像指定物件」選樣目標為候選、**未拍板**。狀態:待 `/po` 收斂 PRD 後放行 `/architect`。
- (2026-07-05) **Task 0 維護修復(M9/M10 選樣目標名稱不匹配)**:GUI `_OBJ` 曾列 novelty/balanced/confusion/pure
  4 項,但 `al_batch.run_batched` 只收 novelty/uncertain/confusion → 選 balanced/pure 直接
  「分批掃描失敗:objective must be…」(現有 E2E 只測 novelty 未攔)。修:app.py 加 `_AL_ENGINE_OBJ`
  映射(balanced→uncertain、pure→novelty),批次/監看兩處呼叫套用;選單砍同義的「純 novelty」成 3 項誠實選單;
  watch 區塊補 head 閘(與批次一致,無 head 選 balanced/confusion → 反灰而非掃描時 raise)。
  驗:新增 E2E `test_al_batch_gui_e2e.py::test_g7_balanced_objective_maps_to_uncertain`(含 head 模型選
  三訊號均衡 → 掃出佇列、無 ValueError)**1/1 綠(58s)**。維護模式(無 role)完成,不走 U-Net 五層。
- (2026-07-05) **M11 `/po` 收斂 PRD**:[2_PO_PRD/prelabel_prd.md](2_PO_PRD/prelabel_prd.md) —— v1 只做「已有 bbox 的
  分類預標」(Must:prelabel 三純函式 + GUI 預覽/類別過濾/匯出另選目錄;Should:M10 佇列採納建議;
  Won't:無標籤影像的偵測預標=v2)。模組分解:**11 `prelabel`(Tier B,無 GUI→gate=單元+真實檔案系統測試)**
  + M11 GUI 接線(Tier B,真實 E2E)。appetite ≤1 模組+GUI。給架構師 4 個開放問題(正常物件剔除/conf 定義/
  C6 檢查粒度/類別對齊以 head 為準)。放行 `/architect`。
- (2026-07-05) **M11 完成(architect→pm→pg 一輪)**:**11 `prelabel`**(設計
  [11_prelabel.md](3_Architect_Design/11_prelabel.md);純函式 `prelabel_records`/`to_yolo_lines`/`export_prelabels`
  + `assert_safe_prelabel_dir`):閘控判定→可接受預標→安全 YOLO 匯出。**架構師 4 開放問題定案**:①正常物件剔除
  ②conf=head 最大類別機率、Unknown 一律不 accept ③C6 檢查=含 images/ 或與來源有祖先/後代/等於關係即拒
  ④cls_id 以 head 類別空間為準。**關鍵決策**:prelabel 需**專屬** `assert_safe_prelabel_dir`(既有
  `assert_safe_bank_dir` 見 labels/ 會誤擋重複匯出)。`gate.py prelabel` **18 綠**(含 AC17 來源零寫入不變量、
  AC18 冪等)。**M11-GUI**(③ 挑樣加「🏷️ 預標」expander;設計 [M11_gui_wiring.md](3_Architect_Design/M11_gui_wiring.md)):
  min_conf slider + 目標類別 multiselect + 預覽縮圖牆(pred·信心·✅收)+ 匯出到**另選**目錄(絕不寫回來源);
  無 head 友善降級。**真實 Playwright E2E `test_prelabel_gui_e2e.py` 2/2 綠**(AC-E1~E5:預覽 accept 數==實際
  匯出行數、讀回 labels/*.txt 驗 cls_id、來源資料夾前後檔案集合不變=C6 真實不變量、無 head 無匯出鈕)。
  回歸:M9 批次 GUI E2E 5/5、M10 監看 GUI E2E 3/3 全綠(Task 0 改動無回歸)。**Should AC-E6 未做進候選。**
- (2026-07-05) **/user 起 M12「找相似」拍板(A1+A3)**:使用者在 app 找不到「找相似」→ 確認該功能原為 Feature A
  「候選/未拍板」(只在規劃文件),遂拍板開發,scope=**A1(③ 互動)+ A3(批次:al_batch+M9/M10)**。PRD
  [2_PO_PRD/similar_objective_prd.md](2_PO_PRD/similar_objective_prd.md)。模組:12 `similarity`(Tier A 純函式)
  + 08 `al_batch` 加法改造(objective=similar+ref_vector,既有 17 測無回歸鐵則)+ M9/M10 GUI + 09/10 profile 小改。
  A2(建模物件 provenance / ref_meta.json)本輪不做,參考來源限當前②結果。appetite:A1 綠+commit → 再 A3。放行 `/architect`。
- (2026-07-05) **M12a(A1)完成**:**12 `similarity`**(設計 [12_similarity.md](3_Architect_Design/12_similarity.md);
  `cosine_similarity_to_ref` / `similarity_priority`,降冪=最像,_minmax 與 active_learning 一致)——
  `gate.py similarity` **8 綠**(含 AC7 正尺度不變、AC8 同向最大衍生測)。**③ GUI「🔎 找相似」**(設計
  [M12a_gui_wiring.md](3_Architect_Design/M12a_gui_wiring.md)):挑參考物件索引 → cosine 相似度降冪縮圖牆
  (排除參考自身)+ 加購物車;無 obj_emb / <2 物件友善降級。**真實 E2E `test_similar_gui_e2e.py` 1/1 綠**
  (AC-S1 相似佇列同群佔多數、AC-S2 換到另一群當參考 → 佇列多數翻群 = 排序真跟參考走,真實 DINOv2 非 element 存在)。
  A1 段 commit 後接 A3(al_batch objective=similar + M9/M10)。
- (2026-07-05) **M12b(A3)完成**:**08 `al_batch` 加法擴充**(設計增補於 [08_al_batch.md](3_Architect_Design/08_al_batch.md)):
  `run_batched(..., ref_vector=)` + objective=`similar`;per-item cosine 存 shard `ref_sim` 欄(C8-safe)、
  merge 端 `_minmax(ref_sim)` 排序、reason「相似度」、run 身分納入 `sha256(ref_vector)`(換參考=另一 run)。
  復用模組 12。`gate.py al_batch` **21 綠**(17 既有無回歸 + AC-SIM1~4:相似排序/缺 ref 拒/換參考重算翻群/分批==一次跑)。
  **09/10 服務**:profile 加 `reference_vector_file`;`init_workspace(reference_vector=)` 存 `reference.npy`(atomic)+
  profile 欄位;`run_once` similar 載入傳引擎(`gate.py al_service` **9 綠**,AC-SVC-SIM)。**M9/M10 GUI**(設計
  [M12b_gui_wiring.md](3_Architect_Design/M12b_gui_wiring.md)):批次/監看選單加「🔎 找相似」+ 參考物件索引(取自②結果 obj_emb)、
  無②結果友善降級;`_AL_ENGINE_OBJ` 加 similar;批次每個參考各自 checkpoint。**M9 similar 真實 E2E `test_g8` 綠**
  (reason 含相似度、無例外),M9/M10 既有 GUI E2E 6/6+3/3 無回歸。M10 similar 核心以 al_service 單元 AC-SVC-SIM 覆蓋。
  **M12(找相似 A1+A3)收斂。**
- (2026-07-05) **M12 UX 精修 + 選樣目標人話化(使用者回饋)**:使用者反映「參考物件索引」裸數字看不懂、
  且 novelty/N_min/含 head 等術語難懂。改:(1)新純函式 `similarity.class_centroid`(gate similarity **11 綠**);
  (2)③/M9/M10 共用參考挑選器 `_anomaly_pick_reference`——「參考依據」切換**像某一類**(用該類平均向量)/
  **像某一個物件**(下拉顯示「第N·類別·檔名」+縮圖,不再裸索引);(3)選樣目標/警告/取樣矩陣/N_min 全改人話
  (novelty→「抓沒看過的異常」、需分類頭→「需會分辨瑕疵種類的模型/每種至少 8 張」)。E2E:③ 類別模式
  `test_s1_s2_class_reference_follows` 綠、M9 g8 改類別模式綠、g7 標籤更新綠;wizard/prelabel/M9(6)/M10(3)
  全回歸綠(「無分類頭」提示來自未改的取樣佇列行)。**誠實界定**:①建模狀態列「含/無分類頭」因被 ~10 條
  E2E 斷言依賴,本輪未改(留候選,要動需連同更新那批測試)。
- (2026-07-05) **維護修復:長時操作進度條不更新(使用者回報)**:①建模、②套用、M9 分批掃描、M10 監看掃描
  四個都把 `st.progress` 放在 **on_click callback** 裡——Streamlit **不會從 callback 即時串流**進度更新
  (卡到整個操作跑完才一次更新,大資料時看起來像凍住)。修:改「callback 只設 pending flag → 主體
  (script body)執行長時操作」讓 progress 真的會動;M9 分批另把 batch_size 1000→64(每 64 張更新一次)
  + on_identity_mismatch='restart'(換參數不報錯改重跑)。回歸:build/apply/batch/watch 相關 **15 條 E2E 全綠**
  (完成標記照舊出現)。維護模式(無 role)。
- (2026-07-05) **進度條後續(使用者回報)**:① 漏改 M10 監看/離線服務的 batch_size —— 它讀 profile 預設
  (`al_workspace._DEFAULTS`),仍是 1000 → 每 1000 張才更新;改預設 1000→**20**(GUI 批次亦 64→20,兩端一致
  每 20 張更新;既有 profile 需重按「初始化監看」才生效)。② build/apply 的 run_pipeline 進度本來就逐物件
  回報(不受影響,只需前一輪的 callback→主體修復)。③ 四處長時操作加 `st.spinner`(動畫)+「首次先載入模型
  (約 10~30 秒)」文字,補冷啟動空窗的狀態提示。④ 監看佇列 ✅/🏷/⏭ 按鈕改「icon+字」且不再撐滿寬度。
  gate al_workspace 14 / al_service 9 綠、M10 GUI E2E 3/3 綠。
- (2026-07-05) **/user 起 M13「以樣搜樣」候選**:使用者拍板核心工作流「小樣本 → 海掃大資料 → 撈同類 →
  YOLO 預標 → 人確認」為主要日常,要求成**獨立功能分類**(第 9 工具,避免與瑕疵偵測「找異常」方向混淆——
  M12「像某一類」重疊教訓)。定位=教會模型**之前**的 bootstrapping 撈資料工具,與 M11(有 head 後)分工、
  確認累積導流回①訓頭。需求與設計素材已落檔(見 M13 條目);4 個開放問題(無標資料佔比/樣本規模/
  監看版/匯出形式)待拍板後 `/po` 收斂。**只記錄、未開發。**
- (2026-07-05) **M13 規格定案(二輪問答)**:①大資料**有低信心六欄 YOLO 粗框**(cls cx cy w h conf;
  parse_yolo_boxes_conf 已支援)→ 框生成從 Won't 變不需要,粗框=proposal;②**粗框類別不可信,只用框位置**,
  類別一律由樣本比對決定;③樣本=每類 5~10 張×約 4 類 → max-cosine + 「確認加回樣本集」滾大;
  ④**監看版 v1 一起做**(profile 擴充 sample_bank_dir,復用 M10);⑤匯出=YOLO labels(沿用粗框幾何)
  **+ retrieval_report.csv**(欄位已釘死)。規格全落 FEWSHOT_SEARCH_AND_PRELABEL_PLAN.md §0/§2/§4/§6,
  含既有教訓(進度條 pending-flag 模式、術語人話、C6/C8)。**未開發,等指令。**
- (2026-07-05) **M12 設計修正(多 agent 討論,使用者觸發)**:使用者質疑「找相似的參考該來自①而非②」→
  4 視角 + 綜合裁決:**「像某一個物件」(by-example)對且是旗艦**(novel 物件只能來自②);
  **「像某一類」錯**——(a) silent-wrong:class_centroid 吃②的 YOLO label,②常無標 → 全 '—' → 退化成全體平均
  不報錯;(b) 與 M11 預標重疊(有 head 找已知類該用 head,centroid 是未校準弱版)。拍板:**移除「像某一類」**、
  找已知種類導流「🏷️ 預標」;「找相似」改名「**🔎 找同款**」(by-example 專責);參考來源顯性切換
  **② 這次掃描 / ① 建模範例**(①同 session 用 anomaly_train_result,免持久化;跨 session=候選)。
  head coef_ 當類別代表 = 語義誤導,排除;PLAN A2「對 bank 向量加 provenance」= 誤規格(bank 是 patch coreset
  非 obj_emb 空間),棄用。
