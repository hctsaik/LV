# 0_discussion — AL 迴圈 UX/協作討論 導讀

> **正典摘要已更新：**請先讀 [`../5_active_learning_product_review_2026-07-12/00_結論與建議.md`](../5_active_learning_product_review_2026-07-12/00_結論與建議.md)。本資料夾保留 2026-07-11 的逐頁證據、multi-agent 辯論與主席裁決全文；其中 `05_debate_log.md` 的「不做／緩做」仍具治理約束，翻案需提出新證據並寫入 ROADMAP 決策日誌。

> **狀態校正（2026-07-12）：**08／09 的純邏輯核心與單元測試已完成；歷史文件中的「未開工」應理解為 GUI 接線與 E2E 尚未完成。最新排程用語統一為「M9 GUI 接線」。

> **本資料夾為 2026-07-11 multi-agent 討論產出,尚未開發。**
> 所有內容是「建議」不是「開發指令」;任何採納項目必須走 U-Net 五階段流程
> (`/user` → `/po` → …)立案後才可動工,詳見文末〈下一步〉。

## 本文件回答什麼

1. 這次 multi-agent 討論的**目標與方法**(8 盤點 + 3 視角 + 2 反方 + 1 主席)。
2. 主席裁決的**核心結論**:P0 項目全文列出,P1/P2 摘要,均標定位(既有元件重排/已規劃/新需求須立案)。
3. 資料夾內 5 份文件的**檔案地圖**與閱讀順序。
4. 若採納建議,**下一步該怎麼走**(哪些必須 `/user` → `/po` 立案,哪些待 User 放行)。

---

## 1. 討論目標與方法

**目標**:LV(`scripts/app.py`,約 8200 行單檔 Streamlit)已長出六個可見工具
(Visualize、Compare、完整度熱力圖、瑕疵偵測、匯出、📥 標註回饋)與三個隱藏工具
(組考卷、灰帶覆核、評估),並透過 `labeling_handoff` 與外部標註系統 ANnoTation 單向送出/讀回。
本次討論回答兩個問題:**(a)** 電腦視覺工程師如何更順地跑完整條 active learning 迴圈;
**(b)** User(標註者/領域專家)與 CV 工程師的協作權責如何顯性化。

**方法**(共 14 個 agent 角色):

- **8 個盤點 agent**:逐頁/逐功能實讀程式碼(Read/Grep,引用一律附 file:line),
  盤點每頁的輸入/輸出/與 AL 迴圈的接點/模糊點 → 產出 `01_page_inventory.md`。
- **3 個視角 agent**(A/B/C):分別從「CV 工程師動線」「User/標註者權責」「半導體產線場景」
  提出合併與重構提案 → `02_merge_proposals.md`、`03_al_workflow_roles.md`、`04_semiconductor_scenarios.md`。
- **2 個反方 agent**:反方一守「工程現實」(E2E 選擇器脆弱性、st.tabs rerun flaky、
  歷史更名疤痕、vendored 雙副本同步);反方二守「產線現實」(審計可追溯、採購級門檻、
  裁量權不可籠統外包)。逐條攻擊三視角提案 → 記錄於 `05_debate_log.md`。
- **1 個主席**:複核全部 file:line 引用後收斂裁決,產出處置清單、目標資訊架構、
  分工矩陣、半導體 Top 5 與不做/緩做清單(本 README §2 即其摘要)。

---

## 2. 主席裁決核心結論

### 2.1 P0 項目(全文)

| # | 項目 | 定位 | 內容 |
|---|------|------|------|
| P0-1 | **讀回套用落地層** | **新需求須立案**(08–13 規格均未涵蓋) | 現行「套用」只改記憶體中的 records(labeling_handoff.py:400-413),重 Run 後白標蒸發。裁決:落地 `.lv_cache` 標籤覆蓋層 + append-only 變更日誌(舊標/新標/annotator/時間戳),匯出可選「含修正」。四方一致無異議,同時化解視角 A 的最高缺口與反方二的「審計否決項」。 |
| P0-2 | **「不用回 LV」文案統一** | 既有元件重排(文案修正) | 7 處送標文案寫「不用回 LV」(app.py:2412、2712、2787、3289、6336、7205、7234),與 📥 標註回饋分頁「把新標籤套進目前載入的資料」(app.py:7257)正典矛盾。統一為「送出 → ANT 標 → 回 LV 套用」;同 PR 掃 `tests/e2e` 可見字串避免 E2E 紅。 |
| P0-3 | **probe_eval 接 GUI** | 已規劃(`3_Architect_Design/12_probe_eval.md`),GUI 接線部分依規格 | 目前 app.py 只從 active_loop import 一個函式 `confusion_targeted_priority`(app.py:1541),「還要標多少/自動率多少」在 GUI 答不出。裁決:M12 probe_eval 接 GUI + per-class recall + overkill/underkill CSV。反方二列為採購級門檻,無異議。 |
| P0-4 | **pool_registry / round_ledger 依規格開工** | 已規劃(`3_Architect_Design/08_pool_registry.md`、`09_round_ledger.md`),**仍待 User 放行** | 現況:重複送標無防呆、回合不可追、換班失憶。開工時順手拆 ANT 端 module_014 的全批 `read_back` 語義(014_process.py:63-67 把所有未讀批次一次標成 `read_back`,無法逐批)。開工程序走 CLAUDE.md 閘門(User 放行 → PO 記決策日誌 → 五階段)。 |
| P0-5 | **清假地圖文案 + 刪死碼** | 既有元件重排 | 工具列說明(app.py:8101 一帶)與功能地圖仍指向已隱藏工具(session 殘值在 app.py:8094-8099 被正規化回 Visualize);`_render_cmp_panel`(app.py:4596)定義後零呼叫,刪除。**保留** `_send_to_labeling_ui` 的灰帶 handoff 呼叫(app.py:7642、7791,`TASK_ADJUDICATE` 唯二存活點,是活契約)。 |

### 2.2 P1 摘要

| 項目 | 定位 | 一句話 |
|------|------|--------|
| 框級 shapes 讀回**分段** | 前段=P0-1 一部分;後段=新需求(配 M13 預標框) | 讀回目前只取 `shapes[0].get("label")`(labeling_handoff.py:287-288),幾何全丟;先讓分類字串迴圈可審計地轉起來,幾何讀回配 M13 再做。 |
| task 顯式 selectbox | 既有元件重排 | task 現由排序語境隱式推導(app.py:2403-2404),標註者不知任務;改顯式選單,預設=現行推導值(行為不變)。 |
| 四態/round_id/annotator/選樣理由顯性化 | 部分已規劃(09)+ 小案立案 | `sent→annotating→exported→applied` 用新增狀態值不改名;向後相容 + 三處同步(含 nativeApp vendored 副本)。 |
| 瑕疵③補直送鈕 | 既有元件重排 | 瑕疵挑樣佇列目前只能進購物車(app.py:1968-1970),最該送標的頁沒有直送出口;復用 `_send_to_labeling_ui`(app.py:7190),送原圖路徑非 crop、key 進 anomaly 命名空間。 |
| 取樣訊號後端統一 | 已規劃(M11 hybrid_sampler) | Visualize 選樣與瑕疵③ 2×2 **後端**共用 M11 純函式;兩頁 UI 各自保留(M7 索引空間污染 + st.tabs flaky 教訓,不合併 UI)。 |

### 2.3 P2 摘要

| 項目 | 定位 | 一句話 |
|------|------|--------|
| 三個「匯出」撞名 | 既有元件重排(僅文案) | 不造新名,收斂到既有「匯出清單」;E2E 選擇器 8 處/10 檔 + 舊名「匯出子集」更名疤痕(app.py:8096)為戒。 |
| 行話白話化 | 既有元件重排(僅 caption/help) | 「桶①佔比」等只改 caption/help;widget label 緩(E2E 文字選擇器命中)。 |
| 隱藏三工具處置 | 新需求須立案(ROADMAP 決策) | 評估→M10/M12 回歸、灰帶裁決掛回 📥、組考卷降級 kappa;**PO 先記決策日誌**再動。 |

### 2.4 其他裁決要點(詳見各檔)

- **資訊架構落點**:📥 分頁擴建為「🔁 標註迴圈」(狀態列 + ①挑樣②送標③回讀④評估⑤下一輪),
  **不新增第七工具列項**(肌肉記憶 + app.py:8098 正規化耦合)。→ `02_merge_proposals.md`
- **User 三分**:複判員/製程工程師/品保三種角色,各有 RACI;裁量權整包移交籠統「User」駁回。→ `03_al_workflow_roles.md`
- **半導體 Top 5**:ADC 複判主線、極端不平衡 rare-event、Golden 樣本庫(需立案)、
  稀有模式發現、漂移監控(只做可匯出報表);**前置盲區**=lot/wafer/die metadata schema
  為全新需求,先於任何 UI 重構立案。→ `04_semiconductor_scenarios.md`
- **不做/緩做**:挑樣 UI 元件合併、第七工具列項、lot 漂移自動排程、標註 UI→ANT schema
  自動同步、裁量權整包移交——均不做;更名、shapes 幾何讀回、隱藏工具結構性移除——緩做。→ `05_debate_log.md`

---

## 3. 檔案地圖

| 檔案 | 內容 | 建議閱讀對象 |
|------|------|--------------|
| `01_page_inventory.md` | 頁面盤點:六可見 + 三隱藏工具逐頁的輸入/輸出/AL 接點/模糊點,全數附 file:line | 想知道「現況到底長怎樣」的人,先讀這份 |
| `02_merge_proposals.md` | 合併提案:📥 擴建為標註迴圈的資訊架構、哪些元件重排、哪些提案被反方否決 | CV 工程師、PO |
| `03_al_workflow_roles.md` | AL 協作工作流:User 三分(複判員/製程/品保)× CV 的 RACI 矩陣、最小顯性化資訊集(round_id/task/四態/annotator) | 導入協作流程的主管、PO |
| `04_semiconductor_scenarios.md` | 半導體場景:Top 5 場景逐一對映現有能力與缺口、前置盲區(metadata schema) | 評估產線適用性的人 |
| `05_debate_log.md` | 討論記錄:三視角提案 × 兩反方逐條攻防 × 主席裁決理由,含不做/緩做的完整論證 | 想知道「為什麼這樣裁」的人 |

建議順序:`README`(本文)→ `01` → `02`/`03`/`04`(依關注面向)→ `05`(查裁決理由)。

---

## 4. 下一步:採納 ≠ 開工

**本資料夾不是開發指令。** 依 CLAUDE.md 協定,任何採納項目必須走 U-Net 流程,不可直接開發:

1. **必須 `/user` → `/po` 全新立案**(現有 08–13 規格均未涵蓋):
   - P0-1 讀回落地層 + append-only 變更日誌;
   - sidecar 預標寫入 handoff(dino_head/瑕疵分數 + M13 框,小案);
   - Golden 樣本庫的 bank 版本/血統/簽核(變更管理需求);
   - lot/wafer/die metadata schema(**先於任何 UI 重構**立案評估);
   - 隱藏三工具的結構性處置(PO 先在 ROADMAP 決策日誌記一行)。
2. **已規劃、仍待 User 放行**:08–13(pool_registry/round_ledger/gt_pred_diff/
   hybrid_sampler/probe_eval/heatmap_to_boxes)設計 + AC 已在 `3_Architect_Design/`
   備齊,但 User 指示未開工;放行後由 PO 記決策日誌、跑 `python verify/gate.py --snapshot`
   再走 `/pm` → `/pg`。09 需**補規格**:「作用中回合常駐顯示」為其自承缺口。
3. **既有元件重排/文案類**(P0-2、P0-5、P1 task selectbox、P1 直送鈕、P2 文案):
   規模小仍不豁免——由 PO 判定粒度(可併入相鄰 module 或開小案),同 PR 必掃
   `tests/e2e` 可見字串,涉及共用模組時記得 nativeApp vendored 副本雙向同步。
4. **不做清單具有約束力**:`05_debate_log.md` 所列不做項(挑樣 UI 合併、第七工具列項等)
   若日後想翻案,須帶新證據回到討論,不可默默實作。
