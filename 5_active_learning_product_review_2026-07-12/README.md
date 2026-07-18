# VisualLatent Active Learning 產品檢視

整理日期：2026-07-12  
範圍：`visuallatent` 全部可見與隱藏頁面，以及 `C:\code\claude\ANnoTation` 的標註交接面。  
狀態：**討論稿，未開發、未修改產品程式碼。**

## 一句話結論

目前最需要的不是再增加分析頁，而是把「為何挑這些樣本 → 誰要做什麼 → 標到哪裡 → 回來後改了什麼 → 這一輪是否值得」串成可追溯的 Active Learning 回合。

## 建議閱讀順序

1. [00_結論與建議.md](00_結論與建議.md)：本次結論的單一 Markdown 總整理。
2. [01_逐頁檢視與合併建議.md](01_逐頁檢視與合併建議.md)：每個頁面的定位、模糊點、保留／合併／移轉建議。
3. [02_目標工作流與角色邊界.md](02_目標工作流與角色邊界.md)：User、標註者、領域專家、CV 工程師如何協作。
4. [03_半導體工業檢測情境.md](03_半導體工業檢測情境.md)：真正適合 Active Learning 的現場情境與必要資料。
5. [04_優先序與待決策事項.md](04_優先序與待決策事項.md)：先後順序、非目標，以及開發前需要確認的產品決策。
6. [05_ANnoTation逐頁檢視.md](05_ANnoTation逐頁檢視.md)：標註端每頁責任、重疊與 LV 交接風險。
7. [06_開發交接指南.md](06_開發交接指南.md)：給接手開發 AI 的任務卡（T0–T6）、已驗證現況快照、資料契約對表與硬規則；放行開發後從這份開工。
8. [07_review_truth_trace.md](07_review_truth_trace.md)：T6 唯讀調查——ANnoTation review 權威真相追查。**修正 05 文件的「P0:Review 兩套真相」定性**：sidecar 與 DB 從不雙寫、不會漂移；真問題是匯出不 gate 於 review。
9. [08_平台概念差距分析.md](08_平台概念差距分析.md)：比對平台策略補強稿與現有設計，區分已有、部分已有與真正缺少的概念，並裁決 Registry／Audit、Promotion Gate、Backbone、Storage 與治理的先後順序。**2026-07-19 更新**：合併第二份獨立分析（A/B/C 尺度分類、單機最小落地形態、反向盤點）＋第三方逐條程式碼查核附錄；修正 read-back 過時項。
10. [09_半導體分析與Active_Learning下一階段功能.md](09_半導體分析與Active_Learning下一階段功能.md)：在 M15–M20 已完成的基線上，重新盤點半導體 entity、group/time split、selection bias、風險選樣、校準拒判、Unknown／taxonomy／bank 生命週期，以及 wafer／multi-view 等下一階段缺口。**2026-07-19 合併**：併入第二份獨立盤點（群組批次標註、Tool/Recipe 顯著性與成本工作點兩個 quick win、植入式驗證設計原則、最小版設計表）；兩份獨立收斂＝缺口可信。

> **開發進度(2026-07-12,User `/goal` 放行後)**：T0 gate 治理、T1 文案/死碼、T2 M9 GUI 接線（pool+round
> 進送標/📥,真實 E2E 7/7 綠）、T4 ANnoTation closeout 修正、T5 送標顯性化+瑕疵③直送、T6 review 調查、
> **T3 讀回落地層（M18 `readback_store`,gate 11 綠＋E2E 3/3）— 全部完成**。詳見 `ROADMAP.md` 決策日誌。
11. [KICKOFF.md](KICKOFF.md)：貼給接手 AI 的開工提示詞（非正典；含今日任務解鎖狀態快照與操作迴圈，交付前先核對是否過期）。

## 本次盤點範圍

- 可見頂層工具：Visualize Embeddings、Compare Distributions、完整度熱力圖、瑕疵偵測、匯出、📥 標註回饋。
- 程式仍存在但導航隱藏：組考卷、灰帶覆核、評估。
- 標註端：任務取得、影像／框標註、匯出、VisualLatent handoff 與 read-back。

## 主裁決

- 不把 Visualize、完整度與瑕疵偵測的操作 UI 生硬合併；三者目的分別是探索、覆蓋補洞、異常風險排序。
- 合併共享的候選資料契約、去重、取樣策略、送標狀態、讀回與 round ledger。
- 將「📥 標註回饋」提升為「標註迴圈」中樞，不新增另一個頂層頁面。
- 「組考卷」的派件與多人重複標註移到 ANnoTation；LV 保留一致性分析。
- 「灰帶覆核」併入標註迴圈的裁決佇列；「評估」併入每輪成效報告。
- 半導體導入前，先讓 `lot / wafer / die / tool / chamber / recipe / layer` 成為一等欄位，而不是依賴資料夾命名。
- ANnoTation 的「匯出資料」與「提交／完成 Active Learning 回合」必須拆成兩個語意，且只能關閉當前 handoff。

## 與既有文件的關係

本資料夾是本次重新核對後的產品決策摘要；保留既有 `0_discussion/` 詳細盤點，不覆寫使用者現有草稿。實作前仍應回看其中的檔案／行號證據與既有設計文件。

治理關係：`0_discussion/05_debate_log.md` 保存主席裁決、反方論證與「翻案須帶新證據」的完整脈絡；本資料夾是較新的正典摘要。若兩者衝突，先標出差異並回到 ROADMAP／設計契約決策，不可靜默選一份。
