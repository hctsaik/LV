# 07 — ANnoTation「審查決定(review)」權威真相追查

> 2026-07-12 唯讀調查產出(對應 `06_開發交接指南.md` T6)。**只讀不改任何程式碼。**
> 目的:釐清 `05_ANnoTation逐頁檢視.md` 標為「P0:Review 有兩套真相」的風險——`.review.json`
> sidecar vs DB `ReviewDecision` 到底哪邊權威、會不會分歧。所有主張附 `file:line`(路徑相對
> `C:\code\claude\ANnoTation`)。

## 本文件回答什麼

1. sidecar 與 DB 兩套 review 各自的生命週期(誰寫、誰讀、做什麼)。
2. 兩者是否雙寫、是否有同步、是否會分歧——**證實/未證實**分清楚。
3. 對「另一個修正任務(匯出/交棒是否尊重審查)」的意涵。

## 一句話結論

**這不是「同一筆記錄兩套真相會漂移」的 bug,而是「兩套互不相通的系統各管各的」。** 對使用者實際
操作的 Streamlit 審查流程而言,**唯一權威真相是 `.review.json` sidecar**;DB 的 `ReviewDecision`
屬於另一套只有 MCP 入口、GUI 完全不碰、且**整個 repo 只寫不讀**的平台領域層(真正被讀的是
`annotation_set.state`)。兩者**從不雙寫、無同步、無交集**,所以**不會就同一筆記錄互相覆蓋而分歧**。

## 兩套系統對照

| | 系統 A:`.review.json` sidecar | 系統 B:DB `ReviewDecision` |
|---|---|---|
| 使用者實際入口 | ✅ Streamlit「🖼️ Review Gallery」(module_018) | ❌ 僅 MCP tool,無 GUI |
| 儲存 | 影像旁 `<影像名>.review.json` | `annotation_workspace` 內另一個 sqlite 的 `review_decisions` 表 + annotation_set 狀態機 |
| 讀取端 | module_017 QA 統計、module_018 畫廊 | **全 repo 無任何讀取者**;真正被讀的是 `annotation_set.state` |
| module_014 匯出是否讀 | ❌ 不讀 | ❌ 不讀 |

## 1. `.review.json` sidecar 生命週期

- **路徑**:`module_018/018_output.py:54-55` `_review_path()` = `<影像父目錄>/<影像名>.review.json`
  (寫在**使用者資料夾內**、影像旁)。
- **唯一寫入點**:`018_output.py:68-78` `_set_review()`,欄位 `status`(approved/rejected/pending,
  見 `:48`)、`comment`、`reviewer`(= OS 登入帳號 `_current_reviewer()` `:39-45`,刻意不接平台
  auth)、`timestamp`。原子寫(`.tmp`+`os.replace` `:76-78`)。
- **UI 動作**(全在 module_018 Review Gallery):核准/退回(`:362-368`、`:427-436`);↩️重置=**直接
  刪 sidecar 檔**(`rp.unlink()` `:379-381`)。
- **讀取者**:`_get_review()`(`:58-65`,檔不存在預設 pending)。被 module_018 自身(計數 `:289-290`、
  狀態邊框 `:156-157/214-220`、面板 `:350-358`)與 **module_017 QA 統計** `_render_review_summary()`
  (`017_output.py:192-278`)讀。**關鍵細節**:module_017 的 item 清單來自 **manifest DB**
  (`:200` `_mdb.get_manifest_items`),但 review 狀態值逐一到影像旁讀 **sidecar**(`:213-216`)。
- **用途**:純顯示/統計(QA dashboard),**不影響匯出、不影響 handoff closeout**。

## 2. DB `ReviewDecision` 生命週期

- **定義**:`domain/core/models.py:294-306`,欄位 `target_type`(annotation_set/task/export)、
  `decision`(approved/rejected/**changes_requested**——注意與 sidecar 的 pending 不同)、`actor_id`、
  `comment`、`decided_at`。狀態機 `domain/core/states.py:6-13`。
- **寫入**:`states.py:33-54` `apply_review_decision()` 轉 annotation_set 狀態並回一個 ReviewDecision;
  `domain/services.py:497-502` `review_task()` 呼叫它 + `save_review_decision()`(存 review_decisions 表,
  `sqlite_store.py:428-441`,表定義 `:97`)。**唯一對外入口是 MCP**:`mcp/server.py:110-111` →
  `mcp/handlers.py:200-201`;**沒有任何 Streamlit module 呼叫 review_task**。
- **讀取**:`review_decisions` 表**全 repo 只有 save、無任何 SELECT 讀取者**——它實質是**稽核日誌**。
  DB 路徑真正「讀來做決定」的是 `annotation_set.state`,例:`services.py:642` `create_export()` 在
  purpose=training/publish 時 gate `state != "approved"`。此路徑的實體 DB(`SqliteMetadataStore`,位於
  `CIM_LOG_DIR/annotation_workspace`,見 `module_022/022_process.py:18`)**與 module_014/017/018 用的
  `CIM_LOG_DIR/db/manifest.sqlite` 是不同檔案**。

## 3. 雙寫?同步?分歧?

- **雙寫**:否,只寫其一。module_018 動作→只寫 sidecar;MCP review_task→只寫 DB。沒有一行把一次
  動作同時落兩邊。
- **讀取端**:QA 統計/畫廊讀 sidecar;DB 匯出閘門讀 annotation_set.state。**無「sidecar 蓋過 DB」或
  反向**——各讀各的。
- **同步**:**不存在**。全 repo 無任何程式把 sidecar sync 進 DB 或反向。manifest.sqlite schema
  (`_manifest_db.py:18-90`)**完全沒有 review 欄位**,佐證 Streamlit 側 review 只能存在於 sidecar。
- **分歧**:
  - 「同一筆記錄互相覆蓋而漂移」= **不會**(不雙寫、無交集、無同步)。
  - 「兩套系統互盲」= **架構上必然成立**(若同一資料同時走兩條路,彼此看不到對方);但**未證實**
    是否真有流程把同一資料集同時餵給 module_018 sidecar 審查與 MCP annotation_set 審查——兩者操作
    的是不同物件模型(逐影像檔 vs annotation_set)與不同 DB,無證據顯示曾套用於同一底層資料。

## 4. 結論(對 05 文件「P0:Review 兩套真相」的修正)

- **權威真相**:使用者實際流程 = **`.review.json` sidecar 唯一權威**;DB `ReviewDecision` 是只寫不讀的
  平台稽核層,真正權威是 `annotation_set.state`。
- **是否分歧**:「數值漂移」**不會**;「兩系統互盲」**架構成立但未證實實際重疊**。因此 05 文件把它列為
  「P0 收斂 canonical truth」**過度定性**——現況不是收斂問題,而是「兩套系統是否該共存/是否有資料重疊」
  的產品問題。建議把它從「P0 收斂」降級為「待確認:是否有資料同時走兩條審查路」。
- **仍缺、待查**:(a) 是否有流程讓同一資料集同時走 module_018 sidecar 與 MCP annotation_set 審查;
  (b) modules 022–026 import 了 `AnnotationService` 但經查未呼叫 review_task/submit_for_review,其與
  sidecar 審查是否有間接關係尚未細追。

## 5. 附帶(對 T4/其他修正任務關鍵):module_014 匯出/交棒**完全不讀 review**

**明確答案:sidecar 與 DB 都不讀。** `module_014/014_process.py:682` `execute_logic()` 的資料來源只有
manifest items(`:740`)、分類結果(`:745`)、影像旁 `.json` 標註 shapes(`:758-764`,讀 `<影像>.json`
**不是** `.review.json`)。**被 module_018 標為 rejected 的影像照樣匯出**(COCO/YOLO/VOC/ImageFolder/CSV
全部)。`_retire_lv_handoffs()`(`:44-74`)只處理 LV 批次生命週期(標 read_back),**與 QA review 無關**。

> **這其實是比「兩套真相」更該優先的真問題**:review 對下游匯出**毫無效力**。若要讓匯出尊重審查,
> 主線(module_014 Streamlit 匯出)須接 review;依「使用者實際入口」判斷,應接 **sidecar**
> (`018_output.py:54-65` 的 `_review_path`/`_get_review`)。此為新需求,須另立案(不在本次修正範圍)。
> (注:另一條 MCP 匯出 `services.create_export` `:642` **有** gate `state=="approved"`,但那不是
> 使用者實際用的那條。)
