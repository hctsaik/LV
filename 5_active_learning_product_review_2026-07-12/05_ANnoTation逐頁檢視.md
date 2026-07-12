# ANnoTation 逐頁檢視

## 現況摘要

主 Sheet 是四步：資料來源 → 標注工作台 → 審查 → 匯出／回傳。但 manifest 另有多個 enabled 功能未進主流程，並保留 Data Feeder、舊 Annotation、Sync Back 等舊語彙，因此「功能存在」不等於「使用者找得到」，也不等於「責任只有一處」。

## 主流程頁面

| 頁面 | 現況 | 模糊／重疊 | 建議 |
|---|---|---|---|
| 資料來源（module_026） | local/remote/iWISC，建立 DatasetManifest；可接 LV handoff、轉換 YOLO seed、帶入 classes | 多 round 時偏向自動取 newest pending，缺少批次 inbox、owner、任務理由與進度；其他模組仍使用 Data Feeder 舊名 | **保留為唯一 intake，升級概念為 Round Inbox**；讓人選 batch，顯示來源、reason、task、priority、owner；`manifest_id` 綁定唯一 `handoff_id` |
| 標注工作台（module_012） | schema、快速分類、X-AnyLabeling、canvas、AI pre-label、confidence filter、QA 快捷操作、瀏覽與進度都在同頁 | 與 module_016 AI Pre-labeling、module_009 統一平台及 module_018 Gallery 部分重疊；標註者可見過多模型工程決策 | **收斂為 Annotate**：只執行／修正任務。模型、策略與批次推論由 CV 在 round 設定；標註者可調整預標但不任意換模型 |
| 審查（module_018） | approved/rejected/pending、備註、可開回 X-AnyLabeling 修正 | `.review.json` sidecar 與 domain core/DB review state 同時存在；尚未追完整同步資料流，屬概念風險，未證實已分歧；另未明確禁止自己審自己 | **保留並擴為 Review & Adjudicate**；先追查 canonical source 與衝突優先序。若確有雙寫，再以 DB/core `ReviewDecision` 為真相、sidecar 作交換 artifact |
| 匯出／回傳（module_014） | 格式轉換、split、iWISC delivery、LV handoff closeout | 「產生檔案」和「完成回合交付」混在一起；目前一次 export 可能把 registry 中所有 open LV handoff 標成 read_back，而非只關閉當前 batch；部分 label validation 只是 warning | **同頁拆兩個明確動作**：「建立資料版本」與「提交本回合」。提交前過 QA gate，只更新當前 handoff，禁止 export 等同 applied |

## 流程外功能的處置

| 功能 | 問題 | 建議 |
|---|---|---|
| module_013 Sync Back | 與 module_014 iWISC 回傳重疊 | 收斂到 Deliver Round 的 destination/action，不再有第二個交付真相 |
| module_015 Dashboard | 與 module_017 前半統計重疊 | 合併為 Engineer Monitor：round throughput、退回率、coverage、策略收益、drift |
| module_017 Label Manager | 同時放 Dashboard、QA、Sync、label governance，名稱無法表達責任 | 拆出 Governance：taxonomy、alias/merge、guideline/schema version、migration impact；監控與交付回到各自主頁 |
| module_016 AI Pre-labeling | 與工作台內逐圖／整批 AI 預標重疊 | 模型與批量推論歸 round setup；工作台只顯示／修正預標 |
| module_009、010、011、019、022–025 | 新舊入口與停用流程並存 | 先建立保留／移轉／刪除決策表，清掉所有仍指向停用模組的導引，不讓使用者猜入口 |

## 兩端契約的 P0 風險

1. **錯關批次**：ANnoTation 匯出時不可把所有 open handoff 一次標成 `read_back`。狀態應至少拆成 submitted、reviewed、applied、rejected，且只作用於目前 `handoff_id`。
2. **Review 真相待查證**：已確認 sidecar 與 DB/core 機制並存，但尚未證實兩者會獨立分歧。先追查寫入、讀取、同步與衝突路徑；若兩者都是權威來源，再收斂至 DB/core canonical truth。
3. **Detection read-back 不完整**：LV 目前偏向只讀第一個 shape 的 label；多框／polygon 任務必須回收全部 geometry、attributes 與 review decision。
4. **缺 round lineage**：handoff 需補 `round_id`、model/data/bank/schema version、selection strategy/score、priority、required annotation type 與 QA policy。
5. **缺角色與時限**：主 local/LV 流程需有 Round Owner、Annotator、Reviewer、due date、instruction version，並能避免標註者自審。

## 建議目標資訊架構

```text
Round Inbox
  → Annotate
  → Review & Adjudicate
  → Deliver Round

Engineer Monitor（跨 round 監控）
Governance（schema / guideline / taxonomy）
```

這個結構讓 ANnoTation 專心處理「任務執行與品質」，VisualLatent 專心處理「挑樣、模型與回合成效」，兩者以明確 batch contract 串接。
