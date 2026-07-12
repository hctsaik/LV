# PRD:讀回落地層(Read-back Persistence)

對應需求:[1_user_needs/readback_persistence.md](../1_user_needs/readback_persistence.md) · 里程碑 M12
(AL Loop 收尾:讓迴圈真的閉合)

## 問題

`apply_readback`(labeling_handoff.py)只 mutate 記憶體中的 `records[i]["label"]`;重 Run 後
records 從資料夾重建,修正蒸發。迴圈的「回讀」這段等於白做。這是 `5_active_learning_product_review`
與 `0_discussion` 一致認定的 **#1 P0**。

## 目標(這一輪要達成)

把讀回的修正落成**持久、可稽核、不碰使用者資料集**的覆蓋層,並在載入時自動套回。

## 範圍(MoSCoW)

- **Must**
  - 新模組 `readback_store`(純 stdlib,對齊 08/09 體質):以 **sha256** 為鍵的
    **append-only 變更日誌**(old/new/annotator/timestamp/reason/batch/round),與由它折疊出的
    **覆蓋層**(sha→最新修正標籤)。落 `.lv_cache`,**絕不寫使用者資料夾**(鐵則 [[no-dataset-writes]])。
  - GUI 接線:①「📥 套用讀回結果」成功後,把 changes 記入 `readback_store`;②Visualize Run 建完
    records 後,套用覆蓋層(修正跨 Run/重啟存活);③📥 分頁長出「修正稽核」小區(列出/匯出變更日誌)。
- **Should**
  - 「建立資料版本(含修正)」動作:把變更日誌收成一份帶版本 id 的**修正清單**產物(CSV/JSONL)於
    `.lv_cache`,作為可交接、不可變的版本 delta。
- **Could**
  - 覆蓋層徽章:被覆蓋的圖在畫廊標「已修正(原:X)」。
- **Won't(本輪)**
  - 全影像重匯出成新資料夾(那條走既有 cart 匯出,不在本模組)。
  - 覆蓋層的衝突解消 UI（多來源改同一 sha）——採「最新 append 勝」,不做互動式合併。
  - 框幾何(bbox/polygon)的讀回落地——本輪只落**分類標籤**;幾何配 M13 預標框另議。

## 模組分解

- **M12 = 14 `readback_store`**(Tier B:核心純函式 Tier A + GUI 接線 Tier B)。
  相依:被 app.py 送標讀回(labeling_handoff apply_readback)與 Visualize Run 呼叫;概念上接
  09 round_ledger 的 record_readback(同一次讀回,round 記帳 + 修正落地並存,不互斥)。

## 成功門檻

- 單元 gate `python verify/gate.py readback_store` 印 GREEN。
- 真實 E2E:讀回改標 → 重啟 server → 重 Run → 該圖顯示修正後標籤;原始資料夾 label 檔 byte 不變;
  變更日誌可讀可匯出。

## 放行程序

PO 核 PRD → `/architect` 出 14 設計含 AC → `/pm` 落紅測 → `python verify/gate.py --snapshot` → `/pg`。
