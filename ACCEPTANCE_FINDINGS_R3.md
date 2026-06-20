# 驗收報告 R3：/loop 五輪（驗收 → 找問題 → 修）

> 日期：2026-06-20　|　接續 R1/R2
> 模式：每輪聚焦一個尚未深挖的區域，用 valid/test 真實數據走查，找到真問題就**當場修**；
> 沒有真問題就**誠實記錄健康、不為修而修**。
> 真實資料：valid（55 圖/309 物件）、test（107 圖/550 物件）、10 類，valid 無 couch。

---

## 五輪總覽

| 輪 | 區域 | 結果 |
|---|---|---|
| 1 | Compare Distributions | 🔴 找到並修：**整類差異盲點** |
| 2 | Visualize（搜圖/洩漏/離群） | ✅ 驗證健康（無 bug） |
| 3 | 邊界/錯誤處理 | 🔴 找到並修：**評估巢狀佈局類別解析** |
| 4 | QA 鏈（組考卷↔評估契約） | ✅ 驗證健康（無 bug） |
| 5 | 系統性/橫切 | 🔴 找到並修：**類別解析深度不一致** |

3 個真修復、2 個驗證健康。

---

## Loop 1 — Compare：整類差異被靜默忽略 🔴→✅

**問題**：`_compute` 用 `classes = set(la) & set(lb)`（**只取交集**）逐類別比漂移。
valid 無 couch、test 有 58 個 couch → couch 不在交集 → **drift 表完全沒有 couch**。
使用者比 valid vs test，看不到「test 多了一整類 couch」這個**最大的分佈差異**，只有一句
「類別名需一致」。

**修**：計算 `a_only`/`b_only`（一邊有、另一邊完全沒有的類＋數量），存 session，
在漂移表上方用警示列出「只在 A：…／只在 B：couch(58)…」。實測 b_only=`couch`。

## Loop 2 — Visualize：驗證健康 ✅

- **以文搜圖**：非 CLIP 模型下**其實有提示**「切換 chinese-clip 即可使用」（line 1281）
  ——R1 的 finding I 是誤判，撤回。
- **重複/洩漏掃描**：`splits` 取自資料夾、`cross_split_only` 過濾邏輯正確；
  實測 valid↔test phash（hamming≤4）**0 洩漏**、0 完全重複 → 資料乾淨、功能正確。
- 無需修改。

## Loop 3 — 邊界/錯誤：評估在巢狀佈局解析不到 classes.txt 🔴→✅

**問題**：評估用 `read_classes_txt(folder) or read_classes_txt(folder/"_")` 解析類別名，
這兩個探的是 `[Small]/classes.txt` 與 `test/classes.txt`——**都不存在**（classes.txt 在
`indoor/`，第三層）。結果 `names=[]` → GT 類別退化成 id 字串；若預測 CSV 用類別**名**，
class-aware 比對 `"7"!="couch"` → **全部誤判為漏抓、recall 假性=0**。

**修**：加 `folder.parent` 探測（後續併入 Loop 5 的共用 helper）。

**順帶驗證已 guard、非 bug**：KMeans 區塊 `kk=max(1,min(k,len))`（2 物件不崩）；
投影 small-N 有 `n_samples<4` guard；參照分佈聚合表的一鍵補只列 `未覆蓋>0` 的類。

## Loop 4 — QA 鏈：組考卷↔評估共識契約健康 ✅

組考卷匯出共識 CSV 的 `consensus` 欄是 Python bool→字串 "True"/"False"。
經典陷阱是消費端 `bool("False")==True`。但 `consensus_flags` 用
`truthy={"true","1","yes","共識","consensus"}` 比對（line 104），"False"→"false"∉truthy
→**正確**。box-level 以 IoU 對回 GT。契約健全，無需修。

## Loop 5 — 系統性：類別解析深度全 app 不一致 🔴→✅

Loop 3 的巢狀佈局暴露出更大的問題——「資料夾→類別名」各工具探測深度不同：

| 位置 | 原本探測 | 巢狀佈局([Small]) |
|---|---|---|
| `_cov_class_names`（完整度/objcov） | root, root.parent（2 層） | ✅ 找得到 |
| Visualize 自動偵測（2623） | `folders[0]` 只 1 層 | ❌ 漏 |
| 評估（6290） | folder, folder/"_" | ❌ 漏 |
| 評估 demo（6217） | folder, folder/"_" | （demo 剛好 OK） |

同一份 [Small] 資料，在完整度找得到類別、在 Visualize 卻偵測不到——體驗不一致。

**修**：抽共用 helper `_classes_txt_nested(folder)`（依序探 folder.parent/、folder/、
folder.parent.parent/ 的 classes.txt），把 Visualize 自動偵測／評估／評估 demo **三處統一**
走它。實測 `_classes_txt_nested(valid)` → 正確回 10 類。

---

## 驗證

- compile + import OK；`_classes_txt_nested(valid)` 回 10 類。
- 回歸（各自單獨跑全綠）：cov 10、objcov 11、cmpc 6、compare_by_class 6。
  （合併跑時 compare_by_class 2 紅＝既有測試隔離污染：cov/objcov harness 全域 patch
  `st.plotly_chart`，與本次修改無關。）
- 順手修了一個過時測試斷言（cmpc 投影方法 options 已擴含監督法）。

## 一句話總結

這輪把焦點從「功能夠不夠」轉到「**真實資料下會不會出錯**」。最有價值的發現是一條
**橫切主題**：好幾個工具各自實作「資料夾→類別名」，深度不一，巢狀佈局（…/[Small]/split）
會在某些工具靜默失效。收斂成單一 helper 後，類別解析在全 app 一致。Compare 的整類差異
盲點則和 R2 的完整度盲點同源——**系統擅長比「共同的部分」，卻容易對「一邊獨有」這種
最大差異視而不見**，這輪也補上了。
