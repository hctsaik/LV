# Export Subset / 我的子集 — 設計文件

> 2026-06-21 · **匯出子集仍屬設計；React 持久 cart 已完成第一個 Visualize 切片，完整功能尚未完成**。
> 承接:[匯出清單 × 組考卷設計討論](export_list_quiz_design_discussion.md);守 `defect_*.md`「探索與治理分離」「誠實命名」紀律。
> 重用:`scripts/labeling_handoff.py`(已會把 sha256 命名的子集影像複製到資料夾)、`scripts/manifest.py`(每圖 sha256/phash/labels,存 `.lv_cache`)。
> 硬約束:**no-dataset-writes** —— app 只寫 `.lv_cache`;絕不寫回使用者資料集。

---

## 0. 定位（一句話）

把使用者在各工具(Visualize/完整度/物件覆蓋/灰帶覆核/組考卷)陸續挑出的資料,**彙整成單一視圖,並能原封不動匯出成一個 subset 子資料集** ——
> 「它幫你把挑好的東西原封不動搬出去,絕不偷改、絕不偷丟、每一筆都記帳;你的資料集,它一個位元都不碰。」

**這頁是 cart 這個唯一收集器的「臉」+ 一個新的 export-to-folder 出口,不是第二個收集器。**

### 現況校正（Round 1 查證）
目前的 Streamlit `viz_export_list` 是 `session_state` dict，只跨同一 session 的 rerun／Run 保留；`scripts/app.py` 的 `output/curation_log.jsonl` 記錄選取歷史與理由，不是購物車。React M24 現在將 Visualize 手選項目存入 `LV_CACHE_DIR/curation_log.jsonl`，可跨 worker 重開讀回。Streamlit session cart 尚無自動匯入流程；CSV／JSON 匯入與其它工具接入仍待實作。**目前沒有「export to folder」,也沒有 dst-in-source 防護** —— no-dataset-writes 目前靠既有輸出固定導向 `.lv_cache` 的設計保證,不是 runtime 檢查。本功能第一次讓使用者把任意寫入路徑交給 app,故 dst 防護是新的硬需求。

---

## 1. 目的（Purpose）

跨工具挑出的資料散落、無單一歸宿、無法一鍵收成資料夾。本功能:**統一收集 → 單一視圖複查 →（明確發起的）export-to-folder 成可用 YOLO 子集**,全程不碰原資料集。

---

## 2. 核心不變式 INV-1（單一 store・release gate）

**React cart 的唯一 on-disk store = `LV_CACHE_DIR/curation_log.jsonl`**。React UI 不維護另一份持久化清單。Streamlit session state 與它是兩個尚未完成退場／匯入的執行路徑；Streamlit 的 `output/curation_log.jsonl` 是另一種選取歷史日誌，不能當成 cart 讀取。

- **INV-1**(release gate):刪掉/清空 `curation_log.jsonl` 後,「我的子集 / Export」頁面必須**一無所有**(零筆/零類/零來源)。
- **INV-1a**:磁碟上不存在第二份持久化收集清單(grep 不到第二個寫檔點)。
- **INV-1b**:各工具「加入子集」鈕收斂為**唯一一個 `cart.add()` 寫入口**。
- **INV-1c**:頁面只**讀** store + 提供 export 出口,**無自己的 add/remove 私有狀態**(它是視圖,不是收集器)。

### 2.1 Streamlit session cart 遷移（尚未實作）
`viz_export_list` 沒有持久化副本，worker 關閉後無法自動讀取；也不能把 `output/curation_log.jsonl` 的選取歷史誤當成 cart。切換前需要明確的 session handoff 或使用者匯出後匯入流程，匯入需去重、列帳並顯示部分失敗。React 目前提供 CSV／JSON 匯出，但尚無匯入入口；此項仍是切換／發版前的資料延續工作。

---

## 3. 統一資料模型（SelectionSet）

`curation_log.jsonl`(append-only 事件日誌)折疊後的記憶體視圖 = `SelectionSet`。每筆:

React M24 目前已實作的 cart event snapshot 欄位為 `id`、`index`、`filename`、`path`、`label`、`split`、`sha256`、`source`、`score`、`reason`、`level`、`bbox`、`obj_index`、`added_at`。`id` 以正規化來源路徑產生，object level 再含 `obj_index`；SHA-256 用於匯出識別，不是目前 cart 去重鍵。下表的 `source_tool`、`source_tag`、`dataset_id`、跨來源 object ID 聯集與 `to_selection_set()` 仍屬完整跨工具／匯出資料模型，不能視為已由此切片實作。

| 欄位 | 說明 |
|---|---|
| `sha256` | **去重/身分鍵**(內容定址)。同圖不同路徑→同 sha256→去重一筆 |
| `source_tool` | visualize / completeness / objcov / graytone / examset |
| `source_tag` | 稀疏／離群／重複／分歧／候選(provenance,可多值聯集) |
| `reason` | 工具自動帶的判據(如「距質心 3.2σ」「dup-group #14」「分歧 0.7」) |
| `level` | image / object |
| `object_ids` / `bbox` | level=object 時,挑中的框(聯集合併,不覆蓋) |
| `dataset_id` | 來源資料集(manifest 的 `<name>_<hash>`) |
| `src_path` / `rel_path` | 複製來源 / 還原 YOLO 結構用(provenance,非鍵) |
| `added_at` | 時間 |

**sha256 歸一化只發生在 `to_selection_set()` 這唯一邊界**(cart 內部維持 path-keyed,避免高風險重 key);sha256 從 manifest 查,**查不到 → `unresolved[]` → export 擋下,不靜默丟**(留 `TECH-DEBT(D3)` 註記)。

---

## 4. 統一入口「加入子集」

React M24 現在只有 Visualize 的手動選取按鈕接入該 store，透過 run ID／record index 建立 snapshot；同一路徑重複加入會略過且保留第一次 provenance。清單頁已提供來源篩選、理由、單筆移除、確認清空、CSV／JSON／ZIP 下載。完整度、物件覆蓋、灰帶覆核與組考卷尚未接入；下列 UI 命名與自動 provenance 是目標契約，非已完成狀態。

- 各工具的收集鈕**統一命名「加入子集」**,餵同一個 store(INV-1b)。觸發點:縮圖角落 `+ 加入子集`(已加入→`✓ 已在子集`,再點移除)、表格列尾/表頭、批次 action bar。
- **加入時自動帶 provenance**(source_tool/tag/reason/level/bbox),使用者**不可手填**(保證可信)。
- 回饋用 `st.toast`(不彈 modal、不跳頁);重複加入同一筆只提示、不重寫。
- **與灰帶處置的關係**:被標**排除(exclude)的項硬擋不可加入**(hover 提示先取消排除);**灰帶(borderline)**可加入但帶琥珀警示與邊框。

---

## 5. export_subset 契約

```python
def export_subset(selection: SelectionSet, dst: Path,
                  mode: "copy"|"symlink"|"manifest-only",
                  layout: "yolo"|"crop-out") -> ExportReport
```
framework-free（純 stdlib,可單測);**`labeling_handoff` 反過來成為它的 caller(layout=labeling-handoff)** —— 一份 copy 引擎,多個出口。

- **mode**:**copy(預設,Windows 唯一可靠)** / symlink(進階,偵測 Windows 無權限即反灰) / manifest-only(只出清單 CSV/JSON,不搬位元組)。
- **layout=yolo**:`images/ + labels/ + classes.txt + data.yaml`,**保留 train/val/test split**;檔名預設 sha256(原名留 manifest);多來源走 **class_remap**(以類別**名稱**union 合併→新連續 id;同名不同義需使用者確認,**衝突未解擋下 export**)。
- **dst-containment 硬擋(不可關)**:`dst.resolve()` 不得等於、也不得是任一來源資料集根的後代(反向亦然);命中即拒絕並說明「會污染你的資料集真值」。
- **絕不靜默覆寫(不可關)**:dst 非空 → dry-run 顯示衝突,**預設 skip(計入 `skipped[]`)**;改名/覆寫須使用者顯式選。
- **強制 provenance manifest**(`manifest.csv` 給人看 + `lineage.json` 給機器重建,含 `unresolved[]`/`skipped[]`/class_remap/挑選理由/模式旗標)+ **偏斜警告** + **帶時間戳資料夾名**。
- **`ExportReport` 全列帳**:`{exported, skipped, unresolved, deduped, dst}` —— 不靜默吃資料。

---

## 6. object-level（兩條乾淨路徑）

- **① 整圖 + 全 labels(預設)**:box 選擇只記進 manifest/lineage,**label 檔原封不動**(訓練語義正確、零竄改真值)。
- **② crop-out**:每框裁成獨立影像 = 乾淨的**物件/分類集**,`task: classification`(≠detection),佈局 `images/<class>/<sha>_<boxidx>.jpg`(ImageFolder 原生),附**物件級 manifest**(crop_sha256/source_sha256/bbox/class/pad/source_tag),context **pad 預設 0**(可選、clip 邊界、寫入 manifest)。
- **「selected-boxes-only 改寫 label 丟掉沒挑的框」永久禁止於資料夾匯出** —— 會把沒挑的真物件變背景 = 系統性負標籤污染。要選框改真值,只能走 **labeling_handoff 任務規格** 或 manifest-only 清單。

---

## 7. UX 版面

### 7.1「我的子集」頁面（cart 的視圖,零暫存清單）
- **總覽帶**(sticky):總數 / image vs object 分計 / 跨幾工具 / 估大小(隨 export 設定動態);**去重橫幅**(sha256 精確重複給數字 + 檢視;phash 近似僅警示)、**偏斜橫幅**(最大類別占比超門檻常駐)。
- **控制列**:分組(預設依**來源工具**,次維度 source tag facet 多選 chip)、檢視(縮圖牆/逐類別表)、排序。
- **縮圖卡**必顯「**從哪挑**(來源工具+tag 角標)/**為什麼挑**(reason)」+ 重複標記;object-level 摺成**底圖 + N 框**徽章,並標當前搬「整圖」還是「裁切圖」。

### 7.2 Export 對話框（強制 dry-run）
目的地(新/空資料夾,指到原資料集內→紅字硬擋、確認鈕反灰)、mode、layout、object 搬運模式、衝突處置;**dry-run 預覽**(寫幾張/幾 label/總大小/衝突,改任一設定即重算)、**確認按鈕帶實際數字**、**偏斜警告常駐**。

### 7.3 管理
移除單筆(可復原)、清空(打字二次確認)、依來源/tag 篩選、去重視圖(sha256 group 聚合;phash 僅人工逐組決定不自動刪)、object vs image 分計呈現。

---

## 8. 驗收標準（給 /pm）

React 的第一個 cart slice 已有 API 與可見 UI 證據，包含不同 worker PID 重開讀回；目前只涵蓋 Visualize 手選，未完成 Streamlit session 匯入、其它工具共用、`export_subset` 資料夾匯出與目的地保護。以下仍是整個「我的子集」功能的驗收門檻，不能以 cart UI 已完成替代。

1. **INV-1 release gate**:刪 `curation_log` → 頁面一無所有;+ INV-1a/b/c(無第二份持久化、單一 `cart.add()` 寫入口、頁面無私有 add/remove 狀態)。
2. **一次性遷移**:升級時 viz_export_list 一次性匯入 curation_log(去重)、列帳、完成後停讀封存;部分失敗 surface 不靜默。
3. **dst-containment 硬擋**:dst 落在任一來源資料集內(雙向)→ 拒絕,不可關。
4. **絕不靜默覆寫**:dst 非空預設 skip + 列帳;覆寫須顯式確認。
5. **不靜默吃資料**:unresolved(查不到 sha256)→ 擋下 + 列 manifest,不給「照樣輸出」逃生門;ExportReport 全列帳。
6. **object-level**:資料夾匯出**永不輸出子集化/被改寫的 label .txt**;crop-out 走 `task:classification` 且 lineage 可還原(原圖+box+pad)。
7. **多來源 class_remap**:衝突未解 → 擋下;不重疊類各自獨立列(缺席側 0,不冒充「召回掉 0」)。
8. **去重**:sha256 精確去重(自動);phash 近似**僅警示,絕不自動刪**。
9. **可重現**:光看 `lineage.json` + 來源還在 → 能 byte-for-byte 重建這個 subset。
10. **偏斜警告**:匯出物必帶「刻意偏斜的診斷子集,直接訓練/評估會誤導」警語 + 帶時間戳資料夾名。

---

## 9. 範圍（MoSCoW，二元收斂)

**Must**:① 跨工具集合視圖(單一 store,滿足 INV-1)② 統一「加入子集」③ export-to-folder(`export_subset`,copy 預設、yolo layout、sha256 去重、class_remap、dst 硬擋、provenance+偏斜警告)④ object-level 整圖 & crop-out ⑤ 一次性遷移。

**Won't**:① selected-boxes-only 改寫 label ② 寫回原資料集 ③ 第二個 store / 影子清單 ④ COCO/VOC 格式轉換 ⑤ N 路 ⑥ 集合頁內二次改標。

---

## 10. 與 no-dataset-writes 對齊

收集只寫 `.lv_cache`(curation_log);export 只寫**使用者指定的新 dst**,dst-containment runtime 雙向硬擋確保不碰來源;近似重複僅警示不自動刪;一切動作列帳不靜默 —— 全程零位元寫回使用者資料集。

---

## 11. 剩餘風險（PG 階段,由驗收/e2e 兜底）

Windows symlink 權限、大資料 copy 的時間/空間、跨來源 classes.txt 同名不同義的人工對映、phash 門檻的業務語意、偏斜 subset 被誤用於 benchmark(靠警語+獨立 hold-out 文件提醒)。
