# 14 — readback_store 技術設計(讀回落地層:覆蓋層 + 變更日誌)

對應 PRD:[2_PO_PRD/readback_persistence_prd.md](../2_PO_PRD/readback_persistence_prd.md) · 里程碑 M12 · Tier A(核心)+ B(接線)

## 1. 目的(Purpose)

以影像 **sha256** 為鍵,把「標註讀回的修正」落成 **append-only 變更日誌**(不可變歷史、可稽核)
與由它折疊出的**覆蓋層**(sha→最新修正標籤),存在 `.lv_cache`(**絕不寫使用者資料夾**)。
Visualize Run 建完 records 後套用覆蓋層,讓修正跨 Run / 跨 app 重啟存活——閉合主動學習迴圈的「回讀」段。

## 2. I/O 契約

### 2.1 新模組 `scripts/readback_store.py`(純 stdlib;**不得 import numpy/streamlit/PIL**)

儲存:**append-only JSONL**,一行一「修正事件」。路徑 call-time 讀 env(對齊 pool_registry):

```python
def changelog_path() -> Path:
    """<LV_CACHE_DIR 或 <repo>/.lv_cache>/readback_changelog.jsonl(跨資料集全域)。"""
```

事件 schema(多餘鍵容忍;缺 `sha256` 或 `new_label` 的事件跳過、不計入):

```json
{"ts": "2026-07-12T12:00:00+00:00", "sha256": "<64hex>",
 "old_label": "<改前>", "new_label": "<改後>", "filename": "<供人讀>",
 "reason": "", "batch_id": "", "round_id": "", "annotator": ""}
```

API(全部純函式 + 檔案 IO 隔離在這一層):

```python
def record_corrections(changes: list[dict], *, batch_id: str = "",
                       round_id: str = "", annotator: str = "") -> int
    # changes 每筆至少含 sha256 + new_label(old_label/filename/reason 選填)。
    # 逐筆 append(自動補 ts=UTC isoformat);缺 sha256 或 new_label 的跳過。
    # 回實際寫入筆數。目錄不存在先建。

def load_overlay(shas: Iterable[str] | None = None) -> dict[str, str]
    # 折疊變更日誌 → {sha: 最新 new_label}(依 append 順序,後到覆蓋先到 = 最新勝)。
    # 空/檔不存在 → {};壞行(非 JSON / 缺鍵)靜默跳過。給 shas 只回命中的。

def apply_overlay(records: list[dict], sha_of: Callable[[dict], str | None]) -> int
    # 對每筆 record:sha = sha_of(rec);sha 命中覆蓋層且覆蓋值 != 現 label 時,
    # 就地設 record["label"] = 覆蓋值;回實際覆蓋筆數。sha_of 由呼叫端提供(保持本模組
    # framework-free、不自行算 sha)。sha_of 回 None → 該筆跳過。

def load_changelog() -> list[dict]
    # 全部合法事件、依 append 順序(供稽核視圖 / 匯出);壞行跳過。

def corrections_summary() -> dict
    # {"n_events": 合法事件數, "n_images": len(overlay), "by_new_label": {label: 圖數}}。

def export_version(out_dir: Path, version_id: str) -> Path
    # (Should)把目前變更日誌收成一份帶版本 id 的修正清單 CSV(欄:sha256,filename,
    # old_label,new_label,annotator,ts),寫到 out_dir/readback_version_<version_id>.csv。
    # 回檔案路徑。純讀日誌 + 寫 .lv_cache 內指定 out_dir,不碰使用者資料夾。
```

### 2.2 GUI 接線(Tier B,`scripts/app.py`)

- **記入(readback 成功後)**:`_record_readback_bookkeeping`(T2 已建)裡,對 apply_readback 回傳的
  changes **補算 sha**(每筆 change 的 records 路徑取 sha,重用 `labeling_handoff._sha_for`),
  呼叫 `readback_store.record_corrections(changes_with_sha, batch_id, round_id, annotator)`。
  annotator 取自 handoff 結果(`read_labeling_results` 的 annotator 欄,取眾數或第一個非空)。
- **套用(Visualize Run 後)**:records 建好、寫完 manifest 後,呼叫
  `readback_store.apply_overlay(records, sha_of=<manifest 命中或現算 file_sha256>)`。
  被覆蓋的圖 label 即為修正後值。**只動 records 記憶體,不寫資料夾**。
- **稽核視圖(📥 分頁)**:「🔁 回合」時間軸下方加「🩹 讀回修正(稽核)」小區:
  `corrections_summary()` 顯示「已修正 N 圖 / M 筆事件」,展開列 `load_changelog()` 表格,
  一鍵「⬇ 匯出修正清單」呼叫 `export_version(.lv_cache 版本夾, 時間戳版本 id)`。

## 3. 資料流

選樣 → 送標 → 標註 → 📥 套用讀回(改記憶體 records)→ `record_corrections`(落 `.lv_cache` 日誌)
→ …重 Run / 重啟… → Visualize Run 建 records → `apply_overlay`(修正復現)→ 畫廊顯示修正後標籤。

## 4. 邊界條件與錯誤處理

| 情況 | 預期行為 |
|---|---|
| 日誌檔不存在 | `load_overlay()=={}`;`apply_overlay` 不改任何 record;`record_corrections` 自建 |
| 壞行 | 跳過壞行、其餘照常 |
| 同 sha 多次修正 | 覆蓋層取**最新 append**;日誌保留全部歷史(可稽核) |
| change 缺 sha256 或 new_label | 不寫(不製造無鍵/空修正) |
| sha_of 回 None(算不出 sha) | 該 record 跳過(不覆蓋) |
| new_label == old_label | 仍可記(呼叫端通常已濾;本模組不強制) |
| 覆蓋值 == record 現 label | `apply_overlay` 不計入(no-op),避免假覆蓋數 |
| LV_CACHE_DIR 有設 | 日誌跟著進該目錄(與其他快取同根) |
| 使用者刪 `.lv_cache` 日誌 | 修正歸零但 app 不崩潰(等同從未修正);原始資料夾始終不受影響 |

## 5. Acceptance Criteria

**單元(`tests/test_readback_store.py`;tmp_path + monkeypatch LV_CACHE_DIR,不碰真快取)**

- **AC1(記入→覆蓋)**:`record_corrections([{"sha256":"a"*64,"old_label":"x","new_label":"y","filename":"f.jpg"}])`
  後 `load_overlay()["a"*64]=="y"`,回傳筆數==1。
- **AC2(最新勝 + 歷史保留)**:同 sha 依序 new_label "y"→"z" 後 `load_overlay()[sha]=="z"`;
  `load_changelog()` 長度==2(append-only,兩筆都在)。
- **AC3(apply_overlay)**:記入 sha=a→"z" 後,`records=[{"path":"/p/f.jpg","label":"x"},{"path":"/p/g.jpg","label":"w"}]`,
  `apply_overlay(records, sha_of=lambda r:{"/p/f.jpg":"a"*64}.get(r["path"]))` 回 1、`records[0]["label"]=="z"`、
  `records[1]["label"]=="w"`(未命中不動)。
- **AC4(append-only + 稽核欄)**:兩次 record_corrections(帶 annotator/batch_id/round_id)後檔案行數==Σ筆數,
  每行含 `ts`/`annotator`/`batch_id`/`round_id`/`old_label`/`new_label`。
- **AC5(壞行容忍)**:塞 `not json\n` 與缺 sha256 的合法 JSON 各一行後,`load_overlay()`/`load_changelog()`
  正常回其餘;不崩潰。
- **AC6(檔不存在)**:全新 tmp `load_overlay()=={}`;`apply_overlay([{"path":"x","label":"l"}], lambda r:"a"*64)==0`;
  `corrections_summary()["n_events"]==0`。
- **AC7(缺鍵跳過)**:`record_corrections([{"sha256":"a"*64}])`(缺 new_label)與 `[{"new_label":"y"}]`(缺 sha)
  都回 0、`load_overlay()=={}`。
- **AC8(no-op 不假覆蓋)**:overlay sha=a→"z";`apply_overlay([{"path":"p","label":"z"}], lambda r:"a"*64)==0`
  (現 label 已是 z,不計入)。
- **AC9(export_version)**:記入 1 筆後 `export_version(tmp,"v1")` 產出 CSV,含表頭 + 1 資料列
  (sha256/filename/old_label/new_label 欄值正確);回傳路徑存在。
- **AC10(無重依賴)**:乾淨子行程 `import readback_store` 後 `"numpy" not in sys.modules`。
- **AC11(filter shas)**:記 a、b 後 `load_overlay(["a"*64])` 只含 a。

**GUI / 整合(真實 E2E,`tests/e2e/test_readback_store_e2e.py`,`@pytest.mark.e2e`)**

- **AC-G1(跨重啟存活)**:走 cart 送標 → 對 handoff 寫 sidecar(改成 classB)→ 📥 套用讀回 →
  **重啟 server** → 重 Run Visualize 同資料夾 → 該圖在畫廊/記錄顯示 **classB**(overlay 套用生效)。
- **AC-G2(不寫使用者資料集)**:上流程跑完,原始資料夾**沒有**新增/變更任何 label 檔
  (該資料集本無 label 檔;斷言資料夾內容 byte 級不變)、`.lv_cache/readback_changelog.jsonl` 存在且含該筆。
- **AC-G3(稽核可讀可匯出)**:📥「🩹 讀回修正」顯示「已修正 ≥1 圖」;點「⬇ 匯出修正清單」產出 CSV(含該筆)。

## 6. 給 /pm 的提示

- 核心純函式不需影像檔:sha 用 `"a"*64` 假值即可(AC1-AC11)。
- E2E 重用 T2 的 `al_isolated_server` + `_al_loop_harness`(cart 送標 + 寫 sidecar + 套用讀回 + 重啟)。
  重啟後「重 Run」要能觀察到修正——斷言可讀畫廊縮圖標籤,或體檢卡/記錄區的 label 文字。
- 覆蓋層「最新勝」(AC2)與 apply_overlay「no-op 不假覆蓋」(AC8)是防 silent-wrong 的關鍵,務必釘死。
- 與 09 的 record_readback **並存不互斥**:同一次讀回,round_ledger 記帳 + readback_store 落地各記各的。
