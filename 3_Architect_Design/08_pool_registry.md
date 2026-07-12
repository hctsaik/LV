# 08 — pool_registry 技術設計(全域樣本池記帳)

對應 PRD:[2_PO_PRD/al_loop_infrastructure_prd.md](../2_PO_PRD/al_loop_infrastructure_prd.md) · 里程碑 M9 · Tier A(核心)+ B(接線)

## 1. 目的(Purpose)

以影像 **sha256** 為身分鍵的全域狀態機,讓「送標/已標/排除」狀態跨工具、跨回合、跨 app 重啟存活;
任何送標入口在送出前查帳,**預設不重複送標**,且把攔截的理由攤在 UI 上(不靜默丟棄)。

## 2. I/O 契約

### 2.1 新模組 `scripts/pool_registry.py`(純 stdlib;**不得 import numpy/streamlit**)

儲存:**append-only JSONL**,一行一事件。路徑:

```python
def registry_path() -> Path:
    """<LV_CACHE_DIR 或 repo 根/.lv_cache>/pool_registry.jsonl(跨資料集全域)。"""
```

事件 schema(一行一個 JSON object;多餘鍵容忍、缺必要鍵跳過):

```json
{"ts": "2026-07-04T12:00:00+00:00", "sha256": "<64hex>",
 "event": "sent" | "labeled" | "excluded" | "reset",
 "batch_id": "<handoff 批次 id 或空字串>", "tool": "<來源工具名>",
 "path_hint": "<當時的影像路徑,僅供人讀>"}
```

API(全部純函式 + 檔案 IO 隔離在這一層):

```python
def record_events(events: list[dict]) -> None
    # append 寫入;自動補 ts(UTC isoformat);目錄不存在先建。

def load_states(shas: Iterable[str] | None = None) -> dict[str, dict]
    # 折疊事件流 → **乾淨** {sha: {"state": str, "batch_id": str, "tool": str, "ts": str}}。
    # 鍵一律是 64-hex sha(不混入任何診斷欄位;空 registry / 檔不存在 → {})。
    # 狀態優先序(高蓋低):labeled > excluded > sent;"reset" 清回 unseen(之後的事件再累積)。
    # 同優先序後到蓋先到。shas=None 回全部;給 shas 只回命中的(不含 unseen)。
    # 壞行(非 JSON / 缺 sha256 / 缺 event)靜默跳過(不進回傳);計數由 corrupt_line_count() 另取。

def corrupt_line_count() -> int
    # 掃 registry 檔,回無法解析 / 缺必要鍵的行數(供 UI 顯示健康度);檔不存在回 0。

def partition_new(shas: list[str]) -> tuple[list[str], dict[str, dict]]
    # (從未見過的 sha 順序保留, {已有狀態的 sha: state dict})。第二元素同 load_states 乾淨形。

def mark_sent(shas: list[str], batch_id: str, tool: str, paths: dict[str, str] | None = None) -> None
def mark_labeled(shas: list[str], batch_id: str) -> None
def mark_excluded(shas: list[str], tool: str, reason: str = "") -> None
    # 三個便捷包裝,都走 record_events。
```

### 2.2 GUI 接線(Tier B,`scripts/app.py`)

- **送標入口統一攔截**:`_send_to_labeling_ui`(所有工具共用的送標元件)在送出前:
  1. 對 batch 內每張圖取 sha256(cart snapshot / handoff 既有欄位;缺 sha 時現算,重用 export 的 `_file_sha256` 邏輯)。
  2. `partition_new(shas)` → UI caption:`本批 N 張:M 張新樣本;K 張已送標(批次 …)/已標,預設略過`。
  3. checkbox `key=f"{key}_include_handled"`「仍包含已送標/已標的樣本」預設 **False**;送出集合依此過濾。
  4. 送出成功後 `mark_sent(實際送出的 shas, batch_id, tool)`。
- **回讀入口**:📥 標註回饋「套用讀回」成功後,對該批次**有結果檔**的 sha `mark_labeled(shas, batch_id)`。
- **手動排除**:灰帶覆核的「🚫 排除」動作加掛 `mark_excluded`(tool="gray")。
- 顯示層:選樣畫廊的徽章可加灰字「已送標」(Could,不擋驗收)。

## 3. 資料流

選樣(任何工具)→ 送標按鈕 → sha256 → `partition_new` → UI 攤牌 → 過濾後送 Labeling
→ `mark_sent` → …標註… → 📥 回讀套用 → `mark_labeled` → 下輪選樣時同 sha 被攔。

## 4. 邊界條件與錯誤處理

| 情況 | 預期行為 |
|---|---|
| registry 檔不存在 | `load_states` 回 `{}`(全 unseen);`record_events` 自建目錄+檔 |
| 檔內有壞行 | 跳過壞行、其餘照常;`_corrupt_lines` 計數回報 |
| 同 sha 多事件 | 依優先序+時間折疊(見 2.1) |
| `reset` 事件 | 該 sha 回 unseen;之後事件重新累積 |
| shas=[] | `partition_new([])==([], {})`,不讀檔也可(短路) |
| LV_CACHE_DIR 有設 | registry 跟著進該目錄(與其他快取同根) |
| 送標批全部被攔 | 送出鈕 disabled + 理由 caption,**不送空批** |
| 圖檔已移動(path_hint 失效) | 不影響:身分是 sha;path_hint 僅顯示 |

## 5. Acceptance Criteria

**單元(`tests/test_pool_registry.py`;用 tmp_path + monkeypatch LV_CACHE_DIR,不碰真 registry)**

- **AC1(往返)**:`mark_sent(["a"*64], "b1", "viz")` 後 `load_states()["a"*64]["state"]=="sent"` 且 `batch_id=="b1"`。
- **AC2(優先序)**:同 sha 依序 sent→labeled 後 state=="labeled";再 excluded 後**仍**=="labeled"(labeled 最高)。
- **AC3(reset)**:sent→reset 後該 sha 不在 `load_states()`(unseen);reset→sent 後 =="sent"。
- **AC4(partition_new)**:registry 有 {A:sent};`partition_new([B,A,C]) == ([B,C], {A:{...state:"sent"...}})`(順序保留)。
- **AC5(壞行容忍)**:檔案手動塞入 `not json\n` 與一行缺 `event` 鍵的合法 JSON 後,`load_states()` 正常回其餘狀態(乾淨、鍵皆 sha、不含診斷欄位),且 `corrupt_line_count()==2`。
- **AC6(檔不存在)**:全新 tmp 目錄 `load_states()=={}`;`partition_new([X])==([X],{})`。
- **AC7(append-only)**:呼叫兩次 mark_sent 後檔案行數==2(不覆寫、不重排)。
- **AC8(無重依賴)**:`import pool_registry` 後 `"numpy" not in sys.modules`(在乾淨子行程驗,防依賴汙染)。

**GUI / 整合(真實 E2E,`tests/e2e/test_pool_registry_e2e.py`,`@pytest.mark.e2e`)**

- **AC-G1(第二次送標被攔)**:同一資料夾走送標流程兩次(harness 走既有 `_send_to_labeling_ui` 的可驅動路徑):
  第二次 UI 出現「已送標」字樣與數量,且預設送出集合為空(送出鈕 disabled 或送出數==0)。
- **AC-G2(覆寫開關)**:勾「仍包含已送標」後可送出,數量==原批大小。
- **AC-G3(回讀後升級)**:模擬 handoff 結果檔(重用 `test_lv_to_labeling` 的 seed/result fixture 手法)→
  套用讀回 → registry 內該 sha state=="labeled"(直接讀檔斷言)。

## 6. 給 /pm 的提示

- 核心是純邏輯,單元測試不需要影像檔:sha 用 `"a"*64` 之類假值即可。
- E2E 的送標路徑:`_send_to_labeling_ui` 是共用元件,挑一個已可 headless 驅動的工具(viz 選樣)走通即可,
  不必每工具各測一次(接線是同一個元件)。
- 鎖契約點:優先序表(AC2)與 `partition_new` 順序保留(AC4)是防 silent-wrong 的關鍵。
