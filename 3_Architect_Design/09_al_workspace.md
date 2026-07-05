# 09 — al_workspace 技術設計(離線服務工作區持久化 + 增量掃描 + 鎖)

對應 PRD:[2_PO_PRD/al_scale_offline_prd.md](../2_PO_PRD/al_scale_offline_prd.md) · 里程碑 **M10** · Tier B
設計素材(非契約):[AL_SCALE_AND_OFFLINE_SERVICE_PLAN.md](../AL_SCALE_AND_OFFLINE_SERVICE_PLAN.md) §6.3

> 純持久化 + 增量掃描 + 鎖;**無選樣邏輯**(那是 al_batch)、**無編排**(那是 al_service)。
> 承擔憲法 **C5 單寫者**(服務寫 queue/state,GUI 只 append labels)、**C6 不寫來源**、**C8**(不做全域運算)。

## 1. 目的(Purpose)

管理離線監看服務的**全部持久化檔案**(設定 / 掃描狀態 / 佇列 / 標註 / 每輪摘要 / 執行鎖)與**增量掃描狀態**,
讓 al_service 每輪只處理新/變更檔、GUI 與服務**單寫者**互不踩踏、設定檔可攜(跨機換路徑即可)。

## 2. I/O 契約

### 2.1 工作區目錄佈局

```
<workspace_dir>/                 # .lv_cache/al_workspace/<name>/ 或使用者另指定(須過 assert_safe)
  profile.yaml                   # 設定(人可讀可手改;GUI 匯出的「可攜設定檔」= 這份)
  state.json                     # 增量掃描狀態:{path: {mtime, size}} + last_run + schema_version(服務寫)
  queue.jsonl                    # 佇列(服務**原子重寫**;GUI 只讀)
  labels.jsonl                   # 人工決定(GUI **append**;服務只讀 + 下輪合併)
  runs/<ts>.json                 # 每輪摘要(服務寫;唯讀審計)
  lock                           # 執行鎖(PID + 開始時間;過期可搶)
```

**單寫者(C5)**:`queue.jsonl` / `state.json` / `runs/` 只由**服務**寫;`labels.jsonl` 只由 **GUI append**。兩邊永不寫同一檔。

### 2.2 公開函式(放 `scripts/al_workspace.py`)

```python
# ── 設定(profile;可攜、人可編)──
def load_profile(workspace_dir) -> dict          # 讀 profile.yaml → 驗 schema + 填預設;缺檔/壞檔 raise
def save_profile(workspace_dir, profile: dict) -> None   # 原子寫 profile.yaml(schema_version 補齊)
def default_profile(*, name, watch_folders, model_dir, **over) -> dict   # 產一份含預設值的 profile(K=100 等)

# ── 增量掃描狀態 ──
def scan_incremental(folders, state: dict) -> dict
    # 回 {"new":[Path...], "changed":[Path...], "deleted":[str...], "state":<更新後 state>}
    # new/changed = 依 (mtime,size) 雙鍵比對;deleted = state 有但檔已不在。只讀來源(C6)。
def load_state(workspace_dir) -> dict            # 讀 state.json;缺/壞 → 回空初始 state(視為全新)
def save_state(workspace_dir, state: dict) -> None   # 原子寫

# ── 佇列(服務寫、GUI 讀)──
def read_queue(workspace_dir) -> list[dict]      # 讀 queue.jsonl(逐行 json;壞行跳過並計數)
def write_queue(workspace_dir, items: list[dict]) -> None   # 原子重寫整份(temp+rename)

# ── 標註(GUI append、服務讀+合併)──
def append_label(workspace_dir, label: dict) -> None   # 對 labels.jsonl **append 一行**(不重寫)
def read_labels(workspace_dir) -> list[dict]     # 讀全部;同 item_id 取**最後一筆**(後覆前)
def merge_labels_into_pending(queue_items, labels) -> list[dict]
    # 純函式:把已標(decision∈good/defect)或 deleted 的 item 從 pending 移出;回留下的 pending
    #（給 al_service 合併用;不碰檔案)

# ── 每輪摘要 ──
def write_run_summary(workspace_dir, summary: dict, *, ts: str) -> Path   # 原子寫 runs/<ts>.json(ts 由 caller 給)

# ── 執行鎖(單一服務實例)──
def acquire_lock(workspace_dir, *, pid: int, started: float, stale_after: float = 3600.0) -> bool
    # 寫 lock(PID+started);已有且未過期且 PID 存活 → False;過期/PID 死 → 搶佔並回 True
def release_lock(workspace_dir) -> None          # 移除 lock(僅移除本 PID 的;非本 PID 不動)

# ── 安全目錄(C6)──
def assert_safe_workspace(workspace_dir, watch_folders=()) -> None
    # 重用 anomaly_bank_store.assert_safe_bank_dir(workspace_dir, selected_folders=watch_folders)
```

### 2.3 profile.yaml schema(可攜設定檔;GUI 唯一編輯面 → 匯出給服務)

```yaml
schema_version: 1
name: line3-pills                 # profile 名(工作區目錄名)
watch_folders: ["D:/data/incoming"]   # 只讀來源(可多個;跨機換路徑即可,不改碼)
model_dir: "D:/lv_models/pill_v3"     # GUI 一鍵存出的凍結模型目錄
objective: novelty                    # novelty | uncertain | confusion(對齊 al_batch)
k: 100                                # 使用者拍板預設
score_mode: patch                     # 對齊模型 meta(僅記錄;實際以模型 meta 為準)
target_res: 224
batch_size: 1000
filters: {min_score: null, classes: []}   # Could(前置篩選;v1 可留空)
```

- **可攜性(C:跨機)**:profile 是**唯一可攜設定**;`state.json`/`queue.jsonl` 等是本機衍生(不進「匯出」)。
  跨機部署 = 換 `watch_folders`/`model_dir`/`workspace_dir` 路徑,不改碼(§4 開放問題已由 PO 定:服務為設定檔純消費者)。
- **schema_version**:未來欄位演進的向前相容鎖;load 時版本不符 → raise 明確訊息(不靜默誤讀)。

### 2.4 queue.jsonl / labels.jsonl 單筆 schema(對齊 al_batch topk_records + labeling_handoff 命名)

```json
// queue.jsonl(服務寫;每行一筆;欄位對齊 al_batch topk_records + 服務加的 run/status)
{"id":"<al_batch item_id>","image_path":"...","bbox":[cx,cy,w,h],"obj_index":0,
 "label":"...","score":0.87,"priority":0.91,"reason":"很不像正常樣本…",
 "model_version":"<hash>","run_id":"20260705T031201","status":"pending","created_at":"..."}
// labels.jsonl(GUI append;每行一筆)
{"id":"<item_id>","decision":"good|defect|skip","label":"刮痕",
 "decided_at":"...","model_version_at_decision":"<hash>"}
```

- **`id` = al_batch 的 `item_id`**(影像內容 sha256 + bbox + label + obj_index;跨模型版本不變)→ 重評分**原地對應**(憲法 C4/C1)。
- 欄位命名參考 `scripts/labeling_handoff.py`(未來平台 Labeling 對接留鉤子;不強耦合)。

## 3. 資料流(Data Flow)

1. **服務一輪**(al_service 呼叫本模組):`acquire_lock` → `load_profile` → `load_state` →
   `scan_incremental(watch_folders, state)` 得 new/changed/deleted → (al_batch 評分 new/changed)→
   `read_labels` + `merge_labels_into_pending`(移出已標/deleted)→ `write_queue`(原子重寫)→
   `save_state`(記新 mtime/size)→ `write_run_summary` → `release_lock`。
2. **GUI 消費**:`read_queue`(只讀)顯示佇列;使用者決定 → `append_label`(只 append,不碰 queue/state)。
3. **單寫者不變式**:服務不碰 labels.jsonl(只讀+合併);GUI 不碰 queue/state/runs。→ 免鎖檔地獄(lock 只擋「兩個服務實例並跑」)。

## 4. 邊界條件與錯誤處理

| 情況 | 預期行為 |
|---|---|
| `workspace_dir` 落在 watch_folder 內 / 含 images\|labels | `assert_safe_workspace` raise(C6)|
| `profile.yaml` 缺 / schema_version 不符 / 缺必填(name/watch_folders/model_dir)| `load_profile` raise 明確訊息(不靜默填危險預設)|
| `state.json` 缺 / 壞(torn)| `load_state` 回空初始 state → 視為全新(全部檔算 new)|
| `queue.jsonl` 某行壞 | `read_queue` 跳過壞行、計數;不整份失敗 |
| `write_queue` / `save_state` 途中失敗 | temp+rename 原子 → 舊檔完好(重用 `_atomic_text`)|
| 網路磁碟 mtime 不可靠 | state 用 **mtime + size 雙鍵**;文件註明網路磁碟限制;Could:可選 hash 模式(v1 不做)|
| 檔案寫到一半被掃到(不可讀)| `scan_incremental` 只比對 mtime/size(不解碼)→ 交 al_batch 的 `partition_readable` 剔除;下輪重試 |
| deleted 檔的佇列項 | `merge_labels_into_pending` 移出 pending(標 stale,不刪 labels 審計)——對齊 PO 裁決 |
| `acquire_lock` 已被持有且存活未過期 | 回 False(第二實例不啟動)|
| lock 的 PID 已死 / 超過 stale_after | 搶佔並回 True |
| `append_label` 對同 id 多次 | 都 append;`read_labels` 取**最後一筆**(後覆前;審計保留全部)|

## 5. Acceptance Criteria(釘死;給 /pm。全用 **tmp 目錄**,不觸真模型/真掃描)

**設定(profile)**
- **AC1(round-trip)**:`save_profile(ws, p)` 後 `load_profile(ws) == p`(補齊 schema_version/預設後相等);`default_profile(...)["k"] == 100`。
- **AC2(缺/壞 profile)**:無 profile.yaml → `load_profile` raise;缺 `model_dir` → raise;`schema_version` 設成未知值 → raise(訊息含 "schema")。

**增量掃描**
- **AC3(new/changed/deleted 分類)**:tmp 造 3 檔建 state → 加 1 檔(new)、touch 改 1 檔的 mtime(changed)、刪 1 檔(deleted)→
  `scan_incremental` 的 new/changed/deleted 各含**正確且僅含**該檔;第二次對更新後 state 掃 → new/changed/deleted **全空**(冪等)。
- **AC4(size 也算)**:mtime 不變但檔案 size 變(改內容保持 mtime,或 size 不同)→ 歸入 changed(雙鍵)。
- **AC5(只讀來源,C6)**:`scan_incremental` 跑完,來源資料夾檔案清單 + mtime 快照**零變化**。

**佇列 / 標註 / 合併**
- **AC6(queue 原子 round-trip)**:`write_queue(ws, items)` 後 `read_queue(ws) == items`(逐筆相等,順序保留)。
- **AC7(queue 壞行容錯)**:queue.jsonl 插一行非法 json → `read_queue` 回其餘完好筆數,不 raise。
- **AC8(labels append + 後覆前)**:`append_label` ×3(其中兩筆同 id 不同 decision)→ labels.jsonl 有 3 行;
  `read_labels` 對該 id 取**最後一筆**。
- **AC9(merge 移出已標/deleted)**:queue 5 筆,labels 標其中 2 筆(good/defect)+ deleted 集含 1 筆 →
  `merge_labels_into_pending` 回**剩 2 筆 pending**(移出 2 已標 + 1 deleted);skip 決定**留在 pending**(可再看)。

**原子性 / 鎖 / 安全**
- **AC10(原子:途中失敗舊檔完好)**:monkeypatch 使 `write_queue` 在 rename 前丟例外 → queue.jsonl **維持舊內容**(未半寫)。
- **AC11(安全目錄)**:`assert_safe_workspace(watch/子目錄, watch_folders=[watch])` raise;`.lv_cache` 下的 workspace 通過。
- **AC12(鎖互斥 + 過期搶佔)**:`acquire_lock(pid=A, started=now)` → True;`acquire_lock(pid=B)`(A 未過期且模擬存活)→ False;
  把 lock 的 started 設成 `now - 2*stale_after`(或 PID 設不存在)→ `acquire_lock(pid=B)` → True(搶佔)。
- **AC13(release 只移本 PID)**:A 持鎖,`release_lock`(以 B 身分,設計為只移本 PID 的鎖)→ 若非本 PID 不移除(避免誤放他人鎖)。

**推導(PM 自加)**
- **AC-D1(單寫者不變式)**:一連串「服務 write_queue/save_state」與「GUI append_label」交錯操作後,
  labels.jsonl 只被 append(行數單調增、既有行不變);queue.jsonl 只被服務重寫;兩者互不污染(以檔案內容斷言)。

## 6. 給 /pm 的提示

- 單元/整合測試 `tests/test_al_workspace.py`;無 GUI → 不需 E2E(al_service 的 CLI 與 GUI 消費另驗)。
- 全用 `tmp_path` 造 workspace + 假來源資料夾;`scan_incremental` 用 `os.utime` 改 mtime、寫不同內容改 size。
- **重用鎖點**:`_atomic_text`(anomaly_bank_store)、`assert_safe_bank_dir`;profile 用 YAML(若無 `pyyaml`,PG 走反向閘門回報,退 JSON,不阻擋)。
- **契約鎖點**:增量分類正確+冪等(AC3)、單寫者不互污(AC-D1)、item_id 原地對應(queue/labels 的 `id`=al_batch item_id)、原子性(AC10)是防 silent-wrong 核心。
- 開跑前放行 PG 記得 `python verify/gate.py --snapshot` 重建 baseline。

> 完成後停下等審查,再放行 `/pm`。10 `al_service` 的設計待 09 定案(al_service 依賴 09+08)。
