# 10 — al_service 技術設計(離線監看選樣服務:CLI run-once 編排)

對應 PRD:[2_PO_PRD/al_scale_offline_prd.md](../2_PO_PRD/al_scale_offline_prd.md) · 里程碑 **M10** · Tier B
依賴:**08 `al_batch`**(選樣引擎)+ **09 `al_workspace`**(持久化/掃描/鎖)。設計素材 §6.4。

> **run-once + 外部排程,不做常駐 daemon**(憲法 C7)。跑一輪就退出;排程交工作排程器/手動/未來平台。
> 只**編排**——所有持久化委派 09、所有選樣委派 08、不自寫檔、不自算分數。

## 1. 目的(Purpose)

在**不開互動畫面**下,對一個(持續更新的)監看資料夾跑一輪:增量掃描 → 用凍結模型分批選樣 →
合併人工標註(移出已標)→ 物化 Top-K 標註佇列 → 寫摘要。**GUI 與服務共用 08 引擎、以 09 工作區為單寫者契約。**

## 2. I/O 契約

### 2.1 公開函式(放 `scripts/al_service.py`)

```python
def run_once(workspace_dir, *, embed_fn=None, extractor=None,
             max_batches=None, now=None) -> dict:
    """跑一輪離線選樣。now: float(可注入時間,測試用;預設 time.time())。
    embed_fn/extractor: 透傳給 al_batch(測試注入;正式為 None → 惰性建真模型)。
    回 {"status": "ok"|"skipped"|"error", "reason": str,
        "new": int, "changed": int, "deleted": int,
        "objects_scored": int, "queue_len": int,
        "model_version": str, "run_id": str, "summary_path": str|None}。"""

def status(workspace_dir) -> dict:
    """讀最近一輪摘要 + 佇列長度 + 鎖狀態(給人/排程健檢;不跑任何運算)。"""

def init_workspace(workspace_dir, *, name, watch_folders, model_dir, **over) -> dict:
    """初始化工作區:assert_safe → 寫 default_profile。回 profile。"""

def main(argv=None) -> int:
    """CLI 入口:run / status / init。回退出碼(0=ok/skipped,非0=error)。"""
```

### 2.2 CLI

```
python scripts/al_service.py run    --profile <workspace_dir>   # 跑一輪;exit 0(ok/skipped)/ 非0(error)
python scripts/al_service.py status --profile <workspace_dir>   # 印最近摘要 + 佇列 + 鎖
python scripts/al_service.py init   --profile <workspace_dir> --model-dir <..> --watch <..> [--k 100 ...]
```

## 3. `run_once` 編排流程(嚴格順序)

1. **安全**:`al_workspace.assert_safe_workspace(workspace_dir, watch_folders)`(watch 讀自 profile;profile 缺則先讀失敗)。
   實務:先 `load_profile`(缺/壞 → 回 `{"status":"error","reason":...}`,exit 非0),再 assert_safe(checkpoint/工作區安全)。
2. **鎖**(C7 單實例):`acquire_lock(ws, pid=os.getpid(), started=now)`。False → 回 `{"status":"skipped","reason":"另一實例執行中"}`(exit 0)。
   取得後 **try/finally `release_lock(ws, pid)`**(例外也放鎖)。
3. **驗模型**:`al_batch.load_frozen_model(profile["model_dir"])`——缺檔/不相容 → error(放鎖後退出)。取 `object_source = disk meta`。
4. **增量掃描**:`state=load_state(ws)`;`scan = scan_incremental(watch_folders, state)` 得 new/changed/deleted。
5. **選樣**(委派 08):蒐集**當前全部**影像(state 更新後的 cur ∪ new/changed;= scan 後的所有現存來源影像)`all_paths`;
   `labels = read_labels(ws)`;`al_batch.run_batched(all_paths, model_dir, checkpoint_dir=<ws>/al_batch_ck,
   objective=profile.objective, k=profile.k + len(labels), batch_size=profile.batch_size,
   object_source=<disk meta>, class_names=<classes_for>, dataset_dirs=watch_folders,
   embed_fn, extractor, resume=True, on_identity_mismatch="restart", max_batches=max_batches)`。
   - **增量性**:`.lv_cache`(al_batch 的 embedding 快取,在 checkpoint 內)→ 只有**新圖真的 embed**;分數重算 O(N)、便宜(無模型)。
   - **模型換版**:model_dir 變 → al_batch `model_version` 變 → `resume_identity` 變 → `restart` 重評分(item_id 內容不變 → 佇列原地更新)= 拍板「重評分」。
6. **合併標註**(委派 09 純函式):對 `run_batched` 的 `topk_records`:
   `pending = merge_labels_into_pending(records, labels, deleted_ids=scan["deleted"])`;取 `pending[:profile.k]`。
7. **蓋章物化**:每筆 queue item 加 `model_version`(al_batch 回傳的)、`run_id`(=run_id)、`status="pending"`、`created_at`;
   `write_queue(ws, queued)`(09 原子重寫)。
8. **落狀態 + 摘要**:`save_state(ws, scan["state"])`;`write_run_summary(ws, {counts...,model_version,run_id,duration}, ts=run_id)`。
9. **放鎖**、回 `{"status":"ok",...}`。

> `run_id` = 由 `now` 產生的時間戳字串(如 `20260705T031201`)——**由 caller/now 決定,不在模組內呼叫 `time`/`datetime` 於測試路徑**(可注入)。

## 4. 邊界條件與錯誤處理

| 情況 | 預期行為 |
|---|---|
| profile 缺/壞 | `run_once` 回 `{"status":"error",...}`;CLI exit 非0;**不建鎖、不寫任何檔** |
| watch_folder 不存在 | scan 略過該資料夾(0 檔);若全部不存在 → 0 new → 佇列空但 `status="ok"`(非 error;空掃描合法) |
| model_dir 缺/不相容 | load_frozen_model raise → error(**已取鎖則先放鎖**)、exit 非0 |
| 另一實例執行中(鎖新鮮) | `status="skipped"`、exit 0、不動任何檔 |
| 執行中例外 | try/finally 放鎖;回 error;已寫的原子檔各自完好(無半寫)|
| deleted 檔 | 從 pending 移出(標 stale 隱藏);labels 審計保留 |
| 空來源(0 影像) | queue 空、`status="ok"`、寫摘要(objects_scored=0) |
| checkpoint_dir 落來源內 | al_batch 的 assert_safe_bank_dir raise → error |

## 5. Acceptance Criteria(釘死;給 /pm。合成資料 + 注入 embed_fn/extractor,**不觸真 DINOv2**;凍結模型用 `save_bank` 造)

> 重用 al_batch 測試的 graded extractor(紅→分數)+ 合成 bank;`now` 注入固定時間戳。全 tmp 目錄。

- **AC1(首輪)**:空 state + 6 張合成圖(2 張高紅=異常)→ `init_workspace` + `run_once` →
  `status=="ok"`;`read_queue` 有 `min(k, objects)` 筆;**高紅 2 張排最前**;每筆含 `id/score/reason/model_version/run_id/status=="pending"`;`runs/<run_id>.json` 存在。
- **AC2(增量只 embed 新圖)**:AC1 後再加 3 張新圖 → 第二輪(注入計數 extractor)→ **extractor 呼叫增量 == 3**(舊 6 張靠 .lv_cache 不重 embed);佇列重排含新圖。
- **AC3(閉環:標註移出)**:對佇列前 2 筆 `append_label`(good/defect)→ 下一輪 → 這 2 筆 `id` **不在** queue;摘要 counts 正確。
- **AC4(模型換版 → 重評分、item_id 原地)**:換一個內容不同的 model_dir(仍相容 model/res)→ run_once →
  佇列項 `model_version` **改變**;同一物件的 `id` **不變**(內容定址),非新增重複項。
- **AC5(壞輸入)**:(a) 無 profile → `status=="error"`、無鎖檔、無 queue/state 檔;(b) model_dir 缺 → error、**鎖已釋放**(lock 不殘留)。
- **AC6(不寫來源,C6)**:run_once 跑完,來源資料夾檔案清單 + mtime 快照零變化。
- **AC7(單實例鎖)**:先手動寫一個新鮮 lock(他人 pid)→ `run_once` 回 `status=="skipped"`、**不動 queue/state**;移除 lock 後再跑 → `ok`。
- **AC8(CLI 退出碼)**:`main(["run","--profile",ws])` 回 0(ok);`main(["run","--profile",<無 profile>])` 回非0;`main(["init",...])` 建出 profile.yaml。

## 6. 給 /pm 的提示

- 測試 `tests/test_al_service.py`;可直接呼叫 `run_once`/`main`(不必真 subprocess;CLI smoke 用 `main(argv)` 即可)。
- 凍結模型 fixture 同 al_batch(`save_bank` 合成 bank);合成圖用紅通道編碼分數;注入 graded `extractor` + 計數包裝驗 AC2。
- `now` 注入固定值產生確定 `run_id`;AC4 換 model_dir 用不同 bank 向量列數使 `model_version` 變。
- **契約鎖點**:增量只 embed 新圖(AC2)、標註閉環移出(AC3)、換版重評分 item_id 原地(AC4)、鎖釋放/error 不殘留(AC5)、不寫來源(AC6)、單實例(AC7)是防 silent-wrong 核心。
- 開跑前放行 PG 記得 `python verify/gate.py --snapshot`。

> GUI 端(④ 監看設定/匯出/狀態/立即掃描 + Step③ 佇列消費)屬 **M10-GUI 接線**,以真實 E2E 驗收,不在本模組。
