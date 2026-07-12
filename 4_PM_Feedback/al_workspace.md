# 09 al_workspace — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/09_al_workspace.md](../3_Architect_Design/09_al_workspace.md) · 里程碑 **M10** · Tier B
單元/整合測試:`tests/test_al_workspace.py`(gate:`python verify/gate.py al_workspace`)
E2E:**不適用**(無 GUI;純持久化。al_service CLI 與 GUI 消費另驗)。

## 單元 / 整合 AC ↔ 測試(全 tmp 目錄,不觸真模型/真掃描)

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | profile round-trip;`default_profile` k==100 | `test_ac1_profile_roundtrip_and_default_k` | 🔴 |
| AC2 | 缺 profile / 缺 model_dir / schema_version 未知 → raise | `test_ac2_missing_or_bad_profile_raises` | 🔴 |
| AC-WS-DEC(M14) | `objective=="retrieve"` 時 model_dir 可空不 raise(特徵器身分改由 sample_bank);非 retrieve 仍必填 | `test_ac_ws_dec_retrieve_model_dir_optional` | 🔴 |
| AC3 | scan_incremental new/changed/deleted 正確且僅含該檔;第二次掃冪等全空 | `test_ac3_scan_incremental_classes_and_idempotent` | 🔴 |
| AC4 | mtime 不變、size 變 → changed(雙鍵) | `test_ac4_size_change_is_changed` | 🔴 |
| AC5 | scan 只讀來源,快照零變化(C6) | `test_ac5_scan_never_writes_source` | 🔴 |
| AC6 | queue 原子 round-trip 逐筆相等順序保留 | `test_ac6_queue_roundtrip` | 🔴 |
| AC7 | queue.jsonl 壞行跳過、不整份失敗 | `test_ac7_queue_bad_line_tolerated` | 🔴 |
| AC8 | labels append + 同 id 後覆前(審計保留全部) | `test_ac8_labels_append_and_last_wins` | 🔴 |
| AC9 | merge_labels_into_pending 移出已標/deleted;skip 留 pending | `test_ac9_merge_labels_into_pending` | 🔴 |
| AC10 | write_queue rename 前失敗 → 舊檔完好(原子) | `test_ac10_write_queue_atomic_on_failure` | 🔴 |
| AC11 | assert_safe_workspace:落 watch 內 raise、.lv_cache 下通過 | `test_ac11_assert_safe_workspace` | 🔴 |
| AC12 | 鎖互斥(fresh held→False)+ 時間過期搶佔 | `test_ac12_lock_mutex_and_stale_steal` | 🔴 |
| AC13 | release 只移本 PID 的鎖 | `test_ac13_release_only_own_pid` | 🔴 |

## 推導(PM 自加施壓)

| 測試 | 施壓的不變量 |
|------|-------------|
| `test_derived_single_writer_invariant` | 憲法 C5 單寫者:labels 只被 append(單調增、既有行不變)、queue 只被服務重寫、兩者互不污染(檔案內容斷言) |

## 契約微調(PM 依設計意圖,給可測介面)

- `acquire_lock(ws, *, pid, started, stale_after=3600.0)` / `release_lock(ws, *, pid)`:鎖帶**顯式 pid**(al_service 傳自身;測試可控),使「只移本 PID」「過期搶佔」可驗(設計 §2.2 的 release_lock(ws) 補 pid 參數)。
- `merge_labels_into_pending(queue_items, labels, *, deleted_ids=())`:deleted 集以參數傳入(純函式,不碰檔案)。

## done 判準(Tier B,無 GUI)

- `python verify/gate.py al_workspace` 印 `GREEN`(收集>0、退出碼 0、無 fail/error、3_/4_ 契約未竄改)。
- 真實行為由對真實檔案系統(tmp 目錄、原子寫、mtime/size 增量、鎖檔)的整合斷言涵蓋;無瀏覽器 E2E。
