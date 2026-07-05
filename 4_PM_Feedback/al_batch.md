# 08 al_batch — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/08_al_batch.md](../3_Architect_Design/08_al_batch.md) · 里程碑 **M9** · Tier B
單元/整合測試:`tests/test_al_batch.py`(gate:`python verify/gate.py al_batch`,自動對應 `tests/test_{module}.py`)
E2E:**不適用**——al_batch 無 GUI;Tier B 的 done = 單元 + **真實檔案系統整合測試**綠(不需 Playwright)。
M9-GUI 接線(三模式顯示 / 暫定 Top-K / 續跑)另以 `@pytest.mark.e2e` 驗,不屬本模組。

## 驗收特性(全程注入 extractor/embed_fn + 合成向量凍結模型,不觸真 DINOv2)

- 影像色彩雙訊號:**紅→異常分數**(graded extractor,patch 與 bank-A 距離隨紅單調)、**藍→head 預測類別**(class embed_fn,obj_emb 落 X/Y 區);兩訊號正交 → 分數與類別可各自控制。
- 凍結模型:`save_bank`(50 個正常 A 向量)+(可選)`train_head`/`save_head` 2 類 head。

## 單元 / 整合 AC ↔ 測試

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | 分批(bs=7)最終 topk **逐一等於**一次跑完(bs=40) | `test_ac1_batched_equals_single` | 🔴 |
| AC2 | 中斷後續跑,已處理批 extractor 呼叫增量 0;結果==未中斷 | `test_ac2_resume_no_recompute` | 🔴 |
| AC3 | 跑一批後 `provisional_topk` = 已處理併集依原始分數前 k;`provisional=True` | `test_ac3_provisional_topk_is_true_topk_of_processed` | 🔴 |
| AC4 | 全 tie 下 topk 跨 batch_size **位元相同**(item_id canonical 排序,非陣列位置) | `test_ac4_topk_deterministic_across_batch_size_on_ties` | 🔴 |
| AC5 | 改 batch_size 續跑 → `error` raise;`restart` 清 checkpoint 重跑 | `test_ac5_resume_refuses_on_identity_drift` | 🔴 |
| AC6 | `objective="uncertain"` 無 head → 開跑前 raise,extractor 呼叫 0 次 | `test_ac6_objective_requires_head_before_embedding` | 🔴 |
| AC7 | (a) 同物件跨 model_dir → 同 item_id;(b) 同 stem 不同內容 → 不同 item_id | `test_ac7_item_id_content_addressed` | 🔴 |
| AC8 | 跑完來源資料夾檔案清單 + mtime 快照零變化(C6) | `test_ac8_never_writes_dataset_folder` | 🔴 |
| AC9 | `checkpoint_dir` 落來源資料夾內 → `assert_safe_bank_dir` raise | `test_ac9_rejects_unsafe_checkpoint_dir` | 🔴 |
| AC10 | yolo 某批 0 物件 → 空批不崩、commit、`objects_scored` 不計空批 | `test_ac10_empty_and_zero_object_batch` | 🔴 |
| AC11 | 模擬 crash(orphan shard + state 未登錄)續跑冪等,無雙計 | `test_ac11_crash_between_shard_and_state_is_idempotent` | 🔴 |
| AC12 | 只改 manifest.json → model_version 不變;改 bank.npz → 變 | `test_ac12_model_version_excludes_manifest` | 🔴 |
| AC13 | novelty 暫定 Top-K 成員 == 已處理併集分數前 k 成員(_minmax 單調) | `test_ac13_novelty_provisional_is_final_membership` | 🔴 |
| AC14 | 有 head → `diversity_applied=True` 且同預測類 ≤ cap(2);無 head → False | `test_ac14_diversity_flag_and_cluster_cap` | 🔴 |
| AC15 | 回傳含 §2.4 所有鍵;`topk_records` 欄位齊全且 `reason` 非空字串 | `test_ac15_return_shape_and_reason` | 🔴 |

> 本表 AC 集合 `{1..15}` 與設計 §5 一致,無跳號。

## 推導 / 反向稽查測試(設計未明列,PM 自加施壓)

| 測試 | 施壓的不變量 |
|------|-------------|
| `test_derived_multi_resume_converges` | property/metamorphic:任意切成多次「一批續跑」最終 == 一次跑完(比 AC1/AC2 更強的續跑收斂性) |
| `test_derived_no_forbidden_global_ops_in_batch` | 反向稽查:讀 `scripts/al_batch.py` 原始碼,斷言**不含** `run_pipeline(` / `candidates_in_reference=True` / `cluster_objects(`(設計 §2.6 禁止事項、C8) |

## 誠實定位(別把單元綠當「已被獨立驗證」)

- PM 與 architect 同一模型、共享盲區:單元測試防**實作 bug**,不防「設計與驗收同時漏掉」的概念盲點——後者靠前三階段人審 + M9-GUI 真實 E2E 攔。
- 反向稽查(`test_derived_no_forbidden_global_ops_in_batch`)是同源仍能逼出「PG 為省事包 run_pipeline」這種 C8 違規的守門。

## done 判準(Tier B,無 GUI)

- **單元/整合綠**:`python verify/gate.py al_batch` 印 `GREEN`(收集>0、退出碼 0、無 fail/error、`3_/4_` 契約未竄改)。
- al_batch **無 GUI → 無瀏覽器 E2E**;其「真實行為」由 AC1/AC2/AC3/AC8/AC11 等對真實檔案系統(tmp 目錄、原子寫、續跑)的整合斷言涵蓋。
- M9 里程碑完成另需 **M9-GUI 接線**(消費 al_batch 的三模式顯示/暫定 Top-K/續跑)通過真實 Playwright E2E——那是 GUI 接線的 done 要件,不在本模組。

## 增補(M12b / A3):objective="similar"
| AC | 說明 | 測試 |
|----|------|------|
| AC-SIM1 | ref=E2 → topk 全為 X 群、priority 降冪、reason 含「相似度」 | `test_ac_sim1_similar_ranks_by_reference` |
| AC-SIM2 | similar 缺 ref_vector → ValueError(含 ref_vector) | `test_ac_sim2_similar_requires_ref_vector` |
| AC-SIM3 | 換參考=另一 run(error 拒;restart 重算翻群) | `test_ac_sim3_change_ref_is_new_run` |
| AC-SIM4 | similar 分批==一次跑(C8) | `test_ac_sim4_similar_batched_equals_single` |
| AC-SIM5 | 既有 17 測無回歸(novelty/uncertain/confusion 不變) | 既有全測保持綠(gate 判) |

## 增補(M13 Task3):objective="retrieve"(以樣搜樣多參考)
| AC | 說明 | 測試 |
|----|------|------|
| AC-RET1 | 建議類別由樣本決定(忽略粗框 cls)+ reason 含相似度 + suggested_class 欄 | `test_ac_ret1_suggested_class_ignores_proposal_cls` |
| AC-RET2 | 缺 ref_vectors → ValueError | `test_ac_ret2_requires_ref_vectors` |
| AC-RET3 | 換樣本集=另一 run | `test_ac_ret3_change_samples_new_run` |
| AC-RET4 | 分批==一次跑(C8) | `test_ac_ret4_batched_equals_single` |
| AC-RET5 | conf 預篩丟低信心粗框 | `test_ac_ret5_conf_prefilter` |
| AC-RET6 | 既有 21 測無回歸 | 既有全測綠(gate 判) |

## 增補(M14 E2):retrieve 免整包模型(feature_extractor lite 載入)
| AC | 說明 | 測試 |
|----|------|------|
| AC-RL-1 | model_dir=None + feature_extractor(免 bank/projection)也能跑、records 帶建議類別 | `test_ac_rl1_retrieve_without_bank` |
| AC-RL-2 | retrieve 不算 anomaly 分數 → 每筆 score==0.0 | `test_ac_rl2_retrieve_no_anomaly_score` |
| AC-RL-3 | `_extractor_version(model,res,object_source)` 穩定且隨三者變 | `test_ac_rl3_extractor_version_identity` |
| AC-RL-4 | 換特徵器 → 續跑身分變 → 不吃 stale shard(error 拒/restart 重算) | `test_ac_rl4_change_extractor_not_stale` |
| AC-RL-5 | retrieve 缺 model_dir 且缺 feature_extractor → ValueError | `test_ac_rl5_requires_extractor_or_model_dir` |
| AC-RL-6 | 不變量:既有 26 測(其他 objective)無回歸 | 既有全綠(gate 判) |
