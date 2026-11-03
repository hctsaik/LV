# AC ↔ 測試對應表:class_similarity_records

設計:[3_Architect_Design/32_class_similarity_records.md](../3_Architect_Design/32_class_similarity_records.md)。單元:`tests/test_class_similarity_records.py`。

| AC | 說明 | 測試 |
|----|------|------|
| AC06 | 顯示切換不改分; K/label/raw/recipe/split 失效 | `test_ac06_display_does_not_rescore` `test_ac06_inputs_invalidate_key` |
| AC08 | 過濾後 original_index; 過期事件拒絕 | `test_ac08_index_after_filter` `test_ac08_stale_event_rejected` |
| AC29 | 路徑 vs crop policy; 暫存 ID; crop 衝突 | `test_ac29_path_vs_crop_policy_id` `test_ac29_invalid_temp_ids` `test_ac29_crop_collision_integrity` |
| AC30 | 讀回、flat、unknown/mix 資料夾、A→B、restore | `test_ac30_readback_effective_label` `test_ac30_flat_fallback_unresolved` `test_ac30_literal_unknown_mix_single` `test_ac30_mode_restore_rejects_old_event` |
| AC03 | 重複身份停止 | `test_ac03_duplicate_identity_error` |
| AC34 | 生成後改圖 stale; 未驗 generation 禁 PR 標記 | `test_ac34_generation_hash_mismatch` `test_ac34_unverified_generation` |
