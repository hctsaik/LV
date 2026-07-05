# AC ↔ 測試對應表:13 sample_bank

設計:[3_Architect_Design/13_sample_bank.md](../3_Architect_Design/13_sample_bank.md)。單元:`tests/test_sample_bank.py`。

| AC | 說明 | 測試 |
|----|------|------|
| AC-B1 | save→load 往返(vectors/labels/provenance/model/res) | `test_ac_b1_save_load_roundtrip` |
| AC-B2 | append 接尾 N→N+k | `test_ac_b2_append` |
| AC-B3 | 模型/解析度相容檢查 | `test_ac_b3_model_compat` |
| AC-B4 | load 缺檔 → raise | `test_ac_b4_load_missing` |
| AC-B5 | append 維度不符 → raise | `test_ac_b5_append_dim_mismatch` |
| AC-B6 | build(注入 embed_fn)→ vectors/labels/provenance/model | `test_ac_b6_build_injected` |
| AC-B7 | build 無物件 → ValueError | `test_ac_b7_build_no_objects` |
| AC-B8(衍生) | 往返後可直接餵 multi_ref_similarity | `test_ac_b8_load_feeds_multi_ref` |
| AC-STHR-1 | ≥2 類 × 每類 ≥8 → ready(釘死回傳 dict) | `test_ac_sthr_1_two_classes_reach` |
| AC-STHR-2 | 一類差 1 顆 → 未達標、只達標類進 ready_classes | `test_ac_sthr_2_one_class_short` |
| AC-STHR-3 | 只 1 類(數量多也不夠)→ 未達標 | `test_ac_sthr_3_single_class` |
| AC-STHR-4 | 空 labels → 全空、未達標 | `test_ac_sthr_4_empty` |
| AC-STHR-5(衍生) | 門檻可調 + per_class/ready_classes 依類名排序 | `test_ac_sthr_5_threshold_param` |

## 增補(M14 E1):樣本集自描述 object_source
| AC | 說明 | 測試 |
|----|------|------|
| AC-OS-1 | build 回傳帶 object_source | `test_ac_os1_build_records_object_source` |
| AC-OS-2 | save→load 往返保留 object_source | `test_ac_os2_object_source_roundtrip` |
| AC-OS-3 | 舊集(meta 無此欄)load 回填 "yolo",不 raise | `test_ac_os3_load_legacy_defaults_yolo` |
| AC-OS-4 | 不變量:既有 13 測無回歸 | 既有全綠(gate 判) |
