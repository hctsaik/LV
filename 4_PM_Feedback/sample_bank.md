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
