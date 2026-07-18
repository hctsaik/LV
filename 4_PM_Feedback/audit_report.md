# 24 audit_report — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/24_audit_report.md](../3_Architect_Design/24_audit_report.md) · 里程碑 M20 · Tier A(純函式)
單元測試:`tests/test_audit_report.py`(gate:`python verify/gate.py audit_report`)

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | exact 手算:shas [a,a,a,b,c,c] → value==3、2 組、組員路徑正確 | `test_ac1_exact_dup_handcalc` | 🔴 |
| AC2 | near_dup 濾同 sha 對 → 只算 (3,4) | `test_ac2_near_dup_excludes_exact` | 🔴 |
| AC3 | leakage value==2、items 帶 (split_i,split_j) | `test_ac3_leakage_pairs` | 🔴 |
| AC4 | conflict:同 sha 異標 1 組;同 sha 同標 0;空標籤不觸發 | `test_ac4_label_conflict` | 🔴 |
| AC5 | coverage:val missing==["classB","classC"](字典序)、value==2;單 split → None+「不適用」 | `test_ac5_class_coverage` | 🔴 |
| AC6 | outliers top_n 降冪;None → None+「未計算」 | `test_ac6_outliers_topn_and_skipped` | 🔴 |
| AC7 | metadata 末節:value/text/groups/n_missing;meta=None → 無此節 | `test_ac7_metadata_section` | 🔴 |
| AC8 | sections 順序固定(overview…outliers,含 emb_near_dup 插位)、method 全非空、totals | `test_ac8_structure_contract` | 🔴 |
| AC9 | 白話句含數字;pairs==max_pairs → text 含「顯示上限」(no silent caps) | `test_ac9_text_numbers_and_cap_honesty` | 🔴 |
| AC10 | shas/paths 長度不一 → ValueError | `test_ac10_length_mismatch_raises` | 🔴 |

## done 判準
- `python verify/gate.py audit_report` 印 `GREEN`(Tier A,無 E2E 要件)。
