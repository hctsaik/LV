# 23 meta_join — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/23_meta_join.md](../3_Architect_Design/23_meta_join.md) · 里程碑 M20 · Tier B(僅 CSV 讀檔)
單元測試:`tests/test_meta_join.py`(gate:`python verify/gate.py meta_join`)

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | 檔名匹配手算:同 basename 2 張全套用、n_matched==3、未匹配影像==1、bogus 列 idx==[2]、fields==["tool"] | `test_ac1_filename_match_handcalc` | 🔴 |
| AC2 | sha 優先:檔名亂寫 sha 對 → 配上;sha/檔名指不同影像 → sha 勝 | `test_ac2_sha_takes_priority` | 🔴 |
| AC3 | 未匹配雙向明講:5 bogus 列 + 2 張無列 | `test_ac3_unmatched_both_directions` | 🔴 |
| AC4 | 欄名不分大小寫(FileName/SHA256);無鍵欄 → ValueError 列出實際欄位 | `test_ac4_column_detection` | 🔴 |
| AC5 | group_counts:max_groups=3 → 前 2 組(4/3 張)+"(其他)"(4 張聯集);missing 含無值影像 | `test_ac5_group_counts_cap_and_missing` | 🔴 |
| AC6 | utf-8-sig(BOM)可讀;空檔 ValueError | `test_ac6_bom_and_empty` | 🔴 |
| AC7 | 多列同圖後列勝 + warnings 非空 | `test_ac7_later_row_wins_with_warning` | 🔴 |

註:AC5 fixture 刻意用**組大小嚴格遞減**(4>3>2=1=1)避免同大小 tie 在截斷邊界的
順序歧義——設計只保證「依組大小降冪」,tie 順序不是契約。

## done 判準
- `python verify/gate.py meta_join` 印 `GREEN`(無 GUI,無 E2E 要件;E2E 由 25 的 AC-G 連帶覆蓋)。
