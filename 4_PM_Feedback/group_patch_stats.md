# 20 group_patch_stats — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/20_group_patch_stats.md](../3_Architect_Design/20_group_patch_stats.md) · 里程碑 M19 · Tier A(純陣列)
單元測試:`tests/test_group_patch_stats.py`(gate:`python verify/gate.py group_patch_stats`)

## 單元 AC ↔ 測試

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | 整群植入位移 → 該 patch `score`/`z` 皆 Top-1、verdict True、p<0.05 | `test_ac1_planted_top1_and_verdict` | 🔴 |
| AC2 | null 誠實:同分布 10 組(seed 100–109)→ verdict True ≤ 2 組(α=0.05 校準;**用集合而非單組,把 5% 固有偽陽率變成可斷言的校準性質**) | `test_ac2_null_honest` | 🔴 |
| AC3 | 40 張 Bad 僅 1 張位移 → 偶然雜訊:verdict False 且 `stability[7] < 0.2` | `test_ac3_single_outlier_not_stable` | 🔴 |
| AC4 | 80% Bad 位移 → `stability[7] ≈ 0.8 ± 0.15`、仍 Top-1、verdict True | `test_ac4_stability_quantifies_fraction` | 🔴 |
| AC5 | 群內散度入分母:同位移、noise 0.05 vs 0.45 → 低噪組分數 > 1.5× 高噪組 | `test_ac5_within_spread_in_denominator` | 🔴 |
| AC6 | Ng<2 / P 不一致 / D 不一致 / NaN / n_perm<20 → ValueError | `test_ac6_validation_errors` | 🔴 |
| AC7 | 同參數同 seed → `score`/`z`/`stability`/`p_global` 完全相等 | `test_ac7_deterministic` | 🔴 |
| AC8 | 輸出鍵/形狀契約(設計 §2 表)+ stability ∈ [0,1] | `test_ac8_output_shape_contract` | 🔴 |

## 推導測試(設計未明列,PM 自加施壓)

| 測試 | 施壓的不變量 |
|------|-------------|
| `test_derived_swap_symmetry` | `score` 對群交換對稱(分子/分母皆對稱)→ 逼出「單向距離當分數」「good/bad 角色寫死」類 bug;Ng==Nb 時 verdict 一致 |

## 誠實注記(de-risk)

- AC2/AC3 的 verdict 斷言依賴 permutation p 值在固定 seed 下的**確定性實現**。理論上
  正確實作也有小機率在特定 seed 落在錯側;若 PG 以正確實作觸紅且能論證為 seed 不幸,
  走反向閘門 `/pg → /pm` 換 fixture seed 並記決策日誌——**不得由 PG 自行改測試**。

## done 判準

- `python verify/gate.py group_patch_stats` 印 `GREEN`(Tier A,無 E2E 要件)。
