# 09 round_ledger — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/09_round_ledger.md](../3_Architect_Design/09_round_ledger.md) · 里程碑 M9 · Tier A(核心)+ B(接線)
單元測試:`tests/test_round_ledger.py`(gate:`python verify/gate.py round_ledger`)
E2E:`tests/e2e/test_round_ledger_e2e.py`(`@pytest.mark.e2e`,**不進 gate / PG 自主修綠迴圈**)

> 設計反向閘門紀錄:PM 發現 (a) `load_rounds` 宣告回 list 又被 AC4 要 `_orphans` 鍵、(b) round_id「2 位亂數尾」
> 無法滿足 AC3「100 次全異」。architect 修為「load_rounds 回乾淨 list、孤兒下沉 `orphan_events()`」與
> 「round_id 用 uuid4 尾提供足夠熵」(見 ROADMAP 決策日誌 2026-07-04)。

## 單元 AC ↔ 測試

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | 生命週期:start→batch→readback→metric,四類欄位值全對(釘死) | `test_ac1_lifecycle_pinned` | 🔴 |
| AC2 | 多批多讀回累加:n_sent==7、n_labeled==7 | `test_ac2_accumulate` | 🔴 |
| AC3 | round_id 連呼 100 次全異、皆以 `r` 開頭 | `test_ac3_round_id_unique` | 🔴 |
| AC4 | 孤兒:掛不存在 round 的事件進 `orphan_events()`(len 1),load_rounds 乾淨 | `test_ac4_orphan` | 🔴 |
| AC5 | learning_curve:累積 x + 缺 metric 回合跳過 → `([10,60],[0.5,0.8])` | `test_ac5_learning_curve_pinned` | 🔴 |
| AC6 | 壞行容忍:load_rounds 照常 | `test_ac6_corrupt_tolerated` | 🔴 |
| AC7 | 無重依賴:乾淨子行程 import 不拉 numpy | `test_ac7_no_heavy_dependency` | 🔴 |

## 推導測試(設計未明列,PM 自加施壓)

| 測試 | 施壓的不變量 |
|------|-------------|
| `test_acD1_metric_last_wins_and_x_monotonic` | (1) 同名 metric 掛兩次→曲線取**最後一筆**(非 first/平均);(2) 曲線 x(累積標註量)**單調不減** → 逼出「metric 取 first / x 未累積」bug |

## GUI / 整合 E2E AC(Tier B done 要件;`/ux-test` 跑)

| AC | 內容 | 測試 |
|----|------|------|
| AC-G1 | 送標即開回合:📥「🔁 回合」出現卡,可見策略名與 `送出 N` | `test_send_opens_round_e2e` | ✅ |
| AC-G2 | 回讀入帳:同卡出現 `回讀 M · 變更 K`(數字對 fixture) | `test_readback_into_round_e2e` | ✅ |
| AC-G3 | 重啟 server 後時間軸卡仍在(讀自 rounds.jsonl) | `test_round_survives_restart_e2e` | ✅ |
| AC-G4 | 兩回合各掛 "acc" → 折線圖渲染 + 掛指標表單寫入帳本 | `test_learning_curve_renders_e2e` | ✅ |

> **E2E 狀態(2026-07-12)**:四條 AC-G 全綠、連兩次全套穩定綠。策略名經 cart 送標路徑=`cart`
> (`_send_to_labeling_ui` 的 `strategy` 空字串時沿用 source)。AC-G4 拆兩段:折線圖渲染(seed 兩回合
> 帶 acc 直接寫 rounds.jsonl)+ 掛指標表單驗證(填欄位 blur commit → 點掛上 → 驗帳本出現該事件、重試)。
> 共用 harness 與隔離 server 見 08 pool_registry 對應表註記。

## done 判準(Tier A 核心 + Tier B 接線)

- **單元綠**:`python verify/gate.py round_ledger` 印 `GREEN`。
- **AND 真實 E2E 綠**(接線完成後):`tests/e2e/test_round_ledger_e2e.py -m e2e`(AC-G1~G4)。
- 08 的 `mark_labeled` 整合在 08 的 AC-G3 驗,本表不重複。
