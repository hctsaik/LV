# 11 hybrid_sampler — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/11_hybrid_sampler.md](../3_Architect_Design/11_hybrid_sampler.md) · 里程碑 M11 · Tier A(核心)+ B(接線)
單元測試:`tests/test_hybrid_sampler.py`(gate:`python verify/gate.py hybrid_sampler`)
E2E:`tests/e2e/test_hybrid_sampler_e2e.py`(`@pytest.mark.e2e`,**不進 gate**)

> 實作註記:`interaction.farthest_point_sampling(emb, n, seed_indices)` 回的是**局部索引**且
> **不含 seed**。依設計 §6 授權,`hybrid_sampler` 內包一層薄 adapter(對候選池的子矩陣跑 FPS
> 再映回原索引),**不改既有函式簽名**。

## 單元 AC ↔ 測試

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | 長度與範圍:N=100,k=10 → 10 個相異索引 ∈ [0,100) | `test_ac1_length_and_range` | 🔴 |
| AC2 | 不確定門檻:選中皆 ∈ top-(oversample×k),候選池外永不入選 | `test_ac2_uncertainty_pool` | 🔴 |
| AC3 | **多樣性勝 top-K(核心)**:純 top-K 近重複 45 對;混合選樣 **< 其半數** | `test_ac3_diversity_beats_topk` | 🔴 |
| AC4 | exclude:top-3 剔除後與選中集合交集為空,仍補滿到 k | `test_ac4_exclude` | 🔴 |
| AC5 | 配額:quota={rare:2}、池內 rare 3 張 → 選中 rare≥2 且正是池內最不確定的 2 張 | `test_ac5_quota` | 🔴 |
| AC6 | 配額不足**誠實**:要 5 只有 2 → rare==2(不硬湊),總數仍==k(FPS 補滿) | `test_ac6_quota_underfilled_honest` | 🔴 |
| AC7 | 決定論:同輸入重呼 20 次逐位相等 | `test_ac7_deterministic` | 🔴 |
| AC8 | 防呆:len 不符 → ValueError;k=0 → `[]` | `test_ac8_guards` | 🔴 |

## 推導測試(設計未明列,PM 自加施壓)

| 測試 | 施壓的不變量 |
|------|-------------|
| `test_md1_monotone_uncertainty_invariance` | 選樣只依 uncertainty 的**排序**;正仿射變換(a·u+b, a>0)不改名次 → 輸出逐位相等 → 逼出「拿絕對值進 FPS / 門檻」 |
| `test_md2_near_dup_monotonic_in_threshold` | `near_duplicate_pairs` 對 thr **單調非遞減**(20 → 45 跨兩個距離尺度)→ 逼出「`<` 寫成 `>`」「距離取負」 |

## GUI / 整合 E2E AC(Tier B done 要件)

| AC | 內容 | 測試 |
|----|------|------|
| AC-G1 | 策略可用:選「混合（不確定×多樣）」、K=6 → 畫廊恰 6 張、caption 出現「近重複對」數字 | `test_hybrid_strategy_runs_e2e` |
| AC-G2 | **接 08**:先把部分樣本直寫 registry 標為已送標 → 重新選樣該樣本不在結果;取消勾「剔除已送標」→ 回到可入選 | `test_hybrid_excludes_handled_e2e` |
| AC-G3 | **接 09**:送標後回合卡策略名含 `hybrid:` | `test_hybrid_send_opens_round_e2e` |

> AC-G1 依設計 §6 提示用「離群度」當不確定來源(既有訊號,不跨模組阻塞);「低信心」來源需
> 側欄填了模型預測資料夾(10 號)才出現,由 10 號的 E2E 覆蓋。

## done 判準

- **單元綠**:`python verify/gate.py hybrid_sampler` 印 `GREEN`。
- **AND 真實 E2E 綠**:AC-G1~G3。AC-G1 的「近重複對」caption 是**把改善攤在使用者眼前**的
  產品承諾——不是裝飾,故列 E2E 斷言。
