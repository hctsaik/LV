# 20 — group_patch_stats 技術設計(Good/Bad 群組 patch 差異統計 + null 校準)

對應 PRD:[2_PO_PRD/goodbad_patch_diff_prd.md](../2_PO_PRD/goodbad_patch_diff_prd.md) · 里程碑 M19 · **Tier A(純陣列運算,無 I/O)**

## 1. 目的(Purpose)

回答 M19 的核心統計問題:給定兩群**同 grid 對齊**的 patch 特徵,每個 patch 位置
「Good 群 vs Bad 群」差多少、差得穩不穩定、以及**整體到底有沒有超出隨機的差異**
(null 校準;G2 誠實判定的唯一依據)。純函式、決定論(給 seed)、只吃 numpy 陣列。

設計取捨(回答 PRD 開放問題 1):
- **差異分數用 Fisher 型分離度**(群間均值距 ÷ 群內散度),不是裸平均距離——
  PRD Must-3 要求含群內散度,否則群內雜訊大的 patch 會假高分。
- **穩定度**獨立於分數另計:「幾成 Bad 落在 Good 自身分布之外」,直接對應
  使用者要的白話句(「Bad 群 82% 在此處…」)。
- **null 校準用 permutation + max-統計量**:把兩群混洗重貼標籤 n_perm 次,
  取每次「全 patch 最大 z」構成 null 分布 → 控制多重比較(256 個 patch 同時檢定,
  不校正就會假陽性),p_global 對應「整體有無穩定差異」verdict。

## 2. I/O 契約

### `scripts/group_patch_stats.py`(依賴:numpy;禁 I/O、禁 import 其他專案模組)

```python
def group_diff_stats(
    feats_good: "np.ndarray",   # (Ng, P, D) float32/64,每列 L2 正規化(呼叫端保證)
    feats_bad: "np.ndarray",    # (Nb, P, D) 同上;P、D 必須與 good 一致
    *,
    n_perm: int = 200,          # permutation 次數(≥ 20,否則 ValueError)
    alpha: float = 0.05,        # 全域顯著水準(verdict = p_global < alpha)
    stab_q: float = 0.95,       # 穩定度門檻 = Good 自身距離分布的此分位
    seed: int = 0,              # np.random.default_rng(seed);全輸出決定論
) -> dict
```

回傳 dict(鍵名即契約):

| 鍵 | 形狀 | 意義 |
|---|---|---|
| `score` | (P,) | Fisher 分離度 `‖μg−μb‖ / (sg + sb + 1e-6)`;`s = sqrt(max(0, 1−‖μ‖²))`(L2 正規化下 `E‖x−μ‖² = 1−‖μ‖²`,不必展開逐樣本差) |
| `z` | (P,) | 校準分數 `(score − null_mean) / (null_std + 1e-9)`,null 統計逐 patch 由 permutation 取得 |
| `stability` | (P,) | Bad 中「到 `μg/‖μg‖` 的 cosine 距離 > Good 自身該距離的第 `stab_q` 分位」的比例 ∈ [0,1] |
| `dist_good` | (Ng, P) | 每張 Good 每 patch 到 `μg` 的 cosine 距離(給 21 挑代表樣本) |
| `dist_bad` | (Nb, P) | 每張 Bad 同上 |
| `p_global` | float | `(1 + #{perm_max_z ≥ obs_max_z}) / (1 + n_perm)` |
| `verdict` | bool | `p_global < alpha`(True=有穩定差異) |
| `threshold_z` | float | null「max-z」分布的 `1−alpha` 分位——**給 21 當進區域門檻**,天然控多重比較 |
| `null_max_z` | (n_perm,) | 供報表/稽核 |

實作要點:
- permutation 逐 patch 迴圈 + 矩陣化(perm 掩碼 `(n_perm, N) @ X_p (N, D)` 一次算完
  n_perm 個群均值),記憶體 O(N·D + n_perm·D),不整包展開 (n_perm, P, D)。
- null_mean/null_std 與 max-z 用同一批 permutation(省算;輕微樂觀偏差可接受,文件誠實註明)。
- 分數/距離全 float64 累加(fp16 快取讀回後先升精度,防 500 張累加誤差)。

## 3. 邊界條件與錯誤處理

| 情況 | 預期行為 |
|---|---|
| Ng < 2 或 Nb < 2 | ValueError(訊息含實際數量) |
| P 或 D 兩群不一致 | ValueError |
| 輸入含 NaN/Inf | ValueError(不靜默清洗——上游壞圖已由 safe_io 補零,零向量合法) |
| n_perm < 20 | ValueError |
| 某 patch 兩群完全相同(score=0, null_std=0) | z 定義為 0(分母 +1e-9 已保證不炸) |
| feats 未正規化 | 不檢查(呼叫端契約;22 保證) |

## 4. Acceptance Criteria

**單元(`tests/test_group_patch_stats.py`;合成陣列、免模型、全部固定 seed)**

共用 fixture:`_synth(Ng=40, Nb=40, P=25, D=16, shift_patch=None, shift_frac=1.0,
shift_scale=1.2, noise=0.15, seed=7)` — 每 patch 位置一個基底單位向量 + 高斯雜訊後
逐列 L2 正規化;`shift_patch` 給定時,Bad 群**前 `shift_frac` 比例**的樣本在該 patch
加固定位移向量(scale=shift_scale)再正規化。

- **AC1(植入 Top-1 + verdict)**:`shift_patch=7, shift_frac=1.0` →
  `argmax(score)==7`、`argmax(z)==7`、`verdict is True`、`p_global < 0.05`。
- **AC2(null 誠實)**:`shift_patch=None`(兩群同分布、不同抽樣)→ `verdict is False`
  且 `p_global ≥ 0.05`。
- **AC3(單張偶然 ≠ 穩定差異)**:`shift_frac=1/40`(40 張 Bad 只 1 張有位移)→
  `verdict is False` 且 `stability[7] < 0.2`。
- **AC4(穩定度量化)**:`shift_frac=0.8` → `abs(stability[7] − 0.8) ≤ 0.15`,
  且 `argmax(z)==7`、`verdict is True`。
- **AC5(群內散度入分母)**:同位移、`noise=0.05` vs `noise=0.45` 兩組 →
  `score_lownoise[7] > 1.5 * score_highnoise[7]`(平均距離裸分數無此性質)。
- **AC6(防呆)**:Ng=1 → ValueError;P 不一致 → ValueError;D 不一致 → ValueError;
  含 NaN → ValueError;n_perm=5 → ValueError。
- **AC7(決定論)**:同參數同 seed 跑兩次 → `score`/`z`/`p_global`/`stability`
  完全相等(np.array_equal / ==)。
- **AC8(輸出形狀契約)**:各鍵存在且形狀如 §2 表(一條測試掃全表)。

## 5. 給 /pm 的提示

- fixture 的位移 patch、比例、seed 全釘死,斷言才可精確;AC4 容差 ±0.15 是給
  正規化後幾何的餘裕,不是躲 flaky。
- AC5 兩組共用同一 seed 生成基底,只差 noise,比較才公平。
- 本模組禁 import scripts 其他模組——PM 測試直接 `import group_patch_stats`
  (tests/conftest.py 已把 scripts/ 入 sys.path)。
