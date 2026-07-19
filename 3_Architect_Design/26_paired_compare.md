# 26 `paired_compare` — 成對比較統計引擎(Tier A)

> M21。PRD [2_PO_PRD/recipe_compare_cost_workpoint_prd.md](../2_PO_PRD/recipe_compare_cost_workpoint_prd.md)。
> 純函式、無 I/O、無 streamlit;只吃 list/np.ndarray。實作 `scripts/paired_compare.py`。

## 1. 目的

兩份「同項目、不同做法」的品質數值 → 效應量+支撐數+(blocked)符號翻轉 permutation p
→ 三值結論,保守方向預設「看不出來」。

## 2. I/O 契約

```python
def paired_compare(
    a: Sequence[float], b: Sequence[float], *,
    higher_is_better: bool,
    groups: Sequence[str] | None = None,   # 每項所屬群組;None=不分組
    alpha: float = 0.05,
    n_min: int = 10,                       # 支撐門檻:項目數
    g_min: int = 6,                        # 支撐門檻:群組數(僅分組時檢)
    n_perm: int = 9999,                    # Monte Carlo 抽樣數(單位數 >12 時)
    seed: int = 0,
) -> dict
```

回傳(鍵固定):

```python
{
  "verdict": "a_better" | "b_better" | "inconclusive",
  "mean_diff": float,      # 定向效應量:>0 = A 較好(見資料流)
  "win_rate": float,       # 非平手項目中 A 較好的比例;全平手時 0.0
  "n": int,                # 配對項目數
  "n_ties": int,           # d_i == 0 的項目數
  "n_units": int,          # permutation 單位數(分組=群組數;不分組=n)
  "p": float,              # 雙尾 p
  "alpha": float,
  "exact": bool,           # True=窮舉(單位數<=12,2^U 全枚舉);False=Monte Carlo
  "blocked": bool,         # groups is not None
  "reasons": list[str],    # inconclusive 的理由(可多條;空=顯著結論)
  "method": str,           # method line(見 §5 AC8 固定片段)
}
```

## 3. 資料流

1. **定向差**:`d_i = a_i - b_i`(若 `higher_is_better`)否則 `d_i = b_i - a_i`
   → 恆為「正 = A 較好」。`mean_diff = mean(d)`。
2. **permutation 單位**:不分組=每個 i 一單位;分組=同群組所有 `d_i` **同時**翻號
   (blocked sign-flip,整群同動)。統計量=翻號後的 `mean(d)`。
3. **p 值**:單位數 `U ≤ 12` → 窮舉全部 `2^U` 翻號組合,
   `p = #(|stat_perm| >= |stat_obs|) / 2^U`(恆等式:觀測本身含在枚舉內,p>0)。
   `U > 12` → Monte Carlo `n_perm` 次(`np.random.default_rng(seed)`),
   `p = (#(|stat_perm| >= |stat_obs|) + 1) / (n_perm + 1)`。
4. **verdict(保守優先,依序檢查,命中即記 reason;所有 reason 都要收集不短路)**:
   - `n < n_min` → reason `"樣本不足"`
   - `groups` 給了且 `n_units < g_min` → reason `"群組不足"`
   - `mean_diff == 0` → reason `"無差異"`
   - `p > alpha` → reason `"未達顯著"`
   - reasons 非空 → `"inconclusive"`;否則 `mean_diff > 0 → "a_better"`,`< 0 → "b_better"`。

## 4. 邊界條件與錯誤處理

- `len(a) != len(b)`、任一為空、`groups` 給了但長度 ≠ `len(a)` → `ValueError`。
- `a`/`b` 含非有限值(NaN/Inf)→ `ValueError`(清洗是呼叫端 `run_pairing` 的事)。
- `alpha` 不在 (0,1)、`n_perm < 1` → `ValueError`。
- 全部 `d_i == 0`:`mean_diff = 0.0`、`p = 1.0`、verdict inconclusive(reason 無差異)。
- groups 只有 1 個群組:`n_units = 1 < g_min` → inconclusive(群組不足);不 crash。

## 5. Acceptance Criteria

- **AC1(G1 效應檢出,exact)**:`paired_compare([1.5]*12, [1.0]*12, higher_is_better=True)`
  → `verdict=="a_better"`、`mean_diff==0.5`、`win_rate==1.0`、`n==12`、`n_units==12`、
  `exact is True`、`p == 2/4096`、`reasons==[]`。
- **AC2(無差異誠實)**:`a=[1,2]*6`、`b=[2,1]*6`(d=+1,−1 交錯,mean=0),
  `higher_is_better=True` → `mean_diff==0.0`、`verdict=="inconclusive"`、
  `"無差異" in reasons`。
- **AC3(批次對照=G2②核心,exact)**:`d` 由 `a-b` 構成:前 6 項 `+1.0`(群組 `"W1"`)、
  後 6 項 `0.0`(群組 `"W2"`)(如 `a=[2]*6+[1]*6`, `b=[1]*6+[1]*6`):
  - 不分組:`p == 128/4096`(=0.03125 ≤ α)、`verdict=="a_better"`——天真法被批次騙。
  - 分組 `["W1"]*6+["W2"]*6`:`blocked is True`、`n_units==2`、`p==1.0`、
    `verdict=="inconclusive"`、`"群組不足" in reasons`——blocked 誠實說不知道。
- **AC4(blocked 可過門檻,exact)**:d 全 `+0.5`,12 項分 6 群組各 2 項
  → `n_units==6`、`exact is True`、`p == 2/64`、`n==12`、`verdict=="a_better"`。
- **AC5(樣本不足優先)**:d 全 `+1.0`、N=6、不分組 → `p == 2/64 ≤ α` 但
  `verdict=="inconclusive"` 且 `"樣本不足" in reasons`(保守蓋過小 p)。
- **AC6(方向翻轉)**:`paired_compare([0.1]*12, [0.2]*12, higher_is_better=False)`
  → `verdict=="a_better"`、`mean_diff==pytest.approx(0.1)`。
- **AC7(錯誤路徑)**:長度不一致 / 含 NaN / groups 長度不符 / alpha=1.5
  各自 `raises ValueError`。
- **AC8(method line 固定片段)**:任一成功呼叫的 `method` 同時含子字串
  `"單位="`、`"N="`、`"p="`、`"α="`、`"exploratory"`;分組時另含 `"群組"`。
- **AC9(MC 決定性)**:N=20 隨機 d(`default_rng(7)` 生成)呼叫兩次(同 `seed=0`)
  → 兩次回傳的 `p` 完全相等且 `exact is False`。
- **AC10(null 假陽性率,統計驗收=10 文件分級表)**:200 組 null 資料
  (`default_rng(s)`,s=0..199,N=20,d~Normal(0,1),不分組)→
  `mean(p <= 0.05) <= 0.075`(名目 α + 蒙地卡羅裕度)。
