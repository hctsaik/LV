"""成對比較統計引擎(M21 模組 26,Tier A)。

兩份「同項目、不同做法」的品質數值 → 效應量 + 支撐數 +(blocked)符號翻轉
permutation p → 三值結論;保守方向預設「看不出來」。
設計:3_Architect_Design/26_paired_compare.md。純函式、無 I/O、無 streamlit。

統計語義(exploratory):產線資料交換性通常不成立,p 值只用於探索性排序,
不是嚴格推論;有群組欄位時整群同動(blocked sign-flip)防批次效應假差異。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

_EXHAUSTIVE_MAX_UNITS = 12   # 2^12=4096 → 單位數以內全枚舉,p 可 exact


def paired_compare(
    a: Sequence[float], b: Sequence[float], *,
    higher_is_better: bool,
    groups: Sequence[str] | None = None,
    alpha: float = 0.05,
    n_min: int = 10,
    g_min: int = 6,
    n_perm: int = 9999,
    seed: int = 0,
) -> dict:
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    if a_arr.ndim != 1 or a_arr.shape != b_arr.shape or a_arr.size == 0:
        raise ValueError("a/b 必須是等長非空一維序列")
    if not (np.isfinite(a_arr).all() and np.isfinite(b_arr).all()):
        raise ValueError("a/b 含非有限值(NaN/Inf);請先在呼叫端清洗(run_pairing)")
    n = int(a_arr.size)
    if groups is not None and len(groups) != n:
        raise ValueError(f"groups 長度({len(groups)})必須等於項目數({n})")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha 必須在 (0,1):{alpha}")
    if int(n_perm) < 1:
        raise ValueError(f"n_perm 必須 ≥1:{n_perm}")

    # 定向差:恆為「正 = A 較好」
    d = (a_arr - b_arr) if higher_is_better else (b_arr - a_arr)
    mean_diff = float(d.mean())
    nonzero = d[d != 0.0]
    n_ties = int(n - nonzero.size)
    win_rate = float((nonzero > 0).mean()) if nonzero.size else 0.0

    # permutation 單位:不分組=每項一單位;分組=同群組整包同動(blocked)
    blocked = groups is not None
    if blocked:
        sums: dict = {}
        for g, val in zip(groups, d):
            sums[g] = sums.get(g, 0.0) + float(val)
        unit_sums = np.asarray(list(sums.values()), dtype=float)
    else:
        unit_sums = d.astype(float)
    n_units = int(unit_sums.size)

    # 統計量用 |sum(d)|(與 |mean| 等價:n 固定);窮舉含觀測本身 → p>0
    obs = abs(float(unit_sums.sum()))
    eps = 1e-12
    if n_units <= _EXHAUSTIVE_MAX_UNITS:
        exact = True
        m = 1 << n_units
        # 4096×12 以內,直接矩陣枚舉
        masks = np.arange(m)[:, None] >> np.arange(n_units)[None, :]
        signs = 1.0 - 2.0 * (masks & 1)          # bit=0 → +1、bit=1 → −1
        stats = np.abs(signs @ unit_sums)
        p = float((stats >= obs - eps).sum()) / m
    else:
        exact = False
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_perm), n_units))
        stats = np.abs(signs @ unit_sums)
        p = (int((stats >= obs - eps).sum()) + 1) / (int(n_perm) + 1)

    # verdict:保守優先,理由全收集不短路
    reasons: list[str] = []
    if n < int(n_min):
        reasons.append(f"樣本不足(N={n} < {int(n_min)})")
    if blocked and n_units < int(g_min):
        reasons.append(f"群組不足(G={n_units} < {int(g_min)})")
    if mean_diff == 0.0:
        reasons.append("無差異")
    if p > float(alpha) and mean_diff != 0.0:
        reasons.append(f"未達顯著(p={p:.4g} > α={float(alpha):g})")
    if reasons:
        verdict = "inconclusive"
    else:
        verdict = "a_better" if mean_diff > 0 else "b_better"

    unit_desc = f"群組×{n_units}" if blocked else f"項目×{n_units}"
    method = (f"配對符號翻轉 permutation(單位={unit_desc},N={n},"
              f"{'窮舉' if exact else f'MC {int(n_perm)}'},p={p:.4g},"
              f"α={float(alpha):g};exploratory)")

    return {
        "verdict": verdict,
        "mean_diff": mean_diff,
        "win_rate": win_rate,
        "n": n,
        "n_ties": n_ties,
        "n_units": n_units,
        "p": float(p),
        "alpha": float(alpha),
        "exact": exact,
        "blocked": blocked,
        "reasons": reasons,
        "method": method,
    }
