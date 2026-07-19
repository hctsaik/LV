"""成本工作點 what-if 引擎(M21 模組 27,Tier A)。

異常分數 + 人工確認標籤 + 相對成本比 → 閾值掃描的 FN/FP/工作量/預期相對代價
曲線與最低代價點。**只算不套**:輸出建議,絕不寫任何設定(GUI 端 G4 鐵則)。
設計:3_Architect_Design/27_cost_curve.md。純函式、無 I/O、無 streamlit。
判定規則(釘死):score >= t → 判異常(flag)。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def cost_curve(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    cost_miss: float,
    cost_fa: float,
    thresholds: Sequence[float] | None = None,
    current_threshold: float | None = None,
) -> dict:
    s = np.asarray(list(scores), dtype=float)
    lab_raw = list(labels)
    if s.ndim != 1 or s.size == 0 or s.size != len(lab_raw):
        raise ValueError("scores/labels 必須是等長非空序列")
    if not np.isfinite(s).all():
        raise ValueError("scores 含非有限值(NaN/Inf)")
    lab = np.asarray(lab_raw)
    if not set(np.unique(lab).tolist()) <= {0, 1}:
        raise ValueError("labels 僅允許 0(正常)/1(瑕疵)")
    lab = lab.astype(int)
    n = int(s.size)
    n_pos = int((lab == 1).sum())
    n_neg = int((lab == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"需同時有已確認的瑕疵與正常樣本(現況:瑕疵 {n_pos}、正常 {n_neg})")
    if float(cost_miss) <= 0 or float(cost_fa) <= 0:
        raise ValueError("cost_miss/cost_fa 必須 > 0")

    if thresholds is None:
        u = np.unique(s)                      # 升冪唯一分數
        cuts = ([float(u[0]) - 1.0]
                + [float(u[i] + u[i + 1]) / 2.0 for i in range(len(u) - 1)]
                + [float(u[-1]) + 1.0])       # 恰 K+1 點:全標 → 全不標
    else:
        cuts = sorted({float(t) for t in thresholds})
        if not cuts:
            raise ValueError("thresholds 不可為空")

    def _counts(t: float) -> tuple[int, int, int]:
        flag = s >= t
        fn_i = int(((lab == 1) & ~flag).sum())
        fp_i = int(((lab == 0) & flag).sum())
        return fn_i, fp_i, int(flag.sum())

    fn: list[int] = []
    fp: list[int] = []
    flagged: list[int] = []
    cost: list[float] = []
    for t in cuts:
        fn_i, fp_i, fl_i = _counts(t)
        fn.append(fn_i)
        fp.append(fp_i)
        flagged.append(fl_i)
        cost.append(float(cost_miss) * fn_i + float(cost_fa) * fp_i)

    best_idx = int(np.argmin(cost))           # 平手取最小 index(=最鬆之最優)

    current = None
    if current_threshold is not None:
        t = float(current_threshold)
        fn_i, fp_i, fl_i = _counts(t)
        current = {"threshold": t, "fn": fn_i, "fp": fp_i, "flagged": fl_i,
                   "cost": float(cost_miss) * fn_i + float(cost_fa) * fp_i}

    method = (f"判定規則:score >= t → 標記異常;成本比 漏檢:誤報 = "
              f"{float(cost_miss):g}:{float(cost_fa):g};"
              "僅試算,不會改變任何設定")

    return {
        "thresholds": cuts,
        "fn": fn, "fp": fp, "flagged": flagged,
        "cost": cost,
        "best_idx": best_idx,
        "n": n, "n_pos": n_pos, "n_neg": n_neg,
        "current": current,
        "method": method,
    }
