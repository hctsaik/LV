"""anomaly_classify:異常分數(+少量確認標籤)→ 門檻 → 整批好/壞 + 排序。

設計:3_Architect_Design/04_anomaly_classify.md。
門檻三路:有 good+bad → Youden's J(以相鄰分數中點為候選,落在類間);只有 bad → 最低確認壞分數之下;
無確認 → (1-contamination) 百分位。AUROC 僅在有雙類確認時提供。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _youden_threshold(scores: list[float], y: list[int]) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    uniq = np.unique(s)
    cands = list((uniq[:-1] + uniq[1:]) / 2.0) + [uniq[0] - 1e-9, uniq[-1] + 1e-9]
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    best_t, best_j = cands[0], -np.inf
    for t in cands:
        pred = s >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        j = tp / n_pos - fp / n_neg
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)


def classify(scores, *, confirmed: dict | None = None,
             contamination: float = 0.05) -> dict:
    scores = np.asarray(scores, dtype=float)
    N = int(len(scores))
    if N == 0:
        return {"threshold": float("inf"), "labels": np.array([], dtype=object),
                "ranking": [], "auroc": None, "n_bad": 0}

    contamination = min(max(float(contamination), 0.0), 1.0)
    valid: dict[int, str] = {}
    for i, v in (confirmed or {}).items():
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        if 0 <= i < N and v in ("good", "bad"):
            valid[i] = v
    goods = [i for i in valid if valid[i] == "good"]
    bads = [i for i in valid if valid[i] == "bad"]

    auroc = None
    if goods and bads:
        keys = sorted(valid)
        y = [1 if valid[i] == "bad" else 0 for i in keys]
        s = [float(scores[i]) for i in keys]
        auroc = float(roc_auc_score(y, s))
        threshold = _youden_threshold(s, y)
    elif bads:
        threshold = float(min(scores[i] for i in bads)) - 1e-9
    else:
        threshold = float(np.quantile(scores, 1.0 - contamination))

    labels = np.where(scores >= threshold, "bad", "good").astype(object)
    ranking = sorted(range(N), key=lambda i: (-scores[i], i))
    return {"threshold": threshold, "labels": labels, "ranking": ranking,
            "auroc": auroc, "n_bad": int((labels == "bad").sum())}
