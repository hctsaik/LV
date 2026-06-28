"""active_learning:主動學習取樣 Priority Score(架構文件第九節)。

主動學習的本質不是找模型最猶豫的樣本,而是找**最值得標註、最能幫系統成長**的樣本。
softmax entropy 對 OOD/未知常過度自信,故**距離訊號(novelty)一定納入**,不單靠 head 信心。

Priority = w_nov·Novelty(Normal Bank 異常分數) + w_bnd·Near-Decision-Boundary(head top1/top2 接近)
         + w_dis·Model-Disagreement(Normal Bank 說異常但 head 沒把握 = Unknown,資訊量高)
         ,再 × Production-Risk-Weight(關鍵 layer/tool)。
選樣時加 Cluster-Diversity(每群上限,挑各群代表,不挑一堆幾乎一樣的)。
(Temporal-Persistence 屬時序差分,本工具是單批影像 → 不在此模組;有的話可乘進 risk_weights。)
"""
from __future__ import annotations

import numpy as np


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def boundary_score(head_proba) -> np.ndarray:
    """近決策邊界程度:1 - (top1 - top2 機率差)。確定(one-hot)→0;前兩類接近/均勻 →趨近 1。"""
    P = np.asarray(head_proba, dtype=np.float32)
    if P.ndim != 2 or P.shape[1] < 2:
        return np.zeros(len(P), dtype=np.float32)
    part = np.sort(P, axis=1)[:, ::-1]
    margin = part[:, 0] - part[:, 1]
    return np.clip(1.0 - margin, 0.0, 1.0).astype(np.float32)


def priority_score(anomaly_scores, *, head_proba=None, anomaly_threshold=None,
                   risk_weights=None, w_novelty: float = 1.0,
                   w_boundary: float = 1.0, w_disagreement: float = 1.0) -> np.ndarray:
    """每物件主動學習優先分數(越高越值得送人工標註)。"""
    a = np.asarray(anomaly_scores, dtype=np.float32)
    N = len(a)
    novelty = _minmax(a)                                    # 距離訊號:離正常越遠越優先
    boundary = (boundary_score(head_proba) if head_proba is not None
                else np.zeros(N, dtype=np.float32))
    if head_proba is not None and anomaly_threshold is not None:
        conf = np.asarray(head_proba, dtype=np.float32).max(axis=1)
        # 被 Normal Bank 判異常,但 head 沒把握 → 不知道是哪種(Unknown),資訊量最高
        disagreement = np.where(a >= float(anomaly_threshold), 1.0 - conf, 0.0).astype(np.float32)
    else:
        disagreement = np.zeros(N, dtype=np.float32)
    pri = (float(w_novelty) * novelty + float(w_boundary) * boundary
           + float(w_disagreement) * disagreement)
    if risk_weights is not None:
        pri = pri * np.asarray(risk_weights, dtype=np.float32)
    return pri.astype(np.float32)


def select_for_labeling(priority, *, k: int, cluster_labels=None,
                        max_per_cluster: int = 2) -> list:
    """依 priority 取前 k,加 cluster diversity:每群最多 max_per_cluster 個(挑各群代表)。
    若因群上限湊不足 k,再放寬補滿。回選中的物件索引(依優先序)。"""
    pri = np.asarray(priority, dtype=np.float32)
    order = list(np.argsort(-pri, kind="stable"))
    k = int(min(max(k, 0), len(order)))
    if cluster_labels is None:
        return [int(i) for i in order[:k]]
    cl = np.asarray(cluster_labels)
    cnt: dict = {}
    out: list = []
    for i in order:
        c = cl[int(i)]
        if cnt.get(c, 0) >= max_per_cluster:
            continue
        out.append(int(i))
        cnt[c] = cnt.get(c, 0) + 1
        if len(out) >= k:
            return out
    for i in order:                                        # 群上限湊不足 → 放寬補滿
        if int(i) not in out:
            out.append(int(i))
            if len(out) >= k:
                break
    return out
