"""active_loop:主動學習標註迴圈(設計 3_Architect_Design/M5_active_loop.md)。

把 M2 Normal Bank + M3 分類頭 + M4 取樣佇列 從「開迴圈組件」串成**閉迴圈**:
佇列選樣 → 人工標 → 回流(擴 Normal Bank + 重訓 head)→ 量測學習曲線 → 重複,
並提供「弱類定向優先(分類頭最混淆者最該標)」與「曲線走平就停」的可量測準則。

實證(雙-split 完整測試 S7):uncertainty sampling 比隨機省 ~60-75% 標註(跨 valid/test 都成立)。
"""
from __future__ import annotations

import numpy as np


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def entropy_score(head_proba) -> np.ndarray:
    """歸一化預測熵 ∈ [0,1](÷log C):one-hot→0、均勻→1。分類頭不確定度,高=最該標。"""
    P = np.clip(np.asarray(head_proba, dtype=float), 1e-12, 1.0)
    if P.ndim != 2 or P.shape[1] < 2:
        return np.zeros(len(P), dtype=np.float32)
    H = -(P * np.log(P)).sum(axis=1)
    return (H / np.log(P.shape[1])).astype(np.float32)


def confusion_targeted_priority(anomaly_scores, head_proba, *, anomaly_threshold=None,
                                w_novelty: float = 1.0, w_entropy: float = 1.0) -> np.ndarray:
    """弱類定向主動學習優先 = w_novelty·Novelty(離正常,minmax)+ w_entropy·Entropy(分類頭最混淆)。
    對齊「最該標的是分類頭最不確定/混淆 + 離正常遠的未知」。head_proba=None → 熵項為 0(純 novelty)。"""
    a = np.asarray(anomaly_scores, dtype=float)
    nov = _minmax(a)
    ent = (entropy_score(head_proba) if head_proba is not None
           else np.zeros(len(a), dtype=np.float32))
    return (float(w_novelty) * nov + float(w_entropy) * ent).astype(np.float32)


def label_efficiency_curve(pool_emb, pool_labels, eval_emb, eval_labels, *,
                           strategy: str = "active", seed_n: int = 60, batch: int = 60,
                           rounds: int = 8, seed: int = 0) -> list:
    """模擬標註迴圈:從 pool 以 strategy 逐輪選樣標註、重訓 head、在 eval 量 balanced_acc。
    strategy: 'active'(uncertainty sampling,挑最低信心)或 'random'。回 [[n_labels, acc], ...]。"""
    from sklearn.metrics import balanced_accuracy_score

    from dino_head import predict_head, train_head
    pe = np.asarray(pool_emb, dtype=np.float32)
    pl = np.asarray(pool_labels)
    ee = np.asarray(eval_emb, dtype=np.float32)
    rng = np.random.default_rng(seed)
    classes = sorted(set(pl.tolist()))
    seed_idx = [int(rng.choice(np.where(pl == c)[0])) for c in classes]   # 每類至少 1 個
    rest = np.array([i for i in range(len(pl)) if i not in set(seed_idx)])
    if len(rest) and seed_n > len(seed_idx):
        seed_idx += [int(i) for i in rng.choice(rest, size=min(seed_n - len(seed_idx), len(rest)),
                                                replace=False)]
    labeled = set(seed_idx)
    curve: list = []
    for _ in range(int(rounds)):
        ls = sorted(labeled)
        if len(set(pl[ls].tolist())) < 2:
            break
        h = train_head(pe[ls], pl[ls])
        pa, _, _ = predict_head(h, ee)
        curve.append([len(labeled), round(float(balanced_accuracy_score(eval_labels, pa)), 4)])
        unl = np.array([i for i in range(len(pl)) if i not in labeled])
        if len(unl) == 0:
            break
        if strategy == "active":                              # uncertainty sampling:最低信心最該標
            _, cf, _ = predict_head(h, pe[unl])
            pick = unl[np.argsort(cf)[:batch]]
        else:                                                 # random baseline
            pick = rng.choice(unl, size=min(batch, len(unl)), replace=False)
        labeled |= {int(i) for i in pick}
    return curve


def should_stop_labeling(curve, *, patience: int = 2, min_delta: float = 0.01) -> bool:
    """停止準則:連續 patience 輪 balanced_acc 提升 < min_delta(曲線走平)→ True 建議停止。"""
    if len(curve) < int(patience) + 1:
        return False
    accs = [a for _, a in curve]
    deltas = [accs[i] - accs[i - 1] for i in range(len(accs) - int(patience), len(accs))]
    return all(d < float(min_delta) for d in deltas)


def round_summary(confirmed_labels) -> dict:
    """本輪標註狀態:每類已標數 + 總數(給迴圈面板)。接受 list 或 dict(value 為 label)。"""
    vals = (list(confirmed_labels.values()) if isinstance(confirmed_labels, dict)
            else list(confirmed_labels))
    per: dict = {}
    for v in vals:
        per[v] = per.get(v, 0) + 1
    return {"total": len(vals), "per_class": per}
