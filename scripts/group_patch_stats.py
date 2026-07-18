"""group_patch_stats:兩群對齊 patch 特徵 → 每位置差異分數/穩定度/null 校準 verdict。

設計:3_Architect_Design/20_group_patch_stats.md(M19)。純 numpy、無 I/O、決定論(seed)。
分數 = Fisher 型分離度(群間均值距 ÷ 群內散度);null 校準 = permutation + max-z
(控 P 個 patch 同時檢定的多重比較);穩定度 = Bad 落在 Good 自身距離分布之外的比例。
null_mean/std 與 max-z 用同一批 permutation(省算;輕微樂觀偏差,設計已誠實註明)。
"""
from __future__ import annotations

import numpy as np


def _validate(feats_good, feats_bad, n_perm: int):
    g = np.asarray(feats_good)
    b = np.asarray(feats_bad)
    if g.ndim != 3 or b.ndim != 3:
        raise ValueError(f"feats 應為 3 維 (N,P,D):good {g.shape}、bad {b.shape}")
    if g.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError(f"每群至少 2 張:Ng={g.shape[0]}、Nb={b.shape[0]}")
    if g.shape[1:] != b.shape[1:]:
        raise ValueError(f"P/D 不一致:good {g.shape[1:]} vs bad {b.shape[1:]}")
    if not (np.isfinite(g).all() and np.isfinite(b).all()):
        raise ValueError("輸入含 NaN/Inf")
    if n_perm < 20:
        raise ValueError(f"n_perm 至少 20(現為 {n_perm})")
    return g.astype(np.float64), b.astype(np.float64)


def _fisher(mu_g: np.ndarray, mu_b: np.ndarray) -> np.ndarray:
    """群均值 → 分離度。L2 正規化列下 E‖x−μ‖² = 1−‖μ‖² → 群內散度免展開逐樣本。"""
    sep = np.linalg.norm(mu_g - mu_b, axis=-1)
    sg = np.sqrt(np.clip(1.0 - np.sum(mu_g * mu_g, axis=-1), 0.0, None))
    sb = np.sqrt(np.clip(1.0 - np.sum(mu_b * mu_b, axis=-1), 0.0, None))
    return sep / (sg + sb + 1e-6)


def group_diff_stats(feats_good, feats_bad, *, n_perm: int = 200,
                     alpha: float = 0.05, stab_q: float = 0.95,
                     seed: int = 0) -> dict:
    g, b = _validate(feats_good, feats_bad, n_perm)
    Ng, P, _D = g.shape
    Nb = b.shape[0]
    N = Ng + Nb

    mu_g = g.mean(axis=0)                       # (P, D)
    mu_b = b.mean(axis=0)
    score = _fisher(mu_g, mu_b)                 # (P,)

    # 穩定度:到 μg 方向的 cosine 距離,門檻 = Good 自身分布第 stab_q 分位
    mu_dir = mu_g / np.clip(np.linalg.norm(mu_g, axis=1, keepdims=True), 1e-12, None)
    dist_good = 1.0 - np.einsum("npd,pd->np", g, mu_dir)    # (Ng, P)
    dist_bad = 1.0 - np.einsum("npd,pd->np", b, mu_dir)     # (Nb, P)
    thr = np.quantile(dist_good, stab_q, axis=0)            # (P,)
    stability = (dist_bad > thr[None, :]).mean(axis=0)      # (P,)

    # permutation null:逐 patch、掩碼矩陣化(記憶體 O(N·D + n_perm·N))
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([g, b], axis=0)                 # (N, P, D)
    masks = np.zeros((n_perm, N), dtype=np.float64)
    for m in range(n_perm):
        masks[m, rng.permutation(N)[:Ng]] = 1.0
    inv = 1.0 - masks
    null_scores = np.empty((n_perm, P), dtype=np.float64)
    for p in range(P):
        Xp = pooled[:, p, :]                                # (N, D)
        null_scores[:, p] = _fisher(masks @ Xp / Ng, inv @ Xp / Nb)

    null_mean = null_scores.mean(axis=0)
    null_std = null_scores.std(axis=0)
    z = (score - null_mean) / (null_std + 1e-9)
    null_z = (null_scores - null_mean[None, :]) / (null_std[None, :] + 1e-9)
    null_max_z = null_z.max(axis=1)                         # (n_perm,)
    p_global = float((1 + int(np.sum(null_max_z >= float(z.max())))) / (1 + n_perm))

    return {
        "score": score,
        "z": z,
        "stability": stability,
        "dist_good": dist_good,
        "dist_bad": dist_bad,
        "p_global": p_global,
        "verdict": bool(p_global < alpha),
        "threshold_z": float(np.quantile(null_max_z, 1.0 - alpha)),
        "null_max_z": null_max_z,
    }
