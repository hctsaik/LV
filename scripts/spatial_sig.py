"""單片晶圓空間簽名統計引擎(M22 模組 30,Tier A)。

單片 die 格點+缺陷標記 → 三簽名(聚一團/偏邊緣/線狀)統計量 + MC p → 四值誠實結論。
設計:3_Architect_Design/30_spatial_sig.md。純函式、無 I/O、無 streamlit。

統計鐵則(09/10 裁決):null=within-wafer / die-mask 受限重排——只在**這片實際
存在的 die 位置**之間重排缺陷標記、保缺陷數 k;嚴禁全域 shuffle、嚴禁假設完整格點。
G2 精神:門檻不足/未達顯著一律「看不出模式」。
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12
_CHUNK = 256          # 重排批次(clustered 兩兩距離的記憶體上限控制)
_TIE_ORDER = ("linear", "edge", "clustered")   # 平手優先序:線狀最特異


def _stats_for(rel: np.ndarray, r_norm: np.ndarray, idx: np.ndarray):
    """idx: (P, k) 各重排的缺陷 die 索引 → (clu, edge, lin) 各 (P,)。"""
    pts = rel[idx]                                  # (P, k, 2)
    k = pts.shape[1]
    # clustered:兩兩歐氏距離平均(越小越聚)
    diff = pts[:, :, None, :] - pts[:, None, :, :]  # (P, k, k, 2)
    dd = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(k, 1)
    clu = dd[:, iu[0], iu[1]].mean(axis=1)
    # edge:到質心距離 / R 的平均(越大越偏邊)
    edge = r_norm[idx].mean(axis=1)
    # linear:2×2 共變異數 λ1/(λ1+λ2)(閉式特徵值)
    c = pts - pts.mean(axis=1, keepdims=True)
    a = (c[:, :, 0] ** 2).mean(axis=1)
    d = (c[:, :, 1] ** 2).mean(axis=1)
    b = (c[:, :, 0] * c[:, :, 1]).mean(axis=1)
    tr = a + d
    root = np.sqrt(np.maximum((a - d) ** 2 + 4 * b ** 2, 0.0))
    lam1 = (tr + root) / 2.0
    lin = np.where(tr <= _EPS, 0.5, lam1 / np.where(tr <= _EPS, 1.0, tr))
    return clu, edge, lin


def spatial_signature(
    dies: list, *, alpha: float = 0.05,
    n_min: int = 20, k_min: int = 5,
    n_perm: int = 2000, seed: int = 0,
) -> dict:
    if not dies:
        raise ValueError("dies 為空")
    try:
        coords = np.asarray([[d["x"], d["y"]] for d in dies], dtype=float)
        flags = np.asarray([d["flag"] for d in dies])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"dies 需含 x/y/flag 且可轉數值:{e}") from e
    if not np.isfinite(coords).all():
        raise ValueError("座標含非有限值(NaN/Inf)")
    if not set(np.unique(flags).tolist()) <= {0, 1}:
        raise ValueError("flag 僅允許 0/1")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha 必須在 (0,1):{alpha}")
    if int(n_perm) < 1 or int(n_min) < 1 or int(k_min) < 1:
        raise ValueError("n_perm/n_min/k_min 必須 ≥1")

    flags = flags.astype(int)
    n = int(len(dies))
    k = int(flags.sum())
    method = (f"空間簽名 permutation(within-wafer die-mask 受限重排、保 k;"
              f"N={n},k={k},MC {int(n_perm)},α={float(alpha):g}"
              f"(Bonferroni /3);exploratory)")
    base = {"n": n, "k": k, "alpha": float(alpha), "method": method}

    # 保守門檻(收集全部理由,不短路)
    reasons: list[str] = []
    if n < int(n_min):
        reasons.append(f"die 太少(N={n} < {int(n_min)})")
    if k < int(k_min):
        reasons.append(f"缺陷太少(k={k} < {int(k_min)})")
    if n - k < 5:
        reasons.append(f"無對比(正常 die 僅 {n - k})")
    centroid = coords.mean(axis=0)
    rel = coords - centroid
    R = float(np.sqrt((rel ** 2).sum(axis=1)).max())
    if not reasons and R <= _EPS:
        reasons.append("座標退化(所有 die 同一點)")
    if reasons:
        return {**base, "verdict": "none", "p": None, "stats": None,
                "reasons": reasons}

    r_norm = np.sqrt((rel ** 2).sum(axis=1)) / R
    obs_idx = np.flatnonzero(flags == 1)[None, :]
    oc, oe, ol = (v[0] for v in _stats_for(rel, r_norm, obs_idx))

    rng = np.random.default_rng(int(seed))
    cnt = {"clustered": 0, "edge": 0, "linear": 0}
    done = 0
    while done < int(n_perm):
        m = min(_CHUNK, int(n_perm) - done)
        # 每列=一次受限重排:argsort 隨機值取前 k(不放回、只在本片 die 位置內)
        idx = np.argsort(rng.random((m, n)), axis=1)[:, :k]
        pc, pe, pl = _stats_for(rel, r_norm, idx)
        cnt["clustered"] += int((pc <= oc + _EPS).sum())
        cnt["edge"] += int((pe >= oe - _EPS).sum())
        cnt["linear"] += int((pl >= ol - _EPS).sum())
        done += m
    p = {s: (cnt[s] + 1) / (int(n_perm) + 1) for s in cnt}
    stats = {"clustered": float(oc), "edge": float(oe), "linear": float(ol)}

    thr = float(alpha) / 3.0
    sig = [s for s in _TIE_ORDER if p[s] <= thr]
    if not sig:
        return {**base, "verdict": "none", "p": p, "stats": stats,
                "reasons": [f"未達顯著(最小 p={min(p.values()):.4g} > α/3={thr:.4g})"]}
    best = min(sig, key=lambda s: (p[s], _TIE_ORDER.index(s)))
    return {**base, "verdict": best, "p": p, "stats": stats, "reasons": []}
