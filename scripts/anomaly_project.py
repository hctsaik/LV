"""anomaly_project:跨資料夾投影器(把新資料映射回舊資料分佈)。

綜合設計裁決:
- **transform-into-fixed-basis**(非 re-fit-together):建 bank 那次對舊 obj_emb 算
  PCA-via-SVD,把 mean + components(=Vt[:2])存下來;讀回那次對新資料算
  (emb_new - mean) @ components.T,新點投到舊座標軸 → 「映射『回』舊分佈」。
- **l2norm=False**(預設,寫死):對齊 app.py:1301 現況散點(raw obj_emb 的 SVD)。
- **密集/離群判定走 D 維 cosine**(classify_dense),2D 座標只負責畫位置(2D 壓扁會假陰)。
- tau(密集區門檻)對「舊 good 對舊 good」算 kNN 分位,**去自身**(self-dist=0 會拉低)。

投影用 obj_emb(物件級,patch / object 兩種分數依據都有);與 patch memory bank 是不同層級。
"""
from __future__ import annotations

import numpy as np


def _l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def fit_projector(obj_emb, good_mask=None, *, l2norm: bool = False, dim_out: int = 2,
                  max_good: int = 2000, max_ref: int = 2000,
                  tau_k: int = 5, tau_percentile: float = 95.0,
                  seed: int = 42) -> dict:
    """對舊資料 obj_emb 擬合固定投影基底 + 密度判定所需的 good 子集 + tau。"""
    X = np.asarray(obj_emb, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("empty obj_emb")
    if l2norm:
        X = _l2n(X)
    Dd = X.shape[1]
    mean = X.mean(axis=0)
    c = X - mean
    try:
        u, s, vt = np.linalg.svd(c, full_matrices=False)
        comp = vt[:dim_out]
        ref_coords = u[:, :dim_out] * s[:dim_out]
    except np.linalg.LinAlgError:                      # 退化 → 取原始前 dim_out 維
        comp = np.eye(dim_out, Dd, dtype=np.float32)
        ref_coords = c[:, :dim_out]
    comp = np.ascontiguousarray(comp[:dim_out], dtype=np.float32)
    if comp.shape[0] < dim_out:                        # rank < dim_out → 補零行
        comp = np.vstack([comp, np.zeros((dim_out - comp.shape[0], Dd), np.float32)])

    rng = np.random.default_rng(seed)
    good = X if good_mask is None else X[np.asarray(good_mask, dtype=bool)]
    if len(good) == 0:
        good = X
    if len(good) > max_good:
        good = good[np.sort(rng.choice(len(good), max_good, replace=False))]
    tau = compute_tau(good, k=tau_k, percentile=tau_percentile)

    rc = np.ascontiguousarray(ref_coords[:, :dim_out], dtype=np.float32)
    if len(rc) > max_ref:
        rc = rc[np.sort(rng.choice(len(rc), max_ref, replace=False))]

    return {
        "mean": mean.astype(np.float32),
        "components": comp,
        "ref_coords": rc,                              # 舊投影座標(畫灰底)
        "good_obj_emb": good.astype(np.float16).astype(np.float32),  # 密度判定底(fp16 存)
        "tau": float(tau),
        "l2norm": bool(l2norm),
    }


def transform_new(basis: dict, emb_new) -> np.ndarray:
    """把新資料 obj_emb 投到舊基底:(emb_new - mean) @ components.T → (N, dim_out)。"""
    X = np.asarray(emb_new, dtype=np.float32)
    mean = np.asarray(basis["mean"], dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != mean.shape[0]:
        raise ValueError(f"obj_emb 維度 {X.shape} 與投影基底 D={mean.shape[0]} 不符")
    if basis.get("l2norm"):
        X = _l2n(X)
    return ((X - mean) @ np.asarray(basis["components"], np.float32).T).astype(np.float32)


def compute_tau(good, *, k: int = 5, percentile: float = 95.0,
                include_self: bool = False) -> float:
    """舊 good 對舊 good 的 kNN(cosine)平均距離之分位 → 密集區門檻。
    預設 include_self=False(去自身;否則 self-dist=0 拉低分位 → 新點普遍誤判離群)。"""
    g = np.asarray(good, dtype=np.float32)
    if g.ndim != 2 or g.shape[0] < 2:
        return float("inf")                            # 無法判定 → 不擋
    from sklearn.neighbors import NearestNeighbors
    kk = min(k + 1, len(g))
    nn = NearestNeighbors(n_neighbors=kk, metric="cosine").fit(g)
    d, _ = nn.kneighbors(g)                            # 第 0 欄是自己(dist≈0)
    use = d[:, :max(1, kk - 1)] if include_self else d[:, 1:]
    return float(np.percentile(use.mean(axis=1), percentile))


def classify_dense(emb_new, good_obj_emb, tau, *, k: int = 5) -> np.ndarray:
    """新資料每點:對舊 good 的 kNN(cosine)平均距離 ≤ tau → 落在密集區(正常)。
    用 **D 維 cosine**,不是 2D 座標(2D 壓扁會把離群假陰)。"""
    g = np.asarray(good_obj_emb, dtype=np.float32)
    X = np.asarray(emb_new, dtype=np.float32)
    if g.shape[0] == 0 or not np.isfinite(tau):
        return np.ones(len(X), dtype=bool)             # 無法判定 → 不擋
    from sklearn.neighbors import NearestNeighbors
    kk = min(k, len(g))
    nn = NearestNeighbors(n_neighbors=kk, metric="cosine").fit(g)
    d, _ = nn.kneighbors(X)
    return d.mean(axis=1) <= float(tau)


def nearest_ref(emb_new, ref_emb):
    """新資料每點最像哪個舊參照點:回 (indices, cosine 距離)。連線端點用,不是 mean-of-k。"""
    R = np.asarray(ref_emb, dtype=np.float32)
    X = np.asarray(emb_new, dtype=np.float32)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(R)
    d, idx = nn.kneighbors(X)
    return idx[:, 0], d[:, 0]


def save_projector(path, basis: dict) -> None:
    np.savez_compressed(
        str(path),
        mean=np.asarray(basis["mean"], np.float32),
        components=np.asarray(basis["components"], np.float32),
        ref_coords=np.asarray(basis["ref_coords"], np.float32),
        good_obj_emb=np.asarray(basis["good_obj_emb"], np.float16),
        tau=np.float32(basis["tau"]),
        l2norm=np.array(bool(basis["l2norm"])),
    )


def load_projector(path) -> dict:
    d = np.load(str(path))
    return {
        "mean": d["mean"].astype(np.float32),
        "components": d["components"].astype(np.float32),
        "ref_coords": d["ref_coords"].astype(np.float32),
        "good_obj_emb": d["good_obj_emb"].astype(np.float32),
        "tau": float(d["tau"]),
        "l2norm": bool(d["l2norm"]),
    }
