"""bootstrap_cluster:物件 embedding 分群 → 正常密群 vs 離群候選。

設計:3_Architect_Design/03_bootstrap_cluster.md。先 L2 正規化再用 euclidean(等價 cosine)。
候選 = noise 或所屬群小於 small_thresh;small_thresh 含 `0.5*最大群` 項,
讓「明顯小於主群的少數群」一律成候選(對 1-stage 抓少數缺陷關鍵)。
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN


def _l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def cluster_objects(emb, *, min_cluster_size: int | None = None,
                    small_cluster_frac: float = 0.05) -> dict:
    emb = np.asarray(emb, dtype=np.float32)
    N = int(len(emb))
    if N < 2:
        cand = np.zeros(N, dtype=bool)
        return {"labels": np.zeros(N, dtype=int), "candidate_mask": cand,
                "n_clusters": (1 if N == 1 else 0), "normal_mask": ~cand}

    mcs = int(min_cluster_size or max(5, round(0.05 * N)))
    labels = HDBSCAN(min_cluster_size=mcs, metric="euclidean").fit_predict(_l2n(emb))

    sizes: dict[int, int] = {}
    for lab in labels.tolist():
        sizes[lab] = sizes.get(lab, 0) + 1
    cluster_sizes = [s for lab, s in sizes.items() if lab != -1]
    largest = max(cluster_sizes) if cluster_sizes else 0
    small_thresh = max(mcs, int(np.ceil(small_cluster_frac * N)),
                       int(np.ceil(0.5 * largest)))

    cand = np.array(
        [(labels[i] == -1) or (sizes[int(labels[i])] < small_thresh)
         for i in range(N)], dtype=bool)
    return {"labels": labels, "candidate_mask": cand,
            "n_clusters": len(cluster_sizes), "normal_mask": ~cand}
