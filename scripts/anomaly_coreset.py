"""anomaly_coreset:greedy k-center coreset(farthest-point sampling)。

取代純隨機子抽樣建 memory bank —— greedy 讓小 budget 也能覆蓋所有正常模式,
代表性優於純隨機(PatchCore 風格,對齊「以更準為原則」)。為撐「非常大量 good」,
N 超過 oversample_cap 時先隨機抽到 cap 再 greedy(近似但仍遠優於純隨機)。

對已 L2 正規化的向量(memory bank 建前都會正規化),L2 平方距離與 cosine 距離
單調等價,故用 L2 平方(BLAS 友善、無需逐步正規化)。
"""
from __future__ import annotations

import numpy as np


def _sqdist(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    """每列到中心 c 的 L2 平方距離。"""
    diff = X - c
    return np.einsum("ij,ij->i", diff, diff)


def greedy_coreset(vectors, budget: int, *, seed: int = 42,
                   oversample_cap: "int | None" = 50_000) -> np.ndarray:
    """回傳選中的原始列索引(已排序,長度 = min(budget, 有效列數))。

    k-center greedy:起點隨機,之後每步選「離已選集最遠」的點,最小化覆蓋半徑。
    時間 O(k · M · D)(k=budget、M=抽樣後列數);大量場景靠 oversample_cap 先降 M。
    """
    X = np.asarray(vectors, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("empty coreset input")
    N = X.shape[0]
    rng = np.random.default_rng(seed)

    # 撐大量:先隨機抽到 cap(O(N)),再對子集做 greedy。pool 保留原始索引。
    pool = np.arange(N)
    if oversample_cap is not None and N > oversample_cap:
        pool = np.sort(rng.choice(N, size=oversample_cap, replace=False))
        X = X[pool]

    M = X.shape[0]
    k = int(min(max(budget, 1), M))
    if k >= M:
        return pool.copy()

    selected = np.empty(k, dtype=np.int64)
    first = int(rng.integers(M))
    selected[0] = first
    min_d = _sqdist(X, X[first])              # 各點到已選集的最近距離(平方)
    for i in range(1, k):
        nxt = int(np.argmax(min_d))           # 離已選集最遠者
        selected[i] = nxt
        np.minimum(min_d, _sqdist(X, X[nxt]), out=min_d)
    return np.sort(pool[selected])
