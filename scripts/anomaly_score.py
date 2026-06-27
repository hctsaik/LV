"""anomaly_score:正常 patch memory bank(coreset)+ patch 最近鄰異常分數。

設計:3_Architect_Design/02_anomaly_score.md。AnomalyDINO 的評分核心:
每個查詢 patch 到正常 memory bank 的最近鄰 cosine 距離 → 聚合成物件異常分數。
"""
from __future__ import annotations

import numpy as np

# 小 bank → 精確 matmul(BLAS,零召回問題);大 bank → hnswlib 近似(快、撐十萬級)。
# 近似搜尋在「大量相同向量」的退化情況會漏找稀有點,真實 DINOv2 特徵不會退化;
# 用精確路徑守住正確性,近似路徑守住大規模互動延遲(G3)。
_EXACT_MAX = 8192


def _l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class MemoryBank:
    """正常 patch 的不透明 handle:持有正規化向量 + cosine 最近鄰搜尋。"""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.size = int(self.vectors.shape[0])
        self._exact = self.size <= _EXACT_MAX
        self._index = None
        if not self._exact:
            from interaction import build_nn_index
            self._index = build_nn_index(self.vectors)

    def nn_dist(self, queries: np.ndarray) -> np.ndarray:
        """每個 query(已 L2 正規化)到 bank 的最近鄰 cosine 距離(clamp≥0)。"""
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if self._exact:
            out = np.empty(len(q), dtype=np.float32)
            for s in range(0, len(q), 8192):  # 分塊矩陣乘:cos 相似度 → 距離
                sims = q[s:s + 8192] @ self.vectors.T
                out[s:s + 8192] = 1.0 - sims.max(axis=1)
            return np.maximum(out, 0.0)
        self._index.set_ef(128)
        _, dists = self._index.knn_query(q, k=1)
        return np.maximum(dists[:, 0], 0.0)


def build_memory_bank(normal_patch_feats, *, budget: int = 100_000,
                      seed: int = 42, method: str = "greedy") -> MemoryBank:
    """正常 patch → memory bank。超過 budget 時抽 coreset:
    method='greedy'(預設,更準的 k-center 代表性子集,以更準為原則)或 'random'(純隨機)。
    greedy 對「非常大量 good」靠 oversample_cap 先降規模(見 anomaly_coreset);CPU 上極大規模
    會 graceful 退化成隨機子抽樣。"""
    feats = np.asarray(normal_patch_feats, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[0] == 0:
        raise ValueError("empty memory bank")
    feats = _l2n(feats)
    if feats.shape[0] > budget:
        if method == "greedy":
            from anomaly_coreset import greedy_coreset
            idx = greedy_coreset(feats, budget, seed=seed)
        else:
            idx = np.random.default_rng(seed).choice(feats.shape[0], size=budget,
                                                      replace=False)
        feats = feats[idx]
    return MemoryBank(feats)


def score_object(query_feats, grid, bank: MemoryBank, *, q: float = 0.01):
    """回 (object_score, patch_scores[gh,gw])。

    patch_scores[r,c] = 該 patch 的最近鄰 cosine 距離 ∈[0,2]。
    object_score = 最高 ceil(q*P) 個 patch 距離之平均(至少 1 個)。
    """
    gh, gw = int(grid[0]), int(grid[1])
    query = np.asarray(query_feats, dtype=np.float32)
    if query.ndim != 2 or query.shape[0] == 0:
        return 0.0, np.zeros((gh, gw), dtype=np.float32)
    dists = bank.nn_dist(_l2n(query))
    pmap = dists.reshape(gh, gw).astype(np.float32)
    P = int(dists.shape[0])
    qq = min(max(float(q), 1e-9), 1.0)
    k = min(P, max(1, int(np.ceil(qq * P))))
    topk = np.sort(dists)[::-1][:k]
    return float(topk.mean()), pmap
