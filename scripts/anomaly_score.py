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
                      seed: int = 42, method: str = "random") -> MemoryBank:
    """正常 patch → memory bank。超過 budget 時抽 coreset。

    method='random'(**預設**):純隨機保留大 budget → 覆蓋廣。對「非常大量 good」,最近鄰評分
      靠 bank 覆蓋密度,點多(budget≈10萬)實際比小 greedy 子集更穩 → 對大量場景才是真的更準。
    method='greedy':k-center 代表性子集。CPU 上 farthest-point 對大 M 極慢(實測 40k→12k 要 ~10 分),
      故只在受控候選池 pool=5000、bank≈2500 內跑;適合中小規模或想要分散代表,極大量時點數太少反不如
      大 budget 隨機。GPU/近似加速為 backlog。"""
    feats = np.asarray(normal_patch_feats, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[0] == 0:
        raise ValueError("empty memory bank")
    feats = _l2n(feats)
    if feats.shape[0] > budget:
        if method == "greedy":
            from anomaly_coreset import greedy_coreset
            # FPS k-center 純 CPU 對大 M 慢 → 用受控候選池 pool 讓建 bank 在可接受時間完成,
            # 且 eff_budget < pool 確保 FPS 真的執行(否則 k>=M 早退 = 退化純隨機,greedy 形同 no-op)。
            # 極大量 good:在代表性候選池內 greedy 選最分散子集,仍優於純隨機(CPU 規模上限,GPU 加速為 backlog)。
            pool = 5_000
            eff_budget = min(int(budget), pool // 2)
            idx = greedy_coreset(feats, eff_budget, seed=seed, oversample_cap=pool)
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
