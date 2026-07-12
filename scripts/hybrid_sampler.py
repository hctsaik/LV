"""不確定 × 多樣性 × 類配額 混合選樣。

設計:3_Architect_Design/11_hybrid_sampler.md

修掉批次主動學習的經典陷阱:「最不確定的前 K 張」常是近重複,選 50 張等於選 8 張。
兩段式:**先取 top-(oversample×K) 不確定,再在其 embedding 上跑 FPS 取 K**;可選類配額
(弱類優先)。零新演算法依賴 —— FPS 復用 ``interaction.farthest_point_sampling``。

純函式、決定論、不 import streamlit。
"""
from __future__ import annotations

import numpy as np

from interaction import farthest_point_sampling


def hybrid_select(
    emb: np.ndarray,
    uncertainty: np.ndarray,
    k: int,
    *,
    oversample: int = 3,
    exclude: set[int] | None = None,
    labels: list[str] | None = None,
    class_quota: dict[str, int] | None = None,
    seed_idx: list[int] | None = None,
) -> list[int]:
    """回傳最多 ``k`` 個「值得送標」的原始索引,依選中順序。

    ``uncertainty`` 越大越該標(來源不限:標籤分歧 / 離群度 / 異常分數 / 1−conf)。
    只用它的**排序**,不用絕對值 —— 故對任何保序變換不變。

    步驟(全決定論):
      1. 候選池 = 剔除 ``exclude`` 後、最不確定的 ``oversample*k`` 個。
      2. 有 ``class_quota`` → 先從池內每個配額類取「該類最不確定的 quota 張」。
         該類樣本不足就有多少拿多少(缺口誠實留給第 3 步,不硬湊)。
      3. 其餘名額在池內跑 FPS 補滿(起點 = 已選者 ∪ ``seed_idx``∩池;皆空則取池內最不確定者)。
    """
    emb = np.asarray(emb, dtype=float)
    unc = np.asarray(uncertainty, dtype=float)
    if len(emb) != len(unc):
        raise ValueError(
            f"emb and uncertainty length mismatch: {len(emb)} != {len(unc)}")
    if k <= 0 or len(emb) == 0:
        return []

    exclude = set(exclude or ())
    # 1) 候選池:stable argsort 使「不確定度相同」時退回索引序(決定論)
    order = [int(i) for i in np.argsort(-unc, kind="stable")]
    cand = [i for i in order if i not in exclude][:max(oversample, 1) * k]
    if not cand:
        return []

    picked: list[int] = []

    # 2) 類配額:弱類保底。池內該類不足 → 有多少拿多少。
    if class_quota and labels is not None:
        for cls, quota in class_quota.items():
            in_cls = [i for i in cand if labels[i] == cls]  # cand 已依不確定度降冪
            for i in in_cls[:max(int(quota), 0)]:
                if i not in picked and len(picked) < k:
                    picked.append(i)

    # 3) 其餘名額用 FPS 補滿。FPS 吃「候選池的子矩陣」+ 局部 seed 索引,回局部索引且不含 seed。
    if len(picked) < k:
        pos = {g: loc for loc, g in enumerate(cand)}
        seeds_local = [pos[i] for i in picked]
        seeds_local += [pos[i] for i in (seed_idx or []) if i in pos and pos[i] not in seeds_local]
        if not seeds_local:
            # 無起點 → 池內最不確定者當第一張(cand[0]),它同時是 FPS 的覆蓋起點
            picked.append(cand[0])
            seeds_local = [0]
        need = k - len(picked)
        if need > 0:
            for loc in farthest_point_sampling(emb[cand], need, seed_indices=seeds_local):
                g = cand[loc]
                if g not in picked:
                    picked.append(g)

    return picked[:k]


def near_duplicate_pairs(emb: np.ndarray, idxs: list[int], thr: float = 0.05) -> int:
    """選中集合內 cosine 距離 < ``thr`` 的無序對數 —— 把「這批有多重複」攤在使用者眼前。

    O(k²),k 小(一批送標量)故可接受。
    """
    idxs = list(idxs)
    if len(idxs) < 2:
        return 0
    v = np.asarray(emb, dtype=float)[idxs]
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    dist = 1.0 - v @ v.T
    iu = np.triu_indices(len(idxs), k=1)  # 只數上三角 = 無序對
    return int((dist[iu] < thr).sum())
