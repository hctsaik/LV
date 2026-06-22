"""驗收:bootstrap_cluster(設計 03_bootstrap_cluster.md)。先紅:scripts/bootstrap_cluster.py 未實作。"""
import numpy as np
import pytest

from bootstrap_cluster import cluster_objects

D = 16


def _blob(rng, center, n, sigma=0.05):
    return center[None, :] + sigma * rng.standard_normal((n, D)).astype(np.float32)


def _centers(rng, k, spread=10.0):
    c = rng.standard_normal((k, D)).astype(np.float32)
    return c / np.linalg.norm(c, axis=1, keepdims=True) * spread


def test_three_clean_blobs():  # AC1
    rng = np.random.default_rng(0)
    cs = _centers(rng, 3)
    emb = np.vstack([_blob(rng, cs[i], 50) for i in range(3)])
    r = cluster_objects(emb)
    assert r["n_clusters"] == 3
    assert not r["candidate_mask"].any()
    # 同團兩點同群(partition 一致)
    lab = r["labels"]
    assert lab[0] == lab[10] and lab[60] == lab[70]


def test_far_outliers_flagged():  # AC2
    rng = np.random.default_rng(1)
    cs = _centers(rng, 3)
    emb = np.vstack([_blob(rng, cs[i], 50) for i in range(3)])
    # 從隨機池挑「離所有群中心最遠」的 3 個方向當離群點 —— 16 維隨機單位向量平均
    # 距離僅 ~1.41,單純亂取可能恰好貼到某群;挑最孤立的才保證真的離群(→ noise)。
    csn = cs / np.linalg.norm(cs, axis=1, keepdims=True)
    pool = rng.standard_normal((200, D)).astype(np.float32)
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    min_cos_dist = (1.0 - pool @ csn.T).min(axis=1)
    outliers = pool[np.argsort(min_cos_dist)[::-1][:3]]  # 最孤立的 3 個
    emb = np.vstack([emb, outliers])
    r = cluster_objects(emb)
    assert (r["labels"][-3:] == -1).all()
    assert r["candidate_mask"][-3:].all()
    assert not r["candidate_mask"][:150].any()


def test_small_cluster_is_candidate():  # AC3(少數壞的聚成小群也算候選)
    rng = np.random.default_rng(2)
    cs = _centers(rng, 3)
    big = np.vstack([_blob(rng, cs[0], 60), _blob(rng, cs[1], 60)])
    small = _blob(rng, cs[2], 4)
    emb = np.vstack([big, small])
    r = cluster_objects(emb)
    assert r["candidate_mask"][-4:].all()          # 小群成員=候選
    assert not r["candidate_mask"][:120].any()      # 大群成員非候選


def test_shapes_and_types():  # AC4
    rng = np.random.default_rng(3)
    emb = _blob(rng, np.zeros(D, dtype=np.float32), 40)
    r = cluster_objects(emb)
    assert r["labels"].shape == (40,)
    assert r["candidate_mask"].dtype == bool
    assert np.array_equal(r["normal_mask"], ~r["candidate_mask"])


def test_degenerate_small_n():  # AC5
    for n in (0, 1):
        emb = np.zeros((n, D), dtype=np.float32)
        r = cluster_objects(emb)
        assert r["labels"].shape == (n,)
        assert not r["candidate_mask"].any()


def test_deterministic():  # AC6(推導:決定性 — partition+candidate 完全一致)
    rng = np.random.default_rng(4)
    cs = _centers(rng, 3)
    emb = np.vstack([_blob(rng, cs[i], 50) for i in range(3)])
    r1 = cluster_objects(emb)
    r2 = cluster_objects(emb)
    assert np.array_equal(r1["candidate_mask"], r2["candidate_mask"])
    assert np.array_equal(r1["labels"], r2["labels"])
