"""驗收:anomaly_coreset —— greedy k-center coreset(取代純隨機抽樣,求更準代表性)。
先紅:scripts/anomaly_coreset.py 尚未實作。

核心價值(對齊 PatchCore):小 budget 也能覆蓋所有正常模式,覆蓋半徑優於純隨機。
"""
import numpy as np
import pytest

from anomaly_coreset import greedy_coreset


def _clusters(n_clusters=6, per=200, D=16, sep=20.0, seed=0):
    """n_clusters 個沿不同軸分離的簇(簇間距 >> 簇內噪聲)。"""
    rng = np.random.default_rng(seed)
    X, labels = [], []
    for c in range(n_clusters):
        center = np.zeros(D, dtype=np.float32)
        center[c % D] = sep * (c + 1)
        X.append(center + rng.normal(0, 0.5, size=(per, D)).astype(np.float32))
        labels += [c] * per
    return np.vstack(X), np.array(labels)


def _maxmin(X, sel):
    """覆蓋半徑:每點到選中集的最近距離,取最大值(k-center 目標)。"""
    d = np.linalg.norm(X[:, None, :] - X[sel][None, :, :], axis=2)
    return float(d.min(axis=1).max())


def test_budget_respected():  # AC1:回傳 budget 個唯一有效索引
    X, _ = _clusters()
    idx = greedy_coreset(X, 50)
    assert len(idx) == 50
    assert len(set(idx.tolist())) == 50
    assert idx.min() >= 0 and idx.max() < len(X)


def test_budget_ge_n_returns_all():  # AC2:budget≥N → 全選
    X, _ = _clusters(n_clusters=2, per=5)  # N=10
    idx = greedy_coreset(X, 999)
    assert sorted(idx.tolist()) == list(range(len(X)))


def test_deterministic():  # AC3:同 seed 同輸入 → 同結果
    X, _ = _clusters()
    assert np.array_equal(greedy_coreset(X, 30, seed=7),
                          greedy_coreset(X, 30, seed=7))


def test_covers_all_modes():  # AC4:greedy 的核心 —— budget=簇數 即涵蓋所有簇
    X, lab = _clusters(n_clusters=6, per=200)
    idx = greedy_coreset(X, 6, seed=1)
    covered = set(lab[idx].tolist())
    assert covered == set(range(6)), f"greedy 應涵蓋全部 6 簇,實際 {covered}"


def test_coverage_radius_le_random():  # AC5:k-center 性質 —— 覆蓋半徑 ≤ 純隨機
    X, _ = _clusters(n_clusters=6, per=200)
    g = _maxmin(X, greedy_coreset(X, 8, seed=2))
    r = _maxmin(X, np.random.default_rng(2).choice(len(X), 8, replace=False))
    assert g <= r, f"greedy 覆蓋半徑 {g:.3f} 應 ≤ 隨機 {r:.3f}"


def test_oversample_cap_handles_large():  # AC6:N 超過 cap → 先抽樣再 greedy,不爆且索引有效
    X, _ = _clusters(n_clusters=4, per=500)  # N=2000
    idx = greedy_coreset(X, 20, oversample_cap=300)
    assert len(idx) == 20
    assert idx.min() >= 0 and idx.max() < len(X)
    assert len(set(idx.tolist())) == 20


def test_empty_raises():  # AC7:空輸入 → ValueError
    with pytest.raises(ValueError):
        greedy_coreset(np.zeros((0, 8), dtype=np.float32), 5)
