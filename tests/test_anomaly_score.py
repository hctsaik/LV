"""驗收:anomaly_score(設計 03_.../02_anomaly_score.md)。先紅:scripts/anomaly_score.py 尚未實作。
A=e0、B=e1(正規化),cos(A,B)=0 → patch 距離=1.0。"""
import numpy as np
import pytest

from anomaly_score import build_memory_bank, score_object

D = 384


def _A():
    v = np.zeros(D, dtype=np.float32); v[0] = 1.0; return v


def _B():
    v = np.zeros(D, dtype=np.float32); v[1] = 1.0; return v


def test_identical_query_scores_zero():  # AC1
    bank = build_memory_bank(np.repeat(_A()[None, :], 1000, axis=0))
    q = np.repeat(_A()[None, :], 256, axis=0)
    obj, pmap = score_object(q, (16, 16), bank, q=0.01)
    assert pmap.shape == (16, 16)
    assert np.max(pmap) < 1e-6
    assert obj < 1e-6


def test_single_orthogonal_patch_scores_one():  # AC2 + AC3
    bank = build_memory_bank(np.repeat(_A()[None, :], 1000, axis=0))
    q = np.repeat(_A()[None, :], 100, axis=0)
    q[37] = _B()  # one anomalous patch at flat index 37
    obj, pmap = score_object(q, (10, 10), bank, q=0.01)  # k=ceil(0.01*100)=1
    assert obj == pytest.approx(1.0, abs=1e-6)
    assert pmap.shape == (10, 10)
    assert int(np.argmax(pmap)) == 37            # AC3 位置正確
    assert pmap.flat[37] == pytest.approx(1.0, abs=1e-6)


def test_coreset_budget_respected():  # AC4
    rng = np.random.default_rng(0)
    M = rng.standard_normal((1000, D)).astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    assert build_memory_bank(M, budget=50).size == 50
    assert build_memory_bank(M, budget=5000).size == 1000


def test_object_score_monotone_in_defect_count():  # AC5(推導:單調/飽和不變量)
    bank = build_memory_bank(np.repeat(_A()[None, :], 1000, axis=0))
    scores = []
    for c in (1, 5, 10):
        q = np.repeat(_A()[None, :], 100, axis=0)
        q[:c] = _B()
        # q 設成涵蓋全部壞 patch(k>=c):用 q=c/100
        obj, _ = score_object(q, (10, 10), bank, q=c / 100.0)
        scores.append(obj)
    assert scores[0] <= scores[1] <= scores[2]
    assert scores[2] == pytest.approx(1.0, abs=1e-6)  # 全是 B 的 top-c 平均=1


def test_defect_strictly_separable_from_normal():  # AC6(可分性,餵 G1)
    rng = np.random.default_rng(1)
    A = _A()
    normal_patches = A[None, :] + 0.01 * rng.standard_normal((4000, D)).astype(np.float32)
    normal_patches /= np.linalg.norm(normal_patches, axis=1, keepdims=True)
    bank = build_memory_bank(normal_patches)
    normal_scores, defect_scores = [], []
    for _ in range(30):
        qn = A[None, :] + 0.01 * rng.standard_normal((64, D)).astype(np.float32)
        qn /= np.linalg.norm(qn, axis=1, keepdims=True)
        normal_scores.append(score_object(qn, (8, 8), bank, q=0.05)[0])
        qd = qn.copy(); qd[0] = _B()
        defect_scores.append(score_object(qd, (8, 8), bank, q=0.05)[0])
    assert min(defect_scores) > max(normal_scores)


def test_empty_bank_and_empty_query():  # AC7(邊界)
    with pytest.raises(ValueError):
        build_memory_bank(np.zeros((0, D), dtype=np.float32))
    bank = build_memory_bank(np.repeat(_A()[None, :], 10, axis=0))
    obj, pmap = score_object(np.zeros((0, D), dtype=np.float32), (3, 4), bank)
    assert obj == 0.0 and pmap.shape == (3, 4)
