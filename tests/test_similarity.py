"""驗收:12 similarity(對參考物件的 cosine 相似優先分數)。

設計:3_Architect_Design/12_similarity.md;對應表:4_PM_Feedback/similarity.md。
每測註對應 # ACn。延遲匯入 similarity(scripts/similarity.py 由 /pg 建立)→ 先紅為 failure 非 collection error。
"""
import numpy as np
import pytest


def _sim():
    import similarity
    return similarity


def test_ac1_cosine_values():  # AC1:cosine 值(同向=1、正交=0)
    s = _sim()
    got = np.asarray(s.cosine_similarity_to_ref([[1, 0], [0, 1], [3, 0]], [1, 0]))
    for g, e in zip(got, [1.0, 0.0, 1.0]):
        assert abs(float(g) - e) < 1e-6


def test_ac2_similarity_order_range():  # AC2:相似降冪排序 + 範圍 [0,1]
    s = _sim()
    p = np.asarray(s.similarity_priority([[1, 0], [0, 1], [0.9, 0.1]], [1, 0]), dtype=float)
    assert list(np.argsort(-p)) == [0, 2, 1]
    assert abs(float(p.min()) - 0.0) < 1e-6 and abs(float(p.max()) - 1.0) < 1e-6
    assert all(0.0 <= float(x) <= 1.0 for x in p)


def test_ac3_ref_zero_raises():  # AC3:參考全零 → raise
    s = _sim()
    with pytest.raises(ValueError):
        s.cosine_similarity_to_ref([[1, 0]], [0, 0])
    with pytest.raises(ValueError):
        s.similarity_priority([[1, 0]], [0, 0])


def test_ac4_empty():  # AC4:空輸入 → 形狀 (0,)
    s = _sim()
    out = np.asarray(s.similarity_priority(np.zeros((0, 2), dtype=np.float32), [1, 0]))
    assert out.shape == (0,)


def test_ac5_dim_mismatch():  # AC5:維度不符 → raise
    s = _sim()
    with pytest.raises(ValueError):
        s.cosine_similarity_to_ref([[1, 0]], [1, 0, 0])


def test_ac6_single():  # AC6:單一物件 → (1,) 值 0.0(minmax 退化)
    s = _sim()
    out = np.asarray(s.similarity_priority([[1, 0]], [1, 0]), dtype=float)
    assert out.shape == (1,) and abs(float(out[0]) - 0.0) < 1e-6


def test_ac7_positive_scale_invariance():  # AC7 衍生:cosine 對正尺度不變
    s = _sim()
    X = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.3, 0.7]]
    a = np.asarray(s.similarity_priority(X, [1, 0]), dtype=float)
    b = np.asarray(s.similarity_priority((2.0 * np.asarray(X)).tolist(), [1, 0]), dtype=float)
    assert np.allclose(a, b, atol=1e-6)


def test_ac8_ref_aligned_is_max():  # AC8 衍生:與參考同向者優先分數最大
    s = _sim()
    X = [[0.0, 1.0], [2.0, 0.0], [0.5, 0.5]]   # 第 1 列 = 2×參考方向
    p = np.asarray(s.similarity_priority(X, [1, 0]), dtype=float)
    assert int(np.argmax(p)) == 1 and abs(float(p[1]) - 1.0) < 1e-6
