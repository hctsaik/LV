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


# ── class_centroid(依類別當參考,UX 精修)──────────────────────────────────────
def test_ac9_class_centroid():  # AC9:類別平均向量
    s = _sim()
    ca = np.asarray(s.class_centroid([[1, 0], [0, 1], [1, 0]], ["a", "b", "a"], "a"), dtype=float)
    cb = np.asarray(s.class_centroid([[1, 0], [0, 1], [1, 0]], ["a", "b", "a"], "b"), dtype=float)
    assert np.allclose(ca, [1, 0], atol=1e-6) and np.allclose(cb, [0, 1], atol=1e-6)


def test_ac10_class_centroid_missing():  # AC10:不存在的類別 → raise
    s = _sim()
    with pytest.raises(ValueError):
        s.class_centroid([[1, 0], [0, 1]], ["a", "b"], "c")


def test_ac11_centroid_similarity_same_class():  # AC11 衍生:同類物件對該類 centroid 最像
    s = _sim()
    X = [[1.0, 0.02], [0.98, 0.0], [0.0, 1.0], [0.02, 0.98]]   # a,a,b,b(兩群可分)
    labs = ["a", "a", "b", "b"]
    p = np.asarray(s.similarity_priority(X, s.class_centroid(X, labs, "a")), dtype=float)
    assert int(np.argmax(p)) in (0, 1), "屬於 a 的物件應對 a-centroid 最像"


# ── multi_ref_similarity(M13 Task1:多樣本 per-class max-cosine)────────────────
def test_ac_m1_multi_ref_basic():  # AC-M1
    s = _sim()
    labs, sims = s.multi_ref_similarity([[1, 0]], [[1, 0], [0, 1]], ["a", "b"])
    assert list(np.asarray(labs)) == ["a"]
    assert abs(float(np.asarray(sims)[0]) - 1.0) < 1e-6


def test_ac_m2_max_not_mean():  # AC-M2:同類兩樣本取 max(非平均)
    s = _sim()
    labs, sims = s.multi_ref_similarity([[0, 1]], [[1, 0], [0, 1]], ["a", "a"])
    assert list(np.asarray(labs)) == ["a"]
    assert abs(float(np.asarray(sims)[0]) - 1.0) < 1e-6   # max(0,1)=1,不是平均 0.5


def test_ac_m3_picks_closer_class():  # AC-M3:歸最像的類
    s = _sim()
    labs, _ = s.multi_ref_similarity([[0.9, 0.1]], [[1, 0], [0, 1]], ["a", "b"])
    assert list(np.asarray(labs)) == ["a"]


def test_ac_m4_empty_objects():  # AC-M4:空物件 → 空
    s = _sim()
    labs, sims = s.multi_ref_similarity(np.zeros((0, 2), dtype=np.float32), [[1, 0]], ["a"])
    assert np.asarray(labs).shape == (0,) and np.asarray(sims).shape == (0,)


def test_ac_m5_empty_refs():  # AC-M5:無樣本 → raise
    s = _sim()
    with pytest.raises(ValueError):
        s.multi_ref_similarity([[1, 0]], np.zeros((0, 2), dtype=np.float32), [])


def test_ac_m6_dim_mismatch():  # AC-M6:維度不符 → raise
    s = _sim()
    with pytest.raises(ValueError):
        s.multi_ref_similarity([[1, 0]], [[1, 0, 0]], ["a"])


def test_ac_m7_zero_ref():  # AC-M7:全零樣本列 → raise
    s = _sim()
    with pytest.raises(ValueError):
        s.multi_ref_similarity([[1, 0]], [[0, 0]], ["a"])


def test_ac_m8_matches_single_ref():  # AC-M8 衍生:單類單顆 == cosine_similarity_to_ref
    s = _sim()
    X = [[1.0, 0.02], [0.0, 1.0], [0.9, 0.1]]
    _, sims = s.multi_ref_similarity(X, [[1, 0]], ["a"])
    single = s.cosine_similarity_to_ref(X, [1, 0])
    assert np.allclose(np.asarray(sims, dtype=float), np.asarray(single, dtype=float), atol=1e-6)
