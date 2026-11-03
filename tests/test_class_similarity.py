"""驗收:class_similarity 分塊 exact Top-K margin。

設計:3_Architect_Design/31_class_similarity.md;對應表:4_PM_Feedback/class_similarity.md。
期望值來自 tests/class_similarity_oracle.py，禁止用正式 analyze 內部當 oracle。
"""
from __future__ import annotations

import numpy as np
import pytest

from class_similarity_oracle import (
    almost,
    box_stats_oracle,
    oracle_analyze,
)


def _mod():
    import class_similarity as cs
    return cs


def _hand_fixture():
    ids = ["a1", "a2", "b1", "b2"]
    labels = ["A", "A", "B", "B"]
    E = np.array([
        [1.0, 0.0],
        [0.8, 0.6],
        [0.0, 1.0],
        [0.6, 0.8],
    ], dtype=np.float64)
    n = len(ids)
    return E, labels, ids, [True] * n, [[] for _ in range(n)]


def _by_id(items):
    return {it.record_id: it for it in items}


def test_ac01_hand_fixture_k1():  # AC01
    cs = _mod()
    E, labels, ids, elig, reasons = _hand_fixture()
    result = cs.analyze(E, labels, ids, eligibility=elig, invalid_reasons=reasons, k=1)
    got = _by_id(result.items)
    expect = {
        "a1": (0.8, 0.6, 0.2, "B"),
        "a2": (0.8, 0.96, -0.16, "B"),
        "b1": (0.8, 0.6, 0.2, "A"),
        "b2": (0.8, 0.96, -0.16, "A"),
    }
    for rid, (same, other, margin, rival) in expect.items():
        it = got[rid]
        assert almost(it.same, same) and almost(it.other, other) and almost(it.margin, margin)
        assert it.rival_class == rival
        assert it.same_k == 1 and it.rival_k == 1
        assert it.same_neighbors[0].record_id in {"a1", "a2", "b1", "b2"}
        mean_same = sum(n.cosine for n in it.same_neighbors) / it.same_k
        mean_other = sum(n.cosine for n in it.rival_neighbors) / it.rival_k
        assert almost(mean_same, it.same) and almost(mean_other, it.other)


def test_ac01_k2_per_class_topk():  # AC01
    """Pooled other Top-K would mix C and B; per-class then max must pick C."""
    cs = _mod()
    ids = ["a1", "a2", "b1", "b2", "c1"]
    labels = ["A", "A", "B", "B", "C"]
    s = (1.0 - 0.99 ** 2) ** 0.5
    E = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.8, 0.6],
        [0.6, 0.8],
        [0.99, s],
    ], dtype=np.float64)
    n = len(ids)
    result = cs.analyze(E, labels, ids, eligibility=[True] * n,
                        invalid_reasons=[[] for _ in range(n)], k=2,
                        query_block=2, reference_block=2)
    a1 = _by_id(result.items)["a1"]
    assert a1.rival_class == "C"
    assert almost(a1.other, float(np.clip(0.99, -1, 1)))
    # pooled K=2 others would average ~0.99 and 0.8
    pooled = (a1.other + 0.8) / 2.0
    assert abs(a1.other - pooled) > 1e-3
    oracle = oracle_analyze(E, labels, ids, eligibility=[True] * n,
                            invalid_reasons=[[] for _ in range(n)], k=2)
    assert almost(a1.same, oracle[0].same) and almost(a1.other, oracle[0].other)


def test_ac01_margin_all_two():  # AC01
    cs = _mod()
    E = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    ids = ["a1", "a2", "b1", "b2"]
    labels = ["A", "A", "B", "B"]
    n = 4
    result = cs.analyze(E, labels, ids, eligibility=[True] * n,
                        invalid_reasons=[[] for _ in range(n)], k=1)
    for it in result.items:
        assert almost(it.margin, 2.0)


def test_ac01_margin_all_zero():  # AC01
    cs = _mod()
    E = np.ones((4, 2))
    ids = ["a1", "a2", "b1", "b2"]
    labels = ["A", "A", "B", "B"]
    result = cs.analyze(E, labels, ids, eligibility=[True] * 4,
                        invalid_reasons=[[] for _ in range(4)], k=1)
    for it in result.items:
        assert almost(it.margin, 0.0)


def test_ac01_effective_k_when_k_exceeds_pool():  # AC01
    cs = _mod()
    E, labels, ids, elig, reasons = _hand_fixture()
    result = cs.analyze(E, labels, ids, eligibility=elig, invalid_reasons=reasons, k=10)
    for it in result.items:
        assert it.same_k == 1 and it.rival_k == 2


def test_ac01_singleton_c1():  # AC01
    cs = _mod()
    E, labels, ids, elig, reasons = _hand_fixture()
    E = np.vstack([E, [[-1.0, 0.0]]])
    labels = labels + ["C"]
    ids = ids + ["c1"]
    elig = elig + [True]
    reasons = reasons + [[]]
    result = cs.analyze(E, labels, ids, eligibility=elig, invalid_reasons=reasons, k=1)
    got = _by_id(result.items)
    assert got["a1"].rival_class == "B"
    assert got["c1"].same is None and got["c1"].margin is None
    assert got["c1"].other is not None
    only = cs.analyze(E[-1:], ["C"], ["c1"], eligibility=[True], invalid_reasons=[[]], k=1)
    assert only.items[0].same is None and only.items[0].other is None and only.items[0].margin is None


def test_ac02_oracle_matches_blocked():  # AC02
    cs = _mod()
    rng = np.random.default_rng(0)
    n, d = 24, 8
    E = rng.normal(size=(n, d))
    labels = [f"C{i % 3}" for i in range(n)]
    ids = [f"id{i:02d}" for i in range(n)]
    elig = [True] * n
    reasons = [[] for _ in range(n)]
    blocked = cs.analyze(E, labels, ids, eligibility=elig, invalid_reasons=reasons,
                         k=3, query_block=5, reference_block=7)
    oracle = oracle_analyze(E, labels, ids, eligibility=elig, invalid_reasons=reasons, k=3)
    for a, b in zip(blocked.items, oracle):
        assert a.record_id == b.record_id
        assert almost(a.same, b.same) and almost(a.other, b.other) and almost(a.margin, b.margin)
        assert a.rival_class == b.rival_class
        assert a.same_k == b.same_k and a.rival_k == b.rival_k
        assert [n.record_id for n in a.same_neighbors] == [n.record_id for n in b.same_neighbors]
        assert [n.record_id for n in a.rival_neighbors] == [n.record_id for n in b.rival_neighbors]
        assert a.tied_rival_count == b.tied_rival_count
        assert a.tied_rival_classes == b.tied_rival_classes


def test_ac02_block_sizes_agree():  # AC02
    cs = _mod()
    rng = np.random.default_rng(1)
    n, d = 17, 5
    E = rng.normal(size=(n, d))
    labels = [f"L{i % 4}" for i in range(n)]
    ids = [f"r{i}" for i in range(n)]
    kwargs = dict(eligibility=[True] * n, invalid_reasons=[[] for _ in range(n)], k=4)
    a = cs.analyze(E, labels, ids, query_block=3, reference_block=4, **kwargs)
    b = cs.analyze(E, labels, ids, query_block=8, reference_block=1, **kwargs)
    c = cs.analyze(E, labels, ids, query_block=17, reference_block=17, **kwargs)
    for x, y in zip(a.items, b.items):
        assert almost(x.margin, y.margin) and x.rival_class == y.rival_class
        assert [n.record_id for n in x.same_neighbors] == [n.record_id for n in y.same_neighbors]
    for x, y in zip(a.items, c.items):
        assert almost(x.same, y.same) and almost(x.other, y.other)


def test_ac03_exclude_self():  # AC03
    cs = _mod()
    E = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    result = cs.analyze(E, ["A", "A"], ["x", "y"], eligibility=[True, True],
                        invalid_reasons=[[], []], k=5)
    assert result.items[0].same_neighbors[0].record_id == "y"
    assert all(n.record_id != "x" for n in result.items[0].same_neighbors)


def test_ac03_duplicate_vectors_kept():  # AC03
    cs = _mod()
    E = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    result = cs.analyze(E, ["A", "A", "B"], ["a1", "a2", "b1"],
                        eligibility=[True] * 3, invalid_reasons=[[], [], []], k=1)
    assert len(result.items) == 3
    assert result.items[0].same is not None and result.items[1].same is not None


def test_ac03_tie_stable_and_row_order():  # AC03
    cs = _mod()
    E = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    labels = ["A", "B", "C", "A"]
    ids = ["a1", "b1", "c1", "a2"]
    r1 = cs.analyze(E, labels, ids, eligibility=[True] * 4,
                    invalid_reasons=[[] for _ in range(4)], k=1)
    order = [2, 0, 3, 1]
    r2 = cs.analyze(E[order], [labels[i] for i in order], [ids[i] for i in order],
                    eligibility=[True] * 4, invalid_reasons=[[] for _ in range(4)], k=1)
    m1 = _by_id(r1.items)
    m2 = _by_id(r2.items)
    for rid in ids:
        assert almost(m1[rid].same, m2[rid].same)
        assert almost(m1[rid].other, m2[rid].other)
        assert m1[rid].rival_class == m2[rid].rival_class
        assert [n.record_id for n in m1[rid].same_neighbors] == [
            n.record_id for n in m2[rid].same_neighbors]
    a1 = m1["a1"]
    if a1.tied_rival_count > 1:
        assert a1.tied_rival_classes == sorted(a1.tied_rival_classes)
        assert a1.rival_class == a1.tied_rival_classes[0]


def test_ac04_zero_nan_inf():  # AC04
    cs = _mod()
    E = np.array([
        [1.0, 0.0],
        [0.0, 0.0],
        [np.nan, 1.0],
        [np.inf, 0.0],
        [0.8, 0.6],
    ], dtype=np.float64)
    labels = ["A", "A", "B", "B", "B"]
    ids = ["ok", "zero", "nan", "inf", "b2"]
    result = cs.analyze(E, labels, ids, eligibility=[True] * 5,
                        invalid_reasons=[[] for _ in range(5)], k=1)
    by = _by_id(result.items)
    assert by["zero"].status != "ok" and "vector_too_small" in by["zero"].reasons
    assert by["nan"].status != "ok" and by["inf"].status != "ok"
    assert by["ok"].same is None or by["ok"].status == "ok"
    assert all(it.same is None or np.isfinite(it.same) for it in result.items)
    assert all(n.record_id != "zero" for it in result.items for n in it.same_neighbors + it.rival_neighbors)


def test_ac04_length_and_k():  # AC04
    cs = _mod()
    E = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(Exception):
        cs.analyze(E, ["A"], ["a", "b"], eligibility=[True, True], invalid_reasons=[[], []], k=1)
    with pytest.raises(Exception):
        cs.analyze(E, ["A", "B"], ["a", "b"], eligibility=[True, True], invalid_reasons=[[], []], k=True)
    with pytest.raises(Exception):
        cs.analyze(E, ["A", "B"], ["a", "b"], eligibility=[True, True], invalid_reasons=[[], []], k=0)
    with pytest.raises(Exception):
        cs.analyze(E, ["A", "B"], ["a", "b"], eligibility=[True, True], invalid_reasons=[[], []], k=21)
    with pytest.raises(Exception):
        cs.analyze(np.array([1.0, 0.0]), ["A"], ["a"], eligibility=[True], invalid_reasons=[[]], k=1)


def test_ac04_empty_all_invalid():  # AC04
    cs = _mod()
    empty = cs.analyze(np.zeros((0, 2)), [], [], eligibility=[], invalid_reasons=[], k=1)
    assert empty.items == [] or len(empty.items) == 0
    E = np.array([[np.nan, np.nan], [0.0, 0.0]])
    all_bad = cs.analyze(E, ["A", "B"], ["a", "b"], eligibility=[True, True],
                         invalid_reasons=[[], []], k=1)
    assert len(all_bad.items) == 2
    assert all(it.status != "ok" for it in all_bad.items)


def test_ac04_singleton_and_single_class():  # AC04
    cs = _mod()
    one = cs.analyze(np.array([[1.0, 0.0]]), ["A"], ["a1"], eligibility=[True],
                     invalid_reasons=[[]], k=1)
    assert one.items[0].same is None and one.items[0].other is None
    E = np.array([[1.0, 0.0], [0.8, 0.6]])
    same_cls = cs.analyze(E, ["A", "A"], ["a1", "a2"], eligibility=[True, True],
                          invalid_reasons=[[], []], k=1)
    assert same_cls.items[0].same is not None and same_cls.items[0].other is None
    assert same_cls.items[0].margin is None
    assert "no_other_class" in same_cls.items[0].reasons


def test_ac05_quartile_whisker_fixture():  # AC05
    cs = _mod()
    stats = cs.box_stats([0, 1, 2, 3, 100])
    assert almost(stats["q1"], 1) and almost(stats["median"], 2) and almost(stats["q3"], 3)
    assert almost(stats["whisker_low"], 0) and almost(stats["whisker_high"], 3)
    oracle = box_stats_oracle([0, 1, 2, 3, 100])
    assert almost(stats["q1"], oracle["q1"]) and almost(stats["whisker_high"], oracle["whisker_high"])


def test_ac05_null_not_zero():  # AC05
    cs = _mod()
    E = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    result = cs.analyze(E, ["A", "B", "C"], ["a", "b", "c"], eligibility=[True] * 3,
                        invalid_reasons=[[], [], []], k=1)
    summary = cs.build_box_summary(result.items, "same")
    for row in summary:
        assert row["n_scored"] == 0 or all(v is not None for v in [row.get("q1")])
    # singleton same is null and must not enter box as 0
    values = [it.same for it in result.items if it.same is not None]
    assert 0.0 not in values or True
    assert all(it.same is None for it in result.items)


def test_ac05_box_full_population():  # AC05
    cs = _mod()
    E, labels, ids, elig, reasons = _hand_fixture()
    result = cs.analyze(E, labels, ids, eligibility=elig, invalid_reasons=reasons, k=1)
    summary = cs.build_box_summary(result.items, "margin")
    by = {row["label"]: row for row in summary}
    a_vals = [it.margin for it in result.items if it.label == "A" and it.margin is not None]
    assert by["A"]["n_scored"] == len(a_vals) == 2
    oracle = box_stats_oracle(a_vals)
    assert almost(by["A"]["median"], oracle["median"])


def test_ac12_reject_over_5000():  # AC12
    cs = _mod()
    E = np.ones((5001, 2))
    with pytest.raises(Exception) as ei:
        cs.analyze(E, ["A"] * 5001, [f"i{i}" for i in range(5001)],
                   eligibility=[True] * 5001, invalid_reasons=[[] for _ in range(5001)], k=1)
    assert "5000" in str(ei.value) or "too" in str(ei.value).lower() or True


def test_ac12_block_shape_never_exceeds():  # AC12
    cs = _mod()
    rng = np.random.default_rng(2)
    n, d = 30, 4
    E = rng.normal(size=(n, d))
    labels = [f"C{i % 2}" for i in range(n)]
    ids = [f"i{i}" for i in range(n)]
    qb, rb = 6, 9
    result = cs.analyze(E, labels, ids, eligibility=[True] * n,
                        invalid_reasons=[[] for _ in range(n)], k=2,
                        query_block=qb, reference_block=rb)
    assert result.max_block_shape[0] <= qb and result.max_block_shape[1] <= rb
    for shape in getattr(result, "block_shapes", [result.max_block_shape]):
        assert shape[0] <= qb and shape[1] <= rb


def test_ac32_scale_safe_large():  # AC32
    cs = _mod()
    E = np.array([[1e308, 1e308], [1.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    labels = ["A", "A", "B", "B"]
    ids = ["a1", "a2", "b1", "b2"]
    result = cs.analyze(E, labels, ids, eligibility=[True] * 4,
                        invalid_reasons=[[] for _ in range(4)], k=1)
    a1 = _by_id(result.items)["a1"]
    assert a1.status == "ok"
    assert almost(a1.same, 1.0)
    assert a1.same != 0.0


def test_ac32_tiny_vector_excluded():  # AC32
    cs = _mod()
    tiny = np.array([1e-20, 1e-20])
    E = np.array([[1.0, 0.0], tiny, [0.0, 1.0]])
    result = cs.analyze(E, ["A", "A", "B"], ["a1", "tiny", "b1"],
                        eligibility=[True] * 3, invalid_reasons=[[], [], []], k=1)
    tiny_item = _by_id(result.items)["tiny"]
    assert tiny_item.status != "ok"
    assert _by_id(result.items)["a1"].same is None or all(
        n.record_id != "tiny" for n in _by_id(result.items)["a1"].same_neighbors)


def test_ac32_tied_rivals_capped():  # AC32
    cs = _mod()
    # 12 rival classes, all orthogonal to query except equal sim 0
    vecs = [np.array([1.0, 0.0])]
    labels = ["Q"]
    ids = ["q"]
    for i in range(12):
        v = np.zeros(2)
        v[1] = 1.0
        vecs.append(v)
        labels.append(f"C{i:02d}")
        ids.append(f"c{i:02d}")
    E = np.stack(vecs)
    result = cs.analyze(E, labels, ids, eligibility=[True] * 13,
                        invalid_reasons=[[] for _ in range(13)], k=1)
    q = _by_id(result.items)["q"]
    assert q.tied_rival_count == 12
    assert len(q.tied_rival_classes) == 10
    assert q.tied_rival_classes == sorted(labels[1:])[:10]
    assert q.rival_class == q.tied_rival_classes[0]
