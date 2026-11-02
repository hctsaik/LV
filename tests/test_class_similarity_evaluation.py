"""驗收:人工評估、PR、digest、heldout。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _thr():
    import class_similarity_threshold as m
    return m


def _exp():
    import class_similarity_export as m
    return m


def _items(rows):
    """rows: (id, margin, label)"""
    return [SimpleNamespace(record_id=a, margin=b, label=c, status="ok", reasons=[])
            for a, b, c in rows]


def _eval_fixture():
    items = _items([
        ("r1", -0.5, "A"), ("r2", -0.1, "A"),
        ("r3", 0.1, "B"), ("r4", 0.4, "B"),
    ])
    evaluation = {
        "schema_version": 1,
        "evaluation_id": "e", "evaluation_version": 1, "purpose": "tuning",
        "snapshot_key": "snap",
        "frozen_ids": ["r1", "r2", "r3", "r4"],
        "reviews": [
            {"record_id": "r1", "original_label": "A", "verdict": "incorrect",
             "review_source": "human", "review_version": 1},
            {"record_id": "r2", "original_label": "A", "verdict": "correct",
             "review_source": "human", "review_version": 1},
            {"record_id": "r3", "original_label": "B", "verdict": "incorrect",
             "review_source": "human", "review_version": 1},
            {"record_id": "r4", "original_label": "B", "verdict": "correct",
             "review_source": "human", "review_version": 1},
        ],
    }
    return items, evaluation


def test_ac20_pr_hand_fixture():  # AC20
    thr = _thr()
    items, ev = _eval_fixture()
    c = thr.evaluate_threshold(items, thr.candidate_mask(items, {"mode": "threshold", "threshold": 0.1}), ev)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    assert c.precision == 0.5 and c.recall == 0.5
    c2 = thr.evaluate_threshold(items, thr.candidate_mask(items, {"mode": "threshold", "threshold": 0.2}), ev)
    assert (c2.tp, c2.fp, c2.fn, c2.tn) == (2, 1, 0, 1)
    assert abs(c2.precision - 2 / 3) < 1e-12 and c2.recall == 1
    none = thr.evaluate_threshold(items, thr.candidate_mask(items, {"mode": "none"}), ev)
    assert (none.tp, none.fp, none.fn, none.tn) == (0, 0, 2, 2)
    assert none.precision is None and none.recall == 0
    all_e = thr.evaluate_threshold(items, thr.candidate_mask(items, {"mode": "all_eligible"}), ev)
    assert (all_e.tp, all_e.fp, all_e.fn, all_e.tn) == (2, 2, 0, 0)
    assert all_e.precision == 0.5 and all_e.recall == 1


def test_ac20_zero_denominator():  # AC20
    thr = _thr()
    items = _items([("a", 0.5, "A"), ("b", 0.6, "A")])
    ev = {"frozen_ids": ["a", "b"],
          "reviews": [{"record_id": "a", "verdict": "correct", "original_label": "A"},
                      {"record_id": "b", "verdict": "correct", "original_label": "A"}]}
    c = thr.evaluate_threshold(items, thr.candidate_mask(items, {"mode": "none"}), ev)
    assert c.recall is None and c.precision is None
    assert c.status == "ready"
    pr = thr.precision_recall_curve(items, ev)
    # all-correct: recall undefined for all points with no incorrect → curve empty or no P,R
    assert all(p.get("recall") is not None for p in pr) or pr == [] or True
    # E all correct → recall null at none; skip drawing those points
    assert all(p["recall"] is not None and p["precision"] is not None for p in pr)


def test_ac20_no_truth_unavailable():  # AC20
    thr = _thr()
    items = _items([("a", -0.1, "A")])
    c = thr.evaluate_threshold(items, [True], None)
    assert c.status == "not_configured" and c.precision is None


def test_ac19_manifest_validation():  # AC19
    thr = _thr()
    items, ev = _eval_fixture()
    snap = {"snapshot_key": "snap", "original_labels": {"r1": "A", "r2": "A", "r3": "B", "r4": "B"}}
    ok = thr.validate_evaluation_manifest(ev, snap, items)
    assert ok["truth_digest"]
    bad = dict(ev)
    bad["reviews"] = list(ev["reviews"]) + [{"record_id": "nope", "verdict": "correct"}]
    with pytest.raises(thr.EvaluationError):
        thr.validate_evaluation_manifest(bad, snap, items)
    dup = dict(ev, frozen_ids=["r1", "r1"])
    with pytest.raises(thr.EvaluationError):
        thr.validate_evaluation_manifest(dup, snap, items)
    unrev = dict(ev)
    unrev["reviews"] = ev["reviews"][:3]  # r4 filled unreviewed
    filled = thr.validate_evaluation_manifest(unrev, snap, items)
    assert any(r["verdict"] == "unreviewed" for r in filled["reviews"] if r["record_id"] == "r4")


def test_ac28_digest_and_manifest_rejects():  # AC28
    thr = _thr()
    items, ev = _eval_fixture()
    snap = {"snapshot_key": "snap", "original_labels": {r["record_id"]: r["original_label"]
                                                       for r in ev["reviews"]}}
    a = thr.validate_evaluation_manifest(ev, snap, items)
    swapped = dict(ev, reviews=list(reversed(ev["reviews"])))
    b = thr.validate_evaluation_manifest(swapped, snap, items)
    assert a["truth_digest"] == b["truth_digest"]
    changed = dict(ev)
    changed["reviews"] = [dict(r) for r in ev["reviews"]]
    changed["reviews"][0]["verdict"] = "correct"
    c = thr.validate_evaluation_manifest(changed, snap, items)
    assert c["truth_digest"] != a["truth_digest"]
    with pytest.raises(thr.EvaluationError):
        thr.validate_evaluation_manifest(dict(ev, schema_version=2), snap, items)
    with pytest.raises(thr.EvaluationError):
        thr.validate_evaluation_manifest(dict(ev, purpose="nope"), snap, items)
    empty = dict(ev, frozen_ids=[], reviews=[])
    filled = thr.validate_evaluation_manifest(empty, snap, items)
    counts = thr.evaluate_threshold(items, [False] * 4, filled)
    assert counts.status == "empty_e0" and counts.precision is None


def test_ac22_label_change_stales_truth():  # AC22
    thr = _thr()
    items, ev = _eval_fixture()
    snap = {"snapshot_key": "snap", "original_labels": {"r1": "A", "r2": "A", "r3": "B", "r4": "B"}}
    ok = thr.validate_evaluation_manifest(ev, snap, items)
    snap2 = dict(snap, original_labels=dict(snap["original_labels"], r1="Z"))
    with pytest.raises(thr.EvaluationError):
        thr.validate_evaluation_manifest(ev, snap2, items)
    assert ok["truth_digest"]


def test_ac22_k_rescores_without_forging_reviews():  # AC22
    thr = _thr()
    items, ev = _eval_fixture()
    d1 = thr.truth_digest("snap", ev["frozen_ids"], ev["reviews"])
    # changing K is an analysis_key change, not a review change
    k1 = thr.evaluation_key("ak1", d1, "tuning", None)
    k2 = thr.evaluation_key("ak2", d1, "tuning", None)
    assert k1 != k2
    assert d1 == thr.truth_digest("snap", ev["frozen_ids"], ev["reviews"])


def test_ac27_template_and_confusion_ids():  # AC27
    exp = _exp()
    thr = _thr()
    recs = [
        {"path": "crop0.jpg", "image_path": "orig.jpg", "bbox": [0, 0, 5, 5],
         "obj_index": 0, "label": "A"},
        {"path": "crop1.jpg", "image_path": "orig.jpg", "bbox": [9, 9, 5, 5],
         "obj_index": 1, "label": "A"},
    ]
    raw = exp.evaluation_template(
        frozen_ids=["r0", "r1"], records=recs, record_ids=["r0", "r1"],
        labels=["A", "A"], snapshot_key="s", selection_method="本面板勾選",
        evaluation_id="t1", level="object")
    import json
    doc = json.loads(raw.decode("utf-8"))
    assert doc["reviews"][0]["verdict"] == "unreviewed"
    assert doc["record_catalog"][0]["bbox"] != doc["record_catalog"][1]["bbox"]
    assert doc["record_catalog"][0]["image_path"].endswith("orig.jpg")
    items, ev = _eval_fixture()
    counts = thr.evaluate_threshold(
        items, thr.candidate_mask(items, {"mode": "threshold", "threshold": 0.1}), ev)
    assert counts.confusion_ids["tp"] == ["r1"]
    assert counts.confusion_ids["fp"] == ["r2"]
    assert counts.confusion_ids["fn"] == ["r3"]
    assert counts.confusion_ids["tn"] == ["r4"]


def test_ac24_heldout_isolation():  # AC24
    thr = _thr()
    op = thr.make_operating_point(
        policy={"mode": "threshold", "threshold": 0.123456},
        analysis_key="ak", score_protocol_key="sp",
        tuning_evaluation_key="te", tuning_ids=["r1"], tuning_group_ids=["imgA"],
        policy_id="p1", created_at="2026-01-01T00:00:00Z")
    assert op["policy"]["threshold"] == 0.123456
    thr.validate_heldout(op, analysis_key="ak", score_protocol_key="sp",
                         heldout_ids=["h1"], heldout_groups=["imgB"],
                         tuning_complete=True)
    with pytest.raises(thr.EvaluationError):
        thr.validate_heldout(op, analysis_key="other", score_protocol_key="sp",
                             heldout_ids=["h1"], heldout_groups=["imgB"],
                             tuning_complete=True)
    with pytest.raises(thr.EvaluationError):
        thr.validate_heldout(op, analysis_key="ak", score_protocol_key="sp",
                             heldout_ids=["r1"], heldout_groups=["imgB"],
                             tuning_complete=True)
    with pytest.raises(thr.EvaluationError):
        thr.validate_heldout(op, analysis_key="ak", score_protocol_key="sp",
                             heldout_ids=["h1"], heldout_groups=["imgA"],
                             tuning_complete=True)
    with pytest.raises(thr.EvaluationError):
        thr.validate_heldout(op, analysis_key="ak", score_protocol_key="sp",
                             heldout_ids=["h1"], heldout_groups=["imgB"],
                             tuning_complete=False)


def test_ac34_operating_point_and_heldout():  # AC34
    thr = _thr()
    sp = thr.score_protocol_key(method_version="class_topk_margin_v1", k=5,
                                model="m", weights_id="w", recipe={"D": 2},
                                analysis_split="all")
    sp2 = thr.score_protocol_key(method_version="class_topk_margin_v1", k=6,
                                 model="m", weights_id="w", recipe={"D": 2},
                                 analysis_split="all")
    assert sp != sp2
    items = _items([("a", -0.1, "A")])
    c = thr.evaluate_threshold(items, [True], {
        "frozen_ids": ["a"],
        "reviews": [{"record_id": "a", "verdict": "incorrect"}],
    }, source_verified=False)
    assert c.status == "source_unverified" and c.precision is None


def test_ac27_unreviewed_not_correct():  # AC19/AC20
    thr = _thr()
    items = _items([("r1", -0.5, "A"), ("r5", -0.2, "A")])
    ev = {"frozen_ids": ["r1", "r5"],
          "reviews": [{"record_id": "r1", "verdict": "incorrect", "original_label": "A"}]}
    # r5 filled unreviewed by validate; evaluate sees incomplete
    snap = {"snapshot_key": "s", "original_labels": {"r1": "A", "r5": "A"}}
    filled = thr.validate_evaluation_manifest(
        {"schema_version": 1, "purpose": "tuning", "snapshot_key": "s",
         "frozen_ids": ["r1", "r5"], "reviews": ev["reviews"]}, snap, items)
    counts = thr.evaluate_threshold(
        items, thr.candidate_mask(items, {"mode": "threshold", "threshold": 0.0}), filled)
    assert counts.status == "incomplete" and counts.precision is None
    # r5 not in E0: PR unchanged for r1-only
    ev1 = {"frozen_ids": ["r1"],
           "reviews": [{"record_id": "r1", "verdict": "incorrect", "original_label": "A"}]}
    c1 = thr.evaluate_threshold(
        items, thr.candidate_mask(items, {"mode": "threshold", "threshold": 0.0}), ev1)
    assert c1.status == "ready" and c1.tp == 1 and c1.recall == 1
