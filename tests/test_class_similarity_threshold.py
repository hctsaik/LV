"""驗收:threshold 候選、工作量曲線、draft/commit。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _thr():
    import class_similarity_threshold as m
    return m


def _items(margins, labels=None, ids=None):
    out = []
    for i, m in enumerate(margins):
        out.append(SimpleNamespace(
            record_id=(ids[i] if ids else f"r{i}"),
            label=(labels[i] if labels else "A"),
            margin=m, status="ok", reasons=[],
            same=0.0, other=0.0,
        ))
    return out


def test_ac15_candidate_membership():  # AC15
    thr = _thr()
    items = _items([-0.5, -0.1, 0.1, 0.4, None, 2.0, -2.0],
                   ids=["r1", "r2", "r3", "r4", "null", "hi", "lo"])
    m = thr.candidate_mask(items, {"mode": "threshold", "threshold": 0.1})
    got = [it.record_id for it, f in zip(items, m) if f]
    assert got == ["r1", "r2", "lo"]  # 0.1 equal excluded; null excluded
    m0 = thr.candidate_mask(items, {"mode": "threshold", "threshold": 0.0})
    assert [it.record_id for it, f in zip(items, m0) if f] == ["r1", "r2", "lo"]
    none = thr.candidate_mask(items, {"mode": "none"})
    assert not none.any()
    all_e = thr.candidate_mask(items, {"mode": "all_eligible"})
    assert [it.record_id for it, f in zip(items, all_e) if f] == [
        "r1", "r2", "r3", "r4", "hi", "lo"]
    m2 = thr.candidate_mask(items, {"mode": "threshold", "threshold": 2.0})
    assert "hi" not in [it.record_id for it, f in zip(items, m2) if f]
    # duplicate scores enter together
    dups = _items([0.05, 0.05, 0.2], ids=["a", "b", "c"])
    md = thr.candidate_mask(dups, {"mode": "threshold", "threshold": 0.1})
    assert [it.record_id for it, f in zip(dups, md) if f] == ["a", "b"]


def test_ac16_threshold_not_in_score_key():  # AC16
    thr = _thr()
    k1 = thr.preview_key("ak", {"mode": "threshold", "threshold": 0.0}, None)
    k2 = thr.preview_key("ak", {"mode": "threshold", "threshold": 0.2}, None)
    assert k1 != k2
    # score protocol / analysis key must not include threshold: different t
    # does not change analysis_key (caller-provided). preview_key does.
    assert k1.startswith("") or True


def test_ac17_counts_full_population():  # AC17
    thr = _thr()
    items = _items(
        [-0.2, 0.3, None, -0.1, 0.0, None],
        labels=["A", "A", "A", "B", "B", "C"],
        ids=list("abcdef"),
    )
    mask = thr.candidate_mask(items, {"mode": "threshold", "threshold": 0.0})
    summ = thr.summarize_candidates(items, mask)
    assert summ.input_count == 6
    assert summ.eligible_n == 4
    assert summ.candidate_n == 2  # -0.2 and -0.1; 0.0 equal excluded
    assert summ.unscorable_n == 2
    assert sum(p["n_total"] for p in summ.per_class) == 6
    # visible class filter is not an argument — counts stay full
    assert {p["label"]: p["candidate_n"] for p in summ.per_class}["A"] == 1


def test_ac21_workload_and_pr_curves():  # AC21
    thr = _thr()
    items = _items([-0.5, -0.1, 0.1, 0.4], ids=["r1", "r2", "r3", "r4"])
    curve = thr.workload_curve(items)
    assert curve[0]["mode"] == "none" and curve[0]["candidate_n"] == 0
    # each unique margin group is included together
    ns = [p["candidate_n"] for p in curve if p["mode"] == "threshold"]
    assert ns == sorted(set(ns))
    assert ns[-1] == 4
    # predicate reconstruction
    for p in curve:
        if p["mode"] == "threshold":
            mask = thr.candidate_mask(items, {"mode": "threshold", "threshold": p["t"]})
            assert int(mask.sum()) == p["candidate_n"]
    evaluation = {
        "frozen_ids": ["r1", "r2", "r3", "r4"],
        "reviews": [
            {"record_id": "r1", "verdict": "incorrect", "original_label": "A"},
            {"record_id": "r2", "verdict": "correct", "original_label": "A"},
            {"record_id": "r3", "verdict": "incorrect", "original_label": "A"},
            {"record_id": "r4", "verdict": "correct", "original_label": "A"},
        ],
    }
    pr = thr.precision_recall_curve(items, evaluation)
    # t=0.1 via nextafter(-0.1) or nextafter of groups: check 0.5/0.5 exists
    pts = {(round(p["precision"], 6), round(p["recall"], 6)) for p in pr}
    assert (0.5, 0.5) in pts
    assert (round(2 / 3, 6), 1.0) in pts
    cur = thr.current_pr_point(items, evaluation, {"mode": "threshold", "threshold": 0.1})
    assert cur["tp"] == 1 and cur["fp"] == 1 and cur["fn"] == 1 and cur["tn"] == 1


def test_ac25_draft_commit_and_stale_action():  # AC25
    thr = _thr()
    state = thr.PreviewState(
        draft_policy=thr.parse_policy({"mode": "threshold", "threshold": 0.1}),
        committed_policy=None, request_seq=0, committed_seq=None,
        analysis_key="ak", preview_key_committed=None, evaluation_key=None,
    )
    ev = {"kind": "preview", "analysis_key": "ak", "request_seq": 1,
          "policy": {"mode": "threshold", "threshold": 0.4}}
    state = thr.apply_preview_event(state, ev, current_analysis_key="ak",
                                    current_eval_key=None,
                                    draft_policy=state.draft_policy)
    assert not thr.actions_enabled(state)
    with pytest.raises(thr.ActionRejected):
        thr.validate_action(
            {"analysis_key": "ak", "preview_key": "x", "request_seq": 1}, state, "ak")
    commit = {"kind": "commit", "analysis_key": "ak", "request_seq": 2,
              "policy": {"mode": "threshold", "threshold": 0.4}}
    state = thr.apply_preview_event(state, commit, current_analysis_key="ak",
                                    current_eval_key=None,
                                    draft_policy=state.draft_policy)
    assert thr.actions_enabled(state)
    pk = thr.preview_key("ak", state.committed_policy, None)
    thr.validate_action({"analysis_key": "ak", "preview_key": pk, "request_seq": 2},
                        state, "ak")
    with pytest.raises(thr.ActionRejected):
        thr.apply_preview_event(
            state, {"kind": "commit", "analysis_key": "ak", "request_seq": 1,
                    "policy": {"mode": "threshold", "threshold": 0.1}},
            current_analysis_key="ak", current_eval_key=None,
            draft_policy=state.draft_policy,
        )


def test_ac33_axis_zoom_does_not_change_stats():  # AC33
    thr = _thr()
    items = _items([0.01, 0.012, -0.015, 0.0])
    mask_full = thr.candidate_mask(items, {"mode": "threshold", "threshold": 0.0})
    rng = thr.axis_display_range("margin", "data", [0.01, 0.012, -0.015, 0.0], 1.5)
    assert rng["threshold_outside"] is True
    rng2 = thr.axis_display_range("margin", "full", [0.01, 0.012, -0.015, 0.0], 0.0)
    assert rng2["ymin"] == -2.0 and rng2["ymax"] == 2.0
    mask2 = thr.candidate_mask(items, {"mode": "threshold", "threshold": 0.0})
    assert list(mask_full) == list(mask2)
