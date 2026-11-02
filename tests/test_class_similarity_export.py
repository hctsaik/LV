"""驗收:class_similarity_export schema 2。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from class_similarity_oracle import almost


def _cs():
    import class_similarity as m
    return m


def _exp():
    import class_similarity_export as m
    return m


def _thr():
    import class_similarity_threshold as m
    return m


def _analyze(E, labels, ids, elig=None, reasons=None, k=1):
    n = len(ids)
    return _cs().analyze(
        E, labels, ids,
        eligibility=[True] * n if elig is None else elig,
        invalid_reasons=[[] for _ in range(n)] if reasons is None else reasons,
        k=k,
    )


def test_ac09_two_roi_same_image(tmp_path):  # AC09
    exp = _exp()
    img = tmp_path / "orig.jpg"
    img.write_bytes(b"img")
    recs = [
        {"path": tmp_path / "c0.jpg", "image_path": img, "bbox": [0, 0, 10, 10],
         "obj_index": 0, "label": "A", "split": "train"},
        {"path": tmp_path / "c1.jpg", "image_path": img, "bbox": [20, 20, 10, 10],
         "obj_index": 1, "label": "B", "split": "train"},
    ]
    E = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = _analyze(E, ["A", "B"], ["r0", "r1"])
    csv_b, js_b = exp.export_report(result, recs, scope="all", level="object",
                                    analysis_key="k")
    data = json.loads(js_b.decode("utf-8"))
    rec_by_id = {r["record_id"]: r for r in data["records"]}
    assert rec_by_id["r0"]["bbox"] != rec_by_id["r1"]["bbox"]
    assert rec_by_id["r0"]["obj_index"] == 0 and rec_by_id["r1"]["obj_index"] == 1
    assert rec_by_id["r0"]["image_path"].endswith("orig.jpg")
    assert rec_by_id["r0"]["path"] != rec_by_id["r1"]["path"]


def test_ac09_missing_image_reported(tmp_path):  # AC09
    exp = _exp()
    recs = [{"path": tmp_path / "missing.jpg", "label": "A", "split": "train"},
            {"path": tmp_path / "also.jpg", "label": "B", "split": "train"}]
    E = np.eye(2)
    result = _analyze(E, ["A", "B"], ["a", "b"])
    _, js_b = exp.export_report(result, recs, scope="all", analysis_key="k")
    data = json.loads(js_b.decode("utf-8"))
    assert data["records"][0]["path"].endswith("missing.jpg")
    assert not Path(data["records"][0]["path"]).exists()


def test_ac11_csv_json_roundtrip():  # AC11
    exp = _exp()
    recs = [
        {"path": "圖/甲.jpg", "label": "甲類", "split": "train"},
        {"path": "圖/乙.jpg", "label": "乙類", "split": "train"},
    ]
    E = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = _analyze(E, ["甲類", "乙類"], ["id甲", "id乙"])
    result.items[0].same = None  # keep a null
    csv_b, js_b = exp.export_report(result, recs, scope="all", analysis_key="鍵")
    data = json.loads(js_b.decode("utf-8"))
    assert data["schema_version"] == 2
    assert data["items"][0]["same"] is None
    assert "甲類" in js_b.decode("utf-8")
    text = csv_b.decode("utf-8-sig")
    assert "甲類" in text and "id甲" in text
    rec_ids = {row["record_id"] for row in data["records"]}
    for it in data["items"]:
        for n in it["same_neighbors"] + it["rival_neighbors"]:
            assert n["record_id"] in rec_ids


def test_ac11_recompute_mean_from_neighbors():  # AC11
    exp = _exp()
    E = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]])
    ids = ["a1", "a2", "b1", "b2"]
    recs = [{"path": f"{i}.jpg", "label": lab, "split": "train"}
            for i, lab in enumerate(["A", "A", "B", "B"])]
    result = _analyze(E, ["A", "A", "B", "B"], ids)
    _, js_b = exp.export_report(result, recs, scope="all", analysis_key="k")
    data = json.loads(js_b.decode("utf-8"))
    for it in data["items"]:
        if it["same_k"]:
            mean = sum(n["cosine"] for n in it["same_neighbors"]) / it["same_k"]
            assert almost(mean, it["same"])
        if it["rival_k"]:
            mean = sum(n["cosine"] for n in it["rival_neighbors"]) / it["rival_k"]
            assert almost(mean, it["other"])
        for n in it["same_neighbors"] + it["rival_neighbors"]:
            assert any(r["record_id"] == n["record_id"] for r in data["records"])


def test_ac31_selected_scope_full_index():  # AC31
    exp = _exp()
    thr = _thr()
    labels = ["A", "A", "B", "B", "C"]
    ids = ["a1", "a2", "b1", "b2", "c1"]
    E = np.array([
        [1.0, 0.0], [1.0, 0.0],
        [0.0, 1.0], [0.0, 1.0],
        [-1.0, 0.0],
    ])
    recs = [{"path": f"{i}.jpg", "label": lab, "split": "train"}
            for i, lab in enumerate(labels)]
    elig = [True, True, True, True, True]
    reasons = [[] for _ in ids]
    # extra invalid row
    E = np.vstack([E, [[0.0, 0.0]]])
    labels = labels + [None]
    ids = ids + ["inv"]
    recs = recs + [{"path": "bad.jpg", "label": None, "split": "train"}]
    elig = elig + [False]
    reasons = reasons + [["missing_provenance"]]
    result = _cs().analyze(E, labels, ids, eligibility=elig, invalid_reasons=reasons, k=1)
    policy = {"mode": "threshold", "threshold": 2.0}
    mask = thr.candidate_mask(result.items, policy)
    csv_sel, js_sel = exp.export_report(
        result, recs, scope="selected", local_selected_ids=["a1", "a2"],
        committed_preview=policy, analysis_key="k", level="image")
    data = json.loads(js_sel.decode("utf-8"))
    qids = {it["record_id"] for it in data["items"]}
    assert qids <= {"a1", "a2"}
    neighbor_ids = {n["record_id"] for it in data["items"]
                    for n in it["same_neighbors"] + it["rival_neighbors"]}
    assert "c1" not in qids and "c1" not in neighbor_ids
    assert "inv" not in qids
    assert {row["record_id"] for row in data["score_index"]} == set(ids)
    rebuilt = exp.rebuild_counts_from_score_index(data["score_index"], policy)
    full_sum = thr.summarize_candidates(result.items, mask)
    assert rebuilt["candidate_n"] == full_sum.candidate_n
    assert rebuilt["eligible_n"] == full_sum.eligible_n
    _, js_all = exp.export_report(result, recs, scope="all", committed_preview=policy,
                                  analysis_key="k")
    all_data = json.loads(js_all.decode("utf-8"))
    assert any(it["record_id"] == "inv" for it in all_data["items"])
    _, js_cand = exp.export_report(result, recs, scope="candidates",
                                   committed_preview=policy, analysis_key="k")
    cand = json.loads(js_cand.decode("utf-8"))
    assert "inv" not in {it["record_id"] for it in cand["items"]}
    for blob in (data, all_data, cand):
        rb = exp.rebuild_counts_from_score_index(blob["score_index"], policy)
        assert rb["input_count"] == len(ids)
        labels_seen = {p["label"] for p in rb["per_class"]}
        assert "A" in labels_seen and "B" in labels_seen and "C" in labels_seen


def test_ac23_export_rebuilds_candidates_and_pr():  # AC23
    exp = _exp()
    thr = _thr()
    ids = ["r1", "r2", "r3", "r4"]
    items_margins = [-0.5, -0.1, 0.1, 0.4]
    E = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]])
    labels = ["A", "A", "B", "B"]
    recs = [{"path": f"{i}.jpg", "label": lab, "split": "train"}
            for i, lab in enumerate(labels)]
    result = _analyze(E, labels, ids)
    # overwrite margins to the PR fixture (export must still persist exact values)
    for it, m in zip(result.items, items_margins):
        it.margin = m
        it.status = "ok"
    policy = {"mode": "threshold", "threshold": 0.123456}
    evaluation = {
        "evaluation_id": "e1", "evaluation_version": 1, "purpose": "tuning",
        "truth_digest": "d", "frozen_ids": ids,
        "reviews": [
            {"record_id": "r1", "original_label": "A", "verdict": "incorrect"},
            {"record_id": "r2", "original_label": "A", "verdict": "correct"},
            {"record_id": "r3", "original_label": "B", "verdict": "incorrect"},
            {"record_id": "r4", "original_label": "B", "verdict": "correct"},
        ],
    }
    mask = thr.candidate_mask(result.items, policy)
    counts = thr.evaluate_threshold(result.items, mask, evaluation)
    _, js_b = exp.export_report(result, recs, scope="candidates",
                                committed_preview=policy, evaluation=evaluation,
                                evaluation_counts=counts, analysis_key="k")
    data = json.loads(js_b.decode("utf-8"))
    assert data["policy"]["threshold"] == 0.123456
    rebuilt_mask_ids = [row["record_id"] for row in data["score_index"]
                        if row["margin"] is not None and row["margin"] < 0.123456]
    assert set(data["candidate_ids"]) == set(rebuilt_mask_ids)
    # rebuild PR from evaluation_items + score_index
    from types import SimpleNamespace
    fake = [SimpleNamespace(record_id=r["record_id"], margin=r["margin"],
                            label=r["label"], status="ok", reasons=[])
            for r in data["score_index"]]
    ev = {
        "frozen_ids": [r["record_id"] for r in data["evaluation_items"]],
        "reviews": data["evaluation_items"],
    }
    counts2 = thr.evaluate_threshold(fake, thr.candidate_mask(fake, policy), ev)
    assert counts2.tp == counts.tp and counts2.fp == counts.fp
    assert counts2.precision == counts.precision
