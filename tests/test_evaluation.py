"""Unit tests for the detection-evaluation gate (scripts/evaluation.py) and
the prediction ingestion (interaction.load_predictions_csv)."""
from __future__ import annotations

import pytest

from evaluation import (
    consensus_flags,
    evaluate_detections,
    iou_xywh,
    match_image,
)
from interaction import load_predictions_csv


def box(cls, cx, cy, w=0.2, h=0.2, score=None):
    b = {"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h}
    if score is not None:
        b["score"] = score
    return b


# ── IoU ─────────────────────────────────────────────────────────────────

def test_iou_identical_disjoint_partial():
    a = box("x", 0.5, 0.5)
    assert iou_xywh(a, a) == pytest.approx(1.0)
    assert iou_xywh(a, box("x", 0.9, 0.9)) == pytest.approx(0.0)
    assert iou_xywh(a, box("x", 0.6, 0.5)) == pytest.approx(1 / 3, abs=1e-6)


# ── single-image matching ───────────────────────────────────────────────

def test_match_perfect_fp_fn():
    gt = [box("d", 0.3, 0.3), box("d", 0.7, 0.7)]
    preds = [box("d", 0.3, 0.3, score=0.9),  # TP
             box("d", 0.71, 0.69, score=0.8),  # TP (high IoU)
             box("d", 0.1, 0.9, score=0.5)]  # FP (no GT)
    tp, fn, fp = match_image(gt, preds)
    assert len(tp) == 2 and fn == [] and len(fp) == 1


def test_match_class_aware_and_conf_threshold():
    gt = [box("crack", 0.5, 0.5)]
    # wrong-class prediction → GT unmatched (FN) + that pred is FP
    tp, fn, fp = match_image(gt, [box("scratch", 0.5, 0.5, score=0.9)])
    assert tp == [] and fn == [0] and fp == [0]
    # right class but below conf threshold → filtered → GT becomes FN
    tp, fn, fp = match_image(gt, [box("crack", 0.5, 0.5, score=0.2)],
                             conf_thresh=0.5)
    assert tp == [] and fn == [0] and fp == []


# ── full evaluation ─────────────────────────────────────────────────────

def test_evaluate_recall_and_escape_list():
    gt = {"a.jpg": [box("d", 0.5, 0.5)], "b.jpg": [box("d", 0.3, 0.3)]}
    pred = {"a.jpg": [box("d", 0.5, 0.5, score=0.9)], "b.jpg": []}  # b missed
    r = evaluate_detections(gt, pred)
    assert r["per_class"]["d"]["recall"] == 0.5
    assert r["overall"]["fn"] == 1 and r["overall"]["tp"] == 1
    assert r["false_negatives"] == [
        {"filename": "b.jpg", "cls": "d", "box": {"cx": 0.3, "cy": 0.3, "w": 0.2, "h": 0.2}}]


def test_evaluate_perfect_no_escapes():
    gt = {"a.jpg": [box("d", 0.5, 0.5)]}
    pred = {"a.jpg": [box("d", 0.5, 0.5, score=0.9)]}
    r = evaluate_detections(gt, pred)
    assert r["overall"]["recall"] == 1.0 and r["false_negatives"] == []


def test_consensus_filter_excludes_gray_band_from_recall():
    # one image, one GT box that annotators did NOT agree on, and no prediction
    gt = {"a.jpg": [box("d", 0.5, 0.5)]}
    pred = {"a.jpg": []}
    # without consensus info → counts as a real escape
    base = evaluate_detections(gt, pred)
    assert base["per_class"]["d"]["fn"] == 1 and base["gray"]["total"] == 0
    # mark the box gray (no consensus) → it leaves the signable recall entirely
    r = evaluate_detections(gt, pred, consensus_by_image={"a.jpg": [False]})
    assert r["per_class"] == {} and r["false_negatives"] == []
    assert r["gray"] == {"tp": 0, "fn": 1, "total": 1}
    assert r["overall"]["recall"] == 0.0 and r["overall"]["n_gt"] == 0


def test_confusion_records_cross_class_when_not_class_aware():
    gt = {"a.jpg": [box("crack", 0.5, 0.5)]}
    pred = {"a.jpg": [box("scratch", 0.5, 0.5, score=0.9)]}
    r = evaluate_detections(gt, pred, class_aware=False)
    assert ("crack", "scratch") in r["confusion"]  # predicted scratch for a crack


# ── consensus → per-box flags (M3) ──────────────────────────────────────

def test_consensus_flags_image_level():
    gt = {"a.jpg": [box("d", 0.5, 0.5)],
          "b.jpg": [box("d", 0.3, 0.3), box("d", 0.7, 0.7)]}
    rows = [{"filename": "a.jpg", "consensus": "True"},
            {"filename": "b.jpg", "consensus": "False"}]
    cby, n_c, n_g = consensus_flags(rows, gt)
    assert cby == {"a.jpg": [True], "b.jpg": [False, False]}
    assert n_c == 1 and n_g == 2


def test_consensus_flags_box_level_iou_match():
    # two GT boxes; a box-level consensus row matches only the first
    gt = {"a.jpg": [box("d", 0.5, 0.5), box("d", 0.2, 0.2)]}
    rows = [{"filename": "a.jpg", "consensus": "true",
             "cx": "0.5", "cy": "0.5", "w": "0.2", "h": "0.2"}]
    cby, n_c, n_g = consensus_flags(rows, gt)
    assert cby["a.jpg"] == [True, False]  # 2nd GT box has no consensus → gray
    assert n_c == 1 and n_g == 1
    # a box that annotators disagreed on (consensus=false) → its GT box is gray
    rows2 = [{"filename": "a.jpg", "consensus": "false",
              "cx": "0.5", "cy": "0.5", "w": "0.2", "h": "0.2"}]
    cby2, n_c2, _ = consensus_flags(rows2, gt)
    assert cby2["a.jpg"] == [False, False] and n_c2 == 0


# ── predictions ingestion (M5) ──────────────────────────────────────────

def test_load_predictions_csv(tmp_path):
    p = tmp_path / "pred.csv"
    p.write_text(
        "filename,class,cx,cy,w,h,score\n"
        "a.jpg,crack,0.5,0.5,0.2,0.2,0.91\n"
        "a.jpg,scratch,0.2,0.2,0.1,0.1,0.4\n"
        "bad,row,only\n", encoding="utf-8")
    out = load_predictions_csv(p)
    assert set(out) == {"a.jpg"} and len(out["a.jpg"]) == 2
    assert out["a.jpg"][0] == {"cls": "crack", "cx": 0.5, "cy": 0.5,
                               "w": 0.2, "h": 0.2, "score": 0.91}


def test_load_predictions_csv_confidence_alias_and_missing(tmp_path):
    p = tmp_path / "pred2.csv"
    p.write_text("filename,class,cx,cy,w,h,confidence\nx.png,d,0.5,0.5,0.3,0.3,0.7\n",
                 encoding="utf-8")
    out = load_predictions_csv(p)
    assert out["x.png"][0]["score"] == 0.7
    assert load_predictions_csv(tmp_path / "nope.csv") == {}
