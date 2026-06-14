"""Unit tests for scripts/quiz.py — the annotator-agreement quiz (§1)."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from quiz import (
    SKINS,
    build_quiz,
    cohen_kappa,
    consensus_labels,
    fleiss_kappa,
    geometric_skin,
    score_quiz,
    self_consistency,
    vs_golden_accuracy,
)


# ── consensus aggregation (M1: 組考卷 → consensus subset / gray band) ────

def test_consensus_unanimous_vs_split():
    # qid 1: all agree "缺陷" → consensus; qid 2: 2-2 split → gray band
    r1 = {1: "缺陷", 2: "缺陷", 3: "OK"}
    r2 = {1: "缺陷", 2: "OK", 3: "OK"}
    r3 = {1: "缺陷", 2: "缺陷"}
    r4 = {1: "缺陷", 2: "OK"}
    out = consensus_labels([r1, r2, r3, r4])
    assert out[1] == {"label": "缺陷", "agreement": 1.0, "n_votes": 4,
                      "consensus": True}
    assert out[2]["agreement"] == 0.5 and out[2]["consensus"] is False
    assert out[3]["label"] == "OK" and out[3]["consensus"] is True  # 2/2 agree


def test_consensus_threshold_and_min_votes():
    maps = [{1: "A"}, {1: "A"}, {1: "A"}, {1: "B"}]  # 3/4 = 0.75
    # unanimous default → not consensus
    assert consensus_labels(maps)[1]["consensus"] is False
    # 0.7 threshold → consensus, label = majority A
    relaxed = consensus_labels(maps, agree_thresh=0.7)
    assert relaxed[1]["consensus"] is True and relaxed[1]["label"] == "A"
    # a qid only one rater answered is omitted (can't judge agreement)
    assert consensus_labels([{5: "A"}, {6: "B"}]) == {}


# ── geometric_skin whitelist ────────────────────────────────────────────

def _img(w=40, h=30):
    arr = np.random.default_rng(0).integers(0, 255, (h, w, 3)).astype("uint8")
    return Image.fromarray(arr)

def test_geometric_skin_rotations_and_flips_preserve_pixels():
    img = _img()
    assert geometric_skin(img, "rot180").size == img.size
    assert geometric_skin(img, "rot90").size == (img.height, img.width)
    # flip is reversible → same pixels back
    once = geometric_skin(img, "fliplr")
    twice = geometric_skin(once, "fliplr")
    assert np.array_equal(np.asarray(twice), np.asarray(img))

def test_geometric_skin_crop_shrinks():
    img = _img(40, 30)
    out = geometric_skin(img, "crop90")
    assert out.size[0] < 40 and out.size[1] < 30

def test_geometric_skin_rejects_photometric():
    for bad in ("contrast", "brightness", "sharpen", "blur"):
        with pytest.raises(ValueError):
            geometric_skin(_img(), bad)
    assert set(SKINS).isdisjoint({"contrast", "brightness", "sharpen"})


# ── build_quiz ──────────────────────────────────────────────────────────

def _records(n, classes=("A", "B")):
    return [{"label": classes[i % len(classes)], "path": f"x{i}.jpg"}
            for i in range(n)]

def test_build_quiz_mix_and_repeat_pairs():
    recs = _records(40)
    dis = np.linspace(1, 0, 40)  # first records most disputed
    quiz = build_quiz(recs, dis, n_questions=16, seed=1)
    qs = quiz["questions"]
    assert len(qs) <= 16 + 4  # base + repeats appended
    kinds = {q["kind"] for q in qs}
    assert "dispute" in kinds and "golden" in kinds
    # every repeat pair references a real original + a skinned repeat
    by_qid = {q["qid"]: q for q in qs}
    for a, b in quiz["repeat_pairs"]:
        assert by_qid[a]["skin"] is None and by_qid[b]["skin"] in SKINS
        assert by_qid[a]["record_idx"] == by_qid[b]["record_idx"]
    # golden/distractor questions carry a golden label; disputes do not
    for q in qs:
        if q["kind"] in ("golden", "distractor"):
            assert q["golden_label"] is not None and q["qid"] in quiz["key"]
        if q["kind"] in ("dispute", "repeat"):
            assert q["golden_label"] is None

def test_build_quiz_empty():
    q = build_quiz([], np.zeros(0))
    assert q["questions"] == [] and q["key"] == {}

def test_build_quiz_blind_shuffle_breaks_adjacency():
    recs = _records(30)
    quiz = build_quiz(recs, np.linspace(1, 0, 30), n_questions=20, seed=2)
    # at least one repeat pair should not be adjacent after the shuffle
    qids = [q["qid"] for q in quiz["questions"]]
    pos = {qid: i for i, qid in enumerate(qids)}
    if quiz["repeat_pairs"]:
        gaps = [abs(pos[a] - pos[b]) for a, b in quiz["repeat_pairs"]]
        assert max(gaps) > 1


# ── scoring ─────────────────────────────────────────────────────────────

def test_self_consistency():
    pairs = [(0, 1), (2, 3), (4, 5)]
    answers = {0: "A", 1: "A", 2: "A", 3: "B", 4: "B", 5: "B"}
    assert self_consistency(answers, pairs) == pytest.approx(2 / 3)
    assert self_consistency({}, pairs) == 0.0

def test_vs_golden_accuracy():
    key = {0: "A", 1: "B", 2: "A"}
    answers = {0: "A", 1: "B", 2: "B"}  # 2/3 correct
    assert vs_golden_accuracy(answers, key) == pytest.approx(2 / 3)
    assert vs_golden_accuracy({}, key) == 0.0

def test_score_quiz_pass_lines():
    quiz = {"questions": [{"qid": i} for i in range(10)],
            "repeat_pairs": [(0, 1), (2, 3)], "key": {4: "A", 5: "B"}}
    answers = {0: "A", 1: "A", 2: "B", 3: "B", 4: "A", 5: "B"}
    r = score_quiz(answers, quiz)
    assert r["self_consistency"] == 1.0 and r["self_pass"] is True
    assert r["vs_golden"] == 1.0 and r["golden_pass"] is True


# ── kappa ───────────────────────────────────────────────────────────────

def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]) == pytest.approx(1.0)
    # total disagreement on a 2-class balanced set → negative kappa
    assert cohen_kappa(["A", "A", "B", "B"], ["B", "B", "A", "A"]) < 0
    assert cohen_kappa([], []) == 0.0

def test_fleiss_kappa_perfect_and_random():
    # 3 items, 4 raters, everyone agrees → kappa 1.0
    perfect = np.array([[4, 0], [0, 4], [4, 0]])
    assert fleiss_kappa(perfect) == pytest.approx(1.0)
    # split votes everywhere → near 0 / negative
    split = np.array([[2, 2], [2, 2], [2, 2]])
    assert fleiss_kappa(split) < 0.1
    assert fleiss_kappa(np.zeros((0, 2))) == 0.0

def test_fleiss_kappa_too_few_raters():
    assert fleiss_kappa(np.array([[1, 0], [0, 1]])) == 0.0  # 1 rater → undefined
