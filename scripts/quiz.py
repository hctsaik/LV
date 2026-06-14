"""Annotator-agreement quiz (defect-mechanisms v2 §1).

Turns disputed samples into a blind consistency test: re-skinned repeats
measure a rater's self-consistency, distractors (similar images with a
KNOWN different label) separate "judgment drift" from "this item is
genuinely hard", golden anchors measure agreement vs the standard, and
Fleiss/Cohen kappa measures agreement between raters.

Honest boundary (per the design): this measures rater stability and
agreement on EXISTING cases — not detection of novel defects, nor whether
the golden answers are themselves correct.

Augmentation whitelist is GEOMETRIC ONLY (crop/rotate/flip) so re-skinning
perturbs visual memory without changing the defect's physical evidence —
contrast/brightness/sharpen are forbidden.

Framework-free: no streamlit imports, unit-testable.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from PIL import Image

# geometric-only skin transforms (defect-mechanisms v2 §1.2 whitelist)
SKINS = ["rot90", "rot180", "rot270", "fliplr", "flipud", "crop90"]


def geometric_skin(img: Image.Image, transform: str) -> Image.Image:
    """Apply ONE whitelisted geometric transform. Forbidden transforms
    (anything photometric) raise ValueError — they could shift a gray-zone
    judgment and would measure the wrong thing."""
    if transform == "rot90":
        return img.rotate(90, expand=True)
    if transform == "rot180":
        return img.rotate(180, expand=True)
    if transform == "rot270":
        return img.rotate(270, expand=True)
    if transform == "fliplr":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if transform == "flipud":
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if transform == "crop90":
        w, h = img.size
        dx, dy = int(w * 0.05), int(h * 0.05)
        return img.crop((dx, dy, w - dx, h - dy))
    raise ValueError(f"transform {transform!r} not in geometric whitelist {SKINS}")


def build_quiz(
    records: Sequence[dict],
    disagreement: np.ndarray,
    n_questions: int = 16,
    distractor_ratio: float = 0.25,
    repeat_ratio: float = 0.2,
    golden_ratio: float = 0.12,
    seed: int = 0,
) -> dict:
    """Assemble one blind quiz form (defect-mechanisms v2 §1.1 mix).

    ``disagreement`` is the per-record kNN label-disagreement (see
    interaction.compute_label_disagreement): high = disputed, low = clear.
    Returns {questions, repeat_pairs, key} where each question is
    {qid, record_idx, skin, kind, golden_label|None} and ``key`` maps qid
    -> golden_label for the scorable (golden + distractor) questions.
    """
    rng = np.random.default_rng(seed)
    n = len(records)
    if n == 0:
        return {"questions": [], "repeat_pairs": [], "key": {}}
    order = np.argsort(disagreement)[::-1]            # disputed first
    disputed = [int(i) for i in order]
    clear = [int(i) for i in order[::-1]]             # clearest first
    labels = [r.get("label", "") for r in records]

    n_q = min(n_questions, n)
    n_repeat = max(0, int(round(n_q * repeat_ratio)))
    n_golden = max(0, int(round(n_q * golden_ratio)))
    n_distract = max(0, int(round(n_q * distractor_ratio)))
    n_dispute = max(1, n_q - n_repeat - n_golden - n_distract)

    questions: list[dict] = []
    used: set[int] = set()
    qid = 0

    def _take(pool, k):
        out = []
        for i in pool:
            if len(out) >= k:
                break
            if i not in used:
                used.add(i)
                out.append(i)
        return out

    dispute_items = _take(disputed, n_dispute)
    for i in dispute_items:
        questions.append({"qid": qid, "record_idx": i, "skin": None,
                          "kind": "dispute", "golden_label": None})
        qid += 1

    # golden anchors: clearest items, their own label is the golden answer
    for i in _take(clear, n_golden):
        questions.append({"qid": qid, "record_idx": i, "skin": None,
                          "kind": "golden", "golden_label": labels[i]})
        qid += 1

    # distractors: clear items of a label different from the dispute items'
    dispute_labels = {labels[i] for i in dispute_items}
    distract_pool = [i for i in clear
                     if labels[i] not in dispute_labels or len(set(labels)) <= 1]
    for i in _take(distract_pool, n_distract):
        questions.append({"qid": qid, "record_idx": i, "skin": None,
                          "kind": "distractor", "golden_label": labels[i]})
        qid += 1

    # re-skinned repeats of a few dispute items (self-consistency pairs)
    repeat_pairs: list[tuple[int, int]] = []
    for i in dispute_items[:n_repeat]:
        orig_qid = next(q["qid"] for q in questions if q["record_idx"] == i
                        and q["kind"] == "dispute")
        skin = SKINS[int(rng.integers(len(SKINS)))]
        questions.append({"qid": qid, "record_idx": i, "skin": skin,
                          "kind": "repeat", "golden_label": None})
        repeat_pairs.append((orig_qid, qid))
        qid += 1

    rng.shuffle(questions)  # blind: break the original→repeat adjacency
    key = {q["qid"]: q["golden_label"] for q in questions
           if q["golden_label"] is not None}
    return {"questions": questions, "repeat_pairs": repeat_pairs, "key": key}


def self_consistency(answers: dict, repeat_pairs: Sequence[tuple[int, int]]) -> float:
    """Fraction of re-skin pairs answered identically (intra-rater).
    Pairs with a missing answer are skipped; no answered pair → 0.0."""
    hits, total = 0, 0
    for a_qid, b_qid in repeat_pairs:
        if a_qid in answers and b_qid in answers:
            total += 1
            if answers[a_qid] == answers[b_qid]:
                hits += 1
    return hits / total if total else 0.0


def vs_golden_accuracy(answers: dict, key: dict) -> float:
    """Accuracy on golden + distractor questions (agreement vs standard)."""
    graded = [(answers[q], g) for q, g in key.items() if q in answers]
    if not graded:
        return 0.0
    return sum(a == g for a, g in graded) / len(graded)


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa between two raters' label sequences (paired)."""
    a, b = list(a), list(b)
    if not a or len(a) != len(b):
        return 0.0
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    ntot = m.sum()
    po = np.trace(m) / ntot
    pe = float((m.sum(0) * m.sum(1)).sum()) / (ntot * ntot)
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else 1.0


def fleiss_kappa(rating_counts: np.ndarray) -> float:
    """Fleiss' kappa from an (items × categories) matrix of per-item rater
    counts (each row sums to the number of raters). Measures agreement
    among ≥2 raters; constant ratings → 1.0."""
    m = np.asarray(rating_counts, dtype=float)
    if m.size == 0:
        return 0.0
    n_raters = m.sum(axis=1)
    if not np.allclose(n_raters, n_raters[0]) or n_raters[0] < 2:
        # unequal rater counts or <2 raters — undefined; clamp gracefully
        n = n_raters[0] if len(n_raters) else 0
        if n < 2:
            return 0.0
    n = n_raters[0]
    N = len(m)
    p_j = m.sum(axis=0) / (N * n)
    P_i = (np.sum(m * m, axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = float(np.sum(p_j * p_j))
    return (P_bar - P_e) / (1 - P_e) if (1 - P_e) > 1e-12 else 1.0


def consensus_labels(
    answer_maps: Sequence[dict],
    *,
    agree_thresh: float = 1.0,
    min_votes: int = 2,
) -> dict:
    """Aggregate raters' ``{qid: answer}`` maps into per-question consensus.

    This is what turns 組考卷 from "a kappa number" into "a usable subset":
    kappa says whether the ruler is stable overall, but the per-question vote
    tally says WHICH items the raters actually agree on (trustworthy ground
    truth) versus which are split (the gray band, no ground truth).

    For every qid answered by at least ``min_votes`` raters, tallies the votes:
      - ``label``: the majority answer,
      - ``agreement``: top-vote share ∈ (0,1],
      - ``n_votes``: how many raters answered it,
      - ``consensus``: ``agreement >= agree_thresh`` (1.0 = unanimous).

    Returns ``{qid: {label, agreement, n_votes, consensus}}``. Items below
    ``min_votes`` are omitted (can't judge agreement from one vote). The
    caller joins qid → image/box and exports the consensus subset (for the
    evaluation gate) and routes non-consensus items to gray-zone review.
    """
    qids: set = set()
    for m in answer_maps:
        qids |= set(m)
    out: dict = {}
    for qid in sorted(qids):
        votes = [m[qid] for m in answer_maps if qid in m]
        if len(votes) < min_votes:
            continue
        label, top = Counter(votes).most_common(1)[0]
        agreement = top / len(votes)
        out[qid] = {
            "label": label,
            "agreement": round(agreement, 3),
            "n_votes": len(votes),
            "consensus": agreement >= agree_thresh,
        }
    return out


def score_quiz(answers: dict, quiz: dict) -> dict:
    """Single-rater report: self-consistency, vs-golden accuracy, counts.
    Pass lines (defect-mechanisms v2 §1.3): self ≥0.90, vs-golden ≥0.85."""
    sc = self_consistency(answers, quiz["repeat_pairs"])
    vg = vs_golden_accuracy(answers, quiz["key"])
    return {
        "self_consistency": round(sc, 3),
        "vs_golden": round(vg, 3),
        "self_pass": sc >= 0.90,
        "golden_pass": vg >= 0.85,
        "n_answered": len(answers),
        "n_questions": len(quiz["questions"]),
        "n_repeat_pairs": len(quiz["repeat_pairs"]),
    }
