"""Independent numeric oracle for class-similarity tests.

This is a test-only nested-loop implementation. It must not import
``class_similarity.analyze`` or any production top-K helper.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

EPS = 1e-12


def scale_safe_normalize(row) -> np.ndarray | None:
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    if row.size == 0 or not np.all(np.isfinite(row)):
        return None
    a = float(np.max(np.abs(row)))
    if a == 0.0:
        return None
    v = row / a
    b = float(np.linalg.norm(v))
    if b == 0.0:
        return None
    if a <= EPS / b:
        return None
    u = v / b
    if not np.all(np.isfinite(u)):
        return None
    return u


def cosine(u, v) -> float:
    return float(np.clip(np.dot(u, v), -1.0, 1.0))


@dataclass
class OracleNeighbor:
    record_id: str
    cosine: float
    index: int


@dataclass
class OracleItem:
    record_id: str
    input_index: int
    label: str | None
    same: float | None
    other: float | None
    margin: float | None
    rival_class: str | None
    same_k: int
    rival_k: int
    same_neighbors: list[OracleNeighbor]
    rival_neighbors: list[OracleNeighbor]
    tied_rival_count: int
    tied_rival_classes: list[str]
    status: str
    reasons: list[str]


def oracle_analyze(embeddings, labels, record_ids, *, eligibility, invalid_reasons,
                   k=5):
    """O(N^2) full pairwise oracle. Allowed in tests only."""
    E = np.asarray(embeddings, dtype=np.float64)
    n = E.shape[0]
    items: list[OracleItem] = []
    units: list[np.ndarray | None] = []
    can_use = []
    for i in range(n):
        reasons = list(invalid_reasons[i] or [])
        u = None
        usable = bool(eligibility[i])
        if not usable and not reasons:
            reasons.append("ineligible")
        if usable:
            u = scale_safe_normalize(E[i])
            if u is None:
                usable = False
                if not np.all(np.isfinite(np.asarray(E[i], dtype=np.float64))):
                    reasons.append("non_finite_vector")
                else:
                    reasons.append("vector_too_small")
        lab = labels[i]
        if usable and (lab is None or str(lab).strip() == ""):
            usable = False
            reasons.append("missing_label")
        units.append(u if usable else None)
        can_use.append(usable)
        status = "ok" if usable else ("invalid" if not eligibility[i] else "unscorable")
        items.append(OracleItem(
            record_id=record_ids[i], input_index=i, label=lab,
            same=None, other=None, margin=None, rival_class=None,
            same_k=0, rival_k=0, same_neighbors=[], rival_neighbors=[],
            tied_rival_count=0, tied_rival_classes=[],
            status=status, reasons=reasons,
        ))

    def class_topk(i, cls):
        cands: list[OracleNeighbor] = []
        for j in range(n):
            if not can_use[j]:
                continue
            if labels[j] != cls:
                continue
            if record_ids[j] == record_ids[i]:
                continue
            cands.append(OracleNeighbor(
                record_id=record_ids[j],
                cosine=cosine(units[i], units[j]),
                index=j,
            ))
        cands.sort(key=lambda c: (-c.cosine, c.record_id))
        k_eff = min(k, len(cands))
        chosen = cands[:k_eff]
        if not chosen:
            return None, 0, []
        mean = float(sum(c.cosine for c in chosen) / k_eff)
        return mean, k_eff, chosen

    classes = sorted({labels[i] for i in range(n) if can_use[i] and labels[i] is not None})
    for i in range(n):
        if not can_use[i]:
            continue
        own = labels[i]
        same, same_k, same_nb = class_topk(i, own)
        others = []
        for c in classes:
            if c == own:
                continue
            s, kk, nb = class_topk(i, c)
            if s is not None:
                others.append((c, s, kk, nb))
        if same is None:
            items[i].reasons.append("insufficient_same")
        items[i].same = same
        items[i].same_k = same_k
        items[i].same_neighbors = same_nb
        if not others:
            items[i].other = None
            items[i].rival_class = None
            items[i].rival_k = 0
            items[i].rival_neighbors = []
            items[i].tied_rival_count = 0
            items[i].tied_rival_classes = []
            items[i].reasons.append("no_other_class")
        else:
            best = max(s for _, s, _, _ in others)
            tied = sorted(c for c, s, _, _ in others if s == best)
            rival = tied[0]
            picked = next(t for t in others if t[0] == rival)
            items[i].other = picked[1]
            items[i].rival_class = rival
            items[i].rival_k = picked[2]
            items[i].rival_neighbors = picked[3]
            items[i].tied_rival_count = len(tied)
            items[i].tied_rival_classes = tied[:10]
            if len(tied) > 1:
                items[i].reasons.append("tied_rivals")
        if items[i].same is not None and items[i].other is not None:
            items[i].margin = items[i].same - items[i].other
        else:
            items[i].margin = None
    return items


def box_stats_oracle(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return None
    q1, med, q3 = np.percentile(arr, [25, 50, 75], method="linear")
    iqr = float(q3 - q1)
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = arr[(arr >= lo) & (arr <= hi)]
    return {
        "q1": float(q1), "median": float(med), "q3": float(q3), "iqr": iqr,
        "whisker_low": float(inside.min()) if inside.size else float(arr.min()),
        "whisker_high": float(inside.max()) if inside.size else float(arr.max()),
        "n": int(arr.size),
    }


def canonical_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def almost(a, b, tol=1e-6):
    if a is None or b is None:
        return a is None and b is None
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol
