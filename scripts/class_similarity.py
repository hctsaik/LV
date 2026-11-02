"""Blocked exact Top-K class similarity (same / rival-other / margin).

Pure arrays. No Streamlit, disk I/O, or model loading.
"""
from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

SCHEMA_VERSION = 2
METHOD_VERSION = "class_topk_margin_v1"
MAX_ROWS = 5000
DEFAULT_K = 5
QUERY_BLOCK = 128
REFERENCE_BLOCK = 512
NORM_EPS = 1e-12
TIED_RIVAL_LIMIT = 10
K_MIN, K_MAX = 1, 20


class ClassSimilarityError(ValueError):
    pass


class InputShapeError(ClassSimilarityError):
    pass


class InvalidKError(ClassSimilarityError):
    pass


class TooManyRowsError(ClassSimilarityError):
    pass


def canonical_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


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
    if a <= NORM_EPS / b:
        return None
    u = v / b
    if not np.all(np.isfinite(u)):
        return None
    return u


def _as_int_k(k) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise InvalidKError("K must be an integer 1–20 (bool is not allowed)")
    k = int(k)
    if k < K_MIN or k > K_MAX:
        raise InvalidKError(f"K must be an integer {K_MIN}–{K_MAX}")
    return k


@dataclass
class NeighborEvidence:
    record_id: str
    cosine: float
    input_index: int | None = None


@dataclass
class AnalysisItem:
    record_id: str
    input_index: int
    label: str | None
    same: float | None
    other: float | None
    margin: float | None
    rival_class: str | None
    same_k: int
    rival_k: int
    same_neighbors: list[NeighborEvidence]
    rival_neighbors: list[NeighborEvidence]
    tied_rival_count: int
    tied_rival_classes: list[str]
    status: str
    reasons: list[str]


@dataclass
class AnalysisResult:
    items: list[AnalysisItem]
    k: int
    query_block: int
    reference_block: int
    max_block_shape: tuple[int, int]
    block_shapes: list[tuple[int, int]] = field(default_factory=list)
    input_count: int = 0
    valid_count: int = 0
    excluded_count: int = 0
    per_class_counts: dict[str, int] = field(default_factory=dict)
    method_version: str = METHOD_VERSION
    schema_version: int = SCHEMA_VERSION


class _RevStr:
    __slots__ = ("s",)

    def __init__(self, s: str):
        self.s = s

    def __lt__(self, other: "_RevStr") -> bool:
        return self.s > other.s

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _RevStr) and self.s == other.s


def _push_topk(heap: list, sim: float, record_id: str, index: int, k: int) -> None:
    item = (sim, _RevStr(record_id), record_id, index)
    if len(heap) < k:
        heapq.heappush(heap, item)
        return
    worst_sim, _, worst_id, _ = heap[0]
    if sim > worst_sim or (sim == worst_sim and record_id < worst_id):
        heapq.heapreplace(heap, item)


def _heap_neighbors(heap: list) -> tuple[float | None, int, list[NeighborEvidence]]:
    cands = sorted(((sim, rid, idx) for sim, _, rid, idx in heap),
                   key=lambda t: (-t[0], t[1]))
    if not cands:
        return None, 0, []
    k_eff = len(cands)
    mean = float(sum(s for s, _, _ in cands) / k_eff)
    nbs = [NeighborEvidence(record_id=rid, cosine=float(sim), input_index=int(idx))
           for sim, rid, idx in cands]
    return mean, k_eff, nbs


def analyze(
    embeddings,
    labels,
    record_ids,
    *,
    eligibility,
    invalid_reasons,
    k: int = DEFAULT_K,
    query_block: int = QUERY_BLOCK,
    reference_block: int = REFERENCE_BLOCK,
    progress: Callable[[int, int], None] | None = None,
) -> AnalysisResult:
    k = _as_int_k(k)
    if query_block < 1 or reference_block < 1:
        raise InputShapeError("block sizes must be >= 1")
    E = np.asarray(embeddings, dtype=np.float64)
    if E.ndim != 2:
        raise InputShapeError(f"embeddings must be 2D (N,D); got {getattr(E, 'shape', None)}")
    n, d = E.shape
    if d <= 0:
        raise InputShapeError("embedding dimension D must be > 0")
    if n > MAX_ROWS:
        raise TooManyRowsError(
            f"analysis universe has {n} rows; first version cap is {MAX_ROWS} "
            "(narrow the analysis split; rows are not truncated)")
    if not (len(labels) == len(record_ids) == len(eligibility) == len(invalid_reasons) == n):
        raise InputShapeError("labels, record_ids, eligibility and invalid_reasons must match N")
    if n == 0:
        return AnalysisResult(items=[], k=k, query_block=query_block,
                              reference_block=reference_block, max_block_shape=(0, 0))

    units = np.zeros((n, d), dtype=np.float64)
    can_use = np.zeros(n, dtype=bool)
    items: list[AnalysisItem] = []
    for i in range(n):
        reasons = [str(r) for r in (invalid_reasons[i] or [])]
        adapter_ok = bool(eligibility[i])
        if not adapter_ok and not reasons:
            reasons.append("ineligible")
        u = None
        usable = adapter_ok
        if usable:
            u = scale_safe_normalize(E[i])
            if u is None:
                usable = False
                raw = np.asarray(E[i], dtype=np.float64)
                if not np.all(np.isfinite(raw)):
                    reasons.append("non_finite_vector")
                else:
                    reasons.append("vector_too_small")
        lab = labels[i]
        if lab is not None:
            lab = str(lab)
        if usable and (lab is None or lab.strip() == ""):
            usable = False
            reasons.append("missing_label")
        if usable:
            units[i] = u
            can_use[i] = True
        status = "ok" if usable else ("invalid" if not adapter_ok else "unscorable")
        items.append(AnalysisItem(
            record_id=str(record_ids[i]), input_index=i, label=lab if lab != "" else None,
            same=None, other=None, margin=None, rival_class=None,
            same_k=0, rival_k=0, same_neighbors=[], rival_neighbors=[],
            tied_rival_count=0, tied_rival_classes=[],
            status=status, reasons=reasons,
        ))

    query_indices = [i for i in range(n) if can_use[i]]
    ref_indices = query_indices
    block_shapes: list[tuple[int, int]] = []
    max_shape = (0, 0)

    if query_indices:
        # heaps[local_q][class] = heap
        from collections import defaultdict
        heaps: list[dict[str, list]] = [defaultdict(list) for _ in query_indices]
        qmap = {qi: loc for loc, qi in enumerate(query_indices)}
        nq = len(query_indices)
        for qs in range(0, nq, query_block):
            q_block = query_indices[qs:qs + query_block]
            Q = units[q_block]
            for rs in range(0, len(ref_indices), reference_block):
                r_block = ref_indices[rs:rs + reference_block]
                R = units[r_block]
                sims = np.clip(Q @ R.T, -1.0, 1.0)
                shape = (int(sims.shape[0]), int(sims.shape[1]))
                block_shapes.append(shape)
                if shape[0] * shape[1] >= max_shape[0] * max_shape[1]:
                    max_shape = shape if shape[0] > max_shape[0] or shape[1] > max_shape[1] else max_shape
                max_shape = (max(max_shape[0], shape[0]), max(max_shape[1], shape[1]))
                if shape[0] > query_block or shape[1] > reference_block:
                    raise ClassSimilarityError(
                        f"pairing block {shape} exceeds ({query_block}, {reference_block})")
                for loc_q, qi in enumerate(q_block):
                    qid = items[qi].record_id
                    row = sims[loc_q]
                    for loc_r, rj in enumerate(r_block):
                        if record_ids[rj] == qid:
                            continue
                        lab = items[rj].label
                        if lab is None:
                            continue
                        _push_topk(heaps[qmap[qi]][lab], float(row[loc_r]),
                                   items[rj].record_id, rj, k)
            if progress is not None:
                progress(min(qs + query_block, nq), nq)

        for loc, qi in enumerate(query_indices):
            item = items[qi]
            own = item.label
            class_scores: dict[str, tuple[float, int, list[NeighborEvidence]]] = {}
            for cls, heap in heaps[loc].items():
                mean, k_eff, nbs = _heap_neighbors(heap)
                if mean is not None:
                    class_scores[cls] = (mean, k_eff, nbs)
            if own in class_scores:
                mean, k_eff, nbs = class_scores[own]
                item.same, item.same_k, item.same_neighbors = mean, k_eff, nbs
            else:
                item.reasons.append("insufficient_same")
            others = [(c, *class_scores[c]) for c in class_scores if c != own]
            if not others:
                item.reasons.append("no_other_class")
            else:
                best = max(t[1] for t in others)
                tied = sorted(c for c, s, _, _ in others if s == best)
                rival = tied[0]
                mean, k_eff, nbs = class_scores[rival]
                item.other = mean
                item.rival_k = k_eff
                item.rival_neighbors = nbs
                item.rival_class = rival
                item.tied_rival_count = len(tied)
                item.tied_rival_classes = tied[:TIED_RIVAL_LIMIT]
                if len(tied) > 1:
                    item.reasons.append("tied_rivals")
            if item.same is not None and item.other is not None:
                item.margin = float(item.same - item.other)

    per_class: dict[str, int] = {}
    for it in items:
        if it.label is not None and can_use[it.input_index]:
            per_class[it.label] = per_class.get(it.label, 0) + 1
    valid_count = int(can_use.sum())
    return AnalysisResult(
        items=items, k=k, query_block=query_block, reference_block=reference_block,
        max_block_shape=max_shape, block_shapes=block_shapes,
        input_count=n, valid_count=valid_count, excluded_count=n - valid_count,
        per_class_counts=per_class,
    )


def box_stats(values: Iterable) -> dict[str, float | int] | None:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                     dtype=np.float64)
    if arr.size == 0:
        return None
    q1, med, q3 = np.percentile(arr, [25, 50, 75], method="linear")
    iqr = float(q3 - q1)
    lo, hi = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
    inside = arr[(arr >= lo) & (arr <= hi)]
    return {
        "q1": float(q1), "median": float(med), "q3": float(q3), "iqr": iqr,
        "whisker_low": float(inside.min()) if inside.size else float(arr.min()),
        "whisker_high": float(inside.max()) if inside.size else float(arr.max()),
        "n": int(arr.size),
    }


def build_box_summary(items: Sequence[AnalysisItem], metric: str) -> list[dict]:
    if metric not in {"same", "other", "margin"}:
        raise InputShapeError(f"metric must be same|other|margin, got {metric}")
    groups: dict[str, list] = {}
    totals: dict[str, int] = {}
    unscored: dict[str, int] = {}
    for it in items:
        lab = it.label if it.label is not None else "未分類／標籤無效"
        totals[lab] = totals.get(lab, 0) + 1
        val = getattr(it, metric)
        if val is None or not np.isfinite(val):
            unscored[lab] = unscored.get(lab, 0) + 1
            continue
        groups.setdefault(lab, []).append(float(val))
    out = []
    for lab in sorted(totals):
        stats = box_stats(groups.get(lab, []))
        row = {
            "label": lab,
            "n_total": totals[lab],
            "n_scored": 0 if stats is None else stats["n"],
            "n_unscored": unscored.get(lab, 0),
            "metric": metric,
        }
        if stats:
            row.update(stats)
        else:
            row.update({"q1": None, "median": None, "q3": None, "iqr": None,
                        "whisker_low": None, "whisker_high": None, "n": 0})
        out.append(row)
    return out
