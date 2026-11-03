"""Margin threshold preview, evaluation manifests, PR and curves."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from class_similarity import AnalysisItem, ClassSimilarityError, canonical_dumps
from class_similarity_records import sha256_canonical

COMPARATOR = "lt"
SCORE_NAME = "margin"
SCHEMA_EVAL = 1


class EvaluationError(ClassSimilarityError):
    pass


class ActionRejected(ClassSimilarityError):
    pass


@dataclass(frozen=True)
class ThresholdPolicy:
    mode: str = "threshold"  # threshold | none | all_eligible
    threshold: float | None = 0.0
    comparator: str = COMPARATOR
    score: str = SCORE_NAME


@dataclass
class CandidateSummary:
    eligible_n: int
    candidate_n: int
    unscorable_n: int
    ratio: float | None
    per_class: list[dict]
    input_count: int


@dataclass
class EvaluationCounts:
    status: str
    unavailable_reason: str | None
    coverage: float | None
    e0_n: int
    e_n: int
    known_incorrect_n: int
    unknown_n: int
    unscorable_in_e0: int
    tp: int | None
    fp: int | None
    fn: int | None
    tn: int | None
    precision: float | None
    recall: float | None
    confusion_ids: dict[str, list[str]] = field(default_factory=dict)


def _finite_margin(item: AnalysisItem) -> bool:
    return item.margin is not None and np.isfinite(item.margin)


def parse_policy(policy) -> ThresholdPolicy:
    if isinstance(policy, ThresholdPolicy):
        pol = policy
    else:
        policy = policy or {}
        pol = ThresholdPolicy(
            mode=str(policy.get("mode") or "threshold"),
            threshold=policy.get("threshold", 0.0),
            comparator=str(policy.get("comparator") or COMPARATOR),
            score=str(policy.get("score") or SCORE_NAME),
        )
    if pol.mode not in {"threshold", "none", "all_eligible"}:
        raise ClassSimilarityError(f"unknown policy mode {pol.mode}")
    if pol.comparator != COMPARATOR:
        raise ClassSimilarityError("comparator must be lt")
    if pol.score != SCORE_NAME:
        raise ClassSimilarityError("score must be margin")
    if pol.mode == "threshold":
        if pol.threshold is None or not np.isfinite(float(pol.threshold)):
            raise ClassSimilarityError("threshold must be a finite number")
        t = float(pol.threshold)
        if t < -2.0 or t > 2.0:
            raise ClassSimilarityError("threshold must be in [-2, 2]")
        pol = ThresholdPolicy(mode="threshold", threshold=t,
                              comparator=COMPARATOR, score=SCORE_NAME)
    else:
        pol = ThresholdPolicy(mode=pol.mode, threshold=None,
                              comparator=COMPARATOR, score=SCORE_NAME)
    return pol


def candidate_mask(items: Sequence[AnalysisItem], policy) -> np.ndarray:
    pol = parse_policy(policy)
    n = len(items)
    mask = np.zeros(n, dtype=bool)
    if pol.mode == "none":
        return mask
    for i, it in enumerate(items):
        if not _finite_margin(it):
            continue
        if pol.mode == "all_eligible":
            mask[i] = True
        else:
            mask[i] = bool(it.margin < pol.threshold)
    return mask


def summarize_candidates(items: Sequence[AnalysisItem], mask) -> CandidateSummary:
    mask = np.asarray(mask, dtype=bool)
    eligible = [i for i, it in enumerate(items) if _finite_margin(it)]
    unscorable = [i for i, it in enumerate(items) if not _finite_margin(it)]
    cand = [i for i in eligible if bool(mask[i])]
    classes = sorted({it.label for it in items if it.label is not None})
    per = []
    classified = 0
    for lab in classes:
        idxs = [i for i, it in enumerate(items) if it.label == lab]
        el = [i for i in idxs if _finite_margin(items[i])]
        ca = [i for i in el if bool(mask[i])]
        un = len(idxs) - len(el)
        ratio = (len(ca) / len(el)) if el else None
        per.append({
            "label": lab, "n_total": len(idxs), "eligible_n": len(el),
            "candidate_n": len(ca), "unscorable_n": un, "ratio": ratio,
        })
        classified += len(idxs)
    if classified < len(items):
        idxs = [i for i, it in enumerate(items) if it.label is None]
        el = [i for i in idxs if _finite_margin(items[i])]
        ca = [i for i in el if bool(mask[i])]
        per.append({
            "label": "未分類／標籤無效", "n_total": len(idxs), "eligible_n": len(el),
            "candidate_n": len(ca), "unscorable_n": len(idxs) - len(el),
            "ratio": (len(ca) / len(el)) if el else None,
        })
    eligible_n = len(eligible)
    cand_n = len(cand)
    return CandidateSummary(
        eligible_n=eligible_n, candidate_n=cand_n, unscorable_n=len(unscorable),
        ratio=(cand_n / eligible_n) if eligible_n else None,
        per_class=per, input_count=len(items),
    )


def workload_curve(items: Sequence[AnalysisItem]) -> list[dict]:
    values = sorted({float(it.margin) for it in items if _finite_margin(it)})
    points = [{"t": None, "mode": "none", "candidate_n": 0,
               "label": "none"}]
    eligible_n = sum(1 for it in items if _finite_margin(it))
    for v in values:
        t = math.nextafter(v, math.inf)
        mask = candidate_mask(items, {"mode": "threshold", "threshold": t})
        label = "全部可計分" if v == 2.0 else None
        points.append({
            "t": t, "mode": "threshold", "source_margin": v,
            "candidate_n": int(mask.sum()), "label": label,
        })
    points.append({
        "t": None, "mode": "all_eligible", "candidate_n": eligible_n,
        "label": "all_eligible",
    })
    return points


def preview_key(analysis_key: str, policy, evaluation_key: str | None) -> str:
    pol = parse_policy(policy)
    return sha256_canonical({
        "analysis_key": analysis_key,
        "mode": pol.mode,
        "threshold": pol.threshold,
        "comparator": pol.comparator,
        "evaluation_key": evaluation_key,
    })


def truth_digest(snapshot_key: str, frozen_ids: Sequence[str], reviews: Sequence[dict]) -> str:
    by_id = {r["record_id"]: r for r in reviews}
    rows = []
    for rid in sorted(frozen_ids):
        r = by_id.get(rid) or {}
        rows.append({
            "record_id": rid,
            "original_label": r.get("original_label"),
            "verdict": r.get("verdict") or "unreviewed",
            "review_source": r.get("review_source"),
            "review_version": r.get("review_version"),
        })
    return sha256_canonical({
        "snapshot_key": snapshot_key,
        "frozen_ids": list(sorted(frozen_ids)),
        "reviews": rows,
    })


def evaluation_key(analysis_key: str, digest: str, purpose: str,
                   frozen_policy_id: str | None) -> str:
    return sha256_canonical({
        "analysis_key": analysis_key,
        "truth_digest": digest,
        "purpose": purpose,
        "frozen_policy_id": frozen_policy_id,
    })


_ALLOWED_VERDICTS = {"incorrect", "correct", "unresolved", "unreviewed"}
_ALLOWED_PURPOSE = {"tuning", "heldout"}


def validate_evaluation_manifest(manifest: dict, snapshot: dict,
                                 items: Sequence[AnalysisItem]) -> dict:
    if not isinstance(manifest, dict):
        raise EvaluationError("manifest must be an object")
    errors = []
    if manifest.get("schema_version") != SCHEMA_EVAL:
        errors.append("schema_version must be 1")
    purpose = manifest.get("purpose")
    if purpose not in _ALLOWED_PURPOSE:
        errors.append("purpose must be tuning or heldout")
    frozen = manifest.get("frozen_ids")
    if frozen is None:
        frozen = []
    if not isinstance(frozen, list) or any(not isinstance(x, str) or x == "" for x in frozen):
        errors.append("frozen_ids must be a list of non-empty strings")
        frozen = []
    if len(frozen) != len(set(frozen)):
        errors.append("frozen_ids must not contain duplicates")
    if isinstance(manifest.get("evaluation_version"), bool):
        errors.append("evaluation_version must not be bool")
    reviews = manifest.get("reviews") or []
    if not isinstance(reviews, list):
        errors.append("reviews must be a list")
        reviews = []
    by_item = {it.record_id: it for it in items}
    snap_key = snapshot.get("snapshot_key")
    if manifest.get("snapshot_key") not in (None, snap_key) and snap_key is not None:
        errors.append("snapshot_key mismatch")
    labels = snapshot.get("original_labels") or {}
    seen_review = set()
    cleaned_reviews = []
    for r in reviews:
        if not isinstance(r, dict):
            errors.append("review is not an object")
            continue
        rid = r.get("record_id")
        if rid in seen_review:
            errors.append(f"duplicate review {rid}")
        seen_review.add(rid)
        if rid not in frozen:
            errors.append(f"review {rid} is not in frozen_ids")
        if rid not in by_item:
            errors.append(f"review {rid} is not in analysis universe")
        verdict = r.get("verdict")
        if verdict not in _ALLOWED_VERDICTS:
            errors.append(f"illegal verdict for {rid}")
        expected_label = labels.get(rid)
        if expected_label is not None and r.get("original_label") not in (None, expected_label):
            if r.get("original_label") != expected_label:
                errors.append(f"original_label mismatch for {rid}")
        cleaned_reviews.append(r)
    if errors:
        raise EvaluationError("evaluation manifest rejected: " + "; ".join(errors))
    # fill unreviewed
    have = {r["record_id"] for r in cleaned_reviews}
    for rid in frozen:
        if rid not in have:
            item = by_item.get(rid)
            cleaned_reviews.append({
                "record_id": rid,
                "original_label": labels.get(rid) or (item.label if item else None),
                "verdict": "unreviewed",
                "review_source": None,
                "review_version": None,
            })
    digest = truth_digest(manifest.get("snapshot_key") or snap_key or "", frozen, cleaned_reviews)
    return {
        "schema_version": SCHEMA_EVAL,
        "evaluation_id": manifest.get("evaluation_id"),
        "evaluation_version": manifest.get("evaluation_version"),
        "purpose": purpose,
        "snapshot_key": manifest.get("snapshot_key") or snap_key,
        "selection_method": manifest.get("selection_method"),
        "frozen_ids": list(frozen),
        "reviews": cleaned_reviews,
        "record_catalog": manifest.get("record_catalog") or [],
        "truth_digest": digest,
        "status": "imported",
    }


def evaluate_threshold(items: Sequence[AnalysisItem], mask,
                       evaluation: dict | None,
                       *, source_verified: bool = True) -> EvaluationCounts:
    if not evaluation:
        return EvaluationCounts(
            status="not_configured", unavailable_reason="no_evaluation",
            coverage=None, e0_n=0, e_n=0, known_incorrect_n=0, unknown_n=0,
            unscorable_in_e0=0, tp=None, fp=None, fn=None, tn=None,
            precision=None, recall=None,
        )
    if not source_verified:
        return EvaluationCounts(
            status="source_unverified", unavailable_reason="source_unverified",
            coverage=None, e0_n=len(evaluation.get("frozen_ids") or []),
            e_n=0, known_incorrect_n=0, unknown_n=0, unscorable_in_e0=0,
            tp=None, fp=None, fn=None, tn=None, precision=None, recall=None,
        )
    frozen = list(evaluation.get("frozen_ids") or [])
    if not frozen:
        return EvaluationCounts(
            status="empty_e0", unavailable_reason="empty_e0",
            coverage=None, e0_n=0, e_n=0, known_incorrect_n=0, unknown_n=0,
            unscorable_in_e0=0, tp=None, fp=None, fn=None, tn=None,
            precision=None, recall=None,
        )
    mask = np.asarray(mask, dtype=bool)
    by_id = {it.record_id: (i, it) for i, it in enumerate(items)}
    reviews = {r["record_id"]: r for r in evaluation.get("reviews") or []}
    e0 = []
    missing = []
    for rid in frozen:
        if rid not in by_id:
            missing.append(rid)
            continue
        e0.append(rid)
    if missing:
        return EvaluationCounts(
            status="unavailable", unavailable_reason="ids_not_in_universe",
            coverage=None, e0_n=len(frozen), e_n=0, known_incorrect_n=0,
            unknown_n=len(frozen), unscorable_in_e0=0,
            tp=None, fp=None, fn=None, tn=None, precision=None, recall=None,
        )
    eligible_e = []
    unscorable = []
    unknown = 0
    known_incorrect = 0
    for rid in e0:
        idx, it = by_id[rid]
        verdict = (reviews.get(rid) or {}).get("verdict") or "unreviewed"
        if verdict == "incorrect":
            known_incorrect += 1
        if not _finite_margin(it):
            unscorable.append(rid)
            continue
        if verdict in {"unreviewed", "unresolved"}:
            unknown += 1
        eligible_e.append((rid, idx, verdict, bool(mask[idx])))
    e_n = len(eligible_e)
    coverage = (e_n / len(e0)) if e0 else None
    binary = all(v in {"correct", "incorrect"} for _, _, v, _ in eligible_e)
    if e_n == 0 or not binary:
        return EvaluationCounts(
            status="incomplete",
            unavailable_reason="incomplete_binary_verdicts" if e_n else "e_empty",
            coverage=coverage, e0_n=len(e0), e_n=e_n,
            known_incorrect_n=known_incorrect, unknown_n=unknown,
            unscorable_in_e0=len(unscorable),
            tp=None, fp=None, fn=None, tn=None, precision=None, recall=None,
        )
    tp_ids, fp_ids, fn_ids, tn_ids = [], [], [], []
    for rid, idx, verdict, flag in eligible_e:
        if verdict == "incorrect" and flag:
            tp_ids.append(rid)
        elif verdict == "correct" and flag:
            fp_ids.append(rid)
        elif verdict == "incorrect" and not flag:
            fn_ids.append(rid)
        else:
            tn_ids.append(rid)
    tp, fp, fn, tn = map(len, (tp_ids, fp_ids, fn_ids, tn_ids))
    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    return EvaluationCounts(
        status="ready", unavailable_reason=None, coverage=coverage,
        e0_n=len(e0), e_n=e_n, known_incorrect_n=known_incorrect,
        unknown_n=unknown, unscorable_in_e0=len(unscorable),
        tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
        confusion_ids={"tp": tp_ids, "fp": fp_ids, "fn": fn_ids, "tn": tn_ids},
    )


def precision_recall_curve(items: Sequence[AnalysisItem], evaluation: dict,
                           *, source_verified: bool = True) -> list[dict]:
    if not evaluation:
        return []
    values = sorted({float(it.margin) for i, it in enumerate(items)
                     if it.record_id in set(evaluation.get("frozen_ids") or [])
                     and _finite_margin(it)})
    out = []
    policies = [{"mode": "none"}]
    for v in values:
        policies.append({"mode": "threshold", "threshold": math.nextafter(v, math.inf)})
    policies.append({"mode": "all_eligible"})
    for pol in policies:
        mask = candidate_mask(items, pol)
        ev = evaluate_threshold(items, mask, evaluation, source_verified=source_verified)
        if ev.precision is None or ev.recall is None:
            continue
        out.append({
            "policy": parse_policy(pol).__dict__,
            "precision": ev.precision,
            "recall": ev.recall,
            "tp": ev.tp, "fp": ev.fp, "fn": ev.fn, "tn": ev.tn,
        })
    return out


def current_pr_point(items, evaluation, policy, *, source_verified=True) -> dict:
    mask = candidate_mask(items, policy)
    ev = evaluate_threshold(items, mask, evaluation, source_verified=source_verified)
    return {
        "precision": ev.precision, "recall": ev.recall,
        "tp": ev.tp, "fp": ev.fp, "fn": ev.fn, "tn": ev.tn,
        "status": ev.status, "unavailable_reason": ev.unavailable_reason,
    }


@dataclass
class PreviewState:
    draft_policy: ThresholdPolicy
    committed_policy: ThresholdPolicy | None
    request_seq: int
    committed_seq: int | None
    analysis_key: str
    preview_key_committed: str | None
    evaluation_key: str | None


def apply_preview_event(state: PreviewState, event: dict, *, now_seq: int | None = None,
                        current_analysis_key: str, current_eval_key: str | None,
                        draft_policy: ThresholdPolicy) -> PreviewState:
    """Reject out-of-order / stale events. Commit only matching seq+keys."""
    if event.get("analysis_key") != current_analysis_key:
        raise ActionRejected("analysis_key mismatch")
    seq = event.get("request_seq")
    if seq is None:
        raise ActionRejected("missing request_seq")
    seq = int(seq)
    if seq < state.request_seq:
        raise ActionRejected("stale request_seq")
    kind = event.get("kind") or "preview"
    pol = parse_policy(event.get("policy") or draft_policy)
    new_state = PreviewState(
        draft_policy=pol,
        committed_policy=state.committed_policy,
        request_seq=max(state.request_seq, seq),
        committed_seq=state.committed_seq,
        analysis_key=current_analysis_key,
        preview_key_committed=state.preview_key_committed,
        evaluation_key=current_eval_key,
    )
    if kind == "commit":
        if event.get("preview_key") not in (
                None, preview_key(current_analysis_key, pol, current_eval_key)):
            # allow server to compute
            pass
        new_state.committed_policy = pol
        new_state.committed_seq = seq
        new_state.preview_key_committed = preview_key(
            current_analysis_key, pol, current_eval_key)
    return new_state


def actions_enabled(state: PreviewState) -> bool:
    if state.committed_policy is None or state.committed_seq is None:
        return False
    return (state.draft_policy == state.committed_policy
            and state.request_seq == state.committed_seq)


def validate_action(action: dict, state: PreviewState, current_analysis_key: str) -> None:
    if not actions_enabled(state):
        raise ActionRejected("preview not committed")
    if action.get("analysis_key") != current_analysis_key:
        raise ActionRejected("analysis_key mismatch")
    if action.get("preview_key") != state.preview_key_committed:
        raise ActionRejected("preview_key mismatch")
    if int(action.get("request_seq")) != state.committed_seq:
        raise ActionRejected("request_seq mismatch")


def source_group_id(record: dict, level: str) -> str:
    if level == "object":
        src = record.get("image_path") or record.get("path")
    else:
        src = record.get("path")
    from class_similarity_records import canonical_path
    extra = record.get("source_group_id")
    base = canonical_path(src) if src not in (None, "") else "unknown"
    if extra:
        return f"{base}|{extra}"
    return base


def validate_heldout(frozen_policy: dict, *, analysis_key: str,
                     score_protocol_key: str, heldout_ids: Sequence[str],
                     heldout_groups: Sequence[str],
                     tuning_complete: bool) -> None:
    if not frozen_policy:
        raise EvaluationError("heldout requires a saved operating point")
    if frozen_policy.get("analysis_key") != analysis_key:
        raise EvaluationError("heldout analysis_key mismatch")
    if frozen_policy.get("score_protocol_key") != score_protocol_key:
        raise EvaluationError("heldout score protocol mismatch")
    if not tuning_complete:
        raise EvaluationError("tuning evaluation is incomplete")
    tuning_ids = set(frozen_policy.get("tuning_ids") or [])
    if set(heldout_ids) & tuning_ids:
        raise EvaluationError("heldout IDs overlap tuning IDs")
    tuning_groups = set(frozen_policy.get("tuning_group_ids") or [])
    if set(heldout_groups) & tuning_groups:
        raise EvaluationError("heldout source groups overlap tuning groups")


def score_protocol_key(*, method_version: str, k: int, model: str, weights_id,
                       recipe: dict, analysis_split: str) -> str:
    return sha256_canonical({
        "method_version": method_version, "k": k, "model": model,
        "weights_id": weights_id, "recipe": recipe,
        "comparator": COMPARATOR, "score": SCORE_NAME,
        "analysis_split": analysis_split, "reference": "same_universe",
    })


def make_operating_point(*, policy, analysis_key: str, score_protocol_key: str,
                         tuning_evaluation_key: str | None, tuning_ids: Sequence[str],
                         tuning_group_ids: Sequence[str], policy_id: str,
                         created_at: str) -> dict:
    pol = parse_policy(policy)
    return {
        "policy_id": policy_id,
        "created_at": created_at,
        "policy": {
            "mode": pol.mode, "threshold": pol.threshold,
            "comparator": pol.comparator, "score": pol.score,
        },
        "analysis_key": analysis_key,
        "score_protocol_key": score_protocol_key,
        "tuning_evaluation_key": tuning_evaluation_key,
        "tuning_ids": list(tuning_ids),
        "tuning_group_ids": list(tuning_group_ids),
    }


def axis_display_range(metric: str, mode: str, values: Sequence[float],
                       threshold: float | None) -> dict:
    full = (-2.0, 2.0) if metric == "margin" else (-1.0, 1.0)
    if mode != "data":
        out = {"ymin": full[0], "ymax": full[1], "mode": "full"}
    else:
        arr = [float(v) for v in values if v is not None and np.isfinite(v)]
        if not arr:
            out = {"ymin": full[0], "ymax": full[1], "mode": "data"}
        else:
            lo, hi = min(arr), max(arr)
            pad = max((hi - lo) * 0.05, 1e-4)
            out = {"ymin": lo - pad, "ymax": hi + pad, "mode": "data"}
    warn = False
    if metric == "margin" and threshold is not None and np.isfinite(threshold):
        if threshold < out["ymin"] or threshold > out["ymax"]:
            warn = True
    out["threshold_outside"] = warn
    out["threshold"] = threshold
    return out
