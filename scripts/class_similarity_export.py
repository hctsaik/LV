"""CSV / JSON evidence bytes for class-similarity reports. No dataset writes."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from class_similarity import (
    METHOD_VERSION,
    SCHEMA_VERSION,
    AnalysisItem,
    AnalysisResult,
    ClassSimilarityError,
)
from class_similarity_records import canonical_path
from class_similarity_threshold import (
    CandidateSummary,
    EvaluationCounts,
    ThresholdPolicy,
    candidate_mask,
    parse_policy,
    summarize_candidates,
)


class ExportError(ClassSimilarityError):
    pass


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_float(v):
    if v is None:
        return None
    x = float(v)
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return x


def _bbox_json(bbox) -> str | None:
    if bbox is None:
        return None
    return json.dumps(list(bbox), ensure_ascii=False, separators=(",", ":"))


def _record_index_entry(record: dict, record_id: str, original_index: int,
                        level: str) -> dict:
    path = record.get("path")
    image_path = record.get("image_path")
    entry = {
        "record_id": record_id,
        "original_index": original_index,
        "level": level,
        "path": str(path) if path is not None else None,
        "image_path": str(image_path) if image_path is not None else None,
        "bbox": list(record["bbox"]) if record.get("bbox") is not None else None,
        "obj_index": record.get("obj_index"),
        "label": record.get("label"),
        "split": record.get("split"),
        "canonical_path": canonical_path(path) if path is not None else None,
    }
    if level == "object" and image_path is not None:
        entry["canonical_image_path"] = canonical_path(image_path)
    return entry


def _item_payload(it: AnalysisItem) -> dict:
    return {
        "record_id": it.record_id,
        "same": _json_float(it.same),
        "other": _json_float(it.other),
        "rival_class": it.rival_class,
        "margin": _json_float(it.margin),
        "same_k": it.same_k,
        "rival_k": it.rival_k,
        "status": it.status,
        "tied_rival_count": it.tied_rival_count,
        "tied_rival_classes": list(it.tied_rival_classes),
        "same_neighbors": [{"record_id": n.record_id, "cosine": _json_float(n.cosine)}
                           for n in it.same_neighbors],
        "rival_neighbors": [{"record_id": n.record_id, "cosine": _json_float(n.cosine)}
                            for n in it.rival_neighbors],
        "reasons": list(it.reasons),
    }


def _score_index_row(it: AnalysisItem, label_status: str | None = None) -> dict:
    return {
        "record_id": it.record_id,
        "label": it.label,
        "label_status": label_status,
        "status": it.status,
        "reasons": list(it.reasons),
        "margin": _json_float(it.margin),
    }


def _eval_payload(evaluation: dict | None, counts: EvaluationCounts | None) -> dict | None:
    if not evaluation and not counts:
        return {
            "evaluation_id": None, "evaluation_version": None, "purpose": None,
            "truth_digest": None, "frozen_ids": [], "scored_ids": [],
            "coverage": None, "status": "not_configured",
            "tp": None, "fp": None, "fn": None, "tn": None,
            "precision": None, "recall": None,
            "unavailable_reason": "not_configured",
        }
    evaluation = evaluation or {}
    counts = counts
    payload = {
        "evaluation_id": evaluation.get("evaluation_id"),
        "evaluation_version": evaluation.get("evaluation_version"),
        "purpose": evaluation.get("purpose"),
        "truth_digest": evaluation.get("truth_digest"),
        "frozen_ids": list(evaluation.get("frozen_ids") or []),
        "scored_ids": [],
        "coverage": None, "status": "not_configured",
        "tp": None, "fp": None, "fn": None, "tn": None,
        "precision": None, "recall": None,
        "unavailable_reason": "not_configured",
    }
    if counts:
        payload.update({
            "coverage": counts.coverage, "status": counts.status,
            "tp": counts.tp, "fp": counts.fp, "fn": counts.fn, "tn": counts.tn,
            "precision": counts.precision, "recall": counts.recall,
            "unavailable_reason": counts.unavailable_reason,
            "e0_n": counts.e0_n, "e_n": counts.e_n,
        })
        if counts.confusion_ids:
            payload["scored_ids"] = (
                counts.confusion_ids.get("tp", []) + counts.confusion_ids.get("fp", [])
                + counts.confusion_ids.get("fn", []) + counts.confusion_ids.get("tn", [])
            )
    return payload


def export_report(
    result: AnalysisResult,
    records: Sequence[dict],
    *,
    scope: str,
    local_selected_ids: Sequence[str] | None = None,
    committed_preview=None,
    evaluation: dict | None = None,
    evaluation_counts: EvaluationCounts | None = None,
    prepared=None,
    level: str = "image",
    analysis_key: str | None = None,
    preview_key: str | None = None,
    visible_class_filter: Sequence[str] | None = None,
    frozen_policy: dict | None = None,
    created_at: str | None = None,
    config: dict | None = None,
    label_status: Sequence[str] | None = None,
) -> tuple[bytes, bytes]:
    if scope not in {"all", "candidates", "selected"}:
        raise ExportError(f"scope must be all|candidates|selected, got {scope}")
    items = result.items
    if len(records) != len(items):
        raise ExportError("records and result.items must be aligned")
    policy = None
    mask = None
    if committed_preview is not None:
        policy = parse_policy(committed_preview if not isinstance(committed_preview, dict)
                              or "mode" in committed_preview
                              else committed_preview.get("policy"))
        mask = candidate_mask(items, policy)
    elif scope == "candidates":
        raise ExportError("candidates scope requires a committed preview")

    selected = list(local_selected_ids or [])
    selected_set = set(selected)
    query_indices = list(range(len(items)))
    if scope == "candidates":
        query_indices = [i for i, flag in enumerate(mask) if flag and not str(items[i].record_id).startswith("invalid:")]
    elif scope == "selected":
        query_indices = [i for i, it in enumerate(items)
                         if it.record_id in selected_set
                         and not str(it.record_id).startswith("invalid:")]

    # neighbor ids needed for records index
    needed = set()
    detail_items = []
    for i in query_indices:
        it = items[i]
        if scope != "all" and it.record_id.startswith("invalid:"):
            continue
        if scope == "all" or not it.record_id.startswith("invalid:"):
            detail_items.append(it)
            needed.add(it.record_id)
            for n in it.same_neighbors + it.rival_neighbors:
                needed.add(n.record_id)
    if scope == "all":
        detail_items = [_item_payload(it) for it in items]
        needed = {it.record_id for it in items}
        for it in items:
            for n in it.same_neighbors + it.rival_neighbors:
                needed.add(n.record_id)
    else:
        # invalid detailed items only in all
        detail_items = [_item_payload(it) for it in items
                        if it.record_id in {items[i].record_id for i in query_indices}
                        and not it.record_id.startswith("invalid:")]
        needed = set()
        for payload in detail_items:
            needed.add(payload["record_id"])
            for n in payload["same_neighbors"] + payload["rival_neighbors"]:
                needed.add(n["record_id"])

    id_to_row = {it.record_id: (i, it) for i, it in enumerate(items)}
    orig_index = list(getattr(prepared, "original_index", range(len(items))))
    records_index = []
    for rid in sorted(needed):
        if rid not in id_to_row:
            continue
        i, it = id_to_row[rid]
        rec = records[i]
        records_index.append(_record_index_entry(
            rec, rid, orig_index[i] if i < len(orig_index) else i, level))

    summary = {
        "input_count": result.input_count,
        "valid_count": result.valid_count,
        "excluded_count": result.excluded_count,
        "per_class_counts": result.per_class_counts,
    }
    cand_summary = None
    candidate_ids = []
    if mask is not None:
        summ = summarize_candidates(items, mask)
        cand_summary = {
            "eligible_n": summ.eligible_n, "candidate_n": summ.candidate_n,
            "unscorable_n": summ.unscorable_n, "ratio": summ.ratio,
            "per_class": summ.per_class,
        }
        candidate_ids = [items[i].record_id for i, flag in enumerate(mask)
                         if flag and not items[i].record_id.startswith("invalid:")]

    statuses = list(label_status) if label_status is not None else (
        list(getattr(prepared, "label_status", [None] * len(items))))
    score_index = [_score_index_row(it, statuses[i] if i < len(statuses) else None)
                   for i, it in enumerate(items)]
    evaluation_items = []
    if evaluation:
        rev = {r["record_id"]: r for r in evaluation.get("reviews") or []}
        for rid in evaluation.get("frozen_ids") or []:
            r = rev.get(rid) or {}
            it = id_to_row.get(rid)
            evaluation_items.append({
                "record_id": rid,
                "margin": _json_float(it[1].margin) if it else None,
                "original_label": r.get("original_label"),
                "verdict": r.get("verdict") or "unreviewed",
                "review_source": r.get("review_source"),
                "review_version": r.get("review_version"),
            })

    exclusions = []
    for it in items:
        if it.status != "ok":
            exclusions.append({
                "record_id": it.record_id,
                "original_index": orig_index[it.input_index] if it.input_index < len(orig_index) else it.input_index,
                "reason": list(it.reasons),
            })

    key = analysis_key or getattr(prepared, "analysis_key", None)
    pol_obj = policy.__dict__ if isinstance(policy, ThresholdPolicy) else (
        parse_policy(policy).__dict__ if policy is not None else {
            "mode": None, "threshold": None, "comparator": "lt", "score": "margin",
        })
    if policy is None:
        pol_payload = {"mode": None, "threshold": None, "comparator": "lt", "score": "margin"}
    else:
        pol_payload = {
            "mode": pol_obj["mode"], "threshold": pol_obj["threshold"],
            "comparator": "lt", "score": "margin",
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "method_version": METHOD_VERSION,
        "analysis_key": key,
        "created_at_utc": created_at or _iso_now(),
        "scope": scope,
        "config": config or {
            "K": result.k, "analysis_split": getattr(prepared, "analysis_split", "all"),
            "model": getattr(prepared, "model", None),
            "embedding_recipe": getattr(prepared, "embedding_recipe", None),
            "exact": True,
        },
        "summary": summary,
        "records": records_index,
        "items": detail_items,
        "exclusions": exclusions,
        "policy": pol_payload,
        "preview_key": preview_key,
        "candidate_ids": candidate_ids,
        "selected_ids": selected,
        "visible_class_filter": list(visible_class_filter) if visible_class_filter else [],
        "candidate_summary": cand_summary,
        "evaluation": _eval_payload(evaluation, evaluation_counts),
        "score_index": score_index,
        "evaluation_items": evaluation_items,
        "frozen_policy": frozen_policy,
        "note": "含全批 ID／分數與評估索引，不附全批影像",
    }
    json_bytes = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")

    # CSV of query rows in this scope
    buf = io.StringIO()
    fields = [
        "analysis_key", "record_id", "original_index", "level", "path", "image_path",
        "bbox_json", "obj_index", "label", "split", "same", "other", "rival_class",
        "margin", "same_k", "rival_k", "status", "policy_mode", "threshold",
        "comparator", "is_candidate",
    ]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    csv_indices = list(range(len(items))) if scope == "all" else query_indices
    if scope == "all":
        csv_indices = list(range(len(items)))
    for i in csv_indices:
        it = items[i]
        if scope != "all" and it.record_id.startswith("invalid:"):
            continue
        rec = records[i]
        is_cand = "" if mask is None else ("1" if bool(mask[i]) else "0")
        writer.writerow({
            "analysis_key": key or "",
            "record_id": it.record_id,
            "original_index": orig_index[i] if i < len(orig_index) else i,
            "level": level,
            "path": "" if rec.get("path") is None else str(rec.get("path")),
            "image_path": "" if rec.get("image_path") is None else str(rec.get("image_path")),
            "bbox_json": _bbox_json(rec.get("bbox")) or "",
            "obj_index": "" if rec.get("obj_index") is None else rec.get("obj_index"),
            "label": it.label or "",
            "split": rec.get("split") or "",
            "same": "" if it.same is None else repr(float(it.same)),
            "other": "" if it.other is None else repr(float(it.other)),
            "rival_class": it.rival_class or "",
            "margin": "" if it.margin is None else repr(float(it.margin)),
            "same_k": it.same_k,
            "rival_k": it.rival_k,
            "status": it.status,
            "policy_mode": pol_payload["mode"] or "",
            "threshold": "" if pol_payload["threshold"] is None else repr(pol_payload["threshold"]),
            "comparator": "lt",
            "is_candidate": is_cand,
        })
    csv_bytes = buf.getvalue().encode("utf-8-sig")
    return csv_bytes, json_bytes


def evaluation_template(*, frozen_ids: Sequence[str], records: Sequence[dict],
                        record_ids: Sequence[str], labels: Sequence[str | None],
                        snapshot_key: str, purpose: str = "tuning",
                        selection_method: str, evaluation_id: str,
                        evaluation_version: int = 1, level: str = "image") -> bytes:
    catalog = []
    id_to_row = {rid: i for i, rid in enumerate(record_ids)}
    for rid in frozen_ids:
        i = id_to_row.get(rid)
        rec = records[i] if i is not None else {}
        catalog.append({
            "record_id": rid,
            "path": None if rec.get("path") is None else str(rec.get("path")),
            "image_path": None if rec.get("image_path") is None else str(rec.get("image_path")),
            "bbox": list(rec["bbox"]) if rec.get("bbox") is not None else None,
            "obj_index": rec.get("obj_index"),
            "original_label": labels[i] if i is not None else None,
            "source_group_id": str(rec.get("image_path") or rec.get("path") or ""),
        })
    reviews = [{
        "record_id": rid,
        "original_label": next((c["original_label"] for c in catalog if c["record_id"] == rid), None),
        "verdict": "unreviewed",
        "review_source": None,
        "review_version": None,
    } for rid in frozen_ids]
    doc = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "evaluation_version": evaluation_version,
        "purpose": purpose,
        "snapshot_key": snapshot_key,
        "selection_method": selection_method,
        "frozen_ids": list(frozen_ids),
        "reviews": reviews,
        "record_catalog": catalog,
    }
    return json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8")


def rebuild_counts_from_score_index(score_index: Sequence[dict], policy) -> dict:
    """Independent recount from exported score_index (no items required)."""
    from types import SimpleNamespace
    fake_items = []
    for row in score_index:
        fake_items.append(SimpleNamespace(
            margin=row.get("margin"), record_id=row.get("record_id"),
            label=row.get("label"), status=row.get("status"), reasons=row.get("reasons") or [],
        ))
    mask = candidate_mask(fake_items, policy)  # uses _finite_margin on .margin
    from class_similarity_threshold import summarize_candidates
    summ = summarize_candidates(fake_items, mask)
    return {
        "eligible_n": summ.eligible_n,
        "candidate_n": summ.candidate_n,
        "unscorable_n": summ.unscorable_n,
        "per_class": summ.per_class,
        "input_count": summ.input_count,
    }
