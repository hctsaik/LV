"""Record identity, analysis key, eligibility and selection validation."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from class_similarity import (
    METHOD_VERSION,
    SCHEMA_VERSION,
    AnalysisResult,
    ClassSimilarityError,
    canonical_dumps,
    scale_safe_normalize,
)


class DuplicateIdentityError(ClassSimilarityError):
    pass


class CropIdentityConflictError(ClassSimilarityError):
    pass


class StaleSelectionError(ClassSimilarityError):
    pass


def canonical_path(path) -> str:
    p = Path(path)
    try:
        resolved = p.resolve(strict=False)
    except OSError:
        resolved = p.absolute()
    return os.path.normcase(str(resolved))


def _finite_bbox(bbox) -> list:
    if bbox is None:
        raise ValueError("missing bbox")
    vals = list(bbox)
    if len(vals) != 4:
        raise ValueError("bbox must have 4 numbers")
    out = []
    for x in vals:
        if isinstance(x, (bool, np.bool_)):
            raise ValueError("bbox values must not be bool")
        n = float(x)
        if not np.isfinite(n):
            raise ValueError("bbox must be finite")
        if isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)):
            out.append(int(x))
        else:
            out.append(n)
    return out


def sha256_canonical(obj: Any) -> str:
    return hashlib.sha256(canonical_dumps(obj).encode("utf-8")).hexdigest()


def make_record_id(record: dict, *, level: str, data_token: str,
                   original_index: int | None = None) -> tuple[str, list[str]]:
    reasons: list[str] = []
    try:
        if level == "image":
            if record.get("path") in (None, ""):
                raise ValueError("missing path")
            payload = {"level": "image", "path": canonical_path(record["path"])}
        elif level == "object":
            missing = []
            if record.get("image_path") in (None, ""):
                missing.append("image_path")
            if record.get("bbox") is None:
                missing.append("bbox")
            if record.get("obj_index") is None:
                missing.append("obj_index")
            if missing:
                raise ValueError("missing:" + ",".join(missing))
            payload = {
                "level": "object",
                "image_path": canonical_path(record["image_path"]),
                "bbox": _finite_bbox(record["bbox"]),
                "obj_index": int(record["obj_index"]),
            }
        else:
            raise ValueError(f"unknown level {level}")
        return sha256_canonical(payload), reasons
    except (TypeError, ValueError, OSError) as exc:
        idx = 0 if original_index is None else int(original_index)
        reasons.append(f"missing_provenance:{exc}")
        return f"invalid:{data_token}:{idx}", reasons


def _parent_name(path) -> str:
    return Path(path).parent.name


def infer_label_status(record: dict, *, level: str, viz_mode: str) -> tuple[str, str, list[str]]:
    """Return (label_status, label_origin, reasons)."""
    reasons: list[str] = []
    label = record.get("label")
    label_s = "" if label is None else str(label)
    split = str(record.get("split") or "")
    if level == "object":
        if record.get("image_path") in (None, "") or record.get("bbox") is None \
                or record.get("obj_index") is None:
            return "unresolved", "missing_object_provenance", ["missing_provenance"]
        if label_s.strip() == "":
            return "missing", "empty_label", ["missing_label"]
        return "single", "object_annotation", reasons
    # image-level
    if viz_mode == "Object Detector":
        if label_s.strip() == "" or label_s == "unknown":
            return "missing", "detector_unknown", ["missing_label"]
        if label_s == "mix":
            return "multi", "detector_mix", ["multi_label"]
        return "single", "detector_image", reasons
    # classifier
    if label_s.strip() == "":
        return "missing", "empty_label", ["missing_label"]
    path = record.get("path")
    parent = _parent_name(path) if path is not None else ""
    if parent == split:
        return "unresolved", "flat_fallback", ["flat_fallback"]
    if parent == label_s:
        return "single", "class_folder", reasons
    return "unresolved", "unconfirmed_folder", ["unconfirmed_folder"]


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _crop_identity(record: dict) -> tuple:
    return (
        canonical_path(record.get("image_path")),
        tuple(_finite_bbox(record["bbox"])) if record.get("bbox") is not None else None,
        record.get("obj_index"),
    )


@dataclass
class PreparedInput:
    records: list[dict]
    embeddings: np.ndarray
    labels: list[str | None]
    record_ids: list[str]
    original_index: list[int]
    eligibility: list[bool]
    invalid_reasons: list[list[str]]
    label_status: list[str]
    source_label: list[str | None]
    effective_label: list[str | None]
    label_origin: list[str]
    analysis_key: str
    scope_fp: str
    source_snapshot_key: str
    source_verified: bool
    source_status: str
    analysis_split: str
    k: int
    data_token: str
    model: str
    embedding_recipe: dict
    event_generation: int = 0
    per_row_source_verified: list[bool] = field(default_factory=list)

    def as_validate_context(self) -> dict:
        return {
            "analysis_key": self.analysis_key,
            "event_generation": self.event_generation,
            "record_ids": list(self.record_ids),
            "original_index": list(self.original_index),
        }


def _embedding_digest(embeddings, eligibility, labels) -> bytes:
    """Hash raw float64 bytes plus scale-safe normalized valid rows.

    Raw bytes catch any replacement of the stored vectors. Normalized
    valid rows match the scoring space used by analyze().
    """
    E = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float64))
    parts = [E.astype("<f8", copy=False).tobytes()]
    for i in range(len(E)):
        if i < len(eligibility) and not eligibility[i]:
            continue
        u = scale_safe_normalize(E[i])
        if u is None:
            continue
        parts.append(np.asarray(u, dtype="<f8").tobytes())
    return b"".join(parts)


def _check_crop_conflicts(records: Sequence[dict], level: str) -> None:
    if level != "object":
        return
    by_crop: dict[str, list[tuple]] = {}
    for rec in records:
        path = rec.get("path")
        if path in (None, ""):
            continue
        key = canonical_path(path)
        try:
            ident = _crop_identity(rec)
        except (TypeError, ValueError):
            continue
        by_crop.setdefault(key, [])
        if ident not in by_crop[key]:
            by_crop[key].append(ident)
    conflicts = {k: v for k, v in by_crop.items() if len(v) > 1}
    if conflicts:
        sample = next(iter(conflicts.items()))
        raise CropIdentityConflictError(
            f"crop identity conflict: {sample[0]} maps to {len(sample[1])} source identities")


def _source_paths(record: dict, level: str) -> list[str]:
    if level == "object":
        src = record.get("image_path") or record.get("path")
    else:
        src = record.get("path")
    if src in (None, ""):
        return []
    return [canonical_path(src)]


def _evaluate_generation(records, context) -> tuple[str, bool, str]:
    generation = context.get("generation")
    level = context.get("level") or "image"
    if not generation:
        return "source_unverified", False, sha256_canonical({"status": "unverified"})
    hashes = generation.get("source_hashes") or {}
    snapshot_parts = []
    mismatched = False
    missing = False
    seen = set()
    for rec in records:
        for p in _source_paths(rec, level):
            if p in seen:
                continue
            seen.add(p)
            expected = hashes.get(p)
            if expected is None:
                missing = True
                snapshot_parts.append({"path": p, "status": "missing_generation_hash"})
                continue
            try:
                current = file_sha256(p)
            except OSError:
                mismatched = True
                snapshot_parts.append({"path": p, "status": "unreadable"})
                continue
            if current != expected:
                mismatched = True
                snapshot_parts.append({"path": p, "status": "mismatch",
                                       "generation": expected, "current": current})
            else:
                snapshot_parts.append({"path": p, "sha256": current})
    snap_key = sha256_canonical({"generation_hashes": hashes, "check": snapshot_parts,
                                 "recipe": generation.get("recipe")})
    if mismatched:
        return "source_stale", False, snap_key
    if missing:
        return "source_unverified", False, snap_key
    return "source_verified", True, snap_key


def prepare_records(viz_records: Sequence[dict], raw, context: dict) -> PreparedInput:
    records = list(viz_records)
    E = np.asarray(raw, dtype=np.float64)
    if E.ndim != 2:
        raise ClassSimilarityError("raw embeddings must be 2D")
    if len(records) != E.shape[0]:
        raise ClassSimilarityError(
            f"records ({len(records)}) and embeddings ({E.shape[0]}) are not aligned")
    level = str(context.get("level") or "image")
    data_token = str(context.get("data_token") or "")
    viz_mode = str(context.get("viz_mode") or "Image Classifier")
    analysis_split = context.get("analysis_split") or "all"
    k = int(context.get("k") or 5)
    _check_crop_conflicts(records, level)

    keep = list(range(len(records)))
    if analysis_split not in (None, "", "all"):
        keep = [i for i, r in enumerate(records) if str(r.get("split") or "") == str(analysis_split)]

    out_records = []
    labels: list[str | None] = []
    ids: list[str] = []
    orig = []
    eligibility = []
    invalid_reasons = []
    label_status = []
    source_label = []
    effective_label = []
    label_origin = []
    id_to_index: dict[str, int] = {}

    for new_i, old_i in enumerate(keep):
        rec = dict(records[old_i])
        status, origin, extra = infer_label_status(rec, level=level, viz_mode=viz_mode)
        rid, reasons = make_record_id(rec, level=level, data_token=data_token,
                                      original_index=old_i)
        reasons = list(reasons) + list(extra)
        eligible = status == "single" and not rid.startswith("invalid:")
        if not eligible and status == "single" and rid.startswith("invalid:"):
            eligible = False
        if rid in id_to_index and not rid.startswith("invalid:"):
            raise DuplicateIdentityError(
                f"duplicate record identity {rid} at rows {id_to_index[rid]} and {old_i}")
        if not rid.startswith("invalid:"):
            id_to_index[rid] = old_i
        eff = rec.get("label")
        eff_s = None if eff is None or str(eff).strip() == "" else str(eff)
        src = rec.get("source_label", eff)
        src_s = None if src is None or str(src).strip() == "" else str(src)
        out_records.append(rec)
        labels.append(eff_s if eligible else eff_s)
        ids.append(rid)
        orig.append(old_i)
        eligibility.append(bool(eligible))
        invalid_reasons.append(reasons)
        label_status.append(status)
        source_label.append(src_s)
        effective_label.append(eff_s)
        label_origin.append(origin)

    emb = E[keep] if keep else np.zeros((0, E.shape[1]), dtype=np.float64)
    source_status, source_verified, snap_key = _evaluate_generation(
        [records[i] for i in keep], context)
    recipe = dict(context.get("embedding_recipe") or {})
    meta = {
        "schema_version": SCHEMA_VERSION,
        "method_version": METHOD_VERSION,
        "data_token": data_token,
        "record_ids": ids,
        "labels": effective_label,
        "splits": [str(out_records[i].get("split") or "") for i in range(len(out_records))],
        "source_snapshot_key": snap_key,
        "eligibility": eligibility,
        "label_status": label_status,
        "model": context.get("model"),
        "weights_id": context.get("model_weights_id") or "unknown",
        "embedding_recipe": recipe,
        "k": k,
        "analysis_split": analysis_split,
    }
    digest = _embedding_digest(emb, eligibility, labels)
    analysis_key = hashlib.sha256(canonical_dumps(meta).encode("utf-8") + digest).hexdigest()
    scope_fp = sha256_canonical({
        "record_ids": ids,
        "effective_label": effective_label,
        "label_status": label_status,
        "analysis_split": analysis_split,
        "data_token": data_token,
    })
    return PreparedInput(
        records=out_records, embeddings=emb, labels=labels, record_ids=ids,
        original_index=orig, eligibility=eligibility, invalid_reasons=invalid_reasons,
        label_status=label_status, source_label=source_label,
        effective_label=effective_label, label_origin=label_origin,
        analysis_key=analysis_key, scope_fp=scope_fp, source_snapshot_key=snap_key,
        source_verified=source_verified, source_status=source_status,
        analysis_split=str(analysis_split), k=k, data_token=data_token,
        model=str(context.get("model") or ""), embedding_recipe=recipe,
        event_generation=int(context.get("event_generation") or 0),
        per_row_source_verified=[source_verified] * len(ids),
    )


def scope_fingerprint(prepared: PreparedInput, **_display_ignored) -> str:
    return prepared.scope_fp


def analysis_inputs_fingerprint(prepared: PreparedInput, **_display_ignored) -> str:
    return prepared.analysis_key


def validate_selection(event, result: AnalysisResult | None, current_context) -> list[int]:
    if event is None:
        raise StaleSelectionError("missing selection event")
    if isinstance(current_context, PreparedInput):
        ctx = current_context.as_validate_context()
        record_ids = current_context.record_ids
        orig = current_context.original_index
        key = current_context.analysis_key
        gen = current_context.event_generation
    else:
        ctx = dict(current_context or {})
        record_ids = list(ctx.get("record_ids") or [])
        orig = list(ctx.get("original_index") or range(len(record_ids)))
        key = ctx.get("analysis_key")
        gen = int(ctx.get("event_generation") or 0)
    if event.get("analysis_key") != key:
        raise StaleSelectionError("analysis_key mismatch")
    if event.get("event_generation") is not None and int(event["event_generation"]) != gen:
        raise StaleSelectionError("event generation mismatch")
    rid = event.get("record_id")
    index = event.get("original_index")
    if index is None:
        raise StaleSelectionError("missing original_index")
    index = int(index)
    # original_index is viz_records position; map back to prepared row
    if orig and index in orig:
        local = orig.index(index)
    elif 0 <= index < len(record_ids):
        local = index
    else:
        raise StaleSelectionError("original_index out of range")
    if record_ids[local] != rid:
        raise StaleSelectionError("record_id does not match original_index")
    if str(rid).startswith("invalid:"):
        raise StaleSelectionError("invalid identity cannot be selected")
    if result is not None and local < len(result.items):
        if result.items[local].record_id != rid:
            raise StaleSelectionError("result row identity mismatch")
    return [index]


def selection_original_indices(prepared: PreparedInput, record_ids: Iterable[str]) -> list[int]:
    want = set(record_ids)
    out = []
    for rid, orig, elig in zip(prepared.record_ids, prepared.original_index, prepared.eligibility):
        if rid in want and elig and not str(rid).startswith("invalid:"):
            out.append(orig)
    return out
