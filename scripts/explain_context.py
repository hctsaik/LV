"""Context resolution, geometry, compatibility, and stale-result guards."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from explain_contracts import (
    ExplainContextV1,
    ExplainResultV1,
    ExplainRunSpecV1,
    ContractError,
    canonical_payload,
    deterministic_compute_key,
    identity_status,
    is_legacy_identity,
    mark_legacy_identity,
    stable_record_key,
)


def normalize_bbox(bbox: Sequence[float], *, source_dimensions: Sequence[int] | None = None) -> tuple[float, float, float, float]:
    """Validate and round an xyxy box in normalized source coordinates."""
    if len(bbox) != 4:
        raise ContractError("bbox must contain x0, y0, x1, y1")
    values = tuple(round(float(x), 12) for x in bbox)
    if any(x < 0 or x > 1 for x in values) or values[2] < values[0] or values[3] < values[1]:
        raise ContractError("bbox must be an xyxy box in [0, 1]")
    return values


def bbox_to_pixels(bbox_normalized: Sequence[float], source_dimensions: Sequence[int]) -> tuple[int, int, int, int]:
    bbox = normalize_bbox(bbox_normalized, source_dimensions=source_dimensions)
    width, height = (int(source_dimensions[0]), int(source_dimensions[1]))
    if width <= 0 or height <= 0:
        raise ContractError("source dimensions must be positive")
    return (round(bbox[0] * width), round(bbox[1] * height), round(bbox[2] * width), round(bbox[3] * height))


def record_key_for(record: Mapping[str, Any]) -> str:
    """Resolve stable identity from a record without trusting array position."""
    source = record.get("source_sha256") or record.get("sha256") or record.get("image_sha256")
    if not source:
        raise ContractError("record has no source SHA-256")
    bbox = record.get("bbox_normalized") or record.get("bbox")
    index = record.get("object_index")
    crop = record.get("crop_policy", "full_frame")
    if bbox is None and index is None:
        return stable_record_key(str(source))
    return stable_record_key(str(source), object_index=int(index or 0), bbox_normalized=normalize_bbox(bbox or (0, 0, 1, 1)), crop_policy=crop)


def resolve_record(records: Sequence[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]], record_key: str) -> Mapping[str, Any] | None:
    """Find a record by content identity after arbitrary record reordering."""
    values = records.values() if isinstance(records, Mapping) else records
    for record in values:
        try:
            if str(record.get("record_key") or record_key_for(record)) == str(record_key):
                return record
        except (ContractError, TypeError, ValueError):
            continue
    return None


def make_explain_context(**values: Any) -> ExplainContextV1:
    """Construct a context and derive its record key when possible."""
    data = dict(values)
    if data.get("source_dimensions") is not None:
        data["source_dimensions"] = tuple(data["source_dimensions"])
    if data.get("bbox_normalized") is not None:
        data["bbox_normalized"] = normalize_bbox(data["bbox_normalized"])
    if data.get("bbox_pixel") is not None:
        data["bbox_pixel"] = tuple(int(x) for x in data["bbox_pixel"])
    if not data.get("record_key") and data.get("source_sha256"):
        data["record_key"] = stable_record_key(
            data["source_sha256"],
            object_index=data.get("object_index"),
            bbox_normalized=data.get("bbox_normalized"),
            crop_policy=data.get("crop_policy", "full_frame"),
        ) if data.get("object_index") is not None or data.get("bbox_normalized") is not None else stable_record_key(data["source_sha256"])
    return ExplainContextV1(**data)


def context_contract(value: ExplainContextV1 | Mapping[str, Any]) -> ExplainContextV1:
    """Coerce an app/session mapping into the versioned reference contract.

    UI state may carry display-only fields (for example ``record_index`` or
    a resolved ``source_path``).  Keep those fields outside the contract while
    ensuring compatibility checks and compute keys use the same canonical
    serializer everywhere.
    """
    if isinstance(value, ExplainContextV1):
        return value
    if not isinstance(value, Mapping):
        raise ContractError("context must be a mapping or ExplainContextV1")
    allowed = set(ExplainContextV1.__dataclass_fields__)
    data = {key: item for key, item in value.items() if key in allowed}
    return ExplainContextV1.from_dict(data)


@dataclass(frozen=True)
class CompatibilityCheck:
    status: str
    stale_fields: tuple[str, ...] = ()
    incompatible_fields: tuple[str, ...] = ()

    @property
    def current(self) -> bool:
        return self.status == "current"

    @property
    def stale(self) -> bool:
        return bool(self.stale_fields) or self.status == "stale"

    @property
    def incompatible(self) -> bool:
        return bool(self.incompatible_fields) or self.status == "incompatible"

    @property
    def reason(self) -> str:
        fields = self.incompatible_fields + self.stale_fields
        return ", ".join(fields) if fields else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stale_fields": list(self.stale_fields),
            "incompatible_fields": list(self.incompatible_fields),
            "reason": self.reason,
        }


_STALE_CONTEXT_FIELDS = (
    "origin_generation",
    "dataset_fingerprint",
    "source_sha256",
    "record_key",
)
_INCOMPATIBLE_CONTEXT_FIELDS = (
    "model_id",
    "checkpoint_id",
    "checkpoint_digest",
    "scope",
    "source_granularity",
    "embedding_recipe",
    "head_ref",
    "head_digest",
    "head_provenance",
    "normal_bank_ref",
    "normal_bank_digest",
)
_INCOMPATIBLE_RUN_FIELDS = (
    "scope",
    "target_spec",
    "reference_spec",
    "target_scalar",
    "attribution_method",
    "analysis_target_res",
    "pooling",
    "keep_aspect",
    "crop_policy",
    "faithfulness_profile",
    "occlusion_baseline",
    "seed",
    "checkpoint_digest",
    "head_digest",
    "normal_bank_digest",
    "embedding_recipe",
    "engine_schema_version",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, ExplainContextV1 | ExplainRunSpecV1):
        return value.to_dict()
    if isinstance(value, Mapping):
        return value
    raise ContractError("expected a context or run mapping")


def check_compatibility(
    expected_context: ExplainContextV1 | Mapping[str, Any],
    current_context: ExplainContextV1 | Mapping[str, Any],
    run: ExplainRunSpecV1 | Mapping[str, Any] | None = None,
    current_run: ExplainRunSpecV1 | Mapping[str, Any] | None = None,
) -> CompatibilityCheck:
    """Compare a result context to the current context.

    Incompatible recipe/head fields take precedence over stale generation or
    source fields, so callers cannot accidentally run an incompatible result
    merely because it is also old.  Either state blocks publication.
    """
    # Accept the intuitive (context, run) ordering if a caller passes the
    # arguments positionally in the opposite order during migration.
    if isinstance(expected_context, ExplainRunSpecV1) and isinstance(current_context, ExplainContextV1):
        expected_context, current_context = current_context, expected_context
    expected = _mapping(expected_context)
    current = _mapping(current_context)
    stale = [name for name in _STALE_CONTEXT_FIELDS if expected.get(name) not in (None, "") and current.get(name) != expected.get(name)]
    incompatible = [name for name in _INCOMPATIBLE_CONTEXT_FIELDS if expected.get(name) not in (None, "") and current.get(name) != expected.get(name)]
    if run is not None:
        run_data = _mapping(run)
        compare_to = _mapping(current_run) if current_run is not None else current
        aliases = {"analysis_target_res": ("analysis_target_res", "target_res")}
        for name in _INCOMPATIBLE_RUN_FIELDS:
            run_value = run_data.get(name)
            current_value = compare_to.get(name)
            if current_value in (None, ""):
                for alias in aliases.get(name, ()):
                    current_value = compare_to.get(alias)
                    if current_value not in (None, ""):
                        break
            if run_value not in (None, "") and current_value not in (None, "") and run_value != current_value:
                incompatible.append(f"run.{name}")
    incompatible = list(dict.fromkeys(incompatible))
    stale = list(dict.fromkeys(stale))
    status = "incompatible" if incompatible else ("stale" if stale else "current")
    return CompatibilityCheck(status, tuple(stale), tuple(incompatible))


def validate_context(*args: Any, **kwargs: Any) -> CompatibilityCheck:
    return check_compatibility(*args, **kwargs)


def stale_reason(*args: Any, **kwargs: Any) -> str:
    return check_compatibility(*args, **kwargs).reason


def validate_late_result(
    result: ExplainResultV1 | Mapping[str, Any],
    current_context: ExplainContextV1 | Mapping[str, Any],
    *,
    current_run_key: str | None = None,
) -> CompatibilityCheck:
    """Guard publication of a result that completed after a context switch."""
    item = result.to_dict() if isinstance(result, ExplainResultV1) else dict(result)
    current = _mapping(current_context)
    stale = []
    checks = (
        ("context_id", item.get("context_id"), current.get("context_id")),
        ("requested_generation", item.get("requested_generation"), current.get("origin_generation")),
        ("source_sha256", item.get("source_sha256"), current.get("source_sha256")),
    )
    for field, old, new in checks:
        if old not in (None, "") and new not in (None, "") and old != new:
            stale.append(field)
    if current_run_key not in (None, "") and item.get("run_key") not in (None, "") and item.get("run_key") != current_run_key:
        stale.append("run_key")
    stale = tuple(dict.fromkeys(stale))
    return CompatibilityCheck("stale" if stale else "current", stale_fields=stale)


def late_result_is_current(*args: Any, **kwargs: Any) -> bool:
    return validate_late_result(*args, **kwargs).current


def result_status_for(
    result: ExplainResultV1 | Mapping[str, Any],
    current_context: ExplainContextV1 | Mapping[str, Any],
    *,
    current_run_key: str | None = None,
) -> str:
    check = validate_late_result(result, current_context, current_run_key=current_run_key)
    return check.status


def classify_legacy_cache(identity: Any) -> str:
    """Return ``legacy`` for path/size/mtime records; never silently upgrade."""
    return identity_status(identity)


__all__ = [
    "CompatibilityCheck",
    "ExplainContextV1",
    "ExplainResultV1",
    "ExplainRunSpecV1",
    "bbox_to_pixels",
    "check_compatibility",
    "classify_legacy_cache",
    "deterministic_compute_key",
    "late_result_is_current",
    "make_explain_context",
    "normalize_bbox",
    "record_key_for",
    "resolve_record",
    "result_status_for",
    "stable_record_key",
    "stale_reason",
    "validate_context",
    "validate_late_result",
    "is_legacy_identity",
    "mark_legacy_identity",
]
