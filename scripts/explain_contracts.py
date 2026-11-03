"""Versioned, serialisable contracts for model diagnostics.

The contracts in this module are intentionally framework free.  They carry
references and provenance only; images, models, tensors, and bank vectors do
not belong in an :class:`ExplainContextV1` payload.  All identities used for
reuse are content based and all JSON used in a key is canonical.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence


CONTRACT_SCHEMA_VERSION = 1
CONTEXT_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
RENDER_SCHEMA_VERSION = 1
MAX_CONTEXT_PAYLOAD_BYTES = 64 * 1024

CANONICAL_SIGNED_GRADCAM = "canonical_signed_gradcam"
EXPERIMENTAL_TOKEN_WISE = "experimental_token_wise_attribution"
TOKEN_GRADIENT_PRODUCT = EXPERIMENTAL_TOKEN_WISE

HEAD_EVIDENCE_STATES = frozenset(
    {"incompatible_or_stale", "validation_insufficient", "sample_level_audit_available", "grouped_supported"}
)
FAITHFULNESS_STATES = frozenset(
    {"unavailable", "unsupported", "inconclusive", "single_sample_smoke_supported", "batch_supported"}
)
LOCALIZATION_STATES = frozenset(
    {"unavailable", "incompatible", "unverified", "mask_agreement_supported"}
)
RESULT_STATES = frozenset({"current", "cached", "stale", "historical", "incompatible", "failed"})
JOB_STATES = frozenset({"queued", "running", "cancel_requested", "partial", "complete", "failed", "stale"})


class ContractError(ValueError):
    """Raised when a contract cannot be safely represented or validated."""


class EvidenceState(str, Enum):
    HEAD_INCOMPATIBLE_OR_STALE = "incompatible_or_stale"
    HEAD_VALIDATION_INSUFFICIENT = "validation_insufficient"
    HEAD_SAMPLE_LEVEL_AUDIT = "sample_level_audit_available"
    HEAD_GROUPED_SUPPORTED = "grouped_supported"
    FAITHFULNESS_UNAVAILABLE = "unavailable"
    FAITHFULNESS_UNSUPPORTED = "unsupported"
    FAITHFULNESS_INCONCLUSIVE = "inconclusive"
    FAITHFULNESS_SMOKE = "single_sample_smoke_supported"
    FAITHFULNESS_BATCH = "batch_supported"
    LOCALIZATION_INCOMPATIBLE = "incompatible"
    LOCALIZATION_UNVERIFIED = "unverified"
    LOCALIZATION_MASK_SUPPORTED = "mask_agreement_supported"


def _jsonable(value: Any, *, reject_binary: bool = False) -> Any:
    """Convert safe scalar/mapping values into deterministic JSON values.

    Arrays and tensors are represented with their shape, dtype, and byte
    digest for digest payloads.  They are rejected for context payloads.
    """
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("canonical JSON does not permit NaN or infinity")
        return value
    if isinstance(value, Enum):
        return _jsonable(value.value, reject_binary=reject_binary)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes | bytearray | memoryview):
        if reject_binary:
            raise ContractError("binary data is not allowed in a context reference")
        raw = bytes(value)
        return {"__bytes_sha256__": hashlib.sha256(raw).hexdigest(), "length": len(raw)}
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value), reject_binary=reject_binary)
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError("canonical mapping keys must be strings")
            out[key] = _jsonable(item, reject_binary=reject_binary)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        if isinstance(value, (set, frozenset)):
            value = sorted(value, key=lambda item: str(item))
        return [_jsonable(item, reject_binary=reject_binary) for item in value]

    # numpy scalar values (and similar scalar wrappers) are safe once reduced
    # to their Python equivalent.  This keeps digest helpers dependency-free.
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item(), reject_binary=reject_binary)
        except (TypeError, ValueError):
            pass

    # Optional numpy/torch support without importing either package.
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None and hasattr(value, "tobytes"):
        if reject_binary:
            raise ContractError("array/tensor data is not allowed in a context reference")
        raw = value.tobytes()
        return {
            "__array_sha256__": hashlib.sha256(raw).hexdigest(),
            "dtype": str(dtype),
            "shape": [int(x) for x in shape],
        }
    raise ContractError(f"unsupported value in canonical payload: {type(value).__name__}")


def canonical_payload(value: Any, *, reject_binary: bool = False) -> Any:
    """Return the canonical Python representation of ``value``."""
    return _jsonable(value, reject_binary=reject_binary)


def canonical_json(value: Any, *, reject_binary: bool = False) -> str:
    """Serialise with stable ordering and separators for hashes/downloads."""
    return json.dumps(
        canonical_payload(value, reject_binary=reject_binary),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def to_canonical_json(value: Any, *, reject_binary: bool = False) -> str:
    """Named alias used by callers that distinguish canonical from display JSON."""
    return canonical_json(value, reject_binary=reject_binary)


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(bytes(value)).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_payload(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def content_digest(value: Any) -> str:
    """Digest bytes/files or a canonical representation of structured data."""
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        if path.is_file():
            return sha256_file(path)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return sha256_bytes(value)
    return _digest_payload(value)


def digest_checkpoint(checkpoint: Any) -> str:
    """Return a strong checkpoint content digest, never path/mtime identity."""
    return content_digest(checkpoint)


def checkpoint_content_digest(checkpoint: Any) -> str:
    return digest_checkpoint(checkpoint)


def _model_payload(model: Any) -> Any:
    if isinstance(model, Mapping):
        return dict(model)
    attrs = {}
    for name in ("coef_", "intercept_", "classes_", "temperature", "weight", "bias"):
        if hasattr(model, name):
            attrs[name] = getattr(model, name)
    return attrs or repr(model)


def digest_classifier_head(head: Any) -> str:
    """Digest weights, classes, calibration, and all supplied provenance."""
    if isinstance(head, Mapping):
        payload = dict(head)
        if "model" in payload:
            payload["model"] = _model_payload(payload["model"])
    else:
        payload = _model_payload(head)
    return _digest_payload(payload)


def head_content_digest(head: Any) -> str:
    return digest_classifier_head(head)


def digest_normal_bank(bank: Any) -> str:
    """Digest bank vectors plus metadata/provenance that affects scoring."""
    return content_digest(bank)


def normal_bank_content_digest(bank: Any) -> str:
    return digest_normal_bank(bank)


def _hex_digest(value: str, name: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ContractError(f"{name} must be a SHA-256 hexadecimal digest")
    return text


def _as_tuple(value: Sequence[Any] | None, *, length: int | None = None) -> tuple[Any, ...] | None:
    if value is None:
        return None
    result = tuple(value)
    if length is not None and len(result) != length:
        raise ContractError(f"expected {length} values, got {len(result)}")
    return result


def stable_record_key(
    source_sha256: str,
    *,
    object_index: int | None = None,
    bbox_normalized: Sequence[float] | None = None,
    crop_policy: str | Mapping[str, Any] | None = None,
) -> str:
    """Build a stable whole-image or object-record key.

    A whole image intentionally uses its source digest directly.  Object
    records add object index, normalized geometry, and crop recipe.
    """
    source = _hex_digest(source_sha256, "source_sha256")
    if object_index is None and bbox_normalized is None and crop_policy in (None, "", "full_frame"):
        return source
    if object_index is None:
        raise ContractError("object_index is required for an object record")
    bbox = tuple(round(float(x), 12) for x in (_as_tuple(bbox_normalized, length=4) or ()))
    if len(bbox) != 4 or any(x < 0 or x > 1 for x in bbox) or bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        raise ContractError("bbox_normalized must be an xyxy box in [0, 1]")
    payload = {
        "source_sha256": source,
        "object_index": int(object_index),
        "bbox_normalized": bbox,
        "crop_policy": crop_policy or "object_crop",
    }
    return f"{source}:{_digest_payload(payload)}"


def whole_image_key(source_sha256: str) -> str:
    return stable_record_key(source_sha256)


def object_record_key(source_sha256: str, object_index: int, bbox_normalized: Sequence[float], crop_policy: Any) -> str:
    return stable_record_key(
        source_sha256,
        object_index=object_index,
        bbox_normalized=bbox_normalized,
        crop_policy=crop_policy,
    )


class _ContractMixin:
    contract_name: ClassVar[str]
    schema_version: int

    def to_dict(self) -> dict[str, Any]:
        return canonical_payload(dataclasses.asdict(self), reject_binary=False)

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def canonical_json(self) -> str:
        return self.to_json()

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()

    @property
    def payload_bytes(self) -> int:
        return len(self.to_json().encode("utf-8"))


@dataclass(frozen=True)
class ExplainContextV1(_ContractMixin):
    """Reference-only context passed between Visualize and diagnostics."""

    contract_name: ClassVar[str] = "ExplainContextV1"
    context_id: str = ""
    context_version: int = CONTEXT_SCHEMA_VERSION
    origin_tool: str = "visualize"
    origin_view: str = "viewer"
    origin_generation: str | int | None = None
    record_key: str = ""
    source_path: str | None = None
    source_relative_path: str | None = None
    source_sha256: str = ""
    source_dimensions: tuple[int, int] | None = None
    dataset_root_identity: str | None = None
    dataset_fingerprint: str | None = None
    manifest_ref: str | None = None
    scope: str = "whole_image"
    source_granularity: str = "classifier_image"
    object_index: int | None = None
    bbox_normalized: tuple[float, float, float, float] | None = None
    bbox_pixel: tuple[int, int, int, int] | None = None
    crop_policy: str | Mapping[str, Any] = "full_frame"
    label: str | None = None
    split: str | None = None
    model_id: str = ""
    checkpoint_id: str | None = None
    checkpoint_digest: str | None = None
    embedding_recipe: Mapping[str, Any] = field(default_factory=dict)
    embedding_ref: str | None = None
    head_ref: str | None = None
    head_digest: str | None = None
    head_provenance: Mapping[str, Any] = field(default_factory=dict)
    mask_ref: str | None = None
    normal_bank_ref: str | None = None
    normal_bank_digest: str | None = None
    engine_schema_version: str = "1"
    created_at: str | None = None
    schema_version: int = CONTEXT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.source_sha256:
            _hex_digest(self.source_sha256, "source_sha256")
        for name, value in (("checkpoint_digest", self.checkpoint_digest), ("head_digest", self.head_digest), ("normal_bank_digest", self.normal_bank_digest)):
            if value:
                _hex_digest(value, name)
        if self.scope not in {"whole_image", "object_crop"}:
            raise ContractError("scope must be whole_image or object_crop")
        if self.source_dimensions is not None:
            dims = _as_tuple(self.source_dimensions, length=2)
            if any(int(x) <= 0 for x in dims):
                raise ContractError("source_dimensions must be positive")
        if self.bbox_normalized is not None:
            stable_record_key(self.source_sha256 or "0" * 64, object_index=self.object_index or 0, bbox_normalized=self.bbox_normalized, crop_policy=self.crop_policy)
        if self.record_key and any(token in self.record_key.lower() for token in ("mtime", "file_size")):
            raise ContractError("record_key cannot be based on path/size/mtime")
        # Explicitly reject accidental image/model/array payloads.
        canonical_json(dataclasses.asdict(self), reject_binary=True)
        if self.payload_bytes > MAX_CONTEXT_PAYLOAD_BYTES:
            raise ContractError(f"context payload exceeds {MAX_CONTEXT_PAYLOAD_BYTES} bytes")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExplainContextV1":
        data = dict(value)
        data.pop("contract_name", None)
        data["source_dimensions"] = _as_tuple(data.get("source_dimensions"), length=2)
        for name in ("bbox_normalized", "bbox_pixel"):
            data[name] = _as_tuple(data.get(name), length=4)
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "ExplainContextV1":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class ExplainRunSpecV1(_ContractMixin):
    contract_name: ClassVar[str] = "ExplainRunSpecV1"
    kind: str = "gradcam"
    run_version: int = RUN_SCHEMA_VERSION
    scope: str = "whole_image"
    target_spec: str | int = "auto_top1"
    reference_spec: str | int = "strongest_runner_up"
    target_scalar: str = "logit_margin"
    attribution_method: str = CANONICAL_SIGNED_GRADCAM
    analysis_target_res: int = 224
    pooling: str = "cls"
    keep_aspect: bool = True
    crop_policy: str | Mapping[str, Any] = "full_frame"
    faithfulness_profile: str = "none"
    occlusion_baseline: str = "mean_color"
    seed: int = 0
    checkpoint_digest: str | None = None
    head_digest: str | None = None
    normal_bank_digest: str | None = None
    embedding_recipe: Mapping[str, Any] = field(default_factory=dict)
    engine_schema_version: str = "1"
    schema_version: int = RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.kind not in {"attention", "gradcam", "anomaly"}:
            raise ContractError("kind must be attention, gradcam, or anomaly")
        if self.attribution_method not in {CANONICAL_SIGNED_GRADCAM, EXPERIMENTAL_TOKEN_WISE, "global_channel", "token_gradient_product"}:
            raise ContractError(f"unsupported attribution method: {self.attribution_method}")
        if int(self.analysis_target_res) <= 0:
            raise ContractError("analysis_target_res must be positive")
        for name, value in (("checkpoint_digest", self.checkpoint_digest), ("head_digest", self.head_digest), ("normal_bank_digest", self.normal_bank_digest)):
            if value:
                _hex_digest(value, name)
        canonical_json(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExplainRunSpecV1":
        data = dict(value)
        data.pop("contract_name", None)
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "ExplainRunSpecV1":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class RenderSpecV1(_ContractMixin):
    contract_name: ClassVar[str] = "RenderSpecV1"
    display_mode: str = "signed"
    render_version: int = RENDER_SCHEMA_VERSION
    palette: str = "blue_white_red"
    opacity: float = 0.50
    show_grid: bool = False
    overlay_size: tuple[int, int] | None = None
    schema_version: int = RENDER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.display_mode not in {"signed", "target_positive"}:
            raise ContractError("display_mode must be signed or target_positive")
        if not 0 <= float(self.opacity) <= 1:
            raise ContractError("opacity must be between 0 and 1")
        if self.overlay_size is not None and any(int(x) <= 0 for x in self.overlay_size):
            raise ContractError("overlay_size must be positive")
        canonical_json(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RenderSpecV1":
        data = dict(value)
        data.pop("contract_name", None)
        if data.get("overlay_size") is not None:
            data["overlay_size"] = tuple(data["overlay_size"])
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "RenderSpecV1":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class ExplainResultV1(_ContractMixin):
    contract_name: ClassVar[str] = "ExplainResultV1"
    context_id: str = ""
    requested_generation: str | int | None = None
    source_sha256: str = ""
    run_key: str = ""
    compute_key: str = ""
    status: str = "current"
    evidence: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    raw_patch_scores: Any = None
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in RESULT_STATES:
            raise ContractError(f"unknown result status: {self.status}")
        if self.source_sha256:
            _hex_digest(self.source_sha256, "source_sha256")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExplainResultV1":
        data = dict(value)
        data.pop("contract_name", None)
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "ExplainResultV1":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class ExplainJobV1(_ContractMixin):
    contract_name: ClassVar[str] = "ExplainJobV1"
    job_id: str = ""
    context_id: str = ""
    requested_generation: str | int | None = None
    requested_source_sha256: str = ""
    requested_run_key: str = ""
    status: str = "queued"
    error: str | None = None
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in JOB_STATES:
            raise ContractError(f"unknown job status: {self.status}")
        if self.requested_source_sha256:
            _hex_digest(self.requested_source_sha256, "requested_source_sha256")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExplainJobV1":
        data = dict(value)
        data.pop("contract_name", None)
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "ExplainJobV1":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class EvidenceStatusV1(_ContractMixin):
    contract_name: ClassVar[str] = "EvidenceStatusV1"
    head: str = "validation_insufficient"
    faithfulness: str = "unavailable"
    localization: str = "unavailable"
    decision_ambiguous: bool = False
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.head not in HEAD_EVIDENCE_STATES:
            raise ContractError(f"unknown head evidence state: {self.head}")
        if self.faithfulness not in FAITHFULNESS_STATES:
            raise ContractError(f"unknown faithfulness state: {self.faithfulness}")
        if self.localization not in LOCALIZATION_STATES:
            raise ContractError(f"unknown localization state: {self.localization}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceStatusV1":
        data = dict(value)
        data.pop("contract_name", None)
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "EvidenceStatusV1":
        return cls.from_dict(json.loads(value))


# Descriptive aliases for callers that prefer status-specific class names.
HeadValidationStatus = EvidenceStatusV1
FaithfulnessStatus = EvidenceStatusV1
LocalizationStatus = EvidenceStatusV1


def deterministic_compute_key(context: ExplainContextV1 | Mapping[str, Any], run: ExplainRunSpecV1 | Mapping[str, Any]) -> str:
    """Hash only source/model/analysis inputs; render settings are excluded."""
    context_payload = context.to_dict() if isinstance(context, _ContractMixin) else dict(context)
    run_payload = run.to_dict() if isinstance(run, _ContractMixin) else dict(run)
    # Deliberately select fields rather than hashing the entire handoff.  UI
    # bookkeeping (context id, origin view, timestamps, and paths) must never
    # cause a model recomputation; generation/fingerprint remain here because
    # a new dataset snapshot must not reuse an old result.
    context_payload = {
        name: context_payload.get(name)
        for name in (
            "origin_generation", "dataset_fingerprint", "source_sha256",
            "source_dimensions", "record_key", "scope", "source_granularity",
            "object_index", "bbox_normalized", "bbox_pixel", "crop_policy",
            "model_id", "checkpoint_digest", "head_digest", "head_provenance",
            "embedding_recipe", "normal_bank_digest", "engine_schema_version",
        )
    }
    run_payload = {
        name: run_payload.get(name)
        for name in (
            "run_version", "kind", "scope", "target_spec", "reference_spec",
            "target_scalar", "attribution_method", "analysis_target_res",
            "pooling", "keep_aspect", "crop_policy", "faithfulness_profile",
            "occlusion_baseline", "seed", "checkpoint_digest", "head_digest",
            "normal_bank_digest", "embedding_recipe", "engine_schema_version",
        )
    }
    payload = {
        "contract": "ExplainComputeKeyV1",
        "context": context_payload,
        "run": run_payload,
    }
    return _digest_payload(payload)


def compute_key(context: ExplainContextV1 | Mapping[str, Any], run: ExplainRunSpecV1 | Mapping[str, Any]) -> str:
    return deterministic_compute_key(context, run)


def run_key(context: ExplainContextV1 | Mapping[str, Any], run: ExplainRunSpecV1 | Mapping[str, Any]) -> str:
    return deterministic_compute_key(context, run)


def is_legacy_identity(identity: Any) -> bool:
    """Detect path/size/mtime identities without promoting them."""
    if not isinstance(identity, Mapping):
        return False
    if identity.get("identity_schema_version") in {0, "0", "legacy", "path_size_mtime"}:
        return True
    has_content = any(identity.get(k) for k in ("sha256", "content_sha256", "checkpoint_digest", "head_digest"))
    has_legacy = any(k in identity for k in ("path", "source_path", "mtime", "mtime_ns", "size", "file_size"))
    return bool(has_legacy and not has_content)


def identity_status(identity: Any) -> str:
    if is_legacy_identity(identity):
        return "legacy"
    if isinstance(identity, Mapping) and any(identity.get(k) for k in ("sha256", "content_sha256", "checkpoint_digest", "head_digest")):
        return "content_verified"
    return "unverified"


def mark_legacy_identity(identity: Mapping[str, Any], *, reason: str = "path_size_mtime_identity") -> dict[str, Any]:
    result = dict(identity)
    result["identity_status"] = "legacy"
    result["legacy_reason"] = reason
    result["reusable"] = False
    return result


def checkpoint_identity(checkpoint: Any) -> str:
    """Compatibility name for callers migrating from path metadata identity."""
    return digest_checkpoint(checkpoint)


# Explicit aliases make migration from existing helper naming unsurprising.
source_content_digest = content_digest
source_digest = content_digest
sha256_path = sha256_file
head_digest = digest_classifier_head
bank_digest = digest_normal_bank
record_identity = stable_record_key
