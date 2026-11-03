"""Versioned detector identities. Existing DINO V1 contracts are untouched."""
from __future__ import annotations

import hashlib
import json
import math
import copy
from dataclasses import asdict, dataclass, field


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class DetectionTargetV2:
    model_sha256: str
    input_sha256: str
    output_index: int
    class_index: int
    class_name: str
    box: tuple[float, float, float, float]
    scalar: str = "confidence"
    schema_version: int = 2

    def __post_init__(self):
        object.__setattr__(self,"box",tuple(self.box))
        if self.schema_version != 2 or not self.class_name:
            raise ValueError("Detector V2 requires a class name and schema_version=2")
        for value in (self.model_sha256, self.input_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Detector target requires SHA-256 identities")
        if type(self.output_index) is not int or self.output_index < 0 or type(self.class_index) is not int or self.class_index < 0:
            raise ValueError("Detector target indices must be nonnegative integers")
        if self.scalar not in {"confidence", "class", "objectness"}:
            raise ValueError("Unsupported detector score expression")
        if len(self.box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in self.box):
            raise ValueError("Target box must use normalized source coordinates")
        if self.box[0] >= self.box[2] or self.box[1] >= self.box[3]:
            raise ValueError("Target box must have positive area")

    def as_dict(self):
        return asdict(self)

    @property
    def key(self):
        return identity(self.as_dict())


@dataclass(frozen=True)
class DetectorRunSpecV2:
    target: DetectionTargetV2
    geometry: dict
    layer: dict
    method: str = "positive"
    profile_revision: str = ""
    source_sha256: str = ""
    case_context: dict = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self):
        for name in ("geometry","layer","case_context"):
            object.__setattr__(self,name,copy.deepcopy(getattr(self,name)))
        if self.schema_version != 2:
            raise ValueError("Detector run requires schema_version=2")
        if self.method not in {"positive", "signed"}:
            raise ValueError("Detector CAM method must be positive or signed")
        identity(self.as_dict())

    def as_dict(self):
        return asdict(self)

    @property
    def key(self):
        return identity(self.as_dict())


def is_current(result, spec):
    """Late results cannot replace a different model, target, layer or geometry."""
    return result.get("compute_id") == spec.key


@dataclass(frozen=True)
class DetectionTargetV3:
    """Modern final-row identity plus the immutable pre-top-k score target."""
    model_sha256: str
    input_sha256: str
    output_index: int
    class_index: int
    class_name: str
    box: tuple[float, float, float, float]
    native_candidate_index: int
    execution: dict
    scalar: str = "confidence"
    schema_version: int = 3

    def __post_init__(self):
        DetectionTargetV2(self.model_sha256, self.input_sha256, self.output_index,
                          self.class_index, self.class_name, self.box, self.scalar)
        if self.schema_version != 3 or type(self.native_candidate_index) is not int or self.native_candidate_index < 0:
            raise ValueError("Modern target requires a nonnegative native candidate")
        object.__setattr__(self, "box", tuple(self.box))
        object.__setattr__(self, "execution", copy.deepcopy(self.execution))
        if not all(self.execution.get(k) for k in ("runtime_id", "runtime_release", "head_path", "output_contract")):
            raise ValueError("Modern target requires actual runtime/head/output identity")
        identity(self.as_dict())

    def as_dict(self):
        return asdict(self)

    @property
    def key(self):
        return identity(self.as_dict())


@dataclass(frozen=True)
class DetectorRunSpecV3:
    target: DetectionTargetV3
    geometry: dict
    layer: dict
    method: str = "positive"
    profile_revision: str = ""
    source_sha256: str = ""
    case_context: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    schema_version: int = 3

    def __post_init__(self):
        if self.schema_version != 3 or self.method not in {"positive", "signed"}:
            raise ValueError("Invalid modern CAM specification")
        for name in ("geometry", "layer", "case_context", "validation"):
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))
        identity(self.as_dict())

    def as_dict(self):
        return asdict(self)

    @property
    def key(self):
        return identity(self.as_dict())
