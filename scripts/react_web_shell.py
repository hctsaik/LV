"""Loopback FastAPI adapter for LV's React web interface."""
from __future__ import annotations

import csv
import base64
import copy
import hashlib
import json
import logging
import math
import mimetypes
import os
import tempfile
import threading
import uuid
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from starlette.responses import JSONResponse, Response
from starlette.staticfiles import StaticFiles
from PIL import Image

from _utils import (
    available_models,
    load_model,
    load_text_encoder,
    supports_text_query,
    yaml_class_names,
)
from interaction import (
    crop_bbox,
    draw_yolo_boxes,
    find_duplicate_pairs_embedding,
    find_duplicate_pairs_phash,
    nearest_labels,
    rank_gap_fillers,
    reference_coverage,
    records_to_csv,
    snapshots_to_csv,
    zip_selected_images,
    yolo_label_path_for,
)
from curation_store import CurationStore, CurationStoreError
from fp_case_store import FalsePositiveCaseStore, public_case
from fp_external_models import ExternalModelRegistry, MAX_MODEL_BYTES
from fp_external_runtime import Runtime as ExternalOnnxRuntime, compare_reference as compare_external_reference
from fp_external_evaluation import export_training as export_fp_training
from fp_external_evaluation import paired_evaluation as run_fp_paired_evaluation
from fp_case_search import add_candidate as add_fp_search_candidate
from fp_case_search import model_recipe as fp_case_model_recipe
from fp_case_search import search_case as run_fp_case_search
from fp_case_report import report_json as fp_case_report_json
from fp_case_report import report_markdown as fp_case_report_markdown
from fp_case_report import set_reply_status as set_fp_case_reply_status
from fp_case_report import upsert_claim as upsert_fp_case_claim
from fp_case_review import apply_review_return as apply_fp_case_review_return
from fp_case_review import link_round as link_fp_case_review_round
from fp_case_review import prepare_batch as prepare_fp_case_review_batch
from fp_case_review import review_candidate as review_fp_case_candidate
from fp_case_review import review_counts as fp_case_review_counts
from fp_case_review import review_manifest as fp_case_review_manifest
from fp_case_review import review_zip as fp_case_review_zip
from completeness import (
    bucketize,
    build_completeness,
    categorical_buckets,
    cell_centroid,
    image_stats,
    mine_candidates,
)
from react_compare import ComparisonInputError, project_comparison, run_comparison
from react_coverage import (
    coverage_projection,
    coverage_sparsity,
    sparse_threshold,
)
from react_evaluation import (
    EvaluationInputError,
    run_detection_evaluation,
    run_detection_evaluation_demo,
)
from class_similarity import analyze as analyze_class_similarity
from class_similarity import build_box_summary, ClassSimilarityError
from class_similarity_export import evaluation_template as class_similarity_evaluation_template
from class_similarity_export import export_report as export_class_similarity_report
from class_similarity_records import canonical_path as canonical_class_similarity_path
from class_similarity_records import prepare_records as prepare_class_similarity_records
from class_similarity_visuals import jitter_x as class_similarity_jitter_x
from class_similarity_visuals import sample_scatter_rows as sample_class_similarity_scatter_rows
from class_similarity_threshold import (
    candidate_mask as class_similarity_candidate_mask,
    evaluate_threshold as evaluate_class_similarity_threshold,
    parse_policy as parse_class_similarity_policy,
    precision_recall_curve as class_similarity_precision_recall_curve,
    preview_key as class_similarity_preview_key,
    make_operating_point as make_class_similarity_operating_point,
    score_protocol_key as class_similarity_score_protocol_key,
    source_group_id as class_similarity_source_group_id,
    summarize_candidates as summarize_class_similarity_candidates,
    validate_evaluation_manifest as validate_class_similarity_manifest,
    validate_heldout as validate_class_similarity_heldout,
    workload_curve as class_similarity_workload_curve,
)
from react_visualize import JobCancelled, VisualizationInputError, run_image_visualization
from safe_io import safe_open_image

API_VERSION = "1"
logger = logging.getLogger("lv.react_web")


def _configured_api_port() -> int:
    """Use the explicit LV setting, then NativeApp's dynamically chosen port."""
    return int(os.environ.get("LV_WEB_API_PORT") or os.environ.get("PORT") or "8765")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ComparisonObjectPolicy(BaseModel):
    pad: float = Field(ge=0.0, le=0.5)
    target_res: Literal[168, 196, 224, 280, 336, 392, 448]
    head: Literal["cls", "meanpool"]


class CreateVisualizationJob(BaseModel):
    folders: list[str] = Field(min_length=1, max_length=32)
    folder_grant_ids: list[str | None] = Field(default_factory=list, max_length=32)
    mode: Literal["classifier", "detector"] = "classifier"
    unit: Literal["image", "object"] = "image"
    class_names: list[str] = Field(default_factory=list, max_length=256)
    object_policy: ComparisonObjectPolicy | None = None
    model_id: str | None = Field(default=None, min_length=1, max_length=128)
    method: str = "pca"
    model_ids: list[str] = Field(default_factory=list, max_length=8)
    methods: list[str] = Field(default_factory=list, max_length=5)
    umap_reference: bool = False
    umap_rebuild: bool = False


class CreateObjectPolicyAutotuneJob(BaseModel):
    folders: list[str] = Field(min_length=1, max_length=32)
    folder_grant_ids: list[str | None] = Field(default_factory=list, max_length=32)
    class_names: list[str] = Field(default_factory=list, max_length=256)
    model_id: str = Field(min_length=1, max_length=128)


class SaveObjectPolicyProfileRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)


class ObjectPolicyProfileMatchRequest(BaseModel):
    folders: list[str] = Field(min_length=1, max_length=32)
    folder_grant_ids: list[str | None] = Field(default_factory=list, max_length=32)
    class_names: list[str] = Field(default_factory=list, max_length=256)


class CreateClassSimilarityJob(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)
    k: int = Field(default=5, ge=1, le=20)
    analysis_split: str = Field(default="all", max_length=128)


class ClassSimilarityEvaluationTemplateRequest(BaseModel):
    scope: Literal["all", "selected"] = "all"
    purpose: Literal["tuning", "heldout"] = "tuning"
    selected_ids: list[str] = Field(default_factory=list, max_length=10000)


class ClassSimilarityEvaluationRequest(BaseModel):
    manifest: dict
    policy: dict = Field(default_factory=lambda: {"mode": "threshold", "threshold": 0.0})


class ClassSimilarityEvaluationPreviewRequest(BaseModel):
    policy: dict = Field(default_factory=lambda: {"mode": "threshold", "threshold": 0.0})


class ClassSimilarityReportRequest(BaseModel):
    scope: Literal["all", "candidates", "selected"] = "all"
    selected_ids: list[str] = Field(default_factory=list, max_length=10000)
    policy: dict = Field(default_factory=lambda: {"mode": "threshold", "threshold": 0.0})
    format: Literal["csv", "json"]


class ClassSimilarityOperatingPointRequest(BaseModel):
    policy: dict


class CreateDetectionEvaluationRequest(BaseModel):
    folders: list[str] = Field(default_factory=list, max_length=32)
    folder_grant_ids: list[str | None] = Field(default_factory=list, max_length=32)
    prediction_csv: str = Field(min_length=1)
    consensus_csv: str | None = None
    iou_thresh: float = Field(default=0.5, ge=0.1, le=0.9)
    conf_thresh: float = Field(default=0.0, ge=0.0, le=1.0)
    class_aware: bool = True


def _resolve_nativeapp_folder_grant(grant_id: str, app_id: str) -> Path:
    """Resolve a renderer's opaque Host grant through the public NativeApp API."""
    base = os.environ.get("NATIVEAPP_API", "").rstrip("/")
    parsed = urllib.parse.urlsplit(base)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise _folder_permission_error(
            503,
            "NATIVEAPP_HOST_UNAVAILABLE",
            "The NativeApp host connection is unavailable. Reopen the app from NativeApp and choose the folder again.",
        )
    body = json.dumps({"app_id": app_id, "grant_id": grant_id}).encode("utf-8")
    request = urllib.request.Request(
        base + "/host/grants/resolve",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise _folder_permission_error(
            403,
            "FOLDER_GRANT_REJECTED",
            "NativeApp rejected this folder permission. Choose the folder again.",
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _folder_permission_error(
            503,
            "NATIVEAPP_HOST_UNAVAILABLE",
            "The NativeApp host could not verify folder permission. Reopen the app and choose the folder again.",
        ) from exc
    if not isinstance(payload, dict) or payload.get("app_id") != app_id:
        raise _folder_permission_error(
            403,
            "FOLDER_GRANT_REJECTED",
            "NativeApp returned a folder permission for a different app.",
        )
    try:
        selected = Path(str(payload.get("path") or "")).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _folder_permission_error(
            403,
            "FOLDER_GRANT_REJECTED",
            "The granted folder is no longer available. Choose it again.",
        ) from exc
    if not selected.is_dir():
        raise _folder_permission_error(
            403,
            "FOLDER_GRANT_REJECTED",
            "The granted location is not a folder. Choose a folder again.",
        )
    return selected


def _folder_permission_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "request_id": uuid.uuid4().hex},
    )


def _resolve_nativeapp_folder_list(
    folders: list[str], folder_grant_ids: list[str | None],
) -> list[str]:
    app_id = os.environ.get("NATIVEAPP_APP_ID", "").strip()
    if not app_id:
        if any(folder_grant_ids):
            raise _folder_permission_error(
                403,
                "FOLDER_GRANT_REJECTED",
                "A NativeApp folder permission cannot be used outside its managed app session.",
            )
        return folders
    if len(folder_grant_ids) != len(folders):
        raise _folder_permission_error(
            403,
            "FOLDER_GRANT_REQUIRED",
            "Choose each image folder with the NativeApp folder picker before running the analysis.",
        )

    resolved: list[str] = []
    for submitted_path, grant_id in zip(folders, folder_grant_ids):
        if not grant_id:
            raise _folder_permission_error(
                403,
                "FOLDER_GRANT_REQUIRED",
                "Choose each image folder with the NativeApp folder picker before running the analysis.",
            )
        granted_path = _resolve_nativeapp_folder_grant(grant_id, app_id)
        try:
            displayed_path = Path(submitted_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise _folder_permission_error(
                403,
                "FOLDER_GRANT_MISMATCH",
                "The selected folder no longer matches its NativeApp permission. Choose it again.",
            ) from exc
        if (
            not displayed_path.is_dir()
            or os.path.normcase(os.path.normpath(str(displayed_path)))
            != os.path.normcase(os.path.normpath(str(granted_path)))
        ):
            raise _folder_permission_error(
                403,
                "FOLDER_GRANT_MISMATCH",
                "The selected folder no longer matches its NativeApp permission. Choose it again.",
            )
        resolved.append(str(granted_path))
    return resolved


def _resolve_request_folders(
    request: (
        CreateVisualizationJob
        | CreateObjectPolicyAutotuneJob
        | ObjectPolicyProfileMatchRequest
        | CreateDetectionEvaluationRequest
        | CreateCoverageJob
        | CreateCoverageCandidateJob
        | CoverageCompletenessCandidateRequest
    ),
) -> list[str]:
    return _resolve_nativeapp_folder_list(request.folders, request.folder_grant_ids)


def _resolve_comparison_folders(request: CreateComparisonJob) -> tuple[list[str], list[str]]:
    return (
        _resolve_nativeapp_folder_list(request.folders_a, request.folder_grant_ids_a),
        _resolve_nativeapp_folder_list(request.folders_b, request.folder_grant_ids_b),
    )


class CreateDiagnosticReferenceRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=128)
    record_index: int = Field(ge=0)
    model_id: str = Field(min_length=1, max_length=128)


class CreateDiagnosticHeadRequest(BaseModel):
    current_run_id: str = Field(min_length=1, max_length=128)
    resolution: Literal[224, 336, 448]


class CreateDiagnosticExplanationRequest(BaseModel):
    current_run_id: str = Field(min_length=1, max_length=128)
    resolution: Literal[224, 336, 448]
    method: Literal["attention", "gradcam"]
    attention_method: Literal["last_layer_head_mean", "single_head", "rollout"] = "last_layer_head_mean"
    head_index: int | None = Field(default=None, ge=0, le=63)
    target_class: str | None = Field(default=None, min_length=1, max_length=256)
    display_mode: Literal["signed", "positive"] = "signed"


class CreateFalsePositiveCaseRequest(BaseModel):
    source_name: str = Field(min_length=1, max_length=512)
    source_kind: Literal["original", "screenshot"]
    source_base64: str = Field(min_length=1, max_length=70_000_000)
    title: str = Field(default="", max_length=240)
    description: str = Field(default="", max_length=10000)
    external_model_id: str = Field(default="", max_length=240)
    threshold: str = Field(default="", max_length=120)
    raw_output: str = Field(default="", max_length=20000)


class AddFalsePositiveSourceRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    source_name: str = Field(min_length=1, max_length=512)
    source_kind: Literal["original", "screenshot"]
    source_base64: str = Field(min_length=1, max_length=70_000_000)


class AddFalsePositiveRegionRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    source_id: str = Field(min_length=1, max_length=64)
    roi: list[float] = Field(min_length=4, max_length=4)
    label: str = Field(default="", max_length=256)
    score: float | None = None


class ImportFalsePositivePredictionsRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    source_id: str = Field(min_length=1, max_length=64)
    filename: str = Field(min_length=1, max_length=512)
    content: str = Field(min_length=1, max_length=2_000_000)


class ReviewFalsePositiveRegionRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    verdict: Literal["pending", "normal", "defect", "disputed", "insufficient"]
    reviewer: str = Field(default="", max_length=256)
    reason: str = Field(default="", max_length=4000)


class CreateFalsePositiveSearchRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    item_id: str = Field(min_length=1, max_length=64)
    normal_folder: str = Field(default="", max_length=4096)
    defect_folder: str = Field(default="", max_length=4096)
    candidate_scope: Literal["whole_image", "same_roi"] = "whole_image"
    limit: int = Field(default=300, ge=1, le=5000)
    top_k: int = Field(default=6, ge=1, le=100)
    include_history: bool = True


class AddFalsePositiveSearchCandidateRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    category: Literal["normal", "defect", "history"]


class UpsertFalsePositiveClaimRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    claim_id: str = Field(default="", max_length=64)
    kind: Literal["fact", "hypothesis", "action"]
    text: str = Field(min_length=1, max_length=10_000)
    evidence_ids: list[str] = Field(min_length=1, max_length=100)


class SetFalsePositiveReplyStatusRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    status: Literal["draft", "replied"]


class ReviewFalsePositiveCandidateRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    verdict: Literal["pending", "normal", "defect", "disputed", "insufficient"]
    reviewer: str = Field(default="", max_length=256)
    reason: str = Field(default="", max_length=4000)


class PrepareFalsePositiveReviewBatchRequest(BaseModel):
    expected_revision: int = Field(ge=1)


class ReturnFalsePositiveReviewBatchRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    manifest: dict


class CreateExternalModelProfileRequest(BaseModel):
    name: str = Field(min_length=1, max_length=240)
    asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: dict


class CreateAnomalyModelJobRequest(BaseModel):
    train_folders: list[str] = Field(min_length=1, max_length=32)
    model_id: str = Field(min_length=1, max_length=128)
    score_mode: Literal["patch", "object"] = "patch"
    target_res: Literal[224, 336, 448] = 224
    sample_n: int = Field(default=64, ge=1, le=10000)
    label_semantic: Literal["object", "defect"] = "object"
    n_min: int = Field(default=8, ge=2, le=10000)
    object_source: Literal["yolo", "whole_image"] = "yolo"
    class_names: list[str] = Field(default_factory=list, max_length=256)
    confirmed: dict[str, Literal["good", "bad"]] = Field(default_factory=dict)


class CreateFewshotSampleBankJobRequest(BaseModel):
    sample_folders: list[str] = Field(min_length=1, max_length=32)
    model_id: str = Field(min_length=1, max_length=128)
    object_source: Literal["yolo", "whole_image"] = "yolo"
    class_names: list[str] = Field(default_factory=list, max_length=256)


class CreateFewshotScanJobRequest(BaseModel):
    sample_bank_id: str = Field(min_length=36, max_length=36)
    target_folders: list[str] = Field(min_length=1, max_length=32)
    k: int = Field(default=100, ge=10, le=500, multiple_of=10)
    min_proposal_conf: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)


class AddFewshotCandidatesRequest(BaseModel):
    sample_bank_id: str = Field(min_length=36, max_length=36)
    scan_id: str = Field(min_length=36, max_length=36)
    item_ids: list[str] = Field(min_length=1, max_length=500)
    theta: float = Field(default=0.3, ge=0.0, le=1.0, allow_inf_nan=False)
    class_overrides: dict[str, str] = Field(default_factory=dict, max_length=500)


class ExportFewshotCandidatesRequest(BaseModel):
    sample_bank_id: str = Field(min_length=36, max_length=36)
    scan_id: str = Field(min_length=36, max_length=36)
    accepted_item_ids: list[str] = Field(default_factory=list, max_length=500)
    theta: float = Field(default=0.3, ge=0.0, le=1.0, allow_inf_nan=False)
    class_overrides: dict[str, str] = Field(default_factory=dict, max_length=500)
    output_dir: str = Field(min_length=1, max_length=4096)
    copy_images: bool = True


class CreateFewshotMonitorWorkspaceRequest(BaseModel):
    sample_bank_id: str = Field(min_length=36, max_length=36)
    target_folders: list[str] = Field(min_length=1, max_length=32)
    workspace_dir: str = Field(min_length=1, max_length=4096)
    k: int = Field(default=100, ge=10, le=500, multiple_of=10)


class CreateFewshotMonitorScanRequest(BaseModel):
    monitor_workspace_id: str = Field(min_length=36, max_length=36)


class LabelFewshotMonitorItemRequest(BaseModel):
    item_id: str = Field(min_length=1, max_length=256)
    decision: Literal["defect", "skip"]
    label: str = Field(default="", max_length=128)


class CreateAnomalyScoreJobRequest(BaseModel):
    target_folders: list[str] = Field(min_length=1, max_length=32)
    contamination: float = Field(default=0.05, ge=0.0, le=1.0, allow_inf_nan=False)
    class_names: list[str] = Field(default_factory=list, max_length=256)


class ReapplyAnomalyRunRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    contamination: float = Field(default=0.05, ge=0.0, le=1.0, allow_inf_nan=False)
    confirmed: dict[str, Literal["good", "bad", ""]] = Field(default_factory=dict, max_length=10000)


class AnomalyPrelabelRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    min_conf: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    target_classes: list[str] | None = Field(default=None, max_length=256)


class AnomalyPrelabelPreviewRequest(AnomalyPrelabelRequest):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=40, ge=1, le=200)


class AnomalyPrelabelExportRequest(AnomalyPrelabelRequest):
    output_dir: str = Field(min_length=1, max_length=4096)


class LoadAnomalyModelRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=64)


class ReviseExternalModelProfileRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    name: str | None = Field(default=None, max_length=240)
    asset_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    config: dict | None = None


class RunExternalModelRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    source_id: str = Field(min_length=1, max_length=64)
    profile_id: str = Field(min_length=1, max_length=64)
    profile_revision: int = Field(default=1, ge=1)
    reference: dict | None = None
    score_atol: float = Field(default=0.001, ge=0.0, le=0.1)
    box_atol: float = Field(default=0.001, ge=0.0, le=0.1)


class RunFalsePositiveEvaluationRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    baseline_profile_id: str = Field(min_length=1, max_length=64)
    baseline_profile_revision: int = Field(default=1, ge=1)
    candidate_profile_id: str = Field(min_length=1, max_length=64)
    candidate_profile_revision: int = Field(default=1, ge=1)
    manifest: dict
    base_dir: str = Field(default="", max_length=4096)
    min_support: int = Field(default=10, ge=1, le=5000)
    max_miss_increase: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    min_fp_reduction: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)


class ExportFalsePositiveTrainingRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    complete_images: list[str] = Field(min_length=1, max_length=5000)
    class_assignments: dict[str, str] = Field(default_factory=dict, max_length=10000)
    class_names: list[str] = Field(min_length=1, max_length=256)
    reviewer: str = Field(min_length=1, max_length=256)
    reason: str = Field(min_length=1, max_length=4000)
    confirmed_complete: bool


class CreateComparisonJob(BaseModel):
    folders_a: list[str] = Field(min_length=1, max_length=32)
    folder_grant_ids_a: list[str | None] = Field(default_factory=list, max_length=32)
    folders_b: list[str] = Field(min_length=1, max_length=32)
    folder_grant_ids_b: list[str | None] = Field(default_factory=list, max_length=32)
    model_id: str = Field(min_length=1, max_length=128)
    unit: Literal["whole", "object"] = "object"
    class_names: list[str] = Field(default_factory=list, max_length=256)
    object_policy: ComparisonObjectPolicy | None = None


class CreateCoverageJob(BaseModel):
    folders: list[str] = Field(min_length=1, max_length=32)
    folder_grant_ids: list[str | None] = Field(default_factory=list, max_length=32)
    model_id: str = Field(min_length=1, max_length=128)
    unit: Literal["image", "object"] = "image"
    method: Literal["pca", "tsne", "umap", "lda", "sumap"] = "sumap"
    dimension: Literal[2, 3] = 2
    k: int = Field(default=10, ge=1, le=1000)
    object_policy: ComparisonObjectPolicy | None = None


class CreateGroupDiffJob(BaseModel):
    good_dirs: list[str] = Field(min_length=1, max_length=32)
    bad_dirs: list[str] = Field(min_length=1, max_length=32)
    model_id: str = Field(min_length=1, max_length=128)
    n_perm: Literal[100, 200, 500] = 200
    alpha: Literal[0.01, 0.05, 0.1] = 0.05
    top_k: int = Field(default=5, ge=1, le=10)
    seed: int = Field(default=0, ge=0, le=9999)


class GroupDiffExportRequest(BaseModel):
    output_dir: str = Field(min_length=1, max_length=4096)


class CoverageProjectionRequest(BaseModel):
    method: Literal["pca", "tsne", "umap", "lda", "sumap"]
    dimension: Literal[2, 3] = 2


class CoverageSparsityRequest(BaseModel):
    k: int = Field(ge=1, le=1000)


class CoverageCompletenessFrequency(BaseModel):
    x: int = Field(ge=0, le=8192)
    y: int = Field(ge=0, le=8192)
    frequency: Literal["高", "中", "低", "不適用"]


class CoverageCompletenessRequest(BaseModel):
    axis_x: Literal["label", "split", "brightness", "contrast", "sharpness", "aspect"] = "label"
    axis_y: Literal["label", "split", "brightness", "contrast", "sharpness", "aspect"] = "brightness"
    bins: int = Field(default=3, ge=2, le=8)
    t_abs: int = Field(default=10, ge=1, le=1_000_000)
    d_star: float = Field(default=0.6, ge=0.0, le=1.0)
    frequencies: list[CoverageCompletenessFrequency] = Field(default_factory=list, max_length=4096)


class CoverageCompletenessCandidateRequest(BaseModel):
    folders: list[str] = Field(min_length=1, max_length=32)
    folder_grant_ids: list[str | None] = Field(default_factory=list, max_length=32)
    axis_x: Literal["label", "split", "brightness", "contrast", "sharpness", "aspect"] = "label"
    axis_y: Literal["label", "split", "brightness", "contrast", "sharpness", "aspect"] = "brightness"
    bins: int = Field(default=3, ge=2, le=8)
    x: int = Field(ge=0, le=8192)
    y: int = Field(ge=0, le=8192)


class CoverageObjectViewRequest(BaseModel):
    class_label: str | None = Field(default=None, max_length=256)
    min_short_px: int = Field(default=0, ge=0, le=4096)
    confidence_min: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_max: float = Field(default=1.0, ge=0.0, le=1.0)
    method: Literal["pca", "tsne", "umap", "lda", "sumap"] = "pca"
    dimension: Literal[2, 3] = 2
    k: int = Field(default=10, ge=1, le=1000)
    block_k: int = Field(default=8, ge=3, le=20)


class CreateCoverageCandidateJob(BaseModel):
    folders: list[str] = Field(min_length=1, max_length=32)
    folder_grant_ids: list[str | None] = Field(default_factory=list, max_length=32)
    class_names: list[str] = Field(default_factory=list, max_length=256)


class CoverageCandidateViewRequest(BaseModel):
    role: Literal["pool", "reference"] = "pool"
    class_label: str | None = Field(default=None, max_length=256)
    min_short_px: int = Field(default=0, ge=0, le=4096)
    confidence_min: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_max: float = Field(default=1.0, ge=0.0, le=1.0)
    method: Literal["pca", "tsne", "umap", "lda", "sumap"] = "pca"
    dimension: Literal[2, 3] = 2
    k: int = Field(default=10, ge=1, le=1000)
    limit: int = Field(default=12, ge=1, le=60)


class AddCoverageCandidateCurationRequest(BaseModel):
    indices: list[int] = Field(min_length=1, max_length=10000)
    source: Literal["reference", "reference_class", "gap_filler"]
    scores: dict[int, float] = Field(default_factory=dict)
    reason: str = Field(default="", max_length=2000)


class ComparisonCurationSelection(BaseModel):
    group: Literal["A", "B"]
    index: int = Field(ge=0)


class AddComparisonCurationRequest(BaseModel):
    selections: list[ComparisonCurationSelection] = Field(min_length=1, max_length=10000)
    reason: str = Field(default="", max_length=2000)


class ProjectComparisonRequest(BaseModel):
    label: str = Field(min_length=1, max_length=256)
    method: str = "sumap"
    dimension: int = Field(default=2, ge=2, le=3)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    max_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    min_short_px: int = Field(default=0, ge=0, le=4096)


class KMeansBlocksRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)
    method: Literal["pca", "tsne", "umap", "lda", "sumap"]
    k: int = Field(default=12, ge=4, le=30)
    label_filter: str | None = Field(default=None, max_length=256)
    split_filter: str | None = Field(default=None, max_length=128)
    block_id: int | None = Field(default=None, ge=0, le=29)


class SimilarSearchRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)
    query_index: int = Field(ge=0)
    k: int = Field(default=9, ge=1, le=10000)


class TextSearchRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=512)
    k: int = Field(default=9, ge=1, le=10000)


class DuplicateScanRequest(BaseModel):
    method: Literal["phash", "embedding"] = "phash"
    model_id: str | None = Field(default=None, min_length=1, max_length=128)
    max_hamming: int = Field(default=4, ge=0, le=16)
    max_distance: float = Field(default=0.05, ge=0.0, le=0.5)
    cross_split_only: bool = False
    max_pairs: int = Field(default=200, ge=1, le=200)


class DuplicateLabelingHandoffRequest(DuplicateScanRequest):
    include_handled: bool = False
    new_round: bool = True
    round_id: str | None = Field(default=None, min_length=1, max_length=128)


class AddCurationRequest(BaseModel):
    indices: list[int] = Field(min_length=1, max_length=10000)
    source: str = Field(default="manual", min_length=1, max_length=128)
    scores: dict[int, float] = Field(default_factory=dict)
    reason: str = Field(default="", max_length=2000)


class AddAnomalyCurationRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    indices: list[int] = Field(min_length=1, max_length=10000)


class SetCurationReasonRequest(BaseModel):
    item_ids: list[str] = Field(min_length=1, max_length=10000)
    reason: str = Field(min_length=1, max_length=2000)


class CurationSubsetPreviewRequest(BaseModel):
    source: str = Field(default="all", max_length=128)
    destination: str = Field(min_length=1, max_length=4096)
    mode: Literal["copy", "symlink", "manifest-only"] = "copy"
    on_exists: Literal["skip", "rename", "overwrite"] = "skip"


class CurationSubsetExportRequest(CurationSubsetPreviewRequest):
    preview_token: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


class CurationHandoffRequest(BaseModel):
    item_ids: list[str] = Field(min_length=1, max_length=10000)
    include_handled: bool = False


class SendCurationHandoffRequest(CurationHandoffRequest):
    new_round: bool = True
    round_id: str | None = Field(default=None, min_length=1, max_length=128)


class AttachRoundMetricRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    value: float = Field(allow_inf_nan=False)
    source: str = Field(default="manual", min_length=1, max_length=128)
    note: str = Field(default="", max_length=2000)


class AttachEvaluationToRoundRequest(BaseModel):
    evaluation_id: str = Field(min_length=1, max_length=128)


class ApplyLabelingReadbackRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=128)


class RunRoundProbeRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=128)


@dataclass
class _Job:
    job_id: str
    request: CreateVisualizationJob
    state: str = "queued"
    phase: str = "queued"
    done: int = 0
    total: int = 0
    message: str = "Waiting to start"
    run_id: str | None = None
    error: dict | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class _ObjectPolicyAutotuneJob:
    job_id: str
    request: CreateObjectPolicyAutotuneJob
    state: str = "queued"
    phase: str = "queued"
    done: int = 0
    total: int = 0
    message: str = "Waiting to start"
    result: dict | None = None
    error: dict | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class _ClassSimilarityJob:
    job_id: str
    run_id: str
    request: CreateClassSimilarityJob
    state: str = "queued"
    done: int = 0
    total: int = 0
    result: dict | None = None
    error: dict | None = None
    prepared: object | None = None
    analysis: object | None = None
    source_hashes: dict[str, str] = field(default_factory=dict)
    evaluation: dict | None = None
    tuning_evaluation: dict | None = None
    frozen_policy: dict | None = None
    heldout_used_ids: set[str] = field(default_factory=set)
    heldout_used_groups: set[str] = field(default_factory=set)
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class _ComparisonJob:
    job_id: str
    request: CreateComparisonJob
    state: str = "queued"
    phase: str = "queued"
    done: int = 0
    total: int = 0
    message: str = "Waiting to start"
    run_id: str | None = None
    error: dict | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class _CoverageJob:
    job_id: str
    request: CreateCoverageJob
    state: str = "queued"
    phase: str = "queued"
    done: int = 0
    total: int = 0
    message: str = "Waiting to start"
    run_id: str | None = None
    error: dict | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class _GroupDiffJob:
    job_id: str
    request: CreateGroupDiffJob
    state: str = "queued"
    phase: str = "queued"
    done: int = 0
    total: int = 100
    message: str = "Waiting to start"
    run_id: str | None = None
    error: dict | None = None


@dataclass
class _CoverageCandidateJob:
    job_id: str
    run_id: str
    request: CreateCoverageCandidateJob
    state: str = "queued"
    phase: str = "queued"
    done: int = 0
    total: int = 0
    message: str = "Waiting to start"
    error: dict | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class _AnomalyModelJob:
    job_id: str
    request: CreateAnomalyModelJobRequest
    state: str = "queued"
    phase: str = "queued"
    done: int = 0
    total: int = 0
    message: str = "Waiting to start"
    model_id: str | None = None
    result: dict | None = None
    error: dict | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class _FewshotSampleBankJob:
    job_id: str
    request: CreateFewshotSampleBankJobRequest
    state: str = "queued"
    phase: str = "queued"
    done: int = 0
    total: int = 0
    message: str = "Waiting to start"
    result: dict | None = None
    error: dict | None = None


@dataclass
class _FewshotScanJob:
    job_id: str
    request: CreateFewshotScanJobRequest
    state: str = "queued"
    phase: str = "queued"
    done: int = 0
    total: int = 0
    message: str = "Waiting to start"
    result: dict | None = None
    error: dict | None = None


@dataclass
class _FewshotMonitorScanJob:
    job_id: str
    monitor_workspace_id: str
    state: str = "queued"
    phase: str = "queued"
    done: int = 0
    total: int = 0
    message: str = "Waiting to start"
    result: dict | None = None
    error: dict | None = None


@dataclass
class _AnomalyScoreJob:
    job_id: str
    model_id: str
    request: CreateAnomalyScoreJobRequest
    state: str = "queued"
    phase: str = "queued"
    done: int = 0
    total: int = 0
    message: str = "Waiting to start"
    run_id: str | None = None
    result: dict | None = None
    error: dict | None = None


def _job_view(
    job: (
        _Job
        | _ObjectPolicyAutotuneJob
        | _ComparisonJob
        | _CoverageJob
        | _CoverageCandidateJob
        | _GroupDiffJob
    ),
) -> dict:
    return {
        "job_id": job.job_id,
        "state": job.state,
        "phase": job.phase,
        "done": job.done,
        "total": job.total,
        "percent": (job.done / job.total * 100.0) if job.total else 0.0,
        "message": job.message,
        "run_id": getattr(job, "run_id", None),
        "result": getattr(job, "result", None),
        "error": job.error,
        "can_cancel": job.state in {"queued", "running", "cancelling"} and not isinstance(job, _GroupDiffJob),
    }


def _anomaly_job_view(job: _AnomalyModelJob) -> dict:
    return {
        "job_id": job.job_id,
        "state": job.state,
        "phase": job.phase,
        "done": job.done,
        "total": job.total,
        "percent": (job.done / job.total * 100.0) if job.total else 0.0,
        "message": job.message,
        "model_id": job.model_id,
        "result": job.result,
        "error": job.error,
        "can_cancel": False,
    }


def _fewshot_sample_bank_job_view(
    job: _FewshotSampleBankJob | _FewshotScanJob | _FewshotMonitorScanJob,
) -> dict:
    return {
        "job_id": job.job_id,
        "state": job.state,
        "phase": job.phase,
        "done": job.done,
        "total": job.total,
        "percent": (job.done / job.total * 100.0) if job.total else 0.0,
        "message": job.message,
        "result": job.result,
        "error": job.error,
        "can_cancel": False,
    }


def _anomaly_score_job_view(job: _AnomalyScoreJob) -> dict:
    return {
        "job_id": job.job_id,
        "state": job.state,
        "phase": job.phase,
        "done": job.done,
        "total": job.total,
        "percent": (job.done / job.total * 100.0) if job.total else 0.0,
        "message": job.message,
        "run_id": job.run_id,
        "result": job.result,
        "error": job.error,
        "can_cancel": False,
    }


def create_app(
    *,
    model_catalog: Callable[[], list[str]] | None = None,
    model_loader: Callable[[str], Callable] | None = None,
    text_model_loader: Callable[[str], Callable[[str], np.ndarray]] | None = None,
    object_model_loader: Callable | None = None,
    diagnostic_model_loader: Callable[[str, int, bool, str], Callable] | None = None,
    groupdiff_extractor: Callable | None = None,
    case_search_model_provider: Callable[[], tuple[dict, Callable]] | None = None,
    anomaly_model_builder: Callable | None = None,
    anomaly_scorer: Callable | None = None,
) -> FastAPI:
    catalog = model_catalog or available_models
    has_custom_model_loader = model_loader is not None
    has_custom_object_model_loader = object_model_loader is not None
    loader = model_loader or load_model
    text_loader = text_model_loader or load_text_encoder
    if object_model_loader is not None:
        object_loader = object_model_loader
    elif model_loader is None:
        object_loader = load_model
    else:
        # Test and host integrations often inject a simple one-argument loader;
        # it still exercises the exact crop and aggregation code without weights.
        object_loader = lambda model_id, **_kwargs: loader(model_id)
    app = FastAPI(title="VisualLatent Local API", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.jobs = {}
    app.state.object_policy_jobs = {}
    app.state.class_similarity_jobs = {}
    app.state.runs = {}
    app.state.current_run_id = None
    app.state.diagnostic_references = {}
    app.state.diagnostic_heads = {}
    app.state.raw_embeddings = {}
    app.state.ranking_scores = {}
    app.state.text_encoders = {}
    app.state.text_vectors = {}
    app.state.block_cache = {}
    app.state.comparison_jobs = {}
    app.state.comparison_runs = {}
    app.state.comparison_projections = {}
    app.state.coverage_jobs = {}
    app.state.coverage_runs = {}
    app.state.coverage_embeddings = {}
    app.state.coverage_sparsity = {}
    app.state.coverage_image_stats = {}
    app.state.coverage_completeness_candidates = {}
    app.state.coverage_candidate_jobs = {}
    app.state.coverage_candidates = {}
    app.state.groupdiff_jobs = {}
    app.state.groupdiff_runs = {}
    app.state.anomaly_jobs = {}
    app.state.anomaly_score_jobs = {}
    app.state.fewshot_sample_bank_jobs = {}
    app.state.fewshot_scan_jobs = {}
    app.state.fewshot_monitor_scan_jobs = {}
    app.state.fewshot_scans = {}
    app.state.fewshot_monitor_workspaces = {}
    app.state.fewshot_lock = threading.RLock()
    app.state.anomaly_runs = {}
    app.state.current_anomaly_run_id = None
    app.state.anomaly_models = {}
    app.state.current_anomaly_model_id = None
    app.state.anomaly_model_builder = anomaly_model_builder
    app.state.anomaly_scorer = anomaly_scorer
    app.state.curation_store = CurationStore()
    app.state.fp_case_store = FalsePositiveCaseStore()
    app.state.external_model_registry = ExternalModelRegistry(app.state.fp_case_store.root)
    app.state.case_search_model_provider = case_search_model_provider
    app.state.fp_case_lock = threading.RLock()
    app.state.lock = threading.RLock()
    app.state.evaluation_runs = {}

    def has_active_job() -> bool:
        jobs = (
            list(app.state.jobs.values())
            + list(app.state.object_policy_jobs.values())
            + list(app.state.class_similarity_jobs.values())
            + list(app.state.comparison_jobs.values())
            + list(app.state.coverage_jobs.values())
            + list(app.state.groupdiff_jobs.values())
            + list(app.state.coverage_candidate_jobs.values())
            + list(app.state.anomaly_jobs.values())
            + list(app.state.anomaly_score_jobs.values())
            + list(app.state.fewshot_sample_bank_jobs.values())
            + list(app.state.fewshot_scan_jobs.values())
            + list(app.state.fewshot_monitor_scan_jobs.values())
        )
        return any(job.state in {"queued", "running", "cancelling"} for job in jobs)

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and {"code", "message", "request_id"} <= exc.detail.keys():
            payload = exc.detail
        else:
            code = {404: "NOT_FOUND", 409: "STATE_CONFLICT", 422: "INVALID_REQUEST"}.get(
                exc.status_code, "LOCAL_API_ERROR"
            )
            payload = {
                "code": code,
                "message": str(exc.detail) if isinstance(exc.detail, str) else "The request could not be completed.",
                "request_id": uuid.uuid4().hex,
            }
        return JSONResponse(status_code=exc.status_code, content=payload, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={
            "code": "INVALID_REQUEST",
            "message": "The request does not match the local API contract.",
            "request_id": uuid.uuid4().hex,
        })

    @app.get("/api/v1/health")
    def health() -> dict:
        try:
            models_available = len(catalog())
            model_catalog_ready = True
        except Exception:  # a broken model directory must not leak a traceback
            models_available = 0
            model_catalog_ready = False
        return {
            "ok": True,
            "api_version": API_VERSION,
            "model_catalog_ready": model_catalog_ready,
            "models_available": models_available,
        }

    def get_fp_case(case_id: str) -> dict:
        try:
            return app.state.fp_case_store.load(case_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="找不到誤逮案件") from exc

    def check_fp_revision(case: dict, expected_revision: int) -> None:
        if case["revision"] != expected_revision:
            raise HTTPException(status_code=409, detail={
                "code": "CASE_REVISION_CONFLICT",
                "message": "案件已在另一個視窗更新，請重新載入後再儲存。",
                "request_id": uuid.uuid4().hex,
            })

    def save_fp_case(case: dict) -> dict:
        try:
            return public_case(app.state.fp_case_store.save(case))
        except ValueError as exc:
            if "另一個視窗更新" in str(exc):
                raise HTTPException(status_code=409, detail={
                    "code": "CASE_REVISION_CONFLICT",
                    "message": str(exc),
                    "request_id": uuid.uuid4().hex,
                }) from exc
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def decode_fp_image(source_base64: str) -> bytes:
        try:
            return base64.b64decode(source_base64, validate=True)
        except Exception as exc:
            raise HTTPException(status_code=422, detail="影像內容不是有效的 Base64") from exc

    @app.get("/api/v1/fp-cases")
    def list_fp_cases() -> dict:
        with app.state.fp_case_lock:
            cases = app.state.fp_case_store.list()
        rows = [public_case(case) for case in cases]
        return {"items": rows, "total": len(rows)}

    @app.post("/api/v1/fp-cases", status_code=201)
    def create_fp_case(request: CreateFalsePositiveCaseRequest) -> dict:
        try:
            source = app.state.fp_case_store.ingest(
                decode_fp_image(request.source_base64), request.source_name, request.source_kind,
            )
            with app.state.fp_case_lock:
                case = app.state.fp_case_store.create(
                    source,
                    title=request.title,
                    description=request.description,
                    model_id=request.external_model_id,
                    threshold=request.threshold,
                    raw_output=request.raw_output,
                )
            return public_case(case)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/fp-cases/{case_id}")
    def get_fp_case_route(case_id: str) -> dict:
        with app.state.fp_case_lock:
            return public_case(get_fp_case(case_id))

    @app.get("/api/v1/fp-cases/{case_id}/sources/{source_id}/image")
    def get_fp_case_source_image(case_id: str, source_id: str) -> Response:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            source = next((row for row in case["sources"] if row["id"] == source_id), None)
            if source is None:
                raise HTTPException(status_code=404, detail="找不到案件影像來源")
            try:
                path = app.state.fp_case_store.image_path(source)
                content = path.read_bytes()
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return Response(content=content, media_type=media_type, headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        })

    @app.post("/api/v1/fp-cases/{case_id}/sources", status_code=201)
    def add_fp_case_source(case_id: str, request: AddFalsePositiveSourceRequest) -> dict:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            check_fp_revision(case, request.expected_revision)
            try:
                source = app.state.fp_case_store.ingest(
                    decode_fp_image(request.source_base64), request.source_name, request.source_kind,
                )
                app.state.fp_case_store.add_source(case, source)
            except HTTPException:
                raise
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return save_fp_case(case)

    @app.post("/api/v1/fp-cases/{case_id}/regions", status_code=201)
    def add_fp_case_region(case_id: str, request: AddFalsePositiveRegionRequest) -> dict:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            check_fp_revision(case, request.expected_revision)
            try:
                item = app.state.fp_case_store.add_region(
                    case, request.source_id, request.roi, request.label, request.score,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            saved = save_fp_case(case)
            return {"case": saved, "item_id": item["id"]}

    @app.post("/api/v1/fp-cases/{case_id}/predictions/yolo", status_code=201)
    def import_fp_case_yolo(case_id: str, request: ImportFalsePositivePredictionsRequest) -> dict:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            check_fp_revision(case, request.expected_revision)
            try:
                count = app.state.fp_case_store.import_yolo(case, request.source_id, request.content)
                case["evidence"].append({
                    "id": "e_" + uuid.uuid4().hex[:12],
                    "role": "prediction_file",
                    "text": "匯入外部 YOLO 預測",
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "source_id": request.source_id,
                    "filename": Path(request.filename.replace("\\", "/")).name[:240],
                    "sha256": hashlib.sha256(request.content.encode("utf-8")).hexdigest(),
                    "region_count": count,
                })
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return {"case": save_fp_case(case), "imported": count}

    @app.patch("/api/v1/fp-cases/{case_id}/regions/{item_id}/review")
    def review_fp_case_region(
        case_id: str,
        item_id: str,
        request: ReviewFalsePositiveRegionRequest,
    ) -> dict:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            check_fp_revision(case, request.expected_revision)
            try:
                app.state.fp_case_store.review_region(
                    case, item_id, request.verdict, request.reviewer, request.reason,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return save_fp_case(case)

    @app.post("/api/v1/fp-cases/{case_id}/searches", status_code=201)
    def search_fp_case(case_id: str, request: CreateFalsePositiveSearchRequest) -> dict:
        with app.state.fp_case_lock:
            current = get_fp_case(case_id)
            check_fp_revision(current, request.expected_revision)
            case_snapshot = copy.deepcopy(current)
        try:
            if app.state.case_search_model_provider is not None:
                recipe, embed_fn = app.state.case_search_model_provider()
            else:
                recipe, models_root = fp_case_model_recipe()
                embed_fn = load_model(
                    "dinov2_vits14",
                    models_dir=models_root,
                    keep_aspect=True,
                    target_res=224,
                    head="cls",
                )
            result = run_fp_case_search(
                case_snapshot,
                request.item_id,
                app.state.fp_case_store,
                request.normal_folder,
                request.defect_folder,
                candidate_scope=request.candidate_scope,
                limit=request.limit,
                top_k=request.top_k,
                include_history=request.include_history,
                embed_fn=embed_fn,
                recipe=recipe,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # local weights/backend failures must not leak paths or traces
            logger.exception("LV false-positive case retrieval failed")
            raise HTTPException(
                status_code=503,
                detail="本機 DINOv2 外觀搜尋失敗；案件原有內容未變更。",
            ) from exc
        with app.state.fp_case_lock:
            latest = get_fp_case(case_id)
            check_fp_revision(latest, request.expected_revision)
            saved = save_fp_case(case_snapshot)
        return {"case": saved, "search_id": result["id"]}

    def fp_case_search_row(case: dict, search_id: str, result_id: str, category: str | None = None) -> dict:
        result = next((row for row in case.get("searches", []) if row["id"] == search_id), None)
        if result is None:
            raise HTTPException(status_code=404, detail="找不到案件搜尋紀錄")
        for group_name, rows in result.get("groups", {}).items():
            if category is not None and group_name != category:
                continue
            for row in rows:
                if row["id"] == result_id:
                    return row
        raise HTTPException(status_code=404, detail="找不到對照項目")

    @app.get("/api/v1/fp-cases/{case_id}/searches/{search_id}/results/{result_id}/image")
    def get_fp_case_search_image(case_id: str, search_id: str, result_id: str) -> Response:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            row = fp_case_search_row(case, search_id, result_id)
            source = row.get("source")
            if not isinstance(source, dict):
                raise HTTPException(status_code=404, detail="對照影像副本不存在")
            try:
                path = app.state.fp_case_store.image_path(source)
                content = path.read_bytes()
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return Response(content=content, media_type=media_type, headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        })

    @app.post("/api/v1/fp-cases/{case_id}/searches/{search_id}/results/{result_id}/candidate", status_code=201)
    def add_fp_case_search_candidate(
        case_id: str,
        search_id: str,
        result_id: str,
        request: AddFalsePositiveSearchCandidateRequest,
    ) -> dict:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            check_fp_revision(case, request.expected_revision)
            fp_case_search_row(case, search_id, result_id, request.category)
            try:
                added = add_fp_search_candidate(case, search_id, result_id, request.category)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            saved = save_fp_case(case) if added else public_case(case)
            return {"case": saved, "added": added}

    @app.put("/api/v1/fp-cases/{case_id}/claims")
    def save_fp_case_claim_route(case_id: str, request: UpsertFalsePositiveClaimRequest) -> dict:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            check_fp_revision(case, request.expected_revision)
            try:
                claim = upsert_fp_case_claim(
                    case,
                    request.kind,
                    request.text,
                    request.evidence_ids,
                    request.claim_id,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return {"case": save_fp_case(case), "claim_id": claim["id"]}

    @app.patch("/api/v1/fp-cases/{case_id}/reply-status")
    def update_fp_case_reply_status(case_id: str, request: SetFalsePositiveReplyStatusRequest) -> dict:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            check_fp_revision(case, request.expected_revision)
            try:
                changed = set_fp_case_reply_status(case, request.status)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            saved = save_fp_case(case) if changed else public_case(case)
            return {"case": saved, "changed": changed}

    @app.get("/api/v1/fp-cases/{case_id}/report.md")
    def download_fp_case_markdown(case_id: str) -> Response:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            content = fp_case_report_markdown(case).encode("utf-8")
        return Response(content, media_type="text/markdown; charset=utf-8", headers={
            "Content-Disposition": f'attachment; filename="{case_id}.md"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        })

    @app.get("/api/v1/fp-cases/{case_id}/report.json")
    def download_fp_case_json(case_id: str) -> Response:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            content = fp_case_report_json(case).encode("utf-8")
        return Response(content, media_type="application/json", headers={
            "Content-Disposition": f'attachment; filename="{case_id}.json"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        })

    @app.patch("/api/v1/fp-cases/{case_id}/candidates/{candidate_id}/review")
    def review_fp_case_candidate_route(
        case_id: str,
        candidate_id: str,
        request: ReviewFalsePositiveCandidateRequest,
    ) -> dict:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            check_fp_revision(case, request.expected_revision)
            try:
                review_fp_case_candidate(case, candidate_id, request.verdict, request.reviewer, request.reason)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return save_fp_case(case)

    @app.get("/api/v1/fp-cases/{case_id}/candidates/{candidate_id}/image")
    def get_fp_case_candidate_image(case_id: str, candidate_id: str) -> Response:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            candidate = next((row for row in case.get("candidates", []) if row.get("id") == candidate_id), None)
            if candidate is None:
                raise HTTPException(status_code=404, detail="找不到覆核候選")
            try:
                path = app.state.fp_case_store.image_path(candidate["source"])
                content = path.read_bytes()
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return Response(content, media_type=media_type, headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        })

    @app.post("/api/v1/fp-cases/{case_id}/batches", status_code=201)
    def prepare_fp_case_review_batch_route(case_id: str, request: PrepareFalsePositiveReviewBatchRequest) -> dict:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            check_fp_revision(case, request.expected_revision)
            try:
                batch = prepare_fp_case_review_batch(case)
                round_id = link_fp_case_review_round(case, batch["id"])
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            saved = save_fp_case(case)
            return {
                "case": saved,
                "batch_id": batch["id"],
                "round_id": round_id,
                "counts": fp_case_review_counts(saved["candidates"]),
            }

    @app.get("/api/v1/fp-cases/{case_id}/batches/{batch_id}/manifest")
    def download_fp_case_review_manifest(case_id: str, batch_id: str) -> Response:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            try:
                manifest = fp_case_review_manifest(case, batch_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            content = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        return Response(content, media_type="application/json", headers={
            "Content-Disposition": f'attachment; filename="{batch_id}.json"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        })

    @app.get("/api/v1/fp-cases/{case_id}/batches/{batch_id}/archive.zip")
    def download_fp_case_review_zip(case_id: str, batch_id: str) -> Response:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            try:
                manifest = fp_case_review_manifest(case, batch_id)
                content = fp_case_review_zip(app.state.fp_case_store, case, manifest)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(content, media_type="application/zip", headers={
            "Content-Disposition": f'attachment; filename="{batch_id}.zip"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        })

    @app.post("/api/v1/fp-cases/{case_id}/batches/{batch_id}/return")
    def read_fp_case_review_return(
        case_id: str,
        batch_id: str,
        request: ReturnFalsePositiveReviewBatchRequest,
    ) -> dict:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            check_fp_revision(case, request.expected_revision)
            try:
                staged, applied = apply_fp_case_review_return(case, batch_id, request.manifest)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return {"case": save_fp_case(staged), "applied": applied,
                    "counts": fp_case_review_counts(staged["candidates"]), "training_ready_images": 0}

    @app.post("/api/v1/external-model-assets", status_code=201)
    async def upload_external_model_asset(filename: str, request: Request) -> dict:
        """Stream ONNX bytes to local content-addressed storage without JSON/base64 expansion."""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_MODEL_BYTES:
                    raise HTTPException(status_code=413, detail="ONNX 檔案超過 1 GiB 匯入上限")
            except ValueError as exc:
                if isinstance(exc, HTTPException):
                    raise
                raise HTTPException(status_code=400, detail="Content-Length 格式無效") from exc
        registry = app.state.external_model_registry
        staged = None
        try:
            fd, staged_name = tempfile.mkstemp(prefix=".upload-", suffix=".tmp", dir=registry.assets)
            staged = Path(staged_name)
            total = 0
            with os.fdopen(fd, "wb") as stream:
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > MAX_MODEL_BYTES:
                        raise HTTPException(status_code=413, detail="ONNX 檔案超過 1 GiB 匯入上限")
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            asset = registry.ingest_file(staged, filename)
            return {key: asset[key] for key in ("sha256", "name", "format", "size", "created_at")}
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)

    @app.get("/api/v1/external-model-profiles")
    def list_external_model_profiles() -> dict:
        return {"items": app.state.external_model_registry.list()}

    @app.post("/api/v1/external-model-profiles", status_code=201)
    def create_external_model_profile(request: CreateExternalModelProfileRequest) -> dict:
        try:
            profile = app.state.external_model_registry.create(
                request.name, request.asset_sha256, request.config,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return profile

    @app.patch("/api/v1/external-model-profiles/{profile_id}")
    def revise_external_model_profile(
        profile_id: str,
        request: ReviseExternalModelProfileRequest,
    ) -> dict:
        try:
            return app.state.external_model_registry.revise(
                profile_id,
                request.expected_revision,
                name=request.name,
                asset_sha256=request.asset_sha256,
                config=request.config,
            )
        except ValueError as exc:
            status = 404 if "找不到" in str(exc) else 409 if "另一個視窗" in str(exc) else 422
            raise HTTPException(status_code=status, detail=str(exc)) from exc

    @app.post("/api/v1/fp-cases/{case_id}/external-runs", status_code=201)
    def run_external_model(case_id: str, request: RunExternalModelRequest) -> dict:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            check_fp_revision(case, request.expected_revision)
            source = next((row for row in case["sources"] if row["id"] == request.source_id), None)
            if source is None:
                raise HTTPException(status_code=404, detail="找不到案件影像來源")
            case_snapshot = copy.deepcopy(case)
            source_snapshot = copy.deepcopy(source)
        if request.reference is not None and len(json.dumps(request.reference, ensure_ascii=False).encode("utf-8")) > 2_000_000:
            raise HTTPException(status_code=413, detail="歷史部署紀錄不可超過 2 MB")
        registry = app.state.external_model_registry
        try:
            profile = registry.get(request.profile_id, request.profile_revision)
            config = registry.runtime_config(request.profile_id, request.profile_revision)
            source_bytes = app.state.fp_case_store.image_path(source_snapshot).read_bytes()
            runtime = ExternalOnnxRuntime(config)
            run = runtime.infer(
                source_bytes,
                app.state.fp_case_store.root / "external" / case_id,
            )
            parity = compare_external_reference(
                run,
                request.reference,
                source_snapshot["kind"],
                score_atol=request.score_atol,
                box_atol=request.box_atol,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("LV external ONNX inference failed (case=%s)", case_id)
            raise HTTPException(status_code=503, detail={
                "code": "EXTERNAL_MODEL_FAILED",
                "message": "外部模型執行失敗；請確認本機 ONNX Runtime 與模型契約後重試。",
                "request_id": uuid.uuid4().hex,
            }) from exc

        safe_config = {key: value for key, value in runtime.config.items() if key != "model_path"}
        run.update({
            "case_id": case_id,
            "source_id": source_snapshot["id"],
            "source_context_id": source_snapshot["sha256"],
            "source_kind": source_snapshot["kind"],
            "profile_id": profile["id"],
            "profile_revision": profile["revision"],
            "profile_name": profile["name"],
            "profile_asset_sha256": profile["asset"]["sha256"],
            "config": safe_config,
            "parity": parity,
        })
        if request.reference is not None:
            run["historical_reference"] = copy.deepcopy(request.reference)
        evidence_id = "e_" + uuid.uuid4().hex[:12]
        run["evidence_id"] = evidence_id
        case_snapshot.setdefault("external_runs", []).append(run)
        case_snapshot.setdefault("evidence", []).append({
            "id": evidence_id,
            "role": "external_inference",
            "text": "外部 ONNX 本機推論與完整歷史輸出比對",
            "at": run["at"],
            "source_id": source_snapshot["id"],
            "run_id": run["id"],
            "contract_id": run["contract_id"],
            "parity_status": parity["status"],
        })
        run_root = Path(run["artifact_dir"])
        (run_root / "case_run.json").write_text(
            json.dumps(public_case(run), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        with app.state.fp_case_lock:
            latest = get_fp_case(case_id)
            check_fp_revision(latest, request.expected_revision)
            current_source = next((row for row in latest["sources"] if row["id"] == request.source_id), None)
            if current_source is None or current_source["sha256"] != source_snapshot["sha256"]:
                raise HTTPException(status_code=409, detail="案件影像已變更，請重新載入後重跑")
            saved = save_fp_case(case_snapshot)
        saved_run = next(row for row in saved["external_runs"] if row["id"] == run["id"])
        return {"case": saved, "run": saved_run}

    @app.post("/api/v1/fp-cases/{case_id}/evaluations", status_code=201)
    def evaluate_fp_case_models(
        case_id: str, request: RunFalsePositiveEvaluationRequest
    ) -> dict:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            check_fp_revision(case, request.expected_revision)
            case_snapshot = copy.deepcopy(case)
        registry = app.state.external_model_registry
        try:
            baseline_profile = registry.get(
                request.baseline_profile_id, request.baseline_profile_revision
            )
            candidate_profile = registry.get(
                request.candidate_profile_id, request.candidate_profile_revision
            )
            result = run_fp_paired_evaluation(
                case_snapshot,
                app.state.fp_case_store,
                registry.runtime_config(
                    request.baseline_profile_id, request.baseline_profile_revision
                ),
                registry.runtime_config(
                    request.candidate_profile_id, request.candidate_profile_revision
                ),
                request.manifest,
                base_dir=request.base_dir or None,
                min_support=request.min_support,
                max_miss_increase=request.max_miss_increase,
                min_fp_reduction=request.min_fp_reduction,
                baseline_profile={
                    "id": baseline_profile["id"],
                    "revision": baseline_profile["revision"],
                    "name": baseline_profile["name"],
                    "asset_sha256": baseline_profile["asset"]["sha256"],
                },
                candidate_profile={
                    "id": candidate_profile["id"],
                    "revision": candidate_profile["revision"],
                    "name": candidate_profile["name"],
                    "asset_sha256": candidate_profile["asset"]["sha256"],
                },
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("LV paired external-model evaluation failed (case=%s)", case_id)
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "EXTERNAL_EVALUATION_FAILED",
                    "message": "固定資料集評估失敗；請確認模型、資料集與本機執行環境後重試。",
                    "request_id": uuid.uuid4().hex,
                },
            ) from exc
        with app.state.fp_case_lock:
            latest = get_fp_case(case_id)
            check_fp_revision(latest, request.expected_revision)
            saved = save_fp_case(case_snapshot)
        saved_result = next(
            row for row in saved["evaluations"] if row["id"] == result["id"]
        )
        return {"case": saved, "evaluation": saved_result}

    @app.post("/api/v1/fp-cases/{case_id}/training-exports", status_code=201)
    def create_fp_training_export(
        case_id: str, request: ExportFalsePositiveTrainingRequest
    ) -> dict:
        if request.confirmed_complete is not True:
            raise HTTPException(
                status_code=422,
                detail="訓練交接需要明確確認整張影像與標註完整性",
            )
        if len(request.complete_images) != len(set(request.complete_images)):
            raise HTTPException(status_code=422, detail="完整影像清單不可重複")
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            check_fp_revision(case, request.expected_revision)
            case_snapshot = copy.deepcopy(case)
        try:
            record, payload = export_fp_training(
                case_snapshot,
                app.state.fp_case_store,
                complete_images=request.complete_images,
                class_assignments=request.class_assignments,
                class_names=request.class_names,
                reviewer=request.reviewer,
                reason=request.reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        training_root = (app.state.fp_case_store.root / "training").resolve()
        export_root = (training_root / case_id).resolve()
        if not export_root.is_relative_to(training_root):
            raise HTTPException(status_code=422, detail="案件訓練交接位置無效")
        export_root.mkdir(parents=True, exist_ok=True)
        archive_path = (export_root / record["zip_file"]).resolve()
        if not archive_path.is_relative_to(export_root):
            raise HTTPException(status_code=422, detail="訓練交接檔案名稱無效")
        staged_path = archive_path.with_name("." + archive_path.name + ".tmp")
        try:
            staged_path.write_bytes(payload)
            os.replace(staged_path, archive_path)
            with app.state.fp_case_lock:
                latest = get_fp_case(case_id)
                check_fp_revision(latest, request.expected_revision)
                saved = save_fp_case(case_snapshot)
        except HTTPException:
            archive_path.unlink(missing_ok=True)
            raise
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise
        finally:
            staged_path.unlink(missing_ok=True)
        saved_record = next(
            row
            for row in saved["training_exports"]
            if row["id"] == record["id"]
        )
        return {"case": saved, "training_export": saved_record}

    @app.get(
        "/api/v1/fp-cases/{case_id}/training-exports/{export_id}/archive.zip"
    )
    def download_fp_training_export(case_id: str, export_id: str) -> Response:
        with app.state.fp_case_lock:
            case = get_fp_case(case_id)
            record = next(
                (
                    row
                    for row in case.get("training_exports", [])
                    if row.get("id") == export_id
                    and row.get("zip_file") == export_id + ".zip"
                ),
                None,
            )
            if record is None:
                raise HTTPException(status_code=404, detail="找不到訓練交接 ZIP")
            training_root = (app.state.fp_case_store.root / "training").resolve()
            export_root = (training_root / case_id).resolve()
            if not export_root.is_relative_to(training_root):
                raise HTTPException(status_code=409, detail="案件訓練交接位置無效")
            archive_path = (export_root / record["zip_file"]).resolve()
            if not archive_path.is_relative_to(export_root) or not archive_path.is_file():
                raise HTTPException(status_code=404, detail="訓練交接 ZIP 不存在")
            payload = archive_path.read_bytes()
        return Response(
            payload,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{export_id}.zip"',
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    def get_external_run(case_id: str, run_id: str) -> tuple[dict, dict, Path]:
        case = get_fp_case(case_id)
        run = next((row for row in case.get("external_runs", []) if row.get("id") == run_id), None)
        if run is None:
            raise HTTPException(status_code=404, detail="找不到外部推論紀錄")
        case_root = (app.state.fp_case_store.root / "external" / case_id).resolve()
        run_root = Path(str(run.get("artifact_dir", ""))).resolve()
        if not run_root.is_relative_to(case_root):
            raise HTTPException(status_code=409, detail="推論證據位置無效")
        return case, run, run_root

    @app.get("/api/v1/fp-cases/{case_id}/external-runs/{run_id}/input-preview.png")
    def get_external_model_input_preview(case_id: str, run_id: str) -> Response:
        with app.state.fp_case_lock:
            _case, _run, run_root = get_external_run(case_id, run_id)
            path = run_root / "model_input.png"
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise HTTPException(status_code=404, detail="模型輸入預覽不存在") from exc
        return Response(content, media_type="image/png", headers={
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
        })

    @app.get("/api/v1/fp-cases/{case_id}/external-runs/{run_id}/evidence.json")
    def download_external_model_evidence_json(case_id: str, run_id: str) -> Response:
        with app.state.fp_case_lock:
            _case, run, _run_root = get_external_run(case_id, run_id)
            content = json.dumps(public_case(run), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        return Response(content, media_type="application/json", headers={
            "Content-Disposition": f'attachment; filename="{run_id}.json"',
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
        })

    @app.get("/api/v1/fp-cases/{case_id}/external-runs/{run_id}/evidence.zip")
    def download_external_model_evidence_zip(case_id: str, run_id: str) -> Response:
        with app.state.fp_case_lock:
            _case, run, run_root = get_external_run(case_id, run_id)
            output = BytesIO()
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("evidence.json", json.dumps(public_case(run), ensure_ascii=False, indent=2, allow_nan=False))
                for name in ("model_input.png", "source.png", "tensors.npz"):
                    path = run_root / name
                    if path.is_file():
                        archive.write(path, name)
            content = output.getvalue()
        return Response(content, media_type="application/zip", headers={
            "Content-Disposition": f'attachment; filename="{run_id}.zip"',
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
        })

    @app.get("/api/v1/models")
    def models() -> dict:
        try:
            names = catalog()
        except Exception as exc:
            raise HTTPException(status_code=503, detail={
                "code": "MODEL_CATALOG_UNAVAILABLE",
                "message": "The local model catalog could not be read.",
                "request_id": uuid.uuid4().hex,
            }) from exc
        return {"items": [{"model_id": name, "name": name, "available": True} for name in names]}

    def fewshot_sample_bank_path(sample_bank_id: str) -> Path:
        if (len(sample_bank_id) != 36 or not sample_bank_id.startswith("fsb_")
                or any(char not in "0123456789abcdef" for char in sample_bank_id[4:])):
            raise HTTPException(status_code=404, detail="找不到此樣本集")
        root = (app.state.curation_store.path.parent / "fewshot_sample_banks").resolve()
        target = (root / sample_bank_id).resolve()
        if target.parent != root:
            raise HTTPException(status_code=404, detail="找不到此樣本集")
        return target

    def fewshot_sample_bank_view(sample_bank_id: str, bank: dict) -> dict:
        from collections import Counter

        labels = [str(label) for label in np.asarray(bank["labels"]).tolist()]
        vectors = np.asarray(bank["vectors"])
        return {
            "sample_bank_id": sample_bank_id,
            "model_id": str(bank["model"]),
            "target_res": int(bank["target_res"]),
            "object_source": str(bank.get("object_source") or "yolo"),
            "sample_count": len(labels),
            "class_counts": dict(sorted(Counter(labels).items())),
            "feature_dimension": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        }

    def execute_fewshot_sample_bank_job(job: _FewshotSampleBankJob) -> None:
        try:
            with app.state.lock:
                job.state, job.phase = "running", "embedding"
                job.message = "正在讀取樣本影像並準備特徵器"
                job.total = 100

            embedder: Callable | None = None

            def embed_fn(crop):
                nonlocal embedder
                if embedder is None:
                    if has_custom_object_model_loader or not has_custom_model_loader:
                        embedder = object_loader(
                            job.request.model_id, keep_aspect=True, target_res=224, head="cls",
                        )
                    else:
                        embedder = loader(job.request.model_id)
                return embedder(crop)

            def update_progress(done: int, total: int) -> None:
                with app.state.lock:
                    job.phase = "embedding"
                    job.done = max(0, int(done))
                    job.total = max(1, int(total))
                    job.message = f"正在建立樣本特徵 {job.done}/{job.total}"

            import sample_bank
            bank = sample_bank.build_sample_bank(
                [Path(folder) for folder in job.request.sample_folders],
                model=job.request.model_id,
                target_res=224,
                class_names=job.request.class_names or None,
                object_source=job.request.object_source,
                embed_fn=embed_fn,
                progress=update_progress,
            )
            sample_bank_id = "fsb_" + uuid.uuid4().hex
            bank_dir = fewshot_sample_bank_path(sample_bank_id)
            with app.state.lock:
                job.phase, job.message = "saving", "正在保存樣本集至 LV_CACHE_DIR"
            sample_bank.save_sample_bank(bank_dir, bank)
            result = fewshot_sample_bank_view(sample_bank_id, bank)
            with app.state.lock:
                job.result = result
                job.done = job.total = 100
                job.state, job.phase, job.message = "completed", "completed", "樣本集已建立並保存"
        except FileNotFoundError:
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "MODEL_UNAVAILABLE"
                job.error = {
                    "code": "MODEL_UNAVAILABLE",
                    "message": "所選本機特徵模型權重無法載入。",
                    "request_id": job.job_id,
                }
        except ValueError:
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "INVALID_SAMPLE_DATA"
                job.error = {
                    "code": "INVALID_SAMPLE_DATA",
                    "message": "找不到可用樣本；請確認影像格式、YOLO labels，或改選整張影像模式。",
                    "request_id": job.job_id,
                }
        except Exception:
            logger.exception("LV few-shot sample bank build failed (job=%s)", job.job_id)
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "SAMPLE_BANK_BUILD_FAILED"
                job.error = {
                    "code": "SAMPLE_BANK_BUILD_FAILED",
                    "message": "建立樣本集失敗。請檢查本機模型權重、資料格式與 worker 記錄。",
                    "request_id": job.job_id,
                }

    @app.post("/api/v1/fewshot/sample-banks/jobs", status_code=202)
    def create_fewshot_sample_bank_job(request: CreateFewshotSampleBankJobRequest) -> dict:
        try:
            available = set(catalog())
        except Exception as exc:
            raise HTTPException(status_code=503, detail="無法讀取本機模型清單") from exc
        if request.model_id not in available:
            raise HTTPException(status_code=422, detail="所選本機特徵模型不存在或尚未安裝")
        folders = []
        for value in request.sample_folders:
            try:
                folder = Path(value).expanduser().resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise HTTPException(status_code=422, detail="樣本資料夾不存在或無法讀取") from exc
            if not folder.is_dir():
                raise HTTPException(status_code=422, detail="樣本資料夾不存在或無法讀取")
            folders.append(str(folder))
        normalized = request.model_copy(update={"sample_folders": folders})
        job_id = uuid.uuid4().hex
        job = _FewshotSampleBankJob(job_id=job_id, request=normalized)
        with app.state.lock:
            if has_active_job():
                raise HTTPException(status_code=409, detail={
                    "code": "JOB_ALREADY_RUNNING",
                    "message": "請等目前分析完成後再建立樣本集。",
                    "request_id": job_id,
                })
            app.state.fewshot_sample_bank_jobs[job_id] = job
        threading.Thread(
            target=execute_fewshot_sample_bank_job,
            args=(job,),
            daemon=True,
            name=f"lv-fewshot-{job_id[:8]}",
        ).start()
        return {"job_id": job_id, "state": job.state}

    @app.get("/api/v1/fewshot/jobs/{job_id}")
    def get_fewshot_sample_bank_job(job_id: str) -> dict:
        with app.state.lock:
            job = (app.state.fewshot_sample_bank_jobs.get(job_id)
                   or app.state.fewshot_scan_jobs.get(job_id)
                   or app.state.fewshot_monitor_scan_jobs.get(job_id))
            if job is None:
                raise HTTPException(status_code=404, detail="找不到以樣搜樣工作")
            return _fewshot_sample_bank_job_view(job)

    @app.get("/api/v1/fewshot/sample-banks/{sample_bank_id}")
    def get_fewshot_sample_bank(sample_bank_id: str) -> dict:
        bank_dir = fewshot_sample_bank_path(sample_bank_id)
        if not bank_dir.is_dir():
            raise HTTPException(status_code=404, detail="找不到此樣本集")
        try:
            import sample_bank
            bank = sample_bank.load_sample_bank(bank_dir)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail="樣本集損毀或格式不相容") from exc
        return {"sample_bank": fewshot_sample_bank_view(sample_bank_id, bank)}

    def fewshot_scan_public_records(records: list[dict]) -> list[dict]:
        output = []
        for index, record in enumerate(records):
            output.append({
                "index": index,
                "item_id": str(record["item_id"]),
                "filename": Path(record["image_path"]).name,
                "bbox": [float(value) for value in record["bbox"]],
                "obj_index": int(record.get("obj_index") or 0),
                "label": str(record.get("label") or ""),
                "suggested_class": str(record.get("suggested_class") or ""),
                "similarity": float(record.get("similarity") or 0.0),
                "reason": str(record.get("reason") or ""),
            })
        return output

    def execute_fewshot_scan_job(
        job: _FewshotScanJob, bank: dict, roots: list[Path], image_paths: list[Path],
        class_names: list[str] | None,
    ) -> None:
        try:
            with app.state.lock:
                job.state, job.phase = "running", "scanning"
                job.total = len(image_paths)
                job.message = f"正在海掃 {job.total} 張影像"

            embedder: Callable | None = None

            def embed_fn(crop):
                nonlocal embedder
                if embedder is None:
                    if has_custom_object_model_loader or not has_custom_model_loader:
                        embedder = object_loader(
                            bank["model"], keep_aspect=True,
                            target_res=int(bank["target_res"]), head="cls",
                        )
                    else:
                        embedder = loader(bank["model"])
                return embedder(crop)

            def update_progress(data: dict) -> None:
                with app.state.lock:
                    job.phase = "scanning"
                    job.done = max(0, int(data.get("images_processed") or 0))
                    job.total = max(1, int(data.get("images_total") or len(image_paths)))
                    scored = int(data.get("objects_scored") or 0)
                    job.message = f"海掃 {job.done}/{job.total} 張影像 · 已評分 {scored} 個候選"

            from object_eval import dataset_cache_dir
            import al_batch

            confidence_key = round(float(job.request.min_proposal_conf) * 1000)
            checkpoint_dir = dataset_cache_dir(
                roots[0],
                f"fewshot_retrieve_{job.request.sample_bank_id}_k{job.request.k}_c{confidence_key}",
            )
            result = al_batch.run_batched(
                image_paths,
                model_dir=None,
                feature_extractor={"model": bank["model"], "target_res": bank["target_res"]},
                checkpoint_dir=checkpoint_dir,
                objective="retrieve",
                ref_vectors=bank["vectors"],
                ref_labels=[str(label) for label in bank["labels"]],
                k=int(job.request.k),
                object_source=str(bank.get("object_source") or "yolo"),
                class_names=class_names,
                dataset_dirs=roots,
                embed_fn=embed_fn,
                progress=update_progress,
                resume=True,
                on_identity_mismatch="restart",
                min_proposal_conf=float(job.request.min_proposal_conf),
                batch_size=20,
            )
            records_by_id = {}
            for record in result.get("topk_records") or []:
                records_by_id.setdefault(str(record.get("item_id") or ""), record)
            records = [record for item_id, record in records_by_id.items() if item_id]
            scan_id = "fss_" + uuid.uuid4().hex
            public_records = fewshot_scan_public_records(records)
            with app.state.fewshot_lock:
                app.state.fewshot_scans[scan_id] = {
                    "sample_bank_id": job.request.sample_bank_id,
                    "model_id": str(bank["model"]),
                    "target_res": int(bank["target_res"]),
                    "object_source": str(bank.get("object_source") or "yolo"),
                    "target_folders": [str(root) for root in roots],
                    "records": records,
                }
            with app.state.lock:
                job.result = {
                    "scan_id": scan_id,
                    "sample_bank_id": job.request.sample_bank_id,
                    "model_id": str(bank["model"]),
                    "target_res": int(bank["target_res"]),
                    "object_source": str(bank.get("object_source") or "yolo"),
                    "k": int(job.request.k),
                    "images_total": int(result.get("images_total") or len(image_paths)),
                    "images_processed": int(result.get("images_processed") or 0),
                    "objects_scored": int(result.get("objects_scored") or 0),
                    "candidate_count": len(public_records),
                    "done": bool(result.get("done")),
                    "candidates": public_records,
                }
                job.done = job.total = len(image_paths)
                job.state, job.phase, job.message = "completed", "completed", "海掃完成"
        except FileNotFoundError:
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "MODEL_UNAVAILABLE"
                job.error = {
                    "code": "MODEL_UNAVAILABLE",
                    "message": "樣本集所綁定的本機特徵模型權重無法載入。",
                    "request_id": job.job_id,
                }
        except ValueError:
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "SCAN_INPUT_INVALID"
                job.error = {
                    "code": "SCAN_INPUT_INVALID",
                    "message": "無法海掃此資料集；請確認樣本集、標註格式與特徵模型相容。",
                    "request_id": job.job_id,
                }
        except Exception:
            logger.exception("LV few-shot scan failed (job=%s)", job.job_id)
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "SCAN_FAILED"
                job.error = {
                    "code": "SCAN_FAILED",
                    "message": "海掃失敗。請檢查資料集、模型權重與 worker 記錄。",
                    "request_id": job.job_id,
                }

    @app.post("/api/v1/fewshot/scans/jobs", status_code=202)
    def create_fewshot_scan_job(request: CreateFewshotScanJobRequest) -> dict:
        bank_dir = fewshot_sample_bank_path(request.sample_bank_id)
        try:
            import sample_bank
            bank = sample_bank.load_sample_bank(bank_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="找不到此樣本集") from exc
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail="樣本集損毀或格式不相容") from exc
        try:
            available = set(catalog())
        except Exception as exc:
            raise HTTPException(status_code=503, detail="無法讀取本機模型清單") from exc
        if str(bank["model"]) not in available:
            raise HTTPException(status_code=422, detail="樣本集所綁定的本機特徵模型尚未安裝")

        roots = []
        image_paths: list[Path] = []
        class_names = None
        for value in request.target_folders:
            try:
                root = Path(value).expanduser().resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise HTTPException(status_code=422, detail="目標資料夾不存在或無法讀取") from exc
            if not root.is_dir():
                raise HTTPException(status_code=422, detail="目標資料夾不存在或無法讀取")
            roots.append(root)
        from object_eval import classes_for, list_images
        for root in roots:
            image_paths.extend(list_images(root))
            class_names = class_names or classes_for(root)
        image_paths = sorted(set(image_paths), key=lambda path: str(path))
        if not image_paths:
            raise HTTPException(status_code=422, detail="目標資料夾找不到支援的影像")

        normalized = request.model_copy(update={"target_folders": [str(root) for root in roots]})
        job_id = uuid.uuid4().hex
        job = _FewshotScanJob(job_id=job_id, request=normalized, total=len(image_paths))
        with app.state.lock:
            if has_active_job():
                raise HTTPException(status_code=409, detail={
                    "code": "JOB_ALREADY_RUNNING",
                    "message": "請等目前分析完成後再開始海掃。",
                    "request_id": job_id,
                })
            app.state.fewshot_scan_jobs[job_id] = job
        threading.Thread(
            target=execute_fewshot_scan_job,
            args=(job, bank, roots, image_paths, class_names),
            daemon=True,
            name=f"lv-fewshot-scan-{job_id[:8]}",
        ).start()
        return {"job_id": job_id, "state": job.state}

    @app.get("/api/v1/fewshot/scans/{scan_id}/records/{index}/preview.jpg")
    def get_fewshot_scan_candidate_preview(scan_id: str, index: int) -> Response:
        if (len(scan_id) != 36 or not scan_id.startswith("fss_")
                or any(char not in "0123456789abcdef" for char in scan_id[4:])):
            raise HTTPException(status_code=404, detail="找不到此海掃結果")
        with app.state.fewshot_lock:
            scan = app.state.fewshot_scans.get(scan_id)
            if scan is None or index < 0 or index >= len(scan["records"]):
                raise HTTPException(status_code=404, detail="找不到此海掃候選")
            record = scan["records"][index]
        try:
            image_path = Path(record["image_path"]).resolve(strict=True)
            if not image_path.is_file() or _sha256_file(image_path)[:16] != str(record["item_id"]).split(":", 1)[0]:
                raise OSError("source changed")
        except (KeyError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail="候選原始影像已變更或無法讀取，請重新海掃") from exc
        image = safe_open_image(image_path)
        if image is None:
            raise HTTPException(status_code=404, detail="候選原始影像無法解碼")
        crop = crop_bbox(image, *record["bbox"], pad=0.1)
        crop.thumbnail((768, 768))
        output = BytesIO()
        crop.convert("RGB").save(output, format="JPEG", quality=86, optimize=True)
        return Response(content=output.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/v1/fewshot/sample-banks/add-candidates")
    def add_fewshot_candidates_to_bank(request: AddFewshotCandidatesRequest) -> dict:
        bank_dir = fewshot_sample_bank_path(request.sample_bank_id)
        with app.state.fewshot_lock:
            scan = app.state.fewshot_scans.get(request.scan_id)
            if scan is None:
                raise HTTPException(status_code=404, detail="找不到此 worker 的海掃結果；請重新海掃")
            if scan["sample_bank_id"] != request.sample_bank_id:
                raise HTTPException(status_code=409, detail="海掃結果與所選樣本集不相符")
            try:
                import sample_bank
                bank = sample_bank.load_sample_bank(bank_dir)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=422, detail="樣本集損毀或格式不相容") from exc
            if (str(bank["model"]) != scan["model_id"]
                    or int(bank["target_res"]) != scan["target_res"]
                    or str(bank.get("object_source") or "yolo") != scan["object_source"]):
                raise HTTPException(status_code=409, detail="樣本集特徵身分已變更；請重新海掃")

            records_by_id = {str(record["item_id"]): record for record in scan["records"]}
            unique_ids = list(dict.fromkeys(request.item_ids))
            missing = [item_id for item_id in unique_ids if item_id not in records_by_id]
            if missing:
                raise HTTPException(status_code=422, detail="海掃候選清單已過期；請重新整理後再試")
            eligible = {
                item_id for item_id in unique_ids
                if float(records_by_id[item_id].get("similarity") or 0.0) >= request.theta
            }
            if len(eligible) != len(unique_ids):
                raise HTTPException(status_code=422, detail="只可採納達目前相似度門檻的候選")
            if set(request.class_overrides) - set(unique_ids):
                raise HTTPException(status_code=422, detail="類別修改清單包含未選取的候選")

            known_classes = {str(label) for label in np.asarray(bank["labels"]).tolist()}
            for label in request.class_overrides.values():
                if label not in known_classes:
                    raise HTTPException(status_code=422, detail="候選類別必須來自目前樣本集")
            existing_ids = {
                str(item["item_id"]) for item in bank.get("provenance") or [] if item.get("item_id")
            }
            additions = [item_id for item_id in unique_ids if item_id not in existing_ids]
            skipped = len(request.item_ids) - len(additions)
            if not additions:
                return {
                    "added": 0,
                    "skipped": skipped,
                    "sample_bank": fewshot_sample_bank_view(request.sample_bank_id, bank),
                }

            objmeta = []
            provenance = []
            for item_id in additions:
                record = records_by_id[item_id]
                label = request.class_overrides.get(item_id) or str(record.get("suggested_class") or "")
                if not label:
                    raise HTTPException(status_code=422, detail="候選沒有可採納的建議類別")
                try:
                    path = Path(record["image_path"]).resolve(strict=True)
                    if not path.is_file() or _sha256_file(path)[:16] != item_id.split(":", 1)[0]:
                        raise OSError("source changed")
                except (KeyError, OSError, RuntimeError) as exc:
                    raise HTTPException(status_code=409, detail="候選原始影像已變更或無法讀取，請重新海掃") from exc
                bbox = [float(value) for value in record["bbox"]]
                objmeta.append({
                    "image_path": str(path),
                    "bbox": bbox,
                    "label": label,
                    "obj_index": int(record.get("obj_index") or 0),
                })
                provenance.append({
                    "image_path": str(path),
                    "bbox": bbox,
                    "label": label,
                    "item_id": item_id,
                })

            embedder: Callable | None = None

            def embed_fn(crop):
                nonlocal embedder
                if embedder is None:
                    if has_custom_object_model_loader or not has_custom_model_loader:
                        embedder = object_loader(
                            bank["model"], keep_aspect=True,
                            target_res=int(bank["target_res"]), head="cls",
                        )
                    else:
                        embedder = loader(bank["model"])
                return embedder(crop)

            from anomaly_tool import _object_embeddings
            vectors = np.asarray(
                _object_embeddings(objmeta, bank["model"], embed_fn), dtype=np.float32,
            )
            merged = sample_bank.append_sample(
                bank_dir,
                vectors=vectors,
                labels=[item["label"] for item in objmeta],
                provenance=provenance,
            )
            return {
                "added": len(additions),
                "skipped": skipped,
                "sample_bank": fewshot_sample_bank_view(request.sample_bank_id, merged),
            }

    @app.post("/api/v1/fewshot/scans/exports")
    def export_fewshot_scan(request: ExportFewshotCandidatesRequest) -> dict:
        with app.state.fewshot_lock:
            scan = app.state.fewshot_scans.get(request.scan_id)
            if scan is None:
                raise HTTPException(status_code=404, detail="找不到此 worker 的海掃結果；請重新海掃")
            if scan["sample_bank_id"] != request.sample_bank_id:
                raise HTTPException(status_code=409, detail="海掃結果與所選樣本集不相符")
            try:
                import sample_bank
                bank_dir = fewshot_sample_bank_path(request.sample_bank_id)
                bank = sample_bank.load_sample_bank(bank_dir)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=422, detail="樣本集損毀或格式不相容") from exc
            if (str(bank["model"]) != scan["model_id"]
                    or int(bank["target_res"]) != scan["target_res"]
                    or str(bank.get("object_source") or "yolo") != scan["object_source"]):
                raise HTTPException(status_code=409, detail="樣本集特徵身分已變更；請重新海掃")

            records = scan["records"]
            records_by_id = {str(record["item_id"]): record for record in records}
            accepted_ids = set(request.accepted_item_ids)
            if accepted_ids - set(records_by_id):
                raise HTTPException(status_code=422, detail="海掃候選清單已過期；請重新海掃")
            eligible_ids = {
                item_id for item_id, record in records_by_id.items()
                if float(record.get("similarity") or 0.0) >= request.theta
            }
            if accepted_ids - eligible_ids:
                raise HTTPException(status_code=422, detail="只可匯出達目前相似度門檻的候選")
            if set(request.class_overrides) - accepted_ids:
                raise HTTPException(status_code=422, detail="類別修改清單包含未採納的候選")

            class_names = sorted({str(label) for label in np.asarray(bank["labels"]).tolist()})
            if any(label not in class_names for label in request.class_overrides.values()):
                raise HTTPException(status_code=422, detail="候選類別必須來自目前樣本集")
            decisions = []
            for index, record in enumerate(records):
                item_id = str(record["item_id"])
                similarity = float(record.get("similarity") or 0.0)
                if similarity < request.theta:
                    decision, final_class = "pending", None
                elif item_id not in accepted_ids:
                    decision, final_class = "skipped", None
                else:
                    suggested = str(record.get("suggested_class") or "")
                    final_class = request.class_overrides.get(item_id, suggested)
                    if not final_class:
                        raise HTTPException(status_code=422, detail="採納候選沒有可匯出的類別")
                    decision = "relabeled" if final_class != suggested else "accepted"
                    try:
                        source = Path(record["image_path"]).resolve(strict=True)
                        if not source.is_file() or _sha256_file(source)[:16] != item_id.split(":", 1)[0]:
                            raise OSError("source changed")
                    except (KeyError, OSError, RuntimeError) as exc:
                        raise HTTPException(status_code=409, detail="候選原始影像已變更或無法讀取，請重新海掃") from exc
                decisions.append({"item": index, "decision": decision, "final_class": final_class})

            try:
                import retrieval_export
                out_dir = Path(request.output_dir).expanduser().resolve()
                result = retrieval_export.export_retrieval(
                    records,
                    decisions,
                    out_dir,
                    class_names=class_names,
                    source_dirs=scan["target_folders"],
                    copy_images=request.copy_images,
                )
            except ValueError as exc:
                # The helper checks C6 before writing. Keep local source/output paths out of API errors.
                raise HTTPException(status_code=422, detail="輸出資料夾不安全或無法匯出") from exc
            except (OSError, RuntimeError) as exc:
                raise HTTPException(status_code=500, detail="寫入匯出資料時失敗") from exc
            return result

    def fewshot_monitor_workspace(monitor_workspace_id: str) -> dict:
        if (len(monitor_workspace_id) != 36 or not monitor_workspace_id.startswith("fwm_")
                or any(char not in "0123456789abcdef" for char in monitor_workspace_id[4:])):
            raise HTTPException(status_code=404, detail="找不到此監看工作區")
        with app.state.fewshot_lock:
            workspace = app.state.fewshot_monitor_workspaces.get(monitor_workspace_id)
        if workspace is None:
            raise HTTPException(status_code=404, detail="找不到此 worker 的監看工作區；請重新初始化")
        return workspace

    def fewshot_monitor_status(workspace: dict) -> dict:
        import al_service
        import al_workspace

        status = al_service.status(workspace["path"])
        queue = al_workspace.read_queue(workspace["path"])
        labels_by_id = {str(item.get("id") or ""): item for item in al_workspace.read_labels(workspace["path"])}
        last = status.get("last_run") or {}
        last_run = ({key: last.get(key) for key in (
            "run_id", "at", "new", "changed", "deleted", "objects_scored",
            "queue_len", "model_version", "objective", "done",
        )} if last else None)
        items = []
        for index, item in enumerate(queue):
            human = labels_by_id.get(str(item.get("id") or ""), {})
            items.append({
                "index": index,
                "item_id": str(item.get("id") or ""),
                "filename": Path(str(item.get("image_path") or "candidate")).name,
                "bbox": [float(value) for value in item.get("bbox", [0.5, 0.5, 1.0, 1.0])],
                "obj_index": int(item.get("obj_index") or 0),
                "suggested_class": str(item.get("suggested_class") or ""),
                "similarity": float(item.get("similarity") or 0.0),
                "reason": str(item.get("reason") or ""),
                "run_id": str(item.get("run_id") or ""),
                "status": str(item.get("status") or "pending"),
                "human_decision": human.get("decision"),
                "human_label": str(human.get("label") or ""),
            })
        return {
            "monitor_workspace_id": workspace["id"],
            "last_run": last_run,
            "queue_len": len(queue),
            "locked": bool(status.get("locked")),
            "items": items,
        }

    @app.post("/api/v1/fewshot/monitors/workspaces", status_code=201)
    def create_fewshot_monitor_workspace(request: CreateFewshotMonitorWorkspaceRequest) -> dict:
        try:
            import sample_bank
            bank_dir = fewshot_sample_bank_path(request.sample_bank_id)
            bank = sample_bank.load_sample_bank(bank_dir)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=404, detail="找不到此樣本集或樣本集格式不相容") from exc
        roots: list[Path] = []
        for value in request.target_folders:
            try:
                root = Path(value).expanduser().resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise HTTPException(status_code=422, detail="目標資料夾不存在或無法讀取") from exc
            if not root.is_dir():
                raise HTTPException(status_code=422, detail="目標資料夾不存在或無法讀取")
            roots.append(root)
        try:
            workspace_path = Path(request.workspace_dir).expanduser().resolve()
            bank_path = Path(bank_dir).resolve()
            if (workspace_path == bank_path or workspace_path in bank_path.parents
                    or bank_path in workspace_path.parents):
                raise ValueError("monitor workspace overlaps sample bank storage")
            with app.state.fewshot_lock:
                prior_ids = [key for key, prior in app.state.fewshot_monitor_workspaces.items()
                             if prior["path"] == workspace_path]
            with app.state.lock:
                if any(job.monitor_workspace_id in prior_ids
                       and job.state in {"queued", "running", "cancelling"}
                       for job in app.state.fewshot_monitor_scan_jobs.values()):
                    raise HTTPException(status_code=409, detail="監看掃描進行中，無法重新初始化工作區")
            import al_service
            import al_workspace
            al_workspace.assert_safe_workspace(workspace_path, roots)
            al_service.init_workspace(
                workspace_path,
                name=roots[0].name or "以樣搜樣監看",
                watch_folders=[str(root) for root in roots],
                model_dir="",
                objective="retrieve",
                k=int(request.k),
                sample_bank_dir=str(bank_dir),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="監看工作區不安全或無法初始化") from exc
        with app.state.fewshot_lock:
            workspace_id = next((key for key, prior in app.state.fewshot_monitor_workspaces.items()
                                  if prior["path"] == workspace_path), "")
        workspace_id = workspace_id or ("fwm_" + uuid.uuid4().hex)
        workspace = {
            "id": workspace_id,
            "path": workspace_path,
            "sample_bank_id": request.sample_bank_id,
            "sample_bank_dir": bank_dir,
            "target_folders": [str(root) for root in roots],
            "k": int(request.k),
            "classes": sorted({str(label) for label in np.asarray(bank["labels"]).tolist()}),
        }
        with app.state.fewshot_lock:
            app.state.fewshot_monitor_workspaces[workspace_id] = workspace
        return fewshot_monitor_status(workspace)

    def execute_fewshot_monitor_scan_job(job: _FewshotMonitorScanJob, workspace: dict) -> None:
        try:
            with app.state.lock:
                job.state, job.phase, job.message = "running", "scanning", "正在執行以樣搜樣監看掃描"
            import al_service

            embedder: Callable | None = None

            def embed_fn(crop):
                nonlocal embedder
                if embedder is None:
                    import sample_bank
                    bank = sample_bank.load_sample_bank(workspace["sample_bank_dir"])
                    if has_custom_object_model_loader or not has_custom_model_loader:
                        embedder = object_loader(
                            bank["model"], keep_aspect=True,
                            target_res=int(bank["target_res"]), head="cls",
                        )
                    else:
                        embedder = loader(bank["model"])
                return embedder(crop)

            def update_progress(data: dict) -> None:
                with app.state.lock:
                    job.phase = "scanning"
                    job.done = max(0, int(data.get("images_processed") or 0))
                    job.total = max(1, int(data.get("images_total") or 1))
                    job.message = f"監看掃描 {job.done}/{job.total} 張影像"

            result = al_service.run_once(workspace["path"], embed_fn=embed_fn, progress=update_progress)
            if result.get("status") != "ok":
                with app.state.lock:
                    job.state, job.phase, job.message = "failed", "failed", "監看掃描未完成"
                    job.error = {
                        "code": "MONITOR_SCAN_FAILED",
                        "message": "監看掃描未完成。請確認工作區、樣本集與目標資料夾。",
                        "request_id": job.job_id,
                    }
                return
            status = fewshot_monitor_status(workspace)
            summary = status.get("last_run") or {}
            with app.state.lock:
                job.result = {
                    "status": "ok",
                    "run_id": summary.get("run_id") or result.get("run_id"),
                    "new": int(summary.get("new") or 0),
                    "changed": int(summary.get("changed") or 0),
                    "deleted": int(summary.get("deleted") or 0),
                    "objects_scored": int(summary.get("objects_scored") or 0),
                    "queue_len": status["queue_len"],
                    "done": bool(summary.get("done")),
                }
                job.done = job.total = max(job.total, 1)
                job.state, job.phase, job.message = "completed", "completed", "監看掃描完成"
        except Exception:
            logger.exception("LV few-shot monitor scan failed (job=%s)", job.job_id)
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "監看掃描失敗"
                job.error = {
                    "code": "MONITOR_SCAN_FAILED",
                    "message": "監看掃描失敗。請檢查資料夾、樣本集與 worker 記錄。",
                    "request_id": job.job_id,
                }

    @app.post("/api/v1/fewshot/monitors/scans/jobs", status_code=202)
    def create_fewshot_monitor_scan_job(request: CreateFewshotMonitorScanRequest) -> dict:
        workspace = fewshot_monitor_workspace(request.monitor_workspace_id)
        job_id = uuid.uuid4().hex
        job = _FewshotMonitorScanJob(job_id=job_id, monitor_workspace_id=request.monitor_workspace_id)
        with app.state.lock:
            if has_active_job():
                raise HTTPException(status_code=409, detail={
                    "code": "JOB_ALREADY_RUNNING",
                    "message": "請等目前分析完成後再開始監看掃描。",
                    "request_id": job_id,
                })
            app.state.fewshot_monitor_scan_jobs[job_id] = job
        threading.Thread(
            target=execute_fewshot_monitor_scan_job,
            args=(job, workspace),
            daemon=True,
            name=f"lv-fewshot-monitor-{job_id[:8]}",
        ).start()
        return {"job_id": job_id, "state": job.state}

    @app.get("/api/v1/fewshot/monitors/workspaces/{monitor_workspace_id}")
    def get_fewshot_monitor_workspace(monitor_workspace_id: str) -> dict:
        return fewshot_monitor_status(fewshot_monitor_workspace(monitor_workspace_id))

    @app.get("/api/v1/fewshot/monitors/workspaces/{monitor_workspace_id}/queue/{index}/preview.jpg")
    def get_fewshot_monitor_queue_preview(monitor_workspace_id: str, index: int) -> Response:
        workspace = fewshot_monitor_workspace(monitor_workspace_id)
        import al_workspace
        queue = al_workspace.read_queue(workspace["path"])
        if index < 0 or index >= len(queue):
            raise HTTPException(status_code=404, detail="找不到此監看候選")
        record = queue[index]
        try:
            image_path = Path(record["image_path"]).resolve(strict=True)
            item_id = str(record.get("id") or "")
            if not image_path.is_file() or _sha256_file(image_path)[:16] != item_id.split(":", 1)[0]:
                raise OSError("source changed")
        except (KeyError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail="候選原始影像已變更或無法讀取，請重新監看掃描") from exc
        image = safe_open_image(image_path)
        if image is None:
            raise HTTPException(status_code=404, detail="候選原始影像無法解碼")
        crop = crop_bbox(image, *record.get("bbox", [0.5, 0.5, 1.0, 1.0]), pad=0.1)
        crop.thumbnail((768, 768))
        output = BytesIO()
        crop.convert("RGB").save(output, format="JPEG", quality=86, optimize=True)
        return Response(content=output.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/v1/fewshot/monitors/workspaces/{monitor_workspace_id}/labels")
    def label_fewshot_monitor_item(
        monitor_workspace_id: str, request: LabelFewshotMonitorItemRequest,
    ) -> dict:
        workspace = fewshot_monitor_workspace(monitor_workspace_id)
        import al_workspace
        queue = al_workspace.read_queue(workspace["path"])
        if not any(str(item.get("id") or "") == request.item_id for item in queue):
            raise HTTPException(status_code=404, detail="找不到此監看候選")
        label = request.label.strip()
        if request.decision == "defect" and label not in workspace["classes"]:
            raise HTTPException(status_code=422, detail="採納類別必須來自目前樣本集")
        with app.state.lock:
            if any(job.monitor_workspace_id == monitor_workspace_id
                   and job.state in {"queued", "running", "cancelling"}
                   for job in app.state.fewshot_monitor_scan_jobs.values()):
                raise HTTPException(status_code=409, detail="監看掃描進行中，請完成後再標記佇列")
        import time
        al_workspace.append_label(workspace["path"], {
            "id": request.item_id,
            "decision": request.decision,
            "label": label,
            "decided_at": time.time(),
        })
        return {"decision": request.decision, "item_id": request.item_id}

    @app.get("/api/v1/fewshot/monitors/workspaces/{monitor_workspace_id}/profile.yaml")
    def export_fewshot_monitor_profile(monitor_workspace_id: str) -> Response:
        workspace = fewshot_monitor_workspace(monitor_workspace_id)
        profile = Path(workspace["path"]) / "profile.yaml"
        if not profile.is_file():
            raise HTTPException(status_code=404, detail="找不到監看設定；請重新初始化")
        return Response(
            content=profile.read_bytes(),
            media_type="application/yaml",
            headers={"Content-Disposition": f'attachment; filename="al_profile_{monitor_workspace_id}.yaml"'},
        )

    def execute(job: _Job) -> None:
        try:
            with app.state.lock:
                job.state, job.phase, job.message = "running", "discovery", "Scanning image folders"
            selected_models = job.request.model_ids or (
                [job.request.model_id] if job.request.model_id else []
            )
            selected_methods = job.request.methods or [job.request.method]
            available_models = set(catalog())
            if not selected_models or any(name not in available_models for name in selected_models):
                raise VisualizationInputError("MODEL_UNAVAILABLE", "A selected model is not installed.")

            def update_progress(done: int, total: int) -> None:
                with app.state.lock:
                    job.phase = "embedding"
                    job.done = done
                    job.total = total
                    job.message = f"Extracting embeddings ({done}/{total})"

            label_overrides = {}
            if job.request.mode == "classifier" and job.request.unit == "image":
                try:
                    import readback_store as _readback_store

                    label_overrides = _readback_store.load_overlay()
                except OSError:
                    logger.exception("LV read-back overlay could not be loaded")

            result = run_image_visualization(
                job.request.folders,
                model_names=selected_models,
                methods=selected_methods,
                mode=job.request.mode,
                unit=job.request.unit,
                class_names=job.request.class_names,
                label_overrides=label_overrides,
                object_policy=(
                    job.request.object_policy.model_dump()
                    if job.request.object_policy is not None else None
                ),
                umap_reference=job.request.umap_reference,
                umap_rebuild=job.request.umap_rebuild,
                include_raw_embeddings=True,
                model_loader=loader,
                object_model_loader=object_loader,
                progress=update_progress,
                cancelled=job.cancel_event.is_set,
            )
            if job.cancel_event.is_set():
                raise JobCancelled()

            run_id = uuid.uuid4().hex
            raw_embeddings = result.pop("_raw_embeddings", {})
            ranking_scores = result.pop("_ranking_scores", {})
            run = {key: value for key, value in result.items() if key != "image_paths"}
            run["image_paths"] = result["image_paths"]
            with app.state.lock:
                # Retain high-dimensional vectors only for the active visual run;
                # old run summaries stay readable without pinning model arrays.
                app.state.raw_embeddings.clear()
                app.state.ranking_scores.clear()
                app.state.block_cache.clear()
                app.state.runs[run_id] = run
                app.state.current_run_id = run_id
                app.state.raw_embeddings[run_id] = raw_embeddings
                app.state.ranking_scores[run_id] = ranking_scores
                job.run_id = run_id
                model_count = len(result["models"])
                expected = len(result["records"]) * model_count
                job.state, job.phase, job.done, job.total = "completed", "completed", expected, expected
                job.message = "Completed"
        except JobCancelled:
            with app.state.lock:
                job.state, job.phase, job.message = "cancelled", "cancelled", "Cancelled"
        except VisualizationInputError as exc:
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", exc.code
                job.error = {"code": exc.code, "message": str(exc), "request_id": job.job_id}
        except FileNotFoundError:
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "MODEL_UNAVAILABLE"
                job.error = {
                    "code": "MODEL_UNAVAILABLE",
                    "message": "The selected model weights are unavailable.",
                    "request_id": job.job_id,
                }
        except Exception:
            # Never return local paths, model internals, or a Python traceback.
            logger.exception("LV web visualization failed (job=%s)", job.job_id)
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "RUN_FAILED"
                job.error = {
                    "code": "RUN_FAILED",
                    "message": "The analysis failed. Check the local worker log and retry.",
                    "request_id": job.job_id,
                }

    def store_detection_evaluation(evaluated: dict, *, demo: bool = False) -> dict:
        evaluation_id = uuid.uuid4().hex
        escape_csv = StringIO()
        escape_writer = csv.writer(escape_csv)
        escape_writer.writerow(["filename", "cls", "cx", "cy", "w", "h"])
        for item in evaluated["escapes"]:
            box = item["box"]
            escape_writer.writerow([item["filename"], item["cls"], box["cx"], box["cy"], box["w"], box["h"]])
        preview_escapes = evaluated["escapes"][:40]
        preview_image_names = {item["filename"] for item in preview_escapes}
        with app.state.lock:
            app.state.evaluation_runs[evaluation_id] = {
                "escape_csv": escape_csv.getvalue(),
                "summary": evaluated["summary"],
                "parameters": evaluated.get("parameters", {}),
                "demo": demo,
                "image_by_name": {
                    name: evaluated["image_by_name"][name]
                    for name in preview_image_names
                    if name in evaluated["image_by_name"]
                },
                "escapes": preview_escapes,
            }
            while len(app.state.evaluation_runs) > 16:
                app.state.evaluation_runs.pop(next(iter(app.state.evaluation_runs)))
        response = {"evaluation_id": evaluation_id, **evaluated["summary"]}
        if demo:
            response["demo"] = True
        return response

    def detection_evaluation_error(exc: EvaluationInputError) -> HTTPException:
        return HTTPException(status_code=400, detail={
            "code": exc.code,
            "message": str(exc),
            "request_id": uuid.uuid4().hex,
        })

    @app.post("/api/v1/evaluations")
    def create_detection_evaluation(request: CreateDetectionEvaluationRequest) -> dict:
        folders = _resolve_request_folders(request)
        try:
            evaluated = run_detection_evaluation(
                folders,
                request.prediction_csv,
                request.consensus_csv,
                iou_thresh=request.iou_thresh,
                conf_thresh=request.conf_thresh,
                class_aware=request.class_aware,
            )
        except EvaluationInputError as exc:
            raise detection_evaluation_error(exc) from exc
        evaluated["parameters"] = {
            "iou_thresh": request.iou_thresh,
            "conf_thresh": request.conf_thresh,
            "class_aware": request.class_aware,
        }
        return store_detection_evaluation(evaluated)

    @app.post("/api/v1/evaluations/demo")
    def create_detection_evaluation_demo() -> dict:
        sample_root = Path(
            os.environ.get("LV_CACHE_DIR")
            or (Path(tempfile.gettempdir()) / "visuallatent")
        ) / "_samples"
        try:
            evaluated = run_detection_evaluation_demo(sample_root)
        except EvaluationInputError as exc:
            raise detection_evaluation_error(exc) from exc
        evaluated["parameters"] = {
            "iou_thresh": 0.5, "conf_thresh": 0.0, "class_aware": True,
        }
        return store_detection_evaluation(evaluated, demo=True)

    @app.get("/api/v1/evaluations/{evaluation_id}/escapes.csv")
    def detection_evaluation_escapes_csv(evaluation_id: str) -> Response:
        with app.state.lock:
            evaluated = app.state.evaluation_runs.get(evaluation_id)
        if evaluated is None:
            raise HTTPException(status_code=404, detail={
                "code": "EVALUATION_NOT_FOUND",
                "message": "This evaluation is no longer available. Run it again.",
                "request_id": uuid.uuid4().hex,
            })
        return Response(
            content=evaluated["escape_csv"],
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="escapes.csv"'},
        )

    @app.get("/api/v1/evaluations/{evaluation_id}/escapes/{index}/thumbnail.jpg")
    def detection_evaluation_escape_thumbnail(evaluation_id: str, index: int) -> Response:
        with app.state.lock:
            evaluated = app.state.evaluation_runs.get(evaluation_id)
        if evaluated is None or index < 0 or index >= len(evaluated["escapes"]):
            raise HTTPException(status_code=404, detail={
                "code": "EVALUATION_IMAGE_NOT_FOUND",
                "message": "This escape image is no longer available.",
                "request_id": uuid.uuid4().hex,
            })
        item = evaluated["escapes"][index]
        image_path = evaluated["image_by_name"].get(item["filename"])
        image = safe_open_image(Path(image_path)) if image_path else None
        if image is None:
            raise HTTPException(status_code=404, detail={
                "code": "EVALUATION_IMAGE_NOT_FOUND",
                "message": "This escape image is no longer available.",
                "request_id": uuid.uuid4().hex,
            })
        box = item["box"]
        crop = crop_bbox(image, box["cx"], box["cy"], box["w"], box["h"], pad=0.4)
        crop.thumbnail((180, 180))
        buffer = BytesIO()
        crop.convert("RGB").save(buffer, format="JPEG", quality=88)
        return Response(content=buffer.getvalue(), media_type="image/jpeg")

    @app.post("/api/v1/visualize/jobs", status_code=202)
    def create_visualization_job(request: CreateVisualizationJob) -> dict:
        resolved_folders = _resolve_request_folders(request)
        request = request.model_copy(update={
            "folders": resolved_folders,
            "folder_grant_ids": [],
        })
        job_id = uuid.uuid4().hex
        job = _Job(job_id=job_id, request=request)
        with app.state.lock:
            if has_active_job():
                raise HTTPException(status_code=409, detail={
                    "code": "JOB_ALREADY_RUNNING",
                    "message": "Wait for the active analysis to finish before starting another.",
                    "request_id": job_id,
                })
            app.state.jobs[job_id] = job
            app.state.current_run_id = None
            app.state.diagnostic_heads.clear()
        threading.Thread(target=execute, args=(job,), daemon=True, name=f"lv-web-{job_id[:8]}").start()
        return {"job_id": job_id, "state": job.state}

    def load_object_policy_profiles(oe) -> dict[str, dict]:
        profile_path = app.state.curation_store.path.parent / "object_policy_profiles.json"

        def usable_profiles(items: list[dict]) -> dict[str, dict]:
            normalized = {}
            for item in items:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                fingerprint = item.get("fingerprint")
                if not isinstance(fingerprint, dict):
                    continue
                size_q, ar_q = fingerprint.get("size_q"), fingerprint.get("ar_q")
                if not isinstance(size_q, list) or len(size_q) != 3:
                    continue
                if not isinstance(ar_q, list) or len(ar_q) != 3:
                    continue
                try:
                    size_q = [float(value) for value in size_q]
                    ar_q = [float(value) for value in ar_q]
                    if not all(math.isfinite(value) for value in size_q + ar_q):
                        continue
                    policy = ComparisonObjectPolicy.model_validate(item.get("policy")).model_dump()
                    name = str(item["name"]).strip()
                    if not name:
                        continue
                    normalized[name] = {
                        **item,
                        "name": name,
                        "fingerprint": {
                            "n_objects": int(fingerprint["n_objects"]),
                            "n_classes": int(fingerprint["n_classes"]),
                            "size_q": size_q,
                            "ar_q": ar_q,
                        },
                        "policy": policy,
                    }
                except (KeyError, TypeError, ValueError):
                    continue
            return normalized

        profiles_by_name = usable_profiles(oe.load_profiles(profile_path))
        legacy_profile_path = Path(oe.PROFILES_PATH)
        if legacy_profile_path.resolve() != profile_path.resolve():
            legacy_profiles = usable_profiles(oe.load_profiles(legacy_profile_path))
            profiles_by_name = legacy_profiles | profiles_by_name
        return profiles_by_name

    def match_object_policy_profile(oe, fingerprint: dict) -> dict | None:
        try:
            profile, similarity = oe.match_profile(
                fingerprint,
                list(load_object_policy_profiles(oe).values()),
            )
        except (KeyError, TypeError, ValueError):
            logger.info("Ignoring unreadable LV object policy profiles")
            return None
        if not profile or similarity < 0.6:
            return None
        return {
            "name": str(profile["name"]),
            "similarity": float(similarity),
            "policy": ComparisonObjectPolicy.model_validate(profile["policy"]).model_dump(),
        }

    @app.post("/api/v1/object-policy/profile-match")
    def get_object_policy_profile_match(request: ObjectPolicyProfileMatchRequest) -> dict:
        import object_eval as oe

        folders = [Path(folder) for folder in _resolve_request_folders(request)]
        images = [image for folder in folders for image in oe.list_images(folder)]
        if not images:
            raise HTTPException(
                status_code=422,
                detail="The selected folder contains no readable image files.",
            )
        class_names = oe.classes_for(folders[0]) or request.class_names or None
        try:
            fingerprint = oe.dataset_fingerprint(images, class_names)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=422,
                detail="The selected folder's images or labels could not be read.",
            ) from exc
        return {
            "fingerprint": fingerprint,
            "similar_profile": match_object_policy_profile(oe, fingerprint),
        }

    def execute_object_policy_job(job: _ObjectPolicyAutotuneJob) -> None:
        try:
            import object_eval as oe

            folders = [Path(folder) for folder in job.request.folders]
            images = [image for folder in folders for image in oe.list_images(folder)]
            if not images:
                with app.state.lock:
                    job.state, job.phase, job.message = "failed", "failed", "找不到可用的圖片"
                    job.error = {
                        "code": "OBJECT_POLICY_NO_IMAGES",
                        "message": "The selected folder contains no readable image files.",
                        "request_id": job.job_id,
                    }
                return
            class_names = oe.classes_for(folders[0]) or job.request.class_names or None
            candidate_count = len(oe.DEFAULT_CANDIDATES)

            with app.state.lock:
                job.state, job.phase, job.message = "running", "fingerprint", "正在檢查物件標註"
                job.total = candidate_count

            def update_progress(ci: int, count: int, policy: dict, done: int, total: int) -> None:
                if job.cancel_event.is_set():
                    raise JobCancelled()
                with app.state.lock:
                    job.phase = "embedding"
                    job.total = max(1, int(count) * max(1, int(total)))
                    job.done = min(job.total, max(0, int(ci) * max(1, int(total)) + int(done)))
                    job.message = (
                        f"[{ci + 1}/{count}] {oe.policy_tag(policy)} · 物件 {done}/{total}"
                    )[:240]

            rows, best_policy, fingerprint = oe.run_autotune(
                images,
                class_names,
                model=job.request.model_id,
                cache_dir=oe.dataset_cache_dir(folders[0], "object_crops") / "_autotune",
                progress=update_progress,
                model_loader=object_loader,
            )
            if job.cancel_event.is_set():
                raise JobCancelled()
            if not rows or best_policy is None:
                with app.state.lock:
                    job.state, job.phase, job.message = "failed", "failed", "資料中沒有可評估的標註物件"
                    job.error = {
                        "code": "OBJECT_POLICY_NO_OBJECTS",
                        "message": "No labeled objects were found. Check the YOLO labels and class names.",
                        "request_id": job.job_id,
                    }
                return

            suggested_profile = match_object_policy_profile(oe, fingerprint)
            result = {
                "rows": rows,
                "best_policy": best_policy,
                "fingerprint": fingerprint,
                "similar_profile": suggested_profile,
            }
            with app.state.lock:
                job.result = result
                job.done = job.total
                job.state, job.phase, job.message = "completed", "completed", "自動設定搜尋完成"
        except JobCancelled:
            with app.state.lock:
                job.state, job.phase, job.message = "cancelled", "cancelled", "已取消自動設定搜尋"
        except (ValueError, FileNotFoundError):
            logger.info("LV object policy autotune input rejected (job=%s)", job.job_id)
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "自動設定搜尋失敗"
                job.error = {
                    "code": "OBJECT_POLICY_INVALID_INPUT",
                    "message": "The image folder, labels, or class names could not be read.",
                    "request_id": job.job_id,
                }
        except Exception:
            logger.exception("LV object policy autotune failed (job=%s)", job.job_id)
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "自動設定搜尋失敗"
                job.error = {
                    "code": "OBJECT_POLICY_AUTOTUNE_FAILED",
                    "message": "Object policy tuning failed. Check the local worker log and model installation.",
                    "request_id": job.job_id,
                }

    @app.post("/api/v1/object-policy/autotune/jobs", status_code=202)
    def create_object_policy_job(request: CreateObjectPolicyAutotuneJob) -> dict:
        if request.model_id not in set(catalog()):
            raise HTTPException(status_code=422, detail="選取的模型未安裝")
        resolved_folders = _resolve_request_folders(request)
        request = request.model_copy(update={
            "folders": resolved_folders,
            "folder_grant_ids": [],
        })
        job_id = uuid.uuid4().hex
        job = _ObjectPolicyAutotuneJob(job_id=job_id, request=request)
        with app.state.lock:
            if has_active_job():
                raise HTTPException(status_code=409, detail={
                    "code": "JOB_ALREADY_RUNNING",
                    "message": "Wait for the active analysis to finish before tuning object policy.",
                    "request_id": job_id,
                })
            app.state.object_policy_jobs[job_id] = job
        threading.Thread(
            target=execute_object_policy_job,
            args=(job,),
            daemon=True,
            name=f"lv-object-policy-{job_id[:8]}",
        ).start()
        return {"job_id": job_id, "state": job.state}

    @app.post("/api/v1/object-policy/autotune/jobs/{job_id}/profile")
    def save_object_policy_profile(job_id: str, request: SaveObjectPolicyProfileRequest) -> dict:
        with app.state.lock:
            job = app.state.object_policy_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Object policy tuning result not found")
            if job.state != "completed" or not job.result:
                raise HTTPException(status_code=409, detail="Complete an object policy search before saving a profile")
            result = copy.deepcopy(job.result)
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="Profile name cannot be empty")
        import object_eval as oe

        profile_path = app.state.curation_store.path.parent / "object_policy_profiles.json"
        try:
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            oe.save_profile(
                name,
                result["fingerprint"],
                result["best_policy"],
                result["rows"][0],
                path=profile_path,
            )
        except Exception as exc:
            logger.exception("LV object policy profile could not be saved")
            raise HTTPException(
                status_code=500,
                detail="Could not save the object policy profile in the app cache",
            ) from exc
        return {"name": name, "policy": result["best_policy"]}

    def execute_anomaly_model_job(job: _AnomalyModelJob) -> None:
        try:
            with app.state.lock:
                job.state, job.phase, job.message = "running", "building", "Preparing anomaly model"
                job.total = 100

            def update_progress(done: int, total: int, message: str) -> None:
                with app.state.lock:
                    job.phase = "building"
                    job.done = max(0, min(int(done), int(total)))
                    job.total = max(1, int(total))
                    job.message = str(message)[:240]

            if app.state.anomaly_model_builder is not None:
                bundle = app.state.anomaly_model_builder(job.request.model_dump(), progress=update_progress)
            else:
                from react_anomaly import build_model
                bundle = build_model(job.request.model_dump(), progress=update_progress)
            if job.cancel_event.is_set():
                raise JobCancelled()
            model_id = "am_" + uuid.uuid4().hex[:16]
            bundle["_selected_folders"] = list(job.request.train_folders)
            from react_anomaly import _view
            result = _view(model_id, bundle, source="built")
            with app.state.lock:
                # Streamlit owns one active anomaly model slot. Drop prior
                # in-memory banks instead of pinning potentially large arrays.
                app.state.anomaly_models.clear()
                app.state.anomaly_models[model_id] = bundle
                app.state.current_anomaly_model_id = model_id
                app.state.anomaly_runs.clear()
                app.state.current_anomaly_run_id = None
                job.model_id = model_id
                job.result = result
                job.done = job.total = 100
                job.state, job.phase, job.message = "completed", "completed", "模型已建立"
        except JobCancelled:
            with app.state.lock:
                job.state, job.phase, job.message = "cancelled", "cancelled", "已取消"
        except (ValueError, FileNotFoundError) as exc:
            logger.info("LV anomaly model input rejected (job=%s): %s", job.job_id, exc)
            message = str(exc)
            for selected in job.request.train_folders:
                for candidate in {selected, os.path.abspath(os.path.expanduser(selected))}:
                    if candidate:
                        message = message.replace(candidate, "選取資料夾")
            if isinstance(exc, FileNotFoundError):
                message = "訓練資料夾不存在，請重新選擇。"
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "模型建立失敗"
                job.error = {
                    "code": "ANOMALY_MODEL_INVALID",
                    "message": message[:500],
                    "request_id": job.job_id,
                }
        except Exception:
            logger.exception("LV anomaly model build failed (job=%s)", job.job_id)
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "模型建立失敗"
                job.error = {
                    "code": "ANOMALY_MODEL_FAILED",
                    "message": "建立 Normal Bank 失敗；請檢查本機模型權重、資料集標註及 worker 記錄後重試。",
                    "request_id": job.job_id,
                }

    @app.post("/api/v1/anomaly/models/jobs", status_code=202)
    def create_anomaly_model_job(request: CreateAnomalyModelJobRequest) -> dict:
        if request.model_id not in set(catalog()):
            raise HTTPException(status_code=422, detail="選取的模型未安裝")
        job_id = uuid.uuid4().hex
        job = _AnomalyModelJob(job_id=job_id, request=request)
        with app.state.lock:
            if has_active_job():
                raise HTTPException(status_code=409, detail={
                    "code": "JOB_ALREADY_RUNNING",
                    "message": "請等目前分析完成後再建立瑕疵偵測模型。",
                    "request_id": job_id,
                })
            app.state.anomaly_jobs[job_id] = job
        threading.Thread(
            target=execute_anomaly_model_job,
            args=(job,),
            daemon=True,
            name=f"lv-anomaly-{job_id[:8]}",
        ).start()
        return {"job_id": job_id, "state": job.state}

    @app.get("/api/v1/anomaly/jobs/{job_id}")
    def get_anomaly_model_job(job_id: str) -> dict:
        with app.state.lock:
            job = app.state.anomaly_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="找不到瑕疵偵測建模工作")
            return _anomaly_job_view(job)

    def execute_anomaly_score_job(job: _AnomalyScoreJob, bundle: dict) -> None:
        try:
            with app.state.lock:
                job.state, job.phase, job.message = "running", "scoring", "準備套用已載入模型"
                job.total = 100

            def update_progress(done: int, total: int, message: str) -> None:
                with app.state.lock:
                    job.phase = "scoring"
                    job.done = max(0, min(int(done), int(total)))
                    job.total = max(1, int(total))
                    job.message = str(message)[:240]

            scorer = app.state.anomaly_scorer
            if scorer is None:
                from react_anomaly import score_targets
                scorer = score_targets
            run = scorer(job.request.model_dump(), bundle, progress=update_progress)
            if not run.get("records"):
                raise ValueError("找不到符合模型物件來源的目標影像或標註")
            run_id = uuid.uuid4().hex
            run["run_id"] = run_id
            run["model_bundle_id"] = job.model_id
            run["target_folders"] = list(job.request.target_folders)
            run["labels"] = np.asarray(run["labels"], dtype=object)
            run["revision"] = 1
            from react_anomaly import anomaly_run_view
            view = anomaly_run_view(run)
            with app.state.lock:
                if app.state.current_anomaly_model_id != job.model_id:
                    raise ValueError("目前載入模型已變更，請重新套用偵測")
                app.state.anomaly_runs.clear()
                app.state.anomaly_runs[run_id] = run
                app.state.current_anomaly_run_id = run_id
                job.run_id = run_id
                job.result = view
                job.done = job.total = 100
                job.state, job.phase, job.message = "completed", "completed", "目標資料夾評分完成"
        except (ValueError, FileNotFoundError) as exc:
            logger.info("LV anomaly scoring input rejected (job=%s): %s", job.job_id, exc)
            message = str(exc)
            for selected in job.request.target_folders:
                for candidate in {selected, os.path.abspath(os.path.expanduser(selected))}:
                    if candidate:
                        message = message.replace(candidate, "選取資料夾")
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "目標資料夾評分失敗"
                job.error = {
                    "code": "ANOMALY_SCORE_INVALID",
                    "message": message[:500],
                    "request_id": job.job_id,
                }
        except Exception:
            logger.exception("LV anomaly scoring failed (job=%s)", job.job_id)
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "目標資料夾評分失敗"
                job.error = {
                    "code": "ANOMALY_SCORE_FAILED",
                    "message": "套用模型失敗；請檢查本機模型權重、資料集標註及 worker 記錄後重試。",
                    "request_id": job.job_id,
                }

    @app.post("/api/v1/anomaly/score-jobs", status_code=202)
    def create_anomaly_score_job(request: CreateAnomalyScoreJobRequest) -> dict:
        job_id = uuid.uuid4().hex
        with app.state.lock:
            model_id = app.state.current_anomaly_model_id
            bundle = app.state.anomaly_models.get(model_id) if model_id else None
            if bundle is None:
                raise HTTPException(status_code=409, detail="請先建立或載入瑕疵偵測模型")
            if has_active_job():
                raise HTTPException(status_code=409, detail={
                    "code": "JOB_ALREADY_RUNNING",
                    "message": "請等目前分析完成後再套用瑕疵偵測模型。",
                    "request_id": job_id,
                })
            job = _AnomalyScoreJob(job_id=job_id, model_id=model_id, request=request)
            app.state.anomaly_score_jobs[job_id] = job
        threading.Thread(
            target=execute_anomaly_score_job,
            args=(job, bundle),
            daemon=True,
            name=f"lv-anomaly-score-{job_id[:8]}",
        ).start()
        return {"job_id": job_id, "state": job.state}

    @app.get("/api/v1/anomaly/score-jobs/{job_id}")
    def get_anomaly_score_job(job_id: str) -> dict:
        with app.state.lock:
            job = app.state.anomaly_score_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="找不到瑕疵偵測套用工作")
            return _anomaly_score_job_view(job)

    @app.get("/api/v1/anomaly/runs/{run_id}")
    def get_anomaly_run(run_id: str) -> dict:
        from react_anomaly import anomaly_run_view
        with app.state.lock:
            run = app.state.anomaly_runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="找不到目前 worker 的瑕疵偵測結果")
            return anomaly_run_view(run)

    @app.get("/api/v1/anomaly/runs/{run_id}/objects")
    def get_anomaly_run_objects(
        run_id: str,
        offset: int = 0,
        limit: int = 60,
        verdict: Literal["good", "bad"] | None = None,
        class_label: str | None = None,
        score_min: float | None = None,
        score_max: float | None = None,
    ) -> dict:
        from react_anomaly import anomaly_run_objects
        if score_min is not None and score_max is not None and score_min > score_max:
            raise HTTPException(status_code=422, detail="最低分數不可高於最高分數")
        if any(value is not None and not np.isfinite(value) for value in (score_min, score_max)):
            raise HTTPException(status_code=422, detail="分數範圍必須是有限數值")
        with app.state.lock:
            run = app.state.anomaly_runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="找不到目前 worker 的瑕疵偵測結果")
            return anomaly_run_objects(
                run, offset=offset, limit=limit, verdict=verdict,
                class_label=class_label, score_min=score_min, score_max=score_max,
            )

    def anomaly_prelabel_state(run_id: str, expected_revision: int) -> dict:
        with app.state.lock:
            run = app.state.anomaly_runs.get(run_id)
            if run is None or app.state.current_anomaly_run_id != run_id:
                raise HTTPException(status_code=404, detail={
                    "code": "ANOMALY_RUN_NOT_FOUND",
                    "message": "找不到目前 worker 的瑕疵偵測結果，請重新套用模型。",
                    "request_id": uuid.uuid4().hex,
                })
            if int(run["revision"]) != expected_revision:
                raise HTTPException(status_code=409, detail={
                    "code": "ANOMALY_RUN_REVISION_CONFLICT",
                    "message": "偵測結果已更新，請重新預覽預標。",
                    "request_id": uuid.uuid4().hex,
                })
            bundle_id = app.state.current_anomaly_model_id
            bundle = app.state.anomaly_models.get(bundle_id) if bundle_id else None
            if not bundle or run.get("model_bundle_id") != bundle_id:
                raise HTTPException(status_code=409, detail={
                    "code": "PRELABEL_MODEL_UNAVAILABLE",
                    "message": "目前載入的模型與這次偵測結果不一致，請重新套用模型。",
                    "request_id": uuid.uuid4().hex,
                })
            head = bundle.get("head")
            if not head:
                raise HTTPException(status_code=409, detail={
                    "code": "PRELABEL_HEAD_UNAVAILABLE",
                    "message": "此模型沒有瑕疵分類頭，無法預標。請使用瑕疵類別語義建立並載入分類頭模型。",
                    "request_id": uuid.uuid4().hex,
                })
            records = list(run.get("records") or [])
            raw_embeddings = run.get("obj_emb")
            raw_scores = run.get("scores")
            if raw_embeddings is None or raw_scores is None:
                raise HTTPException(status_code=409, detail={
                    "code": "PRELABEL_EMBEDDINGS_UNAVAILABLE",
                    "message": "這次結果沒有可用的物件 embedding，無法預標；請重新套用物件層級模型。",
                    "request_id": uuid.uuid4().hex,
                })
            embeddings = np.asarray(raw_embeddings, dtype=np.float32)
            scores = np.asarray(raw_scores, dtype=np.float64)
            if (
                embeddings.ndim != 2 or embeddings.shape[0] != len(records)
                or scores.ndim != 1 or len(scores) != len(records)
                or not np.isfinite(embeddings).all() or not np.isfinite(scores).all()
            ):
                raise HTTPException(status_code=409, detail={
                    "code": "PRELABEL_EMBEDDINGS_UNAVAILABLE",
                    "message": "物件 embedding 與偵測結果沒有對齊，請重新套用模型。",
                    "request_id": uuid.uuid4().hex,
                })
            return {
                "run": run,
                "records": records,
                "embeddings": embeddings,
                "scores": scores,
                "threshold": float(run["threshold"]),
                "head": head,
                "class_names": [str(value) for value in (head.get("classes") or [])],
                "source_dirs": list(run.get("target_folders") or []),
                "model_bundle_id": bundle_id,
                "revision": int(run["revision"]),
            }

    def anomaly_prelabel_decisions(state: dict, request: AnomalyPrelabelRequest) -> list[dict]:
        classes = set(state["class_names"])
        if request.target_classes is not None and not set(request.target_classes) <= classes:
            raise HTTPException(status_code=422, detail={
                "code": "PRELABEL_CLASS_SELECTION_INVALID",
                "message": "預標類別不在目前模型的分類頭中，請重新選擇。",
                "request_id": uuid.uuid4().hex,
            })
        try:
            import prelabel as _prelabel
            return _prelabel.prelabel_records(
                state["records"], state["embeddings"], state["scores"], state["head"],
                anomaly_threshold=state["threshold"], min_conf=request.min_conf,
                target_classes=request.target_classes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={
                "code": "PRELABEL_EVALUATION_INVALID",
                "message": f"無法計算預標：{exc}",
                "request_id": uuid.uuid4().hex,
            }) from exc

    @app.post("/api/v1/anomaly/runs/{run_id}/prelabel/preview")
    def preview_anomaly_prelabels(
        run_id: str, request: AnomalyPrelabelPreviewRequest,
    ) -> dict:
        state = anomaly_prelabel_state(run_id, request.expected_revision)
        decisions = anomaly_prelabel_decisions(state, request)
        start = min(request.offset, len(decisions))
        rows = []
        for decision in decisions[start:start + request.limit]:
            index = int(decision["item"])
            record = state["records"][index]
            rows.append({
                "index": index,
                "filename": Path(str(record.get("image_path") or "")).name,
                "label": str(record.get("label") or ""),
                "obj_index": int(record.get("obj_index") or 0),
                "bbox": [float(value) for value in record["bbox"]],
                "score": float(state["scores"][index]),
                "pred": str(decision["pred"]),
                "confidence": float(decision["conf"]),
                "accept": bool(decision["accept"]),
            })
        return {
            "run_id": run_id,
            "revision": state["revision"],
            "min_conf": request.min_conf,
            "target_classes": request.target_classes if request.target_classes is not None else state["class_names"],
            "class_names": state["class_names"],
            "total": len(decisions),
            "accepted": sum(1 for decision in decisions if decision["accept"]),
            "offset": start,
            "limit": request.limit,
            "items": rows,
        }

    @app.post("/api/v1/anomaly/runs/{run_id}/prelabel/export")
    def export_anomaly_prelabels(
        run_id: str, request: AnomalyPrelabelExportRequest,
    ) -> dict:
        state = anomaly_prelabel_state(run_id, request.expected_revision)
        decisions = anomaly_prelabel_decisions(state, request)
        try:
            import prelabel as _prelabel
            lines = _prelabel.to_yolo_lines(
                state["records"], decisions, class_names=state["class_names"],
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={
                "code": "PRELABEL_EXPORT_INVALID",
                "message": f"無法準備 YOLO 預標：{exc}",
                "request_id": uuid.uuid4().hex,
            }) from exc
        with app.state.lock:
            current = app.state.anomaly_runs.get(run_id)
            if (
                current is None or app.state.current_anomaly_run_id != run_id
                or int(current["revision"]) != state["revision"]
                or app.state.current_anomaly_model_id != state["model_bundle_id"]
            ):
                raise HTTPException(status_code=409, detail={
                    "code": "ANOMALY_RUN_REVISION_CONFLICT",
                    "message": "偵測結果或模型已更新，請重新預覽預標後再匯出。",
                    "request_id": uuid.uuid4().hex,
                })
            try:
                result = _prelabel.export_prelabels(
                    lines, request.output_dir, class_names=state["class_names"],
                    source_dirs=state["source_dirs"],
                )
            except ValueError as exc:
                logger.info("LV prelabel export rejected (run=%s): %s", run_id, exc)
                raise HTTPException(status_code=422, detail={
                    "code": "PRELABEL_EXPORT_UNSAFE",
                    "message": "匯出安全檢查未通過：請選來源資料夾之外、且不含 images 子資料夾的輸出位置。",
                    "request_id": uuid.uuid4().hex,
                }) from exc
            except OSError:
                logger.exception("LV prelabel export failed (run=%s)", run_id)
                raise HTTPException(status_code=500, detail={
                    "code": "PRELABEL_EXPORT_FAILED",
                    "message": "無法寫入預標輸出位置。請確認資料夾可寫入後重試。",
                    "request_id": uuid.uuid4().hex,
                }) from None
        return {
            "run_id": run_id,
            "revision": state["revision"],
            "accepted": sum(1 for decision in decisions if decision["accept"]),
            "written": int(result["written"]),
            "objects": int(result["objects"]),
            "class_names": state["class_names"],
        }

    @app.post("/api/v1/anomaly/runs/{run_id}/reapply")
    def reapply_anomaly_run(run_id: str, request: ReapplyAnomalyRunRequest) -> dict:
        from react_anomaly import anomaly_run_view, reclassify_anomaly_run
        with app.state.lock:
            run = app.state.anomaly_runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="找不到目前 worker 的瑕疵偵測結果")
            try:
                reclassify_anomaly_run(
                    run,
                    expected_revision=request.expected_revision,
                    confirmed=request.confirmed,
                    contamination=request.contamination,
                )
            except ValueError as exc:
                status = 409 if "已更新" in str(exc) else 422
                raise HTTPException(status_code=status, detail=str(exc)) from exc
            return anomaly_run_view(run)

    @app.post("/api/v1/anomaly/runs/{run_id}/curation/items", status_code=201)
    def add_anomaly_candidates_to_curation(
        run_id: str,
        request: AddAnomalyCurationRequest,
    ) -> dict:
        with app.state.lock:
            run = app.state.anomaly_runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="找不到目前 worker 的瑕疵偵測結果")
            if request.expected_revision != int(run["revision"]):
                raise HTTPException(status_code=409, detail={
                    "code": "ANOMALY_RUN_REVISION_CONFLICT",
                    "message": "偵測結果已更新，請重新載入後再加入策展清單。",
                    "request_id": uuid.uuid4().hex,
                })
            records = run["records"]
            labels = np.asarray(run["labels"], dtype=object)
            scores = np.asarray(run["scores"], dtype=float)
            if any(index < 0 or index >= len(records) or index >= len(labels) or index >= len(scores)
                   for index in request.indices):
                raise HTTPException(status_code=422, detail={
                    "code": "INVALID_ANOMALY_RECORD_INDEX",
                    "message": "選取的異常候選不屬於目前這次評分結果。",
                    "request_id": uuid.uuid4().hex,
                })
            if any(labels[index] != "bad" for index in request.indices):
                raise HTTPException(status_code=422, detail={
                    "code": "ANOMALY_CANDIDATE_NOT_BAD",
                    "message": "只能將目前判為異常候選的物件加入策展清單；請重新整理分數清單。",
                    "request_id": uuid.uuid4().hex,
                })

            entries = []
            for index in dict.fromkeys(request.indices):
                record = records[index]
                try:
                    path = Path(record["image_path"]).expanduser().resolve(strict=True)
                    if not path.is_file():
                        raise OSError("source is not a file")
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                except (KeyError, OSError) as exc:
                    raise HTTPException(status_code=404, detail={
                        "code": "CURATION_IMAGE_UNAVAILABLE",
                        "message": "選取的原始影像已無法讀取，沒有加入策展清單。",
                        "request_id": uuid.uuid4().hex,
                    }) from exc
                bbox = [float(value) for value in record.get("bbox", (0, 0, 1, 1))]
                if len(bbox) != 4 or not np.isfinite(bbox).all():
                    raise HTTPException(status_code=409, detail="異常候選的原始影像框資料無效")
                source = {
                    "index": index,
                    "filename": path.name,
                    "path": str(path),
                    "image_path": str(path),
                    "label": str(record.get("label") or ""),
                    "split": str(record.get("split") or ""),
                    "sha256": digest.hexdigest(),
                    "source": "anomaly",
                    "source_run_id": run_id,
                    "score": float(scores[index]),
                    "reason": "Anomaly score candidate; human review required.",
                    "level": "object",
                    "bbox": bbox,
                    "obj_index": int(record.get("obj_index") or 0),
                }
                entries.append(source)
            try:
                added, skipped = app.state.curation_store.add(entries)
                items = app.state.curation_store.list_items()
            except CurationStoreError as exc:
                raise curation_store_error() from exc
        return {
            "added": added,
            "skipped": skipped,
            "items": [public_curation_item(item) for item in items],
            "total": len(items),
        }

    def anomaly_object_crop(run_id: str, index: int):
        with app.state.lock:
            run = app.state.anomaly_runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="找不到目前 worker 的瑕疵偵測結果")
            if index < 0 or index >= len(run["records"]):
                raise HTTPException(status_code=404, detail="找不到指定物件")
            record = run["records"][index]
        image = safe_open_image(record["image_path"])
        if image is None:
            raise HTTPException(status_code=404, detail="目標影像已無法讀取")
        from object_eval import DEFAULT_POLICY, _adaptive_pad_px
        pad = float((run["model_meta"].get("embedding_recipe") or {}).get("pad", DEFAULT_POLICY["pad"]))
        bbox = tuple(float(value) for value in record["bbox"])
        return run, record, crop_bbox(image, *bbox, pad_px=_adaptive_pad_px(bbox, *image.size, pad))

    @app.get("/api/v1/anomaly/runs/{run_id}/objects/{index}/preview.jpg")
    def get_anomaly_object_preview(run_id: str, index: int) -> Response:
        _run, _record, crop = anomaly_object_crop(run_id, index)
        crop.thumbnail((768, 768))
        output = BytesIO()
        crop.convert("RGB").save(output, format="JPEG", quality=86, optimize=True)
        return Response(content=output.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/anomaly/runs/{run_id}/objects/{index}/heatmap.png")
    def get_anomaly_object_heatmap(run_id: str, index: int) -> Response:
        run, _record, crop = anomaly_object_crop(run_id, index)
        if run["score_mode"] != "patch" or run.get("bank") is None:
            raise HTTPException(status_code=409, detail="物件分數模式沒有 patch 熱圖")
        try:
            from anomaly_heatmap import render_heatmap
            from anomaly_score import score_object
            from patch_features import extract_patch_grid
            meta = run["model_meta"]
            patch = extract_patch_grid(crop, target_res=int(meta["target_res"]), model=str(meta["model"]))
            _score, patch_scores = score_object(patch["feats"], patch["grid"], run["bank"])
            heatmap = render_heatmap(patch_scores, crop)
        except Exception as exc:
            logger.exception("LV anomaly heatmap failed (run=%s, index=%s)", run_id, index)
            raise HTTPException(status_code=500, detail="無法產生物件熱圖；請檢查本機模型權重與 worker 記錄") from exc
        heatmap.thumbnail((768, 768))
        output = BytesIO()
        heatmap.save(output, format="PNG", optimize=True)
        return Response(content=output.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/anomaly/models")
    def list_anomaly_models() -> dict:
        from react_anomaly import list_saved
        items = list_saved()
        known = {row["model_id"] for row in items}
        with app.state.lock:
            current_id = app.state.current_anomaly_model_id
            current = app.state.anomaly_models.get(current_id) if current_id else None
            if current_id and current and current_id not in known:
                from react_anomaly import _view
                items.insert(0, _view(current_id, current, source="built"))
        for item in items:
            item["active"] = item["model_id"] == current_id
        return {"items": items}

    @app.post("/api/v1/anomaly/models/{model_id}/save")
    def save_anomaly_model(model_id: str) -> dict:
        from react_anomaly import save_bundle
        with app.state.lock:
            bundle = app.state.anomaly_models.get(model_id)
            if bundle is None:
                raise HTTPException(status_code=404, detail="目前 worker 沒有此模型；請重新載入已保存模型")
            selected_folders = bundle.get("_selected_folders", [])
        try:
            saved = save_bundle(model_id, bundle, selected_folders)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("LV anomaly model save failed (model=%s)", model_id)
            raise HTTPException(status_code=500, detail="模型保存失敗；請檢查 LV_CACHE_DIR 可用空間與權限") from exc
        return {"model": saved}

    @app.post("/api/v1/anomaly/models/load")
    def load_anomaly_model(request: LoadAnomalyModelRequest) -> dict:
        from react_anomaly import load_bundle
        try:
            bundle, view = load_bundle(request.model_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="找不到完整的已保存模型") from exc
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail="模型檔案不完整、損毀或不相容，請重新選擇有效模型") from exc
        with app.state.lock:
            if has_active_job():
                raise HTTPException(status_code=409, detail="請等目前工作完成後再切換模型")
            app.state.anomaly_models.clear()
            app.state.anomaly_models[request.model_id] = bundle
            app.state.current_anomaly_model_id = request.model_id
            app.state.anomaly_runs.clear()
            app.state.current_anomaly_run_id = None
        return {"model": view}

    def coverage_points(records: list[dict], coordinates: np.ndarray) -> list[dict]:
        return [
            {
                **record,
                "index": int(record.get("index", index)),
                "x": float(coordinates[index, 0]),
                "y": float(coordinates[index, 1]),
                "z": float(coordinates[index, 2]) if coordinates.shape[1] > 2 else 0.0,
            }
            for index, record in enumerate(records)
        ]

    def public_coverage_run(run: dict) -> dict:
        return {
            key: value for key, value in run.items()
            if key != "image_paths"
        }

    def coverage_attribute_axes(
        run_id: str,
        axis_x: str,
        axis_y: str,
        bins: int,
    ) -> tuple[dict, np.ndarray, list[int], list[str], list[int], list[str]]:
        with app.state.lock:
            run = app.state.coverage_runs.get(run_id)
            vectors = app.state.coverage_embeddings.get(run_id)
            if run is None or vectors is None:
                raise HTTPException(status_code=404, detail="Coverage run not found")
            if run.get("unit") != "image":
                raise HTTPException(status_code=409, detail="Attribute completeness requires image-level coverage")
            records = run["records"]
            paths: list[Path] = run["image_paths"]
            stats = app.state.coverage_image_stats.get(run_id)
        if len(records) != len(paths) or len(records) != len(vectors):
            raise HTTPException(status_code=409, detail="Coverage run records are inconsistent")

        needs_stats = axis_x not in {"label", "split"} or axis_y not in {"label", "split"}
        if needs_stats and stats is None:
            computed = []
            for path in paths:
                try:
                    computed.append(image_stats(Path(path)))
                except (OSError, Image.DecompressionBombError):
                    computed.append({axis: 0.0 for axis in ("brightness", "contrast", "sharpness", "aspect")})
            with app.state.lock:
                if app.state.coverage_runs.get(run_id) is not run:
                    raise HTTPException(status_code=404, detail="Coverage run expired")
                stats = app.state.coverage_image_stats.setdefault(run_id, computed)

        def bucket_axis(axis: str) -> tuple[list[int], list[str]]:
            if axis in {"label", "split"}:
                return categorical_buckets([str(record.get(axis, "")) for record in records])
            assert stats is not None
            return bucketize(
                [float(row.get(axis, 0.0)) for row in stats],
                bins,
                method="quantile",
            )

        bucket_x, labels_x = bucket_axis(axis_x)
        bucket_y, labels_y = bucket_axis(axis_y)
        return run, vectors, bucket_x, labels_x, bucket_y, labels_y

    def execute_coverage(job: _CoverageJob) -> None:
        try:
            with app.state.lock:
                job.state, job.phase, job.message = "running", "discovery", "Scanning image folders"
            if job.request.model_id not in set(catalog()):
                raise VisualizationInputError("MODEL_UNAVAILABLE", "The selected model is not installed.")

            def update_progress(done: int, total: int) -> None:
                with app.state.lock:
                    job.phase = "embedding"
                    job.done = done
                    job.total = total
                    job.message = f"Extracting embeddings ({done}/{total})"

            result = run_image_visualization(
                job.request.folders,
                model_names=[job.request.model_id],
                methods=["pca"],
                mode="detector" if job.request.unit == "object" else "classifier",
                unit=job.request.unit,
                object_policy=(
                    job.request.object_policy.model_dump()
                    if job.request.object_policy is not None else None
                ),
                model_loader=loader,
                object_model_loader=object_loader,
                include_raw_embeddings=True,
                progress=update_progress,
                cancelled=job.cancel_event.is_set,
            )
            if job.cancel_event.is_set():
                raise JobCancelled()

            vectors = result.pop("_raw_embeddings", {}).get(job.request.model_id)
            if vectors is None or len(vectors) != len(result["records"]):
                raise VisualizationInputError("INVALID_EMBEDDINGS", "The coverage embeddings are unavailable.")
            records = result["records"]
            labels = [str(record.get("label", "")) for record in records]
            coordinates, used_method, projection_warnings = coverage_projection(
                vectors,
                labels,
                job.request.method,
                job.request.dimension,
            )
            k = min(job.request.k, max(1, len(vectors) - 1))
            scores = coverage_sparsity(vectors, k)
            threshold, sparse_count = sparse_threshold(scores)
            run_id = uuid.uuid4().hex
            warnings = list(result.get("warnings", [])) + projection_warnings
            run = {
                "run_id": run_id,
                "model_id": job.request.model_id,
                "unit": job.request.unit,
                "object_policy": result.get("object_policy"),
                "class_names": result.get("class_names", []),
                "method": used_method,
                "requested_method": job.request.method,
                "dimension": job.request.dimension,
                "records": records,
                "image_paths": result["image_paths"],
                "points": coverage_points(records, coordinates),
                "embedding_shape": result["embedding_shape"],
                "k": k,
                "sparsity": np.asarray(scores, dtype=float).tolist(),
                "sparse_threshold": threshold,
                "sparse_count": sparse_count,
                "sparse_percentile": 90,
                "warnings": warnings,
            }
            with app.state.lock:
                app.state.coverage_runs.clear()
                app.state.coverage_embeddings.clear()
                app.state.coverage_sparsity.clear()
                app.state.coverage_image_stats.clear()
                app.state.coverage_completeness_candidates.clear()
                app.state.coverage_candidates.clear()
                app.state.coverage_runs[run_id] = run
                app.state.coverage_embeddings[run_id] = np.asarray(vectors, dtype=np.float32)
                app.state.coverage_sparsity[run_id] = {k: run["sparsity"]}
                job.run_id = run_id
                job.state, job.phase = "completed", "completed"
                job.done = job.total = len(records)
                job.message = "Completed"
        except JobCancelled:
            with app.state.lock:
                job.state, job.phase, job.message = "cancelled", "cancelled", "Cancelled"
        except VisualizationInputError as exc:
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", exc.code
                job.error = {"code": exc.code, "message": str(exc), "request_id": job.job_id}
        except FileNotFoundError:
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "MODEL_UNAVAILABLE"
                job.error = {
                    "code": "MODEL_UNAVAILABLE",
                    "message": "The selected model weights are unavailable.",
                    "request_id": job.job_id,
                }
        except Exception:
            logger.exception("LV web coverage analysis failed (job=%s)", job.job_id)
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "RUN_FAILED"
                job.error = {
                    "code": "RUN_FAILED",
                    "message": "The coverage analysis failed. Check the local worker log and retry.",
                    "request_id": job.job_id,
                }

    @app.post("/api/v1/coverage/jobs", status_code=202)
    def create_coverage_job(request: CreateCoverageJob) -> dict:
        resolved_folders = _resolve_request_folders(request)
        request = request.model_copy(update={
            "folders": resolved_folders,
            "folder_grant_ids": [],
        })
        job_id = uuid.uuid4().hex
        job = _CoverageJob(job_id=job_id, request=request)
        with app.state.lock:
            if has_active_job():
                raise HTTPException(status_code=409, detail={
                    "code": "JOB_ALREADY_RUNNING",
                    "message": "Wait for the active analysis to finish before starting another.",
                    "request_id": job_id,
                })
            app.state.coverage_jobs[job_id] = job
        threading.Thread(
            target=execute_coverage,
            args=(job,),
            daemon=True,
            name=f"lv-coverage-{job_id[:8]}",
        ).start()
        return {"job_id": job_id, "state": job.state}

    @app.get("/api/v1/coverage/jobs/{job_id}")
    def get_coverage_job(job_id: str) -> dict:
        with app.state.lock:
            job = app.state.coverage_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Coverage job not found")
            return _job_view(job)

    @app.delete("/api/v1/coverage/jobs/{job_id}", status_code=202)
    def cancel_coverage_job(job_id: str) -> dict:
        with app.state.lock:
            job = app.state.coverage_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Coverage job not found")
            if job.state in {"queued", "running"}:
                job.state, job.phase, job.message = "cancelling", "cancelling", "Cancellation requested"
                job.cancel_event.set()
            return {"job_id": job.job_id, "state": job.state}

    def execute_coverage_candidate_job(job: _CoverageCandidateJob) -> None:
        try:
            with app.state.lock:
                run = app.state.coverage_runs.get(job.run_id)
            if run is None:
                raise VisualizationInputError("RUN_NOT_FOUND", "The coverage run is no longer available.")
            if run["model_id"] not in set(catalog()):
                raise VisualizationInputError("MODEL_UNAVAILABLE", "The coverage model is no longer installed.")

            records: list[dict] = []
            image_paths: list[Path] = []
            embedding_batches: list[np.ndarray] = []
            warnings: list[str] = []
            mode = "detector" if run["unit"] == "object" else "classifier"
            fallback_names = job.request.class_names or run.get("class_names", [])
            for folder_value in job.request.folders:
                if job.cancel_event.is_set():
                    raise JobCancelled()
                base_done = job.done

                def update_candidate_progress(done: int, total: int) -> None:
                    with app.state.lock:
                        job.phase = "embedding"
                        job.done = base_done + done
                        job.total = base_done + total
                        job.message = f"Embedding candidate folder ({job.done}/{job.total})"

                try:
                    result = run_image_visualization(
                        [folder_value],
                        model_names=[run["model_id"]],
                        methods=["pca"],
                        mode=mode,
                        unit=run["unit"],
                        class_names=fallback_names,
                        object_policy=run.get("object_policy"),
                        model_loader=loader,
                        object_model_loader=object_loader,
                        include_raw_embeddings=True,
                        progress=update_candidate_progress,
                        cancelled=job.cancel_event.is_set,
                    )
                except VisualizationInputError as exc:
                    if exc.code in {"NO_IMAGES", "NO_OBJECTS", "INVALID_DETECTOR_DATASET"}:
                        warnings.append(f"{Path(folder_value).name}: {exc}")
                        continue
                    raise
                vectors = result.pop("_raw_embeddings", {}).get(run["model_id"])
                if vectors is None or len(vectors) != len(result["records"]):
                    raise VisualizationInputError(
                        "INVALID_EMBEDDINGS", "Candidate embeddings could not be aligned to their records.",
                    )
                offset = len(records)
                records.extend({**record, "index": offset + index}
                               for index, record in enumerate(result["records"])
                )
                image_paths.extend(Path(path) for path in result["image_paths"])
                embedding_batches.append(np.asarray(vectors, dtype=np.float32))
                warnings.extend(result.get("warnings", []))

            if not records or not embedding_batches:
                detail = "No readable candidate images were found in the selected folders."
                if warnings:
                    detail = "No candidate records were found. " + " ".join(warnings)
                raise VisualizationInputError("NO_CANDIDATES", detail)
            vectors = np.vstack(embedding_batches).astype(np.float32, copy=False)
            if vectors.ndim != 2 or vectors.shape[0] != len(records) or not np.isfinite(vectors).all():
                raise VisualizationInputError("INVALID_EMBEDDINGS", "Candidate embeddings are invalid.")
            candidate_set = {
                "run_id": job.run_id,
                "unit": run["unit"],
                "model_id": run["model_id"],
                "records": records,
                "image_paths": image_paths,
                "embeddings": vectors,
                "embedding_shape": [int(vectors.shape[0]), int(vectors.shape[1])],
                "warnings": warnings,
            }
            with app.state.lock:
                if app.state.coverage_runs.get(job.run_id) is None:
                    raise VisualizationInputError("RUN_NOT_FOUND", "The coverage run is no longer available.")
                app.state.coverage_candidates[job.run_id] = candidate_set
                job.state, job.phase, job.message = "completed", "completed", "Completed"
                job.done = job.total = len(records)
        except JobCancelled:
            with app.state.lock:
                job.state, job.phase, job.message = "cancelled", "cancelled", "Cancelled"
        except VisualizationInputError as exc:
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", exc.code
                job.error = {"code": exc.code, "message": str(exc), "request_id": job.job_id}
        except FileNotFoundError:
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "MODEL_UNAVAILABLE"
                job.error = {
                    "code": "MODEL_UNAVAILABLE",
                    "message": "The selected model weights are unavailable.",
                    "request_id": job.job_id,
                }
        except Exception:
            logger.exception("LV web coverage candidate analysis failed (job=%s)", job.job_id)
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "RUN_FAILED"
                job.error = {
                    "code": "RUN_FAILED",
                    "message": "Candidate analysis failed. Check the local worker log and retry.",
                    "request_id": job.job_id,
                }

    @app.post("/api/v1/coverage/runs/{run_id}/candidates/jobs", status_code=202)
    def create_coverage_candidate_job(run_id: str, request: CreateCoverageCandidateJob) -> dict:
        resolved_folders = _resolve_request_folders(request)
        request = request.model_copy(update={
            "folders": resolved_folders,
            "folder_grant_ids": [],
        })
        job_id = uuid.uuid4().hex
        with app.state.lock:
            if app.state.coverage_runs.get(run_id) is None:
                raise HTTPException(status_code=404, detail="Coverage run not found")
            if has_active_job():
                raise HTTPException(status_code=409, detail={
                    "code": "JOB_ALREADY_RUNNING",
                    "message": "Wait for the active analysis to finish before adding candidates.",
                    "request_id": job_id,
                })
            job = _CoverageCandidateJob(job_id=job_id, run_id=run_id, request=request)
            app.state.coverage_candidate_jobs[job_id] = job
        threading.Thread(
            target=execute_coverage_candidate_job,
            args=(job,),
            daemon=True,
            name=f"lv-coverage-candidates-{job_id[:8]}",
        ).start()
        return {"job_id": job_id, "state": job.state}

    @app.get("/api/v1/coverage/candidate-jobs/{job_id}")
    def get_coverage_candidate_job(job_id: str) -> dict:
        with app.state.lock:
            job = app.state.coverage_candidate_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Coverage candidate job not found")
            result = _job_view(job)
            result["run_id"] = job.run_id
            return result

    @app.delete("/api/v1/coverage/candidate-jobs/{job_id}", status_code=202)
    def cancel_coverage_candidate_job(job_id: str) -> dict:
        with app.state.lock:
            job = app.state.coverage_candidate_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Coverage candidate job not found")
            if job.state in {"queued", "running"}:
                job.state, job.phase, job.message = "cancelling", "cancelling", "Cancellation requested"
                job.cancel_event.set()
            return {"job_id": job.job_id, "state": job.state}

    @app.get("/api/v1/coverage/runs/{run_id}")
    def get_coverage_run(run_id: str) -> dict:
        with app.state.lock:
            run = app.state.coverage_runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Coverage run not found")
            return public_coverage_run(run)

    @app.post("/api/v1/coverage/runs/{run_id}/projections")
    def update_coverage_projection(run_id: str, request: CoverageProjectionRequest) -> dict:
        with app.state.lock:
            run = app.state.coverage_runs.get(run_id)
            vectors = app.state.coverage_embeddings.get(run_id)
            if run is None or vectors is None:
                raise HTTPException(status_code=404, detail="Coverage run not found")
            labels = [str(record.get("label", "")) for record in run["records"]]
            records = run["records"]
        try:
            coordinates, used_method, warnings = coverage_projection(
                vectors, labels, request.method, request.dimension,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with app.state.lock:
            current = app.state.coverage_runs.get(run_id)
            if current is None:
                raise HTTPException(status_code=404, detail="Coverage run not found")
            current["method"] = used_method
            current["requested_method"] = request.method
            current["dimension"] = request.dimension
            current["points"] = coverage_points(records, coordinates)
            current["warnings"] = warnings
            return {
                "method": used_method,
                "requested_method": request.method,
                "dimension": request.dimension,
                "points": current["points"],
                "warnings": warnings,
            }

    @app.post("/api/v1/coverage/runs/{run_id}/sparsity")
    def update_coverage_sparsity(run_id: str, request: CoverageSparsityRequest) -> dict:
        with app.state.lock:
            run = app.state.coverage_runs.get(run_id)
            vectors = app.state.coverage_embeddings.get(run_id)
            if run is None or vectors is None:
                raise HTTPException(status_code=404, detail="Coverage run not found")
            cache = app.state.coverage_sparsity.setdefault(run_id, {})
            k = min(request.k, max(1, len(vectors) - 1))
            scores = cache.get(k)
        if scores is None:
            try:
                scores = coverage_sparsity(vectors, k)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            scores = np.asarray(scores, dtype=float).tolist()
            with app.state.lock:
                app.state.coverage_sparsity.setdefault(run_id, {})[k] = scores
        threshold, sparse_count = sparse_threshold(np.asarray(scores, dtype=float))
        with app.state.lock:
            current = app.state.coverage_runs.get(run_id)
            if current is None:
                raise HTTPException(status_code=404, detail="Coverage run not found")
            current.update({
                "k": k,
                "sparsity": scores,
                "sparse_threshold": threshold,
                "sparse_count": sparse_count,
            })
        return {
            "k": k,
            "scores": scores,
            "threshold": threshold,
            "sparse_count": sparse_count,
            "sparse_percentile": 90,
        }

    @app.post("/api/v1/coverage/runs/{run_id}/completeness")
    def get_coverage_completeness(run_id: str, request: CoverageCompletenessRequest) -> dict:
        run, vectors, bucket_x, labels_x, bucket_y, labels_y = coverage_attribute_axes(
            run_id, request.axis_x, request.axis_y, request.bins,
        )
        records = run["records"]

        freq_classes: dict[tuple[int, int], str] = {}
        for item in request.frequencies:
            key = (item.x, item.y)
            if item.x >= len(labels_x) or item.y >= len(labels_y):
                raise HTTPException(status_code=422, detail="A frequency prior refers to a cell outside this grid")
            if key in freq_classes:
                raise HTTPException(status_code=422, detail="A grid cell can have only one frequency prior")
            freq_classes[key] = item.frequency

        result = build_completeness(
            records,
            vectors,
            bucket_x,
            bucket_y,
            labels_x,
            labels_y,
            t_abs=request.t_abs,
            d_star=request.d_star,
            freq_classes=freq_classes or None,
        )

        def public_cell(cell: dict) -> dict:
            return {
                "x": int(cell["x"]),
                "y": int(cell["y"]),
                "x_label": str(cell["x_label"]),
                "y_label": str(cell["y_label"]),
                "n": int(cell["n"]),
                "d": float(cell["d"]),
                "t": float(cell["t"]),
                "state": str(cell["state"]),
                "sample_indices": [int(index) for index in cell["indices"][:30]],
            }

        public_cells = [public_cell(cell) for cell in result["cells"]]
        top_gaps = [
            {**public_cell(gap), "shortfall": float(gap["shortfall"])}
            for gap in result["health"]["top_gaps"]
        ]
        health = {**result["health"], "top_gaps": top_gaps}
        with app.state.lock:
            if app.state.coverage_runs.get(run_id) is not run:
                raise HTTPException(status_code=404, detail="Coverage run expired")
        return {
            "run_id": run_id,
            "axis_x": request.axis_x,
            "axis_y": request.axis_y,
            "labels_x": labels_x,
            "labels_y": labels_y,
            "bins": request.bins,
            "t_abs": request.t_abs,
            "d_star": request.d_star,
            "calibrated": result["calibrated"],
            "frequency_classes": [
                {"x": x, "y": y, "frequency": frequency}
                for (x, y), frequency in sorted(freq_classes.items())
            ],
            "cells": public_cells,
            "health": health,
        }

    @app.post("/api/v1/coverage/runs/{run_id}/completeness/candidates")
    def mine_coverage_completeness_candidates(
        run_id: str,
        request: CoverageCompletenessCandidateRequest,
    ) -> dict:
        resolved_folders = _resolve_request_folders(request)
        request = request.model_copy(update={
            "folders": resolved_folders,
            "folder_grant_ids": [],
        })
        run, vectors, bucket_x, labels_x, bucket_y, labels_y = coverage_attribute_axes(
            run_id, request.axis_x, request.axis_y, request.bins,
        )
        if request.x >= len(labels_x) or request.y >= len(labels_y):
            raise HTTPException(status_code=422, detail="The selected cell is outside this grid")
        completeness = build_completeness(
            run["records"], vectors, bucket_x, bucket_y, labels_x, labels_y,
            t_abs=1,
        )
        cell = next(
            cell for cell in completeness["cells"]
            if (cell["x"], cell["y"]) == (request.x, request.y)
        )
        query = cell_centroid(vectors, cell["indices"])
        if query is None:
            # For an empty cell, use the full X-axis marginal as its query seed.
            # With the default label axis this matches the legacy class fallback.
            if request.axis_x == "label":
                fallback_indices = [
                    index for index, record in enumerate(run["records"])
                    if str(record.get("label", "")) == cell["x_label"]
                ]
            else:
                fallback_indices = [
                    index for index, bucket in enumerate(bucket_x)
                    if bucket == request.x
                ]
            query = cell_centroid(vectors, fallback_indices)
        if query is None:
            raise HTTPException(status_code=422, detail="This cell has no sample or class-marginal query vector")
        try:
            candidate_run = run_image_visualization(
                request.folders,
                model_name=run["model_id"],
                methods=["pca"],
                mode="classifier",
                unit="image",
                model_loader=loader,
                include_raw_embeddings=True,
            )
        except VisualizationInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        candidate_vectors = candidate_run["_raw_embeddings"][run["model_id"]]
        candidate_indices, distances = mine_candidates(candidate_vectors, query, k=12)
        pool_indices = [int(index) for index in candidate_indices]
        items = [
            {
                "index": index,
                "filename": str(candidate_run["records"][index]["filename"]),
                "split": str(candidate_run["records"][index]["split"]),
                "label": str(candidate_run["records"][index]["label"]),
                "distance": float(distance),
            }
            for index, distance in zip(pool_indices, distances)
        ]
        with app.state.lock:
            if app.state.coverage_runs.get(run_id) is not run:
                raise HTTPException(status_code=404, detail="Coverage run expired")
            app.state.coverage_completeness_candidates[run_id] = {
                "cell": (request.x, request.y),
                "items": items,
                "records": candidate_run["records"],
                "image_paths": candidate_run["image_paths"],
            }
        return {
            "run_id": run_id,
            "cell": {"x": request.x, "y": request.y},
            "items": items,
            "total": len(items),
        }

    @app.get("/api/v1/coverage/runs/{run_id}/completeness/candidates/{index}/thumbnail")
    def get_coverage_completeness_candidate_thumbnail(run_id: str, index: int) -> Response:
        with app.state.lock:
            candidate_set = app.state.coverage_completeness_candidates.get(run_id)
            if candidate_set is None:
                raise HTTPException(status_code=404, detail="Cell candidates not found")
            items = candidate_set["items"]
            if not any(int(item["index"]) == index for item in items):
                raise HTTPException(status_code=404, detail="Candidate image not found")
            path = candidate_set["image_paths"][index]
        thumbnail = safe_open_image(path)
        if thumbnail is None:
            raise HTTPException(status_code=404, detail="Candidate image is no longer available")
        thumbnail.thumbnail((256, 256))
        output = BytesIO()
        thumbnail.save(output, format="JPEG", quality=84, optimize=True)
        return Response(
            content=output.getvalue(),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/v1/coverage/runs/{run_id}/completeness/candidates.csv")
    def export_coverage_completeness_candidates(run_id: str) -> Response:
        with app.state.lock:
            candidate_set = app.state.coverage_completeness_candidates.get(run_id)
            if candidate_set is None:
                raise HTTPException(status_code=404, detail="Cell candidates not found")
            paths: list[Path] = candidate_set["image_paths"]
            rows = [
                [str(paths[int(item["index"])]), f"{float(item['distance']):.6f}"]
                for item in candidate_set["items"]
            ]
        buffer = StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["path", "distance"])
        writer.writerows(rows)
        return Response(
            content=buffer.getvalue().encode("utf-8-sig"),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="cell_candidates.csv"'},
        )

    @app.post("/api/v1/coverage/runs/{run_id}/object-view")
    def get_coverage_object_view(run_id: str, request: CoverageObjectViewRequest) -> dict:
        if request.confidence_min > request.confidence_max:
            raise HTTPException(status_code=422, detail="Minimum confidence must not exceed maximum confidence")
        with app.state.lock:
            run = app.state.coverage_runs.get(run_id)
            vectors = app.state.coverage_embeddings.get(run_id)
            if run is None or vectors is None:
                raise HTTPException(status_code=404, detail="Coverage run not found")
            if run.get("unit") != "object":
                raise HTTPException(status_code=409, detail="Object filters require object-level coverage")
            records = run["records"]
            vectors = np.asarray(vectors)

        kept = []
        for index, record in enumerate(records):
            short_px = int(record.get("short_px", 0) or 0)
            score = record.get("score")
            if short_px < request.min_short_px:
                continue
            if score is not None and not request.confidence_min <= float(score) <= request.confidence_max:
                continue
            kept.append(index)

        labels_after_object_filters = [str(records[index].get("label", "")) for index in kept]
        class_summary = []
        for label in sorted(set(labels_after_object_filters)):
            class_indices = [index for index in kept if str(records[index].get("label", "")) == label]
            class_vectors = vectors[class_indices]
            if len(class_indices) >= 2:
                class_scores = coverage_sparsity(class_vectors, min(request.k, len(class_indices) - 1))
                threshold = float(np.percentile(class_scores, 80))
                sparse_count = int(np.sum(np.asarray(class_scores) >= threshold))
                max_sparsity = float(np.max(class_scores))
            else:
                sparse_count = 0
                max_sparsity = 0.0
            class_summary.append({
                "label": label,
                "count": len(class_indices),
                "analyzable": len(class_indices) >= 4,
                "sparse_count": sparse_count,
                "max_sparsity": max_sparsity,
            })
        class_summary.sort(key=lambda row: (row["count"], -row["max_sparsity"], row["label"]))

        selected_indices = [
            index for index in kept
            if request.class_label is None
            or str(records[index].get("label", "")) == request.class_label
        ]
        selected_records = [records[index] for index in selected_indices]
        selected_vectors = vectors[selected_indices]
        selected_labels = [str(record.get("label", "")) for record in selected_records]
        warnings: list[str] = []
        if selected_records:
            try:
                coordinates, used_method, projection_warnings = coverage_projection(
                    selected_vectors,
                    selected_labels,
                    request.method,
                    request.dimension,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            warnings.extend(projection_warnings)
        else:
            coordinates = np.zeros((0, request.dimension), dtype=np.float64)
            used_method = request.method
            warnings.append("No objects match the current class and filters.")

        k = min(request.k, max(1, len(selected_vectors) - 1))
        scores = coverage_sparsity(selected_vectors, k)
        threshold, sparse_count = sparse_threshold(np.asarray(scores, dtype=np.float64), 80.0)
        block_summaries: list[dict] = []
        if request.class_label is not None and len(selected_vectors) >= 2:
            if len(selected_vectors) < 4:
                warnings.append(
                    f"{request.class_label} has only {len(selected_vectors)} objects; block coverage may be unstable."
                )
            normalized = selected_vectors / np.clip(
                np.linalg.norm(selected_vectors, axis=1, keepdims=True), 1e-12, None,
            )
            block_count = min(request.block_k, len(selected_vectors))
            fitted = KMeans(n_clusters=block_count, n_init=4, random_state=42).fit(normalized)
            points = coverage_points(selected_records, coordinates)
            labels_by_block = fitted.labels_.astype(int)
            for point, block_id in zip(points, labels_by_block):
                point["block_id"] = int(block_id)
            for block_id in sorted(set(labels_by_block.tolist())):
                members = np.flatnonzero(labels_by_block == block_id)
                block_summaries.append({
                    "block_id": int(block_id),
                    "count": int(len(members)),
                    "mean_sparsity": float(np.mean(np.asarray(scores)[members])),
                })
        else:
            points = coverage_points(selected_records, coordinates)
            if request.class_label is not None and len(selected_vectors) < 2:
                warnings.append(
                    f"{request.class_label} has fewer than two objects; block analysis is unavailable."
                )

        return {
            "class_label": request.class_label,
            "method": used_method,
            "requested_method": request.method,
            "dimension": request.dimension,
            "records": selected_records,
            "points": points,
            "k": k,
            "sparsity": np.asarray(scores, dtype=float).tolist(),
            "sparse_threshold": threshold,
            "sparse_count": sparse_count,
            "class_summary": class_summary,
            "block_k": min(request.block_k, len(selected_vectors)) if len(selected_vectors) >= 2 else 0,
            "blocks": block_summaries,
            "has_confidence": any(record.get("score") is not None for record in records),
            "warnings": warnings,
        }

    @app.post("/api/v1/coverage/runs/{run_id}/candidate-view")
    def get_coverage_candidate_view(run_id: str, request: CoverageCandidateViewRequest) -> dict:
        if request.confidence_min > request.confidence_max:
            raise HTTPException(status_code=422, detail="Minimum confidence must not exceed maximum confidence")
        with app.state.lock:
            run = app.state.coverage_runs.get(run_id)
            main_vectors = app.state.coverage_embeddings.get(run_id)
            candidate_set = app.state.coverage_candidates.get(run_id)
            if run is None or main_vectors is None:
                raise HTTPException(status_code=404, detail="Coverage run not found")
            if candidate_set is None:
                raise HTTPException(status_code=409, detail="Project candidate folders before requesting a candidate view")
            main_records = run["records"]
            candidate_records = candidate_set["records"]
            main_vectors = np.asarray(main_vectors)
            candidate_vectors = np.asarray(candidate_set["embeddings"])

        def keep_record(record: dict) -> bool:
            if run["unit"] != "object":
                return True
            short_px = int(record.get("short_px", 0) or 0)
            if short_px < request.min_short_px:
                return False
            score = record.get("score")
            if score is not None and not request.confidence_min <= float(score) <= request.confidence_max:
                return False
            return True

        main_indices = [
            index for index, record in enumerate(main_records)
            if keep_record(record)
            and (request.class_label is None or str(record.get("label", "")) == request.class_label)
        ]
        candidate_indices = [
            index for index, record in enumerate(candidate_records)
            if keep_record(record)
            and (request.class_label is None or str(record.get("label", "")) == request.class_label)
        ]
        selected_main_records = [main_records[index] for index in main_indices]
        selected_candidate_records = [candidate_records[index] for index in candidate_indices]
        selected_main_vectors = main_vectors[main_indices]
        selected_candidate_vectors = candidate_vectors[candidate_indices]
        main_labels = [str(record.get("label", "")) for record in selected_main_records]
        candidate_labels = [str(record.get("label", "")) for record in selected_candidate_records]
        warnings = list(candidate_set.get("warnings", []))
        if request.class_label is not None and not selected_candidate_records:
            warnings.append(
                f"No candidate objects match {request.class_label}. Check candidate class names or choose another class."
            )
        elif not selected_candidate_records:
            warnings.append("No candidates match the current confidence and source-size filters.")
        if not selected_main_records:
            warnings.append("No main records match the current class and object filters; coverage cannot be measured.")

        bounded_k = min(request.k, max(1, len(selected_main_vectors) - 1))
        main_sparsity = coverage_sparsity(selected_main_vectors, bounded_k)
        radius = 0.0
        if len(selected_main_vectors) >= 2:
            nn = NearestNeighbors(metric="cosine", n_neighbors=2).fit(selected_main_vectors)
            distances, _ = nn.kneighbors(selected_main_vectors)
            radius = float(np.percentile(distances[:, 1], 75))

        provisional_labels = nearest_labels(
            selected_candidate_vectors, selected_main_vectors, main_labels,
        )
        candidate_scores_by_local: dict[int, float] = {}
        if not len(selected_main_vectors):
            ranked_local: list[int] = []
            work_local: list[int] = []
            recall = None
            warnings.append("Candidate ranking needs at least one main record.")
        elif not len(selected_candidate_vectors):
            ranked_local = []
            work_local = []
            recall = None
            if request.role == "reference":
                warnings.append("Reference coverage is unavailable because no candidate rows remain after filtering.")
        elif request.role == "pool":
            ranked_local, ranked_scores = rank_gap_fillers(
                selected_candidate_vectors, selected_main_vectors, k=bounded_k,
            )
            candidate_scores_by_local = {
                index: float(score) for index, score in zip(ranked_local, ranked_scores)
            }
            work_local = ranked_local
            recall = None
        else:
            uncovered_local, recall, distances = reference_coverage(
                selected_main_vectors, selected_candidate_vectors, radius,
            )
            candidate_scores_by_local = {
                index: float(distances[index]) for index in range(len(distances))
            }
            work_local = uncovered_local
            ranked_local = uncovered_local

        combined_vectors = np.vstack([selected_main_vectors, selected_candidate_vectors])
        combined_labels = main_labels + provisional_labels
        if len(combined_vectors):
            try:
                coordinates, used_method, projection_warnings = coverage_projection(
                    combined_vectors,
                    combined_labels,
                    request.method,
                    request.dimension,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            warnings.extend(projection_warnings)
            main_coordinates = coordinates[:len(selected_main_vectors)]
            candidate_coordinates = coordinates[len(selected_main_vectors):]
        else:
            used_method = request.method
            main_coordinates = np.zeros((0, request.dimension), dtype=np.float64)
            candidate_coordinates = np.zeros((0, request.dimension), dtype=np.float64)
        main_points = coverage_points(selected_main_records, main_coordinates)
        candidate_points = []
        score_by_global_index: dict[int, float] = {}
        work_indices = []
        work_scores: dict[int, float] = {}
        for local_index in ranked_local:
            global_index = candidate_indices[local_index]
            score = candidate_scores_by_local.get(local_index, 0.0)
            score_by_global_index[global_index] = score
        for local_index in work_local:
            global_index = candidate_indices[local_index]
            work_indices.append(global_index)
            work_scores[global_index] = candidate_scores_by_local.get(local_index, 0.0)
        for local_index, record in enumerate(selected_candidate_records):
            candidate_points.append({
                **record,
                "index": int(record.get("index", candidate_indices[local_index])),
                "x": float(candidate_coordinates[local_index, 0]),
                "y": float(candidate_coordinates[local_index, 1]),
                "z": float(candidate_coordinates[local_index, 2])
                if candidate_coordinates.shape[1] > 2 else 0.0,
                "score": score_by_global_index.get(candidate_indices[local_index]),
                "provisional_label": provisional_labels[local_index],
                "role": request.role,
            })

        class_gaps = []
        if run["unit"] == "object" and request.role == "reference" and request.class_label is None:
            main_counts: dict[str, int] = {}
            candidate_counts: dict[str, int] = {}
            uncovered_counts: dict[str, int] = {}
            uncovered_set = set(work_indices)
            for record in selected_main_records:
                label = str(record.get("label", ""))
                main_counts[label] = main_counts.get(label, 0) + 1
            for record in selected_candidate_records:
                label = str(record.get("label", ""))
                candidate_counts[label] = candidate_counts.get(label, 0) + 1
                if int(record.get("index", -1)) in uncovered_set:
                    uncovered_counts[label] = uncovered_counts.get(label, 0) + 1
            for label in sorted(candidate_counts, key=lambda item: (
                main_counts.get(item, 0) != 0, -uncovered_counts.get(item, 0), item,
            )):
                count = candidate_counts[label]
                uncovered_count = uncovered_counts.get(label, 0)
                if main_counts.get(label, 0) == 0:
                    state = "missing"
                elif uncovered_count / max(count, 1) > 0.6:
                    state = "sparse"
                else:
                    state = "covered"
                class_gaps.append({
                    "label": label,
                    "main_count": main_counts.get(label, 0),
                    "candidate_count": count,
                    "uncovered_count": uncovered_count,
                    "state": state,
                })

        selected_work = [
            {
                **candidate_records[index],
                "score": work_scores.get(index, 0.0),
                "provisional_label": next((
                    point["provisional_label"] for point in candidate_points
                    if point["index"] == index
                ), ""),
            }
            for index in work_indices[:request.limit]
        ]
        main_threshold, main_sparse_count = sparse_threshold(
            np.asarray(main_sparsity, dtype=np.float64), 80.0,
        )
        return {
            "role": request.role,
            "class_label": request.class_label,
            "method": used_method,
            "requested_method": request.method,
            "dimension": request.dimension,
            "main_count": len(selected_main_records),
            "candidate_count": len(selected_candidate_records),
            "main_points": main_points,
            "candidate_points": candidate_points,
            "main_sparsity": np.asarray(main_sparsity, dtype=float).tolist(),
            "main_sparse_threshold": main_threshold,
            "main_sparse_count": main_sparse_count,
            "k": bounded_k,
            "radius": radius if len(selected_main_vectors) >= 2 else None,
            "recall": recall,
            "uncovered_count": len(work_indices) if request.role == "reference" else 0,
            "work_indices": work_indices,
            "work_scores": work_scores,
            "work_records": selected_work,
            "class_gaps": class_gaps,
            "warnings": warnings,
        }

    @app.get("/api/v1/coverage/runs/{run_id}/images/{index}/thumbnail")
    def get_coverage_thumbnail(run_id: str, index: int) -> Response:
        with app.state.lock:
            run = app.state.coverage_runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Coverage run not found")
            paths: list[Path] = run["image_paths"]
            if index < 0 or index >= len(paths):
                raise HTTPException(status_code=404, detail="Image not found")
            path = paths[index]
        thumbnail = safe_open_image(path)
        if thumbnail is None:
            raise HTTPException(status_code=404, detail="Image is no longer available")
        if run.get("unit") == "object":
            record = run["records"][index]
            bbox = record.get("bbox")
            if bbox is not None:
                width, height = thumbnail.size
                long_side = max(float(bbox[2]) * width, float(bbox[3]) * height)
                pad_px = int(min(64, max(8, round(0.12 * long_side))))
                thumbnail = crop_bbox(thumbnail, *bbox, pad_px=pad_px)
        thumbnail.thumbnail((256, 256))
        output = BytesIO()
        thumbnail.save(output, format="JPEG", quality=84, optimize=True)
        return Response(
            content=output.getvalue(),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/v1/coverage/runs/{run_id}/candidates/{index}/thumbnail")
    def get_coverage_candidate_thumbnail(run_id: str, index: int) -> Response:
        with app.state.lock:
            candidate_set = app.state.coverage_candidates.get(run_id)
            if candidate_set is None:
                raise HTTPException(status_code=404, detail="Coverage candidates not found")
            paths: list[Path] = candidate_set["image_paths"]
            if index < 0 or index >= len(paths):
                raise HTTPException(status_code=404, detail="Candidate image not found")
            path = paths[index]
            record = candidate_set["records"][index]
            unit = candidate_set["unit"]
        thumbnail = safe_open_image(path)
        if thumbnail is None:
            raise HTTPException(status_code=404, detail="Candidate image is no longer available")
        if unit == "object":
            bbox = record.get("bbox")
            if bbox is not None:
                width, height = thumbnail.size
                long_side = max(float(bbox[2]) * width, float(bbox[3]) * height)
                pad_px = int(min(64, max(8, round(0.12 * long_side))))
                thumbnail = crop_bbox(thumbnail, *bbox, pad_px=pad_px)
        thumbnail.thumbnail((256, 256))
        output = BytesIO()
        thumbnail.save(output, format="JPEG", quality=84, optimize=True)
        return Response(
            content=output.getvalue(),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    def execute_comparison(job: _ComparisonJob) -> None:
        try:
            with app.state.lock:
                job.state, job.phase, job.message = "running", "discovery", "Scanning both image groups"
            if job.request.model_id not in set(catalog()):
                raise ComparisonInputError("MODEL_UNAVAILABLE", "The selected model is not installed.")

            def update_progress(phase: str, done: int, total: int, message: str) -> None:
                with app.state.lock:
                    job.phase = phase
                    job.done = done
                    job.total = total
                    job.message = message

            result = run_comparison(
                job.request.folders_a,
                job.request.folders_b,
                job.request.model_id,
                unit=job.request.unit,
                class_names=job.request.class_names,
                object_policy=(
                    job.request.object_policy.model_dump()
                    if job.request.object_policy is not None else None
                ),
                model_loader=loader,
                object_model_loader=object_loader,
                progress=update_progress,
                cancelled=job.cancel_event.is_set,
            )
            if job.cancel_event.is_set():
                raise JobCancelled()

            run_id = uuid.uuid4().hex
            with app.state.lock:
                app.state.comparison_runs[run_id] = result
                job.run_id = run_id
                job.state, job.phase = "completed", "completed"
                job.done = job.total = max(job.total, len(result["records_a"]) + len(result["records_b"]))
                job.message = "Completed"
        except JobCancelled:
            with app.state.lock:
                job.state, job.phase, job.message = "cancelled", "cancelled", "Cancelled"
        except ComparisonInputError as exc:
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", exc.code
                job.error = {"code": exc.code, "message": str(exc), "request_id": job.job_id}
        except FileNotFoundError:
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "MODEL_UNAVAILABLE"
                job.error = {
                    "code": "MODEL_UNAVAILABLE",
                    "message": "The selected model weights are unavailable.",
                    "request_id": job.job_id,
                }
        except Exception:
            logger.exception("LV web comparison failed (job=%s)", job.job_id)
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "RUN_FAILED"
                job.error = {
                    "code": "RUN_FAILED",
                    "message": "The comparison failed. Check the local worker log and retry.",
                    "request_id": job.job_id,
                }

    @app.post("/api/v1/compare/jobs", status_code=202)
    def create_comparison_job(request: CreateComparisonJob) -> dict:
        folders_a, folders_b = _resolve_comparison_folders(request)
        request = request.model_copy(update={
            "folders_a": folders_a,
            "folder_grant_ids_a": [],
            "folders_b": folders_b,
            "folder_grant_ids_b": [],
        })
        job_id = uuid.uuid4().hex
        job = _ComparisonJob(job_id=job_id, request=request)
        with app.state.lock:
            if has_active_job():
                raise HTTPException(status_code=409, detail={
                    "code": "JOB_ALREADY_RUNNING",
                    "message": "Wait for the active analysis to finish before starting another.",
                    "request_id": job_id,
                })
            app.state.comparison_jobs[job_id] = job
        threading.Thread(
            target=execute_comparison, args=(job,), daemon=True,
            name=f"lv-compare-{job_id[:8]}",
        ).start()
        return {"job_id": job_id, "state": job.state}

    def groupdiff_result(run_id: str) -> dict:
        with app.state.lock:
            result = app.state.groupdiff_runs.get(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="差異分析結果不存在或已過期")
        return result

    def groupdiff_run_view(run_id: str, result: dict) -> dict:
        regions = []
        for region in result["regions"]:
            regions.append({
                key: region[key]
                for key in ("rank", "peak_z", "mean_z", "stability", "pct", "text", "bbox_px")
            } | {
                "good_count": len(region.get("good_paths", [])),
                "bad_count": len(region.get("bad_paths", [])),
            })
        return {
            "run_id": run_id,
            "grid": [int(value) for value in result["grid"]],
            "score_map": np.asarray(result["score_map"], dtype=float).tolist(),
            "z_map": np.asarray(result["z_map"], dtype=float).tolist(),
            "stability_map": np.asarray(result["stability_map"], dtype=float).tolist(),
            "verdict": bool(result["verdict"]),
            "p_global": float(result["p_global"]),
            "threshold_z": float(result["threshold_z"]),
            "regions": regions,
            "n_good": int(result["n_good"]),
            "n_bad": int(result["n_bad"]),
            "skipped": dict(result["skipped"]),
            "alignment_warning": bool(result["alignment_warning"]),
            "warn_msgs": list(result["warn_msgs"]),
            "meta": dict(result["meta"]),
            "bases_good": [Path(path).name for path in result["good_paths"]],
            "bases_bad": [Path(path).name for path in result["bad_paths"]],
            "base_good_index": result["good_paths"].index(result["base_good"]),
            "base_bad_index": result["bad_paths"].index(result["base_bad"]),
        }

    def safe_groupdiff_error(message: str, request: CreateGroupDiffJob) -> str:
        safe = message
        for folder in (*request.good_dirs, *request.bad_dirs):
            try:
                safe = safe.replace(str(Path(folder).expanduser().resolve()), "selected folder")
            except (OSError, RuntimeError):
                safe = safe.replace(folder, "selected folder")
        return safe

    def execute_groupdiff(job: _GroupDiffJob) -> None:
        try:
            with app.state.lock:
                job.state, job.phase, job.message = "running", "discovery", "正在掃描 Good／Bad 影像"
            from groupdiff_pipeline import run_groupdiff

            def update_progress(frac: float, message: str) -> None:
                with app.state.lock:
                    job.phase = "analysis"
                    job.done = max(job.done, min(100, int(float(frac) * 100)))
                    job.message = str(message)

            result = run_groupdiff(
                job.request.good_dirs,
                job.request.bad_dirs,
                model=job.request.model_id,
                n_perm=job.request.n_perm,
                alpha=job.request.alpha,
                top_k=job.request.top_k,
                seed=job.request.seed,
                extractor=groupdiff_extractor,
                progress=update_progress,
            )
            run_id = uuid.uuid4().hex
            with app.state.lock:
                app.state.groupdiff_runs[run_id] = result
                while len(app.state.groupdiff_runs) > 8:
                    app.state.groupdiff_runs.pop(next(iter(app.state.groupdiff_runs)))
                job.run_id = run_id
                job.state, job.phase = "completed", "completed"
                job.done = job.total = 100
                job.message = "分析完成"
        except FileNotFoundError:
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "MODEL_UNAVAILABLE"
                job.error = {
                    "code": "MODEL_UNAVAILABLE",
                    "message": "所選 DINOv2 模型權重無法載入。",
                    "request_id": job.job_id,
                }
        except ValueError as exc:
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "INVALID_GROUPDIFF_INPUT"
                job.error = {
                    "code": "INVALID_GROUPDIFF_INPUT",
                    "message": safe_groupdiff_error(str(exc), job.request),
                    "request_id": job.job_id,
                }
        except Exception:
            logger.exception("LV web group difference analysis failed (job=%s)", job.job_id)
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "GROUPDIFF_FAILED"
                job.error = {
                    "code": "GROUPDIFF_FAILED",
                    "message": "群組差異分析失敗。請檢查本機模型權重、影像資料與 worker 記錄。",
                    "request_id": job.job_id,
                }

    @app.post("/api/v1/groupdiff/jobs", status_code=202)
    def create_groupdiff_job(request: CreateGroupDiffJob) -> dict:
        try:
            available = set(catalog())
        except Exception as exc:
            raise HTTPException(status_code=503, detail="無法讀取本機模型清單") from exc
        if request.model_id not in available:
            raise HTTPException(status_code=422, detail="所選 DINOv2 模型不存在或尚未安裝")
        if groupdiff_extractor is None and not request.model_id.startswith("dinov2_vit"):
            raise HTTPException(status_code=422, detail="群組 patch 差異分析目前只支援 DINOv2 模型")

        def normalize_folders(values: list[str]) -> list[str]:
            normalized: list[str] = []
            seen: set[str] = set()
            for value in values:
                try:
                    folder = Path(value).expanduser().resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    raise HTTPException(status_code=422, detail="Good／Bad 資料夾不存在或無法讀取") from exc
                if not folder.is_dir():
                    raise HTTPException(status_code=422, detail="Good／Bad 資料夾不存在或無法讀取")
                key = os.path.normcase(str(folder))
                if key not in seen:
                    seen.add(key)
                    normalized.append(str(folder))
            return normalized

        normalized = request.model_copy(update={
            "good_dirs": normalize_folders(request.good_dirs),
            "bad_dirs": normalize_folders(request.bad_dirs),
        })
        job_id = uuid.uuid4().hex
        job = _GroupDiffJob(job_id=job_id, request=normalized)
        with app.state.lock:
            if has_active_job():
                raise HTTPException(status_code=409, detail={
                    "code": "JOB_ALREADY_RUNNING",
                    "message": "請等目前分析完成後再開始 Good／Bad 差異分析。",
                    "request_id": job_id,
                })
            app.state.groupdiff_jobs[job_id] = job
        threading.Thread(
            target=execute_groupdiff,
            args=(job,),
            daemon=True,
            name=f"lv-groupdiff-{job_id[:8]}",
        ).start()
        return {"job_id": job_id, "state": job.state}

    @app.get("/api/v1/groupdiff/jobs/{job_id}")
    def get_groupdiff_job(job_id: str) -> dict:
        with app.state.lock:
            job = app.state.groupdiff_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="找不到差異分析工作")
            return _job_view(job)

    @app.get("/api/v1/groupdiff/runs/{run_id}")
    def get_groupdiff_run(run_id: str) -> dict:
        return groupdiff_run_view(run_id, groupdiff_result(run_id))

    @app.get("/api/v1/groupdiff/runs/{run_id}/bases/{side}/{index}")
    def get_groupdiff_base(run_id: str, side: str, index: int,
                           overlay: bool = False, alpha: float = 0.55) -> Response:
        result = groupdiff_result(run_id)
        if side not in {"good", "bad"} or not 0.0 <= alpha <= 1.0:
            raise HTTPException(status_code=422, detail="底圖選項無效")
        paths = result["good_paths"] if side == "good" else result["bad_paths"]
        if index < 0 or index >= len(paths):
            raise HTTPException(status_code=404, detail="找不到差異分析底圖")
        from safe_io import safe_open_image
        image = safe_open_image(paths[index])
        if image is None:
            raise HTTPException(status_code=404, detail="差異分析底圖已無法讀取")
        image = image.convert("RGB")
        if overlay:
            from anomaly_heatmap import render_heatmap
            vmax = float(max(float(np.max(result["z_map"])), result["threshold_z"], 1e-9))
            image = render_heatmap(result["z_map"], image, alpha=alpha, vmin=0.0, vmax=vmax)
        image.thumbnail((1400, 1400))
        output = BytesIO()
        image.save(output, format="PNG", optimize=True)
        return Response(output.getvalue(), media_type="image/png", headers={
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
        })

    @app.get("/api/v1/groupdiff/runs/{run_id}/regions/{rank}/images/{side}/{index}.jpg")
    def get_groupdiff_region_image(run_id: str, rank: int, side: str, index: int) -> Response:
        result = groupdiff_result(run_id)
        if side not in {"good", "bad"}:
            raise HTTPException(status_code=422, detail="群組選項無效")
        region = next((item for item in result["regions"] if item["rank"] == rank), None)
        if region is None or index < 0 or index >= len(region.get(f"{side}_paths", [])):
            raise HTTPException(status_code=404, detail="找不到此差異區域代表圖")
        from groupdiff_pipeline import region_compare_images
        selected_region = dict(region)
        selected_region["good_paths"] = [region["good_paths"][index]] if side == "good" else []
        selected_region["bad_paths"] = [region["bad_paths"][index]] if side == "bad" else []
        images = region_compare_images(result, selected_region, n_each=1)
        if not images[side]:
            raise HTTPException(status_code=404, detail="差異區域代表圖已無法讀取")
        image = images[side][0]
        image.thumbnail((900, 900))
        output = BytesIO()
        image.save(output, format="JPEG", quality=88, optimize=True)
        return Response(output.getvalue(), media_type="image/jpeg", headers={
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
        })

    @app.get("/api/v1/groupdiff/runs/{run_id}/pca")
    def get_groupdiff_pca(run_id: str) -> dict:
        result = groupdiff_result(run_id)
        from groupdiff_pipeline import pca2d
        try:
            projected = pca2d(result["gmean_good"], result["gmean_bad"])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "xy_good": np.asarray(projected["xy_good"], dtype=float).tolist(),
            "xy_bad": np.asarray(projected["xy_bad"], dtype=float).tolist(),
            "evr": np.asarray(projected["evr"], dtype=float).tolist(),
        }

    @app.post("/api/v1/groupdiff/runs/{run_id}/exports")
    def export_groupdiff_run(run_id: str, request: GroupDiffExportRequest) -> dict:
        result = groupdiff_result(run_id)
        output_dir = request.output_dir.strip()
        if not output_dir:
            raise HTTPException(status_code=422, detail="請選擇差異分析報告的輸出資料夾")
        try:
            from groupdiff_pipeline import export_report
            report = export_report(result, output_dir.strip())
        except (ValueError, OSError) as exc:
            raise HTTPException(
                status_code=422,
                detail="匯出失敗。請確認輸出資料夾位於 Good／Bad 來源之外且有寫入權限。",
            ) from exc
        return {"count": len(report["files"]), "files": [Path(path).name for path in report["files"]]}

    @app.get("/api/v1/compare/jobs/{job_id}")
    def get_comparison_job(job_id: str) -> dict:
        with app.state.lock:
            job = app.state.comparison_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            return _job_view(job)

    @app.delete("/api/v1/compare/jobs/{job_id}", status_code=202)
    def cancel_comparison_job(job_id: str) -> dict:
        with app.state.lock:
            job = app.state.comparison_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            if job.state not in {"queued", "running", "cancelling"}:
                raise HTTPException(status_code=409, detail="Job is already finished")
            job.cancel_event.set()
            job.state, job.message = "cancelling", "Cancellation requested"
            return {"job_id": job_id, "state": job.state}

    @app.get("/api/v1/comparisons/{run_id}")
    def get_comparison(run_id: str) -> dict:
        with app.state.lock:
            comparison = app.state.comparison_runs.get(run_id)
            if comparison is None:
                raise HTTPException(status_code=404, detail="Comparison not found")
            return {key: value for key, value in comparison.items() if not key.startswith("_")}

    @app.post("/api/v1/comparisons/{run_id}/curation/items", status_code=201)
    def add_comparison_records_to_curation(
        run_id: str,
        request: AddComparisonCurationRequest,
    ) -> dict:
        with app.state.lock:
            comparison = app.state.comparison_runs.get(run_id)
            if comparison is None:
                raise HTTPException(status_code=404, detail="Comparison not found")
            selections = list(dict.fromkeys(
                (selection.group, selection.index) for selection in request.selections
            ))
            for group, index in selections:
                side = group.lower()
                records = comparison[f"records_{side}"]
                paths = comparison[f"_record_paths_{side}"]
                if index >= len(records) or index >= len(paths):
                    raise HTTPException(status_code=422, detail={
                        "code": "INVALID_COMPARISON_RECORD_INDEX",
                        "message": "A selected comparison record is no longer available.",
                        "request_id": uuid.uuid4().hex,
                    })

            entries = []
            seen_paths: set[str] = set()
            for group, index in selections:
                side = group.lower()
                record = comparison[f"records_{side}"][index]
                try:
                    path = Path(comparison[f"_record_paths_{side}"][index]).expanduser().resolve(strict=True)
                    if not path.is_file():
                        raise OSError("source is not a file")
                    path_key = os.path.normcase(str(path))
                    if path_key in seen_paths:
                        continue
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                except OSError as exc:
                    raise HTTPException(status_code=404, detail={
                        "code": "CURATION_IMAGE_UNAVAILABLE",
                        "message": "A selected comparison image is no longer available.",
                        "request_id": uuid.uuid4().hex,
                    }) from exc
                seen_paths.add(path_key)
                entries.append({
                    "index": index,
                    "filename": str(record.get("filename", path.name)),
                    "path": str(path),
                    "label": str(record.get("label", "")),
                    "split": str(record.get("split", "")),
                    "sha256": digest.hexdigest(),
                    "source": "compare",
                    "score": record.get("score"),
                    "reason": request.reason.strip(),
                    # Streamlit Compare adds the selected source image to its image-level cart;
                    # several selected boxes from one file therefore produce one cart item.
                    "level": "image",
                })
            try:
                added, skipped = app.state.curation_store.add(entries)
                items = app.state.curation_store.list_items()
            except CurationStoreError as exc:
                raise curation_store_error() from exc
        return {
            "added": added,
            "skipped": skipped,
            "items": [public_curation_item(item) for item in items],
            "total": len(items),
        }

    @app.post("/api/v1/comparisons/{run_id}/projections")
    def get_comparison_projection(run_id: str, request: ProjectComparisonRequest) -> dict:
        with app.state.lock:
            comparison = app.state.comparison_runs.get(run_id)
            if comparison is None:
                raise HTTPException(status_code=404, detail="Comparison not found")
        key = (run_id, request.label, request.method, request.dimension,
               request.min_confidence, request.max_confidence, request.min_short_px)
        with app.state.lock:
            if key in app.state.comparison_projections:
                return app.state.comparison_projections[key]
        try:
            projected = project_comparison(
                comparison,
                label=request.label,
                method=request.method,
                dimension=request.dimension,
                min_confidence=request.min_confidence,
                max_confidence=request.max_confidence,
                min_short_px=request.min_short_px,
            )
        except ComparisonInputError as exc:
            raise HTTPException(status_code=422, detail={
                "code": exc.code,
                "message": str(exc),
                "request_id": uuid.uuid4().hex,
            }) from exc
        with app.state.lock:
            app.state.comparison_projections[key] = projected
        return projected

    @app.get("/api/v1/comparisons/{run_id}/images/{side}/{index}")
    def get_comparison_image(run_id: str, side: str, index: int) -> Response:
        if side not in {"a", "b"}:
            raise HTTPException(status_code=404, detail="Image not found")
        with app.state.lock:
            comparison = app.state.comparison_runs.get(run_id)
            if comparison is None:
                raise HTTPException(status_code=404, detail="Comparison not found")
            paths: list[Path] = comparison[f"_record_paths_{side}"]
            if index < 0 or index >= len(paths):
                raise HTTPException(status_code=404, detail="Image not found")
            path = paths[index]
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise HTTPException(status_code=404, detail="Image is no longer available") from exc
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return Response(content=content, media_type=media_type, headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/runs/{run_id}/images/{index}/thumbnail")
    def get_run_thumbnail(run_id: str, index: int) -> Response:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
            paths: list[Path] = run["image_paths"]
            if index < 0 or index >= len(paths):
                raise HTTPException(status_code=404, detail="Image not found")
            path = paths[index]
        thumbnail = safe_open_image(path)
        if thumbnail is None:
            raise HTTPException(status_code=404, detail="Image is no longer available")
        thumbnail.thumbnail((256, 256))
        output = BytesIO()
        thumbnail.save(output, format="JPEG", quality=84, optimize=True)
        return Response(
            content=output.getvalue(),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/v1/runs/{run_id}/images/{index}/overlay")
    def get_run_image_overlay(run_id: str, index: int) -> Response:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
            if run.get("mode") != "detector" or run.get("unit") != "image":
                raise HTTPException(
                    status_code=409,
                    detail="Annotation overlay requires detector whole-image mode",
                )
            paths: list[Path] = run["image_paths"]
            if index < 0 or index >= len(paths):
                raise HTTPException(status_code=404, detail="Image not found")
            path = paths[index]
            class_names = run.get("class_names") or None
        image = draw_yolo_boxes(path, yolo_label_path_for(path), class_names)
        if image is None:
            raise HTTPException(status_code=404, detail="Image is no longer available")
        output = BytesIO()
        image.save(output, format="PNG")
        return Response(
            content=output.getvalue(),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        with app.state.lock:
            job = app.state.jobs.get(job_id) or app.state.object_policy_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            return _job_view(job)

    @app.delete("/api/v1/jobs/{job_id}", status_code=202)
    def cancel_job(job_id: str) -> dict:
        with app.state.lock:
            job = app.state.jobs.get(job_id) or app.state.object_policy_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            if job.state not in {"queued", "running", "cancelling"}:
                raise HTTPException(status_code=409, detail="Job is already finished")
            job.cancel_event.set()
            job.state, job.message = "cancelling", "Cancellation requested"
            return {"job_id": job_id, "state": job.state}

    @app.get("/api/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
            return {
                key: value for key, value in run.items()
                if key not in {"image_paths", "_raw_embeddings", "_phashes", "_source_sha256"}
            }

    def class_similarity_job_view(job: _ClassSimilarityJob) -> dict:
        percent = int(job.done * 100 / job.total) if job.total else (100 if job.state == "completed" else 0)
        return {
            "job_id": job.job_id,
            "run_id": job.run_id,
            "state": job.state,
            "done": job.done,
            "total": job.total,
            "percent": max(0, min(100, percent)),
            "result": job.result,
            "error": job.error,
            "can_cancel": job.state in {"queued", "running"},
        }

    def execute_class_similarity_job(
        job: _ClassSimilarityJob,
        run: dict,
        raw_embeddings: np.ndarray,
    ) -> None:
        try:
            with app.state.lock:
                job.state = "running"
                job.total = max(1, len(run.get("records", [])))

            unit = str(run.get("unit") or "image")
            mode = str(run.get("mode") or "classifier")
            records = [dict(record) for record in run.get("records", [])]
            image_paths = [Path(path).resolve() for path in run.get("image_paths", [])]
            source_hashes = list(run.get("_source_sha256") or [])
            if len(records) != len(image_paths) or len(records) != len(source_hashes):
                raise ClassSimilarityError("The Visualize source identity is incomplete.")

            generation_hashes = {}
            for index, (record, path, expected_hash) in enumerate(zip(records, image_paths, source_hashes)):
                if not path.is_file() or _sha256_file(path) != expected_hash:
                    raise ClassSimilarityError("A source image changed after this Visualize run.")
                generation_hashes[canonical_class_similarity_path(path)] = str(expected_hash)
                if unit == "object":
                    record["image_path"] = str(path)
                else:
                    record["path"] = str(path)

            context = {
                "level": unit,
                "data_token": f"{job.run_id}:{job.request.model_id}",
                "viz_mode": "Object Detector" if mode == "detector" else "Image Classifier",
                "analysis_split": job.request.analysis_split,
                "k": job.request.k,
                "model": job.request.model_id,
                "model_weights_id": job.request.model_id,
                "embedding_recipe": {
                    "model_id": job.request.model_id,
                    "object_policy": run.get("object_policy"),
                },
                "generation": {
                    "source_hashes": generation_hashes,
                    "recipe": {"run_id": job.run_id, "model_id": job.request.model_id},
                },
            }
            prepared = prepare_class_similarity_records(records, raw_embeddings, context)
            if not prepared.source_verified:
                raise ClassSimilarityError("The Visualize source could not be verified.")

            def update_progress(done: int, total: int) -> None:
                if job.cancel_event.is_set():
                    raise JobCancelled()
                with app.state.lock:
                    job.done = max(0, min(int(done), int(total)))
                    job.total = max(1, int(total))

            analysis = analyze_class_similarity(
                prepared.embeddings,
                prepared.labels,
                prepared.record_ids,
                eligibility=prepared.eligibility,
                invalid_reasons=prepared.invalid_reasons,
                k=job.request.k,
                progress=update_progress,
            )
            if job.cancel_event.is_set():
                raise JobCancelled()

            def public_neighbor(neighbor) -> dict:
                index = int(neighbor.input_index)
                record = prepared.records[index]
                return {
                    "index": int(prepared.original_index[index]),
                    "filename": str(record.get("filename") or ""),
                    "label": record.get("label"),
                    "cosine": float(neighbor.cosine),
                }

            items = []
            for item in analysis.items:
                index = int(item.input_index)
                record = prepared.records[index]
                items.append({
                    "record_id": item.record_id,
                    "index": int(prepared.original_index[index]),
                    "filename": str(record.get("filename") or ""),
                    "label": item.label,
                    "split": str(record.get("split") or ""),
                    "bbox": record.get("bbox"),
                    "obj_index": record.get("obj_index"),
                    "same": item.same,
                    "other": item.other,
                    "margin": item.margin,
                    "rival_class": item.rival_class,
                    "same_k": item.same_k,
                    "rival_k": item.rival_k,
                    "status": item.status,
                    "reasons": item.reasons,
                    "same_neighbors": [public_neighbor(value) for value in item.same_neighbors],
                    "rival_neighbors": [public_neighbor(value) for value in item.rival_neighbors],
                    "tied_rival_count": item.tied_rival_count,
                    "tied_rival_classes": item.tied_rival_classes,
                })

            summaries = {
                metric: build_box_summary(analysis.items, metric)
                for metric in ("margin", "same", "other")
            }
            scatter_items = {
                metric: [{
                    "record_id": item.record_id,
                    "index": int(prepared.original_index[item.input_index]),
                    "label": item.label,
                    "value": float(getattr(item, metric)),
                    "jitter": class_similarity_jitter_x(prepared.analysis_key, item.record_id),
                } for item in sample_class_similarity_scatter_rows(
                    analysis.items, metric, prepared.analysis_key,
                )]
                for metric in ("margin", "same", "other")
            }
            result = {
                "analysis_key": prepared.analysis_key,
                "run_id": job.run_id,
                "model_id": job.request.model_id,
                "unit": unit,
                "k": job.request.k,
                "analysis_split": prepared.analysis_split,
                "source_status": prepared.source_status,
                "summary": {
                    "input_count": analysis.input_count,
                    "valid_count": analysis.valid_count,
                    "excluded_count": analysis.excluded_count,
                    "per_class_counts": analysis.per_class_counts,
                },
                "summaries": summaries,
                "scatter_items": scatter_items,
                "items": items,
            }
            with app.state.lock:
                if app.state.current_run_id != job.run_id:
                    raise ClassSimilarityError("The Visualize run changed during class-similarity analysis.")
                job.prepared = prepared
                job.analysis = analysis
                job.source_hashes = generation_hashes
                job.result = result
                job.done = job.total = len(items)
                job.state = "completed"
        except JobCancelled:
            with app.state.lock:
                job.state = "cancelled"
        except ClassSimilarityError:
            with app.state.lock:
                job.state = "failed"
                job.error = {
                    "code": "CLASS_SIMILARITY_INVALID",
                    "message": "The selected records or source snapshot are invalid. Create a new Visualize run and retry.",
                    "request_id": job.job_id,
                }
        except Exception:
            logger.exception("LV class-similarity analysis failed (job=%s)", job.job_id)
            with app.state.lock:
                job.state = "failed"
                job.error = {
                    "code": "CLASS_SIMILARITY_FAILED",
                    "message": "Class-similarity analysis failed. Check the local worker log and retry.",
                    "request_id": job.job_id,
                }

    @app.post("/api/v1/runs/{run_id}/class-similarity/jobs", status_code=202)
    def create_class_similarity_job(run_id: str, request: CreateClassSimilarityJob) -> dict:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            raw_by_model = app.state.raw_embeddings.get(run_id, {})
            raw = raw_by_model.get(request.model_id)
            if run is None or app.state.current_run_id != run_id or raw is None:
                raise HTTPException(status_code=409, detail={
                    "code": "CLASS_SIMILARITY_RUN_UNAVAILABLE",
                    "message": "Create a current Visualize run with the selected model before calculating class similarity.",
                    "request_id": uuid.uuid4().hex,
                })
            if request.model_id not in set(run.get("models", [])):
                raise HTTPException(status_code=422, detail="The selected model is not part of this Visualize run.")
            if has_active_job():
                raise HTTPException(status_code=409, detail={
                    "code": "JOB_ALREADY_RUNNING",
                    "message": "Wait for the active local analysis to finish before calculating class similarity.",
                    "request_id": uuid.uuid4().hex,
                })
            run_snapshot = copy.deepcopy(run)
            raw_snapshot = np.array(raw, copy=True)
            job_id = uuid.uuid4().hex
            job = _ClassSimilarityJob(job_id=job_id, run_id=run_id, request=request)
            app.state.class_similarity_jobs[job_id] = job
        threading.Thread(
            target=execute_class_similarity_job,
            args=(job, run_snapshot, raw_snapshot),
            daemon=True,
            name=f"lv-class-similarity-{job_id[:8]}",
        ).start()
        return {"job_id": job_id, "state": job.state, "run_id": run_id}

    @app.get("/api/v1/class-similarity/jobs/{job_id}")
    def get_class_similarity_job(job_id: str) -> dict:
        with app.state.lock:
            job = app.state.class_similarity_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Class-similarity job not found")
            return class_similarity_job_view(job)

    @app.delete("/api/v1/class-similarity/jobs/{job_id}", status_code=202)
    def cancel_class_similarity_job(job_id: str) -> dict:
        with app.state.lock:
            job = app.state.class_similarity_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Class-similarity job not found")
            if job.state not in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="Class-similarity job is already finished")
            job.cancel_event.set()
            job.state = "cancelling"
            return {"job_id": job_id, "state": job.state}

    def current_class_similarity_job(job_id: str, *, verify_sources: bool = False) -> _ClassSimilarityJob:
        with app.state.lock:
            job = app.state.class_similarity_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Class-similarity job not found")
            if job.state != "completed" or job.prepared is None or job.analysis is None:
                raise HTTPException(status_code=409, detail="Class-similarity analysis is not ready")
            if app.state.current_run_id != job.run_id:
                raise HTTPException(status_code=409, detail="The Visualize run changed; recalculate class similarity.")
        if verify_sources:
            for source_path, expected_hash in job.source_hashes.items():
                path = Path(source_path)
                try:
                    if not path.is_file() or _sha256_file(path) != expected_hash:
                        raise HTTPException(status_code=409, detail="A Visualize source changed after analysis.")
                except OSError as exc:
                    raise HTTPException(status_code=409, detail="A Visualize source is no longer available.") from exc
        return job

    def class_similarity_evaluation_payload(job: _ClassSimilarityJob, policy: dict) -> dict:
        prepared = job.prepared
        analysis = job.analysis
        parsed = parse_class_similarity_policy(policy)
        mask = class_similarity_candidate_mask(analysis.items, parsed)
        summary = summarize_class_similarity_candidates(analysis.items, mask)
        evaluation = job.evaluation
        counts = evaluate_class_similarity_threshold(
            analysis.items, mask, evaluation, source_verified=prepared.source_verified,
        ) if evaluation else None
        precision_recall = []
        if evaluation and evaluation.get("purpose") != "heldout" and prepared.source_verified:
            for point in class_similarity_precision_recall_curve(
                analysis.items, evaluation, source_verified=prepared.source_verified,
            ):
                point_policy = point.get("policy") or {}
                precision_recall.append({
                    "threshold": point_policy.get("threshold")
                    if point_policy.get("mode") == "threshold" else None,
                    "precision": point["precision"],
                    "recall": point["recall"],
                    "tp": point["tp"], "fp": point["fp"], "fn": point["fn"],
                })
        return {
            "policy": {"mode": parsed.mode, "threshold": parsed.threshold,
                       "comparator": parsed.comparator, "score": parsed.score},
            "summary": {
                "eligible_n": summary.eligible_n,
                "candidate_n": summary.candidate_n,
                "unscorable_n": summary.unscorable_n,
                "ratio": summary.ratio,
                "per_class": summary.per_class,
                "input_count": summary.input_count,
            },
            "evaluation": ({
                "evaluation_id": evaluation.get("evaluation_id"),
                "evaluation_version": evaluation.get("evaluation_version"),
                "purpose": evaluation.get("purpose"),
                "truth_digest": evaluation.get("truth_digest"),
                "frozen_ids": evaluation.get("frozen_ids") or [],
                "pending_n": sum(1 for review in evaluation.get("reviews") or []
                                 if review.get("verdict") == "unreviewed"),
            } if evaluation else None),
            "frozen_policy": job.frozen_policy,
            "counts": ({
                "status": counts.status,
                "unavailable_reason": counts.unavailable_reason,
                "coverage": counts.coverage,
                "e0_n": counts.e0_n,
                "e_n": counts.e_n,
                "known_incorrect_n": counts.known_incorrect_n,
                "unknown_n": counts.unknown_n,
                "unscorable_in_e0": counts.unscorable_in_e0,
                "tp": counts.tp,
                "fp": counts.fp,
                "fn": counts.fn,
                "tn": counts.tn,
                "precision": counts.precision,
                "recall": counts.recall,
                "confusion_ids": counts.confusion_ids,
            } if counts else None),
            "workload_curve": class_similarity_workload_curve(analysis.items),
            "precision_recall_curve": precision_recall,
        }

    def class_similarity_protocol_key(job: _ClassSimilarityJob) -> str:
        prepared = job.prepared
        return class_similarity_score_protocol_key(
            method_version="class_topk_margin_v1", k=prepared.k,
            model=prepared.model, weights_id="unknown",
            recipe=prepared.embedding_recipe, analysis_split=prepared.analysis_split,
        )

    def class_similarity_policy_for_job(job: _ClassSimilarityJob, policy: dict):
        requested = parse_class_similarity_policy(policy)
        if job.evaluation and job.evaluation.get("purpose") == "heldout":
            if not job.frozen_policy:
                raise HTTPException(status_code=409, detail="Heldout policy is unavailable; restore a valid tuning point.")
            frozen = parse_class_similarity_policy(job.frozen_policy.get("policy"))
            if requested != frozen:
                raise HTTPException(status_code=409, detail="Heldout threshold is locked to the saved operating point.")
            return frozen
        return requested

    @app.post("/api/v1/class-similarity/jobs/{job_id}/evaluation/template")
    def download_class_similarity_evaluation_template(
        job_id: str, request: ClassSimilarityEvaluationTemplateRequest,
    ) -> Response:
        job = current_class_similarity_job(job_id, verify_sources=True)
        prepared = job.prepared
        if request.scope == "selected":
            selected = set(request.selected_ids)
            ids = [record_id for record_id in prepared.record_ids
                   if record_id in selected and not record_id.startswith("invalid:")]
            if len(ids) != len(selected):
                raise HTTPException(status_code=422, detail="Selection contains unavailable records.")
            method = "React panel selection (manual sample; not random or population-wide)"
        else:
            ids = [record_id for record_id, eligible in zip(prepared.record_ids, prepared.eligibility)
                   if eligible and not record_id.startswith("invalid:")]
            method = "Full eligible Visualize population"
        payload = class_similarity_evaluation_template(
            frozen_ids=ids, records=prepared.records, record_ids=prepared.record_ids,
            labels=prepared.effective_label, snapshot_key=prepared.source_snapshot_key,
            purpose=request.purpose, selection_method=method,
            evaluation_id=f"eval-{prepared.analysis_key[:8]}-{request.purpose}",
            level=str(job.result.get("unit") or "image"),
        )
        return Response(
            content=payload, media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="class_similarity_evaluation.json"'},
        )

    @app.put("/api/v1/class-similarity/jobs/{job_id}/evaluation")
    def import_class_similarity_evaluation(
        job_id: str, request: ClassSimilarityEvaluationRequest,
    ) -> dict:
        job = current_class_similarity_job(job_id, verify_sources=True)
        prepared = job.prepared
        snapshot = {
            "snapshot_key": prepared.source_snapshot_key,
            "original_labels": {record_id: label for record_id, label in
                                 zip(prepared.record_ids, prepared.effective_label)},
        }
        try:
            evaluation = validate_class_similarity_manifest(
                request.manifest, snapshot, job.analysis.items,
            )
            parse_class_similarity_policy(request.policy)
            level = str(job.result.get("unit") or "image")
            if evaluation.get("purpose") == "heldout":
                frozen = job.frozen_policy
                if not frozen:
                    raise ClassSimilarityError("Heldout review requires a saved operating point.")
                tuning = job.tuning_evaluation
                if not tuning or tuning.get("purpose") != "tuning":
                    raise ClassSimilarityError("Heldout review requires a validated tuning set.")
                frozen_policy = frozen.get("policy") or {}
                tuning_mask = class_similarity_candidate_mask(job.analysis.items, frozen_policy)
                tuning_counts = evaluate_class_similarity_threshold(
                    job.analysis.items, tuning_mask, tuning,
                    source_verified=prepared.source_verified,
                )
                record_by_id = {record_id: prepared.records[index]
                                for index, record_id in enumerate(prepared.record_ids)}
                heldout_ids = list(evaluation.get("frozen_ids") or [])
                heldout_groups = [class_similarity_source_group_id(record_by_id[record_id], level)
                                  for record_id in heldout_ids]
                validate_class_similarity_heldout(
                    frozen, analysis_key=prepared.analysis_key,
                    score_protocol_key=class_similarity_protocol_key(job),
                    heldout_ids=heldout_ids, heldout_groups=heldout_groups,
                    tuning_complete=(tuning_counts.status == "ready"),
                )
                if set(heldout_ids) & job.heldout_used_ids:
                    raise ClassSimilarityError("Heldout IDs were already evaluated in this session.")
                if set(heldout_groups) & job.heldout_used_groups:
                    raise ClassSimilarityError("Heldout source groups were already evaluated in this session.")
                effective_policy = frozen_policy
            else:
                effective_policy = request.policy
        except ClassSimilarityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with app.state.lock:
            job.evaluation = evaluation
            if evaluation.get("purpose") == "heldout":
                ids = list(evaluation.get("frozen_ids") or [])
                groups = [class_similarity_source_group_id(record_by_id[record_id], level)
                          for record_id in ids]
                job.heldout_used_ids.update(ids)
                job.heldout_used_groups.update(groups)
            else:
                job.tuning_evaluation = evaluation
                if job.frozen_policy and job.frozen_policy.get("tuning_evaluation_key") != evaluation.get("truth_digest"):
                    job.frozen_policy = None
                    job.heldout_used_ids.clear()
                    job.heldout_used_groups.clear()
        return class_similarity_evaluation_payload(job, effective_policy)

    @app.post("/api/v1/class-similarity/jobs/{job_id}/operating-point")
    def save_class_similarity_operating_point(
        job_id: str, request: ClassSimilarityOperatingPointRequest,
    ) -> dict:
        job = current_class_similarity_job(job_id, verify_sources=True)
        prepared = job.prepared
        evaluation = job.tuning_evaluation
        if not evaluation or evaluation.get("purpose") != "tuning":
            raise HTTPException(status_code=409, detail="Import a tuning review set before saving an operating point.")
        try:
            policy = parse_class_similarity_policy(request.policy)
            mask = class_similarity_candidate_mask(job.analysis.items, policy)
            counts = evaluate_class_similarity_threshold(
                job.analysis.items, mask, evaluation,
                source_verified=prepared.source_verified,
            )
            if counts.status != "ready":
                raise ClassSimilarityError("The tuning review set must have complete binary verdicts before saving.")
            record_by_id = {record_id: prepared.records[index]
                            for index, record_id in enumerate(prepared.record_ids)}
            tuning_ids = list(evaluation.get("frozen_ids") or [])
            level = str(job.result.get("unit") or "image")
            tuning_groups = [class_similarity_source_group_id(record_by_id[record_id], level)
                             for record_id in tuning_ids]
            preview = class_similarity_preview_key(
                prepared.analysis_key, policy, evaluation.get("truth_digest"),
            )
            frozen = make_class_similarity_operating_point(
                policy=policy, analysis_key=prepared.analysis_key,
                score_protocol_key=class_similarity_protocol_key(job),
                tuning_evaluation_key=evaluation.get("truth_digest"),
                tuning_ids=tuning_ids, tuning_group_ids=tuning_groups,
                policy_id=preview[:12], created_at=datetime.now(timezone.utc).isoformat(),
            )
        except ClassSimilarityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with app.state.lock:
            job.frozen_policy = frozen
            job.heldout_used_ids.clear()
            job.heldout_used_groups.clear()
        return {"frozen_policy": frozen, **class_similarity_evaluation_payload(job, policy.__dict__)}

    @app.post("/api/v1/class-similarity/jobs/{job_id}/evaluation/preview")
    def preview_class_similarity_evaluation(
        job_id: str, request: ClassSimilarityEvaluationPreviewRequest,
    ) -> dict:
        job = current_class_similarity_job(job_id)
        try:
            policy = class_similarity_policy_for_job(job, request.policy)
            return class_similarity_evaluation_payload(job, policy.__dict__)
        except HTTPException:
            raise
        except ClassSimilarityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/class-similarity/jobs/{job_id}/report")
    def download_class_similarity_report(job_id: str, request: ClassSimilarityReportRequest) -> Response:
        job = current_class_similarity_job(job_id, verify_sources=True)
        prepared = job.prepared
        analysis = job.analysis
        try:
            policy = class_similarity_policy_for_job(job, request.policy)
            mask = class_similarity_candidate_mask(analysis.items, policy)
            counts = evaluate_class_similarity_threshold(
                analysis.items, mask, job.evaluation,
                source_verified=prepared.source_verified,
            ) if job.evaluation else None
            preview = class_similarity_preview_key(
                prepared.analysis_key, policy,
                job.evaluation.get("evaluation_key") if job.evaluation else None,
            )
            csv_bytes, json_bytes = export_class_similarity_report(
                analysis, prepared.records, scope=request.scope,
                local_selected_ids=request.selected_ids,
                committed_preview=policy, evaluation=job.evaluation,
                evaluation_counts=counts, prepared=prepared,
                level=str(job.result.get("unit") or "image"),
                analysis_key=prepared.analysis_key, preview_key=preview,
                frozen_policy=job.frozen_policy, label_status=prepared.label_status,
            )
        except ClassSimilarityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        content, media_type, filename = (
            (csv_bytes, "text/csv; charset=utf-8", "class_similarity.csv")
            if request.format == "csv" else
            (json_bytes, "application/json", "class_similarity.json")
        )
        return Response(content=content, media_type=media_type,
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    def public_diagnostic_reference(
        context: dict,
        *,
        current_run_id: str | None = None,
        check_run: bool = False,
    ) -> dict:
        status = "current"
        reason = "The selected record still matches its Visualize source."
        if check_run and current_run_id != context["run_id"]:
            status = "incompatible"
            reason = "Visualize produced a different run; this reference remains historical."
        else:
            source_path = Path(context["_source_path"])
            expected_hash = context.get("source_sha256")
            try:
                actual_hash = _sha256_file(source_path) if source_path.is_file() else None
            except OSError:
                actual_hash = None
            if not expected_hash or actual_hash != expected_hash:
                status = "stale"
                reason = (
                    "The source image is missing or its SHA-256 changed after the Visualize run."
                )
        return {
            key: context[key]
            for key in (
                "context_id", "run_id", "record_index", "filename", "label", "split",
                "unit", "model_id", "object_index", "bbox", "source_sha256",
            )
        } | {"status": status, "reason": reason}

    @app.post("/api/v1/diagnostic-references", status_code=201)
    def create_diagnostic_reference(request: CreateDiagnosticReferenceRequest) -> dict:
        with app.state.lock:
            run = app.state.runs.get(request.run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
            if request.run_id != app.state.current_run_id:
                raise HTTPException(status_code=409, detail={
                    "code": "INCOMPATIBLE_DIAGNOSTIC_RUN",
                    "message": "The selected Visualize run is no longer current.",
                    "request_id": uuid.uuid4().hex,
                })
            records = run.get("records", [])
            paths = run.get("image_paths", [])
            source_hashes = run.get("_source_sha256", [])
            if request.record_index >= len(records) or request.record_index >= len(paths):
                raise HTTPException(status_code=422, detail={
                    "code": "INVALID_RECORD_INDEX",
                    "message": "The selected record is not part of this Visualize run.",
                    "request_id": uuid.uuid4().hex,
                })
            if request.model_id not in run.get("models", []):
                raise HTTPException(status_code=422, detail={
                    "code": "MODEL_NOT_IN_RUN",
                    "message": "The selected model is not part of this Visualize run.",
                    "request_id": uuid.uuid4().hex,
                })
            record = dict(records[request.record_index])
            source_path = Path(paths[request.record_index]).expanduser().resolve()
            source_hash = (
                str(source_hashes[request.record_index])
                if request.record_index < len(source_hashes) else None
            )
        context_id = uuid.uuid4().hex
        context = {
            "context_id": context_id,
            "run_id": request.run_id,
            "record_index": request.record_index,
            "filename": str(record.get("filename", source_path.name)),
            "label": str(record.get("label", "")),
            "split": str(record.get("split", "")),
            "unit": str(run.get("unit", "image")),
            "model_id": request.model_id,
            "object_index": record.get("obj_index"),
            "bbox": record.get("bbox"),
            "source_sha256": source_hash,
            "_source_path": str(source_path),
        }
        with app.state.lock:
            app.state.diagnostic_references[context_id] = context
        return public_diagnostic_reference(context)

    @app.get("/api/v1/diagnostic-references/{context_id}")
    def get_diagnostic_reference(
        context_id: str,
        current_run_id: str | None = None,
    ) -> dict:
        with app.state.lock:
            context = app.state.diagnostic_references.get(context_id)
            actual_run_id = app.state.current_run_id
            if context is None:
                raise HTTPException(status_code=404, detail="Diagnostic reference not found")
        if current_run_id is None:
            checked_run_id = actual_run_id
        elif current_run_id == actual_run_id:
            checked_run_id = actual_run_id
        else:
            checked_run_id = ""
        return public_diagnostic_reference(
            context,
            current_run_id=checked_run_id,
            check_run=True,
        )

    def diagnostic_run_sources_current(run: dict) -> bool:
        paths = run.get("image_paths", [])
        expected_hashes = run.get("_source_sha256", [])
        if not paths or len(paths) != len(expected_hashes):
            return False
        try:
            return all(
                Path(path).is_file() and _sha256_file(Path(path)) == expected
                for path, expected in zip(paths, expected_hashes)
            )
        except OSError:
            return False

    def diagnostic_head_key(run_id: str, run: dict, model_id: str, resolution: int) -> str:
        policy = dict(run.get("object_policy") or {})
        payload = {
            "run_id": run_id,
            "source_sha256": run.get("_source_sha256", []),
            "model_id": model_id,
            "unit": run.get("unit", "image"),
            "resolution": resolution,
            "pooling": str(policy.get("head") or "cls"),
            "object_policy": policy if run.get("unit") == "object" else None,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def diagnostic_head_summary(head_entry: dict, *, reused: bool) -> dict:
        diagnostics = dict(head_entry["head"].get("explain_diagnostics") or {})
        return {
            "status": "ready",
            "resolution": head_entry["resolution"],
            "model_id": head_entry["model_id"],
            "unit": head_entry["unit"],
            "pooling": head_entry["pooling"],
            "classes": [str(value) for value in (head_entry["head"].get("classes") or [])],
            "training_samples": int(head_entry["embeddings"].shape[0]),
            "embedding_dim": int(head_entry["embeddings"].shape[1]),
            "oof_available": bool(diagnostics.get("available")),
            "oof_balanced_accuracy": diagnostics.get("balanced_accuracy"),
            "reused": reused,
        }

    def diagnostic_head_state(
        context_id: str,
        current_run_id: str | None,
        resolution: int,
    ) -> dict:
        with app.state.lock:
            context = app.state.diagnostic_references.get(context_id)
            if context is None:
                raise HTTPException(status_code=404, detail="Diagnostic reference not found")
            run = app.state.runs.get(context["run_id"])
            if run is None:
                return {"status": "stale", "resolution": resolution, "reason": "The Visualize run is no longer available."}
            actual_run_id = app.state.current_run_id
            requested_run_id = actual_run_id if current_run_id is None else current_run_id
            if requested_run_id != actual_run_id or context["run_id"] != actual_run_id:
                return {"status": "incompatible", "resolution": resolution, "reason": "Visualize produced a different run."}
            ref_status = public_diagnostic_reference(context)
            if ref_status["status"] != "current":
                return {"status": "stale", "resolution": resolution, "reason": "The selected source image changed or is unavailable."}
            key = diagnostic_head_key(context["run_id"], run, context["model_id"], resolution)
            head_entry = app.state.diagnostic_heads.get(key)
        if head_entry is None:
            return {"status": "not_ready", "resolution": resolution, "reason": "No matching exact-resolution head has been prepared."}
        return diagnostic_head_summary(head_entry, reused=True)

    @app.get("/api/v1/diagnostic-references/{context_id}/heads/{resolution}")
    def get_diagnostic_head_status(
        context_id: str,
        resolution: str,
        current_run_id: str | None = None,
    ) -> dict:
        try:
            parsed_resolution = int(resolution)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Unsupported classifier head resolution") from exc
        if parsed_resolution not in {224, 336, 448}:
            raise HTTPException(status_code=422, detail="Unsupported classifier head resolution")
        return diagnostic_head_state(context_id, current_run_id, parsed_resolution)

    @app.post("/api/v1/diagnostic-references/{context_id}/heads")
    def build_diagnostic_head(
        context_id: str,
        request: CreateDiagnosticHeadRequest,
    ) -> dict:
        with app.state.lock:
            context = app.state.diagnostic_references.get(context_id)
            if context is None:
                raise HTTPException(status_code=404, detail="Diagnostic reference not found")
            if request.current_run_id != context["run_id"] or app.state.current_run_id != context["run_id"]:
                raise HTTPException(status_code=409, detail={
                    "code": "INCOMPATIBLE_DIAGNOSTIC_RUN",
                    "message": "The diagnostics reference belongs to a different Visualize run.",
                    "request_id": uuid.uuid4().hex,
                })
            run = app.state.runs.get(context["run_id"])
            raw_by_model = app.state.raw_embeddings.get(context["run_id"], {})
            raw_embeddings = raw_by_model.get(context["model_id"])
            if run is None:
                raise HTTPException(status_code=409, detail={
                    "code": "STALE_DIAGNOSTIC_RUN",
                    "message": "The Visualize run is no longer available.",
                    "request_id": uuid.uuid4().hex,
                })
            run_snapshot = dict(run)
            records = [dict(record) for record in run.get("records", [])]
            paths = [Path(item) for item in run.get("image_paths", [])]
            run_id = context["run_id"]
            model_id = context["model_id"]
            resolution = request.resolution
            unit = str(run.get("unit", "image"))
            pooling = str((run.get("object_policy") or {}).get("head") or "cls")
            key = diagnostic_head_key(run_id, run, model_id, resolution)
            existing = app.state.diagnostic_heads.get(key)
        if not diagnostic_run_sources_current(run_snapshot):
            raise HTTPException(status_code=409, detail={
                "code": "STALE_DIAGNOSTIC_SOURCE",
                "message": "A source image changed or is unavailable; create a new Visualize run before preparing a head.",
                "request_id": uuid.uuid4().hex,
            })
        if existing is not None:
            return diagnostic_head_summary(existing, reused=True)
        if len(records) < 2 or len(paths) != len(records) or len({str(item.get("label") or "") for item in records}) < 2:
            raise HTTPException(status_code=422, detail={
                "code": "DIAGNOSTIC_HEAD_DATA_INSUFFICIENT",
                "message": "At least two readable records from two classes are required to prepare a classifier head.",
                "request_id": uuid.uuid4().hex,
            })

        try:
            from dino_head import evaluate_head_oof, train_head

            policy = dict(run_snapshot.get("object_policy") or {})
            keep_aspect = unit == "object"

            def exact_model_loader(selected_model: str, **_kwargs):
                if diagnostic_model_loader is not None:
                    return diagnostic_model_loader(selected_model, resolution, keep_aspect, pooling)
                if unit == "object" and object_model_loader is not None:
                    return object_model_loader(
                        selected_model,
                        keep_aspect=keep_aspect,
                        target_res=resolution,
                        head=pooling,
                    )
                if has_custom_model_loader:
                    if resolution != 224:
                        raise ValueError(
                            "The host has no resolution-aware diagnostics model loader."
                        )
                    return loader(selected_model)
                return load_model(
                    selected_model,
                    keep_aspect=keep_aspect,
                    target_res=resolution,
                    head=pooling,
                )

            if unit == "object":
                from object_eval import embed_objects

                object_records = [
                    {
                        "image_path": paths[index],
                        "bbox": record.get("bbox"),
                        "obj_index": record.get("obj_index", index),
                    }
                    for index, record in enumerate(records)
                ]
                exact_policy = {
                    **policy,
                    "target_res": resolution,
                    "head": pooling,
                }
                embeddings, _short_px = embed_objects(
                    object_records,
                    model_id,
                    exact_policy,
                    cache_path=None,
                    model_loader=exact_model_loader,
                )
            elif resolution == 224 and raw_embeddings is not None:
                embeddings = np.asarray(raw_embeddings, dtype=np.float32)
            else:
                embed_fn = exact_model_loader(
                    model_id,
                    keep_aspect=False,
                    target_res=resolution,
                    head=pooling,
                )
                embeddings = np.stack([
                    np.asarray(embed_fn(path), dtype=np.float32) for path in paths
                ])
            embeddings = np.asarray(embeddings, dtype=np.float32)
            labels = np.asarray([str(record.get("label") or "") for record in records])
            if embeddings.ndim != 2 or embeddings.shape[0] != len(labels) or not np.isfinite(embeddings).all():
                raise ValueError("The selected model returned invalid embeddings.")
            head = train_head(embeddings, labels)
            head["explain_diagnostics"] = evaluate_head_oof(
                embeddings,
                labels,
                source_ids=[
                    f"{run.get('_source_sha256', [])[index]}:{record.get('obj_index', '')}"
                    for index, record in enumerate(records)
                ],
            )
            head["provenance"] = {
                "run_id": run_id,
                "model_id": model_id,
                "unit": unit,
                "pooling": pooling,
                "target_res": resolution,
                "source_fingerprint": key,
                "exact_resolution": True,
                "resolution_mode": "exact_head",
            }
            checkpoint = None
            if model_loader is None and object_model_loader is None and diagnostic_model_loader is None:
                from dino_explain import _resolve_checkpoint

                try:
                    checkpoint = _resolve_checkpoint(model_id, None)
                except Exception:
                    # The head itself can still be retained from already computed
                    # Visualize features. Explanation preflight will refuse Grad-CAM
                    # until the model identity can be established.
                    checkpoint = None
            # An injected host/test embedder does not currently publish a weight
            # identity, so it must not be represented as the default DINO checkpoint.
            from dino_explain import checkpoint_identity
            head["explain_provenance"] = {
                "model_id": model_id,
                "checkpoint_id": checkpoint_identity(checkpoint),
                "embedding_dim": int(embeddings.shape[1]),
                "keep_aspect": keep_aspect,
                "target_res": resolution,
                "pooling": pooling,
                "l2norm": bool(head.get("l2norm", True)),
                "scope": "object_crop" if unit == "object" else "whole_image",
                "pad": float(policy.get("pad", 0.12)) if unit == "object" else None,
            }
            entry = {
                "head": head,
                "embeddings": embeddings,
                "resolution": resolution,
                "model_id": model_id,
                "unit": unit,
                "pooling": pooling,
            }
            if not diagnostic_run_sources_current(run_snapshot):
                raise RuntimeError("A source image changed while the classifier head was being prepared.")
        except Exception as exc:
            logger.exception("LV diagnostics classifier head preparation failed")
            raise HTTPException(status_code=422, detail={
                "code": "DIAGNOSTIC_HEAD_BUILD_FAILED",
                "message": "The exact-resolution classifier head could not be prepared. Check the local model and dataset, then retry.",
                "request_id": uuid.uuid4().hex,
            }) from exc
        with app.state.lock:
            if app.state.current_run_id != run_id:
                raise HTTPException(status_code=409, detail={
                    "code": "INCOMPATIBLE_DIAGNOSTIC_RUN",
                    "message": "Visualize changed while the classifier head was being prepared.",
                    "request_id": uuid.uuid4().hex,
                })
            stored = app.state.diagnostic_heads.setdefault(key, entry)
        return diagnostic_head_summary(stored, reused=stored is not entry)

    @app.post("/api/v1/diagnostic-references/{context_id}/explanations")
    def create_diagnostic_explanation(
        context_id: str,
        request: CreateDiagnosticExplanationRequest,
    ) -> dict:
        with app.state.lock:
            context = app.state.diagnostic_references.get(context_id)
            if context is None:
                raise HTTPException(status_code=404, detail="Diagnostic reference not found")
            run = app.state.runs.get(context["run_id"])
            if run is None:
                raise HTTPException(status_code=409, detail={
                    "code": "STALE_DIAGNOSTIC_RUN",
                    "message": "The Visualize run is no longer available.",
                    "request_id": uuid.uuid4().hex,
                })
            if request.current_run_id != context["run_id"] or app.state.current_run_id != context["run_id"]:
                raise HTTPException(status_code=409, detail={
                    "code": "INCOMPATIBLE_DIAGNOSTIC_RUN",
                    "message": "The diagnostics reference belongs to a different Visualize run.",
                    "request_id": uuid.uuid4().hex,
                })
            if public_diagnostic_reference(context)["status"] != "current":
                raise HTTPException(status_code=409, detail={
                    "code": "STALE_DIAGNOSTIC_SOURCE",
                    "message": "The selected source image changed or is unavailable.",
                    "request_id": uuid.uuid4().hex,
                })
            record_index = int(context["record_index"])
            records = run.get("records", [])
            paths = run.get("image_paths", [])
            if record_index >= len(records) or record_index >= len(paths):
                raise HTTPException(status_code=409, detail={
                    "code": "STALE_DIAGNOSTIC_RUN",
                    "message": "The selected record is no longer part of this Visualize run.",
                    "request_id": uuid.uuid4().hex,
                })
            model_id = str(context["model_id"])
            resolution = int(request.resolution)
            key = diagnostic_head_key(context["run_id"], run, model_id, resolution)
            head_entry = app.state.diagnostic_heads.get(key)
            record = dict(records[record_index])
            source_path = Path(paths[record_index])
            expected_hash = context.get("source_sha256")
            run_snapshot = dict(run)

        if head_entry is None:
            raise HTTPException(status_code=409, detail={
                "code": "DIAGNOSTIC_HEAD_NOT_READY",
                "message": "Prepare the exact-resolution classifier head before running an explanation.",
                "request_id": uuid.uuid4().hex,
            })
        try:
            image = safe_open_image(source_path)
            if image is None:
                raise ValueError("The selected source image cannot be read.")
            original_size = image.size
            unit = str(context.get("unit") or "image")
            policy = dict(run_snapshot.get("object_policy") or {})
            scope = "whole_image"
            crop_bbox_px = None
            pad = None
            input_image = image
            keep_aspect = False
            if unit == "object":
                bbox = context.get("bbox")
                if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                    raise ValueError("The selected object has no valid bounding box.")
                from object_eval import _adaptive_pad_px
                from interaction import bbox_to_pixels, crop_bbox

                scope = "object_crop"
                pad = float(policy.get("pad", 0.12))
                pad_px = _adaptive_pad_px(bbox, *image.size, pad)
                crop_bbox_px = bbox_to_pixels(*bbox, *image.size, pad_px=pad_px)
                input_image = crop_bbox(image, *bbox, pad_px=pad_px)
                keep_aspect = True

            from dino_explain import (
                ExplainRequest,
                _resolve_checkpoint,
                attention_map,
                get_explainer,
                max_norm,
            )

            checkpoint = _resolve_checkpoint(model_id, None)
            explainer, checkpoint = get_explainer(
                model_id, checkpoint, head=str(head_entry["pooling"]),
            )
            source_id = f"{expected_hash}:{record.get('obj_index', '')}"
            explain_request = ExplainRequest(
                image=input_image,
                source_id=source_id,
                scope=scope,
                model_id=model_id,
                checkpoint_path=checkpoint,
                keep_aspect=keep_aspect,
                target_res=resolution,
                pooling=str(head_entry["pooling"]),
                method=request.attention_method,
                head_index=request.head_index,
                original_size=original_size,
                crop_bbox_px=crop_bbox_px,
                pad=pad,
            )
            embeddings = np.asarray(head_entry["embeddings"], dtype=np.float32)
            if record_index >= len(embeddings):
                raise ValueError("The prepared head does not contain the selected record.")
            from dino_head import predict_head

            predicted, confidence, probabilities = predict_head(
                head_entry["head"], embeddings[record_index:record_index + 1],
            )
            classes = [str(value) for value in head_entry["head"].get("classes", [])]
            payload = {
                "status": "ready",
                "run_id": context["run_id"],
                "context_id": context_id,
                "record_index": record_index,
                "resolution": resolution,
                "model_id": model_id,
                "unit": unit,
                "scope": scope,
                "classes": classes,
                "predicted_class": str(predicted[0]),
                "predicted_probability": float(confidence[0]),
                "class_probabilities": [
                    {"class_name": name, "probability": float(probabilities[0][index])}
                    for index, name in enumerate(classes)
                ],
                "warning": "Patch evidence is a coarse model explanation, not a defect mask or pixel-accurate localization.",
            }
            if request.method == "attention":
                result = attention_map(
                    explain_request,
                    explainer=explainer,
                )
                display_scores = np.asarray(result.display_patch_scores, dtype=np.float32)
                payload.update({
                    "method": "attention",
                    "attention_method": result.method,
                    "grid": list(result.grid),
                    "patch_size": int(result.patch_size),
                    "score_mode": "positive",
                    "display_scores": display_scores.tolist(),
                    "raw_scores": np.asarray(result.raw_patch_scores, dtype=np.float32).tolist(),
                    "warnings": list(result.warnings),
                })
            else:
                from gradcam_explain import token_gradcam_map

                result = token_gradcam_map(
                    explain_request,
                    head_entry["head"],
                    target_class=request.target_class,
                    explainer=explainer,
                )
                raw_scores = np.asarray(result.raw_patch_scores, dtype=np.float32)
                if request.display_mode == "positive":
                    display_scores = max_norm(np.clip(raw_scores, 0.0, None))
                else:
                    display_scores = np.asarray(result.display_patch_scores, dtype=np.float32)
                payload.update({
                    "method": "gradcam",
                    "attention_method": None,
                    "grid": list(result.grid),
                    "patch_size": int(result.patch_size),
                    "score_mode": request.display_mode,
                    "display_scores": display_scores.tolist(),
                    "raw_scores": raw_scores.tolist(),
                    "target_class": result.target_class,
                    "target_score": float(result.target_score),
                    "reference_class": result.reference_class,
                    "predicted_class": result.predicted_class,
                    "predicted_probability": float(result.predicted_probability),
                    "runner_up_class": result.runner_up_class,
                    "runner_up_probability": float(result.runner_up_probability),
                    "probability_margin": float(result.probability_margin),
                    "target_probability": float(result.target_probability),
                    "reference_probability": float(result.reference_probability),
                    "is_predicted_target": bool(result.is_predicted_target),
                    "positive_mass": float(result.positive_mass),
                    "negative_mass": float(result.negative_mass),
                    "faithfulness": result.faithfulness,
                    "head_diagnostics": result.head_diagnostics,
                    "sample_oof": result.sample_oof,
                    "warnings": list(result.warnings),
                })
            from explanation_render import export_png_bytes, render_overlay

            signed_view = request.method == "gradcam" and request.display_mode == "signed"
            overlay = render_overlay(
                result.input_image,
                display_scores,
                alpha=0.45,
                cmap="blue_white_red" if request.method == "gradcam" else "viridis",
                signed=signed_view,
            )
            payload.update({
                "source_sha256": expected_hash,
                "model_input_png_base64": base64.b64encode(
                    export_png_bytes(result.input_image)
                ).decode("ascii"),
                "overlay_png_base64": base64.b64encode(
                    export_png_bytes(overlay)
                ).decode("ascii"),
            })
            try:
                source_is_current = (
                    source_path.is_file()
                    and bool(expected_hash)
                    and _sha256_file(source_path) == expected_hash
                )
            except OSError:
                source_is_current = False
            if not source_is_current:
                raise HTTPException(status_code=409, detail={
                    "code": "STALE_DIAGNOSTIC_SOURCE",
                    "message": "The selected source image changed while the explanation was running.",
                    "request_id": uuid.uuid4().hex,
                })
            with app.state.lock:
                if app.state.current_run_id != context["run_id"]:
                    raise HTTPException(status_code=409, detail={
                        "code": "INCOMPATIBLE_DIAGNOSTIC_RUN",
                        "message": "Visualize changed while the explanation was running.",
                        "request_id": uuid.uuid4().hex,
                    })
            return payload
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("LV diagnostics explanation failed")
            code = "DIAGNOSTIC_EXPLANATION_FAILED"
            message = "The explanation could not be generated. Check the exact-resolution head and local DINO model, then retry."
            if isinstance(exc, ValueError) and str(exc) and not any(
                token in str(exc).lower() for token in ("path", "directory", "checkpoint")
            ):
                message = str(exc)
            raise HTTPException(status_code=422, detail={
                "code": code,
                "message": message,
                "request_id": uuid.uuid4().hex,
            }) from exc

    def curation_store_error() -> HTTPException:
        logger.exception("LV curation store operation failed")
        return HTTPException(status_code=500, detail={
            "code": "CURATION_STORE_ERROR",
            "message": "The local curation list could not be read or saved. Existing cache data was preserved.",
            "request_id": uuid.uuid4().hex,
        })

    def public_curation_item(item: dict) -> dict:
        return {key: value for key, value in item.items() if key not in {"path", "image_path"}}

    def curation_snapshot() -> list[dict]:
        try:
            return app.state.curation_store.list_items()
        except CurationStoreError as exc:
            raise curation_store_error() from exc

    @app.get("/api/v1/curation/items")
    def get_curation_items() -> dict:
        items = curation_snapshot()
        return {"items": [public_curation_item(item) for item in items], "total": len(items)}

    def curation_handoff_plan(item_ids: list[str], include_handled: bool) -> dict:
        """Resolve cart rows to readable original images and apply the shared pool guard."""
        import pool_registry as _pool_registry

        selected = curation_snapshot()
        by_id = {str(item.get("id", "")): item for item in selected}
        requested_ids = list(dict.fromkeys(item_ids))
        missing_ids = [item_id for item_id in requested_ids if item_id not in by_id]
        if missing_ids:
            raise HTTPException(status_code=409, detail={
                "code": "CURATION_HANDOFF_ITEMS_CHANGED",
                "message": "The curation list changed. Reload it before sending to Labeling.",
                "request_id": uuid.uuid4().hex,
            })

        records: list[dict] = []
        manifest: dict[str, dict] = {}
        unavailable = 0
        for item_id in requested_ids:
            item = by_id[item_id]
            raw_path = item.get("image_path") if item.get("level") == "object" else item.get("path")
            try:
                path = Path(raw_path or "").expanduser().resolve(strict=True)
                if not path.is_file():
                    raise OSError("source is not a file")
                digest = _sha256_file(path)
            except (OSError, RuntimeError, ValueError):
                unavailable += 1
                continue
            records.append({
                "path": str(path),
                "filename": str(item.get("filename") or path.name),
                "label": str(item.get("label") or ""),
                "split": str(item.get("split") or ""),
            })
            manifest[str(path)] = {"sha256": digest}

        sha_by_index = {
            index: manifest[record["path"]]["sha256"]
            for index, record in enumerate(records)
        }
        unique_shas = list(dict.fromkeys(sha_by_index.values()))
        new_shas, handled = _pool_registry.partition_new(unique_shas)
        new_set = set(new_shas)
        send_indices = [
            index for index, sha in sha_by_index.items()
            if include_handled or sha in new_set
        ]
        send_shas = list(dict.fromkeys(sha_by_index[index] for index in send_indices))

        # Reuse dataset class order when the source contains classes.txt/data.yaml;
        # cart-only or non-classifier images fall back to their recorded labels.
        class_options: list[str] = []
        class_seen: set[str] = set()
        for record in records:
            image_path = Path(record["path"])
            for folder in list(image_path.parents)[:7]:
                names: list[str] = []
                class_file = folder / "classes.txt"
                if class_file.is_file():
                    try:
                        names = [
                            line.strip()
                            for line in class_file.read_text(encoding="utf-8-sig").splitlines()
                            if line.strip()
                        ]
                    except OSError:
                        names = []
                if not names:
                    names = yaml_class_names(folder / "data.yaml")
                if names:
                    for name in names:
                        if name not in class_seen:
                            class_options.append(name)
                            class_seen.add(name)
                    break
        for record in records:
            name = record["label"]
            if name and name not in class_seen:
                class_options.append(name)
                class_seen.add(name)
        if not class_options:
            class_options = ["object"]

        return {
            "records": records,
            "manifest": manifest,
            "sha_by_index": sha_by_index,
            "send_indices": send_indices,
            "send_shas": send_shas,
            "class_options": class_options,
            "selected": len(requested_ids),
            "available": len(records),
            "unavailable": unavailable,
            "unique_images": len(unique_shas),
            "deduplicated": max(0, len(records) - len(unique_shas)),
            "new_images": len(new_shas),
            "handled_images": len(handled),
            "sendable_images": len(send_shas),
            "skipped_handled": 0 if include_handled else len(handled),
        }

    @app.post("/api/v1/curation/labeling-handoff/preview")
    def preview_curation_handoff(request: CurationHandoffRequest) -> dict:
        plan = curation_handoff_plan(request.item_ids, request.include_handled)
        return {key: plan[key] for key in (
            "selected", "available", "unavailable", "unique_images", "deduplicated",
            "new_images", "handled_images", "sendable_images", "skipped_handled",
        )}

    @app.post("/api/v1/curation/labeling-handoff/send", status_code=201)
    def send_curation_handoff(request: SendCurationHandoffRequest) -> dict:
        import labeling_handoff as _labeling_handoff
        import pool_registry as _pool_registry
        import round_ledger as _round_ledger

        with app.state.lock:
            plan = curation_handoff_plan(request.item_ids, request.include_handled)
            rounds = _round_ledger.load_rounds()
            round_ids = {row["round_id"] for row in rounds}
            target_round = request.round_id
            if not request.new_round and target_round is not None and target_round not in round_ids:
                raise HTTPException(status_code=404, detail={
                    "code": "ROUND_NOT_FOUND",
                    "message": "The selected active-learning round no longer exists.",
                    "request_id": uuid.uuid4().hex,
                })
            if not plan["send_indices"]:
                raise HTTPException(status_code=422, detail={
                    "code": "CURATION_HANDOFF_EMPTY",
                    "message": "There are no available new images to send. Enable the explicit override to include previously handled samples.",
                    "request_id": uuid.uuid4().hex,
                })
            try:
                handoff_dir = _labeling_handoff.send_to_labeling(
                    plan["records"],
                    plan["send_indices"],
                    source="cart",
                    task=_labeling_handoff.TASK_RELABEL,
                    class_options=plan["class_options"],
                    manifest=plan["manifest"],
                    original_labels={
                        index: record["label"]
                        for index, record in enumerate(plan["records"])
                    },
                    payload={"curation_item_ids": list(dict.fromkeys(request.item_ids))},
                )
                if handoff_dir is None:
                    raise ValueError("no images were materialized")
                spec = _labeling_handoff.load_spec(handoff_dir)
                sent_shas = [item["sha256"] for item in spec.get("items", [])]
                batch_id = str(spec.get("handoff_id") or Path(handoff_dir).name)
                if not sent_shas:
                    raise ValueError("the handoff contains no readable images")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.exception("LV curation handoff could not be materialized")
                raise HTTPException(status_code=422, detail={
                    "code": "CURATION_HANDOFF_FAILED",
                    "message": "The selected images could not be prepared for Labeling. Check that the source files are still readable.",
                    "request_id": uuid.uuid4().hex,
                }) from exc

            _pool_registry.mark_sent(sent_shas, batch_id, tool="cart")
            created_round = request.new_round or not rounds
            if created_round:
                round_id = _round_ledger.start_round("cart", {}, "cart")
            else:
                round_id = target_round or rounds[-1]["round_id"]
            _round_ledger.attach_batch(round_id, batch_id, len(sent_shas))
            return {
                "handoff_id": batch_id,
                "task": str(spec.get("task") or _labeling_handoff.TASK_RELABEL),
                "sent": len(sent_shas),
                "deduplicated": max(0, len(plan["send_indices"]) - len(sent_shas)),
                "skipped_handled": plan["skipped_handled"],
                "round_id": round_id,
                "created_round": created_round,
                "saved_to_local_inbox": True,
            }

    @app.get("/api/v1/rounds")
    def get_active_learning_rounds() -> dict:
        import round_ledger as _round_ledger

        return {
            "rounds": _round_ledger.load_rounds(),
            "orphan_events": _round_ledger.orphan_events(),
        }

    @app.post("/api/v1/rounds/{round_id}/metrics")
    def add_active_learning_metric(round_id: str, request: AttachRoundMetricRequest) -> dict:
        import round_ledger as _round_ledger

        with app.state.lock:
            rounds = _round_ledger.load_rounds()
            if not any(row["round_id"] == round_id for row in rounds):
                raise HTTPException(status_code=404, detail={
                    "code": "ROUND_NOT_FOUND",
                    "message": "The selected active-learning round no longer exists.",
                    "request_id": uuid.uuid4().hex,
                })
            _round_ledger.attach_metric(
                round_id,
                request.name.strip(),
                request.value,
                request.source,
                request.note,
            )
            return {"rounds": _round_ledger.load_rounds()}

    @app.post("/api/v1/rounds/{round_id}/evaluations")
    def attach_detection_evaluation_to_round(
        round_id: str, request: AttachEvaluationToRoundRequest,
    ) -> dict:
        import round_ledger as _round_ledger

        with app.state.lock:
            rounds = _round_ledger.load_rounds()
            target_round = next((row for row in rounds if row["round_id"] == round_id), None)
            if target_round is None:
                raise HTTPException(status_code=404, detail={
                    "code": "ROUND_NOT_FOUND",
                    "message": "The selected active-learning round no longer exists.",
                    "request_id": uuid.uuid4().hex,
                })
            evaluation = app.state.evaluation_runs.get(request.evaluation_id)
            if evaluation is None:
                raise HTTPException(status_code=404, detail={
                    "code": "EVALUATION_NOT_FOUND",
                    "message": "This evaluation is no longer available. Run it again.",
                    "request_id": uuid.uuid4().hex,
                })
            if evaluation.get("demo"):
                raise HTTPException(status_code=422, detail={
                    "code": "SAMPLE_EVALUATION_NOT_ATTACHABLE",
                    "message": "Sample evaluations cannot be added to an active-learning round.",
                    "request_id": uuid.uuid4().hex,
                })
            prior_round = next((
                row for row in rounds
                if any(saved.get("evaluation_id") == request.evaluation_id
                       for saved in row.get("evaluations", []))
            ), None)
            if prior_round is not None and prior_round["round_id"] != round_id:
                raise HTTPException(status_code=409, detail={
                    "code": "EVALUATION_ALREADY_ATTACHED",
                    "message": "This evaluation is already attached to another round.",
                    "request_id": uuid.uuid4().hex,
                })
            attached = prior_round is None
            if attached:
                summary = evaluation.get("summary", {})
                safe_summary = {
                    key: summary.get(key)
                    for key in (
                        "overall", "per_class", "escape_count", "confusion",
                        "gray_gt_count", "consensus_gt_count", "image_count",
                        "prediction_image_count", "used_consensus", "warnings",
                    )
                }
                _round_ledger.attach_evaluation(round_id, {
                    "evaluation_id": request.evaluation_id,
                    "parameters": evaluation.get("parameters", {}),
                    "summary": safe_summary,
                })
            return {
                "round_id": round_id,
                "attached": attached,
                "rounds": _round_ledger.load_rounds(),
            }

    @app.post("/api/v1/rounds/{round_id}/probe")
    def run_active_learning_probe(round_id: str, request: RunRoundProbeRequest) -> dict:
        import probe_eval as _probe_eval
        import round_ledger as _round_ledger

        with app.state.lock:
            if not any(row["round_id"] == round_id for row in _round_ledger.load_rounds()):
                raise HTTPException(status_code=404, detail={
                    "code": "ROUND_NOT_FOUND",
                    "message": "The selected active-learning round no longer exists.",
                    "request_id": uuid.uuid4().hex,
                })
            run = app.state.runs.get(request.run_id)
            if run is None or request.run_id != app.state.current_run_id:
                raise HTTPException(status_code=409, detail={
                    "code": "PROBE_RUN_NOT_CURRENT",
                    "message": "Load a current Visualize run before running a probe evaluation.",
                    "request_id": uuid.uuid4().hex,
                })
            if run.get("mode") != "classifier" or run.get("unit") != "image":
                raise HTTPException(status_code=422, detail={
                    "code": "PROBE_REQUIRES_IMAGE_CLASSIFIER",
                    "message": "Probe evaluation currently requires Image Classifier at whole-image level.",
                    "request_id": uuid.uuid4().hex,
                })
            model_id = str(run.get("model_id") or "")
            model_vectors = app.state.raw_embeddings.get(request.run_id, {}).get(model_id)
            records = run.get("records", [])
            if model_vectors is None or len(model_vectors) != len(records):
                raise HTTPException(status_code=409, detail={
                    "code": "PROBE_EMBEDDINGS_UNAVAILABLE",
                    "message": "Frozen embeddings for the current run are unavailable. Run Visualize again before evaluating.",
                    "request_id": uuid.uuid4().hex,
                })
            keep = [
                index for index, record in enumerate(records)
                if (str(record.get("label") or "").strip() not in ("", "unknown"))
            ]
            labels = [str(records[index]["label"]) for index in keep]
            try:
                result = _probe_eval.train_probe(np.asarray(model_vectors)[keep], labels)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail={
                    "code": "PROBE_INSUFFICIENT_SAMPLES",
                    "message": f"Probe unavailable: {exc}",
                    "request_id": uuid.uuid4().hex,
                }) from exc
            _round_ledger.attach_metric(round_id, "probe_acc", round(result["acc"], 4), "probe")
            _round_ledger.attach_metric(round_id, "probe_macro_f1", round(result["macro_f1"], 4), "probe")
            return {"round_id": round_id, "model_id": model_id, **result}

    @app.get("/api/v1/rounds/learning-curve/advice")
    def get_active_learning_curve_advice(metric_name: str) -> dict:
        import probe_eval as _probe_eval
        import round_ledger as _round_ledger

        name = metric_name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="A metric name is required")
        xs, ys = _round_ledger.learning_curve(_round_ledger.load_rounds(), name)
        return {
            "metric_name": name,
            "xs": xs,
            "ys": ys,
            "advice": _probe_eval.plateau_advice(xs, ys),
        }

    @app.get("/api/v1/labeling/handoffs")
    def get_labeling_handoffs() -> dict:
        import labeling_handoff as _labeling_handoff

        rows = _labeling_handoff.list_pending()
        return {"items": [
            {
                key: row.get(key)
                for key in ("handoff_id", "source", "task", "created_at", "status", "n_total", "n_annotated")
            }
            for row in rows
        ]}

    @app.post("/api/v1/labeling/handoffs/{handoff_id}/readback")
    def apply_labeling_readback(handoff_id: str, request: ApplyLabelingReadbackRequest) -> dict:
        import labeling_handoff as _labeling_handoff
        import pool_registry as _pool_registry
        import readback_store as _readback_store
        import round_ledger as _round_ledger
        from collections import Counter

        with app.state.lock:
            run = app.state.runs.get(request.run_id)
            if run is None or request.run_id != app.state.current_run_id:
                raise HTTPException(status_code=409, detail={
                    "code": "READBACK_RUN_NOT_CURRENT",
                    "message": "Load the dataset in Visualize again before applying Labeling results.",
                    "request_id": uuid.uuid4().hex,
                })
            if run.get("mode") != "classifier" or run.get("unit") != "image":
                raise HTTPException(status_code=422, detail={
                    "code": "READBACK_REQUIRES_IMAGE_CLASSIFIER",
                    "message": "Read-back into the current run currently requires Image Classifier at whole-image level.",
                    "request_id": uuid.uuid4().hex,
                })
            row = next((item for item in _labeling_handoff.list_pending()
                        if item.get("handoff_id") == handoff_id), None)
            if row is None:
                raise HTTPException(status_code=404, detail={
                    "code": "LABELING_HANDOFF_NOT_FOUND",
                    "message": "The selected Labeling handoff could not be found.",
                    "request_id": uuid.uuid4().hex,
                })
            if row.get("status") == _labeling_handoff.STATUS_READ:
                raise HTTPException(status_code=409, detail={
                    "code": "LABELING_READBACK_ALREADY_APPLIED",
                    "message": "This Labeling handoff has already been fully applied.",
                    "request_id": uuid.uuid4().hex,
                })
            handoff_root = _labeling_handoff.handoff_root().resolve(strict=False)
            try:
                handoff_dir = Path(row["dir"]).resolve(strict=True)
            except (KeyError, OSError, RuntimeError) as exc:
                raise HTTPException(status_code=404, detail={
                    "code": "LABELING_HANDOFF_UNAVAILABLE",
                    "message": "The selected Labeling handoff folder is unavailable.",
                    "request_id": uuid.uuid4().hex,
                }) from exc
            if not handoff_dir.is_relative_to(handoff_root):
                raise HTTPException(status_code=404, detail={
                    "code": "LABELING_HANDOFF_NOT_FOUND",
                    "message": "The selected Labeling handoff could not be found.",
                    "request_id": uuid.uuid4().hex,
                })

            paths = run.get("image_paths", [])
            hashes = run.get("_source_sha256", [])
            public_records = run.get("records", [])
            if len(paths) != len(public_records) or len(hashes) != len(public_records):
                raise HTTPException(status_code=409, detail={
                    "code": "READBACK_RUN_IDENTITY_UNAVAILABLE",
                    "message": "The active run no longer has complete source-image identity data. Reload the dataset before applying read-back.",
                    "request_id": uuid.uuid4().hex,
                })

            readback_records = [
                {**record, "path": str(paths[index]), "sha256": hashes[index]}
                for index, record in enumerate(public_records)
            ]
            try:
                results = _labeling_handoff.read_labeling_results(handoff_dir)
                changes = _labeling_handoff.apply_readback(handoff_dir, readback_records)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.exception("LV could not read Labeling results for handoff %s", handoff_id)
                raise HTTPException(status_code=422, detail={
                    "code": "LABELING_READBACK_INVALID",
                    "message": "The handoff or its annotation sidecars could not be read. Check the Labeling export and try again.",
                    "request_id": uuid.uuid4().hex,
                }) from exc

            labeled_shas = [sha for sha, result in results.items() if result.get("label")]
            if not labeled_shas:
                raise HTTPException(status_code=409, detail={
                    "code": "LABELING_READBACK_NOT_READY",
                    "message": "No completed labels were found in this handoff yet.",
                    "request_id": uuid.uuid4().hex,
                })
            if labeled_shas:
                _pool_registry.mark_labeled(labeled_shas, handoff_id)
            summary = {
                "batch_id": handoff_id,
                "n_labeled": len(labeled_shas),
                "n_changed": len(changes),
                "changed_by_class": dict(Counter(change["new_label"] for change in changes)),
            }
            matching_round = next((round_row["round_id"] for round_row in _round_ledger.load_rounds()
                                   if any(batch.get("batch_id") == handoff_id
                                          for batch in round_row.get("batches", []))), None)
            if matching_round:
                _round_ledger.record_readback(matching_round, summary)

            annotator = next((result.get("annotator") for result in results.values()
                              if result.get("annotator")), "")
            persisted_changes = []
            for change in changes:
                index = change.get("lv_index")
                if isinstance(index, int) and 0 <= index < len(hashes):
                    persisted_changes.append({**change, "sha256": hashes[index]})
            _readback_store.record_corrections(
                persisted_changes,
                batch_id=handoff_id,
                round_id=matching_round or "",
                annotator=annotator,
            )

            changes_by_index = {change["lv_index"]: change for change in changes}
            for index, change in changes_by_index.items():
                public_records[index]["label"] = change["new_label"]
            for point in run.get("points", []):
                change = changes_by_index.get(point.get("index"))
                if change:
                    point["label"] = change["new_label"]
            for view in run.get("views", []):
                for point in view.get("points", []):
                    change = changes_by_index.get(point.get("index"))
                    if change:
                        point["label"] = change["new_label"]
            complete = len(labeled_shas) >= int(row.get("n_total") or 0)
            if complete:
                _labeling_handoff.update_pending(
                    handoff_id,
                    status=_labeling_handoff.STATUS_READ,
                    read_back_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
            return {
                "handoff_id": handoff_id,
                "run_id": request.run_id,
                "round_id": matching_round,
                "n_labeled": len(labeled_shas),
                "n_changed": len(changes),
                "complete": complete,
                "annotator": annotator,
                "orphan_round": matching_round is None,
                "changes": [
                    {key: change.get(key) for key in ("filename", "old_label", "new_label")}
                    for change in changes
                ],
            }

    @app.get("/api/v1/readback/audit")
    def get_readback_audit() -> dict:
        import readback_store as _readback_store

        return {
            "summary": _readback_store.corrections_summary(),
            "events": [
                {
                    key: row.get(key, "")
                    for key in ("ts", "sha256", "filename", "old_label", "new_label", "batch_id", "round_id", "annotator")
                }
                for row in _readback_store.load_changelog()
            ],
        }

    @app.get("/api/v1/readback/audit.csv")
    def export_readback_audit() -> Response:
        import readback_store as _readback_store

        buffer = StringIO()
        fields = ["sha256", "filename", "old_label", "new_label", "annotator", "ts"]
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        for row in _readback_store.load_changelog():
            writer.writerow({key: row.get(key, "") for key in fields})
        return Response(
            content="\ufeff" + buffer.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="labeling_readback_corrections.csv"'},
        )

    def build_curation_subset_plan(source: str) -> tuple[list, list[Path], dict, str]:
        """Translate the saved cart into the framework-free subset contract."""
        import export_subset as _export_subset

        selected = curation_snapshot()
        if source != "all":
            selected = [
                item for item in selected
                if (item.get("source") or "manual") == source
            ]
        if not selected:
            raise HTTPException(status_code=422, detail={
                "code": "CURATION_SUBSET_EMPTY",
                "message": "There are no curation items in the selected source.",
                "request_id": uuid.uuid4().hex,
            })

        export_items = []
        fingerprints = []
        available_hashes: set[str] = set()
        missing_files: list[str] = []
        label_files: set[str] = set()
        source_roots: set[str] = set()
        available = 0

        def source_root_for(image_path: Path, split: str) -> Path:
            ancestors = list(image_path.parents)
            for folder in ancestors:
                if folder.name.casefold() == "images":
                    return folder.parent
            if split:
                for folder in ancestors:
                    if folder.name.casefold() == split.casefold():
                        return folder.parent
            return image_path.parent

        def class_names_for(image_path: Path) -> list[str]:
            # Follow dataset ancestors instead of importing Streamlit's app.py
            # helper; classes.txt and all supported data.yaml names forms use
            # the shared framework-free parser.
            for folder in list(image_path.parents)[:7]:
                classes_file = folder / "classes.txt"
                if classes_file.is_file():
                    try:
                        names = [
                            line.strip()
                            for line in classes_file.read_text(
                                encoding="utf-8-sig",
                            ).splitlines()
                            if line.strip()
                        ]
                    except OSError:
                        names = []
                    if names:
                        return names
                names = yaml_class_names(folder / "data.yaml")
                if names:
                    return names
            return []

        for item in selected:
            is_object = item.get("level") == "object" and item.get("image_path")
            image_path = Path(
                item["image_path"] if is_object else item["path"],
            ).expanduser()
            image_path = image_path.resolve(strict=False)
            source_root = source_root_for(
                image_path, str(item.get("split") or ""),
            ).resolve(strict=False)
            source_roots.add(str(source_root))
            label_path = yolo_label_path_for(image_path)
            names = class_names_for(image_path)
            image_sha = None
            image_state = "missing"
            try:
                if image_path.is_file():
                    image_sha = _sha256_file(image_path)
                    image_state = "available"
                    available += 1
                    available_hashes.add(image_sha)
                else:
                    missing_files.append(image_path.name)
            except OSError:
                image_state = "unreadable"
                missing_files.append(image_path.name)

            label_sha = None
            label_state = "missing"
            if label_path.is_file():
                try:
                    label_sha = _sha256_file(label_path)
                    label_state = "available"
                    label_files.add(str(label_path.resolve(strict=False)))
                except OSError:
                    label_state = "unreadable"

            object_index = item.get("obj_index") if is_object else None
            object_ids = (
                [int(object_index)]
                if isinstance(object_index, int) and object_index >= 0 else None
            )
            export_items.append(_export_subset.ExportItem(
                image_path=image_path,
                label_path=label_path if label_path.is_file() else None,
                class_names=names,
                split=(str(item.get("split") or "") or None),
                source_tool=str(item.get("source") or "manual"),
                source_tag=str(item.get("source") or "manual"),
                reason=str(item.get("reason") or ""),
                level="object" if is_object else "image",
                object_ids=object_ids,
                # Re-hash at export time so a stale cart digest cannot mask a
                # changed or missing source image.
                sha256=None,
            ))
            fingerprints.append({
                "id": item.get("id"),
                "path": str(image_path),
                "image_state": image_state,
                "image_sha256": image_sha,
                "label_path": str(label_path.resolve(strict=False)),
                "label_state": label_state,
                "label_sha256": label_sha,
                "class_names": names,
                "source": item.get("source", "manual"),
                "reason": item.get("reason", ""),
                "split": item.get("split", ""),
                "level": item.get("level", "image"),
                "obj_index": object_index,
            })

        summary = {
            "selected": len(selected),
            "available": available,
            "unique_images": len(available_hashes),
            "deduplicated": max(0, available - len(available_hashes)),
            "missing": len(missing_files),
            "missing_filenames": missing_files[:50],
            "label_files": len(label_files),
            "object_items": sum(
                1 for item in selected if item.get("level") == "object"
            ),
        }
        fingerprint = json.dumps(
            fingerprints,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return (
            export_items,
            [Path(root) for root in sorted(source_roots)],
            summary,
            fingerprint,
        )

    def prepare_curation_subset(
        source: str, destination: str, mode: str, on_exists: str,
    ):
        import export_subset as _export_subset

        export_items, source_roots, summary, fingerprint = build_curation_subset_plan(
            source,
        )
        try:
            resolved_destination = _export_subset.validate_export_destination(
                export_items, Path(destination), source_roots=source_roots,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={
                "code": "CURATION_SUBSET_DESTINATION_OVERLAP",
                "message": (
                    "The destination overlaps a source dataset. "
                    "Choose a separate output folder."
                ),
                "request_id": uuid.uuid4().hex,
            }) from exc
        token_input = json.dumps(
            {
                "fingerprint": fingerprint,
                "destination": str(resolved_destination),
                "mode": mode,
                "on_exists": on_exists,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        token = hashlib.sha256(token_input.encode("utf-8")).hexdigest()
        return export_items, source_roots, summary, token, resolved_destination

    def curation_subset_preview(
        source: str, destination: str, mode: str, on_exists: str,
    ) -> dict:
        _items, _roots, summary, token, _destination = prepare_curation_subset(
            source, destination, mode, on_exists,
        )
        return {**summary, "preview_token": token, "destination_safe": True}

    @app.post("/api/v1/curation/subset/preview")
    def preview_curation_subset(request: CurationSubsetPreviewRequest) -> dict:
        return curation_subset_preview(
            request.source, request.destination, request.mode, request.on_exists,
        )

    @app.post("/api/v1/curation/subset/export")
    def export_curation_subset(request: CurationSubsetExportRequest) -> dict:
        import export_subset as _export_subset

        (
            export_items,
            source_roots,
            _summary,
            current_token,
            destination,
        ) = prepare_curation_subset(
            request.source,
            request.destination,
            request.mode,
            request.on_exists,
        )
        if current_token != request.preview_token:
            raise HTTPException(status_code=409, detail={
                "code": "CURATION_SUBSET_PREVIEW_STALE",
                "message": (
                    "The curation items or source files changed after preview. "
                    "Preview again before exporting."
                ),
                "request_id": uuid.uuid4().hex,
            })
        try:
            report = _export_subset.export_subset(
                export_items,
                destination,
                mode=request.mode,
                layout="yolo",
                source_roots=source_roots,
                on_exists=request.on_exists,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={
                "code": "CURATION_SUBSET_EXPORT_REJECTED",
                "message": (
                    "The export destination or options are no longer valid. "
                    "Preview again and choose a separate output folder."
                ),
                "request_id": uuid.uuid4().hex,
            }) from exc
        return {
            "exported": report.exported,
            "skipped": len(report.skipped),
            "deduplicated": report.deduped,
            "unresolved": len(report.unresolved),
            "corrupt": len(report.corrupt),
            "warnings": len(report.warnings),
            "errors": len(report.errors),
            "annotations_preserved": report.annotations_preserved,
            "partial": bool(report.unresolved or report.corrupt or report.errors),
            "training_warning": "This is a deliberately skewed diagnostic subset; evaluate with an independent hold-out set.",
        }

    @app.post("/api/v1/coverage/runs/{run_id}/candidates/curation", status_code=201)
    def add_coverage_candidates_to_curation(
        run_id: str,
        request: AddCoverageCandidateCurationRequest,
    ) -> dict:
        with app.state.lock:
            candidate_set = app.state.coverage_candidates.get(run_id)
            if candidate_set is None:
                raise HTTPException(status_code=404, detail="Coverage candidates not found")
            records = candidate_set["records"]
            paths: list[Path] = candidate_set["image_paths"]
            if any(index < 0 or index >= len(records) or index >= len(paths) for index in request.indices):
                raise HTTPException(status_code=422, detail={
                    "code": "INVALID_RECORD_INDEX",
                    "message": "A selected candidate is not part of this coverage run.",
                    "request_id": uuid.uuid4().hex,
                })
            entries = []
            for index in dict.fromkeys(request.indices):
                record = records[index]
                try:
                    path = Path(paths[index]).expanduser().resolve(strict=True)
                    if not path.is_file():
                        raise OSError("source is not a file")
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                except OSError as exc:
                    raise HTTPException(status_code=404, detail={
                        "code": "CURATION_IMAGE_UNAVAILABLE",
                        "message": "A selected candidate image is no longer available.",
                        "request_id": uuid.uuid4().hex,
                    }) from exc
                entry = {
                    "index": index,
                    "filename": str(record.get("filename", path.name)),
                    "path": str(path),
                    "label": str(record.get("label", "")),
                    "split": str(record.get("split", "")),
                    "sha256": digest.hexdigest(),
                    "source": request.source,
                    "score": request.scores.get(index),
                    "reason": request.reason.strip(),
                    "level": candidate_set["unit"],
                }
                if candidate_set["unit"] == "object":
                    entry.update({
                        "image_path": str(path),
                        "bbox": record.get("bbox"),
                        "obj_index": record.get("obj_index"),
                    })
                entries.append(entry)
            try:
                added, skipped = app.state.curation_store.add(entries)
                items = app.state.curation_store.list_items()
            except CurationStoreError as exc:
                raise curation_store_error() from exc
        return {
            "added": added,
            "skipped": skipped,
            "items": [public_curation_item(item) for item in items],
            "total": len(items),
        }

    @app.post("/api/v1/runs/{run_id}/curation/items", status_code=201)
    def add_run_records_to_curation(run_id: str, request: AddCurationRequest) -> dict:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
            records = run.get("records", [])
            paths: list[Path] = run.get("image_paths", [])
            if any(index < 0 or index >= len(records) or index >= len(paths) for index in request.indices):
                raise HTTPException(status_code=422, detail={
                    "code": "INVALID_RECORD_INDEX",
                    "message": "A selected record is not part of this run.",
                    "request_id": uuid.uuid4().hex,
                })
            selected_indices = set(request.indices)
            if any(index not in selected_indices for index in request.scores):
                raise HTTPException(status_code=422, detail={
                    "code": "CURATION_SCORE_INDEX_NOT_SELECTED",
                    "message": "A curation score must refer to a selected record.",
                    "request_id": uuid.uuid4().hex,
                })
            if any(not math.isfinite(score) for score in request.scores.values()):
                raise HTTPException(status_code=422, detail={
                    "code": "CURATION_SCORE_NOT_FINITE",
                    "message": "Curation scores must be finite numbers.",
                    "request_id": uuid.uuid4().hex,
                })
            entries = []
            for index in dict.fromkeys(request.indices):
                record = records[index]
                try:
                    path = Path(paths[index]).expanduser().resolve(strict=True)
                    if not path.is_file():
                        raise OSError("source is not a file")
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                except OSError as exc:
                    raise HTTPException(status_code=404, detail={
                        "code": "CURATION_IMAGE_UNAVAILABLE",
                        "message": "A selected image is no longer available.",
                        "request_id": uuid.uuid4().hex,
                    }) from exc
                entry = {
                    "index": index,
                    "filename": str(record.get("filename", path.name)),
                    "path": str(path),
                    "label": str(record.get("label", "")),
                    "split": str(record.get("split", "")),
                    "sha256": digest.hexdigest(),
                    "source": request.source,
                    "score": request.scores.get(index),
                    "reason": request.reason.strip(),
                    "level": run.get("unit", "image"),
                }
                if run.get("unit") == "object":
                    entry["image_path"] = str(path)
                    entry["bbox"] = record.get("bbox")
                    entry["obj_index"] = record.get("obj_index")
                entries.append(entry)
            try:
                added, skipped = app.state.curation_store.add(entries)
                items = app.state.curation_store.list_items()
            except CurationStoreError as exc:
                raise curation_store_error() from exc
        return {
            "added": added,
            "skipped": skipped,
            "items": [public_curation_item(item) for item in items],
            "total": len(items),
        }

    @app.patch("/api/v1/curation/items/reason")
    def set_curation_reason(request: SetCurationReasonRequest) -> dict:
        reason = request.reason.strip()
        if not reason:
            raise HTTPException(status_code=422, detail="A non-empty reason is required")
        try:
            updated = app.state.curation_store.set_reason(request.item_ids, reason)
            items = app.state.curation_store.list_items()
        except CurationStoreError as exc:
            raise curation_store_error() from exc
        return {"updated": updated, "items": [public_curation_item(item) for item in items], "total": len(items)}

    @app.delete("/api/v1/curation/items/{item_id}")
    def remove_curation_item(item_id: str) -> dict:
        if len(item_id) != 64:
            raise HTTPException(status_code=404, detail="Curation item not found")
        try:
            removed = app.state.curation_store.remove(item_id)
            items = app.state.curation_store.list_items()
        except CurationStoreError as exc:
            raise curation_store_error() from exc
        if not removed:
            raise HTTPException(status_code=404, detail="Curation item not found")
        return {"removed": True, "items": [public_curation_item(item) for item in items], "total": len(items)}

    @app.delete("/api/v1/curation/items")
    def clear_curation_items() -> dict:
        try:
            cleared = app.state.curation_store.clear()
        except CurationStoreError as exc:
            raise curation_store_error() from exc
        return {"cleared": cleared, "items": [], "total": 0}

    @app.get("/api/v1/curation/items/{item_id}/thumbnail")
    def get_curation_thumbnail(item_id: str) -> Response:
        item = next((row for row in curation_snapshot() if row.get("id") == item_id), None)
        if item is None:
            raise HTTPException(status_code=404, detail="Curation item not found")
        path = Path(item["path"])
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise HTTPException(status_code=404, detail="Image is no longer available") from exc
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return Response(content=content, media_type=media_type, headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/curation.csv")
    def export_curation_csv() -> Response:
        items = curation_snapshot()
        content = snapshots_to_csv(items).encode("utf-8")
        return Response(content=content, media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="curation_cart.csv"'})

    @app.get("/api/v1/curation.json")
    def export_curation_json() -> Response:
        items = curation_snapshot()
        content = json.dumps(items, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        return Response(content=content, media_type="application/json; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="curation_cart.json"'})

    @app.get("/api/v1/curation.zip")
    def export_curation_zip() -> Response:
        items = curation_snapshot()
        records = [
            {"path": Path(item["path"]), "label": item.get("label", ""), "split": item.get("split", "")}
            for item in items
        ]
        content = zip_selected_images(records, list(range(len(records))))
        return Response(content=content, media_type="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="curation_cart_images.zip"'})

    def ranked_rows(
        run_id: str,
        model_id: str,
        metric: Literal["outlier", "disagreement"],
        label_filter: str | None = None,
        split_filter: str | None = None,
    ) -> tuple[dict, int, list[dict]]:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            by_model = app.state.ranking_scores.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if model_id not in run.get("models", []):
            raise HTTPException(status_code=404, detail="Model is not part of this run")
        scores_by_metric = (by_model or {}).get(model_id)
        if scores_by_metric is None:
            raise HTTPException(
                status_code=409,
                detail="Ranking signals require at least three records in this run",
            )
        scores = np.asarray(scores_by_metric.get(metric), dtype=np.float64)
        records = run.get("records", [])
        if scores.ndim != 1 or len(scores) != len(records) or not np.isfinite(scores).all():
            raise HTTPException(status_code=409, detail="Ranking signals do not match this run")
        order = sorted(range(len(records)), key=lambda index: (-float(scores[index]), index))
        if label_filter is not None:
            order = [index for index in order if records[index].get("label") == label_filter]
        if split_filter is not None:
            order = [index for index in order if records[index].get("split") == split_filter]
        rows = [
            {"rank": rank, "index": index, "score": float(scores[index])}
            for rank, index in enumerate(order, start=1)
        ]
        return run, int(scores_by_metric["k"]), rows

    @app.get("/api/v1/runs/{run_id}/rankings")
    def get_rankings(
        run_id: str,
        model_id: str,
        metric: Literal["outlier", "disagreement"] = "outlier",
        label_filter: str | None = None,
        split_filter: str | None = None,
    ) -> dict:
        _run, k, rows = ranked_rows(run_id, model_id, metric, label_filter, split_filter)
        descriptions = {
            "outlier": (
                f"Mean cosine distance to the {k} nearest records. Higher values are "
                "novelty clues for review, not a quality score or error verdict."
            ),
            "disagreement": (
                f"Fraction of the {k} nearest records with a different label. Higher "
                "values flag neighborhoods to review; they do not confirm a mislabel."
            ),
        }
        return {
            "model_id": model_id,
            "metric": metric,
            "k": k,
            "score_direction": "descending",
            "interpretation": descriptions[metric],
            "results": rows,
        }

    @app.get("/api/v1/runs/{run_id}/rankings.csv")
    def export_rankings(
        run_id: str,
        model_id: str,
        metric: Literal["outlier", "disagreement"] = "outlier",
        label_filter: str | None = None,
        split_filter: str | None = None,
    ) -> Response:
        run, _k, rows = ranked_rows(run_id, model_id, metric, label_filter, split_filter)
        output = StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow([
            "rank", "score", "index", "filename", "split", "label",
            "class_id", "obj_index", "bbox",
        ])
        for row in rows:
            record = run["records"][row["index"]]
            writer.writerow([
                row["rank"], f"{row['score']:.12g}", row["index"],
                record.get("filename", ""), record.get("split", ""),
                record.get("label", ""), record.get("class_id", ""),
                record.get("obj_index", ""),
                "" if record.get("bbox") is None else json.dumps(record["bbox"], separators=(",", ":")),
            ])
        return Response(
            content=output.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{metric}_ranking.csv"'},
        )

    def similar_result(
        run_id: str, request: SimilarSearchRequest,
    ) -> tuple[dict, list[int], list[float]]:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            raw_by_model = app.state.raw_embeddings.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.get("mode") != "classifier" or run.get("unit") != "image":
            raise HTTPException(
                status_code=409,
                detail="Similar-image search requires classifier image embeddings",
            )
        raw = (raw_by_model or {}).get(request.model_id)
        if raw is None:
            raise HTTPException(status_code=404, detail="Raw embeddings are unavailable for this model")
        vectors = np.asarray(raw, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(run["records"]):
            raise HTTPException(
                status_code=409,
                detail="The selected model embeddings do not match this run",
            )
        if request.query_index >= len(vectors):
            raise HTTPException(status_code=422, detail="The query image is outside this run")

        query = vectors[request.query_index]
        norms = np.linalg.norm(vectors, axis=1)
        query_norm = norms[request.query_index]
        denominators = norms * query_norm
        similarities = np.zeros(len(vectors), dtype=np.float32)
        np.divide(vectors @ query, denominators, out=similarities, where=denominators > 0)
        distances = np.maximum(0.0, 1.0 - np.clip(similarities, -1.0, 1.0))
        indices = np.asarray([index for index in range(len(vectors)) if index != request.query_index])
        order = np.lexsort((indices, distances[indices]))[:request.k]
        ranked_indices = indices[order].astype(int).tolist()
        ranked_distances = distances[ranked_indices].astype(float).tolist()
        return run, ranked_indices, ranked_distances

    @app.post("/api/v1/runs/{run_id}/similar")
    def find_similar_images(run_id: str, request: SimilarSearchRequest) -> dict:
        _run, indices, distances = similar_result(run_id, request)
        return {
            "query_index": request.query_index,
            "model_id": request.model_id,
            "results": [
                {"index": index, "distance": distance}
                for index, distance in zip(indices, distances)
            ],
        }

    @app.post("/api/v1/runs/{run_id}/similar.csv")
    def export_similar_images(run_id: str, request: SimilarSearchRequest) -> Response:
        run, indices, _distances = similar_result(run_id, request)
        paths: list[Path] = run["image_paths"]
        records = [
            {**record, "path": paths[index]}
            for index, record in enumerate(run["records"])
        ]
        csv_text = records_to_csv(records, [request.query_index, *indices])
        return Response(
            content=csv_text,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="similar_group.csv"'},
        )

    def text_search_result(run_id: str, request: TextSearchRequest) -> tuple[dict, list[int], list[float]]:
        query_text = request.query.strip()
        if not query_text:
            raise HTTPException(status_code=422, detail="The text query cannot be blank")
        with app.state.lock:
            run = app.state.runs.get(run_id)
            raw_by_model = app.state.raw_embeddings.get(run_id)
            text_encoder = app.state.text_encoders.get(request.model_id)
            cached_query = app.state.text_vectors.get(request.model_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.get("mode") != "classifier" or run.get("unit") != "image":
            raise HTTPException(status_code=409, detail="Text search requires classifier image embeddings")
        if not supports_text_query(request.model_id):
            raise HTTPException(status_code=409, detail="Text search requires a Chinese-CLIP model")
        raw = (raw_by_model or {}).get(request.model_id)
        if raw is None:
            raise HTTPException(status_code=404, detail="Chinese-CLIP image embeddings are unavailable for this run")
        if cached_query is not None and cached_query[0] == query_text:
            query = cached_query[1]
        else:
            if text_encoder is None:
                try:
                    text_encoder = text_loader(request.model_id)
                except Exception as exc:
                    raise HTTPException(
                        status_code=503,
                        detail="The local Chinese-CLIP text encoder could not be loaded",
                    ) from exc
                with app.state.lock:
                    text_encoder = app.state.text_encoders.setdefault(request.model_id, text_encoder)
            try:
                query = np.asarray(text_encoder(query_text), dtype=np.float32).reshape(-1)
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail="The Chinese-CLIP text query could not be encoded",
                ) from exc
            with app.state.lock:
                app.state.text_vectors[request.model_id] = (query_text, query)
        vectors = np.asarray(raw, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(run["records"]):
            raise HTTPException(status_code=409, detail="The selected model embeddings do not match this run")
        if (
            query.shape[0] != vectors.shape[1]
            or not np.isfinite(query).all()
            or not np.linalg.norm(query)
        ):
            raise HTTPException(status_code=422, detail="The text encoder returned an invalid embedding")
        norms = np.linalg.norm(vectors, axis=1)
        query_norm = np.linalg.norm(query)
        denominators = norms * query_norm
        similarities = np.zeros(len(vectors), dtype=np.float32)
        np.divide(vectors @ query, denominators, out=similarities, where=denominators > 0)
        distances = np.maximum(0.0, 1.0 - np.clip(similarities, -1.0, 1.0))
        indices = np.arange(len(vectors), dtype=np.int64)
        order = np.lexsort((indices, distances))[:request.k]
        ranked_indices = indices[order].astype(int).tolist()
        ranked_distances = distances[ranked_indices].astype(float).tolist()
        return run, ranked_indices, ranked_distances

    @app.post("/api/v1/runs/{run_id}/text-search")
    def search_images_by_text(run_id: str, request: TextSearchRequest) -> dict:
        _run, indices, distances = text_search_result(run_id, request)
        return {
            "query": request.query.strip(),
            "model_id": request.model_id,
            "results": [
                {"index": index, "distance": distance}
                for index, distance in zip(indices, distances)
            ],
        }

    @app.post("/api/v1/runs/{run_id}/text-search.csv")
    def export_text_search(run_id: str, request: TextSearchRequest) -> Response:
        run, indices, _distances = text_search_result(run_id, request)
        paths: list[Path] = run["image_paths"]
        records = [
            {**record, "path": paths[index]}
            for index, record in enumerate(run["records"])
        ]
        return Response(
            content=records_to_csv(records, indices),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="text_search.csv"'},
        )

    def duplicate_scan_result(run_id: str, request: DuplicateScanRequest) -> tuple[dict, list[tuple[int, int, int | float]]]:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            raw_by_model = app.state.raw_embeddings.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.get("mode") != "classifier" or run.get("unit") != "image":
            raise HTTPException(status_code=409, detail="Duplicate review requires classifier image embeddings")
        splits = [str(record["split"]) for record in run["records"]]
        if request.method == "phash":
            phashes = run.get("_phashes")
            if not isinstance(phashes, list) or len(phashes) != len(splits):
                raise HTTPException(status_code=409, detail="Perceptual hashes are unavailable for this run")
            pairs = find_duplicate_pairs_phash(
                phashes,
                max_hamming=request.max_hamming,
                splits=splits,
                cross_split_only=request.cross_split_only,
                max_pairs=request.max_pairs,
            )
        else:
            if not request.model_id:
                raise HTTPException(status_code=422, detail="Choose a model for embedding duplicate review")
            raw = (raw_by_model or {}).get(request.model_id)
            if raw is None:
                raise HTTPException(status_code=404, detail="The selected model embeddings are unavailable for this run")
            pairs = find_duplicate_pairs_embedding(
                raw,
                max_distance=request.max_distance,
                splits=splits,
                cross_split_only=request.cross_split_only,
                max_pairs=request.max_pairs,
            )
        return run, pairs

    @app.post("/api/v1/runs/{run_id}/duplicates")
    def scan_duplicate_images(run_id: str, request: DuplicateScanRequest) -> dict:
        _run, pairs = duplicate_scan_result(run_id, request)
        return {
            "method": request.method,
            "model_id": request.model_id if request.method == "embedding" else None,
            "cross_split_only": request.cross_split_only,
            "results": [
                {"left_index": left, "right_index": right, "distance": distance}
                for left, right, distance in pairs
            ],
        }

    @app.post("/api/v1/runs/{run_id}/duplicates.csv")
    def export_duplicate_pairs(run_id: str, request: DuplicateScanRequest) -> Response:
        run, pairs = duplicate_scan_result(run_id, request)
        paths: list[Path] = run["image_paths"]
        output = StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow([
            "left_index", "left_filename", "left_path", "left_split",
            "right_index", "right_filename", "right_path", "right_split", "distance",
        ])
        for left, right, distance in pairs:
            a, b = run["records"][left], run["records"][right]
            writer.writerow([
                left, a["filename"], str(paths[left]), a["split"],
                right, b["filename"], str(paths[right]), b["split"], distance,
            ])
        return Response(
            content=output.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="duplicate_pairs.csv"'},
        )

    @app.post("/api/v1/runs/{run_id}/duplicates/labeling-handoff", status_code=201)
    def send_duplicate_pairs_to_labeling(
        run_id: str, request: DuplicateLabelingHandoffRequest,
    ) -> dict:
        import labeling_handoff as _labeling_handoff
        import pool_registry as _pool_registry
        import round_ledger as _round_ledger

        with app.state.lock:
            run, pairs = duplicate_scan_result(run_id, request)
            if not pairs:
                raise HTTPException(status_code=422, detail={
                    "code": "DUPLICATE_HANDOFF_EMPTY",
                    "message": "There are no duplicate pairs to send for review.",
                    "request_id": uuid.uuid4().hex,
                })

            records: list[dict] = []
            manifest: dict[str, dict] = {}
            local_index_by_run_index: dict[int, int] = {}
            sha_by_run_index: dict[int, str] = {}
            expected_hashes = run.get("_source_sha256") or []
            run_indices = sorted({index for left, right, _ in pairs for index in (left, right)})
            for run_index in run_indices:
                try:
                    path = Path(run["image_paths"][run_index]).expanduser().resolve(strict=True)
                    if not path.is_file():
                        raise OSError("source is not a file")
                    sha = _sha256_file(path)
                except (OSError, RuntimeError, ValueError, IndexError, TypeError) as exc:
                    raise HTTPException(status_code=409, detail={
                        "code": "DUPLICATE_SOURCE_UNAVAILABLE",
                        "message": "A duplicate-review image is no longer available. Re-run the scan before sending it.",
                        "request_id": uuid.uuid4().hex,
                    }) from exc
                expected = expected_hashes[run_index] if run_index < len(expected_hashes) else None
                if expected and sha != expected:
                    raise HTTPException(status_code=409, detail={
                        "code": "DUPLICATE_SOURCE_CHANGED",
                        "message": "A duplicate-review image changed after the scan. Re-run the scan before sending it.",
                        "request_id": uuid.uuid4().hex,
                    })
                source_record = run["records"][run_index]
                local_index_by_run_index[run_index] = len(records)
                sha_by_run_index[run_index] = sha
                records.append({
                    "path": str(path),
                    "filename": str(source_record.get("filename") or path.name),
                    "label": str(source_record.get("label") or ""),
                    "split": str(source_record.get("split") or ""),
                })
                manifest[str(path)] = {"sha256": sha}

            unique_shas = list(dict.fromkeys(sha_by_run_index.values()))
            new_shas, handled_shas = _pool_registry.partition_new(unique_shas)
            new_set = set(new_shas)
            eligible_pairs = [
                (left, right, distance)
                for left, right, distance in pairs
                if request.include_handled
                or (sha_by_run_index[left] in new_set and sha_by_run_index[right] in new_set)
            ]
            if not eligible_pairs:
                raise HTTPException(status_code=422, detail={
                    "code": "DUPLICATE_HANDOFF_EMPTY",
                    "message": "All duplicate pairs contain previously sent images. Enable the explicit override to include them.",
                    "request_id": uuid.uuid4().hex,
                })

            participating_indices = {
                index
                for left, right, _ in eligible_pairs
                for index in (left, right)
            }
            send_indices = [
                local_index_by_run_index[index]
                for index in sorted(participating_indices)
            ]
            send_shas = {sha_by_run_index[index] for index in participating_indices}
            handoff_pairs = []
            for left, right, distance in eligible_pairs:
                left_record = run["records"][left]
                right_record = run["records"][right]
                handoff_pairs.append({
                    "left_sha256": sha_by_run_index[left],
                    "right_sha256": sha_by_run_index[right],
                    "left_filename": str(left_record.get("filename") or Path(run["image_paths"][left]).name),
                    "right_filename": str(right_record.get("filename") or Path(run["image_paths"][right]).name),
                    "left_split": str(left_record.get("split") or ""),
                    "right_split": str(right_record.get("split") or ""),
                    "distance": int(distance) if request.method == "phash" else float(distance),
                })

            rounds = _round_ledger.load_rounds()
            round_ids = {row["round_id"] for row in rounds}
            target_round = request.round_id
            if not request.new_round and target_round is None and rounds:
                target_round = rounds[-1]["round_id"]
            if not request.new_round and target_round is not None and target_round not in round_ids:
                raise HTTPException(status_code=404, detail={
                    "code": "ROUND_NOT_FOUND",
                    "message": "The selected active-learning round no longer exists.",
                    "request_id": uuid.uuid4().hex,
                })

            class_options = list(dict.fromkeys(
                records[local_index_by_run_index[index]]["label"]
                for index in sorted(participating_indices)
                if records[local_index_by_run_index[index]]["label"]
            )) or ["object"]
            payload = {
                "kind": "duplicate_pair_review",
                "method": request.method,
                "model_id": request.model_id if request.method == "embedding" else None,
                "cross_split_only": request.cross_split_only,
                "pairs": handoff_pairs,
            }
            try:
                handoff_dir = _labeling_handoff.send_to_labeling(
                    records,
                    send_indices,
                    source="duplicate",
                    task=_labeling_handoff.TASK_VERIFY,
                    class_options=class_options,
                    manifest=manifest,
                    original_labels={
                        index: record["label"]
                        for index, record in enumerate(records)
                    },
                    payload=payload,
                    instructions=(
                        "Review each supplied pair together as a possible duplicate or split leak. "
                        "Keep/drop decisions are human review only; do not delete source images automatically."
                    ),
                )
                if handoff_dir is None:
                    raise ValueError("no images were materialized")
                spec = _labeling_handoff.load_spec(handoff_dir)
                sent_shas = [item["sha256"] for item in spec.get("items", [])]
                batch_id = str(spec.get("handoff_id") or Path(handoff_dir).name)
                if not sent_shas or not send_shas.intersection(sent_shas):
                    raise ValueError("the handoff contains no selected duplicate-review images")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.exception("LV duplicate-pair handoff could not be materialized")
                raise HTTPException(status_code=422, detail={
                    "code": "DUPLICATE_HANDOFF_FAILED",
                    "message": "The duplicate-review images could not be prepared for Labeling. Check that the source files are still readable.",
                    "request_id": uuid.uuid4().hex,
                }) from exc

            _pool_registry.mark_sent(sent_shas, batch_id, tool="duplicate")
            created_round = request.new_round or not rounds
            if created_round:
                round_id = _round_ledger.start_round(
                    "duplicate",
                    {
                        "method": request.method,
                        "model_id": request.model_id if request.method == "embedding" else None,
                        "cross_split_only": request.cross_split_only,
                        "max_hamming": request.max_hamming,
                        "max_distance": request.max_distance,
                    },
                    "duplicate",
                )
            else:
                round_id = target_round or rounds[-1]["round_id"]
            _round_ledger.attach_batch(round_id, batch_id, len(sent_shas))
            return {
                "handoff_id": batch_id,
                "task": str(spec.get("task") or _labeling_handoff.TASK_VERIFY),
                "sent": len(sent_shas),
                "pair_count": len(handoff_pairs),
                "skipped_handled": len(handled_shas) if not request.include_handled else 0,
                "skipped_pairs": len(pairs) - len(eligible_pairs),
                "round_id": round_id,
                "created_round": created_round,
                "saved_to_local_inbox": True,
            }

    @app.post("/api/v1/runs/{run_id}/blocks")
    def get_kmeans_blocks(run_id: str, request: KMeansBlocksRequest) -> dict:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            raw_by_model = app.state.raw_embeddings.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.get("mode") not in {"classifier", "detector"} or run.get("unit") not in {"image", "object"}:
            raise HTTPException(status_code=409, detail="KMeans blocks require a supported visualization run")
        raw = (raw_by_model or {}).get(request.model_id)
        if raw is None:
            raise HTTPException(status_code=404, detail="Raw embeddings are unavailable for this model")
        view = next((item for item in run["views"]
                     if item["model_id"] == request.model_id and item["method"] == request.method), None)
        if view is None:
            raise HTTPException(status_code=404, detail="The requested projection is unavailable")

        view_points = [
            point for point in view["points"]
            if (request.label_filter is None or point["label"] == request.label_filter)
            and (request.split_filter is None or point["split"] == request.split_filter)
        ]
        if not view_points:
            raise HTTPException(status_code=422, detail="No images match the current filters")

        cache_key = (
            run_id, request.model_id, request.method, request.k,
            request.label_filter, request.split_filter,
        )
        with app.state.lock:
            base = app.state.block_cache.get(cache_key)
        if base is None:
            indices = np.asarray([int(point["index"]) for point in view_points], dtype=np.int64)
            coordinates = np.asarray([
                [float(point["x"]), float(point["y"]), float(point["z"])]
                for point in view_points
            ], dtype=np.float64)
            kk = max(1, min(request.k, len(indices)))
            fitted = KMeans(n_clusters=kk, n_init=4, random_state=42).fit(coordinates)
            counts = np.bincount(fitted.labels_, minlength=kk)
            centers = [
                {
                    "block_id": block_id,
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "z": float(center[2]),
                    "count": int(counts[block_id]),
                }
                for block_id, center in enumerate(fitted.cluster_centers_)
            ]
            base = {
                "k": kk,
                "centers": centers,
                "assignments": {
                    int(index): int(block_id)
                    for index, block_id in zip(indices, fitted.labels_)
                },
                "points_by_index": {int(point["index"]): point for point in view_points},
            }
            with app.state.lock:
                app.state.block_cache[cache_key] = base

        block_points = [
            {**base["points_by_index"][index], "block_id": block_id}
            for index, block_id in sorted(base["assignments"].items())
        ]
        drilldown_points: list[dict] = []
        if request.block_id is not None:
            if request.block_id >= base["k"]:
                raise HTTPException(status_code=422, detail="The selected block is outside the current KMeans result")
            members = [
                index for index, block_id in sorted(base["assignments"].items())
                if block_id == request.block_id
            ]
            member_points = [base["points_by_index"][index] for index in members]
            member_vectors = raw[members]
            if len(members) > 2 and float(np.var(member_vectors)) > 0.0:
                projected = PCA(n_components=2, random_state=42).fit_transform(member_vectors)
            else:
                projected = np.asarray([[point["x"], point["y"]] for point in member_points])
            drilldown_points = [
                {**point, "x": float(xy[0]), "y": float(xy[1]), "z": 0.0}
                for point, xy in zip(member_points, projected)
            ]

        return {
            "k": base["k"],
            "centers": base["centers"],
            "points": block_points,
            "block_id": request.block_id,
            "drilldown_points": drilldown_points,
        }

    @app.get("/api/v1/runs/{run_id}/images/{index}")
    def get_run_image(run_id: str, index: int) -> Response:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
            paths: list[Path] = run["image_paths"]
            if index < 0 or index >= len(paths):
                raise HTTPException(status_code=404, detail="Image not found")
            path = paths[index]
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise HTTPException(status_code=404, detail="Image is no longer available") from exc
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return Response(content=content, media_type=media_type, headers={"Cache-Control": "no-store"})

    static_dir = Path(__file__).resolve().parent.parent / "web" / "dist"
    if static_dir.is_dir():
        # NativeApp can serve the same build through its host, while direct local
        # runs and browser E2E use this same-origin fallback.
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="react-web")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=_configured_api_port())
