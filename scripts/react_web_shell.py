"""Loopback FastAPI adapter for LV's React web interface."""
from __future__ import annotations

import csv
import base64
import copy
import hashlib
import json
import logging
import mimetypes
import os
import threading
import uuid
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

from _utils import available_models, load_model, load_text_encoder, supports_text_query
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
from fp_case_search import add_candidate as add_fp_search_candidate
from fp_case_search import model_recipe as fp_case_model_recipe
from fp_case_search import search_case as run_fp_case_search
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
from react_visualize import JobCancelled, VisualizationInputError, run_image_visualization
from safe_io import safe_open_image

API_VERSION = "1"
logger = logging.getLogger("lv.react_web")


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


class CreateComparisonJob(BaseModel):
    folders_a: list[str] = Field(min_length=1, max_length=32)
    folders_b: list[str] = Field(min_length=1, max_length=32)
    model_id: str = Field(min_length=1, max_length=128)
    unit: Literal["whole", "object"] = "object"
    class_names: list[str] = Field(default_factory=list, max_length=256)
    object_policy: ComparisonObjectPolicy | None = None


class CreateCoverageJob(BaseModel):
    folders: list[str] = Field(min_length=1, max_length=32)
    model_id: str = Field(min_length=1, max_length=128)
    unit: Literal["image", "object"] = "image"
    method: Literal["pca", "tsne", "umap", "lda", "sumap"] = "sumap"
    dimension: Literal[2, 3] = 2
    k: int = Field(default=10, ge=1, le=1000)


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


class AddCurationRequest(BaseModel):
    indices: list[int] = Field(min_length=1, max_length=10000)
    source: str = Field(default="manual", min_length=1, max_length=128)
    reason: str = Field(default="", max_length=2000)


class SetCurationReasonRequest(BaseModel):
    item_ids: list[str] = Field(min_length=1, max_length=10000)
    reason: str = Field(min_length=1, max_length=2000)


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


def _job_view(job: _Job | _ComparisonJob | _CoverageJob | _CoverageCandidateJob) -> dict:
    return {
        "job_id": job.job_id,
        "state": job.state,
        "phase": job.phase,
        "done": job.done,
        "total": job.total,
        "percent": (job.done / job.total * 100.0) if job.total else 0.0,
        "message": job.message,
        "run_id": job.run_id,
        "error": job.error,
        "can_cancel": job.state in {"queued", "running", "cancelling"},
    }


def create_app(
    *,
    model_catalog: Callable[[], list[str]] | None = None,
    model_loader: Callable[[str], Callable] | None = None,
    text_model_loader: Callable[[str], Callable[[str], np.ndarray]] | None = None,
    object_model_loader: Callable | None = None,
    diagnostic_model_loader: Callable[[str, int, bool, str], Callable] | None = None,
    case_search_model_provider: Callable[[], tuple[dict, Callable]] | None = None,
) -> FastAPI:
    catalog = model_catalog or available_models
    has_custom_model_loader = model_loader is not None
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
    app.state.curation_store = CurationStore()
    app.state.fp_case_store = FalsePositiveCaseStore()
    app.state.case_search_model_provider = case_search_model_provider
    app.state.fp_case_lock = threading.RLock()
    app.state.lock = threading.RLock()

    def has_active_job() -> bool:
        jobs = (
            list(app.state.jobs.values())
            + list(app.state.comparison_jobs.values())
            + list(app.state.coverage_jobs.values())
            + list(app.state.coverage_candidate_jobs.values())
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

            result = run_image_visualization(
                job.request.folders,
                model_names=selected_models,
                methods=selected_methods,
                mode=job.request.mode,
                unit=job.request.unit,
                class_names=job.request.class_names,
                object_policy=(
                    job.request.object_policy.model_dump()
                    if job.request.object_policy is not None else None
                ),
                umap_reference=job.request.umap_reference,
                umap_rebuild=job.request.umap_rebuild,
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

    @app.post("/api/v1/visualize/jobs", status_code=202)
    def create_visualization_job(request: CreateVisualizationJob) -> dict:
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
            job = app.state.jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            return _job_view(job)

    @app.delete("/api/v1/jobs/{job_id}", status_code=202)
    def cancel_job(job_id: str) -> dict:
        with app.state.lock:
            job = app.state.jobs.get(job_id)
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
                    "score": None,
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

    @app.post("/api/v1/runs/{run_id}/blocks")
    def get_kmeans_blocks(run_id: str, request: KMeansBlocksRequest) -> dict:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            raw_by_model = app.state.raw_embeddings.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.get("mode") != "classifier" or run.get("unit") != "image":
            raise HTTPException(status_code=409, detail="KMeans blocks require classifier image embeddings")
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

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("LV_WEB_API_PORT", "8765")))
