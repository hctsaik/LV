"""Loopback FastAPI adapter for LV's React web interface."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import mimetypes
import os
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from starlette.responses import JSONResponse, Response
from starlette.staticfiles import StaticFiles

from _utils import available_models, load_model, load_text_encoder, supports_text_query
from interaction import (
    crop_bbox,
    draw_yolo_boxes,
    find_duplicate_pairs_embedding,
    find_duplicate_pairs_phash,
    records_to_csv,
    snapshots_to_csv,
    zip_selected_images,
    yolo_label_path_for,
)
from curation_store import CurationStore, CurationStoreError
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


class CoverageObjectViewRequest(BaseModel):
    class_label: str | None = Field(default=None, max_length=256)
    min_short_px: int = Field(default=0, ge=0, le=4096)
    confidence_min: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_max: float = Field(default=1.0, ge=0.0, le=1.0)
    method: Literal["pca", "tsne", "umap", "lda", "sumap"] = "pca"
    dimension: Literal[2, 3] = 2
    k: int = Field(default=10, ge=1, le=1000)
    block_k: int = Field(default=8, ge=3, le=20)


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


def _job_view(job: _Job | _ComparisonJob | _CoverageJob) -> dict:
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
) -> FastAPI:
    catalog = model_catalog or available_models
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
    app.state.curation_store = CurationStore()
    app.state.lock = threading.RLock()

    def has_active_job() -> bool:
        jobs = (
            list(app.state.jobs.values())
            + list(app.state.comparison_jobs.values())
            + list(app.state.coverage_jobs.values())
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
                if key not in {"image_paths", "_raw_embeddings", "_phashes"}
            }

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
