"""Loopback FastAPI adapter for LV's React web interface."""
from __future__ import annotations

import csv
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
    draw_yolo_boxes,
    find_duplicate_pairs_embedding,
    find_duplicate_pairs_phash,
    records_to_csv,
    yolo_label_path_for,
)
from react_compare import ComparisonInputError, project_comparison, run_comparison
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


def _job_view(job: _Job) -> dict:
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
    app.state.text_encoders = {}
    app.state.text_vectors = {}
    app.state.block_cache = {}
    app.state.comparison_jobs = {}
    app.state.comparison_runs = {}
    app.state.comparison_projections = {}
    app.state.lock = threading.RLock()

    def has_active_job() -> bool:
        jobs = list(app.state.jobs.values()) + list(app.state.comparison_jobs.values())
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
            run = {key: value for key, value in result.items() if key != "image_paths"}
            run["image_paths"] = result["image_paths"]
            with app.state.lock:
                # Retain high-dimensional vectors only for the active visual run;
                # old run summaries stay readable without pinning model arrays.
                app.state.raw_embeddings.clear()
                app.state.block_cache.clear()
                app.state.runs[run_id] = run
                app.state.raw_embeddings[run_id] = raw_embeddings
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
