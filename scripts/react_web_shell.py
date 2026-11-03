"""Loopback FastAPI adapter for LV's React web interface."""
from __future__ import annotations

import logging
import mimetypes
import os
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, Response
from starlette.staticfiles import StaticFiles

from _utils import available_models, load_model, load_text_encoder, supports_text_query
from react_compare import CompareCancelled, CompareInputError, project_comparison, run_image_comparison
from react_visualize import JobCancelled, VisualizationInputError, run_visualization

API_VERSION = "1"
logger = logging.getLogger("lv.react_web")


class CreateObjectPolicy(BaseModel):
    pad: float = Field(default=0.12, ge=0, le=0.5)
    target_res: Literal[168, 196, 224, 280, 336, 392, 448] = 224
    head: Literal["cls", "meanpool"] = "cls"


class CreateVisualizationJob(BaseModel):
    folders: list[str] = Field(min_length=1, max_length=32)
    model_id: str | None = Field(default=None, min_length=1, max_length=128)
    method: str = "pca"
    model_ids: list[str] = Field(default_factory=list, max_length=8)
    methods: list[str] = Field(default_factory=list, max_length=5)
    mode: Literal["image_classifier", "object_detector"] = "image_classifier"
    granularity: Literal["whole_image", "object"] = "whole_image"
    classes_file_text: str | None = Field(default=None, max_length=128_000)
    class_names: list[str] = Field(default_factory=list, max_length=512)
    object_policy: CreateObjectPolicy = Field(default_factory=CreateObjectPolicy)
    umap_reference: bool = False
    rebuild_umap_reference: bool = False


class CreateCompareJob(BaseModel):
    folders_a: list[str] = Field(min_length=1, max_length=32)
    folders_b: list[str] = Field(min_length=1, max_length=32)
    model_id: str = Field(min_length=1, max_length=128)


class CreateCompareProjection(BaseModel):
    method: str = Field(min_length=1, max_length=16)
    dimension: int = Field(default=2, ge=2, le=3)


class CreateRunBlocks(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)
    method: str = Field(min_length=1, max_length=16)
    indices: list[int] = Field(min_length=1, max_length=100_000)
    k: int = Field(default=12, ge=4, le=30)


class CreateBlockProjection(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)
    method: str = Field(min_length=1, max_length=16)
    indices: list[int] = Field(min_length=1, max_length=100_000)


class CreateSimilarSearch(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)
    query_index: int = Field(ge=0)
    k: int = Field(default=9, ge=1, le=100_000)


class CreateTextSearch(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=512)
    k: int = Field(default=9, ge=1, le=100_000)


class CreateDuplicateScan(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)
    method: Literal["phash", "embedding"] = "phash"
    cross_split_only: bool = False
    phash_threshold: int = Field(default=4, ge=0, le=16)
    embedding_threshold: float = Field(default=0.05, ge=0, le=0.5)


class CreateRanking(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)
    criterion: Literal["outlier", "disagreement", "filename", "spatial"]
    indices: list[int] | None = Field(default=None, max_length=100_000)
    method: str = Field(default="pca", min_length=1, max_length=16)


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
class _CompareJob:
    job_id: str
    request: CreateCompareJob
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


def _run_embedding_matrix(run: dict, model_id: str) -> np.ndarray:
    """Read one whole-image model matrix from its run-bound app cache."""
    records = run.get("records", [])
    if run.get("source_granularity") != "whole_image":
        raise HTTPException(status_code=409, detail={
            "code": "WHOLE_IMAGE_RUN_REQUIRED",
            "message": "This search is available for whole-image runs. Run whole-image analysis first.",
            "request_id": uuid.uuid4().hex,
        })
    if model_id not in run.get("models", []):
        raise HTTPException(status_code=404, detail={
            "code": "MODEL_NOT_IN_RUN",
            "message": "Run the analysis with this feature model before searching across models.",
            "request_id": uuid.uuid4().hex,
        })
    try:
        expected_shape = tuple(int(value) for value in run["embedding_shapes"][model_id])
    except (KeyError, TypeError, ValueError):
        expected_shape = ()
    if len(expected_shape) != 2 or expected_shape[0] != len(records) or expected_shape[1] <= 0:
        raise HTTPException(status_code=409, detail={
            "code": "EMBEDDING_CACHE_UNAVAILABLE",
            "message": "The feature cache for this model is unavailable. Run the analysis again.",
            "request_id": uuid.uuid4().hex,
        })

    matrix = np.empty(expected_shape, dtype=np.float32)
    seen_indices: set[int] = set()
    cache_error = False
    for source in run.get("_embedding_sources", {}).get(model_id, []):
        try:
            indices = [int(value) for value in source.get("indices", [])]
            keys = [str(value) for value in source.get("keys", [])]
        except (TypeError, ValueError):
            cache_error = True
            break
        if len(indices) != len(keys) or len(set(indices)) != len(indices):
            cache_error = True
            break
        try:
            with np.load(str(source["cache_path"]), allow_pickle=False) as cache:
                cache_keys = [str(key) for key in cache["keys"].tolist()]
                cache_rows = {key: row for row, key in enumerate(cache_keys)}
                embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
                if (embeddings.ndim != 2 or embeddings.shape[0] != len(cache_keys)
                        or embeddings.shape[1] != expected_shape[1]):
                    cache_error = True
                    break
                for index, key in zip(indices, keys):
                    if (index < 0 or index >= len(records) or index in seen_indices
                            or key not in cache_rows):
                        cache_error = True
                        break
                    matrix[index] = embeddings[cache_rows[key]]
                    seen_indices.add(index)
        except (OSError, KeyError, ValueError, TypeError):
            cache_error = True
        if cache_error:
            break
    if cache_error or seen_indices != set(range(len(records))):
        raise HTTPException(status_code=409, detail={
            "code": "EMBEDDING_CACHE_UNAVAILABLE",
            "message": "The feature cache for this model is unavailable. Run the analysis again.",
            "request_id": uuid.uuid4().hex,
        })

    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(matrix).all() or np.any(norms <= 0):
        raise HTTPException(status_code=409, detail={
            "code": "INVALID_EMBEDDING_CACHE",
            "message": "The feature cache contains invalid vectors. Run the analysis again.",
            "request_id": uuid.uuid4().hex,
        })
    return matrix


def _rank_cosine(matrix: np.ndarray, query_vector: np.ndarray, k: int,
                 excluded_index: int | None = None) -> list[tuple[int, float]]:
    matrix = np.asarray(matrix, dtype=np.float32)
    query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if matrix.ndim != 2 or query_vector.shape[0] != matrix.shape[1]:
        raise HTTPException(status_code=422, detail={
            "code": "QUERY_DIMENSION_MISMATCH",
            "message": "The query vector is incompatible with this model's image features.",
            "request_id": uuid.uuid4().hex,
        })
    query_norm = float(np.linalg.norm(query_vector))
    row_norms = np.linalg.norm(matrix, axis=1)
    if (not np.isfinite(query_vector).all() or query_norm <= 0
            or not np.isfinite(matrix).all() or np.any(row_norms <= 0)):
        raise HTTPException(status_code=422, detail={
            "code": "INVALID_QUERY_VECTOR",
            "message": "The query encoder returned an invalid feature vector.",
            "request_id": uuid.uuid4().hex,
        })
    cosine = (matrix @ query_vector) / (row_norms * query_norm)
    distances = np.clip(1.0 - cosine.astype(np.float64), 0.0, 2.0)
    candidate_count = len(matrix) - (1 if excluded_index is not None else 0)
    indices = sorted(
        (index for index in range(len(matrix)) if index != excluded_index),
        key=lambda index: (float(distances[index]), index),
    )[: min(k, max(0, candidate_count))]
    return [(index, float(distances[index])) for index in indices]


def _similar_search(run: dict, model_id: str, query_index: int, k: int) -> dict:
    """Rank whole-image records by exact cosine distance from one run record."""
    records = run.get("records", [])
    if k < 1:
        raise HTTPException(status_code=422, detail={
            "code": "INVALID_RESULT_COUNT",
            "message": "The number of similar images must be at least one.",
            "request_id": uuid.uuid4().hex,
        })
    if run.get("source_granularity") != "whole_image":
        raise HTTPException(status_code=409, detail={
            "code": "WHOLE_IMAGE_RUN_REQUIRED",
            "message": "Image similarity is available for whole-image runs. Run whole-image analysis first.",
            "request_id": uuid.uuid4().hex,
        })
    if model_id not in run.get("models", []):
        raise HTTPException(status_code=404, detail={
            "code": "MODEL_NOT_IN_RUN",
            "message": "Run the analysis with this feature model before searching across models.",
            "request_id": uuid.uuid4().hex,
        })
    if not 0 <= query_index < len(records):
        raise HTTPException(status_code=422, detail={
            "code": "INVALID_QUERY_INDEX",
            "message": "The selected query image is outside this run.",
            "request_id": uuid.uuid4().hex,
        })
    if len(records) < 2:
        return {"model_id": model_id, "query_index": query_index, "results": []}

    matrix = _run_embedding_matrix(run, model_id)
    ranked = _rank_cosine(matrix, matrix[query_index], k, excluded_index=query_index)
    return {
        "model_id": model_id,
        "query_index": query_index,
        "results": [
            {**records[index], "distance": distance}
            for index, distance in ranked
        ],
    }


def _text_search(run: dict, model_id: str, query: str, k: int,
                 text_encoder: Callable[[str], np.ndarray]) -> dict:
    query = query.strip()
    if not query:
        raise HTTPException(status_code=422, detail={
            "code": "EMPTY_TEXT_QUERY",
            "message": "Enter a text query before searching.",
            "request_id": uuid.uuid4().hex,
        })
    if len(query) > 512:
        raise HTTPException(status_code=422, detail={
            "code": "TEXT_QUERY_TOO_LONG",
            "message": "The text query must be 512 characters or fewer.",
            "request_id": uuid.uuid4().hex,
        })
    if not supports_text_query(model_id):
        raise HTTPException(status_code=409, detail={
            "code": "TEXT_QUERY_UNAVAILABLE",
            "message": "Text search requires a Chinese-CLIP model included in this run.",
            "request_id": uuid.uuid4().hex,
        })
    if model_id not in run.get("models", []):
        raise HTTPException(status_code=404, detail={
            "code": "MODEL_NOT_IN_RUN",
            "message": "Run the analysis with this text-capable model before searching.",
            "request_id": uuid.uuid4().hex,
        })
    if k < 1:
        raise HTTPException(status_code=422, detail={
            "code": "INVALID_RESULT_COUNT",
            "message": "The number of search results must be at least one.",
            "request_id": uuid.uuid4().hex,
        })
    if not run.get("records"):
        return {"model_id": model_id, "query": query, "results": []}
    try:
        query_vector = text_encoder(query)
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail={
            "code": "TEXT_MODEL_UNAVAILABLE",
            "message": "The selected local text model is unavailable.",
            "request_id": uuid.uuid4().hex,
        }) from None
    except Exception:
        logger.exception("LV React text query failed (model=%s)", model_id)
        raise HTTPException(status_code=503, detail={
            "code": "TEXT_QUERY_FAILED",
            "message": "The local text query could not be encoded. Check the worker log and retry.",
            "request_id": uuid.uuid4().hex,
        }) from None
    matrix = _run_embedding_matrix(run, model_id)
    ranked = _rank_cosine(matrix, query_vector, k)
    records = run["records"]
    return {
        "model_id": model_id,
        "query": query,
        "results": [
            {**records[index], "distance": distance}
            for index, distance in ranked
        ],
    }


def _duplicate_scan(run: dict, request: CreateDuplicateScan) -> dict:
    records = run.get("records", [])
    if run.get("source_granularity") != "whole_image":
        raise HTTPException(status_code=409, detail={
            "code": "WHOLE_IMAGE_RUN_REQUIRED",
            "message": "Duplicate review is available for whole-image runs. Run whole-image analysis first.",
            "request_id": uuid.uuid4().hex,
        })
    if len(records) < 2:
        return {
            "model_id": request.model_id,
            "method": request.method,
            "cross_split_only": request.cross_split_only,
            "total_pairs": 0,
            "pairs": [],
        }
    splits = [str(record.get("split", "")) for record in records]
    if request.method == "phash":
        phashes = run.get("_phashes", [])
        if len(phashes) != len(records):
            raise HTTPException(status_code=409, detail={
                "code": "PHASH_CACHE_UNAVAILABLE",
                "message": "Perceptual hashes are unavailable for this run. Run the analysis again.",
                "request_id": uuid.uuid4().hex,
            })
        from interaction import find_duplicate_pairs_phash

        pairs = find_duplicate_pairs_phash(
            phashes,
            max_hamming=request.phash_threshold,
            splits=splits,
            cross_split_only=request.cross_split_only,
        )
    else:
        from interaction import find_duplicate_pairs_embedding

        matrix = _run_embedding_matrix(run, request.model_id)
        pairs = find_duplicate_pairs_embedding(
            matrix,
            max_distance=request.embedding_threshold,
            splits=splits,
            cross_split_only=request.cross_split_only,
        )
    return {
        "model_id": request.model_id,
        "method": request.method,
        "cross_split_only": request.cross_split_only,
        "total_pairs": len(pairs),
        "pairs": [
            {
                "left": records[left],
                "right": records[right],
                "distance": distance,
            }
            for left, right, distance in pairs
        ],
    }
    cosine = (matrix @ matrix[query_index]) / (norms * norms[query_index])
    distances = np.clip(1.0 - cosine.astype(np.float64), 0.0, 2.0)
    ranked = sorted(
        (index for index in range(len(records)) if index != query_index),
        key=lambda index: (float(distances[index]), index),
    )[: min(k, len(records) - 1)]
    return {
        "model_id": model_id,
        "query_index": query_index,
        "results": [
            {**records[index], "distance": float(distances[index])}
            for index in ranked
        ],
    }


def create_app(
    *,
    model_catalog: Callable[[], list[str]] | None = None,
    model_loader: Callable[[str], Callable] | None = None,
    object_model_loader: Callable[[str, dict], Callable] | None = None,
    comparison_runner: Callable[..., dict] | None = None,
    text_encoder_loader: Callable[[str], Callable[[str], np.ndarray]] | None = None,
) -> FastAPI:
    catalog = model_catalog or available_models
    loader = model_loader or load_model
    object_loader = object_model_loader
    if object_loader is None and model_loader is not None:
        object_loader = lambda model_id, _policy: loader(model_id)
    text_loader = text_encoder_loader or load_text_encoder
    app = FastAPI(title="VisualLatent Local API", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.jobs = {}
    app.state.runs = {}
    app.state.compare_jobs = {}
    app.state.compare_runs = {}
    app.state.text_encoders = {}
    app.state.text_vectors = {}
    app.state.compare_projection_active = False
    app.state.lock = threading.RLock()

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

            result = run_visualization(
                job.request.folders,
                model_names=selected_models,
                methods=selected_methods,
                mode=job.request.mode,
                granularity=job.request.granularity,
                classes_file_text=job.request.classes_file_text,
                manual_class_names=job.request.class_names,
                object_policy=(
                    job.request.object_policy.model_dump()
                    if hasattr(job.request.object_policy, "model_dump")
                    else job.request.object_policy.dict()
                ),
                object_model_loader=object_loader,
                umap_reference=job.request.umap_reference,
                rebuild_umap_reference=job.request.rebuild_umap_reference,
                model_loader=loader,
                progress=update_progress,
                cancelled=job.cancel_event.is_set,
            )
            if job.cancel_event.is_set():
                raise JobCancelled()

            run_id = uuid.uuid4().hex
            run = {
                key: value for key, value in result.items() if key != "image_paths"
            }
            run["image_paths"] = result["image_paths"]
            with app.state.lock:
                app.state.runs[run_id] = run
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
        selected_methods = request.methods or [request.method]
        if request.rebuild_umap_reference and (
            not request.umap_reference or "umap" not in selected_methods
        ):
            raise HTTPException(status_code=422, detail={
                "code": "INVALID_UMAP_REFERENCE",
                "message": "Rebuild requires the fixed UMAP reference option and UMAP projection.",
                "request_id": uuid.uuid4().hex,
            })
        job_id = uuid.uuid4().hex
        job = _Job(job_id=job_id, request=request)
        with app.state.lock:
            if app.state.compare_projection_active or any(
                item.state in {"queued", "running", "cancelling"}
                for item in [*app.state.jobs.values(), *app.state.compare_jobs.values()]
            ):
                raise HTTPException(status_code=409, detail={
                    "code": "JOB_ALREADY_RUNNING",
                    "message": "Wait for the active analysis to finish before starting another.",
                    "request_id": job_id,
                })
            app.state.jobs[job_id] = job
        threading.Thread(target=execute, args=(job,), daemon=True, name=f"lv-web-{job_id[:8]}").start()
        return {"job_id": job_id, "state": job.state}

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

    def execute_compare(job: _CompareJob) -> None:
        try:
            with app.state.lock:
                if job.cancel_event.is_set():
                    raise JobCancelled()
                job.state, job.phase, job.message = "running", "discovery", "Scanning comparison folders"
            if job.request.model_id not in set(catalog()):
                raise CompareInputError("MODEL_UNAVAILABLE", "The selected feature model is not installed.")

            def update_progress(phase: str, done: int, total: int, message: str) -> None:
                with app.state.lock:
                    job.phase = phase
                    job.done = done
                    job.total = total
                    job.message = message

            if comparison_runner is None:
                result = run_image_comparison(
                    job.request.folders_a,
                    job.request.folders_b,
                    job.request.model_id,
                    model_loader=loader,
                    progress=update_progress,
                    cancelled=job.cancel_event.is_set,
                )
            else:
                result = comparison_runner(
                    job.request.folders_a,
                    job.request.folders_b,
                    job.request.model_id,
                    progress=update_progress,
                    cancelled=job.cancel_event.is_set,
                )
            if job.cancel_event.is_set():
                raise JobCancelled()

            run_id = uuid.uuid4().hex
            with app.state.lock:
                app.state.compare_runs[run_id] = result
                job.run_id = run_id
                job.state, job.phase = "completed", "completed"
                job.done = job.total or 1
                job.total = job.total or 1
                job.message = "Completed"
        except (JobCancelled, CompareCancelled):
            with app.state.lock:
                job.state, job.phase, job.message = "cancelled", "cancelled", "Cancelled"
        except CompareInputError as exc:
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", exc.code
                job.error = {"code": exc.code, "message": str(exc), "request_id": job.job_id}
        except FileNotFoundError:
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "MODEL_UNAVAILABLE"
                job.error = {
                    "code": "MODEL_UNAVAILABLE",
                    "message": "The selected local feature model or its weights are unavailable.",
                    "request_id": job.job_id,
                }
        except Exception:
            logger.exception("LV React comparison failed (job=%s)", job.job_id)
            with app.state.lock:
                job.state, job.phase, job.message = "failed", "failed", "COMPARE_FAILED"
                job.error = {
                    "code": "COMPARE_FAILED",
                    "message": "The comparison failed. Check the local worker log and retry.",
                    "request_id": job.job_id,
                }

    @app.post("/api/v1/compare/jobs", status_code=202)
    def create_compare_job(request: CreateCompareJob) -> dict:
        job_id = uuid.uuid4().hex
        job = _CompareJob(job_id=job_id, request=request)
        with app.state.lock:
            if app.state.compare_projection_active or any(
                item.state in {"queued", "running", "cancelling"}
                for item in [*app.state.jobs.values(), *app.state.compare_jobs.values()]
            ):
                raise HTTPException(status_code=409, detail={
                    "code": "JOB_ALREADY_RUNNING",
                    "message": "Wait for the active analysis to finish before starting another.",
                    "request_id": job_id,
                })
            app.state.compare_jobs[job_id] = job
        threading.Thread(target=execute_compare, args=(job,), daemon=True,
                         name=f"lv-compare-{job_id[:8]}").start()
        return {"job_id": job_id, "state": job.state}

    @app.get("/api/v1/compare/jobs/{job_id}")
    def get_compare_job(job_id: str) -> dict:
        with app.state.lock:
            job = app.state.compare_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Comparison job not found")
            return _job_view(job)

    @app.delete("/api/v1/compare/jobs/{job_id}", status_code=202)
    def cancel_compare_job(job_id: str) -> dict:
        with app.state.lock:
            job = app.state.compare_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Comparison job not found")
            if job.state not in {"queued", "running", "cancelling"}:
                raise HTTPException(status_code=409, detail="Comparison job is already finished")
            job.cancel_event.set()
            job.state, job.message = "cancelling", "Cancellation requested"
            return {"job_id": job_id, "state": job.state}

    @app.get("/api/v1/compare/runs/{run_id}")
    def get_compare_run(run_id: str) -> dict:
        with app.state.lock:
            run = app.state.compare_runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Comparison run not found")
            public_fields = {
                "folder_a", "folder_b", "model_id", "drift", "records_a", "records_b",
                "warnings", "overlap_count",
            }
            return {key: value for key, value in run.items() if key in public_fields}

    @app.post("/api/v1/compare/runs/{run_id}/projections")
    def create_compare_projection(run_id: str, request: CreateCompareProjection) -> dict:
        with app.state.lock:
            run = app.state.compare_runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Comparison run not found")
            if app.state.compare_projection_active or any(
                item.state in {"queued", "running", "cancelling"}
                for item in [*app.state.jobs.values(), *app.state.compare_jobs.values()]
            ):
                raise HTTPException(status_code=409, detail={
                    "code": "JOB_ALREADY_RUNNING",
                    "message": "Wait for the active local analysis to finish before projecting this run.",
                    "request_id": uuid.uuid4().hex,
                })
            app.state.compare_projection_active = True
            cache_key = (request.method, request.dimension)
            cached = run.get("projections", {}).get(cache_key)
            embeddings_a = run["embeddings_a"]
            embeddings_b = run["embeddings_b"]
        if cached is not None:
            with app.state.lock:
                app.state.compare_projection_active = False
            return cached
        try:
            view = project_comparison(
                embeddings_a,
                embeddings_b,
                request.method,
                request.dimension,
            )
            with app.state.lock:
                run.setdefault("projections", {})[cache_key] = view
            return view
        except CompareInputError as exc:
            raise HTTPException(status_code=422, detail={
                "code": exc.code,
                "message": str(exc),
                "request_id": uuid.uuid4().hex,
            }) from exc
        except (ImportError, ModuleNotFoundError) as exc:
            method_name = request.method.upper()
            raise HTTPException(status_code=422, detail={
                "code": "METHOD_UNAVAILABLE",
                "message": f"{method_name} is unavailable in this local runtime.",
                "request_id": uuid.uuid4().hex,
            }) from exc
        except Exception as exc:
            logger.exception("LV React comparison projection failed (run=%s)", run_id)
            raise HTTPException(status_code=500, detail={
                "code": "PROJECTION_FAILED",
                "message": "The projection failed. Check the local worker log and retry.",
                "request_id": uuid.uuid4().hex,
            }) from exc
        finally:
            with app.state.lock:
                app.state.compare_projection_active = False

    @app.get("/api/v1/compare/runs/{run_id}/images/{group}/{index}")
    def get_compare_image(run_id: str, group: str, index: int) -> Response:
        with app.state.lock:
            run = app.state.compare_runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Comparison run not found")
            paths_key = {"a": "image_paths_a", "b": "image_paths_b"}.get(group)
            if paths_key is None:
                raise HTTPException(status_code=404, detail="Image not found")
            paths: list[Path] = run[paths_key]
            if index < 0 or index >= len(paths):
                raise HTTPException(status_code=404, detail="Image not found")
            path = paths[index]
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise HTTPException(status_code=404, detail="Image is no longer available") from exc
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return Response(content=content, media_type=media_type,
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
            return {
                key: value for key, value in run.items()
                if key != "image_paths" and not key.startswith("_")
            }

    @app.post("/api/v1/runs/{run_id}/rankings")
    def rank_run_images(run_id: str, request: CreateRanking) -> dict:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
        records = run.get("records", [])
        if request.model_id not in run.get("models", []):
            raise HTTPException(status_code=404, detail={
                "code": "MODEL_NOT_IN_RUN",
                "message": "Run the analysis with this feature model before ranking its records.",
                "request_id": uuid.uuid4().hex,
            })

        indices = request.indices
        if indices is not None:
            if any(index < 0 or index >= len(records) for index in indices):
                raise HTTPException(status_code=422, detail={
                    "code": "INVALID_RECORD_INDEX",
                    "message": "A ranking record index is outside this run.",
                    "request_id": uuid.uuid4().hex,
                })
            indices = list(dict.fromkeys(indices))

        criterion = request.criterion
        if indices is None and criterion in {"filename", "spatial"}:
            # The legacy unselected grid always falls back to outlier Top-N.
            criterion = "outlier"
        metric_scores = run.get("_ranking_scores", {}).get(request.model_id, {})
        scores = metric_scores.get(criterion) if criterion in {"outlier", "disagreement"} else None
        if indices is None:
            if scores is None:
                return {
                    "model_id": request.model_id,
                    "criterion": request.criterion,
                    "effective_criterion": criterion,
                    "available": False,
                    "total": len(records),
                    "items": [],
                }
            order = [int(index) for index in np.argsort(np.asarray(scores))[::-1][:50]]
        elif criterion in {"outlier", "disagreement"}:
            if scores is None:
                # Streamlit falls back to spatial order when score arrays do not exist.
                criterion = "spatial"
            else:
                order = sorted(indices, key=lambda index: -float(scores[index]))
        if indices is not None and criterion == "filename":
            order = sorted(indices, key=lambda index: str(records[index].get("filename", "")))
        elif indices is not None and criterion == "spatial":
            from interaction import spatial_order

            view = next((item for item in run.get("views", [])
                         if item.get("model_id") == request.model_id
                         and item.get("method") == request.method), None)
            if view is None or len(view.get("points", [])) != len(records):
                raise HTTPException(status_code=404, detail={
                    "code": "VIEW_NOT_IN_RUN",
                    "message": "The selected projection is unavailable for this ranking.",
                    "request_id": uuid.uuid4().hex,
                })
            coords = np.asarray([[point["x"], point["y"]] for point in view["points"]], dtype=float)
            order = spatial_order(coords, indices)

        return {
            "model_id": request.model_id,
            "criterion": request.criterion,
            "effective_criterion": criterion,
            "available": True,
            "total": len(records) if indices is None else len(indices),
            "items": [
                {"index": index, "score": float(scores[index]) if scores is not None else None}
                for index in order
            ],
        }

    @app.post("/api/v1/runs/{run_id}/similar")
    def search_similar(run_id: str, request: CreateSimilarSearch) -> dict:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
        return _similar_search(run, request.model_id, request.query_index, request.k)

    @app.get("/api/v1/runs/{run_id}/similar.csv")
    def export_similar_csv(run_id: str, model_id: str, query_index: int, k: int = 9) -> Response:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
        result = _similar_search(run, model_id, query_index, k)
        ordered_indices = [query_index, *[row["index"] for row in result["results"]]]
        records_with_paths = [
            {**record, "path": run["image_paths"][index]}
            for index, record in enumerate(run["records"])
        ]
        from interaction import records_to_csv

        content = records_to_csv(records_with_paths, ordered_indices)
        return Response(
            content=content,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="similar_group.csv"'},
        )

    def get_text_encoder(model_id: str) -> Callable[[str], np.ndarray]:
        with app.state.lock:
            encoder = app.state.text_encoders.get(model_id)
        if encoder is not None:
            return encoder
        try:
            encoder = text_loader(model_id)
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail={
                "code": "TEXT_MODEL_UNAVAILABLE",
                "message": "The selected local text model is unavailable.",
                "request_id": uuid.uuid4().hex,
            }) from None
        except Exception:
            logger.exception("LV React could not load text model (model=%s)", model_id)
            raise HTTPException(status_code=503, detail={
                "code": "TEXT_MODEL_UNAVAILABLE",
                "message": "The selected local text model could not be loaded. Check the worker log.",
                "request_id": uuid.uuid4().hex,
            }) from None
        with app.state.lock:
            return app.state.text_encoders.setdefault(model_id, encoder)

    def encode_text_query(model_id: str, query: str) -> np.ndarray:
        with app.state.lock:
            cached = app.state.text_vectors.get(model_id)
            if cached is not None and cached[0] == query:
                return cached[1]
        vector = np.asarray(get_text_encoder(model_id)(query), dtype=np.float32).reshape(-1)
        with app.state.lock:
            app.state.text_vectors[model_id] = (query, vector)
        return vector

    @app.post("/api/v1/runs/{run_id}/text-search")
    def search_by_text(run_id: str, request: CreateTextSearch) -> dict:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
        return _text_search(
            run, request.model_id, request.query, request.k,
            lambda query: encode_text_query(request.model_id, query),
        )

    @app.get("/api/v1/runs/{run_id}/text-search.csv")
    def export_text_search_csv(run_id: str, model_id: str, query: str, k: int = 9) -> Response:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
        result = _text_search(
            run, model_id, query, k,
            lambda text: encode_text_query(model_id, text),
        )
        ordered_indices = [row["index"] for row in result["results"]]
        records_with_paths = [
            {**record, "path": run["image_paths"][index]}
            for index, record in enumerate(run["records"])
        ]
        from interaction import records_to_csv

        content = records_to_csv(records_with_paths, ordered_indices)
        return Response(
            content=content,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="text_search.csv"'},
        )

    @app.post("/api/v1/runs/{run_id}/duplicates")
    def scan_duplicates(run_id: str, request: CreateDuplicateScan) -> dict:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
        return _duplicate_scan(run, request)

    @app.post("/api/v1/runs/{run_id}/blocks")
    def create_run_blocks(run_id: str, request: CreateRunBlocks) -> dict:
        if len(set(request.indices)) != len(request.indices):
            raise HTTPException(status_code=422, detail="Point indices must be unique")
        with app.state.lock:
            run = app.state.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
            view = next((item for item in run["views"]
                         if item["model_id"] == request.model_id
                         and item["method"] == request.method), None)
            if view is None:
                raise HTTPException(status_code=404, detail="Projection view not found")
            by_index = {int(point["index"]): point for point in view["points"]}
            if any(index not in by_index for index in request.indices):
                raise HTTPException(status_code=422, detail="Point index is outside this projection")
            coordinates = np.asarray([
                [by_index[index]["x"], by_index[index]["y"], by_index[index]["z"]]
                for index in request.indices
            ], dtype=np.float64)

        from sklearn.cluster import KMeans

        cluster_count = min(request.k, len(request.indices))
        fitted = KMeans(n_clusters=cluster_count, n_init=4, random_state=42).fit(coordinates)
        block_indices = [
            [request.indices[row] for row, label in enumerate(fitted.labels_) if label == block_id]
            for block_id in range(cluster_count)
        ]
        return {
            "model_id": request.model_id,
            "method": request.method,
            "requested_k": request.k,
            "effective_k": cluster_count,
            "blocks": [
                {"block_id": block_id, "count": len(indices), "indices": indices}
                for block_id, indices in enumerate(block_indices) if indices
            ],
            "centers": [
                {"block_id": block_id, "x": float(center[0]),
                 "y": float(center[1]), "z": float(center[2])}
                for block_id, center in enumerate(fitted.cluster_centers_)
            ],
        }

    @app.post("/api/v1/runs/{run_id}/block-projection")
    def create_block_projection(run_id: str, request: CreateBlockProjection) -> dict:
        if len(set(request.indices)) != len(request.indices):
            raise HTTPException(status_code=422, detail="Point indices must be unique")
        requested_set = set(request.indices)
        with app.state.lock:
            run = app.state.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
            view = next((item for item in run["views"]
                         if item["model_id"] == request.model_id
                         and item["method"] == request.method), None)
            if view is None:
                raise HTTPException(status_code=404, detail="Projection view not found")
            by_index = {int(point["index"]): point for point in view["points"]}
            if any(index not in by_index for index in request.indices):
                raise HTTPException(status_code=422, detail="Point index is outside this projection")
            view_points = [by_index[index] for index in request.indices]
            sources = run.get("_embedding_sources", {}).get(request.model_id, [])

        projection_source = "projection"
        coordinates = np.asarray([[point["x"], point["y"]] for point in view_points], dtype=np.float64)
        if len(request.indices) > 2:
            rows_by_index: dict[int, np.ndarray] = {}
            cache_error = False
            for source in sources:
                source_indices = source.get("indices", [])
                source_keys = source.get("keys", [])
                needed = [(int(index), str(key)) for index, key in zip(source_indices, source_keys)
                          if int(index) in requested_set]
                if not needed:
                    continue
                try:
                    with np.load(str(source["cache_path"]), allow_pickle=False) as cache:
                        cache_keys = [str(key) for key in cache["keys"].tolist()]
                        cache_rows = {key: row for row, key in enumerate(cache_keys)}
                        embeddings = cache["embeddings"]
                        for index, key in needed:
                            if key not in cache_rows:
                                cache_error = True
                                break
                            rows_by_index[index] = embeddings[cache_rows[key]]
                except (OSError, KeyError, ValueError):
                    cache_error = True
                if cache_error:
                    break
            if not cache_error and all(index in rows_by_index for index in request.indices):
                from sklearn.decomposition import PCA

                raw = np.stack([rows_by_index[index] for index in request.indices])
                coordinates = PCA(n_components=2, random_state=42).fit_transform(raw)
                projection_source = "embedding_pca"
            else:
                raise HTTPException(
                    status_code=409,
                    detail="Embedding cache is unavailable for this block. Run the analysis again.",
                )
        return {
            "model_id": request.model_id,
            "method": request.method,
            "projection_source": projection_source,
            "points": [
                {"index": index, "x": float(point[0]), "y": float(point[1])}
                for index, point in zip(request.indices, coordinates)
            ],
        }

    @app.get("/api/v1/runs/{run_id}/images/{index}")
    def get_run_image(
        run_id: str,
        index: int,
        overlay: bool = False,
        view: Literal["selected", "source"] = "selected",
    ) -> Response:
        with app.state.lock:
            run = app.state.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
            paths: list[Path] = run["image_paths"]
            if index < 0 or index >= len(paths):
                raise HTTPException(status_code=404, detail="Image not found")
            path = paths[index]
            is_object = run.get("source_granularity") == "object"
            object_record = run["records"][index] if is_object else None
        try:
            if is_object and object_record is not None and view == "selected":
                from io import BytesIO

                from interaction import crop_bbox
                from object_eval import _adaptive_pad_px
                from safe_io import safe_open_image

                image = safe_open_image(path)
                if image is None:
                    raise OSError("The source image is unreadable")
                bbox = object_record["bbox"]
                policy = run["_object_policy"]
                crop = crop_bbox(
                    image,
                    *bbox,
                    pad_px=_adaptive_pad_px(bbox, *image.size, policy["pad"]),
                )
                output = BytesIO()
                crop.convert("RGB").save(output, format="JPEG", quality=90)
                content = output.getvalue()
                media_type = "image/jpeg"
            elif overlay and run.get("source_mode") == "object_detector" and run.get("class_names"):
                from io import BytesIO

                from interaction import draw_yolo_boxes, yolo_label_path_for

                rendered = draw_yolo_boxes(path, yolo_label_path_for(path), run["class_names"])
                if rendered is None:
                    raise OSError("The source image is unreadable")
                output = BytesIO()
                rendered.convert("RGB").save(output, format="JPEG", quality=92)
                content = output.getvalue()
                media_type = "image/jpeg"
            else:
                content = path.read_bytes()
                media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        except OSError as exc:
            raise HTTPException(status_code=404, detail="Image is no longer available") from exc
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
