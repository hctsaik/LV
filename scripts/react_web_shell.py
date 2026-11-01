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

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, Response
from starlette.staticfiles import StaticFiles

from _utils import available_models, load_model
from react_compare import ComparisonInputError, project_comparison, run_comparison
from react_visualize import JobCancelled, VisualizationInputError, run_image_visualization

API_VERSION = "1"
logger = logging.getLogger("lv.react_web")


class CreateVisualizationJob(BaseModel):
    folders: list[str] = Field(min_length=1, max_length=32)
    mode: Literal["classifier", "detector"] = "classifier"
    class_names: list[str] = Field(default_factory=list, max_length=256)
    model_id: str | None = Field(default=None, min_length=1, max_length=128)
    method: str = "pca"
    model_ids: list[str] = Field(default_factory=list, max_length=8)
    methods: list[str] = Field(default_factory=list, max_length=5)


class ComparisonObjectPolicy(BaseModel):
    pad: float = Field(ge=0.0, le=0.5)
    target_res: Literal[168, 196, 224, 280, 336, 392, 448]
    head: Literal["cls", "meanpool"]


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
    object_model_loader: Callable | None = None,
) -> FastAPI:
    catalog = model_catalog or available_models
    loader = model_loader or load_model
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
                class_names=job.request.class_names,
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
            return {key: value for key, value in run.items() if key != "image_paths"}

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
