from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from react_web_shell import create_app


def fixture_builder(request: dict, *, progress=None) -> dict:
    if progress:
        progress(100, 100, "Fixture Normal Bank ready")
    return {
        "meta": {
            "model": request["model_id"],
            "target_res": 224,
            "score_mode": "patch",
            "object_source": "yolo",
            "patch_dim": 3,
            "obj_dim": 2,
            "n_objects": 5,
            "label_semantic": "object",
            "embedding_recipe": {"model_id": request["model_id"], "checkpoint_id": "fixture-sha"},
        },
        "bank_vectors": np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        "projection": {
            "mean": np.asarray([0, 0], dtype=np.float32),
            "components": np.asarray([[1, 0], [0, 1]], dtype=np.float32),
            "ref_coords": np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
            "good_obj_emb": np.asarray([[1, 0], [0, 1]], dtype=np.float32),
            "tau": 0.25,
            "l2norm": False,
        },
        "fewshot": [{"stem": "sample", "bbox": [0.1, 0.1, 0.8, 0.8], "label": "good", "verdict": "good"}],
        "head": None,
        "built_at": 1_798_000_000.0,
        "summary": {
            "n_images": 5,
            "n_objects": 5,
            "has_normal_bank": True,
            "has_head": False,
            "head_status": "只做異常偵測(不分種類)",
            "eligible_classes": [],
        },
    }


def wait_for_job(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.get(f"/api/v1/anomaly/jobs/{job_id}").json()
        if result["state"] not in {"queued", "running", "cancelling"}:
            return result
        time.sleep(0.02)
    raise AssertionError("anomaly model job did not finish")


def make_app(monkeypatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "cases"))
    return TestClient(create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: lambda _path: np.zeros(2, dtype=np.float32),
        anomaly_model_builder=fixture_builder,
    ))


def test_anomaly_bank_build_save_and_fresh_worker_load(monkeypatch, tmp_path: Path):
    train = tmp_path / "training"
    train.mkdir()
    with make_app(monkeypatch, tmp_path) as first_worker:
        created = first_worker.post("/api/v1/anomaly/models/jobs", json={
            "train_folders": [str(train)],
            "model_id": "fixture-model",
            "score_mode": "patch",
            "target_res": 224,
            "sample_n": 64,
            "label_semantic": "object",
            "n_min": 8,
            "object_source": "yolo",
        })
        assert created.status_code == 202, created.text
        job = wait_for_job(first_worker, created.json()["job_id"])
        assert job["state"] == "completed", job
        built = job["result"]
        assert built["summary"]["has_normal_bank"] is True
        assert str(tmp_path) not in str(job)
        model_id = built["model_id"]

        saved = first_worker.post(f"/api/v1/anomaly/models/{model_id}/save")
        assert saved.status_code == 200, saved.text
        assert saved.json()["model"]["saved"] is True
        model_dir = tmp_path / "cache" / "react_anomaly_models" / model_id
        assert {"bank.npz", "projection.npz", "fewshot.json", "meta.json", "manifest.json"} <= {
            path.name for path in model_dir.iterdir()
        }

    # A new FastAPI worker has no in-memory model arrays; the saved package is enough to reopen it.
    with make_app(monkeypatch, tmp_path) as reopened_worker:
        listed = reopened_worker.get("/api/v1/anomaly/models").json()["items"]
        assert [item["model_id"] for item in listed] == [model_id]
        assert listed[0]["active"] is False
        loaded = reopened_worker.post("/api/v1/anomaly/models/load", json={"model_id": model_id})
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()["model"]["source"] == "loaded"
        assert loaded.json()["model"]["meta"] == built["meta"]
        assert reopened_worker.get("/api/v1/anomaly/models").json()["items"][0]["active"] is True

    (model_dir / "bank.npz").unlink()
    with make_app(monkeypatch, tmp_path) as damaged_worker:
        rejected = damaged_worker.post("/api/v1/anomaly/models/load", json={"model_id": model_id})
        assert rejected.status_code == 422
        assert "不完整" in rejected.json()["message"]


def test_anomaly_model_rejects_unknown_model_without_creating_job(monkeypatch, tmp_path: Path):
    with make_app(monkeypatch, tmp_path) as client:
        response = client.post("/api/v1/anomaly/models/jobs", json={
            "train_folders": [str(tmp_path)],
            "model_id": "not-installed",
        })
        assert response.status_code == 422
        assert client.get("/api/v1/anomaly/models").json()["items"] == []
