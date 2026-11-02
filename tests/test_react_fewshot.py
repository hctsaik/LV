from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from react_web_shell import create_app


def _samples(root: Path, *, with_labels: bool) -> Path:
    images = root / "images"
    images.mkdir(parents=True)
    if with_labels:
        labels = root / "labels"
        labels.mkdir()
        (root / "classes.txt").write_text("scratch\nstain\n", encoding="utf-8")
    for index, color in enumerate(((220, 30, 40), (20, 180, 80))):
        Image.new("RGB", (48, 40), color).save(images / f"sample-{index}.png")
    if with_labels:
        (root / "labels" / "sample-0.txt").write_text(
            "0 0.5 0.5 0.5 0.5\n1 0.25 0.25 0.25 0.25\n", encoding="utf-8",
        )
        (root / "labels" / "sample-1.txt").write_text(
            "0 0.5 0.5 0.75 0.75\n", encoding="utf-8",
        )
    return root


def _embed(crop: Image.Image) -> np.ndarray:
    pixels = np.asarray(crop.convert("RGB"), dtype=np.float32)
    return np.asarray([
        pixels[..., 0].mean() / 255.0,
        pixels[..., 1].mean() / 255.0,
        pixels[..., 2].mean() / 255.0,
        crop.width / 48.0,
    ], dtype=np.float32)


def _client(monkeypatch, tmp_path: Path, *, model_loader=None) -> TestClient:
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    return TestClient(create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=model_loader or (lambda _model: _embed),
        anomaly_model_builder=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sample bank creation must not build an anomaly model")
        ),
    ))


def _wait_job(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.get(f"/api/v1/fewshot/jobs/{job_id}").json()
        if result["state"] not in {"queued", "running", "cancelling"}:
            return result
        time.sleep(0.02)
    raise AssertionError("few-shot sample-bank job did not finish")


def test_fewshot_builds_yolo_bank_without_anomaly_model_and_reopens_it(
    monkeypatch, tmp_path: Path,
):
    samples = _samples(tmp_path / "yolo-samples", with_labels=True)
    source_hashes = {path: path.read_bytes() for path in samples.rglob("*") if path.is_file()}
    with _client(monkeypatch, tmp_path) as worker:
        created = worker.post("/api/v1/fewshot/sample-banks/jobs", json={
            "sample_folders": [str(samples)],
            "model_id": "fixture-model",
            "object_source": "yolo",
        })
        assert created.status_code == 202, created.text
        job = _wait_job(worker, created.json()["job_id"])
        assert job["state"] == "completed", job
        summary = job["result"]
        assert summary["object_source"] == "yolo"
        assert summary["sample_count"] == 3
        assert summary["class_counts"] == {"scratch": 2, "stain": 1}
        assert summary["feature_dimension"] == 4
        assert str(tmp_path) not in json.dumps(job)
        assert worker.get("/api/v1/anomaly/models").json()["items"] == []

        bank_dir = tmp_path / "cache" / "fewshot_sample_banks" / summary["sample_bank_id"]
        assert {path.name for path in bank_dir.iterdir()} == {"emb.npz", "meta.json"}

    # The ID resolves from the shared cache after a separate worker starts.
    with _client(monkeypatch, tmp_path) as reopened_worker:
        reopened = reopened_worker.get(
            f"/api/v1/fewshot/sample-banks/{summary['sample_bank_id']}"
        )
        assert reopened.status_code == 200, reopened.text
        assert reopened.json()["sample_bank"] == summary
        assert str(tmp_path) not in reopened.text

    assert {path: path.read_bytes() for path in source_hashes} == source_hashes


def test_fewshot_whole_image_mode_needs_no_yolo_labels(monkeypatch, tmp_path: Path):
    samples = _samples(tmp_path / "whole-image-samples", with_labels=False)
    with _client(monkeypatch, tmp_path) as worker:
        created = worker.post("/api/v1/fewshot/sample-banks/jobs", json={
            "sample_folders": [str(samples)],
            "model_id": "fixture-model",
            "object_source": "whole_image",
        })
        assert created.status_code == 202, created.text
        job = _wait_job(worker, created.json()["job_id"])
        assert job["state"] == "completed", job
        summary = job["result"]
        assert summary["object_source"] == "whole_image"
        assert summary["sample_count"] == 2
        assert summary["class_counts"] == {"image": 2}

        bank_dir = tmp_path / "cache" / "fewshot_sample_banks" / summary["sample_bank_id"]
        meta = json.loads((bank_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["object_source"] == "whole_image"
        assert [item["bbox"] for item in meta["provenance"]] == [
            [0.5, 0.5, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0],
        ]


def test_fewshot_rejects_uninstalled_model_and_invalid_bank_ids(monkeypatch, tmp_path: Path):
    samples = _samples(tmp_path / "samples", with_labels=True)
    with _client(monkeypatch, tmp_path) as worker:
        rejected = worker.post("/api/v1/fewshot/sample-banks/jobs", json={
            "sample_folders": [str(samples)],
            "model_id": "not-installed",
            "object_source": "yolo",
        })
        assert rejected.status_code == 422
        assert worker.get("/api/v1/fewshot/sample-banks/invalid").status_code == 404


def test_fewshot_yolo_without_objects_fails_before_loading_model(monkeypatch, tmp_path: Path):
    samples = _samples(tmp_path / "unlabeled", with_labels=False)

    def unexpected_model_load(_model):
        raise AssertionError("an empty YOLO sample set must be rejected before loading weights")

    with _client(monkeypatch, tmp_path, model_loader=unexpected_model_load) as worker:
        created = worker.post("/api/v1/fewshot/sample-banks/jobs", json={
            "sample_folders": [str(samples)],
            "model_id": "fixture-model",
            "object_source": "yolo",
        })
        assert created.status_code == 202, created.text
        job = _wait_job(worker, created.json()["job_id"])
        assert job["state"] == "failed"
        assert job["error"]["code"] == "INVALID_SAMPLE_DATA"
        assert not (tmp_path / "cache" / "fewshot_sample_banks").exists()
