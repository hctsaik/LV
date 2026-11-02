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


def make_app(monkeypatch, tmp_path: Path, *, anomaly_scorer=None) -> TestClient:
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "cases"))
    return TestClient(create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: lambda _path: np.zeros(2, dtype=np.float32),
        anomaly_model_builder=fixture_builder,
        anomaly_scorer=anomaly_scorer,
    ))


def wait_for_score_job(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.get(f"/api/v1/anomaly/score-jobs/{job_id}").json()
        if result["state"] not in {"queued", "running", "cancelling"}:
            return result
        time.sleep(0.02)
    raise AssertionError("anomaly scoring job did not finish")


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


def test_anomaly_score_rank_confirm_reapply_and_heatmap(monkeypatch, tmp_path: Path):
    from PIL import Image

    image_path = tmp_path / "target.png"
    Image.new("RGB", (96, 64), (120, 140, 160)).save(image_path)

    def scorer(request, bundle, *, progress=None):
        if progress:
            progress(90, 100, "Fixture target scores ready")
        records = [
            {"image_path": str(image_path), "bbox": [0.1, 0.1, 0.8, 0.8], "label": "part", "split": "test", "obj_index": index}
            for index in range(2)
        ]
        return {
            "records": records,
            "scores": np.asarray([0.9, 0.1], dtype=np.float64),
            "labels": np.asarray(["bad", "good"], dtype=object),
            "ranking": [0, 1],
            "threshold": 0.5,
            "n_bad": 1,
            "auroc": None,
            "score_mode": "patch",
            "model_id": bundle["meta"]["model"],
            "model_meta": bundle["meta"],
            "model_bank_sha256": "a" * 64,
            "object_source": "yolo",
            "bank": object(),
            "skipped_count": 0,
            "fewshot_aligned": {},
            "fewshot_unmatched": 0,
            "confirmed": {},
            "contamination": float(request["contamination"]),
            "created_at": 1_798_000_000.0,
        }

    import anomaly_score
    import patch_features

    monkeypatch.setattr(
        patch_features,
        "extract_patch_grid",
        lambda _crop, **_kwargs: {"feats": np.ones((2, 3), dtype=np.float32), "grid": (1, 2)},
    )
    monkeypatch.setattr(
        anomaly_score,
        "score_object",
        lambda _feats, _grid, _bank: (0.9, np.asarray([[0.1, 0.9]], dtype=np.float32)),
    )

    with make_app(monkeypatch, tmp_path, anomaly_scorer=scorer) as client:
        created = client.post("/api/v1/anomaly/models/jobs", json={
            "train_folders": [str(tmp_path)],
            "model_id": "fixture-model",
            "score_mode": "patch",
            "object_source": "yolo",
        })
        assert created.status_code == 202, created.text
        model_job = wait_for_job(client, created.json()["job_id"])
        assert model_job["state"] == "completed", model_job

        scored = client.post("/api/v1/anomaly/score-jobs", json={
            "target_folders": [str(tmp_path)],
            "contamination": 0.05,
        })
        assert scored.status_code == 202, scored.text
        score_job = wait_for_score_job(client, scored.json()["job_id"])
        assert score_job["state"] == "completed", score_job
        run = score_job["result"]
        assert run["n_objects"] == 2
        assert run["n_bad"] == 1
        assert str(image_path) not in str(score_job)
        run_id = run["run_id"]

        page = client.get(f"/api/v1/anomaly/runs/{run_id}/objects?verdict=bad&limit=1")
        assert page.status_code == 200, page.text
        assert page.json()["total"] == 1
        assert page.json()["items"][0]["filename"] == "target.png"
        assert str(image_path) not in page.text
        preview = client.get(f"/api/v1/anomaly/runs/{run_id}/objects/0/preview.jpg")
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/jpeg"
        heatmap = client.get(f"/api/v1/anomaly/runs/{run_id}/objects/0/heatmap.png")
        assert heatmap.status_code == 200
        assert heatmap.headers["content-type"] == "image/png"

        reapply = client.post(f"/api/v1/anomaly/runs/{run_id}/reapply", json={
            "expected_revision": 1,
            "contamination": 0.05,
            "confirmed": {"0": "good", "1": "bad"},
        })
        assert reapply.status_code == 200, reapply.text
        assert reapply.json()["revision"] == 2
        assert reapply.json()["confirmed_count"] == 2
        assert reapply.json()["n_bad"] == 2
        assert reapply.json()["auroc"] == 0.0
        assert reapply.json()["model_bank_sha256"] == run["model_bank_sha256"]
        stale = client.post(f"/api/v1/anomaly/runs/{run_id}/reapply", json={
            "expected_revision": 1,
            "contamination": 0.05,
            "confirmed": {},
        })
        assert stale.status_code == 409


def test_score_targets_uses_locked_object_model_contract(monkeypatch, tmp_path: Path):
    import anomaly_tool
    import object_eval
    from react_anomaly import score_targets

    target = tmp_path / "targets"
    target.mkdir()
    image_path = target / "sample.png"
    image_path.touch()
    captured = {}

    def run_pipeline(image_paths, class_names, **kwargs):
        captured.update(kwargs)
        assert image_paths == [image_path]
        assert class_names == ["part"]
        return {
            "records": [{
                "image_path": str(image_path), "bbox": (0.1, 0.2, 0.5, 0.6),
                "label": "part", "split": "", "obj_index": 0,
            }],
            "scores": np.asarray([0.25], dtype=np.float32),
            "ranking": [0], "threshold": 0.25, "n_bad": 0,
            "auroc": None, "bank": None, "skipped": [],
        }

    monkeypatch.setattr(object_eval, "list_images", lambda _root: [image_path])
    monkeypatch.setattr(object_eval, "classes_for", lambda _root: ["part"])
    monkeypatch.setattr(anomaly_tool, "run_pipeline", run_pipeline)
    ref = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    bundle = {
        "meta": {
            "model": "locked-model", "target_res": 448, "score_mode": "object",
            "object_source": "whole_image", "obj_dim": 2,
            "embedding_recipe": {"model_id": "locked-model", "pad": 0.12},
        },
        "bank_vectors": None,
        "projection": {"good_obj_emb": ref},
        "fewshot": [],
    }
    run = score_targets({"target_folders": [str(target)], "contamination": 0.1}, bundle)
    assert captured["score_mode"] == "object"
    assert captured["model"] == "locked-model"
    assert captured["target_res"] == 448
    assert captured["object_source"] == "whole_image"
    assert captured["external_bank"] is None
    np.testing.assert_array_equal(captured["external_ref"], ref)
    assert run["score_mode"] == "object"
    assert run["model_bank_sha256"]
    assert run["records"][0]["image_path"] == str(image_path)
