from __future__ import annotations

import hashlib
import csv
import io
import json
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from class_similarity import analyze, build_box_summary
from react_web_shell import create_app


def test_topk_same_rival_margin_matches_hand_calculated_fixture():
    embeddings = np.asarray([
        [1.0, 0.0],
        [0.8, 0.6],
        [0.0, 1.0],
        [0.6, 0.8],
        [-1.0, 0.0],
    ])
    labels = ["A", "A", "B", "B", "C"]
    ids = ["a1", "a2", "b1", "b2", "c1"]
    result = analyze(
        embeddings,
        labels,
        ids,
        eligibility=[True] * len(ids),
        invalid_reasons=[[] for _ in ids],
        k=1,
        query_block=2,
        reference_block=3,
    )

    by_id = {item.record_id: item for item in result.items}
    assert np.isclose(by_id["a1"].same, 0.8)
    assert np.isclose(by_id["a1"].other, 0.6)
    assert np.isclose(by_id["a1"].margin, 0.2)
    assert np.isclose(by_id["a2"].same, 0.8)
    assert np.isclose(by_id["a2"].other, 0.96)
    assert np.isclose(by_id["a2"].margin, -0.16)
    assert by_id["c1"].same is None
    assert by_id["c1"].margin is None
    assert result.max_block_shape[0] <= 2
    assert result.max_block_shape[1] <= 3

    summary = {item["label"]: item for item in build_box_summary(result.items, "margin")}
    assert summary["A"]["n_scored"] == 2
    assert summary["C"]["n_unscored"] == 1


def _fixture(root: Path) -> tuple[list[Path], dict[str, str]]:
    specs = [
        ("train", "apple", "a1.png", (220, 35, 45)),
        ("train", "apple", "a2.png", (200, 55, 50)),
        ("train", "banana", "b1.png", (35, 65, 215)),
        ("test", "banana", "b2.png", (50, 80, 200)),
    ]
    folders = {name: root / name for name in ("train", "test")}
    hashes: dict[str, str] = {}
    for split, label, filename, color in specs:
        path = folders[split] / label / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color).save(path)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return list(folders.values()), hashes


def _embedder(path: Path) -> np.ndarray:
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    vector = np.zeros(384, dtype=np.float32)
    vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
    vector[3] = len(path.name) / 10.0
    return vector


def _wait_for_job(client: TestClient, path: str, job_id: str) -> dict:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        response = client.get(path.format(job_id=job_id))
        assert response.status_code == 200, response.text
        body = response.json()
        if body["state"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.02)
    raise AssertionError("job did not reach a terminal state")


def test_class_similarity_api_uses_current_raw_embeddings_and_keeps_paths_private(
    tmp_path: Path, monkeypatch,
):
    folders, source_hashes = _fixture(tmp_path / "dataset")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "cases"))
    app = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: _embedder,
    )

    with TestClient(app) as client:
        created = client.post("/api/v1/visualize/jobs", json={
            "folders": [str(folder) for folder in folders],
            "model_id": "fixture-model",
            "method": "pca",
        })
        assert created.status_code == 202, created.text
        run_job = _wait_for_job(client, "/api/v1/jobs/{job_id}", created.json()["job_id"])
        assert run_job["state"] == "completed", run_job
        run_id = run_job["run_id"]

        started = client.post(f"/api/v1/runs/{run_id}/class-similarity/jobs", json={
            "model_id": "fixture-model",
            "k": 1,
            "analysis_split": "all",
        })
        assert started.status_code == 202, started.text
        analysis = _wait_for_job(
            client,
            "/api/v1/class-similarity/jobs/{job_id}",
            started.json()["job_id"],
        )
        assert analysis["state"] == "completed", analysis
        result = analysis["result"]
        assert result["summary"] == {
            "input_count": 4,
            "valid_count": 4,
            "excluded_count": 0,
            "per_class_counts": {"apple": 2, "banana": 2},
        }
        assert len(result["items"]) == 4
        assert all(item["same_neighbors"] for item in result["items"])
        assert all(item["rival_neighbors"] for item in result["items"])
        assert str(tmp_path) not in analysis.__repr__()
        assert {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in source_hashes} == source_hashes


def test_class_similarity_evaluation_template_review_and_scoped_exports(tmp_path: Path, monkeypatch):
    folders, source_hashes = _fixture(tmp_path / "evaluation-dataset")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "cases"))
    app = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: _embedder,
    )
    with TestClient(app) as client:
        created = client.post("/api/v1/visualize/jobs", json={
            "folders": [str(folder) for folder in folders],
            "model_id": "fixture-model", "method": "pca",
        })
        run_job = _wait_for_job(client, "/api/v1/jobs/{job_id}", created.json()["job_id"])
        assert run_job["state"] == "completed", run_job
        run_id = run_job["run_id"]
        started = client.post(f"/api/v1/runs/{run_id}/class-similarity/jobs", json={
            "model_id": "fixture-model", "k": 1,
        })
        analysis = _wait_for_job(
            client, "/api/v1/class-similarity/jobs/{job_id}", started.json()["job_id"],
        )
        assert analysis["state"] == "completed", analysis
        job_id = started.json()["job_id"]

        template_response = client.post(
            f"/api/v1/class-similarity/jobs/{job_id}/evaluation/template",
            json={"scope": "all", "selected_ids": []},
        )
        assert template_response.status_code == 200, template_response.text
        template = json.loads(template_response.content)
        assert len(template["frozen_ids"]) == 4
        assert all(review["verdict"] == "unreviewed" for review in template["reviews"])
        for index, review in enumerate(template["reviews"]):
            review["verdict"] = "incorrect" if index == 0 else "correct"

        policy = {"mode": "threshold", "threshold": 2.0}
        imported = client.put(
            f"/api/v1/class-similarity/jobs/{job_id}/evaluation",
            json={"manifest": template, "policy": policy},
        )
        assert imported.status_code == 200, imported.text
        payload = imported.json()
        assert payload["evaluation"]["pending_n"] == 0
        assert payload["counts"]["status"] == "ready"
        assert payload["counts"]["coverage"] == 1.0
        assert payload["counts"]["precision"] is not None
        assert payload["counts"]["recall"] is not None
        assert payload["precision_recall_curve"]

        preview = client.post(
            f"/api/v1/class-similarity/jobs/{job_id}/evaluation/preview",
            json={"policy": {"mode": "none"}},
        )
        assert preview.status_code == 200, preview.text
        assert preview.json()["summary"]["candidate_n"] == 0
        assert preview.json()["counts"]["status"] == "ready"

        csv_response = client.post(
            f"/api/v1/class-similarity/jobs/{job_id}/report",
            json={"scope": "candidates", "selected_ids": [], "policy": policy, "format": "csv"},
        )
        assert csv_response.status_code == 200, csv_response.text
        csv_rows = list(csv.DictReader(io.StringIO(csv_response.content.decode("utf-8-sig"))))
        assert len(csv_rows) == 4
        assert all(row["is_candidate"] == "1" for row in csv_rows)

        json_response = client.post(
            f"/api/v1/class-similarity/jobs/{job_id}/report",
            json={"scope": "selected", "selected_ids": [template["frozen_ids"][0]], "policy": policy, "format": "json"},
        )
        assert json_response.status_code == 200, json_response.text
        report = json.loads(json_response.content)
        assert report["scope"] == "selected"
        assert report["evaluation"]["status"] == "ready"
        assert len(report["items"]) == 1

        source = folders[0] / "apple" / "a1.png"
        Image.new("RGB", (32, 32), (1, 2, 3)).save(source)
        stale_export = client.post(
            f"/api/v1/class-similarity/jobs/{job_id}/report",
            json={"scope": "all", "selected_ids": [], "policy": policy, "format": "json"},
        )
        assert stale_export.status_code == 409
        assert "source changed" in stale_export.json()["message"]

    assert any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
               for path, digest in source_hashes.items())


def test_class_similarity_saved_operating_point_requires_disjoint_heldout_review(
    tmp_path: Path, monkeypatch,
):
    folders, _source_hashes = _fixture(tmp_path / "heldout-dataset")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "cases"))
    app = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: _embedder,
    )
    with TestClient(app) as client:
        created = client.post("/api/v1/visualize/jobs", json={
            "folders": [str(folder) for folder in folders],
            "model_id": "fixture-model", "method": "pca",
        })
        run_job = _wait_for_job(client, "/api/v1/jobs/{job_id}", created.json()["job_id"])
        assert run_job["state"] == "completed", run_job
        run_id = run_job["run_id"]
        started = client.post(f"/api/v1/runs/{run_id}/class-similarity/jobs", json={
            "model_id": "fixture-model", "k": 1,
        })
        analysis = _wait_for_job(
            client, "/api/v1/class-similarity/jobs/{job_id}", started.json()["job_id"],
        )
        assert analysis["state"] == "completed", analysis
        job_id = started.json()["job_id"]
        record_ids = [item["record_id"] for item in analysis["result"]["items"]]
        tuning_ids, heldout_ids = record_ids[:2], record_ids[2:]

        tuning_template_response = client.post(
            f"/api/v1/class-similarity/jobs/{job_id}/evaluation/template",
            json={"scope": "selected", "purpose": "tuning", "selected_ids": tuning_ids},
        )
        assert tuning_template_response.status_code == 200, tuning_template_response.text
        tuning_template = json.loads(tuning_template_response.content)
        for index, review in enumerate(tuning_template["reviews"]):
            review["verdict"] = "incorrect" if index == 0 else "correct"
        tuning = client.put(
            f"/api/v1/class-similarity/jobs/{job_id}/evaluation",
            json={"manifest": tuning_template, "policy": {"mode": "threshold", "threshold": 0.5}},
        )
        assert tuning.status_code == 200, tuning.text
        assert tuning.json()["evaluation"]["purpose"] == "tuning"
        assert tuning.json()["counts"]["status"] == "ready"

        saved = client.post(
            f"/api/v1/class-similarity/jobs/{job_id}/operating-point",
            json={"policy": {"mode": "threshold", "threshold": 0.5}},
        )
        assert saved.status_code == 200, saved.text
        frozen = saved.json()["frozen_policy"]
        assert frozen["policy"]["threshold"] == 0.5
        assert set(frozen["tuning_ids"]) == set(tuning_ids)

        heldout_template_response = client.post(
            f"/api/v1/class-similarity/jobs/{job_id}/evaluation/template",
            json={"scope": "selected", "purpose": "heldout", "selected_ids": heldout_ids},
        )
        assert heldout_template_response.status_code == 200, heldout_template_response.text
        heldout_template = json.loads(heldout_template_response.content)
        assert heldout_template["purpose"] == "heldout"
        heldout_review_seed = heldout_template["reviews"]
        heldout_template["frozen_ids"] = [tuning_ids[0]]
        heldout_template["reviews"] = [{"record_id": tuning_ids[0], "verdict": "incorrect"}]
        overlap = client.put(
            f"/api/v1/class-similarity/jobs/{job_id}/evaluation",
            json={"manifest": heldout_template, "policy": {"mode": "threshold", "threshold": -1.0}},
        )
        assert overlap.status_code == 422
        assert "overlap tuning IDs" in overlap.json()["message"]

        heldout_template["frozen_ids"] = heldout_ids
        heldout_template["reviews"] = heldout_review_seed
        for index, review in enumerate(heldout_template["reviews"]):
            review["verdict"] = "incorrect" if index == 0 else "correct"
        heldout = client.put(
            f"/api/v1/class-similarity/jobs/{job_id}/evaluation",
            json={"manifest": heldout_template, "policy": {"mode": "threshold", "threshold": -1.0}},
        )
        assert heldout.status_code == 200, heldout.text
        heldout_payload = heldout.json()
        assert heldout_payload["evaluation"]["purpose"] == "heldout"
        assert heldout_payload["policy"]["threshold"] == 0.5
        assert heldout_payload["counts"]["status"] == "ready"
        assert heldout_payload["precision_recall_curve"] == []

        reused = client.put(
            f"/api/v1/class-similarity/jobs/{job_id}/evaluation",
            json={"manifest": heldout_template, "policy": {"mode": "threshold", "threshold": 0.5}},
        )
        assert reused.status_code == 422
        assert "already evaluated" in reused.json()["message"]


def test_class_similarity_rejects_a_run_after_its_source_changes(tmp_path: Path, monkeypatch):
    folders, _source_hashes = _fixture(tmp_path / "dataset")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "cases"))
    app = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: _embedder,
    )
    with TestClient(app) as client:
        created = client.post("/api/v1/visualize/jobs", json={
            "folders": [str(folder) for folder in folders],
            "model_id": "fixture-model",
            "method": "pca",
        })
        run_job = _wait_for_job(client, "/api/v1/jobs/{job_id}", created.json()["job_id"])
        assert run_job["state"] == "completed", run_job
        run_id = run_job["run_id"]

        source = folders[0] / "apple" / "a1.png"
        Image.new("RGB", (32, 32), (1, 2, 3)).save(source)
        started = client.post(f"/api/v1/runs/{run_id}/class-similarity/jobs", json={
            "model_id": "fixture-model",
            "k": 1,
        })
        assert started.status_code == 202, started.text
        analysis = _wait_for_job(
            client,
            "/api/v1/class-similarity/jobs/{job_id}",
            started.json()["job_id"],
        )
        assert analysis["state"] == "failed"
        assert analysis["error"]["code"] == "CLASS_SIMILARITY_INVALID"
        assert str(tmp_path) not in str(analysis)
