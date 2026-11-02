from __future__ import annotations

import csv
import io
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


def test_fewshot_scan_threshold_confirm_append_and_reuse_updated_bank(monkeypatch, tmp_path: Path):
    samples = _samples(tmp_path / "sample-set", with_labels=True)
    targets = _samples(tmp_path / "target-set", with_labels=True)
    source_hashes = {
        path: path.read_bytes()
        for root in (samples, targets)
        for path in root.rglob("*") if path.is_file()
    }
    with _client(monkeypatch, tmp_path) as worker:
        created_bank = worker.post("/api/v1/fewshot/sample-banks/jobs", json={
            "sample_folders": [str(samples)],
            "model_id": "fixture-model",
            "object_source": "yolo",
        })
        assert created_bank.status_code == 202, created_bank.text
        bank_job = _wait_job(worker, created_bank.json()["job_id"])
        assert bank_job["state"] == "completed", bank_job
        bank = bank_job["result"]

        created_scan = worker.post("/api/v1/fewshot/scans/jobs", json={
            "sample_bank_id": bank["sample_bank_id"],
            "target_folders": [str(targets)],
            "k": 20,
            "min_proposal_conf": 0.0,
        })
        assert created_scan.status_code == 202, created_scan.text
        scan_job = _wait_job(worker, created_scan.json()["job_id"])
        assert scan_job["state"] == "completed", scan_job
        scan = scan_job["result"]
        assert scan["candidate_count"] == 3
        assert all(item["similarity"] >= 0.99 for item in scan["candidates"])
        assert str(tmp_path) not in json.dumps(scan_job)
        preview = worker.get(
            f"/api/v1/fewshot/scans/{scan['scan_id']}/records/0/preview.jpg"
        )
        assert preview.status_code == 200
        assert Image.open(__import__("io").BytesIO(preview.content)).size[0] > 0

        selected = [scan["candidates"][0], scan["candidates"][2]]
        appended = worker.post("/api/v1/fewshot/sample-banks/add-candidates", json={
            "sample_bank_id": bank["sample_bank_id"],
            "scan_id": scan["scan_id"],
            "item_ids": [item["item_id"] for item in selected],
            "theta": 0.9,
            "class_overrides": {selected[0]["item_id"]: "stain"},
        })
        assert appended.status_code == 200, appended.text
        updated_bank = appended.json()["sample_bank"]
        assert appended.json()["added"] == 2
        assert updated_bank["sample_count"] == bank["sample_count"] + 2
        expected_stain = bank["class_counts"]["stain"] + 1 + int(
            selected[1]["suggested_class"] == "stain"
        )
        assert updated_bank["class_counts"]["stain"] == expected_stain

        storage = tmp_path / "cache" / "fewshot_sample_banks" / bank["sample_bank_id"]
        import sample_bank
        persisted = sample_bank.load_sample_bank(storage)
        assert persisted["object_source"] == "yolo"
        assert {item["item_id"] for item in persisted["provenance"][-2:]} == {
            item["item_id"] for item in selected
        }

        duplicate = worker.post("/api/v1/fewshot/sample-banks/add-candidates", json={
            "sample_bank_id": bank["sample_bank_id"],
            "scan_id": scan["scan_id"],
            "item_ids": [selected[0]["item_id"]],
            "theta": 0.9,
            "class_overrides": {},
        })
        assert duplicate.status_code == 200, duplicate.text
        assert duplicate.json()["added"] == 0
        assert duplicate.json()["skipped"] == 1

        reused_scan = worker.post("/api/v1/fewshot/scans/jobs", json={
            "sample_bank_id": bank["sample_bank_id"],
            "target_folders": [str(targets)],
            "k": 20,
            "min_proposal_conf": 0.0,
        })
        assert reused_scan.status_code == 202, reused_scan.text
        reused_job = _wait_job(worker, reused_scan.json()["job_id"])
        assert reused_job["state"] == "completed", reused_job
        assert reused_job["result"]["sample_bank_id"] == bank["sample_bank_id"]
        assert worker.get(
            f"/api/v1/fewshot/sample-banks/{bank['sample_bank_id']}"
        ).json()["sample_bank"]["sample_count"] == updated_bank["sample_count"]

    assert {path: path.read_bytes() for path in source_hashes} == source_hashes


def test_fewshot_whole_image_scan_keeps_full_image_boxes_and_source_identity(monkeypatch, tmp_path: Path):
    samples = _samples(tmp_path / "whole-samples", with_labels=False)
    targets = _samples(tmp_path / "whole-targets", with_labels=False)
    with _client(monkeypatch, tmp_path) as worker:
        created_bank = worker.post("/api/v1/fewshot/sample-banks/jobs", json={
            "sample_folders": [str(samples)],
            "model_id": "fixture-model",
            "object_source": "whole_image",
        })
        assert created_bank.status_code == 202, created_bank.text
        bank = _wait_job(worker, created_bank.json()["job_id"])["result"]
        created_scan = worker.post("/api/v1/fewshot/scans/jobs", json={
            "sample_bank_id": bank["sample_bank_id"],
            "target_folders": [str(targets)],
            "k": 10,
            "min_proposal_conf": 0.0,
        })
        assert created_scan.status_code == 202, created_scan.text
        result = _wait_job(worker, created_scan.json()["job_id"])["result"]
        assert result["object_source"] == "whole_image"
        assert result["candidate_count"] == 2
        assert all(item["bbox"] == [0.5, 0.5, 1.0, 1.0] for item in result["candidates"])


def test_fewshot_export_respects_acceptance_theta_relabel_and_c6(monkeypatch, tmp_path: Path):
    samples = _samples(tmp_path / "sample-set", with_labels=True)
    targets = _samples(tmp_path / "target-set", with_labels=True)
    source_hashes = {
        path: path.read_bytes()
        for root in (samples, targets)
        for path in root.rglob("*") if path.is_file()
    }
    with _client(monkeypatch, tmp_path) as worker:
        created_bank = worker.post("/api/v1/fewshot/sample-banks/jobs", json={
            "sample_folders": [str(samples)],
            "model_id": "fixture-model",
            "object_source": "yolo",
        })
        assert created_bank.status_code == 202, created_bank.text
        bank = _wait_job(worker, created_bank.json()["job_id"])["result"]
        created_scan = worker.post("/api/v1/fewshot/scans/jobs", json={
            "sample_bank_id": bank["sample_bank_id"],
            "target_folders": [str(targets)],
            "k": 20,
            "min_proposal_conf": 0.0,
        })
        assert created_scan.status_code == 202, created_scan.text
        scan = _wait_job(worker, created_scan.json()["job_id"])["result"]
        candidates = scan["candidates"]
        assert len(candidates) == 3
        relabel = next(item for item in candidates if item["suggested_class"] == "scratch")
        other = next(item for item in candidates if item["item_id"] != relabel["item_id"])
        accepted = [relabel["item_id"], other["item_id"]]
        output_dir = tmp_path / "retrieval-export"

        exported = worker.post("/api/v1/fewshot/scans/exports", json={
            "sample_bank_id": bank["sample_bank_id"],
            "scan_id": scan["scan_id"],
            "accepted_item_ids": accepted,
            "theta": 0.9,
            "class_overrides": {relabel["item_id"]: "stain"},
            "output_dir": str(output_dir),
            "copy_images": True,
        })
        assert exported.status_code == 200, exported.text
        assert exported.json()["objects"] == 2
        assert exported.json()["csv_rows"] == 3
        assert (output_dir / "classes.txt").read_text(encoding="utf-8").splitlines() == ["scratch", "stain"]
        label_files = list((output_dir / "labels").glob("*.txt"))
        assert sum(len(path.read_text(encoding="utf-8").splitlines()) for path in label_files) == 2
        label_rows = [line for path in label_files for line in path.read_text(encoding="utf-8").splitlines()]
        assert "1 0.500000 0.500000 0.500000 0.500000" in label_rows
        with (output_dir / "retrieval_report.csv").open(encoding="utf-8", newline="") as stream:
            report = list(csv.DictReader(stream))
        assert len(report) == 3
        assert {row["decision"] for row in report} == {"accepted", "relabeled", "skipped"}
        relabeled_row = next(row for row in report if row["image_path"].endswith("sample-0.png")
                             and row["final_class"] == "stain")
        assert relabeled_row["decision"] == "relabeled"
        copied_images = list((output_dir / "images").iterdir())
        assert copied_images

        unsafe_dir = targets / "nested-output"
        rejected = worker.post("/api/v1/fewshot/scans/exports", json={
            "sample_bank_id": bank["sample_bank_id"],
            "scan_id": scan["scan_id"],
            "accepted_item_ids": accepted,
            "theta": 0.9,
            "class_overrides": {},
            "output_dir": str(unsafe_dir),
            "copy_images": False,
        })
        assert rejected.status_code == 422
        assert str(tmp_path) not in rejected.text
        assert not unsafe_dir.exists()

    assert {path: path.read_bytes() for path in source_hashes} == source_hashes


def test_fewshot_monitor_runs_without_anomaly_model_and_persists_queue_labels(monkeypatch, tmp_path: Path):
    samples = _samples(tmp_path / "sample-set", with_labels=True)
    targets = _samples(tmp_path / "target-set", with_labels=True)
    source_hashes = {
        path: path.read_bytes()
        for root in (samples, targets)
        for path in root.rglob("*") if path.is_file()
    }
    with _client(monkeypatch, tmp_path) as worker:
        created_bank = worker.post("/api/v1/fewshot/sample-banks/jobs", json={
            "sample_folders": [str(samples)],
            "model_id": "fixture-model",
            "object_source": "yolo",
        })
        assert created_bank.status_code == 202, created_bank.text
        bank = _wait_job(worker, created_bank.json()["job_id"])["result"]
        workspace_dir = tmp_path / "fewshot-monitor"
        unsafe_workspace = tmp_path / "cache" / "fewshot_sample_banks" / bank["sample_bank_id"] / "monitor"
        unsafe = worker.post("/api/v1/fewshot/monitors/workspaces", json={
            "sample_bank_id": bank["sample_bank_id"],
            "target_folders": [str(targets)],
            "workspace_dir": str(unsafe_workspace),
            "k": 20,
        })
        assert unsafe.status_code == 422
        assert str(tmp_path) not in unsafe.text
        assert not unsafe_workspace.exists()
        initialized = worker.post("/api/v1/fewshot/monitors/workspaces", json={
            "sample_bank_id": bank["sample_bank_id"],
            "target_folders": [str(targets)],
            "workspace_dir": str(workspace_dir),
            "k": 20,
        })
        assert initialized.status_code == 201, initialized.text
        workspace_id = initialized.json()["monitor_workspace_id"]
        assert initialized.json()["queue_len"] == 0
        assert str(tmp_path) not in initialized.text
        reinitialized = worker.post("/api/v1/fewshot/monitors/workspaces", json={
            "sample_bank_id": bank["sample_bank_id"],
            "target_folders": [str(targets)],
            "workspace_dir": str(workspace_dir),
            "k": 20,
        })
        assert reinitialized.status_code == 201, reinitialized.text
        assert reinitialized.json()["monitor_workspace_id"] == workspace_id
        profile = worker.get(f"/api/v1/fewshot/monitors/workspaces/{workspace_id}/profile.yaml")
        assert profile.status_code == 200
        assert "objective: retrieve" in profile.text
        assert "model_dir: ''" in profile.text

        first_run = worker.post("/api/v1/fewshot/monitors/scans/jobs", json={
            "monitor_workspace_id": workspace_id,
        })
        assert first_run.status_code == 202, first_run.text
        first_job = _wait_job(worker, first_run.json()["job_id"])
        assert first_job["state"] == "completed", first_job
        assert first_job["result"]["new"] == 2
        status = worker.get(f"/api/v1/fewshot/monitors/workspaces/{workspace_id}")
        assert status.status_code == 200, status.text
        view = status.json()
        assert view["queue_len"] == 3
        assert len(view["items"]) == 3
        assert str(tmp_path) not in status.text
        preview = worker.get(
            f"/api/v1/fewshot/monitors/workspaces/{workspace_id}/queue/0/preview.jpg"
        )
        assert preview.status_code == 200
        assert Image.open(__import__("io").BytesIO(preview.content)).size[0] > 0

        accepted, skipped = view["items"][:2]
        accepted_response = worker.post(
            f"/api/v1/fewshot/monitors/workspaces/{workspace_id}/labels",
            json={"item_id": accepted["item_id"], "decision": "defect", "label": "stain"},
        )
        assert accepted_response.status_code == 200, accepted_response.text
        skipped_response = worker.post(
            f"/api/v1/fewshot/monitors/workspaces/{workspace_id}/labels",
            json={"item_id": skipped["item_id"], "decision": "skip", "label": "scratch"},
        )
        assert skipped_response.status_code == 200, skipped_response.text
        labels_view = worker.get(f"/api/v1/fewshot/monitors/workspaces/{workspace_id}").json()
        decisions = {item["item_id"]: item["human_decision"] for item in labels_view["items"]}
        assert decisions[accepted["item_id"]] == "defect"
        assert decisions[skipped["item_id"]] == "skip"

        second_run = worker.post("/api/v1/fewshot/monitors/scans/jobs", json={
            "monitor_workspace_id": workspace_id,
        })
        assert second_run.status_code == 202, second_run.text
        second_job = _wait_job(worker, second_run.json()["job_id"])
        assert second_job["state"] == "completed", second_job
        next_queue = worker.get(f"/api/v1/fewshot/monitors/workspaces/{workspace_id}").json()
        assert next_queue["queue_len"] == 2
        assert accepted["item_id"] not in {item["item_id"] for item in next_queue["items"]}
        assert skipped["item_id"] in {item["item_id"] for item in next_queue["items"]}
        assert worker.get("/api/v1/anomaly/models").json()["items"] == []

    assert {path: path.read_bytes() for path in source_hashes} == source_hashes
