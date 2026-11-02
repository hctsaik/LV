from __future__ import annotations

import base64
import copy
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fp_external_runtime import DEFAULT_CONFIG, compare_reference, decode_output, prepare_input
from fp_onnx_demo import make_demo_model
from react_web_shell import create_app


def image_bytes(value: int = 220, size: tuple[int, int] = (80, 60)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (value, value, value)).save(output, format="PNG")
    return output.getvalue()


def create_test_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "cases"))
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    return TestClient(create_app(model_catalog=lambda: [], model_loader=lambda _model: lambda _path: np.zeros(8)))


def create_case(client: TestClient, image: bytes, *, kind: str = "original") -> dict:
    response = client.post("/api/v1/fp-cases", json={
        "source_name": "reported.png",
        "source_kind": kind,
        "source_base64": base64.b64encode(image).decode("ascii"),
    })
    assert response.status_code == 201, response.text
    return response.json()


def add_profile(client: TestClient, model: bytes, config: dict, *, name: str = "baseline") -> dict:
    uploaded = client.post(
        "/api/v1/external-model-assets?filename=local-model.onnx",
        content=model,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert uploaded.status_code == 201, uploaded.text
    asset = uploaded.json()
    response = client.post("/api/v1/external-model-profiles", json={
        "name": name,
        "asset_sha256": asset["sha256"],
        "config": config,
    })
    assert response.status_code == 201, response.text
    return response.json()


def classifier_config() -> dict:
    return {
        "model_version": "fixture-v1",
        "output_format": "classification",
        "class_names": ["good", "crack"],
        "normal_classes": ["good"],
        "input_size": [32, 32],
        "resize": "stretch",
        "threshold": 0.5,
    }


def test_explicit_preprocessing_and_detection_layouts():
    config = {
        **DEFAULT_CONFIG,
        "input_size": [32, 32],
        "color": "BGR",
        "mean": [0.1, 0.2, 0.3],
        "std": [1, 2, 3],
    }
    tensor, _preview, geometry = prepare_input(Image.new("RGB", (64, 32), (255, 128, 0)), config)
    assert geometry["padding_xy"] == [0, 8]
    assert tensor[0, :, 16, 16] == pytest.approx([-.1, (128 / 255 - .2) / 2, (1 - .3) / 3])

    yolo_v8 = np.array([[[16, 16], [16, 16], [20, 20], [20, 20], [.9, .8]]])
    decoded = decode_output(yolo_v8, {**config, "output_format": "yolo_v8"}, geometry)
    assert len(decoded["detections"]) == 1
    assert decoded["candidate_count"] == 2
    assert decoded["raw_candidates_available"] is True

    yolo_v5 = np.array([[[16, 16, 20, 20, .8, .9]]])
    decoded_v5 = decode_output(yolo_v5, {**config, "output_format": "yolo_v5"}, geometry)
    assert decoded_v5["detections"][0]["score"] == pytest.approx(.72)
    already_postprocessed = decode_output(
        np.array([[[0, 8, 32, 24, .9, 0]]]),
        {**config, "output_format": "xyxy", "score_activation": "none"},
        geometry,
    )
    assert already_postprocessed["detections"][0]["box"] == [0, 0, 1, 1]
    assert already_postprocessed["nms_applied"] is False


def test_historical_replay_requires_original_contract_and_complete_boxes():
    row = {"class": "crack", "score": .91, "box": [.1, .2, .4, .6]}
    run = {
        "image_sha256": "image-sha",
        "contract_id": "contract-sha",
        "prediction": {"task": "detection", "alarm": True, "detections": [row]},
    }
    reference = {
        "origin": "deployment",
        "recorded_at": "2026-09-01T00:00:00Z",
        "image_sha256": "image-sha",
        "contract_id": "contract-sha",
        "prediction": {"task": "detection", "alarm": True, "detections": [copy.deepcopy(row)]},
    }
    assert compare_reference(run, reference)["status"] == "matched"
    assert compare_reference(run, reference, source_kind="screenshot")["status"] == "unavailable"
    changed_box = copy.deepcopy(reference)
    changed_box["prediction"]["detections"][0]["box"][0] += .05
    assert compare_reference(run, changed_box)["status"] == "mismatched"
    incomplete = copy.deepcopy(reference)
    incomplete["prediction"]["detections"] = []
    assert compare_reference(run, incomplete)["status"] == "mismatched"
    assert compare_reference(run, {**reference, "origin": "lv_local_inference"})["status"] == "unavailable"


def test_react_model_upload_inference_history_and_evidence_are_durable_and_path_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    pytest.importorskip("onnx")
    model_path = make_demo_model(tmp_path / "demo-classifier.onnx")
    model = model_path.read_bytes()
    source_data = image_bytes()

    with create_test_client(monkeypatch, tmp_path) as client:
        case = create_case(client, source_data)
        profile = add_profile(client, model, classifier_config())
        assert profile["latest_revision"] == profile["revision"]
        assert profile["asset"]["sha256"]
        assert "relative_path" not in json.dumps(profile)
        assert str(tmp_path) not in json.dumps(profile)

        first_response = client.post(f"/api/v1/fp-cases/{case['id']}/external-runs", json={
            "expected_revision": case["revision"],
            "source_id": case["sources"][0]["id"],
            "profile_id": profile["id"],
            "profile_revision": profile["revision"],
            "score_atol": 0.001,
            "box_atol": 0.001,
        })
        assert first_response.status_code == 201, first_response.text
        first = first_response.json()
        run = first["run"]
        assert run["prediction"]["class"] == "crack"
        assert run["prediction"]["score"] == pytest.approx(220 / 255, abs=1e-5)
        assert run["parity"]["status"] == "unavailable"
        assert len(run["contract_id"]) == 64
        assert run["contract"]["model_sha256"] == profile["asset"]["sha256"]
        assert run["image_sha256"] == run["source_context_id"]
        assert "model_path" not in json.dumps(first)
        assert "artifact_dir" not in json.dumps(first)
        assert str(tmp_path) not in json.dumps(first)
        assert first["case"]["revision"] == 2
        assert len(first["case"]["external_runs"]) == 1
        assert first["case"]["evidence"][-1]["role"] == "external_inference"

        replay_reference = {
            "origin": "deployment",
            "recorded_at": "2026-09-01T00:00:00Z",
            "image_sha256": run["image_sha256"],
            "contract_id": run["contract_id"],
            "prediction": run["prediction"],
        }
        replay_response = client.post(f"/api/v1/fp-cases/{case['id']}/external-runs", json={
            "expected_revision": 2,
            "source_id": case["sources"][0]["id"],
            "profile_id": profile["id"],
            "profile_revision": profile["revision"],
            "reference": replay_reference,
            "score_atol": 0.001,
            "box_atol": 0.001,
        })
        assert replay_response.status_code == 201, replay_response.text
        replayed = replay_response.json()
        assert replayed["run"]["parity"]["status"] == "matched"

        preview = client.get(f"/api/v1/fp-cases/{case['id']}/external-runs/{run['id']}/input-preview.png")
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/png"
        evidence = client.get(f"/api/v1/fp-cases/{case['id']}/external-runs/{run['id']}/evidence.json")
        assert evidence.status_code == 200
        assert "model_path" not in evidence.text and str(tmp_path) not in evidence.text
        with ZipFile(io.BytesIO(client.get(
            f"/api/v1/fp-cases/{case['id']}/external-runs/{run['id']}/evidence.zip"
        ).content)) as archive:
            assert set(archive.namelist()) == {"evidence.json", "model_input.png", "source.png", "tensors.npz"}
            exported = archive.read("evidence.json").decode("utf-8")
            assert "model_path" not in exported and str(tmp_path) not in exported

    # A new worker can reopen both the stored profile and case run, then execute
    # the same profile revision without needing the original upload path.
    with create_test_client(monkeypatch, tmp_path) as reopened_client:
        profiles = reopened_client.get("/api/v1/external-model-profiles").json()["items"]
        reopened_case = reopened_client.get(f"/api/v1/fp-cases/{case['id']}").json()
        assert profiles[0]["id"] == profile["id"]
        assert len(reopened_case["external_runs"]) == 2


def test_bad_output_layout_does_not_publish_a_case_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    pytest.importorskip("onnx")
    model_path = make_demo_model(tmp_path / "demo-classifier.onnx")
    config = {**classifier_config(), "output_format": "yolo_v8"}
    with create_test_client(monkeypatch, tmp_path) as client:
        case = create_case(client, image_bytes())
        profile = add_profile(client, model_path.read_bytes(), config)
        response = client.post(f"/api/v1/fp-cases/{case['id']}/external-runs", json={
            "expected_revision": 1,
            "source_id": case["sources"][0]["id"],
            "profile_id": profile["id"],
            "profile_revision": profile["revision"],
        })
        assert response.status_code == 422
        assert "4+C" in response.json()["message"]
        saved = client.get(f"/api/v1/fp-cases/{case['id']}").json()
        assert saved["revision"] == 1
        assert saved["external_runs"] == []
