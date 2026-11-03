from __future__ import annotations

import base64
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
from fp_case_store import FalsePositiveCaseStore
from fp_external_evaluation import validate_evaluation_manifest
from fp_external_models import ExternalModelRegistry
from fp_onnx_demo import make_demo_model
from react_web_shell import create_app


def image_bytes(value: int, size: tuple[int, int] = (80, 60)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (value, value, value)).save(output, format="PNG")
    return output.getvalue()


def client_for(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "cases"))
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    return TestClient(
        create_app(
            model_catalog=lambda: [],
            model_loader=lambda _model: lambda _path: np.zeros(8),
        )
    )


def create_case(client: TestClient, data: bytes, kind: str = "original") -> dict:
    response = client.post(
        "/api/v1/fp-cases",
        json={
            "source_name": "reported.png",
            "source_kind": kind,
            "source_base64": base64.b64encode(data).decode("ascii"),
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def model_profile(client: TestClient, model_path: Path, name: str) -> dict:
    asset_response = client.post(
        "/api/v1/external-model-assets?filename=" + name + ".onnx",
        content=model_path.read_bytes(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert asset_response.status_code == 201, asset_response.text
    config = {
        "model_version": name,
        "output_format": "classification",
        "class_names": ["good", "crack"],
        "normal_classes": ["good"],
        "input_size": [32, 32],
        "resize": "stretch",
        "threshold": 0.5,
    }
    response = client.post(
        "/api/v1/external-model-profiles",
        json={"name": name, "asset_sha256": asset_response.json()["sha256"], "config": config},
    )
    assert response.status_code == 201, response.text
    return response.json()


def manifest_for(tmp_path: Path) -> dict:
    normal = tmp_path / "heldout-normal.png"
    defect = tmp_path / "heldout-defect.png"
    normal.write_bytes(image_bytes(80))
    defect.write_bytes(image_bytes(230))
    return {
        "schema_version": 1,
        "purpose": "independent",
        "independent_declared": True,
        "split_basis": "separate held-out lots; not used for training or threshold choice",
        "items": [
            {"id": "normal-1", "path": str(normal), "truth": "normal", "verified": True, "group_id": "lot-a"},
            {"id": "defect-1", "path": str(defect), "truth": "defect", "verified": True, "group_id": "lot-b"},
        ],
    }


def test_react_paired_evaluation_is_saved_and_uses_same_holdout_for_both_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    pytest.importorskip("onnx")
    baseline_path = make_demo_model(tmp_path / "baseline.onnx", bias=0.3)
    candidate_path = make_demo_model(tmp_path / "candidate.onnx")
    with client_for(monkeypatch, tmp_path) as client:
        case = create_case(client, image_bytes(220))
        baseline = model_profile(client, baseline_path, "baseline")
        candidate = model_profile(client, candidate_path, "candidate")
        manifest = manifest_for(tmp_path)
        originals = {
            Path(row["path"]): Path(row["path"]).read_bytes()
            for row in manifest["items"]
        }
        response = client.post(
            "/api/v1/fp-cases/" + case["id"] + "/evaluations",
            json={
                "expected_revision": case["revision"],
                "baseline_profile_id": baseline["id"],
                "baseline_profile_revision": baseline["revision"],
                "candidate_profile_id": candidate["id"],
                "candidate_profile_revision": candidate["revision"],
                "manifest": manifest,
                "min_support": 1,
                "max_miss_increase": 0,
                "min_fp_reduction": 0.5,
            },
        )
        assert response.status_code == 201, response.text
        payload = response.json()
        result = payload["evaluation"]
        assert result["baseline"]["false_alarms"] == 1
        assert result["candidate"]["false_alarms"] == 0
        assert result["baseline"]["misses"] == result["candidate"]["misses"] == 0
        assert result["gate"]["status"] == "passed"
        assert len(result["dataset_fingerprint"]) == 64
        assert payload["case"]["revision"] == case["revision"] + 1
        assert payload["case"]["evaluations"][-1]["evidence_id"] in {
            row["id"] for row in payload["case"]["evidence"]
        }
        assert "path" not in json.dumps(result["manifest"])
        assert str(tmp_path) not in response.text
        assert "model_path" not in response.text
        assert all(path.read_bytes() == data for path, data in originals.items())

        stale = client.post(
            "/api/v1/fp-cases/" + case["id"] + "/evaluations",
            json={
                "expected_revision": case["revision"],
                "baseline_profile_id": baseline["id"],
                "candidate_profile_id": candidate["id"],
                "manifest": manifest,
                "min_support": 1,
            },
        )
        assert stale.status_code == 409


def test_react_paired_evaluation_endpoint_with_local_runtime_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import fp_external_evaluation

    class FixtureRuntime:
        def __init__(self, config: dict):
            self.config = config
            marker = Path(config["model_path"]).read_bytes()
            self.bias = 0.3 if marker == b"baseline-fixture" else 0.0
            self.identity = {
                "contract_id": "a" * 64 if self.bias else "b" * 64,
                "model_sha256": "c" * 64 if self.bias else "d" * 64,
            }

        def infer(self, data: bytes) -> dict:
            with Image.open(io.BytesIO(data)) as image:
                value = sum(image.convert("RGB").getpixel((0, 0))) / 3 / 255
            crack = min(1.0, value + self.bias)
            alarm = crack >= float(self.config["threshold"])
            return {
                "prediction": {
                    "task": "classification",
                    "alarm": alarm,
                    "class": "crack" if alarm else "good",
                    "decision": "crack" if alarm else "good",
                    "scores": {"good": 1.0 - crack, "crack": crack},
                    "detections": [],
                }
            }

    monkeypatch.setattr(fp_external_evaluation, "Runtime", FixtureRuntime)
    baseline_path = tmp_path / "baseline-fixture.onnx"
    candidate_path = tmp_path / "candidate-fixture.onnx"
    baseline_path.write_bytes(b"baseline-fixture")
    candidate_path.write_bytes(b"candidate-fixture")
    with client_for(monkeypatch, tmp_path) as client:
        case = create_case(client, image_bytes(220))
        baseline = model_profile(client, baseline_path, "baseline")
        candidate = model_profile(client, candidate_path, "candidate")
        response = client.post(
            "/api/v1/fp-cases/" + case["id"] + "/evaluations",
            json={
                "expected_revision": case["revision"],
                "baseline_profile_id": baseline["id"],
                "candidate_profile_id": candidate["id"],
                "manifest": manifest_for(tmp_path),
                "min_support": 1,
                "min_fp_reduction": 0.5,
            },
        )
        assert response.status_code == 201, response.text
        saved = response.json()
        assert saved["evaluation"]["gate"]["status"] == "passed"
        assert saved["evaluation"]["baseline"]["false_alarms"] == 1
        assert saved["evaluation"]["candidate"]["false_alarms"] == 0
        assert saved["case"]["revision"] == 2
        assert str(tmp_path) not in response.text
        assert "model_path" not in response.text


def test_independent_holdout_rejects_case_content_and_duplicate_files(tmp_path: Path):
    store = FalsePositiveCaseStore(tmp_path / "store")
    case_source = store.ingest(image_bytes(200), "case.png", "original")
    case = store.create(case_source, title="overlap")
    store.save(case)
    overlap_path = tmp_path / "overlap.png"
    overlap_path.write_bytes(image_bytes(200))
    manifest = {
        "schema_version": 1,
        "purpose": "independent",
        "independent_declared": True,
        "split_basis": "separate lot",
        "items": [
            {"id": "overlap", "path": str(overlap_path), "truth": "normal", "verified": True}
        ],
    }
    with pytest.raises(ValueError, match="重疊"):
        validate_evaluation_manifest(manifest, store)

    other = tmp_path / "other.png"
    other.write_bytes(image_bytes(80))
    manifest["items"] = [
        {"id": "one", "path": str(other), "truth": "normal", "verified": True},
        {"id": "two", "path": str(other), "truth": "defect", "verified": True},
    ]
    with pytest.raises(ValueError, match="重複影像"):
        validate_evaluation_manifest(manifest, store)


def test_react_training_export_requires_explicit_full_review_and_downloads_zip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    with client_for(monkeypatch, tmp_path) as client:
        case = create_case(client, image_bytes(190))
        created = client.post(
            "/api/v1/fp-cases/" + case["id"] + "/regions",
            json={
                "expected_revision": case["revision"],
                "source_id": case["sources"][0]["id"],
                "roi": [0.2, 0.25, 0.6, 0.75],
                "label": "predicted-crack",
            },
        )
        assert created.status_code == 201, created.text
        case = created.json()["case"]
        item_id = case["items"][0]["id"]
        reviewed = client.patch(
            "/api/v1/fp-cases/" + case["id"] + "/regions/" + item_id + "/review",
            json={
                "expected_revision": case["revision"],
                "verdict": "defect",
                "reviewer": "Engineer",
                "reason": "Visually confirmed a crack.",
            },
        )
        assert reviewed.status_code == 200, reviewed.text
        case = reviewed.json()
        request = {
            "expected_revision": case["revision"],
            "complete_images": [case["sources"][0]["sha256"]],
            "class_assignments": {"item:" + item_id: "crack"},
            "class_names": ["crack"],
            "reviewer": "Engineer",
            "reason": "Full image checked; all true defect boxes labeled.",
            "confirmed_complete": True,
        }
        response = client.post(
            "/api/v1/fp-cases/" + case["id"] + "/training-exports",
            json=request,
        )
        assert response.status_code == 201, response.text
        payload = response.json()
        export = payload["training_export"]
        assert export["n_images"] == 1
        assert export["n_boxes"] == 1
        assert export["n_negative_images"] == 0
        assert export["evidence_id"] in {row["id"] for row in payload["case"]["evidence"]}
        archive_response = client.get(
            "/api/v1/fp-cases/" + case["id"] + "/training-exports/"
            + export["id"] + "/archive.zip"
        )
        assert archive_response.status_code == 200
        with ZipFile(io.BytesIO(archive_response.content)) as archive:
            image_name = "images/" + case["sources"][0]["sha256"] + ".png"
            label_name = "labels/" + case["sources"][0]["sha256"] + ".txt"
            assert image_name in archive.namelist()
            assert archive.read(label_name).decode().startswith(
                "0 0.400000000 0.500000000 0.400000000 0.500000000"
            )
            provenance = json.loads(archive.read("provenance.json"))
            assert provenance["regions"][0]["review"]["verdict"] == "defect"

        request["expected_revision"] = payload["case"]["revision"]
        request["confirmed_complete"] = False
        declined = client.post(
            "/api/v1/fp-cases/" + case["id"] + "/training-exports",
            json=request,
        )
        assert declined.status_code == 422


def test_partial_normal_cannot_be_exported_as_a_negative_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    with client_for(monkeypatch, tmp_path) as client:
        case = create_case(client, image_bytes(180))
        region = client.post(
            "/api/v1/fp-cases/" + case["id"] + "/regions",
            json={
                "expected_revision": case["revision"],
                "source_id": case["sources"][0]["id"],
                "roi": [0.1, 0.1, 0.5, 0.5],
                "label": "reflection",
            },
        ).json()
        case = region["case"]
        item_id = case["items"][0]["id"]
        case = client.patch(
            "/api/v1/fp-cases/" + case["id"] + "/regions/" + item_id + "/review",
            json={
                "expected_revision": case["revision"],
                "verdict": "normal",
                "reviewer": "Engineer",
                "reason": "This ROI is reflection only.",
            },
        ).json()
        response = client.post(
            "/api/v1/fp-cases/" + case["id"] + "/training-exports",
            json={
                "expected_revision": case["revision"],
                "complete_images": [case["sources"][0]["sha256"]],
                "class_assignments": {},
                "class_names": ["crack"],
                "reviewer": "Engineer",
                "reason": "Checked selected region.",
                "confirmed_complete": True,
            },
        )
        assert response.status_code == 422
        assert "局部正常" in response.json()["message"]


def test_only_full_image_normal_review_exports_an_empty_label_and_blocks_case_conflicts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    data = image_bytes(170)
    with client_for(monkeypatch, tmp_path) as client:
        normal_case = create_case(client, data)
        full_region = client.post(
            "/api/v1/fp-cases/" + normal_case["id"] + "/regions",
            json={
                "expected_revision": normal_case["revision"],
                "source_id": normal_case["sources"][0]["id"],
                "roi": [0, 0, 1, 1],
                "label": "whole-image-review",
            },
        ).json()
        normal_case = full_region["case"]
        normal_item_id = normal_case["items"][0]["id"]
        normal_case = client.patch(
            "/api/v1/fp-cases/" + normal_case["id"] + "/regions/" + normal_item_id + "/review",
            json={
                "expected_revision": normal_case["revision"],
                "verdict": "normal",
                "reviewer": "Engineer",
                "reason": "The complete original image contains no defect.",
            },
        ).json()
        normal_export = client.post(
            "/api/v1/fp-cases/" + normal_case["id"] + "/training-exports",
            json={
                "expected_revision": normal_case["revision"],
                "complete_images": [normal_case["sources"][0]["sha256"]],
                "class_assignments": {},
                "class_names": ["crack"],
                "reviewer": "Engineer",
                "reason": "The entire image was reviewed.",
                "confirmed_complete": True,
            },
        )
        assert normal_export.status_code == 201, normal_export.text
        export = normal_export.json()["training_export"]
        assert export["n_negative_images"] == 1
        zip_response = client.get(
            "/api/v1/fp-cases/" + normal_case["id"] + "/training-exports/"
            + export["id"] + "/archive.zip"
        )
        with ZipFile(io.BytesIO(zip_response.content)) as archive:
            sha = normal_case["sources"][0]["sha256"]
            assert archive.read("labels/" + sha + ".txt") == b""

        defect_case = create_case(client, data)
        defect_region = client.post(
            "/api/v1/fp-cases/" + defect_case["id"] + "/regions",
            json={
                "expected_revision": defect_case["revision"],
                "source_id": defect_case["sources"][0]["id"],
                "roi": [0.2, 0.2, 0.5, 0.5],
                "label": "suggested-class",
            },
        ).json()
        defect_case = defect_region["case"]
        defect_item_id = defect_case["items"][0]["id"]
        defect_case = client.patch(
            "/api/v1/fp-cases/" + defect_case["id"] + "/regions/" + defect_item_id + "/review",
            json={
                "expected_revision": defect_case["revision"],
                "verdict": "defect",
                "reviewer": "Engineer",
                "reason": "The ROI contains a confirmed crack.",
            },
        ).json()
        conflict = client.post(
            "/api/v1/fp-cases/" + defect_case["id"] + "/training-exports",
            json={
                "expected_revision": defect_case["revision"],
                "complete_images": [defect_case["sources"][0]["sha256"]],
                "class_assignments": {"item:" + defect_item_id: "crack"},
                "class_names": ["crack"],
                "reviewer": "Engineer",
                "reason": "Full image and defect region reviewed.",
                "confirmed_complete": True,
            },
        )
        assert conflict.status_code == 422
        assert "其他案件有整圖正常裁決" in conflict.json()["message"]


def test_screenshot_source_is_rejected_for_training_handoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    with client_for(monkeypatch, tmp_path) as client:
        case = create_case(client, image_bytes(150), kind="screenshot")
        region = client.post(
            "/api/v1/fp-cases/" + case["id"] + "/regions",
            json={
                "expected_revision": case["revision"],
                "source_id": case["sources"][0]["id"],
                "roi": [0, 0, 1, 1],
                "label": "whole-image",
            },
        ).json()
        case = region["case"]
        item_id = case["items"][0]["id"]
        case = client.patch(
            "/api/v1/fp-cases/" + case["id"] + "/regions/" + item_id + "/review",
            json={
                "expected_revision": case["revision"],
                "verdict": "normal",
                "reviewer": "Engineer",
                "reason": "Checked the visible screenshot.",
            },
        ).json()
        response = client.post(
            "/api/v1/fp-cases/" + case["id"] + "/training-exports",
            json={
                "expected_revision": case["revision"],
                "complete_images": [case["sources"][0]["sha256"]],
                "class_assignments": {},
                "class_names": ["crack"],
                "reviewer": "Engineer",
                "reason": "Checked screenshot only.",
                "confirmed_complete": True,
            },
        )
        assert response.status_code == 422
        assert "截圖不可" in response.json()["message"]
