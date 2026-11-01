"""React web API acceptance tests; case status is tracked in the feedback matrix."""

from __future__ import annotations

import csv
import base64
import hashlib
from io import BytesIO, StringIO
import os
import sys
import time
import types
from pathlib import Path
from threading import Event
from zipfile import ZipFile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from react_web_shell import create_app
from interaction import compute_label_disagreement, compute_outlier_scores
from visualize_embeddings import discover_images_classifier


class _DeterministicUMAP:
    """Small picklable reducer used to exercise the reference-frame contract."""

    fit_calls = 0
    transform_calls = 0

    def __init__(self, *, n_components: int, n_neighbors: int, random_state: int):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.offset = 0.0

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        type(self).fit_calls += 1
        self.offset = float(type(self).fit_calls)
        return np.asarray(values[:, :self.n_components], dtype=np.float64) + self.offset

    def transform(self, values: np.ndarray) -> np.ndarray:
        type(self).transform_calls += 1
        return np.asarray(values[:, :self.n_components], dtype=np.float64) + self.offset + 10.0


@pytest.fixture
def classifier_fixture(tmp_path: Path) -> dict:
    """Four readable images plus one corrupt file in the legacy classifier layout."""
    specs = [
        ("train", "apple", "01.jpg", (220, 35, 45)),
        ("train", "apple", "02.jpg", (30, 190, 70)),
        ("train", "banana", "03.png", (35, 65, 215)),
        ("test", "banana", "04.jpeg", (210, 180, 25)),
    ]
    folders = {name: tmp_path / name for name in ("train", "test")}
    hashes = {}
    for split, label, filename, color in specs:
        image_path = folders[split] / label / filename
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color).save(image_path)
        hashes[str(image_path)] = hashlib.sha256(image_path.read_bytes()).hexdigest()

    broken = folders["train"] / "apple" / "broken.jpg"
    broken.write_bytes(b"not an image")
    return {"folders": list(folders.values()), "hashes": hashes, "root": tmp_path}


def _fake_embedder(calls: list[str] | None = None):
    def embed(path: Path) -> np.ndarray:
        if calls is not None:
            calls.append(str(path))
        pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        vector = np.zeros(384, dtype=np.float32)
        vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
        vector[3] = len(path.name) / 10.0
        return vector

    return embed


def _client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    embedder=None,
    models=None,
    object_model_loader=None,
    diagnostic_model_loader=None,
    case_search_model_provider=None,
):
    cache_root = tmp_path.parent / f"{tmp_path.name}-lv-cache"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path.parent / f"{tmp_path.name}-fp-cases"))
    return TestClient(create_app(
        model_catalog=lambda: models if models is not None else ["dinov2_vits14"],
        model_loader=lambda _model: embedder or _fake_embedder(),
        object_model_loader=object_model_loader,
        diagnostic_model_loader=diagnostic_model_loader,
        case_search_model_provider=case_search_model_provider,
    ))


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (80, 60), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_case_lv_web_05_case_01_case_sources_yolo_regions_and_revision_conflict(
    monkeypatch, tmp_path
):
    """CASE-01 keeps source identity and region decisions separate across revisions."""
    screenshot_path = tmp_path / "reported screenshot.png"
    screenshot_bytes = _png_bytes((210, 80, 40))
    screenshot_path.write_bytes(screenshot_bytes)
    screenshot_hash_before = hashlib.sha256(screenshot_path.read_bytes()).hexdigest()
    original_bytes = _png_bytes((40, 110, 190))

    with _client(monkeypatch, tmp_path / "fp-case-journey") as client:
        created_response = client.post("/api/v1/fp-cases", json={
            "source_name": "reported screenshot.html",
            "source_kind": "screenshot",
            "source_base64": base64.b64encode(screenshot_bytes).decode("ascii"),
            "title": "Reflective surface report",
            "description": "The deployed model flagged a normal reflection.",
            "external_model_id": "detector-2026.10",
            "threshold": "0.82",
            "raw_output": "class=crack score=0.94",
        })
        assert created_response.status_code == 201, created_response.text
        created = created_response.json()
        assert created["revision"] == 1
        assert created["external"] == {
            "model_id": "detector-2026.10",
            "threshold": "0.82",
            "raw_output": "class=crack score=0.94",
            "replay_status": "not_replayed",
        }
        assert created["sources"][0]["kind"] == "screenshot"
        assert "asset" not in created["sources"][0]
        assert "source_path" not in created["sources"][0]
        assert str(tmp_path) not in created_response.text

        image_response = client.get(
            f"/api/v1/fp-cases/{created['id']}/sources/{created['sources'][0]['id']}/image"
        )
        assert image_response.status_code == 200
        assert image_response.content == screenshot_bytes
        assert image_response.headers["content-type"] == "image/png"
        assert image_response.headers["x-content-type-options"] == "nosniff"

        source_response = client.post(f"/api/v1/fp-cases/{created['id']}/sources", json={
            "expected_revision": 1,
            "source_name": "original.png",
            "source_kind": "original",
            "source_base64": base64.b64encode(original_bytes).decode("ascii"),
        })
        assert source_response.status_code == 201, source_response.text
        sourced = source_response.json()
        original = next(source for source in sourced["sources"] if source["kind"] == "original")
        assert sourced["revision"] == 2
        assert original["sha256"] == hashlib.sha256(original_bytes).hexdigest()
        assert len(sourced["sources"]) == 2

        invalid_import = client.post(f"/api/v1/fp-cases/{created['id']}/predictions/yolo", json={
            "expected_revision": 2,
            "source_id": original["id"],
            "filename": "predictions.txt",
            "content": "0 0.50 0.50 0.40 0.40 0.91\n1 0.2 0.2 0.1\n",
        })
        assert invalid_import.status_code == 422
        assert client.get(f"/api/v1/fp-cases/{created['id']}").json()["revision"] == 2
        assert client.get(f"/api/v1/fp-cases/{created['id']}").json()["items"] == []

        imported_response = client.post(f"/api/v1/fp-cases/{created['id']}/predictions/yolo", json={
            "expected_revision": 2,
            "source_id": original["id"],
            "filename": "predictions.txt",
            "content": "0 0.50 0.50 0.40 0.40 0.91\n1 0.30 0.40 0.20 0.30 0.76\n",
        })
        assert imported_response.status_code == 201, imported_response.text
        imported = imported_response.json()["case"]
        assert imported_response.json()["imported"] == 2
        assert imported["revision"] == 3
        assert [row["review"]["verdict"] for row in imported["items"]] == ["pending", "pending"]
        assert [row["label"] for row in imported["items"]] == ["0", "1"]
        assert imported["items"][0]["roi"] == [0.3, 0.3, 0.7, 0.7]
        assert imported["items"][0]["source_id"] == original["id"]
        assert imported["evidence"][-1]["filename"] == "predictions.txt"

        first, second = imported["items"]
        reviewed_response = client.patch(
            f"/api/v1/fp-cases/{created['id']}/regions/{first['id']}/review",
            json={
                "expected_revision": 3,
                "verdict": "normal",
                "reviewer": "Engineer A",
                "reason": "This is an allowed reflection under the product specification.",
            },
        )
        assert reviewed_response.status_code == 200, reviewed_response.text
        reviewed = reviewed_response.json()
        assert reviewed["revision"] == 4
        assert reviewed["items"][0]["review"]["verdict"] == "normal"
        assert reviewed["items"][1]["review"]["verdict"] == "pending"

        stale_response = client.patch(
            f"/api/v1/fp-cases/{created['id']}/regions/{second['id']}/review",
            json={
                "expected_revision": 3,
                "verdict": "defect",
                "reviewer": "Engineer B",
                "reason": "A second window's stale decision must not overwrite this case.",
            },
        )
        assert stale_response.status_code == 409
        assert stale_response.json()["code"] == "CASE_REVISION_CONFLICT"
        after_stale = client.get(f"/api/v1/fp-cases/{created['id']}").json()
        assert after_stale["items"][0]["review"]["verdict"] == "normal"
        assert after_stale["items"][1]["review"]["verdict"] == "pending"

        second_review = client.patch(
            f"/api/v1/fp-cases/{created['id']}/regions/{second['id']}/review",
            json={
                "expected_revision": 4,
                "verdict": "defect",
                "reviewer": "Engineer B",
                "reason": "Independent review found a material crack in this ROI.",
            },
        )
        assert second_review.status_code == 200, second_review.text
        final_case = second_review.json()
        assert [row["review"]["verdict"] for row in final_case["items"]] == ["normal", "defect"]
        assert "whole_image_label" not in final_case
        assert "label" not in final_case
        assert screenshot_path.read_bytes() == screenshot_bytes
        assert hashlib.sha256(screenshot_path.read_bytes()).hexdigest() == screenshot_hash_before


def test_case_lv_web_05_case_01_manual_roi_and_review_validation(monkeypatch, tmp_path):
    image_bytes = _png_bytes((30, 140, 90))
    encoded = base64.b64encode(image_bytes).decode("ascii")
    with _client(monkeypatch, tmp_path / "fp-case-manual") as client:
        created = client.post("/api/v1/fp-cases", json={
            "source_name": "report.png",
            "source_kind": "original",
            "source_base64": encoded,
        }).json()
        bad_roi = client.post(f"/api/v1/fp-cases/{created['id']}/regions", json={
            "expected_revision": 1,
            "source_id": created["sources"][0]["id"],
            "roi": [0.7, 0.2, 0.3, 0.8],
            "label": "scratch",
        })
        assert bad_roi.status_code == 422

        added = client.post(f"/api/v1/fp-cases/{created['id']}/regions", json={
            "expected_revision": 1,
            "source_id": created["sources"][0]["id"],
            "roi": [0.1, 0.2, 0.6, 0.8],
            "label": "scratch",
        })
        assert added.status_code == 201, added.text
        case = added.json()["case"]
        item = case["items"][0]
        assert item["roi"] == [0.1, 0.2, 0.6, 0.8]
        assert item["review"]["verdict"] == "pending"

        missing_reviewer = client.patch(
            f"/api/v1/fp-cases/{created['id']}/regions/{item['id']}/review",
            json={"expected_revision": 2, "verdict": "normal", "reviewer": "", "reason": "reason"},
        )
        assert missing_reviewer.status_code == 422
        assert client.get(f"/api/v1/fp-cases/{created['id']}").json()["items"][0]["review"]["verdict"] == "pending"

        disputed = client.patch(
            f"/api/v1/fp-cases/{created['id']}/regions/{item['id']}/review",
            json={"expected_revision": 2, "verdict": "disputed", "reviewer": "Reviewer", "reason": "Specification needs clarification."},
        )
        assert disputed.status_code == 200, disputed.text
        insufficient = client.patch(
            f"/api/v1/fp-cases/{created['id']}/regions/{item['id']}/review",
            json={"expected_revision": 3, "verdict": "insufficient", "reviewer": "Reviewer", "reason": "The available image does not show enough detail."},
        )
        assert insufficient.status_code == 200, insufficient.text
        reopened = client.get(f"/api/v1/fp-cases/{created['id']}").json()
        assert reopened["revision"] == 4
        assert reopened["items"][0]["review"]["verdict"] == "insufficient"
        with client.app.state.fp_case_store._connect() as connection:
            history_count = connection.execute(
                "SELECT COUNT(*) FROM history WHERE id = ?", (created["id"],)
            ).fetchone()[0]
        assert history_count == 4


def test_case_lv_web_05_retrieve_01_existing_cases_receive_empty_retrieval_defaults(
    monkeypatch, tmp_path
):
    """Cases saved before retrieval fields existed still open and can be edited."""
    with _client(monkeypatch, tmp_path / "fp-retrieve-legacy") as client:
        created = client.post("/api/v1/fp-cases", json={
            "source_name": "legacy.png",
            "source_kind": "original",
            "source_base64": base64.b64encode(_png_bytes((150, 70, 30))).decode("ascii"),
            "title": "Existing case",
        }).json()
        with client.app.state.fp_case_store._connect() as connection:
            connection.execute(
                "UPDATE cases SET body = json_remove(body, "
                "'$.candidates', '$.searches', '$.claims', '$.batches', "
                "'$.reply_status', '$.cause_status', '$.improvement_status', "
                "'$.effort_minutes', '$.effort_log') WHERE id = ?",
                (created["id"],),
            )

        reopened = client.get(f"/api/v1/fp-cases/{created['id']}")
        assert reopened.status_code == 200
        assert reopened.json()["candidates"] == []
        assert reopened.json()["searches"] == []
        listed = client.get("/api/v1/fp-cases").json()["items"][0]
        assert listed["claims"] == [] and listed["batches"] == []
        assert listed["reply_status"] == "draft"
        assert listed["cause_status"] == "unknown"

        added = client.post(f"/api/v1/fp-cases/{created['id']}/regions", json={
            "expected_revision": 1,
            "source_id": created["sources"][0]["id"],
            "roi": [0.1, 0.1, 0.8, 0.8],
            "label": "legacy-region",
        })
        assert added.status_code == 201, added.text
        assert added.json()["case"]["searches"] == []


def test_case_lv_web_05_retrieve_01_balanced_results_cache_evidence_and_candidate_copy(
    monkeypatch, tmp_path
):
    normal = tmp_path / "normal"
    defect = tmp_path / "defect"
    normal.mkdir()
    defect.mkdir()
    (normal / "normal.png").write_bytes(_png_bytes((210, 15, 10)))
    (normal / "normal-copy.png").write_bytes((normal / "normal.png").read_bytes())
    (defect / "defect.png").write_bytes(_png_bytes((10, 10, 230)))
    (defect / "label-conflict.png").write_bytes((normal / "normal.png").read_bytes())
    (normal / "broken.png").write_bytes(b"not an image")
    reference_files = [*normal.iterdir(), *defect.iterdir()]
    reference_before = {path: path.read_bytes() for path in reference_files}
    calls: list[tuple[int, int]] = []

    def embed(image):
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        calls.append((image.width, image.height))
        return pixels.mean(axis=(0, 1)) + 1.0

    recipe = {"model": "fixture-mean", "checkpoint_sha256": "a" * 64, "target_res": 224}
    with _client(
        monkeypatch,
        tmp_path / "fp-retrieve-balanced",
        case_search_model_provider=lambda: (recipe, embed),
    ) as client:
        query_bytes = _png_bytes((220, 10, 10))
        created = client.post("/api/v1/fp-cases", json={
            "source_name": "reported.png",
            "source_kind": "screenshot",
            "source_base64": base64.b64encode(query_bytes).decode("ascii"),
            "title": "Color comparison report",
            "external_model_id": "detector-2026.10",
        }).json()
        added = client.post(f"/api/v1/fp-cases/{created['id']}/regions", json={
            "expected_revision": 1,
            "source_id": created["sources"][0]["id"],
            "roi": [0.2, 0.2, 0.8, 0.8],
            "label": "crack",
            "score": 0.94,
        })
        assert added.status_code == 201, added.text
        item = added.json()["case"]["items"][0]

        first_response = client.post(f"/api/v1/fp-cases/{created['id']}/searches", json={
            "expected_revision": 2,
            "item_id": item["id"],
            "normal_folder": str(normal),
            "defect_folder": str(defect),
            "candidate_scope": "whole_image",
            "limit": 20,
            "top_k": 6,
            "include_history": False,
        })
        assert first_response.status_code == 201, first_response.text
        first_payload = first_response.json()
        first_case = first_payload["case"]
        search = first_case["searches"][-1]
        assert first_case["revision"] == 3
        assert len(search["groups"]["normal"]) == 1
        assert len(search["groups"]["defect"]) == 2
        assert search["groups"]["history"] == []
        assert search["groups"]["normal"][0]["label_conflict"] is True
        assert search["sources"]["normal"]["label_status"] == "supplied_folder_label"
        assert search["sources"]["normal"]["folder_sha256"]
        assert search["skipped_total"] == 1
        assert search["skipped"][0]["name"] == "broken.png"
        assert search["recipe"]["checkpoint_sha256"] == recipe["checkpoint_sha256"]
        assert str(tmp_path) not in first_response.text
        result_row = search["groups"]["normal"][0]
        assert "asset" not in result_row["source"] and "path" not in result_row
        image = client.get(
            f"/api/v1/fp-cases/{created['id']}/searches/{search['id']}/results/{result_row['id']}/image"
        )
        assert image.status_code == 200
        assert image.content == (normal / "normal.png").read_bytes()
        calls_after_first = len(calls)

        second_response = client.post(f"/api/v1/fp-cases/{created['id']}/searches", json={
            "expected_revision": 3,
            "item_id": item["id"],
            "normal_folder": str(normal),
            "defect_folder": str(defect),
            "candidate_scope": "whole_image",
            "limit": 20,
            "top_k": 6,
            "include_history": False,
        })
        assert second_response.status_code == 201, second_response.text
        second_case = second_response.json()["case"]
        assert len(second_case["searches"]) == 2
        assert len(calls) == calls_after_first, "recipe+content+ROI cache recomputed an unchanged search"

        second_search = second_case["searches"][-1]
        result = second_search["groups"]["normal"][0]
        candidate_response = client.post(
            f"/api/v1/fp-cases/{created['id']}/searches/{second_search['id']}/results/{result['id']}/candidate",
            json={"expected_revision": 4, "category": "normal"},
        )
        assert candidate_response.status_code == 201, candidate_response.text
        candidate_case = candidate_response.json()["case"]
        assert candidate_response.json()["added"] is True
        assert candidate_case["revision"] == 5
        assert candidate_case["candidates"][0]["id"] == result["id"]
        assert candidate_case["candidates"][0]["source"]["sha256"] == result["sha256"]
        assert candidate_case["candidates"][0]["review"]["verdict"] == "pending"

        duplicate_response = client.post(
            f"/api/v1/fp-cases/{created['id']}/searches/{second_search['id']}/results/{result['id']}/candidate",
            json={"expected_revision": 5, "category": "normal"},
        )
        assert duplicate_response.status_code == 201
        assert duplicate_response.json()["added"] is False
        assert duplicate_response.json()["case"]["revision"] == 5

    assert reference_before == {path: path.read_bytes() for path in reference_files}


def test_case_lv_web_05_retrieve_01_history_keeps_verdict_model_and_empty_categories(
    monkeypatch, tmp_path
):
    recipe = {"model": "fixture-mean", "checkpoint_sha256": "b" * 64, "target_res": 224}

    def embed(image):
        return np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) + 1.0

    with _client(
        monkeypatch,
        tmp_path / "fp-retrieve-history",
        case_search_model_provider=lambda: (recipe, embed),
    ) as client:
        old_bytes = _png_bytes((180, 30, 40))
        old = client.post("/api/v1/fp-cases", json={
            "source_name": "old-screenshot.png",
            "source_kind": "screenshot",
            "source_base64": base64.b64encode(old_bytes).decode("ascii"),
            "title": "Earlier reviewed case",
            "external_model_id": "detector-v1",
        }).json()
        old_region = client.post(f"/api/v1/fp-cases/{old['id']}/regions", json={
            "expected_revision": 1,
            "source_id": old["sources"][0]["id"],
            "roi": [0.0, 0.0, 0.5, 0.5],
            "label": "scratch",
        }).json()["case"]["items"][0]
        old_review = client.patch(
            f"/api/v1/fp-cases/{old['id']}/regions/{old_region['id']}/review",
            json={"expected_revision": 2, "verdict": "normal", "reviewer": "Old reviewer", "reason": "Accepted reflection."},
        )
        assert old_review.status_code == 200

        current = client.post("/api/v1/fp-cases", json={
            "source_name": "new-report.png",
            "source_kind": "original",
            "source_base64": base64.b64encode(_png_bytes((190, 30, 40))).decode("ascii"),
            "title": "Current case",
            "external_model_id": "detector-v2",
        }).json()
        current_region = client.post(f"/api/v1/fp-cases/{current['id']}/regions", json={
            "expected_revision": 1,
            "source_id": current["sources"][0]["id"],
            "roi": [0.1, 0.1, 0.8, 0.8],
            "label": "crack",
        }).json()["case"]["items"][0]
        searched = client.post(f"/api/v1/fp-cases/{current['id']}/searches", json={
            "expected_revision": 2,
            "item_id": current_region["id"],
            "candidate_scope": "same_roi",
            "include_history": True,
            "limit": 20,
            "top_k": 6,
        })
        assert searched.status_code == 201, searched.text
        case = searched.json()["case"]
        result = case["searches"][-1]
        assert result["groups"]["normal"] == []
        assert result["groups"]["defect"] == []
        historical = result["groups"]["history"]
        assert len(historical) == 1
        assert historical[0]["external_model"] == "detector-v1"
        assert historical[0]["label_status"] == "reviewed_normal"
        assert historical[0]["review"]["reviewer"] == "Old reviewer"
        assert historical[0]["roi"] == [0.0, 0.0, 0.5, 0.5]
        assert case["external"]["model_id"] == "detector-v2"
        assert case["cause_status"] == "unknown"
        assert result["sources"]["normal"]["n_files"] == 0
        assert result["sources"]["defect"]["n_files"] == 0
        assert str(tmp_path) not in searched.text


def test_case_lv_web_05_retrieve_01_same_roi_and_truncated_source_are_explicit(
    monkeypatch, tmp_path
):
    folder = tmp_path / "references"
    folder.mkdir()
    for index, color in enumerate(((210, 20, 30), (200, 30, 40), (180, 50, 60))):
        (folder / f"candidate-{index}.png").write_bytes(_png_bytes(color))
    recipe = {"model": "fixture-mean", "checkpoint_sha256": "c" * 64, "target_res": 224}

    def embed(image):
        return np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) + 1.0

    with _client(
        monkeypatch,
        tmp_path / "fp-retrieve-roi",
        case_search_model_provider=lambda: (recipe, embed),
    ) as client:
        created = client.post("/api/v1/fp-cases", json={
            "source_name": "query.png",
            "source_kind": "original",
            "source_base64": base64.b64encode(_png_bytes((205, 25, 35))).decode("ascii"),
        }).json()
        region = client.post(f"/api/v1/fp-cases/{created['id']}/regions", json={
            "expected_revision": 1,
            "source_id": created["sources"][0]["id"],
            "roi": [0.2, 0.1, 0.7, 0.9],
            "label": "scratch",
        }).json()["case"]["items"][0]
        response = client.post(f"/api/v1/fp-cases/{created['id']}/searches", json={
            "expected_revision": 2,
            "item_id": region["id"],
            "normal_folder": str(folder),
            "candidate_scope": "same_roi",
            "include_history": False,
            "limit": 1,
            "top_k": 1,
        })
        assert response.status_code == 201, response.text
        result = response.json()["case"]["searches"][-1]
        assert result["sources"]["normal"]["truncated"] is True
        assert result["sources"]["normal"]["n_files"] == 1
        assert result["groups"]["normal"][0]["roi"] == region["roi"]
        assert result["groups"]["defect"] == []


def _run(client: TestClient, folders: list[Path], *, model: str = "dinov2_vits14") -> dict:
    response = client.post("/api/v1/visualize/jobs", json={
        "folders": [str(folder) for folder in folders],
        "model_id": model,
        "method": "pca",
    })
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = client.get(f"/api/v1/jobs/{job_id}").json()
        if status["state"] in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(0.02)
    pytest.fail("job did not reach a terminal state within 15 seconds")


def _run_coverage(
    client: TestClient,
    folders: list[Path],
    *,
    model: str = "dinov2_vits14",
    unit: str = "image",
    method: str = "pca",
    dimension: int = 2,
    k: int = 10,
) -> dict:
    response = client.post("/api/v1/coverage/jobs", json={
        "folders": [str(folder) for folder in folders],
        "model_id": model,
        "unit": unit,
        "method": method,
        "dimension": dimension,
        "k": k,
    })
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = client.get(f"/api/v1/coverage/jobs/{job_id}").json()
        if status["state"] in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(0.02)
    pytest.fail("coverage job did not reach a terminal state within 15 seconds")


def test_case_lv_web_03_img_01_coverage_sparsity_projection_selection_inputs(
    classifier_fixture, monkeypatch, tmp_path,
):
    """Image coverage keeps high-dimensional sparsity separate from projection."""
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in classifier_fixture["folders"]
        for path in folder.rglob("*")
        if path.is_file()
    }
    with _client(monkeypatch, tmp_path) as client:
        job = _run_coverage(client, classifier_fixture["folders"], k=2)
        assert job["state"] == "completed", job
        run_id = job["run_id"]
        run_response = client.get(f"/api/v1/coverage/runs/{run_id}")
        assert run_response.status_code == 200
        run = run_response.json()
        vectors = client.app.state.coverage_embeddings[run_id]
        expected = compute_outlier_scores(
            vectors, vectors, k=2, candidates_in_reference=True,
        )
        assert np.allclose(run["sparsity"], expected)
        assert run["sparse_threshold"] == pytest.approx(np.percentile(expected, 90))
        assert run["sparse_count"] == int(np.sum(expected >= np.percentile(expected, 90)))
        assert len(run["points"]) == len(run["records"]) == 4
        assert "image_paths" not in run
        assert run["requested_method"] == "pca"
        assert all(np.isfinite([point["x"], point["y"], point["z"]]).all() for point in run["points"])

        switched = client.post(
            f"/api/v1/coverage/runs/{run_id}/projections",
            json={"method": "tsne", "dimension": 2},
        )
        assert switched.status_code == 200, switched.text
        projection = switched.json()
        assert projection["requested_method"] == "tsne"
        assert projection["method"] == "tsne"
        assert len(projection["points"]) == 4
        assert all(np.isfinite([point["x"], point["y"]]).all() for point in projection["points"])

        changed_k = client.post(
            f"/api/v1/coverage/runs/{run_id}/sparsity",
            json={"k": 1},
        )
        assert changed_k.status_code == 200, changed_k.text
        sparsity = changed_k.json()
        expected_k1 = compute_outlier_scores(
            vectors, vectors, k=1, candidates_in_reference=True,
        )
        assert sparsity["k"] == 1
        assert np.allclose(sparsity["scores"], expected_k1)

        thumbnail = client.get(f"/api/v1/coverage/runs/{run_id}/images/0/thumbnail")
        assert thumbnail.status_code == 200
        assert thumbnail.headers["content-type"].startswith("image/jpeg")
        assert client.get(f"/api/v1/coverage/runs/{run_id}/images/999/thumbnail").status_code == 404
        assert client.get("/api/v1/coverage/runs/missing").status_code == 404

    assert source_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_hashes
    }


def test_case_lv_web_03_attr_01_attribute_grid_calibration_and_cell_candidate_readback(
    classifier_fixture, monkeypatch, tmp_path,
):
    """Attribute cells preserve legacy binning, priors, index-only assets, and mining export."""
    source_paths = [
        path for folder in classifier_fixture["folders"] for path in folder.rglob("*")
        if path.is_file()
    ]
    pool = tmp_path / "cell-pool"
    pool.mkdir()
    for index, color in enumerate(((220, 50, 30), (20, 180, 70), (30, 40, 220))):
        Image.new("RGB", (36, 28), color).save(pool / f"pool-{index}.png")
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*source_paths, *pool.iterdir()]
    }

    with _client(monkeypatch, tmp_path / "attribute-grid") as client:
        job = _run_coverage(client, classifier_fixture["folders"], k=2)
        assert job["state"] == "completed", job
        run_id = job["run_id"]
        base = f"/api/v1/coverage/runs/{run_id}/completeness"
        response = client.post(base, json={
            "axis_x": "label", "axis_y": "brightness", "bins": 3,
            "t_abs": 10, "d_star": 0.6,
        })
        assert response.status_code == 200, response.text
        default = response.json()
        assert default["labels_x"] == ["apple", "banana"]
        assert len(default["labels_y"]) == 3
        assert len(default["cells"]) == 6
        assert sum(cell["n"] for cell in default["cells"]) == 4
        assert all(cell["t"] == 10 for cell in default["cells"])
        assert default["calibrated"] is False
        assert all(len(cell["sample_indices"]) <= 30 for cell in default["cells"])
        assert "image_paths" not in default
        assert all("path" not in cell for cell in default["cells"])

        applied = client.post(base, json={
            "axis_x": "label", "axis_y": "brightness", "bins": 3,
            "t_abs": 10, "d_star": 0.6,
            "frequencies": [
                {"x": 0, "y": 0, "frequency": "低"},
                {"x": 1, "y": 0, "frequency": "不適用"},
                {"x": 0, "y": 1, "frequency": "高"},
            ],
        })
        assert applied.status_code == 200, applied.text
        calibrated = applied.json()
        assert calibrated["calibrated"] is True
        assert next(cell for cell in calibrated["cells"] if (cell["x"], cell["y"]) == (0, 0))["t"] == 4
        assert next(cell for cell in calibrated["cells"] if (cell["x"], cell["y"]) == (0, 1))["t"] == 20
        assert next(cell for cell in calibrated["cells"] if (cell["x"], cell["y"]) == (1, 0))["state"] == "不適用"
        assert calibrated["health"]["counts"]["不適用"] == 1
        invalid_cell = client.post(base, json={
            "axis_x": "label", "axis_y": "brightness", "bins": 3,
            "t_abs": 10, "d_star": 0.6,
            "frequencies": [{"x": 999, "y": 0, "frequency": "低"}],
        })
        assert invalid_cell.status_code == 422

        selected = next(cell for cell in default["cells"] if cell["n"] > 0)
        mined = client.post(f"{base}/candidates", json={
            "folders": [str(pool)], "axis_x": "label", "axis_y": "brightness",
            "bins": 3, "x": selected["x"], "y": selected["y"],
        })
        assert mined.status_code == 200, mined.text
        candidates = mined.json()
        assert candidates["total"] == 3
        assert [item["distance"] for item in candidates["items"]] == sorted(
            item["distance"] for item in candidates["items"]
        )
        candidate_index = candidates["items"][0]["index"]
        assert client.get(f"{base}/candidates/{candidate_index}/thumbnail").status_code == 200
        csv_response = client.get(f"{base}/candidates.csv")
        assert csv_response.status_code == 200
        assert csv_response.text.lstrip("\ufeff").splitlines()[0] == "path,distance"
        assert str(pool) in csv_response.text

        empty = next(cell for cell in default["cells"] if cell["n"] == 0)
        empty_query = client.post(f"{base}/candidates", json={
            "folders": [str(pool)], "axis_x": "label", "axis_y": "brightness",
            "bins": 3, "x": empty["x"], "y": empty["y"],
        })
        assert empty_query.status_code == 200, empty_query.text
        assert empty_query.json()["total"] == 3

    assert source_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_hashes
    }


def _object_coverage_fixture(root: Path) -> tuple[list[Path], dict[str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    train, test = root / "train", root / "test"
    specs = [
        (train, "01.png", 0, 0.50, 0.50, 0.95),
        (train, "02.png", 0, 0.10, 0.10, 0.40),
        (train, "03.png", 0, 0.40, 0.40, 0.85),
        (train, "04.png", 0, 0.08, 0.08, 0.75),
        (train, "05.png", 0, 0.60, 0.60, 0.92),
        (train, "06.png", 0, 0.45, 0.45, 0.70),
        (test, "07.png", 0, 0.70, 0.70, 0.88),
        (test, "08.png", 1, 0.50, 0.50, 0.90),
    ]
    hashes: dict[str, str] = {}
    for index, (folder, filename, class_id, width, height, confidence) in enumerate(specs):
        image_dir, label_dir = folder / "images", folder / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / filename
        Image.new("RGB", (64, 64), (25 + index * 23, 55 + index * 17, 180 - index * 13)).save(image_path)
        label_path = label_dir / f"{Path(filename).stem}.txt"
        label_path.write_text(
            f"{class_id} 0.5 0.5 {width} {height} {confidence}\n",
            encoding="utf-8",
        )
        hashes[str(image_path)] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        hashes[str(label_path)] = hashlib.sha256(label_path.read_bytes()).hexdigest()
    hashes[str(root / "classes.txt")] = hashlib.sha256((root / "classes.txt").read_bytes()).hexdigest()
    return [train, test], hashes


def _candidate_object_coverage_fixture(root: Path) -> tuple[list[Path], dict[str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("beta\nalpha\ngamma\n", encoding="utf-8")
    train, test = root / "train", root / "test"
    specs = [
        (train, "11.png", 1, 0.55, 0.55, 0.92),
        (train, "12.png", 2, 0.45, 0.40, 0.90),
        (test, "13.png", 0, 0.50, 0.50, 0.95),
    ]
    hashes: dict[str, str] = {}
    colors = [(235, 35, 45), (25, 225, 65), (30, 55, 240)]
    for index, (folder, filename, class_id, width, height, confidence) in enumerate(specs):
        image_dir, label_dir = folder / "images", folder / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / filename
        Image.new("RGB", (64, 64), colors[index]).save(image_path)
        label_path = label_dir / f"{Path(filename).stem}.txt"
        label_path.write_text(
            f"{class_id} 0.5 0.5 {width} {height} {confidence}\n",
            encoding="utf-8",
        )
        hashes[str(image_path)] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        hashes[str(label_path)] = hashlib.sha256(label_path.read_bytes()).hexdigest()
    hashes[str(root / "classes.txt")] = hashlib.sha256((root / "classes.txt").read_bytes()).hexdigest()
    return [train, test], hashes


def test_case_lv_web_03_obj_01_object_coverage_filters_keep_global_alignment(
    monkeypatch, tmp_path,
):
    """Object filters preserve global indices, supervised labels, class guards, and cropped readback."""
    folders, source_hashes = _object_coverage_fixture(tmp_path / "object-coverage")

    def object_loader(_model_id: str, **_kwargs):
        def embed(crop: Image.Image) -> np.ndarray:
            pixels = np.asarray(crop.convert("RGB"), dtype=np.float32)
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
            vector[3] = crop.width / 100.0
            vector[4] = crop.height / 100.0
            return vector
        return embed

    original_projection = __import__("react_web_shell").coverage_projection
    projected_labels: list[list[str]] = []

    def capture_projection(embeddings, labels, method, dimension):
        assert len(embeddings) == len(labels)
        projected_labels.append(list(labels))
        return original_projection(embeddings, labels, method, dimension)

    monkeypatch.setattr("react_web_shell.coverage_projection", capture_projection)
    source_root = folders[0].parent
    with _client(
        monkeypatch,
        tmp_path / "object-coverage-run",
        object_model_loader=object_loader,
    ) as client:
        job = _run_coverage(client, folders, unit="object", method="pca", k=10)
        assert job["state"] == "completed", job
        run_response = client.get(f"/api/v1/coverage/runs/{job['run_id']}")
        assert run_response.status_code == 200
        run = run_response.json()
        assert run["unit"] == "object"
        assert run["object_policy"] == {"pad": 0.12, "target_res": 224, "head": "cls"}
        assert len(run["records"]) == len(run["points"]) == 8
        assert [record["index"] for record in run["records"]] == list(range(8))
        assert client.post(
            f"/api/v1/coverage/runs/{job['run_id']}/completeness", json={},
        ).status_code == 409
        assert [record["score"] for record in run["records"]] == pytest.approx(
            [0.95, 0.40, 0.85, 0.75, 0.92, 0.70, 0.88, 0.90]
        )
        assert [record["short_px"] for record in run["records"]] == [32, 6, 26, 5, 38, 29, 45, 32]

        filtered_response = client.post(
            f"/api/v1/coverage/runs/{job['run_id']}/object-view",
            json={
                "class_label": None,
                "min_short_px": 10,
                "confidence_min": 0.8,
                "confidence_max": 0.95,
                "method": "lda",
                "dimension": 2,
                "k": 10,
                "block_k": 8,
            },
        )
        assert filtered_response.status_code == 200, filtered_response.text
        filtered = filtered_response.json()
        assert [record["index"] for record in filtered["records"]] == [0, 2, 4, 6, 7]
        assert [point["index"] for point in filtered["points"]] == [0, 2, 4, 6, 7]
        assert [record["label"] for record in filtered["records"]] == [
            "alpha", "alpha", "alpha", "alpha", "beta",
        ]
        assert filtered["method"] == "lda"
        assert projected_labels[-1] == ["alpha", "alpha", "alpha", "alpha", "beta"]
        assert filtered["k"] == 4
        assert [row["label"] for row in filtered["class_summary"]] == ["beta", "alpha"]
        assert [(row["label"], row["count"], row["analyzable"]) for row in filtered["class_summary"]] == [
            ("beta", 1, False), ("alpha", 4, True),
        ]

        alpha_response = client.post(
            f"/api/v1/coverage/runs/{job['run_id']}/object-view",
            json={
                "class_label": "alpha",
                "min_short_px": 10,
                "confidence_min": 0.8,
                "confidence_max": 0.95,
                "method": "pca",
                "dimension": 2,
                "k": 10,
                "block_k": 8,
            },
        )
        alpha = alpha_response.json()
        assert [record["index"] for record in alpha["records"]] == [0, 2, 4, 6]
        assert alpha["block_k"] == 4
        assert len(alpha["blocks"]) == 4
        assert all(point.get("block_id") is not None for point in alpha["points"])

        beta_response = client.post(
            f"/api/v1/coverage/runs/{job['run_id']}/object-view",
            json={
                "class_label": "beta",
                "min_short_px": 10,
                "confidence_min": 0.8,
                "confidence_max": 0.95,
                "method": "pca",
                "dimension": 2,
                "k": 10,
                "block_k": 8,
            },
        )
        beta = beta_response.json()
        assert [record["index"] for record in beta["records"]] == [7]
        assert beta["blocks"] == []
        assert any("fewer than two objects" in warning for warning in beta["warnings"])

        thumbnail = client.get(f"/api/v1/coverage/runs/{job['run_id']}/images/0/thumbnail")
        assert thumbnail.status_code == 200
        cropped = Image.open(BytesIO(thumbnail.content))
        assert cropped.size[0] < 64 and cropped.size[1] < 64
        invalid = client.post(
            f"/api/v1/coverage/runs/{job['run_id']}/object-view",
            json={"confidence_min": 0.9, "confidence_max": 0.1},
        )
        assert invalid.status_code == 422

    assert source_hashes == {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in source_hashes
    }


def test_case_lv_web_03_ref_01_candidate_pool_reference_and_class_name_alignment(
    monkeypatch, tmp_path,
):
    """Candidate mining and reference roles preserve taxonomy, indices, coverage and cart identity."""
    main_folders, main_hashes = _object_coverage_fixture(tmp_path / "reference-main")
    candidate_folders, candidate_hashes = _candidate_object_coverage_fixture(
        tmp_path / "reference-candidates",
    )

    def object_loader(_model_id: str, **_kwargs):
        def embed(crop: Image.Image) -> np.ndarray:
            pixels = np.asarray(crop.convert("RGB"), dtype=np.float32)
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
            vector[3] = crop.width / 100.0
            vector[4] = crop.height / 100.0
            return vector
        return embed

    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "candidate-cache"))
    app = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=lambda _model: _fake_embedder(),
        object_model_loader=object_loader,
    )
    with TestClient(app) as client:
        main_job = _run_coverage(client, main_folders, unit="object", method="pca", k=2)
        assert main_job["state"] == "completed", main_job
        run_id = main_job["run_id"]
        created = client.post(
            f"/api/v1/coverage/runs/{run_id}/candidates/jobs",
            json={"folders": [str(folder) for folder in candidate_folders]},
        )
        assert created.status_code == 202, created.text
        job_id = created.json()["job_id"]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            candidate_job = client.get(f"/api/v1/coverage/candidate-jobs/{job_id}").json()
            if candidate_job["state"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.02)
        else:
            pytest.fail("coverage candidate job did not finish within 15 seconds")
        assert candidate_job["state"] == "completed", candidate_job
        assert candidate_job["run_id"] == run_id

        base_request = {
            "role": "pool",
            "class_label": None,
            "min_short_px": 10,
            "confidence_min": 0.8,
            "confidence_max": 1.0,
            "method": "lda",
            "dimension": 2,
            "k": 2,
            "limit": 12,
        }
        pool_response = client.post(
            f"/api/v1/coverage/runs/{run_id}/candidate-view",
            json=base_request,
        )
        assert pool_response.status_code == 200, pool_response.text
        pool = pool_response.json()
        assert pool["role"] == "pool"
        assert pool["method"] == "lda"
        assert pool["candidate_count"] == 3
        assert [point["label"] for point in pool["candidate_points"]] == [
            "alpha", "gamma", "beta",
        ]
        assert [point["index"] for point in pool["candidate_points"]] == [0, 1, 2]
        assert all(point["provisional_label"] in {"alpha", "beta"}
                   for point in pool["candidate_points"])
        assert pool["work_indices"]
        assert pool["work_scores"]
        assert all(np.isfinite([point["x"], point["y"], point["score"]]).all()
                   for point in pool["candidate_points"])

        reference_response = client.post(
            f"/api/v1/coverage/runs/{run_id}/candidate-view",
            json={**base_request, "role": "reference"},
        )
        assert reference_response.status_code == 200, reference_response.text
        reference = reference_response.json()
        assert reference["recall"] is not None
        assert 0.0 <= reference["recall"] <= 1.0
        assert reference["uncovered_count"] == len(reference["work_indices"])
        gamma = next(row for row in reference["class_gaps"] if row["label"] == "gamma")
        assert gamma == {
            "label": "gamma",
            "main_count": 0,
            "candidate_count": 1,
            "uncovered_count": gamma["uncovered_count"],
            "state": "missing",
        }

        missing_class = client.post(
            f"/api/v1/coverage/runs/{run_id}/candidate-view",
            json={**base_request, "role": "reference", "class_label": "ghost"},
        )
        assert missing_class.status_code == 200, missing_class.text
        assert missing_class.json()["candidate_count"] == 0
        assert any("No candidate objects match ghost" in warning
                   for warning in missing_class.json()["warnings"])

        candidate_only_class = client.post(
            f"/api/v1/coverage/runs/{run_id}/candidate-view",
            json={**base_request, "role": "reference", "class_label": "gamma"},
        )
        assert candidate_only_class.status_code == 200, candidate_only_class.text
        candidate_only = candidate_only_class.json()
        assert candidate_only["main_count"] == 0
        assert candidate_only["candidate_count"] == 1
        assert candidate_only["recall"] is None
        assert candidate_only["candidate_points"][0]["provisional_label"] == ""
        assert any("coverage cannot be measured" in warning
                   for warning in candidate_only["warnings"])

        empty_reference = client.post(
            f"/api/v1/coverage/runs/{run_id}/candidate-view",
            json={**base_request, "role": "reference", "min_short_px": 1000},
        )
        assert empty_reference.status_code == 200, empty_reference.text
        assert empty_reference.json()["candidate_count"] == 0
        assert empty_reference.json()["recall"] is None
        assert any("no candidates match the current confidence" in warning.lower()
                   for warning in empty_reference.json()["warnings"])

        thumbnail = client.get(
            f"/api/v1/coverage/runs/{run_id}/candidates/0/thumbnail",
        )
        assert thumbnail.status_code == 200
        assert Image.open(BytesIO(thumbnail.content)).size[0] < 64
        added = client.post(
            f"/api/v1/coverage/runs/{run_id}/candidates/curation",
            json={"indices": [1], "source": "reference_class", "scores": {"1": 0.37}},
        )
        assert added.status_code == 201, added.text
        assert added.json()["added"] == 1
        item = added.json()["items"][0]
        assert item["label"] == "gamma"
        assert item["source"] == "reference_class"
        assert item["score"] == pytest.approx(0.37)
        assert item["level"] == "object"
        assert item["obj_index"] == 0

    for path, digest in {**main_hashes, **candidate_hashes}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


def test_case_lv_web_01_ic_01_discovery(classifier_fixture, monkeypatch, tmp_path):
    """AC1: React API preserves Streamlit classifier order, labels, and splits."""
    baseline = discover_images_classifier(classifier_fixture["folders"], allow_flat=True)
    baseline_readable = [
        record for record in baseline
        if Path(record["path"]).name != "broken.jpg"
    ]

    with _client(monkeypatch, tmp_path) as client:
        job = _run(client, classifier_fixture["folders"])
        assert job["state"] == "completed", job
        run = client.get(f"/api/v1/runs/{job['run_id']}").json()

    actual = [(r["filename"], r["split"], r["label"]) for r in run["records"]]
    expected = [(Path(r["path"]).name, r["split"], r["label"]) for r in baseline_readable]
    assert actual == expected
    assert run["embedding_shape"] == [4, 384]
    assert run["embedding_finite"] is True
    assert run["embedding_nonzero"] is True
    assert len(run["points"]) == 4
    assert all(np.isfinite([point["x"], point["y"]]).all() for point in run["points"])


def test_case_lv_web_01_ic_01_selected_image_is_run_bound(
    classifier_fixture, monkeypatch, tmp_path
):
    """AC3: a selected scatter point serves only its matching run record image."""
    with _client(monkeypatch, tmp_path) as client:
        job = _run(client, classifier_fixture["folders"])
        run = client.get(f"/api/v1/runs/{job['run_id']}").json()
        index = next(i for i, row in enumerate(run["records"]) if row["filename"] == "03.png")
        image = client.get(f"/api/v1/runs/{job['run_id']}/images/{index}")
        invalid = client.get(f"/api/v1/runs/{job['run_id']}/images/999")

    assert image.status_code == 200
    assert image.content == (classifier_fixture["root"] / "train" / "banana" / "03.png").read_bytes()
    assert invalid.status_code == 404
    assert set(invalid.json()) == {"code", "message", "request_id"}
    assert invalid.json()["code"] == "NOT_FOUND"


def test_case_lv_web_04_ctx_01_diagnostic_reference_tracks_source_and_run(
    classifier_fixture, monkeypatch, tmp_path
):
    calls: list[str] = []
    with _client(monkeypatch, tmp_path, embedder=_fake_embedder(calls)) as client:
        job = _run(client, classifier_fixture["folders"])
        assert job["state"] == "completed", job
        run_id = job["run_id"]
        run = client.get(f"/api/v1/runs/{run_id}").json()
        index = next(i for i, record in enumerate(run["records"]) if record["filename"] == "03.png")
        source = classifier_fixture["root"] / "train" / "banana" / "03.png"
        expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        inference_calls_after_run = len(calls)

        created = client.post("/api/v1/diagnostic-references", json={
            "run_id": run_id,
            "record_index": index,
            "model_id": "dinov2_vits14",
        })
        assert created.status_code == 201, created.text
        reference = created.json()
        assert reference["status"] == "current"
        assert reference["filename"] == "03.png"
        assert reference["label"] == "banana"
        assert reference["split"] == "train"
        assert reference["source_sha256"] == expected_hash
        assert "source_path" not in reference
        assert str(classifier_fixture["root"]) not in created.text

        current = client.get(
            f"/api/v1/diagnostic-references/{reference['context_id']}",
            params={"current_run_id": run_id},
        ).json()
        assert current["status"] == "current"

        source.write_bytes(source.read_bytes() + b"changed after the Visualize run")
        stale = client.get(
            f"/api/v1/diagnostic-references/{reference['context_id']}",
            params={"current_run_id": run_id},
        ).json()
        incompatible = client.get(
            f"/api/v1/diagnostic-references/{reference['context_id']}",
            params={"current_run_id": "new-visualize-run"},
        ).json()
        invalid_index = client.post("/api/v1/diagnostic-references", json={
            "run_id": run_id,
            "record_index": 999,
            "model_id": "dinov2_vits14",
        })
        public_run = client.get(f"/api/v1/runs/{run_id}").json()

    assert stale["status"] == "stale"
    assert "SHA-256 changed" in stale["reason"]
    assert incompatible["status"] == "incompatible"
    assert invalid_index.status_code == 422
    assert invalid_index.json()["code"] == "INVALID_RECORD_INDEX"
    assert "_source_sha256" not in public_run
    assert len(calls) == inference_calls_after_run


def test_case_lv_web_04_head_01_exact_resolution_head_build_and_reuse(
    classifier_fixture, monkeypatch, tmp_path
):
    diagnostic_calls: list[tuple[int, str, bool, str]] = []

    def diagnostic_loader(model_id: str, resolution: int, keep_aspect: bool, pooling: str):
        diagnostic_calls.append((resolution, model_id, keep_aspect, pooling))

        def embed(path: Path) -> np.ndarray:
            pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = pixels.mean(axis=(0, 1)) / 255.0
            vector[3] = resolution / 448.0
            return vector

        return embed

    with _client(
        monkeypatch,
        tmp_path,
        diagnostic_model_loader=diagnostic_loader,
    ) as client:
        job = _run(client, classifier_fixture["folders"])
        assert job["state"] == "completed", job
        run_id = job["run_id"]
        run = client.get(f"/api/v1/runs/{run_id}").json()
        reference = client.post("/api/v1/diagnostic-references", json={
            "run_id": run_id,
            "record_index": 0,
            "model_id": "dinov2_vits14",
        }).json()
        context_id = reference["context_id"]

        before = client.get(
            f"/api/v1/diagnostic-references/{context_id}/heads/224",
            params={"current_run_id": run_id},
        ).json()
        head_224 = client.post(
            f"/api/v1/diagnostic-references/{context_id}/heads",
            json={"current_run_id": run_id, "resolution": 224},
        )
        reused_224 = client.post(
            f"/api/v1/diagnostic-references/{context_id}/heads",
            json={"current_run_id": run_id, "resolution": 224},
        ).json()
        head_336 = client.post(
            f"/api/v1/diagnostic-references/{context_id}/heads",
            json={"current_run_id": run_id, "resolution": 336},
        ).json()
        head_448 = client.post(
            f"/api/v1/diagnostic-references/{context_id}/heads",
            json={"current_run_id": run_id, "resolution": 448},
        ).json()
        ready_336 = client.get(
            f"/api/v1/diagnostic-references/{context_id}/heads/336",
            params={"current_run_id": run_id},
        ).json()
        incompatible = client.post(
            f"/api/v1/diagnostic-references/{context_id}/heads",
            json={"current_run_id": "different-run", "resolution": 224},
        )
        public_run = client.get(f"/api/v1/runs/{run_id}").json()
        replacement_run = _run(client, classifier_fixture["folders"])
        old_run_status = client.get(
            f"/api/v1/diagnostic-references/{context_id}/heads/224",
            params={"current_run_id": run_id},
        ).json()

    assert replacement_run["state"] == "completed", replacement_run
    assert old_run_status["status"] == "incompatible"
    assert before.get("status") == "not_ready", before
    assert head_224.status_code == 200, head_224.text
    assert head_224.json()["status"] == "ready"
    assert head_224.json()["resolution"] == 224
    assert head_224.json()["training_samples"] == len(run["records"])
    assert set(head_224.json()["classes"]) == {"apple", "banana"}
    assert reused_224["reused"] is True
    assert head_336["resolution"] == 336
    assert head_448["resolution"] == 448
    assert ready_336["status"] == "ready"
    assert ready_336["reused"] is True
    assert diagnostic_calls.count((336, "dinov2_vits14", False, "cls")) == 1
    assert diagnostic_calls.count((448, "dinov2_vits14", False, "cls")) == 1
    assert all(call[0] in {336, 448} for call in diagnostic_calls)
    assert incompatible.status_code == 409
    assert incompatible.json()["code"] == "INCOMPATIBLE_DIAGNOSTIC_RUN"
    assert "_source_sha256" not in public_run


def test_case_lv_web_04_expl_01_explicit_attention_and_signed_gradcam_evidence(
    classifier_fixture, monkeypatch, tmp_path
):
    """Explanation is explicit, tied to the exact head, and never exposes paths or masks."""
    import dino_explain
    import gradcam_explain

    attention_calls = []
    gradcam_calls = []
    checkpoint = classifier_fixture["root"] / "fixture-checkpoint.pth"
    checkpoint.write_bytes(b"test checkpoint identity")

    class FakeExplainer:
        pass

    def fake_attention(request, *, explainer=None):
        attention_calls.append(request)
        return types.SimpleNamespace(
            raw_patch_scores=np.array([[0.2, 0.8], [0.1, 0.4]], dtype=np.float32),
            display_patch_scores=np.array([[0.25, 1.0], [0.125, 0.5]], dtype=np.float32),
            grid=(2, 2),
            patch_size=14,
            method=request.method,
            input_image=request.image,
            warnings=("Attention is not a defect mask.",),
        )

    def fake_gradcam(request, head, *, target_class, explainer=None):
        gradcam_calls.append((request, target_class, head, explainer))
        return types.SimpleNamespace(
            raw_patch_scores=np.array([[-1.0, 2.0], [0.5, -0.25]], dtype=np.float32),
            display_patch_scores=np.array([[-0.5, 1.0], [0.25, -0.125]], dtype=np.float32),
            grid=(2, 2),
            patch_size=14,
            input_image=request.image,
            target_class=target_class or "banana",
            target_score=0.75,
            reference_class="apple",
            predicted_class="banana",
            predicted_probability=0.7,
            runner_up_class="apple",
            runner_up_probability=0.3,
            probability_margin=0.4,
            target_probability=0.7,
            reference_probability=0.3,
            is_predicted_target=(target_class in (None, "banana")),
            positive_mass=2.5,
            negative_mass=1.25,
            faithfulness={"available": True, "status": "supported"},
            head_diagnostics={"available": False},
            sample_oof=None,
            warnings=("Signed evidence is not segmentation.",),
        )

    monkeypatch.setattr(dino_explain, "_resolve_checkpoint", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(
        dino_explain,
        "get_explainer",
        lambda *_args, **_kwargs: (FakeExplainer(), checkpoint),
    )
    monkeypatch.setattr(dino_explain, "attention_map", fake_attention)
    monkeypatch.setattr(gradcam_explain, "token_gradcam_map", fake_gradcam)

    def diagnostic_loader(model_id: str, resolution: int, keep_aspect: bool, pooling: str):
        def embed(path: Path) -> np.ndarray:
            pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = pixels.mean(axis=(0, 1)) / 255.0
            vector[3] = resolution / 448.0
            return vector
        return embed

    with _client(
        monkeypatch,
        tmp_path,
        diagnostic_model_loader=diagnostic_loader,
    ) as client:
        job = _run(client, classifier_fixture["folders"])
        run_id = job["run_id"]
        run = client.get(f"/api/v1/runs/{run_id}").json()
        reference = client.post("/api/v1/diagnostic-references", json={
            "run_id": run_id, "record_index": 0, "model_id": "dinov2_vits14",
        }).json()
        context_id = reference["context_id"]
        unavailable = client.post(
            f"/api/v1/diagnostic-references/{context_id}/explanations",
            json={
                "current_run_id": run_id, "resolution": 224,
                "method": "attention",
            },
        )
        head = client.post(
            f"/api/v1/diagnostic-references/{context_id}/heads",
            json={"current_run_id": run_id, "resolution": 224},
        )
        source_path = classifier_fixture["root"] / run["records"][0]["split"] / run["records"][0]["label"] / run["records"][0]["filename"]
        source_before = hashlib.sha256(source_path.read_bytes()).hexdigest()
        attention = client.post(
            f"/api/v1/diagnostic-references/{context_id}/explanations",
            json={
                "current_run_id": run_id,
                "resolution": 224,
                "method": "attention",
                "attention_method": "single_head",
                "head_index": 2,
            },
        )
        gradcam = client.post(
            f"/api/v1/diagnostic-references/{context_id}/explanations",
            json={
                "current_run_id": run_id,
                "resolution": 224,
                "method": "gradcam",
                "target_class": "apple",
                "display_mode": "positive",
            },
        )
        incompatible = client.post(
            f"/api/v1/diagnostic-references/{context_id}/explanations",
            json={
                "current_run_id": "different-run", "resolution": 224,
                "method": "attention",
            },
        )
        source_after = hashlib.sha256(source_path.read_bytes()).hexdigest()

    assert unavailable.status_code == 409
    assert unavailable.json()["code"] == "DIAGNOSTIC_HEAD_NOT_READY"
    assert head.status_code == 200, head.text
    assert attention.status_code == 200, attention.text
    assert gradcam.status_code == 200, gradcam.text
    assert incompatible.status_code == 409
    assert incompatible.json()["code"] == "INCOMPATIBLE_DIAGNOSTIC_RUN"
    assert attention.json()["method"] == "attention"
    assert attention.json()["attention_method"] == "single_head"
    assert attention.json()["grid"] == [2, 2]
    assert attention_calls[0].head_index == 2
    assert attention_calls[0].method == "single_head"
    assert "mask" in attention.json()["warning"]
    gradcam_payload = gradcam.json()
    assert gradcam_payload["target_class"] == "apple"
    assert gradcam_payload["reference_class"] == "apple"
    assert gradcam_payload["score_mode"] == "positive"
    assert gradcam_payload["display_scores"] == [[0.0, 1.0], [0.25, 0.0]]
    assert gradcam_payload["faithfulness"]["status"] == "supported"
    assert gradcam_calls[0][1] == "apple"
    assert str(classifier_fixture["root"]) not in attention.text + gradcam.text
    assert "image_path" not in gradcam_payload and "source_path" not in gradcam_payload
    assert gradcam_payload["source_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    input_png = Image.open(BytesIO(base64.b64decode(gradcam_payload["model_input_png_base64"])))
    overlay_png = Image.open(BytesIO(base64.b64decode(gradcam_payload["overlay_png_base64"])))
    assert input_png.format == overlay_png.format == "PNG"
    assert input_png.size == overlay_png.size == (32, 32)
    assert source_before == source_after


def test_case_lv_web_01_rank_01_legacy_scores_stable_order_filters_and_csv(
    classifier_fixture, monkeypatch, tmp_path
):
    """Ranking values match LV's legacy helpers and exports the displayed order."""
    with _client(monkeypatch, tmp_path) as client:
        job = _run(client, classifier_fixture["folders"])
        run_id = job["run_id"]
        run_response = client.get(f"/api/v1/runs/{run_id}")
        run = run_response.json()
        vectors = client.app.state.raw_embeddings[run_id]["dinov2_vits14"]
        labels = [record["label"] for record in run["records"]]
        expected_scores = {
            "outlier": compute_outlier_scores(
                vectors, vectors, k=min(5, len(vectors) - 1), candidates_in_reference=True,
            ),
            "disagreement": compute_label_disagreement(
                vectors, labels, k=min(5, len(vectors) - 1),
            ),
        }

        for metric, expected in expected_scores.items():
            response = client.get(
                f"/api/v1/runs/{run_id}/rankings",
                params={"model_id": "dinov2_vits14", "metric": metric},
            )
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["metric"] == metric
            assert payload["score_direction"] == "descending"
            assert "not a quality score" in payload["interpretation"] or "do not confirm a mislabel" in payload["interpretation"]
            expected_order = sorted(range(len(expected)), key=lambda index: (-float(expected[index]), index))
            assert [row["index"] for row in payload["results"]] == expected_order
            assert [row["rank"] for row in payload["results"]] == list(range(1, len(expected) + 1))
            np.testing.assert_allclose(
                [row["score"] for row in payload["results"]],
                [expected[index] for index in expected_order],
                rtol=0,
                atol=1e-12,
            )

            filtered = client.get(
                f"/api/v1/runs/{run_id}/rankings",
                params={
                    "model_id": "dinov2_vits14", "metric": metric,
                    "label_filter": "apple", "split_filter": "train",
                },
            ).json()
            expected_filtered = [index for index in expected_order if labels[index] == "apple" and run["records"][index]["split"] == "train"]
            assert [row["index"] for row in filtered["results"]] == expected_filtered
            assert [row["rank"] for row in filtered["results"]] == list(range(1, len(expected_filtered) + 1))

            exported = client.get(
                f"/api/v1/runs/{run_id}/rankings.csv",
                params={
                    "model_id": "dinov2_vits14", "metric": metric,
                    "label_filter": "apple", "split_filter": "train",
                },
            )
            rows = list(csv.DictReader(StringIO(exported.text)))
            assert [int(row["index"]) for row in rows] == expected_filtered
            assert [int(row["rank"]) for row in rows] == list(range(1, len(expected_filtered) + 1))
            assert str(classifier_fixture["root"]) not in exported.text

        assert "_ranking_scores" not in run
        assert "raw_embeddings" not in run_response.text
        assert client.get(
            f"/api/v1/runs/{run_id}/rankings",
            params={"model_id": "missing-model", "metric": "outlier"},
        ).status_code == 404


def test_case_lv_web_01_search_01_cosine_ranking_model_switch_and_csv(
    classifier_fixture, monkeypatch, tmp_path
):
    """F3 API ranks by the active model, excludes the query, and exports ranked rows."""
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "similar-search-cache"))
    model_vectors = {
        "model-a": {
            "01.jpg": [1.0, 0.0], "02.jpg": [0.8, 0.6],
            "03.png": [0.0, 1.0], "04.jpeg": [-1.0, 0.0],
        },
        "model-b": {
            "01.jpg": [1.0, 0.0], "02.jpg": [-1.0, 0.0],
            "03.png": [0.8, 0.6], "04.jpeg": [0.0, 1.0],
        },
    }

    def loader(model_id: str):
        def embed(path: Path) -> np.ndarray:
            return np.asarray(model_vectors[model_id][path.name], dtype=np.float32)
        return embed

    with TestClient(create_app(
        model_catalog=lambda: list(model_vectors), model_loader=loader,
    )) as client:
        job = _run_custom(client, {
            "folders": [str(folder) for folder in classifier_fixture["folders"]],
            "model_ids": list(model_vectors),
            "methods": ["pca"],
        })
        assert job["state"] == "completed", job
        run_id = job["run_id"]
        run = client.get(f"/api/v1/runs/{run_id}").json()
        first_model = client.post(f"/api/v1/runs/{run_id}/similar", json={
            "model_id": "model-a", "query_index": 0, "k": 9,
        })
        second_model = client.post(f"/api/v1/runs/{run_id}/similar", json={
            "model_id": "model-b", "query_index": 0, "k": 9,
        })
        csv_response = client.post(f"/api/v1/runs/{run_id}/similar.csv", json={
            "model_id": "model-a", "query_index": 0, "k": 9,
        })
        missing_model = client.post(f"/api/v1/runs/{run_id}/similar", json={
            "model_id": "missing", "query_index": 0, "k": 9,
        })
        invalid_index = client.post(f"/api/v1/runs/{run_id}/similar", json={
            "model_id": "model-a", "query_index": 99, "k": 9,
        })

    ranked_a = first_model.json()["results"]
    ranked_b = second_model.json()["results"]
    assert [item["index"] for item in ranked_a] == [1, 2, 3]
    assert [item["distance"] for item in ranked_a] == sorted(item["distance"] for item in ranked_a)
    assert ranked_a[0]["distance"] == pytest.approx(0.2)
    assert ranked_a[-1]["distance"] == pytest.approx(2.0)
    assert [item["index"] for item in ranked_b] == [2, 3, 1]
    assert 0 not in [item["index"] for item in ranked_a]
    assert run["records"][0]["filename"] == "01.jpg"
    assert "raw_embeddings" not in run
    rows = list(csv.reader(csv_response.text.splitlines()))
    assert csv_response.status_code == 200
    assert rows[0] == ["index", "filename", "path", "label", "split"]
    assert [row[1] for row in rows[1:]] == ["01.jpg", "02.jpg", "03.png", "04.jpeg"]
    assert missing_model.status_code == 404
    assert invalid_index.status_code == 422


def test_case_lv_web_01_search_02_text_search_and_duplicate_leakage_review(
    classifier_fixture, monkeypatch, tmp_path
):
    """Chinese-CLIP text ranking and both duplicate methods preserve run order and split filters."""
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "search-02-cache"))
    model_vectors = {
        "01.jpg": [1.0, 0.0],
        "02.jpg": [0.0, 1.0],
        "03.png": [-1.0, 0.0],
        "04.jpeg": [0.998, 0.063],
    }
    encoded_queries: list[str] = []

    def model_loader(_model_id: str):
        def embed(path: Path) -> np.ndarray:
            return np.asarray(model_vectors[path.name], dtype=np.float32)
        return embed

    def text_model_loader(_model_id: str):
        def encode(query: str) -> np.ndarray:
            encoded_queries.append(query)
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return encode

    with TestClient(create_app(
        model_catalog=lambda: ["chinese-clip-fixture"],
        model_loader=model_loader,
        text_model_loader=text_model_loader,
    )) as client:
        job = _run_custom(client, {
            "folders": [str(folder) for folder in classifier_fixture["folders"]],
            "model_ids": ["chinese-clip-fixture"],
            "methods": ["pca"],
        })
        assert job["state"] == "completed", job
        run_id = job["run_id"]
        run = client.get(f"/api/v1/runs/{run_id}").json()
        text_result = client.post(f"/api/v1/runs/{run_id}/text-search", json={
            "model_id": "chinese-clip-fixture", "query": "  紅色樣本  ", "k": 4,
        })
        text_csv = client.post(f"/api/v1/runs/{run_id}/text-search.csv", json={
            "model_id": "chinese-clip-fixture", "query": "紅色樣本", "k": 4,
        })
        phash_result = client.post(f"/api/v1/runs/{run_id}/duplicates", json={
            "method": "phash", "max_hamming": 0, "cross_split_only": True,
        })
        embedding_result = client.post(f"/api/v1/runs/{run_id}/duplicates", json={
            "method": "embedding", "model_id": "chinese-clip-fixture",
            "max_distance": 0.05, "cross_split_only": True,
        })
        duplicate_csv = client.post(f"/api/v1/runs/{run_id}/duplicates.csv", json={
            "method": "embedding", "model_id": "chinese-clip-fixture",
            "max_distance": 0.05, "cross_split_only": True,
        })
        unsupported_model = client.post(f"/api/v1/runs/{run_id}/text-search", json={
            "model_id": "dinov2_vits14", "query": "red", "k": 2,
        })
        blank_query = client.post(f"/api/v1/runs/{run_id}/text-search", json={
            "model_id": "chinese-clip-fixture", "query": "   ", "k": 2,
        })

    assert [item["index"] for item in text_result.json()["results"]] == [0, 3, 1, 2]
    distances = [item["distance"] for item in text_result.json()["results"]]
    assert distances == sorted(distances)
    assert text_result.json()["query"] == "紅色樣本"
    assert encoded_queries == ["紅色樣本"]
    assert "_phashes" not in run and "image_paths" not in run and "_raw_embeddings" not in run
    assert all("path" not in item for item in text_result.json()["results"])
    text_rows = list(csv.reader(text_csv.text.splitlines()))
    assert text_csv.status_code == 200
    assert [row[1] for row in text_rows[1:]] == [run["records"][i]["filename"] for i in [0, 3, 1, 2]]

    phash_pairs = phash_result.json()["results"]
    assert [(item["left_index"], item["right_index"]) for item in phash_pairs] == [(0, 3), (1, 3), (2, 3)]
    embedding_pairs = embedding_result.json()["results"]
    assert all("path" not in item for item in embedding_pairs)
    assert [(item["left_index"], item["right_index"]) for item in embedding_pairs] == [(0, 3)]
    assert embedding_pairs[0]["distance"] == pytest.approx(
        1 - 0.998 / np.sqrt(0.998**2 + 0.063**2), abs=5e-7,
    )
    duplicate_rows = list(csv.reader(duplicate_csv.text.splitlines()))
    assert duplicate_csv.status_code == 200
    assert duplicate_rows[0][:4] == ["left_index", "left_filename", "left_path", "left_split"]
    assert len(duplicate_rows) == 2
    assert unsupported_model.status_code == 409
    assert blank_query.status_code == 422
    for path, expected_hash in classifier_fixture["hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_hash


def test_case_lv_web_01_ic_01_manifest_cache_and_source_immutability(
    classifier_fixture, monkeypatch, tmp_path
):
    """AC5: manifest SHA keys reuse embeddings without writing to source folders."""
    calls: list[str] = []
    with _client(monkeypatch, tmp_path, embedder=_fake_embedder(calls)) as client:
        first = _run(client, classifier_fixture["folders"])
        second = _run(client, classifier_fixture["folders"])

    assert first["state"] == second["state"] == "completed"
    assert len(calls) == 4
    for path, expected_hash in classifier_fixture["hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_hash
    source_files = {p.resolve() for p in classifier_fixture["root"].rglob("*") if p.is_file()}
    assert source_files == {
        Path(path).resolve() for path in classifier_fixture["hashes"]
    } | {(classifier_fixture["root"] / "train" / "apple" / "broken.jpg").resolve()}
    assert not (classifier_fixture["root"] / "train" / "manifest.jsonl").exists()


def test_case_lv_web_01_ic_01_changed_content_invalidates_embedding(
    classifier_fixture, monkeypatch, tmp_path
):
    """AC5: a changed image with the same path cannot reuse its old embedding."""
    calls: list[str] = []
    image_path = classifier_fixture["root"] / "train" / "apple" / "01.jpg"
    with _client(monkeypatch, tmp_path, embedder=_fake_embedder(calls)) as client:
        first = _run(client, classifier_fixture["folders"])
        old_stat = image_path.stat()
        Image.new("RGB", (37, 35), (15, 230, 175)).save(image_path)
        os.utime(image_path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 2_000_000_000))
        second = _run(client, classifier_fixture["folders"])

    assert first["state"] == second["state"] == "completed"
    assert len(calls) == 5


def test_case_lv_web_01_ic_01_empty_dataset_and_missing_model_are_explicit(
    monkeypatch, tmp_path
):
    empty = tmp_path / "empty"
    empty.mkdir()
    with _client(monkeypatch, tmp_path / "empty-run") as client:
        empty_job = _run(client, [empty])
    assert empty_job["state"] == "failed"
    assert empty_job["error"]["code"] == "NO_IMAGES"

    dataset = tmp_path / "data" / "apple"
    dataset.mkdir(parents=True)
    Image.new("RGB", (16, 16), (20, 40, 60)).save(dataset / "sample.png")
    with _client(monkeypatch, tmp_path / "missing-model", models=[]) as client:
        missing_job = _run(client, [dataset.parent])
    assert missing_job["state"] == "failed"
    assert missing_job["error"]["code"] == "MODEL_UNAVAILABLE"


def test_case_lv_web_01_ic_01_cancel_stops_after_current_image(
    classifier_fixture, monkeypatch, tmp_path
):
    """AC4: cancel is observable and prevents the worker from reporting success."""
    started, release = Event(), Event()

    def slow_embed(path: Path) -> np.ndarray:
        started.set()
        release.wait(timeout=5)
        return _fake_embedder()(path)

    with _client(monkeypatch, tmp_path, embedder=slow_embed) as client:
        created = client.post("/api/v1/visualize/jobs", json={
            "folders": [str(folder) for folder in classifier_fixture["folders"]],
            "model_id": "dinov2_vits14",
            "method": "pca",
        })
        assert created.status_code == 202
        job_id = created.json()["job_id"]
        assert started.wait(timeout=3)
        assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 202
        release.set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = client.get(f"/api/v1/jobs/{job_id}").json()
            if status["state"] in {"cancelled", "completed", "failed"}:
                break
            time.sleep(0.02)

    assert status["state"] == "cancelled"

def _run_custom(client: TestClient, payload: dict) -> dict:
    response = client.post("/api/v1/visualize/jobs", json=payload)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        status = client.get("/api/v1/jobs/" + job_id).json()
        if status["state"] in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(0.02)
    pytest.fail("configured visualization job did not finish within 120 seconds")


def test_case_lv_web_01_ic_02_multi_model_projection_views(
    classifier_fixture, monkeypatch, tmp_path
):
    """Selected models and projections keep one stable record order and separate caches."""
    calls: list[str] = []
    selected_models = ["dinov2_vits14", "resnet50"]
    selected_methods = ["pca", "tsne", "umap", "lda", "sumap"]

    with _client(
        monkeypatch, tmp_path, embedder=_fake_embedder(calls), models=selected_models
    ) as client:
        job = _run_custom(client, {
            "folders": [str(folder) for folder in classifier_fixture["folders"]],
            "model_ids": selected_models,
            "methods": selected_methods,
        })
        assert job["state"] == "completed", job
        run = client.get("/api/v1/runs/" + job["run_id"]).json()

    assert run["models"] == selected_models
    assert run["embedding_shapes"] == {
        "dinov2_vits14": [4, 384],
        "resnet50": [4, 384],
    }
    assert len(calls) == 8
    view_keys = {(view["model_id"], view["method"]) for view in run["views"]}
    for model_id in selected_models:
        assert (model_id, "pca") in view_keys
        assert (model_id, "tsne") in view_keys
        assert (model_id, "lda") in view_keys
        for method in ("umap", "sumap"):
            if (model_id, method) not in view_keys:
                assert any(method in warning.lower() for warning in run["warnings"])
    for view in run["views"]:
        assert len(view["points"]) == len(run["records"]) == 4
        assert view["embedding_shape"] == [4, 384]
        assert view["embedding_finite"] is True
        assert view["embedding_nonzero"] is True
        assert all(np.isfinite([point["x"], point["y"], point["z"]]).all()
                   for point in view["points"])
        assert [point["index"] for point in view["points"]] == list(range(4))
    assert run["model_id"] == selected_models[0]
    assert run["method"] == "pca"


def test_case_lv_web_01_ic_03_fixed_umap_reference_frame(
    classifier_fixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Known images stay fixed; new images transform into the frame; rebuild refits."""
    fake_umap = types.ModuleType("umap")
    fake_umap.UMAP = _DeterministicUMAP
    monkeypatch.setitem(sys.modules, "umap", fake_umap)
    if "umap_ref" in sys.modules:
        monkeypatch.setattr(sys.modules["umap_ref"], "umap", fake_umap)
    _DeterministicUMAP.fit_calls = 0
    _DeterministicUMAP.transform_calls = 0

    payload = {
        "folders": [str(folder) for folder in classifier_fixture["folders"]],
        "model_ids": ["dinov2_vits14"],
        "methods": ["umap"],
        "umap_reference": True,
    }

    def run(client: TestClient, request: dict = payload) -> dict:
        job = _run_custom(client, request)
        assert job["state"] == "completed", job
        return client.get("/api/v1/runs/" + job["run_id"]).json()

    def umap_points(result: dict) -> dict[str, tuple[float, float, float]]:
        view = next(view for view in result["views"] if view["method"] == "umap")
        return {
            point["filename"]: (point["x"], point["y"], point["z"])
            for point in view["points"]
        }

    with _client(monkeypatch, tmp_path, embedder=_fake_embedder()) as client:
        first = run(client)
        assert first["umap_reference"] == {
            "enabled": True,
            "by_model": {"dinov2_vits14": {"refitted": True, "new_points": 0}},
        }
        first_points = umap_points(first)

        repeated = run(client)
        assert repeated["umap_reference"]["by_model"]["dinov2_vits14"] == {
            "refitted": False, "new_points": 0,
        }
        assert umap_points(repeated) == first_points

        new_image = classifier_fixture["folders"][0] / "apple" / "99.png"
        Image.new("RGB", (32, 32), (180, 90, 35)).save(new_image)
        extended = run(client)
        extension_status = extended["umap_reference"]["by_model"]["dinov2_vits14"]
        assert extension_status == {"refitted": False, "new_points": 1}
        extended_points = umap_points(extended)
        assert all(extended_points[name] == coords for name, coords in first_points.items())
        assert "99.png" in extended_points
        assert _DeterministicUMAP.transform_calls == 1

        new_image.unlink()
        reduced = run(client)
        assert reduced["umap_reference"]["by_model"]["dinov2_vits14"] == {
            "refitted": False, "new_points": 0,
        }
        assert umap_points(reduced) == first_points

        reference_files = list((tmp_path.parent / f"{tmp_path.name}-lv-cache").rglob("umap_ref.pkl"))
        assert len(reference_files) == 1
        assert not reference_files[0].is_relative_to(classifier_fixture["root"])

        rebuilt = run(client, {**payload, "umap_rebuild": True})
        assert rebuilt["umap_reference"]["by_model"]["dinov2_vits14"] == {
            "refitted": True, "new_points": 0,
        }
        assert _DeterministicUMAP.fit_calls == 2


def test_case_lv_web_01_ic_03_real_umap_reference_integration(
    classifier_fixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The real reducer keeps its first coordinate dimensionality as data grows."""
    try:
        import umap  # noqa: F401
    except ImportError:
        pytest.skip("install the declared umap-learn dependency for real reducer validation")

    payload = {
        "folders": [str(folder) for folder in classifier_fixture["folders"]],
        "model_ids": ["dinov2_vits14"],
        "methods": ["umap"],
        "umap_reference": True,
    }

    def run(client: TestClient, request: dict = payload) -> dict:
        job = _run_custom(client, request)
        assert job["state"] == "completed", job
        return client.get("/api/v1/runs/" + job["run_id"]).json()

    def umap_points(result: dict) -> dict[str, np.ndarray]:
        view = next(view for view in result["views"] if view["method"] == "umap")
        return {
            point["filename"]: np.asarray([point["x"], point["y"], point["z"]])
            for point in view["points"]
        }

    with _client(monkeypatch, tmp_path, embedder=_fake_embedder()) as client:
        first = run(client)
        assert first["umap_reference"]["by_model"]["dinov2_vits14"]["refitted"] is True
        first_points = umap_points(first)
        assert all(np.isfinite(point).all() for point in first_points.values())

        new_image = classifier_fixture["folders"][0] / "apple" / "99.png"
        Image.new("RGB", (32, 32), (180, 90, 35)).save(new_image)
        extended = run(client)
        assert extended["umap_reference"]["by_model"]["dinov2_vits14"] == {
            "refitted": False, "new_points": 1,
        }
        extended_points = umap_points(extended)
        assert all(np.array_equal(extended_points[name], coords)
                   for name, coords in first_points.items())
        assert np.isfinite(extended_points["99.png"]).all()

        new_image.unlink()
        reduced = run(client)
        assert reduced["umap_reference"]["by_model"]["dinov2_vits14"] == {
            "refitted": False, "new_points": 0,
        }
        assert all(np.array_equal(umap_points(reduced)[name], coords)
                   for name, coords in first_points.items())

        rebuilt = run(client, {**payload, "umap_rebuild": True})
        assert rebuilt["umap_reference"]["by_model"]["dinov2_vits14"]["refitted"] is True


def test_case_lv_web_01_ic_04_kmeans_blocks_and_pca_drilldown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """KMeans block IDs use the selected 3D view and drilldown projects private raw vectors."""
    folder = tmp_path / "classifier-blocks" / "train" / "apple"
    folder.mkdir(parents=True)
    for cluster in range(4):
        for replicate in range(3):
            Image.new("RGB", (24, 24), (30 + cluster * 45 + replicate, 40 + replicate * 20, 120)).save(
                folder / f"cluster-{cluster}-{replicate}.png"
            )

    def clustered_embedder(_path: Path) -> np.ndarray:
        name = _path.stem.split("-")
        cluster, replicate = int(name[1]), int(name[2])
        vector = np.zeros(384, dtype=np.float32)
        vector[cluster] = 10.0
        vector[4] = replicate * 0.1 + 0.01
        vector[5] = (cluster + 1) * (replicate + 1) * 0.01
        return vector

    with _client(
        monkeypatch, tmp_path / "block-api", embedder=clustered_embedder
    ) as client:
        job = _run(client, [folder.parent])
        assert job["state"] == "completed", job
        run = client.get(f"/api/v1/runs/{job['run_id']}").json()
        clustered = client.post(f"/api/v1/runs/{job['run_id']}/blocks", json={
            "model_id": "dinov2_vits14",
            "method": "pca",
            "k": 4,
            "label_filter": "apple",
            "split_filter": "train",
        })
        assert clustered.status_code == 200, clustered.text
        blocks = clustered.json()
        selected_block = max(blocks["centers"], key=lambda center: center["count"])["block_id"]
        drilldown = client.post(f"/api/v1/runs/{job['run_id']}/blocks", json={
            "model_id": "dinov2_vits14",
            "method": "pca",
            "k": 4,
            "label_filter": "apple",
            "split_filter": "train",
            "block_id": selected_block,
        })
        invalid_model = client.post(f"/api/v1/runs/{job['run_id']}/blocks", json={
            "model_id": "not-selected", "method": "pca", "k": 4,
        })

    assert "_raw_embeddings" not in run and "raw_embeddings" not in run
    assert blocks["k"] == 4
    assert len(blocks["centers"]) == 4
    assert len(blocks["points"]) == 12
    assert sum(center["count"] for center in blocks["centers"]) == 12
    assert all(np.isfinite([center["x"], center["y"], center["z"]]).all()
               for center in blocks["centers"])
    assert drilldown.status_code == 200, drilldown.text
    drilldown_points = drilldown.json()["drilldown_points"]
    expected_members = {
        point["index"] for point in blocks["points"]
        if point["block_id"] == selected_block
    }
    assert {point["index"] for point in drilldown_points} == expected_members
    assert len(drilldown_points) > 2
    assert all(np.isfinite([point["x"], point["y"]]).all() for point in drilldown_points)

    assert invalid_model.status_code == 404


def test_case_lv_web_01_ic_02_single_class_supervision_falls_back_to_pca(
    monkeypatch, tmp_path
):
    """Supervised-only selection with one class remains usable and labels the fallback."""
    folder = tmp_path / "flat-one-class"
    folder.mkdir()
    for index, color in enumerate(((200, 20, 30), (30, 200, 40), (40, 50, 210), (210, 190, 20))):
        Image.new("RGB", (24, 24), color).save(folder / ("image-" + str(index) + ".png"))

    with _client(monkeypatch, tmp_path, models=["dinov2_vits14"]) as client:
        job = _run_custom(client, {
            "folders": [str(folder)],
            "model_ids": ["dinov2_vits14"],
            "methods": ["lda", "sumap"],
        })
        assert job["state"] == "completed", job
        run = client.get("/api/v1/runs/" + job["run_id"]).json()

    assert run["methods"] == ["pca"]
    assert len(run["points"]) == 4
    assert any("lda skipped because at least two classes are required" in warning
               for warning in run["warnings"])
    assert any("sumap skipped because at least two classes are required" in warning
               for warning in run["warnings"])
    assert any("PCA was used" in warning for warning in run["warnings"])


def test_case_lv_web_01_edge_01_supervised_projection_skips_small_sample(
    monkeypatch, tmp_path
):
    """Legacy supervised projections need at least four images, even with two classes."""
    folder = tmp_path / "small-two-class"
    folder.mkdir()
    for index, (label, color) in enumerate((
        ("apple", (200, 20, 30)),
        ("apple", (30, 200, 40)),
        ("banana", (40, 50, 210)),
    )):
        class_folder = folder / label
        class_folder.mkdir(exist_ok=True)
        Image.new("RGB", (24, 24), color).save(class_folder / ("image-" + str(index) + ".png"))

    with _client(monkeypatch, tmp_path, models=["dinov2_vits14"]) as client:
        job = _run_custom(client, {
            "folders": [str(folder)],
            "model_ids": ["dinov2_vits14"],
            "methods": ["lda"],
        })
        assert job["state"] == "completed", job
        run = client.get("/api/v1/runs/" + job["run_id"]).json()

    assert run["methods"] == ["pca"]
    assert len(run["points"]) == 3
    assert any("lda skipped because at least four images are required" in warning
               for warning in run["warnings"])
    assert any("PCA was used" in warning for warning in run["warnings"])


def _detector_fixture(root: Path, *, with_classes: bool = True) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    if with_classes:
        (root / "classes.txt").write_text("scratch\nstain\n", encoding="utf-8")
    folders = [root / "train", root / "test"]
    images = [
        (folders[0], "01.png", "0 0.5 0.5 0.4 0.4\n"),
        (folders[0], "02.png", "0 0.5 0.5 0.4 0.4\n1 0.5 0.5 0.2 0.2\n"),
        (folders[1], "03.png", "1 0.5 0.5 0.3 0.3\n"),
        (folders[1], "04.png", "8 0.5 0.5 0.2 0.2\n"),
    ]
    for index, (folder, filename, label_text) in enumerate(images):
        (folder / "images").mkdir(parents=True, exist_ok=True)
        (folder / "labels").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (30 + index * 30, 70, 160)).save(folder / "images" / filename)
        (folder / "labels" / (Path(filename).stem + ".txt")).write_text(label_text, encoding="utf-8")
    return folders


def test_case_lv_web_01_od_01_whole_image_detector_taxonomy_and_models(
    monkeypatch, tmp_path
):
    """Detector whole-image points keep YOLO class IDs, class-file precedence, and model order."""
    folders = _detector_fixture(tmp_path / "detector")
    source_root = tmp_path / "detector"
    source_before = {
        path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_root.rglob("*") if path.is_file()
    }
    calls: list[str] = []
    selected_models = ["dinov2_vits14", "resnet50"]
    with _client(
        monkeypatch,
        tmp_path / "detector-run",
        embedder=_fake_embedder(calls),
        models=selected_models,
    ) as client:
        job = _run_custom(client, {
            "folders": [str(folder) for folder in folders],
            "mode": "detector",
            "class_names": ["manual-wrong-name"],
            "model_ids": selected_models,
            "methods": ["pca"],
        })
        assert job["state"] == "completed", job
        run = client.get("/api/v1/runs/" + job["run_id"]).json()
        run_response = client.get("/api/v1/runs/" + job["run_id"])
        thumbnail = client.get(f"/api/v1/runs/{job['run_id']}/images/0/thumbnail")
        overlay = client.get(f"/api/v1/runs/{job['run_id']}/images/0/overlay")
        original = client.get(f"/api/v1/runs/{job['run_id']}/images/0")

    assert run["mode"] == "detector"
    assert run["class_names"] == ["scratch", "stain"]
    assert run["models"] == selected_models
    assert str(source_root) not in run_response.text
    assert [(row["filename"], row["split"], row["label"]) for row in run["records"]] == [
        ("01.png", "train", "scratch"),
        ("02.png", "train", "mix"),
        ("03.png", "test", "stain"),
        ("04.png", "test", "class_8"),
    ]
    assert len(calls) == 8
    assert all(len(view["points"]) == 4 for view in run["views"])
    assert thumbnail.status_code == 200 and thumbnail.headers["content-type"] == "image/jpeg"
    assert Image.open(BytesIO(thumbnail.content)).size == (32, 32)
    assert overlay.status_code == 200 and overlay.headers["content-type"] == "image/png"
    assert not np.array_equal(
        np.asarray(Image.open(BytesIO(overlay.content)).convert("RGB")),
        np.asarray(Image.open(BytesIO(original.content)).convert("RGB")),
    )
    assert source_before == {
        path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_root.rglob("*") if path.is_file()
    }


def test_case_lv_web_01_od_01_missing_classes_are_explicit(monkeypatch, tmp_path):
    folders = _detector_fixture(tmp_path / "detector", with_classes=False)
    with _client(monkeypatch, tmp_path / "detector-run") as client:
        missing = _run_custom(client, {
            "folders": [str(folder) for folder in folders],
            "mode": "detector",
            "model_ids": ["dinov2_vits14"],
            "methods": ["pca"],
        })
        fallback = _run_custom(client, {
            "folders": [str(folder) for folder in folders],
            "mode": "detector",
            "class_names": ["scratch", "stain"],
            "model_ids": ["dinov2_vits14"],
            "methods": ["pca"],
        })
        fallback_run = client.get("/api/v1/runs/" + fallback["run_id"]).json()

    assert missing["state"] == "failed"
    assert missing["error"]["code"] == "CLASS_NAMES_UNAVAILABLE"
    assert fallback["state"] == "completed", fallback
    assert fallback_run["class_names"] == ["scratch", "stain"]


def test_case_lv_web_01_od_02_object_detector_uses_crops_and_content_cache(
    monkeypatch, tmp_path
):
    """Object points preserve YOLO order/boxes and cache crops by content, box, model, and policy."""
    folders = _detector_fixture(tmp_path / "detector")
    source_root = tmp_path / "detector"
    source_before = {
        path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_root.rglob("*") if path.is_file()
    }
    crop_calls: list[tuple[int, int]] = []

    def object_loader(_model_id: str, **_kwargs):
        def embed(crop: Image.Image) -> np.ndarray:
            crop_calls.append(crop.size)
            pixels = np.asarray(crop.convert("RGB"), dtype=np.float32)
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
            vector[3] = crop.width / 100.0
            return vector
        return embed

    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "app-cache"))
    app = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=lambda _model_id: _fake_embedder(),
        object_model_loader=object_loader,
    )
    client = TestClient(app)
    policy = {"pad": 0.12, "target_res": 224, "head": "cls"}
    payload = {
        "folders": [str(folder) for folder in folders],
        "mode": "detector",
        "unit": "object",
        "class_names": ["wrong-fallback"],
        "model_ids": ["dinov2_vits14"],
        "methods": ["pca"],
        "object_policy": policy,
    }

    runs = []
    for _ in range(2):
        job = _run_custom(client, payload)
        assert job["state"] == "completed", job
        runs.append(client.get(f"/api/v1/runs/{job['run_id']}").json())

    run = runs[-1]
    assert run["unit"] == "object"
    assert run["object_policy"] == policy
    assert run["class_names"] == ["scratch", "stain"]
    assert run["embedding_shape"] == [5, 384]
    assert run["embedding_finite"] is True
    assert run["embedding_nonzero"] is True
    assert [
        (record["filename"], record["split"], record["label"], record["class_id"], record["obj_index"])
        for record in run["records"]
    ] == [
        ("01.png", "train", "scratch", 0, 0),
        ("02.png", "train", "scratch", 0, 0),
        ("02.png", "train", "stain", 1, 1),
        ("03.png", "test", "stain", 1, 0),
        ("04.png", "test", "class_8", 8, 0),
    ]
    assert run["records"][0]["bbox"] == [0.5, 0.5, 0.4, 0.4]
    assert len(crop_calls) == 5
    assert all(width < 32 and height < 32 for width, height in crop_calls)
    assert runs[0]["records"] == run["records"]
    assert source_before == {
        path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_root.rglob("*") if path.is_file()
    }

    # Moving one existing box changes only that object's cache key.
    label_path = source_root / "train" / "labels" / "01.txt"
    label_path.write_text("0 0.6 0.5 0.4 0.4\n", encoding="utf-8")
    job = _run_custom(client, payload)
    assert job["state"] == "completed", job
    changed_run = client.get(f"/api/v1/runs/{job['run_id']}").json()
    assert changed_run["records"][0]["bbox"] == [0.6, 0.5, 0.4, 0.4]
    object_rankings = client.get(
        f"/api/v1/runs/{job['run_id']}/rankings",
        params={"model_id": "dinov2_vits14", "metric": "disagreement"},
    )
    assert object_rankings.status_code == 200, object_rankings.text
    assert len(object_rankings.json()["results"]) == len(changed_run["records"]) == 5
    assert len(crop_calls) == 6
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == source_before[path.relative_to(source_root).as_posix()]
        for path in source_root.rglob("*.png")
    )


def test_case_lv_web_01_cart_01_curation_round_trip_and_worker_reopen(
    classifier_fixture, monkeypatch, tmp_path
):
    """The React curation cart preserves legacy identity/provenance and survives worker recreation."""
    paths_by_name = {Path(path).name: Path(path) for path in classifier_fixture["hashes"]}
    with _client(monkeypatch, tmp_path) as client:
        job = _run(client, classifier_fixture["folders"])
        run_id = job["run_id"]
        run = client.get(f"/api/v1/runs/{run_id}").json()
        indices = [
            next(index for index, record in enumerate(run["records"]) if record["filename"] == filename)
            for filename in ("01.jpg", "02.jpg")
        ]
        assert client.post(
            f"/api/v1/runs/{run_id}/curation/items", json={"indices": []},
        ).status_code == 422

        added = client.post(f"/api/v1/runs/{run_id}/curation/items", json={"indices": indices})
        assert added.status_code == 201, added.text
        assert (added.json()["added"], added.json()["skipped"], added.json()["total"]) == (2, 0, 2)
        first_items = added.json()["items"]
        assert all("path" not in item and "image_path" not in item for item in first_items)
        assert [item["source"] for item in first_items] == ["manual", "manual"]
        assert [item["sha256"] for item in first_items] == [
            classifier_fixture["hashes"][str(paths_by_name[filename])]
            for filename in ("01.jpg", "02.jpg")
        ]

        duplicate = client.post(
            f"/api/v1/runs/{run_id}/curation/items",
            json={"indices": indices, "source": "search", "reason": "do not replace first provenance"},
        )
        assert (duplicate.json()["added"], duplicate.json()["skipped"]) == (0, 2)
        assert [item["source"] for item in duplicate.json()["items"]] == ["manual", "manual"]
        assert [item["reason"] for item in duplicate.json()["items"]] == ["", ""]

        item_ids = [item["id"] for item in first_items]
        reason = client.patch("/api/v1/curation/items/reason", json={
            "item_ids": item_ids,
            "reason": "check the unusual edge cases",
        })
        assert reason.status_code == 200, reason.text
        assert reason.json()["updated"] == 2
        assert all(item["reason"] == "check the unusual edge cases" for item in reason.json()["items"])

        listing = client.get("/api/v1/curation/items").json()
        assert listing["total"] == 2
        assert [item["filename"] for item in listing["items"]] == ["01.jpg", "02.jpg"]
        image = client.get(f"/api/v1/curation/items/{item_ids[0]}/thumbnail")
        assert image.status_code == 200
        assert image.content == paths_by_name["01.jpg"].read_bytes()

        csv_response = client.get("/api/v1/curation.csv")
        assert csv_response.status_code == 200
        with StringIO(csv_response.text.lstrip("\ufeff")) as csv_file:
            rows = list(csv.DictReader(csv_file))
        assert csv_response.text.lstrip("\ufeff").splitlines()[0] == (
            "index,filename,path,label,split,sha256,source,score,reason"
        )
        assert len(rows) == 2
        assert [row["filename"] for row in rows] == ["01.jpg", "02.jpg"]
        assert all(len(row["sha256"]) == 64 and row["source"] == "manual" for row in rows)
        assert all(row["reason"] == "check the unusual edge cases" for row in rows)
        assert all(Path(row["path"]).is_file() for row in rows)

        json_items = client.get("/api/v1/curation.json").json()
        assert [item["filename"] for item in json_items] == ["01.jpg", "02.jpg"]
        assert all(Path(item["path"]).is_file() for item in json_items)
        assert all(item["reason"] == "check the unusual edge cases" for item in json_items)

        with ZipFile(BytesIO(client.get("/api/v1/curation.zip").content)) as archive:
            names = archive.namelist()
            assert "manifest.csv" in names
            image_names = [name for name in names if name.startswith("images/")]
            assert len(image_names) == 2
            assert {archive.read(name) for name in image_names} == {
                paths_by_name["01.jpg"].read_bytes(), paths_by_name["02.jpg"].read_bytes(),
            }

        for path, expected_hash in classifier_fixture["hashes"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_hash

    # A new API worker instance uses the same app cache and reconstructs the cart.
    with TestClient(create_app(model_catalog=lambda: [], model_loader=lambda _model: None)) as restarted:
        readback = restarted.get("/api/v1/curation/items").json()
        assert readback["total"] == 2
        assert [item["filename"] for item in readback["items"]] == ["01.jpg", "02.jpg"]
        removed = restarted.delete(f"/api/v1/curation/items/{readback['items'][0]['id']}")
        assert removed.status_code == 200 and removed.json()["total"] == 1
        cleared = restarted.delete("/api/v1/curation/items")
        assert cleared.status_code == 200 and cleared.json()["cleared"] == 1
        assert restarted.get("/api/v1/curation/items").json() == {"items": [], "total": 0}


def test_case_lv_web_01_cart_01_object_records_keep_object_identity(
    monkeypatch, tmp_path
):
    """Selecting two boxes from one image retains both object IDs in the cart."""
    folders = _detector_fixture(tmp_path / "detector")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "object-cart-cache"))
    object_loader = lambda _model_id, **_kwargs: (lambda _crop: np.ones(384, dtype=np.float32))
    with TestClient(create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model_id: _fake_embedder(),
        object_model_loader=object_loader,
    )) as client:
        job = _run_custom(client, {
            "folders": [str(folder) for folder in folders],
            "mode": "detector",
            "unit": "object",
            "model_id": "fixture-model",
            "method": "pca",
        })
        assert job["state"] == "completed", job
        run = client.get(f"/api/v1/runs/{job['run_id']}").json()
        indices = [
            index for index, record in enumerate(run["records"])
            if record["filename"] == "02.png"
        ]
        assert len(indices) == 2
        response = client.post(
            f"/api/v1/runs/{job['run_id']}/curation/items", json={"indices": indices},
        )
        assert response.status_code == 201, response.text
        items = response.json()["items"]
        assert [item["obj_index"] for item in items] == [0, 1]
        assert all(item["level"] == "object" for item in items)
        assert items[0]["id"] != items[1]["id"]
        assert all("image_path" not in item and "path" not in item for item in items)
