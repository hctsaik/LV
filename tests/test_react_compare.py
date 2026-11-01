"""Unit and local API acceptance for LV Compare Distributions."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from react_web_shell import create_app


def _image(folder: Path, name: str, color: tuple[int, int, int]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    Image.new("RGB", (32, 28), color).save(path)
    return path


def _whole_embedder(path: Path) -> np.ndarray:
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    result = np.zeros(384, dtype=np.float32)
    result[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
    result[3] = len(path.name) / 10.0
    return result


def _object_embedder(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    result = np.zeros(384, dtype=np.float32)
    result[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
    result[3] = image.width / max(image.height, 1)
    return result


def _completed_run(client: TestClient, payload: dict) -> tuple[dict, dict]:
    response = client.post("/api/v1/compare/jobs", json=payload)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        job = client.get(f"/api/v1/compare/jobs/{job_id}").json()
        if job["state"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.02)
    else:
        pytest.fail("comparison job did not finish within 45 seconds")
    assert job["state"] == "completed", job
    result = client.get(f"/api/v1/comparisons/{job['run_id']}")
    assert result.status_code == 200, result.text
    return job, result.json()


def test_case_lv_web_02_img_01_whole_image_drift_projection_and_path_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    a = [_image(tmp_path / "a", f"a-{i}.png", (230, 20 + i, 30)) for i in range(3)]
    b = [_image(tmp_path / "b", f"b-{i}.png", (20, 35, 225 - i)) for i in range(3)]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in a + b}
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "app-cache"))
    client = TestClient(create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model_id: _whole_embedder,
    ))

    job, result = _completed_run(client, {
        "folders_a": [str(tmp_path / "a")],
        "folders_b": [str(tmp_path / "b")],
        "model_id": "fixture-model",
        "unit": "whole",
    })
    projection = client.post(f"/api/v1/comparisons/{job['run_id']}/projections", json={
        "label": "(全部影像)", "method": "pca", "dimension": 2,
    })

    assert projection.status_code == 200, projection.text
    view = projection.json()
    from sklearn.decomposition import PCA

    # Independently reproduce the legacy Streamlit means, normalization, and
    # 2D PCA on the same fixed fixture embeddings.
    embeddings_a = np.stack([_whole_embedder(path) for path in a])
    embeddings_b = np.stack([_whole_embedder(path) for path in b])
    mean_a = embeddings_a.mean(axis=0)
    mean_a = mean_a / (np.linalg.norm(mean_a) + 1e-9)
    mean_b = embeddings_b.mean(axis=0)
    mean_b = mean_b / (np.linalg.norm(mean_b) + 1e-9)
    expected_drift = round(max(0.0, 1.0 - float(mean_a @ mean_b)), 4)
    expected_projection = PCA(n_components=2, random_state=42).fit_transform(
        np.vstack([embeddings_a, embeddings_b])
    )

    assert len(result["records_a"]) == len(result["records_b"]) == 3
    assert result["rows"] == [{
        "label": "(全部影像)", "a_count": 3, "b_count": 3, "drift": result["rows"][0]["drift"],
    }]
    assert result["rows"][0]["drift"] == expected_drift
    assert len(view["points"]) == 6
    assert {point["group"] for point in view["points"]} == {"A", "B"}
    assert np.allclose(
        [[point["x"], point["y"]] for point in view["points"]],
        expected_projection,
        rtol=1e-5,
        atol=1e-5,
    )
    assert all(np.isfinite([point["x"], point["y"]]).all() for point in view["points"])
    assert "_paths_a" not in result and "image_paths" not in result
    image = client.get(f"/api/v1/comparisons/{job['run_id']}/images/a/0")
    invalid = client.get(f"/api/v1/comparisons/{job['run_id']}/images/a/99")
    assert image.status_code == 200 and image.content == a[0].read_bytes()
    assert invalid.status_code == 404
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in a + b} == before
    assert list((tmp_path / "app-cache").rglob("*.npz"))


def test_case_lv_web_02_obj_01_compares_shared_classes_and_keeps_boxes_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    roots = [tmp_path / "det-a", tmp_path / "det-b"]
    colors = [(210, 40, 35), (35, 50, 210)]
    for root, color in zip(roots, colors):
        (root / "labels").mkdir(parents=True)
        (root / "classes.txt").write_text("widget\n", encoding="utf-8")
        _image(root / "images", "same-name.png", color)
        (root / "labels" / "same-name.txt").write_text("0 0.5 0.5 0.7 0.6\n", encoding="utf-8")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "app-cache"))
    client = TestClient(create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model_id: _whole_embedder,
        object_model_loader=lambda _model_id, **_kwargs: _object_embedder,
    ))

    job, result = _completed_run(client, {
        "folders_a": [str(roots[0])],
        "folders_b": [str(roots[1])],
        "model_id": "fixture-model",
        "unit": "object",
        "object_policy": {"pad": 0.2, "target_res": 280, "head": "meanpool"},
    })
    projection = client.post(f"/api/v1/comparisons/{job['run_id']}/projections", json={
        "label": "widget", "method": "pca", "dimension": 2,
        "min_confidence": 0.0, "max_confidence": 1.0, "min_short_px": 0,
    })

    assert projection.status_code == 200, projection.text
    assert result["rows"][0]["label"] == "widget"
    assert result["rows"][0]["a_count"] == result["rows"][0]["b_count"] == 1
    assert result["policy"] == {"pad": 0.2, "target_res": 280, "head": "meanpool"}
    assert result["policy_source"] == "manual"
    assert len(result["autotune_rows"]) == 6
    assert result["records_a"][0]["bbox"] == [0.5, 0.5, 0.7, 0.6]
    assert "path" not in result["records_a"][0]
    assert len(projection.json()["points"]) == 2
    assert np.allclose(
        [[point["x"], point["y"]] for point in projection.json()["points"]],
        np.zeros((2, 2)),
    )
    projection_3d = client.post(f"/api/v1/comparisons/{job['run_id']}/projections", json={
        "label": "widget", "method": "pca", "dimension": 3,
        "min_confidence": 0.0, "max_confidence": 1.0, "min_short_px": 0,
    })
    assert projection_3d.status_code == 200, projection_3d.text
    assert np.allclose(
        [[point["x"], point["y"], point["z"]] for point in projection_3d.json()["points"]],
        np.zeros((2, 3)),
    )
    assert client.get(f"/api/v1/comparisons/{job['run_id']}/images/a/0").status_code == 200


def test_case_lv_web_02_state_01_compare_selection_adds_unique_original_images_to_curation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    roots = [tmp_path / "det-a", tmp_path / "det-b"]
    (roots[0] / "images").mkdir(parents=True)
    (roots[0] / "labels").mkdir()
    (roots[1] / "images").mkdir(parents=True)
    (roots[1] / "labels").mkdir()
    before = []
    for index, root in enumerate(roots):
        image_path = root / "images" / f"group-{index}.png"
        Image.new("RGB", (64, 56), (70 + index * 90, 60, 150)).save(image_path)
        (root / "classes.txt").write_text("widget\n", encoding="utf-8")
        label_rows = (
            ["0 0.3 0.4 0.4 0.5 0.40", "0 0.7 0.6 0.3 0.4 0.45"]
            if index == 0 else
            ["0 0.3 0.4 0.4 0.5 0.40", "0 0.7 0.6 0.3 0.4 0.90"]
        )
        (root / "labels" / f"group-{index}.txt").write_text("\n".join(label_rows) + "\n", encoding="utf-8")
        before.extend([image_path, root / "classes.txt", root / "labels" / f"group-{index}.txt"])
    hashes_before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "app-cache"))
    client = TestClient(create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model_id: _whole_embedder,
        object_model_loader=lambda _model_id, **_kwargs: _object_embedder,
    ))

    job, result = _completed_run(client, {
        "folders_a": [str(roots[0])],
        "folders_b": [str(roots[1])],
        "model_id": "fixture-model",
        "unit": "object",
    })
    assert [record["score"] for record in result["records_a"]] == [0.4, 0.45]
    assert [record["score"] for record in result["records_b"]] == [0.4, 0.9]
    response = client.post(
        f"/api/v1/comparisons/{job['run_id']}/curation/items",
        json={"selections": [
            {"group": "A", "index": 0}, {"group": "A", "index": 1},
            {"group": "B", "index": 0},
        ]},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert (payload["added"], payload["skipped"], payload["total"]) == (2, 0, 2)
    assert [
        (item["filename"], item["label"], item["source"], item["score"], item["level"])
        for item in payload["items"]
    ] == [
        ("group-0.png", "widget", "compare", 0.4, "image"),
        ("group-1.png", "widget", "compare", 0.4, "image"),
    ]
    assert all(not {"path", "image_path", "bbox", "obj_index"} & item.keys() for item in payload["items"])

    invalid = client.post(
        f"/api/v1/comparisons/{job['run_id']}/curation/items",
        json={"selections": [{"group": "A", "index": 0}, {"group": "B", "index": 9}]},
    )
    assert invalid.status_code == 422
    assert client.get("/api/v1/curation/items").json()["total"] == 2
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before} == hashes_before


def test_case_lv_web_02_rejects_missing_groups_and_invalid_projection(tmp_path, monkeypatch):
    empty_a, empty_b = tmp_path / "empty-a", tmp_path / "empty-b"
    empty_a.mkdir()
    empty_b.mkdir()
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "app-cache"))
    client = TestClient(create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model_id: _whole_embedder,
    ))
    invalid_policy = client.post("/api/v1/compare/jobs", json={
        "folders_a": [str(empty_a)], "folders_b": [str(empty_b)],
        "model_id": "fixture-model", "unit": "object",
        "object_policy": {"pad": 0.7, "target_res": 280, "head": "meanpool"},
    })
    assert invalid_policy.status_code == 422
    job_response = client.post("/api/v1/compare/jobs", json={
        "folders_a": [str(empty_a)], "folders_b": [str(empty_b)],
        "model_id": "fixture-model", "unit": "whole",
    })
    job_id = job_response.json()["job_id"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f"/api/v1/compare/jobs/{job_id}").json()
        if job["state"] in {"failed", "completed"}:
            break
        time.sleep(0.02)
    assert job["state"] == "failed"
    assert job["error"]["code"] == "NO_IMAGES_A"
    assert "C:\\" not in job["error"]["message"]


def test_case_lv_web_02_obj_01_no_shared_classes_is_explicit_and_path_safe(tmp_path, monkeypatch):
    roots = [tmp_path / "det-a", tmp_path / "det-b"]
    for root, label in zip(roots, ("alpha", "beta")):
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        (root / "classes.txt").write_text(f"{label}\n", encoding="utf-8")
        _image(root / "images", "item.png", (60, 100, 180))
        (root / "labels" / "item.txt").write_text("0 0.5 0.5 0.7 0.6\n", encoding="utf-8")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "app-cache"))
    client = TestClient(create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model_id: _whole_embedder,
        object_model_loader=lambda _model_id, **_kwargs: _object_embedder,
    ))

    response = client.post("/api/v1/compare/jobs", json={
        "folders_a": [str(roots[0])], "folders_b": [str(roots[1])],
        "model_id": "fixture-model", "unit": "object",
    })
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        job = client.get(f"/api/v1/compare/jobs/{job_id}").json()
        if job["state"] in {"failed", "completed", "cancelled"}:
            break
        time.sleep(0.02)
    else:
        pytest.fail("comparison job did not report the missing shared classes")

    assert job["state"] == "failed"
    assert job["error"]["code"] == "NO_COMMON_CLASSES"
    assert "no shared detection classes" in job["error"]["message"].lower()
    assert str(tmp_path) not in job["error"]["message"]
