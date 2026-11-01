"""AC tests for LV-WEB-01-IC-01; see 4_PM_Feedback/react_web_shell.md."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from react_web_shell import create_app
from visualize_embeddings import discover_images_classifier


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


def _client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, embedder=None, models=None):
    cache_root = tmp_path.parent / f"{tmp_path.name}-lv-cache"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache_root))
    return TestClient(create_app(
        model_catalog=lambda: models if models is not None else ["dinov2_vits14"],
        model_loader=lambda _model: embedder or _fake_embedder(),
    ))


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
