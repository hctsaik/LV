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
from visualize_embeddings import discover_images, discover_images_classifier


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
):
    cache_root = tmp_path.parent / f"{tmp_path.name}-lv-cache"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache_root))
    return TestClient(create_app(
        model_catalog=lambda: models if models is not None else ["dinov2_vits14"],
        model_loader=lambda _model: embedder or _fake_embedder(),
        object_model_loader=object_model_loader,
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


def test_case_lv_web_01_ic_03_umap_reference_survives_data_growth(
    classifier_fixture, monkeypatch, tmp_path
):
    """Fixed UMAP keeps prior image coordinates and transforms only newly added content."""
    cache_root = tmp_path.parent / f"{tmp_path.name}-lv-cache"
    with _client(monkeypatch, tmp_path, embedder=_fake_embedder()) as client:
        payload = {
            "folders": [str(folder) for folder in classifier_fixture["folders"]],
            "model_ids": ["dinov2_vits14"],
            "methods": ["umap"],
            "umap_reference": True,
        }
        pre_existing_image = classifier_fixture["root"] / "train" / "cherry" / "pre-existing-image.png"
        pre_existing_image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (60, 90, 170)).save(pre_existing_image)
        first = _run_custom(client, payload)
        assert first["state"] == "completed", first
        first_run = client.get(f"/api/v1/runs/{first['run_id']}").json()
        first_points = {
            point["filename"]: (point["x"], point["y"], point["z"])
            for point in first_run["points"]
        }
        assert any("Fixed UMAP reference rebuilt" in warning for warning in first_run["warnings"])

        second = _run_custom(client, payload)
        assert second["state"] == "completed", second
        second_run = client.get(f"/api/v1/runs/{second['run_id']}").json()
        second_points = {
            point["filename"]: (point["x"], point["y"], point["z"])
            for point in second_run["points"]
        }
        assert first_points == second_points
        assert any("Fixed UMAP reference reused (+0 new images)" in warning
                   for warning in second_run["warnings"])

        new_image = classifier_fixture["root"] / "train" / "cherry" / "new-reference-image.png"
        Image.new("RGB", (32, 32), (25, 140, 200)).save(new_image)
        grown = _run_custom(client, payload)
        assert grown["state"] == "completed", grown
        grown_run = client.get(f"/api/v1/runs/{grown['run_id']}").json()
        grown_points = {
            point["filename"]: (point["x"], point["y"], point["z"])
            for point in grown_run["points"]
        }
        assert len(grown_points) == 6
        assert {name: grown_points[name] for name in first_points} == first_points
        assert any("Fixed UMAP reference reused (+1 new images)" in warning
                   for warning in grown_run["warnings"])

        rebuild = _run_custom(client, {**payload, "rebuild_umap_reference": True})
        assert rebuild["state"] == "completed", rebuild
        rebuild_run = client.get(f"/api/v1/runs/{rebuild['run_id']}").json()
        assert any("Fixed UMAP reference rebuilt" in warning for warning in rebuild_run["warnings"])

    references = list(cache_root.rglob("umap_ref.pkl"))
    assert len(references) == 1
    assert references[0].is_file()
    assert not list(classifier_fixture["root"].rglob("umap_ref.pkl"))


def test_case_lv_web_01_ic_03_rebuild_requires_fixed_umap_option(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as client:
        response = client.post("/api/v1/visualize/jobs", json={
            "folders": [str(tmp_path)],
            "model_ids": ["dinov2_vits14"],
            "methods": ["pca"],
            "rebuild_umap_reference": True,
        })
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_UMAP_REFERENCE"


def test_case_lv_web_01_ic_04_kmeans_blocks_and_embedding_drilldown(
    classifier_fixture, monkeypatch, tmp_path
):
    with _client(monkeypatch, tmp_path, embedder=_fake_embedder()) as client:
        job = _run_custom(client, {
            "folders": [str(folder) for folder in classifier_fixture["folders"]],
            "model_ids": ["dinov2_vits14"],
            "methods": ["pca"],
        })
        assert job["state"] == "completed", job
        run_response = client.get(f"/api/v1/runs/{job['run_id']}")
        run = run_response.json()
        assert not any(key.startswith("_") for key in run)

        blocks_response = client.post(f"/api/v1/runs/{job['run_id']}/blocks", json={
            "model_id": "dinov2_vits14",
            "method": "pca",
            "indices": [0, 1, 2, 3],
            "k": 12,
        })
        invalid_index = client.post(f"/api/v1/runs/{job['run_id']}/blocks", json={
            "model_id": "dinov2_vits14",
            "method": "pca",
            "indices": [0, 999],
            "k": 4,
        })
        duplicate_index = client.post(f"/api/v1/runs/{job['run_id']}/blocks", json={
            "model_id": "dinov2_vits14",
            "method": "pca",
            "indices": [0, 0],
            "k": 4,
        })
        projection_response = client.post(f"/api/v1/runs/{job['run_id']}/block-projection", json={
            "model_id": "dinov2_vits14",
            "method": "pca",
            "indices": [0, 1, 2],
        })

    assert blocks_response.status_code == 200, blocks_response.text
    blocks = blocks_response.json()
    assert blocks["effective_k"] == 4
    flattened = [index for block in blocks["blocks"] for index in block["indices"]]
    assert sorted(flattened) == [0, 1, 2, 3]
    assert all(block["count"] == len(block["indices"]) for block in blocks["blocks"])
    assert all(np.isfinite([center["x"], center["y"], center["z"]]).all()
               for center in blocks["centers"])
    assert invalid_index.status_code == 422
    assert duplicate_index.status_code == 422
    assert projection_response.status_code == 200, projection_response.text
    projection = projection_response.json()
    assert projection["projection_source"] == "embedding_pca"
    assert [point["index"] for point in projection["points"]] == [0, 1, 2]
    assert all(np.isfinite([point["x"], point["y"]]).all()
               for point in projection["points"])


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


def test_case_lv_web_01_od_01_detector_whole_image_class_sources_and_models(
    monkeypatch, tmp_path
):
    """Detector whole-image runs preserve class-source priority and multi-model views."""
    root = tmp_path / "detector"
    root.mkdir()
    (root / "classes.txt").write_text("good\nrotten\n", encoding="utf-8")
    folders = [root / "train", root / "val"]
    source_hashes = {}
    for split_index, folder in enumerate(folders):
        images = folder / "images"
        labels = folder / "labels"
        labels.mkdir(parents=True)
        for class_id, color in enumerate(((210, 35, 45), (35, 180, 70))):
            image_path = images / f"{split_index}-{class_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24, 24), color).save(image_path)
            (labels / f"{image_path.stem}.txt").write_text(
                f"{class_id} 0.5 0.5 0.6 0.6\n", encoding="utf-8"
            )
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }
    models = ["dinov2_vits14", "resnet50"]

    with _client(monkeypatch, tmp_path, models=models) as client:
        selected_file = _run_custom(client, {
            "folders": [str(folder) for folder in folders],
            "model_ids": models,
            "methods": ["pca"],
            "mode": "object_detector",
            "classes_file_text": "override_good\noverride_bad\n",
            "class_names": ["manual_good", "manual_bad"],
        })
        assert selected_file["state"] == "completed", selected_file
        selected_run = client.get(f"/api/v1/runs/{selected_file['run_id']}").json()

        auto_file = _run_custom(client, {
            "folders": [str(folder) for folder in folders],
            "model_ids": [models[0]],
            "methods": ["pca"],
            "mode": "object_detector",
            "class_names": ["manual_good", "manual_bad"],
        })
        assert auto_file["state"] == "completed", auto_file
        auto_run = client.get(f"/api/v1/runs/{auto_file['run_id']}").json()

        yaml_root = tmp_path / "yaml-detector"
        yaml_root.mkdir()
        (yaml_root / "data.yaml").write_text(
            "names: [yaml_good, yaml_bad]\n", encoding="utf-8"
        )
        yaml_folder = yaml_root / "train"
        yaml_images, yaml_labels = yaml_folder / "images", yaml_folder / "labels"
        yaml_labels.mkdir(parents=True)
        for class_id, color in enumerate(((195, 40, 50), (40, 185, 65))):
            image_path = yaml_images / f"yaml-{class_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24, 24), color).save(image_path)
            (yaml_labels / f"{image_path.stem}.txt").write_text(
                f"{class_id} 0.5 0.5 0.6 0.6\n", encoding="utf-8"
            )
        yaml_job = _run_custom(client, {
            "folders": [str(yaml_folder)],
            "model_ids": [models[0]],
            "methods": ["pca"],
            "mode": "object_detector",
            "class_names": ["manual_good", "manual_bad"],
        })
        assert yaml_job["state"] == "completed", yaml_job
        yaml_run = client.get(f"/api/v1/runs/{yaml_job['run_id']}").json()

        manual_root = tmp_path / "manual-detector"
        manual_folder = manual_root / "train"
        images, labels = manual_folder / "images", manual_folder / "labels"
        labels.mkdir(parents=True)
        for class_id, color in enumerate(((200, 45, 55), (45, 190, 75))):
            image_path = images / f"manual-{class_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24, 24), color).save(image_path)
            (labels / f"{image_path.stem}.txt").write_text(
                f"{class_id} 0.5 0.5 0.6 0.6\n", encoding="utf-8"
            )
        manual = _run_custom(client, {
            "folders": [str(manual_folder)],
            "model_ids": [models[0]],
            "methods": ["pca"],
            "mode": "object_detector",
            "class_names": ["manual_good", "manual_bad"],
        })
        assert manual["state"] == "completed", manual
        manual_run = client.get(f"/api/v1/runs/{manual['run_id']}").json()

        missing = _run_custom(client, {
            "folders": [str(manual_folder)],
            "model_ids": [models[0]],
            "methods": ["pca"],
            "mode": "object_detector",
        })

    assert selected_run["source_mode"] == "object_detector"
    assert selected_run["class_source"] == "selected_classes_file"
    assert selected_run["class_names"] == ["override_good", "override_bad"]
    assert [row["label"] for row in selected_run["records"]] == [
        "override_good", "override_bad", "override_good", "override_bad"
    ]
    assert selected_run["models"] == models
    assert {(view["model_id"], view["method"]) for view in selected_run["views"]} == {
        (model, "pca") for model in models
    }
    assert "classes_file_text" not in selected_run
    assert auto_run["class_source"] == "auto_classes_txt"
    assert auto_run["class_names"] == ["good", "rotten"]
    assert [row["label"] for row in auto_run["records"]] == [
        "good", "rotten", "good", "rotten"
    ]
    assert yaml_run["class_source"] == "auto_data_yaml"
    assert yaml_run["class_names"] == ["yaml_good", "yaml_bad"]
    assert manual_run["class_source"] == "manual"
    assert manual_run["class_names"] == ["manual_good", "manual_bad"]
    assert missing["state"] == "failed"
    assert missing["error"]["code"] == "CLASSES_REQUIRED"
    assert "classes.txt" in missing["error"]["message"]
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest
               for path, digest in source_hashes.items())
    assert {(Path(record["path"]).name, record["split"], record["label"])
            for record in discover_images(folders, ["good", "rotten"])} == {
        (row["filename"], row["split"], row["label"]) for row in auto_run["records"]
    }


def test_case_lv_web_01_od_02_object_embeddings_policy_order_and_crop_readback(
    monkeypatch, tmp_path
):
    """Object runs crop each bbox, apply the chosen policy and preserve row identity."""
    root = tmp_path / "detector-objects"
    root.mkdir()
    (root / "classes.txt").write_text("good\nbad\n", encoding="utf-8")
    folders = [root / "train", root / "val"]
    image_specs = [
        (folders[0], "train-01.png", (220, 35, 45)),
        (folders[1], "val-01.png", (35, 180, 70)),
    ]
    for folder, filename, color in image_specs:
        images, labels = folder / "images", folder / "labels"
        labels.mkdir(parents=True)
        image_path = images / filename
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (100, 60), color)
        image.paste((35, 65, 215), (40, 0, 70, 60))
        image.save(image_path)
        rows = (
            "0 0.20 0.50 0.20 0.40\n"
            "1 0.72 0.50 0.30 0.60\n"
            "0 0.48 0.50 0.12 0.20\n"
            if folder == folders[0]
            else "1 0.50 0.50 0.50 0.50\n"
        )
        (labels / f"{image_path.stem}.txt").write_text(rows, encoding="utf-8")
    source_before = {
        path: path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    model_calls: list[tuple[str, dict]] = []
    crop_sizes: list[tuple[int, int]] = []

    def object_loader(model_id: str, policy: dict):
        model_calls.append((model_id, dict(policy)))

        def embed(crop: Image.Image) -> np.ndarray:
            assert isinstance(crop, Image.Image)
            crop_sizes.append(crop.size)
            pixels = np.asarray(crop.convert("RGB"), dtype=np.float32)
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
            vector[3] = crop.width / 100.0
            vector[4] = crop.height / 100.0
            return vector

        return embed

    policy = {"pad": 0.25, "target_res": 336, "head": "meanpool"}
    with _client(
        monkeypatch,
        tmp_path,
        object_model_loader=object_loader,
    ) as client:
        job = _run_custom(client, {
            "folders": [str(folder) for folder in folders],
            "model_ids": ["dinov2_vits14"],
            "methods": ["pca"],
            "mode": "object_detector",
            "granularity": "object",
            "object_policy": policy,
        })
        assert job["state"] == "completed", job
        run = client.get(f"/api/v1/runs/{job['run_id']}").json()
        crop_response = client.get(f"/api/v1/runs/{job['run_id']}/images/0")

    assert run["source_mode"] == "object_detector"
    assert run["source_granularity"] == "object"
    assert run["object_policy"] == policy
    assert [row["label"] for row in run["records"]] == ["good", "bad", "good", "bad"]
    assert [row["object_index"] for row in run["records"]] == [0, 1, 2, 0]
    assert [row["index"] for row in run["points"]] == [0, 1, 2, 3]
    assert [row["filename"] for row in run["records"]] == [
        "train-01.png", "train-01.png", "train-01.png", "val-01.png"
    ]
    assert model_calls == [("dinov2_vits14", policy)]
    assert len(crop_sizes) == 4
    assert crop_response.status_code == 200
    assert crop_response.headers["content-type"].startswith("image/jpeg")
    from io import BytesIO
    crop = Image.open(BytesIO(crop_response.content))
    assert crop.size == crop_sizes[0]
    assert crop.width < 100 and crop.height < 60
    assert {path: path.read_bytes() for path in source_before} == source_before
    assert {path for path in root.rglob("*") if path.is_file()} == set(source_before)


def test_case_lv_web_01_od_02_rejects_invalid_object_policy(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as client:
        response = client.post("/api/v1/visualize/jobs", json={
            "folders": [str(tmp_path)],
            "model_ids": ["dinov2_vits14"],
            "mode": "object_detector",
            "granularity": "object",
            "object_policy": {"pad": 0.8, "target_res": 224, "head": "cls"},
        })
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_REQUEST"


def test_case_lv_web_01_view_01_detector_image_route_draws_yolo_overlay(
    monkeypatch, tmp_path
):
    root = tmp_path / "overlay-detector"
    folder = root / "train"
    images, labels = folder / "images", folder / "labels"
    labels.mkdir(parents=True)
    (root / "classes.txt").write_text("part\n", encoding="utf-8")
    image_path = images / "overlay.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), (40, 90, 150)).save(image_path)
    label_path = labels / "overlay.txt"
    label_path.write_text("0 0.5 0.5 0.6 0.6\n", encoding="utf-8")
    second_image = images / "overlay-second.png"
    Image.new("RGB", (64, 64), (190, 35, 45)).save(second_image)
    second_label = labels / "overlay-second.txt"
    second_label.write_text("0 0.3 0.3 0.3 0.3\n", encoding="utf-8")
    originals = {
        path: path.read_bytes()
        for path in (image_path, label_path, second_image, second_label, root / "classes.txt")
    }

    with _client(monkeypatch, tmp_path) as client:
        job = _run_custom(client, {
            "folders": [str(folder)],
            "model_ids": ["dinov2_vits14"],
            "methods": ["pca"],
            "mode": "object_detector",
            "class_names": ["part"],
        })
        assert job["state"] == "completed", job
        base = f"/api/v1/runs/{job['run_id']}/images/0"
        raw = client.get(base, params={"overlay": "false"})
        overlay = client.get(base, params={"overlay": "true"})

    assert raw.status_code == overlay.status_code == 200
    assert raw.headers["content-type"].startswith("image/png")
    assert overlay.headers["content-type"].startswith("image/jpeg")
    from io import BytesIO
    raw_pixels = np.asarray(Image.open(BytesIO(raw.content)).convert("RGB"))
    overlay_pixels = np.asarray(Image.open(BytesIO(overlay.content)).convert("RGB"))
    assert raw_pixels.shape == overlay_pixels.shape == (64, 64, 3)
    assert not np.array_equal(raw_pixels, overlay_pixels)
    assert {path: path.read_bytes() for path in originals} == originals
