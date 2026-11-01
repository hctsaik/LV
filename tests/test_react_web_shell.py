"""AC tests for LV-WEB-01-IC-01; see 4_PM_Feedback/react_web_shell.md."""

from __future__ import annotations

import hashlib
import os
import sys
import time
import types
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from react_web_shell import create_app
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
    assert len(crop_calls) == 6
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == source_before[path.relative_to(source_root).as_posix()]
        for path in source_root.rglob("*.png")
    )
