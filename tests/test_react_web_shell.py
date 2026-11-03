"""AC tests for LV-WEB-01-IC-01; see 4_PM_Feedback/react_web_shell.md."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import time
import zipfile
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


def test_case_lv_web_01_search_01_similarity_reorders_by_model_and_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """SEARCH-01: nearest images sort by cosine distance and recompute per model."""
    folder = tmp_path / "train" / "classA"
    folder.mkdir(parents=True)
    filenames = ["a_query.png", "b_near_a.png", "c_near_b.png", "d_far.png"]
    colors = [(220, 35, 45), (210, 40, 50), (35, 65, 215), (30, 200, 70)]
    for filename, color in zip(filenames, colors):
        Image.new("RGB", (32, 32), color).save(folder / filename)

    vectors = {
        "model_a": {
            "a_query.png": [1.0, 0.0, 0.0],
            "b_near_a.png": [0.99, 0.1, 0.0],
            "c_near_b.png": [0.0, 1.0, 0.0],
            "d_far.png": [0.0, 0.0, 1.0],
        },
        "model_b": {
            "a_query.png": [1.0, 0.0, 0.0],
            "b_near_a.png": [0.0, 1.0, 0.0],
            "c_near_b.png": [0.99, 0.1, 0.0],
            "d_far.png": [0.0, 0.0, 1.0],
        },
    }

    def model_loader(model_id: str):
        return lambda path: np.asarray(vectors[model_id][path.name], dtype=np.float32)

    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    client = TestClient(create_app(
        model_catalog=lambda: ["model_a", "model_b"],
        model_loader=model_loader,
    ))
    created = client.post("/api/v1/visualize/jobs", json={
        "folders": [str(tmp_path / "train")],
        "model_ids": ["model_a", "model_b"],
        "methods": ["pca"],
    })
    assert created.status_code == 202, created.text
    job_id = created.json()["job_id"]
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = client.get(f"/api/v1/jobs/{job_id}").json()
        if status["state"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.02)
    assert status["state"] == "completed", status
    run_id = status["run_id"]
    run = client.get(f"/api/v1/runs/{run_id}").json()
    query_index = next(row["index"] for row in run["records"] if row["filename"] == "a_query.png")

    model_a = client.post(f"/api/v1/runs/{run_id}/similar", json={
        "model_id": "model_a", "query_index": query_index, "k": 3,
    })
    model_b = client.post(f"/api/v1/runs/{run_id}/similar", json={
        "model_id": "model_b", "query_index": query_index, "k": 3,
    })
    assert model_a.status_code == model_b.status_code == 200
    result_a = model_a.json()["results"]
    result_b = model_b.json()["results"]
    assert [row["filename"] for row in result_a][0] == "b_near_a.png"
    assert [row["filename"] for row in result_b][0] == "c_near_b.png"
    assert set(result_a[0]) == {"index", "filename", "split", "label", "distance"}
    assert all(row["index"] != query_index for row in result_a + result_b)
    for rows in (result_a, result_b):
        distances = [row["distance"] for row in rows]
        assert distances == sorted(distances)
        assert all(np.isfinite(distance) and 0 <= distance <= 2 for distance in distances)

    export = client.get(f"/api/v1/runs/{run_id}/similar.csv", params={
        "model_id": "model_b", "query_index": query_index, "k": 3,
    })
    assert export.status_code == 200
    csv_rows = list(csv.reader(io.StringIO(export.text)))
    assert csv_rows[0] == ["index", "filename", "path", "label", "split"]
    assert len(csv_rows) == 1 + 1 + len(result_b)
    assert csv_rows[1][1] == "a_query.png"
    assert [row[1] for row in csv_rows[2:]][0] == "c_near_b.png"
    assert (folder / "a_query.png").is_file()

    unavailable_model = client.post(f"/api/v1/runs/{run_id}/similar", json={
        "model_id": "not-in-run", "query_index": query_index, "k": 3,
    })
    invalid_export_count = client.get(f"/api/v1/runs/{run_id}/similar.csv", params={
        "model_id": "model_b", "query_index": query_index, "k": 0,
    })
    assert unavailable_model.status_code == 404
    assert unavailable_model.json()["code"] == "MODEL_NOT_IN_RUN"
    assert invalid_export_count.status_code == 422
    assert invalid_export_count.json()["code"] == "INVALID_RESULT_COUNT"


def test_case_lv_web_01_search_02_text_search_and_duplicate_leakage_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """SEARCH-02: text retrieval and phash/embedding split review use run records."""
    root = tmp_path / "dataset"
    train = root / "train" / "classA"
    valid = root / "val" / "classA"
    train.mkdir(parents=True)
    valid.mkdir(parents=True)
    rng = np.random.default_rng(2026)
    image_bytes = {}
    for index in range(3):
        path = train / f"unique-{index}.png"
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
        image_bytes[path] = path.read_bytes()
    duplicate_path = train / "dup.png"
    Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(duplicate_path)
    image_bytes[duplicate_path] = duplicate_path.read_bytes()
    duplicate_copy = valid / "dup_copy.png"
    duplicate_copy.write_bytes(image_bytes[duplicate_path])
    image_bytes[duplicate_copy] = duplicate_copy.read_bytes()
    unique_val = valid / "unique-val.png"
    Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(unique_val)
    image_bytes[unique_val] = unique_val.read_bytes()

    def feature_for(path: Path) -> np.ndarray:
        return np.frombuffer(hashlib.sha256(path.read_bytes()).digest(), dtype=np.uint8).astype(np.float32) + 1

    text_queries: list[str] = []

    def text_encoder_loader(model_id: str):
        assert model_id == "chinese-clip-test"

        def encode(query: str) -> np.ndarray:
            text_queries.append(query)
            return feature_for(duplicate_path) if query == "斑馬" else feature_for(unique_val)

        return encode

    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    client = TestClient(create_app(
        model_catalog=lambda: ["chinese-clip-test"],
        model_loader=lambda _model: feature_for,
        text_encoder_loader=text_encoder_loader,
    ))
    created = client.post("/api/v1/visualize/jobs", json={
        "folders": [str(root / "train"), str(root / "val")],
        "model_id": "chinese-clip-test",
        "method": "pca",
    })
    assert created.status_code == 202, created.text
    job_id = created.json()["job_id"]
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = client.get(f"/api/v1/jobs/{job_id}").json()
        if status["state"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.02)
    assert status["state"] == "completed", status
    run_id = status["run_id"]
    run = client.get(f"/api/v1/runs/{run_id}").json()
    assert "_phashes" not in run and "_embedding_sources" not in run
    records = run["records"]
    expected_pair = {
        next(row["index"] for row in records if row["filename"] == "dup.png"),
        next(row["index"] for row in records if row["filename"] == "dup_copy.png"),
    }

    text_result = client.post(f"/api/v1/runs/{run_id}/text-search", json={
        "model_id": "chinese-clip-test", "query": "斑馬", "k": 3,
    })
    assert text_result.status_code == 200, text_result.text
    assert text_result.json()["query"] == "斑馬"
    assert text_queries == ["斑馬"]
    assert {row["filename"] for row in text_result.json()["results"][:2]} == {
        "dup.png", "dup_copy.png",
    }
    assert [row["distance"] for row in text_result.json()["results"]] == sorted(
        row["distance"] for row in text_result.json()["results"]
    )
    text_export = client.get(f"/api/v1/runs/{run_id}/text-search.csv", params={
        "model_id": "chinese-clip-test", "query": "斑馬", "k": 3,
    })
    assert text_export.status_code == 200
    text_csv_rows = list(csv.reader(io.StringIO(text_export.text)))
    assert text_csv_rows[0] == ["index", "filename", "path", "label", "split"]
    assert len(text_csv_rows) == 4
    assert text_queries == ["斑馬"]  # encoder instance and query vector are reused
    unsupported_text_model = client.post(f"/api/v1/runs/{run_id}/text-search", json={
        "model_id": "dinov2_vits14", "query": "zebra", "k": 3,
    })
    long_text_query = client.get(f"/api/v1/runs/{run_id}/text-search.csv", params={
        "model_id": "chinese-clip-test", "query": "x" * 513, "k": 3,
    })
    assert unsupported_text_model.status_code == 409
    assert unsupported_text_model.json()["code"] == "TEXT_QUERY_UNAVAILABLE"
    assert long_text_query.status_code == 422
    assert long_text_query.json()["code"] == "TEXT_QUERY_TOO_LONG"

    all_phash = client.post(f"/api/v1/runs/{run_id}/duplicates", json={
        "model_id": "chinese-clip-test", "method": "phash", "phash_threshold": 4,
    })
    cross_phash = client.post(f"/api/v1/runs/{run_id}/duplicates", json={
        "model_id": "chinese-clip-test", "method": "phash", "phash_threshold": 4,
        "cross_split_only": True,
    })
    cross_embedding = client.post(f"/api/v1/runs/{run_id}/duplicates", json={
        "model_id": "chinese-clip-test", "method": "embedding", "embedding_threshold": 0.05,
        "cross_split_only": True,
    })
    for response in (all_phash, cross_phash, cross_embedding):
        assert response.status_code == 200, response.text
        assert all("path" not in pair["left"] and "path" not in pair["right"]
                   for pair in response.json()["pairs"])
    assert any({pair["left"]["index"], pair["right"]["index"]} == expected_pair
               and pair["distance"] == 0 for pair in all_phash.json()["pairs"])
    for result in (cross_phash.json(), cross_embedding.json()):
        assert result["cross_split_only"] is True
        assert all(pair["left"]["split"] != pair["right"]["split"] for pair in result["pairs"])
        assert any({pair["left"]["index"], pair["right"]["index"]} == expected_pair
                   for pair in result["pairs"])
    assert {path: path.read_bytes() for path in image_bytes} == image_bytes
    assert {path for path in root.rglob("*") if path.is_file()} == set(image_bytes)


def test_case_lv_web_01_rank_01_metrics_and_review_orders_match_streamlit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """RANK-01: scores and selected-record orders preserve the Streamlit rules."""
    from interaction import compute_label_disagreement, compute_outlier_scores, spatial_order

    root = tmp_path / "rank-dataset"
    vectors: dict[str, np.ndarray] = {}
    for index, angle in enumerate((0.0, 0.25, 0.9, 1.5, 2.2, 2.8, 3.7, 5.1)):
        split = "train" if index < 6 else "val"
        label = "apple" if index < 4 else "banana"
        folder = root / split / label
        folder.mkdir(parents=True, exist_ok=True)
        filename = f"image-{index:02}.png"
        Image.new("RGB", (32, 32), (20 + index * 20, 80, 160)).save(folder / filename)
        vectors[filename] = np.asarray([
            np.cos(angle), np.sin(angle), 0.15 * (index % 3), 0.05 * index + 0.1,
        ], dtype=np.float32)

    def embed(path: Path) -> np.ndarray:
        return vectors[path.name]

    with _client(monkeypatch, tmp_path, embedder=embed, models=["rank-model"]) as client:
        job = _run(client, [root / "train", root / "val"], model="rank-model")
        assert job["state"] == "completed", job
        run_id = job["run_id"]
        run = client.get(f"/api/v1/runs/{run_id}").json()
        outlier = client.post(f"/api/v1/runs/{run_id}/rankings", json={
            "model_id": "rank-model", "criterion": "outlier",
        })
        disagreement = client.post(f"/api/v1/runs/{run_id}/rankings", json={
            "model_id": "rank-model", "criterion": "disagreement",
        })
        selected_ids = [6, 2, 4, 1]
        selected_outlier = client.post(f"/api/v1/runs/{run_id}/rankings", json={
            "model_id": "rank-model", "criterion": "outlier", "indices": selected_ids,
        })
        selected_disagreement = client.post(f"/api/v1/runs/{run_id}/rankings", json={
            "model_id": "rank-model", "criterion": "disagreement", "indices": selected_ids,
        })
        by_filename = client.post(f"/api/v1/runs/{run_id}/rankings", json={
            "model_id": "rank-model", "criterion": "filename", "indices": selected_ids,
        })
        spatial = client.post(f"/api/v1/runs/{run_id}/rankings", json={
            "model_id": "rank-model", "criterion": "spatial", "indices": selected_ids,
            "method": "pca",
        })
        invalid_index = client.post(f"/api/v1/runs/{run_id}/rankings", json={
            "model_id": "rank-model", "criterion": "outlier", "indices": [999],
        })

    assert "_ranking_scores" not in run and "_embedding_sources" not in run
    records = run["records"]
    matrix = np.stack([vectors[record["filename"]] for record in records])
    labels = [record["label"] for record in records]
    k = min(5, len(records) - 1)
    expected_outlier = compute_outlier_scores(matrix, matrix, k=k, candidates_in_reference=True)
    expected_disagreement = compute_label_disagreement(matrix, labels, k=k)
    for response in (outlier, disagreement, selected_outlier, selected_disagreement, by_filename, spatial):
        assert response.status_code == 200, response.text
    assert [item["index"] for item in outlier.json()["items"]] == np.argsort(expected_outlier)[::-1][:50].tolist()
    assert [item["index"] for item in disagreement.json()["items"]] == np.argsort(expected_disagreement)[::-1][:50].tolist()
    assert np.allclose([item["score"] for item in outlier.json()["items"]],
                       expected_outlier[[item["index"] for item in outlier.json()["items"]]])
    assert [item["index"] for item in selected_outlier.json()["items"]] == sorted(
        selected_ids, key=lambda index: -float(expected_outlier[index]),
    )
    assert [item["index"] for item in selected_disagreement.json()["items"]] == sorted(
        selected_ids, key=lambda index: -float(expected_disagreement[index]),
    )
    assert [item["index"] for item in by_filename.json()["items"]] == sorted(
        selected_ids, key=lambda index: records[index]["filename"],
    )
    coords = np.asarray([[point["x"], point["y"]] for point in run["views"][0]["points"]])
    assert [item["index"] for item in spatial.json()["items"]] == spatial_order(coords, selected_ids)
    assert invalid_index.status_code == 422
    assert invalid_index.json()["code"] == "INVALID_RECORD_INDEX"


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
        ranking_response = client.post(f"/api/v1/runs/{job['run_id']}/rankings", json={
            "model_id": "dinov2_vits14", "criterion": "outlier",
        })

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
    assert ranking_response.status_code == 200, ranking_response.text
    assert ranking_response.json()["available"] is True
    assert len(ranking_response.json()["items"]) == 4
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


def test_case_lv_web_01_cart_01_persists_deduplicates_and_exports(
    classifier_fixture, monkeypatch, tmp_path
):
    """CART-01: the saved cart survives worker recreation and preserves export identity."""
    originals = {
        path: path.read_bytes()
        for folder in classifier_fixture["folders"]
        for path in folder.rglob("*")
        if path.is_file()
    }
    cache_root = tmp_path.parent / f"{tmp_path.name}-lv-cache"

    with _client(monkeypatch, tmp_path) as client:
        job = _run(client, classifier_fixture["folders"])
        assert job["state"] == "completed", job
        run_id = job["run_id"]
        assert client.get("/api/v1/curation-cart").json()["item_count"] == 0

        added = client.post("/api/v1/curation-cart/items", json={
            "run_id": run_id,
            "items": [
                {"index": 0, "source": "manual"},
                {"index": 0, "source": "search", "score": 0.125},
                {"index": 1, "source": "outlier", "score": 0.375},
            ],
        })
        assert added.status_code == 200, added.text
        payload = added.json()
        assert (payload["item_count"], payload["added"], payload["skipped"]) == (2, 2, 1)
        assert all("path" not in item for item in payload["items"])
        assert payload["items"][0]["source"] == "manual"
        assert payload["items"][0]["score"] is None
        assert payload["items"][1]["score"] == 0.375
        assert payload["items"][1]["sha256"]

        item_ids = [item["id"] for item in payload["items"]]
        reason_response = client.patch("/api/v1/curation-cart/reasons", json={
            "item_ids": item_ids,
            "reason": "Review for balanced validation coverage",
        })
        assert reason_response.status_code == 200, reason_response.text
        assert {item["reason"] for item in reason_response.json()["items"]} == {
            "Review for balanced validation coverage"
        }
        assert client.request(
            "DELETE", "/api/v1/curation-cart", json={"confirm": False}
        ).status_code == 409

        csv_response = client.get("/api/v1/curation-cart.csv")
        assert csv_response.status_code == 200
        rows = list(csv.DictReader(io.StringIO(csv_response.text)))
        assert [row["source"] for row in rows] == ["manual", "outlier"]
        assert rows[1]["score"] == "0.375"
        assert all(row["reason"] == "Review for balanced validation coverage" for row in rows)

        zip_response = client.get("/api/v1/curation-cart.zip")
        assert zip_response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
            names = archive.namelist()
            assert "manifest.csv" in names
            assert len([name for name in names if name.startswith("images/")]) == 2
            assert "status,arcname" in archive.read("manifest.csv").decode("utf-8")

        persisted_path = cache_root / "curation_cart.json"
        stored = json.loads(persisted_path.read_text(encoding="utf-8"))
        assert stored["schema_version"] == 1
        assert stored["items"][0]["path"]
        assert stored["items"][0]["reason"] == "Review for balanced validation coverage"

    # A fresh FastAPI app models a worker restart; it reads the app-owned cache.
    with TestClient(create_app()) as restarted_worker:
        restored = restarted_worker.get("/api/v1/curation-cart")
        assert restored.status_code == 200
        assert restored.json()["item_count"] == 2
        json_response = restarted_worker.get("/api/v1/curation-cart.json")
        exported = json_response.json()
        assert exported["schema_version"] == 1
        assert [item["source"] for item in exported["items"]] == ["manual", "outlier"]
        assert all(item["reason"] == "Review for balanced validation coverage" for item in exported["items"])
        assert all(Path(item["path"]).is_absolute() for item in exported["items"])
        removed = restarted_worker.delete(f"/api/v1/curation-cart/items/{exported['items'][0]['id']}")
        assert removed.status_code == 200
        assert removed.json()["item_count"] == 1
        cleared = restarted_worker.request(
            "DELETE", "/api/v1/curation-cart", json={"confirm": True}
        )
        assert cleared.status_code == 200
        assert cleared.json()["item_count"] == 0

    assert {path: path.read_bytes() for path in originals} == originals


def test_case_lv_web_01_cart_01_keeps_object_identity_for_boxes_on_one_image(
    monkeypatch, tmp_path
):
    root = tmp_path / "cart-objects"
    folder = root / "train"
    images, labels = folder / "images", folder / "labels"
    labels.mkdir(parents=True)
    (root / "classes.txt").write_text("part\n", encoding="utf-8")
    image_path = images / "same-image.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (80, 60), (70, 120, 180)).save(image_path)
    (labels / "same-image.txt").write_text(
        "0 0.25 0.5 0.2 0.4\n0 0.75 0.5 0.2 0.4\n", encoding="utf-8"
    )
    source_before = image_path.read_bytes()

    def object_loader(_model: str, _policy: dict):
        def embed(crop: Image.Image) -> np.ndarray:
            pixels = np.asarray(crop.convert("RGB"), dtype=np.float32)
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
            return vector
        return embed

    with _client(monkeypatch, tmp_path, object_model_loader=object_loader) as client:
        job = _run_custom(client, {
            "folders": [str(folder)], "model_ids": ["dinov2_vits14"],
            "methods": ["pca"], "mode": "object_detector", "granularity": "object",
        })
        assert job["state"] == "completed", job
        run = client.get(f"/api/v1/runs/{job['run_id']}").json()
        assert [row["object_index"] for row in run["records"]] == [0, 1]
        added = client.post("/api/v1/curation-cart/items", json={
            "run_id": job["run_id"],
            "items": [{"index": 0}, {"index": 1}],
        })
        assert added.status_code == 200, added.text
        items = added.json()["items"]

    assert len(items) == 2
    assert items[0]["id"] != items[1]["id"]
    assert [item["object_index"] for item in items] == [0, 1]
    assert image_path.read_bytes() == source_before
