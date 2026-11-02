"""Focused contracts for the React whole-image A/B comparison slice.

The injected feature function is deterministic and local. These tests verify
folder, cache, API, projection and readback contracts; they are not real-model
or visible Streamlit-parity evidence.
"""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from react_compare import CompareCancelled, CompareInputError, project_comparison, run_image_comparison
from react_web_shell import create_app


def _solid(path: Path, color: tuple[int, int, int]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 18), color).save(path)
    return path.read_bytes()


def _feature(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        mean = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) / 255.0
    return np.r_[mean, mean.mean(), 1.0].astype(np.float32)


def _runner(folders_a, folders_b, model_id, *, progress=None, cancelled=None):
    return run_image_comparison(
        folders_a, folders_b, model_id,
        model_loader=lambda _model_id: _feature,
        progress=progress,
        cancelled=cancelled,
    )


def _wait_for_compare(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        status = client.get(f"/api/v1/compare/jobs/{job_id}").json()
        if status["state"] in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(0.01)
    pytest.fail("comparison job did not reach a terminal state")


def _create_compare(client: TestClient, folders_a: list[Path], folders_b: list[Path]) -> dict:
    created = client.post("/api/v1/compare/jobs", json={
        "folders_a": [str(folder) for folder in folders_a],
        "folders_b": [str(folder) for folder in folders_b],
        "model_id": "contract-test-model",
    })
    assert created.status_code == 202, created.text
    status = _wait_for_compare(client, created.json()["job_id"])
    assert status["state"] == "completed", status
    return status


def test_case_lv_web_02_img_01_groups_multiple_roots_and_deduplicates_in_picker_order(tmp_path: Path):
    a1, a2, b1 = (tmp_path / name for name in ("a1", "a2", "b1"))
    _solid(a1 / "z.png", (220, 30, 40))
    _solid(a1 / "a.jpg", (210, 40, 50))
    _solid(a2 / "nested" / "b.jpeg", (190, 50, 60))
    _solid(b1 / "target.png", (30, 60, 220))

    run = run_image_comparison(
        [a1, a2, a1], [b1], "contract-test-model",
        model_loader=lambda _model_id: _feature,
    )

    assert run["folder_a"] == "A（2 個資料夾）"
    assert [row["filename"] for row in run["records_a"]] == ["a.jpg", "z.png", "b.jpeg"]
    assert [path.name for path in run["image_paths_a"]] == ["a.jpg", "z.png", "b.jpeg"]
    assert len(run["records_a"]) == len(set(run["image_paths_a"])) == 3
    assert any("subfolders" in warning for warning in run["warnings"])


def test_case_lv_web_02_img_01_centroid_drift_matches_streamlit_arithmetic(tmp_path: Path):
    a, b = tmp_path / "A", tmp_path / "B"
    colors_a = [(255, 0, 0), (255, 255, 0)]
    colors_b = [(0, 0, 255), (0, 255, 255)]
    for index, color in enumerate(colors_a): _solid(a / f"{index}.png", color)
    for index, color in enumerate(colors_b): _solid(b / f"{index}.png", color)
    values_a = np.stack([_feature(a / f"{index}.png") for index in range(2)])
    values_b = np.stack([_feature(b / f"{index}.png") for index in range(2)])
    mean_a = values_a.mean(axis=0); mean_a /= np.linalg.norm(mean_a) + 1e-9
    mean_b = values_b.mean(axis=0); mean_b /= np.linalg.norm(mean_b) + 1e-9
    expected = round(max(0.0, 1.0 - float(mean_a @ mean_b)), 4)

    run = run_image_comparison(
        [a], [b], "contract-test-model", model_loader=lambda _model_id: _feature,
    )

    assert run["drift"] == expected


def test_case_lv_web_02_img_01_app_cache_reuses_embeddings_without_changing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    a, b = tmp_path / "A", tmp_path / "B"
    source = {
        a / "a.png": _solid(a / "a.png", (210, 30, 40)),
        a / "b.png": _solid(a / "b.png", (200, 40, 50)),
        b / "c.png": _solid(b / "c.png", (30, 40, 220)),
        b / "d.png": _solid(b / "d.png", (40, 50, 210)),
    }
    cache_root = tmp_path / "worker-cache"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache_root))
    calls: list[Path] = []

    def embed(path: Path):
        calls.append(path)
        return _feature(path)

    first = run_image_comparison([a], [b], "contract-test-model", model_loader=lambda _id: embed)
    second = run_image_comparison([a], [b], "contract-test-model", model_loader=lambda _id: embed)

    assert len(calls) == 4
    assert second["drift"] == first["drift"]
    assert list(cache_root.rglob("compare_embeddings_contract-test-model/embeddings.npz"))
    assert all(path.read_bytes() == content for path, content in source.items())


def test_case_lv_web_02_img_01_unreadable_images_are_skipped_and_empty_group_is_rejected(tmp_path: Path):
    a, b = tmp_path / "A", tmp_path / "B"
    empty = tmp_path / "empty"
    empty.mkdir()
    _solid(a / "valid.png", (210, 40, 50))
    (a / "broken.png").write_bytes(b"not an image")
    _solid(b / "valid.png", (30, 60, 220))

    run = run_image_comparison([a], [b], "contract-test-model", model_loader=lambda _id: _feature)
    assert len(run["records_a"]) == 1
    assert any("unreadable" in warning for warning in run["warnings"])
    with pytest.raises(CompareInputError) as exc:
        run_image_comparison([empty], [b], "contract-test-model")
    assert exc.value.code == "NO_IMAGES"


def test_case_lv_web_02_img_01_projection_is_index_stable_and_validates_method():
    embeddings_a = np.asarray([[0, 1, 2], [1, 0, 2], [2, 1, 0], [1, 3, 2]], dtype=np.float32)
    embeddings_b = np.asarray([[3, 1, 0], [2, 4, 1], [0, 2, 4], [4, 0, 2]], dtype=np.float32)

    view = project_comparison(embeddings_a, embeddings_b, "pca", 2)
    repeated = project_comparison(embeddings_a, embeddings_b, "pca", 2)

    assert view["method"] == "pca" and view["dimension"] == 2
    assert [(point["group"], point["index"]) for point in view["points"]] == (
        [("a", index) for index in range(4)] + [("b", index) for index in range(4)]
    )
    assert np.asarray([[p["x"], p["y"]] for p in view["points"]]) == pytest.approx(
        np.asarray([[p["x"], p["y"]] for p in repeated["points"]])
    )
    with pytest.raises(CompareInputError) as exc:
        project_comparison(embeddings_a, embeddings_b, "unknown", 2)
    assert exc.value.code == "METHOD_UNAVAILABLE"


def test_case_lv_web_02_img_01_api_hides_embeddings_and_reads_images_by_run_group_index(tmp_path: Path):
    a1, a2, b1 = (tmp_path / name for name in ("A1", "A2", "B1"))
    source_a0 = _solid(a1 / "first.png", (210, 30, 40))
    source_a1 = _solid(a2 / "second.png", (180, 50, 60))
    source_b = _solid(b1 / "blue.png", (30, 40, 220))

    with TestClient(create_app(model_catalog=lambda: ["contract-test-model"], comparison_runner=_runner)) as client:
        status = _create_compare(client, [a1, a2], [b1])
        run_response = client.get(f"/api/v1/compare/runs/{status['run_id']}")
        run = run_response.json()
        projection = client.post(
            f"/api/v1/compare/runs/{status['run_id']}/projections",
            json={"method": "pca", "dimension": 2},
        )
        image_a = client.get(f"/api/v1/compare/runs/{status['run_id']}/images/a/1")
        image_b = client.get(f"/api/v1/compare/runs/{status['run_id']}/images/b/0")
        bad_group = client.get(f"/api/v1/compare/runs/{status['run_id']}/images/c/0")
        bad_index = client.get(f"/api/v1/compare/runs/{status['run_id']}/images/a/99")

    assert run_response.status_code == 200
    assert set(run) == {"folder_a", "folder_b", "model_id", "drift", "records_a", "records_b", "warnings", "overlap_count"}
    assert run["folder_a"] == "A（2 個資料夾）"
    assert str(tmp_path) not in run_response.text
    assert "embeddings_a" not in run_response.text
    assert projection.status_code == 200
    assert projection.json()["method"] == "pca"
    assert len(projection.json()["points"]) == 3
    assert image_a.content == source_a1 and image_b.content == source_b
    assert image_a.headers["cache-control"] == "no-store"
    assert bad_group.status_code == bad_index.status_code == 404
    assert source_a0 != source_a1


def test_case_lv_web_02_img_01_projection_endpoint_rejects_unknown_method(tmp_path: Path):
    a, b = tmp_path / "A", tmp_path / "B"
    _solid(a / "a.png", (210, 30, 40)); _solid(b / "b.png", (30, 40, 220))
    with TestClient(create_app(model_catalog=lambda: ["contract-test-model"], comparison_runner=_runner)) as client:
        status = _create_compare(client, [a], [b])
        response = client.post(
            f"/api/v1/compare/runs/{status['run_id']}/projections",
            json={"method": "not-a-projection", "dimension": 2},
        )
    assert response.status_code == 422
    assert response.json()["code"] == "METHOD_UNAVAILABLE"


def test_case_lv_web_02_img_01_compare_and_visualize_share_worker_slot(tmp_path: Path):
    a, b = tmp_path / "A", tmp_path / "B"
    _solid(a / "a.png", (210, 30, 40)); _solid(b / "b.png", (30, 40, 220))
    started, release = Event(), Event()

    def slow_runner(folders_a, folders_b, model_id, *, progress=None, cancelled=None):
        started.set()
        release.wait(timeout=5)
        return _runner(folders_a, folders_b, model_id, progress=progress, cancelled=cancelled)

    with TestClient(create_app(model_catalog=lambda: ["contract-test-model"], comparison_runner=slow_runner)) as client:
        first = client.post("/api/v1/compare/jobs", json={
            "folders_a": [str(a)], "folders_b": [str(b)], "model_id": "contract-test-model",
        })
        assert first.status_code == 202 and started.wait(timeout=2)
        blocked = client.post("/api/v1/visualize/jobs", json={
            "folders": [str(a)], "model_id": "contract-test-model",
        })
        release.set()
        status = _wait_for_compare(client, first.json()["job_id"])

    assert blocked.status_code == 409
    assert status["state"] == "completed"


def test_case_lv_web_02_img_01_cancellation_is_reported(tmp_path: Path):
    a, b = tmp_path / "A", tmp_path / "B"
    _solid(a / "a.png", (210, 30, 40)); _solid(b / "b.png", (30, 40, 220))
    started = Event()

    def cancellable_runner(_folders_a, _folders_b, _model_id, *, cancelled=None, **_kwargs):
        started.set()
        while not cancelled(): time.sleep(0.005)
        raise CompareCancelled()

    with TestClient(create_app(model_catalog=lambda: ["contract-test-model"], comparison_runner=cancellable_runner)) as client:
        created = client.post("/api/v1/compare/jobs", json={
            "folders_a": [str(a)], "folders_b": [str(b)], "model_id": "contract-test-model",
        })
        assert created.status_code == 202 and started.wait(timeout=2)
        job_id = created.json()["job_id"]
        cancelled = client.delete(f"/api/v1/compare/jobs/{job_id}")
        status = _wait_for_compare(client, job_id)

    assert cancelled.status_code == 202
    assert status["state"] == "cancelled" and status["run_id"] is None
