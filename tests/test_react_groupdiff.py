"""React API coverage for the migrated Good/Bad patch-difference workflow."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from react_web_shell import create_app


D = 16
SHIFT_PATCH = 5


def _make_images(folder: Path, count: int, *, marker: str | None = None, seed0: int = 0) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        rng = np.random.default_rng(seed0 + index)
        pixels = rng.integers(90, 160, (56, 56, 3)).astype("uint8")
        if marker == "bad":
            pixels[:8, :8] = 0
        Image.fromarray(pixels).save(folder / f"img_{seed0 + index:03d}.png")


def _patch_extractor():
    rng = np.random.default_rng(999)
    base = rng.normal(size=(16, D))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    shift = rng.normal(size=D)
    shift /= np.linalg.norm(shift)

    def extract(image: Image.Image):
        pixels = np.asarray(image.convert("RGB"), dtype=float)
        marker = pixels[:16, :16].mean()
        sample_rng = np.random.default_rng(int(pixels.sum()) % (2 ** 32))
        features = base + sample_rng.normal(scale=0.15, size=(16, D))
        if marker < 30:
            features[SHIFT_PATCH] += 1.2 * shift
        features /= np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None)
        return features.astype(np.float32), (4, 4)

    return extract


def _snapshot(*folders: Path) -> dict[str, tuple[str, int]]:
    result = {}
    for folder in folders:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                result[str(path)] = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    return result


def _wait_for_job(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        status = client.get(f"/api/v1/groupdiff/jobs/{job_id}").json()
        if status["state"] in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(0.01)
    raise AssertionError("group difference job did not finish")


def _run(client: TestClient, good: Path, bad: Path) -> dict:
    created = client.post("/api/v1/groupdiff/jobs", json={
        "good_dirs": [str(good)], "bad_dirs": [str(bad)], "model_id": "dinov2_vits14",
        "n_perm": 100, "alpha": 0.05, "top_k": 5, "seed": 0,
    })
    assert created.status_code == 202, created.text
    status = _wait_for_job(client, created.json()["job_id"])
    assert status["state"] == "completed", status
    assert status["can_cancel"] is False
    assert status["run_id"]
    return client.get(f"/api/v1/groupdiff/runs/{status['run_id']}").json()


def test_groupdiff_api_planted_null_heatmap_region_pca_and_safe_export(monkeypatch, tmp_path):
    cache = tmp_path / "lv-cache"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache))
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    _make_images(good, 8, seed0=0)
    _make_images(bad, 8, marker="bad", seed0=100)
    source_before = _snapshot(good, bad)

    all_null = tmp_path / "all-null"
    null_a, null_b = tmp_path / "null-a", tmp_path / "null-b"
    _make_images(all_null, 8, seed0=300)
    null_a.mkdir()
    null_b.mkdir()
    for index, image in enumerate(sorted(all_null.glob("*.png"))):
        (null_a / image.name).write_bytes(image.read_bytes())
        (null_b / image.name).write_bytes(image.read_bytes())
    null_before = _snapshot(null_a, null_b)

    with TestClient(create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=lambda _model: None,
        groupdiff_extractor=_patch_extractor(),
    )) as client:
        result = _run(client, good, bad)
        run_id = result["run_id"]
        assert result["verdict"] is True
        assert result["regions"] and result["regions"][0]["rank"] == 1
        assert "good_paths" not in result and "bad_paths" not in result
        assert str(good) not in str(result) and str(bad) not in str(result)

        base = client.get(f"/api/v1/groupdiff/runs/{run_id}/bases/bad/0?overlay=false")
        heatmap = client.get(f"/api/v1/groupdiff/runs/{run_id}/bases/bad/0?overlay=true&alpha=0.55")
        assert base.status_code == heatmap.status_code == 200
        assert base.headers["content-type"].startswith("image/png")
        assert Image.open(__import__("io").BytesIO(heatmap.content)).size == (56, 56)
        region = client.get(f"/api/v1/groupdiff/runs/{run_id}/regions/1/images/good/0.jpg")
        assert region.status_code == 200 and region.headers["content-type"].startswith("image/jpeg")
        pca = client.get(f"/api/v1/groupdiff/runs/{run_id}/pca").json()
        assert len(pca["xy_good"]) == len(pca["xy_bad"]) == 8
        assert len(pca["evr"]) == 2 and pca["evr"][0] >= pca["evr"][1]

        output = tmp_path / "report"
        exported = client.post(f"/api/v1/groupdiff/runs/{run_id}/exports", json={"output_dir": str(output)})
        assert exported.status_code == 200, exported.text
        assert {"heatmap_good.png", "heatmap_bad.png", "regions.csv", "region_01.png", "summary.json"} <= set(exported.json()["files"])
        assert (output / "regions.csv").is_file() and (output / "summary.json").is_file()
        unsafe = client.post(f"/api/v1/groupdiff/runs/{run_id}/exports", json={"output_dir": str(good / "report")})
        assert unsafe.status_code == 422
        assert not (good / "report").exists()

        null_result = _run(client, null_a, null_b)
        assert null_result["verdict"] is False

    assert _snapshot(good, bad) == source_before
    assert _snapshot(null_a, null_b) == null_before
