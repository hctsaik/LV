from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from react_web_shell import create_app


def _seed_label_quality_run(
    application,
    root: Path,
    labels: list[str],
    embeddings: list[list[float]],
) -> tuple[str, list[Path]]:
    paths = []
    records = []
    source_hashes = []
    for index, label in enumerate(labels):
        path = root / f"image-{index:02}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (index * 20, 40, 80)).save(path)
        paths.append(path)
        source_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        records.append({"index": index, "filename": path.name, "label": label, "split": "train"})

    run_id = "label-quality-run"
    application.state.runs[run_id] = {
        "records": records,
        "image_paths": paths,
        "_source_sha256": source_hashes,
        "unit": "image",
        "mode": "classifier",
    }
    application.state.current_run_id = run_id
    application.state.raw_embeddings[run_id] = {
        "fixture-model": np.asarray(embeddings, dtype=np.float32),
    }
    return run_id, paths


def test_label_quality_api_uses_full_embeddings_and_keeps_paths_private(tmp_path: Path):
    application = create_app(model_catalog=lambda: ["fixture-model"])
    labels = ["cat", "cat", "dog", "dog", "cat"]
    embeddings = [[1.0, 0.0], [0.95, 0.1], [-1.0, 0.0], [-0.95, 0.1], [-1.0, -0.05]]
    run_id, paths = _seed_label_quality_run(application, tmp_path, labels, embeddings)

    with TestClient(application) as client:
        response = client.post(
            f"/api/v1/runs/{run_id}/label-quality",
            json={"model_id": "fixture-model", "k": 2},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["classes"] == ["cat", "dog"]
        assert result["counts"] == [3, 2]
        assert len(result["confusion"]) == 2
        assert result["mismatch_count"] == 3
        assert result["candidates"][0] == {
            "index": 4,
            "filename": paths[4].name,
            "own": "cat",
            "suggested": "dog",
            "suggested_frac": 1.0,
            "own_frac": 0.0,
            "disagreement": 1.0,
            "bbox": None,
            "obj_index": None,
        }
        assert {candidate["index"] for candidate in result["candidates"][1:]} == {2, 3}
        assert str(tmp_path) not in json.dumps(result)

        paths[0].write_bytes(paths[0].read_bytes() + b"changed")
        stale = client.post(
            f"/api/v1/runs/{run_id}/label-quality",
            json={"model_id": "fixture-model", "k": 2},
        )
        assert stale.status_code == 409
        assert stale.json()["code"] == "STALE_LABEL_QUALITY_SOURCE"


def test_label_quality_api_rejects_insufficient_classes(tmp_path: Path):
    application = create_app(model_catalog=lambda: ["fixture-model"])
    run_id, _paths = _seed_label_quality_run(
        application,
        tmp_path,
        ["cat", "cat", "cat"],
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
    )

    with TestClient(application) as client:
        response = client.post(
            f"/api/v1/runs/{run_id}/label-quality",
            json={"model_id": "fixture-model", "k": 2},
        )
    assert response.status_code == 422
    assert response.json()["code"] == "LABEL_QUALITY_DATA_INSUFFICIENT"
