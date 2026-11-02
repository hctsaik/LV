from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from react_web_shell import create_app


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_react_curation_subset_preflight_safe_export_and_object_lineage(
    tmp_path: Path, monkeypatch,
):
    source = tmp_path / "dataset"
    image_dir = source / "train" / "images"
    label_dir = source / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (source / "classes.txt").write_text("zebra\napple\n", encoding="utf-8")
    image = image_dir / "sample.png"
    Image.new("RGB", (24, 24), (100, 40, 180)).save(image)
    other_image = image_dir / "other.png"
    Image.new("RGB", (24, 24), (20, 180, 60)).save(other_image)
    label = label_dir / "sample.txt"
    label.write_text("0 0.5 0.5 0.4 0.4\n1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    missing = image_dir / "missing.png"

    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    app = create_app(model_catalog=lambda: [], model_loader=lambda _model: None)
    with TestClient(app) as client:
        app.state.curation_store.add([
            {
                "path": str(image), "image_path": str(image), "filename": image.name,
                "sha256": _sha256(image), "label": "object", "split": "train",
                "source": "inspect", "reason": "edge case", "level": "object",
                "obj_index": 0,
            },
            {
                "path": str(image), "image_path": str(image), "filename": image.name,
                "sha256": _sha256(image), "label": "object", "split": "train",
                "source": "inspect", "reason": "second box", "level": "object",
                "obj_index": 1,
            },
            {
                "path": str(missing), "filename": missing.name, "sha256": "0" * 64,
                "label": "object", "split": "train", "source": "inspect",
                "reason": "missing source", "level": "image",
            },
            {
                "path": str(other_image), "filename": other_image.name,
                "sha256": _sha256(other_image), "label": "manual class",
                "split": "train", "source": "manual", "reason": "other source",
                "level": "image",
            },
        ])

        outside = tmp_path / "subset-output"
        preview_input = {
            "source": "inspect",
            "destination": str(outside),
            "mode": "copy",
            "on_exists": "rename",
        }
        preview = client.post("/api/v1/curation/subset/preview", json=preview_input)
        assert preview.status_code == 200, preview.text
        preview_json = preview.json()
        assert preview_json["selected"] == 3
        assert preview_json["available"] == 2
        assert preview_json["unique_images"] == 1
        assert preview_json["deduplicated"] == 1
        assert preview_json["missing"] == 1
        assert preview_json["missing_filenames"] == ["missing.png"]
        assert preview_json["label_files"] == 1
        assert preview_json["object_items"] == 2
        assert preview_json["destination_safe"] is True
        assert str(source) not in preview.text

        blocked = client.post("/api/v1/curation/subset/preview", json={
            **preview_input,
            "destination": str(source / "train" / "nested-output"),
        })
        assert blocked.status_code == 422
        assert "destination overlaps a source dataset" in blocked.json()["message"]
        assert str(source) not in blocked.text

        # A source-label change invalidates the preview before any output is written.
        label.write_text("0 0.5 0.5 0.3 0.3\n1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        stale = client.post("/api/v1/curation/subset/export", json={
            **preview_input,
            "preview_token": preview_json["preview_token"],
        })
        assert stale.status_code == 409
        assert stale.json()["code"] == "CURATION_SUBSET_PREVIEW_STALE"
        assert not outside.exists()

        # Exercise the explicit rename policy and ensure the source was untouched.
        output_image_dir = outside / "images" / "train"
        output_image_dir.mkdir(parents=True)
        (output_image_dir / image.name).write_bytes(b"pre-existing destination file")
        before_hash = _sha256(image)
        before_label = label.read_bytes()
        refreshed_preview = client.post("/api/v1/curation/subset/preview", json=preview_input)
        assert refreshed_preview.status_code == 200, refreshed_preview.text
        exported = client.post("/api/v1/curation/subset/export", json={
            **preview_input,
            "preview_token": refreshed_preview.json()["preview_token"],
        })
        assert exported.status_code == 200, exported.text
        report = exported.json()
        assert report["exported"] == 1
        assert report["deduplicated"] == 1
        assert report["unresolved"] == 1
        assert report["skipped"] == 0
        assert report["partial"] is True
        assert str(source) not in exported.text

        output_image = output_image_dir / "sample__1.png"
        output_label = outside / "labels" / "train" / "sample__1.txt"
        assert output_image.is_file()
        assert output_image.read_bytes() == image.read_bytes()
        assert output_label.read_text(encoding="utf-8").splitlines() == [
            "1 0.5 0.5 0.3 0.3", "0 0.5 0.5 0.2 0.2",
        ]
        assert (outside / "classes.txt").read_text(encoding="utf-8").splitlines() == [
            "apple", "zebra",
        ]
        with (outside / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
            manifest = next(csv.DictReader(handle))
        assert manifest["object_ids"] == "0 1"
        assert manifest["source_tag"] == "inspect"
        assert manifest["reason"] == "edge case|second box"
        lineage = json.loads((outside / "lineage.json").read_text(encoding="utf-8"))
        assert "skew" in json.dumps(lineage, ensure_ascii=False).lower()
        assert _sha256(image) == before_hash
        assert label.read_bytes() == before_label

        manifest_only_dir = tmp_path / "manifest-only"
        manifest_only_input = {
            **preview_input,
            "destination": str(manifest_only_dir),
            "mode": "manifest-only",
            "on_exists": "skip",
        }
        manifest_preview = client.post(
            "/api/v1/curation/subset/preview", json=manifest_only_input,
        )
        manifest_export = client.post("/api/v1/curation/subset/export", json={
            **manifest_only_input,
            "preview_token": manifest_preview.json()["preview_token"],
        })
        assert manifest_export.status_code == 200, manifest_export.text
        assert manifest_export.json()["exported"] == 0
        assert (manifest_only_dir / "manifest.csv").is_file()
        assert not list((manifest_only_dir / "images").rglob("sample.png"))

        skip_dir = tmp_path / "skip-existing"
        skip_image = skip_dir / "images" / "train" / image.name
        skip_image.parent.mkdir(parents=True)
        skip_image.write_bytes(b"keep existing destination")
        skip_input = {**preview_input, "destination": str(skip_dir), "on_exists": "skip"}
        skip_preview = client.post("/api/v1/curation/subset/preview", json=skip_input)
        skip_export = client.post("/api/v1/curation/subset/export", json={
            **skip_input,
            "preview_token": skip_preview.json()["preview_token"],
        })
        assert skip_export.status_code == 200, skip_export.text
        assert skip_export.json()["skipped"] == 1
        assert skip_export.json()["exported"] == 0
        assert skip_image.read_bytes() == b"keep existing destination"

        overwrite_dir = tmp_path / "overwrite-existing"
        overwrite_image = overwrite_dir / "images" / "train" / image.name
        overwrite_image.parent.mkdir(parents=True)
        overwrite_image.write_bytes(b"replace only after explicit overwrite choice")
        overwrite_input = {
            **preview_input,
            "destination": str(overwrite_dir),
            "on_exists": "overwrite",
        }
        overwrite_preview = client.post(
            "/api/v1/curation/subset/preview", json=overwrite_input,
        )
        overwrite_export = client.post("/api/v1/curation/subset/export", json={
            **overwrite_input,
            "preview_token": overwrite_preview.json()["preview_token"],
        })
        assert overwrite_export.status_code == 200, overwrite_export.text
        assert overwrite_export.json()["exported"] == 1
        assert overwrite_image.read_bytes() == image.read_bytes()
