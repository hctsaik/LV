from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from react_web_shell import create_app


def test_react_curation_handoff_deduplicates_guards_override_and_records_rounds(
    monkeypatch, tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    inbox_dir = tmp_path / "labeling-inbox"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("LV_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("CIM_LOG_DIR", str(inbox_dir))

    first = tmp_path / "dataset-a" / "train" / "apple" / "a.png"
    second = tmp_path / "dataset-b" / "train" / "apple" / "b.png"
    for path in (first, second):
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 18), (120, 30, 80)).save(first)
    shutil.copyfile(first, second)

    application = create_app(model_catalog=lambda: [], model_loader=lambda _model: None)
    with TestClient(application) as client:
        added, skipped = application.state.curation_store.add([
            {"path": str(first), "filename": first.name, "label": "apple", "split": "train", "source": "manual"},
            {"path": str(second), "filename": second.name, "label": "apple", "split": "train", "source": "manual"},
        ])
        assert (added, skipped) == (2, 0)
        item_ids = [item["id"] for item in application.state.curation_store.list_items()]

        preview = client.post("/api/v1/curation/labeling-handoff/preview", json={
            "item_ids": item_ids,
        })
        assert preview.status_code == 200, preview.text
        assert preview.json() == {
            "selected": 2,
            "available": 2,
            "unavailable": 0,
            "unique_images": 1,
            "deduplicated": 1,
            "new_images": 1,
            "handled_images": 0,
            "sendable_images": 1,
            "skipped_handled": 0,
        }

        sent = client.post("/api/v1/curation/labeling-handoff/send", json={
            "item_ids": item_ids,
        })
        assert sent.status_code == 201, sent.text
        response = sent.json()
        assert response["sent"] == 1
        assert response["deduplicated"] == 1
        assert response["saved_to_local_inbox"] is True
        assert response["created_round"] is True
        assert str(tmp_path) not in sent.text

        handoff_dir = inbox_dir / "lv_labeling_handoff" / response["handoff_id"]
        spec = json.loads((handoff_dir / "_handoff.json").read_text(encoding="utf-8"))
        assert spec["task"] == "re-label"
        assert [item["filename"] for item in spec["items"]] == [first.name]
        assert (handoff_dir / "classes.txt").read_text(encoding="utf-8").splitlines() == ["apple"]

        guarded = client.post("/api/v1/curation/labeling-handoff/preview", json={
            "item_ids": item_ids,
        })
        assert guarded.status_code == 200
        assert guarded.json()["handled_images"] == 1
        assert guarded.json()["sendable_images"] == 0
        assert guarded.json()["skipped_handled"] == 1
        blocked = client.post("/api/v1/curation/labeling-handoff/send", json={
            "item_ids": item_ids,
        })
        assert blocked.status_code == 422
        assert blocked.json()["code"] == "CURATION_HANDOFF_EMPTY"

        overridden = client.post("/api/v1/curation/labeling-handoff/send", json={
            "item_ids": item_ids,
            "include_handled": True,
        })
        assert overridden.status_code == 201, overridden.text
        assert overridden.json()["sent"] == 1
        assert overridden.json()["skipped_handled"] == 0
        assert overridden.json()["handoff_id"] == response["handoff_id"]

        registry_path = cache_dir / "pool_registry.jsonl"
        registry = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines()]
        assert len([row for row in registry if row["event"] == "sent"]) == 2
        assert len({row["sha256"] for row in registry if row["event"] == "sent"}) == 1

        rounds_response = client.get("/api/v1/rounds")
        assert rounds_response.status_code == 200
        rounds = rounds_response.json()["rounds"]
        assert len(rounds) == 2
        assert all(row["n_sent"] == 1 and row["strategy"] == "cart" for row in rounds)

        round_id = rounds[-1]["round_id"]
        metric = client.post(f"/api/v1/rounds/{round_id}/metrics", json={
            "name": "review_acc", "value": 0.875,
        })
        assert metric.status_code == 200, metric.text
        saved_round = next(row for row in metric.json()["rounds"] if row["round_id"] == round_id)
        assert saved_round["metrics"][-1]["name"] == "review_acc"
        assert saved_round["metrics"][-1]["value"] == 0.875


def test_react_labeling_readback_updates_run_overlay_pool_round_and_audit(
    monkeypatch, tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    inbox_dir = tmp_path / "labeling-inbox"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("LV_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("CIM_LOG_DIR", str(inbox_dir))

    image = tmp_path / "dataset" / "train" / "old-label" / "image.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (24, 18), (80, 120, 30)).save(image)
    source_sha = hashlib.sha256(image.read_bytes()).hexdigest()

    application = create_app(model_catalog=lambda: [], model_loader=lambda _model: None)
    with TestClient(application) as client:
        application.state.curation_store.add([{
            "path": str(image), "filename": image.name, "label": "old-label",
            "split": "train", "source": "manual",
        }])
        item_id = application.state.curation_store.list_items()[0]["id"]
        sent = client.post("/api/v1/curation/labeling-handoff/send", json={"item_ids": [item_id]})
        assert sent.status_code == 201, sent.text
        handoff_id = sent.json()["handoff_id"]
        handoff_dir = inbox_dir / "lv_labeling_handoff" / handoff_id
        spec = json.loads((handoff_dir / "_handoff.json").read_text(encoding="utf-8"))
        sidecar = handoff_dir / "images" / Path(spec["items"][0]["image"]).with_suffix(".json").name

        run_id = "current-image-run"
        application.state.runs[run_id] = {
            "mode": "classifier",
            "unit": "image",
            "records": [{"index": 0, "filename": image.name, "label": "old-label"}],
            "points": [{"index": 0, "label": "old-label"}],
            "views": [{"points": [{"index": 0, "label": "old-label"}]}],
            "image_paths": [image],
            "_source_sha256": [source_sha],
        }
        application.state.current_run_id = run_id

        pending = client.get("/api/v1/labeling/handoffs")
        assert pending.status_code == 200
        assert pending.json()["items"][0]["n_annotated"] == 0
        assert str(tmp_path) not in pending.text

        not_ready = client.post(f"/api/v1/labeling/handoffs/{handoff_id}/readback", json={"run_id": run_id})
        assert not_ready.status_code == 409
        assert not_ready.json()["code"] == "LABELING_READBACK_NOT_READY"

        sidecar.write_text(json.dumps({
            "shapes": [{"label": "new-label"}], "annotator": "reviewer-1",
        }), encoding="utf-8")
        readback = client.post(f"/api/v1/labeling/handoffs/{handoff_id}/readback", json={"run_id": run_id})
        assert readback.status_code == 200, readback.text
        result = readback.json()
        assert result["n_labeled"] == 1
        assert result["n_changed"] == 1
        assert result["complete"] is True
        assert result["annotator"] == "reviewer-1"
        assert result["changes"] == [{
            "filename": image.name, "old_label": "old-label", "new_label": "new-label",
        }]
        assert str(tmp_path) not in readback.text

        updated_run = client.get(f"/api/v1/runs/{run_id}").json()
        assert updated_run["records"][0]["label"] == "new-label"
        assert updated_run["points"][0]["label"] == "new-label"
        assert updated_run["views"][0]["points"][0]["label"] == "new-label"

        audit = client.get("/api/v1/readback/audit")
        assert audit.status_code == 200
        assert audit.json()["summary"]["n_images"] == 1
        assert audit.json()["events"][0]["sha256"] == source_sha
        assert audit.json()["events"][0]["annotator"] == "reviewer-1"
        assert str(tmp_path) not in audit.text
        csv_response = client.get("/api/v1/readback/audit.csv")
        assert csv_response.status_code == 200
        assert b"new-label" in csv_response.content

        replay = client.post(f"/api/v1/labeling/handoffs/{handoff_id}/readback", json={"run_id": run_id})
        assert replay.status_code == 409
        assert replay.json()["code"] == "LABELING_READBACK_ALREADY_APPLIED"
        assert hashlib.sha256(image.read_bytes()).hexdigest() == source_sha

