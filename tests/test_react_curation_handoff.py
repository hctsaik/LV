from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

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


def test_duplicate_pairs_handoff_preserves_pair_metadata_and_guards_handled_images(
    monkeypatch, tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    inbox_dir = tmp_path / "labeling-inbox"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("LV_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("CIM_LOG_DIR", str(inbox_dir))

    train = tmp_path / "dataset" / "train" / "apple" / "train.png"
    test = tmp_path / "dataset" / "test" / "apple" / "test.png"
    train.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    Image.new("RGB", (24, 18), (120, 30, 80)).save(train)
    Image.new("RGB", (24, 18), (80, 120, 30)).save(test)
    paths = [train, test]
    source_shas = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]

    application = create_app(model_catalog=lambda: [], model_loader=lambda _model: None)
    run_id = "duplicate-handoff-fixture"
    application.state.runs[run_id] = {
        "mode": "classifier",
        "unit": "image",
        "records": [
            {"index": 0, "filename": train.name, "label": "apple", "split": "train"},
            {"index": 1, "filename": test.name, "label": "apple", "split": "test"},
        ],
        "image_paths": paths,
        "_source_sha256": source_shas,
        "_phashes": ["0000000000000000", "0000000000000000"],
    }

    with TestClient(application) as client:
        response = client.post(f"/api/v1/runs/{run_id}/duplicates/labeling-handoff", json={
            "method": "phash", "max_hamming": 0, "cross_split_only": True,
        })
        assert response.status_code == 201, response.text
        result = response.json()
        assert result["sent"] == 2
        assert result["pair_count"] == 1
        assert result["task"] == "verify"
        assert result["saved_to_local_inbox"] is True
        assert str(tmp_path) not in response.text

        handoff_dir = inbox_dir / "lv_labeling_handoff" / result["handoff_id"]
        spec = json.loads((handoff_dir / "_handoff.json").read_text(encoding="utf-8"))
        assert spec["source"] == "duplicate"
        assert spec["task"] == "verify"
        assert spec["payload"]["kind"] == "duplicate_pair_review"
        assert spec["payload"]["pairs"] == [{
            "left_sha256": source_shas[0],
            "right_sha256": source_shas[1],
            "left_filename": train.name,
            "right_filename": test.name,
            "left_split": "train",
            "right_split": "test",
            "distance": 0,
        }]
        assert {item["sha256"] for item in spec["items"]} == set(source_shas)

        continued = client.post(f"/api/v1/runs/{run_id}/duplicates/labeling-handoff", json={
            "method": "phash", "max_hamming": 0, "cross_split_only": True,
            "include_handled": True, "new_round": False,
            "round_id": result["round_id"],
        })
        assert continued.status_code == 201, continued.text
        assert continued.json()["round_id"] == result["round_id"]
        assert continued.json()["created_round"] is False

        blocked = client.post(f"/api/v1/runs/{run_id}/duplicates/labeling-handoff", json={
            "method": "phash", "max_hamming": 0, "cross_split_only": True,
        })
        assert blocked.status_code == 422
        assert blocked.json()["code"] == "DUPLICATE_HANDOFF_EMPTY"

        overridden = client.post(f"/api/v1/runs/{run_id}/duplicates/labeling-handoff", json={
            "method": "phash", "max_hamming": 0, "cross_split_only": True,
            "include_handled": True,
        })
        assert overridden.status_code == 201, overridden.text
        assert overridden.json()["sent"] == 2
        assert overridden.json()["skipped_handled"] == 0

        train.write_bytes(b"changed after scan")
        changed = client.post(f"/api/v1/runs/{run_id}/duplicates/labeling-handoff", json={
            "method": "phash", "max_hamming": 0, "cross_split_only": True,
            "include_handled": True,
        })
        assert changed.status_code == 409
        assert changed.json()["code"] == "DUPLICATE_SOURCE_CHANGED"


def test_duplicate_pair_answers_csv_counts_and_reads_keep_drop_by_sha256(
    monkeypatch, tmp_path: Path,
):
    output_dir = tmp_path / "output"
    inbox_dir = tmp_path / "labeling-inbox"
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LV_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("CIM_LOG_DIR", str(inbox_dir))

    train = tmp_path / "dataset" / "train" / "apple" / "train.png"
    test = tmp_path / "dataset" / "test" / "apple" / "test.png"
    train.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    Image.new("RGB", (24, 18), (120, 30, 80)).save(train)
    Image.new("RGB", (24, 18), (80, 120, 30)).save(test)
    paths = [train, test]
    source_shas = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]

    application = create_app(model_catalog=lambda: [], model_loader=lambda _model: None)
    run_id = "duplicate-answer-readback"
    application.state.runs[run_id] = {
        "mode": "classifier",
        "unit": "image",
        "records": [
            {"index": 0, "filename": train.name, "label": "apple", "split": "train"},
            {"index": 1, "filename": test.name, "label": "apple", "split": "test"},
        ],
        "image_paths": paths,
        "_source_sha256": source_shas,
        "_phashes": ["0000000000000000", "0000000000000000"],
    }
    application.state.current_run_id = run_id

    with TestClient(application) as client:
        sent = client.post(f"/api/v1/runs/{run_id}/duplicates/labeling-handoff", json={
            "method": "phash", "max_hamming": 0, "cross_split_only": True,
        })
        assert sent.status_code == 201, sent.text
        handoff_id = sent.json()["handoff_id"]
        handoff_dir = inbox_dir / "lv_labeling_handoff" / handoff_id
        (handoff_dir / "answers.csv").write_text(
            "sha256,label\n" + f"{source_shas[0]},keep\n{source_shas[1]},drop\n",
            encoding="utf-8",
        )

        pending = client.get("/api/v1/labeling/handoffs").json()["items"]
        row = next(item for item in pending if item["handoff_id"] == handoff_id)
        assert row["n_annotated"] == 2

        readback = client.post(
            f"/api/v1/labeling/handoffs/{handoff_id}/readback", json={"run_id": run_id},
        )
        assert readback.status_code == 200, readback.text
        result = readback.json()
        assert result["n_labeled"] == 2
        assert result["n_changed"] == 2
        assert result["complete"] is True
        assert [record["label"] for record in application.state.runs[run_id]["records"]] == ["keep", "drop"]
        assert [change["new_label"] for change in result["changes"]] == ["keep", "drop"]
        assert str(tmp_path) not in readback.text


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


def test_react_round_probe_records_metrics_and_reports_insufficient_classes(
    monkeypatch, tmp_path: Path,
):
    output_dir = tmp_path / "output"
    monkeypatch.setenv("LV_OUTPUT_DIR", str(output_dir))
    application = create_app(model_catalog=lambda: [], model_loader=lambda _model: None)
    with TestClient(application) as client:
        import round_ledger

        round_id = round_ledger.start_round("cart", {}, "cart")
        run_id = "probe-run"
        labels = ["apple"] * 4 + ["banana"] * 4
        vectors = np.asarray([
            [1.0, 0.0], [1.1, 0.0], [0.9, 0.1], [1.0, 0.1],
            [-1.0, 0.0], [-1.1, 0.0], [-0.9, -0.1], [-1.0, -0.1],
        ])
        application.state.runs[run_id] = {
            "mode": "classifier", "unit": "image", "model_id": "fixture-model",
            "records": [{"index": index, "filename": f"{index}.png", "label": label}
                        for index, label in enumerate(labels)],
        }
        application.state.current_run_id = run_id
        application.state.raw_embeddings[run_id] = {"fixture-model": vectors}

        response = client.post(f"/api/v1/rounds/{round_id}/probe", json={"run_id": run_id})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["model_id"] == "fixture-model"
        assert result["acc"] == 1.0
        assert result["macro_f1"] == 1.0
        assert result["n_train"] == 6 and result["n_test"] == 2
        assert {metric["name"] for metric in round_ledger.load_rounds()[0]["metrics"]} == {
            "probe_acc", "probe_macro_f1",
        }

        application.state.runs[run_id]["records"] = application.state.runs[run_id]["records"][:6]
        application.state.raw_embeddings[run_id]["fixture-model"] = vectors[:6]
        insufficient = client.post(f"/api/v1/rounds/{round_id}/probe", json={"run_id": run_id})
        assert insufficient.status_code == 422
        assert insufficient.json()["code"] == "PROBE_INSUFFICIENT_SAMPLES"
        assert "insufficient_class:banana" in insufficient.json()["message"]
        assert len(round_ledger.load_rounds()[0]["metrics"]) == 2


def test_react_learning_curve_advice_uses_labeled_counts_and_never_auto_stops(
    monkeypatch, tmp_path: Path,
):
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    monkeypatch.setenv("LV_OUTPUT_DIR", str(output_dir))
    rows = []
    for index, value in enumerate((0.70, 0.80, 0.802), start=1):
        round_id = f"probe-round-{index}"
        stamp = f"2026-07-12T00:00:0{index}+00:00"
        rows.extend([
            {"kind": "start", "round_id": round_id, "ts": stamp, "strategy": "cart",
             "params": {}, "tool": "cart", "note": ""},
            {"kind": "batch", "round_id": round_id, "batch_id": f"b{index}", "n_sent": 100},
            {"kind": "readback", "round_id": round_id, "batch_id": f"b{index}",
             "n_labeled": 100, "n_changed": 10},
            {"kind": "metric", "round_id": round_id, "name": "probe_acc", "value": value,
             "source": "probe", "ts": stamp},
        ])
    (output_dir / "rounds.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
    )

    with TestClient(create_app(model_catalog=lambda: [], model_loader=lambda _model: None)) as client:
        advice = client.get("/api/v1/rounds/learning-curve/advice?metric_name=probe_acc")
        assert advice.status_code == 200, advice.text
        payload = advice.json()
        assert payload["xs"] == [100, 200, 300]
        assert payload["ys"] == [0.70, 0.80, 0.802]
        assert payload["advice"]["status"] == "plateau"
        assert payload["advice"]["slope_per_100"] == pytest.approx(0.002)
        assert "報酬遞減" in payload["advice"]["reason"]

