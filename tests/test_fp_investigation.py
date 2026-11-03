import copy
import io
import json
import zipfile

import pytest

from fp_cases import CaseStore, FULL_ROI, add_item, review_item
from fp_investigation import (run_case, run_experiment, current_run, paired_evaluation,
                             validate_evaluation_manifest, evaluate_predictions, improvement_gate)
from fp_training_export import export_training
from fp_onnx_demo import make_demo_model
from test_fp_external_runtime import config, image_bytes


@pytest.fixture
def investigation(tmp_path):
    store = CaseStore(tmp_path / "cases")
    case = store.create(store.ingest(image_bytes(), "original.png"), "test")
    item = add_item(case, case["sources"][0]["id"], FULL_ROI, "crack")
    store.save(case)
    case = store.load(case["id"])
    model = make_demo_model(tmp_path / "baseline.onnx")
    return store, case, item["id"], config(model)


def test_persisted_runtime_and_stale_context(investigation):
    store, case, item_id, c = investigation
    run = run_case(case, item_id, store, c)
    assert run["parity"]["status"] == "unavailable"
    assert current_run(case, run)
    case = store.save(case)
    loaded = store.load(case["id"])
    assert loaded["external_runs"][0]["evidence_id"] in {e["id"] for e in loaded["evidence"]}
    loaded["items"][0]["roi"] = [0, 0, .5, .5]
    assert not current_run(loaded, run)
    with pytest.raises(ValueError, match="重新執行"):
        run_experiment(loaded, run["id"], store, target_class="crack", intervention_roi=FULL_ROI)


def test_brightness_control_and_no_cause_promotion(investigation):
    store, case, item_id, c = investigation
    baseline = run_case(case, item_id, store, c)
    result = run_experiment(case, baseline["id"], store, target_class="crack", intervention_roi=FULL_ROI,
                            mode="brightness", factor=.5, target_preserved=True)
    assert result["variants"][0]["score_drop"] > .4
    assert result["variants"][1]["score_drop"] == pytest.approx(0)
    assert case["cause_status"] == "unknown" and case["improvement_status"] == "not_evaluated"
    assert result["available_controls"] == 1
    assert result["variants"][0]["run"]["artifact_dir"]


def test_missing_detection_is_not_score_zero(investigation, tmp_path):
    store, case, item_id, _ = investigation
    c = config(make_demo_model(tmp_path / "det.onnx", detection=True), True)
    baseline = run_case(case, item_id, store, c)
    result = run_experiment(case, baseline["id"], store, target_class="crack", intervention_roi=FULL_ROI,
                            fill=0, controls=2, target_preserved=True)
    changed = result["variants"][0]
    assert changed["target"]["status"] == "unmatched"
    assert changed["score_drop"] is None and changed["status"] == "inconclusive"
    assert result["available_controls"] == 0  # cannot invent different same-sized full-image controls


def manifest(tmp_path, values=(80, 230)):
    rows = []
    for i, (value, truth) in enumerate(zip(values, ["normal", "defect"])):
        path = tmp_path / f"eval_{i}.png"
        path.write_bytes(image_bytes(value))
        rows.append({"id": str(i), "path": str(path), "truth": truth, "verified": True, "group_id": f"lot-{i}"})
    return {"schema_version": 1, "purpose": "independent", "independent_declared": True,
            "split_basis": "separate held-out lots", "items": rows}


def test_paired_eval_pass_and_no_source_mutation(investigation, tmp_path):
    store, case, _, _ = investigation
    before = config(make_demo_model(tmp_path / "old.onnx", bias=.3))
    after = config(make_demo_model(tmp_path / "new.onnx"))
    data = manifest(tmp_path)
    from pathlib import Path
    originals = {r["path"]: Path(r["path"]).read_bytes() for r in data["items"]}
    result = paired_evaluation(case, store, before, after, data, min_support=1, min_fp_reduction=.5)
    assert result["baseline"]["false_alarms"] == 1 and result["candidate"]["false_alarms"] == 0
    assert result["candidate"]["misses"] == 0 and result["gate"]["status"] == "passed"
    assert len(result["dataset_fingerprint"]) == 64
    assert all(Path(p).read_bytes() == v for p, v in originals.items())
    assert "case_regression" not in result["gate"]["status"]


def test_more_misses_fail_even_if_false_alarms_drop(investigation, tmp_path):
    store, case, _, _ = investigation
    before = config(make_demo_model(tmp_path / "old.onnx", bias=.3))
    after = config(make_demo_model(tmp_path / "new.onnx", gain=.5))
    result = paired_evaluation(case, store, before, after, manifest(tmp_path), min_support=1)
    assert result["candidate"]["false_alarms"] < result["baseline"]["false_alarms"]
    assert result["candidate"]["misses"] > result["baseline"]["misses"]
    assert result["gate"]["status"] == "failed"
    assert any("漏檢" in r for r in result["gate"]["reasons"])


def test_holdout_overlap_duplicate_and_missing_truth_rejected(investigation, tmp_path):
    store, case, _, _ = investigation
    data = manifest(tmp_path, values=(220, 230))
    with pytest.raises(ValueError, match="重疊"):
        validate_evaluation_manifest(data, store)
    data["purpose"] = "case_regression"
    snapshot, _ = validate_evaluation_manifest(data, store)
    assert snapshot["purpose"] == "case_regression"
    data["items"].append({**data["items"][0], "id": "different-id"})
    with pytest.raises(ValueError, match="重複影像"):
        validate_evaluation_manifest(data, store)
    data = manifest(tmp_path)
    data["items"][0]["verified"] = False
    with pytest.raises(ValueError, match="真值"):
        validate_evaluation_manifest(data, store)


def test_detection_metrics_catch_wrong_location_and_support():
    rows = [{"truth": "normal", "boxes": [], "annotations_complete": True},
            {"truth": "defect", "boxes": [{"class": "crack", "box": [0, 0, .2, .2]}], "annotations_complete": True}]
    preds = [{"task": "detection", "alarm": False, "detections": []},
             {"task": "detection", "alarm": True, "detections": [{"class": "crack", "box": [.6, .6, 1, 1], "score": .99}]}]
    score = evaluate_predictions(rows, preds, normal_classes=[])
    assert score["misses"] == 0  # image-level alarm alone does not locate the actual defect
    assert score["detection"]["fn"] == 1 and score["detection"]["fp"] == 1
    assert improvement_gate(score, score, purpose="independent")["status"] == "insufficient"
    assert improvement_gate(score, score, purpose="case_regression")["status"] == "case_regression_only"


def test_training_export_mixed_regions_preserves_true_box(investigation):
    store, case, first_id, _ = investigation
    case["items"][0]["roi"] = [0, 0, .4, .4]
    review_item(case, first_id, "normal", "E", "normal reflection")
    second = add_item(case, case["sources"][0]["id"], [.5, .5, 1, 1], "predicted-name")
    review_item(case, second["id"], "defect", "E", "true crack")
    sha = case["sources"][0]["sha256"]
    params = dict(complete_images=[sha], class_assignments={"item:" + second["id"]: "crack"},
                  class_names=["crack"], reviewer="E", reason="full image checked")
    record, payload = export_training(case, store, **params)
    assert record["n_boxes"] == 1 and record["n_negative_images"] == 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert archive.read("labels/"+sha+".txt").decode().startswith("0 0.750000000 0.750000000 0.500000000")
    case["items"][0]["roi"] = FULL_ROI.copy()
    with pytest.raises(ValueError, match="矛盾"):
        export_training(case, store, **params)


def test_partial_normal_cannot_be_exported_as_negative(investigation):
    store, case, item_id, _ = investigation
    case["items"][0]["roi"] = [0, 0, .5, .5]
    review_item(case, item_id, "normal", "E", "only this area")
    kwargs = dict(complete_images=[case["sources"][0]["sha256"]], class_assignments={}, class_names=["crack"], reviewer="E", reason="review")
    with pytest.raises(ValueError, match="局部正常"):
        export_training(case, store, **kwargs)
    case["items"][0]["roi"] = FULL_ROI.copy()
    record, data = export_training(case, store, **kwargs)
    assert record["n_negative_images"] == 1
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.read("labels/"+case["sources"][0]["sha256"]+".txt") == b""


def test_screenshot_provenance_blocks_training_even_as_reference(investigation):
    store, case, item_id, _ = investigation
    case["sources"][0]["kind"] = "reference"
    case["sources"][0]["original_kind"] = "screenshot"
    review_item(case, item_id, "normal", "E", "visual check")
    with pytest.raises(ValueError, match="截圖"):
        export_training(case, store, complete_images=[case["sources"][0]["sha256"]], class_assignments={},
                        class_names=["crack"], reviewer="E", reason="checked")
