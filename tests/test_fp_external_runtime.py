import copy
import io

import numpy as np
import pytest
from PIL import Image

from fp_external_runtime import Runtime, DEFAULT_CONFIG, prepare_input, decode_output, compare_reference, config_checked
from fp_onnx_demo import make_demo_model


def image_bytes(value=220, size=(32, 32)):
    out = io.BytesIO()
    Image.new("RGB", size, (value,)*3).save(out, format="PNG")
    return out.getvalue()


def config(path, detection=False):
    return {**DEFAULT_CONFIG, "model_path": str(path), "model_version": "test-v1", "input_size": [32, 32],
            "output_format": "xyxy" if detection else "classification", "resize": "stretch",
            "class_names": ["crack"] if detection else ["good", "crack"],
            "normal_classes": [] if detection else ["good"], "threshold": .5}


def test_real_onnx_classification_and_artifacts(tmp_path):
    runtime = Runtime(config(make_demo_model(tmp_path / "model.onnx")))
    run = runtime.infer(image_bytes(), tmp_path / "runs")
    assert run["prediction"]["class"] == "crack"
    assert run["prediction"]["score"] == pytest.approx(220/255, abs=1e-5)
    assert run["prediction"]["alarm"]
    assert len(run["contract_id"]) == 64
    with np.load(tmp_path / "runs" / run["id"] / "tensors.npz") as arrays:
        assert arrays["input"].shape == (1, 3, 32, 32)
        assert arrays["output"].shape == (1, 2)
    assert runtime.infer(image_bytes(20))["prediction"]["alarm"] is False


def test_letterbox_geometry_color_and_normalization():
    c = {**DEFAULT_CONFIG, "input_size": [32, 32], "color": "BGR", "mean": [.1, .2, .3], "std": [1, 2, 3]}
    tensor, preview, geom = prepare_input(Image.new("RGB", (64, 32), (255, 128, 0)), c)
    assert geom["padding_xy"] == [0, 8] and geom["resized_size"] == [32, 16]
    assert tensor[0, :, 16, 16] == pytest.approx([-.1, (128/255-.2)/2, (1-.3)/3])
    pred = decode_output(np.array([[[0, 8, 32, 24, .9, 0]]]), {**c, "output_format": "xyxy"}, geom)
    assert pred["detections"][0]["box"] == [0, 0, 1, 1]
    assert pred["nms_applied"] is False


def test_yolo_layouts_and_nms_are_explicit():
    c = {**DEFAULT_CONFIG, "input_size": [32, 32]}
    _, _, geom = prepare_input(Image.new("RGB", (32, 32)), c)
    raw = np.array([[[16, 16], [16, 16], [20, 20], [20, 20], [.9, .8]]])
    pred = decode_output(raw, c, geom)
    assert len(pred["detections"]) == 1 and pred["candidate_count"] == 2
    assert pred["raw_candidates_available"]
    rows = np.array([[[16, 16, 20, 20, .8, .9]]])
    pred = decode_output(rows, {**c, "output_format": "yolo_v5"}, geom)
    assert pred["detections"][0]["score"] == pytest.approx(.72)
    with pytest.raises(ValueError, match=r"4\+C"):
        decode_output(rows, c, geom)


def test_full_historical_parity_no_self_certification(tmp_path):
    run = Runtime(config(make_demo_model(tmp_path / "det.onnx", detection=True), True)).infer(image_bytes())
    assert compare_reference(run, run)["status"] == "unavailable"
    ref = {**copy.deepcopy(run), "origin": "deployment", "recorded_at": "2026-09-01T00:00:00Z"}
    assert compare_reference(run, ref)["status"] == "matched"
    assert compare_reference(run, ref, "screenshot")["status"] == "unavailable"
    changed = copy.deepcopy(ref)
    changed["prediction"]["detections"][0]["box"][0] += .05
    assert compare_reference(run, changed)["status"] == "mismatched"
    changed = copy.deepcopy(ref)
    changed["prediction"]["detections"].append(changed["prediction"]["detections"][0])
    assert compare_reference(run, changed)["status"] == "mismatched"
    ref["contract_id"] = "other"
    assert compare_reference(run, ref)["status"] == "unavailable"


def test_model_content_and_settings_change_identity(tmp_path):
    path = make_demo_model(tmp_path / "model.onnx")
    first = Runtime(config(path))
    other = Runtime({**config(path), "threshold": .8})
    assert first.identity["contract_id"] != other.identity["contract_id"]
    make_demo_model(path, bias=.1)
    newer = Runtime(config(path))
    assert first.identity["contract_id"] != newer.identity["contract_id"]
    assert first.infer(image_bytes(100))["prediction"]["score"] != newer.infer(image_bytes(100))["prediction"]["score"]


@pytest.mark.parametrize("updates", [{"std": [1, 0, 1]}, {"threshold": float("nan")},
                                     {"class_names": ["a", "a"]}, {"input_size": [0, 32]}, {"new_field": 1}])
def test_invalid_contract_rejected(updates):
    with pytest.raises(ValueError):
        config_checked({**DEFAULT_CONFIG, **updates})


def test_wrong_input_size_fails_before_prediction(tmp_path):
    r = Runtime({**config(make_demo_model(tmp_path / "model.onnx")), "input_size": [64, 64]})
    with pytest.raises(ValueError, match="輸入尺寸"):
        r.infer(image_bytes())


def test_parity_close_boxes_matching_is_order_independent():
    def row(x):
        return {"class": "crack", "score": .9, "box": [x, .1, .6, .6]}
    run = {"image_sha256": "image", "contract_id": "model", "prediction": {
        "task": "detection", "alarm": True, "detections": [row(.10), row(.14)]}}
    reference = {**copy.deepcopy(run), "origin": "deployment", "recorded_at": "known"}
    reference["prediction"]["detections"] = [row(.12), row(.08)]
    assert compare_reference(run, reference, box_atol=.03)["status"] == "matched"
