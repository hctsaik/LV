"""Pinned 8.4.142 real architectures; engineering correctness, not trained-model efficacy."""
import copy
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from yolo_attribution import AttributionError, YoloRuntime, inspect_model, summarize_faithfulness
from detector_attribution_contracts import DetectionTargetV2, DetectorRunSpecV2, identity

MODERN = {"runtime_id": "modern-8.4.142", "runtime_release": "8.4.142"}


def picture():
    a = np.full((64, 96, 3), 70, dtype=np.uint8)
    a[12:40, 18:54] = [210, 90, 40]
    out = io.BytesIO()
    Image.fromarray(a).save(out, format="PNG")
    return out.getvalue()


@pytest.fixture(scope="module", params=[("yolov8", "one_to_many"), ("yolo11", "one_to_many"),
                                       ("yolo12", "one_to_many"), ("yolo26", "one_to_many"), ("yolo26", "one_to_one")])
def modern(request, tmp_path_factory):
    import torch
    from ultralytics.nn.tasks import DetectionModel
    torch.set_num_threads(1)
    torch.manual_seed(17)
    family, head = request.param
    root = tmp_path_factory.mktemp(family + head)
    model = DetectionModel(family + "n.yaml", ch=3, nc=2, verbose=False)
    if head == "one_to_one":
        # Still untrained: avoid the default near-constant confidence field when
        # testing ordered ONNX correspondence. Exact ties have a separate gate.
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_var.fill_(.15)
        with torch.no_grad():
            for branch in model.model[-1].one2one_cv2:
                branch[-1].weight.zero_()
                branch[-1].bias.fill_(.5)
            for branch in model.model[-1].one2one_cv3:
                branch[-1].weight.mul_(1000.)
    model.names = {0: "scratch", 1: "dent"}
    path = root / "original.pt"
    torch.save({"model": model.eval()}, path)
    info = inspect_model(path, trusted=True, config={**MODERN, "head_path": head})
    config = info["default_config"]
    config.update(input_size=[64, 64], threshold=0., max_detections=20)
    return YoloRuntime(config, trusted=True), path, info, root


def test_real_modern_forward_gradient_trace_and_artifacts(modern):
    runtime, path, info, root = modern
    assert info["family"] == runtime.family
    assert info["runtime_release"] == "8.4.142"
    before = len(runtime.model.model[-1]._forward_pre_hooks)
    run = runtime.infer(picture(), root)
    row = run["prediction"]["detections"][0]
    result = runtime.gradcam(picture(), output_index=row["output_index"], class_index=row["class_index"], method="signed", artifact_root=root)
    assert result["schema_version"] == 3
    assert result["target"]["native_candidate_index"] == row["native_candidate_index"]
    assert result["target"]["execution"]["head_path"] == runtime.head_path
    assert result["forward_parity"]["status"] == "matched"
    assert result["gradient_abs_max"] > 0
    assert np.isclose(result["score"], row["score"], atol=1e-7)
    assert len(runtime.model.model[-1]._forward_pre_hooks) == before
    assert runtime.infer(picture())["output_sha256"] == run["output_sha256"]
    saved = json.loads(Path(result["artifact_dir"], "attribution.json").read_text(encoding="utf-8"))
    assert saved["target"] == json.loads(json.dumps(result["target"]))
    assert result["analysis_id"] == identity({"cam_compute_id": result["compute_id"], "validation_id": None})
    with pytest.raises(AttributionError, match="class differs"):
        runtime.gradcam(picture(), output_index=row["output_index"], class_index=1-row["class_index"])
    assert len(runtime.model.model[-1]._forward_pre_hooks) == before


def test_real_modern_public_export_matches_onnx(modern):
    import torch
    import onnxruntime as ort
    runtime, _, _, root = modern
    class Public(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)[0]
    model = Public(runtime.model).eval()
    x = torch.from_numpy(runtime._prepare(picture())[0])
    with torch.no_grad():
        expected = model(x).numpy()
    assert np.isfinite(expected).all()
    assert runtime.infer(picture())["prediction"]["detections"]
    target = root / "public.onnx"
    torch.onnx.export(model, x, str(target), opset_version=17, input_names=["images"], output_names=["predictions"], dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    actual = ort.InferenceSession(str(target), sess_options=options, providers=["CPUExecutionProvider"]).run(None, {"images": x.numpy()})[0]
    if runtime.head_path == "one_to_one":
        from fp_external_runtime import Runtime
        from fp_attribution import _match_target
        def sorted_unique(a):
            rows = a[0]
            keys = [tuple(row[[5,0,1,2,3]]) for row in rows]
            assert len(set(keys)) == len(keys), "Non-degenerate export gate requires unique class/box identities"
            return rows[sorted(range(len(rows)), key=lambda i: keys[i])]
        np.testing.assert_allclose(sorted_unique(actual), sorted_unique(expected), atol=1e-4, rtol=1e-4)
        deployed = Runtime({**runtime.core_config, "model_path": str(target)}).infer(picture())
        gradient = runtime.infer(picture())
        selected, matched = _match_target(deployed, gradient, deployed["prediction"]["detections"][0])
        assert matched["status"] == "per_case_matched" and "native_candidate_index" in selected
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    assert runtime.infer(picture())["prediction"]["nms_applied"] == (runtime.head_path == "one_to_many")


def test_fixed_native_target_survives_reordered_topk(modern, tmp_path):
    import torch
    runtime, _, _, _ = modern
    row = runtime.infer(picture())["prediction"]["detections"][0]
    cam = runtime.gradcam(picture(), output_index=row["output_index"], method="signed")
    target = cam["target"]
    # Use an explicit compact mask so equal-shape translated controls exist.
    scores = np.zeros((64, 96), dtype=np.float32)
    scores[8:16, 8:16] = 1
    report = runtime.faithfulness(picture(), target, scores, fractions=(.01,), controls=1, artifact_root=tmp_path)
    assert report["protocol_version"] == 2
    assert set(r["fill_method"] for r in report["rows"]) == {"constant", "blur"}
    for record in report["rows"]:
        assert not record.get("unavailable")
        t = torch.from_numpy(runtime._prepare(Path(record["modified_image"]).read_bytes())[0])
        with torch.no_grad():
            _, raw, _ = runtime._forward(t)
            expected = runtime._scalar(raw, target["native_candidate_index"], target["class_index"], target["scalar"]).item()
        assert record["fixed_scalar"] == pytest.approx(expected, abs=1e-9)
    changed_fill = runtime.faithfulness(picture(), target, scores, fractions=(.01,), controls=1, fill_methods=("constant",))
    changed_overlap = runtime.faithfulness(picture(), target, scores, fractions=(.01,), controls=1, max_overlap=0.)
    assert len({report["validation_id"], changed_fill["validation_id"], changed_overlap["validation_id"]}) == 3
    assert len({identity({"cam_compute_id": cam["compute_id"], "validation_id": r["validation_id"]})
                for r in (report, changed_fill, changed_overlap)}) == 3
    changed = copy.deepcopy(target)
    changed["execution"]["head_path"] = "another_branch"
    with pytest.raises(AttributionError, match="contract differs"):
        runtime.faithfulness(picture(), changed, scores, fractions=(.01,), controls=1)


def test_runtime_contract_and_legacy_hash_unchanged(modern):
    runtime, path, info, _ = modern
    with pytest.raises(AttributionError, match="8.3.203"):
        inspect_model(path, trusted=True)
    with pytest.raises(AttributionError, match="release differs"):
        inspect_model(path, trusted=True, config={**MODERN, "runtime_release": "wrong"})
    with pytest.raises(AttributionError, match="conflicts"):
        inspect_model(path, trusted=True, config={**MODERN, "head_path": "one_to_one", "nms": True})
    cfg = copy.deepcopy(info["default_config"])
    cfg["output_contract"] = {"format": "wrong"}
    with pytest.raises(AttributionError, match="contract differs"):
        YoloRuntime(cfg, trusted=True)
    target = DetectionTargetV2("a"*64, "b"*64, 1, 0, "scratch", (.1,.2,.8,.9))
    spec = DetectorRunSpecV2(target, {"input_size": [64,64]}, {"level": 0})
    assert "execution" not in target.as_dict()
    assert spec.key == identity(spec.as_dict())


def test_one2one_duplicate_anchor_classes_trace_and_snapshot(modern):
    import torch
    runtime, _, _, _ = modern
    if runtime.head_path != "one_to_one":
        pytest.skip("one-to-one-specific top-k contract")
    head = runtime.model.model[-1]
    # Top-k can legitimately retain two different classes for one native anchor.
    scores = torch.tensor([[[.99,.98],[.3,.2],[.1,.05]]])
    values, labels, indices = head.get_topk_index(scores, 3)
    assert indices[0,:2].tolist() == [0,0]
    assert labels[0,:2,0].tolist() == [0,1]
    row = runtime.infer(picture())["prediction"]["detections"][0]
    cam = runtime.gradcam(picture(), output_index=row["output_index"], method="signed")
    clone = object.__new__(YoloRuntime)
    clone.__dict__ = {**runtime.__dict__, "model": copy.deepcopy(runtime.model)}
    # Force a different ranking; fixed snapshot must keep original native target.
    with torch.no_grad():
        for branch in clone.model.model[-1].one2one_cv3:
            branch[-1].bias[cam["target"]["class_index"]] -= 20
            branch[-1].bias[1-cam["target"]["class_index"]] += 20
    replacement = next(r for r in clone.infer(picture())["prediction"]["detections"] if r["output_index"] == row["output_index"])
    assert replacement["class_index"] != cam["target"]["class_index"]
    fixed = clone.gradcam(picture(), output_index=row["output_index"], class_index=cam["target"]["class_index"],
                         method="signed", _target_snapshot=cam["target"])
    assert fixed["target"] == cam["target"]
    t = torch.from_numpy(clone._prepare(picture())[0])
    with torch.no_grad():
        _, raw, _ = clone._forward(t)
        expected = clone._scalar(raw, cam["target"]["native_candidate_index"], cam["target"]["class_index"], "confidence")
    assert fixed["score"] == pytest.approx(expected.item())


def test_summary_partitions_execution_and_complete_protocol():
    report = {"status": "measured", "protocol_version": 2, "controls": 1, "fractions": [.1],
              "fill_methods": ["constant", "blur"], "protocol": {"seed": 42},
              "method": "positive", "layer": {"path": "model.23.input[0]", "level": 0},
              "preprocessing": {"input_size": [64,64], "scale": 1/255},
              "target": {"model_sha256": "a"*64, "scalar": "confidence", "class_index": 0, "class_name": "scratch",
                         "execution": {"runtime_id": "modern-8.4.142", "head_path": "one_to_one"}}, "rows": []}
    for fill in report["fill_methods"]:
        for role, value, mask in (("attribution", .8, "x"), ("random_control", .2, "y")):
            report["rows"].append({"fraction": .1, "fill_method": fill, "control_strategy": "translated_mask",
                                   "role": role, "fixed_score_drop": value, "mask_sha256": mask})
    changed_seed = copy.deepcopy(report)
    changed_seed["protocol"]["seed"] = 43
    changed_head = copy.deepcopy(report)
    changed_head["target"]["execution"]["head_path"] = "one_to_many"
    result = summarize_faithfulness([report, changed_seed, changed_head])
    assert len(result["fractions"]) == 6
    assert all(group["support"] == 1 and group["identity"]["execution"] for group in result["fractions"].values())
    changed_scalar = copy.deepcopy(report)
    changed_scalar["target"]["scalar"] = "class"
    changed_scale = copy.deepcopy(report)
    changed_scale["preprocessing"]["scale"] = 1.
    changed_layer = copy.deepcopy(report)
    changed_layer["layer"]["level"] = 1
    assert len(summarize_faithfulness([report, changed_scalar, changed_scale, changed_layer])["fractions"]) == 8


def test_real_one2one_tied_export_is_not_silently_paired(modern, tmp_path):
    import torch
    from fp_external_runtime import Runtime
    from fp_attribution import _match_target
    runtime, _, _, _ = modern
    if runtime.head_path != "one_to_one":
        pytest.skip("one-to-one-specific tied export negative")
    tied = copy.deepcopy(runtime.model)
    for module in tied.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_var.fill_(1.)
    with torch.no_grad():
        for branch in tied.model[-1].one2one_cv3:
            branch[-1].weight.div_(1000.)
        for branch in tied.model[-1].one2one_cv2:
            branch[-1].bias.fill_(2.)
    pt = tmp_path / "tied.pt"
    torch.save({"model": tied}, pt)
    cfg = {**runtime.config, "model_path": str(pt)}
    actual_pt = YoloRuntime(cfg, trusted=True)
    class Public(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)[0]
    public = Public(actual_pt.model).eval()
    x = torch.from_numpy(actual_pt._prepare(picture())[0])
    with torch.no_grad():
        public(x)
    onnx = tmp_path / "tied.onnx"
    torch.onnx.export(public, x, str(onnx), opset_version=17, input_names=["images"], output_names=["predictions"], dynamo=False)
    deployed = Runtime({**runtime.core_config, "model_path": str(onnx)}).infer(picture())
    gradient = actual_pt.infer(picture())
    # Exercise the actual companion-model guard, not only tensor allclose.
    with pytest.raises(ValueError):
        _match_target(deployed, gradient, deployed["prediction"]["detections"][0])


def test_companion_guard_requires_complete_unique_bijection():
    from fp_attribution import _match_target
    a = {"contract_id": "onnx", "input_sha256": "a"*64, "prediction": {"task": "detection", "detections": [
        {"output_index": 0, "class": "scratch", "score": .8, "box": [.1,.1,.2,.2]},
        {"output_index": 1, "class": "scratch", "score": .8, "box": [.6,.6,.8,.8]}]}}
    b = copy.deepcopy(a)
    b["contract_id"] = "pt"
    b["prediction"]["detections"].reverse()
    for i, row in enumerate(b["prediction"]["detections"]):
        row.update(output_index=i, native_candidate_index=100+i, class_index=0, head_path="one_to_one")
    selected, match = _match_target(a, b, a["prediction"]["detections"][0])
    assert match["status"] == "per_case_matched"
    assert selected["output_index"] == 1 and selected["native_candidate_index"] == 101
    duplicate = copy.deepcopy(b)
    duplicate["prediction"]["detections"][0]["box"] = [.1,.1,.2,.2]
    with pytest.raises(ValueError):
        _match_target(a, duplicate, a["prediction"]["detections"][0])
    missing = copy.deepcopy(b)
    missing["prediction"]["detections"].pop()
    with pytest.raises(ValueError):
        _match_target(a, missing, a["prediction"]["detections"][0])
