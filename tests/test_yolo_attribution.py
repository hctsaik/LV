"""Real pinned architectures plus small deterministic geometry/contract gates.

Random initialized architecture fixtures validate adapters, not production accuracy.
No model weights or architectures are downloaded at test time.
"""
import copy
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from detector_attribution_contracts import DetectionTargetV2, DetectorRunSpecV2, is_current
from yolo_attribution import (AttributionError, YoloRuntime, inspect_model,
                              localization_metrics, source_map, _v5_imports)


def picture():
    image = Image.new("RGB", (96, 48), (110, 60, 20))
    pixels = np.asarray(image).copy()
    pixels[8:30, 20:55] = [220, 150, 80]
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture(scope="module", params=["yolov5", "yolov8", "yolo11"])
def real_model(request, tmp_path_factory):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(34)
    root = tmp_path_factory.mktemp(request.param)
    if request.param == "yolov5":
        module = _v5_imports()
        cfg = Path(module.__file__).parent/"yolov5n.yaml"
        model = module.DetectionModel(str(cfg), ch=3, nc=2)
    else:
        from ultralytics.nn.tasks import DetectionModel
        model = DetectionModel("yolov8n.yaml" if request.param == "yolov8" else "yolo11n.yaml", ch=3, nc=2, verbose=False)
    model.names = {0: "scratch", 1: "dent"}
    path = root/"original.pt"
    torch.save({"model": model.eval()}, path)
    info = inspect_model(path, trusted=True)
    config = info["default_config"]
    config.update(input_size=[64,64], threshold=0.0, max_detections=10)
    return request.param, path, config, root


def test_real_architecture_inference_gradcam_and_cleanup(real_model):
    family, path, config, root = real_model
    runtime = YoloRuntime(config, trusted=True)
    before = len(runtime.model.model[-1]._forward_pre_hooks)
    run = runtime.infer(picture(), root)
    assert run["prediction"]["detections"]
    det = run["prediction"]["detections"][0]
    result = runtime.gradcam(picture(), output_index=det["output_index"], method="signed", artifact_root=root)
    assert result["family"] == family
    assert result["target"]["output_index"] == det["output_index"]
    assert result["forward_parity"]["status"] == "matched"
    assert result["gradient_abs_max"] > 0
    assert len(runtime.model.model[-1]._forward_pre_hooks) == before
    with np.load(Path(result["artifact_dir"])/"maps.npz") as maps:
        assert maps["source"].shape == (48,96)
        assert np.isfinite(maps["raw"]).all()
    assert Path(result["artifact_dir"], "overlay.png").is_file()
    assert json.loads(Path(result["artifact_dir"], "attribution.json").read_text(encoding="utf-8"))["compute_id"] == result["compute_id"]
    again = runtime.infer(picture())
    assert again["output_sha256"] == run["output_sha256"]
    with pytest.raises(AttributionError, match="最終偵測框"):
        runtime.gradcam(picture(), output_index=999999)
    assert len(runtime.model.model[-1]._forward_pre_hooks) == before


def test_real_architecture_onnx_export_correspondence(real_model):
    import torch
    import onnxruntime as ort
    family, path, config, root = real_model
    runtime = YoloRuntime(config, trusted=True)
    class Raw(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)[0]
    tensor, _, _ = runtime._prepare(picture())
    x = torch.from_numpy(tensor)
    model = Raw(runtime.model).eval()
    # Warm caches before tracing; export tests the actual adapter architecture.
    with torch.no_grad():
        expected = model(x).numpy()
    target = root/"export.onnx"
    torch.onnx.export(model, x, str(target), input_names=["images"], output_names=["predictions"], opset_version=17, dynamo=False)
    actual = ort.InferenceSession(target.read_bytes(), providers=["CPUExecutionProvider"]).run(None, {"images": tensor})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_v5_confidence_is_objectness_times_class(real_model):
    import torch
    family, _, config, _ = real_model
    runtime = YoloRuntime(config, trusted=True)
    if family == "yolov5":
        raw = torch.tensor([[[1.,2.,3.,4.,.4,.3,.7]]])
        assert float(runtime._scalar(raw,0,1,"confidence")) == pytest.approx(.28)
        assert float(runtime._scalar(raw,0,1,"class")) == pytest.approx(.7)
        assert float(runtime._scalar(raw,0,1,"objectness")) == pytest.approx(.4)
    else:
        with pytest.raises(AttributionError, match="objectness"):
            runtime._scalar(torch.zeros(1,6,1),0,0,"objectness")


def test_untrusted_checkpoint_never_loaded(tmp_path):
    path = tmp_path/"model.pt"
    path.write_bytes(b"not-a-checkpoint")
    with pytest.raises(AttributionError, match="來源確認"):
        inspect_model(path)


def test_zero_gradient_does_not_create_hotspot(real_model):
    import torch
    family, _, config, _ = real_model
    if family != "yolov8":
        pytest.skip("zero-gradient numeric gate uses v8; all families have real CAM smoke")
    runtime = YoloRuntime(config,trusted=True)
    with torch.no_grad():
        for p in runtime.model.model[-1].cv3.parameters():
            p.zero_()
    det = runtime.infer(picture())["prediction"]["detections"][0]
    result = runtime.gradcam(picture(),output_index=det["output_index"])
    assert result["status"] == "zero_gradient"
    assert not np.asarray(result["maps"]["source"]).any()
    assert not runtime.model.model[-1]._forward_pre_hooks


def test_original_v5_namespace_checkpoint(real_model):
    import torch
    family, _, config, root = real_model
    if family != "yolov5":
        pytest.skip("original namespace applies only to original v5")
    runtime = YoloRuntime(config,trusted=True)
    types = {type(m) for m in runtime.model.modules() if type(m).__module__.startswith("yolov5.models.")}
    old = {t:t.__module__ for t in types}
    path = root/"original_training_namespace.pt"
    try:
        for t in types:
            t.__module__ = t.__module__.replace("yolov5.","",1)
        torch.save({"model":runtime.model},path)
    finally:
        for t,module in old.items():
            t.__module__ = module
    assert inspect_model(path,trusted=True)["family"] == "yolov5"


def test_unsupported_segment_and_invalid_settings(real_model):
    import torch
    family, _, config, root = real_model
    if family != "yolov8":
        pytest.skip("unsupported end2end gate uses v8")
    runtime = YoloRuntime(config,trusted=True)
    runtime.model.model[-1].end2end = True
    path = root/"unsupported_end2end.pt"
    torch.save({"model":runtime.model},path)
    with pytest.raises(AttributionError,match="head"):
        inspect_model(path,trusted=True)
    with pytest.raises(AttributionError,match="未知模型設定"):
        YoloRuntime({**config,"threshold_typo":.1},trusted=True)


def test_contract_changes_target_and_preserves_render_identity():
    target = DetectionTargetV2("a"*64,"b"*64,1,0,"scratch",(.1,.2,.8,.9))
    spec = DetectorRunSpecV2(target,{"input_size":[64,64]}, {"level":0})
    assert spec.key == "a2c35c10f4be385a0ef20a2ef00f1cdf3ebd8f05fef251da6fd810879e271004"
    result = {"compute_id": spec.key}
    assert is_current(result,spec)
    other = DetectorRunSpecV2(target,{"input_size":[64,64]}, {"level":1})
    assert not is_current(result,other)
    assert DetectorRunSpecV2(target,spec.geometry,spec.layer,"signed").key != spec.key
    geometry={"input_size":[64,64]}
    copied=DetectorRunSpecV2(target,geometry,{"level":0})
    key=copied.key
    geometry["input_size"][0]=128
    assert copied.key == key


def test_padding_removed_and_mask_must_align():
    geometry = {"source_size":[20,10], "input_size":[20,20], "resized_size":[20,10], "padding_xy":[0,5]}
    native = np.zeros((20,20), np.float32)
    native[:5] = 100
    native[15:] = 100
    assert not source_map(native,geometry).any()
    scores = np.zeros((10,20)); scores[2:4,3:5] = 1
    mask = scores > 0
    measured = localization_metrics(scores,mask)
    assert measured["pointing_hit"] and measured["positive_mass_in_mask"] == 1
    with pytest.raises(AttributionError, match="同尺寸"):
        localization_metrics(scores,np.zeros((5,10)))
    assert localization_metrics(scores,None)["status"] == "unavailable"


def test_fixed_scalar_faithfulness_and_cancel(real_model):
    family, _, config, root = real_model
    if family != "yolov8":
        pytest.skip("fixed-target/randomization numeric gate uses v8")
    runtime = YoloRuntime(config,trusted=True)
    det = runtime.infer(picture())["prediction"]["detections"][0]
    result = runtime.gradcam(picture(),output_index=det["output_index"],method="signed")
    scores = np.abs(np.asarray(result["maps"]["source"]))
    report = runtime.faithfulness(picture(), result["target"], scores, fractions=(.1,.3), controls=2)
    assert report["protocol_version"] == 2
    assert {r["fill_method"] for r in report["rows"]} == {"constant", "blur"}
    assert all(r.get("reason") if r.get("unavailable") else
               "fixed_scalar" in r and "rematched_detection" in r for r in report["rows"])
    assert report["support_images"] == 1
    with pytest.raises(AttributionError, match="取消"):
        runtime.gradcam(picture(),output_index=det["output_index"],cancelled=lambda: True)
    assert not runtime.model.model[-1]._forward_pre_hooks
    before = runtime.infer(picture())["output_sha256"]
    sanity = runtime.randomization_sanity(picture(),result["target"],seed=11)
    assert sanity["reset_modules"]
    assert sanity["status"] in {"measured","inconclusive"}
    assert runtime.infer(picture())["output_sha256"] == before
