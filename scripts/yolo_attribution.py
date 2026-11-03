"""Pinned, local YOLO Detect inference and exact-candidate Grad-CAM.

Only call trusted=True from the explicit, isolated model-inspection/job worker.
Checkpoint pickle is executable: this module does not claim to sandbox torch.load.
No torch.hub, automatic model download, model fusion or DINO fallback is used.
"""
from __future__ import annotations

import copy
import importlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import sys
import threading
import uuid

import numpy as np
from PIL import Image

from detector_attribution_contracts import DetectionTargetV2, DetectorRunSpecV2, DetectionTargetV3, DetectorRunSpecV3, identity
from fp_external_runtime import DEFAULT_CONFIG, canonical, config_checked, decode_output, prepare_input, iou
from fp_cases import digest, utc_now

PINS = {"ultralytics": "8.3.203", "yolov5": "7.0.14"}
RUNTIMES = {"legacy-8.3.203": "8.3.203", "modern-8.4.142": "8.4.142"}
EXTRA_CONFIG = {"family", "backend", "runtime_id", "runtime_release", "head_path", "nms", "output_contract"}
ADAPTER_VERSION = "lv-yolo-detect-2.0"
MODERN_ADAPTER_VERSION = "lv-yolo-detect-modern-3.0"
_LOAD_LOCK = threading.RLock()


class AttributionError(ValueError):
    pass


def _pin(name, runtime_id="legacy-8.3.203"):
    if runtime_id not in RUNTIMES:
        raise AttributionError("Unknown registered YOLO runtime")
    try:
        actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise AttributionError("尚未安裝 YOLO 歸因套件；請安裝 requirements-yolo-attribution.txt") from exc
    expected = RUNTIMES[runtime_id] if name == "ultralytics" else PINS[name]
    if actual != expected:
        raise AttributionError(f"{name} 版本 {actual} 未驗證；本 adapter 需要 {expected}")
    return actual


def _v5_imports():
    _pin("yolov5")
    module = importlib.import_module("yolov5.models.yolo")
    # Original training checkpoints reference models.yolo/models.common, while
    # the pinned wheel namespaces the same original anchor-based architecture.
    for name in ("models", "models.common", "models.experimental", "models.yolo"):
        target = importlib.import_module("yolov5." + name)
        existing = sys.modules.get(name)
        if existing is not None and existing is not target:
            raise AttributionError("YOLOv5 原始模組名稱衝突；請在獨立模型 worker 重試")
        sys.modules[name] = target
    return module


def _load(path, trusted, config=None):
    if not trusted:
        raise AttributionError("PyTorch 模型需要先經來源確認與獨立檢查；不會自動執行未確認 checkpoint")
    import torch
    if torch.__version__.split("+")[0] != "2.6.0":
        raise AttributionError("此 adapter 驗證環境為 PyTorch 2.6.0；請使用專案固定相依版本")
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".pt" or not path.is_file():
        raise AttributionError("請補入存在的受支援 YOLO .pt 原始模型")
    content = path.read_bytes()
    config = config or {}
    runtime_id = config.get("runtime_id", "legacy-8.3.203")
    release = _pin("ultralytics", runtime_id)
    if config.get("runtime_release", release) != release:
        raise AttributionError("Requested runtime release differs from the executing environment")
    modern = runtime_id == "modern-8.4.142"
    # The worker must enforce resource limits. Inspect globals for useful early
    # unsupported-family feedback, not as a security guarantee for pickle.
    with _LOAD_LOCK:
        try:
            globals_ = torch.serialization.get_unsafe_globals_in_checkpoint(io.BytesIO(content))
            if any(x.startswith(("models.", "yolov5.")) for x in globals_):
                _v5_imports()
            checkpoint = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
        except Exception as exc:
            raise AttributionError(f"模型載入失敗；需受支援原始架構及完整 checkpoint：{exc}") from exc
    if isinstance(checkpoint, dict):
        model = checkpoint.get("ema")
        if model is None:
            model = checkpoint.get("model")
    else:
        model = checkpoint
    if not isinstance(model, torch.nn.Module) or not hasattr(model, "model"):
        raise AttributionError("需要含原始 YOLO Detect 架構的 checkpoint，不能只提供 state_dict")
    head = model.model[-1]
    module_name, class_name = type(head).__module__, type(head).__name__
    if class_name != "Detect" or (not modern and getattr(head, "end2end", False)):
        raise AttributionError(f"尚未支援此任務／head：{module_name}.{class_name}；seg、OBB、pose、end-to-end 不會代用偵測 CAM")
    if getattr(head, "xyxy", False) or (getattr(head, "export", False) and getattr(head, "format", None) not in {None, "onnx"}):
        raise AttributionError("此 checkpoint 帶有特殊匯出座標／正規化模式；請提供原始訓練 .pt")
    if module_name in {"yolov5.models.yolo", "models.yolo"} and hasattr(head, "anchors") and hasattr(head, "na"):
        family = "yolov5"
        _pin("yolov5")
    elif module_name == "ultralytics.nn.modules.head":
        # Verify real body topology rather than the user-provided filename.
        blocks = {type(m).__name__ for m in model.modules()}
        if modern and "A2C2f" in blocks and "C3k2" in blocks:
            family = "yolo12"
        elif modern and "C3k2" in blocks and "C2PSA" in blocks and int(head.reg_max) == 1:
            family = "yolo26"
        elif "C3k2" in blocks and "C2PSA" in blocks:
            family = "yolo11"
        elif "C2f" in blocks and not (blocks & {"C3k2", "A2C2f", "PSA"}):
            family = "yolov8"
        else:
            raise AttributionError("此 Detect backbone 尚未驗證；請提供版本與原始架構")
    else:
        raise AttributionError(f"未驗證的 Detect 架構：{module_name}")
    if modern and family == "yolov5":
        raise AttributionError("Original YOLOv5 requires the isolated legacy runtime")
    if modern:
        requested_head = config.get("head_path")
        if requested_head not in {None, "one_to_many", "one_to_one"}:
            raise AttributionError("Unknown detector head path")
        nms = config.get("nms")
        if nms is not None and type(nms) is not bool:
            raise AttributionError("nms must be null or boolean")
        if nms is not None:
            implied = "one_to_many" if nms else "one_to_one"
            if requested_head is not None and implied != requested_head:
                raise AttributionError("Head selection conflicts with NMS intent")
            requested_head = implied
        if requested_head is not None:
            branches = ("one2one_cv2", "one2one_cv3") if requested_head == "one_to_one" else ("cv2", "cv3")
            if any(getattr(head, name, None) is None for name in branches):
                raise AttributionError("Requested detection branch is absent or was removed by fusion")
            head.end2end = requested_head == "one_to_one"
        if getattr(head, "export", False):
            raise AttributionError("Export-mode PT lacks the native trace contract; use the original training checkpoint")
    elif config.get("head_path") not in {None, "one_to_many"} or config.get("nms") is False:
        raise AttributionError("Legacy adapter supports only one-to-many plus NMS")
    allowed = ("torch.nn.", "ultralytics.nn.") if family != "yolov5" else ("torch.nn.", "yolov5.models.", "models.")
    if any(not type(m).__module__.startswith(allowed) for m in model.modules()):
        raise AttributionError("模型含未驗證自訂模組；不能開啟此 adapter")
    model = model.cpu().float().eval()
    for p in model.parameters():
        p.requires_grad_(True)
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        if set(names) != set(range(int(head.nc))):
            raise AttributionError("模型類別 ID 不是連續的 0..C-1")
        names = [str(names[i]) for i in range(int(head.nc))]
    if not isinstance(names, (list, tuple)) or len(names) != int(head.nc):
        names = [str(i) for i in range(int(head.nc))]
    return model, family, list(names), digest(content)


def inspect_model(path, *, trusted=False, config=None):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".onnx":
        import onnxruntime as ort
        session = ort.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        defaults = copy.deepcopy(DEFAULT_CONFIG)
        defaults["model_path"] = str(path)
        if len(inputs) == 1 and len(inputs[0].shape) == 4:
            shape = inputs[0].shape
            if all(isinstance(v, int) for v in shape[2:]):
                defaults["input_size"] = [shape[3], shape[2]]
        return {"family": "onnx", "task": "unknown", "model_sha256": digest(path.read_bytes()),
                "capabilities": {"inference": len(inputs) == 1, "gradcam": False, "perturbation": len(inputs) == 1},
                "reason": "ONNX 本身無可用反向圖；可補入對應 PyTorch 模型", "default_config": defaults,
                "inputs": [{"name": x.name, "shape": x.shape, "type": x.type} for x in inputs],
                "outputs": [{"name": x.name, "shape": x.shape} for x in outputs],
                "settings_provenance": {"output_format": "needs_confirmation", "class_names": "needs_confirmation", "threshold": "lv_default"}}
    config = config or {}
    runtime_id = config.get("runtime_id", "legacy-8.3.203")
    model, family, names, sha = _load(path, trusted, config)
    head_path = "one_to_one" if getattr(model.model[-1], "end2end", False) else "one_to_many"
    defaults = copy.deepcopy(DEFAULT_CONFIG)
    defaults.update(model_path=str(path), family=family, backend="pytorch_yolo", class_names=names,
                    output_format="xyxy" if head_path == "one_to_one" else "yolo_v5" if family == "yolov5" else "yolo_v8")
    if runtime_id != "legacy-8.3.203":
        defaults.update(runtime_id=runtime_id, runtime_release=RUNTIMES[runtime_id], head_path=head_path, nms=config.get("nms"))
    return {"family": family, "task": "detect", "model_sha256": sha, "class_names": names,
            "capabilities": {"inference": True, "gradcam": True, "perturbation": True},
            "default_config": defaults, "package_versions": {**{n: _pin(n, runtime_id) for n in PINS if n == "ultralytics" or family == "yolov5"}, "torch": importlib.metadata.version("torch")},
            "runtime_id": runtime_id, "runtime_release": RUNTIMES[runtime_id], "head_path": head_path,
            "available_head_paths": (["one_to_many"] if family == "yolov5" else
                [name for name, branches in (("one_to_many", ("cv2", "cv3")), ("one_to_one", ("one2one_cv2", "one2one_cv3")))
                 if all(getattr(model.model[-1], attr, None) is not None for attr in branches)]),
            "nms": head_path == "one_to_many", "output_contract": _output_contract(model, family),
            "gradient_capabilities": {"class_score": True, "objectness": family == "yolov5", "box_localization": False,
                                      "feature_source": "Detect input features", "native_candidate_trace": True},
            "adapter_version": MODERN_ADAPTER_VERSION if runtime_id == "modern-8.4.142" else ADAPTER_VERSION,
            "head": type(model.model[-1]).__module__ + ".Detect",
            "settings_provenance": {"class_names": "model", "input_size": "lv_default", "threshold": "lv_default", "resize": "lv_default"}}


def _tensor_hash(array):
    return digest(str(array.dtype).encode()+canonical(list(array.shape)).encode()+array.tobytes())


def _output_contract(model, family):
    head = model.model[-1]
    return {"format": "xyxy_score_class" if getattr(head, "end2end", False) else "xywh_objectness_classes" if family == "yolov5" else "xywh_classes",
            "head_path": "one_to_one" if getattr(head, "end2end", False) else "one_to_many",
            "export": bool(getattr(head, "export", False)), "export_format": getattr(head, "format", None),
            "max_det": int(getattr(head, "max_det", 300)), "agnostic_topk": bool(getattr(head, "agnostic_nms", False))}


def _raw(output):
    value = output[0] if isinstance(output, (list, tuple)) else output
    if not hasattr(value, "shape") or value.ndim != 3:
        raise AttributionError("YOLO 原始輸出拓撲不符合已驗證 adapter")
    return value


def _bundle_record(value, directory):
    """On-disk JSON stays valid after job pending->artifacts rename or ZIP copy."""
    if isinstance(value,dict):
        return {k:_bundle_record(v,directory) for k,v in value.items()}
    if isinstance(value,list):
        return [_bundle_record(v,directory) for v in value]
    if isinstance(value,str) and value.startswith(str(directory.resolve())):
        path=Path(value)
        if path.is_relative_to(directory.resolve()):
            return path.relative_to(directory.resolve()).as_posix()
    return value


def _save_run(run, data, tensor, raw, preview, root, native=None):
    directory = Path(root) / run["id"]
    directory.mkdir(parents=True, exist_ok=False)
    preview.save(directory / "model_input.png")
    with Image.open(io.BytesIO(data)) as image:
        image.convert("RGB").save(directory / "source.png")
    np.savez_compressed(directory / "tensors.npz", input=tensor, output=raw,
                        **({"native_output": native} if native is not None else {}))
    run["artifact_dir"] = str(directory.resolve())
    (directory / "run.json").write_text(json.dumps(_bundle_record(run,directory), ensure_ascii=False, indent=2), encoding="utf-8")


def source_map(native, geometry):
    """Map feature map through actual input dimensions, then remove padding."""
    tw, th = geometry["input_size"]
    w, h = geometry["source_size"]
    nw, nh = geometry["resized_size"]
    left, top = geometry["padding_xy"]
    enlarged = Image.fromarray(np.asarray(native, dtype=np.float32)).resize((tw, th), Image.Resampling.BILINEAR)
    return np.asarray(enlarged.crop((left, top, left+nw, top+nh)).resize((w, h), Image.Resampling.BILINEAR)).copy()


def render_overlay(image, scores, *, method="positive", opacity=.45):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    values = np.asarray(scores, dtype=np.float32)
    maximum = float(np.max(np.abs(values))) if values.size else 0
    normalized = values/maximum if maximum > 0 else np.zeros_like(values)
    heat = np.zeros_like(rgb)
    heat[..., 0] = 255
    if method == "signed":
        heat[normalized < 0] = [0, 90, 255]
    alpha = np.abs(normalized)[..., None]*float(opacity)
    return Image.fromarray(np.clip(rgb*(1-alpha)+heat*alpha, 0, 255).astype(np.uint8))


def localization_metrics(scores, mask):
    if mask is None:
        return {"status": "unavailable", "reason": "未提供對齊的人工像素 mask"}
    values = np.asarray(scores, dtype=float)
    target = np.asarray(mask)
    if target.ndim != 2 or target.shape != values.shape:
        raise AttributionError("人工 mask 必須與原圖同尺寸；不會自動縮放或把框當 mask")
    if not np.isfinite(target).all() or not np.isfinite(values).all():
        raise AttributionError("mask 或 attribution 含非有限數值")
    target = target > 0
    positive = np.maximum(values, 0)
    mass = float(positive.sum())
    if not target.any() or mass <= 0:
        return {"status": "unavailable", "reason": "mask 無前景或沒有正向 attribution"}
    k = max(1, int(math.ceil(target.size*.1)))
    top = np.argsort(-positive.ravel(), kind="stable")[:k]
    return {"status": "measured", "pointing_hit": bool(target.ravel()[int(positive.argmax())]),
            "positive_mass_in_mask": float(positive[target].sum()/mass),
            "top10_precision": float(target.ravel()[top].mean()), "mask_pixels": int(target.sum()),
            "source_pixels": target.size, "limitation": "定位一致性不是忠實度、分割品質或根因證明"}


class YoloRuntime:
    def __init__(self, config, *, trusted=False):
        unknown = set(config)-set(DEFAULT_CONFIG)-EXTRA_CONFIG
        if unknown:
            raise AttributionError("未知模型設定欄位："+", ".join(sorted(unknown)))
        self.runtime_id = config.get("runtime_id", "legacy-8.3.203")
        self.modern = self.runtime_id == "modern-8.4.142"
        self.model, self.family, names, self.sha = _load(config["model_path"], trusted, config)
        self.head_path = "one_to_one" if getattr(self.model.model[-1], "end2end", False) else "one_to_many"
        self.execution = {"runtime_id": self.runtime_id, "runtime_release": RUNTIMES[self.runtime_id],
                          "head_path": self.head_path, "output_contract": _output_contract(self.model, self.family)}
        if config.get("output_contract") is not None and config["output_contract"] != self.execution["output_contract"]:
            raise AttributionError("Requested output/export contract differs from the actual model")
        requested = config.get("family")
        if requested and requested != self.family:
            raise AttributionError(f"指定 {requested} 與實際 {self.family} 架構不同")
        self.config = config_checked({key: value for key, value in config.items() if key in DEFAULT_CONFIG})
        if self.config["class_names"] != names:
            raise AttributionError("設定類別順序與 checkpoint 不同；請核對 class mapping")
        expected = "xyxy" if self.head_path == "one_to_one" else "yolo_v5" if self.family == "yolov5" else "yolo_v8"
        if self.config["output_format"] != expected or self.config["input_layout"] != "NCHW" or self.config["coordinates"] != "pixels" or self.config["score_activation"] != "none":
            raise AttributionError("此 PyTorch adapter 需要對應 YOLO 輸出格式、NCHW、pixel 座標與已啟用機率")
        stride = int(max(self.model.model[-1].stride))
        if any(n % stride for n in self.config["input_size"]):
            raise AttributionError(f"YOLO 輸入寬高須為 stride {stride} 的倍數")
        contract = {k: v for k, v in self.config.items() if k != "model_path"}
        contract.update(model_sha256=self.sha, schema_version=2, input_dtype="tensor(float)", family=self.family,
                        adapter_version=ADAPTER_VERSION, torch_version=importlib.metadata.version("torch"))
        if self.modern:
            contract.update(schema_version=3, adapter_version=MODERN_ADAPTER_VERSION, **copy.deepcopy(self.execution))
        self.identity = {"contract": contract, "contract_id": identity(contract),
                         "runtime": {"name": "pytorch_yolo", "version": _pin("ultralytics", self.runtime_id), "provider": "CPU", "family": self.family}}
        self.core_config = copy.deepcopy(self.config)
        if self.modern:
            self.config.update(**copy.deepcopy(self.execution), nms=config.get("nms"), family=self.family, backend="pytorch_yolo")
        self.lock = threading.RLock()

    def _prepare(self, data):
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return prepare_input(image, self.core_config)

    def _forward(self, tensor):
        """Return untouched public output, differentiable native scores and final-row trace."""
        import torch
        output = self.model(tensor)
        public = _raw(output)
        if not self.modern or self.head_path != "one_to_one":
            return public, public, None
        if not isinstance(output, tuple) or len(output) != 2 or not isinstance(output[1], dict):
            raise AttributionError("Modern head did not expose its native prediction trace")
        head = self.model.model[-1]
        branch = output[1].get("one2one")
        if not isinstance(branch, dict) or not branch.get("feats"):
            raise AttributionError("Selected one-to-one branch lacks native features")
        native = head._inference(branch)
        scores, labels, indices = head.get_topk_index(native[:, 4:, :].permute(0, 2, 1), head.max_det)
        boxes = head._gather(native[:, :4, :].permute(0, 2, 1), indices)
        reconstructed = torch.cat((boxes, scores, labels), dim=-1)
        if reconstructed.shape != public.shape or not torch.allclose(reconstructed.detach(), public.detach(), atol=1e-6, rtol=1e-5):
            raise AttributionError("Native top-k trace differs from the unchanged normal forward output")
        trace = [{"native_candidate_index": int(i), "class_index": int(c)}
                 for i, c in zip(indices[0].detach().tolist(), labels[0, :, 0].detach().tolist())]
        return public, native, trace

    def _decode(self, public, geometry, trace):
        prediction = decode_output(public, self.core_config, geometry)
        if self.modern:
            for key in ("detections", "candidates"):
                for row in prediction[key]:
                    detail = trace[row["output_index"]] if trace is not None else {
                        "native_candidate_index": row["output_index"], "class_index": self.config["class_names"].index(row["class"])}
                    row.update(detail, head_path=self.head_path)
            prediction["native_candidates_available"] = True
            prediction["target_trace"] = "exact_forward_topk" if trace is not None else "native_output_index"
        return prediction

    def infer(self, data, artifact_root=None):
        import torch
        tensor, preview, geometry = self._prepare(data)
        with self.lock, torch.no_grad():
            public, native, trace = self._forward(torch.from_numpy(tensor))
            raw = public.detach().cpu().numpy()
        prediction = self._decode(raw, geometry, trace)
        run = {"id": "run_"+uuid.uuid4().hex[:16], "at": utc_now(), **copy.deepcopy(self.identity),
               "image_sha256": digest(data), "geometry": geometry, "prediction": prediction,
               "input_sha256": _tensor_hash(tensor), "output_sha256": _tensor_hash(raw),
               "input_shape": list(tensor.shape), "output_shape": list(raw.shape), "origin": "lv_local_inference"}
        native_np = native.detach().cpu().numpy() if self.modern else None
        if self.modern:
            run.update(native_output_sha256=_tensor_hash(native_np), native_output_shape=list(native_np.shape),
                       execution=copy.deepcopy(self.execution))
        if artifact_root is not None:
            _save_run(run, data, tensor, raw, preview, artifact_root, native_np)
        return run

    def _scalar(self, raw, index, cls, mode):
        count = raw.shape[1] if self.family == "yolov5" else raw.shape[2]
        if type(index) is not int or not 0 <= index < count or type(cls) is not int or not 0 <= cls < len(self.config["class_names"]):
            raise AttributionError("原始 candidate 或 class index 超出本次輸出")
        if mode not in {"confidence", "class", "objectness"}:
            raise AttributionError("不支援此歸因 scalar")
        if self.family == "yolov5":
            row = raw[0, index]
            return row[4] if mode == "objectness" else row[5+cls] if mode == "class" else row[4]*row[5+cls]
        if mode == "objectness":
            raise AttributionError("此 Detect head 不提供獨立 objectness 分數")
        return raw[0, 4+cls, index]

    def gradcam(self, data, *, output_index, class_index=None, scalar="confidence", method="positive",
                artifact_root=None, fractions=(), controls=3, seed=42, mask=None, cancelled=None,
                profile_revision="", case_context=None, layer_level=None, _target_snapshot=None,
                protocol_version=2, fill_methods=("constant", "blur"), max_overlap=.1, control_strategy="translated_mask"):
        import torch
        if method not in {"positive", "signed"}:
            raise AttributionError("CAM method 必須是 positive 或 signed")
        if protocol_version != 2 or control_strategy != "translated_mask":
            raise AttributionError("Unsupported attribution validation protocol")
        tensor, preview, geometry = self._prepare(data)
        def check_cancel():
            if cancelled and cancelled():
                raise AttributionError("工作已取消")
        check_cancel()
        head = self.model.model[-1]
        captures = []
        def capture(module, args):
            captures.extend(list(args[0]))
        with self.lock:
            handle = head.register_forward_pre_hook(capture)
            try:
                with torch.no_grad():
                    ordinary, ordinary_native, ordinary_trace = self._forward(torch.from_numpy(tensor))
                captures.clear()
                with torch.enable_grad():
                    public, raw, trace = self._forward(torch.from_numpy(tensor).requires_grad_(True))
                    if (not torch.allclose(public.detach(), ordinary, atol=1e-6, rtol=1e-5)
                            or not torch.allclose(raw.detach(), ordinary_native, atol=1e-6, rtol=1e-5) or trace != ordinary_trace):
                        raise AttributionError("保留梯度的 forward 與正常推論不同，拒絕歸因")
                    raw_np = public.detach().cpu().numpy()
                    prediction = self._decode(raw_np, geometry, trace)
                    rows = prediction["detections"]
                    chosen = next((r for r in rows if r["output_index"] == output_index), None)
                    if _target_snapshot is not None:
                        if (_target_snapshot["input_sha256"] != _tensor_hash(tensor) or
                                _target_snapshot["model_sha256"] != self.sha or
                                _target_snapshot["output_index"] != output_index):
                            raise AttributionError("固定目標 snapshot 與輸入不符")
                        if self.modern and _target_snapshot.get("execution") != self.execution:
                            raise AttributionError("Fixed target belongs to a different runtime/head/export path")
                        chosen = {"class": _target_snapshot["class_name"], "box": _target_snapshot["box"], "output_index": output_index,
                                  "native_candidate_index": _target_snapshot.get("native_candidate_index", output_index)}
                    if chosen is None:
                        raise AttributionError("指定 candidate 不是本次最終偵測框；請重新選擇實際框")
                    cls = self.config["class_names"].index(chosen["class"]) if class_index is None else class_index
                    if self.modern and cls != self.config["class_names"].index(chosen["class"]):
                        raise AttributionError("Requested class differs from the selected final detection")
                    native_index = chosen.get("native_candidate_index", output_index)
                    counts = [int(x.shape[-2]*x.shape[-1])*(int(head.na) if self.family == "yolov5" else 1) for x in captures]
                    offset, native_level = 0, None
                    for level, count in enumerate(counts):
                        if offset <= native_index < offset+count:
                            native_level = level
                            break
                        offset += count
                    if native_level is None:
                        raise AttributionError("candidate 無法映射至真實 feature level")
                    level = native_level if layer_level is None else int(layer_level)
                    if not 0 <= level < len(captures):
                        raise AttributionError("feature level 超出模型分支")
                    feature = captures[level]
                    score = self._scalar(raw, native_index, cls, scalar)
                    gradient = torch.autograd.grad(score, feature, allow_unused=True)[0]
                    check_cancel()
                    if gradient is None:
                        raise AttributionError("此 scalar 對選定層沒有梯度；請選原始 candidate 所屬層")
                    if not torch.isfinite(gradient).all() or not torch.isfinite(feature).all():
                        raise AttributionError("activation 或 gradient 含 NaN/Inf")
                    raw_map = (gradient.mean(dim=(2,3), keepdim=True)*feature).sum(1)[0].detach().cpu().numpy()
                    gradient_max = float(gradient.abs().max())
            finally:
                handle.remove()
        selected_map = np.maximum(raw_map, 0) if method == "positive" else raw_map.copy()
        mapped = source_map(selected_map, geometry)
        status = "zero_gradient" if gradient_max == 0 else "no_positive_attribution" if method == "positive" and not np.any(selected_map > 0) else "zero_attribution" if not np.any(selected_map) else "ok"
        layer = {"path": f"model.{len(self.model.model)-1}.input[{level}]", "level": level,
                 "native_candidate_level": native_level, "shape": list(feature.shape),
                 "stride_xy": [tensor.shape[3]/feature.shape[3], tensor.shape[2]/feature.shape[2]],
                 "candidate_offset": offset, "candidate_level_counts": counts}
        if self.modern:
            target = DetectionTargetV3(self.sha, _tensor_hash(tensor), output_index, cls, self.config["class_names"][cls],
                                       tuple(chosen["box"]), native_index, self.execution, scalar)
            spec = DetectorRunSpecV3(target, geometry, layer, method, profile_revision, digest(data), case_context or {})
        else:
            target = DetectionTargetV2(self.sha, _tensor_hash(tensor), output_index, cls, self.config["class_names"][cls], tuple(chosen["box"]), scalar)
            spec = DetectorRunSpecV2(target, geometry, layer, method, profile_revision, digest(data), case_context or {})
        result = {"id": "cam_"+uuid.uuid4().hex[:16], "at": utc_now(), "schema_version": spec.schema_version, "kind": "detector_gradcam",
                  "compute_id": spec.key, "spec": spec.as_dict(), "status": status, "method": method,
                  "target": target.as_dict(), "layer": layer, "geometry": geometry, "score": float(score.detach()),
                  "gradient_abs_max": gradient_max, "family": self.family, **copy.deepcopy(self.identity),
                  "prediction": prediction, "forward_parity": {"status": "matched", "atol": 1e-6, "rtol": 1e-5},
                  "history_parity": {"status": "unavailable"},
                  "localization": localization_metrics(mapped, mask),
                  "legend": "紅色：對指定 scalar 的正向貢獻；藍色：負向貢獻（不是另一類別）" if method == "signed" else "紅色：對指定框分數的正向貢獻；強度為此圖相對值",
                  "limitation": "解釋此 PyTorch 模型的指定框分數；不是 ONNX 梯度、機台最終判定或已確認根因"}
        with Image.open(io.BytesIO(data)) as image:
            original = image.convert("RGB")
        if fractions and status == "ok" and np.any(mapped > 0):
            result["faithfulness"] = self.faithfulness(data, target.as_dict(), mapped, fractions=fractions, controls=controls, seed=seed, cancelled=cancelled,
                                                      artifact_root=Path(artifact_root)/result["id"]/"perturbations" if artifact_root else None,
                                                      fill_methods=fill_methods, max_overlap=max_overlap, layer=layer, method=method)
        else:
            result["faithfulness"] = {"status": "unavailable", "reason": "尚未執行或沒有有效歸因"}
        result["execution"] = copy.deepcopy(self.execution)
        result["analysis_id"] = identity({"cam_compute_id": result["compute_id"],
                                          "validation_id": result["faithfulness"].get("validation_id")})
        if artifact_root is not None:
            directory = Path(artifact_root)/result["id"]
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory/"maps.npz", raw=raw_map, selected=selected_map, source=mapped)
            np.savez_compressed(directory/"tensors.npz", input=tensor, output=raw_np,
                                **({"native_output": raw.detach().cpu().numpy()} if self.modern else {}))
            preview.save(directory/"model_input.png")
            original.save(directory/"source.png")
            render_overlay(original, mapped, method=method).save(directory/"overlay.png")
            result["artifact_dir"] = str(directory.resolve())
            (directory/"attribution.json").write_text(json.dumps(_bundle_record(result,directory), ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            result["maps"] = {"raw": raw_map.tolist(), "source": mapped.tolist()}
        return result

    def faithfulness(self, data, target, scores, *, fractions=(.1,.25,.5), controls=3, seed=42, cancelled=None, artifact_root=None,
                     fill_methods=("constant", "blur"), max_overlap=.1, layer=None, method=None):
        import torch
        from attribution_controls import iter_variants, assess_faithfulness, PROTOCOL_VERSION, CONTROL_STRATEGY
        tensor, _, geometry = self._prepare(data)
        if target["model_sha256"] != self.sha or target["input_sha256"] != _tensor_hash(tensor):
            raise AttributionError("Fixed target differs from model/input")
        if self.modern and target.get("execution") != self.execution:
            raise AttributionError("Fixed target runtime/head/export contract differs")
        with Image.open(io.BytesIO(data)) as image:
            pixels = np.asarray(image.convert("RGB")).copy()
        values = np.maximum(np.asarray(scores), 0)
        if values.shape != pixels.shape[:2] or not np.isfinite(values).all() or not values.any():
            raise AttributionError("Perturbation requires a finite positive source-space map")
        rows = []
        def observe(content):
            t, _, g = self._prepare(content)
            with self.lock, torch.no_grad():
                public, raw, trace = self._forward(torch.from_numpy(t))
                fixed = float(self._scalar(raw, target.get("native_candidate_index", target["output_index"]), target["class_index"], target["scalar"]))
                pred = self._decode(public.numpy(), g, trace)
            matches = [d for d in pred["detections"] if d["class"] == target["class_name"] and iou(d["box"], target["box"]) >= .1]
            rematch = max(matches, key=lambda d: iou(d["box"], target["box"])) if matches else None
            return fixed, rematch
        baseline, _ = observe(data)
        for variant in iter_variants(pixels, values, fractions=fractions, controls=controls, seed=seed,
                                     pad_value=int(self.config["pad_value"]), fill_methods=fill_methods, max_overlap=max_overlap):
            if cancelled and cancelled():
                raise AttributionError("Cancelled")
            if variant.get("unavailable"):
                rows.append(variant)
                continue
            modified = variant.pop("pixels")
            output = io.BytesIO()
            Image.fromarray(modified).save(output, format="PNG")
            content = output.getvalue()
            fixed, rematch = observe(content)
            record = {**variant, "modified_sha256": digest(content), "fixed_scalar": fixed,
                      "fixed_score_drop": baseline-fixed, "rematched_detection": rematch}
            if artifact_root is not None:
                directory = Path(artifact_root)
                directory.mkdir(parents=True, exist_ok=True)
                name = f"variant_{len(rows):03d}.png"
                (directory/name).write_bytes(content)
                record["modified_image"] = str((directory/name).resolve())
            rows.append(record)
        protocol = {"protocol_version": PROTOCOL_VERSION, "control_strategy": CONTROL_STRATEGY,
                    "fractions": list(fractions), "controls": controls, "seed": seed, "fill_methods": list(fill_methods),
                    "max_overlap": max_overlap, "pad_value": int(self.config["pad_value"])}
        preprocessing = {k: copy.deepcopy(self.core_config[k]) for k in
                         ("input_size", "resize", "interpolation", "color", "input_layout", "scale", "mean", "std", "pad_value")}
        preprocessing["input_dtype"] = "float32"
        stable_layer = {k: copy.deepcopy(layer[k]) for k in ("path", "level", "stride_xy", "shape") if layer and k in layer}
        report = {"status": "measured", "baseline_scalar": baseline, "target": target, **protocol, "protocol": protocol,
                  "execution": copy.deepcopy(self.execution),
                  "preprocessing": preprocessing, "layer": stable_layer, "method": method,
                  "validation_id": identity({"target": target, "map_sha256": _tensor_hash(values), "protocol": protocol,
                                             "preprocessing": preprocessing, "layer": stable_layer, "method": method}),
                  "rows": rows, "support_images": 1,
                  "limitation": "Fixed native score and rematched detections are separate; perturbations can alter the object and do not prove a cause."}
        report["assessment"] = assess_faithfulness(report)
        report["stability"] = report["assessment"]["stability"]
        return report

    def randomization_sanity(self, data, target, *, seed=42, cancelled=None):
        """Weight-dependence check on a private copy; never changes checkpoint/run."""
        import torch
        with self.lock:
            original = self.gradcam(data, output_index=target["output_index"], class_index=target["class_index"],
                                    scalar=target["scalar"], method="signed", _target_snapshot=target, cancelled=cancelled)
            clone = object.__new__(YoloRuntime)
            clone.__dict__ = {**self.__dict__, "model": copy.deepcopy(self.model), "lock": threading.RLock()}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            reset = []
            for name, module in clone.model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    module.reset_parameters()
                    reset.append(name)
            randomized = clone.gradcam(data, output_index=target["output_index"], class_index=target["class_index"],
                                       scalar=target["scalar"], method="signed", _target_snapshot=target, cancelled=cancelled)
        a = np.asarray(original["maps"]["source"]).ravel()
        b = np.asarray(randomized["maps"]["source"]).ravel()
        correlation = float(np.corrcoef(a,b)[0,1]) if np.std(a) > 0 and np.std(b) > 0 else None
        return {"status": "measured" if correlation is not None else "inconclusive", "seed": seed,
                "protocol": "reset all Conv2d/Linear weights on a private model copy; fixed native scalar",
                "reset_modules": reset, "signed_map_correlation": correlation,
                "original_status": original["status"], "randomized_status": randomized["status"],
                "limitation": "權重依賴 sanity check；沒有通用通過門檻，不代表模型正確或根因成立"}


def summarize_faithfulness(reports):
    """Bounded caller-owned batch summaries; keep failures and support explicit."""
    reports = list(reports)
    if not reports or len(reports) > 200:
        raise AttributionError("批次報告須含 1–200 筆影像結果")
    differences = []
    protocol_groups = {}
    protocol_identities = {}
    for report in reports:
        if report.get("status") != "measured":
            continue
        if report.get("protocol_version") == 2:
            from attribution_controls import assess_faithfulness
            target = report.get("target", {})
            group_identity = {"model_sha256": target.get("model_sha256"), "execution": target.get("execution") or report.get("execution"),
                              "protocol": report.get("protocol"), "scalar": target.get("scalar"),
                              "class_index": target.get("class_index"), "class_name": target.get("class_name"),
                              "method": report.get("method"), "layer": report.get("layer"), "preprocessing": report.get("preprocessing")}
            if any(not group_identity[k] for k in ("model_sha256", "execution", "protocol", "scalar", "method", "layer", "preprocessing")):
                continue
            group_id = identity(group_identity)
            for comparison in assess_faithfulness(report)["comparisons"]:
                key = f"v2/{group_id}/{comparison['fraction']}/{comparison['fill_method']}/{comparison['control_strategy']}"
                protocol_groups.setdefault(key, []).append(comparison["excess_drop"])
                protocol_identities[key] = {**group_identity, "fraction": comparison["fraction"],
                                            "fill_method": comparison["fill_method"], "control_strategy": comparison["control_strategy"]}
            continue
        for fraction in sorted({r["fraction"] for r in report["rows"]}):
            rows = [r for r in report["rows"] if r["fraction"] == fraction]
            selected = [r["fixed_score_drop"] for r in rows if r["role"] == "attribution"]
            controls = [r["fixed_score_drop"] for r in rows if r["role"] == "random_control"]
            if selected and controls:
                differences.append({"fraction": fraction, "drop_minus_random_mean": selected[0]-float(np.mean(controls))})
    groups = {}
    for fraction in sorted({r["fraction"] for r in differences}):
        values = [r["drop_minus_random_mean"] for r in differences if r["fraction"] == fraction]
        groups[str(fraction)] = {"support": len(values), "mean": float(np.mean(values)),
                                 "std": float(np.std(values,ddof=1)) if len(values)>1 else None,
                                 "scope": "unverified legacy pooled reports; execution/protocol identity unavailable"}
    for key, values in protocol_groups.items():
        groups[key] = {"support": len(values), "mean": float(np.mean(values)),
                       "std": float(np.std(values,ddof=1)) if len(values)>1 else None,
                       "protocol_version": 2, "support_unit": "reports_not_independent_images", "identity": protocol_identities[key]}
    return {"status": "measured" if groups else "inconclusive", "requested_images": len(reports),
            "measured_images": sum(r.get("status") == "measured" for r in reports), "fractions": groups,
            "limitation": "支持數與樣本變異；不使用任意 CAM 品質通過分數，不證明根因"}
