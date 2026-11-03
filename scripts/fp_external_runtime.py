"""Explicit ONNX inference contracts. No architecture guessing or automatic downloads."""
from __future__ import annotations

import copy
import io
import json
import math
import os
import threading
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from fp_cases import digest, utc_now

DEFAULT_CONFIG = {
    "model_path": "", "model_version": "", "output_format": "yolo_v8",
    "class_names": ["defect"], "normal_classes": [], "input_size": [640, 640],
    "resize": "letterbox", "interpolation": "bilinear", "color": "RGB", "input_layout": "NCHW",
    "scale": 1/255, "mean": [0, 0, 0], "std": [1, 1, 1], "pad_value": 114,
    "score_activation": "none", "coordinates": "pixels", "threshold": .25,
    "nms_iou": .45, "class_agnostic_nms": False, "max_detections": 300,
    "input_name": "", "output_name": "",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_checked(config):
    if not isinstance(config, dict) or set(config) - set(DEFAULT_CONFIG):
        raise ValueError("模型設定必須為 JSON 物件，且不得含未支援欄位")
    c = copy.deepcopy(DEFAULT_CONFIG)
    c.update(config)
    for key in ["model_path", "model_version", "input_name", "output_name"]:
        if not isinstance(c[key], str):
            raise ValueError(f"{key} 須為文字")
    enums = {"output_format": {"classification", "yolo_v8", "yolo_v5", "xyxy"},
             "resize": {"letterbox", "stretch"}, "interpolation": {"bilinear", "nearest", "bicubic"},
             "color": {"RGB", "BGR"}, "input_layout": {"NCHW", "NHWC"},
             "score_activation": {"none", "softmax", "sigmoid"}, "coordinates": {"pixels", "normalized"}}
    for key, allowed in enums.items():
        if c[key] not in allowed:
            raise ValueError(f"不支援的 {key}：{c[key]}")
    if c["output_format"] != "classification" and c["score_activation"] == "softmax":
        raise ValueError("YOLO 類別分數不支援 softmax；請確認匯出端的分數語意")
    if c["output_format"] == "xyxy" and c["score_activation"] != "none":
        raise ValueError("已後處理 xyxy 輸出須提供 0–1 分數，不能再次套用 activation")
    names = c["class_names"]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError("class_names 必須為不重複的非空類別名稱陣列，順序須與模型一致")
    if not isinstance(c["normal_classes"], list) or not set(c["normal_classes"]) <= set(names):
        raise ValueError("normal_classes 必須是 class_names 的子集")
    if c["output_format"] == "classification" and (not c["normal_classes"] or len(c["normal_classes"]) == len(names)):
        raise ValueError("分類模型須明確指定正常類別，且至少保留一個瑕疵類別")
    size = c["input_size"]
    if not isinstance(size, list) or len(size) != 2 or not all(type(x) is int and 8 <= x <= 4096 for x in size):
        raise ValueError("input_size 須為 [width,height]，各介於 8–4096")
    for key in ["mean", "std"]:
        if not isinstance(c[key], list) or len(c[key]) != 3 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in c[key]):
            raise ValueError(f"{key} 須為三個有限數值")
    if min(c["std"]) <= 0:
        raise ValueError("std 必須大於零")
    for key in ["scale", "threshold", "nms_iou", "pad_value"]:
        if not isinstance(c[key], (int, float)) or not math.isfinite(c[key]):
            raise ValueError(f"{key} 須為有限數值")
    if c["scale"] <= 0 or not 0 <= c["threshold"] <= 1 or not 0 <= c["nms_iou"] <= 1:
        raise ValueError("scale 須為正值；threshold 與 nms_iou 須介於 0–1")
    if not 0 <= c["pad_value"] <= 255 or int(c["pad_value"]) != c["pad_value"]:
        raise ValueError("pad_value 須為 0–255 整數")
    if type(c["max_detections"]) is not int or not 1 <= c["max_detections"] <= 1000 or type(c["class_agnostic_nms"]) is not bool:
        raise ValueError("max_detections 須介於 1–1000，NMS 類別開關須為布林值")
    return c


@lru_cache(maxsize=2)
def _session(path, sha):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ValueError("請安裝 requirements-external-models.txt 後再執行外部模型") from exc
    data = Path(path).read_bytes()
    if digest(data) != sha:
        raise ValueError("模型檔在載入期間變更，請重新執行")
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    # Loading bytes binds all executed weights to this digest. External-data ONNX is not silently loaded.
    try:
        session = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ValueError("ONNX 載入失敗；需要單檔內嵌權重、CPU 可執行模型。" + str(exc)) from exc
    return session, threading.Lock(), ort.__version__


def prepare_input(image, config, dtype="tensor(float)"):
    c = config_checked(config)
    image = image.convert("RGB")
    w, h = image.size
    tw, th = c["input_size"]
    resample = {"bilinear": Image.Resampling.BILINEAR, "nearest": Image.Resampling.NEAREST,
                "bicubic": Image.Resampling.BICUBIC}[c["interpolation"]]
    if c["resize"] == "letterbox":
        ratio = min(tw/w, th/h)
        nw, nh = max(1, round(w*ratio)), max(1, round(h*ratio))
        left, top = (tw-nw)//2, (th-nh)//2
        preview = Image.new("RGB", (tw, th), (int(c["pad_value"]),)*3)
        preview.paste(image.resize((nw, nh), resample), (left, top))
    else:
        nw, nh, left, top = tw, th, 0, 0
        preview = image.resize((tw, th), resample)
    array = np.asarray(preview).astype(np.float32)
    if c["color"] == "BGR":
        array = array[..., ::-1]
    array = (array*c["scale"] - np.asarray(c["mean"], dtype=np.float32)) / np.asarray(c["std"], dtype=np.float32)
    types = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(uint8)": np.uint8}
    if dtype not in types:
        raise ValueError(f"不支援的輸入 tensor 型別：{dtype}")
    if dtype == "tensor(uint8)" and (array.min() < 0 or array.max() > 255 or not np.all(array == np.round(array))):
        raise ValueError("uint8 輸入不可含小數正規化值；請核對 scale、mean、std")
    array = array.astype(types[dtype])
    if c["input_layout"] == "NCHW":
        array = array.transpose(2, 0, 1)
    return np.ascontiguousarray(array[None]), preview, {
        "source_size": [w, h], "input_size": [tw, th], "resized_size": [nw, nh], "padding_xy": [left, top],
        "scale_xy": [nw/w, nh/h], "orientation": "stored_pixels_no_exif_rotation"}


def iou(a, b):
    inter = max(0.0, min(a[2], b[2])-max(a[0], b[0])) * max(0.0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/union if union > 0 else 0.0


def _activation(values, mode):
    if mode == "softmax":
        exp = np.exp(values - np.max(values, axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)
    if mode == "sigmoid":
        return 1 / (1 + np.exp(-np.clip(values, -80, 80)))
    return values


def _probabilities(values):
    if np.any(values < -1e-6) or np.any(values > 1+1e-6):
        raise ValueError("分數不在 0–1；請確認 logits/probability 與 activation 設定")
    return np.clip(values, 0, 1)


def decode_output(raw, config, geometry):
    c = config_checked(config)
    arr = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("模型輸出含 NaN 或 infinity")
    names = c["class_names"]
    fmt = c["output_format"]
    if fmt == "classification":
        if arr.shape not in {(len(names),), (1, len(names))}:
            raise ValueError("分類輸出須為 [1,C] 或 [C]，C 須符合 class_names")
        scores = _probabilities(_activation(arr.reshape(-1), c["score_activation"]))
        if c["score_activation"] != "sigmoid" and not np.isclose(scores.sum(), 1, atol=1e-3):
            raise ValueError("單類別分類機率总和須為 1；請確認 softmax／輸出語意")
        index = int(np.argmax(scores))
        decision = "abstain" if scores[index] < c["threshold"] else names[index]
        return {"task": "classification", "scores": {n: float(v) for n, v in zip(names, scores)},
                "class": names[index], "score": float(scores[index]), "decision": decision,
                "alarm": decision != "abstain" and decision not in c["normal_classes"], "detections": [],
                "raw_candidates_available": False}
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError("偵測輸出須為 batch=1 的二维候選矩陣")
    if fmt == "yolo_v8":
        if arr.shape[0] != 4+len(names):
            raise ValueError("YOLO class-score 格式須為 [1,4+C,N]；請核對類別數與輸出格式")
        arr = arr.T
    elif arr.shape[1] != (6 if fmt == "xyxy" else 5+len(names)):
        raise ValueError("輸出欄數與 xyxy 或 YOLO objectness 格式不相符")
    if arr.shape[0] > 100000:
        raise ValueError("候選框超過 100000，請先核對輸出布局")
    if fmt == "xyxy":
        classes = arr[:, 5]
        if np.any(classes != np.floor(classes)) or np.any(classes < 0) or np.any(classes >= len(names)):
            raise ValueError("xyxy 輸出的 class ID 超出 class_names")
        scores = _probabilities(arr[:, 4])
        labels = classes.astype(int)
        boxes = arr[:, :4].copy()
    else:
        score_start = 4 if fmt == "yolo_v8" else 5
        probs = _probabilities(_activation(arr[:, score_start:], c["score_activation"]))
        labels = np.argmax(probs, axis=1)
        scores = probs[np.arange(len(arr)), labels]
        if fmt == "yolo_v5":
            scores = scores * _probabilities(_activation(arr[:, 4], c["score_activation"]))
        boxes = arr[:, :4].copy()
        boxes[:, :2] = arr[:, :2] - arr[:, 2:4]/2
        boxes[:, 2:4] = arr[:, :2] + arr[:, 2:4]/2
    if c["coordinates"] == "normalized":
        boxes *= np.asarray(c["input_size"]*2)
    left, top = geometry["padding_xy"]
    sx, sy = geometry["scale_xy"]
    w, h = geometry["source_size"]
    boxes = (boxes - np.array([left, top, left, top])) / np.array([sx*w, sy*h, sx*w, sy*h])
    boxes = np.clip(boxes, 0, 1)
    order = np.argsort(-scores, kind="stable")
    candidates = []
    for index in order:
        b = boxes[index].tolist()
        if b[0] >= b[2] or b[1] >= b[3]:
            continue
        candidates.append({"class": names[labels[index]], "score": float(scores[index]), "box": b,
                           "output_index": int(index)})
    # Raw vectors live in NPZ; retain enough sorted candidates for target inspection in JSON.
    final = []
    for candidate in candidates:
        if candidate["score"] < c["threshold"]:
            break
        if fmt != "xyxy" and any((c["class_agnostic_nms"] or old["class"] == candidate["class"]) and
                                  iou(old["box"], candidate["box"]) > c["nms_iou"] for old in final):
            continue
        final.append(candidate)
        if len(final) == c["max_detections"]:
            break
    return {"task": "detection", "detections": final, "candidates": candidates[:10000],
            "candidate_count": len(candidates), "candidate_json_truncated": len(candidates) > 10000,
            "raw_candidates_available": fmt != "xyxy", "nms_applied": fmt != "xyxy",
            "alarm": any(d["class"] not in c["normal_classes"] for d in final), "decision": "detections"}


class Runtime:
    def __init__(self, config):
        self.config = config_checked(config)
        path = Path(self.config["model_path"]).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".onnx":
            raise ValueError("請指定存在的 .onnx 模型檔")
        sha = digest(path.read_bytes())
        self.session, self.lock, version = _session(str(path), sha)
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1:
            raise ValueError("目前需單一影像 tensor 輸入；請先匯出包含其他固定輸入的 ONNX")
        self.input = inputs[0]
        if self.config["input_name"] and self.input.name != self.config["input_name"]:
            raise ValueError("input_name 與模型不符")
        output_name = self.config["output_name"]
        if not output_name and len(outputs) != 1:
            raise ValueError("模型有多個輸出，請明確指定 output_name")
        self.output_name = output_name or outputs[0].name
        if self.output_name not in {o.name for o in outputs}:
            raise ValueError("output_name 與模型不符")
        contract = {k: v for k, v in self.config.items() if k != "model_path"}
        contract.update(model_sha256=sha, schema_version=1, input_name=self.input.name,
                        output_name=self.output_name, input_dtype=self.input.type)
        self.identity = {"contract": contract, "contract_id": digest(canonical(contract).encode()),
                         "runtime": {"name": "onnxruntime", "version": version, "provider": "CPUExecutionProvider"}}

    def infer(self, data, artifact_root=None):
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            tensor, preview, geometry = prepare_input(image, self.config, self.input.type)
        if len(self.input.shape) != len(tensor.shape) or any(isinstance(a, int) and a != b for a, b in zip(self.input.shape, tensor.shape)):
            raise ValueError(f"輸入尺寸不符：模型 {self.input.shape}，設定 {list(tensor.shape)}")
        with self.lock:
            try:
                raw = self.session.run([self.output_name], {self.input.name: tensor})[0]
            except Exception as exc:
                raise ValueError("外部模型推論失敗：" + str(exc)) from exc
        prediction = decode_output(raw, self.config, geometry)
        run = {"id": "run_" + uuid.uuid4().hex[:16], "at": utc_now(), **copy.deepcopy(self.identity),
               "image_sha256": digest(data), "geometry": geometry, "prediction": prediction,
               "input_sha256": digest(str(tensor.dtype).encode()+canonical(list(tensor.shape)).encode()+tensor.tobytes()),
               "output_sha256": digest(str(raw.dtype).encode()+canonical(list(raw.shape)).encode()+raw.tobytes()),
               "input_shape": list(tensor.shape), "output_shape": list(raw.shape), "origin": "lv_local_inference"}
        if artifact_root is not None:
            root = Path(artifact_root) / run["id"]
            root.mkdir(parents=True, exist_ok=False)
            preview.save(root / "model_input.png")
            with Image.open(io.BytesIO(data)) as source:
                source.convert("RGB").save(root / "source.png")
            np.savez_compressed(root / "tensors.npz", input=tensor, output=raw)
            run["artifact_dir"] = str(root.resolve())
            (root / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        return run


def compare_reference(run, reference, source_kind="original", *, score_atol=.001, box_atol=.001):
    base = {"status": "unavailable", "reasons": [], "score_atol": score_atol, "box_atol": box_atol}
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= .1 for v in [score_atol, box_atol]):
        raise ValueError("重播容許誤差須介於 0–0.1")
    if source_kind != "original":
        return {**base, "reasons": ["來源不是原圖，不能確認當時判定已重現"]}
    if not isinstance(reference, dict) or reference.get("origin") != "deployment" or not reference.get("recorded_at"):
        return {**base, "reasons": ["需要當時部署輸出、記錄時間與模型／前後處理契約；本次輸出不能自我驗證"]}
    for key in ["image_sha256", "contract_id"]:
        if not reference.get(key) or reference[key] != run[key]:
            return {**base, "reasons": [f"歷史 {key} 缺少或不一致"]}
    expected, actual = reference.get("prediction"), run["prediction"]
    if not isinstance(expected, dict):
        raise ValueError("歷史紀錄缺少 prediction")
    reasons = []
    if expected.get("task") != actual["task"] or type(expected.get("alarm")) is not bool:
        raise ValueError("歷史 prediction 須提供同型 task 與布林 alarm")
    if expected["alarm"] != actual["alarm"]:
        reasons.append("最後警報不同")
    if actual["task"] == "classification":
        scores = expected.get("scores")
        if not isinstance(scores, dict) or set(scores) != set(actual["scores"]):
            raise ValueError("歷史分類紀錄須包含全部類別分數")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in scores.values()):
            raise ValueError("歷史分類分數須為有限數值")
        if expected.get("class") != actual["class"] or expected.get("decision") != actual["decision"]:
            reasons.append("分類／拒判結果不同")
        if any(abs(scores[k]-actual["scores"][k]) > score_atol for k in scores):
            reasons.append("分類分數超過容許誤差")
    else:
        rows = expected.get("detections")
        if not isinstance(rows, list):
            raise ValueError("歷史偵測紀錄須含完整 detections")
        actual_rows = actual["detections"]
        edges = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("class"), str):
                raise ValueError("歷史偵測格式不正確")
            from fp_cases import roi_checked
            box = roi_checked(row.get("box", []))
            score = row.get("score")
            if not isinstance(score, (float, int)) or not math.isfinite(score):
                raise ValueError("歷史分數須為有限數值")
            matches = [j for j, r in enumerate(actual_rows) if r["class"] == row["class"] and
                       max(abs(a-b) for a, b in zip(r["box"], box)) <= box_atol and abs(r["score"]-score) <= score_atol]
            edges.append(matches)
        # Maximum bipartite matching avoids an order-dependent false mismatch for close boxes.
        matched = {}
        for start in range(len(edges)):
            queue, parent, visited_right = [start], {start: None}, set()
            found = None
            for left in queue:
                for right in edges[left]:
                    if right in visited_right:
                        continue
                    visited_right.add(right)
                    if right not in matched:
                        found = (left, right)
                        break
                    following = matched[right]
                    if following not in parent:
                        parent[following] = (left, right)
                        queue.append(following)
                if found is not None:
                    break
            if found is not None:
                left, right = found
                while True:
                    matched[right] = left
                    step = parent[left]
                    if step is None:
                        break
                    left, right = step
        if len(matched) < len(rows):
            reasons.append("歷史框／類別／分數沒有相符輸出")
        if len(matched) < len(actual_rows):
            reasons.append("本次出現額外輸出框")
    return {**base, "status": "mismatched" if reasons else "matched", "reasons": reasons or ["完整輸出在設定誤差內相符"],
            "reference_sha256": digest(canonical(reference).encode()), "reference_recorded_at": reference["recorded_at"]}
