"""Real subprocess/ONNX gates; modern YOLO numerical gates live separately."""
import io
import os

from PIL import Image
import pytest

import model_runtimes as runtimes
from model_runtime_rpc import RuntimeProxy
from fp_external_runtime import DEFAULT_CONFIG, Runtime
from fp_onnx_demo import make_demo_model


@pytest.fixture
def modern_ready():
    if not next(row for row in runtimes.available() if row["id"] == runtimes.MODERN)["installed"]:
        pytest.skip("Install the separately pinned modern environment first")


def test_real_package_origins_and_versions_are_isolated(modern_ready):
    old = runtimes.probe(runtimes.LEGACY)
    new = runtimes.probe(runtimes.MODERN)
    assert old["release"] == "8.3.203" and new["release"] == "8.4.142"
    assert ".venv" in old["package_origin"]
    assert ".runtimes" in new["package_origin"]
    assert old["python"] != new["python"]


def test_real_inference_rpc_reuses_worker_and_matches_local_prediction(tmp_path, modern_ready):
    path = make_demo_model(tmp_path / "model.onnx", detection=True)
    config = {**DEFAULT_CONFIG, "model_path": str(path), "input_size": [32, 32], "resize": "stretch",
              "class_names": ["scratch"], "output_format": "xyxy", "threshold": .1}
    image = io.BytesIO()
    Image.new("RGB", (32, 32), (240, 150, 150)).save(image, "PNG")
    data = image.getvalue()
    baseline = Runtime(config).infer(data)
    proxy = RuntimeProxy({**config, "runtime_id": runtimes.MODERN}, trusted=True)
    pid = proxy._process.pid
    try:
        first = proxy.infer(data, tmp_path / "artifacts")
        second = proxy.infer(data)
        assert first["prediction"] == second["prediction"] == baseline["prediction"]
        assert first["execution_runtime"]["runtime_id"] == runtimes.MODERN
        assert proxy._process.pid == pid and pid != os.getpid()
        assert proxy.identity["execution_runtime"]["runtime_release"] == "8.4.142"
        with pytest.raises(ValueError, match="Unsupported"):
            proxy._call({"op": "execute", "python": "arbitrary.py"})
    finally:
        proxy.close()
    assert proxy._process.poll() is not None


def test_real_paired_evaluation_onnx_and_modern_yolo(tmp_path, modern_ready):
    """Reuse a real modern numerical-gate fixture, without generating extra models."""
    from pathlib import Path
    from fp_cases import CaseStore
    from fp_investigation import paired_evaluation
    from test_fp_investigation import manifest
    checkpoint = Path(os.environ.get("LV_TEST_MODERN_PT", "__missing_modern_fixture__"))
    if not checkpoint.is_file():
        pytest.skip("Set LV_TEST_MODERN_PT to a validated modern two-class fixture")
    store = CaseStore(tmp_path / "cases")
    image = io.BytesIO()
    Image.new("RGB", (33, 29), (77, 91, 100)).save(image, "PNG")
    source = store.ingest(image.getvalue(), "case.png")
    case = store.create(source, "Cross-runtime evaluation")
    onnx = make_demo_model(tmp_path / "baseline.onnx", detection=True)
    common = {**DEFAULT_CONFIG, "input_size": [64, 64], "resize": "stretch", "class_names": ["scratch", "dent"],
              "normal_classes": [], "threshold": 0., "max_detections": 20}
    baseline = {**common, "model_path": str(onnx), "output_format": "xyxy", "input_size": [32, 32]}
    candidate = {**common, "model_path": str(checkpoint.resolve()), "output_format": "yolo_v8", "backend": "pytorch",
                 "runtime_id": runtimes.MODERN, "runtime_release": "8.4.142", "head_path": "one_to_many", "nms": None}
    result = paired_evaluation(case, store, baseline, candidate, manifest(tmp_path), min_support=1)
    assert result["candidate_identity"]["execution_runtime"]["runtime_id"] == runtimes.MODERN
    assert result["candidate_identity"]["runtime"]["version"] == "8.4.142"
    assert result["baseline_identity"]["runtime"]["name"] != result["candidate_identity"]["runtime"]["name"]
    assert len(result["per_image"]) == 2
    assert result["candidate_identity"]["contract_id"] != result["baseline_identity"]["contract_id"]
    # Returning from the evaluation releases the RPC object and closes its child.
    import psutil
    assert not [p for p in psutil.Process().children(recursive=True)
                if p.is_running() and "model_runtime_rpc.py" in " ".join(p.cmdline())]
