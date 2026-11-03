import io
import time

from PIL import Image
from streamlit.testing.v1 import AppTest

from fp_cases import CaseStore
from fp_onnx_demo import make_demo_model
from investigation_jobs import JobManager
from model_profiles import ModelRegistry
import pytest


def inspection_fixture_task(payload, context):
    """A UI schema fixture only; real model execution is tested separately."""
    return {"inspection": payload["inspection"]}


def _wait(manager, job_id):
    until = time.monotonic() + 40
    while time.monotonic() < until:
        result = manager.store.get(job_id)
        if result["state"] in {"succeeded", "failed"}:
            assert result["state"] == "succeeded", result
            return manager.store.result(job_id)
        time.sleep(.05)
    raise AssertionError("Worker did not complete")


def _button(app, label):
    return next(item for item in app.button if item.label == label)


def test_second_inspection_selects_new_onnx_after_pt_history(tmp_path, monkeypatch):
    """Controlled inspection results exercise widget cleanup without workers."""
    from types import SimpleNamespace
    from investigation_jobs import JobStore
    jobs = JobStore(tmp_path)
    registry = ModelRegistry(tmp_path)
    results = {}

    def submit(task, payload, compute_key):
        job = jobs.submit(task, payload, compute_key)
        asset = registry.asset(payload["asset_sha"])
        results[job["id"]] = {"inspection": {
            "format": asset["format"], "task": "detect", "class_names": ["scratch"],
            "input_size": [32, 32], "capabilities": {"inference": True, "gradcam": asset["format"] == "pt"},
            "default_config": {"output_format": "yolo_v8" if asset["format"] == "pt" else "xyxy"}}}
        return jobs._update(job["id"], {"state": "succeeded"})

    monkeypatch.setattr(jobs, "result", lambda job_id: results[job_id])
    manager = SimpleNamespace(store=jobs, submit=submit)
    monkeypatch.setattr(JobManager, "get", lambda *args, **kwargs: manager)
    app = AppTest.from_string(f'''
from fp_cases import CaseStore
from investigation_jobs import JobManager
from investigation_models_page import render_model_library
store = CaseStore({str(tmp_path)!r})
render_model_library(store, JobManager.get(store.root, {{}}))
''', default_timeout=30).run()
    for suffix in ("pt", "onnx"):
        path = tmp_path / ("second-inspection." + suffix)
        path.write_bytes(("controlled-" + suffix).encode())
        next(field for field in app.text_input if field.label == "LV 主機模型路徑").set_value(str(path)).run()
        _button(app, "檢查模型").click().run()
        assert not app.exception
        newest = jobs.list()[0]
        assert app.session_state["model_inspect_selection"] == newest["id"]
        assert next(field for field in app.text_input if field.label == "模型名稱").value == path.name
        assert registry.asset(newest["payload"]["asset_sha"])["format"] == suffix
        next(field for field in app.checkbox if field.label == "我已核對類別與現場輸入／判定設定").check()
        _button(app, "儲存模型設定").click().run()
        assert not app.exception
        selected_profile = registry.get(app.session_state["model_library_selected"])
        assert selected_profile["name"] == path.name
        assert app.session_state["model_revision_" + selected_profile["id"]] == 1
    assert len(jobs.list()) == 2


def test_actual_onnx_inspection_guided_save_trial_preview_and_promotion(tmp_path):
    from fp_attribution import HANDLERS, completed_result
    store = CaseStore(tmp_path)
    registry = ModelRegistry(tmp_path)
    model = make_demo_model(tmp_path / "fixture.onnx", detection=True)
    asset = registry.ingest(model.read_bytes(), model.name)
    manager = JobManager.get(tmp_path, HANDLERS)
    try:
        job = manager.submit("inspect", {"asset_sha": asset["sha256"], "asset_suffix": ".onnx", "workspace_root": str(tmp_path)}, "inspect-fixture")
        result = _wait(manager, job["id"])
        assert result["inspection"]["capabilities"]["inference"]
        script = f'''
from fp_cases import CaseStore
from fp_attribution import HANDLERS
from investigation_jobs import JobManager
from investigation_models_page import render_model_library
store = CaseStore({str(tmp_path)!r})
render_model_library(store, JobManager.get(store.root, HANDLERS))
'''
        app = AppTest.from_string(script, default_timeout=30).run()
        assert not app.exception
        next(field for field in app.text_input if field.label == "產線／站點名稱").set_value("Line A")
        next(field for field in app.text_area if field.label == "機台後續判定規則（不知道可留空）").set_value("Three consecutive alarms")
        next(c for c in app.checkbox if c.label == "我已核對類別與現場輸入／判定設定").check()
        _button(app, "儲存模型設定").click().run()
        assert not app.exception
        profile = registry.list()[0]
        assert app.session_state["model_library_selected"] == profile["id"]
        assert app.session_state["model_revision_" + profile["id"]] == profile["revision"]
        assert profile["deployment_asset"]["sha256"] == asset["sha256"]
        assert profile["metadata"]["station_name"] == "Line A"
        assert profile["metadata"]["machine_rules"] == "Three consecutive alarms"
        assert profile["metadata"]["field_provenance"]["machine_rules"] == "user_reported_unverified"
        data = io.BytesIO()
        Image.new("RGB", (48, 24), (220, 220, 220)).save(data, format="PNG")
        source = store.ingest(data.getvalue(), "image.png", kind="original")
        job = manager.submit("infer", {"profile_id": profile["id"], "profile_revision": profile["revision"], "source": source, "case_id": None}, "infer-fixture")
        _wait(manager, job["id"])
        resolved = completed_result(store, job["id"])
        assert resolved["run"]["prediction"]["detections"]
        app.run()
        assert not app.exception
        _button(app, "用此圖建立案件").click().run()
        assert not app.exception
        cases = store.list()
        assert len(cases) == 1
        case = store.load(cases[0]["id"])
        assert len(case["external_runs"]) == 1
        assert len(manager.store.list(states={"succeeded"})) == 2  # Promotion did not infer again.
        assert app.session_state["investigation_next_case"] == case["id"]
        assert manager.store.get(job["id"])["publication"]["state"] == "published"
        _button(app, "開啟已保存的案件").click().run()
        assert not app.exception
        assert len(store.list()) == 1
    finally:
        manager.close()


def test_backup_controls_work_even_before_importing_any_model(tmp_path):
    from investigation_backup import inspect_backup
    manager = JobManager.get(tmp_path, {})
    try:
        app = AppTest.from_string(f'''
from fp_cases import CaseStore
from investigation_jobs import JobManager
from investigation_models_page import render_model_library
store = CaseStore({str(tmp_path)!r})
render_model_library(store, JobManager.get(store.root, {{}}))
''', default_timeout=30).run()
        assert not app.exception
        _button(app, "建立工作區備份").click().run()
        assert not app.exception
        ready = app.session_state["workspace_backup_ready"]
        assert inspect_backup(ready["data"])["includes_models"] is False
    finally:
        manager.close()


@pytest.mark.parametrize("first_format", ["onnx", "pt"])
def test_companion_import_keeps_distinct_deployment_and_gradient_recipes(tmp_path, monkeypatch, first_format):
    from pathlib import Path
    from fp_external_runtime import DEFAULT_CONFIG
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    registry = ModelRegistry(tmp_path)
    original = registry.ingest(b"original", "original." + first_format)
    second_format = "pt" if first_format == "onnx" else "onnx"
    companion = registry.ingest(b"companion", "companion." + second_format)
    config = {**DEFAULT_CONFIG, "class_names": ["scratch"], "input_size": [32, 32], "threshold": .63,
              "output_format": "xyxy" if first_format == "onnx" else "yolo_v5"}
    profile = registry.create("Line A", original, original if first_format == "pt" else None, config,
                              {"inference": True, "gradcam": first_format == "pt"}, {"family": "yolov5"} if first_format == "pt" else {})
    inspection = {"format": second_format, "family": "yolov5" if second_format == "pt" else "onnx", "task": "detect",
                  "class_names": ["scratch"], "input_size": [32, 32], "warnings": [],
                  "capabilities": {"inference": True, "gradcam": second_format == "pt"},
                  "default_config": {**config, "output_format": "yolo_v5" if second_format == "pt" else "xyxy"}}
    manager = JobManager.get(tmp_path, {"inspect": "test_investigation_models_page:inspection_fixture_task"})
    try:
        job = manager.submit("inspect", {"asset_sha": companion["sha256"], "inspection": inspection}, "fixture-pair")
        _wait(manager, job["id"])
        app = AppTest.from_string(f'''
from fp_cases import CaseStore
from investigation_jobs import JobManager
from investigation_models_page import render_model_library
store = CaseStore({str(tmp_path)!r})
render_model_library(store, JobManager.get(store.root, {{"inspect": "test_investigation_models_page:inspection_fixture_task"}}))
''', default_timeout=30).run()
        assert not app.exception
        next(s for s in app.selectbox if s.label == "儲存位置").set_value(profile).run()
        next(c for c in app.checkbox if c.label == "我已核對類別與現場輸入／判定設定").check()
        _button(app, "儲存模型設定").click().run()
        assert not app.exception
        revised = registry.get(profile["id"])
        assert app.session_state["model_library_selected"] == revised["id"]
        assert app.session_state["model_revision_" + revised["id"]] == revised["revision"]
        assert revised["revision"] == 2
        assert revised["deployment_asset"]["format"] == "onnx"
        assert revised["gradient_asset"]["format"] == "pt"
        assert revised["runtime_config"]["output_format"] == "xyxy"
        assert revised["metadata"]["gradient_config"]["output_format"] == "yolo_v5"
        assert revised["metadata"]["gradient_family"] == "yolov5"
        assert revised["capabilities"]["gradcam"]
        assert registry.get(profile["id"], 1)["runtime_config"] == {k:v for k,v in config.items() if k != "model_path"}
    finally:
        manager.close()
