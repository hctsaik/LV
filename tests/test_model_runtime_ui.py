"""UI contract fixtures; actual modern execution is validated separately."""
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from investigation_jobs import JobManager, JobStore
from model_profiles import ModelRegistry
import model_runtimes


def test_inspection_explicit_runtime_head_survive_profile_save(tmp_path, monkeypatch):
    jobs, registry = JobStore(tmp_path), ModelRegistry(tmp_path)
    results = {}
    def submit(task, payload, compute_key):
        job = jobs.submit(task, payload, compute_key)
        results[job["id"]] = {"inspection": {
            "family": "yolo26", "task": "detect", "capabilities": {"inference": True, "gradcam": True},
            "default_config": {"class_names": ["scratch"], "input_size": [32, 32], "output_format": "xyxy",
                "runtime_id": payload["runtime_id"], "runtime_release": payload["runtime_release"],
                "head_path": payload["head_path"], "nms": payload["nms"], "backend": "pytorch"}}}
        return jobs._update(job["id"], {"state": "succeeded"})
    monkeypatch.setattr(jobs, "result", lambda job_id: results[job_id])
    manager = SimpleNamespace(store=jobs, submit=submit)
    monkeypatch.setattr(JobManager, "get", lambda *args, **kwargs: manager)
    monkeypatch.setattr(model_runtimes, "available", lambda: [
        {"id": identifier, "installed": True, "state": "ready"} for identifier in model_runtimes.RELEASES])
    app = AppTest.from_string(f'''
from fp_cases import CaseStore
from investigation_jobs import JobManager
from investigation_models_page import render_model_library
store = CaseStore({str(tmp_path)!r})
render_model_library(store, JobManager.get(store.root, {{}}))
''', default_timeout=30).run()
    app.selectbox(key="model_import_runtime").set_value(model_runtimes.MODERN).run()
    app.selectbox(key="model_import_head").set_value("one_to_one").run()
    path = tmp_path / "controlled.pt"
    path.write_bytes(b"controlled fixture")
    app.text_input(key="model_import_path").set_value(str(path)).run()
    next(b for b in app.button if b.label == "檢查模型").click().run()
    assert not app.exception
    submitted = jobs.list()[0]["payload"]
    assert submitted["runtime_id"] == model_runtimes.MODERN
    assert submitted["head_path"] == "one_to_one" and submitted["nms"] is False
    next(c for c in app.checkbox if c.label == "我已核對類別與現場輸入／判定設定").check()
    next(b for b in app.button if b.label == "儲存模型設定").click().run()
    assert not app.exception
    config = registry.list()[0]["runtime_config"]
    assert config["runtime_id"] == model_runtimes.MODERN and config["runtime_release"] == "8.4.142"
    assert config["head_path"] == "one_to_one" and config["nms"] is False
