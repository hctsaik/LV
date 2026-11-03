from pathlib import Path
import os

import pytest

import model_runtimes as runtimes


def test_runtime_ids_are_closed_and_release_is_not_silently_changed():
    assert runtimes.runtime_id({}) == runtimes.LEGACY
    assert runtimes.runtime_id({"runtime_id": runtimes.MODERN, "runtime_release": "8.4.142"}) == runtimes.MODERN
    for identifier in ("../../python.exe", "C:/arbitrary/python.exe", "modern", "latest"):
        with pytest.raises(ValueError):
            runtimes.runtime_id({"runtime_id": identifier})
    with pytest.raises(ValueError):
        runtimes.runtime_id({"runtime_id": runtimes.MODERN, "runtime_release": "8.3.203"})


def test_modern_worker_cannot_inherit_legacy_site_packages(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "C:/legacy/Lib/site-packages")
    monkeypatch.setenv("PYTHONHOME", "C:/other-python")
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    env = runtimes.environment(runtimes.MODERN)
    assert env["PYTHONPATH"].split(os.pathsep) == [str(runtimes.ROOT / "scripts"), str(runtimes.ROOT)]
    assert "PYTHONHOME" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["MKL_NUM_THREADS"] == "2"
    assert os.environ["PYTHONPATH"] == "C:/legacy/Lib/site-packages"


def test_missing_modern_environment_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(runtimes, "ROOT", tmp_path)
    assert not next(r for r in runtimes.available() if r["id"] == runtimes.MODERN)["installed"]
    with pytest.raises(ValueError, match="setup_model_runtime"):
        runtimes.interpreter(runtimes.MODERN)


def test_both_interpreters_are_repository_owned(tmp_path, monkeypatch):
    monkeypatch.setattr(runtimes, "ROOT", tmp_path)
    suffix = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    for identifier, folder in ((runtimes.LEGACY, ".venv"), (runtimes.MODERN, ".runtimes/yolo-modern-8.4.142")):
        expected = tmp_path / folder / suffix
        expected.parent.mkdir(parents=True)
        expected.write_bytes(b"placeholder")
        if identifier == runtimes.MODERN:
            with pytest.raises(ValueError, match="尚未完成"):
                runtimes.interpreter(identifier)
            (expected.parent.parent / "runtime-ready.json").write_text('{"release":"8.4.142","pip_check":true}', encoding="utf-8")
        assert runtimes.interpreter(identifier) == expected


def test_profile_runtime_is_immutable_and_rejects_executable_id(tmp_path):
    from model_profiles import ModelRegistry
    from fp_attribution import job_runtime
    registry = ModelRegistry(tmp_path)
    asset = registry.ingest(b"model bytes", "model.pt")
    first = registry.create("Original", asset["sha256"], runtime_config={})
    second = registry.revise(first["id"], runtime_config={"runtime_id": runtimes.MODERN, "runtime_release": "8.4.142"})
    assert job_runtime("infer", {"profile_id": first["id"], "profile_revision": 1}, tmp_path) == runtimes.LEGACY
    assert job_runtime("infer", {"profile_id": second["id"], "profile_revision": 2}, tmp_path) == runtimes.MODERN
    assert job_runtime("comparison", {"runtime_id": runtimes.MODERN}, tmp_path) == runtimes.LEGACY
    with pytest.raises(ValueError):
        registry.revise(first["id"], runtime_config={"runtime_id": "C:/arbitrary/python.exe"})
    assert registry.get(first["id"])["revision"] == 2


def test_runtime_for_routes_modern_through_controlled_proxy(monkeypatch):
    import model_runtime_rpc
    from explain_service import runtime_for
    monkeypatch.setenv("LV_MODEL_RUNTIME_ID", runtimes.LEGACY)
    calls = []
    monkeypatch.setattr(model_runtime_rpc, "RuntimeProxy", lambda config, trusted: calls.append((config, trusted)) or "proxy")
    assert runtime_for({"runtime_id": runtimes.MODERN}, trusted=True) == "proxy"
    assert calls == [({"runtime_id": runtimes.MODERN}, True)]


def test_new_control_protocol_is_frozen_before_job_deduplication():
    from fp_attribution import prepare_job_payload
    original = {"case_id": "case", "item_id": "item"}
    prepared = prepare_job_payload("gradcam", original)
    assert "protocol_version" not in original
    assert prepared["protocol_version"] == 2
    assert prepared["fill_methods"] == ["constant", "blur"]
    assert prepared["max_overlap"] == .1
    batch = prepare_job_payload("batch_gradcam", {"items": [original]})
    assert batch["items"] == [prepared]


def test_onnx_inspection_preserves_export_recipe_without_guessing_gradient(tmp_path):
    import onnx
    from fp_onnx_demo import make_demo_model
    from fp_attribution import inspect_onnx
    path = make_demo_model(tmp_path / "processed.onnx", detection=True)
    model = onnx.load(path)
    for key, value in {"version": "8.4.142", "task": "detect", "args": "{'nms': False, 'max_det': 300}"}.items():
        row = model.metadata_props.add()
        row.key, row.value = key, value
    onnx.save(model, path)
    result = inspect_onnx(path)
    assert result["export_recipe"]["declared_args"]["nms"] is False
    assert result["export_recipe"]["head_verification"] == "unverified"
    assert not result["capabilities"]["gradcam"]
    next(row for row in model.metadata_props if row.key == "task").value = "segment"
    onnx.save(model, path)
    assert not inspect_onnx(path)["capabilities"]["inference"]
