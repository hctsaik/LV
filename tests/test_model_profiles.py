import io
import json
import zipfile

import pytest

from model_profiles import ModelRegistry


def test_upload_identity_filenames_size_and_no_execution(tmp_path):
    registry = ModelRegistry(tmp_path)
    first = registry.ingest(b"checkpoint-one", "../../unsafe/model.pt")
    duplicate = registry.ingest(b"checkpoint-one", "other.pt")
    different = registry.ingest(b"checkpoint-two", "model.pt")
    assert first == duplicate
    assert first["sha256"] != different["sha256"]
    assert first["name"] == "model.pt"
    assert first["inspection_status"] == "not_inspected"
    assert registry.asset_path(first).is_relative_to(tmp_path / "models")
    for data, name in [(b"", "model.pt"), (b"abc", "model.py"), (b"12345", "model.pt")]:
        with pytest.raises(ValueError):
            registry.ingest(data, name, max_bytes=4)
    assert not list((tmp_path / "models").rglob(".pending-*"))


def test_revisions_detach_inputs_and_remain_portable(tmp_path):
    registry = ModelRegistry(tmp_path)
    onnx = registry.ingest(b"onnx-model", "v1.onnx")
    pt = registry.ingest(b"pt-model", "v1.pt")
    config = {"model_path": "C:/old/server/model.onnx", "threshold": .5, "class_names": ["scratch"]}
    first = registry.create("Line A", onnx, runtime_config=config, capabilities={"replay": True})
    config["class_names"].append("caller-mutation")
    second = registry.revise(first["id"], gradient_asset=pt, expected_revision=1)
    assert second["revision"] == 2
    assert registry.get(first["id"], 1)["gradient_asset"] is None
    assert first["runtime_config"]["class_names"] == ["scratch"]
    assert "model_path" not in second["runtime_config"]
    assert registry.runtime_config(first["id"], 1)["model_path"] == str(registry.asset_path(onnx))
    with pytest.raises(ValueError, match="another session"):
        registry.revise(first["id"], expected_revision=1, metadata={})
    registry.rename(first["id"], "New display name")
    assert registry.get(first["id"], 1)["name"] == "New display name"
    assert registry.archive(first["id"])["archived"]
    assert registry.list() == []
    assert registry.get(first["id"], 1)["runtime_config"]["threshold"] == .5
    assert registry.restore(first["id"])["revision"] == 2
    assert len(registry.list()) == 1


def test_restore_requires_original_content_and_availability_detects_corruption(tmp_path):
    registry = ModelRegistry(tmp_path)
    asset = registry.ingest(b"weights", "test.pt")
    model = registry.create("Demo", gradient_asset=asset)
    path = registry.asset_path(asset)
    path.write_bytes(b"broken")
    assert not registry.availability(model["id"])["gradient_asset"]["available"]
    with pytest.raises(ValueError, match="content identity"):
        registry.restore_asset(asset["sha256"], b"other")
    registry.restore_asset(asset["sha256"], b"weights")
    assert registry.asset_path(asset).read_bytes() == b"weights"
    path.unlink()
    registry.ingest(b"weights", "renamed.pt")
    assert registry.asset_path(asset).read_bytes() == b"weights"


@pytest.mark.parametrize("include_models", [True, False])
def test_backup_into_different_workspace_and_explicit_omitted_models(tmp_path, include_models):
    source = ModelRegistry(tmp_path / "source")
    asset = source.ingest(b"weights", "test.pt")
    model = source.create("Production", gradient_asset=asset)
    source.revise(model["id"], metadata={"station": "A"})
    target = ModelRegistry(tmp_path / "target")
    result = target.restore_backup(source.backup(include_models=include_models))
    assert result["includes_models"] is include_models
    assert target.get(model["id"])["metadata"] == {"station": "A"}
    assert target.get(model["id"], 1)["metadata"] == {}
    assert target.availability(model["id"])["gradient_asset"]["available"] is include_models
    if not include_models:
        target.restore_asset(asset["sha256"], b"weights")
    assert target.asset_path(asset).is_relative_to(tmp_path / "target")


def test_backup_conflict_and_tampering_do_not_replace_profile(tmp_path):
    registry = ModelRegistry(tmp_path)
    asset = registry.ingest(b"weights", "test.pt")
    model = registry.create("Saved", gradient_asset=asset)
    backup = registry.backup(include_models=True)
    registry.rename(model["id"], "Current")
    with pytest.raises(ValueError, match="conflicts"):
        registry.restore_backup(backup)
    assert registry.get(model["id"])["name"] == "Current"
    payload = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(backup)) as original, zipfile.ZipFile(payload, "w") as modified:
        manifest = json.loads(original.read("manifest.json"))
        manifest["assets"][0]["relative_path"] = "../../outside.pt"
        modified.writestr("manifest.json", json.dumps(manifest))
    with pytest.raises(ValueError, match="asset path"):
        ModelRegistry(tmp_path / "target").restore_backup(payload.getvalue())
