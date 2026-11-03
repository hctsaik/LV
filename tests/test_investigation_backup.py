import hashlib
import io
import json
from pathlib import Path
import sqlite3
import stat
import zipfile

from PIL import Image
import pytest

from fp_cases import CaseStore
from investigation_backup import create_backup, inspect_backup, managed_restore_destination, restore_backup
from investigation_jobs import JobStore
from model_profiles import ModelRegistry


def _workspace(root):
    store = CaseStore(root)
    image = io.BytesIO()
    Image.new("RGB", (12, 8), "white").save(image, format="PNG")
    source = store.ingest(image.getvalue(), "case.png", kind="original")
    case = store.create(source, "Restorable case")
    registry = ModelRegistry(root)
    asset = registry.ingest(b"model-fixture", "model.pt")
    profile = registry.create("Station A", asset, asset, {"threshold": .4}, {"gradcam": True}, {"station_name": "A"})
    registry.revise(profile["id"], metadata={"station_name": "A", "machine_rules": "three consecutive alarms"})
    jobs = JobStore(root)
    job = jobs.submit("infer", {"workspace_root": str(root), "source": source}, "test-input", context={"case_id": case["id"]})
    artifacts = jobs.artifact_dir(job["id"])
    artifacts.mkdir(parents=True)
    result = {"input": str(store.image_path(source)), "artifact_dir": str(artifacts),
              "portable": "@artifact/evidence.txt", "outside_provenance": "Z:/deployment/original.png"}
    (artifacts / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (artifacts / "evidence.txt").write_text("Evidence remains unchanged", encoding="utf-8")
    manifest = [{"path": p.name, "size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts.iterdir()]
    jobs._update(job["id"], {"state": "succeeded", "artifacts": manifest})
    jobs.mark_published(job["id"], case_id=case["id"], error="Original case publication conflict")
    case["external_runs"] = [{"id": "run-fixture", "artifact_dir": str(artifacts), "artifact_relative": artifacts.relative_to(root).as_posix()}]
    case = store.save(case)
    datasets = root / "datasets"
    datasets.mkdir()
    (datasets / "dataset_example.json").write_text(json.dumps({"id": "dataset_example", "rows": [{"source": source, "path": str(store.image_path(source))}]}), encoding="utf-8")
    return store, case, registry, profile, asset, jobs, job


@pytest.mark.parametrize("include_models", [True, False])
def test_portable_complete_workspace_backup_preserves_case_history_and_results(tmp_path, include_models):
    origin = tmp_path / "original"
    store, case, registry, profile, asset, jobs, job = _workspace(origin)
    before = jobs.result(job["id"])
    backup = create_backup(origin, include_models=include_models)
    manifest = inspect_backup(backup)
    assert manifest["includes_models"] is include_models
    assert len(manifest["omitted_model_weights"]) == (0 if include_models else 1)
    destination = tmp_path / "restored"
    result = restore_backup(backup, destination)
    assert result["workspace_root"] == str(destination)
    restored = CaseStore(destination)
    loaded = restored.load(case["id"])
    assert restored.image_path(loaded["sources"][0]).read_bytes() == store.image_path(case["sources"][0]).read_bytes()
    assert loaded["external_runs"][0]["artifact_relative"] == case["external_runs"][0]["artifact_relative"]
    assert loaded["external_runs"][0]["artifact_dir"].startswith(str(destination))
    with sqlite3.connect(restored.db) as con:
        assert con.execute("SELECT COUNT(*) FROM history").fetchone()[0] >= 2
    restored_jobs = JobStore(destination)
    completed = restored_jobs.result(job["id"])
    assert completed["input"].startswith(str(destination))
    assert completed["portable"] == "@artifact/evidence.txt"
    assert completed["outside_provenance"] == "Z:/deployment/original.png"
    assert restored_jobs.get(job["id"])["publication"]["state"] == "conflict"
    assert restored_jobs.get(job["id"])["payload"]["workspace_root"] == str(destination)
    restored_models = ModelRegistry(destination)
    assert restored_models.get(profile["id"], 1)["metadata"] == {"station_name": "A"}
    assert restored_models.get(profile["id"])["revision"] == 2
    assert restored_models.availability(profile["id"])["deployment_asset"]["available"] is include_models
    if not include_models:
        restored_models.restore_asset(asset["sha256"], b"model-fixture")
    assert restored_models.asset_path(asset).is_relative_to(destination)
    dataset = json.loads((destination / "datasets/dataset_example.json").read_text(encoding="utf-8"))
    assert dataset["rows"][0]["path"].startswith(str(destination))
    assert jobs.result(job["id"]) == before  # Restore never changes the original workspace.
    with pytest.raises(ValueError, match="fresh workspace"):
        restore_backup(backup, origin)


def test_live_jobs_missing_sources_and_corrupt_results_block_snapshot(tmp_path):
    store, case, registry, profile, asset, jobs, job = _workspace(tmp_path)
    queued = jobs.submit("infer", {}, "queued")
    with pytest.raises(ValueError, match="queued/running"):
        create_backup(tmp_path)
    jobs.cancel(queued["id"])
    evidence = jobs.artifact_dir(job["id"]) / "evidence.txt"
    evidence.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        create_backup(tmp_path)


def _rewrite(backup, *, member=None, replacement=None, update_manifest=False, extra=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(backup)) as old, zipfile.ZipFile(buffer, "w") as new:
        manifest = json.loads(old.read("manifest.json"))
        for name in old.namelist():
            if name == "manifest.json":
                continue
            data = replacement if name == member else old.read(name)
            new.writestr(name, data)
            if name == member and update_manifest:
                item = next(i for i in manifest["files"] if i["path"] == name)
                item.update(size=len(data), sha256=hashlib.sha256(data).hexdigest())
        if extra:
            info, content = extra
            new.writestr(info, content)
        new.writestr("manifest.json", json.dumps(manifest))
    return buffer.getvalue()


def test_checksums_and_atomic_sqlite_failure_leave_destination_untouched(tmp_path):
    origin = tmp_path / "source"
    _workspace(origin)
    backup = create_backup(origin)
    corrupted = _rewrite(backup, member="cases.sqlite3", replacement=b"not a database")
    with pytest.raises(ValueError):
        restore_backup(corrupted, tmp_path / "target")
    assert not (tmp_path / "target").exists()
    valid_digest_bad_db = _rewrite(backup, member="cases.sqlite3", replacement=b"not a database", update_manifest=True)
    with pytest.raises(sqlite3.DatabaseError):
        restore_backup(valid_digest_bad_db, tmp_path / "target")
    assert not (tmp_path / "target").exists()
    assert not list(tmp_path.glob(".lv-restore-*"))


@pytest.mark.parametrize("name", ["../outside.txt", "/absolute.txt", "assets/../../outside.txt", "assets\\escape.txt", "assets/C:drive.txt", "assets/CON.txt", "assets/trailing."])
def test_archive_paths_cannot_escape_or_use_windows_aliases(tmp_path, name):
    _workspace(tmp_path / "source")
    backup = create_backup(tmp_path / "source")
    bad = _rewrite(backup, extra=(name, b"bad"))
    with pytest.raises(ValueError):
        restore_backup(bad, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_links_duplicates_and_expanded_size_limits_are_rejected(tmp_path):
    _workspace(tmp_path / "source")
    backup = create_backup(tmp_path / "source")
    info = zipfile.ZipInfo("assets/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError, match="links"):
        inspect_backup(_rewrite(backup, extra=(info, b"../../outside")))
    with zipfile.ZipFile(io.BytesIO(backup)) as archive:
        asset_name = next(name for name in archive.namelist() if name.startswith("assets/"))
    with pytest.raises(ValueError, match="colliding"):
        inspect_backup(_rewrite(backup, extra=("assets/" + asset_name[7:].upper(), b"duplicate")))
    with pytest.raises(ValueError, match="limits"):
        inspect_backup(backup, max_bytes=10)
    destination = managed_restore_destination(tmp_path / "source")
    assert destination.parent == tmp_path and not destination.exists()
