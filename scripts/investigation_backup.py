"""Checksummed, portable investigation workspace snapshots.

Restoration is deliberately into a fresh directory, never a merge over an active
workspace. SQLite online backups preserve committed cases/history, model revisions
and job records. Live jobs must be completed/cancelled before creating a snapshot.
Checksums detect damage; they are not signatures proving a backup's author.
"""
from __future__ import annotations

from contextlib import ExitStack, closing
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import stat
import tempfile
import uuid
import zipfile

MAX_EXPANDED_BYTES = 8 * 1024**3
MAX_FILES = 100000
DB_PATHS = ("cases.sqlite3", "models/registry.sqlite", "investigation_jobs/jobs.sqlite")
INCLUDED_DIRS = {"assets", "datasets", "models", "investigation_jobs", "external"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _safe_name(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Backup contains an invalid archive path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value:
        raise ValueError("Backup contains a noncanonical archive path")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    for part in path.parts:
        if part in {".", ".."} or ":" in part or part.endswith((".", " ")) or part.split(".")[0].upper() in reserved:
            raise ValueError("Backup contains an unsafe archive path")
        if any(ord(c) < 32 for c in part):
            raise ValueError("Backup contains an invalid archive path")
    if value != "manifest.json" and value not in DB_PATHS and path.parts[0] not in INCLUDED_DIRS:
        raise ValueError("Backup contains files outside the investigation workspace")
    return value


def _locks_and_snapshots(root, temporary, stack):
    locks = {}
    for relative in DB_PATHS:
        source = root / relative
        if source.exists():
            if source.is_symlink():
                raise ValueError("Workspace database symlinks are not supported")
            lock = sqlite3.connect(source, timeout=5)
            stack.callback(lock.close)
            stack.enter_context(lock)
            # Reserve the writer slot; a separate reader performs SQLite backup.
            lock.execute("BEGIN IMMEDIATE")
            locks[relative] = lock
    jobs = locks.get("investigation_jobs/jobs.sqlite")
    if jobs and jobs.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]:
        raise ValueError("Finish or cancel queued/running investigation jobs before backing up")
    snapshots = {}
    for relative in locks:
        destination = temporary / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(root / relative)) as source, closing(sqlite3.connect(destination)) as target:
            source.backup(target)
        snapshots[relative] = destination
    return snapshots


def _verify_referenced_assets(root, snapshots):
    verified = {}
    def verify(relative, sha):
        _safe_name(relative)
        if relative not in verified:
            path = root / relative
            if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
                raise ValueError("A referenced source or completed artifact is missing")
            verified[relative] = _sha(path.read_bytes())
        if verified[relative] != sha:
            raise ValueError("A referenced source or completed artifact failed integrity verification")
    def visit(value):
        if isinstance(value, dict):
            if "asset" in value and "sha256" in value:
                verify("assets/" + value["asset"], value["sha256"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for relative, tables in (("cases.sqlite3", ("cases", "history")), ("investigation_jobs/jobs.sqlite", ("jobs",))):
        if relative not in snapshots:
            continue
        with closing(sqlite3.connect(snapshots[relative])) as con:
            for table in tables:
                for row in con.execute(f"SELECT body FROM {table}"):
                    value = json.loads(row[0])
                    visit(value)
                    if table == "jobs" and value.get("state") == "succeeded":
                        for artifact in value.get("artifacts", []):
                            verify(f"investigation_jobs/{value['id']}/artifacts/{artifact['path']}", artifact["sha256"])
    for dataset in (root / "datasets").glob("*.json"):
        visit(json.loads(dataset.read_text(encoding="utf-8")))


def create_backup(workspace_root, *, include_models=False, max_bytes=MAX_EXPANDED_BYTES):
    root = Path(workspace_root).resolve()
    if not (root / "cases.sqlite3").is_file():
        raise ValueError("Select an existing investigation workspace")
    buffer = io.BytesIO()
    with tempfile.TemporaryDirectory(prefix="lv-snapshot-") as temporary, ExitStack() as stack:
        snapshots = _locks_and_snapshots(root, Path(temporary), stack)
        _verify_referenced_assets(root, snapshots)
        omitted = []
        model_db = snapshots.get("models/registry.sqlite")
        if model_db:
            with closing(sqlite3.connect(model_db)) as con:
                assets = [json.loads(row[0]) for row in con.execute("SELECT body FROM assets")]
            for asset in assets:
                relative = _safe_name("models/" + asset["relative_path"])
                if include_models:
                    path = root / relative
                    if path.is_symlink() or not path.is_file() or _sha(path.read_bytes()) != asset["sha256"]:
                        raise ValueError("A model weight file is missing or corrupted; restore it or omit model weights")
                else:
                    omitted.append({"path": relative, "sha256": asset["sha256"], "size": asset["size"]})
        files = dict(snapshots)
        for directory in INCLUDED_DIRS:
            base = root / directory
            if not base.exists():
                continue
            if base.is_symlink() or not base.resolve().is_relative_to(root):
                raise ValueError("Workspace directory links are not supported by portable backup")
            for path in base.rglob("*"):
                relative = path.relative_to(root).as_posix()
                if "pending" in path.relative_to(root).parts or path.name.startswith(".pending-"):
                    continue
                if not include_models and relative.startswith("models/assets/"):
                    continue
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    raise ValueError("Workspace symlinks are not supported by portable backup")
                if not path.is_file() or relative in DB_PATHS or path.name.endswith(("-wal", "-shm", "-journal")):
                    continue
                _safe_name(relative)
                files[relative] = path
        if len(files) > MAX_FILES or sum(path.stat().st_size for path in files.values()) > max_bytes:
            raise ValueError("Workspace backup exceeds configured file/size limits")
        entries = []
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for relative, path in sorted(files.items()):
                data = path.read_bytes()
                archive.writestr(relative, data)
                entries.append({"path": relative, "size": len(data), "sha256": _sha(data)})
            manifest = {"schema_version": 1, "kind": "lv_investigation_workspace", "created_at": datetime.now(timezone.utc).isoformat(),
                        "origin_workspace": str(root), "includes_models": bool(include_models), "omitted_model_weights": omitted,
                        "files": entries, "excluded": ["recomputable embedding caches", "unfinished job staging files", "SQLite journal/WAL sidecars", "files outside the workspace (original links remain provenance)"],
                        "restore_policy": "fresh_workspace_only"}
            archive.writestr("manifest.json", _json(manifest))
    return buffer.getvalue()


def inspect_backup(data, *, max_bytes=MAX_EXPANDED_BYTES):
    """Validate the archive envelope and every checksum before creating a target."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_FILES + 1 or sum(info.file_size for info in infos) > max_bytes:
            raise ValueError("Backup exceeds expanded file/size limits")
        names = []
        for info in infos:
            _safe_name(info.filename)
            mode = info.external_attr >> 16
            if info.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG}):
                raise ValueError("Backup links and special files are not supported")
            names.append(info.filename)
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("Backup contains duplicate or case-colliding paths")
        if "manifest.json" not in names or archive.getinfo("manifest.json").file_size > 10 * 1024**2:
            raise ValueError("Backup manifest is missing or too large")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version") != 1 or manifest.get("kind") != "lv_investigation_workspace":
            raise ValueError("Unsupported workspace backup schema")
        if not isinstance(manifest.get("origin_workspace"), str) or not isinstance(manifest.get("includes_models"), bool):
            raise ValueError("Invalid backup workspace metadata")
        entries = manifest.get("files", [])
        if not isinstance(entries, list) or len(entries) > MAX_FILES:
            raise ValueError("Invalid backup manifest files")
        expected = []
        for entry in entries:
            name = _safe_name(entry["path"])
            if name == "manifest.json":
                raise ValueError("Manifest cannot include itself")
            expected.append(name)
            if name not in names or archive.getinfo(name).file_size != entry["size"]:
                raise ValueError("Backup inventory does not match its manifest")
            digest = hashlib.sha256()
            count = 0
            with archive.open(name) as stream:
                while block := stream.read(1024 * 1024):
                    count += len(block)
                    if count > entry["size"]:
                        raise ValueError("Backup member exceeds declared size")
                    digest.update(block)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError("Backup checksum verification failed")
        if len(set(expected)) != len(expected) or set(names) != set(expected) | {"manifest.json"}:
            raise ValueError("Backup contains unlisted or duplicated manifest files")
        if "cases.sqlite3" not in expected:
            raise ValueError("Backup does not contain its cases database")
    return manifest


def _relocate(value, origin, destination):
    if isinstance(value, dict):
        return {key: _relocate(item, origin, destination) for key, item in value.items()}
    if isinstance(value, list):
        return [_relocate(item, origin, destination) for item in value]
    if isinstance(value, str) and origin:
        normalized, base = value.replace("\\", "/"), origin.replace("\\", "/").rstrip("/")
        left, right = (normalized.casefold(), base.casefold()) if len(base) > 1 and base[1] == ":" else (normalized, base)
        if left == right:
            return str(destination)
        if left.startswith(right + "/"):
            suffix = normalized[len(base) + 1:]
            if ".." in PurePosixPath(suffix).parts:
                raise ValueError("Backup contains an unsafe owned path reference")
            return str(destination / suffix)
    return value


def _restore_portability(staging, destination, manifest):
    origin = manifest.get("origin_workspace", "")
    # Rewrite owned paths in JSON artifacts; @artifact/ and relative paths stay intact.
    for path in staging.rglob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        moved = _relocate(value, origin, destination)
        if moved != value:
            path.write_text(_json(moved), encoding="utf-8")
    for relative, tables in (("cases.sqlite3", ("cases", "history")), ("models/registry.sqlite", ("assets", "revisions")),
                             ("investigation_jobs/jobs.sqlite", ("jobs",))):
        path = staging / relative
        if not path.exists():
            continue
        with closing(sqlite3.connect(path)) as con, con:
            if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Backup SQLite database is corrupt")
            known = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in tables:
                if table not in known:
                    raise ValueError("Backup is missing a required workspace table")
                for rowid, body in con.execute(f"SELECT rowid, body FROM {table}").fetchall():
                    value = json.loads(body)
                    moved = _relocate(value, origin, destination)
                    if table == "jobs":
                        if moved["state"] in {"queued", "running"}:
                            moved.update(state="interrupted", stage="interrupted", error="Restored unfinished job; retry in the restored workspace")
                        moved["owner_pid"] = moved["worker_pid"] = None
                        moved["request_hash"] = _sha(_json({"task": moved["task"], "payload": moved["payload"], "context": moved.get("context", {})}).encode())
                        for artifact in moved.get("artifacts", []):
                            name = _safe_name(f"investigation_jobs/{moved['id']}/artifacts/{artifact['path']}")
                            artifact_path = staging / name
                            if not artifact_path.is_file():
                                raise ValueError("Backup is missing a referenced completed job artifact")
                            artifact.update(size=artifact_path.stat().st_size, sha256=_sha(artifact_path.read_bytes()))
                        con.execute("UPDATE jobs SET state=?,body=? WHERE rowid=?", (moved["state"], _json(moved), rowid))
                    elif moved != value:
                        con.execute(f"UPDATE {table} SET body=? WHERE rowid=?", (_json(moved), rowid))


def restore_backup(data, destination, *, max_bytes=MAX_EXPANDED_BYTES):
    """Restore only to a nonexistent or empty directory; publish after validation."""
    requested = Path(destination).expanduser()
    if requested.is_symlink():
        raise ValueError("Restore destination cannot be a symlink")
    destination = requested.resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Restore requires a fresh workspace; existing data will not be overwritten")
    manifest = inspect_backup(data, max_bytes=max_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".lv-restore-", dir=destination.parent)).resolve()
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for entry in manifest["files"]:
                path = staging / entry["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry["path"]) as source, path.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        snapshots = {relative: staging / relative for relative in DB_PATHS if (staging / relative).exists()}
        _verify_referenced_assets(staging, snapshots)
        if manifest["includes_models"] and "models/registry.sqlite" in snapshots:
            with closing(sqlite3.connect(snapshots["models/registry.sqlite"])) as con:
                for row in con.execute("SELECT body FROM assets"):
                    asset = json.loads(row[0])
                    path = staging / _safe_name("models/" + asset["relative_path"])
                    if not path.is_file() or _sha(path.read_bytes()) != asset["sha256"]:
                        raise ValueError("Restored model does not match its recorded content identity")
        _restore_portability(staging, destination, manifest)
        # A relocation receipt describes why local JSON checksums may differ.
        receipt = {"schema_version": 1, "restored_at": datetime.now(timezone.utc).isoformat(),
                   "original_manifest_sha256": _sha(_json(manifest).encode()), "includes_models": manifest["includes_models"],
                   "omitted_model_weights": manifest.get("omitted_model_weights", []), "paths_relocated": True}
        (staging / "restore_receipt.json").write_text(_json(receipt), encoding="utf-8")
        if destination.exists():
            destination.rmdir()  # Only the verified empty destination; no recursive deletion.
        os.replace(staging, destination)
    finally:
        if staging.exists():
            if staging.parent != destination.parent or not staging.name.startswith(".lv-restore-"):
                raise ValueError("Unexpected restore staging directory")
            shutil.rmtree(staging)
    return {"workspace_root": str(destination), "files": len(manifest["files"]), "includes_models": manifest["includes_models"],
            "omitted_model_weights": manifest.get("omitted_model_weights", [])}


def managed_restore_destination(workspace_root):
    """UI-selected fresh sibling location; the user does not enter a server path."""
    root = Path(workspace_root).resolve()
    return root.parent / (root.name + "_restored_" + uuid.uuid4().hex[:10])
