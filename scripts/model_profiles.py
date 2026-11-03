"""Workspace-owned model bytes and immutable, portable model profile revisions.

Importing bytes does not inspect or execute a checkpoint. Capability snapshots
are supplied by the explicit inspection job, never inferred from an extension.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone

DEFAULT_MAX_BYTES = 1024 * 1024 * 1024


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _sha(value):
    value = str(value)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid model content identity")
    return value


def _atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ModelRegistry:
    def __init__(self, workspace_root):
        self.workspace_root = Path(workspace_root).resolve()
        self.root = self.workspace_root / "models"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "registry.sqlite"
        with self._connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS assets (sha TEXT PRIMARY KEY, body TEXT NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, name TEXT NOT NULL, archived INTEGER NOT NULL, revision INTEGER NOT NULL, created_at TEXT NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS revisions (id TEXT NOT NULL, revision INTEGER NOT NULL, body TEXT NOT NULL, PRIMARY KEY(id,revision))")

    def _connect(self):
        con = sqlite3.connect(self.db, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def ingest(self, data, name, *, max_bytes=DEFAULT_MAX_BYTES):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValueError("Upload must contain bytes")
        if not data or len(data) > max_bytes:
            raise ValueError(f"Model upload must be nonempty and at most {max_bytes} bytes")
        name = str(name).replace("\\", "/").split("/")[-1]
        if not name or any(ord(c) < 32 for c in name):
            raise ValueError("Invalid model filename")
        suffix = Path(name).suffix.lower()
        if suffix not in {".pt", ".onnx"}:
            raise ValueError("Supported model uploads: .pt and .onnx")
        data = bytes(data)
        sha = _digest(data)
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            existing = con.execute("SELECT body FROM assets WHERE sha=?", (sha,)).fetchone()
            if existing:
                asset = json.loads(existing[0])
                path = self.root / asset["relative_path"]
                if not path.exists() or _digest(path.read_bytes()) != sha:
                    _atomic(path, data)
                return asset
            asset = {"sha256": sha, "name": name, "format": suffix[1:], "size": len(data),
                     "relative_path": f"assets/{sha}{suffix}", "created_at": _now(), "inspection_status": "not_inspected"}
            _atomic(self.root / asset["relative_path"], data)
            con.execute("INSERT INTO assets VALUES (?,?)", (sha, _json(asset)))
        return asset

    def import_path(self, path, *, max_bytes=DEFAULT_MAX_BYTES):
        path = Path(path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size > max_bytes:
            raise ValueError("Model path is unavailable or exceeds the upload limit")
        with path.open("rb") as stream:
            data = stream.read(max_bytes + 1)
        return self.ingest(data, path.name, max_bytes=max_bytes)

    def asset(self, asset_or_sha):
        sha = _sha(asset_or_sha["sha256"] if isinstance(asset_or_sha, dict) else asset_or_sha)
        with self._connect() as con:
            row = con.execute("SELECT body FROM assets WHERE sha=?", (sha,)).fetchone()
        if not row:
            raise ValueError("Unknown model asset")
        return json.loads(row[0])

    def asset_path(self, asset_or_sha, *, verify=True):
        asset = self.asset(asset_or_sha)
        path = (self.root / asset["relative_path"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Invalid model asset path")
        if not path.is_file():
            raise ValueError("Model file is missing; reattach the original file")
        if verify and _digest(path.read_bytes()) != asset["sha256"]:
            raise ValueError("Model file failed integrity verification; reattach the original file")
        return path

    def restore_asset(self, sha, data):
        asset = self.asset(sha)
        if _digest(data) != asset["sha256"]:
            raise ValueError("Reattached model does not match the required content identity")
        _atomic(self.root / asset["relative_path"], data)
        return asset

    def _snapshot(self, profile_id, revision, deployment_asset, gradient_asset, runtime_config, capabilities, metadata):
        deployment = self.asset(deployment_asset) if deployment_asset else None
        gradient = self.asset(gradient_asset) if gradient_asset else None
        if not deployment and not gradient:
            raise ValueError("A model profile requires a deployment or gradient asset")
        config = dict(runtime_config or {})
        from model_runtimes import runtime_id
        runtime_id(config)
        if (metadata or {}).get("gradient_config"):
            runtime_id(metadata["gradient_config"])
        config.pop("model_path", None)  # A portable revision never retains a server path.
        value = {"schema_version": 1, "id": profile_id, "revision": revision, "created_at": _now(),
                 "deployment_asset": deployment, "gradient_asset": gradient, "runtime_config": config,
                 "capabilities": capabilities or {}, "metadata": metadata or {}}
        # Validate and detach nested mutable caller objects.
        return json.loads(_json(value))

    def create(self, name, deployment_asset=None, gradient_asset=None, runtime_config=None, capabilities=None, metadata=None):
        if not str(name).strip():
            raise ValueError("A model name is required")
        profile_id = "model_" + uuid.uuid4().hex[:16]
        value = self._snapshot(profile_id, 1, deployment_asset, gradient_asset, runtime_config, capabilities, metadata)
        with self._connect() as con:
            con.execute("INSERT INTO profiles VALUES (?,?,0,1,?)", (profile_id, str(name).strip(), value["created_at"]))
            con.execute("INSERT INTO revisions VALUES (?,?,?)", (profile_id, 1, _json(value)))
        return self.get(profile_id)

    def revise(self, profile_id, *, expected_revision=None, **changes):
        allowed = {"deployment_asset", "gradient_asset", "runtime_config", "capabilities", "metadata"}
        if set(changes) - allowed:
            raise ValueError("Unsupported profile revision fields")
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT revision FROM profiles WHERE id=?", (profile_id,)).fetchone()
            if not row:
                raise ValueError("Unknown model profile")
            current = row[0]
            if expected_revision is not None and current != expected_revision:
                raise ValueError("Model changed in another session; reload before saving a revision")
            old = json.loads(con.execute("SELECT body FROM revisions WHERE id=? AND revision=?", (profile_id, current)).fetchone()[0])
            fields = {key: changes.get(key, old[key]) for key in allowed}
            value = self._snapshot(profile_id, current + 1, **fields)
            con.execute("INSERT INTO revisions VALUES (?,?,?)", (profile_id, current + 1, _json(value)))
            con.execute("UPDATE profiles SET revision=? WHERE id=?", (current + 1, profile_id))
        return self.get(profile_id)

    def get(self, profile_id, revision=None):
        with self._connect() as con:
            row = con.execute("SELECT * FROM profiles WHERE id=?", (profile_id,)).fetchone()
            if not row:
                raise ValueError("Unknown model profile")
            snap = con.execute("SELECT body FROM revisions WHERE id=? AND revision=?", (profile_id, revision or row["revision"])).fetchone()
        if not snap:
            raise ValueError("Unknown model revision")
        value = json.loads(snap[0])
        value.update(name=row["name"], archived=bool(row["archived"]), latest_revision=row["revision"])
        return value

    def list(self, *, include_archived=False):
        with self._connect() as con:
            rows = con.execute("SELECT id FROM profiles " + ("" if include_archived else "WHERE archived=0 ") + "ORDER BY created_at DESC").fetchall()
        return [self.get(row[0]) for row in rows]

    def rename(self, profile_id, name):
        if not str(name).strip():
            raise ValueError("A model name is required")
        self.get(profile_id)
        with self._connect() as con:
            con.execute("UPDATE profiles SET name=? WHERE id=?", (str(name).strip(), profile_id))
        return self.get(profile_id)

    def archive(self, profile_id):
        self.get(profile_id)
        with self._connect() as con:
            con.execute("UPDATE profiles SET archived=1 WHERE id=?", (profile_id,))
        return self.get(profile_id)

    def restore(self, profile_id):
        self.get(profile_id)
        with self._connect() as con:
            con.execute("UPDATE profiles SET archived=0 WHERE id=?", (profile_id,))
        return self.get(profile_id)

    def runtime_config(self, profile_id, revision=None):
        profile = self.get(profile_id, revision)
        asset = profile["deployment_asset"] or profile["gradient_asset"]
        return {**profile["runtime_config"], "model_path": str(self.asset_path(asset))}

    def availability(self, profile_id, revision=None):
        profile = self.get(profile_id, revision)
        result = {}
        for role in ("deployment_asset", "gradient_asset"):
            asset = profile[role]
            try:
                if asset:
                    self.asset_path(asset)
                result[role] = {"available": bool(asset), "reason": "" if asset else "not_attached"}
            except ValueError as exc:
                result[role] = {"available": False, "reason": str(exc)}
        return result

    def backup(self, *, include_models=False):
        """Portable registry ZIP. Omitting weights explicitly lists missing assets."""
        with self._connect() as con:
            manifest = {"schema_version": 1, "includes_models": bool(include_models),
                        "profiles": [dict(r) for r in con.execute("SELECT * FROM profiles")],
                        "revisions": [dict(r) for r in con.execute("SELECT * FROM revisions")],
                        "assets": [json.loads(r[0]) for r in con.execute("SELECT body FROM assets")]}
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", _json(manifest))
            if include_models:
                for asset in manifest["assets"]:
                    archive.write(self.asset_path(asset), asset["relative_path"])
        return output.getvalue()

    def restore_backup(self, data, *, max_bytes=4 * DEFAULT_MAX_BYTES):
        """Merge a verified backup without replacing existing conflicting records."""
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sum(info.file_size for info in archive.infolist()) > max_bytes:
                raise ValueError("Backup exceeds expanded size limit")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("schema_version") != 1:
                raise ValueError("Unsupported model backup schema")
            verified = []
            for asset in manifest["assets"]:
                sha = _sha(asset["sha256"])
                expected = f"assets/{sha}.{asset['format']}"
                if asset["format"] not in {"pt", "onnx"} or asset["relative_path"] != expected:
                    raise ValueError("Invalid backup asset path")
                content = archive.read(expected) if manifest["includes_models"] else None
                if content is not None and (_digest(content) != sha or len(content) != asset["size"]):
                    raise ValueError("Backup model integrity mismatch")
                verified.append((asset, content))
            with self._connect() as con:
                con.execute("BEGIN IMMEDIATE")
                for record in manifest["profiles"]:
                    old = con.execute("SELECT * FROM profiles WHERE id=?", (record["id"],)).fetchone()
                    if old and dict(old) != record:
                        raise ValueError("Existing model profile conflicts with backup")
                for record in manifest["revisions"]:
                    body = json.loads(record["body"])
                    if body["id"] != record["id"] or body["revision"] != record["revision"]:
                        raise ValueError("Invalid backup revision")
                    old = con.execute("SELECT body FROM revisions WHERE id=? AND revision=?", (record["id"], record["revision"])).fetchone()
                    if old and json.loads(old[0]) != body:
                        raise ValueError("Existing model revision conflicts with backup")
                for asset, content in verified:
                    if content is not None:
                        _atomic(self.root / asset["relative_path"], content)
                    con.execute("INSERT OR IGNORE INTO assets VALUES (?,?)", (asset["sha256"], _json(asset)))
                for r in manifest["profiles"]:
                    con.execute("INSERT OR IGNORE INTO profiles VALUES (?,?,?,?,?)", (r["id"], r["name"], r["archived"], r["revision"], r["created_at"]))
                for r in manifest["revisions"]:
                    con.execute("INSERT OR IGNORE INTO revisions VALUES (?,?,?)", (r["id"], r["revision"], r["body"]))
        return {"profiles": len(manifest["profiles"]), "assets": len(verified), "includes_models": manifest["includes_models"]}
