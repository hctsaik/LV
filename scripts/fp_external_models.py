"""Content-addressed ONNX model storage and immutable inference profiles."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fp_case_store import digest
from fp_external_runtime import config_checked

MAX_MODEL_BYTES = 1024 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _safe_filename(value: str) -> str:
    name = str(value or "").replace("\\", "/").split("/")[-1].strip()
    if not name or any(ord(char) < 32 for char in name):
        raise ValueError("模型檔名無效")
    if Path(name).suffix.lower() != ".onnx":
        raise ValueError("外部模型目前只接受單檔 ONNX（.onnx）")
    return name[:240]


def _digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class ExternalModelRegistry:
    """Store uploaded model bytes by SHA-256 and immutable profile revisions."""

    def __init__(self, workspace_root: str | Path):
        self.root = (Path(workspace_root).resolve() / "external_models")
        self.assets = self.root / "assets"
        self.assets.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "registry.sqlite3"
        with self._connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS assets (sha TEXT PRIMARY KEY, body TEXT NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, created_at TEXT NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS revisions (id TEXT NOT NULL, revision INTEGER NOT NULL, body TEXT NOT NULL, PRIMARY KEY(id, revision))")

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.db, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    def ingest_file(self, staged_path: str | Path, filename: str, *, max_bytes: int = MAX_MODEL_BYTES) -> dict:
        name = _safe_filename(filename)
        staged = Path(staged_path).resolve()
        if not staged.is_file():
            raise ValueError("模型上傳內容不存在")
        if staged.parent != self.assets.resolve():
            raise ValueError("模型暫存位置無效")
        size = staged.stat().st_size
        if size <= 0 or size > max_bytes:
            raise ValueError(f"模型檔案須介於 1 byte 與 {max_bytes} bytes")
        sha = hashlib.sha256()
        with staged.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(chunk)
        content_sha = sha.hexdigest()
        destination = self.assets / f"{content_sha}.onnx"
        if destination.exists():
            if _digest_file(destination) != content_sha:
                raise ValueError("已保存的模型內容指紋不符")
            staged.unlink(missing_ok=True)
        else:
            # The API creates the temporary file under this same directory so
            # replace is atomic and never exposes a partial model asset.
            os.replace(staged, destination)
        asset = {
            "sha256": content_sha,
            "name": name,
            "format": "onnx",
            "size": size,
            "relative_path": f"assets/{content_sha}.onnx",
            "created_at": _now(),
        }
        with self._connect() as con:
            con.execute("INSERT OR IGNORE INTO assets VALUES (?,?)", (content_sha, _json(asset)))
            row = con.execute("SELECT body FROM assets WHERE sha=?", (content_sha,)).fetchone()
        return json.loads(row[0])

    def ingest(self, data: bytes, filename: str, *, max_bytes: int = MAX_MODEL_BYTES) -> dict:
        """Convenience entry point for small fixtures and direct API tests."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValueError("模型上傳內容必須是位元組")
        name = _safe_filename(filename)
        if not data or len(data) > max_bytes:
            raise ValueError(f"模型檔案須介於 1 byte 與 {max_bytes} bytes")
        fd, temporary = tempfile.mkstemp(prefix=".upload-", suffix=".tmp", dir=self.assets)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
            return self.ingest_file(temporary, name, max_bytes=max_bytes)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def asset(self, sha: str) -> dict:
        if len(str(sha)) != 64 or any(char not in "0123456789abcdef" for char in str(sha)):
            raise ValueError("模型內容指紋格式無效")
        with self._connect() as con:
            row = con.execute("SELECT body FROM assets WHERE sha=?", (sha,)).fetchone()
        if row is None:
            raise ValueError("找不到已匯入的模型內容")
        return json.loads(row[0])

    def asset_path(self, sha: str, *, verify: bool = True) -> Path:
        asset = self.asset(sha)
        path = (self.root / asset["relative_path"]).resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_file():
            raise ValueError("模型檔案不存在，請重新匯入原始 ONNX")
        if verify and _digest_file(path) != asset["sha256"]:
            raise ValueError("模型檔案完整性檢查失敗，請重新匯入原始 ONNX")
        return path

    @staticmethod
    def _profile_config(config: dict) -> dict:
        if not isinstance(config, dict) or "model_path" in config:
            raise ValueError("模型設定不得指定本機路徑；請匯入 ONNX 檔案")
        validated = config_checked({**copy.deepcopy(config), "model_path": "internal-content-addressed.onnx"})
        validated.pop("model_path", None)
        return validated

    def create(self, name: str, asset_sha256: str, config: dict) -> dict:
        label = str(name).strip()[:240]
        if not label:
            raise ValueError("請輸入模型設定名稱")
        asset = self.asset(asset_sha256)
        self.asset_path(asset_sha256)
        profile_id = "onnx_" + uuid.uuid4().hex[:16]
        body = {
            "schema_version": 1,
            "id": profile_id,
            "revision": 1,
            "name": label,
            "created_at": _now(),
            "asset": {key: asset[key] for key in ("sha256", "name", "format", "size")},
            "config": self._profile_config(config),
        }
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute("INSERT INTO profiles VALUES (?,?,?)", (profile_id, 1, body["created_at"]))
            con.execute("INSERT INTO revisions VALUES (?,?,?)", (profile_id, 1, _json(body)))
        return self.get(profile_id)

    def get(self, profile_id: str, revision: int | None = None) -> dict:
        with self._connect() as con:
            profile = con.execute("SELECT revision FROM profiles WHERE id=?", (profile_id,)).fetchone()
            if profile is None:
                raise ValueError("找不到外部模型設定")
            selected = revision or int(profile[0])
            row = con.execute("SELECT body FROM revisions WHERE id=? AND revision=?", (profile_id, selected)).fetchone()
        if row is None:
            raise ValueError("找不到外部模型設定版本")
        body = json.loads(row[0])
        body["latest_revision"] = int(profile[0])
        return body

    def list(self) -> list[dict]:
        with self._connect() as con:
            ids = con.execute("SELECT id FROM profiles ORDER BY created_at DESC").fetchall()
        return [self.get(row[0]) for row in ids]

    def revise(
        self,
        profile_id: str,
        expected_revision: int,
        *,
        name: str | None = None,
        asset_sha256: str | None = None,
        config: dict | None = None,
    ) -> dict:
        current = self.get(profile_id)
        if current["revision"] != expected_revision:
            raise ValueError("模型設定已在另一個視窗更新，請重新載入")
        next_name = current["name"] if name is None else str(name).strip()[:240]
        if not next_name:
            raise ValueError("請輸入模型設定名稱")
        sha = asset_sha256 or current["asset"]["sha256"]
        asset = self.asset(sha)
        self.asset_path(sha)
        body = {
            **current,
            "revision": expected_revision + 1,
            "name": next_name,
            "asset": {key: asset[key] for key in ("sha256", "name", "format", "size")},
            "config": self._profile_config(config if config is not None else current["config"]),
        }
        body.pop("latest_revision", None)
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT revision FROM profiles WHERE id=?", (profile_id,)).fetchone()
            if row is None or int(row[0]) != expected_revision:
                raise ValueError("模型設定已在另一個視窗更新，請重新載入")
            con.execute("INSERT INTO revisions VALUES (?,?,?)", (profile_id, expected_revision + 1, _json(body)))
            con.execute("UPDATE profiles SET revision=? WHERE id=?", (expected_revision + 1, profile_id))
        body["latest_revision"] = expected_revision + 1
        return body

    def runtime_config(self, profile_id: str, revision: int | None = None) -> dict:
        profile = self.get(profile_id, revision)
        path = self.asset_path(profile["asset"]["sha256"])
        return {**profile["config"], "model_path": str(path)}
