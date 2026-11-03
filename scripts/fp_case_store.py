"""Small durable store for the React false-positive case intake journey."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


VERDICTS = {"pending", "normal", "defect", "disputed", "insufficient"}
SOURCE_KINDS = {"original", "screenshot", "reference"}
MAX_IMAGE_BYTES = 50 * 1024 * 1024
CASE_DEFAULTS = {
    "candidates": [],
    "searches": [],
    "claims": [],
    "batches": [],
    "reply_status": "draft",
    "cause_status": "unknown",
    "improvement_status": "not_evaluated",
    "effort_minutes": 0.0,
    "effort_log": [],
    "external_runs": [],
    "evaluations": [],
    "training_exports": [],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_name(name: str) -> str:
    value = str(name or "image").replace("\\", "/").split("/")[-1].strip()
    return (value or "image")[:240]


def checked_roi(roi: list[float] | tuple[float, ...]) -> list[float]:
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ValueError("區域須使用 normalized x1、y1、x2、y2 四個座標")
    values = [float(value) for value in roi]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError("區域座標必須是 0 到 1 之間的有限數值")
    x1, y1, x2, y2 = values
    if x1 >= x2 or y1 >= y2:
        raise ValueError("區域右下座標必須大於左上座標")
    return values


class FalsePositiveCaseStore:
    """SQLite revision snapshots plus immutable, content-hash named image copies."""

    def __init__(self, root: str | Path | None = None):
        default = Path(os.environ.get("LV_OUTPUT_DIR") or Path(__file__).resolve().parents[1] / "output") / "fp_cases"
        self.root = Path(root or os.environ.get("LV_CASES_DIR") or default).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets = self.root / "assets"
        self.assets.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "cases.sqlite3"
        with self._connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, body TEXT NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS history (id TEXT NOT NULL, revision INTEGER NOT NULL, body TEXT NOT NULL, PRIMARY KEY(id, revision))")

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.db, timeout=15)
        try:
            with con:
                yield con
        finally:
            con.close()

    def ingest(self, data: bytes, name: str, kind: str) -> dict:
        if kind not in SOURCE_KINDS:
            raise ValueError("影像來源類型必須是原圖或截圖")
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError("圖片必須介於 1 byte 與 50 MB")
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                width, height = image.size
                image_format = image.format
        except Exception as exc:
            raise ValueError("無法讀取圖片；請選擇 PNG、JPEG、BMP、TIFF 或 WebP") from exc
        extensions = {"JPEG": ".jpg", "PNG": ".png", "BMP": ".bmp", "TIFF": ".tif", "WEBP": ".webp"}
        extension = extensions.get(image_format or "")
        if extension is None:
            raise ValueError("支援 PNG、JPEG、BMP、TIFF、WebP 圖片")
        sha = digest(data)
        asset = sha + extension
        path = self.assets / asset
        if not path.exists():
            temporary = self.assets / (uuid.uuid4().hex + ".tmp")
            try:
                temporary.write_bytes(data)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return {
            "id": "s_" + uuid.uuid4().hex[:12],
            "sha256": sha,
            "asset": asset,
            "name": _safe_name(name),
            "kind": kind,
            "size": [width, height],
            "added_at": utc_now(),
        }

    def create(
        self,
        source: dict,
        title: str = "",
        description: str = "",
        model_id: str = "",
        threshold: str = "",
        raw_output: str = "",
    ) -> dict:
        now = utc_now()
        case = {
            "schema_version": 1,
            "id": "case_" + uuid.uuid4().hex[:16],
            "revision": 0,
            "created_at": now,
            "updated_at": now,
            "title": str(title).strip()[:240] or source["name"],
            "description": str(description)[:10000],
            "sources": [source],
            "items": [],
            "candidates": [],
            "searches": [],
            "claims": [],
            "batches": [],
            "external": {
                "model_id": str(model_id).strip()[:240],
                "threshold": str(threshold).strip()[:120],
                "raw_output": str(raw_output)[:20000],
                "replay_status": "not_replayed",
            },
            "reply_status": "draft",
            "cause_status": "unknown",
            "improvement_status": "not_evaluated",
            "effort_minutes": 0.0,
            "effort_log": [],
            "evidence": [{
                "id": "e_" + uuid.uuid4().hex[:12],
                "role": "reported_output",
                "text": "報案敘述與外部模型輸出（尚未重播）",
                "at": now,
                "source_id": source["id"],
            }],
        }
        return self.save(case)

    def save(self, case: dict) -> dict:
        value = copy.deepcopy(case)
        for key, default in CASE_DEFAULTS.items():
            value.setdefault(key, copy.deepcopy(default))
        old_revision = int(value["revision"])
        value["revision"] = old_revision + 1
        value["updated_at"] = utc_now()
        body = _json(value)
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            if old_revision == 0:
                con.execute("INSERT INTO cases VALUES (?, ?, ?)", (value["id"], 1, body))
            else:
                cursor = con.execute(
                    "UPDATE cases SET revision = ?, body = ? WHERE id = ? AND revision = ?",
                    (old_revision + 1, body, value["id"], old_revision),
                )
                if cursor.rowcount != 1:
                    raise ValueError("案件已在另一個視窗更新，請重新載入後再儲存")
            con.execute("INSERT INTO history VALUES (?, ?, ?)", (value["id"], old_revision + 1, body))
        return value

    def load(self, case_id: str) -> dict:
        with self._connect() as con:
            row = con.execute("SELECT body FROM cases WHERE id = ?", (case_id,)).fetchone()
        if row is None:
            raise KeyError("找不到案件")
        value = json.loads(row[0])
        for key, default in CASE_DEFAULTS.items():
            value.setdefault(key, copy.deepcopy(default))
        return value

    def list(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("SELECT body FROM cases ORDER BY rowid DESC").fetchall()
        values = [json.loads(row[0]) for row in rows]
        for value in values:
            for key, default in CASE_DEFAULTS.items():
                value.setdefault(key, copy.deepcopy(default))
        return values

    def image_path(self, source: dict) -> Path:
        asset = source.get("asset", "")
        if not asset or Path(asset).name != asset:
            raise ValueError("案件影像來源識別無效")
        path = self.assets / asset
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ValueError("案件影像副本不存在，請重新補入來源") from exc
        if digest(data) != source.get("sha256"):
            raise ValueError("案件影像副本指紋不符，請重新補入來源")
        return path

    def add_source(self, case: dict, source: dict) -> None:
        if any(row["sha256"] == source["sha256"] and row["kind"] == source["kind"] for row in case["sources"]):
            raise ValueError("此類型的影像來源已在案件中")
        case["sources"].append(source)
        case["evidence"].append({
            "id": "e_" + uuid.uuid4().hex[:12],
            "role": "source",
            "text": "補入影像來源",
            "at": utc_now(),
            "source_id": source["id"],
        })

    def add_region(
        self,
        case: dict,
        source_id: str,
        roi: list[float],
        label: str = "",
        score: float | None = None,
        prediction_id: str = "",
        origin: str = "manual",
    ) -> dict:
        source = next((row for row in case["sources"] if row["id"] == source_id), None)
        if source is None:
            raise ValueError("找不到指定影像來源")
        checked = checked_roi(roi)
        if score is not None and not math.isfinite(float(score)):
            raise ValueError("預測分數必須是有限數值")
        if any(row["source_id"] == source_id and row["roi"] == checked and row["label"] == label for row in case["items"]):
            raise ValueError("同一來源、區域與預測類別已存在")
        item = {
            "id": "i_" + uuid.uuid4().hex[:12],
            "source_id": source_id,
            "roi": checked,
            "label": str(label)[:256],
            "score": None if score is None else float(score),
            "prediction_id": str(prediction_id)[:256],
            "origin": str(origin),
            "review": {"verdict": "pending", "reviewer": "", "reason": "", "at": utc_now()},
        }
        item["evidence_id"] = "e_" + uuid.uuid4().hex[:12]
        case["items"].append(item)
        case["evidence"].append({
            "id": item["evidence_id"],
            "role": "reported_region",
            "text": "爭議區域／預測紀錄（未覆核）",
            "at": utc_now(),
            "item_id": item["id"],
            "source_id": source_id,
        })
        return item

    def import_yolo(self, case: dict, source_id: str, text: str) -> int:
        """Validate and stage all six-column rows before changing the case."""
        staged = copy.deepcopy(case)
        added = 0
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            cells = line.split()
            if len(cells) != 6:
                raise ValueError(f"第 {line_number} 行須包含 class cx cy width height confidence 六欄")
            try:
                class_id, cx, cy, width, height, confidence = map(float, cells)
            except ValueError as exc:
                raise ValueError(f"第 {line_number} 行含非數值欄位") from exc
            values = (class_id, cx, cy, width, height, confidence)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"第 {line_number} 行含非有限數值")
            if class_id < 0 or not class_id.is_integer() or width <= 0 or height <= 0 or not 0 <= confidence <= 1:
                raise ValueError(f"第 {line_number} 行的類別、尺寸或分數不合法")
            self.add_region(
                staged,
                source_id,
                [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
                label=str(int(class_id)),
                score=confidence,
                prediction_id=str(added),
                origin="yolo_prediction_file",
            )
            added += 1
        if not added:
            raise ValueError("預測檔沒有有效列")
        case.clear()
        case.update(staged)
        return added

    def review_region(self, case: dict, item_id: str, verdict: str, reviewer: str, reason: str) -> dict:
        if verdict not in VERDICTS:
            raise ValueError("未知的覆核結果")
        reviewer, reason = str(reviewer).strip(), str(reason).strip()
        if verdict != "pending" and (not reviewer or not reason):
            raise ValueError("非待覆核結果需要填寫覆核者與判斷依據")
        item = next((row for row in case["items"] if row["id"] == item_id), None)
        if item is None:
            raise ValueError("找不到覆核區域")
        item["review"] = {"verdict": verdict, "reviewer": reviewer, "reason": reason, "at": utc_now()}
        case["evidence"].append({
            "id": "e_" + uuid.uuid4().hex[:12],
            "role": "human_review",
            "text": "人工逐區域覆核",
            "at": utc_now(),
            "item_id": item_id,
            "verdict": verdict,
        })
        return item


def public_case(case: dict) -> dict:
    """Return case data without private asset names or filesystem locations."""
    private_fields = {"asset", "path", "source_path", "checkpoint_path", "model_path", "artifact_dir", "relative_path"}

    def public_value(value):
        if isinstance(value, dict):
            return {
                key: public_value(item)
                for key, item in value.items()
                if key not in private_fields
            }
        if isinstance(value, list):
            return [public_value(item) for item in value]
        return value

    return public_value(copy.deepcopy(case))
