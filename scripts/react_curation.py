"""Worker-owned durable storage for the VisualLatent curation cart."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class CurationCartError(Exception):
    """Raised when the app-owned curation cart cannot be read or saved."""


def curation_cart_path() -> Path:
    cache_root = Path(os.environ.get("LV_CACHE_DIR")
                      or (Path(__file__).resolve().parent.parent / ".lv_cache"))
    return cache_root / "curation_cart.json"


def curation_item_id(path: str | Path, object_index: int | None = None) -> str:
    resolved = str(Path(path).resolve())
    object_key = "" if object_index is None else str(int(object_index))
    return hashlib.sha256(f"{resolved}\0{object_key}".encode("utf-8")).hexdigest()


def _validate_items(items: Any) -> list[dict]:
    if not isinstance(items, list):
        raise CurationCartError("The curation cart format is invalid.")
    validated: list[dict] = []
    seen: set[str] = set()
    required = {"id", "index", "filename", "path", "label", "split", "sha256", "source", "score", "reason"}
    for item in items:
        if not isinstance(item, dict) or not required <= item.keys():
            raise CurationCartError("The curation cart contains an invalid item.")
        try:
            item_id = str(item["id"])
            path = str(item["path"])
            index = int(item["index"])
            object_index = item.get("object_index")
            object_index = None if object_index is None else int(object_index)
        except (TypeError, ValueError):
            raise CurationCartError("The curation cart contains an invalid item.") from None
        if (not Path(path).is_absolute() or index < 0 or not item_id
                or item_id != curation_item_id(path, object_index) or item_id in seen):
            raise CurationCartError("The curation cart contains an invalid identity.")
        sha256 = item.get("sha256")
        if sha256 is not None and (not isinstance(sha256, str) or len(sha256) != 64):
            raise CurationCartError("The curation cart contains an invalid content hash.")
        score = item.get("score")
        if score is not None:
            try:
                score = float(score)
            except (TypeError, ValueError):
                raise CurationCartError("The curation cart contains an invalid score.") from None
            if not math.isfinite(score):
                raise CurationCartError("The curation cart contains an invalid score.")
        for field in ("filename", "label", "split", "source", "reason"):
            if not isinstance(item.get(field), str):
                raise CurationCartError("The curation cart contains invalid metadata.")
        item["index"] = index
        item["path"] = path
        item["object_index"] = object_index
        item["score"] = score
        validated.append(item)
        seen.add(item_id)
    return validated


def load_curation_cart(path: Path | None = None) -> list[dict]:
    target = Path(path or curation_cart_path())
    if not target.exists():
        return []
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise CurationCartError("The saved curation cart could not be read.") from None
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise CurationCartError("The saved curation cart uses an unsupported format.")
    return _validate_items(document.get("items"))


def save_curation_cart(items: list[dict], path: Path | None = None) -> None:
    target = Path(path or curation_cart_path())
    validated = _validate_items(items)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=target.parent,
            prefix=target.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump({"schema_version": SCHEMA_VERSION, "items": validated},
                      stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    except (OSError, TypeError, ValueError):
        raise CurationCartError("The curation cart could not be saved to the app cache.") from None
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass
