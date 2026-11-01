"""Durable, app-local store for the React curation cart.

The Streamlit reference keeps its cart in session state. The React worker
stores append-only events under ``LV_CACHE_DIR`` so a restarted worker can
rebuild the same cart without writing into image datasets.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


class CurationStoreError(RuntimeError):
    """The local curation event log cannot be read or updated safely."""


def default_curation_log_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    cache_root = Path(os.environ.get("LV_CACHE_DIR") or (repo_root / ".lv_cache"))
    return cache_root.expanduser().resolve() / "curation_log.jsonl"


def _item_id(item: dict) -> str:
    try:
        path = Path(item["path"]).expanduser().resolve(strict=False)
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError("A curation item must include a valid source path.") from exc
    identity = os.path.normcase(str(path))
    if item.get("level") == "object":
        obj_index = item.get("obj_index")
        if not isinstance(obj_index, int) or obj_index < 0:
            raise ValueError("An object-level curation item must include its object index.")
        identity += f"\0object:{obj_index}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


class CurationStore:
    """A path-keyed cart backed by one append-only JSONL event file."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else default_curation_log_path()
        self._lock = threading.RLock()

    def _read_events(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise CurationStoreError("The curation event log could not be read.") from exc
        events = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CurationStoreError(
                    f"The curation event log has invalid JSON on line {line_number}."
                ) from exc
            if not isinstance(event, dict) or event.get("event") not in {
                "add", "remove", "clear", "reason",
            }:
                raise CurationStoreError(
                    f"The curation event log has an unknown event on line {line_number}."
                )
            events.append(event)
        return events

    def _fold(self, events: Iterable[dict]) -> list[dict]:
        items: dict[str, dict] = {}
        for event in events:
            event_type = event["event"]
            if event_type == "add":
                item = event.get("item")
                item_id = event.get("id")
                if not isinstance(item, dict) or not isinstance(item_id, str):
                    raise CurationStoreError("The curation event log contains an invalid add event.")
                items[item_id] = item
            elif event_type == "remove":
                items.pop(str(event.get("id", "")), None)
            elif event_type == "clear":
                items.clear()
            elif event_type == "reason":
                reason = event.get("reason", "")
                for item_id in event.get("ids", []):
                    if item_id in items:
                        items[item_id]["reason"] = reason
        return list(items.values())

    def list_items(self) -> list[dict]:
        with self._lock:
            return self._fold(self._read_events())

    def _append(self, event: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, TypeError, ValueError) as exc:
            raise CurationStoreError("The curation event log could not be updated safely.") from exc

    def add(self, entries: Iterable[dict]) -> tuple[int, int]:
        with self._lock:
            current = {item["id"] for item in self._fold(self._read_events())}
            added = skipped = 0
            for entry in entries:
                item = dict(entry)
                item_id = _item_id(item)
                if item_id in current:
                    skipped += 1
                    continue
                item["id"] = item_id
                item.setdefault("reason", "")
                item.setdefault("added_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
                self._append({"event": "add", "id": item_id, "item": item})
                current.add(item_id)
                added += 1
            return added, skipped

    def remove(self, item_id: str) -> bool:
        with self._lock:
            current = {item["id"] for item in self._fold(self._read_events())}
            if item_id not in current:
                return False
            self._append({"event": "remove", "id": item_id})
            return True

    def set_reason(self, item_ids: Iterable[str], reason: str) -> int:
        with self._lock:
            current = {item["id"] for item in self._fold(self._read_events())}
            ids = list(dict.fromkeys(item_id for item_id in item_ids if item_id in current))
            if ids:
                self._append({"event": "reason", "ids": ids, "reason": reason})
            return len(ids)

    def clear(self) -> int:
        with self._lock:
            count = len(self._fold(self._read_events()))
            if count:
                self._append({"event": "clear"})
            return count
