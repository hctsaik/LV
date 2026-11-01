from __future__ import annotations

import json
from pathlib import Path

import pytest

from curation_store import CurationStore, CurationStoreError


def _entry(path: Path, *, source: str = "manual", level: str = "image", obj_index: int | None = None) -> dict:
    return {
        "index": 0,
        "filename": path.name,
        "path": str(path),
        "label": "scratch",
        "split": "train",
        "sha256": "a" * 64,
        "source": source,
        "score": None,
        "reason": "",
        "level": level,
        **({"obj_index": obj_index, "bbox": [0.5, 0.5, 0.25, 0.25]} if level == "object" else {}),
    }


def test_curation_store_preserves_first_path_keyed_entry_and_replays_events(tmp_path: Path):
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    log = tmp_path / "cache" / "curation_log.jsonl"
    store = CurationStore(log)

    added, skipped = store.add([_entry(image)])
    assert (added, skipped) == (1, 0)
    added, skipped = store.add([_entry(image, source="search")])
    assert (added, skipped) == (0, 1)

    item = store.list_items()[0]
    assert item["source"] == "manual"
    assert item["reason"] == ""
    assert len(item["sha256"]) == 64
    assert store.set_reason([item["id"], item["id"], "missing"], "review later") == 1

    restarted = CurationStore(log)
    assert restarted.list_items()[0]["reason"] == "review later"
    assert restarted.remove(item["id"]) is True
    assert CurationStore(log).list_items() == []


def test_curation_store_distinguishes_objects_from_the_same_image(tmp_path: Path):
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    store = CurationStore(tmp_path / "curation_log.jsonl")
    added, skipped = store.add([
        _entry(image, level="object", obj_index=0),
        _entry(image, level="object", obj_index=1),
    ])
    assert (added, skipped) == (2, 0)
    items = store.list_items()
    assert [item["obj_index"] for item in items] == [0, 1]
    assert items[0]["id"] != items[1]["id"]


def test_curation_store_surfaces_corrupt_log_without_overwriting_it(tmp_path: Path):
    log = tmp_path / "curation_log.jsonl"
    original = '{"event":"add"}\nnot-json\n'
    log.write_text(original, encoding="utf-8")
    store = CurationStore(log)

    with pytest.raises(CurationStoreError, match="line 2"):
        store.list_items()
    with pytest.raises(CurationStoreError):
        store.clear()
    assert log.read_text(encoding="utf-8") == original


def test_curation_store_writes_one_json_event_per_line(tmp_path: Path):
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    log = tmp_path / "curation_log.jsonl"
    CurationStore(log).add([_entry(image)])

    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "add"
    assert rows[0]["item"]["filename"] == "image.png"
