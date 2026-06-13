"""Unit tests for scripts/manifest.py (F1 data contract) and the
content-keyed cache validation in _utils.extract_embeddings."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from _utils import extract_embeddings
from manifest import (
    compute_phash,
    file_sha256,
    load_manifest,
    manifest_path_for,
    rel_key,
    set_embedding_refs,
    update_manifest,
    write_manifest,
)


# ── helpers ─────────────────────────────────────────────────────────────

def _image(dirpath: Path, name: str, seed: int = 0, size=(64, 64)) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    arr = np.random.default_rng(seed).integers(0, 255, (*size[::-1], 3)).astype("uint8")
    Image.fromarray(arr).save(p)
    return p


def _records(folder: Path, paths: list[Path], label: str = "classA") -> list[dict]:
    return [{"path": p, "split": folder.name, "label": label} for p in paths]


# ── module hygiene（與 interaction.py 同一紀律）─────────────────────────

def test_manifest_no_streamlit_import():
    import manifest
    src = Path(manifest.__file__).read_text(encoding="utf-8")
    assert "import streamlit" not in src


# ── hashes ──────────────────────────────────────────────────────────────

def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello manifest" * 1000)
    assert file_sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_phash_deterministic_and_distinct(tmp_path):
    a = _image(tmp_path, "a.jpg", seed=1)
    b = _image(tmp_path, "b.jpg", seed=2)
    ha = compute_phash(a)
    assert ha is not None and len(ha) == 16
    assert compute_phash(a) == ha
    assert compute_phash(b) != ha  # different random images → different dHash


def test_phash_unreadable_returns_none(tmp_path):
    p = tmp_path / "broken.jpg"
    p.write_bytes(b"not an image")
    assert compute_phash(p) is None


# ── update / load / write round trip ────────────────────────────────────

def test_update_manifest_creates_full_entries(tmp_path):
    folder = tmp_path / "train"
    paths = [_image(folder / "classA", f"img{i}.jpg", seed=i) for i in range(3)]
    entries = update_manifest(folder, _records(folder, paths))
    assert len(entries) == 3
    e = entries[rel_key(folder, paths[0])]
    assert e["path"] == "classA/img0.jpg"
    assert len(e["sha256"]) == 64
    assert e["phash"] is not None
    assert e["split"] == "train"
    assert e["labels"] == ["classA"]
    assert e["source"] == "discovered"
    assert e["size"] > 0 and e["mtime_ns"] > 0
    assert e["embedding_refs"] == {}
    assert e["thumb_ref"] is None
    assert "T" in e["captured_at"]  # ISO-8601


def test_write_and_load_round_trip_unicode(tmp_path):
    folder = tmp_path / "訓練集"
    paths = [_image(folder / "類別A", "影像 1.jpg", seed=3)]
    entries = update_manifest(folder, _records(folder, paths, label="類別A"))
    write_manifest(folder, entries)
    assert manifest_path_for(folder).exists()
    loaded = load_manifest(folder)
    assert loaded == entries
    # human-readable: unicode is not escaped on disk
    raw = manifest_path_for(folder).read_text(encoding="utf-8")
    assert "類別A" in raw


def test_update_is_incremental_unchanged_files_keep_hashes(tmp_path, monkeypatch):
    folder = tmp_path / "train"
    paths = [_image(folder / "c", f"i{i}.jpg", seed=i) for i in range(2)]
    recs = _records(folder, paths)
    first = update_manifest(folder, recs)
    write_manifest(folder, first)

    import manifest as mod
    monkeypatch.setattr(mod, "file_sha256",
                        lambda *a, **k: pytest.fail("unchanged file was rehashed"))
    second = update_manifest(folder, recs)
    for k in first:
        assert second[k]["sha256"] == first[k]["sha256"]


def test_update_detects_changed_content_same_name(tmp_path):
    folder = tmp_path / "train"
    p = _image(folder / "c", "same.jpg", seed=1)
    recs = _records(folder, [p])
    e1 = update_manifest(folder, recs)
    write_manifest(folder, e1)
    _image(folder / "c", "same.jpg", seed=99)  # overwrite content, same name
    os.utime(p)  # ensure mtime moves even on coarse filesystems
    e2 = update_manifest(folder, recs)
    assert e2[rel_key(folder, p)]["sha256"] != e1[rel_key(folder, p)]["sha256"]


def test_update_drops_removed_records(tmp_path):
    folder = tmp_path / "train"
    paths = [_image(folder / "c", f"i{i}.jpg", seed=i) for i in range(3)]
    write_manifest(folder, update_manifest(folder, _records(folder, paths)))
    entries = update_manifest(folder, _records(folder, paths[:1]))
    assert len(entries) == 1


def test_embedding_refs_set_and_preserved_across_update(tmp_path):
    folder = tmp_path / "train"
    paths = [_image(folder / "c", f"i{i}.jpg", seed=i) for i in range(3)]
    recs = _records(folder, paths)
    entries = update_manifest(folder, recs)
    set_embedding_refs(entries, folder, "resnet18", paths)
    set_embedding_refs(entries, folder, "dinov2", list(reversed(paths)))
    assert entries[rel_key(folder, paths[1])]["embedding_refs"] == {
        "resnet18": 1, "dinov2": 1}
    write_manifest(folder, entries)
    # unchanged files keep their refs on the next update
    again = update_manifest(folder, recs)
    assert again[rel_key(folder, paths[0])]["embedding_refs"]["dinov2"] == 2


def test_thumb_lookup_recorded_relative(tmp_path):
    folder = tmp_path / "train"
    p = _image(folder / "c", "i.jpg")
    thumb = folder / "c" / ".thumbs" / "256" / "abc.webp"
    thumb.parent.mkdir(parents=True)
    thumb.write_bytes(b"x")
    entries = update_manifest(folder, _records(folder, [p]),
                              thumb_lookup=lambda _: thumb)
    assert entries[rel_key(folder, p)]["thumb_ref"] == "c/.thumbs/256/abc.webp"


def test_load_skips_corrupt_lines(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir(parents=True)
    manifest_path_for(folder).write_text(
        '{"path": "ok.jpg", "sha256": "x"}\nnot json\n{"no_path": 1}\n',
        encoding="utf-8")
    loaded = load_manifest(folder)
    assert list(loaded) == ["ok.jpg"]


# ── extract_embeddings content-keyed cache（修「同名換圖吃舊向量」bug）──

def _fake_embed(path: Path) -> np.ndarray:
    # embedding derived from CONTENT so stale-cache bugs are observable
    data = Path(path).read_bytes()
    return np.array([float(sum(data) % 1000), float(len(data)), 1.0, 2.0])


def test_cache_keys_hit_and_reorder(tmp_path):
    paths = [_image(tmp_path, f"i{i}.jpg", seed=i) for i in range(3)]
    keys = [file_sha256(p) for p in paths]
    cache = tmp_path / "emb" / "embeddings.npz"
    first = extract_embeddings(paths, _fake_embed, cache_path=cache, cache_keys=keys)
    calls = []
    out = extract_embeddings(
        list(reversed(paths)), _fake_embed, cache_path=cache,
        cache_keys=list(reversed(keys)),
        progress_cb=lambda d, t: calls.append((d, t)))
    assert calls == [(3, 3)]  # cache hit
    assert np.array_equal(out, first[::-1])


def test_cache_keys_miss_when_content_changes(tmp_path):
    p = _image(tmp_path, "same.jpg", seed=1)
    cache = tmp_path / "emb" / "embeddings.npz"
    old = extract_embeddings([p], _fake_embed, cache_path=cache,
                             cache_keys=[file_sha256(p)])
    _image(tmp_path, "same.jpg", seed=99)  # same name, new content
    new = extract_embeddings([p], _fake_embed, cache_path=cache,
                             cache_keys=[file_sha256(p)])
    assert not np.array_equal(old, new)
    assert np.array_equal(new, np.stack([_fake_embed(p)]))


def test_cache_keys_treat_legacy_cache_as_stale(tmp_path):
    p = _image(tmp_path, "i.jpg", seed=1)
    cache = tmp_path / "emb" / "embeddings.npz"
    extract_embeddings([p], _fake_embed, cache_path=cache)  # legacy: no keys
    calls = []
    extract_embeddings([p], _fake_embed, cache_path=cache,
                       cache_keys=[file_sha256(p)],
                       progress_cb=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 1)]  # re-extracted (1 of 1), not a single-shot hit
    data = np.load(cache)
    assert "keys" in data.files  # upgraded in place


def test_cache_keys_duplicate_content_exact_order_only(tmp_path):
    a = _image(tmp_path / "a", "dup.jpg", seed=5)
    b = tmp_path / "b" / "dup.jpg"
    b.parent.mkdir()
    b.write_bytes(a.read_bytes())  # byte-identical duplicate
    keys = [file_sha256(a), file_sha256(b)]
    assert keys[0] == keys[1]
    cache = tmp_path / "emb" / "embeddings.npz"
    first = extract_embeddings([a, b], _fake_embed, cache_path=cache, cache_keys=keys)
    # same sequence → exact-order hit
    calls = []
    out = extract_embeddings([a, b], _fake_embed, cache_path=cache, cache_keys=keys,
                             progress_cb=lambda d, t: calls.append((d, t)))
    assert calls == [(2, 2)]
    assert np.array_equal(out, first)


def test_cache_keys_length_mismatch_raises(tmp_path):
    p = _image(tmp_path, "i.jpg")
    with pytest.raises(ValueError):
        extract_embeddings([p], _fake_embed, cache_keys=[])


def test_cache_keys_none_keeps_legacy_behaviour(tmp_path):
    paths = [_image(tmp_path, f"i{i}.jpg", seed=i) for i in range(2)]
    cache = tmp_path / "emb" / "embeddings.npz"
    first = extract_embeddings(paths, _fake_embed, cache_path=cache)
    calls = []
    out = extract_embeddings(paths, _fake_embed, cache_path=cache,
                             progress_cb=lambda d, t: calls.append((d, t)))
    assert calls == [(2, 2)]
    assert np.array_equal(out, first)


# ── snapshots_to_csv（匯出清單帶 sha256）────────────────────────────────

def test_snapshots_to_csv_header_and_rows():
    from interaction import snapshots_to_csv
    csv_text = snapshots_to_csv([
        {"filename": "a.jpg", "path": "C:/x/a.jpg", "label": "cat",
         "split": "train", "sha256": "ab" * 32,
         "source": "sparse", "score": 0.42, "reason": "blind spot"},
        {"filename": "b.jpg", "path": "C:/x/b.jpg", "label": "dog",
         "split": "val", "sha256": None},
    ])
    lines = csv_text.splitlines()
    assert lines[0] == "index,filename,path,label,split,sha256,source,score,reason"
    # provenance columns carry through
    assert lines[1].split(",")[6:] == ["sparse", "0.42", "blind spot"]
    assert "ab" * 32 in lines[1]
    # missing sha256/source/score/reason → empty cells, not "None"
    assert lines[2].endswith(",,,")
