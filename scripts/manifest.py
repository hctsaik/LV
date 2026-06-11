"""Dataset manifest — the single data contract for curation features (F1).

One ``manifest.jsonl`` per dataset folder (the split folder users load).
Each line describes one image:

    {
      "path":          relative POSIX path from the manifest's folder,
      "sha256":        content hash — cache / lineage / dedup key,
      "phash":         16-hex perceptual difference-hash, or null for
                       unreadable images (near-duplicate detection, F4),
      "split":         folder name,
      "labels":        list of class names for this image,
      "source":        where the entry came from ("discovered" for now),
      "captured_at":   file mtime as ISO-8601 (proxy until real capture
                       metadata exists),
      "size":          file size in bytes,
      "mtime_ns":      mtime in ns — incremental-update key: entries whose
                       (size, mtime_ns) are unchanged keep their hashes
                       without re-reading the file,
      "embedding_refs": {model_name: row index in that folder's
                        embeddings_<model>/embeddings.npz},
      "thumb_ref":     relative path of the cached thumbnail, or null
    }

Like interaction.py this module is framework-free: no streamlit imports,
every function unit-testable without a browser.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

MANIFEST_NAME = "manifest.jsonl"
MANIFEST_SOURCE_DISCOVERED = "discovered"


def manifest_path_for(folder: Path) -> Path:
    return Path(folder) / MANIFEST_NAME


def rel_key(folder: Path, path: Path) -> str:
    """Manifest key for an image: POSIX-style path relative to the folder."""
    return Path(path).resolve().relative_to(Path(folder).resolve()).as_posix()


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_phash(path: Path, hash_size: int = 8) -> str | None:
    """64-bit difference hash (dHash) as 16 hex chars; None if unreadable.

    Near-duplicate images yield small Hamming distances — good enough for
    duplicate candidate generation without an extra dependency.
    """
    try:
        img = Image.open(path).convert("L").resize(
            (hash_size + 1, hash_size), Image.LANCZOS)
    except OSError:
        return None
    px = np.asarray(img, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:0{hash_size * hash_size // 4}x}"


def load_manifest(folder: Path) -> dict[str, dict]:
    """Read manifest.jsonl → {rel_path: entry}. Missing file → {}.

    Unparseable lines are skipped (a corrupt line must not take down the
    whole dataset — those entries are simply rebuilt on the next update).
    """
    mpath = manifest_path_for(folder)
    if not mpath.exists():
        return {}
    entries: dict[str, dict] = {}
    for line in mpath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            entries[entry["path"]] = entry
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return entries


def write_manifest(folder: Path, entries: dict[str, dict]) -> Path:
    """Write entries (sorted by path for stable diffs) atomically."""
    mpath = manifest_path_for(folder)
    tmp = mpath.with_suffix(".jsonl.tmp")
    lines = [json.dumps(entries[k], ensure_ascii=False, sort_keys=True)
             for k in sorted(entries)]
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.replace(mpath)
    return mpath


def update_manifest(
    folder: Path,
    records: Sequence[dict],
    thumb_lookup: Callable[[Path], Path | None] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, dict]:
    """Build/refresh manifest entries for ``records`` of one folder.

    Incremental: an existing entry whose (size, mtime_ns) is unchanged
    keeps its sha256/phash/embedding_refs without re-reading the file.
    Entries for records no longer present are dropped. The caller decides
    when to persist via write_manifest (typically after embedding_refs
    are filled in).
    """
    folder = Path(folder)
    old = load_manifest(folder)
    entries: dict[str, dict] = {}
    n_total = len(records)
    for i, r in enumerate(records):
        p = Path(r["path"])
        key = rel_key(folder, p)
        stat = p.stat()
        prev = old.get(key)
        if prev is not None and prev.get("size") == stat.st_size \
                and prev.get("mtime_ns") == stat.st_mtime_ns:
            entry = dict(prev)
        else:
            entry = {
                "path": key,
                "sha256": file_sha256(p),
                "phash": compute_phash(p),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "captured_at": datetime.fromtimestamp(
                    stat.st_mtime).isoformat(timespec="seconds"),
                "source": MANIFEST_SOURCE_DISCOVERED,
                "embedding_refs": {},
            }
        entry["split"] = r.get("split", folder.name)
        label = r.get("label", "")
        entry["labels"] = [label] if label else []
        if thumb_lookup is not None:
            thumb = thumb_lookup(p)
            entry["thumb_ref"] = (
                rel_key(folder, thumb) if thumb is not None else None)
        else:
            entry.setdefault("thumb_ref", None)
        entries[key] = entry
        if progress_cb is not None:
            progress_cb(i + 1, n_total)
    return entries


def set_embedding_refs(
    entries: dict[str, dict],
    folder: Path,
    model_name: str,
    image_paths: Sequence[Path],
) -> None:
    """Record each image's row index in the folder's npz for ``model_name``.

    ``image_paths`` must be in npz row order (the order they were passed to
    extract_embeddings).
    """
    for row, p in enumerate(image_paths):
        key = rel_key(folder, Path(p))
        if key in entries:
            entries[key].setdefault("embedding_refs", {})[model_name] = row
