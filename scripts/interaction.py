"""Pure interaction logic for the Streamlit GUI — no streamlit imports.

Ported/adapted from VIX's framework-agnostic ``core/`` pattern: every
function here is unit-testable without a browser or a Streamlit session.
"""
from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.neighbors import NearestNeighbors


def parse_folder_paths(text: str) -> list[Path]:
    """Parse a newline-separated folder list into Path objects.

    One path per line; blank/whitespace-only lines are ignored; surrounding
    whitespace (including trailing ``\\r`` from CRLF input) is stripped.
    No existence check is performed — the caller validates.
    """
    if not text:
        return []
    paths = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            paths.append(Path(line))
    return paths


def build_nn_index(emb_matrix: np.ndarray) -> NearestNeighbors:
    """Fit a cosine NearestNeighbors index over ``emb_matrix`` (N, D)."""
    nn = NearestNeighbors(metric="cosine")
    nn.fit(np.asarray(emb_matrix))
    return nn


def find_similar_indices(
    emb_matrix: np.ndarray,
    query_idx: int,
    k: int = 9,
    nn_index: NearestNeighbors | None = None,
) -> tuple[list[int], list[float]]:
    """Return (indices, cosine_distances) of the k nearest neighbours to
    ``emb_matrix[query_idx]``, EXCLUDING the query itself.

    Results are sorted by ascending distance. ``k`` is clamped to N-1.
    Raises IndexError for an out-of-range ``query_idx``.
    """
    emb_matrix = np.asarray(emb_matrix)
    n = len(emb_matrix)
    if not 0 <= query_idx < n:
        raise IndexError(f"query_idx {query_idx} out of range for {n} embeddings")
    k = max(0, min(k, n - 1))
    if k == 0:
        return [], []
    if nn_index is None:
        nn_index = build_nn_index(emb_matrix)
    dist, idx = nn_index.kneighbors(emb_matrix[query_idx : query_idx + 1], n_neighbors=k + 1)
    out_idx, out_dist = [], []
    for d, i in zip(dist[0], idx[0]):
        if int(i) == query_idx:
            continue
        out_idx.append(int(i))
        out_dist.append(float(d))
    # if the query wasn't among the k+1 (duplicate rows), trim to k
    return out_idx[:k], out_dist[:k]


def compute_outlier_scores(
    candidates: np.ndarray,
    reference: np.ndarray,
    k: int = 5,
    candidates_in_reference: bool = False,
) -> np.ndarray:
    """Outlier-ness = mean cosine distance from each candidate row to its
    k nearest neighbours in ``reference`` (label-free novelty).

    Higher = more unlike the reference set. NOT a probability or an error
    verdict — the UI must disclaim this. ``k`` is clamped to the reference
    size. When ``candidates_in_reference`` is True the closest neighbour of
    each candidate (its own row) is dropped before averaging.
    """
    candidates = np.asarray(candidates)
    reference = np.asarray(reference)
    if len(candidates) == 0:
        return np.zeros(0, dtype=float)
    extra = 1 if candidates_in_reference else 0
    k_eff = max(1, min(k + extra, len(reference)))
    nn = NearestNeighbors(metric="cosine")
    nn.fit(reference)
    dist, _ = nn.kneighbors(candidates, n_neighbors=k_eff)
    if candidates_in_reference and dist.shape[1] > 1:
        dist = dist[:, 1:]
    return dist.mean(axis=1)


def selection_points_to_indices(points: list[dict]) -> list[int]:
    """Extract global record indices from a Streamlit plotly selection.

    ``points`` is ``event.selection.points``; each point carries our global
    index in ``customdata`` (either ``[i]`` or scalar ``i``). Missing/None
    customdata entries are skipped. Returns a de-duplicated list preserving
    first-seen order, values as plain python ints.
    """
    seen: set[int] = set()
    out: list[int] = []
    for pt in points or []:
        cd = pt.get("customdata")
        if cd is None:
            continue
        if isinstance(cd, (list, tuple)):
            if not cd:
                continue
            cd = cd[0]
        try:
            i = int(cd)
        except (TypeError, ValueError):
            continue
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


_CSV_HEADER = ["index", "filename", "path", "label", "split"]
_EXPORT_HEADER = [*_CSV_HEADER, "sha256"]


def snapshots_to_csv(snapshots: list[dict]) -> str:
    """Serialize export-list snapshots to CSV (header always emitted).

    Each snapshot carries its manifest sha256 when available, so exported
    lists are content-addressed — traceable across renames and moves.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_EXPORT_HEADER)
    for i, s in enumerate(snapshots):
        writer.writerow([
            i, s.get("filename", ""), s.get("path", ""),
            s.get("label", ""), s.get("split", ""), s.get("sha256") or "",
        ])
    return buf.getvalue()


def records_to_csv(records: list[dict], indices: list[int]) -> str:
    """Serialize selected records to CSV text (header always emitted).

    Columns: index, filename, path, label, split. ``indices`` order is
    preserved.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_CSV_HEADER)
    for i in indices:
        r = records[i]
        p = Path(r["path"])
        writer.writerow([i, p.name, str(p), r.get("label", ""), r.get("split", "")])
    return buf.getvalue()


def zip_selected_images(records: list[dict], indices: list[int]) -> bytes:
    """Build an in-memory ZIP of the selected image files + a manifest.csv.

    Each image is stored under ``images/<split>/<filename>``; name
    collisions are disambiguated with a numeric suffix. Files missing on
    disk are skipped and recorded with ``status=missing`` in manifest.csv.
    """
    buf = io.BytesIO()
    manifest = io.StringIO()
    mwriter = csv.writer(manifest, lineterminator="\n")
    mwriter.writerow(_CSV_HEADER + ["status", "arcname"])
    used: set[str] = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in indices:
            r = records[i]
            p = Path(r["path"])
            arcname = f"images/{r.get('split', '')}/{p.name}"
            stem, suffix, n_try = p.stem, p.suffix, 1
            while arcname in used:
                arcname = f"images/{r.get('split', '')}/{stem}_{n_try}{suffix}"
                n_try += 1
            if p.exists():
                zf.write(p, arcname)
                used.add(arcname)
                status = "ok"
            else:
                status, arcname = "missing", ""
            mwriter.writerow(
                [i, p.name, str(p), r.get("label", ""), r.get("split", ""), status, arcname]
            )
        zf.writestr("manifest.csv", manifest.getvalue())
    return buf.getvalue()


def thumbnail_path_for(image_path: Path, size: int = 256) -> Path:
    """Deterministic cache location for an image's thumbnail.

    Lives in ``<image_dir>/.thumbs/<size>/<sha1(abspath|mtime|fsize)>.webp``
    so the key invalidates whenever the source file changes, and same-named
    files in different directories can never collide (per-dir cache).
    Raises OSError if the source file is missing.
    """
    image_path = Path(image_path)
    stat = image_path.stat()
    digest = hashlib.sha1(
        f"{image_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}".encode()
    ).hexdigest()[:16]
    return image_path.parent / ".thumbs" / str(size) / f"{digest}.webp"


def make_thumbnail(image_path: Path, size: int = 256) -> Path:
    """Create (or reuse) the cached thumbnail for ``image_path``.

    Returns the thumbnail path. Raises OSError for missing/unreadable
    sources — callers render a placeholder card instead of dropping it.
    """
    image_path = Path(image_path)
    out = thumbnail_path_for(image_path, size)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((size, size))
    img.save(out, "WEBP", quality=80)
    return out


def ensure_thumbnails(
    image_paths: Sequence[Path],
    size: int = 256,
    progress_cb: Callable[[int, int], None] | None = None,
) -> int:
    """Pre-generate thumbnails for all paths; returns how many are usable.

    Broken/missing sources are skipped (the grid shows a placeholder for
    them later). ``progress_cb(done, total)`` is called after each path.
    """
    n_total = len(image_paths)
    n_ok = 0
    for i, p in enumerate(image_paths):
        try:
            make_thumbnail(Path(p), size)
            n_ok += 1
        except OSError:
            pass
        if progress_cb is not None:
            progress_cb(i + 1, n_total)
    return n_ok


def spatial_order(
    coords: np.ndarray, indices: Sequence[int], n_rows: int = 12
) -> list[int]:
    """Order ``indices`` to mirror the scatter's spatial layout.

    Top of the plot first (high y), left-to-right (ascending x) within each
    of ``n_rows`` horizontal bands — so the thumbnail grid reads roughly
    like the chart, preserving spatial gestalt without in-plot highlights.
    """
    indices = list(indices)
    if not indices:
        return []
    ys = np.asarray([coords[i, 1] for i in indices], dtype=float)
    xs = np.asarray([coords[i, 0] for i in indices], dtype=float)
    y_min, y_max = float(ys.min()), float(ys.max())
    if y_max == y_min:
        rows = np.zeros(len(indices), dtype=int)
    else:
        rows = ((y_max - ys) / (y_max - y_min) * (n_rows - 1e-9)).astype(int)
    order = sorted(range(len(indices)), key=lambda j: (int(rows[j]), float(xs[j])))
    return [indices[j] for j in order]


def yolo_label_path_for(image_path: Path) -> Path:
    """Map an image path under ``<root>/images/x.jpg`` to its YOLO label
    file ``<root>/labels/x.txt`` (detector-mode dataset layout)."""
    image_path = Path(image_path)
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def draw_yolo_boxes(
    image_path: Path,
    label_path: Path,
    class_names: list[str] | None = None,
) -> Image.Image:
    """Return the image with its YOLO boxes drawn (red, 2px, class tag).

    A missing/empty label file yields the unmodified image. Lines that fail
    to parse are skipped.
    """
    img = Image.open(image_path).convert("RGB")
    label_path = Path(label_path)
    if not label_path.exists():
        return img
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x1, y1 = (cx + bw / 2) * w, (cy + bh / 2) * h
        draw.rectangle([x0, y0, x1, y1], outline="#e74c3c", width=2)
        name = (
            class_names[cid]
            if class_names and 0 <= cid < len(class_names)
            else f"class_{cid}"
        )
        draw.text((x0 + 2, max(0, y0 - 12)), name, fill="#e74c3c")
    return img
