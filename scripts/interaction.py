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

import hnswlib
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


def build_nn_index(emb_matrix: np.ndarray) -> hnswlib.Index:
    """Build a cosine HNSW index (hnswlib) over ``emb_matrix`` (N, D).

    The interactive query layer for F3 image-query and F7 text-query.
    HNSW is approximate by design; with these parameters (M=16,
    ef_construction=200, query ef >= 4k) recall is effectively 100% at the
    few-thousand-image scale this tool targets. Batch statistics
    (outlier-ness, label disagreement, dup radius scan) stay on sklearn
    exact search.
    """
    emb = np.ascontiguousarray(np.asarray(emb_matrix), dtype=np.float32)
    n, dim = emb.shape
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=max(n, 1), ef_construction=200, M=16,
                     random_seed=42)
    if n:
        index.add_items(emb, np.arange(n))
    return index


def _knn_query(nn_index: hnswlib.Index, vec: np.ndarray, k: int):
    nn_index.set_ef(max(64, k * 4))
    labels, dists = nn_index.knn_query(
        np.ascontiguousarray(vec, dtype=np.float32).reshape(1, -1), k=k)
    # float32 rounding can give ~-1e-7 for identical vectors — clamp
    return labels[0], np.maximum(dists[0], 0.0)


def find_similar_indices(
    emb_matrix: np.ndarray,
    query_idx: int,
    k: int = 9,
    nn_index: hnswlib.Index | None = None,
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
    idx, dist = _knn_query(nn_index, emb_matrix[query_idx], min(k + 1, n))
    out_idx, out_dist = [], []
    for i, d in zip(idx, dist):
        if int(i) == query_idx:
            continue
        out_idx.append(int(i))
        out_dist.append(float(d))
    # if the query wasn't among the k+1 (duplicate rows), trim to k
    return out_idx[:k], out_dist[:k]


def find_similar_to_vector(
    emb_matrix: np.ndarray,
    query_vec: np.ndarray,
    k: int = 9,
    nn_index: hnswlib.Index | None = None,
) -> tuple[list[int], list[float]]:
    """Return (indices, cosine_distances) of the k nearest rows to an
    EXTERNAL query vector — e.g. a Chinese-CLIP text embedding (F7).

    No self-exclusion (the query is not a library row). ``k`` is clamped
    to N. The query's dimensionality must match the matrix.
    """
    emb_matrix = np.asarray(emb_matrix)
    n = len(emb_matrix)
    if n == 0:
        return [], []
    query_vec = np.asarray(query_vec).reshape(-1)
    if query_vec.shape[0] != emb_matrix.shape[1]:
        raise ValueError(
            f"query dim {query_vec.shape[0]} != embedding dim {emb_matrix.shape[1]}")
    k = max(1, min(k, n))
    if nn_index is None:
        nn_index = build_nn_index(emb_matrix)
    idx, dist = _knn_query(nn_index, query_vec, k)
    return [int(i) for i in idx], [float(d) for d in dist]


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


def select_gray_zone(scores: np.ndarray, k: int) -> list[int]:
    """Gray-zone review queue (§3): the k most ambiguous items by score
    (e.g. kNN label-disagreement), highest first. ``k`` clamped to N."""
    s = np.asarray(scores)
    if len(s) == 0 or k <= 0:
        return []
    return [int(i) for i in np.argsort(s)[::-1][:min(k, len(s))]]


def nearest_anchor(
    embeddings: np.ndarray, idx: int, anchor_indices: Sequence[int],
) -> tuple[int | None, float]:
    """Closest anchor to ``idx`` by cosine distance (which 明確是/明確否
    example this gray-zone item sits nearest to). Returns (anchor_idx, d);
    (None, inf) when there are no anchors."""
    anchors = [a for a in anchor_indices if a != idx]
    if not anchors:
        return None, float("inf")
    emb = np.asarray(embeddings, dtype=float)
    q = emb[idx] / (np.linalg.norm(emb[idx]) + 1e-12)
    best, best_d = None, float("inf")
    for a in anchors:
        v = emb[a] / (np.linalg.norm(emb[a]) + 1e-12)
        d = float(1.0 - q @ v)
        if d < best_d:
            best, best_d = a, d
    return best, best_d


_GRAY_DECISION_HEADER = [
    "path", "soft_label", "confidence", "anchor", "reason",
    "proposer", "approver", "status",
]


def gray_decision_csv(decisions: Sequence[dict]) -> str:
    """Serialize confirmed gray-zone decisions to CSV — the four required
    provenance fields (who confirmed, anchor compared, soft label, reason)
    plus status, so an approved decision is auditable downstream."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(_GRAY_DECISION_HEADER)
    for d in decisions:
        w.writerow([d.get(k, "") for k in _GRAY_DECISION_HEADER])
    return buf.getvalue()


def farthest_point_sampling(
    embeddings: np.ndarray,
    n: int,
    seed_indices: Sequence[int] | None = None,
) -> list[int]:
    """k-center greedy / farthest-point sampling (F6 diversity selection).

    Iteratively pick the row whose cosine distance to the already-covered
    set is largest — the maximally-diverse subset to label next. With
    ``seed_indices`` (already-labeled rows) the picks COVER the gaps:
    points far from every seed are chosen first, which is the active-
    learning use (complements the heatmap's per-cell candidate mining).
    Without seeds it starts from the most peripheral point.

    Returns up to ``n`` indices in pick order (most diverse first), never
    including a seed.
    """
    emb = np.asarray(embeddings, dtype=float)
    N = len(emb)
    if N == 0 or n <= 0:
        return []
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    excluded: set[int] = set()
    selected: list[int] = []
    seeds = [i for i in (seed_indices or []) if 0 <= i < N]
    if seeds:
        min_d = (1.0 - norm @ norm[seeds].T).min(axis=1)
        excluded.update(seeds)
    else:
        centroid = norm.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        first = int(np.argmax(1.0 - norm @ centroid))
        selected.append(first)
        excluded.add(first)
        min_d = 1.0 - norm @ norm[first]
    n = min(n, N - len(excluded) + (0 if seeds else 1))
    while len(selected) < n:
        md = min_d.copy()
        if excluded:
            md[list(excluded)] = -1.0
        nxt = int(np.argmax(md))
        if md[nxt] < 0:
            break
        selected.append(nxt)
        excluded.add(nxt)
        min_d = np.minimum(min_d, 1.0 - norm @ norm[nxt])
    return selected


def load_scores_csv(csv_path: Path) -> dict[str, tuple[float, float | None]]:
    """Optional detection-score ingestion for the escape card (N4 gate).

    Reads a ``scores.csv`` with columns ``filename,score[,threshold]`` →
    ``{filename: (score, threshold|None)}``. This is how a detection model's
    output enters LV without LV running the detector. Missing file or
    unparseable rows yield an empty / partial map (the card degrades to
    embedding-only attribution).
    """
    csv_path = Path(csv_path)
    out: dict[str, tuple[float, float | None]] = {}
    if not csv_path.exists():
        return out
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("filename") or row.get("file") or "").strip()
            if not name:
                continue
            try:
                score = float(row["score"])
            except (KeyError, TypeError, ValueError):
                continue
            thr = row.get("threshold")
            try:
                thr_v = float(thr) if thr not in (None, "") else None
            except ValueError:
                thr_v = None
            out[name] = (score, thr_v)
    return out


def neighbor_hit_density(
    emb_matrix: np.ndarray,
    query_idx: int,
    radius: float,
    nn_index=None,
) -> int:
    """N2 signal: how many OTHER rows fall within cosine ``radius`` of the
    query (excluding itself). Low density = the model was shown few
    similar examples → sample scarcity candidate."""
    emb = np.asarray(emb_matrix)
    n = len(emb)
    if not 0 <= query_idx < n or n < 2:
        return 0
    q = emb[query_idx]
    qn = q / (np.linalg.norm(q) + 1e-12)
    en = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    dist = 1.0 - en @ qn
    return int(np.sum(dist <= radius)) - 1  # drop self (distance 0)


def neighbor_label_entropy(
    emb_matrix: np.ndarray,
    labels: Sequence[str],
    query_idx: int,
    k: int = 20,
) -> float:
    """N3 signal: normalized Shannon entropy ∈ [0,1] of the k nearest
    neighbours' labels (self excluded). High = neighbours disagree on the
    label → standard-drift / labeling-dispute candidate."""
    emb = np.asarray(emb_matrix)
    n = len(emb)
    if not 0 <= query_idx < n or n < 2:
        return 0.0
    k_eff = max(1, min(k, n - 1))
    nn = NearestNeighbors(metric="cosine")
    nn.fit(emb)
    _, idx = nn.kneighbors(emb[query_idx:query_idx + 1], n_neighbors=min(k_eff + 1, n))
    neigh = [int(j) for j in idx[0] if int(j) != query_idx][:k_eff]
    if not neigh:
        return 0.0
    arr = np.asarray(labels, dtype=object)[neigh]
    _, counts = np.unique(arr, return_counts=True)
    p = counts / counts.sum()
    ent = -float(np.sum(p * np.log(p)))
    max_ent = np.log(len(counts)) if len(counts) > 1 else 1.0
    return float(ent / max_ent) if max_ent > 0 else 0.0


# escape attribution classes (defect-mechanisms decision tree A–E)
ESCAPE_A = "A 標準漂移"        # neighbours disagree on label (N3 high)
ESCAPE_B = "B 樣本稀缺"        # few similar training examples (N2 low)
ESCAPE_C = "C 邊界擦邊"        # model score sits next to the threshold (N4)
ESCAPE_D = "D 新型態"          # no neighbours + outlier (N2 zero, novelty high)
ESCAPE_REVIEW = "需人工覆核"    # signals insufficient to attribute


def attribute_escape(
    hit_density: int,
    label_entropy: float,
    outlier_pct: float,
    score: float | None = None,
    threshold: float | None = None,
    entropy_thr: float = 0.8,
    density_thr: int = 3,
) -> dict:
    """Preliminary escape attribution from embedding signals (+ optional
    model score). Honest by construction: N0/N1 need a human and N4 needs
    a detection score, so without a score this only separates A/B/D and
    otherwise defers to 需人工覆核.

    Returns {class, confidence (0-1), reasons[list]}.
    """
    reasons: list[str] = []
    if label_entropy >= entropy_thr:
        reasons.append(f"鄰居標籤分歧高（熵 {label_entropy:.2f} ≥ {entropy_thr}）")
        return {"class": ESCAPE_A, "confidence": round(min(label_entropy, 1.0), 2),
                "reasons": reasons}
    if hit_density == 0 and outlier_pct >= 0.9:
        reasons.append(f"訓練集無相似鄰居，且離群度居前 {(1 - outlier_pct) * 100:.0f}%")
        return {"class": ESCAPE_D, "confidence": round(outlier_pct, 2),
                "reasons": reasons}
    if score is not None and threshold is not None:
        margin = abs(score - threshold)
        rel = margin / (abs(threshold) + 1e-9)
        if rel <= 0.1:
            reasons.append(f"模型分數貼近閾值（|{score:.3f}−{threshold:.3f}| 相對 {rel*100:.0f}%）")
            return {"class": ESCAPE_C, "confidence": round(1 - rel, 2), "reasons": reasons}
    if hit_density <= density_thr:
        reasons.append(f"訓練集相似鄰居稀少（{hit_density} ≤ {density_thr}）")
        return {"class": ESCAPE_B, "confidence": round(1 - hit_density / (density_thr + 1), 2),
                "reasons": reasons}
    reasons.append("embedding 訊號不足以歸因（鄰居充足且標籤一致）；"
                   "需 N0 品質/N1 定義/N4 分數判定")
    return {"class": ESCAPE_REVIEW, "confidence": 0.3, "reasons": reasons}


def compute_label_disagreement(
    embeddings: np.ndarray,
    labels: Sequence[str],
    k: int = 5,
) -> np.ndarray:
    """Label audit (F5): for each row, the fraction of its k nearest
    neighbours (self excluded) carrying a DIFFERENT label.

    0 = neighbourhood agrees, 1 = neighbourhood disagrees. This is a
    neighbourhood statistic, NOT a mislabel verdict — the UI must keep the
    honest framing. ``k`` is clamped to N-1; with fewer than 2 rows (or a
    single class) every score is 0.
    """
    embeddings = np.asarray(embeddings)
    labels = list(labels)
    n = len(labels)
    if n < 2:
        return np.zeros(n, dtype=float)
    k_eff = max(1, min(k, n - 1))
    nn = NearestNeighbors(metric="cosine")
    nn.fit(embeddings)
    # +1 so we can drop each row's own entry (byte-identical duplicates may
    # shuffle who comes first, so drop by index, not by position)
    _, idx = nn.kneighbors(embeddings, n_neighbors=min(k_eff + 1, n))
    arr = np.asarray(labels, dtype=object)
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        neigh = [j for j in idx[i] if j != i][:k_eff]
        if neigh:
            scores[i] = float(np.mean(arr[neigh] != arr[i]))
    return scores


def hamming_distance_hex(a: str, b: str) -> int:
    """Hamming distance between two equal-length hex hash strings."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _filter_pair(i: int, j: int, splits: Sequence[str] | None,
                 cross_split_only: bool) -> bool:
    if not cross_split_only:
        return True
    return splits is not None and splits[i] != splits[j]


def find_duplicate_pairs_phash(
    phashes: Sequence[str | None],
    max_hamming: int = 4,
    splits: Sequence[str] | None = None,
    cross_split_only: bool = False,
    max_pairs: int = 200,
) -> list[tuple[int, int, int]]:
    """Duplicate candidates (F4) by perceptual hash.

    Returns (i, j, hamming) with i < j, sorted by distance then indices,
    capped at ``max_pairs``. None hashes (unreadable images) are skipped.
    With cross_split_only=True only pairs spanning different splits are
    returned — i.e. train/val leakage candidates. O(N²) in vectorized
    chunks; fine for the few-thousand-image datasets this tool targets.
    """
    idx = [i for i, h in enumerate(phashes) if h]
    if len(idx) < 2:
        return []
    vals = np.array([np.uint64(int(phashes[i], 16)) for i in idx], dtype=np.uint64)
    pairs: list[tuple[int, int, int]] = []
    for a in range(len(idx) - 1):
        xor = (vals[a] ^ vals[a + 1:]).astype(np.uint64)
        dists = np.unpackbits(xor.view(np.uint8)).reshape(len(xor), -1).sum(axis=1)
        for off in np.nonzero(dists <= max_hamming)[0]:
            i, j = idx[a], idx[a + 1 + off]
            if _filter_pair(i, j, splits, cross_split_only):
                pairs.append((i, j, int(dists[off])))
    pairs.sort(key=lambda p: (p[2], p[0], p[1]))
    return pairs[:max_pairs]


def find_duplicate_pairs_embedding(
    embeddings: np.ndarray,
    max_distance: float = 0.05,
    splits: Sequence[str] | None = None,
    cross_split_only: bool = False,
    max_pairs: int = 200,
) -> list[tuple[int, int, float]]:
    """Duplicate candidates (F4) by embedding cosine distance.

    Semantic near-duplicates that survive resizing/re-encoding, which
    phash misses. Same return contract as the phash variant.
    """
    embeddings = np.asarray(embeddings)
    if len(embeddings) < 2:
        return []
    nn = NearestNeighbors(metric="cosine", radius=max_distance)
    nn.fit(embeddings)
    dists, idxs = nn.radius_neighbors(embeddings)
    pairs: list[tuple[int, int, float]] = []
    for i, (ds, js) in enumerate(zip(dists, idxs)):
        for d, j in zip(ds, js):
            if j <= i:
                continue
            if _filter_pair(i, int(j), splits, cross_split_only):
                pairs.append((i, int(j), float(d)))
    pairs.sort(key=lambda p: (p[2], p[0], p[1]))
    return pairs[:max_pairs]


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
