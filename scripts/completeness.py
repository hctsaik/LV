"""Model-coverage completeness heatmap (defect-mechanisms v2 §5).

Bin a concept into an **attribute-axis grid** (interpretable cells the
user can act on — NOT a raw-embedding or UMAP grid, which distort
density), measure each cell's quantity *and* embedding diversity, and
roll the whole thing into one Coverage Health number plus the guards
that stop a single number from lying.

Design decisions (from the multi-agent review, docs/defect_mechanisms_v2.md):
- axes are interpretable attributes: class label / split (free) or
  bucketized image statistics (brightness/contrast/sharpness/aspect)
- embedding is the in-cell QUALITY probe, never the grid: a cell can be
  "full but fake" (many near-duplicates) — caught by low diversity ``d``
- the denominator is the real-world target ``t`` per cell; without a
  frequency prior it falls back to a uniform target, explicitly marked
  "uncalibrated"

Framework-free: no streamlit imports, every function unit-testable.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

# ── cell states ─────────────────────────────────────────────────────────
STATE_MISSING = "缺"        # n < 0.5t
STATE_LOW = "偏缺"          # 0.5t <= n < t
STATE_HEALTHY = "健康"      # n >= t and d >= d*
STATE_FAKE = "假完整"       # n >= t and d < d*  (near-duplicates padding)
STATE_OVER = "過多"         # n >= 1.5t and d >= d* (may subdivide)
STATE_EMPTY = "空"          # n == 0 (a missing cell with literally nothing)
STATE_NA = "不適用"          # 真實分佈中此組合不存在，排除於完整度分母外

_STATE_ORDER = [STATE_EMPTY, STATE_MISSING, STATE_LOW, STATE_HEALTHY,
                STATE_FAKE, STATE_OVER, STATE_NA]

# 粗分級頻率先驗 → 目標數乘數（討論共識：高/中/低三檔即可起步）
FREQ_MULT = {"高": 2.0, "中": 1.0, "低": 0.4}


# ── per-image attribute statistics (for auto axes) ──────────────────────

def image_stats(path: Path) -> dict[str, float]:
    """Cheap interpretable image properties for completeness axes.

    Computed on a downscaled grayscale load (a cached thumbnail is a fine
    proxy). Returns brightness (mean 0-1), contrast (std 0-1), sharpness
    (Laplacian variance, unbounded ≥0) and aspect (w/h). Raises OSError on
    unreadable sources so the caller can skip them.
    """
    with Image.open(path) as im:
        w, h = im.size
        g = np.asarray(im.convert("L").resize((128, 128)), dtype=np.float64) / 255.0
    # discrete Laplacian variance ≈ focus / edge energy
    lap = (-4 * g
           + np.roll(g, 1, 0) + np.roll(g, -1, 0)
           + np.roll(g, 1, 1) + np.roll(g, -1, 1))
    return {
        "brightness": float(g.mean()),
        "contrast": float(g.std()),
        "sharpness": float(lap.var()),
        "aspect": float(w / h) if h else 1.0,
    }


def bucketize(
    values: Sequence[float], n_bins: int, method: str = "quantile",
) -> tuple[list[int], list[str]]:
    """Map continuous values to ``n_bins`` ordered buckets.

    Returns (bucket index per value, human-readable bucket labels).
    ``method`` "quantile" gives equal-population buckets (robust to
    skew); "uniform" gives equal-width buckets. Degenerate input (all
    equal, or fewer distinct values than bins) collapses to one bucket.
    """
    arr = np.asarray(values, dtype=float)
    n_bins = max(1, int(n_bins))
    if arr.size == 0:
        return [], []
    if method == "quantile":
        qs = np.quantile(arr, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(qs)
    else:
        lo, hi = float(arr.min()), float(arr.max())
        edges = np.unique(np.linspace(lo, hi, n_bins + 1))
    if len(edges) < 2:  # all identical
        return [0] * arr.size, ["全部"]
    inner = edges[1:-1]
    idx = np.searchsorted(inner, arr, side="right").astype(int)
    n_actual = len(edges) - 1
    idx = np.clip(idx, 0, n_actual - 1)
    labels = [f"{edges[b]:.3g}–{edges[b + 1]:.3g}" for b in range(n_actual)]
    return idx.tolist(), labels


def categorical_buckets(values: Sequence) -> tuple[list[int], list[str]]:
    """Map categorical values to ordered bucket indices + labels."""
    labels = sorted({str(v) for v in values})
    pos = {lbl: i for i, lbl in enumerate(labels)}
    return [pos[str(v)] for v in values], labels


# ── embedding diversity probe ───────────────────────────────────────────

def global_mean_nn_distance(embeddings: np.ndarray) -> float:
    """Median nearest-neighbour cosine distance over all rows — the scale
    against which a cell's spread is judged. 0 for <2 rows."""
    from sklearn.neighbors import NearestNeighbors
    emb = np.asarray(embeddings)
    if len(emb) < 2:
        return 0.0
    nn = NearestNeighbors(metric="cosine", n_neighbors=2).fit(emb)
    dist, _ = nn.kneighbors(emb)
    return float(np.median(dist[:, 1]))


def effective_rank(embeddings: np.ndarray) -> float:
    """Effective rank = exp(entropy of normalized covariance eigenvalues).

    A cell whose points spread across many directions has high effective
    rank; near-duplicates collapse toward 1. Returns a value in
    [1, min(n, dim)]; 1.0 for <2 rows.
    """
    emb = np.asarray(embeddings, dtype=float)
    if len(emb) < 2:
        return 1.0
    centered = emb - emb.mean(axis=0, keepdims=True)
    # singular values of the centered matrix → covariance spectrum
    sv = np.linalg.svd(centered, compute_uv=False)
    ev = sv ** 2
    total = ev.sum()
    if total <= 0:
        return 1.0
    p = ev / total
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def cell_diversity(
    embeddings: np.ndarray, indices: Sequence[int], global_mean_nn: float,
) -> float:
    """In-cell diversity d ∈ [0,1] (defect-mechanisms v2 §5.2).

    Geometric mean of two normalized signals — spread (mean NN distance
    relative to the global scale) and dimensionality (effective rank
    relative to its ceiling) — so a cell that is large but all
    near-duplicates scores low (the "fake completeness" signal).
    A cell with <2 samples returns 0.0 (undefined diversity).
    """
    idx = list(indices)
    if len(idx) < 2:
        return 0.0
    emb = np.asarray(embeddings)[idx]
    # spread: cell mean-NN distance vs global, squashed to [0,1]
    cell_nn = global_mean_nn_distance(emb)
    scale = global_mean_nn if global_mean_nn > 0 else 1.0
    spread = min(cell_nn / (1.5 * scale), 1.0)
    # dimensionality: effective rank vs its ceiling
    ceiling = min(len(idx) - 1, emb.shape[1])
    dim = effective_rank(emb) / ceiling if ceiling > 0 else 0.0
    dim = min(max(dim, 0.0), 1.0)
    if spread <= 0.0 or dim <= 0.0:   # near-duplicates → genuinely zero
        return 0.0
    return float(np.sqrt(spread * dim))


# ── targets, classification, health ─────────────────────────────────────

def compute_target(
    n_cells: int,
    t_abs: int,
    freq_weight: float | None = None,
    n_total: int | None = None,
    p: float = 1.0,
    boundary_w: float = 1.0,
) -> float:
    """Target sample count ``t`` for one cell (defect-mechanisms v2 §5.4).

    ``t = max(t_abs, p · f_g · N_target) · w_g``. Without a frequency
    prior (``freq_weight`` None) it falls back to the absolute floor
    ``t_abs`` — the uniform, *uncalibrated* target. ``boundary_w`` is the
    decision-boundary multiplier (1.0 unless supplied).
    """
    base = float(t_abs)
    if freq_weight is not None and n_total is not None and n_cells > 0:
        base = max(base, p * freq_weight * n_total)
    return base * boundary_w


def classify_cell(n: int, d: float, t: float, d_star: float = 0.6) -> str:
    """Five-state cell verdict (priority order from v2 §5.2).

    Low diversity dominates: a full-but-duplicated cell is FAKE before it
    is OVER, because the actionable problem is its redundancy.
    """
    if n == 0:
        return STATE_EMPTY
    if n < 0.5 * t:
        return STATE_MISSING
    if n < t:
        return STATE_LOW
    if d < d_star:
        return STATE_FAKE          # n >= t but near-duplicates
    if n >= 1.5 * t:
        return STATE_OVER          # plenty and diverse → may subdivide
    return STATE_HEALTHY


def coverage_health(cells: list[dict], d_star: float = 0.6) -> dict:
    """Roll per-cell {n, d, t, state} up into the headline number + guards.

    Cells flagged STATE_NA (real distribution says this combination does
    not exist) are excluded from both numerator and denominator so they
    never count as missing.

    Returns:
      - ``coverage_health``: 0-100, Σ(min(n/t,1)·quality) / n_scored, where
        quality = 0.7 when a filled cell is fake (low diversity)
      - ``counts``: per-state cell tally
      - ``gini``: inequality of cell counts (balance guard)
      - ``fake_ratio``: share of scored cells flagged fake-complete
      - ``top_gaps``: cells sorted by shortfall (t - n), the worklist
    """
    counts: dict[str, int] = {s: 0 for s in _STATE_ORDER}
    score_sum = 0.0
    n_scored = 0
    n_vals, gaps = [], []
    n_fake = 0
    for c in cells:
        counts[c["state"]] = counts.get(c["state"], 0) + 1
        if c["state"] == STATE_NA:
            continue
        n, d, t = c["n"], c["d"], max(c["t"], 1e-9)
        n_scored += 1
        quality = 0.7 if (n >= t and d < d_star) else 1.0
        score_sum += min(n / t, 1.0) * quality
        n_vals.append(n)
        if c["state"] == STATE_FAKE:
            n_fake += 1
        shortfall = t - n
        if shortfall > 0:
            gaps.append({**c, "shortfall": round(shortfall, 1)})
    gaps.sort(key=lambda g: -g["shortfall"])
    if n_scored == 0:
        return {"coverage_health": 0.0, "counts": counts, "gini": 0.0,
                "fake_ratio": 0.0, "top_gaps": gaps}
    return {
        "coverage_health": round(100.0 * score_sum / n_scored, 1),
        "counts": counts,
        "gini": round(_gini(n_vals), 3),
        "fake_ratio": round(n_fake / n_scored, 3),
        "top_gaps": gaps,
    }


def cell_centroid(embeddings: np.ndarray, indices: Sequence[int]) -> np.ndarray | None:
    """Mean embedding of a cell's members, or None for an empty cell."""
    idx = list(indices)
    if not idx:
        return None
    return np.asarray(embeddings)[idx].mean(axis=0)


def mine_candidates(
    pool_embeddings: np.ndarray,
    query_vec: np.ndarray,
    k: int = 12,
    max_distance: float | None = None,
) -> tuple[list[int], list[float]]:
    """Rank candidate-pool rows by cosine proximity to ``query_vec`` (F4/F6:
    fill a missing cell from an unlabeled pool).

    Returns (pool row indices, cosine distances) sorted ascending, capped
    at k; rows beyond ``max_distance`` (when given) are dropped so a cell
    with no real match returns fewer/zero candidates rather than noise.
    """
    pool = np.asarray(pool_embeddings)
    if len(pool) == 0 or query_vec is None:
        return [], []
    q = np.asarray(query_vec, dtype=float).reshape(-1)
    qn = q / (np.linalg.norm(q) + 1e-12)
    pn = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-12)
    dist = 1.0 - pn @ qn
    order = np.argsort(dist)[: max(k, 0)]
    out_idx, out_dist = [], []
    for i in order:
        d = float(dist[i])
        if max_distance is not None and d > max_distance:
            break
        out_idx.append(int(i))
        out_dist.append(d)
    return out_idx, out_dist


def _gini(values: Sequence[float]) -> float:
    """Gini inequality of cell counts ∈ [0,1]; 0 = perfectly balanced."""
    arr = np.sort(np.asarray(values, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def build_completeness(
    records: list[dict],
    embeddings: np.ndarray,
    bucket_x: Sequence[int],
    bucket_y: Sequence[int],
    labels_x: list[str],
    labels_y: list[str],
    t_abs: int = 10,
    d_star: float = 0.6,
    freq_classes: dict[tuple[int, int], str] | None = None,
) -> dict:
    """Assemble the full completeness grid for two attribute axes.

    ``bucket_x``/``bucket_y`` are per-record bucket indices. Returns a
    dict with ``cells`` (one per (x,y) in the full grid, including empty
    cells so missing combinations show up), the ``health`` summary, and
    the axis labels.

    ``freq_classes`` maps a cell to a coarse real-world frequency prior —
    "高"/"中"/"低" scale the target up/down, "不適用" excludes the cell
    from the completeness denominator (a combination reality never
    produces, so it must not count as missing). When omitted every cell
    uses the uniform target ``t_abs`` (marked uncalibrated).
    """
    nx, ny = len(labels_x), len(labels_y)
    members: dict[tuple[int, int], list[int]] = {}
    for i, (bx, by) in enumerate(zip(bucket_x, bucket_y)):
        members.setdefault((int(bx), int(by)), []).append(i)

    global_nn = global_mean_nn_distance(embeddings)
    cells: list[dict] = []
    for xi in range(nx):
        for yi in range(ny):
            idx = members.get((xi, yi), [])
            n = len(idx)
            d = cell_diversity(embeddings, idx, global_nn)
            fclass = freq_classes.get((xi, yi)) if freq_classes else None
            if fclass == STATE_NA:
                cells.append({
                    "x": xi, "y": yi, "x_label": labels_x[xi], "y_label": labels_y[yi],
                    "n": n, "d": round(d, 3), "t": 0.0,
                    "state": STATE_NA, "indices": idx,
                })
                continue
            t = float(t_abs) * FREQ_MULT.get(fclass, 1.0)
            cells.append({
                "x": xi, "y": yi, "x_label": labels_x[xi], "y_label": labels_y[yi],
                "n": n, "d": round(d, 3), "t": round(t, 1),
                "state": classify_cell(n, d, t, d_star),
                "indices": idx,
            })
    health = coverage_health(cells, d_star=d_star)
    calibrated = bool(freq_classes) and any(
        v != "中" for v in freq_classes.values())
    return {
        "cells": cells, "health": health,
        "labels_x": labels_x, "labels_y": labels_y,
        "calibrated": calibrated,
    }
