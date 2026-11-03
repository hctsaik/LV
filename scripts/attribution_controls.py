"""Deterministic, geometry-matched perturbations; no inference or causal claims."""
from __future__ import annotations

import hashlib
import math

import numpy as np
from PIL import Image, ImageFilter


PROTOCOL_VERSION = 2
CONTROL_STRATEGY = "translated_mask"


def _mask_digest(mask):
    return hashlib.sha256(str(mask.shape).encode() + np.packbits(mask).tobytes()).hexdigest()


def iter_variants(pixels, scores, *, fractions=(.1, .25, .5), controls=3,
                  seed=42, pad_value=114, fill_methods=("constant", "blur"), max_overlap=.1):
    """Yield variant metadata plus pixels, or an explicit unavailable row.

    Translation never wraps/clips. Search is bounded to 512 deterministic candidate
    placements per fraction; failure means no control found within that search,
    not a proof that none exists. All fills use exactly the same selected masks.
    """
    pixels, scores = np.asarray(pixels), np.asarray(scores)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("pixels must be uint8 RGB")
    if scores.shape != pixels.shape[:2] or not np.isfinite(scores).all() or np.any(scores < 0) or not np.any(scores > 0):
        raise ValueError("scores must be finite, nonnegative, source-sized and nonzero")
    fractions, fill_methods = tuple(fractions), tuple(fill_methods)
    if not fractions or len(fractions) > 10 or len(set(fractions)) != len(fractions) or any(not math.isfinite(f) or not 0 < f < 1 for f in fractions):
        raise ValueError("fractions must be unique values in (0, 1)")
    if isinstance(controls, bool) or not isinstance(controls, int) or not 1 <= controls <= 10:
        raise ValueError("controls must be 1..10")
    if len(set(fill_methods)) != len(fill_methods) or not fill_methods or set(fill_methods) - {"constant", "blur"}:
        raise ValueError("unsupported or duplicate fill method")
    if not 0 <= max_overlap <= .1 or not 0 <= pad_value <= 255:
        raise ValueError("invalid overlap or fill")
    h, w = scores.shape
    rank = np.argsort(-scores.ravel(), kind="stable")
    rng = np.random.default_rng(seed)
    blur_radius = max(1., min(h, w) / 20.)
    fills = {"constant": np.full_like(pixels, pad_value)}
    if "blur" in fill_methods:
        fills["blur"] = np.asarray(Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(blur_radius)))
    for fraction in fractions:
        count = max(1, int(h * w * fraction))
        mask = np.zeros((h, w), dtype=bool)
        mask.ravel()[rank[:count]] = True
        yy, xx = np.nonzero(mask)
        ymin, ymax, xmin, xmax = int(yy.min()), int(yy.max()), int(xx.min()), int(xx.max())
        lo_y, hi_y, lo_x, hi_x = -ymin, h - 1 - ymax, -xmin, w - 1 - xmax
        placements = {(dy, dx) for dy in (lo_y, 0, hi_y) for dx in (lo_x, 0, hi_x)}
        feasible_count = (hi_y - lo_y + 1) * (hi_x - lo_x + 1)
        if feasible_count <= 512:
            placements.update((dy, dx) for dy in range(lo_y, hi_y + 1) for dx in range(lo_x, hi_x + 1))
        else:
            placements.update(zip(rng.integers(lo_y, hi_y + 1, 503).tolist(), rng.integers(lo_x, hi_x + 1, 503).tolist()))
        placements = sorted(placements)
        rng.shuffle(placements)
        selected = [(mask, 0, (0, 0), 1.)]
        for dy, dx in placements:
            if dy == dx == 0:
                continue
            overlap = float(mask[yy + dy, xx + dx].sum() / count)
            if overlap > max_overlap:
                continue
            control_mask = np.zeros_like(mask)
            control_mask[yy + dy, xx + dx] = True
            selected.append((control_mask, len(selected), (dy, dx), overlap))
            if len(selected) == controls + 1:
                break
        for method in fill_methods:
            for variant_mask, control, (dy, dx), overlap in selected:
                modified = pixels.copy()
                modified[variant_mask] = fills[method][variant_mask]
                digest = _mask_digest(variant_mask)
                yield {"fraction": fraction, "role": "attribution" if control == 0 else "random_control",
                       "control": control, "fill_method": method, "control_strategy": CONTROL_STRATEGY,
                       "pair_id": f"{fraction}:{control}:{digest[:16]}", "mask_sha256": digest,
                       "masked_pixels": count, "translation": [dy, dx], "overlap": overlap,
                       "mask_bbox": [xmin + dx, ymin + dy, xmax + dx + 1, ymax + dy + 1],
                       "blur_radius": blur_radius if method == "blur" else None,
                       "pixels": modified}
            if len(selected) - 1 < controls:
                yield {"fraction": fraction, "fill_method": method, "control_strategy": CONTROL_STRATEGY,
                       "role": "random_control", "unavailable": True, "requested_controls": controls,
                       "available_controls": len(selected) - 1,
                       "reason": "No additional distinct bounded translation found within deterministic search and overlap limit."}


def assess_faithfulness(report):
    """Assess observed fixed-score direction; never diagnose why a defect exists."""
    unknown = {"status": "inconclusive", "support": 0, "comparisons": [],
               "limitations": ["Matched multi-fill controls are unavailable or unverified."],
               "stability": {"status": "unknown", "methods": []}}
    if report.get("status") != "measured" or report.get("protocol_version") != PROTOCOL_VERSION:
        return unknown
    fractions = report.get("fractions", (report.get("protocol") or {}).get("fractions"))
    fills = report.get("fill_methods", (report.get("protocol") or {}).get("fill_methods"))
    if not isinstance(fractions, (list, tuple)) or not fractions or not isinstance(fills, (list, tuple)) or set(fills) != {"constant", "blur"}:
        return unknown
    if any(isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f) or not 0 < f < 1 for f in fractions):
        return unknown
    groups = {}
    incomplete = False
    for row in report.get("rows", []):
        if row.get("unavailable"):
            incomplete = True
            continue
        if row.get("control_strategy") != CONTROL_STRATEGY or not row.get("mask_sha256") or row.get("fill_method") not in {"constant", "blur"}:
            incomplete = True
            continue
        try:
            drop = float(row["fixed_score_drop"])
            fraction = float(row["fraction"])
            if not math.isfinite(drop) or not math.isfinite(fraction):
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            incomplete = True
            continue
        key = (fraction, row.get("fill_method"))
        groups.setdefault(key, []).append((row, drop))
    comparisons = []
    requested = report.get("controls", 3)
    if isinstance(requested, bool) or not isinstance(requested, int) or not 1 <= requested <= 10:
        return unknown
    for (fraction, method), rows in sorted(groups.items()):
        selected = [(r, d) for r, d in rows if r.get("role") == "attribution"]
        controls = [(r, d) for r, d in rows if r.get("role") == "random_control"]
        if len(selected) != 1 or len(controls) < requested or len({r["mask_sha256"] for r, _ in controls}) != len(controls):
            incomplete = True
            continue
        delta = selected[0][1] - float(np.mean([d for _, d in controls]))
        comparisons.append({"fraction": fraction, "fill_method": method, "control_strategy": CONTROL_STRATEGY,
                            "control_count": len(controls), "excess_drop": delta,
                            "direction": "supports" if delta > 1e-8 else "opposes" if delta < -1e-8 else "inconclusive"})
    methods = sorted({r["fill_method"] for r in comparisons if r["fill_method"]})
    directions = {r["direction"] for r in comparisons}
    coverage = {(r["fraction"], r["fill_method"]) for r in comparisons} == {(f, method) for f in fractions for method in fills}
    unstable = "supports" in directions and "opposes" in directions
    status = "unstable" if unstable else next(iter(directions)) if len(directions) == 1 else "inconclusive"
    if incomplete or not coverage or set(methods) != {"constant", "blur"}:
        status = "inconclusive"
    return {"status": status, "support": len(comparisons), "comparisons": comparisons,
            "limitations": ["Observed fixed-score sensitivity for this target; not a causal defect explanation."] +
                (["Requested matched controls or fill coverage are incomplete."] if incomplete or not coverage else []),
            "stability": {"status": "unstable" if unstable else "stable" if status in {"supports", "opposes"} else "unknown", "methods": methods}}
