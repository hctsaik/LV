"""Deterministic, bounded plot sampling for class-similarity views."""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from class_similarity import AnalysisItem


def sample_scatter_rows(
    items: Sequence[AnalysisItem], metric: str, analysis_key: str,
    per_class_cap: int = 200, margin_keep: int = 20,
) -> list[AnalysisItem]:
    if metric not in {"same", "other", "margin"}:
        raise ValueError("metric must be same|other|margin")
    if per_class_cap < 1 or margin_keep < 0:
        raise ValueError("sampling limits are invalid")
    groups: dict[str, list[AnalysisItem]] = {}
    for item in items:
        value = getattr(item, metric)
        if value is None or not np.isfinite(value):
            continue
        groups.setdefault(item.label or "未分類／標籤無效", []).append(item)
    chosen: list[AnalysisItem] = []
    for label in sorted(groups):
        rows = groups[label]
        if len(rows) <= per_class_cap:
            chosen.extend(rows)
            continue
        with_margin = [row for row in rows if row.margin is not None and np.isfinite(row.margin)]
        with_margin.sort(key=lambda row: (row.margin, row.record_id))
        keep = list(with_margin[:margin_keep])
        if len(keep) < margin_keep:
            rest = sorted(rows, key=lambda row: (getattr(row, metric), row.record_id))
            seen = {row.record_id for row in keep}
            for row in rest:
                if row.record_id not in seen:
                    keep.append(row)
                    seen.add(row.record_id)
                if len(keep) >= margin_keep:
                    break
        seen = {row.record_id for row in keep}
        remaining = [row for row in rows if row.record_id not in seen]
        remaining.sort(key=lambda row: hashlib.sha256(
            f"{analysis_key}{row.record_id}".encode("utf-8")).hexdigest())
        keep.extend(remaining[:max(0, per_class_cap - len(keep))])
        chosen.extend(keep)
    return chosen


def jitter_x(analysis_key: str, record_id: str, scale: float = 0.18) -> float:
    digest = hashlib.sha256(f"{analysis_key}{record_id}".encode("utf-8")).digest()
    uniform = int.from_bytes(digest[:8], "big") / float(2 ** 64)
    return (uniform - 0.5) * 2.0 * scale
