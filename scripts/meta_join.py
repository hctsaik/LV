"""meta_join:metadata CSV ↔ 影像清單對照(sha256/檔名雙鍵)+ 分組索引。

設計:3_Architect_Design/23_meta_join.md(M20)。CSV 讀檔是唯一 I/O;
匹配失敗**雙向明講**(哪些列沒配到、哪些影像沒資料),絕不默略。
"""
from __future__ import annotations

import csv
from pathlib import Path

_SHA_COLS = ("sha256", "sha")
_FILE_COLS = ("filename", "file", "image", "image_name", "path", "filepath", "檔名")


def load_metadata_csv(csv_path) -> list[dict]:
    p = Path(csv_path)
    with open(p, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"CSV 無表頭或為空:{p}")
        rows = []
        for row in reader:
            clean = {(k or "").strip(): (v or "").strip() for k, v in row.items()
                     if k is not None}
            if any(clean.values()):
                rows.append(clean)
    return rows


def _detect(fieldnames, explicit, candidates):
    if explicit:
        return explicit
    low = {f.strip().lower(): f for f in fieldnames}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def join_metadata(rows, paths, sha_of=None, *,
                  filename_col=None, sha_col=None) -> dict:
    fieldnames = list(rows[0].keys()) if rows else []
    sc = _detect(fieldnames, sha_col, _SHA_COLS)
    fc = _detect(fieldnames, filename_col, _FILE_COLS)
    if sc is None and fc is None:
        raise ValueError("CSV 找不到鍵欄(需 sha256/sha 或 filename/file/image/"
                         f"path/檔名),實際欄位:{fieldnames}")
    key_cols = {c for c in (sc, fc) if c}
    fields = [f for f in fieldnames if f not in key_cols]

    by_sha: dict[str, list[int]] = {}
    if sha_of:
        for idx, s in sha_of.items():
            by_sha.setdefault(str(s), []).append(int(idx))
    by_name: dict[str, list[int]] = {}
    for i, p in enumerate(paths):
        by_name.setdefault(Path(str(p)).name.lower(), []).append(i)

    per_image: dict[int, dict] = {}
    matched_rows: set[int] = set()
    warnings: list[str] = []
    for ri, row in enumerate(rows):
        targets: list[int] = []
        if sc and sha_of is not None:
            v = row.get(sc, "")
            if v and v in by_sha:
                targets = by_sha[v]
        if not targets and fc:
            v = row.get(fc, "")
            if v:
                targets = by_name.get(Path(v).name.lower(), [])
        if not targets:
            continue
        matched_rows.add(ri)
        vals = {f: row.get(f, "") for f in fields}
        for idx in targets:
            if idx in per_image:
                warnings.append(f"影像 {Path(str(paths[idx])).name} 對上多列,"
                                f"取第 {ri + 1} 列(後列覆蓋)")
            per_image[idx] = dict(vals)

    return {"fields": fields, "per_image": per_image, "n_rows": len(rows),
            "n_matched_images": len(per_image),
            "n_unmatched_images": len(paths) - len(per_image),
            "unmatched_row_idx": [i for i in range(len(rows))
                                  if i not in matched_rows],
            "warnings": warnings}


def group_counts(per_image, n_images, field, *, max_groups: int = 20) -> dict:
    raw: dict[str, list[int]] = {}
    for idx, vals in per_image.items():
        v = (vals.get(field) or "").strip()
        if v:
            raw.setdefault(v, []).append(int(idx))
    for v in raw:
        raw[v].sort()
    ordered = sorted(raw.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    n_keep = max(int(max_groups) - 1, 0)
    keep, rest = ordered[:n_keep], ordered[n_keep:]
    # 沒有溢出時不需要 "(其他)" 桶——全部原組直出(含恰好等於 max_groups-1 的情況)
    if len(ordered) <= int(max_groups):
        keep, rest = ordered, []
    groups = {v: idxs for v, idxs in keep}
    if rest:
        groups["(其他)"] = sorted({i for _, idxs in rest for i in idxs})
    valued = {i for idxs in raw.values() for i in idxs}
    return {"groups": groups,
            "missing": [i for i in range(int(n_images)) if i not in valued]}
