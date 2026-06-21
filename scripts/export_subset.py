"""Export Subset — framework-free copy engine for "我的子集 / Export".

Takes a list of curated selections (cross-tool: visualize / completeness /
objcov / graytone / examset) and materialises them, byte-for-byte, into a
**new** destination folder as a usable YOLO subset — never touching the user's
source dataset. Like manifest.py / evaluation.py / labeling_handoff.py this
module is framework-free (pure stdlib + PIL for crop/size), so the whole
contract is unit-testable without a browser, and the Streamlit page is just a
thin caller around ``export_subset``.

Design contract: ``docs/export_subset_design.md`` §5/§6/§8. The five honesty
rules baked in here:

* **dst-containment (hard, non-disable)** — the destination may not be, equal,
  or sit inside any source dataset root, nor may a source root sit inside the
  destination. A hit raises ``ValueError`` and writes nothing (§8.3).
* **never silently overwrite** — a non-empty dst defaults to ``skip`` (logged in
  ``ExportReport.skipped``); rename/overwrite must be asked for explicitly (§8.4).
* **never silently eat data** — items whose image can't be resolved go to
  ``unresolved``; bad label rows / out-of-range coords / unknown class ids
  become ``warnings``; symlink failures become ``errors``. Nothing is dropped
  without a ledger entry (§8.5).
* **object-level never rewrites labels** — yolo layout always copies the FULL
  label file (整圖 + 全 labels); selected boxes only live in the manifest.
  crop-out emits a ``task: classification`` ImageFolder instead (§6).
* **reproducible + skew-warned** — every export writes ``manifest.csv`` (human),
  ``lineage.json`` (machine-rebuildable: class_remap, full sha list, self-hash)
  and carries the "刻意偏斜的診斷子集" warning (§8.9/§8.10).

Identity / dedup key is the image's **sha256** (content-addressed), so the same
image reached through two paths or two tools is exported once and its
``source_tag`` / ``object_ids`` are unioned into the manifest (§5).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ── constants ────────────────────────────────────────────────────────────────
LV_VERSION = "0.0.0-dev"  # placeholder until a real release stamp exists
MANIFEST_NAME = "manifest.csv"
OBJECT_MANIFEST_NAME = "objects.csv"
LINEAGE_NAME = "lineage.json"
SKEW_WARNING = (
    "刻意偏斜的診斷子集,直接訓練/評估會誤導"
    " (deliberately skewed diagnostic subset — training/benchmarking on it "
    "as-is will mislead)."
)
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


# ── public data model ────────────────────────────────────────────────────────
@dataclass
class ExportItem:
    """One curated selection. ``sha256`` is the identity/dedup key; when None
    it is computed from the image bytes at export time.

    ``level="object"`` items still export the WHOLE image in yolo layout (the
    picked ``object_ids`` only annotate the manifest); they drive per-box crops
    only in ``crop-out`` layout. ``object_ids`` are 0-based line indices into
    ``label_path`` (the YOLO .txt). ``class_names`` is the *source* dataset's
    class list, used to remap class ids into the merged subset ``classes.txt``.
    """
    image_path: Path
    label_path: Path | None = None
    class_names: list[str] = field(default_factory=list)
    split: str | None = None
    source_tool: str = ""
    source_tag: str = ""
    reason: str = ""
    level: str = "image"            # "image" | "object"
    object_ids: list[int] | None = None
    dataset_id: str = ""
    sha256: str | None = None


@dataclass
class ExportReport:
    """Full ledger of an export — nothing is dropped without a line here."""
    exported: int = 0
    skipped: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    deduped: int = 0
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    dst: str = ""


# ── small helpers ────────────────────────────────────────────────────────────
def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_ext(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".jpeg":
        return ".jpg"
    return ext if ext in _IMG_EXTS else ".jpg"


def _infer_source_root(image_path: Path) -> Path:
    """Best-effort dataset root for an image, so dst-containment can compare.

    YOLO datasets put images under ``.../images[/split]/<file>``. If an
    ``images`` component is present, the root is its parent; otherwise fall back
    to the image's own parent (still enough to catch dst == that folder).
    """
    p = Path(image_path).resolve()
    parts = p.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            return Path(*parts[:i]) if i > 0 else Path(parts[0])
    return p.parent


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _check_containment(dst: Path, source_roots: list[Path]) -> None:
    """Raise ValueError if dst overlaps any source root (either direction)."""
    dst_r = Path(dst).resolve()
    for root in source_roots:
        root_r = Path(root).resolve()
        if dst_r == root_r or _is_relative_to(dst_r, root_r) \
                or _is_relative_to(root_r, dst_r):
            raise ValueError(
                "Export destination overlaps a source dataset root — "
                "拒絕匯出:會污染來源資料集 (dst-containment hard guard). "
                f"dst={dst_r} 與來源根 {root_r} 重疊;請改用一個全新的空資料夾。"
            )


def _parse_label_lines(text: str) -> tuple[list[list[float]], list[str]]:
    """Parse a YOLO .txt → (rows, warnings). Each row is ``[cls, cx, cy, w, h]``
    with cls kept as the original (possibly-float-looking) int. Malformed lines
    are reported, never silently skipped."""
    rows: list[list[float]] = []
    warns: list[str] = []
    for ln, raw in enumerate(text.splitlines()):
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5:
            warns.append(f"label line {ln}: expected 5 cols, got {len(parts)} "
                         f"({s!r}) — dropped")
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(x) for x in parts[1:5]]
        except ValueError:
            warns.append(f"label line {ln}: non-numeric ({s!r}) — dropped")
            continue
        rows.append([cls, *coords])
    return rows, warns


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── class remap ──────────────────────────────────────────────────────────────
def _build_class_remap(items: list[ExportItem]) -> tuple[list[str], dict[str, dict[int, int]]]:
    """Union all source ``class_names`` by NAME into a stable-sorted subset
    palette, and build per-dataset ``old_id -> new_id`` maps.

    Keyed by ``dataset_id`` (falls back to a hash of the class-name tuple so two
    in-memory sources with the same names but no id still map correctly).
    """
    names: set[str] = set()
    for it in items:
        names.update(it.class_names or [])
    subset_classes = sorted(names)
    new_of = {name: i for i, name in enumerate(subset_classes)}

    remaps: dict[str, dict[int, int]] = {}
    for it in items:
        key = _dataset_key(it)
        if key in remaps:
            continue
        remaps[key] = {old: new_of[name]
                       for old, name in enumerate(it.class_names or [])}
    return subset_classes, remaps


def _dataset_key(it: ExportItem) -> str:
    if it.dataset_id:
        return it.dataset_id
    return "names:" + "|".join(it.class_names or [])


# ── dedup ────────────────────────────────────────────────────────────────────
@dataclass
class _Merged:
    """One image after sha256 dedup: the union of every selection on it."""
    sha256: str
    item: ExportItem                      # representative (first seen)
    source_tags: list[str] = field(default_factory=list)
    source_tools: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    levels: set[str] = field(default_factory=set)
    object_ids: list[int] = field(default_factory=list)


def _resolve_and_dedup(
    items: list[ExportItem], report: ExportReport
) -> dict[str, _Merged]:
    """sha256 each item (resolving the image), merging duplicates. Unreadable /
    missing images go to ``report.unresolved`` and are excluded (§8.5)."""
    merged: dict[str, _Merged] = {}
    for it in items:
        img = Path(it.image_path)
        sha = it.sha256
        if not sha:
            if not img.exists():
                report.unresolved.append({
                    "image_path": str(img), "reason": "image not found",
                    "source_tool": it.source_tool, "source_tag": it.source_tag,
                })
                continue
            try:
                sha = _file_sha256(img)
            except OSError as e:
                report.unresolved.append({
                    "image_path": str(img), "reason": f"unreadable: {e}",
                    "source_tool": it.source_tool, "source_tag": it.source_tag,
                })
                continue
        m = merged.get(sha)
        if m is None:
            m = _Merged(sha256=sha, item=it)
            merged[sha] = m
        else:
            report.deduped += 1
        for val, bucket in ((it.source_tag, m.source_tags),
                            (it.source_tool, m.source_tools),
                            (it.reason, m.reasons)):
            if val and val not in bucket:
                bucket.append(val)
        m.levels.add(it.level)
        for oid in (it.object_ids or []):
            if oid not in m.object_ids:
                m.object_ids.append(oid)
    for m in merged.values():
        m.object_ids.sort()
    return merged


# ── on_exists ────────────────────────────────────────────────────────────────
def _resolve_target(target: Path, on_exists: str, report: ExportReport,
                    sha: str) -> Path | None:
    """Apply on_exists policy to a would-be output path. Returns the final path
    to write, or None when the item is skipped (logged)."""
    if not target.exists():
        return target
    if on_exists == "overwrite":
        return target
    if on_exists == "rename":
        stem, suffix = target.stem, target.suffix
        i = 1
        while True:
            cand = target.with_name(f"{stem}__{i}{suffix}")
            if not cand.exists():
                return cand
            i += 1
    # default: skip
    report.skipped.append({"sha256": sha, "target": str(target),
                           "reason": "exists; on_exists=skip"})
    return None


# ── writers ──────────────────────────────────────────────────────────────────
def _copy_or_link(src: Path, dst: Path, mode: str, report: ExportReport,
                  sha: str) -> bool:
    """Materialise one image per ``mode``. symlink failure is an error, never a
    silent copy-fallback (§5). Returns True on success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return True
        except OSError as e:
            report.errors.append({"sha256": sha, "src": str(src),
                                  "reason": f"symlink failed: {e}"})
            return False
    try:
        shutil.copy2(src, dst)
        return True
    except OSError as e:
        report.errors.append({"sha256": sha, "src": str(src),
                              "reason": f"copy failed: {e}"})
        return False


def _remap_label_text(
    rows: list[list[float]], remap: dict[int, int], report: ExportReport,
    sha: str, drop_invalid: bool,
) -> str:
    """Remap class ids and validate coords; out-of-range / unknown-id rows are
    warned (and optionally dropped), never silently passed (§8.7)."""
    out_lines: list[str] = []
    for row in rows:
        cls = int(row[0])
        coords = row[1:5]
        new_cls = remap.get(cls)
        if new_cls is None:
            report.warnings.append({
                "sha256": sha, "kind": "unknown-class-id",
                "detail": f"class id {cls} not in source class_names"})
            if drop_invalid:
                continue
            new_cls = cls
        if any(not (0.0 <= c <= 1.0) for c in coords):
            report.warnings.append({
                "sha256": sha, "kind": "coord-out-of-range",
                "detail": f"coords {coords} outside [0,1]"})
            if drop_invalid:
                continue
        out_lines.append(
            " ".join([str(new_cls), *(f"{c:.6g}" for c in coords)]))
    return "\n".join(out_lines) + ("\n" if out_lines else "")


# ── crop-out ─────────────────────────────────────────────────────────────────
def _crop_box(img_w: int, img_h: int, box: list[float], pad: float
              ) -> tuple[int, int, int, int]:
    """YOLO normalized [cx,cy,w,h] (+pad fraction of box size) → clipped pixel
    box (left, top, right, bottom)."""
    cx, cy, w, h = box
    w = w * (1.0 + 2 * pad)
    h = h * (1.0 + 2 * pad)
    left = (cx - w / 2) * img_w
    top = (cy - h / 2) * img_h
    right = (cx + w / 2) * img_w
    bottom = (cy + h / 2) * img_h
    left = max(0, int(round(left)))
    top = max(0, int(round(top)))
    right = min(img_w, int(round(right)))
    bottom = min(img_h, int(round(bottom)))
    if right <= left:
        right = min(img_w, left + 1)
    if bottom <= top:
        bottom = min(img_h, top + 1)
    return left, top, right, bottom


# ── lineage / manifest ───────────────────────────────────────────────────────
def _write_manifest_csv(dst: Path, rows: list[dict]) -> None:
    cols = ["dst", "sha256", "source_path", "source_tool", "source_tag",
            "reason", "split", "level"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore",
                       lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    (dst / MANIFEST_NAME).write_text(buf.getvalue(), encoding="utf-8")


def _write_object_manifest_csv(dst: Path, rows: list[dict]) -> None:
    cols = ["crop", "crop_sha256", "source_sha256", "source_path", "bbox",
            "class", "pad", "source_tag"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore",
                       lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    (dst / OBJECT_MANIFEST_NAME).write_text(buf.getvalue(), encoding="utf-8")


def _write_lineage(dst: Path, *, mode: str, layout: str, on_exists: str,
                   pad: float, subset_classes: list[str],
                   class_remap: dict[str, dict[int, int]],
                   shas: list[str], manifest_rows: list[dict],
                   report: ExportReport) -> None:
    lineage = {
        "lv_version": LV_VERSION,
        "warning": SKEW_WARNING,
        "exported_at_utc": _utc_now(),
        "mode": mode,
        "layout": layout,
        "on_exists": on_exists,
        "pad": pad,
        "subset_classes": subset_classes,
        "class_remap": {k: {str(o): n for o, n in v.items()}
                        for k, v in class_remap.items()},
        "sha256_list": sorted(shas),
        "sources": sorted({r["source_path"] for r in manifest_rows}),
        "report": {
            "exported": report.exported,
            "deduped": report.deduped,
            "skipped": len(report.skipped),
            "unresolved": len(report.unresolved),
            "warnings": len(report.warnings),
            "errors": len(report.errors),
        },
    }
    # self-hash over the canonical content (excluding the field itself)
    payload = json.dumps(lineage, ensure_ascii=False, sort_keys=True)
    lineage["export_self_hash"] = hashlib.sha256(
        payload.encode("utf-8")).hexdigest()
    (dst / LINEAGE_NAME).write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")


def _write_data_yaml(dst: Path, subset_classes: list[str], layout: str) -> None:
    """Minimal hand-rolled data.yaml (no pyyaml dependency)."""
    lines = [f"# {SKEW_WARNING}"]
    if layout == "crop-out":
        lines += ["task: classification", "path: ."]
    else:
        lines += ["path: .", "train: images", "val: images"]
    lines.append(f"nc: {len(subset_classes)}")
    names = ", ".join(f"'{c}'" for c in subset_classes)
    lines.append(f"names: [{names}]")
    (dst / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── main entry point ─────────────────────────────────────────────────────────
def export_subset(
    items: list[ExportItem],
    dst: Path,
    *,
    mode: str = "copy",
    layout: str = "yolo",
    source_roots: list[Path] | None = None,
    on_exists: str = "skip",
    pad: float = 0.0,
    drop_invalid_labels: bool = False,
) -> ExportReport:
    """Materialise ``items`` into ``dst`` as a YOLO subset (or crop-out
    classification set). See module docstring for the contract.

    Raises ``ValueError`` (writing nothing) when ``dst`` overlaps a source
    dataset root, or for an unknown ``mode`` / ``layout``. Everything else is
    accounted for in the returned :class:`ExportReport`.
    """
    if mode not in ("copy", "symlink", "manifest-only"):
        raise ValueError(f"unknown mode {mode!r}")
    if layout not in ("yolo", "crop-out"):
        raise ValueError(f"unknown layout {layout!r}")

    dst = Path(dst)
    report = ExportReport(dst=str(dst.resolve()))

    # 1) dst-containment hard guard — BEFORE writing anything.
    roots = ([Path(r) for r in source_roots] if source_roots
             else [_infer_source_root(it.image_path) for it in items])
    _check_containment(dst, roots)

    # 2) sha256 resolve + dedup (unresolved logged, excluded).
    merged = _resolve_and_dedup(items, report)

    # 3) class remap palette (by name union, stable).
    subset_classes, class_remap = _build_class_remap(items)

    dst.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    object_rows: list[dict] = []

    if layout == "crop-out":
        _export_crop_out(merged, dst, mode, pad, report, manifest_rows,
                         object_rows)
    else:
        _export_yolo(merged, dst, mode, on_exists, class_remap, report,
                     manifest_rows, drop_invalid_labels)

    # 4) provenance — always written, even for manifest-only.
    _write_manifest_csv(dst, manifest_rows)
    if layout == "crop-out":
        _write_object_manifest_csv(dst, object_rows)
    (dst / "classes.txt").write_text(
        "\n".join(subset_classes) + ("\n" if subset_classes else ""),
        encoding="utf-8")
    _write_data_yaml(dst, subset_classes, layout)
    _write_lineage(dst, mode=mode, layout=layout, on_exists=on_exists, pad=pad,
                   subset_classes=subset_classes, class_remap=class_remap,
                   shas=list(merged.keys()), manifest_rows=manifest_rows,
                   report=report)
    return report


def _export_yolo(merged, dst, mode, on_exists, class_remap, report,
                 manifest_rows, drop_invalid_labels) -> None:
    """yolo layout: whole image + FULL (remapped) label file (§6 path ①)."""
    img_root = dst / "images"
    lbl_root = dst / "labels"
    for sha, m in merged.items():
        it = m.item
        src_img = Path(it.image_path)
        ext = _norm_ext(src_img)
        split = it.split
        img_dir = img_root / split if split else img_root
        lbl_dir = lbl_root / split if split else lbl_root
        target_img = img_dir / f"{sha}{ext}"

        man_row = _manifest_row(sha, m, level=("object" if "object" in m.levels
                                               else "image"))
        if mode == "manifest-only":
            manifest_rows.append(man_row)
            continue

        final_img = _resolve_target(target_img, on_exists, report, sha)
        if final_img is None:
            continue
        if not _copy_or_link(src_img, final_img, mode, report, sha):
            continue
        report.exported += 1
        man_row["dst"] = str(final_img.relative_to(dst).as_posix())
        manifest_rows.append(man_row)

        # FULL label, class ids remapped — never subset/rewrite boxes (§6).
        if it.label_path and Path(it.label_path).exists():
            rows, warns = _parse_label_lines(
                Path(it.label_path).read_text(encoding="utf-8"))
            for w in warns:
                report.warnings.append({"sha256": sha, "kind": "label-parse",
                                        "detail": w})
            remap = class_remap.get(_dataset_key(it), {})
            text = _remap_label_text(rows, remap, report, sha,
                                     drop_invalid_labels)
            final_lbl = lbl_dir / f"{Path(final_img).stem}.txt"
            final_lbl.parent.mkdir(parents=True, exist_ok=True)
            final_lbl.write_text(text, encoding="utf-8")


def _export_crop_out(merged, dst, mode, pad, report, manifest_rows,
                     object_rows) -> None:
    """crop-out layout: one ImageFolder JPG per picked box (§6 path ②)."""
    from PIL import Image

    img_root = dst / "images"
    for sha, m in merged.items():
        it = m.item
        src_img = Path(it.image_path)
        man_row = _manifest_row(sha, m, level="object")
        manifest_rows.append(man_row)
        if mode == "manifest-only":
            continue
        if "object" not in m.levels or not it.label_path \
                or not Path(it.label_path).exists():
            # No boxes to crop — record it, no crop emitted.
            report.warnings.append({"sha256": sha, "kind": "crop-skip",
                                    "detail": "no object_ids/label for crop-out"})
            continue
        rows, warns = _parse_label_lines(
            Path(it.label_path).read_text(encoding="utf-8"))
        for w in warns:
            report.warnings.append({"sha256": sha, "kind": "label-parse",
                                    "detail": w})
        try:
            img = Image.open(src_img).convert("RGB")
        except OSError as e:
            report.errors.append({"sha256": sha, "src": str(src_img),
                                  "reason": f"crop open failed: {e}"})
            continue
        iw, ih = img.size
        names = it.class_names or []
        for oid in m.object_ids:
            if oid < 0 or oid >= len(rows):
                report.warnings.append({
                    "sha256": sha, "kind": "object-id-oob",
                    "detail": f"object_id {oid} out of label range {len(rows)}"})
                continue
            cls_id = int(rows[oid][0])
            cls_name = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
            box = rows[oid][1:5]
            left, top, right, bottom = _crop_box(iw, ih, box, pad)
            crop = img.crop((left, top, right, bottom))
            cls_dir = img_root / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            crop_name = f"{sha}_{oid}.jpg"
            crop_path = cls_dir / crop_name
            crop.save(crop_path, quality=92)
            report.exported += 1
            object_rows.append({
                "crop": str(crop_path.relative_to(dst).as_posix()),
                "crop_sha256": _file_sha256(crop_path),
                "source_sha256": sha,
                "source_path": str(src_img),
                "bbox": json.dumps(box),
                "class": cls_name,
                "pad": pad,
                "source_tag": "|".join(m.source_tags),
            })


def _manifest_row(sha: str, m: _Merged, level: str) -> dict:
    return {
        "dst": "",
        "sha256": sha,
        "source_path": str(Path(m.item.image_path)),
        "source_tool": "|".join(m.source_tools),
        "source_tag": "|".join(m.source_tags),
        "reason": "|".join(m.reasons),
        "split": m.item.split or "",
        "level": level,
    }
