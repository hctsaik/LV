"""Unified LV → Labeling hand-over: one writer, one reader, one registry.

Every LV feature (cart / label-disagreement / outlier / near-dup / diversity /
completeness-gap / compare-novel / gray-zone / quiz / viewer) hands a subset of
images to the Labeling tool through *this* module, so they all produce one
canonical, content-addressed artifact and read back through one parser.

Why a separate, framework-free module (no streamlit, like interaction.py /
manifest.py): it must be unit-testable without a browser, and — critically —
the hand-over is **asynchronous**. ``ToolProcessManager.start()`` stops the
current tool before starting the next, so when LV launches Labeling, LV itself
is torn down: nothing in ``st.session_state`` survives. The durable channel is
therefore the **handoff folder on disk** plus a ``_pending.json`` registry under
``CIM_LOG_DIR`` (shared by both tools on the same device). Read-back is a later,
user-initiated step after they navigate back to LV.

Contract (the join key in both directions is the image's **sha256**):

    <CIM_LOG_DIR>/lv_labeling_handoff/
      _pending.json                         registry of every handoff + status
      <source>_<ts>_<uid>/
        _handoff.json                       the typed task spec (see send_to_labeling)
        classes.txt                         label palette for xAnyLabeling
        images/<sha256>.<ext>               the subset (filename stem == sha256)
        images/<sha256>.json                ← written by Labeling (xAnyLabeling sidecar)
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import manifest as _manifest  # file_sha256 / compute_phash — framework-free

HANDOFF_ROOT_NAME = "lv_labeling_handoff"
HANDOFF_SPEC_NAME = "_handoff.json"
PENDING_NAME = "_pending.json"
RESULTS_NAME = "_results.json"

# Task types — the one abstraction that makes 10 features feel coherent. The
# value travels in _handoff.json so Labeling can hint the annotator's mode and
# LV can pick the right reconcile action on read-back.
TASK_RELABEL = "re-label"       # pick the correct class (quiz/disagreement/cart)
TASK_VERIFY = "verify"          # approve / reject / triage an existing label
TASK_ADJUDICATE = "adjudicate"  # binding boundary decision against an anchor (gray-zone)
TASK_FRESH = "fresh"            # no prior label: annotate from scratch (gaps/diversity/compare)
TASK_QUIZ = "blind-quiz"        # measurement only — never writes labels back

STATUS_SENT = "sent"
STATUS_ANNOTATING = "annotating"
STATUS_READ = "read_back"


# ── paths / registry ────────────────────────────────────────────────────────
def handoff_root(log_dir: str | os.PathLike | None = None) -> Path:
    base = (str(log_dir) if log_dir else None) or os.environ.get("CIM_LOG_DIR") \
        or str(Path(__file__).parent.parent / "output")
    return Path(base) / HANDOFF_ROOT_NAME


def _pending_path(log_dir=None) -> Path:
    return handoff_root(log_dir) / PENDING_NAME


def _load_pending(log_dir=None) -> dict:
    p = _pending_path(log_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_pending(reg: dict, log_dir=None) -> None:
    p = _pending_path(log_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        tmp.replace(p)
    except OSError:  # cross-device CIM_LOG_DIR mounts: fall back to a plain write
        p.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.unlink(missing_ok=True)


def update_pending(handoff_id: str, log_dir=None, **fields) -> None:
    reg = _load_pending(log_dir)
    reg.setdefault(handoff_id, {}).update(fields)
    _save_pending(reg, log_dir)


def delete_pending(handoff_id: str, log_dir=None) -> None:
    """Remove a handoff from the registry entirely (no tombstone) so the inbox
    count actually shrinks. Caller is responsible for rmtree-ing the folder."""
    reg = _load_pending(log_dir)
    if reg.pop(handoff_id, None) is not None:
        _save_pending(reg, log_dir)


def list_pending(log_dir=None) -> list[dict]:
    """All known handoffs, newest first, each enriched with live status counts."""
    reg = _load_pending(log_dir)
    out = []
    for hid, info in reg.items():
        row = dict(info, handoff_id=hid)
        d = Path(info.get("dir", ""))
        if d.exists():
            n_ann = _count_annotated(d)
            row["n_annotated"] = n_ann
            if info.get("status") != STATUS_READ:
                row["status"] = STATUS_ANNOTATING if n_ann else info.get("status", STATUS_SENT)
        out.append(row)
    out.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return out


def peek_pending_handoff(log_dir=None) -> dict | None:
    """Newest not-yet-read handoff — used by module_026 to auto-prefill the
    local-folder path so the curator doesn't paste it by hand."""
    for row in list_pending(log_dir):
        if row.get("status") != STATUS_READ and Path(row.get("images_dir", "")).exists():
            return row
    return None


# ── send ────────────────────────────────────────────────────────────────────
def _sha_for(rec: dict, manifest: dict | None) -> str:
    p = Path(rec["path"])
    if manifest:
        entry = manifest.get(str(p.resolve())) or manifest.get(str(p))
        if entry and entry.get("sha256"):
            return entry["sha256"]
    return _manifest.file_sha256(p)


def _item_set_hash(items: list[dict]) -> str:
    import hashlib
    h = hashlib.sha256()
    for s in sorted(it["sha256"] for it in items):
        h.update(s.encode())
    return h.hexdigest()[:16]


def send_to_labeling(
    records: list[dict],
    indices: Sequence[int],
    *,
    source: str,
    task: str,
    class_options: Sequence[str],
    manifest: dict | None = None,
    original_labels: dict[int, str] | None = None,
    golden_labels: dict[int, str] | None = None,
    candidate_labels: dict[int, list[str]] | None = None,
    payload: dict | None = None,
    skin_fn: Callable | None = None,
    skins: dict[int, str] | None = None,
    instructions: str = "",
    log_dir=None,
) -> Path | None:
    """Materialise a content-addressed handoff folder Labeling can ingest as a
    local dataset; register it; return its path. Pure: no streamlit, no engine.

    Returns None when ``indices`` is empty (caller shows a guard) — never writes
    an empty/phantom folder. Idempotent: re-sending the same (source, sha-set)
    while still open reuses the existing folder instead of duplicating it.
    """
    idxs = [i for i in indices if 0 <= i < len(records)]
    if not idxs:
        return None

    from PIL import Image  # local: only needed when actually exporting

    original_labels = original_labels or {}
    golden_labels = golden_labels or {}
    candidate_labels = candidate_labels or {}
    skins = skins or {}

    # pre-compute the item list (sha-keyed, de-duplicated by content)
    items: list[dict] = []
    seen: set[str] = set()
    for i in idxs:
        rec = records[i]
        sha = _sha_for(rec, manifest)
        if sha in seen:
            continue
        seen.add(sha)
        man_entry = (manifest or {}).get(str(Path(rec["path"]).resolve())) or {}
        items.append({
            "item_id": sha, "sha256": sha,
            "phash": man_entry.get("phash"),
            "lv_index": int(i),
            "filename": Path(rec["path"]).name,
            "split": rec.get("split", ""),
            "original_label": original_labels.get(i, rec.get("label", "")),
            "golden_label": golden_labels.get(i),
            "candidate_labels": candidate_labels.get(i),
            "skin": skins.get(i),
            "_src": str(Path(rec["path"])),
        })
    if not items:
        return None

    # idempotency: an open handoff with the same source + content set → reuse it
    set_hash = _item_set_hash(items)
    for row in list_pending(log_dir):
        if (row.get("source") == source and row.get("set_hash") == set_hash
                and row.get("status") != STATUS_READ and Path(row.get("dir", "")).exists()):
            return Path(row["dir"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    hid = f"{source}_{ts}_{uuid.uuid4().hex[:6]}"
    out = handoff_root(log_dir) / hid
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for it in items:
        src = Path(it.pop("_src"))
        ext = src.suffix.lower() if src.suffix.lower() in (".jpg", ".jpeg", ".png") else ".jpg"
        dst = img_dir / f"{it['sha256']}{ext}"
        try:
            if it.get("skin") and skin_fn is not None:
                img = Image.open(src).convert("RGB")
                skin_fn(img, it["skin"]).save(dst, quality=92)
            elif ext in (".jpg", ".jpeg"):
                shutil.copyfile(src, dst)
            else:
                Image.open(src).convert("RGB").save(dst)
        except Exception:  # noqa: BLE001  skip one bad image; never orphan a half-folder
            continue
        it["image"] = f"images/{dst.name}"

    items = [it for it in items if it.get("image")]
    if not items:
        shutil.rmtree(out, ignore_errors=True)
        return None

    (out / "classes.txt").write_text("\n".join(class_options) + "\n", encoding="utf-8")
    spec = {
        "handoff_id": hid, "version": 1, "source": source, "task": task,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "class_options": list(class_options),
        "instructions": instructions,
        "payload": payload or {},
        "items": items,
    }
    (out / HANDOFF_SPEC_NAME).write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    update_pending(hid, log_dir=log_dir, source=source, task=task,
                   dir=str(out), images_dir=str(img_dir),
                   created_at=spec["created_at"], status=STATUS_SENT,
                   n_total=len(items), set_hash=set_hash)
    return out


# ── read-back ───────────────────────────────────────────────────────────────
def _label_from_sidecar(ann_path: Path) -> tuple[str | None, str | None]:
    """(label, annotator) from one xAnyLabeling/LabelMe sidecar JSON."""
    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    label = None
    shapes = data.get("shapes")
    if isinstance(shapes, list) and shapes:
        label = shapes[0].get("label")
    if not label and data.get("label"):
        label = data["label"]
    if not label and isinstance(data.get("flags"), dict):
        on = [k for k, v in data["flags"].items() if v]
        label = on[0] if on else None
    return label, data.get("annotator") or data.get("annotated_by")


def _count_annotated(handoff_dir: Path) -> int:
    try:
        spec = json.loads((handoff_dir / HANDOFF_SPEC_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    img_dir = handoff_dir / "images"
    n = 0
    for it in spec.get("items", []):
        ann = (img_dir / Path(it["image"]).name).with_suffix(".json")
        if ann.exists():
            lbl, _ = _label_from_sidecar(ann)
            if lbl:
                n += 1
    return n


def read_labeling_results(handoff_dir: str | os.PathLike) -> dict[str, dict]:
    """Return {sha256 -> {label, status, annotator, filename, original_label}}
    by reading each item's xAnyLabeling sidecar (then image-level label/flags,
    then an answers.csv fallback). Keyed by sha256 so callers reconcile back to
    their records rename/move-proof."""
    handoff_dir = Path(handoff_dir)
    spec = json.loads((handoff_dir / HANDOFF_SPEC_NAME).read_text(encoding="utf-8"))
    img_dir = handoff_dir / "images"
    out: dict[str, dict] = {}

    csv_map: dict[str, str] = {}
    csv_path = handoff_dir / "answers.csv"
    if csv_path.exists():
        import csv as _csv
        import io as _io
        rows = list(_csv.reader(_io.StringIO(csv_path.read_text(encoding="utf-8"))))
        for parts in rows[1:]:  # csv.reader handles commas-in-labels / quoting
            if len(parts) >= 2 and parts[0].strip():
                csv_map[parts[0].strip()] = parts[1].strip()

    for it in spec.get("items", []):
        sha = it["sha256"]
        ann = (img_dir / Path(it["image"]).name).with_suffix(".json")
        label, annotator = (_label_from_sidecar(ann) if ann.exists() else (None, None))
        if not label and sha in csv_map:
            label = csv_map[sha]
        out[sha] = {
            "label": label,
            "status": "annotated" if label else "pending",
            "annotator": annotator,
            "filename": it.get("filename"),
            "original_label": it.get("original_label"),
            "golden_label": it.get("golden_label"),
            "lv_index": it.get("lv_index"),
        }
    return out


def handoff_status(handoff_dir: str | os.PathLike) -> dict:
    """{n_total, n_annotated} for progress display without reopening Labeling."""
    handoff_dir = Path(handoff_dir)
    try:
        spec = json.loads((handoff_dir / HANDOFF_SPEC_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"n_total": 0, "n_annotated": 0}
    return {"n_total": len(spec.get("items", [])),
            "n_annotated": _count_annotated(handoff_dir)}


def reconcile_to_records(results: dict[str, dict], sha_to_index: dict[str, int]) -> dict[int, dict]:
    """Map sha256-keyed read-back results onto current LV record indices, content-
    addressed. Results whose sha256 is absent from the current run are dropped
    (a handoff made on a different dataset degrades, never mis-attaches)."""
    out: dict[int, dict] = {}
    for sha, res in results.items():
        idx = sha_to_index.get(sha)
        if idx is not None and res.get("label"):
            out[idx] = res
    return out


def load_spec(handoff_dir: str | os.PathLike) -> dict:
    return json.loads((Path(handoff_dir) / HANDOFF_SPEC_NAME).read_text(encoding="utf-8"))
