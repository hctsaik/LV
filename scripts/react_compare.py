"""Framework-free local comparison pipeline for the LV React interface."""
from __future__ import annotations

import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from _utils import extract_embeddings, load_model
from interaction import discover_yolo_objects
from object_eval import DEFAULT_CANDIDATES, DEFAULT_POLICY, classes_for, embed_objects, run_autotune
from react_visualize import JobCancelled, _project_embeddings
from safe_io import partition_readable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CACHE_PARTS = ("embeddings_", "object_crops", ".thumbs")
PROJECTION_METHODS = {"pca", "tsne", "umap", "lda", "sumap"}
WHOLE_IMAGE_LABEL = "(全部影像)"


class ComparisonInputError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise JobCancelled()


def _resolve_folders(values: Sequence[str | Path], side: str) -> list[Path]:
    if not values or len(values) > 32:
        raise ComparisonInputError(
            f"FOLDERS_{side}_REQUIRED",
            f"Choose between one and 32 folders for group {side}.",
        )
    folders: list[Path] = []
    seen: set[str] = set()
    for value in values:
        try:
            folder = Path(value).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ComparisonInputError(
                f"INVALID_FOLDER_{side}", f"A selected folder in group {side} is unavailable."
            ) from exc
        if not folder.is_dir():
            raise ComparisonInputError(
                f"INVALID_FOLDER_{side}", f"A selected path in group {side} is not a folder."
            )
        key = os.path.normcase(str(folder))
        if key not in seen:
            seen.add(key)
            folders.append(folder)
    return folders


def _images_in_folder(folder: Path) -> tuple[list[Path], str | None]:
    # Preserve the current Streamlit rule: direct image files win; if none are
    # present, scan nested folders while ignoring app-generated caches.
    direct = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if direct:
        return direct, None
    nested = sorted(
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.startswith(CACHE_PARTS) for part in path.parts)
    )
    if nested:
        return nested, f"Found {len(nested)} images in nested folders."
    return [], None


def _resolve_group(values: Sequence[str | Path], side: str):
    folders = _resolve_folders(values, side)
    paths: list[Path] = []
    notes: list[str] = []
    seen: set[str] = set()
    for folder in folders:
        found, note = _images_in_folder(folder)
        if note:
            notes.append(note)
        for path in found:
            token = os.path.normcase(str(path.resolve()))
            if token not in seen:
                seen.add(token)
                paths.append(path.resolve())
    readable, unreadable = partition_readable(paths)
    if not readable:
        code = f"NO_IMAGES_{side}"
        message = f"Group {side} has no readable .jpg, .jpeg, or .png images."
        raise ComparisonInputError(code, message)
    if unreadable:
        notes.append(f"Skipped {len(unreadable)} unreadable image(s).")
    name = folders[0].name if len(folders) == 1 else f"{side} ({len(folders)} folders)"
    return folders, readable, name, notes, len(unreadable)


def _content_keys(paths: Sequence[Path]) -> list[str]:
    output = []
    for path in paths:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise ComparisonInputError(
                "IMAGE_UNAVAILABLE", "A selected image could not be read. Run the comparison again."
            ) from exc
        output.append(digest.hexdigest())
    return output


def _cache_root() -> Path:
    repo = Path(__file__).resolve().parent.parent
    return Path(os.environ.get("LV_CACHE_DIR") or (repo / ".lv_cache")) / "react_web" / "compare"


def _cache_key(paths: Sequence[Path], model: str, unit: str, content_keys=None) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(unit.encode("utf-8"))
    for path, key in zip(paths, content_keys or _content_keys(paths)):
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(key.encode("ascii"))
    return digest.hexdigest()


def _group_embedding(paths, model, loader, cache_path, progress, offset, total, cancelled, content_keys=None):
    keys = content_keys or _content_keys(paths)
    embed_fn = loader(model)

    def on_item(done: int, count: int) -> None:
        _check_cancelled(cancelled)
        if progress is not None:
            progress("embedding", offset + done, total,
                     f"Extracting group embeddings ({offset + done}/{total})")

    embeddings = extract_embeddings(
        list(paths), embed_fn, cache_path=cache_path,
        cache_keys=keys, progress_cb=on_item,
    )
    if embeddings.ndim != 2 or len(embeddings) != len(paths) or not np.isfinite(embeddings).all():
        raise ComparisonInputError("INVALID_EMBEDDINGS", "The selected model returned invalid embeddings.")
    if not np.all(np.linalg.norm(embeddings, axis=1) > 0):
        raise ComparisonInputError("INVALID_EMBEDDINGS", "The selected model returned an empty embedding.")
    return embeddings.astype(np.float32, copy=False)


def _public_records(paths, labels, *, object_meta=None, short_px=None):
    records = []
    for index, path in enumerate(paths):
        item = {
            "index": index,
            "filename": path.name,
            "label": str(labels[index]),
            "source_id": hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:20],
        }
        if object_meta is not None:
            meta = object_meta[index]
            item.update({
                "class_id": int(meta["class_id"]),
                "bbox": [float(v) for v in meta["bbox"]],
                "score": None if meta.get("score") is None else float(meta["score"]),
                "short_px": int(round(float(short_px[index]))),
            })
        else:
            item.update({"class_id": None, "bbox": None, "score": None, "short_px": None})
        records.append(item)
    return records


def _centroid_drift(a: np.ndarray, b: np.ndarray) -> float:
    mean_a = np.asarray(a, dtype=np.float64).mean(axis=0)
    mean_b = np.asarray(b, dtype=np.float64).mean(axis=0)
    mean_a /= np.linalg.norm(mean_a) + 1e-9
    mean_b /= np.linalg.norm(mean_b) + 1e-9
    return round(max(0.0, 1.0 - float(mean_a @ mean_b)), 4)


def run_comparison(
    folders_a: Sequence[str | Path],
    folders_b: Sequence[str | Path],
    model: str,
    *,
    unit: str = "object",
    class_names: Sequence[str] | None = None,
    model_loader: Callable[[str], Callable] = load_model,
    object_model_loader: Callable | None = None,
    progress: Callable[[str, int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Run current LV Compare semantics without importing the Streamlit UI."""
    if unit not in {"whole", "object"}:
        raise ComparisonInputError("INVALID_UNIT", "Choose whole-image or object-by-class comparison.")
    roots_a, paths_a, name_a, notes_a, _bad_a = _resolve_group(folders_a, "A")
    roots_b, paths_b, name_b, notes_b, _bad_b = _resolve_group(folders_b, "B")
    _check_cancelled(cancelled)
    overlap = len({str(p).casefold() for p in paths_a} & {str(p).casefold() for p in paths_b})
    warnings = notes_a + notes_b
    if overlap:
        warnings.append(
            f"Groups A and B share {overlap} image(s); the drift may be understated."
        )

    if unit == "whole":
        total = len(paths_a) + len(paths_b)
        root = _cache_root() / "whole" / model
        keys_a, keys_b = _content_keys(paths_a), _content_keys(paths_b)
        embedding_a = _group_embedding(
            paths_a, model, model_loader,
            root / (_cache_key(paths_a, model, unit, keys_a) + ".npz"),
            progress, 0, total, cancelled, keys_a,
        )
        embedding_b = _group_embedding(
            paths_b, model, model_loader,
            root / (_cache_key(paths_b, model, unit, keys_b) + ".npz"),
            progress, len(paths_a), total, cancelled, keys_b,
        )
        labels_a = [WHOLE_IMAGE_LABEL] * len(paths_a)
        labels_b = [WHOLE_IMAGE_LABEL] * len(paths_b)
        records_a = _public_records(paths_a, labels_a)
        records_b = _public_records(paths_b, labels_b)
        record_paths_a, record_paths_b = paths_a, paths_b
        rows = [{
            "label": WHOLE_IMAGE_LABEL,
            "a_count": len(paths_a),
            "b_count": len(paths_b),
            "drift": _centroid_drift(embedding_a, embedding_b),
        }]
        policy = None
        autotune_rows = []
        a_only: dict[str, int] = {}
        b_only: dict[str, int] = {}
    else:
        names_override = [str(name).strip() for name in (class_names or []) if str(name).strip()]
        names_a = names_override or classes_for(roots_a[0])
        names_b = names_override or classes_for(roots_b[0])
        meta_a = discover_yolo_objects(paths_a, names_a)
        meta_b = discover_yolo_objects(paths_b, names_b)
        if not meta_a:
            raise ComparisonInputError(
                "NO_OBJECTS_A", "Group A has no supported detection boxes. Use a detection dataset or switch to whole-image comparison."
            )
        if not meta_b:
            raise ComparisonInputError(
                "NO_OBJECTS_B", "Group B has no supported detection boxes. Use a detection dataset or switch to whole-image comparison."
            )

        small_paths = paths_a if len(paths_a) <= len(paths_b) else paths_b
        small_names = names_a if len(paths_a) <= len(paths_b) else names_b
        small_meta = meta_a if len(paths_a) <= len(paths_b) else meta_b
        tuned_count = min(800, len(small_meta))
        full_total = len(meta_a) + len(meta_b)
        autotune_total = len(DEFAULT_CANDIDATES) * tuned_count
        total = max(1, autotune_total + full_total)
        content_a = _cache_key(paths_a, model, "object")
        content_b = _cache_key(paths_b, model, "object")
        object_loader = object_model_loader or load_model
        tune_dir = _cache_root() / "object" / model / hashlib.sha256(
            (content_a + content_b).encode("ascii")
        ).hexdigest() / "autotune"

        def tune_progress(candidate, candidate_count, policy_value, done, count):
            _check_cancelled(cancelled)
            if progress is not None:
                stage_done = min(candidate_count * tuned_count, candidate * tuned_count + done)
                progress("autotune", stage_done, total,
                         f"Tuning object settings ({candidate + 1}/{candidate_count}) · {done}/{count}")

        autotune_rows, policy, _fingerprint = run_autotune(
            small_paths, small_names, model=model, cache_dir=tune_dir,
            progress=tune_progress, model_loader=object_loader,
        )
        if not policy:
            policy = dict(DEFAULT_POLICY)
        if autotune_rows:
            best = autotune_rows[0].get("macro_purity")
            if best is None:
                warnings.append("Object settings used the best available fallback policy; no class purity score was available.")
        embeddings = []
        records = []
        offset = autotune_total
        for side, paths, meta, names, cache_key in (
            ("A", paths_a, meta_a, names_a, content_a),
            ("B", paths_b, meta_b, names_b, content_b),
        ):
            _check_cancelled(cancelled)
            cache_path = _cache_root() / "object" / model / cache_key / (
                f"{policy['pad']}_{policy['target_res']}_{policy['head']}.npz"
            )
            local_done = 0

            def on_object(done, count, current_side=side):
                nonlocal local_done
                _check_cancelled(cancelled)
                local_done = done
                if progress is not None:
                    progress("embedding", offset + done, total,
                             f"Embedding {current_side} objects ({done}/{count})")

            emb, short = embed_objects(
                meta, model, policy, cache_path=cache_path, progress=on_object,
                model_loader=object_loader,
            )
            if emb.ndim != 2 or len(emb) != len(meta) or not np.isfinite(emb).all():
                raise ComparisonInputError("INVALID_EMBEDDINGS", "The selected model returned invalid object embeddings.")
            embeddings.append(emb.astype(np.float32, copy=False))
            records.append(_public_records(
                [Path(item["image_path"]).resolve() for item in meta],
                [item["label"] for item in meta], object_meta=meta, short_px=short,
            ))
            offset += len(meta)
        embedding_a, embedding_b = embeddings
        records_a, records_b = records
        record_paths_a = [Path(item["image_path"]).resolve() for item in meta_a]
        record_paths_b = [Path(item["image_path"]).resolve() for item in meta_b]
        counts_a, counts_b = Counter(item["label"] for item in records_a), Counter(
            item["label"] for item in records_b
        )
        a_only = {label: counts_a[label] for label in sorted(counts_a.keys() - counts_b.keys())}
        b_only = {label: counts_b[label] for label in sorted(counts_b.keys() - counts_a.keys())}
        common = sorted(counts_a.keys() & counts_b.keys())
        if not common:
            raise ComparisonInputError(
                "NO_COMMON_CLASSES",
                "Groups A and B have no shared detection classes. Check the class names or use the same classes.txt.",
            )
        rows = []
        for label in common:
            idx_a = [i for i, item in enumerate(records_a) if item["label"] == label]
            idx_b = [i for i, item in enumerate(records_b) if item["label"] == label]
            rows.append({
                "label": label,
                "a_count": len(idx_a),
                "b_count": len(idx_b),
                "drift": _centroid_drift(embedding_a[idx_a], embedding_b[idx_b]),
            })
        rows.sort(key=lambda row: row["drift"], reverse=True)

    return {
        "unit": unit,
        "model": model,
        "name_a": name_a,
        "name_b": name_b,
        "records_a": records_a,
        "records_b": records_b,
        "rows": rows,
        "a_only": a_only,
        "b_only": b_only,
        "policy": policy,
        "autotune_rows": [
            {
                "policy": row["policy"],
                "macro_purity": row.get("macro_purity"),
                "buckets": row.get("buckets", {}),
            }
            for row in autotune_rows
        ],
        "warnings": warnings,
        "_paths_a": paths_a,
        "_paths_b": paths_b,
        "_record_paths_a": record_paths_a,
        "_record_paths_b": record_paths_b,
        "_embeddings_a": embedding_a,
        "_embeddings_b": embedding_b,
    }


def project_comparison(
    comparison: dict,
    *,
    label: str,
    method: str,
    dimension: int = 2,
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
    min_short_px: int = 0,
) -> dict:
    if method not in PROJECTION_METHODS:
        raise ComparisonInputError("METHOD_UNAVAILABLE", "The selected projection method is unavailable.")
    if dimension not in {2, 3}:
        raise ComparisonInputError("INVALID_DIMENSION", "Choose a 2D or 3D projection.")
    if not (0.0 <= min_confidence <= max_confidence <= 1.0) or min_short_px < 0:
        raise ComparisonInputError("INVALID_FILTER", "The object filter is invalid.")

    keep_a = [
        i for i, row in enumerate(comparison["records_a"])
        if row["label"] == label
        and (row.get("score") is None or min_confidence <= row["score"] <= max_confidence)
        and (row.get("short_px") is None or row["short_px"] >= min_short_px)
    ]
    keep_b = [
        i for i, row in enumerate(comparison["records_b"])
        if row["label"] == label
        and (row.get("score") is None or min_confidence <= row["score"] <= max_confidence)
        and (row.get("short_px") is None or row["short_px"] >= min_short_px)
    ]
    if not keep_a and not keep_b:
        raise ComparisonInputError("NO_FILTERED_OBJECTS", "The current filters hide every object in this class.")
    emb_a = comparison["_embeddings_a"][keep_a]
    emb_b = comparison["_embeddings_b"][keep_b]
    combined = np.vstack([emb_a, emb_b])
    groups = ["A"] * len(emb_a) + ["B"] * len(emb_b)
    projections, warnings = _project_embeddings(combined, groups, [method])
    actual_method = method if method in projections else next(iter(projections))
    coordinates = projections[actual_method]
    if coordinates.shape[1] < dimension:
        coordinates = np.pad(coordinates, ((0, 0), (0, dimension - coordinates.shape[1])))
    coordinates = coordinates[:, :dimension]
    points = []
    cursor = 0
    for side, indices, rows in (("A", keep_a, comparison["records_a"]),
                                ("B", keep_b, comparison["records_b"])):
        for index in indices:
            record = rows[index]
            point = {
                "group": side,
                "index": index,
                "filename": record["filename"],
                "label": record["label"],
                "x": float(coordinates[cursor, 0]),
                "y": float(coordinates[cursor, 1]),
                "z": float(coordinates[cursor, 2]) if dimension == 3 else 0.0,
            }
            points.append(point)
            cursor += 1
    return {
        "label": label,
        "method": actual_method,
        "requested_method": method,
        "dimension": dimension,
        "points": points,
        "warnings": warnings,
        "a_count": len(keep_a),
        "b_count": len(keep_b),
    }
