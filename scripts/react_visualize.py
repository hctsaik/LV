"""Framework-free LV React visualization pipeline for LV-WEB-01-IC-01."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from _utils import extract_embeddings, load_model
from manifest import (
    manifest_path_for,
    rel_key,
    set_embedding_refs,
    update_manifest,
    write_manifest,
)
from safe_io import partition_readable, safe_read_text
from visualize_embeddings import discover_images, discover_images_classifier


class JobCancelled(Exception):
    """Raised when a web job observes its cancellation request."""


class VisualizationInputError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _is_under(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def _project_embeddings(embeddings: np.ndarray, labels: Sequence[str], methods: Sequence[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Project image embeddings with the same parameters as the legacy Visualize flow."""
    from sklearn.manifold import TSNE

    allowed = {"pca", "tsne", "umap", "lda", "sumap"}
    n_samples = len(embeddings)
    n_components = min(3, max(1, n_samples - 2))
    projections: dict[str, np.ndarray] = {}
    warnings: list[str] = []

    def pad_to_2d(values: np.ndarray) -> np.ndarray:
        return values if values.shape[1] >= 2 else np.hstack([values, np.zeros((len(values), 1))])

    class_names = np.unique(np.asarray(labels))
    label_ids = np.searchsorted(class_names, np.asarray(labels))
    for method in methods:
        if method not in allowed:
            raise VisualizationInputError("METHOD_UNAVAILABLE", "A selected projection method is unavailable.")
        if method in {"tsne", "umap", "lda", "sumap"} and n_samples < 4:
            warnings.append(f"{method} skipped because at least four images are required.")
            continue
        if method in {"lda", "sumap"} and len(class_names) < 2:
            warnings.append(f"{method} skipped because at least two classes are required.")
            continue
        try:
            if method == "pca":
                coordinates = PCA(n_components=n_components, random_state=42).fit_transform(embeddings)
            elif method == "tsne":
                perplexity = min(30, max(1, n_samples - 1))
                coordinates = TSNE(
                    n_components=n_components,
                    random_state=42,
                    perplexity=perplexity,
                ).fit_transform(embeddings)
            elif method == "umap":
                import umap

                neighbors = min(15, max(2, n_samples - 1))
                coordinates = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=neighbors,
                    random_state=42,
                ).fit_transform(embeddings)
            elif method == "lda":
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

                components = min(n_components, len(class_names) - 1)
                if components < 1:
                    warnings.append("lda skipped because the labels have one class.")
                    continue
                coordinates = LinearDiscriminantAnalysis(n_components=components).fit_transform(
                    embeddings, label_ids
                )
            else:
                import umap

                neighbors = min(15, max(2, n_samples - 1))
                coordinates = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=neighbors,
                    random_state=42,
                ).fit_transform(embeddings, y=label_ids)
            coordinates = np.asarray(coordinates, dtype=np.float32)
            if coordinates.ndim != 2 or coordinates.shape[0] != n_samples:
                warnings.append(f"{method} did not return one coordinate row per image.")
                continue
            if not np.isfinite(coordinates).all():
                warnings.append(f"{method} returned invalid coordinates.")
                continue
            projections[method] = pad_to_2d(coordinates)
        except (ImportError, ModuleNotFoundError):
            warnings.append(f"{method} is unavailable in this local runtime.")
        except ValueError as exc:
            # Preserve usable methods and report a projection-specific boundary.
            # The legacy UI also skips undefined small-sample projections.
            warnings.append(f"{method} could not be computed: {str(exc).splitlines()[0][:180]}")

    if not projections:
        projections["pca"] = pad_to_2d(
            PCA(n_components=n_components, random_state=42).fit_transform(embeddings)
        )
        warnings.append("PCA was used because none of the selected projections were available.")
    return projections, warnings


def _detector_class_names(folders: Sequence[Path], manual: Sequence[str]) -> list[str]:
    """Resolve detector labels with legacy classes-file precedence.

    The legacy UI checks the dataset-adjacent classes.txt before its manual
    names field. A shared taxonomy must resolve identically for every selected
    split; silently mixing class-id maps would corrupt point labels.
    """
    detected: list[list[str]] = []
    for folder in folders:
        names: list[str] | None = None
        for candidate in (folder.parent / "classes.txt", folder / "classes.txt", folder.parent.parent / "classes.txt"):
            if not candidate.is_file():
                continue
            lines = [line.strip() for line in safe_read_text(candidate).splitlines() if line.strip()]
            if lines:
                names = lines
                break
        if names is not None:
            detected.append(names)
    if detected:
        first = detected[0]
        if any(names != first for names in detected[1:]):
            raise VisualizationInputError(
                "CLASS_NAMES_MISMATCH",
                "Selected detector folders use different classes.txt taxonomies.",
            )
        return first

    fallback = list(dict.fromkeys(name.strip() for name in manual if name.strip()))
    if fallback:
        return fallback
    raise VisualizationInputError(
        "CLASS_NAMES_UNAVAILABLE",
        "No classes.txt was found. Enter class names separated by commas to continue.",
    )


def run_image_visualization(
    folders: Sequence[str | Path],
    model_name: str | None = None,
    *,
    model_names: Sequence[str] | None = None,
    methods: Sequence[str] = ("pca",),
    mode: str = "classifier",
    unit: str = "image",
    class_names: Sequence[str] = (),
    object_policy: dict | None = None,
    model_loader: Callable[[str], Callable[[Path], np.ndarray]] = load_model,
    object_model_loader: Callable[..., Callable] | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Run a Visualize flow without Streamlit state.

    Discovery, per-model content-hash embedding caches, projection parameters,
    object crops, and label/split alignment follow the existing Visualize path.
    Every write is routed through LV's app-side manifest/cache helpers, never
    into selected source folders.
    """
    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise JobCancelled()

    resolved_folders: list[Path] = []
    seen: set[str] = set()
    for value in folders:
        folder = Path(value).expanduser()
        try:
            folder = folder.resolve(strict=True)
        except OSError as exc:
            raise VisualizationInputError("INVALID_FOLDER", "A selected folder is unavailable.") from exc
        if not folder.is_dir():
            raise VisualizationInputError("INVALID_FOLDER", "A selected folder is not a directory.")
        key = str(folder).casefold()
        if key not in seen:
            seen.add(key)
            resolved_folders.append(folder)
    if not resolved_folders:
        raise VisualizationInputError("NO_FOLDERS", "Choose at least one image folder.")

    selected_models = list(model_names or ([model_name] if model_name else []))
    selected_models = list(dict.fromkeys(selected_models))
    if not selected_models or len(selected_models) > 8:
        raise VisualizationInputError("MODEL_UNAVAILABLE", "Choose between one and eight installed models.")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in selected_models):
        raise VisualizationInputError("MODEL_UNAVAILABLE", "A selected model is unavailable.")

    selected_methods = list(dict.fromkeys(methods))
    if not selected_methods or len(selected_methods) > 5:
        raise VisualizationInputError("METHOD_UNAVAILABLE", "Choose at least one supported projection method.")
    if any(method not in {"pca", "tsne", "umap", "lda", "sumap"} for method in selected_methods):
        raise VisualizationInputError("METHOD_UNAVAILABLE", "A selected projection method is unavailable.")

    if unit not in {"image", "object"}:
        raise VisualizationInputError("INVALID_UNIT", "Choose whole-image or object-level analysis.")
    if unit == "object" and mode != "detector":
        raise VisualizationInputError("INVALID_UNIT", "Object-level analysis requires Object Detector mode.")

    resolved_object_policy: dict | None = None
    if unit == "object":
        from object_eval import DEFAULT_POLICY

        resolved_object_policy = dict(DEFAULT_POLICY)
        if object_policy is not None:
            resolved_object_policy.update(object_policy)
        if (
            not isinstance(resolved_object_policy.get("pad"), (int, float))
            or not 0 <= float(resolved_object_policy["pad"]) <= 0.5
            or resolved_object_policy.get("target_res") not in {168, 196, 224, 280, 336, 392, 448}
            or resolved_object_policy.get("head") not in {"cls", "meanpool"}
        ):
            raise VisualizationInputError("INVALID_OBJECT_POLICY", "The object crop policy is invalid.")

    check_cancelled()
    if mode == "classifier":
        resolved_class_names: list[str] = []
        records = discover_images_classifier(resolved_folders, allow_flat=True)
    elif mode == "detector":
        resolved_class_names = _detector_class_names(resolved_folders, class_names)
        invalid = [folder for folder in resolved_folders
                   if not (folder / "images").is_dir()
                   and not (folder / "JPEGImages").is_dir()]
        if invalid:
            raise VisualizationInputError(
                "INVALID_DETECTOR_DATASET",
                "Each detector folder must contain an images/ or JPEGImages/ directory.",
            )
        records = discover_images(resolved_folders, resolved_class_names)
    else:
        raise VisualizationInputError("INVALID_MODE", "Choose Image Classifier or Object Detector.")
    if not records:
        raise VisualizationInputError("NO_IMAGES", "No images were found in the selected folders.")

    readable, unreadable = partition_readable([record["path"] for record in records])
    readable_keys = {str(path.resolve()) for path in readable}
    records = [record for record in records if str(Path(record["path"]).resolve()) in readable_keys]
    if not records:
        raise VisualizationInputError("NO_IMAGES", "No readable images were found in the selected folders.")
    source_records = records
    object_records: list[dict] | None = None
    if unit == "object":
        from interaction import discover_yolo_objects

        object_records = discover_yolo_objects(
            [Path(record["path"]) for record in source_records],
            resolved_class_names,
        )
        if not object_records:
            raise VisualizationInputError("NO_OBJECTS", "No supported annotated objects were found in the selected folders.")
        source_by_path = {str(Path(record["path"]).resolve()): record for record in source_records}
        records = [
            {
                "path": Path(item["image_path"]),
                "split": str(source_by_path[str(Path(item["image_path"]).resolve())]["split"]),
                "label": str(item["label"]),
                "class_id": int(item["class_id"]) if item.get("class_id") is not None else None,
                "bbox": [float(value) for value in item["bbox"]],
                "obj_index": int(item["obj_index"]),
                "score": float(item["score"]) if item.get("score") is not None else None,
            }
            for item in object_records
        ]
    check_cancelled()

    manifest_by_folder: dict[Path, dict[str, dict]] = {}
    paths_by_folder: dict[Path, list[Path]] = {}
    for folder in resolved_folders:
        folder_records = [record for record in source_records if _is_under(Path(record["path"]), folder)]
        if not folder_records:
            continue
        manifest_by_folder[folder] = update_manifest(folder, folder_records)
        paths_by_folder[folder] = [Path(record["path"]) for record in folder_records]

    completed = 0
    total = len(records) * len(selected_models)
    embeddings_by_model: dict[str, np.ndarray] = {}
    projections_by_model: dict[str, dict[str, np.ndarray]] = {}
    embedding_shapes: dict[str, list[int]] = {}
    warnings: list[str] = []
    labels = [str(record["label"]) for record in records]
    if mode == "classifier":
        resolved_class_names = sorted(set(labels))

    for selected_model in selected_models:
        check_cancelled()
        def on_item(done: int, _item_total: int) -> None:
            check_cancelled()
            if progress is not None:
                progress(completed + done, total)

        if unit == "object" and object_records is not None and resolved_object_policy is not None:
            from object_eval import embed_objects, policy_tag

            cache_keys = []
            for record in records:
                path = Path(record["path"]).resolve()
                folder = next(folder for folder in resolved_folders if _is_under(path, folder))
                source_sha = manifest_by_folder[folder][rel_key(folder, path)]["sha256"]
                key_data = {
                    "path": str(path),
                    "sha256": source_sha,
                    "bbox": record["bbox"],
                    "obj_index": record["obj_index"],
                    "model": selected_model,
                    "policy": resolved_object_policy,
                }
                cache_keys.append(hashlib.sha256(
                    json.dumps(key_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest())
            cache_path = (
                manifest_path_for(resolved_folders[0]).parent
                / f"object_embeddings_{selected_model}_{policy_tag(resolved_object_policy)}"
                / "embeddings.npz"
            )
            raw, _short_px = embed_objects(
                object_records,
                selected_model,
                resolved_object_policy,
                cache_path=cache_path,
                progress=on_item,
                model_loader=object_model_loader or model_loader,
                cache_keys=cache_keys,
            )
            completed += len(records)
        else:
            embed_fn = model_loader(selected_model)
            all_embeddings: list[np.ndarray] = []
            for folder in resolved_folders:
                paths = paths_by_folder.get(folder, [])
                if not paths:
                    continue
                check_cancelled()
                entries = manifest_by_folder[folder]
                keys = [entries[rel_key(folder, path)]["sha256"] for path in paths]
                cache_path = manifest_path_for(folder).parent / f"embeddings_{selected_model}" / "embeddings.npz"

                folder_embeddings = extract_embeddings(
                    paths,
                    embed_fn,
                    cache_path=cache_path,
                    progress_cb=on_item,
                    cache_keys=keys,
                )
                set_embedding_refs(entries, folder, selected_model, paths)
                all_embeddings.append(folder_embeddings)
                completed += len(paths)
            raw = np.vstack(all_embeddings).astype(np.float32, copy=False)
        if raw.ndim != 2 or raw.shape[0] != len(records) or not np.isfinite(raw).all():
            raise VisualizationInputError("INVALID_EMBEDDINGS", "The selected model returned invalid embeddings.")
        nonzero = bool(np.all(np.linalg.norm(raw, axis=1) > 0))
        if not nonzero:
            raise VisualizationInputError("INVALID_EMBEDDINGS", "The selected model returned an empty embedding.")
        embeddings_by_model[selected_model] = raw
        embedding_shapes[selected_model] = [int(raw.shape[0]), int(raw.shape[1])]
        projections, projection_warnings = _project_embeddings(raw, labels, selected_methods)
        projections_by_model[selected_model] = projections
        warnings.extend(f"{selected_model}: {warning}" for warning in projection_warnings)
        check_cancelled()

    for folder, entries in manifest_by_folder.items():
        write_manifest(folder, entries)

    public_records = [
        {
            "index": index,
            "filename": Path(record["path"]).name,
            "split": str(record["split"]),
            "label": str(record["label"]),
            **({
                "class_id": record["class_id"],
                "bbox": record["bbox"],
                "obj_index": record["obj_index"],
                "score": record["score"],
            } if unit == "object" else {}),
        }
        for index, record in enumerate(records)
    ]

    views: list[dict] = []
    for selected_model in selected_models:
        for method, coordinates in projections_by_model[selected_model].items():
            points = [
                {
                    **public_records[index],
                    "x": float(coordinates[index, 0]),
                    "y": float(coordinates[index, 1]),
                    "z": float(coordinates[index, 2]) if coordinates.shape[1] > 2 else 0.0,
                }
                for index in range(len(records))
            ]
            views.append({
                "model_id": selected_model,
                "method": method,
                "points": points,
                "embedding_shape": embedding_shapes[selected_model],
                "embedding_finite": bool(np.isfinite(embeddings_by_model[selected_model]).all()),
                "embedding_nonzero": True,
            })
    if not views:
        raise VisualizationInputError("PROJECTION_UNAVAILABLE", "No selected projection could be computed.")

    primary = views[0]
    base_warnings = ([f"Skipped {len(unreadable)} unreadable image(s)."] if unreadable else [])
    return {
        "mode": mode,
        "unit": unit,
        "object_policy": resolved_object_policy,
        "class_names": resolved_class_names,
        "model_id": primary["model_id"],
        "method": primary["method"],
        "records": public_records,
        "points": primary["points"],
        "views": views,
        "models": selected_models,
        "methods": list(dict.fromkeys(view["method"] for view in views)),
        "embedding_shapes": embedding_shapes,
        "embedding_shape": primary["embedding_shape"],
        "embedding_finite": primary["embedding_finite"],
        "embedding_nonzero": primary["embedding_nonzero"],
        "warnings": base_warnings + warnings,
        "image_paths": [Path(record["path"]).resolve() for record in records],
    }
