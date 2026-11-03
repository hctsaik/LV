"""Framework-free LV React visualization pipeline for LV-WEB-01-IC-01."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from _utils import extract_embeddings, load_model, yaml_class_names
from interaction import compute_label_disagreement, compute_outlier_scores
from label_formats import folder_has_annotations
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


def _compute_rank_scores(
    embeddings_by_model: dict[str, np.ndarray], labels: Sequence[str],
) -> dict[str, dict[str, list[float]]]:
    """Cache Streamlit's exact review scores without exposing feature vectors."""
    if len(labels) < 3:
        return {}
    k = min(5, len(labels) - 1)
    results: dict[str, dict[str, list[float]]] = {}
    for model_id, embeddings in embeddings_by_model.items():
        results[model_id] = {
            "outlier": compute_outlier_scores(
                embeddings, embeddings, k=k, candidates_in_reference=True,
            ).astype(float).tolist(),
            "disagreement": compute_label_disagreement(
                embeddings, labels, k=k,
            ).astype(float).tolist(),
        }
    return results


def _is_under(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def _project_embeddings(
    embeddings: np.ndarray,
    labels: Sequence[str],
    methods: Sequence[str],
    *,
    umap_reference_path: Path | None = None,
    umap_reference_keys: Sequence[str] | None = None,
    rebuild_umap_reference: bool = False,
) -> tuple[dict[str, np.ndarray], list[str]]:
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
                neighbors = min(15, max(2, n_samples - 1))
                if umap_reference_path is not None and umap_reference_keys is not None:
                    from umap_ref import stable_umap

                    coordinates, new_count, refitted = stable_umap(
                        embeddings,
                        umap_reference_keys,
                        umap_reference_path,
                        n_components,
                        neighbors,
                        rebuild=rebuild_umap_reference,
                    )
                    state = "rebuilt" if refitted else f"reused (+{new_count} new images)"
                    warnings.append(f"Fixed UMAP reference {state}.")
                else:
                    import umap

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


def _class_names_from_text(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _detector_classes_from_dataset(folder: Path) -> tuple[list[str] | None, str | None]:
    """Match the legacy first-folder classes.txt/data.yaml discovery order."""
    for root in (folder.parent, folder, folder.parent.parent):
        classes_path = root / "classes.txt"
        if classes_path.is_file():
            names = _class_names_from_text(safe_read_text(classes_path))
            if names:
                return names, "auto_classes_txt"
        names = yaml_class_names(root / "data.yaml")
        if names:
            return names, "auto_data_yaml"
    return None, None


_OBJECT_RESOLUTIONS = {168, 196, 224, 280, 336, 392, 448}


def _normalize_object_policy(policy: dict | None) -> dict:
    from object_eval import DEFAULT_POLICY

    value = {**DEFAULT_POLICY, **(policy or {})}
    try:
        pad = float(value["pad"])
        target_res = int(value["target_res"])
        head = str(value["head"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VisualizationInputError("INVALID_OBJECT_POLICY", "The object policy is invalid.") from exc
    if not np.isfinite(pad) or not 0 <= pad <= 0.5:
        raise VisualizationInputError("INVALID_OBJECT_POLICY", "Object padding must be between 0 and 0.5.")
    if target_res not in _OBJECT_RESOLUTIONS:
        raise VisualizationInputError("INVALID_OBJECT_POLICY", "The object resolution is unsupported.")
    if head not in {"cls", "meanpool"}:
        raise VisualizationInputError("INVALID_OBJECT_POLICY", "The object feature head is unsupported.")
    return {"pad": pad, "target_res": target_res, "head": head}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_object_visualization(
    image_records: Sequence[dict],
    folders: Sequence[Path],
    *,
    class_names: list[str],
    class_source: str,
    model_names: Sequence[str],
    methods: Sequence[str],
    policy: dict,
    object_model_loader: Callable[[str, dict], Callable] | None,
    progress: Callable[[int, int], None] | None,
    cancelled: Callable[[], bool] | None,
    unreadable_count: int,
) -> dict:
    """Run bbox-level detector embeddings while keeping source data read-only."""
    from interaction import discover_yolo_objects
    from object_eval import dataset_cache_dir, embed_objects, policy_tag

    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise JobCancelled()

    split_by_path = {
        str(Path(record["path"]).resolve()): str(record["split"])
        for record in image_records
    }
    objects_by_folder: dict[Path, list[dict]] = {}
    ordered_objects: list[dict] = []
    for folder in folders:
        folder_paths = [
            Path(record["path"]).resolve()
            for record in image_records
            if _is_under(Path(record["path"]), folder)
        ]
        if not folder_paths:
            continue
        found = discover_yolo_objects(folder_paths, class_names)
        rows: list[dict] = []
        for item in found:
            check_cancelled()
            path = Path(item["image_path"]).resolve()
            row = {
                **item,
                "image_path": path,
                "path": path,
                "split": split_by_path.get(str(path), folder.name),
            }
            rows.append(row)
            ordered_objects.append(row)
        if rows:
            objects_by_folder[folder] = rows
    if not ordered_objects:
        raise VisualizationInputError("NO_OBJECTS", "No labeled objects were found in the selected images.")

    source_hashes = {
        path: _sha256_file(path)
        for path in dict.fromkeys(row["image_path"] for row in ordered_objects)
    }
    cache_keys_by_folder: dict[Path, list[str]] = {}
    for folder, rows in objects_by_folder.items():
        cache_keys: list[str] = []
        for row in rows:
            identity = {
                "image_sha256": source_hashes[row["image_path"]],
                "obj_index": int(row["obj_index"]),
                "class_id": row["class_id"],
                "bbox": [round(float(value), 8) for value in row["bbox"]],
            }
            cache_keys.append(hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest())
        cache_keys_by_folder[folder] = cache_keys

    labels = [str(row["label"]) for row in ordered_objects]
    completed = 0
    total = len(ordered_objects) * len(model_names)
    embeddings_by_model: dict[str, np.ndarray] = {}
    embedding_shapes: dict[str, list[int]] = {}
    projections_by_model: dict[str, dict[str, np.ndarray]] = {}
    embedding_nonzero: dict[str, bool] = {}
    warnings: list[str] = []

    for selected_model in model_names:
        check_cancelled()
        all_embeddings: list[np.ndarray] = []
        shared_embedder: Callable | None = None

        def load_object_embedder(model_id: str, selected_policy: dict) -> Callable:
            nonlocal shared_embedder
            if shared_embedder is None:
                if object_model_loader is None:
                    shared_embedder = load_model(
                        model_id,
                        keep_aspect=True,
                        target_res=selected_policy["target_res"],
                        head=selected_policy["head"],
                    )
                else:
                    shared_embedder = object_model_loader(model_id, selected_policy)
            return shared_embedder

        for folder, rows in objects_by_folder.items():
            base_done = completed

            def on_item(done: int, _folder_total: int) -> None:
                check_cancelled()
                if progress is not None:
                    progress(base_done + done, total)

            cache_path = (
                dataset_cache_dir(folder, "object_crops")
                / policy_tag(policy)
                / f"embeddings_{selected_model}.npz"
            )
            folder_embeddings, _short_px = embed_objects(
                rows,
                selected_model,
                policy,
                cache_path=cache_path,
                progress=on_item,
                cache_keys=cache_keys_by_folder[folder],
                embedder_factory=load_object_embedder,
            )
            if folder_embeddings.ndim != 2 or folder_embeddings.shape[0] != len(rows):
                raise VisualizationInputError("INVALID_EMBEDDINGS", "The selected model returned invalid object embeddings.")
            all_embeddings.append(folder_embeddings)
            completed += len(rows)
        raw = np.vstack(all_embeddings).astype(np.float32, copy=False)
        if raw.ndim != 2 or raw.shape[0] != len(ordered_objects) or not np.isfinite(raw).all():
            raise VisualizationInputError("INVALID_EMBEDDINGS", "The selected model returned invalid object embeddings.")
        nonzero = bool(np.all(np.linalg.norm(raw, axis=1) > 0))
        if not nonzero:
            warnings.append(f"{selected_model}: some object embeddings are empty.")
        embeddings_by_model[selected_model] = raw
        embedding_nonzero[selected_model] = nonzero
        embedding_shapes[selected_model] = [int(raw.shape[0]), int(raw.shape[1])]
        projections, projection_warnings = _project_embeddings(raw, labels, methods)
        projections_by_model[selected_model] = projections
        warnings.extend(f"{selected_model}: {warning}" for warning in projection_warnings)
        check_cancelled()

    public_records: list[dict] = []
    for index, row in enumerate(ordered_objects):
        public_records.append({
            "index": index,
            "filename": row["image_path"].name,
            "split": str(row["split"]),
            "label": str(row["label"]),
            "object_index": int(row["obj_index"]),
            "class_id": int(row["class_id"]) if row["class_id"] is not None else None,
            "bbox": [float(value) for value in row["bbox"]],
            "score": float(row["score"]) if row.get("score") is not None else None,
        })

    views: list[dict] = []
    for selected_model in model_names:
        for method, coordinates in projections_by_model[selected_model].items():
            points = [
                {
                    **public_records[index],
                    "x": float(coordinates[index, 0]),
                    "y": float(coordinates[index, 1]),
                    "z": float(coordinates[index, 2]) if coordinates.shape[1] > 2 else 0.0,
                }
                for index in range(len(ordered_objects))
            ]
            views.append({
                "model_id": selected_model,
                "method": method,
                "points": points,
                "embedding_shape": embedding_shapes[selected_model],
                "embedding_finite": bool(np.isfinite(embeddings_by_model[selected_model]).all()),
                "embedding_nonzero": embedding_nonzero[selected_model],
            })
    if not views:
        raise VisualizationInputError("PROJECTION_UNAVAILABLE", "No selected projection could be computed.")

    primary = views[0]
    ranking_scores = _compute_rank_scores(
        embeddings_by_model, [str(row["label"]) for row in ordered_objects],
    )
    return {
        "source_mode": "object_detector",
        "source_granularity": "object",
        "class_names": class_names,
        "class_source": class_source,
        "object_policy": policy,
        "model_id": primary["model_id"],
        "method": primary["method"],
        "records": public_records,
        "points": primary["points"],
        "views": views,
        "models": list(model_names),
        "methods": list(dict.fromkeys(view["method"] for view in views)),
        "embedding_shapes": embedding_shapes,
        "embedding_shape": primary["embedding_shape"],
        "embedding_finite": primary["embedding_finite"],
        "embedding_nonzero": primary["embedding_nonzero"],
        "warnings": ([f"Skipped {unreadable_count} unreadable image(s)."] if unreadable_count else []) + warnings,
        "image_paths": [Path(row["image_path"]).resolve() for row in ordered_objects],
        "_object_policy": policy,
        "_embedding_sources": {name: [] for name in model_names},
        "_ranking_scores": ranking_scores,
        "_sha256s": [source_hashes[row["image_path"]] for row in ordered_objects],
    }


def run_visualization(
    folders: Sequence[str | Path],
    model_name: str | None = None,
    *,
    model_names: Sequence[str] | None = None,
    methods: Sequence[str] = ("pca",),
    mode: str = "image_classifier",
    granularity: str = "whole_image",
    classes_file_text: str | None = None,
    manual_class_names: Sequence[str] = (),
    object_policy: dict | None = None,
    object_model_loader: Callable[[str, dict], Callable] | None = None,
    umap_reference: bool = False,
    rebuild_umap_reference: bool = False,
    model_loader: Callable[[str], Callable[[Path], np.ndarray]] = load_model,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Run the legacy Visualize semantics without Streamlit state.

    Classifier and Object Detector whole-image/object discovery, policy-scoped
    embedding caches, projection parameters, and label/split alignment follow
    the existing Visualize Embeddings path.
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
    if mode not in {"image_classifier", "object_detector"}:
        raise VisualizationInputError("MODE_UNAVAILABLE", "Choose a supported dataset mode.")
    if granularity not in {"whole_image", "object"}:
        raise VisualizationInputError("GRANULARITY_UNAVAILABLE", "Choose a supported analysis granularity.")
    if mode == "image_classifier" and granularity != "whole_image":
        raise VisualizationInputError("GRANULARITY_UNAVAILABLE", "Object granularity requires Object Detector mode.")

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
    if rebuild_umap_reference and (not umap_reference or "umap" not in selected_methods):
        raise VisualizationInputError(
            "INVALID_UMAP_REFERENCE", "Rebuild requires the fixed UMAP reference option and UMAP projection."
        )

    check_cancelled()
    class_names: list[str] = []
    class_source = "classifier_folders"
    if mode == "object_detector":
        invalid_folders = [
            folder for folder in resolved_folders
            if not (folder / "images").is_dir()
            and not (folder / "JPEGImages").is_dir()
            and not folder_has_annotations(folder)
        ]
        if invalid_folders:
            raise VisualizationInputError(
                "DETECTOR_DATASET_INVALID",
                "Each Object Detector folder must contain images or supported image annotations.",
            )
        if classes_file_text is not None:
            class_names = _class_names_from_text(classes_file_text)
            if not class_names:
                raise VisualizationInputError(
                    "CLASSES_FILE_EMPTY", "The selected classes file does not contain class names."
                )
            class_source = "selected_classes_file"
        else:
            detected, detected_source = _detector_classes_from_dataset(resolved_folders[0])
            if detected:
                class_names = detected
                class_source = detected_source or "auto_classes_txt"
            else:
                class_names = [name.strip() for name in manual_class_names if name.strip()]
                class_source = "manual"
            if not class_names:
                raise VisualizationInputError(
                    "CLASSES_REQUIRED",
                    "Choose a classes.txt file, use a dataset with classes.txt or data.yaml, or enter class names.",
                )
        records = discover_images(resolved_folders, class_names)
    else:
        records = discover_images_classifier(resolved_folders, allow_flat=True)
        class_names = sorted({str(record["label"]) for record in records})
    if not records:
        raise VisualizationInputError("NO_IMAGES", "No images were found in the selected folders.")

    readable, unreadable = partition_readable([record["path"] for record in records])
    readable_keys = {str(path.resolve()) for path in readable}
    records = [record for record in records if str(Path(record["path"]).resolve()) in readable_keys]
    if not records:
        raise VisualizationInputError("NO_IMAGES", "No readable images were found in the selected folders.")
    check_cancelled()

    if mode == "object_detector" and granularity == "object":
        policy = _normalize_object_policy(object_policy)
        return _run_object_visualization(
            records,
            resolved_folders,
            class_names=class_names,
            class_source=class_source,
            model_names=selected_models,
            methods=selected_methods,
            policy=policy,
            object_model_loader=object_model_loader,
            progress=progress,
            cancelled=cancelled,
            unreadable_count=len(unreadable),
        )

    manifest_by_folder: dict[Path, dict[str, dict]] = {}
    paths_by_folder: dict[Path, list[Path]] = {}
    for folder in resolved_folders:
        folder_records = [record for record in records if _is_under(Path(record["path"]), folder)]
        if not folder_records:
            continue
        manifest_by_folder[folder] = update_manifest(folder, folder_records)
        paths_by_folder[folder] = [Path(record["path"]) for record in folder_records]

    completed = 0
    total = len(records) * len(selected_models)
    embeddings_by_model: dict[str, np.ndarray] = {}
    projections_by_model: dict[str, dict[str, np.ndarray]] = {}
    embedding_shapes: dict[str, list[int]] = {}
    embedding_sources: dict[str, list[dict[str, object]]] = {}
    warnings: list[str] = []
    labels = [str(record["label"]) for record in records]
    record_index_by_path = {
        str(Path(record["path"]).resolve()): index
        for index, record in enumerate(records)
    }

    for selected_model in selected_models:
        check_cancelled()
        embed_fn = model_loader(selected_model)
        all_embeddings: list[np.ndarray] = []
        all_keys: list[str] = []
        model_sources: list[dict[str, object]] = []
        for folder in resolved_folders:
            paths = paths_by_folder.get(folder, [])
            if not paths:
                continue
            check_cancelled()
            entries = manifest_by_folder[folder]
            keys = [entries[rel_key(folder, path)]["sha256"] for path in paths]
            all_keys.extend(keys)
            cache_path = manifest_path_for(folder).parent / f"embeddings_{selected_model}" / "embeddings.npz"
            model_sources.append({
                "cache_path": str(cache_path.resolve()),
                "indices": [record_index_by_path[str(path.resolve())] for path in paths],
                "keys": keys,
            })

            def on_item(done: int, _folder_total: int) -> None:
                check_cancelled()
                if progress is not None:
                    progress(completed + done, total)

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
        embedding_sources[selected_model] = model_sources
        umap_reference_path = None
        if umap_reference and "umap" in selected_methods:
            from umap_ref import ref_path_for

            umap_reference_path = ref_path_for(resolved_folders[0], selected_model)
        projections, projection_warnings = _project_embeddings(
            raw,
            labels,
            selected_methods,
            umap_reference_path=umap_reference_path,
            umap_reference_keys=all_keys if umap_reference_path is not None else None,
            rebuild_umap_reference=rebuild_umap_reference,
        )
        projections_by_model[selected_model] = projections
        warnings.extend(f"{selected_model}: {warning}" for warning in projection_warnings)
        check_cancelled()

    for folder, entries in manifest_by_folder.items():
        write_manifest(folder, entries)

    phash_by_path = {
        str(path.resolve()): entries.get(rel_key(folder, path), {}).get("phash")
        for folder, entries in manifest_by_folder.items()
        for path in paths_by_folder.get(folder, [])
    }

    public_records = [
        {
            "index": index,
            "filename": Path(record["path"]).name,
            "split": str(record["split"]),
            "label": str(record["label"]),
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
    ranking_scores = _compute_rank_scores(
        embeddings_by_model, [str(record["label"]) for record in records],
    )
    sha256s = [""] * len(records)
    for source in embedding_sources.get(selected_models[0], []):
        for index, key in zip(source.get("indices", []), source.get("keys", [])):
            sha256s[int(index)] = str(key)
    return {
        "source_mode": mode,
        "source_granularity": "whole_image",
        "class_names": class_names,
        "class_source": class_source,
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
        "_embedding_sources": embedding_sources,
        "_phashes": [phash_by_path.get(str(Path(record["path"]).resolve())) for record in records],
        "_ranking_scores": ranking_scores,
        "_sha256s": sha256s,
    }


# Backward-compatible internal name for the original classifier-only caller.
run_image_classifier = run_visualization
