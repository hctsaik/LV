"""Framework-free LV React visualization pipeline for LV-WEB-01-IC-01."""
from __future__ import annotations

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
from safe_io import partition_readable
from visualize_embeddings import discover_images_classifier


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


def run_image_classifier(
    folders: Sequence[str | Path],
    model_name: str,
    *,
    model_loader: Callable[[str], Callable[[Path], np.ndarray]] = load_model,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Run the legacy whole-image classifier flow without Streamlit state.

    Discovery, content-hash embedding cache, model loading, and PCA parameters
    follow the existing Visualize Embeddings path. Every write is routed through
    LV's app-side manifest/cache helpers, never into selected source folders.
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
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_name):
        raise VisualizationInputError("MODEL_UNAVAILABLE", "The selected model is unavailable.")

    check_cancelled()
    records = discover_images_classifier(resolved_folders, allow_flat=True)
    if not records:
        raise VisualizationInputError("NO_IMAGES", "No images were found in the selected folders.")

    readable, unreadable = partition_readable([record["path"] for record in records])
    readable_keys = {str(path.resolve()) for path in readable}
    records = [record for record in records if str(Path(record["path"]).resolve()) in readable_keys]
    if not records:
        raise VisualizationInputError("NO_IMAGES", "No readable images were found in the selected folders.")
    check_cancelled()

    manifest_by_folder: dict[Path, dict[str, dict]] = {}
    paths_by_folder: dict[Path, list[Path]] = {}
    for folder in resolved_folders:
        folder_records = [record for record in records if _is_under(Path(record["path"]), folder)]
        if not folder_records:
            continue
        manifest_by_folder[folder] = update_manifest(folder, folder_records)
        paths_by_folder[folder] = [Path(record["path"]) for record in folder_records]

    embed_fn = model_loader(model_name)
    all_embeddings: list[np.ndarray] = []
    completed = 0
    total = len(records)
    for folder in resolved_folders:
        paths = paths_by_folder.get(folder, [])
        if not paths:
            continue
        check_cancelled()
        entries = manifest_by_folder[folder]
        keys = [entries[rel_key(folder, path)]["sha256"] for path in paths]
        cache_path = manifest_path_for(folder).parent / f"embeddings_{model_name}" / "embeddings.npz"

        def on_item(done: int, _folder_total: int) -> None:
            nonlocal completed
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
        set_embedding_refs(entries, folder, model_name, paths)
        write_manifest(folder, entries)
        all_embeddings.append(folder_embeddings)
        completed += len(paths)

    raw = np.vstack(all_embeddings).astype(np.float32, copy=False)
    check_cancelled()
    if raw.ndim != 2 or raw.shape[0] != len(records) or not np.isfinite(raw).all():
        raise VisualizationInputError("INVALID_EMBEDDINGS", "The selected model returned invalid embeddings.")
    nonzero = bool(np.all(np.linalg.norm(raw, axis=1) > 0))
    if not nonzero:
        raise VisualizationInputError("INVALID_EMBEDDINGS", "The selected model returned an empty embedding.")

    # Match scripts/app.py's main whole-image projection rule, including the
    # zero-padded second axis for one-dimensional small-sample PCA results.
    n_samples = len(records)
    n_components = min(3, max(1, n_samples - 2))
    projected = PCA(n_components=n_components, random_state=42).fit_transform(raw)
    if projected.shape[1] < 2:
        projected = np.hstack([projected, np.zeros((len(records), 1))])
    check_cancelled()

    public_records = [
        {
            "index": index,
            "filename": Path(record["path"]).name,
            "split": str(record["split"]),
            "label": str(record["label"]),
        }
        for index, record in enumerate(records)
    ]
    points = [
        {
            **public_records[index],
            "x": float(projected[index, 0]),
            "y": float(projected[index, 1]),
        }
        for index in range(len(records))
    ]
    return {
        "model_id": model_name,
        "method": "pca",
        "records": public_records,
        "points": points,
        "embedding_shape": [int(raw.shape[0]), int(raw.shape[1])],
        "embedding_finite": bool(np.isfinite(raw).all()),
        "embedding_nonzero": nonzero,
        "warnings": ([f"Skipped {len(unreadable)} unreadable image(s)."] if unreadable else []),
        "image_paths": [Path(record["path"]).resolve() for record in records],
    }
