"""Framework-free whole-image A/B comparison for LV's React local API."""
from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from _utils import extract_embeddings, load_model
from manifest import file_sha256, manifest_path_for
from safe_io import safe_open_image


class CompareInputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class CompareCancelled(Exception):
    """Internal signal that the current local comparison should stop."""


Progress = Callable[[str, int, int, str], None]


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise CompareCancelled()


def _resolved_roots(values: Sequence[str | Path], group: str) -> list[Path]:
    if not values or len(values) > 32:
        raise CompareInputError("INVALID_REQUEST", f"Choose one to 32 folders for Folder {group}.")
    roots: list[Path] = []
    seen: set[str] = set()
    for value in values:
        try:
            folder = Path(value).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise CompareInputError("INVALID_FOLDER", f"A Folder {group} path is unavailable.") from exc
        if not folder.is_dir():
            raise CompareInputError("INVALID_FOLDER", f"A Folder {group} selection is not a directory.")
        key = os.path.normcase(str(folder))
        if key not in seen:
            roots.append(folder)
            seen.add(key)
    return roots


def _images_for_root(folder: Path) -> tuple[list[Path], bool]:
    patterns = ("*.jpg", "*.jpeg", "*.png")
    direct = sorted(path for pattern in patterns for path in folder.glob(pattern) if path.is_file())
    if direct:
        return direct, False
    excluded = ("embeddings_", "compare_embeddings_", "object_crops", ".thumbs")
    nested = sorted({
        path
        for pattern in patterns
        for path in folder.rglob(pattern)
        if path.is_file() and not any(part.startswith(excluded) for part in path.parts)
    })
    return nested, bool(nested)


def _resolve_group(values: Sequence[str | Path], group: str) -> tuple[list[Path], list[tuple[Path, Path]], list[str]]:
    roots = _resolved_roots(values, group)
    image_roots: list[tuple[Path, Path]] = []
    notes: list[str] = []
    seen: set[str] = set()
    for root in roots:
        paths, nested = _images_for_root(root)
        if nested:
            notes.append(f"Folder {group}: images were found in subfolders.")
        for path in paths:
            resolved = path.resolve()
            key = os.path.normcase(str(resolved))
            if key not in seen:
                image_roots.append((root, resolved))
                seen.add(key)
    if not image_roots:
        raise CompareInputError("NO_IMAGES", f"Folder {group} contains no supported images.")
    return roots, image_roots, notes


def _group_name(roots: Sequence[Path], group: str) -> str:
    return roots[0].name if len(roots) == 1 else f"{group}（{len(roots)} 個資料夾）"


def _embedding_cache_path(folder: Path, model_id: str) -> Path:
    # Match LV's app-owned cache hierarchy without writing beside user images.
    return manifest_path_for(folder).parent / f"compare_embeddings_{model_id}" / "embeddings.npz"


def _embed_group(
    image_roots: Sequence[tuple[Path, Path]],
    embed: Callable[[Path], np.ndarray],
    model_id: str,
    *,
    progress: Progress | None,
    completed_before: int,
    total: int,
    cancelled: Callable[[], bool] | None,
) -> tuple[list[Path], np.ndarray]:
    paths: list[Path] = []
    vectors: list[np.ndarray] = []
    by_root: dict[Path, list[Path]] = {}
    for root, path in image_roots:
        by_root.setdefault(root, []).append(path)

    done_before = completed_before
    for root, root_paths in by_root.items():
        _check_cancelled(cancelled)

        def report(done: int, _count: int) -> None:
            _check_cancelled(cancelled)
            if progress:
                progress("embedding", done_before + done, total,
                         f"Extracting Folder features ({done_before + done}/{total})")

        features = extract_embeddings(
            root_paths,
            embed,
            cache_path=_embedding_cache_path(root, model_id),
            cache_keys=[file_sha256(path) for path in root_paths],
            progress_cb=report,
        )
        paths.extend(root_paths)
        vectors.append(features)
        done_before += len(root_paths)
    if not vectors:
        raise CompareInputError("NO_IMAGES", "A comparison folder contains no readable images.")
    return paths, np.vstack(vectors).astype(np.float32, copy=False)


def run_image_comparison(
    folders_a: Sequence[str | Path],
    folders_b: Sequence[str | Path],
    model_id: str,
    *,
    model_loader: Callable[[str], Callable[[Path], np.ndarray]] = load_model,
    progress: Progress | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Extract both whole-image groups and compute the legacy centroid drift.

    Folder selection, duplicate removal, bad-image filtering, cache isolation,
    and drift arithmetic follow the current Streamlit whole-image path.
    Projection is requested separately so the same embeddings can support
    method and dimension changes without loading the model again.
    """
    roots_a, candidates_a, notes_a = _resolve_group(folders_a, "A")
    roots_b, candidates_b, notes_b = _resolve_group(folders_b, "B")
    readable_a: list[tuple[Path, Path]] = []
    readable_b: list[tuple[Path, Path]] = []
    unreadable = 0
    for candidates, readable in ((candidates_a, readable_a), (candidates_b, readable_b)):
        for root, path in candidates:
            _check_cancelled(cancelled)
            if safe_open_image(path) is None:
                unreadable += 1
            else:
                readable.append((root, path))
    if not readable_a or not readable_b:
        raise CompareInputError("NO_IMAGES", "Each comparison group needs at least one readable image.")

    total = len(readable_a) + len(readable_b)
    if progress:
        progress("embedding", 0, total, "Extracting whole-image features")
    embed = model_loader(model_id)
    paths_a, embeddings_a = _embed_group(
        readable_a, embed, model_id, progress=progress, completed_before=0,
        total=total, cancelled=cancelled,
    )
    paths_b, embeddings_b = _embed_group(
        readable_b, embed, model_id, progress=progress, completed_before=len(paths_a),
        total=total, cancelled=cancelled,
    )
    combined = np.vstack([embeddings_a, embeddings_b]).astype(np.float32, copy=False)
    if combined.ndim != 2 or not np.isfinite(combined).all():
        raise CompareInputError("INVALID_EMBEDDINGS", "The selected model returned invalid image features.")

    mean_a = embeddings_a.mean(axis=0)
    mean_b = embeddings_b.mean(axis=0)
    mean_a = mean_a / (np.linalg.norm(mean_a) + 1e-9)
    mean_b = mean_b / (np.linalg.norm(mean_b) + 1e-9)
    drift = max(0.0, 1.0 - float(mean_a @ mean_b))
    overlap = len({os.path.normcase(str(path)) for path in paths_a}
                  & {os.path.normcase(str(path)) for path in paths_b})
    warnings: list[str] = []
    warnings.extend(notes_a)
    warnings.extend(notes_b)
    if unreadable:
        warnings.append(f"Skipped {unreadable} unreadable image(s).")
    if overlap:
        warnings.append(
            f"Folder A and B share {overlap} image(s); the comparison may understate distribution differences."
        )
    if progress:
        progress("completed", total, total, "Whole-image comparison completed")
    return {
        "folder_a": _group_name(roots_a, "A"),
        "folder_b": _group_name(roots_b, "B"),
        "model_id": model_id,
        "drift": round(drift, 4),
        "records_a": [{"index": index, "filename": path.name} for index, path in enumerate(paths_a)],
        "records_b": [{"index": index, "filename": path.name} for index, path in enumerate(paths_b)],
        "warnings": warnings,
        "overlap_count": overlap,
        "image_paths_a": paths_a,
        "image_paths_b": paths_b,
        "embeddings_a": embeddings_a,
        "embeddings_b": embeddings_b,
    }


def _supervised_projection(
    combined: np.ndarray,
    labels: np.ndarray,
    method: str,
    dimension: int,
) -> np.ndarray | None:
    unique = np.unique(labels)
    if len(unique) < 2 or len(combined) < 4:
        return None
    encoded = np.searchsorted(unique, labels)
    if method == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        count = min(dimension, len(unique) - 1)
        return LinearDiscriminantAnalysis(n_components=count).fit_transform(combined, encoded)
    import umap

    neighbors = min(15, max(2, len(combined) - 1))
    return umap.UMAP(
        n_components=dimension,
        n_neighbors=neighbors,
        random_state=42,
    ).fit_transform(combined, y=encoded)


def project_comparison(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    method: str,
    dimension: int,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Create the selected legacy A/B projection and stable group/index map."""
    if method not in {"pca", "tsne", "umap", "lda", "sumap"}:
        raise CompareInputError("METHOD_UNAVAILABLE", "Select a supported projection method.")
    if dimension not in {2, 3}:
        raise CompareInputError("INVALID_REQUEST", "Projection dimension must be 2 or 3.")
    _check_cancelled(cancelled)
    combined = np.vstack([embeddings_a, embeddings_b]).astype(np.float32, copy=False)
    n_samples = len(combined)
    if n_samples <= dimension + 1:
        coordinates = np.zeros((n_samples, dimension), dtype=np.float32)
    elif method == "pca":
        coordinates = PCA(n_components=dimension, random_state=42).fit_transform(combined)
    elif method == "tsne":
        coordinates = TSNE(
            n_components=dimension,
            random_state=42,
            perplexity=min(30, max(2, n_samples - 1)),
        ).fit_transform(combined)
    elif method == "umap":
        import umap

        neighbors = min(15, max(2, n_samples - 1))
        coordinates = umap.UMAP(
            n_components=dimension,
            n_neighbors=neighbors,
            random_state=42,
        ).fit_transform(combined)
    else:
        labels = np.asarray(["a"] * len(embeddings_a) + ["b"] * len(embeddings_b))
        supervised = _supervised_projection(combined, labels, method, dimension)
        if supervised is None:
            coordinates = PCA(n_components=dimension, random_state=42).fit_transform(combined)
        elif supervised.shape[1] < dimension:
            padding = PCA(n_components=dimension - supervised.shape[1], random_state=42).fit_transform(combined)
            coordinates = np.hstack([supervised, padding])
        else:
            coordinates = supervised
    _check_cancelled(cancelled)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape != (n_samples, dimension):
        raise CompareInputError("PROJECTION_UNAVAILABLE", "The selected projection returned an invalid coordinate shape.")
    if not np.isfinite(coordinates).all():
        raise CompareInputError("PROJECTION_UNAVAILABLE", "The selected projection returned invalid coordinates.")
    points = []
    for group, offset, paths, count in (
        ("a", 0, None, len(embeddings_a)),
        ("b", len(embeddings_a), None, len(embeddings_b)),
    ):
        for index in range(count):
            row = offset + index
            points.append({
                "group": group,
                "index": index,
                "x": float(coordinates[row, 0]),
                "y": float(coordinates[row, 1]),
                "z": float(coordinates[row, 2]) if dimension == 3 else 0.0,
            })
    return {"method": method, "dimension": dimension, "points": points}
