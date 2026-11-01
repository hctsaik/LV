"""Stateless calculations for the image-level embedding coverage workflow."""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from interaction import sparsity_scores

CoverageMethod = Literal["pca", "tsne", "umap", "lda", "sumap"]


def _l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Coverage embeddings must be a two-dimensional matrix.")
    return array / (np.linalg.norm(array, axis=1, keepdims=True) + 1e-9)


def coverage_projection(
    embeddings: np.ndarray,
    labels: Sequence[str],
    method: CoverageMethod,
    dimension: int,
) -> tuple[np.ndarray, str, list[str]]:
    """Return deterministic coverage coordinates in the legacy cosine space.

    Projection is a display-only view. Sparsity is calculated separately from
    the original, unprojected feature matrix by ``sparsity_scores``.
    """
    if dimension not in (2, 3):
        raise ValueError("Coverage projection dimension must be 2 or 3.")
    values = _l2_normalize_rows(embeddings)
    if len(values) == 0 or len(labels) != len(values):
        raise ValueError("Coverage records and embeddings must have matching rows.")

    sample_count = len(values)
    component_count = min(dimension, max(1, sample_count - 1), values.shape[1])
    warnings: list[str] = []

    def pca() -> np.ndarray:
        if sample_count == 1:
            coordinates = np.zeros((1, component_count), dtype=np.float64)
        else:
            coordinates = PCA(
                n_components=component_count,
                random_state=42,
            ).fit_transform(values)
        return _pad_coordinates(coordinates, dimension)

    selected_method = method
    unique_labels = np.unique(np.asarray(labels, dtype=str))
    if method in {"tsne", "umap"} and sample_count < 4:
        warnings.append(f"{method} needs at least four images; PCA is shown instead.")
        selected_method = "pca"
    elif method in {"lda", "sumap"} and (sample_count < 4 or len(unique_labels) < 2):
        warnings.append(f"{method} needs at least four images from two classes; PCA is shown instead.")
        selected_method = "pca"

    try:
        if selected_method == "pca":
            coordinates = pca()
        elif selected_method == "tsne":
            coordinates = TSNE(
                n_components=component_count,
                random_state=42,
                perplexity=min(30, max(1, sample_count - 1)),
            ).fit_transform(values)
        elif selected_method == "umap":
            import umap

            coordinates = umap.UMAP(
                n_components=component_count,
                n_neighbors=min(15, max(2, sample_count - 1)),
                random_state=42,
            ).fit_transform(values)
        elif selected_method == "lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            label_ids = np.searchsorted(unique_labels, np.asarray(labels, dtype=str))
            components = min(component_count, len(unique_labels) - 1)
            coordinates = LinearDiscriminantAnalysis(
                n_components=components,
            ).fit_transform(values, label_ids)
        else:  # supervised UMAP
            import umap

            label_ids = np.searchsorted(unique_labels, np.asarray(labels, dtype=str))
            coordinates = umap.UMAP(
                n_components=component_count,
                n_neighbors=min(15, max(2, sample_count - 1)),
                random_state=42,
            ).fit_transform(values, y=label_ids)
    except (ImportError, ModuleNotFoundError):
        warnings.append(f"{selected_method} is unavailable; PCA is shown instead.")
        selected_method = "pca"
        coordinates = pca()
    except ValueError as exc:
        warnings.append(f"{selected_method} could not be computed; PCA is shown instead.")
        selected_method = "pca"
        coordinates = pca()

    coordinates = _pad_coordinates(np.asarray(coordinates, dtype=np.float64), dimension)
    if coordinates.shape[0] != sample_count or not np.isfinite(coordinates).all():
        raise ValueError("Coverage projection did not return finite coordinates for every image.")
    return coordinates, selected_method, warnings


def _pad_coordinates(values: np.ndarray, dimension: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError("Coverage projection must return a two-dimensional matrix.")
    if result.shape[1] >= dimension:
        return result[:, :dimension]
    return np.hstack([result, np.zeros((len(result), dimension - result.shape[1]))])


def coverage_sparsity(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Mean cosine distance to k nearest other images in full feature space."""
    values = np.asarray(embeddings)
    if values.ndim != 2:
        raise ValueError("Coverage embeddings must be a two-dimensional matrix.")
    if len(values) < 2:
        return np.zeros(len(values), dtype=np.float64)
    bounded_k = min(max(1, int(k)), len(values) - 1)
    return sparsity_scores(values, k=bounded_k)


def sparse_threshold(scores: np.ndarray, percentile: float = 90.0) -> tuple[float, int]:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0
    threshold = float(np.percentile(values, percentile))
    return threshold, int(np.sum(values >= threshold))
