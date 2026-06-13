"""Unit tests for scripts/umap_ref.py — the persistent UMAP reference frame."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from umap_ref import load_ref, ref_path_for, stable_umap


def _data(n: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, dim))
    base[: n // 2] += 4.0  # two clusters so UMAP has structure
    return base.astype(np.float32)


def _keys(n: int, prefix: str = "k") -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(n)]


def test_first_run_fits_and_persists(tmp_path):
    emb, keys = _data(20), _keys(20)
    ref_p = tmp_path / "emb" / "umap_ref.pkl"
    coords, n_new, refitted = stable_umap(emb, keys, ref_p, 2, 10)
    assert refitted and n_new == 0
    assert coords.shape == (20, 2) and np.all(np.isfinite(coords))
    assert ref_p.exists()
    ref = load_ref(ref_p)
    assert ref is not None and len(ref["keys"]) == 20


def test_same_data_reuses_frozen_coords_exactly(tmp_path):
    emb, keys = _data(20), _keys(20)
    ref_p = tmp_path / "umap_ref.pkl"
    first, _, _ = stable_umap(emb, keys, ref_p, 2, 10)
    second, n_new, refitted = stable_umap(emb, keys, ref_p, 2, 10)
    assert not refitted and n_new == 0
    assert np.array_equal(first, second)  # frozen, not merely similar


def test_new_points_transform_old_points_do_not_move(tmp_path):
    emb, keys = _data(20), _keys(20)
    ref_p = tmp_path / "umap_ref.pkl"
    first, _, _ = stable_umap(emb, keys, ref_p, 2, 10)
    emb2 = np.vstack([emb, _data(5, seed=99)])
    keys2 = keys + _keys(5, prefix="new")
    coords2, n_new, refitted = stable_umap(emb2, keys2, ref_p, 2, 10)
    assert not refitted and n_new == 5
    assert np.array_equal(coords2[:20], first)  # the frame did not move
    assert np.all(np.isfinite(coords2[20:]))
    # the frame grew: a third run sees no new points
    _, n_new3, refitted3 = stable_umap(emb2, keys2, ref_p, 2, 10)
    assert not refitted3 and n_new3 == 0


def test_subset_and_reordered_keys_keep_their_coords(tmp_path):
    emb, keys = _data(20), _keys(20)
    ref_p = tmp_path / "umap_ref.pkl"
    first, _, _ = stable_umap(emb, keys, ref_p, 2, 10)
    order = [5, 3, 17, 0]
    coords, n_new, refitted = stable_umap(emb[order], [keys[i] for i in order],
                                          ref_p, 2, 10)
    assert not refitted and n_new == 0
    assert np.array_equal(coords, first[order])


def test_rebuild_refits(tmp_path):
    emb, keys = _data(20), _keys(20)
    ref_p = tmp_path / "umap_ref.pkl"
    stable_umap(emb, keys, ref_p, 2, 10)
    _, _, refitted = stable_umap(emb, keys, ref_p, 2, 10, rebuild=True)
    assert refitted


def test_incompatible_ref_triggers_refit(tmp_path):
    emb, keys = _data(20), _keys(20)
    ref_p = tmp_path / "umap_ref.pkl"
    stable_umap(emb, keys, ref_p, 2, 10)
    # embedding dimensionality changed (different model) → refit
    _, _, refitted = stable_umap(_data(20, dim=8), keys, ref_p, 2, 10)
    assert refitted
    # component count changed → refit
    _, _, refitted2 = stable_umap(_data(20, dim=8), keys, ref_p, 3, 10)
    assert refitted2


def test_corrupt_pickle_is_treated_as_missing(tmp_path):
    ref_p = tmp_path / "umap_ref.pkl"
    ref_p.write_bytes(b"not a pickle")
    assert load_ref(ref_p) is None
    coords, _, refitted = stable_umap(_data(12), _keys(12), ref_p, 2, 6)
    assert refitted and coords.shape == (12, 2)


def test_ref_path_for_layout():
    p = ref_path_for(Path("d/train"), "dinov2")
    assert str(p).replace("\\", "/").endswith("d/train/embeddings_dinov2/umap_ref.pkl")
