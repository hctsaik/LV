"""Unit tests for F7 text-to-image search: the hnswlib-backed vector query
layer (pure) plus conditional smoke tests for the Chinese-CLIP towers
(skipped when the weights have not been downloaded)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from interaction import build_nn_index, find_similar_indices, find_similar_to_vector

_CLIP_DIR = (Path(__file__).resolve().parent.parent
             / "models" / "chinese-clip-vit-base-patch16")
_HAS_CLIP = (_CLIP_DIR / "config.json").exists()


# ── find_similar_to_vector (pure, hnswlib-backed) ───────────────────────

def _basis(n: int = 5, dim: int = 8) -> np.ndarray:
    return np.eye(n, dim, dtype=np.float32)


def test_vector_query_finds_matching_row():
    emb = _basis()
    q = np.zeros(8); q[2] = 1.0  # identical direction to row 2
    idxs, dists = find_similar_to_vector(emb, q, k=3)
    assert idxs[0] == 2
    assert dists[0] == pytest.approx(0.0, abs=1e-5)
    assert dists == sorted(dists)


def test_vector_query_k_clamped_to_n():
    emb = _basis(3)
    idxs, _ = find_similar_to_vector(emb, np.ones(8), k=99)
    assert sorted(idxs) == [0, 1, 2]


def test_vector_query_no_self_exclusion_and_reused_index():
    emb = _basis()
    nn = build_nn_index(emb)
    # the same index serves image-query (self-excluded) and text-query (not)
    img_idxs, _ = find_similar_indices(emb, 2, k=2, nn_index=nn)
    assert 2 not in img_idxs
    txt_idxs, _ = find_similar_to_vector(emb, emb[2], k=1, nn_index=nn)
    assert txt_idxs == [2]


def test_vector_query_dim_mismatch_raises():
    with pytest.raises(ValueError):
        find_similar_to_vector(_basis(), np.ones(5), k=1)


def test_vector_query_empty_matrix():
    assert find_similar_to_vector(np.zeros((0, 8)), np.ones(8)) == ([], [])


# ── Chinese-CLIP towers（權重存在才跑）──────────────────────────────────

@pytest.mark.skipif(not _HAS_CLIP, reason="Chinese-CLIP weights not downloaded")
def test_chinese_clip_text_and_image_share_space(tmp_path):
    from PIL import Image
    from _utils import load_model, load_text_encoder, supports_text_query

    name = _CLIP_DIR.name
    assert supports_text_query(name)
    text_fn = load_text_encoder(name)
    v1 = text_fn("斑馬")
    v2 = text_fn("夜間的城市街道")
    assert v1.shape == v2.shape and v1.ndim == 1
    assert np.all(np.isfinite(v1))
    assert not np.allclose(v1, v2)  # different queries → different vectors

    img = tmp_path / "x.jpg"
    arr = np.random.default_rng(0).integers(0, 255, (64, 64, 3)).astype("uint8")
    Image.fromarray(arr).save(img)
    emb_fn = load_model(name)
    iv = emb_fn(img)
    assert iv.shape == v1.shape  # same space → text query over image rows works
    assert np.all(np.isfinite(iv))


@pytest.mark.skipif(not _HAS_CLIP, reason="Chinese-CLIP weights not downloaded")
def test_available_models_lists_chinese_clip():
    from _utils import available_models
    assert _CLIP_DIR.name in available_models()
