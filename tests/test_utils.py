import numpy as np
import pytest
from pathlib import Path


def test_extract_embeddings_shape(tmp_path):
    from _utils import extract_embeddings

    def mock_embed(path: Path) -> np.ndarray:
        return np.ones(64)

    paths = [tmp_path / f"{i}.jpg" for i in range(5)]
    for p in paths:
        p.write_bytes(b"x")

    result = extract_embeddings(paths, mock_embed)
    assert result.shape == (5, 64)


def test_extract_embeddings_preserves_values(tmp_path):
    from _utils import extract_embeddings

    def mock_embed(path: Path) -> np.ndarray:
        return np.array([float(path.stem)])

    paths = [tmp_path / f"{i}.jpg" for i in range(3)]
    for p in paths:
        p.write_bytes(b"x")

    result = extract_embeddings(paths, mock_embed)
    assert result[0, 0] == 0.0
    assert result[1, 0] == 1.0
    assert result[2, 0] == 2.0


