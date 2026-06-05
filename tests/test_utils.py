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


def test_save_figure_creates_html_and_png(tmp_path):
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from _utils import save_figure

    pf = go.Figure(go.Scatter(x=[1], y=[2]))
    mf, ax = plt.subplots()
    ax.plot([1], [2])

    save_figure(pf, mf, tmp_path, "out")
    plt.close(mf)

    assert (tmp_path / "out.html").exists()
    assert (tmp_path / "out.png").exists()


def test_save_figure_creates_output_dir(tmp_path):
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from _utils import save_figure

    out_dir = tmp_path / "new_dir"
    pf = go.Figure(go.Scatter(x=[1], y=[2]))
    mf, _ = plt.subplots()

    save_figure(pf, mf, out_dir, "out")
    plt.close(mf)

    assert (out_dir / "out.html").exists()
