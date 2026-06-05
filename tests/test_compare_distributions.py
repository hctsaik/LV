import numpy as np
import pytest
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _make_images(folder: Path, n: int) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = folder / f"img_{i}.jpg"
        Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        ).save(p)
        paths.append(p)
    return paths


def test_get_image_paths_finds_jpg(tmp_path):
    from compare_distributions import get_image_paths

    _make_images(tmp_path, 3)
    paths = get_image_paths(tmp_path)
    assert len(paths) == 3
    assert all(p.suffix == ".jpg" for p in paths)


def test_get_image_paths_empty_folder(tmp_path):
    from compare_distributions import get_image_paths

    assert get_image_paths(tmp_path) == []


def test_build_projection_figure_has_two_traces():
    from compare_distributions import build_projection_figure

    paths_a = [Path(f"a{i}.jpg") for i in range(3)]
    paths_b = [Path(f"b{i}.jpg") for i in range(2)]
    pca_2d = np.random.rand(5, 2)
    fig = build_projection_figure(paths_a, paths_b, pca_2d, "train", "test", 10.5, 0.35)
    assert len(fig.data) == 2


def test_build_projection_figure_title_contains_metrics():
    from compare_distributions import build_projection_figure

    paths_a = [Path(f"a{i}.jpg") for i in range(2)]
    paths_b = [Path(f"b{i}.jpg") for i in range(2)]
    pca_2d = np.random.rand(4, 2)
    fig = build_projection_figure(paths_a, paths_b, pca_2d, "A", "B", 12.34, 0.56)
    assert "12.34" in fig.layout.title.text
    assert "0.56" in fig.layout.title.text


def test_build_matplotlib_figure_returns_figure():
    from compare_distributions import build_matplotlib_figure

    paths_a = [Path(f"a{i}.jpg") for i in range(2)]
    paths_b = [Path(f"b{i}.jpg") for i in range(2)]
    pca_2d = np.random.rand(4, 2)
    fig = build_matplotlib_figure(paths_a, paths_b, pca_2d, "A", "B", 5.0, 0.3)
    assert fig is not None
    plt.close("all")
