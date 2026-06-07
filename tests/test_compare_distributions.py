import numpy as np
import pytest
from pathlib import Path
from PIL import Image


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
    projections = {"pca": np.random.rand(5, 2)}
    fig = build_projection_figure(paths_a, paths_b, projections, "train", "test", 10.5, 0.35)
    assert len(fig.data) == 2


def test_build_projection_figure_title_contains_metrics():
    from compare_distributions import build_projection_figure

    paths_a = [Path(f"a{i}.jpg") for i in range(2)]
    paths_b = [Path(f"b{i}.jpg") for i in range(2)]
    projections = {"pca": np.random.rand(4, 2)}
    fig = build_projection_figure(
        paths_a, paths_b, projections, "A", "B",
        fid_score=12.34, lpips_score=0.56, kid_score=0.001234, ssim_score=0.75,
    )
    assert "12.34" in fig.layout.title.text
    assert "0.56" in fig.layout.title.text
    assert "0.001234" in fig.layout.title.text
    assert "0.75" in fig.layout.title.text


def test_build_projection_figure_multi_method_buttons():
    from compare_distributions import build_projection_figure

    paths_a = [Path(f"a{i}.jpg") for i in range(3)]
    paths_b = [Path(f"b{i}.jpg") for i in range(2)]
    projections = {
        "pca": np.random.rand(5, 2),
        "tsne": np.random.rand(5, 2),
        "umap": np.random.rand(5, 2),
    }
    fig = build_projection_figure(
        paths_a, paths_b, projections, "A", "B",
        fid_score=1.0, lpips_score=0.1, kid_score=0.001, ssim_score=0.8,
    )
    assert len(fig.layout.updatemenus) == 1
    assert len(fig.layout.updatemenus[0].buttons) == 3


def test_compute_ssim_score_range(tmp_path):
    from compare_distributions import compute_ssim_score

    paths_a = _make_images(tmp_path / "a", 5)
    paths_b = _make_images(tmp_path / "b", 5)
    score = compute_ssim_score(paths_a, paths_b, n_pairs=3)
    assert 0.0 <= score <= 1.0


def test_compute_ssim_score_identical_images(tmp_path):
    from compare_distributions import compute_ssim_score

    # n_pairs=1 forces the single path to be picked from both lists, comparing image to itself
    paths = _make_images(tmp_path / "imgs", 1)
    score = compute_ssim_score(paths, paths, n_pairs=1)
    assert score > 0.99


