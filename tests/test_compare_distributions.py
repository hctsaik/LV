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
    fig = build_projection_figure(paths_a, paths_b, projections, "A", "B", 12.34, 0.56)
    assert "12.34" in fig.layout.title.text
    assert "0.56" in fig.layout.title.text


def test_build_projection_figure_multi_method_buttons():
    from compare_distributions import build_projection_figure

    paths_a = [Path(f"a{i}.jpg") for i in range(3)]
    paths_b = [Path(f"b{i}.jpg") for i in range(2)]
    projections = {
        "pca": np.random.rand(5, 2),
        "tsne": np.random.rand(5, 2),
        "umap": np.random.rand(5, 2),
    }
    fig = build_projection_figure(paths_a, paths_b, projections, "A", "B", 1.0, 0.1)
    assert len(fig.layout.updatemenus) == 1
    assert len(fig.layout.updatemenus[0].buttons) == 3


def test_build_projection_figure_with_labels_splits_by_class():
    from compare_distributions import build_projection_figure

    paths_a = [Path(f"a{i}.jpg") for i in range(4)]
    paths_b = [Path(f"b{i}.jpg") for i in range(3)]
    projections = {"pca": np.random.rand(7, 2)}
    labels_a = ["cat", "cat", "dog", "dog"]
    labels_b = ["cat", "dog", "dog"]
    fig = build_projection_figure(
        paths_a, paths_b, projections, "train", "goal", 5.0, 0.3,
        labels_a=labels_a, labels_b=labels_b,
    )
    # 2 classes × 2 groups = 4 traces (cat/train, cat/goal, dog/train, dog/goal)
    assert len(fig.data) == 4
    trace_names = {t.name for t in fig.data}
    assert "cat (train)" in trace_names
    assert "dog (goal)" in trace_names


def test_build_projection_figure_labels_same_color_per_class():
    from compare_distributions import build_projection_figure

    paths_a = [Path("a0.jpg"), Path("a1.jpg")]
    paths_b = [Path("b0.jpg"), Path("b1.jpg")]
    projections = {"pca": np.random.rand(4, 2)}
    labels_a = ["cat", "dog"]
    labels_b = ["cat", "dog"]
    fig = build_projection_figure(
        paths_a, paths_b, projections, "A", "B", 1.0, 0.1,
        labels_a=labels_a, labels_b=labels_b,
    )
    color_by_name = {t.name: t.marker.color for t in fig.data}
    # Same class should share color across groups
    assert color_by_name["cat (A)"] == color_by_name["cat (B)"]
    assert color_by_name["dog (A)"] == color_by_name["dog (B)"]
    assert color_by_name["cat (A)"] != color_by_name["dog (A)"]


