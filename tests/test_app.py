from pathlib import Path
import numpy as np
import pytest
from app import _build_cmp_figure, _build_viz_figure, parse_folder_paths, read_classes_txt


def test_read_classes_txt_found(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    (tmp_path / "classes.txt").write_text("apple\nbanana\norange\n")
    assert read_classes_txt(folder) == ["apple", "banana", "orange"]


def test_read_classes_txt_not_found(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    assert read_classes_txt(folder) is None


def test_read_classes_txt_empty_file(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    (tmp_path / "classes.txt").write_text("\n\n")
    assert read_classes_txt(folder) is None


def test_parse_folder_paths_basic(tmp_path):
    text = f"{tmp_path}/train\n{tmp_path}/test"
    result = parse_folder_paths(text)
    assert result == [Path(f"{tmp_path}/train"), Path(f"{tmp_path}/test")]


def test_parse_folder_paths_ignores_blank_lines(tmp_path):
    text = f"{tmp_path}/train\n\n  \n{tmp_path}/test"
    result = parse_folder_paths(text)
    assert len(result) == 2


def test_parse_folder_paths_empty():
    assert parse_folder_paths("") == []
    assert parse_folder_paths("   \n  ") == []


# --- _build_viz_figure ---

def _make_records(n: int, split: str = "train", label: str = "cat") -> list[dict]:
    return [{"path": Path(f"{split}/images/img_{i}.jpg"), "split": split, "label": label} for i in range(n)]


def test_build_viz_figure_trace_per_label_split():
    records = _make_records(3, "train", "cat") + _make_records(2, "test", "dog")
    coords = np.random.rand(5, 2)
    indices = list(range(5))
    fig = _build_viz_figure(records, coords, indices, "resnet18", "PCA")
    # One trace per (label × split) combination that has data
    assert len(fig.data) == 2


def test_build_viz_figure_title_contains_model_and_method():
    records = _make_records(2, "train", "apple")
    coords = np.random.rand(2, 2)
    fig = _build_viz_figure(records, coords, [0, 1], "mobilenet", "t-SNE")
    assert "mobilenet" in fig.layout.title.text
    assert "t-SNE" in fig.layout.title.text


def test_build_viz_figure_split_filter():
    records = _make_records(3, "train", "cat") + _make_records(2, "test", "cat")
    coords = np.random.rand(5, 2)
    train_indices = [i for i, r in enumerate(records) if r["split"] == "train"]
    fig = _build_viz_figure(records, coords, train_indices, "resnet18", "PCA")
    # Only train split — one trace
    assert len(fig.data) == 1
    assert all(t.name.endswith("(train)") for t in fig.data)


# --- _build_cmp_figure ---

def test_build_cmp_figure_has_two_traces():
    paths_a = [Path(f"a{i}.jpg") for i in range(3)]
    paths_b = [Path(f"b{i}.jpg") for i in range(2)]
    proj = np.random.rand(5, 2)
    fig = _build_cmp_figure(paths_a, paths_b, proj, "train", "goal")
    assert len(fig.data) == 2


def test_build_cmp_figure_group_names():
    paths_a = [Path("a0.jpg")]
    paths_b = [Path("b0.jpg")]
    proj = np.random.rand(2, 2)
    fig = _build_cmp_figure(paths_a, paths_b, proj, "GroupA", "GroupB")
    names = [t.name for t in fig.data]
    assert "GroupA" in names
    assert "GroupB" in names
