from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from app import (
    _build_cmp_figure,
    _build_viz_figure,
    _cmp_resolve_images,
    parse_folder_paths,
    read_classes_txt,
)


def test_cmp_resolve_images_flat_recurse_empty(tmp_path):
    # flat folder with images → returned as-is, no note (the tool's正解)
    flat = tmp_path / "flat"; flat.mkdir()
    Image.new("RGB", (8, 8)).save(flat / "a.jpg")
    Image.new("RGB", (8, 8)).save(flat / "b.png")
    paths, note = _cmp_resolve_images(flat)
    assert len(paths) == 2 and note is None
    # class-subfolder layout (no direct images) → recurse + 容錯 note
    root = tmp_path / "ds"
    (root / "cat").mkdir(parents=True); (root / "dog").mkdir()
    Image.new("RGB", (8, 8)).save(root / "cat" / "c.jpg")
    Image.new("RGB", (8, 8)).save(root / "dog" / "d.jpg")
    paths2, note2 = _cmp_resolve_images(root)
    assert len(paths2) == 2 and note2 is not None
    # cache dirs (embeddings_*/.thumbs) are skipped by the recursive fallback
    cache = tmp_path / "ds2" / "embeddings_x"; cache.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(cache / "thumb.jpg")
    assert _cmp_resolve_images(tmp_path / "ds2") == ([], None)
    # genuinely empty → nothing, no note
    empty = tmp_path / "empty"; empty.mkdir()
    assert _cmp_resolve_images(empty) == ([], None)


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


def test_build_viz_figure_no_redundant_title():
    # model·method is shown in the Model/Method selectboxes above the chart;
    # the in-figure centred title was removed because it overlapped the
    # top-left 全選/全不選 buttons (排版重疊). The figure must carry no title.
    records = _make_records(2, "train", "apple")
    coords = np.random.rand(2, 2)
    fig = _build_viz_figure(records, coords, [0, 1], "mobilenet", "t-SNE")
    assert not (fig.layout.title.text or "")


def test_build_viz_figure_disagreement_mode():
    # disagreement coloring: a single continuous-coloured points trace (carries
    # customdata for selection) + a cross-class pair-line trace; drag=select.
    records = _make_records(4, "train", "apple")
    coords = np.random.rand(4, 2)
    dis = np.array([0.0, 0.9, 0.3, 0.7])
    fig = _build_viz_figure(records, coords, [0, 1, 2, 3], "m", "PCA",
                            color_by="disagreement", disagreement=dis,
                            pairs=[(0, 1), (2, 3)])
    assert fig.layout.dragmode == "select"
    line_traces = [t for t in fig.data if getattr(t, "mode", "") == "lines"]
    pts = [t for t in fig.data if getattr(t, "mode", "") == "markers"]
    assert line_traces and pts                     # both lines and points present
    assert pts[0].customdata is not None           # selection still maps back
    # class mode keeps the 全選/全不選 buttons; disagreement mode drops them
    fig_cls = _build_viz_figure(records, coords, [0, 1, 2, 3], "m", "PCA")
    assert fig_cls.layout.updatemenus
    assert not fig.layout.updatemenus


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
