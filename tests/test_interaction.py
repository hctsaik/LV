"""Unit/regression tests for scripts/interaction.py and the progress_cb
streaming change in _utils.extract_embeddings."""
from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from interaction import (
    build_nn_index,
    compute_outlier_scores,
    draw_yolo_boxes,
    find_similar_indices,
    parse_folder_paths,
    records_to_csv,
    selection_points_to_indices,
    yolo_label_path_for,
    zip_selected_images,
)


# ── module hygiene ──────────────────────────────────────────────────────

def test_interaction_no_streamlit_import():
    import interaction
    assert "st" not in vars(interaction)
    assert not any(m.startswith("streamlit") for m in vars(interaction).values()
                   if isinstance(m, str))
    src = Path(interaction.__file__).read_text(encoding="utf-8")
    assert "import streamlit" not in src


# ── parse_folder_paths ──────────────────────────────────────────────────

def test_parse_folder_paths_crlf():
    paths = parse_folder_paths("C:/data/a\r\nC:/data/b\r\n")
    assert paths == [Path("C:/data/a"), Path("C:/data/b")]
    assert not str(paths[0]).endswith("\r")


def test_parse_folder_paths_strips_whitespace():
    assert parse_folder_paths("  C:/x  \n\t C:/y \n") == [Path("C:/x"), Path("C:/y")]


def test_parse_folder_paths_blank_and_empty():
    assert parse_folder_paths("") == []
    assert parse_folder_paths("   \n \r\n  ") == []


def test_parse_folder_paths_unicode():
    assert parse_folder_paths("C:/資料/訓練集") == [Path("C:/資料/訓練集")]


# ── find_similar_indices ────────────────────────────────────────────────

@pytest.fixture()
def emb10():
    return np.random.default_rng(0).normal(size=(10, 16))


def test_find_similar_excludes_self(emb10):
    idx, _ = find_similar_indices(emb10, 3, k=5)
    assert 3 not in idx


def test_find_similar_returns_k(emb10):
    idx, dist = find_similar_indices(emb10, 0, k=4)
    assert len(idx) == 4 and len(dist) == 4


def test_find_similar_distances_sorted_ascending(emb10):
    _, dist = find_similar_indices(emb10, 0, k=9)
    assert dist == sorted(dist)


def test_find_similar_k_greater_than_n(emb10):
    idx, _ = find_similar_indices(emb10, 0, k=99)
    assert len(idx) == 9  # n-1 others
    assert 0 not in idx


def test_find_similar_k_zero(emb10):
    idx, dist = find_similar_indices(emb10, 0, k=0)
    assert idx == [] and dist == []


def test_find_similar_out_of_range(emb10):
    with pytest.raises(IndexError):
        find_similar_indices(emb10, 10, k=3)
    with pytest.raises(IndexError):
        find_similar_indices(emb10, -1, k=3)


def test_find_similar_duplicate_embeddings():
    e = np.random.default_rng(1).normal(size=(5, 8))
    e[2] = e[0]  # duplicate of the query
    idx, dist = find_similar_indices(e, 0, k=2)
    assert idx[0] == 2 and dist[0] == pytest.approx(0.0, abs=1e-9)
    assert 0 not in idx


def test_find_similar_with_prebuilt_index(emb10):
    nn = build_nn_index(emb10)
    a = find_similar_indices(emb10, 1, k=3, nn_index=nn)
    b = find_similar_indices(emb10, 1, k=3)
    assert a == b


# ── compute_outlier_scores ──────────────────────────────────────────────

def test_outlier_scores_shape(emb10):
    s = compute_outlier_scores(emb10[:4], emb10[4:], k=3)
    assert s.shape == (4,) and s.dtype.kind == "f"


def test_outlier_scores_deterministic_ordering():
    rng = np.random.default_rng(2)
    base = rng.normal(size=16)
    reference = base + rng.normal(scale=0.01, size=(20, 16))
    # candidates progressively rotated away from the reference direction
    other = rng.normal(size=16)
    candidates = np.stack([
        base + t * other for t in (0.0, 0.5, 2.0, 8.0)
    ])
    s = compute_outlier_scores(candidates, reference, k=5)
    assert list(np.argsort(s)) == [0, 1, 2, 3]


def test_outlier_scores_identical_point_low(emb10):
    cand = np.vstack([emb10[0], emb10[0] * 3.7])  # cosine: identical direction
    s = compute_outlier_scores(cand, emb10, k=1)
    assert s[0] == pytest.approx(0.0, abs=1e-9)
    assert s[1] == pytest.approx(s[0], abs=1e-9)  # scale-invariant (cosine)


def test_outlier_scores_k_clamped(emb10):
    s = compute_outlier_scores(emb10[:3], emb10[:2], k=99)
    assert s.shape == (3,)


def test_outlier_scores_empty_candidates(emb10):
    s = compute_outlier_scores(np.zeros((0, 16)), emb10, k=3)
    assert s.shape == (0,)


def test_outlier_scores_self_exclusion(emb10):
    with_self = compute_outlier_scores(emb10, emb10, k=2, candidates_in_reference=True)
    without = compute_outlier_scores(emb10, emb10, k=2, candidates_in_reference=False)
    # including self deflates the score (self-distance ~0 pulls the mean down)
    assert (with_self >= without - 1e-12).all()
    assert with_self.mean() > without.mean()


# ── selection_points_to_indices ─────────────────────────────────────────

def test_selection_empty():
    assert selection_points_to_indices([]) == []
    assert selection_points_to_indices(None) == []


def test_selection_single_point():
    assert selection_points_to_indices([{"customdata": [5]}]) == [5]


def test_selection_multiple_and_order():
    pts = [{"customdata": [9]}, {"customdata": [2]}, {"customdata": [7]}]
    assert selection_points_to_indices(pts) == [9, 2, 7]


def test_selection_dedupes():
    pts = [{"customdata": [4]}, {"customdata": [4]}, {"customdata": [1]}]
    assert selection_points_to_indices(pts) == [4, 1]


def test_selection_scalar_customdata():
    assert selection_points_to_indices([{"customdata": 7}]) == [7]


def test_selection_missing_customdata_skipped():
    pts = [{"x": 1.0}, {"customdata": None}, {"customdata": []}, {"customdata": [3]}]
    assert selection_points_to_indices(pts) == [3]


def test_selection_returns_python_ints():
    out = selection_points_to_indices([{"customdata": [np.int64(6)]}])
    assert out == [6] and type(out[0]) is int


# ── records_to_csv ──────────────────────────────────────────────────────

def _records(tmp_path: Path | None = None, n: int = 4) -> list[dict]:
    base = tmp_path or Path("ds")
    recs = []
    for i in range(n):
        recs.append({"path": base / f"img_{i}.jpg", "split": "train",
                     "label": f"class{i % 2}"})
    return recs


def test_csv_header():
    text = records_to_csv(_records(), [0, 1])
    assert text.splitlines()[0] == "index,filename,path,label,split"


def test_csv_row_count_and_subset():
    text = records_to_csv(_records(), [2, 0])
    lines = text.splitlines()
    assert len(lines) == 3
    assert lines[1].startswith("2,img_2.jpg")
    assert lines[2].startswith("0,img_0.jpg")


def test_csv_empty_indices():
    text = records_to_csv(_records(), [])
    assert text.splitlines() == ["index,filename,path,label,split"]


def test_csv_unicode_roundtrip():
    recs = [{"path": Path("資料/貓咪 01.jpg"), "split": "訓練", "label": "貓"}]
    text = records_to_csv(recs, [0])
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[1][1] == "貓咪 01.jpg" and rows[1][3] == "貓"


def test_csv_comma_in_name_quoted():
    recs = [{"path": Path("a,b.jpg"), "split": "s", "label": "l"}]
    rows = list(csv.reader(io.StringIO(records_to_csv(recs, [0]))))
    assert rows[1][1] == "a,b.jpg"


# ── zip_selected_images ─────────────────────────────────────────────────

def _make_imgs(tmp_path: Path, names: list[str]) -> list[dict]:
    recs = []
    for name in names:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (200, 30, 30)).save(p)
        recs.append({"path": p, "split": "train", "label": "x"})
    return recs


def test_zip_valid_archive_with_selected(tmp_path):
    recs = _make_imgs(tmp_path, ["a.jpg", "b.jpg", "c.jpg"])
    data = zip_selected_images(recs, [0, 2])
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = zf.namelist()
    assert "manifest.csv" in names
    assert sum(n.startswith("images/") for n in names) == 2
    assert any(n.endswith("a.jpg") for n in names)
    assert any(n.endswith("c.jpg") for n in names)
    assert not any(n.endswith("b.jpg") for n in names)


def test_zip_missing_file_recorded(tmp_path):
    recs = _make_imgs(tmp_path, ["a.jpg"])
    recs.append({"path": tmp_path / "ghost.jpg", "split": "train", "label": "x"})
    data = zip_selected_images(recs, [0, 1])
    zf = zipfile.ZipFile(io.BytesIO(data))
    assert sum(n.startswith("images/") for n in zf.namelist()) == 1
    manifest = zf.read("manifest.csv").decode("utf-8")
    rows = list(csv.reader(io.StringIO(manifest)))
    statuses = {r[1]: r[5] for r in rows[1:]}
    assert statuses["a.jpg"] == "ok"
    assert statuses["ghost.jpg"] == "missing"


def test_zip_empty_indices():
    data = zip_selected_images([], [])
    zf = zipfile.ZipFile(io.BytesIO(data))
    assert zf.namelist() == ["manifest.csv"]


def test_zip_duplicate_filenames_disambiguated(tmp_path):
    recs = _make_imgs(tmp_path, ["a/img.jpg", "b/img.jpg"])
    data = zip_selected_images(recs, [0, 1])
    zf = zipfile.ZipFile(io.BytesIO(data))
    members = [n for n in zf.namelist() if n.startswith("images/")]
    assert len(members) == len(set(members)) == 2


def test_zip_unicode_filename(tmp_path):
    recs = _make_imgs(tmp_path, ["貓咪.jpg"])
    data = zip_selected_images(recs, [0])
    zf = zipfile.ZipFile(io.BytesIO(data))
    assert any("貓咪" in n for n in zf.namelist())


# ── YOLO overlay helpers ────────────────────────────────────────────────

def test_yolo_label_path_for():
    p = yolo_label_path_for(Path("ds/train/images/x 1.jpg"))
    assert p == Path("ds/train/labels/x 1.txt")


def test_draw_yolo_boxes_missing_label(tmp_path):
    img_p = tmp_path / "i.jpg"
    Image.new("RGB", (32, 32)).save(img_p)
    out = draw_yolo_boxes(img_p, tmp_path / "none.txt", ["a"])
    assert out.size == (32, 32)


def test_draw_yolo_boxes_with_label(tmp_path):
    img_p = tmp_path / "i.jpg"
    Image.new("RGB", (64, 64), (255, 255, 255)).save(img_p)
    lbl = tmp_path / "i.txt"
    lbl.write_text("0 0.5 0.5 0.5 0.5\nbroken line\n")
    out = draw_yolo_boxes(img_p, lbl, ["thing"])
    # the box outline must have painted at least one non-white pixel
    arr = np.asarray(out)
    assert (arr != 255).any()


# ── _utils.extract_embeddings progress_cb ───────────────────────────────

from _utils import extract_embeddings  # noqa: E402


def _fake_embed(path: Path) -> np.ndarray:
    return np.full(4, float(len(path.name)))


def _img_paths(tmp_path: Path, n: int = 5) -> list[Path]:
    paths = []
    for i in range(n):
        p = tmp_path / f"f{i}.jpg"
        p.write_bytes(b"x")
        paths.append(p)
    return paths


def test_progress_cb_called_n_times(tmp_path):
    paths = _img_paths(tmp_path, 5)
    calls: list[tuple[int, int]] = []
    extract_embeddings(paths, _fake_embed, progress_cb=lambda d, t: calls.append((d, t)))
    assert len(calls) == 5
    assert [c[0] for c in calls] == [1, 2, 3, 4, 5]
    assert all(c[1] == 5 for c in calls)


def test_progress_cb_monotonic_and_completes(tmp_path):
    paths = _img_paths(tmp_path, 3)
    calls = []
    extract_embeddings(paths, _fake_embed, progress_cb=lambda d, t: calls.append((d, t)))
    dones = [c[0] for c in calls]
    assert dones == sorted(dones)
    assert calls[-1] == (3, 3)


def test_progress_cb_cache_hit_called_once_full(tmp_path):
    paths = _img_paths(tmp_path, 4)
    cache = tmp_path / "emb" / "embeddings.npz"
    extract_embeddings(paths, _fake_embed, cache_path=cache)  # warm
    calls = []
    out = extract_embeddings(paths, _fake_embed, cache_path=cache,
                             progress_cb=lambda d, t: calls.append((d, t)))
    assert calls == [(4, 4)]
    assert out.shape == (4, 4)


def test_progress_cb_none_back_compat(tmp_path):
    paths = _img_paths(tmp_path, 2)
    out = extract_embeddings(paths, _fake_embed)
    assert out.shape == (2, 4)


# ── thumbnail pipeline (UX review W3) ───────────────────────────────────

from interaction import (  # noqa: E402
    ensure_thumbnails,
    make_thumbnail,
    spatial_order,
    thumbnail_path_for,
)


def _real_image(tmp_path: Path, name: str = "img.jpg", size=(640, 480)) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / name
    arr = np.random.default_rng(0).integers(0, 255, (*size[::-1], 3)).astype("uint8")
    Image.fromarray(arr).save(p)
    return p


def test_thumbnail_created_and_cached(tmp_path):
    src = _real_image(tmp_path)
    out1 = make_thumbnail(src, size=256)
    assert out1.exists() and out1.suffix == ".webp"
    assert out1.parent == src.parent / ".thumbs" / "256"
    with Image.open(out1) as im:
        assert max(im.size) <= 256
    mtime = out1.stat().st_mtime_ns
    out2 = make_thumbnail(src, size=256)  # cache hit — not regenerated
    assert out2 == out1
    assert out2.stat().st_mtime_ns == mtime


def test_thumbnail_key_changes_when_source_changes(tmp_path):
    src = _real_image(tmp_path)
    key1 = thumbnail_path_for(src)
    _real_image(tmp_path, size=(320, 240))  # overwrite source
    key2 = thumbnail_path_for(src)
    assert key1 != key2


def test_thumbnail_no_collision_same_name_different_dirs(tmp_path):
    a = _real_image(tmp_path / "a", "same.jpg")
    b = _real_image(tmp_path / "b", "same.jpg")
    assert thumbnail_path_for(a).parent != thumbnail_path_for(b).parent


def test_thumbnail_missing_source_raises(tmp_path):
    with pytest.raises(OSError):
        make_thumbnail(tmp_path / "nope.jpg")


def test_ensure_thumbnails_skips_broken_and_reports_count(tmp_path):
    good = [_real_image(tmp_path, f"g{i}.jpg") for i in range(3)]
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    calls = []
    n_ok = ensure_thumbnails([*good, broken],
                             progress_cb=lambda d, t: calls.append((d, t)))
    assert n_ok == 3
    assert calls[-1] == (4, 4)
    assert len(calls) == 4
    for p in good:
        assert thumbnail_path_for(p).exists()


# ── spatial_order (UX review: grid mirrors the scatter) ─────────────────

def test_spatial_order_top_row_first_then_left_to_right():
    #  layout:   2(top-left)   3(top-right)
    #            0(bot-left)   1(bot-right)
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    assert spatial_order(coords, [0, 1, 2, 3], n_rows=2) == [2, 3, 0, 1]


def test_spatial_order_constant_y_falls_back_to_x():
    coords = np.array([[5.0, 1.0], [1.0, 1.0], [3.0, 1.0]])
    assert spatial_order(coords, [0, 1, 2]) == [1, 2, 0]


def test_spatial_order_empty_and_subset():
    coords = np.array([[0.0, 0.0], [1.0, 5.0], [2.0, 9.0]])
    assert spatial_order(coords, []) == []
    assert spatial_order(coords, [2]) == [2]
