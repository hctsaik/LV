"""Unit tests for scripts/sample_data.py — the on-demand tiny synthetic
datasets that keep every tool's demo working when bundled demo/ is absent."""
from __future__ import annotations

from PIL import Image

from sample_data import (
    ensure_classifier_sample,
    ensure_compare_sample,
    ensure_detection_sample,
)


def test_classifier_sample_structure_and_idempotent(tmp_path):
    train = ensure_classifier_sample(tmp_path / "c", n_per=4)
    classes = sorted(p.name for p in train.iterdir() if p.is_dir())
    assert classes == ["classA", "classB", "classC"]
    assert len(list((train / "classA").glob("*.jpg"))) == 4
    assert ensure_classifier_sample(tmp_path / "c", n_per=4) == train  # idempotent


def test_compare_sample_two_folders(tmp_path):
    a, b = ensure_compare_sample(tmp_path / "cmp", n=5)
    assert len(list(a.glob("*.jpg"))) == 5 and len(list(b.glob("*.jpg"))) == 5


def test_detection_sample_valid_yolo(tmp_path):
    base = ensure_detection_sample(tmp_path / "d", n=6)
    imgs = sorted((base / "images").glob("*.png"))
    assert len(imgs) == 6
    assert (base / "classes.txt").read_text(encoding="utf-8").split() == ["spot", "scratch"]
    for img in imgs:  # each image has a matching one-box YOLO label
        parts = (base / "labels" / f"{img.stem}.txt").read_text().split()
        assert len(parts) == 5
        cid, (cx, cy, w, h) = int(parts[0]), map(float, parts[1:])
        assert cid in (0, 1) and 0 < cx < 1 and 0 < w < 1
    with Image.open(imgs[0]) as im:
        assert im.size == (96, 96)
