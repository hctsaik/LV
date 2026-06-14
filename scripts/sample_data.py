"""On-demand tiny synthetic sample datasets, so every tool's one-click demo
works even when the bundled ``demo/`` data isn't provisioned (it is gitignored,
so a fresh clone / platform deploy has no demo data — the demo buttons would
otherwise be disabled).

Each ``ensure_*`` generates a small, deterministic colored-noise dataset into a
cache dir and returns the path; it is idempotent (skips work if already there).
Framework-free (no streamlit) and unit-testable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _noise_image(path: Path, color_axis: int | None, *, size: int = 64, seed: int = 0):
    """A colored-noise JPG; ``color_axis`` (0=R,1=G,2=B) gives the class a hue."""
    arr = np.random.default_rng(seed).integers(0, 255, (size, size, 3)).astype("uint8")
    if color_axis is not None:
        arr[:, :, color_axis] = 255
    Image.fromarray(arr).save(path, quality=85)


def ensure_classifier_sample(root: Path, *, n_per: int = 8) -> Path:
    """Tiny classifier dataset ``root/train/{classA,classB,classC}/*.jpg`` (for
    完整度熱力圖 / 組考卷 / 灰帶覆核). Returns the ``train`` dir."""
    train = Path(root) / "train"
    classes = ("classA", "classB", "classC")
    ready = all((train / c).exists() and any((train / c).glob("*.jpg"))
                for c in classes)
    if not ready:
        for ci, cls in enumerate(classes):
            d = train / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_per):
                _noise_image(d / f"{cls}_{i:02d}.jpg", ci, seed=ci * 100 + i)
    return train


def ensure_compare_sample(root: Path, *, n: int = 10) -> tuple[Path, Path]:
    """Two image folders ``setA`` (R-biased) / ``setB`` (B-biased) for Compare
    Distributions — distinct enough that FID/coverage-gap are meaningful."""
    a, b = Path(root) / "setA", Path(root) / "setB"
    if not (a.exists() and any(a.glob("*.jpg")) and b.exists() and any(b.glob("*.jpg"))):
        a.mkdir(parents=True, exist_ok=True)
        b.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            _noise_image(a / f"a_{i:02d}.jpg", 0, seed=i)
            _noise_image(b / f"b_{i:02d}.jpg", 2, seed=100 + i)
    return a, b


def ensure_detection_sample(root: Path, *, n: int = 8) -> Path:
    """Tiny YOLO detection dataset ``root/{images,labels}`` + ``classes.txt``
    (1 defect box/image across 2 classes) for Visualize object mode / 評估.
    Returns ``root``."""
    base = Path(root)
    images, labels = base / "images", base / "labels"
    if not (images.exists() and any(images.glob("*.png"))):
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        (base / "classes.txt").write_text("spot\nscratch\n", encoding="utf-8")
        rng = np.random.default_rng(0)
        size = 96
        for i in range(n):
            arr = (rng.normal(0.5, 0.05, (size, size, 3)) * 255).clip(0, 255).astype("uint8")
            cls = i % 2
            cx, cy, bw, bh = float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.3, 0.7)), 0.2, 0.2
            x0, y0 = int((cx - bw / 2) * size), int((cy - bh / 2) * size)
            x1, y1 = int((cx + bw / 2) * size), int((cy + bh / 2) * size)
            arr[y0:y1, x0:x1] = 230 if cls == 0 else 30  # a bright/dark defect patch
            Image.fromarray(arr).save(images / f"img_{i:02d}.png")
            (labels / f"img_{i:02d}.txt").write_text(
                f"{cls} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}\n", encoding="utf-8")
    return base
