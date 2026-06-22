"""驗收:anomaly_heatmap(設計 05_anomaly_heatmap.md)。先紅:scripts/anomaly_heatmap.py 未實作。"""
import numpy as np
import pytest
from PIL import Image

from anomaly_heatmap import render_heatmap


def _green_crop(w, h):
    return Image.new("RGB", (w, h), (0, 170, 0))


def test_output_size_and_mode():  # AC1
    out = render_heatmap(np.zeros((4, 8)), _green_crop(160, 80))
    assert out.size == (160, 80)
    assert out.mode == "RGB"


def test_peak_localization():  # AC2(峰值落在 cell (2,5) 的空間範圍)
    gh, gw = 4, 8
    scores = np.zeros((gh, gw)); scores[2, 5] = 1.0
    W, H = 160, 80
    crop = _green_crop(W, H)
    out = np.asarray(render_heatmap(scores, crop, alpha=0.8)).astype(int)
    base = np.asarray(crop).astype(int)
    diff = np.abs(out - base).sum(axis=2)        # 每像素相對 crop 的改變量
    py, px = np.unravel_index(int(np.argmax(diff)), diff.shape)
    assert 2 * H / gh <= py < 3 * H / gh          # [40,60)
    assert 5 * W / gw <= px < 6 * W / gw          # [100,120)


def test_constant_map_no_nan():  # AC3
    out = np.asarray(render_heatmap(np.full((4, 4), 0.7), _green_crop(56, 56)))
    assert np.isfinite(out).all()


def test_alpha_zero_and_monotone_fade_in():  # AC4(alpha=0→crop;熱度隨分數淡入)
    crop = _green_crop(56, 56)
    # alpha=0 → 逐像素等於 crop
    assert np.abs(np.asarray(render_heatmap(np.random.default_rng(0).random((4, 4)),
                                            crop, alpha=0.0)).astype(int)
                  - np.asarray(crop).astype(int)).max() <= 1
    # 高分 cell 偏離原圖 > 低分區(熱度隨分數淡入,非全圖固定混合)
    gh, gw = 4, 8
    scores = np.zeros((gh, gw)); scores[2, 5] = 1.0
    W, H = 160, 80
    out = np.asarray(render_heatmap(scores, _green_crop(W, H), alpha=0.8)).astype(int)
    diff = np.abs(out - np.asarray(_green_crop(W, H)).astype(int)).sum(axis=2)
    r0, r1 = 2 * H // gh, 3 * H // gh
    c0, c1 = 5 * W // gw, 6 * W // gw
    cell = diff[r0:r1, c0:c1].mean()
    mask = np.ones_like(diff, dtype=bool); mask[r0:r1, c0:c1] = False
    assert cell > diff[mask].mean()


def test_deterministic():  # AC5
    scores = np.random.default_rng(1).random((4, 4))
    crop = _green_crop(56, 56)
    assert np.array_equal(np.asarray(render_heatmap(scores, crop)),
                          np.asarray(render_heatmap(scores, crop)))
