"""驗收:explanation_render —— 色盤與 display 不得沿用 anomaly jet/min-max。"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

import pytest

from dino_explain import AttentionResult, max_norm
from explanation_render import (
    CAPTION_ATTENTION,
    colorize,
    export_json_bytes,
    metadata_dict,
    render_grid,
    render_overlay,
    upsample_scores,
)


def test_viridis_not_jet_blue_at_zero():
    z = np.zeros((4, 4), dtype=np.float32)
    rgb = colorize(z, "viridis")
    # viridis 低點偏紫，不是 jet 的藍 (0,0,255)
    pixel = rgb[0, 0]
    assert pixel[2] < 120
    hi = colorize(np.ones((2, 2), dtype=np.float32), "viridis")[0, 0]
    assert hi[1] > hi[2]  # 高點偏黃綠


def test_copper_distinct_from_viridis():
    x = np.full((2, 2), 0.8, dtype=np.float32)
    v = colorize(x, "viridis")[0, 0]
    c = colorize(x, "copper")[0, 0]
    assert not np.allclose(v, c, atol=5)


def test_gradcam_blue_white_red_ramp():
    x = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    rgb = colorize(x, "blue_white_red")
    low, mid, high = rgb[0]
    assert low[2] > low[0]  # low evidence is blue
    assert np.allclose(mid, [255.0, 255.0, 255.0], atol=1.0)
    assert high[0] > high[2]  # high evidence is red


def test_gradcam_overlay_uses_blue_white_red_palette():
    base = Image.fromarray(np.full((24, 24, 3), 80, dtype=np.uint8))
    scores = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    out = np.asarray(render_overlay(base, scores, alpha=1.0, cmap="blue_white_red"))
    assert out.shape == (24, 24, 3)
    assert out[..., 0].max() > out[..., 2].max()  # red end is visible


def test_signed_gradcam_zero_is_transparent_and_sign_controls_color():
    base = Image.fromarray(np.full((12, 30, 3), 80, dtype=np.uint8))
    scores = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    out = np.asarray(render_overlay(
        base, scores, alpha=1.0, cmap="blue_white_red", signed=True))
    assert out[6, 2, 2] > out[6, 2, 0]       # negative is blue
    assert out[6, -3, 0] > out[6, -3, 2]     # positive is red
    neutral = np.asarray(render_overlay(
        base, np.zeros((1, 1), np.float32), alpha=1.0,
        cmap="blue_white_red", signed=True))
    assert np.allclose(neutral, 80, atol=1)    # exact zero reveals input


def test_signed_upsample_preserves_negative_range():
    up = upsample_scores(np.array([[-1.0, 1.0]], np.float32), (20, 4), signed=True)
    assert up.min() <= -0.99
    assert up.max() >= 0.99


def test_overlay_preserves_size_and_is_rgb():
    base = Image.fromarray(np.full((40, 60, 3), 80, dtype=np.uint8))
    scores = max_norm(np.linspace(0, 1, 12).reshape(3, 4).astype(np.float32))
    out = render_overlay(base, scores, alpha=0.5, cmap="viridis")
    assert out.size == (60, 40)
    assert out.mode == "RGB"


def test_uniform_scores_do_not_create_fake_hotspot():
    scores = np.full((4, 4), 0.3, dtype=np.float32)
    disp = max_norm(scores)
    assert float(disp.min()) == pytest.approx(1.0)
    up = upsample_scores(disp, (32, 32))
    assert float(up.std()) < 1e-5


def test_grid_draws_lines():
    base = Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8))
    g = render_grid(base, (2, 2), color=(255, 0, 0))
    arr = np.asarray(g)
    assert arr[20, :, 0].max() > 200 or arr[:, 20, 0].max() > 200


def test_metadata_json_roundtrip():
    r = AttentionResult(
        raw_patch_scores=np.ones((2, 2), np.float32),
        display_patch_scores=np.ones((2, 2), np.float32),
        grid=(2, 2), patch_size=14, n_register=0, n_cls=1,
        cls_mass=0.1, register_mass=0.0, patch_mass=0.9,
        method="last_layer_head_mean", head_index=None,
        model_id="dinov2_vits14", checkpoint_id="x",
        input_size=(224, 224), original_size=(64, 64),
        crop_bbox_px=None, keep_aspect=False, target_res=224,
        pooling="cls", scope="whole_image", source_id="s",
        input_image=Image.fromarray(np.zeros((8, 8, 3), np.uint8)),
        timing_s=0.01,
    )
    meta = metadata_dict(r)
    json.loads(export_json_bytes(meta))
    assert "input_image" not in meta
    assert CAPTION_ATTENTION.startswith("DINO")
