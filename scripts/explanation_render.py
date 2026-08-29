"""可解釋性疊圖。各方法使用不同語意色盤；display 用 max-norm，不用 min-max。"""
from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# viridis 近似：紫→青→黃。不用 jet。
_VIRIDIS = np.array([
    [0.267, 0.005, 0.329],
    [0.253, 0.265, 0.530],
    [0.127, 0.566, 0.551],
    [0.369, 0.789, 0.383],
    [0.993, 0.906, 0.144],
], dtype=np.float32)

# Grad-CAM：藍→白→紅。紅只表示指定 target 的證據較高，不等於「更壞」。
_BLUE_WHITE_RED = np.array([
    [0.08, 0.28, 0.70],
    [0.35, 0.62, 0.90],
    [1.00, 1.00, 1.00],
    [0.95, 0.50, 0.35],
    [0.70, 0.00, 0.00],
], dtype=np.float32)

# Legacy copper kept for callers that may have saved the old renderer choice.
_COPPER = np.array([
    [0.00, 0.00, 0.00],
    [0.35, 0.15, 0.05],
    [0.70, 0.35, 0.10],
    [0.90, 0.62, 0.28],
    [1.00, 0.90, 0.65],
], dtype=np.float32)

CAPTION_ATTENTION = "DINO 關注區域（僅此圖相對亮度）"
CAPTION_GRADCAM = (
    "對比分類證據（藍＝支持 reference、透明＝中性、紅＝支持 target；"
    "不是全域 anomaly）"
)
CAPTION_ANOMALY = "相對 Normal Bank 的不相似區域（不是 DINO attention）"
CAPTION_INPUT = "模型實際輸入"
CAPTION_ORIGINAL = "原圖（脈絡，未疊圖）"


def _ramp(norm: np.ndarray, stops: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(norm, dtype=np.float32), 0.0, 1.0)
    n = len(stops) - 1
    t = x * n
    i = np.clip(np.floor(t).astype(int), 0, n - 1)
    f = (t - i)[..., None]
    rgb = stops[i] * (1.0 - f) + stops[i + 1] * f
    return (rgb * 255.0).astype(np.float32)


def colorize(norm: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    if cmap == "blue_white_red":
        stops = _BLUE_WHITE_RED
    elif cmap == "copper":
        stops = _COPPER
    else:
        stops = _VIRIDIS
    return _ramp(norm, stops)


def upsample_scores(scores, size: tuple[int, int], *, signed: bool = False) -> np.ndarray:
    """size = (W, H). Bilinear only; preserve the caller's normalization."""
    s = np.asarray(scores, dtype=np.float32)
    w, h = size
    encoded = (np.clip(s, -1, 1) + 1.0) * 0.5 if signed else np.clip(s, 0, 1)
    plane = Image.fromarray((encoded * 255).astype(np.uint8), "L")
    up = plane.resize((w, h), Image.BILINEAR)
    decoded = np.asarray(up, dtype=np.float32) / 255.0
    return decoded * 2.0 - 1.0 if signed else decoded


def render_overlay(base, scores, *, alpha: float = 0.45,
                   cmap: str = "viridis", signed: bool = False) -> Image.Image:
    if not isinstance(base, Image.Image):
        base = Image.fromarray(np.asarray(base).astype(np.uint8))
    base = base.convert("RGB")
    w, h = base.size
    heat_n = upsample_scores(scores, (w, h), signed=signed)
    color_input = (heat_n + 1.0) * 0.5 if signed else heat_n
    heat = colorize(color_input, cmap)
    a = min(max(float(alpha), 0.0), 1.0)
    # Signed zero is neutral evidence, so it must reveal the source image
    # instead of painting an opaque white patch.
    strength = np.abs(heat_n) if signed else heat_n
    wgt = (a * strength)[..., None]
    out = (1.0 - wgt) * np.asarray(base, dtype=np.float32) + wgt * heat
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def render_grid(base, grid: tuple[int, int], *, color=(255, 255, 255),
                width: int = 1) -> Image.Image:
    if not isinstance(base, Image.Image):
        base = Image.fromarray(np.asarray(base).astype(np.uint8))
    im = base.convert("RGB").copy()
    gh, gw = int(grid[0]), int(grid[1])
    if gh <= 0 or gw <= 0:
        return im
    w, h = im.size
    draw = ImageDraw.Draw(im)
    for i in range(1, gw):
        x = int(round(i * w / gw))
        draw.line([(x, 0), (x, h - 1)], fill=color, width=width)
    for j in range(1, gh):
        y = int(round(j * h / gh))
        draw.line([(0, y), (w - 1, y)], fill=color, width=width)
    return im


def metadata_dict(result: Any) -> dict:
    """JSON 可序列化的 provenance（不含影像 pixels）。"""
    d = result if isinstance(result, dict) else getattr(result, "as_dict", lambda: {})()
    skip = {"raw_patch_scores", "display_patch_scores", "input_image", "extras"}
    out = {}
    for k, v in d.items():
        if k in skip:
            continue
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, Image.Image):
            continue
        if isinstance(v, tuple):
            out[k] = list(v)
        else:
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = str(v)
    raw = d.get("raw_patch_scores")
    if isinstance(raw, np.ndarray):
        out["raw_min"] = float(raw.min())
        out["raw_max"] = float(raw.max())
        out["raw_sum"] = float(raw.sum())
        out["grid"] = [int(d.get("grid", (0, 0))[0]), int(d.get("grid", (0, 0))[1])]
    extras = d.get("extras") or {}
    if isinstance(extras, dict):
        out.update({k: v for k, v in extras.items() if isinstance(v, (str, int, float, bool, list))})
    return out


def export_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def export_json_bytes(meta: dict) -> bytes:
    return json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
