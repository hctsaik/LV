"""anomaly_heatmap:把 patch 分數圖上採樣疊回物件裁切圖,產生「哪裡怪」熱區。

設計:3_Architect_Design/05_anomaly_heatmap.md。標準淡入疊加:per-pixel 權重 w=alpha*norm,
低分處幾乎保留原圖、高分處才疊上色階 → 峰值即與原圖差最大處。
色階用零依賴的 numpy jet 近似(本環境未必裝 matplotlib)。
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _jet(norm: np.ndarray) -> np.ndarray:
    """jet 近似:norm (H,W) ∈[0,1] → RGB (H,W,3) 0..255(藍→青→黃→紅)。"""
    x = np.clip(norm, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1) * 255.0


def render_heatmap(patch_scores, crop, *, alpha: float = 0.5,
                   vmin: float | None = None, vmax: float | None = None,
                   cmap: str = "jet") -> Image.Image:
    if not isinstance(crop, Image.Image):
        crop = Image.fromarray(np.asarray(crop).astype(np.uint8))
    crop = crop.convert("RGB")
    W, H = crop.size

    s = np.asarray(patch_scores, dtype=float)
    lo = float(s.min()) if vmin is None else float(vmin)
    hi = float(s.max()) if vmax is None else float(vmax)
    if hi <= lo:
        norm = np.full(s.shape, 0.5, dtype=float)
    else:
        norm = np.clip((s - lo) / (hi - lo), 0.0, 1.0)

    norm_up = (np.asarray(Image.fromarray((norm * 255).astype(np.uint8))
                          .resize((W, H), Image.BILINEAR), dtype=float) / 255.0)
    heat = _jet(norm_up)                                     # (H,W,3) 0..255

    a = min(max(float(alpha), 0.0), 1.0)
    w = (a * norm_up)[..., None]                             # per-pixel 淡入權重
    base = np.asarray(crop, dtype=float)
    out = (1.0 - w) * base + w * heat
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
