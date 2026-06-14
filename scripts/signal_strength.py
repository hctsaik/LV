"""Defect signal-strength gate — the missing 桶① axis.

The H1–H5 root-cause pipeline (``interaction.diagnose_root_cause``) scores
three axes — human consistency (S1), data coverage (S2), model
uncertainty (S3) — but NONE of them asks whether the reported defect's
signal is physically present in the pixels at all. So a defect the
imaging chain never captured (桶①「訊號未進資料」) is indistinguishable
from a coverage gap (H1) and gets the exact wrong prescription: "補這格
的資料——補了大概率有效". No amount of data can recover information that
was lost at the sensor.

This module adds a deliberately MINIMAL-ASSUMPTION signal-strength
estimate (requirements doc R0a): does the reported defect ROI carry
evidence measurably above its LOCAL background? It answers only a coarse
question —

    SIGNAL_OBVIOUS  明顯  strong evidence above background  → not 桶①
    SIGNAL_SUSPECT  疑似  weak / borderline evidence        → inconclusive
    SIGNAL_NONE     確無  no evidence above background       → 桶① candidate
    SIGNAL_UNKNOWN  未知  cannot judge (no ROI / bad input)  → do not guess

— and is meant as a *screen* in front of ``diagnose_root_cause``, not a
physics instrument. It is honest by construction: with no defect ROI it
returns UNKNOWN rather than guessing, and the thresholds below are
STARTING POINTS that must be calibrated in-place on real escapes (R0a's
own caveat: the definition of "evidence above background" is the first
thing that drifts).

The two physical quantities it leans on:
  * **SNR** — mean ROI deviation over the local background NOISE (not a
    global background: a textured part has no meaningful global mean).
  * **frac_signal** — fraction of ROI pixels exceeding 3× the background
    noise. This is a robust small-defect catcher: a coherent defect lights
    up many ROI pixels, whereas background noise only crosses 3σ ~0.3% of
    the time. It is what lets a small defect inside a large ROI still
    register even when its mean SNR is diluted.

Framework-free: no streamlit imports, every function unit-testable.

NB: the module is NOT named ``signal.py`` on purpose — that would shadow
the Python standard-library ``signal`` module, which sits on the same
``sys.path`` as this scripts/ folder.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

SIGNAL_OBVIOUS = "明顯"
SIGNAL_SUSPECT = "疑似"
SIGNAL_NONE = "確無"
SIGNAL_UNKNOWN = "未知"

# quantization noise floor (8-bit): a flat region's "noise" is never < 1 LSB,
# so SNR can't blow up to infinity on a synthetically flat background.
_MIN_NOISE = 1.0 / 255.0

# Default gate thresholds — textbook starting points, overridden by a
# calibration config when one exists (calibrate_signal_gate.py --write).
_DEFAULT_SNR_NONE = 1.5
_DEFAULT_SNR_OBVIOUS = 4.0
_CONFIG_NAME = "signal_gate.json"


def roi_background_metrics(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    ring_frac: float = 0.5,
    min_noise: float = _MIN_NOISE,
    min_bg_px: int = 20,
) -> dict:
    """Local contrast / SNR of an ROI against a surrounding background ring.

    ``gray`` is a 2-D float array in [0,1]. ``bbox`` = (x0, y0, x1, y1) in
    PIXEL coords (the reported defect location). The background is a ring
    around the bbox — the bbox dilated by ``ring_frac`` of its size, minus
    the bbox itself — so the comparison is LOCAL. Returns ``{}`` for a
    degenerate ROI (zero area / out of frame / too little surrounding
    background), which the caller maps to UNKNOWN.

    Keys: snr, frac_signal, weber, peak_weber, mean_roi, mean_bg, noise,
    roi_px, bg_px.
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2:
        return {}
    h, w = g.shape
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    x0, x1 = sorted((max(0, min(x0, w)), max(0, min(x1, w))))
    y0, y1 = sorted((max(0, min(y0, h)), max(0, min(y1, h))))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return {}
    bw, bh = x1 - x0, y1 - y0
    dx, dy = max(1, round(bw * ring_frac)), max(1, round(bh * ring_frac))
    ox0, oy0 = max(0, x0 - dx), max(0, y0 - dy)
    ox1, oy1 = min(w, x1 + dx), min(h, y1 + dy)

    roi = g[y0:y1, x0:x1].reshape(-1)
    outer = g[oy0:oy1, ox0:ox1]
    mask = np.ones(outer.shape, dtype=bool)
    mask[y0 - oy0:y1 - oy0, x0 - ox0:x1 - ox0] = False  # punch out the ROI
    bg = outer[mask]
    if roi.size == 0 or bg.size < min_bg_px:
        return {}

    mean_roi = float(roi.mean())
    mean_bg = float(bg.mean())
    noise = max(float(bg.std()), min_noise)
    eps = 1e-9
    dev = np.abs(roi - mean_bg)
    mean_dev = abs(mean_roi - mean_bg)
    return {
        "snr": mean_dev / noise,
        "frac_signal": float(np.mean(dev > 3.0 * noise)),
        "weber": mean_dev / (mean_bg + eps),
        "peak_weber": float(np.percentile(dev, 99)) / (mean_bg + eps),
        "mean_roi": mean_roi,
        "mean_bg": mean_bg,
        "noise": noise,
        "roi_px": int(roi.size),
        "bg_px": int(bg.size),
    }


def classify_signal(
    metrics: dict,
    *,
    snr_obvious: float = _DEFAULT_SNR_OBVIOUS,
    snr_none: float = _DEFAULT_SNR_NONE,
    frac_obvious: float = 0.5,
    frac_none: float = 0.02,
) -> str:
    """Coarse 三檔 verdict from ``roi_background_metrics`` output.

    Empty metrics → UNKNOWN (never guess). OBVIOUS when the mean SNR is
    strong OR a large fraction of the ROI is lit; NONE only when BOTH the
    mean SNR is buried in noise AND almost no ROI pixel crosses 3σ (so a
    small-but-real defect is not silently called 桶①). Everything between
    is SUSPECT. Thresholds are starting points — calibrate on real escapes.
    """
    if not metrics:
        return SIGNAL_UNKNOWN
    snr = metrics["snr"]
    frac = metrics["frac_signal"]
    if snr >= snr_obvious or frac >= frac_obvious:
        return SIGNAL_OBVIOUS
    if snr <= snr_none and frac <= frac_none:
        return SIGNAL_NONE
    return SIGNAL_SUSPECT


def calibrate_thresholds(
    snr_values,
    *,
    none_pct: float = 5.0,
    obvious_pct: float = 60.0,
    rose_snr: float = 5.0,
) -> dict:
    """Suggest gate thresholds from the SNR distribution of KNOWN defects.

    Self-calibration without ground-truth 桶① labels, under one stated
    assumption: a LABELLED defect box is something an annotator could see,
    so it is (mostly) genuinely detectable — 桶②/③, not 桶①. The labelled
    set's SNR distribution therefore defines what "detectable" looks like,
    and the 確無 line must sit BELOW it:

      snr_obvious = min(rose_snr, p{obvious_pct}) — the Rose criterion
                    (SNR≥5 = reliably detectable) is a HARD physics ceiling,
                    pulled lower if the data's bulk sits below it (a genuinely
                    low-contrast modality is not held to an impossible bar).
      snr_none    = p{none_pct} of defect SNRs, floored at 0.5 and kept at
                    least 0.5 BELOW snr_obvious — anything weaker than the
                    faintest few percent of confirmed defects is a 桶①
                    candidate (its signal is below what real defects show).

    Invariants (hold for any input): 0.5 ≤ snr_none < snr_obvious ≤ rose_snr.

    Returns {snr_none, snr_obvious, n, p5, p50, p95, calibrated}. Empty input
    → the module defaults (1.5 / 4.0), flagged ``calibrated=False`` — never
    silently invent thresholds from nothing.
    """
    snr = np.asarray([s for s in snr_values if np.isfinite(s)], dtype=float)
    if snr.size < 10:  # too few defects to trust a distribution
        return {"snr_none": 1.5, "snr_obvious": 4.0, "n": int(snr.size),
                "p5": None, "p50": None, "p95": None, "calibrated": False}
    p5, p50, p95 = (float(np.percentile(snr, q)) for q in (5, 50, 95))
    # Rose is the hard ceiling; floor of 1.0 stops it collapsing on degenerate
    # (e.g. big-object) data where local contrast is near zero.
    snr_obvious = min(rose_snr, max(float(np.percentile(snr, obvious_pct)), 1.0))
    # 確無 line below the defect bulk, then clamped strictly under snr_obvious.
    snr_none = min(max(0.5, float(np.percentile(snr, none_pct))), snr_obvious - 0.5)
    return {"snr_none": round(snr_none, 3), "snr_obvious": round(snr_obvious, 3),
            "n": int(snr.size), "p5": round(p5, 3), "p50": round(p50, 3),
            "p95": round(p95, 3), "calibrated": True}


# ── calibration config (self-applying thresholds) ──────────────────────

def config_path(config_path: Path | str | None = None) -> Path:
    """Resolve the gate's calibration-config path: explicit arg →
    ``LV_SIGNAL_GATE_CONFIG`` env var → ``<repo>/signal_gate.json``."""
    if config_path is not None:
        return Path(config_path)
    env = os.environ.get("LV_SIGNAL_GATE_CONFIG")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / _CONFIG_NAME


def load_calibration(cfg_path: Path | str | None = None) -> dict | None:
    """Read a calibration config → dict, or None if absent/invalid. A valid
    config must carry numeric ``snr_none`` and ``snr_obvious``."""
    p = config_path(cfg_path)
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    try:
        float(cfg["snr_none"]); float(cfg["snr_obvious"])
    except (KeyError, TypeError, ValueError):
        return None
    return cfg


def save_calibration(cfg: dict, cfg_path: Path | str | None = None) -> Path:
    """Write a calibration config atomically; returns the path written."""
    p = config_path(cfg_path)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def effective_thresholds(
    *,
    image_type: str | None = None,
    cfg_path: Path | str | None = None,
) -> dict:
    """The thresholds actually in force for ``image_type`` (None = global):
    a per-type calibration if one exists for that type, else the config's
    global values, else the module defaults. Different image types (optics /
    material) have very different signal floors, so a single ruler under-
    serves a multi-modality line. Always returns {snr_none, snr_obvious,
    source, calibrated, n, dataset, image_type}."""
    cfg = load_calibration(cfg_path)
    if cfg:
        per = cfg.get("per_type") or {}
        if image_type is not None and image_type in per:
            t = per[image_type]
            return {"snr_none": float(t["snr_none"]),
                    "snr_obvious": float(t["snr_obvious"]),
                    "source": "config:type", "calibrated": True,
                    "n": t.get("n"), "dataset": cfg.get("dataset"),
                    "image_type": image_type}
        return {"snr_none": float(cfg["snr_none"]),
                "snr_obvious": float(cfg["snr_obvious"]),
                "source": "config:global",
                "calibrated": bool(cfg.get("calibrated", True)),
                "n": cfg.get("n"), "dataset": cfg.get("dataset"),
                "image_type": None}
    return {"snr_none": _DEFAULT_SNR_NONE, "snr_obvious": _DEFAULT_SNR_OBVIOUS,
            "source": "default", "calibrated": False, "n": None,
            "dataset": None, "image_type": None}


def signal_level_for_image(
    image_path: Path | str,
    bbox: tuple[int, int, int, int] | None = None,
    *,
    snr_none: float | None = None,
    snr_obvious: float | None = None,
    image_type: str | None = None,
    cfg_path: Path | str | None = None,
    **classify_kwargs,
) -> str:
    """Convenience: load ``image_path`` grayscale, measure the ROI, classify.

    ``bbox`` None (no reported defect location), an unreadable image, or a
    degenerate ROI all yield SIGNAL_UNKNOWN — the gate must not invent a
    桶① verdict it cannot support. SNR thresholds default to the calibrated
    config (``effective_thresholds`` for ``image_type``) when not passed
    explicitly, so a calibration written by calibrate_signal_gate.py applies
    automatically — per image type when a per-type calibration exists.
    """
    if bbox is None:
        return SIGNAL_UNKNOWN
    if snr_none is None or snr_obvious is None:
        thr = effective_thresholds(image_type=image_type, cfg_path=cfg_path)
        snr_none = thr["snr_none"] if snr_none is None else snr_none
        snr_obvious = thr["snr_obvious"] if snr_obvious is None else snr_obvious
    try:
        with Image.open(image_path) as im:
            g = np.asarray(im.convert("L"), dtype=np.float64) / 255.0
    except (OSError, ValueError):
        return SIGNAL_UNKNOWN
    return classify_signal(roi_background_metrics(g, bbox),
                           snr_none=snr_none, snr_obvious=snr_obvious,
                           **classify_kwargs)
