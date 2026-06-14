"""Unit tests for the 桶① signal-strength gate (scripts/signal_strength.py).

These pin the coarse three-level verdict: a real defect on a quiet
background reads OBVIOUS, an ROI that is merely more background reads
NONE (the 桶① case adding data can never fix), and ambiguity reads
SUSPECT. No ROI / unreadable input must read UNKNOWN, never a guess.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from signal_strength import (
    SIGNAL_NONE,
    SIGNAL_OBVIOUS,
    SIGNAL_SUSPECT,
    SIGNAL_UNKNOWN,
    calibrate_thresholds,
    classify_signal,
    effective_thresholds,
    load_calibration,
    roi_background_metrics,
    save_calibration,
    signal_level_for_image,
)


def test_obvious_bright_roi_on_quiet_background():
    rng = np.random.default_rng(0)
    g = 0.5 + rng.normal(0, 0.01, (100, 100))
    g[40:60, 40:60] = 0.95  # bright, ROI-filling defect
    m = roi_background_metrics(g, (40, 40, 60, 60))
    assert m and m["snr"] > 10 and m["frac_signal"] > 0.9
    assert classify_signal(m) == SIGNAL_OBVIOUS


def test_none_when_roi_is_just_more_background():
    # the ROI is statistically identical to its surroundings → 桶①:
    # there is nothing here that data could ever teach the model.
    rng = np.random.default_rng(1)
    g = 0.5 + rng.normal(0, 0.05, (120, 120))
    m = roi_background_metrics(g, (50, 50, 70, 70))
    assert classify_signal(m) == SIGNAL_NONE


def test_none_when_saturated():
    g = np.ones((80, 80))  # whole frame clipped white → no contrast possible
    m = roi_background_metrics(g, (30, 30, 50, 50))
    assert classify_signal(m) == SIGNAL_NONE


def test_suspect_small_weak_defect():
    rng = np.random.default_rng(2)
    g = 0.5 + rng.normal(0, 0.01, (100, 100))
    g[48:52, 48:52] += 0.05  # tiny, low-contrast bump inside a larger ROI
    m = roi_background_metrics(g, (40, 40, 60, 60))
    assert classify_signal(m) == SIGNAL_SUSPECT


def test_unknown_inputs():
    assert classify_signal({}) == SIGNAL_UNKNOWN
    g = np.full((50, 50), 0.5)
    assert roi_background_metrics(g, (10, 10, 10, 10)) == {}        # zero area
    assert roi_background_metrics(g, (100, 100, 110, 110)) == {}    # off-frame


def test_bbox_clipped_to_frame():
    rng = np.random.default_rng(3)
    g = 0.5 + rng.normal(0, 0.01, (60, 60))
    g[0:15, 0:15] = 0.95
    # bbox spills past the top-left corner; it must clip, not crash
    m = roi_background_metrics(g, (-10, -10, 15, 15))
    assert m and classify_signal(m) == SIGNAL_OBVIOUS


def test_signal_level_for_image_end_to_end(tmp_path):
    rng = np.random.default_rng(4)
    arr = 0.5 + rng.normal(0, 0.01, (100, 100))
    arr[40:60, 40:60] = 0.95
    img = tmp_path / "defect.png"
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8")).save(img)
    assert signal_level_for_image(img, bbox=(40, 40, 60, 60)) == SIGNAL_OBVIOUS


def test_signal_level_unknown_without_bbox_or_unreadable(tmp_path):
    img = tmp_path / "x.png"
    Image.fromarray(np.full((40, 40), 128, dtype="uint8")).save(img)
    assert signal_level_for_image(img, bbox=None) == SIGNAL_UNKNOWN
    assert signal_level_for_image(tmp_path / "nope.png", bbox=(0, 0, 10, 10)) \
        == SIGNAL_UNKNOWN


# ── self-calibration from a labelled defect distribution ────────────────

def test_calibrate_places_none_below_defects_and_caps_obvious_at_rose():
    rng = np.random.default_rng(0)
    snr = rng.normal(6.0, 1.0, 500)  # confirmed defects: genuinely detectable
    cal = calibrate_thresholds(snr)
    assert cal["calibrated"]
    assert cal["snr_none"] < cal["p50"]        # 確無 line sits below the bulk
    assert cal["snr_none"] < cal["snr_obvious"]  # band stays ordered
    assert cal["snr_obvious"] <= 5.0           # Rose criterion is the ceiling
    # almost no real defect should be mislabelled 確無 under the suggestion
    mislabelled = np.mean(snr < cal["snr_none"])
    assert mislabelled <= 0.06


def test_calibrate_too_few_defects_is_uncalibrated():
    cal = calibrate_thresholds([8.0, 7.0, 9.0])
    assert not cal["calibrated"]
    assert cal["snr_none"] == 1.5 and cal["snr_obvious"] == 4.0


# ── calibration config: persist + auto-apply ───────────────────────────

def test_calibration_config_roundtrip_and_effective(tmp_path):
    cfg_file = tmp_path / "signal_gate.json"
    # no file → module defaults, flagged uncalibrated
    eff0 = effective_thresholds(cfg_path=cfg_file)
    assert eff0["source"] == "default" and not eff0["calibrated"]
    assert eff0["snr_none"] == 1.5 and eff0["snr_obvious"] == 4.0
    # write a calibration → effective_thresholds reflects it
    rng = np.random.default_rng(0)
    cal = calibrate_thresholds(rng.normal(6.0, 1.0, 500))
    save_calibration({**cal, "dataset": "demo"}, cfg_file)
    eff = effective_thresholds(cfg_path=cfg_file)
    assert eff["source"] == "config:global" and eff["calibrated"]
    assert eff["snr_none"] == cal["snr_none"]
    assert eff["snr_obvious"] == cal["snr_obvious"]
    assert eff["dataset"] == "demo"
    # malformed / incomplete configs are ignored, never crash
    assert load_calibration(tmp_path / "nope.json") is None
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert load_calibration(tmp_path / "bad.json") is None
    (tmp_path / "inc.json").write_text('{"snr_none": 1.0}', encoding="utf-8")
    assert load_calibration(tmp_path / "inc.json") is None


def test_per_type_thresholds_fall_back_to_global_then_default(tmp_path):
    cfg_file = tmp_path / "signal_gate.json"
    save_calibration({
        "snr_none": 1.0, "snr_obvious": 3.0, "calibrated": True, "n": 400,
        "type_from": "parent", "dataset": "demo",
        "per_type": {"metalA": {"snr_none": 0.4, "snr_obvious": 1.2, "n": 120}},
    }, cfg_file)
    # known type → its own thresholds
    a = effective_thresholds(image_type="metalA", cfg_path=cfg_file)
    assert a["source"] == "config:type" and a["snr_none"] == 0.4 and a["n"] == 120
    # unknown type → config global
    b = effective_thresholds(image_type="glassZ", cfg_path=cfg_file)
    assert b["source"] == "config:global" and b["snr_none"] == 1.0
    # no type asked → config global
    c = effective_thresholds(cfg_path=cfg_file)
    assert c["source"] == "config:global" and c["snr_obvious"] == 3.0


def test_signal_level_honours_config_thresholds(tmp_path):
    rng = np.random.default_rng(5)
    g = 0.5 + rng.normal(0, 0.03, (120, 120))
    g[50:70, 50:70] += 0.02  # very faint, ROI-filling → low SNR
    img = tmp_path / "faint.png"
    Image.fromarray((np.clip(g, 0, 1) * 255).astype("uint8")).save(img)
    bbox = (50, 50, 70, 70)
    cfg_file = tmp_path / "signal_gate.json"
    # default thresholds → faint ROI is NOT obvious
    assert signal_level_for_image(img, bbox, cfg_path=cfg_file) != SIGNAL_OBVIOUS
    # a permissive calibrated config lowers the bar → same ROI now obvious
    save_calibration({"snr_none": 0.1, "snr_obvious": 0.5, "calibrated": True},
                     cfg_file)
    assert signal_level_for_image(img, bbox, cfg_path=cfg_file) == SIGNAL_OBVIOUS
