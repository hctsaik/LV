"""契約測試:tools/eval_scenarios.py 的評估「邏輯」(非 DINOv2 管線)。

守護情境評估的數學契約,讓 S1(rare-as-anomaly 可分性)、S6(escape/overkill 單調性)、
bootstrap CI 不靠一次性快照、而是被測試鎖定(回應多 agent 評分對「評估邏輯未被 gate」的點名)。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from eval_scenarios import _ci95, escape_overkill_curve  # noqa: E402


def test_escape_overkill_monotonic():  # 契約:percentile↑ → escape 單調不減、overkill 單調不增
    rng = np.random.default_rng(0)
    good = rng.uniform(0.0, 0.1, 200)        # 良品距離緊密在 0
    bad = rng.uniform(0.3, 1.0, 30)          # 缺陷距離散在遠端
    dist = np.concatenate([good, bad])
    bad_mask = np.array([False] * 200 + [True] * 30)
    curve = escape_overkill_curve(dist, bad_mask, ~bad_mask, good)
    escapes = [e for _, _, e, _ in curve]
    overkills = [o for _, _, _, o in curve]
    assert escapes == sorted(escapes)                      # escape 單調不減
    assert overkills == sorted(overkills, reverse=True)    # overkill 單調不增


def test_separable_low_escape_operating_point():  # recall-first:夠分得開時存在低 escape 操作點
    good = np.zeros(200)
    bad = np.full(30, 0.9)
    dist = np.concatenate([good, bad])
    bad_mask = np.array([False] * 200 + [True] * 30)
    curve = escape_overkill_curve(dist, bad_mask, ~bad_mask, good, percentiles=(70, 99))
    assert curve[0][2] == pytest.approx(0.0)               # 低門檻 → escape≈0(全抓到)


def test_ci95_bounds():
    assert _ci95([0.5] * 100) == [0.5, 0.5]               # 退化 → 上下界相等
    lo, hi = _ci95(list(np.linspace(0.0, 1.0, 1000)))
    assert lo < 0.1 and hi > 0.9                          # 均勻 → CI 近 [0,1]
