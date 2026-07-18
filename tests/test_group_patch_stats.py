"""驗收:20 group_patch_stats(Good/Bad 群組 patch 差異統計 + null 校準)。

設計:3_Architect_Design/20_group_patch_stats.md。全部合成陣列、固定 seed、免模型。
"""
from __future__ import annotations

import numpy as np
import pytest


def _synth(Ng=40, Nb=40, P=25, D=16, shift_patch=None, shift_frac=1.0,
           shift_scale=1.2, noise=0.15, seed=7):
    """兩群同基底 patch 特徵;shift_patch 給定時 Bad 前 shift_frac 比例在該位置加固定位移。

    基底與位移向量只依 seed(與 noise 無關)→ AC5 兩組只差 noise 時基底相同、比較公平。
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(P, D))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    shift_vec = rng.normal(size=(D,))
    shift_vec /= np.linalg.norm(shift_vec)

    def _draw(n, shifted_count):
        x = base[None, :, :] + rng.normal(scale=noise, size=(n, P, D))
        if shift_patch is not None and shifted_count:
            x[:shifted_count, shift_patch, :] += shift_scale * shift_vec
        x /= np.clip(np.linalg.norm(x, axis=2, keepdims=True), 1e-12, None)
        return x.astype(np.float32)

    good = _draw(Ng, 0)
    n_shift = int(round(shift_frac * Nb)) if shift_patch is not None else 0
    bad = _draw(Nb, n_shift)
    return good, bad


def test_ac1_planted_top1_and_verdict():
    """# AC1:整群植入位移 → 該 patch score/z 皆 Top-1、verdict True、p<0.05。"""
    from group_patch_stats import group_diff_stats
    good, bad = _synth(shift_patch=7)
    r = group_diff_stats(good, bad, seed=0)
    assert int(np.argmax(r["score"])) == 7
    assert int(np.argmax(r["z"])) == 7
    assert r["verdict"] is True
    assert r["p_global"] < 0.05


def test_ac2_null_honest():
    """# AC2:同分布兩群(10 組不同 seed)→ verdict True 至多 2 組(α=0.05 校準)。"""
    from group_patch_stats import group_diff_stats
    hits = 0
    for s in range(100, 110):
        good, bad = _synth(Ng=24, Nb=24, seed=s)
        r = group_diff_stats(good, bad, n_perm=100, seed=0)
        hits += int(r["verdict"])
    assert hits <= 2, f"null 資料 10 組中 {hits} 組誤判有差異(校準失敗)"


def test_ac3_single_outlier_not_stable():
    """# AC3:40 張 Bad 只 1 張有位移 → 偶然雜訊,verdict False 且 stability 低。"""
    from group_patch_stats import group_diff_stats
    good, bad = _synth(shift_patch=7, shift_frac=1 / 40)
    r = group_diff_stats(good, bad, seed=0)
    assert r["verdict"] is False
    assert r["stability"][7] < 0.2


def test_ac4_stability_quantifies_fraction():
    """# AC4:80% Bad 有位移 → stability ≈ 0.8(±0.15)、仍 Top-1、verdict True。"""
    from group_patch_stats import group_diff_stats
    good, bad = _synth(shift_patch=7, shift_frac=0.8)
    r = group_diff_stats(good, bad, seed=0)
    assert abs(float(r["stability"][7]) - 0.8) <= 0.15
    assert int(np.argmax(r["z"])) == 7
    assert r["verdict"] is True


def test_ac5_within_spread_in_denominator():
    """# AC5:同位移、群內雜訊小 vs 大 → 分離度分數明顯較高(>1.5 倍)。"""
    from group_patch_stats import group_diff_stats
    g_lo, b_lo = _synth(shift_patch=7, noise=0.05)
    g_hi, b_hi = _synth(shift_patch=7, noise=0.45)
    r_lo = group_diff_stats(g_lo, b_lo, seed=0)
    r_hi = group_diff_stats(g_hi, b_hi, seed=0)
    assert float(r_lo["score"][7]) > 1.5 * float(r_hi["score"][7])


def test_ac6_validation_errors():
    """# AC6:Ng<2 / P 不一致 / D 不一致 / NaN / n_perm<20 → ValueError。"""
    from group_patch_stats import group_diff_stats
    good, bad = _synth(Ng=4, Nb=4, P=4, D=8)
    with pytest.raises(ValueError):
        group_diff_stats(good[:1], bad)
    with pytest.raises(ValueError):
        group_diff_stats(good, bad[:, :3, :])
    with pytest.raises(ValueError):
        group_diff_stats(good, bad[:, :, :4])
    nan_bad = bad.copy()
    nan_bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        group_diff_stats(good, nan_bad)
    with pytest.raises(ValueError):
        group_diff_stats(good, bad, n_perm=5)


def test_ac7_deterministic():
    """# AC7:同參數同 seed 兩次 → 輸出完全相等。"""
    from group_patch_stats import group_diff_stats
    good, bad = _synth(shift_patch=3, Ng=16, Nb=16)
    r1 = group_diff_stats(good, bad, n_perm=50, seed=42)
    r2 = group_diff_stats(good, bad, n_perm=50, seed=42)
    assert np.array_equal(r1["score"], r2["score"])
    assert np.array_equal(r1["z"], r2["z"])
    assert np.array_equal(r1["stability"], r2["stability"])
    assert r1["p_global"] == r2["p_global"]


def test_derived_swap_symmetry():
    """# 推導:score 對群交換對稱(分子分母皆對稱);Ng==Nb 時 verdict 一致。

    施壓「分數把 good/bad 角色寫死」類 bug(例如只用 bad 對 good 的單向距離當分數)。
    """
    from group_patch_stats import group_diff_stats
    good, bad = _synth(shift_patch=7, Ng=20, Nb=20)
    r_ab = group_diff_stats(good, bad, n_perm=80, seed=3)
    r_ba = group_diff_stats(bad, good, n_perm=80, seed=3)
    assert np.allclose(r_ab["score"], r_ba["score"], atol=1e-6)
    assert r_ab["verdict"] == r_ba["verdict"]


def test_ac8_output_shape_contract():
    """# AC8:輸出鍵與形狀契約(設計 §2 表)。"""
    from group_patch_stats import group_diff_stats
    Ng, Nb, P = 8, 6, 9
    good, bad = _synth(Ng=Ng, Nb=Nb, P=P, D=8)
    r = group_diff_stats(good, bad, n_perm=30, seed=1)
    assert r["score"].shape == (P,)
    assert r["z"].shape == (P,)
    assert r["stability"].shape == (P,)
    assert r["dist_good"].shape == (Ng, P)
    assert r["dist_bad"].shape == (Nb, P)
    assert isinstance(r["p_global"], float)
    assert isinstance(r["verdict"], bool)
    assert isinstance(r["threshold_z"], float)
    assert r["null_max_z"].shape == (30,)
    assert np.all((r["stability"] >= 0.0) & (r["stability"] <= 1.0))
