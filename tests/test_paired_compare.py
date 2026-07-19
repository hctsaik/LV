"""驗收:26 paired_compare(成對比較統計引擎:效應量+blocked permutation+三值結論)。

設計:3_Architect_Design/26_paired_compare.md。全合成手算、無 I/O、免模型。
permutation 在單位數 ≤12 時窮舉(2^U),p 可 exact 斷言。
"""
from __future__ import annotations

import numpy as np
import pytest


def test_ac1_effect_detected_exact():
    """# AC1:12 項全 +0.5 → a_better,p=2/4096(僅全正/全負達 |stat|),exact。"""
    from paired_compare import paired_compare
    r = paired_compare([1.5] * 12, [1.0] * 12, higher_is_better=True)
    assert r["verdict"] == "a_better"
    assert r["mean_diff"] == pytest.approx(0.5)
    assert r["win_rate"] == pytest.approx(1.0)
    assert r["n"] == 12 and r["n_units"] == 12
    assert r["exact"] is True and r["blocked"] is False
    assert r["p"] == pytest.approx(2 / 4096)
    assert r["reasons"] == []


def test_ac2_zero_mean_honest():
    """# AC2:d=+1/−1 交錯(mean=0)→ inconclusive+「無差異」。"""
    from paired_compare import paired_compare
    r = paired_compare([1, 2] * 6, [2, 1] * 6, higher_is_better=True)
    assert r["mean_diff"] == pytest.approx(0.0)
    assert r["verdict"] == "inconclusive"
    assert any("無差異" in x for x in r["reasons"])


def test_ac3_batch_confound_naive_vs_blocked():
    """# AC3(G2②核心):效應全在 W1 群 → 天真法誤判 a_better(p=128/4096),
    blocked(2 群)誠實 inconclusive(p=1.0+群組不足)。"""
    from paired_compare import paired_compare
    a = [2.0] * 6 + [1.0] * 6
    b = [1.0] * 12
    naive = paired_compare(a, b, higher_is_better=True)
    assert naive["p"] == pytest.approx(128 / 4096)
    assert naive["verdict"] == "a_better"
    blocked = paired_compare(a, b, higher_is_better=True,
                             groups=["W1"] * 6 + ["W2"] * 6)
    assert blocked["blocked"] is True and blocked["n_units"] == 2
    assert blocked["p"] == pytest.approx(1.0)
    assert blocked["verdict"] == "inconclusive"
    assert any("群組不足" in x for x in blocked["reasons"])


def test_ac4_blocked_can_pass():
    """# AC4:6 群×2 項全 +0.5 → blocked exact p=2/64,a_better。"""
    from paired_compare import paired_compare
    groups = [f"g{i}" for i in range(6) for _ in range(2)]
    r = paired_compare([1.5] * 12, [1.0] * 12, higher_is_better=True, groups=groups)
    assert r["n_units"] == 6 and r["exact"] is True and r["n"] == 12
    assert r["p"] == pytest.approx(2 / 64)
    assert r["verdict"] == "a_better"


def test_ac5_small_n_conservative():
    """# AC5:N=6 全正 → p=2/64 ≤ α 但樣本不足蓋過 → inconclusive。"""
    from paired_compare import paired_compare
    r = paired_compare([2.0] * 6, [1.0] * 6, higher_is_better=True)
    assert r["p"] == pytest.approx(2 / 64)
    assert r["verdict"] == "inconclusive"
    assert any("樣本不足" in x for x in r["reasons"])


def test_ac6_direction_flip():
    """# AC6:lower-is-better 時 a 全低 → a_better,mean_diff=+0.1。"""
    from paired_compare import paired_compare
    r = paired_compare([0.1] * 12, [0.2] * 12, higher_is_better=False)
    assert r["verdict"] == "a_better"
    assert r["mean_diff"] == pytest.approx(0.1)


def test_ac7_error_paths():
    """# AC7:長度不符/NaN/groups 長度不符/alpha 出界 → ValueError。"""
    from paired_compare import paired_compare
    with pytest.raises(ValueError):
        paired_compare([1, 2], [1], higher_is_better=True)
    with pytest.raises(ValueError):
        paired_compare([1, float("nan")], [1, 1], higher_is_better=True)
    with pytest.raises(ValueError):
        paired_compare([1] * 4, [1] * 4, higher_is_better=True, groups=["g"] * 3)
    with pytest.raises(ValueError):
        paired_compare([1] * 12, [2] * 12, higher_is_better=True, alpha=1.5)


def test_ac8_method_line_fragments():
    """# AC8:method line 固定片段(單位=/N=/p=/α=/exploratory;分組另含「群組」)。"""
    from paired_compare import paired_compare
    r = paired_compare([1.5] * 12, [1.0] * 12, higher_is_better=True)
    for frag in ("單位=", "N=", "p=", "α=", "exploratory"):
        assert frag in r["method"], f"method line 缺 {frag}:{r['method']}"
    groups = [f"g{i}" for i in range(6) for _ in range(2)]
    rb = paired_compare([1.5] * 12, [1.0] * 12, higher_is_better=True, groups=groups)
    assert "群組" in rb["method"]


def test_ac9_mc_deterministic():
    """# AC9:N=20(>12 → MC)同 seed 兩次呼叫 p 完全相等,exact=False。"""
    from paired_compare import paired_compare
    rng = np.random.default_rng(7)
    a = rng.normal(0.2, 1.0, 20).tolist()
    b = [0.0] * 20
    r1 = paired_compare(a, b, higher_is_better=True, seed=0)
    r2 = paired_compare(a, b, higher_is_better=True, seed=0)
    assert r1["exact"] is False
    assert r1["p"] == r2["p"]


def test_ac10_null_false_positive_rate():
    """# AC10(統計驗收):200 組 null(N=20, d~N(0,1))→ P(p≤0.05) ≤ 0.075。"""
    from paired_compare import paired_compare
    hits = 0
    for s in range(200):
        rng = np.random.default_rng(s)
        d = rng.normal(0.0, 1.0, 20)
        r = paired_compare(d.tolist(), [0.0] * 20, higher_is_better=True, seed=0)
        hits += int(r["p"] <= 0.05)
    assert hits / 200 <= 0.075, f"null 假陽性率過高:{hits}/200"


# ── PM 推導測試(設計未明列;metamorphic)─────────────────────────────


def test_derived_swap_symmetry():
    """推導:交換 a/b → verdict 鏡像、mean_diff 變號、p 不變(對稱性不變量)。"""
    from paired_compare import paired_compare
    a, b = [1.5] * 12, [1.0] * 12
    r_ab = paired_compare(a, b, higher_is_better=True)
    r_ba = paired_compare(b, a, higher_is_better=True)
    assert r_ab["verdict"] == "a_better" and r_ba["verdict"] == "b_better"
    assert r_ba["mean_diff"] == pytest.approx(-r_ab["mean_diff"])
    assert r_ba["p"] == pytest.approx(r_ab["p"])


def test_derived_scale_invariance():
    """推導:d 整體正尺度縮放 → p 與 verdict 不變(sign-flip 統計量線性)。"""
    from paired_compare import paired_compare
    rng = np.random.default_rng(3)
    base = rng.normal(0.5, 1.0, 12)
    r1 = paired_compare(base.tolist(), [0.0] * 12, higher_is_better=True)
    r2 = paired_compare((base * 7.0).tolist(), [0.0] * 12, higher_is_better=True)
    assert r1["p"] == pytest.approx(r2["p"])
    assert r1["verdict"] == r2["verdict"]


def test_derived_group_relabel_invariance():
    """推導:群組改名(標籤集合置換)不影響結果(只看分割,不看名字)。"""
    from paired_compare import paired_compare
    a, b = [1.5] * 12, [1.0] * 12
    g1 = ["x0", "x0", "x1", "x1", "x2", "x2", "x3", "x3", "x4", "x4", "x5", "x5"]
    g2 = ["W" + g for g in g1]
    r1 = paired_compare(a, b, higher_is_better=True, groups=g1)
    r2 = paired_compare(a, b, higher_is_better=True, groups=g2)
    assert r1["p"] == pytest.approx(r2["p"])
    assert r1["verdict"] == r2["verdict"] and r1["n_units"] == r2["n_units"]
