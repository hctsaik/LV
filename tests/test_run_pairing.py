"""驗收:28 run_pairing(對齊 pair_runs + 比較輸入整形 compare_inputs)。

設計:3_Architect_Design/28_run_pairing.md(含 2026-07-19 反向閘門修訂:
G1 方向性由本模組+26 的純函式鏈鎖死,GUI E2E 只驗接線)。
item_id 約定:f"{path}#{obj_index}"。
"""
from __future__ import annotations

import pytest


def test_ac1_exact_alignment():
    """# AC1:交集對齊+集合差明講。"""
    from run_pairing import pair_runs
    r = pair_runs({"x": 1, "y": 2, "z": 3}, {"y": 5, "z": 6, "w": 7})
    assert r["ids"] == ["y", "z"]
    assert r["a"] == [2.0, 3.0] and r["b"] == [5.0, 6.0]
    assert r["only_a"] == ["x"] and r["only_b"] == ["w"]
    assert r["dropped"] == [] and r["n_common"] == 2


def test_ac2_nonfinite_dropped_visibly():
    """# AC2:NaN 排除且列名(不靜默)。"""
    from run_pairing import pair_runs
    r = pair_runs({"y": 2, "z": 3}, {"y": float("nan"), "z": 6})
    assert r["ids"] == ["z"] and r["dropped"] == ["y"] and r["n_common"] == 1


def test_ac3_empty_ok():
    """# AC3:空輸入照常回傳,不 raise。"""
    from run_pairing import pair_runs
    r = pair_runs({}, {"a": 1})
    assert r["n_common"] == 0 and r["ids"] == [] and r["only_b"] == ["a"]


def test_ac4_deterministic_sorted():
    """# AC4:字典序決定性;同輸入兩次呼叫完全相等。"""
    from run_pairing import pair_runs
    a = {"b": 1.0, "a": 2.0, "c": 3.0}
    b = {"c": 1.0, "b": 2.0, "a": 3.0}
    r1, r2 = pair_runs(a, b), pair_runs(a, b)
    assert r1["ids"] == ["a", "b", "c"]
    assert r1 == r2


def test_ac5_error_paths():
    """# AC5:不可轉 float 的值 / 非 str key → ValueError。"""
    from run_pairing import pair_runs
    with pytest.raises(ValueError):
        pair_runs({"x": "abc"}, {"x": 1})
    with pytest.raises(ValueError):
        pair_runs({1: 1.0}, {"1": 1.0})


# ── compare_inputs(反向閘門新增契約)──────────────────────────────


def _recs(verdicts, prefix="p"):
    return [{"path": f"{prefix}{i}", "obj_index": 0, "verdict": v}
            for i, v in enumerate(verdicts)]


def test_ac6_compare_inputs_exact():
    """# AC6:3 筆 records、confirmed={0:'bad',2:'good'} → 錯誤 dict 手算 ==。"""
    from run_pairing import compare_inputs
    ref = _recs(["bad", "good", "good"])
    ra = _recs(["bad", "good", "good"])     # A:0 判 bad(對)、2 判 good(對)
    rb = _recs(["good", "good", "bad"])     # B:0 漏(錯)、2 誤報(錯)
    r = compare_inputs(ra, rb, {0: "bad", 2: "good"}, ref)
    assert r["run_a_err"] == {"p0#0": 0, "p2#0": 0}
    assert r["run_b_err"] == {"p0#0": 1, "p2#0": 1}
    assert r["n_labeled"] == 2


def test_ac7_g1_direction_chain():
    """# AC7(G1 方向鏈):10 bad+6 good;A 全對、B 全漏 → 26+28 鏈判 a_better,
    mean_diff=10/16(錯誤率差)。GUI E2E 不重驗方向,方向鎖在這裡。"""
    from paired_compare import paired_compare
    from run_pairing import compare_inputs, pair_runs
    truth = ["bad"] * 10 + ["good"] * 6
    ref = _recs(truth)
    ra = _recs(truth)                       # A 全判對
    rb = _recs(["good"] * 16)               # B 全漏(全判 good)
    confirmed = {i: t for i, t in enumerate(truth)}
    ci = compare_inputs(ra, rb, confirmed, ref)
    pr = pair_runs(ci["run_a_err"], ci["run_b_err"])
    assert pr["n_common"] == 16
    r = paired_compare(pr["a"], pr["b"], higher_is_better=False)
    assert r["verdict"] == "a_better"
    assert r["mean_diff"] == pytest.approx(10 / 16)


def test_ac8_stale_index_skipped():
    """# AC8:confirmed 殘留索引(超出 records_ref)→ 跳過不 raise。"""
    from run_pairing import compare_inputs
    ref = _recs(["good", "good", "good"])
    r = compare_inputs(ref, ref, {99: "bad"}, ref)
    assert r["n_labeled"] == 0
    assert r["run_a_err"] == {} and r["run_b_err"] == {}


def test_compare_inputs_error_paths():
    """# 28 §4:confirmed 值非 good/bad → ValueError;records 缺鍵 → ValueError。"""
    from run_pairing import compare_inputs
    ref = _recs(["good"])
    with pytest.raises(ValueError):
        compare_inputs(ref, ref, {0: "maybe"}, ref)
    with pytest.raises(ValueError):
        compare_inputs([{"path": "p0"}], ref, {0: "bad"}, ref)


# ── PM 推導測試(設計未明列;metamorphic)─────────────────────────────


def test_derived_pair_runs_swap_mirror():
    """推導:pair_runs(b,a) 鏡像:ids/dropped 不變,a/b 與 only_* 對調。"""
    from run_pairing import pair_runs
    A = {"x": 1.0, "y": 2.0, "z": 3.0}
    B = {"y": 5.0, "z": 6.0, "w": 7.0}
    r, m = pair_runs(A, B), pair_runs(B, A)
    assert m["ids"] == r["ids"] and m["dropped"] == r["dropped"]
    assert m["a"] == r["b"] and m["b"] == r["a"]
    assert m["only_a"] == r["only_b"] and m["only_b"] == r["only_a"]


def test_derived_identical_runs_zero_error_diff():
    """推導:A==B(同 records)→ 兩錯誤 dict 完全相等(null 的資料層保證,
    對應 GUI E2E AC-E2 的 deterministic「無差異」)。"""
    from run_pairing import compare_inputs
    ref = _recs(["bad", "good", "bad", "good"])
    confirmed = {0: "bad", 1: "good", 2: "good", 3: "bad"}  # 部分標錯也一樣相等
    r = compare_inputs(ref, ref, confirmed, ref)
    assert r["run_a_err"] == r["run_b_err"]
    assert r["n_labeled"] == 4
