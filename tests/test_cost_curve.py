"""驗收:27 cost_curve(閾值掃描 FN/FP/工作量/預期代價曲線 + 最低點)。

設計:3_Architect_Design/27_cost_curve.md。全合成手算(計數類 → exact ==)。
固定小例:scores=[0.1,0.2,0.3,0.4,0.5,0.9]、labels=[0,0,0,1,0,1](P=2、N=4);
自動閾值=[-0.9,0.15,0.25,0.35,0.45,0.7,1.9];判定 score >= t → flag。
"""
from __future__ import annotations

import pytest

SCORES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.9]
LABELS = [0, 0, 0, 1, 0, 1]


def test_ac1_exact_full_arrays():
    """# AC1(G3):cost_miss=5,cost_fa=1 → 全陣列手算 ==。"""
    from cost_curve import cost_curve
    r = cost_curve(SCORES, LABELS, cost_miss=5, cost_fa=1)
    assert r["thresholds"] == pytest.approx([-0.9, 0.15, 0.25, 0.35, 0.45, 0.7, 1.9])
    assert r["fn"] == [0, 0, 0, 0, 1, 1, 2]
    assert r["fp"] == [4, 3, 2, 1, 1, 0, 0]
    assert r["flagged"] == [6, 5, 4, 3, 2, 1, 0]
    assert r["cost"] == pytest.approx([4, 3, 2, 1, 6, 5, 10])
    assert r["best_idx"] == 3
    assert r["n"] == 6 and r["n_pos"] == 2 and r["n_neg"] == 4


def test_ac2_ratio_moves_workpoint():
    """# AC2(G3):cost_miss=1,cost_fa=10 → 最低點從 idx3(0.35)移到 idx5(0.7)。"""
    from cost_curve import cost_curve
    r = cost_curve(SCORES, LABELS, cost_miss=1, cost_fa=10)
    assert r["cost"] == pytest.approx([40, 30, 20, 10, 11, 1, 2])
    assert r["best_idx"] == 5


def test_ac3_current_not_snapped():
    """# AC3:current_threshold=0.5 直接算(不吸附網格);0.5>=0.5 → flag。"""
    from cost_curve import cost_curve
    r = cost_curve(SCORES, LABELS, cost_miss=5, cost_fa=1, current_threshold=0.5)
    assert r["current"] == {"threshold": 0.5, "fn": 1, "fp": 1,
                            "flagged": 2, "cost": 6.0}


def test_ac3b_current_none():
    """# AC3 邊界:current_threshold 未給 → current is None。"""
    from cost_curve import cost_curve
    assert cost_curve(SCORES, LABELS, cost_miss=5, cost_fa=1)["current"] is None


def test_ac4_tie_takes_loosest():
    """# AC4:cost 平手 → 取最小 index(=最鬆);K=1 全同分數 → 兩點曲線。"""
    from cost_curve import cost_curve
    r = cost_curve([0.1, 0.9], [0, 1], cost_miss=1, cost_fa=1)
    assert r["thresholds"] == pytest.approx([-0.9, 0.5, 1.9])
    assert r["cost"] == pytest.approx([1, 0, 1])
    assert r["best_idx"] == 1
    r2 = cost_curve([0.5, 0.5], [0, 1], cost_miss=1, cost_fa=1)
    assert r2["thresholds"] == pytest.approx([-0.5, 1.5])
    assert r2["cost"] == pytest.approx([1, 1])
    assert r2["best_idx"] == 0


def test_ac5_error_paths():
    """# AC5:單類標籤(訊息含指引)/cost<=0/label 出集合/長度不符/空輸入 → ValueError。"""
    from cost_curve import cost_curve
    with pytest.raises(ValueError, match="需同時有已確認的瑕疵與正常"):
        cost_curve([0.1, 0.2], [1, 1], cost_miss=1, cost_fa=1)
    with pytest.raises(ValueError, match="需同時有已確認的瑕疵與正常"):
        cost_curve([0.1, 0.2], [0, 0], cost_miss=1, cost_fa=1)
    with pytest.raises(ValueError):
        cost_curve(SCORES, LABELS, cost_miss=0, cost_fa=1)
    with pytest.raises(ValueError):
        cost_curve([0.1, 0.2], [0, 2], cost_miss=1, cost_fa=1)
    with pytest.raises(ValueError):
        cost_curve([0.1], [0, 1], cost_miss=1, cost_fa=1)
    with pytest.raises(ValueError):
        cost_curve([], [], cost_miss=1, cost_fa=1)
    with pytest.raises(ValueError):
        cost_curve(SCORES, LABELS, cost_miss=1, cost_fa=1, thresholds=[])


def test_ac6_method_line():
    """# AC6:method line 含判定規則、成本比、「僅試算」。"""
    from cost_curve import cost_curve
    m = cost_curve(SCORES, LABELS, cost_miss=5, cost_fa=1)["method"]
    assert "score >= t" in m
    assert "漏檢:誤報" in m
    assert "僅試算" in m


# ── PM 推導測試(設計未明列;不變量/metamorphic)──────────────────────


def test_derived_cost_identity_and_flag_identity():
    """推導:逐點恆等式 cost=miss*fn+fa*fp、flagged=(n_pos-fn)+fp。"""
    from cost_curve import cost_curve
    r = cost_curve(SCORES, LABELS, cost_miss=5, cost_fa=1)
    for i in range(len(r["thresholds"])):
        assert r["cost"][i] == pytest.approx(5 * r["fn"][i] + 1 * r["fp"][i])
        assert r["flagged"][i] == (r["n_pos"] - r["fn"][i]) + r["fp"][i]


def test_derived_cost_scale_invariance():
    """推導:兩種成本同乘正常數 → best_idx 不變、cost 等比放大。"""
    from cost_curve import cost_curve
    r1 = cost_curve(SCORES, LABELS, cost_miss=5, cost_fa=1)
    r2 = cost_curve(SCORES, LABELS, cost_miss=15, cost_fa=3)
    assert r2["best_idx"] == r1["best_idx"]
    assert r2["cost"] == pytest.approx([3 * c for c in r1["cost"]])


def test_derived_score_shift_invariance():
    """推導:所有分數平移 +c → 閾值同平移,fn/fp/flagged/best_idx 不變。"""
    from cost_curve import cost_curve
    c = 10.0
    r1 = cost_curve(SCORES, LABELS, cost_miss=5, cost_fa=1)
    r2 = cost_curve([s + c for s in SCORES], LABELS, cost_miss=5, cost_fa=1)
    assert r2["thresholds"] == pytest.approx([t + c for t in r1["thresholds"]])
    assert r2["fn"] == r1["fn"] and r2["fp"] == r1["fp"]
    assert r2["flagged"] == r1["flagged"] and r2["best_idx"] == r1["best_idx"]


def test_derived_explicit_thresholds_dedup_sorted():
    """推導:顯式 thresholds 去重排序後使用(亂序重複輸入 → 升冪唯一)。"""
    from cost_curve import cost_curve
    r = cost_curve(SCORES, LABELS, cost_miss=5, cost_fa=1,
                   thresholds=[0.7, 0.15, 0.7, 0.35])
    assert r["thresholds"] == pytest.approx([0.15, 0.35, 0.7])
    assert r["fn"] == [0, 0, 1] and r["fp"] == [3, 1, 0]
