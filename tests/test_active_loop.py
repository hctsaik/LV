"""驗收:active_loop —— 主動學習標註迴圈(設計 3_Architect_Design/M5_active_loop.md)。
先紅:scripts/active_loop.py 未實作。每個測試註解標對應 AC。
"""
import numpy as np
import pytest

from active_loop import (confusion_targeted_priority, entropy_score,
                         label_efficiency_curve, round_summary,
                         should_stop_labeling)


def test_entropy_score():  # AC1:歸一化熵 one-hot→0、均勻→1
    proba = np.array([[1.0, 0.0, 0.0],        # 確定 → 0
                      [1 / 3, 1 / 3, 1 / 3],    # 均勻 → 1
                      [0.5, 0.5, 0.0]])         # 介於 → 中間
    e = entropy_score(proba)
    assert e.shape == (3,)
    assert e[0] == pytest.approx(0.0, abs=1e-6)
    assert e[1] == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < e[2] < 1.0


def test_confusion_priority_entropy():  # AC2:分類頭最混淆(高熵)→ 優先高
    a = np.array([0.9, 0.9])
    proba = np.array([[0.98, 0.01, 0.01],   # 確定
                      [0.34, 0.33, 0.33]])   # 最混淆
    p = confusion_targeted_priority(a, proba, w_novelty=0, w_entropy=1)
    assert p[1] > p[0]


def test_confusion_priority_degenerate():  # AC3:w_entropy=0→純 novelty;head_proba=None→熵項 0
    a = np.array([0.1, 0.5, 0.9])
    p_nov = confusion_targeted_priority(a, None, w_entropy=1)        # 無 proba → 只剩 novelty
    assert p_nov[2] > p_nov[1] > p_nov[0]
    p_only = confusion_targeted_priority(a, np.eye(3)[[0, 0, 0]], w_novelty=1, w_entropy=0)
    assert p_only[2] > p_only[0]                                     # w_entropy=0 → 純 novelty


def _blobs(n_per=80, d=12, k=4, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(k):
        ctr = np.zeros(d, dtype=np.float32); ctr[c] = 4.0
        X.append(rng.normal(0, 1, (n_per, d)).astype(np.float32) + ctr)
        y += [f"c{c}"] * n_per
    return np.vstack(X), np.array(y)


def test_label_efficiency_curve_shape():  # AC4:回學習曲線,n_labels 遞增、acc∈[0,1]、長度≤rounds
    Xp, yp = _blobs(seed=1)
    Xe, ye = _blobs(seed=2)
    curve = label_efficiency_curve(Xp, yp, Xe, ye, strategy="active",
                                   seed_n=20, batch=20, rounds=5)
    assert 1 <= len(curve) <= 5
    ns = [n for n, _ in curve]
    assert ns == sorted(ns) and len(set(ns)) == len(ns)             # n_labels 嚴格遞增
    assert all(0.0 <= a <= 1.0 for _, a in curve)


def test_active_not_worse_than_random():  # AC5:active 最終 acc ≥ random(不劣於隨機)
    Xp, yp = _blobs(seed=1)
    Xe, ye = _blobs(seed=2)
    a = label_efficiency_curve(Xp, yp, Xe, ye, strategy="active", seed_n=16, batch=16, rounds=6)
    r = label_efficiency_curve(Xp, yp, Xe, ye, strategy="random", seed_n=16, batch=16, rounds=6)
    assert a[-1][1] >= r[-1][1] - 0.05                              # 不顯著劣於隨機


def test_should_stop_labeling():  # AC6:走平→True;持續上升→False
    plateau = [[20, 0.60], [40, 0.605], [60, 0.606]]
    rising = [[20, 0.40], [40, 0.55], [60, 0.70]]
    assert should_stop_labeling(plateau, patience=2, min_delta=0.01) is True
    assert should_stop_labeling(rising, patience=2, min_delta=0.01) is False


def test_round_summary():  # AC7:每類已標數 + 總數;接受 list 或 dict
    s_list = round_summary(["good", "bad", "good", "good"])
    assert s_list["total"] == 4 and s_list["per_class"]["good"] == 3 and s_list["per_class"]["bad"] == 1
    s_dict = round_summary({0: "good", 5: "bad", 9: "bad"})
    assert s_dict["total"] == 3 and s_dict["per_class"]["bad"] == 2
