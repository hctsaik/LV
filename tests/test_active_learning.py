"""驗收:active_learning —— 主動學習取樣 Priority Score(架構文件第九節)。
先紅:scripts/active_learning.py 尚未實作。

Priority = Novelty Distance + Model Disagreement + Near Decision Boundary,× Production Risk Weight;
選樣再加 Cluster Diversity(每群代表,不挑一堆幾乎一樣的)。距離訊號比 softmax entropy 可靠
(對 OOD/未知,softmax 常過度自信),故 novelty 一定納入。
"""
import numpy as np
import pytest

from active_learning import (boundary_score, priority_score,
                             select_for_labeling)


def test_boundary_score():  # AC1:近決策邊界 = top1/top2 機率接近
    proba = np.array([[1.0, 0.0, 0.0],      # 確定 → 不在邊界
                      [1 / 3, 1 / 3, 1 / 3],  # 全均勻 → 最在邊界
                      [0.5, 0.45, 0.05]])     # 前兩類很接近 → 高
    b = boundary_score(proba)
    assert b[0] == pytest.approx(0.0, abs=1e-6)
    assert b[1] == pytest.approx(1.0, abs=1e-6)
    assert b[2] > 0.9


def test_priority_novelty():  # AC2:異常分數越高(越偏離正常)→ 優先越高
    p = priority_score(np.array([0.1, 0.5, 0.9]), w_boundary=0, w_disagreement=0)
    assert p[2] > p[1] > p[0]


def test_priority_disagreement():  # AC3:異常高但 head 沒把握(Unknown)→ 高優先(值得標)
    a = np.array([0.9, 0.9])
    proba = np.array([[0.95, 0.03, 0.02],   # head 有把握
                      [0.40, 0.35, 0.25]])   # head 沒把握
    p = priority_score(a, head_proba=proba, anomaly_threshold=0.5,
                       w_novelty=0, w_boundary=0, w_disagreement=1)
    assert p[1] > p[0]


def test_risk_weight():  # AC4:同 novelty 下,關鍵 layer/tool 風險權重越高 → 優先越高
    # 3 物件讓 minmax 非退化;後兩者 novelty 相同(都=1),只差風險權重
    p = priority_score(np.array([0.0, 1.0, 1.0]), risk_weights=[1.0, 1.0, 3.0],
                       w_boundary=0, w_disagreement=0)
    assert p[2] > p[1]


def test_select_topk_no_cluster():  # AC5:無分群 → 純取 priority 前 k
    assert select_for_labeling(np.array([0.1, 0.9, 0.5, 0.7]), k=2) == [1, 3]


def test_select_cluster_diversity():  # AC6:每群上限 → 不挑一堆同群的,挑各群代表
    pri = np.array([0.9, 0.85, 0.8, 0.7, 0.1])
    clusters = np.array([0, 0, 0, 1, 1])
    sel = select_for_labeling(pri, k=2, cluster_labels=clusters, max_per_cluster=1)
    assert set(sel) == {0, 3}                       # 群0 取最高(idx0)、群1 取最高(idx3)
    assert sum(clusters[i] == 0 for i in sel) <= 1


def test_priority_combines_signals():  # AC7:三訊號加權合成 + 形狀
    a = np.array([0.2, 0.8, 0.5])
    proba = np.array([[0.9, 0.1, 0.0], [0.34, 0.33, 0.33], [0.6, 0.4, 0.0]])
    p = priority_score(a, head_proba=proba, anomaly_threshold=0.4)
    assert p.shape == (3,) and np.isfinite(p).all()
