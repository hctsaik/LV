"""驗收:anomaly_classify(設計 04_anomaly_classify.md)。先紅:scripts/anomaly_classify.py 未實作。"""
import numpy as np
import pytest

from anomaly_classify import classify


def test_separable_with_both_labels():  # AC1
    rng = np.random.default_rng(0)
    normal = rng.uniform(0.0, 0.2, 40)
    bad = rng.uniform(0.8, 1.0, 10)
    scores = np.concatenate([normal, bad])
    confirmed = {0: "good", 1: "good", 5: "good", 40: "bad", 41: "bad"}
    r = classify(scores, confirmed=confirmed)
    assert 0.2 < r["threshold"] < 0.8
    labels = r["labels"]
    assert all(labels[i] == "good" for i in range(40))
    assert all(labels[i] == "bad" for i in range(40, 50))
    assert r["auroc"] == pytest.approx(1.0)


def test_unlabeled_percentile_exact_count():  # AC2(精確計數)
    scores = np.arange(100, dtype=float)
    r = classify(scores, contamination=0.1)
    assert r["threshold"] == pytest.approx(np.quantile(scores, 0.9))
    assert r["n_bad"] == 10
    bad_idx = {i for i, v in enumerate(r["labels"]) if v == "bad"}
    assert bad_idx == set(range(90, 100))


def test_ranking_desc_stable():  # AC3
    scores = np.array([0.5, 0.9, 0.1, 0.9, 0.3])
    r = classify(scores, contamination=0.2)
    assert r["ranking"][0] == 1            # argmax (tie 1 before 3 by index)
    desc = [scores[i] for i in r["ranking"]]
    assert desc == sorted(desc, reverse=True)
    assert r["ranking"][:2] == [1, 3]      # tie-break 升冪索引


def test_only_bad_confirmed():  # AC4
    scores = np.array([0.1, 0.2, 0.7, 0.9])
    r = classify(scores, confirmed={2: "bad"})
    assert r["threshold"] <= scores[2]
    assert r["labels"][2] == "bad"
    assert all(r["labels"][i] == "bad" for i in range(4) if scores[i] >= r["threshold"])


def test_auroc_matches_sklearn():  # AC5(對齊 sklearn,設計成 0.75)
    from sklearn.metrics import roc_auc_score
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    confirmed = {0: "good", 2: "bad", 4: "good", 6: "good", 7: "bad",
                 1: "good", 3: "bad", 5: "good"}
    y = [1 if confirmed[i] == "bad" else 0 for i in sorted(confirmed)]
    s = [scores[i] for i in sorted(confirmed)]
    r = classify(scores, confirmed=confirmed)
    assert round(r["auroc"], 3) == round(roc_auc_score(y, s), 3)


def test_empty_and_all_invalid_confirmed():  # AC6(推導:退化不崩)
    r = classify(np.array([]), confirmed={9: "bad"})
    assert list(r["labels"]) == [] and r["ranking"] == [] and r["n_bad"] == 0
    r2 = classify(np.array([0.1, 0.5]), confirmed={99: "bad", 0: "weird"})
    assert r2["auroc"] is None  # 全越界/非法 → 退回無標籤路
