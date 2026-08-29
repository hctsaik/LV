"""驗收:dino_head —— 凍結 DINOv2 物件 embedding 上的 closed-set 分類頭 + 閘控級聯。
先紅:scripts/dino_head.py 尚未實作。

對應架構:Classification Head(回答「若屬已知類別,最像哪種瑕疵?」)+ gated cascade
(Normal Bank 異常分數先守門 → 只有離正常遠才信任 head 的已知類別預測,防未知被自信誤分)。
"""
import numpy as np
import pytest

from dino_head import (evaluate_head_oof, fit_temperature, gated_predict, load_head,
                       predict_head, save_head, train_head)

D = 16
CLASSES = ["bubble", "reflection", "particle"]


def _data(n=80, seed=0):
    """3 類,各沿不同軸的線性可分群。"""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c, name in enumerate(CLASSES):
        center = np.zeros(D, dtype=np.float32); center[c] = 5.0
        X.append(center + rng.normal(0, 0.4, (n, D)).astype(np.float32))
        y += [name] * n
    return np.vstack(X), np.array(y)


def test_train_predict_separable():  # AC1:可分資料 → 高準確 + classes 正確
    X, y = _data()
    head = train_head(X, y)
    assert set(head["classes"]) == set(CLASSES)
    pred, conf, proba = predict_head(head, X)
    assert (pred == y).mean() >= 0.95
    assert proba.shape == (len(X), len(CLASSES))


def test_confidence_and_proba_valid():  # AC2:信心 ∈[0,1]、機率列和=1
    X, y = _data()
    _, conf, proba = predict_head(train_head(X, y), X)
    assert conf.min() >= 0.0 and conf.max() <= 1.0
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_needs_two_classes():  # AC3:<2 類 → ValueError(不可訓 closed-set)
    X = np.zeros((10, D), dtype=np.float32)
    with pytest.raises(ValueError):
        train_head(X, np.array(["only"] * 10))


def test_save_load_roundtrip(tmp_path):  # AC4:存讀後預測一致
    X, y = _data()
    head = train_head(X, y)
    p = tmp_path / "sub" / "deep" / "head.joblib"   # 巢狀不存在目錄 → save_head 應自建
    save_head(p, head)
    head2 = load_head(p)
    p1, _, _ = predict_head(head, X)
    p2, _, _ = predict_head(head2, X)
    assert (p1 == p2).all()
    assert head2["classes"] == head["classes"]


def test_gated_cascade():  # AC5:閘控級聯 —— 正常分流 / 已知分類 / Unknown 守門
    X, y = _data()
    head = train_head(X, y)
    bubble = np.zeros((1, D), dtype=np.float32); bubble[0, 0] = 5.0   # 明確 bubble(高信心)
    boundary = np.zeros((1, D), dtype=np.float32)                     # 介於各類間 → 低信心
    boundary[0, 0] = 2.5; boundary[0, 1] = 2.5

    # 異常分數低(像正常)→ 不論 head 怎麼說都先分流成「正常」
    assert gated_predict(head, bubble, [0.1], anomaly_threshold=0.5)[0] == "正常"
    # 異常分數高 + head 有信心 → 給已知類別
    assert gated_predict(head, bubble, [0.9], anomaly_threshold=0.5, min_conf=0.5)[0] == "bubble"
    # 異常分數高 + head 沒把握(邊界/未知)→ Unknown(不硬塞已知類別)
    assert gated_predict(head, boundary, [0.9], anomaly_threshold=0.5, min_conf=0.9)[0] == "Unknown"


def test_predict_dim_mismatch_raises():  # AC6:特徵維度不符 → 明確 raise
    head = train_head(*_data())
    with pytest.raises((ValueError, Exception)):
        predict_head(head, np.zeros((3, D + 5), dtype=np.float32))


def test_temperature_calibration():  # AC7:溫度校準不改預測、降低 NLL(修正過度自信)
    X, y = _data()
    head = train_head(X, y)
    cal = fit_temperature(head, X, y)
    assert "temperature" in cal and cal["temperature"] > 0
    p0, c0, pr0 = predict_head(head, X)
    p1, c1, pr1 = predict_head(cal, X)
    assert list(p0) == list(p1)                       # 校準不改預測(argmax 不變),只縮放信心

    def _nll(proba):
        return float(-np.mean(np.log(proba[np.arange(len(y)), [head["classes"].index(t) for t in y]] + 1e-12)))

    assert _nll(pr1) <= _nll(pr0) + 1e-9              # 校準後 NLL 不增(校準的定義)


def test_calibration_gated_uses_calibrated_conf():  # AC8:閘控吃校準後信心(過度自信被壓→更易進 Unknown)
    X, y = _data()
    head = train_head(X, y)
    # 人工塞極小溫度的反例不需要;用大 T 模擬「壓低信心」→ 邊界點更可能 Unknown
    cal = {**head, "temperature": 5.0}
    boundary = np.zeros((1, D), dtype=np.float32); boundary[0, 0] = 2.5; boundary[0, 1] = 2.5
    _, c_raw, _ = predict_head(head, boundary)
    _, c_cal, _ = predict_head(cal, boundary)
    assert c_cal[0] <= c_raw[0] + 1e-6               # 高溫 → 信心被壓低(過度自信修正方向)


def test_oof_audit_reports_generalization_and_per_sample_result():
    X, y = _data(n=12)
    source_ids = [f"sample-{i}" for i in range(len(y))]
    audit = evaluate_head_oof(X, y, source_ids=source_ids)
    assert audit["available"] is True
    assert audit["balanced_accuracy"] >= 0.95
    assert audit["macro_f1"] >= 0.95
    assert set(audit["samples"]) == set(source_ids)
    assert audit["samples"]["sample-0"]["actual"] == y[0]
    assert "非 grouped holdout" in audit["limitations"]


def test_oof_audit_unavailable_when_a_class_has_one_sample():
    X = np.zeros((3, D), dtype=np.float32)
    audit = evaluate_head_oof(X, np.array(["a", "a", "b"]))
    assert audit["available"] is False
    assert audit["n_splits"] == 1
