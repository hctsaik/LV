"""dino_head:凍結 DINOv2 物件 embedding 上的 closed-set 分類頭 + 閘控級聯。

對應工業落地架構的 Classification Head:回答「若屬已知類別,最像哪一種瑕疵?」。
frozen foundation feature + 輕量 linear head(LogisticRegression)→ 推論快、易部署。

**閘控級聯(gated cascade)**:closed-set head 只會從已知類別選答案,對未知瑕疵會自信誤分。
故用 Normal Bank 的異常分數先守門(gated_predict):異常低=正常分流;異常高且 head 有把握=給已知
類別;異常高但 head 沒把握=Unknown(送人工/active learning),從結構上防止未知被塞進已知 bucket。
"""
from __future__ import annotations

import numpy as np


def _l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def train_head(obj_emb, labels, *, C: float = 1.0, seed: int = 42,
               l2norm: bool = True) -> dict:
    """在凍結特徵上訓練 LogisticRegression 分類頭。需 ≥2 類。
    回 head dict{model, classes, l2norm, dim}。class_weight='balanced' 因瑕疵類別常不均衡。"""
    from sklearn.linear_model import LogisticRegression
    X = np.asarray(obj_emb, dtype=np.float32)
    y = np.asarray(labels)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("空特徵,無法訓練分類頭")
    if len(set(y.tolist())) < 2:
        raise ValueError("closed-set 分類頭至少需要 2 個類別")
    if l2norm:
        X = _l2n(X)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed,
                             class_weight="balanced")
    clf.fit(X, y)
    return {"model": clf, "classes": list(clf.classes_),
            "l2norm": bool(l2norm), "dim": int(X.shape[1])}


def _logits(clf, X):
    d = clf.decision_function(X)
    if d.ndim == 1:                         # 二類:logits=[0, df],softmax 等價 sigmoid
        d = np.column_stack([np.zeros_like(d), d])
    return d


def _softmax_T(logits, T):
    z = np.asarray(logits, dtype=float) / max(float(T), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(head: dict, obj_emb, labels) -> dict:
    """溫度標定(temperature scaling):在校準集上找單一純量 T 最小化 NLL,修正過度自信。
    回新 head(含 'temperature')。**不改預測**(T 不動 argmax),只縮放信心 → 讓閘控 min_conf 守得準。
    應在 held-out / confirmed 標籤上校準(別用訓練集自身,否則低估過度自信)。"""
    from scipy.optimize import minimize_scalar
    X = np.asarray(obj_emb, dtype=np.float32)
    if head.get("l2norm"):
        X = _l2n(X)
    clf = head["model"]
    logits = _logits(clf, X)
    idx = {c: i for i, c in enumerate(list(clf.classes_))}
    y = np.array([idx[l] for l in np.asarray(labels)])

    def _nll(T):
        p = _softmax_T(logits, T)
        return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))

    res = minimize_scalar(_nll, bounds=(0.05, 10.0), method="bounded")
    return {**head, "temperature": float(res.x)}


def predict_head(head: dict, obj_emb):
    """回 (pred_labels, confidence, proba_matrix)。confidence = 最大類別機率。
    若 head 含 'temperature'(已校準)→ 用溫度縮放後的 softmax 算信心(argmax/預測不變)。"""
    X = np.asarray(obj_emb, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != head.get("dim"):
        raise ValueError(f"特徵維度 {X.shape} 與分類頭 dim={head.get('dim')} 不符")
    if head.get("l2norm"):
        X = _l2n(X)
    clf = head["model"]
    classes = np.asarray(clf.classes_)
    T = head.get("temperature")
    proba = _softmax_T(_logits(clf, X), T) if T else clf.predict_proba(X)
    pred = classes[proba.argmax(axis=1)]
    return pred, proba.max(axis=1).astype(np.float32), proba.astype(np.float32)


def evaluate_head_oof(obj_emb, labels, *, source_ids=None, C: float = 1.0,
                      seed: int = 42, l2norm: bool = True,
                      max_splits: int = 5) -> dict:
    """Sample-level stratified OOF audit for an ephemeral classification head.

    This is deliberately labelled as an audit rather than a production
    holdout: near-duplicates or samples from the same acquisition batch can
    still leak across folds when group metadata is unavailable.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold

    X = np.asarray(obj_emb, dtype=np.float32)
    y = np.asarray(labels)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("OOF audit 的 embedding 與 labels 長度不符")
    classes, counts = np.unique(y, return_counts=True)
    n_splits = min(int(max_splits), int(counts.min()) if len(counts) else 0)
    base = {
        "available": False,
        "kind": "sample_stratified_oof",
        "n_samples": int(len(y)),
        "n_splits": int(n_splits),
        "classes": [str(c) for c in classes],
        "limitations": "非 grouped holdout；近重複或同批次資料仍可能跨 fold。",
    }
    if len(classes) < 2 or n_splits < 2:
        return {**base, "reason": "每類至少需要 2 筆樣本才能做 OOF"}

    oof_pred = np.empty(len(y), dtype=object)
    oof_conf = np.zeros(len(y), dtype=np.float32)
    oof_prob = np.zeros((len(y), len(classes)), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    class_to_col = {str(c): i for i, c in enumerate(classes)}
    for train_idx, test_idx in splitter.split(X, y):
        fold = train_head(X[train_idx], y[train_idx], C=C, seed=seed, l2norm=l2norm)
        pred, conf, prob = predict_head(fold, X[test_idx])
        oof_pred[test_idx] = pred
        oof_conf[test_idx] = conf
        for j, name in enumerate(fold["classes"]):
            oof_prob[test_idx, class_to_col[str(name)]] = prob[:, j]

    samples = {}
    ids = list(source_ids) if source_ids is not None else [str(i) for i in range(len(y))]
    if len(ids) != len(y):
        raise ValueError("OOF audit 的 source_ids 與 labels 長度不符")
    for i, source_id in enumerate(ids):
        probs = {str(c): float(oof_prob[i, j]) for j, c in enumerate(classes)}
        samples[str(source_id)] = {
            "actual": str(y[i]),
            "predicted": str(oof_pred[i]),
            "confidence": float(oof_conf[i]),
            "correct": bool(str(oof_pred[i]) == str(y[i])),
            "probabilities": probs,
        }
    return {
        **base,
        "available": True,
        "accuracy": float(accuracy_score(y, oof_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, oof_pred)),
        "macro_f1": float(f1_score(y, oof_pred, average="macro", zero_division=0)),
        "samples": samples,
    }


def gated_predict(head: dict, obj_emb, anomaly_scores, *,
                  anomaly_threshold: float, min_conf: float = 0.5) -> list:
    """閘控級聯:Normal Bank 異常分數先分流,只有離正常遠才信任 head 的已知類別。
    每物件回:'正常'(異常 < 門檻) / 已知類別(異常高 + head 信心 ≥ min_conf) / 'Unknown'(異常高但沒把握)。"""
    pred, conf, _ = predict_head(head, obj_emb)
    a = np.asarray(anomaly_scores, dtype=float)
    out = []
    for i in range(len(pred)):
        if a[i] < float(anomaly_threshold):
            out.append("正常")
        elif conf[i] >= float(min_conf):
            out.append(str(pred[i]))
        else:
            out.append("Unknown")
    return out


def save_head(path, head: dict) -> None:
    import os
    from pathlib import Path as _Path

    import joblib
    final = str(path)
    _Path(final).parent.mkdir(parents=True, exist_ok=True)   # 落點目錄可能還沒建(預設 .lv_cache 子目錄)
    tmp = final + ".tmp"
    joblib.dump(head, tmp)                              # sklearn 模型 + metadata 一起序列化
    os.replace(tmp, final)                              # atomic


def load_head(path) -> dict:
    import joblib
    return joblib.load(str(path))
