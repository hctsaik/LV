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


def predict_head(head: dict, obj_emb):
    """回 (pred_labels, confidence, proba_matrix)。confidence = 最大類別機率。"""
    X = np.asarray(obj_emb, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != head.get("dim"):
        raise ValueError(f"特徵維度 {X.shape} 與分類頭 dim={head.get('dim')} 不符")
    if head.get("l2norm"):
        X = _l2n(X)
    clf = head["model"]
    proba = clf.predict_proba(X)
    classes = np.asarray(clf.classes_)
    pred = classes[proba.argmax(axis=1)]
    return pred, proba.max(axis=1).astype(np.float32), proba.astype(np.float32)


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
