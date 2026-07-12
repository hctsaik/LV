"""linear probe 回合代理評估 + 報酬遞減判讀。

設計:3_Architect_Design/12_probe_eval.md

回讀完 30 秒內回答「這批標註值不值」:frozen embedding + LogisticRegression(dino_head 同款
配方,泛化到任何有 label 的記錄集)訓一個代理 probe,把分數掛進回合帳本(round_ledger),
長出「指標 vs 累積標註量」的報酬遞減視圖與**附理由的**停止建議。

誠實界線:probe 是**方向性代理**,不是最終模型成績(UI 常駐標示)。
**不做自動停** —— plateau 只給建議,是否停止由使用者決定(產品紀律,任何把 plateau 變成
自動行為的實作都算違約)。

純函式、決定論、不 import streamlit。
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split


def train_probe(
    emb: np.ndarray,
    labels: list[str],
    *,
    test_size: float = 0.25,
    min_per_class: int = 4,
    seed: int = 42,
) -> dict:
    """frozen embedding + LogisticRegression → 代理指標。

    分層切分(每類在 train/test 依比例分配),故少數類不會整批落到單側把指標弄假。
    任一類樣本 < ``min_per_class`` → ``ValueError("insufficient_class:<類名>")``
    (不靜默降級:類太少時算出來的分數沒有意義,要讓 UI 誠實地說是哪一類不足)。

    回 ``{"acc", "macro_f1", "per_class": {類: {"n", "recall"}}, "n_train", "n_test", "classes"}``,
    其中 ``per_class`` 的 ``n`` 是該類在 **test** 集的支持數。全決定論。
    """
    emb = np.asarray(emb, dtype=float)
    labels = list(labels)
    if len(emb) != len(labels):
        raise ValueError(f"emb and labels length mismatch: {len(emb)} != {len(labels)}")

    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError(f"insufficient_class:{classes[0] if classes else '<empty>'}"
                         " — 至少要 2 類才能訓 probe")
    for c in classes:
        n = labels.count(c)
        if n < min_per_class:
            raise ValueError(
                f"insufficient_class:{c} — 只有 {n} 張,需要至少 {min_per_class} 張")

    y = np.asarray(labels)
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y)

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(emb[idx_train], y[idx_train])
    pred = clf.predict(emb[idx_test])
    truth = y[idx_test]

    recalls = recall_score(truth, pred, labels=classes, average=None, zero_division=0)
    per_class = {
        c: {"n": int((truth == c).sum()), "recall": float(r)}
        for c, r in zip(classes, recalls)
    }
    return {
        "acc": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=classes,
                                   average="macro", zero_division=0)),
        "per_class": per_class,
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "classes": classes,
    }


def plateau_advice(
    xs: list[int],
    ys: list[float],
    *,
    min_points: int = 3,
    eps_per_100: float = 0.005,
) -> dict:
    """讀 learning_curve(round_ledger)判報酬遞減 → 附數字的**建議**(不自動停)。

    斜率只看最後兩點的**增量**(每 100 張標註的指標增量),故對曲線整體平移不變。
    回 ``{"status": "insufficient"|"improving"|"plateau", "slope_per_100", "reason"}``,
    ``reason`` 是可直接顯示的中文字串。
    """
    n = min(len(xs), len(ys))
    if n < min_points:
        return {
            "status": "insufficient",
            "slope_per_100": None,
            "reason": f"曲線只有 {n} 個點,還差 {min_points - n} 點才能判斷報酬遞減。",
        }

    dx = max(xs[-1] - xs[-2], 1)  # clamp:同標註量的兩點不除零
    slope = (ys[-1] - ys[-2]) / dx * 100.0
    if slope < eps_per_100:
        status = "plateau"
        reason = (f"最近一批每 100 張標註只帶來 {slope:.3f} 的指標增量"
                  f"(低於 {eps_per_100:.3f} 門檻)→ 已進入報酬遞減。")
    else:
        status = "improving"
        reason = (f"最近一批每 100 張標註帶來 {slope:.3f} 的指標增量"
                  f"(高於 {eps_per_100:.3f} 門檻)→ 仍在進步,值得繼續標。")
    return {"status": status, "slope_per_100": float(slope), "reason": reason}
