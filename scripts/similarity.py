"""12 similarity:對參考物件向量的 cosine 相似優先分數(降冪=最像)。

設計:3_Architect_Design/12_similarity.md。純陣列、無 I/O。供 A1(③ 互動找相似)與
A3(al_batch objective=similar)共用。_minmax 與 active_learning 同義(hi<=lo → 全 0)。
"""
from __future__ import annotations

import numpy as np


def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def _minmax(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:                                   # 單一/全等 → 退化為全 0(與 active_learning 一致)
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def cosine_similarity_to_ref(obj_emb, ref_vec) -> np.ndarray:
    """每物件對 ref_vec 的 cosine 相似度 ∈ [-1,1],float32 (N,)。各自 L2 normalize 後內積。
    ref 全零 → ValueError;維度不符 → ValueError;N==0 → 形狀 (0,)。"""
    emb = np.asarray(obj_emb, dtype=np.float32)
    ref = np.asarray(ref_vec, dtype=np.float32).ravel()
    if emb.ndim != 2:
        raise ValueError(f"obj_emb 需為 2D (N,D);得 {emb.shape}")
    if emb.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if emb.shape[1] != ref.shape[0]:
        raise ValueError(f"維度不符:obj_emb D={emb.shape[1]} vs 參考 D={ref.shape[0]}")
    rnorm = float(np.linalg.norm(ref))
    if rnorm < 1e-9:
        raise ValueError("參考向量範數為零(全零),無法算 cosine 相似")
    ref_u = ref / rnorm
    return (_l2norm_rows(emb) @ ref_u).astype(np.float32)


def similarity_priority(obj_emb, ref_vec) -> np.ndarray:
    """= _minmax(cosine_similarity_to_ref(...)),float32 (N,) ∈ [0,1];降冪=最像。
    N==0 → (0,);單一/全等 → 全 0(minmax 退化)。ref 全零/維度不符 → 由 cosine 拋 ValueError。"""
    return _minmax(cosine_similarity_to_ref(obj_emb, ref_vec))


def class_centroid(obj_emb, labels, class_name) -> np.ndarray:
    """回 labels==class_name 的所有物件 embedding 平均向量(未正規化;cosine 內部會正規化),float32 (D,)。
    無此類別 → ValueError(列出可選類別)。供 GUI「參考依據=類別」算參考向量。"""
    emb = np.asarray(obj_emb, dtype=np.float32)
    labs = np.asarray([str(l) for l in labels])
    mask = labs == str(class_name)
    if not mask.any():
        raise ValueError(f"找不到類別 '{class_name}' 的物件(可選:{sorted(set(labs.tolist()))})")
    return emb[mask].mean(axis=0).astype(np.float32)


def multi_ref_similarity(obj_emb, ref_vectors, ref_labels):
    """對每物件回 (best_labels, best_sims):最像哪一類樣本、多像(per-class **max**-cosine)。
    設計:3_Architect_Design/12_similarity.md(M13 Task1)。max 不取平均(樣本少/含次型,平均會糊)。
    N==0→兩空陣列;M==0/維度不符/任一樣本全零→ValueError。"""
    emb = np.asarray(obj_emb, dtype=np.float32)
    refs = np.asarray(ref_vectors, dtype=np.float32)
    labels = np.asarray([str(l) for l in ref_labels])
    if refs.ndim != 2 or refs.shape[0] == 0:
        raise ValueError("需要至少一顆樣本(ref_vectors 為空)")
    if len(labels) != refs.shape[0]:
        raise ValueError(f"ref_labels 長度({len(labels)})需等於樣本數({refs.shape[0]})")
    if emb.ndim != 2:
        raise ValueError(f"obj_emb 需為 2D (N,D);得 {emb.shape}")
    if emb.shape[0] == 0:
        return np.array([], dtype=labels.dtype), np.zeros((0,), dtype=np.float32)
    if emb.shape[1] != refs.shape[1]:
        raise ValueError(f"維度不符:obj_emb D={emb.shape[1]} vs 樣本 D={refs.shape[1]}")
    rnorms = np.linalg.norm(refs, axis=1)
    if float(rnorms.min()) < 1e-9:
        raise ValueError("有樣本向量範數為零(全零),無法比對")
    cos = _l2norm_rows(emb) @ (refs / rnorms[:, None]).T          # (N, M) cosine
    classes = list(dict.fromkeys(labels.tolist()))               # 保出現順序(tie→前者)
    per_class = np.stack([cos[:, labels == c].max(axis=1) for c in classes], axis=1)  # (N, C)
    best_idx = np.argmax(per_class, axis=1)
    best_sims = per_class[np.arange(emb.shape[0]), best_idx].astype(np.float32)
    best_labels = np.array([classes[i] for i in best_idx])
    return best_labels, best_sims
