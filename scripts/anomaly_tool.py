"""anomaly_tool:物件級瑕疵偵測的協調核心(無 streamlit,可單元測試)。

設計:3_Architect_Design/06_anomaly_tool.md。串起 YOLO 物件 → (分群/正常集) →
patch/物件 異常分數 → 分類排序 → records(path=原圖,守 object-export-crop-bug)。
GUI 層(scripts/app.py 的第 8 工具)只做 widget,邏輯一律委派 run_pipeline。
"""
from __future__ import annotations

import numpy as np


def _object_embeddings(meta, model, embed_fn, pad: float = 0.12) -> np.ndarray:
    """物件級 embedding。embed_fn 給定時逐物件裁切後呼叫(測試注入);否則用 embed_objects。"""
    if embed_fn is None:
        from object_eval import DEFAULT_POLICY, embed_objects
        emb, _ = embed_objects(meta, model, DEFAULT_POLICY)
        return emb
    from PIL import Image

    from interaction import crop_bbox
    from object_eval import _adaptive_pad_px

    out, cur_ip, cur = [], None, None
    for m in meta:
        ip = str(m["image_path"])
        if ip != cur_ip:
            try:
                cur = Image.open(ip).convert("RGB")
            except (OSError, Image.DecompressionBombError):
                cur = None
            cur_ip = ip
        if cur is None:
            out.append(np.zeros(384, dtype=np.float32))
            continue
        iw, ih = cur.size
        b = m["bbox"]
        crop = crop_bbox(cur, *b, pad_px=_adaptive_pad_px(b, iw, ih, pad))
        out.append(np.asarray(embed_fn(crop), dtype=np.float32))
    return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)


def run_pipeline(image_paths, class_names, *, mode: str = "two_stage",
                 score_mode: str = "patch", sample_n: int = 64,
                 confirmed: dict | None = None, contamination: float = 0.05,
                 cache_dir=None, model: str = "dinov2_vits14",
                 extractor=None, embed_fn=None, progress=None) -> dict:
    from anomaly_classify import classify
    from bootstrap_cluster import cluster_objects
    from interaction import compute_outlier_scores, discover_yolo_objects

    meta = discover_yolo_objects(list(image_paths), class_names)
    N = len(meta)
    if N == 0:
        return {"records": [], "ranking": [], "threshold": float("inf"),
                "auroc": None, "n_bad": 0, "candidates": [], "cluster": None,
                "coords2d": None}

    obj_emb = _object_embeddings(meta, model, embed_fn)
    cl = cluster_objects(obj_emb)
    candidates = [int(i) for i in np.where(cl["candidate_mask"])[0]]

    confirmed = {int(k): v for k, v in (confirmed or {}).items()
                 if str(v) in ("good", "bad")}
    good_idx = sorted(i for i, v in confirmed.items() if v == "good" and 0 <= i < N)
    bad_idx = {i for i, v in confirmed.items() if v == "bad"}
    # 正常集(memory bank 來源)優先序:有確認 good → 僅用乾淨 few-shot;否則用正常密群。
    if good_idx:
        normal_set = good_idx
    else:
        normal_set = [i for i in range(N) if cl["normal_mask"][i] and i not in bad_idx]

    bank = None
    if score_mode == "object":
        scores = (compute_outlier_scores(obj_emb, obj_emb[normal_set], k=5)
                  if normal_set else np.zeros(N, dtype=float))
    else:
        from anomaly_score import build_memory_bank, score_object
        from patch_features import embed_objects_patch
        all_pf = embed_objects_patch(meta, model, target_res=224,
                                     cache_dir=cache_dir, extractor=extractor,
                                     progress=progress)
        if normal_set:
            bank = build_memory_bank(np.vstack([all_pf[i]["feats"] for i in normal_set]))
            scores = np.array(
                [score_object(all_pf[i]["feats"], all_pf[i]["grid"], bank)[0]
                 for i in range(N)], dtype=float)
        else:
            scores = np.zeros(N, dtype=float)

    res = classify(scores, confirmed=confirmed, contamination=contamination)
    records = []
    for i, m in enumerate(meta):
        records.append({
            "path": str(m["image_path"]), "image_path": str(m["image_path"]),
            "bbox": tuple(m["bbox"]), "obj_index": int(m["obj_index"]),
            "label": m.get("label", ""), "split": "",
            "score": float(scores[i]), "verdict": str(res["labels"][i]),
        })
    return {"records": records, "ranking": res["ranking"],
            "threshold": res["threshold"], "auroc": res["auroc"],
            "n_bad": res["n_bad"], "candidates": candidates, "cluster": cl,
            "coords2d": None, "obj_emb": obj_emb, "normal_set": normal_set,
            "bank": bank, "scores": scores, "score_mode": score_mode}
