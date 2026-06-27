"""anomaly_tool:物件級瑕疵偵測的協調核心(無 streamlit,可單元測試)。

設計:3_Architect_Design/06_anomaly_tool.md。串起 YOLO 物件 → (分群/正常集) →
patch/物件 異常分數 → 分類排序 → records(path=原圖,守 object-export-crop-bug)。
GUI 層(scripts/app.py 的第 8 工具)只做 widget,邏輯一律委派 run_pipeline。
"""
from __future__ import annotations

import numpy as np


def _object_embeddings(meta, model, embed_fn, pad: float = 0.12,
                       progress=None, cache_path=None) -> np.ndarray:
    """物件級 embedding。embed_fn 給定時逐物件裁切後呼叫(測試注入);否則用 embed_objects。
    progress(done, total) 逐物件回報(供 GUI 進度條);cache_path 給定時快取到 npz(重跑免重算)。"""
    if embed_fn is None:
        from object_eval import DEFAULT_POLICY, embed_objects
        emb, _ = embed_objects(meta, model, DEFAULT_POLICY,
                               cache_path=cache_path, progress=progress)
        return emb
    from interaction import crop_bbox
    from object_eval import _adaptive_pad_px
    from patch_features import model_dim
    from safe_io import safe_open_image

    out, cur_ip, cur = [], None, None
    total = len(meta)
    for _i, m in enumerate(meta):
        ip = str(m["image_path"])
        if ip != cur_ip:
            cur = safe_open_image(ip)   # 壞圖回 None → 走下方補零保索引對齊
            cur_ip = ip
        if cur is None:
            out.append(np.zeros(model_dim(model), dtype=np.float32))
        else:
            iw, ih = cur.size
            b = m["bbox"]
            crop = crop_bbox(cur, *b, pad_px=_adaptive_pad_px(b, iw, ih, pad))
            out.append(np.asarray(embed_fn(crop), dtype=np.float32))
        if progress:
            progress(_i + 1, total)
    return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)


def run_pipeline(image_paths, class_names, *, mode: str = "two_stage",
                 score_mode: str = "patch", sample_n: int = 64,
                 confirmed: dict | None = None, contamination: float = 0.05,
                 cache_dir=None, model: str = "dinov2_vits14", target_res: int = 224,
                 extractor=None, embed_fn=None, progress=None,
                 external_bank=None, external_ref=None) -> dict:
    from anomaly_classify import classify
    from bootstrap_cluster import cluster_objects
    from interaction import compute_outlier_scores, discover_yolo_objects
    from safe_io import partition_readable

    # 壞檔前置過濾：擋在 meta 之外,下游索引(records/scores/obj_emb)才一致。
    image_paths, _bad = partition_readable(list(image_paths))
    skipped = [str(b) for b in _bad]

    meta = discover_yolo_objects(list(image_paths), class_names)
    N = len(meta)
    if N == 0:
        return {"records": [], "ranking": [], "threshold": float("inf"),
                "auroc": None, "n_bad": 0, "candidates": [], "cluster": None,
                "coords2d": None, "skipped": skipped}

    def _p(frac, text):
        if progress:
            try:
                progress(float(min(max(frac, 0.0), 1.0)), str(text))
            except Exception:
                pass

    _p(0.02, f"載入模型 / 物件特徵…(共 {N} 物件)")
    _obj_cache = None
    if cache_dir is not None:
        from pathlib import Path
        _obj_cache = Path(cache_dir).parent / f"objemb_{model}.npz"  # 跨 score_mode/res 共用
    obj_emb = _object_embeddings(
        meta, model, embed_fn, cache_path=_obj_cache,
        progress=lambda d, t: _p(0.05 + 0.38 * d / max(t, 1), f"物件特徵 {d}/{t}"))
    _p(0.45, "分群…")
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
    if external_bank is not None and score_mode != "object":
        # 掛載外部 bank(patch 級):跳過 normal_set/建 bank,直接用外部 bank 對所有物件評分。
        # 新資料夾 patch 特徵用「傳入的 model/target_res」(= bank meta 的鎖死值,非 GUI 現值)。
        from anomaly_score import MemoryBank, score_object
        from patch_features import embed_objects_patch
        bank = (external_bank if isinstance(external_bank, MemoryBank)
                else MemoryBank(external_bank))
        _p(0.46, "patch 特徵…(對照已掛載 bank;首次載入模型較久)")
        all_pf = embed_objects_patch(
            meta, model, target_res=target_res, cache_dir=cache_dir, extractor=extractor,
            progress=lambda d, t: _p(0.46 + 0.4 * d / max(t, 1), f"patch 特徵 {d}/{t}"))
        scores = np.empty(N, dtype=float)
        for i in range(N):
            scores[i] = score_object(all_pf[i]["feats"], all_pf[i]["grid"], bank)[0]
            if i % 8 == 0 or i == N - 1:
                _p(0.86 + 0.12 * (i + 1) / N, f"評分 {i + 1}/{N}")
    elif external_ref is not None and score_mode == "object":
        # 掛載外部 ref(物件級):對舊 good 的 obj_emb 算離群(對照已掛載參照,非自身)。
        _p(0.6, "評分(物件級,對照已掛載參照)…")
        ref = np.asarray(external_ref, dtype=np.float32)
        scores = (compute_outlier_scores(obj_emb, ref, k=min(5, max(1, len(ref))),
                                         candidates_in_reference=False)
                  if len(ref) > 0 else np.zeros(N, dtype=float))
    elif not normal_set:
        # 沒有正常參考(無確認、且分群找不到正常密群 → 高維/diffuse 多類別資料常 0 群)。
        # 退回「無監督 leave-one-out 物件級離群」:每個物件對『其他物件』算 k 近鄰距離,
        # 鄰居少的(少見/離群類別)分數自然高。不靠分群、不會 self-match 歸零。
        _p(0.6, "評分(無監督 leave-one-out:找少見/離群物件)…")
        scores = (compute_outlier_scores(obj_emb, obj_emb, k=min(5, max(1, N - 1)),
                                         candidates_in_reference=True)
                  if N >= 2 else np.zeros(N, dtype=float))
    elif score_mode == "object":
        _p(0.6, "評分(物件級,對照正常集)…")
        scores = compute_outlier_scores(obj_emb, obj_emb[normal_set],
                                        k=min(5, len(normal_set)),
                                        candidates_in_reference=True)
    else:
        from anomaly_score import build_memory_bank, score_object
        from patch_features import embed_objects_patch
        _p(0.46, "patch 特徵…(首次載入模型較久)")
        all_pf = embed_objects_patch(
            meta, model, target_res=target_res, cache_dir=cache_dir,
            extractor=extractor,
            progress=lambda d, t: _p(0.46 + 0.39 * d / max(t, 1), f"patch 特徵 {d}/{t}"))
        _p(0.85, "建 memory bank…")
        bank = build_memory_bank(np.vstack([all_pf[i]["feats"] for i in normal_set]))
        scores = np.empty(N, dtype=float)
        for i in range(N):
            scores[i] = score_object(all_pf[i]["feats"], all_pf[i]["grid"], bank)[0]
            if i % 8 == 0 or i == N - 1:
                _p(0.86 + 0.12 * (i + 1) / N, f"評分 {i + 1}/{N}")

    _p(0.99, "整理結果…")
    res = classify(scores, confirmed=confirmed, contamination=contamination)
    records = []
    for i, m in enumerate(meta):
        records.append({
            "path": str(m["image_path"]), "image_path": str(m["image_path"]),
            "bbox": tuple(m["bbox"]), "obj_index": int(m["obj_index"]),
            "label": m.get("label", ""), "split": "",
            "score": float(scores[i]), "verdict": str(res["labels"][i]),
        })
    _p(1.0, "完成")
    return {"records": records, "ranking": res["ranking"],
            "threshold": res["threshold"], "auroc": res["auroc"],
            "n_bad": res["n_bad"], "candidates": candidates, "cluster": cl,
            "coords2d": None, "obj_emb": obj_emb, "normal_set": normal_set,
            "bank": bank, "scores": scores, "score_mode": score_mode,
            "skipped": skipped}
