"""al_batch:分批、可 checkpoint、可續跑的主動學習選樣引擎(設計 3_Architect_Design/08_al_batch.md)。

在**凍結模型**下,把「發現物件 → embedding → 對凍結模型評分」分批跑完大資料,每批落不可變 shard,
中斷可續跑不重算,任何時刻給得出「已處理併集」的(暫定)priority Top-K。GUI 與離線服務共用本引擎。

嚴守憲法(見設計 §2.0):
- C4 凍結模型:model_dir 唯讀,永不 train/build bank。
- C6 不寫使用者資料夾:checkpoint_dir 先過 assert_safe_bank_dir。
- C8 分批正確性:批次迴圈只做 per-item;全域運算(_minmax 正規化、多樣性、ranking)只在合併時做一次。
  **不呼叫 run_pipeline**(它每批無條件跑全域 HDBSCAN/門檻);**不用 candidates_in_reference=True**(全域 LOO)。
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
from pathlib import Path

_SCHEMA = 1


# ── 內容雜湊 helpers ────────────────────────────────────────────────────
def _model_version(model_dir) -> str:
    """凍結模型的「磁碟身分」:承重檔內容雜湊,排除 manifest.json(非原子/含 saved_at)。"""
    from export_subset import _file_sha256
    md = Path(model_dir)
    parts = []
    for name in ("bank.npz", "head.joblib", "meta.json", "projection.npz"):  # 排序後拼
        f = md / name
        if f.exists():
            parts.append(f"{name}:{_file_sha256(f)}")
    return hashlib.sha256("\n".join(sorted(parts)).encode("utf-8")).hexdigest()[:16]


def _img_key(image_path, cache: dict) -> str:
    """影像內容雜湊前 16 碼(不可用 stem —— 跨資料夾會撞名)。以 path 快取,同圖只算一次。"""
    p = str(image_path)
    if p not in cache:
        from export_subset import _file_sha256
        cache[p] = _file_sha256(Path(image_path))[:16]
    return cache[p]


def _item_id(rec, img_key: str) -> str:
    """內容定址、排除任何模型/分數相依 → rescore 跨模型不變、跨資料夾不撞。"""
    bx = ",".join(f"{float(v):.3f}" for v in rec["bbox"])
    return f"{img_key}:{bx}:{rec['label']}:{int(rec['obj_index'])}"


def _resume_identity(sorted_paths, object_source, class_names, batch_size,
                     objective, model_version, ref_key="") -> str:
    sig = json.dumps({
        "paths": [str(p) for p in sorted_paths],
        "object_source": object_source,
        "class_names": list(class_names) if class_names else None,
        "batch_size": int(batch_size),
        "objective": objective,
        "model_version": model_version,
        "ref_key": ref_key,          # similar:換參考向量 = 另一 run(否則 "")
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]


# ── 凍結模型載入(C4)──────────────────────────────────────────────────
def load_frozen_model(model_dir) -> dict:
    from anomaly_bank_store import check_compat, load_bank
    md = Path(model_dir)
    b = load_bank(md)                       # 缺 meta.json 會 raise
    meta = b["meta"]
    model = meta.get("model")
    target_res = meta.get("target_res")
    if not model or target_res in (None, ""):
        raise ValueError("凍結模型 meta 缺 model / target_res")
    check_compat(meta, model=model, target_res=target_res)
    score_mode = meta.get("score_mode", "patch")

    bank = None
    ref = None
    if score_mode == "patch":
        if b["vectors"] is None:
            raise ValueError("patch 模式模型缺 bank.npz 向量")
        from anomaly_score import MemoryBank
        bank = MemoryBank(b["vectors"])
    else:  # object
        proj = b["projection"]
        ref = getattr(proj, "good_obj_emb", None) if proj is not None else None
        if ref is None:
            raise ValueError("object 模式模型缺 projection good_obj_emb 作為參考")
        ref = np.asarray(ref, dtype=np.float32)

    head = None
    hp = md / "head.joblib"
    if hp.exists():
        from dino_head import load_head
        head = load_head(str(hp))

    return {"meta": meta, "model": str(model), "target_res": int(target_res),
            "score_mode": score_mode, "bank": bank, "ref": ref, "head": head,
            "model_version": _model_version(md)}


# ── priority 計算(全域正規化只在合併時做,C8)────────────────────────────
def _priority(objective, score, proba):
    """回 (priority_vec, components_dict)。components 給 reason 用。"""
    from active_learning import boundary_score, priority_score
    from active_loop import confusion_targeted_priority, entropy_score
    nov = priority_score(score, w_novelty=1.0, w_boundary=0.0, w_disagreement=0.0)  # = _minmax(score)
    comps = {"novelty": np.asarray(nov, dtype=np.float32)}
    if objective == "novelty":
        return comps["novelty"], comps
    ent = np.asarray(entropy_score(proba), dtype=np.float32)
    bnd = np.asarray(boundary_score(proba), dtype=np.float32)
    comps["entropy"] = ent
    comps["boundary"] = bnd
    if objective == "uncertain":
        pri = comps["novelty"] + bnd + ent
    else:  # confusion = _minmax(novelty) + entropy
        pri = np.asarray(confusion_targeted_priority(score, proba), dtype=np.float32)
    return pri.astype(np.float32), comps


def _reason(objective, comps, i) -> str:
    if objective == "retrieve":
        return f"最像「{comps['best_class'][i]}」樣本(相似度 {float(comps['best_sim'][i]):.2f})"
    if objective == "similar":
        return f"和參考物件相似度 {float(comps['ref_sim'][i]):.2f}"
    if objective == "novelty":
        return f"很不像正常樣本(異常訊號 {float(comps['novelty'][i]):.2f})"
    cands = [("很不像正常樣本", float(comps["novelty"][i])),
             ("模型對這張拿不準", float(comps["entropy"][i])),
             ("介於兩類邊界", float(comps["boundary"][i]))]
    return max(cands, key=lambda t: t[1])[0]


# ── 批次評分(per-item,直接組子函式,不呼叫 run_pipeline)──────────────────
def _score_batch(meta, fm, cache_dir, extractor, embed_fn, obj_cache,
                 objective="novelty", ref_vector=None, ref_vectors=None, ref_labels=None):
    from anomaly_tool import _object_embeddings
    n = len(meta)
    if fm["score_mode"] == "patch":
        from anomaly_score import score_object
        from patch_features import embed_objects_patch
        all_pf = embed_objects_patch(meta, fm["model"], target_res=fm["target_res"],
                                     cache_dir=str(cache_dir), extractor=extractor)
        scores = np.array([score_object(all_pf[i]["feats"], all_pf[i]["grid"], fm["bank"])[0]
                           for i in range(n)], dtype=np.float32)
    else:
        from interaction import compute_outlier_scores
        obj_emb = _object_embeddings(meta, fm["model"], embed_fn, cache_path=obj_cache)
        ref = fm["ref"]
        scores = compute_outlier_scores(
            obj_emb, ref, k=min(5, max(1, len(ref))),
            candidates_in_reference=False).astype(np.float32)

    proba = None
    if fm["head"] is not None:
        from dino_head import predict_head
        obj_emb = _object_embeddings(meta, fm["model"], embed_fn, cache_path=obj_cache)
        proba = np.asarray(predict_head(fm["head"], obj_emb)[2], dtype=np.float32)
    extra = {}
    if objective == "similar":                       # per-item cosine(C8-safe);物件級 embedding
        from similarity import cosine_similarity_to_ref
        oe = _object_embeddings(meta, fm["model"], embed_fn, cache_path=obj_cache)
        extra["ref_sim"] = cosine_similarity_to_ref(oe, ref_vector).astype(np.float32)
    elif objective == "retrieve":                    # M13:多樣本 per-class max-cosine → 最像的類 + 相似度
        from similarity import multi_ref_similarity
        oe = _object_embeddings(meta, fm["model"], embed_fn, cache_path=obj_cache)
        _bl, _bs = multi_ref_similarity(oe, ref_vectors, ref_labels)
        extra["best_sim"] = np.asarray(_bs, dtype=np.float32)
        extra["best_class"] = np.asarray(_bl)
    return scores, proba, extra


# ── checkpoint 讀寫 ─────────────────────────────────────────────────────
def _load_state(ck):
    sp = Path(ck) / "state.json"
    if not sp.exists():
        return None
    try:
        return json.loads(sp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None  # 半寫/毀損 → 視為全新(設計 §4)


def _clear_checkpoint(ck):
    ck = Path(ck)
    for f in ck.glob("shard_*.npz"):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    for name in ("state.json", "finalize.json"):
        try:
            (ck / name).unlink()
        except FileNotFoundError:
            pass


def _merge_and_topk(ck, objective, k):
    """讀所有 committed shard → 依 item_id canonical 排序 → priority → select_for_labeling。
    回 {topk, topk_records, diversity_applied, objects_scored}。"""
    from active_learning import select_for_labeling
    ck = Path(ck)
    state = _load_state(ck) or {"committed_batches": []}
    cols = {kk: [] for kk in ("item_id", "anomaly_score", "image_path",
                              "bbox", "obj_index", "label")}
    probas = []
    has_proba = True
    ref_sims = []
    has_ref_sim = True
    best_sims = []
    best_classes = []
    has_best = True
    for bi in state.get("committed_batches", []):
        f = ck / f"shard_{bi:06d}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=False)
        if len(d["item_id"]) == 0:
            continue
        for kk in cols:
            cols[kk].append(d[kk])
        if "head_proba" in d.files:
            probas.append(d["head_proba"])
        else:
            has_proba = False
        if "ref_sim" in d.files:
            ref_sims.append(d["ref_sim"])
        else:
            has_ref_sim = False
        if "best_sim" in d.files and "best_class" in d.files:
            best_sims.append(d["best_sim"])
            best_classes.append(d["best_class"])
        else:
            has_best = False

    if not cols["item_id"]:
        return {"topk": [], "topk_records": [], "diversity_applied": False,
                "objects_scored": 0}

    item_id = np.concatenate(cols["item_id"])
    score = np.concatenate(cols["anomaly_score"]).astype(np.float32)
    path = np.concatenate(cols["image_path"])
    bbox = np.concatenate(cols["bbox"])
    oidx = np.concatenate(cols["obj_index"])
    label = np.concatenate(cols["label"])
    proba = np.concatenate(probas) if (has_proba and probas) else None
    ref_sim = np.concatenate(ref_sims) if (has_ref_sim and ref_sims) else None
    best_sim = np.concatenate(best_sims) if (has_best and best_sims) else None
    best_class = np.concatenate(best_classes) if (has_best and best_classes) else None
    objects_scored = int(len(item_id))

    # canonical:依 item_id 排序 → tie-break 變成 item_id 的確定函式(跨前端/續跑一致)
    order = np.argsort(item_id, kind="stable")
    item_id, score, path = item_id[order], score[order], path[order]
    bbox, oidx, label = bbox[order], oidx[order], label[order]
    if proba is not None:
        proba = proba[order]
    if ref_sim is not None:
        ref_sim = ref_sim[order]
    if best_sim is not None:
        best_sim = best_sim[order]
        best_class = best_class[order]

    if objective == "similar":
        from similarity import _minmax as _sim_minmax
        pri = (np.asarray(_sim_minmax(ref_sim), dtype=np.float32) if ref_sim is not None
               else np.zeros(len(item_id), dtype=np.float32))
        comps = {"ref_sim": ref_sim if ref_sim is not None
                 else np.zeros(len(item_id), dtype=np.float32)}
    elif objective == "retrieve":                      # M13:優先=最像樣本相似度;帶建議類別
        from similarity import _minmax as _sim_minmax
        pri = (np.asarray(_sim_minmax(best_sim), dtype=np.float32) if best_sim is not None
               else np.zeros(len(item_id), dtype=np.float32))
        comps = {"best_sim": best_sim if best_sim is not None else np.zeros(len(item_id), dtype=np.float32),
                 "best_class": best_class if best_class is not None else np.array(["—"] * len(item_id))}
    else:
        pri, comps = _priority(objective, score, proba)

    if proba is not None and objective not in ("similar", "retrieve"):   # 純最像不做多樣性分群
        cluster_labels = np.argmax(proba, axis=1)
        diversity_applied = True
    else:
        cluster_labels = None
        diversity_applied = False

    sel = select_for_labeling(pri, k=int(k), cluster_labels=cluster_labels)

    topk = [str(item_id[i]) for i in sel]
    recs = []
    for i in sel:
        rec = {"item_id": str(item_id[i]), "image_path": str(path[i]),
               "bbox": [float(x) for x in bbox[i]], "obj_index": int(oidx[i]),
               "label": str(label[i]), "score": float(score[i]),
               "priority": float(pri[i]), "reason": _reason(objective, comps, i)}
        if objective == "retrieve":                    # M13:給 GUI 顯示 + 匯出(建議類別由樣本決定)
            rec["suggested_class"] = str(comps["best_class"][i])
            rec["similarity"] = float(comps["best_sim"][i])
        recs.append(rec)
    return {"topk": topk, "topk_records": recs,
            "diversity_applied": diversity_applied, "objects_scored": objects_scored}


def _images_processed(sorted_paths, committed, batch_size):
    n = 0
    for bi in committed:
        n += len(sorted_paths[bi * batch_size:(bi + 1) * batch_size])
    return n


# ── 主入口 ─────────────────────────────────────────────────────────────
def run_batched(image_paths, *, model_dir, checkpoint_dir, objective="novelty",
                k=100, batch_size=1000, object_source="yolo", class_names=None,
                dataset_dirs=(), embed_fn=None, extractor=None, progress=None,
                resume=True, on_identity_mismatch="error", max_batches=None,
                ref_vector=None, ref_vectors=None, ref_labels=None,
                min_proposal_conf=0.0) -> dict:
    from anomaly_bank_store import _atomic_npz, _atomic_text, assert_safe_bank_dir
    from interaction import discover_whole_images, discover_yolo_objects
    from safe_io import partition_readable

    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir 必填(不得 None,否則無法續跑/快取)")
    if object_source not in ("yolo", "whole_image"):
        raise ValueError("object_source must be 'yolo' or 'whole_image'")
    if objective not in ("novelty", "uncertain", "confusion", "similar", "retrieve"):
        raise ValueError("objective must be novelty/uncertain/confusion/similar/retrieve")
    if objective == "similar" and ref_vector is None:
        raise ValueError("objective 'similar' 需要 ref_vector(參考物件的 embedding 向量)")
    ref_arr = None if ref_vector is None else np.asarray(ref_vector, dtype=np.float32).ravel()
    ref_vectors_arr = None
    if objective == "retrieve":                    # M13:多樣本檢索,樣本比對決定建議類別
        if ref_vectors is None or ref_labels is None or len(ref_labels) == 0:
            raise ValueError("objective 'retrieve' 需要 ref_vectors + ref_labels(樣本集)")
        ref_vectors_arr = np.asarray(ref_vectors, dtype=np.float32)

    ck = Path(checkpoint_dir)
    # C6:先擋不安全目錄(落在來源資料夾內等),再建目錄
    assert_safe_bank_dir(ck, selected_folders=[Path(d) for d in dataset_dirs])
    ck.mkdir(parents=True, exist_ok=True)

    fm = load_frozen_model(model_dir)
    # objective → 所需凍結產物驗證(在任何 embedding 之前)
    if objective in ("uncertain", "confusion") and fm["head"] is None:
        raise ValueError(f"objective '{objective}' 需要含分類頭的模型(缺 head.joblib)")

    sorted_paths = sorted((Path(p) for p in image_paths), key=lambda p: str(p))
    total_images = len(sorted_paths)
    if objective == "similar" and ref_arr is not None:
        ref_key = hashlib.sha256(ref_arr.tobytes()).hexdigest()[:16]
    elif objective == "retrieve" and ref_vectors_arr is not None:   # 換樣本集=另一 run
        ref_key = hashlib.sha256(ref_vectors_arr.tobytes()
                                 + "|".join(str(l) for l in ref_labels).encode()).hexdigest()[:16]
    else:
        ref_key = ""
    ident = _resume_identity(sorted_paths, object_source, class_names,
                             batch_size, objective, fm["model_version"], ref_key)

    cache_dir = ck / f"cache_{fm['model']}_{fm['target_res']}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    obj_cache = ck / f"objemb_{fm['model']}.npz"

    state = _load_state(ck)
    if state is not None and (not resume or state.get("resume_identity") != ident):
        if not resume or on_identity_mismatch == "restart":
            _clear_checkpoint(ck)
            state = None
        else:
            raise ValueError(
                "續跑身分不符(image_paths / object_source / class_names / batch_size / "
                "objective / model_version 任一有變);改 on_identity_mismatch='restart' 可清掉重跑")
    if state is None:
        state = {"schema_version": _SCHEMA, "resume_identity": ident,
                 "model_version": fm["model_version"], "objective": objective,
                 "k": int(k), "batch_size": int(batch_size),
                 "object_source": object_source, "images_total": total_images,
                 "committed_batches": [], "skipped": [], "done": False}

    n_batches = (total_images + batch_size - 1) // batch_size if total_images else 0
    committed = set(state["committed_batches"])
    key_cache: dict = {}
    ran = 0

    for bi in range(n_batches):
        if bi in committed:
            continue
        if max_batches is not None and ran >= max_batches:
            break
        chunk = sorted_paths[bi * batch_size:(bi + 1) * batch_size]
        good, bad = partition_readable(chunk)
        if bad:
            state["skipped"] = sorted(set(state["skipped"]) | {str(b) for b in bad})
        meta = (discover_whole_images(good) if object_source == "whole_image"
                else discover_yolo_objects(good, class_names))
        if min_proposal_conf > 0:                 # M13:丟低信心粗框(score=六欄 YOLO conf;None 保留)
            meta = [m for m in meta
                    if m.get("score") is None or float(m.get("score")) >= float(min_proposal_conf)]

        shard = ck / f"shard_{bi:06d}.npz"
        if len(meta) == 0:
            _atomic_npz(shard, item_id=np.array([], dtype="<U1"),
                        anomaly_score=np.zeros(0, dtype=np.float32))
        else:
            scores, proba, extra = _score_batch(meta, fm, cache_dir, extractor, embed_fn,
                                                obj_cache, objective, ref_arr,
                                                ref_vectors_arr, ref_labels)
            ids = np.array([_item_id(m, _img_key(m["image_path"], key_cache)) for m in meta])
            arrs = {
                "item_id": ids,
                "anomaly_score": scores.astype(np.float32),
                "image_path": np.array([str(m["image_path"]) for m in meta]),
                "bbox": np.array([[float(x) for x in m["bbox"]] for m in meta], dtype=np.float32),
                "obj_index": np.array([int(m["obj_index"]) for m in meta], dtype=np.int32),
                "label": np.array([str(m["label"]) for m in meta]),
            }
            if proba is not None:
                arrs["head_proba"] = proba.astype(np.float32)
            for _ek, _ev in extra.items():           # similar:ref_sim;retrieve:best_sim/best_class
                arrs[_ek] = _ev
            _atomic_npz(shard, **arrs)

        committed.add(bi)
        state["committed_batches"] = sorted(committed)
        _atomic_text(ck / "state.json", json.dumps(state, ensure_ascii=False))  # 最後寫(sentinel)
        ran += 1

        if progress is not None:
            prov = _merge_and_topk(ck, objective, k)
            progress({"images_processed": _images_processed(sorted_paths, committed, batch_size),
                      "images_total": total_images, "objects_scored": prov["objects_scored"],
                      "provisional_topk": prov["topk"], "done": len(committed) >= n_batches})

    done = len(committed) >= n_batches
    if done and not state.get("done"):
        state["done"] = True
        _atomic_text(ck / "state.json", json.dumps(state, ensure_ascii=False))

    res = _merge_and_topk(ck, objective, k)
    res.update({
        "objective": objective, "k": int(k), "images_total": total_images,
        "images_processed": _images_processed(sorted_paths, committed, batch_size),
        "done": done, "provisional": not done,
        "model_version": fm["model_version"], "skipped": list(state["skipped"]),
    })
    return res


def read_checkpoint(checkpoint_dir, *, k=None) -> dict:
    """不再跑批次,只讀 checkpoint 給出目前(暫定或最終)Top-K(GUI 開頁即讀)。"""
    ck = Path(checkpoint_dir)
    state = _load_state(ck)
    if state is None:
        return {"topk": [], "topk_records": [], "objective": None,
                "k": k, "images_total": 0, "images_processed": 0, "objects_scored": 0,
                "done": False, "provisional": True, "diversity_applied": False,
                "model_version": None, "skipped": []}
    kk = int(k) if k is not None else int(state.get("k", 100))
    res = _merge_and_topk(ck, state.get("objective", "novelty"), kk)
    sorted_committed = state.get("committed_batches", [])
    res.update({
        "objective": state.get("objective"), "k": kk,
        "images_total": state.get("images_total", 0),
        "images_processed": _images_processed(
            [Path("x")] * 0, set(sorted_committed), state.get("batch_size", 1)) or 0,
        "done": bool(state.get("done")), "provisional": not bool(state.get("done")),
        "model_version": state.get("model_version"), "skipped": list(state.get("skipped", [])),
    })
    return res
