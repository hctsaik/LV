"""Explicit, source-traceable LV retrieval for case comparisons (not external XAI)."""
from __future__ import annotations

import io
import json
import os
import threading
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from fp_cases import FULL_ROI, _evidence, digest, region_key, roi_checked, utc_now
from similarity import cosine_similarity_to_ref

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def crop_image(image, roi):
    r = roi_checked(roi)
    w, h = image.size
    left, top = int(r[0] * w), int(r[1] * h)
    right, bottom = max(left + 1, int(r[2] * w)), max(top + 1, int(r[3] * h))
    return image.convert("RGB").crop((left, top, min(w, right), min(h, bottom)))


@lru_cache(maxsize=2)
def _model(model_path, checkpoint_sha):
    from _utils import load_model
    root = Path(model_path).parent.parent if Path(model_path).parent.name == "dinov2_vits14" else Path(model_path).parent
    return load_model("dinov2_vits14", models_dir=root, keep_aspect=True, target_res=224, head="cls"), threading.Lock()


def model_recipe():
    root = Path(os.environ.get("LV_MODELS_DIR") or Path(__file__).resolve().parents[1] / "models")
    path = root / "dinov2_vits14" / "dinov2_vits14.pth"
    if not path.exists():
        path = root / "dinov2_vits14.pth"
    if not path.exists():
        raise ValueError("缺少本機 DINOv2 權重；請依 MODELS.md 安裝 dinov2_vits14 後再搜尋。案件仍可儲存與覆核。")
    sha = digest(path.read_bytes())
    return {"model": "dinov2_vits14", "checkpoint_sha256": sha, "checkpoint_path": str(path.resolve()),
            "target_res": 224, "keep_aspect": True, "head": "cls", "pad": 0,
            "normalization": "ImageNet", "crop": "normalized_xyxy_floor_no_exif_rotation", "version": 1}


def _files(folder, limit):
    if not folder.strip():
        return [], False
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"找不到對照資料夾：{root}")
    found = []
    # Bound traversal as well as inference; disclose truncation rather than pretending full coverage.
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            found.append(path)
            if len(found) > limit:
                return sorted(found[:limit]), True
    return sorted(found), False


def search(case, item_id, store, normal_folder="", defect_folder="", *,
           candidate_scope="whole_image", limit=300, top_k=6, include_history=True,
           embed_fn=None, recipe=None, progress=None, datasets=None):
    if candidate_scope not in {"whole_image", "same_roi"} or not 1 <= limit <= 5000 or not 1 <= top_k <= 100:
        raise ValueError("搜尋範圍或數量不正確")
    item = next(i for i in case["items"] if i["id"] == item_id)
    if item.get("withdrawn"):
        raise ValueError("搜尋區域已撤回")
    source = next(s for s in case["sources"] if s["id"] == item["source_id"])
    query_path = store.image_path(source)
    query_key = region_key(source["sha256"], item["roi"])
    records, sources = [], {}
    for category, folder in (("normal", normal_folder), ("defect", defect_folder)):
        files, truncated = _files(folder, limit)
        sources[category] = {"folder": str(Path(folder).resolve()) if folder.strip() else "",
                             "n_files": len(files), "truncated": truncated, "label_status": "supplied_folder_label"}
        records.extend({"path": str(p), "name":p.name, "category": category, "roi": item["roi"] if candidate_scope == "same_roi" else FULL_ROI,
                        "label_status": "supplied_folder_label", "external_model": "unknown"} for p in files)
    if datasets:
        from investigation_data import DatasetRegistry
        registry = DatasetRegistry(store)
        for category, dataset_id in datasets.items():
            if not dataset_id or category not in {"normal", "defect"}:
                continue
            dataset = registry.load(dataset_id)
            sources[category] = {"dataset_id":dataset_id,"name":dataset["name"],"n_files":min(limit,len(dataset["images"])),"truncated":len(dataset["images"])>limit,"label_status":"supplied_dataset_label"}
            records.extend({"path":str(store.image_path(r["source"])),"name":r["name"],"expected_sha":r["source"]["sha256"],"category":category,
                "roi":item["roi"] if candidate_scope=="same_roi" else FULL_ROI,
                "label_status":"supplied_dataset_label","external_model":"unknown","dataset_id":dataset_id} for r in dataset["images"][:limit])
    if include_history:
        historical = []
        for other in store.list():
            if other["id"] == case["id"]:
                continue
            for previous in other["items"]:
                if previous.get("withdrawn") or previous["review"]["verdict"] not in {"normal", "defect"}:
                    continue
                s = next(s for s in other["sources"] if s["id"] == previous["source_id"])
                historical.append({"path": str(store.root / "assets" / s["asset"]), "expected_sha": s["sha256"],
                                   "name":s["name"],"source_path": s["source_path"], "source_kind": s["kind"], "category": "history", "roi": previous["roi"],
                                   "label_status": "reviewed_" + previous["review"]["verdict"],
                                   "historical_case": other["id"], "historical_revision": other["revision"],
                                   "external_model": other["external"]["model_id"] or "unknown",
                                   "review": previous["review"]})
        sources["history"] = {"n_regions": min(len(historical), limit), "truncated": len(historical) > limit}
        records.extend(historical[:limit])
    lock = threading.Lock()
    if embed_fn is None:
        recipe = model_recipe()
        embed_fn, lock = _model(recipe["checkpoint_path"], recipe["checkpoint_sha256"])
    elif not recipe:
        raise ValueError("注入特徵器須提供實際 recipe 身分")
    recipe = dict(recipe)
    recipe_key = digest(json.dumps(recipe, sort_keys=True).encode())
    cache = store.root / "embeddings" / recipe_key
    cache.mkdir(parents=True, exist_ok=True)

    def vector(data, sha, roi):
        key = region_key(sha, roi)
        path = cache / (key + ".npy")
        if path.exists():
            try:
                value = np.load(path, allow_pickle=False)
                if value.ndim == 1 and np.all(np.isfinite(value)) and np.linalg.norm(value) > 0:
                    return value
            except (OSError, ValueError):
                pass
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            crop = crop_image(image, roi)
        with lock:
            value = np.asarray(embed_fn(crop), dtype=np.float32).reshape(-1)
        if not value.size or not np.all(np.isfinite(value)) or np.linalg.norm(value) == 0:
            raise ValueError("特徵器回傳無效向量")
        tmp = cache / (uuid.uuid4().hex + ".npy")
        try:
            np.save(tmp, value, allow_pickle=False)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return value

    query = vector(query_path.read_bytes(), source["sha256"], item["roi"])
    results, skipped, seen = [], [], set()
    for index, record in enumerate(records):
        try:
            data = Path(record["path"]).read_bytes()
            sha = digest(data)
            if record.get("expected_sha", sha) != sha:
                raise ValueError("歷史影像內容已變更")
            identity = region_key(sha, record["roi"])
            category_key = (record["category"], identity)
            if identity == query_key or category_key in seen:
                continue
            v = vector(data, sha, record["roi"])
            score = float(cosine_similarity_to_ref(v[None, :], query)[0])
            row = {**record, "sha256": sha, "similarity": score, "id": identity,
                   "reason": "與報案區域外觀相似，需人工確認；不是外部模型原因"}
            results.append(row)
            seen.add(category_key)
        except (OSError, ValueError) as exc:
            skipped.append({"path": record["path"], "reason": str(exc)})
        finally:
            if progress:
                progress(index + 1, len(records))
    groups = {}
    for category in ("normal", "defect", "history"):
        groups[category] = sorted((r for r in results if r["category"] == category),
                                  key=lambda r: (-r["similarity"], r["path"]))[:top_k]
    shared = {r["id"] for r in results if r["category"] == "normal"} & {r["id"] for r in results if r["category"] == "defect"}
    for rows in groups.values():
        for r in rows:
            r["label_conflict"] = r["id"] in shared
    return {"id": "search_" + uuid.uuid4().hex[:12], "at": utc_now(), "query_item_id": item_id,
            "query_sha256": source["sha256"], "query_roi": item["roi"], "candidate_scope": candidate_scope,
            "recipe": recipe, "sources": sources, "groups": groups, "skipped": skipped,
            "top_k": top_k, "limit_per_source": limit}


def record_search(case, result):
    for rows in result["groups"].values():
        for row in rows:
            row["evidence_id"] = _evidence(case, "lv_auxiliary", "LV 外觀對照（非正式模型歸因）",
                search_id=result["id"], attribution_evidence_id=result.get("attribution_evidence_id"), query_item_id=result["query_item_id"], query_sha256=result["query_sha256"],
                query_roi=result["query_roi"], recipe=result["recipe"], **row)
    case["searches"].append(result)
