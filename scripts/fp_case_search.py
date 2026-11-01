"""Explicit local LV similarity retrieval for saved false-positive cases."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from fp_case_store import FalsePositiveCaseStore, checked_roi, digest
from similarity import cosine_similarity_to_ref


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FULL_ROI = [0.0, 0.0, 1.0, 1.0]
MAX_IMAGE_BYTES = 50 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def region_key(sha256: str, roi: list[float]) -> str:
    canonical = json.dumps([sha256, checked_roi(roi)], separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def crop_image(image: Image.Image, roi: list[float]) -> Image.Image:
    x1, y1, x2, y2 = checked_roi(roi)
    width, height = image.size
    left, top = int(x1 * width), int(y1 * height)
    right = max(left + 1, int(x2 * width))
    bottom = max(top + 1, int(y2 * height))
    return image.convert("RGB").crop((left, top, min(width, right), min(height, bottom)))


def model_recipe(models_dir: str | Path | None = None) -> tuple[dict, Path]:
    root = Path(models_dir or os.environ.get("LV_MODELS_DIR") or Path(__file__).resolve().parents[1] / "models").expanduser().resolve()
    checkpoint = root / "dinov2_vits14" / "dinov2_vits14.pth"
    if not checkpoint.is_file():
        checkpoint = root / "dinov2_vits14.pth"
    if not checkpoint.is_file():
        raise ValueError("缺少本機 DINOv2-small 權重；請依 MODELS.md 安裝後再搜尋。案件仍可建立與覆核。")
    recipe = {
        "model": "dinov2_vits14",
        "checkpoint_sha256": digest(checkpoint.read_bytes()),
        "target_res": 224,
        "keep_aspect": True,
        "head": "cls",
        "pad": 0,
        "normalization": "ImageNet",
        "crop": "normalized_xyxy_floor_no_exif_rotation",
        "version": 1,
    }
    return recipe, root


def _list_files(folder: str, limit: int) -> tuple[Path, list[Path], bool]:
    if not folder.strip():
        return Path(), [], False
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("找不到指定的對照資料夾，請確認路徑。")
    found: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            found.append(path)
            if len(found) > limit:
                return root, sorted(found[:limit]), True
    return root, sorted(found), False


def _folder_fingerprint(rows: list[tuple[str, str]]) -> str | None:
    if not rows:
        return None
    payload = "\n".join(f"{name}:{sha}" for name, sha in sorted(rows))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _new_evidence(case: dict, result: dict, row: dict) -> str:
    evidence_id = "e_" + uuid.uuid4().hex[:12]
    case["evidence"].append({
        "id": evidence_id,
        "role": "lv_auxiliary",
        "text": "LV 外觀對照（非外部模型歸因）",
        "at": utc_now(),
        "search_id": result["id"],
        "query_item_id": result["query_item_id"],
        "query_sha256": result["query_sha256"],
        "query_roi": result["query_roi"],
        "recipe": result["recipe"],
        "result_id": row["id"],
        "category": row["category"],
        "similarity": row["similarity"],
        "source_sha256": row["sha256"],
        "source_kind": row.get("source_kind", "unknown"),
        "historical_case": row.get("historical_case"),
    })
    return evidence_id


def search_case(
    case: dict,
    item_id: str,
    store: FalsePositiveCaseStore,
    normal_folder: str = "",
    defect_folder: str = "",
    *,
    candidate_scope: str = "whole_image",
    limit: int = 300,
    top_k: int = 6,
    include_history: bool = True,
    embed_fn=None,
    recipe: dict | None = None,
    models_dir: str | Path | None = None,
    progress=None,
) -> dict:
    if candidate_scope not in {"whole_image", "same_roi"} or not 1 <= limit <= 5000 or not 1 <= top_k <= 100:
        raise ValueError("搜尋範圍或數量不正確")
    item = next((row for row in case.get("items", []) if row["id"] == item_id), None)
    if item is None:
        raise ValueError("找不到搜尋起點區域")
    source = next((row for row in case.get("sources", []) if row["id"] == item["source_id"]), None)
    if source is None:
        raise ValueError("搜尋起點的來源影像不存在")
    if item.get("withdrawn"):
        raise ValueError("搜尋區域已撤回")
    query_path = store.image_path(source)
    query_key = region_key(source["sha256"], item["roi"])
    roots: dict[str, Path] = {}
    records: list[dict] = []
    folder_specs = (("normal", normal_folder), ("defect", defect_folder))
    summaries: dict[str, dict] = {}
    for category, folder in folder_specs:
        root, files, truncated = _list_files(folder, limit)
        roots[category] = root
        summaries[category] = {
            "folder_name": root.name if str(root) != "." else "",
            "n_files": len(files),
            "truncated": truncated,
            "label_status": "supplied_folder_label",
            "folder_sha256": None,
        }
        compare_roi = item["roi"] if candidate_scope == "same_roi" else FULL_ROI
        records.extend({
            "path": str(path),
            "relative_name": path.relative_to(root).as_posix(),
            "name": path.name,
            "category": category,
            "roi": compare_roi,
            "label_status": "supplied_folder_label",
            "external_model": "unknown",
            "source_kind": "reference_folder",
        } for path in files)

    if include_history:
        historical: list[dict] = []
        for other in store.list():
            if other["id"] == case["id"]:
                continue
            for previous in other.get("items", []):
                verdict = previous.get("review", {}).get("verdict")
                if previous.get("withdrawn") or verdict not in {"normal", "defect"}:
                    continue
                historical_source = next(
                    (row for row in other.get("sources", []) if row["id"] == previous["source_id"]),
                    None,
                )
                if historical_source is None:
                    continue
                historical.append({
                    "path": str(store.image_path(historical_source)),
                    "expected_sha": historical_source["sha256"],
                    "name": historical_source["name"],
                    "category": "history",
                    "roi": previous["roi"],
                    "label_status": "reviewed_" + verdict,
                    "historical_case": other["id"],
                    "historical_revision": other["revision"],
                    "external_model": other.get("external", {}).get("model_id") or "unknown",
                    "source_kind": historical_source["kind"],
                    "review": copy.deepcopy(previous["review"]),
                })
        summaries["history"] = {
            "n_regions": min(len(historical), limit),
            "truncated": len(historical) > limit,
            "label_status": "human_reviewed_regions",
        }
        records.extend(historical[:limit])
    else:
        summaries["history"] = {
            "n_regions": 0,
            "truncated": False,
            "label_status": "human_reviewed_regions",
        }

    if not records:
        raise ValueError("請提供至少一個對照資料夾，或啟用已覆核歷史案件。")
    if embed_fn is None:
        recipe, model_root = model_recipe(models_dir)
        from _utils import load_model
        embed_fn = load_model(
            "dinov2_vits14",
            models_dir=model_root,
            keep_aspect=True,
            target_res=224,
            head="cls",
        )
    elif not recipe:
        raise ValueError("注入特徵器時須提供模型 recipe 身分")
    recipe = dict(recipe or {})
    if not recipe.get("model") or not recipe.get("checkpoint_sha256"):
        raise ValueError("模型 recipe 缺少模型或權重 SHA-256 身分")
    recipe_key = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    cache_root = store.root / "embeddings" / recipe_key
    cache_root.mkdir(parents=True, exist_ok=True)
    model_lock = threading.Lock()
    manifest_rows: dict[str, list[tuple[str, str]]] = {"normal": [], "defect": []}

    def vector(data: bytes, sha: str, roi: list[float]) -> np.ndarray:
        key = region_key(sha, roi)
        path = cache_root / (key + ".npy")
        if path.is_file():
            try:
                cached = np.load(path, allow_pickle=False)
                if cached.ndim == 1 and cached.size and np.all(np.isfinite(cached)) and np.linalg.norm(cached) > 0:
                    return cached.astype(np.float32, copy=False)
            except (OSError, ValueError):
                pass
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            crop = crop_image(image, roi)
        with model_lock:
            value = np.asarray(embed_fn(crop), dtype=np.float32).reshape(-1)
        if not value.size or not np.all(np.isfinite(value)) or np.linalg.norm(value) == 0:
            raise ValueError("特徵器回傳無效向量")
        temporary = cache_root / (uuid.uuid4().hex + ".npy")
        try:
            np.save(temporary, value, allow_pickle=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return value

    query_bytes = query_path.read_bytes()
    if digest(query_bytes) != source["sha256"]:
        raise ValueError("案件搜尋起點影像指紋不符，請重新補入來源")
    query_vector = vector(query_bytes, source["sha256"], item["roi"])
    scored: list[dict] = []
    skipped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        try:
            data = Path(record["path"]).read_bytes()
            if not data or len(data) > MAX_IMAGE_BYTES:
                raise ValueError("影像為空或超過 50 MB")
            sha = digest(data)
            if record.get("expected_sha") and record["expected_sha"] != sha:
                raise ValueError("歷史來源影像指紋不符")
            if record["category"] in manifest_rows:
                root = roots[record["category"]]
                manifest_rows[record["category"]].append((record["relative_name"], sha))
            identity = region_key(sha, record["roi"])
            category_identity = (record["category"], identity)
            if identity == query_key or category_identity in seen:
                continue
            candidate_vector = vector(data, sha, record["roi"])
            similarity = float(cosine_similarity_to_ref(candidate_vector[None, :], query_vector)[0])
            scored.append({
                **record,
                "id": identity,
                "sha256": sha,
                "similarity": similarity,
                "reason": "與報案區域外觀相似，需人工確認；不是外部模型原因。",
            })
            seen.add(category_identity)
        except (OSError, ValueError) as exc:
            skipped.append({"name": record["name"], "category": record["category"], "reason": str(exc)})
        finally:
            if progress:
                progress(index + 1, len(records))

    for category in ("normal", "defect"):
        summaries[category]["folder_sha256"] = _folder_fingerprint(manifest_rows[category])
    groups: dict[str, list[dict]] = {}
    for category in ("normal", "defect", "history"):
        category_rows = sorted(
            (row for row in scored if row["category"] == category),
            key=lambda row: (-row["similarity"], row["name"], row["id"]),
        )[:top_k]
        persisted_rows: list[dict] = []
        for row in category_rows:
            try:
                data = Path(row["path"]).read_bytes()
                if digest(data) != row["sha256"]:
                    raise ValueError("來源影像在搜尋後改變，請重新搜尋")
                stored_source = store.ingest(data, row["name"], kind="reference")
                stored_source["original_kind"] = row.get("source_kind", "reference")
                public_row = {
                    key: copy.deepcopy(value)
                    for key, value in row.items()
                    if key not in {"path", "relative_name", "expected_sha"}
                }
                public_row["source"] = stored_source
                persisted_rows.append(public_row)
            except (OSError, ValueError) as exc:
                skipped.append({"name": row["name"], "category": row["category"], "reason": str(exc)})
        groups[category] = persisted_rows

    normal_ids = {row["id"] for row in groups["normal"]}
    defect_ids = {row["id"] for row in groups["defect"]}
    shared = normal_ids & defect_ids
    for rows in groups.values():
        for row in rows:
            row["label_conflict"] = row["id"] in shared

    result = {
        "id": "search_" + uuid.uuid4().hex[:12],
        "at": utc_now(),
        "query_item_id": item_id,
        "query_sha256": source["sha256"],
        "query_roi": list(item["roi"]),
        "candidate_scope": candidate_scope,
        "recipe": recipe,
        "sources": summaries,
        "groups": groups,
        "skipped": skipped[:1000],
        "skipped_total": len(skipped),
        "top_k": top_k,
        "limit_per_source": limit,
    }
    for rows in result["groups"].values():
        for row in rows:
            row["evidence_id"] = _new_evidence(case, result, row)
    case.setdefault("searches", []).append(result)
    return result


def add_candidate(case: dict, search_id: str, result_id: str, category: str) -> bool:
    result = next((entry for entry in case.get("searches", []) if entry["id"] == search_id), None)
    if result is None:
        raise ValueError("找不到搜尋紀錄")
    if category not in {"normal", "defect", "history"}:
        raise ValueError("未知對照類別")
    row = next((candidate for candidate in result.get("groups", {}).get(category, [])
                if candidate["id"] == result_id), None)
    if row is None:
        raise ValueError("找不到搜尋對照項目")
    known_evidence = {entry["id"] for entry in case.get("evidence", [])}
    if row.get("evidence_id") not in known_evidence:
        raise ValueError("對照項目的證據不屬於此案件")
    candidates = case.setdefault("candidates", [])
    if any(candidate["id"] == row["id"] for candidate in candidates):
        return False
    candidates.append({
        "id": row["id"],
        "source": copy.deepcopy(row["source"]),
        "roi": copy.deepcopy(row["roi"]),
        "reason": row["reason"],
        "reference_category": row["category"],
        "evidence_id": row["evidence_id"],
        "similarity": row["similarity"],
        "label_status": row["label_status"],
        "review": {"verdict": "pending", "reviewer": "", "reason": "", "at": utc_now()},
    })
    return True
