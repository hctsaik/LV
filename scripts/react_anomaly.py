"""React worker helpers for building and persisting LV anomaly banks."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np


def cache_root() -> Path:
    repo = Path(__file__).resolve().parent.parent
    return Path(os.environ.get("LV_CACHE_DIR") or (repo / ".lv_cache")).resolve()


def build_model(request: dict, *, progress=None) -> dict:
    """Build a Normal Bank using the same framework-free LV pipeline as Streamlit."""
    from anomaly_bank_store import confirmed_to_fewshot
    from anomaly_project import fit_projector
    from anomaly_tool import head_unlock_state, run_pipeline
    from dino_explain import checkpoint_identity
    from dino_head import train_head
    from head_adapter import attach_provenance
    from object_eval import DEFAULT_POLICY, classes_for, dataset_cache_dir, list_images
    from _utils import _DEFAULT_MODELS_DIR

    roots = [Path(value).expanduser().resolve(strict=True) for value in request["train_folders"]]
    if not roots or any(not root.is_dir() for root in roots):
        raise ValueError("請選擇存在的訓練／參考資料夾")
    image_paths: list[Path] = []
    class_names = list(request.get("class_names") or []) or None
    for root in roots:
        image_paths.extend(list_images(root))
        class_names = class_names or classes_for(root)
    if not image_paths:
        raise ValueError("訓練資料夾內找不到可讀影像")

    score_mode = request["score_mode"]
    target_res = int(request["target_res"]) if score_mode == "patch" else 224
    model_name = request["model_id"]
    cache = dataset_cache_dir(roots[0], f"anomaly_patch_{score_mode}_{model_name}_r{target_res}")
    def report(frac, message):
        if progress:
            progress(max(0, min(75, round(float(frac) * 75))), 100, str(message))

    result = run_pipeline(
        image_paths,
        class_names,
        mode="two_stage",
        score_mode=score_mode,
        sample_n=int(request["sample_n"]),
        model=model_name,
        target_res=target_res,
        confirmed={int(key): value for key, value in request.get("confirmed", {}).items()},
        object_source=request["object_source"],
        cache_dir=cache,
        progress=report,
    )
    if not result.get("records") or result.get("obj_emb") is None:
        raise ValueError("找不到可用的 YOLO 物件或影像；請檢查資料集與物件來源設定")

    obj_emb = np.asarray(result["obj_emb"], dtype=np.float32)
    if progress:
        progress(80, 100, "建立固定參考投影與密度基準")
    normal_set = result.get("normal_set") or []
    good_mask = np.zeros(len(obj_emb), dtype=bool)
    if normal_set:
        good_mask[np.asarray(normal_set, dtype=int)] = True
    projection = fit_projector(obj_emb, good_mask if good_mask.any() else None)
    memory_bank = result.get("bank")
    vectors = getattr(memory_bank, "vectors", None)
    if score_mode == "patch" and vectors is None:
        raise ValueError("patch 模式無法建立 Normal Bank；請確認參考資料含正常物件，或改用物件級分數")
    confirmed = request.get("confirmed") or {}
    fewshot = confirmed_to_fewshot(confirmed, result["records"]) if confirmed else None

    semantic = request["label_semantic"]
    labels = [row.get("label", "") for row in result["records"]]
    unlock = head_unlock_state(label_semantic=semantic, labels=labels, n_min=int(request["n_min"]))
    checkpoint = _DEFAULT_MODELS_DIR / model_name / f"{model_name}.pth"
    if not checkpoint.exists():
        checkpoint = _DEFAULT_MODELS_DIR / f"{model_name}.pth"
    recipe = {
        "model_id": model_name,
        "checkpoint_id": checkpoint_identity(checkpoint if checkpoint.exists() else None),
        "embedding_dim": int(obj_emb.shape[1]),
        "keep_aspect": True,
        "target_res": int(DEFAULT_POLICY["target_res"]),
        "pooling": str(DEFAULT_POLICY["head"]),
        "l2norm": True,
        "scope": "object_crop",
        "pad": float(DEFAULT_POLICY["pad"]),
    }
    head = None
    if semantic == "defect" and unlock["unlocked"]:
        if progress:
            progress(88, 100, "訓練語義允許的瑕疵分類頭")
        eligible = set(unlock["eligible_classes"])
        keep = [index for index, label in enumerate(labels) if label in eligible]
        head = train_head(obj_emb[keep], [labels[index] for index in keep])
        head["_model_name"] = model_name
        head["_eligible"] = sorted(eligible)
        recipe["l2norm"] = bool(head.get("l2norm", True))
        head = attach_provenance(head, **recipe)

    meta = {
        "model": model_name,
        "target_res": target_res,
        "score_mode": score_mode,
        "object_source": request["object_source"],
        "patch_dim": int(vectors.shape[1]) if vectors is not None else None,
        "obj_dim": int(obj_emb.shape[1]),
        "n_objects": len(result["records"]),
        "embedding_recipe": recipe,
        "label_semantic": semantic,
    }
    return {
        "meta": meta,
        "bank_vectors": np.asarray(vectors, dtype=np.float32) if vectors is not None else None,
        "projection": projection,
        "fewshot": fewshot,
        "head": head,
        "built_at": time.time(),
        "summary": {
            "n_images": len(image_paths),
            "n_objects": len(result["records"]),
            "has_normal_bank": vectors is not None,
            "has_head": head is not None,
            "head_status": "能分辨瑕疵種類" if head else "只做異常偵測(不分種類)",
            "eligible_classes": (head or {}).get("_eligible") or [],
        },
    }


def score_targets(request: dict, bundle: dict, *, progress=None) -> dict:
    """Apply a built/loaded bank to target folders without changing the model package."""
    from anomaly_score import MemoryBank
    from anomaly_tool import run_pipeline
    from object_eval import classes_for, dataset_cache_dir, list_images

    roots = [Path(value).expanduser().resolve(strict=True) for value in request["target_folders"]]
    if not roots or any(not root.is_dir() for root in roots):
        raise ValueError("請選擇存在的目標資料夾")
    image_paths: list[Path] = []
    class_names = list(request.get("class_names") or []) or None
    for root in roots:
        image_paths.extend(list_images(root))
        class_names = class_names or classes_for(root)
    if not image_paths:
        raise ValueError("目標資料夾內找不到可用影像")

    meta = bundle["meta"]
    model_name = str(meta["model"])
    score_mode = str(meta["score_mode"])
    target_res = int(meta["target_res"])
    object_source = str(meta["object_source"])
    bank_vectors = bundle.get("bank_vectors")
    external_bank = MemoryBank(np.asarray(bank_vectors, dtype=np.float32)) if bank_vectors is not None else None
    projection = bundle.get("projection") or {}
    external_ref = np.asarray(projection.get("good_obj_emb"), dtype=np.float32)
    if score_mode == "patch" and external_bank is None:
        raise ValueError("patch scoring 模型缺少 Normal Bank，請重新載入有效模型")
    if score_mode == "object" and (external_ref.ndim != 2 or len(external_ref) == 0):
        raise ValueError("object scoring 模型缺少正常特徵參照，請重新載入有效模型")

    cache_dir = dataset_cache_dir(roots[0], f"anomaly_patch_{score_mode}_{model_name}_r{target_res}")
    def report(frac, message):
        if progress:
            progress(max(0, min(92, round(float(frac) * 92))), 100, str(message))

    result = run_pipeline(
        image_paths,
        class_names,
        mode="one_stage",
        score_mode=score_mode,
        sample_n=64,
        model=model_name,
        target_res=target_res,
        object_source=object_source,
        cache_dir=cache_dir,
        external_bank=external_bank,
        external_ref=external_ref if score_mode == "object" else None,
        confirmed={},
        contamination=float(request["contamination"]),
        progress=report,
    )
    if not result.get("records"):
        if object_source == "yolo":
            raise ValueError(
                "找不到 YOLO 標註：目標資料夾需含 images/ 與 labels/（YOLO .txt）。"
                "若沒有標註框，請改選「整張影像」模式。"
            )
        raise ValueError("目標資料夾內找不到符合整張影像模式的影像。")
    from anomaly_classify import classify

    scores = np.asarray(result["scores"], dtype=np.float64)
    if len(scores) != len(result["records"]) or not np.isfinite(scores).all():
        raise ValueError("模型產生的目標分數無效或索引未對齊")
    classification = classify(scores, confirmed={}, contamination=float(request["contamination"]))
    fewshot = bundle.get("fewshot") or []
    aligned, unmatched = ({}, 0)
    if fewshot:
        from anomaly_bank_store import match_fewshot_to_indices
        aligned, unmatched = match_fewshot_to_indices(fewshot, result["records"])
    identity_vectors = bank_vectors if bank_vectors is not None else external_ref
    model_bank_sha256 = hashlib.sha256(
        str(identity_vectors.dtype).encode("ascii")
        + json.dumps(list(identity_vectors.shape), separators=(",", ":")).encode("ascii")
        + np.ascontiguousarray(identity_vectors, dtype=np.float32).tobytes()
    ).hexdigest()
    return {
        "records": result["records"],
        # The React prelabel workflow reuses the same object-level embeddings and
        # model head as Streamlit; keep them worker-private, never in the API view.
        "obj_emb": np.asarray(result["obj_emb"], dtype=np.float32)
        if result.get("obj_emb") is not None else None,
        "scores": scores,
        "labels": np.asarray(classification["labels"], dtype=object),
        "ranking": list(classification["ranking"]),
        "threshold": float(classification["threshold"]),
        "n_bad": int(classification["n_bad"]),
        "auroc": classification.get("auroc"),
        "score_mode": score_mode,
        "model_id": model_name,
        "model_meta": copy.deepcopy(meta),
        "model_bank_sha256": model_bank_sha256,
        "object_source": object_source,
        "target_folders": [str(root) for root in roots],
        "cache_dir": str(cache_dir),
        "bank": result.get("bank"),
        "skipped_count": len(result.get("skipped") or []),
        "fewshot_aligned": {str(key): value for key, value in aligned.items()},
        "fewshot_unmatched": int(unmatched),
        "confirmed": {},
        "contamination": float(request["contamination"]),
        "revision": 1,
        "created_at": time.time(),
    }


def anomaly_run_view(run: dict) -> dict:
    scores = np.asarray(run["scores"], dtype=float)
    labels = sorted({str(row.get("label") or "") for row in run["records"] if row.get("label")})
    verdicts = np.asarray(run["labels"], dtype=object)
    return {
        "run_id": run["run_id"],
        "revision": int(run["revision"]),
        "model_id": run["model_id"],
        "model_bundle_id": run.get("model_bundle_id"),
        "model_meta": {key: value for key, value in run["model_meta"].items()},
        "model_bank_sha256": run["model_bank_sha256"],
        "score_mode": run["score_mode"],
        "object_source": run["object_source"],
        "n_objects": len(run["records"]),
        "n_bad": int(run["n_bad"]),
        "threshold": float(run["threshold"]),
        "auroc": run.get("auroc"),
        "score_min": float(scores.min()) if len(scores) else None,
        "score_max": float(scores.max()) if len(scores) else None,
        "class_labels": labels,
        "verdict_counts": {
            "good": int(np.count_nonzero(verdicts == "good")),
            "bad": int(np.count_nonzero(verdicts == "bad")),
        },
        "confirmed_count": len(run.get("confirmed") or {}),
        "fewshot_aligned": len(run.get("fewshot_aligned") or {}),
        "fewshot_unmatched": int(run.get("fewshot_unmatched") or 0),
        "skipped_count": int(run.get("skipped_count") or 0),
        "created_at": run["created_at"],
    }


def anomaly_run_objects(
    run: dict,
    *,
    offset: int = 0,
    limit: int = 60,
    verdict: str | None = None,
    class_label: str | None = None,
    score_min: float | None = None,
    score_max: float | None = None,
) -> dict:
    offset = max(0, int(offset))
    limit = min(200, max(1, int(limit)))
    scores = np.asarray(run["scores"], dtype=float)
    labels = np.asarray(run["labels"], dtype=object)
    visible = []
    for index in run["ranking"]:
        row = run["records"][index]
        score = float(scores[index])
        if verdict and labels[index] != verdict:
            continue
        if class_label and str(row.get("label") or "") != class_label:
            continue
        if score_min is not None and score < score_min:
            continue
        if score_max is not None and score > score_max:
            continue
        visible.append({
            "index": int(index),
            "filename": Path(str(row["image_path"])).name,
            "label": str(row.get("label") or ""),
            "split": str(row.get("split") or ""),
            "bbox": [float(value) for value in row["bbox"]],
            "obj_index": int(row.get("obj_index") or 0),
            "score": score,
            "verdict": str(labels[index]),
            "confirmed": (run.get("confirmed") or {}).get(str(index), ""),
        })
    return {
        "run_id": run["run_id"],
        "revision": int(run["revision"]),
        "offset": offset,
        "limit": limit,
        "total": len(visible),
        "items": visible[offset:offset + limit],
    }


def reclassify_anomaly_run(
    run: dict,
    *,
    expected_revision: int,
    confirmed: dict[str, str],
    contamination: float,
) -> dict:
    if int(expected_revision) != int(run["revision"]):
        raise ValueError("偵測結果已更新，請重新載入後再套用標註")
    from anomaly_classify import classify

    normalized = {str(key): value for key, value in (run.get("confirmed") or {}).items()}
    for raw_index, value in confirmed.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            raise ValueError("標註物件索引無效") from None
        if index < 0 or index >= len(run["scores"]) or value not in {"good", "bad", ""}:
            raise ValueError("標註內容無效，請重新選擇正常或異常")
        if value:
            normalized[str(index)] = value
        else:
            normalized.pop(str(index), None)
    if not 0.0 <= float(contamination) <= 1.0:
        raise ValueError("污染率必須介於 0 與 1")
    result = classify(run["scores"], confirmed={int(k): v for k, v in normalized.items()}, contamination=contamination)
    run["confirmed"] = normalized
    run["labels"] = np.asarray(result["labels"], dtype=object)
    run["ranking"] = list(result["ranking"])
    run["threshold"] = float(result["threshold"])
    run["n_bad"] = int(result["n_bad"])
    run["auroc"] = result.get("auroc")
    run["contamination"] = float(contamination)
    run["revision"] += 1
    return run


def _view(model_id: str, bundle: dict, *, source: str, saved_at=None) -> dict:
    meta = copy.deepcopy(bundle["meta"])
    meta.setdefault("schema_version", 1)
    summary = bundle.get("summary") or {
        "n_images": 0,
        "n_objects": int(meta.get("n_objects") or 0),
        "has_normal_bank": bundle.get("bank_vectors") is not None,
        "has_head": bundle.get("head") is not None,
        "head_status": "能分辨瑕疵種類" if bundle.get("head") else "只做異常偵測(不分種類)",
        "eligible_classes": (bundle.get("head") or {}).get("_eligible") or [],
    }
    return {
        "model_id": model_id,
        "source": source,
        "saved": saved_at is not None,
        "saved_at": saved_at,
        "built_at": bundle.get("built_at"),
        "meta": {key: value for key, value in meta.items() if key != "embedding_recipe"} | {
            "embedding_recipe": copy.deepcopy(meta.get("embedding_recipe") or {})
        },
        "summary": summary,
    }


def save_bundle(model_id: str, bundle: dict, selected_folders=()) -> dict:
    from anomaly_bank_store import assert_safe_bank_dir, save_bank
    from dino_head import save_head

    if not re.fullmatch(r"am_[0-9a-f]{16}", model_id):
        raise ValueError("模型識別碼格式無效")
    root = cache_root() / "react_anomaly_models"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / model_id
    if destination.exists():
        raise ValueError("此模型版本已保存；請重新建模後保存新的版本")
    assert_safe_bank_dir(destination, selected_folders)
    staged = Path(tempfile.mkdtemp(prefix=".model-", dir=root))
    try:
        meta = copy.deepcopy(bundle["meta"])
        save_bank(
            staged,
            vectors=bundle.get("bank_vectors"),
            projection=bundle.get("projection"),
            meta=meta,
            fewshot=bundle.get("fewshot"),
        )
        head = bundle.get("head")
        if head:
            save_head(staged / "head.joblib", head)
        saved_at = time.time()
        manifest = {
            "schema_version": 1,
            "model_id": model_id,
            "model": meta.get("model"),
            "target_res": meta.get("target_res"),
            "score_mode": meta.get("score_mode"),
            "label_semantic": meta.get("label_semantic", "object"),
            "eligible_classes": (head or {}).get("_eligible") or [],
            "has_head": bool(head),
            "built_at": bundle.get("built_at"),
            "saved_at": saved_at,
        }
        (staged / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(staged, destination)
        return _view(model_id, bundle, source="saved", saved_at=saved_at)
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def list_saved() -> list[dict]:
    root = cache_root() / "react_anomaly_models"
    if not root.exists():
        return []
    root_real = root.resolve()
    items = []
    for folder in root.iterdir():
        if not folder.is_dir() or not re.fullmatch(r"am_[0-9a-f]{16}", folder.name):
            continue
        resolved = folder.resolve()
        if folder.is_symlink() or not resolved.is_relative_to(root_real):
            continue
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file() or not (folder / "meta.json").is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        items.append({
            "model_id": folder.name,
            "source": "saved",
            "saved": True,
            "saved_at": manifest.get("saved_at"),
            "built_at": manifest.get("built_at"),
            "meta": {key: value for key, value in meta.items() if key != "embedding_recipe"} | {
                "embedding_recipe": meta.get("embedding_recipe") or {}
            },
            "summary": {
                "n_images": 0,
                "n_objects": int(meta.get("n_objects") or 0),
                "has_normal_bank": (folder / "bank.npz").is_file(),
                "has_head": bool(manifest.get("has_head")) and (folder / "head.joblib").is_file(),
                "head_status": "能分辨瑕疵種類" if manifest.get("has_head") else "只做異常偵測(不分種類)",
                "eligible_classes": manifest.get("eligible_classes") or [],
            },
        })
    return sorted(items, key=lambda item: float(item.get("saved_at") or 0), reverse=True)


def load_bundle(model_id: str) -> tuple[dict, dict]:
    from anomaly_bank_store import load_bank
    from dino_head import load_head

    if not re.fullmatch(r"am_[0-9a-f]{16}", model_id):
        raise ValueError("模型識別碼格式無效")
    root = (cache_root() / "react_anomaly_models").resolve()
    candidate = root / model_id
    if candidate.is_symlink():
        raise ValueError("模型目錄位置無效")
    folder = candidate.resolve(strict=True)
    if not folder.is_relative_to(root):
        raise ValueError("模型目錄位置無效")
    bank = load_bank(folder)
    meta = bank.get("meta") or {}
    if not meta.get("model") or meta.get("target_res") in (None, ""):
        raise ValueError("模型 meta 缺少 model／target_res，無法安全載入")
    if meta.get("score_mode") not in {"patch", "object"}:
        raise ValueError("模型 meta 缺少有效 score_mode")
    vectors = bank.get("vectors")
    if meta["score_mode"] == "patch" and vectors is None:
        raise ValueError("patch scoring 模型缺少 Normal Bank 向量")
    if vectors is not None and (vectors.ndim != 2 or not len(vectors)):
        raise ValueError("bank.npz 維度無效")
    if vectors is not None and meta.get("patch_dim") not in (None, int(vectors.shape[1])):
        raise ValueError("bank.npz 維度與 meta.patch_dim 不符")
    projection = bank.get("projection")
    if projection is None:
        raise ValueError("模型套件缺少固定參考投影")
    obj_dim = int(meta.get("obj_dim") or 0)
    if obj_dim < 1 or np.asarray(projection.get("mean")).shape != (obj_dim,):
        raise ValueError("固定投影維度與 meta.obj_dim 不符")
    if np.asarray(projection.get("components")).ndim != 2 or np.asarray(projection["components"]).shape[1] != obj_dim:
        raise ValueError("固定投影 components 維度無效")
    if np.asarray(projection.get("good_obj_emb")).ndim != 2 or np.asarray(projection["good_obj_emb"]).shape[1] != obj_dim:
        raise ValueError("固定投影 good embedding 維度無效")
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("模型套件缺少 manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != model_id:
        raise ValueError("manifest 模型識別碼不符")
    for key in ("model", "target_res", "score_mode", "label_semantic"):
        if manifest.get(key) != meta.get(key):
            raise ValueError(f"manifest 與 meta.{key} 不符")
    head_path = folder / "head.joblib"
    has_head_file = head_path.is_file()
    if bool(manifest.get("has_head")) != has_head_file:
        raise ValueError("分類頭檔與 manifest 不一致")
    if has_head_file and meta.get("label_semantic", "object") != "defect":
        raise ValueError("物件類別語義的模型不應含分類頭")
    head = load_head(head_path) if has_head_file else None
    bundle = {
        "meta": meta,
        "bank_vectors": vectors,
        "projection": projection,
        "fewshot": bank.get("fewshot"),
        "head": head,
        "built_at": manifest.get("built_at"),
        "summary": {
            "n_images": 0,
            "n_objects": int(meta.get("n_objects") or 0),
            "has_normal_bank": vectors is not None,
            "has_head": head is not None,
            "head_status": "能分辨瑕疵種類" if head else "只做異常偵測(不分種類)",
            "eligible_classes": (head or {}).get("_eligible") or [],
        },
    }
    return bundle, _view(model_id, bundle, source="loaded", saved_at=manifest.get("saved_at"))
