"""Review-only batches for false-positive comparison candidates."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import uuid
import zipfile

from fp_case_store import FalsePositiveCaseStore, VERDICTS, utc_now


def _active(rows: list[dict]) -> list[dict]:
    return [row for row in rows if not row.get("withdrawn")]


def review_counts(rows: list[dict]) -> dict:
    active = _active(rows)
    reviewed = [row for row in active if row.get("review", {}).get("verdict") != "pending"]
    return {
        "total_regions": len(active),
        "reviewed_regions": len(reviewed),
        "pending_regions": len(active) - len(reviewed),
        "normal_regions": sum(row.get("review", {}).get("verdict") == "normal" for row in active),
        "defect_regions": sum(row.get("review", {}).get("verdict") == "defect" for row in active),
        "training_ready_images": 0,
    }


def prepare_batch(case: dict) -> dict:
    candidates = _active(case.get("candidates", []))
    if not candidates:
        raise ValueError("先加入待覆核候選")
    candidate_ids = sorted(row["id"] for row in candidates)
    identity = json.dumps([case["id"], candidate_ids], sort_keys=True, separators=(",", ":"))
    batch_id = "fpb_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    batch = next((row for row in case.setdefault("batches", []) if row.get("id") == batch_id), None)
    if batch is None:
        batch = {
            "id": batch_id,
            "candidate_ids": candidate_ids,
            "round_id": None,
            "at": utc_now(),
            "case_revision": int(case["revision"]) + 1,
            "training_export": False,
            "unique_images": len({row["source"]["sha256"] for row in candidates}),
        }
        case["batches"].append(batch)
    return batch


def link_round(case: dict, batch_id: str) -> str:
    """Create or recover the review-only round and attach its image-count batch."""
    batch = next((row for row in case.get("batches", []) if row.get("id") == batch_id), None)
    if batch is None:
        raise ValueError("找不到覆核批次")
    if batch.get("round_id"):
        return str(batch["round_id"])

    # The stable batch key lets retries recover after a ledger append but before
    # the case snapshot was saved.
    import round_ledger as ledger

    rounds = ledger.load_rounds()
    existing = next(
        (row for row in rounds if row.get("params", {}).get("fp_batch_id") == batch_id),
        None,
    )
    if existing:
        round_id = existing["round_id"]
    else:
        round_id = ledger.start_round(
            "case_fp_review",
            {
                "case_id": case["id"],
                "fp_batch_id": batch_id,
                "review_unit": "region",
                "n_regions": len(batch["candidate_ids"]),
                "training_export": False,
            },
            "誤逮案件",
            "僅送人工覆核；區域覆核不計入已標註影像",
        )
    if not existing or not any(row.get("batch_id") == batch_id for row in existing.get("batches", [])):
        ledger.attach_batch(round_id, batch_id, int(batch["unique_images"]))
    batch["round_id"] = round_id
    return round_id


def batch_candidates(case: dict, batch: dict) -> list[dict]:
    wanted = set(batch.get("candidate_ids", []))
    return [row for row in case.get("candidates", []) if row.get("id") in wanted and not row.get("withdrawn")]


def review_manifest(case: dict, batch_id: str) -> dict:
    batch = next((row for row in case.get("batches", []) if row.get("id") == batch_id), None)
    if batch is None:
        raise ValueError("找不到覆核批次")
    items = []
    for candidate in batch_candidates(case, batch):
        source = candidate["source"]
        items.append({
            "id": candidate["id"],
            "source": {
                key: source[key]
                for key in ("id", "sha256", "name", "kind", "size", "original_kind")
                if key in source
            },
            "roi": copy.deepcopy(candidate["roi"]),
            "reason": candidate.get("reason", ""),
            "reference_category": candidate.get("reference_category", "unknown"),
            "evidence_id": candidate.get("evidence_id"),
            "review": copy.deepcopy(candidate.get("review", {})),
        })
    return {
        "schema_version": 1,
        "purpose": "review_only",
        "case_id": case["id"],
        "batch_id": batch["id"],
        "round_id": batch["round_id"],
        "case_revision": int(case["revision"]),
        "unit": "region",
        "training_export": False,
        "unique_images": batch.get("unique_images", len({item["source"]["sha256"] for item in items})),
        "counts": review_counts(items),
        "items": items,
        "external_model": {
            key: case.get("external", {}).get(key, "")
            for key in ("model_id", "threshold")
        },
        "instructions": "逐區域覆核；未覆核不是正常。保持 case、batch、round、候選 ID、來源 SHA-256 與 ROI 不變。此清單只供人工覆核，不是訓練資料。",
    }


def review_candidate(case: dict, candidate_id: str, verdict: str, reviewer: str, reason: str) -> dict:
    if verdict not in VERDICTS:
        raise ValueError("未知的覆核結果")
    reviewer, reason = str(reviewer).strip(), str(reason).strip()
    if verdict != "pending" and (not reviewer or not reason):
        raise ValueError("非待覆核結果需要填寫覆核者與判斷依據")
    candidate = next((row for row in case.get("candidates", []) if row.get("id") == candidate_id), None)
    if candidate is None:
        raise ValueError("找不到覆核候選")
    if candidate.get("withdrawn"):
        raise ValueError("此候選已撤回，請先恢復再覆核")
    review = {"verdict": verdict, "reviewer": reviewer, "reason": reason, "at": utc_now()}
    candidate["review"] = review
    case.setdefault("evidence", []).append({
        "id": "e_" + uuid.uuid4().hex[:12],
        "role": "human_review",
        "text": "人工逐區域覆核待覆核候選",
        "at": review["at"],
        "candidate_id": candidate_id,
        "verdict": verdict,
        "reviewer": reviewer,
        "reason": reason,
        "roi": copy.deepcopy(candidate["roi"]),
        "source_sha256": candidate["source"]["sha256"],
    })
    return candidate


def apply_review_return(case: dict, batch_id: str, manifest: dict) -> tuple[dict, int]:
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1:
        raise ValueError("回傳檔須為版本 1 的覆核清單物件")
    if (manifest.get("purpose") != "review_only" or manifest.get("training_export") is not False
            or manifest.get("case_id") != case.get("id")):
        raise ValueError("回傳檔不屬於此案件或不是覆核清單")
    batch = next((row for row in case.get("batches", []) if row.get("id") == batch_id), None)
    if batch is None or manifest.get("batch_id") != batch_id or manifest.get("round_id") != batch.get("round_id"):
        raise ValueError("回傳批次／回合不相符")
    case_revision = manifest.get("case_revision")
    if type(case_revision) is not int or case_revision != case.get("revision"):
        raise ValueError("回傳檔版本已過期，請重新下載清單並合併覆核結果")
    updates = manifest.get("items")
    if not isinstance(updates, list) or not updates:
        raise ValueError("回傳檔沒有覆核項目")

    staged = copy.deepcopy(case)
    batch = next(row for row in staged["batches"] if row["id"] == batch_id)
    known = {row["id"]: row for row in staged.get("candidates", [])}
    seen: set[str] = set()
    checked_updates: list[tuple[dict, dict]] = []
    for row in updates:
        if not isinstance(row, dict) or not isinstance(row.get("source"), dict) or not isinstance(row.get("review"), dict):
            raise ValueError("回傳項目缺少來源或覆核欄位")
        candidate_id = row.get("id")
        if not isinstance(candidate_id, str) or candidate_id in seen or candidate_id not in batch["candidate_ids"]:
            raise ValueError("回傳含重複或不屬於批次的候選區域")
        seen.add(candidate_id)
        original = known.get(candidate_id)
        if original is None or original.get("withdrawn"):
            raise ValueError("回傳候選已撤回或不存在，請重新建立覆核批次")
        source = row["source"]
        if source.get("id") != original["source"]["id"] or source.get("sha256") != original["source"]["sha256"]:
            raise ValueError("回傳影像來源身分不相符")
        try:
            roi = [float(value) for value in row.get("roi", [])]
        except (TypeError, ValueError) as exc:
            raise ValueError("回傳 ROI 不正確") from exc
        if roi != original["roi"]:
            raise ValueError("回傳候選 ROI 不相符")
        review = row["review"]
        verdict = review.get("verdict")
        reviewer, reason = review.get("reviewer", ""), review.get("reason", "")
        if not isinstance(reviewer, str) or not isinstance(reason, str) or verdict not in VERDICTS:
            raise ValueError("回傳覆核欄位不正確")
        reviewer, reason = reviewer.strip(), reason.strip()
        if len(reviewer) > 256 or len(reason) > 4000:
            raise ValueError("回傳覆核者或判斷依據超過允許長度")
        if verdict != "pending" and (not reviewer or not reason):
            raise ValueError("非待覆核結果需要填寫覆核者與判斷依據")
        checked_updates.append((original, {
            "verdict": verdict,
            "reviewer": reviewer,
            "reason": reason,
        }))

    applied = 0
    for original, review in checked_updates:
        if review["verdict"] == "pending":
            continue
        review_candidate(staged, original["id"], review["verdict"], review["reviewer"], review["reason"])
        applied += 1
    return staged, applied


def review_zip(store: FalsePositiveCaseStore, case: dict, manifest: dict) -> bytes:
    output = io.BytesIO()
    portable = copy.deepcopy(manifest)
    added: set[str] = set()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        for row in portable["items"]:
            source = next(
                (candidate["source"] for candidate in case.get("candidates", []) if candidate["id"] == row["id"]),
                None,
            )
            if source is None:
                raise ValueError("覆核候選來源不存在")
            image_path = store.image_path(source)
            asset_name = f"assets/{source['sha256']}{image_path.suffix.lower()}"
            row["review_image"] = asset_name
            row["source"]["asset"] = asset_name
            if asset_name not in added:
                bundle.write(image_path, asset_name)
                added.add(asset_name)
        bundle.writestr("review_manifest.json", json.dumps(portable, ensure_ascii=False, indent=2))
        bundle.writestr(
            "README.txt",
            "逐區域人工覆核：修改 review_manifest.json 內 items[].review 的 verdict/reviewer/reason。\n"
            "verdict: pending, normal, defect, disputed, insufficient。請勿修改案件、批次、回合、候選 ID、來源 SHA-256 或 ROI。\n"
            "圖片位於 assets/；ROI 是 normalized xyxy。回傳 JSON 供本機讀回。\n"
            "此包不是訓練資料；局部 normal 不代表整張圖片為正常。\n",
        )
    return output.getvalue()
