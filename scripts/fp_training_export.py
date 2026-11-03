"""Explicitly reviewed complete-image YOLO handoff; originals are never modified."""
from __future__ import annotations

import io
import copy
import json
import uuid
import zipfile

from fp_cases import FULL_ROI, _evidence, utc_now, active_rows
from fp_evidence_lineage import region_provenance


def training_rows(case):
    sources = {s["id"]: s for s in case["sources"]}
    rows = [{"id": "item:" + r["id"], "source": sources[r["source_id"]], "roi": r["roi"], "review": r["review"], "evidence_id": r.get("evidence_id")}
            for r in active_rows(case["items"])]
    rows.extend({"id": "candidate:" + r["id"], "source": r["source"], "roi": r["roi"], "review": r["review"], "evidence_id": r.get("evidence_id")}
                for r in active_rows(case["candidates"]))
    return rows


def _pick(record, keys):
    return {key: copy.deepcopy(record[key]) for key in keys if key in record}




def export_training(case, store, *, complete_images, class_assignments, class_names, reviewer, reason):
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(reason, str) or not reason.strip():
        raise ValueError("訓練交接需要確認人員及完整標註依據")
    if not isinstance(class_names, list) or not class_names or len(set(class_names)) != len(class_names) or not all(isinstance(n, str) and n.strip() for n in class_names):
        raise ValueError("需明確提供不重複的訓練瑕疵類別表")
    if not isinstance(class_assignments, dict):
        raise ValueError("類別對應須為 region_id → class_name 的 JSON 物件")
    rows = training_rows(case)
    by_id = {r["id"]: r for r in rows}
    if not set(class_assignments) <= set(by_id) or not set(class_assignments.values()) <= set(class_names):
        raise ValueError("訓練類別對應含未知區域或類別")
    selected = set(complete_images)
    if not selected or not selected <= {r["source"]["sha256"] for r in rows}:
        raise ValueError("請選取已逐圖確認完整標註的影像")
    grouped = {sha: [r for r in rows if r["source"]["sha256"] == sha] for sha in selected}
    # Include other saved-case verdicts as conflict evidence, not automatic training labels.
    other_rows = [r for old in store.list() if old["id"] != case["id"] for r in training_rows(old)]
    plan = []
    for sha, image_rows in grouped.items():
        if any(r["source"]["kind"] == "screenshot" or r["source"].get("original_kind") == "screenshot" for r in image_rows):
            raise ValueError("截圖不可直接作為正式訓練影像，請補入並覆核原圖")
        if any(r["review"]["verdict"] in {"pending", "disputed", "insufficient"} for r in image_rows):
            raise ValueError("此影像仍有待覆核／爭議／資訊不足區域，不可確認完整標註")
        full_normal = [r for r in image_rows if r["review"]["verdict"] == "normal" and r["roi"] == FULL_ROI]
        defects = [r for r in image_rows if r["review"]["verdict"] == "defect"]
        conflicts = [r for r in other_rows if r["source"]["sha256"] == sha]
        if full_normal and (defects or any(r["review"]["verdict"] == "defect" for r in conflicts)):
            raise ValueError("整圖正常裁決與真瑕疵區域矛盾，不能輸出正常負樣本")
        if defects and any(r["review"]["verdict"] == "normal" and r["roi"] == FULL_ROI for r in conflicts):
            raise ValueError("其他案件有整圖正常裁決，請先解決跨案件衝突")
        if not defects and not full_normal:
            raise ValueError("局部正常不足以輸出整圖負樣本；需要明確整圖正常覆核")
        labels, seen = [], {}
        for row in defects:
            label = class_assignments.get(row["id"])
            if label is None:
                raise ValueError("每個真瑕疵區域都須人工指定訓練類別，不能沿用模型預測猜測")
            box_key = tuple(row["roi"])
            if box_key in seen:
                if seen[box_key] != label:
                    raise ValueError("同一區域的人工類別對應矛盾")
                continue
            seen[box_key] = label
            x1, y1, x2, y2 = row["roi"]
            labels.append(f"{class_names.index(label)} {(x1+x2)/2:.9f} {(y1+y2)/2:.9f} {x2-x1:.9f} {y2-y1:.9f}")
        plan.append({"sha256": sha, "source": _pick(image_rows[0]["source"], ("id", "asset", "sha256", "size", "name", "kind", "original_kind")), "labels": labels,
                     "region_ids": [r["id"] for r in image_rows], "negative": bool(full_normal)})
    record = {"id": "training_" + uuid.uuid4().hex[:12], "at": utc_now(), "case_id": case["id"],
              "case_revision": case["revision"], "format": "yolo_detection", "class_names": class_names,
              "complete_images_declared": True, "reviewer": reviewer, "reason": reason,
              "image_sha256": sorted(selected), "n_images": len(plan), "n_boxes": sum(len(p["labels"]) for p in plan),
              "n_negative_images": sum(p["negative"] for p in plan), "class_assignments": class_assignments,
              "items": plan, "round_ids": [b["round_id"] for b in case["batches"] if b["round_id"]],
              "lineage": region_provenance(case, [r for r in rows if r["source"]["sha256"] in selected])}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for row in plan:
            source = row["source"]
            archive.write(store.image_path(source), "images/" + source["asset"])
            archive.writestr("labels/" + source["sha256"] + ".txt", "\n".join(row["labels"]) + ("\n" if row["labels"] else ""))
        archive.writestr("classes.txt", "\n".join(class_names) + "\n")
        archive.writestr("provenance.json", json.dumps(record, ensure_ascii=False, indent=2))
        archive.writestr("README.txt", "已人工確認完整標註的 YOLO 偵測訓練交接資料。\n"
                        "空 labels 檔只來自整張原圖正常覆核；仍須依訓練框架自行建立 train/val/test 切分。\n"
                        "不可把本批回用為獨立改善評估。訓練完成後，回 LV 比較新舊模型。\n")
    payload = output.getvalue()
    record["evidence_id"] = _evidence(case, "training_handoff", "人工完整覆核訓練交接", export_id=record["id"],
            n_images=record["n_images"], n_boxes=record["n_boxes"], n_negative_images=record["n_negative_images"],
            reviewer=reviewer, reason=reason, image_sha256=record["image_sha256"])
    case.setdefault("training_exports", []).append(record)
    return record, payload
