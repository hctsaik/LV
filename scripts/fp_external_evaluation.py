"""Fixed-dataset evaluation and explicitly reviewed YOLO handoff for React LV."""
from __future__ import annotations

import copy
import io
import json
import math
import uuid
import zipfile
from pathlib import Path

from PIL import Image

from fp_case_store import checked_roi, digest, utc_now
from fp_external_runtime import Runtime, canonical, iou


FULL_ROI = [0.0, 0.0, 1.0, 1.0]
VERDICTS = {"pending", "normal", "defect", "disputed", "insufficient"}


def _active(rows: list[dict]) -> list[dict]:
    return [row for row in rows if not row.get("withdrawn")]


def case_content_hashes(store) -> set[str]:
    hashes: set[str] = set()
    for case in store.list():
        hashes.update(row["sha256"] for row in case.get("sources", []))
        hashes.update(
            row["source"]["sha256"]
            for row in case.get("candidates", [])
            if row.get("source", {}).get("sha256")
        )
        for search in case.get("searches", []):
            hashes.update(
                row["sha256"]
                for rows in search.get("groups", {}).values()
                for row in rows
                if row.get("sha256")
            )
        for export in case.get("training_exports", []):
            hashes.update(export.get("image_sha256", []))
    return hashes


def validate_evaluation_manifest(
    manifest: dict, store, base_dir: str | Path | None = None
) -> tuple[dict, str]:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("purpose") not in {"independent", "case_regression"}
    ):
        raise ValueError(
            "評估 JSON 需要 schema_version=1 與 purpose=independent 或 case_regression"
        )
    independent = manifest["purpose"] == "independent"
    if independent and (
        manifest.get("independent_declared") is not True
        or not str(manifest.get("split_basis", "")).strip()
    ):
        raise ValueError("獨立評估需要確認資料未參與訓練／選樣／調門檻，並說明切分依據")
    items = manifest.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= 5000:
        raise ValueError("評估清單需要 1–5000 筆已確認影像")

    root = Path(base_dir).expanduser().resolve() if base_dir else None
    known = case_content_hashes(store) if independent else set()
    seen: set[str] = set()
    ids: set[str] = set()
    rows = []
    for row in items:
        if (
            not isinstance(row, dict)
            or row.get("truth") not in {"normal", "defect"}
            or row.get("verified") is not True
        ):
            raise ValueError("每張評估圖片須明確 verified=true 並有 normal／defect 真值")
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
            raise ValueError("評估項目須有唯一且非空的 id")
        ids.add(row["id"])
        if not isinstance(row.get("path"), str) or not row["path"]:
            raise ValueError("評估項目須有影像 path")
        path = Path(row["path"]).expanduser()
        if not path.is_absolute():
            if root is None:
                raise ValueError("相對圖片路徑需要指定基準資料夾")
            path = root / path
        path = path.resolve()
        try:
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise ValueError(f"無法讀取評估圖片：{row['id']}") from exc
        sha = digest(data)
        if row.get("sha256") and row["sha256"] != sha:
            raise ValueError("評估圖片與指定 hash 不符")
        if sha in seen:
            raise ValueError("評估清單含重複影像內容，不能重複計入支持數")
        if sha in known:
            raise ValueError(
                "獨立資料與報案／候選／已搜尋對照重疊；請改用 case_regression 或另備獨立資料"
            )
        seen.add(sha)
        copied = copy.deepcopy(row)
        copied.update(path=str(path), sha256=sha)
        if "boxes" in row:
            if not isinstance(row["boxes"], list) or row.get("annotations_complete") is not True:
                raise ValueError("框層級評估須提供 boxes 並確認 annotations_complete=true")
            for box in row["boxes"]:
                if (
                    not isinstance(box, dict)
                    or not isinstance(box.get("class"), str)
                    or not box["class"]
                ):
                    raise ValueError("評估框需要 class 名稱")
                checked_roi(box.get("box", []))
            if (row["truth"] == "normal" and row["boxes"]) or (
                row["truth"] == "defect" and not row["boxes"]
            ):
                raise ValueError("圖片真值與完整瑕疵框標註矛盾")
        rows.append(copied)

    snapshot = {**copy.deepcopy(manifest), "items": rows}
    fingerprint_data = {
        **snapshot,
        "items": [{key: value for key, value in row.items() if key != "path"} for row in rows],
    }
    fingerprint = digest(canonical(fingerprint_data).encode("utf-8"))
    return snapshot, fingerprint


def evaluate_predictions(rows: list[dict], predictions: list[dict], *, normal_classes: list[str], match_iou: float = 0.5) -> dict:
    if len(rows) != len(predictions):
        raise ValueError("評估真值與預測數量不一致")
    normal = sum(row["truth"] == "normal" for row in rows)
    defect = len(rows) - normal
    false_alarms = sum(
        row["truth"] == "normal" and bool(prediction["alarm"])
        for row, prediction in zip(rows, predictions)
    )
    misses = sum(
        row["truth"] == "defect" and not prediction["alarm"]
        for row, prediction in zip(rows, predictions)
    )
    abstentions = sum(prediction.get("decision") == "abstain" for prediction in predictions)
    per_group: dict[str, dict] = {}
    for row, prediction in zip(rows, predictions):
        group = str(row.get("group_id") or "未提供群組")
        counts = per_group.setdefault(
            group, {"normal": 0, "defect": 0, "false_alarms": 0, "misses": 0}
        )
        counts[row["truth"]] += 1
        counts["false_alarms"] += int(
            row["truth"] == "normal" and prediction["alarm"]
        )
        counts["misses"] += int(row["truth"] == "defect" and not prediction["alarm"])
    result = {
        "normal_images": normal,
        "defect_images": defect,
        "false_alarms": false_alarms,
        "misses": misses,
        "false_alarm_rate": false_alarms / normal if normal else None,
        "miss_rate": misses / defect if defect else None,
        "abstentions": abstentions,
        "groups": per_group,
    }
    complete = all(
        "boxes" in row and row.get("annotations_complete") is True for row in rows
    )
    if complete and all(prediction["task"] == "detection" for prediction in predictions):
        tp = fp = fn = 0
        for row, prediction in zip(rows, predictions):
            unmatched = list(row["boxes"])
            for detection in sorted(
                prediction["detections"], key=lambda item: -item["score"]
            ):
                if detection["class"] in normal_classes:
                    continue
                candidates = [
                    (index, iou(gt["box"], detection["box"]))
                    for index, gt in enumerate(unmatched)
                    if gt["class"] == detection["class"]
                ]
                best = max(candidates, key=lambda pair: pair[1]) if candidates else None
                if best and best[1] >= match_iou:
                    tp += 1
                    unmatched.pop(best[0])
                else:
                    fp += 1
            fn += len(unmatched)
        result["detection"] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "iou": match_iou,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
    else:
        result["detection"] = None
    return result


def improvement_gate(
    before: dict,
    after: dict,
    *,
    purpose: str,
    min_support: int = 10,
    max_miss_increase: float = 0.0,
    min_fp_reduction: float = 0.0,
) -> dict:
    rates = [max_miss_increase, min_fp_reduction]
    if (
        type(min_support) is not int
        or min_support < 1
        or not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and 0 <= value <= 1
            for value in rates
        )
    ):
        raise ValueError("驗收支持數須為正整數，率差門檻須介於 0–1")
    policy = {
        "min_support_per_truth": min_support,
        "max_miss_increase": max_miss_increase,
        "min_fp_reduction": min_fp_reduction,
    }
    if purpose != "independent":
        return {
            "status": "case_regression_only",
            "reasons": ["報案回歸不能代表整體模型改善"],
            "policy": policy,
        }
    if min(
        before["normal_images"],
        before["defect_images"],
        after["normal_images"],
        after["defect_images"],
    ) < min_support:
        return {"status": "insufficient", "reasons": ["正常或瑕疵支持數不足"], "policy": policy}
    reasons = []
    if before["abstentions"] or after["abstentions"]:
        reasons.append("有拒判影像，需另行覆核，不能當作正確判定")
    if after["miss_rate"] - before["miss_rate"] > max_miss_increase + 1e-12:
        reasons.append("漏檢率增加超過容許值")
    if before["false_alarm_rate"] - after["false_alarm_rate"] < min_fp_reduction - 1e-12:
        reasons.append("誤逮率改善未達設定目標")
    if before.get("detection") and after.get("detection"):
        old, new = before["detection"]["recall"], after["detection"]["recall"]
        if old is not None and new is not None and old - new > max_miss_increase + 1e-12:
            reasons.append("框層級召回下降超過容許值")
    return {
        "status": "failed" if reasons else "passed",
        "reasons": reasons or ["此固定評估集符合指定門檻；非自動部署許可"],
        "policy": policy,
    }


def paired_evaluation(
    case: dict,
    store,
    baseline_config: dict,
    candidate_config: dict,
    manifest: dict,
    *,
    base_dir: str | Path | None = None,
    min_support: int = 10,
    max_miss_increase: float = 0.0,
    min_fp_reduction: float = 0.0,
    baseline_profile: dict | None = None,
    candidate_profile: dict | None = None,
) -> dict:
    snapshot, fingerprint = validate_evaluation_manifest(manifest, store, base_dir)
    baseline, candidate = Runtime(baseline_config), Runtime(candidate_config)
    if (
        baseline.config["class_names"] != candidate.config["class_names"]
        or baseline.config["normal_classes"] != candidate.config["normal_classes"]
    ):
        raise ValueError("前後模型的類別順序與正常類別定義須一致")
    if (baseline.config["output_format"] == "classification") != (
        candidate.config["output_format"] == "classification"
    ):
        raise ValueError("不能直接比較分類與偵測兩種不同任務")
    for row in snapshot["items"]:
        if any(
            box["class"] not in baseline.config["class_names"]
            or box["class"] in baseline.config["normal_classes"]
            for box in row.get("boxes", [])
        ):
            raise ValueError("完整瑕疵框的類別不在模型瑕疵類別表中")
    improvement_gate(
        {"normal_images": 0, "defect_images": 0},
        {"normal_images": 0, "defect_images": 0},
        purpose="independent",
        min_support=min_support,
        max_miss_increase=max_miss_increase,
        min_fp_reduction=min_fp_reduction,
    )

    output_rows: list[list[dict]] = [[], []]
    details = []
    for row in snapshot["items"]:
        data = Path(row["path"]).read_bytes()
        if digest(data) != row["sha256"]:
            raise ValueError("評估圖片在執行期間變更；本次結果不發布")
        runs = [runtime.infer(data) for runtime in (baseline, candidate)]
        for predictions, run in zip(output_rows, runs):
            predictions.append(run["prediction"])
        details.append(
            {
                "id": row["id"],
                "sha256": row["sha256"],
                "truth": row["truth"],
                "group_id": row.get("group_id"),
                "baseline": {
                    key: value
                    for key, value in runs[0]["prediction"].items()
                    if key != "candidates"
                },
                "candidate": {
                    key: value
                    for key, value in runs[1]["prediction"].items()
                    if key != "candidates"
                },
            }
        )
    if any(digest(Path(row["path"]).read_bytes()) != row["sha256"] for row in snapshot["items"]):
        raise ValueError("評估資料在執行期間變更，請重新評估")
    scores = [
        evaluate_predictions(
            snapshot["items"],
            predictions,
            normal_classes=baseline.config["normal_classes"],
        )
        for predictions in output_rows
    ]
    gate = improvement_gate(
        *scores,
        purpose=snapshot["purpose"],
        min_support=min_support,
        max_miss_increase=max_miss_increase,
        min_fp_reduction=min_fp_reduction,
    )
    safe_manifest = copy.deepcopy(snapshot)
    for row in safe_manifest["items"]:
        row.pop("path", None)
    result = {
        "id": "evaluation_" + uuid.uuid4().hex[:12],
        "at": utc_now(),
        "dataset_fingerprint": fingerprint,
        "manifest": safe_manifest,
        "baseline_profile": copy.deepcopy(baseline_profile or {}),
        "candidate_profile": copy.deepcopy(candidate_profile or {}),
        "baseline_identity": baseline.identity,
        "candidate_identity": candidate.identity,
        "baseline": scores[0],
        "candidate": scores[1],
        "gate": gate,
        "per_image": details,
        "limitation": (
            "獨立性包含使用者切分聲明與本工作區內容重疊檢查；"
            "未取得外部訓練全集，不能自動證明完全無洩漏。"
        ),
    }
    evidence_id = "e_" + uuid.uuid4().hex[:12]
    result["evidence_id"] = evidence_id
    case.setdefault("evaluations", []).append(result)
    case.setdefault("evidence", []).append(
        {
            "id": evidence_id,
            "role": "external_evaluation",
            "text": "固定資料集前後模型評估",
            "at": result["at"],
            "evaluation_id": result["id"],
            "dataset_fingerprint": fingerprint,
            "gate_status": gate["status"],
            "baseline_contract": baseline.identity["contract_id"],
            "candidate_contract": candidate.identity["contract_id"],
        }
    )
    return result


def training_rows(case: dict) -> list[dict]:
    sources = {source["id"]: source for source in case.get("sources", [])}
    rows = [
        {
            "id": "item:" + row["id"],
            "source": sources[row["source_id"]],
            "roi": row["roi"],
            "review": row["review"],
            "evidence_id": row.get("evidence_id"),
        }
        for row in _active(case.get("items", []))
        if row.get("source_id") in sources
    ]
    rows.extend(
        {
            "id": "candidate:" + row["id"],
            "source": row["source"],
            "roi": row["roi"],
            "review": row.get("review", {"verdict": "pending"}),
            "evidence_id": row.get("evidence_id"),
        }
        for row in _active(case.get("candidates", []))
        if row.get("source", {}).get("sha256")
    )
    return rows


def export_training(
    case: dict,
    store,
    *,
    complete_images: list[str],
    class_assignments: dict[str, str],
    class_names: list[str],
    reviewer: str,
    reason: str,
) -> tuple[dict, bytes]:
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(reason, str) or not reason.strip():
        raise ValueError("訓練交接需要確認人員及完整標註依據")
    if (
        not isinstance(class_names, list)
        or not class_names
        or len(set(class_names)) != len(class_names)
        or not all(isinstance(name, str) and name.strip() for name in class_names)
    ):
        raise ValueError("需明確提供不重複的訓練瑕疵類別表")
    if not isinstance(class_assignments, dict):
        raise ValueError("類別對應須為 region_id → class_name 的 JSON 物件")
    rows = training_rows(case)
    by_id = {row["id"]: row for row in rows}
    if not set(class_assignments) <= set(by_id) or not set(class_assignments.values()) <= set(class_names):
        raise ValueError("訓練類別對應含未知區域或類別")
    selected = set(complete_images)
    available = {row["source"]["sha256"] for row in rows}
    if not selected or not selected <= available:
        raise ValueError("請選取已逐圖確認完整標註的影像")
    grouped = {
        sha: [row for row in rows if row["source"]["sha256"] == sha]
        for sha in selected
    }
    other_rows = [
        row
        for old_case in store.list()
        if old_case["id"] != case["id"]
        for row in training_rows(old_case)
    ]
    plan = []
    for sha, image_rows in grouped.items():
        if any(
            row["source"].get("kind") == "screenshot"
            or row["source"].get("original_kind") == "screenshot"
            for row in image_rows
        ):
            raise ValueError("截圖不可直接作為正式訓練影像，請補入並覆核原圖")
        if any(row["review"]["verdict"] in {"pending", "disputed", "insufficient"} for row in image_rows):
            raise ValueError("此影像仍有待覆核／爭議／資訊不足區域，不可確認完整標註")
        full_normal = [
            row
            for row in image_rows
            if row["review"]["verdict"] == "normal" and row["roi"] == FULL_ROI
        ]
        defects = [row for row in image_rows if row["review"]["verdict"] == "defect"]
        conflicts = [row for row in other_rows if row["source"]["sha256"] == sha]
        if full_normal and (
            defects or any(row["review"]["verdict"] == "defect" for row in conflicts)
        ):
            raise ValueError("整圖正常裁決與真瑕疵區域矛盾，不能輸出正常負樣本")
        if defects and any(
            row["review"]["verdict"] == "normal" and row["roi"] == FULL_ROI
            for row in conflicts
        ):
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
            labels.append(
                f"{class_names.index(label)} {(x1 + x2) / 2:.9f} "
                f"{(y1 + y2) / 2:.9f} {x2 - x1:.9f} {y2 - y1:.9f}"
            )
        source = image_rows[0]["source"]
        source_path = store.image_path(source)
        extension = source_path.suffix.lower()
        if extension not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
            extension = ".png"
        plan.append(
            {
                "sha256": sha,
                "source": {
                    key: copy.deepcopy(source[key])
                    for key in ("id", "sha256", "size", "name", "kind")
                    if key in source
                },
                "image_bytes": source_path.read_bytes(),
                "image_extension": extension,
                "labels": labels,
                "region_ids": [row["id"] for row in image_rows],
                "regions": [
                    {
                        "region_id": row["id"],
                        "source_sha256": sha,
                        "roi": copy.deepcopy(row["roi"]),
                        "review": copy.deepcopy(row["review"]),
                        "evidence_id": row.get("evidence_id"),
                        "association_semantics": "historical_selection",
                    }
                    for row in image_rows
                ],
                "negative": bool(full_normal),
            }
        )

    export_id = "training_" + uuid.uuid4().hex[:12]
    record = {
        "id": export_id,
        "at": utc_now(),
        "case_id": case["id"],
        "case_revision": case["revision"],
        "format": "yolo_detection",
        "class_names": list(class_names),
        "complete_images_declared": True,
        "reviewer": reviewer.strip(),
        "reason": reason.strip(),
        "image_sha256": sorted(selected),
        "n_images": len(plan),
        "n_boxes": sum(len(row["labels"]) for row in plan),
        "n_negative_images": sum(row["negative"] for row in plan),
        "zip_file": export_id + ".zip",
        "provenance": [
            region
            for image in plan
            for region in image["regions"]
        ],
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        names = "\n".join(
            f"  {index}: {json.dumps(name, ensure_ascii=False)}"
            for index, name in enumerate(class_names)
        )
        archive.writestr(
            "data.yaml",
            f"path: .\ntrain: images\nval: images\nnames:\n{names}\n",
        )
        archive.writestr(
            "README.txt",
            "已人工確認完整標註的 YOLO 偵測訓練交接資料。\n"
            "空 labels 檔只來自整張原圖正常覆核；仍須依訓練框架自行建立 train/val/test 切分。\n"
            "不可把本批回用為獨立改善評估。訓練完成後，回 LV 比較新舊模型。\n",
        )
        archive.writestr(
            "provenance.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "export": {key: value for key, value in record.items() if key != "provenance"},
                    "regions": record["provenance"],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        for row in plan:
            stem = row["sha256"]
            archive.writestr(
                f"images/{stem}{row['image_extension']}", row["image_bytes"]
            )
            archive.writestr(f"labels/{stem}.txt", "\n".join(row["labels"]))
    payload = output.getvalue()
    evidence_id = "e_" + uuid.uuid4().hex[:12]
    record["evidence_id"] = evidence_id
    case.setdefault("training_exports", []).append(record)
    case.setdefault("evidence", []).append(
        {
            "id": evidence_id,
            "role": "training_handoff",
            "text": "人工完整覆核訓練交接",
            "at": record["at"],
            "export_id": export_id,
            "n_images": record["n_images"],
            "n_boxes": record["n_boxes"],
            "n_negative_images": record["n_negative_images"],
            "reviewer": reviewer.strip(),
            "reason": reason.strip(),
            "image_sha256": record["image_sha256"],
        }
    )
    return record, payload
