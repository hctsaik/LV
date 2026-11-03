"""Case-bound external inference, controlled experiments and paired evaluation."""
from __future__ import annotations

import copy
import io
import json
import math
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageDraw

from fp_cases import _evidence, digest, roi_checked, utc_now
from fp_external_runtime import Runtime, canonical, compare_reference, iou


def _runtime(config):
    if Path(config.get("model_path", "")).suffix.lower() == ".pt":
        from explain_service import runtime_for
        return runtime_for(config, trusted=True)
    return Runtime(config)


def context_key(case, item):
    source = next(s for s in case["sources"] if s["id"] == item["source_id"])
    return digest(canonical({"source_sha256": source["sha256"], "item_id": item["id"],
            "roi": item["roi"], "label": item["label"],
            "reported": {k: case["external"].get(k) for k in ["model_id", "threshold", "raw_output"]}}).encode())


def current_run(case, run):
    if run.get("source_context_id"):
        source = next((s for s in case["sources"] if s["id"] == run.get("source_id")), None)
        return source is not None and source["sha256"] == run["source_context_id"]
    item = next((i for i in case["items"] if i["id"] == run.get("item_id")), None)
    return item is not None and not item.get("withdrawn") and run.get("context_id") == context_key(case, item)


def runtime_root(store, case):
    return store.root / "external" / case["id"]


def run_case(case, item_id, store, config, reference=None, *, score_atol=.001, box_atol=.001):
    item = next(i for i in case["items"] if i["id"] == item_id)
    source = next(s for s in case["sources"] if s["id"] == item["source_id"])
    runtime = _runtime(config)
    run = runtime.infer(store.image_path(source).read_bytes(), runtime_root(store, case))
    parity = compare_reference(run, reference, source["kind"], score_atol=score_atol, box_atol=box_atol)
    run.update(item_id=item_id, source_id=source["id"], context_id=context_key(case, item), parity=parity,
               config=runtime.config)
    if reference is not None:
        run["historical_reference"] = copy.deepcopy(reference)
    details = {"run_id": run["id"], "item_id": item_id, "contract_id": run["contract_id"],
               "model_version": config.get("model_version", ""), "image_sha256": run["image_sha256"],
               "artifact_dir": run["artifact_dir"], "parity": parity,
               "prediction": {k: v for k, v in run["prediction"].items() if k != "candidates"}}
    run["evidence_id"] = _evidence(case, "external_inference", "外部模型本機推論與歷史比對", **details)
    case.setdefault("external_runs", []).append(run)
    # Publish only after inference/validation completed; UI save protects the whole case revision.
    (Path(run["artifact_dir"]) / "case_run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    return run


def target_observation(prediction, label, roi, match_iou=.1):
    if prediction["task"] == "classification":
        score = prediction["scores"].get(label)
        return {"status": "matched" if score is not None else "unavailable", "score": score,
                "final_present": prediction["class"] == label and prediction["decision"] != "abstain",
                "raw_available": True}
    def match(rows):
        options = [r for r in rows if r["class"] == label and iou(r["box"], roi) >= match_iou]
        return max(options, key=lambda r: (iou(r["box"], roi), r["score"])) if options else None
    final = match(prediction["detections"])
    raw_available = prediction["raw_candidates_available"]
    raw = match(prediction.get("candidates", [])) if raw_available else None
    chosen = raw or final
    return {"status": "matched" if chosen else "unmatched", "score": chosen["score"] if chosen else None,
            "matched_box": chosen["box"] if chosen else None, "final_present": final is not None,
            "raw_available": raw_available, "candidate_json_truncated": prediction.get("candidate_json_truncated", False)}


def _png(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def run_experiment(case, baseline_id, store, *, target_class, intervention_roi,
                   mode="occlusion", factor=.8, fill=114, controls=3, seed=42,
                   target_preserved=False, target_item_id=None, progress=None):
    baseline = next(r for r in case.get("external_runs", []) if r["id"] == baseline_id)
    if not current_run(case, baseline):
        raise ValueError("來源／區域或報案資訊已變更，請先重新執行基準推論")
    selected_id = target_item_id or baseline.get('item_id')
    item = next((i for i in case['items'] if i['id'] == selected_id and not i.get('withdrawn')), None)
    if item is None:
        raise ValueError('請先選擇此基準影像的有效目標區域；無檢出不會建立虛構目標')
    baseline_source = baseline.get('source_id')
    if baseline_source is None and baseline.get('item_id'):
        old_item = next((i for i in case['items'] if i['id'] == baseline['item_id']), None)
        baseline_source = old_item['source_id'] if old_item else None
    if item['source_id'] != baseline_source:
        raise ValueError('所選區域不屬於此基準影像')
    binding = item.get('external_target')
    if binding and binding.get('run_id') != baseline_id:
        raise ValueError('所選預測框屬於另一筆推論，請選擇對應的基準紀錄')
    if binding and (binding.get('class') != target_class or item['label'] != target_class):
        raise ValueError('固定目標類別必須與所選預測框一致')
    roi = roi_checked(intervention_roi)
    if mode not in {"occlusion", "brightness"} or not math.isfinite(factor) or not .1 <= factor <= 2:
        raise ValueError("實驗模式或亮度係數不正確")
    if type(controls) is not int or not 1 <= controls <= 10 or type(fill) is not int or not 0 <= fill <= 255:
        raise ValueError("對照數須為 1–10，遮擋填充值須為 0–255 整數")
    runtime = _runtime(baseline["config"])
    if runtime.identity["contract_id"] != baseline["contract_id"]:
        raise ValueError("模型內容已變更，請重新執行基準推論")
    if target_class not in runtime.config["class_names"]:
        raise ValueError("目標類別不在模型類別表中")
    source = next(s for s in case["sources"] if s["id"] == item["source_id"])
    source_data = store.image_path(source).read_bytes()
    with Image.open(io.BytesIO(source_data)) as image:
        original = image.convert("RGB")
    original_target = target_observation(baseline["prediction"], target_class, item["roi"])
    if binding:
        exact = [d for d in baseline['prediction'].get('detections',[]) if d.get('output_index') == binding.get('output_index') and d['class'] == target_class and d['box'] == item['roi']]
        if len(exact) != 1:
            raise ValueError('所選框身分或座標已改變，請重新選框')
        original_target.update(score=exact[0]['score'], matched_box=exact[0]['box'], selected_output_index=exact[0]['output_index'])
    if original_target["score"] is None:
        raise ValueError("基準推論找不到同類別且位置相符的目標，請核對爭議區域／類別")
    # Confirm raw input determinism before comparing transformations against an older result.
    repeated = runtime.infer(source_data)
    if repeated["input_sha256"] != baseline["input_sha256"] or repeated["output_sha256"] != baseline["output_sha256"]:
        raise ValueError("同一來源的基準輸出未重現，請先排查輸入或非決定性推論")
    w, h = original.size
    rng = np.random.default_rng(seed)
    definitions = [("selected", roi, factor)]
    if mode == "occlusion":
        # Fixed-size pixel controls: equal area, distinct locations, same fill.
        rw = max(1, round((roi[2]-roi[0])*w)); rh = max(1, round((roi[3]-roi[1])*h))
        used = set()
        for _ in range(controls*100):
            x = int(rng.integers(0, max(1, w-rw+1))); y = int(rng.integers(0, max(1, h-rh+1)))
            control_roi = [x/w, y/h, (x+rw)/w, (y+rh)/h]
            if (x, y) in used or iou(control_roi, roi) > .8:
                continue
            used.add((x, y))
            definitions.append(("control", control_roi, factor))
            if len(used) == controls:
                break
    else:
        definitions.append(("control", roi, 1.0))
    variants = []
    for index, (role, area, brightness) in enumerate(definitions):
        if progress:
            progress(index,len(definitions))
        modified = original.copy()
        left, top = int(area[0]*w), int(area[1]*h)
        if mode == "occlusion":
            right, bottom = min(w, left+rw), min(h, top+rh)
            actual_roi = [left/w, top/h, right/w, bottom/h]
            ImageDraw.Draw(modified).rectangle([left, top, right-1, bottom-1], fill=(fill,)*3)
        else:
            right, bottom = max(left+1, int(area[2]*w)), max(top+1, int(area[3]*h))
            actual_roi = [left/w, top/h, right/w, bottom/h]
            modified.paste(ImageEnhance.Brightness(modified.crop((left, top, right, bottom))).enhance(brightness), (left, top))
        result = runtime.infer(_png(modified), runtime_root(store, case))
        observed = target_observation(result["prediction"], target_class, item["roi"])
        # A score is still shown as an observation, but no effect claim for hidden/unmatched targets.
        status = "single_image_sensitivity" if target_preserved and observed["score"] is not None else "inconclusive"
        delta = original_target["score"]-observed["score"] if status == "single_image_sensitivity" else None
        variants.append({"role": role, "intervention_roi": actual_roi, "factor": brightness, "run": result,
                         "target": observed, "score_drop": delta, "status": status})
    if progress:
        progress(len(definitions),len(definitions))
    experiment = {"id": "experiment_" + uuid.uuid4().hex[:12], "at": utc_now(), "baseline_id": baseline_id,
                  "context_id": context_key(case,item), "item_id":item['id'], 'source_id':source['id'],
                  "contract_id": baseline["contract_id"],
                  "target_class": target_class, "target_roi": item["roi"], "mode": mode, "seed": seed,
                  "fill": fill, "target_preserved_declared": target_preserved, "baseline_target": original_target,
                  "requested_controls": controls if mode == "occlusion" else 1,
                  "available_controls": len(variants)-1, "variants": variants,
                  "limitation": "僅為此圖片在指定修改下的敏感性證據；未證明根因，框消失不等於原始分數為零。"}
    experiment["evidence_id"] = _evidence(case, "external_experiment", "外部模型對照實驗（單圖敏感性）",
        experiment_id=experiment["id"], baseline_id=baseline_id, target_class=target_class, mode=mode,
        target_preserved_declared=target_preserved, baseline_target=original_target,
        variants=[{k: v for k, v in r.items() if k != "run"} | {"run_id": r["run"]["id"], "artifact_dir": r["run"]["artifact_dir"]} for r in variants],
        limitation=experiment["limitation"])
    case.setdefault("experiments", []).append(experiment)
    return experiment


def case_content_hashes(store):
    hashes = set()
    for case in store.list():
        hashes.update(s["sha256"] for s in case["sources"])
        hashes.update(c["source"]["sha256"] for c in case["candidates"])
        for search in case.get("searches", []):
            hashes.update(r["sha256"] for rows in search["groups"].values() for r in rows)
        for export in case.get("training_exports", []):
            hashes.update(export.get("image_sha256", []))
    return hashes


def validate_evaluation_manifest(manifest, store, base_dir=None):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("purpose") not in {"independent", "case_regression"}:
        raise ValueError("評估 JSON 需要 schema_version=1 與 purpose=independent 或 case_regression")
    independent = manifest["purpose"] == "independent"
    if independent and (manifest.get("independent_declared") is not True or not str(manifest.get("split_basis", "")).strip()):
        raise ValueError("獨立評估需要確認資料未參與訓練／選樣／調門檻，並說明切分依據")
    items = manifest.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= 5000:
        raise ValueError("評估清單需要 1–5000 筆已確認影像")
    known = case_content_hashes(store) if independent else set()
    seen, ids, rows = set(), set(), []
    for row in items:
        if not isinstance(row, dict) or row.get("truth") not in {"normal", "defect"} or row.get("verified") is not True:
            raise ValueError("每張評估圖片須明確 verified=true 並有 normal／defect 真值")
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
            raise ValueError("評估項目須有唯一且非空的 id")
        ids.add(row["id"])
        if not isinstance(row.get("path"), str) or not row["path"]:
            raise ValueError("評估項目須有影像 path")
        path = Path(row["path"])
        if not path.is_absolute():
            if base_dir is None:
                raise ValueError("相對圖片路徑需要指定基準資料夾")
            path = Path(base_dir) / path
        path = path.resolve()
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        sha = digest(data)
        if row.get("sha256") and row["sha256"] != sha:
            raise ValueError("評估圖片與指定 hash 不符")
        if sha in seen:
            raise ValueError("評估清單含重複影像內容，不能重複計入支持數")
        if sha in known:
            raise ValueError("獨立資料與報案／候選／已搜尋對照重疊；請改用 case_regression 或另備獨立資料")
        seen.add(sha)
        copied = copy.deepcopy(row)
        copied.update(path=str(path), sha256=sha)
        if "boxes" in row:
            if not isinstance(row["boxes"], list) or row.get("annotations_complete") is not True:
                raise ValueError("框層級評估須提供 boxes 並確認 annotations_complete=true")
            for box in row["boxes"]:
                if not isinstance(box, dict) or not isinstance(box.get("class"), str) or not box["class"]:
                    raise ValueError("評估框需要 class 名稱")
                roi_checked(box.get("box", []))
            if (row["truth"] == "normal" and row["boxes"]) or (row["truth"] == "defect" and not row["boxes"]):
                raise ValueError("圖片真值與完整瑕疵框標註矛盾")
        rows.append(copied)
    snapshot = {**copy.deepcopy(manifest), "items": rows}
    fingerprint = digest(canonical({**snapshot, "items": [{k: v for k, v in r.items() if k != "path"} for r in rows]}).encode())
    return snapshot, fingerprint


def evaluate_predictions(rows, predictions, *, normal_classes, match_iou=.5):
    if len(rows) != len(predictions):
        raise ValueError("評估真值與預測數量不一致")
    normal = sum(r["truth"] == "normal" for r in rows)
    defect = len(rows)-normal
    false_alarms = sum(r["truth"] == "normal" and bool(p["alarm"]) for r, p in zip(rows, predictions))
    misses = sum(r["truth"] == "defect" and not p["alarm"] for r, p in zip(rows, predictions))
    abstentions = sum(p.get("decision") == "abstain" for p in predictions)
    per_group = {}
    for row, pred in zip(rows, predictions):
        group = str(row.get("group_id") or "未提供群組")
        counts = per_group.setdefault(group, {"normal": 0, "defect": 0, "false_alarms": 0, "misses": 0})
        counts[row["truth"]] += 1
        counts["false_alarms"] += int(row["truth"] == "normal" and pred["alarm"])
        counts["misses"] += int(row["truth"] == "defect" and not pred["alarm"])
    result = {"normal_images": normal, "defect_images": defect, "false_alarms": false_alarms, "misses": misses,
              "false_alarm_rate": false_alarms/normal if normal else None,
              "miss_rate": misses/defect if defect else None, "abstentions": abstentions, "groups": per_group}
    complete = all("boxes" in r and r.get("annotations_complete") is True for r in rows)
    if complete and all(p["task"] == "detection" for p in predictions):
        tp = fp = fn = 0
        for row, pred in zip(rows, predictions):
            unmatched = list(row["boxes"])
            for det in sorted(pred["detections"], key=lambda d: -d["score"]):
                if det["class"] in normal_classes:
                    continue
                candidates = [(j, iou(gt["box"], det["box"])) for j, gt in enumerate(unmatched) if gt["class"] == det["class"]]
                best = max(candidates, key=lambda pair: pair[1]) if candidates else None
                if best and best[1] >= match_iou:
                    tp += 1; unmatched.pop(best[0])
                else:
                    fp += 1
            fn += len(unmatched)
        result["detection"] = {"tp": tp, "fp": fp, "fn": fn, "iou": match_iou,
                               "precision": tp/(tp+fp) if tp+fp else None, "recall": tp/(tp+fn) if tp+fn else None}
    else:
        result["detection"] = None
    return result


def improvement_gate(before, after, *, purpose, min_support=10, max_miss_increase=0.0, min_fp_reduction=0.0):
    if type(min_support) is not int or min_support < 1 or not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in [max_miss_increase, min_fp_reduction]):
        raise ValueError("驗收支持數須為正整數，率差門檻須介於 0–1")
    policy = {"min_support_per_truth": min_support, "max_miss_increase": max_miss_increase, "min_fp_reduction": min_fp_reduction}
    if purpose != "independent":
        return {"status": "case_regression_only", "reasons": ["報案回歸不能代表整體模型改善"], "policy": policy}
    if min(before["normal_images"], before["defect_images"], after["normal_images"], after["defect_images"]) < min_support:
        return {"status": "insufficient", "reasons": ["正常或瑕疵支持數不足"], "policy": policy}
    reasons = []
    if before["abstentions"] or after["abstentions"]:
        reasons.append("有拒判影像，需另行覆核，不能當作正確判定")
    if after["miss_rate"]-before["miss_rate"] > max_miss_increase+1e-12:
        reasons.append("漏檢率增加超過容許值")
    if before["false_alarm_rate"]-after["false_alarm_rate"] < min_fp_reduction-1e-12:
        reasons.append("誤逮率改善未達設定目標")
    if before.get("detection") and after.get("detection"):
        old, new = before["detection"]["recall"], after["detection"]["recall"]
        if old is not None and new is not None and old-new > max_miss_increase+1e-12:
            reasons.append("框層級召回下降超過容許值")
    return {"status": "failed" if reasons else "passed", "reasons": reasons or ["此固定評估集符合指定門檻；非自動部署許可"], "policy": policy}


def paired_evaluation(case, store, baseline_config, candidate_config, manifest, *, base_dir=None,
                      min_support=10, max_miss_increase=0.0, min_fp_reduction=0.0, progress=None):
    snapshot, fingerprint = validate_evaluation_manifest(manifest, store, base_dir)
    baseline, candidate = _runtime(baseline_config), _runtime(candidate_config)
    if baseline.config["class_names"] != candidate.config["class_names"] or baseline.config["normal_classes"] != candidate.config["normal_classes"]:
        raise ValueError("前後模型的類別順序與正常類別定義須一致")
    if (baseline.config["output_format"] == "classification") != (candidate.config["output_format"] == "classification"):
        raise ValueError("不能直接比較分類與偵測兩種不同任務")
    for row in snapshot["items"]:
        if any(b["class"] not in baseline.config["class_names"] or b["class"] in baseline.config["normal_classes"] for b in row.get("boxes", [])):
            raise ValueError("完整瑕疵框的類別不在模型瑕疵類別表中")
    # Validate acceptance policy before any costly inference.
    improvement_gate({"normal_images": 0, "defect_images": 0}, {"normal_images": 0, "defect_images": 0},
                     purpose="independent", min_support=min_support, max_miss_increase=max_miss_increase,
                     min_fp_reduction=min_fp_reduction)
    outputs = [[], []]
    details = []
    for index, row in enumerate(snapshot["items"]):
        data = Path(row["path"]).read_bytes()
        if digest(data) != row["sha256"]:
            raise ValueError("評估圖片在執行期間變更；本次結果不發布")
        runs = [runtime.infer(data) for runtime in [baseline, candidate]]
        for predictions, run in zip(outputs, runs):
            predictions.append(run["prediction"])
        details.append({"id": row["id"], "sha256": row["sha256"], "truth": row["truth"],
                        "baseline": {k: v for k, v in runs[0]["prediction"].items() if k != "candidates"},
                        "candidate": {k: v for k, v in runs[1]["prediction"].items() if k != "candidates"}})
        if progress:
            progress(index+1, len(snapshot["items"]))
    # A final recheck prevents silently accepting a dataset that changed after an early row ran.
    if any(digest(Path(r["path"]).read_bytes()) != r["sha256"] for r in snapshot["items"]):
        raise ValueError("評估資料在執行期間變更，請重新評估")
    scores = [evaluate_predictions(snapshot["items"], preds, normal_classes=baseline.config["normal_classes"]) for preds in outputs]
    gate = improvement_gate(*scores, purpose=snapshot["purpose"], min_support=min_support,
                            max_miss_increase=max_miss_increase, min_fp_reduction=min_fp_reduction)
    result = {"id": "evaluation_" + uuid.uuid4().hex[:12], "at": utc_now(), "dataset_fingerprint": fingerprint,
              "manifest": snapshot, "baseline_identity": baseline.identity, "candidate_identity": candidate.identity,
              "baseline": scores[0], "candidate": scores[1], "gate": gate, "per_image": details,
              "limitation": "獨立性包含使用者切分聲明與本工作區內容重疊檢查；未取得外部訓練全集，不能自動證明完全無洩漏。"}
    result["evidence_id"] = _evidence(case, "external_evaluation", "固定資料集前後模型評估", evaluation_id=result["id"],
          dataset_fingerprint=fingerprint, baseline=scores[0], candidate=scores[1], gate=gate,
          baseline_contract=baseline.identity["contract_id"], candidate_contract=candidate.identity["contract_id"])
    case.setdefault("evaluations", []).append(result)
    # Outcome is scoped to this evaluation, not a blanket case/model root-cause status.
    return result
