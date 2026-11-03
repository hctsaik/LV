"""Durable jobs joining uploaded models, source-first inference and case evidence."""
from __future__ import annotations

import ast
import copy
import io
import json
import math
from pathlib import Path

from fp_cases import CaseStore, digest, _evidence, utc_now, add_item, roi_checked
from fp_external_runtime import DEFAULT_CONFIG, canonical, compare_reference, iou

HANDLERS = {"inspect": "fp_attribution:inspect_task", "infer": "fp_attribution:infer_task",
            "gradcam": "fp_attribution:gradcam_task", "evaluate": "fp_attribution:evaluate_task",
            "comparison": "fp_attribution:comparison_task", "batch_gradcam": "fp_attribution:batch_gradcam_task",
            "experiment": "fp_attribution:experiment_task"}


def manager_for(store):
    from investigation_jobs import JobManager
    return JobManager.get(store.root, HANDLERS, max_workers=1, timeout_seconds=1800,
                          runtime_selector=job_runtime, payload_normalizer=prepare_job_payload)


def prepare_job_payload(task, payload):
    payload = copy.deepcopy(payload)
    if task == "gradcam":
        for key, default in {"protocol_version": 2, "fill_methods": ["constant", "blur"],
                             "max_overlap": .1, "control_strategy": "translated_mask"}.items():
            payload.setdefault(key, default)
    elif task == "batch_gradcam":
        payload["items"] = [prepare_job_payload("gradcam", item) for item in payload["items"]]
    return payload


def job_runtime(task, payload, workspace_root):
    """Resolve only immutable profile settings or explicit inspected runtime IDs."""
    from model_runtimes import LEGACY, runtime_id
    from model_profiles import ModelRegistry
    if task == "inspect":
        return runtime_id(payload)
    registry = ModelRegistry(workspace_root)
    if task == "infer":
        profile = registry.get(payload["profile_id"], payload["profile_revision"])
        return runtime_id(profile["runtime_config"])
    if task == "gradcam":
        case = CaseStore(workspace_root).load(payload["case_id"])
        item = next(i for i in case["items"] if i["id"] == payload["item_id"])
        link = item.get("external_target") or {}
        run = next(r for r in case["external_runs"] if r["id"] == link.get("run_id"))
        profile = registry.get(run["profile_id"], run["profile_revision"])
        config = dict(profile["runtime_config"])
        if profile.get("gradient_asset"):
            config.update(profile.get("metadata", {}).get("gradient_config", {}))
        return runtime_id(config)
    # DINO stays legacy; comparison/evaluation invoke model-specific RPC as needed.
    return LEGACY


def cohort_record(child, report, state):
    """Separate a target's identity from the recipe shared across source images."""
    target = report.get("target", {})
    execution = target.get("execution", report.get("execution", {}))
    layer = report.get("layer", {})
    stable_layer = {key: layer[key] for key in ("path", "level", "stride_xy", "shape") if key in layer}
    recipe = report.get("contract", report.get("config", {}))
    preprocessing = {key: recipe[key] for key in ("input_size", "resize", "interpolation", "color", "input_layout",
                     "scale", "mean", "std", "pad_value", "input_dtype") if key in recipe}
    return {**child["payload"].get("cohort", {}),
        "status": "measured" if state == "succeeded" and report.get("status") == "ok" and report.get("faithfulness", {}).get("status") == "measured" else "unavailable",
        "source_sha256": report.get("source_sha256"), "evidence_id": report.get("id", child["id"]),
        "target_key": digest(canonical({"source_sha256": report.get("source_sha256"),
            "candidate": target.get("native_candidate_index", target.get("output_index")),
            "class_index": target.get("class_index"), "scalar": target.get("scalar"),
            "layer": layer, "method": report.get("method")}).encode()),
        "contrasts": report.get("faithfulness", {}).get("assessment", {}).get("comparisons", [])
                     if type(target.get("class_index")) is int and target.get("class_name") else [],
        "identity": {"model_sha256": target.get("model_sha256", report.get("model_sha256")),
            "runtime_id": execution.get("runtime_id"), "head_path": execution.get("head_path"),
            "method": report.get("method"), "scalar": target.get("scalar"), "layer": stable_layer,
            "class_index": target.get("class_index"), "class_name": target.get("class_name"),
            "preprocessing": preprocessing, "protocol": report.get("faithfulness", {}).get("protocol")}}


def _relative_result(value, root):
    if isinstance(value, dict):
        return {k: _relative_result(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_relative_result(v, root) for v in value]
    if isinstance(value, str) and value.startswith(str(root)):
        path = Path(value)
        if path.is_relative_to(root):
            return "@artifact/" + path.relative_to(root).as_posix()
    return value


def completed_result(store, job_id):
    from investigation_jobs import JobStore
    jobs = JobStore(store.root)
    value = jobs.result(job_id)
    def resolve(node):
        if isinstance(node, dict):
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        if isinstance(node, str) and node.startswith("@artifact/"):
            path = (jobs.artifact_dir(job_id) / node[len("@artifact/"):]).resolve()
            if not path.is_relative_to(jobs.artifact_dir(job_id).resolve()):
                raise ValueError("分析證據位置無效")
            return str(path)
        return node
    result = resolve(value)
    result["job_id"] = job_id
    if result.get("child_jobs"):
        rows=[]
        for child_id in result["child_jobs"]:
            child=jobs.get(child_id)
            attempts=jobs.list()
            while newer:=next((j for j in attempts if j.get("retry_of")==child["id"]),None):
                child=newer
            child_id=child["id"]
            row={"item_id":child["payload"]["item_id"],"status":child["state"],"job_id":child_id,"error":child.get("error")}
            if child["state"]=="succeeded":row["result"]=completed_result(store,child_id)
            rows.append(row)
        from yolo_attribution import summarize_faithfulness
        result.update(items=rows,support=len(rows),succeeded=sum(r["status"]=="succeeded" for r in rows),
            summary=summarize_faithfulness([r.get("result",{}).get("attribution",{}).get("faithfulness",{"status":"unavailable"}) for r in rows]))
        result["correspondence_support"]={"requested_targets":len(rows),"matched_targets":sum(r["status"]=="succeeded" for r in rows),
            "unique_images":len({r["result"]["attribution"].get("source_sha256") for r in rows if r["status"]=="succeeded" and r["result"]["attribution"].get("source_sha256")}),
            "scope":"僅涵蓋本批次成功驗證的影像與目標；失敗和未完成工作不算相符"}
        summary=result["summary"]
        summary["requested_targets"]=summary.pop("requested_images")
        summary["measured_targets"]=summary.pop("measured_images")
        summary["unique_images"]=result["correspondence_support"]["unique_images"]
        summary["limitation"]+="；以目標計數，同圖多框不等於獨立影像。"
        groups={}
        for row in rows:
            attribution=row.get("result",{}).get("attribution",{})
            box=attribution.get("target",{}).get("box")
            if not box:continue
            area=(box[2]-box[0])*(box[3]-box[1])
            bucket="<1%" if area<.01 else "1–10%" if area<.1 else ">=10%"
            key=attribution["target"]["class_name"]+" / "+bucket
            group=groups.setdefault(key,{"targets":0,"localized_targets":0,"pointing_hits":0,"positive_mass_values":[]})
            group["targets"]+=1
            localization=attribution.get("localization",{})
            if localization.get("status")=="measured":
                group["localized_targets"]+=1
                group["pointing_hits"]+=int(localization["pointing_hit"])
                group["positive_mass_values"].append(localization["positive_mass_in_mask"])
        result["localization_groups"]=groups
        from explanation_validation import cohort_summary
        cohort_rows = []
        for row in rows:
            child = jobs.get(row["job_id"])
            report = row.get("result", {}).get("attribution", {})
            cohort_rows.append(cohort_record(child, report, row["status"]))
        result["cohort_summary"] = cohort_summary(cohort_rows)
        result["cohort_summary"].update(job_id=job_id, child_job_ids=[r["job_id"] for r in rows])
    return result


def inspect_onnx(path):
    import onnx
    model = onnx.load_model_from_string(Path(path).read_bytes())
    if any(t.external_data for t in model.graph.initializer):
        raise ValueError("此 ONNX 使用外部權重檔，請匯出內嵌權重模型")
    onnx.checker.check_model(model)
    inputs = [v for v in model.graph.input if v.name not in {i.name for i in model.graph.initializer}]
    if len(inputs) != 1:
        raise ValueError("目前支援單一影像輸入 ONNX；此模型需要專用 adapter")
    def shape(value):
        return [d.dim_value if d.HasField("dim_value") else None for d in value.type.tensor_type.shape.dim]
    dims = shape(inputs[0])
    if len(dims) != 4:
        raise ValueError("目前支援四維影像 tensor 輸入")
    metadata = {p.key: p.value for p in model.metadata_props}
    names = None
    try:
        raw = ast.literal_eval(metadata.get("names", "None"))
        if isinstance(raw, dict):
            names = [str(raw[i]) for i in range(len(raw))]
        elif isinstance(raw, list):
            names = [str(v) for v in raw]
    except (ValueError, SyntaxError, KeyError, TypeError):
        pass
    config = copy.deepcopy(DEFAULT_CONFIG)
    fields, warnings = {}, []
    layout = "NHWC" if dims[-1] == 3 and dims[1] != 3 else "NCHW"
    size = [dims[2], dims[1]] if layout == "NHWC" else [dims[3], dims[2]]
    if all(isinstance(v, int) and 8 <= v <= 4096 for v in size):
        config["input_size"] = size
        fields["input_size"] = "模型讀取"
    else:
        warnings.append("動態輸入尺寸：請核對現場實際使用的寬高")
        fields["input_size"] = "LV 建議預設"
    config.update(input_layout=layout, input_name=inputs[0].name)
    output = model.graph.output[0]
    out = shape(output)
    guessed = "classification" if len(out) == 2 else "xyxy" if len(out) == 3 and out[-1] == 6 else "yolo_v8"
    config.update(output_format=guessed, output_name=output.name)
    if names:
        config["class_names"] = names
        fields["class_names"] = "模型讀取"
    else:
        count = out[-1] if guessed == "classification" else out[1]-4 if guessed == "yolo_v8" and isinstance(out[1], int) else 1
        count = count if isinstance(count, int) and 1 <= count <= 500 else 1
        config["class_names"] = [str(i) for i in range(count)]
        fields["class_names"] = "待核對類別名稱"
        warnings.append("模型未提供類別名稱；請核對類別順序")
    fields["output_format"] = "形狀建議，需核對輸出格式"
    warnings.append("權重不能證明現場前後處理／判定門檻；請核對設定")
    if guessed == "classification":
        config["normal_classes"] = []
    export_args = {}
    try:
        parsed = ast.literal_eval(metadata.get("args", "{}"))
        if isinstance(parsed, dict):
            export_args = {key: parsed[key] for key in ("nms", "end2end", "max_det", "imgsz", "half", "dynamic") if key in parsed}
    except (ValueError, SyntaxError):
        pass
    task = metadata.get("task") or ("classification" if guessed == "classification" else "detect")
    task = "classification" if task == "classify" else task
    supported_task = task in {"classification", "detect"}
    if not supported_task:
        warnings.append(f"此 ONNX 宣告任務為 {task}；目前需要另行支援其輸出解碼，不能當作一般偵測框。")
    if guessed == "xyxy":
        warnings.append("已處理框的輸出形狀不能區分 one-to-one 與內嵌 NMS；請核對匯出設定。沒有原始 candidate 對應時不提供此 ONNX 的梯度。")
    export_recipe = {"declared_release": metadata.get("version"), "declared_task": metadata.get("task"),
                     "declared_args": export_args, "output_shape": out,
                     "native_candidate_trace": False, "head_verification": "unverified"}
    return {"format": "onnx", "task": task,
            "class_names": config["class_names"], "input_size": config["input_size"], "input_shape": dims,
            "output_shape": out, "default_config": config, "fields": fields, "warnings": warnings,
            "capabilities": {"inference": supported_task, "gradcam": False, "perturbation": supported_task},
            "export_recipe": export_recipe, "metadata": metadata}


def inspect_task(payload, context):
    from model_profiles import ModelRegistry
    registry = ModelRegistry(context.workspace_root)
    asset = registry.asset(payload["asset_sha"])
    path = registry.asset_path(asset)
    context.progress(.1, "檢查模型格式與可用功能")
    if asset["format"] == "onnx":
        inspection = inspect_onnx(path)
    else:
        from yolo_attribution import inspect_model
        from model_runtimes import LEGACY, RELEASES, runtime_id
        selected_runtime = runtime_id(payload)
        config = {key: payload[key] for key in ("runtime_id", "runtime_release", "head_path", "nms") if key in payload}
        inspection = inspect_model(path, trusted=True, config=config) if selected_runtime != LEGACY else inspect_model(path, trusted=True)
        inspection.setdefault("default_config", {}).update(runtime_id=selected_runtime, runtime_release=RELEASES[selected_runtime])
    context.progress(.95, "模型檢查完成，等待核對現場設定")
    return {"inspection": inspection, "asset": asset}


def profile_config(registry, profile, *, gradient=False):
    asset = profile.get("gradient_asset") if gradient else profile.get("deployment_asset") or profile.get("gradient_asset")
    if asset is None:
        raise ValueError("尚未補入對應的 PyTorch 模型")
    cfg = copy.deepcopy(profile["runtime_config"])
    if gradient:
        cfg.update(copy.deepcopy(profile.get("metadata", {}).get("gradient_config", {})))
    cfg["model_path"] = str(registry.asset_path(asset))
    if asset["format"] == "pt":
        cfg["backend"] = "pytorch"
        family = profile.get("metadata", {}).get("gradient_family" if gradient else "family")
        if family:
            cfg["family"] = family
    else:
        cfg = {k: v for k, v in cfg.items() if k in DEFAULT_CONFIG}
    return cfg


def infer_task(payload, context):
    from model_profiles import ModelRegistry
    from explain_service import runtime_for
    store, registry = CaseStore(context.workspace_root), ModelRegistry(context.workspace_root)
    source = payload["source"]
    data = store.image_path(source).read_bytes()
    profile = registry.get(payload["profile_id"], payload["profile_revision"])
    context.progress(.1, "載入已選模型與原始影像")
    runtime = runtime_for(profile_config(registry, profile), trusted=True)
    run = runtime.infer(data, context.artifact_dir)
    run.update(config=runtime.config, source_id=source["id"], source_context_id=source["sha256"],
               profile_id=profile["id"], profile_revision=profile["revision"], profile_snapshot=profile)
    reference = payload.get("reference")
    run["parity"] = compare_reference(run, reference, source["kind"])
    if payload.get("case_id"):
        case=store.load(payload["case_id"])
        run["reported_comparisons"]=compare_imported_records(case,source,run)
    run["decision_scope"] = {"model_output": "本次模型推論", "machine_decision": "unverified",
                             "reported_rules": profile.get("metadata", {}).get("machine_rules", ""),
                             "reason": "模型輸出與機台最終規則分開；尚未驗證整條現場判定"}
    context.progress(.95, "推論完成，保留完整輸入與輸出")
    return _relative_result({"run": run, "source": source, "profile": profile}, context.artifact_dir)


def compare_imported_records(case,source,run):
    """Compare supplied fields without inventing deployment/tensor identities."""
    results=[]
    detections=run["prediction"].get("detections",[])
    for evidence in case["evidence"]:
        details=evidence["details"]
        annotation=details.get("annotation",{})
        if details.get("source_id")!=source["id"] or annotation.get("role")!="prediction":continue
        if annotation.get("status") not in {"provided","empty"}:
            results.append({"evidence_id":evidence["id"],"status":"unavailable","reason":"現場預測未提供或格式有誤"})
            continue
        rows=annotation.get("boxes",[])
        used=set(); unmatched=[]
        for row in rows:
            options=[j for j,d in enumerate(detections) if j not in used and d["class"]==row["class"] and iou(d["box"],row["box"])>=.9
                     and (row.get("score") is None or abs(d["score"]-row["score"])<=.01)]
            if len(options)==1:used.add(options[0])
            else:unmatched.append(row)
        results.append({"evidence_id":evidence["id"],"status":"partial_fields_match" if not unmatched and len(used)==len(detections) else "partial_fields_differ",
            "reported_count":len(rows),"current_count":len(detections),"matched_count":len(used),"unmatched":unmatched,
            "missing_identity":["部署模型身分","部署輸入 tensor","部署設定","紀錄時間／來源驗證"],
            "scope":"僅核對紀錄內的類別、位置及已提供分數；不能證明歷史部署或機台判定重現"})
    return results


def target_identity(case, item):
    source = next(s for s in case["sources"] if s["id"] == item["source_id"])
    return digest(canonical({"source": source["sha256"], "id": item["id"], "roi": item["roi"],
                             "label": item["label"], "target": item.get("external_target"),
                             "withdrawn": bool(item.get("withdrawn"))}).encode())


def select_detection(case, run, detection):
    rows = run["prediction"].get("detections", [])
    if detection not in rows:
        raise ValueError("所選框不屬於此推論結果")
    key = {"run_id": run["id"], "output_index": detection["output_index"], "class": detection["class"]}
    existing = next((i for i in case["items"] if not i.get("withdrawn") and i.get("external_target") == key), None)
    if existing:
        return existing
    item = next((i for i in case["items"] if not i.get("withdrawn") and i["source_id"] == run["source_id"]
                 and i["roi"] == detection["box"] and i["label"] == detection["class"]), None)
    if item is None:
        item = add_item(case, run["source_id"], detection["box"], detection["class"], detection["score"],
                        str(detection["output_index"]), "external_inference")
    item["external_target"] = key
    return item


def _match_target(deployment, gradient, selected, *, box_atol=.01, score_atol=.01):
    if deployment["prediction"]["task"] != "detection" or gradient["prediction"]["task"] != "detection":
        raise ValueError("此功能需要偵測模型輸出")
    a, b = deployment["prediction"]["detections"], gradient["prediction"]["detections"]
    if deployment.get("input_sha256") and gradient.get("input_sha256") and deployment["input_sha256"] != gradient["input_sha256"]:
        raise ValueError("配套模型的實際輸入 tensor 不同，請先核對縮放、色彩與正規化；不能把其他輸入的熱圖套回原模型")
    if deployment["contract_id"] == gradient["contract_id"] and deployment.get("input_sha256") and deployment.get("output_sha256") and deployment["input_sha256"] == gradient.get("input_sha256") and deployment["output_sha256"] == gradient.get("output_sha256") and a == b:
        return b[a.index(selected)], {"status":"native_matched", "scope":"相同模型、設定與完整輸出；使用原始候選索引",
                                     "deployment_contract":deployment["contract_id"], "gradient_contract":gradient["contract_id"]}
    matches = []
    for row in a:
        choices = [r for r in b if r["class"] == row["class"]
                   and max(abs(x-y) for x,y in zip(r["box"], row["box"])) <= box_atol
                   and abs(r["score"]-row["score"]) <= score_atol]
        if len(choices) != 1:
            raise ValueError("部署與歸因模型的框無唯一對應；請核對設定或在配套模型另建調查")
        matches.append(choices[0])
    if len(a) != len(b) or len({(r["output_index"],r["class"]) for r in matches}) != len(b):
        raise ValueError("部署與歸因模型完整輸出不一致，不能直接沿用原框")
    index = a.index(selected)
    return matches[index], {"status": "per_case_matched", "box_atol": box_atol, "score_atol": score_atol,
            "deployment_contract": deployment["contract_id"], "gradient_contract": gradient["contract_id"],
            "scope": "本圖完整輸出對齊；不代表兩個計算圖或梯度等價"}


def gradcam_task(payload, context):
    from model_profiles import ModelRegistry
    from explain_service import runtime_for
    store, registry = CaseStore(context.workspace_root), ModelRegistry(context.workspace_root)
    case = store.load(payload["case_id"])
    item = next(i for i in case["items"] if i["id"] == payload["item_id"])
    if item.get("withdrawn") or target_identity(case, item) != payload["target_identity"]:
        raise ValueError("所選區域已改變，請重新選框")
    link = item.get("external_target")
    if not link:
        raise ValueError("請先選取模型的實際預測框；人工圈區不是模型輸出")
    run = next(r for r in case["external_runs"] if r["id"] == link["run_id"])
    source = next(s for s in case["sources"] if s["id"] == item["source_id"])
    profile = registry.get(run["profile_id"], run["profile_revision"])
    asset = profile.get("gradient_asset") or profile.get("deployment_asset")
    if not asset or asset["format"] != "pt":
        raise ValueError("此模型目前只有推論能力；請在我的模型補入對應 PyTorch 模型，或使用對照實驗")
    config = profile_config(registry, profile, gradient=bool(profile.get("gradient_asset")))
    runtime = runtime_for(config, trusted=True)
    data = store.image_path(source).read_bytes()
    selected = next(r for r in run["prediction"]["detections"] if r["output_index"] == link["output_index"] and r["class"] == link["class"])
    context.progress(.1, "核對完整原圖與所選框")
    current = runtime.infer(data, context.artifact_dir)
    mapped, correspondence = _match_target(run, current, selected)
    context.progress(.3, "計算所選框的模型歸因")
    mask = None
    if payload.get("mask_source"):
        from PIL import Image
        import numpy as np
        with Image.open(store.image_path(payload["mask_source"])) as image:
            mask = np.asarray(image.convert("L")) > 0
    result = runtime.gradcam(data, output_index=mapped["output_index"],
        class_index=runtime.config["class_names"].index(mapped["class"]),
        scalar=payload.get("scalar", "confidence"), method=payload.get("method", "positive"),
        artifact_root=context.artifact_dir, fractions=tuple(payload.get("fractions", [.1,.25,.5])),
        controls=payload.get("controls",3), mask=mask, cancelled=context.check_cancelled,
        protocol_version=payload.get("protocol_version", 2), fill_methods=tuple(payload.get("fill_methods", ["constant", "blur"])),
        max_overlap=payload.get("max_overlap", .1), control_strategy=payload.get("control_strategy", "translated_mask"),
        profile_revision=str(profile["revision"]), case_context={"case_id":case["id"],"item_id":item["id"],"source_id":source["id"],"target_identity":payload["target_identity"],"run_id":run["id"]})
    result["randomization_sanity"] = runtime.randomization_sanity(data, result["target"], cancelled=context.check_cancelled) if payload.get("randomization") else {"status":"unavailable", "reason":"未執行權重隨機化檢查"}
    if payload.get("mask_source"):
        mask_source=payload["mask_source"]
        path=Path(result["artifact_dir"])/"localization_mask.png"
        from PIL import Image
        Image.fromarray((mask.astype("uint8")*255)).save(path)
        result["localization_mask"]={"source_sha256":mask_source["sha256"],"artifact":"localization_mask.png",
                                     "artifact_sha256":digest(path.read_bytes()),"conversion":"L > 0","size":mask_source["size"]}
    result.update(case_id=case["id"], item_id=item["id"], source_id=source["id"],
        source_sha256=source["sha256"], run_id=run["id"], target_identity=payload["target_identity"],
        profile_id=profile["id"], profile_revision=profile["revision"], profile_snapshot=profile,
        correspondence=correspondence, historical_parity=run.get("parity"),
        attribution_source="PyTorch 模型歸因", deployment_detection=selected)
    context.progress(.98, "保存熱圖原始數值與檢查結果")
    return _relative_result({"attribution": result, "source": source}, context.artifact_dir)


def _portable_paths(row, store):
    if row.get("artifact_dir"):
        path = Path(row["artifact_dir"]).resolve()
        if path.is_relative_to(store.root):
            row["artifact_relative"] = path.relative_to(store.root).as_posix()


def attach_result(store, case_id, result):
    """Publish onto a freshly loaded compatible case; a concurrent save remains a visible conflict."""
    case = store.load(case_id)
    source = result.get("source")
    if source and not any(s["id"] == source["id"] and s["sha256"] == source["sha256"] for s in case["sources"]):
        raise ValueError("此結果影像不屬於目前案件；結果仍保留在工作紀錄")
    if "run" in result:
        row = copy.deepcopy(result["run"])
        if any(r["id"] == row["id"] for r in case.get("external_runs", [])):
            return case
        row["evidence_id"] = _evidence(case, "external_inference", "外部模型原圖推論", run_id=row["id"],
            source_id=row["source_id"], profile_id=row["profile_id"], profile_revision=row["profile_revision"],
            prediction=row["prediction"], parity=row.get("parity"), decision_scope=row.get("decision_scope"))
        _portable_paths(row, store)
        case.setdefault("external_runs", []).append(row)
        case["model_profile"] = {"id": row["profile_id"], "revision": row["profile_revision"]}
    elif "attribution" in result:
        row = copy.deepcopy(result["attribution"])
        item = next((i for i in case["items"] if i["id"] == row["item_id"]), None)
        if item is None or item.get("withdrawn") or target_identity(case,item) != row["target_identity"]:
            raise ValueError("區域已改變；已完成熱圖仍保留在工作紀錄，不能覆蓋目前區域")
        row.setdefault("id", "cam_" + digest(canonical(row).encode())[:16])
        if any(r["id"] == row["id"] for r in case.get("attributions", [])):
            return case
        _portable_paths(row,store)
        overlay=Path(row["artifact_dir"])/"overlay.png"
        if not overlay.resolve().is_relative_to(store.root):
            raise ValueError("熱圖資產不屬於此工作區")
        row["overlay_sha256"]=digest(overlay.read_bytes())
        row["evidence_id"] = _evidence(case, "model_attribution", "所選框的 PyTorch 模型歸因", item_id=item["id"],
            attribution_id=row["id"], profile_id=row["profile_id"], profile_revision=row["profile_revision"],
            correspondence=row["correspondence"], historical_parity=row.get("historical_parity"),
            target_identity=row["target_identity"], limitation="熱圖不是分割真值或已確認根因")
        case.setdefault("attributions", []).append(row)
    elif "evaluation" in result or "experiment" in result:
        kind = "evaluation" if "evaluation" in result else "experiment"
        row = copy.deepcopy(result[kind])
        if result.get("case_id") != case_id:
            raise ValueError("工作不屬於此案件")
        if kind == "experiment":
            item = next((i for i in case["items"] if i["id"] == row["item_id"]), None)
            if item is None or item.get("withdrawn") or (result.get("target_identity") and target_identity(case,item) != result["target_identity"]):
                raise ValueError("實驗區域已變更；結果保留於工作紀錄")
        collection = "evaluations" if kind == "evaluation" else "experiments"
        if any(r["id"] == row["id"] for r in case.get(collection, [])):
            return case
        row["evidence_id"] = _evidence(case, "external_"+kind, "改善評估" if kind == "evaluation" else "外部模型對照實驗", **{kind:copy.deepcopy(row)})
        case.setdefault(collection, []).append(row)
    elif "comparison" in result:
        from fp_case_search import record_search
        row = copy.deepcopy(result["comparison"])
        item = next((i for i in case["items"] if i["id"] == row["query_item_id"]), None)
        if item is None or item.get("withdrawn") or (result.get("target_identity") and target_identity(case,item) != result["target_identity"]):
            raise ValueError("搜尋區域已變更；結果保留於工作紀錄")
        if row.get("attribution_evidence_id"):
            context = row.get("attribution_context")
            if not context or context.get("attribution_evidence_id") != row["attribution_evidence_id"]:
                raise ValueError("熱圖搜尋來源缺少一致的目標紀錄；結果保留於工作紀錄")
            validate_attribution_context(case, item, context)
        if any(r["id"] == row["id"] for r in case["searches"]):
            return case
        record_search(case,row)
    else:
        raise ValueError("此工作沒有可加入案件的推論或歸因結果")
    return store.save(case)


def promote_trial(store, result, *, title=""):
    source = result["source"]
    store.image_path(source)
    case = store.create(source, title or source["name"], "由模型試跑建立案件")
    return attach_result(store,case["id"],result)


def evaluate_task(payload, context):
    from fp_investigation import paired_evaluation
    from model_profiles import ModelRegistry
    store, registry = CaseStore(context.workspace_root), ModelRegistry(context.workspace_root)
    case = store.load(payload["case_id"])
    configs = [profile_config(registry,registry.get(p["id"],p["revision"])) for p in payload["profiles"]] if payload.get("profiles") else payload["configs"]
    result = paired_evaluation(case,store,*configs,payload["manifest"],
        min_support=payload.get("min_support",10), max_miss_increase=payload.get("max_miss_increase",0),
        min_fp_reduction=payload.get("min_fp_reduction",0),
        progress=lambda d,t: context.progress(d/max(t,1), f"比較新舊模型 {d}/{t}"))
    return _relative_result({"evaluation": result, "case_id": case["id"]},context.artifact_dir)


def experiment_task(payload, context):
    from fp_investigation import run_experiment
    store = CaseStore(context.workspace_root)
    case = store.load(payload["case_id"])
    item = next(i for i in case["items"] if i["id"] == payload["target_item_id"])
    if target_identity(case,item) != payload["target_identity"] or item.get("withdrawn"):
        raise ValueError("實驗目標已變更")
    context.check_cancelled()
    result = run_experiment(case,payload["baseline_id"],store,target_item_id=item["id"],
                            progress=lambda d,t:context.progress(d/max(t,1),f"對照實驗 {d}/{t}"), **payload["kwargs"])
    return {"experiment":result,"case_id":case["id"],"target_identity":payload["target_identity"],
            "source":next(s for s in case["sources"] if s["id"]==item["source_id"])}


def validate_attribution_context(case, item, context):
    """Bind a CAM association to both the current query and its original model target."""
    if not isinstance(context, dict) or not context.get("attribution_evidence_id"):
        raise ValueError("熱圖搜尋來源缺少目標紀錄，請從模型診斷重新選擇區域")
    if (item.get("withdrawn") or context.get("item_id") != item["id"]
            or context.get("target_identity") != target_identity(case, item)):
        raise ValueError("熱圖搜尋區域已變更，請從模型診斷重新選擇區域")
    ref = context["attribution_evidence_id"]
    evidence = next((e for e in case["evidence"] if e["id"] == ref and e["role"] == "model_attribution"), None)
    cam = next((r for r in case.get("attributions", []) if r.get("evidence_id") == ref), None)
    if not evidence or not cam or evidence["details"].get("attribution_id") != cam["id"]:
        raise ValueError("熱圖搜尋來源不存在或不屬於此案件")
    original = next((i for i in case["items"] if i["id"] == cam.get("item_id")), None)
    if (original is None or original.get("withdrawn") or not cam.get("target_identity")
            or target_identity(case, original) != cam["target_identity"]):
        raise ValueError("原熱圖模型目標已變更，請重新分析後再找對照")
    if item["id"] != original["id"]:
        linked = any(e["role"] == "attribution_region"
                     and e["details"].get("item_id") == item["id"]
                     and e["details"].get("roi") == item["roi"]
                     and e["details"].get("attribution_evidence_id") == ref for e in case["evidence"])
        if item.get("origin") != "cam_region" or item["source_id"] != original["source_id"] or not linked:
            raise ValueError("此搜尋區域未連結到所選熱圖，請重新選擇區域")
    return copy.deepcopy(context)


def comparison_task(payload, context):
    from fp_case_search import search
    store=CaseStore(context.workspace_root)
    case=store.load(payload["case_id"])
    item = next((i for i in case["items"] if i["id"] == payload["item_id"]), None)
    if item is None or item.get("withdrawn") or (payload.get("target_identity") and target_identity(case, item) != payload["target_identity"]):
        raise ValueError("搜尋區域已變更，請重新選擇搜尋目標")
    attribution_context = payload.get("attribution_context")
    if payload.get("attribution_evidence_id") or attribution_context:
        if not isinstance(attribution_context, dict) or attribution_context.get("attribution_evidence_id") != payload.get("attribution_evidence_id"):
            raise ValueError("熱圖搜尋來源缺少一致的目標紀錄")
        attribution_context = validate_attribution_context(case, item, attribution_context)
    result=search(case,payload["item_id"],store,payload.get("normal_folder",""),payload.get("defect_folder",""),
        candidate_scope=payload.get("candidate_scope","whole_image"),limit=payload.get("limit",300),
        top_k=payload.get("top_k",6),include_history=payload.get("include_history",True),datasets=payload.get("datasets"),
        progress=lambda d,t:context.progress(d/max(t,1),f"找對照 {d}/{t}"))
    result["attribution_evidence_id"]=payload.get("attribution_evidence_id")
    if attribution_context:
        result["attribution_context"] = attribution_context
    return {"comparison":result,"case_id":case["id"],"target_identity":payload.get("target_identity")}


def batch_gradcam_task(payload, context):
    if not 1 <= len(payload["items"]) <= 200:
        raise ValueError("每批需有 1–200 個目標")
    from investigation_jobs import JobStore
    jobs=JobStore(context.workspace_root)
    children=[]
    for item in payload["items"]:
        context.check_cancelled()
        item = prepare_job_payload("gradcam", item)
        job=jobs.submit("gradcam",item,digest(canonical(item).encode()),context={"case_id":item["case_id"],"item_id":item["item_id"]})
        children.append(job["id"])
    return {"child_jobs":children,"scope":"已安排獨立目標工作；逐筆保存，取消或失敗不丟失其他已完成結果"}
