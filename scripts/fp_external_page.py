"""External model setup, evidence experiments, evaluation and training handoff UI."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw
import streamlit as st

from fp_cases import VERDICTS
from fp_external_runtime import DEFAULT_CONFIG, config_checked
from fp_investigation import run_case, run_experiment, current_run, paired_evaluation
from fp_training_export import training_rows, export_training

FORMAT_LABELS = {"yolo_v8": "YOLO 類別分數 [1,4+C,N]", "yolo_v5": "YOLO objectness [1,N,5+C]",
                 "xyxy": "已後處理框 [1,N,6]：xyxy／score／class", "classification": "分類 [1,C]"}
PARITY_LABELS = {"unavailable": "尚無足夠歷史資料可驗證", "matched": "當時完整輸出已重現", "mismatched": "與當時判定不一致"}


def _save(store, case, message):
    store.save(case)
    st.session_state["fp_notice"] = message
    st.rerun()


def _config_form(store, case, profile, title):
    previous = case.get("runtime_profiles", {}).get(profile, DEFAULT_CONFIG)
    with st.expander(title, expanded=profile not in case.get("runtime_profiles", {})):
        st.caption("單一影像輸入、內嵌權重 ONNX；選定格式後才解讀輸出，不會猜測模型架構。")
        with st.form(f"fp_runtime_config_{case['id']}_{profile}"):
            path = st.text_input("ONNX 模型路徑", value=previous["model_path"])
            version = st.text_input("部署／模型版本名稱", value=previous["model_version"])
            formats = list(FORMAT_LABELS)
            fmt = st.selectbox("模型輸出格式", formats, index=formats.index(previous["output_format"]), format_func=FORMAT_LABELS.get)
            names = st.text_input("類別順序（逗號分隔，須與模型輸出一致）", value=",".join(previous["class_names"]))
            normal = st.text_input("正常類別（分類模型必填，逗號分隔）", value=",".join(previous["normal_classes"]))
            a, b = st.columns(2)
            width = a.number_input("輸入寬度", 8, 4096, previous["input_size"][0])
            height = b.number_input("輸入高度", 8, 4096, previous["input_size"][1])
            resize = st.selectbox("縮放方式", ["letterbox", "stretch"], index=["letterbox", "stretch"].index(previous["resize"]),
                                  format_func=lambda r: "保留比例＋補邊" if r == "letterbox" else "縮放至指定寬高")
            threshold = st.number_input("判定分數門檻", 0.0, 1.0, float(previous["threshold"]), step=.01)
            main_keys = {"model_path", "model_version", "output_format", "class_names", "normal_classes", "input_size", "resize", "threshold"}
            extra = {k: v for k, v in previous.items() if k not in main_keys}
            advanced = st.text_area("前後處理設定 JSON", value=json.dumps(extra, ensure_ascii=False, indent=2), height=220,
                                   help="score_activation：none=已是機率，softmax=分類 logits，sigmoid=獨立類別 logits。NMS、RGB/BGR、normalization 必須與部署一致。")
            submit = st.form_submit_button("儲存模型設定")
        if submit:
            extra = json.loads(advanced)
            if not isinstance(extra, dict) or set(extra) & main_keys:
                raise ValueError("進階設定只能包含下方前後處理欄位，不能覆蓋上方模型設定")
            configured = {**extra, "model_path": path, "model_version": version, "output_format": fmt,
                          "class_names": [n.strip() for n in names.split(",") if n.strip()],
                          "normal_classes": [n.strip() for n in normal.split(",") if n.strip()],
                          "input_size": [int(width), int(height)], "resize": resize, "threshold": threshold}
            case.setdefault("runtime_profiles", {})[profile] = config_checked(configured)
            label = "基準模型" if profile == "baseline" else "候選模型"
            _save(store, case, f"{label}設定已儲存；尚未執行推論")


def _run_preview(run, *, caption="實際模型輸入"):
    root = Path(run["artifact_dir"])
    if not (root / "model_input.png").is_file():
        st.warning("本次模型輸入圖片已不在原位置；請核對案件備份。")
        return
    left, right = st.columns(2)
    with left:
        st.image(str(root / "model_input.png"), caption=caption, width="stretch")
    with right:
        with Image.open(root / "source.png") as original:
            image = original.convert("RGB")
        draw = ImageDraw.Draw(image)
        w, h = image.size
        for det in run["prediction"]["detections"]:
            box = det["box"]
            draw.rectangle([box[0]*w, box[1]*h, min(w-1, box[2]*w), min(h-1, box[3]*h)], outline="red", width=max(2, w//200))
        image.thumbnail((700, 500))
        st.image(image, caption="原始幾何上的最終輸出框", width="stretch")
    p = run["prediction"]
    if p["task"] == "classification":
        st.write(f"預測 {p['class']} · 分數 {p['score']:.5f} · {'拒判' if p['decision'] == 'abstain' else '警報' if p['alarm'] else '未警報'}")
        st.json(p["scores"], expanded=False)
    else:
        st.write(f"最終輸出 {len(p['detections'])} 個框 · {'警報' if p['alarm'] else '未警報'}")
        if not p["raw_candidates_available"]:
            st.caption("此模型輸出已經過後處理；沒有 NMS 前候選分數，不會再執行一次 NMS。")
        st.dataframe(p["detections"], hide_index=True)


def _artifact_zip(run):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("evidence.json", json.dumps(run, ensure_ascii=False, indent=2))
        for name in ["model_input.png", "source.png", "tensors.npz"]:
            path = Path(run["artifact_dir"]) / name
            if path.exists():
                archive.write(path, name)
    return output.getvalue()


def _durable_analysis(store, case, task):
    from fp_attribution import manager_for, completed_result, attach_result
    manager = manager_for(store)
    jobs = [j for j in manager.store.list(case_id=case['id']) if j['task'] == task]
    if not jobs:
        return
    @st.fragment(run_every='2s')
    def status_panel():
        job = manager.store.get(jobs[0]['id'])
        labels = {'queued':'等待執行','running':'執行中','succeeded':'已完成','failed':'失敗','cancelled':'已取消','interrupted':'中斷'}
        st.caption(labels[job['state']]+' · '+str(job.get('stage','')))
        if job['state'] in {'queued','running'}:
            st.progress(float(job.get('progress',0)))
            if st.button('取消分析工作',key='analysis_cancel_'+job['id']):
                manager.cancel(job['id'])
                st.rerun()
        elif job['state'] == 'succeeded':
            publication=job.get('publication') or {}
            if publication.get('case_id') == case['id'] and not publication.get('error'):
                st.caption('分析結果已加入案件')
            elif st.button('將完成結果加入案件',key='analysis_publish_'+job['id']):
                try:
                    attach_result(store,case['id'],completed_result(store,job['id']))
                    manager.store.mark_published(job['id'],case_id=case['id'])
                    st.rerun()
                except (ValueError,RuntimeError) as exc:
                    st.error(str(exc)+'；結果仍保留，可重新載入案件後再加入。')
        else:
            st.error(job.get('error') or '工作未完成，已保存資料仍保留')
            if st.button('重試分析工作',key='analysis_retry_'+job['id']):
                manager.retry(job['id'])
                st.rerun()
    status_panel()


def render_external(store, case):
    st.markdown("**外部模型重播與對照實驗**")
    mode = st.segmented_control("外部分析步驟", ["模型設定", "執行與歷史比對", "對照實驗"], default="模型設定",
                                key="fp_external_mode_" + case["id"]) or "模型設定"
    if mode == "模型設定":
        from investigation_models_page import render_model_library
        from fp_attribution import manager_for
        render_model_library(store,manager_for(store))
        with st.expander('進階本機模型設定'):
            st.caption('舊版設定可在下方保留；一般操作請使用上傳模型。')
        if st.checkbox('顯示舊版模型設定',key='external_legacy_'+case['id']):
            _config_form(store, case, "baseline", "目前／基準模型")
            _config_form(store, case, "candidate", "改善後／候選模型（可稍後設定）")
        return
    if mode == "執行與歷史比對":
        from investigation_workspace import render_inference
        source = st.selectbox('選擇推論影像',case['sources'],format_func=lambda s:s['name'],key='external_source_'+case['id'])
        render_inference(store,case,source)
        legacy = bool(case.get('items') and case.get('runtime_profiles')) and st.checkbox('進階：舊版歷史紀錄比對',key='external_replay_legacy_'+case['id'])
        if legacy:
            _legacy_replay_form(store,case)
        if not case.get('external_runs'):
            st.info('尚無推論結果；不需要先建立爭議框即可推論。')
            return
        run = st.selectbox('外部推論紀錄',list(reversed(case['external_runs'])),format_func=lambda r:r['id'])
        _run_preview(run)
        st.write(PARITY_LABELS[run.get('parity',{}).get('status','unavailable')])
        for reason in run.get('parity',{}).get('reasons',[]):
            st.caption(reason)
        for comparison in run.get('reported_comparisons',[]):
            label={'partial_fields_match':'已提供的紀錄欄位相符','partial_fields_differ':'已提供的紀錄欄位有差異','unavailable':'紀錄尚不可比較'}[comparison['status']]
            st.write(label)
            st.caption(comparison.get('scope',comparison.get('reason','')))
            if 'reported_count' in comparison:
                st.write(f"紀錄 {comparison['reported_count']} 框 · 本次 {comparison['current_count']} 框 · 匹配 {comparison['matched_count']} 框")
        st.download_button('下載本次推論 JSON',json.dumps(run,ensure_ascii=False,indent=2),file_name=run['id']+'.json',mime='application/json',key='external_selected_json_'+case['id'])
        if st.button("準備完整推論證據包"):
            st.session_state["external_run_zip_"+case["id"]]=(run["id"],_artifact_zip(run))
        ready=st.session_state.get("external_run_zip_"+case["id"])
        if ready and ready[0]==run["id"]:
            st.download_button("下載推論證據 ZIP",ready[1],file_name=run["id"]+".zip",mime="application/zip",on_click="ignore")
        return
    _durable_analysis(store,case,'experiment')
    _render_experiment(store,case)


def _legacy_replay_form(store,case):
        with st.form("fp_replay_" + case["id"]):
            item = st.selectbox("要調查的爭議區域", case["items"], format_func=lambda r: f"{r['label'] or '未命名'} · {r['id']}")
            profile = st.selectbox("使用模型設定", list(case["runtime_profiles"]), format_func=lambda k: "目前／基準模型" if k == "baseline" else "改善後／候選模型")
            reference_file = st.file_uploader("當時部署輸出 JSON（可留空）", type=["json"])
            reference_text = st.text_area("或貼上當時部署輸出 JSON", help="需要 origin=deployment、recorded_at、image_sha256、contract_id、完整 prediction。沒有歷史資料仍可執行本次推論。")
            a, b = st.columns(2)
            score_tol = a.number_input("分數容許誤差", 0.0, .1, .001, step=.001, format="%.4f")
            box_tol = b.number_input("框座標容許誤差（0–1）", 0.0, .1, .001, step=.001, format="%.4f")
            submit = st.form_submit_button("執行外部模型並比對", type="primary")
        if submit:
            reference = json.loads(reference_file.getvalue().decode("utf-8-sig")) if reference_file else json.loads(reference_text) if reference_text.strip() else None
            with st.spinner("載入外部模型、執行推論並核對完整輸出…"):
                run_case(case, item["id"], store, case["runtime_profiles"][profile], reference,
                         score_atol=score_tol, box_atol=box_tol)
            _save(store, case, "外部推論與比對證據已儲存")

def _render_experiment(store,case):
    baselines = [r for r in case.get("external_runs", []) if current_run(case, r)]
    if not baselines:
        st.info("先執行一筆符合目前來源／區域的基準推論。")
        return
    baseline = st.selectbox("實驗基準紀錄", list(reversed(baselines)), format_func=lambda r: f"{r['id']} · {r['contract']['model_version']}")
    source_id = baseline.get('source_id')
    items = [i for i in case['items'] if not i.get('withdrawn') and i['source_id']==source_id and (not i.get('external_target') or i['external_target'].get('run_id')==baseline['id'])]
    detections = baseline['prediction'].get('detections',[])
    if detections:
        chosen = st.selectbox('從本次輸出選擇目標框',detections,format_func=lambda d:f"{d['class']} · {d['score']:.3f} · {d['box']}",key='experiment_detection_'+baseline['id'])
        if st.button('使用此框作為實驗目標',key='experiment_select_'+baseline['id']):
            from fp_attribution import select_detection
            select_detection(case,baseline,chosen)
            _save(store,case,'已選定本次推論的目標框')
    if not items:
        st.info('請選取本次輸出框，或在案件影像圈出要調查的區域。沒有檢出時仍可找對照與回覆，不會虛構目標。')
        return
    item = st.selectbox('固定觀察的區域',items,format_func=lambda i:f"{i['label']} · {i['roi']}",key='experiment_item_'+baseline['id'])
    st.caption("基準模型、前後處理與目標位置保持固定；下方座標指定要修改的影像區域。")
    with st.form("fp_experiment_" + case["id"] + baseline["id"]):
        names=baseline['config']['class_names']
        target = st.selectbox("固定觀察的模型類別", names,index=names.index(item['label']) if item['label'] in names else 0)
        mode_value = st.selectbox("實驗方法", ["brightness", "occlusion"], format_func=lambda m: "亮度對照（包含原亮度控制）" if m == "brightness" else "區域遮擋（包含等面積位置對照）")
        cols = st.columns(4)
        roi = [col.number_input(label, 0.0, 100.0, float(value*100), step=1.0) / 100 for col, label, value in
               zip(cols, ["修改左 %", "修改上 %", "修改右 %", "修改下 %"], item["roi"])]
        a, b, c = st.columns(3)
        factor = a.number_input("亮度倍數", .1, 2.0, .8, step=.1)
        fill = b.number_input("遮擋灰階填充值", 0, 255, 114)
        controls = c.number_input("遮擋位置對照數", 1, 10, 3)
        preserved = st.checkbox("此修改範圍與方式保留了待判斷目標的有效外觀（仍須檢查結果圖）")
        submit = st.form_submit_button("執行對照實驗", type="primary")
    if submit:
        from fp_attribution import manager_for,target_identity
        from fp_cases import digest
        payload={'case_id':case['id'],'baseline_id':baseline['id'],'target_item_id':item['id'],'target_identity':target_identity(case,item),
            'kwargs':{'target_class':target,'intervention_roi':roi,'mode':mode_value,'factor':factor,'fill':int(fill),'controls':int(controls),'target_preserved':preserved}}
        manager_for(store).submit('experiment',payload,digest(json.dumps(payload,sort_keys=True).encode()),context={'case_id':case['id'],'item_id':item['id']})
        st.rerun()
    if case.get("experiments"):
        experiment = st.selectbox("對照實驗紀錄", list(reversed(case["experiments"])), format_func=lambda e: e["id"] + " · " + e["mode"])
        st.info(experiment["limitation"])
        if experiment["available_controls"] < experiment["requested_controls"]:
            st.warning("此影像範圍不足以建立所需的不同等面積對照；未虛構對照結果。")
        st.write(f"基準目標分數：{experiment['baseline_target']['score']:.5f}")
        for index, variant in enumerate(experiment["variants"]):
            with st.expander("指定修改" if variant["role"] == "selected" else f"對照組 {index}", expanded=index == 0):
                _run_preview(variant["run"], caption="修改後真正送入模型的圖片")
                st.write(f"目標分數：{variant['target']['score'] if variant['target']['score'] is not None else '沒有匹配目標'}")
                delta = variant['score_drop']
                if delta is None:
                    st.write("分數變化：不可判讀")
                elif delta < 0:
                    st.write(f"分數上升：{-delta:.5f}")
                elif delta > 0:
                    st.write(f"分數下降：{delta:.5f}")
                else:
                    st.write("分數未變")
                st.caption("單圖敏感性觀察" if variant["status"] == "single_image_sensitivity" else "目標外觀未確認保留或無法匹配；不判定改善。")
        st.download_button("下載對照實驗 JSON", json.dumps(experiment, ensure_ascii=False, indent=2),
                           file_name=experiment["id"]+".json", mime="application/json", on_click="ignore")


def render_evaluation(store, case):
    st.markdown("**改善前後：固定資料集評估**")
    st.caption("同一批圖片、同一組真值比較兩個模型；報案回歸與獨立評估分開。沒有足夠正常／瑕疵支持數時，不判定改善通過。")
    from model_profiles import ModelRegistry
    from investigation_data_page import evaluation_manifest_form
    registry = ModelRegistry(store.root)
    profiles = registry.list()
    _durable_analysis(store,case,'evaluate')
    if profiles:
        a,b = st.columns(2)
        baseline = a.selectbox("改善前模型", profiles, format_func=lambda p:p['name'], key='eval_base_'+case['id'])
        candidate = b.selectbox("改善後模型", profiles, format_func=lambda p:p['name'], key='eval_candidate_'+case['id'])
        baseline_revision=a.selectbox("改善前設定版本",list(range(baseline['latest_revision'],0,-1)),key='eval_base_revision_'+case['id']+baseline['id'])
        candidate_revision=b.selectbox("改善後設定版本",list(range(candidate['latest_revision'],0,-1)),key='eval_candidate_revision_'+case['id']+candidate['id'])
        baseline=registry.get(baseline['id'],baseline_revision)
        candidate=registry.get(candidate['id'],candidate_revision)
        if st.button("使用這兩個模型設定",key='eval_models_'+case['id']):
            case.setdefault('runtime_profiles',{}).update(baseline=registry.runtime_config(baseline['id'],baseline_revision),candidate=registry.runtime_config(candidate['id'],candidate_revision))
            _save(store,case,'評估模型設定已保存')
    if st.checkbox('進階：本機模型設定',key='eval_local_config_'+case['id']):
        _config_form(store, case, "candidate", "設定改善後／候選模型")
    if not profiles and not {"baseline", "candidate"} <= set(case.get("runtime_profiles", {})):
        st.info("請先在「外部模型」設定目前／基準模型，並在此設定候選模型。")
        return
    input_mode = st.radio('評估資料來源',['選擇已上傳資料集','進階 JSON 清單'],key='eval_input_'+case['id'])
    manifest = None
    if input_mode == '選擇已上傳資料集':
        purpose = st.selectbox('評估用途',['case_regression','independent'],format_func=lambda p:'報案回歸' if p=='case_regression' else '獨立改善驗收',key='eval_purpose_'+case['id'])
        manifest = evaluation_manifest_form(store,'eval_data_'+case['id'],purpose)
    with st.expander("評估 JSON 範例"):
        st.code(json.dumps({"schema_version": 1, "purpose": "independent", "independent_declared": True,
            "split_basis": "不同批次，未用於訓練／選樣／調門檻", "items": [
                {"id": "normal-1", "path": "normal.png", "truth": "normal", "verified": True, "group_id": "lot-1"},
                {"id": "defect-1", "path": "defect.png", "truth": "defect", "verified": True, "group_id": "lot-2"}]}, ensure_ascii=False, indent=2), language="json")
        st.caption("完整框評估可另附 annotations_complete=true、boxes=[{class,box:[x1,y1,x2,y2]}]。只有影像真值時，結果只代表圖片警報率。")
    with st.form("fp_evaluation_" + case["id"]):
        uploaded, pasted, base_dir = None, '', ''
        if input_mode == '進階 JSON 清單':
            uploaded = st.file_uploader("評估清單 JSON", type=["json"])
            pasted = st.text_area("或貼上評估清單 JSON")
            base_dir = st.text_input("相對影像路徑的基準資料夾（絕對路徑可留空）")
        cols = st.columns(3)
        support = cols[0].number_input("正常／瑕疵各自最低支持數", 1, 5000, 10)
        miss = cols[1].number_input("容許漏檢率增加（百分點）", 0.0, 100.0, 0.0, step=.1)
        fp_drop = cols[2].number_input("要求誤逮率下降（百分點）", 0.0, 100.0, 0.0, step=.1)
        submit = st.form_submit_button("執行前後模型評估", type="primary")
    if submit:
        if input_mode == '進階 JSON 清單':
            manifest = json.loads(uploaded.getvalue().decode("utf-8-sig")) if uploaded else json.loads(pasted)
        if manifest is None:
            raise ValueError('請先逐張確認並建立評估清單')
        from fp_attribution import manager_for
        from fp_investigation import validate_evaluation_manifest
        from fp_cases import digest
        manifest,_=validate_evaluation_manifest(manifest,store,base_dir or None)
        payload={'case_id':case['id'],'manifest':manifest,'min_support':int(support),
                 'max_miss_increase':miss/100,'min_fp_reduction':fp_drop/100}
        if profiles:
            payload['profiles']=[{'id':p['id'],'revision':p['revision']} for p in [baseline,candidate]]
        else:
            payload['configs']=[case['runtime_profiles'][k] for k in ['baseline','candidate']]
        manager_for(store).submit('evaluate',payload,digest(json.dumps(payload,sort_keys=True).encode()),context={'case_id':case['id']})
        st.rerun()
    if case.get("evaluations"):
        result = st.selectbox("評估紀錄", list(reversed(case["evaluations"])), format_func=lambda r: r["at"][:19] + " · " + r["manifest"]["purpose"])
        labels = {"passed": "符合本次固定資料集驗收門檻", "failed": "未通過改善驗收", "insufficient": "支持數不足", "case_regression_only": "僅報案回歸結果"}
        st.write(labels[result["gate"]["status"]])
        for reason in result["gate"]["reasons"]:
            st.caption(reason)
        rows = [{"項目": label, "基準": result["baseline"][key], "候選": result["candidate"][key]} for key, label in
                [("normal_images", "正常影像數"), ("defect_images", "瑕疵影像數"), ("false_alarms", "誤逮數"),
                 ("misses", "漏檢數"), ("false_alarm_rate", "誤逮率（0–1）"), ("miss_rate", "漏檢率（0–1）"), ("abstentions", "拒判數")]]
        st.dataframe(rows, hide_index=True)
        if result["baseline"]["detection"]:
            st.json({"框層級基準": result["baseline"]["detection"], "框層級候選": result["candidate"]["detection"]})
        st.caption(result["limitation"])
        with st.expander("逐圖結果與群組支持數"):
            st.json({"baseline_groups": result["baseline"]["groups"], "candidate_groups": result["candidate"]["groups"], "per_image": result["per_image"]})
        st.download_button("下載前後評估 JSON", json.dumps(result, ensure_ascii=False, indent=2), file_name=result["id"]+".json", mime="application/json", on_click="ignore")


def render_training(store, case):
    st.markdown("**人工完整覆核後的訓練交接**")
    st.caption("此處產生獨立 YOLO 偵測資料包；不修改來源，也不替未知的外部訓練程式啟動重訓。取得新版模型後，可直接回「改善評估」驗證。")
    rows = training_rows(case)
    if not rows:
        st.info("先建立並逐項覆核爭議區域或候選。")
        return
    st.dataframe([{"區域 ID": r["id"], "影像": r["source"]["name"], "影像 SHA": r["source"]["sha256"],
                   "範圍": str(r["roi"]), "覆核": VERDICTS[r["review"]["verdict"]]} for r in rows], hide_index=True)
    sources = {r["source"]["sha256"]: r["source"]["name"] for r in rows}
    config=case.get("runtime_profiles",{}).get("baseline",{})
    if case.get("model_profile"):
        from model_profiles import ModelRegistry
        reference=case["model_profile"]
        config=ModelRegistry(store.root).get(reference["id"],reference["revision"])["runtime_config"]
    suggested=[n for n in config.get("class_names",[]) if n not in config.get("normal_classes",[])]
    names = st.text_input("訓練瑕疵類別順序（逗號分隔）", value=",".join(suggested) or "crack",key='training_classes_'+case['id'])
    class_names = [n.strip() for n in names.split(',') if n.strip()]
    advanced = st.checkbox('進階：直接貼上類別 JSON',key='training_advanced_'+case['id'])
    class_assignments = {}
    if not advanced:
        for row in rows:
            if row['review']['verdict'] == 'defect':
                chosen = st.selectbox('人工確認類別：'+row['source']['name']+' · '+str(row['roi']),['待選擇']+class_names,key='training_class_'+case['id']+row['id'])
                if chosen != '待選擇':
                    class_assignments[row['id']] = chosen
    with st.form("fp_training_" + case["id"]):
        selected = st.multiselect("已確認整張標註完整的影像", list(sources), format_func=lambda sha: sources[sha]+" · "+sha[:12])
        mapping = st.text_area("真瑕疵區域 → 人工確認類別 JSON", value="{}", help='例如 {"item:i_xxx":"crack"}；正常負樣本不需指定類別。') if advanced else None
        reviewer = st.text_input("完整標註確認者")
        reason = st.text_input("完整檢查與類別確認依據")
        confirmed = st.checkbox("我已檢查選取影像的全部內容，真瑕疵框無遺漏，且影像是可用的原始訓練內容")
        submit = st.form_submit_button("建立訓練交接 ZIP", type="primary")
    if submit:
        if not confirmed:
            raise ValueError("訓練交接需要明確確認整張影像與標註完整性")
        record, payload = export_training(case, store, complete_images=selected, class_assignments=json.loads(mapping) if advanced else class_assignments,
                         class_names=class_names, reviewer=reviewer, reason=reason)
        root = store.root / "training" / case["id"]
        root.mkdir(parents=True, exist_ok=True)
        path = root / (record["id"]+".zip")
        path.write_bytes(payload)
        record["zip_path"] = str(path)
        _save(store, case, "已建立人工完整覆核的訓練交接包")
    if case.get("training_exports"):
        result = st.selectbox("訓練交接紀錄", list(reversed(case["training_exports"])), format_func=lambda r: r["id"])
        st.write(f"影像 {result['n_images']} · 真瑕疵框 {result['n_boxes']} · 整圖正常負樣本 {result['n_negative_images']}")
        if result.get("zip_path") and Path(result["zip_path"]).exists():
            st.download_button("下載訓練交接 ZIP", Path(result["zip_path"]).read_bytes(), file_name=result["id"]+".zip", mime="application/zip", on_click="ignore")
