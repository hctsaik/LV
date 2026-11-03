"""Chinese false-positive cases with source-first model diagnostics and reviewed evidence."""
from __future__ import annotations

import copy
import io
import json
import time
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw
import streamlit as st

import fp_cases as domain
from fp_case_search import crop_image, record_search, search


def _commit(store, case, message="已儲存"):
    saved = store.save(case)
    st.session_state["fp_notice"] = message
    return saved


def _intake_bytes(upload, path):
    if upload is not None:
        return upload.getvalue(), upload.name, ""
    if path.strip():
        p = Path(path.strip()).expanduser().resolve()
        return p.read_bytes(), p.name, str(p)
    raise ValueError("請上傳圖片或填寫本機圖片路徑")


def _new_case(store):
    with st.expander("建立誤逮案件", expanded=not store.list()):
        with st.form("fp_new"):
            upload = st.file_uploader("報案圖片或截圖", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])
            path = st.text_input("或填寫本機圖片路徑", key="fp_new_path")
            kind = st.selectbox("圖片來源", ["original", "screenshot"], format_func=lambda x: "原圖" if x == "original" else "截圖")
            title = st.text_input("案件名稱（可留空）")
            description = st.text_area("使用者描述／認為正常的理由")
            a, b = st.columns(2)
            model = a.text_input("外部模型版本（可留空）")
            threshold = b.text_input("判定門檻（可留空）")
            raw = st.text_area("當時模型輸出或紀錄（可留空）")
            submitted = st.form_submit_button("建立案件", type="primary")
        if submitted:
            data, name, source_path = _intake_bytes(upload, path)
            source = store.ingest(data, name, kind=kind, source_path=source_path)
            duplicates = store.duplicates(source["sha256"])
            case = store.create(source, title, description, model, threshold, raw)
            st.session_state["fp_case_id"] = case["id"]
            # Keep the destination outside widget state: the early rerun can
            # clean up the previous case selector before it is rendered again.
            st.session_state["investigation_next_case"] = case["id"]
            st.session_state["fp_notice"] = "案件已建立" + (f"；已有 {len(duplicates)} 件相同影像，可在來源區查看。" if duplicates else "")
            st.rerun()


def _show_image(store, source, roi=None, caption="", *, preview_size=None):
    with Image.open(store.image_path(source)) as original:
        image = original.convert("RGB")
    if roi is not None:
        w, h = image.size
        xy = [roi[0]*w, roi[1]*h, min(w-1, roi[2]*w), min(h-1, roi[3]*h)]
        ImageDraw.Draw(image).rectangle(xy, outline="#ff4b4b", width=max(2, w//200))
    image.thumbnail(preview_size or (1100, 800))
    st.image(image, caption=caption, width=image.width if preview_size else "stretch")


def _source_form(store, case):
    with st.expander("補入原圖／其他來源"):
        with st.form("fp_source_" + case["id"]):
            upload = st.file_uploader("補入圖片", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])
            path = st.text_input("補入圖片本機路徑")
            kind = st.selectbox("補入來源種類", ["original", "screenshot"], format_func=lambda x: "原圖" if x == "original" else "截圖")
            submit = st.form_submit_button("儲存補入來源")
        if submit:
            data, name, original_path = _intake_bytes(upload, path)
            domain.add_source(case, store.ingest(data, name, kind=kind, source_path=original_path))
            _commit(store, case)
            st.rerun()


def _review_form(store, case, item, candidate=False):
    r = item["review"]
    key = f"fp_review_{case['id']}_{item['id']}_{candidate}"
    with st.form(key):
        verdict = st.selectbox("覆核結果", list(domain.VERDICTS), index=list(domain.VERDICTS).index(r["verdict"]), format_func=domain.VERDICTS.get)
        reviewer = st.text_input("覆核者", value=r["reviewer"])
        reason = st.text_area("判斷依據", value=r["reason"])
        done = st.form_submit_button("儲存此區域覆核")
    if done:
        domain.review_item(case, item["id"], verdict, reviewer, reason, candidate=candidate)
        _commit(store, case, "此區域覆核已儲存；其他區域維持原狀")
        st.rerun()


def _case_details(store, case):
    with st.expander("報案資訊／模型輸出", expanded=not case["items"]):
        with st.form("fp_context_" + case["id"]):
            prefix = "fp_context_draft_" + case["id"]
            for field, value in {"description": case["description"], "model": case["external"]["model_id"],
                                 "threshold": str(case["external"]["threshold"]), "raw": case["external"]["raw_output"]}.items():
                st.session_state.setdefault(prefix + field, value)
            desc = st.text_area("報案描述", key=prefix + "description")
            model = st.text_input("模型版本", key=prefix + "model")
            threshold = st.text_input("門檻", key=prefix + "threshold")
            raw = st.text_area("原始輸出紀錄", key=prefix + "raw")
            save = st.form_submit_button("儲存報案資訊")
        if save:
            domain.update_context(case, desc, model, threshold, raw)
            _commit(store, case)
            st.rerun()
    source_names = {"original": "原圖", "screenshot": "截圖", "reference": "對照圖"}
    source_index=next((j for j,s in enumerate(case["sources"]) if s["id"]==st.query_params.get("inv_source")),0)
    source = st.selectbox("目前影像來源", case["sources"],index=source_index, format_func=lambda s: f"{s['name']} · {source_names[s['kind']]}", key="fp_source_select_" + case["id"])
    st.query_params["inv_source"]=source["id"]
    from investigation_workspace import render_inference, graphical_regions, render_region_actions, render_case_import
    try:
        store.image_path(source)
    except (OSError, ValueError):
        st.warning("這張案件圖片遺失或內容已改變。請補回同一份檔案，恢復原有區域與證據的對應。")
        with st.form("fp_restore_source_" + case["id"] + source["id"]):
            replacement = st.file_uploader("補回原始檔案", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])
            restore = st.form_submit_button("驗證並修復案件圖片")
        if restore:
            if replacement is None:
                st.error("請先選擇要補回的圖片")
            else:
                try:
                    store.restore_source(source, replacement.getvalue())
                except (OSError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    st.session_state["fp_notice"] = "原始圖片已驗證並修復；案件區域與既有證據維持原有對應"
                    st.rerun()
        _source_form(store, case)
        return
    render_inference(store, case, source)
    if source["kind"] == "screenshot":
        st.info("依截圖比對，無法重現當時判定。框線、文字與縮放可能影響相似搜尋，可用局部區域避開。")
    duplicates = [c for c in store.duplicates(source["sha256"]) if c["id"] != case["id"]]
    if duplicates:
        st.caption("相同影像歷史案件：" + "；".join(c["title"] + " · " + c["id"] for c in duplicates))
    left, right = st.columns([1.15, 1])
    with left:
        drawn_roi = graphical_regions(store, case, source)
        st.caption(f"影像尺寸 {source['size'][0]} × {source['size'][1]}；來源 {source['source_path'] or '上傳副本'}")
    with right:
        st.markdown("**新增爭議區域**")
        st.caption("可在圖上圈選；未圈選時使用整張圖。未標註不代表正常。")
        coordinate_key="fp_coordinates_"+case["id"]+source["id"]
        drawing_identity=[drawn_roi,st.session_state.get("inv_picker_"+case["id"]+source["id"]+"_draw_nonce")]
        if st.session_state.get(coordinate_key+"_draw")!=drawing_identity:
            for j,v in enumerate(drawn_roi or domain.FULL_ROI):st.session_state[coordinate_key+str(j)]=float(v*100)
            st.session_state[coordinate_key+"_draw"]=drawing_identity
        with st.form("fp_region_" + case["id"] + source["id"]):
            with st.expander("精確座標（可選）"):
                cols = st.columns(4)
                defaults = [v*100 for v in (drawn_roi or domain.FULL_ROI)]
                values = [c.number_input(label, 0.0, 100.0, default, step=1.0,key=coordinate_key+str(j)) for j,(c,label,default) in
                          enumerate(zip(cols, ["左 %", "上 %", "右 %", "下 %"], defaults))]
            label = st.text_input("模型預測類別／爭議名稱")
            score = st.text_input("預測分數（未知可留空）")
            pred_id = st.text_input("預測框 ID（可留空）")
            add = st.form_submit_button("加入爭議區域", type="primary")
        if add:
            domain.add_item(case, source["id"], [v/100 for v in values], label,
                            None if not score.strip() else float(score), pred_id)
            _commit(store, case)
            st.rerun()
        with st.expander("匯入 YOLO 預測框"):
            st.caption("每行：class cx cy width height confidence；座標介於 0–1。匯入後仍為待覆核。")
            with st.form("fp_yolo_" + case["id"] + source["id"]):
                uploaded = st.file_uploader("YOLO 六欄預測檔", type=["txt"])
                pasted = st.text_area("或貼上預測內容")
                submit = st.form_submit_button("匯入預測區域")
            if submit:
                text = uploaded.getvalue().decode("utf-8-sig") if uploaded else pasted
                n = domain.import_yolo(case, source["id"], text)
                _commit(store, case, f"已匯入 {n} 個待覆核區域")
                st.rerun()
    _source_form(store, case)
    render_case_import(store, case)
    active = domain.active_rows(case["items"])
    if active:
        st.divider()
        by_id = {r["id"]: r for r in active}
        selection_key = "fp_review_selected_id_" + case["id"]
        if preferred := st.session_state.pop("inv_selected_item_" + case["id"], None):
            st.session_state[selection_key] = preferred
        if st.session_state.get(selection_key) not in by_id:
            st.session_state[selection_key] = st.query_params.get("inv_item") if st.query_params.get("inv_item") in by_id else next(iter(by_id))
        selected_id = st.selectbox("選擇爭議區域", list(by_id), key=selection_key,
                            format_func=lambda key: f"{by_id[key]['label'] or '未命名'} · {domain.VERDICTS[by_id[key]['review']['verdict']]}")
        item = by_id[selected_id]
        st.query_params["inv_item"]=selected_id
        selected_source = next(s for s in case["sources"] if s["id"] == item["source_id"])
        a, b = st.columns(2)
        with a:
            _show_image(store, selected_source, item["roi"], "紅框為本次覆核範圍；不代表整張圖的判定")
            origin={"yolo_prediction_file":"YOLO 預測檔","external_inference":"本次模型推論","imported_prediction":"匯入現場預測","imported_ground_truth":"匯入人工標註（仍須核對）"}.get(item["origin"],"人工指定區域")
            st.caption(f"預測分數：{item['score'] if item['score'] is not None else '未知'} · 來源：{origin}")
        with b:
            _review_form(store, case, item)
            render_region_actions(store, case, item)


def _comparison(store, case):
    active = domain.active_rows(case["items"])
    if not active:
        st.info("先在「案件與區域」加入一個爭議區域，再找對照。")
        return
    st.caption("選擇已上傳的正常／瑕疵資料集；也可使用本機資料夾。來源標籤仍須人工覆核。")
    with st.expander("搜尋設定", expanded=not case["searches"]), st.form("fp_search_" + case["id"]):
        preferred=st.session_state.get("inv_search_item_"+case["id"])
        item = st.selectbox("作為搜尋起點的區域", active, index=next((j for j,r in enumerate(active) if r["id"]==preferred),0), format_func=lambda i: f"{i['label'] or '未命名'} · {i['id']}")
        from investigation_data import DatasetRegistry
        datasets=DatasetRegistry(store).list()
        a,b=st.columns(2)
        normal_ds=a.selectbox("正常對照資料集",[None]+datasets,format_func=lambda d:"未選擇" if d is None else d["name"])
        defect_ds=b.selectbox("瑕疵對照資料集",[None]+datasets,format_func=lambda d:"未選擇" if d is None else d["name"])
        a, b = st.columns(2)
        normal = a.text_input("正常對照資料夾（可留空）")
        defect = b.text_input("真瑕疵對照資料夾（可留空）")
        scope = st.selectbox("對照圖片的比對範圍", ["whole_image", "same_roi"],
                             format_func=lambda x: "整張對照圖（適合已裁切物件）" if x == "whole_image" else "與報案相同比例位置（適合已對齊影像）")
        history = st.checkbox("包含已覆核歷史案件", value=True)
        a, b = st.columns(2)
        limit = a.number_input("每個來源最多搜尋圖片／區域數", 1, 5000, 300)
        top_k = b.number_input("每欄最多顯示對照數", 1, 100, 6)
        run = st.form_submit_button("找對照案例", type="primary")
    if run:
        from fp_attribution import manager_for,target_identity
        attribution_context=st.session_state.get("inv_search_context_"+case["id"])
        if attribution_context:
            from fp_attribution import validate_attribution_context
            try:
                validate_attribution_context(case,item,attribution_context)
            except ValueError:
                attribution_context=None
                st.session_state.pop("inv_search_context_"+case["id"],None)
        st.session_state.pop("inv_search_evidence_"+case["id"],None)
        payload={"case_id":case["id"],"item_id":item["id"],"target_identity":target_identity(case,item),
                 "normal_folder":normal,"defect_folder":defect,"candidate_scope":scope,"limit":int(limit),"top_k":int(top_k),
                 "include_history":history,"datasets":{"normal":normal_ds["id"] if normal_ds else None,"defect":defect_ds["id"] if defect_ds else None},
                 "attribution_evidence_id":attribution_context["attribution_evidence_id"] if attribution_context else None,
                 "attribution_context":attribution_context}
        job=manager_for(store).submit("comparison",payload,domain.digest(json.dumps(payload,sort_keys=True).encode()),context={"case_id":case["id"]})
        st.session_state["inv_comparison_job_"+case["id"]]=job["id"]
        st.rerun()
    from investigation_workspace import render_case_job
    render_case_job(store,case,"comparison")
    if not case["searches"]:
        st.info("按下「找對照案例」才開始計算；不需要訓練 LV 分類 head。")
        return
    result = st.selectbox("搜尋紀錄", list(reversed(case["searches"])),
                          format_func=lambda r: f"{r['at'][:19]} · {r['query_item_id']}", key="fp_search_result_" + case["id"])
    query_item = next(i for i in case["items"] if i["id"] == result["query_item_id"])
    query_source = next(s for s in case["sources"] if s["id"] == query_item["source_id"])
    preview_col, context_col = st.columns([1, 3])
    with preview_col:
        with Image.open(store.image_path(query_source)) as image:
            query_crop = crop_image(image, result["query_roi"])
        query_crop.thumbnail((380, 260))
        st.image(query_crop, caption="報案區域：本次搜尋起點", width="stretch")
    with context_col:
        st.markdown("**對照這個報案區域**")
        st.write(f"{query_source['name']} · 預測 {query_item['label'] or '未知'} · 分數 {query_item['score'] if query_item['score'] is not None else '未知'}")
        st.write(domain.VERDICTS[query_item["review"]["verdict"]])
        st.caption("下方同時呈現正常、瑕疵與歷史對照；可加入支持或反駁目前假設的案例。")
        if query_source["kind"] == "screenshot":
            st.caption("來源為截圖；外部模型原始輸入與判定尚未重現。")
    st.caption("LV cosine 相似度只表示這個特徵空間中的外觀接近程度，不是誤逮機率或外部模型信心。")
    with st.expander("搜尋範圍與可比條件"):
        st.json({k: result[k] for k in ["query_roi", "candidate_scope", "recipe", "sources", "skipped"]})
    if any(s.get("truncated") for s in result["sources"].values()):
        st.warning("部分來源超過搜尋上限；目前結果不代表整個資料庫。可縮小資料夾或提高上限。")
    if result["skipped"]:
        st.warning(f"有 {len(result['skipped'])} 個檔案無法比對；詳細原因見搜尋範圍。")
    for col, category, title in zip(st.columns(3), ["normal", "defect", "history"], ["正常對照", "真瑕疵對照", "歷史案件"]):
        with col:
            st.markdown(f"**{title}**")
            rows = result["groups"][category]
            if not rows:
                st.info("本次搜尋範圍內沒有可用對照。")
            for row in rows:
                with st.container(border=True):
                    try:
                        data = Path(row["path"]).read_bytes()
                        if domain.digest(data) != row["sha256"]:
                            raise ValueError("影像內容已變更，請重新搜尋")
                        with Image.open(io.BytesIO(data)) as image:
                            preview = crop_image(image, row["roi"])
                            preview.thumbnail((380, 300))
                        st.image(preview, width="stretch")
                    except (OSError, ValueError) as exc:
                        st.warning(str(exc))
                    st.write(row.get("name") or Path(row["path"]).name)
                    status_label = {"supplied_folder_label": "資料夾提供標籤（未覆核）",
                                    "reviewed_normal": "歷史：已確認此區域正常",
                                    "reviewed_defect": "歷史：已確認有瑕疵", "supplied_dataset_label":"資料集提供標籤（未覆核）"}[row["label_status"]]
                    st.caption(f"cosine {row['similarity']:.3f} · {status_label}")
                    if row["label_conflict"]:
                        st.warning("同一區域同時出現在正常與瑕疵來源，請優先確認標籤。")
                    if category == "history":
                        st.caption(f"歷史案件 {row['historical_case']} · 外部模型 {row['external_model']}")
                        st.caption("歷史原因不會套用到本案；請核對模型版本與情境。")
                    with st.expander("來源與證據"):
                        st.write(row["path"])
                        st.code(row["evidence_id"])
                    already = any(c["id"] == row["id"] for c in case["candidates"])
                    if st.button("已加入待覆核" if already else "加入待覆核", disabled=already,
                                 key=f"fp_add_{case['id']}_{result['id']}_{category}_{row['id']}"):
                        domain.add_candidate(case, row, store)
                        _commit(store, case, "已加入候選，仍須逐區域覆核")
                        st.rerun()


def _explanation(store, case):
    from investigation_data_page import render_reply_editor
    saved = render_reply_editor(store, case)
    if saved and saved["revision"] != case["revision"]:
        st.rerun()
    st.caption("把確認事實、待驗證假設與下一步分開。引用 LV 對照時，只能支持外觀線索，不能證明外部模型根因。")
    choices = [None] + case["claims"]
    editing = st.selectbox("新增或修改說明", choices, format_func=lambda c: "新增一段說明" if c is None else f"{domain.KINDS[c['kind']]}：{c['text'][:55]}")
    edit_id = editing["id"] if editing else "new"
    by_id = {e["id"]: e for e in case["evidence"]}
    with st.form(f"fp_claim_{case['id']}_{edit_id}"):
        kinds = list(domain.KINDS)
        kind = st.selectbox("說明類型", kinds, index=kinds.index(editing["kind"]) if editing else 0, format_func=domain.KINDS.get)
        text = st.text_area("說明內容", value=editing["text"] if editing else "")
        refs = st.multiselect("引用證據", list(by_id), default=editing["refs"] if editing else [],
                              format_func=lambda key: f"{key} · {by_id[key]['text']} · {by_id[key]['details'].get('item_id', by_id[key]['details'].get('path', ''))}")
        save = st.form_submit_button("儲存說明")
    if save:
        domain.add_claim(case, kind, text, refs, editing["id"] if editing else None)
        _commit(store, case)
        st.rerun()
    with st.expander("檢視證據詳細內容"):
        chosen = st.selectbox("證據紀錄", list(by_id), format_func=lambda key: f"{key} · {by_id[key]['text']}")
        st.json(by_id[chosen])
    replied = st.checkbox("已回覆使用者", value=case["reply_status"] == "replied", key="fp_replied_" + case["id"])
    if replied != (case["reply_status"] == "replied"):
        case["reply_status"] = "replied" if replied else "draft"
        _commit(store, case)
        st.rerun()
    st.caption("已回覆不等於原因已確認。外部推論、對照實驗與改善驗收均保留各自的證據及適用範圍。")
    report = domain.report_markdown(case)
    with st.expander("案件說明預覽", expanded=True):
        st.markdown(report)
    a, b = st.columns(2)
    a.download_button("下載案件說明 Markdown", report, file_name=case["id"] + ".md", mime="text/markdown", on_click="ignore")
    b.download_button("下載案件 JSON", json.dumps(case, ensure_ascii=False, indent=2),
                      file_name=case["id"] + ".json", mime="application/json", on_click="ignore")


def _review_zip(store, manifest, provenance=None):
    output = io.BytesIO()
    portable = copy.deepcopy(manifest)
    # The editable return contract only needs source identity and review fields.
    # Local acquisition paths/history are not part of the portable handoff.
    portable['external_model'] = {k:v for k,v in portable.get('external_model',{}).items() if k in {'model_id','threshold'}}
    portable['items'] = [{k:v for k,v in row.items() if k in {'id','source','roi','review','reason','reference_category','evidence_id','withdrawn'}} for row in portable['items']]
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        added = set()
        for row in portable["items"]:
            source = row["source"]
            asset = "assets/" + source["asset"]
            row["review_image"] = asset
            if asset not in added:
                bundle.write(store.image_path(source), asset)
                added.add(asset)
            row['source'] = {k:v for k,v in source.items() if k in {'id','asset','sha256','size','name','kind','original_kind'}}
        bundle.writestr("review_manifest.json", json.dumps(portable, ensure_ascii=False, indent=2))
        if provenance is not None:
            bundle.writestr("provenance.json", json.dumps(provenance, ensure_ascii=False, indent=2))
        bundle.writestr("README.txt", "逐區域覆核：修改 review_manifest.json 中 items[].review 的 verdict/reviewer/reason。\n"
                        "verdict: pending, normal, defect, disputed, insufficient。保持身分欄位不變。\n"
                        "圖片位於 assets/；roi 是 normalized xyxy。將 JSON 回傳 LV「覆核批次」。\n"
                        "provenance.json 記錄選樣證據與當次來源／區域／覆核快照；歷史熱圖或相似度不代表目前真值。\n"
                        "此包不是訓練資料；正常 ROI 不代表整張圖正常。\n")
    return output.getvalue()


def _batches(store, case):
    candidates=domain.active_rows(case["candidates"])
    counts = domain.review_counts(candidates)
    st.write(f"候選 {counts['total_regions']} 個區域 · 已覆核 {counts['reviewed_regions']} · 待覆核 {counts['pending_regions']}")
    st.caption("覆核區域不計為已標註訓練影像。候選來源標籤不會自動成為覆核結果。")
    if candidates:
        row = st.selectbox("待覆核候選", candidates,
                           format_func=lambda r: f"{r['source']['name']} · {r['id']} · {domain.VERDICTS[r['review']['verdict']]}")
        a, b = st.columns(2)
        with a:
            _show_image(store, row["source"], row["roi"], "本次覆核範圍")
            st.caption(row["reason"])
        with b:
            _review_form(store, case, row, candidate=True)
            reason=st.text_input("撤回候選原因",key="inv_candidate_reason")
            if st.button("撤回候選"):
                domain.revise_region(case,row["id"],candidate=True,withdraw=True,reason=reason)
                _commit(store,case);st.rerun()
        if st.button("建立／連結 Active Learning 覆核回合", type="primary"):
            batch_id = domain.prepare_batch(case)
            case = _commit(store, case)
            domain.link_round(case, batch_id)
            _commit(store, case, "已連結覆核回合；尚未寫入任何訓練標籤")
            st.rerun()
    if case["batches"]:
        batch = st.selectbox("覆核批次", list(reversed(case["batches"])), format_func=lambda b: b["id"])
        st.caption(f"回合：{batch['round_id'] or '未連結，請重試上方連結按鈕'}")
        manifest = domain.review_manifest(case, batch["id"])
        st.download_button("下載覆核清單 JSON", json.dumps(manifest, ensure_ascii=False, indent=2),
                           file_name=batch["id"] + ".json", mime="application/json", on_click="ignore")
        if st.button("準備含圖片的覆核包"):
            st.session_state["fp_zip_" + case["id"]] = (case["revision"], batch["id"], _review_zip(store, manifest, domain.review_provenance(case,batch["id"])))
        ready = st.session_state.get("fp_zip_" + case["id"])
        if ready and ready[:2] == (case["revision"], batch["id"]):
            st.download_button("下載覆核包 ZIP", ready[2], file_name=batch["id"] + ".zip", mime="application/zip", on_click="ignore")
        returned = st.file_uploader("回傳已覆核清單 JSON", type=["json"],
                                    key="fp_return_" + case["id"] + batch["id"])
        submit = st.button("驗證並讀回覆核", disabled=returned is None)
        if submit:
            if returned is None:
                raise ValueError("請先選取回傳 JSON")
            domain.apply_review_return(case, json.loads(returned.getvalue().decode("utf-8-sig")))
            _commit(store, case, "覆核已讀回；未更新訓練標籤或模型成績")
            st.rerun()


def _effort(store, case):
    with st.expander(f"調查工時：已記錄 {case['effort_minutes']:.1f} 分鐘"):
        key = "fp_timer_" + case["id"]
        running = st.session_state.get(key)
        st.caption("手動開始／暫停；切換頁面後仍會計時，離開前請暫停。未暫停的時間不自動入帳，可補登或修正。")
        if running is None:
            if st.button("開始調查計時"):
                st.session_state[key] = time.monotonic()
                st.rerun()
        else:
            st.info("調查計時中")
            if st.button("暫停並記錄工時"):
                domain.record_effort(case, (time.monotonic() - running)/60, note="手動計時")
                _commit(store, case)
                st.session_state.pop(key, None)
                st.rerun()
        with st.form("fp_effort_" + case["id"]):
            minutes = st.number_input("修正後累計分鐘數", min_value=0.0, value=float(case["effort_minutes"]), step=1.0)
            reason = st.text_input("補登／修正原因")
            save = st.form_submit_button("儲存修正工時")
        if save:
            domain.record_effort(case, minutes, mode="set", note=reason)
            _commit(store, case)
            st.rerun()
        with st.form("inv_effort_protocol_"+case["id"]):
            st.caption("固定案例比較：兩個流程使用同一研究、配對、案例群組與調查者；記錄執行先後，並由另一位覆核者檢查品質。未完成的紀錄保留，但不列入時間差比較。")
            study = st.text_input("研究編號")
            pair = st.text_input("配對編號")
            group = st.text_input("案例群組")
            reviewer = st.text_input("調查者")
            setup=st.number_input("首次設定分鐘",min_value=0.0,step=1.0)
            st.caption("請填這一次流程的操作時間；上方累計計時可能包含多次調查，不自動代入配對測量。")
            active=st.number_input("本次實際操作分鐘",min_value=0.0,value=0.0,step=1.0)
            waiting=st.number_input("本次等待運算分鐘",min_value=0.0,step=1.0)
            repeated=st.number_input("重複輸入次數",min_value=0,step=1)
            wrong=st.number_input("選錯目標重作次數",min_value=0,step=1)
            returned=st.number_input("證據不足退回次數",min_value=0,step=1)
            condition=st.selectbox("測量流程",["原有流程","LV 查案流程"])
            order=st.selectbox("配對中的執行順序",[1,2])
            completed=st.checkbox("本次調查已完成")
            quality_reviewer=st.text_input("獨立品質覆核者")
            quality_reviewed=st.checkbox("已完成獨立品質覆核")
            errors=st.number_input("覆核發現的錯誤數",min_value=0,step=1)
            unsupported=st.number_input("缺乏證據的原因陳述數",min_value=0,step=1)
            if st.form_submit_button("記錄本次固定案例測量"):
                case.setdefault("effort_measurements",[]).append({"schema_version":2,"at":domain.utc_now(),"condition":condition,
                    "study_id":study.strip(),"pair_id":pair.strip(),"case_group":group.strip(),"reviewer":reviewer.strip(),
                    "active_minutes":active,"waiting_minutes":waiting,"order":order,"completed":completed,
                    "quality_reviewer":quality_reviewer.strip(),"quality_reviewed":quality_reviewed,
                    "quality_errors":errors,"unsupported_claims":unsupported,
                    "setup_minutes":setup,"case_minutes":case["effort_minutes"],"repeated_inputs":repeated,
                    "wrong_target_rework":wrong,"evidence_returns":returned})
                _commit(store,case);st.rerun()
        from explanation_validation import compare_effort
        measurement = compare_effort(case.get("effort_measurements", []))
        if measurement["pairs"]:
            st.write(f"已配對 {measurement['matched_pairs']} 組；原有流程減 LV 的平均操作時間差：{measurement['mean_active_minutes_difference']:.1f} 分鐘。正值表示本批 LV 操作時間較少。")
            st.write(f"品質未變差：{measurement['quality_not_worse_pairs']}／{measurement['matched_pairs']} 組。")
            st.dataframe([{"配對":p["pair_id"],"調查者":p["reviewer"],"操作時間差（分鐘）":p["active_minutes_difference"],
                           "等待時間差（分鐘）":p["waiting_minutes_difference"],"品質未變差":p["quality_not_worse"]} for p in measurement["pairs"]],hide_index=True)
            if not measurement["order_balanced"]:
                st.caption("目前未涵蓋兩種執行先後，可能受到熟悉案例的影響。")
        else:
            st.caption("尚無完成配對且通過獨立覆核的工時紀錄，無法估計操作時間差。")
        if measurement["excluded"]:
            st.caption(f"另有 {len(measurement['excluded'])} 筆歷史、未完成或未配對紀錄，未納入比較。")
        if case.get("effort_measurements"):
            with st.expander("查看工時與品質原始紀錄"):
                st.json({"records":case["effort_measurements"],"comparison":measurement})
                with st.form("effort_correction_"+case["id"]):
                    index=st.selectbox("修正測量紀錄",list(range(len(case["effort_measurements"]))),
                        format_func=lambda i:f"{i+1} · {case['effort_measurements'][i].get('condition','')} · {case['effort_measurements'][i].get('pair_id','未配對')}")
                    correction=st.text_input("撤回／恢復測量原因")
                    action=st.selectbox("測量修正操作",["撤回","恢復"])
                    if st.form_submit_button("儲存測量修正"):
                        if not correction.strip():
                            st.error("請填寫測量修正原因")
                        else:
                            record=case["effort_measurements"][index]
                            record["withdrawn"]=action=="撤回"
                            record.setdefault("corrections",[]).append({"at":domain.utc_now(),"action":action,"reason":correction.strip()})
                            _commit(store,case);st.rerun()


def render_case_workspace():
    from investigation_workspace import render_navigation, render_case_summary, render_case_import, workspace_store, render_case_version_notice
    store = workspace_store()
    if render_navigation(store):
        return
    st.subheader("誤逮案件診斷")
    st.caption("從報案找對照、留下證據，再把同類問題送入覆核。LV 輔助分析與外部模型判定分開記錄。")
    try:
        if notice := st.session_state.pop("fp_notice", None):
            st.success(notice)
        _new_case(store)
        render_case_import(store)
        cases = store.list()
        if not cases:
            st.info("先建立第一個案件。只有截圖或缺少模型版本也可以開始。")
            return
        with st.expander("搜尋／篩選案件"):
            query = st.text_input("搜尋案件名稱或描述")
            status = st.selectbox("回覆狀態", ["全部", "待回覆", "已回覆"])
        visible = [c for c in cases if (not query or query.casefold() in (c["title"]+" "+c["description"]).casefold())
                   and (status == "全部" or (c["reply_status"] == "replied") == (status == "已回覆"))]
        if not visible:
            st.info("沒有符合條件的案件，請調整篩選。")
            return
        names = {c["id"]: f"{c['title']} · {c['id']}" for c in visible}
        if st.session_state.get("fp_case_id") not in names:
            st.session_state["fp_case_id"] = visible[0]["id"]
        case_id = st.selectbox("開啟案件", list(names), format_func=names.get, key="fp_case_id")
        case = store.load(case_id)
        render_case_version_notice(store, case_id)
        render_case_summary(store, case)
        runtime_note = f"已有 {len(case.get('external_runs', []))} 次外部推論，詳見各筆歷史比對結果" if case.get("external_runs") else "尚未重播外部判定"
        st.caption(f"版本 {case['revision']} · 報案模型 {case['external']['model_id'] or '未知'} · {runtime_note}")
        _effort(store, case)
        section = st.segmented_control("案件步驟", ["案件與區域", "找對照", "外部模型", "說明與回覆", "覆核批次", "訓練交接", "改善評估"],
                                       default="案件與區域", key="fp_section_" + case_id) or "案件與區域"
        if section in {"外部模型", "改善評估", "訓練交接"}:
            from fp_external_page import render_external, render_evaluation, render_training
            {"外部模型": render_external, "改善評估": render_evaluation, "訓練交接": render_training}[section](store, case)
        else:
            {"案件與區域": _case_details, "找對照": _comparison,
             "說明與回覆": _explanation, "覆核批次": _batches}[section](store, case)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        st.error(f"無法完成操作：{exc}")
        st.caption("已儲存的案件仍保留。請修正輸入或重新開啟案件後再試。")
