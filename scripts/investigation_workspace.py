"""Engineer-facing navigation, source-first inference and the shared detector diagnostics view."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import zipfile

import streamlit as st

import fp_cases as domain
from explanation_summary import FILL_LABELS, PARITY_LABELS
from fp_attribution import (manager_for, completed_result, attach_result, select_detection,
                             target_identity, profile_config)


def workspace_store():
    default=domain.CaseStore()
    selected=st.session_state.get("investigation_workspace_root")
    name=st.query_params.get("inv_workspace")
    if not selected and name:
        path=(default.root.parent/name).resolve()
        if Path(name).name!=name or path.parent!=default.root.parent or not name.startswith(default.root.name+"_restored_") or not (path/"restore_receipt.json").is_file():
            st.error("還原工作區位置無效，請由模型庫開啟已還原工作區")
            st.stop()
        selected=str(path)
        st.session_state["investigation_workspace_root"]=selected
    if selected:
        st.query_params["inv_workspace"]=Path(selected).name
    store = domain.CaseStore(selected) if selected else default
    return _guard_case_session(store)


def _case_version_key(store):
    return "inv_case_versions_" + domain.digest(str(store.root).encode())[:16]


def _guard_case_session(store):
    """Keep form defaults on the version this tab observed, not a silent fresh load."""
    key = _case_version_key(store)
    versions = st.session_state.setdefault(key, {})

    def observe(value):
        entry = versions.setdefault(value["id"], {"observed": copy.deepcopy(value), "latest_revision": value["revision"]})
        entry["latest_revision"] = value["revision"]
        # A stale tab must retain its original widget defaults and unsaved inputs.
        return copy.deepcopy(entry["observed"])

    def before(value):
        if value["revision"] == 0:
            return
        entry = versions.get(value["id"])
        if entry and (entry["latest_revision"] != entry["observed"]["revision"]
                      or value["revision"] != entry["observed"]["revision"]):
            raise ValueError("案件已在另一個視窗更新；此頁草稿已保留，請先開啟最新版本並核對後再儲存")

    def after(value):
        versions[value["id"]] = {"observed": copy.deepcopy(value), "latest_revision": value["revision"]}

    store.on_load, store.before_save, store.after_save = observe, before, after
    return store


def _open_latest_case(root, case_id):
    # Explicit user action advances this tab's baseline; no draft widget keys are cleared.
    plain = domain.CaseStore(root)
    latest = plain.load(case_id)
    st.session_state.setdefault(_case_version_key(plain), {})[case_id] = {
        "observed": copy.deepcopy(latest), "latest_revision": latest["revision"], "draft_preserved": True}


def render_case_version_notice(store, case_id):
    entry = st.session_state.get(_case_version_key(store), {}).get(case_id)
    if not entry:
        return
    stale = entry["latest_revision"] != entry["observed"]["revision"]
    if stale:
        st.warning(f"另一個視窗已儲存版本 {entry['latest_revision']}；此頁仍保留版本 {entry['observed']['revision']} 的草稿，尚未覆寫任何資料。")
        st.button("開啟最新版本（保留此頁草稿）", on_click=_open_latest_case,
                  args=(str(store.root), case_id), key="inv_open_latest_" + case_id)
    elif entry.get("draft_preserved"):
        st.info("已開啟最新版本；此頁草稿仍保留，請與目前已儲存內容核對後再儲存。")
    if stale or entry.get("draft_preserved"):
        latest = domain.CaseStore(store.root).load(case_id)
        with st.expander("目前已儲存的報案資訊", expanded=True):
            st.write({"版本": latest["revision"], "報案描述": latest["description"],
                      "模型版本": latest["external"]["model_id"], "門檻": latest["external"]["threshold"],
                      "原始輸出紀錄": latest["external"]["raw_output"]})


def go_area(area, case_id=None, section=None):
    st.session_state["investigation_next_area"] = area
    if case_id:
        st.session_state["investigation_next_case"] = case_id
    if section:
        st.session_state["investigation_next_section"] = section


def _navigate_diagnostics(case_id, item_id, source_id=None):
    st.session_state["investigation_diagnostic"] = {"case_id":case_id,"item_id":item_id}
    st.session_state["tool_switch"] = "Model Diagnostics"
    st.query_params["diagnostic_case"] = case_id
    st.query_params["diagnostic_item"] = item_id
    st.query_params["inv_item"]=item_id
    if source_id:st.query_params["inv_source"]=source_id


def render_navigation(store):
    if st.session_state.get("investigation_workspace_root"):
        st.caption("目前使用獨立還原工作區")
        if st.button("返回原工作區"):
            st.session_state.pop("investigation_workspace_root",None)
            st.query_params.pop("inv_workspace",None)
            for key in list(st.session_state):
                if key.startswith(("inv_","model_","fp_case_id")):st.session_state.pop(key,None)
            st.rerun()
    if area := st.session_state.pop("investigation_next_area", None):
        st.session_state["investigation_area"] = area
    if case_id := st.session_state.pop("investigation_next_case", None):
        st.session_state["fp_case_id"] = case_id
    if "fp_case_id" not in st.session_state and st.query_params.get("inv_case"):
        st.session_state["fp_case_id"] = st.query_params["inv_case"]
    area = st.segmented_control("查案工作區",["案件","我的模型","資料庫","工作紀錄"],
                                default="案件",key="investigation_area") or "案件"
    if area == "案件":
        return False
    try:
        if area == "我的模型":
            from investigation_models_page import render_model_library
            render_model_library(store,manager_for(store))
        elif area == "資料庫":
            from investigation_data_page import render_data_library
            render_data_library(store)
        else:
            render_jobs(store)
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:
        st.error(f"無法完成操作：{exc}")
        st.caption("已儲存的模型、案件與完整結果仍保留。修正後可重試。")
    return True


def _job_status(store,job_id,*,key):
    manager=manager_for(store)
    @st.fragment(run_every="2s")
    def panel():
        job=manager.store.get(job_id)
        labels={"queued":"等待執行","running":"執行中","succeeded":"已完成","failed":"失敗",
                "cancelled":"已取消","interrupted":"中斷，可重試"}
        st.caption(f"{labels[job['state']]} · {job.get('stage','')}")
        if job["state"] in {"queued","running"}:
            st.progress(float(job.get("progress",0)))
            if st.button("取消工作",key=key+"_cancel"):
                manager.cancel(job_id)
                st.rerun()
        elif job["state"]=="succeeded":
            if st.session_state.get(key+"_seen") != job_id:
                st.session_state[key+"_seen"]=job_id
                st.rerun()
        else:
            st.error(job.get("error") or "此工作未完成，原始案件仍保留。")
            if st.button("重試這項工作",key=key+"_retry"):
                fresh=manager.retry(job_id)
                st.session_state[key+"_retry_job"]=fresh["id"]
                if key.startswith("inv_job_"):st.session_state["inv_job_next"]=fresh["id"]
                st.rerun()
    panel()


def render_case_job(store,case,task):
    manager=manager_for(store)
    jobs=[j for j in manager.store.list(case_id=case["id"]) if j["task"]==task]
    if not jobs:return
    key="inv_"+task+"_job_"+case["id"]
    job_id=st.session_state.pop(key+"_retry_job",None) or st.session_state.get(key,jobs[0]["id"])
    st.session_state[key]=job_id
    _job_status(store,job_id,key=key)
    job=manager.store.get(job_id)
    if job["state"]=="succeeded" and job.get("publication",{}).get("state","not_published")=="not_published":
        try:
            attach_result(store,case["id"],completed_result(store,job_id))
            manager.store.mark_published(job_id,case_id=case["id"])
            if task=="comparison":st.session_state["fp_notice"]="對照結果與來源已儲存"
            st.rerun()
        except ValueError as exc:
            manager.store.mark_published(job_id,case_id=case["id"],error=str(exc))
            st.error(str(exc))


def render_jobs(store):
    st.subheader("工作紀錄")
    manager=manager_for(store)
    jobs=manager.store.list()
    if not jobs:
        st.info("尚未執行模型工作。先建立案件或匯入模型即可開始。")
        return
    by_id={j["id"]:j for j in jobs}
    if pending:=st.session_state.pop("inv_job_next",None):st.session_state["inv_job_selected"]=pending
    task_names={"inspect":"模型檢查","infer":"圖片推論","gradcam":"框歸因","batch_gradcam":"批次歸因","evaluate":"改善評估","experiment":"對照實驗","comparison":"找對照"}
    job_id=st.selectbox("查看工作",list(by_id),format_func=lambda k:f"{by_id[k]['created_at'][:19]} · {task_names.get(by_id[k]['task'],by_id[k]['task'])} · {by_id[k]['state']}",key="inv_job_selected")
    job=by_id[job_id]
    _job_status(store,job["id"],key="inv_job_"+job["id"])
    current=manager.store.get(job["id"])
    if current["state"]=="succeeded":
        result=completed_result(store,job["id"])
        st.success("計算結果已保存；加入案件與計算完成分開記錄。")
        if current.get("publication"):
            publication=current["publication"]
            st.caption({"not_published":"尚未加入案件","published":"已加入案件","conflict":"加入案件時發生衝突，計算結果仍保存"}.get(publication.get("state"),"可核對案件保存狀態"))
        if current["task"]=="batch_gradcam":
            _render_cohort_summary(result.get("cohort_summary"))
            if result.get('cohort_summary') and current.get('context',{}).get('case_id'):
                if st.button('保存跨批次核對摘要',key='inv_cohort_save_'+job['id']):
                    from explanation_summary import save_cohort_summary
                    try:
                        cohort_case=store.load(current['context']['case_id'])
                        save_cohort_summary(cohort_case,job['id'],result['cohort_summary'])
                        store.save(cohort_case)
                        st.success('已保存歷史批次證據，可在回覆中選取引用。')
                    except (ValueError,OSError) as exc:
                        st.error(str(exc))
            st.write(f"完成 {result['succeeded']}／{result['support']} 個目標；失敗項目不列入有效樣本。")
            st.dataframe([{"目標":r["item_id"],"狀態":r["status"],"原因":r.get("error","")} for r in result["items"]],hide_index=True)
            summary=result.get("summary",{})
            st.dataframe([{"遮擋比例":v.get('identity',{}).get('fraction',k),"填補":v.get('identity',{}).get('fill_method','舊協定'),"模型":v.get('identity',{}).get('model_sha256','未知')[:12],"執行環境":v.get('identity',{}).get('execution',{}).get('runtime_id','未知'),"Head":v.get('identity',{}).get('execution',{}).get('head_path','未知'),"目標支持數":v["support"],"熱區減分－控制平均":v["mean"],"樣本標準差":v["std"]} for k,v in summary.get("fractions",{}).items()],hide_index=True)
            st.caption(summary.get("limitation",""))
            st.caption(f"有效結果涵蓋 {summary.get('unique_images',0)} 張不同影像。")
            if result.get("localization_groups"):
                with st.expander("依類別與目標面積分組的定位檢查"):
                    st.dataframe([{"群組":k,"目標數":v["targets"],"有定位真值":v["localized_targets"],"熱點命中":v["pointing_hits"]} for k,v in result["localization_groups"].items()],hide_index=True)
                    st.caption("未提供有效 mask 的目標不列為定位通過或失敗。")
            if any(r["status"] in {"queued","running"} for r in result["items"]):
                st.button("更新批次進度")
                if st.button("取消尚未完成的批次目標"):
                    for row in result["items"]:
                        if row["status"] in {"queued","running"}:manager.cancel(row["job_id"])
                    st.rerun()
            failed={r["item_id"] for r in result["items"] if r["status"] in {"failed","interrupted","cancelled"}}
            if failed and st.button("只重試失敗目標"):
                for row in result["items"]:
                    if row["item_id"] in failed:manager.retry(row["job_id"])
                st.rerun()
            if st.button("將完成的目標逐筆加入案件"):
                for row in result["items"]:
                    if row["status"]!="succeeded":continue
                    try:
                        value=row["result"]
                        attach_result(store,value["attribution"]["case_id"],value)
                    except ValueError as exc:
                        st.warning(f"{row['item_id']}：{exc}")
                st.success("可相容的結果已保存；衝突目標保留工作結果供核對。")
        case_id=current.get("context",{}).get("case_id")
        if case_id and any(k in result for k in ["run","attribution","evaluation","comparison","experiment"]):
            if st.button("重新核對並加入案件"):
                try:
                    saved=attach_result(store,case_id,result)
                    manager.store.mark_published(job["id"],case_id=case_id)
                    go_area("案件",case_id)
                    st.rerun()
                except ValueError as exc:
                    manager.store.mark_published(job["id"],case_id=case_id,error=str(exc))
                    st.error(str(exc))
        with st.expander("工作技術資訊"):
            st.json(result)


def _render_cohort_summary(summary):
    if not summary:
        st.info('這份批次結果尚無獨立影像／群組核對資料。')
        return
    st.write(f"不同影像 {summary.get('unique_images',0)} 張；已核實 {summary.get('verified_images',0)} 張，人工宣告群組 {summary.get('unique_groups',0)} 組。")
    st.caption('不同模型、head、方法與控制協定分開核對；群組宣告不等於已證明獨立性，也不代表現場效果。')
    rows=[]
    for partition in summary.get('partitions',[]):
        for role,counts in partition.get('roles',{}).items():
            rows.append({'執行組別':partition.get('identity_key','')[:12],'樣本角色':role,'不同影像':counts['unique_images'],'人工群組':counts['unique_groups']})
    if rows:st.dataframe(rows,hide_index=True)
    contrasts=[]
    for partition in summary.get('partitions',[]):
        for row in partition.get('contrasts',{}).get('rows',[]):
            contrasts.append({'執行組別':partition.get('identity_key','')[:12],'角色':row['role'],'遮擋比例':row['fraction'],'填補':row['fill_method'],'不同影像':row['unique_images'],'人工群組':row['unique_groups'],'群組平均相對減分':row['mean_group_excess_drop']})
    if contrasts:
        st.dataframe(contrasts,hide_index=True)
        st.caption('先去重指定候選，再於每張影像、每個人工群組內平均；此為描述性分數差異，不是獨立性或根因證明。')
    if summary.get('excluded'):st.warning(f"{len(summary['excluded'])} 個目標因資料未核實、衝突或執行身分不完整而排除。")


def render_case_summary(store,case):
    st.query_params["inv_case"]=case["id"]
    if section:=st.session_state.pop("investigation_next_section",None):
        st.session_state["fp_section_"+case["id"]]=section
    missing=[]
    if not any(s["kind"]=="original" for s in case["sources"]):
        missing.append("原圖")
    if not case.get("model_profile") and not case.get("runtime_profiles"):
        missing.append("模型")
    if not domain.active_rows(case["items"]):
        action="先選擇圖片推論，或圈選爭議區域；也可先整理待補資料回覆。"
    elif any(i["review"]["verdict"]=="pending" for i in domain.active_rows(case["items"])):
        action="先逐區域確認現場規格，再選擇對照或模型分析。"
    elif case["reply_status"]!="replied":
        action="可整理已有證據與尚待確認項目，產生回覆。"
    else:
        action="已回覆；仍可補證據、覆核候選與驗證改善。"
    st.info(action+("　待補："+"、".join(missing)+"（不阻擋保存或回覆）" if missing else ""))
    a,b,c=st.columns(3)
    a.button("選擇／匯入模型",on_click=go_area,args=("我的模型",case["id"]),key="inv_model_cta_"+case["id"])
    b.button("準備圖文回覆",on_click=go_area,args=("案件",case["id"],"說明與回覆"),key="inv_reply_cta_"+case["id"])
    c.button("參考圖片與資料集",on_click=go_area,args=("資料庫",case["id"]),key="inv_data_cta_"+case["id"])
    targets=[i for i in domain.active_rows(case["items"]) if i.get("external_target")]
    if targets:
        with st.expander("批次檢查已選模型框"):
            st.caption("批次群組由人工宣告（例如不同生產批號）；同圖多框不算獨立影像。角色與群組須經人工核實，未確認資料不列入跨批次支持。")
            st.caption('這裡只核對已有指定模型框的樣本；沒有偵測框的正常圖請使用資料庫改善評估，不列為 CAM 支持。')
            cohort_item=st.selectbox("設定批次核對資料的目標",targets,format_func=lambda i:f"{i['label']} · {i['id']}",key="inv_cohort_item_"+case['id'])
            cohort_item=next(i for i in targets if i['id']==cohort_item['id'])
            from explanation_summary import snapshot_cohort
            current_cohort=snapshot_cohort(case,cohort_item)
            if current_cohort.get('validation_reason'):st.warning(current_cohort['validation_reason'])
            metadata=cohort_item.get('cohort',{})
            roles={'normal':'正常樣本','true_defect':'確認瑕疵','false_positive':'確認誤逮'}
            with st.form('inv_cohort_'+cohort_item['id']):
                group=st.text_input('獨立批次／群組',value=metadata.get('group_id',''))
                role=st.selectbox('已核對樣本角色',list(roles),index=list(roles).index(metadata.get('role','normal')),format_func=roles.get)
                reviewer=st.text_input('批次資料覆核者',value=metadata.get('reviewer',''))
                reason=st.text_input('角色與群組依據',value=metadata.get('reason',''))
                verified=st.checkbox('我已核實角色與群組',value=metadata.get('verified') is True)
                if st.form_submit_button('保存批次核對資料'):
                    from explanation_summary import verified_item_review
                    review_evidence=verified_item_review(case,cohort_item)
                    expected='defect' if role=='true_defect' else 'normal'
                    if verified and (not review_evidence or review_evidence['details']['verdict']!=expected):
                        st.error('請先完成人工區域覆核，並確認樣本角色與正常／瑕疵判讀一致。')
                    elif verified and not all(v.strip() for v in (group,reviewer,reason)):
                        st.error('核實資料需要群組、覆核者與依據。')
                    else:
                        cohort_item['cohort']=snapshot_cohort(case,cohort_item,metadata={'group_id':group.strip(),'role':role,'reviewer':reviewer.strip(),'reason':reason.strip(),'verified':verified})
                        store.save(case)
                        st.success('已保存人工核對資料。')
            selected=st.multiselect("批次分析目標",targets,format_func=lambda i:f"{i['label']} · {i['id']}")
            selected_ids={i['id'] for i in selected}
            selected=[i for i in targets if i['id'] in selected_ids]
            batch_small=st.radio('批次遮擋比例',['標準：10%、25%、50%','較小：2%、5%、10%'],key='inv_batch_fractions_'+case['id']).startswith('較小')
            if st.button("執行批次熱圖與遮擋驗證",disabled=not selected):
                payload={"items":[{"case_id":case["id"],"item_id":i["id"],"target_identity":target_identity(case,i),"method":"positive","fractions":[.02,.05,.1] if batch_small else [.1,.25,.5],"controls":3,"cohort":snapshot_cohort(case,i)} for i in selected]}
                manager_for(store).submit("batch_gradcam",payload,domain.digest(json.dumps(payload,sort_keys=True).encode()),context={"case_id":case["id"]})
                go_area("工作紀錄",case["id"]);st.rerun()
    withdrawn=[(i,candidate) for candidate,key in [(False,"items"),(True,"candidates")] for i in case[key] if i.get("withdrawn")]
    if withdrawn:
        with st.expander("已撤回區域與候選"):
            selected=st.selectbox("恢復項目",withdrawn,format_func=lambda pair:pair[0].get("label") or pair[0]["id"])
            reason=st.text_input("恢復原因")
            if st.button("恢復並重新覆核"):
                row,candidate=selected
                domain.revise_region(case,row["id"],candidate=candidate,reason=reason)
                row["review"]=domain._review("pending","","")
                store.save(case);st.rerun()


def render_case_import(store,case=None):
    from investigation_data import DatasetRegistry,prepare_import
    title="批次收案／由資料集建立案件" if case is None else "補入多張圖片／匯入紀錄"
    with st.expander(title):
        datasets=DatasetRegistry(store).list()
        dataset=st.selectbox("選擇已匯入資料集",[None]+datasets,format_func=lambda d:"上傳圖片" if d is None else d["name"],key="inv_intake_ds_"+str(case is not None))
        uploads=st.file_uploader("報案圖片（可多選）",type=["png","jpg","jpeg","bmp","tif","tiff","webp"],accept_multiple_files=True,key="inv_intake_images_"+str(case is not None))
        preview=None
        duplicate_names=bool(uploads and len({f.name for f in uploads})!=len(uploads))
        if duplicate_names:st.error("本批有重複檔名，請改名，或到資料庫上傳保留目錄結構的 ZIP。")
        elif uploads:
            preview=prepare_import({f.name:f.getvalue() for f in uploads})
            st.dataframe([{"影像":r["name"],"尺寸":f"{r['size'][0]} × {r['size'][1]}","紀錄":{"missing":"未提供","provided":"已提供","empty":"已提供、零框","error":"格式錯誤"}.get(r["annotation"]["status"],r["annotation"]["status"])} for r in preview["images"]],hide_index=True)
            for error in preview.get("errors",[]):
                st.warning(f"{error.get('name',error.get('path','檔案'))}：{error.get('error',error.get('reason','格式不符'))}" if isinstance(error,dict) else str(error))
        kind=st.selectbox("本批影像來源",["original","screenshot"],format_func=lambda k:"原圖" if k=="original" else "截圖",key="inv_intake_kind_"+str(case is not None))
        name=st.text_input("本批案件名稱",value=case["title"] if case else "",key="inv_intake_name_"+str(case is not None))
        split=st.checkbox("每張圖片建立獨立案件",disabled=case is not None,key="inv_intake_split_"+str(case is not None))
        if dataset:
            st.dataframe([{"影像":r["name"],"紀錄角色":r["annotation"]["role"],"紀錄狀態":r["annotation"]["status"],"框數":len(r["annotation"]["boxes"])} for r in dataset["images"]],hide_index=True)
            st.caption("預測與人工標註保留來源角色；匯入不直接等於已覆核。")
        if st.button("確認匯入案件",disabled=duplicate_names or (not dataset and not (preview and preview["images"])),key="inv_confirm_import_"+str(case is not None)):
            if uploads:
                dataset=DatasetRegistry(store).save(name or "報案圖片",preview,kind=kind)
            saved=import_dataset_cases(store,dataset,name=name,kind=kind,case_id=case["id"] if case else None,split=split)
            go_area("案件",saved[0]["id"])
            st.rerun()


def import_dataset_cases(store,dataset,*,name="",kind="original",case_id=None,split=False):
    current=store.load(case_id) if case_id else None
    saved=[]
    for row in dataset["images"]:
        source_kind="screenshot" if row["source"].get("kind")=="screenshot" or row["source"].get("original_kind")=="screenshot" else kind
        if current is None or split:
            src=store.ingest(store.image_path(row["source"]).read_bytes(),row["name"],kind=source_kind)
            current=store.create(src,(name+" · "+row["name"] if name and split else name) or row["name"])
        else:
            src=next((s for s in current["sources"] if s["sha256"]==row["source"]["sha256"] and s["kind"]==source_kind),None)
            if src is None:
                src=store.ingest(store.image_path(row["source"]).read_bytes(),row["name"],kind=source_kind)
                domain.add_source(current,src)
        import_key=domain.digest((dataset["id"]+row["id"]+src["id"]).encode())
        if import_key not in current.setdefault("material_imports",[]):
            annotation=row["annotation"]
            domain._evidence(current,"imported_record","匯入現場紀錄／標註（尚未驗證歷史模型）",
                             source_id=src["id"],dataset_id=dataset["id"],annotation=annotation)
            for box in annotation.get("boxes",[]):
                try:
                    domain.add_item(current,src["id"],box["box"],box["class"],box.get("score"),origin="imported_"+annotation["role"])
                except ValueError as exc:
                    if "已存在" not in str(exc):
                        raise
            current["material_imports"].append(import_key)
            current=store.save(current)
        if split:
            saved.append(current)
    if not split and current:
        saved=[current]
    if not saved:
        raise ValueError("沒有可匯入的圖片")
    return saved


def render_inference(store,case,source):
    from model_profiles import ModelRegistry
    registry=ModelRegistry(store.root)
    profiles=registry.list()
    if not profiles:
        st.caption("可以先圈區與記錄。要直接推論，請至「我的模型」上傳原本的模型。")
        return
    ids=[p["id"] for p in profiles]
    preferred=case.get("model_profile",{}).get("id")
    selected=st.selectbox("此圖使用的模型／部署設定",ids,index=ids.index(preferred) if preferred in ids else 0,
        format_func=lambda pid:next(f"{p['name']} · v{p['revision']}" for p in profiles if p['id']==pid),key="inv_infer_profile_"+case["id"])
    latest=registry.get(selected)
    revisions=list(range(latest["latest_revision"],0,-1))
    previous=case.get("model_profile",{})
    preferred_revision=previous.get("revision") if previous.get("id")==selected else revisions[0]
    revision=st.selectbox("部署設定版本",revisions,index=revisions.index(preferred_revision) if preferred_revision in revisions else 0,
        format_func=lambda r:f"v{r} · "+str(registry.get(selected,r).get("metadata",{}).get("station_name") or registry.get(selected,r)["runtime_config"].get("model_version") or "未填站點"),key="inv_infer_revision_"+case["id"]+selected)
    profile=registry.get(selected,revision)
    manager=manager_for(store)
    if st.button("推論這張圖片",type="primary",key="inv_infer_"+source["id"]):
        payload={"profile_id":selected,"profile_revision":profile["revision"],"source":source,"case_id":case["id"]}
        job=manager.submit("infer",payload,domain.digest(json.dumps(payload,sort_keys=True).encode()),context={"case_id":case["id"]})
        st.session_state["inv_active_infer_"+case["id"]]=job["id"]
        st.rerun()
    job_id=st.session_state.get("inv_active_infer_"+case["id"])
    if not job_id:
        jobs=[j for j in manager.store.list(case_id=case["id"]) if j["task"]=="infer" and j["payload"].get("source",{}).get("id")==source["id"]]
        job_id=jobs[0]["id"] if jobs else None
    if job_id:
        if retry:=st.session_state.pop("inv_infer_status_"+case["id"]+"_retry_job",None):
            job_id=retry
            st.session_state["inv_active_infer_"+case["id"]]=retry
        job=manager.store.get(job_id)
        if job["payload"].get("source",{}).get("id")!=source["id"]:
            return
        _job_status(store,job_id,key="inv_infer_status_"+case["id"])
        job=manager.store.get(job_id)
        if job["state"]=="succeeded" and job.get("publication",{}).get("state","not_published")=="not_published":
            try:
                result=completed_result(store,job_id)
                attach_result(store,case["id"],result)
                st.session_state["inv_visible_run_"+source["id"]]=result["run"]["id"]
                st.session_state["inv_pending_visible_run_"+source["id"]]=result["run"]["id"]
                manager.store.mark_published(job_id,case_id=case["id"])
                st.rerun()
            except ValueError as exc:
                manager.store.mark_published(job_id,case_id=case["id"],error=str(exc))
                st.error(str(exc))


def graphical_regions(store,case,source):
    from region_picker import image_region_picker
    runs=[r for r in case.get("external_runs",[]) if r.get("source_id")==source["id"]]
    run=runs[-1] if runs else None
    if runs:
        ids=[r["id"] for r in runs]
        run_key="inv_visible_run_"+source["id"]
        pending=st.session_state.pop("inv_pending_visible_run_"+source["id"],None)
        if pending in ids:st.session_state[run_key]=pending
        preferred=st.query_params.get("inv_run")
        selected=st.selectbox("顯示的推論紀錄",ids,index=ids.index(preferred) if preferred in ids else len(ids)-1,
            format_func=lambda rid:next(f"{r.get('at','')[:19]} · {r.get('profile_snapshot',{}).get('name','模型')} · v{r.get('profile_revision','?')}" for r in runs if r['id']==rid),key=run_key)
        run=next(r for r in runs if r["id"]==selected)
        st.query_params["inv_run"]=selected
    rows=run["prediction"].get("detections",[]) if run else []
    boxes=[{"id":"det:"+str(j),"roi":r["box"],"label":f"{r['class']} {r['score']:.3f}","kind":"prediction"} for j,r in enumerate(rows)]
    boxes.extend({"id":r["id"],"roi":r["roi"],"label":r["label"] or "爭議區域","kind":"truth" if r["origin"]=="imported_ground_truth" else "manual"} for r in domain.active_rows(case["items"]) if r["source_id"]==source["id"] and not r.get("external_target"))
    key="inv_picker_"+case["id"]+source["id"]
    if st.session_state.get(key+"_run")!=(run["id"] if run else None):
        st.session_state[key+"_selected"]=None
        st.session_state[key+"_run"]=run["id"] if run else None
    event=image_region_picker(store.image_path(source),boxes=boxes,selected_id=st.session_state.get(key+"_selected"),
                               roi=st.session_state.get(key+"_roi"),key=key)
    if event and event.get("nonce")!=st.session_state.get(key+"_nonce"):
        st.session_state[key+"_nonce"]=event.get("nonce")
        if event.get("action")=="draw":
            st.session_state[key+"_roi"]=domain.roi_checked(event["roi"])
            st.session_state[key+"_draw_nonce"]=event["nonce"]
            st.rerun()
        elif event.get("action")=="select":
            selected=event["id"]
            st.session_state[key+"_selected"]=selected
            if selected.startswith("det:"):
                index=int(selected.split(":")[1])
                item=select_detection(case,run,rows[index])
                saved=store.save(case)
                st.session_state["inv_selected_item_"+case["id"]]=item["id"]
            else:
                st.session_state["inv_selected_item_"+case["id"]]=selected
            st.rerun()
    if rows:
        choice=st.selectbox("或從清單選擇預測框",[None]+list(range(len(rows))),format_func=lambda j:"請選框" if j is None else f"{j+1}. {rows[j]['class']} · {rows[j]['score']:.3f}",key=key+"_list")
        if st.button("選取此預測框",disabled=choice is None,key=key+"_pick"):
            item=select_detection(case,run,rows[choice])
            store.save(case)
            st.session_state[key+"_selected"]="det:"+str(choice)
            st.session_state["inv_selected_item_"+case["id"]]=item["id"]
            st.rerun()
    elif run:
        st.info("本次模型完整輸出沒有偵測框。可圈出報案位置繼續對照；這不等於影像已確認正常。")
    return st.session_state.get(key+"_roi")


def render_region_actions(store,case,item):
    if item.get("external_target"):
        st.button("解釋這個框",type="primary",on_click=_navigate_diagnostics,args=(case["id"],item["id"],item["source_id"]),key="inv_explain_"+item["id"])
    with st.expander("修正／撤回這個區域"):
        draw=st.session_state.get("inv_picker_"+case["id"]+item["source_id"]+"_roi")
        use_draw=st.checkbox("使用上方最新圈選範圍",disabled=draw is None,key="inv_use_draw_"+item["id"])
        label=st.text_input("修正類別／爭議名稱",value=item["label"],key="inv_edit_label_"+item["id"])
        reason=st.text_input("修正／撤回原因",key="inv_edit_reason_"+item["id"])
        a,b=st.columns(2)
        if a.button("儲存區域修正",key="inv_edit_"+item["id"]):
            domain.revise_region(case,item["id"],roi=draw if use_draw else item["roi"],label=label,reason=reason)
            store.save(case);st.rerun()
        if b.button("撤回區域",key="inv_withdraw_"+item["id"]):
            domain.revise_region(case,item["id"],withdraw=True,reason=reason)
            store.save(case);st.rerun()


def render_detector_diagnostics():
    store=workspace_store()
    context=st.session_state.get("investigation_diagnostic") or {"case_id":st.query_params.get("diagnostic_case"),"item_id":st.query_params.get("diagnostic_item")}
    try:
        case=store.load(context["case_id"])
        render_case_version_notice(store, case["id"])
        if notice := st.session_state.pop("inv_cam_saved_notice_"+case["id"], None):
            st.success(notice)
        item=next((i for i in case["items"] if i["id"]==context["item_id"] and not i.get("withdrawn")),None)
        if item is None:
            st.warning("所選區域已撤回或不存在，請返回案件選擇。")
            _return_case(case["id"])
            return
        st.title("模型診斷")
        st.caption(f"案件：{case['title']} · 目標：{item['label']} · 外部 YOLO 模型歸因")
        _return_case(case["id"])
        mode=st.segmented_control("分析來源",["YOLO 模型歸因","LV 輔助對照"],default="YOLO 模型歸因",key="inv_diagnostic_source") or "YOLO 模型歸因"
        if mode=="LV 輔助對照":
            st.info("LV 對照提供外觀線索；不是外部 YOLO 的內部歸因。")
            st.button("以此區域找對照",on_click=_open_aux_comparison,args=(case["id"],item["id"]))
            return
        source=next(s for s in case["sources"] if s["id"]==item["source_id"])
        from fp_case_page import _show_image
        caption="本次輸入：截圖（可能含標記或縮放，未確認是部署原圖）" if source["kind"]=="screenshot" else "完整原圖"
        _show_image(store,source,item["roi"],caption+"；框是分析目標，不會自動裁切推論輸入",preview_size=(600,280))
        link=item.get("external_target")
        run=next((r for r in case.get("external_runs",[]) if link and r["id"]==link["run_id"]),None)
        if run is None or not run.get("profile_snapshot"):
            st.info("請先使用我的模型推論原圖，並選取該次結果中的框。")
            return
        profile=run["profile_snapshot"]
        detection=next((d for d in run["prediction"].get("detections",[]) if d["output_index"]==link["output_index"] and d["class"]==link["class"]),None)
        st.write(f"模型：{profile.get('name',profile['id'])} · v{profile['revision']} · 框分數：{detection['score'] if detection else '目標已不可用'}")
        historical_profile=bool(case.get("model_profile") and case["model_profile"]!={"id":profile["id"],"revision":profile["revision"]})
        if historical_profile:st.warning("案件最近使用過另一個模型設定；此頁仍解釋所選框原本的模型版本。要分析新版，請回案件選擇新版推論的框。")
        st.caption("此頁解釋框分數；現場機台規則與根因需另外核對。")
        asset=profile.get("gradient_asset") or profile.get("deployment_asset")
        if not asset or asset["format"]!="pt":
            st.info("Grad-CAM 需要對應的 PyTorch 模型。目前仍可使用外部模型的亮度／遮擋對照實驗。")
            from model_profiles import ModelRegistry
            latest=ModelRegistry(store.root).get(profile["id"])
            if latest["revision"]!=profile["revision"] and latest.get("gradient_asset"):
                st.info(f"模型庫已有 v{latest['revision']} 配套模型；請回案件使用新版設定重新推論，舊分析保留原版本。")
                st.button("回案件選擇新版模型",on_click=_back_to_case,args=(case["id"],"案件與區域"))
            st.button("補入對應模型",on_click=_back_to_case,args=(case["id"],None,"我的模型"))
            st.button("進行對照實驗",on_click=_back_to_case,args=(case["id"],"外部模型"))
            return
        method=st.selectbox("熱圖方式",["positive","signed"],format_func=lambda x:"Grad-CAM 正向貢獻" if x=="positive" else "正負貢獻診斷",key="inv_cam_method")
        checks=st.checkbox("同時執行多比例遮擋與隨機對照",value=True,key="inv_cam_checks")
        with st.expander("定位真值與進階檢查"):
            mask_upload=st.file_uploader("對齊原圖的瑕疵 mask（可留空）",type=["png"],key="inv_cam_mask")
            st.caption("沒有 mask 仍可分析；定位驗證會標示尚無真值。")
            controls=st.number_input("隨機對照數",1,10,3,key="inv_cam_controls")
            small_fractions=st.radio('遮擋比例',['標準：10%、25%、50%','較小：2%、5%、10%'],key='inv_cam_fractions').startswith('較小')
            fractions=([.02,.05,.1] if small_fractions else [.1,.25,.5]) if checks else []
            scalar=st.selectbox("解釋分數",["confidence","class"] if profile.get("metadata",{}).get("family")=="yolov5" else ["confidence"],
                format_func=lambda s:"偵測框分數" if s=="confidence" else "僅類別分數（不含 objectness）")
            randomization=st.checkbox("檢查熱圖是否依賴模型權重（額外計算）",key="inv_cam_randomization")
        manager=manager_for(store)
        if st.button("分析所選框",type="primary"):
            payload={"case_id":case["id"],"item_id":item["id"],"target_identity":target_identity(case,item),
                     "method":method,"fractions":fractions,"controls":controls,"scalar":scalar,"randomization":randomization}
            if mask_upload:
                payload["mask_source"]=store.ingest(mask_upload.getvalue(),mask_upload.name)
            job=manager.submit("gradcam",payload,domain.digest(json.dumps(payload,sort_keys=True).encode()),context={"case_id":case["id"],"item_id":item["id"]})
            st.session_state["inv_cam_job_"+item["id"]]=job["id"]
            st.rerun()
        jobs=[j for j in manager.store.list(case_id=case["id"]) if j["task"]=="gradcam" and j["payload"]["item_id"]==item["id"]]
        if jobs:
            job_id=st.session_state.pop("inv_cam_status_"+item["id"]+"_retry_job",None) or st.session_state.get("inv_cam_job_"+item["id"],jobs[0]["id"])
            st.session_state["inv_cam_job_"+item["id"]]=job_id
            _job_status(store,job_id,key="inv_cam_status_"+item["id"])
            job=manager.store.get(job_id)
            if job["state"]=="succeeded":
                result=completed_result(store,job_id)
                desired={"method":method,"scalar":scalar,"fractions":fractions,"controls":controls,"randomization":randomization}
                changed=historical_profile or any(job["payload"].get(k,"confidence" if k=="scalar" else False if k=="randomization" else None)!=v for k,v in desired.items())
                current_mask=domain.digest(mask_upload.getvalue()) if mask_upload else None
                changed=changed or current_mask!=job["payload"].get("mask_source",{}).get("sha256")
                if changed:st.warning("分析設定已變更，以下保留前一次結果；請重新分析以取得目前設定的結果。")
                _render_attribution_result(store,case,item,result["attribution"],method)
                if st.button("保存歷史分析為案件證據" if changed else "保存為案件證據"):
                    try:
                        attach_result(store,case["id"],result)
                        manager.store.mark_published(job_id,case_id=case["id"])
                        st.session_state["inv_cam_saved_notice_"+case["id"]]="熱圖證據已加入案件，可於圖文回覆引用。"
                        st.rerun()
                    except ValueError as exc:
                        manager.store.mark_published(job_id,case_id=case["id"],error=str(exc));st.error(str(exc))
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:
        st.error(f"無法完成分析：{exc}")
        st.caption("案件與已完成結果仍保留，可返回案件修正後再試。")


def _open_aux_comparison(case_id, item_id):
    st.session_state["inv_search_item_" + case_id] = item_id
    st.session_state.pop("inv_search_context_" + case_id, None)
    st.session_state.pop("inv_search_evidence_" + case_id, None)
    _back_to_case(case_id, "找對照")


def _cam_comparison_region(case, item, attribution_evidence_id, selected_roi):
    """Reuse a drawn region while retaining each CAM-to-region evidence link."""
    if selected_roi is None:
        return item, False
    selected_roi = domain.roi_checked(selected_roi)
    search_item = next((i for i in domain.active_rows(case["items"])
                        if i["source_id"] == item["source_id"] and i["roi"] == selected_roi
                        and i.get("origin") == "cam_region"), None)
    if search_item is None:
        search_item = domain.add_item(case, item["source_id"], selected_roi, "熱圖待查區域", origin="cam_region")
    evidence_id = next((e["id"] for e in reversed(case["evidence"])
                        if e["role"] == "attribution_region" and e["details"].get("item_id") == search_item["id"]
                        and e["details"].get("roi") == selected_roi
                        and e["details"].get("attribution_evidence_id") == attribution_evidence_id), None)
    if evidence_id is None:
        evidence_id = domain._evidence(case, "attribution_region", "人工由熱圖圈選的待查區域",
                                      item_id=search_item["id"], attribution_evidence_id=attribution_evidence_id,
                                      roi=selected_roi)
    changed = search_item.get("evidence_id") != evidence_id
    search_item["evidence_id"] = evidence_id
    return search_item, changed


def _back_to_case(case_id,section=None,area="案件"):
    st.session_state.pop("investigation_diagnostic",None)
    for key in ["diagnostic_case","diagnostic_item"]:
        st.query_params.pop(key,None)
    st.session_state["tool_switch"]="誤逮案件"
    go_area(area,case_id,section)


def _return_case(case_id):
    st.button("返回原案件",on_click=_back_to_case,args=(case_id,),key="inv_return_case")


def _open_cam_comparison(store,case,item,result,selection_key):
    """Run navigation before tool_switch is instantiated in the next rerun."""
    error_key="inv_cam_navigation_error_"+case["id"]
    try:
        latest=domain.CaseStore(store.root).load(case["id"])
        current=next((r for r in latest["items"] if r["id"]==item["id"] and not r.get("withdrawn")),None)
        if current is None or target_identity(latest,current)!=result["target_identity"]:
            raise ValueError("原熱圖模型目標已變更，請開啟最新版本並重新核對")
        saved=attach_result(store,case["id"],{"attribution":result})
        row=next(r for r in saved["attributions"] if r["id"]==result["id"])
        search_item,changed=_cam_comparison_region(saved,item,row["evidence_id"],st.session_state.get(selection_key))
        if changed:
            saved=store.save(saved)
        context={"item_id":search_item["id"],"target_identity":target_identity(saved,search_item),
                 "attribution_evidence_id":row["evidence_id"]}
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        st.session_state[error_key]="無法開啟對照搜尋："+str(exc)
        return
    st.session_state.pop(error_key,None)
    st.session_state.pop("inv_search_evidence_"+case["id"],None)
    st.session_state["inv_search_context_"+case["id"]]=context
    st.session_state["inv_search_item_"+case["id"]]=search_item["id"]
    _back_to_case(case["id"],"找對照")


def _render_attribution_result(store,case,item,result,requested_method):
    from explanation_summary import QUESTION_TYPES,QUESTION_NEEDS
    with st.expander('調查問題與說明範圍'):
        question=case.get('explanation_question',{}).get('type','class_score')
        chosen=st.selectbox('本次調查問題',list(QUESTION_TYPES),index=list(QUESTION_TYPES).index(question) if question in QUESTION_TYPES else 0,format_func=QUESTION_TYPES.get,key='inv_question_'+case['id'])
        if st.button('保存說明題型',key='inv_save_question_'+case['id']):
            case['explanation_question']={'type':chosen}
            case=store.save(case)
        if chosen in QUESTION_NEEDS:
            st.info(QUESTION_NEEDS[chosen])
    if result["target_identity"]!=target_identity(case,item) or result.get("method")!=requested_method:
        st.warning("以下是歷史分析；目標或方法已變更，需重新分析才是目前結果。")
    st.write(result.get("legend",""))
    status=result.get("status")
    if status!="ok":
        st.warning({"zero_gradient":"目標梯度為零","no_positive_attribution":"沒有正向歸因"}.get(status,str(status)))
    root=Path(result["artifact_dir"])
    a,b=st.columns(2)
    a.image(str(root/"model_input.png"),caption="實際模型輸入",width="stretch")
    opacity=b.slider("熱圖透明度",0.0,1.0,.45,.05,key="inv_cam_opacity")
    import numpy as np
    from PIL import Image
    from yolo_attribution import render_overlay
    with np.load(root/"maps.npz",allow_pickle=False) as maps:
        scores=maps["source"]
    with Image.open(root/"source.png") as image:
        overlay=render_overlay(image.convert("RGB"),scores,method=result["method"],opacity=opacity)
    b.image(overlay,caption="指定框分數的模型歸因",width="stretch")
    st.caption(result.get("limitation",""))
    st.write("配套模型對齊："+result.get("correspondence",{}).get("scope","尚未驗證"))
    st.write("歷史部署重播："+PARITY_LABELS.get((result.get("historical_parity") or {}).get("status"),'尚未驗證'))
    if result.get("faithfulness"):
        with st.expander("遮擋與隨機對照結果",expanded=True):
            faith=result["faithfulness"]
            if faith.get("status")!="measured":
                st.info(faith.get("reason","尚無有效遮擋對照"))
            else:
                st.caption("追蹤原本指定框的固定分數；重新找到的框另外列出，避免把替代框當成原框。")
                st.metric("原始固定目標分數",f"{faith['baseline_scalar']:.5f}")
                rows=[]
                for r in faith["rows"]:
                    if r.get('unavailable'):
                        st.warning(f"比例 {r['fraction']}／{r.get('fill_method','')}：公平控制可用 {r.get('available_controls',0)}／{r.get('requested_controls',0)}。有限搜尋未找到足夠低重疊同形狀區域；可嘗試較小比例或其他原圖。")
                        continue
                    found=r.get("rematched_detection")
                    rows.append({"遮擋比例":r["fraction"],"填補方式":r.get('fill_method','舊協定'),"方法":"熱區" if r["role"]=="attribution" else f"同形狀對照 {r['control']}",
                                 "原目標分數":r["fixed_scalar"],"分數下降":r["fixed_score_drop"],
                                 "重新匹配框":f"{found['class']} {found['score']:.4f}" if found else "未找到"})
                st.dataframe(rows,hide_index=True)
                st.caption(faith["limitation"])
                variants=[r for r in faith["rows"] if r.get("modified_image")]
                if variants:
                    variant=st.selectbox("檢查實際遮擋圖片",variants,format_func=lambda r:f"{r['fraction']:.0%} · {r.get('fill_method','舊協定')} · "+("熱區" if r["role"]=="attribution" else f"同形狀對照 {r['control']}"))
                    modified=Path(variant["modified_image"])
                    if not modified.resolve().is_relative_to(root.resolve()) or domain.digest(modified.read_bytes())!=variant["modified_sha256"]:
                        raise ValueError("遮擋圖片與原紀錄不符，請重新產生分析")
                    st.image(str(modified),caption="真正送入前處理的修改圖；請人工確認修改是否保留目標",width="stretch")
    localization=result.get("localization",{})
    if localization.get("status")=="measured":
        st.write("定位真值對照", {k:v for k,v in localization.items() if k not in {"status","limitation"}})
    else:
        st.caption("定位驗證：尚無有效且對齊原圖的真值 mask。")
    with st.expander("定位與計算參數"):
        st.json({k:result.get(k) for k in ["localization","randomization_sanity","layer","target","compute_id","forward_parity"]})
    output=io.BytesIO()
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob("*"):
            if path.is_file():archive.write(path,path.relative_to(root).as_posix())
        archive.writestr("case_evidence.json",json.dumps(result,ensure_ascii=False,indent=2))
    st.download_button("下載完整熱圖證據包",output.getvalue(),file_name=result["id"]+".zip",mime="application/zip",on_click="ignore")
    _render_explanation_assessment(store,case,result)
    from region_picker import image_region_picker
    selection_key="inv_cam_search_roi_"+result["id"]
    with st.expander("圈選熱圖中的可疑區域找對照"):
        st.caption("可圈反光、背景或物體細節；這會建立待查區域，不改變原本解釋的模型框。未圈選時沿用原框。")
        event=image_region_picker(root/"overlay.png",roi=st.session_state.get(selection_key),key=selection_key+"_picker")
        if event and event.get("action")=="draw" and event.get("nonce")!=st.session_state.get(selection_key+"_nonce"):
            st.session_state[selection_key]=domain.roi_checked(event["roi"])
            st.session_state[selection_key+"_nonce"]=event["nonce"]
            st.rerun()
    if error:=st.session_state.get("inv_cam_navigation_error_"+case["id"]):
        st.error(error)
    st.button("從此區域找同類案例",on_click=_open_cam_comparison,args=(store,case,item,result,selection_key))


def _render_explanation_assessment(store,case,result,*,show_variants=False):
    from explanation_summary import build_explanation_summary,render_summary_text,record_preservation_review,variant_hashes
    saved=next((a for a in case.get('attributions',[]) if a['id']==result['id']),None)
    if saved is None:
        st.info('先保存為案件證據，才能建立可引用摘要與修改圖人工確認。')
        return
    hashes=variant_hashes(saved)
    with st.expander('人工確認修改圖是否保留目標外觀'):
        if not hashes:
            st.info('本次沒有可供確認的修改圖指紋；不將舊結果視為已確認。')
        else:
            if show_variants:
                root=(store.root/str(saved.get('artifact_relative',''))).resolve()
                variants=[r for r in saved.get('faithfulness',{}).get('rows',[]) if r.get('modified_image')]
                for variant in variants:
                    path=Path(variant['modified_image']).resolve()
                    if not root.is_relative_to(store.root.resolve()) or not path.is_relative_to(root) or not path.is_file() or domain.digest(path.read_bytes())!=variant.get('modified_sha256'):
                        st.error('修改圖檔案與保存指紋不符，請恢復資產或重新產生分析。')
                        return case
                    st.image(str(path),caption=f"比例 {variant.get('fraction')} · {FILL_LABELS.get(variant.get('fill_method'),'舊協定未記錄填補方式')} · "+('熱區修改' if variant.get('role')=='attribution' else '同形狀控制修改')+'，請核對目標外觀',width='stretch')
                if len({r.get('modified_sha256') for r in variants})!=len(hashes):
                    st.error('部分修改圖未保存，不能確認全部目標外觀。')
                    return case
            with st.form('inv_preservation_'+saved['id']):
                verdict=st.selectbox('修改圖判讀',['uncertain','preserved','changed'],format_func=lambda v:{'uncertain':'不確定','preserved':'仍保留目標外觀','changed':'已改變目標外觀'}[v])
                reviewer=st.text_input('修改圖覆核者')
                reason=st.text_input('目標保留判讀依據')
                checked=st.checkbox(f'我已核對本次全部 {len(hashes)} 張不同修改圖')
                submit=st.form_submit_button('保存修改圖人工確認')
            if submit:
                try:
                    if not checked:
                        raise ValueError('請先核對全部修改圖再保存確認')
                    record_preservation_review(case,saved['id'],verdict=verdict,reviewer=reviewer,reason=reason)
                    case=store.save(case)
                    st.success('人工確認已綁定本次修改圖指紋；不代表根因已確認')
                except (ValueError,OSError) as exc:
                    st.error(str(exc))
    with st.expander('指定分數的受控證據摘要',expanded=True):
        summary=build_explanation_summary(case,[saved['evidence_id']],attribution_id=saved['id'])
        st.text(render_summary_text(summary))
    with st.expander('逐張修改圖分數詳細紀錄'):
        for observation in summary['observations']:
            if observation.get('kind')=='variant':st.write(observation['text'])
    return case
