"""Upload-first model library. All model execution is submitted to local jobs."""
from __future__ import annotations

import hashlib
import json

from model_profiles import DEFAULT_MAX_BYTES, ModelRegistry


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _names(inspection):
    value = inspection.get("class_names", inspection.get("names", []))
    if isinstance(value, dict):
        value = [v for _, v in sorted(value.items(), key=lambda row: int(row[0]))]
    return [str(v) for v in value or []]


def _status(st, manager, job, *, prefix):
    """Poll only job metadata; inspect/result loading remains an explicit page action."""
    @st.fragment(run_every="1s")
    def progress_panel():
        current = manager.store.get(job["id"])
        labels = {"queued": "等待執行", "running": "執行中", "succeeded": "已完成", "failed": "執行失敗",
                  "cancelled": "已取消", "interrupted": "執行中斷"}
        st.caption(f"{labels[current['state']]} · {current.get('stage', '')}")
        if current["state"] in {"queued", "running"}:
            st.progress(float(current.get("progress", 0)))
            if st.button("取消這項工作", key=prefix + "_cancel"):
                manager.cancel(current["id"])
                st.rerun()
        elif current["state"] == "succeeded":
            if st.session_state.get(prefix + "_seen") != current["id"]:
                st.session_state[prefix + "_seen"] = current["id"]
                st.rerun()
        else:
            if current.get("error"):
                st.error(current["error"])
            if st.button("重試工作", key=prefix + "_retry"):
                new = manager.retry(current["id"])
                st.session_state[prefix + "_pending_job"] = new["id"]
                st.rerun()
    progress_panel()


def _config_fields(st, previous, *, prefix):
    from fp_external_runtime import DEFAULT_CONFIG
    from fp_external_page import FORMAT_LABELS
    config = {**DEFAULT_CONFIG, **(previous or {})}
    classes = st.text_area("模型類別（依模型順序，每行一個）", value="\n".join(config["class_names"]), key=prefix + "_classes")
    cols = st.columns(2)
    width = cols[0].number_input("輸入寬度", min_value=16, max_value=8192, value=int(config["input_size"][0]), step=32, key=prefix + "_width")
    height = cols[1].number_input("輸入高度", min_value=16, max_value=8192, value=int(config["input_size"][1]), step=32, key=prefix + "_height")
    threshold = st.slider("判定門檻", 0.0, 1.0, float(config["threshold"]), .01, key=prefix + "_threshold")
    with st.expander("進階部署設定"):
        resize = st.selectbox("縮放方式", ["letterbox", "stretch"], index=0 if config["resize"] == "letterbox" else 1, key=prefix + "_resize")
        nms = st.slider("重疊框抑制 IoU", 0.0, 1.0, float(config["nms_iou"]), .01, key=prefix + "_nms")
        formats = list(FORMAT_LABELS)
        fmt = st.selectbox("輸出結構", formats, index=formats.index(config["output_format"]) if config["output_format"] in formats else 0,
                           format_func=FORMAT_LABELS.get, key=prefix + "_format")
        st.caption("色彩、正規化與其他設定沿用模型檢查結果；若現場有自訂處理，請核對後再認定重播相符。")
    config.update(class_names=[v.strip() for v in classes.replace(",", "\n").splitlines() if v.strip()],
                  input_size=[int(width), int(height)], threshold=threshold, nms_iou=nms, resize=resize, output_format=fmt)
    if fmt == "classification":
        config["normal_classes"] = st.multiselect("正常類別（不觸發瑕疵警報）", config["class_names"],
            default=[v for v in config["normal_classes"] if v in config["class_names"]], key=prefix + "_normal_classes")
    if not config["class_names"]:
        st.info("請填入模型類別，才能儲存設定與試跑。")
    return config


def _import_panel(st, store, manager, registry):
    st.subheader("新增或補入模型")
    upload_limit = min(DEFAULT_MAX_BYTES, int(st.get_option("server.maxUploadSize")) * 1024**2)
    st.caption(f"可上傳 .pt／.onnx；瀏覽器上傳最多 {upload_limit // (1024**2)} MB，LV 主機檔案最多 {DEFAULT_MAX_BYTES // (1024**2)} MB。上傳後按「檢查模型」才會載入。")
    upload = st.file_uploader("上傳模型", type=["pt", "onnx"], key="model_import_upload")
    with st.expander("大型模型：使用 LV 主機上的檔案"):
        server_path = st.text_input("LV 主機模型路徑", key="model_import_path")
    from model_runtimes import LEGACY, MODERN, RELEASES, available
    runtime_states = {row["id"]: row for row in available()}
    installed = {key: row["installed"] for key, row in runtime_states.items()}
    selected_runtime = st.selectbox("模型執行版本", [LEGACY, MODERN],
        format_func=lambda value: "既有 YOLOv5／v8／11 · 8.3.203" if value == LEGACY else "新版 YOLO12／26／v8／11 · 8.4.142",
        key="model_import_runtime")
    head_path = None
    if selected_runtime == MODERN:
        head_path = st.selectbox("推論 head", [None, "one_to_many", "one_to_one"],
            format_func=lambda value: {None: "保留模型現有 head", "one_to_many": "One-to-many＋NMS", "one_to_one": "One-to-one（NMS-free）"}[value],
            key="model_import_head")
        st.caption("檢查後會顯示實際 head 與輸出結構；變更 head 代表另一條推論路徑，須另行核對部署結果。")
    if not installed[selected_runtime]:
        state = "安裝尚未完成" if runtime_states[selected_runtime]["state"] == "incomplete" else "尚未安裝"
        st.warning(f"此執行版本{state}。請由 LV 主機執行 .venv/Scripts/python.exe scripts/setup_model_runtime.py 後重試。")
    profiles = registry.list()
    targets = [None] + profiles
    target = st.selectbox("儲存位置", targets, format_func=lambda p: "建立新模型" if p is None else f"補入 {p['name']}（建立新版本）", key="model_import_target")
    if st.button("檢查模型", type="primary", disabled=(upload is None and not server_path.strip()) or not installed[selected_runtime]):
        with st.spinner("保存模型檔案…"):
            asset = registry.ingest(upload.getvalue(), upload.name, max_bytes=upload_limit) if upload is not None else registry.import_path(server_path)
        payload = {"asset_sha": asset["sha256"], "asset_suffix": "." + asset["format"], "workspace_root": str(store.root)}
        payload.update(runtime_id=selected_runtime, runtime_release=RELEASES[selected_runtime])
        if head_path:
            payload.update(head_path=head_path, nms=False if head_path == "one_to_one" else None)
        job = manager.submit("inspect", payload, _key({"inspect": payload}))
        st.session_state["model_inspect_job"] = job["id"]
        st.session_state["model_inspect_selection"] = job["id"]
        st.session_state["model_inspect_pending_job"] = job["id"]
        st.rerun()
    imports = [j for j in manager.store.list() if j["task"] == "inspect"]
    if not imports:
        return
    ids = [j["id"] for j in imports]
    if pending := st.session_state.pop("model_inspect_pending_job", None):
        st.session_state["model_inspect_job"] = pending
        st.session_state["model_inspect_selection"] = pending
    preferred = st.session_state.get("model_inspect_job", ids[0])
    selected = st.selectbox("模型檢查紀錄", ids, index=ids.index(preferred) if preferred in ids else 0,
                            format_func=lambda key: registry.asset(manager.store.get(key)["payload"]["asset_sha"])["name"] + " · " + manager.store.get(key)["created_at"][:19], key="model_inspect_selection")
    job = manager.store.get(selected)
    _status(st, manager, job, prefix="model_inspect")
    job = manager.store.get(selected)
    if job["state"] != "succeeded":
        return
    inspection = manager.store.result(selected)["inspection"]
    asset = registry.asset(job["payload"]["asset_sha"])
    capabilities = inspection.get("capabilities") or {}
    inspected_config = inspection.get("default_config") or {}
    if inspected_config.get("runtime_release"):
        st.caption(f"實際執行版本：{inspected_config['runtime_release']} · head：{inspected_config.get('head_path', '傳統 Detect')} · 輸出：{inspection.get('output_contract', inspected_config.get('output_format', '未辨識'))}")
    st.write(f"{inspection.get('family', inspection.get('format', asset['format']))} · {inspection.get('task', '尚未辨識任務')}")
    st.caption("可推論：" + ("是" if capabilities.get("inference") else "尚未可用") + " · 可 Grad-CAM：" + ("是" if capabilities.get("gradcam") else "尚未可用"))
    for warning in inspection.get("warnings", []):
        st.warning(str(warning))
    if not capabilities.get("inference"):
        st.info("此檢查結果尚無可用的推論能力；請依上方原因補入支援模型。原案件不受影響。")
        return
    default_config = dict(inspection.get("default_config") or {})
    default_config.setdefault("class_names", _names(inspection))
    if inspection.get("input_size"):
        default_config.setdefault("input_size", inspection["input_size"])
    st.caption("模型可讀取欄位已帶入。門檻與縮放等現場設定仍需由你核對；建議預設不代表已驗證部署設定。")
    gradient_pair = bool(target and (target.get("deployment_asset") or {}).get("format") == "onnx" and asset["format"] == "pt")
    form_config = default_config
    if target:
        # Share the confirmed image recipe, but every newly attached backend
        # retains its inspected output topology and tensor names.
        shared = {key: value for key, value in target["runtime_config"].items()
                  if key not in {"output_format", "output_name", "input_name", "score_activation", "coordinates", "model_path",
                                 "runtime_id", "runtime_release", "head_path", "nms", "output_contract", "family"}}
        form_config = {**default_config, **shared}
    if gradient_pair:
        st.info("此檔案將成為對應的 Grad-CAM 模型；原 ONNX 部署設定保留，兩個模型另行比對。")
    prefix = "model_save_" + selected + (target["id"] if target else "new")
    with st.form(prefix):
        name = st.text_input("模型名稱", value=target["name"] if target else asset["name"], key=prefix + "_name")
        config = _config_fields(st, form_config, prefix=prefix)
        version = st.text_input("現場版本名稱", value=config.get("model_version") or "", key=prefix + "_version")
        station = st.text_input("產線／站點名稱", value=(target.get("metadata", {}) if target else {}).get("station_name", ""), key=prefix + "_station")
        machine_rules = st.text_area("機台後續判定規則（不知道可留空）", value=(target.get("metadata", {}) if target else {}).get("machine_rules", ""), key=prefix + "_machine_rules",
                                     help="例如連續幾張警報、面積排除或 PLC 邏輯。此欄記錄現場描述，不代表 LV 已重播或驗證這些規則。")
        confirmed = st.checkbox("我已核對類別與現場輸入／判定設定", key=prefix + "_confirm")
        save = st.form_submit_button("儲存模型設定", type="primary")
    if save:
        if not confirmed or not name.strip() or not config["class_names"]:
            st.error("請填入名稱、模型類別並確認現場設定。")
            return
        config["model_version"] = version
        metadata = {**(target.get("metadata", {}) if target else {}), "inspection": inspection,
                    "settings_source": "user_confirmed", "inspection_job_id": selected,
                    "station_name": station.strip(), "machine_rules": machine_rules.strip(),
                    "field_provenance": {"station_name": "user_reported" if station.strip() else "unknown",
                                         "machine_rules": "user_reported_unverified" if machine_rules.strip() else "unknown"}}
        if inspection.get("family") and asset["format"] == "pt":
            metadata["gradient_family" if target and (target.get("deployment_asset") or {}).get("format") == "onnx" else "family"] = inspection["family"]
        if gradient_pair:
            metadata["gradient_config"] = dict(config)
        elif asset["format"] == "onnx" and target and target.get("gradient_asset"):
            metadata.setdefault("gradient_config", dict(target["runtime_config"]))
            if metadata.get("family"):
                metadata.setdefault("gradient_family", metadata["family"])
        deployment = target["deployment_asset"] if target else None
        gradient = target["gradient_asset"] if target else None
        if asset["format"] == "onnx":
            deployment = asset
        else:
            gradient = asset
            if not deployment or deployment["format"] == "pt":
                deployment = asset
        if target:
            capabilities = {**target.get("capabilities", {}), **capabilities,
                            "inference": bool(capabilities.get("inference") or target.get("capabilities", {}).get("inference")),
                            "gradcam": bool(gradient and (capabilities.get("gradcam") or
                                (asset["format"] == "onnx" and target.get("capabilities", {}).get("gradcam"))))}
            profile = registry.revise(target["id"], deployment_asset=deployment, gradient_asset=gradient,
                                      runtime_config=target["runtime_config"] if gradient_pair else config,
                                      capabilities=capabilities, metadata=metadata, expected_revision=target["revision"])
            registry.rename(target["id"], name)
        else:
            profile = registry.create(name, deployment, gradient, config, capabilities, metadata)
        st.session_state["model_library_selected"] = profile["id"]
        st.session_state["model_library_pending"] = {"id": profile["id"], "revision": profile["revision"]}
        st.session_state["model_library_notice"] = "模型已儲存，可在下方試跑或在案件中直接選用。"
        st.rerun()


def _trial(st, store, manager, profile):
    st.subheader("試跑一張圖片")
    upload = st.file_uploader("試跑原圖", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"], key="model_trial_upload_" + profile["id"])
    cases = store.list()
    case = st.selectbox("或選既有案件圖片", [None] + cases, format_func=lambda c: "使用上傳原圖" if c is None else c["title"], key="model_trial_case_" + profile["id"])
    source = None
    if case:
        case = store.load(case["id"])
        source = st.selectbox("案件影像", case["sources"], format_func=lambda s: s["name"], key="model_trial_source_" + profile["id"])
    if st.button("試跑模型", type="primary", disabled=upload is None and source is None):
        if upload is not None and source is None:
            source = store.ingest(upload.getvalue(), upload.name, kind="original")
        payload = {"profile_id": profile["id"], "profile_revision": profile["revision"], "source": source, "case_id": case["id"] if case else None}
        job = manager.submit("infer", payload, _key(payload), context={"case_id": case["id"] if case else None})
        st.session_state["model_trial_job"] = job["id"]
        st.session_state["model_trial_selection_" + profile["id"]] = job["id"]
        st.session_state["model_trial_pending_job"] = job["id"]
        st.rerun()
    jobs = [j for j in manager.store.list() if j["task"] == "infer" and j["payload"].get("profile_id") == profile["id"]]
    if not jobs:
        return
    ids = [j["id"] for j in jobs]
    if (pending := st.session_state.pop("model_trial_pending_job", None)) in ids:
        st.session_state["model_trial_job"] = pending
        st.session_state["model_trial_selection_" + profile["id"]] = pending
    preferred = st.session_state.get("model_trial_job", ids[0])
    selected = st.selectbox("試跑紀錄", ids, index=ids.index(preferred) if preferred in ids else 0, key="model_trial_selection_" + profile["id"])
    job = manager.store.get(selected)
    _status(st, manager, job, prefix="model_trial")
    if manager.store.get(selected)["state"] != "succeeded":
        return
    from fp_attribution import completed_result
    result = completed_result(store, selected)
    run = result["run"]
    st.success("試跑完成；結果已保存。這不代表已核對歷史部署判定。")
    from fp_external_page import _run_preview
    _run_preview(run)
    prediction = run.get("prediction") or {}
    detections = prediction.get("detections") or []
    if detections:
        st.dataframe(detections, hide_index=True, use_container_width=True)
    else:
        st.info("本次沒有偵測框。可調整現場設定建立新版本再試；這不是缺少預測檔。")
    original_case_id = job["payload"].get("case_id")
    publication = manager.store.get(selected).get("publication", {})
    published_case = publication.get("case_id") if publication.get("state") == "published" else None
    action_label = "開啟已保存的案件" if published_case else "儲存結果並返回原案件" if original_case_id else "用此圖建立案件"
    if st.button(action_label, key="model_promote_" + selected):
        from fp_attribution import attach_result, promote_trial
        try:
            created = store.load(published_case) if published_case else attach_result(store, original_case_id, result) if original_case_id else promote_trial(store, result, title="模型試跑：" + profile["name"])
            manager.store.mark_published(selected, case_id=created["id"])
        except ValueError as exc:
            manager.store.mark_published(selected, case_id=original_case_id, error=exc)
            raise
        st.session_state["investigation_next_area"] = "案件"
        st.session_state["investigation_next_case"] = created["id"]
        st.rerun()


def render_model_library(store, manager):
    import streamlit as st
    registry = ModelRegistry(store.root)
    st.header("我的模型")
    st.caption("上傳一次、跨案件選用。部署模型與對應的 Grad-CAM 模型保留各自身分。")
    if notice := st.session_state.pop("model_library_notice", None):
        st.success(notice)
    try:
        _workspace_backup(st, store)
        with st.expander("匯入／補入模型", expanded=not registry.list()):
            _import_panel(st, store, manager, registry)
        include_archived = st.checkbox("顯示已封存模型", key="model_show_archived")
        profiles = registry.list(include_archived=include_archived)
        if not profiles:
            st.info("尚無可選模型，請先在上方上傳並檢查。")
            return
        by_id = {p["id"]: p for p in profiles}
        pending = st.session_state.pop("model_library_pending", None)
        if pending and pending["id"] in by_id:
            st.session_state["model_library_selected"] = pending["id"]
            st.session_state["model_revision_" + pending["id"]] = pending["revision"]
        if st.session_state.get("model_library_selected") not in by_id:
            st.session_state["model_library_selected"] = next(iter(by_id))
        selected = st.selectbox("選擇模型", list(by_id), format_func=lambda key: by_id[key]["name"] + (" · 已封存" if by_id[key]["archived"] else ""), key="model_library_selected")
        latest = by_id[selected]
        revision = st.selectbox("設定版本", list(range(latest["latest_revision"], 0, -1)), key="model_revision_" + selected)
        profile = registry.get(selected, revision)
        st.caption(f"版本 {revision} · {profile['runtime_config'].get('model_version', '')}")
        for role, label in (("deployment_asset", "部署模型"), ("gradient_asset", "Grad-CAM 模型")):
            asset = profile[role]
            if asset:
                st.write(f"{label}：{asset['name']} · {asset['format']}")
        availability = registry.availability(selected, revision)
        for role, state in availability.items():
            if profile[role] and not state["available"]:
                st.error(state["reason"])
                restored = st.file_uploader("重新附入相同模型檔", type=["pt", "onnx"], key="model_restore_" + role)
                if restored and st.button("驗證並恢復檔案", key="model_restore_btn_" + role):
                    registry.restore_asset(profile[role]["sha256"], restored.getvalue())
                    st.rerun()
        with st.expander("管理模型"):
            name = st.text_input("顯示名稱", value=latest["name"], key="model_rename_" + selected)
            if st.button("儲存名稱"):
                registry.rename(selected, name)
                st.rerun()
            if st.button("恢復模型" if latest["archived"] else "封存模型"):
                (registry.restore if latest["archived"] else registry.archive)(selected)
                st.rerun()
            st.caption("封存保留舊案件引用與模型檔案。改名不會更改模型權重或設定版本。")
        if not latest["archived"]:
            with st.expander("修改現場設定（建立新版本）"):
                with st.form("model_edit_" + selected + str(revision)):
                    config = _config_fields(st, profile["runtime_config"], prefix="model_edit_" + selected + str(revision))
                    version = st.text_input("現場版本名稱", value=config.get("model_version", ""), key="model_edit_version_" + selected + str(revision))
                    station = st.text_input("產線／站點名稱", value=profile["metadata"].get("station_name", ""), key="model_edit_station_" + selected + str(revision))
                    machine_rules = st.text_area("機台後續判定規則（不知道可留空）", value=profile["metadata"].get("machine_rules", ""), key="model_edit_rules_" + selected + str(revision),
                                                 help="保留現場描述；未驗證的後處理規則不能算完整機台重播。")
                    confirmed = st.checkbox("已核對這次設定變更", key="model_edit_confirm_" + selected + str(revision))
                    save = st.form_submit_button("儲存新版本")
                if save:
                    if not confirmed or not config["class_names"]:
                        st.error("請填入類別並確認設定變更。")
                    else:
                        config["model_version"] = version
                        updated = registry.revise(selected, runtime_config=config, expected_revision=latest["revision"],
                                                  metadata={**profile["metadata"], "settings_source": "user_confirmed", "based_on_revision": revision,
                                                            "station_name": station.strip(), "machine_rules": machine_rules.strip(),
                                                            "field_provenance": {**profile["metadata"].get("field_provenance", {}),
                                                              "station_name": "user_reported" if station.strip() else "unknown",
                                                              "machine_rules": "user_reported_unverified" if machine_rules.strip() else "unknown"}})
                        st.session_state["model_library_notice"] = f"已建立版本 {updated['revision']}；舊案件證據保留原版本。"
                        st.session_state["model_library_pending"] = {"id": selected, "revision": updated["revision"]}
                        st.rerun()
        if not latest["archived"] and any(v["available"] for v in availability.values()):
            _trial(st, store, manager, profile)
    except (ValueError, OSError, KeyError) as exc:
        st.error(str(exc))


def _workspace_backup(st, store):
    import sqlite3
    import zipfile
    from investigation_backup import create_backup, inspect_backup, managed_restore_destination, restore_backup
    root_key = str(store.root)
    with st.expander("工作區備份與還原"):
        st.caption("備份包含案件／歷史、原圖、資料庫、模型設定與已完成工作證據。還原建立獨立工作區，不覆寫目前案件。")
        include = st.checkbox("包含模型權重（檔案較大；不勾選時可於還原後補回相同模型）", key="workspace_backup_models")
        try:
            if st.button("建立工作區備份"):
                with st.spinner("建立一致性快照並驗證檔案…"):
                    data = create_backup(store.root, include_models=include)
                st.session_state["workspace_backup_ready"] = {"root": root_key, "data": data, "includes_models": include}
            ready = st.session_state.get("workspace_backup_ready")
            if ready and ready["root"] == root_key:
                st.caption("此備份" + ("包含模型權重。" if ready["includes_models"] else "不含模型權重；模型版本與來源身分仍保留。"))
                st.download_button("下載完整工作區備份", ready["data"], file_name="lv-investigation-workspace.zip", mime="application/zip", on_click="ignore")
            st.caption(f"還原 ZIP 上傳上限：{st.get_option('server.maxUploadSize')} MB。原工作區將保留。")
            uploaded = st.file_uploader("上傳工作區備份 ZIP", type=["zip"], key="workspace_restore_upload")
            if st.button("檢查備份內容", disabled=uploaded is None):
                data = uploaded.getvalue()
                with st.spinner("檢查清單、大小與每個檔案的校驗碼…"):
                    manifest = inspect_backup(data)
                st.session_state["workspace_restore_checked"] = {"sha": hashlib.sha256(data).hexdigest(), "manifest": manifest}
            checked = st.session_state.get("workspace_restore_checked")
            if checked and uploaded is not None and checked["sha"] == hashlib.sha256(uploaded.getvalue()).hexdigest():
                manifest = checked["manifest"]
                st.write(f"備份時間：{manifest['created_at']} · {len(manifest['files'])} 個檔案")
                st.caption("模型權重：" + ("已包含" if manifest["includes_models"] else "未包含，還原後可補回"))
                if st.button("還原成新工作區", type="primary"):
                    destination = managed_restore_destination(store.root)
                    with st.spinner("驗證並還原至新的工作區…"):
                        result = restore_backup(uploaded.getvalue(), destination)
                    st.session_state["workspace_restore_result"] = result
            restored = st.session_state.get("workspace_restore_result")
            if restored:
                st.success("備份已還原至獨立工作區，目前工作區保留。")
                if st.button("開啟還原工作區"):
                    from fp_cases import CaseStore
                    st.session_state["investigation_workspace_root"] = restored["workspace_root"]
                    st.session_state["investigation_next_area"] = "案件"
                    cases = CaseStore(restored["workspace_root"]).list()
                    if cases:
                        st.session_state["investigation_next_case"] = cases[0]["id"]
                    st.rerun()
        except (ValueError, OSError, KeyError, sqlite3.DatabaseError, zipfile.BadZipFile) as exc:
            st.error(f"備份／還原未完成：{exc}")
