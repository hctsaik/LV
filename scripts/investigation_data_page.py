"""Ordinary uploaded-data and reply workflows, independent of model backends."""
from __future__ import annotations

import json
import streamlit as st
import streamlit.components.v1 as components

from investigation_data import DatasetRegistry, prepare_import, build_evaluation_manifest
from investigation_reply import factual_draft, render_reply_html, make_reply_revision, engineering_appendix, reply_review_warnings, reply_attribution_images


def render_data_library(store):
    st.subheader('調查資料庫')
    st.caption('上傳圖片、標註／預測或 ZIP。最多 5,000 檔、解壓後合計 250 MB；先預覽再保存。相對路徑須精確配對。')
    uploads = st.file_uploader('圖片與紀錄', type=['png','jpg','jpeg','bmp','tif','tiff','webp','zip','txt','csv','json'], accept_multiple_files=True, key='inv_materials')
    a,b = st.columns(2)
    name = a.text_input('資料集名稱', key='inv_dataset_name')
    role = b.selectbox('紀錄角色', ['prediction','ground_truth'], format_func=lambda x:'模型預測' if x=='prediction' else '人工標註', key='inv_role')
    a,b,c = st.columns(3)
    fmt = a.selectbox('紀錄格式', ['yolo','csv','json'], format_func=lambda x:{'yolo':'YOLO TXT','csv':'CSV','json':'JSON'}[x], key='inv_format')
    coordinates = b.selectbox('座標單位', ['normalized','pixels'], format_func=lambda x:'0–1 比例' if x=='normalized' else '像素', key='inv_coordinates')
    box_format = c.selectbox('框格式', ['cxcywh','xyxy'], format_func=lambda x:'中心點＋寬高' if x=='cxcywh' else '左上＋右下', key='inv_box_format')
    mapping = st.data_editor([{'原始類別':'0','LV 類別名稱':''}], num_rows='dynamic', key='inv_classes', hide_index=True)
    class_map = {str(r['原始類別']).strip():str(r['LV 類別名稱']).strip() for r in mapping if str(r['LV 類別名稱']).strip()}
    field_map = {}
    if fmt != 'yolo':
        st.caption('填入檔案的欄位名稱；共用清單需指定圖片欄位，值須為上傳包內完整相對名稱。')
        cols = st.columns(3)
        for index,(key,label,default) in enumerate([('image','圖片欄位（單圖檔可留空）',''),('class','類別欄位','class'),('x','X／左欄位','x'),('y','Y／上欄位','y'),('w','寬／右欄位','w'),('h','高／下欄位','h'),('score','分數欄位','score')]):
            value = cols[index%3].text_input(label, value=default, key='inv_field_'+key)
            if value.strip():
                field_map[key] = value.strip()
    with st.expander('自訂圖片與紀錄配對（路徑不同時）'):
        pairing_rows = st.data_editor([{'圖片相對名稱':'','紀錄相對名稱':'','紀錄角色':'同上'}],num_rows='dynamic',hide_index=True,key='inv_pairings',column_config={'紀錄角色':st.column_config.SelectboxColumn(options=['同上','模型預測','人工標註'])})
    pairings = {r['圖片相對名稱'].strip():r['紀錄相對名稱'].strip() for r in pairing_rows if r['圖片相對名稱'].strip() and r['紀錄相對名稱'].strip()}
    annotation_roles = {r['紀錄相對名稱'].strip():{'模型預測':'prediction','人工標註':'ground_truth'}[r['紀錄角色']] for r in pairing_rows if r['紀錄相對名稱'].strip() and r['紀錄角色'] in {'模型預測','人工標註'}}
    if st.button('預覽匯入', disabled=not uploads, type='primary'):
        st.session_state['inv_preview'] = prepare_import([(u.name,u.getvalue()) for u in uploads], annotation_role=role, annotation_format=fmt, class_map=class_map, coordinates=coordinates, box_format=box_format, field_map=field_map,pairings=pairings,annotation_roles=annotation_roles)
    preview = st.session_state.get('inv_preview')
    if preview:
        rows = [{'圖片':r['name'],'紀錄':r['annotation_name'] or '未提供','角色':'模型預測' if r['annotation']['role']=='prediction' else '人工標註','狀態':{'empty':'已提供，零框','missing':'缺紀錄','provided':'配對成功','error':'紀錄有誤'}[r['annotation']['status']], '框數':len(r['annotation']['boxes']), '重複':bool(r['duplicate_of'])} for r in preview['images']]
        st.dataframe(rows, hide_index=True)
        for col,row in zip(st.columns(3),preview['images'][:3]):
            col.image(row['data'],caption=row['name'],width='stretch')
        for error in preview['errors']:
            st.warning(error['name']+'：'+error['error'])
        if preview['unpaired']:
            st.info('未配對紀錄：'+', '.join(preview['unpaired']))
        st.caption('錯誤紀錄不會成為真值；有效圖片可先保存，修正紀錄後可重新匯入。')
        if st.button('保存資料集', disabled=not name.strip() or not preview['images']):
            dataset = DatasetRegistry(store).save(name,preview)
            st.success('已保存：'+dataset['name'])
            st.session_state.pop('inv_preview',None)
    datasets = DatasetRegistry(store).list()
    if datasets:
        st.dataframe([{'名稱':d['name'],'圖片數':len(d['images']),'建立時間':d['created_at']} for d in datasets],hide_index=True)
    return datasets


def evaluation_manifest_form(store, key, purpose='case_regression'):
    datasets = DatasetRegistry(store).list()
    if not datasets:
        st.info('先在「調查資料庫」上傳評估圖片與人工標註。')
        return None
    dataset = st.selectbox('評估資料集', datasets, format_func=lambda d:d['name'], key=key+'_dataset')
    st.caption('逐張確認整圖真值；預測不算人工標註，正常區域不代表整張正常。重複內容只選一份。')
    initial = [{'納入':not bool(r['duplicate_of']),'圖片':r['name'],'真值':'待確認','已人工確認':False,'覆核者':'','框標註完整':False,'批次／群組':''} for r in dataset['images']]
    review_key = key+'_review_'+dataset['id']
    if review_key not in st.session_state:
        st.session_state[review_key] = initial
    edited = [dict(row) for row in st.session_state[review_key]]
    mode = st.radio('真值確認方式',['逐張看圖確認','表格編輯'],horizontal=True,key=review_key+'_mode')
    if mode == '表格編輯':
        edited = st.data_editor(edited,key=key+'_truth_'+dataset['id']+'_'+str(st.session_state.get(review_key+'_version',0)),hide_index=True,disabled=['圖片'],
            column_config={'真值':st.column_config.SelectboxColumn(options=['待確認','正常','瑕疵'])})
    elif edited:
        shared_reviewer = st.text_input('常用覆核者（新圖片預填）',key=review_key+'_reviewer')
        select_key = review_key+'_selected'
        pending = st.session_state.pop(review_key+'_next',None)
        if pending is not None:
            st.session_state[select_key] = pending
        index = st.selectbox('確認哪張圖片',list(range(len(edited))),format_func=lambda i:edited[i]['圖片'],key=select_key)
        record = dataset['images'][index]
        row = edited[index]
        try:
            st.image(str(store.image_path(record['source'])),caption=row['圖片'],width=600)
        except (ValueError,OSError):
            st.error('無法讀取此圖片，請先修復來源圖片再覆核。')
        st.caption('請檢查整張圖片，再決定真值；已有標註不會自動代替人工確認。')
        # A form keeps values from one image together. New keys after switching
        # editor modes ensure table corrections are also shown in this form.
        field_key = review_key+'_'+str(index)+'_'+str(st.session_state.get(review_key+'_form_version',0))
        with st.form(field_key):
            row['納入'] = st.checkbox('納入評估',value=row['納入'])
            row['真值'] = st.selectbox('整張圖片真值',['待確認','正常','瑕疵'],index=['待確認','正常','瑕疵'].index(row['真值']))
            row['已人工確認'] = st.checkbox('已人工檢查整張圖片',value=row['已人工確認'])
            row['覆核者'] = st.text_input('本張覆核者',value=row['覆核者'] or shared_reviewer)
            row['框標註完整'] = st.checkbox('已確認整張框標註完整',value=row['框標註完整'])
            row['批次／群組'] = st.text_input('本張批次／群組',value=row['批次／群組'])
            save = st.form_submit_button('保存本張確認')
            next_image = st.form_submit_button('保存並看下一張')
        if save or next_image:
            st.session_state[review_key] = edited
            st.session_state[review_key+'_version'] = st.session_state.get(review_key+'_version',0)+1
            if next_image:
                st.session_state[review_key+'_next'] = (index+1)%len(edited)
                st.rerun()
            st.success('本張確認已保存')
        else:
            edited = [dict(row) for row in st.session_state[review_key]]
    if mode == '表格編輯':
        if edited != st.session_state[review_key]:
            st.session_state[review_key+'_form_version'] = st.session_state.get(review_key+'_form_version',0)+1
        st.session_state[review_key] = edited
    independent = False
    split = ''
    if purpose == 'independent':
        independent = st.checkbox('確認未用於訓練、選樣或調整門檻', key=key+'_independent')
        split = st.text_input('獨立切分依據', key=key+'_split')
    signature = json.dumps([dataset['id'],purpose,edited,independent,split],ensure_ascii=False,sort_keys=True)
    if st.button('確認並建立評估清單', key=key+'_build'):
        confirmations = {}
        for image,row in zip(dataset['images'],edited):
            confirmations[image['id']] = {'include':row['納入'],'truth':{'正常':'normal','瑕疵':'defect'}.get(row['真值']), 'verified':row['已人工確認'],'reviewer':row['覆核者'],'annotations_complete':row['框標註完整'],'group':row['批次／群組']}
        try:
            result = build_evaluation_manifest(dataset,store,confirmations,purpose=purpose,independent_declared=independent,split_basis=split)
            st.session_state[key+'_manifest'] = result
            st.session_state[key+'_manifest_signature'] = signature
            st.success('評估清單已通過真值與資料重疊檢查')
        except (ValueError,OSError) as exc:
            st.error(str(exc))
    cached = st.session_state.get(key+'_manifest')
    return cached if cached and st.session_state.get(key+'_manifest_signature') == signature else None


def render_reply_editor(store, case):
    st.subheader('圖文回覆')
    key = 'inv_reply_'+case['id']
    from explanation_summary import QUESTION_TYPES,build_explanation_summary,render_summary_text
    with st.expander('這次要解釋什麼'):
        current_question=case.get('explanation_question',{}).get('type','class_score')
        question=st.selectbox('調查問題',list(QUESTION_TYPES),index=list(QUESTION_TYPES).index(current_question) if current_question in QUESTION_TYPES else 0,format_func=QUESTION_TYPES.get,key=key+'_question')
        if st.button('保存調查問題',key=key+'_save_question'):
            case['explanation_question']={'type':question}
            try:
                case=store.save(case)
                st.success('調查問題已保存；人工回覆文字維持原內容')
            except (ValueError,OSError) as exc:
                st.error(str(exc))
    evidence = {e['id']:e for e in case.get('evidence',[])}
    previous = case.get('reply_revisions',[])[-1] if case.get('reply_revisions') else {}
    selected = st.multiselect('附上哪些證據',list(evidence),default=[x for x in previous.get('evidence_ids',[]) if x in evidence],format_func=lambda x:evidence[x]['text'],key=key+'_evidence')
    images = {s['id']:s for s in case.get('sources',[])}
    image_ids = st.multiselect('附圖',list(images),default=[x for x in previous.get('image_ids',[]) if x in images],format_func=lambda x:images[x]['name'],key=key+'_images')
    relations = {}
    labels = {'support':'支持','opposition':'反駁','pending':'待查'}
    for eid in selected:
        prior_relation = previous.get('relations',{}).get(eid,'pending')
        relations[eid] = st.selectbox(evidence[eid]['text'],list(labels),index=list(labels).index(prior_relation),format_func=labels.get,key=key+'_rel_'+eid)
    for warning in reply_review_warnings(case,selected):
        st.warning(warning)
    for attribution in case.get('attributions',[]):
        if attribution.get('evidence_id') in selected and attribution.get('faithfulness',{}).get('rows'):
            from investigation_workspace import _render_explanation_assessment
            case=_render_explanation_assessment(store,case,attribution,show_variants=True) or case
    with st.expander('可核對的證據摘要',expanded=bool(selected)):
        st.text(render_summary_text(build_explanation_summary(case,selected)))
    if key+'_text' not in st.session_state:
        revisions = case.get('reply_revisions',[])
        st.session_state[key+'_text'] = revisions[-1]['text'] if revisions else factual_draft(case,selected)
    if st.button('依選取證據重新產生事實草稿',key=key+'_draft'):
        st.session_state[key+'_text'] = factual_draft(case,selected)
    text = st.text_area('回覆內容（可選取複製）',height=260,key=key+'_text')
    try:
        html = render_reply_html(case,store,text=text,evidence_ids=selected,image_ids=image_ids,relations=relations)
    except ValueError as exc:
        st.error(str(exc))
        st.info('請取消選取有問題的證據，或恢復分析資產後重試。已編輯文字保留。')
        return case
    with st.expander('對外圖文預覽',expanded=True):
        components.html(html,height=550,scrolling=True)
    replied = st.checkbox('此版本已回覆使用者',key=key+'_replied')
    if st.button('保存回覆版本',key=key+'_save',type='primary'):
        revision = make_reply_revision(case,text=text,evidence_ids=selected,image_ids=image_ids,relations=relations)
        revision['attribution_images']=[{k:v for k,v in image.items() if k!='data'} for image in reply_attribution_images(case,store,selected)]
        revision['reply_status'] = 'replied' if replied else 'draft'
        case.setdefault('reply_revisions',[]).append(revision)
        case['reply_status'] = revision['reply_status']
        case = store.save(case)
        st.success('回覆版本已保存；不代表根因或改善已確認')
    a,b = st.columns(2)
    a.download_button('下載含圖片回覆 HTML',html,file_name='case-reply.html',mime='text/html',key=key+'_html',on_click='ignore')
    b.download_button('下載工程證據附錄',engineering_appendix(case),file_name='case-evidence.zip',mime='application/zip',key=key+'_appendix',on_click='ignore')
    return case
