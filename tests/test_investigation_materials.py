import io
import json
import zipfile

import pytest
from PIL import Image

from fp_cases import CaseStore
from investigation_data import prepare_import, DatasetRegistry, build_evaluation_manifest, parse_annotations
from investigation_reply import render_reply_html, factual_draft, engineering_appendix, make_reply_revision


def picture(color='white'):
    stream = io.BytesIO()
    Image.new('RGB', (20, 10), color).save(stream, 'PNG')
    return stream.getvalue()


def test_mixed_archive_pairing_missing_empty_duplicate_and_errors():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        z.writestr('../escape.png', picture())
        z.writestr('a/same.png', picture())
        z.writestr('a/same.txt', '')
        z.writestr('b/same.png', picture())
        z.writestr('bad.png', b'not an image')
    result = prepare_import({'materials.zip':stream.getvalue()})
    a,b = result['images']
    assert a['annotation']['status'] == 'empty'
    assert b['annotation']['status'] == 'missing'
    assert b['duplicate_of'] == a['id']
    assert len(result['errors']) == 2


def test_formats_and_roles():
    with pytest.raises(ValueError, match='五欄'):
        parse_annotations(b'0 .5 .5 .2 .2', role='prediction', class_map={'0':'crack'})
    row = parse_annotations(b'label,left,top,right,bottom,confidence\ncrack,2,1,10,5,.8',
        role='prediction', format='csv', class_map={'crack':'scratch'}, coordinates='pixels',
        size=[20,10], box_format='xyxy', field_map={'class':'label','x':'left','y':'top','w':'right','h':'bottom','score':'confidence'})
    assert row['boxes'] == [{'class':'scratch','box':[.1,.1,.5,.5],'score':.8}]
    assert parse_annotations(b'[]', role='ground_truth', format='json')['status'] == 'empty'
    with pytest.raises(ValueError, match='類別'):
        parse_annotations(b'0 .5 .5 .2 .2 .9', role='prediction')


def test_registry_manifest_requires_human_truth_and_retains_roles(tmp_path):
    store = CaseStore(tmp_path)
    p = prepare_import({'normal.png':picture(),'normal.txt':b''}, annotation_role='ground_truth')
    registry = DatasetRegistry(store)
    dataset = registry.save('inspection', p)
    assert registry.load(dataset['id'])['images'][0]['annotation']['status'] == 'empty'
    key = dataset['images'][0]['id']
    confirmation = {key:{'truth':'normal','verified':True,'reviewer':'engineer','annotations_complete':True,'group':'lot-A'}}
    m = build_evaluation_manifest(dataset,store,confirmation)
    assert m['items'][0]['boxes'] == []
    assert m['items'][0]['group_id'] == 'lot-A'
    confirmation[key]['verified'] = False
    with pytest.raises(ValueError, match='人工'):
        build_evaluation_manifest(dataset,store,confirmation)


def test_reply_is_portable_escaped_selected_and_not_causal(tmp_path):
    store = CaseStore(tmp_path)
    source = store.ingest(picture(), '<script>.png', kind='screenshot')
    case = store.create(source, '<script>alert(1)</script>', 'Customer says good')
    case['evidence'].append({'id':'test','role':'auxiliary','text':'<img onerror=x> LV match', 'details':{'path':'C:/private/secret'},'at':'today'})
    result = render_reply_html(case,store,text='<script>bad</script>',evidence_ids=['test'],image_ids=[source['id']],relations={'test':'opposition'})
    assert '<script>' not in result and '&lt;script&gt;' in result
    assert 'data:image/png;base64,' in result and 'C:/private/secret' not in result
    assert '反駁' in result
    assert '根因' not in factual_draft(case)
    with zipfile.ZipFile(io.BytesIO(engineering_appendix(case))) as z:
        assert json.loads(z.read('case-evidence.json'))['id'] == case['id']
    with pytest.raises(ValueError):
        make_reply_revision(case,text='',evidence_ids=['missing'],image_ids=[])


def test_zip_limits_keep_valid_other_files():
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        z.writestr('large.txt',b'x'*1000)
    result=prepare_import({'bad.zip':stream.getvalue(),'ok.png':picture()},max_bytes=500)
    assert len(result['images']) == 1
    assert result['errors']


def test_master_csv_pairs_relative_names_and_maps_classes():
    result = prepare_import({'a/a.png':picture(),'b/a.png':picture('red'),
        'pred.csv':b'image,label,x,y,w,h,score\na/a.png,0,.5,.5,.2,.2,.8\nb/a.png,1,.5,.5,.4,.4,.9'},
        annotation_format='csv', class_map={'0':'crack','1':'scratch'},field_map={'image':'image','class':'label'})
    assert not result['errors']
    assert [r['annotation']['boxes'][0]['class'] for r in result['images']] == ['crack','scratch']
    assert result['unpaired'] == []


def test_bad_annotation_retains_image_and_raw_source():
    result=prepare_import({'a.png':picture(),'a.txt':b'0 .5 .5 .2 .2'},class_map={'0':'crack'})
    assert result['images'][0]['annotation']['status'] == 'error'
    assert result['images'][0]['annotation']['raw_text'] == '0 .5 .5 .2 .2'


def test_explicit_mixed_roles_do_not_invent_prediction_scores():
    result=prepare_import({'a.png':picture(),'a.txt':b'0 .5 .5 .2 .2',
                           'b.png':picture('red'),'b.txt':b'0 .5 .5 .2 .2 .9'},
                          class_map={'0':'crack'},annotation_roles={'a.txt':'ground_truth'})
    assert not result['errors']
    assert 'score' not in result['images'][0]['annotation']['boxes'][0]
    assert result['images'][1]['annotation']['boxes'][0]['score'] == .9


def test_data_library_and_reply_editor_render_without_json_requirement(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('LV_CASES_DIR',str(tmp_path))
    app=AppTest.from_string('from fp_cases import CaseStore\nfrom investigation_data_page import render_data_library\nrender_data_library(CaseStore())').run(timeout=30)
    assert not app.exception
    assert any(w.label == '資料集名稱' for w in app.text_input)
    store=CaseStore(tmp_path)
    store.create(store.ingest(picture(),'a.png'),'test','customer report')
    app=AppTest.from_string('from fp_cases import CaseStore\nfrom investigation_data_page import render_reply_editor\ns=CaseStore()\nrender_reply_editor(s,s.list()[0])').run(timeout=30)
    assert not app.exception
    assert any(w.label == '回覆內容（可選取複製）' for w in app.text_area)


def test_photo_truth_review_requires_confirmation_and_invalidates_manifest(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('LV_CASES_DIR',str(tmp_path))
    store=CaseStore(tmp_path)
    DatasetRegistry(store).save('photos',prepare_import({'a.png':picture(),'a.txt':b''},annotation_role='ground_truth'))
    app=AppTest.from_string('from fp_cases import CaseStore\nfrom investigation_data_page import evaluation_manifest_form\nr=evaluation_manifest_form(CaseStore(),"photo")\nif r: st.write("VALID_MANIFEST")\n'.replace('from fp_cases','import streamlit as st\nfrom fp_cases')).run(timeout=30)
    def widget(kind,label):
        return next(w for w in getattr(app,kind) if w.label==label)
    assert not app.exception
    assert widget('radio','真值確認方式').value=='逐張看圖確認'
    assert not widget('checkbox','已人工檢查整張圖片').value
    widget('button','確認並建立評估清單').click().run(timeout=30)
    assert app.error
    widget('selectbox','整張圖片真值').select('正常')
    widget('checkbox','已人工檢查整張圖片').check()
    widget('checkbox','已確認整張框標註完整').check()
    widget('text_input','本張覆核者').input('Engineer')
    widget('button','保存本張確認').click().run(timeout=30)
    widget('button','確認並建立評估清單').click().run(timeout=30)
    assert any(m.value=='VALID_MANIFEST' for m in app.markdown)
    widget('checkbox','已人工檢查整張圖片').uncheck()
    widget('button','保存本張確認').click().run(timeout=30)
    assert not any(m.value=='VALID_MANIFEST' for m in app.markdown)
    assert not app.exception


def test_reply_heatmap_checked_embedded_and_stale_claim_excluded(tmp_path):
    from fp_cases import digest
    store=CaseStore(tmp_path/'cases')
    case=store.create(store.ingest(picture(),'a.png'),'case')
    artifact=store.root/'cam'
    artifact.mkdir()
    data=picture('red')
    (artifact/'overlay.png').write_bytes(data)
    case['attributions']=[{'id':'cam1','artifact_dir':str(artifact),'overlay_sha256':digest(data),
                          'method':'positive','legend':'positive contribution','limitation':'not root cause'}]
    case['evidence'].append({'id':'cam-e','role':'model_attribution','text':'model heatmap',
                             'details':{'attribution_id':'cam1'},'at':'today'})
    case['claims']=[{'kind':'fact','text':'old wrong claim','refs':['cam-e'],'needs_review':True}]
    assert 'old wrong claim' not in factual_draft(case,['cam-e'])
    rendered=render_reply_html(case,store,evidence_ids=['cam-e'])
    assert '選取的模型歸因熱圖' in rendered and '需要重新核對' in rendered
    assert str(artifact) not in rendered
    (artifact/'overlay.png').write_bytes(picture('blue'))
    with pytest.raises(ValueError,match='完整性'):
        render_reply_html(case,store,evidence_ids=['cam-e'])
    case['attributions'][0]['artifact_dir']=str(tmp_path)
    with pytest.raises(ValueError,match='工作區'):
        render_reply_html(case,store,evidence_ids=['cam-e'])
