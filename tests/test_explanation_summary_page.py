import io
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
from fp_cases import CaseStore, add_item, review_item, _evidence


def seeded_case(tmp_path,monkeypatch):
    monkeypatch.setenv('LV_CASES_DIR',str(tmp_path))
    store=CaseStore(tmp_path)
    image=io.BytesIO()
    Image.new('RGB',(32,32),'white').save(image,format='PNG')
    source=store.ingest(image.getvalue(),'sample.png')
    case=store.create(source)
    item=add_item(case,source['id'],[0,0,1,1],'suspect')
    review_item(case,item['id'],'normal','engineer','checked original')
    eid=_evidence(case,'model_attribution','saved CAM')
    case['attributions']=[dict(id='attr',item_id=item['id'],evidence_id=eid,target={},faithfulness=dict(protocol_version=1,rows=[dict(modified_sha256='a'*64)]))]
    case=store.save(case)
    return store,case,item


def test_preservation_requires_explicit_confirmation_and_persists(tmp_path,monkeypatch):
    store,case,item=seeded_case(tmp_path,monkeypatch)
    app=AppTest.from_string('''
from fp_cases import CaseStore
from investigation_workspace import _render_explanation_assessment
store=CaseStore()
case=store.list()[0]
_render_explanation_assessment(store,case,case['attributions'][0])
''',default_timeout=30).run()
    app.selectbox[0].set_value('preserved')
    app.text_input[0].set_value('reviewer')
    app.text_input[1].set_value('target remains')
    app.button[0].click().run()
    assert app.error
    assert not any(e['role']=='target_preservation_review' for e in store.load(case['id'])['evidence'])
    app.checkbox[0].check()
    app.button[0].click().run()
    assert not app.exception and not app.error
    saved=store.load(case['id'])
    review=next(e for e in saved['evidence'] if e['role']=='target_preservation_review')
    assert review['details']['variant_sha256']==['a'*64]
    assert review['details']['verdict']=='preserved'


def test_question_changes_and_reruns_preserve_human_reply(tmp_path,monkeypatch):
    store,case,item=seeded_case(tmp_path,monkeypatch)
    app=AppTest.from_string('''
from fp_cases import CaseStore
from investigation_data_page import render_reply_editor
store=CaseStore()
render_reply_editor(store,store.list()[0])
''',default_timeout=30).run()
    app.text_area[0].set_value('Engineer authored reply <keep>').run()
    next(w for w in app.selectbox if w.label=='調查問題').set_value('machine_alarm')
    next(w for w in app.button if w.label=='保存調查問題').click().run()
    assert not app.exception
    assert app.text_area[0].value=='Engineer authored reply <keep>'
    next(w for w in app.button if w.label=='保存回覆版本').click().run()
    saved=store.load(case['id'])
    assert saved['reply_revisions'][-1]['text']=='Engineer authored reply <keep>'
    assert saved['reply_revisions'][-1]['explanation_summary']['question']['type']=='machine_alarm'


def test_cohort_form_requires_matching_review_and_saves_metadata(tmp_path,monkeypatch):
    store,case,item=seeded_case(tmp_path,monkeypatch)
    item=case['items'][0]
    item['external_target']={'run_id':'seeded'}
    store.save(case)
    app=AppTest.from_string('''
from fp_cases import CaseStore
from investigation_workspace import render_case_summary
store=CaseStore()
render_case_summary(store,store.list()[0])
''',default_timeout=30).run()
    next(w for w in app.text_input if w.label=='獨立批次／群組').set_value('lot-2')
    next(w for w in app.text_input if w.label=='批次資料覆核者').set_value('reviewer')
    next(w for w in app.text_input if w.label=='角色與群組依據').set_value('production record')
    next(w for w in app.selectbox if w.label=='已核對樣本角色').set_value('true_defect')
    next(w for w in app.checkbox if w.label=='我已核實角色與群組').check()
    next(w for w in app.button if w.label=='保存批次核對資料').click().run()
    assert app.error
    assert not store.load(case['id'])['items'][0].get('cohort')
    next(w for w in app.selectbox if w.label=='已核對樣本角色').set_value('false_positive')
    next(w for w in app.button if w.label=='保存批次核對資料').click().run()
    assert not app.exception and not app.error
    metadata=store.load(case['id'])['items'][0]['cohort']
    assert metadata['verified'] is True and metadata['group_id']=='lot-2'


def test_cohort_descriptive_contrast_table_does_not_pool_targets():
    app=AppTest.from_string('''
from investigation_workspace import _render_cohort_summary
_render_cohort_summary(dict(unique_images=3,verified_images=3,unique_groups=2,partitions=[dict(identity_key='abc',roles={},contrasts=dict(rows=[dict(role='false_positive',fraction=.1,fill_method='blur',unique_images=3,unique_groups=2,mean_group_excess_drop=.125)]))]))
''',default_timeout=30).run()
    assert not app.exception
    table=app.dataframe[0].value
    assert table.iloc[0]['不同影像']==3 and table.iloc[0]['人工群組']==2
    assert table.iloc[0]['群組平均相對減分']==.125
    assert any('不是獨立性或根因證明' in c.value for c in app.caption)


@pytest.mark.parametrize('change',['review','roi'])
def test_batch_submission_invalidates_changed_cohort_without_deleting_history(tmp_path,monkeypatch,change):
    from explanation_summary import snapshot_cohort
    from types import SimpleNamespace
    import investigation_workspace
    store,case,_=seeded_case(tmp_path,monkeypatch)
    item=case['items'][0]
    item['external_target']={'run_id':'seeded'}
    item['cohort']=snapshot_cohort(case,item,metadata=dict(group_id='lot',role='false_positive',reviewer='engineer',reason='original checked',verified=True))
    original=dict(item['cohort'])
    if change=='review':review_item(case,item['id'],'defect','second reviewer','correction')
    else:item['roi']=[.1,.1,.9,.9]
    store.save(case)
    captured=[]
    def submit(task,payload,*args,**kwargs):captured.append(payload)
    monkeypatch.setattr(investigation_workspace,'manager_for',lambda store:SimpleNamespace(submit=submit))
    app=AppTest.from_string('''
from fp_cases import CaseStore
from investigation_workspace import render_case_summary
store=CaseStore()
render_case_summary(store,store.list()[0])
''',default_timeout=30).run()
    next(w for w in app.multiselect if w.label=='批次分析目標').set_value([store.load(case['id'])['items'][0]])
    next(w for w in app.button if w.label=='執行批次熱圖與遮擋驗證').click().run()
    assert not app.exception
    frozen=captured[0]['items'][0]['cohort']
    assert frozen['verified'] is False and frozen['validation_reason']
    assert frozen['human_review_evidence_id']==original['human_review_evidence_id']
    assert store.load(case['id'])['items'][0]['cohort']==original
