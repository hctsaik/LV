import io
from types import SimpleNamespace

import pytest
from PIL import Image

from fp_cases import CaseStore, add_item, FULL_ROI
from fp_onnx_demo import make_demo_model
from fp_investigation import run_case, run_experiment


@pytest.fixture
def source_run(tmp_path):
    store=CaseStore(tmp_path/'cases')
    out=io.BytesIO()
    Image.new('RGB',(32,32),(150,150,150)).save(out,'PNG')
    case=store.create(store.ingest(out.getvalue(),'source.png'))
    item=add_item(case,case['sources'][0]['id'],FULL_ROI,'crack')
    path=make_demo_model(tmp_path/'m.onnx')
    config={'model_path':str(path),'output_format':'classification','input_size':[32,32],
            'class_names':['good','crack'],'normal_classes':['good'],'score_activation':'softmax'}
    run=run_case(case,item['id'],store,config)
    run.pop('item_id')
    run.pop('context_id')
    run['source_context_id']=case['sources'][0]['sha256']
    case.pop('runtime_profiles',None)
    return store,case,item,run


def test_source_first_uses_explicit_selected_item(source_run):
    store,case,item,run=source_run
    progress=[]
    result=run_experiment(case,run['id'],store,target_item_id=item['id'],target_class='crack',
        intervention_roi=[0,0,.3,.3],mode='brightness',target_preserved=True,progress=lambda d,t:progress.append((d,t)))
    assert result['item_id']==item['id']
    assert result['source_id']==item['source_id']
    assert result['context_id']
    assert progress[-1]==(2,2)


def test_durable_experiment_executes_saved_source_first_run(source_run):
    from fp_attribution import experiment_task, target_identity
    store, case, item, run = source_run
    store.save(case)
    progress = []
    context = SimpleNamespace(workspace_root=store.root, check_cancelled=lambda: None,
        progress=lambda fraction, message: progress.append((fraction, message)))
    payload = {"case_id": case["id"], "baseline_id": run["id"],
        "target_item_id": item["id"], "target_identity": target_identity(case, item),
        "kwargs": {"target_class": "crack", "intervention_roi": [0, 0, .3, .3],
                   "mode": "brightness", "target_preserved": True}}
    result = experiment_task(payload, context)
    assert result["experiment"]["item_id"] == item["id"]
    assert result["experiment"]["source_id"] == item["source_id"]
    assert result["source"]["sha256"] == case["sources"][0]["sha256"]
    assert progress[-1][0] == 1


def test_missing_withdrawn_and_wrong_source_rejected(source_run):
    store,case,item,run=source_run
    kwargs={'target_class':'crack','intervention_roi':[0,0,.3,.3]}
    with pytest.raises(ValueError,match='目標區域'):
        run_experiment(case,run['id'],store,**kwargs)
    item['withdrawn']=True
    with pytest.raises(ValueError,match='目標區域'):
        run_experiment(case,run['id'],store,target_item_id=item['id'],**kwargs)
    item['withdrawn']=False
    original=item['source_id']
    item['source_id']='other-source'
    with pytest.raises(ValueError,match='不屬於'):
        run_experiment(case,run['id'],store,target_item_id=item['id'],**kwargs)
    item['source_id']=original
    item['external_target']={'run_id':'other-run','class':'crack'}
    with pytest.raises(ValueError,match='另一筆'):
        run_experiment(case,run['id'],store,target_item_id=item['id'],**kwargs)


def test_source_first_experiment_page_needs_no_legacy_profiles(source_run,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import fp_attribution
    store,case,item,run=source_run
    case=store.save(case)
    monkeypatch.setenv('LV_CASES_DIR',str(store.root))
    monkeypatch.setattr(fp_attribution,'manager_for',lambda store:SimpleNamespace(store=SimpleNamespace(list=lambda **kwargs:[])))
    app=AppTest.from_string("import streamlit as st\nfrom fp_cases import CaseStore\nfrom fp_external_page import render_external\ns=CaseStore(); c=s.list()[0]\nst.session_state['fp_external_mode_'+c['id']]='對照實驗'\nrender_external(s,c)").run(timeout=30)
    assert not app.exception
    assert any(s.label=='固定觀察的區域' for s in app.selectbox)
    assert any(b.label=='執行對照實驗' for b in app.button)


def test_increased_score_is_visible_without_promoting_cause(source_run, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import fp_attribution
    store, case, item, run = source_run
    result = run_experiment(case, run['id'], store, target_item_id=item['id'],
        target_class='crack', intervention_roi=FULL_ROI, mode='brightness',
        factor=1.5, target_preserved=True)
    assert result['variants'][0]['score_drop'] < 0
    store.save(case)
    monkeypatch.setenv('LV_CASES_DIR', str(store.root))
    monkeypatch.setattr(fp_attribution, 'manager_for', lambda store: SimpleNamespace(
        store=SimpleNamespace(list=lambda **kwargs: [])))
    app = AppTest.from_string("import streamlit as st\nfrom fp_cases import CaseStore\nfrom fp_external_page import render_external\ns=CaseStore(); c=s.list()[0]\nst.session_state['fp_external_mode_'+c['id']]='對照實驗'\nrender_external(s,c)", default_timeout=30).run()
    assert not app.exception
    text = '\n'.join(m.value for m in app.markdown)
    assert '分數上升：' in text and '分數未變' in text
    assert '分數下降：-' not in text
    assert store.load(case['id'])['cause_status'] == 'unknown'
