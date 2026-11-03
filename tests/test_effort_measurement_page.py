from PIL import Image
from streamlit.testing.v1 import AppTest

from fp_cases import CaseStore


def page():
    import os
    from fp_cases import CaseStore
    from fp_case_page import _effort
    store=CaseStore(os.environ["LV_CASES_DIR"])
    _effort(store,store.load(store.list()[0]["id"]))


def widget(items,label):
    return next(w for w in items if w.label==label)


def test_paired_effort_form_persists_quality_and_does_not_claim_unpaired_savings(tmp_path,monkeypatch):
    monkeypatch.setenv("LV_CASES_DIR",str(tmp_path/"cases"))
    store=CaseStore(tmp_path/"cases")
    image=tmp_path/"source.png";Image.new("RGB",(16,16),"white").save(image)
    case=store.create(store.ingest(image.read_bytes(),image.name),"effort")
    at=AppTest.from_function(page,default_timeout=60).run()
    assert not at.exception
    for label,value in [("研究編號","study"),("配對編號","pair"),("案例群組","lot"),("調查者","Alice"),("獨立品質覆核者","Bob")]:
        widget(at.text_input,label).set_value(value)
    widget(at.number_input,"本次實際操作分鐘").set_value(20.)
    widget(at.checkbox,"本次調查已完成").check()
    widget(at.checkbox,"已完成獨立品質覆核").check()
    widget(at.button,"記錄本次固定案例測量").click().run()
    assert not at.exception
    assert any("尚無完成配對" in x.value for x in at.caption)
    widget(at.selectbox,"測量流程").select("LV 查案流程")
    widget(at.selectbox,"配對中的執行順序").select(2)
    widget(at.number_input,"本次實際操作分鐘").set_value(10.)
    widget(at.number_input,"覆核發現的錯誤數").set_value(1)
    widget(at.button,"記錄本次固定案例測量").click().run()
    assert not at.exception
    records=store.load(case["id"])["effort_measurements"]
    assert len(records)==2 and records[1]["quality_errors"]==1
    assert any("品質未變差：0／1" in x.value for x in at.markdown)
    reopened=AppTest.from_function(page,default_timeout=60).run()
    assert not reopened.exception
    assert any("已配對 1 組" in x.value for x in reopened.markdown)
