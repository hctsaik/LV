import io

from PIL import Image
from streamlit.testing.v1 import AppTest

from fp_cases import CaseStore


def _app():
    from fp_case_page import render_case_workspace
    render_case_workspace()


def test_drawing_updates_existing_form_but_preserves_manual_coordinate_edits(tmp_path, monkeypatch):
    monkeypatch.setenv("LV_CASES_DIR",str(tmp_path))
    store=CaseStore(tmp_path)
    output=io.BytesIO()
    Image.new("RGB",(64,64),"orange").save(output,format="PNG")
    store.create(store.ingest(output.getvalue(),"reported.png"))
    import investigation_workspace as workspace
    def drawing(*args):
        import streamlit as st
        return st.session_state.get("test_drawing")
    monkeypatch.setattr(workspace,"graphical_regions",drawing)
    app=AppTest.from_function(_app,default_timeout=30).run()
    assert not app.exception
    app.session_state["test_drawing"]=[.1,.2,.7,.8]
    app.run()
    coordinate=lambda label:next(w for w in app.number_input if w.label==label)
    assert coordinate("左 %").value==10.0
    assert coordinate("右 %").value==70.0
    next(b for b in app.button if b.label=="加入爭議區域").click().run()
    assert not app.exception and not app.error
    assert store.list()[0]["items"][0]["roi"]==[.1,.2,.7,.8]
    coordinate("左 %").set_value(20.0)
    next(b for b in app.button if b.label=="加入爭議區域").click().run()
    assert not app.exception and not app.error
    assert store.list()[0]["items"][-1]["roi"]==[.2,.2,.7,.8]


def test_newly_published_run_replaces_previous_visible_run(tmp_path, monkeypatch):
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path))
    store = CaseStore(tmp_path)
    output = io.BytesIO()
    Image.new("RGB", (64, 64), "orange").save(output, format="PNG")
    source = store.ingest(output.getvalue(), "two-runs.png")
    case = store.create(source)
    case["external_runs"] = [{"id": rid, "source_id": source["id"], "prediction": {"detections": []}}
                             for rid in ["old-run", "new-run"]]
    store.save(case)
    app = AppTest.from_string('''
from fp_cases import CaseStore
from investigation_workspace import graphical_regions
store = CaseStore()
case = store.list()[0]
graphical_regions(store, case, case['sources'][0])
''', default_timeout=30)
    app.query_params["inv_run"] = "old-run"
    app.run()
    assert app.selectbox[0].value == "old-run"
    app.session_state["inv_pending_visible_run_" + source["id"]] = "new-run"
    app.run()
    assert not app.exception
    assert app.selectbox[0].value == "new-run"
    assert app.query_params["inv_run"] == ["new-run"]
