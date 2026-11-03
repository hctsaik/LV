"""Changing the user's query must not carry over a previous CAM citation."""
import io
from types import SimpleNamespace

from PIL import Image
from streamlit.testing.v1 import AppTest

from fp_cases import CaseStore, add_item


def test_selecting_another_query_clears_previous_cam_context(tmp_path, monkeypatch):
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path))
    store = CaseStore(tmp_path)
    data = io.BytesIO()
    Image.new("RGB", (64, 64), "orange").save(data, format="PNG")
    source = store.ingest(data.getvalue(), "reported.png")
    case = store.create(source)
    first = add_item(case, source["id"], [.1, .1, .4, .8], "first")
    second = add_item(case, source["id"], [.6, .1, .9, .8], "second")
    store.save(case)
    import fp_attribution
    import investigation_workspace
    captured = []
    def submit(task, payload, compute_key, context=None):
        captured.append(payload)
        return {"id": "captured-only"}
    monkeypatch.setattr(fp_attribution, "manager_for", lambda store: SimpleNamespace(submit=submit))
    monkeypatch.setattr(investigation_workspace, "render_case_job", lambda *args: None)
    app = AppTest.from_string('''
from fp_cases import CaseStore
from fp_case_page import _comparison
store = CaseStore()
_comparison(store, store.list()[0])
''', default_timeout=30)
    context_key = "inv_search_context_" + case["id"]
    app.session_state[context_key] = {"item_id": first["id"],
        "target_identity": fp_attribution.target_identity(case, first), "attribution_evidence_id": "previous-cam"}
    app.run()
    next(w for w in app.selectbox if w.label == "作為搜尋起點的區域").set_value(second)
    next(b for b in app.button if b.label == "找對照案例").click().run()
    assert not app.exception and not app.error
    assert captured[0]["item_id"] == second["id"]
    assert captured[0]["attribution_context"] is None
    assert captured[0]["attribution_evidence_id"] is None
    assert context_key not in app.session_state
