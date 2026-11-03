"""Controlled completed-job UI fixture: save must immediately reload assessment."""
import io
from types import SimpleNamespace

from PIL import Image
from streamlit.testing.v1 import AppTest

from fp_cases import CaseStore, add_item, _evidence
import investigation_workspace as workspace


def test_first_cam_save_refreshes_summary_without_second_user_action(tmp_path, monkeypatch):
    store = CaseStore(tmp_path)
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    source = store.ingest(image.getvalue(), "sample.png")
    case = store.create(source)
    item = add_item(case, source["id"], [0, 0, 1, 1], "scratch")
    item["external_target"] = {"run_id": "run", "output_index": 0, "class": "scratch"}
    case["external_runs"] = [{"id": "run", "profile_snapshot": {
        "id": "profile", "revision": 1, "gradient_asset": {"format": "pt"}},
        "prediction": {"detections": [{"output_index": 0, "class": "scratch", "score": .5}]}}]
    case = store.save(case)
    report = {"id": "cam", "item_id": item["id"], "target": {}, "faithfulness": {}}
    job = {"id": "job", "task": "gradcam", "state": "succeeded", "payload": {
        "item_id": item["id"], "method": "positive", "scalar": "confidence",
        "fractions": [.1, .25, .5], "controls": 3, "randomization": False}}
    publications = []
    manager = SimpleNamespace(store=SimpleNamespace(
        list=lambda **kwargs: [job], get=lambda job_id: job,
        mark_published=lambda *args, **kwargs: publications.append(args[0])))
    monkeypatch.setattr(workspace, "workspace_store", lambda: store)
    monkeypatch.setattr(workspace, "manager_for", lambda value: manager)
    monkeypatch.setattr(workspace, "_job_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(workspace, "completed_result", lambda *args: {"attribution": report})
    # Keep the actual assessment UI, replacing only image/ZIP presentation.
    monkeypatch.setattr(workspace, "_render_attribution_result",
                        lambda store, case, item, result, method:
                        workspace._render_explanation_assessment(store, case, result))
    def attach(store, case_id, result):
        fresh = store.load(case_id)
        eid = _evidence(fresh, "model_attribution", "controlled completed CAM")
        fresh["attributions"] = [{**result["attribution"], "evidence_id": eid}]
        return store.save(fresh)
    monkeypatch.setattr(workspace, "attach_result", attach)
    app = AppTest.from_string("from investigation_workspace import render_detector_diagnostics\nrender_detector_diagnostics()",
                             default_timeout=30)
    app.session_state["investigation_diagnostic"] = {"case_id": case["id"], "item_id": item["id"]}
    app.run()
    assert not app.exception
    assert any("先保存為案件證據" in message.value for message in app.info)
    next(button for button in app.button if button.label == "保存為案件證據").click().run()
    assert not app.exception and not app.error
    assert publications == ["job"]
    assert not any("先保存為案件證據" in message.value for message in app.info)
    assert any(message.value == "熱圖證據已加入案件，可於圖文回覆引用。" for message in app.success)
    assert any(expander.label == "指定分數的受控證據摘要" for expander in app.expander)
    app.run()
    assert not app.exception and not app.success  # flash notice is consumed once
    assert publications == ["job"]
