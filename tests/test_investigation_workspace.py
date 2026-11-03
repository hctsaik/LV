"""Workflow contracts without loading any inference or gradient model."""
import copy
import hashlib
import io
import json
from types import SimpleNamespace

from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

import fp_cases as cases
import investigation_workspace as workspace
from investigation_jobs import JobStore


def _source(store, color="white", kind="original"):
    content = io.BytesIO()
    Image.new("RGB", (24, 16), color).save(content, format="PNG")
    return store.ingest(content.getvalue(), color + ".png", kind=kind)


def _dataset(store, *, kind="original", annotations=None):
    annotations = annotations or [
        {"role": "prediction", "status": "provided", "boxes": [{"box": [.1, .2, .6, .8], "class": "scratch", "score": .8}]},
        {"role": "ground_truth", "status": "missing", "boxes": []},
    ]
    return {"id": "dataset-workflow", "name": "Imported set", "images": [
        {"id": f"image-{i}", "name": f"image-{i}.png", "source": _source(store, color, kind), "annotation": annotation}
        for i, (color, annotation) in enumerate(zip(["white", "black", "red"], annotations))]}


def test_multi_image_case_import_is_idempotent_and_preserves_pending_review(tmp_path):
    store = cases.CaseStore(tmp_path)
    dataset = _dataset(store)
    created = workspace.import_dataset_cases(store, dataset, name="Incoming report")
    assert len(created) == 1
    case = created[0]
    assert len(case["sources"]) == 2 and len(case["items"]) == 1
    assert case["items"][0]["review"]["verdict"] == "pending"
    assert case["items"][0]["origin"] == "imported_prediction"
    assert "external_target" not in case["items"][0]
    before = copy.deepcopy(case)
    repeated = workspace.import_dataset_cases(store, dataset, case_id=case["id"])[0]
    assert repeated == before
    assert len(store.list()) == 1


def test_split_import_and_empty_prediction_do_not_invent_normal_truth(tmp_path):
    store = cases.CaseStore(tmp_path)
    dataset = _dataset(store, annotations=[
        {"role": "prediction", "status": "empty", "boxes": []},
        {"role": "ground_truth", "status": "missing", "boxes": []},
    ])
    created = workspace.import_dataset_cases(store, dataset, name="Batch", split=True)
    assert len(created) == 2
    for case in created:
        assert len(case["sources"]) == 1
        assert case["items"] == []
        assert case["candidates"] == []
        assert not case.get("training_exports")
        assert len([e for e in case["evidence"] if e["role"] == "imported_record"]) == 1


def test_imported_ground_truth_boxes_still_require_case_review(tmp_path):
    store = cases.CaseStore(tmp_path)
    dataset = _dataset(store, annotations=[{"role": "ground_truth", "status": "provided",
        "boxes": [{"box": [.1, .2, .6, .8], "class": "scratch"}]}])
    case = workspace.import_dataset_cases(store, dataset)[0]
    assert case["items"][0]["origin"] == "imported_ground_truth"
    assert case["items"][0]["score"] is None
    assert case["items"][0]["review"]["verdict"] == "pending"


def test_known_screenshot_cannot_become_an_original_through_dataset_import(tmp_path):
    store = cases.CaseStore(tmp_path)
    dataset = _dataset(store, kind="screenshot")
    try:
        imported = workspace.import_dataset_cases(store, dataset, kind="original")
    except ValueError:
        return  # Rejecting incompatible origin claims is also acceptable.
    assert all(source["kind"] == "screenshot" or source.get("original_kind") == "screenshot"
               for case in imported for source in case["sources"])


def test_region_revision_invalidates_target_review_and_related_claim_without_rewriting_history(tmp_path):
    store = cases.CaseStore(tmp_path)
    source = _source(store)
    case = store.create(source)
    item = cases.add_item(case, source["id"], [.1, .2, .6, .8], "scratch", .8)
    item["external_target"] = {"run_id": "r1", "output_index": 3, "class": "scratch"}
    cases.review_item(case, item["id"], "normal", "Engineer", "Normal texture")
    cases.add_claim(case, "fact", "Reviewed the reported region", [item["evidence_id"]])
    before = copy.deepcopy(item)
    old_evidence = copy.deepcopy(case["evidence"])
    previous_identity = workspace.target_identity(case, item)
    cases.revise_region(case, item["id"], roi=[.2, .2, .7, .8], reason="Correct the region")
    assert item["review"]["verdict"] == "pending"
    assert "external_target" not in item
    assert workspace.target_identity(case, item) != previous_identity
    assert case["region_history"][-1]["item"] == before
    assert case["evidence"][:len(old_evidence)] == old_evidence
    assert case["claims"][0]["needs_review"]
    cases.revise_region(case, item["id"], withdraw=True, reason="Wrong region")
    assert cases.active_rows(case["items"]) == []
    cases.revise_region(case, item["id"], reason="Restore for fresh review")
    assert len(cases.active_rows(case["items"])) == 1
    assert item["review"]["verdict"] == "pending"


def test_invalid_or_duplicate_region_correction_has_no_partial_mutation(tmp_path):
    store = cases.CaseStore(tmp_path)
    source = _source(store)
    case = store.create(source)
    first = cases.add_item(case, source["id"], [.1, .1, .4, .4], "scratch")
    cases.add_item(case, source["id"], [.5, .5, .8, .8], "scratch")
    before = copy.deepcopy(case)
    for roi in ([.5, .5, .8, .8], [2, .2, .8, .8]):
        with pytest.raises(ValueError):
            cases.revise_region(case, first["id"], roi=roi, reason="Correction")
        assert case == before


def test_navigation_returns_to_same_case_without_starting_work(tmp_path, monkeypatch):
    import investigation_models_page
    store = cases.CaseStore(tmp_path)
    case = store.create(_source(store), "Navigation case")
    jobs = JobStore(tmp_path)
    monkeypatch.setattr(workspace, "manager_for", lambda _store: SimpleNamespace(store=jobs))
    def library(_store, _manager):
        import streamlit as st
        st.write("Model library route reached")
        st.button("Return", on_click=workspace.go_area, args=("案件", case["id"]))
    monkeypatch.setattr(investigation_models_page, "render_model_library", library)
    app = AppTest.from_string(f'''
import streamlit as st
from fp_cases import CaseStore
from investigation_workspace import go_area, render_navigation
store = CaseStore({str(tmp_path)!r})
st.button("Models", on_click=go_area, args=("我的模型", {case['id']!r}))
render_navigation(store)
''').run(timeout=30)
    next(b for b in app.button if b.label == "Models").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["fp_case_id"] == case["id"]
    assert app.session_state["investigation_area"] == "我的模型"
    next(b for b in app.button if b.label == "Return").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["investigation_area"] == "案件"
    assert app.session_state["fp_case_id"] == case["id"]
    assert jobs.list() == []


def test_new_completed_job_is_published_by_case_worker_view(tmp_path, monkeypatch):
    store = cases.CaseStore(tmp_path)
    case = store.create(_source(store))
    jobs = JobStore(tmp_path)
    job = jobs.submit("comparison", {}, "comparison-input", context={"case_id": case["id"]})
    jobs._update(job["id"], {"state": "succeeded"})
    monkeypatch.setattr(workspace, "manager_for", lambda _store: SimpleNamespace(store=jobs))
    monkeypatch.setattr(workspace, "completed_result", lambda _store, _id: {"comparison": {"id": "search"}})
    attached = []
    monkeypatch.setattr(workspace, "attach_result", lambda _store, case_id, result: attached.append(case_id))
    app = AppTest.from_string(f'''
from fp_cases import CaseStore
from investigation_workspace import render_case_job
store = CaseStore({str(tmp_path)!r})
render_case_job(store, store.load({case['id']!r}), "comparison")
''').run(timeout=30)
    assert not app.exception
    assert attached == [case["id"]]
    assert jobs.get(job["id"])["publication"]["state"] == "published"


def _complete(jobs, job_id, result):
    directory = jobs.artifact_dir(job_id)
    directory.mkdir(parents=True)
    data = json.dumps(result).encode()
    (directory / "result.json").write_bytes(data)
    jobs._update(job_id, {"state": "succeeded", "artifacts": [{"path": "result.json", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}]})


def test_batch_coordinator_keeps_completed_child_when_other_child_cancelled_and_retried(tmp_path):
    from fp_attribution import batch_gradcam_task, completed_result
    store = cases.CaseStore(tmp_path)
    jobs = JobStore(tmp_path)
    payload = {"items": [{"case_id": "case-a", "item_id": "one", "target_identity": "target-one"},
                         {"case_id": "case-a", "item_id": "two", "target_identity": "target-two"}]}
    parent = jobs.submit("batch_gradcam", payload, "batch-input")
    context = SimpleNamespace(workspace_root=tmp_path, check_cancelled=lambda: None)
    scheduled = batch_gradcam_task(payload, context)
    assert len(scheduled["child_jobs"]) == 2
    assert all(jobs.get(job_id)["state"] == "queued" for job_id in scheduled["child_jobs"])
    _complete(jobs, parent["id"], scheduled)
    first, second = scheduled["child_jobs"]
    _complete(jobs, first, {"attribution": {"item_id": "one", "faithfulness": {"status": "unavailable"}}})
    jobs.cancel(second)
    partial = completed_result(store, parent["id"])
    assert partial["succeeded"] == 1 and partial["support"] == 2
    assert partial["items"][0]["result"]["attribution"]["item_id"] == "one"
    assert partial["items"][1]["status"] == "cancelled"
    retry = jobs.retry(second)
    _complete(jobs, retry["id"], {"attribution": {"item_id": "two", "faithfulness": {"status": "unavailable"}}})
    recovered = completed_result(store, parent["id"])
    assert recovered["succeeded"] == 2
    assert recovered["items"][0]["job_id"] == first
    assert recovered["items"][1]["job_id"] == retry["id"]


def test_imported_prediction_comparison_does_not_require_or_invent_historical_identity(tmp_path):
    from fp_attribution import compare_imported_records
    store = cases.CaseStore(tmp_path)
    source = _source(store)
    case = store.create(source)
    annotation = {"role": "prediction", "status": "provided", "boxes": [{"box": [.1, .1, .8, .8], "class": "scratch"}]}
    cases._evidence(case, "imported_record", "User supplied prediction", source_id=source["id"], annotation=annotation)
    run = {"prediction": {"detections": [{"box": [.1, .1, .8, .8], "class": "scratch", "score": .8}]}}
    compared = compare_imported_records(case, source, run)[0]
    assert compared["status"] == "partial_fields_match"
    assert compared["missing_identity"]
    assert "contract_id" not in run and "image_sha256" not in annotation
    run["prediction"]["detections"][0]["class"] = "other"
    assert compare_imported_records(case, source, run)[0]["status"] == "partial_fields_differ"


def test_evaluation_publication_preserves_gate_and_correct_collection(tmp_path):
    from fp_attribution import attach_result
    store = cases.CaseStore(tmp_path)
    case = store.create(_source(store))
    evaluation = {"id": "evaluation-fixture", "gate": {"status": "failed", "reasons": ["Miss rate increased"]},
                  "manifest": {"purpose": "independent"}, "dataset_fingerprint": "fixture", "baseline": {}, "candidate": {}}
    saved = attach_result(store, case["id"], {"evaluation": evaluation, "case_id": case["id"]})
    assert saved["evaluations"][0]["gate"] == evaluation["gate"]
    assert "evaluation" not in saved and "evaluationss" not in saved
    assert "failed" in cases.report_markdown(saved)
    same = attach_result(store, case["id"], {"evaluation": evaluation, "case_id": case["id"]})
    assert len(same["evaluations"]) == 1


def test_fresh_session_opens_restored_workspace_from_url_and_can_return_to_original(tmp_path, monkeypatch):
    from investigation_backup import create_backup, restore_backup
    default_root = tmp_path / "cases"
    monkeypatch.setenv("LV_CASES_DIR", str(default_root))
    store = cases.CaseStore()
    original = store.create(_source(store), "Original workspace case")
    restored_root = tmp_path / "cases_restored_session"
    restore_backup(create_backup(default_root), restored_root)
    restored = cases.CaseStore(restored_root)
    revised = restored.load(original["id"])
    revised["title"] = "Restored workspace case"
    restored.save(revised)
    script = '''
import streamlit as st
from investigation_workspace import workspace_store, render_navigation
store = workspace_store()
render_navigation(store)
st.session_state["observed_root"] = str(store.root)
st.write(store.list()[0]["title"])
'''
    fresh = AppTest.from_string(script, default_timeout=30)
    fresh.query_params["inv_workspace"] = restored_root.name
    fresh.query_params["inv_case"] = original["id"]
    fresh.run(timeout=30)
    assert not fresh.exception
    assert fresh.session_state["observed_root"] == str(restored_root)
    assert fresh.session_state["fp_case_id"] == original["id"]
    assert any(m.value == "Restored workspace case" for m in fresh.markdown)
    next(b for b in fresh.button if b.label == "返回原工作區").click().run(timeout=30)
    assert not fresh.exception
    assert fresh.session_state["observed_root"] == str(default_root)
    assert "inv_workspace" not in fresh.query_params
    assert any(m.value == "Original workspace case" for m in fresh.markdown)
    assert store.load(original["id"])["title"] == "Original workspace case"


@pytest.mark.parametrize("query", ["../outside", "cases_restored_missing", "C:/outside", "cases_restored_bad/../outside"])
def test_workspace_url_rejects_traversal_or_missing_restore_receipt(tmp_path, monkeypatch, query):
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "cases"))
    app = AppTest.from_string('''
import streamlit as st
from investigation_workspace import workspace_store
workspace_store()
st.write("Unexpected workspace access")
''', default_timeout=30)
    app.query_params["inv_workspace"] = query
    app.run(timeout=30)
    assert not app.exception
    assert any("還原工作區位置無效" in item.value for item in app.error)
    assert not any(item.value == "Unexpected workspace access" for item in app.markdown)
    assert not (tmp_path / "outside").exists()
