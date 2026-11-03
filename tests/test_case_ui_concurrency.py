"""Two real Streamlit sessions must not overwrite one another's form drafts."""
from PIL import Image
from streamlit.testing.v1 import AppTest

from fp_cases import CaseStore, add_item
from fp_attribution import attach_result, target_identity


def page():
    from fp_case_page import render_case_workspace
    render_case_workspace()


def widget(items, label):
    return next(w for w in items if w.label == label)


def job_page():
    import streamlit as st
    from investigation_workspace import workspace_store, render_case_job, render_case_version_notice
    store = workspace_store()
    case = store.load(store.list()[0]["id"])
    render_case_version_notice(store, case["id"])
    st.session_state.setdefault("report_draft", case["description"])
    st.text_area("Draft", key="report_draft")
    render_case_job(store, case, "comparison")
    st.write("Published searches: " + str(len(case["searches"])))


class CompletedJobs:
    def __init__(self):
        self.job = {"id": "job_comparison", "state": "queued", "task": "comparison", "stage": "waiting",
                    "progress": 0, "publication": {"state": "not_published"}}
    def list(self, **kwargs):
        return [self.job]
    def get(self, job_id):
        return self.job
    def mark_published(self, job_id, **kwargs):
        self.job["publication"] = {"state": "failed" if kwargs.get("error") else "published"}


def test_two_tabs_keep_stale_description_draft_without_overwrite(tmp_path, monkeypatch):
    root = tmp_path / "cases"
    monkeypatch.setenv("LV_CASES_DIR", str(root))
    image = tmp_path / "report.png"
    Image.new("RGB", (32, 32), "red").save(image)
    store = CaseStore(root)
    case = store.create(store.ingest(image.read_bytes(), image.name), "Concurrent case", "original description")
    a = AppTest.from_function(page, default_timeout=60).run()
    b = AppTest.from_function(page, default_timeout=60).run()
    widget(b.text_area, "報案描述").set_value("B unsaved draft")
    widget(a.text_area, "報案描述").set_value("A saved first")
    widget(a.button, "儲存報案資訊").click().run()
    assert store.load(case["id"])["description"] == "A saved first"
    first_revision = store.load(case["id"])["revision"]
    widget(b.button, "儲存報案資訊").click().run()
    assert store.load(case["id"])["description"] == "A saved first"
    assert store.load(case["id"])["revision"] == first_revision
    assert widget(b.text_area, "報案描述").value == "B unsaved draft"
    assert b.error
    widget(b.button, "開啟最新版本（保留此頁草稿）").click().run()
    assert not b.error and not b.exception
    assert widget(b.text_area, "報案描述").value == "B unsaved draft"
    assert store.load(case["id"])["description"] == "A saved first"
    widget(b.text_area, "報案描述").set_value("A saved first; B reviewed and added")
    widget(b.button, "儲存報案資訊").click().run()
    assert not b.error and not b.exception
    assert store.load(case["id"])["description"] == "A saved first; B reviewed and added"
    # This tab's own writes advance its observed version without repeated permission steps.
    widget(b.text_area, "報案描述").set_value("B second intentional save")
    widget(b.button, "儲存報案資訊").click().run()
    assert not b.error and store.load(case["id"])["description"] == "B second intentional save"


def test_background_job_publication_advances_ui_baseline_without_losing_draft(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import investigation_workspace as workspace
    root = tmp_path / "cases"
    monkeypatch.setenv("LV_CASES_DIR", str(root))
    image = tmp_path / "source.png"
    Image.new("RGB", (32, 32), "red").save(image)
    plain = CaseStore(root)
    case = plain.create(plain.ingest(image.read_bytes(), image.name), "Job case", "initial description")
    item = add_item(case, case["sources"][0]["id"], [0, 0, .5, .5], "query")
    case = plain.save(case)
    result = {"comparison": {"id": "search_job", "query_item_id": item["id"], "query_roi": item["roi"],
                "query_sha256": case["sources"][0]["sha256"], "recipe": {}, "groups": {"normal": []}},
              "case_id": case["id"], "target_identity": target_identity(case, item)}
    jobs = CompletedJobs()
    monkeypatch.setattr(workspace, "manager_for", lambda store: SimpleNamespace(store=jobs))
    monkeypatch.setattr(workspace, "completed_result", lambda store, job_id: result)
    at = AppTest.from_function(job_page, default_timeout=60).run()
    widget(at.text_area, "Draft").set_value("unsaved description")
    jobs.job.update(state="succeeded", stage="completed", progress=1)
    at.run()
    assert not at.exception and not at.error
    assert jobs.job["publication"]["state"] == "published"
    assert plain.load(case["id"])["searches"][0]["id"] == "search_job"
    assert any("Published searches: 1" in m.value for m in at.markdown)
    assert widget(at.text_area, "Draft").value == "unsaved description"
    assert not any(b.label == "開啟最新版本（保留此頁草稿）" for b in at.button)
    # A plain store publication (for example another tab) remains recoverable explicitly.
    external = {**result, "comparison": {**result["comparison"], "id": "search_other_tab"}}
    attach_result(plain, case["id"], external)
    at.run()
    assert widget(at.text_area, "Draft").value == "unsaved description"
    assert any("Published searches: 1" in m.value for m in at.markdown)
    widget(at.button, "開啟最新版本（保留此頁草稿）").click().run()
    assert not at.exception and not at.error
    assert any("Published searches: 2" in m.value for m in at.markdown)
    assert widget(at.text_area, "Draft").value == "unsaved description"
