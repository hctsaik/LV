"""CAM selection lineage is evidence, never an automatic training label."""
import copy
import io
import json
import zipfile
from types import SimpleNamespace

import pytest
from PIL import Image

from fp_cases import CaseStore, _evidence, add_item, add_candidate, review_item, revise_region, FULL_ROI
from fp_attribution import target_identity, validate_attribution_context, comparison_task, attach_result
from fp_training_export import export_training, training_rows
from fp_evidence_lineage import region_provenance


def image_bytes(color):
    data = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(data, format="PNG")
    return data.getvalue()


@pytest.fixture
def lineage(tmp_path):
    store = CaseStore(tmp_path / "cases")
    source = store.ingest(image_bytes("red"), "report.png")
    case = store.create(source, "CAM lineage")
    original = add_item(case, source["id"], [.1, .1, .4, .4], "scratch")
    cam = {"id": "cam_test", "item_id": original["id"], "target_identity": target_identity(case, original),
           "compute_id": "compute_test", "profile_id": "profile_test", "profile_revision": 2,
           "method": "signed", "target": {"model_sha256": "a" * 64, "input_sha256": "b" * 64,
           "class_name": "scratch", "output_index": 4, "box": original["roi"]},
           "artifact_dir": "C:/private/customer/models", "correspondence": {"status": "per_case_matched"},
           "historical_parity": {"status": "unavailable"}}
    cam["evidence_id"] = _evidence(case, "model_attribution", "model score evidence", item_id=original["id"], attribution_id=cam["id"])
    case["attributions"] = [cam]
    derived = add_item(case, source["id"], [.5, .5, .75, .75], "reflection", origin="cam_region")
    derived["evidence_id"] = _evidence(case, "attribution_region", "selected CAM region", item_id=derived["id"],
                                      attribution_evidence_id=cam["evidence_id"], roi=derived["roi"])
    context = {"item_id": derived["id"], "target_identity": target_identity(case, derived), "attribution_evidence_id": cam["evidence_id"]}
    reference = tmp_path / "private-reference.png"
    reference.write_bytes(image_bytes("green"))
    ref_source = store.ingest(reference.read_bytes(), reference.name, source_path=str(reference))
    row = {"path": str(reference), "source_path": str(reference), "sha256": ref_source["sha256"],
           "roi": FULL_ROI.copy(), "reason": "normal visual reference", "category": "normal", "source_kind": "original"}
    row["evidence_id"] = _evidence(case, "lv_auxiliary", "appearance only", search_id="search_test",
                                   attribution_evidence_id=cam["evidence_id"], query_item_id=derived["id"], query_roi=derived["roi"], **row)
    case["searches"].append({"id": "search_test", "query_item_id": derived["id"], "query_roi": derived["roi"],
                              "query_sha256": source["sha256"], "recipe": {"model_path": "C:/private/model"},
                              "attribution_evidence_id": cam["evidence_id"], "groups": {"normal": [row]}})
    add_candidate(case, row, store)
    candidate = case["candidates"][0]
    review_item(case, candidate["id"], "normal", "Engineer", "entire reference checked", candidate=True)
    unrelated = _evidence(case, "reported_prediction", "unrelated private information", path="C:/private/unrelated")
    case["revision"] = store.save(case)["revision"]
    return SimpleNamespace(store=store, case=case, original=original, derived=derived, cam=cam,
                           context=context, candidate=candidate, row=row, unrelated=unrelated)


def test_context_accepts_original_and_exact_derived_region(lineage):
    f = lineage
    assert validate_attribution_context(f.case, f.derived, f.context) == f.context
    original_context = {**f.context, "item_id": f.original["id"], "target_identity": target_identity(f.case, f.original)}
    assert validate_attribution_context(f.case, f.original, original_context)
    other = add_item(f.case, f.original["source_id"], [.8, .8, .9, .9], "unrelated")
    forged = {**f.context, "item_id": other["id"], "target_identity": target_identity(f.case, other)}
    with pytest.raises(ValueError, match="未連結"):
        validate_attribution_context(f.case, other, forged)
    with pytest.raises(ValueError, match="不存在"):
        validate_attribution_context(f.case, f.derived, {**f.context, "attribution_evidence_id": f.unrelated})


def test_context_rejects_changed_query_and_original_target(lineage):
    f = lineage
    revise_region(f.case, f.derived["id"], roi=[.5, .5, .8, .8], reason="new query")
    with pytest.raises(ValueError, match="搜尋區域已變更"):
        validate_attribution_context(f.case, f.derived, f.context)
    renewed = {**f.context, "target_identity": target_identity(f.case, f.derived)}
    with pytest.raises(ValueError, match="未連結"):
        validate_attribution_context(f.case, f.derived, renewed)
    f.derived["roi"] = [.5, .5, .75, .75]
    revise_region(f.case, f.original["id"], withdraw=True, reason="wrong original box")
    with pytest.raises(ValueError, match="原熱圖模型目標已變更"):
        validate_attribution_context(f.case, f.derived, f.context)


def test_comparison_requires_context_before_search_and_revalidates_publication(lineage, monkeypatch):
    f = lineage
    calls = []
    def fake_search(*args, **kwargs):
        calls.append(True)
        return {"id": "search_new", "query_item_id": f.derived["id"], "query_roi": f.derived["roi"],
                "query_sha256": f.case["sources"][0]["sha256"], "recipe": {}, "groups": {"normal": []}}
    monkeypatch.setattr("fp_case_search.search", fake_search)
    context = SimpleNamespace(workspace_root=f.store.root, progress=lambda *args: None)
    payload = {"case_id": f.case["id"], "item_id": f.derived["id"], "target_identity": f.context["target_identity"],
               "attribution_evidence_id": f.cam["evidence_id"]}
    with pytest.raises(ValueError, match="目標紀錄"):
        comparison_task(payload, context)
    assert not calls
    result = comparison_task({**payload, "attribution_context": f.context}, context)
    assert result["comparison"]["attribution_context"] == f.context
    revise_region(f.case, f.original["id"], withdraw=True, reason="changed while searching")
    f.store.save(f.case)
    with pytest.raises(ValueError, match="原熱圖模型目標已變更"):
        attach_result(f.store, f.case["id"], result)


def test_training_zip_carries_only_selected_transitive_lineage(lineage):
    f = lineage
    candidate_row = next(r for r in training_rows(f.case) if r["id"].startswith("candidate:"))
    assert candidate_row["evidence_id"] == f.candidate["evidence_id"]
    record, data = export_training(f.case, f.store, complete_images=[f.candidate["source"]["sha256"]],
                                  class_assignments={}, class_names=["scratch"], reviewer="Engineer", reason="complete image review")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        portable = json.loads(archive.read("provenance.json"))
        assert archive.read("labels/" + f.candidate["source"]["sha256"] + ".txt") == b""
    chain = portable["lineage"]
    assert {e["id"] for e in chain["evidence"]} == {f.candidate["evidence_id"], f.cam["evidence_id"]}
    assert chain["searches"][0]["id"] == "search_test"
    assert chain["attributions"][0]["compute_id"] == "compute_test"
    assert chain["attributions"][0]["target_current"] is True
    assert chain["regions"][0]["review"]["reviewer"] == "Engineer"
    assert chain["regions"][0]["association_semantics"] == "historical_selection"
    serialized = json.dumps(portable)
    assert "C:/private" not in serialized and str(f.store.root) not in serialized
    assert f.unrelated not in serialized
    assert portable["lineage"] == record["lineage"]


def test_corrected_drawn_region_keeps_historical_chain_without_current_claim(lineage):
    f = lineage
    old_ref = f.derived["evidence_id"]
    revise_region(f.case, f.derived["id"], roi=[.5, .5, .8, .8], reason="adjust reflection")
    revise_region(f.case, f.original["id"], withdraw=True, reason="old model box withdrawn")
    row = next(r for r in training_rows(f.case) if r["id"] == "item:" + f.derived["id"])
    result = region_provenance(f.case, [row])
    assert old_ref in {e["id"] for e in result["evidence"]}
    assert result["attributions"][0]["target_current"] is False
    assert result["regions"][0]["roi"] == [.5, .5, .8, .8]
    assert result["regions"][0]["review"]["verdict"] == "pending"


def auxiliary_navigation_page():
    import streamlit as st
    from investigation_workspace import render_detector_diagnostics, workspace_store
    from fp_case_page import _comparison
    if st.session_state.get("investigation_diagnostic"):
        render_detector_diagnostics()
    else:
        store = workspace_store()
        _comparison(store, store.load(st.session_state["investigation_next_case"]))


def test_auxiliary_navigation_selects_current_region_and_clears_cam_context(lineage, monkeypatch):
    from streamlit.testing.v1 import AppTest
    f = lineage
    # This story starts a fresh search; the minimal lineage fixture is not a UI search result.
    f.case["searches"] = []
    f.case = f.store.save(f.case)
    monkeypatch.setenv("LV_CASES_DIR", str(f.store.root))
    at = AppTest.from_function(auxiliary_navigation_page, default_timeout=60)
    at.session_state["investigation_diagnostic"] = {"case_id": f.case["id"], "item_id": f.derived["id"]}
    at.session_state["inv_diagnostic_source"] = "LV 輔助對照"
    at.session_state["inv_search_item_" + f.case["id"]] = f.original["id"]
    at.session_state["inv_search_context_" + f.case["id"]] = f.context
    at.session_state["inv_search_evidence_" + f.case["id"]] = f.cam["evidence_id"]
    at.run()
    next(b for b in at.button if b.label == "以此區域找對照").click().run()
    assert not at.exception and not at.error
    selected = next(w for w in at.selectbox if w.label == "作為搜尋起點的區域").value
    assert selected["id"] == f.derived["id"]
    assert "inv_search_context_" + f.case["id"] not in at.session_state.filtered_state
    assert "inv_search_evidence_" + f.case["id"] not in at.session_state.filtered_state


def test_reusing_drawn_roi_links_new_cam_without_losing_previous_lineage(lineage):
    from investigation_workspace import _cam_comparison_region
    f = lineage
    old_region_evidence = f.derived["evidence_id"]
    second = {**copy.deepcopy(f.cam), "id": "cam_second", "compute_id": "second_compute"}
    second["evidence_id"] = _evidence(f.case, "model_attribution", "second CAM", item_id=f.original["id"], attribution_id=second["id"])
    f.case["attributions"].append(second)
    count = len(f.case["items"])
    selected, changed = _cam_comparison_region(f.case, f.original, second["evidence_id"], f.derived["roi"])
    assert changed and selected["id"] == f.derived["id"] and len(f.case["items"]) == count
    assert selected["evidence_id"] != old_region_evidence
    assert old_region_evidence in {e["id"] for e in f.case["evidence"]}
    context = {"item_id": selected["id"], "target_identity": target_identity(f.case, selected),
               "attribution_evidence_id": second["evidence_id"]}
    assert validate_attribution_context(f.case, selected, context) == context
    links = [e for e in f.case["evidence"] if e["role"] == "attribution_region" and e["details"].get("item_id") == selected["id"]]
    assert {e["details"]["attribution_evidence_id"] for e in links} == {f.cam["evidence_id"], second["evidence_id"]}
    before = len(f.case["evidence"])
    assert _cam_comparison_region(f.case, f.original, second["evidence_id"], f.derived["roi"])[1] is False
    assert len(f.case["evidence"]) == before


def cam_navigation_widget_page():
    import streamlit as st
    from investigation_workspace import _render_attribution_result,workspace_store
    st.selectbox('Tool',['Model Diagnostics','誤逮案件'],key='tool_switch')
    store=workspace_store()
    case=store.load(st.session_state['test_case_id'])
    if st.session_state['tool_switch']=='Model Diagnostics':
        item=next(i for i in case['items'] if i['id']==st.session_state['test_item_id'])
        _render_attribution_result(store,case,item,st.session_state['test_cam'],'signed')
    else:
        st.write('Returned to comparison')


@pytest.mark.parametrize('withdraw_target',[False,True])
def test_cam_navigation_callback_after_real_tool_widget(lineage,monkeypatch,tmp_path,withdraw_target):
    import numpy as np
    from streamlit.testing.v1 import AppTest
    f=lineage
    root=tmp_path/'cam'; root.mkdir()
    for name in ['source.png','model_input.png','overlay.png']:
        (root/name).write_bytes(image_bytes('red'))
    np.savez(root/'maps.npz',source=np.ones((32,32),dtype=np.float32))
    cam={**f.cam,'artifact_dir':str(root),'status':'ok'}
    monkeypatch.setenv('LV_CASES_DIR',str(f.store.root))
    at=AppTest.from_function(cam_navigation_widget_page,default_timeout=60)
    at.session_state['test_case_id']=f.case['id']
    at.session_state['test_item_id']=f.original['id']
    at.session_state['test_cam']=cam
    at.run()
    assert not at.exception
    if withdraw_target:
        revise_region(f.case,f.original['id'],withdraw=True,reason='changed in other tab')
        f.store.save(f.case)
    next(b for b in at.button if b.label=='從此區域找同類案例').click().run()
    assert not at.exception
    if withdraw_target:
        assert at.session_state['tool_switch']=='Model Diagnostics'
        assert any('無法開啟對照搜尋' in e.value for e in at.error)
        assert 'inv_search_context_'+f.case['id'] not in at.session_state.filtered_state
    else:
        assert at.session_state['tool_switch']=='誤逮案件'
        context=at.session_state['inv_search_context_'+f.case['id']]
        assert context['item_id']==f.original['id']
        assert context['attribution_evidence_id']==f.cam['evidence_id']
        assert not at.error
