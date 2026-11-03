"""Behavioral gates for case isolation, partial human review and safe AL linkage."""
import copy
import io
import json
import sqlite3
import zipfile

import pytest
from PIL import Image

import fp_cases as fp


def png(color="red"):
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), color).save(buf, format="PNG")
    return buf.getvalue()


def test_repair_source_preserves_identity_and_rejects_replacement(tmp_path):
    store = fp.CaseStore(tmp_path / "repair")
    original = png()
    source = store.ingest(original, "original.png")
    case = store.create(source)
    path = store.image_path(source)
    path.unlink()
    with pytest.raises(ValueError, match="指紋"):
        store.restore_source(source, png("blue"))
    assert not path.exists()
    assert store.restore_source(source, original).read_bytes() == original
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="指紋"):
        store.restore_source(source, png("blue"))
    assert path.read_bytes() == b"corrupted"
    store.restore_source(source, original)
    assert store.load(case["id"]) == case
    assert store.image_path(source).read_bytes() == original


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("LV_OUTPUT_DIR", str(tmp_path / "out"))
    store = fp.CaseStore(tmp_path / "cases")
    source = store.ingest(png(), "screen.png", kind="screenshot")
    case = store.create(source, "反光報案", "使用者認為正常")
    return store, case


def test_intake_resume_duplicate_and_source_nonmutation(workspace, tmp_path):
    store, case = workspace
    original = tmp_path / "source.png"
    original.write_bytes(png("blue"))
    data = original.read_bytes()
    s = store.ingest(data, "original.png", source_path=original)
    fp.add_source(case, s)
    saved = store.save(case)
    assert fp.CaseStore(store.root).load(case["id"]) == saved
    assert original.read_bytes() == data
    assert len(saved["sources"]) == 2
    assert store.duplicates(s["sha256"])[0]["id"] == case["id"]
    original.unlink()
    assert store.image_path(s).read_bytes() == data  # case-owned copy survives
    assert "截圖" in fp.report_markdown(saved)
    assert saved["external"]["model_id"] == ""


def test_stale_writer_preserves_latest_and_history(workspace):
    store, case = workspace
    stale = copy.deepcopy(case)
    case["title"] = "new"
    saved = store.save(case)
    stale["title"] = "lost update"
    with pytest.raises(ValueError, match="另一個視窗"):
        store.save(stale)
    assert store.load(case["id"])["title"] == "new"
    with sqlite3.connect(store.db) as con:
        assert con.execute("SELECT COUNT(*) FROM history").fetchone()[0] == 2
    assert saved["revision"] == 2


def test_mixed_image_verdict_is_per_region_not_whole_image(workspace):
    store, case = workspace
    s = case["sources"][0]
    left = fp.add_item(case, s["id"], [0, 0, .4, 1], "crack", .99)
    right = fp.add_item(case, s["id"], [.5, 0, 1, 1], "crack", .95)
    assert left["review"]["verdict"] == "pending"
    fp.review_item(case, left["id"], "normal", "Engineer", "正常反光")
    fp.review_item(case, right["id"], "defect", "Engineer", "真裂痕")
    saved = store.save(case)
    assert [i["review"]["verdict"] for i in saved["items"]] == ["normal", "defect"]
    assert "verdict" not in saved["sources"][0]
    assert not list(store.root.rglob("*.txt"))
    report = fp.report_markdown(saved)
    assert "確認此區域正常" in report and "確認有瑕疵" in report


@pytest.mark.parametrize("roi", [[0, 0, 0, 1], [-.1, 0, 1, 1], [0, 0, float("nan"), 1], [0, 1]])
def test_invalid_geometry_rejected(workspace, roi):
    _, case = workspace
    with pytest.raises(ValueError):
        fp.add_item(case, case["sources"][0]["id"], roi)
    assert not case["items"]


def test_yolo_import_atomic_and_unknown_metadata(workspace):
    _, case = workspace
    sid = case["sources"][0]["id"]
    before = copy.deepcopy(case)
    with pytest.raises(ValueError):
        fp.import_yolo(case, sid, "0 .5 .5 .4 .4 .99\nmalformed")
    assert case == before
    assert fp.import_yolo(case, sid, "0 .5 .5 .4 .4 .99\n1 .15 .2 .1 .1 .8") == 2
    assert all(i["review"]["verdict"] == "pending" for i in case["items"])
    assert case["items"][0]["roi"] == [.3, .3, .7, .7]


def test_review_needs_human_and_reason(workspace):
    _, case = workspace
    item = fp.add_item(case, case["sources"][0]["id"], fp.FULL_ROI)
    with pytest.raises(ValueError, match="覆核者"):
        fp.review_item(case, item["id"], "normal", "", "looks good")
    assert item["review"]["verdict"] == "pending"


def test_claim_edit_citations_and_reply_not_cause(workspace):
    _, case = workspace
    ref = case["evidence"][0]["id"]
    with pytest.raises(ValueError, match="引用"):
        fp.add_claim(case, "fact", "a cause", ["invented"])
    fp.add_claim(case, "hypothesis", "可能為反光混淆，尚待驗證", [ref])
    claim_id = case["claims"][0]["id"]
    fp.add_claim(case, "hypothesis", "可能為外觀混淆", [ref], claim_id)
    assert len(case["claims"]) == 1
    case["reply_status"] = "replied"
    assert case["cause_status"] == "hypothesis"
    assert case["improvement_status"] == "not_evaluated"
    report = fp.report_markdown(case)
    assert ref in report and "外部模型尚未重播" in report


def make_candidates(store, case, n=4):
    for i in range(n):
        src = store.ingest(png((i+1, 30, 40)), f"{i}.png", kind="reference")
        case["candidates"].append({"id": fp.region_key(src["sha256"], fp.FULL_ROI), "source": src,
              "roi": fp.FULL_ROI.copy(), "review": fp._review("pending", "", ""), "reason": "相似待覆核",
              "reference_category": "normal", "evidence_id": case["evidence"][0]["id"]})


def test_review_zip_portable_transitive_provenance_and_return_compatible(workspace):
    from fp_case_page import _review_zip
    store,case=workspace
    make_candidates(store,case,1)
    candidate=case['candidates'][0]
    candidate['source']['source_path']='C:/private/source.png'
    candidate['evidence_id']='revision-e'
    case['external']['raw_output']='C:/private/log.json'
    case['evidence'].extend([
        {'id':'cam-e','role':'model_attribution','at':'then','details':{'attribution_id':'cam-1'}},
        {'id':'search-e','role':'lv_auxiliary','at':'then','details':{'search_id':'search-1','attribution_evidence_id':'cam-e','path':'C:/private/source.png'}},
        {'id':'revision-e','role':'region_revision','at':'now','details':{'before':{'evidence_id':'search-e'}}},
        {'id':'unrelated-e','role':'lv_auxiliary','at':'then','details':{'search_id':'other-search'}}])
    case['searches']=[{'id':'search-1','query_roi':[.1,.2,.6,.8],'recipe':{'path':'C:/private/references'},'attribution_evidence_id':'cam-e'},
                      {'id':'other-search','recipe':{}}]
    case['attributions']=[{'id':'cam-1','item_id':'historical-item','target_identity':'old',
                           'target':{'model_sha256':'model-hash','input_sha256':'image-hash'},'artifact_dir':'C:/private/cam'}]
    batch=fp.prepare_batch(case)
    manifest=fp.review_manifest(case,batch)
    provenance=fp.review_provenance(case,batch)
    package=_review_zip(store,manifest,provenance)
    candidate['review']['verdict']='normal'
    assert provenance['regions'][0]['review']['verdict']=='pending'
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        lineage=json.loads(archive.read('provenance.json'))
        returned=json.loads(archive.read('review_manifest.json'))
        all_json=archive.read('provenance.json')+archive.read('review_manifest.json')
        assert b'C:/private' not in all_json and b'unrelated-e' not in all_json
        assert {e['id'] for e in lineage['evidence']}=={'revision-e','search-e','cam-e'}
        assert [s['id'] for s in lineage['searches']]==['search-1']
        assert lineage['attributions'][0]['target_current'] is False
        assert lineage['regions'][0]['roi']==candidate['roi']
        assert lineage['regions'][0]['source_sha256']==candidate['source']['sha256']
        assert archive.read(returned['items'][0]['review_image'])==store.image_path(candidate['source']).read_bytes()
    returned['items'][0]['review']={'verdict':'normal','reviewer':'Engineer','reason':'entire region checked'}
    fp.apply_review_return(case,returned)
    assert case['candidates'][0]['review']['reviewer']=='Engineer'


def test_partial_group_manifest_and_ledger_units(workspace):
    import round_ledger as ledger
    store, case = workspace
    make_candidates(store, case, 100)
    for c in case["candidates"][:3]:
        fp.review_item(case, c["id"], "normal", "E", "依規格確認", candidate=True)
    bid = fp.prepare_batch(case)
    rid = fp.link_round(case, bid)
    assert fp.prepare_batch(case) == bid
    assert fp.link_round(case, bid) == rid
    manifest = fp.review_manifest(case, bid)
    assert manifest["counts"]["reviewed_regions"] == 3
    assert manifest["counts"]["pending_regions"] == 97
    assert manifest["counts"]["training_ready_images"] == 0
    rounds = ledger.load_rounds()
    assert len(rounds) == 1 and rounds[0]["n_labeled"] == 0
    assert len(rounds[0]["batches"]) == 1
    # Interrupted case-save recovery uses stable batch identity, not another round.
    case["batches"][0]["round_id"] = None
    assert fp.link_round(case, bid) == rid
    assert len(ledger.load_rounds()) == 1


def test_return_partial_validated_atomically_and_stale_rejected(workspace):
    store, case = workspace
    make_candidates(store, case)
    bid = fp.prepare_batch(case)
    fp.link_round(case, bid)
    m = fp.review_manifest(case, bid)
    m["items"][0]["review"] = {"verdict": "normal", "reviewer": "E", "reason": "確認"}
    m["items"][1]["source"]["sha256"] = "bad"
    before = copy.deepcopy(case)
    with pytest.raises(ValueError, match="身分"):
        fp.apply_review_return(case, m)
    assert case == before
    m["items"] = m["items"][:1]
    fp.apply_review_return(case, m)
    assert fp.review_counts(case["candidates"])["reviewed_regions"] == 1
    saved = store.save(case)
    with pytest.raises(ValueError, match="過期"):
        fp.apply_review_return(saved, m)
    m["case_id"] = "wrong-case"
    with pytest.raises(ValueError, match="不屬"):
        fp.apply_review_return(saved, m)


def test_candidate_dedup_and_changed_reference(workspace, tmp_path):
    store, case = workspace
    image = tmp_path / "ref.png"
    image.write_bytes(png("blue"))
    result = {"path": str(image), "sha256": fp.digest(image.read_bytes()), "roi": fp.FULL_ROI,
              "reason": "相似候選", "category": "normal", "evidence_id": case["evidence"][0]["id"]}
    assert fp.add_candidate(case, result, store)
    assert not fp.add_candidate(case, result, store)
    assert case["candidates"][0]["review"]["verdict"] == "pending"
    image.write_bytes(png("green"))
    with pytest.raises(ValueError, match="已變更"):
        fp.add_candidate(case, result, store)


def test_effort_logged_and_correctable(workspace):
    store, case = workspace
    fp.record_effort(case, 2.5, note="timer")
    fp.record_effort(case, 1.0)
    fp.record_effort(case, 2.0, mode="set", note="扣除離席")
    saved = store.save(case)
    assert saved["effort_minutes"] == 2 and len(saved["effort_log"]) == 3
    with pytest.raises(ValueError):
        fp.record_effort(case, float("nan"))
    assert json.loads(json.dumps(saved))["effort_minutes"] == 2
