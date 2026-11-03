import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import fp_cases as fp
from fp_case_search import search, record_search


def _image(path, color):
    Image.new("RGB", (48, 48), color).save(path)
    return path.read_bytes()


def _embed(image):
    # Test geometry/ranking/caching independently of the real model smoke test.
    return np.asarray(image).mean(axis=(0, 1)) + 1


def test_uploaded_dataset_filename_survives_search_candidate_and_handoffs(tmp_path):
    import json
    import zipfile
    from investigation_data import DatasetRegistry,prepare_import
    from fp_case_page import _review_zip
    from fp_training_export import export_training
    store=fp.CaseStore(tmp_path/'cases')
    query=_image(tmp_path/'query.png','red')
    reference=_image(tmp_path/'defect-ref.png','yellow')
    dataset=DatasetRegistry(store).save('uploaded',prepare_import({'defect-ref.png':reference}))
    case=store.create(store.ingest(query,'query.png'))
    item=fp.add_item(case,case['sources'][0]['id'],fp.FULL_ROI)
    result=search(case,item['id'],store,datasets={'defect':dataset['id']},include_history=False,embed_fn=_embed,recipe={'model':'test'})
    row=result['groups']['defect'][0]
    assert row['name']=='defect-ref.png'
    assert Path(row['path']).name!=row['name']
    record_search(case,result)
    fp.add_candidate(case,row,store)
    candidate=case['candidates'][0]
    assert candidate['source']['name']=='defect-ref.png'
    assert candidate['source']['sha256']==fp.digest(reference)
    assert store.image_path(candidate['source']).read_bytes()==reference
    fp.review_item(case,candidate['id'],'defect','Engineer','complete reference',candidate=True)
    batch=fp.prepare_batch(case)
    with zipfile.ZipFile(io.BytesIO(_review_zip(store,fp.review_manifest(case,batch),fp.review_provenance(case,batch)))) as archive:
        manifest=json.loads(archive.read('review_manifest.json'))
        assert manifest['items'][0]['source']['name']=='defect-ref.png'
    record,data=export_training(case,store,complete_images=[candidate['source']['sha256']],
        class_assignments={'candidate:'+candidate['id']:'crack'},class_names=['crack'],reviewer='Engineer',reason='complete')
    assert record['items'][0]['source']['name']=='defect-ref.png'


def test_balanced_retrieval_conflicts_dedup_provenance_and_cache(tmp_path):
    normal, defect = tmp_path / "normal", tmp_path / "defect"
    normal.mkdir(); defect.mkdir()
    data = _image(tmp_path / "query.png", (220, 10, 10))
    _image(normal / "normal.png", (210, 15, 10))
    _image(defect / "defect.png", (10, 10, 230))
    (normal / "copy.png").write_bytes((normal / "normal.png").read_bytes())
    (defect / "conflict.png").write_bytes((normal / "normal.png").read_bytes())
    (normal / "broken.png").write_bytes(b"bad image")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.png")}
    store = fp.CaseStore(tmp_path / "cases")
    case = store.create(store.ingest(data, "q.png"))
    item = fp.add_item(case, case["sources"][0]["id"], fp.FULL_ROI)
    recipe = {"model": "test-mean", "checkpoint_sha256": "test-v1"}
    calls = []
    def embed(image):
        calls.append(image.size)
        return _embed(image)
    result = search(case, item["id"], store, str(normal), str(defect), embed_fn=embed, recipe=recipe)
    assert len(result["groups"]["normal"]) == 1
    assert len(result["groups"]["defect"]) == 2
    assert not result["groups"]["history"]
    assert result["groups"]["normal"][0]["label_conflict"]
    assert len(result["skipped"]) == 1
    count = len(calls)
    search(case, item["id"], store, str(normal), str(defect), embed_fn=embed, recipe=recipe)
    assert len(calls) == count  # content/ROI/recipe cache, not paths alone
    record_search(case, result)
    assert result["groups"]["normal"][0]["evidence_id"] in {e["id"] for e in case["evidence"]}
    assert case["evidence"][-1]["role"] == "lv_auxiliary"
    assert all(p.read_bytes() == contents for p, contents in before.items())


def test_history_retains_external_version_and_region_review(tmp_path):
    store = fp.CaseStore(tmp_path / "cases")
    data = _image(tmp_path / "old.png", (210, 30, 40))
    old = store.create(store.ingest(data, "old.png"), model_id="YOLO-old")
    old_item = fp.add_item(old, old["sources"][0]["id"], [0, 0, .5, .5])
    fp.review_item(old, old_item["id"], "normal", "E", "反光")
    store.save(old)
    data = _image(tmp_path / "new.png", (220, 30, 40))
    case = store.create(store.ingest(data, "new.png"), model_id="YOLO-new")
    item = fp.add_item(case, case["sources"][0]["id"], [0, 0, .5, .5])
    result = search(case, item["id"], store, embed_fn=_embed, recipe={"model": "test"})
    row = result["groups"]["history"][0]
    assert row["external_model"] == "YOLO-old"
    assert row["roi"] == [0, 0, .5, .5]
    assert row["label_status"] == "reviewed_normal"
    assert not case["claims"] and case["cause_status"] == "unknown"


def test_limits_and_empty_results_are_explicit(tmp_path):
    folder = tmp_path / "normal"
    folder.mkdir()
    for i in range(4):
        _image(folder / f"{i}.png", (i*20+1, 20, 30))
    data = _image(tmp_path / "q.png", (255, 0, 0))
    store = fp.CaseStore(tmp_path / "cases")
    case = store.create(store.ingest(data, "q.png"))
    item = fp.add_item(case, case["sources"][0]["id"], [0, 0, .5, .5])
    result = search(case, item["id"], store, str(folder), limit=1, candidate_scope="same_roi",
                    embed_fn=_embed, recipe={"model": "test"})
    assert result["sources"]["normal"]["truncated"]
    assert result["groups"]["defect"] == []
    assert result["groups"]["normal"][0]["roi"] == item["roi"]
    with pytest.raises(ValueError, match="資料夾"):
        search(case, item["id"], store, "missing-folder", embed_fn=_embed, recipe={"model": "test"})
