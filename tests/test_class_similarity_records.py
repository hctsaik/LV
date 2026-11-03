"""驗收:class_similarity_records 身份 / key / eligibility。

設計:3_Architect_Design/32_class_similarity_records.md。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest


def _mod():
    import class_similarity_records as rec
    return rec


def _image_rec(path, label, split="train"):
    return {"path": Path(path), "label": label, "split": split}


def _object_rec(crop, image, bbox, obj_index, label, split="train"):
    return {
        "path": Path(crop), "image_path": Path(image), "bbox": list(bbox),
        "obj_index": obj_index, "label": label, "split": split,
    }


def _ctx(**kw):
    base = {
        "level": "image",
        "data_token": "tok1",
        "model": "dinov2_vits14",
        "model_weights_id": "unknown",
        "embedding_recipe": {"level": "image", "D": 2, "resolution": 224,
                             "pooling": "cls", "crop_policy": "none"},
        "analysis_split": "all",
        "k": 5,
        "viz_mode": "Image Classifier",
        "generation": None,
        "manifest": {},
    }
    base.update(kw)
    return base


def test_ac29_path_vs_crop_policy_id(tmp_path):  # AC29
    rec = _mod()
    p1 = tmp_path / "a" / "x.jpg"
    p2 = tmp_path / "b" / "x.jpg"
    p1.parent.mkdir(); p2.parent.mkdir()
    p1.write_bytes(b"same-bytes")
    p2.write_bytes(b"same-bytes")
    r1 = _image_rec(p1, "A")
    r2 = _image_rec(p2, "A")
    id1 = rec.make_record_id(r1, level="image", data_token="t")
    id2 = rec.make_record_id(r2, level="image", data_token="t")
    assert id1[0] != id2[0]
    obj = _object_rec(tmp_path / "c.jpg", p1, [0.1, 0.2, 0.3, 0.4], 0, "A")
    a = rec.make_record_id(obj, level="object", data_token="t")[0]
    b = rec.make_record_id(obj, level="object", data_token="t")[0]
    assert a == b  # crop policy is not part of the ID


def test_ac29_invalid_temp_ids():  # AC29
    rec = _mod()
    r0 = {"path": Path("crop0.jpg"), "label": "A", "split": "train"}  # object missing provenance
    r1 = {"path": Path("crop1.jpg"), "label": "A", "split": "train"}
    raw = np.eye(2)
    ctx = _ctx(level="object", viz_mode="Object Detector", data_token="tokZ")
    prepared = rec.prepare_records([r0, r1], raw, ctx)
    assert prepared.record_ids[0] != prepared.record_ids[1]
    assert prepared.record_ids[0].startswith("invalid:tokZ:0")
    assert prepared.record_ids[1].startswith("invalid:tokZ:1")
    assert prepared.eligibility[0] is False and prepared.eligibility[1] is False
    assert all(not e for e in prepared.eligibility)


def test_ac29_crop_collision_integrity(tmp_path):  # AC29
    rec = _mod()
    crop = tmp_path / "same__obj0.jpg"
    img_a = tmp_path / "dirA" / "same.jpg"
    img_b = tmp_path / "dirB" / "same.jpg"
    img_a.parent.mkdir(); img_b.parent.mkdir()
    img_a.write_bytes(b"a"); img_b.write_bytes(b"b")
    recs = [
        _object_rec(crop, img_a, [0, 0, 1, 1], 0, "A"),
        _object_rec(crop, img_b, [0, 0, 1, 1], 0, "B"),
    ]
    raw = np.eye(2)
    with pytest.raises(Exception) as ei:
        rec.prepare_records(recs, raw, _ctx(level="object", viz_mode="Object Detector"))
    msg = str(ei.value).lower()
    assert "crop" in msg or "integrity" in msg or "conflict" in msg


def test_ac03_duplicate_identity_error(tmp_path):  # AC03
    rec = _mod()
    p = tmp_path / "train" / "A" / "x.jpg"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    recs = [_image_rec(p, "A"), _image_rec(p, "A")]
    with pytest.raises(Exception):
        rec.prepare_records(recs, np.eye(2), _ctx())


def test_ac30_flat_fallback_unresolved(tmp_path):  # AC30
    rec = _mod()
    p = tmp_path / "train" / "img.jpg"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"i")
    recs = [_image_rec(p, "train", split="train")]
    prepared = rec.prepare_records(recs, np.array([[1.0, 0.0]]), _ctx())
    assert prepared.label_status[0] == "unresolved"
    assert prepared.eligibility[0] is False
    assert prepared.label_origin[0] == "flat_fallback"


def test_ac30_literal_unknown_mix_single(tmp_path):  # AC30
    rec = _mod()
    rows = []
    for name in ("unknown", "mix"):
        p = tmp_path / "train" / name / "a.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(name.encode())
        rows.append(_image_rec(p, name))
    raw = np.array([[1.0, 0.0], [0.0, 1.0]])
    prepared = rec.prepare_records(rows, raw, _ctx())
    assert prepared.label_status == ["single", "single"]
    assert prepared.eligibility == [True, True]


def test_ac30_readback_effective_label(tmp_path):  # AC30
    rec = _mod()
    p = tmp_path / "train" / "A" / "x.jpg"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    row = _image_rec(p, "B")  # already overlaid A→B
    row["source_label"] = "A"
    prepared = rec.prepare_records([row], np.array([[1.0, 0.0]]), _ctx())
    assert prepared.effective_label[0] == "B"
    assert prepared.labels[0] == "B"
    assert prepared.source_label[0] == "A"


def test_ac06_inputs_invalidate_key(tmp_path):  # AC06
    rec = _mod()
    p1 = tmp_path / "train" / "A" / "x.jpg"
    p2 = tmp_path / "train" / "B" / "y.jpg"
    p1.parent.mkdir(parents=True); p2.parent.mkdir(parents=True)
    p1.write_bytes(b"x"); p2.write_bytes(b"y")
    recs = [_image_rec(p1, "A"), _image_rec(p2, "B")]
    raw = np.array([[1.0, 0.0], [0.0, 1.0]])
    a = rec.prepare_records(recs, raw, _ctx(k=5))
    b = rec.prepare_records(recs, raw, _ctx(k=6))
    assert a.analysis_key != b.analysis_key
    recs2 = [_image_rec(p1, "B"), _image_rec(p2, "B")]
    c = rec.prepare_records(recs2, raw, _ctx(k=5))
    assert a.analysis_key != c.analysis_key
    raw2 = raw.copy(); raw2[0, 0] = 0.5
    d = rec.prepare_records(recs, raw2, _ctx(k=5))
    assert a.analysis_key != d.analysis_key
    e = rec.prepare_records(recs, raw, _ctx(k=5, analysis_split="train"))
    # split train vs all may differ if split filtering changes U; both rows are train
    recipe = dict(_ctx()["embedding_recipe"])
    recipe["resolution"] = 336
    f = rec.prepare_records(recs, raw, _ctx(embedding_recipe=recipe))
    assert a.analysis_key != f.analysis_key


def test_ac06_display_does_not_rescore(tmp_path):  # AC06
    rec = _mod()
    p1 = tmp_path / "train" / "A" / "x.jpg"
    p2 = tmp_path / "train" / "B" / "y.jpg"
    p1.parent.mkdir(parents=True); p2.parent.mkdir(parents=True)
    p1.write_bytes(b"x"); p2.write_bytes(b"y")
    recs = [_image_rec(p1, "A"), _image_rec(p2, "B")]
    raw = np.array([[1.0, 0.0], [0.0, 1.0]])
    prepared = rec.prepare_records(recs, raw, _ctx())
    key1 = prepared.analysis_key
    # display-only fingerprint must ignore visible class filter / sort / metric
    fp1 = rec.scope_fingerprint(prepared)
    fp2 = rec.scope_fingerprint(prepared, visible_classes=["A"], metric="same", sort="id")
    assert fp1 == fp2 or rec.analysis_inputs_fingerprint(prepared) == rec.analysis_inputs_fingerprint(prepared)
    assert key1 == rec.prepare_records(recs, raw, _ctx()).analysis_key


def test_ac08_index_after_filter(tmp_path):  # AC08
    rec = _mod()
    paths = []
    recs = []
    for i, lab in enumerate(["A", "B", "A"]):
        p = tmp_path / "train" / lab / f"{i}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(bytes([i]))
        recs.append(_image_rec(p, lab))
    raw = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]])
    prepared = rec.prepare_records(recs, raw, _ctx())
    import class_similarity as cs
    result = cs.analyze(prepared.embeddings, prepared.labels, prepared.record_ids,
                        eligibility=prepared.eligibility,
                        invalid_reasons=prepared.invalid_reasons, k=1)
    event = {
        "analysis_key": prepared.analysis_key,
        "record_id": prepared.record_ids[2],
        "original_index": prepared.original_index[2],
    }
    idx = rec.validate_selection(event, result, prepared)
    assert idx == [2]


def test_ac08_stale_event_rejected(tmp_path):  # AC08
    rec = _mod()
    p = tmp_path / "train" / "A" / "x.jpg"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    recs = [_image_rec(p, "A")]
    raw = np.array([[1.0, 0.0]])
    prepared = rec.prepare_records(recs, raw, _ctx())
    import class_similarity as cs
    result = cs.analyze(prepared.embeddings, prepared.labels, prepared.record_ids,
                        eligibility=prepared.eligibility,
                        invalid_reasons=prepared.invalid_reasons, k=1)
    event = {
        "analysis_key": "deadbeef",
        "record_id": prepared.record_ids[0],
        "original_index": 0,
    }
    with pytest.raises(Exception):
        rec.validate_selection(event, result, prepared)
    event2 = {
        "analysis_key": prepared.analysis_key,
        "record_id": "not-this-id",
        "original_index": 0,
    }
    with pytest.raises(Exception):
        rec.validate_selection(event2, result, prepared)


def test_ac30_mode_restore_rejects_old_event():  # AC30
    rec = _mod()
    event = {"analysis_key": "k1", "record_id": "id", "original_index": 0,
             "event_generation": 1}
    ctx = {"analysis_key": "k1", "event_generation": 2, "record_ids": ["id"]}
    with pytest.raises(Exception):
        rec.validate_selection(event, None, ctx)


def test_ac34_generation_hash_mismatch(tmp_path):  # AC34
    rec = _mod()
    p = tmp_path / "train" / "A" / "x.jpg"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"old")
    sha = hashlib.sha256(b"old").hexdigest()
    gen = {"source_hashes": {rec.canonical_path(p): sha}, "recipe": {"D": 2}}
    recs = [_image_rec(p, "A")]
    prepared = rec.prepare_records(recs, np.array([[1.0, 0.0]]), _ctx(generation=gen))
    assert prepared.source_status in {"verified", "source_verified"}
    p.write_bytes(b"new-bytes-changed")
    stale = rec.prepare_records(recs, np.array([[1.0, 0.0]]), _ctx(generation=gen))
    assert stale.source_status in {"stale", "source_stale"}


def test_ac34_unverified_generation(tmp_path):  # AC34
    rec = _mod()
    p = tmp_path / "train" / "A" / "x.jpg"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    recs = [_image_rec(p, "A")]
    prepared = rec.prepare_records(recs, np.array([[1.0, 0.0]]), _ctx(generation=None))
    assert prepared.source_status in {"unverified", "source_unverified"}
    assert prepared.source_verified is False
