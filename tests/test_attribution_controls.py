import copy

import numpy as np
import pytest

from attribution_controls import assess_faithfulness, iter_variants


def fixture():
    pixels = np.arange(30*40*3, dtype=np.uint16).reshape(30,40,3).astype(np.uint8)
    scores = np.zeros((30,40)); scores[10:15,15:20] = 1
    return pixels,scores


def test_controls_preserve_geometry_area_and_outside_pixels_across_fills():
    pixels,scores=fixture()
    variants=list(iter_variants(pixels,scores,fractions=(25/1200,),controls=3))
    repeat=list(iter_variants(pixels,scores,fractions=(25/1200,),controls=3))
    assert len(variants)==8
    for row,again in zip(variants,repeat):
        assert not row.get("unavailable")
        assert row["mask_sha256"]==again["mask_sha256"]
        np.testing.assert_array_equal(row["pixels"],again["pixels"])
        dy,dx=row["translation"]
        mask=np.zeros((30,40),bool);mask[10+dy:15+dy,15+dx:20+dx]=True
        assert row["masked_pixels"]==mask.sum()==25
        np.testing.assert_array_equal(row["pixels"][~mask],pixels[~mask])
        if row["control"]:assert row["overlap"]<=.1
    assert [r["mask_sha256"] for r in variants[:4]]==[r["mask_sha256"] for r in variants[4:]]
    assert len({r["mask_sha256"] for r in variants[:4]})==4


def test_full_extent_mask_cannot_fall_back_to_scattered_pixels():
    pixels=np.zeros((10,10,3),np.uint8);scores=np.zeros((10,10));scores[::9,::9]=1
    rows=list(iter_variants(pixels,scores,fractions=(.04,),controls=2))
    assert len(rows)==4
    assert sum(bool(r.get("unavailable")) for r in rows)==2
    assert all(r.get("available_controls")==0 for r in rows if r.get("unavailable"))


def measured():
    p,s=fixture();rows=list(iter_variants(p,s,fractions=(25/1200,),controls=2))
    for r in rows:r.pop("pixels");r["fixed_score_drop"]=.5 if r["role"]=="attribution" else .1
    return {"status":"measured","protocol_version":2,"controls":2,"rows":rows,"fractions":[25/1200],"fill_methods":["constant","blur"]}


def test_assessment_rejects_missing_controls_and_detects_fill_reversal():
    report=measured()
    assert assess_faithfulness(report)["status"]=="supports"
    other=copy.deepcopy(report)
    for row in other["rows"]:
        if row["fill_method"]=="blur" and row["role"]=="attribution":row["fixed_score_drop"]=-.2
    assert assess_faithfulness(other)["status"]=="unstable"
    report["rows"].pop()
    assert assess_faithfulness(report)["status"]=="inconclusive"
    assert assess_faithfulness({"status":"measured","rows":[]})["status"]=="inconclusive"


def test_missing_entire_requested_fraction_is_inconclusive():
    report=measured();report["fractions"].append(.5)
    assert assess_faithfulness(report)["status"]=="inconclusive"


@pytest.mark.parametrize("kwargs",[{"fill_methods":("noise",)},{"max_overlap":.5},{"controls":True},{"fractions":(.1,.1)}])
def test_invalid_protocol_is_rejected(kwargs):
    with pytest.raises(ValueError):list(iter_variants(*fixture(),**kwargs))
