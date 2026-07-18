"""驗收:24 audit_report(訊號原始結果 → 量化報告資料結構)。

設計:3_Architect_Design/24_audit_report.md。全合成手算、無 I/O、免模型。
"""
from __future__ import annotations

import numpy as np
import pytest

ORDER = ["overview", "unreadable", "exact_dup", "near_dup", "emb_near_dup",
         "leakage", "label_conflict", "class_coverage", "outliers"]


def _inputs(n=4, **over):
    d = {
        "paths": [f"p{i}.png" for i in range(n)],
        "splits": ["train"] * n,
        "labels": [""] * n,
        "shas": [f"s{i}" for i in range(n)],
        "n_unreadable": 0,
        "phash_pairs": [],
        "leak_pairs": [],
        "emb_pairs": None,
        "outlier_scores": None,
        "meta": None,
        "meta_groups": None,
        "params": {"model": "m", "phash_max_hamming": 4, "emb_max_distance": 0.05,
                   "outlier_k": 5, "top_n": 10, "max_pairs": 200,
                   "use_embedding": True},
    }
    d.update(over)
    return d


def _sec(report, key):
    for s in report["sections"]:
        if s["key"] == key:
            return s
    raise AssertionError(f"缺節:{key}")


def test_ac1_exact_dup_handcalc():
    """# AC1:shas [a,a,a,b,c,c] → 額外複本 3(=2+1)、2 組、組員正確。"""
    from audit_report import build_report
    r = build_report(_inputs(6, shas=["a", "a", "a", "b", "c", "c"]))
    s = _sec(r, "exact_dup")
    assert s["value"] == 3
    assert len(s["items"]) == 2
    grp = {g["sha"]: sorted(g["paths"]) for g in s["items"]}
    assert grp["a"] == ["p0.png", "p1.png", "p2.png"]
    assert grp["c"] == ["p4.png", "p5.png"]


def test_ac2_near_dup_excludes_exact():
    """# AC2:phash 對含同 sha 的一對 → 只算另一對。"""
    from audit_report import build_report
    r = build_report(_inputs(5, shas=["dup", "dup", "x", "y", "z"],
                             phash_pairs=[(0, 1, 0), (3, 4, 2)]))
    s = _sec(r, "near_dup")
    assert s["value"] == 1
    assert s["items"][0]["path_i"] == "p3.png" and s["items"][0]["path_j"] == "p4.png"


def test_ac3_leakage_pairs():
    """# AC3:2 對跨 split → value 2,items 帶 (split_i, split_j)。"""
    from audit_report import build_report
    r = build_report(_inputs(7, splits=["train"] * 5 + ["val"] * 2,
                             leak_pairs=[(0, 5, 0), (1, 6, 1)]))
    s = _sec(r, "leakage")
    assert s["value"] == 2
    assert s["items"][0]["split_i"] == "train" and s["items"][0]["split_j"] == "val"


def test_ac4_label_conflict():
    """# AC4:同 sha 異標 1 組;同 sha 同標 0;空標籤不觸發。"""
    from audit_report import build_report
    r = build_report(_inputs(4, shas=["x", "x", "y", "y"],
                             labels=["A", "B", "C", "C"]))
    s = _sec(r, "label_conflict")
    assert s["value"] == 1
    assert sorted(s["items"][0]["labels"]) == ["A", "B"]
    r2 = build_report(_inputs(2, shas=["z", "z"], labels=["A", ""]))
    assert _sec(r2, "label_conflict")["value"] == 0


def test_ac5_class_coverage():
    """# AC5:train{A,B,C} val{A} → 基準 train、缺類 2、val missing 排序;單 split 不適用。"""
    from audit_report import build_report
    splits = ["train"] * 6 + ["val"] * 2
    labels = ["classA", "classA", "classA", "classB", "classB", "classC",
              "classA", "classA"]
    r = build_report(_inputs(8, splits=splits, labels=labels))
    s = _sec(r, "class_coverage")
    assert s["value"] == 2
    val_item = next(it for it in s["items"] if it["split"] == "val")
    assert val_item["missing"] == ["classB", "classC"]
    r2 = build_report(_inputs(3, labels=["A", "B", "A"]))
    s2 = _sec(r2, "class_coverage")
    assert s2["value"] is None and "不適用" in s2["text"]


def test_ac6_outliers_topn_and_skipped():
    """# AC6:分數降冪取 top_n;None → 未計算。"""
    from audit_report import build_report
    inp = _inputs(4, outlier_scores=np.array([.1, .9, .5, .7]))
    inp["params"]["top_n"] = 2
    r = build_report(inp)
    s = _sec(r, "outliers")
    assert s["value"] == 2
    assert [it["path"] for it in s["items"]] == ["p1.png", "p3.png"]
    r2 = build_report(_inputs(4, outlier_scores=None))
    s2 = _sec(r2, "outliers")
    assert s2["value"] is None and "未計算" in s2["text"]


def test_ac7_metadata_section():
    """# AC7:meta 給定 → 末節 metadata,數字正確;meta=None → 無此節。"""
    from audit_report import build_report
    meta = {"fields": ["tool"], "per_image": {0: {"tool": "T1"}}, "n_rows": 3,
            "n_matched_images": 1, "n_unmatched_images": 2,
            "unmatched_row_idx": [1, 2], "warnings": []}
    groups = {"tool": {"groups": {"T1": [0]}, "missing": [1, 2]}}
    r = build_report(_inputs(3, meta=meta, meta_groups=groups))
    assert r["sections"][-1]["key"] == "metadata"
    s = r["sections"][-1]
    assert s["value"] == 1
    assert "2" in s["text"]
    tool_item = next(it for it in s["items"] if it["field"] == "tool")
    assert tool_item["groups"] == {"T1": 1}
    assert tool_item["n_missing"] == 2
    r2 = build_report(_inputs(3))
    assert all(sec["key"] != "metadata" for sec in r2["sections"])


def test_ac8_structure_contract():
    """# AC8:sections 順序固定、每節 method 非空、overview value=n_images。"""
    from audit_report import build_report
    r = build_report(_inputs(5))
    assert [s["key"] for s in r["sections"]] == ORDER
    assert all(isinstance(s["method"], str) and s["method"] for s in r["sections"])
    assert _sec(r, "overview")["value"] == 5
    assert r["totals"]["n_images"] == 5
    assert r["totals"]["split_counts"] == {"train": 5}


def test_ac9_text_numbers_and_cap_honesty():
    """# AC9:白話句含數字;pairs 達 max_pairs → text 註明顯示上限。"""
    from audit_report import build_report
    r = build_report(_inputs(6, shas=["a", "a", "a", "b", "c", "c"]))
    assert "3" in _sec(r, "exact_dup")["text"]
    inp = _inputs(6, shas=[f"u{i}" for i in range(6)],
                  phash_pairs=[(0, 1, 1), (2, 3, 2)])
    inp["params"]["max_pairs"] = 2
    r2 = build_report(inp)
    assert "顯示上限" in _sec(r2, "near_dup")["text"]


def test_ac10_length_mismatch_raises():
    """# AC10:shas 與 paths 長度不一 → ValueError。"""
    from audit_report import build_report
    with pytest.raises(ValueError):
        build_report(_inputs(4, shas=["a", "b"]))
