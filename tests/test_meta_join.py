"""驗收:23 meta_join(metadata CSV ↔ 影像對照 + 分組索引)。

設計:3_Architect_Design/23_meta_join.md。CSV 為唯一 I/O(tmp 檔),其餘純合成。
"""
from __future__ import annotations

import pytest


def _write_csv(path, text, encoding="utf-8"):
    path.write_text(text, encoding=encoding)
    return path


PATHS = [r"C:\data\a\x.png", r"C:\data\b\x.png", r"C:\data\c\y.png", r"C:\data\d\z.png"]


def test_ac1_filename_match_handcalc():
    """# AC1:檔名匹配——同 basename 2 張全套用;n_matched=3、未匹配影像=1、bogus 列=1。"""
    from meta_join import join_metadata
    rows = [{"filename": "x.png", "tool": "T1"},
            {"filename": "y.png", "tool": "T2"},
            {"filename": "nope.png", "tool": "T3"}]
    r = join_metadata(rows, PATHS)
    assert r["per_image"][0]["tool"] == "T1"
    assert r["per_image"][1]["tool"] == "T1"
    assert r["per_image"][2]["tool"] == "T2"
    assert 3 not in r["per_image"]
    assert r["n_matched_images"] == 3
    assert r["n_unmatched_images"] == 1
    assert r["unmatched_row_idx"] == [2]
    assert r["fields"] == ["tool"]
    assert r["n_rows"] == 3


def test_ac2_sha_takes_priority():
    """# AC2:sha 正確但檔名亂寫 → 仍配上;sha 與檔名指向不同影像 → sha 勝。"""
    from meta_join import join_metadata
    paths = [r"C:\d\q.png", r"C:\d\r.png"]
    sha_of = {0: "aaa", 1: "bbb"}
    rows = [{"sha256": "aaa", "filename": "r.png", "tool": "T9"}]
    r = join_metadata(rows, paths, sha_of)
    assert r["per_image"].get(0, {}).get("tool") == "T9"
    assert 1 not in r["per_image"]


def test_ac3_unmatched_both_directions():
    """# AC3:5 列 bogus → unmatched_row_idx 長 5;2 張影像無列 → n_unmatched_images=2。"""
    from meta_join import join_metadata
    paths = [r"C:\d\a.png", r"C:\d\b.png", r"C:\d\c.png"]
    rows = [{"filename": "a.png", "tool": "T1"}] + \
           [{"filename": f"bogus{i}.png", "tool": "T1"} for i in range(5)]
    r = join_metadata(rows, paths)
    assert len(r["unmatched_row_idx"]) == 5
    assert r["n_unmatched_images"] == 2
    assert r["n_matched_images"] == 1


def test_ac4_column_detection():
    """# AC4:欄名不分大小寫;sha 欄 + sha_of 可用;無鍵欄 ValueError 列出欄位。"""
    from meta_join import join_metadata
    r = join_metadata([{"FileName": "x.png", "Tool": "T1"}], PATHS[:1])
    assert r["per_image"][0]["Tool"] == "T1"
    r2 = join_metadata([{"SHA256": "s0", "Tool": "T2"}], PATHS[:1], {0: "s0"})
    assert r2["per_image"][0]["Tool"] == "T2"
    with pytest.raises(ValueError) as ei:
        join_metadata([{"foo": "1", "bar": "2"}], PATHS[:1])
    assert "foo" in str(ei.value)


def test_ac5_group_counts_cap_and_missing():
    """# AC5:max_groups 截斷 → 前 max_groups-1 組 + "(其他)";missing 含無值影像。"""
    from meta_join import group_counts
    per_image = {}
    idx = 0
    sizes = {"va": 4, "vb": 3, "vc": 2, "vd": 1, "ve": 1}
    for val, n in sizes.items():
        for _ in range(n):
            per_image[idx] = {"tool": val}
            idx += 1
    n_images = idx + 1                       # 最後一張無 tool 值
    g = group_counts(per_image, n_images, "tool", max_groups=3)
    assert list(g["groups"].keys()) == ["va", "vb", "(其他)"]
    assert g["groups"]["va"] == [0, 1, 2, 3]
    assert g["groups"]["vb"] == [4, 5, 6]
    assert sorted(g["groups"]["(其他)"]) == [7, 8, 9, 10]
    assert g["missing"] == [idx]


def test_ac6_bom_and_empty(tmp_path):
    """# AC6:utf-8-sig(BOM)可讀;空檔 ValueError。"""
    from meta_join import load_metadata_csv
    f = _write_csv(tmp_path / "m.csv", "filename,tool\nx.png,T1\n", encoding="utf-8-sig")
    rows = load_metadata_csv(f)
    assert rows == [{"filename": "x.png", "tool": "T1"}]
    empty = _write_csv(tmp_path / "e.csv", "")
    with pytest.raises(ValueError):
        load_metadata_csv(empty)


def test_ac7_later_row_wins_with_warning():
    """# AC7:多列配同圖 → 後列覆蓋,warnings 非空。"""
    from meta_join import join_metadata
    rows = [{"filename": "x.png", "tool": "T1"},
            {"filename": "x.png", "tool": "T2"}]
    r = join_metadata(rows, PATHS[:1])
    assert r["per_image"][0]["tool"] == "T2"
    assert r["warnings"]
