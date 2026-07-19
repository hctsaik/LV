"""驗收:29 wafer_grid(欄名猜測 + die 格點整形 + PIL 點陣渲染)。

設計:3_Architect_Design/29_wafer_grid.md。全合成手算、無 I/O、免模型。
G6 鐵則:零欄位假設——缺 wafer 欄且未明示「同屬一片」必須 raise,不得默默猜。
"""
from __future__ import annotations

import pytest


def test_ac1_guess_fields_exact():
    """# AC1:欄名 casefold 完全相等才中;同表多欄取先出現;沒中=None。"""
    from wafer_grid import guess_fields
    assert guess_fields(["LOT", "Wafer_ID", "Die_X", "Die_Y", "tool"]) == {
        "wafer": "Wafer_ID", "x": "Die_X", "y": "Die_Y", "lot": "LOT"}
    assert guess_fields(["a", "b"]) == {
        "wafer": None, "x": None, "y": None, "lot": None}
    assert guess_fields(["x", "col"])["x"] == "x"


def test_ac2_build_exact():
    """# AC2:物件聚合(flag/score 取 max)、缺座標與查無列分開列名。"""
    from wafer_grid import build_wafer_grids
    items = [
        {"item_id": "p0#0", "path": "p0", "flag": 0, "score": 0.2},
        {"item_id": "p0#1", "path": "p0", "flag": 1, "score": 0.9},
        {"item_id": "p1#0", "path": "p1", "flag": 0, "score": 0.1},
        {"item_id": "p2#0", "path": "p2", "flag": 0, "score": 0.3},
    ]
    meta = {"p0": {"w": "W1", "cx": "0", "cy": "0"},
            "p1": {"w": "W1", "cx": "1", "cy": "bad"}}
    r = build_wafer_grids(items, meta, wafer_field="w", x_field="cx", y_field="cy")
    assert r["wafers"] == [{
        "wafer_id": "W1", "lot": None,
        "dies": [{"x": 0.0, "y": 0.0, "flag": 1, "path": "p0", "score": 0.9}],
        "n": 1, "k": 1}]
    assert r["missing_coord"] == ["p1"]
    assert r["unmatched"] == ["p2"]


def test_ac3_g6_no_guessing():
    """# AC3(G6):無 wafer 欄且未明示 → ValueError(含「同屬一片」);
    明示後 wafer_id 固定 "(單片)"。"""
    from wafer_grid import build_wafer_grids
    items = [{"item_id": "p0#0", "path": "p0", "flag": 1, "score": None}]
    meta = {"p0": {"cx": "0", "cy": "0"}}
    with pytest.raises(ValueError, match="同屬一片"):
        build_wafer_grids(items, meta, wafer_field=None, x_field="cx", y_field="cy")
    r = build_wafer_grids(items, meta, wafer_field=None, x_field="cx", y_field="cy",
                          assume_single_wafer=True)
    assert r["wafers"][0]["wafer_id"] == "(單片)"


def test_ac4_grouping_sorted_and_lot():
    """# AC4:交錯輸入 → wafer_id 字典序;n/k 手算;lot 帶出。"""
    from wafer_grid import build_wafer_grids
    items = [{"item_id": f"p{i}#0", "path": f"p{i}", "flag": i % 2, "score": None}
             for i in range(4)]
    meta = {"p0": {"w": "W2", "cx": "0", "cy": "0", "lot": "L1"},
            "p1": {"w": "W1", "cx": "1", "cy": "0", "lot": "L1"},
            "p2": {"w": "W2", "cx": "2", "cy": "0", "lot": "L1"},
            "p3": {"w": "W1", "cx": "3", "cy": "0", "lot": "L1"}}
    r = build_wafer_grids(items, meta, wafer_field="w", x_field="cx", y_field="cy",
                          lot_field="lot")
    assert [w["wafer_id"] for w in r["wafers"]] == ["W1", "W2"]
    w1, w2 = r["wafers"]
    assert (w1["n"], w1["k"]) == (2, 2)   # p1(flag1)+p3(flag1)
    assert (w2["n"], w2["k"]) == (2, 0)   # p0(flag0)+p2(flag0)
    assert w1["lot"] == "L1" and w2["lot"] == "L1"


def test_ac5_png_pixels_exact():
    """# AC5(G3 像素):3 顆 die、cell=10、pad=0 → 尺寸與格中心顏色 exact。"""
    from wafer_grid import wafer_map_png
    wafer = {"wafer_id": "W1", "lot": None, "n": 3, "k": 1,
             "dies": [{"x": 0.0, "y": 0.0, "flag": 1, "path": "a", "score": None},
                      {"x": 1.0, "y": 0.0, "flag": 0, "path": "b", "score": None},
                      {"x": 0.0, "y": 1.0, "flag": 0, "path": "c", "score": None}]}
    img = wafer_map_png(wafer, cell=10, pad=0)
    assert img.size == (20, 20)
    px = img.convert("RGB")
    assert px.getpixel((5, 5)) == (220, 60, 60)      # flag=1
    assert px.getpixel((15, 5)) == (120, 160, 220)   # flag=0
    assert px.getpixel((15, 15)) == (245, 245, 245)  # 空格=背景


def test_ac6_missing_wafer_value():
    """# AC6:wafer 欄值空字串 → 進 missing_wafer、不進任何 wafer。"""
    from wafer_grid import build_wafer_grids
    items = [{"item_id": "p0#0", "path": "p0", "flag": 0, "score": None},
             {"item_id": "p1#0", "path": "p1", "flag": 0, "score": None}]
    meta = {"p0": {"w": "", "cx": "0", "cy": "0"},
            "p1": {"w": "W1", "cx": "1", "cy": "1"}}
    r = build_wafer_grids(items, meta, wafer_field="w", x_field="cx", y_field="cy")
    assert r["missing_wafer"] == ["p0"]
    assert [w["wafer_id"] for w in r["wafers"]] == ["W1"]
    assert r["wafers"][0]["n"] == 1


def test_ac7_error_paths():
    """# AC7:x_field=None → ValueError;wafer_map_png 空 dies → ValueError。"""
    from wafer_grid import build_wafer_grids, wafer_map_png
    with pytest.raises(ValueError):
        build_wafer_grids([], {}, wafer_field="w", x_field=None, y_field="cy")
    with pytest.raises(ValueError):
        wafer_map_png({"wafer_id": "W", "dies": []})


def test_ac8_deterministic():
    """# AC8:同輸入兩次呼叫完全相等。"""
    from wafer_grid import build_wafer_grids
    items = [{"item_id": "p0#0", "path": "p0", "flag": 1, "score": 0.5}]
    meta = {"p0": {"w": "W1", "cx": "2", "cy": "3"}}
    kw = dict(wafer_field="w", x_field="cx", y_field="cy")
    assert build_wafer_grids(items, meta, **kw) == build_wafer_grids(items, meta, **kw)


# ── PM 推導測試(設計未明列)─────────────────────────────────────────


def test_derived_empty_items_ok():
    """推導:items 空 → 各清單空、wafers 空、不 raise(§4 邊界)。"""
    from wafer_grid import build_wafer_grids
    r = build_wafer_grids([], {}, wafer_field="w", x_field="cx", y_field="cy")
    assert r == {"wafers": [], "unmatched": [], "missing_coord": [],
                 "missing_wafer": []}


def test_derived_png_translation_invariance():
    """推導:所有 die 座標整體平移 → 圖完全相同(線性映射到畫布)。"""
    from wafer_grid import wafer_map_png
    base = [{"x": 0.0, "y": 0.0, "flag": 1, "path": "a", "score": None},
            {"x": 1.0, "y": 1.0, "flag": 0, "path": "b", "score": None}]
    shifted = [{**d, "x": d["x"] + 100.0, "y": d["y"] - 50.0} for d in base]
    w1 = {"wafer_id": "W", "lot": None, "n": 2, "k": 1, "dies": base}
    w2 = {"wafer_id": "W", "lot": None, "n": 2, "k": 1, "dies": shifted}
    assert wafer_map_png(w1, cell=8, pad=1).tobytes() == \
        wafer_map_png(w2, cell=8, pad=1).tobytes()


def test_derived_single_die_png():
    """推導:單一 die → 1 格畫布照畫(§4:唯一值 step=1)。"""
    from wafer_grid import wafer_map_png
    w = {"wafer_id": "W", "lot": None, "n": 1, "k": 1,
         "dies": [{"x": 7.0, "y": 9.0, "flag": 1, "path": "a", "score": None}]}
    img = wafer_map_png(w, cell=10, pad=0)
    assert img.size == (10, 10)
    assert img.convert("RGB").getpixel((5, 5)) == (220, 60, 60)
