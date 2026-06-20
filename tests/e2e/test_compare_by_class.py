"""Targeted tests for the reworked Compare → object-by-class scatter
(`_render_compare_by_class` in scripts/app.py).

The full Compare path (autotune + crop+embed of every YOLO object on CPU) is
far too slow for E2E, so this module covers the *new* surface two cheaper ways:

1. RENDER (seeded session state):  a Streamlit `AppTest` harness
   (_cmpc_harness.py) seeds the `cmpc_*` keys with a few fake records pointing
   at tiny real jpgs + small numpy arrays, then calls the real renderer. We
   assert the table, projection/dimension controls, and selection grid render
   in BOTH 2D and 3D (where the KMeans block picker + 2D sub-scatter appear)
   with zero exceptions.

2. DECODE (pure logic):  the customdata sentinel routing — codes < _CMPC_BOFF
   are group-A indices into cmpc_oa, codes >= _CMPC_BOFF are (code-BOFF) group-B
   indices into cmpc_ob — exercised through the same `selection_points_to_indices`
   the app uses, including the 3D block→2D `all_codes` re-indexing.

These do not need a browser; they run in-process and import scripts/app.py.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# import the app module (and the shared decode helper) directly
_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import app  # noqa: E402
from interaction import selection_points_to_indices  # noqa: E402

_HARNESS = Path(__file__).resolve().parent / "_cmpc_harness.py"
_BOFF = app._CMPC_BOFF


# ── fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cmpc_fixture(tmp_path_factory) -> Path:
    """A folder with 8 tiny jpgs the harness splits 4/4 into groups A and B."""
    d = tmp_path_factory.mktemp("cmpc_imgs")
    for i in range(8):
        arr = np.random.default_rng(i).integers(0, 255, (48, 48, 3)).astype("uint8")
        Image.fromarray(arr).save(d / f"img_{i}.jpg", quality=85)
    return d


@pytest.fixture()
def harness(cmpc_fixture):
    """An AppTest bound to _cmpc_harness.py with the fixture dir wired in."""
    from streamlit.testing.v1 import AppTest
    os.environ["CMPC_FIXTURE"] = str(cmpc_fixture)
    warnings.filterwarnings("ignore")  # silence use_container_width deprecation
    at = AppTest.from_file(str(_HARNESS), default_timeout=90)
    return at


# ── (1) render: 2D ──────────────────────────────────────────────────────

def test_render_2d_no_exception(harness):
    at = harness.run()
    assert not at.exception, [e.value for e in at.exception]
    # the per-class table + the two selectboxes + the 2D/3D radio all render
    assert any("逐類別比較" in s.value for s in at.subheader)
    assert len(at.dataframe) == 1
    labels = [s.label for s in at.selectbox]
    assert "投影方法" in labels and any("進入某一類" in l for l in labels)
    dim = at.radio("cmpc_dim")
    assert dim.value == "2D" and dim.options == ["2D", "3D"]
    # selection-grid empty state ("未框選 → 顯示前 12 個樣本…") is present
    assert any("未框選" in c.value for c in at.caption)


# ── (1) render: 3D → KMeans block picker → 2D sub-scatter ───────────────

def test_render_3d_block_picker_no_exception(harness):
    at = harness.run()
    assert not at.exception, [e.value for e in at.exception]
    at = at.radio("cmpc_dim").set_value("3D").run()
    assert not at.exception, [e.value for e in at.exception]
    # 3D adds the "區塊數 K" slider and the block-picker selectbox feeding a
    # native 2D box-select sub-scatter (plotly 3D has no on_select)
    assert any(s.label == "區塊數 K" for s in at.slider), [s.label for s in at.slider]
    assert any("進入區塊" in s.label for s in at.selectbox), \
        [s.label for s in at.selectbox]
    assert any("3D 旋轉看區塊" in c.value for c in at.caption)
    # selection grid still renders below the 3D controls
    assert any("未框選" in c.value for c in at.caption)


# ── (1) render: projection method switch survives ───────────────────────

def test_render_projection_switch(harness):
    at = harness.run()
    proj = at.selectbox(key="cmpc_proj")
    # _METHOD_KEY 已擴含監督法；只要核心三法在即可（避免過度脆裂）
    assert {"PCA", "t-SNE"} <= set(proj.options)
    at = proj.set_value("t-SNE").run()
    assert not at.exception, [e.value for e in at.exception]


# ── (2) decode: sentinel routes A vs B correctly (2D path) ──────────────

def _split(sel_codes):
    """The exact A/B split _render_compare_by_class applies to decoded codes."""
    sa = sorted({v for v in sel_codes if v < _BOFF})
    sb = sorted({v - _BOFF for v in sel_codes if v >= _BOFF})
    return sa, sb


def test_decode_sentinel_routes_a_and_b():
    # class 'pick' → A object indices ia, B object indices ib (into oa/ob)
    ia, ib = [3, 7], [1, 4, 9]
    code_a = [[ia[j]] for j in range(len(ia))]
    code_b = [[ib[j] + _BOFF] for j in range(len(ib))]

    # every A code is below the sentinel, every B code is at/above it
    assert all(c[0] < _BOFF for c in code_a)
    assert all(c[0] >= _BOFF for c in code_b)

    # box-select one A point (oa idx 7) + one B point (ob idx 9)
    pts = [{"customdata": [7]}, {"customdata": [9 + _BOFF]}]
    sa, sb = _split(selection_points_to_indices(pts))
    assert sa == [7], sa            # group-A index straight through
    assert sb == [9], sb            # group-B index = code - BOFF

    # select all A → no B leakage, and vice-versa
    sa, sb = _split(selection_points_to_indices([{"customdata": c} for c in code_a]))
    assert sa == sorted(ia) and sb == []
    sa, sb = _split(selection_points_to_indices([{"customdata": c} for c in code_b]))
    assert sa == [] and sb == sorted(ib)


def test_decode_boundary_value():
    # exactly BOFF is group-B index 0 (boundary is inclusive on the B side)
    sa, sb = _split(selection_points_to_indices([{"customdata": [_BOFF]}]))
    assert sa == [] and sb == [0]
    # BOFF-1 is the largest possible group-A index
    sa, sb = _split(selection_points_to_indices([{"customdata": [_BOFF - 1]}]))
    assert sa == [_BOFF - 1] and sb == []


# ── (3) decode: 3D block → 2D sub-scatter customdata routing ────────────

def test_3d_block_to_2d_customdata_routing():
    """In 3D, the block's 2D sub-scatter is built with
    `customdata=[[all_codes[i]] for i in _bidx]`, where `all_codes` is indexed
    by COMBINED position (0..n_a-1 = A, n_a.. = B). Selecting in that sub-scatter
    must decode back to the right oa/ob indices."""
    ia, ib = [3, 7], [1, 4, 9]
    n_a = len(ia)
    code_a = [[ia[j]] for j in range(len(ia))]
    code_b = [[ib[j] + _BOFF] for j in range(len(ib))]
    all_codes = [c[0] for c in code_a] + [c[0] for c in code_b]
    # all_codes maps combined position → sentinel code
    assert all_codes == [3, 7, 1 + _BOFF, 4 + _BOFF, 9 + _BOFF]

    # a KMeans block holds combined positions 1 (A: oa idx 7) and 3 (B: ob idx 4)
    bidx = [1, 3]
    sub_customdata = [[all_codes[i]] for i in bidx]
    assert sub_customdata == [[7], [4 + _BOFF]]

    sa, sb = _split(selection_points_to_indices(
        [{"customdata": c} for c in sub_customdata]))
    assert sa == [7], sa            # combined pos 1 → group-A oa idx 7
    assert sb == [4], sb            # combined pos 3 → group-B ob idx 4

    # the A/B colour split in the sub-scatter keys off the same n_a boundary
    colors = ["#3498db" if i < n_a else "#e74c3c" for i in bidx]
    assert colors == ["#3498db", "#e74c3c"]
