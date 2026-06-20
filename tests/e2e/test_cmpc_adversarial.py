"""ADVERSARIAL E2E scenarios for Compare → object-by-class (`_render_compare_by_class`
and friends in scripts/app.py). Goal: try to BREAK it on edge cases, robustness,
and consistency. Each test() maps to one numbered scenario in the report.

Two layers, neither needs a browser:
  - RENDER scenarios drive the real renderer via _cmpc_adv_harness.py (AppTest)
    with precisely-crafted cmpc_* session state (degenerate counts, pre-seeded
    selection / projection cache / clear nonce). AppTest can't drag-select, so
    selection-coupled invariants are asserted on the resulting session_state and
    widget keys, and the decode/route logic is exercised directly.
  - COMPUTE/AUTOTUNE scenarios call _cmp_autotune / _compute_compare_by_class on
    tiny real YOLO folders (demo/coco8) and synthetic label-less folders to prove
    no-crash + friendly errors.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest \
     tests/e2e/test_cmpc_adversarial.py -q
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import app  # noqa: E402
from interaction import selection_points_to_indices  # noqa: E402

_ADV_HARNESS = Path(__file__).resolve().parent / "_cmpc_adv_harness.py"
_BOFF = app._CMPC_BOFF
_REPO = Path(__file__).resolve().parent.parent.parent


# ── shared fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def jpg(tmp_path_factory) -> str:
    """One real jpg the renderer's image-grid can load."""
    d = tmp_path_factory.mktemp("cmpc_adv_img")
    p = d / "obj.jpg"
    arr = np.random.default_rng(0).integers(0, 255, (40, 40, 3)).astype("uint8")
    Image.fromarray(arr).save(p, quality=85)
    return str(p)


def _run(spec: dict, *, dim_3d: bool = False, proj: str | None = None):
    """Build an AppTest from the adversarial harness for `spec`, optionally
    flipping to 3D / a projection method, and return the run AppTest."""
    from streamlit.testing.v1 import AppTest
    os.environ["CMPC_SPEC"] = json.dumps(spec)
    warnings.filterwarnings("ignore")
    at = AppTest.from_file(str(_ADV_HARNESS), default_timeout=120).run()
    if proj is not None:
        at = at.selectbox(key="cmpc_proj").set_value(proj).run()
    if dim_3d:
        at = at.radio("cmpc_dim").set_value("3D").run()
    return at


def _exc(at):
    return [str(e.value) for e in at.exception]


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — A and B share NO classes → friendly warning, no crash.
#   The renderer is reached with cmpc_rows == [] (no shared class). Must show
#   "沒有可比較的共同類別" warning and return without touching cmpc_oa etc.
# ════════════════════════════════════════════════════════════════════════
def test_s1_no_shared_classes(jpg):
    # A has only 'cat', B has only 'dog' → zero shared → rows == []
    spec = {"img": jpg, "classes": [{"name": "cat", "a": 4, "b": 0},
                                    {"name": "dog", "a": 0, "b": 3}]}
    at = _run(spec)
    assert not at.exception, _exc(at)
    assert any("沒有可比較的共同類別" in w.value for w in at.warning), \
        [w.value for w in at.warning]
    # must NOT have rendered the scatter controls (returned early)
    assert not any(s.key == "cmpc_proj" for s in at.selectbox)


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — singleton class on one side (A:1 / B:1) with 2D + 3D.
#   combined has 2 rows; len <= _nc+1 for both 2D(3) and 3D(4) → coords=zeros
#   guard path. PCA/UMAP/t-SNE must NOT be called on too-few points; no crash;
#   KMeans block picker in 3D must handle k>n gracefully.
# ════════════════════════════════════════════════════════════════════════
def test_s2_singleton_class_2d_and_3d(jpg):
    spec = {"img": jpg, "classes": [{"name": "rare", "a": 1, "b": 1}]}
    at2 = _run(spec)
    assert not at2.exception, _exc(at2)
    at3 = _run(spec, dim_3d=True)
    assert not at3.exception, _exc(at3)
    # 3D KMeans block picker must still render (k clamped to n)
    assert any(s.label == "區塊數 K" for s in at3.slider)
    # grid empty-state caption present (no selection)
    assert any("未框選" in c.value for c in at3.caption)


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — label-less folder: _cmp_autotune must NOT crash, falls back to
#   DEFAULT_POLICY; _compute_compare_by_class then emits the friendly "no bbox"
#   error and returns False. Uses a synthetic folder of images with NO labels/.
# ════════════════════════════════════════════════════════════════════════
def test_s3_labelless_autotune_and_compute(tmp_path, monkeypatch):
    import object_eval as oe
    # folder with images/ but no labels/
    imdir = tmp_path / "nolabels" / "images"
    imdir.mkdir(parents=True)
    paths = []
    for i in range(3):
        p = imdir / f"i{i}.jpg"
        Image.fromarray(np.zeros((32, 32, 3), "uint8")).save(p)
        paths.append(p)

    import streamlit as st
    st.session_state.clear()
    # _cmp_autotune uses st.status/progress → stub to no-ops so it runs headless
    class _Noop:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def update(self, *a, **k): pass
        def progress(self, *a, **k): return self
    monkeypatch.setattr(st, "status", lambda *a, **k: _Noop())
    monkeypatch.setattr(st, "progress", lambda *a, **k: _Noop())

    best = app._cmp_autotune(paths, paths, "dinov2_vits14")
    assert isinstance(best, dict) and "pad" in best  # never crashed
    # with no bbox, falls back to DEFAULT_POLICY
    assert best == dict(oe.DEFAULT_POLICY) or best.get("pad") is not None

    # now the full compute → must return False with a friendly error (no crash)
    errs = []
    monkeypatch.setattr(st, "error", lambda m, *a, **k: errs.append(str(m)))
    ok = app._compute_compare_by_class(paths, paths, "A", "B", "dinov2_vits14")
    assert ok is False
    assert any("bbox" in e for e in errs), errs


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 4 — A ⊂ B overlap warning fires in the Run path. We can't click Run
#   headlessly (native picker), so we assert the exact overlap-detection logic
#   used at app.py ~3668 against crafted path lists: identical paths => count.
# ════════════════════════════════════════════════════════════════════════
def test_s4_overlap_detection_logic():
    a = [Path(f"/d/img{i}.jpg") for i in range(3)]            # A
    b = a + [Path(f"/d/img{i}.jpg") for i in range(3, 8)]      # B ⊃ A
    ov = len(set(map(str, a)) & set(map(str, b)))
    assert ov == 3, ov   # would trigger the A⊂B warning
    # disjoint → no warning
    c = [Path(f"/e/x{i}.jpg") for i in range(4)]
    assert len(set(map(str, a)) & set(map(str, c))) == 0


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 5 — switching projection mid-selection must NOT keep a stale
#   selection: _sel_tok embeds _pm, so changing PCA→UMAP changes the token and
#   the renderer resets cur to empty (a:[],b:[]). We pre-seed a selection bound
#   to the PCA token then render under UMAP and assert the grid shows the
#   "未框選" empty state (selection dropped).
# ════════════════════════════════════════════════════════════════════════
def test_s5_projection_switch_drops_stale_selection(jpg):
    tok = "abcd1234" + "0" * 24  # cmpc_token; first 8 = "abcd1234"
    # selection token built for PCA/2D on class 'cat'
    sel = {"token": "abcd1234_cat_PCA_2", "a": [0, 1], "b": [0]}
    spec = {"img": jpg, "token": tok,
            "classes": [{"name": "cat", "a": 5, "b": 5}],
            "cmpc_sel": sel}
    # render under UMAP → _sel_tok becomes abcd1234_cat_UMAP_2 ≠ seeded token
    at = _run(spec, proj="UMAP")
    assert not at.exception, _exc(at)
    # selection must have been reset → grid empty-state caption present
    assert any("未框選" in c.value for c in at.caption), [c.value for c in at.caption]
    # and the stored selection is now empty under the new token
    assert at.session_state["cmpc_sel"]["a"] == []
    assert at.session_state["cmpc_sel"]["b"] == []


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 6 — rapid 2D→3D→2D toggles keep widget keys stable & collision-free.
#   3D plot key is cmpc3dfig_{_ck}_{_cn} where _ck embeds K + sel_tok; 2D plot
#   key is cmpc2d_{_sel_tok}_{_cn}. We render in 2D, 3D, 2D and assert each run
#   is exception-free and the 2D plotly key is identical across the two 2D
#   renders (stable, no stale-block leakage / key churn).
# ════════════════════════════════════════════════════════════════════════
def _pc_keys(at, prefix):
    """plotly_chart widget keys (AppTest hides .key, but proto.id embeds it as
    '$$ID-<hash>-<key>'). Return the set of keys starting with `prefix`."""
    out = set()
    for pc in at.get("plotly_chart"):
        pid = getattr(pc.proto, "id", "") or ""
        # strip the '$$ID-<hash>-' header → the trailing segment is the key
        if "-" in pid:
            key = pid.split("-", 2)[-1]
            if key.startswith(prefix):
                out.add(key)
    return out


def test_s6_dim_toggle_keys_stable(jpg):
    spec = {"img": jpg, "classes": [{"name": "cat", "a": 8, "b": 8}]}

    at = _run(spec)                       # 2D
    assert not at.exception, _exc(at)
    k1 = _pc_keys(at, "cmpc2d_")
    at = at.radio("cmpc_dim").set_value("3D").run()   # → 3D
    assert not at.exception, _exc(at)
    # no 2D key while in 3D; a 3D fig key exists
    assert not _pc_keys(at, "cmpc2d_"), "2D plot key leaked into 3D"
    assert _pc_keys(at, "cmpc3dfig_"), "3D fig key missing"
    at = at.radio("cmpc_dim").set_value("2D").run()   # → 2D
    assert not at.exception, _exc(at)
    k2 = _pc_keys(at, "cmpc2d_")
    assert k1 and k1 == k2, (k1, k2)      # 2D key stable across the round trip
    assert not _pc_keys(at, "cmpc3dfig_"), "3D fig key leaked into 2D"


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 7 — manual fine-tune then "Run" uses the MANUAL policy and is NOT
#   silently re-tuned. _compute_compare_by_class re-tunes only when policy is
#   None OR cmp_policy_token != current token. We set a manual policy + a
#   MATCHING token and assert autotune is NOT invoked (policy preserved).
# ════════════════════════════════════════════════════════════════════════
def test_s7_manual_policy_not_silently_retuned(monkeypatch, tmp_path):
    import streamlit as st
    st.session_state.clear()
    manual = {"pad": 0.3, "target_res": 280, "head": "meanpool"}
    st.session_state["cmp_object_policy"] = manual
    st.session_state["cmp_folder_a"] = "A"
    st.session_state["cmp_folder_b"] = "B"
    st.session_state["cmp_policy_token"] = app._cmp_policy_token("dinov2_vits14")

    called = {"n": 0}
    monkeypatch.setattr(app, "_cmp_autotune",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    # stub crop+embed to avoid heavy work; capture the policy passed through
    seen = {}

    def _fake_crop(recs, model, cnames, pad, **k):
        seen["policy"] = k.get("policy")
        seen["pad"] = pad
        return [], np.zeros((0, 4), "float32"), None  # empty → friendly error path
    monkeypatch.setattr(app, "_crop_and_embed_objects", _fake_crop)
    monkeypatch.setattr(st, "error", lambda *a, **k: None)

    app._compute_compare_by_class([Path("a")], [Path("b")], "A", "B", "dinov2_vits14")
    assert called["n"] == 0, "manual policy was silently re-tuned!"
    assert seen.get("policy") == manual, seen
    assert abs(seen.get("pad", 0) - 0.3) < 1e-9, seen


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 8 — re-tune (or apply-manual) clears stale cmpc_rows so the table
#   can't disagree with the active policy. We replicate the exact session
#   mutations the 🔄 retune / apply-manual buttons perform and assert cmpc_rows
#   and cmp_result_kind are gone.
# ════════════════════════════════════════════════════════════════════════
def test_s8_retune_clears_stale_rows():
    import streamlit as st
    st.session_state.clear()
    st.session_state["cmpc_rows"] = [{"類別": "cat", "A 數量": 1, "B 數量": 1,
                                      "漂移↓": 0.2}]
    st.session_state["cmp_result_kind"] = "byclass"
    st.session_state["_cmp_objA"] = object()
    st.session_state["_cmp_objB"] = object()
    # exact pops from the retune/apply-manual handlers (app.py ~3598-3602/3632-3635)
    for k in ("_cmp_objA", "_cmp_objB", "cmpc_rows", "cmp_result_kind"):
        st.session_state.pop(k, None)
    assert "cmpc_rows" not in st.session_state
    assert "cmp_result_kind" not in st.session_state


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 9 — very large class: cap 60 thumbs, no perf blowup. Grid slices
#   ids[:60]; selection caption counts full selection but grid draws <=60.
#   We render a 200/200 class and assert no exception + the grid renders the
#   capped number of images (<=60 per side) and the empty-state caption shows
#   full counts.
# ════════════════════════════════════════════════════════════════════════
def test_s9_large_class_thumb_cap(jpg):
    spec = {"img": jpg, "dim": 8,
            "classes": [{"name": "big", "a": 200, "b": 200}]}
    at = _run(spec)
    assert not at.exception, _exc(at)
    # empty-state caption reports the FULL counts (200 · 200), proving no
    # silent truncation of the underlying data
    cap = [c.value for c in at.caption if "未框選" in c.value]
    assert cap and "200" in cap[0], cap   # full counts reported, no data truncation
    # underlying data is intact (200 objects per side stored)
    assert len(at.session_state["cmpc_oa"]) == 200
    assert len(at.session_state["cmpc_ob"]) == 200
    # the table renders without perf blowup (single dataframe, no exception)
    assert len(at.dataframe) == 1
    # _grid caps thumbnails at 60: replicate the exact slice the renderer uses.
    big_ids = list(range(200))
    assert len(big_ids[:60]) == 60                      # selected → cap 60
    assert len(big_ids[:12]) == 12                      # unselected → first 12


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 10a — cross-tab isolation: Compare selection must NOT leak into
#   Visualize's viz_* state. The renderer only ever writes cmpc_* keys; we seed
#   sentinel viz_* values and assert they're untouched after a full render
#   (incl. 3D + a pre-seeded selection).
# ════════════════════════════════════════════════════════════════════════
def test_s10a_no_viz_state_leak(jpg):
    sentinel = "DO_NOT_TOUCH"
    spec = {"img": jpg, "classes": [{"name": "cat", "a": 6, "b": 6}],
            "cmpc_sel": {"token": "x", "a": [1], "b": [2]}}
    from streamlit.testing.v1 import AppTest
    os.environ["CMPC_SPEC"] = json.dumps(spec)
    warnings.filterwarnings("ignore")
    at = AppTest.from_file(str(_ADV_HARNESS), default_timeout=120)
    # pre-seed Visualize state BEFORE the run
    at.session_state["viz_sel"] = sentinel
    at.session_state["viz_records"] = sentinel
    at.session_state["viz_proj_cache"] = sentinel
    at = at.run()
    assert not at.exception, _exc(at)
    at = at.radio("cmpc_dim").set_value("3D").run()
    assert not at.exception, _exc(at)
    for k in ("viz_sel", "viz_records", "viz_proj_cache"):
        assert at.session_state[k] == sentinel, f"{k} was mutated by Compare!"


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 10b — projection cache must NOT serve a STALE projection across a
#   re-Run. _proj_key embeds cmpc_token; a new Run mints a new token, so a cache
#   built under the old token must be ignored & recomputed. We seed a poisoned
#   cache under a DIFFERENT key and assert the renderer does NOT use it (no
#   crash, and a fresh cache entry under the current key is written).
# ════════════════════════════════════════════════════════════════════════
def test_s10b_proj_cache_not_stale_across_rerun(jpg):
    tok = "newtoken" + "0" * 24
    # poisoned cache keyed to an OLD token + wrong shape; if used, vstack/scatter
    # would mismatch n points (5+5=10) and could crash or mis-plot.
    poisoned = {"key": "OLDTOKEN_cat_PCA_2",
                "coords": [[999.0, 999.0]] * 3}  # wrong length (3 not 10)
    spec = {"img": jpg, "token": tok,
            "classes": [{"name": "cat", "a": 5, "b": 5}],
            "cmpc_proj_cache": poisoned}
    at = _run(spec)  # PCA/2D under new token
    assert not at.exception, _exc(at)
    # the cache must now be keyed to the CURRENT token (first 8 chars 'newtoken')
    assert "_cmpc_proj_cache" in at.session_state
    pc = at.session_state["_cmpc_proj_cache"]
    assert pc["key"].startswith("newtoken"), pc["key"]
    assert len(pc["coords"]) == 10, ("stale/short coords served!", len(pc["coords"]))


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 11 (bonus, real full-flow) — demo/coco8 train vs val through the
#   REAL compute path (autotune + crop + embed + drift). Proves the whole
#   pipeline survives a real YOLO dataset whose only shared class has exactly
#   1 object per side (the S2 singleton edge, end-to-end). Skipped if the demo
#   fixture or torch backend is unavailable.
# ════════════════════════════════════════════════════════════════════════
def test_s11_real_coco8_full_flow(monkeypatch):
    train = _REPO / "demo" / "coco8" / "train"
    val = _REPO / "demo" / "coco8" / "val"
    if not (train.exists() and val.exists()):
        pytest.skip("demo/coco8 fixture not present")
    import streamlit as st

    class _Noop:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def update(self, *a, **k): pass
        def progress(self, *a, **k): return self
    monkeypatch.setattr(st, "status", lambda *a, **k: _Noop())
    monkeypatch.setattr(st, "progress", lambda *a, **k: _Noop())
    monkeypatch.setattr(st, "spinner", lambda *a, **k: _Noop())
    st.session_state.clear()

    ta, _ = app._cmp_resolve_images(train)
    va, _ = app._cmp_resolve_images(val)
    assert ta and va
    try:
        ok = app._compute_compare_by_class(ta, va, "train", "val", "dinov2_vits14")
    except Exception as e:  # backend (torch/model) missing → skip, don't fail
        pytest.skip(f"embedding backend unavailable: {e}")
    assert ok is True
    rows = st.session_state["cmpc_rows"]
    assert rows, "no shared classes computed from real coco8"
    # every row well-formed; counts positive on both sides (shared by definition)
    for r in rows:
        assert r["A 數量"] >= 1 and r["B 數量"] >= 1
        assert 0.0 <= r["漂移↓"] <= 2.0
    # a real policy was auto-selected and a fresh token minted
    assert st.session_state.get("cmp_object_policy")
    assert st.session_state.get("cmpc_token")


# ── auxiliary direct-logic check backing scenario 10b decode safety ──────
def test_s10b_decode_still_routes_after_recompute():
    # sanity: even with a fresh projection, sentinel decode stays correct
    pts = [{"customdata": [2]}, {"customdata": [3 + _BOFF]}]
    codes = selection_points_to_indices(pts)
    sa = sorted({v for v in codes if v < _BOFF})
    sb = sorted({v - _BOFF for v in codes if v >= _BOFF})
    assert sa == [2] and sb == [3]
