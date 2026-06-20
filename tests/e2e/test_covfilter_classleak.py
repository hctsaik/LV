"""STRICT UI-level verification (real renderers via AppTest) that the candidate
class filter in BOTH coverage views lets through ONLY the picked class.

THE BUG (reported 3x): filtering the MAIN object set to one YOLO class did NOT
filter the CANDIDATE objects projected from another folder, so the
overlay/selection/gallery showed foreign classes (window/door/...).

These tests drive the REAL renderers:
  - _render_coverage_view   (嵌入覆蓋圖)        via _cov_harness.py
  - _render_objcov_view     (按物件・逐類別補洞)  via _objcov_harness.py
with a class filter active + candidates projected, then read back the EXACT
candidate set the renderer handed to the figure builder / overlay trace and
assert ZERO foreign-class candidates.

Data: MAIN = indoor/valid, CANDIDATE = indoor/test (10 shared YOLO classes).
Skips gracefully if the dataset is absent.

Run:
  set PYTHONIOENCODING=utf-8 & set PYTHONUTF8=1
  .venv\\Scripts\\python.exe -m pytest tests/e2e/test_covfilter_classleak.py -v -s
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from streamlit.testing.v1 import AppTest  # noqa: E402

COV_HARNESS = str(Path(__file__).resolve().parent / "_cov_harness.py")
OBJCOV_HARNESS = str(Path(__file__).resolve().parent / "_objcov_harness.py")

MAIN = Path("C:/code/dataset/indoor/valid")
CAND = Path("C:/code/dataset/indoor/test")

pytestmark = pytest.mark.skipif(
    not (MAIN.exists() and CAND.exists()),
    reason="indoor valid/test dataset not present")

# Classes shared by both folders with foreign neighbours in the candidate pool.
TEST_CLASSES = ["cabinet", "cabinetDoor", "chair"]


_LEAKY_ENV = ("LV_CACHE_DIR", "COV_CAPTURE", "OBJCOV_CAPTURE",
              "COV_SPEC", "OBJCOV_SPEC")


@pytest.fixture(autouse=True)
def _restore_env():
    """These tests set process-wide env vars (LV_CACHE_DIR, *_CAPTURE, *_SPEC) and
    the harness monkeypatches module-level globals on `app` (load_model ->
    16-dim fake, _build_cov_scatter wrapper, plotly add_trace wrapper). All of
    those persist in the shared pytest process and would otherwise break later
    real-model E2E tests (16-vs-384 NearestNeighbors mismatch). Snapshot &
    restore everything after each test."""
    import app  # noqa
    import plotly.graph_objects as go
    saved_env = {k: os.environ.get(k) for k in _LEAKY_ENV}
    saved_load = app.load_model
    saved_scatter = app._build_cov_scatter
    saved_addtrace = go.Figure.add_trace
    saved_chart = app.st.plotly_chart   # harness patches this for fake selections
    try:
        yield
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        app.load_model = saved_load
        app._build_cov_scatter = saved_scatter
        go.Figure.add_trace = saved_addtrace
        app.st.plotly_chart = saved_chart


def exc_summary(at: AppTest) -> str:
    return " | ".join(f"{e.type}: {repr(e.value)[:300]}" for e in at.exception)


def ss_get(at: AppTest, key: str, default=None):
    try:
        return at.session_state[key]
    except (KeyError, Exception):
        return default


# ──────────────────── VIEW 1: 嵌入覆蓋圖 ────────────────────
@pytest.mark.parametrize("cls", TEST_CLASSES)
def test_coverage_view_no_foreign_candidate(cls, tmp_path):
    # Isolated cache → fake 16-dim embeddings are used (never stale 384-dim npz).
    os.environ["LV_CACHE_DIR"] = str(tmp_path / "lvcache")
    os.environ["COV_CAPTURE"] = "1"
    os.environ["COV_SPEC"] = json.dumps({
        "main_dir": str(MAIN),
        "cov_token": f"covleak_{cls}",
        "granularity": "物件級（YOLO）",
        "class_filter": cls,                 # the 物件類別 selectbox = pick C
        "proj_method": "PCA",
        "cand_folders": [str(CAND)],
        "preembed_cand": True,               # project candidate folder objects
        "fake_model": True,                  # labels are real; embeddings fake
    })
    at = AppTest.from_file(COV_HARNESS, default_timeout=600)
    at.run()
    assert not at.exception, f"coverage render crashed for {cls}: {exc_summary(at)}"

    labels = ss_get(at, "_cap_cov_cand_labels")
    n_plotted = ss_get(at, "_cap_cov_cand_n")
    assert labels is not None, ("figure builder was never reached — candidate "
                                "capture missing (no candidates projected?)")
    counts = Counter(labels)
    foreign = {k: v for k, v in counts.items() if k != cls}
    print(f"[VIEW1 嵌入覆蓋圖] pick={cls!r} candidate overlay labels={dict(counts)} "
          f"plotted_pts={n_plotted} foreign={foreign} "
          f"-> {'PASS' if not foreign else 'FAIL'}")
    assert labels, f"no candidate of class {cls} reached the overlay (filter too aggressive?)"
    assert not foreign, f"FOREIGN class leaked into 嵌入覆蓋圖 overlay: {foreign}"
    # overlay point count must equal kept candidates of class C
    assert n_plotted == len(labels), \
        f"plotted candidate count {n_plotted} != filtered records {len(labels)}"


# ── Task 3: a SELECTED candidate's gallery entry is always the picked class ──
@pytest.mark.parametrize("cls", ["cabinet", "chair"])
def test_coverage_selected_candidate_is_picked_class(cls, tmp_path):
    """Inject a candidate box-selection (the gallery branch the user reaches by
    lassoing black diamonds). The gallery caption is cand_records[i]['label'];
    since cand_records is the FILTERED list, every selected candidate caption must
    equal the picked class. Selecting indices that span the whole filtered range
    proves the index->record mapping points into the filtered list."""
    os.environ["LV_CACHE_DIR"] = str(tmp_path / "lvcache")
    os.environ["COV_CAPTURE"] = "1"
    # First render to learn how many filtered candidates exist.
    base_spec = {
        "main_dir": str(MAIN), "cov_token": f"covsel_{cls}",
        "granularity": "物件級（YOLO）", "class_filter": cls,
        "proj_method": "PCA", "cand_folders": [str(CAND)],
        "preembed_cand": True, "fake_model": True,
    }
    os.environ["COV_SPEC"] = json.dumps(base_spec)
    at = AppTest.from_file(COV_HARNESS, default_timeout=600)
    at.run()
    assert not at.exception, f"render crashed: {exc_summary(at)}"
    n_filtered = len(ss_get(at, "_cap_cov_cand_labels") or [])
    assert n_filtered > 0

    # Re-render with a candidate selection across the full filtered index range.
    sel = list(range(min(n_filtered, 24)))
    os.environ["COV_SPEC"] = json.dumps({**base_spec, "fake_cand_selection": sel})
    at2 = AppTest.from_file(COV_HARNESS, default_timeout=600)
    at2.run()
    assert not at2.exception, f"render(sel) crashed: {exc_summary(at2)}"

    # Gallery captions for selected candidates are the resolved labels.
    captions = [c.value for c in at2.caption]
    gallery_labels = [c for c in captions if c in
                      ("cabinet", "cabinetDoor", "refrigeratorDoor", "window",
                       "chair", "table", "couch", "door", "openedDoor", "pole")]
    foreign = [c for c in gallery_labels if c != cls]
    print(f"[VIEW1 selection] pick={cls!r} selected={len(sel)} "
          f"gallery_label_captions={Counter(gallery_labels)} foreign={foreign} "
          f"-> {'PASS' if not foreign else 'FAIL'}")
    assert gallery_labels, ("no candidate label captions rendered — selection "
                            "branch not reached")
    assert not foreign, f"selected candidate gallery showed FOREIGN class: {foreign}"


# ──────────────────── VIEW 2: 按物件・逐類別補洞 ────────────────────
@pytest.mark.parametrize("cls", TEST_CLASSES)
def test_objcov_view_no_foreign_candidate(cls, tmp_path):
    os.environ["LV_CACHE_DIR"] = str(tmp_path / "lvcache")
    os.environ["OBJCOV_CAPTURE"] = "1"
    os.environ["OBJCOV_SPEC"] = json.dumps({
        "main_dir": str(MAIN),
        "cov_token": f"objleak_{cls}",
        "cand_folders": [str(CAND)],
        "preembed_cand": True,   # _objcov_embed_candidates: real labels, fake emb
    })
    at = AppTest.from_file(OBJCOV_HARNESS, default_timeout=600)
    at.run()
    assert not at.exception, f"objcov render crashed (default pick): {exc_summary(at)}"

    # Drive the class selectbox to C (the renderer recomputes cand_ci for `pick`).
    picked = False
    for sb in at.selectbox:
        if sb.label and sb.label.startswith("選類別"):
            sb.set_value(cls)
            picked = True
    assert picked, "class-pick selectbox not found in objcov view"
    at.run()
    assert not at.exception, f"objcov render crashed (pick={cls}): {exc_summary(at)}"

    labels = ss_get(at, "_cap_objcov_cand_labels")
    n_plotted = ss_get(at, "_cap_objcov_cand_n")
    cand = ss_get(at, "objcov_cand") or {}
    full = Counter(cand.get("labels") or [])
    assert labels is not None, ("candidate overlay trace was never drawn — "
                                "no candidate of any class? full pool="
                                f"{dict(full)}")
    counts = Counter(labels)
    foreign = {k: v for k, v in counts.items() if k != cls}
    print(f"[VIEW2 按物件逐類別] pick={cls!r} full_cand_pool={dict(full)} "
          f"overlay labels={dict(counts)} plotted_pts={n_plotted} "
          f"foreign={foreign} -> {'PASS' if not foreign else 'FAIL'}")
    assert labels, f"no candidate of class {cls} reached the overlay"
    assert not foreign, f"FOREIGN class leaked into 按物件逐類別 overlay: {foreign}"
    assert n_plotted == len(labels), \
        f"plotted candidate count {n_plotted} != cand_ci size {len(labels)}"
    # Sanity: the FULL candidate pool DID contain foreign classes (otherwise the
    # test is vacuous — the filter had nothing to remove).
    pool_foreign = {k: v for k, v in full.items() if k != cls}
    assert pool_foreign, (f"candidate pool had no foreign classes for {cls}; "
                          "test is vacuous")
    print(f"    (pool had {sum(pool_foreign.values())} foreign objects across "
          f"{len(pool_foreign)} classes that the filter correctly removed)")
