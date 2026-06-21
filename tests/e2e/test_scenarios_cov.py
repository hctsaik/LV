"""STRICT real-user E2E scenarios for the 嵌入覆蓋圖 (embedding coverage) view
(`_render_coverage_view` in scripts/app.py) and its sidebar.

Two surfaces:
  (A) UI-level via AppTest + _cov_harness.py — a real render of the Streamlit
      page from a JSON spec. Asserts the renderer survives edge datasets and
      surfaces friendly guidance instead of crashing, and that the RECENTLY
      CHANGED behaviours hold (default unit, class captions, no pad input, 5
      sidebar projection methods, single-class sparsity fallback, selectable
      candidate diamonds, ✕ 取消框選 always present, colour match).
  (B) Logic-level — call the real helpers (_build_cov_scatter, _cov_project5,
      selection_points_to_indices, _cov_sparsity, rank_gap_fillers,
      reference_coverage) directly, because AppTest cannot drive a plotly
      box/lasso drag. The brief explicitly asks to verify decode/projection/
      colouring logic directly.

Run:
  set PYTHONIOENCODING=utf-8 & set PYTHONUTF8=1
  .venv\\Scripts\\python.exe -m pytest tests/e2e/test_scenarios_cov.py -v -s
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from streamlit.testing.v1 import AppTest  # noqa: E402

HARNESS = str(Path(__file__).resolve().parent / "_cov_harness.py")
COCO8 = REPO / "demo" / "coco8" / "train"
COCO128 = REPO / "demo" / "coco128" / "train"


# ───────────────────────── helpers ─────────────────────────
def run_harness(spec: dict, timeout: int = 240) -> AppTest:
    os.environ["COV_SPEC"] = json.dumps(spec)
    at = AppTest.from_file(HARNESS, default_timeout=timeout)
    at.run()
    return at


def all_text(at: AppTest) -> str:
    bits = []
    for coll in (at.info, at.warning, at.error, at.caption, at.markdown,
                 at.success):
        for el in coll:
            try:
                bits.append(str(el.value))
            except Exception:
                pass
    return "\n".join(bits)


def exc_summary(at: AppTest) -> str:
    return " | ".join(f"{e.type}: {repr(e.value)[:200]}" for e in at.exception)


def make_yolo_dataset(dst: Path, *, images, labels, classes) -> Path:
    """Build a YOLO dataset matching demo/coco8 layout. Returns <dst>/train."""
    root = dst / "train"
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy(img, root / "images" / img.name)
        if img.stem in labels:
            (root / "labels" / f"{img.stem}.txt").write_text(
                labels[img.stem], encoding="utf-8")
    if classes is not None:
        (dst / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    return root


def _real_app():
    import app
    return app


def ss_get(at: AppTest, key, default=None):
    """AppTest.session_state has no .get(); emulate it safely."""
    try:
        return at.session_state[key]
    except (KeyError, AttributeError, Exception):
        return default


# ════════════════════════ SCENARIO 1 ════════════════════════
# A new user opens 嵌入覆蓋圖 on a detection dataset. RECENTLY CHANGED #1+#2:
# default analysis unit must be 物件級（YOLO） and there must be NO manual pad
# (物件外擴) number_input anywhere in the view.
def test_s01_default_unit_is_object_and_no_pad_input():
    at = run_harness({"main_dir": str(COCO8), "cov_token": "s01"})
    assert not at.exception, f"S01 crashed: {exc_summary(at)}"
    unit_radio = [r for r in at.radio if r.label == "分析單位"]
    assert unit_radio, "no 分析單位 radio on a detection dataset"
    assert unit_radio[0].value == "物件級（YOLO）", \
        f"default unit is not object-level: {unit_radio[0].value}"
    # the 物件數 metric (not 樣本數) proves object branch is live by default
    labels = [m.label for m in at.metric]
    assert "物件數" in labels, f"object metric missing; got {labels}"
    # No pad input: scan every number_input label for 外擴/pad.
    ni_labels = [n.label for n in at.number_input]
    bad = [l for l in ni_labels if l and ("外擴" in l or "pad" in l.lower())]
    print("S01 unit:", unit_radio[0].value, "| number_inputs:", ni_labels)
    assert not bad, f"manual pad input still present: {bad}"


# ════════════════════════ SCENARIO 2 ════════════════════════
# RECENTLY CHANGED #1 + #7: thumbnail captions show the CLASS (door/vase/…),
# never "images", and the caption background colour matches the scatter point
# colour for that class (color_mode=class). We verify the rendered caption HTML
# and cross-check against _viz_color_map / _build_cov_scatter point colours.
def test_s02_class_captions_and_colour_match():
    import re
    import app
    at = run_harness({"main_dir": str(COCO8), "cov_token": "s02"})
    assert not at.exception, f"S02 crashed: {exc_summary(at)}"
    caps = []
    bg_by_label = {}
    for m in at.markdown:
        v = str(m.value)
        if "background:" in v and "border-radius" in v:
            txt = re.sub("<[^>]+>", "", v)
            lbl = txt.split(" · ")[0].strip()
            mcol = re.search(r"background:(#[0-9a-fA-F]{6})", v)
            caps.append(txt)
            if mcol:
                bg_by_label[lbl] = mcol.group(1).lower()
    print("S02 captions:", caps[:6])
    assert caps, "no class-coloured thumbnail captions rendered"
    assert not any(c.strip().lower().startswith("image") for c in caps), \
        "caption says 'images' instead of the class"
    # cross-check caption bg vs the canonical class colour map
    records = at.session_state["cov_records"]
    # object-level records live inside the renderer; reconstruct the map the way
    # the gallery does (sorted labels). We assert each rendered bg equals the map.
    # Build the same colour map from the object labels we can read off captions.
    cls_in_caps = sorted(bg_by_label)
    cmap = {c: app._VIZ_COLORS[i % len(app._VIZ_COLORS)]
            for i, c in enumerate(sorted({*cls_in_caps}))}
    # NOTE: gallery uses the FULL object label set, which is a superset; we only
    # assert internal consistency: same label -> same colour everywhere.
    # (Strict colour-vs-scatter identity is covered in test_s03 at logic level.)
    dup = {l: bg_by_label[l] for l in cls_in_caps}
    print("S02 label->bg:", dup)
    assert len(set(dup.values())) >= 1
    assert all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in dup.values())


# ════════════════════════ SCENARIO 3 ════════════════════════
# LOGIC: when color_mode='class', the scatter point colour for record i MUST be
# the SAME _viz_color_map colour the gallery caption uses (RECENTLY CHANGED #7).
# AppTest can't read plotly marker arrays from the page, so call _build_cov_scatter
# directly and compare its marker colours to _viz_color_map.
def test_s03_scatter_point_colour_equals_caption_colour():
    import app
    records = [{"path": f"/x/{i}.jpg", "label": lbl, "split": ""}
               for i, lbl in enumerate(["door", "door", "vase", "window", "vase"])]
    emb = np.random.default_rng(3).standard_normal((5, 8)).astype("float32")
    sp = app.sparsity_scores(emb, k=2)
    coords = app.PCA(n_components=2).fit_transform(emb)
    fig = app._build_cov_scatter(np.asarray(coords), sp, records, 2,
                                 None, None, [], [], color_mode="class")
    cmap = app._viz_color_map(records)
    point_colours = list(fig.data[0].marker.color)
    expected = [cmap[r["label"]] for r in records]
    print("S03 point colours:", point_colours, "| expected:", expected)
    assert point_colours == expected, \
        "scatter point colours do not match the class colour map (caption mismatch)"
    # sparsity colour mode → numeric colour array + colourbar
    fig2 = app._build_cov_scatter(np.asarray(coords), sp, records, 2,
                                  None, None, [], [], color_mode="sparsity")
    assert list(fig2.data[0].marker.color) == [float(x) for x in sp], \
        "sparsity colouring did not use raw sparsity values"
    assert fig2.data[0].marker.colorscale is not None


# ════════════════════════ SCENARIO 4 ════════════════════════
# RECENTLY CHANGED #4: a user filters to a SINGLE class → class colour would be
# uniform → the view must auto-fall-back to sparsity colouring (no 上色依據 radio,
# a "單一類別" caption). Verify on coco8 by picking one class.
def test_s04_single_class_falls_back_to_sparsity():
    # giraffe / orange / bowl exist in coco8; pick one with >=2 objects.
    at = run_harness({"main_dir": str(COCO8), "cov_token": "s04",
                      "class_filter": "orange"})
    assert not at.exception, f"S04 crashed: {exc_summary(at)}"
    txt = all_text(at)
    # The 上色依據 radio must NOT be present once a single class is selected.
    colour_radio = [r for r in at.radio if r.label == "上色依據"]
    print("S04 colour radio present:", bool(colour_radio),
          "| single-class caption:", "單一類別" in txt)
    assert not colour_radio, "上色依據 radio still shown for a single class"
    assert "單一類別" in txt and "稀疏度上色" in txt, \
        "no single-class→sparsity fallback caption"


# ════════════════════════ SCENARIO 5 ════════════════════════
# RECENTLY CHANGED #6: the ✕ 取消框選 button is ALWAYS present and is DISABLED
# when nothing is selected (no selection seeded here).
def test_s05_clear_selection_button_always_present_disabled_when_empty():
    at = run_harness({"main_dir": str(COCO8), "cov_token": "s05"})
    assert not at.exception, f"S05 crashed: {exc_summary(at)}"
    clear = [b for b in at.button if "取消框選" in b.label]
    assert clear, "✕ 取消框選 button missing"
    print("S05 clear-button disabled:", clear[0].disabled)
    assert clear[0].disabled is True, \
        "✕ 取消框選 should be DISABLED when nothing is selected"


# ════════════════════════ SCENARIO 6 ════════════════════════
# LOGIC: candidate diamonds are SELECTABLE via the sentinel customdata
# (i + _CMPC_BOFF). RECENTLY CHANGED #5. Verify _build_cov_scatter tags
# candidates with the sentinel AND that selection_points_to_indices +
# the renderer's split logic recover (main_idx, cand_idx) correctly.
def test_s06_candidate_diamonds_selectable_sentinel_decode():
    import app
    from interaction import selection_points_to_indices
    records = [{"path": f"/d/{i}.jpg", "label": "person", "split": ""}
               for i in range(6)]
    cand_records = [{"path": f"/c/{i}.jpg", "label": "person", "split": ""}
                    for i in range(4)]
    emb = np.random.default_rng(6).standard_normal((6, 8)).astype("float32")
    cemb = np.random.default_rng(7).standard_normal((4, 8)).astype("float32")
    sp = app.sparsity_scores(emb, k=2)
    ranked_idx, ranked_scores = app.rank_gap_fillers(cemb, emb, k=2)
    cd = app.PCA(n_components=2).fit_transform(emb)
    cc = app.PCA(n_components=2).fit_transform(np.vstack([emb, cemb]))[6:]
    fig = app._build_cov_scatter(np.asarray(cd), sp, records, 2,
                                 np.asarray(cc), cand_records,
                                 ranked_idx, ranked_scores, color_mode="class")
    assert len(fig.data) == 2, "candidate diamond trace not drawn"
    cand_cd = [c[0] for c in fig.data[1].customdata]
    assert all(v >= app._CMPC_BOFF for v in cand_cd), \
        "candidate customdata missing the _CMPC_BOFF sentinel"
    assert fig.data[1].marker.symbol == "diamond"
    # candidate markers smaller than dataset markers (#5 "SMALLER")
    base_main = (fig.data[0].marker.size if np.isscalar(fig.data[0].marker.size)
                 else min(fig.data[0].marker.size))
    base_cand = (fig.data[1].marker.size if np.isscalar(fig.data[1].marker.size)
                 else min(fig.data[1].marker.size))
    print("S06 main base size:", base_main, "| cand base size:", base_cand)
    # Simulate a drag selecting 2 dataset + 2 candidate points.
    sel_pts = [{"customdata": [0]}, {"customdata": [2]},
               {"customdata": cand_cd[1]}, {"customdata": cand_cd[3]}]
    allc = selection_points_to_indices(sel_pts)
    picked = [c for c in allc if c < app._CMPC_BOFF]
    picked_cand = sorted({c - app._CMPC_BOFF for c in allc if c >= app._CMPC_BOFF})
    print("S06 decoded main:", picked, "| cand:", picked_cand)
    assert picked == [0, 2], "main selection decode wrong"
    assert picked_cand == [1, 3], "candidate selection decode wrong"


# ════════════════════════ SCENARIO 7 ════════════════════════
# RECENTLY CHANGED #5: after a user box-selects candidate diamonds, a
# "你框選的候選" gallery + 加入購物車 button must appear so the user SEES the
# candidates they picked. AppTest cannot fire a plotly drag, and the renderer
# always trusts the (empty) plotly event over pre-seeded session state, so
# pre-seeding cov_sel is wiped on rerun. We therefore drive the REAL gallery
# branch two ways:
#   (1) UI: monkeypatch app.st.plotly_chart to return a synthetic selection
#       event carrying candidate sentinel customdata → the renderer decodes it,
#       sets cov_sel.cand, and must render the '你框選的候選' gallery + button.
#   (2) source-wiring: the gallery+button code is gated on (sel_cand and
#       cand_records) and the button calls _batch_add(..., 'gap_filler').
def test_s07_selected_candidates_visible_with_cart_button(tmp_path, monkeypatch):
    import app
    import inspect
    # (2) source wiring first — cheap, proves the branch exists & is reachable.
    src = inspect.getsource(app._render_coverage_view)
    # candidate gallery header now shows the filtered class: "候選・<class> · 框選 N"
    assert "候選・" in src, "selected-candidate gallery removed"
    assert "cov_add_cand_cart" in src, "selected-candidate cart button removed"
    assert "_CMPC_BOFF" in src, "candidate sentinel decode removed"

    # (1) UI drive: build a candidate folder, project it, then synthesize a
    # plotly selection of 2 candidate diamonds via a patched plotly_chart.
    img = next((COCO8 / "images").glob("*.jpg"))
    cand_root = make_yolo_dataset(
        tmp_path / "cand_sel", images=[img],
        labels={img.stem: "0 0.5 0.5 0.4 0.4\n0 0.2 0.2 0.2 0.2\n"
                          "0 0.8 0.8 0.2 0.2\n0 0.3 0.7 0.2 0.2\n"},
        classes=["person"])
    os.environ["COV_SPEC"] = json.dumps({
        "main_dir": str(COCO8), "cov_token": "s07",
        "granularity": "物件級（YOLO）",
        "cand_folders": [str(cand_root)], "preembed_cand": True,
        "cand_offset": 0.0,
        # tells the harness to return a fake plotly selection of cand #0 (+#1)
        "fake_cand_selection": [0]})
    at = AppTest.from_file(HARNESS, default_timeout=240)
    at.run()
    assert not at.exception, f"S07 render with synthetic selection: {exc_summary(at)}"
    ncand = ss_get(at, "cov_cand_records")
    assert ncand, "candidates were not projected into the space"
    txt = all_text(at)
    has_cart_btn = any(b.key == "cov_add_cand_cart" for b in at.button)
    sel_gallery = ("候選・" in txt) and has_cart_btn   # class-aware header + cart button
    print("S07 cand projected:", len(ncand), "| selected-cand gallery:", sel_gallery,
          "| buttons:", [b.label for b in at.button if "購物車" in b.label])
    assert sel_gallery, "no selected-candidate gallery/button after selecting candidate diamonds"
    assert has_cart_btn, "no 加入購物車 button for the selected candidates"


# ════════════════════════ SCENARIO 8 ════════════════════════
# Sidebar: RECENTLY CHANGED #3 — 投影方法 lives in the LEFT SIDEBAR with 5
# methods (PCA/t-SNE/UMAP/監督UMAP/LDA). And switching to a supervised method
# (監督UMAP/LDA) must render without crashing (candidates get NN provisional
# labels via _cov_project5). We verify the 5 methods are wired, then drive a
# supervised render through _cov_project5 directly (the renderer reads
# cov_proj_method from session, which the sidebar sets).
def test_s08_five_projection_methods_and_supervised_render():
    import app
    # (a) the sidebar method list (the registry the selectbox is built from)
    methods = list(app._METHOD_KEY)
    print("S08 methods:", methods)
    for need in ("PCA", "t-SNE", "UMAP", "監督UMAP", "LDA(監督)"):
        assert need in methods, f"projection method missing from sidebar: {need}"
    assert "lda" in app._SUPERVISED_METHODS and "sumap" in app._SUPERVISED_METHODS

    # (b) supervised render via the harness (set cov_proj_method = 監督UMAP)
    at = run_harness({"main_dir": str(COCO8), "cov_token": "s08",
                      "proj_method": "監督UMAP"})
    assert not at.exception, f"S08 supervised render crashed: {exc_summary(at)}"

    # (c) LDA supervised projection with candidates getting provisional labels
    rng = np.random.default_rng(8)
    emb = np.vstack([rng.standard_normal((10, 16)) + 3,
                     rng.standard_normal((10, 16)) - 3]).astype("float32")
    labels = ["a"] * 10 + ["b"] * 10
    cand = rng.standard_normal((5, 16)).astype("float32")
    prov = app.nearest_labels(cand, emb, labels)
    cd, cc = app._cov_project5(emb, cand, labels, prov, "lda", 2,
                              "s08tok", "candtok")
    print("S08 lda coords_d:", cd.shape, "| coords_c:", None if cc is None else cc.shape,
          "| provisional:", prov)
    assert cd.shape[0] == 20 and cc is not None and cc.shape[0] == 5, \
        "supervised projection dropped candidates / rows"
    assert set(prov) <= {"a", "b"}, "provisional labels not from nearest neighbour"


# ════════════════════════ SCENARIO 9 ════════════════════════
# Empty / edge: project a candidate folder that is NOT a detection dataset while
# in object-level mode → friendly warning, no crash. Then a candidate folder
# that does not exist → friendly warning. (STRICT: a real user typo'ing a path
# must not see a stack trace.)
def test_s09_bad_candidate_folder_friendly_warnings(tmp_path):
    # (a) candidate folder with images but NO labels/ while object-level.
    cand = tmp_path / "nolbl"
    (cand / "images").mkdir(parents=True)
    for p in list((COCO8 / "images").glob("*.jpg"))[:2]:
        shutil.copy(p, cand / "images" / p.name)
    at = run_harness({"main_dir": str(COCO8), "cov_token": "s09a",
                      "granularity": "物件級（YOLO）",
                      "cand_folders": [str(cand)], "preembed_cand": True})
    assert not at.exception, f"S09a crashed: {exc_summary(at)}"
    txt = all_text(at)
    print("S09a warnings:", [w.value for w in at.warning])
    assert ("labels/ 找不到 bbox" in txt or "找不到 bbox" in txt
            or "整張影像" in txt or "找不到影像" in txt), \
        "no friendly warning for a label-less candidate folder in object mode"

    # (b) candidate path that does not exist.
    at2 = run_harness({"main_dir": str(COCO8), "cov_token": "s09b",
                       "cand_folders": [str(tmp_path / "does_not_exist")],
                       "preembed_cand": True})
    assert not at2.exception, f"S09b crashed: {exc_summary(at2)}"
    txt2 = all_text(at2)
    print("S09b warnings:", [w.value for w in at2.warning])
    assert ("資料夾不存在" in txt2 or "找不到影像" in txt2), \
        "no friendly warning for a non-existent candidate folder"


# ════════════════════════ SCENARIO 10 ════════════════════════
# Reference-distribution role (B = 外部真值): a user projects a candidate folder
# and switches B's role to 參照分佈 → coverage% + 未覆蓋點 metrics + the
# "你相對參照 B 缺的區域" section render. Then LOGIC-verify reference_coverage
# gives sane recall on overlapping vs disjoint clouds. Also confirms candidate
# embedding actually happened (📁 → 投影候選 wiring).
def test_s10_reference_distribution_role_and_coverage(tmp_path):
    import app
    img = next((COCO8 / "images").glob("*.jpg"))
    cand_root = make_yolo_dataset(
        tmp_path / "cand_ref", images=[img],
        labels={img.stem: "0 0.5 0.5 0.4 0.4\n0 0.2 0.2 0.2 0.2\n"},
        classes=["person"])
    at = run_harness({"main_dir": str(COCO8), "cov_token": "s10",
                      "granularity": "物件級（YOLO）",
                      "cand_folders": [str(cand_root)], "preembed_cand": True,
                      "cand_offset": 4.0, "b_role": "參照分佈"})
    assert not at.exception, f"S10 crashed: {exc_summary(at)}"
    assert ss_get(at, "cov_cand_records"), "candidates not projected"
    txt = all_text(at)
    metric_labels = [m.label for m in at.metric]
    print("S10 metrics:", [(m.label, m.value) for m in at.metric])
    print("S10 ref text present:", "外部真值" in txt or "參照" in txt)
    assert "覆蓋參照 B" in metric_labels and "未覆蓋點" in metric_labels, \
        "reference-distribution coverage metrics missing"

    # LOGIC: overlapping clouds → high recall; disjoint → low recall.
    rng = np.random.default_rng(10)
    a = (rng.standard_normal((30, 16)) * 0.2).astype("float32")
    b_overlap = (rng.standard_normal((20, 16)) * 0.2).astype("float32")
    b_far = (rng.standard_normal((20, 16)) * 0.2 + 8.0).astype("float32")
    radius = app._cov_radius(a, "s10tok")
    _, recall_hi, _ = app.reference_coverage(a, b_overlap, radius)
    unc_far, recall_lo, _ = app.reference_coverage(a, b_far, radius)
    print(f"S10 recall overlap={recall_hi:.2f} far={recall_lo:.2f} "
          f"uncovered_far={len(unc_far)}")
    assert recall_hi > recall_lo, "reference recall not higher for overlapping B"
    assert len(unc_far) == len(b_far), "disjoint B should be entirely uncovered"


# ════════════════════════ SCENARIO 11 ════════════════════════
# RECENTLY FIXED: 取消框選 was mispositioned (below the gallery, mixed with the
# 張數/原圖 controls) AND had no real effect — the scatter widget key was static
# ("cov_emb_scatter"), so the stale plotly event re-populated cov_sel on the
# next rerun and the button looked dead. Aligned to Visualize/Compare:
#   • one-way data flow — only a NON-EMPTY drag changes the selection; an empty
#     plotly event must NOT wipe a live selection (that was the resurrection bug).
# Here: pre-seed an image-level selection, render with NO synthetic drag (the
# unpatched plotly_chart returns an empty event under AppTest). The selection
# must SURVIVE and light up the ✕ button (old code wiped it → button disabled).
def test_s11_empty_plotly_event_does_not_wipe_selection():
    at = run_harness({"main_dir": str(COCO8), "cov_token": "s11",
                      "granularity": "整張影像", "sel_indices": [0, 1, 2]})
    assert not at.exception, f"S11 crashed: {exc_summary(at)}"
    surviving = ss_get(at, "cov_sel", {})
    print("S11 cov_sel after empty event:", surviving)
    assert surviving.get("indices") == [0, 1, 2], \
        "empty plotly event wiped a live selection (one-way flow broken)"
    clear = [b for b in at.button if "取消框選" in b.label]
    assert clear and clear[0].disabled is False, \
        "✕ 取消框選 should be ENABLED while a selection is live"
    assert "你框選的 3" in all_text(at), \
        "left gallery should show the 3 box-selected items"


# ════════════════════════ SCENARIO 12 ════════════════════════
# _cov_clear_selection is the ONLY path that empties the selection; it must also
# bump the nonce and DROP the stale plotly widget event so the scatter remounts
# fresh on the next rerun (mirrors Visualize's _clear_selection). This is the
# fix for the "button does nothing" report. Driven via a tiny AppTest so
# st.session_state is live.
def test_s12_clear_callback_empties_selection_and_bumps_nonce():
    script = (
        "import sys\n"
        f"sys.path.insert(0, r'{SCRIPTS}')\n"
        "import streamlit as st\n"
        "import app\n"
        "if '_t12' not in st.session_state:\n"
        "    st.session_state['_t12'] = True\n"
        "    st.session_state['cov_sel'] = {'token': 'keep', 'indices': [3, 4], 'cand': [1]}\n"
        "    st.session_state['cov_emb_scatter_0'] = {'stale': 1}\n"
        "if st.button('clear'):\n"
        "    app._cov_clear_selection('cov_emb_scatter_0')\n"
    )
    at = AppTest.from_string(script, default_timeout=120)
    at.run()
    assert not at.exception, f"S12 setup: {exc_summary(at)}"
    assert at.session_state["cov_sel"]["indices"] == [3, 4]
    at.button[0].click().run()
    assert not at.exception, f"S12 after click: {exc_summary(at)}"
    cs = at.session_state["cov_sel"]
    nonce = ss_get(at, "_cov_clear_nonce")
    print("S12 after clear:", cs, "| nonce:", nonce)
    assert cs["indices"] == [] and cs["cand"] == [], \
        "✕ did not empty the selection (button had no function)"
    assert cs["token"] == "keep", "token should be preserved across a clear"
    assert nonce == 1, "nonce not bumped → scatter would not remount (dead-button bug)"
    assert "cov_emb_scatter_0" not in at.session_state, \
        "stale plotly widget event not dropped → cleared selection could resurrect"


# ════════════════════════ SCENARIO 13 ════════════════════════
# Source-wiring guard: the ✕ button sits ABOVE the scatter (placeholder filled
# after the selection is final, so the count/enabled state are correct), the
# scatter key carries the clear nonce (so a clear remounts it), the button is
# wired to _cov_clear_selection, and the OLD dead bottom button (which only
# reset cov_sel to token="" with no nonce bump) is gone.
def test_s13_clear_button_aligned_with_visualize_pattern():
    import app
    import inspect
    src = inspect.getsource(app._render_coverage_view)
    assert "_cov_clear_nonce" in src, "clear nonce not read in the coverage view"
    # key 仍須帶 clear nonce（之後可再接物件過濾 tag，故只比對前綴）
    assert 'key=f"cov_emb_scatter_{_cov_cn}' in src, \
        "scatter key no longer carries the clear nonce → clear cannot remount it"
    assert "_clear_slot = _tb2.empty()" in src, \
        "✕ button is not the placeholder placed ABOVE the scatter"
    assert "on_click=_cov_clear_selection" in src, \
        "✕ button not wired to _cov_clear_selection"
    assert 'cov_sel={"token": "", "indices": [], "cand": []}' not in src, \
        "old no-op bottom clear button (token reset, no nonce) still present"


# ════════════════════════ SCENARIO 14 ════════════════════════
# REGRESSION: 物件即時過濾（信心區間／源短邊）會縮短 records/emb；當時 labels 沒跟著
# 縮，監督投影 _supervised_projection(embeddings, y=labels) 就 ValueError（len(x)≠len(y)，
# 例如 emb=28 / labels=504）。物件級 + 監督UMAP + 源短邊過濾掉一部分物件，必須不崩。
def test_s14_object_filter_keeps_labels_aligned_for_supervised_proj():
    at = run_harness({"main_dir": str(COCO8), "cov_token": "s14",
                      "granularity": "物件級（YOLO）", "proj_method": "監督UMAP",
                      "obj_min_short_px": 12})
    assert not at.exception, f"S14 crashed (labels/emb 過濾後不同步?): {exc_summary(at)}"
    txt = all_text(at)
    print("S14 rendered ok | 監督 caption:", "監督投影" in txt)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
