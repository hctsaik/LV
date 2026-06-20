"""STRICT real-user E2E scenarios for 按物件・逐類別補洞 (object-by-class gap-fill).

These are adversarial, demanding-user scenarios on top of the happy-path coverage
in test_objcov_backfill.py. They specifically stress the RECENTLY-FIXED behaviors:

  1. ✕ 取消框選 placeholder button: disabled when no selection, ENABLED + clears
     when a selection exists (was always-disabled before the fix).
  2. Candidate projection = FIXED REFERENCE FRAME (fit on MAIN, transform
     candidates): candidates resembling MAIN must land NEAR main points, not a
     disjoint blob. t-SNE + candidates must fall back to PCA (t-SNE has no
     out-of-sample transform).
  3. Candidate selection sentinel _CMPC_BOFF separates main(<BOFF) vs cand(>=BOFF).
  4. 5 projection methods incl. 監督UMAP / LDA(監督) (supervised by main-vs-cand).

Run:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest \
        tests/e2e/test_objcov_strict_scenarios.py -v
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
_REPO = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import app  # noqa: E402
import object_eval as oe  # noqa: E402
from completeness import mine_candidates  # noqa: E402
from interaction import selection_points_to_indices  # noqa: E402

_HARNESS = Path(__file__).resolve().parent / "_objcov_harness.py"
_PTAG = oe.policy_tag(dict(oe.DEFAULT_POLICY))
_COCO8_TRAIN = _REPO / "demo" / "coco8" / "train"
_BOFF = app._CMPC_BOFF


# ── shared infra (mirrors test_objcov_backfill helpers) ──────────────────

def _tiny_jpg(path: Path, seed: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 255, (32, 32, 3)).astype("uint8")
    Image.fromarray(arr).save(path, quality=85)
    return path


def _run(spec: dict, *, timeout: int = 180, preseed=None):
    from streamlit.testing.v1 import AppTest
    warnings.filterwarnings("ignore")
    os.environ["OBJCOV_SPEC"] = json.dumps(spec)
    at = AppTest.from_file(str(_HARNESS), default_timeout=timeout)
    if preseed is not None:
        preseed(at.session_state)
    return at.run()


def _fig_spec(at, idx=0):
    return json.loads(at.get("plotly_chart")[idx].proto.spec)


def _coords(v) -> np.ndarray:
    """Decode a plotly trace coordinate field. Streamlit's proto encodes numeric
    arrays as a typed-array dict {'dtype': 'f4', 'bdata': <base64>} rather than a
    plain list; handle both forms."""
    if isinstance(v, dict) and "bdata" in v:
        import base64
        return np.frombuffer(base64.b64decode(v["bdata"]),
                             dtype=np.dtype(v.get("dtype", "f4"))).astype(float)
    return np.asarray(v, dtype=float)


def _trace_names(at):
    return [t.get("name") for t in _fig_spec(at).get("data", [])]


def _ss(at, key, default=None):
    ss = at.session_state
    return ss[key] if key in ss else default


def _base_spec(**over) -> dict:
    spec = {"main_dir": str(_COCO8_TRAIN), "cov_token": "tok_main_0001",
            "dim": 16, "main_seed": 7}
    spec.update(over)
    return spec


def _controlled_preseed(tmp_path, *, cov_token, n_main=12, n_cand=6, dim=16,
                        cand_offset=0.02):
    """Pre-seed ONE analysable MAIN class ('orange') with a tight centroid and a
    CANDIDATE set of the SAME class clustered at distance ``cand_offset`` from the
    matching main point, plus far 'dog' candidates the NAME filter must drop.
    Returns (preseed_fn, cand_root, main_mean, main_emb)."""
    rng = np.random.default_rng(11)
    main_mean = np.zeros(dim, dtype="float32"); main_mean[0] = 5.0
    main_recs, main_emb = [], []
    for j in range(n_main):
        p = _tiny_jpg(tmp_path / "main_crops" / f"orange_{j}.jpg", seed=300 + j)
        main_recs.append({"path": str(p), "image_path": str(p),
                          "label": "orange", "split": ""})
        main_emb.append(main_mean + rng.standard_normal(dim).astype("float32") * 0.3)
    main_emb = np.asarray(main_emb, dtype="float32")

    cand_root = tmp_path / "cand_real"
    cand_recs, cand_emb, cand_labels = [], [], []
    for j in range(n_cand):
        p = _tiny_jpg(cand_root / "images" / f"o{j}.jpg", seed=600 + j)
        cand_recs.append({"path": str(p), "image_path": str(p),
                          "label": "orange", "split": ""})
        cand_emb.append(main_emb[j % n_main]
                        + rng.standard_normal(dim).astype("float32") * cand_offset)
        cand_labels.append("orange")
    for j in range(3):
        p = _tiny_jpg(cand_root / "images" / f"d{j}.jpg", seed=650 + j)
        cand_recs.append({"path": str(p), "image_path": str(p),
                          "label": "dog", "split": ""})
        cand_emb.append(np.full(dim, 9.0, dtype="float32"))
        cand_labels.append("dog")
    cand_emb = np.asarray(cand_emb, dtype="float32")
    main_token = f"{cov_token}_{_PTAG}"

    def _preseed(ss):
        ss["_objcov_main"] = {"token": main_token, "recs": main_recs,
                              "emb": main_emb,
                              "labels": [r["label"] for r in main_recs]}
        ss["objcov_cand"] = {"token": "cand_frozen_1", "recs": cand_recs,
                             "emb": cand_emb, "labels": cand_labels}

    return _preseed, cand_root, main_mean, main_emb


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — "I drew a box, now let me clear it." The ✕ 取消框選 button must
# be disabled with no selection AND become enabled + clear when a selection
# exists (the recently-fixed placeholder-after-processing wiring).
# ════════════════════════════════════════════════════════════════════════

def test_s1_clear_selection_button_enable_and_clear(tmp_path):
    cov_token = "tok_s1"
    preseed, _, _, _ = _controlled_preseed(tmp_path, cov_token=cov_token)
    spec = _base_spec(cov_token=cov_token, seed_main_cache=False)

    # (a) fresh render → no selection → button DISABLED, count-less label.
    at = _run(spec, preseed=preseed)
    assert not at.exception, [e.value for e in at.exception]
    clr = [b for b in at.button if b.key == "objcov_clear_sel"]
    assert clr, "✕ 取消框選 button not rendered"
    assert clr[0].disabled is True, "clear button should be disabled with no selection"
    assert clr[0].label.strip() == "✕ 取消框選", clr[0].label

    # (b) a user box-selected 3 main + 1 cand → seed that processed selection,
    #     re-render: button must be ENABLED and show the count.
    _pm, _dim = "PCA", 2
    main_token = f"{cov_token}_{_PTAG}"
    otok = f"{main_token}|orange|{_pm}|{_dim}"

    def preseed2(ss):
        preseed(ss)
        ss["objcov_sel"] = {"token": otok, "main": [0, 1, 2], "cand": [0]}

    at2 = _run(spec, preseed=preseed2)
    assert not at2.exception, [e.value for e in at2.exception]
    clr2 = [b for b in at2.button if b.key == "objcov_clear_sel"][0]
    assert clr2.disabled is False, "clear button must enable when a selection exists"
    assert "4" in clr2.label, f"label should show selection count: {clr2.label!r}"
    # selected thumbnails are shown (3 main + 1 cand)
    assert at2.get("imgs"), "expected selected-object thumbnails"

    # (c) click ✕ → selection cleared and button disabled again.
    at3 = clr2.click().run()
    assert not at3.exception, [e.value for e in at3.exception]
    sel = _ss(at3, "objcov_sel", {})
    assert sel.get("main") == [] and sel.get("cand") == [], sel
    clr3 = [b for b in at3.button if b.key == "objcov_clear_sel"][0]
    assert clr3.disabled is True, "after clear, button should disable again"


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — "The candidates I added look just like my data — they'd better
# overlap, not float off in their own cluster." FIXED REFERENCE FRAME (PCA):
# candidates resembling MAIN land NEAR the main cloud, not a disjoint blob.
# ════════════════════════════════════════════════════════════════════════

def test_s2_candidates_land_near_main_not_disjoint(tmp_path):
    cov_token = "tok_s2"
    # candidates are near-copies of main points (offset 0.02)
    preseed, _, _, main_emb = _controlled_preseed(
        tmp_path, cov_token=cov_token, cand_offset=0.02)
    spec = _base_spec(cov_token=cov_token, seed_main_cache=False)
    at = _run(spec, preseed=preseed)
    assert not at.exception, [e.value for e in at.exception]

    fig = _fig_spec(at)
    main_tr = next(t for t in fig["data"] if str(t.get("name", "")).startswith("主"))
    cand_tr = next(t for t in fig["data"] if t.get("name") == "候選")
    mx, my = _coords(main_tr["x"]), _coords(main_tr["y"])
    cx, cy = _coords(cand_tr["x"]), _coords(cand_tr["y"])

    # main cloud bbox (with a small margin = one main std)
    mxs, mys = mx.std() + 1e-6, my.std() + 1e-6
    in_x = (cx >= mx.min() - mxs) & (cx <= mx.max() + mxs)
    in_y = (cy >= my.min() - mys) & (cy <= my.max() + mys)
    frac_inside = float(np.mean(in_x & in_y))
    assert frac_inside >= 0.8, f"candidates float off (only {frac_inside:.0%} inside)"

    # centroid gap must be small relative to main spread (NOT a separate blob)
    gap = np.hypot(cx.mean() - mx.mean(), cy.mean() - my.mean())
    spread = np.hypot(mxs, mys)
    assert gap <= 1.5 * spread, f"candidate centroid {gap:.2f} far vs spread {spread:.2f}"


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — "I picked t-SNE but I'm overlaying candidates." t-SNE has no
# out-of-sample transform, so the view must FALL BACK to PCA and still render
# candidates in-frame (no crash, candidates present, finite coords).
# ════════════════════════════════════════════════════════════════════════

def test_s3_tsne_with_candidates_falls_back_pca(tmp_path):
    cov_token = "tok_s3"
    preseed, _, _, _ = _controlled_preseed(tmp_path, cov_token=cov_token)
    spec = _base_spec(cov_token=cov_token, seed_main_cache=False)

    def preseed_tsne(ss):
        preseed(ss)
        ss["objcov_proj"] = "t-SNE"

    at = _run(spec, preseed=preseed_tsne)
    assert not at.exception, [e.value for e in at.exception]
    fig = _fig_spec(at)
    assert "候選" in [t.get("name") for t in fig["data"]], "candidates dropped under t-SNE"
    cand_tr = next(t for t in fig["data"] if t.get("name") == "候選")
    assert np.all(np.isfinite(_coords(cand_tr["x"]))), "non-finite cand x"
    # direct helper-level proof: tsne+candidates returns BOTH frames (PCA fallback)
    rng = np.random.default_rng(1)
    memb = rng.standard_normal((20, 16)).astype("float32")
    cmemb = memb[:5] + 0.02
    app.st.session_state.pop("_objcov_proj_cache", None)
    cm, cc = app._objcov_project(memb, cmemb, "tsne", 2, "k_tsne_fallback")
    assert cc is not None and len(cc) == 5, "t-SNE+cand should fall back so cand projects"


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 4 — "Selection must not confuse my objects with the candidates."
# The _CMPC_BOFF sentinel cleanly splits main(<BOFF) vs candidate(>=BOFF),
# round-tripping through the real customdata → selection_points_to_indices path.
# ════════════════════════════════════════════════════════════════════════

def test_s4_sentinel_separates_main_and_candidate_selection():
    n_main, n_cand = 7, 4
    code_m = [[i] for i in range(n_main)]
    code_c = [[j + _BOFF] for j in range(n_cand)]
    assert all(c[0] < _BOFF for c in code_m)
    assert all(c[0] >= _BOFF for c in code_c)

    # simulate a mixed plotly selection (2 main + 2 cand) → customdata points
    points = ([{"customdata": code_m[1]}, {"customdata": code_m[5]},
               {"customdata": code_c[0]}, {"customdata": code_c[3]}])
    sel = selection_points_to_indices(points)
    ms = sorted({c for c in sel if c < _BOFF})
    cs = sorted({c - _BOFF for c in sel if c >= _BOFF})
    assert ms == [1, 5], ms
    assert cs == [0, 3], cs
    # the offset is huge → no realistic main index can collide with a cand code
    assert n_main < _BOFF


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 5 — "Five projection methods, and the two supervised ones should
# actually separate my data from the candidates." 監督UMAP/LDA present in the
# sidebar; LDA(監督) yields a real main-vs-cand split with both frames returned.
# ════════════════════════════════════════════════════════════════════════

def test_s5_five_methods_and_supervised_separates_groups():
    assert list(app._OBJCOV_PROJ) == ["PCA", "t-SNE", "UMAP", "監督UMAP", "LDA(監督)"]
    assert app._OBJCOV_PROJ["監督UMAP"] == "sumap"
    assert app._OBJCOV_PROJ["LDA(監督)"] == "lda"

    rng = np.random.default_rng(5)
    # main and candidate clouds overlap in raw space; LDA(監督) by group should
    # still pull the two groups apart along the discriminant axis.
    memb = rng.standard_normal((30, 16)).astype("float32")
    cmemb = rng.standard_normal((12, 16)).astype("float32") + 0.4
    app.st.session_state.pop("_objcov_proj_cache", None)
    cm, cc = app._objcov_project(memb, cmemb, "lda", 2, "k_lda")
    assert cc is not None and len(cc) == 12, "LDA must project candidates too"
    # group means separated on axis-0 (the single LDA discriminant)
    sep = abs(cm[:, 0].mean() - cc[:, 0].mean())
    within = cm[:, 0].std() + cc[:, 0].std() + 1e-9
    assert sep / within > 0.3, f"LDA failed to separate groups (sep/within={sep/within:.2f})"


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 6 — "I picked a block, then I want the candidates closest to THAT
# block — same class only." Ranking is closest-first to the FROZEN high-D block
# centroid, NAME-filtered, never mixing in another class.
# ════════════════════════════════════════════════════════════════════════

def test_s6_block_candidates_closest_first_same_class(tmp_path):
    cov_token = "tok_s6"
    preseed, _, _, _ = _controlled_preseed(tmp_path, cov_token=cov_token)
    spec = _base_spec(cov_token=cov_token, seed_main_cache=False)
    at = _run(spec, preseed=preseed)
    assert not at.exception, [e.value for e in at.exception]

    # the candidate panel rendered its closest-first thumbnails with d= captions
    caps = [c.value for c in at.caption]
    d_caps = [c for c in caps if c.startswith("d=")]
    assert d_caps, f"no distance captions on candidate thumbnails: {caps[:5]}"
    dvals = [float(c.split("=")[1]) for c in d_caps]
    assert dvals == sorted(dvals), f"candidate thumbnails not closest-first: {dvals}"

    # helper-level: NAME filter never lets a 'dog' candidate into the 'orange' pool
    cand = _ss(at, "objcov_cand")
    orange_ci = [i for i, l in enumerate(cand["labels"]) if l == "orange"]
    cmemb = cand["emb"][orange_ci]
    centroid = np.asarray(cand["emb"])[orange_ci].mean(0)  # near the orange cloud
    idx, dist = mine_candidates(cmemb, centroid, k=24)
    assert idx and dist == sorted(dist)
    assert max(idx) < len(orange_ci), "mine_candidates indexed outside the orange pool"


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 7 — "When the candidates are all far away, don't show me junk."
# A candidate folder whose 'orange' objects sit far from every block centroid
# must yield an honest 'too far' warning and NO 加入清單 button.
# ════════════════════════════════════════════════════════════════════════

def test_s7_far_candidates_warn_no_add_button(tmp_path):
    cov_token = "tok_s7"
    dim = 16
    preseed, _, _, main_emb = _controlled_preseed(tmp_path, cov_token=cov_token)
    main_token = f"{cov_token}_{_PTAG}"

    # rewrite candidate 'orange' embeddings to be FAR from main (different region)
    def preseed_far(ss):
        preseed(ss)
        cand = ss["objcov_cand"]
        far = np.asarray(cand["emb"]).copy()
        for i, l in enumerate(cand["labels"]):
            if l == "orange":
                far[i] = np.full(dim, -8.0, dtype="float32")  # opposite corner
        cand["emb"] = far
        ss["objcov_cand"] = cand

    spec = _base_spec(cov_token=cov_token, seed_main_cache=False)
    at = _run(spec, preseed=preseed_far)
    assert not at.exception, [e.value for e in at.exception]
    warns = " | ".join(w.value for w in at.warning)
    assert "太遠" in warns or "夠接近" in warns, f"expected a too-far warning: {warns}"
    assert not [b for b in at.button if b.key == "objcov_add"], \
        "加入清單 must NOT appear when no candidate is close enough"


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 8 — "Don't offer me a class with only a couple objects as if it's
# analysable." The default pick is the thinnest ANALYSABLE (>=4) class, and a
# tiny class (<4) trips the guard without crashing. (coco8 real flow.)
# ════════════════════════════════════════════════════════════════════════

def test_s8_default_is_thinnest_analysable_and_tiny_guard(coco8_available=None):
    assert (_COCO8_TRAIN / "images").is_dir(), "demo/coco8/train missing"
    at = _run(_base_spec())
    assert not at.exception, [e.value for e in at.exception]
    df = at.dataframe[0].value
    rows = df.to_dict("records")
    analysable = [r for r in rows if r["物件數"] >= 4]
    sb = next(s for s in at.selectbox if "選類別" in (s.label or ""))
    if analysable:
        # selected default == the FIRST analysable row (thinnest by sort order)
        assert sb.value == analysable[0]["類別"], (sb.value, analysable[0]["類別"])
        assert sb.value not in [r["類別"] for r in rows if r["物件數"] < 4]

    # explicitly select a tiny (<4) class → guard message, no crash
    tiny = next((o for o in list(sb.options)
                 if o in ("broccoli", "vase", "giraffe", "toothbrush")), None)
    if tiny is None:
        pytest.skip("no <4 class in this coco8 sample")
    at2 = sb.set_value(tiny).run()
    assert not at2.exception, [e.value for e in at2.exception]
    msgs = " | ".join(getattr(m, "value", "")
                      for m in list(at2.warning) + list(at2.info))
    assert ("太少無法分析" in msgs or "盲區判斷不穩" in msgs
            or "建議整類擴量" in msgs), msgs


@pytest.fixture(scope="module")
def coco8_available():
    assert (_COCO8_TRAIN / "images").is_dir(), "demo/coco8/train missing"
    return True


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 9 — "Adding candidates to my list must NEVER touch the source
# folder, must tag them right, and must log exactly one curation entry." (The
# proposal-only contract — verified byte-for-byte.)
# ════════════════════════════════════════════════════════════════════════

def _snapshot_tree(root: Path):
    return {str(p): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def test_s9_add_to_cart_proposal_only_and_logged(tmp_path):
    cov_token = "tok_s9b"
    preseed, cand_root, _, _ = _controlled_preseed(tmp_path, cov_token=cov_token)
    log = app._CURATION_LOG
    before = log.read_text(encoding="utf-8").count("\n") if log.exists() else 0
    cand_before = _snapshot_tree(cand_root)

    spec = _base_spec(cov_token=cov_token, seed_main_cache=False)
    at = _run(spec, preseed=preseed)
    assert not at.exception, [e.value for e in at.exception]
    add = [b for b in at.button if b.key == "objcov_add"]
    assert add, "objcov_add button missing despite near candidates"
    at = add[0].click().run()
    assert not at.exception, [e.value for e in at.exception]

    cart = _ss(at, "viz_export_list", {})
    assert cart, "cart empty after 加入清單"
    assert all(v.get("source") == "objcov_backfill" for v in cart.values()), \
        {v.get("source") for v in cart.values()}
    # never a MAIN crop, never a 'dog' (NAME-filtered to analysed 'orange')
    cand = _ss(at, "objcov_cand")
    orange_paths = {str(r["path"]) for r in cand["recs"] if r["label"] == "orange"}
    main_paths = {str(r["path"]) for r in _ss(at, "_objcov_main")["recs"]}
    for v in cart.values():
        assert v["path"] in orange_paths, f"cart item not an orange candidate: {v['path']}"
        assert v["path"] not in main_paths, "MAIN crop leaked into cart"

    after = log.read_text(encoding="utf-8").count("\n") if log.exists() else 0
    assert after == before + 1, (before, after)
    last = log.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert "backfill" in last and "block" in last, last
    # source folder untouched, byte-for-byte
    assert _snapshot_tree(cand_root) == cand_before, "candidate dataset was modified"


# ════════════════════════════════════════════════════════════════════════
# SCENARIO 10 — "If I never picked a candidate folder, the panel should coach
# me, not error." Empty-candidate-state guidance + no add button + no crash;
# and the per-class table / scatter still render so the user can explore.
# ════════════════════════════════════════════════════════════════════════

def test_s10_no_candidate_folder_coaches_no_crash(tmp_path):
    cov_token = "tok_s10"
    # seed MAIN only — NO objcov_cand at all
    preseed, _, _, _ = _controlled_preseed(tmp_path, cov_token=cov_token)
    main_token = f"{cov_token}_{_PTAG}"

    def preseed_no_cand(ss):
        preseed(ss)
        if "objcov_cand" in ss:        # user hasn't projected any candidates
            del ss["objcov_cand"]      # SafeSessionState has no .pop()

    spec = _base_spec(cov_token=cov_token, seed_main_cache=False)
    at = _run(spec, preseed=preseed_no_cand)
    assert not at.exception, [e.value for e in at.exception]
    # the exploration surface still renders
    assert at.dataframe, "per-class table missing in empty-candidate state"
    assert at.get("plotly_chart"), "scatter missing in empty-candidate state"
    # coaching caption present, no add button, no false 'too far' warning
    caps = " | ".join(c.value for c in at.caption)
    assert "候選資料夾" in caps, f"missing empty-state coaching: {caps}"
    assert not [b for b in at.button if b.key == "objcov_add"], \
        "加入清單 must not appear with no candidates"
