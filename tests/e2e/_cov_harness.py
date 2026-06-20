"""AppTest harness for `_render_coverage_view` (嵌入覆蓋圖 / embedding coverage).

Reads a JSON spec from COV_SPEC describing the MAIN dataset plus optional
pre-projected candidate folders, seeds the session the way `_completeness_ui`
would after pressing ▶ 開始分析, then calls the REAL renderer from scripts/app.py.
Any exception inside the renderer surfaces as an AppTest exception the test asserts.

Design (fast + deterministic, no real DINOv2 weights):
- `app.load_model` is monkeypatched to a deterministic fake embedder, so the
  object-crop path (_crop_and_embed_objects) and candidate projection run
  end-to-end without weights but stay reproducible. ``cand_offset`` shifts a
  candidate folder toward/away from the MAIN cloud.
- MAIN whole-image emb is synthesized (image-level path); when the spec uses
  object level, the renderer crops real coco8 bboxes (tiny) and embeds via the
  fake embedder — exercising the true is_obj branch.

Spec keys:
  main_dir          (str)   dataset root, e.g. demo/coco8/train  [required]
  cov_token         (str)   stable token (default "covtok")
  dim               (int)   fake embedder width (default 16)
  granularity       (str)   "物件級（YOLO）" | "整張影像"  (sets cov_granularity)
  class_filter      (str)   value for cov_class_filter selectbox (optional)
  proj_method       (str)   cov_proj_method, e.g. "PCA"/"監督UMAP" (default PCA)
  proj_dim          (str)   "2D"|"3D"  (sets cov_proj_dim)
  color_mode        (str)   "類別（與縮圖同色）"|"稀疏度" (sets cov_color_mode)
  cand_folders      ([str]) candidate folders to project (writes cov_cand_text)
  preembed_cand     (bool)  call _cov_embed_candidates before render
  cand_offset       (float) push candidate cloud (fake embedder offset)
  b_role            (str)   "採礦池"|"參照分佈" (sets cov_b_role)
  sel_indices       ([int]) pre-seed a box-selection of MAIN points (cov_sel)
  sel_cand          ([int]) pre-seed a box-selection of candidate diamonds
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

import app  # noqa: E402

# Capture the TRUE pristine st.plotly_chart exactly once per process and stash it
# on the streamlit module so a later fake-selection patch can't overwrite it.
# (This harness script re-executes for every test, so a plain module-global would
# re-grab whatever a previous test left PATCHED — the leak that made a fake
# selection bleed across tests.) Restored at the start of each _seed.
if not hasattr(app.st, "_lv_pristine_plotly_chart"):
    app.st._lv_pristine_plotly_chart = app.st.plotly_chart


def _fake_embedder_factory(dim: int, offset: float):
    """Deterministic embed_fn(path_or_PIL) -> vector. PIL crops hashed by pixel
    bytes so the same bbox maps to the same vector; ``offset`` shifts a folder."""
    def embed_fn(x):
        try:
            if isinstance(x, Image.Image):
                buf = io.BytesIO()
                x.convert("RGB").resize((8, 8)).save(buf, format="PNG")
                h = hashlib.sha1(buf.getvalue()).hexdigest()
            else:
                h = hashlib.sha1(str(x).encode("utf-8")).hexdigest()
        except Exception:
            h = hashlib.sha1(str(x).encode("utf-8")).hexdigest()
        rng = np.random.default_rng(int(h[:8], 16))
        return rng.standard_normal(dim).astype("float32") * 0.5 + offset
    return embed_fn


def _seed() -> None:
    if st.session_state.get("_cov_seeded"):
        return
    spec = json.loads(os.environ["COV_SPEC"])
    st.session_state["_cov_seeded"] = True

    dim = int(spec.get("dim", 16))
    model = spec.get("model", "dinov2_vits14")
    st.session_state["_cov_model"] = model

    # Model loader. By default use the REAL model so MAIN (which may reuse an
    # on-disk 384-dim crop cache) and freshly-embedded candidates share the same
    # dimensionality — a patched fake only on one side causes a 16-vs-384 NN
    # mismatch. Set spec["fake_model"]=true to force the deterministic fake
    # (only safe when no on-disk MAIN crop cache exists / dims agree).
    if spec.get("fake_model"):
        app.load_model = lambda *a, **k: _fake_embedder_factory(
            dim, float(spec.get("cand_offset", 0.0)))

    from visualize_embeddings import discover_images, discover_images_classifier
    main_dir = Path(spec["main_dir"])
    recs = discover_images_classifier([main_dir]) or discover_images([main_dir])
    st.session_state["cov_records"] = recs
    st.session_state["cov_token"] = spec.get("cov_token", "covtok")

    # Whole-image embeddings (image-level path / object level re-crops on top).
    rng = np.random.default_rng(int(spec.get("main_seed", 7)))
    labels = sorted({r.get("label", "") for r in recs})
    centers = {c: rng.standard_normal(dim).astype("float32") * 2.0 for c in labels}
    emb = np.zeros((len(recs), dim), dtype="float32")
    for i, r in enumerate(recs):
        emb[i] = (centers.get(r.get("label", ""), np.zeros(dim, "float32"))
                  + rng.standard_normal(dim).astype("float32") * 0.3)
    st.session_state["cov_emb"] = emb

    # objcov_policy drives the auto pad used by _crop_and_embed_objects.
    import object_eval as oe
    st.session_state["objcov_policy"] = dict(oe.DEFAULT_POLICY)

    # Sidebar selections (the renderer reads these as session keys).
    if spec.get("granularity"):
        st.session_state["cov_granularity"] = spec["granularity"]
    st.session_state["cov_proj_method"] = spec.get("proj_method", "PCA")
    if spec.get("proj_dim"):
        st.session_state["cov_proj_dim"] = spec["proj_dim"]
    if spec.get("color_mode"):
        st.session_state["cov_color_mode"] = spec["color_mode"]
    if spec.get("class_filter"):
        st.session_state["cov_class_filter"] = spec["class_filter"]
    if spec.get("b_role"):
        st.session_state["cov_b_role"] = spec["b_role"]

    # Candidate folders.
    if spec.get("cand_folders"):
        st.session_state["cov_cand_text"] = "\n".join(str(p) for p in spec["cand_folders"])
    if spec.get("preembed_cand") and spec.get("cand_folders"):
        app._cov_embed_candidates(model)

    # Synthetic plotly selection: AppTest can't fire a real drag, so wrap
    # st.plotly_chart to return an event selecting the given candidate diamonds
    # (by candidate index → sentinel customdata i + _CMPC_BOFF) and/or main
    # points. This exercises the REAL decode + gallery branch in the renderer.
    # Reset to the pristine chart first → a fake-selection patch from a previous
    # test in this process cannot bleed into this render (test order independence).
    app.st.plotly_chart = app.st._lv_pristine_plotly_chart
    fcs = spec.get("fake_cand_selection")
    fms = spec.get("fake_main_selection")
    if fcs is not None or fms is not None:
        _orig_chart = app.st._lv_pristine_plotly_chart

        def _patched_chart(*a, **k):
            real = _orig_chart(*a, **k)
            if not k.get("on_select"):
                return real
            pts = []
            for i in (fms or []):
                pts.append({"customdata": [int(i)]})
            for i in (fcs or []):
                pts.append({"customdata": [int(i) + app._CMPC_BOFF]})
            return {"selection": {"points": pts}}
        app.st.plotly_chart = _patched_chart

    # Pre-seed a selection (AppTest can't fire plotly drag; we set cov_sel the
    # exact shape the decode produces, so the gallery/cart branches render).
    if spec.get("sel_indices") is not None or spec.get("sel_cand") is not None:
        # token must match what the renderer derives; for image level it's
        # cov_token, for object level it's the freshly-minted obj token, so we
        # leave token mismatch to be reset by the renderer UNLESS image level.
        st.session_state["_cov_pre_sel"] = {
            "indices": list(spec.get("sel_indices") or []),
            "cand": list(spec.get("sel_cand") or []),
        }


def _install_cand_capture() -> None:
    """Capture the EXACT cand_records the renderer hands to the figure builder
    (after the _cov_cls candidate filter at app.py:4331-4334), so a test can
    assert no foreign-class candidate reaches the overlay/gallery. Writes the
    captured labels to session key '_cap_cov_cand_labels'."""
    if not os.environ.get("COV_CAPTURE"):
        return
    _orig = app._build_cov_scatter

    def _wrapped(coords_d, sparsity, records, dim, coords_c, cand_records,
                 ranked_idx, ranked_scores, **kw):
        st.session_state["_cap_cov_cand_labels"] = [
            (r.get("label", "") if isinstance(r, dict) else "")
            for r in (cand_records or [])]
        st.session_state["_cap_cov_cand_n"] = (
            0 if coords_c is None else len(coords_c))
        return _orig(coords_d, sparsity, records, dim, coords_c, cand_records,
                     ranked_idx, ranked_scores, **kw)
    app._build_cov_scatter = _wrapped


_install_cand_capture()
_seed()
# If a pre-selection was requested at IMAGE level, set cov_sel with the token the
# renderer will use (cov_token) so it survives the token check.
_pre = st.session_state.get("_cov_pre_sel")
if _pre and st.session_state.get("cov_granularity", "整張影像") == "整張影像":
    st.session_state["cov_sel"] = {
        "token": st.session_state["cov_token"],
        "indices": _pre["indices"], "cand": _pre["cand"]}

app._render_coverage_view(
    st.session_state["cov_records"],
    st.session_state["cov_emb"],
    st.session_state.get("_cov_model", "dinov2_vits14"))
