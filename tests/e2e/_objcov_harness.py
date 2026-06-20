"""AppTest harness for `_render_objcov_view` (按物件・逐類別補洞).

Reads a JSON spec from OBJCOV_SPEC describing the MAIN detection dataset plus
optional pre-seeded candidate folders, then calls the real renderer from
scripts/app.py. Any exception inside the renderer surfaces as an AppTest
exception the test can assert against.

Design (fast + deterministic, no real DINOv2 weights):
- When the spec asks (seed_main_cache, default True) we seed the `_objcov_main`
  session cache directly with a token matching what the renderer derives, so
  `_crop_and_embed_objects` is never called for MAIN. MAIN object embeddings are
  generated deterministically, clustered per class so KMeans blocks form.
- `app.load_model` is monkeypatched to a deterministic fake embedder, so the
  CANDIDATE embedding path runs end-to-end without real weights but stays
  reproducible. ``cand_offset`` shifts candidate vectors toward/away from MAIN.

The spec owns the adversarial intent; this file only translates spec -> session.
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
import object_eval as oe  # noqa: E402


def _fake_embedder_factory(dim: int, offset: float):
    """Deterministic embed_fn(path_or_PIL) -> vector. PIL crops are hashed by
    pixel bytes so the same bbox always maps to the same vector; ``offset`` pushes
    a candidate folder near/far from the MAIN cloud (centered ~0 in tests)."""
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
        return rng.standard_normal(dim).astype("float32") * 0.05 + offset
    return embed_fn


def _seed() -> None:
    if st.session_state.get("_objcov_seeded"):
        return
    spec = json.loads(os.environ["OBJCOV_SPEC"])
    st.session_state["_objcov_seeded"] = True

    dim = int(spec.get("dim", 16))
    model = spec.get("model", "dinov2_vits14")
    st.session_state["_objcov_model"] = model

    # Patch model loader used by the candidate embedding path.
    app.load_model = lambda *a, **k: _fake_embedder_factory(
        dim, float(spec.get("cand_offset", 0.0)))

    main_dir = Path(spec["main_dir"])
    from visualize_embeddings import discover_images, discover_images_classifier
    recs = discover_images_classifier([main_dir]) or discover_images([main_dir])
    st.session_state["cov_records"] = recs
    st.session_state["cov_token"] = spec.get("cov_token", "tok_main_0001")
    st.session_state["objcov_policy"] = dict(oe.DEFAULT_POLICY)

    if spec.get("seed_main_cache", True) and recs and app._is_detection_dataset(recs):
        class_names = app._cov_class_names(recs)
        from interaction import discover_yolo_objects
        meta = discover_yolo_objects([Path(r["path"]) for r in recs], class_names)
        if meta:
            orecs = [{
                "path": o["image_path"], "image_path": o["image_path"],
                "split": "", "label": o["label"], "class_id": o["class_id"],
                "bbox": o["bbox"], "obj_index": o["obj_index"],
            } for o in meta]
            rng = np.random.default_rng(int(spec.get("main_seed", 7)))
            labels = sorted({r["label"] for r in orecs})
            centers = {c: rng.standard_normal(dim).astype("float32") * 2.0
                       for c in labels}
            emb = np.zeros((len(orecs), dim), dtype="float32")
            for i, r in enumerate(orecs):
                emb[i] = centers[r["label"]] + rng.standard_normal(dim).astype("float32") * 0.3
            main_token = (f"{st.session_state['cov_token']}_"
                          f"{oe.policy_tag(st.session_state['objcov_policy'])}")
            st.session_state["_objcov_main"] = {
                "token": main_token, "recs": orecs, "emb": emb,
                "labels": [r["label"] for r in orecs]}

    if spec.get("cand_folders"):
        st.session_state["objcov_cand_list"] = [str(p) for p in spec["cand_folders"]]
    if spec.get("preembed_cand") and spec.get("cand_folders"):
        app._objcov_embed_candidates(
            [Path(p) for p in spec["cand_folders"]], model,
            st.session_state["objcov_policy"], app._cov_class_names(recs))

    # Positive-mine path: synthesize candidate object embeddings that sit RIGHT
    # ON the MAIN cloud of `near_class`, so mine_candidates returns hits and the
    # rendered "🛒 加入清單" button is reachable (exercises the real button wiring,
    # not just the helpers). Candidate crops reuse a real on-disk image so the UI
    # thumbnails render. Requires seed_main_cache.
    if spec.get("cand_from_main_class") and st.session_state.get("_objcov_main"):
        nc = spec["cand_from_main_class"]
        mc = st.session_state["_objcov_main"]
        idxs = [i for i, l in enumerate(mc["labels"]) if l == nc]
        if idxs:
            base = mc["emb"][idxs]
            rng = np.random.default_rng(99)
            # jitter EXACTLY around the main points (same scale) → cosine-close
            cemb = (base[rng.integers(0, len(base), size=min(8, len(base)))]
                    + rng.standard_normal((min(8, len(base)), base.shape[1])) * 0.02
                    ).astype("float32")
            timg = mc["recs"][idxs[0]]["path"]
            crecs = [{"path": timg, "image_path": timg, "split": "",
                      "label": nc, "class_id": 0, "bbox": (0.5, 0.5, 0.3, 0.3),
                      "obj_index": j} for j in range(len(cemb))]
            import uuid as _uuid
            st.session_state["objcov_cand"] = {
                "token": _uuid.uuid4().hex, "recs": crecs, "emb": cemb,
                "labels": [nc] * len(cemb)}


def _install_cand_capture() -> None:
    """Capture the candidate overlay trace the renderer actually draws (after the
    cand_ci class filter at app.py:4846). The candidate trace is named '候選' and
    its text is [f'候選 #{i}' for i in cand_ci]; those i index the FULL unfiltered
    cand['labels'], so we resolve each back to its class and stash the labels in
    session key '_cap_objcov_cand_labels'. Also captures the count of candidate
    points actually plotted ('_cap_objcov_cand_n')."""
    if not os.environ.get("OBJCOV_CAPTURE"):
        return
    import plotly.graph_objects as go
    _orig_add = go.Figure.add_trace

    def _wrapped(self, trace, *a, **k):
        try:
            if getattr(trace, "name", None) == "候選":
                txt = list(getattr(trace, "text", None) or [])
                cand = st.session_state.get("objcov_cand") or {}
                full_labels = cand.get("labels") or []
                idxs = []
                for t in txt:
                    # t == "候選 #<i>"
                    try:
                        idxs.append(int(str(t).split("#")[-1]))
                    except Exception:
                        pass
                st.session_state["_cap_objcov_cand_labels"] = [
                    full_labels[i] for i in idxs if 0 <= i < len(full_labels)]
                st.session_state["_cap_objcov_cand_n"] = len(idxs)
        except Exception:
            pass
        return _orig_add(self, trace, *a, **k)
    go.Figure.add_trace = _wrapped


_install_cand_capture()
_seed()
app._render_objcov_view(st.session_state["cov_records"],
                        st.session_state.get("_objcov_model", "dinov2_vits14"))
