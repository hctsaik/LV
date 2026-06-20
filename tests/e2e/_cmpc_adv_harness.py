"""Adversarial AppTest harness for `_render_compare_by_class`.

Unlike _cmpc_harness.py (which seeds a fixed 2-class fixture), this driver reads
a JSON spec from the CMPC_SPEC env var describing exactly the cmpc_* session
state to seed (counts per class per side, embedding dim, degenerate cases, etc.)
plus optional pre-seeded selection / projection-cache / clear-nonce so a single
adversarial scenario can drive a precise edge condition before calling the real
renderer. Any exception inside the renderer surfaces as an AppTest exception.

The spec is intentionally low-level so the *test* owns the adversarial intent;
this file only translates spec → session_state and calls the real fn.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

import app  # noqa: E402


def _seed() -> None:
    if st.session_state.get("cmpc_rows") is not None:
        return
    spec = json.loads(os.environ["CMPC_SPEC"])
    img = spec["img"]  # a real jpg path usable by st.image in the grid
    dim = int(spec.get("dim", 16))
    seed = int(spec.get("seed", 0))
    rng = np.random.default_rng(seed)

    # spec["classes"] = list of {"name", "a", "b"} object counts per side.
    classes = spec["classes"]
    oa, ob, la, lb = [], [], [], []
    for c in classes:
        for _ in range(int(c["a"])):
            oa.append({"path": img, "label": c["name"]})
            la.append(c["name"])
        for _ in range(int(c["b"])):
            ob.append({"path": img, "label": c["name"]})
            lb.append(c["name"])

    ea = rng.standard_normal((len(oa), dim)).astype("float32") if oa else \
        np.zeros((0, dim), "float32")
    eb = rng.standard_normal((len(ob), dim)).astype("float32") if ob else \
        np.zeros((0, dim), "float32")

    # rows = per-shared-class drift table (only classes present on BOTH sides),
    # matching what _compute_compare_by_class would emit.
    shared = sorted({c["name"] for c in classes if c["a"] > 0}
                    & {c["name"] for c in classes if c["b"] > 0})
    rows = []
    for name in shared:
        na = sum(1 for l in la if l == name)
        nb = sum(1 for l in lb if l == name)
        rows.append({"類別": name, "A 數量": na, "B 數量": nb,
                     "漂移↓": round(float(rng.random()) * 0.4, 4)})
    rows.sort(key=lambda r: r["漂移↓"], reverse=True)

    state = {
        "cmpc_oa": oa, "cmpc_ea": ea, "cmpc_ob": ob, "cmpc_eb": eb,
        "cmpc_la": la, "cmpc_lb": lb, "cmpc_rows": rows,
        "cmpc_names": tuple(spec.get("names", ("setA", "setB"))),
        "cmpc_model": spec.get("model", "dinov2_vits14"),
        "cmpc_token": spec.get("token", "deadbeef" * 4),
        "cmp_result_kind": "byclass",
        "cmp_object_policy": spec.get("policy"),
    }
    # optional pre-seeded interaction state for selection/projection scenarios
    if "cmpc_proj_cache" in spec:
        state["_cmpc_proj_cache"] = spec["cmpc_proj_cache"]
    if "cmpc_sel" in spec:
        state["cmpc_sel"] = spec["cmpc_sel"]
    if "clear_nonce" in spec:
        state["_cmpc_clear_nonce"] = spec["clear_nonce"]
    if "cmp_autotune" in spec:
        state["cmp_autotune"] = spec["cmp_autotune"]
    st.session_state.update(state)


_seed()
app._render_compare_by_class()
