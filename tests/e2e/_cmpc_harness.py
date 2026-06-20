"""Streamlit harness driven by AppTest to render ONLY `_render_compare_by_class`
with seeded `cmpc_*` session state — no autotune / crop+embed (too slow for CI).

The test (test_compare_by_class.py) sets up session_state via AppTest before the
first run; this script reads those seeded keys and calls the real renderer from
scripts/app.py. Any exception inside the renderer surfaces as an AppTest
exception, which the test asserts against.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `import app` resolve to scripts/app.py regardless of CWD.
_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

import app  # noqa: E402


def _seed_if_needed() -> None:
    """Populate cmpc_* keys from a tiny on-disk fixture if not already present.

    The fixture dir is passed via the CMPC_FIXTURE env var (a folder holding a
    few jpgs). Group A = first half, group B = second half; two fake classes.
    """
    if st.session_state.get("cmpc_rows"):
        return
    fixture = Path(os.environ["CMPC_FIXTURE"])
    jpgs = sorted(fixture.glob("*.jpg"))
    assert len(jpgs) >= 6, f"need >=6 jpgs in {fixture}, found {len(jpgs)}"
    half = len(jpgs) // 2
    a_paths, b_paths = jpgs[:half], jpgs[half:]

    # two classes, interleaved so each class spans both groups
    def _labels(paths):
        return ["cat" if i % 2 == 0 else "dog" for i in range(len(paths))]

    la, lb = _labels(a_paths), _labels(b_paths)
    oa = [{"path": str(p), "label": la[i]} for i, p in enumerate(a_paths)]
    ob = [{"path": str(p), "label": lb[i]} for i, p in enumerate(b_paths)]
    rng = np.random.default_rng(0)
    dim = 16
    ea = rng.standard_normal((len(oa), dim)).astype("float32")
    eb = rng.standard_normal((len(ob), dim)).astype("float32")

    rows = []
    for c in ("cat", "dog"):
        ia = [i for i, l in enumerate(la) if l == c]
        ib = [i for i, l in enumerate(lb) if l == c]
        rows.append({"類別": c, "A 數量": len(ia), "B 數量": len(ib), "漂移↓": 0.1})

    st.session_state.update({
        "cmpc_oa": oa, "cmpc_ea": ea, "cmpc_ob": ob, "cmpc_eb": eb,
        "cmpc_la": la, "cmpc_lb": lb, "cmpc_rows": rows,
        "cmpc_names": ("setA", "setB"), "cmpc_model": "dinov2_vits14",
        "cmpc_token": "deadbeef" * 4, "cmp_result_kind": "byclass",
        "cmp_object_policy": None,  # skip the policy caption branch
    })


_seed_if_needed()
app._render_compare_by_class()
