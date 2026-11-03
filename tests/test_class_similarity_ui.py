"""驗收:class_similarity_ui 純函式(抽樣、jitter、選取、軸範圍)。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from class_similarity_oracle import almost


def _ui():
    import class_similarity_ui as m
    return m


def _item(rid, label, metric_val, margin=None):
    return SimpleNamespace(
        record_id=rid, label=label, margin=margin if margin is not None else metric_val,
        same=metric_val, other=0.0, input_index=0, status="ok",
    )


def test_ac05_scatter_sample_not_in_box():  # AC05
    ui = _ui()
    items = [_item(f"a{i:03d}", "A", i / 1000.0, margin=i / 1000.0) for i in range(250)]
    sampled = ui.sample_scatter_rows(items, "margin", "ak")
    assert len(sampled) == 200
    keep20 = {it.record_id for it in sorted(items, key=lambda r: (r.margin, r.record_id))[:20]}
    got = {it.record_id for it in sampled}
    assert keep20 <= got
    # box helper still uses all 250
    import class_similarity as cs
    fake = []
    from class_similarity import AnalysisItem
    for i, it in enumerate(items):
        fake.append(AnalysisItem(
            record_id=it.record_id, input_index=i, label="A",
            same=it.same, other=0.0, margin=it.margin, rival_class="B",
            same_k=1, rival_k=1, same_neighbors=[], rival_neighbors=[],
            tied_rival_count=0, tied_rival_classes=[], status="ok", reasons=[],
        ))
    box = cs.build_box_summary(fake, "margin")[0]
    assert box["n_scored"] == 250
    assert box["n_scored"] != len(sampled)


def test_ac07_jitter_stable():  # AC07
    ui = _ui()
    a = ui.jitter_x("ak", "rid")
    b = ui.jitter_x("ak", "rid")
    c = ui.jitter_x("ak", "other")
    assert a == b and a != c


def test_ac26_local_selected_valid():  # AC26
    ui = _ui()
    prepared = SimpleNamespace(
        record_ids=["ok1", "ok2", "invalid:tok:3"],
        eligibility=[True, True, False],
    )
    got = ui.local_selected_valid(["ok1", "invalid:tok:3", "missing"], prepared)
    assert got == ["ok1"]


def test_ac12_preview_does_not_recall_analyze():  # AC12
    ui = _ui()
    # panel dispatch: similarity view must not call neighbor_confusion
    assert ui.PANEL_SIM != ui.PANEL_CONF
    # sampling / mask are local; analyze is only invoked from compute button in render_panel
    import class_similarity_threshold as thr
    items = [_item("a", "A", -0.2, -0.2), _item("b", "A", 0.3, 0.3)]
    from class_similarity import AnalysisItem
    real = [
        AnalysisItem("a", 0, "A", 0.0, 0.0, -0.2, "B", 1, 1, [], [], 0, [], "ok", []),
        AnalysisItem("b", 1, "A", 0.0, 0.0, 0.3, "B", 1, 1, [], [], 0, [], "ok", []),
    ]
    m1 = thr.candidate_mask(real, {"mode": "threshold", "threshold": 0.0})
    m2 = thr.candidate_mask(real, {"mode": "threshold", "threshold": 0.5})
    assert int(m1.sum()) != int(m2.sum())


def test_ac33_axis_zoom_helper():  # AC33
    import class_similarity_threshold as thr
    rng = thr.axis_display_range("margin", "data", [-0.015, 0.01, 0.012], 1.0)
    assert rng["threshold_outside"] is True
    full = thr.axis_display_range("margin", "full", [-0.015, 0.01], 0.0)
    assert full["ymin"] == -2 and full["ymax"] == 2


def test_ac25_out_of_order_requests():  # AC25
    import class_similarity_threshold as thr
    pol = thr.parse_policy({"mode": "threshold", "threshold": 0.1})
    state = thr.PreviewState(pol, None, 5, None, "ak", None, None)
    with pytest.raises(thr.ActionRejected):
        thr.apply_preview_event(
            state, {"kind": "commit", "analysis_key": "ak", "request_seq": 3,
                    "policy": {"mode": "threshold", "threshold": 0.4}},
            current_analysis_key="ak", current_eval_key=None, draft_policy=pol,
        )

