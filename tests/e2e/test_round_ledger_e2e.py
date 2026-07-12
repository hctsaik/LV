"""E2E for 09 round_ledger GUI wiring (AC-G1..G4).

Real Streamlit server with isolated durable roots (al_isolated_server), driven
through the real Visualize→cart→匯出 send → 📥 timeline path. Asserts on both
rounds.jsonl (ground truth) and the rendered 「🔁 回合」 cards / learning curve.
Design: 3_Architect_Design/09_round_ledger.md.
"""
from __future__ import annotations

import json
import re

import pytest
from playwright.sync_api import expect

from .conftest import (load_app, newest_handoff_dir, wait_idle,
                       write_annotation_sidecars)
from ._al_loop_harness import (apply_readback_click, cart_send_count, goto_cart, goto_readback,
                               read_jsonl, run_viz, sample_to_cart, send_cart)

pytestmark = pytest.mark.e2e
expect.set_options(timeout=15000)


def test_send_opens_round_e2e(al_isolated_server, page, synthetic_dataset):
    """AC-G1: a send opens a round; 📥「🔁 回合」 shows a card with 送出 N."""
    srv = al_isolated_server
    load_app(page, srv.base_url)
    run_viz(page, synthetic_dataset)
    sample_to_cart(page)
    goto_cart(page)
    n = cart_send_count(page)
    assert n >= 2
    send_cart(page)

    rounds = read_jsonl(srv.rounds_path)
    starts = [r for r in rounds if r.get("kind") == "start"]
    batches = [r for r in rounds if r.get("kind") == "batch"]
    assert len(starts) == 1, f"send must open exactly one round: {rounds}"
    assert sum(b.get("n_sent", 0) for b in batches) == n

    goto_readback(page)
    expect(page.get_by_text(re.compile(r"送出\s*%d" % n)).first).to_be_visible()
    # the cart send path carries strategy 'cart' onto the round card
    expect(page.get_by_text(re.compile("策略：cart")).first).to_be_visible()


def test_readback_into_round_e2e(al_isolated_server, page, synthetic_dataset):
    """AC-G2: applying read-back records it onto the same round card (回讀 M)."""
    srv = al_isolated_server
    load_app(page, srv.base_url)
    run_viz(page, synthetic_dataset)
    sample_to_cart(page)
    goto_cart(page)
    n = cart_send_count(page)
    assert n >= 2
    send_cart(page)

    handoff = newest_handoff_dir(srv)
    write_annotation_sidecars(handoff, "classB")

    goto_readback(page)
    apply_readback_click(page)

    # ground truth: a readback event landed on the round with n_labeled == n
    readbacks = [r for r in read_jsonl(srv.rounds_path) if r.get("kind") == "readback"]
    assert readbacks, "read-back must append a readback event to the ledger"
    assert readbacks[-1].get("n_labeled") == n
    assert readbacks[-1].get("n_changed", 0) >= 1

    expect(page.get_by_text(re.compile(r"回讀\s*%d" % n)).first).to_be_visible()


def test_round_survives_restart_e2e(al_isolated_server, page, synthetic_dataset):
    """AC-G3: the round card survives a full server restart (read from disk)."""
    srv = al_isolated_server
    load_app(page, srv.base_url)
    run_viz(page, synthetic_dataset)
    sample_to_cart(page)
    goto_cart(page)
    n = cart_send_count(page)
    assert n >= 2
    send_cart(page)
    assert [r for r in read_jsonl(srv.rounds_path) if r.get("kind") == "start"]

    srv.restart()                       # new process, SAME isolated LV_OUTPUT_DIR
    load_app(page, srv.base_url)
    goto_readback(page)
    expect(page.get_by_text(re.compile(r"送出\s*%d" % n)).first).to_be_visible()


def test_learning_curve_renders_e2e(al_isolated_server, page, synthetic_dataset):
    """AC-G4: two rounds carrying the same metric render a learning curve; the
    「➕ 掛指標」 form appends a metric event (attach_metric wiring)."""
    srv = al_isolated_server
    # seed two rounds each with an 'acc' metric directly into the isolated ledger
    seed = [
        {"kind": "start", "round_id": "rSEED1", "ts": "2026-07-12T00:00:01+00:00",
         "strategy": "cart", "params": {}, "tool": "cart", "note": ""},
        {"kind": "batch", "round_id": "rSEED1", "batch_id": "b1", "n_sent": 5},
        {"kind": "readback", "round_id": "rSEED1", "batch_id": "b1",
         "n_labeled": 5, "n_changed": 2},
        {"kind": "metric", "round_id": "rSEED1", "name": "acc", "value": 0.5,
         "source": "manual", "ts": "2026-07-12T00:00:01+00:00"},
        {"kind": "start", "round_id": "rSEED2", "ts": "2026-07-12T00:00:02+00:00",
         "strategy": "cart", "params": {}, "tool": "cart", "note": ""},
        {"kind": "batch", "round_id": "rSEED2", "batch_id": "b2", "n_sent": 5},
        {"kind": "readback", "round_id": "rSEED2", "batch_id": "b2",
         "n_labeled": 5, "n_changed": 1},
        {"kind": "metric", "round_id": "rSEED2", "name": "acc", "value": 0.8,
         "source": "manual", "ts": "2026-07-12T00:00:02+00:00"},
    ]
    srv.rounds_path.parent.mkdir(parents=True, exist_ok=True)
    srv.rounds_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in seed) + "\n",
        encoding="utf-8")

    load_app(page, srv.base_url)
    goto_readback(page)

    # the learning curve renders (>=2 rounds share metric 'acc')
    expect(page.locator('.st-key-al_learning_curve')).to_be_visible()

    # the per-round metric form appends a metric event (attach_metric wiring).
    # Blur each field (Tab) so Streamlit commits the widget value before the
    # button's rerun reads it — a bare fill sets the DOM value but doesn't fire the
    # change event. Verify against the ledger and retry: the fill/commit/click
    # sequence is timing-sensitive under load.
    def _acc_0_9() -> bool:
        return any(abs(float(m.get("value", 0)) - 0.9) < 1e-9
                   for m in read_jsonl(srv.rounds_path)
                   if m.get("kind") == "metric" and m.get("round_id") == "rSEED2")

    for _ in range(4):
        name = page.locator('.st-key-al_metric_name_rSEED2 input')
        name.click()
        name.fill("acc")
        name.press("Tab")
        wait_idle(page)
        val = page.locator('.st-key-al_metric_val_rSEED2 input')
        val.click()
        val.fill("0.9")
        val.press("Tab")
        wait_idle(page)
        page.locator('.st-key-al_metric_add_rSEED2 button').click()
        wait_idle(page, timeout=60000)
        if _acc_0_9():
            break
    assert _acc_0_9(), (
        "metric form must append acc=0.9 to rSEED2: "
        f"{[m for m in read_jsonl(srv.rounds_path) if m.get('kind') == 'metric']}")
