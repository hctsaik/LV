"""E2E for 08 pool_registry GUI wiring (AC-G1..G3).

Real Streamlit server with LV_CACHE_DIR/LV_OUTPUT_DIR/CIM_LOG_DIR isolated to
temp dirs (al_isolated_server), driven through the real Visualize→cart→匯出 send
path. Each test asserts on BOTH the on-disk registry (deterministic ground truth)
and the rendered blocked/override UI. Design: 3_Architect_Design/08_pool_registry.md.

Not in the PG auto-green loop (@pytest.mark.e2e; gate runs -m "not e2e").
"""
from __future__ import annotations

import pytest
from playwright.sync_api import expect

from .conftest import load_app, newest_handoff_dir, wait_idle, write_annotation_sidecars
from ._al_loop_harness import (apply_readback_click, blocked_caption, cart_send_count, goto_cart,
                               goto_readback, read_jsonl, run_viz, sample_to_cart,
                               send_cart, wait_cart_blocked)

pytestmark = pytest.mark.e2e
expect.set_options(timeout=15000)


def test_second_send_blocked_e2e(al_isolated_server, page, synthetic_dataset):
    """AC-G1: send a cart once, then the same cart re-renders as 「已送標」 with an
    empty send set (button disabled / （0）)."""
    srv = al_isolated_server
    load_app(page, srv.base_url)
    run_viz(page, synthetic_dataset)
    sample_to_cart(page)
    goto_cart(page)
    n = cart_send_count(page)
    assert n >= 2, f"need a cart of >=2 to exercise a batch, got {n}"

    send_cart(page)

    # ground truth: the pool registry recorded the n distinct images as 'sent'
    sent_shas = {e["sha256"] for e in read_jsonl(srv.registry_path)
                 if e.get("event") == "sent"}
    assert len(sent_shas) == n, f"expected {n} distinct sent sha, got {len(sent_shas)}"

    # the same cart is now intercepted on re-render: send fully blocked
    wait_cart_blocked(page)
    expect(blocked_caption(page)).to_be_visible()


def test_include_handled_override_e2e(al_isolated_server, page, synthetic_dataset):
    """AC-G2: ticking 「仍包含已送標」 re-enables the send with count == original."""
    srv = al_isolated_server
    load_app(page, srv.base_url)
    run_viz(page, synthetic_dataset)
    sample_to_cart(page)
    goto_cart(page)
    n = cart_send_count(page)
    assert n >= 2

    send_cart(page)                          # first send → all n now 'sent'
    wait_cart_blocked(page)
    expect(blocked_caption(page)).to_be_visible()

    # tick the override → send set is the full original batch again
    page.get_by_text("仍包含已送標／已標的樣本").first.click()
    wait_idle(page)
    assert cart_send_count(page) == n, "override count must equal the original batch"
    send_cart(page)                          # click the re-enabled send
    wait_idle(page, timeout=60000)
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)


def test_readback_marks_labeled_e2e(al_isolated_server, page, synthetic_dataset):
    """AC-G3: simulate annotation sidecars → apply read-back → registry marks
    those sha 'labeled' (read straight off disk)."""
    srv = al_isolated_server
    load_app(page, srv.base_url)
    run_viz(page, synthetic_dataset)
    sample_to_cart(page)
    goto_cart(page)
    n = cart_send_count(page)
    assert n >= 2
    send_cart(page)

    # simulate Labeling finishing every image with a (different) class
    handoff = newest_handoff_dir(srv)
    n_ann = write_annotation_sidecars(handoff, "classB")
    assert n_ann == n

    goto_readback(page)
    apply_readback_click(page)

    labeled = [e for e in read_jsonl(srv.registry_path) if e.get("event") == "labeled"]
    labeled_shas = {e["sha256"] for e in labeled}
    assert len(labeled_shas) == n, \
        f"read-back must mark {n} sha labeled, got {len(labeled_shas)}"
