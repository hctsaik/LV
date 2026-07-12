"""E2E for 14 readback_store GUI wiring (AC-G1..G3).

Read-back corrections must (G1) survive a full server restart, (G2) never touch
the user's dataset, and (G3) be auditable + exportable. Drives the real
cart→send→sidecar→apply-readback path with isolated durable roots, asserting on
both the on-disk overlay/changelog and the 📥 audit view.
Design: 3_Architect_Design/14_readback_store.md.

The apply_overlay transformation itself (latest-wins, no-op-not-counted) is pinned
by the unit tests (tests/test_readback_store.py AC3/AC8); here we prove the
integration: recorded on read-back, persisted across restart, dataset untouched.
"""
from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

_CORRECTED = re.compile("已修正")

from .conftest import (load_app, newest_handoff_dir, wait_idle,
                       write_annotation_sidecars)
from ._al_loop_harness import (apply_readback_click, cart_send_count, goto_cart,
                               goto_readback, read_jsonl, run_viz, sample_to_cart,
                               send_cart)

pytestmark = pytest.mark.e2e
expect.set_options(timeout=15000)


def _dataset_snapshot(folder) -> dict:
    """map of every file under the dataset -> size, to prove it is untouched."""
    from pathlib import Path
    return {str(p.relative_to(folder)): p.stat().st_size
            for p in Path(folder).rglob("*") if p.is_file()}


def _do_readback(srv, page, synthetic_dataset) -> int:
    """Shared prelude: send a cart, annotate it all as classB, apply read-back.
    Returns the changelog event count recorded."""
    load_app(page, srv.base_url)
    run_viz(page, synthetic_dataset)
    sample_to_cart(page)
    goto_cart(page)
    n = cart_send_count(page)
    assert n >= 2
    send_cart(page)
    write_annotation_sidecars(newest_handoff_dir(srv), "classB")
    goto_readback(page)
    apply_readback_click(page)
    return n


def test_readback_survives_restart_e2e(al_isolated_server, page, synthetic_dataset):
    """AC-G1: read-back corrections are recorded and survive a server restart."""
    srv = al_isolated_server
    _do_readback(srv, page, synthetic_dataset)

    changelog = srv.cache_dir / "readback_changelog.jsonl"
    events = read_jsonl(changelog)
    assert events, "read-back must append corrections to the overlay changelog"
    assert all(e.get("sha256") and e.get("new_label") == "classB" for e in events)

    srv.restart()                          # new process, SAME isolated LV_CACHE_DIR
    load_app(page, srv.base_url)
    # re-Run applies the overlay on load (no crash); the audit view reads the
    # persisted changelog and still shows the corrections after the restart.
    run_viz(page, synthetic_dataset)
    goto_readback(page)
    expect(page.get_by_text(_CORRECTED).first).to_be_visible()
    assert read_jsonl(changelog), "changelog must persist across restart"


def test_readback_no_dataset_write_e2e(al_isolated_server, page, synthetic_dataset):
    """AC-G2: applying read-back never writes the user's dataset folder; the
    overlay lives entirely under LV_CACHE_DIR."""
    srv = al_isolated_server
    before = _dataset_snapshot(synthetic_dataset)
    _do_readback(srv, page, synthetic_dataset)
    after = _dataset_snapshot(synthetic_dataset)

    assert after == before, "no-dataset-writes: the dataset folder must be untouched"
    assert (srv.cache_dir / "readback_changelog.jsonl").exists(), \
        "the correction overlay must land in .lv_cache"


def test_readback_audit_export_e2e(al_isolated_server, page, synthetic_dataset):
    """AC-G3: the 📥 audit shows the corrections and exports a version CSV."""
    srv = al_isolated_server
    _do_readback(srv, page, synthetic_dataset)

    # audit section visible with a correction count
    expect(page.get_by_text(_CORRECTED).first).to_be_visible()
    # click the export button, verifying the CSV lands (retry — the click→rerun
    # can miss on timing, same as the send/apply buttons).
    versions: list = []
    for _ in range(4):
        page.locator('.st-key-readback_export_version button').click()
        wait_idle(page, timeout=30000)
        versions = list((srv.cache_dir / "readback_versions").glob("readback_version_*.csv"))
        if versions:
            break
    assert versions, "export must produce a version correction-list CSV under .lv_cache"
    body = versions[0].read_text(encoding="utf-8")
    assert body.splitlines()[0].startswith("sha256")     # header
    assert "classB" in body                              # the recorded correction
