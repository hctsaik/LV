"""Shared Playwright driving helpers for the M9 AL-loop E2E (pool_registry +
round_ledger GUI wiring). Both suites drive the same real send / read-back /
round-timeline path through the CART, not the scatter.

Why the cart path: the send bookkeeping lives in the ONE shared component
`_send_to_labeling_ui`. It is reachable from the Visualize selection panel (inside
the `@st.fragment` `_render_right_panel`) AND from the curation cart in the 匯出
tool (`_export_subset_ui`, NOT a fragment). Selecting via the scatter fires
`on_select="rerun"` (a full-app rerun) which re-renders the fragment from outside
and transiently leaves a STALE copy of the send button in the DOM — clicking it
is a no-op. Driving via the cart (sampling panel add-all → switch to 匯出 → send)
only ever touches the live, non-fragment button, so it is deterministic.

Contract strings the wiring (scripts/app.py) must produce:
  · cart send component (key=cart_to_labeling): button label ends with 「（n）」,
    disabled when 0; partition caption contains 「已送標」 when part of the batch was
    already sent; override checkbox label 含「仍包含」.
  · 📥 round timeline: each round shows 「送出 N」/「回讀 M」/「變更 K」 and 「策略：cart」.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from .conftest import _add_folder, wait_idle

CART_SEND_KEY = "cart_to_labeling"


def run_viz(page, folder) -> None:
    """Load the synthetic dataset in Visualize (Image Classifier) and Run. The
    first AL test pays a cold DINOv2 embedding compute (~90s); later tests hit the
    shared cache, so wait generously for the scatter to appear."""
    page.locator('.st-key-viz_mode').get_by_text("Image Classifier").click()
    wait_idle(page)
    # the mode switch briefly mounts two copies of the folder-input during its
    # rerun; wait for it to settle to one so _add_folder's strict locator is safe.
    page.wait_for_function(
        """() => document.querySelectorAll('.st-key-viz_folder_list_add input').length === 1""",
        timeout=20000)
    _add_folder(page, "viz_folder_list", str(folder))
    page.locator('.st-key-run_viz button').click()
    deadline = time.time() + 260
    while time.time() < deadline:
        if page.locator('.st-key-viz_scatter_wrap g.points path').count() > 0:
            break
        time.sleep(0.3)
    wait_idle(page, timeout=120000)


def sample_to_cart(page) -> None:
    """Pick diverse samples via the 選樣 panel and add them all to the curation
    cart. Uses only fragment-internal reruns (panel switch + sampling buttons),
    so no stale-copy duplication — same path as the passing test_x_diversity."""
    page.locator('.st-key-viz_panel_view').get_by_text("選樣", exact=True).click()
    wait_idle(page)
    page.locator('.st-key-viz_sampling_btn button').click()
    wait_idle(page, timeout=90000)
    page.locator('.st-key-viz_sampling_addall button').click()
    wait_idle(page)


def goto_readback(page) -> None:
    wait_idle(page, timeout=30000)
    # partial match: the segmented-control label carries an emoji + spacing that
    # an exact match trips over; scoped to .st-key-tool_switch so the toolbar
    # caption's copy of the same text isn't matched.
    page.locator('.st-key-tool_switch').get_by_text("標註回饋").first.click()
    wait_idle(page, timeout=60000)


def goto_cart(page) -> None:
    """Open the 匯出清單 (curation cart) right-panel view — reached via the
    fragment-internal panel switcher, so the cart send button renders clean."""
    page.locator('.st-key-viz_panel_view').get_by_text("匯出清單", exact=True).click()
    wait_idle(page)


_CART_CSS = f'.st-key-{CART_SEND_KEY} button'


def _wait_cart(page, want_enabled: bool, timeout: int = 30000) -> None:
    # clicking the cart send fires a full st.rerun which can transiently duplicate
    # the fragment button — poll for the wanted enabled/disabled state.
    page.wait_for_function(
        """([sel, we]) => {
            const bs = [...document.querySelectorAll(sel)]
                .filter(b => b.offsetParent !== null);
            if (!bs.length) return false;
            return we ? bs.some(b => !b.disabled) : bs.every(b => b.disabled);
        }""",
        arg=[_CART_CSS, want_enabled], timeout=timeout)


def cart_send_button(page):
    return page.locator(f'{_CART_CSS}:visible').first


def cart_send_count(page) -> int:
    _wait_cart(page, want_enabled=True)
    m = re.search(r"（(\d+)）", cart_send_button(page).inner_text())
    return int(m.group(1)) if m else 0


def blocked_caption(page):
    return page.get_by_text(re.compile("已送標")).first


def wait_cart_blocked(page) -> None:
    _wait_cart(page, want_enabled=False, timeout=25000)


def _send_confirmed(page, timeout: int = 9000) -> bool:
    # main() renders `_render_send_confirmation` ('✅ 已送 N 張到 Labeling…') after a
    # successful send — the reliable signal the click actually fired the callback.
    try:
        page.wait_for_function(
            """() => { const t = document.body.innerText;
                       return t.includes('已送') && t.includes('到 Labeling'); }""",
            timeout=timeout)
        return True
    except Exception:
        return False


def send_cart(page, include_handled: bool = False) -> None:
    """Click the cart send and CONFIRM it fired (retrying the click). A plotly/
    fragment rerun can leave the click landing on a stale button copy that does
    nothing; retry until the send confirmation appears."""
    if include_handled:
        page.get_by_text("仍包含已送標／已標的樣本").first.click()
        wait_idle(page)
    for _ in range(4):
        _wait_cart(page, want_enabled=True)
        page.locator(f'{_CART_CSS}:visible:enabled').first.click()
        wait_idle(page, timeout=60000)
        if _send_confirmed(page):
            return


def apply_readback_click(page) -> None:
    """Click 「📥 套用讀回結果」 and confirm it fired (retrying). Success renders
    either '已套用 N 筆變更' or '沒有新的標籤變更'."""
    for _ in range(4):
        page.locator('.st-key-readback_apply_btn button').click()
        wait_idle(page, timeout=60000)
        try:
            page.wait_for_function(
                """() => { const t = document.body.innerText;
                           return t.includes('已套用') || t.includes('沒有新的標籤變更'); }""",
                timeout=8000)
            return
        except Exception:
            continue


def read_jsonl(path: Path) -> list[dict]:
    import json
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
