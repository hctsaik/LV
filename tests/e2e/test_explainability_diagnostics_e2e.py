"""Browser-level gates for the Model Diagnostics route and Visualize handoff."""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import pytest
from playwright.sync_api import expect

from .conftest import _ensure_sidebar, load_app, wait_idle

pytestmark = pytest.mark.e2e
expect.set_options(timeout=30000)


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _click_key_button(page, key: str, timeout: int = 30000) -> None:
    deadline = time.time() + timeout / 1000
    buttons = page.locator(f".st-key-{key} button")
    while time.time() < deadline:
        # Streamlit can retain a hidden copy of a widget for one rerun.  The
        # newest mounted copy is last; choose the last visible/enabled button
        # so a click cannot land on the obsolete DOM instance.
        for i in range(buttons.count() - 1, -1, -1):
            button = buttons.nth(i)
            try:
                visible = button.is_visible(timeout=250)
                enabled = button.is_enabled(timeout=250)
            except Exception:
                continue
            if visible and enabled:
                try:
                    button.click(timeout=1000)
                    return
                except Exception:
                    # A Streamlit rerun may detach this copy between the
                    # visibility check and the click.  Re-query on the next
                    # poll rather than failing on a transient duplicate.
                    continue
        page.wait_for_timeout(100)
    raise AssertionError(f"no visible enabled button for {key!r}")


def _has_visible_enabled_button(page, key: str) -> bool:
    buttons = page.locator(f".st-key-{key} button")
    for i in range(buttons.count() - 1, -1, -1):
        try:
            if buttons.nth(i).is_visible(timeout=250) and buttons.nth(i).is_enabled(timeout=250):
                return True
        except Exception:
            continue
    return False


def _try_click_key_button(page, key: str, timeout: int = 30000) -> bool:
    """Click an optional action, tolerating rerun/remount races."""
    try:
        _click_key_button(page, key, timeout=timeout)
    except AssertionError:
        return False
    return True


def _click_text_button(page, pattern: str, timeout: int = 30000) -> None:
    locator = page.get_by_role("button", name=re.compile(pattern))
    deadline = time.time() + timeout / 1000
    while time.time() < deadline:
        for i in range(locator.count() - 1, -1, -1):
            candidate = locator.nth(i)
            try:
                if candidate.is_visible(timeout=250) and candidate.is_enabled(timeout=250):
                    candidate.click(timeout=1000)
                    return
            except Exception:
                continue
        page.wait_for_timeout(100)
    raise AssertionError(f"no visible enabled button matching {pattern!r}")


def _has_text_button(page, pattern: str) -> bool:
    locator = page.get_by_role("button", name=re.compile(pattern))
    for i in range(locator.count() - 1, -1, -1):
        try:
            if locator.nth(i).is_visible(timeout=250) and locator.nth(i).is_enabled(timeout=250):
                return True
        except Exception:
            continue
    return False


def _has_text(page, pattern: str) -> bool:
    locator = page.get_by_text(pattern, exact=False)
    for index in range(locator.count() - 1, -1, -1):
        try:
            if locator.nth(index).is_visible(timeout=250):
                return True
        except Exception:
            continue
    return False


def _extract_identity(page, label: str) -> str:
    """Read a short content identity rendered by the diagnostics header."""
    pattern = re.compile(
        rf"{re.escape(label)}\s*[:：]\s*`?([0-9a-f]{{12,64}})", re.I
    )
    # Streamlit can keep a hidden delta during a widget rerun. Prefer the
    # newest visible caption, then fall back to body text_content so a valid
    # caption is not lost just because its parent is in a transient delta.
    candidates = page.get_by_text(re.compile(re.escape(label), re.I), exact=False)
    for index in range(candidates.count() - 1, -1, -1):
        candidate = candidates.nth(index)
        try:
            if candidate.is_visible(timeout=250):
                match = pattern.search(candidate.text_content() or "")
                if match:
                    return match.group(1)
        except Exception:
            continue
    body = page.locator("body").text_content() or ""
    match = pattern.search(body)
    if not match:
        raise AssertionError(f"{label} was not rendered in diagnostics header:\n{body[-2500:]}")
    return match.group(1)


def _switch_tool(page, label: str) -> None:
    def tool_is_active() -> bool:
        active = page.locator(
            '.st-key-tool_switch button[kind="segmented_controlActive"]'
        )
        for index in range(active.count() - 1, -1, -1):
            try:
                if active.nth(index).get_by_text(label, exact=True).is_visible(timeout=250):
                    return True
            except Exception:
                continue
        return False

    def route_ready() -> bool:
        if not tool_is_active():
            return False
        if label == "Visualize Embeddings":
            candidates = [page.locator(".st-key-viz_mode")]
        elif label == "Model Diagnostics":
            # The title can remain mounted as a stale Streamlit delta after a
            # fragment handoff.  Use route-owned content instead: empty,
            # current, stale, and incompatible diagnostics states are all
            # explicit and visible only on this route.
            candidates = [
                page.get_by_text("Context current", exact=False),
                page.get_by_text("Context stale", exact=False),
                page.get_by_text("Context incompatible", exact=False),
                page.get_by_text(
                    "請先到 Visualize 執行資料分析，再選取一張圖片",
                    exact=False,
                ),
            ]
        else:
            candidates = []
        for candidate in candidates:
            for index in range(candidate.count() - 1, -1, -1):
                try:
                    if candidate.nth(index).is_visible(timeout=250):
                        return True
                except Exception:
                    continue
        return False

    deadline = time.time() + 30000 / 1000
    while time.time() < deadline:
        if route_ready():
            return
        switch = page.locator(".st-key-tool_switch")
        buttons = switch.get_by_text(label, exact=True)
        for index in range(buttons.count() - 1, -1, -1):
            button = buttons.nth(index)
            try:
                if button.is_visible(timeout=250) and button.is_enabled(timeout=250):
                    button.click(timeout=1000)
                    wait_idle(page)
                    break
            except Exception:
                continue
        page.wait_for_timeout(100)
    raise AssertionError(f"route {label!r} did not become ready")


def _open_diagnostics_details(page) -> None:
    """Open the progressive-disclosure panel when assertions need provenance."""
    deadline = time.time() + 30000 / 1000
    while time.time() < deadline:
        trigger = page.get_by_text(
            "查看分析 context、可靠度與 OOF 詳細資料", exact=True
        )
        for index in range(trigger.count() - 1, -1, -1):
            candidate = trigger.nth(index)
            try:
                if candidate.is_visible(timeout=250):
                    candidate.click(timeout=1000)
                    wait_idle(page)
                    return
            except Exception:
                continue
        page.wait_for_timeout(100)
    raise AssertionError(
        "diagnostics detail expander did not mount:\n"
        + page.locator("body").inner_text()[-3000:]
    )


def _close_diagnostics_details(page) -> None:
    trigger = page.get_by_text(
        "查看分析 context、可靠度與 OOF 詳細資料", exact=True
    ).last
    if trigger.is_visible():
        trigger.click()
        wait_idle(page)


def _add_folder_visible(page, list_key: str, path: Path) -> None:
    """Enter a folder through the newest visible Streamlit input instance.

    App-scope handoffs can leave one old fragment input mounted for a brief
    window.  Selecting the visible/enabled copy keeps the second Visualize
    generation deterministic without changing the shared test fixture.
    """
    selector = f".st-key-{list_key}_add input"
    # Forward slashes are accepted on Windows and survive Markdown captions
    # unchanged (a Windows backslash before a dot is Markdown escaping).
    entered_path = path.as_posix()
    deadline = time.time() + 30000 / 1000
    while time.time() < deadline:
        inputs = page.locator(selector)
        for index in range(inputs.count() - 1, -1, -1):
            candidate = inputs.nth(index)
            try:
                if candidate.is_visible(timeout=250) and candidate.is_enabled(timeout=250):
                    candidate.fill(entered_path)
                    candidate.press("Enter")
                    page.wait_for_function(
                        "([s, p]) => Array.from(document.querySelectorAll(s))"
                        ".filter(el => el.offsetParent !== null)"
                        ".every(el => el.value !== p)",
                        arg=[selector, entered_path],
                        timeout=15000,
                    )
                    wait_idle(page)
                    # A remount can clear the input without committing its
                    # value. Only the rendered folder caption proves that the
                    # server accepted this submission; retry the real action
                    # if a simultaneous mode rerun discarded it.
                    expect(page.get_by_text(entered_path, exact=True).last).to_be_visible(
                        timeout=5000)
                    return
            except Exception:
                continue
        page.wait_for_timeout(100)
    raise AssertionError(f"no visible enabled folder input for {list_key!r}")


def _run_visualize(page, dataset: Path) -> None:
    mode = page.locator('.st-key-viz_mode').get_by_text("Image Classifier", exact=True).last
    mode.click()
    wait_idle(page)
    # A top-level route remount can restore the default detector radio after
    # the text click.  Confirm the actual radio state before entering inputs.
    radios = page.get_by_role("radio", name=re.compile("Image Classifier"))
    if radios.count() and not radios.last.is_checked():
        radios.last.check()
        wait_idle(page)
    _add_folder_visible(page, "viz_folder_list", dataset)
    methods = page.locator('.st-key-viz_methods')
    methods.get_by_role("button", name="Clear all").click()
    wait_idle(page)
    methods.locator('[data-baseweb="select"]').click()
    page.get_by_role("option", name="PCA", exact=True).click()
    page.keyboard.press("Escape")
    wait_idle(page)
    _ensure_sidebar(page)
    _click_key_button(page, "run_viz", timeout=30000)
    # Do not mistake the previous scatter for the new run.  Observe at least
    # one running-status transition (when the embedding pass is long enough)
    # before accepting the scatter as committed.
    page.wait_for_timeout(250)
    running_seen = False
    running_wait_started = time.time()
    run_deadline = time.time() + 180
    while time.time() < run_deadline:
        status_widget = page.locator('[data-testid="stStatusWidget"]')
        try:
            running = status_widget.count() > 0 and status_widget.first.is_visible(timeout=250)
        except Exception:
            running = False
        if running:
            running_seen = True
        elif running_seen:
            break
        elif time.time() - running_wait_started > 20:
            break
        page.wait_for_timeout(100)
    wait_idle(page, timeout=120000)
    deadline = time.time() + 180
    while time.time() < deadline:
        wrap = page.locator('.st-key-viz_scatter_wrap').filter(
            has=page.locator('g.points path'))
        if wrap.count() > 0 and wrap.last.locator('g.points path').count() > 0:
            wait_idle(page, timeout=120000)
            page.locator('.st-key-viz_grid').wait_for(state="visible", timeout=30000)
            return
        page.wait_for_timeout(250)
    raise AssertionError("Visualize scatter never appeared after Run")


def test_model_diagnostics_route_handoff_and_direct_heatmap(
    explain_app_server, browser, synthetic_dataset, e2e_cache_dir,
):
    """Exercise the route, direct heatmap action, and render-only behavior.

    1. Empty top-level route is reachable.
    2. Page entry does not start inference.
    3. Visualize exposes the CTA only for a selected record and hands off a
       reference-only context at app scope.
    4. The shell exposes conservative head/mask/bank statuses without a page
       engine or implicit attribution.
    Stale-generation compatibility remains covered by the context/unit suite;
    this browser gate stays focused on the user-facing diagnostics action.
    """
    source_before = _fingerprint(synthetic_dataset)
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    try:
        load_app(page, explain_app_server)

        # Scenario 1: the new top-level route is reachable with no context.
        _switch_tool(page, "Model Diagnostics")
        expect(page.get_by_text("模型診斷", exact=True).first).to_be_visible()
        expect(page.get_by_text(
            "請先到 Visualize 執行資料分析，再選取一張圖片按「分析這張圖」",
            exact=False,
        )).to_be_visible()

        # Scenario 2: page entry remains an empty, no-auto-run state.
        expect(page.get_by_text("尚未開始模型推論", exact=False)).to_be_visible()
        expect(page.get_by_text(
            "尚未開始 inference / OOF / faithfulness", exact=False
        )).to_have_count(0)

        # Prepare one Visualize run for the handoff scenarios.
        _switch_tool(page, "Visualize Embeddings")
        _run_visualize(page, synthetic_dataset)
        cache_after_run = _fingerprint(e2e_cache_dir)

        # Scenario 3: no record means no handoff CTA; selecting one enables it.
        expect(page.locator(".st-key-viz_slot_model_diagnostics")).to_have_count(0)
        cards = page.locator('.st-key-viz_grid [class*="st-key-viz_card_"] button')
        first_card = next(
            cards.nth(i) for i in range(cards.count() - 1, -1, -1)
            if cards.nth(i).is_visible() and cards.nth(i).is_enabled()
        )
        first_card.click()
        wait_idle(page)
        handoff = page.locator(".st-key-viz_slot_model_diagnostics button")
        deadline = time.time() + 30000 / 1000
        visible_handoff = None
        while time.time() < deadline:
            for i in range(handoff.count() - 1, -1, -1):
                candidate = handoff.nth(i)
                if candidate.is_visible() and candidate.is_enabled():
                    visible_handoff = candidate
                    break
            if visible_handoff is not None:
                break
            page.wait_for_timeout(100)
        if visible_handoff is None:
            raise AssertionError(
                "Visualize handoff CTA did not appear after selecting a card:\n"
                + page.locator("body").inner_text()[-3000:]
            )
        expect(visible_handoff).to_be_visible()
        handoff = visible_handoff
        handoff.click()
        wait_idle(page)
        expect(page.get_by_text("Context current", exact=False)).to_be_visible()
        expect(page.get_by_text("分析結論", exact=True).last).to_be_visible()
        _open_diagnostics_details(page)
        expect(page.get_by_text("目前分析對象", exact=True)).to_be_visible()
        expect(page.get_by_text("資料集", exact=True)).to_be_visible()
        assert "影像" in page.locator("body").inner_text()
        # Scenario 4: conservative capability state and no implicit inference.
        expect(page.get_by_text("分類模型", exact=True)).to_be_visible()
        expect(page.get_by_text("unavailable", exact=False).first).to_be_visible()
        expect(page.get_by_text("Mask：unavailable", exact=False).first).to_be_visible()
        expect(page.get_by_text("Normal Bank：unavailable", exact=False).first).to_be_visible()
        expect(page.get_by_text("尚未準備分類模型", exact=False).first).to_be_visible()
        selector = page.get_by_role("combobox", name="分析精細度")
        expect(selector).to_be_visible()
        selector.click()
        expect(page.get_by_role("option", name=re.compile(r"^336×336"))).to_be_visible()
        expect(page.get_by_role("option", name=re.compile(r"^448×448"))).to_be_visible()
        page.keyboard.press("Escape")
        # Changing resolution is configuration only.  It must expose the
        # combined action without training a 336 head as a selector side
        # effect; the test returns to 224 before running the expensive path.
        selector.click()
        page.get_by_role("option", name=re.compile(r"^336×336")).click()
        wait_idle(page)
        expect(page.get_by_role(
            "button", name=re.compile(r"建立 336×336 模型並分析此圖")
        )).to_be_visible(timeout=30000)
        assert not _has_text(page, "分類模型已準備：336×336")
        selector = page.get_by_role("combobox", name="分析精細度")
        selector.click()
        page.get_by_role("option", name=re.compile(r"^224×224")).click()
        wait_idle(page)
        assert _fingerprint(e2e_cache_dir) == cache_after_run, \
            "context handoff must not write existing cache artifacts"
        # Keep the provenance expander closed while the single analysis action
        # runs; this avoids mixing old expanded fragments with the committed
        # head/result in the browser DOM.
        _close_diagnostics_details(page)

        # One primary action owns both exact-head preparation and canonical
        # signed Grad-CAM.  There is no intermediate training click and no
        # research acknowledgement click.
        assert not _has_text_button(page, "我知道這是研究用途")
        expect(page.get_by_role("button", name=re.compile(r"建立 224×224 模型並分析此圖"))).to_be_visible(
            timeout=30000)
        assert page.get_by_role("button").filter(has_text="建立 224×224").count() == 1
        _click_key_button(page, "diag_analyze", timeout=300000)
        expect(page.get_by_text("分類模型已準備：224×224", exact=False)).to_be_visible(
            timeout=180000)
        expect(page.get_by_text(
            "Blue = lowers the target-reference margin", exact=False,
        )).to_be_visible(timeout=180000)
        expect(page.get_by_text("分類模型可靠度", exact=True).first).to_be_visible()
        compute_key_before_render_only = _extract_identity(page, "Compute key")
        # Grid/opacity are render-only controls.  Toggling the grid causes a
        # Streamlit rerun but must preserve the same model compute identity.
        grid = page.get_by_text("顯示 patch grid", exact=True).last
        if grid.is_visible():
            grid.click()
            wait_idle(page)
            expect(page.get_by_text("Blue = lowers the target-reference margin", exact=False)).to_be_visible()
            expect(page.get_by_text("Compute key", exact=False)).to_be_visible(timeout=30000)
            assert _extract_identity(page, "Compute key") == compute_key_before_render_only

        # The same single action must also build and explain each missing
        # exact-resolution head. This covers prepared-head reuse at 224 and
        # missing-head work at both higher resolutions.
        for resolution in (336, 448):
            selector = page.get_by_role("combobox", name="分析精細度")
            selector.click()
            page.get_by_role(
                "option", name=re.compile(rf"^{resolution}×{resolution}")
            ).click()
            wait_idle(page)
            expect(page.get_by_role(
                "button", name=re.compile(
                    rf"建立 {resolution}×{resolution} 模型並分析此圖"
                )
            )).to_be_visible(timeout=30000)
            assert not _has_text(page, f"分類模型已準備：{resolution}×{resolution}")
            _click_key_button(page, "diag_analyze", timeout=300000)
            expect(page.get_by_text(
                f"分類模型已準備：{resolution}×{resolution}", exact=False
            )).to_be_visible(timeout=240000)
            expect(page.get_by_text(
                "Blue = lowers the target-reference margin", exact=False,
            )).to_be_visible(timeout=240000)
        assert _fingerprint(e2e_cache_dir) == cache_after_run, \
            "diagnostics inference must not write existing cache artifacts"

        assert _fingerprint(synthetic_dataset) == source_before, \
            "diagnostics route must not modify the source dataset"
    finally:
        ctx.close()
