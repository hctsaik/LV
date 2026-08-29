"""E2E：Visualize 單張可解釋性（Attention + session head Grad-CAM）。

真實 Streamlit + Playwright。選圖不自動推論；按「分析此圖」才跑。
不寫使用者資料集。
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pytest
from playwright.sync_api import expect

from .conftest import _add_folder, _ensure_sidebar, load_app, wait_idle

pytestmark = pytest.mark.e2e

expect.set_options(timeout=20000)


def _no_exception(page) -> None:
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)


def _folder_fingerprint(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _open_expander(page, title: str) -> None:
    """Open an expander through its real summary interaction, not JS state."""
    summary = page.locator("details summary", has_text=title).first
    summary.wait_for(state="visible", timeout=15000)
    is_open = page.evaluate(
        """(t) => {
            const d = [...document.querySelectorAll('details')].find(x => {
                const s = x.querySelector('summary');
                return s && s.textContent.includes(t);
            });
            return Boolean(d && d.open);
        }""",
        title,
    )
    if not is_open:
        summary.click()
    page.wait_for_timeout(250)


def _visible_key_button(page, key: str, timeout: int = 30000, *, enabled: bool = False):
    """Return a visible (optionally enabled) Streamlit button after reruns."""
    deadline = time.time() + timeout / 1000
    while time.time() < deadline:
        buttons = page.locator(f".st-key-{key} button")
        for i in range(buttons.count()):
            button = buttons.nth(i)
            if button.is_visible() and (not enabled or button.is_enabled()):
                return button
        page.wait_for_timeout(100)
    state = "visible enabled" if enabled else "visible"
    raise AssertionError(f"no {state} button for Streamlit key {key!r}")


def _click_key_button(page, key: str) -> None:
    """Click the one actionable Streamlit duplicate without bypassing Playwright.

    Streamlit can retain an inert duplicate of a keyed control during a rerun.
    A strict locator correctly rejects that ambiguity, so select the visible
    candidate through Playwright and still use a normal user-facing click.
    """
    button = _visible_key_button(page, key, enabled=True)
    if not button.is_enabled():
        panel = page.locator("body").inner_text()
        raise AssertionError(
            f"visible button for {key!r} is disabled; viewer text:\n{panel[-3000:]}"
        )
    button.click()


def _run_visualize(page, app_server, dataset) -> None:
    load_app(page, app_server)
    page.locator('.st-key-viz_mode').get_by_text("Image Classifier").click()
    wait_idle(page)
    _add_folder(page, "viz_folder_list", str(dataset))
    # The default includes supervised UMAP.  Explainability does not exercise
    # projection quality, so use exactly PCA to keep this real browser test
    # bounded on CPU-only machines.
    methods = page.locator('.st-key-viz_methods')
    methods.get_by_role("button", name="Clear all").click()
    wait_idle(page)
    methods.locator('[data-baseweb="select"]').click()
    page.get_by_role("option", name="PCA", exact=True).click()
    page.keyboard.press("Escape")
    wait_idle(page)
    _ensure_sidebar(page)
    # Folder-entry reruns can leave an inert duplicate behind.  Use the same
    # visible/enabled selection guard as the explainability action buttons.
    _click_key_button(page, "run_viz")
    deadline = time.time() + 180
    scatter_ready = False
    while time.time() < deadline:
        wrap = page.locator('.st-key-viz_scatter_wrap').filter(
            has=page.locator('g.points path'))
        if wrap.count() > 0 and wrap.last.locator('g.points path').count() > 0:
            scatter_ready = True
            break
        time.sleep(0.2)
    assert scatter_ready, "scatter plot never appeared after Run"
    wait_idle(page, timeout=60000)
    _no_exception(page)


def test_explainability_attention_and_gradcam(
    explain_app_server, browser, synthetic_dataset, e2e_cache_dir,
):
    before = _folder_fingerprint(synthetic_dataset)
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    try:
        _run_visualize(page, explain_app_server, synthetic_dataset)

        # 選圖不會自動推論：點卡片後不該立刻出現 DINO 關注 overlay caption
        page.locator('.st-key-viz_grid [class*="st-key-viz_card_"] button').first.click()
        wait_idle(page)
        viewer = page.locator('.st-key-viz_image_viewer')
        expect(viewer.locator('[data-testid="stImage"] img').first).to_be_visible()
        body_before = page.inner_text("body")
        assert "模型實際輸入" not in body_before

        viewer.scroll_into_view_if_needed()
        _open_expander(page, "可解釋性")
        wait_idle(page)
        expect(page.get_by_text("DINO Attention", exact=False).first).to_be_visible()

        # Missing Normal Bank must be an actionable guide, not a dead-end error.
        page.locator('.st-key-viz_explain_mode').locator(
            "text=Anomaly Heatmap").last.click()
        wait_idle(page)
        _open_expander(page, "可解釋性")
        expect(page.get_by_text(
            "如何建立並使用 Anomaly Heatmap", exact=True).first).to_be_visible()
        expect(page.get_by_text(
            "目前尚未建立 patch 模式的 Normal Bank", exact=False).first).to_be_visible()
        page.locator('.st-key-viz_explain_mode').locator(
            "text=DINO Attention").last.click()
        wait_idle(page)
        _open_expander(page, "可解釋性")
        cache_before_explain = _folder_fingerprint(e2e_cache_dir)
        _click_key_button(page, "viz_explain_analyze")
        page.wait_for_selector(
            "text=模型實際輸入", timeout=180000)
        wait_idle(page, timeout=120000)
        _no_exception(page)
        expect(page.get_by_text("DINO 關注區域", exact=False).first).to_be_visible()
        expect(page.get_by_text("不是瑕疵", exact=False).first).to_be_visible()
        expect(page.get_by_text("重建", exact=False).first).to_be_visible()
        _visible_key_button(page, "viz_explain_dl_png")
        _visible_key_button(page, "viz_explain_dl_json")

        # session head → Grad-CAM
        _open_expander(page, "可解釋性")
        page.locator('.st-key-viz_explain_mode').locator(
            "text=分類理由").last.click()
        wait_idle(page)
        _open_expander(page, "可解釋性")
        _click_key_button(page, "viz_explain_train_head")
        wait_idle(page, timeout=120000)
        expect(page.get_by_text("session head", exact=False).first).to_be_visible()
        _open_expander(page, "可解釋性")
        _click_key_button(page, "viz_explain_analyze")
        page.wait_for_selector("text=實際 top-1", timeout=180000)
        wait_idle(page, timeout=120000)
        _no_exception(page)
        expect(page.get_by_text("對比分類證據", exact=False).first).to_be_visible()
        expect(page.get_by_text("正常 / Unknown", exact=False).first).to_be_visible()
        assert _folder_fingerprint(e2e_cache_dir) == cache_before_explain, \
            "單張 Attention / Grad-CAM 不得寫入 LV cache"

        # Verify the guide's primary action after the rest of this Visualize
        # workflow is complete; navigation intentionally leaves this tool.
        _open_expander(page, "可解釋性")
        page.locator('.st-key-viz_explain_mode').locator(
            "text=Anomaly Heatmap").last.click()
        wait_idle(page)
        _open_expander(page, "可解釋性")
        _click_key_button(page, "viz_explain_normal_bank_guide")
        wait_idle(page)
        expect(page.get_by_text("① 建模 / 載入模型", exact=False).first).to_be_visible()
        expect(page.get_by_text("分數依據", exact=True).first).to_be_visible()

        after = _folder_fingerprint(synthetic_dataset)
        assert after == before, "可解釋性不得寫入使用者資料集"
    finally:
        ctx.close()
