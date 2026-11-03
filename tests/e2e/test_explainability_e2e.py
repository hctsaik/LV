"""E2E：Visualize 單張可解釋性（Attention + session head Grad-CAM）。

真實 Streamlit + Playwright。選圖不自動推論；按「分析此圖」才跑。
不寫使用者資料集。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import pytest
from playwright.sync_api import expect

from .conftest import _add_folder, _ensure_sidebar, load_app, wait_idle

pytestmark = pytest.mark.e2e

expect.set_options(timeout=20000)


@pytest.fixture(scope="session")
def synthetic_dataset(synthetic_dataset):
    reuse = os.environ.get("LV_DINO_E2E_REUSE_ROOT")
    if not reuse:
        return synthetic_dataset
    previous = Path(reuse).resolve() / "ds0" / "train"
    assert previous.is_dir(), "explicit DINO reuse dataset is missing"
    assert _folder_fingerprint(previous) == _folder_fingerprint(synthetic_dataset), \
        "reused dataset must exactly match the deterministic fixture"
    return previous


@pytest.fixture(scope="session")
def e2e_cache_dir(e2e_cache_dir, synthetic_dataset):
    reuse = os.environ.get("LV_DINO_E2E_REUSE_ROOT")
    if not reuse:
        return e2e_cache_dir
    previous = Path(reuse).resolve() / "lv_cache0"
    assert previous.is_dir(), "explicit DINO reuse cache is missing"
    # Dataset bytes and original paths are verified above. The application's
    # cache still validates model/recipe identities; session heads are rebuilt.
    return previous


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
        # Streamlit keeps the previous fragment visible while streaming its
        # replacement. Prefer the newest actionable button, as Diagnostics
        # does, so the click reaches the current callback registration.
        for i in range(buttons.count() - 1, -1, -1):
            button = buttons.nth(i)
            try:
                current = button.evaluate("el => !el.closest('[data-stale=\"true\"]')")
                if current and button.is_visible() and (not enabled or button.is_enabled()):
                    return button
            except Exception:
                continue  # A rerun can detach a previous fragment instance.
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


def _run_visualize(page, app_server, dataset, *, artifacts=None) -> None:
    # Share the verified route/radio/rerun guards with the Diagnostics entry.
    from .test_explainability_diagnostics_e2e import _run_visualize as run_visualize
    load_app(page, app_server)
    try:
        run_visualize(page, dataset)
    except Exception:
        if artifacts is not None:
            artifacts = Path(artifacts)
            artifacts.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(artifacts / "visualize_after_run.png"), full_page=True)
            (artifacts / "visualize_after_run.txt").write_text(page.inner_text("body"), encoding="utf-8")
            (artifacts / "visualize_after_run.html").write_text(page.content(), encoding="utf-8")
        raise
    _no_exception(page)


def test_explainability_attention_and_gradcam(
    explain_app_server, browser, synthetic_dataset, e2e_cache_dir, tmp_path,
):
    before = _folder_fingerprint(synthetic_dataset)
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    def stage(value):
        (tmp_path / "legacy_stage.txt").write_text(value, encoding="utf-8")
    try:
        stage("Visualize")
        _run_visualize(page, explain_app_server, synthetic_dataset,
                       artifacts=tmp_path / "legacy_visualize_failure")
        stage("Attention and Normal Bank guide")

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
        stage("336 exact-resolution head")
        wait_idle(page, timeout=120000)
        expect(page.get_by_text("session head", exact=False).first).to_be_visible()
        # Verify that a high-resolution probe creates a native 24x24 grid.
        res_box = page.locator(
            ".st-key-viz_explain_gradcam_res_v1 [data-baseweb='select']"
        ).first
        res_box.click()
        # BaseWeb renders the open menu in a portal, so scope by semantic role
        # rather than by the selectbox's DOM subtree.
        resolution_options = page.get_by_role("option").all_text_contents()
        for expected_res, expected_grid in (("224", "16"), ("336", "24"), ("448", "32")):
            assert any(
                expected_res in option and expected_grid in option
                for option in resolution_options
            ), f"missing Grad-CAM resolution option {expected_res}×{expected_res}: {resolution_options!r}"
        page.get_by_role("option", name=re.compile(r"336")).last.click()
        page.keyboard.press("Escape")
        wait_idle(page)
        expect(res_box).to_contain_text("336")
        expect(page.get_by_text(
            "已建立 336×336 exact-resolution session head", exact=False
        ).first).to_be_visible(timeout=180000)
        assert "只提高空間採樣" not in page.inner_text("body")
        stage("336 Grad-CAM result")
        _open_expander(page, "可解釋性")
        _click_key_button(page, "viz_explain_analyze")
        # "實際 top-1" is also the target selector's default option. Wait
        # for a result-only caption so a queued fragment run cannot pass.
        expect(page.get_by_text("對比分類證據", exact=False).first).to_be_visible(timeout=180000)
        wait_idle(page, timeout=120000)
        _no_exception(page)
        expect(page.get_by_text("對比分類證據", exact=False).first).to_be_visible()
        expect(page.get_by_text("正常 / Unknown", exact=False).first).to_be_visible()
        expect(page.get_by_text("分析解析度：336→24×24 patch", exact=False).first).to_be_visible()
        page.screenshot(path=str(tmp_path / "legacy_gradcam.png"), full_page=True)

        cache_after_explain = _folder_fingerprint(e2e_cache_dir)
        assert cache_after_explain == cache_before_explain, \
            "單張 Attention / Grad-CAM 不得寫入 LV cache"

        # Verify the guide's primary action after the rest of this Visualize
        stage("Normal Bank navigation and non-mutation")
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
        (tmp_path / "immutability.json").write_text(json.dumps({
            "dataset_before": before, "dataset_after": after,
            "cache_before_explanation": cache_before_explain,
            "cache_after_explanation": cache_after_explain,
            "dataset_unchanged": True, "cache_unchanged": True,
        }, indent=2), encoding="utf-8")
        stage("passed")
    except Exception:
        artifacts = tmp_path / "legacy_inspector_failure"
        artifacts.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(artifacts / "inspector.png"), full_page=True)
        (artifacts / "inspector.txt").write_text(page.inner_text("body"), encoding="utf-8")
        (artifacts / "inspector.html").write_text(page.content(), encoding="utf-8")
        raise
    finally:
        ctx.close()
