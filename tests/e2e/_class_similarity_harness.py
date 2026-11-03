"""Helpers for class-similarity Streamlit + Playwright E2E.

Isolated server: reuse ``explain_app_server`` (LV_CACHE_DIR per session).
Do not change shared conftest behaviour.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import expect

from .conftest import _ensure_sidebar, load_app, wait_idle

EXPANDER_TITLE = "標錯偵測"
PANEL_CONF = "類別混淆"
PANEL_BAD = "疑似標錯"
PANEL_SIM = "類別相似度分布"


def source_snapshot(root: Path) -> tuple[list[str], str]:
    """Directory listing + content hash. Do not use atime."""
    names: list[str] = []
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        names.append(rel + ("/" if path.is_dir() else ""))
        if path.is_file():
            digest.update(rel.encode("utf-8"))
            digest.update(path.read_bytes())
    return names, digest.hexdigest()


def make_detector_dataset(root: Path, *, n_per_class: int = 8) -> Path:
    """YOLO layout with two classes and unique stems (avoid crop-name collisions)."""
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (root / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
    rng = np.random.default_rng(1)
    for ci, cls in enumerate(("cat", "dog")):
        for i in range(n_per_class):
            stem = f"{cls}_{i:02d}"
            arr = rng.integers(0, 255, (64, 64, 3)).astype("uint8")
            arr[:, :, ci] = 255
            Image.fromarray(arr).save(images / f"{stem}.jpg", quality=85)
            (labels / f"{stem}.txt").write_text(
                f"{ci} 0.5 0.5 0.6 0.6\n", encoding="utf-8")
    return root


def _no_exception(page) -> None:
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)


def _click_key_button(page, key: str, timeout: int = 30000) -> None:
    deadline = time.time() + timeout / 1000
    buttons = page.locator(f".st-key-{key} button")
    while time.time() < deadline:
        for i in range(buttons.count() - 1, -1, -1):
            button = buttons.nth(i)
            try:
                if button.is_visible(timeout=250) and button.is_enabled(timeout=250):
                    button.click(timeout=1000)
                    return
            except Exception:
                continue
        page.wait_for_timeout(100)
    raise AssertionError(f"no visible enabled button for {key!r}")


def _add_folder_visible(page, list_key: str, path: Path) -> None:
    selector = f".st-key-{list_key}_add input"
    entered_path = path.as_posix()
    deadline = time.time() + 30
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
                    expect(page.get_by_text(entered_path, exact=True).last).to_be_visible(
                        timeout=8000)
                    return
            except Exception:
                continue
        page.wait_for_timeout(100)
    raise AssertionError(f"no visible enabled folder input for {list_key!r}")


def _select_pca_only(page) -> None:
    methods = page.locator(".st-key-viz_methods")
    clear = methods.get_by_role("button", name="Clear all")
    if clear.count():
        for i in range(clear.count() - 1, -1, -1):
            btn = clear.nth(i)
            try:
                if btn.is_visible(timeout=250):
                    btn.click(timeout=1000)
                    wait_idle(page)
                    break
            except Exception:
                continue
    methods.locator('[data-baseweb="select"]').last.click()
    page.get_by_role("option", name="PCA", exact=True).click()
    page.keyboard.press("Escape")
    wait_idle(page)


def _wait_scatter(page, timeout: int = 180) -> None:
    page.wait_for_timeout(250)
    running_seen = False
    running_wait_started = time.time()
    deadline = time.time() + timeout
    while time.time() < deadline:
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
    deadline = time.time() + timeout
    while time.time() < deadline:
        wrap = page.locator(".st-key-viz_scatter_wrap").filter(
            has=page.locator("g.points path"))
        if wrap.count() > 0 and wrap.last.locator("g.points path").count() > 0:
            wait_idle(page, timeout=120000)
            return
        page.wait_for_timeout(250)
    raise AssertionError("Visualize scatter never appeared after Run")


def run_image_visualize(page, dataset: Path) -> None:
    _ensure_sidebar(page)
    mode = page.locator(".st-key-viz_mode").get_by_text("Image Classifier", exact=True).last
    mode.click()
    wait_idle(page)
    radios = page.get_by_role("radio", name=re.compile("Image Classifier"))
    if radios.count() and not radios.last.is_checked():
        radios.last.check()
        wait_idle(page)
    page.wait_for_function(
        "() => document.querySelectorAll('.st-key-viz_folder_list_add input').length >= 1",
        timeout=20000,
    )
    _add_folder_visible(page, "viz_folder_list", dataset)
    _select_pca_only(page)
    _ensure_sidebar(page)
    _click_key_button(page, "run_viz", timeout=30000)
    _wait_scatter(page)
    _no_exception(page)


def run_object_visualize(page, dataset: Path) -> None:
    _ensure_sidebar(page)
    mode = page.locator(".st-key-viz_mode").get_by_text("Object Detector", exact=True).last
    mode.click()
    wait_idle(page)
    page.wait_for_function(
        "() => document.querySelectorAll('.st-key-viz_folder_list_add input').length >= 1",
        timeout=20000,
    )
    _add_folder_visible(page, "viz_folder_list", dataset)
    _select_pca_only(page)
    _ensure_sidebar(page)
    _click_key_button(page, "run_viz", timeout=30000)
    _wait_scatter(page)
    _no_exception(page)


def open_label_quality(page) -> None:
    summary = page.locator("details summary", has_text=EXPANDER_TITLE).first
    summary.wait_for(state="visible", timeout=30000)
    summary.scroll_into_view_if_needed()
    is_open = page.evaluate(
        """(t) => {
            const d = [...document.querySelectorAll('details')].find(x => {
                const s = x.querySelector('summary');
                return s && s.textContent.includes(t);
            });
            return Boolean(d && d.open);
        }""",
        EXPANDER_TITLE,
    )
    if not is_open:
        summary.click()
        page.wait_for_timeout(250)
    wait_idle(page)


def select_quality_panel(page, name: str) -> None:
    open_label_quality(page)
    page.locator(".st-key-cs_panel_mode").get_by_text(name, exact=True).last.click()
    wait_idle(page)
    open_label_quality(page)


def compute_similarity(page) -> None:
    select_quality_panel(page, PANEL_SIM)
    btn = page.get_by_role("button", name="計算類別相似度")
    deadline = time.time() + 30
    clicked = False
    while time.time() < deadline:
        for i in range(btn.count() - 1, -1, -1):
            cand = btn.nth(i)
            try:
                if cand.is_visible(timeout=250) and cand.is_enabled(timeout=250):
                    cand.click(timeout=1000)
                    clicked = True
                    break
            except Exception:
                continue
        if clicked:
            break
        page.wait_for_timeout(100)
    if not clicked:
        raise AssertionError("計算類別相似度 button not found")
    wait_idle(page, timeout=120000)
    expect(page.get_by_text("候選 / 可計分", exact=False).first).to_be_visible(timeout=60000)
    _no_exception(page)


def metric_value(page, label: str) -> str:
    card = page.locator('[data-testid="stMetric"]').filter(has_text=label).first
    return card.locator('[data-testid="stMetricValue"]').inner_text().strip()


def candidate_n(page) -> int:
    raw = metric_value(page, "候選 / 可計分")
    return int(raw.split("/")[0].replace(",", "").strip())


def threshold_caption(page) -> str:
    loc = page.get_by_text(re.compile(r"目前門檻 t="))
    loc.first.wait_for(state="visible", timeout=15000)
    return loc.first.inner_text()


def slider_locator(page):
    for frame in page.frames:
        handle = frame.locator('[data-testid="cs-threshold-slider"]')
        try:
            if handle.count() and handle.first.is_visible():
                return handle.first
        except Exception:
            continue
    raise AssertionError("cs-threshold-slider not found in any iframe")


def toggle_first_record_checkbox(page) -> None:
    """Click the first visible table-row checkbox (not the filter checkboxes).

    Streamlit hides the native input; click the visible widget instead.
    """
    open_label_quality(page)
    checks = page.locator('[data-testid="stCheckbox"]')
    deadline = time.time() + 20
    while time.time() < deadline:
        for i in range(checks.count()):
            box = checks.nth(i)
            try:
                if not box.is_visible(timeout=250):
                    continue
                label = box.inner_text()
                if "僅候選" in label or "margin" in label:
                    continue
                box.scroll_into_view_if_needed()
                box.click(timeout=2000)
                wait_idle(page)
                return
            except Exception:
                continue
        page.wait_for_timeout(150)
    raise AssertionError("no visible class-similarity row checkbox")


def new_page(browser, app_server):
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    load_app(page, app_server)
    return ctx, page
