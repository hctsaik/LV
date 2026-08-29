"""Opt-in real-image E2E for the five MVTec explainability cases."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import pytest
from playwright.sync_api import expect

from .conftest import wait_idle
from .test_explainability_e2e import (
    _click_key_button,
    _open_expander,
    _run_visualize,
    _visible_key_button,
)

pytestmark = pytest.mark.e2e

_CASE_NAMES = (
    "000_regular.png",
    "000_underexposed.png",
    "000_overexposed.png",
    "000_shift_1.png",
    "001_regular.png",
)
_DEFAULT_BAD_DIR = Path(
    r"C:\code\dataset\MVTec2\ad2_test_public_gold_v1\mvtec_ad_2\can\test_public\bad"
)
_USER_BAD_DIR = Path(
    r"C:\code\dataset\MVTec2\ad2\_test\_public\_gold\_v1\mvtec\_ad\_2\can\test\_public\bad"
)
_DEFAULT_CAPSULE_TEST_DIR = Path(r"C:\code\dataset\MVTec\capsule\test")
_CAPSULE_CLASSES = ("good", "scratch", "crack")
_CAPSULE_SAMPLES_PER_CLASS = 6
_CAPSULE_TARGET = "scratch"


def _fingerprint(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def _image_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


@pytest.fixture(scope="session")
def mvtec_capsule_gradcam_dataset(tmp_path_factory) -> tuple[Path, Path, str]:
    """Copy a small, labelled real-MVTec classifier set without touching source data."""
    if os.environ.get("LV_RUN_REAL_MVTEC_GRADCAM_E2E") != "1":
        pytest.skip("set LV_RUN_REAL_MVTEC_GRADCAM_E2E=1 for real MVTec Grad-CAM E2E")
    source = Path(
        os.environ.get("LV_MVTEC_CAPSULE_TEST_DIR", str(_DEFAULT_CAPSULE_TEST_DIR))
    )
    selected: dict[str, list[Path]] = {}
    for label in _CAPSULE_CLASSES:
        files = _image_files(source / label) if (source / label).is_dir() else []
        if len(files) < _CAPSULE_SAMPLES_PER_CLASS:
            pytest.skip(
                f"MVTec capsule class {label!r} needs at least "
                f"{_CAPSULE_SAMPLES_PER_CLASS} images: {source / label}"
            )
        selected[label] = files[:_CAPSULE_SAMPLES_PER_CLASS]

    target = tmp_path_factory.mktemp("mvtec_capsule_gradcam") / "test"
    for label, files in selected.items():
        target_dir = target / label
        target_dir.mkdir(parents=True)
        for path in files:
            shutil.copy2(path, target_dir / path.name)
    return source, target, selected[_CAPSULE_TARGET][0].name


@pytest.fixture(scope="session")
def mvtec_bad_dataset(tmp_path_factory) -> tuple[Path, Path]:
    if os.environ.get("LV_RUN_REAL_MVTEC_E2E") != "1":
        pytest.skip("set LV_RUN_REAL_MVTEC_E2E=1 to run real MVTec E2E")
    source = Path(os.environ.get("LV_MVTEC_BAD_DIR", str(_DEFAULT_BAD_DIR)))
    if not source.is_dir() and _USER_BAD_DIR.is_dir():
        source = _USER_BAD_DIR
    missing = [name for name in _CASE_NAMES if not (source / name).is_file()]
    if missing:
        pytest.skip(f"MVTec bad folder missing {missing}: {source}")
    target = tmp_path_factory.mktemp("mvtec_bad") / "bad"
    target.mkdir()
    for name in _CASE_NAMES:
        shutil.copy2(source / name, target / name)
    return source, target


def _click_card(page, name: str) -> None:
    """Click the newest enabled visible card whose text contains ``name``."""
    buttons = page.locator('.st-key-viz_grid [class*="st-key-viz_card_"] button')
    # During a rerun Streamlit can briefly retain an older visible node.  The
    # last enabled candidate is the newest DOM node and carries the live click
    # callback for that card.
    for i in range(buttons.count() - 1, -1, -1):
        button = buttons.nth(i)
        if (button.is_visible() and button.is_enabled()
                and name in button.inner_text()):
            button.click()
            return
    raise AssertionError(f"no enabled visible MVTec card for {name!r}")


def _wait_viewer_case(page, expected: str, timeout: int = 30000) -> None:
    """Wait until the viewer identifies the card selected in this context."""
    viewer = page.locator(".st-key-viz_image_viewer")
    deadline = time.time() + timeout / 1000
    while time.time() < deadline:
        if viewer.count() and expected in viewer.inner_text():
            return
        page.wait_for_timeout(150)
    text = viewer.inner_text() if viewer.count() else "<viewer missing>"
    raise AssertionError(
        f"viewer did not identify expected MVTec case {expected!r}: {text[:1000]}"
    )


def _click_card_for_class(page, label: str, filename: str) -> None:
    """Pick a card by both label and file name: MVTec reuses names per class."""
    buttons = page.locator('.st-key-viz_grid [class*="st-key-viz_card_"] button')
    for i in range(buttons.count() - 1, -1, -1):
        button = buttons.nth(i)
        if not (button.is_visible() and button.is_enabled()):
            continue
        text = button.inner_text()
        if label in text and filename in text:
            button.click()
            return
    raise AssertionError(f"no enabled card for {label}/{filename}")


def _artifact_subdir(name: str) -> Path | None:
    root = os.environ.get("LV_E2E_ARTIFACT_DIR", "").strip()
    if not root:
        return None
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_real_mvtec_five_case_explainability(
    explain_app_server, browser, mvtec_bad_dataset, e2e_cache_dir,
):
    source, dataset = mvtec_bad_dataset
    source_before = _fingerprint(source)
    cache_baseline: str | None = None
    scores: list[int] = []
    artifact_root = os.environ.get("LV_E2E_ARTIFACT_DIR", "").strip()
    artifact_dir = Path(artifact_root) if artifact_root else None
    record_trace = os.environ.get("LV_E2E_RECORD_TRACE") == "1"
    record_video = os.environ.get("LV_E2E_RECORD_VIDEO") == "1"
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    case_results: list[dict[str, object]] = []

    # Each image gets a fresh browser context/session.  Streamlit reruns can
    # retain stale card callbacks in one session; isolating cases keeps the
    # selection assertion about the requested filename, while still exercising
    # the production UI and the same on-demand analysis path.
    for expected in _CASE_NAMES:
        case_slug = Path(expected).stem
        case_artifact_dir = artifact_dir / case_slug if artifact_dir else None
        if case_artifact_dir:
            case_artifact_dir.mkdir(parents=True, exist_ok=True)
        context_options = {"viewport": {"width": 1920, "height": 1080}}
        if case_artifact_dir and record_video:
            context_options["record_video_dir"] = str(case_artifact_dir / "video")
        ctx = browser.new_context(**context_options)
        page = ctx.new_page()
        page.set_default_timeout(30000)
        tracing_started = False
        try:
            if case_artifact_dir and record_trace:
                ctx.tracing.start(screenshots=True, snapshots=True, sources=True)
                tracing_started = True
            _run_visualize(page, explain_app_server, dataset)
            if cache_baseline is None:
                # The normal Visualize run is allowed to materialize its
                # embedding cache.  Explainability itself must not mutate it.
                cache_baseline = _fingerprint(e2e_cache_dir)
            else:
                assert _fingerprint(e2e_cache_dir) == cache_baseline, \
                    f"repeat Visualize changed LV cache for {expected}"
            cards = page.locator('.st-key-viz_grid [class*="st-key-viz_card_"] button')
            visible_cards = [
                cards.nth(i) for i in range(cards.count()) if cards.nth(i).is_visible()
            ]
            assert len(visible_cards) >= len(_CASE_NAMES), \
                f"visible cards={len(visible_cards)}"
            _click_card(page, expected)
            wait_idle(page)
            _wait_viewer_case(page, expected)
            viewer = page.locator(".st-key-viz_image_viewer")
            expect(viewer.locator('[data-testid="stImage"] img').first).to_be_visible()
            if case_artifact_dir:
                page.screenshot(
                    path=str(case_artifact_dir / "01-selected.png"), full_page=True
                )
            score = 20  # selected real image is visible

            summary = viewer.locator("details summary").first
            summary.wait_for(state="visible")
            summary.scroll_into_view_if_needed()
            if not summary.evaluate("el => el.parentElement.open"):
                summary.click()
            wait_idle(page)
            _click_key_button(page, "viz_explain_analyze")
            _visible_key_button(page, "viz_explain_dl_png", timeout=180000)
            _visible_key_button(page, "viz_explain_dl_json", timeout=30000)
            score += 30  # analysis and both exports are present
            expect(viewer.locator('[data-testid="stImage"] img')).to_have_count(4)
            score += 20  # original, input, overlay, and the viewer image render
            wait_idle(page, timeout=120000)
            expect(page.locator('[data-testid="stException"]')).to_have_count(0)
            score += 15  # no Streamlit exception
            assert _fingerprint(e2e_cache_dir) == cache_baseline, \
                f"single-image Attention changed LV cache for {expected}"
            score += 15  # cache invariant for this case
            scores.append(score)
            if case_artifact_dir:
                page.screenshot(
                    path=str(case_artifact_dir / "02-analyzed.png"), full_page=True
                )
            case_results.append({"case": expected, "score": score})
            print(f"MVTec case {expected}: {score}/100")
        finally:
            if case_artifact_dir and tracing_started:
                ctx.tracing.stop(path=str(case_artifact_dir / "trace.zip"))
            ctx.close()

    assert _fingerprint(source) == source_before, \
        "real MVTec source image changed"
    mean = sum(scores) / len(scores)
    print(f"MVTec explainability mean: {mean:.2f}/100")
    assert mean > 90, f"MVTec explainability mean {mean:.2f} must exceed 90"
    if artifact_dir:
        (artifact_dir / "case-results.json").write_text(
            json.dumps(
                {
                    "dataset": str(source),
                    "cases": case_results,
                    "mean": mean,
                    "score_threshold": 90,
                    "source_unchanged": _fingerprint(source) == source_before,
                    "cache_unchanged_after_analysis": (
                        _fingerprint(e2e_cache_dir) == cache_baseline
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def test_real_mvtec_capsule_gradcam(
    explain_app_server, browser, mvtec_capsule_gradcam_dataset, e2e_cache_dir,
):
    """Real images must reach a session head and the selected Grad-CAM target."""
    source, dataset, selected_name = mvtec_capsule_gradcam_dataset
    source_before = _fingerprint(source)
    dataset_before = _fingerprint(dataset)
    artifact_dir = _artifact_subdir("capsule-gradcam")
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    try:
        _run_visualize(page, explain_app_server, dataset)
        cache_before_explain = _fingerprint(e2e_cache_dir)

        _click_card_for_class(page, _CAPSULE_TARGET, selected_name)
        wait_idle(page)
        viewer = page.locator(".st-key-viz_image_viewer")
        expect(viewer.locator('[data-testid="stImage"] img').first).to_be_visible()
        assert selected_name in viewer.inner_text()

        viewer.scroll_into_view_if_needed()
        _open_expander(page, "可解釋性")
        page.locator(".st-key-viz_explain_mode").locator(
            "text=分類理由"
        ).last.click()
        wait_idle(page)
        _open_expander(page, "可解釋性")
        _click_key_button(page, "viz_explain_train_head")
        wait_idle(page, timeout=120000)
        expect(page.get_by_text("session head", exact=False).first).to_be_visible()

        _open_expander(page, "可解釋性")
        target_box = page.locator(
            ".st-key-viz_explain_target [data-baseweb='select']"
        ).first
        target_box.click()
        page.get_by_role("option", name=_CAPSULE_TARGET, exact=True).click()
        page.keyboard.press("Escape")
        wait_idle(page)
        _click_key_button(page, "viz_explain_analyze")
        page.wait_for_selector("text=對比分類證據", timeout=180000)
        wait_idle(page, timeout=120000)

        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
        expect(page.get_by_text("對比分類證據", exact=False).first).to_be_visible()
        expect(
            page.get_by_text(
                re.compile(rf"(?:實際決策|反事實假說).*{_CAPSULE_TARGET}"), exact=False,
            ).first
        ).to_be_visible()
        expect(viewer.locator('[data-testid="stImage"] img')).to_have_count(4)
        _visible_key_button(page, "viz_explain_dl_png")
        _visible_key_button(page, "viz_explain_dl_json")
        assert _fingerprint(e2e_cache_dir) == cache_before_explain, \
            "real MVTec Grad-CAM must not write LV cache"
        assert _fingerprint(dataset) == dataset_before, \
            "real MVTec Grad-CAM must not write the copied classifier dataset"

        if artifact_dir:
            page.screenshot(
                path=str(artifact_dir / "01-gradcam-scratch.png"), full_page=True
            )
            (artifact_dir / "result.json").write_text(
                json.dumps(
                    {
                        "source_dataset": str(source),
                        "copied_dataset": str(dataset),
                        "classes": list(_CAPSULE_CLASSES),
                        "samples_per_class": _CAPSULE_SAMPLES_PER_CLASS,
                        "selected_class": _CAPSULE_TARGET,
                        "selected_file": selected_name,
                        "target": _CAPSULE_TARGET,
                        "source_unchanged": _fingerprint(source) == source_before,
                        "cache_unchanged_after_gradcam": (
                            _fingerprint(e2e_cache_dir) == cache_before_explain
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    finally:
        ctx.close()

    assert _fingerprint(source) == source_before, "original MVTec capsule data changed"
