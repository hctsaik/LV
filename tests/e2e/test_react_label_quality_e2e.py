from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pytest
from PIL import Image

from react_web_shell import create_app
from .test_react_web_shell_e2e import (
    _free_port,
    _pinned_dinov2_models_dir,
    _start_api,
    _start_streamlit_for_react_parity,
    _worker_fixture_model,
)

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]


def _label_quality_fixture(root: Path) -> Path:
    train = root / "train"
    specs = [("apple", (220 - index % 5, 35 + index % 6, 45 + index % 4)) for index in range(12)]
    # Deliberately mislabeled: its image features match the banana group.
    specs.append(("apple", (30, 190, 70)))
    specs.extend(
        ("banana", (30 + index % 5, 190 + index % 6, 70 + index % 4))
        for index in range(10)
    )
    for index, (label, color) in enumerate(specs):
        folder = train / label
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color).save(folder / f"img-{index:02}.png")
    return train


def _source_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def _add_visible_streamlit_folder(page, path: Path) -> None:
    """Enter a folder through the visible Streamlit control after transient reruns."""
    selector = ".st-key-viz_folder_list_add input"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        inputs = page.locator(selector)
        for index in range(inputs.count() - 1, -1, -1):
            candidate = inputs.nth(index)
            try:
                if not candidate.is_visible(timeout=250) or not candidate.is_enabled(timeout=250):
                    continue
                candidate.fill(str(path))
                candidate.press("Enter")
                page.wait_for_function(
                    "([s, p]) => Array.from(document.querySelectorAll(s))"
                    ".filter(el => el.offsetParent !== null)"
                    ".every(el => el.value !== p)",
                    arg=[selector, str(path)],
                    timeout=15000,
                )
                from .conftest import wait_idle

                wait_idle(page)
                from playwright.sync_api import expect

                expect(page.get_by_text(str(path), exact=True).last).to_be_visible(timeout=8000)
                return
            except Exception:
                continue
        page.wait_for_timeout(100)
    raise AssertionError(f"no stable visible Streamlit folder input for {path}")


def test_lv_web_01_lq_01_visible_confusion_and_mislabel_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The React label-quality screen finds a planted label issue from raw features."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the React workspace with npm run build before this E2E")

    train = _label_quality_fixture(tmp_path / "label-quality-fixture")
    before = _source_hashes(train)
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=_worker_fixture_model,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1450, "height": 1050})
            page.set_default_timeout(30000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("model-choice-fixture-model").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            page.get_by_label("Image folder").fill(str(train))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=30000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(23)

            page.get_by_test_id("open-label-quality").click()
            expect(page.get_by_test_id("label-quality-workspace")).to_be_visible()
            expect(page.get_by_label("Label quality K")).to_have_value("10")
            page.get_by_test_id("label-quality-calculate").click()
            expect(page.get_by_test_id("label-quality-confusion-matrix")).to_be_visible()
            expect(page.get_by_test_id("label-quality-summary")).to_contain_text("1 suspected label issue")
            candidate = page.get_by_test_id("label-quality-candidate")
            expect(candidate).to_have_count(1)
            expect(candidate).to_contain_text("apple → banana")
            expect(candidate.get_by_role("img", name="img-12.png")).to_be_visible()

            page.get_by_test_id("label-quality-select-candidates").click()
            expect(page.get_by_role("status")).to_contain_text("Added 1 suspected record(s)")
            page.get_by_role("button", name="Return to Visualize").click()
            expect(page.get_by_test_id("selection-count")).to_have_text("1 selected")
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert _source_hashes(train) == before


def test_lv_web_01_lq_01_visible_streamlit_react_real_model_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Run the same pinned-model label-quality journey in both visible UIs."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the React workspace with npm run build before this E2E")

    import base64
    import numpy as np
    from _utils import available_models, load_model
    from .conftest import _ensure_sidebar, _stop_process_tree, load_app, wait_idle

    models_dir = _pinned_dinov2_models_dir(tmp_path)
    train = _label_quality_fixture(tmp_path / "shared-label-quality-fixture")
    before = _source_hashes(train)
    cache_dir = tmp_path / "shared-cache"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache_dir))
    application = create_app(
        model_catalog=lambda: available_models(models_dir),
        model_loader=lambda model_id: load_model(model_id, models_dir=models_dir),
    )
    server = server_thread = legacy_process = legacy_log = browser = None
    try:
        server, server_thread, base_url = _start_api(application)
        parity_root = tmp_path / "streamlit-reference"
        parity_root.mkdir()
        legacy_process, legacy_log, legacy_url = _start_streamlit_for_react_parity(
            models_dir, parity_root, cache_dir=cache_dir,
        )

        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            react_page = browser.new_page(viewport={"width": 1500, "height": 1100})
            react_page.set_default_timeout(30000)
            label_quality_payloads = []

            def capture_label_quality(response):
                if response.ok and "/label-quality" in response.url:
                    label_quality_payloads.append(response.json())

            react_page.on("response", capture_label_quality)
            react_page.goto(base_url, wait_until="networkidle")
            react_page.get_by_test_id("model-choice-dinov2_vits14").check()
            react_page.get_by_label("Supervised UMAP").uncheck()
            react_page.get_by_label("PCA").check()
            react_page.get_by_label("Image folder").fill(str(train))
            react_page.get_by_role("button", name="Add folder").click()
            react_page.get_by_role("button", name="Run analysis").click()
            expect(react_page.get_by_test_id("job-status")).to_contain_text(
                "Completed", timeout=300000,
            )
            expect(react_page.get_by_test_id("embedding-point")).to_have_count(23)
            react_page.get_by_test_id("open-label-quality").click()
            react_page.get_by_test_id("label-quality-calculate").click()
            expect(react_page.get_by_test_id("label-quality-confusion-matrix")).to_be_visible()
            assert label_quality_payloads, "React did not request label-quality results"
            react_result = label_quality_payloads[-1]
            assert react_result["model_id"] == "dinov2_vits14"
            assert react_result["k"] == 10
            assert react_result["record_count"] == 23
            react_candidates = react_page.get_by_test_id("label-quality-candidate").all_inner_texts()
            assert len(react_candidates) == len(react_result["candidates"])
            for candidate, visible_candidate in zip(
                react_result["candidates"], react_candidates, strict=True,
            ):
                assert f"{candidate['own']} → {candidate['suggested']}" in visible_candidate
                assert candidate["filename"] in visible_candidate
                suggested_pct = round(candidate["suggested_frac"] * 100)
                assert f"Suggested class in {suggested_pct}% of neighbors" in visible_candidate

            react_matrix = react_page.get_by_test_id("label-quality-confusion-matrix")
            react_visible_confusion = [
                [float(value.removesuffix("%")) for value in row.locator("td").all_text_contents()]
                for row in react_matrix.locator("tbody tr").all()
            ]
            np.testing.assert_array_equal(
                np.asarray(react_visible_confusion),
                np.round(np.asarray(react_result["confusion"]) * 100, 1),
            )

            legacy_page = browser.new_page(viewport={"width": 1920, "height": 1080})
            legacy_page.set_default_timeout(30000)
            load_app(legacy_page, legacy_url)
            legacy_page.locator(".st-key-viz_mode").get_by_text("Image Classifier").click()
            wait_idle(legacy_page)
            _add_visible_streamlit_folder(legacy_page, train)
            methods = legacy_page.locator(".st-key-viz_methods")
            methods.get_by_role("button", name="Clear all").click()
            wait_idle(legacy_page)
            methods.locator("[data-baseweb='select']").click()
            legacy_page.get_by_role("option", name="PCA", exact=True).click()
            legacy_page.keyboard.press("Escape")
            wait_idle(legacy_page)
            _ensure_sidebar(legacy_page)
            legacy_page.locator(".st-key-run_viz button").click()
            legacy_page.wait_for_selector(
                ".st-key-viz_scatter_wrap g.points path", timeout=300000,
            )
            wait_idle(legacy_page, timeout=60000)

            expander = legacy_page.locator(
                "details summary", has_text="標錯偵測 / 類別混淆",
            ).first
            expander.click()
            legacy_page.get_by_role("tab", name="🩹 疑似標錯").click()
            legacy_visible_text = legacy_page.locator("[data-testid='stMain']").inner_text()
            expected_candidates = react_result["candidates"]
            if expected_candidates:
                assert f"共 {len(expected_candidates)} 個物件" in legacy_visible_text
                for candidate in expected_candidates:
                    pair = f"{candidate['own']} → {candidate['suggested']}"
                    pct = f"({round(candidate['suggested_frac'] * 100):.0f}%)"
                    assert pair in legacy_visible_text, pair
                    assert pct in legacy_visible_text, (pair, pct)
            else:
                assert "沒有明顯的疑似標錯" in legacy_visible_text

            legacy_page.get_by_role("tab", name="🔀 類別混淆 / 分離度").click()
            legacy_plot = legacy_page.locator(
                ".st-key-conf_hm_dinov2_vits14 .js-plotly-plot",
            )
            legacy_plot.wait_for(state="visible")
            streamlit_visible_confusion = legacy_plot.evaluate("plot => plot.data[0].z")
            if isinstance(streamlit_visible_confusion, dict) and "bdata" in streamlit_visible_confusion:
                encoded = base64.b64decode(streamlit_visible_confusion["bdata"])
                values = np.frombuffer(encoded, dtype=streamlit_visible_confusion["dtype"])
                shape = tuple(
                    int(value) for value in streamlit_visible_confusion.get("shape", "").split(",")
                    if value
                )
                streamlit_visible_confusion = values.reshape(shape)
            np.testing.assert_array_equal(
                np.asarray(streamlit_visible_confusion, dtype=float),
                np.round(np.asarray(react_result["confusion"]) * 100, 1),
            )
            assert _source_hashes(train) == before
            browser.close()
            browser = None
    finally:
        if legacy_process is not None:
            _stop_process_tree(legacy_process)
        if legacy_log is not None:
            legacy_log.close()
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)
    assert _source_hashes(train) == before
