"""Visible LV-WEB-01-IC-01 acceptance against a real DINOv2 model."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import uvicorn

from _utils import available_models, load_model
from react_compare import CompareCancelled
from react_web_shell import create_app

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
DINOV2_VITS14_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
FIXTURE_SHA256 = {
    "train/apple/image-01.png": "be4582401ec897d3d653f2f52275802d4485ab5106e0e6125344711a3d32b6a3",
    "train/apple/image-02.png": "ed47d5fdeb3b4ad1fb5ff833846ffab6b8fc690d96f5d44a348196b0c1f5d731",
    "train/banana/image-03.png": "1ed14b03efb7653fadc8bc56445702b7ea39ae67974957dee0d4f3c5ad04f0dd",
    "test/banana/image-04.png": "246ec8e9c65eda52b30d0da34ccd0c35294e6575299430218d65f2803d990060",
}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_http(url: str, server_thread: threading.Thread, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not server_thread.is_alive():
            pytest.fail(f"local API server exited early: {url}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status < 500:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.2)
    pytest.fail(f"service did not become ready: {url}")


def _fixture(root: Path) -> Path:
    colors = [(220, 35, 45), (30, 190, 70), (35, 65, 215), (210, 180, 25)]
    for i, color in enumerate(colors):
        folder = root / ("test" if i == 3 else "train") / ("apple" if i < 2 else "banana")
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color).save(folder / f"image-{i + 1:02}.png")
    actual = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*.png")
    }
    assert actual == FIXTURE_SHA256
    return root


def _start_api(application):
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(
        application, host="127.0.0.1", port=port, log_level="error", lifespan="off",
    ))
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{port}"
    _wait_http(f"{base_url}/api/v1/health", server_thread)
    return server, server_thread, base_url


def test_lv_web_01_ic_01_ic_02_real_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """AC2/AC3: real DINOv2 vectors, projection switching, 3D view and image readback."""
    models_dir = Path(os.environ.get("LV_MODELS_DIR", ROOT / "models")).resolve()
    weights = models_dir / "dinov2_vits14" / "dinov2_vits14.pth"
    if not weights.is_file():
        pytest.skip(f"real DINOv2-small weights are unavailable: {weights}")
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == DINOV2_VITS14_SHA256, \
        "the local model weights differ from the LV-WEB-01-IC-01 baseline"
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    dataset = _fixture(tmp_path / "classifier-fixture")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: available_models(models_dir),
        model_loader=lambda model_id: load_model(model_id, models_dir=models_dir),
    )
    server = None
    server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)

        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            run_payloads = []

            def capture_run(response):
                if response.request.method == "GET" and "/api/v1/runs/" in response.url and "/images/" not in response.url and response.ok:
                    run_payloads.append(response.json())

            page.on("response", capture_run)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("model-choice-dinov2_vits14").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            page.get_by_label("t-SNE").check()
            page.get_by_label("LDA (supervised)").check()
            page.get_by_label("Image folder").fill(str(dataset / "train"))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_label("Image folder").fill(str(dataset / "test"))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text(
                "Completed", timeout=180_000
            )
            points = page.get_by_test_id("embedding-point")
            expect(points).to_have_count(4)
            assert run_payloads, "React UI did not load a run result from the local API"
            assert run_payloads[-1]["embedding_shape"] == [4, 384]
            assert run_payloads[-1]["embedding_finite"] is True
            assert run_payloads[-1]["embedding_nonzero"] is True
            assert {(view["model_id"], view["method"]) for view in run_payloads[-1]["views"]} >= {
                ("dinov2_vits14", "pca"),
                ("dinov2_vits14", "tsne"),
                ("dinov2_vits14", "lda"),
            }
            page.get_by_label("Filter by class").select_option(label="apple")
            expect(page.get_by_test_id("embedding-point")).to_have_count(2)
            page.get_by_label("Filter by class").select_option(label="All classes")
            page.get_by_label("Filter by split").select_option(label="test")
            expect(page.get_by_test_id("embedding-point")).to_have_count(1)
            page.get_by_label("Filter by split").select_option(label="All splits")
            page.get_by_test_id("embedding-point").nth(2).click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text("image-03.png")
            page.get_by_label("Display method").select_option("tsne")
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)
            expect(page.get_by_test_id("selected-image-name")).to_have_text("image-03.png")
            page.get_by_label("Display method").select_option("lda")
            expect(page.get_by_test_id("selected-image-name")).to_have_text("image-03.png")
            page.get_by_test_id("dimension-3d").click()
            expect(page.get_by_test_id("embedding-plot-3d")).to_be_visible(timeout=60_000)
            expect(page.locator("[data-testid='embedding-plot-3d'] .js-plotly-plot")).to_be_attached(timeout=60_000)
            points = page.get_by_test_id("embedding-point")
            expect(points).to_have_count(4, timeout=60_000)
            points.nth(3).focus()
            points.nth(3).press("Enter")
            expect(page.get_by_test_id("selected-image-name")).to_have_text("image-04.png")
            page.get_by_test_id("dimension-2d").click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text("image-04.png")
            image = page.get_by_test_id("selected-image")
            expect(image).to_be_visible()
            expect(image).to_have_js_property("naturalWidth", 64)

            for folder in (dataset / "train", dataset / "test"):
                page.get_by_role("button", name=f"Remove {folder}").click()
            empty_folder = tmp_path / "empty-folder"
            empty_folder.mkdir()
            page.get_by_label("Image folder").fill(str(empty_folder))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_role("alert")).to_contain_text("No images were found", timeout=10_000)
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)


def test_lv_web_01_ic_03_fixed_umap_frame_keeps_old_images_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The React control persists a real UMAP reducer and preserves old image coordinates."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")
    dataset = _fixture(tmp_path / "umap-reference-fixture")
    pre_existing_image = dataset / "train" / "cherry" / "pre-existing-image.png"
    pre_existing_image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (60, 90, 170)).save(pre_existing_image)
    cache_dir = tmp_path / "lv-app-cache"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache_dir))

    def model_loader(_model_id: str):
        def embed(path: Path) -> np.ndarray:
            pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
            vector[3] = len(path.name) / 10.0
            return vector
        return embed

    application = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=model_loader,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            run_payloads: list[dict] = []

            def capture_run(response):
                if response.request.method == "GET" and "/api/v1/runs/" in response.url and "/images/" not in response.url and response.ok:
                    run_payloads.append(response.json())

            page.on("response", capture_run)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("model-choice-dinov2_vits14").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_test_id("projection-umap").check()
            page.get_by_label("Keep UMAP reference frame").check()
            for folder in (dataset / "train", dataset / "test"):
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()

            def run_and_wait(expected_count: int) -> dict:
                page.get_by_role("button", name="Run analysis").click()
                expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=180_000)
                deadline = time.monotonic() + 20
                while len(run_payloads) < expected_count and time.monotonic() < deadline:
                    time.sleep(0.05)
                assert len(run_payloads) == expected_count
                return run_payloads[-1]

            first_run = run_and_wait(1)
            first_points = {
                point["filename"]: (point["x"], point["y"], point["z"])
                for point in first_run["points"]
            }
            expect(page.get_by_text("Fixed UMAP reference rebuilt", exact=False)).to_be_visible()

            second_run = run_and_wait(2)
            second_points = {
                point["filename"]: (point["x"], point["y"], point["z"])
                for point in second_run["points"]
            }
            assert first_points == second_points
            expect(page.get_by_text("Fixed UMAP reference reused (+0 new images)", exact=False)).to_be_visible()

            new_image = dataset / "train" / "cherry" / "reference-added.png"
            new_image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), (25, 140, 200)).save(new_image)
            grown_run = run_and_wait(3)
            grown_points = {
                point["filename"]: (point["x"], point["y"], point["z"])
                for point in grown_run["points"]
            }
            assert len(grown_points) == 6
            assert {name: grown_points[name] for name in first_points} == first_points
            expect(page.get_by_text("Fixed UMAP reference reused (+1 new images)", exact=False)).to_be_visible()

            page.get_by_role("button", name="Rebuild UMAP reference").click()
            expect(page.get_by_role("button", name="Rebuild queued for next run")).to_be_visible()
            rebuilt_run = run_and_wait(4)
            assert any("Fixed UMAP reference rebuilt" in warning for warning in rebuilt_run["warnings"])
            assert len(list(cache_dir.rglob("umap_ref.pkl"))) == 1
            assert not list(dataset.rglob("umap_ref.pkl"))
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)


def test_lv_web_01_ic_01_missing_model_is_visible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """AC4: a missing local model is presented as a recoverable UI state."""
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(model_catalog=lambda: [], model_loader=lambda _model_id: None)
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(base_url, wait_until="networkidle")
            expect(page.get_by_text("Install a supported LV model to enable analysis.")).to_be_visible()
            expect(page.get_by_role("button", name="Run analysis")).to_be_disabled()
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)


def test_lv_web_01_ic_01_cancel_is_visible_in_react(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """AC4: the visible UI can cancel a running job and shows its terminal state."""
    dataset = _fixture(tmp_path / "classifier-fixture")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    loader_started, release_loader = threading.Event(), threading.Event()

    def slow_loader(_model_id: str):
        loader_started.set()
        release_loader.wait(timeout=10)
        return lambda _path: [1.0] * 384

    application = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=slow_loader,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Image folder").fill(str(dataset / "train"))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_label("Image folder").fill(str(dataset / "test"))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            assert loader_started.wait(timeout=10), "worker did not start the model-loading stage"
            page.get_by_role("button", name="Cancel analysis").click()
            release_loader.set()
            expect(page.get_by_test_id("job-status")).to_contain_text("Cancelled", timeout=10_000)
            browser.close()
    finally:
        release_loader.set()
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)


def test_lv_web_01_ic_04_kmeans_block_selection_round_trips_across_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The React 3D block labels drill into 2D selection and keep the prior selection."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")
    dataset = _fixture(tmp_path / "kmeans-block-fixture")
    colors = [
        (210, 35, 45), (200, 45, 55), (220, 30, 40),
        (35, 190, 70), (45, 200, 60), (30, 180, 80),
        (35, 65, 215), (45, 75, 205), (25, 55, 225),
        (210, 180, 25), (220, 170, 35), (200, 190, 15),
    ]
    for index, color in enumerate(colors, start=5):
        folder = dataset / ("test" if index % 4 == 0 else "train") / ("apple" if index % 2 else "banana")
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color).save(folder / f"block-image-{index:02}.png")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "lv-app-cache"))

    def model_loader(_model_id: str):
        def embed(path: Path) -> np.ndarray:
            pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
            vector[3] = len(path.name) / 10.0
            return vector
        return embed

    application = create_app(model_catalog=lambda: ["dinov2_vits14"], model_loader=model_loader)
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            block_responses = []

            def capture_blocks(response):
                if response.url.endswith("/blocks"):
                    block_responses.append(response)

            page.on("response", capture_blocks)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("model-choice-dinov2_vits14").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in (dataset / "train", dataset / "test"):
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=180_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(16)

            page.get_by_test_id("embedding-point").nth(0).click()
            expect(page.get_by_test_id("selection-summary")).to_have_text("Selected 1 images")
            page.get_by_test_id("dimension-3d").click()
            expect(page.get_by_test_id("block-choice")).to_be_visible(timeout=60_000)
            deadline = time.monotonic() + 20
            while not block_responses and time.monotonic() < deadline:
                time.sleep(0.05)
            assert block_responses, [
                (response.status, response.url) for response in block_responses
            ]
            assert block_responses[-1].ok, block_responses[-1].text()
            blocks_result = block_responses[-1].json()
            plot3d = page.locator("[data-testid='embedding-plot-3d'] .js-plotly-plot")
            expect(plot3d).to_be_visible()
            label_count = plot3d.evaluate("el => el._fullLayout.scene.annotations.length")
            assert label_count == blocks_result["effective_k"]
            expect(page.get_by_test_id("selection-summary")).to_have_text("Selected 1 images")

            selected_block = next(
                block for block in blocks_result["blocks"]
                if 0 not in block["indices"] and block["count"] > 1
            )
            page.get_by_test_id("block-choice").select_option(str(selected_block["block_id"]))
            expect(page.get_by_test_id("block-selection-plot")).to_be_visible(timeout=60_000)
            plot2d = page.locator("[data-testid='block-selection-plot'] .js-plotly-plot")
            plot2d.scroll_into_view_if_needed()
            page.wait_for_timeout(150)
            box = plot2d.bounding_box()
            assert box is not None
            page.mouse.move(box["x"] + 55, box["y"] + 60)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] - 35, box["y"] + box["height"] - 45, steps=8)
            page.mouse.up()
            expect(page.get_by_test_id("selection-summary")).to_have_text("Selected 2 images", timeout=10_000)
            plot2d.locator('.modebar-btn[data-title="Lasso Select"]').first.click()
            drag = plot2d.locator(".nsewdrag")
            drag_box = drag.bounding_box()
            assert drag_box is not None
            lasso_path = [
                (drag_box["x"] + 2, drag_box["y"] + 2),
                (drag_box["x"] + drag_box["width"] - 2, drag_box["y"] + 2),
                (drag_box["x"] + drag_box["width"] - 2, drag_box["y"] + drag_box["height"] - 2),
                (drag_box["x"] + 2, drag_box["y"] + drag_box["height"] - 2),
                (drag_box["x"] + 2, drag_box["y"] + 2),
            ]
            page.mouse.move(*lasso_path[0])
            page.mouse.down()
            for x, y in lasso_path[1:]:
                page.mouse.move(x, y, steps=8)
            page.mouse.up()
            expect(page.get_by_test_id("selection-summary")).to_have_text(
                f"Selected {selected_block['count'] + 1} images", timeout=10_000
            )
            page.get_by_test_id("dimension-2d").click()
            expect(page.get_by_test_id("selection-summary")).to_have_text(
                f"Selected {selected_block['count'] + 1} images"
            )
            page.get_by_role("button", name="Clear selection").click()
            expect(page.get_by_test_id("selection-summary")).to_have_text("Selected 0 images")
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)


def test_lv_web_02_img_01_visible_compare_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Visible React whole-image A/B controls, projections and image readback.

    The local feature function is deterministic. This is API/UI wiring coverage,
    not evidence of real-model or Streamlit numerical parity.
    """
    import numpy as np
    from playwright.sync_api import expect, sync_playwright
    from react_compare import run_image_comparison

    folders = {
        name: tmp_path / name for name in ("A1", "A2", "B1", "B2")
    }
    source_bytes = {}
    for group_index, (name, color) in enumerate((
        ("A1", (220, 30, 45)), ("A2", (200, 50, 60)),
        ("B1", (30, 80, 220)), ("B2", (50, 100, 200)),
    )):
        folders[name].mkdir()
        for index in range(2):
            path = folders[name] / f"image-{index}.png"
            pixel = tuple(min(255, channel + index * 7 + group_index) for channel in color)
            Image.new("RGB", (24, 18), pixel).save(path)
            source_bytes[path] = path.read_bytes()

    slow_mode = {"enabled": False}
    compare_started = threading.Event()

    def embed(path: Path):
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        mean = image.mean(axis=(0, 1)) / 255.0
        return np.r_[mean, mean.mean(), 1.0].astype(np.float32)

    def comparison_runner(folders_a, folders_b, model_id, *, progress=None, cancelled=None):
        if slow_mode["enabled"]:
            compare_started.set()
            while not cancelled():
                time.sleep(0.01)
            raise CompareCancelled()
        return run_image_comparison(
            folders_a, folders_b, model_id,
            model_loader=lambda _model_id: embed,
            progress=progress,
            cancelled=cancelled,
        )

    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: ["contract-test-model"],
        comparison_runner=comparison_runner,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(base_url, wait_until="networkidle")
            page.get_by_role("button", name="Compare", exact=True).click()
            expect(page.get_by_role("heading", name="Compare Distributions")).to_be_visible()

            for group, names in (("A", ("A1", "A2")), ("B", ("B1", "B2"))):
                section = page.locator(f'.compare-folder-group[aria-label="Folder {group}"]')
                for name in names:
                    page.get_by_label(f"Folder {group} path").fill(str(folders[name]))
                    section.get_by_role("button", name="Add").click()
                expect(section.locator(".compare-folder-list li")).to_have_count(2)

            page.get_by_label("Compare model").select_option("contract-test-model")
            page.get_by_role("button", name="Run comparison").click()
            expect(page.get_by_test_id("compare-drift")).to_be_visible(timeout=20_000)
            expect(page.locator(".drift-count strong").nth(0)).to_have_text("4")
            expect(page.locator(".drift-count strong").nth(1)).to_have_text("4")
            page.get_by_label("Compare projection method").select_option("pca")
            expect(page.get_by_test_id("compare-point")).to_have_count(8, timeout=10_000)

            page.get_by_test_id("compare-point").first.click()
            expect(page.locator(".compare-selection-summary")).to_contain_text("Selected 1 from A")
            preview_a = page.get_by_test_id("compare-image-a-0")
            preview_b = page.get_by_test_id("compare-image-b-0")
            expect(preview_a).to_be_visible()
            expect(preview_a).to_have_js_property("naturalWidth", 24)
            expect(preview_b).to_have_js_property("naturalWidth", 24)

            plot = page.locator("svg.compare-scatter")
            box = plot.bounding_box()
            assert box is not None
            page.mouse.move(box["x"] + 10, box["y"] + 10)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] - 10, box["y"] + box["height"] - 10, steps=8)
            page.mouse.up()
            expect(page.locator(".compare-selection-summary")).to_contain_text(
                "Selected 4 from A · 4 from B", timeout=5_000
            )

            page.get_by_role("group", name="Compare projection dimension").get_by_role("button", name="3D").click()
            expect(page.get_by_test_id("embedding-plot-3d")).to_be_visible(timeout=15_000)
            page.get_by_role("group", name="Compare projection dimension").get_by_role("button", name="2D").click()
            expect(page.get_by_test_id("compare-plot")).to_be_visible()

            slow_mode["enabled"] = True
            page.get_by_role("button", name="Run comparison").click()
            assert compare_started.wait(timeout=5)
            page.get_by_role("button", name="Cancel comparison").click()
            expect(page.get_by_text(
                "Comparison cancelled. Adjust the folders and run it again.", exact=True
            )).to_be_visible(timeout=10_000)
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert all(path.read_bytes() == content for path, content in source_bytes.items())


def test_lv_web_01_search_01_visible_chain_cross_model_and_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """SEARCH-01: visible search sorts, pivots, recomputes per model, and exports."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")
    from playwright.sync_api import expect, sync_playwright

    folder = tmp_path / "train" / "classA"
    folder.mkdir(parents=True)
    filenames = ["a_query.png", "b_near_a.png", "c_near_b.png", "d_far.png"]
    colors = [(220, 35, 45), (210, 40, 50), (35, 65, 215), (30, 200, 70)]
    source_bytes = {}
    for filename, color in zip(filenames, colors):
        image_path = folder / filename
        Image.new("RGB", (32, 32), color).save(image_path)
        source_bytes[image_path] = image_path.read_bytes()

    vectors = {
        "model_a": {
            "a_query.png": [1.0, 0.0, 0.0],
            "b_near_a.png": [0.99, 0.1, 0.0],
            "c_near_b.png": [0.0, 1.0, 0.0],
            "d_far.png": [0.0, 0.0, 1.0],
        },
        "model_b": {
            "a_query.png": [1.0, 0.0, 0.0],
            "b_near_a.png": [0.0, 1.0, 0.0],
            "c_near_b.png": [0.99, 0.1, 0.0],
            "d_far.png": [0.0, 0.0, 1.0],
        },
    }

    def model_loader(model_id: str):
        return lambda path: np.asarray(vectors[model_id][path.name], dtype=np.float32)

    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "lv-cache"))
    application = create_app(
        model_catalog=lambda: ["model_a", "model_b"],
        model_loader=model_loader,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            run_payloads = []
            search_payloads = []

            def capture_response(response):
                if ("/api/v1/runs/" in response.url and "/images/" not in response.url
                        and not response.url.endswith(".csv") and response.ok):
                    if response.request.method == "POST" and response.url.endswith("/similar"):
                        search_payloads.append(response.json())
                    elif response.request.method == "GET":
                        run_payloads.append(response.json())

            page.on("response", capture_response)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("model-choice-model_b").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            page.get_by_label("Image folder").fill(str(tmp_path / "train"))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)
            assert run_payloads
            run = run_payloads[-1]
            query_index = next(row["index"] for row in run["records"] if row["filename"] == "a_query.png")
            pivot_index = next(row["index"] for row in run["records"] if row["filename"] == "b_near_a.png")

            page.get_by_test_id("embedding-point").nth(query_index).click()
            page.get_by_test_id("find-similar").click()
            expect(page.get_by_test_id("similar-results")).to_be_visible()
            expect(page.get_by_test_id("similar-result")).to_have_count(3)
            expect(page.get_by_test_id("similar-result").first).to_contain_text("b_near_a.png")
            expect(page.get_by_test_id("similar-result").first).to_contain_text("d=")

            page.get_by_role("button", name="Search from b_near_a.png").click()
            expect(page.get_by_test_id("similar-query-step")).to_have_count(2)
            page.get_by_label("Display model").select_option("model_b")
            expect(page.get_by_test_id("similar-model")).to_have_text("model_b")
            expect(page.get_by_test_id("similar-result").first).to_contain_text("c_near_b.png")
            assert any(payload["model_id"] == "model_b" for payload in search_payloads)

            export = page.get_by_test_id("similar-export")
            expect(export).to_have_attribute("href", re.compile(r"model_id=model_b.*query_index=" + str(pivot_index)))
            with page.expect_download() as download:
                export.click()
            rows = list(csv.reader(io.StringIO(Path(download.value.path()).read_text(encoding="utf-8"))))
            assert len(rows) == 1 + 1 + 3
            assert rows[1][1] == "b_near_a.png"
            assert rows[2][1] == "c_near_b.png"

            page.get_by_role("button", name="Query step 1: a_query.png").click()
            expect(page.get_by_test_id("similar-query-step")).to_have_count(1)
            page.get_by_role("button", name="Close similar search").click()
            expect(page.get_by_test_id("similar-panel")).to_have_count(0)
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert {path: path.read_bytes() for path in source_bytes} == source_bytes
    assert {path for path in (tmp_path / "train").rglob("*") if path.is_file()} == set(source_bytes)


def test_lv_web_01_search_02_visible_text_and_duplicate_leakage_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """SEARCH-02: Chinese-CLIP text results and duplicate pairs remain reviewable."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")
    from playwright.sync_api import expect, sync_playwright

    root = tmp_path / "leakage-dataset"
    train, valid = root / "train" / "classA", root / "val" / "classA"
    train.mkdir(parents=True)
    valid.mkdir(parents=True)
    rng = np.random.default_rng(2026)
    source_bytes = {}
    for index in range(3):
        image_path = train / f"unique-{index}.png"
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(image_path)
        source_bytes[image_path] = image_path.read_bytes()
    duplicate = train / "dup.png"
    Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(duplicate)
    source_bytes[duplicate] = duplicate.read_bytes()
    duplicate_copy = valid / "dup_copy.png"
    duplicate_copy.write_bytes(source_bytes[duplicate])
    source_bytes[duplicate_copy] = duplicate_copy.read_bytes()
    unique_val = valid / "unique-val.png"
    Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(unique_val)
    source_bytes[unique_val] = unique_val.read_bytes()

    def feature_for(path: Path) -> np.ndarray:
        return np.frombuffer(hashlib.sha256(path.read_bytes()).digest(), dtype=np.uint8).astype(np.float32) + 1

    text_queries: list[str] = []

    def encode_text(query: str) -> np.ndarray:
        text_queries.append(query)
        return feature_for(duplicate) if query == "斑馬" else feature_for(unique_val)

    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "lv-cache"))
    application = create_app(
        model_catalog=lambda: ["chinese-clip-test"],
        model_loader=lambda _model: feature_for,
        text_encoder_loader=lambda _model: encode_text,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            run_payloads = []
            page.on("response", lambda response: run_payloads.append(response.json())
                    if response.request.method == "GET" and "/api/v1/runs/" in response.url
                    and "/images/" not in response.url and not response.url.endswith(".csv")
                    and response.ok else None)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in (root / "train", root / "val"):
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(6)
            assert run_payloads

            page.get_by_test_id("search-by-text").click()
            expect(page.get_by_test_id("similar-panel")).to_be_visible()
            page.get_by_label("Text search query").fill("斑馬")
            page.get_by_role("button", name="Search", exact=True).click()
            expect(page.get_by_test_id("text-search-results")).to_be_visible()
            expect(page.get_by_test_id("text-search-result")).to_have_count(6)
            first_names = [
                page.get_by_test_id("text-search-result").nth(index).inner_text()
                for index in range(2)
            ]
            assert "dup.png" in first_names[0] or "dup_copy.png" in first_names[0]
            assert "dup.png" in first_names[1] or "dup_copy.png" in first_names[1]
            assert first_names[0] != first_names[1]
            assert text_queries == ["斑馬"]

            with page.expect_download() as download:
                page.get_by_test_id("text-search-export").click()
            text_csv = list(csv.reader(io.StringIO(Path(download.value.path()).read_text(encoding="utf-8"))))
            assert len(text_csv) == 1 + 6
            assert text_queries == ["斑馬"]

            page.get_by_role("button", name=re.compile("Use .* as image query")).first.click()
            expect(page.get_by_test_id("similar-query-step")).to_have_count(1)
            expect(page.get_by_test_id("similar-results")).to_be_visible()

            page.get_by_test_id("open-duplicate-review").click()
            page.get_by_label("Only cross split").check()
            page.get_by_test_id("scan-duplicates").click()
            expect(page.get_by_test_id("duplicate-pairs")).to_be_visible()
            pair_rows = page.get_by_test_id("duplicate-pair")
            expect(pair_rows.first).to_contain_text("dup.png")
            expect(pair_rows.first).to_contain_text("dup_copy.png")
            for index in range(pair_rows.count()):
                row_text = pair_rows.nth(index).inner_text()
                assert ("train" in row_text and "val" in row_text)

            page.get_by_label("Duplicate method").select_option("embedding")
            page.get_by_label("Embedding distance threshold").fill("0.05")
            page.get_by_test_id("scan-duplicates").click()
            expect(page.get_by_test_id("duplicate-pair").first).to_contain_text("dup_copy.png")
            page.get_by_role("button", name="Review right dup_copy.png").click()
            expect(page.get_by_test_id("selected-image-name")).to_contain_text("dup_copy.png")
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert text_queries == ["斑馬"]
    assert {path: path.read_bytes() for path in source_bytes} == source_bytes
    assert {path for path in root.rglob("*") if path.is_file()} == set(source_bytes)


def test_lv_web_01_rank_01_visible_ranked_gallery_and_viewer_navigation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """RANK-01: ranking criteria order a visible gallery and drive its viewer."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")
    from playwright.sync_api import expect, sync_playwright

    root = tmp_path / "ranked-dataset"
    vectors: dict[str, np.ndarray] = {}
    source_bytes = {}
    for index, angle in enumerate((0.0, 0.25, 0.9, 1.5, 2.2, 2.8, 3.7, 5.1)):
        split = "train" if index < 6 else "val"
        label = "apple" if index < 4 else "banana"
        folder = root / split / label
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"image-{index:02}.png"
        Image.new("RGB", (48, 48), (20 + index * 20, 80, 160)).save(path)
        source_bytes[path] = path.read_bytes()
        vectors[path.name] = np.asarray([
            np.cos(angle), np.sin(angle), 0.15 * (index % 3), 0.05 * index + 0.1,
        ], dtype=np.float32)

    def embed(path: Path) -> np.ndarray:
        return vectors[path.name]

    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "lv-cache"))
    application = create_app(
        model_catalog=lambda: ["rank-test"],
        model_loader=lambda _model: embed,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            run_payloads: list[dict] = []
            rank_payloads: list[dict] = []
            page.on("response", lambda response: run_payloads.append(response.json())
                    if response.request.method == "GET" and "/api/v1/runs/" in response.url
                    and "/images/" not in response.url and response.ok else None)
            page.on("response", lambda response: rank_payloads.append(response.json())
                    if response.request.method == "POST" and response.url.endswith("/rankings")
                    and response.ok else None)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in (root / "train", root / "val"):
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("ranked-review-item")).to_have_count(8)
            expect(page.get_by_test_id("ranking-status")).to_contain_text("top 8 by outlier score")
            assert run_payloads and rank_payloads

            initial_order = [int(page.get_by_test_id("ranked-review-item").nth(i).get_attribute("data-index"))
                             for i in range(8)]
            assert initial_order == [item["index"] for item in rank_payloads[-1]["items"]]
            page.get_by_test_id("ranked-review-item").first.click()
            expect(page.get_by_test_id("viewer-position")).to_have_text("1 / 8")
            expect(page.get_by_test_id("selected-image-name")).to_have_text(
                run_payloads[-1]["records"][initial_order[0]]["filename"]
            )
            page.get_by_role("button", name="Next ranked image").click()
            expect(page.get_by_test_id("viewer-position")).to_have_text("2 / 8")
            expect(page.get_by_test_id("selected-image-name")).to_have_text(
                run_payloads[-1]["records"][initial_order[1]]["filename"]
            )

            with page.expect_response(
                lambda response: response.request.method == "POST"
                and response.url.endswith("/rankings"), timeout=10_000,
            ) as rank_response:
                page.get_by_label("Rank review images").select_option("disagreement")
            assert rank_response.value.json()["criterion"] == "disagreement"
            expect(page.get_by_test_id("ranking-status")).to_contain_text("label disagreement")
            expect(page.get_by_test_id("ranked-review-item")).to_have_count(8)
            visible_order = [int(page.get_by_test_id("ranked-review-item").nth(i).get_attribute("data-index"))
                             for i in range(8)]
            assert visible_order == [item["index"] for item in rank_payloads[-1]["items"]]
            expect(page.get_by_text("not a mislabel verdict")).to_be_visible()

            train_index = next(row["index"] for row in run_payloads[-1]["records"]
                               if row["split"] == "train")
            val_index = next(row["index"] for row in run_payloads[-1]["records"]
                             if row["split"] == "val")
            page.locator(f'[data-testid="embedding-point"][data-index="{train_index}"]').click()
            page.locator(f'[data-testid="embedding-point"][data-index="{val_index}"]').click(modifiers=["Shift"])
            expect(page.get_by_test_id("selection-summary")).to_have_text("Selected 2 images")
            page.get_by_label("Filter by split").select_option(label="val")
            expect(page.get_by_test_id("ranking-status")).to_contain_text("Selected 2 · showing 1/1")
            expect(page.get_by_test_id("ranked-review-item")).to_have_count(1)
            assert int(page.get_by_test_id("ranked-review-item").first.get_attribute("data-index")) == val_index
            page.get_by_label("Filter by split").select_option(label="All splits")
            expect(page.get_by_test_id("ranked-review-item")).to_have_count(2)

            with page.expect_response(
                lambda response: response.request.method == "POST"
                and response.url.endswith("/rankings"), timeout=10_000,
            ) as filename_response:
                page.get_by_label("Rank review images").select_option("filename")
            assert filename_response.value.json()["criterion"] == "filename"
            selected_indices = [train_index, val_index]
            expected_filename_order = sorted(
                selected_indices,
                key=lambda index: run_payloads[-1]["records"][index]["filename"],
            )
            expect(page.get_by_test_id("ranked-review-item")).to_have_count(2)
            assert [int(page.get_by_test_id("ranked-review-item").nth(i).get_attribute("data-index"))
                    for i in range(2)] == expected_filename_order
            expect(page.get_by_test_id("selection-summary")).to_have_text("Selected 2 images")
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert {path: path.read_bytes() for path in source_bytes} == source_bytes
    assert {path for path in root.rglob("*") if path.is_file()} == set(source_bytes)


def test_lv_web_01_od_02_visible_object_embeddings_policy_filters_and_3d(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Visible React OD-02 flow preserves object order through filters and dimensions."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")
    from playwright.sync_api import expect, sync_playwright

    dataset = tmp_path / "object-dataset"
    folder = dataset / "train"
    images, labels = folder / "images", folder / "labels"
    labels.mkdir(parents=True)
    (dataset / "classes.txt").write_text("good\nbad\n", encoding="utf-8")
    image_path = images / "objects.png"
    image_path.parent.mkdir(parents=True)
    source = Image.new("RGB", (96, 64), (220, 35, 45))
    source.paste((35, 65, 215), (40, 0, 72, 64))
    source.save(image_path)
    label_path = labels / "objects.txt"
    label_path.write_text(
        "0 0.20 0.50 0.22 0.50\n"
        "1 0.72 0.50 0.30 0.70\n"
        "0 0.48 0.50 0.12 0.20\n",
        encoding="utf-8",
    )
    source_bytes = {path: path.read_bytes() for path in (image_path, label_path, dataset / "classes.txt")}
    loader_calls: list[tuple[str, dict]] = []

    def object_loader(model_id: str, policy: dict):
        loader_calls.append((model_id, dict(policy)))

        def embed(crop: Image.Image) -> np.ndarray:
            pixels = np.asarray(crop.convert("RGB"), dtype=np.float32)
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
            vector[3] = crop.width / 100.0
            vector[4] = crop.height / 100.0
            return vector

        return embed

    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "lv-cache"))
    application = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        object_model_loader=object_loader,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            run_payloads = []

            def capture_run(response):
                if response.request.method == "GET" and "/api/v1/runs/" in response.url and "/images/" not in response.url and response.ok:
                    run_payloads.append(response.json())

            page.on("response", capture_run)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Dataset mode").select_option("object_detector")
            page.get_by_label("Detection granularity").select_option("object")
            page.get_by_label("BBox padding").fill("0.25")
            page.get_by_label("Object input size").select_option("336")
            page.get_by_label("Object feature head").select_option("meanpool")
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            page.get_by_label("Image folder").fill(str(folder))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(3)
            expect(page.get_by_test_id("dataset-mode-indicator")).to_contain_text("object-level")
            assert run_payloads
            assert run_payloads[-1]["source_granularity"] == "object"
            assert run_payloads[-1]["object_policy"] == {
                "pad": 0.25, "target_res": 336, "head": "meanpool"
            }
            assert [record["label"] for record in run_payloads[-1]["records"]] == ["good", "bad", "good"]
            assert [record["object_index"] for record in run_payloads[-1]["records"]] == [0, 1, 2]
            assert loader_calls == [("dinov2_vits14", {"pad": 0.25, "target_res": 336, "head": "meanpool"})]

            page.get_by_label("Filter by class").select_option(label="bad")
            expect(page.get_by_test_id("embedding-point")).to_have_count(1)
            page.get_by_label("Filter by class").select_option(label="All classes")
            page.get_by_test_id("embedding-point").nth(1).click()
            expect(page.get_by_test_id("selected-object-number")).to_have_text("2")
            expect(page.get_by_test_id("selected-image-name")).to_contain_text("objects.png · object 2")
            preview = page.get_by_test_id("selected-image")
            expect(preview).to_be_visible()
            expect(preview).to_have_js_property("naturalWidth", 51)
            expect(page.get_by_label("Show YOLO boxes")).to_have_count(0)
            page.get_by_label("Object preview mode").get_by_label("Source image").check()
            expect(page.get_by_label("Show YOLO boxes")).to_be_visible()
            expect(preview).to_have_attribute("src", re.compile("overlay=true.*view=source"))
            expect(page.get_by_test_id("selected-object-bbox")).to_have_text("0.720, 0.500, 0.300, 0.700")
            page.get_by_test_id("dimension-3d").click()
            expect(page.locator("[data-testid='embedding-plot-3d'] .js-plotly-plot")).to_be_visible(timeout=60_000)
            expect(page.get_by_test_id("block-explorer")).to_have_count(0)
            expect(page.get_by_test_id("selected-object-number")).to_have_text("2")
            page.get_by_test_id("dimension-2d").click()
            expect(page.get_by_test_id("selected-object-number")).to_have_text("2")
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert {path: path.read_bytes() for path in source_bytes} == source_bytes
    assert {path for path in dataset.rglob("*") if path.is_file()} == set(source_bytes)


def test_lv_web_01_view_01_selection_gallery_yolo_navigation_and_mode_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Selection order, viewer navigation, YOLO toggle and mode reset stay aligned."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")
    from playwright.sync_api import expect, sync_playwright

    detector_root = tmp_path / "detector-viewer"
    detector_folder = detector_root / "train"
    detector_images, detector_labels = detector_folder / "images", detector_folder / "labels"
    detector_labels.mkdir(parents=True)
    (detector_root / "classes.txt").write_text("good\nbad\n", encoding="utf-8")
    source_files = [detector_root / "classes.txt"]
    for index, (label, color) in enumerate((
        (0, (220, 35, 45)), (1, (35, 180, 70)), (0, (35, 65, 215)),
    )):
        image_path = detector_images / f"detector-{index}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color).save(image_path)
        label_path = detector_labels / f"detector-{index}.txt"
        label_path.write_text(f"{label} 0.5 0.5 0.6 0.6\n", encoding="utf-8")
        source_files.extend((image_path, label_path))
    source_before = {path: path.read_bytes() for path in source_files}

    classifier_folder = tmp_path / "classifier" / "train"
    for index, (label, color) in enumerate((
        ("apple", (200, 35, 45)), ("banana", (35, 180, 70)), ("apple", (35, 65, 215)),
    )):
        image_path = classifier_folder / label / f"classifier-{index}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color).save(image_path)

    def embed(path: Path) -> np.ndarray:
        pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        vector = np.zeros(384, dtype=np.float32)
        vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
        vector[3] = len(path.name) / 100.0
        return vector

    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "lv-cache"))
    application = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=lambda _model_id: embed,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            run_payloads = []

            def capture_run(response):
                if response.request.method == "GET" and "/api/v1/runs/" in response.url and "/images/" not in response.url and response.ok:
                    run_payloads.append(response.json())

            page.on("response", capture_run)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Dataset mode").select_option("object_detector")
            page.get_by_label("Detection granularity").select_option("whole_image")
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            page.get_by_label("Image folder").fill(str(detector_folder))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(3)
            points = page.get_by_test_id("embedding-point")
            points.nth(0).click()
            points.nth(1).click(modifiers=["Shift"])
            expect(page.get_by_test_id("selection-summary")).to_have_text("Selected 2 images")
            gallery = page.get_by_test_id("selection-gallery-item")
            expect(gallery).to_have_count(2)
            display_order = [int(gallery.nth(i).get_attribute("data-index")) for i in range(2)]
            assert len(set(display_order)) == 2
            gallery.first.click()
            record_by_index = {row["index"]: row for row in run_payloads[-1]["records"]}
            expect(page.get_by_test_id("selected-image-name")).to_have_text(
                record_by_index[display_order[0]]["filename"]
            )
            overlay_toggle = page.get_by_label("Show YOLO boxes")
            expect(overlay_toggle).to_be_checked()
            expect(page.get_by_test_id("selected-image")).to_have_attribute("src", re.compile("overlay=true"))
            overlay_toggle.uncheck()
            expect(page.get_by_test_id("selected-image")).to_have_attribute("src", re.compile("overlay=false"))
            page.get_by_role("button", name="Next selected image").click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text(
                record_by_index[display_order[1]]["filename"]
            )
            expect(page.get_by_label("Show YOLO boxes")).not_to_be_checked()
            page.get_by_role("button", name="Close image viewer").click()
            expect(page.get_by_test_id("selected-image")).to_have_count(0)
            expect(page.get_by_test_id("selection-gallery-item")).to_have_count(2)

            page.get_by_role("button", name=f"Remove {detector_folder}").click()
            page.get_by_label("Dataset mode").select_option("image_classifier")
            page.get_by_label("Image folder").fill(str(classifier_folder))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(3)
            page.get_by_test_id("embedding-point").first.click()
            expect(page.get_by_test_id("selected-image")).to_be_visible()
            page.get_by_label("Dataset mode").select_option("object_detector")
            expect(page.get_by_test_id("empty-canvas")).to_be_visible()
            page.get_by_label("Dataset mode").select_option("image_classifier")
            expect(page.get_by_test_id("empty-canvas")).to_be_visible()
            expect(page.get_by_test_id("embedding-point")).to_have_count(0)
            expect(page.get_by_test_id("selected-image")).to_have_count(0)
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert {path: path.read_bytes() for path in source_before} == source_before
    assert {path for path in detector_root.rglob("*") if path.is_file()} == set(source_before)


def test_lv_web_01_od_01_detector_whole_image_and_class_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Visible detector whole-image flow covers selected, automatic, and missing class sources."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")
    dataset = tmp_path / "detector-dataset"
    dataset.mkdir()
    (dataset / "classes.txt").write_text("good\nrotten\n", encoding="utf-8")
    source_files = []
    for split_index, split in enumerate(("train", "val")):
        images = dataset / split / "images"
        labels = dataset / split / "labels"
        labels.mkdir(parents=True)
        for class_id, color in enumerate(((210, 35, 45), (35, 180, 70))):
            image_path = images / f"{split_index}-{class_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), color).save(image_path)
            label_path = labels / f"{image_path.stem}.txt"
            label_path.write_text(f"{class_id} 0.5 0.5 0.6 0.6\n", encoding="utf-8")
            source_files.extend((image_path, label_path))
    source_files.append(dataset / "classes.txt")
    source_bytes = {path: path.read_bytes() for path in source_files}

    def embed(path: Path) -> np.ndarray:
        pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        vector = np.zeros(384, dtype=np.float32)
        vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
        vector[3] = len(path.name) / 10.0
        return vector

    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "lv-cache"))
    models = ["dinov2_vits14", "resnet50"]
    application = create_app(
        model_catalog=lambda: models,
        model_loader=lambda _model_id: embed,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            run_payloads = []

            def capture_run(response):
                if response.request.method == "GET" and "/api/v1/runs/" in response.url and "/images/" not in response.url and response.ok:
                    run_payloads.append(response.json())

            page.on("response", capture_run)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Dataset mode").select_option("object_detector")
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            page.get_by_test_id("model-choice-resnet50").check()
            classes_input = page.get_by_label("Classes file")
            classes_input.set_input_files({
                "name": "override-classes.txt",
                "mimeType": "text/plain",
                "buffer": b"custom-good\ncustom-bad\n",
            })
            expect(page.get_by_test_id("selected-classes-file")).to_contain_text("override-classes.txt")
            page.get_by_label("Manual class names").fill("manual-good,manual-bad")
            for folder in (dataset / "train", dataset / "val"):
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()

            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)
            expect(page.get_by_test_id("dataset-mode-indicator")).to_contain_text(
                "2 classes · selected classes file"
            )
            assert run_payloads[-1]["models"] == models
            assert run_payloads[-1]["class_names"] == ["custom-good", "custom-bad"]
            assert [row["label"] for row in run_payloads[-1]["records"]] == [
                "custom-good", "custom-bad", "custom-good", "custom-bad"
            ]
            page.get_by_label("Filter by class").select_option(label="custom-good")
            expect(page.get_by_test_id("embedding-point")).to_have_count(2)
            page.get_by_label("Filter by class").select_option(label="All classes")
            page.get_by_label("Display model").select_option("resnet50")
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)

            classes_input.set_input_files([])
            expect(page.get_by_test_id("selected-classes-file")).to_have_count(0)
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("dataset-mode-indicator")).to_contain_text(
                "2 classes · auto classes txt"
            )
            assert run_payloads[-1]["class_names"] == ["good", "rotten"]
            assert run_payloads[-1]["class_source"] == "auto_classes_txt"

            missing_dataset = tmp_path / "detector-without-classes"
            missing_train = missing_dataset / "train"
            missing_images, missing_labels = missing_train / "images", missing_train / "labels"
            missing_labels.mkdir(parents=True)
            for class_id, color in enumerate(((200, 45, 55), (45, 190, 75))):
                image_path = missing_images / f"missing-{class_id}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), color).save(image_path)
                (missing_labels / f"{image_path.stem}.txt").write_text(
                    f"{class_id} 0.5 0.5 0.6 0.6\n", encoding="utf-8"
                )
            for folder in (dataset / "train", dataset / "val"):
                page.get_by_role("button", name=f"Remove {folder}").click()
            page.get_by_label("Image folder").fill(str(missing_train))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_label("Manual class names").fill("")
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_role("alert")).to_contain_text(
                "Choose a classes.txt file", timeout=20_000
            )
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert all(path.read_bytes() == content for path, content in source_bytes.items())
