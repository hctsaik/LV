"""Visible LV-WEB-01-IC-01 acceptance against a real DINOv2 model."""

from __future__ import annotations

import os
import hashlib
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

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
                if "/api/v1/runs/" in response.url and "/images/" not in response.url and response.ok:
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
