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

import numpy as np
import pytest
from PIL import Image
import uvicorn

from _utils import available_models, load_model
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


def _detector_fixture(root: Path, *, with_classes: bool = True) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    if with_classes:
        (root / "classes.txt").write_text("scratch\nstain\n", encoding="utf-8")
    folders = [root / "train", root / "test"]
    samples = [
        (folders[0], "01.png", "0 0.5 0.5 0.4 0.4\n"),
        (folders[0], "02.png", "0 0.5 0.5 0.4 0.4\n1 0.5 0.5 0.2 0.2\n"),
        (folders[1], "03.png", "1 0.5 0.5 0.3 0.3\n"),
        (folders[1], "04.png", "8 0.5 0.5 0.2 0.2\n"),
    ]
    for index, (folder, filename, label_text) in enumerate(samples):
        (folder / "images").mkdir(parents=True, exist_ok=True)
        (folder / "labels").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (30 + index * 30, 70, 160)).save(folder / "images" / filename)
        (folder / "labels" / (Path(filename).stem + ".txt")).write_text(label_text, encoding="utf-8")
    return folders


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


def test_lv_web_01_od_01_detector_runs_through_visible_react_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The browser UI sends detector mode, preserves classes.txt priority, and displays YOLO labels."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")
    dataset = tmp_path / "detector-fixture"
    folders = _detector_fixture(dataset)
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))

    def fake_loader(_model_id: str):
        def embed(path: Path) -> np.ndarray:
            pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
            vector[3] = len(path.name) / 10.0
            return vector
        return embed

    application = create_app(
        model_catalog=lambda: ["dinov2_vits14", "resnet50"],
        model_loader=fake_loader,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            run_payloads = []
            page.on(
                "response",
                lambda response: run_payloads.append(response.json())
                if "/api/v1/runs/" in response.url and "/images/" not in response.url and response.ok
                else None,
            )
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Analysis mode").select_option("detector")
            page.get_by_label("Detector class names fallback").fill("manual-wrong-name")
            page.get_by_test_id("model-choice-resnet50").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in folders:
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=30_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)
            assert run_payloads
            assert run_payloads[-1]["mode"] == "detector"
            assert run_payloads[-1]["class_names"] == ["scratch", "stain"]
            assert run_payloads[-1]["models"] == ["dinov2_vits14", "resnet50"]

            page.get_by_label("Filter by class").select_option(label="scratch")
            expect(page.get_by_test_id("embedding-point")).to_have_count(1)
            page.get_by_label("Filter by class").select_option(label="All classes")
            page.get_by_test_id("embedding-point").nth(2).click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text("03.png")
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)


def test_lv_web_01_od_01_missing_class_names_is_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Detector mode does not invent class names when both metadata and fallback are absent."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")
    folders = _detector_fixture(tmp_path / "detector-fixture", with_classes=False)
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=lambda _model_id: lambda _path: np.array([1.0] + [0.0] * 383, dtype=np.float32),
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Analysis mode").select_option("detector")
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in folders:
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_role("alert")).to_contain_text("No classes.txt was found", timeout=10_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(0)
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)
