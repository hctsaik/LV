"""Visible LV React acceptance journeys against fixture and real model data."""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import os
import socket
import sys
import threading
import time
import types
import urllib.error
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

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
DETECTOR_FIXTURE_SHA256 = {
    "classes.txt": "6790a8ef4600c53847701c63920499d8ea56375232a8de32fc15b3e26b9d6ba4",
    "test/images/03.png": "d1db8754763264c772e91c2d290fa34eb4fccde5347547a0b88508a03a78e2a8",
    "test/images/04.png": "2efc4e09db0248ef37f99e869efbf46b3831f0a8a32aebdc15b91fca3a7bd10d",
    "test/labels/03.txt": "bbad46a203bf72fa2666bd7b28fb48bdad2e401c2f4d52fef2f8395d13fc6a6d",
    "test/labels/04.txt": "d2a7909432cfee5af8b863bfff69743ee65dd5971a12510c0d1f2bccc97e7b3c",
    "train/images/01.png": "2bf522ccb76e9a416106fb84e8ce91398058fc5a0df49ee3a2a05a0db0391323",
    "train/images/02.png": "8acebfda3ba1cbfccce4ed478aa4920f53ae3b7c7e65b8a338a08c432bf63035",
    "train/labels/01.txt": "27870ce778e19c56104999640fb7fcdc18751cab266b594e4c462c4960284748",
    "train/labels/02.txt": "4ea54c88845a9d9e59da0f008b0d72db82edd68ec37f615a00b016e9727aed22",
}
COVERAGE_REF_FIXTURE_TREE_SHA256 = {
    "main": "d372c90e9203550f627f044a29a0f35241a97df216e024f96e45d7d9267ba337",
    "candidate": "2f6e47474450aae0443d6b0e755657f36f81bcd753f6420e0cd03d0c0376782e",
}
DETECTOR_FIXTURE_SHA256 = {
    "classes.txt": "6790a8ef4600c53847701c63920499d8ea56375232a8de32fc15b3e26b9d6ba4",
    "test/images/03.png": "d1db8754763264c772e91c2d290fa34eb4fccde5347547a0b88508a03a78e2a8",
    "test/images/04.png": "2efc4e09db0248ef37f99e869efbf46b3831f0a8a32aebdc15b91fca3a7bd10d",
    "test/labels/03.txt": "bbad46a203bf72fa2666bd7b28fb48bdad2e401c2f4d52fef2f8395d13fc6a6d",
    "test/labels/04.txt": "d2a7909432cfee5af8b863bfff69743ee65dd5971a12510c0d1f2bccc97e7b3c",
    "train/images/01.png": "2bf522ccb76e9a416106fb84e8ce91398058fc5a0df49ee3a2a05a0db0391323",
    "train/images/02.png": "8acebfda3ba1cbfccce4ed478aa4920f53ae3b7c7e65b8a338a08c432bf63035",
    "train/labels/01.txt": "27870ce778e19c56104999640fb7fcdc18751cab266b594e4c462c4960284748",
    "train/labels/02.txt": "4ea54c88845a9d9e59da0f008b0d72db82edd68ec37f615a00b016e9727aed22",
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


def _worker_fixture_model(_model_id: str):
    def embed(path: Path) -> np.ndarray:
        pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        vector = np.zeros(384, dtype=np.float32)
        vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
        vector[3] = len(path.name) / 10.0
        return vector
    return embed


def _serve_curation_worker(port: int, cache_root: str) -> None:
    os.environ["LV_CACHE_DIR"] = cache_root
    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=_worker_fixture_model,
    )
    uvicorn.run(application, host="127.0.0.1", port=port, log_level="error", access_log=False)


def _start_curation_worker(cache_root: Path):
    port = _free_port()
    process = multiprocessing.get_context("spawn").Process(
        target=_serve_curation_worker,
        args=(port, str(cache_root)),
        daemon=True,
    )
    process.start()
    base_url = f"http://127.0.0.1:{port}"
    _wait_http(f"{base_url}/api/v1/health", process)
    return process, base_url


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


def _object_coverage_fixture(root: Path) -> tuple[list[Path], dict[str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    train, test = root / "train", root / "test"
    specs = [
        (train, "01.png", 0, 0.50, 0.50, 0.95),
        (train, "02.png", 0, 0.10, 0.10, 0.40),
        (train, "03.png", 0, 0.40, 0.40, 0.85),
        (train, "04.png", 0, 0.08, 0.08, 0.75),
        (train, "05.png", 0, 0.60, 0.60, 0.92),
        (train, "06.png", 0, 0.45, 0.45, 0.70),
        (test, "07.png", 0, 0.70, 0.70, 0.88),
        (test, "08.png", 1, 0.50, 0.50, 0.90),
    ]
    hashes: dict[str, str] = {}
    for index, (folder, filename, class_id, width, height, confidence) in enumerate(specs):
        image_dir, label_dir = folder / "images", folder / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / filename
        Image.new("RGB", (64, 64), (25 + index * 23, 55 + index * 17, 180 - index * 13)).save(image_path)
        label_path = label_dir / f"{Path(filename).stem}.txt"
        label_path.write_text(
            f"{class_id} 0.5 0.5 {width} {height} {confidence}\n",
            encoding="utf-8",
        )
        hashes[str(image_path)] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        hashes[str(label_path)] = hashlib.sha256(label_path.read_bytes()).hexdigest()
    classes_path = root / "classes.txt"
    hashes[str(classes_path)] = hashlib.sha256(classes_path.read_bytes()).hexdigest()
    return [train, test], hashes


def _candidate_object_coverage_fixture(root: Path) -> tuple[list[Path], dict[str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("beta\nalpha\ngamma\n", encoding="utf-8")
    train, test = root / "train", root / "test"
    specs = [
        (train, "11.png", 1, 0.55, 0.55, 0.92),
        (train, "12.png", 2, 0.45, 0.40, 0.90),
        (test, "13.png", 0, 0.50, 0.50, 0.95),
    ]
    colors = [(235, 35, 45), (25, 225, 65), (30, 55, 240)]
    hashes: dict[str, str] = {}
    for index, (folder, filename, class_id, width, height, confidence) in enumerate(specs):
        image_dir, label_dir = folder / "images", folder / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / filename
        Image.new("RGB", (64, 64), colors[index]).save(image_path)
        label_path = label_dir / f"{Path(filename).stem}.txt"
        label_path.write_text(
            f"{class_id} 0.5 0.5 {width} {height} {confidence}\n",
            encoding="utf-8",
        )
        hashes[str(image_path)] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        hashes[str(label_path)] = hashlib.sha256(label_path.read_bytes()).hexdigest()
    classes_path = root / "classes.txt"
    hashes[str(classes_path)] = hashlib.sha256(classes_path.read_bytes()).hexdigest()
    return [train, test], hashes


def _fixture_tree_digest(root: Path, source_hashes: dict[str, str]) -> str:
    rows = sorted(
        f"{Path(path).relative_to(root).as_posix()}:{digest}"
        for path, digest in source_hashes.items()
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


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


def _is_run_summary_response(response) -> bool:
    path = urlparse(response.url).path.rstrip("/")
    prefix = "/api/v1/runs/"
    return (
        response.request.method == "GET"
        and path.startswith(prefix)
        and "/" not in path[len(prefix):]
    )


class _DeterministicUMAP:
    """Picklable reducer double; this E2E validates the React/API journey, not UMAP math."""

    fit_calls = 0
    transform_calls = 0

    def __init__(self, *, n_components: int, n_neighbors: int, random_state: int):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.offset = 0.0

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        type(self).fit_calls += 1
        self.offset = float(type(self).fit_calls)
        return np.asarray(values[:, :self.n_components], dtype=np.float64) + self.offset

    def transform(self, values: np.ndarray) -> np.ndarray:
        type(self).transform_calls += 1
        return np.asarray(values[:, :self.n_components], dtype=np.float64) + self.offset + 10.0


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
                if _is_run_summary_response(response) and response.ok:
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


def test_lv_web_01_rank_01_visible_ranking_viewer_and_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Visible React ranks both legacy review signals and navigates in rank order."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    dataset = _fixture(tmp_path / "ranking-fixture")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    def fixture_loader(_model_id: str):
        def embed(path: Path) -> np.ndarray:
            values = {
                "image-01.png": [1.0, 0.0, 0.0],
                "image-02.png": [0.9, 0.1, 0.0],
                "image-03.png": [0.0, 0.9, 0.1],
                "image-04.png": [0.0, 0.0, 1.0],
            }
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = values[path.name]
            return vector
        return embed

    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=fixture_loader,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.set_default_timeout(10_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("model-choice-fixture-model").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            page.get_by_label("Image folder").fill(str(dataset / "train"))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_label("Image folder").fill(str(dataset / "test"))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()

            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("ranking-workspace")).to_be_visible()
            rows = page.get_by_test_id("ranked-result")
            expect(rows).to_have_count(4)
            assert "not determine image quality" in page.get_by_test_id("ranking-interpretation").inner_text()

            outlier_names = page.locator("[data-testid='ranked-result'] strong").all_inner_texts()
            rows.first.get_by_role("button").click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text(outlier_names[0])
            expect(page.get_by_test_id("selected-image-navigation")).to_contain_text("1 / 4 ranked")
            page.get_by_role("button", name="Next ranked image").click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text(outlier_names[1])
            expect(page.get_by_test_id("selected-image-navigation")).to_contain_text("2 / 4 ranked")

            page.get_by_label("Ranking criterion").select_option("disagreement")
            expect(page.get_by_test_id("ranking-status")).to_contain_text("highest first")
            expect(page.get_by_test_id("ranked-result")).to_have_count(4)
            disagreement_names = page.locator("[data-testid='ranked-result'] strong").all_inner_texts()
            page.get_by_test_id("ranked-result").first.get_by_role("button").click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text(disagreement_names[0])
            page.get_by_role("button", name="Next ranked image").click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text(disagreement_names[1])

            with page.expect_download() as download_info:
                page.get_by_test_id("ranking-export").click()
            download = download_info.value
            assert download.suggested_filename == "disagreement_ranking.csv"
            with Path(download.path()).open(encoding="utf-8", newline="") as handle:
                rows_csv = list(csv.DictReader(handle))
            assert [row["filename"] for row in rows_csv] == disagreement_names
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)


def test_lv_web_01_ic_04_kmeans_drilldown_and_multiselection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Visible React fixture keeps additive selection and drills into numbered 3D blocks."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    dataset = tmp_path / "classifier-blocks"
    image_folder = dataset / "train" / "apple"
    colors = [(220, 35, 45), (30, 190, 70), (35, 65, 215), (210, 180, 25)]
    for cluster, color in enumerate(colors):
        for replicate in range(3):
            folder = image_folder
            folder.mkdir(parents=True, exist_ok=True)
            distinct_color = (color[0] + replicate, color[1], color[2])
            Image.new("RGB", (64, 64), distinct_color).save(folder / f"cluster-{cluster}-{replicate}.png")

    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))

    def fixture_loader(_model_id: str):
        def embed(path: Path) -> np.ndarray:
            cluster, replicate = (int(part) for part in path.stem.split("-")[1:])
            vector = np.zeros(384, dtype=np.float32)
            vector[cluster] = 10.0
            vector[4] = replicate * 0.1 + 0.01
            vector[5] = (cluster + 1) * (replicate + 1) * 0.01
            return vector
        return embed

    application = create_app(
        model_catalog=lambda: ["fixture-model"], model_loader=fixture_loader
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.set_default_timeout(10_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("model-choice-fixture-model").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            page.get_by_label("Image folder").fill(str(dataset / "train"))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=180_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(12)

            for point_index in (0, 1):
                point = page.get_by_test_id("embedding-point").nth(point_index)
                point.focus()
                point.press("Enter")
            expect(page.get_by_test_id("selection-count")).to_have_text("2 selected")
            page.get_by_test_id("dimension-3d").click()
            expect(page.get_by_test_id("embedding-plot-3d")).to_be_visible(timeout=60_000)
            page.get_by_test_id("kmeans-count").press("Home")
            expect(page.get_by_test_id("kmeans-status")).to_contain_text("4 個區塊", timeout=30_000)
            page.get_by_test_id("cluster-block-select").select_option(index=0)
            expect(page.get_by_test_id("block-point")).to_have_count(3, timeout=30_000)

            page.get_by_test_id("block-selection-mode-box").click()
            box_plot = page.locator("[data-testid='block-plot'] .js-plotly-plot")
            box_bounds = box_plot.bounding_box()
            assert box_bounds is not None
            page.mouse.move(box_bounds["x"] + 75, box_bounds["y"] + 55)
            page.mouse.down()
            page.mouse.move(box_bounds["x"] + box_bounds["width"] - 20,
                            box_bounds["y"] + box_bounds["height"] - 25, steps=12)
            page.mouse.up()
            expect(page.get_by_test_id("selection-count")).not_to_have_text("2 selected")

            page.get_by_role("button", name="Clear selection").click()
            expect(page.get_by_test_id("selection-count")).to_have_text("0 selected")
            page.get_by_test_id("block-selection-mode-lasso").click()
            lasso_plot = page.locator("[data-testid='block-plot'] .js-plotly-plot")
            lasso_bounds = lasso_plot.bounding_box()
            assert lasso_bounds is not None
            path = [
                (lasso_bounds["x"] + 75, lasso_bounds["y"] + 55),
                (lasso_bounds["x"] + lasso_bounds["width"] - 20, lasso_bounds["y"] + 55),
                (lasso_bounds["x"] + lasso_bounds["width"] - 20, lasso_bounds["y"] + lasso_bounds["height"] - 25),
                (lasso_bounds["x"] + 75, lasso_bounds["y"] + lasso_bounds["height"] - 25),
                (lasso_bounds["x"] + 75, lasso_bounds["y"] + 55),
            ]
            page.mouse.move(*path[0])
            page.mouse.down()
            for x, y in path[1:]:
                page.mouse.move(x, y, steps=5)
            page.mouse.up()
            expect(page.get_by_test_id("selection-count")).not_to_have_text("0 selected")
            selected_before_switch = page.get_by_test_id("selection-count").inner_text()
            page.get_by_test_id("dimension-2d").click()
            expect(page.get_by_test_id("selection-count")).to_have_text(selected_before_switch)
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)


def test_lv_web_01_search_01_similar_images_chain_model_switch_and_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Visible F3 flow keeps ranked results fresh as the query and model change."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    dataset = tmp_path / "similar-search"
    image_folder = dataset / "train" / "apple"
    image_folder.mkdir(parents=True)
    for index, color in enumerate(((210, 40, 50), (30, 190, 70), (35, 65, 215), (210, 180, 25), (170, 70, 180))):
        Image.new("RGB", (48, 48), color).save(image_folder / f"image-{index:02}.png")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))

    vectors = {
        "fixture-model-a": [
            [1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0], [0.6, 0.8, 0.0],
        ],
        "fixture-model-b": [
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.8, 0.6, 0.0], [0.6, 0.8, 0.0],
        ],
    }

    def fixture_loader(model_id: str):
        def embed(path: Path) -> np.ndarray:
            index = int(path.stem.split("-")[-1])
            return np.asarray(vectors[model_id][index], dtype=np.float32)
        return embed

    application = create_app(
        model_catalog=lambda: list(vectors), model_loader=fixture_loader,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.set_default_timeout(10_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("model-choice-fixture-model-b").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            page.get_by_label("Image folder").fill(str(image_folder))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=180_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(5)

            page.get_by_test_id("embedding-point").nth(0).click()
            page.get_by_test_id("similar-start").click()
            panel = page.get_by_test_id("similar-panel")
            expect(panel.get_by_test_id("similar-result")).to_have_count(4)
            initial_order = [int(item.get_attribute("data-index")) for item in panel.get_by_test_id("similar-result").all()]
            assert initial_order == [1, 4, 2, 3]

            page.get_by_role("button", name="Use image-01.png as query").click()
            expect(panel.get_by_test_id("similar-query-chip")).to_have_count(2)
            expect(panel).to_contain_text("Query: image-01.png")
            page.get_by_label("Display model").select_option("fixture-model-b")
            expect(panel.get_by_test_id("similar-result")).to_have_count(4)
            switched_order = [int(item.get_attribute("data-index")) for item in panel.get_by_test_id("similar-result").all()]
            assert switched_order == [2, 4, 3, 0]

            with page.expect_download() as download_info:
                page.get_by_test_id("similar-export").click()
            rows = list(csv.reader(Path(download_info.value.path()).read_text(encoding="utf-8").splitlines()))
            assert rows[0] == ["index", "filename", "path", "label", "split"]
            assert len(rows) == 6
            assert rows[1][0:2] == ["1", "image-01.png"]
            assert [int(row[0]) for row in rows[2:]] == switched_order
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)


def test_lv_web_01_search_02_text_and_duplicate_review_visible_journey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Visible React flow searches by Chinese-CLIP text and reviews split leakage pairs."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    dataset = tmp_path / "search-02"
    train_apple = dataset / "train" / "apple"
    train_banana = dataset / "train" / "banana"
    test_banana = dataset / "test" / "banana"
    train_apple.mkdir(parents=True)
    train_banana.mkdir(parents=True)
    test_banana.mkdir(parents=True)
    duplicate_pixels = np.random.default_rng(17).integers(0, 255, (48, 48, 3), dtype=np.uint8)
    Image.fromarray(duplicate_pixels).save(train_apple / "image-01.png")
    Image.fromarray(np.random.default_rng(23).integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(train_apple / "image-02.png")
    Image.fromarray(np.random.default_rng(31).integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(train_banana / "image-03.png")
    (test_banana / "image-04.png").write_bytes((train_apple / "image-01.png").read_bytes())
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))

    vectors = {
        "image-01.png": [1.0, 0.0, 0.0],
        "image-02.png": [0.0, 1.0, 0.0],
        "image-03.png": [-1.0, 0.0, 0.0],
        "image-04.png": [0.998, 0.063, 0.0],
    }

    def fixture_loader(_model_id: str):
        def embed(path: Path) -> np.ndarray:
            return np.asarray(vectors[path.name], dtype=np.float32)
        return embed

    def fixture_text_loader(_model_id: str):
        def encode(_query: str) -> np.ndarray:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        return encode

    application = create_app(
        model_catalog=lambda: ["chinese-clip-fixture"],
        model_loader=fixture_loader,
        text_model_loader=fixture_text_loader,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.set_default_timeout(10_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in (dataset / "train", dataset / "test"):
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=180_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)

            page.get_by_label("Chinese-CLIP text query").fill("反光樣本")
            page.get_by_role("button", name="Search images").click()
            expect(page.get_by_test_id("text-search-result")).to_have_count(4)
            text_order = [int(item.get_attribute("data-index")) for item in page.get_by_test_id("text-search-result").all()]
            assert text_order == [0, 3, 1, 2]
            with page.expect_download() as text_download:
                page.get_by_test_id("text-search-export").click()
            text_rows = list(csv.reader(Path(text_download.value.path()).read_text(encoding="utf-8").splitlines()))
            assert [int(row[0]) for row in text_rows[1:]] == text_order
            page.get_by_role("button", name="Use image-04.png as visual search query").click()
            similar_panel = page.get_by_test_id("similar-panel")
            expect(similar_panel.get_by_test_id("similar-result")).to_have_count(3)

            page.get_by_label("Only cross-split pairs").check()
            page.get_by_role("button", name="Scan candidates").click()
            expect(page.get_by_test_id("duplicate-result")).to_have_count(1)
            assert page.get_by_test_id("duplicate-result").first.get_attribute("data-pair") == "0-3"
            thumbnail_url = page.get_by_test_id("duplicate-result").first.locator("img").first.get_attribute("src")
            assert thumbnail_url and page.request.get(base_url + thumbnail_url).status == 200

            page.get_by_label("Duplicate method").select_option("embedding")
            page.get_by_role("button", name="Scan candidates").click()
            expect(page.get_by_test_id("duplicate-result")).to_have_count(1)
            with page.expect_download() as duplicate_download:
                page.get_by_test_id("duplicate-export").click()
            duplicate_rows = list(csv.reader(Path(duplicate_download.value.path()).read_text(encoding="utf-8").splitlines()))
            assert len(duplicate_rows) == 2
            assert duplicate_rows[1][0:2] == ["0", "image-01.png"]
            assert duplicate_rows[1][4:6] == ["3", "image-04.png"]
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)


def test_lv_web_01_view_01_selected_gallery_navigation_and_yolo_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Visible selection thumbnails navigate in order and preserve overlay state."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")
    folders = _detector_fixture(tmp_path / "detector-viewer")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))

    def fixture_loader(_model_id: str):
        def embed(path: Path) -> np.ndarray:
            index = int(path.stem)
            vector = np.zeros(384, dtype=np.float32)
            vector[index] = 1.0
            vector[10] = index / 10.0
            return vector
        return embed

    application = create_app(model_catalog=lambda: ["fixture-model"], model_loader=fixture_loader)
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.set_default_timeout(10_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Analysis mode").select_option("detector")
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in folders:
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=180_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)

            page.get_by_test_id("embedding-point").nth(0).click()
            page.get_by_test_id("embedding-point").nth(1).click()
            expect(page.get_by_test_id("selection-gallery")).to_be_visible()
            thumbs = page.get_by_test_id("selection-thumbnail")
            expect(thumbs).to_have_count(2)
            assert [int(thumb.get_attribute("data-index")) for thumb in thumbs.all()] == [0, 1]
            assert page.request.get(base_url + thumbs.nth(0).locator("img").get_attribute("src")).status == 200

            thumbs.nth(0).click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text("01.png")
            page.get_by_test_id("selected-image-next").click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text("02.png")
            selected_image = page.get_by_test_id("selected-image")
            overlay_url = selected_image.get_attribute("src")
            assert overlay_url and overlay_url.endswith("/images/1/overlay")
            overlay_response = page.request.get(base_url + overlay_url)
            original_response = page.request.get(base_url + overlay_url.removesuffix("/overlay"))
            overlaid = np.asarray(Image.open(BytesIO(overlay_response.body())).convert("RGB"))
            original = np.asarray(Image.open(BytesIO(original_response.body())).convert("RGB"))
            assert not np.array_equal(overlaid, original)

            page.get_by_label("Show annotation boxes").uncheck()
            expect(selected_image).not_to_have_attribute("src", overlay_url)
            page.get_by_label("Show annotation boxes").check()
            expect(selected_image).to_have_attribute("src", overlay_url)

            for mode, point_index, expected_count in (("box", 2, 3), ("lasso", 3, 4)):
                page.get_by_test_id("embedding-selection-mode-" + mode).click()
                point_bounds = page.locator(f'[data-testid="embedding-point"][data-index="{point_index}"]').bounding_box()
                assert point_bounds is not None
                x0, y0 = point_bounds["x"] + point_bounds["width"] / 2, point_bounds["y"] + point_bounds["height"] / 2
                page.mouse.move(x0 - 10, y0 - 10)
                page.mouse.down()
                if mode == "box":
                    page.mouse.move(x0 + 10, y0 + 10, steps=5)
                else:
                    page.mouse.move(x0 + 10, y0 - 10, steps=3)
                    page.mouse.move(x0 + 10, y0 + 10, steps=3)
                    page.mouse.move(x0 - 10, y0 + 10, steps=3)
                    page.mouse.move(x0 - 10, y0 - 10, steps=3)
                page.mouse.up()
                expect(page.get_by_test_id("selection-count")).to_have_text(f"{expected_count} selected")

            page.get_by_label("Analysis mode").select_option("classifier")
            expect(page.get_by_test_id("selected-image")).to_have_count(0)
            expect(page.get_by_test_id("selection-gallery")).to_have_count(0)
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)


def test_lv_web_03_img_01_visible_coverage_projection_box_selection_and_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Coverage runs independently, colors classes, switches projection and reads selected images back."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    dataset = _fixture(tmp_path / "coverage-fixture")
    folders = [dataset / "train", dataset / "test"]
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in folders
        for path in folder.rglob("*")
        if path.is_file()
    }
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))

    def fixture_loader(_model_id: str):
        def embed(path: Path) -> np.ndarray:
            vectors = {
                "image-01.png": [1.0, 0.0, 0.0],
                "image-02.png": [0.9, 0.1, 0.0],
                "image-03.png": [0.0, 1.0, 0.0],
                "image-04.png": [0.0, 0.0, 1.0],
            }
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = vectors[path.name]
            return vector
        return embed

    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=fixture_loader,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.set_default_timeout(10_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_role("button", name="Coverage", exact=True).click()
            for folder in folders:
                page.get_by_label("Coverage image folder").fill(str(folder))
                page.get_by_role("button", name="Add coverage folder").click()
            expect(page.get_by_label("Feature model")).to_have_value("fixture-model")
            page.get_by_label("Projection method").select_option("pca")
            page.get_by_role("button", name="Run coverage analysis").click()
            expect(page.get_by_test_id("coverage-job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("coverage-point")).to_have_count(4)
            expect(page.get_by_test_id("coverage-image-count")).to_have_text("4")
            expect(page.get_by_test_id("coverage-class-legend-item")).to_have_count(2)
            legend = page.get_by_test_id("coverage-class-legend-item")
            assert legend.nth(0).get_attribute("data-label") == "apple"
            assert legend.nth(0).locator("i").evaluate("node => getComputedStyle(node).backgroundColor") == "rgb(231, 76, 60)"
            assert legend.nth(1).get_attribute("data-label") == "banana"
            assert legend.nth(1).locator("i").evaluate("node => getComputedStyle(node).backgroundColor") == "rgb(243, 156, 18)"
            page.locator("input[name='coverage-color-mode'][value='sparsity']").check()
            expect(page.locator(".coverage-legend-gradient")).to_be_visible()
            expect(page.get_by_test_id("coverage-class-legend-item")).to_have_count(0)
            page.locator("input[name='coverage-color-mode'][value='class']").check()

            page.get_by_test_id("coverage-selection-mode-box").click()
            plot = page.locator("[data-testid='coverage-plot'] .js-plotly-plot")
            bounds = plot.bounding_box()
            assert bounds is not None
            page.mouse.move(bounds["x"] + 58, bounds["y"] + 20)
            page.mouse.down()
            page.mouse.move(bounds["x"] + bounds["width"] - 18,
                            bounds["y"] + bounds["height"] - 45, steps=8)
            page.mouse.up()
            expect(page.get_by_test_id("coverage-selected-count")).to_contain_text("selected")
            expect(page.get_by_test_id("coverage-selected-image")).not_to_have_count(0)
            selected = page.get_by_test_id("coverage-selected-image").first
            thumbnail_src = selected.locator("img").get_attribute("src")
            assert thumbnail_src
            assert page.request.get(base_url + thumbnail_src).status == 200

            with page.expect_response(lambda response: "/coverage/runs/" in response.url
                                      and response.url.endswith("/projections")
                                      and response.request.method == "POST" and response.ok):
                page.get_by_label("Projection method").select_option("tsne")
            expect(page.get_by_test_id("coverage-point")).to_have_count(4)
            expect(page.get_by_test_id("coverage-selected-count")).to_contain_text("selected")

            with page.expect_response(lambda response: "/coverage/runs/" in response.url
                                      and response.url.endswith("/projections")
                                      and response.request.method == "POST" and response.ok):
                page.get_by_test_id("coverage-dimension-3d").click()
            expect(page.locator("[data-testid='coverage-plot'] [data-testid='coverage-plot-3d']")).to_be_visible()
            expect(page.get_by_test_id("coverage-point")).to_have_count(4)
            page.get_by_test_id("coverage-point").nth(0).focus()
            page.get_by_test_id("coverage-point").nth(0).press("Enter")
            expect(page.get_by_test_id("coverage-selected-count")).to_have_text("3 selected")
            with page.expect_response(lambda response: "/coverage/runs/" in response.url
                                      and response.url.endswith("/projections")
                                      and response.request.method == "POST" and response.ok):
                page.get_by_test_id("coverage-dimension-2d").click()
            expect(page.get_by_test_id("coverage-selected-count")).to_have_text("3 selected")

            with page.expect_response(lambda response: response.url.endswith("/sparsity")
                                      and response.request.method == "POST" and response.ok):
                page.get_by_test_id("coverage-k").fill("1")
            expect(page.get_by_test_id("coverage-k-summary")).to_have_text("1")

            page.get_by_test_id("coverage-clear-selection").click()
            expect(page.get_by_test_id("coverage-selected-count")).to_have_text("0 selected")
            expect(page.get_by_test_id("coverage-selected-image")).to_have_count(0)
            assert page.get_by_test_id("coverage-clear-selection").is_disabled()

            page.get_by_role("button", name="Visualize", exact=True).click()
            page.get_by_role("button", name="Coverage", exact=True).click()
            expect(page.get_by_test_id("coverage-image-count")).to_have_text("4")
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_hashes
    }


def test_lv_web_03_obj_01_visible_object_coverage_filters_blocks_and_crop_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Visible object coverage filters aligned YOLO rows, guards tiny classes, and selects cropped samples."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    folders, source_hashes = _object_coverage_fixture(tmp_path / "object-coverage-fixture")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))

    def object_loader(_model_id: str, **_kwargs):
        def embed(crop: Image.Image) -> np.ndarray:
            pixels = np.asarray(crop.convert("RGB"), dtype=np.float32)
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
            vector[3] = crop.width / 100.0
            vector[4] = crop.height / 100.0
            return vector
        return embed

    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model_id: lambda _path: np.array([1.0] + [0.0] * 383, dtype=np.float32),
        object_model_loader=object_loader,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.set_default_timeout(15_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_role("button", name="Coverage", exact=True).click()
            page.get_by_label("Analysis unit").select_option("object")
            page.get_by_label("Projection method").select_option("lda")
            for folder in folders:
                page.get_by_label("Coverage image folder").fill(str(folder))
                page.get_by_role("button", name="Add coverage folder").click()
            page.get_by_role("button", name="Run coverage analysis").click()
            expect(page.get_by_test_id("coverage-job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("coverage-point")).to_have_count(8)
            expect(page.get_by_test_id("coverage-image-count")).to_have_text("8")
            expect(page.get_by_role("heading", name="Object coverage by class")).to_be_visible()

            page.get_by_test_id("coverage-object-class").select_option("alpha")
            page.get_by_test_id("coverage-min-short-px").fill("10")
            page.get_by_test_id("coverage-confidence-min").fill("0.8")
            page.get_by_test_id("coverage-confidence-max").fill("0.95")
            page.get_by_test_id("coverage-apply-object-filters").click()
            expect(page.get_by_test_id("coverage-point")).to_have_count(4)
            expect(page.get_by_test_id("coverage-image-count")).to_have_text("4")
            point_labels = [
                page.get_by_test_id("coverage-point").nth(index).get_attribute("aria-label")
                for index in range(4)
            ]
            assert [label.split(",", 1)[0] for label in point_labels] == [
                "Select 01.png", "Select 03.png", "Select 05.png", "Select 07.png",
            ]
            beta_summary = page.locator(
                '[data-testid="coverage-class-summary-row"][data-label="beta"]'
            )
            expect(beta_summary).to_contain_text("1")
            expect(page.get_by_test_id("coverage-object-blocks")).to_contain_text("K=4")
            expect(page.get_by_test_id("coverage-object-block")).to_have_count(4)

            page.get_by_test_id("coverage-object-class").select_option("beta")
            page.get_by_test_id("coverage-apply-object-filters").click()
            expect(page.get_by_test_id("coverage-point")).to_have_count(1)
            expect(page.get_by_test_id("coverage-no-blocks")).to_be_visible()
            expect(page.locator(".coverage-warning").filter(
                has_text="fewer than two objects",
            )).to_be_visible()

            page.get_by_test_id("coverage-object-class").select_option("alpha")
            page.get_by_test_id("coverage-apply-object-filters").click()
            expect(page.get_by_test_id("coverage-object-block")).to_have_count(4)
            page.get_by_test_id("coverage-object-block").first.click()
            expect(page.get_by_test_id("coverage-selected-count")).to_have_text("1 selected")
            selected = page.get_by_test_id("coverage-selected-image").first
            assert int(selected.get_attribute("data-index")) in {0, 2, 4, 6}
            image = selected.locator("img")
            expect(image).to_have_js_property("complete", True)
            assert 0 < image.evaluate("node => node.naturalWidth") < 64
            expect(selected.locator("small").nth(1)).to_contain_text("Object ")
            thumbnail_src = image.get_attribute("src")
            assert thumbnail_src and page.request.get(base_url + thumbnail_src).status == 200
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in source_hashes
    }


def test_lv_web_03_ref_01_visible_candidate_pool_reference_and_curation_journey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """React projects a separate candidate pool/reference and keeps class, crop, and curation identity aligned."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    main_root = tmp_path / "reference-main"
    candidate_root = tmp_path / "reference-candidates"
    main_folders, main_hashes = _object_coverage_fixture(main_root)
    candidate_folders, candidate_hashes = _candidate_object_coverage_fixture(candidate_root)
    assert _fixture_tree_digest(main_root, main_hashes) == COVERAGE_REF_FIXTURE_TREE_SHA256["main"]
    assert _fixture_tree_digest(candidate_root, candidate_hashes) == COVERAGE_REF_FIXTURE_TREE_SHA256["candidate"]
    source_hashes = {**main_hashes, **candidate_hashes}
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))

    def object_loader(_model_id: str, **_kwargs):
        def embed(crop: Image.Image) -> np.ndarray:
            pixels = np.asarray(crop.convert("RGB"), dtype=np.float32)
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
            vector[3] = crop.width / 100.0
            vector[4] = crop.height / 100.0
            return vector
        return embed

    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model_id: lambda _path: np.array([1.0] + [0.0] * 383, dtype=np.float32),
        object_model_loader=object_loader,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1540, "height": 1200}, accept_downloads=True)
            page.set_default_timeout(20_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_role("button", name="Coverage", exact=True).click()
            page.get_by_label("Analysis unit").select_option("object")
            page.get_by_label("Projection method").select_option("pca")
            for folder in main_folders:
                page.get_by_label("Coverage image folder").fill(str(folder))
                page.get_by_role("button", name="Add coverage folder").click()
            page.get_by_role("button", name="Run coverage analysis").click()
            expect(page.get_by_test_id("coverage-job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("coverage-point")).to_have_count(8)

            for folder in candidate_folders:
                page.get_by_test_id("coverage-candidate-folder-path").fill(str(folder))
                page.get_by_role("button", name="Add candidate folder").click()
            with page.expect_response(
                lambda response: response.url.endswith("/candidate-view")
                and response.request.method == "POST" and response.ok,
                timeout=60_000,
            ) as pool_view_response:
                page.get_by_test_id("coverage-project-candidates").click()
            expect(page.get_by_test_id("coverage-candidate-job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("coverage-candidate-counts")).to_contain_text("3 candidates")
            expect(page.get_by_test_id("coverage-candidate-point")).to_have_count(3)
            expect(page.get_by_test_id("coverage-candidate-work-item")).to_have_count(3)
            pool_view = pool_view_response.value.json()
            assert [point["label"] for point in pool_view["candidate_points"]] == ["alpha", "gamma", "beta"]
            for records in (pool_view["main_points"], pool_view["candidate_points"], pool_view["work_records"]):
                assert all("path" not in record and "image_path" not in record for record in records)

            pool_point = page.get_by_test_id("coverage-candidate-point")
            pool_point.nth(0).focus()
            pool_point.nth(0).press("Enter")
            expect(page.get_by_test_id("coverage-selected-candidate-count")).to_have_text("1 selected")
            selected_candidate = page.get_by_test_id("coverage-selected-candidate").first
            candidate_image = selected_candidate.locator("img")
            expect(candidate_image).to_have_js_property("complete", True)
            assert 0 < candidate_image.evaluate("node => node.naturalWidth") < 64
            page.get_by_test_id("coverage-add-selected-candidates").click()
            expect(page.locator(".notice-banner")).to_contain_text("Added 1 candidate(s) to curation")

            page.get_by_test_id("coverage-dimension-3d").click()
            expect(page.locator("[data-testid='coverage-plot'] [data-testid='coverage-plot-3d']")).to_be_visible()
            expect(page.get_by_test_id("coverage-candidate-point")).to_have_count(3, timeout=60_000)
            pool_point = page.get_by_test_id("coverage-candidate-point")
            pool_point.nth(1).focus()
            pool_point.nth(1).press("Enter")
            expect(page.get_by_test_id("coverage-selected-candidate-count")).to_have_text("2 selected")
            page.get_by_test_id("coverage-dimension-2d").click()
            expect(page.get_by_test_id("coverage-candidate-point")).to_have_count(3, timeout=60_000)
            expect(page.get_by_test_id("coverage-selected-candidate-count")).to_have_text("2 selected")
            page.get_by_test_id("coverage-add-selected-candidates").click()
            page.get_by_test_id("coverage-add-top-candidates").click()
            expect(page.locator(".notice-banner")).to_contain_text("Added 1 candidate(s) to curation; skipped 2 duplicate(s).")
            pool_items_response = page.request.get(base_url + "/api/v1/curation/items")
            pool_items = pool_items_response.json()["items"]
            assert len(pool_items) == 3
            assert {item["source"] for item in pool_items} == {"gap_filler"}
            assert all(item["level"] == "object" and item.get("obj_index") is not None for item in pool_items)

            page.get_by_test_id("open-curation").click()
            expect(page.get_by_test_id("curation-item")).to_have_count(3)
            page.get_by_role("button", name="Clear list").click()
            page.get_by_role("button", name="Confirm clear list").click()
            expect(page.get_by_test_id("curation-empty")).to_be_visible()
            page.get_by_role("button", name="Coverage", exact=True).click()

            page.get_by_test_id("coverage-candidate-role").select_option("reference")
            with page.expect_response(
                lambda response: response.url.endswith("/candidate-view")
                and response.request.method == "POST" and response.ok,
                timeout=60_000,
            ) as reference_view_response:
                page.get_by_test_id("coverage-refresh-candidate-view").click()
            reference_view = reference_view_response.value.json()
            assert reference_view["recall"] is not None
            assert 0.0 <= reference_view["recall"] <= 1.0
            gamma = next(row for row in reference_view["class_gaps"] if row["label"] == "gamma")
            assert gamma["main_count"] == 0 and gamma["candidate_count"] == 1 and gamma["state"] == "missing"
            expect(page.get_by_test_id("coverage-reference-recall")).not_to_have_text("Unavailable")
            expect(page.locator('[data-testid="coverage-class-gap-row"][data-label="gamma"]')).to_contain_text("missing")
            reference_point = page.get_by_test_id("coverage-candidate-point")
            reference_point.nth(0).focus()
            reference_point.nth(0).press("Enter")
            expect(page.get_by_test_id("coverage-selected-candidate")).to_have_count(1)
            page.get_by_test_id("coverage-add-selected-candidates").click()
            reference_items = page.request.get(base_url + "/api/v1/curation/items").json()["items"]
            assert len(reference_items) == 1
            assert reference_items[0]["source"] == "reference"

            page.get_by_test_id("coverage-object-class").select_option("gamma")
            page.get_by_test_id("coverage-apply-object-filters").click()
            expect(page.get_by_test_id("coverage-candidate-counts")).to_contain_text("0 main")
            expect(page.get_by_test_id("coverage-candidate-counts")).to_contain_text("1 candidates")
            expect(page.get_by_test_id("coverage-image-count")).to_have_text("0")
            expect(page.get_by_test_id("coverage-reference-recall")).to_have_text("Unavailable")
            expect(page.locator(".coverage-warning").filter(has_text="No main records match")).to_be_visible()
            expect(page.get_by_test_id("coverage-candidate-point")).to_have_count(1)
            gamma_point = page.get_by_test_id("coverage-candidate-point")
            gamma_point.focus()
            gamma_point.press("Enter")
            page.get_by_test_id("coverage-add-selected-candidates").click()
            gamma_items = page.request.get(base_url + "/api/v1/curation/items").json()["items"]
            assert len(gamma_items) == 2
            gamma_item = next(item for item in gamma_items if item["label"] == "gamma")
            assert gamma_item["source"] == "reference_class"
            assert gamma_item["level"] == "object" and gamma_item["obj_index"] == 0
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in source_hashes
    }


def test_lv_web_01_cart_01_visible_curation_exports_and_new_pid_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Visible React preserves selection/cart semantics through run and worker restarts."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    dataset = _fixture(tmp_path / "curation-fixture")
    source_before = {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }
    cache_root = tmp_path / "curation-cache"
    worker = None
    browser = None
    playwright = None
    try:
        worker, base_url = _start_curation_worker(cache_root)
        first_pid = worker.pid
        from playwright.sync_api import expect, sync_playwright

        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1100}, accept_downloads=True)
        page.set_default_timeout(15_000)
        page.goto(base_url, wait_until="networkidle")
        page.get_by_test_id("model-choice-fixture-model").check()
        page.get_by_label("Supervised UMAP").uncheck()
        page.get_by_label("PCA").check()
        page.get_by_label("Image folder").fill(str(dataset / "train"))
        page.get_by_role("button", name="Add folder").click()
        page.get_by_label("Image folder").fill(str(dataset / "test"))
        page.get_by_role("button", name="Add folder").click()
        page.get_by_role("button", name="Run analysis").click()
        expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
        expect(page.get_by_test_id("embedding-point")).to_have_count(4)

        page.get_by_test_id("open-curation").click()
        expect(page.get_by_test_id("curation-empty")).to_be_visible()
        expect(page.get_by_test_id("curation-export-csv")).to_be_disabled()
        page.get_by_role("button", name="Visualize").click()
        page.get_by_test_id("embedding-point").nth(0).click()
        page.get_by_test_id("embedding-point").nth(1).click()
        expect(page.get_by_test_id("selection-count")).to_have_text("2 selected")

        page.get_by_test_id("add-selection-to-curation").click()
        expect(page.locator(".alert-info")).to_contain_text("Added 2 item(s) to curation; skipped 0 duplicate(s).")
        page.get_by_test_id("add-selection-to-curation").click()
        expect(page.locator(".alert-info")).to_contain_text("Added 0 item(s) to curation; skipped 2 duplicate(s).")

        page.get_by_role("button", name="Run analysis").click()
        expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
        expect(page.get_by_test_id("selection-count")).to_have_text("0 selected")
        page.get_by_test_id("open-curation").click()
        expect(page.get_by_test_id("curation-item")).to_have_count(2)
        expect(page.get_by_test_id("open-curation")).to_contain_text("2")
        page.get_by_label("Reason for visible curation items").fill("checked in the visible React review")
        page.get_by_role("button", name="Apply reason").click()
        expect(page.locator(".curation-item-reason")).to_have_count(2)
        expect(page.locator(".curation-item-reason").first).to_have_text("checked in the visible React review")

        with page.expect_download() as csv_download:
            page.get_by_test_id("curation-export-csv").click()
        with Path(csv_download.value.path()).open(encoding="utf-8-sig", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        assert len(csv_rows) == 2
        assert all(row["source"] == "manual" for row in csv_rows)
        assert all(row["reason"] == "checked in the visible React review" for row in csv_rows)
        assert all(len(row["sha256"]) == 64 for row in csv_rows)

        with page.expect_download() as json_download:
            page.get_by_test_id("curation-export-json").click()
        json_items = json.loads(Path(json_download.value.path()).read_text(encoding="utf-8"))
        assert len(json_items) == 2
        assert all(Path(item["path"]).is_file() for item in json_items)
        assert all(item["reason"] == "checked in the visible React review" for item in json_items)

        with page.expect_download() as zip_download:
            page.get_by_test_id("curation-export-zip").click()
        with zipfile.ZipFile(zip_download.value.path()) as archive:
            assert "manifest.csv" in archive.namelist()
            assert len([name for name in archive.namelist() if name.startswith("images/")]) == 2

        worker.terminate()
        worker.join(timeout=10)
        assert not worker.is_alive()
        worker, restarted_url = _start_curation_worker(cache_root)
        assert worker.pid != first_pid
        page.goto(restarted_url, wait_until="networkidle")
        page.get_by_test_id("open-curation").click()
        expect(page.get_by_test_id("curation-item")).to_have_count(2)
        expect(page.locator(".curation-item-reason").first).to_have_text("checked in the visible React review")
        assert worker.pid != first_pid
    finally:
        if browser is not None:
            browser.close()
        if playwright is not None:
            playwright.stop()
        if worker is not None and worker.is_alive():
            worker.terminate()
            worker.join(timeout=10)
        assert source_before == {
            path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in dataset.rglob("*") if path.is_file()
        }


def test_lv_web_01_ic_03_umap_reference_visible_journey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Visible React controls report fit, reuse, transform, and rebuild transitions."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    fake_umap = types.ModuleType("umap")
    fake_umap.UMAP = _DeterministicUMAP
    monkeypatch.setitem(sys.modules, "umap", fake_umap)
    if "umap_ref" in sys.modules:
        monkeypatch.setattr(sys.modules["umap_ref"], "umap", fake_umap)
    _DeterministicUMAP.fit_calls = 0
    _DeterministicUMAP.transform_calls = 0

    dataset = tmp_path / "classifier-reference"
    image_folder = dataset / "train" / "apple"
    image_folder.mkdir(parents=True)
    for index, color in enumerate(((220, 35, 45), (30, 190, 70), (35, 65, 215), (210, 180, 25))):
        Image.new("RGB", (64, 64), color).save(image_folder / f"image-{index:02}.png")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))

    def fixture_loader(_model_id: str):
        def embed(path: Path) -> np.ndarray:
            pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
            vector[3] = len(path.name) / 10.0
            return vector
        return embed

    application = create_app(model_catalog=lambda: ["fixture-model"], model_loader=fixture_loader)
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1450, "height": 1000})
            page.set_default_timeout(10_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("model-choice-fixture-model").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_test_id("projection-umap").check()
            page.get_by_label("Image folder").fill(str(dataset / "train"))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_label("Keep UMAP reference frame").check()

            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("umap-reference-result")).to_contain_text(
                "UMAP reference fitted for fixture-model."
            )
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)

            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("umap-reference-result")).to_contain_text(
                "UMAP reference reused for fixture-model; 0 new point(s) transformed."
            )

            Image.new("RGB", (64, 64), (180, 90, 35)).save(image_folder / "image-99.png")
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("umap-reference-result")).to_contain_text(
                "UMAP reference reused for fixture-model; 1 new point(s) transformed."
            )
            expect(page.get_by_test_id("embedding-point")).to_have_count(5)

            (image_folder / "image-99.png").unlink()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("umap-reference-result")).to_contain_text(
                "UMAP reference reused for fixture-model; 0 new point(s) transformed."
            )
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)

            page.get_by_role("button", name="Rebuild UMAP reference next run").click()
            expect(page.locator(".umap-rebuild-pending")).to_contain_text("next analysis will refit")
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("umap-reference-result")).to_contain_text(
                "UMAP reference fitted for fixture-model."
            )
            assert _DeterministicUMAP.fit_calls == 2
            assert _DeterministicUMAP.transform_calls == 1
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


def test_lv_web_01_edge_01_single_unicode_path_and_unreadable_file_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A single image in a Unicode path remains selectable when a sibling file is corrupt."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")
    image_folder = tmp_path / "資料 集" / "train" / "類別A"
    image_folder.mkdir(parents=True)
    image_path = image_folder / "影像 [1].jpg"
    Image.new("RGB", (48, 36), (127, 64, 192)).save(image_path)
    unreadable_path = image_folder / "壞檔.jpg"
    unreadable_path.write_bytes(b"not an image")
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (image_path, unreadable_path)
    }
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model_id: lambda _path: np.array([1.0] + [0.0] * 383, dtype=np.float32),
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
                if _is_run_summary_response(response) and response.ok:
                    run_payloads.append(response.json())

            page.on("response", capture_run)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Image folder").fill(str(image_folder.parent))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=30_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(1)
            assert run_payloads and run_payloads[-1]["embedding_finite"] is True
            assert run_payloads[-1]["method"] == "pca"
            record = run_payloads[-1]["records"][0]
            assert (record["filename"], record["split"], record["label"]) == (
                image_path.name, "train", "類別A",
            )
            assert all(np.isfinite([point["x"], point["y"]]).all() for point in run_payloads[-1]["points"])
            expect(page.locator(".alert-info")).to_contain_text("Skipped 1 unreadable image(s).")
            page.get_by_test_id("embedding-point").click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text(image_path.name)
            image = page.get_by_test_id("selected-image")
            expect(image).to_be_visible()
            expect(image).to_have_js_property("naturalWidth", 48)
            assert {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (image_path, unreadable_path)
            } == source_hashes
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
                if _is_run_summary_response(response) and response.ok
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


def test_lv_web_01_od_02_object_flow_through_visible_react_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """React selects object granularity, filters boxes, and previews the selected source bbox."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")
    folders = _detector_fixture(tmp_path / "detector-object-fixture")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))

    def object_loader(_model_id: str, **_kwargs):
        def embed(crop: Image.Image) -> np.ndarray:
            pixels = np.asarray(crop.convert("RGB"), dtype=np.float32)
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
            vector[3] = crop.width / 100.0
            return vector
        return embed

    application = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=lambda _model_id: lambda _path: np.array([1.0] + [0.0] * 383, dtype=np.float32),
        object_model_loader=object_loader,
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
                if _is_run_summary_response(response) and response.ok
                else None,
            )
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Analysis mode").select_option("detector")
            page.get_by_label("Detection unit").select_option("object")
            page.get_by_label("Detector class names fallback").fill("wrong-fallback")
            page.get_by_test_id("model-choice-dinov2_vits14").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in folders:
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=30_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(5)

            assert run_payloads
            run = run_payloads[-1]
            assert run["unit"] == "object"
            assert run["object_policy"] == {"pad": 0.12, "target_res": 224, "head": "cls"}
            assert [(record["filename"], record["label"], record["obj_index"]) for record in run["records"]] == [
                ("01.png", "scratch", 0),
                ("02.png", "scratch", 0),
                ("02.png", "stain", 1),
                ("03.png", "stain", 0),
                ("04.png", "class_8", 0),
            ]

            page.get_by_label("Filter by class").select_option(label="stain")
            expect(page.get_by_test_id("embedding-point")).to_have_count(2)
            page.get_by_test_id("embedding-point").nth(0).click()
            expect(page.get_by_role("heading", name="Selected object")).to_be_visible()
            expect(page.get_by_test_id("selected-image-name")).to_have_text("02.png")
            expect(page.get_by_test_id("selected-object-index")).to_have_text("2")
            expect(page.get_by_test_id("selected-object-box")).to_have_text("0.500, 0.500, 0.200, 0.200")
            expect(page.get_by_test_id("selected-object-bbox")).to_be_visible()
            expect(page.get_by_test_id("selected-image")).to_have_js_property("naturalWidth", 32)

            page.get_by_test_id("dimension-3d").click()
            expect(page.get_by_test_id("embedding-plot-3d")).to_be_visible()
            points_3d = page.get_by_test_id("embedding-point")
            expect(points_3d).to_have_count(2)
            expect(points_3d.nth(0)).to_have_attribute("aria-pressed", "true")
            expect(points_3d.nth(0)).to_have_attribute("aria-label", "Select 02.png, object 2, stain, train")

            points_3d.nth(1).focus()
            page.keyboard.press("Enter")
            expect(page.get_by_test_id("selected-image-name")).to_have_text("03.png")
            expect(page.get_by_test_id("selected-object-index")).to_have_text("1")
            expect(page.get_by_test_id("selected-object-bbox")).to_be_visible()

            page.get_by_test_id("dimension-2d").click()
            expect(page.locator('[data-testid="embedding-point"][data-index="3"]')).to_have_class("point point-active")
            expect(page.get_by_test_id("selected-image-name")).to_have_text("03.png")
            expect(page.get_by_test_id("selected-object-index")).to_have_text("1")
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)


def test_lv_web_01_od_02_real_model_object_embeddings_on_arm64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Visible object-level React flow uses the pinned DINOv2 model on real crops."""
    models_dir = Path(os.environ.get("LV_MODELS_DIR", ROOT / "models")).resolve()
    weights = models_dir / "dinov2_vits14" / "dinov2_vits14.pth"
    if not weights.is_file():
        pytest.skip(f"real DINOv2-small weights are unavailable: {weights}")
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == DINOV2_VITS14_SHA256, \
        "the local model weights differ from the LV-WEB-01 baseline"
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    dataset = tmp_path / "detector-object-real-fixture"
    folders = _detector_fixture(dataset)
    source_hashes = {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }
    assert source_hashes == DETECTOR_FIXTURE_SHA256
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: available_models(models_dir),
        model_loader=lambda model_id: load_model(model_id, models_dir=models_dir),
        object_model_loader=lambda model_id, **kwargs: load_model(model_id, models_dir=models_dir, **kwargs),
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
                if _is_run_summary_response(response) and response.ok
                else None,
            )
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Analysis mode").select_option("detector")
            page.get_by_label("Detection unit").select_option("object")
            page.get_by_test_id("model-choice-dinov2_vits14").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in folders:
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=180_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(5)

            assert run_payloads, "React UI did not load the real-model object result"
            run = run_payloads[-1]
            assert run["unit"] == "object"
            assert run["object_policy"] == {"pad": 0.12, "target_res": 224, "head": "cls"}
            assert run["models"] == ["dinov2_vits14"]
            assert run["embedding_shape"] == [5, 384]
            assert run["embedding_finite"] is True
            assert run["embedding_nonzero"] is True
            assert [(row["filename"], row["label"], row["obj_index"]) for row in run["records"]] == [
                ("01.png", "scratch", 0),
                ("02.png", "scratch", 0),
                ("02.png", "stain", 1),
                ("03.png", "stain", 0),
                ("04.png", "class_8", 0),
            ]
            page.get_by_test_id("embedding-point").nth(2).click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text("02.png")
            expect(page.get_by_test_id("selected-object-index")).to_have_text("2")
            expect(page.get_by_test_id("selected-object-bbox")).to_be_visible()
            expect(page.get_by_test_id("selected-image")).to_have_js_property("naturalWidth", 32)
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }, "object-level React analysis changed the source dataset"


def test_lv_web_01_od_01_real_model_visible_detector_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Run detector mode through React with the pinned DINOv2 model and YOLO fixture."""
    models_dir = Path(os.environ.get("LV_MODELS_DIR", ROOT / "models")).resolve()
    weights = models_dir / "dinov2_vits14" / "dinov2_vits14.pth"
    if not weights.is_file():
        pytest.skip(f"real DINOv2-small weights are unavailable: {weights}")
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == DINOV2_VITS14_SHA256, \
        "the local model weights differ from the LV-WEB-01 baseline"
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    dataset = tmp_path / "detector-real-model-fixture"
    folders = _detector_fixture(dataset)
    fixture_hashes = {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }
    assert fixture_hashes == DETECTOR_FIXTURE_SHA256
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: available_models(models_dir),
        model_loader=lambda model_id: load_model(model_id, models_dir=models_dir),
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
                if _is_run_summary_response(response) and response.ok
                else None,
            )
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Analysis mode").select_option("detector")
            page.get_by_label("Detector class names fallback").fill("manual-wrong-name")
            page.get_by_test_id("model-choice-dinov2_vits14").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in folders:
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=180_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)

            assert run_payloads, "React UI did not load the real-model detector result"
            run = run_payloads[-1]
            assert run["mode"] == "detector"
            assert run["class_names"] == ["scratch", "stain"]
            assert run["models"] == ["dinov2_vits14"]
            assert run["embedding_shape"] == [4, 384]
            assert run["embedding_finite"] is True
            assert run["embedding_nonzero"] is True
            assert [(row["filename"], row["split"], row["label"]) for row in run["records"]] == [
                ("01.png", "train", "scratch"),
                ("02.png", "train", "mix"),
                ("03.png", "test", "stain"),
                ("04.png", "test", "class_8"),
            ]

            page.get_by_label("Filter by class").select_option(label="scratch")
            expect(page.get_by_test_id("embedding-point")).to_have_count(1)
            page.get_by_label("Filter by class").select_option(label="All classes")
            page.get_by_test_id("embedding-point").nth(2).click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text("03.png")
            expect(page.get_by_test_id("selected-image")).to_have_js_property("naturalWidth", 32)
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert fixture_hashes == {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }, "React analysis changed the detector source files"
