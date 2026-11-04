"""Visible LV React acceptance journeys against fixture and real model data."""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import subprocess
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
CHINESE_CLIP_MODEL_ID = "chinese-clip-vit-base-patch16"
CHINESE_CLIP_PYTORCH_SHA256 = "7b7b583c210c867410bc6bdb8a55fe14eec62999e0a9ea31ff222dc501f9cfbe"
FIXTURE_SHA256 = {
    "train/apple/image-01.png": "be4582401ec897d3d653f2f52275802d4485ab5106e0e6125344711a3d32b6a3",
    "train/apple/image-02.png": "ed47d5fdeb3b4ad1fb5ff833846ffab6b8fc690d96f5d44a348196b0c1f5d731",
    "train/banana/image-03.png": "1ed14b03efb7653fadc8bc56445702b7ea39ae67974957dee0d4f3c5ad04f0dd",
    "test/banana/image-04.png": "246ec8e9c65eda52b30d0da34ccd0c35294e6575299430218d65f2803d990060",
}
DETECTOR_FIXTURE_SHA256 = {
    "classes.txt": "6790a8ef4600c53847701c63920499d8ea56375232a8de32fc15b3e26b9d6ba4",
    "test/images/03.png": "6c3f44cb683248541a6592a7e241731df440e74d3490a164f9ccfdda5326da37",
    "test/images/04.png": "9959fd8060eb6de599f05151cc4aad1143fe0f681d25fa7b2fefa49739a1c21c",
    "test/labels/03.txt": "bbad46a203bf72fa2666bd7b28fb48bdad2e401c2f4d52fef2f8395d13fc6a6d",
    "test/labels/04.txt": "d2a7909432cfee5af8b863bfff69743ee65dd5971a12510c0d1f2bccc97e7b3c",
    "train/images/01.png": "075839d8bf45fd16e41e9ba9f2ba7f6767c86bc2da7599c3c9b2a090d7423f00",
    "train/images/02.png": "9cc58e0682dfdede3ffcd894909d4dc24ae2c0f78f4512cf4837fb119447015b",
    "train/labels/01.txt": "27870ce778e19c56104999640fb7fcdc18751cab266b594e4c462c4960284748",
    "train/labels/02.txt": "4ea54c88845a9d9e59da0f008b0d72db82edd68ec37f615a00b016e9727aed22",
}
COVERAGE_REF_FIXTURE_TREE_SHA256 = {
    "main": "d372c90e9203550f627f044a29a0f35241a97df216e024f96e45d7d9267ba337",
    "candidate": "2f6e47474450aae0443d6b0e755657f36f81bcd753f6420e0cd03d0c0376782e",
}
REAL_KMEANS_FIXTURE_TREE_SHA256 = "819744ee86027ee129e254fb041c2f19520d1f094e7592ea25edb7a18a925b11"


def _pinned_dinov2_models_dir(tmp_path: Path) -> Path:
    configured_models = Path(
        os.environ.get("LV_TEST_MODELS_DIR") or os.environ.get("LV_MODELS_DIR") or ROOT / "models"
    ).expanduser().resolve()
    configured_weights = (
        configured_models / "dinov2_vits14" / "dinov2_vits14.pth",
        configured_models / "dinov2_vits14.pth",
    )
    cache_weights = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vits14_pretrain.pth"
    source_weights = next((path for path in configured_weights if path.is_file()), None)
    if source_weights is None and cache_weights.is_file():
        source_weights = cache_weights
    if source_weights is None:
        pytest.skip("pinned real DINOv2-small weights are unavailable in LV models or the local torch cache")
    assert hashlib.sha256(source_weights.read_bytes()).hexdigest() == DINOV2_VITS14_SHA256, \
        "the local model weights differ from the LV DINOv2-small baseline"

    models_dir = tmp_path / "models"
    model_dir = models_dir / "dinov2_vits14"
    model_dir.mkdir(parents=True)
    weights = model_dir / "dinov2_vits14.pth"
    try:
        os.link(source_weights, weights)
    except OSError:
        shutil.copyfile(source_weights, weights)
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == DINOV2_VITS14_SHA256
    return models_dir


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pinned_chinese_clip_models_dir() -> Path:
    configured_models = Path(
        os.environ.get("LV_TEST_MODELS_DIR")
        or os.environ.get("LV_MODELS_DIR")
        or ROOT / "models"
    ).expanduser().resolve()
    model_dir = configured_models / CHINESE_CLIP_MODEL_ID
    weights = model_dir / "pytorch_model.bin"
    if not (model_dir / "config.json").is_file() or not weights.is_file():
        pytest.skip("pinned Chinese-CLIP weights are unavailable in the configured model directory")
    assert _sha256_file(weights) == CHINESE_CLIP_PYTORCH_SHA256, (
        "the local Chinese-CLIP weights differ from the pinned LV baseline"
    )
    return configured_models


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


def _fixture_anomaly_model_builder(request: dict, *, progress=None) -> dict:
    """Stable artifact fixture for visible React persistence and restart coverage."""
    if progress:
        progress(100, 100, "Fixture Normal Bank ready")
    return {
        "meta": {
            "model": request["model_id"], "target_res": 224, "score_mode": "patch",
            "object_source": "yolo", "patch_dim": 3, "obj_dim": 2, "n_objects": 5,
            "label_semantic": "object",
            "embedding_recipe": {"model_id": request["model_id"], "checkpoint_id": "fixture-sha"},
        },
        "bank_vectors": np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        "projection": {
            "mean": np.asarray([0, 0], dtype=np.float32),
            "components": np.asarray([[1, 0], [0, 1]], dtype=np.float32),
            "ref_coords": np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
            "good_obj_emb": np.asarray([[1, 0], [0, 1]], dtype=np.float32),
            "tau": 0.25,
            "l2norm": False,
        },
        "fewshot": [], "head": None, "built_at": 1_798_000_000.0,
        "summary": {
            "n_images": 5, "n_objects": 5, "has_normal_bank": True,
            "has_head": False, "head_status": "只做異常偵測(不分種類)", "eligible_classes": [],
        },
    }


def _serve_anomaly_worker(port: int, cache_root: str) -> None:
    os.environ["LV_CACHE_DIR"] = cache_root
    os.environ["LV_CASES_DIR"] = str(Path(cache_root) / "cases")
    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: _worker_fixture_model,
        anomaly_model_builder=_fixture_anomaly_model_builder,
    )
    uvicorn.run(application, host="127.0.0.1", port=port, log_level="error", access_log=False)


def _start_anomaly_worker(cache_root: Path):
    port = _free_port()
    process = multiprocessing.get_context("spawn").Process(
        target=_serve_anomaly_worker,
        args=(port, str(cache_root)),
        daemon=True,
    )
    process.start()
    base_url = f"http://127.0.0.1:{port}"
    _wait_http(f"{base_url}/api/v1/health", process)
    return process, base_url


def _stop_worker(process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def _serve_curation_worker(port: int, cache_root: str) -> None:
    os.environ["LV_CACHE_DIR"] = cache_root
    os.environ["LV_OUTPUT_DIR"] = str(Path(cache_root) / "output")
    os.environ["CIM_LOG_DIR"] = str(Path(cache_root) / "labeling-inbox")
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


def _start_annotation_react_server(
    repo_root: Path,
    app_data_dir: Path,
    port: int,
    log_path: Path,
):
    """Run the checked-out ANnoTation React app and its real local service."""
    web_dir = repo_root / "web" / "dist"
    if not (repo_root / "annotation_web" / "server.py").is_file():
        pytest.fail(f"ANnoTation repo has no local web service: {repo_root}")
    if not (web_dir / "index.html").is_file():
        pytest.fail(f"Build the ANnoTation React app before this E2E: {web_dir}")
    python = os.environ.get("ANNOTATION_PYTHON") or sys.executable
    command = [
        python, "-m", "annotation_web.server",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--app-data-dir", str(app_data_dir),
        "--web-dir", str(web_dir),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    deadline = time.monotonic() + 30
    health_url = f"http://127.0.0.1:{port}/api/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(
                "ANnoTation local service exited before becoming ready:\n"
                + log_path.read_text(encoding="utf-8", errors="replace")
            )
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    return process, f"http://127.0.0.1:{port}"
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    _stop_annotation_react_server(process)
    pytest.fail(
        "ANnoTation local service did not become ready:\n"
        + log_path.read_text(encoding="utf-8", errors="replace")
    )


def _stop_annotation_react_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


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
    """AC2/AC3/CS-01: real DINOv2 projection and class-similarity numerical journey."""
    tests_path = str(ROOT / "tests")
    if tests_path not in sys.path:
        sys.path.insert(0, tests_path)
    from class_similarity_oracle import almost, oracle_analyze

    models_dir = _pinned_dinov2_models_dir(tmp_path)
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
            run_ids = []
            label_quality_payloads = []

            def capture_run(response):
                if _is_run_summary_response(response) and response.ok:
                    run_ids.append(urlparse(response.url).path.rstrip("/").rsplit("/", 1)[-1])
                    run_payloads.append(response.json())

            def capture_label_quality(response):
                if response.ok and "/label-quality" in urlparse(response.url).path:
                    label_quality_payloads.append(response.json())

            page.on("response", capture_run)
            page.on("response", capture_label_quality)
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

            real_run_id = run_ids[-1]
            with application.state.lock:
                raw_embeddings = application.state.raw_embeddings[real_run_id]["dinov2_vits14"].copy()
            assert raw_embeddings.shape == (4, 384)
            assert np.isfinite(raw_embeddings).all() and np.any(raw_embeddings != 0)
            page.get_by_test_id("open-class-similarity").click()
            expect(page.get_by_test_id("class-similarity-workspace")).to_be_visible()
            page.get_by_role("button", name="Calculate class similarity").click()
            expect(page.get_by_test_id("class-similarity-summary")).to_contain_text("eligible", timeout=30_000)

            with application.state.lock:
                similarity_jobs = [
                    job for job in application.state.class_similarity_jobs.values()
                    if job.run_id == real_run_id
                ]
            assert len(similarity_jobs) == 1
            similarity_job = similarity_jobs[0]
            assert similarity_job.state == "completed"
            prepared = similarity_job.prepared
            actual_items = similarity_job.analysis.items
            assert prepared.source_verified is True
            assert np.array_equal(prepared.embeddings, raw_embeddings)
            expected_items = oracle_analyze(
                prepared.embeddings,
                prepared.labels,
                prepared.record_ids,
                eligibility=prepared.eligibility,
                invalid_reasons=prepared.invalid_reasons,
                k=similarity_job.request.k,
            )
            assert len(actual_items) == len(expected_items) == 4
            for actual, expected in zip(actual_items, expected_items, strict=True):
                assert actual.record_id == expected.record_id
                assert actual.rival_class == expected.rival_class
                assert actual.same_k == expected.same_k
                assert actual.rival_k == expected.rival_k
                assert almost(actual.same, expected.same)
                assert almost(actual.other, expected.other)
                assert almost(actual.margin, expected.margin)
                assert [neighbor.record_id for neighbor in actual.same_neighbors] == [
                    neighbor.record_id for neighbor in expected.same_neighbors
                ]
                assert all(
                    almost(actual_neighbor.cosine, expected_neighbor.cosine)
                    for actual_neighbor, expected_neighbor in zip(
                        actual.same_neighbors, expected.same_neighbors, strict=True
                    )
                )
                assert [neighbor.record_id for neighbor in actual.rival_neighbors] == [
                    neighbor.record_id for neighbor in expected.rival_neighbors
                ]
                assert all(
                    almost(actual_neighbor.cosine, expected_neighbor.cosine)
                    for actual_neighbor, expected_neighbor in zip(
                        actual.rival_neighbors, expected.rival_neighbors, strict=True
                    )
                )
            page.get_by_role("button", name="Return to Visualize").click()

            page.get_by_test_id("open-label-quality").click()
            expect(page.get_by_test_id("label-quality-workspace")).to_be_visible()
            expect(page.get_by_label("Label quality K")).to_have_value("10")
            page.get_by_test_id("label-quality-calculate").click()
            expect(page.get_by_test_id("label-quality-summary")).to_contain_text("K=10", timeout=30_000)
            assert label_quality_payloads, "React UI did not request label-quality results"
            label_quality = label_quality_payloads[-1]
            assert label_quality["run_id"] == real_run_id
            assert label_quality["model_id"] == "dinov2_vits14"
            assert label_quality["k"] == 10
            with application.state.lock:
                label_records = list(application.state.runs[real_run_id]["records"])
                label_embeddings = application.state.raw_embeddings[real_run_id]["dinov2_vits14"].copy()
            from object_eval import neighbor_confusion

            expected_label_quality = neighbor_confusion(
                label_embeddings,
                [str(record.get("label") or "") for record in label_records],
                k=10,
            )
            assert label_quality["classes"] == expected_label_quality["classes"]
            np.testing.assert_array_equal(label_quality["counts"], expected_label_quality["counts"])
            np.testing.assert_allclose(
                label_quality["confusion"], expected_label_quality["confusion"],
                rtol=1e-7, atol=1e-8,
            )
            expected_mismatches = sorted(
                (item for item in expected_label_quality["per_object"] if item["mismatch"]),
                key=lambda item: item["suggested_frac"], reverse=True,
            )[:30]
            assert len(label_quality["candidates"]) == len(expected_mismatches)
            for actual, expected in zip(label_quality["candidates"], expected_mismatches, strict=True):
                assert actual["index"] == expected["idx"]
                assert actual["own"] == expected["own"]
                assert actual["suggested"] == expected["suggested"]
                assert actual["suggested_frac"] == pytest.approx(expected["suggested_frac"])
                assert actual["own_frac"] == pytest.approx(expected["own_frac"])
                assert actual["disagreement"] == pytest.approx(expected["disagreement"])
            page.get_by_role("button", name="Return to Visualize").click()

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


def test_lv_web_01_ic_01_nativeapp_folder_grant_visible_react_journey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """React forwards the opaque picker grant and the worker resolves it before analysis."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    dataset = _fixture(tmp_path / "nativeapp-grant-fixture")
    selected_folder = dataset / "train"
    source_hashes = {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }
    cache_root = tmp_path / "nativeapp-grant-cache"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("NATIVEAPP_APP_ID", "visuallatent")
    monkeypatch.setenv("NATIVEAPP_API", "http://127.0.0.1:8765")
    application = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=_worker_fixture_model,
    )
    server = server_thread = None
    nativeapp_calls = []
    try:
        server, server_thread, base_url = _start_api(application)

        def resolve_folder_grant(request, timeout):
            nativeapp_calls.append(request)
            assert request.full_url == "http://127.0.0.1:8765/host/grants/resolve"
            assert request.get_method() == "POST"
            assert request.get_header("Origin") is None
            assert timeout == 10
            assert json.loads(request.data) == {
                "app_id": "visuallatent", "grant_id": "hostgrant_fixture",
            }
            return BytesIO(json.dumps({
                "app_id": "visuallatent",
                "grant_id": "hostgrant_fixture",
                "path": str(selected_folder),
                "display_name": selected_folder.name,
            }).encode("utf-8"))

        monkeypatch.setattr(urllib.request, "urlopen", resolve_folder_grant)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_timeout(15_000)
            folder = json.dumps(str(selected_folder))
            page.add_init_script(
                "window.cimHost = { chooseFile: async () => ({ path: " + folder
                + ", grant_id: 'hostgrant_fixture', display_name: 'train' }) };"
            )
            run_payloads = []
            create_payloads = []
            page.on(
                "response",
                lambda response: run_payloads.append(response.json())
                if _is_run_summary_response(response) and response.ok else None,
            )
            page.on(
                "request",
                lambda request: create_payloads.append(json.loads(request.post_data))
                if request.method == "POST" and urlparse(request.url).path == "/api/v1/visualize/jobs"
                else None,
            )
            page.goto(base_url, wait_until="networkidle")
            page.locator(".host-picker").click()
            expect(page.get_by_role("list", name="Selected folders")).to_contain_text("train")
            page.get_by_test_id("model-choice-dinov2_vits14").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=30_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(3)
            assert len(create_payloads) == 1
            assert create_payloads[0]["folder_grant_ids"] == ["hostgrant_fixture"]
            assert len(nativeapp_calls) == 1
            assert run_payloads and run_payloads[-1]["embedding_shape"] == [3, 384]
            assert "hostgrant_fixture" not in json.dumps(run_payloads[-1])
            assert str(dataset) not in json.dumps(run_payloads[-1])
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }


def test_lv_web_01_cs_01_visible_class_similarity_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Visible React flow ranks same/rival evidence and reads the selected source image."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    dataset = _fixture(tmp_path / "class-similarity-fixture")
    source_hashes = {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "cases"))
    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda model_id: _worker_fixture_model(model_id),
    )
    server = None
    server_thread = None
    restarted_worker = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1000})
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("model-choice-fixture-model").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in (dataset / "train", dataset / "test"):
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=30_000)

            page.get_by_test_id("open-class-similarity").click()
            expect(page.get_by_test_id("class-similarity-workspace")).to_be_visible()
            page.get_by_role("button", name="Calculate class similarity").click()
            expect(page.get_by_test_id("class-similarity-summary")).to_contain_text("eligible", timeout=15_000)
            expect(page.get_by_test_id("class-similarity-row")).to_have_count(4)
            expect(page.get_by_role("img", name="margin distribution scatter plot")).to_be_visible()
            expect(page.get_by_role("img", name="Workload curve")).to_be_visible()
            page.get_by_label("Visible classes").select_option(label=["apple"])
            expect(page.get_by_test_id("class-similarity-candidate-count")).to_contain_text("4 candidates")
            page.get_by_label("Visible classes").select_option(label=["apple", "banana"])
            page.get_by_label("Class similarity axis range").select_option("data")
            page.get_by_role("button", name="Reset full range").click()
            candidate_count = page.get_by_test_id("class-similarity-candidate-count")
            page.get_by_label("Class similarity threshold", exact=True).fill("-2")
            expect(candidate_count).to_contain_text("0 candidates")
            page.get_by_label("Only candidates").check()
            expect(page.get_by_test_id("class-similarity-row")).to_have_count(0)
            page.get_by_label("Only candidates").uncheck()
            expect(page.get_by_test_id("class-similarity-row")).to_have_count(4)
            page.get_by_label("Class similarity threshold", exact=True).fill("2")
            expect(candidate_count).to_contain_text("4 candidates")
            page.get_by_label("Visible classes").select_option(label=["apple"])
            page.get_by_role("button", name="Add all candidates to Visualize selection (4)").click()
            expect(page.get_by_role("status").filter(has_text="Added 4 current candidate(s) across all classes")).to_be_visible()
            page.get_by_label("Visible classes").select_option(label=["apple", "banana"])
            page.get_by_label("Only margin < 0").check()
            assert page.get_by_test_id("class-similarity-row").count() <= 4
            page.get_by_label("Only margin < 0").uncheck()
            expect(page.get_by_test_id("class-similarity-row")).to_have_count(4)
            page.get_by_role("button", name="Select current candidates (4)").click()
            page.get_by_role("button", name="Add selected to Curation").click()
            expect(page.get_by_role("status").filter(has_text="Added 4; skipped 0")).to_be_visible(timeout=15_000)

            page.get_by_role("button", name="Clear selection (4)").click()
            scatter = page.locator("svg[aria-label='margin distribution scatter plot']")
            scatter.scroll_into_view_if_needed()
            scatter_box = scatter.bounding_box()
            assert scatter_box is not None
            page.mouse.move(scatter_box["x"] + scatter_box["width"] * 72 / 800,
                            scatter_box["y"] + scatter_box["height"] * 18 / 390)
            page.mouse.down()
            page.mouse.move(scatter_box["x"] + scatter_box["width"] * 780 / 800,
                            scatter_box["y"] + scatter_box["height"] * 342 / 390, steps=5)
            page.mouse.up()
            box_selected = page.get_by_role("button", name=re.compile(r"Clear selection \([1-4]\)"))
            expect(box_selected).to_be_enabled()
            box_selected.click()
            rows = page.get_by_test_id("class-similarity-row")
            rows.nth(0).locator("input[type=checkbox]").check()
            rows.nth(1).locator("input[type=checkbox]").check()
            page.get_by_label("Evaluation template scope").select_option("selected")
            with page.expect_download() as download_info:
                page.get_by_role("button", name="Download evaluation template").click()
            tuning_download = download_info.value
            tuning_path = tmp_path / "class_similarity_tuning.json"
            tuning_template = json.loads(Path(tuning_download.path()).read_text(encoding="utf-8"))
            assert tuning_template["purpose"] == "tuning"
            assert len(tuning_template["frozen_ids"]) == 2
            for index, review in enumerate(tuning_template["reviews"]):
                review["verdict"] = "incorrect" if index == 0 else "correct"
            tuning_path.write_text(json.dumps(tuning_template), encoding="utf-8")
            page.get_by_label("Upload class similarity evaluation JSON").set_input_files(str(tuning_path))
            expect(page.get_by_test_id("class-similarity-evaluation-results")).to_be_visible(timeout=15_000)
            expect(page.get_by_test_id("class-similarity-evaluation-results")).to_contain_text("TP")
            expect(page.get_by_test_id("class-similarity-evaluation-results")).to_contain_text("precision")
            expect(page.get_by_role("img", name="Precision recall curve")).to_be_visible()
            page.get_by_role("button", name="TP 1").click()
            expect(page.get_by_test_id("class-similarity-row")).to_have_count(1)
            page.get_by_role("button", name="TP 1").click()

            page.get_by_role("button", name="Save operating point").click()
            expect(page.get_by_role("button", name="Restore saved operating point")).to_be_visible()
            page.get_by_label("Class similarity threshold", exact=True).fill("-1")
            page.get_by_role("button", name="Restore saved operating point").click()
            expect(page.get_by_label("Class similarity threshold", exact=True)).to_have_value("2")

            page.get_by_role("button", name="Clear selection (2)").click()
            rows.nth(2).locator("input[type=checkbox]").check()
            rows.nth(3).locator("input[type=checkbox]").check()
            page.get_by_label("Evaluation purpose").select_option("heldout")
            with page.expect_download() as heldout_download_info:
                page.get_by_role("button", name="Download evaluation template").click()
            heldout_template = json.loads(Path(heldout_download_info.value.path()).read_text(encoding="utf-8"))
            assert heldout_template["purpose"] == "heldout"
            assert set(heldout_template["frozen_ids"]).isdisjoint(tuning_template["frozen_ids"])
            assert {row["source_group_id"] for row in heldout_template["record_catalog"]}.isdisjoint(
                {row["source_group_id"] for row in tuning_template["record_catalog"]}
            )
            for index, review in enumerate(heldout_template["reviews"]):
                review["verdict"] = "incorrect" if index == 0 else "correct"
            heldout_path = tmp_path / "class_similarity_heldout.json"
            heldout_path.write_text(json.dumps(heldout_template), encoding="utf-8")
            page.get_by_label("Upload class similarity evaluation JSON").set_input_files(str(heldout_path))
            expect(page.get_by_role("status").filter(has_text="Heldout verdicts validated")).to_be_visible(timeout=15_000)
            expect(page.get_by_role("img", name="Precision recall curve")).to_have_count(0)
            expect(page.get_by_label("Class similarity threshold", exact=True)).to_be_disabled()
            expect(page.get_by_role("button", name="No candidates")).to_be_disabled()

            with page.expect_download() as report_download_info:
                page.get_by_role("button", name="CSV", exact=True).first.click()
            report_download = report_download_info.value
            exported_rows = list(csv.DictReader(Path(report_download.path()).open(encoding="utf-8-sig")))
            assert len(exported_rows) == 4
            assert all(row["is_candidate"] == "1" for row in exported_rows)

            page.get_by_test_id("class-similarity-row").first.get_by_role("button").click()
            preview = page.get_by_test_id("class-similarity-preview")
            expect(preview).to_be_visible()
            expect(preview).to_have_js_property("naturalWidth", 64)
            expect(page.get_by_role("heading", name="Same-class neighbors")).to_be_visible()
            expect(page.get_by_role("heading", name="Rival-class neighbors")).to_be_visible()
            selected_name = page.get_by_test_id("class-similarity-row").first.locator("button").inner_text()
            page.get_by_role("button", name="Open record in Visualize").click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text(selected_name)
            expect(page.get_by_test_id("selection-count")).to_have_text("4 selected")
            page.get_by_test_id("open-curation").click()
            expect(page.get_by_test_id("curation-item")).to_have_count(4)
            expect(page.get_by_test_id("curation-item").first).to_contain_text("類別相似度")

            server.should_exit = True
            server_thread.join(timeout=10)
            assert not server_thread.is_alive()
            server = None
            server_thread = None
            restarted_worker, restarted_url = _start_curation_worker(cache_root)
            assert restarted_worker.pid != os.getpid()
            page.goto(restarted_url, wait_until="networkidle")
            page.get_by_test_id("open-curation").click()
            expect(page.get_by_test_id("curation-item")).to_have_count(4)
            expect(page.get_by_test_id("curation-item").first).to_contain_text("類別相似度")
            browser.close()
    finally:
        if restarted_worker is not None:
            _stop_worker(restarted_worker)
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }


def test_lv_web_01_ic_02_multi_model_projections_visible_journey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Visible React keeps record identity across two model views and all five projections."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    dataset = _fixture(tmp_path / "classifier-multimodel")
    source_before = {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    selected_models = ["dinov2_vits14", "resnet50"]
    embed_calls: list[tuple[str, str]] = []

    def fixture_loader(model_id: str):
        def embed(path: Path) -> np.ndarray:
            pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = pixels.mean(axis=(0, 1)) / 255.0 + 0.01
            vector[3] = len(path.name) / 10.0
            vector[4] = 0.25 if model_id == selected_models[0] else -0.25
            embed_calls.append((model_id, str(path)))
            return vector
        return embed

    application = create_app(
        model_catalog=lambda: selected_models,
        model_loader=fixture_loader,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.set_default_timeout(15_000)
            run_payloads = []
            page.on(
                "response",
                lambda response: run_payloads.append(response.json())
                if _is_run_summary_response(response) and response.ok
                else None,
            )
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("model-choice-dinov2_vits14").check()
            page.get_by_test_id("model-choice-resnet50").check()
            for projection in ("pca", "tsne", "umap", "lda"):
                page.get_by_test_id("projection-" + projection).check()
            assert page.get_by_label("Supervised UMAP").is_checked()
            for folder in (dataset / "train", dataset / "test"):
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()

            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=120_000)
            points = page.get_by_test_id("embedding-point")
            expect(points).to_have_count(4)
            assert run_payloads, "React UI did not load the completed local API run"
            run = run_payloads[-1]
            assert run["models"] == selected_models
            assert run["embedding_shapes"] == {
                model_id: [4, 384] for model_id in selected_models
            }
            assert len(embed_calls) == 8
            expected_views = {
                (model_id, method)
                for model_id in selected_models
                for method in ("pca", "tsne", "umap", "lda", "sumap")
            }
            actual_views = {(view["model_id"], view["method"]) for view in run["views"]}
            assert actual_views == expected_views
            for view in run["views"]:
                assert [point["index"] for point in view["points"]] == list(range(4))
                assert all(np.isfinite([point["x"], point["y"], point["z"]]).all()
                           for point in view["points"])

            page.get_by_label("Filter by class").select_option(label="apple")
            expect(page.get_by_test_id("embedding-point")).to_have_count(2)
            page.get_by_label("Filter by class").select_option(label="All classes")
            page.get_by_label("Filter by split").select_option("test")
            expect(page.get_by_test_id("embedding-point")).to_have_count(1)
            page.get_by_label("Filter by split").select_option("all")

            page.get_by_test_id("embedding-point").nth(2).click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text("image-03.png")
            page.get_by_label("Display model").select_option("resnet50")
            expect(page.get_by_test_id("selected-image-name")).to_have_text("image-03.png")
            page.get_by_label("Display method").select_option("umap")
            expect(page.get_by_test_id("selected-image-name")).to_have_text("image-03.png")
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)
            page.get_by_test_id("dimension-3d").click()
            expect(page.get_by_test_id("embedding-plot-3d")).to_be_visible(timeout=60_000)
            expect(page.get_by_test_id("selected-image-name")).to_have_text("image-03.png")
            page.get_by_test_id("dimension-2d").click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text("image-03.png")

            source_after = {
                path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in dataset.rglob("*")
                if path.is_file()
            }
            assert source_after == source_before
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
            run_id = next(iter(application.state.runs))
            assert "not determine image quality" in page.get_by_test_id("ranking-interpretation").inner_text()

            outlier_names = page.locator("[data-testid='ranked-result'] strong").all_inner_texts()
            rows.first.locator("button[aria-label^='Open rank']").click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text(outlier_names[0])
            expect(page.get_by_test_id("selected-image-navigation")).to_contain_text("1 / 4 ranked")
            page.get_by_role("button", name="Next ranked image").click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text(outlier_names[1])
            expect(page.get_by_test_id("selected-image-navigation")).to_contain_text("2 / 4 ranked")

            page.get_by_label("Ranking criterion").select_option("disagreement")
            expect(page.get_by_test_id("ranking-status")).to_contain_text("highest first")
            expect(page.get_by_test_id("ranked-result")).to_have_count(4)
            disagreement_names = page.locator("[data-testid='ranked-result'] strong").all_inner_texts()
            first_disagreement = page.get_by_test_id("ranked-result").first
            curated_index = int(first_disagreement.get_attribute("data-index"))
            displayed_score = float(first_disagreement.locator("small").inner_text().split("·")[-1].strip())
            page.get_by_test_id(f"curate-ranked-{curated_index}").click()
            expect(page.locator(".alert-info")).to_contain_text("Added 1 item(s) to curation")
            curation = page.request.get(base_url + "/api/v1/curation/items").json()
            ranked_item = next(item for item in curation["items"] if item["index"] == curated_index)
            assert ranked_item["source"] == "disagreement"
            assert ranked_item["score"] == pytest.approx(displayed_score, abs=0.00005)
            page.get_by_test_id("ranked-result").first.locator("button[aria-label^='Open rank']").click()
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


def test_lv_web_01_rank_01_real_dinov2_visible_streamlit_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Pinned DINOv2 rankings match Streamlit scores and keep their honest framing."""
    import app as legacy_app

    models_dir = _pinned_dinov2_models_dir(tmp_path)
    dataset = _fixture(tmp_path / "ranking-real")
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*.png")
    }
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "ranking-real-cache"))
    application = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=lambda model_id: load_model(model_id, models_dir=models_dir),
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.set_default_timeout(15_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("model-choice-dinov2_vits14").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in (dataset / "train", dataset / "test"):
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text(
                "Completed", timeout=180_000
            )
            expect(page.get_by_test_id("ranking-workspace")).to_be_visible()
            expect(page.get_by_test_id("ranked-result")).to_have_count(4)

            run_id = next(iter(application.state.runs))
            with application.state.lock:
                run = application.state.runs[run_id]
                raw_embeddings = application.state.raw_embeddings[run_id]["dinov2_vits14"].copy()
            assert raw_embeddings.shape == (4, 384)
            assert np.isfinite(raw_embeddings).all() and np.any(raw_embeddings != 0)
            labels = [record["label"] for record in run["records"]]
            k = min(5, len(run["records"]) - 1)
            legacy_scores = {
                "outlier": legacy_app.compute_outlier_scores(
                    raw_embeddings, raw_embeddings, k=k, candidates_in_reference=True
                ),
                "disagreement": legacy_app.compute_label_disagreement(
                    raw_embeddings, labels, k=k
                ),
            }

            def assert_metric_matches_streamlit(metric: str) -> list[int]:
                response = page.request.get(
                    base_url + f"/api/v1/runs/{run_id}/rankings",
                    params={"model_id": "dinov2_vits14", "metric": metric},
                )
                assert response.status == 200
                payload = response.json()
                assert payload["k"] == k
                assert payload["score_direction"] == "descending"
                scores = legacy_scores[metric]
                expected_indices = sorted(
                    range(len(scores)), key=lambda index: (-float(scores[index]), index)
                )
                actual_results = payload["results"]
                assert [row["index"] for row in actual_results] == expected_indices
                np.testing.assert_allclose(
                    [row["score"] for row in actual_results],
                    [scores[index] for index in expected_indices],
                    rtol=1e-7,
                    atol=1e-7,
                )
                return expected_indices

            def assert_visible_rows(metric: str, expected_indices: list[int]) -> None:
                rows = page.get_by_test_id("ranked-result")
                expect(rows).to_have_count(len(expected_indices))
                assert [int(row.get_attribute("data-index")) for row in rows.all()] == expected_indices
                visible_scores = [
                    float(row.locator("small").inner_text().split("·")[-1].strip())
                    for row in rows.all()
                ]
                np.testing.assert_allclose(
                    visible_scores,
                    [legacy_scores[metric][index] for index in expected_indices],
                    atol=0.00005,
                )

            outlier_indices = assert_metric_matches_streamlit("outlier")
            assert_visible_rows("outlier", outlier_indices)
            assert "not determine image quality" in page.get_by_test_id(
                "ranking-interpretation"
            ).inner_text()

            with page.expect_response(
                lambda response: "/rankings?" in response.url
                and "metric=disagreement" in response.url
                and response.ok
            ):
                page.get_by_label("Ranking criterion").select_option("disagreement")
            expect(page.get_by_test_id("ranking-status")).to_contain_text("highest first")
            disagreement_indices = assert_metric_matches_streamlit("disagreement")
            assert_visible_rows("disagreement", disagreement_indices)

            with page.expect_download() as download_info:
                page.get_by_test_id("ranking-export").click()
            download = download_info.value
            assert download.suggested_filename == "disagreement_ranking.csv"
            with Path(download.path()).open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            assert [int(row["index"]) for row in csv_rows] == disagreement_indices
            np.testing.assert_allclose(
                [float(row["score"]) for row in csv_rows],
                [legacy_scores["disagreement"][index] for index in disagreement_indices],
                rtol=1e-7,
                atol=1e-7,
            )
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*.png")
    }, "real-model ranking changed its source images"


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


def test_lv_web_01_ic_04_real_model_matches_streamlit_kmeans_and_drilldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Pinned DINOv2 UI run matches Streamlit KMeans and private-vector drilldown."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    import app as legacy_app

    models_dir = _pinned_dinov2_models_dir(tmp_path)
    dataset = tmp_path / "classifier-real-kmeans" / "train" / "apple"
    colors = [(220, 35, 45), (30, 190, 70), (35, 65, 215), (210, 180, 25)]
    for cluster, color in enumerate(colors):
        for replicate in range(3):
            dataset.mkdir(parents=True, exist_ok=True)
            varied = (color[0] + replicate, color[1] + replicate * 2, color[2])
            Image.new("RGB", (64, 64), varied).save(
                dataset / f"cluster-{cluster}-{replicate}.png"
            )

    source_root = dataset.parent.parent
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_root.rglob("*.png")
    }
    assert _fixture_tree_digest(source_root, source_hashes) == REAL_KMEANS_FIXTURE_TREE_SHA256
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "real-kmeans-cache"))
    application = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=lambda model_id: load_model(model_id, models_dir=models_dir),
    )
    server = server_thread = None
    block_responses: list[tuple[dict, dict]] = []
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.set_default_timeout(15_000)

            def capture_blocks(response):
                if not response.ok or not urlparse(response.url).path.endswith("/blocks"):
                    return
                request = response.request.post_data_json
                if request:
                    try:
                        payload = response.json()
                    except Exception:
                        return
                    block_responses.append((request, payload))

            page.on("response", capture_blocks)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("model-choice-dinov2_vits14").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            page.get_by_label("Image folder").fill(str(dataset.parent))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text(
                "Completed", timeout=180_000
            )
            expect(page.get_by_test_id("embedding-point")).to_have_count(12)

            with application.state.lock:
                run_id = next(reversed(application.state.runs))
                run = application.state.runs[run_id]
                raw_embeddings = application.state.raw_embeddings[run_id][
                    "dinov2_vits14"
                ].copy()
            assert raw_embeddings.shape == (12, 384)
            assert np.isfinite(raw_embeddings).all() and np.any(raw_embeddings != 0)
            pca_view = next(
                view for view in run["views"]
                if view["model_id"] == "dinov2_vits14" and view["method"] == "pca"
            )
            view_points = sorted(pca_view["points"], key=lambda point: point["index"])
            assert len(view_points) == 12
            assert all(
                np.isfinite([point["x"], point["y"], point["z"]]).all()
                for point in view_points
            )

            page.get_by_label("Display method").select_option("pca")
            page.get_by_test_id("dimension-3d").click()
            expect(page.get_by_test_id("embedding-plot-3d")).to_be_visible(timeout=60_000)
            page.get_by_test_id("kmeans-count").focus()
            page.get_by_test_id("kmeans-count").press("Home")
            expect(page.get_by_test_id("kmeans-status")).to_contain_text(
                "4 個區塊", timeout=30_000
            )
            expect(page.get_by_test_id("cluster-block-select")).to_be_enabled()

            base_request, base_response = next(
                (request, response)
                for request, response in reversed(block_responses)
                if request.get("k") == 4
                and request.get("method") == "pca"
                and request.get("block_id") is None
            )
            assert base_request["model_id"] == "dinov2_vits14"
            assert base_response["k"] == 4
            indices = [int(point["index"]) for point in view_points]
            coordinates = np.asarray(
                [[point["x"], point["y"], point["z"]] for point in view_points],
                dtype=float,
            )
            monkeypatch.setattr(legacy_app.st, "session_state", {})
            legacy_blocks, legacy_centers = legacy_app._viz_kmeans_blocks(
                coordinates, indices, 4, f"real-dinov2-{run_id}"
            )
            react_blocks = {
                int(point["index"]): int(point["block_id"])
                for point in base_response["points"]
            }
            assert react_blocks == legacy_blocks
            react_centers = sorted(
                base_response["centers"], key=lambda center: center["block_id"]
            )
            np.testing.assert_allclose(
                [[center["x"], center["y"], center["z"]] for center in react_centers],
                legacy_centers,
                rtol=1e-6,
                atol=1e-6,
            )

            largest = max(base_response["centers"], key=lambda center: center["count"])
            assert largest["count"] >= 3
            page.get_by_test_id("cluster-block-select").select_option(
                str(largest["block_id"])
            )
            expect(page.get_by_test_id("block-point")).to_have_count(
                largest["count"], timeout=30_000
            )
            selected_request, selected_response = next(
                (request, response)
                for request, response in reversed(block_responses)
                if request.get("k") == 4
                and request.get("block_id") == largest["block_id"]
            )
            assert selected_request["method"] == "pca"
            members = sorted(
                index for index, block_id in legacy_blocks.items()
                if block_id == largest["block_id"]
            )
            assert [point["index"] for point in selected_response["drilldown_points"]] == members
            expected_drilldown = legacy_app.PCA(
                n_components=2, random_state=42
            ).fit_transform(raw_embeddings[members])
            np.testing.assert_allclose(
                [
                    [point["x"], point["y"]]
                    for point in selected_response["drilldown_points"]
                ],
                expected_drilldown,
                rtol=1e-6,
                atol=1e-6,
            )
            expect(page.get_by_test_id("block-drilldown")).to_be_visible()
            expect(page.get_by_test_id("block-drilldown")).to_contain_text(
                f"#{largest['block_id']}"
            )
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_root.rglob("*.png")
    }, "real-model KMeans flow changed its source images"


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
            run_id = next(iter(application.state.runs))

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

            curated_index = switched_order[0]
            expected_search = page.request.post(
                base_url + f"/api/v1/runs/{run_id}/similar",
                data={"model_id": "fixture-model-b", "query_index": 1, "k": 4},
            )
            assert expected_search.status == 200
            expected_score = next(
                item["distance"] for item in expected_search.json()["results"]
                if item["index"] == curated_index
            )
            page.get_by_test_id(f"curate-similar-{curated_index}").click()
            expect(page.locator(".alert-info")).to_contain_text("Added 1 item(s) to curation")
            curation = page.request.get(base_url + "/api/v1/curation/items").json()
            similar_item = next(item for item in curation["items"] if item["index"] == curated_index)
            assert similar_item["source"] == "similar"
            assert similar_item["score"] == pytest.approx(expected_score)

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


def test_lv_web_01_search_01_real_model_visible_streamlit_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Real DINOv2 React search matches Streamlit neighbors, distances, and CSV."""
    import app as legacy_app

    models_dir = _pinned_dinov2_models_dir(tmp_path)
    dataset = _fixture(tmp_path / "similar-real")
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*.png")
    }
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "similar-real-cache"))
    application = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=lambda model_id: load_model(model_id, models_dir=models_dir),
    )
    server = server_thread = None
    similar_responses: list[dict] = []
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.set_default_timeout(15_000)

            def capture_similar(response):
                if (
                    response.ok
                    and response.request.method == "POST"
                    and urlparse(response.url).path.endswith("/similar")
                ):
                    similar_responses.append(response.json())

            page.on("response", capture_similar)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("model-choice-dinov2_vits14").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in (dataset / "train", dataset / "test"):
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text(
                "Completed", timeout=180_000
            )
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)
            run_id = next(iter(application.state.runs))
            with application.state.lock:
                raw_embeddings = application.state.raw_embeddings[run_id]["dinov2_vits14"].copy()
                run = application.state.runs[run_id]
            assert raw_embeddings.shape == (4, 384)
            assert np.isfinite(raw_embeddings).all() and np.any(raw_embeddings != 0)

            def assert_visible_search_matches_streamlit(query_index: int) -> list[int]:
                class ExactCosineIndex:
                    """Deterministic exact adapter for the optional HNSW query seam."""

                    def set_ef(self, _value: int) -> None:
                        return None

                    def knn_query(self, query: np.ndarray, k: int):
                        vectors = np.asarray(raw_embeddings, dtype=np.float32)
                        query_vector = np.asarray(query, dtype=np.float32).reshape(-1)
                        norms = np.linalg.norm(vectors, axis=1)
                        denominator = norms * np.linalg.norm(query_vector)
                        similarities = np.zeros(len(vectors), dtype=np.float32)
                        np.divide(
                            vectors @ query_vector,
                            denominator,
                            out=similarities,
                            where=denominator > 0,
                        )
                        distances = np.maximum(
                            0.0, 1.0 - np.clip(similarities, -1.0, 1.0)
                        )
                        indices = np.arange(len(vectors))
                        order = np.lexsort((indices, distances))[:k]
                        return order.reshape(1, -1), distances[order].reshape(1, -1)

                expected_indices, expected_distances = legacy_app.find_similar_indices(
                    raw_embeddings,
                    query_index,
                    k=len(raw_embeddings) - 1,
                    nn_index=ExactCosineIndex(),
                )
                payload = next(
                    item for item in reversed(similar_responses)
                    if item["query_index"] == query_index
                )
                assert payload["model_id"] == "dinov2_vits14"
                actual_indices = [item["index"] for item in payload["results"]]
                actual_distances = [item["distance"] for item in payload["results"]]
                assert actual_indices == expected_indices
                np.testing.assert_allclose(
                    actual_distances, expected_distances, rtol=1e-6, atol=1e-6
                )

                panel = page.get_by_test_id("similar-panel")
                rows = panel.get_by_test_id("similar-result")
                expect(rows).to_have_count(len(expected_indices))
                visible_indices = [int(row.get_attribute("data-index")) for row in rows.all()]
                assert visible_indices == expected_indices
                visible_distances = [
                    row.locator(".similar-result-info span").inner_text().rsplit("d=", 1)[1]
                    for row in rows.all()
                ]
                assert visible_distances == [f"{distance:.4f}" for distance in expected_distances]
                return expected_indices

            page.get_by_test_id("embedding-point").nth(0).click()
            page.get_by_test_id("similar-start").click()
            panel = page.get_by_test_id("similar-panel")
            expect(panel.get_by_test_id("similar-status")).to_contain_text("3 result(s)")
            first_query = int(page.get_by_test_id("embedding-point").nth(0).get_attribute("data-index"))
            first_indices = assert_visible_search_matches_streamlit(first_query)

            expected_records = [
                {**record, "path": path}
                for record, path in zip(run["records"], run["image_paths"], strict=True)
            ]
            expected_csv = legacy_app.records_to_csv(
                expected_records, [first_query, *first_indices]
            )
            with page.expect_download() as download_info:
                page.get_by_test_id("similar-export").click()
            actual_csv = Path(download_info.value.path()).read_text(encoding="utf-8")
            assert actual_csv == expected_csv

            next_query = first_indices[0]
            previous_response_count = len(similar_responses)
            next_filename = run["records"][next_query]["filename"]
            panel.get_by_role("button", name=f"Use {next_filename} as query").click()
            expect(panel).to_contain_text(f"Query: {next_filename}")
            deadline = time.monotonic() + 15
            while len(similar_responses) <= previous_response_count and time.monotonic() < deadline:
                time.sleep(0.05)
            assert len(similar_responses) > previous_response_count
            assert_visible_search_matches_streamlit(next_query)
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*.png")
    }, "real-model similar-image search changed its source images"


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
    monkeypatch.setenv("LV_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("CIM_LOG_DIR", str(tmp_path / "labeling-inbox"))

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
            run_id = next(iter(application.state.runs))

            page.get_by_label("Chinese-CLIP text query").fill("反光樣本")
            page.get_by_role("button", name="Search images").click()
            expect(page.get_by_test_id("text-search-result")).to_have_count(4)
            text_order = [int(item.get_attribute("data-index")) for item in page.get_by_test_id("text-search-result").all()]
            assert text_order == [0, 3, 1, 2]
            text_scores = page.request.post(
                base_url + f"/api/v1/runs/{run_id}/text-search",
                data={"model_id": "chinese-clip-fixture", "query": "反光樣本", "k": 4},
            ).json()["results"]
            page.get_by_test_id("curate-text-1").click()
            expect(page.locator(".alert-info")).to_contain_text("Added 1 item(s) to curation")
            text_curation = page.request.get(base_url + "/api/v1/curation/items").json()
            text_item = next(item for item in text_curation["items"] if item["index"] == 1)
            assert text_item["source"] == "search"
            assert text_item["score"] == pytest.approx(
                next(item["distance"] for item in text_scores if item["index"] == 1)
            )
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
            expected_pairs = page.request.post(
                base_url + f"/api/v1/runs/{run_id}/duplicates",
                data={
                    "method": "embedding", "model_id": "chinese-clip-fixture",
                    "max_distance": 0.05, "cross_split_only": True,
                },
            ).json()["results"]
            page.get_by_test_id("curate-duplicate-0-3").click()
            expect(page.locator(".alert-info")).to_contain_text("Added 2 item(s) to curation")
            duplicate_curation = page.request.get(base_url + "/api/v1/curation/items").json()
            duplicate_items = {item["index"]: item for item in duplicate_curation["items"]}
            expected_distance = expected_pairs[0]["distance"]
            assert duplicate_items[0]["source"] == duplicate_items[3]["source"] == "duplicate"
            assert duplicate_items[0]["score"] == duplicate_items[3]["score"] == pytest.approx(expected_distance)
            assert duplicate_items[1]["source"] == "search"
            with page.expect_download() as duplicate_download:
                page.get_by_test_id("duplicate-export").click()
            duplicate_rows = list(csv.reader(Path(duplicate_download.value.path()).read_text(encoding="utf-8").splitlines()))
            assert len(duplicate_rows) == 2
            assert duplicate_rows[1][0:2] == ["0", "image-01.png"]
            assert duplicate_rows[1][4:6] == ["3", "image-04.png"]
            page.get_by_test_id("duplicate-labeling-handoff").click()
            expect(page.get_by_test_id("duplicate-handoff-status")).to_contain_text("Saved 1 pair(s) with 1 unique image(s) to the Labeling inbox")
            handoff_specs = list((tmp_path / "labeling-inbox" / "lv_labeling_handoff").glob("*/_handoff.json"))
            assert len(handoff_specs) == 1
            handoff = json.loads(handoff_specs[0].read_text(encoding="utf-8"))
            assert handoff["task"] == "verify"
            assert handoff["payload"]["kind"] == "duplicate_pair_review"
            assert handoff["payload"]["pairs"][0]["left_filename"] == "image-01.png"
            assert handoff["payload"]["pairs"][0]["right_filename"] == "image-04.png"
            assert handoff["payload"]["pairs"][0]["distance"] == pytest.approx(1 - 0.998 / np.sqrt(0.998**2 + 0.063**2), abs=5e-7)
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)


def test_lv_web_01_search_02_real_chinese_clip_text_search_visible_streamlit_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Real Chinese-CLIP text ranking and image-query pivot match Streamlit."""
    import app as legacy_app
    from _utils import load_text_encoder

    models_dir = _pinned_chinese_clip_models_dir()
    weights_path = models_dir / CHINESE_CLIP_MODEL_ID / "pytorch_model.bin"
    weights_hash = _sha256_file(weights_path)
    dataset = _fixture(tmp_path / "search-02-real")
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*.png")
    }
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "search-02-real-cache"))
    monkeypatch.setenv("LV_OUTPUT_DIR", str(tmp_path / "search-02-real-output"))
    application = create_app(
        model_catalog=lambda: [CHINESE_CLIP_MODEL_ID],
        model_loader=lambda model_id: load_model(model_id, models_dir=models_dir),
        text_model_loader=lambda model_id: load_text_encoder(model_id, models_dir=models_dir),
    )
    server = server_thread = browser = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.set_default_timeout(15_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id(f"model-choice-{CHINESE_CLIP_MODEL_ID}").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in (dataset / "train", dataset / "test"):
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text(
                "Completed", timeout=300_000
            )
            expect(page.get_by_test_id("search-review-workspace")).to_be_visible()
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)

            run_id = next(iter(application.state.runs))
            with application.state.lock:
                run = application.state.runs[run_id]
                raw_embeddings = application.state.raw_embeddings[run_id][
                    CHINESE_CLIP_MODEL_ID
                ].copy()
            assert raw_embeddings.shape == (4, 512)
            assert np.isfinite(raw_embeddings).all()
            assert np.all(np.linalg.norm(raw_embeddings, axis=1) > 0)

            text_encoder = load_text_encoder(CHINESE_CLIP_MODEL_ID, models_dir=models_dir)
            query = "長頸鹿"
            query_vector = text_encoder(query)
            simplified_vector = text_encoder("长颈鹿")
            np.testing.assert_allclose(query_vector, simplified_vector, rtol=1e-6, atol=1e-7)

            class ExactCosineIndex:
                def __init__(self, vectors: np.ndarray):
                    self.vectors = np.asarray(vectors, dtype=np.float32)

                def set_ef(self, _ef: int) -> None:
                    pass

                def knn_query(self, values: np.ndarray, k: int):
                    vector = np.asarray(values, dtype=np.float32).reshape(-1)
                    denominators = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(vector)
                    similarities = np.zeros(len(self.vectors), dtype=np.float32)
                    np.divide(
                        self.vectors @ vector,
                        denominators,
                        out=similarities,
                        where=denominators > 0,
                    )
                    distances = np.maximum(0.0, 1.0 - np.clip(similarities, -1.0, 1.0))
                    indices = np.lexsort((np.arange(len(self.vectors)), distances))[:k]
                    return indices.reshape(1, -1), distances[indices].reshape(1, -1)

            exact_index = ExactCosineIndex(raw_embeddings)
            legacy_indices, legacy_distances = legacy_app.find_similar_to_vector(
                raw_embeddings, query_vector, k=len(raw_embeddings), nn_index=exact_index
            )
            api_response = page.request.post(
                base_url + f"/api/v1/runs/{run_id}/text-search",
                data={"model_id": CHINESE_CLIP_MODEL_ID, "query": query, "k": 4},
            )
            assert api_response.status == 200
            api_payload = api_response.json()
            assert api_payload["query"] == query
            assert [row["index"] for row in api_payload["results"]] == legacy_indices
            np.testing.assert_allclose(
                [row["distance"] for row in api_payload["results"]],
                legacy_distances,
                rtol=1e-6,
                atol=1e-7,
            )

            page.get_by_label("Chinese-CLIP text query").fill(query)
            page.get_by_role("button", name="Search images").click()
            text_rows = page.get_by_test_id("text-search-result")
            expect(text_rows).to_have_count(4)
            visible_text_indices = [int(row.get_attribute("data-index")) for row in text_rows.all()]
            assert visible_text_indices == legacy_indices
            visible_text_distances = [
                float(row.locator("small").inner_text().split("d=")[-1])
                for row in text_rows.all()
            ]
            np.testing.assert_allclose(
                visible_text_distances, legacy_distances, atol=0.00005
            )

            records = [
                {**record, "path": path}
                for record, path in zip(run["records"], run["image_paths"])
            ]
            with page.expect_download() as text_download_info:
                page.get_by_test_id("text-search-export").click()
            text_download = text_download_info.value
            assert Path(text_download.path()).read_bytes() == legacy_app.records_to_csv(
                records, legacy_indices
            ).encode("utf-8")

            text_rows.first.locator("button[aria-label^='Use ']").click()
            similar_rows = page.get_by_test_id("similar-result")
            expect(similar_rows).to_have_count(3)
            visual_query_index = legacy_indices[0]
            similar_indices, similar_distances = legacy_app.find_similar_indices(
                raw_embeddings,
                visual_query_index,
                k=3,
                nn_index=exact_index,
            )
            visible_similar_indices = [
                int(row.get_attribute("data-index")) for row in similar_rows.all()
            ]
            assert visible_similar_indices == similar_indices
            visible_similar_distances = [
                float(row.locator("span").inner_text().split("d=")[-1])
                for row in similar_rows.all()
            ]
            np.testing.assert_allclose(
                visible_similar_distances, similar_distances, atol=0.00005
            )
            browser.close()
            browser = None
    finally:
        if browser is not None:
            browser.close()
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*.png")
    }, "real Chinese-CLIP search changed its source images"
    assert _sha256_file(weights_path) == weights_hash == CHINESE_CLIP_PYTORCH_SHA256


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

            with page.expect_response(lambda response: "/coverage/runs/" in response.url
                                      and response.url.endswith("/projections")
                                      and response.request.method == "POST" and response.ok) as lda_fallback_response:
                page.get_by_label("Projection method").select_option("lda")
            lda_fallback = lda_fallback_response.value.json()
            assert lda_fallback["requested_method"] == "lda"
            assert lda_fallback["method"] == "pca"
            assert any("at least 5 images for a 3d layout" in warning.lower()
                       for warning in lda_fallback["warnings"])
            expect(page.get_by_role("status").filter(
                has_text="lda needs at least 5 images for a 3d layout",
            )).to_be_visible()
            expect(page.get_by_test_id("coverage-point")).to_have_count(4)
            page.get_by_test_id("coverage-point").nth(0).focus()
            page.get_by_test_id("coverage-point").nth(0).press("Enter")
            expect(page.get_by_test_id("coverage-selected-count")).to_have_text("3 selected")
            with page.expect_response(lambda response: "/coverage/runs/" in response.url
                                      and response.url.endswith("/projections")
                                      and response.request.method == "POST" and response.ok):
                page.get_by_test_id("coverage-dimension-2d").click()
            expect(page.get_by_test_id("coverage-selected-count")).to_have_text("3 selected")

            with page.expect_response(lambda response: "/coverage/runs/" in response.url
                                      and response.url.endswith("/projections")
                                      and response.request.method == "POST" and response.ok):
                page.get_by_label("Projection method").select_option("umap")
            with page.expect_response(lambda response: "/coverage/runs/" in response.url
                                      and response.url.endswith("/projections")
                                      and response.request.method == "POST" and response.ok):
                page.get_by_test_id("coverage-dimension-3d").click()
            expect(page.locator(".coverage-warning[role='status']")
                   .filter(has_text="at least 5 images for a 3D layout")).to_be_visible()
            expect(page.locator("[data-testid='coverage-plot'] [data-testid='coverage-plot-3d']")).to_be_visible()
            expect(page.get_by_test_id("coverage-point")).to_have_count(4)
            with page.expect_response(lambda response: "/coverage/runs/" in response.url
                                      and response.url.endswith("/projections")
                                      and response.request.method == "POST" and response.ok):
                page.get_by_test_id("coverage-dimension-2d").click()
            expect(page.get_by_test_id("coverage-point")).to_have_count(4)

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


def test_lv_web_03_img_01_real_model_visible_streamlit_sparsity_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The visible React coverage run preserves legacy PCA and high-dimensional sparsity values."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    from interaction import compute_outlier_scores

    models_dir = _pinned_dinov2_models_dir(tmp_path)
    dataset = _fixture(tmp_path / "coverage-real-model")
    folders = [dataset / "train", dataset / "test"]
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in folders
        for path in folder.rglob("*")
        if path.is_file()
    }
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "coverage-real-cache"))
    application = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=lambda model_id: load_model(model_id, models_dir),
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
            for folder in folders:
                page.get_by_label("Coverage image folder").fill(str(folder))
                page.get_by_role("button", name="Add coverage folder").click()
            expect(page.get_by_label("Feature model")).to_have_value("dinov2_vits14")
            page.get_by_label("Projection method").select_option("pca")
            page.get_by_role("button", name="Run coverage analysis").click()
            expect(page.get_by_test_id("coverage-job-status")).to_contain_text("Completed", timeout=300_000)
            expect(page.get_by_test_id("coverage-point")).to_have_count(4)
            expect(page.get_by_test_id("coverage-image-count")).to_have_text("4")
            expect(page.get_by_test_id("coverage-class-legend-item")).to_have_count(2)

            with application.state.lock:
                assert len(application.state.coverage_runs) == 1
                run_id, stored_run = next(iter(application.state.coverage_runs.items()))
                run = dict(stored_run)
                vectors = application.state.coverage_embeddings[run_id].copy()
            assert vectors.shape == (4, 384)
            assert run["embedding_shape"] == [4, 384]
            assert {record["label"] for record in run["records"]} == {"apple", "banana"}
            assert {record["split"] for record in run["records"]} == {"train", "test"}

            # Use the actual Streamlit projection helper as the numerical oracle.
            # Reset its session cache so this comparison is bound to this model run.
            import app as legacy_app

            monkeypatch.setattr(legacy_app.st, "session_state", {})
            legacy_2d, _ = legacy_app._cov_projection(
                vectors, None, "pca", 2, f"img01-real-{run_id}", "",
            )
            legacy_3d, _ = legacy_app._cov_projection(
                vectors, None, "pca", 3, f"img01-real-{run_id}", "",
            )
            legacy_tsne_2d, _ = legacy_app._cov_projection(
                vectors, None, "tsne", 2, f"img01-real-{run_id}", "",
            )
            legacy_tsne_3d, _ = legacy_app._cov_projection(
                vectors, None, "tsne", 3, f"img01-real-{run_id}", "",
            )
            react_2d = np.asarray(
                [[point["x"], point["y"]] for point in run["points"]],
            )
            np.testing.assert_allclose(react_2d, legacy_2d, rtol=1e-5, atol=1e-5)

            with page.expect_response(
                lambda response: response.url.endswith("/projections")
                and response.request.method == "POST" and response.ok,
            ) as projection_3d_response:
                page.get_by_test_id("coverage-dimension-3d").click()
            projection_3d = projection_3d_response.value.json()
            assert projection_3d["dimension"] == 3
            react_3d = np.asarray([
                [point["x"], point["y"], point["z"]]
                for point in projection_3d["points"]
            ])
            np.testing.assert_allclose(react_3d, legacy_3d, rtol=1e-5, atol=1e-5)
            expect(page.get_by_test_id("coverage-dimension-3d")).to_have_attribute(
                "aria-pressed", "true",
            )
            expect(page.locator(
                "[data-testid='coverage-plot'] [data-testid='coverage-plot-3d']",
            )).to_be_visible()
            expect(page.get_by_test_id("coverage-point")).to_have_count(4)
            with page.expect_response(
                lambda response: response.url.endswith("/projections")
                and response.request.method == "POST" and response.ok,
            ) as projection_2d_response:
                page.get_by_test_id("coverage-dimension-2d").click()
            projection_2d = projection_2d_response.value.json()
            react_2d_switched = np.asarray([
                [point["x"], point["y"]]
                for point in projection_2d["points"]
            ])
            np.testing.assert_allclose(react_2d_switched, legacy_2d, rtol=1e-5, atol=1e-5)
            expect(page.get_by_test_id("coverage-dimension-2d")).to_have_attribute(
                "aria-pressed", "true",
            )
            expect(page.get_by_test_id("coverage-point")).to_have_count(4)

            with page.expect_response(
                lambda response: response.url.endswith("/projections")
                and response.request.method == "POST" and response.ok,
            ) as tsne_2d_response:
                page.get_by_label("Projection method").select_option("tsne")
            tsne_2d = tsne_2d_response.value.json()
            assert tsne_2d["method"] == "tsne"
            react_tsne_2d = np.asarray([
                [point["x"], point["y"]]
                for point in tsne_2d["points"]
            ])
            np.testing.assert_allclose(react_tsne_2d, legacy_tsne_2d, rtol=1e-5, atol=1e-5)
            expect(page.get_by_test_id("coverage-point")).to_have_count(4)
            with page.expect_response(
                lambda response: response.url.endswith("/projections")
                and response.request.method == "POST" and response.ok,
            ) as tsne_3d_response:
                page.get_by_test_id("coverage-dimension-3d").click()
            tsne_3d = tsne_3d_response.value.json()
            react_tsne_3d = np.asarray([
                [point["x"], point["y"], point["z"]]
                for point in tsne_3d["points"]
            ])
            np.testing.assert_allclose(react_tsne_3d, legacy_tsne_3d, rtol=1e-5, atol=1e-5)
            expect(page.locator(
                "[data-testid='coverage-plot'] [data-testid='coverage-plot-3d']",
            )).to_be_visible()
            with page.expect_response(
                lambda response: response.url.endswith("/projections")
                and response.request.method == "POST" and response.ok,
            ):
                page.get_by_test_id("coverage-dimension-2d").click()
            with page.expect_response(
                lambda response: response.url.endswith("/projections")
                and response.request.method == "POST" and response.ok,
            ) as restored_pca_response:
                page.get_by_label("Projection method").select_option("pca")
            restored_pca = restored_pca_response.value.json()
            restored_pca_2d = np.asarray([
                [point["x"], point["y"]]
                for point in restored_pca["points"]
            ])
            np.testing.assert_allclose(restored_pca_2d, legacy_2d, rtol=1e-5, atol=1e-5)
            expect(page.get_by_test_id("coverage-point")).to_have_count(4)

            # Streamlit's _cov_sparsity delegates to this legacy distance helper.
            expected_default = compute_outlier_scores(
                vectors, vectors, k=run["k"], candidates_in_reference=True,
            )
            np.testing.assert_allclose(run["sparsity"], expected_default, rtol=1e-7, atol=1e-8)
            public_run_response = page.request.get(f"{base_url}/api/v1/coverage/runs/{run_id}")
            assert public_run_response.status == 200
            public_run = public_run_response.json()
            np.testing.assert_allclose(public_run["sparsity"], expected_default, rtol=1e-7, atol=1e-8)
            assert "image_paths" not in public_run

            page.locator("input[name='coverage-color-mode'][value='sparsity']").check()
            expect(page.locator(".coverage-legend-gradient")).to_be_visible()
            with page.expect_response(lambda response: response.url.endswith("/sparsity")
                                      and response.request.method == "POST" and response.ok) as sparsity_response:
                page.get_by_test_id("coverage-k").fill("2")
            changed = sparsity_response.value.json()
            expected_k2 = compute_outlier_scores(
                vectors, vectors, k=2, candidates_in_reference=True,
            )
            assert changed["k"] == 2
            np.testing.assert_allclose(changed["scores"], expected_k2, rtol=1e-7, atol=1e-8)
            expect(page.get_by_test_id("coverage-k-summary")).to_have_text("2")

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
            selected = page.get_by_test_id("coverage-selected-image").first
            expect(selected).to_be_visible()
            thumbnail_src = selected.locator("img").get_attribute("src")
            assert thumbnail_src and page.request.get(base_url + thumbnail_src).status == 200
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_hashes
    }, "real-model coverage changed the source dataset"


def test_lv_web_03_img_01_real_model_projection_visible_streamlit_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Real-model React UMAP and supervised projections match Streamlit in 2D and 3D."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")
    pytest.importorskip("umap")

    import app as legacy_app

    models_dir = _pinned_dinov2_models_dir(tmp_path)
    dataset = _fixture(tmp_path / "coverage-real-umap")
    Image.new("RGB", (48, 32), (40, 70, 180)).save(dataset / "train" / "apple" / "image-05.png")
    folders = [dataset / "train", dataset / "test"]
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in folders
        for path in folder.rglob("*")
        if path.is_file()
    }
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "coverage-real-umap-cache"))
    application = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=lambda model_id: load_model(model_id, models_dir),
    )
    server = server_thread = browser = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.set_default_timeout(120_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_role("button", name="Coverage", exact=True).click()
            for folder in folders:
                page.get_by_label("Coverage image folder").fill(str(folder))
                page.get_by_role("button", name="Add coverage folder").click()
            expect(page.get_by_label("Feature model")).to_have_value("dinov2_vits14")
            page.get_by_label("Projection method").select_option("umap")
            page.get_by_role("button", name="Run coverage analysis").click()
            expect(page.get_by_test_id("coverage-job-status")).to_contain_text(
                "Completed", timeout=300_000,
            )
            expect(page.get_by_test_id("coverage-point")).to_have_count(5)

            with application.state.lock:
                assert len(application.state.coverage_runs) == 1
                run_id, stored_run = next(iter(application.state.coverage_runs.items()))
                run = dict(stored_run)
                vectors = application.state.coverage_embeddings[run_id].copy()
            assert vectors.shape == (5, 384)
            assert run["method"] == run["requested_method"] == "umap"
            assert run["dimension"] == 2
            labels = [record["label"] for record in run["records"]]

            monkeypatch.setattr(legacy_app.st, "session_state", {})
            legacy_2d, _ = legacy_app._cov_projection(
                vectors, None, "umap", 2, f"img01-real-umap-{run_id}", "",
            )
            legacy_3d, _ = legacy_app._cov_projection(
                vectors, None, "umap", 3, f"img01-real-umap-{run_id}", "",
            )
            react_2d = np.asarray([[point["x"], point["y"]] for point in run["points"]])
            np.testing.assert_allclose(react_2d, legacy_2d, rtol=1e-5, atol=1e-5)

            with page.expect_response(
                lambda response: response.url.endswith("/projections")
                and response.request.method == "POST" and response.ok,
            ) as projection_3d_response:
                page.get_by_test_id("coverage-dimension-3d").click()
            projection_3d = projection_3d_response.value.json()
            assert projection_3d["method"] == "umap"
            react_3d = np.asarray([
                [point["x"], point["y"], point["z"]]
                for point in projection_3d["points"]
            ])
            np.testing.assert_allclose(react_3d, legacy_3d, rtol=1e-5, atol=1e-5)
            expect(page.locator(
                "[data-testid='coverage-plot'] [data-testid='coverage-plot-3d']",
            )).to_be_visible()
            expect(page.get_by_test_id("coverage-point")).to_have_count(5)

            with page.expect_response(
                lambda response: response.url.endswith("/projections")
                and response.request.method == "POST" and response.ok,
            ) as projection_2d_response:
                page.get_by_test_id("coverage-dimension-2d").click()
            projection_2d = projection_2d_response.value.json()
            react_2d_switched = np.asarray([
                [point["x"], point["y"]]
                for point in projection_2d["points"]
            ])
            np.testing.assert_allclose(react_2d_switched, legacy_2d, rtol=1e-5, atol=1e-5)
            expect(page.get_by_test_id("coverage-dimension-2d")).to_have_attribute(
                "aria-pressed", "true",
            )
            expect(page.get_by_test_id("coverage-point")).to_have_count(5)

            def assert_supervised_parity(response, method: str, dimension: int) -> None:
                projection = response.json()
                assert projection["requested_method"] == method
                assert projection["method"] == method
                monkeypatch.setattr(legacy_app.st, "session_state", {})
                expected, _ = legacy_app._cov_project5(
                    vectors, None, labels, None, method, dimension,
                    f"img01-real-{method}-{dimension}-{run_id}", "",
                )
                actual = np.asarray([
                    [point[axis] for axis in ("x", "y", "z")[:dimension]]
                    for point in projection["points"]
                ])
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
                expect(page.get_by_test_id("coverage-point")).to_have_count(5)

            with page.expect_response(
                lambda response: response.url.endswith("/projections")
                and response.request.method == "POST" and response.ok,
            ) as lda_2d_response:
                page.get_by_label("Projection method").select_option("lda")
            assert_supervised_parity(lda_2d_response.value, "lda", 2)

            with page.expect_response(
                lambda response: response.url.endswith("/projections")
                and response.request.method == "POST" and response.ok,
            ) as lda_3d_response:
                page.get_by_test_id("coverage-dimension-3d").click()
            assert_supervised_parity(lda_3d_response.value, "lda", 3)

            with page.expect_response(
                lambda response: response.url.endswith("/projections")
                and response.request.method == "POST" and response.ok,
            ) as sumap_3d_response:
                page.get_by_label("Projection method").select_option("sumap")
            assert_supervised_parity(sumap_3d_response.value, "sumap", 3)

            with page.expect_response(
                lambda response: response.url.endswith("/projections")
                and response.request.method == "POST" and response.ok,
            ) as sumap_2d_response:
                page.get_by_test_id("coverage-dimension-2d").click()
            assert_supervised_parity(sumap_2d_response.value, "sumap", 2)
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_hashes
    }, "real-model coverage projections changed the source dataset"


def test_lv_web_03_attr_01_visible_attribute_grid_calibration_and_candidate_mining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The React attribute grid exposes priors, missing cells, samples, and candidate export."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    dataset = _fixture(tmp_path / "attribute-fixture")
    folders = [dataset / "train", dataset / "test"]
    pool = tmp_path / "cell-candidate-pool"
    pool.mkdir()
    for index, color in enumerate(((200, 30, 40), (20, 190, 55), (35, 60, 220), (190, 160, 25))):
        Image.new("RGB", (48, 32), color).save(pool / f"candidate-{index}.png")
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in [*folders, pool]
        for path in folder.rglob("*")
        if path.is_file()
    }
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))

    def fixture_loader(_model_id: str):
        def embed(path: Path) -> np.ndarray:
            values = {
                "image-01.png": [1.0, 0.0, 0.0],
                "image-02.png": [0.9, 0.1, 0.0],
                "image-03.png": [0.0, 1.0, 0.0],
                "image-04.png": [0.0, 0.0, 1.0],
            }
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = values.get(
                path.name,
                np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) / 255.0 + 0.01,
            )
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
            page.set_default_timeout(15_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_role("button", name="Coverage", exact=True).click()
            for folder in folders:
                page.get_by_label("Coverage image folder").fill(str(folder))
                page.get_by_role("button", name="Add coverage folder").click()
            page.get_by_label("Projection method").select_option("pca")
            page.get_by_role("button", name="Run coverage analysis").click()
            expect(page.get_by_test_id("coverage-job-status")).to_contain_text("Completed", timeout=60_000)
            page.get_by_test_id("coverage-view-attributes").click()
            expect(page.get_by_role("heading", name="Attribute completeness grid")).to_be_visible()
            expect(page.get_by_test_id("completeness-health")).to_be_visible()
            expect(page.get_by_test_id("completeness-uncalibrated")).to_be_visible()
            expect(page.get_by_test_id("coverage-completeness-cell")).to_have_count(6)
            expect(page.get_by_test_id("coverage-completeness-gap")).not_to_have_count(0)

            populated = page.locator('[data-testid="coverage-completeness-cell"]:not([data-count="0"])').first
            populated.click()
            expect(page.get_by_test_id("coverage-completeness-sample")).not_to_have_count(0)
            sample_image = page.get_by_test_id("coverage-completeness-sample").first.locator("img")
            expect(sample_image).to_have_js_property("complete", True)
            assert sample_image.evaluate("node => node.naturalWidth") > 0

            page.get_by_test_id("completeness-frequency-0-0").select_option("低")
            with page.expect_response(lambda response: response.url.endswith("/completeness")
                                      and response.request.method == "POST" and response.ok):
                page.get_by_test_id("completeness-apply-calibration").click()
            expect(page.get_by_test_id("completeness-uncalibrated")).to_have_count(0)
            cell_00 = page.locator('[data-testid="coverage-completeness-cell"][data-index="0:0"]')
            expect(cell_00).to_contain_text("t=4.0")
            page.locator('[data-testid="coverage-completeness-cell"][data-count="0"]').first.click()
            expect(page.get_by_text("This cell has no images.", exact=True)).to_be_visible()

            page.get_by_test_id("completeness-candidate-folder-path").fill(str(pool))
            page.get_by_test_id("completeness-add-candidate-folder").click()
            with page.expect_response(lambda response: response.url.endswith("/completeness/candidates")
                                      and response.request.method == "POST" and response.ok):
                page.get_by_test_id("completeness-mine-candidates").click()
            expect(page.get_by_test_id("completeness-candidate-count")).to_contain_text("candidates")
            expect(page.get_by_test_id("completeness-candidate")).to_have_count(4)
            candidate_image = page.get_by_test_id("completeness-candidate").first.locator("img")
            expect(candidate_image).to_have_js_property("complete", True)
            assert candidate_image.evaluate("node => node.naturalWidth") > 0
            with page.expect_download() as download_info:
                page.get_by_test_id("completeness-export-candidates").click()
            assert download_info.value.suggested_filename == "cell_candidates.csv"
            candidate_src = candidate_image.get_attribute("src")
            assert candidate_src and "/coverage/runs/" in candidate_src
            run_id = candidate_src.split("/coverage/runs/", 1)[1].split("/", 1)[0]
            csv_response = page.request.get(base_url + f"/api/v1/coverage/runs/{run_id}/completeness/candidates.csv")
            assert csv_response.status == 200
            assert csv_response.text().lstrip("\ufeff").splitlines()[0] == "path,distance"
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


def test_lv_web_03_obj_01_crop_policy_controls_match_manual_and_autotune_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Coverage policy controls reach the worker and autotune updates the next run."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    folders, source_hashes = _object_coverage_fixture(tmp_path / "object-policy-controls")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "object-policy-controls-cache"))
    grant_ids = ["hostgrant_coverage_train", "hostgrant_coverage_test"]
    granted_paths = dict(zip(grant_ids, folders, strict=True))
    nativeapp_calls = []
    monkeypatch.setenv("NATIVEAPP_APP_ID", "visuallatent")
    monkeypatch.setenv("NATIVEAPP_API", "http://127.0.0.1:8765")

    def resolve_folder_grant(request, timeout):
        payload = json.loads(request.data)
        nativeapp_calls.append(payload)
        assert request.full_url == "http://127.0.0.1:8765/host/grants/resolve"
        assert request.get_method() == "POST"
        assert request.get_header("Origin") is None
        assert timeout == 10
        assert payload["app_id"] == "visuallatent"
        path = granted_paths[payload["grant_id"]]
        return BytesIO(json.dumps({
            "app_id": "visuallatent",
            "grant_id": payload["grant_id"],
            "path": str(path),
            "display_name": path.name,
        }).encode("utf-8"))

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
        monkeypatch.setattr(urllib.request, "urlopen", resolve_folder_grant)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.set_default_timeout(20_000)
            coverage_requests = []
            autotune_requests = []
            page.on(
                "request",
                lambda request: coverage_requests.append(request.post_data_json)
                if request.url.endswith("/api/v1/coverage/jobs") and request.method == "POST"
                else None,
            )
            page.on(
                "request",
                lambda request: autotune_requests.append(request.post_data_json)
                if request.url.endswith("/api/v1/object-policy/autotune/jobs") and request.method == "POST"
                else None,
            )
            page.add_init_script(
                "const folderPicks = " + json.dumps([
                    {"path": str(path), "grant_id": grant_id, "display_name": path.name}
                    for path, grant_id in zip(folders, grant_ids, strict=True)
                ]) + "; window.cimHost = { chooseFile: async () => folderPicks.shift() };"
            )
            page.goto(base_url, wait_until="networkidle")
            page.get_by_role("button", name="Coverage", exact=True).click()
            page.get_by_label("Analysis unit").select_option("object")
            expect(page.get_by_test_id("coverage-object-policy")).to_be_visible()
            for folder in folders:
                page.locator(".host-picker").first.click()
                expect(page.get_by_role("list", name="Coverage folders")).to_contain_text(folder.name)

            page.get_by_test_id("coverage-object-pad").fill("0.22")
            page.get_by_test_id("coverage-object-resolution").select_option("280")
            page.get_by_test_id("coverage-object-head").select_option("meanpool")
            page.get_by_test_id("coverage-object-policy-apply").click()
            expect(page.get_by_test_id("coverage-object-policy-active")).to_contain_text(
                "Active policy: pad 0.22 · 280px · meanpool."
            )

            page.get_by_role("button", name="Run coverage analysis").click()
            expect(page.get_by_test_id("coverage-job-status")).to_contain_text(
                "Completed", timeout=60_000,
            )
            manual_policy = {"pad": 0.22, "target_res": 280, "head": "meanpool"}
            assert coverage_requests[-1]["object_policy"] == manual_policy
            assert coverage_requests[-1]["folder_grant_ids"] == grant_ids
            assert len(nativeapp_calls) == 2
            with application.state.lock:
                first_run = next(iter(application.state.coverage_runs.values()))
                assert first_run["object_policy"] == manual_policy

            page.get_by_test_id("coverage-object-policy-autotune").click()
            expect(page.get_by_text(
                "Applied the best object crop policy. Run coverage analysis to use it."
            )).to_be_visible(timeout=120_000)
            assert len(autotune_requests) == 1
            assert autotune_requests[-1]["folder_grant_ids"] == grant_ids
            assert len(nativeapp_calls) == 4
            active_policy_text = page.get_by_test_id("coverage-object-policy-active").inner_text()
            import re

            matched = re.search(
                r"Active policy: pad ([0-9.]+) · ([0-9]+)px · (cls|meanpool)",
                active_policy_text,
            )
            assert matched, active_policy_text
            best_policy = {
                "pad": float(matched.group(1)),
                "target_res": int(matched.group(2)),
                "head": matched.group(3),
            }

            page.get_by_role("button", name="Run coverage analysis").click()
            expect(page.get_by_test_id("coverage-job-status")).to_contain_text(
                "Completed", timeout=60_000,
            )
            assert coverage_requests[-1]["object_policy"] == best_policy
            with application.state.lock:
                latest_run = list(application.state.coverage_runs.values())[-1]
                assert latest_run["object_policy"] == best_policy
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


def test_lv_web_03_obj_01_real_dinov2_visible_streamlit_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Real DINO object crops, projection, sparsity, and filtering match Streamlit."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    import app as legacy_app
    from visualize_embeddings import discover_images

    models_dir = _pinned_dinov2_models_dir(tmp_path)
    folders, source_hashes = _object_coverage_fixture(tmp_path / "object-coverage-real")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "object-coverage-real-cache"))
    weights = models_dir / "dinov2_vits14" / "dinov2_vits14.pth"
    weights_hash = _sha256_file(weights)
    application = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=lambda model_id: load_model(model_id, models_dir=models_dir),
        object_model_loader=lambda model_id, **kwargs: load_model(
            model_id, models_dir=models_dir, **kwargs,
        ),
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.set_default_timeout(20_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_role("button", name="Coverage", exact=True).click()
            page.get_by_label("Analysis unit").select_option("object")
            page.get_by_label("Projection method").select_option("pca")
            for folder in folders:
                page.get_by_label("Coverage image folder").fill(str(folder))
                page.get_by_role("button", name="Add coverage folder").click()
            expect(page.get_by_label("Feature model")).to_have_value("dinov2_vits14")
            page.get_by_role("button", name="Run coverage analysis").click()
            expect(page.get_by_test_id("coverage-job-status")).to_contain_text(
                "Completed", timeout=300_000,
            )
            expect(page.get_by_test_id("coverage-point")).to_have_count(8)
            expect(page.get_by_test_id("coverage-image-count")).to_have_text("8")

            with application.state.lock:
                assert len(application.state.coverage_runs) == 1
                run_id, stored_run = next(iter(application.state.coverage_runs.items()))
                run = dict(stored_run)
                react_vectors = application.state.coverage_embeddings[run_id].copy()
            assert run["unit"] == "object"
            assert run["object_policy"] == {"pad": 0.12, "target_res": 224, "head": "cls"}
            assert react_vectors.shape == (8, 384)
            assert np.isfinite(react_vectors).all()
            assert np.all(np.linalg.norm(react_vectors, axis=1) > 0)

            class_names = ["alpha", "beta"]
            legacy_source_records = discover_images(folders, class_names)
            monkeypatch.setattr(
                legacy_app.st, "session_state", {},
            )
            monkeypatch.setattr(
                legacy_app,
                "load_model",
                lambda model_id, **kwargs: load_model(
                    model_id, models_dir=models_dir, **kwargs,
                ),
            )
            legacy_records, legacy_vectors, _legacy_token = legacy_app._crop_and_embed_objects(
                legacy_source_records,
                "dinov2_vits14",
                class_names,
                0.12,
                base_token="react-web-real-object-coverage",
            )
            assert len(legacy_records) == len(run["records"]) == 8
            np.testing.assert_allclose(react_vectors, legacy_vectors, rtol=1e-6, atol=1e-7)
            for expected, actual in zip(legacy_records, run["records"], strict=True):
                assert actual["filename"] == expected["image_path"].name
                assert actual["label"] == expected["label"]
                assert actual["class_id"] == expected["class_id"]
                assert actual["obj_index"] == expected["obj_index"]
                assert actual["score"] == pytest.approx(expected["score"], abs=1e-7)
                assert actual["short_px"] == expected["short_px"]
                np.testing.assert_allclose(actual["bbox"], expected["bbox"], rtol=0, atol=1e-7)

            monkeypatch.setattr(legacy_app.st, "session_state", {})
            legacy_2d, _ = legacy_app._cov_projection(
                legacy_vectors, None, "pca", 2, f"objcov-real-{run_id}", "",
            )
            react_2d = np.asarray([[point["x"], point["y"]] for point in run["points"]])
            np.testing.assert_allclose(react_2d, legacy_2d, rtol=1e-5, atol=1e-5)
            from interaction import sparsity_scores

            expected_sparsity = sparsity_scores(legacy_vectors, k=run["k"])
            np.testing.assert_allclose(run["sparsity"], expected_sparsity, rtol=1e-7, atol=1e-8)
            public_run = page.request.get(
                f"{base_url}/api/v1/coverage/runs/{run_id}",
            ).json()
            assert "image_paths" not in public_run
            assert "image_path" not in str(public_run)

            page.get_by_test_id("coverage-object-class").select_option("alpha")
            page.get_by_test_id("coverage-min-short-px").fill("10")
            page.get_by_test_id("coverage-confidence-min").fill("0.8")
            page.get_by_test_id("coverage-confidence-max").fill("0.95")
            with page.expect_response(
                lambda response: response.url.endswith("/object-view")
                and response.request.method == "POST" and response.ok,
            ) as object_view_response:
                page.get_by_test_id("coverage-apply-object-filters").click()
            object_view = object_view_response.value.json()
            expect(page.get_by_test_id("coverage-point")).to_have_count(4)
            expect(page.get_by_test_id("coverage-image-count")).to_have_text("4")

            monkeypatch.setattr(
                legacy_app.st,
                "session_state",
                {"obj_conf_range": (0.8, 0.95), "obj_min_short_px": 10},
            )
            legacy_indices = [
                index for index in legacy_app._obj_view_indices(legacy_records)
                if legacy_records[index]["label"] == "alpha"
            ]
            assert [point["index"] for point in object_view["points"]] == legacy_indices
            assert [point["filename"] for point in object_view["points"]] == [
                legacy_records[index]["image_path"].name for index in legacy_indices
            ]
            for point, index in zip(object_view["points"], legacy_indices, strict=True):
                legacy_record = legacy_records[index]
                assert point["label"] == legacy_record["label"] == "alpha"
                assert point["class_id"] == legacy_record["class_id"]
                assert point["obj_index"] == legacy_record["obj_index"]
                assert point["short_px"] == legacy_record["short_px"]
                assert point["score"] == pytest.approx(legacy_record["score"], abs=1e-7)
                np.testing.assert_allclose(point["bbox"], legacy_record["bbox"], rtol=0, atol=1e-7)
            legacy_filtered = legacy_vectors[legacy_indices]
            legacy_filtered_labels = [legacy_records[index]["label"] for index in legacy_indices]
            expected_filtered_sparsity = sparsity_scores(
                legacy_filtered, k=object_view["k"],
            )
            np.testing.assert_allclose(
                object_view["sparsity"], expected_filtered_sparsity, rtol=1e-7, atol=1e-8,
            )
            monkeypatch.setattr(legacy_app.st, "session_state", {})
            expected_filtered_2d, _ = legacy_app._cov_projection(
                legacy_filtered,
                None,
                "pca",
                2,
                f"objcov-real-filtered-{run_id}",
                "",
            )
            react_filtered_2d = np.asarray([
                [point["x"], point["y"]] for point in object_view["points"]
            ])
            np.testing.assert_allclose(
                react_filtered_2d, expected_filtered_2d, rtol=1e-5, atol=1e-5,
            )
            assert legacy_filtered_labels == [point["label"] for point in object_view["points"]]
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in source_hashes
    }, "real-model object coverage changed source images, labels, or classes"
    assert _sha256_file(weights) == weights_hash


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
    # The visible journey exercises both a successful probe and the honest low-sample
    # path after readback changes labels. Keep four images per source class initially.
    for relative, color in (
        ("train/apple/image-03.png", (224, 36, 45)),
        ("train/apple/image-04.png", (210, 45, 52)),
        ("train/banana/image-04.png", (32, 72, 215)),
        ("train/banana/image-05.png", (40, 68, 205)),
    ):
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color).save(path)
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
        expect(page.get_by_test_id("embedding-point")).to_have_count(8)

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

        expect(page.get_by_test_id("curation-labeling-preview")).to_contain_text("2 sendable · 2 new · 0 already handled")
        page.get_by_test_id("curation-send-labeling").click()
        expect(page.locator(".alert-info")).to_contain_text("Saved 2 image(s) to the local Labeling inbox")
        expect(page.get_by_test_id("curation-labeling-preview")).to_contain_text("0 sendable · 0 new · 2 already handled")
        expect(page.get_by_test_id("curation-send-labeling")).to_be_disabled()
        page.get_by_label("Include previously handled samples").check()
        expect(page.get_by_test_id("curation-labeling-preview")).to_contain_text("2 sendable · 0 new · 2 already handled")
        page.get_by_test_id("curation-send-labeling").click()
        expect(page.locator(".alert-info")).to_contain_text("Saved 2 image(s) to the local Labeling inbox")

        rounds = page.request.get(base_url + "/api/v1/rounds").json()["rounds"]
        assert len(rounds) == 2
        assert [row["n_sent"] for row in rounds] == [2, 2]
        for round_row, value in zip(rounds, (0.75, 0.9), strict=True):
            page.get_by_label("Metric round").select_option(round_row["round_id"])
            page.get_by_label("Metric name").fill("review_accuracy")
            page.get_by_label("Metric value").fill(str(value))
            page.get_by_role("button", name="Add metric").click()
            expect(page.locator(".alert-info")).to_contain_text("Added review_accuracy to round")
        expect(page.get_by_test_id("active-learning-curve-chart")).to_be_visible()
        expect(page.get_by_test_id("active-learning-round")).to_have_count(2)
        probe_round_id = rounds[0]["round_id"]
        page.get_by_test_id(f"active-learning-probe-{probe_round_id}").click()
        expect(page.locator(".alert-info")).to_contain_text("Probe complete (fixture-model)")
        round_after_probe = next(
            row for row in page.request.get(base_url + "/api/v1/rounds").json()["rounds"]
            if row["round_id"] == probe_round_id
        )
        successful_probe_metrics = [
            metric for metric in round_after_probe["metrics"] if metric.get("source") == "probe"
        ]
        assert {metric["name"] for metric in successful_probe_metrics} == {
            "probe_acc", "probe_macro_f1",
        }
        probe_round_card = page.get_by_test_id("active-learning-round").filter(has_text=probe_round_id)
        expect(probe_round_card).to_contain_text("probe_acc=")
        handoff_root = cache_root / "labeling-inbox" / "lv_labeling_handoff"
        handoff_specs = list(handoff_root.glob("*/_handoff.json"))
        assert len(handoff_specs) == 1
        handoff_spec_path = handoff_specs[0]
        handoff_spec = json.loads(handoff_spec_path.read_text(encoding="utf-8"))
        assert len(handoff_spec["items"]) == 2
        for item in handoff_spec["items"]:
            sidecar = handoff_spec_path.parent / "images" / Path(item["image"]).with_suffix(".json").name
            sidecar.write_text(json.dumps({
                "shapes": [{"label": "reviewed-label"}], "annotator": "Fixture Annotator",
            }), encoding="utf-8")
        page.get_by_test_id("refresh-labeling-inbox").click()
        expect(page.get_by_test_id("labeling-handoff-row")).to_contain_text("2/2 labeled")
        handoff_id = handoff_spec["handoff_id"]
        page.get_by_test_id(f"labeling-readback-{handoff_id}").click()
        expect(page.locator(".alert-info")).to_contain_text("Applied 2 labeled image(s)")
        expect(page.locator(".alert-info")).to_contain_text("2 label change(s) · batch complete · annotator Fixture Annotator")
        expect(page.get_by_test_id("labeling-readback-audit")).to_contain_text("2 corrected image(s) · 2 audit event(s)")
        with page.expect_download() as audit_download:
            page.get_by_role("button", name="Export correction CSV").click()
        audit_csv = Path(audit_download.value.path()).read_text(encoding="utf-8-sig")
        assert audit_csv.count("reviewed-label") == 2
        assert "Fixture Annotator" in audit_csv
        assert page.request.get(base_url + "/api/v1/readback/audit").json()["summary"]["n_images"] == 2

        subset_dir = tmp_path / "curation-yolo-subset"
        page.get_by_label("Subset destination folder").fill(str(subset_dir))
        page.get_by_test_id("curation-subset-preview-button").click()
        expect(page.get_by_test_id("curation-subset-preview")).to_contain_text(
            "Preflight: 2 available · 2 unique · 0 duplicate(s) · 0 missing"
        )
        page.get_by_test_id("curation-subset-export-button").click()
        expect(page.locator(".alert-info")).to_contain_text("YOLO subset exported: 2 image(s)")
        assert (subset_dir / "manifest.csv").is_file()
        assert (subset_dir / "lineage.json").is_file()
        assert len(list((subset_dir / "images").rglob("*.png"))) == 2

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
        expect(page.get_by_test_id("active-learning-round")).to_have_count(2)
        assert len(page.request.get(restarted_url + "/api/v1/rounds").json()["rounds"]) == 2
        expect(page.get_by_test_id("labeling-readback-audit")).to_contain_text("2 corrected image(s)")
        page.get_by_role("button", name="Visualize").click()
        page.get_by_test_id("model-choice-fixture-model").check()
        page.get_by_label("Supervised UMAP").uncheck()
        page.get_by_label("PCA").check()
        page.get_by_label("Image folder").fill(str(dataset / "train"))
        page.get_by_role("button", name="Add folder").click()
        page.get_by_label("Image folder").fill(str(dataset / "test"))
        page.get_by_role("button", name="Add folder").click()
        page.get_by_role("button", name="Run analysis").click()
        expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
        reviewed_points = [
            point.get_attribute("aria-label") or ""
            for point in page.get_by_test_id("embedding-point").all()
        ]
        assert sum("reviewed-label" in label for label in reviewed_points) == 2
        page.get_by_test_id("open-curation").click()
        expect(page.get_by_test_id("active-learning-round")).to_have_count(2)
        insufficient_round_id = rounds[1]["round_id"]
        page.get_by_test_id(f"active-learning-probe-{insufficient_round_id}").click()
        expect(page.locator(".alert-error")).to_contain_text("insufficient_class:", timeout=20_000)
        round_after_warning = next(
            row for row in page.request.get(restarted_url + "/api/v1/rounds").json()["rounds"]
            if row["round_id"] == insufficient_round_id
        )
        assert not any(metric.get("source") == "probe" for metric in round_after_warning["metrics"])
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


@pytest.mark.parametrize("handoff_flow", ["curation-item", "duplicate-pair"])
def test_lv_web_10_send_readback_cross_repo_react_browser_journey(
    tmp_path: Path, handoff_flow: str,
):
    """Round-trip relabel and duplicate-pair decisions through both React apps."""
    annotation_root_value = os.environ.get("ANNOTATION_REPO_ROOT")
    if not annotation_root_value:
        pytest.skip("set ANNOTATION_REPO_ROOT to a built ANnoTation checkout for cross-repo E2E")
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.fail("Build LV React with npm run build before running this cross-repo E2E")

    annotation_root = Path(annotation_root_value).expanduser().resolve()
    if not (annotation_root / "web" / "dist" / "index.html").is_file():
        pytest.fail(
            "Build ANnoTation React with npm --prefix web run build before this E2E: "
            f"{annotation_root / 'web' / 'dist'}"
        )

    dataset = _fixture(tmp_path / "cross-repo-source")
    if handoff_flow == "duplicate-pair":
        from PIL.PngImagePlugin import PngInfo

        left_image = dataset / "train" / "apple" / "image-01.png"
        right_image = dataset / "test" / "banana" / "image-04.png"
        duplicate_info = PngInfo()
        duplicate_info.add_text("fixture_copy", "second file identity")
        with Image.open(left_image) as source:
            source.convert("RGB").save(right_image, pnginfo=duplicate_info)
        assert left_image.read_bytes() != right_image.read_bytes()
        assert np.array_equal(np.asarray(Image.open(left_image)), np.asarray(Image.open(right_image)))
    source_before = {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }
    cache_root = tmp_path / "cross-repo-lv-cache"
    annotation_data = tmp_path / "cross-repo-annotation-data"
    annotation_data.mkdir(parents=True)
    annotation_log = tmp_path / "annotation-service.log"
    annotation_port = _free_port()
    worker = None
    annotation_process = None
    browser = None
    playwright = None
    try:
        worker, lv_url = _start_curation_worker(cache_root)
        from playwright.sync_api import expect, sync_playwright

        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        lv_page = browser.new_page(viewport={"width": 1500, "height": 1100})
        annotation_page = browser.new_page(viewport={"width": 1440, "height": 900})
        lv_page.set_default_timeout(15_000)
        annotation_page.set_default_timeout(15_000)
        lv_page.goto(lv_url, wait_until="networkidle")
        lv_page.get_by_test_id("model-choice-fixture-model").check()
        lv_page.get_by_label("Supervised UMAP").uncheck()
        lv_page.get_by_label("PCA").check()
        for folder in (dataset / "train", dataset / "test"):
            lv_page.get_by_label("Image folder").fill(str(folder))
            lv_page.get_by_role("button", name="Add folder").click()
        lv_page.get_by_role("button", name="Run analysis").click()
        expect(lv_page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
        expect(lv_page.get_by_test_id("embedding-point")).to_have_count(4)
        if handoff_flow == "curation-item":
            lv_page.get_by_test_id("embedding-point").first.click()
            expect(lv_page.get_by_test_id("selection-count")).to_have_text("1 selected")
            lv_page.get_by_test_id("add-selection-to-curation").click()
            expect(lv_page.locator(".alert-info")).to_contain_text("Added 1 item(s) to curation")
            lv_page.get_by_test_id("open-curation").click()
            expect(lv_page.get_by_test_id("curation-item")).to_have_count(1)
            expect(lv_page.get_by_test_id("curation-labeling-preview")).to_contain_text(
                "1 sendable · 1 new · 0 already handled"
            )
            lv_page.get_by_test_id("curation-send-labeling").click()
            expect(lv_page.locator(".alert-info")).to_contain_text(
                "Saved 1 image(s) to the local Labeling inbox"
            )
        else:
            lv_page.get_by_label("Duplicate method").select_option("embedding")
            lv_page.get_by_label("Only cross-split pairs").check()
            lv_page.get_by_role("button", name="Scan candidates").click()
            expect(lv_page.get_by_test_id("duplicate-result")).to_have_count(1)
            assert lv_page.get_by_test_id("duplicate-result").first.get_attribute("data-pair") == "0-3"
            lv_page.get_by_test_id("duplicate-labeling-handoff").click()
            expect(lv_page.get_by_test_id("duplicate-handoff-status")).to_contain_text(
                "Saved 1 pair(s) with 2 unique image(s) to the Labeling inbox"
            )

        handoff_specs = list((cache_root / "labeling-inbox" / "lv_labeling_handoff").glob("*/_handoff.json"))
        assert len(handoff_specs) == 1
        handoff_spec_path = handoff_specs[0]
        handoff_spec = json.loads(handoff_spec_path.read_text(encoding="utf-8"))
        handoff_id = handoff_spec["handoff_id"]
        images_dir = handoff_spec_path.parent / "images"
        if handoff_flow == "curation-item":
            assert handoff_spec["task"] == "re-label"
            assert len(handoff_spec["items"]) == 1
            item = handoff_spec["items"][0]
            image_path = images_dir / Path(item["image"]).name
            assert image_path.is_file()
            assert hashlib.sha256(image_path.read_bytes()).hexdigest() == item["sha256"]
        else:
            assert handoff_spec["task"] == "verify"
            assert handoff_spec["source"] == "duplicate"
            assert handoff_spec["payload"]["kind"] == "duplicate_pair_review"
            assert len(handoff_spec["payload"]["pairs"]) == 1
            pair = handoff_spec["payload"]["pairs"][0]
            assert pair["left_filename"] == "image-01.png"
            assert pair["right_filename"] == "image-04.png"
            assert pair["left_sha256"] != pair["right_sha256"]
            assert len(handoff_spec["items"]) == 2
            for item in handoff_spec["items"]:
                image_path = images_dir / Path(item["image"]).name
                assert image_path.is_file()
                assert hashlib.sha256(image_path.read_bytes()).hexdigest() == item["sha256"]

        annotation_process, annotation_url = _start_annotation_react_server(
            annotation_root, annotation_data, annotation_port, annotation_log
        )
        annotation_page.goto(annotation_url, wait_until="networkidle")
        workflow = annotation_page.get_by_role("navigation", name="工作流程")
        workflow.get_by_role("button", name=re.compile("資料集")).click()
        annotation_page.get_by_label("資料夾路徑").fill(str(images_dir))
        annotation_page.get_by_label("資料集名稱").fill("LV React handoff browser journey")
        annotation_page.get_by_role("button", name=re.compile("建立資料集並開始標注")).click()
        expect(annotation_page.get_by_role("heading", name="標注工作台")).to_be_visible()
        if handoff_flow == "curation-item":
            expect(annotation_page.get_by_role("button", name=re.compile(re.escape(image_path.name)))).to_be_visible()
        else:
            pair_row = annotation_page.get_by_test_id("handoff-review-pair")
            expect(pair_row).to_contain_text("image-01.png")
            expect(pair_row).to_contain_text("image-04.png")
            expect(annotation_page.get_by_test_id("handoff-review-classification-guidance")).to_be_visible()

        workflow.get_by_role("button", name=re.compile("資料集")).click()
        annotation_page.get_by_role("button", name="標籤管理").click()
        if handoff_flow == "curation-item":
            annotation_page.get_by_label("框選標籤").fill("reviewed-label")
            annotation_page.get_by_label("整張圖片分類").fill("OK\nNG")
        else:
            annotation_page.get_by_label("框選標籤").fill("keep\ndrop")
            annotation_page.get_by_label("整張圖片分類").fill("keep\ndrop")
        annotation_page.get_by_label("外部標注工具").select_option("labelme")
        annotation_page.get_by_role("button", name=re.compile("保存工作區設定")).click()
        expect(annotation_page.get_by_text("工作區設定已保存")).to_be_visible()
        workflow.get_by_role("button", name="標注工作台").click()
        if handoff_flow == "curation-item":
            annotation_page.get_by_role("button", name=re.compile(re.escape(image_path.name))).click()
            canvas = annotation_page.get_by_label("圖片與標注框")
            expect(canvas).to_be_visible()
            bounds = canvas.bounding_box()
            assert bounds is not None
            annotation_page.mouse.move(bounds["x"] + bounds["width"] * 0.2, bounds["y"] + bounds["height"] * 0.2)
            annotation_page.mouse.down()
            annotation_page.mouse.move(bounds["x"] + bounds["width"] * 0.8, bounds["y"] + bounds["height"] * 0.8)
            annotation_page.mouse.up()
            expect(annotation_page.get_by_label("第 1 個框選類別")).to_have_value("reviewed-label")
            annotation_page.get_by_role("button", name=re.compile("保存標注")).click()
            expect(annotation_page.get_by_text("已同步")).to_be_visible()
            sidecar_path = image_path.with_suffix(".json")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            assert [shape["label"] for shape in sidecar["shapes"]] == ["reviewed-label"]
            assert hashlib.sha256(image_path.read_bytes()).hexdigest() == item["sha256"]
        else:
            pair_row = annotation_page.get_by_test_id("handoff-review-pair")
            pair_row.get_by_test_id("handoff-pair-left").click()
            with annotation_page.expect_response(lambda response: "/classification" in response.url):
                annotation_page.get_by_label("分類結果").select_option(label="keep")
            pair_row.get_by_test_id("handoff-pair-right").click()
            with annotation_page.expect_response(lambda response: "/classification" in response.url):
                annotation_page.get_by_label("分類結果").select_option(label="drop")
            with (handoff_spec_path.parent / "answers.csv").open(encoding="utf-8", newline="") as handle:
                answers = {row["sha256"]: row["label"] for row in csv.DictReader(handle)}
            assert answers == {
                pair["left_sha256"]: "keep", pair["right_sha256"]: "drop",
            }

        _stop_annotation_react_server(annotation_process)
        annotation_process, restarted_annotation_url = _start_annotation_react_server(
            annotation_root, annotation_data, annotation_port, annotation_log
        )
        annotation_page.goto(restarted_annotation_url, wait_until="networkidle")
        expect(annotation_page.get_by_role("heading", name="標注工作台")).to_be_visible()
        if handoff_flow == "curation-item":
            annotation_page.get_by_role("button", name=re.compile(re.escape(image_path.name))).click()
            expect(annotation_page.get_by_label("第 1 個框選類別")).to_have_value("reviewed-label")
            expect(annotation_page.get_by_text("已同步")).to_be_visible()
        else:
            pair_row = annotation_page.get_by_test_id("handoff-review-pair")
            pair_row.get_by_test_id("handoff-pair-left").click()
            expect(annotation_page.get_by_label("分類結果")).to_have_value("keep")
            pair_row.get_by_test_id("handoff-pair-right").click()
            expect(annotation_page.get_by_label("分類結果")).to_have_value("drop")

        _stop_worker(worker)
        worker = None
        worker, restarted_lv_url = _start_curation_worker(cache_root)
        lv_page.goto(restarted_lv_url, wait_until="networkidle")
        lv_page.get_by_test_id("model-choice-fixture-model").check()
        lv_page.get_by_label("Supervised UMAP").uncheck()
        lv_page.get_by_label("PCA").check()
        for folder in (dataset / "train", dataset / "test"):
            lv_page.get_by_label("Image folder").fill(str(folder))
            lv_page.get_by_role("button", name="Add folder").click()
        lv_page.get_by_role("button", name="Run analysis").click()
        expect(lv_page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
        lv_page.get_by_test_id("open-curation").click()
        lv_page.get_by_test_id("refresh-labeling-inbox").click()
        expected_labeled = 1 if handoff_flow == "curation-item" else 2
        expect(lv_page.get_by_test_id("labeling-handoff-row")).to_contain_text(
            f"{expected_labeled}/{expected_labeled} labeled"
        )
        lv_page.get_by_test_id(f"labeling-readback-{handoff_id}").click()
        expect(lv_page.locator(".alert-info")).to_contain_text(
            f"Applied {expected_labeled} labeled image(s)"
        )
        expect(lv_page.get_by_test_id("labeling-readback-audit")).to_contain_text(
            f"{expected_labeled} corrected image(s) · {expected_labeled} audit event(s)"
        )
        lv_page.get_by_role("button", name="Visualize").click()
        lv_page.get_by_role("button", name="Run analysis").click()
        expect(lv_page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=60_000)
        point_labels = [
            point.get_attribute("aria-label") or ""
            for point in lv_page.get_by_test_id("embedding-point").all()
        ]
        if handoff_flow == "curation-item":
            assert sum("reviewed-label" in label for label in point_labels) == 1
        else:
            assert any("image-01.png, keep, train" in label for label in point_labels)
            assert any("image-04.png, drop, test" in label for label in point_labels)
            ledger = [
                json.loads(line)
                for line in (cache_root / "output" / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            duplicate_readbacks = [
                event for event in ledger
                if event.get("kind") == "readback" and event.get("batch_id") == handoff_id
            ]
            assert len(duplicate_readbacks) == 1
            assert duplicate_readbacks[0]["n_labeled"] == 2
            assert duplicate_readbacks[0]["n_changed"] == 2
        assert source_before == {
            path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in dataset.rglob("*") if path.is_file()
        }
    finally:
        if browser is not None:
            browser.close()
        if playwright is not None:
            playwright.stop()
        if worker is not None:
            _stop_worker(worker)
        if annotation_process is not None:
            _stop_annotation_react_server(annotation_process)


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


def test_lv_web_01_edge_01_unreadable_only_folder_is_rejected_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A folder containing only corrupt images fails visibly without touching its source."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    image_folder = tmp_path / "資料 集" / "train" / "損壞影像"
    image_folder.mkdir(parents=True)
    unreadable_path = image_folder / "無法解碼.jpg"
    unreadable_path.write_bytes(b"not an image")
    source_hash = hashlib.sha256(unreadable_path.read_bytes()).hexdigest()
    embed_calls: list[Path] = []
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))

    def embed(path: Path) -> np.ndarray:
        embed_calls.append(path)
        return np.array([1.0] + [0.0] * 383, dtype=np.float32)

    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model_id: embed,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Image folder").fill(str(image_folder.parent))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()

            expect(page.get_by_role("alert")).to_contain_text(
                "No readable images were found", timeout=10_000
            )
            expect(page.locator(".completion-label")).to_have_count(0)
            expect(page.get_by_test_id("embedding-point")).to_have_count(0)
            assert embed_calls == []
            assert hashlib.sha256(unreadable_path.read_bytes()).hexdigest() == source_hash
            assert sorted(path.name for path in image_folder.iterdir()) == [unreadable_path.name]
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
    """React object curation previews source boxes and exercises grant-bound copy/symlink export."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")
    folders = _detector_fixture(tmp_path / "detector-object-fixture")
    selected_image = folders[0] / "images" / "02.png"
    selected_yolo = folders[0] / "labels" / "02.txt"
    selected_sidecar = selected_image.with_suffix(".json")
    labelme_annotation = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [{
            "label": "reviewed-defect",
            "points": [[3.0, 4.0], [12.0, 16.0]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        }],
        "imagePath": "02.png",
        "imageData": None,
        "imageHeight": 32,
        "imageWidth": 32,
    }
    selected_sidecar.write_text(json.dumps(labelme_annotation), encoding="utf-8")
    source_hashes = {
        path.relative_to(folders[0].parent.parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in folders
        for path in folder.rglob("*")
        if path.is_file()
    }
    source_image_bytes = selected_image.read_bytes()
    source_yolo_lines = selected_yolo.read_text(encoding="utf-8").splitlines()
    source_sidecar_bytes = selected_sidecar.read_bytes()
    subset_dir = tmp_path / "visible-object-subset"
    subset_dir.mkdir()
    symlink_dir = tmp_path / "visible-object-symlink-subset"
    symlink_dir.mkdir()
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("NATIVEAPP_APP_ID", raising=False)

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
            page.add_init_script(
                "const curationFolderChoices = "
                + json.dumps([
                    {"path": str(subset_dir), "grant_id": "hostgrant_subset"},
                    {"path": str(symlink_dir), "grant_id": "hostgrant_symlink"},
                ])
                + "; let curationFolderIndex = 0;"
                + " window.cimHost = {chooseFile: async () => curationFolderChoices["
                + "Math.min(curationFolderIndex++, curationFolderChoices.length - 1)]};"
            )
            run_payloads = []
            subset_requests = []
            subset_export_results = []
            page.on(
                "response",
                lambda response: run_payloads.append(response.json())
                if _is_run_summary_response(response) and response.ok
                else None,
            )
            page.on(
                "response",
                lambda response: subset_export_results.append(response.json())
                if response.url.endswith("/api/v1/curation/subset/export") and response.ok
                else None,
            )
            page.on(
                "request",
                lambda request: subset_requests.append(json.loads(request.post_data or "{}"))
                if request.url.endswith((
                    "/api/v1/curation/subset/preview",
                    "/api/v1/curation/subset/export",
                ))
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

            expect(page.get_by_test_id("selection-count")).to_have_text("2 selected")
            page.get_by_test_id("add-selection-to-curation").click()
            expect(page.locator(".alert-info")).to_contain_text(
                "Added 2 item(s) to curation; skipped 0 duplicate(s)."
            )
            curation = page.request.get(base_url + "/api/v1/curation/items").json()
            assert curation["total"] == 2
            assert [
                (item["filename"], item["label"], item["level"], item["obj_index"], item["bbox"])
                for item in curation["items"]
            ] == [
                ("02.png", "stain", "object", 1, run["records"][2]["bbox"]),
                ("03.png", "stain", "object", 0, run["records"][3]["bbox"]),
            ]
            assert all("path" not in item and "image_path" not in item for item in curation["items"])

            page.get_by_test_id("open-curation").click()
            expect(page.get_by_test_id("curation-item")).to_have_count(2)
            item_headings = page.locator(".curation-item-heading strong")
            expect(item_headings.nth(0)).to_have_text("02.png")
            expect(item_headings.nth(1)).to_have_text("03.png")
            item_metadata = page.locator(".curation-item-meta")
            expect(item_metadata.nth(0)).to_contain_text("stain · train · manual · object 2")
            expect(item_metadata.nth(1)).to_contain_text("stain · test · manual · object 1")

            import react_web_shell

            monkeypatch.setenv("NATIVEAPP_APP_ID", "visuallatent")

            grant_destinations = {
                "hostgrant_subset": subset_dir,
                "hostgrant_symlink": symlink_dir,
            }

            def resolve_subset_grant(grant_id: str, app_id: str) -> Path:
                assert grant_id in grant_destinations
                assert app_id == "visuallatent"
                return grant_destinations[grant_id].resolve()

            monkeypatch.setattr(
                react_web_shell,
                "_resolve_nativeapp_folder_grant",
                resolve_subset_grant,
            )
            page.get_by_test_id("curation-subset-choose-folder").click()
            expect(page.get_by_label("Subset destination folder")).to_have_value(str(subset_dir))
            page.get_by_test_id("curation-subset-preview-button").click()
            expect(page.get_by_test_id("curation-subset-preview")).to_contain_text(
                "Preflight: 2 available · 2 unique · 0 duplicate(s) · 0 missing"
            )
            expect(page.get_by_test_id("curation-subset-preview")).to_contain_text(
                "2 YOLO label file(s) · 2 object selection(s)"
            )
            page.get_by_test_id("curation-subset-export-button").click()
            expect(page.locator(".alert-info")).to_contain_text(
                "YOLO subset exported: 2 image(s)"
            )
            assert subset_export_results[-1]["exported"] == 2
            assert subset_export_results[-1]["errors"] == 0
            assert [request["destination_folder_grant_id"] for request in subset_requests] == [
                "hostgrant_subset", "hostgrant_subset",
            ]

            exported_train_image = subset_dir / "images" / "train" / "02.png"
            exported_test_image = subset_dir / "images" / "test" / "03.png"
            exported_train_label = subset_dir / "labels" / "train" / "02.txt"
            exported_test_label = subset_dir / "labels" / "test" / "03.txt"
            exported_sidecar = exported_train_image.with_suffix(".json")
            assert exported_train_image.read_bytes() == source_image_bytes
            assert exported_test_image.read_bytes() == (folders[1] / "images" / "03.png").read_bytes()
            assert exported_sidecar.read_bytes() == source_sidecar_bytes
            assert len(exported_train_label.read_text(encoding="utf-8").splitlines()) == len(source_yolo_lines) == 2
            assert len(exported_test_label.read_text(encoding="utf-8").splitlines()) == 1

            page.get_by_label("Subset image handling").select_option("symlink")
            page.get_by_test_id("curation-subset-choose-folder").click()
            expect(page.get_by_label("Subset destination folder")).to_have_value(str(symlink_dir))
            page.get_by_test_id("curation-subset-preview-button").click()
            expect(page.get_by_test_id("curation-subset-preview")).to_contain_text(
                "Preflight: 2 available · 2 unique · 0 duplicate(s) · 0 missing"
            )
            with page.expect_response(
                lambda response: response.url.endswith("/api/v1/curation/subset/export")
                and response.ok
            ) as symlink_export_response:
                page.get_by_test_id("curation-subset-export-button").click()
            symlink_result = symlink_export_response.value.json()
            symlinked_images = [
                symlink_dir / "images" / "train" / "02.png",
                symlink_dir / "images" / "test" / "03.png",
            ]
            successful_symlinks = sum(path.is_symlink() for path in symlinked_images)
            assert symlink_result["exported"] == successful_symlinks
            assert symlink_result["errors"] == len(symlinked_images) - successful_symlinks
            assert all(not path.exists() or path.is_symlink() for path in symlinked_images)
            if symlink_result["errors"]:
                expect(page.locator(".alert-info")).to_contain_text(
                    "Export completed with issues: 0 image(s)"
                )
                expect(page.locator(".alert-info")).to_contain_text(
                    f"{symlink_result['errors']} error(s)"
                )
                assert symlink_result["partial"] is True
            else:
                expect(page.locator(".alert-info")).to_contain_text(
                    "YOLO subset exported: 2 image(s)"
                )
                assert symlink_result["partial"] is False
                assert symlink_result["exported"] == 2
                symlinked_train_image, symlinked_test_image = symlinked_images
                assert symlinked_train_image.is_symlink()
                assert symlinked_train_image.resolve() == selected_image.resolve()
                assert symlinked_train_image.read_bytes() == source_image_bytes
                assert symlinked_test_image.is_symlink()
                assert symlinked_test_image.resolve() == (folders[1] / "images" / "03.png").resolve()
            assert [request["destination_folder_grant_id"] for request in subset_requests] == [
                "hostgrant_subset", "hostgrant_subset",
                "hostgrant_symlink", "hostgrant_symlink",
            ]

            with (subset_dir / "manifest.csv").open(encoding="utf-8", newline="") as handle:
                manifest_rows = list(csv.DictReader(handle))
            assert {row["dst"] for row in manifest_rows} == {
                "images/train/02.png", "images/test/03.png",
            }
            assert {row["object_ids"] for row in manifest_rows} == {"0", "1"}
            assert all(row["level"] == "object" for row in manifest_rows)
            lineage = json.loads((subset_dir / "lineage.json").read_text(encoding="utf-8"))
            assert "skew" in lineage["warning"].lower()
            assert sorted(lineage["sha256_list"]) == sorted(
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (selected_image, folders[1] / "images" / "03.png")
            )
            assert source_hashes == {
                path.relative_to(folders[0].parent.parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for folder in folders
                for path in folder.rglob("*")
                if path.is_file()
            }
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)


def test_lv_web_01_od_02_real_model_object_embeddings_on_arm64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Pinned object flow matches Streamlit vectors and KMeans drilldown in React."""
    import app as legacy_app

    models_dir = _pinned_dinov2_models_dir(tmp_path)
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
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            run_payloads = []
            block_responses: list[tuple[dict, dict]] = []
            page.on(
                "response",
                lambda response: run_payloads.append(response.json())
                if _is_run_summary_response(response) and response.ok
                else None,
            )

            def capture_blocks(response):
                if not response.ok or not urlparse(response.url).path.endswith("/blocks"):
                    return
                request = response.request.post_data_json
                if request:
                    try:
                        payload = response.json()
                    except Exception:
                        return
                    block_responses.append((request, payload))

            page.on("response", capture_blocks)
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
            with application.state.lock:
                run_id = application.state.current_run_id
                stored_run = application.state.runs[run_id]
                raw_embeddings = application.state.raw_embeddings[run_id][
                    "dinov2_vits14"
                ].copy()
            assert raw_embeddings.shape == (5, 384)
            class_names = list(run["class_names"])
            monkeypatch.setattr(legacy_app.st, "session_state", {})
            monkeypatch.setattr(
                legacy_app,
                "load_model",
                lambda model_id, **kwargs: load_model(
                    model_id, models_dir=models_dir, **kwargs,
                ),
            )
            legacy_records, legacy_vectors, _legacy_token = legacy_app._crop_and_embed_objects(
                legacy_app.discover_images(folders, class_names),
                "dinov2_vits14",
                class_names,
                0.12,
                base_token=f"react-web-od-02-{run_id}",
                policy=run["object_policy"],
            )
            assert len(legacy_records) == len(stored_run["records"]) == 5
            np.testing.assert_allclose(raw_embeddings, legacy_vectors, rtol=1e-6, atol=1e-7)
            for expected, actual in zip(legacy_records, stored_run["records"], strict=True):
                assert actual["filename"] == expected["image_path"].name
                assert actual["label"] == expected["label"]
                assert actual["class_id"] == expected["class_id"]
                assert actual["obj_index"] == expected["obj_index"]
                np.testing.assert_allclose(actual["bbox"], expected["bbox"], rtol=0, atol=1e-7)
            page.get_by_label("Filter by class").select_option(label="stain")
            stain_points = page.get_by_test_id("embedding-point")
            expect(stain_points).to_have_count(2)
            stain_points.nth(0).click()
            expect(page.get_by_test_id("selected-image-name")).to_have_text("02.png")
            expect(page.get_by_test_id("selected-object-index")).to_have_text("2")
            expect(page.get_by_test_id("selected-object-bbox")).to_be_visible()
            expect(page.get_by_test_id("selected-image")).to_have_js_property("naturalWidth", 32)

            stain_points.nth(1).click()
            expect(page.get_by_test_id("selection-count")).to_have_text("2 selected")
            page.get_by_test_id("add-selection-to-curation").click()
            expect(page.locator(".alert-info")).to_contain_text(
                "Added 2 item(s) to curation; skipped 0 duplicate(s)."
            )
            curation = page.request.get(base_url + "/api/v1/curation/items").json()
            assert curation["total"] == 2
            assert [
                (item["filename"], item["label"], item["level"], item["obj_index"], item["bbox"])
                for item in curation["items"]
            ] == [
                ("02.png", "stain", "object", 1, run["records"][2]["bbox"]),
                ("03.png", "stain", "object", 0, run["records"][3]["bbox"]),
            ]
            assert all("path" not in item and "image_path" not in item for item in curation["items"])

            page.get_by_label("Filter by class").select_option(label="All classes")
            page.get_by_label("Display method").select_option("pca")
            page.get_by_test_id("dimension-3d").click()
            expect(page.get_by_test_id("embedding-plot-3d")).to_be_visible(timeout=60_000)
            kmeans_count = page.get_by_test_id("kmeans-count")
            kmeans_count.focus()
            kmeans_count.press("Home")
            expect(page.get_by_test_id("kmeans-status")).to_contain_text(
                "4 個區塊", timeout=30_000,
            )
            expect(page.get_by_test_id("cluster-block-select")).to_be_enabled()

            pca_view = next(
                view for view in stored_run["views"]
                if view["model_id"] == "dinov2_vits14" and view["method"] == "pca"
            )
            view_points = sorted(pca_view["points"], key=lambda point: point["index"])
            indices = [int(point["index"]) for point in view_points]
            coordinates = np.asarray(
                [[point["x"], point["y"], point["z"]] for point in view_points],
                dtype=float,
            )
            monkeypatch.setattr(legacy_app.st, "session_state", {})
            legacy_blocks, legacy_centers = legacy_app._viz_kmeans_blocks(
                coordinates, indices, 4, f"real-object-{run_id}",
            )
            base_responses = [
                (request, response)
                for request, response in reversed(block_responses)
                if request.get("k") == 4
            ]
            assert base_responses, f"Captured KMeans requests: {[item[0] for item in block_responses]}"
            base_request, base_response = base_responses[0]
            assert base_request["model_id"] == "dinov2_vits14"
            assert base_request["method"] == "pca"
            react_blocks = {
                int(point["index"]): int(point["block_id"])
                for point in base_response["points"]
            }
            assert react_blocks == legacy_blocks
            react_centers = sorted(base_response["centers"], key=lambda item: item["block_id"])
            np.testing.assert_allclose(
                [[center["x"], center["y"], center["z"]] for center in react_centers],
                legacy_centers,
                rtol=1e-6,
                atol=1e-6,
            )
            largest = max(base_response["centers"], key=lambda center: center["count"])
            page.get_by_test_id("cluster-block-select").select_option(str(largest["block_id"]))
            expect(page.get_by_test_id("block-point")).to_have_count(
                largest["count"], timeout=30_000,
            )
            selected_response = next(
                response
                for request, response in reversed(block_responses)
                if request.get("k") == 4 and request.get("block_id") == largest["block_id"]
            )
            members = sorted(
                index for index, block_id in legacy_blocks.items()
                if block_id == largest["block_id"]
            )
            if len(members) > 2 and float(np.var(legacy_vectors[members])) > 0.0:
                expected_drilldown = legacy_app.PCA(
                    n_components=2, random_state=42,
                ).fit_transform(legacy_vectors[members])
            else:
                expected_drilldown = coordinates[members, :2]
            assert [point["index"] for point in selected_response["drilldown_points"]] == members
            np.testing.assert_allclose(
                [[point["x"], point["y"]] for point in selected_response["drilldown_points"]],
                expected_drilldown,
                rtol=1e-6,
                atol=1e-6,
            )
            expect(page.get_by_test_id("block-drilldown")).to_be_visible()
            expect(page.get_by_test_id("block-drilldown")).to_contain_text(
                f"#{largest['block_id']}"
            )

            page.get_by_test_id("open-curation").click()
            expect(page.get_by_test_id("curation-item")).to_have_count(2)
            item_headings = page.locator(".curation-item-heading strong")
            expect(item_headings.nth(0)).to_have_text("02.png")
            expect(item_headings.nth(1)).to_have_text("03.png")
            item_metadata = page.locator(".curation-item-meta")
            expect(item_metadata.nth(0)).to_contain_text("stain · train · manual · object 2")
            expect(item_metadata.nth(1)).to_contain_text("stain · test · manual · object 1")
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
    """Run detector mode through React and compare it with the legacy Visualize path."""
    models_dir = _pinned_dinov2_models_dir(tmp_path)
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

            # The legacy Streamlit Visualize path calls these same public
            # discovery/model helpers, then fits sklearn PCA with this seed and
            # the sample-count dimensionality rule. Compare both its raw model
            # vectors and visible React coordinates on the exact same files.
            from sklearn.decomposition import PCA
            from _utils import extract_embeddings
            from visualize_embeddings import discover_images

            legacy_records = discover_images(folders, ["scratch", "stain"])
            legacy_embeddings = extract_embeddings(
                [record["path"] for record in legacy_records],
                load_model("dinov2_vits14", models_dir=models_dir),
            )
            legacy_dimensions = min(3, max(1, len(legacy_records) - 2))
            legacy_pca = PCA(
                n_components=legacy_dimensions, random_state=42
            ).fit_transform(legacy_embeddings)
            active_run_id = application.state.current_run_id
            assert active_run_id is not None
            react_embeddings = application.state.raw_embeddings[active_run_id]["dinov2_vits14"]
            np.testing.assert_allclose(
                react_embeddings, legacy_embeddings, rtol=1e-5, atol=1e-6
            )
            pca_view = next(view for view in run["views"] if view["method"] == "pca")
            react_pca = np.asarray(
                [[point["x"], point["y"]] for point in pca_view["points"]],
                dtype=np.float32,
            )
            np.testing.assert_allclose(
                react_pca, legacy_pca[:, :2], rtol=1e-5, atol=1e-5
            )
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


def test_lv_web_04_ctx_01_diagnostics_reference_stale_and_run_incompatible_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Visible diagnostics handoff rechecks identity and never infers on page entry."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    dataset = _fixture(tmp_path / "diagnostic-fixture")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    inference_calls: list[str] = []
    diagnostic_attempts: list[int] = []

    def fixture_loader(_model_id: str):
        def embed(path: Path) -> np.ndarray:
            inference_calls.append(str(path))
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = np.asarray(Image.open(path).convert("RGB")).mean(axis=(0, 1)) / 255.0
            return vector
        return embed

    def diagnostic_loader(_model_id: str, resolution: int, _keep_aspect: bool, _pooling: str):
        diagnostic_attempts.append(resolution)
        if resolution == 448 and diagnostic_attempts.count(448) == 1:
            raise RuntimeError("fixture first 448 attempt failed")

        def embed(path: Path) -> np.ndarray:
            vector = np.zeros(384, dtype=np.float32)
            vector[0] = float(Image.open(path).convert("RGB").getpixel((0, 0))[0]) / 255.0
            vector[1] = resolution / 448.0
            return vector
        return embed

    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=fixture_loader,
        diagnostic_model_loader=diagnostic_loader,
    )
    server = server_thread = None
    source_path: Path | None = None
    original_source: bytes | None = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.set_default_timeout(15_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in (dataset / "train", dataset / "test"):
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()

            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=180_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)
            page.get_by_test_id("embedding-point").nth(0).click()

            with page.expect_response(lambda response: response.url.endswith("/api/v1/diagnostic-references")
                                      and response.request.method == "POST") as created_response:
                page.get_by_test_id("open-diagnostics-reference").click()
            created = created_response.value
            assert created.status == 201
            assert str(dataset) not in (created.request.post_data or "")
            reference = created.json()
            assert reference["status"] == "current"
            assert reference["source_sha256"]
            assert str(dataset) not in created.text()
            expect(page.get_by_test_id("diagnostic-reference-status")).to_contain_text("Context current")
            expect(page.get_by_test_id("diagnostics-no-auto-inference")).to_be_visible()
            expect(page.get_by_test_id("diagnostic-head-status")).to_contain_text("Head not prepared")
            expect(page.get_by_test_id("diagnostic-head-build")).to_be_enabled()

            inference_calls_at_handoff = len(inference_calls)
            page.get_by_test_id("diagnostic-recheck").click()
            expect(page.get_by_test_id("diagnostic-reference-status")).to_contain_text("Context current")
            assert len(inference_calls) == inference_calls_at_handoff
            assert diagnostic_attempts == []

            source_path = next(dataset.rglob(reference["filename"]))
            original_source = source_path.read_bytes()
            Image.new("RGB", (64, 64), (2, 3, 4)).save(source_path)
            page.get_by_test_id("diagnostic-recheck").click()
            expect(page.get_by_test_id("diagnostic-reference-status")).to_contain_text("Context stale")
            assert len(inference_calls) == inference_calls_at_handoff
            source_path.write_bytes(original_source)
            page.get_by_test_id("diagnostic-recheck").click()
            expect(page.get_by_test_id("diagnostic-reference-status")).to_contain_text("Context current")
            expect(page.get_by_test_id("diagnostic-head-status")).to_contain_text("Head not prepared")

            # Preparing 224 reuses the matching Visualize vectors. Higher
            # resolutions build their own features only after an explicit click.
            page.get_by_test_id("diagnostic-head-build").click()
            expect(page.get_by_test_id("diagnostic-head-status")).to_contain_text("Head ready: 224×224")
            assert diagnostic_attempts == []
            assert len(inference_calls) == inference_calls_at_handoff

            page.get_by_label("Analysis resolution").select_option("336")
            expect(page.get_by_test_id("diagnostic-head-status")).to_contain_text("Head not prepared")
            assert diagnostic_attempts == []
            page.get_by_test_id("diagnostic-head-build").click()
            expect(page.get_by_test_id("diagnostic-head-status")).to_contain_text("Head ready: 336×336")
            assert diagnostic_attempts == [336]

            page.get_by_label("Analysis resolution").select_option("448")
            expect(page.get_by_test_id("diagnostic-head-status")).to_contain_text("Head not prepared")
            page.get_by_test_id("diagnostic-head-build").click()
            expect(page.get_by_role("alert")).to_contain_text("could not be prepared")
            expect(page.get_by_test_id("diagnostic-head-build")).to_be_enabled()
            page.get_by_test_id("diagnostic-head-build").click()
            expect(page.get_by_test_id("diagnostic-head-status")).to_contain_text("Head ready: 448×448")
            assert diagnostic_attempts == [336, 448, 448]
            assert len(inference_calls) == inference_calls_at_handoff

            page.get_by_role("button", name="Visualize", exact=True).click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=180_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)
            page.get_by_test_id("open-diagnostics").click()
            expect(page.get_by_test_id("diagnostic-reference-status")).to_contain_text(
                "Context incompatible"
            )
            browser.close()
    finally:
        if source_path is not None and original_source is not None:
            source_path.write_bytes(original_source)
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert len(inference_calls) > 0, "the fixture run should have invoked its model before handoff"


def test_lv_web_04_expl_01_visible_attention_and_gradcam_journey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Visible explanation controls are explicit and keep signed evidence distinct from masks."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    import dino_explain
    import gradcam_explain

    dataset = _fixture(tmp_path / "diagnostic-explanation-fixture")
    fixture_hashes = {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    inference_calls: list[str] = []
    attention_calls: list[object] = []
    gradcam_calls: list[tuple[object, str | None]] = []
    checkpoint = tmp_path / "fixture-checkpoint.pth"
    checkpoint.write_bytes(b"fixture checkpoint identity")

    class FakeExplainer:
        pass

    def fake_attention(request, *, explainer=None):
        attention_calls.append(request)
        return types.SimpleNamespace(
            raw_patch_scores=np.array([[0.2, 0.8], [0.1, 0.4]], dtype=np.float32),
            display_patch_scores=np.array([[0.25, 1.0], [0.125, 0.5]], dtype=np.float32),
            grid=(2, 2), patch_size=14, method=request.method,
            input_image=request.image,
            warnings=("Attention is not a defect mask.",),
        )

    def fake_gradcam(request, head, *, target_class, explainer=None):
        gradcam_calls.append((request, target_class))
        return types.SimpleNamespace(
            raw_patch_scores=np.array([[-1.0, 2.0], [0.5, -0.25]], dtype=np.float32),
            display_patch_scores=np.array([[-0.5, 1.0], [0.25, -0.125]], dtype=np.float32),
            grid=(2, 2), patch_size=14,
            input_image=request.image,
            target_class=target_class or "apple", target_score=0.75,
            reference_class="banana", predicted_class="apple",
            predicted_probability=0.7, runner_up_class="banana",
            runner_up_probability=0.3, probability_margin=0.4,
            target_probability=0.7, reference_probability=0.3,
            is_predicted_target=target_class in (None, "apple"),
            positive_mass=2.5, negative_mass=1.25,
            faithfulness={"available": True, "status": "supported"},
            head_diagnostics={"available": False}, sample_oof=None,
            warnings=("Signed evidence is not segmentation.",),
        )

    def fixture_loader(_model_id: str):
        def embed(path: Path) -> np.ndarray:
            inference_calls.append(str(path))
            vector = np.zeros(384, dtype=np.float32)
            vector[:3] = np.asarray(Image.open(path).convert("RGB")).mean(axis=(0, 1)) / 255.0
            return vector
        return embed

    def diagnostic_loader(_model_id: str, resolution: int, _keep_aspect: bool, _pooling: str):
        def embed(path: Path) -> np.ndarray:
            vector = np.zeros(384, dtype=np.float32)
            vector[0] = float(Image.open(path).convert("RGB").getpixel((0, 0))[0]) / 255.0
            vector[1] = resolution / 448.0
            return vector
        return embed

    monkeypatch.setattr(dino_explain, "_resolve_checkpoint", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(
        dino_explain, "get_explainer",
        lambda *_args, **_kwargs: (FakeExplainer(), checkpoint),
    )
    monkeypatch.setattr(dino_explain, "attention_map", fake_attention)
    monkeypatch.setattr(gradcam_explain, "token_gradcam_map", fake_gradcam)

    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=fixture_loader,
        diagnostic_model_loader=diagnostic_loader,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1450, "height": 1050})
            page.set_default_timeout(15_000)
            explanation_responses = []
            page.on(
                "response",
                lambda response: explanation_responses.append(response)
                if response.url.endswith("/explanations")
                else None,
            )
            page.goto(base_url, wait_until="networkidle")
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in (dataset / "train", dataset / "test"):
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=180_000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)
            page.get_by_test_id("embedding-point").nth(0).click()
            page.get_by_test_id("open-diagnostics-reference").click()
            expect(page.get_by_test_id("diagnostic-reference-status")).to_contain_text("Context current")
            expect(page.get_by_test_id("diagnostic-head-status")).to_contain_text("Head not prepared")
            expect(page.get_by_test_id("diagnostics-no-auto-inference")).to_be_visible()
            expect(page.get_by_test_id("diagnostic-explain-run")).to_be_disabled()
            assert not attention_calls and not gradcam_calls and not explanation_responses

            page.get_by_test_id("diagnostic-head-build").click()
            expect(page.get_by_test_id("diagnostic-head-status")).to_contain_text("Head ready: 224×224")
            expect(page.get_by_test_id("diagnostic-explain-run")).to_be_enabled()
            assert not attention_calls and not gradcam_calls

            with page.expect_response(
                lambda response: response.url.endswith("/explanations")
                and response.request.method == "POST"
            ) as attention_response:
                page.get_by_test_id("diagnostic-explain-run").click()
            attention_payload = attention_response.value.json()
            assert attention_payload["method"] == "attention"
            assert attention_payload["attention_method"] == "last_layer_head_mean"
            assert str(dataset) not in attention_response.value.text()
            expect(page.get_by_test_id("diagnostic-evidence-grid")).to_be_visible()
            expect(page.get_by_test_id("diagnostic-evidence-grid").locator("span")).to_have_count(4)
            expect(page.get_by_test_id("diagnostic-explanation-result")).to_contain_text("not a defect mask")
            assert len(attention_calls) == 1

            page.get_by_test_id("diagnostic-method").select_option("gradcam")
            page.get_by_test_id("diagnostic-target-class").select_option("banana")
            page.get_by_test_id("diagnostic-display-mode").select_option("signed")
            with page.expect_response(
                lambda response: response.url.endswith("/explanations")
                and response.request.method == "POST"
            ) as gradcam_response:
                page.get_by_test_id("diagnostic-explain-run").click()
            gradcam_payload = gradcam_response.value.json()
            assert gradcam_payload["method"] == "gradcam"
            assert gradcam_payload["target_class"] == "banana"
            assert gradcam_payload["score_mode"] == "signed"
            assert gradcam_payload["display_scores"][0][0] < 0
            assert gradcam_payload["display_scores"][0][1] > 0
            assert str(dataset) not in gradcam_response.value.text()
            expect(page.get_by_test_id("diagnostic-target-result")).to_contain_text("banana")
            expect(page.get_by_test_id("diagnostic-faithfulness")).to_contain_text("supported")
            expect(page.get_by_test_id("diagnostic-model-input-image")).to_be_visible()
            expect(page.get_by_test_id("diagnostic-overlay-image")).to_have_js_property("naturalWidth", 64)
            with page.expect_download() as png_download:
                page.get_by_test_id("diagnostic-download-png").click()
            png_file = png_download.value.path()
            assert png_file and Image.open(png_file).format == "PNG"
            with page.expect_download() as json_download:
                page.get_by_test_id("diagnostic-download-json").click()
            json_file = json_download.value.path()
            assert json_file
            downloaded_metadata = json.loads(Path(json_file).read_text(encoding="utf-8"))
            assert downloaded_metadata["target_class"] == "banana"
            assert "overlay_png_base64" not in downloaded_metadata
            assert str(dataset) not in Path(json_file).read_text(encoding="utf-8")
            assert len(gradcam_calls) == 1 and gradcam_calls[0][1] == "banana"
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert fixture_hashes == {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }, "diagnostics changed the source dataset"


def test_lv_web_04_expl_01_real_dinov2_attention_and_gradcam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Real pinned DINO weights execute both evidence methods at the exact classifier resolution."""
    from fastapi.testclient import TestClient
    import _utils
    from dino_explain import ExplainRequest, _resolve_checkpoint, attention_map, get_explainer
    from gradcam_explain import token_gradcam_map

    models_dir = Path(
        os.environ.get("LV_MODELS_DIR", r"C:\code\claude\LV\visuallatent\models")
    ).resolve()
    weights = models_dir / "dinov2_vits14" / "dinov2_vits14.pth"
    if not weights.is_file():
        pytest.skip(f"pinned real DINOv2-small weights are unavailable: {weights}")
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == DINOV2_VITS14_SHA256, \
        "the local DINOv2-small weights differ from the pinned diagnostics baseline"
    monkeypatch.setattr(_utils, "_DEFAULT_MODELS_DIR", models_dir)
    monkeypatch.setattr(_utils.available_models, "__defaults__", (models_dir,))
    monkeypatch.setattr(
        _utils.load_model,
        "__defaults__",
        (models_dir, *_utils.load_model.__defaults__[1:]),
    )
    cache_root = tmp_path / "real-diagnostics-cache"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache_root))

    dataset = _fixture(tmp_path / "real-diagnostics-fixture")
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }
    application = create_app()
    with TestClient(application) as client:
        response = client.post("/api/v1/visualize/jobs", json={
            "folders": [str(dataset / "train"), str(dataset / "test")],
            "model_id": "dinov2_vits14",
            "method": "pca",
        })
        assert response.status_code == 202, response.text
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            job = client.get(f"/api/v1/jobs/{job_id}").json()
            if job["state"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        assert job["state"] == "completed", job
        run_id = job["run_id"]
        run = client.get(f"/api/v1/runs/{run_id}").json()
        reference = client.post("/api/v1/diagnostic-references", json={
            "run_id": run_id,
            "record_index": 0,
            "model_id": "dinov2_vits14",
        }).json()
        head = client.post(
            f"/api/v1/diagnostic-references/{reference['context_id']}/heads",
            json={"current_run_id": run_id, "resolution": 224},
        )
        assert head.status_code == 200, head.text
        cache_before_explanation = hashlib.sha256(
            b"".join(
                str(path.relative_to(cache_root)).encode("utf-8") + path.read_bytes()
                for path in sorted(cache_root.rglob("*")) if path.is_file()
            )
        ).hexdigest()
        attention = client.post(
            f"/api/v1/diagnostic-references/{reference['context_id']}/explanations",
            json={
                "current_run_id": run_id,
                "resolution": 224,
                "method": "attention",
                "attention_method": "last_layer_head_mean",
            },
        )
        gradcam = client.post(
            f"/api/v1/diagnostic-references/{reference['context_id']}/explanations",
            json={
                "current_run_id": run_id,
                "resolution": 224,
                "method": "gradcam",
                "target_class": "banana",
                "display_mode": "signed",
            },
        )
        internal_run = client.app.state.runs[run_id]
        context = client.app.state.diagnostic_references[reference["context_id"]]
        head_entry = next(
            entry for entry in client.app.state.diagnostic_heads.values()
            if entry["resolution"] == 224 and entry["model_id"] == "dinov2_vits14"
        )
        source_path = Path(internal_run["image_paths"][0])
        source_image = Image.open(source_path).convert("RGB")
        source_id = f"{context['source_sha256']}:{internal_run['records'][0].get('obj_index', '')}"
        checkpoint = _resolve_checkpoint("dinov2_vits14", None)
        explainer, checkpoint = get_explainer("dinov2_vits14", checkpoint, head="cls")
        direct_request = ExplainRequest(
            image=source_image,
            source_id=source_id,
            model_id="dinov2_vits14",
            checkpoint_path=checkpoint,
            keep_aspect=False,
            target_res=224,
            pooling="cls",
            original_size=source_image.size,
        )
        direct_attention = attention_map(direct_request, explainer=explainer)
        direct_gradcam = token_gradcam_map(
            direct_request,
            head_entry["head"],
            target_class="banana",
            explainer=explainer,
        )
        cache_after_explanation = hashlib.sha256(
            b"".join(
                str(path.relative_to(cache_root)).encode("utf-8") + path.read_bytes()
                for path in sorted(cache_root.rglob("*")) if path.is_file()
            )
        ).hexdigest()

    assert attention.status_code == 200, attention.text
    assert gradcam.status_code == 200, gradcam.text
    attention_payload = attention.json()
    gradcam_payload = gradcam.json()
    assert run["embedding_shape"] == [4, 384]
    assert attention_payload["grid"] == [16, 16]
    assert gradcam_payload["grid"] == [16, 16]
    assert attention_payload["predicted_class"] in {"apple", "banana"}
    assert gradcam_payload["target_class"] == "banana"
    assert gradcam_payload["reference_class"] in {"apple", "banana"}
    assert np.isfinite(np.asarray(attention_payload["display_scores"], dtype=float)).all()
    assert np.isfinite(np.asarray(gradcam_payload["display_scores"], dtype=float)).all()
    assert "not a defect mask" in gradcam_payload["warning"]
    np.testing.assert_allclose(
        np.asarray(attention_payload["raw_scores"], dtype=np.float32),
        direct_attention.raw_patch_scores,
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(gradcam_payload["raw_scores"], dtype=np.float32),
        direct_gradcam.raw_patch_scores,
        rtol=0,
        atol=1e-7,
    )
    assert str(dataset) not in attention.text + gradcam.text
    assert cache_before_explanation == cache_after_explanation
    assert source_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }


def test_lv_web_04_expl_01_visible_real_dinov2_attention_and_gradcam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Run real pinned DINO diagnostics through the visible React interface."""
    models_dir = _pinned_dinov2_models_dir(tmp_path)
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    import _utils

    monkeypatch.setattr(_utils, "_DEFAULT_MODELS_DIR", models_dir)
    monkeypatch.setattr(_utils.available_models, "__defaults__", (models_dir,))
    monkeypatch.setattr(
        _utils.load_model,
        "__defaults__",
        (models_dir, *_utils.load_model.__defaults__[1:]),
    )
    cache_root = tmp_path / "visible-real-diagnostics-cache"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache_root))

    dataset = _fixture(tmp_path / "visible-real-diagnostics-fixture")
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }
    application = create_app()
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1450, "height": 1050})
            page.set_default_timeout(15_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("model-choice-dinov2_vits14").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            for folder in (dataset / "train", dataset / "test"):
                page.get_by_label("Image folder").fill(str(folder))
                page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text(
                "Completed", timeout=180_000
            )
            expect(page.get_by_test_id("embedding-point")).to_have_count(4)

            page.get_by_test_id("embedding-point").nth(0).click()
            page.get_by_test_id("open-diagnostics-reference").click()
            expect(page.get_by_test_id("diagnostic-reference-status")).to_contain_text(
                "Context current"
            )
            expect(page.get_by_test_id("diagnostics-no-auto-inference")).to_be_visible()
            expect(page.get_by_test_id("diagnostic-explain-run")).to_be_disabled()
            page.get_by_test_id("diagnostic-head-build").click()
            expect(page.get_by_test_id("diagnostic-head-status")).to_contain_text(
                "Head ready: 224×224", timeout=180_000
            )

            with page.expect_response(
                lambda response: response.url.endswith("/explanations")
                and response.request.method == "POST",
                timeout=180_000,
            ) as attention_response:
                page.get_by_test_id("diagnostic-explain-run").click()
            attention_payload = attention_response.value.json()
            assert attention_payload["method"] == "attention"
            assert attention_payload["grid"] == [16, 16]
            assert np.isfinite(
                np.asarray(attention_payload["display_scores"], dtype=np.float32)
            ).all()
            assert "not a defect mask" in attention_payload["warning"]
            expect(page.get_by_test_id("diagnostic-evidence-grid")).to_be_visible()
            expect(
                page.get_by_test_id("diagnostic-evidence-grid").locator("span")
            ).to_have_count(256)

            page.get_by_test_id("diagnostic-method").select_option("gradcam")
            page.get_by_test_id("diagnostic-target-class").select_option("banana")
            page.get_by_test_id("diagnostic-display-mode").select_option("signed")
            with page.expect_response(
                lambda response: response.url.endswith("/explanations")
                and response.request.method == "POST",
                timeout=180_000,
            ) as gradcam_response:
                page.get_by_test_id("diagnostic-explain-run").click()
            gradcam_payload = gradcam_response.value.json()
            assert gradcam_payload["method"] == "gradcam"
            assert gradcam_payload["target_class"] == "banana"
            assert gradcam_payload["score_mode"] == "signed"
            assert gradcam_payload["grid"] == [16, 16]
            assert np.isfinite(
                np.asarray(gradcam_payload["display_scores"], dtype=np.float32)
            ).all()
            expect(page.get_by_test_id("diagnostic-target-result")).to_contain_text(
                "banana"
            )
            expect(page.get_by_test_id("diagnostic-model-input-image")).to_be_visible()
            expect(page.get_by_test_id("diagnostic-overlay-image")).to_have_js_property(
                "naturalWidth", 224
            )
            with page.expect_download() as png_download:
                page.get_by_test_id("diagnostic-download-png").click()
            png_file = png_download.value.path()
            assert png_file and Image.open(png_file).format == "PNG"
            with page.expect_download() as json_download:
                page.get_by_test_id("diagnostic-download-json").click()
            json_file = json_download.value.path()
            assert json_file
            metadata = json.loads(Path(json_file).read_text(encoding="utf-8"))
            assert metadata["method"] == "gradcam"
            assert metadata["target_class"] == "banana"
            assert "overlay_png_base64" not in metadata
            assert str(dataset) not in Path(json_file).read_text(encoding="utf-8")
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }, "real-model diagnostics changed the source dataset"


def test_lv_web_05_case_01_visible_case_intake_source_navigation_and_region_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Visible React intake, source identity, prediction import, review and case reopen."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    from PIL import Image
    from playwright.sync_api import expect, sync_playwright

    cases_root = tmp_path / "fp-cases"
    monkeypatch.setenv("LV_CASES_DIR", str(cases_root))
    screenshot_path = tmp_path / "reported-screen.png"
    original_path = tmp_path / "original-image.png"
    predictions_path = tmp_path / "predictions.txt"
    Image.new("RGB", (80, 60), (220, 80, 35)).save(screenshot_path)
    Image.new("RGB", (120, 90), (35, 110, 190)).save(original_path)
    predictions_path.write_text(
        "0 0.50 0.50 0.40 0.40 0.91\n1 0.30 0.40 0.20 0.30 0.76\n",
        encoding="utf-8",
    )
    reference_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (screenshot_path, original_path, predictions_path)
    }
    application = create_app(model_catalog=lambda: [], model_loader=lambda _model: None)
    server, server_thread, base_url = _start_api(application)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1480, "height": 1024})
            page.set_default_timeout(15000)
            created_cases: list[dict] = []
            added_sources: list[dict] = []
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("open-fp-cases").click()
            expect(page.get_by_role("heading", name="建立誤逮案件")).to_be_visible()
            page.get_by_label("案件名稱（可留空）").fill("Reflective surface report")
            page.get_by_label("新案件來源類型").select_option("screenshot")
            page.get_by_label("建立案件圖片").set_input_files(str(screenshot_path))
            page.get_by_label("使用者描述／報案理由").fill("外部模型把可接受反光標成裂痕。")
            page.get_by_label("外部模型版本").fill("detector-2026.10")
            page.get_by_label("預測門檻").fill("0.82")
            page.get_by_label("外部模型原始預測／輸出").fill("class=crack score=0.94")
            with page.expect_response(lambda response: response.request.method == "POST"
                                      and response.url.rstrip("/").endswith("/api/v1/fp-cases")
                                      and response.ok) as created_response:
                page.get_by_role("button", name="建立案件", exact=True).click()
            created_cases.append(created_response.value.json())
            expect(page.get_by_test_id("fp-case-title-readback")).to_have_text("Reflective surface report")
            expect(page.get_by_test_id("fp-case-notice")).to_contain_text("尚未重播")
            expect(page.get_by_test_id("fp-source-image")).to_have_js_property("naturalWidth", 80)
            assert created_cases and created_cases[0]["sources"][0]["kind"] == "screenshot"
            case_id = created_cases[0]["id"]

            page.get_by_label("補入來源類型").select_option("original")
            page.get_by_label("補入原圖或截圖").set_input_files(str(original_path))
            with page.expect_response(lambda response: response.request.method == "POST"
                                      and response.url.rstrip("/").endswith("/sources")
                                      and response.ok) as source_response:
                page.get_by_role("button", name="加入此案件").click()
            added_sources.append(source_response.value.json())
            expect(page.get_by_test_id("fp-case-notice")).to_contain_text("原圖已加入")
            assert added_sources and len(added_sources[0]["sources"]) == 2
            original_source = next(source for source in added_sources[0]["sources"] if source["kind"] == "original")
            screenshot_source = next(source for source in added_sources[0]["sources"] if source["kind"] == "screenshot")
            page.get_by_test_id(f"fp-source-{original_source['id']}").click()
            expect(page.get_by_test_id("fp-source-image")).to_have_js_property("naturalWidth", 120)
            page.get_by_test_id(f"fp-source-{screenshot_source['id']}").click()
            expect(page.get_by_test_id("fp-source-image")).to_have_js_property("naturalWidth", 80)

            page.get_by_test_id("fp-yolo-file").set_input_files(str(predictions_path))
            page.get_by_role("button", name="匯入 YOLO 預測").click()
            expect(page.get_by_test_id("fp-case-notice")).to_contain_text("已匯入 2 個預測區域")
            expect(page.get_by_test_id("fp-region-count")).to_have_text("2 個區域")
            expect(page.get_by_test_id("fp-selected-region-roi")).to_be_visible()
            region_tabs = page.locator("[data-testid^='fp-region-tab-']")
            expect(region_tabs).to_have_count(2)
            first_region_id = region_tabs.nth(0).get_attribute("data-testid").removeprefix("fp-region-tab-")

            page.get_by_test_id("fp-review-verdict").select_option("normal")
            page.get_by_test_id("fp-reviewer").fill("Engineer A")
            page.get_by_test_id("fp-review-reason").fill("依規格，此 ROI 是可接受反光。")
            page.get_by_test_id("fp-save-review").click()
            expect(page.get_by_test_id("fp-case-notice")).to_contain_text("整圖標籤不會隨之改變")
            page.get_by_test_id(f"fp-region-tab-{first_region_id}").click()
            assert page.get_by_test_id("fp-review-verdict").input_value() == "normal"
            page.locator("[data-testid^='fp-region-tab-']").nth(1).click()
            page.get_by_test_id("fp-review-verdict").select_option("defect")
            page.get_by_test_id("fp-reviewer").fill("Engineer B")
            page.get_by_test_id("fp-review-reason").fill("此 ROI 有可見裂痕，與第一區域分開覆核。")
            page.get_by_test_id("fp-save-review").click()
            expect(page.locator("[data-testid^='fp-region-tab-']").nth(0)).to_contain_text("確認此區域正常")
            expect(page.locator("[data-testid^='fp-region-tab-']").nth(1)).to_contain_text("確認此區域有瑕疵")

            page.get_by_role("button", name="＋ 建立案件").click()
            page.get_by_label("案件名稱（可留空）").fill("Second report")
            page.get_by_label("建立案件圖片").set_input_files(str(original_path))
            with page.expect_response(lambda response: response.request.method == "POST"
                                      and response.url.rstrip("/").endswith("/api/v1/fp-cases")
                                      and response.ok) as second_created_response:
                page.get_by_role("button", name="建立案件", exact=True).click()
            created_cases.append(second_created_response.value.json())
            expect(page.get_by_test_id("fp-case-title-readback")).to_have_text("Second report")
            assert len(created_cases) == 2

            page.get_by_test_id(f"fp-open-case-{case_id}").click()
            expect(page.get_by_test_id("fp-case-title-readback")).to_have_text("Reflective surface report")
            expect(page.locator("[data-testid^='fp-region-tab-']").nth(0)).to_contain_text("確認此區域正常")
            page.locator("[data-testid^='fp-region-tab-']").nth(1).click()
            expect(page.get_by_test_id("fp-review-verdict")).to_have_value("defect")
            assert not page.get_by_test_id("fp-case-error").count()
            browser.close()
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)

    assert reference_hashes == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (screenshot_path, original_path, predictions_path)
    }, "case intake or review modified its source files"


def test_lv_web_07_sample_01_visible_builds_yolo_and_whole_image_banks_without_anomaly_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Build and read back both sample-source modes from the dedicated React workspace."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    yolo_root = tmp_path / "fewshot-yolo"
    (yolo_root / "images").mkdir(parents=True)
    (yolo_root / "labels").mkdir()
    (yolo_root / "classes.txt").write_text("scratch\nstain\n", encoding="utf-8")
    for index, color in enumerate(((210, 40, 50), (30, 190, 70))):
        Image.new("RGB", (48, 40), color).save(yolo_root / "images" / f"sample-{index}.png")
    (yolo_root / "labels" / "sample-0.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n1 0.25 0.25 0.25 0.25\n", encoding="utf-8",
    )
    (yolo_root / "labels" / "sample-1.txt").write_text(
        "0 0.5 0.5 0.75 0.75\n", encoding="utf-8",
    )

    whole_root = tmp_path / "fewshot-whole-image"
    (whole_root / "images").mkdir(parents=True)
    for index, color in enumerate(((190, 80, 50), (50, 90, 210))):
        Image.new("RGB", (52, 36), color).save(whole_root / "images" / f"whole-{index}.png")

    original_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (*yolo_root.rglob("*"), *whole_root.rglob("*")) if path.is_file()
    }
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: lambda crop: np.asarray(
            [np.asarray(crop.convert("RGB"), dtype=np.float32).mean(), crop.width, crop.height, 1],
            dtype=np.float32,
        ),
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(15_000)
            anomaly_model_build_requests = []
            page.on("request", lambda request: anomaly_model_build_requests.append(request.url)
                    if "/api/v1/anomaly/models/jobs" in request.url else None)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("open-fewshot-samples").click()
            expect(page.get_by_role("heading", name="以樣搜樣：建立樣本集")).to_be_visible()

            page.get_by_test_id("fewshot-sample-folders").fill(str(yolo_root))
            page.get_by_test_id("fewshot-object-source").select_option("yolo")
            page.get_by_test_id("fewshot-build-bank").click()
            expect(page.get_by_test_id("fewshot-bank-summary")).to_contain_text("3 個樣本", timeout=60_000)
            expect(page.get_by_test_id("fewshot-bank-summary")).to_contain_text("scratch 2、stain 1")
            first_bank_id = page.get_by_test_id("fewshot-bank-id").input_value()
            page.get_by_test_id("fewshot-reload-bank").click()
            expect(page.get_by_test_id("fewshot-notice")).to_contain_text("重新讀回", timeout=10_000)

            page.get_by_test_id("fewshot-sample-folders").fill(str(whole_root))
            page.get_by_test_id("fewshot-object-source").select_option("whole_image")
            page.get_by_test_id("fewshot-build-bank").click()
            expect(page.get_by_test_id("fewshot-bank-summary")).to_contain_text("2 個樣本", timeout=60_000)
            expect(page.get_by_test_id("fewshot-bank-summary")).to_contain_text("整張影像語義")
            second_bank_id = page.get_by_test_id("fewshot-bank-id").input_value()
            assert first_bank_id != second_bank_id
            assert anomaly_model_build_requests == []
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert original_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in original_hashes
    }


def test_lv_web_05_ext_01_visible_onnx_profile_inference_replay_and_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """React uploads a local ONNX, runs it explicitly, replays full history and downloads evidence."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")
    pytest.importorskip("onnx")

    from fp_onnx_demo import make_demo_model
    from playwright.sync_api import expect, sync_playwright

    cases_root = tmp_path / "fp-cases"
    monkeypatch.setenv("LV_CASES_DIR", str(cases_root))
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    source_path = tmp_path / "original.png"
    model_path = make_demo_model(tmp_path / "demo-classifier.onnx")
    Image.new("RGB", (80, 60), (220, 220, 220)).save(source_path)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

    application = create_app(model_catalog=lambda: [], model_loader=lambda _model: None)
    server, server_thread, base_url = _start_api(application)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1480, "height": 1120}, accept_downloads=True)
            page.set_default_timeout(20_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("open-fp-cases").click()
            page.get_by_label("案件名稱（可留空）").fill("External ONNX replay")
            page.get_by_label("新案件來源類型").select_option("original")
            page.get_by_label("建立案件圖片").set_input_files(str(source_path))
            page.get_by_role("button", name="建立案件", exact=True).click()
            expect(page.get_by_test_id("fp-case-title-readback")).to_have_text("External ONNX replay")
            external = page.get_by_test_id("fp-external-models")
            expect(external).to_be_visible()
            page.get_by_label("外部模型設定名稱").fill("Visible fixture classifier")
            page.get_by_label("保存的模型版本名稱").fill("visible-fixture-v1")
            page.get_by_label("ONNX 模型檔").set_input_files(str(model_path))
            page.get_by_label("外部模型輸入寬度").fill("32")
            page.get_by_label("外部模型輸入高度").fill("32")
            with page.expect_response(
                lambda response: response.url.endswith("/api/v1/external-model-profiles")
                and response.request.method == "POST" and response.ok,
            ) as profile_response:
                page.get_by_test_id("fp-save-external-profile").click()
            profile = profile_response.value.json()
            assert profile["asset"]["sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
            expect(page.get_by_test_id("fp-external-notice")).to_contain_text("不會執行推論")

            with page.expect_response(
                lambda response: response.url.endswith("/external-runs")
                and response.request.method == "POST" and response.ok,
            ) as first_run_response:
                page.get_by_test_id("fp-run-external-model").click()
            first = first_run_response.value.json()
            first_run = first["run"]
            expect(page.get_by_test_id("fp-external-classification")).to_contain_text("crack")
            expect(page.get_by_test_id("fp-external-parity")).to_contain_text("歷史 parity 不可用")
            expect(page.get_by_test_id("fp-external-results")).to_be_visible()
            preview = external.locator(".fp-ext-input-preview")
            expect(preview).to_have_js_property("naturalWidth", 32)
            assert first["case"]["items"] == [], "inference must not silently become a human-reviewed region"
            assert str(tmp_path) not in first_run_response.value.text()

            reference = {
                "origin": "deployment",
                "recorded_at": "2026-09-01T00:00:00Z",
                "image_sha256": first_run["image_sha256"],
                "contract_id": first_run["contract_id"],
                "prediction": first_run["prediction"],
            }
            page.get_by_label("歷史部署輸出 JSON 文字").fill(json.dumps(reference))
            with page.expect_response(
                lambda response: response.url.endswith("/external-runs")
                and response.request.method == "POST" and response.ok,
            ) as replay_response:
                page.get_by_test_id("fp-run-external-model").click()
            replay = replay_response.value.json()
            expect(page.get_by_test_id("fp-external-parity")).to_contain_text("完整歷史輸出相符")
            assert replay["run"]["parity"]["status"] == "matched"

            detector_path = make_demo_model(tmp_path / "demo-detector.onnx", detection=True)
            page.get_by_label("外部模型輸出契約").select_option("xyxy")
            page.get_by_label("外部模型類別順序").fill("crack")
            page.get_by_label("外部模型正常類別").fill("")
            page.get_by_label("ONNX 模型檔").set_input_files(str(detector_path))
            with page.expect_response(
                lambda response: "/api/v1/external-model-profiles/" in response.url
                and response.request.method == "PATCH" and response.ok,
            ) as revised_profile_response:
                page.get_by_test_id("fp-save-external-profile").click()
            revised_profile = revised_profile_response.value.json()
            assert revised_profile["revision"] == 2
            assert revised_profile["asset"]["sha256"] == hashlib.sha256(detector_path.read_bytes()).hexdigest()
            assert revised_profile["asset"]["sha256"] != profile["asset"]["sha256"]
            page.get_by_label("歷史部署輸出 JSON 文字").fill("")
            with page.expect_response(
                lambda response: response.url.endswith("/external-runs")
                and response.request.method == "POST" and response.ok,
            ) as detection_response:
                page.get_by_test_id("fp-run-external-model").click()
            detection = detection_response.value.json()["run"]
            assert detection["profile_revision"] == 2
            assert detection["prediction"]["task"] == "detection"
            expect(page.get_by_test_id("fp-ext-detection-summary")).to_contain_text("最終輸出 1 個框")
            expect(external.locator(".fp-ext-detection-box")).to_have_count(1)
            assert detection["prediction"]["detections"][0]["box"] == pytest.approx(
                [0.125, 0.0, 0.875, 1.0], abs=1e-5,
            )

            with page.expect_download() as zip_download:
                page.get_by_role("button", name="下載完整推論 ZIP").click()
            zip_path = zip_download.value.path()
            assert zip_path
            with zipfile.ZipFile(zip_path) as archive:
                assert set(archive.namelist()) == {"evidence.json", "model_input.png", "source.png", "tensors.npz"}
                exported_evidence = archive.read("evidence.json").decode("utf-8")
                assert "model_path" not in exported_evidence
                assert str(tmp_path) not in exported_evidence
            browser.close()
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)

    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash


def test_lv_web_05_ext_01_visible_paired_evaluation_and_training_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Visible React journey for paired holdout results and reviewed YOLO ZIP export."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    from playwright.sync_api import expect, sync_playwright
    import fp_external_evaluation

    class FixtureRuntime:
        def __init__(self, config: dict):
            self.config = config
            marker = Path(config["model_path"]).read_bytes()
            self.bias = 0.3 if marker == b"baseline-fixture" else 0.0
            self.identity = {
                "contract_id": "a" * 64 if self.bias else "b" * 64,
                "model_sha256": "c" * 64 if self.bias else "d" * 64,
            }

        def infer(self, data: bytes) -> dict:
            pixels = np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.float32)
            crack = min(1.0, float(pixels.mean() / 255.0) + self.bias)
            alarm = crack >= float(self.config["threshold"])
            return {"prediction": {
                "task": "classification",
                "alarm": alarm,
                "class": "crack" if alarm else "good",
                "decision": "crack" if alarm else "good",
                "scores": {"good": 1.0 - crack, "crack": crack},
                "detections": [],
            }}

    monkeypatch.setattr(fp_external_evaluation, "Runtime", FixtureRuntime)
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "fp-cases"))
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))

    baseline_model = tmp_path / "baseline.onnx"
    candidate_model = tmp_path / "candidate.onnx"
    baseline_model.write_bytes(b"baseline-fixture")
    candidate_model.write_bytes(b"candidate-fixture")
    holdout_normal = tmp_path / "normal.png"
    holdout_defect = tmp_path / "defect.png"
    Image.new("RGB", (80, 60), (80, 80, 80)).save(holdout_normal)
    Image.new("RGB", (80, 60), (230, 230, 230)).save(holdout_defect)
    manifest = {
        "schema_version": 1,
        "purpose": "case_regression",
        "items": [
            {"id": "normal", "path": str(holdout_normal), "truth": "normal", "verified": True},
            {"id": "defect", "path": str(holdout_defect), "truth": "defect", "verified": True},
        ],
    }
    application = create_app(model_catalog=lambda: [], model_loader=lambda _model: None)
    server, server_thread, base_url = _start_api(application)

    def post_json(path: str, body: dict) -> dict:
        request = urllib.request.Request(
            base_url + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))

    def create_profile(path: Path, marker_name: str) -> dict:
        upload_request = urllib.request.Request(
            base_url + "/api/v1/external-model-assets?filename="
            + marker_name.replace(" ", "-") + ".onnx",
            data=path.read_bytes(),
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urllib.request.urlopen(upload_request, timeout=20) as response:
            asset = json.loads(response.read().decode("utf-8"))
        config = {
            "model_version": marker_name,
            "output_format": "classification",
            "class_names": ["good", "crack"],
            "normal_classes": ["good"],
            "input_size": [32, 32],
            "resize": "stretch",
            "threshold": 0.5,
        }
        return post_json(
            "/api/v1/external-model-profiles",
            {"name": marker_name, "asset_sha256": asset["sha256"], "config": config},
        )

    try:
        baseline_profile = create_profile(baseline_model, "Visible baseline")
        candidate_profile = create_profile(candidate_model, "Visible candidate")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": 1480, "height": 1120}, accept_downloads=True
            )
            page.set_default_timeout(20_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("open-fp-cases").click()
            page.get_by_label("案件名稱（可留空）").fill("Paired evaluation and training")
            page.get_by_label("新案件來源類型").select_option("original")
            source_path = tmp_path / "reported.png"
            Image.new("RGB", (80, 60), (220, 220, 220)).save(source_path)
            page.get_by_label("建立案件圖片").set_input_files(str(source_path))
            with page.expect_response(
                lambda response: response.url.rstrip("/").endswith("/api/v1/fp-cases")
                and response.request.method == "POST" and response.ok,
            ) as created_response:
                page.get_by_role("button", name="建立案件", exact=True).click()
            created_case = created_response.value.json()
            page.get_by_test_id("fp-open-case-" + created_case["id"]).click()
            expect(page.get_by_test_id("fp-evaluation-training")).to_be_visible()
            page.get_by_label("評估改善前模型").select_option(baseline_profile["id"])
            page.get_by_label("評估改善後模型").select_option(candidate_profile["id"])
            page.get_by_label("評估清單 JSON 內容").fill(json.dumps(manifest))

            with page.expect_response(
                lambda response: response.url.endswith("/evaluations")
                and response.request.method == "POST" and response.ok,
            ) as evaluation_response:
                page.get_by_test_id("fp-run-paired-evaluation").click()
            evaluated = evaluation_response.value.json()
            expect(page.get_by_test_id("fp-paired-evaluation-result")).to_contain_text(
                "case_regression_only"
            )
            assert evaluated["evaluation"]["baseline"]["false_alarms"] == 1
            assert evaluated["evaluation"]["candidate"]["false_alarms"] == 0

            case = evaluated["case"]
            region = post_json(
                "/api/v1/fp-cases/" + case["id"] + "/regions",
                {
                    "expected_revision": case["revision"],
                    "source_id": case["sources"][0]["id"],
                    "roi": [0.2, 0.2, 0.6, 0.7],
                    "label": "model-suggestion",
                },
            )
            case = region["case"]
            item_id = case["items"][0]["id"]
            review_request = urllib.request.Request(
                base_url + "/api/v1/fp-cases/" + case["id"] + "/regions/" + item_id + "/review",
                data=json.dumps({
                    "expected_revision": case["revision"],
                    "verdict": "defect",
                    "reviewer": "Visible reviewer",
                    "reason": "Confirmed a true defect in this region.",
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="PATCH",
            )
            with urllib.request.urlopen(review_request, timeout=20) as response:
                case = json.loads(response.read().decode("utf-8"))
            page.reload(wait_until="networkidle")
            page.get_by_test_id("open-fp-cases").click()
            page.get_by_test_id("fp-open-case-" + case["id"]).click()
            expect(page.get_by_test_id("fp-evaluation-training")).to_be_visible()
            page.locator(".fp-ext-training-images input[type=checkbox]").first.check()
            page.get_by_label("訓練類別 item:" + item_id).select_option("crack")
            page.get_by_label("訓練覆核者").fill("Visible reviewer")
            page.get_by_label("完整標註依據").fill("The whole image was checked and all defect boxes were reviewed.")
            page.get_by_label("確認影像標註完整").check()

            with page.expect_download() as zip_download:
                with page.expect_response(
                    lambda response: response.url.endswith("/training-exports")
                    and response.request.method == "POST" and response.ok,
                ) as export_response:
                    page.get_by_test_id("fp-create-training-export").click()
            exported = export_response.value.json()["training_export"]
            assert exported["n_images"] == 1 and exported["n_boxes"] == 1
            archive_path = zip_download.value.path()
            assert archive_path
            with zipfile.ZipFile(archive_path) as archive:
                sha = case["sources"][0]["sha256"]
                label = archive.read("labels/" + sha + ".txt").decode("utf-8")
                assert label.startswith("0 0.400000000 0.450000000 0.400000000 0.500000000")
                assert "provenance.json" in archive.namelist()
            expect(page.get_by_test_id("fp-evaluation-training")).to_contain_text(
                exported["id"]
            )
            browser.close()
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)


def test_lv_web_05_retrieve_01_visible_explicit_search_and_candidate_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Visible React search keeps empty groups, evidence and pending candidate status."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    from playwright.sync_api import expect, sync_playwright

    cases_root = tmp_path / "fp-cases"
    monkeypatch.setenv("LV_CASES_DIR", str(cases_root))
    normal_folder = tmp_path / "normal-reference"
    normal_folder.mkdir()
    Image.new("RGB", (80, 60), (210, 25, 35)).save(normal_folder / "accepted-reflection.png")
    Image.new("RGB", (80, 60), (30, 70, 210)).save(normal_folder / "blue-background.png")
    (normal_folder / "unreadable.png").write_bytes(b"not an image")
    query_path = tmp_path / "reported.png"
    Image.new("RGB", (80, 60), (220, 20, 30)).save(query_path)
    reference_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*normal_folder.iterdir(), query_path]
    }
    model_calls: list[tuple[int, int]] = []

    def embed(image):
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        model_calls.append((image.width, image.height))
        return pixels.mean(axis=(0, 1)) + 1.0

    recipe = {"model": "fixture-mean", "checkpoint_sha256": "d" * 64, "target_res": 224}
    application = create_app(
        model_catalog=lambda: [],
        model_loader=lambda _model: None,
        case_search_model_provider=lambda: (recipe, embed),
    )
    server, server_thread, base_url = _start_api(application)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1480, "height": 1024})
            page.set_default_timeout(15000)
            search_responses: list[dict] = []

            def capture_search(response):
                if response.request.method == "POST" and "/searches" in response.url:
                    search_responses.append(response.json())

            page.on("response", capture_search)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("open-fp-cases").click()
            page.get_by_label("案件名稱（可留空）").fill("Visible retrieval report")
            page.get_by_label("建立案件圖片").set_input_files(str(query_path))
            page.get_by_role("button", name="建立案件", exact=True).click()
            expect(page.get_by_test_id("fp-case-title-readback")).to_have_text("Visible retrieval report")
            assert model_calls == [] and search_responses == [], "model retrieval must be explicit"

            page.get_by_label("預測類別／爭議名稱").fill("scratch")
            page.get_by_label("ROI x1").fill("0.2")
            page.get_by_label("ROI y1").fill("0.2")
            page.get_by_label("ROI x2").fill("0.8")
            page.get_by_label("ROI y2").fill("0.8")
            page.get_by_role("button", name="加入爭議區域").click()
            expect(page.get_by_test_id("fp-selected-region-roi")).to_be_visible()
            assert model_calls == [] and search_responses == [], "adding a region must not run retrieval"

            page.get_by_label("正常對照資料夾（可留空）").fill(str(normal_folder))
            page.get_by_label("包含已覆核歷史案件").uncheck()
            page.get_by_test_id("fp-run-search").click()
            expect(page.get_by_test_id("fp-search-summary")).to_be_visible()
            expect(page.get_by_test_id("fp-search-normal").get_by_role("heading", name="正常對照")).to_be_visible()
            expect(page.get_by_test_id("fp-search-defect").get_by_text("本次搜尋範圍內沒有可用對照。", exact=True)).to_be_visible()
            expect(page.get_by_text("cosine 相似度", exact=False).first).to_be_visible()
            expect(page.get_by_text("同一內容也出現在另一個標籤來源", exact=False)).to_have_count(0)
            expect(page.get_by_text("1 個檔案無法比對", exact=False)).to_be_visible()
            result_cards = page.locator(".fp-search-result-card")
            expect(result_cards).to_have_count(2)
            expect(result_cards.first.locator("img")).to_be_visible()
            expect(result_cards.first.get_by_role("button", name="加入待覆核")).to_be_visible()
            assert model_calls and len(search_responses) == 1
            assert str(tmp_path) not in json.dumps(search_responses[0], ensure_ascii=False)

            result_cards.first.get_by_role("button", name="加入待覆核").click()
            expect(page.get_by_test_id("fp-case-notice")).to_contain_text("尚未成為人工標籤")
            expect(result_cards.first.get_by_role("button", name="已加入待覆核")).to_be_disabled()
            browser.close()
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)

    assert reference_hashes == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*normal_folder.iterdir(), query_path]
    }, "case search modified its source files"


def test_lv_web_05_report_01_visible_cited_claims_reply_and_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Visible report flow cites evidence and keeps user-reply separate from cause."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    from playwright.sync_api import expect, sync_playwright

    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "fp-cases"))
    query_path = tmp_path / "reported.png"
    Image.new("RGB", (96, 72), (130, 140, 145)).save(query_path)
    application = create_app(model_catalog=lambda: [], model_loader=lambda _model: None)
    server, server_thread, base_url = _start_api(application)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1480, "height": 1024}, accept_downloads=True)
            page.set_default_timeout(15000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("open-fp-cases").click()
            page.get_by_label("案件名稱（可留空）").fill("Cited visible report")
            page.get_by_label("建立案件圖片").set_input_files(str(query_path))
            page.get_by_role("button", name="建立案件", exact=True).click()
            expect(page.get_by_test_id("fp-case-title-readback")).to_have_text("Cited visible report")
            report = page.get_by_test_id("fp-case-report")
            expect(report.get_by_test_id("fp-claims-empty")).to_be_visible()
            reference_select = report.get_by_label("引用案件證據")
            evidence_id = reference_select.locator("option").first.get_attribute("value")
            assert evidence_id

            report.get_by_label("說明類型").select_option("hypothesis")
            report.get_by_label("說明內容").fill("可能是反光造成誤逮；目前仍是待驗證假設。")
            report.get_by_test_id("fp-save-claim").click()
            claim = page.locator("[data-testid^='fp-claim-c_']").first
            expect(claim).to_contain_text("可能是反光造成誤逮")
            expect(report.get_by_text("根因狀態：假設待驗證", exact=True)).to_be_visible()
            claim_test_id = claim.get_attribute("data-testid")
            assert claim_test_id
            claim_id = claim_test_id.removeprefix("fp-claim-")

            claim.get_by_role("button", name="編輯此說明").click()
            report.get_by_label("說明內容").fill("資料已引用，但反光因果尚未確認。")
            report.get_by_test_id("fp-save-claim").click()
            expect(page.get_by_test_id(f"fp-claim-{claim_id}")).to_contain_text("反光因果尚未確認")

            report.get_by_label("使用者回覆狀態").select_option("replied")
            expect(report.get_by_label("使用者回覆狀態")).to_have_value("replied")
            expect(report.get_by_text("根因狀態：假設待驗證", exact=True)).to_be_visible()

            with page.expect_download() as markdown_download:
                report.get_by_test_id("fp-download-report-md").click()
            markdown_path = Path(markdown_download.value.path())
            markdown = markdown_path.read_text(encoding="utf-8")
            assert "反光因果尚未確認" in markdown
            assert evidence_id in markdown and claim_id in markdown
            assert "外部模型尚未重播" in markdown

            with page.expect_download() as json_download:
                report.get_by_test_id("fp-download-report-json").click()
            json_path = Path(json_download.value.path())
            snapshot = json.loads(json_path.read_text(encoding="utf-8"))
            assert snapshot["reply_status"] == "replied"
            assert snapshot["cause_status"] == "hypothesis"
            assert snapshot["claims"][0]["refs"] == [evidence_id]
            assert snapshot["claims"][0]["id"] == claim_id
            assert str(tmp_path) not in json.dumps(snapshot, ensure_ascii=False)
            assert any(row.get("claim_id") == claim_id for row in snapshot["evidence"])
            browser.close()
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)


def test_lv_web_05_al_01_visible_review_batch_zip_partial_return_and_manual_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Visible handoff exports one deduplicated image and reads partial/manual reviews back."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    from playwright.sync_api import expect, sync_playwright

    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "fp-cases"))
    monkeypatch.setenv("LV_OUTPUT_DIR", str(tmp_path / "lv-output"))
    image = BytesIO()
    Image.new("RGB", (96, 72), (120, 90, 55)).save(image, format="PNG")
    payload = image.getvalue()
    application = create_app(model_catalog=lambda: [], model_loader=lambda _model: None)
    store = application.state.fp_case_store
    reported = store.ingest(payload, "reported.png", "screenshot")
    case = store.create(reported, title="Visible review batch")
    reference = store.ingest(payload, "normal-reference.png", "reference")
    for index, roi in enumerate(([0.1, 0.1, 0.4, 0.4], [0.55, 0.5, 0.9, 0.9])):
        case["candidates"].append({
            "id": f"cand_visible_{index}",
            "source": reference,
            "roi": roi,
            "reason": "Retrieved visual neighbor",
            "reference_category": "normal",
            "evidence_id": case["evidence"][0]["id"],
            "similarity": 0.91 - index / 100,
            "label_status": "folder_label_unreviewed",
            "review": {"verdict": "pending", "reviewer": "", "reason": "", "at": ""},
        })
    store.save(case)
    server, server_thread, base_url = _start_api(application)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1480, "height": 1024}, accept_downloads=True)
            page.set_default_timeout(15000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("open-fp-cases").click()
            page.get_by_test_id(f"fp-open-case-{case['id']}").click()
            expect(page.get_by_test_id("fp-case-title-readback")).to_have_text("Visible review batch")
            batches = page.get_by_test_id("fp-review-batches")
            expect(batches.get_by_text("訓練就緒影像：0", exact=True)).to_be_visible()
            expect(batches.get_by_test_id("fp-candidate-roi-cand_visible_0")).to_be_visible()
            batches.get_by_test_id("fp-create-batch").click()
            expect(batches.get_by_test_id("fp-batch-select")).to_be_visible()
            expect(batches.get_by_text("唯一影像 1 張", exact=False)).to_be_visible()

            with page.expect_download() as zip_download:
                batches.get_by_test_id("fp-download-batch-zip").click()
            zip_path = Path(zip_download.value.path())
            with zipfile.ZipFile(zip_path) as archive:
                assert len([name for name in archive.namelist() if name.startswith("assets/")]) == 1
                returned_manifest = json.loads(archive.read("review_manifest.json"))
            returned_manifest["items"] = returned_manifest["items"][:1]
            returned_manifest["items"][0]["review"] = {
                "verdict": "normal", "reviewer": "Visible reviewer", "reason": "Checked against the approved reference.",
            }
            partial_path = tmp_path / "partial-review.json"
            partial_path.write_text(json.dumps(returned_manifest, ensure_ascii=False), encoding="utf-8")
            batches.get_by_label("回傳已覆核清單 JSON").set_input_files(str(partial_path))
            batches.get_by_test_id("fp-apply-review-return").click()
            expect(batches.get_by_test_id("fp-batch-notice")).to_contain_text("已讀回 1 個區域覆核")
            expect(batches.get_by_text("候選 2 個區域 · 已覆核 1 · 待覆核 1", exact=True)).to_be_visible()
            first = batches.get_by_test_id("fp-candidate-review-cand_visible_0")
            expect(first.get_by_label("覆核結果")).to_have_value("normal")
            expect(first.get_by_label("覆核者")).to_have_value("Visible reviewer")

            batches.get_by_label("回傳已覆核清單 JSON").set_input_files(str(partial_path))
            batches.get_by_test_id("fp-apply-review-return").click()
            expect(batches.get_by_test_id("fp-batch-error")).to_contain_text("版本已過期")

            second = batches.get_by_test_id("fp-candidate-review-cand_visible_1")
            second.get_by_label("覆核結果").select_option("defect")
            second.get_by_label("覆核者").fill("Visible reviewer")
            second.get_by_label("判斷依據").fill("The defect is visible in the candidate region.")
            second.get_by_role("button", name="儲存此候選覆核").click()
            expect(batches.get_by_text("候選 2 個區域 · 已覆核 2 · 待覆核 0", exact=True)).to_be_visible()
            expect(batches.get_by_text("訓練就緒影像：0", exact=True)).to_be_visible()

            persisted = store.load(case["id"])
            assert {row["id"]: row["review"]["verdict"] for row in persisted["candidates"]} == {
                "cand_visible_0": "normal", "cand_visible_1": "defect",
            }
            assert persisted["items"] == []
            assert persisted["improvement_status"] == "not_evaluated"
            browser.close()
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)


def test_lv_web_06_bank_01_visible_build_save_worker_restart_and_load(
    tmp_path: Path,
):
    """Visible bank journey persists the package and reloads it in a fresh worker PID."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    from playwright.sync_api import expect, sync_playwright

    cache_root = tmp_path / "persistent-cache"
    train_folder = tmp_path / "training-data"
    train_folder.mkdir()
    first_worker, first_url = _start_anomaly_worker(cache_root)
    first_pid = first_worker.pid
    model_id = ""
    api_bodies: list[str] = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(15000)
            page.on("response", lambda response: api_bodies.append(response.text())
                    if "/api/v1/anomaly/" in response.url and response.ok else None)
            page.goto(first_url, wait_until="networkidle")
            page.get_by_test_id("open-anomaly-bank").click()
            expect(page.get_by_test_id("anomaly-model")).to_be_visible()
            page.get_by_test_id("anomaly-train-folders").fill(str(train_folder))
            page.get_by_test_id("anomaly-model").select_option("fixture-model")
            page.get_by_test_id("anomaly-build").click()
            expect(page.get_by_test_id("anomaly-model-status")).to_contain_text("模型已建立", timeout=15000)
            expect(page.get_by_test_id("anomaly-model-status")).to_contain_text("5 個物件")
            model_id = page.get_by_test_id("anomaly-model-status").inner_text().split("模型 ID ")[-1].strip()
            page.get_by_test_id("anomaly-save").click()
            expect(page.get_by_test_id("anomaly-notice")).to_contain_text("已保存至 LV_CACHE_DIR")
            expect(page.get_by_test_id("anomaly-model-status")).to_contain_text("模型已保存")
            model_dir = cache_root / "react_anomaly_models" / model_id
            assert {"bank.npz", "projection.npz", "fewshot.json", "meta.json", "manifest.json"} <= {
                path.name for path in model_dir.iterdir()
            }
            page.close()
            browser.close()
    finally:
        _stop_worker(first_worker)

    second_worker, second_url = _start_anomaly_worker(cache_root)
    try:
        assert second_worker.pid != first_pid
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(15000)
            page.on("response", lambda response: api_bodies.append(response.text())
                    if "/api/v1/anomaly/" in response.url and response.ok else None)
            page.goto(second_url, wait_until="networkidle")
            page.get_by_test_id("open-anomaly-bank").click()
            page.get_by_test_id("anomaly-saved-model").select_option(model_id)
            page.get_by_test_id("anomaly-load").click()
            expect(page.get_by_test_id("anomaly-model-status")).to_contain_text("模型已載入")
            expect(page.get_by_test_id("anomaly-model-status")).to_contain_text(model_id)
            assert all(str(tmp_path) not in body for body in api_bodies)
            browser.close()
    finally:
        _stop_worker(second_worker)


def _start_streamlit_for_react_parity(
    models_dir: Path, run_root: Path, *, cache_dir: Path | None = None,
):
    """Launch the visible legacy UI with isolated app state and an optional shared cache."""
    from .conftest import _process_group_options

    log_path = run_root / "streamlit.log"
    log_file = log_path.open("w", encoding="utf-8")
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "LV_MODELS_DIR": str(models_dir),
        "LV_CACHE_DIR": str(cache_dir or run_root / "cache"),
        "LV_CASES_DIR": str(run_root / "cases"),
        "LV_UI_STATE": str(run_root / "no-persist" / "ui_state.json"),
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_SERVER_HEADLESS": "true",
    }
    process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "scripts/app.py",
            "--server.port", str(port), "--server.headless", "true",
            "--server.fileWatcherType", "none",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        **_process_group_options(),
    )
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_file.close()
            pytest.fail("Streamlit reference exited early:\n" + log_path.read_text(encoding="utf-8")[-3000:])
        try:
            with urllib.request.urlopen(f"{base_url}/_stcore/health", timeout=1) as response:
                if response.read().decode().strip() == "ok":
                    return process, log_file, base_url
        except OSError:
            time.sleep(0.2)
    from .conftest import _stop_process_tree

    _stop_process_tree(process)
    log_file.close()
    pytest.fail("Streamlit reference did not become healthy:\n" + log_path.read_text(encoding="utf-8")[-3000:])


def test_lv_web_06_bank_01_real_dinov2_normal_bank_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, yolo_defect_at_nmin,
):
    """Compare visible real-model scoring, filtering, and prelabel behavior across both UIs."""
    models_dir = _pinned_dinov2_models_dir(tmp_path)
    weights = models_dir / "dinov2_vits14" / "dinov2_vits14.pth"
    if not weights.is_file():
        pytest.skip(f"real DINOv2-small weights are unavailable: {weights}")
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == DINOV2_VITS14_SHA256, \
        "the local model weights differ from the LV-WEB-06-BANK-01 baseline"
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    from playwright.sync_api import expect, sync_playwright
    import _utils

    monkeypatch.setattr(_utils, "_DEFAULT_MODELS_DIR", models_dir)
    monkeypatch.setattr(_utils.available_models, "__defaults__", (models_dir,))
    monkeypatch.setattr(
        _utils.load_model,
        "__defaults__",
        (models_dir, *_utils.load_model.__defaults__[1:]),
    )

    monkeypatch.setenv("LV_MODELS_DIR", str(models_dir))
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "cases"))
    root = yolo_defect_at_nmin["root"]
    # Keep the training fixture untouched; per-image target stripes create a
    # non-degenerate score range for the real-model filter parity case.
    target_root = tmp_path / "score-target"
    shutil.copytree(root, target_root)
    from PIL import ImageDraw

    for index, image_path in enumerate(sorted((target_root / "images").glob("*.jpg"))):
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        draw = ImageDraw.Draw(image)
        stripe_x = 78 + (index % 5) * 8
        stripe_width = 4 + (index % 6) * 3
        level = (index * 47 + 19) % 256
        draw.rectangle(
            (stripe_x, 0, min(127, stripe_x + stripe_width), 127),
            fill=(level, 255 - level, (level * 3) % 256),
        )
        image.save(image_path, quality=92)

    source_hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }
    target_hashes = {
        path.relative_to(target_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in target_root.rglob("*") if path.is_file()
    }
    application = create_app(
        model_catalog=lambda: available_models(models_dir),
        model_loader=lambda model_id: load_model(model_id, models_dir=models_dir),
    )
    server, server_thread, base_url = _start_api(application)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(300000)
            job_payloads: list[dict] = []
            score_payloads: list[dict] = []
            prelabel_payloads: list[dict] = []

            def capture_job(response):
                if response.request.method == "GET" and "/api/v1/anomaly/jobs/" in response.url and response.ok:
                    job_payloads.append(response.json())

            def capture_score_job(response):
                if response.request.method == "GET" and "/api/v1/anomaly/score-jobs/" in response.url and response.ok:
                    score_payloads.append(response.json())

            def capture_prelabel_preview(response):
                if "/api/v1/anomaly/runs/" in response.url and "/prelabel/preview" in response.url and response.ok:
                    prelabel_payloads.append(response.json())

            page.on("response", capture_job)
            page.on("response", capture_score_job)
            page.on("response", capture_prelabel_preview)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("open-anomaly-bank").click()
            page.get_by_test_id("anomaly-train-folders").fill(str(root))
            page.get_by_test_id("anomaly-model").select_option("dinov2_vits14")
            page.get_by_test_id("anomaly-build").click()
            expect(page.get_by_test_id("anomaly-model-status")).to_contain_text("模型已建立", timeout=300000)
            expect(page.get_by_test_id("anomaly-model-status")).to_contain_text("16 個物件")
            expect(page.get_by_test_id("anomaly-model-status")).to_contain_text("embedding D=384")
            expect(page.get_by_test_id("anomaly-model-status")).to_contain_text("只做異常偵測")
            assert any(
                payload.get("state") == "completed"
                and payload.get("result", {}).get("meta", {}).get("patch_dim") == 384
                and payload.get("result", {}).get("summary", {}).get("has_head") is False
                for payload in job_payloads
            )
            page.get_by_test_id("anomaly-label-semantic").select_option("defect")
            page.get_by_test_id("anomaly-build").click()
            expect(page.get_by_test_id("anomaly-model-status")).to_contain_text("能分辨瑕疵種類", timeout=300000)
            assert any(
                payload.get("state") == "completed"
                and payload.get("result", {}).get("summary", {}).get("has_head") is True
                and payload.get("result", {}).get("summary", {}).get("eligible_classes") == ["scratch", "stain"]
                for payload in job_payloads
            )
            page.get_by_test_id("anomaly-save").click()
            expect(page.get_by_test_id("anomaly-notice")).to_contain_text(
                "已保存至 LV_CACHE_DIR", timeout=30000,
            )
            bundle_id = application.state.current_anomaly_model_id
            react_model_dir = tmp_path / "cache" / "react_anomaly_models" / bundle_id

            page.get_by_test_id("anomaly-target-folders").fill(str(target_root))
            page.get_by_test_id("anomaly-score").click()
            expect(page.get_by_test_id("anomaly-run-summary")).to_contain_text("16", timeout=300000)
            expect(page.get_by_test_id("anomaly-review-workspace")).to_be_visible()
            expect(page.locator(".anomaly-object").first).to_be_visible()
            assert any(
                payload.get("state") == "completed"
                and payload.get("result", {}).get("n_objects") == 16
                and payload.get("result", {}).get("score_mode") == "patch"
                for payload in score_payloads
            )
            run_id = application.state.current_anomaly_run_id
            run = application.state.anomaly_runs[run_id]
            expect(page.locator(".anomaly-object")).to_have_count(len(run["records"]))
            react_score_cards = page.locator(".anomaly-object").all()
            react_rank = [
                int(card.get_attribute("data-testid").rsplit("-", 1)[-1])
                for card in react_score_cards
            ]
            assert react_rank == [int(index) for index in run["ranking"]]
            for index, card in zip(react_rank, react_score_cards):
                record = run["records"][index]
                text = card.inner_text()
                score_match = re.search(r"異常分數\s+([0-9.]+)", text)
                assert score_match, text
                assert float(score_match.group(1)) == pytest.approx(
                    float(run["scores"][index]), abs=5.1e-7,
                )
                assert Path(record["image_path"]).name in text
                assert str(record["label"]) in text
                assert ("異常候選" if run["labels"][index] == "bad" else "正常候選") in text

            def expected_inspector_options(indices: list[int]) -> list[str]:
                return [
                    f"{float(run['scores'][index]):.3f} · "
                    f"{Path(run['records'][index]['image_path']).name} "
                    f"#{int(run['records'][index].get('obj_index') or 0)} · "
                    f"{run['records'][index]['label']}"
                    for index in indices
                ]

            def read_streamlit_inspector_options(expected_count: int) -> list[str]:
                selector = legacy_page.locator(
                    '.st-key-anomaly_inspect [data-baseweb="select"]:visible',
                ).first
                for _ in range(3):
                    selector.click()
                    try:
                        legacy_page.wait_for_selector('[role="option"]', timeout=6000)
                        break
                    except Exception:
                        legacy_page.keyboard.press("Escape")
                        wait_idle(legacy_page)
                else:
                    raise AssertionError("Streamlit inspector options did not open")

                seen: list[str] = []
                for _ in range(12):
                    for value in legacy_page.get_by_role("option").all_inner_texts():
                        normalized = " ".join(value.split())
                        if normalized and normalized not in seen:
                            seen.append(normalized)
                    if len(seen) >= expected_count:
                        break
                    moved = legacy_page.evaluate("""() => {
                        const option = document.querySelector('[role="option"]');
                        const scrollable = [];
                        for (let node = option?.parentElement; node && node !== document.body;
                             node = node.parentElement) {
                            const style = getComputedStyle(node);
                            if (node.scrollHeight > node.clientHeight + 1
                                && /auto|scroll/.test(style.overflowY)) {
                                scrollable.push(node);
                            }
                        }
                        const list = scrollable[0];
                        if (!list) return false;
                        const before = list.scrollTop;
                        list.scrollTop = Math.min(
                            list.scrollHeight, before + Math.max(220, list.clientHeight),
                        );
                        return list.scrollTop !== before;
                    }""")
                    legacy_page.wait_for_timeout(220)
                    if not moved:
                        before_count = len(seen)
                        legacy_page.get_by_role("option").last.hover()
                        legacy_page.mouse.wheel(0, 320)
                        legacy_page.wait_for_timeout(220)
                        for value in legacy_page.get_by_role("option").all_inner_texts():
                            normalized = " ".join(value.split())
                            if normalized and normalized not in seen:
                                seen.append(normalized)
                        if len(seen) == before_count:
                            break
                legacy_page.keyboard.press("Escape")
                wait_idle(legacy_page)
                return seen

            def streamlit_range_sliders():
                legacy_page.wait_for_function("""() => Array.from(document.querySelectorAll(
                    '.st-key-anomaly_heat_filter_apply',
                )).filter(element => element.offsetParent !== null).length === 1""")
                widget = legacy_page.locator(
                    ".st-key-anomaly_heat_filter_apply:visible",
                ).first
                sliders = widget.locator('[role="slider"]:visible')
                assert sliders.count() == 2, {
                    "visible_widgets": legacy_page.locator(
                        ".st-key-anomaly_heat_filter_apply:visible",
                    ).count(),
                    "visible_sliders": sliders.count(),
                }
                return sliders

            # Exercise real-model prelabel in the React UI and compare every
            # visible row and exported YOLO line with the existing prelabel core.
            def assert_react_class_filter(label: str) -> None:
                page.get_by_label("瑕疵類別篩選").select_option(label)
                page.get_by_role("button", name="套用篩選").click()
                expected_indices = [
                    int(index) for index in run["ranking"]
                    if run["records"][int(index)]["label"] == label
                ]
                expect(page.locator(".anomaly-object")).to_have_count(len(expected_indices))
                actual_indices = [
                    int(card.get_attribute("data-testid").rsplit("-", 1)[-1])
                    for card in page.locator(".anomaly-object").all()
                ]
                assert actual_indices == expected_indices
                assert all(run["records"][index]["label"] == label for index in actual_indices)

            def assert_react_score_range(score_min: float, score_max: float) -> list[int]:
                page.get_by_label("瑕疵類別篩選").select_option("")
                page.get_by_label("瑕疵最低分數").fill(str(score_min))
                page.get_by_label("瑕疵最高分數").fill(str(score_max))
                page.get_by_role("button", name="套用篩選").click()
                expected_indices = [
                    int(index) for index in run["ranking"]
                    if score_min <= float(run["scores"][int(index)]) <= score_max
                ]
                expect(page.locator(".anomaly-object")).to_have_count(len(expected_indices))
                actual_indices = [
                    int(card.get_attribute("data-testid").rsplit("-", 1)[-1])
                    for card in page.locator(".anomaly-object").all()
                ]
                assert actual_indices == expected_indices
                expect(page.get_by_test_id("anomaly-review-workspace")).to_contain_text(
                    f"篩選結果 {len(expected_indices)} 筆",
                )
                return expected_indices

            def clear_react_score_filters() -> None:
                page.get_by_label("瑕疵類別篩選").select_option("")
                page.get_by_label("瑕疵最低分數").fill("")
                page.get_by_label("瑕疵最高分數").fill("")
                page.get_by_role("button", name="套用篩選").click()
                expect(page.locator(".anomaly-object")).to_have_count(len(run["records"]))

            assert_react_class_filter("scratch")
            clear_react_score_filters()

            page.get_by_test_id("anomaly-prelabel-preview").click()
            expect(page.get_by_test_id("anomaly-prelabel-summary")).to_be_visible(timeout=30000)
            expect(page.get_by_test_id("anomaly-prelabel-item-0")).to_be_visible()
            page.wait_for_function(
                "document.querySelector('[data-testid=\"anomaly-prelabel-item-0\"] img')?.naturalWidth > 0",
                timeout=30000,
            )
            assert prelabel_payloads, "the visible preview should request the local prelabel API"
            preview = prelabel_payloads[-1]
            bundle = application.state.anomaly_models[bundle_id]
            import prelabel as legacy_prelabel

            expected_decisions = legacy_prelabel.prelabel_records(
                run["records"], run["obj_emb"], run["scores"], bundle["head"],
                anomaly_threshold=run["threshold"], min_conf=preview["min_conf"],
                target_classes=preview["target_classes"],
            )
            assert preview["total"] == len(expected_decisions) == 16
            assert preview["accepted"] == sum(item["accept"] for item in expected_decisions)
            assert len(preview["items"]) == len(expected_decisions)
            for row, decision in zip(preview["items"], expected_decisions):
                index = decision["item"]
                record = run["records"][index]
                assert row["index"] == index
                assert row["filename"] == Path(record["image_path"]).name
                assert row["label"] == record["label"]
                assert row["obj_index"] == record["obj_index"]
                assert row["bbox"] == pytest.approx(record["bbox"], abs=1e-7)
                assert row["score"] == pytest.approx(run["scores"][index], abs=1e-7)
                assert row["pred"] == decision["pred"]
                assert row["confidence"] == pytest.approx(decision["conf"], abs=1e-7)
                assert row["accept"] is decision["accept"]
                item = page.get_by_test_id(f"anomaly-prelabel-item-{index}")
                expect(item).to_contain_text(f"預測 {decision['pred']}")
                expect(item).to_contain_text("接受" if decision["accept"] else "不接受")

            output = tmp_path / "real-dino-prelabel-output"
            page.get_by_test_id("anomaly-prelabel-output-dir").fill(str(output))
            page.get_by_test_id("anomaly-prelabel-export").click()
            expect(page.get_by_test_id("anomaly-notice")).to_contain_text("預標匯出完成", timeout=30000)
            expected_lines = legacy_prelabel.to_yolo_lines(
                run["records"], expected_decisions,
                class_names=bundle["head"]["classes"],
            )
            expected_files = {
                f"{Path(image_path).stem}.txt": "\n".join(lines) + "\n"
                for image_path, lines in expected_lines.items()
            }
            actual_files = {
                path.name: path.read_text(encoding="utf-8")
                for path in (output / "labels").glob("*.txt")
            }
            assert actual_files == expected_files
            assert (output / "classes.txt").read_text(encoding="utf-8").splitlines() == bundle["head"]["classes"]

            def file_tree_bytes(folder: Path) -> dict[str, bytes]:
                return {
                    path.relative_to(folder).as_posix(): path.read_bytes()
                    for path in folder.rglob("*") if path.is_file()
                }

            class_names = list(bundle["head"]["classes"])
            assert set(class_names) == {"scratch", "stain"}
            react_prelabel_cases = {
                "all_050": {
                    "files": file_tree_bytes(output),
                    "accepted": sorted(
                        (item["pred"], f"{item['confidence']:.2f}")
                        for item in preview["items"] if item["accept"]
                    ),
                    "accepted_count": preview["accepted"],
                }
            }

            def set_range_value(slider, desired: float) -> None:
                current = float(slider.input_value())
                steps = round((desired - current) / 0.05)
                key = "ArrowRight" if steps > 0 else "ArrowLeft"
                for _ in range(abs(steps)):
                    slider.press(key)
                assert float(slider.input_value()) == pytest.approx(desired, abs=1e-5)

            def run_react_prelabel_case(
                case_id: str, min_conf: float, target_classes: list[str],
            ) -> None:
                set_range_value(page.get_by_test_id("anomaly-prelabel-min-conf"), min_conf)
                for label in class_names:
                    page.get_by_role("checkbox", name=f"預標類別 {label}").set_checked(
                        label in target_classes,
                    )
                with page.expect_response(
                    lambda response: "/prelabel/preview" in response.url
                    and response.request.method == "POST"
                    and response.ok,
                    timeout=30000,
                ) as response_info:
                    page.get_by_test_id("anomaly-prelabel-preview").click()
                expect(page.get_by_test_id("anomaly-prelabel-summary")).to_be_visible(timeout=30000)
                case_preview = response_info.value.json()
                assert case_preview["min_conf"] == pytest.approx(min_conf)
                assert set(case_preview["target_classes"]) == set(target_classes)
                case_decisions = legacy_prelabel.prelabel_records(
                    run["records"], run["obj_emb"], run["scores"], bundle["head"],
                    anomaly_threshold=run["threshold"], min_conf=min_conf,
                    target_classes=target_classes,
                )
                assert case_preview["total"] == len(case_decisions) == 16
                assert case_preview["accepted"] == sum(item["accept"] for item in case_decisions)
                for row, decision in zip(case_preview["items"], case_decisions):
                    assert row["index"] == decision["item"]
                    assert row["pred"] == decision["pred"]
                    assert row["confidence"] == pytest.approx(decision["conf"], abs=1e-7)
                    assert row["accept"] is decision["accept"]
                    item = page.get_by_test_id(f"anomaly-prelabel-item-{decision['item']}")
                    expect(item).to_contain_text(f"預測 {decision['pred']}")
                    expect(item).to_contain_text("接受" if decision["accept"] else "不接受")

                case_output = tmp_path / f"react-prelabel-{case_id}"
                page.get_by_test_id("anomaly-prelabel-output-dir").fill(str(case_output))
                page.get_by_test_id("anomaly-prelabel-export").click()
                expect(page.get_by_test_id("anomaly-notice")).to_contain_text("預標匯出完成", timeout=30000)
                expected_case_lines = legacy_prelabel.to_yolo_lines(
                    run["records"], case_decisions, class_names=class_names,
                )
                expected_case_files = {
                    f"{Path(image_path).stem}.txt": "\n".join(lines) + "\n"
                    for image_path, lines in expected_case_lines.items()
                }
                case_files = file_tree_bytes(case_output)
                actual_case_labels = {
                    Path(name).name: (case_output / name).read_text(encoding="utf-8")
                    for name, content in case_files.items() if name.startswith("labels/")
                }
                assert actual_case_labels == expected_case_files
                assert case_files["classes.txt"].decode("utf-8").splitlines() == class_names
                react_prelabel_cases[case_id] = {
                    "files": case_files,
                    "accepted": sorted(
                        (item["pred"], f"{item['confidence']:.2f}")
                        for item in case_preview["items"] if item["accept"]
                    ),
                    "accepted_count": case_preview["accepted"],
                }

            # Cover low/default/high confidence against all, each individual,
            # and no selected target classes through the visible React controls.
            prelabel_cases = [
                ("all_000", 0.00, class_names),
                ("all_095", 0.95, class_names),
                ("scratch_000", 0.00, ["scratch"]),
                ("scratch_050", 0.50, ["scratch"]),
                ("scratch_095", 0.95, ["scratch"]),
                ("stain_000", 0.00, ["stain"]),
                ("stain_050", 0.50, ["stain"]),
                ("stain_095", 0.95, ["stain"]),
                ("none_000", 0.00, []),
                ("none_050", 0.50, []),
                ("none_095", 0.95, []),
            ]
            for case_id, min_conf, target_classes in prelabel_cases:
                run_react_prelabel_case(case_id, min_conf, target_classes)

            original_bank_sha = page.get_by_test_id("anomaly-run-summary").inner_text().split("bank SHA-256 ")[-1].split("…")[0]
            page.locator(".anomaly-object").first.get_by_role("button", name="顯示 Patch 熱圖").click()
            heatmap = page.locator(".anomaly-object-heatmap").first
            expect(heatmap).to_be_visible(timeout=300000)
            page.wait_for_function("document.querySelector('.anomaly-object-heatmap')?.naturalWidth > 0", timeout=300000)
            page.locator(".anomaly-object").first.locator("select").select_option("good")
            page.get_by_test_id("anomaly-reapply").click()
            expect(page.get_by_test_id("anomaly-run-summary")).to_contain_text("revision 2")
            updated_bank_sha = page.get_by_test_id("anomaly-run-summary").inner_text().split("bank SHA-256 ")[-1].split("…")[0]
            assert updated_bank_sha == original_bank_sha, "reclassifying target scores must not rebuild or replace the Normal Bank"
            after_hashes = {
                path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*") if path.is_file()
            }
            assert after_hashes == source_hashes, "anomaly model build must leave the selected dataset unchanged"
            after_target_hashes = {
                path.relative_to(target_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in target_root.rglob("*") if path.is_file()
            }
            assert after_target_hashes == target_hashes, "anomaly scoring and export must leave target images unchanged"

            # Re-run the same 16-object, same-weight workflow through the
            # legacy Streamlit UI. Sharing feature caches keeps both cold-start
            # model builds on the same canonical cached vectors before comparing
            # the visible decisions and exported files.
            from ._anomaly_wizard import (
                TAB_APPLY, apply_model, build_model, click_tab, enter_anomaly, set_model_dir,
            )
            from .conftest import _stop_process_tree, wait_idle
            from .test_prelabel_gui_e2e import _accept_count, _open_prelabel

            parity_root = tmp_path / "streamlit-parity"
            parity_root.mkdir()
            legacy_process, legacy_log, legacy_url = _start_streamlit_for_react_parity(
                models_dir, parity_root, cache_dir=tmp_path / "cache",
            )
            legacy_page = browser.new_page(viewport={"width": 1920, "height": 1080})
            try:
                legacy_page.set_default_timeout(300000)
                enter_anomaly(legacy_page, legacy_url)
                message = build_model(legacy_page, root, semantic_text="瑕疵類別")
                assert "能分辨瑕疵種類" in message
                legacy_model_dir = tmp_path / "cache" / "streamlit-parity-model"
                set_model_dir(legacy_page, legacy_model_dir)
                legacy_page.locator('.st-key-anomaly_save_model_btn button').click()
                wait_idle(legacy_page)
                expect(legacy_page.locator('[data-testid="stMain"]')).to_contain_text("已存:")

                # Independent builds must persist equivalent model state. The
                # shared feature cache makes the expected bank and projector
                # vectors exact on this same-input run.
                from anomaly_bank_store import load_bank
                from dino_head import load_head

                legacy_bundle = load_bank(legacy_model_dir)
                react_bundle = load_bank(react_model_dir)
                for key in (
                    "model", "target_res", "score_mode", "object_source",
                    "patch_dim", "obj_dim", "n_objects", "embedding_recipe",
                    "label_semantic",
                ):
                    assert legacy_bundle["meta"].get(key) == react_bundle["meta"].get(key), key
                np.testing.assert_array_equal(legacy_bundle["vectors"], react_bundle["vectors"])
                for key in ("mean", "components", "good_obj_emb", "ref_coords"):
                    np.testing.assert_array_equal(
                        legacy_bundle["projection"][key], react_bundle["projection"][key],
                    )
                assert legacy_bundle["projection"]["l2norm"] == react_bundle["projection"]["l2norm"]
                assert legacy_bundle["projection"]["tau"] == react_bundle["projection"]["tau"]
                legacy_head = load_head(legacy_model_dir / "head.joblib")
                react_head = load_head(react_model_dir / "head.joblib")
                np.testing.assert_array_equal(legacy_head["classes"], react_head["classes"])
                np.testing.assert_array_equal(
                    legacy_head["model"].coef_, react_head["model"].coef_,
                )
                np.testing.assert_array_equal(
                    legacy_head["model"].intercept_, react_head["model"].intercept_,
                )

                # Exercise the legacy loader against the React-saved model
                # package, then compare both visible apply/preview/export runs.
                set_model_dir(legacy_page, react_model_dir)
                legacy_page.locator('.st-key-anomaly_load_model_btn button').click()
                wait_idle(legacy_page)
                expect(legacy_page.locator('[data-testid="stMain"]')).to_contain_text(
                    "模型已載入",
                )
                apply_model(legacy_page, target_root)
                legacy_main = legacy_page.locator('[data-testid="stMain"]').inner_text()
                legacy_bad_match = re.search(r"判為可疑\(bad\)\s*(\d+)\s*個", legacy_main)
                assert legacy_bad_match, legacy_main[-2000:]
                assert int(legacy_bad_match.group(1)) == int(run["n_bad"])
                assert read_streamlit_inspector_options(len(run["records"])) == expected_inspector_options(
                    [int(index) for index in run["ranking"]],
                )

                range_thumbs = streamlit_range_sliders()
                initial_min = float(range_thumbs.nth(0).get_attribute("aria-valuenow"))
                range_thumbs.nth(0).focus()
                for _ in range(1):
                    legacy_page.keyboard.press("PageUp")
                wait_idle(legacy_page)
                click_tab(legacy_page, TAB_APPLY)
                range_thumbs = streamlit_range_sliders()
                filtered_min = float(range_thumbs.nth(0).get_attribute("aria-valuenow"))
                filtered_max = float(range_thumbs.nth(1).get_attribute("aria-valuenow"))
                assert filtered_min > initial_min
                score_indices = [
                    int(index) for index in run["ranking"]
                    if filtered_min <= float(run["scores"][int(index)]) <= filtered_max
                ]
                assert 0 < len(score_indices) < len(run["records"]), {
                    "initial_min": initial_min,
                    "filtered_min": filtered_min,
                    "filtered_max": filtered_max,
                    "scores": sorted(float(value) for value in run["scores"]),
                }
                assert assert_react_score_range(filtered_min, filtered_max) == score_indices
                assert read_streamlit_inspector_options(len(score_indices)) == expected_inspector_options(
                    score_indices,
                )

                range_thumbs.nth(0).focus()
                for _ in range(1):
                    legacy_page.keyboard.press("PageDown")
                wait_idle(legacy_page)
                click_tab(legacy_page, TAB_APPLY)
                restored_min = float(
                    streamlit_range_sliders().nth(0).get_attribute("aria-valuenow")
                )
                assert restored_min == pytest.approx(initial_min, abs=1e-5)
                clear_react_score_filters()

                class_filter = legacy_page.locator(".st-key-anomaly_class_filter_apply")
                class_filter.get_by_role("combobox").first.click()
                legacy_page.get_by_role("option", name="scratch", exact=True).click()
                wait_idle(legacy_page)
                click_tab(legacy_page, TAB_APPLY)
                scratch_indices = [
                    int(index) for index in run["ranking"]
                    if run["records"][int(index)]["label"] == "scratch"
                ]
                assert len(scratch_indices) == 8
                assert read_streamlit_inspector_options(len(scratch_indices)) == expected_inspector_options(
                    scratch_indices,
                )

                _open_prelabel(legacy_page)
                legacy_preview_text = legacy_page.locator('.st-key-anomaly_prelabel_preview').inner_text()
                legacy_accepted = _accept_count(legacy_page)
                legacy_captions = re.findall(
                    r"([^\n]+?)\s*·\s*信心\s*([0-9.]+)\s*·\s*✅收",
                    legacy_preview_text,
                )
                react_captions = sorted(react_prelabel_cases["all_050"]["accepted"][:12])
                assert legacy_accepted == preview["accepted"]

                legacy_output = parity_root / "prelabel-output"
                legacy_page.locator('.st-key-anomaly_prelabel_out_dir input').first.fill(str(legacy_output))
                legacy_page.locator('.st-key-anomaly_prelabel_out_dir input').first.press("Enter")
                wait_idle(legacy_page)
                _open_prelabel(legacy_page)
                legacy_page.locator('.st-key-anomaly_prelabel_export_btn button').click()
                legacy_page.wait_for_function(
                    "() => (document.querySelector('[data-testid=\"stMain\"]')?.innerText || '').includes('預標匯出完成')",
                    timeout=60000,
                )
                wait_idle(legacy_page)

                legacy_files = file_tree_bytes(legacy_output)
                react_files = file_tree_bytes(output)
                assert legacy_files == react_files, json.dumps({
                    name: {
                        "streamlit": legacy_files.get(name, b"").decode("utf-8", errors="replace"),
                        "react": react_files.get(name, b"").decode("utf-8", errors="replace"),
                    }
                    for name in sorted(set(legacy_files) | set(react_files))
                    if legacy_files.get(name) != react_files.get(name)
                }, ensure_ascii=False)
                assert sorted(legacy_captions) == sorted(react_captions), json.dumps({
                    "streamlit": sorted(legacy_captions),
                    "react": sorted(react_captions),
                }, ensure_ascii=False)

                def selected_legacy_classes() -> set[str]:
                    widget = legacy_page.locator(".st-key-anomaly_prelabel_classes")
                    tag_texts = widget.locator('[data-baseweb="tag"]').all_text_contents()
                    return {
                        label for label in class_names
                        if any(label in text for text in tag_texts)
                    }

                def set_legacy_classes(target_classes: list[str]) -> None:
                    target = set(target_classes)
                    current = selected_legacy_classes()
                    while current:
                        widget = legacy_page.locator(".st-key-anomaly_prelabel_classes")
                        combo = widget.get_by_role("combobox").first
                        combo.click()
                        combo.press("Backspace")
                        legacy_page.wait_for_function(
                            "expected => document.querySelectorAll("
                            "'.st-key-anomaly_prelabel_classes [data-baseweb=\"tag\"]'"
                            ").length < expected",
                            arg=len(current),
                            timeout=10000,
                        )
                        wait_idle(legacy_page)
                        _open_prelabel(legacy_page)
                        updated = selected_legacy_classes()
                        assert len(updated) < len(current), {
                            "before": sorted(current),
                            "after": sorted(updated),
                        }
                        current = updated
                    for label in sorted(target):
                        widget = legacy_page.locator(".st-key-anomaly_prelabel_classes")
                        expected_count = len(selected_legacy_classes()) + 1
                        widget.locator('[role="combobox"]').first.click()
                        legacy_page.get_by_role("option", name=label, exact=True).click()
                        legacy_page.wait_for_function(
                            "expected => document.querySelectorAll("
                            "'.st-key-anomaly_prelabel_classes [data-baseweb=\"tag\"]'"
                            ").length === expected",
                            arg=expected_count,
                            timeout=10000,
                        )
                        wait_idle(legacy_page)
                        _open_prelabel(legacy_page)
                    assert selected_legacy_classes() == target, {
                        "expected": sorted(target),
                        "actual": sorted(selected_legacy_classes()),
                    }

                def run_legacy_prelabel_case(
                    case_id: str, min_conf: float, target_classes: list[str],
                ) -> None:
                    _open_prelabel(legacy_page)
                    # Streamlit renders sliders as role=slider elements, not
                    # native input elements. Keep the lookup scoped to the
                    # keyed widget so other sliders on the page cannot match.
                    slider = legacy_page.locator(
                        ".st-key-anomaly_prelabel_min_conf",
                    ).get_by_role("slider")
                    assert slider.count() == 1, f"{case_id}: expected one min-confidence slider"
                    current = float(slider.get_attribute("aria-valuenow"))
                    steps = round((min_conf - current) / 0.05)
                    key = "ArrowRight" if steps > 0 else "ArrowLeft"
                    for _ in range(abs(steps)):
                        slider.press(key)
                    wait_idle(legacy_page)
                    _open_prelabel(legacy_page)
                    assert float(slider.get_attribute("aria-valuenow")) == pytest.approx(
                        min_conf, abs=1e-5,
                    )
                    set_legacy_classes(target_classes)
                    _open_prelabel(legacy_page)

                    react_case = react_prelabel_cases[case_id]
                    legacy_page.wait_for_function(
                        """expected => {
                            const preview = document.querySelector(
                                '.st-key-anomaly_prelabel_preview',
                            );
                            const text = preview?.innerText || '';
                            const count = text.match(/接受\\s*(\\d+)/);
                            const captions = [...text.matchAll(
                                /([^\\n]+?)\\s*·\\s*信心\\s*([0-9.]+)\\s*·\\s*✅收/g,
                            )].map(match => [match[1].trim(), match[2]])
                                .sort((a, b) => (a[0] + '\\0' + a[1]).localeCompare(b[0] + '\\0' + b[1]));
                            return count
                                && Number(count[1]) === expected.accepted_count
                                && JSON.stringify(captions) === JSON.stringify(expected.accepted);
                        }""",
                        arg={
                            "accepted_count": react_case["accepted_count"],
                            "accepted": [
                                list(item) for item in sorted(react_case["accepted"][:12])
                            ],
                        },
                        timeout=30000,
                    )

                    case_preview_text = legacy_page.locator(
                        ".st-key-anomaly_prelabel_preview",
                    ).inner_text()
                    case_accepted = _accept_count(legacy_page)
                    case_captions = sorted(re.findall(
                        r"([^\n]+?)\s*·\s*信心\s*([0-9.]+)\s*·\s*✅收",
                        case_preview_text,
                    ))
                    assert case_accepted == react_case["accepted_count"]
                    assert case_captions == sorted(react_case["accepted"][:12]), json.dumps({
                        "case": case_id,
                        "streamlit": case_captions,
                        "react": sorted(react_case["accepted"][:12]),
                    }, ensure_ascii=False)

                    case_output = parity_root / f"prelabel-output-{case_id}"
                    legacy_page.locator('.st-key-anomaly_prelabel_out_dir input').first.fill(
                        str(case_output),
                    )
                    legacy_page.locator('.st-key-anomaly_prelabel_out_dir input').first.press("Enter")
                    wait_idle(legacy_page)
                    _open_prelabel(legacy_page)
                    legacy_page.locator('.st-key-anomaly_prelabel_export_btn button').click()
                    legacy_page.wait_for_function(
                        "() => (document.querySelector('[data-testid=\"stMain\"]')?.innerText || '').includes('預標匯出完成')",
                        timeout=60000,
                    )
                    wait_idle(legacy_page)
                    case_files = file_tree_bytes(case_output)
                    assert case_files == react_case["files"], json.dumps({
                        "case": case_id,
                        "streamlit_files": sorted(case_files),
                        "react_files": sorted(react_case["files"]),
                        "different_files": [
                            name for name in sorted(set(case_files) | set(react_case["files"]))
                            if case_files.get(name) != react_case["files"].get(name)
                        ],
                    }, ensure_ascii=False)
                    expect(legacy_page.locator('[data-testid="stException"]')).to_have_count(0)

                for case_id, min_conf, target_classes in prelabel_cases:
                    run_legacy_prelabel_case(case_id, min_conf, target_classes)

                assert source_hashes == {
                    path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in root.rglob("*") if path.is_file()
                }
                expect(legacy_page.locator('[data-testid="stException"]')).to_have_count(0)
            finally:
                legacy_page.close()
                _stop_process_tree(legacy_process)
                legacy_log.close()

            browser.close()
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)


def test_lv_web_06_review_01_visible_anomaly_candidate_cart_exports_original_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """React sends a scored object to curation and ZIP exports the untouched source image."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    from playwright.sync_api import expect, sync_playwright

    cache_root = tmp_path / "cache"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "cases"))
    training = tmp_path / "training"
    training.mkdir()
    target = tmp_path / "target"
    (target / "images").mkdir(parents=True)
    source_image = target / "images" / "panel.png"
    Image.new("RGB", (128, 96), (145, 80, 55)).save(source_image)
    source_bytes = source_image.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    def score_candidate(request: dict, bundle: dict, *, progress=None) -> dict:
        if progress:
            progress(90, 100, "Fixture anomaly candidate ready")
        return {
            "records": [{
                "image_path": str(source_image),
                "bbox": [0.25, 0.2, 0.75, 0.8],
                "label": "part",
                "split": "test",
                "obj_index": 0,
            }],
            "scores": np.asarray([0.95], dtype=np.float64),
            "labels": np.asarray(["bad"], dtype=object),
            "ranking": [0],
            "threshold": 0.5,
            "n_bad": 1,
            "auroc": None,
            "score_mode": "patch",
            "model_id": bundle["meta"]["model"],
            "model_meta": bundle["meta"],
            "model_bank_sha256": "a" * 64,
            "object_source": "yolo",
            "bank": None,
            "skipped_count": 0,
            "fewshot_aligned": {},
            "fewshot_unmatched": 0,
            "confirmed": {},
            "contamination": float(request["contamination"]),
            "created_at": 1_798_000_000.0,
        }

    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: lambda _path: np.zeros(2, dtype=np.float32),
        anomaly_model_builder=_fixture_anomaly_model_builder,
        anomaly_scorer=score_candidate,
    )
    server, server_thread, base_url = _start_api(application)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
            page.set_default_timeout(20000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("open-anomaly-bank").click()
            page.get_by_test_id("anomaly-train-folders").fill(str(training))
            page.get_by_test_id("anomaly-build").click()
            expect(page.get_by_test_id("anomaly-model-status")).to_contain_text("模型已建立")

            page.get_by_test_id("anomaly-target-folders").fill(str(target))
            page.get_by_test_id("anomaly-score").click()
            expect(page.get_by_test_id("anomaly-review-workspace")).to_be_visible()
            candidate = page.get_by_test_id("anomaly-object-0")
            expect(candidate).to_be_visible()
            candidate.get_by_label("選取 panel.png 物件 1 加入策展清單").check()
            page.get_by_test_id("anomaly-add-selected-curation").click()
            expect(page.get_by_test_id("anomaly-notice")).to_contain_text("已加入策展清單 1 個原圖候選")
            expect(page.get_by_test_id("open-curation")).to_contain_text("Curation (1)")

            page.get_by_test_id("open-curation").click()
            expect(page.get_by_test_id("curation-item")).to_contain_text("panel.png")
            with page.expect_download() as download:
                page.get_by_test_id("curation-export-zip").click()
            with zipfile.ZipFile(download.value.path()) as archive:
                exported = [name for name in archive.namelist() if name.lower().endswith(".png")]
                assert len(exported) == 1
                exported_bytes = archive.read(exported[0])
            assert hashlib.sha256(exported_bytes).hexdigest() == source_sha256
            with Image.open(BytesIO(exported_bytes)) as image:
                assert image.size == (128, 96)
            assert source_image.read_bytes() == source_bytes
            browser.close()
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)


def test_lv_web_10_prelabel_01_visible_preview_safe_export_and_no_head_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """M11 preview/count/readback and C6 source protection are visible in React."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    from playwright.sync_api import expect, sync_playwright

    cache_root = tmp_path / "cache"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("LV_CASES_DIR", str(tmp_path / "cases"))
    training = tmp_path / "training"
    training.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    source_image = target / "panel.png"
    Image.new("RGB", (128, 96), (145, 80, 55)).save(source_image)
    source_before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in target.iterdir()}
    output = tmp_path / "prelabel-output"

    def prelabel_builder(request: dict, *, progress=None) -> dict:
        bundle = _fixture_anomaly_model_builder(request, progress=progress)
        if request.get("label_semantic") == "defect":
            bundle["head"] = {"classes": ["scratch", "stain"], "dim": 2}
            bundle["summary"].update({
                "has_head": True,
                "head_status": "能分辨瑕疵種類",
                "eligible_classes": ["scratch", "stain"],
            })
        return bundle

    def score_prelabel_targets(request: dict, bundle: dict, *, progress=None) -> dict:
        if progress:
            progress(90, 100, "Fixture anomaly candidates ready")
        return {
            "records": [
                {"image_path": str(source_image), "bbox": [0.5, 0.5, 0.4, 0.6], "label": "part", "split": "test", "obj_index": index}
                for index in range(2)
            ],
            "obj_emb": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "scores": np.asarray([0.9, 0.95], dtype=np.float64),
            "labels": np.asarray(["bad", "bad"], dtype=object),
            "ranking": [1, 0],
            "threshold": 0.5,
            "n_bad": 2,
            "auroc": None,
            "score_mode": "patch",
            "model_id": bundle["meta"]["model"],
            "model_meta": bundle["meta"],
            "model_bank_sha256": "c" * 64,
            "object_source": "yolo",
            "bank": None,
            "skipped_count": 0,
            "fewshot_aligned": {},
            "fewshot_unmatched": 0,
            "confirmed": {},
            "contamination": float(request["contamination"]),
            "created_at": 1_798_000_000.0,
        }

    import dino_head

    def predict_head(_head, embeddings):
        rows = np.asarray(embeddings)
        labels = np.where(rows[:, 0] > 0, "scratch", "stain")
        confidence = np.asarray([0.91, 0.83], dtype=np.float32)
        probabilities = np.asarray([[0.91, 0.09], [0.17, 0.83]], dtype=np.float32)
        return labels, confidence, probabilities

    def gated_predict(head, embeddings, scores, *, anomaly_threshold, min_conf=0.5):
        labels, confidence, _ = predict_head(head, embeddings)
        return [
            label if score >= anomaly_threshold and conf >= min_conf else "Unknown"
            for label, conf, score in zip(labels, confidence, scores)
        ]

    monkeypatch.setattr(dino_head, "predict_head", predict_head)
    monkeypatch.setattr(dino_head, "gated_predict", gated_predict)
    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: lambda _path: np.zeros(2, dtype=np.float32),
        anomaly_model_builder=prelabel_builder,
        anomaly_scorer=score_prelabel_targets,
    )
    server, server_thread, base_url = _start_api(application)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(15000)
            api_responses: list[str] = []

            def capture_api_json(response):
                if (
                    "/api/v1/anomaly/" in response.url and response.ok
                    and response.headers.get("content-type", "").startswith("application/json")
                ):
                    api_responses.append(response.text())

            page.on("response", capture_api_json)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("open-anomaly-bank").click()
            page.get_by_test_id("anomaly-train-folders").fill(str(training))
            page.get_by_test_id("anomaly-model").select_option("fixture-model")

            # No classification head: friendly message and no export control.
            page.get_by_test_id("anomaly-build").click()
            expect(page.get_by_test_id("anomaly-model-status")).to_contain_text("只做異常偵測")
            page.get_by_test_id("anomaly-target-folders").fill(str(target))
            page.get_by_test_id("anomaly-score").click()
            expect(page.get_by_test_id("anomaly-prelabel-unavailable")).to_contain_text("沒有瑕疵分類頭")
            expect(page.get_by_test_id("anomaly-prelabel-export")).to_have_count(0)

            # A defect head enables M11 preview and safe export.
            page.get_by_test_id("anomaly-label-semantic").select_option("defect")
            page.get_by_test_id("anomaly-build").click()
            expect(page.get_by_test_id("anomaly-model-status")).to_contain_text("能分辨瑕疵種類")
            page.get_by_test_id("anomaly-score").click()
            expect(page.get_by_test_id("anomaly-prelabel-workspace")).to_be_visible()
            page.get_by_label("預標類別 stain").uncheck()
            page.get_by_test_id("anomaly-prelabel-preview").click()
            expect(page.get_by_test_id("anomaly-prelabel-summary")).to_contain_text("接受 1 / 2 個物件")
            expect(page.get_by_test_id("anomaly-prelabel-item-0")).to_contain_text("scratch")
            expect(page.get_by_test_id("anomaly-prelabel-item-1")).to_contain_text("不接受")
            page.get_by_label("預標類別 stain").check()
            expect(page.get_by_test_id("anomaly-prelabel-summary")).to_have_count(0)
            page.get_by_label("預標類別 stain").uncheck()
            page.get_by_test_id("anomaly-prelabel-preview").click()
            expect(page.get_by_test_id("anomaly-prelabel-summary")).to_contain_text("接受 1 / 2 個物件")
            page.wait_for_function(
                "document.querySelector('[data-testid=\"anomaly-prelabel-item-0\"] img')?.naturalWidth > 0",
                timeout=15000,
            )
            page.get_by_test_id("anomaly-prelabel-output-dir").fill(str(output))
            page.get_by_test_id("anomaly-prelabel-export").click()
            expect(page.get_by_test_id("anomaly-notice")).to_contain_text("預標匯出完成")

            label_file = output / "labels" / "panel.txt"
            assert label_file.read_text(encoding="utf-8").splitlines() == [
                "0 0.500000 0.500000 0.400000 0.600000",
            ]
            assert (output / "classes.txt").read_text(encoding="utf-8").splitlines() == ["scratch", "stain"]
            assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in target.iterdir()} == source_before
            assert all(str(tmp_path) not in body for body in api_responses)
            browser.close()
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)


def test_lv_web_07_scan_01_visible_reviews_and_appends_fewshot_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Review scan candidates in React, append accepted samples, then scan again."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    sample_root = tmp_path / "fewshot-samples"
    (sample_root / "images").mkdir(parents=True)
    (sample_root / "labels").mkdir()
    (sample_root / "classes.txt").write_text("scratch\nstain\n", encoding="utf-8")
    Image.new("RGB", (48, 40), (220, 30, 40)).save(sample_root / "images" / "sample-0.png")
    Image.new("RGB", (48, 40), (20, 180, 80)).save(sample_root / "images" / "sample-1.png")
    (sample_root / "labels" / "sample-0.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n1 0.25 0.25 0.25 0.25\n", encoding="utf-8",
    )
    (sample_root / "labels" / "sample-1.txt").write_text(
        "0 0.5 0.5 0.75 0.75\n", encoding="utf-8",
    )

    target_root = tmp_path / "fewshot-targets"
    (target_root / "images").mkdir(parents=True)
    (target_root / "labels").mkdir()
    (target_root / "classes.txt").write_text("scratch\nstain\n", encoding="utf-8")
    Image.new("RGB", (48, 40), (220, 30, 40)).save(target_root / "images" / "target-0.png")
    Image.new("RGB", (48, 40), (20, 180, 80)).save(target_root / "images" / "target-1.png")
    (target_root / "labels" / "target-0.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n1 0.25 0.25 0.25 0.25\n", encoding="utf-8",
    )
    (target_root / "labels" / "target-1.txt").write_text(
        "0 0.5 0.5 0.75 0.75\n", encoding="utf-8",
    )
    source_files = [path for root in (sample_root, target_root) for path in root.rglob("*") if path.is_file()]
    source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}

    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: lambda crop: np.asarray(
            [np.asarray(crop.convert("RGB"), dtype=np.float32).mean(), crop.width, crop.height, 1],
            dtype=np.float32,
        ),
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1050})
            page.set_default_timeout(20_000)
            anomaly_model_build_requests: list[str] = []
            scan_requests: list[dict] = []
            page.on("request", lambda request: anomaly_model_build_requests.append(request.url)
                    if "/api/v1/anomaly/models/jobs" in request.url else None)
            page.on("request", lambda request: scan_requests.append(json.loads(request.post_data or "{}"))
                    if request.method == "POST"
                    and request.url.rstrip("/").endswith("/api/v1/fewshot/scans/jobs") else None)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("open-fewshot-samples").click()
            expect(page.get_by_role("heading", name="以樣搜樣：建立樣本集")).to_be_visible()

            page.get_by_test_id("fewshot-sample-folders").fill(str(sample_root))
            page.get_by_test_id("fewshot-object-source").select_option("yolo")
            page.get_by_test_id("fewshot-build-bank").click()
            expect(page.get_by_test_id("fewshot-bank-summary")).to_contain_text("3 個樣本", timeout=60_000)
            bank_id = page.get_by_test_id("fewshot-bank-id").input_value()

            page.get_by_test_id("fewshot-target-folders").fill(str(target_root))
            page.get_by_test_id("fewshot-theta").fill("0.9")
            page.get_by_test_id("fewshot-scan").click()
            expect(page.get_by_test_id("fewshot-scan-summary")).to_contain_text("候選 3", timeout=60_000)
            candidates = page.get_by_test_id("fewshot-candidate")
            expect(candidates).to_have_count(3)
            preview = candidates.nth(0).get_by_test_id("fewshot-candidate-preview")
            page.wait_for_function(
                "element => element.complete && element.naturalWidth > 0",
                arg=preview.element_handle(),
                timeout=15_000,
            )

            candidates.nth(1).get_by_role("checkbox").uncheck()
            candidates.nth(0).get_by_role("combobox").select_option(label="stain")
            page.get_by_test_id("fewshot-add-selected").click()
            expect(page.get_by_test_id("fewshot-scan-notice")).to_contain_text("已加入 2 顆", timeout=30_000)
            expect(page.get_by_test_id("fewshot-bank-summary")).to_contain_text("5 個樣本")
            updated_summary = page.get_by_test_id("fewshot-bank-summary").inner_text()
            assert "stain 2" in updated_summary or "stain 3" in updated_summary

            page.get_by_test_id("fewshot-scan").click()
            expect(page.get_by_test_id("fewshot-scan-summary")).to_contain_text("候選 3", timeout=60_000)
            assert [item["sample_bank_id"] for item in scan_requests] == [bank_id, bank_id]
            assert anomaly_model_build_requests == []
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_hashes}


def test_lv_web_07_scan_01_real_dinov2_visible_reviews_and_appends_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Run the complete visible few-shot journey with the pinned real DINOv2 model."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    configured_models = Path(
        os.environ.get("LV_TEST_MODELS_DIR") or os.environ.get("LV_MODELS_DIR") or ROOT / "models"
    ).expanduser().resolve()
    configured_weights = (
        configured_models / "dinov2_vits14" / "dinov2_vits14.pth",
        configured_models / "dinov2_vits14.pth",
    )
    cache_weights = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vits14_pretrain.pth"
    source_weights = next((path for path in configured_weights if path.is_file()), None)
    if source_weights is None and cache_weights.is_file():
        source_weights = cache_weights
    if source_weights is None:
        pytest.skip("pinned real DINOv2-small weights are unavailable in LV models or the local torch cache")
    assert hashlib.sha256(source_weights.read_bytes()).hexdigest() == DINOV2_VITS14_SHA256, \
        "the local model weights differ from the LV-WEB-06-BANK-01 baseline"

    models_dir = tmp_path / "models"
    model_dir = models_dir / "dinov2_vits14"
    model_dir.mkdir(parents=True)
    weights = model_dir / "dinov2_vits14.pth"
    try:
        os.link(source_weights, weights)
    except OSError:
        shutil.copyfile(source_weights, weights)
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == DINOV2_VITS14_SHA256

    sample_root = tmp_path / "real-fewshot-samples"
    (sample_root / "images").mkdir(parents=True)
    (sample_root / "labels").mkdir()
    (sample_root / "classes.txt").write_text("scratch\nstain\n", encoding="utf-8")
    colors = {"scratch": (205, 38, 46), "stain": (26, 104, 188)}
    sample_hashes = {}
    for index, (label, color) in enumerate(colors.items()):
        image_path = sample_root / "images" / f"sample-{label}.png"
        image = Image.new("RGB", (96, 96), color)
        for offset in range(12, 84, 12):
            image.putpixel((offset, (offset * 3) % 84 + 6), (245, 220, 45))
        image.save(image_path)
        (sample_root / "labels" / f"sample-{label}.txt").write_text(
            f"{index} 0.5 0.5 0.75 0.75\n", encoding="utf-8",
        )
        sample_hashes[image_path.name] = hashlib.sha256(image_path.read_bytes()).hexdigest()

    target_root = tmp_path / "real-fewshot-targets"
    (target_root / "images").mkdir(parents=True)
    (target_root / "labels").mkdir()
    (target_root / "classes.txt").write_text("scratch\nstain\n", encoding="utf-8")
    for index, label in enumerate(colors):
        shutil.copyfile(sample_root / "images" / f"sample-{label}.png",
                        target_root / "images" / f"target-{label}.png")
        (target_root / "labels" / f"target-{label}.txt").write_text(
            f"{index} 0.5 0.5 0.75 0.75\n", encoding="utf-8",
        )
    source_files = [
        path for root in (sample_root, target_root) for path in root.rglob("*") if path.is_file()
    ]
    source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}

    monkeypatch.setenv("LV_MODELS_DIR", str(models_dir))
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: available_models(models_dir),
        model_loader=lambda model_id: load_model(model_id, models_dir=models_dir),
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1050})
            page.set_default_timeout(300_000)
            job_responses: list[dict] = []
            add_responses: list[dict] = []
            scan_requests: list[dict] = []
            fewshot_response_bodies: list[str] = []
            anomaly_model_requests: list[str] = []

            def capture_response(response):
                if not response.ok:
                    return
                if "/api/v1/fewshot/jobs/" in response.url:
                    job_responses.append(response.json())
                    fewshot_response_bodies.append(response.text())
                elif response.request.method == "POST" and response.url.rstrip("/").endswith(
                    "/api/v1/fewshot/sample-banks/add-candidates"
                ):
                    add_responses.append(response.json())
                    fewshot_response_bodies.append(response.text())
                elif response.request.method == "GET" and "/api/v1/fewshot/sample-banks/" in response.url:
                    fewshot_response_bodies.append(response.text())

            page.on("response", capture_response)
            page.on("request", lambda request: scan_requests.append(json.loads(request.post_data or "{}"))
                    if request.method == "POST"
                    and request.url.rstrip("/").endswith("/api/v1/fewshot/scans/jobs") else None)
            page.on("request", lambda request: anomaly_model_requests.append(request.url)
                    if "/api/v1/anomaly/models/jobs" in request.url else None)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("open-fewshot-samples").click()
            expect(page.get_by_role("heading", name="以樣搜樣：建立樣本集")).to_be_visible()

            page.get_by_test_id("fewshot-sample-folders").fill(str(sample_root))
            page.get_by_test_id("fewshot-object-source").select_option("yolo")
            page.get_by_test_id("fewshot-model").select_option("dinov2_vits14")
            page.get_by_test_id("fewshot-build-bank").click()
            expect(page.get_by_test_id("fewshot-bank-summary")).to_contain_text("2 個樣本", timeout=300_000)
            expect(page.get_by_test_id("fewshot-bank-summary")).to_contain_text("特徵維度 D=384")
            bank_id = page.get_by_test_id("fewshot-bank-id").input_value()
            built_jobs = [job for job in job_responses if job.get("state") == "completed"
                          and job.get("result", {}).get("sample_bank_id") == bank_id]
            assert built_jobs
            assert built_jobs[-1]["result"]["model_id"] == "dinov2_vits14"
            assert built_jobs[-1]["result"]["feature_dimension"] == 384
            assert built_jobs[-1]["result"]["class_counts"] == {"scratch": 1, "stain": 1}

            page.get_by_test_id("fewshot-reload-bank").click()
            expect(page.get_by_test_id("fewshot-notice")).to_contain_text("重新讀回", timeout=15_000)
            expect(page.get_by_test_id("fewshot-bank-summary")).to_contain_text("D=384")
            page.get_by_test_id("fewshot-target-folders").fill(str(target_root))
            page.get_by_test_id("fewshot-theta").fill("0.95")
            page.get_by_test_id("fewshot-scan").click()
            expect(page.get_by_test_id("fewshot-scan-summary")).to_contain_text("候選 2", timeout=300_000)
            candidates = page.get_by_test_id("fewshot-candidate")
            expect(candidates).to_have_count(2)
            completed_scan_jobs = [job for job in job_responses if job.get("state") == "completed"
                                   and job.get("result", {}).get("scan_id")]
            assert completed_scan_jobs
            scan_result = completed_scan_jobs[-1]["result"]
            assert scan_result["sample_bank_id"] == bank_id
            assert scan_result["model_id"] == "dinov2_vits14"
            assert scan_result["objects_scored"] == 2
            assert scan_result["candidate_count"] == 2
            matches = {Path(candidate["filename"]).stem: candidate for candidate in scan_result["candidates"]}
            assert set(matches) == {"target-scratch", "target-stain"}
            assert all(candidate["similarity"] >= 0.99 for candidate in matches.values())
            assert matches["target-scratch"]["suggested_class"] == "scratch"
            assert matches["target-stain"]["suggested_class"] == "stain"
            for index in range(2):
                preview = candidates.nth(index).get_by_test_id("fewshot-candidate-preview")
                page.wait_for_function(
                    "element => element.complete && element.naturalWidth > 0",
                    arg=preview.element_handle(), timeout=30_000,
                )

            # Confirm the suggested labels in the visible review controls and append both vectors.
            visible_labels = {}
            for index in range(2):
                item = candidates.nth(index)
                item_text = item.inner_text()
                filename = next(name for name in ("target-scratch.png", "target-stain.png") if name in item_text)
                visible_labels[filename] = item.get_by_role("combobox").input_value()
                expect(item.get_by_role("checkbox")).to_be_checked()
            assert visible_labels == {"target-scratch.png": "scratch", "target-stain.png": "stain"}
            page.get_by_test_id("fewshot-add-selected").click()
            expect(page.get_by_test_id("fewshot-scan-notice")).to_contain_text("已加入 2 顆", timeout=300_000)
            expect(page.get_by_test_id("fewshot-bank-summary")).to_contain_text("4 個樣本", timeout=30_000)
            assert add_responses and add_responses[-1]["sample_bank"]["sample_count"] == 4
            assert add_responses[-1]["sample_bank"]["feature_dimension"] == 384
            page.get_by_test_id("fewshot-reload-bank").click()
            expect(page.get_by_test_id("fewshot-notice")).to_contain_text("重新讀回", timeout=15_000)
            expect(page.get_by_test_id("fewshot-bank-summary")).to_contain_text("4 個樣本")
            page.get_by_test_id("fewshot-scan").click()
            expect(page.get_by_test_id("fewshot-scan-summary")).to_contain_text("候選 2", timeout=300_000)
            assert [request["sample_bank_id"] for request in scan_requests] == [bank_id, bank_id]
            assert anomaly_model_requests == []
            assert all(str(tmp_path) not in body for body in fewshot_response_bodies)
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_hashes}
    assert source_hashes[sample_root / "images" / "sample-scratch.png"] == \
        sample_hashes["sample-scratch.png"]


def test_lv_web_07_export_01_visible_monitor_queue_without_anomaly_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Initialize and run the local monitor, review queue items, and download its profile."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    sample_root = tmp_path / "fewshot-samples"
    (sample_root / "images").mkdir(parents=True)
    (sample_root / "labels").mkdir()
    (sample_root / "classes.txt").write_text("scratch\nstain\n", encoding="utf-8")
    Image.new("RGB", (48, 40), (220, 30, 40)).save(sample_root / "images" / "sample-0.png")
    Image.new("RGB", (48, 40), (20, 180, 80)).save(sample_root / "images" / "sample-1.png")
    (sample_root / "labels" / "sample-0.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n1 0.25 0.25 0.25 0.25\n", encoding="utf-8",
    )
    (sample_root / "labels" / "sample-1.txt").write_text(
        "0 0.5 0.5 0.75 0.75\n", encoding="utf-8",
    )
    target_root = tmp_path / "fewshot-targets"
    (target_root / "images").mkdir(parents=True)
    (target_root / "labels").mkdir()
    (target_root / "classes.txt").write_text("scratch\nstain\n", encoding="utf-8")
    Image.new("RGB", (48, 40), (220, 30, 40)).save(target_root / "images" / "target-0.png")
    Image.new("RGB", (48, 40), (20, 180, 80)).save(target_root / "images" / "target-1.png")
    (target_root / "labels" / "target-0.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n1 0.25 0.25 0.25 0.25\n", encoding="utf-8",
    )
    (target_root / "labels" / "target-1.txt").write_text(
        "0 0.5 0.5 0.75 0.75\n", encoding="utf-8",
    )
    source_files = [path for root in (sample_root, target_root) for path in root.rglob("*") if path.is_file()]
    source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}

    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: lambda crop: np.asarray(
            [np.asarray(crop.convert("RGB"), dtype=np.float32).mean(), crop.width, crop.height, 1],
            dtype=np.float32,
        ),
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1100}, accept_downloads=True)
            page.set_default_timeout(20_000)
            anomaly_model_build_requests: list[str] = []
            page.on("request", lambda request: anomaly_model_build_requests.append(request.url)
                    if "/api/v1/anomaly/models/jobs" in request.url else None)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("open-fewshot-samples").click()
            page.get_by_test_id("fewshot-sample-folders").fill(str(sample_root))
            page.get_by_test_id("fewshot-object-source").select_option("yolo")
            page.get_by_test_id("fewshot-build-bank").click()
            expect(page.get_by_test_id("fewshot-bank-summary")).to_contain_text("3 個樣本", timeout=60_000)

            page.get_by_test_id("fewshot-target-folders").fill(str(target_root))
            page.get_by_test_id("fewshot-monitor-directory").fill(str(tmp_path / "fewshot-monitor"))
            page.get_by_test_id("fewshot-monitor-scan").click()
            expect(page.get_by_test_id("fewshot-monitor-summary")).to_contain_text("待標佇列 3 項", timeout=60_000)
            items = page.get_by_test_id("fewshot-monitor-queue-item")
            expect(items).to_have_count(3)
            preview = items.nth(0).get_by_test_id("fewshot-monitor-preview")
            page.wait_for_function(
                "element => element.complete && element.naturalWidth > 0",
                arg=preview.element_handle(),
                timeout=15_000,
            )

            items.nth(0).get_by_role("combobox").select_option(label="stain")
            page.get_by_test_id("fewshot-monitor-accept-0").click()
            expect(items.nth(0)).to_contain_text("已採納 stain", timeout=15_000)
            page.get_by_test_id("fewshot-monitor-skip-1").click()
            expect(items.nth(1)).to_contain_text("已略過（下輪仍保留）", timeout=15_000)

            with page.expect_download() as profile_download:
                page.get_by_test_id("fewshot-monitor-export-profile").click()
            profile_path = Path(profile_download.value.path())
            assert "objective: retrieve" in profile_path.read_text(encoding="utf-8")
            assert anomaly_model_build_requests == []
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_hashes}


def test_lv_web_07_export_01_visible_yolo_csv_export_tracks_unchecked_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Export checked scan results to a separate YOLO+CSV directory without source writes."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    sample_root = tmp_path / "fewshot-samples"
    (sample_root / "images").mkdir(parents=True)
    (sample_root / "labels").mkdir()
    (sample_root / "classes.txt").write_text("scratch\nstain\n", encoding="utf-8")
    Image.new("RGB", (48, 40), (220, 30, 40)).save(sample_root / "images" / "sample-0.png")
    Image.new("RGB", (48, 40), (20, 180, 80)).save(sample_root / "images" / "sample-1.png")
    (sample_root / "labels" / "sample-0.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n1 0.25 0.25 0.25 0.25\n", encoding="utf-8",
    )
    (sample_root / "labels" / "sample-1.txt").write_text(
        "0 0.5 0.5 0.75 0.75\n", encoding="utf-8",
    )
    target_root = tmp_path / "fewshot-targets"
    (target_root / "images").mkdir(parents=True)
    (target_root / "labels").mkdir()
    (target_root / "classes.txt").write_text("scratch\nstain\n", encoding="utf-8")
    Image.new("RGB", (48, 40), (220, 30, 40)).save(target_root / "images" / "target-0.png")
    Image.new("RGB", (48, 40), (20, 180, 80)).save(target_root / "images" / "target-1.png")
    (target_root / "labels" / "target-0.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n1 0.25 0.25 0.25 0.25\n", encoding="utf-8",
    )
    (target_root / "labels" / "target-1.txt").write_text(
        "0 0.5 0.5 0.75 0.75\n", encoding="utf-8",
    )
    source_files = [path for root in (sample_root, target_root) for path in root.rglob("*") if path.is_file()]
    source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}

    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: lambda crop: np.asarray(
            [np.asarray(crop.convert("RGB"), dtype=np.float32).mean(), crop.width, crop.height, 1],
            dtype=np.float32,
        ),
    )
    server = server_thread = None
    out_all = tmp_path / "fewshot-export-all"
    out_less = tmp_path / "fewshot-export-less"
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1050})
            page.set_default_timeout(20_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("open-fewshot-samples").click()
            page.get_by_test_id("fewshot-sample-folders").fill(str(sample_root))
            page.get_by_test_id("fewshot-object-source").select_option("yolo")
            page.get_by_test_id("fewshot-build-bank").click()
            expect(page.get_by_test_id("fewshot-bank-summary")).to_contain_text("3 個樣本", timeout=60_000)

            page.get_by_test_id("fewshot-target-folders").fill(str(target_root))
            page.get_by_test_id("fewshot-theta").fill("0.9")
            page.get_by_test_id("fewshot-scan").click()
            expect(page.get_by_test_id("fewshot-scan-summary")).to_contain_text("候選 3", timeout=60_000)
            expect(page.get_by_test_id("fewshot-export-copy-images")).to_be_checked()
            page.get_by_test_id("fewshot-export-directory").fill(str(out_all))
            page.get_by_test_id("fewshot-export").click()
            expect(page.get_by_test_id("fewshot-export-notice")).to_contain_text("已匯出 3 個 YOLO 標註", timeout=30_000)

            candidates = page.get_by_test_id("fewshot-candidate")
            expect(candidates).to_have_count(3)
            candidates.nth(0).get_by_role("checkbox").uncheck()
            page.get_by_test_id("fewshot-export-directory").fill(str(out_less))
            page.get_by_test_id("fewshot-export").click()
            expect(page.get_by_test_id("fewshot-export-notice")).to_contain_text("已匯出 2 個 YOLO 標註", timeout=30_000)
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    def exported_object_count(folder: Path) -> int:
        return sum(len(path.read_text(encoding="utf-8").splitlines())
                   for path in (folder / "labels").glob("*.txt"))

    assert exported_object_count(out_all) == 3
    assert exported_object_count(out_less) == 2
    for folder in (out_all, out_less):
        assert (folder / "retrieval_report.csv").is_file()
        assert (folder / "classes.txt").read_text(encoding="utf-8").splitlines() == ["scratch", "stain"]
        image_stems = {path.stem for path in (folder / "images").iterdir() if path.is_file()}
        label_stems = {path.stem for path in (folder / "labels").glob("*.txt")}
        assert image_stems and label_stems <= image_stems
    assert source_hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_hashes}


def test_lv_web_10_eval_01_visible_react_detection_evaluation_and_escape_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the web workspace with npm run build before running this E2E")

    dataset = tmp_path / "evaluation-dataset"
    dataset.mkdir()
    (dataset / "classes.txt").write_text("dent\n", encoding="utf-8")
    for filename, color in (
        ("matched.png", (20, 110, 210)),
        ("missed.png", (210, 80, 40)),
        ("gray.png", (80, 180, 70)),
    ):
        (dataset / "images").mkdir(exist_ok=True)
        (dataset / "labels").mkdir(exist_ok=True)
        Image.new("RGB", (64, 48), color).save(dataset / "images" / filename)
        (dataset / "labels" / f"{Path(filename).stem}.txt").write_text(
            "0 0.5 0.5 0.25 0.25\n", encoding="utf-8",
        )
    source_hashes = {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }
    import round_ledger

    output_dir = tmp_path / "evaluation-output"
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "evaluation-cache"))
    monkeypatch.setenv("LV_OUTPUT_DIR", str(output_dir))
    round_id = round_ledger.start_round("fixture", {}, "curation")
    application = create_app()
    server = server_thread = browser = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_timeout(15_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("open-evaluation").click()
            expect(page.get_by_test_id("evaluation-workspace")).to_be_visible()
            page.get_by_test_id("run-evaluation-demo").click()
            expect(page.get_by_test_id("evaluation-results")).to_be_visible()
            expect(page.get_by_role("status").filter(has_text="範例資料")).to_be_visible()
            expect(page.get_by_test_id("evaluation-escape-count")).not_to_have_text("0")
            expect(page.get_by_test_id("sample-evaluation-not-attachable")).to_be_visible()
            expect(page.get_by_test_id("attach-evaluation-to-round")).to_have_count(0)

            page.get_by_label("Dataset folder path").fill(str(dataset))
            page.get_by_role("button", name="Add dataset folder").click()
            expect(page.get_by_test_id("evaluation-folder-list")).to_contain_text(str(dataset))

            page.get_by_test_id("run-detection-evaluation").click()
            expect(page.get_by_role("alert")).to_contain_text("predictions.csv")

            predictions = (
                "filename,class,cx,cy,w,h,score\n"
                "matched.png,dent,0.5,0.5,0.25,0.25,0.9\n"
            )
            consensus = (
                "filename,consensus,cx,cy,w,h\n"
                "matched.png,true,0.5,0.5,0.25,0.25\n"
                "missed.png,true,0.5,0.5,0.25,0.25\n"
                "gray.png,false,0.5,0.5,0.25,0.25\n"
            )
            page.get_by_test_id("evaluation-predictions-file").set_input_files({
                "name": "predictions.csv", "mimeType": "text/csv", "buffer": predictions.encode(),
            })
            page.get_by_test_id("evaluation-consensus-file").set_input_files({
                "name": "consensus_set.csv", "mimeType": "text/csv", "buffer": consensus.encode(),
            })
            page.get_by_test_id("run-detection-evaluation").click()

            expect(page.get_by_test_id("evaluation-results")).to_be_visible()
            expect(page.get_by_test_id("evaluation-overall-recall")).to_have_text("50.0%")
            expect(page.get_by_test_id("evaluation-overall-precision")).to_have_text("100.0%")
            expect(page.get_by_test_id("evaluation-escape-count")).to_have_text("1")
            expect(page.get_by_test_id("evaluation-per-class-table").locator("tbody tr")).to_have_count(1)
            expect(page.get_by_text("共識 GT 2 · 灰帶 GT 1")).to_be_visible()
            page.get_by_label("Round to attach evaluation").select_option(round_id)
            page.get_by_test_id("attach-evaluation-to-round").click()
            expect(page.get_by_test_id("evaluation-round-attached")).to_contain_text(round_id)
            image = page.locator("img[alt*='missed.png']")
            expect(image).to_be_visible()
            page.wait_for_function(
                "() => { const image = document.querySelector(\"img[alt*='missed.png']\"); "
                "return image && image.complete && image.naturalWidth > 0; }",
            )
            with page.expect_download() as download_info:
                page.get_by_test_id("download-evaluation-escapes").click()
            download = download_info.value
            assert download.suggested_filename == "escapes.csv"
            exported = Path(download.path()).read_text(encoding="utf-8")
            assert "missed.png,dent,0.5,0.5,0.25,0.25" in exported
            page.get_by_test_id("open-curation").click()
            report = page.get_by_test_id("active-learning-evaluation-report")
            expect(report).to_contain_text("50.0% recall")
            report.locator("summary").click()
            expect(report).to_contain_text("IoU 0.50")
            expect(report).to_contain_text("FN 1")
            expect(report.locator("tbody tr")).to_have_count(1)
            browser.close()
            browser = None
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert source_hashes == {
        path.relative_to(dataset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.rglob("*") if path.is_file()
    }
