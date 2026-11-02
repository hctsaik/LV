"""Visible browser workflow for the local Compare Distributions page."""
from __future__ import annotations

import hashlib
import os
import shutil
import socket
import threading
import time
from pathlib import Path

import uvicorn
from PIL import Image, ImageDraw
from playwright.sync_api import expect, sync_playwright

from react_web_shell import create_app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _embed(path: Path):
    import numpy as np

    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    vec = np.zeros(384, dtype=np.float32)
    vec[:3] = arr.mean(axis=(0, 1)) / 255.0 + 0.01
    vec[3] = len(path.name) / 10
    return vec


def _embed_object(image: Image.Image):
    import numpy as np

    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    vec = np.zeros(384, dtype=np.float32)
    vec[:3] = arr.mean(axis=(0, 1)) / 255.0 + 0.01
    vec[3] = image.width / max(image.height, 1)
    return vec


def test_lv_web_02_img_01_react_compare_flow_with_local_worker(tmp_path: Path, monkeypatch):
    for side, base in (("a", (220, 30, 40)), ("b", (30, 50, 220))):
        root = tmp_path / side
        root.mkdir()
        for index in range(4):
            Image.new("RGB", (48, 48), (base[0], min(255, base[1] + index), base[2])).save(root / f"{side}-{index}.png")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: _embed,
    )
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(application, host="127.0.0.1", port=port, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                import urllib.request
                with urllib.request.urlopen(base_url + "/api/v1/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("local worker did not start")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(base_url, wait_until="networkidle")
            page.get_by_role("button", name="Compare").click()
            page.get_by_label("Comparison unit").select_option("whole")
            page.get_by_label("Image folder", exact=True).nth(0).fill(str(tmp_path / "a"))
            page.get_by_role("button", name="Add Folder A").click()
            page.get_by_label("Image folder", exact=True).nth(1).fill(str(tmp_path / "b"))
            page.get_by_role("button", name="Add Folder B").click()
            page.get_by_label("Feature model").select_option("fixture-model")
            page.get_by_role("button", name="Run comparison").click()
            expect(page.get_by_test_id("compare-job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_role("heading", name="Overall image drift")).to_be_visible()
            expect(page.get_by_role("button", name="(全部影像)", exact=True)).to_be_visible()
            expect(page.get_by_test_id("comparison-point")).to_have_count(8, timeout=60_000)
            page.get_by_test_id("comparison-point").nth(0).click()
            expect(page.get_by_test_id("comparison-focused-record")).to_contain_text("Group A")
            expect(page.get_by_test_id("comparison-focused-record")).to_be_visible()
            page.get_by_label("Comparison projection", exact=True).select_option("pca")
            expect(page.get_by_test_id("comparison-point")).to_have_count(8, timeout=60_000)
            expect(page.locator(".compare-gallery-grid img").first).to_be_visible()
            page.get_by_label("Comparison unit").select_option("object")
            expect(page.get_by_role("heading", name="Your comparison workspace is ready")).to_be_visible()
            expect(page.get_by_role("heading", name="Overall image drift")).to_have_count(0)
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_lv_web_02_img_01_real_dinov2_visible_streamlit_parity(tmp_path: Path, monkeypatch):
    """Compare the visible React result with the legacy whole-image Streamlit formula."""
    import json

    import numpy as np
    import pytest
    from sklearn.decomposition import PCA

    from _utils import load_model

    root = Path(__file__).resolve().parents[2]
    configured_models = Path(
        os.environ.get("LV_TEST_MODELS_DIR") or os.environ.get("LV_MODELS_DIR") or root / "models"
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
    expected_sha256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
    assert hashlib.sha256(source_weights.read_bytes()).hexdigest() == expected_sha256, \
        "the local model weights differ from the LV-WEB-01/06 baseline"

    models_dir = tmp_path / "models"
    model_dir = models_dir / "dinov2_vits14"
    model_dir.mkdir(parents=True)
    weights = model_dir / "dinov2_vits14.pth"
    try:
        os.link(source_weights, weights)
    except OSError:
        shutil.copyfile(source_weights, weights)
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == expected_sha256

    roots = [tmp_path / "group-a", tmp_path / "group-b"]
    source_files = []
    for side, root_dir, color in (("a", roots[0], (215, 36, 42)), ("b", roots[1], (28, 58, 212))):
        root_dir.mkdir()
        for index in range(4):
            image = Image.new("RGB", (80, 64), (color[0], min(255, color[1] + index * 9), color[2]))
            draw = ImageDraw.Draw(image)
            draw.rectangle((8 + index * 3, 10, 32 + index * 3, 42), fill=(240, 196, 34))
            draw.line((4, 56 - index, 74, 8 + index), fill=(30 + index * 20, 220, 130), width=2)
            path = root_dir / f"{side}-{index}.png"
            image.save(path)
            source_files.append(path)
    source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    monkeypatch.setenv("LV_MODELS_DIR", str(models_dir))
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))

    embed = load_model("dinov2_vits14", models_dir=models_dir)
    application = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=lambda model_id: load_model(model_id, models_dir=models_dir),
    )
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(
        application, host="127.0.0.1", port=port, log_level="error", lifespan="off",
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                import urllib.request

                with urllib.request.urlopen(base_url + "/api/v1/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("local worker did not start")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1050})
            page.set_default_timeout(300_000)
            comparison_results: list[dict] = []
            projection_results: list[dict] = []
            visible_image_bodies: dict[str, bytes] = {}

            def capture_response(response):
                if not response.ok:
                    return
                path = response.url.split("?", 1)[0]
                if response.request.method == "GET" and "/api/v1/comparisons/" in path \
                        and not path.endswith("/projections") and "/images/" not in path:
                    comparison_results.append(response.json())
                elif response.request.method == "POST" and path.endswith("/projections"):
                    projection_results.append(response.json())
                elif "/api/v1/comparisons/" in path and "/images/" in path:
                    image_parts = path.rsplit("/", 2)
                    visible_image_bodies[f"{image_parts[-2]}/{image_parts[-1]}"] = response.body()

            page.on("response", capture_response)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_role("button", name="Compare").click()
            page.get_by_label("Comparison unit").select_option("whole")
            for index, root_dir in enumerate(roots):
                page.get_by_label("Image folder", exact=True).nth(index).fill(str(root_dir))
                page.get_by_role("button", name=f"Add Folder {'A' if index == 0 else 'B'}").click()
            page.get_by_label("Feature model").select_option("dinov2_vits14")
            page.get_by_role("button", name="Run comparison").click()
            expect(page.get_by_test_id("compare-job-status")).to_contain_text("Completed", timeout=300_000)
            expect(page.get_by_role("heading", name="Overall image drift")).to_be_visible()
            expect(page.get_by_test_id("comparison-point")).to_have_count(8, timeout=300_000)
            page.get_by_label("Comparison projection", exact=True).select_option("pca")
            expect(page.get_by_test_id("comparison-point")).to_have_count(8, timeout=300_000)
            expect(page.locator(".compare-gallery-grid img").first).to_be_visible()
            page.get_by_test_id("comparison-point").nth(0).click()
            expect(page.get_by_test_id("comparison-focused-record")).to_contain_text("Group A")

            assert comparison_results and projection_results
            result = comparison_results[-1]
            projection = projection_results[-1]
            assert result["unit"] == "whole" and result["model"] == "dinov2_vits14"
            paths_a = sorted(roots[0].glob("*.png"))
            paths_b = sorted(roots[1].glob("*.png"))
            embeddings_a = np.stack([embed(path) for path in paths_a])
            embeddings_b = np.stack([embed(path) for path in paths_b])
            assert embeddings_a.shape == embeddings_b.shape == (4, 384)
            assert np.isfinite(embeddings_a).all() and np.isfinite(embeddings_b).all()

            # Match scripts/app.py::_compute_compare_whole using identical real embeddings.
            mean_a = embeddings_a.mean(axis=0)
            mean_a /= np.linalg.norm(mean_a) + 1e-9
            mean_b = embeddings_b.mean(axis=0)
            mean_b /= np.linalg.norm(mean_b) + 1e-9
            expected_drift = round(max(0.0, 1.0 - float(mean_a @ mean_b)), 4)
            assert result["rows"] == [{
                "label": "(全部影像)", "a_count": 4, "b_count": 4, "drift": expected_drift,
            }]
            expect(page.get_by_text(f"{expected_drift:.4f}", exact=True)).to_be_visible()
            expected_pca = PCA(n_components=2, random_state=42).fit_transform(
                np.vstack([embeddings_a, embeddings_b])
            )
            actual_pca = np.asarray([[point["x"], point["y"]] for point in projection["points"]])
            assert projection["requested_method"] == "pca"
            assert np.allclose(actual_pca, expected_pca, rtol=1e-5, atol=1e-5)
            assert len(result["records_a"]) == len(result["records_b"]) == 4
            public_payload = json.dumps({"comparison": result, "projection": projection})
            assert str(tmp_path) not in public_payload
            assert "a/0" in visible_image_bodies
            assert visible_image_bodies["a/0"] == paths_a[0].read_bytes()
            assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files} == source_hashes
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_lv_web_02_obj_01_real_dinov2_visible_streamlit_parity(tmp_path: Path, monkeypatch):
    """Match real DINOv2 object drift and PCA with the legacy Streamlit calculation."""
    import json

    import numpy as np
    import pytest
    from sklearn.decomposition import PCA

    from _utils import load_model

    repository_root = Path(__file__).resolve().parents[2]
    configured_models = Path(
        os.environ.get("LV_TEST_MODELS_DIR") or os.environ.get("LV_MODELS_DIR") or repository_root / "models"
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
    expected_sha256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
    assert hashlib.sha256(source_weights.read_bytes()).hexdigest() == expected_sha256, \
        "the local model weights differ from the LV-WEB-01/06 baseline"

    models_dir = tmp_path / "models"
    model_dir = models_dir / "dinov2_vits14"
    model_dir.mkdir(parents=True)
    weights = model_dir / "dinov2_vits14.pth"
    try:
        os.link(source_weights, weights)
    except OSError:
        shutil.copyfile(source_weights, weights)
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == expected_sha256

    roots = [tmp_path / "objects-a", tmp_path / "objects-b"]
    source_files = []
    for side, root_dir, color in (("a", roots[0], (210, 42, 34)), ("b", roots[1], (36, 54, 211))):
        (root_dir / "images").mkdir(parents=True)
        (root_dir / "labels").mkdir()
        (root_dir / "classes.txt").write_text("widget\n", encoding="utf-8")
        for index in range(2):
            image = Image.new("RGB", (96, 80), (color[0], min(255, color[1] + index * 13), color[2]))
            draw = ImageDraw.Draw(image)
            draw.rectangle((12 + index * 5, 13, 57 + index * 3, 61 - index * 4), fill=(239, 193, 41))
            draw.line((7, 70 - index * 3, 89, 9 + index * 2), fill=(32, 210, 126), width=3)
            path = root_dir / "images" / f"{side}-{index}.png"
            image.save(path)
            (root_dir / "labels" / f"{side}-{index}.txt").write_text(
                f"0 {0.45 + index * 0.1:.2f} 0.50 0.62 0.58\n", encoding="utf-8",
            )
            source_files.extend((path, root_dir / "labels" / f"{side}-{index}.txt"))
        source_files.append(root_dir / "classes.txt")
    source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    monkeypatch.setenv("LV_MODELS_DIR", str(models_dir))
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))

    policy = {"pad": 0.2, "target_res": 280, "head": "meanpool"}
    embed_object = load_model("dinov2_vits14", models_dir=models_dir, keep_aspect=True,
                              target_res=policy["target_res"], head=policy["head"])
    application = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=lambda model_id: load_model(model_id, models_dir=models_dir),
        object_model_loader=lambda model_id, **kwargs: load_model(model_id, models_dir=models_dir, **kwargs),
    )
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(
        application, host="127.0.0.1", port=port, log_level="error", lifespan="off",
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                import urllib.request

                with urllib.request.urlopen(base_url + "/api/v1/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("local worker did not start")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1050})
            page.set_default_timeout(300_000)
            comparison_results: list[dict] = []
            projection_results: list[dict] = []
            comparison_requests: list[dict] = []
            visible_image_bodies: dict[str, bytes] = {}

            def capture_response(response):
                if not response.ok:
                    return
                path = response.url.split("?", 1)[0]
                if response.request.method == "GET" and "/api/v1/comparisons/" in path \
                        and not path.endswith("/projections") and "/images/" not in path:
                    comparison_results.append(response.json())
                elif response.request.method == "POST" and path.endswith("/projections"):
                    projection_results.append(response.json())
                elif "/api/v1/comparisons/" in path and "/images/" in path:
                    image_parts = path.rsplit("/", 2)
                    visible_image_bodies[f"{image_parts[-2]}/{image_parts[-1]}"] = response.body()

            page.on("response", capture_response)
            page.on("request", lambda request: comparison_requests.append(json.loads(request.post_data or "{}"))
                    if request.method == "POST"
                    and request.url.rstrip("/").endswith("/api/v1/compare/jobs") else None)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_role("button", name="Compare").click()
            page.get_by_label("Image folder", exact=True).nth(0).fill(str(roots[0]))
            page.get_by_role("button", name="Add Folder A").click()
            page.get_by_label("Image folder", exact=True).nth(1).fill(str(roots[1]))
            page.get_by_role("button", name="Add Folder B").click()
            page.get_by_label("Feature model").select_option("dinov2_vits14")
            page.get_by_label("Object padding").fill("0.2")
            page.get_by_label("Object resolution").select_option("280")
            page.get_by_label("Object embedding head").select_option("meanpool")
            page.get_by_role("button", name="Apply manual settings").click()
            expect(page.get_by_text("Manual policy active: pad 0.2 · 280 · meanpool.")).to_be_visible()
            page.get_by_role("button", name="Run comparison").click()
            expect(page.get_by_test_id("compare-job-status")).to_contain_text("Completed", timeout=300_000)
            expect(page.get_by_role("heading", name="Drift by shared class")).to_be_visible()
            expect(page.get_by_test_id("compare-policy")).to_contain_text(
                "Manual object policy: pad 0.2 · resolution 280 · meanpool head",
            )
            expect(page.get_by_test_id("comparison-point")).to_have_count(4, timeout=300_000)
            page.get_by_label("Comparison projection", exact=True).select_option("pca")
            expect(page.get_by_test_id("comparison-point")).to_have_count(4, timeout=300_000)
            page.get_by_test_id("comparison-point").nth(0).click()
            expect(page.get_by_test_id("comparison-focused-record")).to_contain_text("Group A")

            assert comparison_results and projection_results and comparison_requests
            result = comparison_results[-1]
            projection = projection_results[-1]
            assert result["unit"] == "object" and result["model"] == "dinov2_vits14"
            assert result["policy"] == policy and result["policy_source"] == "manual"
            assert comparison_requests[-1]["object_policy"] == policy

            from interaction import crop_bbox, discover_yolo_objects
            from object_eval import _adaptive_pad_px

            def legacy_embeddings(root_dir: Path) -> np.ndarray:
                image_paths = sorted((root_dir / "images").glob("*.png"))
                metadata = discover_yolo_objects(image_paths, ["widget"])
                vectors = []
                for item in metadata:
                    with Image.open(item["image_path"]) as source:
                        original = source.convert("RGB")
                    bbox = item["bbox"]
                    pad_px = _adaptive_pad_px(bbox, original.width, original.height, policy["pad"])
                    crop = crop_bbox(original, *bbox, pad_px=pad_px)
                    vectors.append(embed_object(crop))
                return np.stack(vectors)

            embeddings_a = legacy_embeddings(roots[0])
            embeddings_b = legacy_embeddings(roots[1])
            assert embeddings_a.shape == embeddings_b.shape == (2, 384)
            assert np.isfinite(embeddings_a).all() and np.isfinite(embeddings_b).all()
            mean_a = embeddings_a.mean(axis=0)
            mean_a /= np.linalg.norm(mean_a) + 1e-9
            mean_b = embeddings_b.mean(axis=0)
            mean_b /= np.linalg.norm(mean_b) + 1e-9
            expected_drift = round(max(0.0, 1.0 - float(mean_a @ mean_b)), 4)
            assert result["rows"] == [{
                "label": "widget", "a_count": 2, "b_count": 2, "drift": expected_drift,
            }]
            expect(page.get_by_text(f"{expected_drift:.4f}", exact=True)).to_be_visible()
            expected_pca = PCA(n_components=2, random_state=42).fit_transform(
                np.vstack([embeddings_a, embeddings_b])
            )
            actual_pca = np.asarray([[point["x"], point["y"]] for point in projection["points"]])
            assert projection["requested_method"] == "pca"
            assert np.allclose(actual_pca, expected_pca, rtol=1e-5, atol=1e-5)
            assert len(result["records_a"]) == len(result["records_b"]) == 2
            assert all(record["bbox"] == [0.45 + index * 0.1, 0.5, 0.62, 0.58]
                       for index, record in enumerate(result["records_a"]))
            public_payload = json.dumps({"comparison": result, "projection": projection})
            assert str(tmp_path) not in public_payload
            assert "a/0" in visible_image_bodies
            assert visible_image_bodies["a/0"] == sorted((roots[0] / "images").glob("*.png"))[0].read_bytes()
            assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files} == source_hashes
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_lv_web_02_obj_01_react_object_compare_flow_with_local_worker(tmp_path: Path, monkeypatch):
    roots = [tmp_path / "det-a", tmp_path / "det-b"]
    colors = [(210, 40, 35), (35, 50, 210)]
    for root, color in zip(roots, colors):
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        (root / "classes.txt").write_text("widget\na_only\nb_only\n", encoding="utf-8")
        Image.new("RGB", (64, 56), color).save(root / "images" / "item.png")
        unmatched_label = "1" if root == roots[0] else "2"
        (root / "labels" / "item.txt").write_text(
            f"0 0.5 0.5 0.7 0.6\n{unmatched_label} 0.25 0.25 0.2 0.2\n",
            encoding="utf-8",
        )
    source_files = [path for root in roots for path in root.rglob("*") if path.is_file()]
    hashes_before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: _embed,
        object_model_loader=lambda _model, **_kwargs: _embed_object,
    )
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(application, host="127.0.0.1", port=port, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                import urllib.request

                with urllib.request.urlopen(base_url + "/api/v1/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("local worker did not start")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(base_url, wait_until="networkidle")
            page.get_by_role("button", name="Compare").click()
            page.get_by_label("Image folder", exact=True).nth(0).fill(str(roots[0]))
            page.get_by_role("button", name="Add Folder A").click()
            page.get_by_label("Image folder", exact=True).nth(1).fill(str(roots[1]))
            page.get_by_role("button", name="Add Folder B").click()
            page.get_by_label("Feature model").select_option("fixture-model")
            page.get_by_role("button", name="Run comparison").click()
            expect(page.get_by_test_id("compare-job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_role("heading", name="Drift by shared class")).to_be_visible()
            warning = page.locator(".alert-warning")
            expect(warning).to_contain_text("a_only (1)")
            expect(warning).to_contain_text("b_only (1)")
            expect(page.get_by_role("button", name="widget", exact=True)).to_be_visible()
            expect(page.get_by_label("Comparison class", exact=True)).to_have_value("widget")
            expect(page.get_by_test_id("compare-policy")).to_contain_text("Automatic object policy")
            expect(page.get_by_test_id("autotune-results")).to_be_visible()
            expect(page.get_by_test_id("comparison-point")).to_have_count(2, timeout=60_000)
            first_point = page.get_by_test_id("comparison-point").nth(0)
            first_point.focus()
            first_point.press("Enter")
            expect(page.get_by_test_id("comparison-focused-record")).to_be_visible()
            page.get_by_label("Object padding").fill("0.2")
            page.get_by_label("Object resolution").select_option("280")
            page.get_by_label("Object embedding head").select_option("meanpool")
            page.get_by_role("button", name="Apply manual settings").click()
            expect(page.get_by_role("heading", name="Your comparison workspace is ready")).to_be_visible()
            expect(page.get_by_test_id("comparison-focused-record")).to_have_count(0)
            expect(page.get_by_test_id("comparison-point")).to_have_count(0)
            page.get_by_role("button", name="Run comparison").click()
            expect(page.get_by_test_id("compare-job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("compare-policy")).to_contain_text("Manual object policy: pad 0.2 · resolution 280 · meanpool head")
            page.get_by_label("Comparison projection", exact=True).select_option("pca")
            expect(page.get_by_test_id("comparison-point")).to_have_count(2, timeout=60_000)
            expect(page.locator(".compare-gallery-grid img")).to_have_count(2)
            expect(page.locator(".compare-image-canvas i")).to_have_count(2)
            first_point = page.get_by_test_id("comparison-point").nth(0)
            first_point.focus()
            first_point.press("Enter")
            expect(page.get_by_test_id("comparison-focused-record")).to_contain_text("Box")
            browser.close()
        assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files} == hashes_before
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_lv_web_02_obj_01_no_shared_classes_shows_visible_error(tmp_path: Path, monkeypatch):
    roots = [tmp_path / "det-a", tmp_path / "det-b"]
    for root, label in zip(roots, ("alpha", "beta")):
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        (root / "classes.txt").write_text(f"{label}\n", encoding="utf-8")
        Image.new("RGB", (64, 56), (70, 100, 150)).save(root / "images" / "item.png")
        (root / "labels" / "item.txt").write_text("0 0.5 0.5 0.7 0.6\n", encoding="utf-8")
    source_files = [path for root in roots for path in root.rglob("*") if path.is_file()]
    hashes_before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: _embed,
        object_model_loader=lambda _model, **_kwargs: _embed_object,
    )
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(application, host="127.0.0.1", port=port, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                import urllib.request

                with urllib.request.urlopen(base_url + "/api/v1/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("local worker did not start")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(base_url, wait_until="networkidle")
            page.get_by_role("button", name="Compare").click()
            page.get_by_label("Image folder", exact=True).nth(0).fill(str(roots[0]))
            page.get_by_role("button", name="Add Folder A").click()
            page.get_by_label("Image folder", exact=True).nth(1).fill(str(roots[1]))
            page.get_by_role("button", name="Add Folder B").click()
            page.get_by_label("Feature model").select_option("fixture-model")
            page.get_by_role("button", name="Run comparison").click()
            alert = page.get_by_role("alert")
            expect(alert).to_contain_text("Groups A and B have no shared detection classes", timeout=60_000)
            expect(alert).to_contain_text("same classes.txt")
            expect(alert).not_to_contain_text(str(tmp_path))
            expect(page.get_by_test_id("comparison-point")).to_have_count(0)
            expect(page.get_by_test_id("compare-policy")).to_have_count(0)
            browser.close()
        assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files} == hashes_before
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_lv_web_02_state_01_filters_reset_selection_and_compare_images_handoff_visible_journey(
    tmp_path: Path, monkeypatch,
):
    roots = [tmp_path / "det-a", tmp_path / "det-b"]
    for index, root in enumerate(roots):
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        (root / "classes.txt").write_text("widget\ngear\n", encoding="utf-8")
        Image.new("RGB", (64, 56), (70 + index * 90, 60, 150)).save(root / "images" / f"group-{index}.png")
        label_rows = (
            ["0 0.3 0.4 0.4 0.5 0.40", "0 0.7 0.6 0.3 0.4 0.45", "1 0.5 0.5 0.2 0.3 0.30"]
            if index == 0 else
            ["0 0.3 0.4 0.4 0.5 0.40", "0 0.7 0.6 0.3 0.4 0.90", "1 0.5 0.5 0.2 0.3 0.30"]
        )
        (root / "labels" / f"group-{index}.txt").write_text("\n".join(label_rows) + "\n", encoding="utf-8")
    source_files = [path for root in roots for path in root.rglob("*") if path.is_file()]
    source_before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=lambda _model: _embed,
        object_model_loader=lambda _model, **_kwargs: _embed_object,
    )
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(application, host="127.0.0.1", port=port, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                import urllib.request

                with urllib.request.urlopen(base_url + "/api/v1/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("local worker did not start")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1050})
            page.set_default_timeout(15_000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_role("button", name="Compare").click()
            for side, root in enumerate(roots):
                page.get_by_label("Image folder", exact=True).nth(side).fill(str(root))
                page.get_by_role("button", name=f"Add Folder {'A' if side == 0 else 'B'}").click()
            page.get_by_label("Feature model").select_option("fixture-model")
            page.get_by_role("button", name="Run comparison").click()
            expect(page.get_by_test_id("compare-job-status")).to_contain_text("Completed", timeout=60_000)
            page.get_by_label("Comparison class").select_option(label="widget")
            points = page.get_by_test_id("comparison-point")
            expect(points).to_have_count(4, timeout=60_000)
            points.nth(0).click()
            expect(page.get_by_test_id("comparison-focused-record")).to_be_visible()
            page.get_by_label("Comparison class").select_option(label="gear")
            expect(points).to_have_count(2, timeout=60_000)
            expect(page.get_by_test_id("comparison-focused-record")).to_have_count(0)
            expect(page.locator(".clear-selection")).to_have_count(0)
            page.get_by_label("Comparison class").select_option(label="widget")
            expect(points).to_have_count(4, timeout=60_000)

            points.nth(0).click()
            expect(page.get_by_test_id("comparison-focused-record")).to_be_visible()
            page.get_by_label("Comparison projection", exact=True).select_option("pca")
            expect(points).to_have_count(4, timeout=60_000)
            expect(page.get_by_test_id("comparison-focused-record")).to_have_count(0)
            expect(page.locator(".clear-selection")).to_have_count(0)

            page.get_by_role("button", name="3D", exact=True).click()
            expect(points).to_have_count(4, timeout=60_000)
            expect(page.get_by_test_id("comparison-focused-record")).to_have_count(0)
            expect(page.locator(".clear-selection")).to_have_count(0)

            high_confidence = page.get_by_role("button", name="Select group-1.png from group B", exact=True).nth(1)
            high_confidence.focus()
            page.keyboard.press("Enter")
            expect(page.get_by_test_id("comparison-focused-record")).to_contain_text("group-1.png")
            expect(high_confidence).to_have_attribute("aria-pressed", "true")
            max_confidence = page.get_by_label("Maximum confidence")
            max_confidence.focus()
            for _ in range(10):
                max_confidence.press("ArrowLeft")
            expect(page.get_by_text("Confidence 0.00–0.50")).to_be_visible()
            expect(points).to_have_count(3, timeout=60_000)
            expect(page.get_by_test_id("comparison-focused-record")).to_have_count(0)
            expect(page.locator(".clear-selection")).to_have_count(0)
            expect(page.locator('.screen-reader-points button[aria-pressed="true"]')).to_have_count(0)

            page.get_by_role("button", name="2D", exact=True).click()
            expect(page.locator(".compare-scatter")).to_be_visible(timeout=60_000)
            expect(points).to_have_count(3, timeout=60_000)
            chart = page.locator(".compare-scatter").bounding_box()
            assert chart is not None
            page.mouse.move(chart["x"] + 2, chart["y"] + 2)
            page.mouse.down()
            page.mouse.move(chart["x"] + chart["width"] - 2, chart["y"] + chart["height"] - 2, steps=10)
            page.mouse.up()
            expect(page.get_by_role("heading", name="Selected records (3)")).to_be_visible()
            page.get_by_test_id("add-comparison-selection-to-curation").click()
            expect(page.locator(".alert-info")).to_contain_text(
                "Added 2 image(s) to curation; skipped 0 duplicate(s)."
            )
            expect(page.get_by_test_id("open-curation")).to_contain_text("(2)")

            curation = page.request.get(base_url + "/api/v1/curation/items").json()
            assert curation["total"] == 2
            assert [(item["filename"], item["source"], item["level"]) for item in curation["items"]] == [
                ("group-0.png", "compare", "image"),
                ("group-1.png", "compare", "image"),
            ]
            assert all(not {"path", "image_path", "bbox", "obj_index"} & item.keys() for item in curation["items"])

            page.get_by_test_id("open-curation").click()
            expect(page.get_by_test_id("curation-item")).to_have_count(2)
            expect(page.locator(".curation-item-heading strong").nth(0)).to_have_text("group-0.png")
            expect(page.locator(".curation-item-heading strong").nth(1)).to_have_text("group-1.png")
            expect(page.locator(".curation-item-meta").nth(0)).to_contain_text("widget · Unsplit · compare")
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files} == source_before
