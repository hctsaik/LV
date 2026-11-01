"""Visible browser workflow for the local Compare Distributions page."""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import uvicorn
from PIL import Image
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


def test_lv_web_02_obj_01_react_object_compare_flow_with_local_worker(tmp_path: Path, monkeypatch):
    roots = [tmp_path / "det-a", tmp_path / "det-b"]
    colors = [(210, 40, 35), (35, 50, 210)]
    for root, color in zip(roots, colors):
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        (root / "classes.txt").write_text("widget\n", encoding="utf-8")
        Image.new("RGB", (64, 56), color).save(root / "images" / "item.png")
        (root / "labels" / "item.txt").write_text("0 0.5 0.5 0.7 0.6\n", encoding="utf-8")
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
            expect(page.get_by_test_id("compare-policy")).to_contain_text("Automatic object policy")
            expect(page.get_by_test_id("autotune-results")).to_be_visible()
            page.get_by_label("Object padding").fill("0.2")
            page.get_by_label("Object resolution").select_option("280")
            page.get_by_label("Object embedding head").select_option("meanpool")
            page.get_by_role("button", name="Apply manual settings").click()
            expect(page.get_by_role("heading", name="Your comparison workspace is ready")).to_be_visible()
            page.get_by_role("button", name="Run comparison").click()
            expect(page.get_by_test_id("compare-job-status")).to_contain_text("Completed", timeout=60_000)
            expect(page.get_by_test_id("compare-policy")).to_contain_text("Manual object policy: pad 0.2 · resolution 280 · meanpool head")
            page.get_by_label("Comparison projection", exact=True).select_option("pca")
            expect(page.get_by_test_id("comparison-point")).to_have_count(2, timeout=60_000)
            expect(page.locator(".compare-gallery-grid img")).to_have_count(2)
            expect(page.locator(".compare-image-canvas i")).to_have_count(2)
            page.get_by_test_id("comparison-point").nth(0).click()
            expect(page.get_by_test_id("comparison-focused-record")).to_contain_text("Box")
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
