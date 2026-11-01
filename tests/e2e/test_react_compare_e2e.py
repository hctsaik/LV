"""Visible browser workflow for the local Compare Distributions page."""
from __future__ import annotations

import hashlib
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
