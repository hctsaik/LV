"""Visible React journey for planted and null Good/Bad patch-difference cases."""
from __future__ import annotations

import hashlib
import re
import shutil
import socket
import threading
import time
from pathlib import Path

import numpy as np
import uvicorn
from PIL import Image
from playwright.sync_api import expect, sync_playwright

from react_web_shell import create_app


D = 16
SHIFT_PATCH = 5


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_images(folder: Path, count: int, *, marker: str | None = None, seed0: int = 0) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        rng = np.random.default_rng(seed0 + index)
        pixels = rng.integers(90, 160, (56, 56, 3)).astype("uint8")
        if marker == "bad":
            pixels[:8, :8] = 0
        Image.fromarray(pixels).save(folder / f"img_{seed0 + index:03d}.png")


def _patch_extractor():
    rng = np.random.default_rng(999)
    base = rng.normal(size=(16, D))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    shift = rng.normal(size=D)
    shift /= np.linalg.norm(shift)

    def extract(image: Image.Image):
        pixels = np.asarray(image.convert("RGB"), dtype=float)
        marker = pixels[:16, :16].mean()
        sample_rng = np.random.default_rng(int(pixels.sum()) % (2 ** 32))
        features = base + sample_rng.normal(scale=0.15, size=(16, D))
        if marker < 30:
            features[SHIFT_PATCH] += 1.2 * shift
        features /= np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None)
        return features.astype(np.float32), (4, 4)

    return extract


def _snapshot(*folders: Path) -> dict[str, tuple[str, int]]:
    result = {}
    for folder in folders:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                result[str(path)] = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    return result


def _run_visible(page, good: Path, bad: Path, expected: str) -> None:
    inputs = page.get_by_label("資料夾路徑")
    inputs.nth(0).fill(str(good))
    inputs.nth(0).press("Enter")
    inputs.nth(1).fill(str(bad))
    inputs.nth(1).press("Enter")
    page.get_by_test_id("groupdiff-run").click()
    expect(page.get_by_test_id("groupdiff-verdict")).to_contain_text(expected, timeout=60_000)


def test_lv_web_08_diff_01_visible_signal_null_heatmap_regions_and_export(tmp_path: Path, monkeypatch):
    good, bad = tmp_path / "good", tmp_path / "bad"
    _make_images(good, 8, seed0=0)
    _make_images(bad, 8, marker="bad", seed0=100)
    all_null = tmp_path / "all-null"
    null_a, null_b = tmp_path / "null-a", tmp_path / "null-b"
    _make_images(all_null, 8, seed0=300)
    null_a.mkdir()
    null_b.mkdir()
    for image in sorted(all_null.glob("*.png")):
        shutil.copy2(image, null_a / image.name)
        shutil.copy2(image, null_b / image.name)
    source_before = _snapshot(good, bad, null_a, null_b)
    report_dir = tmp_path / "report"
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))

    application = create_app(
        model_catalog=lambda: ["dinov2_vits14"],
        model_loader=lambda _model: None,
        groupdiff_extractor=_patch_extractor(),
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
            page.get_by_test_id("open-groupdiff").click()
            page.get_by_label("特徵模型").select_option("dinov2_vits14")
            _run_visible(page, good, bad, "兩群存在穩定的局部差異")
            expect(page.get_by_test_id("groupdiff-top-region")).to_be_visible()
            expect(page.get_by_test_id("groupdiff-heatmap")).to_be_visible()
            page.wait_for_function("""() => { const image = document.querySelector('[data-testid="groupdiff-heatmap"]'); return image && image.complete && image.naturalWidth > 0; }""")
            expect(page.get_by_test_id("groupdiff-good-region-image").first).to_be_visible()
            expect(page.get_by_test_id("groupdiff-bad-region-image").first).to_be_visible()
            page.get_by_label("熱圖底圖").select_option("good:0")
            alpha = page.get_by_label("熱力圖濃度")
            alpha.focus()
            alpha.press("ArrowRight")
            expect(page.get_by_test_id("groupdiff-heatmap")).to_have_attribute("src", re.compile(r"alpha=0\.6"))
            width = page.get_by_label("顯示寬度")
            width.focus()
            width.press("ArrowRight")
            assert page.locator(".groupdiff-heat-images").evaluate(
                "element => getComputedStyle(element).getPropertyValue('--groupdiff-width').trim()"
            ) == "660px"
            page.get_by_text("📎 整體旁證（PCA，非監督）").click()
            expect(page.get_by_test_id("groupdiff-pca")).to_be_visible()

            page.get_by_label("報告輸出資料夾").fill(str(report_dir))
            page.get_by_role("button", name="📦 匯出報告").click()
            expect(page.get_by_text("heatmap_good.png", exact=False)).to_be_visible(timeout=30_000)
            assert (report_dir / "regions.csv").is_file()
            assert (report_dir / "region_01.png").stat().st_size > 0

            for path in (good, bad):
                page.get_by_role("button", name=f"Remove {path}").click()
            _run_visible(page, null_a, null_b, "未發現穩定差異")
            expect(page.get_by_test_id("groupdiff-verdict")).not_to_contain_text("兩群存在穩定的局部差異")
            assert page.get_by_test_id("groupdiff-heatmap-details").evaluate("element => element.open") is False
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    assert _snapshot(good, bad, null_a, null_b) == source_before
