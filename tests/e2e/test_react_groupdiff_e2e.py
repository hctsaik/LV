"""Visible React journey for planted and null Good/Bad patch-difference cases."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import socket
import threading
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pytest
import uvicorn
from PIL import Image
from playwright.sync_api import expect, sync_playwright

from react_web_shell import create_app


ROOT = Path(__file__).resolve().parents[2]
DINO_MODEL_ID = "dinov2_vits14"
DINO_WEIGHT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
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


def _make_streamlit_groupdiff_fixture(folder: Path, count: int, *, defect: bool, seed0: int) -> None:
    """Reproduce the random 224px images used by the original Streamlit E2E."""
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        rng = np.random.default_rng(seed0 + index)
        pixels = rng.integers(110, 150, (224, 224, 3)).astype("uint8")
        pixels[::28, :, :] = 165
        pixels[:, ::28, :] = 165
        if defect:
            block = rng.integers(20, 45, (56, 56, 3)).astype("uint8")
            block[..., 0] = np.clip(block[..., 0].astype(int) + 30, 0, 255).astype("uint8")
            pixels[56:112, 56:112] = block
        Image.fromarray(pixels).save(folder / f"{folder.name}_{index:02d}.png")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pinned_dino_weights() -> tuple[Path, Path]:
    configured = Path(
        os.environ.get("LV_TEST_MODELS_DIR")
        or os.environ.get("LV_MODELS_DIR")
        or ROOT / "models"
    ).expanduser()
    candidates = [
        configured / DINO_MODEL_ID / f"{DINO_MODEL_ID}.pth",
        configured / f"{DINO_MODEL_ID}.pth",
        Path(r"C:\code\claude\LV\visuallatent\models") / DINO_MODEL_ID / f"{DINO_MODEL_ID}.pth",
    ]
    weights = next((path.resolve() for path in candidates if path.is_file()), None)
    if weights is None:
        pytest.skip("the pinned DINOv2-small weights are unavailable")
    assert _sha256(weights) == DINO_WEIGHT_SHA256, "DINOv2 weights differ from the pinned Streamlit baseline"
    models_dir = weights.parent.parent if weights.parent.name == DINO_MODEL_ID else weights.parent
    return models_dir, weights


def _assert_streamlit_result_parity(baseline: dict, visible_result: dict) -> None:
    assert visible_result["grid"] == list(baseline["grid"])
    assert visible_result["verdict"] is baseline["verdict"]
    assert visible_result["n_good"] == baseline["n_good"]
    assert visible_result["n_bad"] == baseline["n_bad"]
    assert visible_result["skipped"] == baseline["skipped"]
    assert visible_result["alignment_warning"] is baseline["alignment_warning"]
    assert visible_result["warn_msgs"] == baseline["warn_msgs"]
    assert visible_result["meta"] == baseline["meta"]
    assert visible_result["bases_good"] == [Path(path).name for path in baseline["good_paths"]]
    assert visible_result["bases_bad"] == [Path(path).name for path in baseline["bad_paths"]]
    assert visible_result["base_good_index"] == baseline["good_paths"].index(baseline["base_good"])
    assert visible_result["base_bad_index"] == baseline["bad_paths"].index(baseline["base_bad"])
    assert visible_result["p_global"] == pytest.approx(baseline["p_global"], abs=1e-12)
    assert visible_result["threshold_z"] == pytest.approx(baseline["threshold_z"], abs=1e-12)
    for key in ("score_map", "z_map", "stability_map"):
        np.testing.assert_allclose(visible_result[key], baseline[key], rtol=1e-7, atol=1e-7)

    assert len(visible_result["regions"]) == len(baseline["regions"])
    for expected, actual in zip(baseline["regions"], visible_result["regions"], strict=True):
        for key in ("rank", "text", "pct"):
            assert actual[key] == expected[key]
        for key in ("peak_z", "mean_z", "stability"):
            assert actual[key] == pytest.approx(expected[key], abs=1e-7)
        np.testing.assert_allclose(actual["bbox_px"], expected["bbox_px"], rtol=0, atol=1e-12)
        assert actual["good_count"] == len(expected["good_paths"])
        assert actual["bad_count"] == len(expected["bad_paths"])


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
    server = uvicorn.Server(
        uvicorn.Config(
            application, host="127.0.0.1", port=port, log_level="error", lifespan="off"
        )
    )
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
            page.wait_for_function(
                """() => {
                    const image = document.querySelector('[data-testid="groupdiff-heatmap"]');
                    return image && image.complete && image.naturalWidth > 0;
                }"""
            )
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


def test_lv_web_08_diff_01_real_dino_visible_streamlit_parity(tmp_path: Path, monkeypatch):
    """Compare visible React output with Streamlit's original fixtures and DINO weights."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    models_dir, weights = _pinned_dino_weights()

    import _utils
    from groupdiff_pipeline import run_groupdiff

    monkeypatch.setattr(_utils, "_DEFAULT_MODELS_DIR", models_dir)
    root = tmp_path / "streamlit-groupdiff"
    good, bad = root / "good", root / "bad"
    null_a, null_b = root / "null_a", root / "null_b"
    _make_streamlit_groupdiff_fixture(good, 16, defect=False, seed0=0)
    _make_streamlit_groupdiff_fixture(bad, 12, defect=True, seed0=100)
    _make_streamlit_groupdiff_fixture(null_a, 8, defect=False, seed0=300)
    _make_streamlit_groupdiff_fixture(null_b, 8, defect=False, seed0=400)
    source_before = _snapshot(good, bad, null_a, null_b)
    weights_hash = _sha256(weights)

    options = {"model": DINO_MODEL_ID, "n_perm": 200, "alpha": 0.05, "top_k": 5, "seed": 0}
    streamlit_planted = run_groupdiff(good, bad, cache_root=tmp_path / "streamlit-cache", **options)
    streamlit_null = run_groupdiff(
        null_a, null_b, cache_root=tmp_path / "streamlit-cache", **options
    )
    assert streamlit_planted["verdict"] is True and streamlit_planted["regions"]
    assert streamlit_null["verdict"] is False

    application = create_app(model_catalog=lambda: [DINO_MODEL_ID])
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            application, host="127.0.0.1", port=port, log_level="error", lifespan="off"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                with urlopen(base_url + "/api/v1/health", timeout=1) as response:
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
            visible_runs: list[dict] = []

            def capture_run(response):
                if response.status == 200 and re.search(r"/api/v1/groupdiff/runs/[^/?]+$", response.url):
                    visible_runs.append(response.json())

            page.on("response", capture_run)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("open-groupdiff").click()
            page.get_by_label("特徵模型").select_option(DINO_MODEL_ID)
            _run_visible(page, good, bad, "兩群存在穩定的局部差異")
            expect(page.get_by_test_id("groupdiff-top-region")).to_be_visible()
            expect(page.get_by_test_id("groupdiff-heatmap")).to_be_visible()
            page.wait_for_function(
                """() => {
                    const image = document.querySelector('[data-testid="groupdiff-heatmap"]');
                    return image && image.complete && image.naturalWidth > 0;
                }"""
            )
            assert visible_runs, "React UI did not read back the completed planted result"
            _assert_streamlit_result_parity(streamlit_planted, visible_runs[-1])
            assert page.get_by_test_id("groupdiff-heatmap-details").evaluate(
                "element => element.open"
            ) is True

            for path in (good, bad):
                page.get_by_role("button", name=f"Remove {path}").click()
            _run_visible(page, null_a, null_b, "未發現穩定差異")
            expect(page.get_by_test_id("groupdiff-verdict")).not_to_contain_text("兩群存在穩定的局部差異")
            assert page.get_by_test_id("groupdiff-heatmap-details").evaluate(
                "element => element.open"
            ) is False
            assert len(visible_runs) >= 2, "React UI did not read back the completed null result"
            _assert_streamlit_result_parity(streamlit_null, visible_runs[-1])
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    assert _snapshot(good, bad, null_a, null_b) == source_before
    assert _sha256(weights) == weights_hash
