from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from react_web_shell import create_app
from .test_react_web_shell_e2e import _free_port, _start_api, _worker_fixture_model

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]


def _label_quality_fixture(root: Path) -> Path:
    train = root / "train"
    specs = [
        ("apple", (220, 35, 45)),
        ("apple", (218, 40, 48)),
        ("apple", (225, 30, 43)),
        ("apple", (215, 45, 50)),
        # Deliberately mislabeled: its image features match the banana group.
        ("apple", (30, 190, 70)),
        ("banana", (32, 188, 72)),
        ("banana", (28, 194, 68)),
        ("banana", (35, 186, 75)),
        ("banana", (30, 192, 65)),
    ]
    for index, (label, color) in enumerate(specs):
        folder = train / label
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color).save(folder / f"img-{index:02}.png")
    return train


def _source_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def test_lv_web_01_lq_01_visible_confusion_and_mislabel_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The React label-quality screen finds a planted label issue from raw features."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the React workspace with npm run build before this E2E")

    train = _label_quality_fixture(tmp_path / "label-quality-fixture")
    before = _source_hashes(train)
    monkeypatch.setenv("LV_CACHE_DIR", str(tmp_path / "cache"))
    application = create_app(
        model_catalog=lambda: ["fixture-model"],
        model_loader=_worker_fixture_model,
    )
    server = server_thread = None
    try:
        server, server_thread, base_url = _start_api(application)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1450, "height": 1050})
            page.set_default_timeout(30000)
            page.goto(base_url, wait_until="networkidle")
            page.get_by_test_id("model-choice-fixture-model").check()
            page.get_by_label("Supervised UMAP").uncheck()
            page.get_by_label("PCA").check()
            page.get_by_label("Image folder").fill(str(train))
            page.get_by_role("button", name="Add folder").click()
            page.get_by_role("button", name="Run analysis").click()
            expect(page.get_by_test_id("job-status")).to_contain_text("Completed", timeout=30000)
            expect(page.get_by_test_id("embedding-point")).to_have_count(9)

            page.get_by_test_id("open-label-quality").click()
            expect(page.get_by_test_id("label-quality-workspace")).to_be_visible()
            page.get_by_label("Label quality K").fill("5")
            page.get_by_test_id("label-quality-calculate").click()
            expect(page.get_by_test_id("label-quality-confusion-matrix")).to_be_visible()
            expect(page.get_by_test_id("label-quality-summary")).to_contain_text("1 suspected label issue")
            candidate = page.get_by_test_id("label-quality-candidate")
            expect(candidate).to_have_count(1)
            expect(candidate).to_contain_text("apple → banana")
            expect(candidate.get_by_role("img", name="img-04.png")).to_be_visible()

            page.get_by_test_id("label-quality-select-candidates").click()
            expect(page.get_by_role("status")).to_contain_text("Added 1 suspected record(s)")
            page.get_by_role("button", name="Return to Visualize").click()
            expect(page.get_by_test_id("selection-count")).to_have_text("1 selected")
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)

    assert _source_hashes(train) == before
