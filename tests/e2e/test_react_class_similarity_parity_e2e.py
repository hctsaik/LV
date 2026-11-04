from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from react_web_shell import create_app
from .test_react_web_shell_e2e import (
    _pinned_dinov2_models_dir,
    _start_api,
    _start_streamlit_for_react_parity,
)

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
STREAMLIT_REFERENCE_COMMIT = "bf246f266b5f299b962e91d0e358c185361cd8bb"


def _fixture(root: Path) -> Path:
    colors = [(220, 35, 45), (30, 190, 70), (35, 65, 215), (210, 180, 25)]
    for index, color in enumerate(colors):
        split = "test" if index == 3 else "train"
        label = "apple" if index < 2 else "banana"
        folder = root / split / label
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color).save(folder / f"image-{index + 1:02}.png")
    return root


def _source_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def _create_streamlit_reference_worktree(path: Path) -> Path:
    resolved = path.resolve()
    available = subprocess.run(
        ["git", "cat-file", "-e", f"{STREAMLIT_REFERENCE_COMMIT}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if available.returncode:
        pytest.skip(
            "fetch LV source commit "
            f"{STREAMLIT_REFERENCE_COMMIT} to run the Streamlit comparison"
        )
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(resolved), STREAMLIT_REFERENCE_COMMIT],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return resolved


def _remove_streamlit_reference_worktree(path: Path, *, temp_root: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != temp_root.resolve() or resolved.name != "streamlit-reference":
        raise ValueError(f"refusing to remove unexpected reference worktree: {resolved}")
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(resolved)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def test_lv_web_01_cs_01_visible_react_matches_pinned_streamlit_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Compare React with the checked-in Streamlit source snapshot on one real-model run."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("build the React workspace with npm run build before this E2E")
    try:
        import hnswlib  # noqa: F401 - required by the historical Streamlit app imports
    except ImportError:
        pytest.skip("install chroma-hnswlib from the historical LV requirements to run Streamlit")

    from _utils import available_models, load_model
    from .conftest import (
        _add_folder,
        _ensure_sidebar,
        _stop_process_tree,
        load_app,
        wait_idle,
    )
    from .test_react_web_shell_e2e import _is_run_summary_response

    models_dir = _pinned_dinov2_models_dir(tmp_path)
    dataset = _fixture(tmp_path / "class-similarity-parity-fixture")
    source_before = _source_hashes(dataset)
    cache_dir = tmp_path / "shared-cache"
    monkeypatch.setenv("LV_CACHE_DIR", str(cache_dir))
    application = create_app(
        model_catalog=lambda: available_models(models_dir),
        model_loader=lambda model_id: load_model(model_id, models_dir=models_dir),
    )

    legacy_root = _create_streamlit_reference_worktree(tmp_path / "streamlit-reference")
    server = server_thread = legacy_process = legacy_log = None
    try:
        server, server_thread, base_url = _start_api(application)
        legacy_process, legacy_log, legacy_url = _start_streamlit_for_react_parity(
            models_dir,
            tmp_path / "streamlit-run",
            cache_dir=cache_dir,
            source_root=legacy_root,
        )

        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            react_page = browser.new_page(viewport={"width": 1500, "height": 1050})
            react_page.set_default_timeout(30000)
            run_ids = []

            def capture_run(response):
                if _is_run_summary_response(response) and response.ok:
                    run_ids.append(response.url.rstrip("/").rsplit("/", 1)[-1])

            react_page.on("response", capture_run)
            react_page.goto(base_url, wait_until="networkidle")
            react_page.get_by_test_id("model-choice-dinov2_vits14").check()
            react_page.get_by_label("Supervised UMAP").uncheck()
            react_page.get_by_label("PCA").check()
            for split in ("train", "test"):
                folder = dataset / split
                react_page.get_by_label("Image folder").fill(str(folder))
                react_page.get_by_role("button", name="Add folder").click()
            react_page.get_by_role("button", name="Run analysis").click()
            expect(react_page.get_by_test_id("job-status")).to_contain_text(
                "Completed", timeout=300000,
            )
            expect(react_page.get_by_test_id("embedding-point")).to_have_count(4)
            assert run_ids, "React did not produce a Visualize run"
            run_id = run_ids[-1]

            react_page.get_by_test_id("open-class-similarity").click()
            react_page.get_by_role("button", name="Calculate class similarity").click()
            expect(react_page.get_by_test_id("class-similarity-summary")).to_contain_text(
                "eligible", timeout=30000,
            )
            expect(react_page.get_by_test_id("class-similarity-row")).to_have_count(4)
            with application.state.lock:
                jobs = [
                    job for job in application.state.class_similarity_jobs.values()
                    if job.run_id == run_id
                ]
            assert len(jobs) == 1
            prepared = jobs[0].prepared
            analysis_items = jobs[0].analysis.items
            assert prepared.source_verified is True
            assert len(analysis_items) == 4

            react_rows = {}
            for row in react_page.get_by_test_id("class-similarity-row").all():
                cells = row.locator("td").all_text_contents()
                react_rows[cells[0]] = cells
            record_by_id = {
                item.record_id: prepared.records[item.input_index]
                for item in analysis_items
            }
            for item in analysis_items:
                record = record_by_id[item.record_id]
                filename = Path(str(record["path"])).name
                expected_cells = react_rows[filename]
                assert expected_cells[1] == (item.label or "—")
                assert expected_cells[2] == f"{item.same:.4f}"
                assert expected_cells[3] == f"{item.other:.4f}"
                assert expected_cells[4] == f"{item.margin:.4f}"

            legacy_page = browser.new_page(viewport={"width": 1920, "height": 1080})
            legacy_page.set_default_timeout(30000)
            load_app(legacy_page, legacy_url)
            legacy_page.locator(".st-key-viz_mode").get_by_text(
                "Image Classifier", exact=True,
            ).click()
            wait_idle(legacy_page)
            for split in ("train", "test"):
                _add_folder(legacy_page, "viz_folder_list", str(dataset / split))
            methods = legacy_page.locator(".st-key-viz_methods")
            methods.get_by_role("button", name="Clear all").click()
            wait_idle(legacy_page)
            methods.locator("[data-baseweb='select']").click()
            legacy_page.get_by_role("option", name="PCA", exact=True).click()
            legacy_page.keyboard.press("Escape")
            wait_idle(legacy_page)
            _ensure_sidebar(legacy_page)
            legacy_page.locator(".st-key-run_viz button").click()
            legacy_page.wait_for_selector(
                ".st-key-viz_scatter_wrap g.points path", timeout=300000,
            )
            wait_idle(legacy_page, timeout=60000)
            summary = legacy_page.locator("details summary", has_text="標錯偵測").first
            summary.scroll_into_view_if_needed()
            if not summary.evaluate("element => element.parentElement.open"):
                summary.click()
            legacy_page.locator(".st-key-cs_panel_mode").get_by_text(
                "類別相似度分布", exact=True,
            ).last.click()
            wait_idle(legacy_page)
            legacy_page.get_by_role("button", name="計算類別相似度").last.click()
            expect(legacy_page.get_by_text("分析母體：", exact=False).first).to_be_visible(
                timeout=120000,
            )
            expect(legacy_page.get_by_label("K（Top-K 平均）")).to_have_value("5")
            assert int(
                legacy_page.locator('[data-testid="stMetric"]').filter(
                    has_text="分析總數",
                ).first.locator('[data-testid="stMetricValue"]').inner_text()
            ) == 4

            for item in analysis_items:
                row_id = item.record_id[:12] + "…"
                id_cell = legacy_page.get_by_text(row_id, exact=True).last
                legacy_row = id_cell.locator(
                    "xpath=ancestor::div[@data-testid='stHorizontalBlock'][1]",
                )
                legacy_text = legacy_row.inner_text()
                assert f"{item.same:.4f}" in legacy_text
                assert item.rival_class in legacy_text
                assert f"{item.other:.4f}" in legacy_text
                assert f"{item.margin:.4f}" in legacy_text

            chosen = analysis_items[0]
            chosen_record = record_by_id[chosen.record_id]
            chosen_filename = Path(str(chosen_record["path"])).name
            react_row = react_page.get_by_test_id("class-similarity-row").filter(
                has_text=chosen_filename,
            )
            react_row.get_by_role("button", name=chosen_filename).click()
            react_evidence = react_page.locator(".class-similarity-evidence")
            expect(react_evidence).to_contain_text("Same-class neighbors")
            expect(react_evidence).to_contain_text("Rival-class neighbors")

            legacy_row_id = legacy_page.get_by_text(
                chosen.record_id[:12] + "…", exact=True,
            ).last
            legacy_row = legacy_row_id.locator(
                "xpath=ancestor::div[@data-testid='stHorizontalBlock'][1]",
            )
            legacy_row.get_by_role("button", name="證據").click()
            wait_idle(legacy_page)
            summary = legacy_page.locator("details summary", has_text="標錯偵測").first
            if not summary.evaluate("element => element.parentElement.open"):
                summary.click()
                wait_idle(legacy_page)
            expect(legacy_page.get_by_text("分析母體：", exact=False).last).to_be_visible()
            expect(legacy_page.get_by_text("同類 Top-K", exact=False).last).to_be_visible()
            legacy_evidence_text = legacy_page.locator("[data-testid='stMain']").inner_text()
            expected_neighbor_count = len(chosen.same_neighbors) + len(chosen.rival_neighbors)
            assert legacy_evidence_text.count(" cos=") == expected_neighbor_count
            assert react_evidence.locator(".class-similarity-neighbor").count() == expected_neighbor_count
            for neighbor in chosen.same_neighbors + chosen.rival_neighbors:
                assert f"{neighbor.record_id[:10]} cos={neighbor.cosine:.4f}" in legacy_evidence_text
                neighbor_record = record_by_id[neighbor.record_id]
                neighbor_filename = Path(str(neighbor_record["path"])).name
                assert neighbor_filename in react_evidence.inner_text()
                assert f"cosine {neighbor.cosine:.4f}" in react_evidence.inner_text()

            assert _source_hashes(dataset) == source_before
            browser.close()
    finally:
        if legacy_process is not None:
            _stop_process_tree(legacy_process)
        if legacy_log is not None:
            legacy_log.close()
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)
        _remove_streamlit_reference_worktree(legacy_root, temp_root=tmp_path)

    assert _source_hashes(dataset) == source_before
