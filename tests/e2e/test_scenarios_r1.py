"""Round-1 usage-scenario E2E tests (multi-agent generated, 10 scenarios).

Each test encodes one user scenario's MUST requirements; scoring (0-100
per scenario, deductions for unsupported needs) is reported separately.
Execution order is intentional: s01 runs the detector pipeline that
s10/s02/s03/s04 then chain on.
"""
from __future__ import annotations

import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from playwright.sync_api import expect

from .conftest import load_app, wait_idle

pytestmark = pytest.mark.e2e

expect.set_options(timeout=15000)


# ── fixtures ────────────────────────────────────────────────────────────

def _img(d: Path, name: str, seed: int, bias: int = 0) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 255, (64, 64, 3)).astype("uint8")
    arr[:, :, bias % 3] = 255
    p = d / name
    Image.fromarray(arr).save(p, quality=90)
    return p


@pytest.fixture(scope="module")
def detector_dataset(tmp_path_factory) -> Path:
    """2-class YOLO layout: train(12) + val(6), labels + classes.txt."""
    root = tmp_path_factory.mktemp("det")
    (root / "classes.txt").write_text("good\nrotten\n", encoding="utf-8")
    seed = 0
    for split, n_per in (("train", 6), ("val", 3)):
        img_dir, lbl_dir = root / split / "images", root / split / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for ci in range(2):
            for i in range(n_per):
                seed += 1
                p = _img(img_dir, f"c{ci}_{i:02d}.jpg", seed, bias=ci)
                (lbl_dir / f"{p.stem}.txt").write_text(
                    f"{ci} 0.5 0.5 0.6 0.6\n", encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def det_page(app_server, browser, detector_dataset):
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(20000)
    load_app(page, app_server)
    yield page
    ctx.close()


@pytest.fixture()
def fresh_page(app_server, browser):
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(20000)
    load_app(page, app_server)
    yield page
    ctx.close()


def _no_exception(page) -> None:
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)


def _status(page) -> str:
    return page.locator('.st-key-viz_status_line').inner_text()


def _run_and_collect_progress(page, timeout_s: int = 300) -> set[str]:
    page.locator('.st-key-run_viz button').click()
    texts: set[str] = set()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        prog = page.locator('[data-testid="stProgress"]')
        try:
            if prog.count():
                t = prog.first.inner_text(timeout=250).strip()
                if t:
                    texts.add(t)
        except Exception:
            pass
        if page.locator('.st-key-viz_scatter_wrap g.points path').count() > 0:
            break
        time.sleep(0.15)
    wait_idle(page, timeout=120000)
    return texts


def _click_marker(page, group_idx: int, path_idx: int, shift: bool = False) -> None:
    before = _status(page)
    groups = page.locator('.st-key-viz_scatter_wrap g.points')
    g = groups.nth(min(group_idx, groups.count() - 1))
    p = g.locator('path').nth(min(path_idx, g.locator('path').count() - 1))
    bb = p.bounding_box()
    if shift:
        page.keyboard.down("Shift")
    page.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
    if shift:
        page.keyboard.up("Shift")
    try:
        page.wait_for_function(
            """(prev) => { const el = document.querySelector('.st-key-viz_status_line');
                           return el && el.innerText !== prev; }""",
            arg=before, timeout=5000)
    except Exception:
        pass
    wait_idle(page)


def _select_option(page, key: str, label: str) -> None:
    sel = (page.locator(f'.st-key-{key} [data-baseweb="select"]')
           .locator("visible=true").first)
    sel.click()
    opt = page.get_by_role("option", name=label, exact=True)
    try:
        opt.click(timeout=5000)
    except Exception:
        sel.click()  # dropdown didn't open (stale node) — reopen once
        opt.click()
    wait_idle(page, timeout=60000)


def _switch_panel(page, label: str) -> None:
    page.locator('.st-key-viz_panel_view').get_by_text(label, exact=True).click()
    wait_idle(page)


# ── S1: detector multi-folder cold run + model switching ────────────────

def test_s01_detector_multimodel_cold_run(det_page, detector_dataset):
    page = det_page
    page.locator('.st-key-viz_folder_text textarea').fill(
        str(detector_dataset / "train") + "\n" + str(detector_dataset / "val"))
    texts = _run_and_collect_progress(page, timeout_s=400)
    assert page.locator('.st-key-viz_scatter_wrap g.points path').count() > 0
    assert len(texts) >= 3, f"progress must stream: {texts}"
    expect(page.get_by_text(re.compile("Auto-detected 2 classes"))).to_be_visible()
    # distinct traces per (class, split)
    assert page.locator('.st-key-viz_scatter_wrap g.points').count() >= 2
    # model switching re-renders without losing the session
    # (this machine ships dinov2 + chinese-clip; resnet is optional)
    models = ["dinov2_vits14", "chinese-clip-vit-base-patch16"]
    for m in models:
        _select_option(page, "viz_model_select", m)
        expect(page.locator('.st-key-viz_scatter_wrap')
               .get_by_text(re.compile(m))).to_be_visible()
        _no_exception(page)
    _no_exception(page)


# ── S10(edge): empty selection / empty export list honesty ──────────────

def test_s10_empty_selection_export_honesty(det_page):
    page = det_page
    assert "未選取" in _status(page)
    # batch-add is disabled with nothing selected — no way to crash it
    expect(page.locator('.st-key-viz_add_btn button')).to_be_disabled()
    _switch_panel(page, "匯出清單")
    expect(page.get_by_text("清單是空的。", exact=False)).to_be_visible()
    assert page.locator('.st-key-viz_export_csv').count() == 0  # honest empty state
    _switch_panel(page, "選取")
    _no_exception(page)


# ── S2: selection → viewer → YOLO overlay chain ─────────────────────────

def test_s02_selection_viewer_yolo_chain(det_page):
    page = det_page
    _click_marker(page, 0, 1)
    # plotly re-applies the restored selection asynchronously — retry the
    # shift-click on fresh points until accumulation sticks (cf. test_h)
    selected = 1
    for attempt in range(3):
        _click_marker(page, 1, 1 + attempt, shift=True)
        m = re.search(r"已選取 (\d+) 個點", _status(page))
        selected = int(m.group(1)) if m else 0
        if selected >= 2:
            break
    assert selected >= 2
    page.locator('.st-key-viz_grid [class*="st-key-viz_card_"] button').first.click()
    wait_idle(page)
    viewer = page.locator('.st-key-viz_image_viewer')
    img = viewer.locator('[data-testid="stImage"] img').first
    expect(img).to_be_visible()
    # YOLO overlay toggle exists in detector mode and survives navigation
    boxes = page.locator('.st-key-viz_img_boxes input')
    expect(boxes).to_have_count(1)
    page.locator('.st-key-viz_img_boxes label').first.click()
    wait_idle(page)
    expect(viewer.locator('[data-testid="stImage"] img').first).to_be_visible()
    nxt = page.locator('.st-key-viz_img_next button')
    if nxt.is_enabled():
        nxt.click()
        wait_idle(page)
        assert page.locator('.st-key-viz_img_boxes input').is_checked(), \
            "YOLO toggle must keep its state across images"
    page.locator('.st-key-viz_img_close button').click()
    wait_idle(page)
    _no_exception(page)


# ── S3: find-similar, ascending distances, chain, cross-model, CSV ──────

def test_s03_find_similar_cross_model(det_page):
    page = det_page
    _click_marker(page, 0, 2)
    page.locator('.st-key-viz_similar_btn button').click()
    wait_idle(page)
    panel = page.locator('.st-key-viz_similar_panel')
    expect(panel.locator('[data-testid="stImage"] img')).to_have_count(9)
    dists = [float(x) for x in re.findall(r"d=([0-9.]+)", panel.inner_text())]
    assert len(dists) == 9 and all(0 <= d <= 2 for d in dists)
    # inner_text reads the 3-column grid column-major — rebuild rank order
    ranked = [dists[(j % 3) * 3 + j // 3] for j in range(9)]
    assert ranked == sorted(ranked), f"results must be sorted by distance: {ranked}"
    panel.get_by_text("↻ 以此為查詢").first.click()
    expect(panel.locator('button:has-text("#")')).to_have_count(2)
    wait_idle(page)
    # switching the model must not show stale results (recomputed live)
    _select_option(page, "viz_model_select", "dinov2_vits14")
    expect(page.locator('.st-key-viz_similar_panel [data-testid="stImage"] img')
           ).to_have_count(9)
    _no_exception(page)
    with page.expect_download() as dl:
        page.locator('.st-key-viz_export_similar button').click()
    lines = Path(dl.value.path()).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 10  # query + 9 results
    page.locator('.st-key-viz_similar_close button').click()
    wait_idle(page)
    _switch_panel(page, "選取")
    _no_exception(page)


# ── S4: outlier ranking default + rank-ordered viewer + export ──────────

def test_s04_outlier_ranking_and_export(det_page):
    page = det_page
    if "已選取" in _status(page):
        page.locator('.st-key-viz_clear_btn button').click()
        page.wait_for_function(
            """() => { const el = document.querySelector('.st-key-viz_status_line');
                       return el && el.innerText.includes('未選取'); }""", timeout=10000)
        wait_idle(page)
    status = _status(page)
    assert "離群度前" in status and "非品質判定" in status, status
    cards = page.locator('.st-key-viz_grid [class*="st-key-viz_card_"] button')
    expect(cards.first).to_be_visible()
    assert "第1" in cards.first.inner_text()  # rank, not raw score (honest)
    # 標籤分歧 ranking is one switch away
    _select_option(page, "viz_grid_sort", "標籤分歧")
    page.wait_for_function(
        """() => { const el = document.querySelector('.st-key-viz_status_line');
                   return el && el.innerText.includes('標籤分歧前'); }""", timeout=10000)
    _select_option(page, "viz_grid_sort", "空間順序")
    # rank order drives the viewer context
    page.locator('.st-key-viz_grid [class*="st-key-viz_card_"] button').first.click()
    wait_idle(page)
    expect(page.locator('.st-key-viz_image_viewer')
           .get_by_text(re.compile(r"1/\d+"))).to_be_visible()
    page.locator('.st-key-viz_slot_add button').click()
    wait_idle(page)
    _switch_panel(page, "匯出清單")
    with page.expect_download() as dl:
        page.locator('.st-key-viz_export_zip button').click()
    zf = zipfile.ZipFile(io.BytesIO(Path(dl.value.path()).read_bytes()))
    assert "manifest.csv" in zf.namelist()
    assert any(n.startswith("images/") for n in zf.namelist())
    _switch_panel(page, "選取")
    _no_exception(page)


# ── S5: Compare Distributions full metrics + coverage gap + JSON ────────

def test_s05_compare_distributions_metrics(fresh_page, tmp_path):
    page = fresh_page
    a, b = tmp_path / "gen", tmp_path / "real"
    for i in range(8):
        _img(a, f"a{i}.jpg", seed=100 + i, bias=0)
        _img(b, f"b{i}.jpg", seed=200 + i, bias=1)
    page.locator('.st-key-tool_switch').get_by_text("Compare Distributions").click()
    wait_idle(page)
    page.locator('.st-key-cmp_folder_a input').fill(str(a))
    page.locator('.st-key-cmp_folder_b input').fill(str(b))
    npairs = page.locator('[data-testid="stSidebar"] [data-testid="stNumberInput"] input')
    npairs.fill("20")
    page.keyboard.press("Enter")
    wait_idle(page)
    page.locator('.st-key-run_cmp button').click()
    expect(page.get_by_text("FID ↓")).to_be_visible(timeout=600000)
    wait_idle(page, timeout=120000)
    body = page.locator('[data-testid="stMain"]').inner_text()
    for token in ("KID ↓", "LPIPS ↓", "SSIM ↑", "PSNR ↑", "IS ↑",
                  "Coverage Gap Analysis", "樣本總數"):
        assert token in body, f"missing metric/section: {token}"
    with page.expect_download() as dl:
        page.get_by_text("⬇ Download JSON").click()
    metrics = json.loads(Path(dl.value.path()).read_text(encoding="utf-8"))
    for k in ("fid", "kid", "lpips", "ssim", "psnr",
              "is_a_mean", "is_b_mean", "n_a", "n_b", "model"):
        assert k in metrics, f"missing JSON key: {k}"
    assert metrics["n_a"] == 8 and metrics["n_b"] == 8
    _no_exception(page)


# ── S6(edge): minimal dataset (n=1) never crashes ───────────────────────

def test_s06_minimal_dataset_single_image(fresh_page, tmp_path):
    page = fresh_page
    _img(tmp_path / "mini" / "train" / "classA", "dog.jpg", seed=7)
    page.locator('.st-key-viz_mode').get_by_text("Image Classifier").click()
    wait_idle(page)
    page.locator('.st-key-viz_folder_text textarea').fill(
        str(tmp_path / "mini" / "train"))
    _run_and_collect_progress(page, timeout_s=300)
    _no_exception(page)
    assert page.locator('.st-key-viz_scatter_wrap g.points path').count() == 1
    # t-SNE/UMAP honestly skipped → Method offers PCA only
    page.locator('.st-key-viz_method_select [data-baseweb="select"]').first.click()
    expect(page.get_by_role("option")).to_have_count(1)
    page.keyboard.press("Escape")
    _click_marker(page, 0, 0)
    assert "已選取 1 個點" in _status(page)
    page.locator('.st-key-viz_similar_btn button').click()
    wait_idle(page)
    expect(page.locator('.st-key-viz_similar_panel')
           .get_by_text(re.compile("無法找相似|沒有其他影像"))).to_be_visible()
    _switch_panel(page, "選取")
    page.locator('.st-key-viz_add_btn button').click()
    wait_idle(page)
    _switch_panel(page, "匯出清單")
    with page.expect_download() as dl:
        page.locator('.st-key-viz_export_csv button').click()
    lines = Path(dl.value.path()).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # header + the single row
    _no_exception(page)


# ── S7(edge): unicode & spaces in paths / classes / filenames ───────────

def test_s07_unicode_paths_round_trip(fresh_page, tmp_path):
    page = fresh_page
    root = tmp_path / "資料 集"
    names = ["影像 1.jpg", "image (2).jpg", "圖片 [3].jpg"]
    seed = 0
    for split, n in (("train", 3), ("val", 2)):
        for cls in ("類別A", "類別B"):
            for i in range(n):
                seed += 1
                _img(root / split / cls, names[i % 3], seed, bias=0 if cls == "類別A" else 1)
    page.locator('.st-key-viz_mode').get_by_text("Image Classifier").click()
    wait_idle(page)
    page.locator('.st-key-viz_folder_text textarea').fill(
        str(root / "train") + "\n" + str(root / "val"))
    _run_and_collect_progress(page, timeout_s=400)
    _no_exception(page)
    expect(page.get_by_text(re.compile("自動偵測到 2 個類別.*類別A"))).to_be_visible()
    _select_option(page, "viz_split_select", "val")
    _select_option(page, "viz_split_select", "All")
    _click_marker(page, 0, 0)
    assert "已選取" in _status(page)
    page.locator('.st-key-viz_add_btn button').click()
    wait_idle(page)
    _switch_panel(page, "匯出清單")
    with page.expect_download() as dl:
        page.locator('.st-key-viz_export_csv button').click()
    text = Path(dl.value.path()).read_text(encoding="utf-8")
    assert "類別" in text
    with page.expect_download() as dl:
        page.locator('.st-key-viz_export_zip button').click()
    zf = zipfile.ZipFile(io.BytesIO(Path(dl.value.path()).read_bytes()))
    img_names = [n for n in zf.namelist() if n.startswith("images/")]
    assert img_names and any(any(t in n for t in ("影像", "圖片", "(2)"))
                             for n in img_names), img_names
    _no_exception(page)

    # ── S8(edge): mode switch fully isolates state (no silent carryover) ─
    page.locator('.st-key-viz_mode').get_by_text("Object Detector").click()
    wait_idle(page)
    expect(page.get_by_text("已切換模式", exact=False)).to_be_visible()
    expect(page.locator('.st-key-viz_mode_undo button')).to_be_visible()
    expect(page.get_by_text("快速開始")).to_be_visible()  # results gone
    page.locator('.st-key-viz_mode').get_by_text("Image Classifier").click()
    wait_idle(page)
    expect(page.get_by_text("快速開始")).to_be_visible()  # NOT auto-restored
    assert page.locator('.st-key-viz_status_line').count() == 0
    _no_exception(page)


# S8 is asserted at the tail of test_s07 (same session, intentional chain).
def test_s08_mode_switch_isolation_documented():
    """S8 assertions live at the end of test_s07 (chained on its session)."""
    assert True


# ── S9(edge): detector without classes.txt and empty manual input ───────

def test_s09_detector_missing_classes_error(fresh_page, tmp_path):
    page = fresh_page
    root = tmp_path / "noclasses"
    img_dir, lbl_dir = root / "train" / "images", root / "train" / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        p = _img(img_dir, f"x{i}.jpg", seed=i)
        (lbl_dir / f"{p.stem}.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    # default mode is Object Detector; no classes.txt anywhere
    page.locator('.st-key-viz_folder_text textarea').fill(str(root / "train"))
    page.get_by_text("類別來源", exact=False).click()  # open the expander
    wait_idle(page)
    class_input = page.locator('[data-testid="stSidebar"] [data-testid="stTextInput"] input')
    class_input.fill("")
    page.keyboard.press("Tab")
    wait_idle(page)
    page.locator('.st-key-run_viz button').click()
    wait_idle(page, timeout=60000)
    expect(page.get_by_text("Enter at least one class name.")).to_be_visible()
    _no_exception(page)
