"""Playwright E2E flows for the two-column linked-view UI (run in file order).

The module shares one browser page + one Streamlit session: the pipeline is
Run once (cold, proving progress streaming) and the interaction features are
exercised on the resulting state. Logic-level guarantees (sortedness, dedupe,
csv columns, thumbnails...) live in tests/test_interaction.py; these tests
prove the GUI wiring works end-to-end, including the UX-review acceptance
criteria: zero-scroll linked view, selection persistence across views,
default outlier grid with honest disclaimer, and export-list round trips.
"""
from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path

import pytest
from playwright.sync_api import expect

from .conftest import load_app, wait_idle

pytestmark = pytest.mark.e2e

expect.set_options(timeout=15000)


@pytest.fixture(scope="module")
def flow_page(app_server, browser, synthetic_dataset):
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(20000)
    load_app(page, app_server)
    yield page
    ctx.close()


def _no_exception(page) -> None:
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)


def _status_text(page) -> str:
    return page.locator('.st-key-viz_status_line').inner_text()


def _selected_count(page) -> int:
    m = re.search(r"已選取 (\d+) 個點", _status_text(page))
    return int(m.group(1)) if m else 0


def _grid_imgs(page):
    return page.locator('.st-key-viz_grid [data-testid="stImage"] img')


def _switch_panel(page, label: str) -> None:
    page.locator('.st-key-viz_panel_view').get_by_text(label, exact=True).click()
    wait_idle(page)


def _click_wait_status(page, css: str, timeout: int = 10000) -> None:
    """Click a button whose effect shows up in the status line.

    Fragment reruns can be fast enough that wait_idle alone races them —
    explicitly wait for the status text to change before returning.
    """
    before = _status_text(page)
    page.locator(css).click()
    try:
        page.wait_for_function(
            """(prev) => {
                const el = document.querySelector('.st-key-viz_status_line');
                return el && el.innerText !== prev;
            }""",
            arg=before, timeout=timeout,
        )
    except Exception:
        pass
    wait_idle(page)


def _select_option(page, key: str, label: str) -> None:
    page.locator(f'.st-key-{key} [data-baseweb="select"]').click()
    page.get_by_role("option", name=label, exact=True).click()
    wait_idle(page, timeout=60000)


def _click_marker(page, group_idx: int, path_idx: int, shift: bool = False) -> None:
    """Click the centre of one scatter marker (bbox re-queried fresh).

    The click → rerun → status-line update is asynchronous; wait for the
    status line to actually change before returning (clicking an already
    selected point legitimately changes nothing — swallow that timeout).
    """
    before = _status_text(page)
    groups = page.locator('.st-key-viz_scatter_wrap g.points')
    g = groups.nth(min(group_idx, groups.count() - 1))
    paths = g.locator('path')
    p = paths.nth(min(path_idx, paths.count() - 1))
    bb = p.bounding_box()
    assert bb is not None
    if shift:
        page.keyboard.down("Shift")
    page.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
    if shift:
        page.keyboard.up("Shift")
    try:
        page.wait_for_function(
            """(prev) => {
                const el = document.querySelector('.st-key-viz_status_line');
                return el && el.innerText !== prev;
            }""",
            arg=before, timeout=5000,
        )
    except Exception:
        pass
    wait_idle(page)


# ── (a) cold load ───────────────────────────────────────────────────────

def test_a_cold_load(flow_page):
    expect(flow_page.get_by_text("Dataset Analysis Tools")).to_be_visible()
    expect(flow_page.locator('[data-testid="stSidebar"]')).to_be_visible()
    # feature discoverability: the feature map popover is one click away
    popover_btn = flow_page.get_by_test_id("stPopoverButton")
    expect(popover_btn).to_be_visible()
    popover_btn.click()
    expect(flow_page.get_by_text("以文搜圖", exact=False).first).to_be_visible()
    flow_page.keyboard.press("Escape")
    _no_exception(flow_page)


# ── (b) run pipeline; prove streaming progress; scatter appears ────────

def test_b_run_with_streaming_progress(flow_page, synthetic_dataset):
    page = flow_page
    page.locator('.st-key-viz_mode').get_by_text("Image Classifier").click()
    wait_idle(page)
    page.locator('.st-key-viz_folder_text textarea').fill(str(synthetic_dataset))
    page.locator('.st-key-run_viz button').click()

    progress_texts: set[str] = set()
    deadline = time.time() + 150
    scatter_ready = False
    while time.time() < deadline:
        prog = page.locator('[data-testid="stProgress"]')
        try:
            if prog.count():
                t = prog.first.inner_text(timeout=300).strip()
                if t:
                    progress_texts.add(t)
        except Exception:
            pass
        if page.locator('.st-key-viz_scatter_wrap g.points path').count() > 0:
            scatter_ready = True
            break
        time.sleep(0.15)

    assert scatter_ready, "scatter plot never appeared after Run"
    assert len(progress_texts) >= 2, f"progress did not stream: {progress_texts}"
    wait_idle(page, timeout=60000)
    _no_exception(page)
    expect(page.get_by_text(re.compile("自動偵測到 2 個類別"))).to_be_visible()

    # F1 data contract: Run writes/updates manifest.jsonl in the dataset
    # folder — one line per image, content-hashed, with embedding refs
    import json
    mpath = synthetic_dataset / "manifest.jsonl"
    assert mpath.exists(), "Run must write manifest.jsonl"
    entries = [json.loads(ln) for ln in
               mpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(entries) == 24
    for e in entries[:3]:
        assert len(e["sha256"]) == 64
        assert e["embedding_refs"], "embedding_refs must be filled after Run"
        assert e["labels"] and e["split"] == "train"


# ── (c) zero-scroll linked view + default outlier grid ──────────────────

def test_c_two_column_default_outlier_grid(flow_page):
    page = flow_page
    # acceptance: scatter and grid share the screen with no page scroll
    assert page.evaluate(
        "() => document.body.scrollHeight <= window.innerHeight + 1"
    ), "page must not scroll: scatter and thumbnail wall share one screen"
    # honest default: top-N by outlier-ness, explicitly NOT a quality verdict
    status = _status_text(page)
    assert "未選取" in status and "非品質判定" in status, status
    expect(_grid_imgs(page)).to_have_count(24)  # min(50, n_records)
    # viewer slot is present (empty placeholder state)
    expect(page.locator('.st-key-viz_image_viewer')).to_be_visible()
    _no_exception(page)


# ── (d) click a point → selection feeds the grid, zero scroll ───────────

def test_d_click_point_selects(flow_page):
    page = flow_page
    _click_marker(page, 0, 2)
    assert _selected_count(page) == 1
    expect(_grid_imgs(page)).to_have_count(1)
    # acceptance: first thumbnail fully inside the viewport without scrolling
    assert page.evaluate(
        """() => {
            const img = document.querySelector('.st-key-viz_grid img');
            if (!img) return false;
            const r = img.getBoundingClientRect();
            return r.top >= 0 && r.bottom <= window.innerHeight;
        }"""
    ), "selected thumbnail must be visible without page scrolling"
    _no_exception(page)


# ── (e) card click → viewer slot swaps content ──────────────────────────

def test_e_card_click_opens_viewer_slot(flow_page):
    page = flow_page
    # the grid also contains st.image fullscreen buttons — target the card
    # button via its widget key class
    page.locator('.st-key-viz_grid [class*="st-key-viz_card_"] button').first.click()
    wait_idle(page)
    viewer = page.locator('.st-key-viz_image_viewer')
    img = viewer.locator('[data-testid="stImage"] img').first
    expect(img).to_be_visible()
    assert page.evaluate("el => el.naturalWidth", img.element_handle()) > 0
    expect(viewer.get_by_text(re.compile(r"1/1"))).to_be_visible()
    # manifest provenance (sha256/phash/refs) is one click away
    viewer.get_by_text("📄 Manifest").click()
    expect(page.get_by_text("sha256：", exact=False).first).to_be_visible()
    page.keyboard.press("Escape")
    _no_exception(page)


# ── (f) selection persists across model/method/split/dim views (W2) ─────

def test_f_selection_persists_across_views(flow_page):
    page = flow_page
    n = _selected_count(page)
    assert n >= 1
    _select_option(page, "viz_method_select", "t-SNE")
    assert _selected_count(page) == n, "selection must survive a method change"
    _select_option(page, "viz_split_select", "train")
    assert _selected_count(page) == n, "selection must survive a split change"
    _select_option(page, "viz_split_select", "All")
    _select_option(page, "viz_method_select", "PCA")
    assert _selected_count(page) == n
    _no_exception(page)


# ── (g) find similar: panel switch, k results, chain re-query ───────────

def test_g_find_similar(flow_page):
    page = flow_page
    page.locator('.st-key-viz_similar_btn button').click()
    wait_idle(page)
    panel = page.locator('.st-key-viz_similar_panel')
    expect(panel).to_be_visible()
    imgs = panel.locator('[data-testid="stImage"] img')
    expect(imgs).to_have_count(9)  # default k = min(9, n-1)
    # chain re-query: click ↻ on the first result → 2 chips
    panel.get_by_text("↻ 以此為查詢").first.click()
    expect(panel.locator('button:has-text("#")')).to_have_count(2)  # chain grew
    wait_idle(page)
    expect(panel.locator('[data-testid="stImage"] img')).to_have_count(9)
    page.locator('.st-key-viz_similar_close button').click()
    wait_idle(page)
    _switch_panel(page, "選取")
    _no_exception(page)


# ── (h) shift-click accumulation (regression: fragment must not eat
#         plotly selection state) ────────────────────────────────────────

def test_h_multi_select(flow_page):
    page = flow_page
    chart = page.locator('.st-key-viz_scatter_wrap')
    chart.scroll_into_view_if_needed()
    drag = chart.locator('.nsewdrag').first
    bb = drag.bounding_box()
    page.mouse.dblclick(bb["x"] + bb["width"] * 0.5, bb["y"] + 5)
    wait_idle(page)

    _click_marker(page, 0, 2)
    assert _selected_count(page) == 1, "first click should select exactly 1 point"
    # plotly re-applies the restored selection asynchronously after the
    # rerun re-mounts the chart; retry shift-clicks on new points.
    selected = 1
    for attempt in range(3):
        _click_marker(page, 1, 2 + attempt, shift=True)
        selected = _selected_count(page)
        if selected >= 2:
            break
    assert selected >= 2, f"shift-click accumulated only {selected} points"
    _no_exception(page)


# ── (i) batch add → export list → CSV + ZIP round trip (W6) ─────────────

def test_i_export_list_round_trip(flow_page):
    page = flow_page
    n_sel = _selected_count(page)
    assert n_sel >= 2
    page.locator('.st-key-viz_add_btn button').click()
    wait_idle(page)
    _switch_panel(page, "匯出清單")
    expect(page.get_by_text(re.compile(rf"共 {n_sel} 張"))).to_be_visible()

    with page.expect_download() as dl:
        page.locator('.st-key-viz_export_csv button').click()
    text = Path(dl.value.path()).read_text(encoding="utf-8")
    lines = text.splitlines()
    # 策展購物車 CSV now carries provenance columns after sha256
    assert lines[0] == "index,filename,path,label,split,sha256,source,score,reason"
    assert len(lines) == 1 + n_sel, "CSV rows must equal export-list size"
    for row in lines[1:]:
        cells = row.split(",")
        sha = cells[5]
        assert len(sha) == 64, "exported rows must be content-addressed (manifest sha256)"
        assert cells[6] == "manual", "box-selection adds carry source=manual"

    with page.expect_download() as dl:
        page.locator('.st-key-viz_export_zip button').click()
    data = Path(dl.value.path()).read_bytes()
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = zf.namelist()
    assert "manifest.csv" in names
    assert sum(1 for x in names if x.startswith("images/")) == n_sel

    _switch_panel(page, "選取")
    _no_exception(page)


# ── (j) 3D mode keeps the selection (was: invalidated) ──────────────────

def test_j_3d_mode_preserves_selection(flow_page):
    page = flow_page
    n = _selected_count(page)
    assert n >= 1
    page.locator('.st-key-viz_dim_radio').get_by_text("3D").click()
    wait_idle(page, timeout=30000)
    # 3D now highlights the current 2D selection (重評 #3): the caption names
    # the highlighted count
    expect(page.get_by_text(re.compile(rf"3D 看：黑圈為目前選取的 {n} 點"))).to_be_visible()
    _no_exception(page)
    page.locator('.st-key-viz_dim_radio').get_by_text("2D").click()
    wait_idle(page, timeout=30000)
    assert _selected_count(page) == n, "selection must survive a 2D↔3D round trip"
    _no_exception(page)


# ── (k) explicit clear → back to honest default grid ────────────────────

def test_k_clear_selection(flow_page):
    page = flow_page
    assert _selected_count(page) >= 1
    _click_wait_status(page, '.st-key-viz_clear_btn button')
    status = _status_text(page)
    assert "未選取" in status and "非品質判定" in status, status
    expect(_grid_imgs(page)).to_have_count(24)
    # one-way data flow: an unrelated full rerun must NOT resurrect the
    # cleared selection from a stale widget event
    page.locator('.st-key-viz_dim_radio').get_by_text("3D").click()
    wait_idle(page, timeout=30000)
    page.locator('.st-key-viz_dim_radio').get_by_text("2D").click()
    wait_idle(page, timeout=30000)
    assert "未選取" in _status_text(page), "cleared selection must stay cleared"
    _no_exception(page)


# ── (k2) F5 label audit: ranking criterion switch on the default grid ───

def test_k2_label_disagreement_ranking(flow_page):
    page = flow_page
    assert "未選取" in _status_text(page)
    page.locator('.st-key-viz_grid_sort [data-baseweb="select"]').click()
    page.get_by_role("option", name="標籤分歧", exact=True).click()
    page.wait_for_function(
        """() => {
            const el = document.querySelector('.st-key-viz_status_line');
            return el && el.innerText.includes('標籤分歧前');
        }""",
        timeout=10000,
    )
    status = _status_text(page)
    assert "非品質判定" in status, status  # honest framing survives the switch
    expect(_grid_imgs(page)).to_have_count(24)
    # back to the default criterion
    page.locator('.st-key-viz_grid_sort [data-baseweb="select"]').click()
    page.get_by_role("option", name="空間順序", exact=True).click()
    page.wait_for_function(
        """() => {
            const el = document.querySelector('.st-key-viz_status_line');
            return el && el.innerText.includes('離群度前');
        }""",
        timeout=10000,
    )
    _no_exception(page)


# ── (l) re-Run: selection resets, export list survives (data-token) ─────

def test_l_rerun_resets_selection_keeps_export_list(flow_page):
    page = flow_page
    _click_marker(page, 0, 1)
    assert _selected_count(page) >= 1
    page.locator('.st-key-run_viz button').click()
    # the warm re-Run can finish before wait_idle even sees the runner —
    # wait directly for the reset status text instead
    page.wait_for_function(
        """() => {
            const el = document.querySelector('.st-key-viz_status_line');
            return el && el.innerText.includes('未選取');
        }""",
        timeout=120000,
    )
    wait_idle(page, timeout=120000)
    assert "未選取" in _status_text(page), "re-Run (new data token) must reset the selection"
    # export list is keyed by image path — it survives a re-Run
    _switch_panel(page, "匯出清單")
    expect(page.get_by_text(re.compile(r"共 [1-9]\d* 張"))).to_be_visible()
    assert page.locator('.st-key-viz_export_grid [data-testid="stImage"] img').count() >= 1
    _switch_panel(page, "選取")
    _no_exception(page)


# ── (m) selection feedback latency (soft SLA: warm, 24 cards) ───────────

def test_m_selection_latency(flow_page):
    page = flow_page
    timings = []
    for path_idx in (1, 3, 5):
        # clean slate so the click flips the status 未選取 → 已選取
        if _selected_count(page) >= 1:
            _click_wait_status(page, '.st-key-viz_clear_btn button')
        groups = page.locator('.st-key-viz_scatter_wrap g.points')
        p = groups.nth(0).locator('path').nth(path_idx)
        bb = p.bounding_box()
        t0 = time.perf_counter()
        page.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
        page.wait_for_function(
            """() => {
                const el = document.querySelector('.st-key-viz_status_line');
                return el && el.innerText.includes('已選取');
            }""",
            timeout=10000,
        )
        timings.append(time.perf_counter() - t0)
        wait_idle(page)
    timings.sort()
    median = timings[len(timings) // 2]
    assert median <= 2.5, f"selection→grid feedback too slow: median {median:.2f}s of {timings}"
    _no_exception(page)


# ── (n) error path: nonexistent folder → st.error, no traceback ─────────

def test_n_invalid_folder_error(app_page):
    page = app_page
    page.locator('.st-key-viz_mode').get_by_text("Image Classifier").click()
    wait_idle(page)
    page.locator('.st-key-viz_folder_text textarea').fill(r"C:\does\not\exist\nope")
    page.locator('.st-key-run_viz button').click()
    wait_idle(page, timeout=30000)
    expect(page.get_by_text(re.compile("資料夾不存在"))).to_be_visible()
    _no_exception(page)


# ── (o) projection-method opt-out: only PCA runs, no t-SNE anywhere ─────

def test_o_projection_method_skip(app_page, synthetic_dataset):
    page = app_page
    page.locator('.st-key-viz_mode').get_by_text("Image Classifier").click()
    wait_idle(page)
    page.locator('.st-key-viz_folder_text textarea').fill(str(synthetic_dataset))
    # drop t-SNE and UMAP from the 投影方法 multiselect. Each removal
    # triggers a rerun that can swallow the next keypress — retry until
    # only PCA's tag remains.
    ms_input = page.locator('.st-key-viz_methods input')
    tags = page.locator('.st-key-viz_methods span[data-baseweb="tag"]')
    for _ in range(8):
        if tags.count() <= 1:
            break
        ms_input.click()
        page.keyboard.press("Backspace")
        page.wait_for_timeout(400)
        wait_idle(page)
    page.keyboard.press("Escape")
    expect(tags).to_have_count(1)
    page.locator('.st-key-run_viz button').click()

    progress_texts: set[str] = set()
    deadline = time.time() + 120
    while time.time() < deadline:
        prog = page.locator('[data-testid="stProgress"]')
        try:
            if prog.count():
                t = prog.first.inner_text(timeout=300).strip()
                if t:
                    progress_texts.add(t)
        except Exception:
            pass
        if page.locator('.st-key-viz_scatter_wrap g.points path').count() > 0:
            break
        time.sleep(0.1)
    wait_idle(page, timeout=60000)

    assert not any("t-SNE" in t or "UMAP" in t for t in progress_texts), progress_texts
    # Method dropdown offers only PCA
    page.locator('.st-key-viz_method_select').click()
    options = page.get_by_role("option")
    expect(options).to_have_count(1)
    expect(options.first).to_have_text("PCA")
    page.keyboard.press("Escape")
    _no_exception(page)


# ── (p) F4 duplicate / leakage scan on a dataset with a real bit-copy ───

@pytest.fixture()
def leakage_dataset(tmp_path):
    """train has 3 unique images + 1 source; val has a BYTE-IDENTICAL copy
    of that source (classic train/val leakage) + 1 unique image."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(7)

    def _img(d, name, seed):
        d.mkdir(parents=True, exist_ok=True)
        arr = np.random.default_rng(seed).integers(0, 255, (64, 64, 3)).astype("uint8")
        Image.fromarray(arr).save(d / name, quality=90)
        return d / name

    root = tmp_path / "leakds"
    for i in range(3):
        _img(root / "train" / "classA", f"u{i}.jpg", seed=10 + i)
    src = _img(root / "train" / "classA", "dup.jpg", seed=42)
    val_dir = root / "val" / "classA"
    val_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / "dup_copy.jpg").write_bytes(src.read_bytes())
    _img(root / "val" / "classA", "v0.jpg", seed=99)
    return root


def test_p_duplicate_leakage_scan(app_page, leakage_dataset):
    page = app_page
    page.locator('.st-key-viz_mode').get_by_text("Image Classifier").click()
    wait_idle(page)
    page.locator('.st-key-viz_folder_text textarea').fill(
        str(leakage_dataset / "train") + "\n" + str(leakage_dataset / "val"))
    # PCA only — the dup scan does not depend on projections, keep it fast
    ms_input = page.locator('.st-key-viz_methods input')
    tags = page.locator('.st-key-viz_methods span[data-baseweb="tag"]')
    for _ in range(8):
        if tags.count() <= 1:
            break
        ms_input.click()
        page.keyboard.press("Backspace")
        page.wait_for_timeout(400)
        wait_idle(page)
    page.keyboard.press("Escape")
    page.locator('.st-key-run_viz button').click()
    page.wait_for_selector('.st-key-viz_scatter_wrap g.points path', timeout=180000)
    wait_idle(page, timeout=120000)

    _switch_panel(page, "重複")
    expect(page.locator('.st-key-viz_dup_panel')).to_be_visible()
    # default method: phash, hamming <= 4 — the bit-copy must surface
    page.locator('.st-key-viz_dup_scan button').click()
    expect(page.get_by_text(re.compile(r"找到 \d+ 對候選"))).to_be_visible()
    pair_imgs = page.locator('.st-key-viz_dup_list [data-testid="stImage"] img')
    expect(pair_imgs.nth(1)).to_be_visible()  # auto-waits: pair renders side by side
    _no_exception(page)

    # leakage filter: the copy spans train/val so it must survive 僅跨 split
    page.locator('.st-key-viz_dup_cross label').first.click()
    wait_idle(page)
    page.locator('.st-key-viz_dup_scan button').click()
    expect(page.get_by_text(re.compile(r"找到 \d+ 對候選"))).to_be_visible()
    _no_exception(page)

    # review one side in the viewer slot
    page.locator('.st-key-viz_dup_list [class*="st-key-viz_dup_0_"] button').first.click()
    viewer = page.locator('.st-key-viz_image_viewer')
    expect(viewer.locator('[data-testid="stImage"] img').first).to_be_visible()

    # one-click exclusion list: add all right-hand sides, then verify
    page.locator('.st-key-viz_dup_add_all button').click()
    wait_idle(page)
    _switch_panel(page, "匯出清單")
    expect(page.get_by_text(re.compile(r"共 [1-9]\d* 張"))).to_be_visible()
    _no_exception(page)


# ── (q) F7 text-to-image search (needs Chinese-CLIP weights on disk) ────

_CLIP_DIR = (Path(__file__).resolve().parent.parent.parent
             / "models" / "chinese-clip-vit-base-patch16")


@pytest.mark.skipif(not (_CLIP_DIR / "config.json").exists(),
                    reason="Chinese-CLIP weights not downloaded")
def test_q_text_to_image_search(flow_page):
    page = flow_page
    _select_option(page, "viz_model_select", _CLIP_DIR.name)
    _switch_panel(page, "相似")
    box = page.locator('.st-key-viz_text_query input')
    expect(box).to_be_visible()
    box.fill("斑馬")
    page.keyboard.press("Enter")
    wait_idle(page, timeout=120000)  # first query loads the text tower
    panel = page.locator('.st-key-viz_similar_panel')
    expect(page.get_by_text(re.compile("「斑馬」的前 \\d+ 名"))).to_be_visible()
    expect(panel.locator('[data-testid="stImage"] img')).to_have_count(9)
    # pivot: a text hit becomes the root of an image query chain
    panel.get_by_text("↻ 以此圖續查").first.click()
    expect(panel.locator('button:has-text("#")').first).to_be_visible()
    expect(page.locator('.st-key-viz_text_query input')).to_have_value("")
    _no_exception(page)


# ── (r) layout review: quick-start cards + one-click demo (coco8) ───────

def test_r_quick_start_demo(app_page):
    page = app_page
    # cold start: three step cards + the demo button replace the dead white
    expect(page.get_by_text("快速開始")).to_be_visible()
    for step in ("① 選資料", "② 跑分析", "③ 探索"):
        expect(page.get_by_text(step)).to_be_visible()
    demo_btn = page.locator('.st-key-viz_demo_btn button')
    expect(demo_btn).to_be_visible()
    # brand line survives the title removal (toolbar row)
    expect(page.get_by_text("Dataset Analysis Tools")).to_be_visible()

    demo_btn.click()
    page.wait_for_selector('.st-key-viz_scatter_wrap g.points path', timeout=600000)
    wait_idle(page, timeout=300000)
    # the demo run lands in the normal linked view with the default grid
    assert "未選取" in _status_text(page)
    expect(_grid_imgs(page).first).to_be_visible()
    _no_exception(page)


# ── (s) Compare Distributions linked view: click → thumbnails → viewer ──

def test_s_compare_linked_view(app_page, tmp_path):
    import numpy as np
    from PIL import Image
    page = app_page
    for name, bias, seed0 in (("setA", 0, 300), ("setB", 1, 400)):
        d = tmp_path / name
        d.mkdir()
        for i in range(6):
            arr = np.random.default_rng(seed0 + i).integers(0, 255, (64, 64, 3)).astype("uint8")
            arr[:, :, bias] = 255
            Image.fromarray(arr).save(d / f"{name}_{i}.jpg", quality=90)

    page.locator('.st-key-tool_switch').get_by_text("Compare Distributions").click()
    wait_idle(page)
    page.locator('.st-key-cmp_folder_a input').fill(str(tmp_path / "setA"))
    page.locator('.st-key-cmp_folder_b input').fill(str(tmp_path / "setB"))
    # viz-only keeps this fast — the linked view must work without metrics
    page.locator('[data-testid="stSidebar"] [data-testid="stCheckbox"] label').first.click()
    wait_idle(page)
    page.locator('.st-key-run_cmp button').click()
    page.wait_for_selector('.st-key-cmp_scatter_wrap g.points path', timeout=300000)
    wait_idle(page, timeout=120000)

    # empty state is honest, hint visible
    expect(page.get_by_text("對應影像會立即顯示在這裡", exact=False)).to_be_visible()
    # click one marker → thumbnail appears in the right panel
    pth = page.locator('.st-key-cmp_scatter_wrap g.points').nth(0).locator('path').nth(1)
    bb = pth.bounding_box()
    page.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
    page.wait_for_function(
        """() => {
            const el = document.querySelector('.st-key-cmp_grid');
            return el && el.innerText.includes('已選取');
        }""", timeout=10000)
    wait_idle(page)
    expect(page.locator('.st-key-cmp_grid [data-testid="stImage"] img').first).to_be_visible()
    # card click → viewer slot shows the full image with group metadata
    page.locator('.st-key-cmp_grid [class*="st-key-cmp_card_"] button').first.click()
    wait_idle(page)
    viewer = page.locator('.st-key-cmp_image_viewer')
    img = viewer.locator('[data-testid="stImage"] img').first
    expect(img).to_be_visible()
    assert page.evaluate("el => el.naturalWidth", img.element_handle()) > 0
    expect(viewer.get_by_text(re.compile("setA|setB"))).to_be_visible()
    _no_exception(page)


# ── (t) persistent UMAP reference frame (固定參考系) ─────────────────────

def test_t_umap_reference_frame(app_page, tmp_path):
    import numpy as np
    from PIL import Image
    page = app_page
    train = tmp_path / "ds" / "train"
    for ci, cls in enumerate(("a", "b")):
        d = train / cls
        d.mkdir(parents=True)
        for i in range(3):
            arr = np.random.default_rng(ci * 10 + i).integers(0, 255, (64, 64, 3)).astype("uint8")
            arr[:, :, ci] = 255
            Image.fromarray(arr).save(d / f"{cls}{i}.jpg", quality=90)

    page.locator('.st-key-viz_mode').get_by_text("Image Classifier").click()
    wait_idle(page)
    page.locator('.st-key-viz_folder_text textarea').fill(str(train))
    page.locator('.st-key-viz_umap_ref label').first.click()
    wait_idle(page)
    page.locator('.st-key-run_viz button').click()
    page.wait_for_selector('.st-key-viz_scatter_wrap g.points path', timeout=300000)
    wait_idle(page, timeout=120000)
    _no_exception(page)
    refs = list(train.glob("embeddings_*/umap_ref.pkl"))
    assert refs, "the fitted UMAP reference frame must be persisted to disk"

    # second Run reuses the frozen frame without refitting or crashing
    page.locator('.st-key-run_viz button').click()
    page.wait_for_function(
        """() => {
            const el = document.querySelector('.st-key-viz_status_line');
            return el && el.innerText.includes('未選取');
        }""", timeout=300000)
    wait_idle(page, timeout=120000)
    _no_exception(page)
    assert list(train.glob("embeddings_*/umap_ref.pkl")), "frame must survive re-Run"


# ── (u) completeness heatmap tool (defect-mechanisms v2 §5) ─────────────

def test_u_completeness_heatmap(app_page, tmp_path):
    import numpy as np
    from PIL import Image
    page = app_page
    # 3 classes, each split bright vs dark → an interpretable label×brightness grid
    root = tmp_path / "concept" / "train"
    for ci, cls in enumerate(("alpha", "beta", "gamma")):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(8):
            base = 210 if i < 4 else 40       # bright half / dark half
            arr = np.full((64, 64, 3), base, "uint8")
            arr[:, :, ci] = (arr[:, :, ci].astype(int) + 30).clip(0, 255)
            Image.fromarray(arr).save(d / f"{cls}_{i}.jpg", quality=90)

    page.locator('.st-key-tool_switch').get_by_text("完整度熱力圖").click()
    wait_idle(page)
    page.locator('.st-key-cov_folder_text textarea').fill(str(root))
    # X = label, Y = brightness (default index 2 already = brightness)
    page.locator('.st-key-run_cov button').click()
    page.wait_for_selector('.st-key-cov_heatmap', timeout=300000)
    wait_idle(page, timeout=120000)
    _no_exception(page)

    # the headline number + missing-cell stats render
    expect(page.get_by_text("Coverage Health", exact=True)).to_be_visible()
    expect(page.get_by_text("缺格數", exact=True)).to_be_visible()
    expect(page.get_by_text("假完整格", exact=True)).to_be_visible()
    # the heatmap has cells (one trace, label×brightness)
    assert page.locator('.st-key-cov_heatmap').count() == 1

    # pick a populated cell → its images render in the side panel
    page.locator('.st-key-cov_cell_pick [data-baseweb="select"]').click()
    page.get_by_role("option").nth(1).click()  # first real cell
    wait_idle(page)
    expect(page.locator('.st-key-cov_cell_close')).to_be_visible()
    _no_exception(page)


# ── (v) escape health card tab (defect-mechanisms §4) ───────────────────

def test_v_health_card(flow_page):
    page = flow_page
    _switch_panel(page, "選取")  # status line + grid live on this panel
    # ensure something is selected (chain off the shared session)
    if _selected_count(page) == 0:
        _click_marker(page, 0, 1)
    # open a card subject by clicking a grid card → sets viz_active_image
    page.locator('.st-key-viz_grid [class*="st-key-viz_card_"] button').first.click()
    wait_idle(page)
    page.locator('.st-key-viz_panel_view').get_by_text("體檢卡", exact=True).click()
    wait_idle(page)
    panel = page.locator('.st-key-viz_card_panel')
    # three-orthogonal-signal H1–H5 diagnosis
    expect(panel.get_by_text(re.compile("根因："))).to_be_visible()
    expect(panel.get_by_text(re.compile("補資料有效性"))).to_be_visible()
    expect(panel.get_by_text("S1 人類一致性", exact=True)).to_be_visible()
    expect(panel.get_by_text("S2 命中密度", exact=True)).to_be_visible()
    expect(panel.get_by_text("S3 模型不確定度", exact=True)).to_be_visible()
    # no scores.csv → S3 falls back to the labeled proxy, stated honestly
    expect(panel.get_by_text(re.compile("代理"))).to_be_visible()
    expect(page.locator('.st-key-viz_card_export')).to_be_visible()
    page.locator('.st-key-viz_panel_view').get_by_text("選取", exact=True).click()
    wait_idle(page)
    _no_exception(page)


# ── (w) completeness calibration + candidate mining ─────────────────────

def test_w_completeness_calibration_and_mining(app_page, tmp_path):
    import numpy as np
    from PIL import Image
    page = app_page
    root = tmp_path / "cov2" / "train"
    pool = tmp_path / "cov2" / "pool"
    pool.mkdir(parents=True)
    for ci, cls in enumerate(("p", "q")):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(6):
            arr = np.full((64, 64, 3), 210 if i < 3 else 40, "uint8")
            arr[:, :, ci] = 200
            Image.fromarray(arr).save(d / f"{cls}_{i}.jpg", quality=90)
    # candidate pool: a few extra images to mine from
    for i in range(5):
        arr = np.random.default_rng(i).integers(0, 255, (64, 64, 3)).astype("uint8")
        Image.fromarray(arr).save(pool / f"cand_{i}.jpg", quality=90)

    page.locator('.st-key-tool_switch').get_by_text("完整度熱力圖").click()
    wait_idle(page)
    # Run-time sidebar is just folder + model + run; tuning lives post-Run
    page.locator('.st-key-cov_folder_text textarea').fill(str(root))
    page.locator('.st-key-run_cov button').click()
    page.wait_for_selector('.st-key-cov_heatmap', timeout=300000)
    wait_idle(page, timeout=120000)
    _no_exception(page)

    # tuning row is in the main area now: lower the per-cell floor live (no re-Run)
    page.locator('.st-key-cov_t_abs input').fill("4")
    page.keyboard.press("Tab")
    wait_idle(page)
    expect(page.locator('.st-key-cov_heatmap')).to_be_visible()  # re-rendered, no re-Run

    # (a) calibration editor is present (uncalibrated warning visible first)
    expect(page.get_by_text(re.compile("未校正真實分佈"))).to_be_visible()
    # open the calibration expander via its summary (avoid matching the
    # same words in the feature-map popover)
    page.locator('[data-testid="stExpander"] summary'
                 ).filter(has_text="真實分佈校正").first.click()
    wait_idle(page)
    expect(page.locator('.st-key-cov_apply_freq')).to_be_visible()

    # (b) pick a cell → set the candidate pool in the popover, mine, export
    page.locator('.st-key-cov_cell_pick [data-baseweb="select"]').click()
    page.get_by_role("option").nth(1).click()
    wait_idle(page)
    page.get_by_text("🔎 撈候選補此格").click()  # open the pool popover
    pool_box = page.locator('.st-key-cov_pool_text textarea')
    expect(pool_box).to_be_visible()  # wait for the popover body to mount
    pool_box.fill(str(pool))
    page.keyboard.press("Tab")
    wait_idle(page)
    mine = page.locator('.st-key-cov_mine_btn button')
    expect(mine).to_be_enabled()
    mine.click()
    wait_idle(page, timeout=120000)
    expect(page.locator('.st-key-cov_cand_csv')).to_be_visible()
    _no_exception(page)


# ── (x) F6 diversity sampling / active-learning tab ─────────────────────

def test_x_diversity_sampling(flow_page):
    page = flow_page
    page.locator('.st-key-viz_panel_view').get_by_text("選樣", exact=True).click()
    wait_idle(page)
    panel = page.locator('.st-key-viz_sampling_panel')
    expect(panel).to_be_visible()
    page.locator('.st-key-viz_sampling_btn button').click()
    wait_idle(page)
    # diverse picks render as a ranked thumbnail grid + export
    imgs = panel.locator('[data-testid="stImage"] img')
    expect(imgs.first).to_be_visible()
    expect(page.locator('.st-key-viz_sampling_csv')).to_be_visible()
    # add all to the export list, verify it grew
    page.locator('.st-key-viz_sampling_addall button').click()
    wait_idle(page)
    page.locator('.st-key-viz_panel_view').get_by_text("匯出清單", exact=True).click()
    wait_idle(page)
    expect(page.get_by_text(re.compile(r"共 [1-9]\d* 張"))).to_be_visible()
    page.locator('.st-key-viz_panel_view').get_by_text("選取", exact=True).click()
    wait_idle(page)
    _no_exception(page)


# ── (y) §1 annotator-agreement quiz tool ────────────────────────────────

def test_y_quiz_tool(app_page, tmp_path):
    import numpy as np
    from PIL import Image
    page = app_page
    root = tmp_path / "quizds" / "train"
    for ci, cls in enumerate(("alpha", "beta")):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(10):
            arr = np.random.default_rng(ci * 50 + i).integers(0, 255, (64, 64, 3)).astype("uint8")
            arr[:, :, ci] = 220
            Image.fromarray(arr).save(d / f"{cls}_{i}.jpg", quality=90)

    page.locator('.st-key-tool_switch').get_by_text("組考卷", exact=True).click()
    wait_idle(page)
    page.locator('.st-key-quiz_folder_text textarea').fill(str(root))
    page.locator('.st-key-run_quiz button').click()
    wait_idle(page, timeout=300000)
    _no_exception(page)

    # generate a quiz, then answer every question blind
    page.locator('.st-key-quiz_gen button').click()
    page.wait_for_selector('[class*="st-key-quiz_ans_"] button', timeout=30000)
    done = page.get_by_text(re.compile("作答完成"))
    for _ in range(60):  # bounded; answer until the report appears
        if done.count():
            break
        btns = page.locator('[class*="st-key-quiz_ans_"] button')
        if btns.count() == 0:
            break
        btns.first.click()
        wait_idle(page)
    expect(done).to_be_visible()
    expect(page.get_by_text("自我一致率", exact=True)).to_be_visible()
    expect(page.get_by_text("vs golden 一致", exact=True)).to_be_visible()
    expect(page.locator('.st-key-quiz_answers_csv')).to_be_visible()
    _no_exception(page)


# ── (z) §3 gray-zone review (propose → approve double sign-off) ─────────

def test_z_gray_zone_review(app_page, tmp_path):
    import numpy as np
    from PIL import Image
    page = app_page
    root = tmp_path / "grayds" / "train"
    for ci, cls in enumerate(("p", "q")):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(10):
            arr = np.random.default_rng(ci * 30 + i).integers(0, 255, (64, 64, 3)).astype("uint8")
            arr[:, :, ci] = 200
            Image.fromarray(arr).save(d / f"{cls}_{i}.jpg", quality=90)

    page.locator('.st-key-tool_switch').get_by_text("灰帶覆核", exact=True).click()
    wait_idle(page)
    page.locator('.st-key-gray_folder_text textarea').fill(str(root))
    page.locator('.st-key-run_gray button').click()
    wait_idle(page, timeout=300000)
    _no_exception(page)
    # OVERVIEW: one-line backlog + triage grid + the three batch actions
    expect(page.get_by_text(re.compile("灰帶待 audit")).first).to_be_visible()
    expect(page.get_by_text(re.compile("總覽")).first).to_be_visible()
    expect(page.locator('.st-key-gray_to_lbl button').first).to_be_visible()
    expect(page.locator('.st-key-gray_soft_btn button').first).to_be_visible()
    # click 對照 on a grid cell → FOCUS pair view (gray vs anchors)
    page.locator('[class*="st-key-gray_focus_"] button').first.click()
    wait_idle(page)
    expect(page.get_by_text(re.compile("最近他類錨例")).first).to_be_visible()
    expect(page.get_by_text(re.compile("這一張的處置")).first).to_be_visible()
    # single-item disposition → back to overview → disposition export appears
    page.locator('[class*="st-key-gray_f_soft_"] button').first.click()
    wait_idle(page)
    page.locator('.st-key-gray_back button').first.click()
    wait_idle(page)
    expect(page.locator('.st-key-gray_disp_csv')).to_be_visible()
    _no_exception(page)


# ── (aa) curation log: record selection + reason, persist, re-select ────

def test_aa_curation_log(flow_page):
    page = flow_page
    _switch_panel(page, "選取")
    if _selected_count(page) == 0:
        _click_marker(page, 0, 1)
    n = _selected_count(page)
    assert n >= 1

    def _open_log():
        # idempotent: only expand if collapsed (clicking an open expander
        # toggles it closed, and clear keeps the widget's open state)
        if not page.locator('.st-key-viz_cur_reason input').is_visible():
            page.locator('[data-testid="stExpander"] summary'
                         ).filter(has_text="策展日誌").first.click()
            wait_idle(page)

    _open_log()
    page.locator('.st-key-viz_cur_reason input').fill("e2e：疑似灰帶一批")
    page.keyboard.press("Tab")
    wait_idle(page)
    page.locator('.st-key-viz_cur_log button').click()
    wait_idle(page)
    # the entry is now logged (survives restart on disk) + listed with re-select
    expect(page.get_by_text(re.compile("e2e：疑似灰帶一批")).first).to_be_visible()
    expect(page.locator('.st-key-viz_cur_csv')).to_be_visible()
    # clear, then re-select from the log → selection comes back by sha256
    _click_wait_status(page, '.st-key-viz_clear_btn button')
    assert "未選取" in _status_text(page)
    _open_log()
    re_btn = page.locator('[class*="st-key-viz_cur_re_"] button').first
    re_btn.scroll_into_view_if_needed()
    re_btn.click()
    page.wait_for_function(
        """() => { const el = document.querySelector('.st-key-viz_status_line');
                   return el && el.innerText.includes('已選取'); }""", timeout=10000)
    wait_idle(page)
    assert _selected_count(page) == n, "re-select from the log restores the batch"
    _no_exception(page)


# ── (ab) legend select-all / deselect-all buttons (client-side) ─────────

def test_ab_legend_toggle_buttons(flow_page):
    page = flow_page
    _switch_panel(page, "選取")
    wrap = page.locator('.st-key-viz_scatter_wrap')
    # plotly updatemenus render the two buttons inside the chart svg
    expect(wrap.get_by_text("全選類別").first).to_be_visible()
    expect(wrap.get_by_text("全不選").first).to_be_visible()
    # clicking them is a client-side restyle (no Streamlit rerun, no
    # selection reset) — just prove it does not raise
    wrap.get_by_text("全不選").first.click()
    wrap.get_by_text("全選類別").first.click()
    _no_exception(page)


# ── (ac) evaluation tool — one-click demo renders recall + escape gallery ─

def test_ac_evaluation_demo(app_page):
    page = app_page
    page.locator('.st-key-tool_switch').get_by_text("評估", exact=True).click()
    wait_idle(page)
    # the quick-start demo button synthesizes predictions+consensus from coco8
    demo = page.locator('.st-key-eval_demo_btn button')
    expect(demo).to_be_visible()
    demo.click()
    wait_idle(page, timeout=120000)
    # purpose is legible: a recall metric, an escape (FN) count, and the gallery
    expect(page.get_by_text("整體 recall", exact=True)).to_be_visible()
    expect(page.get_by_text("漏抓 FN（escape）", exact=True)).to_be_visible()
    expect(page.get_by_text(re.compile(r"漏抓畫廊（escape"))).to_be_visible()
    _no_exception(page)
