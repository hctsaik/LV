"""E2E fixtures: synthetic dataset + live Streamlit server + loaded page."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory) -> Path:
    """2 classes x 12 colored-noise JPGs in classifier layout (split=train)."""
    root = tmp_path_factory.mktemp("ds")
    train = root / "train"
    rng = np.random.default_rng(0)
    for ci, cls in enumerate(("classA", "classB")):
        d = train / cls
        d.mkdir(parents=True)
        for i in range(12):
            arr = rng.integers(0, 255, (64, 64, 3)).astype("uint8")
            arr[:, :, ci] = 255  # give each class a colour bias
            Image.fromarray(arr).save(d / f"{cls}_{i:02d}.jpg", quality=85)
    return train


@pytest.fixture(scope="session")
def app_server(synthetic_dataset, tmp_path_factory) -> str:
    port = _free_port()
    # 隔離持久化狀態:E2E 用乾淨的 UI-state 檔與 cache 目錄,不共用/不殘留開發者本機狀態,
    # 也避免 pytest 保留前幾次 tmp 目錄導致「上輪殘留資料夾」被還原而污染本輪(跨 run 汙染)。
    _state = tmp_path_factory.mktemp("lv_state")
    env = {
        **os.environ,
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_SERVER_HEADLESS": "true",
        # 父目錄 no_persist/ 刻意不建立 → _ui_state_path 的讀(load)寫(save)皆 OSError→no-op,
        # 等於 E2E 關閉「記住上次資料夾/選項」的持久化:每個測試(獨立 browser context)從乾淨
        # session_state 起跑,folder list 不會經由磁碟檔在測試之間殘留(跨測試/跨 run 汙染根因)。
        # 註:不隔離 LV_CACHE_DIR —— 有測試(test_b/test_t)在「測試行程」用 manifest_path_for/
        # ref_path_for(無此 env)直接驗證 .lv_cache 內容,隔離只會讓測試與 server 看不同目錄。
        "LV_UI_STATE": str(_state / "no_persist" / "ui_state.json"),
    }
    log = open(REPO_ROOT / "tests" / "e2e" / "_server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "scripts/app.py",
         "--server.port", str(port), "--server.headless", "true",
         "--server.fileWatcherType", "none",
         "--browser.gatherUsageStats", "false"],
        cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    base = f"http://localhost:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            log.close()
            raise RuntimeError(
                "streamlit exited early:\n"
                + (REPO_ROOT / "tests" / "e2e" / "_server.log").read_text(encoding="utf-8")[-3000:]
            )
        try:
            with urllib.request.urlopen(f"{base}/_stcore/health", timeout=2) as r:
                if r.read().decode().strip() == "ok":
                    break
        except OSError:
            time.sleep(0.5)
    else:
        raise RuntimeError("streamlit did not become healthy in 60s")
    yield base
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                   capture_output=True)
    log.close()


def wait_idle(page, timeout: int = 30000) -> None:
    """Wait until Streamlit's running-man status widget is gone/hidden."""
    page.wait_for_function(
        """() => {
            const w = document.querySelector('[data-testid="stStatusWidget"]');
            return !w || w.offsetParent === null;
        }""",
        timeout=timeout,
    )


def _add_folder(page, list_key, *paths):
    """Add folder(s) via the small '輸入路徑→Enter' field (replaces the old paste
    textarea). Accepts multiple path args, or a single newline-joined string
    (legacy paste form); each non-empty line is added with fill + Enter, which
    fires the on_change that appends it to the managed folder list."""
    items: list[str] = []
    for p in paths:
        items.extend(str(p).splitlines() or [str(p)])
    sel = f'.st-key-{list_key}_add input'
    inp = page.locator(sel)
    for it in items:
        it = it.strip()
        if not it:
            continue
        inp.fill(it)
        inp.press("Enter")
        # The on_change appends the path and clears the field. Wait for that
        # clear before typing the next path: wait_idle can return in the gap
        # between Enter and the rerun starting, letting the next fill clobber
        # this value before on_change reads it (dropping a folder).
        page.wait_for_function(
            "(s) => { const el = document.querySelector(s); return el && el.value === ''; }",
            arg=sel, timeout=15000)
        wait_idle(page)


def _sidebar_visible(page) -> bool:
    """True once the auto-collapse CSS (visibility:hidden / width:0) is gone."""
    return page.evaluate(
        """() => {
            const sb = document.querySelector('[data-testid="stSidebar"]');
            if (!sb) return false;
            const cs = getComputedStyle(sb);
            return cs.visibility !== 'hidden' && sb.offsetWidth > 0;
        }"""
    )


def _ensure_sidebar(page) -> None:
    """Re-expand the left settings sidebar if a prior Run auto-collapsed it.

    The Run button lives inside the sidebar, so a second Run in the same
    session must first click the '☰ 顯示左側設定列' reopen button (a no-op
    when the sidebar is already open — the button only renders while
    collapsed).

    The reopen click triggers a flag-flip rerun whose status widget can come
    and go faster than wait_idle polls — returning before the collapse CSS is
    actually torn down would leave run_viz (inside the sidebar) hidden and the
    caller's click would time out. So click, then WAIT for the sidebar to be
    genuinely visible again (collapse CSS gone), retrying the click if the
    first one's rerun raced us. Tool-agnostic: it waits on the sidebar itself,
    not on any per-tool widget."""
    for _ in range(3):
        if _sidebar_visible(page):
            return
        btn = page.locator('.st-key-reopen_sidebar button')
        if not (btn.count() and btn.first.is_visible()):
            # No reopen affordance yet (rerun still settling): let it settle
            # and re-check rather than declaring success on a collapsed page.
            wait_idle(page)
            continue
        btn.first.click()
        wait_idle(page)
        try:
            page.wait_for_function(
                """() => {
                    const sb = document.querySelector('[data-testid="stSidebar"]');
                    if (!sb) return false;
                    const cs = getComputedStyle(sb);
                    return cs.visibility !== 'hidden' && sb.offsetWidth > 0;
                }""",
                timeout=20000,
            )
            return
        except Exception:
            # rerun raced the click (flag flip swallowed) — loop and retry
            continue


def load_app(page, base_url: str) -> None:
    """Navigate and wait until the first script run has fully rendered.

    The first run on a fresh server is slow (torch/umap imports); wait for
    the page title instead of a fixed sleep.
    """
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_selector('[data-testid="stAppViewContainer"]')
    page.get_by_text("Dataset Analysis Tools").wait_for(timeout=120000)
    wait_idle(page, timeout=120000)


@pytest.fixture()
def app_page(app_server, page):
    page.set_default_timeout(20000)
    load_app(page, app_server)
    return page
