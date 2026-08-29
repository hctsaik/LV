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
def single_class_dataset(tmp_path_factory) -> Path:
    """1 class x 12 JPGs — probe_eval AC-G3: 類別不足時要誠實 warning、不入帳、不崩潰。"""
    root = tmp_path_factory.mktemp("ds_single")
    train = root / "train"
    rng = np.random.default_rng(7)
    d = train / "onlyclass"
    d.mkdir(parents=True)
    for i in range(12):
        arr = rng.integers(0, 255, (64, 64, 3)).astype("uint8")
        arr[:, :, 0] = 255
        Image.fromarray(arr).save(d / f"only_{i:02d}.jpg", quality=85)
    return train


@pytest.fixture(scope="session")
def e2e_cache_dir(tmp_path_factory) -> Path:
    """All ordinary E2E cache artifacts live outside the developer worktree."""
    return tmp_path_factory.mktemp("lv_cache")


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


@pytest.fixture(scope="session")
def explain_app_server(synthetic_dataset, tmp_path_factory, e2e_cache_dir) -> str:
    """Dedicated server for on-demand explainability E2E.

    The ordinary ``app_server`` intentionally shares the developer cache for
    legacy browser tests.  This one isolates just the new test so its cache
    non-mutation assertion neither reads nor changes that shared state.
    """
    port = _free_port()
    state = tmp_path_factory.mktemp("lv_explain_state")
    log_path = tmp_path_factory.mktemp("lv_explain_log") / "streamlit.log"
    env = {
        **os.environ,
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_SERVER_HEADLESS": "true",
        "LV_CACHE_DIR": str(e2e_cache_dir),
        "LV_UI_STATE": str(state / "no_persist" / "ui_state.json"),
    }
    log = open(log_path, "w", encoding="utf-8")
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
                "streamlit (explainability) exited early:\n"
                + log_path.read_text(encoding="utf-8")[-3000:]
            )
        try:
            with urllib.request.urlopen(f"{base}/_stcore/health", timeout=2) as r:
                if r.read().decode().strip() == "ok":
                    break
        except OSError:
            time.sleep(0.5)
    else:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
        log.close()
        raise RuntimeError("streamlit (explainability) did not become healthy in 60s")
    try:
        yield base
    finally:
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


# ── AL-loop (M9) isolated server ─────────────────────────────────────────────
# pool_registry / round_ledger / labeling_handoff persist to disk keyed off
# LV_CACHE_DIR / LV_OUTPUT_DIR / CIM_LOG_DIR. The shared app_server deliberately
# does NOT isolate those (dev-machine .lv_cache is real), so AL-loop E2E needs a
# dedicated server whose three durable roots are per-test temp dirs — otherwise
# "second send blocked" and "round survives restart" would be polluted by (and
# pollute) real state. This server also supports .restart() for AC-G3.
class _ManagedServer:
    def __init__(self, env: dict, cache_dir: Path, output_dir: Path, log_dir: Path):
        self.env = env
        self.cache_dir = cache_dir      # LV_CACHE_DIR   → pool_registry.jsonl
        self.output_dir = output_dir    # LV_OUTPUT_DIR  → rounds.jsonl
        self.log_dir = log_dir          # CIM_LOG_DIR    → lv_labeling_handoff/
        self._proc = None
        self._log = None
        self.base_url = ""

    @property
    def registry_path(self) -> Path:
        return self.cache_dir / "pool_registry.jsonl"

    @property
    def rounds_path(self) -> Path:
        return self.output_dir / "rounds.jsonl"

    @property
    def handoff_root(self) -> Path:
        return self.log_dir / "lv_labeling_handoff"

    def start(self) -> str:
        port = _free_port()
        self._log = open(REPO_ROOT / "tests" / "e2e" / "_server_al.log", "w",
                         encoding="utf-8")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "scripts/app.py",
             "--server.port", str(port), "--server.headless", "true",
             "--server.fileWatcherType", "none",
             "--browser.gatherUsageStats", "false"],
            cwd=REPO_ROOT, env=self.env, stdout=self._log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        base = f"http://localhost:{port}"
        deadline = time.time() + 60
        while time.time() < deadline:
            if self._proc.poll() is not None:
                self._log.close()
                raise RuntimeError(
                    "streamlit (AL) exited early:\n"
                    + (REPO_ROOT / "tests" / "e2e" / "_server_al.log")
                    .read_text(encoding="utf-8")[-3000:])
            try:
                with urllib.request.urlopen(f"{base}/_stcore/health", timeout=2) as r:
                    if r.read().decode().strip() == "ok":
                        break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("streamlit (AL) did not become healthy in 60s")
        self.base_url = base
        return base

    def stop(self) -> None:
        if self._proc is not None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self._proc.pid)],
                           capture_output=True)
            self._proc = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def restart(self) -> str:
        self.stop()
        return self.start()


@pytest.fixture(scope="session")
def _al_dirs(tmp_path_factory) -> dict:
    # LV_CACHE_DIR is SHARED across AL tests so the DINOv2 embedding cache is
    # computed once (first test) and reused — otherwise a fresh cache per test
    # means a cold ~90s recompute every time, blowing the E2E timeouts. Per-test
    # isolation of the AL state (pool_registry.jsonl / rounds.jsonl / handoffs) is
    # done by al_isolated_server deleting just those files, leaving the cache.
    return {
        "cache": tmp_path_factory.mktemp("al_cache"),
        "output": tmp_path_factory.mktemp("al_output"),
        "cim": tmp_path_factory.mktemp("al_cimlog"),
        "state": tmp_path_factory.mktemp("al_state"),
    }


@pytest.fixture(scope="module")
def al_server(_al_dirs) -> "_ManagedServer":
    env = {
        **os.environ,
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_SERVER_HEADLESS": "true",
        "LV_CACHE_DIR": str(_al_dirs["cache"]),
        "LV_OUTPUT_DIR": str(_al_dirs["output"]),
        "CIM_LOG_DIR": str(_al_dirs["cim"]),
        # disable folder/option persistence (parent no_persist/ intentionally absent)
        "LV_UI_STATE": str(_al_dirs["state"] / "no_persist" / "ui_state.json"),
    }
    srv = _ManagedServer(env, _al_dirs["cache"], _al_dirs["output"], _al_dirs["cim"])
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture()
def al_isolated_server(al_server) -> "_ManagedServer":
    """Per-test clean slate for the AL durable state (pool registry / round ledger
    / handoff folders) while preserving the shared embedding cache. The server
    reads these files fresh on every render, so deleting them between tests is a
    genuine reset without a restart."""
    import shutil
    srv = al_server
    srv.registry_path.unlink(missing_ok=True)
    srv.rounds_path.unlink(missing_ok=True)
    # readback overlay persists across Run by design — reset it per test too, or a
    # prior test's corrections would pre-apply and make this test's read-back a no-op.
    (srv.cache_dir / "readback_changelog.jsonl").unlink(missing_ok=True)
    shutil.rmtree(srv.cache_dir / "readback_versions", ignore_errors=True)
    if srv.handoff_root.exists():
        shutil.rmtree(srv.handoff_root, ignore_errors=True)
    yield srv


def newest_handoff_dir(server: "_ManagedServer") -> Path:
    """The handoff folder just created by a send (newest under CIM_LOG_DIR)."""
    root = server.handoff_root
    dirs = [d for d in root.iterdir() if d.is_dir()] if root.exists() else []
    assert dirs, f"no handoff folder created under {root}"
    return max(dirs, key=lambda d: d.stat().st_mtime)


def write_annotation_sidecars(handoff_dir: Path, label: str) -> int:
    """Simulate Labeling finishing the batch: write an xAnyLabeling sidecar
    (images/<sha>.json with shapes[0].label) next to every exported image, so
    read_labeling_results/apply_readback see a completed annotation. Returns the
    number of items annotated."""
    import json as _json
    spec = _json.loads((handoff_dir / "_handoff.json").read_text(encoding="utf-8"))
    img_dir = handoff_dir / "images"
    n = 0
    for it in spec.get("items", []):
        ann = (img_dir / Path(it["image"]).name).with_suffix(".json")
        ann.write_text(_json.dumps({"shapes": [{"label": label}]}),
                       encoding="utf-8")
        n += 1
    return n
