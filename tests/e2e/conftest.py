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
def app_server(synthetic_dataset) -> str:
    port = _free_port()
    env = {
        **os.environ,
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_SERVER_HEADLESS": "true",
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
