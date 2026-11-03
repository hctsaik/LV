"""Developer-owned interpreter registry, independent of uploaded model data."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LEGACY = "legacy-8.3.203"
MODERN = "modern-8.4.142"
RELEASES = {LEGACY: "8.3.203", MODERN: "8.4.142"}


def runtime_id(config=None):
    config = config or {}
    value = config.get("runtime_id") or LEGACY
    if value not in RELEASES:
        raise ValueError("未知模型執行環境；請使用 LV 提供的版本")
    release = config.get("runtime_release")
    if release and release != RELEASES[value]:
        raise ValueError("模型執行環境與套件版本不一致")
    return value


def interpreter(value=LEGACY, *, require_exists=True):
    value = runtime_id({"runtime_id": value})
    suffix = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    # Resolve legacy from the repository, even when called inside modern workers.
    path = ROOT / ".venv" / suffix if value == LEGACY else ROOT / ".runtimes" / "yolo-modern-8.4.142" / suffix
    if require_exists and not path.is_file():
        raise ValueError("模型執行環境不存在；新版請執行 .venv/Scripts/python.exe scripts/setup_model_runtime.py；舊版請還原專案 .venv")
    if require_exists and value == MODERN and not _ready(path):
        raise ValueError("新版模型環境安裝尚未完成；請執行 .venv/Scripts/python.exe scripts/setup_model_runtime.py")
    return path


def _ready(path):
    try:
        marker = json.loads((path.parent.parent / "runtime-ready.json").read_text(encoding="utf-8"))
        return marker.get("release") == RELEASES[MODERN] and marker.get("pip_check") is True
    except (OSError, ValueError):
        return False


def environment(value=LEGACY):
    """Modern packages must never be shadowed by the legacy virtual environment."""
    value = runtime_id({"runtime_id": value})
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(name, "1")
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["LV_MODEL_RUNTIME_ID"] = value
    if value == MODERN:
        settings = ROOT / ".runtimes" / "yolo-modern-8.4.142" / "settings"
        settings.mkdir(parents=True, exist_ok=True)
        env["YOLO_CONFIG_DIR"] = str(settings)
    paths = [str(ROOT / "scripts"), str(ROOT)]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def available():
    rows = []
    for key, release in RELEASES.items():
        path = interpreter(key, require_exists=False)
        state = "missing" if not path.is_file() else "ready" if key == LEGACY or _ready(path) else "incomplete"
        rows.append({"id": key, "release": release, "installed": state == "ready", "state": state})
    return rows


def probe(value):
    """Read actual package origin/version without importing torch into the UI."""
    code = ("import json,sys,importlib.metadata as m,importlib.util as u; "
            "print(json.dumps({'python':sys.executable,'release':m.version('ultralytics'),"
            "'package_origin':u.find_spec('ultralytics').origin}))")
    result = subprocess.run([str(interpreter(value)), "-c", code], env=environment(value),
                            capture_output=True, text=True, check=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    info = json.loads(result.stdout.strip())
    if info["release"] != RELEASES[value]:
        raise ValueError("模型執行環境套件版本不符，請重新建立環境")
    return {"runtime_id": value, **info}
