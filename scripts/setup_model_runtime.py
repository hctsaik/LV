"""Create LV's separate modern YOLO worker environment without changing legacy.

Run with the base interpreter: python scripts/setup_model_runtime.py.
Resolved dependencies are retained in a repository lock and beside the interpreter;
subsequent runs install the repository lock. The environment never inherits packages
from another environment, and no existing site-packages are edited or linked.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
import venv


def install():
    root = Path(__file__).resolve().parents[1]
    target = root / ".runtimes" / "yolo-modern-8.4.142"
    interpreter = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not interpreter.exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(target)
    marker = target / "runtime-ready.json"
    marker.unlink(missing_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    (target / "settings").mkdir(parents=True, exist_ok=True)
    env["YOLO_CONFIG_DIR"] = str(target / "settings")
    lock = target / "requirements.lock.txt"
    repository_lock = root / "requirements-yolo-modern.lock.txt"
    requirement = repository_lock if repository_lock.exists() else lock if lock.exists() else root / "requirements-yolo-modern.txt"
    subprocess.run([str(interpreter), "-m", "pip", "install", "-r", str(root / "requirements-yolo-modern.txt"), "-r", str(requirement)], env=env, check=True)
    subprocess.run([str(interpreter), "-m", "pip", "check"], env=env, check=True)
    resolved = subprocess.check_output([str(interpreter), "-m", "pip", "freeze"], env=env)
    lock.write_bytes(resolved)
    if not repository_lock.exists():
        repository_lock.write_bytes(b"# Resolved on Windows x64 / Python 3.11; other platforms require their own validation.\n" + resolved)
    subprocess.run([str(interpreter), "-c", "import importlib.metadata as m; assert m.version('ultralytics') == '8.4.142'; import torch,ultralytics; print('Modern runtime ready:',ultralytics.__version__,torch.__version__)"], env=env, check=True)
    pending = marker.with_suffix(".pending")
    pending.write_text(json.dumps({"release": "8.4.142", "pip_check": True, "python": str(interpreter)}), encoding="utf-8")
    pending.replace(marker)
    print(interpreter)


if __name__ == "__main__":
    install()
