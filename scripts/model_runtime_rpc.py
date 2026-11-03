"""Fixed local inference protocol for comparisons spanning pinned environments.

The subprocess accepts no callable names or executable paths. It loads one model,
serves inference serially, and exits when its parent worker disappears.
"""
from __future__ import annotations

import base64
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import weakref

from model_runtimes import RELEASES, environment, interpreter, runtime_id


def _stop(process, log):
    # On Windows the venv launcher has a child interpreter; closing stdin lets
    # that interpreter exit too instead of killing only the launcher.
    if process.stdin:
        try:
            process.stdin.close()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            process.kill()
        process.wait(timeout=5)
    if process.stdout:
        process.stdout.close()
    log.close()


class RuntimeProxy:
    def __init__(self, config, *, trusted=False):
        if not trusted:
            raise ValueError("模型推論需要明確的模型匯入操作")
        import tempfile
        selected = runtime_id(config)
        self.execution_runtime = {"runtime_id": selected, "runtime_release": RELEASES[selected]}
        self._lock = threading.RLock()
        log = tempfile.TemporaryFile()
        self._process = subprocess.Popen([str(interpreter(selected)), str(Path(__file__).resolve()), "--worker", str(os.getpid())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
            env=environment(selected), creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self._finalizer = weakref.finalize(self, _stop, self._process, log)
        try:
            response = self._call({"op": "initialize", "config": config})
            self.config, self.identity = response["config"], response["identity"]
            self.identity["execution_runtime"] = dict(self.execution_runtime)
        except Exception:
            self.close()
            raise

    def _call(self, request):
        with self._lock:
            try:
                self._process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
                self._process.stdin.flush()
                line = self._process.stdout.readline()
            except (OSError, BrokenPipeError) as exc:
                raise RuntimeError("模型子程序中斷，可重試工作") from exc
            if not line:
                raise RuntimeError("模型子程序未回傳結果，可重試工作")
            response = json.loads(line)
            if not response.get("ok"):
                raise ValueError(response.get("error", "模型子程序失敗"))
            return response["result"]

    def infer(self, data, artifact_root=None):
        result = self._call({"op": "infer", "data": base64.b64encode(data).decode("ascii"),
                             "artifact_root": str(artifact_root) if artifact_root is not None else None})
        result["execution_runtime"] = dict(self.execution_runtime)
        return result

    def close(self):
        self._finalizer()


def _serve(parent_pid):
    from investigation_jobs import _pid_alive, _memory_bytes
    deadline = [time.monotonic() + 1800]
    def watchdog():
        while True:
            memory = _memory_bytes(os.getpid())
            if not _pid_alive(parent_pid) or time.monotonic() > deadline[0] or (memory is not None and memory > 4 * 1024**3):
                os._exit(1)
            time.sleep(1)
    threading.Thread(target=watchdog, daemon=True).start()
    model = None
    for line in sys.stdin:
        deadline[0] = time.monotonic() + 1800
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request.get("op") == "initialize" and model is None:
                    config = request["config"]
                    if runtime_id(config) != os.environ["LV_MODEL_RUNTIME_ID"]:
                        raise ValueError("模型執行版本與子程序不一致")
                    from explain_service import runtime_for
                    model = runtime_for(config, trusted=True)
                    result = {"config": model.config, "identity": model.identity}
                elif request.get("op") == "infer" and model is not None:
                    data = base64.b64decode(request["data"], validate=True)
                    result = model.infer(data, request.get("artifact_root"))
                else:
                    raise ValueError("Unsupported model worker operation")
            response = {"ok": True, "result": result}
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--worker":
    _serve(int(sys.argv[2]))
