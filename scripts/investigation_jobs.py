"""Persistent local investigation jobs with cancellable isolated workers.

UI usage: manager = JobManager.get(root, {"inspect": "fp_attribution:inspect_task"});
job = manager.submit("inspect", payload, compute_key, context={"case_id": ...}).
Poll manager.store.get(job['id']); completed results survive case publication errors.
Handlers are developer-registered import paths, never fields from uploaded payloads.
Each handler accepts (payload, context), returns JSON, and may write artifacts under
context.artifact_dir / relative_name or call context.write_artifact(name, bytes).
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

from model_profiles import _atomic

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


def _worker_environment(runtime_id=None):
    """Bound worker CPU pools while preserving explicit deployment settings."""
    if runtime_id is not None:
        from model_runtimes import environment
        return environment(runtime_id)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(name, "1")
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in sys.path if p)
    env["PYTHONUTF8"] = "1"
    return env


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied is not proof of death.
        code = wintypes.DWORD()
        try:
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _memory_bytes(pid):
    """Best-effort resident/private memory accounting; returns None if unavailable."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x1000 | 0x10, False, int(pid))
            if not handle:
                return None
            try:
                counters = Counters()
                counters.cb = ctypes.sizeof(counters)
                psapi = ctypes.WinDLL("psapi", use_last_error=True)
                psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    return max(counters.PrivateUsage, counters.WorkingSetSize)
            finally:
                kernel.CloseHandle(handle)
        elif Path(f"/proc/{pid}/statm").exists():
            return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        pass
    return None


class JobCancelled(Exception):
    pass


class JobStore:
    def __init__(self, workspace_root):
        self.workspace_root = Path(workspace_root).resolve()
        self.root = self.workspace_root / "investigation_jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "jobs.sqlite"
        with self._connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, state TEXT NOT NULL, compute_key TEXT NOT NULL, body TEXT NOT NULL)")
            con.execute("CREATE INDEX IF NOT EXISTS job_compute ON jobs(compute_key)")

    def _connect(self):
        return sqlite3.connect(self.db, timeout=30)

    def submit(self, task, payload, compute_key, *, context=None, retry_of=None):
        # Include task/payload identity to prevent an accidental caller-key collision.
        request_hash = hashlib.sha256(_json({"task": task, "payload": payload, "context": context or {}}).encode()).hexdigest()
        if not task or not compute_key:
            raise ValueError("Job task and compute identity are required")
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            rows = con.execute("SELECT body FROM jobs WHERE compute_key=? AND state IN ('queued','running','succeeded')", (str(compute_key),)).fetchall()
            for row in rows:
                previous = json.loads(row[0])
                if previous["request_hash"] == request_hash:
                    return previous
            job = {"schema_version": 1, "id": "job_" + uuid.uuid4().hex[:20], "task": str(task),
                   "payload": payload, "context": context or {}, "compute_key": str(compute_key),
                   "request_hash": request_hash, "state": "queued", "created_at": _now(),
                   "updated_at": _now(), "progress": 0.0, "stage": "queued", "error": None,
                   "retry_of": retry_of, "owner_pid": None, "worker_pid": None,
                   "publication": {"state": "not_published"}, "artifacts": []}
            con.execute("INSERT INTO jobs VALUES (?,?,?,?)", (job["id"], job["state"], job["compute_key"], _json(job)))
        return json.loads(_json(job))

    def get(self, job_id):
        with self._connect() as con:
            row = con.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise ValueError("Unknown investigation job")
        return json.loads(row[0])

    def list(self, *, case_id=None, states=None):
        with self._connect() as con:
            rows = con.execute("SELECT body FROM jobs ORDER BY rowid DESC").fetchall()
        jobs = [json.loads(r[0]) for r in rows]
        return [j for j in jobs if (case_id is None or j["context"].get("case_id") == case_id)
                and (states is None or j["state"] in states)]

    def _update(self, job_id, changes, *, expected_states=None):
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise ValueError("Unknown investigation job")
            job = json.loads(row[0])
            if expected_states is not None and job["state"] not in expected_states:
                return None
            job.update(changes, updated_at=_now())
            con.execute("UPDATE jobs SET state=?,body=? WHERE id=?", (job["state"], _json(job), job_id))
        return job

    def cancel(self, job_id):
        return self._update(job_id, {"state": "cancelled", "stage": "cancelled", "finished_at": _now()}, expected_states={"queued", "running"}) or self.get(job_id)

    def retry(self, job_id):
        old = self.get(job_id)
        if old["state"] not in {"failed", "cancelled", "interrupted"}:
            raise ValueError("Only failed, cancelled or interrupted jobs can be retried")
        return self.submit(old["task"], old["payload"], old["compute_key"], context=old["context"], retry_of=old["id"])

    def recover_interrupted(self):
        changed = []
        for job in self.list(states={"running"}):
            if not _pid_alive(job.get("owner_pid")):
                value = self._update(job["id"], {"state": "interrupted", "stage": "interrupted", "finished_at": _now(),
                                              "error": "LV stopped before this job completed; retry is available"}, expected_states={"running"})
                if value:
                    changed.append(value)
        return changed

    def artifact_dir(self, job_id):
        self.get(job_id)
        return self.root / job_id / "artifacts"

    def result(self, job_id):
        job = self.get(job_id)
        if job["state"] != "succeeded":
            raise ValueError("Job has no completed result")
        root = self.artifact_dir(job_id)
        for artifact in job["artifacts"]:
            path = (root / artifact["path"]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                raise ValueError("Job artifact is missing or corrupted")
        return json.loads((root / "result.json").read_text(encoding="utf-8"))

    def mark_published(self, job_id, *, case_id=None, evidence_id=None, error=None):
        job = self.get(job_id)
        if job["state"] != "succeeded":
            raise ValueError("Only completed results can be published")
        publication = {"state": "conflict" if error else "published", "case_id": case_id,
                       "evidence_id": evidence_id, "error": str(error) if error else None, "at": _now()}
        return self._update(job_id, {"publication": publication}, expected_states={"succeeded"})


class JobContext:
    def __init__(self, store, job_id):
        self.store, self.job_id = store, job_id
        self.workspace_root = store.workspace_root
        self.artifact_dir = store.root / job_id / "pending"
        self.final_artifact_dir = store.root / job_id / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def check_cancelled(self):
        job = self.store.get(self.job_id)
        if job["state"] != "running" or not _pid_alive(job.get("owner_pid")):
            raise JobCancelled("Investigation was cancelled or its LV session stopped")

    def progress(self, fraction, stage=""):
        self.check_cancelled()
        fraction = float(fraction)
        if not 0 <= fraction <= 1:
            raise ValueError("Progress must be between zero and one")
        self.store._update(self.job_id, {"progress": fraction, "stage": str(stage)}, expected_states={"running"})

    def write_artifact(self, name, data):
        self.check_cancelled()
        path = (self.artifact_dir / name).resolve()
        if not path.is_relative_to(self.artifact_dir.resolve()) or path == self.artifact_dir.resolve():
            raise ValueError("Artifact must be inside the job directory")
        _atomic(path, bytes(data))
        return path


def _worker(root, job_id, handler_path):
    store = JobStore(root)
    context = JobContext(store, job_id)
    watchdog_done = threading.Event()
    def watch_owner():
        while not watchdog_done.wait(1):
            try:
                current = store.get(job_id)
                if not _pid_alive(current.get("owner_pid")):
                    os._exit(1)  # The owning UI process vanished; do not leave orphan inference running.
            except (ValueError, sqlite3.Error):
                pass
    threading.Thread(target=watch_owner, name="lv-worker-owner-watch", daemon=True).start()
    try:
        context.check_cancelled()
        module, function = handler_path.split(":", 1)
        handler = getattr(importlib.import_module(module), function)
        result = handler(store.get(job_id)["payload"], context)
        def relocate(value):
            if isinstance(value, dict):
                return {key: relocate(item) for key, item in value.items()}
            if isinstance(value, list):
                return [relocate(item) for item in value]
            if isinstance(value, str):
                for source, destination in ((str(context.artifact_dir), str(context.final_artifact_dir)),
                                            (context.artifact_dir.as_posix(), context.final_artifact_dir.as_posix())):
                    if value == source or value.startswith(source + "/") or value.startswith(source + "\\"):
                        return destination + value[len(source):]
            return value
        result = relocate(result)
        context.write_artifact("result.json", _json(result).encode("utf-8"))
        context.check_cancelled()
        artifacts = []
        for path in sorted(context.artifact_dir.rglob("*")):
            if path.is_symlink():
                raise ValueError("Artifact symlinks are not supported")
            if path.is_file():
                artifacts.append({"path": path.relative_to(context.artifact_dir).as_posix(), "size": path.stat().st_size,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        destination = store.artifact_dir(job_id)
        os.replace(context.artifact_dir, destination)
        store._update(job_id, {"state": "succeeded", "progress": 1.0, "stage": "completed", "artifacts": artifacts,
                              "finished_at": _now()}, expected_states={"running"})
    except JobCancelled:
        store.cancel(job_id)
    except BaseException as exc:
        store._update(job_id, {"state": "failed", "stage": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_at": _now()}, expected_states={"running"})
    finally:
        watchdog_done.set()


class JobManager:
    _instances = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get(cls, workspace_root, handlers, **kwargs):
        key = str(Path(workspace_root).resolve())
        with cls._instances_lock:
            manager = cls._instances.get(key)
            if manager is None or manager._stop.is_set():
                manager = cls(workspace_root, handlers, **kwargs)
                cls._instances[key] = manager
            else:
                manager.handlers.update(handlers)
                if "runtime_selector" in kwargs:
                    manager.runtime_selector = kwargs["runtime_selector"]
                if "payload_normalizer" in kwargs:
                    manager.payload_normalizer = kwargs["payload_normalizer"]
            return manager

    def __init__(self, workspace_root, handlers, *, max_workers=1, timeout_seconds=900, memory_limit_mb=4096, runtime_selector=None, payload_normalizer=None):
        self.store = JobStore(workspace_root)
        self.handlers = dict(handlers)
        self.runtime_selector = runtime_selector
        self.payload_normalizer = payload_normalizer
        self.max_workers = max(1, int(max_workers))
        self.timeout_seconds = float(timeout_seconds)
        self.memory_limit_bytes = int(memory_limit_mb) * 1024 * 1024
        self._stop = threading.Event()
        self.store.recover_interrupted()
        self._thread = threading.Thread(target=self._run, name="lv-investigation-manager", daemon=True)
        self._thread.start()

    def submit(self, task, payload, compute_key, *, context=None):
        if task not in self.handlers:
            raise ValueError("Unsupported investigation task")
        if self.payload_normalizer:
            payload = self.payload_normalizer(task, payload)
        context = dict(context or {})
        if self.runtime_selector:
            context["execution_runtime_id"] = self.runtime_selector(task, payload, self.store.workspace_root)
        return self.store.submit(task, payload, compute_key, context=context)

    def cancel(self, job_id):
        return self.store.cancel(job_id)

    def retry(self, job_id):
        if self.store.get(job_id)["task"] not in self.handlers:
            raise ValueError("Task adapter is no longer available")
        return self.store.retry(job_id)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self):
        running = {}
        try:
            while not self._stop.is_set():
                for job_id, (process, started, log) in list(running.items()):
                    job = self.store.get(job_id)
                    memory = _memory_bytes(process.pid)
                    memory_exceeded = memory is not None and memory > self.memory_limit_bytes
                    if job["state"] in {"cancelled", "interrupted"} or time.monotonic() - started > self.timeout_seconds or memory_exceeded:
                        if process.poll() is None:
                            process.kill()
                        self.store._update(job_id, {"state": "failed", "stage": "failed", "error": "Worker exceeded memory limit" if memory_exceeded else "Worker exceeded time limit", "finished_at": _now()}, expected_states={"running"})
                    if process.poll() is not None:
                        log.close()
                        self.store._update(job_id, {"state": "failed", "stage": "failed", "error": f"Worker stopped without a result (exit {process.returncode})", "finished_at": _now()}, expected_states={"running"})
                        del running[job_id]
                for job in reversed(self.store.list(states={"queued"})):
                    if len(running) >= self.max_workers:
                        break
                    if job["task"] not in self.handlers:
                        continue
                    claimed = self.store._update(job["id"], {"state": "running", "stage": "starting worker", "owner_pid": os.getpid(), "started_at": _now()}, expected_states={"queued"})
                    if not claimed:
                        continue
                    directory = self.store.root / job["id"]
                    directory.mkdir(exist_ok=True)
                    log = (directory / "worker.log").open("ab")
                    try:
                        selected_runtime = self.runtime_selector(job["task"], job["payload"], self.store.workspace_root) if self.runtime_selector else None
                        if selected_runtime:
                            from model_runtimes import interpreter
                            executable = str(interpreter(selected_runtime))
                        else:
                            executable = sys.executable
                        env = _worker_environment(selected_runtime)
                        process = subprocess.Popen([executable, str(Path(__file__).resolve()), "--worker", str(self.store.workspace_root), job["id"], self.handlers[job["task"]]],
                                                   stdout=log, stderr=log, env=env,
                                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                        self.store._update(job["id"], {"worker_pid": process.pid}, expected_states={"running"})
                        running[job["id"]] = (process, time.monotonic(), log)
                    except Exception as exc:
                        log.close()
                        self.store._update(job["id"], {"state": "failed", "error": f"Could not start worker: {exc}"}, expected_states={"running"})
                self._stop.wait(0.1)
        finally:
            for job_id, (process, _, log) in running.items():
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                log.close()
                self.store._update(job_id, {"state": "interrupted", "stage": "interrupted", "error": "Worker manager stopped; retry is available", "finished_at": _now()}, expected_states={"running"})


if __name__ == "__main__" and len(sys.argv) == 5 and sys.argv[1] == "--worker":
    _worker(sys.argv[2], sys.argv[3], sys.argv[4])
