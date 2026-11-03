import os
from pathlib import Path
import time

import pytest

from investigation_jobs import JobManager, JobStore, _pid_alive


def test_worker_environment_bounds_cpu_pools_and_preserves_overrides(monkeypatch):
    from investigation_jobs import _worker_environment
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    env = _worker_environment()
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "4"
    assert "OMP_NUM_THREADS" not in os.environ


def echo_task(payload, context):
    context.progress(.4, "writing evidence")
    context.write_artifact("evidence.txt", b"original evidence")
    return {"answer": payload["value"], "worker_pid": os.getpid()}


def slow_task(payload, context):
    context.progress(.1, "slow computation")
    time.sleep(30)
    return {"unexpected": True}


def error_task(payload, context):
    raise ValueError("Checkpoint requires a supported architecture")


def escape_task(payload, context):
    context.write_artifact("../../escape.txt", b"no")


def _wait(store, job_id, states, timeout=20):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        job = store.get(job_id)
        if job["state"] in states:
            return job
        time.sleep(.05)
    pytest.fail(f"Job did not reach {states}: {store.get(job_id)}")


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    value = JobManager(tmp_path, {"echo": "test_investigation_jobs:echo_task", "slow": "test_investigation_jobs:slow_task",
                                 "error": "test_investigation_jobs:error_task", "escape": "test_investigation_jobs:escape_task"})
    yield value
    value.close()


def test_completion_isolated_duplicate_and_publish_conflict_preserves_evidence(manager):
    first = manager.submit("echo", {"value": 42}, "same-input", context={"case_id": "case-a"})
    duplicate = manager.submit("echo", {"value": 42}, "same-input", context={"case_id": "case-a"})
    assert duplicate["id"] == first["id"]
    done = _wait(manager.store, first["id"], {"succeeded", "failed"})
    assert done["state"] == "succeeded", done
    result = manager.store.result(first["id"])
    assert result["answer"] == 42
    assert result["worker_pid"] != os.getpid()
    manager.store.mark_published(first["id"], case_id="case-a", error="Case revision changed")
    restarted = JobStore(manager.store.workspace_root)
    assert restarted.get(first["id"])["publication"]["state"] == "conflict"
    assert restarted.result(first["id"])["answer"] == 42
    restarted.mark_published(first["id"], case_id="case-a", evidence_id="e1")
    assert restarted.get(first["id"])["publication"]["state"] == "published"
    assert manager.submit("echo", {"value": 42}, "same-input", context={"case_id": "case-a"})["id"] == first["id"]
    different = manager.submit("echo", {"value": 43}, "same-input", context={"case_id": "case-a"})
    assert different["id"] != first["id"]
    (restarted.artifact_dir(first["id"]) / "evidence.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="corrupted"):
        restarted.result(first["id"])


def test_cancel_kills_running_worker_and_retry_has_new_attempt(manager):
    job = manager.submit("slow", {}, "long")
    running = _wait(manager.store, job["id"], {"running"})
    # Wait until process id has been persisted.
    deadline = time.monotonic() + 10
    while not running["worker_pid"] and time.monotonic() < deadline:
        time.sleep(.05)
        running = manager.store.get(job["id"])
    assert running["worker_pid"]
    manager.cancel(job["id"])
    deadline = time.monotonic() + 10
    while _pid_alive(running["worker_pid"]) and time.monotonic() < deadline:
        time.sleep(.05)
    assert not _pid_alive(running["worker_pid"])
    assert manager.store.get(job["id"])["state"] == "cancelled"
    with pytest.raises(ValueError, match="no completed"):
        manager.store.result(job["id"])
    retry = manager.retry(job["id"])
    assert retry["id"] != job["id"] and retry["retry_of"] == job["id"]
    manager.cancel(retry["id"])


def test_recovery_does_not_interrupt_other_live_owner(tmp_path):
    store = JobStore(tmp_path)
    live = store.submit("test", {}, "live")
    dead = store.submit("test", {}, "dead")
    store._update(live["id"], {"state": "running", "owner_pid": os.getpid()})
    store._update(dead["id"], {"state": "running", "owner_pid": 2147483647})
    assert len(store.recover_interrupted()) == 1
    assert store.get(live["id"])["state"] == "running"
    assert store.get(dead["id"])["state"] == "interrupted"
    assert store.retry(dead["id"])["state"] == "queued"


@pytest.mark.parametrize("task,reason", [("error", "supported architecture"), ("escape", "inside the job")])
def test_failure_is_visible_and_partial_artifacts_never_published(manager, task, reason):
    job = manager.submit(task, {}, task)
    done = _wait(manager.store, job["id"], {"failed", "succeeded"})
    assert done["state"] == "failed" and reason in done["error"]
    assert not manager.store.artifact_dir(job["id"]).exists()
    with pytest.raises(ValueError):
        manager.submit("user_supplied_python", {}, "invalid")


def test_shutdown_marks_interrupted_and_queued_cancel(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    manager = JobManager(tmp_path, {"slow": "test_investigation_jobs:slow_task"})
    first = manager.submit("slow", {}, "one")
    second = manager.submit("slow", {}, "two")
    _wait(manager.store, first["id"], {"running"})
    manager.cancel(second["id"])
    manager.close()
    assert manager.store.get(first["id"])["state"] == "interrupted"
    assert manager.store.get(second["id"])["state"] == "cancelled"
