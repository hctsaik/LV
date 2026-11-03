"""Terminate a small worker with a deliberately tiny budget, without stressing the host."""
import time
from pathlib import Path

import pytest

from investigation_jobs import JobManager, _pid_alive


def waiting_task(payload, context):
    context.write_artifact("partial.txt", b"incomplete")
    time.sleep(30)
    return {"unexpected": True}


@pytest.mark.parametrize("options,reason", [
    ({"memory_limit_mb": 1}, "memory limit"),
    ({"timeout_seconds": .2}, "time limit"),
])
def test_resource_limit_stops_worker_without_publishing(tmp_path, monkeypatch, options, reason):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    manager = JobManager(tmp_path, {"wait": "test_investigation_resource_limits:waiting_task"}, **options)
    try:
        job = manager.submit("wait", {}, "bounded-resource-check")
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            current = manager.store.get(job["id"])
            if current["state"] == "failed":
                break
            time.sleep(.05)
        assert current["state"] == "failed", current
        assert reason in current["error"]
        assert current["worker_pid"] and not _pid_alive(current["worker_pid"])
        assert not manager.store.artifact_dir(job["id"]).exists()
        with pytest.raises(ValueError, match="no completed"):
            manager.store.result(job["id"])
    finally:
        manager.close()
