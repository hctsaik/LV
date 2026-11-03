"""Seeded lifecycle GUI checks; no claim of production-model acceptance.

The queued fixture deliberately uses an unregistered task so no computation
starts before the user cancels it. Retry uses the actual inspect handler with
a missing asset, exercising a reproducible failure without loading a model.
"""
import io
import re
import time

import pytest
from PIL import Image
from playwright.sync_api import expect

from fp_cases import CaseStore
from investigation_jobs import JobStore
from model_profiles import ModelRegistry
from .conftest import load_app, wait_idle
from .test_fp_case_workspace_e2e import fp_server, _click

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def capture_final_lifecycle_state(page, tmp_path):
    yield
    page.screenshot(path=str(tmp_path / "final-state.png"), full_page=True)
    (tmp_path / "final-state.txt").write_text(page.inner_text("body"), encoding="utf-8")
    (tmp_path / "final-state.html").write_text(page.content(), encoding="utf-8")


def _area(page, label):
    _click(page, label)


def test_seeded_queued_cancel_and_failed_inspect_retry(fp_server, page, tmp_path):
    jobs = JobStore(fp_server.case_root)
    registry = ModelRegistry(fp_server.case_root)
    missing = registry.ingest(b"controlled missing weights", "missing.onnx")
    registry.asset_path(missing["sha256"]).unlink()
    failed = jobs.submit("inspect", {"asset_sha": missing["sha256"], "asset_suffix": ".onnx",
                                   "workspace_root": str(fp_server.case_root)}, "seeded-missing-inspect")
    jobs._update(failed["id"], {"state": "failed", "error": "Seeded missing model asset"})
    queued = jobs.submit("seeded_queue_only", {}, "seeded-queued-ui")
    load_app(page, fp_server.base_url)
    _click(page, "誤逮案件")
    _area(page, "工作紀錄")
    expect(page.get_by_text("等待執行", exact=False)).to_be_visible()
    _click(page, "取消工作")
    expect(page.get_by_text("已取消", exact=False)).to_be_visible()
    assert jobs.get(queued["id"])["state"] == "cancelled"
    selector = page.locator(".st-key-inv_job_selected [data-baseweb='select']")
    selector.click()
    page.get_by_role("option", name=re.compile("模型檢查.*failed")).click()
    expect(page.get_by_text("Seeded missing model asset", exact=True)).to_be_visible()
    _click(page, "重試這項工作")
    deadline = time.monotonic() + 30
    retries = []
    while time.monotonic() < deadline:
        retries = [job for job in jobs.list() if job["retry_of"] == failed["id"]]
        if retries:
            break
        page.wait_for_timeout(100)
    assert len(retries) == 1
    assert jobs.get(failed["id"])["state"] == "failed"
    expect(page.locator(".st-key-inv_job_" + retries[0]["id"] + "_retry")).to_be_visible(timeout=60000)
    page.screenshot(path=str(tmp_path / "job-retry.png"), full_page=True)


def test_seeded_model_archive_restore_preserves_revision_history(fp_server, page, tmp_path):
    registry = ModelRegistry(fp_server.case_root)
    asset = registry.ingest(b"seeded-ui-model-not-executable", "history.onnx")
    first = registry.create("歷史模型 GUI", deployment_asset=asset,
                            runtime_config={"model_version": "original", "threshold": .1},
                            metadata={"station_name": "Station A"})
    second = registry.revise(first["id"], runtime_config={"model_version": "revised", "threshold": .5},
                             metadata={"station_name": "Station B"})
    store = CaseStore(fp_server.case_root)
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "gray").save(buffer, format="PNG")
    source = store.ingest(buffer.getvalue(), "history.png")
    case = store.create(source, "Historical model profile fixture")
    case["external_runs"] = [{"id": "seeded-history", "source_id": source["id"],
                              "profile_snapshot": first, "prediction": {"detections": []}}]
    case = store.save(case)
    load_app(page, fp_server.base_url)
    _click(page, "誤逮案件")
    _area(page, "我的模型")
    revision = page.locator(".st-key-model_revision_" + first["id"] + " [data-baseweb='select']")
    settings = page.get_by_test_id("stExpander").filter(has=page.get_by_text("修改現場設定（建立新版本）", exact=True))
    settings.locator("summary").first.click()
    for number, station, threshold in ((1, "Station A", .1), (2, "Station B", .5)):
        revision.click()
        page.get_by_role("option", name=str(number), exact=True).click()
        if settings.locator("details").first.get_attribute("open") is None:
            settings.locator("summary").first.click()
        expect(settings.get_by_role("textbox", name="產線／站點名稱", exact=True)).to_have_value(station)
        expect(settings.locator(".st-key-model_edit_" + first["id"] + str(number) + "_threshold").get_by_role("slider")).to_have_attribute("aria-valuenow", str(threshold))
    expander = page.get_by_test_id("stExpander").filter(has=page.get_by_text("管理模型", exact=True))
    expander.locator("summary").click()
    _click(page, "封存模型")
    expect(page.get_by_text("尚無可選模型", exact=False)).to_be_visible()
    page.get_by_text("顯示已封存模型", exact=True).click()
    wait_idle(page)
    expect(page.get_by_text("歷史模型 GUI · 已封存", exact=True)).to_be_visible()
    revision.click()
    page.get_by_role("option", name="1", exact=True).click()
    expect(page.get_by_text("版本 1 · original", exact=True)).to_be_visible()
    if expander.locator("details").get_attribute("open") is None:
        expander.locator("summary").click()
    _click(page, "恢復模型")
    expect(page.get_by_role("button", name="封存模型", exact=True)).to_be_visible()
    assert registry.get(first["id"], 1)["runtime_config"] == first["runtime_config"]
    assert registry.get(first["id"], 2)["runtime_config"] == second["runtime_config"]
    assert not registry.get(first["id"])["archived"]
    assert store.load(case["id"])["external_runs"][0]["profile_snapshot"] == first
    assert registry.get(first["id"], 2)["deployment_asset"]["sha256"] == first["deployment_asset"]["sha256"]
    page.screenshot(path=str(tmp_path / "model-history-restored.png"), full_page=True)
