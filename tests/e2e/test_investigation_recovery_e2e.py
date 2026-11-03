"""Seeded case UI regressions; these do not claim production-model acceptance."""
import io
import hashlib
import json

import pytest
from PIL import Image
from playwright.sync_api import expect

import fp_cases as cases
from .conftest import load_app, wait_idle
from .test_fp_case_workspace_e2e import fp_server, _click

pytestmark = pytest.mark.e2e


def _image(color):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _expand(page, title):
    details = page.get_by_test_id("stExpander").filter(has=page.get_by_text(title, exact=True))
    if details.locator("details").first.get_attribute("open") is None:
        details.locator("summary").first.click()
    return details


def test_missing_source_repair_rejects_different_image(fp_server, page, tmp_path):
    store = cases.CaseStore(fp_server.case_root)
    original = _image("orange")
    source = store.ingest(original, "recover.png")
    case = store.create(source, "缺圖修復 UI")
    cases.add_item(case, source["id"], [.1, .1, .7, .8], "scratch")
    case = store.save(case)
    store.image_path(source).unlink()
    load_app(page, fp_server.base_url + "?inv_case=" + case["id"])
    expect(page.get_by_text("這張案件圖片遺失或內容已改變。", exact=False)).to_be_visible(timeout=60000)
    uploader = page.get_by_test_id("stFileUploader").filter(has=page.get_by_text("補回原始檔案", exact=True))
    uploader.locator('input[type="file"]').set_input_files({"name": "wrong.png", "mimeType": "image/png", "buffer": _image("blue")})
    expect(uploader.get_by_text("wrong.png", exact=True)).to_be_visible()
    _click(page, "驗證並修復案件圖片")
    expect(page.get_by_text("補回圖片必須與案件原始檔案完全一致", exact=False)).to_be_visible()
    assert store.load(case["id"]) == case
    uploader.locator('input[type="file"]').set_input_files({"name": "recover.png", "mimeType": "image/png", "buffer": original})
    expect(uploader.get_by_text("recover.png", exact=True)).to_be_visible()
    _click(page, "驗證並修復案件圖片")
    expect(page.get_by_text("原始圖片已驗證並修復", exact=False)).to_be_visible()
    assert store.image_path(source).read_bytes() == original
    assert store.load(case["id"]) == case
    page.screenshot(path=str(tmp_path / "source-repaired.png"), full_page=True)


def test_region_correction_withdraw_and_restore_preserves_other_verdict(fp_server, page, tmp_path):
    store = cases.CaseStore(fp_server.case_root)
    source = store.ingest(_image("green"), "two-regions.png")
    case = store.create(source, "逐區域修正 UI")
    cases.add_item(case, source["id"], [.05, .1, .4, .8], "first")
    cases.add_item(case, source["id"], [.6, .1, .95, .8], "second")
    first, second = case["items"]
    cases.review_item(case, first["id"], "normal", "Engineer", "反光")
    cases.review_item(case, second["id"], "defect", "Engineer", "確認裂痕")
    case = store.save(case)
    other_review = case["items"][1]["review"]
    load_app(page, fp_server.base_url + "?inv_case=" + case["id"] + "&inv_item=" + first["id"])
    expect(page.get_by_text("修正／撤回這個區域", exact=True)).to_be_visible(timeout=60000)
    _expand(page, "修正／撤回這個區域")
    page.get_by_role("textbox", name="修正類別／爭議名稱", exact=True).fill("reflection")
    page.get_by_role("textbox", name="修正／撤回原因", exact=True).fill("修正第一次選錯的類別")
    _click(page, "儲存區域修正")
    changed = store.load(case["id"])
    assert changed["items"][0]["label"] == "reflection"
    assert changed["items"][0]["review"]["verdict"] == "pending"
    assert changed["items"][1]["review"] == other_review
    _expand(page, "修正／撤回這個區域")
    page.get_by_role("textbox", name="修正／撤回原因", exact=True).fill("撤回重複爭議")
    _click(page, "撤回區域")
    expect(page.get_by_text("已撤回區域與候選", exact=True)).to_be_visible()
    assert store.load(case["id"])["items"][0]["withdrawn"]
    _expand(page, "已撤回區域與候選")
    page.get_by_role("textbox", name="恢復原因", exact=True).fill("補充資料後重新覆核")
    _click(page, "恢復並重新覆核")
    restored = store.load(case["id"])
    assert not restored["items"][0]["withdrawn"]
    assert restored["items"][0]["review"]["verdict"] == "pending"
    assert restored["items"][1]["review"] == other_review
    assert len(restored["region_history"]) == 3
    _expand(page, "調查工時：已記錄 0.0 分鐘")
    page.get_by_role("spinbutton", name="首次設定分鐘", exact=True).fill("4")
    page.get_by_role("spinbutton", name="重複輸入次數", exact=True).fill("2")
    page.get_by_role("spinbutton", name="選錯目標重作次數", exact=True).fill("1")
    page.get_by_role("spinbutton", name="證據不足退回次數", exact=True).fill("1")
    _click(page, "記錄本次固定案例測量")
    measurement = store.load(case["id"])["effort_measurements"][-1]
    assert measurement["setup_minutes"] == 4
    assert measurement["repeated_inputs"] == 2
    assert measurement["wrong_target_rework"] == 1
    assert measurement["evidence_returns"] == 1
    page.screenshot(path=str(tmp_path / "region-restored.png"), full_page=True)


def test_zero_detection_preserves_reported_region_without_cam_target(fp_server, page):
    """Controlled stored run checks display, not a claim of real inference."""
    store = cases.CaseStore(fp_server.case_root)
    source = store.ingest(_image("purple"), "no-detection.png")
    case = store.create(source, "本次零框 UI", raw_output="現場報告一個 scratch 框")
    cases.add_item(case, source["id"], [.1, .1, .8, .8], "reported scratch")
    case["external_runs"] = [{"id": "controlled-empty-run", "source_id": source["id"],
                              "prediction": {"detections": []}}]
    case = store.save(case)
    load_app(page, fp_server.base_url + "?inv_case=" + case["id"])
    expect(page.get_by_text("本次模型完整輸出沒有偵測框。", exact=False)).to_be_visible(timeout=60000)
    expect(page.get_by_role("button", name="解釋這個框", exact=True)).to_have_count(0)
    current = store.load(case["id"])
    assert len(current["items"]) == 1
    assert "external_target" not in current["items"][0]
    assert current["items"][0]["review"]["verdict"] == "pending"
    _click(page, "準備圖文回覆")
    expect(page.get_by_role("button", name="下載含圖片回覆 HTML", exact=True)).to_be_visible()


def test_zero_gradient_evidence_is_visible_without_recomputation(fp_server, page, tmp_path):
    """Seed zero-valued artifacts to verify the negative-result UI only."""
    import numpy as np
    from fp_attribution import target_identity
    from investigation_jobs import JobStore
    store = cases.CaseStore(fp_server.case_root)
    image = _image("gray")
    source = store.ingest(image, "zero-gradient.png")
    case = store.create(source, "無有效熱圖 UI")
    item = cases.add_item(case, source["id"], [.1, .1, .8, .8], "scratch")
    item["external_target"] = {"run_id": "controlled-run", "output_index": 0, "class": "scratch"}
    case["external_runs"] = [{"id": "controlled-run", "source_id": source["id"],
        "profile_snapshot": {"id": "controlled-profile", "name": "Controlled display fixture", "revision": 1,
                             "gradient_asset": {"format": "pt"}},
        "prediction": {"detections": [{"output_index": 0, "class": "scratch", "score": .5, "box": item["roi"]}]}}]
    case = store.save(case)
    identity = target_identity(case, item)
    jobs = JobStore(store.root)
    payload = {"case_id": case["id"], "item_id": item["id"], "target_identity": identity,
               "method": "positive", "scalar": "confidence", "fractions": [.1, .25, .5], "controls": 3, "randomization": False}
    job = jobs.submit("gradcam", payload, "controlled-zero-ui", context={"case_id": case["id"]})
    jobs._update(job["id"], {"state": "interrupted"})
    directory = jobs.artifact_dir(job["id"])
    directory.mkdir(parents=True)
    for name in ["source.png", "model_input.png", "overlay.png"]:
        (directory / name).write_bytes(image)
    np.savez(directory / "maps.npz", source=np.zeros((48, 64), dtype=np.float32))
    result = {"attribution": {"id": "controlled-zero-cam", "target_identity": identity, "method": "positive",
              "status": "zero_gradient", "artifact_dir": "@artifact/.",
              "legend": "此受控顯示案例沒有有效梯度", "limitation": "不能據此確認誤逮原因",
              "faithfulness": {"status": "unavailable", "reason": "沒有有效歸因可供遮擋對照"}}}
    (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    artifacts = [{"path": p.name, "size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in directory.iterdir()]
    jobs._update(job["id"], {"state": "succeeded", "artifacts": artifacts})
    load_app(page, fp_server.base_url + "?diagnostic_case=" + case["id"] + "&diagnostic_item=" + item["id"])
    expect(page.get_by_text("目標梯度為零", exact=True)).to_be_visible(timeout=60000)
    expect(page.get_by_text("沒有有效歸因可供遮擋對照", exact=True)).to_be_visible()
    expect(page.get_by_text("不能據此確認誤逮原因", exact=True)).to_be_visible()
    expect(page.get_by_text("此頁解釋框分數；現場機台規則與根因需另外核對。", exact=True)).to_be_visible()
    expect(page.get_by_text("歷史部署重播：unavailable", exact=True)).to_be_visible()
    assert len(jobs.list(case_id=case["id"])) == 1
    assert not np.any(np.load(directory / "maps.npz")["source"])
    page.screenshot(path=str(tmp_path / "zero-gradient-state.png"), full_page=True)
