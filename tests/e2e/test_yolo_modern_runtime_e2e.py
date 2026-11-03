"""Real modern upload/head selection/inference/CAM; random weights are integration fixtures."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw
from playwright.sync_api import expect
import pytest

from fp_cases import CaseStore
from model_profiles import ModelRegistry
from model_runtimes import MODERN, environment, interpreter
from .conftest import load_app
from .test_fp_case_workspace_e2e import fp_server, _click
from .test_yolo_investigation_e2e import _select, uploader, expand, inspect_uploaded, save_inspected_model, first_detection

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def capture_browser_state(page, tmp_path):
    yield
    if not page.is_closed():
        page.screenshot(path=str(tmp_path / "browser-state.png"), full_page=True, timeout=10000)
        (tmp_path / "browser-state.txt").write_text(page.locator("body").inner_text(), encoding="utf-8")
        (tmp_path / "browser-state.html").write_text(page.content(), encoding="utf-8")


@pytest.fixture(scope="module")
def modern_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("modern_upload")
    path = root / "inspection-yolo26.pt"
    validated = os.environ.get("LV_TEST_MODERN_BROWSER_PT")
    script = """
import sys,torch
from ultralytics.nn.tasks import DetectionModel
torch.set_num_threads(1)
torch.manual_seed(34)
model=DetectionModel('yolo26n.yaml',ch=3,nc=2,verbose=False).eval()
model.names={0:'scratch',1:'dent'}
torch.save({'model':model},sys.argv[1])
"""
    if validated and Path(validated).is_file():
        path.write_bytes(Path(validated).read_bytes())
    else:
        subprocess.run([str(interpreter(MODERN)), "-c", script, str(path)], env=environment(MODERN),
                       check=True, capture_output=True, text=True, timeout=300)
    image = Image.new("RGB", (96, 48), (110, 60, 20))
    ImageDraw.Draw(image).rectangle([20, 8, 54, 29], fill=(220, 150, 80))
    image.save(root / "report.png")
    return root


@pytest.mark.parametrize("head,label", [("one_to_many", "One-to-many＋NMS"), ("one_to_one", "One-to-one（NMS-free）")])
def test_modern_upload_explicit_head_cam_and_legacy_navigation(modern_files, fp_server, page, tmp_path, head, label):
    page.set_default_timeout(30000)
    page.set_viewport_size({"width": 1600, "height": 1100})
    checkpoint = modern_files / "inspection-yolo26.pt"
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    load_app(page, fp_server.base_url)
    _click(page, "誤逮案件")
    _click(page, "我的模型")
    expand(page, "匯入／補入模型")
    _select(page, "儲存位置", "建立新模型")
    _select(page, "模型執行版本", "新版 YOLO12／26／v8／11 · 8.4.142")
    _select(page, "推論 head", label)
    uploader(page, "上傳模型").set_input_files(str(checkpoint))
    inspect_uploaded(page, checkpoint.name)
    # The previous inspection can share the same filename and capability text.
    # Wait for this head's completed result, not a transient historical caption.
    expect(page.get_by_text(f"實際執行版本：8.4.142 · head：{head}", exact=False)).to_be_visible(timeout=300000)
    expect(page.get_by_text("可推論：是 · 可 Grad-CAM：是", exact=True)).to_be_visible()
    name = "E2E YOLO26 " + head
    save_inspected_model(page, name)
    profile = next(p for p in ModelRegistry(fp_server.case_root).list() if p["name"] == name)
    assert profile["runtime_config"]["runtime_id"] == MODERN
    assert profile["runtime_config"]["head_path"] == head
    uploader(page, "試跑原圖").set_input_files(str(modern_files / "report.png"))
    _click(page, "試跑模型")
    expect(page.get_by_text("試跑完成；結果已保存。這不代表已核對歷史部署判定。", exact=True)).to_be_visible(timeout=300000)
    _click(page, "用此圖建立案件")
    first_detection(page)
    _click(page, "解釋這個框")
    _select(page, "熱圖方式", "正負貢獻診斷")
    # Numerical perturbation protocols have separate gates; this story verifies
    # the ordinary model/head/CAM path without multiplying browser CPU work.
    page.get_by_text("同時執行多比例遮擋與隨機對照", exact=True).click()
    _click(page, "分析所選框")
    expect(page.get_by_role("button", name="保存為案件證據", exact=True)).to_be_visible(timeout=300000)
    _click(page, "保存為案件證據")
    expect(page.get_by_text("熱圖證據已加入案件，可於圖文回覆引用。", exact=True)).to_be_visible()
    cases = CaseStore(fp_server.case_root).list()
    case = next(c for c in cases if any(r.get("profile_id") == profile["id"] for r in c.get("external_runs", [])))
    report = case["attributions"][-1]
    assert report["status"] == "ok"
    assert report["target"]["execution"]["runtime_id"] == MODERN
    assert report["target"]["execution"]["head_path"] == head
    assert isinstance(report["target"]["native_candidate_index"], int)
    assert report["forward_parity"]["status"] == "matched"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checksum
    (tmp_path / "validation.json").write_text(json.dumps({
        "fixture_scope": "Artificial numerical checkpoint; not trained-model efficacy",
        "fixture_sha256": checksum, "source_unchanged": True,
        "runtime_config": profile["runtime_config"], "target": report["target"],
        "status": report["status"], "forward_parity": report["forward_parity"],
        "source_sha256": report["source_sha256"], "compute_id": report["compute_id"]}, indent=2), encoding="utf-8")
    page.screenshot(path=str(tmp_path / (head + "-modern-gradcam.png")), full_page=True)
    _click(page, "返回原案件")
    assert not page.get_by_test_id("stException").count()
