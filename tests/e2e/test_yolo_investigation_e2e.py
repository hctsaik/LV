"""Real browser upload -> real YOLO inspection/inference/CAM -> reusable case evidence.

Random-initialized real architecture validates integration, not production accuracy.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
from PIL import Image, ImageDraw
from playwright.sync_api import expect

from fp_cases import CaseStore
from fp_onnx_demo import make_demo_model
from investigation_jobs import JobStore
from model_profiles import ModelRegistry
from .test_fp_case_workspace_e2e import fp_server, _click, _select as _select_changed
from .conftest import load_app, wait_idle
from .test_investigation_materials_e2e import _new_case

pytestmark=pytest.mark.e2e


@pytest.fixture(scope="module")
def yolo_upload_files(tmp_path_factory):
    root=tmp_path_factory.mktemp("yolo_upload")
    checkpoint=root/"inspection-yolov8.pt"
    existing=os.environ.get("LV_TEST_YOLO_PT")
    if existing and Path(existing).is_file():
        checkpoint.write_bytes(Path(existing).read_bytes())
    else:
        # Keep torch/model memory out of the browser driver's Python process.
        script="""
import sys
import torch
from ultralytics.nn.tasks import DetectionModel
torch.set_num_threads(1)
torch.manual_seed(34)
model=DetectionModel('yolov8n.yaml',ch=3,nc=2,verbose=False).eval()
model.names={0:'scratch',1:'dent'}
torch.save({'model':model},sys.argv[1])
"""
        subprocess.run([sys.executable,"-c",script,str(checkpoint)],check=True,timeout=300,capture_output=True,text=True)
    image=Image.new("RGB",(96,48),(110,60,20))
    ImageDraw.Draw(image).rectangle([20,8,54,29],fill=(220,150,80))
    image.save(root/"report.png")
    ImageDraw.Draw(image).rectangle([60,10,70,20],fill=(190,120,90))
    image.save(root/"second-report.png")
    make_demo_model(root/"deployment.onnx",detection=True)
    return root


def _select(page,label,value):
    control=page.get_by_test_id("stSelectbox").filter(has=page.get_by_text(label,exact=True))
    expect(control).to_be_visible()
    if control.get_by_text(value,exact=True).count():
        return
    _select_changed(page,label,value)


def inspect_uploaded(page,filename):
    _click(page,"檢查模型")
    history=page.get_by_test_id("stSelectbox").filter(has=page.get_by_text("模型檢查紀錄",exact=True))
    expect(history).to_contain_text(filename,timeout=60000)


def uploader(page,label):
    return page.get_by_test_id("stFileUploader").filter(has=page.get_by_text(label,exact=True)).locator('input[type="file"]')


def expand(page,label):
    expander=page.get_by_test_id("stExpander").filter(has=page.locator("summary").filter(has_text=label)).first
    if expander.locator("details").first.get_attribute("open") is None:
        expander.locator("summary").first.click()
    return expander


def first_detection(page):
    control=page.get_by_test_id("stSelectbox").filter(has=page.get_by_text("或從清單選擇預測框",exact=True))
    expect(control).to_be_visible(timeout=300000)
    control.locator('[data-baseweb="select"]').click()
    page.get_by_role("option").nth(1).click()
    _click(page,"選取此預測框")


def save_inspected_model(page,name,*,onnx=False,classes=None,size=None):
    form=page.get_by_test_id("stForm").filter(has=page.get_by_role("textbox",name="模型名稱",exact=True))
    expect(form).to_be_visible(timeout=300000)
    form.get_by_role("textbox",name="模型名稱",exact=True).fill(name)
    if onnx or classes:
        form.get_by_role("textbox",name="模型類別（依模型順序，每行一個）",exact=True).fill(classes or "scratch")
    for label in ("輸入寬度","輸入高度"):
        field=form.get_by_role("spinbutton",name=label,exact=True)
        field.fill(str(size or (32 if onnx else 64)))
        field.press("Tab")
    slider=form.get_by_test_id("stSlider").filter(has=page.get_by_text("判定門檻",exact=True)).get_by_role("slider")
    slider.focus()
    for _ in range(30):
        slider.press("ArrowLeft")
    expect(slider).to_have_attribute("aria-valuenow",re.compile(r"0(?:\.0+)?"))
    form.get_by_text("我已核對類別與現場輸入／判定設定",exact=True).click()
    form.get_by_role("button",name="儲存模型設定",exact=True).click()
    expect(page.get_by_text("模型已儲存，可在下方試跑或在案件中直接選用。",exact=True)).to_be_visible()


def test_upload_trial_cam_download_reuse_and_restart(yolo_upload_files,fp_server,page,tmp_path):
    page.set_default_timeout(30000)
    page.set_viewport_size({"width":1600,"height":1100})
    files=yolo_upload_files
    original_sha=hashlib.sha256((files/"inspection-yolov8.pt").read_bytes()).hexdigest()
    load_app(page,fp_server.base_url)
    _click(page,"誤逮案件")
    _click(page,"我的模型")
    uploader(page,"上傳模型").set_input_files(str(files/"inspection-yolov8.pt"))
    inspect_uploaded(page,"inspection-yolov8.pt")
    expect(page.get_by_text("可推論：是 · 可 Grad-CAM：是",exact=True)).to_be_visible(timeout=300000)
    save_inspected_model(page,"E2E YOLOv8 Station A")
    registry=ModelRegistry(fp_server.case_root)
    profile=next(p for p in registry.list() if p["name"] == "E2E YOLOv8 Station A")
    assert profile["runtime_config"]["input_size"] == [64,64]
    assert profile["runtime_config"]["threshold"] == 0
    uploader(page,"試跑原圖").set_input_files(str(files/"report.png"))
    _click(page,"試跑模型")
    expect(page.get_by_text("試跑完成；結果已保存。這不代表已核對歷史部署判定。",exact=True)).to_be_visible(timeout=300000)
    _click(page,"用此圖建立案件")
    first_detection(page)
    _click(page,"解釋這個框")
    expect(page.get_by_text("模型診斷",exact=True)).to_be_visible()
    _select(page,"熱圖方式","正負貢獻診斷")
    page.get_by_text("同時執行多比例遮擋與隨機對照",exact=True).click()
    _click(page,"分析所選框")
    expect(page.get_by_role("button",name="保存為案件證據",exact=True)).to_be_visible(timeout=300000)
    jobs=JobStore(fp_server.case_root)
    counts_before=[(j["id"],j["task"],j["state"]) for j in jobs.list()]
    opacity=page.get_by_test_id("stSlider").filter(has=page.get_by_text("熱圖透明度",exact=True)).get_by_role("slider")
    old_opacity=opacity.get_attribute("aria-valuenow")
    opacity.press("ArrowRight")
    expect(opacity).not_to_have_attribute("aria-valuenow",old_opacity)
    wait_idle(page,timeout=30000)
    assert [(j["id"],j["task"],j["state"]) for j in jobs.list()] == counts_before,"render-only opacity must not enqueue inference"
    _click(page,"保存為案件證據")
    expect(page.get_by_text("熱圖證據已加入案件，可於圖文回覆引用。",exact=True)).to_be_visible()
    case=next(c for c in CaseStore(fp_server.case_root).list() if c.get("attributions"))
    assert case["attributions"][0]["status"] == "ok"
    assert case["attributions"][0]["target"]["output_index"] == case["items"][0]["external_target"]["output_index"]
    with page.expect_download() as download:
        page.get_by_role("button",name="下載完整熱圖證據包",exact=True).click()
    with zipfile.ZipFile(download.value.path()) as archive:
        assert {"maps.npz","overlay.png","model_input.png","source.png","tensors.npz","case_evidence.json"} <= set(archive.namelist())
        evidence=json.loads(archive.read("case_evidence.json"))
        assert evidence["historical_parity"]["status"] == "unavailable"
    page.get_by_text("模型診斷",exact=True).scroll_into_view_if_needed()
    page.screenshot(path=str(tmp_path/"yolo-case-gradcam.png"),full_page=True)
    _click(page,"返回原案件")
    _click(page,"案件")
    _new_case(page,str(files/"second-report.png"),"Second reusable model case")
    expect(page.get_by_text("案件已建立",exact=False).first).to_be_visible()
    second=next(c for c in CaseStore(fp_server.case_root).list() if c["title"] == "Second reusable model case")
    expect(page.get_by_test_id("stSelectbox").filter(has=page.get_by_text("此圖使用的模型／部署設定",exact=True))).to_contain_text("E2E YOLOv8 Station A")
    inspections_before=len([j for j in jobs.list() if j["task"] == "inspect"])
    _click(page,"推論這張圖片")
    expect(page.get_by_text("或從清單選擇預測框",exact=True)).to_be_visible(timeout=300000)
    second=next(c for c in CaseStore(fp_server.case_root).list() if c["title"] == "Second reusable model case")
    assert second["external_runs"][0]["profile_id"] == profile["id"]
    assert len([j for j in jobs.list() if j["task"] == "inspect"]) == inspections_before
    assert hashlib.sha256((files/"inspection-yolov8.pt").read_bytes()).hexdigest() == original_sha
    fp_server.restart()
    load_app(page,fp_server.base_url)
    _click(page,"誤逮案件")
    assert CaseStore(fp_server.case_root).load(case["id"])["attributions"][0]["compute_id"] == evidence["compute_id"]
    assert not page.get_by_test_id("stException").count()


def test_onnx_upload_keeps_infer_without_fake_gradcam(yolo_upload_files,fp_server,page):
    page.set_default_timeout(30000)
    load_app(page,fp_server.base_url)
    _click(page,"誤逮案件")
    _click(page,"我的模型")
    expand(page,"匯入／補入模型")
    _select(page,"儲存位置","建立新模型")
    uploader(page,"上傳模型").set_input_files(str(yolo_upload_files/"deployment.onnx"))
    inspect_uploaded(page,"deployment.onnx")
    expect(page.get_by_text("可推論：是 · 可 Grad-CAM：尚未可用",exact=True)).to_be_visible(timeout=180000)
    save_inspected_model(page,"E2E ONNX only",onnx=True)
    uploader(page,"試跑原圖").set_input_files(str(yolo_upload_files/"report.png"))
    _click(page,"試跑模型")
    expect(page.get_by_role("button",name="用此圖建立案件",exact=True)).to_be_visible(timeout=180000)
    _click(page,"用此圖建立案件")
    first_detection(page)
    _click(page,"解釋這個框")
    expect(page.get_by_text("Grad-CAM 需要對應的 PyTorch 模型。",exact=False)).to_be_visible()
    expect(page.get_by_role("button",name="進行對照實驗",exact=True)).to_be_visible()
    assert page.get_by_role("button",name="分析所選框",exact=True).count() == 0
    assert not page.get_by_test_id("stException").count()


@pytest.fixture(scope="module")
def paired_onnx(yolo_upload_files):
    path=yolo_upload_files/"paired-yolov8.onnx"
    script="""
import sys
import torch
torch.set_num_threads(1)
model=torch.load(sys.argv[1],map_location='cpu',weights_only=False)['model'].float().eval()
class Raw(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model
    def forward(self,x):
        return self.model(x)[0]
wrapper=Raw(model).eval()
x=torch.zeros(1,3,64,64)
with torch.no_grad():
    wrapper(x)
torch.onnx.export(wrapper,x,sys.argv[2],input_names=['images'],output_names=['predictions'],opset_version=17,dynamo=False)
"""
    subprocess.run([sys.executable,"-c",script,str(yolo_upload_files/"inspection-yolov8.pt"),str(path)],check=True,timeout=300,capture_output=True,text=True)
    return path


def test_pair_existing_onnx_case_and_failed_inspection(paired_onnx,yolo_upload_files,fp_server,page,tmp_path):
    page.set_default_timeout(30000)
    page.set_viewport_size({"width":1600,"height":1100})
    load_app(page,fp_server.base_url)
    _click(page,"誤逮案件")
    _click(page,"我的模型")
    expand(page,"匯入／補入模型")
    _select(page,"儲存位置","建立新模型")
    uploader(page,"上傳模型").set_input_files(str(paired_onnx))
    inspect_uploaded(page,"paired-yolov8.onnx")
    expect(page.get_by_text("可推論：是 · 可 Grad-CAM：尚未可用",exact=True)).to_be_visible(timeout=180000)
    save_inspected_model(page,"E2E paired YOLOv8",onnx=True,classes="scratch\ndent",size=64)
    registry=ModelRegistry(fp_server.case_root)
    profile=next(p for p in registry.list() if p["name"] == "E2E paired YOLOv8")
    uploader(page,"試跑原圖").set_input_files(str(yolo_upload_files/"report.png"))
    _click(page,"試跑模型")
    expect(page.get_by_role("button",name="用此圖建立案件",exact=True)).to_be_visible(timeout=180000)
    _click(page,"用此圖建立案件")
    first_detection(page)
    _click(page,"解釋這個框")
    expect(page.get_by_text("Grad-CAM 需要對應的 PyTorch 模型。",exact=False)).to_be_visible()
    case=next(c for c in CaseStore(fp_server.case_root).list() if c.get("model_profile",{}).get("id") == profile["id"])
    original_run=case["external_runs"][0]["id"]
    _click(page,"補入對應模型")
    expand(page,"匯入／補入模型")
    _select(page,"儲存位置","補入 E2E paired YOLOv8（建立新版本）")
    uploader(page,"上傳模型").set_input_files(str(yolo_upload_files/"inspection-yolov8.pt"))
    inspect_uploaded(page,"inspection-yolov8.pt")
    expect(page.get_by_text("可推論：是 · 可 Grad-CAM：是",exact=True)).to_be_visible(timeout=300000)
    save_inspected_model(page,"E2E paired YOLOv8",classes="scratch\ndent",size=64)
    updated=registry.get(profile["id"])
    assert updated["revision"] == 2 and updated["gradient_asset"] and updated["deployment_asset"]["format"] == "onnx"
    _click(page,"案件")
    _select(page,"開啟案件",case["title"]+" · "+case["id"])
    _select(page,"部署設定版本","v2 · 未填站點")
    _click(page,"推論這張圖片")
    expect(page.get_by_text("已有 2 次外部推論",exact=False)).to_be_visible(timeout=300000)
    first_detection(page)
    _click(page,"解釋這個框")
    _select(page,"熱圖方式","正負貢獻診斷")
    page.get_by_text("同時執行多比例遮擋與隨機對照",exact=True).click()
    _click(page,"分析所選框")
    expect(page.get_by_role("button",name="保存為案件證據",exact=True)).to_be_visible(timeout=300000)
    _click(page,"保存為案件證據")
    saved=CaseStore(fp_server.case_root).load(case["id"])
    attribution=saved["attributions"][-1]
    assert attribution["correspondence"]["status"] == "per_case_matched"
    assert attribution["historical_parity"]["status"] == "unavailable"
    assert saved["external_runs"][0]["id"] == original_run and saved["external_runs"][0]["profile_revision"] == 1
    page.get_by_text("模型診斷",exact=True).scroll_into_view_if_needed()
    page.screenshot(path=str(tmp_path/"paired-onnx-pytorch-gradcam.png"),full_page=True)
    _click(page,"返回原案件")
    _click(page,"我的模型")
    expand(page,"匯入／補入模型")
    _select(page,"儲存位置","建立新模型")
    before=len(registry.list())
    broken=tmp_path/"broken.pt"; broken.write_bytes(b"not a torch checkpoint")
    uploader(page,"上傳模型").set_input_files(str(broken))
    inspect_uploaded(page,"broken.pt")
    expect(page.get_by_text("模型載入失敗",exact=False).first).to_be_visible(timeout=180000)
    assert len(registry.list()) == before
    assert CaseStore(fp_server.case_root).load(case["id"])["attributions"][-1]["compute_id"] == attribution["compute_id"]
    assert not page.get_by_test_id("stException").count()
