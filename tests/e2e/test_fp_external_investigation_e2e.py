"""Actual ONNX + browser: replay, experiment, reviewed export and paired evaluation."""
import json
import zipfile

import pytest
from PIL import Image
from playwright.sync_api import expect

from fp_cases import CaseStore, report_markdown
from fp_onnx_demo import make_demo_model
from .conftest import load_app, wait_idle
from .test_fp_case_workspace_e2e import fp_server, _click, _select

pytestmark = pytest.mark.e2e


def _form_select(page, form, label, option):
    form.get_by_test_id("stSelectbox").filter(has=page.get_by_text(label, exact=True)).locator('[data-baseweb="select"]').click()
    page.get_by_role("option", name=option, exact=True).click()


def _configure(page, path, version, index=0):
    title = "目前／基準模型" if index == 0 else "改善後／候選模型（可稍後設定）"
    panel = page.get_by_test_id("stExpander").filter(has=page.get_by_text(title, exact=True))
    form = panel.get_by_test_id("stForm")
    form.get_by_role("textbox", name="ONNX 模型路徑", exact=True).fill(str(path))
    form.get_by_role("textbox", name="部署／模型版本名稱", exact=True).fill(version)
    _form_select(page, form, "模型輸出格式", "分類 [1,C]")
    form.get_by_role("textbox", name="類別順序（逗號分隔，須與模型輸出一致）", exact=True).fill("good,crack")
    form.get_by_role("textbox", name="正常類別（分類模型必填，逗號分隔）", exact=True).fill("good")
    form.get_by_role("spinbutton", name="輸入寬度", exact=True).fill("32")
    form.get_by_role("spinbutton", name="輸入高度", exact=True).fill("32")
    _form_select(page, form, "縮放方式", "縮放至指定寬高")
    form.get_by_role("button", name="儲存模型設定", exact=True).click()
    wait_idle(page, timeout=30000)
    label = "基準模型" if index == 0 else "候選模型"
    expect(page.get_by_text(f"{label}設定已儲存；尚未執行推論", exact=True)).to_be_visible()


def test_external_replay_experiment_training_and_eval(fp_server, page, tmp_path):
    page.set_default_timeout(30000)
    page.set_viewport_size({"width": 1600, "height": 1100})
    baseline = make_demo_model(tmp_path / "old.onnx", bias=.3)
    candidate = make_demo_model(tmp_path / "new.onnx")
    image = tmp_path / "reported.png"
    Image.new("RGB", (32, 32), (220,)*3).save(image)
    before_bytes = image.read_bytes()
    load_app(page, fp_server.base_url)
    _click(page, "誤逮案件")
    page.get_by_role("textbox", name="或填寫本機圖片路徑", exact=True).fill(str(image))
    page.get_by_role("textbox", name="案件名稱（可留空）", exact=True).fill("外部模型查案 E2E")
    _click(page, "建立案件")
    page.get_by_role("textbox", name="模型預測類別／爭議名稱", exact=True).fill("crack")
    _click(page, "加入爭議區域")
    _select(page, "覆核結果", "確認此區域正常")
    page.get_by_role("textbox", name="覆核者", exact=True).fill("Engineer")
    page.get_by_role("textbox", name="判斷依據", exact=True).fill("整圖依規格確認正常")
    _click(page, "儲存此區域覆核")
    _click(page, "外部模型")
    page.get_by_text("顯示舊版模型設定",exact=True).click()
    _configure(page, baseline, "baseline")
    _configure(page, candidate, "candidate", index=1)
    _click(page, "執行與歷史比對")
    page.get_by_text("進階：舊版歷史紀錄比對",exact=True).click()
    _click(page, "執行外部模型並比對")
    expect(page.get_by_text("外部推論與比對證據已儲存", exact=True)).to_be_visible()
    store = CaseStore(fp_server.case_root)
    case = store.list()[0]
    first = case["external_runs"][0]
    assert first["parity"]["status"] == "unavailable"
    # This fixture simulates a signed-off deployment record for the controlled ONNX model.
    historical = {**first, "origin": "deployment", "recorded_at": "2026-09-01T00:00:00Z"}
    page.get_by_role("textbox", name="或貼上當時部署輸出 JSON", exact=True).fill(json.dumps(historical, ensure_ascii=False))
    _click(page, "執行外部模型並比對")
    expect(page.get_by_text("當時完整輸出已重現", exact=True)).to_be_visible()
    _click(page, "準備完整推論證據包")
    with page.expect_download() as downloaded:
        page.get_by_role("button", name="下載推論證據 ZIP", exact=True).click()
    with zipfile.ZipFile(downloaded.value.path()) as archive:
        assert {"tensors.npz", "model_input.png", "source.png", "evidence.json"} <= set(archive.namelist())
    _click(page, "對照實驗")
    _select(page, "固定觀察的模型類別", "crack")
    page.get_by_role("spinbutton", name="亮度倍數", exact=True).fill("0.5")
    page.get_by_text("此修改範圍與方式保留了待判斷目標的有效外觀（仍須檢查結果圖）", exact=True).click()
    _click(page, "執行對照實驗")
    expect(page.get_by_role("button",name="將完成結果加入案件",exact=True)).to_be_visible(timeout=180000)
    _click(page,"將完成結果加入案件")
    case = store.list()[0]
    assert case["experiments"][0]["variants"][0]["score_drop"] > 0
    assert case["cause_status"] == "unknown"
    _click(page, "訓練交接")
    page.get_by_test_id("stMultiSelect").filter(has=page.get_by_text("已確認整張標註完整的影像", exact=True)).locator('[data-baseweb="select"]').click()
    page.get_by_role("option").first.click()
    page.keyboard.press("Escape")
    page.get_by_role("textbox", name="完整標註確認者", exact=True).fill("Engineer")
    page.get_by_role("textbox", name="完整檢查與類別確認依據", exact=True).fill("全部區域依規格確認正常")
    page.get_by_text("我已檢查選取影像的全部內容，真瑕疵框無遺漏，且影像是可用的原始訓練內容", exact=True).click()
    _click(page, "建立訓練交接 ZIP")
    with page.expect_download() as training_download:
        page.get_by_role("button", name="下載訓練交接 ZIP", exact=True).click()
    with zipfile.ZipFile(training_download.value.path()) as archive:
        provenance = json.loads(archive.read("provenance.json"))
        assert provenance["n_negative_images"] == 1
    normal, defect = tmp_path / "holdout_normal.png", tmp_path / "holdout_defect.png"
    Image.new("RGB", (32, 32), (80,)*3).save(normal)
    Image.new("RGB", (32, 32), (230,)*3).save(defect)
    manifest = {"schema_version": 1, "purpose": "independent", "independent_declared": True,
                "split_basis": "held-out fixture", "items": [
                    {"id": "n", "path": str(normal), "truth": "normal", "verified": True},
                    {"id": "d", "path": str(defect), "truth": "defect", "verified": True}]}
    _click(page, "改善評估")
    page.get_by_text("進階 JSON 清單",exact=True).click()
    page.get_by_role("textbox", name="或貼上評估清單 JSON", exact=True).fill(json.dumps(manifest))
    page.get_by_role("spinbutton", name="正常／瑕疵各自最低支持數", exact=True).fill("1")
    _click(page, "執行前後模型評估")
    expect(page.get_by_role("button",name="將完成結果加入案件",exact=True)).to_be_visible(timeout=180000)
    _click(page,"將完成結果加入案件")
    expect(page.get_by_text("符合本次固定資料集驗收門檻", exact=True)).to_be_visible()
    with page.expect_download() as eval_download:
        page.get_by_role("button", name="下載前後評估 JSON", exact=True).click()
    result = json.loads(eval_download.value.path().read_text(encoding="utf-8"))
    assert result["gate"]["status"] == "passed"
    assert result["baseline"]["false_alarms"] == 1 and result["candidate"]["false_alarms"] == 0
    page.screenshot(path=str(tmp_path / "external_evaluation.png"), full_page=True)
    case = store.list()[0]
    report = report_markdown(case)
    assert "外部模型尚未重播。" not in report and "固定資料集改善評估" in report
    assert image.read_bytes() == before_bytes
    assert not page.get_by_test_id("stException").count()
    fp_server.restart()
    load_app(page, fp_server.base_url)
    _click(page, "誤逮案件")
    _click(page, "改善評估")
    expect(page.get_by_text("符合本次固定資料集驗收門檻", exact=True)).to_be_visible()
