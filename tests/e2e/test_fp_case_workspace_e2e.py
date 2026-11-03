"""Real app + real local DINO: intake to cited report, review ZIP and return."""
from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile

import pytest
from PIL import Image, ImageDraw
from playwright.sync_api import expect

from fp_cases import CaseStore
from .conftest import _ManagedServer, load_app, wait_idle

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def fp_server(tmp_path_factory):
    root = tmp_path_factory.mktemp("fp_browser")
    paths = {name: root / name for name in ["cache", "output", "cim", "cases"]}
    for p in paths.values():
        p.mkdir()
    env = {**os.environ, "LV_CASES_DIR": str(paths["cases"]), "LV_OUTPUT_DIR": str(paths["output"]),
           "LV_CACHE_DIR": str(paths["cache"]), "CIM_LOG_DIR": str(paths["cim"]),
           "LV_UI_STATE": str(root / "missing" / "state.json"),
           "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false", "STREAMLIT_SERVER_HEADLESS": "true"}
    server = _ManagedServer(env, paths["cache"], paths["output"], paths["cim"])
    server.case_root = paths["cases"]
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _click(page, name):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        buttons = page.get_by_role("button", name=name, exact=True)
        clicked = False
        for index in range(buttons.count() - 1, -1, -1):
            button = buttons.nth(index)
            try:
                if (button.is_visible() and button.is_enabled()
                        and button.evaluate("el => !el.closest('[data-stale=\"true\"]')")):
                    button.click(timeout=1000)
                    clicked = True
                    break
            except Exception:
                continue  # The previous fragment may detach before the click.
        if clicked:
            break
        page.wait_for_timeout(100)
    else:
        raise AssertionError(f"No current actionable button: {name}")
    # Give the submitted event a chance to enter its rerun before checking
    # idle; individual tests still assert the action's actual saved result.
    page.wait_for_timeout(250)
    wait_idle(page, timeout=180000)
    page.wait_for_function("""() => !Array.from(document.querySelectorAll('[data-stale="true"]'))
        .some(el => el.offsetParent !== null)""", timeout=30000)


def _select(page, label, choice):
    page.get_by_test_id("stSelectbox").filter(has=page.get_by_text(label, exact=True)).locator('[data-baseweb="select"]').click()
    page.get_by_role("option", name=choice, exact=True).click()


def _snapshot(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file()}


def test_real_case_to_comparison_report_and_review_return(fp_server, page, tmp_path):
    page.set_default_timeout(30000)
    page.set_viewport_size({"width": 1600, "height": 1100})
    references = tmp_path / "references"
    normal, defect = references / "normal", references / "defect"
    normal.mkdir(parents=True); defect.mkdir()
    query = references / "reported.png"
    for path, line_color, offset in [(query, "white", 20), (normal / "normal.png", "white", 24),
                                     (defect / "crack.png", "black", 30)]:
        image = Image.new("RGB", (112, 112), (130, 140, 145))
        draw = ImageDraw.Draw(image)
        draw.ellipse([10, 10, 100, 100], fill=(180, 185, 190))
        draw.line([(offset, 20), (55, 50), (75, 90)], fill=line_color, width=3)
        image.save(path)
    before = _snapshot(references)
    load_app(page, fp_server.base_url)
    _click(page, "誤逮案件")
    expect(page.get_by_text("誤逮案件診斷", exact=True)).to_be_visible()
    page.get_by_role("textbox", name="或填寫本機圖片路徑", exact=True).fill(str(query))
    page.get_by_role("textbox", name="案件名稱（可留空）", exact=True).fill("反光誤逮 E2E")
    _select(page, "圖片來源", "截圖")
    page.get_by_role("textbox", name="使用者描述／認為正常的理由", exact=True).fill("外觀為正常反光，請協助對照")
    _click(page, "建立案件")
    expect(page.get_by_text("依截圖比對", exact=False)).to_be_visible()
    assert not list(fp_server.case_root.rglob("*.npy")), "intake must not run inference"
    page.get_by_role("textbox", name="模型預測類別／爭議名稱", exact=True).fill("crack")
    page.get_by_role("textbox", name="預測分數（未知可留空）", exact=True).fill("0.98")
    _click(page, "加入爭議區域")
    _select(page, "覆核結果", "確認此區域正常")
    page.get_by_role("textbox", name="覆核者", exact=True).fill("Engineer")
    page.get_by_role("textbox", name="判斷依據", exact=True).fill("依規格反光為可接受外觀")
    _click(page, "儲存此區域覆核")
    _click(page, "找對照")
    page.get_by_role("textbox", name="正常對照資料夾（可留空）", exact=True).fill(str(normal))
    page.get_by_role("textbox", name="真瑕疵對照資料夾（可留空）", exact=True).fill(str(defect))
    _click(page, "找對照案例")
    expect(page.get_by_text("對照結果與來源已儲存", exact=True)).to_be_visible(timeout=180000)
    expect(page.get_by_text("正常對照", exact=True)).to_be_visible()
    expect(page.get_by_text("真瑕疵對照", exact=True)).to_be_visible()
    expect(page.get_by_text("本次搜尋範圍內沒有可用對照。", exact=True)).to_be_visible()
    case = CaseStore(fp_server.case_root).list()[0]
    assert case["searches"][0]["recipe"]["model"] == "dinov2_vits14"
    assert len(case["searches"][0]["recipe"]["checkpoint_sha256"]) == 64
    page.get_by_role("button", name="加入待覆核", exact=True).first.click()
    wait_idle(page, timeout=30000)
    page.screenshot(path=str(tmp_path / "comparison.png"), full_page=True)
    _click(page, "說明與回覆")
    _select(page, "說明類型", "待驗證假設")
    page.get_by_role("textbox", name="說明內容", exact=True).fill("可能有反光與裂痕的外觀混淆，正式模型原因尚未確認。")
    page.get_by_test_id("stMultiSelect").filter(has=page.get_by_text("引用證據", exact=True)).locator('[data-baseweb="select"]').click()
    page.get_by_role("option").first.click()
    page.keyboard.press("Escape")
    _click(page, "儲存說明")
    with page.expect_download() as report_download:
        page.get_by_role("button", name="下載案件說明 Markdown", exact=True).click()
    report_path = report_download.value.path()
    assert "外部模型尚未重播" in report_path.read_text(encoding="utf-8")
    _click(page, "覆核批次")
    _click(page, "建立／連結 Active Learning 覆核回合")
    _click(page, "準備含圖片的覆核包")
    with page.expect_download() as zip_download:
        page.get_by_role("button", name="下載覆核包 ZIP", exact=True).click()
    with zipfile.ZipFile(zip_download.value.path()) as archive:
        manifest = json.loads(archive.read("review_manifest.json"))
        assert manifest["purpose"] == "review_only"
        assert len([p for p in archive.namelist() if p.startswith("assets/")]) == 1
        manifest["items"][0]["review"] = {"verdict": "normal", "reviewer": "Reviewer", "reason": "對照規格確認正常"}
    returned = tmp_path / "reviewed.json"
    returned.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    page.get_by_test_id("stFileUploader").filter(has_text="回傳已覆核清單 JSON").locator('input[type="file"]').set_input_files(str(returned))
    _click(page, "驗證並讀回覆核")
    expect(page.get_by_text("覆核已讀回；未更新訓練標籤或模型成績", exact=True)).to_be_visible()
    case = CaseStore(fp_server.case_root).list()[0]
    assert case["candidates"][0]["review"]["verdict"] == "normal"
    assert case["cause_status"] == "hypothesis"
    assert case["improvement_status"] == "not_evaluated"
    ledger = [json.loads(line) for line in fp_server.rounds_path.read_text(encoding="utf-8").splitlines()]
    assert [e["kind"] for e in ledger] == ["start", "batch"]
    assert _snapshot(references) == before
    fp_server.restart()
    load_app(page, fp_server.base_url)
    _click(page, "誤逮案件")
    expect(page.get_by_text("反光誤逮 E2E", exact=False).first).to_be_visible()
    _click(page, "覆核批次")
    expect(page.get_by_text("候選 1 個區域 · 已覆核 1 · 待覆核 0", exact=True)).to_be_visible()
    assert not page.get_by_test_id("stException").count()
