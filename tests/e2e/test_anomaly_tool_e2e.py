"""E2E 真實行為驗收:瑕疵偵測工具(設計 06_anomaly_tool.md 的 E2E-AC*)。

這是「使用者真的能用」的 spec,不進 PG 自主修綠迴圈(gate.py 以 -m "not e2e" 排除),
由 /ux-test 或人觸發。**真實行為斷言**:不是查 element 存在,而是查
「被注入缺陷的物件真的排到前段」「匯出的是原圖」「錯誤路徑顯示中文指引」。

PG 實作 GUI 時需對齊以下契約鍵(目前 UI 未建,故本檔在 UI 完成前為紅/失敗,屬預期):
  - 工具入口:可由文字「瑕疵偵測」進入該工具分頁。
  - 資料夾輸入:沿用既有 `_add_folder(page, "anomaly_folder", <root>)` 模式(list_key=anomaly_folder)。
  - 執行鈕:`.st-key-anomaly_run button`。
  - 排序結果區:`.st-key-anomaly_ranked`(內含每個物件來源檔名文字,順序=由最可疑到最不可疑)。
  - 錯誤訊息區:無 `labels/` 時顯示中文指引(含「YOLO」「標註」字樣),不得出現 stException。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from playwright.sync_api import expect

from .conftest import _add_folder, load_app, wait_idle

pytestmark = pytest.mark.e2e
expect.set_options(timeout=20000)


@pytest.fixture(scope="module")
def anomaly_page(app_server, browser, synthetic_yolo_dataset):
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    load_app(page, app_server)
    page.locator('.st-key-tool_switch').get_by_text("瑕疵偵測", exact=True).click()  # 進入第 8 工具
    wait_idle(page)
    yield page, synthetic_yolo_dataset
    ctx.close()


def _no_exception(page):
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)


def _run(page, root: Path):
    _add_folder(page, "anomaly_folder", str(root))
    wait_idle(page)
    page.locator('.st-key-anomaly_run button').click()
    wait_idle(page, timeout=180000)  # 首跑含模型載入,給足時間


def test_defects_rank_in_top(anomaly_page):  # E2E-AC1(真實行為,非 element 存在)
    page, ds = anomaly_page
    _run(page, ds["root"])
    _no_exception(page)
    ranked_text = page.locator('.st-key-anomaly_ranked').inner_text()
    # 取出排序清單中出現的來源檔 stem,前段(前 2*K 名)應涵蓋全部缺陷檔
    order = re.findall(r'(normal_\d+|defect_\d+)', ranked_text)
    assert order, "排序結果區應列出物件來源檔名"
    k = ds["n_defect"]
    top = order[: 2 * k]
    defect_stems = {s for s, _ in ds["defect_keys"]}
    assert defect_stems.issubset(set(top)), f"缺陷應集中在前段;實際前段={top}"


def test_select_and_add_to_cart(anomaly_page):  # E2E-AC2
    page, ds = anomaly_page
    # 一鍵把離群/可疑物件加入購物車 → 計數應變非零(離群候選 ⊇ 缺陷)
    cart = page.locator('.st-key-anomaly_cart_count')
    wait_idle(page)
    btn = page.locator('.st-key-anomaly_select_outliers button')
    expect(btn).to_be_visible()
    btn.click()
    expect(cart).to_contain_text(re.compile(r"[1-9]"))  # 等 rerun 把計數更新成非零
    _no_exception(page)
    after = int(re.sub(r"\D", "", cart.inner_text()) or 0)
    assert after >= ds["n_defect"]  # 離群候選含全部缺陷


def test_export_is_original_image(anomaly_page, tmp_path):  # E2E-AC3(匯出原圖,非裁切)
    import io
    import zipfile
    from PIL import Image
    page, ds = anomaly_page
    with page.expect_download() as dl:
        page.locator('.st-key-anomaly_export button').click()
    data = Path(dl.value.path()).read_bytes()
    zf = zipfile.ZipFile(io.BytesIO(data))
    imgs = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    assert imgs, "zip 應含影像"
    with Image.open(io.BytesIO(zf.read(imgs[0]))) as im:
        assert im.size == (128, 128)  # 原圖尺寸(夾具 128x128),非裁切小圖
    assert "manifest.csv" in zf.namelist()


def test_confirm_normal_fewshot_rerun(app_server, browser, synthetic_yolo_dataset):
    """2-stage few-shot 確認流程:執行→把最不可疑標為正常範例→用乾淨 bank 重跑(真實行為)。"""
    ds = synthetic_yolo_dataset
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    load_app(page, app_server)
    page.locator('.st-key-tool_switch').get_by_text("瑕疵偵測", exact=True).click()
    wait_idle(page)
    _add_folder(page, "anomaly_folder", str(ds["root"]))
    wait_idle(page)
    page.locator('.st-key-anomaly_run button').click()
    wait_idle(page, timeout=180000)
    expect(page.locator('.st-key-anomaly_ranked')).to_contain_text("defect_")
    # 把最不可疑的標為正常範例(few-shot 種子)→ 應出現「已標記正常」狀態
    seed = page.locator('.st-key-anomaly_autoseed_normal button')
    expect(seed).to_be_visible()
    seed.click()
    expect(page.locator('[data-testid="stAppViewContainer"]')).to_contain_text("已標記正常")
    # 用乾淨 few-shot bank 重跑(第一次執行後 sidebar 收起,等重開鈕出現再點開)
    reopen = page.locator('.st-key-reopen_sidebar button')
    expect(reopen).to_be_visible()
    reopen.click()
    wait_idle(page)
    run2 = page.locator('.st-key-anomaly_run button')
    expect(run2).to_be_visible()
    run2.click()
    wait_idle(page, timeout=180000)
    _no_exception(page)
    expect(page.locator('.st-key-anomaly_ranked')).to_contain_text("defect_")
    ranked = page.locator('.st-key-anomaly_ranked').inner_text()
    order = re.findall(r'(normal_\d+|defect_\d+)', ranked)
    defect_stems = {s for s, _ in ds["defect_keys"]}
    assert defect_stems.issubset(set(order[: 2 * ds["n_defect"]]))  # 乾淨 bank 仍抓到缺陷
    ctx.close()


def test_error_path_no_labels(app_server, browser, tmp_path):  # E2E-AC6(友善錯誤,非 traceback)
    (tmp_path / "images").mkdir()
    from PIL import Image
    Image.new("RGB", (64, 64), (0, 170, 0)).save(tmp_path / "images" / "a.jpg")  # 無 labels/
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
    page = ctx.new_page()
    load_app(page, app_server)
    page.locator('.st-key-tool_switch').get_by_text("瑕疵偵測", exact=True).click()
    wait_idle(page)
    _add_folder(page, "anomaly_folder", str(tmp_path))
    page.locator('.st-key-anomaly_run button').click()
    wait_idle(page)
    container = page.locator('[data-testid="stAppViewContainer"]')
    # 等執行後的 rerun 真的畫出指引再讀(否則 wait_idle 可能在 rerun 前 ~0.5s 就回,
    # 抓到執行前的預設文字 → race);expect 自動重試覆蓋這個 gap。
    expect(container).to_contain_text("找不到 YOLO 標註")
    body = container.inner_text()
    assert ("YOLO" in body and "標註" in body)        # 中文指引
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)  # 無原始 traceback
    ctx.close()
