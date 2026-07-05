"""M10-GUI 接線真實 E2E(設計 3_Architect_Design/M10_gui_wiring.md)。

驗「🛰 持續監看」接線:初始化工作區(profile.yaml)+ 匯出設定 + 立即掃描(→佇列+reason)+
標註閉環(標了下輪移出)+ 未存模型自動存。小合成資料;用檔案系統斷言降低 flaky。

── PG widget key 契約 ──
② _anomaly_tab_apply 內「🛰 持續監看」:
  anomaly_watch_ws          工作區目錄 text_input
  anomaly_watch_init_btn    「初始化監看」
  anomaly_watch_export_btn  「📤 匯出設定」(download_button)
  anomaly_watch_scan_btn    「▶ 立即掃描一次」
  anomaly_watch_status      服務狀態卡容器
  anomaly_watch_queue       監看佇列容器(縮圖卡 + reason)
  anomaly_watch_good_<id>   每卡「✅ 正常」(id=queue item id)
DOM 標記:掃描完成 → 主畫面含「掃描完成」。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from playwright.sync_api import expect

from ._anomaly_wizard import (TAB_APPLY, build_model, click_tab, enter_anomaly,
                              set_model_dir)
from .conftest import _add_folder, wait_idle

pytestmark = pytest.mark.e2e


def _paint(img, defect):
    if defect:
        for x in range(10, 74):
            for y in range(10, 74):
                img.putpixel((x, y), (220, 0, 0))
    return img


@pytest.fixture
def yolo_ds(tmp_path):
    from PIL import Image
    root = tmp_path / "yolo_ds"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("obj\n", encoding="utf-8")
    for i in range(8):
        img = _paint(Image.new("RGB", (128, 128), (0, 170, 0)), i % 3 == 0)
        img.save(root / "images" / f"o{i:02d}.jpg", quality=92)
        (root / "labels" / f"o{i:02d}.txt").write_text("0 0.5 0.5 0.9 0.9\n", encoding="utf-8")
    return root


def _fill(page, key, value):
    inp = page.locator(f'.st-key-{key} input').first
    inp.wait_for(state="visible", timeout=30000)
    inp.click(); inp.fill(str(value)); inp.press("Enter")
    wait_idle(page); page.wait_for_timeout(300)


def _wait_scan_done(page, timeout=300000):
    page.wait_for_function(
        """() => { const m = document.querySelector('[data-testid="stMain"]');
                   return m && (m.innerText || '').includes('掃描完成'); }""", timeout=timeout)
    wait_idle(page, timeout=60000); page.wait_for_timeout(600)


def _setup_watch(page, yolo_ds, model_dir, *, save=True):
    """①建模(可選存)→ ②設目標資料夾。工作區用預設(.lv_cache/al_watch);
    實際 ws 由 _actual_ws 從 UI 讀(不假設能覆蓋預設路徑)。"""
    build_model(page, yolo_ds, semantic_text="物件類別")
    set_model_dir(page, model_dir)
    if save:
        page.locator('.st-key-anomaly_save_model_btn button').click(); wait_idle(page)
    click_tab(page, TAB_APPLY)
    _add_folder(page, "anomaly_target_folder", str(yolo_ds))
    wait_idle(page); click_tab(page, TAB_APPLY)


def _actual_ws(page):
    """從 anomaly_watch_ws input 讀『實際生效』的工作區路徑(做檔案系統斷言)。"""
    return Path(page.locator('.st-key-anomaly_watch_ws input').first.input_value())


def _click(page, key):
    b = page.locator(f'.st-key-{key} button')
    b.wait_for(state="visible", timeout=30000)
    page.wait_for_function(
        f"""() => {{ const b = document.querySelector('.st-key-{key} button');
                    return b && !b.disabled; }}""", timeout=30000)
    b.click()


# ── AC-W1 / AC-W2:初始化(profile.yaml + 匯出)+ 立即掃描(佇列+reason)──────
def test_w1_w2_init_export_scan_queue(app_server, browser, yolo_ds, tmp_path):
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)
        _setup_watch(page, yolo_ds, tmp_path / "mdl", save=True)
        _click(page, "anomaly_watch_init_btn")
        page.wait_for_timeout(800)
        ws_dir = _actual_ws(page)
        # AC-W1:profile.yaml 產生 + 匯出鈕可見
        assert (ws_dir / "profile.yaml").exists(), f"初始化監看應產生 profile.yaml @ {ws_dir}"
        expect(page.locator('.st-key-anomaly_watch_export_btn')).to_be_visible(timeout=15000)
        # AC-W2:立即掃描 → 佇列渲染 + reason
        _click(page, "anomaly_watch_scan_btn")
        _wait_scan_done(page)
        q = page.locator('.st-key-anomaly_watch_queue')
        expect(q.locator('[data-testid="stImage"]').first).to_be_visible(timeout=30000)
        assert any(w in q.inner_text() for w in ("很不像正常", "拿不準", "邊界")), \
            "監看佇列卡缺人話理由(reason)"
        assert (ws_dir / "queue.jsonl").exists()
    finally:
        ctx.close()


# ── AC-W3:標註閉環(標了下輪移出)──────────────────────────────────────────
def test_w3_label_removes_next_round(app_server, browser, yolo_ds, tmp_path):
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)
        _setup_watch(page, yolo_ds, tmp_path / "mdl3", save=True)
        _click(page, "anomaly_watch_init_btn")
        page.wait_for_timeout(400)
        ws_dir = _actual_ws(page)
        _click(page, "anomaly_watch_scan_btn")
        _wait_scan_done(page)
        # 標第一張為正常 → labels.jsonl 多一行
        good = page.locator('[class*="st-key-anomaly_watch_good_"] button').first
        good.wait_for(state="visible", timeout=30000)
        good.click()
        wait_idle(page); page.wait_for_timeout(600)
        labels = [json.loads(l) for l in
                  (ws_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(labels) >= 1, "按『正常』應 append 一筆標註"
        labeled_id = labels[-1]["id"]
        # 再掃一次 → 該 id 不在 queue
        click_tab(page, TAB_APPLY)
        _click(page, "anomaly_watch_scan_btn")
        _wait_scan_done(page)
        q_ids = {json.loads(l)["id"] for l in
                 (ws_dir / "queue.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        assert labeled_id not in q_ids, "已標註的物件下輪不得再出現在佇列(閉環)"
    finally:
        ctx.close()


# ── AC-W5:未存模型 → 自動存 ───────────────────────────────────────────────
def test_w5_autosave(app_server, browser, yolo_ds, tmp_path):
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)
        _setup_watch(page, yolo_ds, tmp_path / "mdl5", save=False)   # 刻意不存
        _click(page, "anomaly_watch_init_btn")
        page.wait_for_timeout(400)
        ws_dir = _actual_ws(page)
        _click(page, "anomaly_watch_scan_btn")
        _wait_scan_done(page)
        # 佇列渲染=掃描成功=al_service 讀到磁碟凍結模型(load_frozen_model 缺 meta.json 會 raise)
        expect(page.locator('.st-key-anomaly_watch_queue [data-testid="stImage"]').first
               ).to_be_visible(timeout=30000)
        # 自動存:profile 的 model_dir 目錄出現 meta.json(auto-save 確實把模型寫到磁碟)
        import yaml
        prof = yaml.safe_load((ws_dir / "profile.yaml").read_text(encoding="utf-8"))
        assert (Path(prof["model_dir"]) / "meta.json").exists(), "未存模型時應自動存(meta.json 應存在)"
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()
