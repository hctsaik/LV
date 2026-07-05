"""M12a-GUI「🔎 找同款」(by-example)真實 E2E。

設計修正(多 agent 討論後):找相似 = 挑**一顆**當範本的 by-example 檢索(不再有「像某一類」——
那是 silent-wrong 且與『分種類/預標』重疊)。參考來源可切換 ② 這次掃描 / ① 建模範例。

真實行為:兩群(scratch 紅左上 / stain 藍左下)→ ② 套用 → ③「找同款」→ 挑一顆 scratch 物件當範本 →
佇列多數 scratch(真實 DINOv2:同群 cosine > 跨群);換挑 stain 物件 → 佇列多數翻 stain(排序跟著範本走)。

── widget key 契約 ──
  anomaly_sim_ref_src   參考來源 radio(② 這次掃描結果 / ① 建模已知範例;>1 來源才出現)
  anomaly_sim_ref_idx   範本物件 selectbox(第N · label · 檔名)
  anomaly_sim_queue     相似佇列(縮圖牆;每卡含 label)
"""
from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from ._anomaly_wizard import (TAB_SAMPLE, apply_model, build_model, click_tab,
                              enter_anomaly, set_model_dir)
from .conftest import wait_idle

pytestmark = pytest.mark.e2e

_LABELS = ("scratch", "stain")


def _open_similar(page):
    click_tab(page, TAB_SAMPLE)
    page.wait_for_timeout(500)
    hdr = page.locator('[data-testid="stExpander"] summary', has_text="找同款")
    hdr.first.wait_for(state="visible", timeout=30000)
    if not page.evaluate(
        """() => { const s=[...document.querySelectorAll('[data-testid=\"stExpander\"] summary')]
                     .find(x=>(x.innerText||'').includes('找同款'));
                   const d=s && s.closest('details'); return !!(d && d.open); }"""):
        hdr.first.click()
        wait_idle(page)
    page.wait_for_timeout(500)


def _pick_ref_object(page, want_label, key="anomaly_sim_ref_idx"):
    """在範本 selectbox 挑第一個 label 含 want_label 的物件(by-example)。"""
    sb = page.locator(f'.st-key-{key} [data-baseweb="select"]')
    sb.wait_for(state="visible", timeout=30000)
    for _ in range(3):
        sb.click()
        try:
            page.wait_for_selector('[role="option"]', timeout=6000)
            break
        except Exception:
            page.keyboard.press("Escape")
            wait_idle(page)
    page.locator('[role="option"]', has_text=want_label).first.click()
    wait_idle(page)
    page.wait_for_timeout(400)
    _open_similar(page)


def _queue_label_counts(page) -> dict:
    txt = page.locator('.st-key-anomaly_sim_queue').inner_text()
    return {lab: len(re.findall(lab, txt)) for lab in _LABELS}


def _majority_label(counts: dict) -> str:
    return max(_LABELS, key=lambda l: counts[l])


# ── 找同款(by-example):範本=某顆物件 → 佇列同群佔多數 + 換範本翻群 ────────────
def test_by_example_queue_follows_reference(app_server, browser, yolo_defect_at_nmin, tmp_path):
    ds = yolo_defect_at_nmin
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)
        build_model(page, ds["root"], semantic_text="物件類別")   # 找同款不需分類頭
        set_model_dir(page, tmp_path / "mdl_sim")
        page.locator('.st-key-anomaly_save_model_btn button').click()
        wait_idle(page)
        apply_model(page, ds["root"])

        _open_similar(page)
        # 挑一顆 scratch 物件當範本(參考來源預設 ② 這次掃描結果)→ 佇列 scratch 佔多數
        _pick_ref_object(page, "scratch")
        queue = page.locator('.st-key-anomaly_sim_queue')
        expect(queue.locator('[data-testid="stImage"]').first).to_be_visible(timeout=30000)
        c0 = _queue_label_counts(page)
        assert _majority_label(c0) == "scratch", f"範本=scratch 物件 → 佇列應以 scratch 佔多數:{c0}"
        # 換挑 stain 物件 → 佇列多數翻成 stain(排序跟著範本走)
        _pick_ref_object(page, "stain")
        c1 = _queue_label_counts(page)
        assert _majority_label(c1) == "stain", f"換範本=stain 物件 → 佇列應翻成 stain 佔多數:{c1}"
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()
