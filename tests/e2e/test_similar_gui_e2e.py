"""M12a-GUI 接線真實 E2E(設計 3_Architect_Design/M12a_gui_wiring.md,含 UX 精修:參考依據切換)。

真實行為:含兩群(scratch 紅左上 / stain 藍左下)的資料 → ② 套用 → ③「🔎 找相似」→
**參考依據=類別**,選 scratch → 佇列裡 scratch 佔多數(真實 DINOv2:同群 cosine > 跨群);
切成 stain → 佇列多數翻成 stain(排序真的跟著參考類別走)。

── PG 必須實作的 widget key 契約 ──
③ 挑樣內:
  anomaly_sim_ref_mode    參考依據 radio(類別 / 具體物件;預設「類別」)
  anomaly_sim_ref_class   參考類別 selectbox(類別模式)
  anomaly_sim_ref_idx     參考物件 selectbox/number(具體物件模式)
  anomaly_sim_queue       相似佇列容器(縮圖牆;每卡含 label)
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
    hdr = page.locator('[data-testid="stExpander"] summary', has_text="找相似")
    hdr.first.wait_for(state="visible", timeout=30000)
    if not page.evaluate(
        """() => { const s=[...document.querySelectorAll('[data-testid=\"stExpander\"] summary')]
                     .find(x=>(x.innerText||'').includes('找相似'));
                   const d=s && s.closest('details'); return !!(d && d.open); }"""):
        hdr.first.click()
        wait_idle(page)
    page.wait_for_timeout(500)


def _pick_ref_class(page, cls, key="anomaly_sim_ref_class"):
    """選參考類別(baseweb selectbox)。"""
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
    page.get_by_role("option", name=cls, exact=True).first.click()
    wait_idle(page)
    page.wait_for_timeout(400)
    _open_similar(page)


def _queue_label_counts(page) -> dict:
    txt = page.locator('.st-key-anomaly_sim_queue').inner_text()
    return {lab: len(re.findall(lab, txt)) for lab in _LABELS}


def _majority_label(counts: dict) -> str:
    return max(_LABELS, key=lambda l: counts[l])


# ── AC-S1 / AC-S2:依「類別」當參考 → 佇列同群佔多數 + 換類別翻群 ─────────────────
def test_s1_s2_class_reference_follows(app_server, browser, yolo_defect_at_nmin, tmp_path):
    ds = yolo_defect_at_nmin
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)
        build_model(page, ds["root"], semantic_text="物件類別")   # 找相似不需分類器
        set_model_dir(page, tmp_path / "mdl_sim")
        page.locator('.st-key-anomaly_save_model_btn button').click()
        wait_idle(page)
        apply_model(page, ds["root"])

        _open_similar(page)
        # 預設參考依據=類別;選 scratch → 佇列 scratch 佔多數
        _pick_ref_class(page, "scratch")
        queue = page.locator('.st-key-anomaly_sim_queue')
        expect(queue.locator('[data-testid="stImage"]').first).to_be_visible(timeout=30000)
        c0 = _queue_label_counts(page)
        assert _majority_label(c0) == "scratch", f"參考類別=scratch → 佇列應以 scratch 佔多數:{c0}"
        # 切成 stain → 佇列多數翻成 stain
        _pick_ref_class(page, "stain")
        c1 = _queue_label_counts(page)
        assert _majority_label(c1) == "stain", f"換參考類別=stain → 佇列應翻成 stain 佔多數:{c1}"
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()
