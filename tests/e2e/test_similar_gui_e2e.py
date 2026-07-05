"""M12a-GUI 接線真實 E2E(設計 3_Architect_Design/M12a_gui_wiring.md)。

真實行為:含兩群(scratch 紅左上 / stain 藍左下)的資料 → ② 套用 → ③「🔎 找相似」→
挑一顆參考 → 佇列裡**同群物件佔多數**(真實 DINOv2:同群 cosine > 跨群);換到另一群當參考 →
佇列多數翻成新群(排序真的跟著參考走)。

── PG 必須實作的 widget key 契約(M12a_gui_wiring.md §3)──
③ 挑樣內:
  anomaly_sim_ref_idx   參考物件索引(number_input / selectbox)
  anomaly_sim_k         取幾個最像
  anomaly_sim_ref       參考顯示容器(含文字「參考:<label>」)
  anomaly_sim_queue     相似佇列容器(縮圖牆;每卡含 label;不含參考自身)
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


def _set_ref_idx(page, idx: int):
    """設參考物件索引(number_input)並等 rerun。"""
    inp = page.locator('.st-key-anomaly_sim_ref_idx input').first
    inp.wait_for(state="visible", timeout=30000)
    inp.fill(str(idx))
    inp.press("Enter")
    wait_idle(page)
    page.wait_for_timeout(400)
    _open_similar(page)


def _ref_label(page) -> str:
    txt = page.locator('.st-key-anomaly_sim_ref').inner_text()
    m = re.search(r"參考[:：]\s*(scratch|stain)", txt)
    assert m, f"參考容器應顯示『參考:<label>』:{txt[:200]}"
    return m.group(1)


def _queue_label_counts(page) -> dict:
    txt = page.locator('.st-key-anomaly_sim_queue').inner_text()
    return {lab: len(re.findall(lab, txt)) for lab in _LABELS}


def _majority_label(counts: dict) -> str:
    return max(_LABELS, key=lambda l: counts[l])


# ── AC-S1 / AC-S2:相似佇列同群佔多數 + 換參考翻群 ────────────────────────────
def test_s1_s2_similar_queue_follows_reference(app_server, browser, yolo_defect_at_nmin, tmp_path):
    ds = yolo_defect_at_nmin
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)
        build_model(page, ds["root"], semantic_text="物件類別")   # 找相似不需 head
        set_model_dir(page, tmp_path / "mdl_sim")
        page.locator('.st-key-anomaly_save_model_btn button').click()
        wait_idle(page)
        apply_model(page, ds["root"])

        _open_similar(page)
        _set_ref_idx(page, 0)
        l0 = _ref_label(page)
        # AC-S1:佇列渲染 + 同群(l0)佔多數
        queue = page.locator('.st-key-anomaly_sim_queue')
        expect(queue.locator('[data-testid="stImage"]').first).to_be_visible(timeout=30000)
        c0 = _queue_label_counts(page)
        assert _majority_label(c0) == l0, \
            f"參考={l0} 時相似佇列應以 {l0} 佔多數(同群 cosine 高);實際 {c0}"

        # AC-S2:掃到一個「另一群」的參考索引 → 佇列多數翻成新群
        other_idx = None
        for j in range(1, 16):
            _set_ref_idx(page, j)
            if _ref_label(page) != l0:
                other_idx = j
                break
        assert other_idx is not None, "資料含兩群,應找得到 label≠參考0 的物件"
        lj = _ref_label(page)   # != l0
        cj = _queue_label_counts(page)
        assert _majority_label(cj) == lj, \
            f"換參考={lj}(idx={other_idx})後佇列應以 {lj} 佔多數(排序跟著參考走);實際 {cj}"
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()
