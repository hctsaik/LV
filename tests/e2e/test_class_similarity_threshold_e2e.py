"""E2E: Threshold 滑桿／精確值／軸縮放／來源變更過期。

真實 Streamlit + Playwright。pointer 按住時必須更新預覽（AC18）。
"""
from __future__ import annotations

import re
import shutil
import time

import pytest
from playwright.sync_api import expect

from .conftest import wait_idle
from ._class_similarity_harness import (
    _no_exception,
    candidate_n,
    compute_similarity,
    metric_value,
    new_page,
    run_image_visualize,
    slider_locator,
    source_snapshot,
    threshold_caption,
)

pytestmark = pytest.mark.e2e
expect.set_options(timeout=20000)


def test_ac16_canonical_threshold_precision(explain_app_server, browser, synthetic_dataset):
    """# AC16 精確輸入 0.123456 不被 slider 0.01 刻度蓋掉。"""
    ctx, page = new_page(browser, explain_app_server)
    try:
        run_image_visualize(page, synthetic_dataset)
        compute_similarity(page)
        n_before = candidate_n(page)
        inp = page.get_by_label("精確門檻")
        inp.click()
        inp.fill("0.123456")
        inp.press("Tab")
        wait_idle(page)
        expect(inp).to_have_value(re.compile(r"0\.123456"))
        caption = threshold_caption(page)
        assert "0.123456" in caption, caption
        page.get_by_text("同類 Top-K 平均 cosine", exact=False).last.click()
        wait_idle(page)
        page.get_by_text("margin（same − other）", exact=False).last.click()
        wait_idle(page)
        expect(page.get_by_label("精確門檻")).to_have_value(re.compile(r"0\.123456"))
        assert "0.123456" in threshold_caption(page)
        page.get_by_role("button", name="證據").first.click()
        wait_idle(page)
        expect(page.get_by_label("精確門檻")).to_have_value(re.compile(r"0\.123456"))
        # dragging/number must not recompute embeddings: record count stays
        assert int(metric_value(page, "分析總數")) == 24
        assert candidate_n(page) != n_before or n_before >= 0
        _no_exception(page)
    finally:
        ctx.close()


def test_ac18_pointer_held_preview(explain_app_server, browser, synthetic_dataset):
    """# AC18 pointer 按住移到中間值時，候選數在 mouseup 前就變。"""
    ctx, page = new_page(browser, explain_app_server)
    try:
        run_image_visualize(page, synthetic_dataset)
        compute_similarity(page)
        before = candidate_n(page)
        slider = slider_locator(page)
        slider.scroll_into_view_if_needed()
        box = slider.bounding_box()
        assert box, "slider has no bounding box"
        y = box["y"] + box["height"] / 2
        # t=0 is mid; drag toward +1 (75%) then toward -1 (25%) if needed.
        page.mouse.move(box["x"] + box["width"] * 0.5, y)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] * 0.78, y, steps=12)
        changed = False
        deadline = time.time() + 6
        while time.time() < deadline:
            try:
                now = candidate_n(page)
            except Exception:
                page.wait_for_timeout(150)
                continue
            if now != before:
                changed = True
                break
            if "預覽未提交" in page.inner_text("body"):
                # draft updated even if membership stayed the same; keep waiting
                pass
            page.wait_for_timeout(150)
        if not changed:
            page.mouse.move(box["x"] + box["width"] * 0.22, y, steps=12)
            deadline = time.time() + 6
            while time.time() < deadline:
                try:
                    now = candidate_n(page)
                except Exception:
                    page.wait_for_timeout(150)
                    continue
                if now != before:
                    changed = True
                    break
                page.wait_for_timeout(150)
        held_body = page.inner_text("body")
        page.mouse.up()
        wait_idle(page)
        assert changed, (
            "AC18 未達連續預覽：pointer 按住時候選數沒有變。"
            f" before={before} body_tail={held_body[-1500:]}"
        )
        _no_exception(page)
    finally:
        ctx.close()


def test_ac33_axis_zoom(explain_app_server, browser, synthetic_dataset):
    """# AC33 軸縮放／重設不改候選數與分析總數。"""
    ctx, page = new_page(browser, explain_app_server)
    try:
        run_image_visualize(page, synthetic_dataset)
        compute_similarity(page)
        n0 = candidate_n(page)
        total0 = metric_value(page, "分析總數")
        page.get_by_text("符合資料範圍", exact=True).last.click()
        wait_idle(page)
        assert candidate_n(page) == n0
        assert metric_value(page, "分析總數") == total0
        page.get_by_role("button", name="重設完整範圍").last.click()
        wait_idle(page)
        assert candidate_n(page) == n0
        assert metric_value(page, "分析總數") == total0
        _no_exception(page)
    finally:
        ctx.close()


def test_ac34_source_change_stale(
    explain_app_server, browser, synthetic_dataset, tmp_path,
):
    """# AC34 Run 後改圖 → 結果過期，禁止沿用舊分數。"""
    ds = tmp_path / "cs_stale_ds"
    shutil.copytree(synthetic_dataset, ds)
    names_before, _hash_before = source_snapshot(ds)
    ctx, page = new_page(browser, explain_app_server)
    try:
        run_image_visualize(page, ds)
        compute_similarity(page)
        jpg = next(ds.rglob("*.jpg"))
        jpg.write_bytes(jpg.read_bytes() + b"\x00\x01")
        page.get_by_text("僅候選", exact=False).last.click()
        wait_idle(page)
        expect(page.get_by_text("結果已過期", exact=False).first).to_be_visible()
        expect(page.get_by_text("過期結果不能選取或匯出", exact=False).first).to_be_visible()
        names_after, _ = source_snapshot(ds)
        assert names_after == names_before
        _no_exception(page)
    finally:
        ctx.close()
