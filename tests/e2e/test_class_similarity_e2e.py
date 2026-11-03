"""E2E: 類別相似度分布面板（image / object、舊工具回歸、購物車）。

真實 Streamlit + Playwright。自備 LV_CACHE_DIR：重用 explain_app_server。
不改共用 conftest 行為、不寫來源資料集。
"""
from __future__ import annotations

import pytest
from playwright.sync_api import expect

from .conftest import wait_idle
from ._class_similarity_harness import (
    PANEL_BAD,
    PANEL_CONF,
    PANEL_SIM,
    _no_exception,
    candidate_n,
    compute_similarity,
    make_detector_dataset,
    metric_value,
    new_page,
    open_label_quality,
    run_image_visualize,
    run_object_visualize,
    select_quality_panel,
    source_snapshot,
    toggle_first_record_checkbox,
)

pytestmark = pytest.mark.e2e
expect.set_options(timeout=20000)


@pytest.fixture(scope="session")
def cs_object_dataset(tmp_path_factory):
    return make_detector_dataset(tmp_path_factory.mktemp("cs_obj"))


def test_ac13_regression_old_tools(explain_app_server, browser, synthetic_dataset):
    """# AC13 舊混淆／疑似標錯、相似、購物車、送標入口仍在。"""
    ctx, page = new_page(browser, explain_app_server)
    try:
        run_image_visualize(page, synthetic_dataset)
        open_label_quality(page)
        select_quality_panel(page, PANEL_CONF)
        expect(page.get_by_text("k 近鄰類別分布", exact=False).last).to_be_visible()
        select_quality_panel(page, PANEL_BAD)
        expect(page.get_by_text("鄰居多數是別類", exact=False).last).to_be_visible()
        select_quality_panel(page, PANEL_SIM)
        expect(page.get_by_role("button", name="計算類別相似度").last).to_be_visible()

        page.locator(".st-key-viz_panel_view").get_by_text("選取", exact=True).click()
        wait_idle(page)
        expect(page.locator(".st-key-viz_similar_btn button").last).to_be_visible()
        expect(page.locator(".st-key-viz_add_btn button").last).to_be_visible()

        page.locator(".st-key-viz_panel_view").get_by_text("匯出清單", exact=True).click()
        wait_idle(page)
        expect(page.get_by_text("策展購物車", exact=False).last).to_be_visible()
        expect(page.get_by_text("購物車是空的", exact=False).last).to_be_visible()

        expect(page.locator(".st-key-viz_clear_sel_plot")).to_have_count(1)
        _no_exception(page)
    finally:
        ctx.close()


def test_ac14_image_and_object_source_readonly(
    explain_app_server, browser, synthetic_dataset, cs_object_dataset,
):
    """# AC14 真實 DINO image 與 object 各一例；來源清單＋內容 hash 不變。"""
    img_before_names, img_before_hash = source_snapshot(synthetic_dataset)
    obj_before_names, obj_before_hash = source_snapshot(cs_object_dataset)
    ctx, page = new_page(browser, explain_app_server)
    try:
        run_image_visualize(page, synthetic_dataset)
        compute_similarity(page)
        expect(page.get_by_text("分析母體：", exact=False).first).to_be_visible()
        n_img = int(metric_value(page, "分析總數"))
        assert n_img == 24, f"image records should be 24, got {n_img}"
        assert candidate_n(page) >= 0
        expect(page.get_by_text("box 使用全量", exact=False).first).to_be_visible()
        img_after_names, img_after_hash = source_snapshot(synthetic_dataset)
        assert img_after_names == img_before_names
        assert img_after_hash == img_before_hash

        run_object_visualize(page, cs_object_dataset)
        compute_similarity(page)
        n_obj = int(metric_value(page, "分析總數"))
        assert n_obj == 16, f"object records should be 16, got {n_obj}"
        expect(page.get_by_text("分析母體：", exact=False).first).to_be_visible()
        obj_after_names, obj_after_hash = source_snapshot(cs_object_dataset)
        assert obj_after_names == obj_before_names
        assert obj_after_hash == obj_before_hash
        _no_exception(page)
    finally:
        ctx.close()


def test_ac07_click_table_and_clear(explain_app_server, browser, synthetic_dataset):
    """# AC07 表格證據可開；清除本面板勾選不復活。"""
    ctx, page = new_page(browser, explain_app_server)
    try:
        run_image_visualize(page, synthetic_dataset)
        compute_similarity(page)
        evidence = page.get_by_role("button", name="證據")
        expect(evidence.last).to_be_visible()
        evidence.last.scroll_into_view_if_needed()
        evidence.last.click()
        wait_idle(page)
        expect(page.get_by_text("同類 Top-K", exact=False).last).to_be_visible()
        expect(page.get_by_text("最接近異類 Top-K", exact=False).last).to_be_visible()

        toggle_first_record_checkbox(page)
        page.get_by_role("button", name="清除本面板勾選").last.click()
        wait_idle(page)
        leftover = []
        boxes = page.locator('[data-testid="stCheckbox"]')
        for i in range(boxes.count()):
            box = boxes.nth(i)
            try:
                stale = box.evaluate(
                    "el => Boolean(el.closest('[data-stale=\"true\"]'))")
                if stale or not box.is_visible():
                    continue
                inp = box.locator("input")
                if not (inp.count() and inp.first.is_checked()):
                    continue
                text = box.inner_text()
                if "僅候選" in text or "margin" in text:
                    continue
                leftover.append(text)
            except Exception:
                continue
        assert leftover == [], leftover
        _no_exception(page)
    finally:
        ctx.close()


def test_ac10_cart_add_skip(explain_app_server, browser, synthetic_dataset):
    """# AC10 購物車新增／重加略過既有、來源=類別相似度。"""
    ctx, page = new_page(browser, explain_app_server)
    try:
        run_image_visualize(page, synthetic_dataset)
        compute_similarity(page)
        toggle_first_record_checkbox(page)
        page.get_by_role("button", name="加入購物車").last.scroll_into_view_if_needed()
        page.get_by_role("button", name="加入購物車").last.click()
        wait_idle(page)
        expect(page.get_by_text("購物車新增 1", exact=False).first).to_be_visible()
        page.get_by_role("button", name="加入購物車").last.click()
        wait_idle(page)
        expect(page.get_by_text("略過 1", exact=False).first).to_be_visible()

        page.locator(".st-key-viz_panel_view").get_by_text("匯出清單", exact=True).click()
        wait_idle(page)
        expect(page.get_by_text("類別相似度", exact=False).first).to_be_visible()
        expect(page.get_by_text("策展購物車 — 共 1 張", exact=False).first).to_be_visible()
        _no_exception(page)
    finally:
        ctx.close()
