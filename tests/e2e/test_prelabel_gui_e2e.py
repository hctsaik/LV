"""M11-GUI 接線真實 E2E(設計 3_Architect_Design/M11_gui_wiring.md)。

真實行為驗收:含 head 模型 → ② 套用 → ③「🏷️ 預標」→ 預覽 accept 數 → 匯出到 tmp 目錄 →
**實際讀回 out_dir/labels/*.txt** 斷言行數==預覽 accept 數、cls_id 正確;來源資料夾零寫入(C6);
無 head 模型 → 友善降級(無匯出鈕、無例外)。

── PG 必須實作的 widget key 契約(M11_gui_wiring.md §3)──
③ 挑樣內:
  anomaly_prelabel_min_conf   信心門檻 slider(預設 0.5)
  anomaly_prelabel_classes    目標類別 multiselect(預設全選)
  anomaly_prelabel_out_dir    輸出資料夾 text_input
  anomaly_prelabel_export_btn 匯出按鈕
  anomaly_prelabel_preview    預覽容器(pred/conf/accept + stImage 縮圖;含文字「接受 N」)
DOM 穩定標記:匯出完成主畫面出現「✅ 預標匯出完成」(含「預標匯出完成」)。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from playwright.sync_api import expect

from ._anomaly_wizard import (TAB_SAMPLE, apply_model, build_model, click_tab,
                              enter_anomaly, set_model_dir)
from .conftest import wait_idle

pytestmark = pytest.mark.e2e


def _open_prelabel(page):
    """切 ③ → 展開「🏷️ 預標」expander(若收合)。"""
    click_tab(page, TAB_SAMPLE)
    page.wait_for_timeout(500)
    hdr = page.locator('[data-testid="stExpander"] summary', has_text="預標")
    hdr.first.wait_for(state="visible", timeout=30000)
    # 展開(details 未開才點,避免點到反而收合)
    if not page.evaluate(
        """() => { const s=[...document.querySelectorAll('[data-testid=\"stExpander\"] summary')]
                     .find(x=>(x.innerText||'').includes('預標'));
                   const d=s && s.closest('details'); return !!(d && d.open); }"""):
        hdr.first.click()
        wait_idle(page)
    page.wait_for_timeout(500)


def _accept_count(page) -> int:
    """從預覽容器讀「接受 N」。"""
    txt = page.locator('.st-key-anomaly_prelabel_preview').inner_text()
    m = re.search(r"接受\s*(\d+)", txt)
    assert m, f"預覽應顯示『接受 N』:{txt[:300]}"
    return int(m.group(1))


def _snapshot_dir(root: Path) -> dict:
    return {str(p.relative_to(root)): p.stat().st_size
            for p in root.rglob("*") if p.is_file()}


# ── AC-E1/E2/E3/E4:預標 → 預覽 → 匯出 → 讀回檔案 + 來源零寫入 ─────────────────
def test_e1_e4_prelabel_export_readback(app_server, browser, yolo_defect_at_nmin, tmp_path):
    ds = yolo_defect_at_nmin   # scratch×8 + stain×8、語義 defect → ① 建模含 head
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)
        build_main = build_model(page, ds["root"], semantic_text="瑕疵類別")
        assert "含分類頭" in build_main, f"應建出含分類頭模型;實際:{build_main[:400]}"
        set_model_dir(page, tmp_path / "mdl")
        page.locator('.st-key-anomaly_save_model_btn button').click()
        wait_idle(page)
        apply_model(page, ds["root"])   # ② 產生 result(records/obj_emb/scores)

        _open_prelabel(page)
        # AC-E1:預覽出現(pred/conf/accept 文字 + 縮圖)
        preview = page.locator('.st-key-anomaly_prelabel_preview')
        expect(preview.locator('[data-testid="stImage"]').first).to_be_visible(timeout=30000)
        n_accept = _accept_count(page)
        assert n_accept >= 1, "contamination 門檻至少 flag 1 物件 + 可分類 → accept 應 ≥ 1"

        # AC-E3:設輸出目錄 → 匯出
        out_dir = tmp_path / "prelabel_out"
        inp = page.locator('.st-key-anomaly_prelabel_out_dir input').first
        inp.wait_for(state="visible", timeout=30000)
        inp.click()
        inp.fill(str(out_dir))
        inp.press("Enter")
        wait_idle(page)
        _open_prelabel(page)   # rerun 可能收合
        src_before = _snapshot_dir(ds["root"])
        btn = page.locator('.st-key-anomaly_prelabel_export_btn button')
        btn.wait_for(state="visible", timeout=30000)
        page.wait_for_function(
            """() => { const b=document.querySelector('.st-key-anomaly_prelabel_export_btn button');
                       return b && !b.disabled; }""", timeout=30000)
        btn.click()
        page.wait_for_function(
            """() => { const m=document.querySelector('[data-testid=\"stMain\"]');
                       return m && (m.innerText||'').includes('預標匯出完成'); }""", timeout=60000)
        wait_idle(page)

        # AC-E2:讀回檔案總行數 == 預覽 accept 數
        label_files = list((out_dir / "labels").glob("*.txt"))
        assert label_files, f"匯出應在 {out_dir}/labels 產生 .txt"
        lines = [ln for f in label_files
                 for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == n_accept, \
            f"匯出行數({len(lines)})應 == 預覽宣稱 accept 數({n_accept})"
        # AC-E3:每行 cls_id ∈ {0,1}、4 座標 ∈ [0,1];classes.txt 含 scratch/stain
        for ln in lines:
            parts = ln.split()
            assert parts[0] in ("0", "1"), f"cls_id 應為 head 類別索引 0/1:{ln}"
            coords = list(map(float, parts[1:5]))
            assert all(0.0 <= c <= 1.0 for c in coords), f"座標應正規化 ∈[0,1]:{ln}"
        classes_txt = (out_dir / "classes.txt").read_text(encoding="utf-8")
        assert "scratch" in classes_txt and "stain" in classes_txt

        # AC-E4:來源資料夾零寫入(C6 真實不變量)
        assert _snapshot_dir(ds["root"]) == src_before, "預標匯出不得改動來源資料夾(C6)"
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()


# ── AC-E5:無 head 模型 → 友善降級(無匯出鈕、無例外)──────────────────────────
def test_e5_no_head_graceful(app_server, browser, yolo_defect_at_nmin, tmp_path):
    ds = yolo_defect_at_nmin
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)
        # 語義=物件類別 → head 不解鎖(object_semantic)→ 無分類頭
        build_model(page, ds["root"], semantic_text="物件類別")
        set_model_dir(page, tmp_path / "mdl_nohead")
        page.locator('.st-key-anomaly_save_model_btn button').click()
        wait_idle(page)
        apply_model(page, ds["root"])
        _open_prelabel(page)
        body = page.locator('[data-testid="stMain"]').inner_text()
        assert "無分類頭" in body, f"無 head 應友善提示『無分類頭…』;實際:{body[-600:]}"
        # 無匯出鈕
        assert page.locator('.st-key-anomaly_prelabel_export_btn').count() == 0, \
            "無 head 時不應出現匯出鈕"
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()
