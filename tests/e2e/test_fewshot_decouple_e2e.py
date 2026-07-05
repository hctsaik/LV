"""M14 以樣搜樣特徵器解耦真實 E2E(設計 3_Architect_Design/M14_extractor_decouple.md)。

旗艦:**乾淨 session(從未建任何 anomaly 模型)**→ 以樣搜樣選樣本 → 建樣本集(預設特徵器)→ 海掃 → 佇列。
證明:AC-D1 死路拆掉(無模型也能到①並建成)+ AC-D2 解耦後檢索正確(相似物件排前、帶建議類別)+ 來源零寫入(C6)。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import expect

from .conftest import _add_folder, load_app, wait_idle
from .test_fewshot_gui_e2e import _click, _fewshot_step, _switch_fewshot

pytestmark = pytest.mark.e2e


def _snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*") if p.is_file()}


def test_ac_d1_d2_no_model_full_flow(app_server, browser, yolo_defect_at_nmin):
    # AC-D1:乾淨 session、從未建 anomaly 模型 → 以樣搜樣全程可用(死路拆掉)
    # AC-D2:解耦後相似物件仍排前 + 帶建議類別;C6 來源零寫入
    ds = yolo_defect_at_nmin
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        load_app(page, app_server)             # 乾淨 session:未經瑕疵偵測、無任何 anomaly 模型
        _switch_fewshot(page)
        # 能導覽到「① 樣本集」內文(舊版死路會提前 return、不渲染步驟列)→ 到得了 = 死路已拆
        _fewshot_step(page, "樣本集", "建立樣本集")
        _add_folder(page, "fewshot_sample_folder", str(ds["root"]))
        wait_idle(page)
        _fewshot_step(page, "樣本集", "建立樣本集")
        src_before = _snapshot(ds["root"])
        _click(page, "fewshot_build_bank_btn")   # 用預設 dinov2_vits14 特徵器,免先建 anomaly 模型
        page.wait_for_function(
            """() => { const m=document.querySelector('[data-testid="stMain"]');
                       const t=(m && m.innerText) || '';
                       return t.includes('樣本集已建立') || t.includes('建立樣本集失敗'); }""",
            timeout=240000)
        wait_idle(page)
        info = page.locator('.st-key-fewshot_bank_info').inner_text()
        assert "樣本集已建立" in info, f"無 anomaly 模型也應能建樣本集(預設特徵器):{info[:200]}"

        # ② 海掃 → 佇列:AC-D2 相似物件排前 + 建議類別
        _fewshot_step(page, "海掃", "相似度門檻")
        _add_folder(page, "fewshot_target_folder", str(ds["root"]))
        wait_idle(page)
        _fewshot_step(page, "海掃", "相似度門檻")
        _click(page, "fewshot_scan_btn")
        page.wait_for_function(
            """() => { const m=document.querySelector('[data-testid=\"stMain\"]');
                       return m && (m.innerText||'').includes('以樣搜樣掃描完成'); }""", timeout=300000)
        wait_idle(page)
        queue = page.locator('.st-key-fewshot_queue')
        expect(queue.locator('[data-testid="stImage"]').first).to_be_visible(timeout=30000)
        qtext = queue.inner_text()
        assert "相似" in qtext and any(c in qtext for c in ("scratch", "stain")), \
            f"解耦後佇列仍應含建議類別+相似度:{qtext[:300]}"
        assert _snapshot(ds["root"]) == src_before, "以樣搜樣全程不得改動來源(C6)"
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()
