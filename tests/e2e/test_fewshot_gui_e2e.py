"""M13 第 9 工具「🎯 以樣搜樣」真實 E2E(設計 3_Architect_Design/M13_gui_wiring.md)。

流程:瑕疵偵測①建模(供凍結模型 _dir)→ 切「以樣搜樣」→ ①建樣本集 → ②海掃 → ③匯出 YOLO+CSV。
真實行為:佇列含建議類別+相似度;匯出讀回 labels/*.txt + retrieval_report.csv;來源零寫入(C6)。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from playwright.sync_api import expect

from ._anomaly_wizard import build_model, enter_anomaly, set_model_dir
from .conftest import _add_folder, load_app, wait_idle

pytestmark = pytest.mark.e2e


def _bank_count(page) -> int:
    """讀「① 樣本集」摘要『共 N 顆樣本』的 N(找不到 → -1)。"""
    txt = page.locator('.st-key-fewshot_bank_info').inner_text()
    m = re.search(r"共\s*(\d+)\s*顆", txt)
    return int(m.group(1)) if m else -1


def _switch_fewshot(page):
    page.locator('.st-key-tool_switch').get_by_text("以樣搜樣", exact=False).first.click()
    wait_idle(page)
    page.wait_for_timeout(800)
    main = page.locator('[data-testid="stMain"]').inner_text()
    assert "以樣搜樣" in main, f"切換工具失敗,未進以樣搜樣;實際:{main[:400]}"


def _fewshot_step(page, name, marker):
    """切到某步:反覆點該步選項直到『步驟內文』的 marker 出現(retry 自動處理 segment 誤 deselect)。"""
    seg = page.locator('.st-key-fewshot_step')
    seg.first.wait_for(state="visible", timeout=15000)
    for _ in range(4):
        if marker in page.locator('[data-testid="stMain"]').inner_text():
            page.wait_for_timeout(300)
            return
        seg.get_by_text(name, exact=False).first.click()
        wait_idle(page)
        page.wait_for_timeout(400)
    raise AssertionError(f"切不到步驟「{name}」(內文找不到 {marker})")


def _click(page, key, timeout=30000):
    b = page.locator(f'.st-key-{key} button').first   # rerun 過場可能暫時雙 DOM → 取 first(單按鈕不受影響)
    b.wait_for(state="visible", timeout=timeout)
    page.wait_for_function(
        f"""() => {{ const b=document.querySelector('.st-key-{key} button'); return b && !b.disabled; }}""",
        timeout=timeout)
    b.click()


def _snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*") if p.is_file()}


def test_f1_f2_sample_scan_export(app_server, browser, yolo_defect_at_nmin, tmp_path):
    ds = yolo_defect_at_nmin
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        # 先建一個凍結模型(以樣搜樣復用 anomaly_model._dir)
        enter_anomaly(page, app_server)
        build_model(page, ds["root"], semantic_text="物件類別")
        set_model_dir(page, tmp_path / "mdl")
        page.locator('.st-key-anomaly_save_model_btn button').click()
        wait_idle(page)
        page.wait_for_timeout(1500)   # 讓存模型(寫 bank + 設 _dir)落定

        _switch_fewshot(page)
        # ① 樣本集
        _fewshot_step(page, "樣本集", "建立樣本集")
        _add_folder(page, "fewshot_sample_folder", str(ds["root"]))
        wait_idle(page)
        _fewshot_step(page, "樣本集", "建立樣本集")
        _click(page, "fewshot_build_bank_btn")
        page.wait_for_function(   # 建樣本集要載模型+embed(冷啟動久),等成功或錯誤標記
            """() => { const m=document.querySelector('[data-testid="stMain"]');
                       const t=(m && m.innerText) || '';
                       return t.includes('樣本集已建立') || t.includes('建立樣本集失敗'); }""",
            timeout=240000)
        wait_idle(page)
        info = page.locator('.st-key-fewshot_bank_info').inner_text()
        assert any(c.isdigit() for c in info), f"樣本集資訊應顯示樣本數:{info[:200]}"

        # ② 海掃
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
            f"佇列應含建議類別 + 相似度:{qtext[:300]}"

        # ③ 確認 / 匯出(預設全採納)
        _fewshot_step(page, "確認", "匯出 YOLO")
        out_dir = tmp_path / "fewshot_out"
        inp = page.locator('.st-key-fewshot_out_dir input').first
        inp.wait_for(state="visible", timeout=30000)
        inp.fill(str(out_dir))
        inp.press("Enter")
        wait_idle(page)
        _fewshot_step(page, "確認", "匯出 YOLO")
        src_before = _snapshot(ds["root"])
        _click(page, "fewshot_export_btn")
        page.wait_for_function(
            """() => { const m=document.querySelector('[data-testid=\"stMain\"]');
                       return m && (m.innerText||'').includes('以樣搜樣匯出完成'); }""", timeout=60000)
        wait_idle(page)

        # AC-F2:讀回雙出 + 來源零寫入
        label_files = list((out_dir / "labels").glob("*.txt"))
        assert label_files, f"應匯出 labels/*.txt @ {out_dir}"
        for f in label_files:
            for ln in f.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    assert ln.split()[0] in ("0", "1"), f"cls_id 應為樣本類別索引:{ln}"
        assert (out_dir / "retrieval_report.csv").exists()
        assert (out_dir / "classes.txt").exists()
        csv_head = (out_dir / "retrieval_report.csv").read_text(encoding="utf-8").splitlines()[0]
        assert csv_head.startswith("image_path,obj_index,cx,cy,w,h")
        assert _snapshot(ds["root"]) == src_before, "以樣搜樣匯出不得改動來源(C6)"
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()


def test_f3_no_model_usable(app_server, browser):
    # M14 解耦後:無 anomaly 模型不再是死路——① 樣本集可直接用(預設特徵器)
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        load_app(page, app_server)
        _switch_fewshot(page)
        _fewshot_step(page, "樣本集", "建立樣本集")   # 到得了①步 = 死路已拆(舊版會提前 return)
        expect(page.locator('.st-key-fewshot_build_bank_btn button')).to_be_visible(timeout=15000)
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()


def test_f4_add_to_sample_bank_grows(app_server, browser, yolo_defect_at_nmin, tmp_path):
    # AC-F4:③把已確認的物件加回樣本集 → 樣本集真的長大(N1>N0)、來源零寫入(C6)
    ds = yolo_defect_at_nmin
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)
        build_model(page, ds["root"], semantic_text="物件類別")
        set_model_dir(page, tmp_path / "mdl")
        page.locator('.st-key-anomaly_save_model_btn button').click()
        wait_idle(page)
        page.wait_for_timeout(1500)

        _switch_fewshot(page)
        # ① 建樣本集 → 記 N0
        _fewshot_step(page, "樣本集", "建立樣本集")
        _add_folder(page, "fewshot_sample_folder", str(ds["root"]))
        wait_idle(page)
        _fewshot_step(page, "樣本集", "建立樣本集")
        _click(page, "fewshot_build_bank_btn")
        page.wait_for_function(
            """() => { const m=document.querySelector('[data-testid="stMain"]');
                       const t=(m && m.innerText) || '';
                       return t.includes('樣本集已建立') || t.includes('建立樣本集失敗'); }""",
            timeout=240000)
        wait_idle(page)
        n0 = _bank_count(page)
        assert n0 > 0, f"樣本集應有初始樣本數:{n0}"

        # ② 海掃
        _fewshot_step(page, "海掃", "相似度門檻")
        _add_folder(page, "fewshot_target_folder", str(ds["root"]))
        wait_idle(page)
        _fewshot_step(page, "海掃", "相似度門檻")
        _click(page, "fewshot_scan_btn")
        page.wait_for_function(
            """() => { const m=document.querySelector('[data-testid=\"stMain\"]');
                       return m && (m.innerText||'').includes('以樣搜樣掃描完成'); }""", timeout=300000)
        wait_idle(page)

        # ③ 確認 → 預設全採納 → ➕ 加入樣本集(真實 re-embed + append)
        _fewshot_step(page, "確認", "加入樣本集")
        src_before = _snapshot(ds["root"])
        _click(page, "fewshot_add_to_bank_btn")
        page.wait_for_function(
            """() => { const m=document.querySelector('[data-testid=\"stMain\"]');
                       return m && (m.innerText||'').includes('已加入'); }""", timeout=120000)
        wait_idle(page)

        # 回 ① 看樣本集摘要 → N1 > N0(真的長大、摘要同步)
        _fewshot_step(page, "樣本集", "建立樣本集")
        n1 = _bank_count(page)
        assert n1 > n0, f"加入樣本集後樣本數應增長:N0={n0} → N1={n1}"
        assert _snapshot(ds["root"]) == src_before, "加入樣本集不得改動來源(C6)"
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    finally:
        ctx.close()
