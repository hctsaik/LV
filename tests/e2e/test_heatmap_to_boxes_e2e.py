"""E2E 真實行為驗收:13 heatmap_to_boxes GUI 接線(AC-G1..G3)。

設計:3_Architect_Design/13_heatmap_to_boxes.md

**AC-G1 是本檔的防 false-green 核心**:不是驗「有產出 txt」,而是驗「產出的框**真的框在
植入的瑕疵上**」——用真實 DINOv2 跑完整 patch 管線,再拿產出的 YOLO 框跟 fixture 裡
已知的瑕疵區算 IoU。

**校準註記(實測)**:`quantile` 預設 0.98 只保留最熱的 2% patch。本 fixture 的瑕疵佔物件約
**25%** 面積,故 0.98 下框雖然**落在瑕疵內(定位正確)**,卻只框到最熱的一小塊 → 對整塊瑕疵區的
IoU 僅 0.116,**達不到設計釘死的 0.3**;把分位調到 0.90(仍在設計允許的 0.90–0.999 內)→ IoU 0.535。
「瑕疵佔比越大、分位要調越低」是真實的物理關係,已寫進 slider 的 help。故本測**明確把分位
拉到 0.90**,並額外驗「良品 conf ≪ 瑕疵 conf」(證明訊號真的來自異常,不是到處亂框)。

E2E 教訓(踩過,寫下來免得重踩):
  · 成功訊號不能用「主畫面含『產出』」—— 面板說明文字本身就有「**產出**與模型預測完全同形…」,
    會在還沒按之前就判定成功(假綠 + 重試迴圈失效)。改用只有成功後才 render 的元素。
  · popover 用 Escape 收合,overlay 仍攔住後續點擊 → 按鈕點擊變靜默 no-op。要再點一次 trigger 收合。
  · BaseWeb slider **不吃 Home/End**;要用滑鼠拖曳。且設完必須驗值,否則會拿著預設值跑出低 IoU。
  · 側欄收合時 `innerText` 回空字串 → 找選項要用 `textContent`。
  · radio 有 format_func → 畫面顯示的是「物件級(快,粗)」而不是 'object'。
  · 不要叫 `_ensure_sidebar`:它依賴「重開側欄」鈕存在,側欄本來就開著時會 timeout。
    一律用 `state="attached"` 選擇器 + JS click,不受側欄收合影響。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from playwright.sync_api import expect

from ._anomaly_wizard import TAB_APPLY, apply_model, build_model, click_tab, enter_anomaly
from .conftest import wait_idle

pytestmark = pytest.mark.e2e
expect.set_options(timeout=20000)

_OUT_RE = re.compile(r"產出\s*(\d+)\s*張\s*/\s*(\d+)\s*框\s*→\s*(\S+)")
_PRED_BTN = '.st-key-anomaly_make_pred button'


def _wait_pred_btn(page, want_enabled: bool, timeout: int = 30000) -> None:
    """rerun 會在 DOM 留下 stale 的按鈕副本 → 不能用 strict locator,改輪詢可見副本的狀態。"""
    page.wait_for_function(
        """([sel, we]) => {
            const bs = [...document.querySelectorAll(sel)].filter(b => b.offsetParent !== null);
            if (!bs.length) return false;
            return we ? bs.some(b => !b.disabled) : bs.every(b => b.disabled);
        }""", arg=[_PRED_BTN, want_enabled], timeout=timeout)


def _quantile_now(page) -> float | None:
    v = page.evaluate(
        """() => { const s = [...document.querySelectorAll(
                       '.st-key-anomaly_pred_q [role="slider"]')]
                     .find(e => e.offsetParent !== null);
                   return s ? s.getAttribute('aria-valuenow') : null; }""")
    return float(v) if v is not None else None


def _set_quantile_to_min(page) -> None:
    """把「參數」popover 內的 quantile slider 拖到**最小值 0.900**(見檔頭校準註記)。"""
    panel = page.locator('.st-key-anomaly_pred_panel')
    panel.get_by_text("參數", exact=True).first.click()          # 開
    wait_idle(page)
    handle = page.locator('.st-key-anomaly_pred_q [role="slider"]:visible').first
    handle.wait_for(state="visible", timeout=15000)
    box = handle.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] - 400, box["y"] + box["height"] / 2, steps=12)   # 越過左端 → clamp
    page.mouse.up()
    wait_idle(page)
    got = _quantile_now(page)
    assert got is not None and abs(got - 0.900) < 1e-6, f"quantile slider 沒到 0.900,實得 {got}"

    panel.get_by_text("參數", exact=True).first.click()          # 收(再點 trigger,不用 Escape)
    wait_idle(page)
    page.wait_for_function(
        """() => ![...document.querySelectorAll('.st-key-anomaly_pred_q [role="slider"]')]
                   .some(e => e.offsetParent !== null)""", timeout=15000)


def _generate_preds(page) -> tuple[int, int, Path]:
    """按「⚡ 產生預標」→ 解析成功訊息回 (n_images, n_boxes, out_root)。"""
    for _ in range(3):
        _wait_pred_btn(page, want_enabled=True)
        page.locator(f'{_PRED_BTN}:visible:enabled').first.click()
        wait_idle(page, timeout=240000)
        try:    # 只有成功後才會 render 的元素 = 唯一可靠的成功訊號
            page.locator('.st-key-anomaly_pred_to_viz button:visible').first.wait_for(
                state="visible", timeout=90000)
            break
        except Exception:  # noqa: BLE001  點到 stale 副本 → 再點一次
            page.wait_for_timeout(800)
    main = page.locator('[data-testid="stMain"]').inner_text()
    m = _OUT_RE.search(main)
    if not m:   # 把畫面上真正說的話帶進失敗訊息,不要只說「沒看到」
        exc = page.locator('[data-testid="stException"]').all_inner_texts()
        panel = page.locator('.st-key-anomaly_pred_panel:visible').all_inner_texts()
        raise AssertionError(
            "產生預標後必須顯示「產出 N 張 / M 框 → <路徑>」。\n"
            f"panel={panel}\nexception={exc}\nmain(head)={main[:1200]}")
    return int(m.group(1)), int(m.group(2)), Path(m.group(3))


def _read_boxes(out_root: Path) -> dict[str, list[tuple]]:
    """讀回產出的 YOLO 6 欄預標 → {stem: [(cid,cx,cy,w,h,conf), …]}。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from interaction import parse_yolo_boxes_conf
    return {p.stem: parse_yolo_boxes_conf(p)
            for p in sorted((out_root / "labels").glob("*.txt"))}


def test_pred_localises_defect_e2e(app_server, browser, synthetic_yolo_dataset):
    """AC-G1(真實行為,防 false-green):產出的框真的框在植入的瑕疵區上。

    三件事缺一不可:
      ① 每張瑕疵圖至少一框與植入瑕疵區 **IoU ≥ 0.3**(定位真的成立);
      ② 瑕疵框的 conf **明顯高於**良品框(訊號真的來自異常,不是到處亂框);
      ③ classes.txt == "defect"(單類語義)。
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from gt_pred_diff import iou_cxcywh          # 測試層 import(產品碼不循環)

    ds = synthetic_yolo_dataset
    x0, y0, x1, y1 = ds["defect_box_norm"]       # 角點 → cxcywh
    planted = ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)

    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    try:
        enter_anomaly(page, app_server)
        build_model(page, ds["root"])            # 側欄預設 score_mode=patch
        apply_model(page, ds["root"])
        click_tab(page, TAB_APPLY)
        _wait_pred_btn(page, want_enabled=True)  # patch 模式下不該反灰
        _set_quantile_to_min(page)               # → 0.900,見檔頭校準註記
        _n_img, n_box, out_root = _generate_preds(page)
        assert n_box > 0, "patch 模式下必須產出框"

        boxes = _read_boxes(out_root)
        assert (out_root / "classes.txt").read_text(encoding="utf-8").strip() == "defect"

        defect_confs, normal_confs = [], []
        for stem, bs in boxes.items():
            if stem.startswith("defect"):
                best = max((iou_cxcywh(b[1:5], planted) for b in bs), default=0.0)
                assert best >= 0.3, (
                    f"{stem}: 產出的框沒框到植入的瑕疵區(best IoU={best:.3f} < 0.3)。"
                    f"框={bs} 植入={planted}")
                defect_confs += [b[5] for b in bs if b[5] is not None]
            elif stem.startswith("normal"):
                normal_confs += [b[5] for b in bs if b[5] is not None]

        assert defect_confs, "瑕疵圖必須有框"
        if normal_confs:    # ② 訊號真的來自異常:最弱的瑕疵框也要比最強的良品框有信心
            assert min(defect_confs) > max(normal_confs), (
                f"瑕疵框 conf({min(defect_confs):.3f}) 必須明顯高於良品框"
                f"({max(normal_confs):.3f}),否則框只是到處亂長")
    finally:
        ctx.close()


def test_pred_feeds_pipeline_e2e(app_server, browser, synthetic_yolo_dataset):
    """AC-G2(餵回管線):按「➡ 填入 Visualize 的模型預測資料夾」→ Visualize 側欄的
    模型預測資料夾真的帶上 out_root(13 的產物可直接當 10 的輸入)。

    AC-G2 的後半「挖錯面板能以此路徑跑出佇列」需要先跑一次 Visualize Run;那是
    gt_pred_diff AC-G1(test_queues_are_real_e2e)用正規 GT+pred fixture 驗的**同一條程式路徑**,
    此處不重複跑一次昂貴的 embedding。本測負責的是**跨工具交棒**這一段。
    """
    ds = synthetic_yolo_dataset
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    try:
        enter_anomaly(page, app_server)
        build_model(page, ds["root"])
        apply_model(page, ds["root"])
        click_tab(page, TAB_APPLY)
        _n, _b, out_root = _generate_preds(page)

        # 點擊後驗證後果並重試(stale 副本會靜默 no-op)。**硬斷言** —— 靜默吞掉失敗會讓
        # 下游的「值是空的」變成無法歸因的謎題(踩過)。
        filled = False
        for _ in range(3):
            page.locator('.st-key-anomaly_pred_to_viz button:visible').first.click()
            wait_idle(page)
            try:
                page.wait_for_function(
                    """() => (document.querySelector('[data-testid="stMain"]')?.innerText || '')
                                .includes('已填入')""", timeout=15000)
                filled = True
                break
            except Exception:  # noqa: BLE001
                page.wait_for_timeout(600)
        assert filled, "「➡ 填入 Visualize 的模型預測資料夾」點了沒生效(沒出現「已填入」確認)"

        # 切回 Visualize → Object Detector(預測資料夾欄位只在偵測模式出現)
        page.locator('.st-key-tool_switch').get_by_text("Visualize").first.click()
        wait_idle(page, timeout=60000)
        page.wait_for_selector('.st-key-viz_mode', state="attached", timeout=60000)
        assert page.evaluate("""() => {
            const root = document.querySelector('.st-key-viz_mode');
            if (!root) return false;
            const opt = [...root.querySelectorAll('label')]
                .filter(l => l.querySelector('input[type="radio"]'))
                .find(l => (l.textContent || '').includes('Object Detector'));
            if (!opt) return false;
            if (!opt.querySelector('input[type="radio"]').checked) opt.click();
            return true;
        }"""), "找不到 Visualize 的 Object Detector 模式選項"
        wait_idle(page, timeout=60000)
        page.wait_for_selector('.st-key-viz_pred_root_input input', state="attached",
                               timeout=60000)

        val = page.locator('.st-key-viz_pred_root_input input').first.input_value()
        assert Path(val) == out_root, f"模型預測資料夾應為 {out_root},實得 {val}"
    finally:
        ctx.close()


def test_object_mode_disabled_and_honest_caption_e2e(app_server, browser,
                                                     synthetic_yolo_dataset):
    """AC-G3:object 模式沒有 patch 熱力圖 → 按鈕反灰 + 理由點名 patch;
    且**誠實界線文案**(patch 網格 / 單一類別 defect / 瑕疵限定)真的出現在 UI,
    不是只寫在設計文件裡。"""
    ds = synthetic_yolo_dataset
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    try:
        enter_anomaly(page, app_server)
        page.wait_for_selector('.st-key-anomaly_score_mode', state="attached", timeout=30000)
        # ⚠ 兩個坑:① radio 有 format_func → 顯示的是「物件級(快,粗)」不是 'object';
        #          ② 側欄收合時 innerText 回空字串 → 要用 textContent。
        assert page.evaluate("""() => {
            const root = document.querySelector('.st-key-anomaly_score_mode');
            if (!root) return false;
            const opt = [...root.querySelectorAll('label')]
                .filter(l => l.querySelector('input[type="radio"]'))
                .find(l => (l.textContent || '').includes('物件級'));
            if (!opt) return false;
            if (!opt.querySelector('input[type="radio"]').checked) opt.click();
            return true;
        }"""), "找不到側欄「分數依據」的『物件級』選項"
        wait_idle(page)
        checked = page.evaluate("""() => [...document.querySelectorAll(
            '.st-key-anomaly_score_mode label')]
              .filter(l => l.querySelector('input[type="radio"]:checked'))
              .map(l => l.textContent).join('|')""")
        assert "物件級" in checked, f"分數依據沒切到物件級(實際:{checked})"

        build_model(page, ds["root"])
        apply_model(page, ds["root"])
        click_tab(page, TAB_APPLY)

        _wait_pred_btn(page, want_enabled=False)   # object 模式無 pmap → 必須反灰
        # 面板文字要**等它真的畫出來**再讀:滿載下 inner_text() 可能讀到還沒 render 完的空字串
        # (讀一次就斷言 → 假失敗。踩過:單跑綠、全套紅)。
        page.wait_for_function(
            """() => { const p = [...document.querySelectorAll('.st-key-anomaly_pred_panel')]
                          .find(e => e.offsetParent !== null);
                       return !!p && (p.innerText || '').includes('patch'); }""",
            timeout=30000)
        panel = page.locator('.st-key-anomaly_pred_panel:visible').first.inner_text()
        tip = page.locator(f'{_PRED_BTN}:visible').first.get_attribute("title") or ""
        assert "patch" in (panel + tip), "反灰理由必須點名 patch 模式"
        for phrase in ("patch", "defect", "瑕疵"):   # 誠實界線真的在畫面上
            assert phrase in panel, f"誠實界線文案缺「{phrase}」:{panel[:200]}"
    finally:
        ctx.close()
