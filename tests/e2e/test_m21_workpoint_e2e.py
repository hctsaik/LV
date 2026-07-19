"""M21 GUI 接線真實 E2E:⚖️ 做法比較(A/B)+ 💰 成本試算(what-if)。

設計:3_Architect_Design/M21_gui_wiring.md §5(2026-07-19 反向閘門修訂版)。
G1 方向性由單元層 26+28 純函式鏈鎖死(test_run_pairing.test_ac7);
本檔驗:接線真實(AC-E1)、null 方向 deterministic(AC-E2)、成本曲線反應性
(AC-E3)、零副作用(AC-E4)、門檻誠實(AC-E5)、可達性(AC-E6)。

真實 Streamlit + Playwright + 真 DINOv2;資料=synthetic_yolo_dataset
(20 normal + 5 defect)。流程沿 test_anomaly_tool_e2e.test_confirm_normal_reapply
先例:①建模 → ②套用 → autoseed → 🔁 就地重評。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from playwright.sync_api import expect

from ._anomaly_wizard import TAB_APPLY, apply_model, build_model, click_tab, enter_anomaly
from .conftest import wait_idle

pytestmark = pytest.mark.e2e


def _no_exception(page) -> None:
    assert page.locator('[data-testid="stException"]').count() == 0, \
        page.locator('[data-testid="stException"]').first.inner_text()[:400]


def _folder_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _open_expander(page, anchor: str) -> None:
    """點 summary 真開 expander(冪等:已開就不再點,避免 toggle 關)。

    ⚠ 不可用 JS `d.open=true` 假開:React 不知道 → 整個 details 攔截 pointer、
    fill 的值會被 React 回寫(mini_num_drive.py 最小重現證實;pitfalls 新條目)。
    只有 summary 真點,expander 內的 fill/click 互動才有效。open attr 由 React
    render 維護,只要全檔不混用 JS-open,拿它判斷「已開」可信。"""
    summ = page.locator("details summary", has_text=anchor).first
    summ.wait_for(state="visible", timeout=15000)
    is_open = page.evaluate(
        """(t) => {
            const ds = Array.from(document.querySelectorAll('details'));
            const d = ds.find(x => { const s = x.querySelector('summary');
                                     return s && s.textContent.includes(t); });
            return d ? d.open : false;
        }""", anchor)
    if not is_open:
        summ.click()
    page.wait_for_timeout(400)


def _click_key_button(page, key: str) -> None:
    """點 .st-key-<key> button(等 enabled;JS click 穿 overlay)。"""
    page.wait_for_function(
        f"""() => {{ const b = document.querySelector('.st-key-{key} button');
                     return b && !b.disabled && b.offsetParent !== null; }}""",
        timeout=30000)
    page.evaluate(f"document.querySelector('.st-key-{key} button').click()")
    page.wait_for_timeout(600)
    wait_idle(page, timeout=120000)
    click_tab(page, TAB_APPLY)   # on_click rerun 會把 st.tabs 彈回① → 切回②


def _card_text(page) -> str:
    el = page.locator('.st-key-rcmp_card')
    el.first.wait_for(state="visible", timeout=30000)
    return el.first.inner_text()


_LIVE_YTICKS_JS = """() => {
    const ws = Array.from(document.querySelectorAll('.st-key-rcost_curve_wrap'))
        .filter(w => w.offsetParent !== null);      // 排除 ghost 副本(pitfalls #11)
    const w = ws[ws.length - 1];
    if (!w) return '';
    return Array.from(w.querySelectorAll('.ytick text, g.ytick text'))
                .map(t => t.textContent).join('|');
}"""


def _ytick_text(page) -> str:
    return page.evaluate(_LIVE_YTICKS_JS)


def test_m21_workpoint_flow(app_server, browser, synthetic_yolo_dataset):
    ds = synthetic_yolo_dataset
    digest_before = _folder_digest(ds["root"])            # AC-E4 零寫入基準
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    try:
        # ── 前置:①建模 → ②套用(AC-E6:全程停在瑕疵偵測工具內)──
        enter_anomaly(page, app_server)
        build_model(page, ds["root"])
        apply_model(page, ds["root"])
        _no_exception(page)

        # ── AC-E1:pin B(未校準)→ autoseed 12 正常 → 就地重評 → pin A → 比較 ──
        _open_expander(page, "做法比較")
        # 契約:未 pin 的槽顯示「(未存)」→ 初始 A/B 兩處,pin B 後剩 1 處
        assert page.locator('[data-testid="stMain"]').inner_text().count("(未存)") == 2
        _click_key_button(page, "rcmp_pin_b")
        _open_expander(page, "做法比較")
        assert page.locator('[data-testid="stMain"]').inner_text().count("(未存)") == 1

        _click_key_button(page, "anomaly_autoseed_normal")
        expect(page.locator('[data-testid="stMain"]')).to_contain_text("已標記正常")
        page.wait_for_function(
            """() => { const bs = Array.from(document.querySelectorAll(
                          '.st-key-anomaly_reapply_btn button'))
                          .filter(b => b.offsetParent !== null);
                       return bs.length === 1; }""", timeout=30000)
        _click_key_button(page, "anomaly_reapply_btn")
        page.locator('[class*="st-key-anomaly_scatter_apply"]').first.wait_for(
            state="visible", timeout=180000)
        wait_idle(page, timeout=120000)
        _no_exception(page)

        _open_expander(page, "做法比較")
        _click_key_button(page, "rcmp_pin_a")
        _open_expander(page, "做法比較")
        _click_key_button(page, "rcmp_go")
        _open_expander(page, "做法比較")
        card = _card_text(page)
        # AC-E1:三值結論之一 + 預設不分組明講;統計細節在「詳細」內
        assert any(k in card for k in ("A 較好", "B 較好", "看不出來")), card
        assert "未考慮批次" in card, card
        _open_expander(page, "詳細")
        detail = _card_text(page)
        assert "exploratory" in detail and "p=" in detail, detail
        _no_exception(page)

        # ── AC-E4 錨點(生效結果快照;compare/試算全程不得改變)──
        cart_btn = page.locator('.st-key-anomaly_select_outliers button')
        state_anchor = cart_btn.inner_text()

        # ── AC-E2:同結果 pin A+pin B → null 方向 deterministic ──
        _click_key_button(page, "rcmp_pin_a")
        _open_expander(page, "做法比較")
        _click_key_button(page, "rcmp_pin_b")
        _open_expander(page, "做法比較")
        _click_key_button(page, "rcmp_go")
        _open_expander(page, "做法比較")
        card2 = _card_text(page)
        assert "看不出來" in card2, card2
        assert "無差異" in card2, card2
        _no_exception(page)

        # ── AC-E3:整圖框選標瑕疵 + autoseed 覆寫正常 → 兩類 ≥5 → 成本曲線 ──
        wrap = page.locator('[class*="st-key-anomaly_scatter_apply"]').first
        wrap.scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        wrap.locator('.modebar-btn[data-title="Box Select"]').first.click()
        page.wait_for_timeout(300)
        drag = wrap.locator('.nsewdrag').first
        bb = drag.bounding_box()
        assert bb is not None
        page.mouse.move(bb["x"] + 2, bb["y"] + 2)
        page.mouse.down()
        page.mouse.move(bb["x"] + bb["width"] - 2, bb["y"] + bb["height"] - 2, steps=10)
        page.mouse.up()
        page.wait_for_timeout(1200)
        wait_idle(page, timeout=60000)
        click_tab(page, TAB_APPLY)
        _click_key_button(page, "anomaly_mark_bad_apply")       # 全部標瑕疵
        expect(page.locator('[data-testid="stMain"]')).to_contain_text("已標記")
        _click_key_button(page, "anomaly_autoseed_normal")      # 最不可疑 12 → 覆寫 good
        wait_idle(page, timeout=60000)

        _open_expander(page, "成本試算")
        page.wait_for_function(
            """() => document.body.innerText.includes('建議工作點')""", timeout=60000)
        assert "僅試算" in page.locator('[data-testid="stMain"]').inner_text()
        page.locator('.st-key-rcost_curve_wrap svg').first.wait_for(
            state="visible", timeout=30000)
        yticks_50 = _ytick_text(page)
        assert yticks_50, "成本曲線 y 軸刻度應存在"
        # 改成本比 50 → 1:曲線尺度必變(argmin exact 移動在 27-AC2,不重驗)
        # ⚠ 陷阱鏈(依序踩過):open details 攔 pointer(click/force 都打到 overlay)
        # → synthetic dispatchEvent 不觸發 commit(DOM value=假訊號)→ JS focus+
        # 鍵盤也進不去。正解=fill+Enter:fill **不做 pointer hit-test**(只檢
        # visible/enabled/editable),是 repo 驅動 Streamlit input 的標準招
        # (_add_folder 同款);第一輪失敗其實是前面的 click 被攔,錯怪了 fill。
        inp = page.locator('.st-key-rcost_ratio input:visible').last
        inp.fill("1")
        inp.press("Enter")
        page.wait_for_timeout(800)
        wait_idle(page, timeout=60000)
        click_tab(page, TAB_APPLY)
        _open_expander(page, "成本試算")
        try:
            page.wait_for_function(
                f"""(prev) => {{ const t = ({_LIVE_YTICKS_JS})();
                                 return t && t !== prev; }}""",
                arg=yticks_50, timeout=30000)
        except Exception:
            diag = page.evaluate(
                """() => {
                    const ws = Array.from(
                        document.querySelectorAll('.st-key-rcost_curve_wrap'));
                    const inps = Array.from(
                        document.querySelectorAll('.st-key-rcost_ratio input'));
                    return {
                        wraps: ws.map(w => ({vis: !!w.offsetParent,
                            ticks: Array.from(w.querySelectorAll('.ytick text'))
                                        .map(t => t.textContent).join('|').slice(0, 80)})),
                        inputs: inps.map(i => ({vis: !!i.offsetParent, val: i.value})),
                        sugg: (document.body.innerText.match(/建議工作點[^\\n]*/g) || []),
                    };
                }""")
            print("DIAG-E3:", diag)
            print("DIAG-E3-yticks_50:", yticks_50[:120])
            raise
        _no_exception(page)

        # ── AC-E4:比較/試算全程沒動生效結果與資料夾 ──
        click_tab(page, TAB_APPLY)
        assert page.locator('.st-key-anomaly_select_outliers button').inner_text() \
            == state_anchor, "成本試算/比較不得改變②的判定結果(G4)"
        assert _folder_digest(ds["root"]) == digest_before, \
            "目標資料夾必須零寫入(G4)"

        # ── AC-E5:清除 confirmed → 兩功能顯示門檻指引,不 crash ──
        _click_key_button(page, "anomaly_clear_confirmed")
        _open_expander(page, "做法比較")
        _click_key_button(page, "rcmp_go")
        _open_expander(page, "做法比較")
        assert "請先" in _card_text(page)
        _open_expander(page, "成本試算")
        expect(page.locator('[data-testid="stMain"]')).to_contain_text("已確認樣本不足")
        _no_exception(page)
    finally:
        ctx.close()
