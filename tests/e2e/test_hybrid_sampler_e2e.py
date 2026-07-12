"""E2E:11 hybrid_sampler GUI 接線(AC-G1..G3)。

設計:3_Architect_Design/11_hybrid_sampler.md

驗「混合(不確定×多樣)」策略在真實 Visualize 選樣面板可用、接得上 08 池帳本(已送標的不再選)
與 09 回合帳本(策略名帶 hybrid:)。

選樣面板在 `@st.fragment`(_render_right_panel)內 —— 只做 fragment 內部互動(切面板 / 按選樣鈕)
就不會產生 stale 副本(同 _al_loop_harness 對 test_x_diversity 的觀察)。
"""
from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from ._al_loop_harness import (cart_send_count, goto_cart, goto_readback,
                               read_jsonl, run_viz, sample_to_cart, send_cart)
from .conftest import load_app, wait_idle

pytestmark = pytest.mark.e2e
expect.set_options(timeout=20000)

_HYBRID = "混合（不確定×多樣）"


def _open_sampling(page) -> None:
    page.locator('.st-key-viz_panel_view').get_by_text("選樣", exact=True).click()
    wait_idle(page)


def _pick_hybrid(page) -> None:
    """選樣策略 radio 切到「混合」。radio 的**群組問題 label** 也含這串字 →
    只點真正含 input[type=radio] 的 option label(誤點群組 label = 靜默 no-op)。"""
    page.wait_for_selector('.st-key-viz_samp_strategy', state="attached", timeout=30000)
    assert page.evaluate("""(want) => {
        const root = document.querySelector('.st-key-viz_samp_strategy');
        if (!root) return false;
        const opt = [...root.querySelectorAll('label')]
            .filter(l => l.querySelector('input[type="radio"]'))
            .find(l => (l.textContent || '').includes('混合'));
        if (!opt) return false;
        if (!opt.querySelector('input[type="radio"]').checked) opt.click();
        return true;
    }""", _HYBRID), "找不到「混合（不確定×多樣）」選樣策略"
    wait_idle(page)
    checked = page.evaluate("""() => [...document.querySelectorAll(
        '.st-key-viz_samp_strategy label')]
          .filter(l => l.querySelector('input[type="radio"]:checked'))
          .map(l => l.textContent).join('|')""")
    assert "混合" in checked, f"策略沒切到混合(實際:{checked})"


def _set_k(page, k: int) -> None:
    inp = page.locator('.st-key-viz_sampling_n input').first
    inp.click()
    inp.fill(str(k))
    inp.press("Tab")          # blur → Streamlit 才會 commit widget 值
    wait_idle(page)


def _run_pick(page) -> None:
    """按「🎯 挑選樣本」並等結果真的畫出來(wait_idle 回來時 fragment 可能還沒 render)。"""
    page.locator('.st-key-viz_sampling_btn button').first.click()
    wait_idle(page, timeout=120000)
    page.wait_for_function(
        """() => { const t = document.querySelector('[data-testid="stMain"]')?.innerText || '';
                   return t.includes('選出') || t.includes('沒有可挑選') || t.includes('沒有'); }""",
        timeout=60000)


def _picked_count(page) -> int:
    """選樣結果的張數,讀面板 caption「選出 N 張」(比數 DOM 節點穩)。"""
    body = page.locator('[data-testid="stMain"]').inner_text()
    m = re.search(r"選出\s*(\d+)\s*張", body)
    assert m, f"選樣後必須顯示「選出 N 張」,實際:{body[:500]}"
    return int(m.group(1))


def test_hybrid_strategy_runs_e2e(al_isolated_server, page, synthetic_dataset):
    """AC-G1:選「混合」、K=6 → 畫廊恰 6 張,且 caption 出現「近重複對」數字
    (把「比純 top-K 少了多少近重複」攤在使用者眼前,是產品承諾,不是裝飾)。"""
    srv = al_isolated_server
    load_app(page, srv.base_url)
    run_viz(page, synthetic_dataset)
    _open_sampling(page)
    _pick_hybrid(page)
    _set_k(page, 6)
    _run_pick(page)

    assert _picked_count(page) == 6, "混合選樣要正好回 K=6 張"
    body = page.locator('[data-testid="stMain"]').inner_text()
    m = re.search(r"本批近重複對：\*{0,2}(\d+)", body) or re.search(r"本批近重複對：(\d+)", body)
    assert m, f"必須顯示「本批近重複對」數字,實際主畫面:{body[:400]}"
    assert re.search(r"純 top-K 不確定會是 \d+ 對", body), "必須同時給出純 top-K 的對照數字"


def test_hybrid_excludes_handled_e2e(al_isolated_server, page, synthetic_dataset):
    """AC-G2(接 08 池帳本):已送標的樣本預設不再被選;取消勾選「剔除已送標」→ 回到可入選。

    先走一次真實送標(cart 路徑)把一批樣本記進 pool_registry,再用混合選樣驗它們被剔除。
    """
    srv = al_isolated_server
    load_app(page, srv.base_url)
    run_viz(page, synthetic_dataset)

    # 1) 先真實送一批(mark_sent 進 registry)。sample_to_cart = 切選樣面板 → 挑 → 全部加入清單
    #    (「加入清單」鈕要挑過一次才會 render —— 直接點會 timeout,踩過)。
    sample_to_cart(page)
    goto_cart(page)
    n_sent = cart_send_count(page)
    assert n_sent >= 2
    send_cart(page)
    sent_shas = {e["sha256"] for e in read_jsonl(srv.registry_path) if e.get("sha256")}
    assert sent_shas, "送標必須把 sha 記進 pool registry"

    # 送標後主畫面被「已送 N 張到 Labeling…」的確認取代 → 重新載入 + 重跑(embedding 快取是熱的),
    # 這也正是真實使用者標完回來的樣子。pool registry 在磁碟上,剔除依然生效。
    load_app(page, srv.base_url)
    run_viz(page, synthetic_dataset)

    # 2) 混合選樣(預設剔除已送標)
    _open_sampling(page)
    _pick_hybrid(page)
    _set_k(page, 6)
    _run_pick(page)
    page.wait_for_function(
        """() => /已剔除 \\d+ 張已送標／已標/.test(
                   document.querySelector('[data-testid="stMain"]')?.innerText || '')""",
        timeout=30000)

    # 3) 取消勾選 → 剔除數字消失(回到可入選)。用 key 定位(label 文字也出現在 help 裡);
    #    JS click 避開「元素在視窗外」的 actionability 失敗。
    assert page.evaluate("""() => {
        const inp = document.querySelector('.st-key-viz_samp_skip_handled input');
        if (!inp) return false;
        if (inp.checked) inp.click();
        return true;
    }"""), "找不到「剔除已送標／已標的樣本」checkbox"
    wait_idle(page)
    _run_pick(page)
    page.wait_for_function(
        """() => !/已剔除 \\d+ 張已送標／已標/.test(
                    document.querySelector('[data-testid="stMain"]')?.innerText || '')""",
        timeout=30000)


def test_hybrid_send_opens_round_e2e(al_isolated_server, page, synthetic_dataset):
    """AC-G3(接 09 回合帳本):從混合選樣送標 → 回合的策略名含 `hybrid:`。"""
    srv = al_isolated_server
    load_app(page, srv.base_url)
    run_viz(page, synthetic_dataset)
    _open_sampling(page)
    _pick_hybrid(page)
    _set_k(page, 6)
    _run_pick(page)

    # 選樣面板自己的送標鈕(strategy=hybrid:<來源>);點擊後驗證後果並重試
    for _ in range(4):
        btn = page.locator('.st-key-viz_sampling_to_lbl button:visible:enabled')
        if btn.count():
            btn.first.click()
            wait_idle(page, timeout=60000)
        starts = [r for r in read_jsonl(srv.rounds_path) if r.get("kind") == "start"]
        if starts:
            break
        page.wait_for_timeout(800)

    starts = [r for r in read_jsonl(srv.rounds_path) if r.get("kind") == "start"]
    assert starts, "送標必須開一個回合"
    assert starts[-1]["strategy"].startswith("hybrid:"), (
        f"回合策略名必須帶 hybrid:,實得 {starts[-1]['strategy']!r}")

    goto_readback(page)
    expect(page.get_by_text(re.compile("策略：hybrid:")).first).to_be_visible()
