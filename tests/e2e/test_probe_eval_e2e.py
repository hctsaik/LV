"""E2E:12 probe_eval GUI 接線(AC-G1..G3)。

設計:3_Architect_Design/12_probe_eval.md

驗「🧪 跑代理評估」按鈕真的把 probe 指標掛進回合帳本、曲線 + 停止建議渲染、
以及類別不足時**誠實 warning**(不入帳、不崩潰)。

**產品紀律(不可違約)**:不做自動停 —— plateau 只給建議。UI 必須常駐
「代理指標僅供方向…是否停止由你決定」,本檔驗它真的在畫面上。

AC-G2 的設計不一致(已與 architect 對齊,見 4_PM_Feedback/probe_eval.md):
設計 §5 寫「兩個回合 → 建議 improving/plateau 其一」,但 §2.1 的 `plateau_advice(min_points=3)`
與 AC5 明定「2 點 → insufficient」。以 **core 契約(AC5)為準** → 本檔用 **3 個回合**驗
improving/plateau 附數字。
"""
from __future__ import annotations

import json
import re

import pytest
from playwright.sync_api import expect

from ._al_loop_harness import (cart_send_count, goto_cart, goto_readback,
                               read_jsonl, run_viz, sample_to_cart, send_cart)
from .conftest import load_app, wait_idle

pytestmark = pytest.mark.e2e
expect.set_options(timeout=20000)


def _seed_rounds(srv, accs: list[float]) -> list[str]:
    """直接把帶 probe_acc 的回合寫進隔離帳本(不必真的跑 N 輪標註迴圈)。

    ⚠ 每回合 `n_labeled=100` 是**刻意的**:`plateau_advice` 的斜率是「每 **100 張**標註的指標
    增量」= (y[-1]-y[-2]) / (x[-1]-x[-2]) * 100。用 n_labeled=10 的話 x=[10,20,30],
    同樣的 y 會算出 0.02(improving)而不是 0.002(plateau) —— 曲線的 x 是**累積標註量**,不是回合序。
    """
    rows, rids = [], []
    for i, acc in enumerate(accs, start=1):
        rid = f"rSEED{i}"
        rids.append(rid)
        ts = f"2026-07-12T00:00:0{i}+00:00"
        rows += [
            {"kind": "start", "round_id": rid, "ts": ts, "strategy": "cart",
             "params": {}, "tool": "cart", "note": ""},
            {"kind": "batch", "round_id": rid, "batch_id": f"b{i}", "n_sent": 100},
            {"kind": "readback", "round_id": rid, "batch_id": f"b{i}",
             "n_labeled": 100, "n_changed": 20},
            {"kind": "metric", "round_id": rid, "name": "probe_acc", "value": acc,
             "source": "probe", "ts": ts},
        ]
    srv.rounds_path.parent.mkdir(parents=True, exist_ok=True)
    srv.rounds_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in rows) + "\n",
        encoding="utf-8")
    return rids


def _click_probe(page, rid: str, expect_text: str) -> None:
    """按某回合卡的「🧪 跑代理評估」,並確認它真的跑了(rerun 可能留 stale 副本 → 靜默 no-op)。"""
    sel = f'.st-key-al_probe_{rid} button'
    page.wait_for_selector(sel, state="attached", timeout=30000)
    for _ in range(4):
        page.locator(f'{sel}:visible').first.click()
        wait_idle(page, timeout=120000)
        try:
            page.wait_for_function(
                """(t) => (document.querySelector('[data-testid="stMain"]')?.innerText || '')
                            .includes(t)""", arg=expect_text, timeout=30000)
            return
        except Exception:  # noqa: BLE001  點到 stale 副本 → 再點一次
            page.wait_for_timeout(800)
    main = page.locator('[data-testid="stMain"]').inner_text()
    raise AssertionError(
        f"「🧪 跑代理評估」點了沒生效(畫面沒出現 {expect_text!r}):{main[:600]}")


def test_probe_attaches_metric_e2e(al_isolated_server, page, synthetic_dataset):
    """AC-G1(一鍵入帳):送標開一個回合 → 按「🧪 跑代理評估」→ 卡上出現 probe_acc,
    且 rounds.jsonl 出現 kind=="metric" 且 source=="probe" 的行(磁碟真相,非只看畫面)。"""
    srv = al_isolated_server
    load_app(page, srv.base_url)
    run_viz(page, synthetic_dataset)
    sample_to_cart(page)
    goto_cart(page)
    assert cart_send_count(page) >= 2
    send_cart(page)

    starts = [r for r in read_jsonl(srv.rounds_path) if r.get("kind") == "start"]
    assert starts, "送標必須開一個回合"
    rid = starts[-1]["round_id"]

    goto_readback(page)
    _click_probe(page, rid, "代理評估完成")

    probes = [m for m in read_jsonl(srv.rounds_path)
              if m.get("kind") == "metric" and m.get("source") == "probe"]
    assert probes, ("「🧪 跑代理評估」必須把 source=='probe' 的 metric 寫進 rounds.jsonl:"
                    f"{read_jsonl(srv.rounds_path)}")
    names = {m["name"] for m in probes}
    assert "probe_acc" in names and "probe_macro_f1" in names, f"指標名不對:{names}"
    assert all(m["round_id"] == rid for m in probes), "指標必須掛在按下去的那個回合上"

    expect(page.get_by_text(re.compile("probe_acc")).first).to_be_visible()


def test_curve_and_advice_render_e2e(al_isolated_server, page, synthetic_dataset):
    """AC-G2(曲線 + 建議):3 個回合各有 probe_acc → 折線圖渲染 + 停止建議 caption(附數字);
    且「不自動停」的誠實界線常駐在畫面上。"""
    srv = al_isolated_server
    _seed_rounds(srv, [0.70, 0.80, 0.802])       # 斜率 0.002 < 0.005 → plateau

    load_app(page, srv.base_url)
    goto_readback(page)

    expect(page.locator('.st-key-al_learning_curve')).to_be_visible()
    body = page.locator('[data-testid="stMain"]').inner_text()
    assert "停止建議" in body, f"必須顯示停止建議:{body[:400]}"
    assert re.search(r"停止建議.*?(報酬遞減|仍在進步)", body, re.S), (
        f"建議必須是 plateau/improving 其一:{body[:600]}")
    assert re.search(r"0\.002", body), f"建議必須附**數字**(每 100 張的指標增量):{body[:600]}"
    # 產品紀律:不做自動停
    assert "是否停止由你決定" in body, "必須常駐「是否停止由你決定」(本工具不自動停)"
    assert "代理" in body, "必須常駐「代理指標僅供方向」的誠實界線"


def test_insufficient_class_warns_e2e(al_isolated_server, page, single_class_dataset):
    """AC-G3(類不足誠實):單一類別的資料 → 按代理評估要出 warning 且**不入帳**、不崩潰。"""
    srv = al_isolated_server
    load_app(page, srv.base_url)
    run_viz(page, single_class_dataset)
    sample_to_cart(page)
    goto_cart(page)
    assert cart_send_count(page) >= 2
    send_cart(page)

    starts = [r for r in read_jsonl(srv.rounds_path) if r.get("kind") == "start"]
    assert starts
    rid = starts[-1]["round_id"]

    goto_readback(page)
    _click_probe(page, rid, "跑不了代理評估")     # 誠實說跑不了,而不是靜默略過

    body = page.locator('[data-testid="stMain"]').inner_text()
    assert "insufficient_class" in body, f"warning 必須點名是哪一類不足:{body[:500]}"
    assert "onlyclass" in body, f"warning 必須含該類名:{body[:500]}"
    probes = [m for m in read_jsonl(srv.rounds_path)
              if m.get("kind") == "metric" and m.get("source") == "probe"]
    assert not probes, "類別不足時不得入帳"
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)   # 不崩潰
