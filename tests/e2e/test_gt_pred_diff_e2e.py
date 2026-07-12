"""E2E:10 gt_pred_diff GUI 接線(AC-G1..G4)。

設計:3_Architect_Design/10_gt_pred_diff.md

合成一個**答案已知**的偵測資料集 + 一份「故意改壞的預測」:
  · miss   — GT 兩框、預測只有一框            → 該進 **漏抓 FN**
  · extra  — GT 一框、預測多一個遠框          → 該進 **誤抓 FP**
  · wrong  — 位置對但類別錯                    → 該進 **類別混淆**
  · perfect— GT 與預測完全一致(高 conf)       → **不該進任何佇列**
「全對那張不在任何佇列」是防 false-green 的關鍵:一個「全部圖都進佇列」的壞實作也能讓
前三條通過,唯有這條擋得住。
"""
from __future__ import annotations

import json
import re

import pytest
from PIL import Image
from playwright.sync_api import expect

from ._al_loop_harness import goto_readback, read_jsonl
from .conftest import _add_folder, load_app, newest_handoff_dir, wait_idle

pytestmark = pytest.mark.e2e
expect.set_options(timeout=20000)


@pytest.fixture(scope="session")
def detect_ds_with_preds(tmp_path_factory):
    """回 (dataset_root, pred_root)。GT 進 labels/;預測是「改壞的副本」放 pred/labels/。"""
    root = tmp_path_factory.mktemp("pd_ds")
    (root / "images").mkdir()
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
    pred = tmp_path_factory.mktemp("pd_pred")
    (pred / "labels").mkdir()

    def img(stem, rgb):
        Image.new("RGB", (128, 128), rgb).save(root / "images" / f"{stem}.jpg", quality=90)

    def gt(stem, text):
        (root / "labels" / f"{stem}.txt").write_text(text, encoding="utf-8")

    def pr(stem, text):
        (pred / "labels" / f"{stem}.txt").write_text(text, encoding="utf-8")

    # miss:GT 兩框,預測只命中第一個 → fn == 1
    img("miss", (200, 60, 60))
    gt("miss", "0 0.25 0.25 0.2 0.2\n0 0.75 0.75 0.2 0.2\n")
    pr("miss", "0 0.25 0.25 0.2 0.2 0.9\n")

    # extra:GT 一框,預測多一個遠框 → fp == 1
    img("extra", (60, 200, 60))
    gt("extra", "0 0.5 0.5 0.3 0.3\n")
    pr("extra", "0 0.5 0.5 0.3 0.3 0.9\n0 0.9 0.1 0.1 0.1 0.85\n")

    # wrong:位置對、類別錯 → confused == 1
    img("wrong", (60, 60, 200))
    gt("wrong", "0 0.5 0.5 0.4 0.4\n")
    pr("wrong", "1 0.5 0.5 0.4 0.4 0.9\n")

    # perfect:完全一致、高 conf → 不進任何佇列
    img("perfect", (200, 200, 60))
    gt("perfect", "0 0.5 0.5 0.4 0.4\n")
    pr("perfect", "0 0.5 0.5 0.4 0.4 0.95\n")

    return root, pred


def _open_viz_detector(page, ds_root, pred_root) -> None:
    """Visualize → Object Detector,填資料夾 + 模型預測資料夾 → Run。"""
    page.locator('.st-key-viz_mode').get_by_text("Object Detector").first.click()
    wait_idle(page)
    page.wait_for_function(
        """() => document.querySelectorAll('.st-key-viz_folder_list_add input').length === 1""",
        timeout=20000)
    _add_folder(page, "viz_folder_list", str(ds_root))

    inp = page.locator('.st-key-viz_pred_root_input input').first
    inp.wait_for(state="attached", timeout=30000)
    inp.click()
    inp.fill(str(pred_root))
    inp.press("Enter")
    wait_idle(page)

    page.locator('.st-key-run_viz button').first.click()
    import time
    deadline = time.time() + 260
    while time.time() < deadline:
        if page.locator('.st-key-viz_scatter_wrap g.points path').count() > 0:
            break
        time.sleep(0.3)
    wait_idle(page, timeout=120000)


def _open_diff_panel(page) -> None:
    page.locator('.st-key-viz_panel_view').get_by_text("挖錯", exact=True).click()
    wait_idle(page, timeout=60000)


def _pick_queue(page, label_prefix: str, qkey: str) -> None:
    """切佇列並**確認真的切過去了**。

    切換訊號不用 aria-checked(這個 segmented_control 沒帶),而用**每個佇列各自唯一的送標鈕 key**
    (`pd_send_fn` / `pd_send_fp` / …)——它出現就代表面板真的在該佇列上。點了沒切卻繼續往下,
    會讓斷言驗到**錯的佇列**,是最典型的誤判來源(踩過)。
    """
    want = f'.st-key-pd_send_{qkey}'
    for _ in range(4):
        if page.locator(want).count():
            return
        page.locator('.st-key-pd_queue').get_by_text(label_prefix).first.click()
        wait_idle(page, timeout=60000)
        try:
            page.wait_for_selector(want, state="attached", timeout=15000)
            return
        except Exception:  # noqa: BLE001
            page.wait_for_timeout(600)
    raise AssertionError(f"佇列沒切到「{label_prefix}」(等不到 {want})")


def _gallery_stems(page) -> set[str]:
    """佇列卡上的檔名(caption 形如 `miss.jpg · 錯誤分 2.0 · 漏1/誤0/混0/低信0`)。"""
    body = page.locator('.st-key-viz_pred_diff_panel').first.inner_text()
    return set(re.findall(r"(\w+)\.jpg\s*·\s*錯誤分", body))


def test_queues_are_real_e2e(app_page, detect_ds_with_preds):
    """AC-G1(佇列真實性):FN/FP/混淆 佇列各含對的那張,且**全對那張不在任何佇列**。"""
    page = app_page
    ds_root, pred_root = detect_ds_with_preds
    _open_viz_detector(page, ds_root, pred_root)
    _open_diff_panel(page)

    seen_all = set()
    for queue, qkey, want in (("漏抓 FN", "fn", "miss"),
                              ("誤抓 FP", "fp", "extra"),
                              ("類別混淆", "confused", "wrong")):
        _pick_queue(page, queue, qkey)
        stems = _gallery_stems(page)
        assert want in stems, f"「{queue}」佇列必須含 {want}.jpg,實得 {stems}"
        seen_all |= stems

    assert "perfect" not in seen_all, (
        "GT 與預測完全一致的那張不該出現在任何佇列 —— 出現了代表在亂塞(false-green 溫床)。"
        f"實際看到:{seen_all}")


def test_two_source_render_e2e(app_page, detect_ds_with_preds):
    """AC-G2(疊框渲染):佇列卡上的影像是「畫了 GT + 預測兩色框」的版本,不是原圖。
    以 bytes 不同斷言(不做像素級比對)。"""
    page = app_page
    ds_root, pred_root = detect_ds_with_preds
    _open_viz_detector(page, ds_root, pred_root)
    _open_diff_panel(page)
    _pick_queue(page, "漏抓 FN", "fn")

    img = page.locator('.st-key-viz_pred_diff_panel img').first
    img.wait_for(state="visible", timeout=30000)
    rendered = page.evaluate("""async (el) => {
        const r = await fetch(el.src);
        const b = await r.arrayBuffer();
        return b.byteLength;
    }""", img.element_handle())
    original = (ds_root / "images" / "miss.jpg").stat().st_size
    assert rendered != original, (
        "佇列卡的影像必須是疊了兩色框的版本(bytes 應與原圖不同),"
        f"rendered={rendered} original={original}")


def test_send_from_queue_opens_round_e2e(al_isolated_server, page, detect_ds_with_preds):
    """AC-G3(送標接回合):從 FN 佇列送標 → 回合卡策略名含 `pred_diff:fn`。"""
    srv = al_isolated_server
    ds_root, pred_root = detect_ds_with_preds
    load_app(page, srv.base_url)
    _open_viz_detector(page, ds_root, pred_root)
    _open_diff_panel(page)
    _pick_queue(page, "漏抓 FN", "fn")

    for _ in range(4):
        btn = page.locator('.st-key-pd_send_fn button:visible:enabled')
        if btn.count():
            btn.first.click()
            wait_idle(page, timeout=60000)
        if [r for r in read_jsonl(srv.rounds_path) if r.get("kind") == "start"]:
            break
        page.wait_for_timeout(800)

    starts = [r for r in read_jsonl(srv.rounds_path) if r.get("kind") == "start"]
    assert starts, "從佇列送標必須開一個回合"
    assert starts[-1]["strategy"] == "pred_diff:fn", (
        f"回合策略名必須是 pred_diff:fn,實得 {starts[-1]['strategy']!r}")

    goto_readback(page)
    expect(page.get_by_text(re.compile("策略：pred_diff:fn")).first).to_be_visible()


def test_seed_pred_lands_e2e(al_isolated_server, page, detect_ds_with_preds):
    """AC-G4(seed=pred 落盤):預標來源選「模型預測」送標 → handoff 的 `images/<sha>.json`
    shapes 數 == 該圖的預測框數(標註者是改框,不是從零畫框)。"""
    srv = al_isolated_server
    ds_root, pred_root = detect_ds_with_preds
    load_app(page, srv.base_url)
    _open_viz_detector(page, ds_root, pred_root)
    _open_diff_panel(page)
    _pick_queue(page, "誤抓 FP", "fp")          # extra.jpg:預測 2 框

    # 預標來源 radio → 模型預測
    assert page.evaluate("""() => {
        const root = document.querySelector('.st-key-pd_send_fp_seed');
        if (!root) return false;
        const opt = [...root.querySelectorAll('label')]
            .filter(l => l.querySelector('input[type="radio"]'))
            .find(l => (l.textContent || '').includes('模型預測'));
        if (!opt) return false;
        if (!opt.querySelector('input[type="radio"]').checked) opt.click();
        return true;
    }"""), "找不到「預標來源 → 模型預測」選項"
    wait_idle(page)

    for _ in range(4):
        btn = page.locator('.st-key-pd_send_fp button:visible:enabled')
        if btn.count():
            btn.first.click()
            wait_idle(page, timeout=60000)
        if [r for r in read_jsonl(srv.rounds_path) if r.get("kind") == "start"]:
            break
        page.wait_for_timeout(800)

    handoff = newest_handoff_dir(srv)
    spec = json.loads((handoff / "_handoff.json").read_text(encoding="utf-8"))
    by_name = {it["filename"]: it for it in spec["items"]}
    assert "extra.jpg" in by_name, f"送出的批次應含 extra.jpg:{list(by_name)}"

    it = by_name["extra.jpg"]
    sidecar = handoff / "images" / f"{it['sha256']}.json"
    assert sidecar.exists(), "seed_source='pred' 必須把預測框預先畫進 sidecar"
    shapes = json.loads(sidecar.read_text(encoding="utf-8"))["shapes"]
    assert len(shapes) == 2, f"extra.jpg 的預測是 2 框,seed 應有 2 個 shape,實得 {len(shapes)}"
    assert it.get("seed_sha"), "spec 必須記下 seed 的內容 hash(供回讀端辨識『人沒動過』)"
