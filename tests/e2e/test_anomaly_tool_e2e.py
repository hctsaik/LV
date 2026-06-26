"""E2E 真實行為驗收:瑕疵偵測工具(設計 06_anomaly_tool.md 的 E2E-AC*)。

這是「使用者真的能用」的 spec,不進 PG 自主修綠迴圈(gate.py 以 -m "not e2e" 排除),
由 /ux-test 或人觸發。**真實行為斷言**:不是查 element 存在,而是查
「被注入缺陷的物件真的排到前段」「匯出的是原圖」「錯誤路徑顯示中文指引」。

PG 實作 GUI 時需對齊以下契約鍵(目前 UI 未建,故本檔在 UI 完成前為紅/失敗,屬預期):
  - 工具入口:可由文字「瑕疵偵測」進入該工具分頁。
  - 資料夾輸入:沿用既有 `_add_folder(page, "anomaly_folder", <root>)` 模式(list_key=anomaly_folder)。
  - 執行鈕:`.st-key-anomaly_run button`。
  - 排序結果區:`.st-key-anomaly_ranked`(內含每個物件來源檔名文字,順序=由最可疑到最不可疑)。
  - 錯誤訊息區:無 `labels/` 時顯示中文指引(含「YOLO」「標註」字樣),不得出現 stException。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from playwright.sync_api import expect

from .conftest import _add_folder, load_app, wait_idle

pytestmark = pytest.mark.e2e
expect.set_options(timeout=20000)


@pytest.fixture(scope="module")
def anomaly_page(app_server, browser, synthetic_yolo_dataset):
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    load_app(page, app_server)
    page.locator('.st-key-tool_switch').get_by_text("瑕疵偵測", exact=True).click()  # 進入第 8 工具
    wait_idle(page)
    yield page, synthetic_yolo_dataset
    ctx.close()


def _no_exception(page):
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)


def _run(page, root: Path):
    _add_folder(page, "anomaly_folder", str(root))
    wait_idle(page)
    page.locator('.st-key-anomaly_run button').click()
    wait_idle(page, timeout=180000)  # 首跑含模型載入,給足時間


def test_defects_rank_in_top(anomaly_page):  # E2E-AC1(真實行為,非 element 存在)
    page, ds = anomaly_page
    _run(page, ds["root"])
    _no_exception(page)
    ranked_text = page.locator('.st-key-anomaly_ranked').inner_text()
    # 取出排序清單中出現的來源檔 stem,前段(前 2*K 名)應涵蓋全部缺陷檔
    order = re.findall(r'(normal_\d+|defect_\d+)', ranked_text)
    assert order, "排序結果區應列出物件來源檔名"
    k = ds["n_defect"]
    top = order[: 2 * k]
    defect_stems = {s for s, _ in ds["defect_keys"]}
    assert defect_stems.issubset(set(top)), f"缺陷應集中在前段;實際前段={top}"


def test_select_and_add_to_cart(anomaly_page):  # E2E-AC2
    page, ds = anomaly_page
    # 一鍵把離群/可疑物件加入購物車 → 計數應變非零(離群候選 ⊇ 缺陷)
    cart = page.locator('.st-key-anomaly_cart_count')
    wait_idle(page)
    btn = page.locator('.st-key-anomaly_select_outliers button')
    expect(btn).to_be_visible()
    btn.click()
    # 點擊觸發 rerun;plotly on_select + 重渲染期間,舊/新 render 會暫時並存(雙份同 key
    # block),而 wait_idle 看的狀態 widget 會在 DOM 換完前就返回。先等計數區收斂成「單一
    # 元素且非零」再斷言,否則 strict-mode 會命中 2 個。斷言內容不變(只是補等待)。
    page.wait_for_function(
        """() => { const e = document.querySelectorAll('.st-key-anomaly_cart_count');
                   return e.length === 1 && /[1-9]/.test(e[0].innerText); }""",
        timeout=30000)
    expect(cart).to_contain_text(re.compile(r"[1-9]"))  # 計數已更新成非零
    _no_exception(page)
    after = int(re.sub(r"\D", "", cart.inner_text()) or 0)
    assert after >= 1  # 判為可疑的已加入(完整偵測力由 test_defects_rank_in_top 驗)


def test_export_is_original_image(anomaly_page, tmp_path):  # E2E-AC3(匯出原圖,非裁切)
    import io
    import zipfile
    from PIL import Image
    page, ds = anomaly_page
    # 等匯出鈕從 rerun 暫態的雙份(新 enabled + 舊 disabled)收斂成單一且可用,再點,
    # 否則 strict-mode 會命中 2 個。
    page.wait_for_function(
        """() => { const b = document.querySelectorAll('.st-key-anomaly_export button');
                   return b.length === 1 && !b[0].disabled; }""",
        timeout=30000)
    with page.expect_download() as dl:
        page.locator('.st-key-anomaly_export button').click()
    data = Path(dl.value.path()).read_bytes()
    zf = zipfile.ZipFile(io.BytesIO(data))
    imgs = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    assert imgs, "zip 應含影像"
    with Image.open(io.BytesIO(zf.read(imgs[0]))) as im:
        assert im.size == (128, 128)  # 原圖尺寸(夾具 128x128),非裁切小圖
    assert "manifest.csv" in zf.namelist()


def test_score_and_class_filters(anomaly_page):  # 篩選真的會篩(真實行為,非 element 存在)
    page, ds = anomaly_page
    # pipeline 已於 test_defects_rank_in_top 跑過(共享 module-scope anomaly_page)
    wait_idle(page)
    total = ds["n_normal"] + ds["n_defect"]  # 25 個物件(20 good + 5 bad)

    def _count() -> int:
        page.wait_for_function(
            r"() => /符合 \d+ \/ \d+ 個/.test(document.body.innerText)", timeout=15000)
        el = page.get_by_text(re.compile(r"符合 \d+ / \d+ 個")).first
        return int(re.search(r"符合 (\d+) /", el.inner_text()).group(1))

    def _stems() -> set[str]:
        return set(re.findall(r'(normal_\d+|defect_\d+)',
                              page.locator('.st-key-anomaly_ranked').inner_text()))

    thumb = page.locator('.st-key-anomaly_heat_filter [role="slider"]')

    def _slider(n_pageup: int) -> None:
        """調分數門檻:n>0 往高(PageUp)、n<0 往低(PageDown)。Streamlit slider 認
        PageUp/PageDown(End/Home 不動;未聚焦的裸 track-click 也不動),故先 focus 再按。"""
        thumb.focus()
        key = "PageUp" if n_pageup > 0 else "PageDown"
        for _ in range(abs(n_pageup)):
            page.keyboard.press(key)
        wait_idle(page)

    # ── (a) 分數門檻 slider 真的會縮小集合(回應「filter 好像沒作用」)──
    base = _count()                      # 預設在最小門檻 → 全部通過
    assert base >= total - 1, f"最小門檻時應幾乎全通過:{base}/{total}"
    _slider(6)                           # 門檻拉高 → 排除低分(正常),只剩可疑
    page.wait_for_function(              # 等符合數真的下降(避開 rerun 過渡態)
        f"() => {{ const m = document.body.innerText.match(/符合 (\\d+) \\/ {total}/);"
        f" return m && +m[1] < {total}; }}", timeout=15000)
    hi = _count()
    assert hi < base, f"分數 slider 沒有篩選作用:{base} -> {hi}"
    _slider(-10)                         # 回到最小門檻,單獨驗下面的類別篩選

    # ── (b) 類別篩選:選 bad 只剩 5 個缺陷、選 good 只剩 20 個正常 ──
    ms = page.locator('.st-key-anomaly_class_filter')

    def _pick_class(name: str) -> None:
        # multiselect 選完選項後下拉仍開著(可連選特性)→ 會蓋住 Clear all/其他控制項,
        # 按 Escape 關掉再繼續,否則後續的 Clear all/選取會點到空白。
        ms.locator('[data-baseweb="select"]').click()
        page.get_by_role("option", name=name, exact=True).click()
        page.keyboard.press("Escape")
        wait_idle(page)

    def _clear_class() -> None:
        ms.get_by_role("button", name="Clear all").click()
        page.wait_for_function(                      # 等回 25/25 確認清空真的生效
            rf"() => /符合 {total} \/ {total} /.test(document.body.innerText)", timeout=15000)
        wait_idle(page)

    def _assert_class(prefix: str, n: int) -> None:
        # 選類別後 rerun 期間 caption 與排序清單可能短暫不同步(舊/新 render 並存)→
        # 等排序清單收斂成「只剩該類」再斷言數量,避免抓到過渡態。
        page.wait_for_function(
            """(p) => {
                const r = document.querySelectorAll('.st-key-anomaly_ranked');
                if (r.length !== 1) return false;
                const s = r[0].innerText.match(/(normal_\\d+|defect_\\d+)/g) || [];
                return s.length > 0 && s.every(x => x.startsWith(p));
            }""", arg=prefix, timeout=15000)
        wait_idle(page)
        assert _count() == n, f"類別篩選後應只剩 {n} 個({prefix})"
        assert all(x.startswith(prefix) for x in _stems())

    _pick_class("bad")                   # 只看 bad → 5 個缺陷
    _assert_class("defect_", ds["n_defect"])
    _clear_class()
    _pick_class("good")                  # 只看 good → 20 個正常
    _assert_class("normal_", ds["n_normal"])
    _no_exception(page)
    _clear_class()                       # 還原篩選,不污染後續


def test_confirm_normal_fewshot_rerun(app_server, browser, synthetic_yolo_dataset):
    """2-stage few-shot 確認流程:執行→把最不可疑標為正常範例→用乾淨 bank 重跑(真實行為)。"""
    ds = synthetic_yolo_dataset
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    load_app(page, app_server)
    page.locator('.st-key-tool_switch').get_by_text("瑕疵偵測", exact=True).click()
    wait_idle(page)
    _add_folder(page, "anomaly_folder", str(ds["root"]))
    wait_idle(page)
    page.locator('.st-key-anomaly_run button').click()
    wait_idle(page, timeout=180000)
    expect(page.locator('.st-key-anomaly_ranked')).to_contain_text("defect_")
    # 把最不可疑的標為正常範例(few-shot 種子)→ 應出現「已標記正常」狀態
    seed = page.locator('.st-key-anomaly_autoseed_normal button')
    expect(seed).to_be_visible()
    seed.click()
    expect(page.locator('[data-testid="stAppViewContainer"]')).to_contain_text("已標記正常")
    # 用乾淨 few-shot bank 重跑(第一次執行後 sidebar 收起,等重開鈕出現再點開)
    reopen = page.locator('.st-key-reopen_sidebar button')
    expect(reopen).to_be_visible()
    reopen.click()
    wait_idle(page)
    run2 = page.locator('.st-key-anomaly_run button')
    expect(run2).to_be_visible()
    run2.click()
    wait_idle(page, timeout=180000)
    _no_exception(page)
    expect(page.locator('.st-key-anomaly_ranked')).to_contain_text("defect_")
    ranked = page.locator('.st-key-anomaly_ranked').inner_text()
    order = re.findall(r'(normal_\d+|defect_\d+)', ranked)
    defect_stems = {s for s, _ in ds["defect_keys"]}
    assert defect_stems.issubset(set(order[: 2 * ds["n_defect"]]))  # 乾淨 bank 仍抓到缺陷
    ctx.close()


def test_error_path_no_labels(app_server, browser, tmp_path):  # E2E-AC6(友善錯誤,非 traceback)
    (tmp_path / "images").mkdir()
    from PIL import Image
    Image.new("RGB", (64, 64), (0, 170, 0)).save(tmp_path / "images" / "a.jpg")  # 無 labels/
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
    page = ctx.new_page()
    load_app(page, app_server)
    page.locator('.st-key-tool_switch').get_by_text("瑕疵偵測", exact=True).click()
    wait_idle(page)
    _add_folder(page, "anomaly_folder", str(tmp_path))
    page.locator('.st-key-anomaly_run button').click()
    wait_idle(page)
    container = page.locator('[data-testid="stAppViewContainer"]')
    # 等執行後的 rerun 真的畫出指引再讀(否則 wait_idle 可能在 rerun 前 ~0.5s 就回,
    # 抓到執行前的預設文字 → race);expect 自動重試覆蓋這個 gap。
    expect(container).to_contain_text("找不到 YOLO 標註")
    body = container.inner_text()
    assert ("YOLO" in body and "標註" in body)        # 中文指引
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)  # 無原始 traceback
    ctx.close()
