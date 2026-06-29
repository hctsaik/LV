"""E2E 真實行為驗收:瑕疵偵測工具(重構成三 tab wizard 後)。

**真實行為斷言**(非查 element 存在):「被注入缺陷的物件真的排到前段」「匯出的是原圖」
「分數/類別 filter 真的會篩」「錯誤路徑顯示中文指引」。

重構後流程(取代舊單資料夾→run):① 建模 → ② 套用偵測(分開的目標資料夾)→ ③ 挑樣。
本檔的偵測結果(散點/排序/篩選/購物車/匯出)都在 **② 套用偵測** tab。
為測流程,訓練(①)與目標(②)用同一個 dataset root(synthetic_yolo_dataset)。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from playwright.sync_api import expect

from ._anomaly_wizard import (TAB_APPLY, apply_model, build_model, click_tab,
                              enter_anomaly)
from .conftest import _add_folder, load_app, wait_idle

pytestmark = pytest.mark.e2e
expect.set_options(timeout=20000)


@pytest.fixture(scope="module")
def anomaly_page(app_server, browser, synthetic_yolo_dataset):
    """module-scope:建一次模型 + 套用一次,後續測試共用 ② 套用結果(快)。
    訓練與目標同 root;語義不指定(預設由啟發式 → 對 synthetic 多框會偏 object,
    但本檔不驗 head,只驗排序/篩選/購物車/匯出 → 不受語義影響)。"""
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    enter_anomaly(page, app_server)
    build_model(page, synthetic_yolo_dataset["root"])
    apply_model(page, synthetic_yolo_dataset["root"])
    yield page, synthetic_yolo_dataset
    ctx.close()


def _no_exception(page):
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)


def _ranked_stems(page, want: int = 14) -> list[str]:
    """讀 ② 「選一個看圖」selectbox(anomaly_inspect)的選項(= shown,由最可疑→最不可疑)
    作為來源檔 stem 順序。BaseWeb 下拉虛擬化 → 開啟後 JS 逐步下捲、累積去重(保留出現順序)。"""
    sb = page.locator('.st-key-anomaly_inspect [data-baseweb="select"]')
    for _ in range(3):                       # 偶有下拉沒開 → 重試
        sb.click()
        try:
            page.wait_for_selector('[role="option"]', timeout=6000)
            break
        except Exception:
            page.keyboard.press("Escape")
            wait_idle(page)
    seen: list[str] = []
    for _ in range(8):
        for t in page.get_by_role("option").all_inner_texts():
            if t not in seen:
                seen.append(t)
        if len(seen) >= want:
            break
        moved = page.evaluate("""() => {
            const o = document.querySelector('[role="option"]');
            const lb = o && o.closest('ul,[role="listbox"],[data-baseweb="menu"]');
            if (!lb) return false;
            const before = lb.scrollTop; lb.scrollTop += 250; return lb.scrollTop !== before;
        }""")
        page.wait_for_timeout(220)
        if not moved:
            break
    page.keyboard.press("Escape")
    wait_idle(page)
    return [m.group(1) for t in seen
            if (m := re.search(r'(normal_\d+|defect_\d+)', t))]


def test_defects_rank_in_top(anomaly_page):  # 真實行為,非 element 存在
    page, ds = anomaly_page
    click_tab(page, TAB_APPLY)            # 確保在②(inspector 在此 tab)再讀排序選單
    _no_exception(page)
    # 排序挑選器(selectbox)的選項順序=由最可疑到最不可疑;前段(前 2*K 名)應涵蓋全部缺陷檔
    order = _ranked_stems(page)
    assert order, "看圖選單應列出物件來源檔名"
    k = ds["n_defect"]
    top = order[: 2 * k]
    defect_stems = {s for s, _ in ds["defect_keys"]}
    assert defect_stems.issubset(set(top)), f"缺陷應集中在前段(選單頂端);實際前段={top}"


def test_select_and_add_to_cart(anomaly_page):
    page, ds = anomaly_page
    # module fixture 共用 page;前一個 test 的 rerun 可能把 st.tabs 留在①→ 先切回②
    click_tab(page, TAB_APPLY)
    # 一鍵把離群/可疑物件加入購物車 → 計數應變非零(離群候選 ⊇ 缺陷)
    cart = page.locator('.st-key-anomaly_cart_count')
    wait_idle(page)
    btn = page.locator('.st-key-anomaly_select_outliers button')
    expect(btn).to_be_visible()
    btn.click()
    # on_click 觸發 rerun → st.tabs 重置回① → 重切回②,讓計數膠囊重新可見再斷言
    click_tab(page, TAB_APPLY)
    # 點擊觸發 rerun;plotly on_select + 重渲染期間,舊/新 render 會暫時並存(雙份同 key
    # block),而 wait_idle 看的狀態 widget 會在 DOM 換完前就返回。先等計數區收斂成「單一
    # 元素且非零」再斷言,否則 strict-mode 會命中 2 個。
    page.wait_for_function(
        """() => { const e = document.querySelectorAll('.st-key-anomaly_cart_count');
                   return e.length === 1 && /[1-9]/.test(e[0].innerText); }""",
        timeout=30000)
    expect(cart).to_contain_text(re.compile(r"[1-9]"))  # 計數已更新成非零
    _no_exception(page)
    after = int(re.sub(r"\D", "", cart.inner_text()) or 0)
    assert after >= 1  # 判為可疑的已加入(完整偵測力由 test_defects_rank_in_top 驗)


def test_export_is_original_image(anomaly_page, tmp_path):  # 匯出原圖,非裁切
    import io
    import zipfile

    from PIL import Image
    page, ds = anomaly_page
    # module fixture 共用 page;前一個 test 的 rerun 可能把 st.tabs 留在① → 先切回②(匯出鈕在②)
    click_tab(page, TAB_APPLY)
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
    click_tab(page, TAB_APPLY)            # 確保在②(篩選器/inspector 在此 tab)
    wait_idle(page)
    total = ds["n_normal"] + ds["n_defect"]  # 25 個物件(20 good + 5 bad)

    def _count() -> int:
        page.wait_for_function(
            r"() => /符合 \d+ \/ \d+ 個/.test(document.body.innerText)", timeout=15000)
        el = page.get_by_text(re.compile(r"符合 \d+ / \d+ 個")).first
        return int(re.search(r"符合 (\d+) /", el.inner_text()).group(1))

    def _stems() -> set[str]:
        return set(_ranked_stems(page))  # 排序清單已移除 → 讀 selectbox 選項(= shown)

    # 雙邊範圍 slider → 有兩個 thumb;取「低界」(.first)。拉高低界即排除低分(正常),
    # 集合縮小,沿用原本「分數 filter 真的會篩」的真實行為斷言。
    thumb = page.locator('.st-key-anomaly_heat_filter_apply [role="slider"]').first

    def _slider(n_pageup: int) -> None:
        """調分數低界:n>0 往高(PageUp)、n<0 往低(PageDown)。Streamlit slider 認
        PageUp/PageDown(End/Home 不動;未聚焦的裸 track-click 也不動),故先 focus 再按。
        ⚠ slider 改值會整頁 rerun → st.tabs 重置回① → 讀「符合」前先重切回② 讓 inspector 重現。"""
        thumb.focus()
        key = "PageUp" if n_pageup > 0 else "PageDown"
        for _ in range(abs(n_pageup)):
            page.keyboard.press(key)
        wait_idle(page)
        click_tab(page, TAB_APPLY)

    # ── (a) 分數門檻 slider 真的會縮小集合 ──
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
    ms = page.locator('.st-key-anomaly_class_filter_apply')

    def _pick_class(name: str) -> None:
        # multiselect 選完選項後下拉仍開著(可連選特性)→ 會蓋住 Clear all/其他控制項,
        # 按 Escape 關掉再繼續,否則後續的 Clear all/選取會點到空白。
        # ⚠ 選項觸發 rerun → st.tabs 重置回① → 選完重切回②,後續讀「符合」才看得到。
        ms.locator('[data-baseweb="select"]').click()
        page.get_by_role("option", name=name, exact=True).click()
        page.keyboard.press("Escape")
        wait_idle(page)
        click_tab(page, TAB_APPLY)

    def _clear_class() -> None:
        ms.get_by_role("button", name="Clear all").click()
        wait_idle(page)
        click_tab(page, TAB_APPLY)                    # rerun 重置 tabs → 重切回② 再確認清空
        page.wait_for_function(                      # 等回 25/25 確認清空真的生效
            rf"() => /符合 {total} \/ {total} /.test(document.body.innerText)", timeout=15000)
        wait_idle(page)

    def _assert_class(prefix: str, n: int) -> None:
        # 選類別後 rerun 期間 caption 可能短暫不同步(舊/新 render 並存)→ 等「符合數」收斂成 n
        # 再斷言,避開過渡態。(排序清單已移除,改以 caption 計數判收斂。)
        page.wait_for_function(
            f"() => {{ const m = document.body.innerText.match(/符合 (\\d+) \\/ {total}/);"
            f" return m && +m[1] === {n}; }}", timeout=15000)
        wait_idle(page)
        assert _count() == n, f"類別篩選後應只剩 {n} 個({prefix})"
        assert all(x.startswith(prefix) for x in _stems()), \
            f"篩選後選單應只剩 {prefix} 類"

    _pick_class("bad")                   # 只看 bad → 5 個缺陷
    _assert_class("defect_", ds["n_defect"])
    _clear_class()
    _pick_class("good")                  # 只看 good → 20 個正常
    _assert_class("normal_", ds["n_normal"])
    _no_exception(page)
    _clear_class()                       # 還原篩選,不污染後續


def test_confirm_normal_reapply(app_server, browser, synthetic_yolo_dataset):
    """2-stage few-shot 確認流程(wizard 版):② 套用後把最不可疑的標為正常範例(few-shot 種子)
    → 按「🔁 就地重評」用乾淨 few-shot bank 在目標資料上重算(不動模型)→ 缺陷仍排前段。
    取代舊『autoseed→重開側欄→重 run』流程(② 就地重評不再經側欄)。"""
    ds = synthetic_yolo_dataset
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.set_default_timeout(30000)
    try:
        enter_anomaly(page, app_server)
        build_model(page, ds["root"])
        apply_model(page, ds["root"])
        assert any(s.startswith("defect_") for s in _ranked_stems(page)), \
            "首次套用應在排序挑選器列出缺陷"

        # 把最不可疑的標為正常範例(few-shot 種子)→ 出現「已標記正常」狀態
        seed = page.locator('.st-key-anomaly_autoseed_normal button')
        expect(seed).to_be_visible()
        seed.click()
        wait_idle(page)
        # on_click rerun → st.tabs 重置回① → 重切回②(狀態文字/重評鈕都在②)
        click_tab(page, TAB_APPLY)
        expect(page.locator('[data-testid="stMain"]')).to_contain_text("已標記正常")

        # 🔁 就地重評(用乾淨 few-shot bank 重算,不動模型)→ 等散點重畫。
        # st.tabs 切換/rerun 過渡期舊+新 fragment 會短暫並存(同 key 雙份)→ 等收斂成
        # 「單一可見」再點,否則 strict-mode 命中 2 個。
        page.wait_for_function(
            """() => { const bs = Array.from(document.querySelectorAll(
                          '.st-key-anomaly_reapply_btn button'))
                          .filter(b => b.offsetParent !== null);
                       return bs.length === 1; }""", timeout=30000)
        reapply = page.locator('.st-key-anomaly_reapply_btn button').locator('visible=true').first
        reapply.click()
        page.wait_for_timeout(1500)
        wait_idle(page, timeout=60000)
        # 重評 rerun 把 tabs 重置回① → 重切回② 才看得到②散點(別用太廣的 anomaly_scatter,
        # 會抓到①隱藏的 _build 散點 → timeout)
        click_tab(page, TAB_APPLY)
        page.locator('[class*="st-key-anomaly_scatter_apply"]').first.wait_for(
            state="visible", timeout=180000)
        wait_idle(page, timeout=60000)
        page.wait_for_timeout(1000)
        _no_exception(page)

        order = _ranked_stems(page)
        assert any(s.startswith("defect_") for s in order), "乾淨 bank 重評後仍應列出缺陷"
        defect_stems = {s for s, _ in ds["defect_keys"]}
        assert defect_stems.issubset(set(order[: 2 * ds["n_defect"]]))  # 乾淨 bank 仍抓到缺陷
    finally:
        ctx.close()


def test_error_path_no_labels(app_server, browser, tmp_path):  # 友善錯誤,非 traceback
    """② 套用偵測吃「無 labels/」的目標資料夾 → 顯中文指引(含 YOLO/標註)、無 traceback。
    需先在①建一個模型(②有前置守門:沒模型只顯引導文案),用 synthetic 之外的乾淨小資料建模。"""
    from PIL import Image

    # ① 訓練資料夾:正常的小 YOLO 資料(建模用)。≥8 張:run_pipeline 的 LOF 需 min_samples
    # (預設 5)≤ 物件數,4 張會「建模失敗:min_samples(5)must be at most …(4)」→ 模型沒建成、
    # ②前置守門擋住、目標 picker 不渲染。用 8 張穩過建模。
    train = tmp_path / "train"
    (train / "images").mkdir(parents=True)
    (train / "labels").mkdir()
    (train / "classes.txt").write_text("good\n", encoding="utf-8")
    for i in range(8):
        Image.new("RGB", (128, 128), (0, 170, 0)).save(train / "images" / f"g{i}.jpg", quality=92)
        (train / "labels" / f"g{i}.txt").write_text("0 0.5 0.5 0.9 0.9\n", encoding="utf-8")

    # ② 目標資料夾:有影像、無 labels/
    target = tmp_path / "target"
    (target / "images").mkdir(parents=True)
    Image.new("RGB", (64, 64), (0, 170, 0)).save(target / "images" / "a.jpg")

    ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
    page = ctx.new_page()
    try:
        enter_anomaly(page, app_server)
        build_model(page, train)             # ① 先建模(過②前置守門)

        # ② 切 tab、加無 labels 的目標、套用 → 應出現中文指引(套用不會畫散點,故不等散點)
        from ._anomaly_wizard import click_tab
        click_tab(page, "② 套用偵測")
        _add_folder(page, "anomaly_target_folder", str(target))
        wait_idle(page)
        # ⚠ _add_folder 的 rerun 把 st.tabs 重置回① → 重切回② 才點得到套用鈕(否則點到隱藏鈕)
        click_tab(page, "② 套用偵測")
        page.locator('.st-key-anomaly_apply_btn button').click()
        wait_idle(page, timeout=60000)
        # ⚠ 套用 callback rerun 把 st.tabs 重置回① → 中文指引在②(_anomaly_tab_apply 內)會被藏起,
        # inner_text 只讀「可見」文字 → 須重切回② 才讀得到。重切回②再斷言。
        click_tab(page, "② 套用偵測")
        container = page.locator('[data-testid="stMain"]')
        # 等②真的畫出指引再讀(expect 自動重試覆蓋 race gap)
        expect(container).to_contain_text("找不到 YOLO")
        body = container.inner_text()
        assert ("YOLO" in body and "標註" in body)        # 中文指引
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)  # 無原始 traceback
    finally:
        ctx.close()
