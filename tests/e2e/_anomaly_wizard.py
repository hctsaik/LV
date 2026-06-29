"""共用 helper:驅動重構後「瑕疵偵測」三 tab wizard(① 建模 → ② 套用 → ③ 挑樣)。

新流程(取代舊單資料夾→run):
- 工具切換:`.st-key-tool_switch` 內點「瑕疵偵測」。
- 三 tab 用 st.tabs(app.py:1998)= ["① 建模 / 載入模型","② 套用偵測","③ 挑樣送人工標"]。
  Playwright 切 tab = 點 `[role="tab"]` 內含該文字的那顆,切完 wait_idle。
- ① 訓練資料夾 list_key=anomaly_train_folder;語義 radio key=anomaly_train_semantic;
  建模鈕 key=anomaly_build_btn;模型暫存目錄 key=anomaly_model_dir;
  存/載 key=anomaly_save_model_btn / anomaly_load_model_btn。
- ② 目標資料夾 list_key=anomaly_target_folder;套用鈕 key=anomaly_apply_btn;
  結果散點 `[class*="st-key-anomaly_scatter"]`。
- ③ 2×2 模式鈕 key=anomaly_qmode_{novelty,confusion,balanced,pure};加購物車 key=anomaly_q_cart。

注意:st.tabs 每次 rerun 三 tab 全跑,跨 tab 狀態落 session_state;切 tab 只是切「可見」的那塊。
"""
from __future__ import annotations

from .conftest import _add_folder, load_app, wait_idle

TAB_BUILD = "① 建模 / 載入模型"
TAB_APPLY = "② 套用偵測"
TAB_SAMPLE = "③ 挑樣送人工標"


def enter_anomaly(page, app_server):
    """載入 app → 切到「瑕疵偵測」工具(停在預設 tab ①)。"""
    page.set_default_timeout(30000)
    load_app(page, app_server)
    page.locator('.st-key-tool_switch').get_by_text("瑕疵偵測", exact=True).click()
    wait_idle(page)


def _step_active(page, name: str) -> bool:
    """該步驟內容是否正在顯示(segmented_control 選中 → 只渲染該步;用主畫面是否含該步標記判斷)。"""
    return page.evaluate(
        """(n) => {
            const seg = document.querySelector('.st-key-anomaly_step');
            if (!seg) return false;
            const btns = Array.from(seg.querySelectorAll('button,[role="radio"],[role="button"]'));
            const t = btns.find(x => (x.innerText || '').trim() === n.trim());
            return !!(t && (t.getAttribute('aria-checked') === 'true'
                            || t.getAttribute('aria-pressed') === 'true'));
        }""", name)


def click_tab(page, name: str):
    """切到步驟選擇器(segmented_control key=anomaly_step)的某一步。

    重構後步驟存 session_state、跨 rerun 保留(不再像 st.tabs 那樣每次 rerun 彈回①),
    所以單擊即可。內容只渲染當前步;點當前步會 deselect→app 用上次步驟回填(內容不變),
    故先檢查是否已在該步、未在才點(避免把當前步 toggle 掉)。"""
    seg = page.locator('.st-key-anomaly_step')
    seg.first.wait_for(state="visible", timeout=15000)
    if not _step_active(page, name):
        seg.get_by_text(name, exact=True).first.click()
        wait_idle(page)
    page.wait_for_timeout(400)


def set_semantic(page, option_text: str):
    """① 設 label 語義 radio(anomaly_train_semantic)。option_text 例:'物件類別' / '瑕疵類別'。
    只點「真正的 radio 選項 label」(含 input[type=radio] 的那種)—— radio 群組的「問題 label」
    同時含『物件類別』與『瑕疵類別』兩字串,若不排除會誤點到它(不切換實際選項 → head 沒訓)。
    設完 wait_idle。回傳是否確實切到目標(選項的 radio input 已 checked)。"""
    checked = page.evaluate(
        """(t) => {
            const root = document.querySelector('.st-key-anomaly_train_semantic') || document;
            // 只取「含 radio input」的 option label(排除群組問題 label)
            const opts = Array.from(root.querySelectorAll('label'))
                .filter(l => l.querySelector('input[type="radio"]'));
            const lab = opts.find(l => (l.innerText || '').includes(t));
            if (!lab) return null;
            const inp = lab.querySelector('input[type="radio"]');
            if (!inp.checked) lab.click();
            return true;
        }""", option_text)
    assert checked, f"找不到 label 語義 radio 選項:{option_text}"
    page.wait_for_timeout(300)
    wait_idle(page)
    page.wait_for_timeout(500)
    # 確認實際 checked 的選項文字含目標(切換真的生效,避免 silent no-op → head 沒訓)
    ok = page.evaluate(
        """(t) => {
            const root = document.querySelector('.st-key-anomaly_train_semantic') || document;
            const opts = Array.from(root.querySelectorAll('label'))
                .filter(l => l.querySelector('input[type="radio"]:checked'));
            return opts.some(l => (l.innerText || '').includes(t));
        }""", option_text)
    assert ok, f"語義 radio 未切到『{option_text}』(實際選項未 checked)"


def set_model_dir(page, model_dir) -> str:
    """① 把「模型暫存目錄」(anomaly_model_dir)填成指定路徑並提交(Enter),回傳填入的值。

    ⚠ 真實 app 行為:text_input 在「還沒加訓練資料夾」的首次 render 就先建好 widget,
    此時 `value=_def_dir` 是空字串(_anomaly_bank_default_dir("")=="")→ key 進 session_state
    後,之後 rerun 的 `value=` 一律被 Streamlit 忽略 → 欄位永遠空 → 💾 存模型鈕 disabled。
    所以(像真實使用者一樣)要先在這欄位「打一個路徑」才存得了。填 `.lv_cache` 下的路徑
    以守 no-dataset-writes。先切回①(欄位在①),填完按 Enter 觸發 on_change rerun。"""
    click_tab(page, TAB_BUILD)
    inp = page.locator('.st-key-anomaly_model_dir input').first
    inp.wait_for(state="visible", timeout=30000)
    inp.click()
    inp.fill(str(model_dir))
    inp.press("Enter")
    wait_idle(page)
    page.wait_for_timeout(400)
    # on_change rerun 會把 st.tabs 重置回①(本就是①,仍重切確保),並等存模型鈕 enabled
    click_tab(page, TAB_BUILD)
    page.wait_for_function(
        """() => { const b = document.querySelector('.st-key-anomaly_save_model_btn button');
                   return b && !b.disabled; }""", timeout=30000)
    return page.locator('.st-key-anomaly_model_dir input').first.input_value()


def build_model(page, train_root, semantic_text: str | None = None):
    """① 加訓練資料夾 → (可選)設語義 → 按「▶ (1) 建立模型」→ 等模型狀態(✅ 模型已建立)出現。
    回傳模型成功訊息所在的 main 容器文字。首跑含模型載入,給足 timeout。"""
    click_tab(page, TAB_BUILD)
    _add_folder(page, "anomaly_train_folder", str(train_root))
    wait_idle(page)
    if semantic_text is not None:
        set_semantic(page, semantic_text)
    # _add_folder / set_semantic 的 rerun 會把 st.tabs 重置回①(雖①本就是預設,仍重切確保)
    click_tab(page, TAB_BUILD)
    btn = page.locator('.st-key-anomaly_build_btn button')
    btn.wait_for(state="visible", timeout=30000)
    # 建模鈕可能被 disabled(資料夾還沒寫進 session)→ 等到 enabled
    page.wait_for_function(
        """() => { const b = document.querySelector('.st-key-anomaly_build_btn button');
                   return b && !b.disabled; }""", timeout=30000)
    btn.click()
    # 等建模完成:成功會出現「✅」「模型已建立」字樣(含模型載入,給足時間)
    page.wait_for_function(
        """() => { const m = document.querySelector('[data-testid="stMain"]');
                   if (!m) return false;
                   const t = m.innerText || '';
                   return t.includes('模型已建立') || t.includes('建模失敗')
                          || t.includes('找不到 YOLO'); }""",
        timeout=240000)
    wait_idle(page, timeout=60000)
    page.wait_for_timeout(800)
    return page.locator('[data-testid="stMain"]').inner_text()


def apply_model(page, target_root):
    """② 切 tab → 加目標資料夾 → 按「▶ (2) 套用偵測」→ 等散點圖出現。"""
    click_tab(page, TAB_APPLY)
    _add_folder(page, "anomaly_target_folder", str(target_root))
    wait_idle(page)
    # ⚠ _add_folder 的 rerun 把 st.tabs 重置回① → 重切回②,否則套用鈕在隱藏 panel 內點不到
    click_tab(page, TAB_APPLY)
    btn = page.locator('.st-key-anomaly_apply_btn button')
    btn.wait_for(state="visible", timeout=30000)
    page.wait_for_function(
        """() => { const b = document.querySelector('.st-key-anomaly_apply_btn button');
                   return b && !b.disabled; }""", timeout=30000)
    btn.click()
    # 套用是長時運算(冷啟動載模型)。先等運算結束(idle),再(若 rerun 重置了 tabs)重切回②,
    # 然後等散點圖在 ② panel 真的可見。
    page.wait_for_timeout(1500)
    wait_idle(page, timeout=240000)
    click_tab(page, TAB_APPLY)
    # ⚠ 鎖定 ② context 的散點(_apply);st.tabs 把 tab① 的 _build 散點留在 DOM 但隱藏,
    # 用太廣的 [class*=anomaly_scatter].first 會抓到隱藏的 _build → timeout。
    page.locator('[class*="st-key-anomaly_scatter_apply"]').first.wait_for(
        state="visible", timeout=120000)
    wait_idle(page, timeout=60000)
    page.wait_for_timeout(1200)
