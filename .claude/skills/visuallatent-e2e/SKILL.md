---
name: visuallatent-e2e
description: >-
  對 c:\code\claude\LV\visuallatent 這個 Streamlit 資料集分析工具跑 / 寫 E2E 測試的操作守則。
  E2E 用 pytest + Playwright,harness 在 tests/e2e/conftest.py(對 scripts/app.py 起真實
  Streamlit server + Playwright 連上)。當要跑瑕疵偵測 / anomaly / bank / 分類頭等 E2E、排查
  E2E 失敗(尤其「偶發紅 / 超時」)、或新增 E2E 案例時讀本 skill。**最重要的一條**:跑 E2E
  前先清掉殘留的 Streamlit server,否則資源競爭會造成假失敗。
---

# visuallatent E2E 操作守則(Streamlit + Playwright)

> 對象:`c:\code\claude\LV\visuallatent`(成熟 Streamlit GUI,程式 `scripts/`、測試 `tests/`)。
> E2E harness 在 `tests/e2e/conftest.py`,對 `scripts/app.py` 起真實 Streamlit + Playwright。

## ⚠️ 跑 E2E 前必做:先清掉殘留的 Streamlit server(最常見的假失敗來源)
為了讓使用者試,常手動 `streamlit run scripts/app.py`,多次重啟會**累積背景 server**;每個都載入
DINOv2 模型,一起**搶 CPU / RAM**。後果:Playwright 動作超時、virtualized 下拉(`_ranked_stems`)
讀到空 → **假紅**。曾見一個 E2E 跑 **28 分鐘**超時假失敗,清掉 **4 個**殘留 server 後 **22 秒**乾淨通過。

**所以:跑 E2E(或排查 E2E 偶發紅)前,先清殘留 server,再跑:**
```bash
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -like '*streamlit*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }"
```
E2E 自己會在 free port 起 **session-scoped** server(`conftest.app_server`),不需要你先手動起一個給它。
若你要保留一個給使用者試(例如 port 8533),**跑 E2E 前先把它停掉**,跑完再開。

## E2E 失敗的排查順序(先排除環境,再懷疑程式)
1. **先問:是不是殘留 server 競爭?** 清掉後**單獨重跑該測試**(`pytest "tests/e2e/x.py::test_y" -m e2e -q`)。
   乾淨環境下過 → 是 flaky,不是你的改動破壞。
2. 才看是不是真 bug(讀斷言、`_no_exception(page)`、實際畫面)。
3. 跑時間異常長(平常 ~30-60s/test,變成數分鐘)= 強烈的資源競爭訊號。

## 怎麼跑
- 單元 gate:`python verify/gate.py <module>`(印 GREEN 才算過)。
- E2E:`python -m pytest tests/e2e/test_<x>_e2e.py -m e2e -q`(標 `@pytest.mark.e2e`;gate 用 `-m "not e2e"` 排除)。

## 驅動 Streamlit GUI 的 Playwright 慣用招(都是踩過的坑)
- **元件定位**:Streamlit 給的 key → `.st-key-<key> button` / `.st-key-<key> input`。
- **expander 內按鈕點不到**(收合 / 被另一 `details` 攔截 pointer):用 JS 展開 + JS click,最穩:
  ```python
  page.evaluate("(t)=>{for(const d of document.querySelectorAll('details')){if(d.textContent.includes(t)) d.open=true;}}", text)
  page.evaluate("(s)=>{const b=document.querySelector(s); if(b) b.click();}", sel)   # 直接觸發 React onClick
  ```
- **run 後側欄收合**:`run` 成功會 `_collapse_sidebar()`;要再點側欄的「執行偵測」或側欄控制項前,先點
  `.st-key-reopen_sidebar button` 重開側欄(否則按鈕 not visible)。
- **always 可見的控制項別放側欄**:像「載入 bank」這種 run 後還要用的,放主畫面 `st.container`(放側欄會被收合擋住)。
- **selectbox 虛擬化**:下拉只渲染前 ~8-13 項。要找特定項用鍵入過濾(`page.keyboard.type("defect")`);
  要拿全部排序用「開下拉 + JS 捲動收集去重」。
- **wait_idle**:重用 conftest 的 `wait_idle(page)`(等 Streamlit running-man 消失);長時運算(首跑載模型)給足 timeout。
- **每個測試起獨立 browser context**(`browser.new_context()`)隔離 session_state。
