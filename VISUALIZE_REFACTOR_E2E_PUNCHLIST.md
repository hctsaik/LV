# Visualize 重構 E2E 收尾清單（punch-list）— ✅ 已收尾

> 背景:`scripts/app.py` 的 Visualize 工具(資料夾系統改版、多模型、UI-state 持久化、
> 面板/視圖、工具列精簡)做了大重構,**先前從未跑過完整 E2E**。先前進度把
> `tests/e2e/test_scenarios_r1.py` + `tests/e2e/test_gui_flows.py` 的失敗從 **35 → 12**。
>
> **本輪把剩餘 12 個全部收掉,整批 E2E 綠(35 passed / 5 skipped / 0 failed)。**
> 5 skip 皆為刻意停用:Compare 影像級流程(test_s05 / test_s)、工具列精簡隱藏的
> 組考卷/灰帶/評估(test_y / test_z / test_ac)。

## 怎麼跑
```
python -m pytest tests/e2e/test_scenarios_r1.py tests/e2e/test_gui_flows.py -m e2e -o addopts="" -rA --tb=short
```
注意:`det_page`(scenarios_r1)與 `flow_page`(gui_flows)是 module-scope 共享頁,
測試**依檔案順序連鎖**;s01 / test_b 跑完會自動收合 sidebar。整批冷跑約 4–7 分(含模型載入)。

## 12 個失敗的根因與修法（本輪）

| 測試 | 根因 | 修法 | 類別 |
|------|------|------|------|
| `s01` | 預設模型由多模型改成只剩 `dinov2_vitb14`(app.py:3126);Run 後 `viz_model_select` 只列「已實際計算」的模型 → 切 vits14/chinese-clip 找不到選項。且圖內標題已刻意移除(app.py:694),舊斷言找不到散點區的模型名 | 加 `_add_models` helper,Run 前先在側欄 `viz_models_sel` 加 vits14+chinese-clip;斷言改驗 `viz_model_select` 下拉值 + 散點重繪;結尾還原預設模型避免污染 s02–s04 | test |
| `s02` | `viz_img_boxes` 預設改為 True(app.py:1499);測試先點掉(True→False)卻斷言 `is_checked()==True`,方向錯 | 擷取點擊後狀態,斷言「跨影像導覽後狀態保留」(忠於原意) | test |
| `s03` | 連鎖:s01 無法切模型 + s01 結尾停在 chinese-clip(散點塌縮、點不到) | 隨 s01 修法(vits14 已算 + 還原預設模型)解決 | test(連帶) |
| `s04` | 切 grid 排序(標籤分歧→空間順序)後 rerun 尚未重新註冊卡片 callback,點到陳舊 on_click args → viewer 開在中段(13/18)而非 1/N | 等狀態翻回「離群度前」+ retry-click 直到 viewer 開在 1/N | test |
| `s07` | 「切粒度清 `viz_records`」邏輯在**切模式**時誤觸發(Classifier→Detector 連帶改 granularity),把 records 偷走 → 切模式快照沒建 → 「已切換模式」banner/復原鈕不顯示 | app:切模式那一輪不在粒度區塊清,留給切模式區塊先快照再清 | **app** |
| `j` | 3D 高亮環(黑圈)仍實作(app.py:3606-3614),但描述用 caption 在重構時掉了 | app:補回 caption `3D 看：黑圈為目前選取的 N 點` | **app** |
| `l` | Run 後 sidebar 以 CSS `visibility:hidden` 收合;`_ensure_sidebar` 點 reopen 後只 `wait_idle`,沒等側欄真的展開 → run_viz 仍隱藏、點擊 20s timeout | conftest:`_ensure_sidebar` 改為「等側欄真的可見 + 重試 reopen」(共享 harness 強化) | conftest |
| `m` | 裸 `page.mouse.click` 點 plotly marker 偶爾 miss,狀態不翻「已選取」 | test:retry 點擊(只計成功命中那次延遲,SLA 意圖不變) | test |
| `o` | 把 `viz_methods` multiselect 縮到只剩 PCA:Backspace 與逐 tag 刪除都會被 rerun 吞掉/誤刪(時而剩 2、時而連 PCA 都刪光) | test:改用 baseweb「Clear all」清空 + 重新加 PCA(確定性) | test |
| `q` | 用共享 `flow_page`,但 test_b 的 Run 只算 vitb14 → 切 chinese-clip 找不到選項;且切模型 rerun 期間 panel segmented-control 暫態渲染兩份 | test:改用獨立 `app_page` + 自帶 Run 並預選 chinese-clip;`_switch_panel` 先沉澱再取 `.first` | test |
| `t` | umap 參考系是 **per-model**;預設只跑 vitb14,但測試查 `ref_path_for(train,"dinov2_vits14")` | test:Run 前加選 `dinov2_vits14` | test |
| `w` | 候選池 popover 只剩原生 📁 picker,**沒有 headless 可填的 `cov_pool_text`**;另校正警告在 t_abs rerun 時暫態渲染兩份 | app:popover 補回 `st.text_area(key="cov_pool_text")`;test:校正警告斷言加 `.first` | **app**+test |

## app.py 真實行為修復(4 處,非僅測試對齊)
- `s07`:切模式不再被「切粒度清除」偷走 `viz_records`(banner/一鍵復原恢復)。
- `j`:補回 3D 高亮環的 caption(行為一直在,描述掉了)。
- `w`:候選池補回 headless 文字輸入(原生 picker 在 headless/E2E 不可填)。

> 其餘多為**測試對齊重構**(預設模型、移除圖內標題、UI 時序)或 **E2E harness 強化**
> (`_ensure_sidebar` 等待側欄真的展開、面板切換沉澱 + `.first`、互動 retry)。
> **無任何 skip/xfail/放寬斷言來假綠**;app 改動只動 Visualize/Completeness UI,
> 不影響單元 gate。
