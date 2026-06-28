# M7 — 瑕疵偵測頁面重設計(引導式 wizard)技術設計

> 來源:使用者 UX 回饋 7 項 + 多 agent 設計鎖定(wgnvly15j)。對象:`scripts/app.py` 的 `_anomaly_ui` 整段重寫。
> 引擎模組(anomaly_tool / anomaly_bank_store / dino_head / anomaly_project / active_learning / active_loop)**公開函式零改動**,僅 anomaly_bank_store.save_bank 的 meta 多帶 `label_semantic`。

## 目標(使用者回饋)
1. 三子分頁 wizard + 按鈕編號 (1)(2)(3):① 建模/載入 → ② 套用偵測 → ③ 挑樣送人工標。
2. 分兩個資料夾:訓練/參考(①)vs 異常目標(②)。
3. 統一模型暫存目錄:選一次 → 按鈕一鍵存 bank + classifier。
4. 取樣佇列 4 模式 → 2×2 矩陣 master-detail(點格 → 右邊大圖牆)。
5. 影像放大 + 更多張 + 有意義標籤。
6. 移除「主動學習迴圈/標註效益曲線」(active_loop 模組函式保留,GUI 不再呼叫)。
7. 拆「進階」popover:語義→提升①白話一步、模式覆寫→移除、N_min→收進①進階。

## 結構
```python
def _anomaly_ui():
    # 標題 + ❓說明(照搬)
    _anomaly_sidebar_settings()   # sidebar 留 model/score_mode/res/mode/sample_n/N_min(跨 tab 共用)
    tb, ta, ts = st.tabs(["① 建模 / 載入模型", "② 套用偵測", "③ 挑樣送人工標"])
    with tb: _anomaly_tab_build()
    with ta: _anomaly_tab_apply()
    with ts: _anomaly_tab_sample()
```
Streamlit st.tabs 每次 rerun 三 tab 全跑 → 跨 tab 狀態**一律落 session_state**;每個 tab function 開頭前置守門(讀不到模型/結果就顯示引導文案 return,不互相重算)。run 只由按鈕 callback 觸發。

## `anomaly_model`(唯一 in-session 模型槽;① built 與 📂 loaded 都寫這,② 永遠讀這)
```python
{"source": "built"|"loaded", "ref_folder": str, "_built_at": float,
 "meta": {"model","target_res","score_mode","patch_dim","obj_dim","n_objects"},
 "label_semantic": "object"|"defect",
 "bank_vectors": np.ndarray|None,   # patch coreset → external_bank
 "projection": dict|None,           # anomaly_project 投影器 + good_obj_emb → external_ref/散點灰底
 "fewshot": list|None, "head": dict|None, "_dir": str|None}
```

## session_state keys(單一真相)
- `anomaly_model`(取代 anomaly_loaded_bank+anomaly_head)、`anomaly_model_dir`、`anomaly_train_folder`、`anomaly_target_folder`、
  `anomaly_train_semantic`(白話 radio,每 rerun 同步寫 `anomaly_active_semantic` 供 _anomaly_train_head 硬守衛)、
  `anomaly_train_result`(①框選標 good 用)、`anomaly_apply_result`(取代 anomaly_result;②③讀)、`anomaly_confirmed`(沿用)、
  `anomaly_q_mode`('novelty'|'confusion'|'balanced'|'pure',預設 novelty)、`anomaly_q_k`、`anomaly_q_mpc`、
  `anomaly_q_cols`(圖牆每列 2-6 預設3)、`anomaly_q_th`(縮圖高 120-400 預設200)、`anomaly_n_min`、`_anomaly_model_err`、`_anomaly_model_saved`。
- 移除:anomaly_folder、anomaly_result、anomaly_loaded_bank、anomaly_head、anomaly_phase_override、anomaly_loop_*、anomaly_al_mode、anomaly_sem_<sig>。

## 新/改 callback(scripts/app.py)
- `_anomaly_build_model()`:① 用 train_folder 跑 run_pipeline(confirmed=anomaly_confirmed)→ 凍 bank/projection/fewshot;若 semantic=="defect" 且 head_unlock_state.unlocked → train_head(eligible) → 一起包進 anomaly_model(source="built", _built_at=time())。**語義硬守衛**(非 defect 不訓 head)。結尾 pop anomaly_apply_result(舊結果 stale)。
- `_anomaly_apply_model()`:② 用 target_folder 跑 run_pipeline,external_bank=model.bank_vectors / external_ref=model.projection.good_obj_emb,model/target_res 鎖 model.meta(非側欄現值)→ anomaly_apply_result(帶 _model_built_at=model._built_at)。
- `_anomaly_save_model(dir)`:save_bank(bank/projection/fewshot/meta+label_semantic)+ save_head(dir/head.joblib)+ 寫 manifest.json(model/res/semantic/類別/built_at)到同一 dir。
- `_anomaly_load_model(dir)`:load_bank + load_head + 讀 manifest → 填 anomaly_model(source="loaded")。舊 bank 無 label_semantic → 預設 "object"(安全側),①提示可重宣告。
- `_anomaly_pick_qmode(mode_id)`:僅設 anomaly_q_mode。
- `_anomaly_priority_for(mode_id, scores, proba, threshold)`:novelty→priority_score(1,.4,.4);balanced→(1,1,1);pure→(1,0,0);confusion→confusion_targeted_priority(scores,proba,1,1)(proba=None 退純 novelty)。
- `_anomaly_queue_labels(sel, records, scores, head, gthr, min_conf)`:每 idx → {label:類別名, verdict:閘控判定(head 有→gated_predict 正常/類別/Unknown;無→record.verdict 對映 可疑/正常), score:異常分數}。

## ③ 2×2 取樣矩陣(master-detail)
- 4 格固定順序(對齊現 radio):(0,0)偏novelty (0,1)弱類定向 (1,0)三訊號均衡 (1,1)純novelty。`r0=st.columns(2);r1=st.columns(2)`,格內一顆 button(模式名 + 徽章),選中格 type="primary"。
- **各別數量徽章(使用者選:佇列成分拆解)**:每格算自己 _sel(該模式 select_for_labeling 取 K)→ 有 head 用 gated_predict 拆「正常N·類別M·未知U」;無 head 拆「可疑N·正常M」(誠實:無 head 不假裝 Unknown)。
- 右側放大圖牆(detail)= 選中模式的 _sel:`st.columns(anomaly_q_cols)` 排版、縮圖 height=anomaly_q_th、可拉 slider 調欄數/高度看更多。
- **影像標籤(使用者選:類別 + 閘控判定 + 異常分數)**:每張三行 `類別名 / 判定:正常·類別·Unknown / 異常 0.XX`(取代 P=score·verdict)。

## 鎖定的 open_decisions(已拍板)
- tab 切換:用原生 st.tabs + 純文案引導(無程式切 tab API);② 無模型/結果顯「請先到①/②」+ 文案。
- ② 框選標記:② 保留就地重評(快回饋,用 confirmed 重算但不動模型);正式重建走 ①(2)。
- 舊 bank 無 label_semantic:有 manifest(新版)一定有;純舊 bank 預設 object(安全)+ ①可重宣告。
- 舊「💾 存成 bank」expander:由 ①(2) 統一取代、移除。
- 側欄 model/res:維持可改(只作用於下次①build);② caption 明示「用模型鎖定的 model@res 非側欄現值」。

## 移除清單
- callback `_anomaly_run_loop_curve`;🔁 主動學習迴圈 expander 整段;⚙進階 popover(語義上移、覆寫刪、N_min 移①);取樣佇列一排 4-radio;sidebar 的執行偵測鈕(移②(3))。
- E2E `test_active_loop_e2e.py` 整檔刪(驗的是被移除的曲線面板)。active_loop.py 模組函式**保留**(GUI 不呼叫,單元測試 test_active_loop.py 仍覆蓋)。

## 驗收
- 單元:gate anomaly_tool 等純邏輯 GREEN。
- E2E(新 tests/e2e/test_anomaly_wizard_e2e.py):①建模存模型 → ②載入套用(兩資料夾)→ ③ 2×2 點格出圖 + 標籤;語義守門(object→①不訓 head)。既有 anomaly E2E 改 tab 化後綠;flaky 隔離照 visuallatent-e2e skill。done = 單元綠 AND E2E 綠。
