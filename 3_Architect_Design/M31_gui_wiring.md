# 設計:M31 GUI 接線(`class_similarity_ui` + `app.py`)

> Tier B。實作:`scripts/class_similarity_ui.py`，最小改 `scripts/app.py`。

## 分派

`_render_label_quality` 入口用持久 segmented selector:`類別混淆` | `疑似標錯` | `類別相似度分布`。分派在舊「<3 筆或 <2 類 return」**之前**。切到新面板不跑 `neighbor_confusion`。

## 面板

計算按鈕才跑 analyze。狀態 empty/computing/ready/stale/error。
Box 預設 margin，軸 [-2,2] 含 y=0; same/other 軸 [-1,1]。顯示縮放不改分數。
散點抽樣規則見 r3 §7.2; customdata 含 analysis_key, record_id, original_index。
點一下=inspect; 框選才併入 local_selected_ids。同步至 LV 選取才改 viz_selection。
Threshold: 單一 canonical; 拖動更新 draft; 未 commit 停用加入全部候選／保存／下載。
action `{type, ids, analysis_key, preview_key?, request_seq?}`。

## app.py 最小接線

generation sidecar、兩條 readback stale、MODE_CLEAR_KEYS、restore 後 event generation++、消費 sync/cart。不呼叫灰帶。不改 Labeling。

## AC

AC07, AC10, AC12(UI call count), AC13, AC14, AC16, AC18, AC25, AC26, AC33, AC34(E2E 部分)。
