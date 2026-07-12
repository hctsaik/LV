# M10-GUI 接線技術設計:離線監看服務(設定 / 匯出 / 立即掃描 / 佇列消費)

對應 PRD:[2_PO_PRD/al_scale_offline_prd.md](../2_PO_PRD/al_scale_offline_prd.md) · 里程碑 **M10** · Tier B(GUI → 真實 E2E)
消費:**10 `al_service`**(run_once / init_workspace / status)+ **09 `al_workspace`**(read_queue / append_label)。引擎公開函式零改。

## 1. 目的(Purpose)

在瑕疵偵測 GUI 加「🛰 持續監看」區塊:用①存出的凍結模型 + ②目標資料夾**初始化一個服務工作區**(可匯出設定檔給離線服務)、
**在畫面按「立即掃描一次」**(in-process 重用 `al_service.run_once`)、看**服務狀態**、消費**監看佇列**(逐張看理由 → 標註 → 下輪移出)。

## 2. 進入點與資料流(重用既有,不重造)

在 ② `_anomaly_tab_apply` 內、`_anomaly_batch_section` 之後,加 `_anomaly_watch_section(model, target_folders)`。
- **工作區目錄**:`session_state["anomaly_watch_ws"]`(text_input,預設 `dataset_cache_dir(target_root, "al_watch")` → .lv_cache)。
- **初始化/存設定**(`anomaly_watch_init_btn`):`al_service.init_workspace(ws, name=<target basename>, watch_folders=[target_root],
  model_dir=model["_dir"], objective=<選單>, k=<slider>)`。**未存模型 → 先自動存**(同 M9-GUI R1 拍板)。
- **📤 匯出設定**(`anomaly_watch_export_btn`):`profile.yaml` 即可攜設定;提供 `st.download_button` 下載其內容(檔名 `al_profile_<name>.yaml`)。
- **▶ 立即掃描一次**(`anomaly_watch_scan_btn`):`al_service.run_once(ws, progress=...)` in-process(阻塞 + 進度,同 M9-GUI 的 callback 模式);
  完成後 `st.session_state["anomaly_watch_last"] = 結果`。**object_source 從磁碟**(al_service 內部已如此)。
- **服務狀態卡**(`anomaly_watch_status`):`al_service.status(ws)` → 上次 run 摘要(new/scored/queued)、佇列長度、鎖狀態;
  **上次成功時間**過久 / **模型版本不符**(profile.model_dir 的 model_version vs GUI 目前模型)→ 顯眼警告(PRD Must「服務健康可見」)。
- **監看佇列消費**(`anomaly_watch_queue`):`al_workspace.read_queue(ws)` → 縮圖牆(重用 crop_bbox+st.image,用 `reason`)+
  每卡三鍵 `anomaly_watch_good_<id>` / `anomaly_watch_defect_<id>` / `anomaly_watch_skip_<id>` →
  `al_workspace.append_label(ws, {"id":..., "decision":..., ...})` → 樂觀即時從畫面移除;分頁(重用 viz_grid_limit idiom)。

## 3. 邊界 / 風險

| # | 風險 | 處理 |
|---|------|------|
| R1 | 未存模型 → al_service 讀不到磁碟模型 | init/scan 前檢查 `model["_dir"]`,None → 自動存(拍板 auto-save;同 M9-GUI)|
| R2 | 工作區落在來源資料夾內 | `al_service.init_workspace` 內 `assert_safe_workspace` raise → 明確錯誤,不靜默 |
| R3 | run_once 阻塞 rerun | 同 M9-GUI:callback + st.progress(阻塞一次 rerun,顯示進度);續掃靠外部排程或重按 |
| R4 | 佇列 id 消費 | 卡片以 queue item 的 `id`(=al_batch item_id)標註;append_label 用同 id → 下輪 al_service merge 移出 |
| R5 | 單寫者 | GUI 只 `append_label`(不碰 queue/state);「立即掃描」= 服務端寫 queue。互不踩踏(憲法 C5)|

## 4. Acceptance Criteria(真實 Playwright E2E;`@pytest.mark.e2e`;給 /pm)

> 小合成資料 + 小資料驗接線行為(非 literal 規模);重用既有 anomaly E2E harness。

- **AC-W1(初始化 + 匯出設定)**:①建模+存 → 設工作區目錄 → 「初始化監看」→ **工作區出現 `profile.yaml`**(磁碟斷言);
  「📤 匯出設定」下載鈕**可見且啟用**(真實可下載,非僅元素存在)。
- **AC-W2(立即掃描 → 佇列渲染)**:「▶ 立即掃描一次」→ 完成後**監看佇列渲染 ≥1 張縮圖卡**,每卡**可見理由(reason)文字**;主畫面出現「掃描完成」。
- **AC-W3(標註閉環:下輪移出)**:對佇列某卡按「✅ 正常」→ `labels.jsonl` 多一行(磁碟斷言)→ 再「立即掃描一次」→ **該卡的 id 不再出現在佇列**。
- **AC-W4(服務狀態卡)**:掃描後狀態卡**可見上輪計數**(new/queued 數字為真實數,非佔位);未掃描前顯示「尚未跑過」之類提示。
- **AC-W5(未存模型自動存)**:①建模但未按存 → 「初始化監看」或「立即掃描」→ 自動存(工作區 profile 的 model_dir 目錄出現 meta.json),無原始 exception。

> AC-W3 是本里程碑閉環的核心(標註真的讓服務下輪不再煩你);AC-W2 的 reason、AC-W1 的 profile.yaml 是防 false-green 的真實斷言。

## 5. 給 /pm 的提示

- E2E 放 `tests/e2e/test_al_service_gui_e2e.py`(`@pytest.mark.e2e`),重用 `_anomaly_wizard.py`(build_model/set_model_dir)。
- 小合成 yolo/whole_image 資料 + 小 k;`al_service.run_once` 走真實 DINOv2(同 M9-GUI E2E,可接受)。
- widget key 契約(PG 照實作):`anomaly_watch_ws / anomaly_watch_init_btn / anomaly_watch_export_btn / anomaly_watch_scan_btn /
  anomaly_watch_status / anomaly_watch_queue / anomaly_watch_good_<id>`;DOM 標記「掃描完成」。
- 契約鎖點:profile.yaml 真的產生(AC-W1)、佇列 reason 真渲染(AC-W2)、標註下輪移出(AC-W3)、狀態計數為真(AC-W4)是防 silent-wrong/false-green 核心。
- 此為整合層接線,無單元 gate;done = 上述 E2E 全綠。
