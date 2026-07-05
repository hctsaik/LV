# M10-GUI 接線 — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/M10_gui_wiring.md](../3_Architect_Design/M10_gui_wiring.md) · 里程碑 **M10** · Tier B(GUI)
E2E:`tests/e2e/test_al_service_gui_e2e.py`(`@pytest.mark.e2e`,交 /ux-test 或人觸發)
消費:10 al_service(run_once/init_workspace/status)+ 09 al_workspace(read_queue/append_label)。

## widget key 契約(PG 實作;先紅)

`anomaly_watch_ws`(工作區目錄)/`anomaly_watch_init_btn`/`anomaly_watch_export_btn`(download_button)/
`anomaly_watch_scan_btn`/`anomaly_watch_status`/`anomaly_watch_queue`/`anomaly_watch_good_<id>`(每卡✅正常);
DOM 標記掃描完成→主畫面含「掃描完成」。

## E2E AC ↔ 測試(小合成資料驗接線)

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC-W1 | 初始化監看 → profile.yaml 產生;匯出設定鈕可見 | `test_w1_w2_init_export_scan_queue` | 🔴 |
| AC-W2 | 立即掃描 → 監看佇列渲染 ≥1 卡 + reason 文字;queue.jsonl 產生 | `test_w1_w2_init_export_scan_queue` | 🔴 |
| AC-W3 | 標「正常」→ labels.jsonl 多一行 → 再掃 → 該 id 不在 queue(閉環) | `test_w3_label_removes_next_round` | 🔴 |
| AC-W5 | 未存模型 → 監看掃描自動存(model_dir meta.json 出現)、無 exception | `test_w5_autosave` | 🔴 |

> AC-W4(服務狀態卡計數)由掃描後主畫面文字間接涵蓋;完整狀態卡細節列 Should,PG 接好後 /ux-test 可補。

## done 判準(Tier B,GUI)

- `pytest tests/e2e/test_al_service_gui_e2e.py -m e2e` 全綠(AC-W1/W2/W3/W5)。
- 跑前清殘留 Streamlit server;偶發紅先單獨重跑判別。
- 真實行為鐵則已落各測(profile.yaml/queue.jsonl/labels.jsonl 真的產生、reason 真渲染、閉環用檔案斷言)。
- **M10 里程碑完成** = 09 al_workspace gate 綠 + 10 al_service gate 綠 + 本 GUI E2E 綠。
