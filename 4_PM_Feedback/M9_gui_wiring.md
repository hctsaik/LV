# M9-GUI 接線 — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/M9_gui_wiring.md](../3_Architect_Design/M9_gui_wiring.md) · 里程碑 **M9** · Tier B(整合/GUI)
E2E:`tests/e2e/test_al_batch_gui_e2e.py`(`@pytest.mark.e2e`,交 `/ux-test` 或人觸發,**不進 PG 自主修綠迴圈**)
消費引擎:`scripts/al_batch.py`(08,已 gate GREEN)

> 此為**整合層接線**,無新增純邏輯模組 → **無單元 gate**;done = 下列真實 Playwright E2E 全綠。
> 引擎公開函式零改;PG 只在 `scripts/app.py` `_anomaly_ui` ② 分頁加接線 + 順手修 `_anomaly_load_model` 掉 object_source 的 latent bug。

## v1 範圍(反向閘門 /pg→/architect 修正,2026-07-05)

大資料分批掃描 v1 **只出「標註佇列」**(縮圖+reason+分數直方圖+分頁+購物車);**探索/完整散點模式延後**
(需 al_batch 未來回傳抽樣 obj_emb + 輕投影才可行——大資料本就不該畫全量散點)。E2E 改直驗佇列 + 「無散點」。

## PG 必須實作的 widget key 契約(E2E 依這些定位;先紅)

| key | 位置 / 作用 |
|---|---|
| `anomaly_batch_scan_btn` | ② 「▶ 大資料分批掃描」按鈕 |
| `anomaly_batch_objective` | ② 選樣目標(novelty/balanced/confusion/pure;無 head 反灰後兩者) |
| `anomaly_batch_resume_btn` | ② 「▶ 繼續上次」(續跑 resume=True) |
| `anomaly_batch_queue` | 標註佇列容器(內含 `stImage` 卡 + 每卡 `reason` 文字) |
| `anomaly_batch_scatter` | v1 **不渲染**(此 class 存在數必為 0 = scale-safe);延後的探索/完整模式才會用 |
| DOM 標記 | 掃描完成主畫面出現「✅ 分批掃描完成」(含「掃描完成」) |

## E2E AC ↔ 測試

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC-G1 | 分批掃描 → 標註佇列渲染出 ≥1 縮圖卡,每卡可見 reason 文字 | `test_g1_g2_batch_scan_queue_and_no_scatter` | 🔴 |
| AC-G2 | 標註模式**不渲染散點**(scale-safe);切「完整」才出現散點 | `test_g1_g2_batch_scan_queue_and_no_scatter` | 🔴 |
| AC-G4 | 停止 → 繼續上次 → done,佇列渲染、無例外(續跑等於一次跑完) | `test_g4_resume_equals_full` | 🔴 |
| AC-G5 | 未存模型 → **自動存**(meta.json 出現)再掃,無原始 exception | `test_g5_autosave_before_scan` | 🔴 |
| AC-G6 | whole_image 建模+存+載入 → 分批掃描仍以整張影像掃(物件數==影像數 8),不被默默當 yolo | `test_g6_object_source_from_disk` | 🔴 |

> AC-G3(暫定 Top-K 於掃描中出現 + images_processed 單調增)、AC-G7(購物車)、AC-G8(無 head 反灰):
> G3 的自續 rerun 時序在 E2E 難穩定斷言,其「暫定→續跑→done」核心由 al_batch 單元 AC3/AC13/多次續跑收斂 +
> 本 E2E 的 `_wait_scan_done` 涵蓋;G7/G8 列 Should(appetite 緊可進候選,PG 接好後由 /ux-test 補)。

## done 判準(Tier B,GUI 接線)

- 無單元 gate;done = `pytest tests/e2e/test_al_batch_gui_e2e.py -m e2e` 全綠(AC-G1/G2/G4/G5/G6)。
- 跑前先清殘留 Streamlit server(見 `.claude/skills/visuallatent-e2e/SKILL.md`);偶發紅先單獨重跑判別(長 session 假紅)。
- 真實行為鐵則已落各測(真的渲染佇列卡 + reason 文字、真的無散點元素、meta.json 真的存在、物件數==影像數),非代理式斷言。
