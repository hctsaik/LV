# M9-GUI 接線技術設計:大資料分批掃描 + 顯示三模式 + 暫定 Top-K + 續跑

對應 PRD:[2_PO_PRD/al_scale_offline_prd.md](../2_PO_PRD/al_scale_offline_prd.md) · 里程碑 **M9** · Tier B(整合/GUI → 真實 Playwright E2E 驗收)
消費引擎:[3_Architect_Design/08_al_batch.md](08_al_batch.md)(`scripts/al_batch.py`,已 gate GREEN)

> 本設計經 workflow 對 `_anomaly_ui` 逐行紮根後定案。**引擎公開函式零改**;M9-GUI 只在 `scripts/app.py`
> `_anomaly_ui` 的 ② 分頁加接線,重用既有 model slot / 資料夾選取器 / 縮圖牆 / 購物車 / 分頁 idiom。

## 1. 目的(Purpose)

讓瑕疵偵測的 ②套用③挑樣在**數萬~十萬物件**下可用:在 ② 加一條「大資料分批掃描」路徑,呼叫 `al_batch.run_batched`
(消費①存出的凍結模型),**跨 Streamlit rerun 分批驅動**(不阻塞、可續跑),並以**顯示三模式**(標註/探索/完整)
呈現,大資料預設走「標註模式」(只佇列+分數直方圖、不算投影、不渲染全量散點)。

## 2. 進入點與資料流(重用既有,不重造)

### 2.1 進入點:② `_anomaly_tab_apply` 內的替代掃描路徑(不新增 wizard step)

在既有「▶ 套用偵測」(`on_click=_anomaly_apply_model`,一次同步跑完 [app.py:1699-1701])**旁邊**加
「▶ 大資料分批掃描」路徑。理由:② 已擁有 al_batch 所需的一切——目標資料夾、image_paths、class_names、鎖定的 model.meta。

### 2.2 al_batch 呼叫的參數來源(全部既有)

| run_batched 參數 | 來源(既有) |
|---|---|
| `model_dir` | **凍結模型磁碟目錄** = `model["_dir"]`(存模型時設 [app.py:1420])。**見 §5 風險 R1:必先存**。 |
| `image_paths` | `② 目標資料夾` → `list_images(root)`(同 `_anomaly_apply_model` [app.py:1338]) |
| `class_names` | `classes_for(root)`(同 ② [app.py:1342]) |
| `object_source` | **從磁碟 meta.json 讀**:`load_bank(model_dir)["meta"].get("object_source","yolo")`——**不可用 in-memory 載入模型的 meta**(載入會掉這個鍵,§5 R2) |
| `checkpoint_dir` | `dataset_cache_dir(target_root, f"al_batch_{objective}")`(同 `_anomaly_bank_default_dir` [app.py:1206])→ 保證 `.lv_cache`,過 `assert_safe_bank_dir` |
| `dataset_dirs` | 目標資料夾 roots(給安全檢查用) |
| `objective` | 由顯示模式/目標選單決定(§3.4 對映) |
| `k` | 預設 **100**(PRD 拍板);GUI slider 可改(重用 `anomaly_q_k` 精神) |
| `batch_size` | 每 rerun 的批預算(§4;預設如 8-16 影像/批,可調) |
| `max_batches` | **每次 rerun 只跑 N 批**(§4 跨 rerun 驅動的核心) |
| `resume` | `True`(續跑) |
| `embed_fn`/`extractor` | `None`(al_batch 內部惰性建真模型,同 run_pipeline 預設 [al_batch.py:131-154]) |
| `progress` | **dict 消費 callback**(§4;**不可**重用 ② 的 `(frac,text)` lambda,契約不同 [app.py:1362]) |

## 3. 顯示模式

> **v1 範圍修正(2026-07-05,反向閘門 /pg→/architect)**:PG 實作時發現「探索/完整=散點」需要 `obj_emb`/投影,
> 但 `al_batch` 的輸出只有 `topk_records`(無 embedding、無投影)——而**大資料本就不該畫全量散點**(標註模式存在的理由)。
> 故 **v1 大資料分批掃描只出「標註模式」佇列**(§3.1);**探索/完整散點模式延後**(需 al_batch 未來回傳抽樣 obj_emb + 一個
> 輕投影才可行,列 M9 候選/M10 後)。小資料仍走既有 ②「▶ 套用偵測」的散點(不受影響)。以下 §3.2/§3.3 為未來設計、v1 不實作。

### 原三模式構想(§3.2/§3.3 延後)

新增顯示模式切換(seeded session_state,key 如 `anomaly_view_mode`,選項 `["標註","探索","完整"]`)。
**大資料自動建議**:當 `objects_scored`(掃描回報)> 5000 → 預設「標註」(§3.1);< 5000 可用「完整」。

### 3.1 標註模式(預設大資料;佇列 + 分數直方圖,**不算投影**)

- **佇列渲染**:重用 ③ 縮圖牆區塊 [app.py:1879-1904],但餵 `al_batch` 的 `topk_records`。
  每筆已含 `image_path/bbox/label/score/reason/priority` → `safe_open_image → crop_bbox → st.image` 路徑不變。
  **caption 用 `rec["reason"]`**(al_batch 自帶人話理由),**不呼叫 `_anomaly_queue_labels`**——它需 `obj_emb`,
  而 checkpoint 只存 `head_proba`、不存 `obj_emb`([al_batch.py:337];§5 R3)。
- **分數直方圖**:對 topk(或 objects_scored)的 `score` 畫 `st.bar_chart`/小 plotly histogram。**無散點、無投影**。
- **分頁**:重用 `viz_grid_limit + 載入更多` idiom [app.py:900-902, 2396-2398](K=100 但使用者可多看);
  遵守 repo Streamlit 效能鐵則(rerun 內禁全量 I/O 迴圈、分頁、禁 sleep-rerun 輪詢)。
- **加入購物車**:重用 `_anomaly_add_to_cart` [app.py:1156],但**先注入 `rec["path"]=rec["image_path"]`**
  (購物車以 `records[i]["path"]` 去重 [app.py:934];物件級 image_path/bbox/obj_index 由 `_export_entry` 自動辨識 [app.py:918-923];§5 R3)。

### 3.2 探索模式(抽樣散點)

- 重用 `_anomaly_render_scatter` [app.py:1907],但傳**抽樣後的 sub-result**:**強制納入 Top-K(與 verdict=='bad')**
  → 其餘隨機/分層抽樣補到**上限 5000** → 組一個 records/scores/obj_emb **索引重排且對齊**的縮小 result。
  加「僅顯示抽樣 N/總 M」caption(對映既有 cap 提示 [app.py:2368-2370])。散點的 selection→index 對映需保持對齊(§5 R6)。

### 3.3 完整模式(現況,小資料 opt-in)

- 對**完整** result 呼叫 `_anomaly_render_scatter(result, context="apply")`,與 ② 現況逐字相同 [app.py:1732]。明確標為小資料 opt-in。

### 3.4 顯示模式/目標 → al_batch objective 對映

- 沿用 ③ 既有目標語彙(novelty / balanced / confusion / pure)映射到 al_batch:
  `novelty→"novelty"`、`balanced→"uncertain"`、`confusion→"confusion"`、`pure→"novelty"`。
- **head 閘控**:`uncertain`/`confusion` 需凍結模型有 `head.joblib`,否則 `run_batched` raise [al_batch.py:277-278]。
  UI 比照 ③ 既有「無 head 反灰 head-依賴模式」[app.py:1850-1868] 反灰這兩個目標;無 head 只給 `novelty`、
  且 `diversity_applied=False`(佇列為純 novelty 排序,無 per-class 分散)。

## 4. 跨 Streamlit rerun 驅動(自續 pattern,無 sleep 輪詢)

> 既有無跨 rerun 長任務迴圈;引入 sanctioned「flag + st.rerun 自續」(**非** time.sleep 輪詢 [app.py 既有 flag→st.rerun(scope='app') pattern 3270-3271])。

- session_state flag `anomaly_al_running`。啟動掃描 = 設 flag。
- **每次 script run 若 flag 開**:呼叫 `run_batched(..., resume=True, max_batches=N, progress=cb)`。
  `cb`(dict 消費)更新 `st.progress`/`st.status` + 暫存 `provisional_topk`。**立即用回傳 `topk_records` 渲染「暫定 Top-K」**。
  - `result["provisional"]`(未 done)→ `st.rerun()`(下一輪處理下一批;這是唯一免違規的續跑,批間讓出 UI)。
  - `done` → 清 flag,顯示最終佇列。
- **「▶ 大資料分批掃描」**:設 flag(新掃描,checkpoint 沿 resume_identity;參數變則 al_batch 依 `on_identity_mismatch` 處置)。
- **「⏸ 停止」按鈕**:清 flag(保留 checkpoint,可續)。
- **「▶ 繼續上次」**:先 `read_checkpoint(checkpoint_dir, k=100)` **即時**顯示現有 Top-K [al_batch.py:367],再設 flag 續跑。
- **防無限迴圈鐵則**:done 清 flag、**例外時也清 flag**(try/finally 或旗標守衛)、有停止鈕。三者缺一不可。

## 5. 邊界條件與風險處理(紮根揪出的真風險,設計層先擋)

| # | 風險 | 處理 |
|---|------|------|
| **R1** | al_batch 讀**磁碟**凍結模型;剛①建好未存的模型 `model["_dir"]==None` [app.py:1300-1306] → 無 model_dir 可傳 | 「大資料分批掃描」啟動前檢查 `model["_dir"]`:None → 先自動呼叫 `_anomaly_save_model`(存到 `anomaly_model_dir`)**或**明確擋下並提示「請先💾存模型」。不可靜默失敗。 |
| **R2** | 📂 載入模型會**掉 `object_source`**(in-memory meta 只留 6 鍵 [app.py:1452-1455]),② 之後默默當 yolo [app.py:1335] | 分批掃描的 `object_source` **一律從磁碟 meta.json 讀**(`load_bank(model_dir)["meta"]`),不信 in-memory。**附帶修**:`_anomaly_load_model` 的 meta comprehension 補回 `object_source`(修既有 latent bug;屬 GUI 接線範圍內順手修,E2E 涵蓋)。 |
| **R3** | `topk_records` 無 `path/verdict/split` 鍵;caption 需 obj_emb(未存) | 佇列 caption 用 `rec["reason"]`;加入購物車前注入 `rec["path"]=rec["image_path"]`。 |
| **R4** | `checkpoint_dir` 落在來源資料夾 → `assert_safe_bank_dir` raise、整輪擋死 [al_batch.py:272] | 一律用 `dataset_cache_dir(target_root, ...)`(.lv_cache);不讓使用者指目標資料夾。 |
| **R5** | objective/head 耦合:無 head 時 uncertain/confusion raise | UI 反灰(§3.4);預設 `novelty`。 |
| **R6** | 探索抽樣若漏掉 Top-K → 掉了使用者最想看的;sub-result 索引須對齊 | 抽樣**強制納入 Top-K/bad**;records/scores/obj_emb 同步重排,散點 selection→index 對映不破。 |
| **R7** | 掃描物件數未知(yolo 每圖框數不定),>5000 自動切模式需要 count | 用**掃描回報的 `objects_scored`** 觸發(第一批後即知),非事前預測;文案說明「大資料已切標註模式」。 |

## 6. Acceptance Criteria(真實 Playwright E2E;`@pytest.mark.e2e`;給 /pm)

> GUI/整合 = false-green 重災區:凡「能看到/能操作」都要**真實行為斷言**(真的渲染出佇列卡片、真的切掉散點、
> 真的續跑出更多)。E2E 用**小資料 + 小 batch_size + 降門檻**驗**接線行為**(非literal 50k 效能);重用 `tests/e2e/conftest.py` harness。

- **AC-G1(分批掃描產出可渲染佇列)**:①建小模型並存 → ②選目標資料夾 → 按「大資料分批掃描」→
  掃描完成後**標註模式渲染出 ≥1 張佇列縮圖卡**,且每卡**可見 `reason` 文字**(真實文字,非僅元素存在)。
- **AC-G2(標註模式不渲染散點 = 大資料 scale-safe)**:標註模式下**散點圖元素不存在/不渲染**(只有佇列+直方圖);
  切「完整」模式才出現散點 → 證明大資料路徑真的避開全量散點。
- **AC-G3(暫定 Top-K 於掃描中出現 + 續跑推進)**:以小 `max_batches` 啟動 → 掃描中畫面**可見「暫定」字樣 + ≥1 張卡**,
  且 `images_processed` 隨 rerun **單調增加**;掃完 `done`。
- **AC-G4(繼續上次)**:掃描中途「⏸停止」→ 佇列保留;按「▶繼續上次」→ 先即時顯示現有 Top-K(read_checkpoint),
  再續跑到 `done`,最終 `objects_scored` == 一次跑完。
- **AC-G5(未存模型擋下)**:①建模但**未存** → 按「大資料分批掃描」→ 出現「請先存模型」提示**或**自動存後才跑
  (二擇一,PG 實作;E2E 驗其一),**不**出現原始 exception。
- **AC-G6(object_source 從磁碟正確)**:以 `whole_image` 建模並存 → 📂載入(in-memory 掉 object_source)→ 分批掃描 →
  **仍以整張影像掃**(物件數 == 影像數),不被默默當 yolo。(坐實 R2 修復。)
- **AC-G7(加入購物車)**:標註佇列某卡「🛒加入購物車」→ 購物車數量 +1(真實狀態變化),匯出清單含該影像。
- **AC-G8(無 head 反灰)**:無 head 的凍結模型 → uncertain/confusion 目標**反灰不可選**,novelty 可跑、`diversity_applied` 為 false 的呈現一致。

> 停止/例外必清 flag 的「無無限迴圈」性質:E2E 以「掃完後畫面穩定、無持續 rerun 抖動」間接驗(AC-G3 done 後 `images_processed` 不再變)。

## 7. 給 /pm 的提示

- E2E 放 `tests/e2e/test_al_batch_gui_e2e.py`(`@pytest.mark.e2e`),重用既有 anomaly E2E harness(`_anomaly_wizard.py` 的建模/存模型/選資料夾 helpers)。
- **小資料驗接線**:合成少量影像(可重用 `synthetic_yolo_dataset` 精神)+ 小 `max_batches` 讓「暫定→續跑→done」在 E2E 可觀察;不需真 50k。
- **降門檻觸發標註模式**:>5000 門檻在 E2E 不可達 → 以**設定注入/session 預設**強制標註模式(不 hardcode 改門檻常數),或直接驗「切到標註模式 → 無散點」。
- 契約鎖點:標註模式**無散點**(AC-G2)、暫定 Top-K 掃描中可見(AC-G3)、續跑等於一次跑完(AC-G4)、object_source 從磁碟(AC-G6)是防 silent-wrong/false-green 的核心。
- 此為**整合層接線**、不新增純邏輯模組;無單元 gate,done = 上述 E2E 全綠(交 `/ux-test` 或人觸發,不進 PG 自主修綠迴圈)。
