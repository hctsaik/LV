# 設計:M13 GUI 接線(第 9 工具「🎯 以樣搜樣」,`scripts/app.py`)

> PRD [2_PO_PRD/fewshot_search_prd.md](../2_PO_PRD/fewshot_search_prd.md)。Tier B → 真實 E2E。
> 復用:sample_bank、al_batch(objective=retrieve)、retrieval_export、M9 進度條 pending-flag 模式、
> M10 佇列樣式、既有 frozen model(**v1 復用 `anomaly_model`**;無則導引先到瑕疵偵測①建模)。

## 1. 工具入口
- `tool_switch` segmented_control 加選項 **「🎯 以樣搜樣」**;dispatch `elif tool == "🎯 以樣搜樣": _fewshot_search_ui()`。
- 三步(用 `st.segmented_control` key=`fewshot_step`,同 anomaly wizard 跨 rerun 保留):
  「① 樣本集」「② 海掃」「③ 確認 / 匯出」。

## 2. 前置:凍結模型
- v1 復用 `st.session_state["anomaly_model"]`(需 `_dir` = 已存模型目錄)。無 → `st.info("先到『瑕疵偵測』①建立/載入一個模型")`,三步 disabled。
- 樣本集與海掃都用此模型的 `meta.model/target_res`(embedding 空間一致)。

## 3. Widget key 契約(PG 必實作)
### ① 樣本集
| key | 元件 | 說明 |
|-----|------|------|
| `fewshot_sample_folder` | folder picker list | 樣本資料夾(YOLO;4 類×5~10 張) |
| `fewshot_build_bank_btn` | button | 建樣本集(pending→主體 build_sample_bank + save .lv_cache) |
| `fewshot_bank_info` | container | 顯示樣本數 + 各類別數(建好後) |

### ② 海掃
| key | 元件 | 說明 |
|-----|------|------|
| `fewshot_target_folder` | folder picker | 要海掃的大資料夾(帶低信心六欄粗框;無框→整張影像) |
| `fewshot_min_conf` | slider 0~1(預設 0) | 粗框信心預篩 |
| `fewshot_theta` | slider 0~1(預設 0.3) | 相似度門檻(佇列只留 best_sim ≥ θ) |
| `fewshot_k` | slider | 取前幾個 |
| `fewshot_scan_btn` | button | 海掃(pending→主體 al_batch retrieve;progress 串流) |
| `fewshot_queue` | container | 佇列(縮圖 + 建議類別 + 相似度) |

### ③ 確認 / 匯出
| key | 元件 | 說明 |
|-----|------|------|
| `fewshot_skip_{i}` | button | 該顆略過(decision=skipped) |
| `fewshot_relabel_{i}` | selectbox(樣本類別) | 改類(decision=relabeled + final_class) |
| `fewshot_out_dir` | text_input | 匯出資料夾(另選) |
| `fewshot_export_btn` | button | 匯出 YOLO+CSV(export_retrieval,C6) |

DOM 穩定標記:海掃完成主畫面含「以樣搜樣掃描完成」;匯出完成含「以樣搜樣匯出完成」。
決定狀態:`st.session_state["fewshot_decisions"]`={item_id:{decision,final_class}};預設(不在 dict)=accepted+suggested_class。

## 4. 邊界(GUI 友善)
| 情境 | 行為 |
|------|------|
| 無 anomaly_model | 三步 info 導引,鈕 disabled |
| 樣本集未建就海掃 | 提示先建樣本集;掃描鈕 disabled |
| 海掃 0 命中(θ 太高) | 佇列空提示「調低相似度門檻」 |
| out_dir 空 / C6 | 匯出鈕 disabled / export_retrieval raise → st.error |

## 5. Acceptance Criteria(真實行為;`tests/e2e/test_fewshot_gui_e2e.py`)
> 用 `yolo_defect_at_nmin`(scratch/stain 兩群)當**樣本資料夾**;海掃同 root(有 YOLO 框)。
> 先用 anomaly wizard `build_model` 建模(供 _dir)→ 切「以樣搜樣」。

- **AC-F1(樣本集 + 海掃 + 佇列)**:① 選樣本夾 → 建樣本集(fewshot_bank_info 顯示樣本數 ≥ 2)→
  ② 選目標夾 → 海掃 → 佇列渲染 ≥1 縮圖,卡片含**建議類別(scratch/stain)+ 相似度**;reason 含「相似度」。
- **AC-F2(匯出雙出 + 讀回)**:③ 設 out_dir → 匯出 → out_dir/labels/*.txt(cls_id∈{0,1})+ retrieval_report.csv
  (表頭正確)+ classes.txt 都在;**來源資料夾零寫入**(C6,前後檔案集合比對)。
- **AC-F3(無模型友善)**:未建/載模型直接切「以樣搜樣」→ 顯示「先到『瑕疵偵測』①…」導引,無 stException。

---
## 增補(M13 Task6):監看模式(背景自動海撈,復用 M10)
> 資料夾持續進新圖 → 背景服務按樣本集自動海撈。復用 al_workspace/al_service(similar 的 reference 機制前例)。

### 服務層契約(09/10 加法)
- `al_workspace._DEFAULTS` 加 `sample_bank_dir: None`(workspace profile 指向 .lv_cache 的樣本集)。
- `al_service.init_workspace(..., sample_bank_dir=None)`:透過 **over 寫進 profile(retrieve 用)。
- `al_service.run_once`:`profile.objective=="retrieve"` 時 → `sample_bank.load_sample_bank(profile.sample_bank_dir)`
  → `ref_vectors=bank.vectors, ref_labels=bank.labels` 傳 `run_batched(objective="retrieve", ...)`;
  缺 sample_bank_dir → 明確 raise。既有 novelty/similar 路徑不變。

### Acceptance(單元;tests/test_al_service.py)
- **AC-SVC-RET**:`init_workspace(objective="retrieve", sample_bank_dir=<存好的樣本集>)` → profile.sample_bank_dir 正確;
  `run_once`(注入 embed_fn;樣本 E2 標 X、E3 標 Y)→ 佇列 `id` 對應 topk 帶 **suggested_class**、依相似度排序、
  X 群圖建議 X。

### GUI(第 9 工具「④ 監看」;沿用 M10 watch 樣式;E2E 以 AC-SVC-RET 覆蓋核心,GUI 接線最小)
- 步驟列加「④ 監看」:工作區目錄 + 🆕初始化(用當前樣本集 + 目標夾)+ 📤匯出設定 + ▶立即掃描一次 + 狀態卡 + 佇列消費(採納/改類/略過→labels.jsonl)。
- profile 可攜:匯出 profile.yaml + sample_bank 目錄給離線服務跑。
