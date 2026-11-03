# 31 `react_web_shell` — AC ↔ 測試對應表

設計：[3_Architect_Design/31_react_web_shell.md](../3_Architect_Design/31_react_web_shell.md) · PRD：[2_PO_PRD/react_web_migration_prd.md](../2_PO_PRD/react_web_migration_prd.md) · Cases：`LV-WEB-01-IC-01`、`LV-WEB-01-IC-02`、`LV-WEB-01-IC-03`、`LV-WEB-01-OD-01`、`LV-WEB-01-OD-02`、`LV-WEB-01-VIEW-01`、`LV-WEB-01-SEARCH-01`、`LV-WEB-02-IMG-01`

| AC | 要證明的行為 | 測試／證據 | 狀態 |
|---|---|---|---|
| AC1 | classifier 資料探索、類別、split 與排序符合 Streamlit 基準 | `tests/test_react_web_shell.py::test_case_lv_web_01_ic_01_discovery` | 通過；API／排序與標籤測試 |
| AC2 | 同一模型與 fixture 產生 4×384 真實 embedding，結果有限且非零 | 真實 DINOv2 E2E：`tests/e2e/test_react_web_shell_e2e.py::test_lv_web_01_ic_01_ic_02_real_model` | 通過；權重 SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`；本機 Python 3.14 AMD64，Windows ARM64 尚待驗 |
| AC3 | React 顯示 4 個投影點；點擊後顯示對應影像、metadata、類別及 split 篩選 | `tests/e2e/test_react_web_shell_e2e.py::test_lv_web_01_ic_01_ic_02_real_model` | 通過；真模型、可見 UI、類別／split 篩選、投影／維度切換時保留選取並讀回原圖 |
| AC4 | 缺模型、空資料與取消可見；不默默成功或替代 backend | API cases + `test_lv_web_01_ic_01_missing_model_is_visible` + `test_lv_web_01_ic_01_cancel_is_visible_in_react` | 通過；缺模型、空資料、取消均有可見狀態 |
| AC5 | manifest SHA-256 embedding 快取可重用，來源圖片 bytes 不變，輸出僅在隔離 cache root | `test_case_lv_web_01_ic_01_manifest_cache_and_source_immutability` + `test_case_lv_web_01_ic_01_changed_content_invalidates_embedding` | 通過；重跑重用四張，改圖只重算一張，來源 hash 不受分析流程更動 |
| AC6 | NativeApp 選資料夾後同源 API 可完成真模型操作，worker 停止後無孤兒程序 | NativeApp visible UI E2E receipt | 待驗；React 尚未接入 NativeApp 發布包、OS folder grant 和 worker lifecycle |

## 完成判準

AC2、AC3、AC6 須以真實權重和可見 UI 通過；AC1、AC4、AC5 可先由 deterministic 單元／API 測試覆蓋。AC1–AC5 已在本機完成；AC6 與 Windows ARM64 驗證仍未完成，因此 `LV-WEB-01-IC-01` 及 M24 不能標為完成。任何 `skip`、fake backend、截圖-only、HTTP-only 或模型健康檢查均不得將此 case 標為完成。

## 既有操作旅程與 React case inventory

本表依 2026-11-01 LV 工具列、scripts/app.py 的 UI 分流、使用指南、已存在的 UI／E2E 測試與本機來源 checkout 的誤逮案件規格盤點，共整理 44 個可驗收 journey。每列是一個可完整操作與回讀的 journey；控制項數量不等於 use case 數量。

Legacy 測試只作為 Streamlit 行為錨點，不算 React 通過證據。React case 必須用固定輸入、固定模型、可見 UI 操作重跑；檢查顯示結果、領域數值、檔案輸出、跨工具 handoff、資料夾副作用及重開後狀態。React 的 LV-WEB-01-IC-01 有本機 AC1–AC5 證據；IC-02 有 deterministic API/core 覆蓋，仍須真實多模型與可見 UI parity receipt。IC-03 有 deterministic feature/API 與可見 UI reference 流程覆蓋，仍須真實模型 parity。IC-04 有 deterministic KMeans API 與可見 React 3D→2D box/lasso selection E2E，仍須真實模型與 NativeApp parity。OD-01 有 deterministic API 與可見 UI E2E 覆蓋 selected-file→auto→manual 類別來源優先序、多模型、缺類別錯誤及資料夾 bytes 不變；仍須真實模型 parity 與 NativeApp picker/worker receipt。OD-02 有 deterministic API 與可見 UI E2E 覆蓋 policy、bbox crop、候選順序、class filter、2D／3D、選取 crop 回讀及來源 bytes 不變；真實模型 parity 與 NativeApp picker/worker receipt 仍待補。VIEW-01 有 deterministic overlay API 與可見 UI E2E，覆蓋 shift 多選、空間排序縮圖、逐張瀏覽、overlay 開關持續及模式切換清除舊 run；來源 bytes 不變。SEARCH-01 有 deterministic API 與可見 UI E2E，覆蓋距離排序、query 排除、候選續查、跨模型重算、CSV 順序及來源 bytes 不變；真實模型排序 parity 與 NativeApp grants／ARM64 仍待驗。LV-WEB-02-IMG-01 的 API／React contract 驗收使用 deterministic feature function；它證明接線、投影、選取、圖片回讀及取消，不證明真實模型數值或 Streamlit parity。逐值／可見結果對照及 ARM64 NativeApp receipt 尚未完成。標成「本機來源」的文件／測試存在於原 LV 使用者工作樹，但尚未併入此隔離分支，不得當成此分支的 CI 證據。

| Case ID | 原有操作旅程與 parity contract | Streamlit 行為錨點 | React 狀態 |
|---|---|---|---|
| LV-WEB-01-IC-01 | Image Classifier 整圖：選來源、DINOv2-small、PCA、執行、選點讀回原圖與類別／split | scripts/app.py::_visualize_embeddings_ui；scripts/visualize_embeddings.py | 本機 AC1–AC5 通過；NativeApp AC6 與 ARM64 待驗 |
| LV-WEB-01-IC-02 | 分類器多資料夾／多模型；切換 PCA、t-SNE、UMAP、LDA 與 Supervised UMAP，保留 split、類別篩選及點選後的記錄身分 | scripts/app.py::_visualize_embeddings_ui；tests/e2e/test_gui_flows.py::test_f_selection_persists_across_views | deterministic multi-model/projection API cases 通過；真模型可見 UI 已驗 PCA、t-SNE、LDA、維度切換及點選讀回，但只用 DINOv2-small；本機 UMAP 套件未安裝，UMAP 與第二個真實模型 UI 仍待驗 |
| LV-WEB-01-IC-03 | 固定 UMAP reference：重跑沿用既有參考座標；新增資料時保持舊點穩定並轉換新點；可手動重建 reference | scripts/app.py::_visualize_embeddings_ui；tests/e2e/test_gui_flows.py::test_t_umap_reference_frame；umap_ref.py::stable_umap | React 控制、worker API 與 reference cache 已接入；deterministic API 和可見 UI E2E 覆蓋首次 fit、重跑、新增圖片、手動重建及來源資料不變；真實模型 parity 與 NativeApp 尚待驗 |
| LV-WEB-01-IC-04 | Image Classifier 3D 導覽：切換維度時保留多點選取，以 KMeans 分區並標記區塊，進入區塊後以 2D 框選／套索加選 | scripts/app.py::_viz_kmeans_blocks；tests/e2e/test_gui_flows.py::test_j_3d_mode_preserves_selection | React/sidecar 已接上 KMeans centers、原始 embedding PCA block projection、box/lasso 加選與跨維度保留；API 和可見 E2E 使用 deterministic features 通過；真實模型 parity 與 NativeApp 尚待驗 |
| LV-WEB-01-OD-01 | Object Detector 整圖：YOLO images／labels、類別來源優先序、多模型，缺少所有類別來源時明確錯誤 | tests/e2e/test_scenarios_r1.py::test_s01_detector_multimodel_cold_run、test_s09_detector_missing_classes_error | React/sidecar 已接入整圖 discovery、selected-file → auto classes.txt/data.yaml → manual names、多模型切換與缺類別錯誤；deterministic API 與可見 E2E 通過；真實模型 parity 與 NativeApp 尚待驗 |
| LV-WEB-01-OD-02 | Object Detector 物件級 embedding：bbox 裁切、object policy、class filter、2D／3D 與候選標記順序一致 | `tests/test_react_web_shell.py::test_case_lv_web_01_od_02_object_embeddings_policy_order_and_crop_readback`; `tests/e2e/test_react_web_shell_e2e.py::test_lv_web_01_od_02_visible_object_embeddings_policy_filters_and_3d` | deterministic API 與可見 E2E 通過；真實模型 parity、NativeApp grants／ARM64 仍待驗 |
| LV-WEB-01-VIEW-01 | 框選／套索後縮圖牆與原圖檢視同序；YOLO overlay 在逐張瀏覽時正確更新；模式切換不沿用舊結果 | `tests/test_react_web_shell.py::test_case_lv_web_01_view_01_detector_image_route_draws_yolo_overlay`; `tests/e2e/test_react_web_shell_e2e.py::test_lv_web_01_view_01_selection_gallery_yolo_navigation_and_mode_isolation` | deterministic overlay API 與可見 E2E 通過；NativeApp grants／ARM64 仍待驗 |
| LV-WEB-01-SEARCH-01 | 以圖搜圖按距離排序、選候選續查、跨模型重算並匯出結果 | `tests/test_react_web_shell.py::test_case_lv_web_01_search_01_similarity_reorders_by_model_and_exports`; `tests/e2e/test_react_web_shell_e2e.py::test_lv_web_01_search_01_visible_chain_cross_model_and_csv`; Streamlit anchor: `tests/e2e/test_scenarios_r1.py::test_s03_find_similar_cross_model` | deterministic API 與可見 UI E2E 通過；真實模型排序 parity、NativeApp grants／ARM64 仍待驗 |
| LV-WEB-01-SEARCH-02 | Chinese-CLIP 以文搜圖，以及 phash／embedding 重複檢查和跨 split 洩漏覆核 | docs/usage_guide.md 的 Visualize「相似／重複」流程；需補固定資料夾 characterization | 未開始；目前沒有直接可對照的 React 或完整 Streamlit E2E |
| LV-WEB-01-RANK-01 | 離群度、標籤分歧排序與排名式檢視；排序只是線索，不得變成品質判定 | tests/e2e/test_scenarios_r1.py::test_s04_outlier_ranking_and_export；tests/e2e/test_gui_flows.py::test_k2_label_disagreement_ranking | 未開始 |
| LV-WEB-01-CART-01 | Visualize 選取／面板候選寫入持久策展清單，CSV／JSON 身分和理由可讀；重複加入有既有規則 | tests/e2e/test_gui_flows.py::test_i_export_list_round_trip；tests/e2e/test_scenarios_r1.py::test_s10_empty_selection_export_honesty | 未開始；需定義跨 worker 持久性 |
| LV-WEB-01-EDGE-01 | 單張／空資料、壞檔、Unicode 路徑、缺模型與缺類別的明確狀態；來源資料 bytes 不變 | tests/e2e/test_scenarios_r1.py::test_s06_minimal_dataset_single_image、test_s07_unicode_paths_round_trip、test_s09_detector_missing_classes_error | 未開始；React 本案例已覆蓋空資料與缺模型子集 |
| LV-WEB-02-IMG-01 | 整圖 A/B：兩側多資料夾、LV feature model、形心餘弦漂移、PCA／t-SNE／UMAP／LDA／Supervised UMAP 投影、2D／3D、點選／2D 框選與 A/B 縮圖讀回 | Streamlit baseline: `scripts/app.py::_compute_compare_whole`、`scripts/app.py::_render_compare_by_class`; React contract: `tests/test_react_compare_api.py::test_case_lv_web_02_img_01_*`、`tests/e2e/test_react_web_shell_e2e.py::test_lv_web_02_img_01_visible_compare_contract`. `tests/e2e/test_scenarios_r1.py::test_s05_compare_distributions_metrics` is skipped and superseded | 部分完成：deterministic API／可見 React contract 覆蓋多資料夾、漂移、投影、選取、圖片回讀與取消；真實模型數值／畫面 parity、NativeApp grants、3D KMeans block drilldown、物件級流程與 cart handoff 尚待驗 |
| LV-WEB-02-OBJ-01 | Object-level A/B 類別配對、分群投影、選取框與雙欄縮圖對照；不匹配類別要誠實提示 | tests/e2e/test_compare_by_class.py；tests/e2e/test_cmpc_adversarial.py | 未開始 |
| LV-WEB-02-STATE-01 | 比較單位、投影、維度或類別改變時清除舊選取；手動門檻套用後失效舊結果；A/B 候選可加入購物車 | tests/e2e/test_cmpc_adversarial.py；tests/e2e/test_scenarios_cmpc.py | 未開始 |
| LV-WEB-03-ATTR-01 | 屬性棋盤軸、分桶、目標覆蓋及頻率先驗校正；顯示缺格及格內影像 | scripts/app.py::_completeness_ui；目前未定位到完整可見 E2E | 未開始；先凍結預設與使用者校正語義 |
| LV-WEB-03-IMG-01 | Image-level embedding coverage：稀疏度、類別顏色、投影切換、框選與清除選取 | tests/e2e/test_scenarios_cov.py::test_s01_default_unit_is_object_and_no_pad_input 至 test_s13_clear_button_aligned_with_visualize_pattern | 未開始 |
| LV-WEB-03-OBJ-01 | Object-level coverage：class/object 過濾、投影／區塊、tiny class guard、框與標籤索引保持一致 | tests/e2e/test_scenarios_cov.py::test_s14_object_filter_keeps_labels_aligned_for_supervised_proj；tests/e2e/test_scenarios_objcov.py | 未開始 |
| LV-WEB-03-REF-01 | 候選池採礦與外部參照 B 是不同角色；比較覆蓋率、類別／物件對齊、候選最近鄰並加入策展清單 | tests/e2e/test_scenarios_cov.py::test_s10_reference_distribution_role_and_coverage；tests/e2e/test_scenarios_objcov.py::test_s04_candidate_missing_selected_class、test_s07_classname_matching_across_id_orderings | 未開始 |
| LV-WEB-04-CTX-01 | 從 Visualize 選取建立診斷 reference；來源 run 改變後 stale／incompatible 明示且阻止錯誤歸因；進頁不自動推論 | scripts/app.py::_open_visualize_diagnostics、_diagnostics_context_status；原來源 tests/test_explain_page.py | 未開始；來源 unit tests 為本機來源 |
| LV-WEB-04-HEAD-01 | 指定 224／336／448 解析度建置或重用對應分類 head；失敗可重試，缺 head 不偽造完成 | 原來源 tests/test_explain_page.py；tests/e2e/test_explainability_e2e.py | 未開始 |
| LV-WEB-04-EXPL-01 | 執行分類結果與 Grad-CAM／Attention evidence；target class、signed／positive 視圖與實驗歸因不混淆；mask 不冒充定位證據 | tests/e2e/test_explainability_e2e.py；tests/e2e/test_explainability_mvtec_e2e.py；原來源 tests/test_explain_page.py | 未開始；原來源 extended tests 為本機來源 |
| LV-WEB-05-CASE-01 | 建立／重開誤逮案件、匯入原圖／截圖與預測、案件導航；逐區域記錄正常／瑕疵／爭議／資訊不足，局部判斷不擴成整圖標籤 | 原來源 tests/e2e/test_fp_case_workspace_e2e.py、tests/test_fp_case_page.py | 未開始；來源文件與測試為本機來源 |
| LV-WEB-05-RETRIEVE-01 | 明確搜尋正常／瑕疵／歷史對照，缺一側仍標示沒有結果；模型版本不同時保留來源身分與不確定性 | 原來源 openspec/changes/add-false-positive-case-workspace 與 tests/e2e/test_fp_case_workspace_e2e.py | 未開始；來源為本機來源 |
| LV-WEB-05-REPORT-01 | 案件說明區分事實／假設／行動，引用證據，匯出人讀報告及 JSON；回覆狀態與根因分開 | 原來源 docs/false-positive-case-workspace.md、tests/e2e/test_fp_case_workspace_e2e.py | 未開始；來源文件為本機來源 |
| LV-WEB-05-AL-01 | 案件候選區域去重後交人工覆核；部分覆核、回傳錯案／過期版本整批拒絕；區域數與已標訓練影像數不混算 | 原來源 tests/e2e/test_fp_case_workspace_e2e.py、test_investigation_recovery_e2e.py | 未開始；來源測試為本機來源 |
| LV-WEB-05-EXT-01 | 外部 ONNX／YOLO 明確契約推論、歷史完整重播、固定目標對照實驗、paired holdout gate、人工確認訓練 ZIP | 原來源 tests/e2e/test_fp_external_investigation_e2e.py、test_yolo_investigation_e2e.py、docs/external-model-investigation.md | 未開始；來源測試為本機來源 |
| LV-WEB-06-BANK-01 | 建立、保存、載入 Normal Bank 並於新程序重開；解析度／head／特徵來源身分一致 | tests/e2e/test_anomaly_bank_ui_e2e.py::test_model_save_load_project；tests/e2e/test_anomaly_wizard_e2e.py::test_build_apply_sample_flow | 未開始 |
| LV-WEB-06-SCORE-01 | 目標資料夾推論、風險排序／熱圖、分數與類別過濾；確認正常樣本後重算，不重建 bank | tests/e2e/test_anomaly_tool_e2e.py::test_defects_rank_in_top、test_score_and_class_filters、test_confirm_normal_reapply | 未開始 |
| LV-WEB-06-REVIEW-01 | 選異常候選加入策展清單並匯出原圖，不可把 bbox crop 當成原始檔；缺 labels 顯示友善錯誤 | tests/e2e/test_anomaly_tool_e2e.py::test_select_and_add_to_cart、test_export_is_original_image、test_error_path_no_labels | 未開始 |
| LV-WEB-07-SAMPLE-01 | 不依賴先建 anomaly model，建立搜尋樣本集／特徵 bank；保留整圖查詢語義 | tests/e2e/test_fewshot_scenarios_e2e.py::test_s01_no_model_full_flow、test_s06_whole_image_mode | 未開始 |
| LV-WEB-07-SCAN-01 | 海掃候選、人工確認類別／框、設定候選門檻；加入 bank 的候選在後續 run 可重用 | tests/e2e/test_fewshot_scenarios_e2e.py::test_s07_add_to_bank_grows、test_s09_theta_applied_to_queue | 未開始 |
| LV-WEB-07-EXPORT-01 | 保留勾選狀態與 C6／整圖輸出規則；匯出 YOLO＋CSV 後可交人工確認，監看佇列顯示任務狀態 | tests/e2e/test_fewshot_scenarios_e2e.py::test_s02_export_dual_and_c6、test_s05_uncheck_excludes_from_export、test_s08_monitor_no_model | 未開始 |
| LV-WEB-08-DIFF-01 | 對齊 Good／Bad 資料後分析 patch 差異；Top-K 熱圖定位 planted signal 並呈現穩定度，不顯著時明示 null | tests/e2e/test_groupdiff_e2e.py::test_g1_planted_difference_found_and_localized、test_g2_null_split_is_honest | 未開始 |
| LV-WEB-08-EXPORT-01 | 選區查看 Good／Bad 同位置對照、調整熱圖呈現，匯出報告至另選目錄且不碰來源 | scripts/app.py::_groupdiff_ui；tests/e2e/test_groupdiff_e2e.py | 未開始；匯出目的地與來源 hash 尚需明確 assertion |
| LV-WEB-09-CART-01 | 跨工具購物車按來源篩選，預覽缺檔／去重數，保留來源、理由與類別 | tests/e2e/test_gui_flows.py::test_i_export_list_round_trip；scripts/app.py::_export_subset_ui | 未開始 |
| LV-WEB-09-SAFE-01 | copy／symlink／manifest-only 與 skip／rename／overwrite；目的地不得位於來源中且衝突結果寫入報告 | tests/e2e/test_export_subset_ui.py；tests/e2e/test_export_subset.py | 未開始 |
| LV-WEB-09-OBJECT-01 | 物件候選匯出的是原圖與完整標註；類別 remap、同 hash 去重、非 YOLO sidecar 及 lineage 保持一致 | tests/e2e/test_export_subset.py::test_crop_out_object_level、test_yolo_object_level_exports_full_image、test_sha256_dedup_merges_source_tags、test_lineage_has_skew_warning_and_self_hash | 未開始 |
| LV-WEB-10-SEND-01 | 送標批次開回合帳本；重送去重、已處理 pool guard 和明確 override | tests/e2e/test_round_ledger_e2e.py::test_send_opens_round_e2e；tests/e2e/test_pool_registry_e2e.py | 未開始 |
| LV-WEB-10-READBACK-01 | 使用者主動匯入並套用 Labeling 回讀；重開後狀態一致、稽核可匯出、來源資料集不變 | tests/e2e/test_readback_store_e2e.py | 未開始 |
| LV-WEB-10-ROUND-01 | 回合／池帳本跨程序保留；顯示 learning curve 與 probe 指標，低樣本明示限制 | tests/e2e/test_round_ledger_e2e.py；tests/e2e/test_probe_eval_e2e.py | 未開始 |
| LV-WEB-10-PRELABEL-01 | 以既有模型產生預標、人工確認再送 Labeling；缺 head graceful error；回讀更新同一 round | tests/e2e/test_prelabel_gui_e2e.py | 未開始 |
| LV-WEB-10-EVAL-01 | 匯入 predictions／consensus、調整 IoU／confidence／class-aware，計算漏抓並匯出 CSV；缺輸入不得偽成功 | tests/test_evaluation.py；scripts/app.py::_evaluation_ui | 未開始；需確認該路徑目前有正式導覽入口 |
| LV-WEB-10-HIDDEN-01 | Quiz 盲測與 Gray-zone review 的測試目前仍存在，但工具列會正規化其 tool_switch；先以可見 UI 確認可達性，再決定保留、併入或標記下架 | tests/e2e/test_gui_flows.py::test_y_quiz_tool、test_z_gray_zone_review；scripts/app.py::main | 尚未裁決；不可默默視為已遷移 |

### Case 完成與證據規則

每個 React case 都要有固定 fixture 內容 hash、model/profile/weight 身分、同樣的操作序列及可比較輸出。輸出至少核對畫面內容與選取身分；有數值時核對領域指標與事先定義的容差；有下載時核對檔案內容；有保存時停止 worker 並以新 PID 讀回；最後比對來源資料夾前後 hash。缺檔、空資料、模型不可用及取消依原行為逐列覆蓋。

只有上表裡的旅程都有對應 React case、legacy 對照與可見 UI receipt 後，才能報告 parity。舊 Streamlit 測試、HTTP status、health check、單張截圖或 mock model 都不構成該旅程的 React 通過證據。隱藏路由、缺少固定 fixture 或只有本機未提交的 legacy 測試列為待釐清，不推定通過。
