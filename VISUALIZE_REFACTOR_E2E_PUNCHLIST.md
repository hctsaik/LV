# Visualize 重構 E2E 收尾清單（punch-list）

> 背景:`scripts/app.py` 的 Visualize 工具於上午做了大重構(資料夾系統改版、
> 多模型、UI-state 持久化、面板/視圖、工具列精簡)。這批重構**先前從未跑過完整 E2E**。
> 本輪已修掉**所有系統性崩潰**,把 `tests/e2e/test_scenarios_r1.py` +
> `tests/e2e/test_gui_flows.py` 的失敗從 **35 → 12**。剩下 12 個是**個別、脆弱的測試對齊**,
> 多數需逐一改測試(或少量 app 行為決定),且彼此會連動。**這些紅測試是上午重構的未收尾,
> 非「壞檔防呆」造成**(已用中立化實驗證明)。

## 已修(本輪)
- viz 資料夾缺 headless 文字輸入 → `_folder_add_input("viz_folder_list")`
- `_folder_picker_list` docstring 宣稱有 add-input 卻沒渲染 → 補上(修 cov/quiz/gray/objcov)+ 移除 anomaly 重複呼叫
- `manifest.update_manifest` 暖快取崩潰:`rel_key(folder, 全域縮圖)` ValueError → 容忍 folder 外路徑
- `update_manifest` 對「不在 folder 下的 record」防呆 skip(同名資料夾/殘留)
- UI-state 持久化造成跨測試/跨 run 殘留資料夾 → `tests/e2e/conftest.py` 關閉 E2E 持久化(`LV_UI_STATE` 指向不存在父目錄 → 讀寫 no-op)
- 移除 `LV_CACHE_DIR` 隔離(test_b/test_t 在測試行程直接驗 `.lv_cache`,隔離會看不同目錄)
- s01 類別偵測訊息常駐化(app G6,`viz_class_msg` + 3534 caption);s07 去掉重複的 `st.success`
- s10 陳舊文字「清單是空的」→「購物車是空的」
- 工具列精簡:`test_y/z/ac`(組考卷/灰帶/評估,commit 319dd80 已自 tool_switch 隱藏)→ `@pytest.mark.skip`

## 剩餘 12 個(待下一輪逐一收)
| 測試 | 根因(推測) | 收尾方向 |
|------|------------|----------|
| `test_scenarios_r1::test_s01_detector_multimodel_cold_run` | 預設模型由多個改成單一 `dinov2_vitb14`(app.py:3126),測試假設多模型可切 | 測試在 `viz_models_sel` multiselect 顯式選 2 個模型(dinov2_vits14 + chinese-clip)再 Run |
| `test_s02_selection_viewer_yolo_chain` | **連鎖**:s01 的 `_select_option` 失敗讓模型下拉卡開,阻塞共享 det_page | 多半隨 s01 修好而解;否則檢查 viz_status_line/panel 狀態 |
| `test_s03_find_similar_cross_model` | 同上連鎖 + 跨模型相似面板 | 隨 s01;檢查 viz_similar_panel/viz_model_select |
| `test_s04_outlier_ranking_and_export` | 同上連鎖 | 隨 s01 |
| `test_s07_unicode_paths_round_trip` | S8 段 `已切換模式`(sidebar 收合/時序)+ 類別訊息 | `_ensure_sidebar` 後再驗;確認 `_viz_mode_snapshot` 觸發 |
| `test_gui_flows::test_j_3d_mode_preserves_selection` | 3D 模式切換保留選取 | 逐一查 3D 切換 UI |
| `test_m_selection_latency` | 選取延遲基準 | 查選取互動時序 |
| `test_o_projection_method_skip` | `viz_methods` multiselect 用 Backspace 移除到剩 PCA → tags 數不收斂 | multiselect 互動脆弱,改穩定移除法 |
| `test_q_text_to_image_search` | CLIP 文搜面板 | 查 viz 文搜 UI(需 CLIP 模型) |
| `test_t_umap_reference_frame` | `ref_path_for` 持久化驗證(移除 cache 隔離後仍紅) | 查 umap_ref 寫入路徑/時序 |
| ~~`test_u_completeness_heatmap`~~ | **✅ 已修(綠)**:檢視預設改成「嵌入覆蓋圖」(app.py:5910/4804),`cov_heatmap`(6070)只在「屬性棋盤」渲染 | 已改:run_cov **前**先切 `cov_view_mode`→「屬性棋盤」(sidebar 仍展開時),跑完直接出熱力圖 |
| `test_w_completeness_calibration_and_mining` | 熱力圖部分已隨 test_u 修法修好;**剩下更深的一段**:挖候選 mining(`🔎 撈候選補此格` popover → `cov_mine_btn` → `cov_cand_csv`)在 mining 後 `cov_cand_csv` 不顯示(無 server crash,line 876) | 查 `_render_cov_candidates`(app.py:4455)/`_mine_cell_candidates`(4412):候選是否為空、CSV 是否在 popover 內(關閉後看不到)。屬獨立問題,非預設檢視 |

## 怎麼跑
```
# 單檔逐一(快):
python -m pytest tests/e2e/test_scenarios_r1.py::test_s01_detector_multimodel_cold_run -m e2e -o addopts="" --tb=long
# 整批 viz e2e:
python -m pytest tests/e2e/test_scenarios_r1.py tests/e2e/test_gui_flows.py -m e2e -o addopts="" -rf --tb=line
```
注意:`det_page` 是 module-scope 共享頁,s01–s04/s10 連鎖;先修 s01。每次冷跑含模型載入,整批約 18 分鐘。
