# T0 現況接線盤點(2026-09-14 實讀程式)

本檔是 r3 設計對目前程式的差異清單。實作必須依此接線，不得用歷史行號硬改。工作區已有 explainability／FP／YOLO 等未提交變更，本功能不得 reset 或覆蓋那些檔案的無關 hunk。

倉庫**沒有** `AGENTS.md`。專案協定以 `CLAUDE.md` 為準(規格疊加、閘門、不寫來源資料集)。

## 實際符號

| 符號 | 現況 | 本功能用法 |
|---|---|---|
| `scripts/app.py::_render_label_quality` | 先 guard「<3 筆或 <2 類就 return」，再 `st.tabs` 跑 `object_eval.neighbor_confusion`(全 N×N) | 新增 segmented 分派必須在此 guard **之前**。新面板可顯示 singleton／單類。舊混淆仍可達，但切到新面板時不得連帶執行 N×N |
| `_visualize_embeddings_ui` | image 與 object 共用尾端 expander「全 384 維」呼叫上述面板 | 兩模式都要接;文案改「原始高維特徵」+實際 D |
| `viz_records` / `viz_raw_embeddings[model]` | row 與 raw 對齊;投影在 `viz_embeddings` | 只用 raw，不用投影座標 |
| `viz_manifest` | image Run 有 path→sha256;object Run 設 `{}` | 空 manifest ≠ 已驗證。object 必須另存 generation sidecar，否則 `source_unverified` |
| `viz_data_token` | 每次 Run 新 hex;mode restore 會連舊 token 一起還原 | 不足單獨當來源證明。analysis key 還要 embedding digest 與 snapshot |
| `_apply_readback_overlay` | Run 後改記憶體 label | 必須使 class-similarity 結果 stale |
| `_render_readback_section` → `LH.apply_readback` | 不經 Run 的即時讀回 | 同樣 stale;不能只靠 token |
| `_MODE_CLEAR_KEYS` | 切模式清空 | 加入本功能 session keys。`cs_event_generation` 在 restore 後必須作廢舊 Plotly 事件 |
| `_export_entry` / `_add_to_export` | 以 `path` 去重;score 四位小數;重加不更新 | `source=class_similarity`;完整證據走 JSON;顯示新增／略過 |
| `_viz_send_to_gray` | 舊灰帶 | **禁止呼叫** |
| `labeling_handoff.py` | Labeling 出口 | 本輪不改契約 |
| `similarity.cosine_similarity_to_ref` | float32 L2，clip 1e-12 | 可參考正規化精神;新模組 float64 scale-safe，不共用此函式畫圖 |
| `similarity_priority` | 查詢內 min-max 到 0–1 | **不可當原始 cosine** |
| `multi_ref_similarity` | 每類 **max** cosine | 不等於逐類 Top-K 平均 |
| `interaction.compute_label_disagreement` | 鄰居異類比例，index 排除自己 | 只參考排除方式 |
| `interaction.find_similar_indices` | HNSW cosine **distance** | 新 exact 統計不用 |
| `object_eval.neighbor_confusion` | 完整 N×N | 不當新 backend;舊 tab 仍用 |
| `visualize_embeddings.discover_images_classifier` | Visualize classifier **固定** `allow_flat=True`(app.py) | flat 列 `label_status=unresolved`;真 class 子資料夾(含字面 unknown/mix)為 `single` |
| `explain_contracts.stable_record_key` | source sha + 正規化 bbox | **不可代用**本功能 ID |
| `_crop_and_embed_objects` | crop 名 `{stem}__obj{obj_index}.jpg` | 同 stem 不同原圖會碰撞。adapter 發現同一 crop path 對到不同原圖／ROI 則**整份分析拒絕** |
| `cost_curve.cost_curve` | `score >= t`，正例=瑕疵 | 比較方向與正例都不同，不代用 |
| `tests/e2e/conftest.py` | 共用 `app_server` 不隔離開發者 cache;`explain_app_server` 才設 `LV_CACHE_DIR` | 本功能 E2E 在測試檔自備隔離 server，**不改**共用 harness 行為 |

## 最小允許 app.py 接線(T4 起才改)

1. `_render_label_quality` 分派 + expander 文案。
2. Run 兩條路徑寫 `viz_embedding_generation`(image 綁 manifest sha;object 在生成當下對原圖去重算內容 hash，空 manifest 不當已驗證)。
3. `_apply_readback_overlay` 與 `LH.apply_readback` 之後使本功能 stale。
4. `_MODE_CLEAR_KEYS` 加入本功能 keys;mode restore 後作廢舊事件。
5. 消費 UI action:同步 `viz_selection`、`_add_to_export`、下載 bytes。
6. 必要時在 records sidecar 提供 loader provenance(`label_origin` 等)，**不改**既有 label 值與列順序。

## 不做

新增頂層工具、復活灰帶、自動改標、改 Labeling schema、先建 N×N 再刪、用投影／centroid／max／min-max 畫本圖、寫回來源資料集、為綠燈改 `tests/e2e/conftest.py` 共用行為。
