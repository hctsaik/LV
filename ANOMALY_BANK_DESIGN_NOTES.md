# 瑕疵偵測:Memory Bank 持久化 + 跨資料夾投影 — 設計與實作

> **狀態:✅ 已實作完成(2026-06-28)。** 經多 agent 三輪設計討論收斂(完整記錄見
> `ANOMALY_BANK_DISCUSSION_RAW.md`)→ 五模組 TDD 落地 → 單元 gate 全綠 + 真實 E2E 綠 +
> 真實 DINOv2 端到端鏈驗證。本檔是定案 + 實作對照。

## User 需求(情境:非常大量 good、少數 bad)
1. **建 + 存**:選資料夾 → 建 memory bank →(可選)存硬碟加速;硬碟也存 few-shot good/bad。
2. **讀 + 評分**:選新資料夾 → 從硬碟讀回 bank → 用舊 bank 對新資料評分。
3. **映射回舊分佈**:新資料投影回舊資料的座標,一眼看出落在舊 good 密集區(正常)或離群(可疑)。

## 最終定案(原則:更準、long term)
**架構/我裁決**:① 評分向量 **fp32**(零漂移、單一真相) ② 投影底圖存 **D 維**(密度判定才準;
視覺 good_obj_emb 用 fp16) ③ **PCA-only 真 transform**(transform-into-fixed-basis,非 re-fit;
umap 進 backlog) ④ stale 警告 backlog ⑤ 移除 patch_grid 硬鍵、meta 原子最後寫、寫硬碟正向白名單
⑥ few-shot **內容定址** `(stem, 量化 bbox, label)`(obj_index 來自 label 行序、重標註會變,只當 tie-break)。
**User 拍板**:🅰 coreset 升級 **greedy/k-center**(更準) 🅱 few-shot **跨資料夾對齊進首版** 🅲 **object 模式也支援**(projection-only profile)。

## bank profile 格式(一個目錄四檔,meta 最後寫=完整性哨兵)
```
<bank_dir>/                     # 預設 dataset_cache_dir(src, "anomaly_bank/<name>") → .lv_cache
  bank.npz        # patch coreset 向量(fp32;object 模式可無 → projection-only)
  projection.npz  # mean / components(=Vt[:2]) / ref_coords / good_obj_emb / tau / l2norm
  fewshot.json    # [{stem, bbox, label, verdict}]  內容定址,跨資料夾可對齊
  meta.json       # 硬鍵 model/target_res/patch_dim/obj_dim(不符拒載)+ 軟鍵(追溯)
```

## 五模組(全 TDD:先紅測 → 實作 → `verify/gate.py <module>` 綠)
| 模組(scripts/) | Tier | 職責 | 測試 |
|---|---|---|---|
| `anomaly_coreset.py` | A | greedy k-center coreset(farthest-point;oversample_cap 撐大量) | `tests/test_anomaly_coreset.py` 7 綠 |
| `anomaly_project.py` | A→B | fit_projector / transform_new(固定舊基底) / nearest_ref / classify_dense(D 維) / compute_tau(去自身) / save·load | 8 綠 |
| `anomaly_bank_store.py` | B | save_bank·load_bank(atomic meta-last) / check_compat(硬鍵拒載) / confirmed_to_fewshot · match_fewshot_to_indices(bbox 反查救行序) / assert_safe_bank_dir(白名單) | 10 綠 |
| `anomaly_tool.run_pipeline`(擴) | B | 加 `external_bank`(patch 短路、不重建)/ `external_ref`(object 對舊 good 離群);cache key 由 meta 鎖死 | `tests/test_anomaly_external_bank.py` 5 綠 + anomaly_tool 回歸 5 綠 |
| `app._anomaly_ui`(擴) | B | 主畫面 bank 面板(載入/掛載膠囊)、結果區存 bank、掛載雙行為、散點疊『舊分佈』灰底、few-shot 對齊 | `tests/e2e/test_anomaly_bank_ui_e2e.py` E2E 1 綠 |

## 驗證(全綠)
- **單元**:5 模組 gate 全 GREEN(7+8+10+5+5 = 35),契約檔未竄改。
- **E2E**:`test_anomaly_bank_ui_e2e`(存→載入掛載→掛載重跑→msg+灰底 trace)綠;既有 anomaly 6 + corrupt 1 回歸綠 = 8/8。
- **真實 DINOv2 端到端鏈**:coco8 train 建 bank → 存 4 檔 → 讀回 cosine 一致 → 換模型被拒 → val 用 train external bank 評分(17 物件)→ val 投影回 train 基底 → few-shot 內容定址不亂貼。ALL-OK。

## GUI 用法
1. 選資料夾 → ▶ 執行偵測 →(可標 good/bad)。
2. 主畫面「💾 把本次結果存成 Memory Bank」→ 落 `.lv_cache`(或自訂目錄,白名單防呆)。
3. 換新資料夾 → 主畫面「💾 讀入既有 Memory Bank」貼路徑 → 載入 → 膠囊「🔗 已掛載 bank」。
4. ▶ 執行偵測 → 用舊 bank 評分;散點圖灰底=舊資料分佈,彩色=新資料(落 good 密集區 vs 離群)。

## 守住的鐵則
絕不寫使用者資料集(.lv_cache 或 User 指定 + 白名單);硬鍵 model/target_res/dim 不符拒載(擋 silent-wrong);
fp32 評分;transform-into-fixed-basis(非 re-fit);D 維密度判定。

## 已整合 / Backlog
- greedy coreset **已接進 `build_memory_bank`(預設 `method='greedy'`)**,建 bank 即用更準的代表性子集;
  CPU 上超大量 good 會 graceful 退化成隨機子抽樣(GPU/近似加速為 backlog)。
- Backlog(未進首版):umap 第二視角(合併重投影)、stale 過期警告、few-shot 對齊的散點視覺強化、最近鄰連線。
