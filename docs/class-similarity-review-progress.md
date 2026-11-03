# 類別相似度功能：進度紀錄

日期：2026-09-15。狀態：**T0–T10 完成**（單元 66 綠 + 真實 Streamlit E2E 8/8 綠）。

契約：`docs/class-similarity-review-implementation-plan.md` r3。

**禁止** `git reset` / `git clean` / 覆蓋工作區裡既有的 explainability／FP／YOLO 未提交變更。

---

## 一句話

Visualize「標錯偵測／類別混淆」expander 已接上「類別相似度分布」：box／散點／Threshold 預覽／購物車／匯出。單元 66 綠；Playwright E2E 8/8 綠（含 image+object、來源唯讀、pointer 按住預覽）。

---

## T0–T10

| 任務 | 狀態 | 證據 |
|------|------|------|
| T0 正式契約 | **完成** | OpenSpec + U-Net 1/2/3/4 + inventory |
| T1 身份／數值測試 | **完成** | `tests/test_class_similarity.py`、`tests/test_class_similarity_records.py` |
| T2 純邏輯實作 | **完成** | `scripts/class_similarity.py`、`scripts/class_similarity_records.py` |
| T3 匯出 | **完成** | `scripts/class_similarity_export.py` |
| T4 唯讀 UI | **完成** | `scripts/class_similarity_ui.py`；expander key=`viz_label_quality` |
| T5 選取／購物車／下載 | **完成** | E2E `test_ac10_cart_add_skip` |
| T6 image/object E2E、AC13/14 | **完成** | `tests/e2e/test_class_similarity_e2e.py` 4/4 |
| T7 Threshold 預覽 | **完成** | pointer-held E2E `test_ac18_pointer_held_preview` |
| T8 人工評估／PR | **完成** | 單元 `test_class_similarity_evaluation.py` |
| T9 曲線／操作點／schema2 | **完成** | 單元 + E2E AC16/33/34 |
| T10 完整回歸與交接 | **完成** | usage_guide、本檔 AC 表、效能數字、E2E 8/8 |

單元（2026-09-15）：

```text
python -m pytest tests/test_class_similarity.py tests/test_class_similarity_records.py tests/test_class_similarity_export.py tests/test_class_similarity_threshold.py tests/test_class_similarity_evaluation.py tests/test_class_similarity_ui.py -q
→ 66 passed
```

E2E（專案 `.venv`，先清殘留 Streamlit）：

```text
.venv\Scripts\python.exe -m pytest tests/e2e/test_class_similarity_e2e.py tests/e2e/test_class_similarity_threshold_e2e.py -m e2e -v
→ 8 passed in 305.63s
```

---

## E2E 對照

| AC | 測試 | 結果 |
|----|------|------|
| 13 | `test_ac13_regression_old_tools` | 綠：混淆熱圖、疑似標錯、相似鈕、購物車、取消框選仍在 |
| 14 | `test_ac14_image_and_object_source_readonly` | 綠：image N=24、object N=16；來源 listing+內容 hash 不變 |
| 07 | `test_ac07_click_table_and_clear` | 綠：證據（同類／異類 Top-K）；清除勾選 |
| 10 | `test_ac10_cart_add_skip` | 綠：新增 1、重加略過 1、來源「類別相似度」 |
| 16 | `test_ac16_canonical_threshold_precision` | 綠：0.123456 不被 0.01 刻度蓋掉 |
| 18 | `test_ac18_pointer_held_preview` | 綠：pointer 按住時候選數已變（mouseup 前） |
| 33 | `test_ac33_axis_zoom` | 綠：縮放／重設不改候選數 |
| 34 | `test_ac34_source_change_stale` | 綠：改圖後「結果已過期」 |

其餘 AC 仍由具名單元測試覆蓋（見 `4_PM_Feedback/`）。

---

## 效能（analyze 純函式，本機）

| N | D | C | K | max_block_shape | 延遲 | 峰值記憶體 |
|---|---|---|---|-----------------|------|------------|
| 24 | 384 | 2 | 5 | (24, 24) | 0.024 s | 0.30 MB |
| 500 | 384 | 5 | 5 | (128, 500) | 5.87 s | 8.6 MB |

query_block=128、reference_block=512；第一版硬上限 5,000 筆。E2E 真實 DINOv2 `dinov2_vits14`：24 張 image + 16 物件，整套 8 測約 5 分鐘（含 embedding 快取命中）。

---

## 啟動

```powershell
.venv\Scripts\python.exe -m streamlit run scripts/app.py
```

Visualize Embeddings → Image Classifier 或 Object Detector → 加資料夾 → Run → 展開「標錯偵測／類別混淆」→ 檢視「類別相似度分布」→「計算類別相似度」。

---

## 本輪實作修正（相對早上交接）

- 預覽 mask 改跟 draft policy（否則拖動中候選數不變，AC18 假綠）。
- expander 加 `key=viz_label_quality`，避免 radio／計算 rerun 把面板收合。
- 購物車 `_SOURCE_LABEL['class_similarity'] = '類別相似度'`。

未做（契約後續）：>5000 ANN、每對類別圖、自動改標、Labeling ROI 原生傳遞、評估 JSON 的瀏覽器匯入。
