# 驗收報告 R4：/loop 第二輪五次（驗收 → 找問題 → 修）

> 日期：2026-06-20　|　接續 R1/R2/R3
> 這輪聚焦前幾輪沒深挖的角落：**檔案寫入路徑、數值正確性、治理流程、匯出健壯性、橫切技術債**。
> 真實資料：valid（55 圖/309 物件）、test（107 圖/550 物件）、10 類。

---

## 五輪總覽

| 輪 | 區域 | 結果 |
|---|---|---|
| 1 | 檔案寫入審計（no-dataset-writes 鐵律） | 🔴 找到並修：**2 個污染源** |
| 2 | 數值正確性（completeness 公式） | ✅ 驗證健康 |
| 3 | 灰帶覆核流程 | ✅ 驗證健康（proposal-only） |
| 4 | 匯出健壯性（CSV） | 🔴 找到並修：**3 處 CSV 引號漏洞** |
| 5 | 橫切技術債 | 🟧 記錄：deprecation（延後安全遷移） |

3 個真修復、2 個驗證健康、1 個記錄延後。

## Loop 1 — 又揪出兩個寫進使用者資料集的源頭 🔴→✅

地毯式 grep 全部 `.write_text/.save/np.savez/.mkdir/open(w)`，比對寫入落點：

1. **manifest.jsonl（嚴重，主流程）**：`manifest_path_for(folder)=folder/"manifest.jsonl"`
   → 寫進**資料集目錄**，而且 **Visualize 每次 Run 都逐資料夾寫**（app.py:2879）。
   這是比之前 object_eval CLI 更常觸發的 active 違規。
2. **umap_ref.pkl**：`ref_path_for(folder,model)=folder/embeddings_<model>/umap_ref.pkl`
   → 寫進資料集（雖被 `**/embeddings_*/` gitignore 擋住 git，但仍在使用者資料夾建檔，
   且與它的 embeddings.npz（在 .lv_cache）分家）。

**修**：兩者都改用 `.lv_cache/<name>_<hash>/`（與 crops/embeddings 同雜湊、同目錄）。
實測：manifest/umap_ref 都落 .lv_cache、資料集保持乾淨、roundtrip 正常。
順帶修了 3 個編碼舊行為的單元測試（2 個原本斷言寫資料集、1 個是 `extract_embeddings`
進度條初始呼叫造成的既有 stale 斷言）。

**其餘寫入點審計皆 OK**：compare `--output-dir`（CLI 指定）、embeddings/crops npz（.lv_cache）、
縮圖（.lv_cache）、profiles/signal_gate（gitignored repo）、Labeling 交棒（使用者匯出夾）、
usage/curation log（output/）。

## Loop 2 — 數值正確性：健康 ✅

`classify_cell`（空/缺/偏缺/假完整/過多 優先序正確）、`coverage_health`
（Σ min(n/t,1)·quality / n_scored，假完整 quality=0.7，NA 排除）、`cell_diversity`
（d∈[0,1]，spread×dim 幾何平均，<2 樣本=0，語意正確：d<d\* 才判 fake）、`_gini`
（實測 empty/single/all-zero → 0.0，無除零/NaN）。公式與邊界都正確，無需修。

## Loop 3 — 灰帶覆核：健康 ✅（proposal-only）

`_gray_dispose` 只寫 session（`gray_disp`），不碰任何檔案；裁決經 download／Labeling
交棒，不寫回資料集——符合「改標只在灰帶、且不直接寫資料集」的設計。無需修。

## Loop 4 — 匯出 CSV 引號漏洞 🔴→✅

主購物車匯出 `snapshots_to_csv` 用 `csv.writer`（安全）。但有 **3 處手寫 f-string CSV**
未跳脫：`cell_candidates.csv`(3834)、`gap_fillers/reference_gaps.csv`(4771)、
`quiz_answers.csv`(5755)。檔名含逗號、或類別名／作答含逗號引號（如 `chair "wood"`）時
**會把 CSV 切錯欄**。**修**：抽 `_rows_to_csv(header, rows)` 走標準 csv 模組，三處統一。
實測 `a,b.jpg` + `chair "wood"` 可正確 roundtrip。

## Loop 5 — 橫切技術債：記錄並延後 🟧

`use_container_width` 在 app.py 有 **157 處**（全 app 更多），已過 Streamlit 宣告的
2025-12-31 移除日 → 跑起來一直冒 deprecation 警告。**但**：在鎖定的 Streamlit 1.58
它仍可運作；且機械式取代有風險（如 `st.image(width=)` 歷史語意是像素寬，盲改可能改壞）。
**判斷**：這是「該遷移、但不該在自動 loop 裡盲改 157+ 處」的技術債——記錄為待辦，
建議用一支**專門、逐元件驗證**的 PR 遷移（`=True`→`width='stretch'`、`=False`→`'content'`），
而非冒險一次性 sed。誠實地不為修而修。

---

## 驗證

- compile + import OK；manifest/umap_ref 落 .lv_cache、資料集乾淨、roundtrip 正常。
- `_rows_to_csv` 對含逗號/引號輸入正確 roundtrip。
- 回歸：manifest+umap 單元 27、cov 10、objcov 11 全綠。

## 一句話總結

延續「**真實資料下會不會出錯**」的視角，這輪最重要的是再揪出**兩個寫進使用者資料集的源頭**
（manifest 是主流程、每次 Run 都寫，影響最大），全部改道 `.lv_cache`——你最在意的第一鐵律
現在三條寫入（crops/embeddings、manifest、umap_ref）終於一致。匯出端也補上了標準 CSV 跳脫。
唯一刻意不動的是 157 處 deprecation：在自動 loop 裡盲改的風險大於效益，留給專門 PR。
