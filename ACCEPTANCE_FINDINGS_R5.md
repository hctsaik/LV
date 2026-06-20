# 驗收報告 R5：Multi-agent /loop 十輪（驗收 → 找問題 → 修）

> 日期：2026-06-20　|　接續 R1–R4
> 模式：每輪派 general-purpose agent 以「完全使用者」立場深度走查一個尚未深挖的角落，
> 用 valid/test 真實數據找**可復現的真 bug**（帶 `檔案:行號`＋複現），修復由主程式做＋回歸。
> 真實資料：valid（55 圖/309 物件）、test（107 圖/550 物件）、10 類。

---

## 十輪總覽

| 輪 | 區域 | 結果 |
|---|---|---|
| 1 | 以圖/以文搜圖鏈 | ✅ 健康 + 修 O1（加入購物車帶 cosine 分數） |
| 2 | 選樣/離群/標籤分歧 計算 | ✅ 健康（公式、index、不均衡全對） |
| 3 | 體檢卡 H1–H5 | ✅ 健康 + 補透明 caveat |
| 4 | autotune / profile match | 🔴 修 `classes_for` 巢狀解析 |
| 5 | Labeling 交棒 | 🔴 修 B1 idempotency 丟 spec |
| 6 | evaluate_detections | ✅ 健康 + 補 ns_mismatch 警示 |
| 7 | quiz 一致性（Fleiss/Cohen） | 🔴 修 `fleiss_kappa` ragged（廣義化） |
| 8 | 路徑/壞圖/邊界 | 🔴🔴 修 超大圖 OOM + bbox 越界崩潰 |
| 9 | manifest/去重/session | 🔴🔴 修 ui_state 啟動崩 + 去重 O(N²) OOM |
| 10 | 3D/fragment/降維邊界 | 🔴🔴 修 監督UMAP+3D+n=4 崩 + fragment 不換頁 |

**真修復 ~11 個、驗證健康 3 個、順帶修 3 個過時測試。** 後半輪 bug 密度明顯升高——越深的角落越少被測到。

## 真修復明細

- **L1 O1**：以圖/以文搜圖「加入清單」沒把 UI 顯示的 cosine 距離寫進購物車 `score`，下游（考卷分歧度、灰帶排序）拿不到 → 補 `score=float(d)`。
- **L4 `classes_for`**（object_eval.py:87）：只探 2 層，漏祖父層 classes.txt（`[Small]` 巢狀）→ autotune/Visualize 面板類別名退化成 `class_N`、與 Compare/ObjCov 不一致 → 探到第三層。
- **L5 B1**（labeling_handoff.py:209）：idempotency 只看 `(source, set_hash)`，同批圖改了 task/class/labels/payload 會**靜默重用舊資料夾、丟棄新 spec** → 把 spec 納入重用鍵（spec_hash）。
- **L7 fleiss_kappa**（quiz.py:173）：每列評分人數不等時靜默回錯值（ragged→1.0）→ 改用**廣義 Fleiss**（per-item n_i），等列時與教科書一致。
- **L8 BUG-1**（manifest/interaction/object_eval/app 6 處）：超大圖(>179MP) 的 `DecompressionBombError` **非 OSError 子類** → `except OSError` 攔不到 → 整批崩 → 各處 except 加上它。
- **L8 BUG-2**（interaction.py:1109）：YOLO 框 cx/cy>1 沒夾上限 → `crop_bbox` ValueError → 夾 x0/y0 到 img-1 並逼 x1>x0，守住「永遠正面積」契約。
- **L9 B1**（app.py:349）：`.lv_ui_state.json` 是合法 JSON 但非 dict（`[]`/`null`）→ `data.items()` AttributeError → **app 啟動崩** → 加 `isinstance(data, dict)` 守門。
- **L9 B2**（interaction.py:622/653）：去重 `max_pairs` 只截回傳、不截 O(N²) 候選 materialize → 近重複密集資料 OOM → 改用**大小受限的堆**只保留 max_pairs 個最近對（結果與原本一致、記憶體 O(max_pairs)）。
- **L10 BUG1**（app.py:_cov_project5 / _objcov_project）：監督UMAP + 3D + 樣本恰=4 → UMAP spectral `eigsh k>=N` 崩（cmpc 有守衛、這兩處漏）→ 補 `len>dim+1` 才走監督、否則退 PCA。
- **L10 BUG2**（app.py:_viz_send_to_gray）：此鈕在右欄 fragment 內，callback 只重跑 fragment、漏設 `_cart_app_rerun` → 「送進灰帶覆核」點了不換頁 → 補旗標跳出做 app rerun。

## 驗證健康（無 bug，不為修而修）

- **L2**：farthest-point 是真 greedy max-min（對照參考實作逐位元相同）、離群度/標籤分歧定義與方向正確、不均衡不誤判少數類、下游 index 無錯位。
- **L3**：H1–H5 三訊號方向/桶分類/邊界/caveat/小樣本守門全對，整類缺失 couch 不被誤歸，HTML 匯出不寫資料集。
- **L6**：IoU/greedy 配對/recall-precision-FN-FP/逐型態/灰帶排除/CSV 容錯全與手算吻合。

## 記錄、未在本輪改（風險/成本考量）

- 去重 embedding 端 sklearn `radius_neighbors` 對極密集叢集仍會先 materialize O(N²) 鄰居（我的堆只界定了回傳側）——徹底解需 ANN/分塊。
- CSV 匯出無 UTF-8 BOM → 中文類別在 Excel 亂碼（純顯示，且改動牽涉多個匯出與既有測試）。
- EXIF 方向未套用（embedding/縮圖看未轉正像素）——修了會使既有 embedding 快取失效。
- kappa prevalence 敏感、N=1 kappa、consensus tie-break 受上傳順序影響——次要統計細節。

## 驗證

- compile-all + import-all OK。
- 單元測試 **138 passed**（含順手修好的 3 個過時 thumbnail/progress 測試）。
- e2e 各自單獨：cov 10、objcov 11 全綠（合併跑的污染是既有 plotly-patch 測試隔離問題）。
- 關鍵修復皆有對抗性實證：bbox(cx=1.5) 不崩且正面積、fleiss ragged=0.216(原 1.0)且教科書值不變、去重堆結果==brute-force、ui_state 非 dict 降級不崩。

## 一句話總結

這十輪把驗收推進到「**最深、最少被測的角落**」，後半段冒出一串會**直接崩潰或靜默吃資料**的真 bug：超大圖/越界框讓整批崩、壞 ui_state 讓 app 開不起來、去重在密集近重複下 OOM、監督UMAP 在 4 樣本崩、fragment 內送灰帶點了不換頁。全部修掉並以對抗性案例驗證；同時確認搜尋、量測、體檢卡、評估這些核心計算在真實數據上數值正確、無需改動。
