# M14c 設計:以樣搜樣匯出一併複製影像(standalone YOLO 資料集)

## 目的
讓以樣搜樣匯出成一個**可直接訓練的 YOLO 資料集**(`images/` + `labels/` + `classes.txt`),
而非只有 labels(需自行對回來源影像)。守 C6:只讀來源、只寫使用者選的 out_dir。

## 背景與安全設計
`prelabel.assert_safe_prelabel_dir` 現有兩道檢查:
1. **out_dir 含 `images/` → raise**(啟發式:看起來像既有資料集,避免覆寫)。
2. **out_dir 等於/位於/包含任一 source_dir → raise**(真正的 anti-clobber)。

要放 `images/` 進 out_dir 會撞到檢查 1。安全解法:**只放寬檢查 1(僅限「明確要建資料集」的匯出),
保留檢查 2**(它才是防寫來源的核心)。因此:
- **不弱化 C6**:來源關係檢查照跑;out_dir 是使用者另選的輸出夾(預設 `.lv_cache/<hash>/fewshot_out`),非來源。

## prelabel(M11)加法
- `assert_safe_prelabel_dir(out_dir, source_dirs=(), allow_images=False)`:
  - `allow_images=False`(**預設,M11 行為不變**):含 `images/` → raise。
  - `allow_images=True`:跳過 `images/` 啟發式;**來源關係檢查仍執行**。
- `export_prelabels(..., allow_images=False)`:透傳給上面的 assert。其餘不變。

## retrieval_export(M13)加法
- `export_retrieval(records, decisions, out_dir, *, class_names, source_dirs=(), copy_images=False) -> dict`:
  - `copy_images=True`:
    1. 呼叫 `export_prelabels(..., allow_images=True)`(允許 out_dir/images/ 存在 → 支援重複匯出)。
    2. 寫完 labels 後,把**每個有匯出標註(decision ∈ {accepted, relabeled} 且有 final_class)的來源影像**,
       以 `shutil.copy2` 複製到 `out_dir/images/<原檔名含副檔名>`。同名去重(集合)。
    3. 回傳增加 `"images_copied": int`(不重複來源影像數)。
  - `copy_images=False`(**預設**):行為與現狀完全相同(無 images/、無 images_copied 或 0),既有測試無回歸。
  - C6:只讀來源影像、只寫 out_dir;略過(skipped/pending)的物件其**獨有**影像不複製。

## GUI(app.py `_fewshot_export_controls` / `_fewshot_export`)
- 匯出區加 checkbox `一併複製影像(images/,可直接訓練)`(key=`fewshot_copy_images`,**預設勾**)。
- `_fewshot_export` 讀該 checkbox → 傳 `copy_images` 給 `export_retrieval`;成功訊息附「+ N 張影像」。

## Acceptance

### prelabel(tests/test_prelabel.py)
- **AC-PIMG-1**:`assert_safe_prelabel_dir(out含 images/, allow_images=True)` 不 raise;
  同目錄 `allow_images=False`(預設)→ raise(ValueError,訊息含「資料集」或「images」)。
- **AC-PIMG-2**:`allow_images=True` 但 out_dir 等於某 source_dir → **仍 raise**(來源關係檢查不受 allow_images 影響)。

### retrieval_export(tests/test_retrieval_export.py)
- **AC-RIMG-1**:`export_retrieval(copy_images=True)`(2 張圖各 1 個 accepted 物件)→ `out_dir/images/` 含這 2 張
  (檔名 == 來源檔名、且與 `labels/<stem>.txt` 對應);回傳 `images_copied == 2`。
- **AC-RIMG-2**:`copy_images=True` 後**來源資料夾快照不變**(C6);對**同一 out_dir 再匯出一次不 raise**
  (images/ 已存在但 allow_images)。
- **AC-RIMG-3**(不變量):`copy_images=False`(預設)→ **無 `images/`**、`images_copied == 0`;
  既有 retrieval_export 測試全綠(無回歸)。
- **AC-RIMG-4**:只有 accepted/relabeled 的影像被複製——某張圖的物件全被 skip → 該圖不進 images/。

### E2E(tests/e2e/test_fewshot_scenarios_e2e.py::test_s02)
- S2 更新:匯出(預設含影像)→ `out_dir/images/` 有影像檔、且與 `labels/*.txt` stem 對應;來源零寫入(C6)。

## 非目標
不做影像格式轉換/縮放、不做 train/val 切分、不處理跨資料夾同名碰撞的重新命名(沿用既有 stem 行為,
碰撞以後覆寫;若日後成問題再處理)。copy_images=False 時一切照舊。
