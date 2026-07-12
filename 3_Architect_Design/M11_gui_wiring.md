# 設計:M11 GUI 接線(預標區塊,`scripts/app.py`)

> 對應 PRD:[2_PO_PRD/prelabel_prd.md](../2_PO_PRD/prelabel_prd.md);純邏輯契約:[11_prelabel.md](11_prelabel.md)。
> Tier B(GUI → false-green 重災區)。**done = 模組 11 gate 綠 AND 真實 Playwright E2E 綠。**
> 本檔只出契約 + 可觀察 AC,不寫實作。

## 1. 目的

在既有瑕疵偵測 wizard 內,讓使用者用凍結分類頭對「② 套用結果」的物件做**分類預標**:
調 min_conf、挑目標類別、**先看預覽表**、再把接受的預標匯出成 YOLO 標註檔到**另選目錄**(C6)。

## 2. 位置與資料來源

- 位置:**③ 挑樣送人工標** tab 內,新增 expander「🏷️ 預標(分類頭代填,人工最終確認)」
  (放在 2×2 取樣矩陣之後;③ 已有 model + apply_result,自然延伸)。
- 讀:`st.session_state["anomaly_model"]`(需 `head`)、`anomaly_apply_result`
  (`records`/`obj_emb`/`scores`/`threshold`)。
- 呼叫 `prelabel.prelabel_records` / `to_yolo_lines` / `export_prelabels`;
  `class_names = head["classes"]`;`source_dirs = anomaly_target_folder`。

## 3. Widget key 契約(PG 必須實作)

| key | 元件 | 說明 |
|-----|------|------|
| `anomaly_prelabel_min_conf` | slider(0.0–1.0,預設 0.5) | 低於此信心 → Unknown → 不 accept |
| `anomaly_prelabel_classes` | multiselect(options=head 類別,預設全選) | 只接受選中類別的預標;空選=不接受任何(全留人工) |
| `anomaly_prelabel_out_dir` | text_input | 輸出資料夾(另選;預設帶 `.lv_cache` 下 `prelabel_out` 建議路徑) |
| `anomaly_prelabel_export_btn` | button(primary) | 匯出;on_click 走 `export_prelabels` |
| `anomaly_prelabel_preview` | container | 預覽表:每列 pred·conf·accept(+ 縮圖 `stImage`) |

DOM 穩定標記(供 E2E 等待,避免 rerun 競態):
- 預覽就緒:預覽容器內含文字「接受 N」(N=accept 數,主畫面可讀)。
- 匯出完成:主畫面出現「✅ 預標匯出完成」(含「預標匯出完成」)。

## 4. 邊界與錯誤路徑(GUI 友善訊息,不吐原始 exception)

| 情境 | GUI 行為 |
|------|---------|
| 模型無 `head` | expander 內顯示 `st.info("此模型無分類頭,無法預標。請到①用**瑕疵類別**標籤(每類達 N_min)重建。")`;**不出匯出鈕**,不 crash |
| 尚無 ② 套用結果 | `st.info("請先到②套用偵測產生結果,才能預標。")` |
| out_dir 空 | 匯出鈕 disabled |
| `export_prelabels` raise(C6/含 images/) | `st.error(f"匯出失敗(安全檢查):{exc}")`,不寫、不 crash |
| target multiselect 全空 | 預覽 accept 數=0;匯出鈕可按但寫 0 檔並提示「無可匯出預標」 |

## 5. Acceptance Criteria(真實行為;給 `/pm` 落 `@pytest.mark.e2e`)

> E2E 用 `yolo_defect_at_nmin`(scratch×8+stain×8,語義 defect → 建模含 head)。
> 流程:enter → build_model(語義=瑕疵類別)→ set_model_dir+存 → apply_model(②)→ click ③。

- **AC-E1(預標區塊 + 預覽出現)**:含 head 模型 + ② 套用後,③ 出現「🏷️ 預標」expander;
  展開後預覽容器 `anomaly_prelabel_preview` 內出現 pred/conf/accept 相關文字與 ≥1 個 `stImage` 縮圖。
- **AC-E2(accept 數與匯出一致)**:設 `min_conf` 低(如 0.0)、目標類別全選 → 預覽顯示「接受 N」(N≥1);
  按匯出 → `out_dir/labels/` 下 `.txt` 檔內**總行數 == N**(預覽宣稱的 accept 數即真實寫出的行數)。
- **AC-E3(匯出真實檔案 + cls_id 正確)**:匯出到 tmp `out_dir` → 讀回任一 `labels/*.txt`,
  每行首欄 cls_id ∈ `{0,1}`(對 head 類別 scratch=0/stain=1),座標 4 欄且 ∈ [0,1];
  `out_dir/classes.txt` 內容含 `scratch` 與 `stain`。
- **AC-E4(C6 來源零寫入)**:完成 AC-E3 後,**來源資料夾 `ds["root"]` 內不存在任何新寫入**
  (無新的 labels 覆寫;比對匯出前後 `ds["root"]` 檔案集合不變)。
- **AC-E5(無 head 友善降級)**:另建**物件類別**語義模型(無 head)→ ③ 預標 expander 顯示
  「無分類頭…」提示、**無匯出鈕**、頁面無 `stException`。
- **AC-E6(Should,可延後)**:M10 監看佇列每項顯示 head 建議(`建議:<類別>`)+「採納建議」→
  `append_label(source="prelabel_accepted")`;下輪掃描該項移出 pending。
  *appetite 內優先 AC-E1~E5;E6 時間許可再做,否則進候選並在 ROADMAP 記一行。*

## 6. 反向閘門備註

若實作發現 ③ 的 `obj_emb`/`scores` 在某模式下不可得(如純 patch 無物件 embedding),
停手回報 `/architect`:預標需要物件級 embedding + head,patch-only 模型不適用 →
應在 expander 明示「此模型型態不支援預標」,而非硬跑。
