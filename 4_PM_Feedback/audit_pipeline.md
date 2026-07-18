# 25 audit_pipeline — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/25_audit_pipeline.md](../3_Architect_Design/25_audit_pipeline.md) · 里程碑 M20 · **Tier B**
單元/整合測試:`tests/test_audit_pipeline.py`(gate:`python verify/gate.py audit_pipeline`)
E2E:`tests/e2e/test_dataset_audit_e2e.py`(`@pytest.mark.e2e`,**不進 gate**,真實 DINOv2)

## 植入手算(單元與 E2E 同構;fixture 錨點)

可讀 25 張(train/classA 12、classB 7、classC 2、val/classA 4)+壞檔 1;
exact 額外複本 **5**(dup×3+leak×1+conflict×1——**全域計**,設計 24 §2 明定);
leak 對 **1**;conflict 組 **1**;val 缺類 **["classB","classC"]**;獨特 sha **20**;
CSV:T1×19、T2×4、bogus×5、classC 2 張無列。

## 單元/整合 AC ↔ 測試(注入 embed_fn)

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | 六訊號植入數字逐項全對(25/1/5/1/1/2+缺類清單) | `test_ac1_planted_numbers_all_exact` | 🔴 |
| AC2 | metadata:unmatched 列 5/影像 2、matched 23、tool 分組 {T1:19,T2:4} | `test_ac2_metadata_join_numbers` | 🔴 |
| AC3 | 無 CSV → 無 metadata 節,其餘照常 | `test_ac3_no_csv_no_metadata_section` | 🔴 |
| AC4 | use_embedding=False → embed_fn 0 呼叫、emb 兩節「未計算」、其餘不變 | `test_ac4_use_embedding_off` | 🔴 |
| AC5 | 內容定址快取:首輪呼叫==20(同 sha 不重算)、重跑 0 | `test_ac5_embedding_cache_by_sha` | 🔴 |
| AC6 | 匯出:三檔非空、json 往返 value==5、issues.csv==21 列(5+1+2+10+2+表頭)、**來源零寫入**、out 在 root 內拒 | `test_ac6_export_roundtrip_zero_source_writes` | 🔴 |
| AC7 | 單 split → coverage「不適用」;平鋪 → conflict 0 | `test_ac7_single_split_and_flat` | 🔴 |
| AC8 | 防呆(路徑入訊息/單張)+ progress 單調終值 ≥0.99 | `test_ac8_validation_and_progress` | 🔴 |
| AC9 | 注入遠向量純色圖 → outliers 第一名 | `test_ac9_outlier_top1_is_planted` | 🔴 |

## GUI / E2E AC(Tier B done 要件;真實 DINOv2)

| AC | 內容 | 測試 |
|----|------|------|
| AC-G1 | 端到端植入數字全對(**以匯出 report.json 斷言**,防 DOM 假綠)+ 完成 banner | `test_audit_g1_g2_g3_end_to_end` |
| AC-G2 | 真實 DINOv2:純色離群圖入 outliers 前 10;emb_near_dup 節存在含 method(數值誠實不釘死) | 同上 |
| AC-G3 | 來源 root rglob+size+mtime 前後全等 | 同上 |

## GUI 契約釘定(PM 具體化;PG 必須照用)

| 元件 | key | 備註 |
|------|-----|------|
| 資料夾輸入 | `adt_root` | text_input,fill+Enter 可驅動 |
| metadata CSV 輸入 | `adt_csv` | text_input(選填);E2E 只走文字輸入 |
| 產生按鈕 | `adt_run` | root 非資料夾時 disabled |
| 輸出資料夾 | `adt_out_dir` | text_input |
| 匯出按鈕 | `adt_export` | 成功以磁碟檔案出現為準 |
| 完成 banner | 含「**體檢報告完成**」 | **唯一訊號**:❓ popover/說明文字禁用此字串([[streamlit-e2e-pitfalls]]) |
| 工具切換 | 「🩺 資料體檢」 | E2E 用去 emoji `get_by_text("資料體檢", exact=False).first`(pitfalls 第 8 條) |

## done 判準

- **單元綠**:`python verify/gate.py audit_pipeline`(連同 `meta_join`、`audit_report`)印 `GREEN`。
- **AND 真實 E2E 綠**:AC-G1 是防 false-green 核心——「畫面有報告」≠「數字對」,
  必須以匯出 JSON 對植入已知數逐項斷言;AC-G2 是 embedding 訊號的真實行為證明。
  跑前先清殘留 streamlit server(skill 鐵則)。
