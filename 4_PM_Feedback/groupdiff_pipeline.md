# 22 groupdiff_pipeline — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/22_groupdiff_pipeline.md](../3_Architect_Design/22_groupdiff_pipeline.md) · 里程碑 M19 · **Tier B**
單元/整合測試:`tests/test_groupdiff_pipeline.py`(gate:`python verify/gate.py groupdiff_pipeline`)
E2E:`tests/e2e/test_groupdiff_e2e.py`(`@pytest.mark.e2e`,**不進 gate**,真實 DINOv2)

## 單元/整合 AC ↔ 測試(注入 extractor,免真模型)

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | 端到端合成:verdict True、rank1 峰=植入 patch 5、z_map (4,4)、text 含 %、路徑/bbox_px/計數契約 | `test_ac1_end_to_end_synthetic` | 🔴 |
| AC2 | null 誠實:同分布兩夾 ×3 組 → verdict True ≤ 1 組 | `test_ac2_null_honest` | 🔴 |
| AC3 | 快取:同參數重跑 extractor 0 呼叫(首輪恰 16 次)、verdict 不變 | `test_ac3_cache_hit_skips_extractor` | 🔴 |
| AC4 | grid 不一致:剔除 + `skipped["grid"]==1` + alignment_warning + warn_msgs,其餘照常 | `test_ac4_grid_mismatch_skipped_with_warning` | 🔴 |
| AC5 | 防呆:不存在資料夾(訊息含路徑)/ 可讀 <2 / export 進來源(含子目錄與相等)→ ValueError | `test_ac5_validation_errors` | 🔴 |
| AC6 | 匯出往返:5 檔存在非空、CSV 行數/rank 序、summary.json 一致、**來源 rglob+大小前後全等** | `test_ac6_export_roundtrip_and_zero_source_writes` | 🔴 |
| AC7 | progress:frac 單調不減、終值 ≥0.99、text 非空 | `test_ac7_progress_monotone` | 🔴 |
| AC8 | pca2d:形狀 (N,2)、evr 遞減;<3 樣本 ValueError | `test_ac8_pca_side_evidence` | 🔴 |
| AC9 | 對照放大圖:good/bad 各 ≤ n_each 張 PIL、尺寸>0;代表圖被刪 → 跳過不炸、數量 -1 | `test_ac9_region_compare_images` | 🔴 |

## GUI / 整合 E2E AC(Tier B done 要件;真實 DINOv2 + 合成植入影像)

| AC | 內容 | 測試 |
|----|------|------|
| AC-G1 | 植入命中:good(16) vs bad(12,固定區塊瑕疵)→ 成功 banner;匯出 regions.csv rank1 中心落在植入區 ±1 patch(**檔案級驗證,防 DOM 假綠**);region_01.png / heatmap_good.png 非空 | `test_g1_planted_difference_found_and_localized` |
| AC-G2 | null 誠實:同源兩夾 → 「未發現穩定差異」警示;成功文案**不得**出現 | `test_g2_null_split_is_honest` |
| AC-G3 | 來源零寫入:兩測各自前後 rglob+size+mtime 全等(嵌在 G1/G2 內) | 同上兩測 |

## GUI 契約釘定(PM 具體化設計 §2.2 的 key 前綴約定;PG 必須照用)

| 元件 | key | E2E 依賴 |
|------|-----|----------|
| Good 資料夾輸入(text_input) | `gpd_good_dir` | `.st-key-gpd_good_dir input` fill+Enter |
| Bad 資料夾輸入(text_input) | `gpd_bad_dir` | 同上 |
| 「🔬 分析差異」按鈕 | `gpd_run` | disabled→enabled 後點擊 |
| 匯出資料夾輸入(text_input) | `gpd_out_dir` | fill+Enter |
| 「📦 匯出報告」按鈕 | `gpd_export` | 點擊後以磁碟檔案出現為成功訊號 |
| verdict 成功 banner 文案 | 含「**兩群存在穩定的局部差異**」 | 唯一成功訊號 |
| verdict null 警示文案 | 含「**未發現穩定差異**」 | 唯一 null 訊號 |

⚠ **唯一訊號紀律**([[streamlit-e2e-pitfalls]]):上兩句文案**只准出現在 verdict banner**,
不得寫進 ❓ 說明 popover / caption / 靜態文字,否則 E2E 誤觸假綠。

## 反向閘門紀錄

- **/pg → /pm(2026-07-18,E2E 首跑)**:`_enter_tool` 用 `get_by_text("🧪 差異探索",
  exact=True)` 抓不到工具鈕——Streamlit 會把 label 的 emoji 拆成獨立節點,`exact=True`
  永遠 miss(fewshot E2E 當初同坑,已證解法=去 emoji + `exact=False` + `.first`)。
  PM 修 locator(僅點擊定位,**所有斷言不變**);app 端無錯。

## 誠實注記(de-risk)

- AC2(單元)與 AC-G2(E2E)的 null verdict 依賴固定 seed 的確定性實現(理論偽陽率
  α=0.05)。單元已用 3 組集合斷言壓低風險;E2E 為單組,若正確實作觸紅且可論證為
  seed 不幸,走 `/pg → /pm` 反向閘門換合成資料 seed 並記決策日誌。

## done 判準

- **單元綠**:`python verify/gate.py groupdiff_pipeline` 印 `GREEN`。
- **AND 真實 E2E 綠**:AC-G1 是防 false-green 核心——「頁面有熱圖」不等於「差異真的在
  植入處」,必須以匯出 CSV 的 rank1 座標對已知植入區做真實行為斷言;AC-G2 是 G2 誠實
  判定的端到端證明。跑 E2E 前先清殘留 streamlit server(`.claude/skills/visuallatent-e2e` 鐵則)。
