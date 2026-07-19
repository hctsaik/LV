# AC ↔ 測試對應:28 run_pairing

設計:[3_Architect_Design/28_run_pairing.md](../3_Architect_Design/28_run_pairing.md)
(含 2026-07-19 反向閘門修訂:`compare_inputs` 契約+G1 方向鏈下沉)
測試:[tests/test_run_pairing.py](../tests/test_run_pairing.py)

| AC | 測試 | 說明 |
|----|------|------|
| AC1 | `test_ac1_exact_alignment` | 交集對齊+only_a/only_b 明講 |
| AC2 | `test_ac2_nonfinite_dropped_visibly` | NaN 排除且列名 |
| AC3 | `test_ac3_empty_ok` | 空輸入不 raise |
| AC4 | `test_ac4_deterministic_sorted` | 字典序決定性 |
| AC5 | `test_ac5_error_paths` | 不可轉 float/非 str key → ValueError |
| AC6 | `test_ac6_compare_inputs_exact` | 錯誤 dict 手算 == |
| AC7 | `test_ac7_g1_direction_chain` | **G1 方向鏈**:A 全對 B 全漏 → 26+28 鏈 a_better(mean_diff=10/16) |
| AC8 | `test_ac8_stale_index_skipped` | 殘留索引跳過不 crash |
| §4 | `test_compare_inputs_error_paths` | confirmed 值非 good/bad、records 缺鍵 → ValueError |
| (推導) | `test_derived_pair_runs_swap_mirror` | 交換鏡像不變量 |
| (推導) | `test_derived_identical_runs_zero_error_diff` | A==B → 錯誤 dict 相等(E2E AC-E2 的資料層保證) |

## GUI 接線(M21_gui_wiring)E2E 對應

E2E:[tests/e2e/test_m21_workpoint_e2e.py](../tests/e2e/test_m21_workpoint_e2e.py)
(單一流程測試 `test_m21_workpoint_flow`,@pytest.mark.e2e,不進 gate 自主修綠迴圈)

| AC | 斷言段落 |
|----|----------|
| AC-E1 | pin B→autoseed→就地重評→pin A→比較:三值結論之一+未考慮批次+詳細含 exploratory/p= |
| AC-E2 | 同結果 pin A/B → 看不出來+無差異(deterministic) |
| AC-E3 | 整圖框選標瑕疵+autoseed → 曲線 SVG+建議工作點+僅試算;ratio 50→1 y 軸刻度改變 |
| AC-E4 | 判可疑數按鈕文字前後 == + 目標資料夾 sha256 前後 ==(零寫入) |
| AC-E5 | 清除 confirmed → 「請先」/「已確認樣本不足」指引,無 exception |
| AC-E6 | 全程停在瑕疵偵測②(無工具切換動作) |
