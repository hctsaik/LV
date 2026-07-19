# AC ↔ 測試對應:30 spatial_sig

設計:[3_Architect_Design/30_spatial_sig.md](../3_Architect_Design/30_spatial_sig.md)
測試:[tests/test_spatial_sig.py](../tests/test_spatial_sig.py)

| AC | 測試 | 說明 |
|----|------|------|
| AC1 | `test_ac1_clustered_center_block` | G1 聚一團(中心塊避開 edge 平手歧義) |
| AC2 | `test_ac2_edge_ring` | G1 偏邊緣 |
| AC3 | `test_ac3_linear_row_and_tie_priority` | G1 線狀 + 平手序 linear 優先 + stats==1.0 |
| AC4 | `test_ac4_random_honest_single_and_fp_rate` | **G2** 隨機單例 none + 200 seeds FP ≤ 0.075 |
| AC5 | `test_ac5_die_mask_contrast` | **G2②裁決核心**:within-mask 誠實 vs 全格點假設誤判 edge 的機器證明 |
| AC6 | `test_ac6_conservative_gates` | die 太少/缺陷太少/無對比 → none+理由 |
| AC7 | `test_ac7_deterministic` | 同 seed 兩次相等 |
| AC8 | `test_ac8_method_fragments` | method line 固定片段(受限重排/Bonferroni/exploratory) |
| AC9 | `test_ac9_error_paths` | 空/壞 flag/NaN/α 出界 → ValueError |
| (推導) | `test_derived_rigid_motion_invariance` | 平移+等比縮放 → stats/p/verdict 不變 |
| (推導) | `test_derived_flag_complement_not_significant_alias` | k_min 邊界 off-by-one 稽查 |

## GUI 接線(M22_gui_wiring)E2E 對應

E2E:[tests/e2e/test_m22_wafer_map_e2e.py](../tests/e2e/test_m22_wafer_map_e2e.py)
(單一流程 `test_m22_wafer_map_flow`,@pytest.mark.e2e,不進 gate 迴圈)

| AC | 斷言段落 |
|----|----------|
| AC-E1 | 猜中欄名→產生→W1「偏邊緣」/W2「看不出模式」;W1 第一 trace 紅點==8(與②一致) |
| AC-E2 | wmap_gaps 含「對不到列 2」「缺座標 1」 |
| AC-E3 | 降級鏈:無 CSV→缺座標欄位(disabled)→無法分片(disabled)→勾同屬一片→"(單片)" |
| AC-E4 | lot L1 + 匯出檔恰 3 件 + CSV verdict W1==edge/W2==none + 來源 sha256 == |
| AC-E5 | 全程在「🧭 晶圓地圖」分頁;工具切換沿 pitfalls #8 |
