# AC ↔ 測試對應:29 wafer_grid

設計:[3_Architect_Design/29_wafer_grid.md](../3_Architect_Design/29_wafer_grid.md)
測試:[tests/test_wafer_grid.py](../tests/test_wafer_grid.py)

| AC | 測試 | 說明 |
|----|------|------|
| AC1 | `test_ac1_guess_fields_exact` | 欄名猜測 casefold 全等/先出現優先/沒中 None |
| AC2 | `test_ac2_build_exact` | 物件聚合 max + unmatched/missing_coord 分列 |
| AC3 | `test_ac3_g6_no_guessing` | **G6**:不猜 wafer;明示才 "(單片)" |
| AC4 | `test_ac4_grouping_sorted_and_lot` | 分片字典序 + n/k 手算 + lot |
| AC5 | `test_ac5_png_pixels_exact` | **G3** PIL 像素 exact(顏色/尺寸釘死) |
| AC6 | `test_ac6_missing_wafer_value` | wafer 值空 → missing_wafer |
| AC7 | `test_ac7_error_paths` | x_field=None / 空 dies → ValueError |
| AC8 | `test_ac8_deterministic` | 同輸入兩次相等 |
| (推導) | `test_derived_empty_items_ok` | 空 items 邊界 |
| (推導) | `test_derived_png_translation_invariance` | 座標平移 → 圖 bytes 相同 |
| (推導) | `test_derived_single_die_png` | 單 die 1 格畫布 |
