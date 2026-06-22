# 對應表:patch_features

設計:[3_Architect_Design/01_patch_features.md](../3_Architect_Design/01_patch_features.md)
測試:`tests/test_patch_features.py`(gate:`python verify/gate.py patch_features`)

| AC | 測試 | 說明 |
|----|------|------|
| AC1 | test_square_crop_grid_and_shape | 224 方形 → grid(16,16)、feats(256,384) |
| AC2 | test_square_crop_grid_and_shape | per-token L2 範數=1 |
| AC3 | test_grid_product_equals_P_invariant | P==gh*gw 不變量(夾具多形狀) |
| AC4 | test_cache_roundtrip_no_recompute | 快取命中不重算、值相等(float16 容差) |
| AC5 | test_no_dataset_pollution | 資料集目錄零新增 [[no-dataset-writes]] |
| AC6 | test_bad_image_yields_zero_grid | 壞圖→grid(1,1)、零向量、不丟例外 |

注:用 fake extractor 注入,免真模型。屬「掛名覆蓋」,行為正確性仍靠人審 + E2E。
