# 對應表:anomaly_heatmap

設計:[3_Architect_Design/05_anomaly_heatmap.md](../3_Architect_Design/05_anomaly_heatmap.md)
測試:`tests/test_anomaly_heatmap.py`(gate:`python verify/gate.py anomaly_heatmap`)

| AC | 測試 | 說明 |
|----|------|------|
| AC1 | test_output_size_and_mode | 輸出尺寸==crop、RGB |
| AC2 | test_peak_localization | 峰值落在指定 cell 的空間範圍內 |
| AC3 | test_constant_map_no_nan | 常數圖無 NaN |
| AC4 | test_alpha_endpoints | 推導:alpha=0→crop、alpha=1→與 crop 無關 |
| AC5 | test_deterministic | 決定性 |
