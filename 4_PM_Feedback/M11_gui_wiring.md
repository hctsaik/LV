# AC ↔ 測試對應表:M11 GUI 接線(預標)

設計:[3_Architect_Design/M11_gui_wiring.md](../3_Architect_Design/M11_gui_wiring.md)。
真實 E2E:`tests/e2e/test_prelabel_gui_e2e.py`(`@pytest.mark.e2e`;done 需綠)。
純邏輯模組對應表見 [prelabel.md](prelabel.md)。

| AC | 說明 | 測試 |
|----|------|------|
| AC-E1 | ③ 出現「🏷️ 預標」+ 預覽(pred/conf/accept + 縮圖) | `test_e1_e4_prelabel_export_readback` |
| AC-E2 | 匯出檔案總行數 == 預覽宣稱 accept 數(所見即所匯) | 同上 |
| AC-E3 | 讀回 labels/*.txt:cls_id∈{0,1}、座標∈[0,1];classes.txt 含 scratch/stain | 同上 |
| AC-E4 | 來源資料夾零寫入(C6 真實不變量,比對前後檔案集合) | 同上 |
| AC-E5 | 無 head 模型 → 友善提示「無分類頭」、無匯出鈕、無 stException | `test_e5_no_head_graceful` |
| AC-E6 | (Should)M10 佇列採納建議 → append_label(source=prelabel_accepted) | **未實作**(appetite 內優先 E1–E5;進候選,ROADMAP 已記) |

真實行為鐵則落實:E2E **不只查 element 存在** —— AC-E2/E3 真的讀回磁碟 .txt 斷言行數與 cls_id,
AC-E4 比對來源資料夾前後檔案集合(C6 不是靠一句 assert,是真實不變量)。E1–E5 已綠(2/2);
E6 未做(進候選)。
