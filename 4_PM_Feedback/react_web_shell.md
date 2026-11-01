# 31 `react_web_shell` — AC ↔ 測試對應表

設計：[3_Architect_Design/31_react_web_shell.md](../3_Architect_Design/31_react_web_shell.md) · PRD：[2_PO_PRD/react_web_migration_prd.md](../2_PO_PRD/react_web_migration_prd.md) · Case：`LV-WEB-01-IC-01`

| AC | 要證明的行為 | 測試／證據 | 狀態 |
|---|---|---|---|
| AC1 | classifier 資料探索、類別、split 與排序符合 Streamlit 基準 | `tests/test_react_web_shell.py::test_case_lv_web_01_ic_01_discovery` | 通過；API／排序與標籤測試 |
| AC2 | 同一模型與 fixture 產生 4×384 真實 embedding，結果有限且非零 | 真實 DINOv2 E2E：`tests/e2e/test_react_web_shell_e2e.py::test_lv_web_01_ic_01_real_model` | 通過；權重 SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`；本機 Python 3.14 AMD64，Windows ARM64 尚待驗 |
| AC3 | React 顯示 4 個投影點；點擊後顯示對應影像、metadata、類別及 split 篩選 | `tests/e2e/test_react_web_shell_e2e.py::test_lv_web_01_ic_01_real_model` | 通過；真模型、可見 UI、篩選與原圖讀回 |
| AC4 | 缺模型、空資料與取消可見；不默默成功或替代 backend | API cases + `test_lv_web_01_ic_01_missing_model_is_visible` + `test_lv_web_01_ic_01_cancel_is_visible_in_react` | 通過；缺模型、空資料、取消均有可見狀態 |
| AC5 | manifest SHA-256 embedding 快取可重用，來源圖片 bytes 不變，輸出僅在隔離 cache root | `test_case_lv_web_01_ic_01_manifest_cache_and_source_immutability` + `test_case_lv_web_01_ic_01_changed_content_invalidates_embedding` | 通過；重跑重用四張，改圖只重算一張，來源 hash 不受分析流程更動 |
| AC6 | NativeApp 選資料夾後同源 API 可完成真模型操作，worker 停止後無孤兒程序 | NativeApp visible UI E2E receipt | 待驗；React 尚未接入 NativeApp 發布包、OS folder grant 和 worker lifecycle |

## 完成判準

AC2、AC3、AC6 須以真實權重和可見 UI 通過；AC1、AC4、AC5 可先由 deterministic 單元／API 測試覆蓋。AC1–AC5 已在本機完成；AC6 與 Windows ARM64 驗證仍未完成，因此 `LV-WEB-01-IC-01` 及 M24 不能標為完成。任何 `skip`、fake backend、截圖-only、HTTP-only 或模型健康檢查均不得將此 case 標為完成。
