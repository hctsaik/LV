# AC ↔ 測試對應表:M12b GUI 接線(M9/M10 找相似)

設計:[3_Architect_Design/M12b_gui_wiring.md](../3_Architect_Design/M12b_gui_wiring.md)。
真實 E2E:`tests/e2e/test_al_batch_gui_e2e.py`(`@pytest.mark.e2e`)。

| AC | 說明 | 測試 |
|----|------|------|
| AC-BSIM | M9 分批選「🔎 找相似」→ 從②結果挑參考 → 掃出佇列、reason 含「相似度」、無例外 | `test_g8_similar_objective_batch` |

M10 GUI similar 的核心(profile 存取 + run_once 排序)以 **AC-SVC-SIM(al_service 單元)** 覆蓋
(見 [al_service.md](al_service.md));GUI 接線沿用既有 watch 骨架,不另立完整 M10-similar Playwright E2E。
