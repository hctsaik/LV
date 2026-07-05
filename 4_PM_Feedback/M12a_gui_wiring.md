# AC ↔ 測試對應表:M12a GUI 接線(③ 找相似)

設計:[3_Architect_Design/M12a_gui_wiring.md](../3_Architect_Design/M12a_gui_wiring.md)。
真實 E2E:`tests/e2e/test_similar_gui_e2e.py`(`@pytest.mark.e2e`;done 需綠)。純邏輯見 [similarity.md](similarity.md)。

| AC | 說明 | 測試 |
|----|------|------|
| AC-S1 | ③ 找相似:參考依據=類別、選 scratch → 佇列 scratch 佔多數(真實 DINOv2)、渲染縮圖 | `test_s1_s2_class_reference_follows` |
| AC-S2 | 換參考類別(scratch→stain)→ 佇列多數翻群(排序跟著參考類別走) | 同上 |
| AC-S3 | (可選)無 obj_emb 結果 → 友善提示、無 stException | 未獨立實作(以 §4 邊界處理涵蓋;A1 主路徑必有 embedding) |

真實行為鐵則:AC-S1/S2 斷言「最像參考的**真的**是同群物件」(用 label 多數決),非只查 element 存在;
換參考觀察佇列翻群 = 排序確實依參考,非固定畫面。

## 修正(2026-07-05):by-example 重設計後的對應
| AC | 說明 | 測試 |
|----|------|------|
| AC-S1' | 範本=scratch 物件 → 佇列 scratch 佔多數(真實 DINOv2) | `test_by_example_queue_follows_reference` |
| AC-S2' | 換範本=stain 物件 → 佇列翻 stain(排序跟著範本) | 同上 |
| — | 「像某一類」已移除(silent-wrong + 與預標重疊);class_centroid 純函式測試(AC9-11)保留 | `tests/test_similarity.py` |
