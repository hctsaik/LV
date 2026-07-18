# （已併入正典）半導體分析 × Active Learning 功能缺口 → 見 5_/09

本文件原為 2026-07-19 的獨立功能缺口盤點(工作台視角:每項附植入驗證法與工程量粗估)。
同日與另一份獨立盤點(決策可信度視角)合併,**唯一正典**:

→ [5_active_learning_product_review_2026-07-12/09_半導體分析與Active_Learning下一階段功能.md](5_active_learning_product_review_2026-07-12/09_半導體分析與Active_Learning下一階段功能.md)

保留本檔僅作路標。合併摘要:

- 兩份獨立收斂:wafer 空間、per-domain(tool/recipe)差異、時間軸/drift、
  promotion gate、Similar Case 處置層、風險/成本受限選樣、不做清單。
- 09 正文抓到本檔漏掉的根本項:**group/time-aware split 洩漏**(同 wafer 跨 train/val)、
  **selection bias/隨機哨兵**、量測系統健康、盲標 audit。
- 本檔併入 09 的獨有內容(見其「2026-07-19 合併」節):**群組批次標註**(P0.5)、
  兩個 quick win(Tool/Recipe 顯著性、成本工作點)、wafer/案例庫/預算器最小版設計、
  **植入式驗證設計原則**、合併後排序共識。
- 待使用者拍板:資料是否有 die/wafer 座標欄位(決定 wafer spatial 的優先序)。
