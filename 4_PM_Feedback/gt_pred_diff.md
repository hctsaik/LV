# 10 gt_pred_diff — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/10_gt_pred_diff.md](../3_Architect_Design/10_gt_pred_diff.md) · 里程碑 M10 · Tier B(核心純函式為 A 級品質)
單元測試:`tests/test_gt_pred_diff.py`(gate:`python verify/gate.py gt_pred_diff`)
E2E:`tests/e2e/test_gt_pred_diff_e2e.py`(`@pytest.mark.e2e`,**不進 gate / PG 自主修綠迴圈**)

> 設計反向閘門紀錄(兩次):
> 1. **AC1 第三式(2026-07-04)**:PM 轉抄時算出「聯集 .3125 → IoU 0.6」與初版 pin 的 3/7 不符;
>    architect 修正為 **0.6**(見設計 §6 自述)。
> 2. **/pg → /architect(2026-07-12)**:seed 寫在 `images/<sha>.json`(xAnyLabeling 唯一會自動載入的
>    路徑)會與既有回讀端的「該檔有 label ⇒ 已標註」語義衝突 → **模型的預測會被當成人工答案讀回**。
>    architect 裁決:seed 內容 sha 記進 spec,回讀兩入口一律跳過未變更的 seed。新增 **AC9**。

## 單元 AC ↔ 測試

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | IoU 釘死:重合 1.0 / 相接 0.0 / 位移 .125 → **0.6** | `test_ac1_iou_identical_and_disjoint` | 🔴 |
| AC2 | tp/fn/fp:命中其一 + 遠框 → tp[(0,0,iou)]、fn[1]、fp[1] | `test_ac2_tp_fn_fp` | 🔴 |
| AC3 | 類別混淆:同位置異類 → confused,tp 為空 | `test_ac3_class_confusion` | 🔴 |
| AC4 | 貪婪決定論:高 conf 搶到 tp;重跑 100 次相同 | `test_ac4_greedy_determinism` | 🔴 |
| AC5 | low_conf(含已匹配者)與 score==4.0 | `test_ac5_low_conf_and_score` | 🔴 |
| AC6 | pred 佈局雙形:labels/ 子目錄 vs 平鋪 | `test_ac6_pred_layout_dual` | 🔴 |
| AC7 | scan 依 score 降冪 + 兩邊皆空剔除 | `test_ac7_scan_orders_by_score_and_drops_empty` | 🔴 |
| AC8 | seed_source:pred 無 pred_root → ValueError;seed 框數==pred 框數且帶 conf | `test_ac8_seed_source_pred_requires_pred_root`<br>`test_ac8_pred_seed_lands_with_conf` | 🔴 |
| AC9 | **未動過的預標不是標註**:送標當下 n_annotated==0、讀回 label 為 None、`apply_readback` 不改任何 label;人動過後才計入 | `test_ac9_untouched_seed_is_not_a_label` | 🔴 |

## 推導測試(設計未明列,PM 自加施壓)

| 測試 | 施壓的不變量 |
|------|-------------|
| `test_metamorphic_iou_properties` | IoU 自反==1 / 範圍[0,1] / 對稱 / 退化框(w≤0)→0 / 自導 partial(位移 .25 → 1/3)→ 逼出「非對稱」「>1」「退化框未防呆」 |
| `test_metamorphic_match_conservation` | match_boxes **分割守恆**:每個 gt/pred 恰落入一桶(tp∪confused∪fn==range(G)、tp∪confused∪fp==range(P))→ 逼出雙重計數 / 漏配 |

## GUI / 整合 E2E AC(Tier B done 要件)

| AC | 內容 | 測試 |
|----|------|------|
| AC-G1 | 佇列真實性:4 圖(漏標/多框/類錯/全對)→ FN/FP/混淆佇列各含對的那張,全對那張不在任何佇列 | `test_queues_are_real_e2e` |
| AC-G2 | 疊框渲染:佇列卡影像 ≠ 原圖 bytes(走到 `draw_two_sources`) | `test_two_source_render_e2e` |
| AC-G3 | 送標接回合:從 FN 佇列送標 → 回合卡策略名含 `pred_diff:fn` | `test_send_from_queue_opens_round_e2e` |
| AC-G4 | seed=pred 落盤:選「模型預測」送標 → `images/<sha>.json` shapes 數 == 該圖 pred 框數 | `test_seed_pred_lands_e2e` |

## done 判準

- **單元綠**:`python verify/gate.py gt_pred_diff` 印 `GREEN`。
- **AND 真實 E2E 綠**:`pytest tests/e2e/test_gt_pred_diff_e2e.py -m e2e`(AC-G1~G4)。
- AC9 是**資料完整性**條款,不是 UI 條款:它在單元層驗死(GUI 不得繞過 `send_to_labeling`)。
