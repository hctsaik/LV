## 1. T0 Contracts

- [x] 1.1 Publish OpenSpec proposal/design/spec/tasks and U-Net 3_/4_ contracts from r3; record live wiring in inventory.md. Do not change the application in T0.

## 2. T1–T2 Identity and scores

- [x] 2.1 Add failing tests for AC01–04, AC06, AC08, AC29, AC30, AC32 (independent oracle, not a copy of production).
- [x] 2.2 Implement `class_similarity` blocked exact Top-K and `class_similarity_records` adapter. Do not edit `app.py` yet.

## 3. T3 Export

- [x] 3.1 Add export tests for AC09–11 and AC31 data layer.
- [x] 3.2 Implement `class_similarity_export` schema 2; subset reports still carry full `score_index` and `evaluation_items`.

## 4. T4–T6 Panel, selection, E2E

- [x] 4.1 UI contract tests for AC05/07/09/12/26/33; implement read-only panel dispatched from `_render_label_quality`.
- [x] 4.2 Wire selection/cart/download (AC08/10/26/30). Never call gray-band. E2E：`test_ac10_cart_add_skip`。
- [x] 4.3 Real image and object Streamlit E2E (AC13/14); source folder listing and content hashes unchanged.

## 5. T7–T9 Threshold and evaluation

- [x] 5.1 Threshold predicate, slider/canonical precision, draft/commit, workload curve (AC15–18/25/26/33). E2E：`test_ac16_canonical_threshold_precision`、`test_ac18_pointer_held_preview`、`test_ac33_axis_zoom`。
- [x] 5.2 Evaluation template/import/digest/PR (AC19/20/22/27/28/34). 純函式綠；UI 匯入以單元為準。
- [x] 5.3 PR curve, confusion drill-down, operating point, heldout isolation, schema-2 round-trip (AC21/23–25/27/31/34).

## 6. T10 Close-out

- [x] 6.1 Pointer-held browser evidence, race tests, regressions, usage guide, AC01–34 named mapping, measured N/D/C/K/block/memory/latency.

進度細節見 `docs/class-similarity-review-progress.md`。
