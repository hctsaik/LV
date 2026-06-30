# 07 whole_image_source — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/07_whole_image_source.md](../3_Architect_Design/07_whole_image_source.md) · 里程碑 M8 · Tier B
單元測試:`tests/test_whole_image_source.py`(gate:`python verify/gate.py whole_image_source`,自動對應 `tests/test_{module}.py`)
E2E:`tests/e2e/test_whole_image_source_e2e.py`(`@pytest.mark.e2e`,**不進 gate / PG 自主修綠迴圈**,交 `/ux-test` 或人觸發)

## 單元 AC ↔ 測試

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | 記錄內容釘死(`image_path/label/class_id/bbox/obj_index/score`) | `test_record_content_exact` | 🔴 |
| AC2 | 欄位集合 == YOLO_KEYS(同形) | `test_keys_are_yolo_shape` | 🔴 |
| AC3 | N 個輸入 → N 筆、順序保留 | `test_count_and_order_preserved` | 🔴 |
| AC4 | `label` 參數(預設 `image`) | `test_label_param` | 🔴 |
| AC5 | 空輸入 → `[]` | `test_empty_input` | 🔴 |
| AC6 | `run_pipeline` 路由 `whole_image`:無 labels 仍 N 筆;`yolo` 對照=`[]` | `test_run_pipeline_whole_image_no_labels` | 🔴 |
| AC7 | 預設 `object_source="yolo"` / 回歸(逐筆相等) | `test_run_pipeline_default_is_yolo_regression` | 🔴 |
| AC8 | 非法 `object_source` → `ValueError` | `test_invalid_object_source_raises` | 🔴 |
| AC10 | 全幅 bbox `crop_bbox(...,0.5,0.5,1,1)` == 整張影像 | `test_full_frame_bbox_is_whole_image` | 🟢¹ |

¹ AC10 驗證的是 PG **不需改、但功能仰賴**的既有 `crop_bbox` clamp 行為(全幅=整張圖);初跑即綠是「下游假設成立」的守門,非假綠。

> 編號由 8 直接跳至 10(無第九條,跳號非遺漏);本表與設計的 AC 集合一致 `{1,2,3,4,5,6,7,8,10}`。

## 推導測試(設計未明列,PM 自加施壓)

| 測試 | 施壓的不變量 |
|------|-------------|
| `test_derived_keys_match_real_yolo_record` | metamorphic:whole_image 記錄欄位 == `discover_yolo_objects` **真實**記錄欄位 → 鎖「下游不漂移」契約 |
| `test_derived_all_records_full_frame_invariant` | property:每筆必 `bbox=(0.5,0.5,1,1)`、`obj_index=0`、`class_id=0`、`score=1.0` |

## GUI / 整合 E2E AC（Tier B done 要件;`/ux-test` 跑)

| AC | 內容 | 測試(待 PG 接好 GUI 後由 /ux-test 跑) |
|----|------|------|
| AC-G1 | 物件來源「整張影像」+ 無 `labels/` 資料夾 → 建模不報「找不到 YOLO 物件」;②套用散點渲染出點、點數==圖數 | `test_whole_image_no_labels_runs_e2e` |
| AC-G2 | 整張影像模式畫面**可見「整張影像」字樣**(真實文字) | `test_mode_label_visible_e2e` |
| AC-G3 | 物件來源「YOLO 物件」+ 有 labels → 行為回歸(物件數==框數、可套用) | `test_yolo_mode_regression_e2e` |
| AC-G4 | whole_image 建模後 `model.meta["object_source"]=="whole_image"`、②鎖定沿用 | `test_object_source_locked_in_meta_e2e` |
| AC-G5(Should) | YOLO 模式 0 物件 → 顯示「改用整張影像」一鍵切換 | `test_fallback_hint_e2e`(Should;appetite 緊可進候選) |

## done 判準(Tier B)

- **單元綠**:`python verify/gate.py whole_image_source` 印 `GREEN`(收集>0、退出碼0、無 fail/error、3_/4_ 契約未竄改)。
- **AND 真實 E2E 綠**:`tests/e2e/test_whole_image_source_e2e.py -m e2e` 全綠(AC-G1~G4;G5 視 appetite)。
- 兩者皆綠才算 M8 完成;PG 先把單元修綠,E2E 交 `/ux-test`。
