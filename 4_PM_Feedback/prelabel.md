# AC ↔ 測試對應表:11 prelabel

設計:[3_Architect_Design/11_prelabel.md](../3_Architect_Design/11_prelabel.md)、
[3_Architect_Design/M11_gui_wiring.md](../3_Architect_Design/M11_gui_wiring.md)。
單元:`tests/test_prelabel.py`;真實 E2E:`tests/e2e/test_prelabel_gui_e2e.py`(`@pytest.mark.e2e`)。

## 單元(模組 11,gate 收)

| AC | 說明 | 測試 |
|----|------|------|
| AC1 | 異常低於門檻 → 「正常」分流、accept=False | `test_ac1_normal_gated_out` |
| AC2 | 異常高但沒把握(min_conf 不可達)→ Unknown、留白 | `test_ac2_unknown_left_blank` |
| AC3 | target_classes 過濾(只接受選定類別) | `test_ac3_target_class_filter` |
| AC4 | head=None → ValueError(含「分類頭」)、不亂猜 | `test_ac4_head_none_raises` |
| AC5 | 長度不符 → ValueError | `test_ac5_length_mismatch_raises` |
| AC6 | 空輸入 → [] | `test_ac6_empty` |
| AC7 | to_yolo_lines 只含 accept + 6 位小數 + cls_id | `test_ac7_to_yolo_only_accept_format` |
| AC8 | 座標往返誤差 < 1e-6 | `test_ac8_roundtrip` |
| AC9 | pred 不在 class_names → ValueError(含該名) | `test_ac9_unknown_class_raises` |
| AC10 | 正常/Unknown 不輸出(雙保險) | `test_ac10_normal_unknown_not_output` |
| AC11 | 多圖分組 + cls_id 對索引 | `test_ac11_multi_image_group` |
| AC12 | export 寫檔 + 讀回 + 回傳計數 | `test_ac12_export_and_readback` |
| AC13 | C6:輸出位於來源內 → raise | `test_ac13_c6_refuse_inside_source` |
| AC14 | C6:輸出含 images/ → raise | `test_ac14_c6_refuse_images_dir` |
| AC15 | 重複匯出(已含 labels/)允許、覆寫 | `test_ac15_reexport_allowed` |
| AC16 | classes.txt 內容 + 空輸入不 crash | `test_ac16_classes_txt_and_empty` |
| AC17(衍生) | C6 真實不變量:完整流程後來源零寫入 | `test_ac17_source_untouched` |
| AC18(衍生) | 純函式冪等 | `test_ac18_prelabel_idempotent` |

雙向覆蓋:AC1–AC16 皆有對應測試;AC17/18 為 PM 自加的不變量/冪等(設計未明列),
逼「C6 真實不變量」與「純函式」而非只逐字轉抄。

## 真實 E2E(M11 GUI 接線)

GUI 接線的 AC-E1~E6 ↔ 測試對應**獨立成檔**(同名對齊慣例,對應設計 `M11_gui_wiring.md`):
見 [M11_gui_wiring.md](M11_gui_wiring.md)。E1–E5 已綠(2/2);E6(Should)未做,進候選。

## 給 PG 的提醒(反向閘門出口)

- 若 ③ 在某模型型態拿不到 `obj_emb`/`scores`(如純 patch 無物件 embedding)→ 停手回報 `/architect`
  (見 M11_gui_wiring.md §6),在 expander 明示「此模型型態不支援預標」,不硬跑。
- `assert_safe_prelabel_dir` 不可用既有 `assert_safe_bank_dir`(後者見 labels/ 誤擋重複匯出,AC15 會紅)。
- 測試基礎設施(conftest/fixtures)PG 唯讀;`yolo_defect_at_nmin` 已在 `tests/conftest.py`(session scope)。
