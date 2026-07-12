# 14 readback_store — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/14_readback_store.md](../3_Architect_Design/14_readback_store.md) · 里程碑 M12 · Tier A(核心)+ B(接線)
單元測試:`tests/test_readback_store.py`(gate:`python verify/gate.py readback_store`)
E2E:`tests/e2e/test_readback_store_e2e.py`(`@pytest.mark.e2e`,**不進 gate / PG 自主修綠迴圈**,交 `/ux-test` 或人觸發)

## 單元 AC ↔ 測試

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | 記入→覆蓋層反映 new_label,回傳筆數 | `test_ac1_record_then_overlay` | 🔴 |
| AC2 | 最新勝 + 歷史保留(append-only 2 筆) | `test_ac2_latest_wins_history_kept` | 🔴 |
| AC3 | apply_overlay 覆蓋命中、未命中不動 | `test_ac3_apply_overlay` | 🔴 |
| AC4 | append-only 行數 + 稽核欄(ts/annotator/batch/round/old/new) | `test_ac4_append_only_audit_fields` | 🔴 |
| AC5 | 壞行容忍(非 JSON / 缺 sha256) | `test_ac5_corrupt_lines_tolerated` | 🔴 |
| AC6 | 檔不存在:空覆蓋 / apply 0 / summary 0 | `test_ac6_missing_file` | 🔴 |
| AC7 | 缺 new_label 或缺 sha256 → 不寫 | `test_ac7_incomplete_skipped` | 🔴 |
| AC8 | 現 label 已等覆蓋值 → apply 不計入(不假覆蓋) | `test_ac8_noop_not_counted` | 🔴 |
| AC9 | export_version 產 CSV(表頭 + 資料列) | `test_ac9_export_version` | 🔴 |
| AC10 | 無重依賴:乾淨子行程 import 不拉 numpy | `test_ac10_no_heavy_dependency` | 🔴 |
| AC11 | load_overlay(shas) 只回命中的 | `test_ac11_filter_shas` | 🔴 |

## GUI / 整合 E2E AC(Tier B done 要件;`/ux-test` 跑,不在 PG 修綠迴圈)

| AC | 內容 | 測試 |
|----|------|------|
| AC-G1 | cart 送標→寫 sidecar(classB)→套用讀回→**重啟 server**→重 Run→修正稽核仍在(overlay 持久) | `test_readback_survives_restart_e2e` | ✅ |
| AC-G2 | 上流程後原始資料夾檔案 byte/大小不變;`.lv_cache/readback_changelog.jsonl` 含該筆 | `test_readback_no_dataset_write_e2e` | ✅ |
| AC-G3 | 📥「🩹 讀回修正」顯示已修正 ≥1 圖;匯出修正清單 CSV(表頭+該筆) | `test_readback_audit_export_e2e` | ✅ |

> **E2E 狀態(2026-07-12)**:三條 AC-G 全綠;與 T2 的 7 條併跑(pool+round+readback)**全套 10/10 綠**。
> 教訓:overlay 依設計跨 Run 持久 → `al_isolated_server` per-test 清理需**連 `readback_changelog.jsonl`
> 一起刪**,否則前一測的修正會在本測 Run 時預先套用、令本測讀回變 no-op(這反而坐實「overlay 真的在
> Run 時套用生效」)。apply_overlay 的 latest-wins / no-op-not-counted 由單元 AC3/AC8 釘死;送標/讀回/
> 匯出點擊皆驗證後果並重試(沿用 T2 harness)。

## done 判準(Tier A 核心 + Tier B 接線)

- **單元綠**:`python verify/gate.py readback_store` 印 `GREEN`(收集>0、退出碼0、無 fail/error、3_/4_ 契約未竄改)。
- **AND 真實 E2E 綠**(接線完成後):`tests/e2e/test_readback_store_e2e.py -m e2e`(AC-G1~G3)。
- PG 先把單元(純邏輯核心)修綠;GUI 接線(record 於讀回、apply 於 Run、📥 稽核區)+ E2E 交 `/ux-test`。
