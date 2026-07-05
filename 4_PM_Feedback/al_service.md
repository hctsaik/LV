# 10 al_service — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/10_al_service.md](../3_Architect_Design/10_al_service.md) · 里程碑 **M10** · Tier B
測試:`tests/test_al_service.py`(gate:`python verify/gate.py al_service`)
依賴:08 al_batch + 09 al_workspace。E2E:**不適用**(CLI/編排;GUI 消費另以 M10-GUI E2E 驗)。

## AC ↔ 測試(合成資料 + 注入 graded extractor + 凍結 bank;now 注入固定 run_id;不觸真 DINOv2)

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | 首輪:queue 有 min(k,objects) 筆、高紅排最前、欄位齊全、runs/<id> 存在 | `test_ac1_first_round` | 🔴 |
| AC2 | 增量:第二輪只 embed 新圖(舊圖 .lv_cache)→ extractor 增量 == 新圖數 | `test_ac2_incremental_only_embeds_new` | 🔴 |
| AC3 | 閉環:標註前 2 筆 → 下一輪那 2 id 不在 queue | `test_ac3_labels_remove_from_queue` | 🔴 |
| AC4 | 換版:內容不同相容 model → model_version 變、同物件 id 不變(原地) | `test_ac4_model_version_rescore_inplace` | 🔴 |
| AC5 | 壞輸入:無 profile→error 無殘留;model 缺→error 鎖已釋放 | `test_ac5_bad_input` | 🔴 |
| AC6 | 不寫來源(C6)快照零變化 | `test_ac6_never_writes_source` | 🔴 |
| AC7 | 單實例:新鮮他人鎖→skipped 不動 queue;移除後→ok | `test_ac7_single_instance_lock` | 🔴 |
| AC8 | CLI:init 建 profile.yaml、run 無 profile→非0 退出碼 | `test_ac8_cli_exit_codes` | 🔴 |

> AC8 的 CLI `run` 完整行為(需真模型)不在單元驗;由 AC1–AC7 的注入路徑涵蓋。CLI 僅驗 init + 壞路徑退出碼。

## done 判準(Tier B,無 GUI)

- `python verify/gate.py al_service` 印 `GREEN`(收集>0、退出碼 0、無 fail/error、3_/4_ 契約未竄改)。
- 真實行為由對真實檔案系統(工作區、原子寫、增量、鎖)+ 對 08/09 真實整合(al_batch 分批評分、al_workspace 持久化)的斷言涵蓋;無瀏覽器 E2E。
- M10 完整完成另需 **M10-GUI 接線**(④ 監看設定/匯出/狀態/立即掃描 + Step③ 佇列消費)通過真實 Playwright E2E。

## 增補(M12b / A3):similar profile
| AC | 說明 | 測試 |
|----|------|------|
| AC-SVC-SIM | init 存 reference.npy + profile.reference_vector_file;run_once similar → 佇列以參考同群為主 | `test_ac_svc_sim_similar_profile_and_ranking` |

## 增補(M13 Task6):retrieve 監看
| AC | 說明 | 測試 |
|----|------|------|
| AC-SVC-RET | init 存 sample_bank_dir;run_once retrieve → 佇列帶 suggested_class、依相似度、X樣本→X群為主 | `test_ac_svc_ret_retrieve_profile_and_queue` |
