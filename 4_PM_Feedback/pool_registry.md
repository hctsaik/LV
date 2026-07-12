# 08 pool_registry — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/08_pool_registry.md](../3_Architect_Design/08_pool_registry.md) · 里程碑 M9 · Tier A(核心)+ B(接線)
單元測試:`tests/test_pool_registry.py`(gate:`python verify/gate.py pool_registry`)
E2E:`tests/e2e/test_pool_registry_e2e.py`(`@pytest.mark.e2e`,**不進 gate / PG 自主修綠迴圈**,交 `/ux-test` 或人觸發)

> 設計反向閘門紀錄:PM 轉抄時發現 `load_states` 契約自我矛盾(AC6 要 `=={}` vs 2.1 帶 `_corrupt_lines`),
> 已由 architect 修為「主回傳乾淨、診斷下沉 `corrupt_line_count()`」(見 ROADMAP 決策日誌 2026-07-04)。

## 單元 AC ↔ 測試

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | 往返:mark_sent 後 load_states 反映 state/batch_id | `test_ac1_round_trip` | 🔴 |
| AC2 | 優先序 labeled>excluded>sent(excluded 不降級 labeled) | `test_ac2_priority` | 🔴 |
| AC3 | reset 清回 unseen,之後事件重新累積 | `test_ac3_reset` | 🔴 |
| AC4 | partition_new:新樣本順序保留 + 已有狀態回報(乾淨形) | `test_ac4_partition_new_order_preserved` | 🔴 |
| AC5 | 壞行容忍 + `corrupt_line_count()==2`;load_states 乾淨 | `test_ac5_corrupt_lines_tolerated` | 🔴 |
| AC6 | 檔不存在:`load_states()=={}`;`partition_new([X])==([X],{})` | `test_ac6_missing_file` | 🔴 |
| AC7 | append-only:兩次 mark_sent → 檔案兩行 | `test_ac7_append_only` | 🔴 |
| AC8 | 無重依賴:乾淨子行程 import 不拉 numpy | `test_ac8_no_heavy_dependency` | 🔴 |

## 推導測試(設計未明列,PM 自加施壓)

| 測試 | 施壓的不變量 |
|------|-------------|
| `test_acD1_priority_monotonic_no_downgrade` | metamorphic:低優先事件晚到不得降級高優先狀態(reset 例外)→ 逼出「last-write-wins 折疊」bug(該實作會讓 AC2 第二段偶爾漏抓) |

## GUI / 整合 E2E AC(Tier B done 要件;`/ux-test` 跑,不在 PG 修綠迴圈)

| AC | 內容 | 測試 |
|----|------|------|
| AC-G1 | 同資料夾走送標流程兩次:第二次 UI 出現「已送標」字樣+數量,預設送出集合為空 | `test_second_send_blocked_e2e` | ✅ |
| AC-G2 | 勾「仍包含已送標」後可送出,數量==原批大小 | `test_include_handled_override_e2e` | ✅ |
| AC-G3 | 模擬 handoff 結果檔→套用讀回→registry 內該 sha state=="labeled"(讀檔斷言) | `test_readback_marks_labeled_e2e` | ✅ |

> **E2E 狀態(2026-07-12,M9 GUI 接線後)**:三條 AC-G 全綠、連兩次全套穩定綠。實作要點/教訓:
> (1) 驅動走 **cart → 匯出清單 → 送標**,不走散點框選——散點 `on_select` 是全頁 rerun,會讓
> `@st.fragment _render_right_panel` 的送標鈕留下**隱藏的 stale 複本**,點到它=無效點擊(假失敗)。
> cart 路徑只碰非 fragment 的送標鈕。(2) 送標/回讀/掛指標點擊都**驗證後果並重試**(送出確認文字/
> 已套用文字/帳本檔出現該事件),吸收 Streamlit rerun 時序的點擊遺失。(3) 用 `al_isolated_server`
> 隔離 `LV_CACHE_DIR`/`LV_OUTPUT_DIR`/`CIM_LOG_DIR` 至 temp;embedding 快取共用(首測冷算 ~90s、
> 其餘暖用),每測只刪 pool/round/handoff 檔重置狀態。harness 見 `tests/e2e/_al_loop_harness.py`。

## done 判準(Tier A 核心 + Tier B 接線)

- **單元綠**:`python verify/gate.py pool_registry` 印 `GREEN`(收集>0、退出碼0、無 fail/error、3_/4_ 契約未竄改)。
- **AND 真實 E2E 綠**(接線完成後):`tests/e2e/test_pool_registry_e2e.py -m e2e`(AC-G1~G3)。
- PG 先把單元(純邏輯核心)修綠;GUI 接線 + E2E 交 `/ux-test`。
