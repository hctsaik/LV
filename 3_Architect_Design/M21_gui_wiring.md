# M21 GUI 接線 — 做法比較(A/B)+ 成本試算(Tier B)

> PRD [2_PO_PRD/recipe_compare_cost_workpoint_prd.md](../2_PO_PRD/recipe_compare_cost_workpoint_prd.md)。
> 組裝 26 `paired_compare`/27 `cost_curve`/28 `run_pairing`,掛**瑕疵偵測 ② 套用**分頁
> (主畫面,套用結果區之後;不放側欄——run 後側欄收合,沿 M2 教訓)。
> app.py 只做讀取/組 id/呼叫引擎/顯示;**判定邏輯零複製**(單一真相源=引擎)。

## 1. 目的

讓使用者在瑕疵偵測畫面內完成「A/B 做法比較」與「成本 what-if 試算」,
全程唯讀生效設定。

## 2. 介面配置與 keys(契約)

②套用結果區之後,兩個 expander:

### ⚖️ 做法比較(A/B)— key 前綴 `rcmp_`

| 元件 | key | 行為 |
|---|---|---|
| 按鈕「📌 存目前結果為 A」 | `rcmp_pin_a` | 快照目前 `anomaly_apply_result`(見 §3) |
| 按鈕「📌 存目前結果為 B」 | `rcmp_pin_b` | 同上存 B 槽 |
| A/B 摘要行 | (caption) | 各槽:配置摘要 + 項目數;未 pin 顯示「(未存)」 |
| 群組欄位 | `rcmp_group_field` | selectbox:`不分組`/`按上層資料夾`/(🩺 已載 metadata CSV 時)其欄位名 |
| 按鈕「開始比較」 | `rcmp_go` | 兩槽齊+同資料夾才 enabled;跑 §3 流程 |
| 結論卡 | container `rcmp_card` | 主句白話+分組明講句+「詳細」expander(method line/p/效應/支撐) |

- 兩槽資料夾不同 → `st.warning`(含 `"不同資料夾"`)且比較鈕 disabled。
- 快照存 `st.session_state["rcmp_slots"]`(dict,keys `"A"`/`"B"`);**session 即可,
  不落地**(user 明說不要歷史)——PRD 開放問題 1 定案。

### 💰 成本試算(what-if)— key 前綴 `rcost_`

| 元件 | key | 行為 |
|---|---|---|
| 成本比輸入 | `rcost_ratio` | `st.number_input`「漏檢:誤報」1–1000,step 1,預設 50(不用 slider:BaseWeb slider E2E 陷阱 #4,且使用者要精確輸入) |
| 曲線 | container `rcost_curve_wrap` | plotly 折線:x=閾值、y=相對代價;最低點 marker + 目前生效閾值 vline |
| 建議句 | (markdown,含錨字 `"建議工作點"`) | 「以你的成本比,代價最低的設定在 t=…(會標記 X/N 張)」 |
| 門檻/教育文案 | `rcost_msg` | 樣本不足時的指引;恆顯 caption 含 `"僅試算，不會改變任何設定"` |

## 3. 資料流

1. **快照(pin)**:從 `anomaly_apply_result.records` 組
   `{f"{r['path']}#{r['obj_index']}": float(r["score"])}` 與
   `{item_id: 1 if r["verdict"]=="bad" else 0}`(判異常決策)+
   配置摘要字串(model.meta 模型/score_mode/分位數現值)+ 目標資料夾路徑。
2. **比較指標(PRD 開放問題 5 定案)**:**已確認樣本上的每項 0/1 錯誤**。
   整形一律走 28 `compare_inputs(records_a, records_b,
   anomaly_confirmed_apply, records_現行)`(錯誤定義在 28 鎖死,app.py 零複製)
   → `pair_runs` 對齊 → 26 `paired_compare(higher_is_better=False)`。
   `only_a/only_b/dropped` 非空時 caption 明講數量。
3. **群組(開放問題 3 定案)**:`按上層資料夾`=`Path(path).parent.name`;
   metadata 欄位=以影像 path 查 🩺 join 結果(session 既有物件,PG 依現況接線;
   查無值的項歸 `"(無資料)"` 群組)。選了群組 → 傳 `groups` 給 26,
   結論卡顯示`「已按〈欄位〉分組比較」`;`不分組` → 顯示
   `「未考慮批次，結論僅供探索」`。
4. **結論卡**:引擎 verdict → 白話主句(A/B 用配置摘要稱呼):
   `a_better`→`「A 較好：錯誤率低 {abs(mean_diff):.0%}（N={n}）」`、
   `b_better` 對稱、`inconclusive`→`「看不出來」+ reasons 白話列點`。
   統計細節只進「詳細」expander(G5)。**GUI 不自設門檻**——一律呼叫引擎、
   顯示 reasons(單一真相源);唯 `n_common==0` 短路顯示
   `「請先在下方畫廊確認一些瑕疵/正常樣本」`。
5. **成本試算(開放問題 4 定案)**:scores=目前 `anomaly_apply_result` 全項;
   labels=confirmed_apply(同映射)。**門檻:瑕疵 ≥5 且 正常 ≥5**,不足顯示
   `「已確認樣本不足（瑕疵 X/5、正常 Y/5）」`(數字即時)。足 → 呼叫 27
   (`cost_miss=ratio`, `cost_fa=1`, `current_threshold=anomaly_apply_result["threshold"]`)
   → 曲線+建議句。改 `rcost_ratio` → 直接重算(fragment 或自然 rerun,PG 依現況)。
6. **G4 鐵則**:本區塊所有程式路徑**只讀** `anomaly_apply_result`/
   `anomaly_confirmed_apply`/model.meta,**絕不寫**它們與任何生效設定 key
   (分位數/模型槽/門檻);絕不寫使用者資料夾([[no-dataset-writes]])。

## 4. 邊界條件與錯誤處理

- 尚無 `anomaly_apply_result` → 兩 expander 顯示指引(`"請先在②執行套用"`),無按鈕功能。
- pin 後又重跑套用(不同配置)→ 槽保留舊快照(摘要行可辨識);同資料夾檢查只看路徑。
- confirmed 被清空後按比較 → `n_common==0` 指引文案,不 crash。
- 引擎 `ValueError` → `st.error` 顯示訊息本文(不吞錯、不 traceback 裸露)。

## 5. Acceptance Criteria(E2E,`tests/e2e/test_m21_workpoint_e2e.py`,@pytest.mark.e2e)

> 2026-07-19 反向閘門修訂:②散點是 Scattergl(WebGL,無 per-point DOM、叢集位置
> 資料相依)→ E2E 無法可靠框選植入瑕疵標 `bad`;且②沒有「分位鬆緊」控制(判定門檻
> 來自 classify 校準)。故 **G1 方向性由 28-AC7 純函式鏈鎖死**,E2E 驗:接線真實、
> null 方向(deterministic)、成本曲線反應性、零副作用、講人話。

真實 Streamlit+Playwright+真 DINOv2;合成 objects 資料集(復用既有 anomaly E2E fixture
手法);①建模→②套用→「✅ 自動把最不可疑的 12 個標為正常範例」(autoseed,一鍵、
決定性)→ 🔁 就地重評(既有 `test_confirm_normal_reapply` 先例)。

- **AC-E1(接線+G5 主流程)**:②套用後 pin B(未校準)→ autoseed 12 正常 →
  就地重評 → pin A(校準後)→ `rcmp_go` → `rcmp_card` 出現三值結論之一
  (`"A 較好"`/`"B 較好"`/`"看不出來"` 至少一者)且含 `"未考慮批次"`(預設不分組);
  「詳細」內文含 `"exploratory"` 與 `"p="`。
- **AC-E2(G2 null 誠實+方向 deterministic)**:同一結果 pin A 再 pin B(不改任何
  東西)→ 比較 → 卡含 `"看不出來"` 且含 `"無差異"`(d 全 0,方向鎖死)。
- **AC-E3(G3 曲線反應性)**:先造出兩類確認樣本(全走真 UI,標記是真的人工標記
  語義):modebar Box Select 對②散點**整圖拖框**(WebGL 不需 per-point DOM)→
  「🔴 框選標為瑕疵範例」(全部標 bad)→ autoseed(最不可疑 12 個**覆寫**為 good)
  → 瑕疵 ≥5 且正常 12 ≥5 成立。成本試算展開,`rcost_ratio` 預設 50 →
  `rcost_curve_wrap` 內出現 plotly SVG 且頁面含 `"建議工作點"` 句與
  `"僅試算"` caption;把 `rcost_ratio` fill 為 1 → 曲線 y 軸刻度文字(SVG)
  **改變**(成本尺度 ~50×變動必反映;argmin 移動的 exact 驗證在 27-AC2,
  不在 GUI 重驗)。
- **AC-E4(G4 零副作用)**:AC-E1~E3 全程前後:②狀態列(n_bad/threshold 顯示文字)
  完全相等;目標資料夾檔案清單+每檔 bytes 完全不變(hash 前後 ==)。
- **AC-E5(門檻誠實)**:清除 confirmed(既有清除鈕)→ 比較顯示指引
  (`"請先"`,n_common==0 短路)、成本試算顯示 `"已確認樣本不足"`,
  無 exception(`_no_exception`)。
- **AC-E6(可達性)**:以上全部在「瑕疵偵測」工具 ② 分頁內完成,無需切換工具。

E2E 陷阱備忘適用:成功訊號用**唯一錨字**(`"建議工作點"`/`"A 較好"` 只在結果卡出現,
說明文字不得含);tabs 點按鈕 rerun 重置 tab 的 flaky 對策沿 M7;expander 用 JS 展開。
