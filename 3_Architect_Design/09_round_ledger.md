# 09 — round_ledger 技術設計(回合帳本 + 回合時間軸)

對應 PRD:[2_PO_PRD/al_loop_infrastructure_prd.md](../2_PO_PRD/al_loop_infrastructure_prd.md) · 里程碑 M9 · Tier A(核心)+ B(接線)

## 1. 目的(Purpose)

給主動學習迴圈一個持久的「記憶」:每一輪選樣策略、送標批次、回讀結果、(可掛的)模型指標,
以 append-only 帳本記錄,並在 📥 標註回饋分頁長出「回合時間軸」視圖。它是停止判準(M11/12)
與學習曲線的資料底座。

## 2. I/O 契約

### 2.1 新模組 `scripts/round_ledger.py`(純 stdlib;不得 import numpy/streamlit)

儲存:`output/rounds.jsonl`(與 `usage_log.jsonl`、`curation_log.jsonl` 同目錄慣例),
一行一事件,round 狀態由事件折疊而成(與 08 同哲學,append-only、不可變歷史)。

```python
def ledger_path() -> Path            # <repo 根>/output/rounds.jsonl(允許 LV_OUTPUT_DIR 覆寫,無則預設)

def start_round(strategy: str, params: dict, tool: str, note: str = "") -> str
    # 產生 round_id = "r" + UTC 時間戳 "%Y%m%d%H%M%S" + "-" + uuid4().hex[:6]
    # (uuid 尾提供足夠熵:同秒內大量連呼也不撞 → AC3 連呼 100 次全異)。
    # app 執行於一般 Python,uuid/time 皆可用(此非 workflow-script 沙盒)。
    # append {"kind":"start", "round_id", "ts", "strategy", "params", "tool", "note"};回 round_id。

def attach_batch(round_id: str, batch_id: str, n_sent: int) -> None
    # {"kind":"batch", ...}。一回合可多批。

def record_readback(round_id: str, summary: dict) -> None
    # summary 至少含 {"batch_id", "n_labeled": int, "n_changed": int,
    #                 "changed_by_class": {label: int}};多鍵容忍。{"kind":"readback", ...}

def attach_metric(round_id: str, name: str, value: float, source: str, note: str = "") -> None
    # {"kind":"metric", ...};source ∈ {"probe","external","manual"}(自由字串,不驗enum)。

def load_rounds() -> list[dict]
    # 折疊成**乾淨的 round 物件 list**(依 start ts 排序;空/檔不存在 → []):
    # {"round_id","ts","strategy","params","tool","note",
    #  "batches":[{"batch_id","n_sent"}...], "n_sent": Σ,
    #  "readbacks":[summary...], "n_labeled": Σ, "n_changed": Σ,
    #  "metrics":[{"name","value","source","ts"}...]}
    # 壞行跳過(不進 list);掛在不存在 round_id 上的事件不丟失、不崩潰,由 orphan_events() 另取。

def orphan_events() -> list[dict]
    # 掛在不存在 round_id 上的事件(batch/readback/metric);正常情況為 []。供 UI 健康度顯示。

def learning_curve(rounds: list[dict], metric_name: str) -> tuple[list[int], list[float]]
    # x = 依回合序的「累積 n_labeled」;y = 該回合此 metric 的**最後一筆** value;
    # 無該 metric 的回合跳過(x/y 均不含)。空輸入回 ([], [])。
```

### 2.2 GUI 接線(Tier B,`scripts/app.py` 📥 標註回饋分頁擴充)

- **開回合**:`_send_to_labeling_ui` 加 checkbox `key=f"{key}_new_round"`「記為新回合」(預設 True)。
  送出成功後:True → `start_round(strategy=呼叫端傳入的策略名, params, tool)` + `attach_batch`;
  False → 掛到「最近一個 round」(session 記住 `_al_active_round`;無則自動 start)。
  呼叫端(viz 選樣/灰帶/瑕疵③…)把自己的策略名與參數傳進 `_send_to_labeling_ui`(新參數
  `strategy: str = ""`, `strategy_params: dict | None = None`;空字串=沿用工具名)。
- **回讀掛帳**:📥 套用讀回成功後,由 `apply_readback` 回傳的 changes 匯總
  `{"n_labeled": 批內有結果檔數, "n_changed": len(changes), "changed_by_class": Counter(新label)}`
  → `record_readback(該批次所屬 round_id, summary)`。批次→round 對映:`load_rounds()` 反查
  batch_id(帳本是唯一真相,不靠 session)。查不到 → 顯示 warning「此批次不屬於任何回合」並允許
  一鍵補掛新回合(不靜默丟)。
- **回合時間軸視圖**:📥 分頁新增子區「🔁 回合」:每回合一張卡(round_id 短碼、策略、
  `送出 n_sent · 回讀 n_labeled · 變更 n_changed`、metrics badges);卡內
  「➕ 掛指標」表單(name / value / source=manual)呼叫 `attach_metric`。
  有 ≥2 個帶同名 metric 的回合時,畫 `learning_curve` 折線(plotly,x=累積標註量)。

## 3. 資料流

選樣 → 送標(strategy 隨手帶入)→ start_round + attach_batch → Labeling 標註 →
📥 套用讀回 → record_readback(+08 的 mark_labeled)→ 時間軸多一張卡 →
(選配)外部重訓後手動掛 metric → 學習曲線多一點。

## 4. 邊界條件與錯誤處理

| 情況 | 預期行為 |
|---|---|
| ledger 檔不存在 | `load_rounds()==[]`;首次 start 自建 |
| 壞行 / 孤兒事件 | 跳過壞行;孤兒進 `_orphans`,UI 顯示計數 |
| 一回合多批次 | batches 累加,n_sent=Σ |
| 同回合多次 readback(分批標完) | readbacks 累加,n_labeled/n_changed=Σ |
| 同名 metric 多筆 | 曲線取該回合最後一筆;卡上全列 |
| round_id 不存在(record/attach) | 照寫事件(append-only),折疊時進 _orphans(見上) |
| 使用者刪 output/rounds.jsonl | 一切歸零但 app 不崩潰(等同全新) |

## 5. Acceptance Criteria

**單元(`tests/test_round_ledger.py`;tmp_path + monkeypatch 輸出路徑)**

- **AC1(生命週期)**:start→attach_batch(n_sent=10)→record_readback(n_labeled=8,n_changed=3)→
  attach_metric("probe_acc",0.9,"probe") 後,`load_rounds()` 恰 1 筆且四類欄位值全對(釘死數值)。
- **AC2(多批多讀回累加)**:兩次 attach_batch(3,4)+兩次 readback(n_labeled=3/4) → n_sent==7、n_labeled==7。
- **AC3(round_id 格式與唯一)**:連呼 start_round 100 次,id 全異且都以 "r" 開頭。
- **AC4(孤兒)**:對不存在 round 記 readback → `orphan_events()` 長度==1,且 `load_rounds()` 回乾淨 list(其餘 round 不受影響、不含孤兒)。
- **AC5(learning_curve)**:三回合 n_labeled=10/20/30、metric "acc"=0.5/缺/0.8 →
  `learning_curve(rounds,"acc") == ([10, 60], [0.5, 0.8])`(x 為累積;缺 metric 回合跳過)。
- **AC6(壞行容忍)**:塞壞行後 load_rounds 正常。
- **AC7(無重依賴)**:同 08-AC8,乾淨子行程 import 不拉 numpy。

**GUI / 整合(真實 E2E,`tests/e2e/test_round_ledger_e2e.py`)**

- **AC-G1(送標即開回合)**:走一次可驅動的送標流程(策略名隨手帶)→ 📥 分頁「🔁 回合」
  出現一張卡,卡上可見策略名與 `送出 N`。
- **AC-G2(回讀入帳)**:對該批模擬結果檔 → 套用讀回 → 同一張卡出現 `回讀 M · 變更 K`(數字與 fixture 一致)。
- **AC-G3(重啟存活)**:重啟 Streamlit server(harness 重起 session)後時間軸卡仍在(讀自 rounds.jsonl)。
- **AC-G4(手動掛指標+曲線)**:對兩個回合各掛 "acc" → 折線圖渲染且點數==2。

## 6. 給 /pm 的提示

- 單元全部不需影像;E2E 重用 `test_lv_to_labeling` 的 handoff/result fixture 手法製造「已標」批次。
- AC5 的曲線數學是本模組唯一「算」的地方,務必釘死數值。
- 與 08 的整合(mark_labeled)在 08 的 AC-G3 驗,這裡不重複。
