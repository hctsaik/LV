# 27 `cost_curve` — 成本工作點 what-if 引擎(Tier A)

> M21。PRD [2_PO_PRD/recipe_compare_cost_workpoint_prd.md](../2_PO_PRD/recipe_compare_cost_workpoint_prd.md)。
> 純函式、無 I/O、無 streamlit。實作 `scripts/cost_curve.py`。**只算不套**:引擎輸出
> 建議點,絕不寫任何設定(寫回是 GUI 明確禁止事項,見 M21_gui_wiring G4)。

## 1. 目的

異常分數+人工確認標籤+相對成本比 → 閾值掃描的 FN/FP/工作量/預期相對代價曲線
與最低代價點。

## 2. I/O 契約

```python
def cost_curve(
    scores: Sequence[float],       # 每項異常分數(越高越異常)
    labels: Sequence[int],         # 1=真瑕疵, 0=正常(人工確認)
    *,
    cost_miss: float,              # 漏檢一項的相對代價(>0)
    cost_fa: float,                # 誤報一項的相對代價(>0)
    thresholds: Sequence[float] | None = None,  # None → 自動導出(見資料流)
    current_threshold: float | None = None,
) -> dict
```

回傳(鍵固定;所有陣列同長、依 threshold 升冪):

```python
{
  "thresholds": list[float],
  "fn": list[int], "fp": list[int], "flagged": list[int],  # flagged = 判異常數
  "cost": list[float],            # cost_miss*fn + cost_fa*fp
  "best_idx": int,                # cost 最小;平手取最小 index(=最鬆之最優)
  "n": int, "n_pos": int, "n_neg": int,
  "current": {"threshold": float, "fn": int, "fp": int,
              "flagged": int, "cost": float} | None,   # current_threshold=None → None
  "method": str,                  # method line(判定規則+成本比,含「僅試算不套用」)
}
```

## 3. 資料流

1. **判定規則(釘死)**:`score >= t` → 判異常(flag)。
   `FN = #(label==1 且 score < t)`;`FP = #(label==0 且 score >= t)`。
2. **自動閾值集**(`thresholds=None`):對唯一分數升冪 `u_1..u_K`,
   取 `[u_1 - 1] + [(u_i + u_{i+1})/2 for i in 1..K-1] + [u_K + 1]`
   → 恰 `K+1` 個切點,涵蓋「全標」到「全不標」。
3. 每切點算 fn/fp/flagged/cost;`best_idx = cost.index(min(cost))`(升冪下первый=最鬆)。
4. `current` 用同一判定規則在 `current_threshold` 直接算(不吸附到網格)。

## 4. 邊界條件與錯誤處理

- 空輸入、`len(scores)!=len(labels)`、labels 含 {0,1} 以外、scores 含非有限值
  → `ValueError`。
- `n_pos==0` 或 `n_neg==0` → `ValueError`(訊息含 `"需同時有已確認的瑕疵與正常"`;
  GUI 據此顯示門檻文案,引擎不猜)。
- `cost_miss<=0` 或 `cost_fa<=0` → `ValueError`。
- 顯式 `thresholds` 給了:去重排序後使用;空列表 → `ValueError`。
- 所有分數相同(K=1):thresholds=`[u-1, u+1]`(全標/全不標兩點),照常回傳。

## 5. Acceptance Criteria

固定小例:`scores=[0.1,0.2,0.3,0.4,0.5,0.9]`、`labels=[0,0,0,1,0,1]`
(P=2:0.4/0.9;N=4)。自動閾值=`[-0.9,0.15,0.25,0.35,0.45,0.7,1.9]`。

- **AC1(G3 exact 全陣列)**:`cost_miss=5, cost_fa=1` →
  `thresholds==pytest.approx([-0.9,0.15,0.25,0.35,0.45,0.7,1.9])`、
  `fn==[0,0,0,0,1,1,2]`、`fp==[4,3,2,1,1,0,0]`、`flagged==[6,5,4,3,2,1,0]`、
  `cost==pytest.approx([4,3,2,1,6,5,10])`、`best_idx==3`、`n_pos==2`、`n_neg==4`。
- **AC2(G3 改成本比→工作點移動)**:同資料 `cost_miss=1, cost_fa=10` →
  `cost==pytest.approx([40,30,20,10,11,1,2])` 且 `best_idx==5`(t=0.7)
  ——成本比改變,最低點從 0.35 移到 0.7。
- **AC3(current 不吸附)**:AC1 設定 + `current_threshold=0.5` →
  `current=={"threshold":0.5,"fn":1,"fp":1,"flagged":2,"cost":6.0}`
  (0.5>=0.5 → flag,含 0.5 本身)。
- **AC4(平手取最鬆)**:`scores=[0.1,0.9]`、`labels=[0,1]`、`cost_miss=1,cost_fa=1`
  → thresholds=`[-0.9,0.5,1.9]`、`cost==pytest.approx([1,0,1])`、`best_idx==1`;
  若 `scores=[0.5,0.5]`、`labels=[0,1]` → thresholds=`[-0.5,1.5]`、
  `cost==pytest.approx([1,1])`、`best_idx==0`(平手取最小 index)。
- **AC5(錯誤路徑)**:全 1 標籤 / 全 0 標籤 → `ValueError` 且訊息含
  `"需同時有已確認的瑕疵與正常"`;`cost_miss=0`、labels 含 2、長度不符、空輸入
  各自 `raises ValueError`。
- **AC6(method line)**:成功呼叫的 `method` 同時含 `"score >= t"`(或等義釘死描述)、
  `"漏檢:誤報"`、`"僅試算"`。
