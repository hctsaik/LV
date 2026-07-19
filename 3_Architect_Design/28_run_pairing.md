# 28 `run_pairing` — 偵測結果對齊與比較輸入整形(Tier A)

> M21。PRD [2_PO_PRD/recipe_compare_cost_workpoint_prd.md](../2_PO_PRD/recipe_compare_cost_workpoint_prd.md)。
> 純函式、無 I/O、無 streamlit;只吃 dict/list。實作 `scripts/run_pairing.py`。
> (2026-07-19 反向閘門 /pm→/architect 修訂:②散點是 Scattergl/WebGL,E2E 無法可靠
> 框選植入瑕疵 → G1 方向性驗證下沉到本模組的純函式鏈,GUI E2E 驗接線與 null 方向。
> 職責仍一句話:把兩份偵測結果與人工確認**整形成引擎可比的配對輸入**。)

## 1. 目的

兩份 `{item_id: 數值}` → 依 item 身分對齊成同序配對陣列,集合差與非有限值**明講**
(丟棄要可見,不靜默);並把 `records`+`confirmed` 整形成 per-item 0/1 錯誤 dict。

## 2. I/O 契約

```python
def pair_runs(run_a: Mapping[str, float], run_b: Mapping[str, float]) -> dict

def compare_inputs(
    records_a: list[dict], records_b: list[dict],
    confirmed_ref: Mapping[int, str],    # {index→"good"|"bad"},index 對 records_ref
    records_ref: list[dict],             # confirmed 索引空間的參考 records(=比較當下的套用結果)
) -> dict
# → {"run_a_err": {item_id: 0|1}, "run_b_err": {item_id: 0|1}, "n_labeled": int}
#   僅含「有人工確認」且該 run 有此 item 的項;
#   error = int(decision != label),decision = int(record["verdict"]=="bad"),
#   label = {"bad":1, "good":0}[confirmed 值];其他 confirmed 值 → ValueError。
#   item_id = f"{record['path']}#{record['obj_index']}"(全 M21 統一)。
```

回傳(鍵固定;`ids`/`a`/`b` 同長同序):

```python
{
  "ids": list[str],        # 兩邊都有且兩邊值皆有限的 item,字典序排序(決定性)
  "a": list[float], "b": list[float],
  "only_a": list[str],     # 僅 A 有(字典序)
  "only_b": list[str],     # 僅 B 有(字典序)
  "dropped": list[str],    # 兩邊都有但任一值非有限(NaN/Inf)→ 排除並列名(字典序)
  "n_common": int,         # == len(ids)
}
```

**item_id 約定(全 M21 統一,GUI 依此組 id)**:`f"{path}#{obj_index}"`——
`path`/`obj_index` 取自 `anomaly_apply_result.records`;物件框來自標註檔,
同資料夾跨配置穩定,故此 id 可跨 run 對齊。

## 3. 資料流

`pair_runs`:交集 → 濾非有限 → 字典序排序 → 抽值。無其他轉換(數值語義由呼叫端決定)。
`compare_inputs`:confirmed index→(經 records_ref)item_id→label;各 run records→
item_id→decision;僅對「有 label 且該 run 有 decision」的 item 產出 error。

## 4. 邊界條件與錯誤處理

- 任一輸入為空 dict:照常回傳(`n_common==0`,不 raise;「可不可比」是 GUI 決策)。
- key 非 str 或值不可轉 float → `ValueError`。
- 兩輸入完全相同物件(同 dict)→ 照常(全交集)。
- `compare_inputs`:confirmed index 超出 records_ref 範圍 → 該筆**跳過**(套用結果
  已換過、殘留索引,不 crash 不猜);confirmed 值非 "good"/"bad" → `ValueError`;
  records 缺 `path`/`obj_index`/`verdict` 鍵 → `ValueError`;
  confirmed 為空 → `n_labeled==0`、兩 err dict 皆空。

## 5. Acceptance Criteria

- **AC1(exact 對齊)**:`pair_runs({"x":1,"y":2,"z":3}, {"y":5,"z":6,"w":7})` →
  `ids==["y","z"]`、`a==[2.0,3.0]`、`b==[5.0,6.0]`、`only_a==["x"]`、
  `only_b==["w"]`、`dropped==[]`、`n_common==2`。
- **AC2(非有限明講)**:`pair_runs({"y":2,"z":3}, {"y":float("nan"),"z":6})` →
  `ids==["z"]`、`dropped==["y"]`、`n_common==1`。
- **AC3(空輸入)**:`pair_runs({}, {"a":1})` → `n_common==0`、`ids==[]`、
  `only_b==["a"]`、不 raise。
- **AC4(決定性排序)**:亂序插入的 dict(如 `{"b":..,"a":..,"c":..}` 兩邊)→
  `ids` 恆為字典序 `["a","b","c"]`;同輸入呼叫兩次回傳完全相等。
- **AC5(錯誤路徑)**:值為 `"abc"`(不可轉 float)或 key 為 int →
  `raises ValueError`。
- **AC6(compare_inputs exact)**:`records_ref = records_a = records_b` 各 3 筆
  (path `p0..p2`、obj_index 0、verdict A=`["bad","good","good"]`、
  B=`["good","good","bad"]`),`confirmed_ref={0:"bad", 2:"good"}` →
  `run_a_err=={"p0#0":0, "p2#0":0}`、`run_b_err=={"p0#0":1, "p2#0":1}`、
  `n_labeled==2`(index 1 未確認 → 不出現)。
- **AC7(G1 方向鏈=植入證明,整合 26+28)**:16 筆 records(10 筆 confirmed
  `"bad"` + 6 筆 `"good"`);A 全判對(bad→"bad"、good→"good"),B 全漏
  (verdict 全 `"good"`)→ `compare_inputs`+`pair_runs`+
  `paired_compare(higher_is_better=False)` 鏈 → `verdict=="a_better"` 且
  `mean_diff==pytest.approx(10/16)`——G1 的方向性在純函式層鎖死
  (GUI E2E 只驗接線,見 M21_gui_wiring 修訂)。
- **AC8(殘留索引跳過)**:`confirmed_ref={99:"bad"}`(超出 3 筆 records_ref)→
  `n_labeled==0`、err dict 皆空、不 raise。
