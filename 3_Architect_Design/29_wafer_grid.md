# 29 `wafer_grid` — 晶圓 die 格點整形與點陣渲染(Tier A)

> M22。PRD [2_PO_PRD/wafer_spatial_signature_prd.md](../2_PO_PRD/wafer_spatial_signature_prd.md)。
> 純函式、無 I/O(PIL Image 物件回傳,不落地)、無 streamlit。實作 `scripts/wafer_grid.py`。
> G6 鐵則:**零欄位假設**——缺 wafer 欄且未獲使用者明示「同屬一片」→ ValueError,
> 絕不默默當同一片;對不上/缺座標逐項列名,絕不默默少畫。

## 1. 目的

metadata 列 + per-item 判定 → 逐片晶圓的 die 格點結構(含對不上清單),
及該結構的 PIL 點陣地圖(匯出用;像素可驗)。

## 2. I/O 契約

```python
_WAFER_ALIASES = ("wafer", "wafer_id", "waferid", "wafer_no", "晶圓", "片號")
_X_ALIASES    = ("die_x", "x", "col", "column", "diex")
_Y_ALIASES    = ("die_y", "y", "row", "diey")
_LOT_ALIASES  = ("lot", "lot_id", "lotid", "lot_no", "批號")

def guess_fields(fieldnames: Sequence[str]) -> dict
# → {"wafer": str|None, "x": str|None, "y": str|None, "lot": str|None}
#   比對規則:欄名 casefold 後與別名表**完全相等**才中;同表多欄中 → 取先出現者。

def build_wafer_grids(
    items: list[dict],            # per-item:{"item_id","path","flag"(0/1),"score"(float|None)}
    meta_lookup: Mapping[str, dict],   # path → metadata row(呼叫端由 meta_join per_image 組)
    *, wafer_field: str | None, x_field: str, y_field: str,
    lot_field: str | None = None,
    assume_single_wafer: bool = False,   # G6:僅使用者顯式勾選才可 True
) -> dict
# → {
#   "wafers": [ {"wafer_id": str, "lot": str|None,
#                "dies": [{"x": float, "y": float, "flag": int,
#                          "path": str, "score": float|None}, ...],  # path 字典序
#                "n": int, "k": int} ],       # wafer_id 字典序;k=Σflag
#   "unmatched": [path...],        # meta_lookup 查無列(字典序去重)
#   "missing_coord": [path...],    # 有列但 x/y 缺或不可轉 float(字典序去重)
#   "missing_wafer": [path...],    # wafer 欄值缺/空字串(僅 wafer_field 非 None)
# }

def wafer_map_png(wafer: dict, *, cell: int = 24, pad: int = 2) -> "PIL.Image.Image"
# 匯出用點陣地圖:die 座標線性映射到畫布格位。顏色釘死:
#   flag=1 → (220, 60, 60);flag=0 → (120, 160, 220);背景 → (245, 245, 245)。
# 畫布大小 = ((x 格數)*cell + 2*pad, (y 格數)*cell + 2*pad);
# x 格位 = round((x - x_min) / step_x),step_x = 相鄰唯一 x 值的最小正差(唯一值 1 個
# 時 step=1);y 同理。y 軸向下(影像座標);同格重複 die → flag 取 max 後畫一次。
```

## 3. 資料流

1. **die 聚合**:items 依 `path` 聚合(一張影像=一顆 die):`flag = max(flags)`、
   `score = max(非 None scores)`(全 None → None)。
2. **查表**:path 依 meta_lookup 取列;查無 → `unmatched`。x/y 缺鍵、空字串或
   不可轉 float → `missing_coord`。wafer_field 給了但值缺/空 → `missing_wafer`。
3. **分片**:wafer_field 給了 → 依欄值分組;None 且 `assume_single_wafer=True`
   → 全部歸一片,`wafer_id="(單片)"`。lot_field 給了 → 取該片第一顆 die 的列值
   (同片多值 → 取字典序最小並照常回傳,不裁決)。
4. 排序:wafers 依 wafer_id 字典序;dies 依 path 字典序(決定性)。

## 4. 邊界條件與錯誤處理

- `wafer_field is None and not assume_single_wafer` → `ValueError`(訊息含
  `"wafer"` 與 `"同屬一片"`;G6:不猜)。
- `x_field is None` 或 `y_field is None` → `ValueError`。
- items 空 → 照常回傳(wafers=[],各清單=[])。
- 全部對不上 → wafers=[]、unmatched 全列。
- `wafer_map_png`:dies 空 → `ValueError`;單一 die → 1 格畫布照畫。

## 5. Acceptance Criteria

- **AC1(guess exact)**:`guess_fields(["LOT","Wafer_ID","Die_X","Die_Y","tool"])`
  `== {"wafer":"Wafer_ID","x":"Die_X","y":"Die_Y","lot":"LOT"}`;
  `guess_fields(["a","b"]) == {"wafer":None,"x":None,"y":None,"lot":None}`;
  `guess_fields(["x","col"])["x"] == "x"`(同表多欄取先出現)。
- **AC2(整形 exact)**:3 個 path(p0/p1/p2),p0 兩個 objects(flag 0/1、
  score 0.2/0.9)、meta 有 p0(W1,x=0,y=0)與 p1(W1,x=1,y=0,y 值 `"bad"`)、
  p2 查無列 → `wafers==[{"wafer_id":"W1",...,"dies":[{"x":0.0,"y":0.0,"flag":1,
  "path":"p0","score":0.9}],"n":1,"k":1}]`、`missing_coord==["p1"]`、
  `unmatched==["p2"]`。
- **AC3(G6 不猜)**:`wafer_field=None, assume_single_wafer=False` →
  `raises ValueError`(訊息含 `"同屬一片"`);`assume_single_wafer=True` →
  `wafers[0]["wafer_id"]=="(單片)"`。
- **AC4(分片與排序)**:兩片交錯輸入(W2/W1/W2/W1)→ wafers 順序 `["W1","W2"]`、
  各片 n/k 正確(手算 ==);lot 欄給了 → 各 wafer["lot"] 正確。
- **AC5(PNG 像素 exact,G3)**:單片 dies=[(0,0,flag1),(1,0,flag0),(0,1,flag0)],
  `cell=10,pad=0` → Image size==(20,20);pixel(5,5)==(220,60,60)(flag1 格中心)、
  pixel(15,5)==(120,160,220)、pixel(15,15)==(245,245,245)(空格=背景)。
- **AC6(missing_wafer)**:wafer 欄值空字串的 path 進 `missing_wafer` 且不進任何
  wafer(該片 n 不含它)。
- **AC7(錯誤路徑)**:x_field=None → ValueError;`wafer_map_png(dies=[])` → ValueError。
- **AC8(決定性)**:同輸入呼叫兩次回傳完全相等。
