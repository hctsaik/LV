# 30 `spatial_sig` — 單片晶圓空間簽名統計引擎(Tier A)

> M22。PRD [2_PO_PRD/wafer_spatial_signature_prd.md](../2_PO_PRD/wafer_spatial_signature_prd.md)。
> 純函式、無 I/O、無 streamlit。實作 `scripts/spatial_sig.py`。
> 統計鐵則(09/10 裁決):null=**within-wafer / die-mask 受限重排**——只在**這片
> 晶圓實際存在的 die 位置**之間隨機重排缺陷標記、**保缺陷數 k**;嚴禁全域 shuffle、
> 嚴禁假設完整格點。G2 精神:寧可說「看不出模式」。

## 1. 目的

單片 die 格點+缺陷標記 → 三簽名(聚一團/偏邊緣/線狀)統計量 + 受限重排 null 的
MC p 值 → 四值誠實結論。

## 2. I/O 契約

```python
def spatial_signature(
    dies: list[dict],            # [{"x": float, "y": float, "flag": 0|1}, ...](單片)
    *, alpha: float = 0.05,
    n_min: int = 20,             # die 太少寧可看不出來
    k_min: int = 5,              # 缺陷太少寧可看不出來
    n_perm: int = 2000,
    seed: int = 0,
) -> dict
# → {
#   "verdict": "clustered" | "edge" | "linear" | "none",
#   "p": {"clustered": float, "edge": float, "linear": float} | None,   # 門檻擋下時 None
#   "stats": {"clustered": float, "edge": float, "linear": float} | None,
#   "n": int, "k": int,
#   "alpha": float,              # 名目 α(判定用 α/3,Bonferroni 三比較)
#   "reasons": list[str],        # verdict=="none" 的理由(可多條);顯著時 []
#   "method": str,               # 固定片段見 AC8
# }
```

## 3. 資料流

1. **門檻(保守,依序收集 reasons 不短路)**:`n < n_min` → `"die 太少"`;
   `k < k_min` → `"缺陷太少"`;`n - k < 5` → `"無對比(幾乎全可疑)"`。
   任一命中 → verdict `"none"`、`p=stats=None`(不硬算)。
2. **三統計量**(座標先以全體 die 質心平移;`R = max_i dist(die_i, 質心)`,
   `R==0` → 加 reason `"座標退化"` 走 none):
   - `clustered`:缺陷 die 兩兩歐氏距離的平均;**越小越聚**。
     單邊 p = P(T_perm ≤ T_obs)。
   - `edge`:缺陷 die 到質心距離 / R 的平均;**越大越偏邊**。p = P(T_perm ≥ T_obs)。
   - `linear`:缺陷 die 座標 2×2 共變異數特徵值 λ1≥λ2,T = λ1/(λ1+λ2);
     **越接近 1 越線狀**。λ1+λ2==0(缺陷全同點)→ T=0.5(中性,聚集已由
     clustered 抓)。p = P(T_perm ≥ T_obs)。
3. **受限重排 null**:`rng = np.random.default_rng(seed)`;每次從**這片的 n 個
   die 位置**中不放回抽 k 個當缺陷(= die-mask 受限、保 k),同一批重排樣本
   同時評三個統計量(共用 draws);`p = (#極端 + 1) / (n_perm + 1)`。
4. **判定**:顯著集合 = `{s : p[s] ≤ alpha/3}`(Bonferroni)。空 → `"none"` +
   reason `"未達顯著"`。非空 → 取 p 最小者;**平手依序 `linear > edge > clustered`**
   (線狀最特異優先——一條線同時也會「聚」,取更有資訊量的稱呼)。

## 4. 邊界條件與錯誤處理

- dies 空、缺 `x`/`y`/`flag` 鍵、座標非有限值 → `ValueError`。
- `flag` 非 {0,1} → `ValueError`。
- `alpha` 不在 (0,1)、`n_perm < 1`、`n_min < 1`、`k_min < 1` → `ValueError`。
- k==0 → 門檻路徑(缺陷太少),不 raise。

## 5. Acceptance Criteria

固定格點:`GRID = [(x, y) for x in range(10) for y in range(10)]`(n=100)。
以下 planted 皆用預設參數(alpha=0.05 → 判定門檻 0.05/3;n_perm=2000, seed=0)。

- **AC1(G1 聚一團)**:缺陷=**中心**緊鄰 6 格 {(4,4),(4,5),(5,4),(5,5),(4,6),(6,4)}
  → `verdict=="clustered"` 且 `p["clustered"] <= 0.05/3`。
  (刻意選中心:角落緊鄰塊會同時極偏邊,兩統計量 p 齊打 MC 下限打平,
  平手序將誤報 edge——中心塊的 edge 方向相反,歧義消失。)
- **AC2(G1 偏邊緣)**:缺陷=最外圈 8 格 {(0,0),(0,9),(9,0),(9,9),(0,5),(9,5),
  (5,0),(5,9)} → `verdict=="edge"` 且 `p["edge"] <= 0.05/3`。
- **AC3(G1 線狀+平手優先序)**:缺陷=一列 {(x,5) for x in 0..7}(8 格共線)
  → `verdict=="linear"`(此例 clustered 亦可能同顯著/同 p,平手序保證報 linear)
  且 `stats["linear"] == pytest.approx(1.0)`。
- **AC4(G2 隨機誠實,單例+多 seed)**:`default_rng(123)` 從 GRID 抽 8 格當缺陷
  → `verdict=="none"`。統計驗收:200 個 seed(0..199)各抽 8 格隨機缺陷 →
  `mean(verdict != "none") <= 0.075`(名目 α+裕度;Bonferroni 後應更低)。
- **AC5(G2② die-mask 對照=裁決核心)**:環狀抽檢 mask:
  `RING = [p for p in GRID if dist(p, 質心) >= 0.7*R_full]`(只有外圈 die 存在);
  缺陷 = `default_rng(7)` 從 RING 抽 6 格(對這片而言=隨機)。
  - `spatial_signature(RING mask)` → `verdict=="none"`(within-mask null 誠實)。
  - **同一批缺陷座標**、但 dies 換成全格點 GRID(=錯誤地假設完整格點的 null)
    → `p["edge"] <= 0.05/3`(會誤判偏邊)。兩者對照即「嚴禁全域/全格點 shuffle」
    的機器證明。
- **AC6(門檻誠實)**:n=10 全格(<20)→ `verdict=="none"` 且 `"die 太少"` in
  reasons、`p is None`;GRID 上 k=2 → `"缺陷太少"`;k=98(n−k=2<5)→
  `"無對比" in "".join(reasons)`。
- **AC7(決定性)**:AC1 輸入呼叫兩次 → 回傳 dict 完全相等。
- **AC8(method line 固定片段)**:任一有算 p 的呼叫,`method` 同時含
  `"受限重排"`、`"N="`、`"k="`、`"α"`、`"Bonferroni"`、`"exploratory"`。
- **AC9(錯誤路徑)**:dies=[]、flag=2、x=NaN、alpha=1.5 各自 `raises ValueError`。
