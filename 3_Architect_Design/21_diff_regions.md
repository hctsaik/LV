# 21 — diff_regions 技術設計(差異分數圖 → Top-K 區域 + 代表樣本 + 白話摘要)

對應 PRD:[2_PO_PRD/goodbad_patch_diff_prd.md](../2_PO_PRD/goodbad_patch_diff_prd.md) · 里程碑 M19 · **Tier A(純陣列運算,無 I/O)**

## 1. 目的(Purpose)

把 20 的逐 patch 統計變成人看得懂的東西:**Top-K 連通差異區域**(排序、可點)、
每區**該看哪幾張圖**(Good 最典型 / Bad 最能展示差異的代表樣本索引)、
以及每區一句**白話穩定度**摘要資料。回答 PRD 開放問題 3/4:區域抽取用
**8 連通合併**(沿 `heatmap_to_boxes` 的 scipy.ndimage 慣例——相鄰高分 patch
是同一塊物理區域,單 patch 排名會把一塊差異拆成好幾名);代表樣本用
**區內平均距離**排(Bad 取最大=最能展示、Good 取最小=最典型)。

## 2. I/O 契約

### `scripts/diff_regions.py`(依賴:numpy + scipy.ndimage;禁 I/O)

```python
def extract_regions(
    z_map: "np.ndarray",          # (gh, gw) float,20 的 z reshape 成 grid
    stability_map: "np.ndarray",  # (gh, gw) float,同 reshape
    *,
    z_thresh: float,              # 進區域門檻(建議餵 20 的 threshold_z)
    top_k: int = 5,
    min_patches: int = 1,         # 區域含 patch 數 < 此值 → 濾掉(雜點)
) -> list[dict]
# mask = z_map >= z_thresh → scipy.ndimage.label(structure=np.ones((3,3)))(8 連通)
# → 每 component:{"patch_idx": list[int](flat row-major 索引,遞增),
#     "bbox_grid": (r0, c0, r1, c1)(inclusive-exclusive,numpy 慣例),
#     "peak_idx": int(區內 z 最大的 flat 索引), "peak_z": float, "mean_z": float,
#     "stability": float(= peak patch 的 stability——峰值處最能代表此區),
#     "rank": int(1 起)}
# 依 peak_z 降冪 → min_patches 過濾 → 取前 top_k、填 rank。全低於門檻 → []。

def pick_representatives(
    region: dict,
    dist_good: "np.ndarray",      # (Ng, P) 20 的輸出
    dist_bad: "np.ndarray",       # (Nb, P)
    *,
    n_each: int = 3,
) -> dict
# {"good_idx": list[int], "bad_idx": list[int]}
# 對區內 patch(region["patch_idx"])的距離取每張圖的平均:
#   bad_idx  = 平均距離**最大**的 n_each 張(降冪)——最能展示差異;
#   good_idx = 平均距離**最小**的 n_each 張(升冪)——最典型的正常長相。
# n_each > 樣本數 → 全取(不重複、不報錯)。

def region_summary(region: dict) -> dict
# {"pct": int, "text": str}
# pct = round(100 * region["stability"]);
# text = f"Bad 群 {pct}% 的圖在此處與 Good 群明顯不同"(GUI/CSV 直接用)。
```

## 3. 邊界條件與錯誤處理

| 情況 | 預期行為 |
|---|---|
| z_map 與 stability_map 形狀不一致 | ValueError |
| z_map 非 2D | ValueError |
| 全部低於 z_thresh | `[]`(誠實空結果,不硬擠) |
| 斜角相鄰的兩個高分 patch | 8 連通 → 同一區域 |
| top_k=0 | `[]` |
| region["patch_idx"] 為空(防禦) | pick_representatives 拋 ValueError |
| dist 矩陣 P 與 patch_idx 超界 | IndexError 自然拋出(不吞) |

## 4. Acceptance Criteria

**單元(`tests/test_diff_regions.py`;合成 z_map / dist 矩陣,全部手算釘死)**

- **AC1(雙熱區釘死)**:6×6 全 0,`z[1:3, 1:3]=5.0`(峰 z[1,1]=6.0)、`z[4, 4]=3.0`;
  stability_map 對應位置 0.9 / 0.5;`z_thresh=2.0` → 恰 2 區,rank1 峰=flat(1,1)=7、
  `bbox_grid==(1,1,3,3)`、`patch_idx` 含 4 個索引 {7,8,13,14}、stability==0.9;
  rank2 峰=flat(4,4)=28、`bbox_grid==(4,4,5,5)`。
- **AC2(空結果)**:全 0 圖、z_thresh=1.0 → `[]`。
- **AC3(8 連通)**:`z[1,1]=3`、`z[2,2]=3`(僅斜角相鄰)→ 1 個區域(patch_idx 兩個)。
- **AC4(top_k 與 min_patches)**:3 個分離熱區(峰 5/4/3)→ top_k=2 只回前兩名且
  rank 連續;min_patches=2 時單 patch 熱區被濾掉。
- **AC5(代表樣本手算)**:region.patch_idx=[0,1];dist_bad=[[.1,.1],[.9,.8],[.5,.5]] →
  bad_idx(n_each=2)==[1,2];dist_good=[[.3,.3],[.05,.05],[.2,.2]] → good_idx==[1,2]。
  n_each=5 > 3 張 → 回全部 3 張、順序正確。
- **AC6(白話句)**:stability=0.82 → pct==82 且 text=="Bad 群 82% 的圖在此處與 Good 群明顯不同";
  stability=0.005 → pct==0(round)。
- **AC7(防呆)**:z_map 3D → ValueError;形狀不一致 → ValueError;空 patch_idx →
  pick_representatives ValueError。

## 5. 給 /pm 的提示

- flat 索引一律 row-major(`np.ravel_multi_index` 同義);AC1 的 {7,8,13,14} 已按
  6 欄寬手算((1,1)→7、(1,2)→8、(2,1)→13、(2,2)→14),寫測試前請再驗一次。
- 21 不 import 20(只吃陣列);測試也不需要 20。
