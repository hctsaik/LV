# 設計:12 `similarity`(對參考物件的 cosine 相似優先分數)

> 對應 PRD:[2_PO_PRD/similar_objective_prd.md](../2_PO_PRD/similar_objective_prd.md)。Tier A(純陣列、無 I/O、無 GUI)。
> 供 A1(③ 互動)與 A3(al_batch objective=similar)共用。**只出契約 + AC,不寫實作。**

## 1. 目的
給定一個參考物件向量,對一批物件算「有多像它」(cosine),正規化成優先分數(降冪=最像),
讓「找長得像指定物件的」變成一個可排序的選樣目標。

## 2. I/O 契約(`scripts/similarity.py`)

```python
def cosine_similarity_to_ref(obj_emb, ref_vec) -> np.ndarray:
    """每物件對 ref_vec 的 cosine 相似度 ∈ [-1, 1],float32,形狀 (N,)。
    obj_emb:(N, D);ref_vec:(D,)。兩者各自 L2 normalize 後內積。
    ref_vec 範數 ~0(全零)→ ValueError;維度不符 → ValueError;N==0 → 形狀 (0,)。"""

def similarity_priority(obj_emb, ref_vec) -> np.ndarray:
    """= _minmax(cosine_similarity_to_ref(obj_emb, ref_vec))(降冪=最像),float32 (N,) ∈ [0,1]。
    _minmax 與 active_learning 同義:hi<=lo(單一/全等)→ 全 0。N==0 → (0,)。"""
```

- `_minmax(x)`:`lo,hi=min,max`;`hi<=lo → zeros_like(x)`;否則 `(x-lo)/(hi-lo)`(與 `active_learning._minmax` 一致)。
- L2 normalize:`v / max(‖v‖, 1e-12)`;ref 範數 < 1e-9 視為全零 → raise。

## 3. 資料流
`obj_emb (N,D)` + `ref_vec (D,)` → 各自 L2 正規化 → 內積得 cosine (N,) → `_minmax` → 優先分數 (N,)。
下游:A1 用它排 `select_for_labeling`;A3 在 `al_batch._merge_and_topk` 用 `_minmax(concat(ref_sim))`(等價)。

## 4. 邊界與錯誤
| 情境 | 行為 |
|------|------|
| `ref_vec` 全零(範數<1e-9) | 兩函式皆 `ValueError`(訊息含「參考」或「零」) |
| `obj_emb` 維度 D ≠ `ref_vec` 維度 | `ValueError` |
| `N==0`(obj_emb (0,D)) | 回形狀 `(0,)` float32 |
| 單一物件 / 全部等距 | `similarity_priority` 回全 0(minmax 退化);cosine 仍給真值 |

## 5. Acceptance Criteria(釘死;給 `/pm`)
- **AC1(cosine 值)**:`cosine_similarity_to_ref([[1,0],[0,1],[3,0]], [1,0])` ≈ `[1.0, 0.0, 1.0]`
  (各差 <1e-6;[3,0] 與 [1,0] 同向 → 1.0)。
- **AC2(相似排序 + 範圍)**:`similarity_priority([[1,0],[0,1],[0.9,0.1]], [1,0])`:
  `argsort(desc) == [0,2,1]`(最像在前);所有值 ∈ [0,1];`max==1.0`、`min==0.0`。
- **AC3(ref 全零)**:`ref_vec=[0,0]` → 兩函式皆 `ValueError`。
- **AC4(空輸入)**:`obj_emb` 形狀 (0,2) → 回形狀 (0,)。
- **AC5(維度不符)**:obj D=2、ref D=3 → `ValueError`。
- **AC6(單一物件)**:`similarity_priority([[1,0]], [1,0])` → 形狀 (1,)、值 `0.0`(minmax 退化),不 crash。
- **AC7(衍生:正尺度不變)**:`similarity_priority(X, r)` 與 `similarity_priority(2*X, r)`
  **逐元素相等**(cosine 對正尺度不變 → 排序與分數不變)。
- **AC8(衍生:參考同向者最像)**:X 含一列 = ref 的正倍數 → 該列 `similarity_priority` 為最大值(==1.0),
  即 `argmax(similarity_priority(X, r))` 指向與 r 同向那列。
