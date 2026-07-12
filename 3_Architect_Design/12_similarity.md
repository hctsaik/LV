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

---

## 增補(M12 UX 精修):`class_centroid`(依類別當參考)

> 使用者回饋:裸索引 UX 差。加「參考依據:類別 / 具體物件」切換。類別參考 = 該類別所有物件 embedding 平均。

### 契約
```python
def class_centroid(obj_emb, labels, class_name) -> np.ndarray:
    """回 labels==class_name 的所有物件 embedding 平均向量(未正規化;cosine 內部會正規化),float32 (D,)。
    無此類別 / 空 → ValueError。"""
```
- `obj_emb`:(N, D);`labels`:長度 N 的類別字串序列;`class_name`:字串。
- 供 GUI「參考依據=類別」時算參考向量,再交給 `similarity_priority` / `al_batch(ref_vector=)`。

### AC
- **AC9**:`class_centroid([[1,0],[0,1],[1,0]], ["a","b","a"], "a")` == `[1,0]`(逐元素);`"b"` == `[0,1]`。
- **AC10**:`class_centroid(..., "c")`(不存在的類別)→ `ValueError`。
- **AC11(衍生)**:`similarity_priority(X, class_centroid(X, labs, c))` 對「屬於 c 的物件」給高分
  (centroid 與同類同向)—— 用可分兩類驗 argmax 落在該類。

---

## 增補(M13 Task 1):`multi_ref_similarity`(多樣本 per-class max-cosine)

> M13「以樣搜樣」用:對每物件算「最像哪一類樣本、多像」。**max**(不取平均)—— 每類樣本少(5~10)且可能
> 含次型,centroid 會糊;max=「像任何一顆就算像」。Tier A 純陣列。

### 契約
```python
def multi_ref_similarity(obj_emb, ref_vectors, ref_labels) -> tuple:
    """回 (best_labels, best_sims):
      best_labels: np.ndarray (N,) 物件字串類別 —— 每物件最像的那一類;
      best_sims:   np.ndarray (N,) float32 ∈ [-1,1] —— 該類的 max-cosine。
    sim(x, 類c) = max over 樣本_j∈c 的 cosine(x, 樣本_j)(各自 L2 normalize)。
    obj_emb:(N,D);ref_vectors:(M,D);ref_labels:長度 M 的類別序列。
    N==0 → 回兩個空陣列;M==0 / 維度不符 / 任一 ref 列範數~0 → ValueError。"""
```

### 資料流
`obj_u=L2(obj_emb)`、`ref_u=L2(ref_vectors)` → `cos=obj_u@ref_u.T`(N,M) →
對每個 distinct 類別,取其 ref 欄的 **max** → per-class 分數 (N, C) → `argmax`=best_label、`max`=best_sim。

### 邊界
| 情境 | 行為 |
|------|------|
| N==0 | 回 `(np.array([]), np.zeros(0,float32))` |
| M==0(無樣本) | `ValueError`(訊息含「樣本」) |
| obj D ≠ ref D | `ValueError` |
| 任一 ref 列範數 < 1e-9(全零樣本) | `ValueError`(不可比) |
| tie(兩類同分) | 取 ref_labels 出現順序較前的類別(argmax stable) |

### Acceptance Criteria(釘死;給 `/pm` 加進 tests/test_similarity.py)
- **AC-M1(基本)**:`multi_ref_similarity([[1,0]], [[1,0],[0,1]], ["a","b"])` →
  best_labels==["a"]、best_sims≈[1.0](差<1e-6)。
- **AC-M2(max 非平均)**:同類兩樣本 `[[1,0],[0,1]]` 都 label "a",query `[[0,1]]` →
  best=("a", 1.0)(max(0,1)=1;若取平均會是 0.5)。**這條鎖死 max 語義。**
- **AC-M3(歸最像的類)**:refs `[[1,0],[0,1]]` labels `["a","b"]`,query `[[0.9,0.1]]` →
  best_labels==["b"?]... 應==["a"](cos 到 [1,0] 較高)。best_sims 為該值。
- **AC-M4(空物件)**:obj (0,2) → best_labels.shape==(0,)、best_sims.shape==(0,)。
- **AC-M5(空樣本)**:`ref_vectors=[]`(M=0)→ ValueError。
- **AC-M6(維度不符)**:obj D=2、ref D=3 → ValueError。
- **AC-M7(全零樣本列)**:某 ref 列 `[0,0]` → ValueError。
- **AC-M8(衍生:與單參考一致)**:單類單顆時,`multi_ref_similarity(X,[r],["a"])` 的 best_sims
  == `cosine_similarity_to_ref(X, r)`(逐元素<1e-6)—— 多參考是單參考的一般化。
