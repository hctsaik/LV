# 設計 02:anomaly_score(Tier A,純函式)

## Purpose
用「正常 patch」建 memory bank(含 coreset 下採樣),計算查詢物件每個 patch 到 bank 的最近鄰 cosine 距離,聚合成**物件異常分數**與 **patch 分數圖**。這是 AnomalyDINO 的評分核心。

## 復用
- cosine 最近鄰索引可用 `interaction.build_nn_index`(hnswlib, space="cosine")或 `sklearn.neighbors.NearestNeighbors(metric="cosine")`。bank 規模(≤budget)下兩者皆可;PG 自選但須符合下列數值 AC。

## I/O 契約
```
class MemoryBank:  # 不透明 handle;持有正規化後的 bank 向量 + NN 索引
    size: int      # 實際 bank 向量數(coreset 後)

def build_memory_bank(
    normal_patch_feats: np.ndarray,   # (M, D) 已 L2 正規化(來自 patch_features)
    *, budget: int = 100_000, seed: int = 42,
) -> MemoryBank
#   M>budget → 隨機(seed 固定)下採樣到 budget;M<=budget → 全用。空輸入 → ValueError。

def score_object(
    query_feats: np.ndarray,          # (P, D) 已 L2 正規化
    grid: tuple[int,int],             # (gh, gw)，gh*gw==P
    bank: MemoryBank,
    *, q: float = 0.01,
) -> tuple[float, np.ndarray]
#   回 (object_score: float, patch_scores: np.ndarray shape==grid)。
#   patch_scores[r,c] = (1 - 最大 cosine 相似度 to bank) = 該 patch 的最近鄰 cosine 距離 ∈ [0,2]。
#   object_score = 取 patch_scores 攤平後「最高的 ceil(q*P) 個」之平均(至少 1 個)。
```

## Data Flow
1. `build_memory_bank`:L2 正規化(防呆,輸入應已正規化)→ 若 M>budget 用 `np.random.default_rng(seed)` 抽 budget 列 → 建 cosine NN 索引。
2. `score_object`:對每個 query patch 查 1-NN cosine 距離 → reshape 成 `grid` 得 `patch_scores` → 取最高 `k=max(1, ceil(q*P))` 個距離平均 = `object_score`。

## 邊界與錯誤處理
- `build_memory_bank` 空輸入(M==0)→ `ValueError("empty memory bank")`。
- `score_object` 空查詢(P==0)→ 回 `(0.0, np.zeros((gh,gw)))`(grid 給定)。
- `q` 夾在 (0,1];`ceil(q*P)` 夾到 [1,P]。
- 距離 float32 捨入可能出現微負 → clamp 到 ≥0(沿用 `_knn_query` 的 `np.maximum(...,0.0)`)。

## Acceptance Criteria(釘死數值;A==B→0、正交→1 等度量恆等式)
設 `A=e1`(D=8 的單位向量 [1,0,…]),`B=e2`([0,1,0,…]);皆已正規化,`cos(A,B)=0 → 距離=1.0`。
- **AC1**:bank 由 1000 個 A 構成。query 全 A(P=256, grid 16×16)→ 所有 `patch_scores≈0`(max<1e-6)、`object_score≈0`(<1e-6)。
- **AC2**:同 bank。query P=100、99 個 A + 1 個 B,`q=0.01`(k=ceil(1)=1)→ `object_score==1.0`(atol 1e-6),且 `patch_scores` 中那顆 B 的值==1.0、為全圖最大。
- **AC3**:`patch_scores.shape == grid` 恆成立;AC2 中 B 放在攤平索引 j → `patch_scores` 攤平後 argmax==j(位置正確)。
- **AC4**(coreset):`build_memory_bank(M=1000 隨機向量, budget=50).size == 50`;budget=5000 時 `.size==1000`(全用)。
- **AC5**(單調):同 bank,query 中 B 的數量 c ∈ {1,5,10},`q` 設成涵蓋(k≥c)→ `object_score` 隨 c 非遞減,且 c=k 時 ==1.0。
- **AC6**(可分性,餵給 G1 的基石):bank=正常分布(A 加小噪音 σ=0.01,正規化);正常 query=同分布、defect query=其中一 patch 換成 B。對 30 正常 + 30 defect 物件算分 → `min(defect_scores) > max(normal_scores)`(嚴格可分)。
- **AC7**:空 bank → `ValueError`;P==0 → `(0.0, zeros(grid))`,不丟例外。
