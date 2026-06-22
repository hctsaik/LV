# 設計 03:bootstrap_cluster(Tier A,純函式)

## Purpose
把物件級 embedding 分群,標出「正常密群」與「離群候選」(供 2-stage bootstrap 抽樣後請使用者確認;1-stage 亦用它定義正常)。

## 復用 / 依賴
- 物件級 embedding 由 `object_eval.embed_objects`(meanpool/cls, D=384)產生。
- 分群:`sklearn.cluster.HDBSCAN`(sklearn 1.9,已確認可用)。**先 L2 正規化再用 euclidean**(等價 cosine,避開 HDBSCAN cosine 後端差異)。
- 抽樣可用 `interaction.farthest_point_sampling`(多樣性抽樣,bootstrap 取樣交給 06,本模組只負責「分群+標候選」)。

## I/O 契約
```
ClusterResult = {
  "labels":          np.ndarray,  # (N,) int,-1 = noise
  "candidate_mask":  np.ndarray,  # (N,) bool,True = 離群候選(待使用者確認壞)
  "n_clusters":      int,         # 非 noise 群數
  "normal_mask":     np.ndarray,  # (N,) bool = ~candidate_mask(密群成員)
}

def cluster_objects(
    emb: np.ndarray,              # (N, D)
    *, min_cluster_size: int | None = None,
    small_cluster_frac: float = 0.05,
) -> ClusterResult
```

## Data Flow
1. L2 正規化 `emb`。
2. `min_cluster_size = min_cluster_size or max(5, round(0.05*N))`(下限 5)。
3. `HDBSCAN(min_cluster_size=…, metric="euclidean").fit_predict` → `labels`。
4. 候選定義:`candidate_mask[i] = (labels[i]==-1)` **或** `i` 所屬群大小 `< small_thresh`,
   其中 `small_thresh = max(min_cluster_size, ceil(small_cluster_frac*N), 0.5*最大群大小)`。其餘為 `normal_mask`。
   **`0.5*最大群`** 這項是關鍵(與資料規模無關):它把「明顯小於主群的少數群」一律視為候選——
   即使少數壞的剛好聚成一個 ≥min_cluster_size 的小群,也會被標出(對 1-stage 尤其重要,否則少數缺陷群會被當正常)。
   **誠實上限**:在 1-stage 無確認下,真正的「少數但良品」類型(如某個量少的好類)也會被誤標候選——這要靠 2-stage 確認校正。
5. `n_clusters = labels 中非 -1 的相異值數`。

## 邊界與錯誤處理
- `N < 2`:`labels=[0]*N`(或單元素 [0])、`candidate_mask` 全 False、`n_clusters = (1 if N>=1 else 0)`,不丟例外。
- 全部相同向量:HDBSCAN 可能整體歸一群或全 noise;契約只保證「不丟例外、形狀正確」,不對此退化輸入斷言群數。
- 比較群相等時用「分群劃分(partition)」而非原始 label id(HDBSCAN label 編號不保證穩定)。

## Acceptance Criteria
資料用固定 `rng=default_rng(0)`,D=16。
- **AC1**(乾淨三群):3 個高斯團(中心相距 ≥10σ,各 50 點)→ `n_clusters==3`;每個團內 `candidate_mask` 全 False;同團兩點在同一群(partition 一致)。
- **AC2**(離群點):AC1 三團 + 3 個遠離所有團的散點(距任一中心 ≥20σ)→ 那 3 點 `labels==-1` 且 `candidate_mask==True`;團成員 `candidate_mask==False`。
- **AC3**(小群即候選):兩大群(各 60 點)+ 一小群(4 點、離大群遠)→ 小群 4 點 `candidate_mask==True`(因群大小 < 門檻),大群成員 False。對應「少數壞的聚成小群也要被挑出」。
- **AC4**(形狀/型別):`labels.shape==(N,)`、`candidate_mask.dtype==bool`、`normal_mask == ~candidate_mask`(逐元素)。
- **AC5**(退化):`N=1` 與 `N=0` 皆不丟例外,形狀正確,`candidate_mask` 全 False。
- **AC6**(決定性):同輸入跑兩次,partition 與 `candidate_mask` 完全一致。
