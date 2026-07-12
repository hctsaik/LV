# 11 — hybrid_sampler 技術設計(不確定 × 多樣性 × 類配額 混合選樣)

對應 PRD:[2_PO_PRD/al_loop_infrastructure_prd.md](../2_PO_PRD/al_loop_infrastructure_prd.md) · 里程碑 M11 · Tier A(核心)+ B(接線)

## 1. 目的(Purpose)

修掉批次主動學習的經典陷阱:「最不確定的前 K 張」常是近重複,選 50 張等於選 8 張。
做法用最穩的兩段式:**先取 top-(oversample×K) 不確定,再在其 embedding 上跑 FPS 取 K**;
可選類配額(弱類優先)。零新演算法依賴——FPS 已存在(`farthest_point_sampling`)。

## 2. I/O 契約

### 2.1 新模組 `scripts/hybrid_sampler.py`(依賴 numpy + interaction.farthest_point_sampling)

```python
def hybrid_select(
    emb: "np.ndarray",                 # (N, D) embedding
    uncertainty: "np.ndarray",         # (N,) 分數,越大越該標(來源不限:分歧/離群/異常/1-conf)
    k: int,
    *,
    oversample: int = 3,               # 候選池 = top-(oversample*k) 不確定
    exclude: "set[int] | None" = None, # 直接剔除的索引(例:pool_registry 已處理者)
    labels: "list[str] | None" = None, # 給類配額用;None=不做配額
    class_quota: "dict[str, int] | None" = None,  # {類名: 至少幾張};None=不做配額
    seed_idx: "list[int] | None" = None,          # FPS 起點(例:已選樣本),預設取池內最不確定者
) -> list[int]
    # 步驟(全決定論):
    # 1) cand = np.argsort(-uncertainty, kind="stable") 過濾 exclude,取前 oversample*k(不足全取)。
    # 2) 若 class_quota:先依配額從 cand 內每類各取「該類最不確定的 quota 張」進 picked,
    #    配額類樣本不足 → 有多少拿多少(缺口誠實留給第 3 步,不硬湊)。
    # 3) 其餘名額:在 cand \ picked 的 emb 上跑 farthest_point_sampling,
    #    起點 = seed_idx∩cand(空則 cand 中最不確定者),補滿到 k。
    # 4) 回傳依「選中順序」的原始索引 list,長度 = min(k, len(cand))。
    # k<=0 → [];N==0 → []。不修改輸入。

def near_duplicate_pairs(emb: "np.ndarray", idxs: list[int], thr: float = 0.05) -> int
    # 選中集合內 cosine 距離 < thr 的無序對數(AC 與 UI 顯示「本批近重複對」用)。O(k²) 可接受(k 小)。
```

### 2.2 GUI 接線(Tier B,`scripts/app.py` Visualize「選樣」panel)

- 既有選樣 panel 的策略選單加一項「**混合(不確定×多樣)**」:
  - 不確定來源 selectbox:`標籤分歧`(既有 disagreement)/ `離群度`(既有 outlier)/
    `低信心`(僅 `viz_pred_root` 已填時出現,=1−conf,吃 10 號的預測);
  - K(既有)、oversample slider(2–10,預設 3);
  - 類配額編輯(Could:先做「弱類自動配額」checkbox = 把最少類保底 `k//類數`,手動編輯不做);
  - `exclude` 自動接 08:`pool_registry` 已 sent/labeled 的索引預設剔除(checkbox 可關,與送標端一致)。
- 選樣結果 caption 顯示:`本批近重複對:X(純 top-K 為 Y)`——把改善攤在使用者眼前。
- 送標時策略名帶 `"hybrid:<不確定來源>"`(接 09 帳本)。

## 3. 資料流

viz Run(embedding 已在)→ 不確定分數(既有訊號之一)→ hybrid_select(剔除已處理)→
畫廊+近重複對 caption → 加購物車/送標(策略名入帳)。

## 4. 邊界條件與錯誤處理

| 情況 | 預期行為 |
|---|---|
| k ≥ 可選數 | 全取(不重複、不報錯) |
| oversample*k ≥ N | 候選=全體(退化為純 FPS-加權起點,行為仍決定論) |
| exclude 蓋掉大半 | 從剩餘者選;全被蓋 → [] + UI 顯示原因 |
| uncertainty 全相等 | stable argsort → 依索引序取候選;FPS 仍給多樣性 |
| labels 給了但 class_quota=None | 等同不配額 |
| quota 類在候選池內樣本不足 | 取到沒有為止,缺口由 FPS 補(不硬湊、不報錯) |
| emb 與 uncertainty 長度不符 | ValueError(明確契約) |

## 5. Acceptance Criteria

**單元(`tests/test_hybrid_sampler.py`;合成 embedding,決定論 seed)**

- **AC1(長度與範圍)**:N=100, k=10 → 回 10 個相異索引,皆 ∈ [0,100)。
- **AC2(不確定門檻)**:所有選中索引的 uncertainty 排名 ≤ oversample*k(候選池外永不入選)。
- **AC3(多樣性勝 top-K,核心 AC)**:造 fixture——3 個緊密叢(叢內 cosine 距 <0.01)各 20 點+
  40 個散點;讓「最不確定前 10」全落在同一叢。則
  `near_duplicate_pairs(emb, hybrid_select(...k=10), 0.05) < near_duplicate_pairs(emb, top10, 0.05) / 2`
  (釘死:混合選樣近重複對數少於純 top-K 的一半)。
- **AC4(exclude)**:把 top-3 不確定索引放進 exclude → 選中集合與其交集為空。
- **AC5(配額)**:labels 三類、class_quota={"rare": 2},候選池內 rare 有 3 張 →
  選中集合內 rare ≥ 2,且該 2 張是池內 rare 中最不確定的 2 張。
- **AC6(配額不足誠實)**:quota={"rare": 5} 但池內 rare 僅 2 → 選中 rare==2,總數仍==k(FPS 補滿)。
- **AC7(決定論)**:同輸入重呼 20 次結果逐位相等。
- **AC8(防呆)**:len(emb)!=len(uncertainty) 拋 ValueError;k=0 回 []。

**GUI / 整合(真實 E2E,`tests/e2e/test_hybrid_sampler_e2e.py`)**

- **AC-G1(策略可用)**:viz Run 後選「混合(不確定×多樣)」策略、K=6 → 畫廊恰 6 張、
  caption 出現「近重複對」數字。
- **AC-G2(接 08)**:先把部分樣本標為已送標(直寫 registry)→ 重新選樣 → 該樣本不在結果
  (以檔名斷言);關掉剔除 checkbox → 回到可入選。
- **AC-G3(接 09)**:送標後回合卡策略名含 "hybrid:"。

## 6. 給 /pm 的提示

- AC3 的 fixture 造法:單位球面上 3 個中心 + 高斯微擾(σ 足夠小),不確定分數 = 到某叢中心的負距離,
  保證 top-10 全在該叢;請把 fixture 寫成 helper 讓 AC3/AC7 共用。
- `farthest_point_sampling` 既有簽名先查 `scripts/interaction.py`,設計假設它吃 (emb, k, start_idx);
  若簽名不合,允許在 hybrid_sampler 內包一層薄 adapter(不改既有函式)。
- 低信心來源依賴 10 號落地;PM 可把 AC-G1 用「標籤分歧」跑,避免跨模組排程阻塞。
