# 設計:31 `class_similarity`(分塊 exact Top-K margin)

> 對應 PRD:[2_PO_PRD/class_similarity_review_prd.md](../2_PO_PRD/class_similarity_review_prd.md)。
> 契約來源:docs/class-similarity-review-implementation-plan.md r3。Tier A 純陣列。
> 實作:`scripts/class_similarity.py`。**只出契約 + AC,不寫實作。**

## I/O

```text
analyze(embeddings, labels, record_ids, *, eligibility, invalid_reasons,
        k=5, query_block=128, reference_block=512, progress=None) -> AnalysisResult

build_box_summary(items, metric) -> list[per-class summary]
box_stats(values) -> {q1, median, q3, iqr, whisker_low, whisker_high, n}
```

- embeddings 可轉 float64 的 (N,D), D>0。外層維度／N 與 labels／ids／eligibility 長度不符 → 整體錯誤，不截斷。
- K 整數 1–20，拒絕 bool。
- 逐列:eligibility False、NaN/Inf、scale-safe 後過小／非有限 → 不進 query/reference。
- scale-safe: a=max(|row|); a=0 排除; v=row/a; b=||v||; 若 a≤1e-12/b 排除; 否則 u=v/b; 再驗 finite。
- cosine = clip(u·v, -1, 1)。
- 對 query i、類 c: J={j: label_j=c 且 id_j≠id_i}; 排序 (-sim, id); k_eff=min(K,|J|); S=前 k_eff 平均，空則 null。
- same=S(label_i); rival=argmax_{c≠label} S(c)，同分 class key 字典序; other=S(rival); margin=same-other。
- tied_rival_count 與按 class key 排序的前 10 個 tied_rival_classes(含選中 rival)。
- items 與輸入列對齊。status=ok|invalid|unscorable。N=0 → items=[]；全無效仍 N 列。
- 生產路徑任一時刻配對矩陣 ≤ (query_block, reference_block)。記錄 max_block_shape。
- 上限 5000 列(含無效)。超過 raise，不截前 5000。
- Q1/Q2/Q3 = numpy linear percentile; 鬚=落在 [Q1-1.5IQR, Q3+1.5IQR] 的實際 min/max。null 不當 0。

## AC

- **AC01** 手算 fixture K=1: a1/b1=(0.8,0.6,0.2), a2/b2=(0.8,0.96,-0.16); 加 c1 後 a1 rival 仍 B; 單 c1 三分數 null; A兩(1,0)+B兩(-1,0) margin=2; 全(1,0) margin=0; K=10 每類2筆 same_k=1 rival_k=2。K=2 逐類算法獨立測例。
- **AC02** 獨立暴力 oracle 與分塊一致; 多種 block size 一致。
- **AC03** 排除自己; 不同 ID 同向量保留; 同分穩定; 改列順序後以 ID 對齊結果不變。
- **AC04** 零/NaN/Inf/缺標/多標/長度錯/非法K/空/singleton/單類。
- **AC05** box 用全量; 抽樣不進四分位; [0,1,2,3,100] → Q1=1, median=2, Q3=3, 鬚 0 與 3。
- **AC12** >5000 拒絕; 中間暫存也不超 block; 記錄 max_block_shape。
- **AC32** [1e308,1e308] 與 [1,1] cosine=1; 微小向量排除; >10 同分 rival 只存前10 + 完整 count。
