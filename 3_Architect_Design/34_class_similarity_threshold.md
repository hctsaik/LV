# 設計:34 `class_similarity_threshold`(候選、評估、曲線)

> Tier A。實作:`scripts/class_similarity_threshold.py`。所有 UI／匯出呼叫同一 predicate。

## I/O

```text
candidate_mask(items, policy={mode, threshold, comparator:"lt"})
summarize_candidates(items, mask)
workload_curve(items)
validate_evaluation_manifest(manifest, snapshot, items)
evaluate_threshold(items, mask, evaluation)
precision_recall_curve(items, evaluation)
preview_key(analysis_key, policy, evaluation_key)
truth_digest(...) / evaluation_key(...)
save_operating_point / validate_heldout
```

## 規則

- eligible = finite margin; candidate = margin < t; 相等不入選; null 永不入選。
- mode none / all_eligible; t=2 仍不含 margin=2。
- 計分 cache key 不含 threshold。
- 工作量曲線: 唯一 margin 排序，同分整組，切點=nextafter(v,+∞)，含 none; max=2 顯示「全部可計分」。
- PR 正例=incorrect。E=E0∩eligible。E 非空且皆 binary 才給 P/R。分母 0 → null。
- 手算: t=0.1 → TP=FP=FN=TN=1, P=R=0.5; t=0.2 → 2/1/0/1, P=2/3, R=1; none P=null R=0; all P=0.5 R=1。
- heldout: 凍結完整 frozen_policy; ID 與 source_group 不得與 tuning 重疊。

## AC

AC15, AC16(數值層), AC17, AC19, AC20, AC21, AC22, AC24, AC25(draft/commit 純函式), AC27, AC28, AC34(評估／操作點)。
