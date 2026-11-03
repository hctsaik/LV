# 設計:33 `class_similarity_export`(CSV／JSON bytes)

> Tier A(純 bytes)。實作:`scripts/class_similarity_export.py`。不寫資料集、不送標。

## I/O

```text
export_report(result, records, *, scope, local_selected_ids,
              committed_preview, evaluation=None) -> (csv_bytes, json_bytes)
evaluation_template(...) -> json_bytes  # schema 1, 另函式可放 threshold 模組
```

scope ∈ {all, candidates, selected}。基礎階段無 committed 候選時 candidates 停用。
selected = local_selected_ids ∩ 當前 U 有效身份。不併入全域 viz_selection。

## JSON schema_version=2

含 method_version=class_topk_margin_v1、analysis_key、created_at_utc、scope、config、summary、records 索引、items、exclusions、policy、preview_key、candidate_summary、evaluation、score_index(全 U)、evaluation_items(完整 E0)、frozen_policy。
null 用 JSON null，禁止 NaN/Inf。中文可往返。鄰居 records 可解析但不列為選中 query。
三種 scope 都保留全 U score_index 與完整 E0 evaluation_items。invalid 詳細 items／CSV 僅 all。
score_index 含 label/status/reasons/margin，以便子集重建每類總數／eligible／候選／無分數與全局 PR。無法分類列進「未分類／標籤無效」bucket。

## CSV

每列一 query: analysis_key, record_id, original_index, level, path, image_path, bbox_json, obj_index, label, split, same, other, rival_class, margin, same_k, rival_k, status, policy_mode, threshold, comparator, is_candidate。無效為空欄不是 0。

## AC

- **AC09** 同原圖兩 ROI 仍兩物件; crop／原圖／bbox 一致; 缺圖明確。
- **AC11** CSV/JSON 可解析; 中文往返; null 合法; 鄰居 ID 可解析; mean 可由 JSON 鄰居重算。
- **AC23** 讀回可重建候選與 PR; candidates 含未畫出資料; threshold 精度保留。
- **AC31** selected 只含 A query 且 C 不在鄰居、另有一 invalid 僅 all 含詳細; 三 scope 都能從 score_index 重算全類統計與全局 PR。
