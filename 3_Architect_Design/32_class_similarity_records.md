# 設計:32 `class_similarity_records`(身份、key、eligibility)

> Tier B(讀 records 結構,不寫資料集)。實作:`scripts/class_similarity_records.py`。

## I/O

```text
prepare_records(viz_records, raw, context) -> PreparedInput
build_analysis_key(prepared, embeddings, metadata) -> hex
scope_fingerprint(records, split, labels, statuses) -> hex
validate_selection(event, result, current_context) -> indices | Rejection
invalidate_if_fingerprint_mismatch(...) -> stale?
```

## 身份

- image: canonical JSON `{level:"image", path:norm}` 的 SHA-256。
- object: `{level:"object", image_path:norm, bbox:原樣有限數值, obj_index}`。
- Windows: resolve + normcase。ID 不含 label／分數／排序。
- 缺 provenance: `invalid:<data_token>:<original_index>`，reasons 記錄，不准進候選／購物車／評估。
- 同一身份多列 → DuplicateIdentityError。
- 同一 crop path 對不同 (image_path,bbox,obj_index) → CropIdentityConflictError，整份拒絕。
- 同內容不同路徑 → 不同 ID。crop policy 不進 ID(由 recipe/analysis key 管)。

## label_status

- `single|missing|multi|unresolved`。不改 viz_records 的 label 值與列序。
- Classifier: parent.name==split → unresolved(flat_fallback); parent.name==label → single(class_folder，含字面 unknown/mix); 空標 missing。
- Detector 整圖: mix→multi, unknown→missing, 單類→single。
- Detector 物件: 有完整 image_path/bbox/obj_index 且單一 class → single; 缺 provenance → invalid。
- 有效 label = 目前 record['label'](讀回覆蓋後); source_label 另存。人工 A→B 以 B 計分。
- 多標不挑第一類。

## analysis key

`SHA256(canonical JSON metadata + 有效列正規化 embedding bytes)`。
metadata 含 schema/method、data_token、順序 IDs/labels/splits、source_snapshot_key、eligibility、model/weights、recipe、K、analysis split。
source_snapshot 綁 **embedding 生成時刻** 的內容 hash。分析只核對。既有 raw 無 generation 證據 → source_unverified。object 空 viz_manifest ≠ 已驗證。

選取事件: analysis_key 必須等於當前非 stale key，且 original_index 的 record_id 相同，否則拒絕。

## AC

- **AC06** 改顯示類別／排序／指標不改分數; 改 K／label／raw／recipe／split 使舊結果失效。
- **AC08** 過濾後仍對到 original_index; 過期事件不能加入選取／購物車。
- **AC29** 同內容不同路徑 ID 不同; crop policy 改 ID 不變; 兩筆缺 provenance 各有不同暫存 ID 且不能進評估; crop 衝突 integrity error。
- **AC30** 讀回使舊結果過期; flat 不當 single; 合法 unknown/mix 資料夾保留; 人工 A→B 用 B; mode restore 舊事件無效。
- **AC34**(records 部分) 生成後改原圖再計算 → stale，不能新 hash 配舊 raw; 路徑或有效 label 變則舊 evaluation／操作點失效; 未驗 generation 無 PR。
