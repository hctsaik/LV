## Context

r3 design in `docs/class-similarity-review-implementation-plan.md` is the implementation contract. This change adds the feature as an extension of Visualize label quality, not a new tool. Workspace already has unrelated uncommitted files; they stay.

## Goals / Non-Goals

**Goals:** Per-record same / rival-other / margin from raw embeddings; box+scatter+table+neighbors; margin threshold preview and workload; human-evaluation Precision/Recall when a frozen binary set exists; schema-2 export; image and object E2E; source datasets unread-write.

**Non-Goals:** Top-N, per-class thresholds, ANN, auto F1, gray-band revival, auto relabel, Labeling schema change, cross-split model generalization.

## Decisions

- **Math:** scale-safe float64 cosine; same = Top-K mean excluding self by record_id; other = max over other classes of that class's Top-K mean; ties by (-sim, record_id) then class key. Production uses query×reference blocks; never a full N×N matrix.
- **Identity:** SHA-256 of canonical JSON `{level, path}` or `{level, image_path, bbox, obj_index}`. Not `explain_contracts.stable_record_key`. Invalid rows get `invalid:<data_token>:<original_index>`. Duplicate identity or crop-path collision aborts the run.
- **Labels:** adapter `label_status=single|missing|multi|unresolved`. Classifier class subfolders (including literal unknown/mix) are single; `allow_flat` parent==split is unresolved. Detector mix=multi, unknown=missing, single ROI=single. Effective label is the current record label after readback.
- **Threshold:** only `margin < t`. Slider and number share one canonical value. Draft vs committed preview. Actions carry analysis_key, preview_key, request_seq.
- **PR:** positive = human verdict `incorrect`. Unknown is not correct. Heldout freezes full operating point and source groups.
- **UI:** segmented panel dispatch before the old 3-record/2-class guard. Local checkbox set ≠ global `viz_selection` ≠ threshold candidates.
- **Generation:** bind source hashes at embedding time. Analysis only verifies. Unverified generation: exploration allowed, PR forbidden.

## Risks / Trade-offs

- [Streamlit slider may commit only on release] → measure with pointer held; add local component if needed; never claim live preview without that evidence.
- [Object crops collide on stem] → refuse analysis, do not silently repair cache names.
- [O(N²D) time] → cap 5000; blocks only bound memory.
- [Cart 4-decimal path key] → cart is a container; JSON is the evidence.

## Migration Plan

Additive panel. Removing the dispatch hides the feature. No dataset migration. Schema 2 is the only analysis report; evaluation import is schema 1.

## Open Questions

None blocking. Performance numbers and live-drag evidence are acceptance outputs, not design guesses.
