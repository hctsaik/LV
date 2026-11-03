## Context

LV already supports source-first replay, fixed-target Grad-CAM, evidence lineage and case replies. Legacy Ultralytics 8.3.203 and original YOLOv5 must remain operational. Newer Detect heads and export paths require explicit contracts. Current scattered controls and unpaired time logs cannot substantiate faithful explanation or effort savings.

## Goals / Non-Goals

Goals: inspect and execute YOLO12/26 in an isolated pinned runtime; preserve native target identity; provide question-aware cited summaries; compare fair perturbations; measure independent sample support and matched effort with review quality.

Non-goals: prove causal defect mechanisms from CAM, invent production efficacy, or automatically send replies. Segmentation/OBB/pose and unsupported exported gradients receive explicit capability results.

## Decisions

1. Registry IDs `legacy-8.3.203` and `modern-8.4.142` select controlled interpreters. Payloads never select arbitrary executable paths. Existing missing IDs default legacy. Modern dependencies are independent and pinned; cross-runtime evaluation dispatches separately per model.
2. Model inspection captures actual release, family, task, head path, NMS mode, export path and gradient capability. The modern contract includes final output row and native candidate/class identity. Perturbation locks the native candidate even when top-k order changes. Preserve legacy hashes.
3. A pure `attribution_controls` module emits deterministic mask variants lazily. Translations preserve exact mask geometry without wrapping/clipping and require at most 10% overlap with the attribution mask. Infeasible controls are explicit unavailable outcomes. Constant and blurred fills share masks; assessment separates fraction/fill/strategy and marks inconsistent directions unstable. Old protocols remain readable but unverified.
4. An `explanation_summary` module composes controlled observations, uncertainty, evidence IDs and next actions. Question types are class score, box localization, postprocessing and machine alarm. Human confirmation references immutable variant hashes as separate evidence. Edited drafts change only through explicit regeneration.
5. `explanation_validation` counts unique source hashes and declared independent groups separately from target/run counts. Cohorts retain verified role, reviewer, model/runtime/head/protocol identity; missing/conflicting metadata is visible and cannot become independent support. Effort comparison requires matched study/pair/reviewer records, active time, waiting time, completion and independent quality assessment; incomplete legacy logs remain visible but excluded from benefit estimates.

## Risks / Trade-offs

- Large CAM masks may have no feasible translated controls → report unavailable instead of substituting a biased control.
- Model score sensitivity can result from distribution shift → multi-fill checks and human preservation review constrain wording.
- New runtime consumes disk and memory → reuse wheel cache, one heavy worker, explicit resource bounds and separate environment.
- Native top-k mapping can silently drift → real forward/backward/export correspondence tests including duplicated anchor classes and reordered outputs.
- Small or correlated samples can exaggerate support → unique image/group counts and no population-level claim.

## Migration Plan

Keep legacy runtime and artifacts unchanged. Install modern runtime through the registry setup command and persist resolved dependencies. Old profiles resolve to legacy. New reports version their protocol and identities. Rollback uses the legacy profile/runtime without rewriting stored results.

## Open Questions

None blocking implementation. Production model efficacy and engineering effort require measured user datasets.
