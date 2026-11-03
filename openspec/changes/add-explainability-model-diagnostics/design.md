## Context

VisualLatent currently renders DINO Attention, classification attribution, and Normal Bank anomaly evidence through `scripts/explain_ui.py`, called from both the Visualize viewer and Anomaly inspector in `scripts/app.py`. The engine already preserves raw and display attribution arrays, supports canonical global-channel and token gradient-times-activation aggregations, validates some head provenance, and can run higher-resolution spatial probes.

The current surface is suitable for quickly inspecting one image, but it mixes model training, validation warnings, attribution method, display mode, analysis resolution, and overlay controls. A user can therefore see a visually concentrated map before understanding that the classifier is weak, the selected target is counterfactual, or the 448 run still uses a 224-trained head.

The capsule exploration motivating this change used a local six-class, 132-image cache and found weak sample-level OOF performance and low `crack` recall. Those numbers are exploratory until a fixed dataset fingerprint, run specification, and machine-readable report are committed. Existing E2E fixtures use a smaller three-class subset and do not reproduce the six-class results.

Constraints:

- Source datasets must remain read-only; application artifacts belong under `.lv_cache` only after an explicit save.
- Existing user modifications in the working tree must be preserved.
- P1 must remain a synchronous, selected-sample workflow; expensive batch and background execution is phased later.
- Current `checkpoint_identity()` uses path metadata and must migrate to content identity before persistent result reuse is trusted.
- Current sample OOF is not grouped validation.
- Existing inline resolution and method tests need an explicit compatibility-to-workspace migration.
- The detailed rationale and full research backlog remain in `docs/explainability-model-diagnostics-workspace-design.md`; the simplified single-action interaction is documented in `docs/model-diagnostics-ui-ux-redesign.md`.

## Goals / Non-Goals

**Goals:**

- Put classifier reliability and decision semantics before attribution rendering.
- Create a top-level Model Diagnostics workspace without duplicating source images, embeddings, heads, or Normal Banks.
- Establish versioned context, run, render, provenance, digest, and stale contracts shared by inline and workspace UI.
- Use canonical signed Grad-CAM as the primary classification-attribution path while retaining token-wise attribution as an explicitly experimental comparison.
- Distinguish selected-sample smoke evidence from grouped classifier evidence, batch faithfulness, and mask localization.
- Use repeated learning curves to investigate whether performance is data-limited.
- Separate spatial probes from true resolution-specific embeddings and classifier heads.
- Make resolution, target, and method selectors side-effect-free and use one stateful primary action for the complete selected-image analysis intent.
- Define testable rollout and release gates from P0 correctness through P5 localization-model escalation.

**Non-Goals:**

- Treat Grad-CAM, DINO Attention, or Normal Bank anomaly evidence as interchangeable.
- Produce a crack mask, bounding box, boundary, or measurement from Grad-CAM.
- Claim that higher resolution is better without paired exact-head validation.
- Infer acquisition groups from filenames or folder hierarchy without an approved data contract.
- Start expensive work on page navigation or display-control changes.
- Deliver background distributed jobs, MIL, or segmentation in P1.

## Decisions

### Decision 1: Use a handoff-first dual-surface product model

The inline panel remains a selected-image summary and handoff surface. The new
top-level workspace owns head preparation, resolution, reliability, provenance,
research controls, attribution rendering, multi-run comparison, and progressively
heavier diagnostics. After replacement coverage exists, the inline primary flow
must not duplicate head training or the full Grad-CAM control stack.

Rationale: duplicating a complete analysis workflow across both surfaces lets old
buttons and state semantics survive in one path after the other path changes.

Alternative considered: remove the inline panel completely. Rejected because the
selected-image summary and handoff remain useful in source workflows.

### Decision 2: Transfer references through `ExplainContextV1`

The handoff stores stable identity and provenance, not PIL images, model objects, embedding arrays, or bank vectors. Whole-image identity is based on source SHA-256; object identity additionally includes object index, normalized geometry, and crop recipe.

The initial P1 source is Visualize classifier context. Anomaly inspector support is P1.1 and has a separate capability-aware flow when no classifier head exists.

Alternative considered: reuse current session indexes and widget state. Rejected because record order and Streamlit reruns can make those references point to different data.

### Decision 3: Separate compute and render specifications

`ExplainRunSpecV1` owns model-affecting choices. `RenderSpecV1` owns opacity, palette, grid, and display size. Render-only changes cannot modify the compute key or rerun the model.

Alternative considered: hash all widget state. Rejected because it creates needless inference and makes equivalent raw evidence appear to be different runs.

### Decision 4: Report orthogonal evidence dimensions

The UI displays independent badges for:

- Head validation: incompatible/stale, insufficient, sample audit available, grouped supported.
- Attribution faithfulness: unavailable, unsupported, inconclusive, single-sample smoke supported, batch supported.
- Localization: unavailable, incompatible, unverified, mask-agreement supported.

Decision ambiguity is a separate property of the current margin. A composite sentence may summarize the badges but cannot hide a failing dimension.

Alternative considered: one ordered status enum. Rejected because P1 can have a supported perturbation smoke test while localization remains unverified and the head remains weak.

### Decision 5: Make canonical signed Grad-CAM the primary attribution

The primary method uses channel-averaged gradients and a target-minus-reference logit margin without ReLU. The signed legend is:

- Blue: lowers the target-minus-reference margin or acts against the target.
- White: near neutral.
- Red: raises the margin or supports the target.

In a multiclass model, blue is not guaranteed to exclusively support the named reference. Token gradient-times-activation remains an advanced, separately named experimental method.

Alternative considered: retain token-wise attribution as the product default because it may look more focused. Rejected because visual concentration is not a validation criterion and the method is not canonical Grad-CAM.

### Decision 6: Migrate attribution semantics before adding the page

P0 freezes formulas, names, raw arrays, signed normalization, deterministic sign tests, and schema identity. Only after those contracts are stable does P1 add the page.

Alternative considered: change aggregation, normalization, page state, and cache identity in one release. Rejected because failures would not be attributable to a single layer.

### Decision 7: Warn on weak-head attribution without hiding research access

Weak classifier evidence does not make the gradient undefined. The system therefore
keeps a concise research-only limitation visible and permits the normal attribution
action directly, while forbidding trusted or localized claims. A second confirmation
button adds friction without adding evidence.

Alternative considered: hide all heatmaps for weak heads. Rejected because they can reveal classifier shortcuts and failure modes.

### Decision 8: Treat P1 faithfulness as a smoke test

P1 fixes target/reference and compares Top-K perturbation with recorded repeated random controls. The result can only become `single_sample_smoke_supported`. Batch deletion curves, AOPC, stability, confidence intervals, and baselines are P4.

Alternative considered: call the current single-random occlusion audit a full faithfulness check. Rejected because it lacks a distribution, uncertainty, and batch evidence.

### Decision 9: Delay visible mask comparison until the localization contract exists

P1 reports mask availability, incompatibility, or not evaluated. It does not default to a mask/Grad-CAM overlay. P4 owns traceable geometry, batch metrics, baselines, uncertainty, and any research thresholds.

Alternative considered: show the mask beside the heatmap in P1. Rejected because it encourages the same visual “looks focused” judgment this change is intended to prevent.

### Decision 10: Distinguish spatial probes from exact heads

A head may run at another resolution only through an explicit research probe. Exact resolution mode requires resolution-specific full-dataset embeddings, head provenance, and validation. Probes never enter the validated leaderboard or recommendation logic.

Alternative considered: treat dimension-compatible embeddings as interchangeable. Rejected because the input distribution, classifier calibration, and validation recipe differ even when embedding dimensions match.

### Decision 11: Make multi-resolution identity recipe-aware

P3 introduces a recipe key containing model, checkpoint digest, target resolution, pooling, scope, crop/pad, aspect policy, normalization, and schema version. Embedding namespaces, manifest references, and head registry entries use this identity.

Alternative considered: append only resolution to the current model-name cache. Rejected because other recipe dimensions can also invalidate embeddings and heads.

### Decision 12: Use content-addressed persistent artifacts only on explicit save

P1 results are session-only but can be downloaded as metadata and raw JSON/NPZ. Later explicit saves write atomic artifacts below `.lv_cache/explainability/runs/<key>/`. Legacy path/size/mtime identities are never silently promoted to content-verified artifacts.

Alternative considered: persist every run automatically. Rejected because it changes current no-cache-write behavior, increases stale risk, and creates data-retention surprises.

### Decision 13: Keep UI orchestration outside the engine

New modules divide responsibilities:

- `explain_contracts.py`: contexts, run/render specs, statuses, canonical serialization.
- `explain_context.py`: handoff, identity, geometry, compatibility, stale validation.
- `explain_engine.py`: framework-free Attention, Grad-CAM, and Anomaly orchestration without implicit disk writes.
- `explain_diagnostics.py`: OOF summaries, learning curves, faithfulness, masks, shortcuts.
- `explain_artifact.py`: atomic content-addressed persistence and migrations.
- `explain_page.py`: Streamlit workspace only.

`explain_ui.py` becomes an inline adapter. Both surfaces call the same engine and renderer.

Alternative considered: import private helpers from `explain_ui.py` into the page. Rejected because preprocessing, target selection, and stale behavior would diverge.

### Decision 14: Phase validation by scientific claim

- P0: contracts and correctness.
- P1: selected-sample research preview from Visualize.
- P1.1: Anomaly inspector context.
- P2: grouped validation and learning curves.
- P3: exact multi-resolution models.
- P4: batch localization, full faithfulness, shortcuts, and evidence packages.
- P5: evidence-driven Normal Bank, patch classifier, MIL, or segmentation work.

Tasks remain unchecked until implementation, required tests, and documentation evidence are complete.

### Decision 15: Treat selected-image analysis as one user action

For an actionable context, the page exposes at most one primary analysis action.
Resolution, target, and method selectors only configure the requested run. When
the selected exact-resolution head is missing, activation of the primary action
runs embeddings, head training/audit, and attribution as stages of one foreground
task and automatically continues to the result. A prepared head skips completed
stages. Weak-head limitations remain visible but never introduce an acknowledgement
action.

The action label communicates the actual work, for example `分析此圖`,
`建立 448 模型並分析此圖`, `使用新設定重新分析`, or `重試分析`.

Alternative considered: keep separate build and attribution buttons because they
map directly to internal functions. Rejected because they split one user intent,
create avoidable clicks, and expose orchestration details as product decisions.

Alternative considered: start 336/448 preparation when the selector changes.
Rejected because selectors should be reversible configuration controls, and a
selection alone does not communicate consent for an expensive foreground job.

## Risks / Trade-offs

- [Users may still interpret any red region as a crack] → Put reliability before rendering, use margin-based legend text, keep warnings persistent, and prohibit localization claims without batch evidence.
- [Canonical Grad-CAM may appear less focused than token-wise attribution] → Preserve token-wise research comparison but do not optimize product defaults for visual sharpness.
- [Existing inline E2E conflicts with control relocation] → Keep P0 compatibility, add page/engine replacement coverage, then migrate or remove only the tests tied to intentionally removed selectors.
- [Local capsule numbers are not a reproducible regression baseline] → Add a fixed fingerprint, run spec, record identity, geometry, and machine-readable report without copying the external dataset into the repository.
- [Mask filenames or dimensions may be misleading] → Require an explicit resolver and traceable geometry; otherwise report unavailable or incompatible.
- [Group-aware validation may be impossible with current metadata] → Report grouped validation unavailable and never infer production generalization from sample-level OOF.
- [Repeated curves and high resolution are expensive] → Require explicit jobs, show phase-specific work estimates and cache hits, and serialize model/device access.
- [One action may contain a long head build] → State the selected resolution and expected work in the action label, show embeddings/head/attribution progress, reuse completed sub-results, and expose one retry action after failure.
- [Background jobs can publish late or leak hooks] → Keep P1 foreground-only; later jobs use generation checks, model leases, cooperative cancellation, and exception cleanup.
- [Content hashing increases initial cost] → Compute and cache strong digests at controlled boundaries; retain legacy identities only as unverified metadata.
- [A single status can hide contradictory evidence] → Keep head, attribution, and localization badges orthogonal.

## Migration Plan

1. Initialize OpenSpec and accept this proposal/spec/design/task set as the normative contract.
2. P0 adds versioned contracts, strong identity, stale rules, signed deterministic tests, method naming, and the shared engine boundary while retaining current inline controls.
3. Produce the fixed capsule exploratory evidence report and clearly separate it from the existing small E2E fixture.
4. P1 adds the Visualize handoff and selected-sample page, consolidates head preparation and attribution behind one primary action, makes selectors side-effect-free, then migrates resolution/method selector coverage from inline UI to page or engine tests.
5. Keep a compatibility path until replacement E2E passes; rollback consists of removing the new route while leaving shared engine contracts and the inline adapter active.
6. P1.1 adds the Anomaly inspector handoff without borrowing an incompatible head or Normal Bank.
7. P2 adds explicit group metadata, grouped validation, leakage audit, and learning curves.
8. P3 migrates embedding references and heads to recipe-aware identities, building exact heads before enabling the validated leaderboard.
9. P4 adds explicit artifacts, batch jobs, mask/faithfulness/shortcut reports, and scientific release gates.
10. Archive the OpenSpec change only after the agreed implementation scope is complete; unfinished later phases may be split into follow-up changes rather than falsely checked off.

## Open Questions

- What approved metadata field will define acquisition groups for P2?
- Should the P1 counterfactual target control be available by default in an advanced section or remain disabled until P2?
- What validation-set policy will replace provisional balanced-accuracy, critical-recall, calibration, and interval-width thresholds?
- Should P3 initially build 224 and 336 only, or generate all 224/336/448 recipes after the registry exists?
- What is the final product objective: explain classification, detect unknown anomalies, or output crack boundaries and measurements?
- Which external capsule dataset snapshot may be referenced by the reproducibility report without copying restricted data into the repository?
