## 1. P0 Versioned Contracts and Identity

- [x] 1.1 Add `ExplainContextV1`, `ExplainRunSpecV1`, `RenderSpecV1`, result, job, and evidence-status contracts with canonical JSON serialization.
- [x] 1.2 Add stable whole-image and object-record keys based on source SHA-256, object index, normalized geometry, and crop recipe.
- [x] 1.3 Add content digests for checkpoints, classifier heads, and Normal Banks, including all behavior-affecting provenance fields.
- [x] 1.4 Define deterministic compute keys and prove that render-only settings do not change them.
- [x] 1.5 Define the complete stale/incompatible matrix and late-result generation checks.
- [x] 1.6 Add a dedicated `explain_page_*` session namespace that survives ordinary tool navigation but becomes stale after a new source generation.
- [x] 1.7 Add legacy path/size/mtime identity handling that marks unverified caches legacy or stale instead of silently promoting them.
- [x] 1.8 Add unit tests for context round-trip, payload size boundaries, identity stability, compute-key sensitivity, and stale precedence.

## 2. P0 Attribution Correctness and Naming

- [x] 2.1 Freeze and document canonical channel-weighted signed Grad-CAM and token gradient-times-activation formulas.
- [x] 2.2 Rename UI and metadata values so token-wise attribution is never presented as standard Grad-CAM or “recommended focus.”
- [x] 2.3 Make canonical signed Grad-CAM the primary product path while retaining token-wise attribution in advanced research controls.
- [x] 2.4 Preserve raw signed scores separately from display scores and target-positive derived rendering.
- [x] 2.5 Implement one symmetric robust signed scale and a flat-map numerical gate without separate positive/negative normalization.
- [x] 2.6 Update the legend to describe blue as lowering the target-reference margin and red as raising it.
- [ ] 2.7 Add deterministic negative/zero/positive color tests and a synthetic single-causal-token attribution test.
- [x] 2.8 Add regression tests for all-negative, all-zero, flat, CLS, mean-pool, and target/reference margin cases.

## 3. P0 Shared Engine Boundary

- [x] 3.1 Add `explain_context.py` for source resolution, scope, geometry, compatibility, and stale validation.
- [ ] 3.2 Add `explain_engine.py` as a framework-free entry point for Attention, Grad-CAM, and Anomaly execution without implicit disk writes.
- [ ] 3.3 Move shared preprocessing, source identity, recipe signature, head signature, and result signature logic out of private UI helpers.
- [ ] 3.4 Refactor `explain_ui.py` into a compact inline adapter that builds contracts and invokes the shared engine.
- [ ] 3.5 Ensure hooks, parameter gradients, locks, and memory are cleaned after success, error, or OOM.
- [x] 3.6 Keep existing inline controls and their tests operational during P0 compatibility migration.
- [x] 3.7 Run focused engine/unit tests and the existing no-dataset-write regression gate.

## 4. Reproducible Capsule Evidence Baseline

- [ ] 4.1 Define the external capsule dataset fingerprint, six-class counts, selected record SHA-256, and source geometry without copying the dataset into the repository.
- [ ] 4.2 Create a versioned run specification for the six-class head audit and `crack/001.png` exploratory analysis.
- [ ] 4.3 Export a machine-readable report for OOF metrics, class support, sample prediction, preprocessing, attribution method, target/reference, and calibration status.
- [ ] 4.4 Mark prior unarchived mask, Top-K, mass, AUC, and occlusion numbers exploratory until they are rerun with complete metadata.
- [ ] 4.5 Add a reproducibility check that distinguishes the full six-class evidence report from the current small three-class E2E fixture.

## 5. P1 Visualize-to-Diagnostics Workspace

- [x] 5.1 Add the top-level **Model Diagnostics** route and app-level navigation from a Visualize fragment.
- [x] 5.2 Implement reference-only Visualize handoff using dataset fingerprint, selected record identity, model, head, scope, recipe, and generation.
- [x] 5.3 Add a fixed context header for dataset, model, head, image, mask status, Normal Bank status, and current/stale result status.
- [x] 5.4 Display ground-truth label, current top-1, runner-up, fixed margin, held-out OOF prediction when available, and calibration status as distinct fields.
- [x] 5.5 Display sample-level OOF split type, sample/fold counts, balanced accuracy, macro-F1, target precision/recall/F1, support, and leakage limitation.
- [x] 5.6 Implement orthogonal head-validation, attribution-faithfulness, and localization badges plus a conservative composite explanation.
- [x] 5.7 Implement the weak-head primary action, concise persistent limitation, and direct attribution action without a redundant confirmation step.
- [x] 5.8 Render source image, actual model input, canonical signed overlay, textual legend, and optional patch grid with default opacity in the 0.45–0.60 range.
- [x] 5.9 Hide counterfactual targets in advanced controls and apply a persistent counterfactual label when the selected target is not current top-1.
- [x] 5.10 Ensure page entry, tab changes, image changes, and render controls never auto-start expensive computation.
- [x] 5.11 Reorganize the workspace into the guided selected-image → resolution selection → matching head preparation → heatmap → interpretation flow, with technical details progressively disclosed.
- [x] 5.12 Replace the separate head-preparation and attribution actions with one stateful primary analysis action that completes missing exact-head preparation and attribution in the same foreground task.
- [x] 5.13 Make 224/336/448 resolution selection side-effect-free; changing the selector updates readiness, estimated work, and action label but starts no embeddings, training, audit, or attribution.
- [x] 5.14 Add dynamic primary-action labels and one phase-specific progress surface for embeddings, head training/audit, attribution, retry, and prepared-head reuse.
- [ ] 5.15 Remove research-acknowledgement widget state, session keys, callbacks, and compatibility branches from workspace and inline paths while retaining persistent reliability limitations.
- [ ] 5.16 Reduce the Visualize inline primary flow to a selected-image summary and Model Diagnostics handoff after equivalent workspace/engine coverage exists.
- [ ] 5.17 Preserve reusable completed sub-results after a failed stage and ensure retry or late completion cannot publish against a newer context or run specification.

## 6. P1 Selected-Sample Evidence and Exports

- [ ] 6.1 Replace the single-random occlusion audit with a recorded Top-K versus repeated-random smoke policy whose target/reference remain fixed.
- [ ] 6.2 Record K/fraction, perturbation baseline, random repeat count, seed, Top-K margin drop, random distribution, and result status.
- [ ] 6.3 Prevent smoke-test success from changing head validation or localization state or removing reliability warnings.
- [ ] 6.4 Add a mask resolver interface and return unavailable until an explicit mapping implementation and fixture exist.
- [ ] 6.5 Show mask available, unavailable, incompatible, or localization-not-evaluated without a default mask/Grad-CAM overlay.
- [ ] 6.6 Add session-only run history keyed by context and deterministic run identity.
- [ ] 6.7 Add metadata JSON plus raw patch JSON/NPZ downloads that can reconstruct the signed display.
- [ ] 6.8 Verify that P1 does not write source datasets, existing embedding caches, anomaly caches, or persistent explanation artifacts.

## 7. P1 UI and E2E Migration Gates

- [x] 7.1 Add UI tests for app-level handoff, no-auto-run, weak-head limitation warnings, direct attribution, and stale context.
- [ ] 7.2 Add tests proving opacity, palette, grid, and display size rerender without model inference.
- [ ] 7.3 Add `crack/001.png` fixed-evidence tests for identity, label, OOF semantics, and 1000-to-224-to-16×16 geometry.
- [x] 7.4 Add tests distinguishing current prediction, label, OOF prediction, target, and reference.
- [x] 7.5 Add Playwright coverage for signed blue-white-red rendering and textual legend.
- [ ] 7.6 Move resolution, method, and target-positive controls to advanced compatibility before removing them from inline primary UI.
- [ ] 7.7 Replace intentionally removed inline selector E2E with equivalent workspace UI or engine integration coverage.
- [x] 7.8 Run all unaffected explainability E2E and no-cache-write regressions before completing P1.
- [x] 7.9 Add one-click Playwright coverage for missing and prepared exact heads at 224/336/448, proving that each selected-image analysis uses one primary action and never inserts a research-confirmation action.
- [ ] 7.10 Add UI tests proving resolution, target, and method selectors have no compute side effects and only update the pending run/action state.
- [ ] 7.11 Add failure/retry coverage proving completed embeddings or heads are reused and a late stage cannot overwrite a newer image or setting.

## 8. P1.1 Anomaly Inspector Context

- [ ] 8.1 Add stable Anomaly inspector handoff with source generation, object geometry, score, scope, model, and Normal Bank identity.
- [ ] 8.2 Display Normal Bank provenance and existing anomaly evidence without treating it as Grad-CAM or Attention.
- [ ] 8.3 Mark Grad-CAM unavailable when no compatible classifier head exists and never borrow another context's head.
- [ ] 8.4 Permit DINO Attention with its own semantics and limitations.
- [ ] 8.5 Add inspector stale, full-frame fallback, no-copy, and no-cache-write E2E coverage.

## 9. P2 Grouped Validation and Learning Curves

- [ ] 9.1 Define an explicit `group_id` contract for approved source, batch, lot, device, acquisition, or object metadata.
- [ ] 9.2 Implement grouped OOF or grouped holdout with group source, group counts, per-split class support, and unavailable reasons.
- [ ] 9.3 Add a fixed audit-set option that cannot be silently reused for repeated threshold tuning.
- [ ] 9.4 Add per-class metrics, confusion matrix, defect-vs-good metrics, AUROC/AUPRC, ECE, Brier, NLL, and confidence intervals where statistically valid.
- [ ] 9.5 Add duplicate and near-duplicate leakage auditing without inferring groups from paths by default.
- [ ] 9.6 Implement explicit repeated learning curves for 10/25/50/75/100 percent training fractions with fixed recipe and audit policy.
- [ ] 9.7 Record seed or grouped-bootstrap variance and late-curve marginal gains.
- [ ] 9.8 Implement conservative data-limited, low-plateau, overfit, high-variance, and class-specific interpretations.
- [ ] 9.9 Generate confusion-pair and diversity-oriented data collection recommendations without claiming a single proven cause.
- [ ] 9.10 Add P2 unit, integration, real-data, and report consistency tests.

## 10. P3 Exact Multi-Resolution Models

- [ ] 10.1 Define the extraction recipe key across model, checkpoint, resolution, pooling, scope, crop/pad, aspect policy, normalization, and schema.
- [ ] 10.2 Add recipe-aware embedding cache namespaces and manifest references with backward-compatible legacy handling.
- [ ] 10.3 Add an exact head registry with resolution-specific head digests and complete provenance.
- [ ] 10.4 Generate or resolve full-dataset 224 embeddings, train an exact 224 head, and run the agreed validation policy.
- [ ] 10.5 Generate full-dataset 336 embeddings, train an exact 336 head, and run the same validation policy.
- [ ] 10.6 Generate full-dataset 448 embeddings, train an exact 448 head, and run the same validation policy.
- [ ] 10.7 Prevent exact-mode fallback to a mismatched-resolution head and expose a separately confirmed probe action.
- [ ] 10.8 Build a paired dataset-level table for classification, per-class recall, calibration, prediction flips, latency, memory, cache, faithfulness, and localization when available.
- [ ] 10.9 Build selected-image comparison modes for each model's actual decision and for a fixed research target/reference.
- [ ] 10.10 Exclude spatial probes from validated ranking and recommendation logic.
- [ ] 10.11 Add cache collision, migration, exact-head selection, 16×16/24×24/32×32 grid, and non-monotonic recommendation tests.

## 11. P4 Mask Localization and Full Faithfulness

- [ ] 11.1 Implement explicit mask discovery and source-to-input-to-patch geometry with alignment validation.
- [ ] 11.2 Add class- and defect-size-stratified pointing, patch AUROC/AUPRC, Top-K precision/recall, attribution mass, and outside-mask false-positive metrics.
- [ ] 11.3 Add research-only thresholded IoU/Dice with validation-derived thresholds and non-segmentation labeling.
- [ ] 11.4 Add random-map, center-bias, DINO Attention, and compatible Normal Bank baselines.
- [ ] 11.5 Implement exhaustive or controlled perturbation ranking, deletion curves, AOPC, Spearman agreement, Top-K overlap, and bootstrap uncertainty.
- [ ] 11.6 Add label-preserving augmentation stability with recorded transformations and policy thresholds.
- [ ] 11.7 Report unavailable rather than supported when faithfulness or localization cannot complete because of mapping or resource failure.
- [ ] 11.8 Add batch real-data tests proving localization and faithfulness statuses cannot be promoted from selected-sample evidence.

## 12. P4 Shortcut Diagnostics, Jobs, and Artifacts

- [ ] 12.1 Implement explicit background-only, object-only, object-masked, text-masked, orientation, color, lighting, blur, translation, and rotation probes.
- [ ] 12.2 Record transformed input recipe, prediction flips, margins, calibration status, and attribution movement for every shortcut probe.
- [ ] 12.3 Extend the P1 foreground stage model to later diagnostic jobs and preserve completed sub-results after partial failure.
- [ ] 12.4 Add a serialized per-model/device worker design, generation checks, cooperative cancellation, and hook/OOM cleanup before enabling background jobs.
- [ ] 12.5 Add explicit atomic save/load under `.lv_cache/explainability/runs/<key>` with corruption detection and schema migration.
- [ ] 12.6 Export consistent HTML, JSON, CSV, PNG, and Markdown evidence packages without embedding source images by default.
- [ ] 12.7 Add report checks that always include limitations for sample OOF, probes, counterfactual targets, missing masks, and non-segmentation attribution.

## 13. P5 Evidence-Driven Localization Escalation

- [ ] 13.1 Establish a compatible patch Normal Bank baseline and evaluate image-, pixel-, and false-alarm metrics across groups.
- [ ] 13.2 Prototype patch classification only if whole-image classification evidence is insufficient for the required spatial task.
- [ ] 13.3 Prototype MIL only when image-level labels are available but mask coverage is insufficient, retaining independent faithfulness evaluation.
- [ ] 13.4 Define segmentation label budget, grouped split, Dice/IoU, boundary F-score, thin-structure recall, and clDice when crack boundaries or measurements are required.
- [ ] 13.5 Compare Normal Bank, patch classifier, MIL, and segmentation against the validated product objective before selecting a production direction.

## 14. Documentation and Release Gates

- [ ] 14.1 Keep `docs/explainability-model-diagnostics-workspace-design.md` synchronized with finalized contract, phase, and evidence decisions.
- [ ] 14.2 Document the exact meaning and allowed claims for every head, attribution, localization, probe, and counterfactual status.
- [ ] 14.3 Run focused tests first, then shared explainability regressions, Playwright E2E, and no-dataset-write checks for each completed phase.
- [ ] 14.4 Record skipped or unavailable validation gates with exact reasons; do not mark the corresponding task complete.
- [ ] 14.5 Do not archive this change as fully complete while later agreed phases remain unchecked; split intentionally deferred phases into follow-up changes if scope is narrowed.
