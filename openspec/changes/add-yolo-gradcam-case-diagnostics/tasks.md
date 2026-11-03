## 1. Existing behavior and provider boundary

- [x] 1.0 Walk through U01–U12 with an interactive workflow prototype and fixed inputs; record primary actions, correction, incomplete-data and return paths before integrating engines (docs/lv-investigation-product-review.md).
- [x] 1.1 Record baseline DINO V1 contract/hash, margin semantics and both UI entry regression results; preserve the existing 224/336/448 path.
- [x] 1.2 Extract provider execution dispatch before DINO head preparation, preserving render-only page behavior and no-auto-inference entry.
- [x] 1.3 Add detector V2 context/target/run/result contracts with method, native target, layer and geometry identity; test stale and late-result isolation without changing V1 hashes.

## 2. Model profiles and verified YOLO adapters

- [x] 2.1 Add reusable immutable model-profile revisions with deployment and gradient backends, snapshot persistence and capability reporting.
- [x] 2.2 Pin optional dependency environments and document actual version/head/task checks, including original v5 versus v5u and unsupported variants.
- [x] 2.3 Implement and verify original YOLOv5 Detect forward/backward with selected objectness-times-class scalar and explicit class-only alternative.
- [x] 2.4 Implement and verify YOLOv8 Detect forward/backward with native candidate/class-score and actual feature-level mapping.
- [x] 2.5 Implement and verify YOLO11 Detect forward/backward with independent compatibility tests and feature-level mapping.
- [x] 2.6 Add per-model execution isolation, hook cleanup, cancellation/error handling and normal-forward/gradient-forward parity tests.
- [x] 2.7 Implement upload-first model import with persistent content-addressed files, visible size limits/progress, duplicate handling, atomic completion and advanced server-path entry; test interrupted, invalid and same-name uploads.
- [x] 2.8 Implement explicit capability inspection through constrained pinned workers and guided required-setting confirmation, preserving metadata/default/user provenance and actionable unsupported-model errors.
- [x] 2.9 Implement one-image trial, reusable My Models cards and save-and-return-to-case flow; support ONNX-only usage and later companion PT attachment without losing case context.

## 3. Detection identity and Grad-CAM

- [x] 3.1 Map selected final detections to native raw targets, testing NMS reordering, adjacent same-class boxes, empty detections and truncated candidate lists.
- [x] 3.2 Implement positive Grad-CAM and distinctly labeled signed-score diagnostics with raw-map persistence and zero/non-finite/saturated-state handling.
- [x] 3.3 Implement layer-to-input-to-original geometry and tested layer selection for non-square images, padding and small targets, without implicit crop inference.
- [x] 3.4 Add PyTorch-to-runtime correspondence records using explicit tensor/output mapping and representative-set support; preserve separate historical deployment parity.

## 4. Unified diagnostics and case evidence

- [x] 4.1 Add case selected-box action and case context resolver independent of Visualize session; reuse model profiles across cases without repeated input.
- [x] 4.2 Extend the existing Diagnostics shell with provider-specific prediction/score semantics, source labels and evidence states; preserve DINO UI and render-only controls.
- [x] 4.3 Implement atomic case attribution artifacts and revision-safe evidence publication, including raw maps, inputs, overlay, model/target/method identities and corruption checks.
- [x] 4.4 Add save-to-case, report claim references, redrawable evidence download and restart recovery; test stale/historical display after target/profile changes.

## 5. Evidence validation and Active Learning

- [x] 5.1 Add fixed original-scalar perturbation traces distinct from rematched final detections, with multi-proportion deletion and seeded equal-area/random controls.
- [x] 5.2 Add bounded batch faithfulness reports with support, variability and inconclusive states; test differing replacement detections and missing targets.
- [x] 5.3 Add explicit mask alignment and localization metrics with separate unavailable states, size/group support and checkpoint-randomization sanity reporting.
- [x] 5.4 Connect selected CAM regions to existing comparison search and review batches with evidence provenance; preserve human review, full-image training and evaluation overlap rules.
- [x] 5.5 Add fixed-case effort measurement protocol and recording for first setup, per-case time, repeated input, wrong-target rework and evidence-review returns.

## 6. Integrated release verification

- [x] 6.1 Run real pinned YOLOv5/v8/11 forward/backward/export tests in addition to synthetic controlled-target tests; publish adapter/version capability results.
- [x] 6.2 Pass browser case-to-box-to-CAM-to-experiment-to-report-to-AL/export/restart flow and unsupported-backend branches without DINO substitution.
- [x] 6.3 Pass both existing DINO browser entries, three-resolution regressions, external replay/training/evaluation regressions and no-source-write checks.
- [x] 6.4 Publish integration guide, supported formats, model-input requirements, evidence semantics, measured validation report and explicit production-model validation status; validate OpenSpec before marking complete.
- [x] 6.5 Pass upload-to-inspection-to-trial-to-selected-box E2E for supported PT, ONNX-only, paired models and failed inspection; verify second-case model reuse and no required internal tensor/layer/JSON input for standard adapters.
- [x] 6.6 Complete U01–U16 acceptance matrix with implementation and real-browser evidence; distinguish actual production-model acceptance from synthetic/adapter validation and do not declare overall completion while ordinary-user paths are incomplete.

## 7. Complete engineer workflow (integrate before release verification)

- [x] 7.1 Implement case-first and model-first navigation with persistent case/source/run/profile selection, case search/status/next actions and trial-to-case promotion without reupload or unnecessary inference (U01–U04).
- [x] 7.2 Decouple source-image inference from pre-existing case items; add graphical detection selection, ROI drawing/editing and candidate withdrawal with historical reference impact handling (U06–U07).
- [x] 7.3 Implement multi-image/prediction/annotation import preview with role, path pairing, explicit class and coordinate mapping, duplicates, per-file errors and confirmed zero-detection state (U05).
- [x] 7.4 Import supported historical CSV/JSON/TXT via structured field mapping without requiring pre-existing LV hashes; retain raw records, missing fields and scope of partial comparison (U03,U05,U07).
- [x] 7.5 Extend deployment revisions to distinguish station settings and recorded model/postprocessing/final-machine-decision stages; unknown downstream rules prevent full-decision replay claims (U04,U09).
- [x] 7.6 Add reusable uploaded reference/evaluation datasets, truth and group/purpose confirmation, GUI-generated manifests and training class dropdowns; preserve independent-set and full-image guards (U11–U12).
- [x] 7.7 Add evidence support/opposition/pending relationships, editable fact-based reply drafts, selected-image preview, copy text and portable HTML reply with separate technical appendix and reply revisions (U08,U10).
- [x] 7.8 Implement persistent queued/running/completed/error/cancelled/interrupted jobs, duplicate submission prevention, navigation recovery, per-item retry and explicit saved-result-versus-case-publication status (U13,U15).
- [x] 7.9 Add non-destructive conflict recovery, model rename/archive/restore and missing-asset reattachment; support workspace backup/restore manifests with explicit model inclusion and protected referenced assets (U14–U15).
