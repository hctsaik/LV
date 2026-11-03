## Context

The first-version case workbench persists region reviews and auxiliary LV comparisons. There is no deployed-model adapter in the repository. The machine has torch but no user-provided deployment model; ONNX is selected as a portable local inference boundary, with exact supported tensor layouts rather than guessed architectures.

## Goals / Non-Goals

**Goals:** Run compatible external models; preserve preprocessing and postprocessing evidence; distinguish local inference from historical parity; compare controlled image modifications; evaluate two recipes on identical labeled images; export reviewed examples safely.

**Non-Goals:** Inventing an absent production checkpoint, automatically certifying a root cause, training an unknown architecture, calling production services, automatic promotion/deployment, or treating case replay as independent validation.

## Decisions

- `fp_external_runtime.py`: single-image embedded-weight ONNX, explicit classification/logits or probabilities, YOLO channels-first class scores, rows with objectness, and already-postprocessed xyxy/score/class outputs. RGB/BGR, stretch/letterbox, dimensions, normalization, confidence and NMS are configurable. No output-layout guessing.
- A contract fingerprint binds checkpoint bytes, class mapping and all preprocessing/postprocessing settings; runtime version and input/output digests are additionally recorded. Historical parity requires source hash and contract fingerprint, verified reference origin, plus complete final output comparison. A locally generated baseline cannot certify its own historical parity.
- Persist model-input preview, input/output NPZ and JSON evidence under the case workspace. Changed settings create new identities. Runs refer to a specific source/item/context digest; mismatched context cannot be used for new experiments without rerun.
- Controlled occlusion and brightness experiments keep the source image geometry, selected class and target ROI fixed. Show original/changed inputs and pre-/post-NMS target matches. Compare target occlusion with equal-size controls; unmatched/hidden targets are inconclusive, never a numeric zero improvement.
- Paired evaluation imports a labeled JSON manifest, hashes all images and labels, rejects duplicate/content overlap and missing verdicts, separates case-regression from declared independent holdout, reports normal-image false-alarm rate and defect-image miss rate, and additionally detection precision/recall when complete boxes exist. Explicit user-set gates cannot be passed by missing class support or increased misses beyond the agreed tolerance.
- Training export copies only reviewed complete examples into a ZIP. Full-frame normal reviews are necessary for negatives; any contradictory defect review blocks them. Positive boxes require explicit human class mapping and explicit full-annotation confirmation. Partial ROI normals and pending/disputed items never become whole-image negatives. Export is a handoff to external training, then the resulting model can be compared in the same workspace.

## Risks / Trade-offs

- [No production checkpoint supplied] → Executable ONNX fixtures verify adapter math and UI workflow; clearly distinguish these tests from the user's actual production parity.
- [Unknown deployment recipe] → Allow inference with an explicit chosen recipe; parity remains unavailable until historical provenance is supplied.
- [Post-NMS-only output] → State raw candidates unavailable; do not apply NMS twice or interpret missing boxes as raw score zero.
- [Perturbation destroys object] → Display modified inputs and invalid/inconclusive statuses; record user scope judgement and never auto-promote cause status.
- [Selection bias] → Hash-based case/candidate overlap check plus explicit independent-manifest declaration and separate case regression reporting; provenance limitations remain in reports.

## Migration Plan

Add new optional fields lazily to existing JSON cases, no destructive migration. Add a runtime dependency file and test-fixture dependency. Existing review-only manifests stay compatible. Update the first-version report wording to describe current external evidence accurately.
