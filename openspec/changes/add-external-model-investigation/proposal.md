## Why

The user requires end-to-end delivery beyond the initial case workspace. Engineers need to run their external model, test suspected visual influences, and check whether a candidate model reduces false alarms without losing real defects.

## What Changes

- Add local ONNX classification and common YOLO detection adapters with explicit preprocessing, output layout, thresholds, NMS, label mapping and content fingerprints.
- Record inference inputs/outputs, compare historical predictions under a verified contract, and persist selected-region controlled perturbation experiments as external-model evidence.
- Add paired current/candidate model evaluation on explicitly labeled, fixed independent datasets, with normal false alarms and defect misses reported separately from case regression.
- Export reviewed, explicitly complete training examples with ROI/image unit safeguards; never auto-label unreviewed or conflicting regions.
- Extend the case UI, evidence report and tests to cover all actions. No external service calls or automatic production deployment.

## Capabilities

### New Capabilities
- `external-model-case-investigation`: Runtime contracts, replay parity, perturbation evidence, paired evaluation and safe reviewed-data export.

### Modified Capabilities
None. Additive extension to the unarchived first-version case change; its evidence semantics remain intact.

## Impact

New framework-independent runtime/evaluation modules and a case UI extension. Optional `onnxruntime` runtime dependency and `onnx` for executable test fixtures. Existing case snapshots remain readable; new collections initialize lazily. Existing LV DINO analysis remains explicitly auxiliary.
