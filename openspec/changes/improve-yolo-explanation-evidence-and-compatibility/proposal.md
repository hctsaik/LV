## Why

False-positive investigations require evidence about the actual deployed model, not merely plausible heatmaps. The existing fixed Ultralytics runtime excludes newer model families, and pixel-scattered controls do not fairly test contiguous attribution regions. Engineers also need explicit uncertainty, independent sample counts, and measured review effort.

## What Changes

- Add isolated, pinned legacy and modern YOLO runtimes with inspected task, head, output path and gradient capabilities; validate YOLO12 and YOLO26 without changing the legacy runtime.
- Preserve final detection to native candidate correspondence through CAM, replay, perturbation, and exported-model comparison.
- Generate editable, cited case summaries distinguishing class-score evidence from localization, postprocessing, and machine-alarm questions.
- Add geometry-matched perturbation controls and fill-method stability, with explicit unavailable/inconclusive outcomes.
- Add independent-image/group accounting and matched effort measurements with quality checks.

## Capabilities

### New Capabilities
- `versioned-yolo-runtimes`: inspected and isolated model execution with faithful target identity.
- `case-explanation-assessment`: question-aware evidence summaries and human confirmation.
- `controlled-attribution-validation`: fair perturbation, sample independence and effort evaluation.

### Modified Capabilities

None. Existing unarchived investigation changes remain compatible.

## Impact

YOLO attribution, model registration, worker dispatch, case evidence/replies, investigation UI, tests and documentation. Modern dependencies live in a separate environment. Existing DINO behavior and legacy result identities remain readable. Actual defect reduction and effort savings require user data and are not inferred from software tests.
