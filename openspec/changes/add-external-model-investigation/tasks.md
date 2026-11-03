## 1. Runtime and replay

- [x] 1.1 Implement strict ONNX runtime contracts, preprocessing, classification/YOLO decoding, NMS and artifact provenance with executable-model tests.
- [x] 1.2 Implement complete-output historical parity, stale-context protection and persisted external evidence; verify missing provenance and changed boxes cannot pass.

## 2. Experiments and improvement

- [x] 2.1 Implement selected-target occlusion/brightness experiments with controls, input previews and unmatched-target limitations.
- [x] 2.2 Implement immutable paired evaluation, independent-set overlap guards, support counts and false-alarm/miss acceptance gates.
- [x] 2.3 Implement reviewed complete-image training ZIP export with explicit class mapping, preserved true boxes and no partial-negative conversion.

## 3. Product delivery

- [x] 3.1 Integrate runtime setup, replay, experiments, evaluation and training handoff into case UI and reports.
- [x] 3.2 Pass focused/regression tests and actual-browser ONNX end-to-end tests, including persisted evidence and downloads.
- [x] 3.3 Update dependency/install instructions, model/evaluation schemas, workflow guide and validation report; validate OpenSpec.
