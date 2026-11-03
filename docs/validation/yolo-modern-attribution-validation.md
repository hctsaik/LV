# Modern YOLO attribution validation — 2026-09-06

These are engineering correctness checks using real, untrained nano architectures and synthetic RGB images. They validate execution contracts, gradients, correspondence, and evidence identities. They do not measure industrial false-positive explanation quality, production accuracy, historical alarm parity, or engineer effort savings.

## Runtime and route matrix

The modern environment uses `ultralytics==8.4.142` and `torch==2.6.0+cpu`. Its separate interpreter, resolved dependency lock and readiness checks are maintained by the runtime registry. The existing `.venv` remains on `ultralytics==8.3.203` with original `yolov5==7.0.14`.

| Runtime / real architecture | Actual route | Normal forward / CAM / hook cleanup | ONNX correspondence | Fixed-target / contract checks |
| --- | --- | --- | --- | --- |
| Modern YOLOv8n | One-to-many + configured LV NMS | Passed | Passed: ordered native output values | Passed |
| Modern YOLO11n | One-to-many + configured LV NMS | Passed | Passed: ordered native output values | Passed |
| Modern YOLO12n | One-to-many + configured LV NMS | Passed | Passed: ordered native output values | Passed |
| Modern YOLO26n | One-to-many + configured LV NMS | Passed | Passed: ordered native output values | Passed |
| Modern YOLO26n | One-to-one, no added NMS | Passed | Passed: complete unique class/box/score correspondence; actual companion guard accepted | Passed: native target retained despite forced final-row class replacement; ambiguous export rejected |
| Legacy original YOLOv5n / YOLOv8n / YOLO11n | Existing routes | Passed | Passed | Passed: V2 identity, cancellation, zero-gradient and existing batch behavior |

One-to-many uses LV's explicit existing preprocessing and postprocessing configuration. This is not blanket equivalence to every option of `YOLO.predict`, arbitrary export pipelines, or machine-side PLC rules. One-to-one preserves the head's actual public postprocess rows and adds LV's configured score threshold and maximum returned detections.

Modern detection rows preserve both the final `output_index` and the `native_candidate_index` plus class and head identity. V3 targets contain the actual runtime release and output/export contract. Native vectors are retained separately in `tensors.npz`. The model forward is unchanged; the adapter reconstructs the head's top-k trace and checks that it matches the actual public output before attribution. Segmentation, OBB, pose, absent/fused-away branches, unsupported backbones, and export-mode PT checkpoints without the required native trace remain unsupported for this CAM adapter.

## Focused run evidence

The modern matrix passed across focused runs, **not one final combined invocation**:

- `.pytest-yolo-modern-numeric-2`: **17 passed, 4 skipped**, then stopped at one-to-one ordered ONNX comparison. The first four routes each passed forward/CAM, export, fixed-target controls and wrong-contract rejection. The one-to-one forward/CAM check also passed.
- `.pytest-yolo-modern-one2one-5`: **5 passed**, then stopped in the new negative test's export wrapper. These five included the final one-to-one positive correspondence gate and actual ONNX-to-PT companion guard, perturbation identity, runtime rejection and native snapshot checks.
- `.pytest-yolo-modern-final-negatives-6`: **4 passed, 28 deselected in 17.53 seconds**, exit 0. This covered the corrected real tied-export rejection, a strengthened snapshot test that explicitly replaces the class occupying the original final row, protocol/execution/scalar/layer/preprocessing summary partitions, and unique-permutation versus duplicate/missing companion cases.
- `.pytest-yolo-legacy-regression-7`: **18 passed, 8 skipped in 61.02 seconds**, exit 0, using `tests/test_yolo_attribution.py` and `tests/test_faithfulness_batch.py`. Skips are deliberately non-applicable family cases. The V2 fixture hash remains `a2c35c10f4be385a0ef20a2ef00f1cdf3ebd8f05fef251da6fd810879e271004`.

The final modern test file contains 24 applicable cases and eight family-specific skips. The route table above states what was exercised across these runs; it does not imply that every historical artifact was regenerated after metadata-only changes.

All heavy tests ran sequentially with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`, using separate repository basetemps. Modern settings used `.runtimes/yolo-modern-8.4.142/settings`; the legacy regression used a pre-created `.pytest-yolo-legacy-settings` directory rather than the user's settings.

## Why the fixture and comparison test changed

Default random YOLO26 weights produced effectively tied confidence values. PyTorch and ONNX Runtime selected or ordered equal-scoring top-k candidates differently. Treating final row numbers as cross-backend identities was therefore an invalid test contract. The test now requires complete unique class/box/score correspondence and separately exercises the real `_match_target` guard. Missing candidate sets or ambiguous matches are rejected; permutation alone is accepted only with a complete unique mapping. Raw tensor tolerances were not widened to hide box mismatches.

The positive one-to-one fixture remains an actual `DetectionModel("yolo26n.yaml")`. Its **test-only weights** use BatchNorm running variance `0.15`, zero final box-convolution weights with distance bias `0.5`, and a `1000` multiplier on final classification-convolution weights. This creates legal, distinguishable boxes and non-degenerate score variation while retaining the real classification gradient path. No production loader changes model weights. The tied negative restores running variance `1`, removes the classification multiplier, and uses distance bias `2` to retain ambiguous clipped boxes.

Two earlier harness issues were corrected: JSON lists were compared canonically with in-memory tuples, and the negative test's export wrapper was explicitly put in evaluation mode so the exporter could not restore its default training mode onto the child model. A trial variance `0.1` saturated scores and invalidated random box outputs; it was discarded. These fixture/harness failures are recorded separately from product adapter defects.

## Reused checkpoints

The browser handoff reuses `.pytest-yolo-modern-one2one-5/yolo26one_to_one0/original.pt`, which retains both heads and is not fused. Its SHA-256 is:

`0cafeba5867c1fb5ca559e87420fb6e486b85f9ef341d98cc35331197ee9f170`

The mixed-runtime evaluation handoff reused `.pytest-yolo-modern-numeric-2/yolov8one_to_many0/original.pt`:

`3c3e2bb6568272821d1231ef3d7be5e989a44a476066b393f000f43c618cefb8`

These private, git-ignored test assets are reusable local evidence, not shipped trained models. Browser assertions and cross-runtime job cleanup are reported separately by their respective gates.

## Controlled evidence and identity

New faithfulness reports use protocol 2: exact-shape translated controls, constant and blurred fills, explicit unavailable placements, and conservative assessment. Fixed native scores remain separate from newly matched final detections. Perturbation reports preserve full protocol, preprocessing, layer and execution context; aggregate groups do not pool different models, heads, scalar expressions, classes, layers, preprocessing or protocols. Old pooled reports remain readable but explicitly unverified.

Existing CAM compute hashes remain unchanged. `validation_id` identifies the target, map, protocol and analysis settings; `analysis_id` binds that validation to the CAM compute identity. These checks establish reproducibility and bounded score sensitivity, not a causal explanation of defects or a measured production benefit.
