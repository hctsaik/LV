# YOLO investigation validation — 2026-09-06

These checks use random-initialized **real** YOLO architectures and generated images. They validate adapter execution, geometry, evidence identity and the operator workflow; they do not establish production accuracy, root cause, or historical deployment equivalence.

## Adapter scope

- Original anchor/objectness YOLOv5 Detect: `yolov5==7.0.14`.
- YOLOv8 and YOLO11 axis-aligned Detect: `ultralytics==8.3.203`.
- Trusted local PyTorch checkpoints only; no hub downloads or unpinned loader execution. Segmentation, OBB, pose, end-to-end/custom heads and unsupported architectures produce explicit unsupported results.
- ONNX is the deployment inference backend, not a gradient backend. An ONNX/PyTorch pair must match the selected case before attribution. A matching pair does not prove that the historical machine decision was reproduced.

Grad-CAM targets one native detection candidate and an explicit scalar. YOLOv5 confidence uses objectness × class score; YOLOv8/11 confidence uses the class score. Positive and signed maps have distinct legends. Full-frame preprocessing and padding removal reuse the deployment geometry. Zero gradients do not produce a fabricated hotspot. Fixed-candidate perturbation results and rematched final detections are reported separately.

## Actual Chromium stories

Each story passed in a focused run. There was no single final all-three run. The paired-only run completed with exit code 0: **1 passed, 2 deselected in 515.30 seconds**. Its browser/trace teardown was slow; the already-finished story's owned Streamlit server was explicitly stopped, and pytest then exited normally. Earlier browser runs exposed and led to fixes for stale case selection, inspection history selection and profile/run selection; test synchronization was also strengthened around file uploads and nested expanders.

| Story | Observed passing actions |
| --- | --- |
| PT upload, CAM and reuse | Upload checkpoint and image; inspect; save model settings; trial inference; create case without uploading again; select model box; compute signed CAM; change opacity without another job; save evidence; download ZIP containing maps, images, tensors and case evidence; create a second case and automatically select it; reuse the model without another inspection; restart the service and retain the saved compute identity. |
| ONNX without PT | Upload ONNX; inspect and save settings; trial inference; create case and select a box; show the corresponding-PyTorch requirement and contrast-experiment action; expose no Grad-CAM execution button. This story did not execute the contrast experiment itself. |
| ONNX with corresponding PT | Export the same real YOLOv8 checkpoint to ONNX; upload and trial it; create case; follow the missing-PT action; upload the corresponding checkpoint; automatically display that inspection; save profile revision 2; explicitly choose deployment revision 2; infer and explain its selected box; save `per_case_matched` evidence while historical parity remains unavailable and revision-1 results remain unchanged; reject a corrupt checkpoint without adding a profile or losing the saved evidence. |

Browser source: `tests/e2e/test_yolo_investigation_e2e.py`. Dedicated temporary roots keep live case assets outside concurrent pytest retention cleanup. CPU thread environment variables were set to 1 for the final focused runs.

Screenshots: [PT story](yolo-case-gradcam-browser.png), [paired story with bounded source preview](yolo-case-paired-browser.png). The PT screenshot predates the source-preview sizing improvement. These are viewport captures, not an assertion that all content fits on one screen.

## Backend and bridge checks

- Initial adapter gate: 15 tests passed, including real YOLOv5/v8/11 forward/backpropagation and exported ONNX numeric correspondence.
- Expanded negative gate: 3 passed, 6 deliberately non-applicable family cases skipped; covers original YOLOv5 checkpoint namespace, zero-gradient handling, unsupported heads and invalid settings. A prior zero-gradient fixture that incorrectly zeroed box decoding as well as classification was corrected before this focused pass.
- Latest lightweight contract gate: 3 passed; trusted-load boundary, V2 identity/deep-copy behavior, letterbox removal and mask geometry.
- Latest non-PyTorch bridge gate: 8 passed; source-first ONNX inference, empty detections, promotion/attachment, actionable missing-PT behavior, target changes, source corruption, reordered/ambiguous cross-backend matching and nonempty input/output digest requirements.
- The actual YOLOv8 bridge test passed earlier; the browser stories subsequently exercised real inference, attribution and durable publication again. The complete expanded backend suite was not rerun as one final combined gate.
- A published CAM's on-disk `attribution.json` was checked after job-directory promotion: `artifact_dir` is relative (`.`), with no stale `.pending` path.

## Acceptance boundaries

U02 has actual browser coverage. U03 has ONNX intake/inference and missing-gradient coverage, but contrast execution belongs to a separate gate. U04 has revision selection and model reuse coverage, not two stations with independently changed preprocessing. U06 correction/withdrawal and U08 zero-map behavior have backend/bridge coverage, not this browser suite. U13 covers completed-result service restart, not active-job cancellation. U14 concurrent edits/archive is not covered here. U15 covers corrupt-checkpoint recovery, not every missing-file or out-of-memory scenario.

## CAM-to-handoff lineage audit

The final provenance audit found that a case-wide session CAM reference could follow a different search region, and that training export's row whitelist discarded candidate evidence IDs. Both were corrected. Comparison execution and publication now validate the chosen query identity, the original CAM target identity, and the exact derived-region evidence link. A changed or withdrawn original model target cannot be silently attached to a later comparison.

`fp_evidence_lineage.region_provenance` preserves only the included regions' source/ROI/review snapshots and the relevant evidence → search → CAM records, without copying private model paths or unrelated case evidence. Training export includes this closure in `provenance.json`. Corrected regions retain their historical chain; `association_semantics` identifies selection history and the CAM record separately reports whether its original target is still current. Human review remains the only source of training labels.

The focused new-lineage plus existing-training gate passed **8 tests**. This is backend provenance validation, not an additional browser traversal of the complete CAM-to-training workflow.

## Two-session draft protection

A real two-session Streamlit AppTest reproduced a concurrency defect: after tab A saved the report description, tab B's stale submit silently lost its draft and wrote another revision containing A's description. Database optimistic locking alone did not catch it because the rerun loaded the new case revision before rendering the old form submission.

UI-configured CaseStore hooks now retain each tab's observed case snapshot, reject stale writes, and advance the baseline after that tab's own successful save. Report-context fields have stable per-case draft keys. The explicit “開啟最新版本（保留此頁草稿）” action presents the current saved information while retaining the report draft for review before another save. Plain worker/CLI stores keep their existing database locking behavior.

The two-session stale-submit → rejection → explicit refresh → retained draft → merged save → second own save test and the three existing case-page tests passed: **4 passed in 19.49 seconds**. This is AppTest coverage of the report-information form; full U14 browser coverage and other forms remain separately tracked.

A follow-up AppTest uses the actual `render_case_job` fragment and `attach_result`, with a completed synthetic comparison job rather than a model invocation. Its own publication advances the UI baseline and displays the result while preserving the draft; a separate plain-store publication displays a version notice and becomes visible after explicit refresh. Both concurrency tests passed together: **2 passed in 16.95 seconds**. Plain-store and bridge regressions also passed: **29 passed, 1 heavy-model test deselected in 23.11 seconds**.
