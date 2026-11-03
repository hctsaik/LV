# Explanation evidence and compatibility validation

Change: `improve-yolo-explanation-evidence-and-compatibility`.

## Evidence contract

- CAM identifies a fixed candidate/class scalar, not a semantic defect cause. Modern targets also identify runtime and actual head/output path.
- Protocol 2 controls preserve mask geometry and area through bounded translations, without wrapping or clipping. Overlap with the selected region is at most 10%; failure to find the requested controls is explicitly inconclusive.
- Constant and blurred fills share masks. The reported contrast is selected score drop minus mean control score drop; positive values mean relatively lower fixed score, not necessarily a larger absolute change.
- A positive contrast is not proof that the model is correct, the target appearance was preserved, or a machine alarm was reproduced. Human preservation review and historical replay scope remain separate evidence.
- Unique images, declared groups, roles and execution protocols are counted separately. Duplicate boxes/runs do not increase image support. Conflicting image metadata is excluded from cohort support.
- Effort results require unique matched records, completed workflows and an independent quality reviewer. Setup, active time, waiting and error counts remain separate. Fewer minutes with more errors remain visible. Withdrawn measurements retain audit history and are excluded.

## Focused verification

Initial controls/cohort/effort backend gate: **11 passed** (`tests/test_attribution_controls.py`, `tests/test_explanation_validation.py`, dedicated `.pytest-explanation-validation-v2`). Covers geometry, deterministic masks across fills, unchanged outside-mask pixels, infeasible full-extent masks, reversed fill direction, missing controls, repeated images, conflicting labels, unreviewed metadata, duplicate pairs, unmatched pairs and self-review.

Initial effort and case UI gate: **4 passed** (`tests/test_effort_measurement_page.py`, `tests/test_fp_case_page.py`, `.pytest-effort-page-v2`). Actual Streamlit form state persists paired time and quality outcomes across restart, unpaired measurements cannot create savings, and case intake/review navigation remains operational.

Final integrated validation results are recorded below as each gate completes. Real production false-positive reduction and effort savings have not been measured; generated model fixtures validate implementation correctness only.

- Controls, cohort, effort and effort UI after reversible measurement correction: **12 passed**, `.pytest-controls-effort-v3` (38.57 s).
- Independent agent review found two additional negative cases: an entire requested masking fraction missing from a report, and a Boolean accepted as workflow order. Both now have explicit rejection and regression coverage.
- Final controls/cohort/effort backend review gate: **12 passed**, `.pytest-controls-review-v4` (2.56 s), including both new counterexamples.
- DINO compatibility gate: **54 passed**, `.pytest-dino-compat-modern-v1` (69.74 s): `test_explain_contracts.py`, `test_explain_ui.py`, `test_gradcam_explain.py`, `test_dino_head.py`, `test_head_adapter.py`.
- Cohort descriptive contrasts first deduplicate native targets, average within each image, then weight declared groups equally. Repeated target runs cannot dominate the result; conflicting repeated-target values are excluded. Each group mean retains its group ID and image support, and member evidence remains traceable.
- Cohort aggregation and audit mapping: **5 passed**, `.pytest-cohort-contrasts-v2` (0.69 s). Final effort form: **1 passed**, `.pytest-effort-final-ui` (10.35 s); cumulative case time is not silently reused as a per-workflow measurement.
- CAM navigation, concurrent case editing and evidence lineage: **12 passed**, `.pytest-cam-navigation-evidence-v2` (7.74 s).
- Third-round review found stale cohort confirmation after human-truth or ROI changes. Submission now revalidates the latest review evidence, target identity and source hash; failed validation preserves history but excludes the cohort claim. See [controlled workflow validation](controlled-explanation-validation.md) for the two actual form-submission regressions and browser evidence.
- Integrated summary, form, cohort and controls gate after the stale-review fix: **24 passed**, `.pytest-explanation-final-integrated` (15.66 s).
- Final visual-review correction: saving CAM immediately refreshes the case and its summary. **1 focused AppTest passed** (4.33 s), including one-time notice and no duplicate publication. Both real YOLO26 head browser workflows and their artifact records are documented in [runtime validation](yolo-modern-runtime-validation.md).

Final OpenSpec strict validation, Python compilation and `git diff --check` passed. All 15 implementation checklist items are complete.

## Source references

- [Ultralytics 8.4.142 release](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.142): pinned modern runtime and NMS interface transition.
- [Ultralytics end-to-end detection guide](https://docs.ultralytics.com/guides/end2end-detection/): head/export path distinctions; version-specific implementation is tested against the pinned package.
- [YOLO26](https://docs.ultralytics.com/models/yolo26/) and [YOLO12](https://docs.ultralytics.com/models/yolo12/): model families; family names alone do not establish checkpoint compatibility.
- [Grad-CAM](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html): visual localization through gradients.
- [Sanity Checks for Saliency Maps](https://papers.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html): visual plausibility alone is insufficient evidence of explanation faithfulness.
