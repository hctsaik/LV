# YOLO explanation improvements

OpenSpec: `improve-yolo-explanation-evidence-and-compatibility`.

## Delivered workflow

1. Upload original weights in the model library, select the registered runtime and inspect the actual model/head capabilities. Modern dependencies are isolated from the legacy environment.
2. Infer the original image and select an actual model detection. CAM retains the native candidate/class identity, including modern one-to-one outputs whose final row order can change.
3. Compare geometry-matched controls across constant and blurred fills. Missing controls and conflicting directions are visible. A smaller masking-fraction option supports cases whose original mask cannot be translated fairly.
4. Inspect the saved modified images and record a separate human preservation review bound to their hashes. Summaries distinguish class-score observations from localization, postprocessing and machine-alarm questions.
5. Save an edited reply and export cited HTML. Automatic summaries stay separate from engineer-authored text; detailed per-variant values remain available without overwhelming the short reply.
6. Assign reviewed cohort metadata and compare unique images and declared groups, with separate model/head/scalar/preprocessing/protocol partitions. Submission revalidates truth and ROI snapshots. Native-target deduplication, image averaging and equal group weighting prevent repeated boxes from inflating descriptive contrasts.
7. Record matched workflow effort with independent quality review. Active time, waiting and setup are separate; faster results with worse quality remain visible. Withdrawn or incomplete measurements are retained without contributing to benefit estimates.

## Validation records

- [Controlled evidence, cohort, effort and DINO compatibility](validation/explanation-evidence-validation.md).
- [Actual browser review and reply workflow](validation/controlled-explanation-validation.md).
- [Isolated runtimes, worker dispatch and modern-model execution](validation/yolo-modern-runtime-validation.md).
- [Modern and legacy YOLO numerical contracts](validation/yolo-modern-attribution-validation.md).

All 15 OpenSpec tasks are complete. Modern and legacy numerical routes, both YOLO26 head browser workflows, the saved-evidence reply browser workflow, mixed-runtime evaluation and focused regression gates passed as documented in the linked records. Tests use generated architectural fixtures and explicit case fixtures. They do not establish production false-positive reduction, semantic root causes, or measured time savings for untested engineers.

## Multi-agent review findings resolved

- Final detection rows and native candidates must remain distinct when modern Top-K ordering changes.
- Missing an entire requested perturbation fraction must not be classified as supporting evidence.
- Model, head, scalar, feature layer, preprocessing and perturbation protocol must not be mixed into one evidence pool; candidate offsets must not fragment an otherwise identical cohort recipe.
- Stored cohort confirmation becomes inapplicable after human-truth or ROI changes; submission rechecks the latest evidence and preserves historical records.
- Edited replies and hash-bound preservation confirmation must survive reruns without being silently overwritten or applied to changed images.
- Effort measurements require pairing and independent quality review; cumulative case time cannot be silently substituted for one workflow's active time.
- Visual review found an obsolete save-first prompt after successful CAM publication. Saving now immediately reloads the case and reveals its evidence summary; a focused AppTest verifies that subsequent reruns do not publish twice.

Existing DINO contracts and original YOLO workflows remain part of the regression gates. Legacy CAM hashes are preserved; modern native-target contracts and perturbation analysis identities are separately versioned.
