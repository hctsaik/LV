## Context

The approved three-round product discussion prioritizes external-model suspected false-positive investigation. Existing similarity and round modules are reusable, but missing labels are not proof of normality and the current training export accepts positive boxes. Existing dirty Model Diagnostics changes must be preserved.

## Goals / Non-Goals

**Goals:** A complete local workflow for creating/resuming cases, region-level review, balanced comparisons, cited explanations, review-only AL handoff and effort records.

**Non-Goals:** External inference replay, automatically proven causes, automatic group labels, production deployment, model retraining, or training-label conversion. These are later phases, not incomplete tasks in this MVP.

## Decisions

- Use independent `fp_cases.py` (SQLite snapshots and domain validation), `fp_case_search.py` (explicit local DINO retrieval), and `fp_case_page.py` (Streamlit). Add only a route to app.py. SQLite transactions and revision checks protect resumed cases against lost writes; immutable revision snapshots preserve prior evidence.
- Store case-owned image copies by content hash under `LV_CASES_DIR` or `LV_OUTPUT_DIR/fp_cases`, outside source datasets. Screenshots retain their identity; added original images are additional sources, not replacements. Source model/threshold may be unknown.
- Disputed items and candidate reviews carry image identity plus normalized xyxy ROI. Verdicts require reviewer/reason. No normal label is inferred from missing annotations, similarities, or a reviewed neighboring ROI.
- Explicitly search user-specified normal and defect reference folders plus reviewed historical cases. Folder membership is a supplied label, not a verified verdict. Reuse local DINO feature extraction and cosine similarity; record model/checkpoint/recipe/search scope and source digests. Query and candidates use the same crop recipe; missing data and changed files remain visible. Computation starts only from the search button.
- Cite immutable evidence records in editable claims. Reports distinguish source predictions, LV auxiliary retrieval, human review, and hypotheses. All first-version reports state external inference was not replayed. Reply status is independent of cause/improvement status.
- Persist review-only manifests with stable candidate identities and per-region verdicts, link existing rounds by case/batch identity, and leave legacy `n_labeled` untouched. Return manifests can update only matching review items after validation; no source annotations are written.
- Effort is explicitly started/paused, with a user-correctable recorded duration. Display review-region counts separately from image/training-ready counts. Never treat page wall time as completed work.

## Risks / Trade-offs

- [Incomplete provenance] → Permit intake, restrict claims, show unknown fields and screenshot limitations.
- [Confirmation bias] → Normal/defect/historical columns, visible source labels, allow empty results; no similarity-as-probability language.
- [Expensive retrieval] → Explicit run, bounded file limit with visible truncation, reusable model object, progress and actionable errors; no automatic head training.
- [Review propagation] → Deduplicate image+ROI; manual individual verdicts, review-only export, no training-ready claims.
- [Cross-tool ledger atomicity] → Stable batch identity and recover existing round by batch parameters on retry; keep review events separate from training-label counts.

## Migration Plan

Additive route and new SQLite workspace. Existing stores/specs are unchanged. Removing the route disables the feature without deleting persisted cases. No dataset migration.

## Open Questions

None blocking MVP. Actual external runtime format and pilot effort reduction target are determined in later phases using real deployment examples and measured baselines.
