## Why

Engineers receive suspected false-positive reports from external deployed models and spend time finding comparisons, recording verdicts, and assembling explanations. LV has retrieval and annotation-loop primitives, but no persistent case connecting the disputed prediction, auxiliary evidence, human review, and follow-up batch.

## What Changes

- Add a Chinese-language false-positive case workspace with image upload/local image intake, optional external prediction metadata, multiple disputed regions, persistent resume, and source provenance.
- Compare normal, defect, and reviewed historical cases using LV embeddings, with explicit source labels and no inference of external-model causes.
- Save per-region human verdicts and cited facts/hypotheses/actions; export a readable Markdown report and machine-readable case evidence.
- Build deduplicated review-only manifests linked to existing AL rounds, without writing training labels or counting reviewed regions as labeled images.
- Record manually controlled active investigation time and review counts; distinguish reply, cause confirmation, and improvement status.

## Capabilities

### New Capabilities
- `false-positive-case-workspace`: Intake, provenance, region verdicts, evidence comparison, report, effort, persistence, and review-only AL handoff.

### Modified Capabilities
None. Existing Model Diagnostics and training-label contracts remain unchanged.

## Impact

New case domain/store, retrieval orchestration, and Streamlit page modules; a small app navigation addition; unit/integration/browser tests; usage documentation. Reuses local DINO extraction, similarity, and round ledger. No new dependency, external service, automatic training, universal runtime adapter, or source-dataset mutation.
