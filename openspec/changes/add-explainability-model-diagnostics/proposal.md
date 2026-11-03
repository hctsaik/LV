## Why

The current inline explainability panel can compute and render attribution maps, but it does not consistently separate computability, classifier reliability, attribution faithfulness, and defect localization. This has caused high-resolution spatial probes and visually concentrated heatmaps to be interpreted as proof that the classifier learned or localized `crack`, even when sample-level OOF performance and `crack` recall are weak.

## What Changes

- Add a top-level **Model Diagnostics** workspace with a reference-only handoff from Visualize and, in a later phase, the Anomaly inspector.
- Show classifier reliability, current prediction, held-out OOF prediction, target/reference margin, calibration status, and validation limitations before rendering attribution.
- Make canonical signed Grad-CAM the primary classification-attribution contract and preserve blue-white-red signed evidence without presenting it as segmentation.
- Add selected-sample perturbation smoke tests, followed by batch faithfulness, mask-localization, and shortcut diagnostics in later phases.
- Add grouped validation and repeated learning curves to distinguish data-limited behavior from feature, label, scope, or classifier limitations.
- Separate experimental 336/448 spatial probes from true 224/336/448 models that have resolution-specific embeddings, heads, OOF evaluation, and provenance.
- Add deterministic context, run, render, cache, stale, and artifact contracts so results cannot be silently reused with incompatible data or settings.
- Add evidence-gated status language and report/export requirements that prohibit unsupported claims such as “the model focused on the crack.”
- Replace separate head-build, research-confirmation, and heatmap actions with one stateful primary analysis action; resolution selection configures the task without starting computation, and a missing exact head is prepared within the same foreground task before attribution.
- **BREAKING**: After compatibility migration and replacement test coverage are complete, resolution, attribution-method, and target-positive selectors move out of the inline primary flow into the diagnostics workspace or advanced research controls.

## Capabilities

### New Capabilities

- `explainability-diagnostics-workspace`: Context handoff, workspace information architecture, weak-head warnings, a single primary analysis action, state lifecycle, progressive disclosure, and report entry points.
- `signed-attribution-evidence`: Canonical signed Grad-CAM semantics, experimental token-wise attribution naming, target/reference handling, render invariants, provenance, and single-sample smoke evidence.
- `model-reliability-diagnostics`: Sample-level and grouped validation, per-class metrics, calibration status, learning curves, leakage controls, and conservative evidence states.
- `multi-resolution-model-evaluation`: Explicit separation of spatial probes from exact resolution-specific embeddings and heads, comparable evaluation, recipe-aware cache identity, and resolution selection.
- `localization-faithfulness-diagnostics`: Mask mapping, batch localization metrics, perturbation faithfulness, stability, shortcut probes, baselines, and escalation to Normal Bank, MIL, or segmentation.

### Modified Capabilities

None. The repository has no existing OpenSpec capability specifications; current behavior is documented outside OpenSpec and is captured here as new capability contracts.

## Impact

- UI and orchestration: `scripts/app.py`, `scripts/explain_ui.py`, and a new diagnostics page.
- Explanation engine and rendering: `scripts/gradcam_explain.py`, `scripts/dino_explain.py`, `scripts/explanation_render.py`, and new framework-free contracts/engine modules.
- Classifier validation and provenance: `scripts/dino_head.py`, `scripts/head_adapter.py`, manifests, embedding recipes, and future head registries.
- State and storage: Streamlit session namespaces, `.lv_cache/explainability`, content digests, recipe-aware embedding namespaces, stale rules, and explicit artifact persistence.
- Validation: focused unit tests, engine integration tests, migrated inline tests, Playwright E2E, fixed capsule evidence reports, and no-dataset-write regressions.
- Documentation: `docs/explainability-model-diagnostics-workspace-design.md` remains the detailed research record, `docs/model-diagnostics-ui-ux-redesign.md` defines the simplified interaction proposal, and this OpenSpec change is the normative implementation contract.
