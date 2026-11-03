## ADDED Requirements

### Requirement: Reference-only diagnostic context handoff
The system SHALL open the Model Diagnostics workspace from Visualize using a versioned context that references, rather than copies, the dataset, selected record, embeddings, classifier head, mask, and Normal Bank.

#### Scenario: Open selected Visualize image
- **WHEN** a user opens Model Diagnostics from a selected Visualize record
- **THEN** the workspace resolves the same dataset fingerprint, image SHA-256, label, split, model, head, scope, and crop recipe without requiring the dataset to be selected again

#### Scenario: Stable record identity after reordering
- **WHEN** records are reordered without changing the selected source content
- **THEN** the selected record is resolved by content identity and object geometry rather than array index

### Requirement: Context compatibility and stale protection
The system MUST validate source content, dataset generation, model, checkpoint, head, scope, pooling, preprocessing, resolution, target, reference, and engine schema before presenting a result as current.

#### Scenario: Dataset changes after a diagnostic run
- **WHEN** the dataset generation or selected source SHA-256 differs from the run context
- **THEN** the previous result is marked stale or historical and MUST NOT overwrite the current context

#### Scenario: Head recipe is incompatible
- **WHEN** the selected head provenance does not match the model, scope, pooling, or preprocessing recipe
- **THEN** Grad-CAM is unavailable and the workspace lists the incompatible fields instead of silently using the head

### Requirement: Reliability-first weak-head warning
The system SHALL present classifier reliability and prediction semantics before
offering classification attribution. A weak head SHALL keep a persistent
research-only limitation visible, but MUST NOT require a redundant acknowledgement
click before the user can inspect the evidence.

#### Scenario: Weak target-class recall
- **WHEN** the applicable audit indicates insufficient classifier performance or weak target-class recall
- **THEN** the page keeps the weak-head warning visible and exposes the normal attribution action directly; the result remains research evidence and cannot be promoted to a trusted or localized claim

#### Scenario: Warning persists after rendering
- **WHEN** a user runs attribution for a weak head
- **THEN** the reliability warning remains visible after the heatmap is rendered and the result is not promoted to a trusted state

### Requirement: Inline and workspace responsibilities
The inline panel and Model Diagnostics workspace SHALL use the same context, engine, render, and reliability contracts while exposing different levels of detail.

#### Scenario: Inline limitation stays visible
- **WHEN** a weak head is selected in the inline panel
- **THEN** the inline panel keeps the research-only limitation visible without adding a second acknowledgement step

#### Scenario: Advanced controls migrate safely
- **WHEN** replacement workspace and engine tests cover resolution, method, and display-mode behavior
- **THEN** those controls may be removed from the inline primary flow without removing the underlying research capability

#### Scenario: Visualize uses a handoff-first summary
- **WHEN** a selected Visualize record can be analyzed in Model Diagnostics
- **THEN** the inline surface shows the selected-image summary and one diagnostics handoff action instead of duplicating head training, resolution, target, method, and full attribution controls

### Requirement: Single primary analysis action
For an actionable selected-image context, the workspace SHALL expose at most one
primary analysis action. The action MUST represent the user's complete intent to
analyze the selected image at the selected resolution; head preparation is an
internal stage of the same task, not a second required user action.

#### Scenario: Matching head does not exist
- **WHEN** the selected resolution has no compatible exact head and the user activates the primary analysis action
- **THEN** the action states that it will build the selected-resolution model and analyze the image, then performs embeddings, head training/audit, and attribution as one foreground task without asking for another click

#### Scenario: Matching head already exists
- **WHEN** the selected resolution has a compatible exact head and the user activates the primary analysis action
- **THEN** the workspace reuses the head, skips completed preparation stages, and produces the attribution without showing a separate model-preparation action

#### Scenario: Weak head remains inspectable
- **WHEN** the prepared head has insufficient reliability evidence
- **THEN** the same primary analysis action remains available, the limitation stays visible, and no research acknowledgement action is inserted before attribution

#### Scenario: Analysis fails
- **WHEN** any stage of the single analysis task fails
- **THEN** the workspace retains reusable completed sub-results, reports the failed stage, and exposes one retry analysis action rather than separate recovery buttons for head and heatmap

### Requirement: Explicit computation lifecycle
The workspace MUST NOT automatically start expensive inference, OOF,
learning-curve, multi-resolution, localization, or shortcut jobs when a page,
tab, image, resolution selector, target selector, method selector, or render
control changes. Only activation of the primary analysis action starts the
selected-image computation lifecycle.

#### Scenario: Select a different resolution
- **WHEN** a user changes the analysis resolution among 224, 336, and 448
- **THEN** the workspace updates the estimated work and primary-action label, marks any incompatible current result as previous or pending refresh, and does not start embeddings, head training, or attribution

#### Scenario: Activate analysis for a non-baseline resolution
- **WHEN** a user selects 336 or 448 and then activates the primary analysis action
- **THEN** the workspace runs the exact-resolution preparation and attribution stages in one foreground task and shows phase-specific progress

#### Scenario: Adjust opacity
- **WHEN** a user changes opacity, palette, grid visibility, or overlay size
- **THEN** the system rerenders the existing raw result without recomputing the model

#### Scenario: Late job completes for an old context
- **WHEN** a job completes after the user has selected a different context or run key
- **THEN** the result is retained only as stale or historical and is not published as the current result

### Requirement: Orthogonal evidence states
The workspace SHALL report head validation, attribution faithfulness, and localization as separate evidence dimensions.

#### Scenario: Faithfulness smoke passes for a weak head
- **WHEN** selected-sample perturbation supports the attribution but classifier validation is insufficient
- **THEN** the workspace reports both facts independently and does not summarize the model as reliable or localized

### Requirement: Anomaly inspector expansion
The system SHALL support Anomaly inspector handoff only after the Visualize classifier flow is complete, with a capability-aware empty state when no classifier head exists.

#### Scenario: Inspector context has no head
- **WHEN** an Anomaly inspector context contains a compatible Normal Bank but no compatible classifier head
- **THEN** the workspace may show Attention, existing anomaly evidence, and Normal Bank provenance while marking Grad-CAM unavailable

### Requirement: No implicit source or cache mutation
The workspace SHALL remain session-only by default and MUST NOT modify source datasets, existing embedding caches, or anomaly caches unless a user explicitly saves a supported diagnostic artifact.

#### Scenario: Run P1 analysis
- **WHEN** a user runs a selected-sample diagnostic without choosing save
- **THEN** no file is written to the source dataset or persistent explanation artifact namespace

### Requirement: Guided selected-sample workflow
The workspace SHALL present one user-facing path ordered as selected image,
resolution selection, one primary analysis action, result, and interpretation.
Matching model preparation MAY occur inside the analysis task but MUST NOT
require a separate user-facing step. Technical identifiers and research-only
controls MUST remain secondary to the primary path.

#### Scenario: Enter diagnostics from a selected image
- **WHEN** a user selects an image in Visualize and opens Model Diagnostics
- **THEN** the page identifies the image first, explains whether the classifier is ready, and shows the next required action without requiring the user to understand context handoff terminology

#### Scenario: Prepare a missing classifier head
- **WHEN** the selected context has no session classifier head
- **THEN** the page shows the 224/336/448 resolution selector immediately, keeps 224 as the default, and exposes one primary action whose label explains that it will build the selected-resolution model and analyze the image

#### Scenario: Configure a non-baseline resolution
- **WHEN** a user selects 336 or 448 before any classifier head exists
- **THEN** the selection itself starts no computation and the primary action changes to describe the matching exact-resolution build-and-analyze task without requiring a 224 head first

#### Scenario: Analyze a finer resolution
- **WHEN** the user activates the primary action with 336 or 448 selected and no matching head exists
- **THEN** the page builds the exact-resolution classifier model, shows embeddings/head/attribution progress, automatically continues to the heatmap, and never offers a second required heatmap action

#### Scenario: Read the result
- **WHEN** signed Grad-CAM completes
- **THEN** the page presents original image, model input, signed overlay, blue/white/red legend, and reliability limitations in that order; technical provenance remains available under a secondary details section
