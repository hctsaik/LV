## ADDED Requirements

### Requirement: Canonical signed Grad-CAM contract
The primary classification-attribution method SHALL compute channel-weighted Grad-CAM for a fixed target-minus-reference logit margin without ReLU so positive and negative evidence remain available.

#### Scenario: Signed evidence is rendered
- **WHEN** canonical signed Grad-CAM returns negative, near-zero, and positive patch scores
- **THEN** the display maps them respectively to blue, white, and red using a symmetric signed scale

#### Scenario: Multiclass negative evidence
- **WHEN** a patch lowers the target-minus-reference margin in a multiclass head
- **THEN** the legend describes it as against the target or lowering the margin and does not assert that the patch exclusively supports the reference class

### Requirement: Experimental token-wise attribution naming
Token gradient-times-activation SHALL remain a separately named experimental attribution method and MUST NOT be labeled standard Grad-CAM or “recommended focus.”

#### Scenario: Compare attribution aggregations
- **WHEN** an advanced user compares canonical and token-wise attribution
- **THEN** image, target scalar, target, reference, hook layer, preprocessing, and normalization are held constant and each method name is recorded

### Requirement: Actual and counterfactual targets
The default attribution SHALL explain the current top-1 class against its strongest eligible runner-up; a manually selected non-top-1 target MUST be labeled counterfactual.

#### Scenario: Explain actual decision
- **WHEN** the user accepts the default target
- **THEN** the result records the current top-1 target, runner-up reference, and fixed target-minus-reference margin

#### Scenario: Explain manually selected crack
- **WHEN** the classifier did not predict `crack` and the user selects `crack`
- **THEN** the workspace states that the map is a counterfactual hypothesis and not an explanation of an actual crack prediction

### Requirement: Raw and display attribution separation
The engine MUST preserve raw signed patch scores independently from display normalization and rendering controls.

#### Scenario: Change display settings
- **WHEN** opacity, palette, grid, or display size changes
- **THEN** raw patch scores and the compute key remain unchanged

#### Scenario: Flat attribution map
- **WHEN** the raw score scale is below the configured numerical threshold
- **THEN** the system reports no effective attribution rather than stretching noise to the full color range

### Requirement: Reproducible attribution provenance
Every attribution result SHALL record model and checkpoint identity, head digest and provenance, scope, pooling, preprocessing, input resolution, native patch grid, target, reference, target scalar, method, hook layer, and engine schema.

#### Scenario: Download P1 evidence
- **WHEN** a user downloads P1 result data
- **THEN** metadata plus a raw JSON or NPZ patch array can reconstruct the signed display without rerunning the model

### Requirement: Selected-sample faithfulness smoke test
P1 SHALL compare fixed-target Top-K patch perturbation with recorded repeated random controls and SHALL report the result only as single-sample smoke evidence.

#### Scenario: Top-K deletion exceeds random controls
- **WHEN** Top-K deletion produces a larger target-margin drop than the recorded random control policy
- **THEN** the attribution badge may be `single_sample_smoke_supported` while classifier and localization states remain unchanged

#### Scenario: Perturbation is unsupported
- **WHEN** Top-K deletion does not outperform random controls or the test cannot run
- **THEN** the attribution is reported as unsupported, inconclusive, or unavailable and no trusted explanation claim is made

### Requirement: Classification attribution is not segmentation
The UI and exports MUST state that signed Grad-CAM explains a classifier margin and is not a crack mask, anomaly map, or pixel segmentation.

#### Scenario: Heatmap overlaps a visible defect
- **WHEN** a heatmap visually overlaps a defect but no validated localization benchmark exists
- **THEN** the system does not claim that the model found or segmented the defect

