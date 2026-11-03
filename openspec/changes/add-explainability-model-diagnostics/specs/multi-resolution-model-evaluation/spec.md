## ADDED Requirements

### Requirement: Spatial probes are explicitly experimental
Running a head trained at one resolution on another input resolution SHALL be labeled a spatial probe and MUST NOT be represented as an exact head for the probe resolution.

#### Scenario: Use a 224 head at 448 input
- **WHEN** an advanced user explicitly runs a 224-trained head with 448 input
- **THEN** the result shows head resolution 224, analysis resolution 448, native grid 32×32, resolution mismatch, and research-only status

#### Scenario: Probe appears in comparison
- **WHEN** spatial probe results exist
- **THEN** they are excluded from the validated resolution leaderboard and cannot generate a resolution recommendation

### Requirement: Exact resolution-specific models
An exact 224, 336, or 448 model SHALL have resolution-specific full-dataset embeddings, preprocessing recipe, classifier head, provenance, and OOF or audit results.

#### Scenario: Run a non-baseline resolution with no exact head
- **WHEN** a user selects 336 or 448 in the diagnostics workspace, no matching session head exists, and the user activates the primary analysis action
- **THEN** the application performs the foreground full-dataset embedding pass, trains/audits the corresponding frozen-backbone classifier head, and continues to Grad-CAM within the same task; it MUST NOT silently reuse the 224 head or require a second click

#### Scenario: Select a non-baseline resolution without running
- **WHEN** a user changes the diagnostics resolution to 336 or 448 but has not activated the primary analysis action
- **THEN** the application performs no embedding pass, head training, audit, or attribution and only updates readiness, cost, and action-label information

#### Scenario: Select an exact 336 head
- **WHEN** a user runs the primary analysis action with exact resolution 336 selected
- **THEN** the system uses only a head whose recorded extraction recipe has target resolution 336 and does not silently fall back to a 224 head

#### Scenario: Exact head is missing
- **WHEN** no compatible head exists for a requested exact resolution
- **THEN** the system offers one explicit build-and-analyze action or a separately labeled advanced probe action instead of producing an exact-model result

### Requirement: Paired resolution benchmark
Validated resolution comparisons SHALL use the same dataset snapshot, group/fold policy, audit records, target/reference policy, and seed policy.

#### Scenario: Compare 224, 336, and 448
- **WHEN** all three exact heads have completed validation
- **THEN** the dataset-level table reports classification, per-class recall, calibration, prediction flips, faithfulness, localization when available, latency, memory, and cache cost before showing sample heatmaps

#### Scenario: 336 outperforms 448
- **WHEN** 336 has stronger validated evidence or a better performance-cost trade-off than 448
- **THEN** the system does not recommend 448 merely because it has a denser patch grid

### Requirement: Recipe-aware embedding and head identity
Embedding caches, manifest references, and head registries SHALL include model, checkpoint, resolution, pooling, scope, crop, aspect policy, normalization, and schema identity.

#### Scenario: Two resolutions use the same model name
- **WHEN** 224 and 448 embeddings are generated for the same backbone
- **THEN** they occupy distinct recipe-aware namespaces and cannot overwrite or reuse each other's rows

#### Scenario: Legacy cache lacks recipe identity
- **WHEN** an older cache cannot prove its full extraction recipe
- **THEN** it is marked legacy or stale and is not silently treated as an exact-resolution artifact

### Requirement: Comparable sample rendering
Exact-head sample comparisons SHALL use a documented signed scale, geometry policy, opacity, and decision-comparison mode.

#### Scenario: Exact heads predict different classes
- **WHEN** resolution-specific heads have different top-1 or runner-up classes
- **THEN** the UI separates “explain each model's actual decision” from “fixed target/reference research comparison” and labels each mode
