## ADDED Requirements

### Requirement: Explicit external runtime contract
The system SHALL run supported single-input local ONNX models only under an explicit preprocessing/output/postprocessing contract and record model, contract and tensor identities.

#### Scenario: Wrong output layout
- **WHEN** tensor dimensions do not match the configured output layout and class mapping
- **THEN** inference evidence is not published as a valid prediction and the user receives an actionable error.

### Requirement: Historical parity is distinct from new inference
The system SHALL compare complete final labels, boxes, scores and alarm decisions under declared tolerances, and SHALL require original-image and historical-contract provenance to mark parity matched.

#### Scenario: Same score different boxes
- **WHEN** a replay has similar scores but different final boxes
- **THEN** the replay is mismatched, not reproduced.

#### Scenario: No historical provenance
- **WHEN** only a screenshot or current local inference is available
- **THEN** local inference remains usable but historical parity is unavailable.

### Requirement: Target-specific controlled experiments
The system SHALL retain original and modified inputs, fixed target class and ROI, control interventions and matching status, and publish only sensitivity evidence.

#### Scenario: Detection disappears
- **WHEN** the original target cannot be matched after a perturbation
- **THEN** target-score change is unavailable and the result does not become a root-cause or model-improvement claim.

### Requirement: Paired independent evaluation
The system SHALL evaluate baseline and candidate contracts on the same immutable labeled manifest, distinguish case regression from independent holdout, and report false alarms and misses with support counts and explicit acceptance gates.

#### Scenario: Fewer case alarms but more independent misses
- **WHEN** candidate miss-rate increase exceeds the configured tolerance
- **THEN** the improvement gate fails regardless of case-regression success.

#### Scenario: Holdout overlaps reviewed cases
- **WHEN** a declared independent manifest contains case or candidate source content
- **THEN** independent evaluation is rejected and the user can evaluate that data only as case regression.

### Requirement: Safe reviewed training handoff
The system SHALL export a separate copied training package only for explicitly complete, individually reviewed annotations with human class mapping, retaining provenance and excluding incomplete or conflicting images.

#### Scenario: Mixed image
- **WHEN** one ROI is normal and another contains a true defect
- **THEN** the image cannot be exported as a normal negative; confirmed positive boxes are preserved when full annotation is explicitly confirmed.
