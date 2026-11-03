## ADDED Requirements

### Requirement: Prediction semantics are distinct
The system SHALL distinguish ground-truth label, current full-fit inference, held-out OOF prediction, target class, and reference class.

#### Scenario: Dataset record has OOF evidence
- **WHEN** the selected record has a held-out OOF prediction
- **THEN** the UI shows it separately from the current full-fit prediction and ground-truth label

#### Scenario: External record has no OOF evidence
- **WHEN** the selected image was not part of the OOF audit
- **THEN** the UI reports OOF unavailable and MUST NOT substitute current inference

### Requirement: Sample-level audit limitations
Until a validated group contract exists, sample-stratified OOF SHALL be labeled an audit with cross-group generalization unknown.

#### Scenario: Display sample-level OOF
- **WHEN** sample-stratified OOF metrics are available
- **THEN** the workspace shows split type, sample count, fold count, class support, and leakage limitations

### Requirement: Reliability metrics and calibration status
The system SHALL report balanced accuracy, macro-F1, per-class precision/recall/F1, confusion, applicable confidence intervals, and calibration status.

#### Scenario: Session head is not calibrated
- **WHEN** no held-out calibration procedure has been applied
- **THEN** score percentages are labeled uncalibrated probabilities and decision ambiguity is assessed primarily using the recorded margin

#### Scenario: Target class has insufficient support
- **WHEN** target-class support cannot sustain the configured validation policy
- **THEN** target-class reliability is inconclusive or unavailable rather than inferred from overall accuracy

### Requirement: Group-aware validation
The system SHALL support group-aware validation only from explicit source, batch, lot, device, acquisition, or object-group metadata and MUST NOT infer groups from filenames or directory structure by default.

#### Scenario: Valid group metadata exists
- **WHEN** explicit group identifiers provide sufficient class support per split
- **THEN** grouped validation records group source, group counts, fold support, metrics, and confidence intervals

#### Scenario: Group metadata is absent
- **WHEN** no approved group identifier exists
- **THEN** grouped validation is unavailable and sample-level OOF is not promoted to cross-acquisition evidence

### Requirement: Repeated learning curves
The system SHALL provide an explicitly started learning-curve experiment using fixed recipes, fixed audit policy, multiple training fractions, and repeated seeds or grouped bootstrap.

#### Scenario: Performance continues to rise
- **WHEN** validation metrics continue to improve through the largest training fraction with uncertainty reported
- **THEN** the system may state that more effective data could help but MUST NOT claim that quantity is the sole cause

#### Scenario: Performance plateaus at a low level
- **WHEN** train and validation metrics remain low and late-curve gains are within the configured uncertainty policy
- **THEN** the system identifies feature, label, scope, or classifier limitations as hypotheses requiring follow-up

#### Scenario: Overall score rises but crack recall does not
- **WHEN** aggregate metrics improve while `crack` recall remains flat
- **THEN** the recommendation prioritizes crack diversity and hard confusion-pair examples rather than indiscriminate data growth

### Requirement: Conservative validation policy
Thresholds for supported states SHALL be versioned, visible, and calibrated on validation data rather than adjusted on the test set.

#### Scenario: Provisional threshold fails
- **WHEN** balanced accuracy, macro-F1, critical-class recall, calibration, or interval width fails the active policy
- **THEN** the head remains validation-insufficient even if an individual heatmap appears plausible

