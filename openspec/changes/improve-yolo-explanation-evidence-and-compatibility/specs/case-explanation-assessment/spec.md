## ADDED Requirements

### Requirement: Question-aware cited assessment
LV SHALL provide controlled evidence summaries containing the question, verified truth, replay scope, observations, uncertainty, missing evidence, next actions and evidence references.

#### Scenario: Class-score sensitivity
- **WHEN** a measured attribution report and controls are available
- **THEN** the summary distinguishes observed sensitivity from causal explanation and cites the underlying report.

#### Scenario: Machine-alarm question
- **WHEN** the investigation asks why a machine alarm occurred
- **THEN** CAM alone is described as insufficient and deployment rules or logs are requested as missing evidence.

### Requirement: Preserve human authorship and review provenance
LV SHALL preserve edited replies until explicit regeneration and store human target-preservation confirmation separately with attribution and variant hashes.

#### Scenario: Rerun after manual edit
- **WHEN** a case page reruns or an evidence job completes
- **THEN** the saved edited reply remains unchanged and any generated summary is separately identifiable.

#### Scenario: Changed perturbation evidence
- **WHEN** variant hashes no longer match a prior human confirmation
- **THEN** the confirmation does not validate the new variants.
