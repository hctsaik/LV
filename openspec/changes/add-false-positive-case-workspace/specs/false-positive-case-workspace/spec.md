## ADDED Requirements

### Requirement: Persistent lightweight case intake
The workspace SHALL accept a readable image or screenshot with optional external output metadata, preserve content identity, and allow resume and additional image sources without modifying source data. Concurrent stale saves MUST be rejected.

#### Scenario: Incomplete screenshot report
- **WHEN** only a screenshot with unknown model version is available
- **THEN** a case can be saved and searched, and reports explicitly state that the original model decision has not been replayed.

#### Scenario: Concurrent edit
- **WHEN** a second session saves an outdated case revision
- **THEN** it receives a reload error and the latest saved verdicts are retained.

### Requirement: Region-scoped review
The workspace SHALL maintain separate disputed prediction regions with pending, normal, defect, disputed or insufficient verdicts and reviewer/reason evidence. Missing annotations MUST NOT become normal verdicts.

#### Scenario: Mixed image
- **WHEN** one region is reviewed normal and another is reviewed defect
- **THEN** their verdicts remain separate and no whole-image negative training label is generated.

### Requirement: Balanced auxiliary retrieval
The workspace SHALL explicitly retrieve comparisons from supplied normal and defect sources and reviewed historical cases, using the same LV embedding recipe for query and candidates, recording provenance and retaining empty categories.

#### Scenario: Missing opposite-class evidence
- **WHEN** the defect reference set is empty
- **THEN** the defect column states no comparisons were found in the searched source, without inventing examples or asserting an out-of-distribution cause.

#### Scenario: Historical different model
- **WHEN** a historical comparison uses a different or unknown deployed model version
- **THEN** its source version remains visible and its cause is not inherited by the current case.

### Requirement: Cited editable case explanation
The workspace SHALL save claims as facts, hypotheses or actions with evidence references, export a human-readable report and a JSON snapshot, and distinguish reply status from cause and improvement status.

#### Scenario: Auxiliary similarity
- **WHEN** an LV similarity result is included in the report
- **THEN** it is labeled auxiliary, not an explanation of the external model's actual internal decision.

### Requirement: Review-only Active Learning handoff
The workspace SHALL persist deduplicated candidate review manifests linked to case and round identities, support validated review return, and never create positive or negative training annotations. ROI review counts MUST remain distinct from image and labeled-training counts.

#### Scenario: Partial group review
- **WHEN** three of one hundred candidate regions have completed human review
- **THEN** the manifest reports three reviewed regions and ninety-seven pending regions and does not mark the entire group normal or add one hundred labeled images to the ledger.

#### Scenario: Return from another batch
- **WHEN** a return manifest names another case, batch, or region identity
- **THEN** the return is rejected without partially applying verdicts.

### Requirement: Measurable investigation effort
The workspace SHALL offer explicit start/pause and correction of investigation time, persist recorded time, and display reviewed-region counts without claiming model-quality improvement.

#### Scenario: Resume investigation
- **WHEN** an engineer reopens a case
- **THEN** saved effort and evidence remain available and no inference starts automatically.
