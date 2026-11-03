## ADDED Requirements

### Requirement: Fair deterministic perturbations
LV SHALL compare identical mask geometry and area using bounded translations, deterministic seeds and multiple fill methods; it SHALL expose control availability and method stability.

#### Scenario: Feasible controls
- **WHEN** a mask can be translated with the configured overlap bound
- **THEN** control geometry and area equal the attribution mask and all pixels outside each mask remain unchanged.

#### Scenario: No feasible controls or inconsistent fill outcomes
- **WHEN** no valid placement exists or fill methods reverse the observed direction
- **THEN** the report is inconclusive or unstable and does not claim faithful explanation.

### Requirement: Independent support accounting
LV SHALL separate targets, unique images and declared independent groups while retaining cohort truth and execution identity.

#### Scenario: Repeated boxes or conflicting labels
- **WHEN** a batch contains multiple boxes from one source or inconsistent cohort metadata
- **THEN** repeated boxes do not increase image support and conflicting metadata cannot support cohort conclusions.

### Requirement: Quality-aware effort comparison
LV SHALL compare matched completed study records with active time, waiting time, reviewer identity and independent quality assessment; unmatched or incomplete records SHALL be visible but excluded from savings estimates.

#### Scenario: Incomplete historical logs
- **WHEN** old effort logs lack pairing or independent quality review
- **THEN** LV shows them as historical observations and does not report measured effort savings from them.

#### Scenario: Completed matched measurements
- **WHEN** both workflows have eligible paired records
- **THEN** LV reports observed time differences alongside quality and error outcomes, without extrapolating to unmeasured users.
