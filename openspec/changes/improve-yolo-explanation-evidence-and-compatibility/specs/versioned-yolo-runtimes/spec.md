## ADDED Requirements

### Requirement: Inspected isolated runtime
LV SHALL register legacy and modern pinned runtimes and inspect actual model task, family, release, head and gradient capability before execution.

#### Scenario: Modern model upload
- **WHEN** a user uploads supported YOLO12 or YOLO26 Detect weights
- **THEN** LV executes inspection and attribution in the modern runtime and displays the detected capabilities without changing legacy dependencies.

#### Scenario: Unsupported task or missing runtime
- **WHEN** a task cannot support the requested gradient path or its runtime is unavailable
- **THEN** LV reports the specific limitation and does not fabricate CAM or silently substitute a head.

### Requirement: Faithful native target correspondence
LV SHALL retain final output row, native candidate, class, runtime and head identity for modern attribution and verify normal forward and exported-model correspondence.

#### Scenario: Top-k reorders under perturbation
- **WHEN** a perturbation changes final detection ordering
- **THEN** fixed-target sensitivity continues to measure the original native candidate and class, and detection rematching remains a separate observation.

#### Scenario: Existing models and cross-runtime comparison
- **WHEN** legacy or DINO workflows execute or two different runtimes are compared
- **THEN** existing legacy identities remain valid and each model runs in its own registered environment.
