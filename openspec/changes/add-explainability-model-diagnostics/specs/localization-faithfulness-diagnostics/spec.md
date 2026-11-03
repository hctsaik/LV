## ADDED Requirements

### Requirement: Mask availability and geometry contract
The system SHALL report a mask as available only when an explicit resolver associates it with the source image and a traceable source-to-input-to-patch transform exists.

#### Scenario: Mask resolver is unavailable
- **WHEN** no validated resolver or fixture exists
- **THEN** mask status is unavailable and the system does not infer availability from filename conventions

#### Scenario: Mask dimensions are incompatible
- **WHEN** mask dimensions or crop geometry cannot be reconciled with the source and model input
- **THEN** localization is incompatible and no overlap metric is computed

### Requirement: P1 does not invite visual localization judgment
P1 SHALL show mask availability and evaluation state but MUST NOT show a default Grad-CAM/mask overlay or claim localization.

#### Scenario: A ground-truth mask exists in P1
- **WHEN** the resolver finds a valid mask before batch localization is implemented
- **THEN** the workspace reports `mask available; localization not evaluated` without presenting a localization pass

### Requirement: Batch localization evaluation
The system SHALL evaluate localization across an approved dataset using pointing, patch ranking, attribution mass, false-positive, uncertainty, and baseline comparisons.

#### Scenario: Run MVTec localization benchmark
- **WHEN** mapped masks and compatible attribution results exist for the audit set
- **THEN** the report includes class- and defect-size-stratified pointing, patch AUROC or AUPRC, Top-K precision/recall, attribution mass, outside-mask false positives, and confidence intervals

#### Scenario: Report thresholded IoU
- **WHEN** a research threshold is used to compute IoU or Dice from a continuous attribution map
- **THEN** the threshold source is recorded and the metric is not labeled segmentation accuracy

### Requirement: Full faithfulness evaluation
The system SHALL support batch perturbation evaluation with fixed targets, deletion curves, AOPC, ranking agreement, random controls, stability, and uncertainty.

#### Scenario: Attribution fails random baseline
- **WHEN** Top-K deletion or AOPC does not significantly exceed the configured random baseline
- **THEN** attribution faithfulness remains unsupported regardless of visual localization

#### Scenario: Perturbation cannot complete
- **WHEN** resource limits or OOM prevent faithfulness evaluation
- **THEN** faithfulness is unavailable and the system does not promote the attribution to a supported state

### Requirement: Shortcut diagnostics
Shortcut tests SHALL be explicit, reproducible jobs that record input transformation, prediction flips, margin changes, confidence status, and attribution movement.

#### Scenario: Background-only retains target score
- **WHEN** a background-only counterfactual preserves a high target margin
- **THEN** the report flags background dependence as a shortcut hypothesis without declaring it the sole cause

#### Scenario: Removing capsule text changes predictions
- **WHEN** masking the `500` text produces substantial class flips
- **THEN** the report records text dependence and recommends validation across text, lighting, orientation, and batch variations

### Requirement: Evidence-driven localization escalation
The system SHALL recommend Normal Bank, MIL, patch classification, or segmentation according to the available labels and the validated failure mode rather than treating them as interchangeable heatmap options.

#### Scenario: Classification improves but localization remains weak
- **WHEN** grouped classification meets policy but batch mask agreement remains unsupported
- **THEN** the next-step recommendation evaluates a localization-specific model rather than collecting only more whole-image classification images

#### Scenario: Product requires crack boundaries or measurements
- **WHEN** the product goal requires a crack mask, boundary, area, or physical measurement
- **THEN** the system requires an independently validated segmentation workflow and does not use Grad-CAM as the production mask

