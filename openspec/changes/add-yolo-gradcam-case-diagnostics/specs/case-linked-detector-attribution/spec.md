## ADDED Requirements

### Requirement: Both case-first and model-first investigation
The system SHALL accept a report before a model is available and SHALL allow source-image inference before creation of a disputed ROI. Model trial images and results SHALL be promotable into a case without reupload. Navigation and refresh SHALL restore the selected case, image, run and profile from durable identities.

#### Scenario: Screenshot without model
- **WHEN** an engineer only has a screenshot and description
- **THEN** the engineer can create a case, select a region, record missing information and prepare a clearly scoped reply without completing model setup

#### Scenario: Inference before selecting a box
- **WHEN** a source image and compatible model are available but no case item exists
- **THEN** inference can produce detections for subsequent selection without requesting fabricated ROI coordinates first

### Requirement: Import real report materials
The system SHALL support multi-file image, prediction and annotation import with pairing preview, declared roles, class mapping and coordinate mapping. Supported CSV/JSON/TXT history SHALL retain original records and missing-provenance status without requiring original logs to contain LV-specific contract identities.

#### Scenario: Mixed filenames and empty predictions
- **WHEN** an import includes same-name images, five-column annotations, six-column predictions and a confirmed empty prediction file
- **THEN** pairing and roles are confirmed before publication, confidence is not fabricated and confirmed zero detections remain distinct from missing or unreadable files
- **AND** retries do not silently duplicate already imported regions

### Requirement: Correctable graphical investigation targets
The system SHALL provide graphical box selection and ROI drawing plus synchronized selection lists, editing and withdrawal. Predictions, reported regions and human labels SHALL remain distinct. Changes SHALL preserve historical evidence and identify affected claims.

#### Scenario: Correct a wrong region
- **WHEN** an engineer changes a mistaken ROI or withdraws a selected candidate after evidence was saved
- **THEN** the current target is corrected, prior evidence remains historical and affected citations are visibly marked for review rather than silently reassigned

### Requirement: Scope final machine-decision claims
The system SHALL distinguish model outputs, supported postprocessing and the reported final machine decision within deployment revisions. Same-checkpoint deployments with different settings SHALL remain distinct. Unknown downstream rules SHALL prevent claims of complete machine-decision reproduction.

#### Scenario: Boxes match but alarm rule is unknown
- **WHEN** replay matches detector boxes but the machine also uses an unspecified ROI or PLC rule
- **THEN** the system can report detector-output correspondence while final machine-decision correspondence remains unverified

### Requirement: No-path no-JSON standard handoff
The system SHALL provide browser import or existing-resource selection for reference and evaluation datasets, structured truth/group/purpose confirmation and training-class selection. Standard supported flows SHALL NOT require server paths, internal item IDs or handwritten JSON.

#### Scenario: Review and compare models using uploaded data
- **WHEN** an engineer uploads supported images and labels and selects old and new models
- **THEN** the interface builds the required manifests and class mappings from confirmed forms while retaining truth completeness, false-negative and independent-set overlap gates

### Requirement: Readable and revisable report delivery
The system SHALL offer selected-evidence factual drafts, editable text, supporting/opposing/pending relationships, image preview, text copying and portable illustrated HTML. Technical appendices SHALL remain available separately; the primary reply SHALL NOT expose server paths or raw internal JSON by default. Reply state SHALL remain separate from cause and improvement states.

#### Scenario: Inconclusive investigation with opposing evidence
- **WHEN** tests conflict with an engineer's hypothesis or no historical detection is reproduced
- **THEN** a reply can explicitly describe confirmed facts, conflicting evidence and missing information without automatically confirming a cause
- **AND** the user can review exactly which text and images are included before downloading, without automatic external sending

### Requirement: Recoverable work and resource lifecycle
The system SHALL persist job and attempt states, prevent duplicate submissions, support cancellation and retry, and mark interrupted jobs after restart. Completed artifacts SHALL survive case revision conflicts without overwriting newer reviews. Referenced model versions SHALL survive rename/archive and workspace restore.

#### Scenario: Completed analysis encounters a save conflict
- **WHEN** another session changes the case while analysis is executing
- **THEN** the completed artifact remains available with an explicit not-yet-published status and can be attached after compatibility revalidation without rerunning or overwriting the newer review

#### Scenario: Restart while a batch is running
- **WHEN** the service restarts with completed and incomplete items
- **THEN** completed items remain available, incomplete work is marked interrupted and retry scope is visible without silently resubmitting all items

### Requirement: Journey-based product acceptance
The release SHALL record implementation and browser evidence for U01–U16 in docs/lv-investigation-product-review.md, including first-use, missing-data, correction, recovery and delivery paths. Algorithm-only and synthetic tests SHALL NOT substitute for ordinary-user or actual deployment acceptance.

#### Scenario: Module tests pass but user must write JSON
- **WHEN** adapter tests pass but a standard training or evaluation path still requires internal IDs or handwritten JSON
- **THEN** overall product acceptance remains incomplete despite successful module validation

### Requirement: Upload-first model onboarding
The system SHALL offer My Models and an equivalent in-case import flow with model upload as the primary entry and a server-local path as an advanced large-file option. Uploaded bytes SHALL be persisted atomically under content identity before becoming usable; filenames SHALL NOT control storage paths or overwrite existing versions.

#### Scenario: Import a model from the user's computer
- **WHEN** the engineer uploads a supported model and names it
- **THEN** progress and applicable size limits are visible and the completed file is stored for reuse without requiring a server filesystem path
- **AND** an interrupted or invalid upload cannot become a usable model or erase the current case

#### Scenario: Duplicate model or duplicate filename
- **WHEN** two uploads share bytes or share only a filename
- **THEN** identical bytes can reuse storage while different bytes receive distinct identities, preserving existing profile snapshots

### Requirement: Guided inspection and trial
The system SHALL provide explicit model inspection, capability results, confirmation of necessary deployment settings, one-image trial and save-and-return-to-case. Model-derived metadata, user settings and suggested defaults SHALL have distinct provenance. Standard adapters SHALL NOT require tensor names, candidate indices, hook layers or JSON on the primary path.

#### Scenario: Inspect then trial a supported model
- **WHEN** a user requests inspection and supplies only missing required deployment settings
- **THEN** the system shows supported operations and performs a trial only on explicit action
- **AND** the trial displays the actual input and predictions before saving the profile and returning to the selected case image
- **AND** trial success does not imply historical deployment parity

#### Scenario: Unsupported checkpoint architecture
- **WHEN** a pinned inspection worker cannot safely load or recognize a supplied architecture within resource limits
- **THEN** it reports the failed stage and specific required information or supported alternatives without running arbitrary supplied loader code or substituting DINO

### Requirement: Progressive companion model attachment
The system SHALL permit ONNX-only investigation and later attachment of a companion gradient model within the same model entry using a new immutable revision. The interface SHALL distinguish deployment and attribution files and their correspondence status.

#### Scenario: Add PT after starting an ONNX case
- **WHEN** an engineer has already investigated a case using ONNX and later uploads a corresponding PT checkpoint
- **THEN** existing evidence is preserved and model correspondence is checked explicitly before presenting paired-model attribution
- **AND** shared filenames alone cannot establish correspondence

### Requirement: Preserve existing DINO diagnostics
The system SHALL preserve the existing DINO Grad-CAM algorithm, V1 serialization and compute identities, inspector entry, and explicit 224/336/448 head preparation. Provider dispatch MUST occur before DINO head preparation.

#### Scenario: External case enters diagnostics
- **WHEN** an engineer opens a YOLO case target in Model Diagnostics
- **THEN** the system uses the external provider without training a DINO head or showing DINO-only OOF/margin semantics
- **AND** the existing DINO routes remain usable independently

### Requirement: Reusable versioned model profiles
The system SHALL store immutable profile revisions containing runtime and gradient-backend identities, versions, capabilities, class mapping and preprocessing/postprocessing recipes. Cases SHALL retain their referenced snapshot.

#### Scenario: Reuse and update a model profile
- **WHEN** an engineer reuses a model on another case and subsequently changes its checkpoint or NMS settings
- **THEN** the second case can reuse the original profile without re-entering settings
- **AND** the change creates a new revision without changing earlier evidence meaning

### Requirement: Explicit detector capabilities
The release SHALL verify original YOLOv5, YOLOv8 and YOLO11 axis-aligned Detect adapters separately using pinned real forward/backward paths. Capability detection SHALL inspect actual task/head/output topology. Unsupported variants and infer-only backends MUST NOT silently produce substitute DINO Grad-CAM.

#### Scenario: Only an ONNX model is available
- **WHEN** the runtime has no verified gradient backend
- **THEN** Grad-CAM is marked unavailable with a specific reason and the existing controlled-experiment action remains available under its own name

#### Scenario: A segmentation or custom head is supplied
- **WHEN** the architecture is outside the tested adapter capability
- **THEN** the system rejects that attribution request without claiming mask, keypoint or angle explanation

### Requirement: Exact selected detection target
The system SHALL trace a selected final detection to an unambiguous native candidate in the gradient backend and preserve its exact score expression in a V2 target contract. Raw candidate indices SHALL NOT be assumed equal across runtimes or confused with truncated JSON list positions.

#### Scenario: Adjacent same-class detections and NMS reordering
- **WHEN** an engineer selects one of several nearby same-class boxes
- **THEN** backpropagation uses the selected candidate's recorded scalar rather than the highest-scoring box, list position or sum of detections
- **AND** ambiguous cross-backend matching prevents that target attribution

#### Scenario: YOLOv5 deployed score
- **WHEN** the detector's alert score is objectness multiplied by class confidence
- **THEN** the default attribution uses and labels that scalar; class-only analysis has a separate target identity

### Requirement: Preserve full-image geometry
The system SHALL use the original runtime input recipe and full frame for detector attribution. Selected ROI SHALL NOT silently become a crop inference. Layer maps SHALL be transformed through exact input geometry with padding excluded from original-image evidence.

#### Scenario: Non-square image with letterbox padding
- **WHEN** the engineer explains a small detection in a padded input
- **THEN** the displayed map aligns to the original image and selected box, with no padding mapped onto source pixels

### Requirement: Honest Grad-CAM methods and cleanup
The system SHALL provide explicit positive Grad-CAM and distinct signed-score semantics, record layers/shapes/scales, and validate normal-forward versus gradient-forward consistency. Hooks and model state SHALL be cleaned up after completion, cancellation and failure.

#### Scenario: No usable positive attribution
- **WHEN** gradients are detached, non-finite, zero, saturated or the positive map is all zero
- **THEN** the system reports the corresponding unavailable or no-positive-evidence state without manufacturing a normalized hotspot or switching scalar silently

### Requirement: Independent correspondence and historical statuses
The system SHALL separate gradient-backend/runtime correspondence from historical deployment replay. A matched companion PyTorch model SHALL retain its own identity and SHALL NOT be described as providing ONNX gradients or proving graph equivalence.

#### Scenario: Companion model agrees but history is absent
- **WHEN** PyTorch and ONNX outputs match on a specified image or recorded representative set but no deployment history is provided
- **THEN** correspondence displays its tested scope and history remains unavailable

### Requirement: Target-specific attribution validation
The system SHALL record fixed original-candidate scalar perturbation traces separately from rematched final-detection outcomes. Faithfulness SHALL include multiple perturbation proportions, matched-size or random controls, seeds, support and uncertainty. Optional explicit masks SHALL produce separate localization metrics with verified geometry.

#### Scenario: Another detection wins after occlusion
- **WHEN** perturbing an image suppresses the selected box and reveals another box
- **THEN** the original scalar trace and rematched detection trace remain separate and no missing score is substituted with zero
- **AND** the result does not automatically confirm a root cause or improvement

#### Scenario: No segmentation mask exists
- **WHEN** a case has boxes but no aligned pixel mask
- **THEN** localization remains unavailable rather than treating the box or CAM as ground truth mask

### Requirement: Shared diagnostics and durable case evidence
The system SHALL provide a case action to explain the selected box in the shared Diagnostics workspace, visibly separate YOLO attribution from LV auxiliary analysis, and persist raw maps, inputs, target/scalar/layers, model identities, methods and verification states as citeable case evidence.

#### Scenario: Save, reopen and export an explanation
- **WHEN** an engineer saves a CAM and restarts the application
- **THEN** the original case can reopen and cite it, and the exported evidence contains sufficient raw data and recipes to redraw it
- **AND** missing or corrupt artifacts are reported without presenting them as valid current evidence

### Requirement: Stale result isolation
The system SHALL include source, model/profile, native target, method, layer and geometry in attribution compute identity and reject late results for a changed target. Display-only settings SHALL NOT trigger recomputation.

#### Scenario: Switch boxes during execution
- **WHEN** a result for box A arrives after the engineer has selected box B
- **THEN** it cannot replace B's current result and can only be retained as explicitly historical evidence

### Requirement: Reviewed active-learning handoff and effort measurement
The system SHALL allow a selected explanation region to seed existing case comparison search with attribution provenance and hand off candidates to the existing review workflow. Heatmaps SHALL NOT create training labels or promote causal status. Effort evaluation SHALL separate initial model setup from subsequent case completion.

#### Scenario: Use a suspicious region to find more cases
- **WHEN** an engineer searches from a CAM-linked region and selects similar candidates
- **THEN** candidates retain source evidence and remain pending until human review
- **AND** existing full-image training and independent-evaluation safeguards remain enforced
