## ADDED Requirements

### Requirement: Raw Top-K margin scores
The system SHALL compute same, other and margin from the current model's raw high-dimensional embeddings using scale-safe float64 cosine. same SHALL be the mean of the top-K cosine neighbors in the query's effective label, excluding any row with the same record_id. other SHALL be the highest per-rival-class Top-K mean. margin SHALL be same−other only when both are finite. Production computation SHALL never allocate a pairing matrix larger than the configured query and reference blocks. N>5000 SHALL be rejected without truncating.

#### Scenario: Hand fixture K=1
- **WHEN** unit vectors a1/A=(1,0), a2/A=(0.8,0.6), b1/B=(0,1), b2/B=(0.6,0.8) are scored with K=1
- **THEN** a1 and b1 are (0.8, 0.6, 0.2) and a2 and b2 are (0.8, 0.96, -0.16)

#### Scenario: Blocked equals oracle
- **WHEN** the blocked algorithm is compared with an independent full-matrix oracle on several block sizes
- **THEN** items, neighbors, effective K and means match within 1e-6 and membership comparisons use exact float64

### Requirement: Stable identity and eligibility
Image record IDs SHALL hash canonical `{level:image, path}`. Object IDs SHALL hash `{level:object, image_path, bbox, obj_index}`. Missing provenance SHALL use `invalid:<data_token>:<original_index>` and MUST NOT enter candidates, cart, or evaluation. Duplicate identities or one crop path mapped to different source identities SHALL abort the analysis. Classifier flat-fallback rows SHALL be unresolved; literal unknown/mix class folders SHALL remain single. Effective labels after readback SHALL be used for scoring.

#### Scenario: Crop cache collision
- **WHEN** two objects from different originals share a crop path
- **THEN** the analysis returns an integrity error and MUST NOT emit normal scores

### Requirement: Threshold preview is a mask
Candidates SHALL be rows with finite margin strictly less than the canonical threshold. Equality SHALL NOT select. Null SHALL NOT be treated as 0. Slider movement SHALL NOT recompute embeddings or neighbors. Uncommitted draft SHALL disable add-all, save operating point, and downloads. Actions MUST present analysis_key, preview_key and request_seq; stale or out-of-order requests SHALL be rejected.

#### Scenario: Equal threshold
- **WHEN** t=0.1 and a row has margin=0.1
- **THEN** that row is not a candidate

### Requirement: Human-evaluation Precision/Recall
A positive example SHALL be a frozen evaluation row whose verdict is incorrect. unreviewed and unresolved SHALL NOT be treated as correct. Precision and Recall SHALL be computed only on E=E0∩eligible when every E row has a binary verdict. Zero denominators SHALL be null with an explicit reason. Unverified embedding generation SHALL forbid PR. Heldout SHALL require the saved operating point, analysis_key, score protocol, disjoint IDs and disjoint source groups.

#### Scenario: Incomplete truth
- **WHEN** E0 contains an unreviewed eligible row
- **THEN** the UI shows coverage and MUST NOT display a fabricated PR

### Requirement: Export and non-mutation
JSON schema_version SHALL be 2 for analysis reports and 1 for evaluation manifests. All scopes SHALL include full-U score_index. Source datasets SHALL NOT be written. Existing confusion, similar search, cart and labeling entry points SHALL remain reachable. Gray-band send SHALL NOT be invoked.

#### Scenario: Selected subset rebuild
- **WHEN** a selected-scope JSON contains only class A queries plus one invalid diagnostic in all-scope
- **THEN** score_index still rebuilds every class count, eligible, candidate and global PR
