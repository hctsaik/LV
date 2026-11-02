export type ModelOption = {
  model_id: string;
  name: string;
  available: boolean;
};

export type ProjectionMethod = "pca" | "tsne" | "umap" | "lda" | "sumap";
export type VisualizeUnit = "image" | "object";
export type ComparisonObjectPolicy = {
  pad: number;
  target_res: 168 | 196 | 224 | 280 | 336 | 392 | 448;
  head: "cls" | "meanpool";
};

export type JobStatus = {
  job_id: string;
  state: "queued" | "running" | "cancelling" | "completed" | "failed" | "cancelled";
  phase: string;
  done: number;
  total: number;
  percent: number;
  message: string;
  run_id: string | null;
  error: { code: string; message: string; request_id: string } | null;
  can_cancel: boolean;
};

export type DetectionEvaluationClass = {
  class_name: string;
  tp: number;
  fp: number;
  fn: number;
  n_gt: number;
  precision: number;
  recall: number;
};

export type DetectionEvaluation = {
  evaluation_id: string;
  demo?: boolean;
  overall: Omit<DetectionEvaluationClass, "class_name">;
  per_class: DetectionEvaluationClass[];
  escape_count: number;
  escapes: Array<{
    index: number;
    filename: string;
    class_name: string;
    box: { cx: number; cy: number; w: number; h: number };
  }>;
  confusion: Array<{ ground_truth: string; predicted: string; count: number }>;
  gray_gt_count: number;
  consensus_gt_count: number | null;
  image_count: number;
  prediction_image_count: number;
  used_consensus: boolean;
  warnings: Array<"NO_PREDICTIONS_MATCHED" | "CLASS_NAMESPACE_MISMATCH" | "CONSENSUS_NOT_PROVIDED">;
};

const nativeFolderGrants = new Map<string, string>();

export type AnomalyModel = {
  model_id: string;
  source: "built" | "saved" | "loaded";
  saved: boolean;
  active?: boolean;
  saved_at: number | null;
  built_at: number | null;
  meta: {
    model: string;
    target_res: number;
    score_mode: "patch" | "object";
    object_source: "yolo" | "whole_image";
    patch_dim: number | null;
    obj_dim: number | null;
    n_objects: number;
    label_semantic: "object" | "defect";
    embedding_recipe: Record<string, unknown>;
  };
  summary: {
    n_images: number;
    n_objects: number;
    has_normal_bank: boolean;
    has_head: boolean;
    head_status: string;
    eligible_classes: string[];
  };
};

export type AnomalyModelJob = {
  job_id: string;
  state: JobStatus["state"];
  phase: string;
  done: number;
  total: number;
  percent: number;
  message: string;
  model_id: string | null;
  result: AnomalyModel | null;
  error: JobStatus["error"];
  can_cancel: boolean;
};

export type AnomalyScoreJob = {
  job_id: string;
  state: JobStatus["state"];
  phase: string;
  done: number;
  total: number;
  percent: number;
  message: string;
  run_id: string | null;
  result: AnomalyRun | null;
  error: JobStatus["error"];
  can_cancel: boolean;
};

export type FewshotSampleBank = {
  sample_bank_id: string;
  model_id: string;
  target_res: number;
  object_source: "yolo" | "whole_image";
  sample_count: number;
  class_counts: Record<string, number>;
  feature_dimension: number;
};

export type FewshotSampleBankJob = {
  job_id: string;
  state: JobStatus["state"];
  phase: string;
  done: number;
  total: number;
  percent: number;
  message: string;
  result: FewshotSampleBank | null;
  error: JobStatus["error"];
  can_cancel: boolean;
};

export type FewshotCandidate = {
  index: number;
  item_id: string;
  filename: string;
  bbox: [number, number, number, number];
  obj_index: number;
  label: string;
  suggested_class: string;
  similarity: number;
  reason: string;
};

export type FewshotScan = {
  scan_id: string;
  sample_bank_id: string;
  model_id: string;
  target_res: number;
  object_source: "yolo" | "whole_image";
  k: number;
  images_total: number;
  images_processed: number;
  objects_scored: number;
  candidate_count: number;
  done: boolean;
  candidates: FewshotCandidate[];
};

export type FewshotScanJob = Omit<FewshotSampleBankJob, "result"> & {
  result: FewshotScan | null;
};

export type FewshotMonitorQueueItem = {
  index: number;
  item_id: string;
  filename: string;
  bbox: [number, number, number, number];
  obj_index: number;
  suggested_class: string;
  similarity: number;
  reason: string;
  run_id: string;
  status: string;
  human_decision: "defect" | "skip" | null;
  human_label: string;
};

export type FewshotMonitorStatus = {
  monitor_workspace_id: string;
  last_run: {
    run_id?: string;
    at?: number;
    new?: number;
    changed?: number;
    deleted?: number;
    objects_scored?: number;
    queue_len?: number;
    model_version?: string | null;
    objective?: string;
    done?: boolean;
  } | null;
  queue_len: number;
  locked: boolean;
  items: FewshotMonitorQueueItem[];
};

export type FewshotMonitorJob = Omit<FewshotSampleBankJob, "result"> & {
  result: {
    status: string;
    run_id: string;
    new: number;
    changed: number;
    deleted: number;
    objects_scored: number;
    queue_len: number;
    done: boolean;
  } | null;
};

export type AnomalyRun = {
  run_id: string;
  revision: number;
  model_id: string;
  model_bundle_id: string | null;
  model_meta: AnomalyModel["meta"];
  model_bank_sha256: string;
  score_mode: "patch" | "object";
  object_source: "yolo" | "whole_image";
  n_objects: number;
  n_bad: number;
  threshold: number;
  auroc: number | null;
  score_min: number | null;
  score_max: number | null;
  class_labels: string[];
  verdict_counts: { good: number; bad: number };
  confirmed_count: number;
  fewshot_aligned: number;
  fewshot_unmatched: number;
  skipped_count: number;
  created_at: number;
};

export type AnomalyObject = {
  index: number;
  filename: string;
  label: string;
  split: string;
  bbox: [number, number, number, number];
  obj_index: number;
  score: number;
  verdict: "good" | "bad";
  confirmed: "good" | "bad" | "";
};

export type AnomalyObjectPage = {
  run_id: string;
  revision: number;
  offset: number;
  limit: number;
  total: number;
  items: AnomalyObject[];
};

export type AnomalyPrelabelItem = {
  index: number;
  filename: string;
  label: string;
  obj_index: number;
  bbox: [number, number, number, number];
  score: number;
  pred: string;
  confidence: number;
  accept: boolean;
};

export type AnomalyPrelabelPreview = {
  run_id: string;
  revision: number;
  min_conf: number;
  target_classes: string[];
  class_names: string[];
  total: number;
  accepted: number;
  offset: number;
  limit: number;
  items: AnomalyPrelabelItem[];
};

export type AnomalyPrelabelOptions = {
  expected_revision: number;
  min_conf: number;
  target_classes: string[];
};

export type AnomalyPrelabelExport = {
  run_id: string;
  revision: number;
  accepted: number;
  written: number;
  objects: number;
  class_names: string[];
};

export type RunRecord = {
  index: number;
  filename: string;
  split: string;
  label: string;
  class_id?: number | null;
  bbox?: [number, number, number, number] | null;
  obj_index?: number | null;
  score?: number | null;
  short_px?: number | null;
};

export type EmbeddingPoint = RunRecord & { x: number; y: number; z: number };

export type VisualizationView = {
  model_id: string;
  method: ProjectionMethod;
  points: EmbeddingPoint[];
  embedding_shape: [number, number];
  embedding_finite: boolean;
  embedding_nonzero: boolean;
};

export type UmapReferenceStatus = {
  refitted: boolean;
  new_points: number;
};

export type DiagnosticHeadStatus = {
  status: "ready" | "not_ready" | "stale" | "incompatible";
  resolution: 224 | 336 | 448;
  reason?: string;
  model_id?: string;
  unit?: VisualizeUnit;
  pooling?: string;
  classes?: string[];
  training_samples?: number;
  embedding_dim?: number;
  oof_available?: boolean;
  oof_balanced_accuracy?: number | null;
  reused?: boolean;
};

export type DiagnosticExplanationResult = {
  status: "ready";
  run_id: string;
  context_id: string;
  record_index: number;
  resolution: 224 | 336 | 448;
  model_id: string;
  unit: VisualizeUnit;
  scope: string;
  method: "attention" | "gradcam";
  attention_method: "last_layer_head_mean" | "single_head" | "rollout" | null;
  grid: [number, number];
  patch_size: number;
  score_mode: "signed" | "positive";
  display_scores: number[][];
  raw_scores: number[][];
  classes: string[];
  predicted_class: string;
  predicted_probability: number;
  class_probabilities: { class_name: string; probability: number }[];
  source_sha256: string;
  model_input_png_base64: string;
  overlay_png_base64: string;
  target_class?: string;
  target_score?: number;
  reference_class?: string;
  runner_up_class?: string;
  runner_up_probability?: number;
  probability_margin?: number;
  target_probability?: number;
  reference_probability?: number;
  is_predicted_target?: boolean;
  positive_mass?: number;
  negative_mass?: number;
  faithfulness?: Record<string, unknown>;
  head_diagnostics?: Record<string, unknown>;
  sample_oof?: Record<string, unknown> | null;
  warning: string;
  warnings: string[];
};

export type DiagnosticExplanationInput = {
  current_run_id: string;
  resolution: 224 | 336 | 448;
  method: "attention" | "gradcam";
  attention_method: "last_layer_head_mean" | "single_head" | "rollout";
  head_index: number | null;
  target_class: string | null;
  display_mode: "signed" | "positive";
};

export type DiagnosticReference = {
  context_id: string;
  run_id: string;
  record_index: number;
  filename: string;
  label: string;
  split: string;
  unit: VisualizeUnit;
  model_id: string;
  object_index: number | null;
  bbox: [number, number, number, number] | null;
  source_sha256: string | null;
  status: "current" | "stale" | "incompatible";
  reason: string;
};

export type VisualizationRun = {
  mode: "classifier" | "detector";
  unit: VisualizeUnit;
  object_policy: ComparisonObjectPolicy | null;
  class_names: string[];
  model_id: string;
  method: ProjectionMethod;
  records: RunRecord[];
  points: EmbeddingPoint[];
  views: VisualizationView[];
  models: string[];
  methods: ProjectionMethod[];
  embedding_shapes: Record<string, [number, number]>;
  embedding_shape: [number, number];
  embedding_finite: boolean;
  embedding_nonzero: boolean;
  umap_reference: { enabled: boolean; by_model: Record<string, UmapReferenceStatus> };
  warnings: string[];
};

export type ClassSimilarityMetric = "margin" | "same" | "other";

export type ClassSimilarityNeighbor = {
  index: number;
  filename: string;
  label: string | null;
  cosine: number;
};

export type ClassSimilarityItem = {
  record_id: string;
  index: number;
  filename: string;
  label: string | null;
  split: string;
  bbox: [number, number, number, number] | null;
  obj_index: number | null;
  same: number | null;
  other: number | null;
  margin: number | null;
  rival_class: string | null;
  same_k: number;
  rival_k: number;
  status: string;
  reasons: string[];
  same_neighbors: ClassSimilarityNeighbor[];
  rival_neighbors: ClassSimilarityNeighbor[];
  tied_rival_count: number;
  tied_rival_classes: string[];
};

export type ClassSimilarityBoxSummary = {
  label: string;
  n_total: number;
  n_scored: number;
  n_unscored: number;
  metric: ClassSimilarityMetric;
  q1?: number;
  median?: number;
  q3?: number;
  iqr?: number;
  whisker_low?: number;
  whisker_high?: number;
};

export type ClassSimilarityScatterItem = {
  record_id: string;
  index: number;
  label: string | null;
  value: number;
  jitter: number;
};

export type ClassSimilarityResult = {
  analysis_key: string;
  run_id: string;
  model_id: string;
  unit: VisualizeUnit;
  k: number;
  analysis_split: string;
  source_status: string;
  summary: {
    input_count: number;
    valid_count: number;
    excluded_count: number;
    per_class_counts: Record<string, number>;
  };
  summaries: Record<ClassSimilarityMetric, ClassSimilarityBoxSummary[]>;
  scatter_items: Record<ClassSimilarityMetric, ClassSimilarityScatterItem[]>;
  items: ClassSimilarityItem[];
};

export type ClassSimilarityJob = {
  job_id: string;
  run_id: string;
  state: JobStatus["state"];
  done: number;
  total: number;
  percent: number;
  result: ClassSimilarityResult | null;
  error: JobStatus["error"];
  can_cancel: boolean;
};

export type ClassSimilarityPolicy = { mode: "threshold" | "none" | "all_eligible"; threshold: number | null; comparator: "lt"; score: "margin" };

export type ClassSimilarityEvaluationPreview = {
  policy: ClassSimilarityPolicy;
  summary: { eligible_n: number; candidate_n: number; unscorable_n: number; ratio: number | null; per_class: Array<Record<string, unknown>>; input_count: number };
  evaluation: { evaluation_id: string; evaluation_version: number; purpose: "tuning" | "heldout"; truth_digest: string; frozen_ids: string[]; pending_n: number } | null;
  frozen_policy: { policy_id: string; created_at: string; policy: ClassSimilarityPolicy; analysis_key: string; score_protocol_key: string; tuning_evaluation_key: string | null; tuning_ids: string[]; tuning_group_ids: string[] } | null;
  counts: {
    status: string; unavailable_reason: string | null; coverage: number | null;
    e0_n: number; e_n: number; known_incorrect_n: number; unknown_n: number; unscorable_in_e0: number;
    tp: number | null; fp: number | null; fn: number | null; tn: number | null;
    precision: number | null; recall: number | null; confusion_ids: Record<string, string[]>;
  } | null;
  workload_curve: Array<{ t: number | null; mode: string; candidate_n: number; label?: string | null }>;
  precision_recall_curve: Array<{ threshold: number | null; precision: number; recall: number; tp: number; fp: number; fn: number }>;
};

export type CoverageRun = {
  run_id: string;
  model_id: string;
  unit: VisualizeUnit;
  object_policy: ComparisonObjectPolicy | null;
  method: ProjectionMethod;
  requested_method: ProjectionMethod;
  dimension: 2 | 3;
  records: RunRecord[];
  points: EmbeddingPoint[];
  embedding_shape: [number, number];
  k: number;
  sparsity: number[];
  sparse_threshold: number;
  sparse_count: number;
  sparse_percentile: number;
  warnings: string[];
};

export type CoverageObjectViewInput = {
  class_label?: string | null;
  min_short_px?: number;
  confidence_min?: number;
  confidence_max?: number;
  method: ProjectionMethod;
  dimension: 2 | 3;
  k: number;
  block_k: number;
};

export type CoverageClassSummary = {
  label: string;
  count: number;
  analyzable: boolean;
  sparse_count: number;
  max_sparsity: number;
};

export type CoverageBlockSummary = {
  block_id: number;
  count: number;
  mean_sparsity: number;
};

export type CoverageObjectView = {
  class_label: string | null;
  method: ProjectionMethod;
  requested_method: ProjectionMethod;
  dimension: 2 | 3;
  records: RunRecord[];
  points: Array<EmbeddingPoint & { block_id?: number }>;
  k: number;
  sparsity: number[];
  sparse_threshold: number;
  sparse_count: number;
  class_summary: CoverageClassSummary[];
  block_k: number;
  blocks: CoverageBlockSummary[];
  has_confidence: boolean;
  warnings: string[];
};

export type CoverageCandidateRole = "pool" | "reference";

export type CoverageCandidateViewInput = {
  role: CoverageCandidateRole;
  class_label: string | null;
  min_short_px: number;
  confidence_min: number;
  confidence_max: number;
  method: ProjectionMethod;
  dimension: 2 | 3;
  k: number;
  limit: number;
};

export type CoverageCandidatePoint = EmbeddingPoint & {
  score: number | null;
  provisional_label: string;
  role: CoverageCandidateRole;
};

export type CoverageCandidateWorkRecord = RunRecord & {
  score: number;
  provisional_label: string;
};

export type CoverageClassGap = {
  label: string;
  main_count: number;
  candidate_count: number;
  uncovered_count: number;
  state: "missing" | "sparse" | "covered";
};

export type CoverageCandidateView = {
  role: CoverageCandidateRole;
  class_label: string | null;
  method: ProjectionMethod;
  requested_method: ProjectionMethod;
  dimension: 2 | 3;
  main_count: number;
  candidate_count: number;
  main_points: EmbeddingPoint[];
  candidate_points: CoverageCandidatePoint[];
  main_sparsity: number[];
  main_sparse_threshold: number;
  main_sparse_count: number;
  k: number;
  radius: number | null;
  recall: number | null;
  uncovered_count: number;
  work_indices: number[];
  work_scores: Record<number, number>;
  work_records: CoverageCandidateWorkRecord[];
  class_gaps: CoverageClassGap[];
  warnings: string[];
};

export type AddCoverageCandidateCurationResult = {
  added: number;
  skipped: number;
  items: CurationItem[];
  total: number;
};

export type CoverageProjectionResult = {
  method: ProjectionMethod;
  requested_method: ProjectionMethod;
  dimension: 2 | 3;
  points: EmbeddingPoint[];
  warnings: string[];
};

export type GroupDiffRegion = {
  rank: number;
  peak_z: number;
  mean_z: number;
  stability: number;
  pct: number;
  text: string;
  bbox_px: [number, number, number, number];
  good_count: number;
  bad_count: number;
};

export type GroupDiffRun = {
  run_id: string;
  grid: [number, number];
  score_map: number[][];
  z_map: number[][];
  stability_map: number[][];
  verdict: boolean;
  p_global: number;
  threshold_z: number;
  regions: GroupDiffRegion[];
  n_good: number;
  n_bad: number;
  skipped: { unreadable: number; grid: number };
  alignment_warning: boolean;
  warn_msgs: string[];
  meta: { model: string; target_res: number; n_perm: number; alpha: number; stab_q: number; seed: number; top_k: number };
  bases_good: string[];
  bases_bad: string[];
  base_good_index: number;
  base_bad_index: number;
};

export type GroupDiffPca = { xy_good: number[][]; xy_bad: number[][]; evr: number[] };

export type CoverageSparsityResult = {
  k: number;
  scores: number[];
  threshold: number;
  sparse_count: number;
  sparse_percentile: number;
};

export type CompletenessAxis = "label" | "split" | "brightness" | "contrast" | "sharpness" | "aspect";
export type CompletenessFrequency = "高" | "中" | "低" | "不適用";

export type CoverageCompletenessInput = {
  axis_x: CompletenessAxis;
  axis_y: CompletenessAxis;
  bins: number;
  t_abs: number;
  d_star: number;
  frequencies: Array<{ x: number; y: number; frequency: CompletenessFrequency }>;
};

export type CoverageCompletenessCell = {
  x: number;
  y: number;
  x_label: string;
  y_label: string;
  n: number;
  d: number;
  t: number;
  state: "空" | "缺" | "偏缺" | "健康" | "假完整" | "過多" | "不適用";
  sample_indices: number[];
  shortfall?: number;
};

export type CoverageCompletenessResult = {
  run_id: string;
  axis_x: CompletenessAxis;
  axis_y: CompletenessAxis;
  labels_x: string[];
  labels_y: string[];
  bins: number;
  t_abs: number;
  d_star: number;
  calibrated: boolean;
  frequency_classes: Array<{ x: number; y: number; frequency: CompletenessFrequency }>;
  cells: CoverageCompletenessCell[];
  health: {
    coverage_health: number;
    counts: Record<string, number>;
    gini: number;
    fake_ratio: number;
    top_gaps: CoverageCompletenessCell[];
  };
};

export type CoverageCompletenessCandidate = {
  index: number;
  filename: string;
  split: string;
  label: string;
  distance: number;
};

export type CoverageCompletenessCandidateResult = {
  run_id: string;
  cell: { x: number; y: number };
  items: CoverageCompletenessCandidate[];
  total: number;
};

export type KMeansCenter = {
  block_id: number;
  x: number;
  y: number;
  z: number;
  count: number;
};

export type KMeansBlocks = {
  k: number;
  centers: KMeansCenter[];
  points: Array<EmbeddingPoint & { block_id: number }>;
  block_id: number | null;
  drilldown_points: EmbeddingPoint[];
};

export type KMeansBlocksInput = {
  model_id: string;
  method: ProjectionMethod;
  k: number;
  label_filter?: string | null;
  split_filter?: string | null;
  block_id?: number | null;
};

export type SimilarSearchResult = {
  query_index: number;
  model_id: string;
  results: Array<{ index: number; distance: number }>;
};

export type TextSearchResult = {
  query: string;
  model_id: string;
  results: Array<{ index: number; distance: number }>;
};

export type DuplicateScanInput = {
  method: "phash" | "embedding";
  model_id?: string;
  max_hamming: number;
  max_distance: number;
  cross_split_only: boolean;
  max_pairs?: number;
};

export type DuplicateScanResult = {
  method: "phash" | "embedding";
  model_id: string | null;
  cross_split_only: boolean;
  results: Array<{ left_index: number; right_index: number; distance: number }>;
};

export type DuplicateLabelingHandoffResult = {
  handoff_id: string;
  task: string;
  sent: number;
  pair_count: number;
  skipped_handled: number;
  skipped_pairs: number;
  round_id: string;
  created_round: boolean;
  saved_to_local_inbox: boolean;
};

export type RankingMetric = "outlier" | "disagreement";
export type RankingResult = {
  model_id: string;
  metric: RankingMetric;
  k: number;
  score_direction: "descending";
  interpretation: string;
  results: Array<{ rank: number; index: number; score: number }>;
};

export type CompareUnit = "whole" | "object";
export type ComparisonRecord = {
  index: number;
  filename: string;
  label: string;
  source_id: string;
  class_id: number | null;
  bbox: [number, number, number, number] | null;
  score: number | null;
  short_px: number | null;
};

export type DriftRow = {
  label: string;
  a_count: number;
  b_count: number;
  drift: number;
};

export type ComparisonRun = {
  unit: CompareUnit;
  model: string;
  name_a: string;
  name_b: string;
  records_a: ComparisonRecord[];
  records_b: ComparisonRecord[];
  rows: DriftRow[];
  a_only: Record<string, number>;
  b_only: Record<string, number>;
  policy: ComparisonObjectPolicy | null;
  policy_source: "automatic" | "manual" | null;
  autotune_rows: Array<{
    policy: ComparisonObjectPolicy;
    macro_purity: number | null;
    buckets: Record<string, { n: number; purity: number | null }>;
  }>;
  warnings: string[];
};

export type ComparisonPoint = {
  group: "A" | "B";
  index: number;
  filename: string;
  label: string;
  x: number;
  y: number;
  z: number;
};

export type ComparisonProjection = {
  label: string;
  method: ProjectionMethod;
  requested_method: ProjectionMethod;
  dimension: 2 | 3;
  points: ComparisonPoint[];
  warnings: string[];
  a_count: number;
  b_count: number;
};

export type ComparisonJobInput = {
  folders_a: string[];
  folder_grant_ids_a?: Array<string | null>;
  folders_b: string[];
  folder_grant_ids_b?: Array<string | null>;
  model_id: string;
  unit: CompareUnit;
  class_names?: string[];
  object_policy?: ComparisonObjectPolicy;
};

export type ComparisonCurationSelection = { group: "A" | "B"; index: number };

export type CurationItem = {
  id: string;
  index: number;
  filename: string;
  label: string;
  split: string;
  sha256: string;
  source: string;
  score: number | null;
  reason: string;
  level: "image" | "object";
  bbox?: [number, number, number, number] | null;
  obj_index?: number | null;
  added_at: string;
};

export type CurationItemsResponse = { items: CurationItem[]; total: number };
export type AddCurationResponse = CurationItemsResponse & { added: number; skipped: number };
export type CurationSubsetOptions = {
  source: string;
  destination: string;
  mode: "copy" | "symlink" | "manifest-only";
  on_exists: "skip" | "rename" | "overwrite";
};
export type CurationSubsetPreview = {
  selected: number;
  available: number;
  unique_images: number;
  deduplicated: number;
  missing: number;
  missing_filenames: string[];
  label_files: number;
  object_items: number;
  preview_token: string;
  destination_safe: boolean;
};
export type CurationSubsetExport = {
  exported: number;
  skipped: number;
  deduplicated: number;
  unresolved: number;
  corrupt: number;
  warnings: number;
  errors: number;
  annotations_preserved: number;
  partial: boolean;
  training_warning: string;
};

export type CurationHandoffPreview = {
  selected: number;
  available: number;
  unavailable: number;
  unique_images: number;
  deduplicated: number;
  new_images: number;
  handled_images: number;
  sendable_images: number;
  skipped_handled: number;
};
export type CurationHandoffResult = {
  handoff_id: string;
  task: string;
  sent: number;
  deduplicated: number;
  skipped_handled: number;
  round_id: string;
  created_round: boolean;
  saved_to_local_inbox: boolean;
};
export type ActiveLearningRound = {
  round_id: string;
  ts: string;
  strategy: string;
  params: Record<string, unknown>;
  tool: string;
  note: string;
  batches: Array<{ batch_id: string; n_sent: number }>;
  n_sent: number;
  readbacks: Array<Record<string, unknown>>;
  n_labeled: number;
  n_changed: number;
  metrics: Array<{ name: string; value: number | null; source: string; ts: string }>;
  evaluations: DetectionEvaluationRoundReport[];
};
export type DetectionEvaluationRoundReport = {
  evaluation_id: string;
  ts: string;
  parameters: { iou_thresh: number; conf_thresh: number; class_aware: boolean };
  summary: {
    overall: Omit<DetectionEvaluationClass, "class_name">;
    per_class: DetectionEvaluationClass[];
    escape_count: number;
    confusion: Array<{ ground_truth: string; predicted: string; count: number }>;
    gray_gt_count: number;
    consensus_gt_count: number | null;
    image_count: number;
    prediction_image_count: number;
    used_consensus: boolean;
    warnings: DetectionEvaluation["warnings"];
  };
};
export type ActiveLearningRoundsResponse = {
  rounds: ActiveLearningRound[];
  orphan_events: Array<Record<string, unknown>>;
};
export type ProbeEvaluationResult = {
  round_id: string;
  model_id: string;
  acc: number;
  macro_f1: number;
  per_class: Record<string, { n: number; recall: number }>;
  n_train: number;
  n_test: number;
  classes: string[];
};
export type LearningCurveAdvice = {
  metric_name: string;
  xs: number[];
  ys: number[];
  advice: { status: "insufficient" | "improving" | "plateau"; slope_per_100: number | null; reason: string };
};
export type LabelingHandoff = {
  handoff_id: string;
  source: string;
  task: string;
  created_at: string;
  status: string;
  n_total: number | null;
  n_annotated: number | null;
};
export type LabelingHandoffsResponse = { items: LabelingHandoff[] };
export type LabelingReadbackResult = {
  handoff_id: string;
  run_id: string;
  round_id: string | null;
  n_labeled: number;
  n_changed: number;
  complete: boolean;
  annotator: string;
  orphan_round: boolean;
  changes: Array<{ filename: string | null; old_label: string | null; new_label: string | null }>;
};
export type ReadbackAuditResponse = {
  summary: { n_events: number; n_images: number; by_new_label: Record<string, number> };
  events: Array<{
    ts: string; sha256: string; filename: string; old_label: string; new_label: string;
    batch_id: string; round_id: string; annotator: string;
  }>;
};

export type FalsePositiveSource = {
  id: string;
  sha256: string;
  name: string;
  kind: "original" | "screenshot" | "reference";
  size: [number, number];
  added_at: string;
};

export type FalsePositiveRegion = {
  id: string;
  source_id: string;
  roi: [number, number, number, number];
  label: string;
  score: number | null;
  prediction_id: string;
  origin: string;
  review: {
    verdict: "pending" | "normal" | "defect" | "disputed" | "insufficient";
    reviewer: string;
    reason: string;
    at: string;
  };
};

export type FalsePositiveSearchRow = {
  id: string;
  sha256: string;
  name: string;
  category: "normal" | "defect" | "history";
  roi: [number, number, number, number];
  similarity: number;
  reason: string;
  label_status: string;
  source_kind: string;
  source: FalsePositiveSource;
  label_conflict: boolean;
  evidence_id: string;
  external_model?: string;
  historical_case?: string;
  historical_revision?: number;
  review?: { verdict: string; reviewer: string; reason: string; at: string };
};

export type FalsePositiveSearchSourceSummary = {
  folder_name?: string;
  n_files?: number;
  n_regions?: number;
  truncated: boolean;
  label_status: string;
  folder_sha256?: string | null;
};

export type FalsePositiveSearch = {
  id: string;
  at: string;
  query_item_id: string;
  query_sha256: string;
  query_roi: [number, number, number, number];
  candidate_scope: "whole_image" | "same_roi";
  recipe: Record<string, unknown>;
  sources: Record<"normal" | "defect" | "history", FalsePositiveSearchSourceSummary>;
  groups: Record<"normal" | "defect" | "history", FalsePositiveSearchRow[]>;
  skipped: Array<{ name: string; category: string; reason: string }>;
  skipped_total: number;
  top_k: number;
  limit_per_source: number;
};

export type FalsePositiveCandidate = {
  id: string;
  source: FalsePositiveSource;
  roi: [number, number, number, number];
  reason: string;
  reference_category: "normal" | "defect" | "history";
  evidence_id: string;
  similarity: number;
  label_status: string;
  review: FalsePositiveRegion["review"];
  withdrawn?: boolean;
};

export type FalsePositiveClaim = {
  id: string;
  kind: "fact" | "hypothesis" | "action";
  text: string;
  refs: string[];
  at: string;
  updated_at?: string;
  authorship: "engineer";
  evidence_id?: string;
  needs_review?: boolean;
  review_reason?: string;
};

export type FalsePositiveBatch = {
  id: string;
  candidate_ids: string[];
  round_id: string;
  at: string;
  case_revision: number;
  training_export: false;
  unique_images: number;
};

export type FalsePositiveExternalProfile = {
  schema_version: number;
  id: string;
  revision: number;
  latest_revision: number;
  name: string;
  created_at: string;
  asset: { sha256: string; name: string; format: "onnx"; size: number };
  config: Record<string, unknown>;
};

export type FalsePositiveExternalRun = {
  id: string;
  at: string;
  case_id: string;
  source_id: string;
  source_context_id: string;
  source_kind: "original" | "screenshot" | "reference";
  profile_id: string;
  profile_revision: number;
  profile_name: string;
  profile_asset_sha256: string;
  image_sha256: string;
  contract: Record<string, unknown>;
  contract_id: string;
  runtime: { name: string; version: string; provider: string };
  input_sha256: string;
  output_sha256: string;
  input_shape: number[];
  output_shape: number[];
  prediction: {
    task: "classification" | "detection";
    alarm: boolean;
    class?: string;
    score?: number;
    decision?: string;
    scores?: Record<string, number>;
    detections: Array<{ class: string; score: number; box: [number, number, number, number]; output_index?: number }>;
    candidates?: Array<{ class: string; score: number; box: [number, number, number, number]; output_index?: number }>;
    candidate_count?: number;
    candidate_json_truncated?: boolean;
    raw_candidates_available?: boolean;
    nms_applied?: boolean;
  };
  geometry: Record<string, unknown>;
  parity: { status: "matched" | "mismatched" | "unavailable"; reasons: string[]; score_atol: number; box_atol: number };
  evidence_id: string;
};

export type FalsePositiveCase = {
  schema_version: number;
  id: string;
  revision: number;
  created_at: string;
  updated_at: string;
  title: string;
  description: string;
  sources: FalsePositiveSource[];
  items: FalsePositiveRegion[];
  candidates: FalsePositiveCandidate[];
  searches: FalsePositiveSearch[];
  claims: FalsePositiveClaim[];
  batches: FalsePositiveBatch[];
  external_runs: FalsePositiveExternalRun[];
  reply_status: "draft" | "replied";
  cause_status: "unknown" | "hypothesis";
  improvement_status: string;
  effort_minutes: number;
  external: {
    model_id: string;
    threshold: string;
    raw_output: string;
    replay_status: "not_replayed";
  };
  evidence: Array<Record<string, unknown>>;
};

export type FalsePositiveCasesResponse = { items: FalsePositiveCase[]; total: number };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const base = window.NATIVEAPP_LOCAL_API || "";
  const response = await fetch(base + path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail;
    const message = body?.message || (typeof detail === "string" ? detail : detail?.message);
    throw new Error(message || "Local API request failed (" + response.status + ")");
  }
  return body as T;
}

async function uploadFile<T>(path: string, file: File): Promise<T> {
  const response = await fetch((window.NATIVEAPP_LOCAL_API || "") + path, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: file,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail;
    const message = body?.message || (typeof detail === "string" ? detail : detail?.message);
    throw new Error(message || "ONNX 模型匯入失敗 (" + response.status + ")");
  }
  return body as T;
}

export const api = {
  health: () => request<{ ok: boolean; api_version: string; models_available: number }>("/api/v1/health"),
  models: () => request<{ items: ModelOption[] }>("/api/v1/models"),
  createClassSimilarityJob: (runId: string, input: { model_id: string; k: number; analysis_split: string }) =>
    request<{ job_id: string; run_id: string; state: string }>(
      "/api/v1/runs/" + encodeURIComponent(runId) + "/class-similarity/jobs",
      { method: "POST", body: JSON.stringify(input) },
    ),
  classSimilarityJob: (jobId: string) => request<ClassSimilarityJob>(
    "/api/v1/class-similarity/jobs/" + encodeURIComponent(jobId),
  ),
  cancelClassSimilarityJob: (jobId: string) => request<{ job_id: string; state: string }>(
    "/api/v1/class-similarity/jobs/" + encodeURIComponent(jobId), { method: "DELETE" },
  ),
  classSimilarityEvaluationTemplate: (jobId: string, input: { scope: "all" | "selected"; purpose: "tuning" | "heldout"; selected_ids: string[] }) =>
    requestBlob("/api/v1/class-similarity/jobs/" + encodeURIComponent(jobId) + "/evaluation/template", input),
  importClassSimilarityEvaluation: (jobId: string, manifest: Record<string, unknown>, policy: ClassSimilarityEvaluationPreview["policy"]) =>
    request<ClassSimilarityEvaluationPreview>("/api/v1/class-similarity/jobs/" + encodeURIComponent(jobId) + "/evaluation", {
      method: "PUT", body: JSON.stringify({ manifest, policy }),
    }),
  previewClassSimilarityEvaluation: (jobId: string, policy: ClassSimilarityEvaluationPreview["policy"]) =>
    request<ClassSimilarityEvaluationPreview>("/api/v1/class-similarity/jobs/" + encodeURIComponent(jobId) + "/evaluation/preview", {
      method: "POST", body: JSON.stringify({ policy }),
    }),
  saveClassSimilarityOperatingPoint: (jobId: string, policy: ClassSimilarityPolicy) =>
    request<ClassSimilarityEvaluationPreview>("/api/v1/class-similarity/jobs/" + encodeURIComponent(jobId) + "/operating-point", {
      method: "POST", body: JSON.stringify({ policy }),
    }),
  classSimilarityReport: (jobId: string, input: { scope: "all" | "candidates" | "selected"; selected_ids: string[]; policy: ClassSimilarityEvaluationPreview["policy"]; format: "csv" | "json" }) =>
    requestBlob("/api/v1/class-similarity/jobs/" + encodeURIComponent(jobId) + "/report", input),
  createAnomalyModelJob: (input: {
    train_folders: string[];
    model_id: string;
    score_mode: "patch" | "object";
    target_res: number;
    sample_n: number;
    label_semantic: "object" | "defect";
    n_min: number;
    object_source: "yolo" | "whole_image";
    class_names?: string[];
  }) => request<{ job_id: string; state: string }>("/api/v1/anomaly/models/jobs", {
    method: "POST",
    body: JSON.stringify(input),
  }),
  createFewshotSampleBankJob: (input: {
    sample_folders: string[];
    model_id: string;
    object_source: "yolo" | "whole_image";
    class_names?: string[];
  }) => request<{ job_id: string; state: string }>("/api/v1/fewshot/sample-banks/jobs", {
    method: "POST",
    body: JSON.stringify(input),
  }),
  fewshotSampleBankJob: (jobId: string) => request<FewshotSampleBankJob>(
    "/api/v1/fewshot/jobs/" + encodeURIComponent(jobId),
  ),
  getFewshotSampleBank: (sampleBankId: string) => request<{ sample_bank: FewshotSampleBank }>(
    "/api/v1/fewshot/sample-banks/" + encodeURIComponent(sampleBankId),
  ),
  createFewshotScanJob: (input: {
    sample_bank_id: string;
    target_folders: string[];
    k: number;
    min_proposal_conf: number;
  }) => request<{ job_id: string; state: string }>("/api/v1/fewshot/scans/jobs", {
    method: "POST",
    body: JSON.stringify(input),
  }),
  fewshotScanJob: (jobId: string) => request<FewshotScanJob>(
    "/api/v1/fewshot/jobs/" + encodeURIComponent(jobId),
  ),
  fewshotCandidatePreviewUrl: (scanId: string, index: number) =>
    (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/fewshot/scans/" + encodeURIComponent(scanId)
      + "/records/" + index + "/preview.jpg",
  addFewshotCandidates: (input: {
    sample_bank_id: string;
    scan_id: string;
    item_ids: string[];
    theta: number;
    class_overrides: Record<string, string>;
  }) => request<{ added: number; skipped: number; sample_bank: FewshotSampleBank }>(
    "/api/v1/fewshot/sample-banks/add-candidates",
    { method: "POST", body: JSON.stringify(input) },
  ),
  exportFewshotCandidates: (input: {
    sample_bank_id: string;
    scan_id: string;
    accepted_item_ids: string[];
    theta: number;
    class_overrides: Record<string, string>;
    output_dir: string;
    copy_images: boolean;
  }) => request<{
    written: number;
    objects: number;
    csv_rows: number;
    out_dir: string;
    images_copied: number;
  }>("/api/v1/fewshot/scans/exports", {
    method: "POST",
    body: JSON.stringify(input),
  }),
  createFewshotMonitorWorkspace: (input: {
    sample_bank_id: string;
    target_folders: string[];
    workspace_dir: string;
    k: number;
  }) => request<FewshotMonitorStatus>("/api/v1/fewshot/monitors/workspaces", {
    method: "POST",
    body: JSON.stringify(input),
  }),
  createFewshotMonitorScanJob: (monitorWorkspaceId: string) => request<{ job_id: string; state: string }>(
    "/api/v1/fewshot/monitors/scans/jobs",
    { method: "POST", body: JSON.stringify({ monitor_workspace_id: monitorWorkspaceId }) },
  ),
  fewshotMonitorScanJob: (jobId: string) => request<FewshotMonitorJob>(
    "/api/v1/fewshot/jobs/" + encodeURIComponent(jobId),
  ),
  fewshotMonitorStatus: (monitorWorkspaceId: string) => request<FewshotMonitorStatus>(
    "/api/v1/fewshot/monitors/workspaces/" + encodeURIComponent(monitorWorkspaceId),
  ),
  fewshotMonitorPreviewUrl: (monitorWorkspaceId: string, index: number) =>
    (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/fewshot/monitors/workspaces/"
      + encodeURIComponent(monitorWorkspaceId) + "/queue/" + index + "/preview.jpg",
  labelFewshotMonitorItem: (monitorWorkspaceId: string, input: {
    item_id: string;
    decision: "defect" | "skip";
    label: string;
  }) => request<{ decision: "defect" | "skip"; item_id: string }>(
    "/api/v1/fewshot/monitors/workspaces/" + encodeURIComponent(monitorWorkspaceId) + "/labels",
    { method: "POST", body: JSON.stringify(input) },
  ),
  fewshotMonitorProfileUrl: (monitorWorkspaceId: string) =>
    (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/fewshot/monitors/workspaces/"
      + encodeURIComponent(monitorWorkspaceId) + "/profile.yaml",
  anomalyJob: (jobId: string) => request<AnomalyModelJob>(
    "/api/v1/anomaly/jobs/" + encodeURIComponent(jobId),
  ),
  anomalyModels: () => request<{ items: AnomalyModel[] }>("/api/v1/anomaly/models"),
  saveAnomalyModel: (modelId: string) => request<{ model: AnomalyModel }>(
    "/api/v1/anomaly/models/" + encodeURIComponent(modelId) + "/save",
    { method: "POST" },
  ),
  loadAnomalyModel: (modelId: string) => request<{ model: AnomalyModel }>(
    "/api/v1/anomaly/models/load",
    { method: "POST", body: JSON.stringify({ model_id: modelId }) },
  ),
  createAnomalyScoreJob: (input: { target_folders: string[]; contamination: number; class_names?: string[] }) =>
    request<{ job_id: string; state: string }>("/api/v1/anomaly/score-jobs", {
      method: "POST",
      body: JSON.stringify(input),
    }),
  anomalyScoreJob: (jobId: string) => request<AnomalyScoreJob>(
    "/api/v1/anomaly/score-jobs/" + encodeURIComponent(jobId),
  ),
  anomalyRunObjects: (runId: string, input: {
    offset?: number;
    limit?: number;
    verdict?: "good" | "bad";
    class_label?: string;
    score_min?: number;
    score_max?: number;
  }) => {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(input)) {
      if (value !== undefined && value !== "") query.set(key, String(value));
    }
    return request<AnomalyObjectPage>(
      "/api/v1/anomaly/runs/" + encodeURIComponent(runId) + "/objects?" + query.toString(),
    );
  },
  previewAnomalyPrelabels: (runId: string, input: AnomalyPrelabelOptions & { offset?: number; limit?: number }) =>
    request<AnomalyPrelabelPreview>(
      "/api/v1/anomaly/runs/" + encodeURIComponent(runId) + "/prelabel/preview",
      { method: "POST", body: JSON.stringify(input) },
    ),
  exportAnomalyPrelabels: (runId: string, input: AnomalyPrelabelOptions & { output_dir: string }) =>
    request<AnomalyPrelabelExport>(
      "/api/v1/anomaly/runs/" + encodeURIComponent(runId) + "/prelabel/export",
      { method: "POST", body: JSON.stringify(input) },
    ),
  reapplyAnomalyRun: (runId: string, input: {
    expected_revision: number;
    contamination: number;
    confirmed: Record<string, "good" | "bad" | "">;
  }) => request<AnomalyRun>(
    "/api/v1/anomaly/runs/" + encodeURIComponent(runId) + "/reapply",
    { method: "POST", body: JSON.stringify(input) },
  ),
  addAnomalyCuration: (runId: string, input: { expected_revision: number; indices: number[] }) =>
    request<AddCurationResponse>(
      "/api/v1/anomaly/runs/" + encodeURIComponent(runId) + "/curation/items",
      { method: "POST", body: JSON.stringify(input) },
    ),
  anomalyObjectImageUrl: (runId: string, index: number, kind: "preview" | "heatmap") => {
    const suffix = kind === "heatmap" ? "heatmap.png" : "preview.jpg";
    return (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/anomaly/runs/" + encodeURIComponent(runId)
      + "/objects/" + index + "/" + suffix;
  },
  createJob: (
    folders: string[],
    modelIds: string[],
    methods: ProjectionMethod[],
    options: {
      mode?: "classifier" | "detector";
      unit?: VisualizeUnit;
      classNames?: string[];
      objectPolicy?: ComparisonObjectPolicy;
      umapReference?: boolean;
      umapRebuild?: boolean;
      folderGrantIds?: Array<string | null>;
    } = {},
  ) =>
    request<{ job_id: string; state: string }>("/api/v1/visualize/jobs", {
      method: "POST",
      body: JSON.stringify({
        folders,
        folder_grant_ids: options.folderGrantIds,
        model_ids: modelIds,
        methods,
        mode: options.mode || "classifier",
        unit: options.unit || "image",
        class_names: options.classNames || [],
        object_policy: options.objectPolicy,
        umap_reference: options.umapReference || false,
        umap_rebuild: options.umapRebuild || false,
      }),
    }),
  rememberFolderGrant: (path: string, grantId: string) => {
    nativeFolderGrants.set(path, grantId);
  },
  folderGrantIds: (paths: string[]) => paths.map((path) => nativeFolderGrants.get(path) || null),
  evaluateDetections: (input: {
    folders: string[];
    folderGrantIds: Array<string | null>;
    predictionCsv: string;
    consensusCsv?: string | null;
    iouThresh: number;
    confThresh: number;
    classAware: boolean;
  }) => request<DetectionEvaluation>("/api/v1/evaluations", {
    method: "POST",
    body: JSON.stringify({
      folders: input.folders,
      folder_grant_ids: input.folderGrantIds,
      prediction_csv: input.predictionCsv,
      consensus_csv: input.consensusCsv,
      iou_thresh: input.iouThresh,
      conf_thresh: input.confThresh,
      class_aware: input.classAware,
    }),
  }),
  demoDetectionEvaluation: () => request<DetectionEvaluation>("/api/v1/evaluations/demo", {
    method: "POST",
  }),
  evaluationEscapesCsvUrl: (evaluationId: string) =>
    (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/evaluations/" + encodeURIComponent(evaluationId) + "/escapes.csv",
  evaluationEscapeThumbnailUrl: (evaluationId: string, index: number) =>
    (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/evaluations/" + encodeURIComponent(evaluationId)
      + "/escapes/" + index + "/thumbnail.jpg",
  job: (jobId: string) => request<JobStatus>("/api/v1/jobs/" + encodeURIComponent(jobId)),
  cancel: (jobId: string) => request<{ job_id: string; state: string }>(
    "/api/v1/jobs/" + encodeURIComponent(jobId),
    { method: "DELETE" },
  ),
  run: (runId: string) => request<VisualizationRun>(
    "/api/v1/runs/" + encodeURIComponent(runId),
  ),
  createDiagnosticReference: (runId: string, recordIndex: number, modelId: string) =>
    request<DiagnosticReference>("/api/v1/diagnostic-references", {
      method: "POST",
      body: JSON.stringify({ run_id: runId, record_index: recordIndex, model_id: modelId }),
    }),
  diagnosticReference: (contextId: string, currentRunId: string | null) => {
    const current = currentRunId === null ? "" : currentRunId;
    return request<DiagnosticReference>(
      "/api/v1/diagnostic-references/" + encodeURIComponent(contextId) +
        "?current_run_id=" + encodeURIComponent(current),
    );
  },
  diagnosticHeadStatus: (contextId: string, resolution: 224 | 336 | 448, currentRunId: string | null) => {
    const current = currentRunId === null ? "" : currentRunId;
    return request<DiagnosticHeadStatus>(
      "/api/v1/diagnostic-references/" + encodeURIComponent(contextId) +
        "/heads/" + resolution + "?current_run_id=" + encodeURIComponent(current),
    );
  },
  buildDiagnosticHead: (contextId: string, resolution: 224 | 336 | 448, currentRunId: string) =>
    request<DiagnosticHeadStatus>(
      "/api/v1/diagnostic-references/" + encodeURIComponent(contextId) + "/heads",
      {
        method: "POST",
        body: JSON.stringify({ current_run_id: currentRunId, resolution }),
      },
    ),
  createDiagnosticExplanation: (
    contextId: string,
    input: DiagnosticExplanationInput,
  ) => request<DiagnosticExplanationResult>(
    "/api/v1/diagnostic-references/" + encodeURIComponent(contextId) + "/explanations",
    { method: "POST", body: JSON.stringify(input) },
  ),
  createCoverageJob: (
    folders: string[],
    modelId: string,
    unit: VisualizeUnit,
    method: ProjectionMethod,
    dimension: 2 | 3,
    k: number,
  ) => request<{ job_id: string; state: string }>("/api/v1/coverage/jobs", {
    method: "POST",
    body: JSON.stringify({ folders, model_id: modelId, unit, method, dimension, k }),
  }),
  coverageJob: (jobId: string) =>
    request<JobStatus>("/api/v1/coverage/jobs/" + encodeURIComponent(jobId)),
  cancelCoverage: (jobId: string) => request<{ job_id: string; state: string }>(
    "/api/v1/coverage/jobs/" + encodeURIComponent(jobId), { method: "DELETE" },
  ),
  createCoverageCandidateJob: (runId: string, folders: string[]) =>
    request<{ job_id: string; state: string }>(
      "/api/v1/coverage/runs/" + encodeURIComponent(runId) + "/candidates/jobs",
      { method: "POST", body: JSON.stringify({ folders }) },
    ),
  coverageCandidateJob: (jobId: string) =>
    request<JobStatus>("/api/v1/coverage/candidate-jobs/" + encodeURIComponent(jobId)),
  cancelCoverageCandidateJob: (jobId: string) =>
    request<{ job_id: string; state: string }>(
      "/api/v1/coverage/candidate-jobs/" + encodeURIComponent(jobId), { method: "DELETE" },
    ),
  coverageCandidateView: (runId: string, input: CoverageCandidateViewInput) =>
    request<CoverageCandidateView>(
      "/api/v1/coverage/runs/" + encodeURIComponent(runId) + "/candidate-view",
      { method: "POST", body: JSON.stringify(input) },
    ),
  addCoverageCandidatesToCuration: (
    runId: string,
    indices: number[],
    source: "reference" | "reference_class" | "gap_filler",
    scores: Record<number, number>,
  ) => request<AddCoverageCandidateCurationResult>(
    "/api/v1/coverage/runs/" + encodeURIComponent(runId) + "/candidates/curation",
    { method: "POST", body: JSON.stringify({ indices, source, scores }) },
  ),
  coverageRun: (runId: string) => request<CoverageRun>(
    "/api/v1/coverage/runs/" + encodeURIComponent(runId),
  ),
  projectCoverage: (runId: string, method: ProjectionMethod, dimension: 2 | 3) =>
    request<CoverageProjectionResult>(
      "/api/v1/coverage/runs/" + encodeURIComponent(runId) + "/projections",
      { method: "POST", body: JSON.stringify({ method, dimension }) },
    ),
  coverageSparsity: (runId: string, k: number) => request<CoverageSparsityResult>(
    "/api/v1/coverage/runs/" + encodeURIComponent(runId) + "/sparsity",
    { method: "POST", body: JSON.stringify({ k }) },
  ),
  createGroupDiffJob: (input: {
    good_dirs: string[]; bad_dirs: string[]; model_id: string;
    n_perm: 100 | 200 | 500; alpha: 0.01 | 0.05 | 0.1; top_k: number; seed: number;
  }) => request<{ job_id: string; state: string }>("/api/v1/groupdiff/jobs", {
    method: "POST", body: JSON.stringify(input),
  }),
  groupDiffJob: (jobId: string) => request<JobStatus>(
    "/api/v1/groupdiff/jobs/" + encodeURIComponent(jobId),
  ),
  groupDiffRun: (runId: string) => request<GroupDiffRun>(
    "/api/v1/groupdiff/runs/" + encodeURIComponent(runId),
  ),
  groupDiffBaseUrl: (runId: string, side: "good" | "bad", index: number, overlay: boolean, alpha: number) => {
    const query = new URLSearchParams({ overlay: String(overlay), alpha: String(alpha) });
    return (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/groupdiff/runs/"
      + encodeURIComponent(runId) + "/bases/" + side + "/" + index + "?" + query.toString();
  },
  groupDiffRegionImageUrl: (runId: string, rank: number, side: "good" | "bad", index: number) =>
    (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/groupdiff/runs/" + encodeURIComponent(runId)
      + "/regions/" + rank + "/images/" + side + "/" + index + ".jpg",
  groupDiffPca: (runId: string) => request<GroupDiffPca>(
    "/api/v1/groupdiff/runs/" + encodeURIComponent(runId) + "/pca",
  ),
  exportGroupDiff: (runId: string, outputDir: string) => request<{ count: number; files: string[] }>(
    "/api/v1/groupdiff/runs/" + encodeURIComponent(runId) + "/exports",
    { method: "POST", body: JSON.stringify({ output_dir: outputDir }) },
  ),
  coverageCompleteness: (runId: string, input: CoverageCompletenessInput) =>
    request<CoverageCompletenessResult>(
      "/api/v1/coverage/runs/" + encodeURIComponent(runId) + "/completeness",
      { method: "POST", body: JSON.stringify(input) },
    ),
  mineCoverageCompletenessCandidates: (
    runId: string,
    input: { folders: string[]; axis_x: CompletenessAxis; axis_y: CompletenessAxis; bins: number; x: number; y: number },
  ) => request<CoverageCompletenessCandidateResult>(
    "/api/v1/coverage/runs/" + encodeURIComponent(runId) + "/completeness/candidates",
    { method: "POST", body: JSON.stringify(input) },
  ),
  coverageCompletenessCandidateThumbnailUrl: (runId: string, index: number) =>
    (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/coverage/runs/"
      + encodeURIComponent(runId) + "/completeness/candidates/" + index + "/thumbnail",
  coverageCompletenessCandidatesCsv: (runId: string) => requestGetBlob(
    "/api/v1/coverage/runs/" + encodeURIComponent(runId) + "/completeness/candidates.csv",
  ),
  coverageObjectView: (runId: string, input: CoverageObjectViewInput) =>
    request<CoverageObjectView>(
      "/api/v1/coverage/runs/" + encodeURIComponent(runId) + "/object-view",
      { method: "POST", body: JSON.stringify(input) },
    ),
  coverageCandidateThumbnailUrl: (runId: string, index: number) =>
    (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/coverage/runs/"
      + encodeURIComponent(runId) + "/candidates/" + index + "/thumbnail",
  coverageThumbnailUrl: (runId: string, index: number) =>
    (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/coverage/runs/"
      + encodeURIComponent(runId) + "/images/" + index + "/thumbnail",
  rankings: (
    runId: string,
    input: { model_id: string; metric: RankingMetric; label_filter?: string; split_filter?: string },
  ) => {
    const query = new URLSearchParams({ model_id: input.model_id, metric: input.metric });
    if (input.label_filter) query.set("label_filter", input.label_filter);
    if (input.split_filter) query.set("split_filter", input.split_filter);
    return request<RankingResult>(
      "/api/v1/runs/" + encodeURIComponent(runId) + "/rankings?" + query.toString(),
    );
  },
  rankingsCsv: (
    runId: string,
    input: { model_id: string; metric: RankingMetric; label_filter?: string; split_filter?: string },
  ) => {
    const query = new URLSearchParams({ model_id: input.model_id, metric: input.metric });
    if (input.label_filter) query.set("label_filter", input.label_filter);
    if (input.split_filter) query.set("split_filter", input.split_filter);
    return requestGetBlob(
      "/api/v1/runs/" + encodeURIComponent(runId) + "/rankings.csv?" + query.toString(),
    );
  },
  kmeansBlocks: (runId: string, input: KMeansBlocksInput) => request<KMeansBlocks>(
    "/api/v1/runs/" + encodeURIComponent(runId) + "/blocks",
    { method: "POST", body: JSON.stringify(input) },
  ),
  similarImages: (runId: string, input: { model_id: string; query_index: number; k: number }) =>
    request<SimilarSearchResult>("/api/v1/runs/" + encodeURIComponent(runId) + "/similar", {
      method: "POST", body: JSON.stringify(input),
    }),
  similarImagesCsv: async (
    runId: string,
    input: { model_id: string; query_index: number; k: number },
  ): Promise<Blob> => {
    const response = await fetch((window.NATIVEAPP_LOCAL_API || "") +
      "/api/v1/runs/" + encodeURIComponent(runId) + "/similar.csv", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input),
      });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new Error(body?.message || body?.detail || "Could not export the similar-image results.");
    }
    return response.blob();
  },
  textSearch: (runId: string, input: { model_id: string; query: string; k: number }) =>
    request<TextSearchResult>("/api/v1/runs/" + encodeURIComponent(runId) + "/text-search", {
      method: "POST", body: JSON.stringify(input),
    }),
  textSearchCsv: (runId: string, input: { model_id: string; query: string; k: number }) =>
    requestBlob("/api/v1/runs/" + encodeURIComponent(runId) + "/text-search.csv", input),
  scanDuplicates: (runId: string, input: DuplicateScanInput) =>
    request<DuplicateScanResult>("/api/v1/runs/" + encodeURIComponent(runId) + "/duplicates", {
      method: "POST", body: JSON.stringify(input),
    }),
  sendDuplicatePairsToLabeling: (
    runId: string,
    input: DuplicateScanInput & { include_handled?: boolean; new_round?: boolean; round_id?: string | null },
  ) => request<DuplicateLabelingHandoffResult>(
    "/api/v1/runs/" + encodeURIComponent(runId) + "/duplicates/labeling-handoff", {
      method: "POST", body: JSON.stringify(input),
    },
  ),
  duplicatePairsCsv: (runId: string, input: DuplicateScanInput) =>
    requestBlob("/api/v1/runs/" + encodeURIComponent(runId) + "/duplicates.csv", input),
  createComparisonJob: (input: ComparisonJobInput) =>
    request<{ job_id: string; state: string }>("/api/v1/compare/jobs", {
      method: "POST",
      body: JSON.stringify(input),
    }),
  comparisonJob: (jobId: string) => request<JobStatus>("/api/v1/compare/jobs/" + encodeURIComponent(jobId)),
  cancelComparison: (jobId: string) => request<{ job_id: string; state: string }>(
    "/api/v1/compare/jobs/" + encodeURIComponent(jobId), { method: "DELETE" },
  ),
  comparison: (runId: string) => request<ComparisonRun>(
    "/api/v1/comparisons/" + encodeURIComponent(runId),
  ),
  addComparisonCuration: (runId: string, selections: ComparisonCurationSelection[], reason = "") =>
    request<AddCurationResponse>("/api/v1/comparisons/" + encodeURIComponent(runId) + "/curation/items", {
      method: "POST",
      body: JSON.stringify({ selections, reason }),
    }),
  projectComparison: (
    runId: string,
    label: string,
    method: ProjectionMethod,
    dimension: 2 | 3,
    filters?: { min_confidence?: number; max_confidence?: number; min_short_px?: number },
  ) => request<ComparisonProjection>(
    "/api/v1/comparisons/" + encodeURIComponent(runId) + "/projections",
    {
      method: "POST",
      body: JSON.stringify({ label, method, dimension, ...filters }),
    },
  ),
  comparisonImageUrl: (runId: string, side: "a" | "b", index: number) =>
    (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/comparisons/" + encodeURIComponent(runId)
      + "/images/" + side + "/" + index,
  curationItems: () => request<CurationItemsResponse>("/api/v1/curation/items"),
  previewCurationHandoff: (itemIds: string[], includeHandled = false) =>
    request<CurationHandoffPreview>("/api/v1/curation/labeling-handoff/preview", {
      method: "POST",
      body: JSON.stringify({ item_ids: itemIds, include_handled: includeHandled }),
    }),
  sendCurationToLabeling: (input: {
    item_ids: string[]; include_handled?: boolean; new_round?: boolean; round_id?: string | null;
  }) => request<CurationHandoffResult>("/api/v1/curation/labeling-handoff/send", {
    method: "POST",
    body: JSON.stringify(input),
  }),
  activeLearningRounds: () => request<ActiveLearningRoundsResponse>("/api/v1/rounds"),
  attachEvaluationToRound: (roundId: string, evaluationId: string) =>
    request<{ round_id: string; attached: boolean; rounds: ActiveLearningRound[] }>(
      "/api/v1/rounds/" + encodeURIComponent(roundId) + "/evaluations", {
        method: "POST",
        body: JSON.stringify({ evaluation_id: evaluationId }),
      }),
  attachRoundMetric: (roundId: string, input: { name: string; value: number; source?: string; note?: string }) =>
    request<ActiveLearningRoundsResponse>("/api/v1/rounds/" + encodeURIComponent(roundId) + "/metrics", {
      method: "POST",
      body: JSON.stringify(input),
    }),
  runRoundProbe: (roundId: string, runId: string) =>
    request<ProbeEvaluationResult>("/api/v1/rounds/" + encodeURIComponent(roundId) + "/probe", {
      method: "POST",
      body: JSON.stringify({ run_id: runId }),
    }),
  activeLearningCurveAdvice: (metricName: string) =>
    request<LearningCurveAdvice>("/api/v1/rounds/learning-curve/advice?metric_name=" + encodeURIComponent(metricName)),
  labelingHandoffs: () => request<LabelingHandoffsResponse>("/api/v1/labeling/handoffs"),
  applyLabelingReadback: (handoffId: string, runId: string) =>
    request<LabelingReadbackResult>("/api/v1/labeling/handoffs/" + encodeURIComponent(handoffId) + "/readback", {
      method: "POST",
      body: JSON.stringify({ run_id: runId }),
    }),
  readbackAudit: () => request<ReadbackAuditResponse>("/api/v1/readback/audit"),
  readbackAuditCsv: () => requestGetBlob("/api/v1/readback/audit.csv"),
  curationSubsetPreview: (input: CurationSubsetOptions) =>
    request<CurationSubsetPreview>("/api/v1/curation/subset/preview", {
      method: "POST",
      body: JSON.stringify(input),
    }),
  curationSubsetExport: (input: CurationSubsetOptions & { preview_token: string }) =>
    request<CurationSubsetExport>("/api/v1/curation/subset/export", {
      method: "POST",
      body: JSON.stringify(input),
    }),
  addCuration: (runId: string, indices: number[], source = "manual", reason = "") =>
    request<AddCurationResponse>("/api/v1/runs/" + encodeURIComponent(runId) + "/curation/items", {
      method: "POST",
      body: JSON.stringify({ indices, source, reason }),
    }),
  setCurationReason: (itemIds: string[], reason: string) =>
    request<CurationItemsResponse & { updated: number }>("/api/v1/curation/items/reason", {
      method: "PATCH",
      body: JSON.stringify({ item_ids: itemIds, reason }),
    }),
  removeCurationItem: (itemId: string) => request<CurationItemsResponse & { removed: boolean }>(
    "/api/v1/curation/items/" + encodeURIComponent(itemId), { method: "DELETE" },
  ),
  clearCuration: () => request<{ cleared: number; items: CurationItem[]; total: number }>(
    "/api/v1/curation/items", { method: "DELETE" },
  ),
  curationExport: (format: "csv" | "json" | "zip") =>
    requestGetBlob("/api/v1/curation." + format),
  curationThumbnailUrl: (itemId: string) =>
    (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/curation/items/" + encodeURIComponent(itemId) + "/thumbnail",
  falsePositiveCases: () => request<FalsePositiveCasesResponse>("/api/v1/fp-cases"),
  externalModelProfiles: () => request<{ items: FalsePositiveExternalProfile[] }>("/api/v1/external-model-profiles"),
  uploadExternalModelAsset: (file: File) => uploadFile<{
    sha256: string; name: string; format: "onnx"; size: number; created_at: string;
  }>("/api/v1/external-model-assets?filename=" + encodeURIComponent(file.name), file),
  createExternalModelProfile: (input: {
    name: string; asset_sha256: string; config: Record<string, unknown>;
  }) => request<FalsePositiveExternalProfile>("/api/v1/external-model-profiles", {
    method: "POST",
    body: JSON.stringify(input),
  }),
  reviseExternalModelProfile: (profileId: string, input: {
    expected_revision: number; name?: string; asset_sha256?: string; config?: Record<string, unknown>;
  }) => request<FalsePositiveExternalProfile>(
    "/api/v1/external-model-profiles/" + encodeURIComponent(profileId), {
      method: "PATCH",
      body: JSON.stringify(input),
    },
  ),
  runExternalModel: (caseId: string, input: {
    expected_revision: number; source_id: string; profile_id: string; profile_revision: number;
    reference?: Record<string, unknown>; score_atol: number; box_atol: number;
  }) => request<{ case: FalsePositiveCase; run: FalsePositiveExternalRun }>(
    "/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/external-runs", {
      method: "POST",
      body: JSON.stringify(input),
    },
  ),
  externalModelInputPreviewUrl: (caseId: string, runId: string) =>
    (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/fp-cases/" + encodeURIComponent(caseId)
      + "/external-runs/" + encodeURIComponent(runId) + "/input-preview.png",
  externalModelEvidence: (caseId: string, runId: string) => requestGetBlob(
    "/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/external-runs/"
      + encodeURIComponent(runId) + "/evidence.json",
  ),
  externalModelEvidenceZip: (caseId: string, runId: string) => requestGetBlob(
    "/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/external-runs/"
      + encodeURIComponent(runId) + "/evidence.zip",
  ),
  falsePositiveCase: (caseId: string) =>
    request<FalsePositiveCase>("/api/v1/fp-cases/" + encodeURIComponent(caseId)),
  createFalsePositiveCase: (input: {
    source_name: string;
    source_kind: "original" | "screenshot";
    source_base64: string;
    title: string;
    description: string;
    external_model_id: string;
    threshold: string;
    raw_output: string;
  }) => request<FalsePositiveCase>("/api/v1/fp-cases", {
    method: "POST",
    body: JSON.stringify(input),
  }),
  addFalsePositiveSource: (caseId: string, expectedRevision: number, input: {
    source_name: string;
    source_kind: "original" | "screenshot";
    source_base64: string;
  }) => request<FalsePositiveCase>("/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/sources", {
    method: "POST",
    body: JSON.stringify({ expected_revision: expectedRevision, ...input }),
  }),
  addFalsePositiveRegion: (caseId: string, expectedRevision: number, input: {
    source_id: string;
    roi: number[];
    label: string;
    score: number | null;
  }) => request<{ case: FalsePositiveCase; item_id: string }>(
    "/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/regions", {
      method: "POST",
      body: JSON.stringify({ expected_revision: expectedRevision, ...input }),
    },
  ),
  importFalsePositiveYolo: (caseId: string, expectedRevision: number, input: {
    source_id: string;
    filename: string;
    content: string;
  }) => request<{ case: FalsePositiveCase; imported: number }>(
    "/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/predictions/yolo", {
      method: "POST",
      body: JSON.stringify({ expected_revision: expectedRevision, ...input }),
    },
  ),
  reviewFalsePositiveRegion: (caseId: string, itemId: string, expectedRevision: number, input: {
    verdict: FalsePositiveRegion["review"]["verdict"];
    reviewer: string;
    reason: string;
  }) => request<FalsePositiveCase>(
    "/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/regions/" + encodeURIComponent(itemId) + "/review", {
      method: "PATCH",
      body: JSON.stringify({ expected_revision: expectedRevision, ...input }),
    },
  ),
  searchFalsePositiveCase: (caseId: string, input: {
    expected_revision: number;
    item_id: string;
    normal_folder: string;
    defect_folder: string;
    candidate_scope: "whole_image" | "same_roi";
    limit: number;
    top_k: number;
    include_history: boolean;
  }) => request<{ case: FalsePositiveCase; search_id: string }>(
    "/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/searches", {
      method: "POST",
      body: JSON.stringify(input),
    },
  ),
  addFalsePositiveSearchCandidate: (
    caseId: string,
    searchId: string,
    resultId: string,
    expectedRevision: number,
    category: FalsePositiveSearchRow["category"],
  ) => request<{ case: FalsePositiveCase; added: boolean }>(
    "/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/searches/"
      + encodeURIComponent(searchId) + "/results/" + encodeURIComponent(resultId) + "/candidate", {
      method: "POST",
      body: JSON.stringify({ expected_revision: expectedRevision, category }),
    },
  ),
  upsertFalsePositiveClaim: (caseId: string, expectedRevision: number, input: {
    claim_id?: string;
    kind: FalsePositiveClaim["kind"];
    text: string;
    evidence_ids: string[];
  }) => request<{ case: FalsePositiveCase; claim_id: string }>(
    "/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/claims", {
      method: "PUT",
      body: JSON.stringify({ expected_revision: expectedRevision, ...input }),
    },
  ),
  setFalsePositiveReplyStatus: (caseId: string, expectedRevision: number, status: "draft" | "replied") =>
    request<{ case: FalsePositiveCase; changed: boolean }>(
      "/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/reply-status", {
        method: "PATCH",
        body: JSON.stringify({ expected_revision: expectedRevision, status }),
      },
    ),
  falsePositiveCaseReport: (caseId: string, format: "md" | "json") => requestGetBlob(
    "/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/report." + format,
  ),
  reviewFalsePositiveCandidate: (caseId: string, candidateId: string, expectedRevision: number, input: {
    verdict: FalsePositiveRegion["review"]["verdict"];
    reviewer: string;
    reason: string;
  }) => request<FalsePositiveCase>(
    "/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/candidates/"
      + encodeURIComponent(candidateId) + "/review", {
      method: "PATCH",
      body: JSON.stringify({ expected_revision: expectedRevision, ...input }),
    },
  ),
  prepareFalsePositiveReviewBatch: (caseId: string, expectedRevision: number) =>
    request<{ case: FalsePositiveCase; batch_id: string; round_id: string; counts: Record<string, number> }>(
      "/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/batches", {
        method: "POST",
        body: JSON.stringify({ expected_revision: expectedRevision }),
      },
    ),
  falsePositiveReviewBatchManifest: (caseId: string, batchId: string) => requestGetBlob(
    "/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/batches/"
      + encodeURIComponent(batchId) + "/manifest",
  ),
  falsePositiveReviewBatchZip: (caseId: string, batchId: string) => requestGetBlob(
    "/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/batches/"
      + encodeURIComponent(batchId) + "/archive.zip",
  ),
  applyFalsePositiveReviewReturn: (caseId: string, batchId: string, expectedRevision: number, manifest: unknown) =>
    request<{ case: FalsePositiveCase; applied: number; counts: Record<string, number>; training_ready_images: 0 }>(
      "/api/v1/fp-cases/" + encodeURIComponent(caseId) + "/batches/"
        + encodeURIComponent(batchId) + "/return", {
        method: "POST",
        body: JSON.stringify({ expected_revision: expectedRevision, manifest }),
      },
    ),
  falsePositiveCandidateImageUrl: (caseId: string, candidateId: string) =>
    (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/fp-cases/" + encodeURIComponent(caseId)
      + "/candidates/" + encodeURIComponent(candidateId) + "/image",
  falsePositiveSourceImageUrl: (caseId: string, sourceId: string) =>
    (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/fp-cases/" + encodeURIComponent(caseId)
      + "/sources/" + encodeURIComponent(sourceId) + "/image",
  falsePositiveSearchImageUrl: (caseId: string, searchId: string, resultId: string) =>
    (window.NATIVEAPP_LOCAL_API || "") + "/api/v1/fp-cases/" + encodeURIComponent(caseId)
      + "/searches/" + encodeURIComponent(searchId) + "/results/" + encodeURIComponent(resultId) + "/image",
};

async function requestBlob(path: string, body: unknown): Promise<Blob> {
  const response = await fetch((window.NATIVEAPP_LOCAL_API || "") + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.message || payload?.detail || "Could not export the results.");
  }
  return response.blob();
}

async function requestGetBlob(path: string): Promise<Blob> {
  const response = await fetch((window.NATIVEAPP_LOCAL_API || "") + path);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.message || payload?.detail || "Could not export the results.");
  }
  return response.blob();
}

export type HostFolderResult = {
  canceled?: boolean;
  path?: string;
  paths?: string[];
  display_name?: string;
  grant_id?: string;
};

declare global {
  interface Window {
    NATIVEAPP_LOCAL_API?: string;
    cimHost?: {
      chooseFile?: () => Promise<HostFolderResult>;
    };
  }
}
