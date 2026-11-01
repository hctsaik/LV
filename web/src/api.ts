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
  folders_b: string[];
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

export const api = {
  health: () => request<{ ok: boolean; api_version: string; models_available: number }>("/api/v1/health"),
  models: () => request<{ items: ModelOption[] }>("/api/v1/models"),
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
    } = {},
  ) =>
    request<{ job_id: string; state: string }>("/api/v1/visualize/jobs", {
      method: "POST",
      body: JSON.stringify({
        folders,
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
  job: (jobId: string) => request<JobStatus>("/api/v1/jobs/" + encodeURIComponent(jobId)),
  cancel: (jobId: string) => request<{ job_id: string; state: string }>(
    "/api/v1/jobs/" + encodeURIComponent(jobId),
    { method: "DELETE" },
  ),
  run: (runId: string) => request<VisualizationRun>(
    "/api/v1/runs/" + encodeURIComponent(runId),
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
