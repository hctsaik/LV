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
  kmeansBlocks: (runId: string, input: KMeansBlocksInput) => request<KMeansBlocks>(
    "/api/v1/runs/" + encodeURIComponent(runId) + "/blocks",
    { method: "POST", body: JSON.stringify(input) },
  ),
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
};

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
