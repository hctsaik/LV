export type ModelOption = {
  model_id: string;
  name: string;
  available: boolean;
};

export type ProjectionMethod = "pca" | "tsne" | "umap" | "lda" | "sumap";

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

export type VisualizationRun = {
  source_mode: "image_classifier" | "object_detector";
  class_names: string[];
  class_source: "classifier_folders" | "selected_classes_file" | "auto_classes_txt" | "auto_data_yaml" | "manual";
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
  warnings: string[];
};

export type KMeansBlock = { block_id: number; count: number; indices: number[] };
export type KMeansCenter = { block_id: number; x: number; y: number; z: number };
export type KMeansBlocks = {
  model_id: string;
  method: ProjectionMethod;
  requested_k: number;
  effective_k: number;
  blocks: KMeansBlock[];
  centers: KMeansCenter[];
};
export type BlockProjectionPoint = { index: number; x: number; y: number };
export type BlockProjection = {
  model_id: string;
  method: ProjectionMethod;
  projection_source: "embedding_pca" | "projection";
  points: BlockProjectionPoint[];
};

export type ComparePoint = {
  group: "a" | "b";
  index: number;
  x: number;
  y: number;
  z: number;
};

export type CompareView = { method: ProjectionMethod; dimension: 2 | 3; points: ComparePoint[] };
export type CompareRecord = { index: number; filename: string };
export type CompareRun = {
  folder_a: string;
  folder_b: string;
  model_id: string;
  drift: number;
  records_a: CompareRecord[];
  records_b: CompareRecord[];
  warnings: string[];
  overlap_count: number;
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
      umapReference?: boolean;
      rebuildUmapReference?: boolean;
      mode?: "image_classifier" | "object_detector";
      classesFileText?: string | null;
      classNames?: string[];
    } = {},
  ) =>
    request<{ job_id: string; state: string }>("/api/v1/visualize/jobs", {
      method: "POST",
      body: JSON.stringify({
        folders,
        model_ids: modelIds,
        methods,
        mode: options.mode ?? "image_classifier",
        classes_file_text: options.classesFileText ?? null,
        class_names: options.classNames ?? [],
        umap_reference: options.umapReference ?? false,
        rebuild_umap_reference: options.rebuildUmapReference ?? false,
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
  runBlocks: (
    runId: string,
    modelId: string,
    method: ProjectionMethod,
    indices: number[],
    k: number,
  ) => request<KMeansBlocks>("/api/v1/runs/" + encodeURIComponent(runId) + "/blocks", {
    method: "POST",
    body: JSON.stringify({ model_id: modelId, method, indices, k }),
  }),
  blockProjection: (
    runId: string,
    modelId: string,
    method: ProjectionMethod,
    indices: number[],
  ) => request<BlockProjection>("/api/v1/runs/" + encodeURIComponent(runId) + "/block-projection", {
    method: "POST",
    body: JSON.stringify({ model_id: modelId, method, indices }),
  }),
  createCompareJob: (foldersA: string[], foldersB: string[], modelId: string) =>
    request<{ job_id: string; state: string }>("/api/v1/compare/jobs", {
      method: "POST",
      body: JSON.stringify({ folders_a: foldersA, folders_b: foldersB, model_id: modelId }),
    }),
  compareJob: (jobId: string) => request<JobStatus>(
    "/api/v1/compare/jobs/" + encodeURIComponent(jobId),
  ),
  cancelCompare: (jobId: string) => request<{ job_id: string; state: string }>(
    "/api/v1/compare/jobs/" + encodeURIComponent(jobId),
    { method: "DELETE" },
  ),
  compareRun: (runId: string) => request<CompareRun>(
    "/api/v1/compare/runs/" + encodeURIComponent(runId),
  ),
  compareProjection: (runId: string, method: ProjectionMethod, dimension: 2 | 3) =>
    request<CompareView>("/api/v1/compare/runs/" + encodeURIComponent(runId) + "/projections", {
      method: "POST",
      body: JSON.stringify({ method, dimension }),
    }),
  compareImageUrl: (runId: string, group: "a" | "b", index: number) => {
    const base = window.NATIVEAPP_LOCAL_API || "";
    return base + "/api/v1/compare/runs/" + encodeURIComponent(runId) + "/images/" + group + "/" + index;
  },
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
