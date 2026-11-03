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
  object_index?: number;
  class_id?: number | null;
  bbox?: [number, number, number, number];
  score?: number | null;
};

export type ObjectPolicy = {
  pad: number;
  target_res: 168 | 196 | 224 | 280 | 336 | 392 | 448;
  head: "cls" | "meanpool";
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
  source_granularity: "whole_image" | "object";
  class_names: string[];
  class_source: "classifier_folders" | "selected_classes_file" | "auto_classes_txt" | "auto_data_yaml" | "manual";
  object_policy?: ObjectPolicy;
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
export type SimilarResult = RunRecord & { distance: number };
export type SimilarSearch = {
  model_id: string;
  query_index: number;
  results: SimilarResult[];
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
      granularity?: "whole_image" | "object";
      classesFileText?: string | null;
      classNames?: string[];
      objectPolicy?: ObjectPolicy;
    } = {},
  ) =>
    request<{ job_id: string; state: string }>("/api/v1/visualize/jobs", {
      method: "POST",
      body: JSON.stringify({
        folders,
        model_ids: modelIds,
        methods,
        mode: options.mode ?? "image_classifier",
        granularity: options.granularity ?? "whole_image",
        classes_file_text: options.classesFileText ?? null,
        class_names: options.classNames ?? [],
        object_policy: options.objectPolicy ?? { pad: 0.12, target_res: 224, head: "cls" },
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
  runImageUrl: (
    runId: string,
    index: number,
    options: { overlay?: boolean; view?: "selected" | "source" } = {},
  ) => {
    const base = window.NATIVEAPP_LOCAL_API || "";
    const query = new URLSearchParams({
      overlay: options.overlay ? "true" : "false",
      view: options.view ?? "selected",
    });
    return base + "/api/v1/runs/" + encodeURIComponent(runId) + "/images/" + index + "?" + query.toString();
  },
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
  searchSimilar: (runId: string, modelId: string, queryIndex: number, k: number) =>
    request<SimilarSearch>("/api/v1/runs/" + encodeURIComponent(runId) + "/similar", {
      method: "POST",
      body: JSON.stringify({ model_id: modelId, query_index: queryIndex, k }),
    }),
  similarCsvUrl: (runId: string, modelId: string, queryIndex: number, k: number) => {
    const base = window.NATIVEAPP_LOCAL_API || "";
    const query = new URLSearchParams({ model_id: modelId, query_index: String(queryIndex), k: String(k) });
    return base + "/api/v1/runs/" + encodeURIComponent(runId) + "/similar.csv?" + query.toString();
  },
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
