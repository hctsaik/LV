import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import {
  api,
  type EmbeddingPoint,
  type KMeansBlocks,
  type BlockProjection,
  type CurationCart,
  type CurationCandidate,
  type DuplicateScan,
  type JobStatus,
  type ImageRanking,
  type ModelOption,
  type ProjectionMethod,
  type RankingCriterion,
  type SimilarSearch,
  type TextSearch,
  type VisualizationRun,
} from "./api";
import Compare from "./Compare";
const LazyScatterPlot3D = lazy(() => import("./ScatterPlot3D"));
const LazyBlockSelectionPlot = lazy(() => import("./ScatterPlot3D").then((module) => ({
  default: module.BlockSelectionPlot,
})));

const labelColors = ["#4976d0", "#e08144", "#38a27c", "#9a6cc3", "#c95567", "#718096"];
const projectionOptions: { value: ProjectionMethod; label: string }[] = [
  { value: "pca", label: "PCA" },
  { value: "tsne", label: "t-SNE" },
  { value: "umap", label: "UMAP" },
  { value: "lda", label: "LDA (supervised)" },
  { value: "sumap", label: "Supervised UMAP" },
];
const rankingSortNames: Record<RankingCriterion, string> = {
  spatial: "spatial order",
  outlier: "outlier score",
  disagreement: "label disagreement",
  filename: "filename",
};

function pause(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function colorMap(points: EmbeddingPoint[]) {
  const labels = [...new Set(points.map((point) => point.label))].sort();
  return new Map(labels.map((label, index) => [label, labelColors[index % labelColors.length]]));
}

function spatialSelectionOrder(indices: number[], points: EmbeddingPoint[], rows = 12) {
  const pointByIndex = new Map(points.map((point) => [point.index, point]));
  const usable = indices.filter((index) => pointByIndex.has(index));
  if (!usable.length) return [];
  const ys = usable.map((index) => pointByIndex.get(index)!.y);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const rowFor = (index: number) => yMax === yMin
    ? 0
    : Math.floor(((yMax - pointByIndex.get(index)!.y) / (yMax - yMin)) * (rows - 1e-9));
  return usable.sort((left, right) =>
    rowFor(left) - rowFor(right) || pointByIndex.get(left)!.x - pointByIndex.get(right)!.x,
  );
}

function ScatterPlot2D({ points, selectedIndices, onSelect }: {
  points: EmbeddingPoint[];
  selectedIndices: number[];
  onSelect: (index: number, additive?: boolean) => void;
}) {
  const width = 860;
  const height = 480;
  const padding = 42;
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const xMin = Math.min(...xs, 0);
  const xMax = Math.max(...xs, 0);
  const yMin = Math.min(...ys, 0);
  const yMax = Math.max(...ys, 0);
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1;
  const colors = colorMap(points);
  const selected = new Set(selectedIndices);

  return (
    <div className="plot-frame" data-testid="embedding-plot">
      <svg className="scatter" viewBox={"0 0 " + width + " " + height} role="img" aria-label="Image embedding scatter plot">
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="axis" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="axis" />
        <text x={width / 2} y={height - 8} className="axis-label">Component 1</text>
        <text x={15} y={height / 2} className="axis-label vertical-label">Component 2</text>
        {points.map((point) => {
          const x = padding + ((point.x - xMin) / xSpan) * (width - padding * 2);
          const y = height - padding - ((point.y - yMin) / ySpan) * (height - padding * 2);
          const color = colors.get(point.label) || labelColors[0];
          const active = selected.has(point.index);
          const shape = point.split.toLowerCase() === "test" ? (
            <rect x={-7} y={-7} width={14} height={14} rx={2} fill={color} />
          ) : point.split.toLowerCase() === "valid" ? (
            <path d="M 0 -9 L 9 0 L 0 9 L -9 0 Z" fill={color} />
          ) : (
            <circle r={7} fill={color} />
          );
          return (
            <g
              key={point.index}
              role="button"
              tabIndex={0}
              aria-label={"Select " + point.filename + ", " + point.label + ", " + point.split}
              data-testid="embedding-point"
              data-index={point.index}
              className={"point " + (active ? "point-active" : "")}
              transform={"translate(" + x + " " + y + ")"}
              onClick={(event) => onSelect(point.index, event.shiftKey)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") onSelect(point.index);
              }}
            >
              <title>{point.filename + " · " + point.label + " · " + point.split}</title>
              {active && <circle r={12} className="point-selection-ring" />}
              {shape}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function ScatterPlot({ points, selectedIndices, clusters, onSelect, dimension, viewKey }: {
  points: EmbeddingPoint[];
  selectedIndices: number[];
  clusters: KMeansBlocks | null;
  onSelect: (index: number, additive?: boolean) => void;
  dimension: 2 | 3;
  viewKey: string;
}) {
  if (dimension === 3) {
    return (
      <Suspense fallback={<div className="plot-frame plot-loading" role="status">Loading 3D plot…</div>}>
        <LazyScatterPlot3D points={points} selectedIndices={selectedIndices} clusters={clusters} onSelect={onSelect} viewKey={viewKey} />
      </Suspense>
    );
  }
  return <ScatterPlot2D points={points} selectedIndices={selectedIndices} onSelect={onSelect} />;
}

export default function App() {
  const [activeTool, setActiveTool] = useState<"visualize" | "compare">("visualize");
  const [models, setModels] = useState<ModelOption[]>([]);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  const [selectedMethods, setSelectedMethods] = useState<ProjectionMethod[]>(["sumap"]);
  const [umapReference, setUmapReference] = useState(false);
  const [rebuildUmapReference, setRebuildUmapReference] = useState(false);
  const [activeModelId, setActiveModelId] = useState("");
  const [activeMethod, setActiveMethod] = useState<ProjectionMethod>("sumap");
  const [dimension, setDimension] = useState<2 | 3>(2);
  const [datasetMode, setDatasetMode] = useState<"image_classifier" | "object_detector">("image_classifier");
  const [detectorGranularity, setDetectorGranularity] = useState<"whole_image" | "object">("whole_image");
  const [objectPad, setObjectPad] = useState(0.12);
  const [objectTargetRes, setObjectTargetRes] = useState<168 | 196 | 224 | 280 | 336 | 392 | 448>(224);
  const [objectHead, setObjectHead] = useState<"cls" | "meanpool">("cls");
  const [selectedClassesFile, setSelectedClassesFile] = useState<{ name: string; text: string } | null>(null);
  const [manualClassNames, setManualClassNames] = useState("apple,banana,orange");
  const [folders, setFolders] = useState<string[]>([]);
  const [folderDraft, setFolderDraft] = useState("");
  const [connected, setConnected] = useState(false);
  const [loadingModels, setLoadingModels] = useState(true);
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [run, setRun] = useState<VisualizationRun | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);
  const [rankingSort, setRankingSort] = useState<RankingCriterion>("outlier");
  const [ranking, setRanking] = useState<ImageRanking | null>(null);
  const [rankingLoading, setRankingLoading] = useState(false);
  const [rankingError, setRankingError] = useState("");
  const [rankingLimit, setRankingLimit] = useState(60);
  const [rankViewerMode, setRankViewerMode] = useState(false);
  const [curationCart, setCurationCart] = useState<CurationCart | null>(null);
  const [cartLoading, setCartLoading] = useState(true);
  const [cartBusy, setCartBusy] = useState(false);
  const [cartError, setCartError] = useState("");
  const [cartNotice, setCartNotice] = useState("");
  const [cartSourceFilter, setCartSourceFilter] = useState("all");
  const [cartReasonDraft, setCartReasonDraft] = useState("");
  const [cartClearConfirm, setCartClearConfirm] = useState(false);
  const [similarOpen, setSimilarOpen] = useState(false);
  const [textSearchMode, setTextSearchMode] = useState(false);
  const [similarChain, setSimilarChain] = useState<number[]>([]);
  const [similarK, setSimilarK] = useState(9);
  const [similarResults, setSimilarResults] = useState<SimilarSearch | null>(null);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [similarError, setSimilarError] = useState("");
  const [textQueryDraft, setTextQueryDraft] = useState("");
  const [submittedTextQuery, setSubmittedTextQuery] = useState("");
  const [textResultK, setTextResultK] = useState(9);
  const [textResults, setTextResults] = useState<TextSearch | null>(null);
  const [textLoading, setTextLoading] = useState(false);
  const [textError, setTextError] = useState("");
  const [duplicateOpen, setDuplicateOpen] = useState(false);
  const [duplicateMethod, setDuplicateMethod] = useState<"phash" | "embedding">("phash");
  const [duplicateCrossSplit, setDuplicateCrossSplit] = useState(false);
  const [phashThreshold, setPhashThreshold] = useState(4);
  const [embeddingThreshold, setEmbeddingThreshold] = useState(0.05);
  const [duplicateResults, setDuplicateResults] = useState<DuplicateScan | null>(null);
  const [duplicateLoading, setDuplicateLoading] = useState(false);
  const [duplicateError, setDuplicateError] = useState("");
  const [showYoloOverlay, setShowYoloOverlay] = useState(true);
  const [viewerImageMode, setViewerImageMode] = useState<"selected" | "source">("selected");
  const [blockCount, setBlockCount] = useState(12);
  const [selectedBlockId, setSelectedBlockId] = useState<number | null>(null);
  const [blocks, setBlocks] = useState<KMeansBlocks | null>(null);
  const [blockProjection, setBlockProjection] = useState<BlockProjection | null>(null);
  const [blocksLoading, setBlocksLoading] = useState(false);
  const [blocksError, setBlocksError] = useState("");
  const [labelFilter, setLabelFilter] = useState("all");
  const [splitFilter, setSplitFilter] = useState("all");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    let active = true;
    Promise.all([api.health(), api.models()])
      .then(([health, response]) => {
        if (!active) return;
        setConnected(health.ok && health.api_version === "1");
        setModels(response.items);
        setSelectedModelIds((current) => {
          if (current.length) return current.filter((id) => response.items.some((item) => item.model_id === id));
          const preferred = response.items.find((item) => item.model_id === "dinov2_vits14") || response.items[0];
          return preferred ? [preferred.model_id] : [];
        });
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "The local worker is unavailable.");
      })
      .finally(() => active && setLoadingModels(false));
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    api.curationCart()
      .then((result) => active && setCurationCart(result))
      .catch((reason: unknown) => active && setCartError(reason instanceof Error ? reason.message : "Unable to load the saved curation cart."))
      .finally(() => active && setCartLoading(false));
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!run) return;
    const preferredModel = run.models.includes("dinov2_vitb14") ? "dinov2_vitb14" : run.models[0];
    const availableForModel = run.views.filter((view) => view.model_id === preferredModel);
    const preferredMethod = availableForModel.find((view) => view.method === "sumap") || availableForModel[0];
    setActiveModelId(preferredModel || "");
    setActiveMethod(preferredMethod?.method || "pca");
  }, [run]);

  const activeViews = useMemo(
    () => run?.views.filter((view) => view.model_id === activeModelId) || [],
    [run, activeModelId],
  );
  const currentView = useMemo(
    () => activeViews.find((view) => view.method === activeMethod) || activeViews[0] || null,
    [activeViews, activeMethod],
  );
  const currentRecords = useMemo(() => {
    if (!currentView) return [];
    return currentView.points.filter((point) =>
      (labelFilter === "all" || point.label === labelFilter) &&
      (splitFilter === "all" || point.split === splitFilter),
    );
  }, [currentView, labelFilter, splitFilter]);
  const labels = useMemo(() => [...new Set(run?.records.map((record) => record.label) || [])].sort(), [run]);
  const splits = useMemo(() => [...new Set(run?.records.map((record) => record.split) || [])].sort(), [run]);
  const selectedRecord = selectedIndex === null ? null : run?.records[selectedIndex] || null;
  const selectedBlock = blocks?.blocks.find((block) => block.block_id === selectedBlockId) || null;
  const selectionGalleryIndices = useMemo(
    () => currentView ? spatialSelectionOrder(selectedIndices, currentView.points) : [],
    [selectedIndices, currentView],
  );
  const rankingRequestIndices = useMemo(() => selectedIndices.length
    ? selectedIndices.filter((index) => splitFilter === "all" || run?.records[index]?.split === splitFilter)
    : null, [selectedIndices, splitFilter, run]);
  const rankingRequestCriterion: RankingCriterion = rankingRequestIndices === null &&
    (rankingSort === "filename" || rankingSort === "spatial") ? "outlier" : rankingSort;
  const rankedIndices = useMemo(() => ranking?.items.map((item) => item.index) || [], [ranking]);
  const rankingShownIndices = useMemo(() => rankingRequestIndices === null
    ? rankedIndices.slice(0, 50)
    : rankedIndices.slice(0, Math.min(rankingLimit, 240)), [rankingRequestIndices, rankedIndices, rankingLimit]);
  const viewerIndices = rankViewerMode ? rankingShownIndices : selectionGalleryIndices;
  const viewerPosition = selectedIndex === null ? -1 : viewerIndices.indexOf(selectedIndex);
  const cartFilteredItems = useMemo(() => curationCart?.items.filter((item) =>
    cartSourceFilter === "all" || item.source === cartSourceFilter,
  ) || [], [curationCart, cartSourceFilter]);
  const similarQueryIndex = similarChain[similarChain.length - 1] ?? null;

  useEffect(() => {
    const runId = job?.run_id;
    if (!runId || !run || !currentView) {
      setRanking(null);
      setRankingLoading(false);
      setRankingError("");
      return;
    }
    let active = true;
    setRanking(null);
    setRankingLoading(true);
    setRankingError("");
    api.rankImages(runId, {
      modelId: activeModelId,
      criterion: rankingRequestCriterion,
      indices: rankingRequestIndices,
      method: currentView.method,
    })
      .then((result) => active && setRanking(result))
      .catch((reason: unknown) => {
        if (active) setRankingError(reason instanceof Error ? reason.message : "Unable to rank review candidates.");
      })
      .finally(() => active && setRankingLoading(false));
    return () => { active = false; };
  }, [job?.run_id, run, currentView, activeModelId, rankingRequestCriterion, rankingRequestIndices, splitFilter]);

  useEffect(() => {
    if (rankViewerMode && selectedIndex !== null && !rankingShownIndices.includes(selectedIndex)) {
      setRankViewerMode(false);
    }
  }, [rankViewerMode, selectedIndex, rankingShownIndices]);

  useEffect(() => {
    const runId = job?.run_id;
    if (dimension !== 3 || run?.source_mode !== "image_classifier" || !runId || !currentView || !currentRecords.length) {
      setBlocks(null);
      setBlockProjection(null);
      setBlocksLoading(false);
      setBlocksError("");
      return;
    }
    let active = true;
    setBlocks(null);
    setBlockProjection(null);
    setBlocksLoading(true);
    setBlocksError("");
    api.runBlocks(runId, activeModelId, currentView.method, currentRecords.map((point) => point.index), blockCount)
      .then((result) => {
        if (!active) return;
        setBlocks(result);
        setSelectedBlockId((current) =>
          result.blocks.some((block) => block.block_id === current)
            ? current
            : result.blocks[0]?.block_id ?? null,
        );
      })
      .catch((reason: unknown) => {
        if (active) setBlocksError(reason instanceof Error ? reason.message : "Unable to partition the visible points.");
      })
      .finally(() => active && setBlocksLoading(false));
    return () => { active = false; };
  }, [dimension, run, job?.run_id, currentView, currentRecords, activeModelId, blockCount]);

  useEffect(() => {
    const runId = job?.run_id;
    if (dimension !== 3 || run?.source_mode !== "image_classifier" || !runId || !selectedBlock || !currentView) {
      setBlockProjection(null);
      return;
    }
    let active = true;
    setBlockProjection(null);
    api.blockProjection(runId, activeModelId, currentView.method, selectedBlock.indices)
      .then((result) => active && setBlockProjection(result))
      .catch((reason: unknown) => {
        if (active) setBlocksError(reason instanceof Error ? reason.message : "Unable to open this KMeans block.");
      });
    return () => { active = false; };
  }, [dimension, run, job?.run_id, selectedBlock, currentView, activeModelId]);

  useEffect(() => {
    const runId = job?.run_id;
    if (!similarOpen || textSearchMode || !runId || !run || run.source_granularity !== "whole_image" || similarQueryIndex === null) {
      setSimilarResults(null);
      setSimilarLoading(false);
      setSimilarError("");
      return;
    }
    let active = true;
    setSimilarResults(null);
    setSimilarLoading(true);
    setSimilarError("");
    api.searchSimilar(runId, activeModelId, similarQueryIndex, similarK)
      .then((result) => active && setSimilarResults(result))
      .catch((reason: unknown) => {
        if (active) {
          setSimilarResults(null);
          setSimilarError(reason instanceof Error ? reason.message : "Unable to search similar images.");
        }
      })
      .finally(() => active && setSimilarLoading(false));
    return () => { active = false; };
  }, [similarOpen, textSearchMode, similarQueryIndex, similarK, activeModelId, run, job?.run_id]);

  useEffect(() => {
    const runId = job?.run_id;
    if (!similarOpen || !textSearchMode || !runId || !run || run.source_granularity !== "whole_image" || !submittedTextQuery) {
      setTextResults(null);
      setTextLoading(false);
      setTextError("");
      return;
    }
    if (!activeModelId.startsWith("chinese-clip")) {
      setTextResults(null);
      setTextLoading(false);
      setTextError("Select a Chinese-CLIP model in this run to search by text.");
      return;
    }
    let active = true;
    setTextResults(null);
    setTextLoading(true);
    setTextError("");
    api.searchByText(runId, activeModelId, submittedTextQuery, textResultK)
      .then((result) => active && setTextResults(result))
      .catch((reason: unknown) => {
        if (active) {
          setTextResults(null);
          setTextError(reason instanceof Error ? reason.message : "Unable to search images by text.");
        }
      })
      .finally(() => active && setTextLoading(false));
    return () => { active = false; };
  }, [similarOpen, textSearchMode, submittedTextQuery, textResultK, activeModelId, run, job?.run_id]);

  useEffect(() => {
    if (duplicateResults?.method === "embedding" && duplicateResults.model_id !== activeModelId) {
      setDuplicateResults(null);
    }
  }, [activeModelId, duplicateResults]);

  function addFolder(value = folderDraft) {
    const path = value.trim();
    if (!path) return;
    setFolders((current) => current.includes(path) ? current : [...current, path]);
    setFolderDraft("");
    setNotice("");
  }

  async function chooseFolder() {
    setError("");
    if (!window.cimHost?.chooseFile) {
      setNotice("Native folder picker is not connected. Enter a local folder path below.");
      return;
    }
    try {
      const result = await window.cimHost.chooseFile();
      if (result.canceled) return;
      const chosen = result.paths?.length ? result.paths : (result.path ? [result.path] : []);
      if (!chosen.length) throw new Error("The host did not return a folder path.");
      chosen.forEach((path) => addFolder(path));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The folder picker failed.");
    }
  }

  function toggleModel(modelId: string) {
    setSelectedModelIds((current) =>
      current.includes(modelId) ? current.filter((id) => id !== modelId) : [...current, modelId],
    );
  }

  function toggleMethod(method: ProjectionMethod) {
    if (method === "umap" && selectedMethods.includes(method)) {
      setUmapReference(false);
      setRebuildUmapReference(false);
    }
    setSelectedMethods((current) =>
      current.includes(method) ? current.filter((item) => item !== method) : [...current, method],
    );
  }

  async function runAnalysis() {
    const selectedFolders = [...folders];
    const draft = folderDraft.trim();
    if (draft && !selectedFolders.includes(draft)) selectedFolders.push(draft);
    if (!selectedFolders.length) {
      setError("Choose at least one image folder.");
      return;
    }
    if (!selectedModelIds.length) {
      setError("Choose at least one installed model.");
      return;
    }
    if (!selectedMethods.length) {
      setError("Choose at least one projection method.");
      return;
    }
    setError("");
    setNotice("");
    setRun(null);
    setSimilarOpen(false);
    setTextSearchMode(false);
    setSimilarChain([]);
    setSimilarResults(null);
    setSubmittedTextQuery("");
    setTextResults(null);
    setDuplicateOpen(false);
    setDuplicateResults(null);
    setSelectedIndex(null);
    setSelectedIndices([]);
    setRankingSort("outlier");
    setRanking(null);
    setRankingError("");
    setRankingLimit(60);
    setRankViewerMode(false);
    setViewerImageMode("selected");
    setShowYoloOverlay(true);
    setSelectedBlockId(null);
    setBlocks(null);
    setBlockProjection(null);
    setLabelFilter("all");
    setSplitFilter("all");
    setDimension(2);
    setBusy(true);
    setJob(null);
    try {
      const created = await api.createJob(selectedFolders, selectedModelIds, selectedMethods, {
        umapReference,
        rebuildUmapReference,
        mode: datasetMode,
        granularity: datasetMode === "object_detector" ? detectorGranularity : "whole_image",
        classesFileText: selectedClassesFile?.text ?? null,
        classNames: manualClassNames.split(",").map((name) => name.trim()).filter(Boolean),
        objectPolicy: { pad: objectPad, target_res: objectTargetRes, head: objectHead },
      });
      setJob({
        job_id: created.job_id,
        state: "queued",
        phase: "queued",
        done: 0,
        total: 0,
        percent: 0,
        message: "Waiting to start",
        run_id: null,
        error: null,
        can_cancel: true,
      });
      let latest: JobStatus | null = null;
      for (let attempt = 0; attempt < 18_000; attempt += 1) {
        latest = await api.job(created.job_id);
        setJob(latest);
        if (["completed", "failed", "cancelled"].includes(latest.state)) break;
        await pause(350);
      }
      if (!latest || !["completed", "failed", "cancelled"].includes(latest.state)) {
        throw new Error("The analysis is taking too long. Cancel it and retry with a smaller dataset.");
      }
      if (latest.state === "failed") throw new Error(latest.error?.message || "The analysis failed.");
      if (latest.state === "cancelled") {
        setNotice("Analysis cancelled. You can update the selection and run it again.");
        return;
      }
      if (!latest.run_id) throw new Error("The worker completed without returning a run result.");
      const result = await api.run(latest.run_id);
      setRun(result);
      setRebuildUmapReference(false);
      setNotice(result.warnings.join(" "));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The analysis failed.");
    } finally {
      setBusy(false);
    }
  }

  async function cancelAnalysis() {
    if (!job?.job_id) return;
    try {
      await api.cancel(job.job_id);
      setJob((current) => current ? { ...current, state: "cancelling", message: "Cancellation requested" } : current);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not cancel the analysis.");
    }
  }

  function selectActiveModel(value: string) {
    setActiveModelId(value);
    const available = run?.views.filter((view) => view.model_id === value) || [];
    if (!available.some((view) => view.method === activeMethod) && available[0]) {
      setActiveMethod(available[0].method);
    }
  }

  function selectPoint(index: number, additive = false) {
    setSelectedIndex(index);
    setSelectedIndices((current) => additive ? [...new Set([...current, index])] : [index]);
    setRankViewerMode(false);
  }

  function startSimilarSearch(index = selectedIndex ?? selectedIndices[0]) {
    if (!run || run.source_granularity !== "whole_image" || index === undefined || index === null) return;
    setSimilarK(Math.min(9, Math.max(1, run.records.length - 1)));
    setSimilarChain([index]);
    setTextSearchMode(false);
    setSubmittedTextQuery("");
    setSimilarOpen(true);
  }

  function openTextSearch() {
    setSimilarChain([]);
    setSimilarResults(null);
    setTextSearchMode(true);
    setTextQueryDraft("");
    setSubmittedTextQuery("");
    setTextResultK(Math.min(9, Math.max(1, run?.records.length || 1)));
    setSimilarOpen(true);
  }

  function pivotTextResult(index: number) {
    setTextSearchMode(false);
    setSubmittedTextQuery("");
    setTextResults(null);
    setSimilarChain([index]);
  }

  function continueSimilarSearch(index: number) {
    setSimilarChain((current) => current[current.length - 1] === index ? current : [...current, index]);
  }

  function truncateSimilarSearch(position: number) {
    setSimilarChain((current) => current.slice(0, position + 1));
  }

  function closeSimilarSearch() {
    setSimilarOpen(false);
    setTextSearchMode(false);
    setSimilarChain([]);
    setSimilarResults(null);
    setSimilarError("");
    setSubmittedTextQuery("");
    setTextResults(null);
    setTextError("");
  }

  async function scanDuplicates() {
    const runId = job?.run_id;
    if (!runId || !run || run.source_granularity !== "whole_image") return;
    setDuplicateLoading(true);
    setDuplicateError("");
    setDuplicateResults(null);
    try {
      const result = await api.scanDuplicates(runId, {
        modelId: activeModelId,
        method: duplicateMethod,
        crossSplitOnly: duplicateCrossSplit,
        phashThreshold,
        embeddingThreshold,
      });
      setDuplicateResults(result);
    } catch (reason) {
      setDuplicateError(reason instanceof Error ? reason.message : "Unable to scan duplicate candidates.");
    } finally {
      setDuplicateLoading(false);
    }
  }

  async function addToCurationCart(candidates: CurationCandidate[]) {
    const runId = job?.run_id;
    if (!runId || !candidates.length) return;
    setCartBusy(true);
    setCartError("");
    setCartNotice("");
    try {
      const result = await api.addCurationItems(runId, candidates);
      setCurationCart(result);
      setCartNotice(`Added ${result.added || 0} · skipped ${result.skipped || 0} already in the cart.`);
    } catch (reason) {
      setCartError(reason instanceof Error ? reason.message : "Unable to add these records to the curation cart.");
    } finally {
      setCartBusy(false);
    }
  }

  async function removeFromCurationCart(itemId: string) {
    setCartBusy(true);
    setCartError("");
    try {
      setCurationCart(await api.removeCurationItem(itemId));
      setCartNotice("Curation item removed.");
    } catch (reason) {
      setCartError(reason instanceof Error ? reason.message : "Unable to remove this curation item.");
    } finally {
      setCartBusy(false);
    }
  }

  async function applyCurationReason() {
    if (!cartFilteredItems.length) return;
    setCartBusy(true);
    setCartError("");
    try {
      const result = await api.setCurationReasons(
        cartFilteredItems.map((item) => item.id), cartReasonDraft.trim(),
      );
      setCurationCart(result);
      setCartNotice(`Reason updated for ${cartFilteredItems.length} items.`);
    } catch (reason) {
      setCartError(reason instanceof Error ? reason.message : "Unable to save the curation reason.");
    } finally {
      setCartBusy(false);
    }
  }

  async function clearCurationCart() {
    if (!cartClearConfirm) return;
    setCartBusy(true);
    setCartError("");
    try {
      setCurationCart(await api.clearCurationCart(true));
      setCartClearConfirm(false);
      setCartNotice("The curation cart was cleared.");
    } catch (reason) {
      setCartError(reason instanceof Error ? reason.message : "Unable to clear the curation cart.");
    } finally {
      setCartBusy(false);
    }
  }

  function navigateViewer(offset: number) {
    const target = viewerIndices[viewerPosition + offset];
    if (target !== undefined) setSelectedIndex(target);
  }

  function addBlockSelection(indices: number[]) {
    if (!indices.length) return;
    setSelectedIndices((current) => [...new Set([...current, ...indices])]);
    setSelectedIndex(indices[indices.length - 1]);
    setRankViewerMode(false);
  }

  function clearSelection() {
    setSelectedIndex(null);
    setSelectedIndices([]);
    setRankViewerMode(false);
  }

  function closeViewer() {
    setSelectedIndex(null);
    setRankViewerMode(false);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
          <div>
            <div className="brand-name">VisualLatent</div>
            <div className="brand-caption">LOCAL DATA WORKSPACE</div>
          </div>
        </div>
        <nav className="tool-nav" aria-label="LV tools">
          <button type="button" aria-pressed={activeTool === "visualize"}
            onClick={() => setActiveTool("visualize")}>Visualize</button>
          <button type="button" aria-pressed={activeTool === "compare"}
            onClick={() => setActiveTool("compare")}>Compare</button>
        </nav>
        <div className={"connection " + (connected ? "connection-ready" : "connection-waiting")}>
          <span className="connection-dot" />
          {connected ? "Local worker connected" : "Connecting to local worker"}
        </div>
      </header>

      <div className="workspace" hidden={activeTool === "compare"}>
        <aside className="control-panel">
          <div className="eyebrow">VISUALIZE EMBEDDINGS</div>
          <h1>Explore your image space</h1>
          <p className="intro">Compare how your local image collections group together using the models already installed on this device.</p>

          <section className="control-section">
            <div className="section-heading"><span className="step-number">01</span><h2>Image folders</h2></div>
            <button className="host-picker" type="button" onClick={chooseFolder}>
              <span className="folder-icon" aria-hidden="true">▱</span>
              Choose folder from device
              <span className="picker-hint">Native</span>
            </button>
            <label className="field-label" htmlFor="folder-path">Image folder</label>
            <div className="folder-input-row">
              <input
                id="folder-path"
                value={folderDraft}
                onChange={(event) => setFolderDraft(event.target.value)}
                onKeyDown={(event) => event.key === "Enter" && addFolder()}
                placeholder="Paste a local folder path"
                autoComplete="off"
              />
              <button type="button" className="add-folder" onClick={() => addFolder()} aria-label="Add folder">+</button>
            </div>
            {folders.length > 0 && (
              <ul className="folder-list" aria-label="Selected folders">
                {folders.map((folder) => (
                  <li key={folder} title={folder}>
                    <span className="folder-chip-dot" />
                    <span className="folder-chip-name">{folder.split(/[\\/]/).filter(Boolean).at(-1) || folder}</span>
                    <button type="button" aria-label={"Remove " + folder} onClick={() => setFolders((current) => current.filter((item) => item !== folder))}>×</button>
                  </li>
                ))}
              </ul>
            )}
            <div className="field-note">Folders stay on this device. LV writes analysis cache to its app cache, not into your dataset.</div>
          </section>

          <section className="control-section model-section">
            <div className="section-heading"><span className="step-number">02</span><h2>Analysis setup</h2></div>
            <label className="field-label" htmlFor="dataset-mode">Dataset mode</label>
            <select
              id="dataset-mode"
              aria-label="Dataset mode"
              data-testid="dataset-mode"
              value={datasetMode}
              disabled={busy}
              onChange={(event) => {
                setDatasetMode(event.target.value as "image_classifier" | "object_detector");
                setRun(null);
                setJob(null);
                clearSelection();
                setViewerImageMode("selected");
                setShowYoloOverlay(true);
                setLabelFilter("all");
                setSplitFilter("all");
              }}
            >
              <option value="image_classifier">Image Classifier</option>
              <option value="object_detector">Object Detector</option>
            </select>
            {datasetMode === "object_detector" && (
              <div className="detector-class-controls" data-testid="detector-class-controls">
                <label className="field-label" htmlFor="detector-granularity">Detection granularity</label>
                <select
                  id="detector-granularity"
                  aria-label="Detection granularity"
                  data-testid="detector-granularity"
                  value={detectorGranularity}
                  disabled={busy}
                  onChange={(event) => {
                    setDetectorGranularity(event.target.value as "whole_image" | "object");
                    setRun(null);
                    setJob(null);
                    clearSelection();
                    setViewerImageMode("selected");
                  }}
                >
                  <option value="whole_image">Whole image</option>
                  <option value="object">Per object (YOLO boxes)</option>
                </select>
                {detectorGranularity === "object" && (
                  <div className="object-policy-controls" data-testid="object-policy-controls">
                    <label className="field-label field-label-spaced" htmlFor="object-pad">BBox padding (0–0.50)</label>
                    <input
                      id="object-pad"
                      aria-label="BBox padding"
                      type="number"
                      min="0"
                      max="0.5"
                      step="0.01"
                      value={objectPad}
                      disabled={busy}
                      onChange={(event) => setObjectPad(Number(event.target.value))}
                    />
                    <label className="field-label field-label-spaced" htmlFor="object-target-res">Object input size</label>
                    <select
                      id="object-target-res"
                      aria-label="Object input size"
                      value={objectTargetRes}
                      disabled={busy}
                      onChange={(event) => setObjectTargetRes(Number(event.target.value) as typeof objectTargetRes)}
                    >
                      {[168, 196, 224, 280, 336, 392, 448].map((size) => <option key={size} value={size}>{size}px</option>)}
                    </select>
                    <label className="field-label field-label-spaced" htmlFor="object-head">Object feature head</label>
                    <select
                      id="object-head"
                      aria-label="Object feature head"
                      value={objectHead}
                      disabled={busy}
                      onChange={(event) => setObjectHead(event.target.value as "cls" | "meanpool")}
                    >
                      <option value="cls">CLS token</option>
                      <option value="meanpool">Mean pool</option>
                    </select>
                    <div className="field-note">Each labeled box becomes one point. Padding is clamped to 8–64 source pixels per side.</div>
                  </div>
                )}
                <label className="field-label field-label-spaced" htmlFor="classes-file">Classes file</label>
                <input
                  id="classes-file"
                  aria-label="Classes file"
                  data-testid="classes-file"
                  type="file"
                  accept=".txt,text/plain"
                  disabled={busy}
                  onChange={(event) => {
                    const file = event.currentTarget.files?.[0];
                    if (!file) {
                      setSelectedClassesFile(null);
                      return;
                    }
                    void file.text().then((text) => setSelectedClassesFile({ name: file.name, text })).catch(() => {
                      setError("Could not read the selected classes file.");
                    });
                  }}
                />
                {selectedClassesFile && <small data-testid="selected-classes-file">Selected: {selectedClassesFile.name}</small>}
                <label className="field-label field-label-spaced" htmlFor="manual-class-names">Manual class names</label>
                <input
                  id="manual-class-names"
                  aria-label="Manual class names"
                  value={manualClassNames}
                  disabled={busy}
                  onChange={(event) => setManualClassNames(event.target.value)}
                  placeholder="name one, name two"
                />
                <div className="field-note">Class names use this order: selected file, dataset classes.txt/data.yaml, then these names.</div>
              </div>
            )}
            <div className="field-label">Feature models</div>
            <div className="choice-list" aria-label="Feature models">
              {models.map((model) => (
                <label className="choice-option" key={model.model_id}>
                  <input
                    type="checkbox"
                    data-testid={"model-choice-" + model.model_id}
                    aria-label={"Model " + model.name}
                    checked={selectedModelIds.includes(model.model_id)}
                    disabled={loadingModels || busy}
                    onChange={() => toggleModel(model.model_id)}
                  />
                  <span>{model.name}</span>
                </label>
              ))}
              {!loadingModels && models.length === 0 && <div className="model-empty">Install a supported LV model to enable analysis.</div>}
            </div>
            <div className="field-note">Each selected model gets its own embedding cache and result view.</div>

            <div className="field-label field-label-spaced">Projection methods</div>
            <div className="choice-list projection-list" aria-label="Projection methods">
              {projectionOptions.map((option) => (
                <label className="choice-option" key={option.value}>
                  <input
                    type="checkbox"
                    data-testid={"projection-" + option.value}
                    aria-label={option.label}
                    checked={selectedMethods.includes(option.value)}
                    disabled={busy}
                    onChange={() => toggleMethod(option.value)}
                  />
                  <span>{option.label}</span>
                </label>
              ))}
            </div>
            <div className="field-note">LDA and Supervised UMAP use class labels to arrange points. Their layout is a visual aid, not proof of separability.</div>
            {selectedMethods.includes("umap") && (
              <div className="umap-reference-controls">
                <label className="choice-option">
                  <input
                    type="checkbox"
                    aria-label="Keep UMAP reference frame"
                    checked={umapReference}
                    disabled={busy}
                    onChange={(event) => {
                      setUmapReference(event.target.checked);
                      if (!event.target.checked) setRebuildUmapReference(false);
                    }}
                  />
                  <span>Keep UMAP reference frame</span>
                </label>
                {umapReference && (
                  <>
                    <p>Known images keep their coordinates between runs. New images are transformed into the saved frame; rebuild after the dataset changes substantially.</p>
                    <button
                      type="button"
                      className="rebuild-reference"
                      aria-pressed={rebuildUmapReference}
                      disabled={busy}
                      onClick={() => setRebuildUmapReference((current) => !current)}
                    >
                      {rebuildUmapReference ? "Rebuild queued for next run" : "Rebuild UMAP reference"}
                    </button>
                  </>
                )}
              </div>
            )}
          </section>

          {error && <div className="alert alert-error" role="alert">{error}</div>}
          {notice && <div className="alert alert-info" role="status">{notice}</div>}

          <button className="run-button" type="button" onClick={runAnalysis} disabled={busy || !connected || loadingModels || !selectedModelIds.length || !selectedMethods.length}>
            <span aria-hidden="true">{busy ? "◌" : "▶"}</span>
            {busy ? "Analysis running" : "Run analysis"}
          </button>
          {busy && <button className="cancel-button" type="button" onClick={cancelAnalysis}>Cancel analysis</button>}
          <div className="privacy-note"><span aria-hidden="true">⌁</span> Local processing · no dataset upload</div>
        </aside>

        <main className="results-area">
          <div className="results-header">
            <div>
              <div className="eyebrow">ANALYSIS CANVAS</div>
              <h2>{run ? "Embedding distribution" : "Your workspace is ready"}</h2>
              <p>{run && currentView ? run.records.length.toLocaleString() + (run.source_granularity === "object" ? " objects" : " images") + " · " + activeModelId + " · " + currentView.method.toUpperCase() : "Choose a folder and run an analysis to see your collection."}</p>
              {run?.source_mode === "object_detector" && (
                <small data-testid="dataset-mode-indicator">
                  Object Detector · {run.source_granularity === "object" ? "object-level" : "whole-image"} · {run.class_names.length} classes · {run.class_source.replaceAll("_", " ")}
                </small>
              )}
            </div>
            {run && <div className="run-badge"><span className="badge-dot" /> Result ready</div>}
          </div>

          {busy && (
            <div className="progress-card" data-testid="job-status" role="status">
              <div className="progress-row"><span>{job?.message || "Preparing analysis"}</span><strong>{Math.round(job?.percent || 0)}%</strong></div>
              <div className="progress-track"><div className="progress-fill" style={{ width: Math.max(3, job?.percent || 0) + "%" }} /></div>
              <div className="progress-foot">{job?.done || 0} / {job?.total || "—"} model-image embeddings</div>
            </div>
          )}
          {!busy && job?.state === "completed" && <div className="completion-label" data-testid="job-status"><span className="completion-check">✓</span> Completed</div>}
          {!busy && job?.state === "cancelled" && <div className="completion-label" data-testid="job-status">Cancelled</div>}

          {run && currentView ? (
            <div className="result-grid">
              <section className="plot-card">
                <div className="plot-toolbar">
                  <div className="plot-title"><span className="chart-glyph" aria-hidden="true">⊙</span><div><strong>{run.source_granularity === "object" ? "Object embeddings" : "Image embeddings"}</strong><small>Click a point to inspect its source</small></div></div>
                  <div className="filters">
                    <label><span>Model</span><select aria-label="Display model" value={activeModelId} onChange={(event) => selectActiveModel(event.target.value)}>{run.models.map((model) => <option key={model}>{model}</option>)}</select></label>
                    <label><span>Method</span><select aria-label="Display method" value={currentView.method} onChange={(event) => setActiveMethod(event.target.value as ProjectionMethod)}>{activeViews.map((view) => <option key={view.method} value={view.method}>{projectionOptions.find((option) => option.value === view.method)?.label || view.method}</option>)}</select></label>
                    <label><span>Split</span><select aria-label="Filter by split" value={splitFilter} onChange={(event) => setSplitFilter(event.target.value)}><option value="all">All splits</option>{splits.map((split) => <option key={split}>{split}</option>)}</select></label>
                  </div>
                </div>
                <div className="plot-subtoolbar">
                  <label><span>Class</span><select aria-label="Filter by class" value={labelFilter} onChange={(event) => setLabelFilter(event.target.value)}><option value="all">All classes</option>{labels.map((label) => <option key={label}>{label}</option>)}</select></label>
                  <div className="dimension-switch" role="group" aria-label="Projection dimension">
                    <button type="button" data-testid="dimension-2d" aria-pressed={dimension === 2} onClick={() => setDimension(2)}>2D</button>
                    <button type="button" data-testid="dimension-3d" aria-pressed={dimension === 3} onClick={() => setDimension(3)}>3D</button>
                  </div>
                  {run.source_granularity === "whole_image" && (
                    <button type="button" className="find-similar-button" data-testid="find-similar"
                      disabled={selectedIndex === null && selectedIndices.length === 0}
                      onClick={() => startSimilarSearch()}>
                      Find similar
                    </button>
                  )}
                  {run.source_granularity === "whole_image" && activeModelId.startsWith("chinese-clip") && (
                    <button type="button" className="find-similar-button" data-testid="search-by-text" onClick={openTextSearch}>
                      Search by text
                    </button>
                  )}
                  {run.source_granularity === "whole_image" && (
                    <button type="button" className="find-similar-button" data-testid="open-duplicate-review"
                      onClick={() => { setDuplicateOpen(true); setDuplicateResults(null); setDuplicateError(""); }}>
                      Find duplicates
                    </button>
                  )}
                </div>
                {currentRecords.length ? (
                  <ScatterPlot
                    points={currentRecords}
                    selectedIndices={selectedIndices}
                    clusters={blocks}
                    onSelect={selectPoint}
                    dimension={dimension}
                    viewKey={activeModelId + "-" + currentView.method + "-" + splitFilter + "-" + labelFilter}
                  />
                ) : <div className="empty-filter">No images match these filters.</div>}
                {dimension === 3 && run.source_mode === "image_classifier" && (
                  <section className="block-explorer" data-testid="block-explorer" aria-label="KMeans block explorer">
                    <div className="block-explorer-heading">
                      <div><strong>Explore KMeans blocks</strong><small>Choose a 3D block, then add points with a 2D box or lasso.</small></div>
                      {blocks && <span data-testid="block-effective-count">{blocks.effective_k} blocks</span>}
                    </div>
                    <label className="block-count-control">
                      <span>Block count K <strong data-testid="block-count-value">{blockCount}</strong></span>
                      <input aria-label="Block count K" type="range" min={4} max={30} step={1} value={blockCount}
                        onChange={(event) => setBlockCount(Number(event.target.value))} />
                    </label>
                    {blocksLoading && <div className="block-loading" role="status">Partitioning visible points…</div>}
                    {blocksError && <div className="alert alert-error" role="alert">{blocksError}</div>}
                    {blocks?.blocks.length ? (
                      <>
                        <label className="block-picker-label"><span>Enter block → 2D box/lasso</span>
                          <select aria-label="Enter block for 2D selection" data-testid="block-choice"
                            value={selectedBlockId ?? ""}
                            onChange={(event) => setSelectedBlockId(Number(event.target.value))}>
                            {blocks.blocks.map((block) => (
                              <option key={block.block_id} value={block.block_id}>
                                #{block.block_id} ({block.count})
                              </option>
                            ))}
                          </select>
                        </label>
                        {selectedBlock && (
                          <>
                            <div className="block-selection-summary" data-testid="block-selection-summary">
                              Selected {selectedIndices.length} images · block #{selectedBlock.block_id} has {selectedBlock.count}
                            </div>
                            {blockProjection ? (
                              <Suspense fallback={<div className="plot-loading" role="status">Opening block…</div>}>
                                <LazyBlockSelectionPlot points={blockProjection.points}
                                  selectedIndices={selectedIndices} onAddSelection={addBlockSelection} />
                              </Suspense>
                            ) : <div className="block-loading" role="status">Preparing block projection…</div>}
                          </>
                        )}
                      </>
                    ) : !blocksLoading && !blocksError && <div className="block-loading">No visible points are available for clustering.</div>}
                  </section>
                )}
                <div className="plot-legend">
                  {labels.map((label) => <span key={label}><i style={{ background: colorMap(currentView.points).get(label) || labelColors[0] }} />{label}</span>)}
                  <span className="legend-separator" />
                  <small>● train　■ test　◆ valid</small>
                  <span data-testid="selection-summary" className="selection-summary">Selected {selectedIndices.length} {run.source_granularity === "object" ? "objects" : "images"}</span>
                  {selectedIndices.length > 0 && <button type="button" className="clear-selection" onClick={clearSelection}>Clear selection</button>}
                </div>
                <section className="selection-gallery" data-testid="selection-gallery" aria-label="Selected image thumbnails">
                  <div className="selection-gallery-heading">
                    <div><strong>Selected {run.source_granularity === "object" ? "objects" : "images"}</strong><span>{selectionGalleryIndices.length}</span></div>
                    <button type="button" data-testid="cart-add-selection" disabled={!selectedIndices.length || cartBusy}
                      onClick={() => void addToCurationCart(selectedIndices.map((index) => ({ index, source: "manual" })))}>
                      Add selection to cart
                    </button>
                  </div>
                  {selectionGalleryIndices.length ? (
                    <div className="selection-gallery-grid">
                      {selectionGalleryIndices.map((index) => {
                        const record = run.records[index];
                        return (
                          <button
                            key={index}
                            type="button"
                            className={"selection-gallery-item" + (selectedIndex === index ? " selection-gallery-item-active" : "")}
                            data-testid="selection-gallery-item"
                            data-index={index}
                            aria-label={"View " + record.filename + (record.object_index === undefined ? "" : ", object " + (record.object_index + 1))}
                            aria-pressed={selectedIndex === index}
                            onClick={() => { setSelectedIndex(index); setRankViewerMode(false); }}
                          >
                            <img src={api.runImageUrl(job?.run_id || "", index)} alt="" loading="lazy" />
                            <span>{record.label}｜{record.filename}{record.object_index === undefined ? "" : ` · object ${record.object_index + 1}`}</span>
                          </button>
                        );
                      })}
                    </div>
                  ) : (
                    <p>Select one or more points to see matching thumbnails here.</p>
                  )}
                </section>
                <section className="ranking-gallery" data-testid="ranking-gallery" aria-label="Ranked review candidates">
                  <div className="ranking-gallery-heading">
                    <div><strong>Ranked review</strong><small>Review cues only; scores are not image-quality judgments.</small></div>
                    <label><span>Sort</span><select aria-label="Rank review images" value={rankingSort}
                      onChange={(event) => { setRankingSort(event.target.value as RankingCriterion); setRankingLimit(60); }}>
                      <option value="spatial">Spatial order</option>
                      <option value="outlier">Outlier score</option>
                      <option value="disagreement">Label disagreement</option>
                      <option value="filename">Filename</option>
                    </select></label>
                  </div>
                  {rankingLoading && <p className="ranking-status" role="status">Calculating review order for {activeModelId}…</p>}
                  {rankingError && <p className="ranking-status ranking-error" role="alert">{rankingError}</p>}
                  {!rankingLoading && !rankingError && ranking && !ranking.available && (
                    <p className="ranking-status" data-testid="ranking-status">At least 3 records are required to rank a run.</p>
                  )}
                  {ranking?.available && (
                    <>
                      <p className="ranking-status" data-testid="ranking-status">
                        {rankingRequestIndices === null
                          ? `No selection · top ${rankingShownIndices.length} by ${rankingSortNames[ranking.effective_criterion]}`
                          : `Selected ${selectedIndices.length} · showing ${rankingShownIndices.length}/${ranking.total} · ${ranking.effective_criterion === "spatial" && rankingSort !== "spatial" ? "scores unavailable; spatial order used" : "sorted by " + rankingSortNames[ranking.effective_criterion]}`}
                        {rankingRequestIndices !== null && ranking.total > 240 && rankingShownIndices.length >= 240 ? " · showing first 240" : ""}
                      </p>
                      {ranking.effective_criterion === "outlier" && (
                        <p className="ranking-note">Higher mean cosine distance to nearest neighbors means more isolated; use it as a review cue, not a quality verdict.</p>
                      )}
                      {ranking.effective_criterion === "disagreement" && (
                        <p className="ranking-note">Higher share of differently labeled nearest neighbors is a review cue, not a mislabel verdict.</p>
                      )}
                      {rankingShownIndices.length ? (
                        <div className="ranking-gallery-grid">
                          {rankingShownIndices.map((index, position) => {
                            const record = run.records[index];
                            const rankedItem = ranking?.items.find((item) => item.index === index);
                            const source: CurationCandidate["source"] = rankingSort === "outlier" || rankingSort === "disagreement"
                              ? rankingSort : "manual";
                            return (
                              <div className="ranking-gallery-entry" key={index}>
                                <button
                                  type="button"
                                  className={"selection-gallery-item" + (rankViewerMode && selectedIndex === index ? " selection-gallery-item-active" : "")}
                                  data-testid="ranked-review-item"
                                  data-index={index}
                                  data-rank={position + 1}
                                  aria-label={`Review ranked candidate ${position + 1}: ${record.filename}`}
                                  aria-pressed={rankViewerMode && selectedIndex === index}
                                  onClick={() => { setSelectedIndex(index); setRankViewerMode(true); }}
                                >
                                  <img src={api.runImageUrl(job?.run_id || "", index, {
                                    overlay: run.source_mode === "object_detector" && showYoloOverlay,
                                    view: run.source_granularity === "object" ? viewerImageMode : "selected",
                                  })} alt="" loading="lazy" />
                                  <span>{record.label}｜{record.filename}{record.object_index === undefined ? "" : ` · object ${record.object_index + 1}`}</span>
                                </button>
                                <button type="button" className="cart-candidate-button" disabled={cartBusy}
                                  aria-label={`Add ${record.filename} to cart`}
                                  onClick={() => void addToCurationCart([{ index, source,
                                    score: source === "manual" ? null : rankedItem?.score }])}>
                                  Add to cart
                                </button>
                              </div>
                            );
                          })}
                        </div>
                      ) : <p className="ranking-status">No records match this split.</p>}
                      {rankingRequestIndices !== null && ranking.total > rankingShownIndices.length && rankingShownIndices.length < 240 && (
                        <button type="button" className="ranking-more" data-testid="ranking-load-more"
                          onClick={() => setRankingLimit((current) => Math.min(current + 60, 240))}>
                          Load more (+60)
                        </button>
                      )}
                    </>
                  )}
                </section>
                {similarOpen && run.source_granularity === "whole_image" && (
                  <section className="similar-panel" data-testid="similar-panel" aria-label="Similar images">
                    <div className="similar-panel-heading">
                      <div><strong>{textSearchMode ? "Search images by text" : "Find similar images"}</strong><small>{textSearchMode ? "Chinese-CLIP maps text into the selected image feature space" : "Recomputed with the active feature model"}</small></div>
                      <span data-testid="similar-model">{activeModelId}</span>
                      <button type="button" aria-label="Close similar search" onClick={closeSimilarSearch}>×</button>
                    </div>
                    {textSearchMode && (
                      <>
                        <form className="text-search-form" onSubmit={(event) => {
                          event.preventDefault();
                          const query = textQueryDraft.trim();
                          if (query) setSubmittedTextQuery(query);
                        }}>
                          <label><span>Text query</span><input aria-label="Text search query" value={textQueryDraft}
                            onChange={(event) => setTextQueryDraft(event.target.value)}
                            placeholder="斑馬、夜間反光、zebra" maxLength={512} /></label>
                          <label><span>Results</span><input aria-label="Text result count" type="number" min={1}
                            max={Math.max(1, run.records.length)} value={textResultK}
                            onChange={(event) => setTextResultK(Math.max(1, Math.min(Number(event.target.value) || 1, run.records.length)))} /></label>
                          <button type="submit" disabled={!textQueryDraft.trim() || !activeModelId.startsWith("chinese-clip")}>Search</button>
                        </form>
                        <p className="text-search-note">Traditional Chinese is normalized by the local text model; English is also supported.</p>
                        {textLoading && <div className="similar-loading" role="status">Searching with {activeModelId}…</div>}
                        {textError && <div className="alert alert-error" role="alert">{textError}</div>}
                        {textResults && (
                          <div className="similar-panel-toolbar">
                            <span data-testid="text-search-query">“{textResults.query}” · {textResults.results.length} results</span>
                            <a data-testid="text-search-export" download="text_search.csv"
                              href={api.textSearchCsvUrl(job?.run_id || "", activeModelId, submittedTextQuery, textResultK)}>
                              Export CSV
                            </a>
                          </div>
                        )}
                        {textResults?.results.length ? (
                          <div className="similar-results" data-testid="text-search-results">
                            {textResults.results.map((record) => (
                              <article className="similar-result" key={record.index} data-testid="text-search-result" data-index={record.index}>
                                <img src={api.runImageUrl(job?.run_id || "", record.index)} alt="" loading="lazy" />
                                <div><strong>{record.label}｜{record.filename}</strong><small>Cosine distance · d={record.distance.toFixed(4)}</small></div>
                                <button type="button" aria-label={`Use ${record.filename} as image query`}
                                  onClick={() => pivotTextResult(record.index)}>Continue by image</button>
                                <button type="button" className="cart-candidate-button" disabled={cartBusy}
                                  onClick={() => void addToCurationCart([{ index: record.index, source: "search", score: record.distance }])}>
                                  Add to cart
                                </button>
                              </article>
                            ))}
                          </div>
                        ) : textResults && !textLoading ? (
                          <p className="similar-empty">No images match this text query.</p>
                        ) : null}
                      </>
                    )}
                    {!textSearchMode && similarQueryIndex !== null && (
                      <>
                    <div className="similar-query-chain" aria-label="Query history">
                      {similarChain.map((index, position) => (
                        <button type="button" key={`${position}-${index}`} data-testid="similar-query-step"
                          aria-label={`Query step ${position + 1}: ${run.records[index]?.filename || index}`}
                          onClick={() => truncateSimilarSearch(position)}>
                          #{index + 1} {run.records[index]?.filename || `Image ${index + 1}`}
                        </button>
                      ))}
                    </div>
                    <div className="similar-panel-toolbar">
                      <label><span>Results</span><input aria-label="Similar result count" type="number" min={1}
                        max={Math.max(1, run.records.length - 1)} value={similarK}
                        onChange={(event) => setSimilarK(Math.max(1, Math.min(Number(event.target.value) || 1, run.records.length - 1)))} /></label>
                      {similarResults && <a data-testid="similar-export" download="similar_group.csv"
                        href={api.similarCsvUrl(job?.run_id || "", activeModelId, similarQueryIndex, similarK)}>
                        Export query + results CSV
                      </a>}
                    </div>
                    {similarLoading && <div className="similar-loading" role="status">Searching this run with {activeModelId}…</div>}
                    {similarError && <div className="alert alert-error" role="alert">{similarError}</div>}
                    {similarResults?.results.length ? (
                      <div className="similar-results" data-testid="similar-results">
                        {similarResults.results.map((record) => (
                          <article className="similar-result" key={record.index} data-testid="similar-result" data-index={record.index}>
                            <img src={api.runImageUrl(job?.run_id || "", record.index)} alt="" loading="lazy" />
                            <div><strong>{record.label}｜{record.filename}</strong><small>Cosine distance · d={record.distance.toFixed(4)}</small></div>
                            <button type="button" aria-label={`Search from ${record.filename}`}
                              onClick={() => continueSimilarSearch(record.index)}>Use as next query</button>
                            <button type="button" className="cart-candidate-button" disabled={cartBusy}
                              onClick={() => void addToCurationCart([{ index: record.index, source: "similar", score: record.distance }])}>
                              Add to cart
                            </button>
                          </article>
                        ))}
                      </div>
                    ) : similarResults && !similarLoading ? (
                      <p className="similar-empty">No other images are available for this query.</p>
                    ) : null}
                      </>
                    )}
                  </section>
                )}
                {duplicateOpen && run.source_granularity === "whole_image" && (
                  <section className="duplicate-panel" data-testid="duplicate-panel" aria-label="Duplicate image review">
                    <div className="similar-panel-heading">
                      <div><strong>Duplicate and split leakage review</strong><small>Candidate pairs only; no automatic deletion or quality decision</small></div>
                      <span>{activeModelId}</span>
                      <button type="button" aria-label="Close duplicate review" onClick={() => setDuplicateOpen(false)}>×</button>
                    </div>
                    <div className="duplicate-controls">
                      <label><span>Method</span><select aria-label="Duplicate method" value={duplicateMethod}
                        onChange={(event) => { setDuplicateMethod(event.target.value as "phash" | "embedding"); setDuplicateResults(null); }}>
                        <option value="phash">Perceptual hash (strict)</option>
                        <option value="embedding">Embedding (semantic)</option>
                      </select></label>
                      {duplicateMethod === "phash" ? (
                        <label><span>Hamming ≤</span><input aria-label="Perceptual hash threshold" type="number" min={0} max={16}
                          value={phashThreshold} onChange={(event) => { setPhashThreshold(Number(event.target.value)); setDuplicateResults(null); }} /></label>
                      ) : (
                        <label><span>Cosine distance ≤</span><input aria-label="Embedding distance threshold" type="number" min={0} max={0.5} step={0.01}
                          value={embeddingThreshold} onChange={(event) => { setEmbeddingThreshold(Number(event.target.value)); setDuplicateResults(null); }} /></label>
                      )}
                      <label className="duplicate-cross-split"><input aria-label="Only cross split" type="checkbox" checked={duplicateCrossSplit}
                        onChange={(event) => { setDuplicateCrossSplit(event.target.checked); setDuplicateResults(null); }} />Only cross split · leakage candidates</label>
                      <button type="button" data-testid="scan-duplicates" onClick={() => void scanDuplicates()}
                        disabled={duplicateLoading}>{duplicateLoading ? "Scanning…" : "Scan duplicates"}</button>
                    </div>
                    {duplicateLoading && <div className="similar-loading" role="status">Scanning this run…</div>}
                    {duplicateError && <div className="alert alert-error" role="alert">{duplicateError}</div>}
                    {duplicateResults && (
                      <div className="duplicate-results-summary" data-testid="duplicate-results-summary">
                        Found {duplicateResults.total_pairs} pair candidates · showing {Math.min(50, duplicateResults.pairs.length)} · manual review only
                      </div>
                    )}
                    {duplicateResults?.pairs.length ? (
                      <div className="duplicate-pairs" data-testid="duplicate-pairs">
                        {duplicateResults.pairs.slice(0, 50).map((pair, index) => (
                          <article className="duplicate-pair" key={`${pair.left.index}-${pair.right.index}`} data-testid="duplicate-pair">
                            <button type="button" aria-label={`Review left ${pair.left.filename}`} onClick={() => { setSelectedIndex(pair.left.index); setRankViewerMode(false); }}>
                              <img src={api.runImageUrl(job?.run_id || "", pair.left.index)} alt="" loading="lazy" />
                              <span>{pair.left.label}｜{pair.left.filename} · {pair.left.split}</span>
                            </button>
                            <div className="duplicate-pair-distance"><small>Pair {index + 1}</small><strong>d={duplicateResults.method === "phash" ? Math.round(pair.distance) : pair.distance.toFixed(4)}</strong>
                              <button type="button" className="cart-candidate-button" disabled={cartBusy}
                                onClick={() => void addToCurationCart([
                                  { index: pair.left.index, source: "duplicate", score: pair.distance },
                                  { index: pair.right.index, source: "duplicate", score: pair.distance },
                                ])}>
                                Add pair
                              </button>
                            </div>
                            <button type="button" aria-label={`Review right ${pair.right.filename}`} onClick={() => { setSelectedIndex(pair.right.index); setRankViewerMode(false); }}>
                              <img src={api.runImageUrl(job?.run_id || "", pair.right.index)} alt="" loading="lazy" />
                              <span>{pair.right.label}｜{pair.right.filename} · {pair.right.split}</span>
                            </button>
                          </article>
                        ))}
                      </div>
                    ) : duplicateResults && !duplicateLoading ? (
                      <p className="similar-empty">No duplicate candidates match these settings.</p>
                    ) : null}
                  </section>
                )}
              </section>

              <aside className="detail-card">
                <div className="detail-card-heading"><div><div className="eyebrow">{run.source_granularity === "object" ? "OBJECT DETAIL" : "IMAGE DETAIL"}</div><h3>{selectedRecord ? (run.source_granularity === "object" ? "Selected object" : "Selected image") : "Select a point"}</h3></div><span className="detail-count">{selectedRecord ? "#" + (selectedRecord.index + 1) : "—"}</span></div>
                {selectedRecord && run ? (
                  <>
                    <div className="viewer-navigation">
                      <button type="button" aria-label={rankViewerMode ? "Previous ranked image" : "Previous selected image"} onClick={() => navigateViewer(-1)} disabled={viewerPosition <= 0}>‹</button>
                      <span data-testid="viewer-position">{viewerPosition >= 0 ? `${viewerPosition + 1} / ${viewerIndices.length}` : "—"}</span>
                      <button type="button" aria-label={rankViewerMode ? "Next ranked image" : "Next selected image"} onClick={() => navigateViewer(1)} disabled={viewerPosition < 0 || viewerPosition >= viewerIndices.length - 1}>›</button>
                      <button type="button" className="viewer-close" aria-label="Close image viewer" onClick={closeViewer}>×</button>
                    </div>
                    {run.source_granularity === "object" && (
                      <fieldset className="object-view-mode" aria-label="Object preview mode">
                        <label><input type="radio" name="object-view-mode" value="selected" checked={viewerImageMode === "selected"} onChange={() => setViewerImageMode("selected")} /> Object crop</label>
                        <label><input type="radio" name="object-view-mode" value="source" checked={viewerImageMode === "source"} onChange={() => setViewerImageMode("source")} /> Source image</label>
                      </fieldset>
                    )}
                    <div className="image-preview"><img data-testid="selected-image" src={api.runImageUrl(job?.run_id || "", selectedRecord.index, {
                      overlay: run.source_mode === "object_detector" && showYoloOverlay,
                      view: run.source_granularity === "object" ? viewerImageMode : "selected",
                    })} alt={selectedRecord.filename} /></div>
                    <div className="image-name" data-testid="selected-image-name" title={selectedRecord.filename}>{selectedRecord.filename}{selectedRecord.object_index !== undefined ? ` · object ${selectedRecord.object_index + 1}` : ""}</div>
                    {run.source_mode === "object_detector" && run.class_names.length > 0 &&
                      (run.source_granularity !== "object" || viewerImageMode === "source") && (
                      <label className="viewer-overlay-toggle">
                        <input type="checkbox" aria-label="Show YOLO boxes" checked={showYoloOverlay} onChange={(event) => setShowYoloOverlay(event.target.checked)} />
                        <span>Show YOLO boxes</span>
                      </label>
                    )}
                    <div className="metadata-list">
                      <div><span>Class</span><strong>{selectedRecord.label}</strong></div>
                      <div><span>Split</span><strong>{selectedRecord.split}</strong></div>
                      {selectedRecord.object_index !== undefined && <div><span>Object</span><strong data-testid="selected-object-number">{selectedRecord.object_index + 1}</strong></div>}
                      {selectedRecord.bbox && <div><span>BBox (cx, cy, w, h)</span><strong data-testid="selected-object-bbox">{selectedRecord.bbox.map((value) => value.toFixed(3)).join(", ")}</strong></div>}
                      {selectedRecord.score !== undefined && selectedRecord.score !== null && <div><span>Score</span><strong>{selectedRecord.score.toFixed(3)}</strong></div>}
                      <div><span>Model</span><strong>{activeModelId}</strong></div>
                      <div><span>Method</span><strong>{currentView.method.toUpperCase()}</strong></div>
                    </div>
                  </>
                ) : (
                  <div className="detail-empty"><div className="image-placeholder" aria-hidden="true">▧</div><p>Select a point or thumbnail to preview the matching image and its metadata.</p></div>
                )}
              </aside>
            </div>
          ) : (
            <div className="empty-canvas" data-testid="empty-canvas">
              <div className="empty-illustration" aria-hidden="true"><span className="orbit orbit-one" /><span className="orbit orbit-two" /><i className="dot dot-one" /><i className="dot dot-two" /><i className="dot dot-three" /><i className="dot dot-four" /><b>LV</b></div>
              <h3>Local intelligence, visualized</h3>
              <p>Once the model finishes extracting features, images will appear here as an interactive map.</p>
              <div className="empty-facts"><span><b>01</b> Choose a dataset</span><span><b>02</b> Run your local model</span><span><b>03</b> Explore each image</span></div>
            </div>
          )}

          <section className="curation-cart-panel" data-testid="curation-cart" aria-label="Curation cart">
            <div className="curation-cart-heading">
              <div><div className="eyebrow">CURATION</div><h3>Curation cart <span data-testid="curation-cart-count">{curationCart?.item_count ?? 0}</span></h3>
                <p>Saved in the LV app cache; available after the local worker restarts.</p></div>
              <div className="curation-cart-downloads">
                <a data-testid="curation-cart-csv" href={api.curationCartCsvUrl()} download="curation_cart.csv">CSV</a>
                <a data-testid="curation-cart-json" href={api.curationCartJsonUrl()} download="curation_cart.json">JSON</a>
                <a data-testid="curation-cart-zip" href={api.curationCartZipUrl()} download="curation_cart_images.zip">Images ZIP</a>
              </div>
            </div>
            {cartLoading ? <p className="curation-cart-status" role="status">Loading saved items…</p> : null}
            {cartError && <p className="curation-cart-status curation-cart-error" role="alert">{cartError}</p>}
            {cartNotice && <p className="curation-cart-status" data-testid="curation-cart-notice" role="status">{cartNotice}</p>}
            <div className="curation-cart-toolbar">
              <label><span>Source</span><select aria-label="Filter curation cart source" value={cartSourceFilter}
                onChange={(event) => setCartSourceFilter(event.target.value)}>
                <option value="all">All sources</option>
                <option value="manual">Manual</option><option value="search">Text search</option>
                <option value="similar">Similar images</option><option value="duplicate">Duplicate pairs</option>
                <option value="outlier">Outlier ranking</option><option value="disagreement">Label disagreement</option>
              </select></label>
              <span data-testid="curation-cart-filter-count">Showing {cartFilteredItems.length} of {curationCart?.item_count ?? 0}</span>
              <label className="curation-reason-input"><span>Reason for visible items</span><input aria-label="Curation reason" maxLength={2000}
                value={cartReasonDraft} onChange={(event) => setCartReasonDraft(event.target.value)}
                placeholder="Why are these images being kept?" /></label>
              <button type="button" data-testid="curation-cart-save-reason" disabled={cartBusy || !cartFilteredItems.length}
                onClick={() => void applyCurationReason()}>Save reason</button>
              <label className="curation-clear-confirm"><input type="checkbox" aria-label="Confirm clear curation cart"
                checked={cartClearConfirm} onChange={(event) => setCartClearConfirm(event.target.checked)} />Confirm clear</label>
              <button type="button" data-testid="curation-cart-clear" disabled={cartBusy || !cartClearConfirm || !curationCart?.item_count}
                onClick={() => void clearCurationCart()}>Clear cart</button>
            </div>
            {cartFilteredItems.length ? (
              <div className="curation-cart-list">
                {cartFilteredItems.map((item) => (
                  <article className="curation-cart-item" key={item.id} data-testid="curation-cart-item" data-item-id={item.id}>
                    <div><strong>{item.label}｜{item.filename}{item.object_index === undefined || item.object_index === null ? "" : ` · object ${item.object_index + 1}`}</strong>
                      <small>{item.split} · {item.source}{item.score === null ? "" : ` · score ${item.score.toFixed(4)}`}{item.sha256 ? ` · sha ${item.sha256.slice(0, 12)}` : ""}</small>
                      {item.reason && <span className="curation-cart-reason">{item.reason}</span>}</div>
                    <button type="button" aria-label={`Remove ${item.filename} from cart`} disabled={cartBusy}
                      onClick={() => void removeFromCurationCart(item.id)}>Remove</button>
                  </article>
                ))}
              </div>
            ) : !cartLoading ? <p className="curation-cart-empty" data-testid="curation-cart-empty">{curationCart?.item_count ? "No saved items match this source." : "Your curation cart is empty."}</p> : null}
          </section>
          <footer className="canvas-footer"><span>VisualLatent · React Web</span><span>Worker API v1</span></footer>
        </main>
      </div>
      <div hidden={activeTool === "visualize"}>
        <Compare models={models} connected={connected} />
      </div>
    </div>
  );
}
