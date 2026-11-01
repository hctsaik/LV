import {
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  api,
  type EmbeddingPoint,
  type KMeansBlocks,
  type ComparisonObjectPolicy,
  type DuplicateScanInput,
  type DuplicateScanResult,
  type JobStatus,
  type ModelOption,
  type ProjectionMethod,
  type RankingMetric,
  type RankingResult,
  type SimilarSearchResult,
  type TextSearchResult,
  type VisualizeUnit,
  type VisualizationRun,
} from "./api";
import CompareDistributions from "./CompareDistributions";
import CurationWorkspace from "./CurationWorkspace";
const LazyScatterPlot2D = lazy(() => import("./ScatterPlot2D"));
const LazyScatterPlot3D = lazy(() => import("./ScatterPlot3D"));
const LazyCoverageWorkspace = lazy(() => import("./CoverageWorkspace"));

const labelColors = ["#4976d0", "#e08144", "#38a27c", "#9a6cc3", "#c95567", "#718096"];
const projectionOptions: { value: ProjectionMethod; label: string }[] = [
  { value: "pca", label: "PCA" },
  { value: "tsne", label: "t-SNE" },
  { value: "umap", label: "UMAP" },
  { value: "lda", label: "LDA (supervised)" },
  { value: "sumap", label: "Supervised UMAP" },
];

function pause(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function colorMap(points: EmbeddingPoint[]) {
  const labels = [...new Set(points.map((point) => point.label))].sort();
  return new Map(labels.map((label, index) => [label, labelColors[index % labelColors.length]]));
}

function MainScatterPlot2D({ points, selectedIndices, onToggleSelection, onAddSelection }: {
  points: EmbeddingPoint[];
  selectedIndices: number[];
  onToggleSelection: (index: number) => void;
  onAddSelection: (indices: number[]) => void;
}) {
  const [selectionMode, setSelectionMode] = useState<false | "box" | "lasso">(false);
  const [drag, setDrag] = useState<{
    start: { x: number; y: number };
    current: { x: number; y: number };
    path: Array<{ x: number; y: number }>;
  } | null>(null);
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

  function position(event: ReactPointerEvent<SVGSVGElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    return {
      x: (event.clientX - bounds.left) * width / bounds.width,
      y: (event.clientY - bounds.top) * height / bounds.height,
    };
  }

  function startSelection(event: ReactPointerEvent<SVGSVGElement>) {
    if (!selectionMode) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = position(event);
    setDrag({ start: point, current: point, path: [point] });
  }

  function updateSelection(event: ReactPointerEvent<SVGSVGElement>) {
    if (!drag) return;
    const point = position(event);
    setDrag((current) => current ? {
      ...current,
      current: point,
      path: selectionMode === "lasso" ? [...current.path, point] : current.path,
    } : current);
  }

  function finishSelection(event: ReactPointerEvent<SVGSVGElement>) {
    if (!drag || !selectionMode) return;
    const end = position(event);
    const path = selectionMode === "lasso" ? [...drag.path, end] : [];
    const insidePolygon = (x: number, y: number) => {
      let inside = false;
      for (let i = 0, j = path.length - 1; i < path.length; j = i++) {
        const a = path[i];
        const b = path[j];
        if ((a.y > y) !== (b.y > y) &&
            x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
      }
      return inside;
    };
    const picked = points.filter((point) => {
      const x = padding + ((point.x - xMin) / xSpan) * (width - padding * 2);
      const y = height - padding - ((point.y - yMin) / ySpan) * (height - padding * 2);
      if (selectionMode === "lasso") return path.length >= 3 && insidePolygon(x, y);
      return x >= Math.min(drag.start.x, end.x) && x <= Math.max(drag.start.x, end.x) &&
        y >= Math.min(drag.start.y, end.y) && y <= Math.max(drag.start.y, end.y);
    }).map((point) => point.index);
    if (picked.length) onAddSelection(picked);
    setDrag(null);
  }

  return (
    <div className="plot-frame" data-testid="embedding-plot">
      <div className="selection-mode" role="group" aria-label="2D selection mode">
        <button type="button" data-testid="embedding-selection-mode-box" aria-pressed={selectionMode === "box"} onClick={() => setSelectionMode((mode) => mode === "box" ? false : "box")}>Box select</button>
        <button type="button" data-testid="embedding-selection-mode-lasso" aria-pressed={selectionMode === "lasso"} onClick={() => setSelectionMode((mode) => mode === "lasso" ? false : "lasso")}>Lasso</button>
      </div>
      <svg
        className={`scatter ${selectionMode ? "scatter-selecting" : ""}`}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="Image embedding scatter plot"
        onPointerDown={startSelection}
        onPointerMove={updateSelection}
        onPointerUp={finishSelection}
        onPointerCancel={() => setDrag(null)}
      >
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="axis" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="axis" />
        <text x={width / 2} y={height - 8} className="axis-label">Component 1</text>
        <text x={15} y={height / 2} className="axis-label vertical-label">Component 2</text>
        {points.map((point) => {
          const x = padding + ((point.x - xMin) / xSpan) * (width - padding * 2);
          const y = height - padding - ((point.y - yMin) / ySpan) * (height - padding * 2);
          const color = colors.get(point.label) || labelColors[0];
          const active = selectedIndices.includes(point.index);
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
              aria-label={`Select ${point.filename}${point.obj_index == null ? "" : `, object ${point.obj_index + 1}`}, ${point.label}, ${point.split}`}
              aria-pressed={active}
              data-testid="embedding-point"
              data-index={point.index}
              className={`point ${active ? "point-active" : ""}`}
              transform={`translate(${x} ${y})`}
              onClick={(event) => {
                if (selectionMode) {
                  event.preventDefault();
                  event.stopPropagation();
                } else onToggleSelection(point.index);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") onToggleSelection(point.index);
              }}
            >
              <title>{`${point.filename} · ${point.label} · ${point.split}`}</title>
              {active && <circle r={12} className="point-selection-ring" />}
              {shape}
            </g>
          );
        })}
        {drag && selectionMode === "box" && (
          <rect
            className="scatter-selection-box"
            x={Math.min(drag.start.x, drag.current.x)}
            y={Math.min(drag.start.y, drag.current.y)}
            width={Math.abs(drag.current.x - drag.start.x)}
            height={Math.abs(drag.current.y - drag.start.y)}
          />
        )}
        {drag && selectionMode === "lasso" && (
          <polyline className="scatter-selection-lasso" points={drag.path.map((point) => `${point.x},${point.y}`).join(" ")} />
        )}
      </svg>
    </div>
  );
}

function ScatterPlot({ points, selectedIndices, onToggleSelection, onAddSelection, dimension, viewKey, clusterCenters, testIdPrefix }: {
  points: EmbeddingPoint[];
  selectedIndices: number[];
  onToggleSelection: (index: number) => void;
  onAddSelection: (indices: number[]) => void;
  dimension: 2 | 3;
  viewKey: string;
  clusterCenters?: KMeansBlocks["centers"];
  testIdPrefix?: string;
}) {
  if (dimension === 3) {
    return (
      <Suspense fallback={<div className="plot-frame plot-loading" role="status">Loading 3D plot…</div>}>
        <LazyScatterPlot3D
          points={points}
          selectedIndices={selectedIndices}
          onToggleSelection={onToggleSelection}
          viewKey={viewKey}
          clusterCenters={clusterCenters}
        />
      </Suspense>
    );
  }
  if (testIdPrefix !== "block") {
    return <MainScatterPlot2D points={points} selectedIndices={selectedIndices} onToggleSelection={onToggleSelection} onAddSelection={onAddSelection} />;
  }
  return (
    <Suspense fallback={<div className="plot-frame plot-loading" role="status">Loading 2D plot…</div>}>
      <LazyScatterPlot2D
        points={points}
        selectedIndices={selectedIndices}
        onToggleSelection={onToggleSelection}
        onAddSelection={onAddSelection}
        viewKey={viewKey}
        testIdPrefix={testIdPrefix}
      />
    </Suspense>
  );
}

export default function App() {
  const [analysisMode, setAnalysisMode] = useState<"classifier" | "detector">("classifier");
  const [analysisUnit, setAnalysisUnit] = useState<VisualizeUnit>("image");
  const [objectPolicy, setObjectPolicy] = useState<ComparisonObjectPolicy>({ pad: 0.12, target_res: 224, head: "cls" });
  const [classNamesDraft, setClassNamesDraft] = useState("");
  const [activeTool, setActiveTool] = useState<"visualize" | "compare" | "curation" | "coverage">("visualize");
  const [coverageMounted, setCoverageMounted] = useState(false);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  const [selectedMethods, setSelectedMethods] = useState<ProjectionMethod[]>(["sumap"]);
  const [umapReferenceEnabled, setUmapReferenceEnabled] = useState(false);
  const [umapRebuildRequested, setUmapRebuildRequested] = useState(false);
  const [activeModelId, setActiveModelId] = useState("");
  const [activeMethod, setActiveMethod] = useState<ProjectionMethod>("sumap");
  const [dimension, setDimension] = useState<2 | 3>(2);
  const [folders, setFolders] = useState<string[]>([]);
  const [folderDraft, setFolderDraft] = useState("");
  const [connected, setConnected] = useState(false);
  const [loadingModels, setLoadingModels] = useState(true);
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [run, setRun] = useState<VisualizationRun | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);
  const [rankedViewerContext, setRankedViewerContext] = useState<number[] | null>(null);
  const [rankingMetric, setRankingMetric] = useState<RankingMetric>("outlier");
  const [rankingResults, setRankingResults] = useState<RankingResult | null>(null);
  const [rankingLoading, setRankingLoading] = useState(false);
  const [rankingError, setRankingError] = useState("");
  const [rankingVisibleCount, setRankingVisibleCount] = useState(24);
  const [showAnnotationOverlay, setShowAnnotationOverlay] = useState(true);
  const [similarChain, setSimilarChain] = useState<number[]>([]);
  const [similarK, setSimilarK] = useState(9);
  const [similarResults, setSimilarResults] = useState<SimilarSearchResult | null>(null);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [similarError, setSimilarError] = useState("");
  const similarRequest = useRef(0);
  const [textQuery, setTextQuery] = useState("");
  const [textResultCount, setTextResultCount] = useState(9);
  const [textResults, setTextResults] = useState<TextSearchResult | null>(null);
  const [textSearchInput, setTextSearchInput] = useState<{ model_id: string; query: string; k: number } | null>(null);
  const [textLoading, setTextLoading] = useState(false);
  const [textError, setTextError] = useState("");
  const textRequest = useRef(0);
  const [duplicateMethod, setDuplicateMethod] = useState<"phash" | "embedding">("phash");
  const [duplicateCrossSplit, setDuplicateCrossSplit] = useState(false);
  const [duplicateHamming, setDuplicateHamming] = useState(4);
  const [duplicateDistance, setDuplicateDistance] = useState(0.05);
  const [duplicateResults, setDuplicateResults] = useState<DuplicateScanResult | null>(null);
  const [duplicateScanUsed, setDuplicateScanUsed] = useState<DuplicateScanInput | null>(null);
  const [duplicateLoading, setDuplicateLoading] = useState(false);
  const [duplicateError, setDuplicateError] = useState("");
  const duplicateRequest = useRef(0);
  const rankingRequest = useRef(0);
  const [blockCount, setBlockCount] = useState(12);
  const [selectedBlock, setSelectedBlock] = useState<number | null>(null);
  const [blockData, setBlockData] = useState<KMeansBlocks | null>(null);
  const [blockLoading, setBlockLoading] = useState(false);
  const blockRequest = useRef(0);
  const [labelFilter, setLabelFilter] = useState("all");
  const [splitFilter, setSplitFilter] = useState("all");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [curationCount, setCurationCount] = useState<number | null>(null);
  const [addingToCuration, setAddingToCuration] = useState(false);

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
    api.curationItems().then((response) => {
      if (active) setCurationCount(response.total);
    }).catch(() => {
      if (active) setCurationCount(null);
    });
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
  const viewerIndices = rankedViewerContext ?? selectedIndices;
  const selectedViewPosition = selectedIndex === null ? -1 : viewerIndices.indexOf(selectedIndex);
  const usesClassifierBlocks = run?.mode === "classifier" && run.unit === "image";
  const canSearchSimilar = run?.mode === "classifier" && run.unit === "image";
  const similarQueryIndex = similarChain[similarChain.length - 1] ?? null;
  const canSearchText = Boolean(
    run?.mode === "classifier" && run.unit === "image" && activeModelId.startsWith("chinese-clip"),
  );

  useEffect(() => {
    textRequest.current += 1;
    rankingRequest.current += 1;
    duplicateRequest.current += 1;
    setTextResults(null);
    setTextSearchInput(null);
    setTextError("");
    setTextLoading(false);
    setDuplicateResults(null);
    setDuplicateScanUsed(null);
    setDuplicateError("");
    setDuplicateLoading(false);
    setRankingResults(null);
    setRankingError("");
    setRankingLoading(false);
    setRankingVisibleCount(24);
    setRankedViewerContext(null);
  }, [job?.run_id]);

  useEffect(() => {
    const requestId = ++rankingRequest.current;
    const runId = job?.run_id;
    setRankedViewerContext(null);
    if (!runId || !run || run.records.length < 3 || !activeModelId) {
      setRankingResults(null);
      setRankingError("");
      setRankingLoading(false);
      return () => { rankingRequest.current += 1; };
    }

    setRankingResults(null);
    setRankingError("");
    setRankingLoading(true);
    api.rankings(runId, {
      model_id: activeModelId,
      metric: rankingMetric,
      label_filter: labelFilter === "all" ? undefined : labelFilter,
      split_filter: splitFilter === "all" ? undefined : splitFilter,
    }).then((result) => {
      if (rankingRequest.current !== requestId) return;
      setRankingResults(result);
      setRankingVisibleCount(24);
    }).catch((reason: unknown) => {
      if (rankingRequest.current !== requestId) return;
      setRankingError(reason instanceof Error ? reason.message : "Could not rank this run.");
    }).finally(() => {
      if (rankingRequest.current === requestId) setRankingLoading(false);
    });
    return () => { rankingRequest.current += 1; };
  }, [job?.run_id, run?.records.length, activeModelId, rankingMetric, labelFilter, splitFilter]);

  useEffect(() => {
    const requestId = ++similarRequest.current;
    const runId = job?.run_id;
    if (!similarChain.length || !runId || !canSearchSimilar || similarQueryIndex === null) {
      setSimilarResults(null);
      setSimilarLoading(false);
      setSimilarError("");
      return () => { similarRequest.current += 1; };
    }

    setSimilarResults(null);
    setSimilarError("");
    setSimilarLoading(true);
    api.similarImages(runId, {
      model_id: activeModelId,
      query_index: similarQueryIndex,
      k: similarK,
    }).then((result) => {
      if (similarRequest.current === requestId) setSimilarResults(result);
    }).catch((reason: unknown) => {
      if (similarRequest.current !== requestId) return;
      setSimilarError(reason instanceof Error ? reason.message : "Could not find similar images.");
    }).finally(() => {
      if (similarRequest.current === requestId) setSimilarLoading(false);
    });
    return () => { similarRequest.current += 1; };
  }, [job?.run_id, canSearchSimilar, similarChain.length, similarQueryIndex, activeModelId, similarK]);

  useEffect(() => {
    const requestId = ++blockRequest.current;
    const runId = job?.run_id;
    if (
      dimension !== 3 || !usesClassifierBlocks || !runId || !currentView || !currentRecords.length
    ) {
      setBlockData(null);
      setSelectedBlock(null);
      setBlockLoading(false);
      return () => { blockRequest.current += 1; };
    }

    const actualBlockCount = Math.min(blockCount, currentRecords.length);
    const blockId = selectedBlock !== null && selectedBlock < actualBlockCount
      ? selectedBlock : null;
    setBlockLoading(true);
    api.kmeansBlocks(runId, {
      model_id: activeModelId,
      method: currentView.method,
      k: blockCount,
      label_filter: labelFilter === "all" ? null : labelFilter,
      split_filter: splitFilter === "all" ? null : splitFilter,
      block_id: blockId,
    }).then((result) => {
      if (blockRequest.current !== requestId) return;
      setBlockData(result);
      const availableBlocks = result.centers.filter((center) => center.count > 0);
      if (selectedBlock === null || !availableBlocks.some((center) => center.block_id === selectedBlock)) {
        setSelectedBlock(availableBlocks[0]?.block_id ?? null);
      }
    }).catch((reason: unknown) => {
      if (blockRequest.current !== requestId) return;
      setBlockData(null);
      setError(reason instanceof Error ? reason.message : "Could not prepare KMeans blocks.");
    }).finally(() => {
      if (blockRequest.current === requestId) setBlockLoading(false);
    });
    return () => { blockRequest.current += 1; };
  }, [
    dimension, usesClassifierBlocks, job?.run_id, currentView, currentRecords.length,
    activeModelId, blockCount, labelFilter, splitFilter, selectedBlock,
  ]);

  function toggleSelection(index: number) {
    setRankedViewerContext(null);
    if (selectedIndices.includes(index)) {
      const next = selectedIndices.filter((current) => current !== index);
      setSelectedIndices(next);
      setSelectedIndex((current) => current === index ? (next[next.length - 1] ?? null) : current);
    } else {
      setSelectedIndices([...selectedIndices, index]);
      setSelectedIndex(index);
    }
  }

  function addSelection(indices: number[]) {
    const additions = [...new Set(indices)].filter((index) => run?.records[index] !== undefined);
    if (!additions.length) return;
    setRankedViewerContext(null);
    setSelectedIndices((current) => [...new Set([...current, ...additions])]);
    setSelectedIndex(additions[additions.length - 1]);
  }

  function clearSelection() {
    setRankedViewerContext(null);
    setSelectedIndices([]);
    setSelectedIndex(null);
  }

  async function addSelectedToCuration() {
    if (!job?.run_id || !selectedIndices.length || addingToCuration) return;
    setAddingToCuration(true);
    setError("");
    setNotice("");
    try {
      const result = await api.addCuration(job.run_id, selectedIndices);
      setCurationCount(result.total);
      setNotice(`Added ${result.added} item(s) to curation; skipped ${result.skipped} duplicate(s).`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not add the selected images to curation.");
    } finally {
      setAddingToCuration(false);
    }
  }

  function moveSelectedImage(direction: -1 | 1) {
    if (!viewerIndices.length) return;
    const current = selectedViewPosition >= 0 ? selectedViewPosition : 0;
    const next = Math.max(0, Math.min(viewerIndices.length - 1, current + direction));
    setSelectedIndex(viewerIndices[next]);
  }

  function openRankedImage(index: number) {
    const indices = rankingResults?.results.map((item) => item.index) || [];
    if (!indices.includes(index)) return;
    setRankedViewerContext(indices);
    setSelectedIndex(index);
  }

  async function exportRanking() {
    if (!job?.run_id || !rankingResults) return;
    try {
      const blob = await api.rankingsCsv(job.run_id, {
        model_id: rankingResults.model_id,
        metric: rankingMetric,
        label_filter: labelFilter === "all" ? undefined : labelFilter,
        split_filter: splitFilter === "all" ? undefined : splitFilter,
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${rankingMetric}_ranking.csv`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (reason) {
      setRankingError(reason instanceof Error ? reason.message : "Could not export the ranking.");
    }
  }

  function startSimilarSearch(index: number) {
    setRankedViewerContext(null);
    setSimilarK(Math.min(9, Math.max(1, (run?.records.length || 1) - 1)));
    setSelectedIndex(index);
    setSimilarChain([index]);
    setSimilarResults(null);
    setSimilarError("");
  }

  function continueSimilarSearch(index: number) {
    setSelectedIndex(index);
    setSimilarChain((current) => current[current.length - 1] === index ? current : [...current, index]);
  }

  function truncateSimilarChain(position: number) {
    setSimilarChain((current) => current.slice(0, position + 1));
  }

  function closeSimilarSearch() {
    setSimilarChain([]);
    setSimilarResults(null);
    setSimilarError("");
  }

  async function exportSimilarSearch() {
    const runId = job?.run_id;
    if (!runId || similarQueryIndex === null) return;
    try {
      const blob = await api.similarImagesCsv(runId, {
        model_id: activeModelId,
        query_index: similarQueryIndex,
        k: similarK,
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "similar_group.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (reason) {
      setSimilarError(reason instanceof Error ? reason.message : "Could not export the similar-image results.");
    }
  }

  function duplicateScanInput(): DuplicateScanInput {
    return {
      method: duplicateMethod,
      ...(duplicateMethod === "embedding" ? { model_id: activeModelId } : {}),
      max_hamming: duplicateHamming,
      max_distance: duplicateDistance,
      cross_split_only: duplicateCrossSplit,
      max_pairs: 200,
    };
  }

  async function searchByText(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const runId = job?.run_id;
    const query = textQuery.trim();
    if (!runId || !query || !canSearchText) return;
    const requestId = ++textRequest.current;
    setTextResults(null);
    setTextSearchInput(null);
    setTextError("");
    setTextLoading(true);
    try {
      const input = {
        model_id: activeModelId,
        query,
        k: textResultCount,
      };
      const result = await api.textSearch(runId, input);
      if (textRequest.current === requestId) {
        setTextResults(result);
        setTextSearchInput(input);
      }
    } catch (reason) {
      if (textRequest.current === requestId) {
        setTextError(reason instanceof Error ? reason.message : "Could not search images by text.");
      }
    } finally {
      if (textRequest.current === requestId) setTextLoading(false);
    }
  }

  async function exportTextSearch() {
    const runId = job?.run_id;
    const query = textResults?.query;
    if (!runId || !query || !textResults || !textSearchInput) return;
    try {
      const blob = await api.textSearchCsv(runId, textSearchInput);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "text_search.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (reason) {
      setTextError(reason instanceof Error ? reason.message : "Could not export text-search results.");
    }
  }

  async function scanDuplicateCandidates() {
    const runId = job?.run_id;
    if (!runId || run?.mode !== "classifier" || run.unit !== "image") return;
    const requestId = ++duplicateRequest.current;
    const input = duplicateScanInput();
    setDuplicateResults(null);
    setDuplicateScanUsed(null);
    setDuplicateError("");
    setDuplicateLoading(true);
    try {
      const result = await api.scanDuplicates(runId, input);
      if (duplicateRequest.current === requestId) {
        setDuplicateResults(result);
        setDuplicateScanUsed(input);
      }
    } catch (reason) {
      if (duplicateRequest.current === requestId) {
        setDuplicateError(reason instanceof Error ? reason.message : "Could not scan duplicate candidates.");
      }
    } finally {
      if (duplicateRequest.current === requestId) setDuplicateLoading(false);
    }
  }

  async function exportDuplicateCandidates() {
    const runId = job?.run_id;
    if (!runId || !duplicateResults || !duplicateScanUsed) return;
    try {
      const blob = await api.duplicatePairsCsv(runId, duplicateScanUsed);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "duplicate_pairs.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (reason) {
      setDuplicateError(reason instanceof Error ? reason.message : "Could not export duplicate candidates.");
    }
  }

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
    setSelectedMethods((current) =>
      current.includes(method) ? current.filter((item) => item !== method) : [...current, method],
    );
  }

  function changeAnalysisMode(mode: "classifier" | "detector") {
    if (mode === analysisMode) return;
    setAnalysisMode(mode);
    setAnalysisUnit("image");
    setUmapReferenceEnabled(false);
    setUmapRebuildRequested(false);
    setFolders([]);
    setFolderDraft("");
    setClassNamesDraft("");
    setRun(null);
    closeSimilarSearch();
    setJob(null);
    setSelectedIndex(null);
    setSelectedIndices([]);
    setSelectedBlock(null);
    setBlockData(null);
    setLabelFilter("all");
    setSplitFilter("all");
    setDimension(2);
    setError("");
    setNotice("Previous results and folder choices were cleared because the analysis mode changed.");
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
    closeSimilarSearch();
    setSelectedIndex(null);
    setSelectedIndices([]);
    setSelectedBlock(null);
    setBlockData(null);
    setLabelFilter("all");
    setSplitFilter("all");
    setDimension(2);
    setBusy(true);
    setJob(null);
    try {
      const classNames = classNamesDraft.split(",").map((name) => name.trim()).filter(Boolean);
      const created = await api.createJob(selectedFolders, selectedModelIds, selectedMethods, {
        mode: analysisMode,
        unit: analysisUnit,
        classNames,
        umapReference: umapReferenceEnabled,
        umapRebuild: umapReferenceEnabled && umapRebuildRequested,
        ...(analysisUnit === "object" ? { objectPolicy } : {}),
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
      if (umapReferenceEnabled && result.umap_reference?.enabled) setUmapRebuildRequested(false);
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
          <button type="button" className={activeTool === "visualize" ? "active" : ""} aria-current={activeTool === "visualize" ? "page" : undefined} onClick={() => setActiveTool("visualize")}>Visualize</button>
          <button type="button" className={activeTool === "compare" ? "active" : ""} aria-current={activeTool === "compare" ? "page" : undefined} onClick={() => setActiveTool("compare")}>Compare</button>
          <button type="button" className={activeTool === "coverage" ? "active" : ""} aria-current={activeTool === "coverage" ? "page" : undefined} onClick={() => { setCoverageMounted(true); setActiveTool("coverage"); }}>Coverage</button>
          <button type="button" data-testid="open-curation" className={activeTool === "curation" ? "active" : ""} aria-current={activeTool === "curation" ? "page" : undefined} onClick={() => setActiveTool("curation")}>Curation{curationCount === null ? "" : ` (${curationCount})`}</button>
        </nav>
        <div className={"connection " + (connected ? "connection-ready" : "connection-waiting")}>
          <span className="connection-dot" />
          {connected ? "Local worker connected" : "Connecting to local worker"}
        </div>
      </header>

      {coverageMounted && (
        <div className="coverage-layer" hidden={activeTool !== "coverage"}>
          <Suspense fallback={<div className="coverage-loading" role="status">Loading coverage workspace…</div>}>
            <LazyCoverageWorkspace
              models={models}
              connected={connected}
              loadingModels={loadingModels}
              onCurationCountChange={setCurationCount}
            />
          </Suspense>
        </div>
      )}

      {activeTool === "curation" ? (
        <CurationWorkspace onCountChange={setCurationCount} />
      ) : activeTool === "compare" ? (
        <CompareDistributions models={models} loadingModels={loadingModels} connected={connected} />
      ) : activeTool === "coverage" ? null : <div className="workspace">
        <aside className="control-panel">
          <div className="eyebrow">VISUALIZE EMBEDDINGS</div>
          <h1>Explore your image space</h1>
          <p className="intro">Compare how your local image collections group together using the models already installed on this device.</p>

          <section className="control-section">
            <div className="section-heading"><span className="step-number">00</span><h2>Analysis mode</h2></div>
            <label className="field-label" htmlFor="analysis-mode">Visualize dataset type</label>
            <select
              id="analysis-mode"
              data-testid="analysis-mode"
              aria-label="Analysis mode"
              value={analysisMode}
              onChange={(event) => changeAnalysisMode(event.target.value as "classifier" | "detector")}
              disabled={busy}
            >
              <option value="classifier">Image Classifier</option>
              <option value="detector">Object Detector (whole image)</option>
            </select>
            {analysisMode === "detector" && (
              <>
                <label className="field-label field-label-spaced" htmlFor="detector-unit">Detection unit</label>
                <select
                  id="detector-unit"
                  data-testid="detector-unit"
                  aria-label="Detection unit"
                  value={analysisUnit}
                  onChange={(event) => {
                    const value = event.target.value as VisualizeUnit;
                    setAnalysisUnit(value);
                    if (value === "object") {
                      setUmapReferenceEnabled(false);
                      setUmapRebuildRequested(false);
                    }
                    setRun(null);
                    setJob(null);
                    setSelectedIndex(null);
                    setLabelFilter("all");
                    setSplitFilter("all");
                  }}
                  disabled={busy}
                >
                  <option value="image">Whole image</option>
                  <option value="object">Each annotated object</option>
                </select>
                {analysisUnit === "object" && (
                  <div className="object-policy-fields" aria-label="Object crop policy">
                    <label className="field-label" htmlFor="object-pad">Crop padding</label>
                    <input
                      id="object-pad"
                      aria-label="Object crop padding"
                      type="number"
                      min="0"
                      max="0.5"
                      step="0.01"
                      value={objectPolicy.pad}
                      onChange={(event) => setObjectPolicy((current) => ({ ...current, pad: Number(event.target.value) }))}
                      disabled={busy}
                    />
                    <label className="field-label" htmlFor="object-resolution">Crop resolution</label>
                    <select
                      id="object-resolution"
                      aria-label="Object crop resolution"
                      value={objectPolicy.target_res}
                      onChange={(event) => setObjectPolicy((current) => ({ ...current, target_res: Number(event.target.value) as ComparisonObjectPolicy["target_res"] }))}
                      disabled={busy}
                    >
                      {[168, 196, 224, 280, 336, 392, 448].map((resolution) => <option key={resolution} value={resolution}>{resolution}px</option>)}
                    </select>
                    <label className="field-label" htmlFor="object-head">Embedding head</label>
                    <select
                      id="object-head"
                      aria-label="Object embedding head"
                      value={objectPolicy.head}
                      onChange={(event) => setObjectPolicy((current) => ({ ...current, head: event.target.value as ComparisonObjectPolicy["head"] }))}
                      disabled={busy}
                    >
                      <option value="cls">CLS token</option>
                      <option value="meanpool">Mean pool</option>
                    </select>
                    <div className="field-note">Each YOLO box is cropped with the selected policy; embeddings are cached by image content and box.</div>
                  </div>
                )}
                <label className="field-label field-label-spaced" htmlFor="detector-class-names">Class names fallback</label>
                <input
                  id="detector-class-names"
                  data-testid="detector-class-names"
                  aria-label="Detector class names fallback"
                  value={classNamesDraft}
                  onChange={(event) => setClassNamesDraft(event.target.value)}
                  placeholder="Used only when classes.txt is unavailable"
                  disabled={busy}
                />
                <div className="field-note">A dataset classes.txt takes priority. Enter comma-separated names only when the dataset has no class file.</div>
              </>
            )}
          </section>

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
            {analysisMode === "classifier" && analysisUnit === "image" && selectedMethods.includes("umap") && (
              <div className="umap-reference-controls" data-testid="umap-reference-controls">
                <label className="choice-option">
                  <input
                    type="checkbox"
                    aria-label="Keep UMAP reference frame"
                    checked={umapReferenceEnabled}
                    disabled={busy}
                    onChange={(event) => {
                      setUmapReferenceEnabled(event.target.checked);
                      if (!event.target.checked) setUmapRebuildRequested(false);
                    }}
                  />
                  <span>Keep UMAP reference frame</span>
                </label>
                <div className="field-note">Freeze the first UMAP layout in the app cache. Later images are transformed into it, so existing points stay fixed. Rebuild after a major dataset change.</div>
                {umapReferenceEnabled && (
                  <button
                    className="secondary-action"
                    type="button"
                    disabled={busy}
                    onClick={() => setUmapRebuildRequested(true)}
                  >
                    Rebuild UMAP reference next run
                  </button>
                )}
                {umapRebuildRequested && <p className="umap-rebuild-pending" role="status">The next analysis will refit and replace each selected model's reference frame.</p>}
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
              <p>{run && currentView ? run.records.length.toLocaleString() + (run.unit === "object" ? " objects" : " images") + " · " + activeModelId + " · " + currentView.method.toUpperCase() : "Choose a folder and run an analysis to see your collection."}</p>
            </div>
            {run && <div className="run-badge"><span className="badge-dot" /> Result ready</div>}
          </div>

          {run?.umap_reference?.enabled && currentView?.method === "umap" && (
            <div className="umap-reference-result" data-testid="umap-reference-result" role="status">
              {run.umap_reference.by_model[activeModelId]?.refitted
                ? `UMAP reference fitted for ${activeModelId}.`
                : run.umap_reference.by_model[activeModelId]
                  ? `UMAP reference reused for ${activeModelId}; ${run.umap_reference.by_model[activeModelId].new_points} new point(s) transformed.`
                  : `UMAP reference was unavailable for ${activeModelId}. Check the run warning.`}
            </div>
          )}

          {busy && (
            <div className="progress-card" data-testid="job-status" role="status">
              <div className="progress-row"><span>{job?.message || "Preparing analysis"}</span><strong>{Math.round(job?.percent || 0)}%</strong></div>
              <div className="progress-track"><div className="progress-fill" style={{ width: Math.max(3, job?.percent || 0) + "%" }} /></div>
              <div className="progress-foot">{job?.done || 0} / {job?.total || "—"} model-{analysisUnit === "object" ? "object" : "image"} embeddings</div>
            </div>
          )}
          {!busy && job?.state === "completed" && <div className="completion-label" data-testid="job-status"><span className="completion-check">✓</span> Completed</div>}
          {!busy && job?.state === "cancelled" && <div className="completion-label" data-testid="job-status">Cancelled</div>}

          {run && currentView ? (
            <>
            <div className="result-grid">
              <section className="plot-card">
                <div className="plot-toolbar">
                  <div className="plot-title"><span className="chart-glyph" aria-hidden="true">⊙</span><div><strong>{run.unit === "object" ? "Object embeddings" : "Image embeddings"}</strong><small>Click a point to inspect its source{run.unit === "object" ? " object" : ""}</small></div></div>
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
                </div>
                {currentRecords.length ? (
                  <ScatterPlot
                    points={currentRecords}
                    selectedIndices={selectedIndices}
                    onToggleSelection={toggleSelection}
                    onAddSelection={addSelection}
                    dimension={dimension}
                    viewKey={activeModelId + "-" + currentView.method + "-" + splitFilter + "-" + labelFilter}
                    clusterCenters={dimension === 3 && usesClassifierBlocks ? blockData?.centers : undefined}
                  />
                ) : <div className="empty-filter">No images match these filters.</div>}
                {dimension === 3 && usesClassifierBlocks && currentRecords.length > 0 && (
                  <section className="block-drilldown" data-testid="kmeans-blocks">
                    <div className="block-controls">
                      <label>
                        <span>區塊數 K: {blockCount}</span>
                        <input
                          type="range"
                          min={4}
                          max={30}
                          value={blockCount}
                          aria-label="KMeans block count"
                          data-testid="kmeans-count"
                          onChange={(event) => setBlockCount(Number(event.target.value))}
                        />
                      </label>
                      <label>
                        <span>進入 3D 區塊</span>
                        <select
                          aria-label="Enter KMeans block"
                          data-testid="cluster-block-select"
                          value={selectedBlock ?? ""}
                          disabled={blockLoading || !blockData?.centers.length}
                          onChange={(event) => setSelectedBlock(Number(event.target.value))}
                        >
                          {blockData?.centers.filter((center) => center.count > 0).map((center) => (
                            <option key={center.block_id} value={center.block_id}>
                              #{center.block_id}（{center.count}）
                            </option>
                          ))}
                        </select>
                      </label>
                      <span className="muted-line" role="status" data-testid="kmeans-status">
                        {blockLoading ? "正在建立區塊…" : blockData ? `3D 已標記 ${blockData.k} 個區塊` : ""}
                      </span>
                    </div>
                    {selectedBlock !== null && (
                      <div className="block-drilldown-chart" data-testid="block-drilldown">
                        <div className="block-drilldown-heading">
                          <strong>區塊 #{selectedBlock} · 2D 框選／套索加入選取</strong>
                          <span>{blockData?.block_id === selectedBlock ? blockData.drilldown_points.length : 0} points</span>
                        </div>
                        {blockData?.block_id === selectedBlock && blockData.drilldown_points.length ? (
                          <ScatterPlot
                            points={blockData.drilldown_points}
                            selectedIndices={selectedIndices}
                            onToggleSelection={toggleSelection}
                            onAddSelection={addSelection}
                            dimension={2}
                            viewKey={`${activeModelId}-${currentView.method}-${selectedBlock}-${blockCount}`}
                            testIdPrefix="block"
                          />
                        ) : (
                          <div className="muted-line" role="status">載入區塊內的 2D 投影…</div>
                        )}
                      </div>
                    )}
                  </section>
                )}
                <div className="plot-legend">
                  {labels.map((label) => <span key={label}><i style={{ background: colorMap(currentView.points).get(label) || labelColors[0] }} />{label}</span>)}
                  <span className="legend-separator" />
                  <small>● train　■ test　◆ valid</small>
                  <span data-testid="selection-count">{selectedIndices.length} selected</span>
                  {selectedIndices.length > 0 && <button type="button" data-testid="add-selection-to-curation" className="add-curation-selection" disabled={addingToCuration || !job?.run_id} onClick={() => void addSelectedToCuration()}>{addingToCuration ? "Adding…" : "Add selected to curation"}</button>}
                  {selectedIndices.length > 0 && <button type="button" className="clear-selection" onClick={clearSelection}>Clear selection</button>}
                </div>
                {selectedIndices.length > 0 && (
                  <section className="selection-gallery" data-testid="selection-gallery" aria-label="Selected image thumbnails">
                    <div className="selection-gallery-heading">
                      <strong>Selected images</strong><span>{selectedIndices.length}</span>
                    </div>
                    <div className="selection-gallery-items">
                      {selectedIndices.map((index) => {
                        const record = run.records[index];
                        if (!record) return null;
                        return (
                          <button
                            type="button"
                            key={index}
                            data-testid="selection-thumbnail"
                            data-index={index}
                            aria-label={`Open ${record.filename}`}
                            aria-pressed={selectedIndex === index}
                            onClick={() => { setRankedViewerContext(null); setSelectedIndex(index); }}
                          >
                            <img
                              src={`/api/v1/runs/${encodeURIComponent(job?.run_id || "")}/images/${index}/thumbnail`}
                              alt=""
                              loading="lazy"
                            />
                            <span title={record.filename}>{record.filename}</span>
                          </button>
                        );
                      })}
                    </div>
                  </section>
                )}
              </section>

              <aside className="detail-card">
                <div className="detail-card-heading"><div><div className="eyebrow">{selectedRecord && run.unit === "object" ? "OBJECT DETAIL" : "IMAGE DETAIL"}</div><h3>{selectedRecord ? (run.unit === "object" ? "Selected object" : "Selected image") : "Select a point"}</h3></div><span className="detail-count">{selectedRecord ? "#" + (selectedRecord.index + 1) : "—"}</span></div>
                {selectedRecord && run ? (
                  <>
                    <div className="image-preview"><div className="image-stage"><img
                      data-testid="selected-image"
                      src={"/api/v1/runs/" + encodeURIComponent(job?.run_id || "") + "/images/" + selectedRecord.index +
                        (run.mode === "detector" && run.unit === "image" && showAnnotationOverlay ? "/overlay" : "")}
                      alt={selectedRecord.filename}
                    />{run.unit === "object" && selectedRecord.bbox && <svg data-testid="selected-object-bbox" aria-label="Selected object bounding box" viewBox="0 0 1 1" preserveAspectRatio="none"><rect x={selectedRecord.bbox[0] - selectedRecord.bbox[2] / 2} y={selectedRecord.bbox[1] - selectedRecord.bbox[3] / 2} width={selectedRecord.bbox[2]} height={selectedRecord.bbox[3]} /></svg>}</div></div>
                    <div className="image-name" data-testid="selected-image-name" title={selectedRecord.filename}>{selectedRecord.filename}</div>
                    {selectedViewPosition >= 0 && (
                      <div className="selected-image-navigation" data-testid="selected-image-navigation">
                        <button type="button" aria-label={rankedViewerContext ? "Previous ranked image" : "Previous selected image"} data-testid="selected-image-previous" disabled={selectedViewPosition <= 0} onClick={() => moveSelectedImage(-1)}>◀</button>
                        <span>{selectedViewPosition + 1} / {viewerIndices.length} {rankedViewerContext ? "ranked" : "selected"}</span>
                        <button type="button" aria-label={rankedViewerContext ? "Next ranked image" : "Next selected image"} data-testid="selected-image-next" disabled={selectedViewPosition >= viewerIndices.length - 1} onClick={() => moveSelectedImage(1)}>▶</button>
                      </div>
                    )}
                    {run.mode === "detector" && run.unit === "image" && run.class_names.length > 0 && (
                      <label className="annotation-overlay-toggle">
                        <input
                          type="checkbox"
                          aria-label="Show annotation boxes"
                          checked={showAnnotationOverlay}
                          onChange={(event) => setShowAnnotationOverlay(event.target.checked)}
                        />
                        <span>Show annotation boxes</span>
                      </label>
                    )}
                    <div className="metadata-list">
                      <div><span>Class</span><strong>{selectedRecord.label}</strong></div>
                      <div><span>Split</span><strong>{selectedRecord.split}</strong></div>
                      {run.unit === "object" && <>
                        <div><span>Object</span><strong data-testid="selected-object-index">{(selectedRecord.obj_index ?? 0) + 1}</strong></div>
                        <div><span>Box (cx, cy, w, h)</span><strong data-testid="selected-object-box">{selectedRecord.bbox?.map((value) => value.toFixed(3)).join(", ")}</strong></div>
                      </>}
                      <div><span>Model</span><strong>{activeModelId}</strong></div>
                      <div><span>Method</span><strong>{currentView.method.toUpperCase()}</strong></div>
                    </div>
                    {canSearchSimilar && (
                      <button
                        type="button"
                        className="similar-start-button"
                        data-testid="similar-start"
                        onClick={() => startSimilarSearch(selectedRecord.index)}
                      >
                        Find similar images
                      </button>
                    )}
                    {canSearchSimilar && similarChain.length > 0 && (
                      <section className="similar-panel" data-testid="similar-panel" aria-label="Similar images">
                        <div className="similar-heading">
                          <div><div className="eyebrow">IMAGE SEARCH</div><h4>Similar images</h4></div>
                          <button type="button" aria-label="Close similar search" onClick={closeSimilarSearch}>×</button>
                        </div>
                        <div className="similar-chain" aria-label="Query history">
                          {similarChain.map((index, position) => (
                            <button
                              type="button"
                              key={`${position}-${index}`}
                              data-testid="similar-query-chip"
                              aria-current={position === similarChain.length - 1 ? "step" : undefined}
                              onClick={() => truncateSimilarChain(position)}
                            >
                              #{index + 1}
                            </button>
                          ))}
                        </div>
                        <p className="similar-caption">
                          Query: {run.records[similarQueryIndex ?? selectedRecord.index]?.filename}. Cosine distance is sorted ascending.
                        </p>
                        <label className="similar-k-field">
                          <span>Results (k)</span>
                          <input
                            aria-label="Similar result count"
                            type="number"
                            min={1}
                            max={Math.max(1, run.records.length - 1)}
                            value={Math.min(similarK, Math.max(1, run.records.length - 1))}
                            onChange={(event) => {
                              const value = Number(event.target.value);
                              if (Number.isInteger(value) && value >= 1) setSimilarK(value);
                            }}
                          />
                        </label>
                        {similarLoading && <div className="similar-status" role="status">Searching {activeModelId} embeddings…</div>}
                        {similarError && <div className="similar-error" role="alert">{similarError}</div>}
                        {!similarLoading && similarResults && (
                          <>
                            <div className="similar-status" data-testid="similar-status">
                              {similarResults.results.length} result(s) · {similarResults.model_id}
                            </div>
                            <div className="similar-results">
                              {similarResults.results.map(({ index, distance }) => {
                                const record = run.records[index];
                                return (
                                  <article className="similar-result" data-testid="similar-result" data-index={index} key={index}>
                                    <img
                                      src={`/api/v1/runs/${encodeURIComponent(job?.run_id || "")}/images/${index}`}
                                      alt={record.filename}
                                      loading="lazy"
                                    />
                                    <div className="similar-result-info">
                                      <strong title={record.filename}>{record.filename}</strong>
                                      <span>{record.label} · d={distance.toFixed(4)}</span>
                                    </div>
                                    <button
                                      type="button"
                                      aria-label={`Use ${record.filename} as query`}
                                      onClick={() => continueSimilarSearch(index)}
                                    >
                                      ↻
                                    </button>
                                  </article>
                                );
                              })}
                            </div>
                            <button
                              type="button"
                              className="similar-export-button"
                              data-testid="similar-export"
                              disabled={similarLoading}
                              onClick={exportSimilarSearch}
                            >
                              Export query and results CSV
                            </button>
                          </>
                        )}
                      </section>
                    )}
                  </>
                ) : (
                  <div className="detail-empty"><div className="image-placeholder" aria-hidden="true">▧</div><p>Select a point in the scatter plot to preview the matching image and its metadata.</p></div>
                )}
              </aside>
            </div>
            {run.mode === "classifier" && run.unit === "image" && (
              <section className="search-workspace" data-testid="search-review-workspace" aria-label="Text search and duplicate review">
                <div className="search-workspace-heading">
                  <div><div className="eyebrow">IMAGE DISCOVERY</div><h3>Text search and duplicate review</h3></div>
                  <span>Cosine and dHash matches are review candidates, not automatic decisions.</span>
                </div>
                <div className="search-tool-grid">
                  <section className="search-tool-card" aria-label="Text search">
                    <div className="search-tool-title"><div><div className="eyebrow">CHINESE-CLIP</div><h4>Search by text</h4></div></div>
                    <p className="search-tool-note">繁體與簡體中文查詢會沿用 Chinese-CLIP 文字塔；英文亦可查詢。</p>
                    <form className="search-form" onSubmit={(event) => void searchByText(event)}>
                      <label>
                        <span>Text query</span>
                        <input
                          aria-label="Chinese-CLIP text query"
                          value={textQuery}
                          maxLength={512}
                          placeholder="例如：斑馬、夜間反光、zebra"
                          onChange={(event) => setTextQuery(event.target.value)}
                        />
                      </label>
                      <label className="search-count-field">
                        <span>Results</span>
                        <input
                          aria-label="Text result count"
                          type="number"
                          min={1}
                          max={Math.max(1, run.records.length)}
                          value={Math.min(textResultCount, Math.max(1, run.records.length))}
                          onChange={(event) => {
                            const value = Number(event.target.value);
                            if (Number.isInteger(value) && value >= 1) setTextResultCount(value);
                          }}
                        />
                      </label>
                      <button type="submit" disabled={!canSearchText || !textQuery.trim() || textLoading || busy}>
                        {textLoading ? "Searching…" : "Search images"}
                      </button>
                    </form>
                    {!canSearchText && <p className="search-tool-note" role="status">Run Chinese-CLIP image embeddings and select that model above to enable text search.</p>}
                    {textError && <div className="search-error" role="alert">{textError}</div>}
                    {textLoading && <div className="search-status" role="status">Encoding query with {activeModelId}…</div>}
                    {!textLoading && textResults && (
                      <>
                        <div className="search-status" data-testid="text-search-status" role="status">
                          “{textResults.query}” · {textResults.results.length} result(s) · {textResults.model_id} · cosine distance ascending
                        </div>
                        {textResults.results.length ? (
                          <div className="text-search-results">
                            {textResults.results.map(({ index, distance }) => {
                              const record = run.records[index];
                              return (
                                <article className="text-search-result" data-testid="text-search-result" data-index={index} key={index}>
                                  <img src={`/api/v1/runs/${encodeURIComponent(job?.run_id || "")}/images/${index}/thumbnail`} alt={record.filename} loading="lazy" />
                                  <strong title={record.filename}>{record.filename}</strong>
                                  <small>{record.label} · {record.split} · d={distance.toFixed(4)}</small>
                                  <button type="button" aria-label={`Use ${record.filename} as visual search query`} onClick={() => startSimilarSearch(index)}>Find similar images</button>
                                </article>
                              );
                            })}
                          </div>
                        ) : <p className="search-tool-note">No matching images were found.</p>}
                        <button type="button" className="search-export-button" data-testid="text-search-export" onClick={() => void exportTextSearch()}>Export text-search CSV</button>
                      </>
                    )}
                  </section>

                  <section className="search-tool-card" aria-label="Duplicate review">
                    <div className="search-tool-title"><div><div className="eyebrow">REVIEW CANDIDATES</div><h4>Duplicates and split leakage</h4></div></div>
                    <p className="search-tool-note">dHash finds visual near-copies; embedding finds semantic near-duplicates. Review each pair before taking action.</p>
                    <div className="duplicate-controls">
                      <label>
                        <span>Method</span>
                        <select aria-label="Duplicate method" value={duplicateMethod} onChange={(event) => setDuplicateMethod(event.target.value as "phash" | "embedding")}>
                          <option value="phash">dHash (visual)</option>
                          <option value="embedding">Embedding (semantic)</option>
                        </select>
                      </label>
                      {duplicateMethod === "phash" ? (
                        <label>
                          <span>Hamming ≤ {duplicateHamming}</span>
                          <input aria-label="dHash distance threshold" type="range" min={0} max={16} value={duplicateHamming} onChange={(event) => setDuplicateHamming(Number(event.target.value))} />
                        </label>
                      ) : (
                        <label>
                          <span>Cosine ≤ {duplicateDistance.toFixed(2)}</span>
                          <input aria-label="Embedding distance threshold" type="range" min={0} max={0.5} step={0.01} value={duplicateDistance} onChange={(event) => setDuplicateDistance(Number(event.target.value))} />
                        </label>
                      )}
                      <label className="duplicate-cross-split">
                        <input type="checkbox" aria-label="Only cross-split pairs" checked={duplicateCrossSplit} onChange={(event) => setDuplicateCrossSplit(event.target.checked)} />
                        <span>Only cross-split pairs</span>
                      </label>
                      <button type="button" onClick={() => void scanDuplicateCandidates()} disabled={duplicateLoading || busy}>
                        {duplicateLoading ? "Scanning…" : "Scan candidates"}
                      </button>
                    </div>
                    {duplicateError && <div className="search-error" role="alert">{duplicateError}</div>}
                    {duplicateLoading && <div className="search-status" role="status">Comparing {duplicateMethod === "phash" ? "perceptual hashes" : `${activeModelId} embeddings`}…</div>}
                    {!duplicateLoading && duplicateResults && (
                      <>
                        <div className="search-status" data-testid="duplicate-status" role="status">
                          {duplicateResults.results.length} pair(s) · {duplicateResults.method === "phash" ? "dHash" : duplicateResults.model_id + " embedding"}
                          {duplicateResults.method === "phash" ? ` · Hamming ≤ ${duplicateScanUsed?.max_hamming}` : ` · cosine ≤ ${duplicateScanUsed?.max_distance?.toFixed(2)}`}
                          {duplicateResults.cross_split_only ? " · cross-split only" : " · all splits"}
                        </div>
                        {duplicateResults.results.length ? (
                          <div className="duplicate-results">
                            {duplicateResults.results.slice(0, 50).map(({ left_index, right_index, distance }) => {
                              return (
                                <article className="duplicate-pair" data-testid="duplicate-result" data-pair={`${left_index}-${right_index}`} key={`${left_index}-${right_index}`}>
                                  {[left_index, right_index].map((index) => {
                                    const record = run.records[index];
                                    return (
                                      <button type="button" className="duplicate-image" key={index} onClick={() => { setRankedViewerContext(null); setSelectedIndex(index); }} title={`Open ${record.filename}`}>
                                        <img src={`/api/v1/runs/${encodeURIComponent(job?.run_id || "")}/images/${index}/thumbnail`} alt="" loading="lazy" />
                                        <strong>{record.filename}</strong>
                                        <small>{record.label} · {record.split}</small>
                                      </button>
                                    );
                                  })}
                                  <span className="duplicate-distance">d={duplicateMethod === "phash" ? distance : distance.toFixed(4)}</span>
                                </article>
                              );
                            })}
                          </div>
                        ) : <p className="search-tool-note">No duplicate candidates under the current settings.</p>}
                        {duplicateResults.results.length > 0 && <button type="button" className="search-export-button" data-testid="duplicate-export" onClick={() => void exportDuplicateCandidates()}>Export candidate pairs CSV</button>}
                      </>
                    )}
                  </section>
                </div>
              </section>
            )}
            {run.records.length >= 3 && (
              <section className="ranking-workspace" data-testid="ranking-workspace" aria-label="Review ranking">
                <div className="search-workspace-heading">
                  <div><div className="eyebrow">REVIEW CLUES</div><h3>Outlier and label-disagreement ranking</h3></div>
                  <label className="ranking-criterion">
                    <span>Rank by</span>
                    <select
                      aria-label="Ranking criterion"
                      data-testid="ranking-criterion"
                      value={rankingMetric}
                      onChange={(event) => setRankingMetric(event.target.value as RankingMetric)}
                    >
                      <option value="outlier">Outlier distance</option>
                      <option value="disagreement">Label disagreement</option>
                    </select>
                  </label>
                </div>
                <p className="ranking-caveat" data-testid="ranking-interpretation">
                  Higher values are review clues only. They do not determine image quality or confirm a labeling error.
                </p>
                {rankingError && <div className="search-error" role="alert">{rankingError}</div>}
                {rankingLoading && <div className="search-status" role="status" data-testid="ranking-loading">Calculating ranking…</div>}
                {!rankingLoading && rankingResults && (
                  <>
                    <div className="search-status" role="status" data-testid="ranking-status">
                      {rankingResults.results.length} record(s) · {rankingResults.model_id} · k={rankingResults.k} · highest first
                    </div>
                    {rankingResults.results.length ? (
                      <div className="ranked-results" data-testid="ranked-results">
                        {rankingResults.results.slice(0, rankingVisibleCount).map(({ rank, index, score }) => {
                          const record = run.records[index];
                          if (!record) return null;
                          return (
                            <article
                              className="ranked-result"
                              data-testid="ranked-result"
                              data-rank={rank}
                              data-index={index}
                              key={index}
                            >
                              <button
                                type="button"
                                aria-label={`Open rank ${rank}: ${record.filename}`}
                                aria-current={selectedIndex === index && rankedViewerContext ? "true" : undefined}
                                onClick={() => openRankedImage(index)}
                              >
                                <img
                                  src={`/api/v1/runs/${encodeURIComponent(job?.run_id || "")}/images/${index}/thumbnail`}
                                  alt=""
                                  loading="lazy"
                                />
                                <span className="ranked-result-number">#{rank}</span>
                                <strong title={record.filename}>{record.filename}</strong>
                                <small>{record.label} · {record.split} · {score.toFixed(4)}</small>
                              </button>
                            </article>
                          );
                        })}
                      </div>
                    ) : (
                      <p className="search-tool-note">No records match the current class and split filters.</p>
                    )}
                    <div className="ranking-actions">
                      {rankingVisibleCount < rankingResults.results.length && (
                        <button type="button" onClick={() => setRankingVisibleCount((count) => count + 24)}>
                          Show more ({rankingResults.results.length - rankingVisibleCount} remaining)
                        </button>
                      )}
                      <button type="button" data-testid="ranking-export" onClick={() => void exportRanking()}>
                        Export ranked CSV
                      </button>
                    </div>
                  </>
                )}
              </section>
            )}
            </>
          ) : (
            <div className="empty-canvas">
              <div className="empty-illustration" aria-hidden="true"><span className="orbit orbit-one" /><span className="orbit orbit-two" /><i className="dot dot-one" /><i className="dot dot-two" /><i className="dot dot-three" /><i className="dot dot-four" /><b>LV</b></div>
              <h3>Local intelligence, visualized</h3>
              <p>Once the model finishes extracting features, images will appear here as an interactive map.</p>
              <div className="empty-facts"><span><b>01</b> Choose a dataset</span><span><b>02</b> Run your local model</span><span><b>03</b> Explore each image</span></div>
            </div>
          )}
          <footer className="canvas-footer"><span>VisualLatent · React Web</span><span>Worker API v1</span></footer>
        </main>
      </div>
      }
    </div>
  );
}
