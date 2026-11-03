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
  type DiagnosticExplanationInput,
  type DiagnosticExplanationResult,
  type DiagnosticHeadStatus,
  type DiagnosticReference,
  type DuplicateScanInput,
  type DuplicateScanResult,
  type DuplicateLabelingHandoffResult,
  type ActiveLearningRound,
  type JobStatus,
  type ModelOption,
  type ObjectPolicyAutotuneResult,
  type ProjectionMethod,
  type RankingMetric,
  type RankingResult,
  type SimilarSearchResult,
  type TextSearchResult,
  type VisualizeUnit,
  type VisualizationRun,
} from "./api";
import ClassSimilarityWorkspace from "./ClassSimilarityWorkspace";
import CompareDistributions from "./CompareDistributions";
import CurationWorkspace from "./CurationWorkspace";
import EvaluationWorkspace from "./EvaluationWorkspace";
import FalsePositiveCaseWorkspace from "./FalsePositiveCaseWorkspace";
const LazyScatterPlot2D = lazy(() => import("./ScatterPlot2D"));
const LazyScatterPlot3D = lazy(() => import("./ScatterPlot3D"));
const LazyGroupDiffWorkspace = lazy(() => import("./GroupDiffWorkspace"));
const LazyCoverageWorkspace = lazy(() => import("./CoverageWorkspace"));
const LazyAnomalyBankWorkspace = lazy(() => import("./AnomalyBankWorkspace"));
const LazyFewshotSampleBankWorkspace = lazy(() => import("./FewshotSampleBankWorkspace"));

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

function objectPolicyLabel(policy: ComparisonObjectPolicy) {
  return "pad" + Math.round(policy.pad * 100) + "_r" + policy.target_res + "_" + policy.head;
}

function colorMap(points: EmbeddingPoint[]) {
  const labels = [...new Set(points.map((point) => point.label))].sort();
  return new Map(labels.map((label, index) => [label, labelColors[index % labelColors.length]]));
}

function DiagnosticEvidenceGrid({ result }: { result: DiagnosticExplanationResult }) {
  const [rows, columns] = result.grid;
  return <div
    className="diagnostic-evidence-grid"
    data-testid="diagnostic-evidence-grid"
    role="img"
    aria-label={`${result.method === "gradcam" ? "Signed Grad-CAM" : "DINO attention"} patch evidence, ${rows} by ${columns}`}
    style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
  >
    {result.display_scores.flatMap((line, row) => line.map((score, column) => {
      const value = Number.isFinite(score) ? Math.max(-1, Math.min(1, score)) : 0;
      const opacity = 0.08 + Math.abs(value) * 0.82;
      const color = result.score_mode === "signed" && value < 0
        ? `rgba(42, 99, 183, ${opacity})`
        : `rgba(204, 54, 45, ${opacity})`;
      return <span
        key={`${row}-${column}`}
        title={`Patch ${row + 1}, ${column + 1}: ${value.toFixed(4)}`}
        style={{ backgroundColor: color }}
      />;
    }))}
  </div>;
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
  const [objectPolicySubmitting, setObjectPolicySubmitting] = useState(false);
  const [objectPolicyJob, setObjectPolicyJob] = useState<JobStatus | null>(null);
  const [objectPolicyTuneResult, setObjectPolicyTuneResult] = useState<{
    jobId: string;
    scope: string;
    result: ObjectPolicyAutotuneResult;
  } | null>(null);
  const [objectPolicyProfileName, setObjectPolicyProfileName] = useState("");
  const [classNamesDraft, setClassNamesDraft] = useState("");
  const [activeTool, setActiveTool] = useState<"visualize" | "diagnostics" | "class_similarity" | "compare" | "groupdiff" | "curation" | "coverage" | "false_positive" | "evaluation" | "anomaly" | "fewshot">("visualize");
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
  const [diagnosticReference, setDiagnosticReference] = useState<DiagnosticReference | null>(null);
  const [diagnosticResolution, setDiagnosticResolution] = useState<224 | 336 | 448>(224);
  const [diagnosticHeadStatus, setDiagnosticHeadStatus] = useState<DiagnosticHeadStatus | null>(null);
  const [diagnosticHeadLoading, setDiagnosticHeadLoading] = useState(false);
  const [diagnosticHeadError, setDiagnosticHeadError] = useState("");
  const [diagnosticMethod, setDiagnosticMethod] = useState<"attention" | "gradcam">("attention");
  const [diagnosticAttentionMethod, setDiagnosticAttentionMethod] = useState<DiagnosticExplanationInput["attention_method"]>("last_layer_head_mean");
  const [diagnosticHeadIndex, setDiagnosticHeadIndex] = useState(0);
  const [diagnosticTargetClass, setDiagnosticTargetClass] = useState("");
  const [diagnosticDisplayMode, setDiagnosticDisplayMode] = useState<"signed" | "positive">("signed");
  const [diagnosticExplanation, setDiagnosticExplanation] = useState<DiagnosticExplanationResult | null>(null);
  const [diagnosticExplanationLoading, setDiagnosticExplanationLoading] = useState(false);
  const [diagnosticExplanationError, setDiagnosticExplanationError] = useState("");
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
  const [duplicateHandoff, setDuplicateHandoff] = useState<DuplicateLabelingHandoffResult | null>(null);
  const [duplicateIncludeHandled, setDuplicateIncludeHandled] = useState(false);
  const [duplicateNewRound, setDuplicateNewRound] = useState(true);
  const [duplicateRounds, setDuplicateRounds] = useState<ActiveLearningRound[]>([]);
  const [duplicateRoundId, setDuplicateRoundId] = useState("");
  const [duplicateHandoffLoading, setDuplicateHandoffLoading] = useState(false);
  const [duplicateLoading, setDuplicateLoading] = useState(false);
  const [duplicateError, setDuplicateError] = useState("");
  const duplicateRequest = useRef(0);
  const duplicateHandoffRequest = useRef(0);
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
  const currentRunId = run && job?.run_id ? job.run_id : null;
  const viewerIndices = rankedViewerContext ?? selectedIndices;
  const selectedViewPosition = selectedIndex === null ? -1 : viewerIndices.indexOf(selectedIndex);

  useEffect(() => {
    if (activeTool !== "diagnostics" || !diagnosticReference) return;
    let active = true;
    api.diagnosticReference(diagnosticReference.context_id, currentRunId)
      .then((latest) => active && setDiagnosticReference(latest))
      .catch(() => active && setDiagnosticReference((current) => current ? {
        ...current,
        status: "stale",
        reason: "The source could not be verified; diagnostics remain blocked.",
      } : current));
    return () => { active = false; };
  }, [activeTool, diagnosticReference?.context_id, currentRunId]);

  useEffect(() => {
    if (activeTool !== "diagnostics" || !diagnosticReference) return;
    let active = true;
    api.diagnosticHeadStatus(
      diagnosticReference.context_id,
      diagnosticResolution,
      currentRunId,
    ).then((status) => active && setDiagnosticHeadStatus(status))
      .catch(() => active && setDiagnosticHeadStatus({
        status: "stale",
        resolution: diagnosticResolution,
        reason: "The run or source could not be verified; head preparation is blocked.",
      }));
    return () => { active = false; };
  }, [
    activeTool,
    diagnosticReference?.context_id,
    diagnosticReference?.status,
    currentRunId,
    diagnosticResolution,
  ]);
  const usesKMeansBlocks = run !== null;
  const canSearchSimilar = run?.mode === "classifier" && run.unit === "image";
  const similarQueryIndex = similarChain[similarChain.length - 1] ?? null;
  const canSearchText = Boolean(
    run?.mode === "classifier" && run.unit === "image" && activeModelId.startsWith("chinese-clip"),
  );

  useEffect(() => {
    textRequest.current += 1;
    rankingRequest.current += 1;
    duplicateRequest.current += 1;
    duplicateHandoffRequest.current += 1;
    setTextResults(null);
    setTextSearchInput(null);
    setTextError("");
    setTextLoading(false);
    setDuplicateResults(null);
    setDuplicateScanUsed(null);
    setDuplicateHandoff(null);
    setDuplicateIncludeHandled(false);
    setDuplicateNewRound(true);
    setDuplicateRounds([]);
    setDuplicateRoundId("");
    setDuplicateHandoffLoading(false);
    setDuplicateError("");
    setDuplicateLoading(false);
    setRankingResults(null);
    setRankingError("");
    setRankingLoading(false);
    setRankingVisibleCount(24);
    setRankedViewerContext(null);
  }, [job?.run_id]);

  useEffect(() => {
    let active = true;
    if (!duplicateResults?.results.length) {
      setDuplicateRounds([]);
      setDuplicateRoundId("");
      return () => { active = false; };
    }
    api.activeLearningRounds().then((response) => {
      if (!active) return;
      setDuplicateRounds(response.rounds);
      setDuplicateRoundId((current) => response.rounds.some((round) => round.round_id === current)
        ? current
        : response.rounds.at(-1)?.round_id || "");
    }).catch(() => {
      if (active) setDuplicateRounds([]);
    });
    return () => { active = false; };
  }, [duplicateResults]);

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
      dimension !== 3 || !usesKMeansBlocks || !runId || !currentView || !currentRecords.length
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
    dimension, usesKMeansBlocks, job?.run_id, currentView, currentRecords.length,
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

  async function addRecordsToCuration(
    indices: number[],
    source: string,
    scores: Record<number, number> = {},
  ) {
    if (!job?.run_id || !indices.length || addingToCuration) return;
    setAddingToCuration(true);
    setError("");
    setNotice("");
    try {
      const result = await api.addCuration(job.run_id, indices, source, "", scores);
      setCurationCount(result.total);
      setNotice(`Added ${result.added} item(s) to curation; skipped ${result.skipped} duplicate(s).`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not add the selected images to curation.");
    } finally {
      setAddingToCuration(false);
    }
  }

  async function addSelectedToCuration() {
    await addRecordsToCuration(selectedIndices, "manual");
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
    duplicateHandoffRequest.current += 1;
    const input = duplicateScanInput();
    setDuplicateResults(null);
    setDuplicateScanUsed(null);
    setDuplicateHandoff(null);
    setDuplicateIncludeHandled(false);
    setDuplicateHandoffLoading(false);
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

  async function sendDuplicateCandidatesToLabeling() {
    const runId = job?.run_id;
    if (!runId || !duplicateResults?.results.length || !duplicateScanUsed) return;
    const requestId = ++duplicateHandoffRequest.current;
    setDuplicateHandoff(null);
    setDuplicateError("");
    setDuplicateHandoffLoading(true);
    try {
      const result = await api.sendDuplicatePairsToLabeling(runId, {
        ...duplicateScanUsed,
        include_handled: duplicateIncludeHandled,
        new_round: duplicateNewRound,
        round_id: duplicateNewRound ? null : duplicateRoundId || null,
      });
      if (duplicateHandoffRequest.current === requestId) {
        setDuplicateHandoff(result);
        if (result.created_round) setDuplicateRoundId(result.round_id);
      }
      void api.activeLearningRounds().then((response) => {
        if (duplicateHandoffRequest.current === requestId) setDuplicateRounds(response.rounds);
      }).catch(() => undefined);
    } catch (reason) {
      if (duplicateHandoffRequest.current === requestId) {
        setDuplicateError(reason instanceof Error ? reason.message : "Could not send duplicate pairs to Labeling.");
      }
    } finally {
      if (duplicateHandoffRequest.current === requestId) setDuplicateHandoffLoading(false);
    }
  }

  function addFolder(value = folderDraft) {
    const path = value.trim();
    if (!path) return;
    setFolders((current) => current.includes(path) ? current : [...current, path]);
    setFolderDraft("");
    setNotice("");
  }

  function effectiveFolders() {
    const selected = [...folders];
    const draft = folderDraft.trim();
    if (draft && !selected.includes(draft)) selected.push(draft);
    return selected;
  }

  function objectPolicyScope(selectedFolders: string[], modelId: string) {
    return JSON.stringify({
      folders: selectedFolders,
      modelId,
      classNames: classNamesDraft.split(",").map((name) => name.trim()).filter(Boolean),
    });
  }

  const objectPolicyTuning = objectPolicySubmitting || (objectPolicyJob !== null
    && ["queued", "running", "cancelling"].includes(objectPolicyJob.state));
  const visibleObjectPolicyResult = objectPolicyTuneResult
    && objectPolicyTuneResult.scope === objectPolicyScope(effectiveFolders(), selectedModelIds[0] || "")
    ? objectPolicyTuneResult.result
    : null;

  async function runObjectPolicyAutotune() {
    const selectedFolders = effectiveFolders();
    const modelId = selectedModelIds[0];
    if (!selectedFolders.length) {
      setError("Choose an image folder before tuning object policy.");
      return;
    }
    if (!modelId) {
      setError("Choose an installed model before tuning object policy.");
      return;
    }
    const classNames = classNamesDraft.split(",").map((name) => name.trim()).filter(Boolean);
    setError("");
    setNotice("");
    setObjectPolicyTuneResult(null);
    setObjectPolicyJob(null);
    setObjectPolicySubmitting(true);
    setObjectPolicyProfileName(selectedFolders[0].split(/[\\/]/).filter(Boolean).at(-1) || "object-policy");
    try {
      const created = await api.createObjectPolicyAutotuneJob({
        folders: selectedFolders,
        folderGrantIds: api.folderGrantIds(selectedFolders),
        classNames,
        modelId,
      });
      setObjectPolicyJob({
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
        setObjectPolicyJob(latest);
        if (["completed", "failed", "cancelled"].includes(latest.state)) break;
        await pause(350);
      }
      if (!latest || !["completed", "failed", "cancelled"].includes(latest.state)) {
        throw new Error("Object policy tuning is taking too long. Cancel it and retry with a smaller dataset.");
      }
      if (latest.state === "failed") throw new Error(latest.error?.message || "Object policy tuning failed.");
      if (latest.state === "cancelled") {
        setNotice("Object policy tuning cancelled.");
        return;
      }
      if (!latest.result) throw new Error("The worker completed without returning tuning results.");
      setObjectPolicyTuneResult({
        jobId: created.job_id,
        scope: objectPolicyScope(selectedFolders, modelId),
        result: latest.result,
      });
      setNotice("Object policy tuning completed. Review the candidate scores before applying a policy.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Object policy tuning failed.");
    } finally {
      setObjectPolicySubmitting(false);
    }
  }

  async function cancelObjectPolicyAutotune() {
    if (!objectPolicyJob?.job_id) return;
    try {
      await api.cancel(objectPolicyJob.job_id);
      setObjectPolicyJob((current) => current
        ? { ...current, state: "cancelling", message: "Cancellation requested" }
        : current);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not cancel object policy tuning.");
    }
  }

  async function saveObjectPolicyProfile() {
    if (!objectPolicyTuneResult) return;
    try {
      const saved = await api.saveObjectPolicyProfile(
        objectPolicyTuneResult.jobId,
        objectPolicyProfileName,
      );
      setNotice("Saved object policy profile “" + saved.name + "”.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not save the object policy profile.");
    }
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
      if (result.grant_id && chosen.length !== 1) {
        throw new Error("Choose one folder at a time so NativeApp can scope its permission.");
      }
      if (result.grant_id) api.rememberFolderGrant(chosen[0], result.grant_id);
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
        folderGrantIds: api.folderGrantIds(selectedFolders),
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

  async function openDiagnosticsReference(recordIndex: number) {
    if (!job?.run_id || !run || !activeModelId) return;
    setError("");
    try {
      const reference = await api.createDiagnosticReference(job.run_id, recordIndex, activeModelId);
      setDiagnosticReference(reference);
      setDiagnosticResolution(224);
      setDiagnosticHeadStatus(null);
      setDiagnosticHeadError("");
      setDiagnosticExplanation(null);
      setDiagnosticExplanationError("");
      setActiveTool("diagnostics");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not create the diagnostics reference.");
    }
  }

  async function recheckDiagnosticsReference() {
    if (!diagnosticReference) return;
    setDiagnosticHeadStatus(null);
    try {
      const latest = await api.diagnosticReference(diagnosticReference.context_id, currentRunId);
      setDiagnosticReference(latest);
    } catch {
      setDiagnosticReference((current) => current ? {
        ...current,
        status: "stale",
        reason: "The source could not be verified; diagnostics remain blocked.",
      } : current);
    }
  }

  async function buildDiagnosticHead() {
    if (!diagnosticReference || !currentRunId || currentRunId !== diagnosticReference.run_id) return;
    setDiagnosticHeadLoading(true);
    setDiagnosticHeadError("");
    setDiagnosticHeadStatus(null);
    try {
      const status = await api.buildDiagnosticHead(
        diagnosticReference.context_id,
        diagnosticResolution,
        currentRunId,
      );
      setDiagnosticHeadStatus(status);
      setDiagnosticExplanation(null);
      setDiagnosticExplanationError("");
    } catch (reason) {
      setDiagnosticHeadError(reason instanceof Error ? reason.message : "The classifier head could not be prepared.");
      setDiagnosticHeadStatus({
        status: "not_ready",
        resolution: diagnosticResolution,
        reason: "No matching head was saved; you can retry this action.",
      });
    } finally {
      setDiagnosticHeadLoading(false);
    }
  }

  async function runDiagnosticExplanation() {
    if (!diagnosticReference || !currentRunId || currentRunId !== diagnosticReference.run_id) return;
    if (diagnosticHeadStatus?.status !== "ready") return;
    setDiagnosticExplanationLoading(true);
    setDiagnosticExplanationError("");
    setDiagnosticExplanation(null);
    try {
      const input: DiagnosticExplanationInput = {
        current_run_id: currentRunId,
        resolution: diagnosticResolution,
        method: diagnosticMethod,
        attention_method: diagnosticAttentionMethod,
        head_index: diagnosticAttentionMethod === "single_head" ? diagnosticHeadIndex : null,
        target_class: diagnosticMethod === "gradcam" && diagnosticTargetClass ? diagnosticTargetClass : null,
        display_mode: diagnosticDisplayMode,
      };
      setDiagnosticExplanation(await api.createDiagnosticExplanation(diagnosticReference.context_id, input));
    } catch (reason) {
      setDiagnosticExplanationError(reason instanceof Error ? reason.message : "The explanation could not be generated.");
    } finally {
      setDiagnosticExplanationLoading(false);
    }
  }

  function downloadDiagnosticJson() {
    if (!diagnosticExplanation) return;
    const metadata = Object.fromEntries(
      Object.entries(diagnosticExplanation).filter(([key]) =>
        key !== "model_input_png_base64" && key !== "overlay_png_base64",
      ),
    );
    const blob = new Blob([JSON.stringify(metadata, null, 2)], { type: "application/json" });
    const objectUrl = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = objectUrl;
    anchor.download = "diagnostic-evidence.json";
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
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
          {run && currentRunId && <button type="button" data-testid="open-class-similarity" className={activeTool === "class_similarity" ? "active" : ""} aria-current={activeTool === "class_similarity" ? "page" : undefined} onClick={() => setActiveTool("class_similarity")}>類別相似度</button>}
          {diagnosticReference && <button type="button" data-testid="open-diagnostics" className={activeTool === "diagnostics" ? "active" : ""} aria-current={activeTool === "diagnostics" ? "page" : undefined} onClick={() => setActiveTool("diagnostics")}>Diagnostics</button>}
          <button type="button" className={activeTool === "compare" ? "active" : ""} aria-current={activeTool === "compare" ? "page" : undefined} onClick={() => setActiveTool("compare")}>Compare</button>
          <button type="button" className={activeTool === "coverage" ? "active" : ""} aria-current={activeTool === "coverage" ? "page" : undefined} onClick={() => { setCoverageMounted(true); setActiveTool("coverage"); }}>Coverage</button>
          <button type="button" data-testid="open-fewshot-samples" className={activeTool === "fewshot" ? "active" : ""} aria-current={activeTool === "fewshot" ? "page" : undefined} onClick={() => setActiveTool("fewshot")}>以樣搜樣</button>
          <button type="button" data-testid="open-anomaly-bank" className={activeTool === "anomaly" ? "active" : ""} aria-current={activeTool === "anomaly" ? "page" : undefined} onClick={() => setActiveTool("anomaly")}>瑕疵偵測</button>
          <button type="button" data-testid="open-groupdiff" className={activeTool === "groupdiff" ? "active" : ""} aria-current={activeTool === "groupdiff" ? "page" : undefined} onClick={() => setActiveTool("groupdiff")}>差異探索</button>
          <button type="button" data-testid="open-evaluation" className={activeTool === "evaluation" ? "active" : ""} aria-current={activeTool === "evaluation" ? "page" : undefined} onClick={() => setActiveTool("evaluation")}>評估</button>
          <button type="button" data-testid="open-curation" className={activeTool === "curation" ? "active" : ""} aria-current={activeTool === "curation" ? "page" : undefined} onClick={() => setActiveTool("curation")}>Curation{curationCount === null ? "" : ` (${curationCount})`}</button>
          <button type="button" data-testid="open-fp-cases" className={activeTool === "false_positive" ? "active" : ""} aria-current={activeTool === "false_positive" ? "page" : undefined} onClick={() => setActiveTool("false_positive")}>誤逮案件</button>
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

      {activeTool === "diagnostics" ? (
        <main className="diagnostics-workspace" aria-labelledby="diagnostics-title">
          <section className="diagnostics-card">
            <div className="eyebrow">MODEL DIAGNOSTICS</div>
            <h1 id="diagnostics-title">Diagnostics reference</h1>
            {diagnosticReference ? <>
              <div
                className={`diagnostic-reference-status diagnostic-status-${diagnosticReference.status}`}
                data-testid="diagnostic-reference-status"
                role={diagnosticReference.status === "current" ? "status" : "alert"}
                aria-live="polite"
              >
                Context {diagnosticReference.status}: {diagnosticReference.reason}
              </div>
              <dl className="diagnostic-reference-details">
                <div><dt>Image</dt><dd data-testid="diagnostic-reference-image">{diagnosticReference.filename}</dd></div>
                <div><dt>Label / split</dt><dd>{diagnosticReference.label || "—"} / {diagnosticReference.split || "—"}</dd></div>
                <div><dt>Model</dt><dd>{diagnosticReference.model_id}</dd></div>
                <div><dt>Visualize run</dt><dd data-testid="diagnostic-reference-run">{diagnosticReference.run_id}</dd></div>
                <div><dt>Source SHA-256</dt><dd data-testid="diagnostic-reference-hash">{diagnosticReference.source_sha256 || "Unavailable"}</dd></div>
                {diagnosticReference.object_index !== null && <div><dt>Object</dt><dd>{diagnosticReference.object_index + 1}</dd></div>}
              </dl>
              <p className="diagnostics-no-auto-inference" data-testid="diagnostics-no-auto-inference">Opening this page only checks the selected source reference. It does not start model inference.</p>
              <section className="diagnostic-head-panel" aria-labelledby="diagnostic-head-title">
                <h2 id="diagnostic-head-title">Classifier head</h2>
                <label htmlFor="diagnostic-resolution">Analysis resolution</label>
                <select
                  id="diagnostic-resolution"
                  data-testid="diagnostic-resolution"
                  value={diagnosticResolution}
                  onChange={(event) => {
                    setDiagnosticResolution(Number(event.target.value) as 224 | 336 | 448);
                    setDiagnosticHeadStatus(null);
                    setDiagnosticHeadError("");
                  }}
                >
                  <option value={224}>224×224 · 16×16 patches</option>
                  <option value={336}>336×336 · 24×24 patches</option>
                  <option value={448}>448×448 · 32×32 patches</option>
                </select>
                {diagnosticHeadStatus && <div
                  className={`diagnostic-head-status diagnostic-status-${diagnosticHeadStatus.status === "ready" ? "current" : diagnosticHeadStatus.status === "not_ready" ? "stale" : diagnosticHeadStatus.status}`}
                  data-testid="diagnostic-head-status"
                  role={diagnosticHeadStatus.status === "ready" || diagnosticHeadStatus.status === "not_ready" ? "status" : "alert"}
                  aria-live="polite"
                >
                  {diagnosticHeadStatus.status === "ready"
                    ? `Head ready: ${diagnosticHeadStatus.resolution}×${diagnosticHeadStatus.resolution} · ${diagnosticHeadStatus.training_samples} records · ${diagnosticHeadStatus.classes?.join(", ") || "no classes"}${diagnosticHeadStatus.reused ? " · reused" : ""}`
                    : diagnosticHeadStatus.status === "not_ready"
                      ? `Head not prepared: ${diagnosticHeadStatus.reason}`
                      : `Head ${diagnosticHeadStatus.status}: ${diagnosticHeadStatus.reason}`}
                </div>}
                {diagnosticHeadError && <div className="alert alert-error" role="alert">{diagnosticHeadError}</div>}
                <button
                  type="button"
                  className="secondary-action diagnostic-head-build"
                  data-testid="diagnostic-head-build"
                  disabled={diagnosticHeadLoading || diagnosticHeadStatus?.status !== "not_ready" || diagnosticReference.status !== "current" || currentRunId !== diagnosticReference.run_id}
                  onClick={buildDiagnosticHead}
                >
                  {diagnosticHeadLoading
                    ? `Preparing ${diagnosticResolution}×${diagnosticResolution} head…`
                    : diagnosticHeadStatus?.status === "ready"
                      ? `Head ready for ${diagnosticResolution}×${diagnosticResolution}`
                      : `Build ${diagnosticResolution}×${diagnosticResolution} classifier head`}
                </button>
                <p className="field-note">A head is created only after this button is pressed. Heads are tied to this Visualize run, model, unit, pooling, and exact resolution; no other resolution is substituted.</p>
              </section>
              <section className="diagnostic-explanation-panel" aria-labelledby="diagnostic-explanation-title">
                <h2 id="diagnostic-explanation-title">Run classification evidence</h2>
                <p className="field-note">Inference runs only when you press the button. Patch evidence is coarse and does not represent a defect mask or pixel-accurate localization.</p>
                <div className="diagnostic-explanation-controls">
                  <label>
                    Evidence method
                    <select
                      data-testid="diagnostic-method"
                      value={diagnosticMethod}
                      onChange={(event) => {
                        setDiagnosticMethod(event.target.value as "attention" | "gradcam");
                        setDiagnosticExplanation(null);
                        setDiagnosticExplanationError("");
                      }}
                    >
                      <option value="attention">DINO Attention</option>
                      <option value="gradcam">Classifier Grad-CAM</option>
                    </select>
                  </label>
                  {diagnosticMethod === "attention" ? <>
                    <label>
                      Attention recipe
                      <select
                        data-testid="diagnostic-attention-method"
                        value={diagnosticAttentionMethod}
                        onChange={(event) => {
                          setDiagnosticAttentionMethod(event.target.value as DiagnosticExplanationInput["attention_method"]);
                          setDiagnosticExplanation(null);
                        }}
                      >
                        <option value="last_layer_head_mean">Last layer, mean of heads</option>
                        <option value="single_head">Single attention head</option>
                        <option value="rollout">Attention rollout (experimental)</option>
                      </select>
                    </label>
                    {diagnosticAttentionMethod === "single_head" && <label>
                      Attention head index
                      <input
                        data-testid="diagnostic-attention-head-index"
                        type="number"
                        min={0}
                        max={63}
                        value={diagnosticHeadIndex}
                        onChange={(event) => setDiagnosticHeadIndex(Math.max(0, Number(event.target.value) || 0))}
                      />
                    </label>}
                  </> : <>
                    <label>
                      Target class
                      <select
                        data-testid="diagnostic-target-class"
                        value={diagnosticTargetClass}
                        onChange={(event) => {
                          setDiagnosticTargetClass(event.target.value);
                          setDiagnosticExplanation(null);
                        }}
                      >
                        <option value="">Actual top-1 (automatic)</option>
                        {(diagnosticHeadStatus?.classes || []).map((name) => <option key={name} value={name}>{name}</option>)}
                      </select>
                    </label>
                    <label>
                      Evidence view
                      <select
                        data-testid="diagnostic-display-mode"
                        value={diagnosticDisplayMode}
                        onChange={(event) => {
                          setDiagnosticDisplayMode(event.target.value as "signed" | "positive");
                          setDiagnosticExplanation(null);
                        }}
                      >
                        <option value="signed">Signed: supports target / reference</option>
                        <option value="positive">Positive target evidence only</option>
                      </select>
                    </label>
                  </>}
                </div>
                {diagnosticExplanationError && <div className="alert alert-error" data-testid="diagnostic-explanation-error" role="alert">{diagnosticExplanationError}</div>}
                <button
                  type="button"
                  className="secondary-action diagnostic-explain-run"
                  data-testid="diagnostic-explain-run"
                  disabled={diagnosticExplanationLoading || diagnosticHeadStatus?.status !== "ready" || diagnosticReference.status !== "current" || currentRunId !== diagnosticReference.run_id}
                  onClick={runDiagnosticExplanation}
                >
                  {diagnosticExplanationLoading ? "Running explanation…" : "Run explanation"}
                </button>
                {diagnosticExplanation && <div className="diagnostic-explanation-result" data-testid="diagnostic-explanation-result">
                  <div className="diagnostic-explanation-summary">
                    <div>
                      <span className="eyebrow">CLASSIFICATION</span>
                      <strong data-testid="diagnostic-predicted-class">{diagnosticExplanation.predicted_class}</strong>
                      <small>{(diagnosticExplanation.predicted_probability * 100).toFixed(1)}% top-1 confidence · {diagnosticExplanation.resolution}×{diagnosticExplanation.resolution} · {diagnosticExplanation.scope}</small>
                    </div>
                    {diagnosticExplanation.method === "gradcam" && <div>
                      <span className="eyebrow">TARGET / REFERENCE</span>
                      <strong data-testid="diagnostic-target-result">{diagnosticExplanation.target_class} / {diagnosticExplanation.reference_class}</strong>
                      <small>{diagnosticExplanation.is_predicted_target ? "Target matches actual top-1" : "Counterfactual target hypothesis"} · margin {diagnosticExplanation.target_score?.toFixed(4)}</small>
                    </div>}
                  </div>
                  <div className="diagnostic-export-actions">
                    <a
                      className="secondary-action"
                      data-testid="diagnostic-download-png"
                      href={`data:image/png;base64,${diagnosticExplanation.overlay_png_base64}`}
                      download="diagnostic-overlay.png"
                    >Download overlay PNG</a>
                    <button
                      type="button"
                      className="secondary-action"
                      data-testid="diagnostic-download-json"
                      onClick={downloadDiagnosticJson}
                    >Download evidence JSON</button>
                  </div>
                  <div className="diagnostic-evidence-layout">
                    <figure className="diagnostic-source-preview">
                      <img
                        src={`/api/v1/runs/${encodeURIComponent(diagnosticExplanation.run_id)}/images/${diagnosticExplanation.record_index}`}
                        alt={diagnosticReference.filename}
                      />
                      <figcaption>Selected source image</figcaption>
                    </figure>
                    <figure className="diagnostic-source-preview">
                      <img
                        data-testid="diagnostic-model-input-image"
                        src={`data:image/png;base64,${diagnosticExplanation.model_input_png_base64}`}
                        alt={`Preprocessed ${diagnosticExplanation.scope} model input`}
                      />
                      <figcaption>Exact preprocessed model input</figcaption>
                    </figure>
                    <figure className="diagnostic-source-preview">
                      <img
                        data-testid="diagnostic-overlay-image"
                        src={`data:image/png;base64,${diagnosticExplanation.overlay_png_base64}`}
                        alt={`${diagnosticExplanation.method} evidence overlay`}
                      />
                      <figcaption>Patch evidence overlay · not a segmentation mask</figcaption>
                    </figure>
                  </div>
                  <div className="diagnostic-grid-panel">
                    <div className="diagnostic-grid-heading">
                      <strong>{diagnosticExplanation.method === "gradcam" ? "Grad-CAM patch evidence" : "DINO Attention patch map"}</strong>
                      <span>{diagnosticExplanation.grid[0]}×{diagnosticExplanation.grid[1]} patches · {diagnosticExplanation.score_mode}</span>
                    </div>
                    <DiagnosticEvidenceGrid result={diagnosticExplanation} />
                    <p className="field-note">{diagnosticExplanation.warning}</p>
                  </div>
                  <ul className="diagnostic-class-probabilities" aria-label="Class probabilities">
                    {diagnosticExplanation.class_probabilities.map((item) => <li key={item.class_name}>
                      <span>{item.class_name}</span><strong>{(item.probability * 100).toFixed(1)}%</strong>
                    </li>)}
                  </ul>
                  {diagnosticExplanation.faithfulness && <p className="diagnostic-faithfulness" data-testid="diagnostic-faithfulness">
                    Occlusion faithfulness: {String(diagnosticExplanation.faithfulness.status || "not available")}.
                  </p>}
                  <details className="diagnostic-warnings">
                    <summary>Interpretation notes</summary>
                    <ul>{diagnosticExplanation.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
                  </details>
                </div>}
              </section>
              <div className="diagnostics-actions">
                <button type="button" className="secondary-action" data-testid="diagnostic-recheck" onClick={recheckDiagnosticsReference}>Check source again</button>
                <button type="button" className="secondary-action" onClick={() => setActiveTool("visualize")}>Return to Visualize</button>
              </div>
            </> : <p>Select a Visualize record to create a diagnostics reference.</p>}
          </section>
        </main>
      ) : activeTool === "false_positive" ? (
        <FalsePositiveCaseWorkspace />
      ) : activeTool === "class_similarity" && run && currentRunId ? (
        <ClassSimilarityWorkspace
          key={`${currentRunId}-${activeModelId}`}
          runId={currentRunId}
          modelId={activeModelId}
          unit={run.unit}
          records={run.records}
          onReturn={() => setActiveTool("visualize")}
          onOpenRecord={(index) => {
            setActiveTool("visualize");
            setSelectedIndex(index);
            setSelectedIndices((current) => current.includes(index) ? current : [...current, index]);
          }}
          onAddSelection={addSelection}
        />
      ) : activeTool === "evaluation" ? (
        <EvaluationWorkspace />
      ) : activeTool === "fewshot" ? (
        <Suspense fallback={<div className="coverage-loading" role="status">Loading few-shot workspace…</div>}>
          <LazyFewshotSampleBankWorkspace models={models} connected={connected} />
        </Suspense>
      ) : activeTool === "anomaly" ? (
        <Suspense fallback={<div className="coverage-loading" role="status">Loading anomaly workspace…</div>}>
          <LazyAnomalyBankWorkspace models={models} connected={connected} onCurationCountChange={setCurationCount} />
        </Suspense>
      ) : activeTool === "curation" ? (
        <CurationWorkspace onCountChange={setCurationCount} currentRunId={currentRunId} onRunUpdated={setRun} />
      ) : activeTool === "compare" ? (
        <CompareDistributions models={models} loadingModels={loadingModels} connected={connected} onCurationCountChange={setCurationCount} />
      ) : activeTool === "groupdiff" ? (
        <Suspense fallback={<div className="coverage-loading" role="status">載入 Good／Bad 差異工作區…</div>}>
          <LazyGroupDiffWorkspace models={models} loadingModels={loadingModels} connected={connected} />
        </Suspense>
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
              disabled={busy || objectPolicyTuning}
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
                  disabled={busy || objectPolicyTuning}
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
                      disabled={busy || objectPolicyTuning}
                    />
                    <label className="field-label" htmlFor="object-resolution">Crop resolution</label>
                    <select
                      id="object-resolution"
                      aria-label="Object crop resolution"
                      value={objectPolicy.target_res}
                      onChange={(event) => setObjectPolicy((current) => ({ ...current, target_res: Number(event.target.value) as ComparisonObjectPolicy["target_res"] }))}
                      disabled={busy || objectPolicyTuning}
                    >
                      {[168, 196, 224, 280, 336, 392, 448].map((resolution) => <option key={resolution} value={resolution}>{resolution}px</option>)}
                    </select>
                    <label className="field-label" htmlFor="object-head">Embedding head</label>
                    <select
                      id="object-head"
                      aria-label="Object embedding head"
                      value={objectPolicy.head}
                      onChange={(event) => setObjectPolicy((current) => ({ ...current, head: event.target.value as ComparisonObjectPolicy["head"] }))}
                      disabled={busy || objectPolicyTuning}
                    >
                      <option value="cls">CLS token</option>
                      <option value="meanpool">Mean pool</option>
                    </select>
                    <div className="field-note">Each YOLO box uses this crop policy. Autotune scores six candidates by macro kNN class purity using the first selected feature model.</div>
                    <button
                      className="secondary-action"
                      type="button"
                      data-testid="object-policy-autotune"
                      disabled={busy || objectPolicyTuning || !connected || !effectiveFolders().length || !selectedModelIds.length}
                      onClick={runObjectPolicyAutotune}
                    >
                      {objectPolicyTuning ? "Tuning object policy…" : "Find best object policy"}
                    </button>
                    {objectPolicyTuning && (
                      <div className="progress-card" data-testid="object-policy-autotune-progress" role="status">
                        <div className="progress-row">
                          <span>{objectPolicyJob?.message || "Preparing object policy search"}</span>
                          <strong>{Math.round(objectPolicyJob?.percent || 0)}%</strong>
                        </div>
                        <div className="progress-track">
                          <div className="progress-fill" style={{ width: Math.max(0, Math.min(100, objectPolicyJob?.percent || 0)) + "%" }} />
                        </div>
                        <button
                          className="cancel-button"
                          type="button"
                          disabled={!objectPolicyJob?.job_id}
                          onClick={cancelObjectPolicyAutotune}
                        >
                          Cancel tuning
                        </button>
                      </div>
                    )}
                    {visibleObjectPolicyResult && objectPolicyTuneResult && (
                      <div className="object-policy-results" data-testid="object-policy-autotune-results">
                        <div className="field-note">
                          {visibleObjectPolicyResult.fingerprint.n_objects.toLocaleString()} labeled objects · {visibleObjectPolicyResult.fingerprint.n_classes} classes · median short side {Math.round(visibleObjectPolicyResult.fingerprint.size_q[1] || 0)} px
                        </div>
                        {visibleObjectPolicyResult.similar_profile && (
                          <div className="alert alert-info" data-testid="object-policy-profile-suggestion" role="status">
                            Similar saved profile: {visibleObjectPolicyResult.similar_profile.name} ({visibleObjectPolicyResult.similar_profile.similarity.toFixed(2)}).
                            <button
                              className="secondary-action"
                              type="button"
                              data-testid="object-policy-apply-profile"
                              onClick={() => {
                                setObjectPolicy(visibleObjectPolicyResult.similar_profile!.policy);
                                setNotice("Applied the similar saved profile. Run analysis to use it.");
                              }}
                            >
                              Apply saved profile
                            </button>
                          </div>
                        )}
                        <div className="object-policy-table-wrap">
                          <table className="object-policy-table">
                            <thead><tr><th>Policy</th><th>Macro</th><th>&lt;32</th><th>32–96</th><th>96–224</th><th>&gt;224</th></tr></thead>
                            <tbody>
                              {visibleObjectPolicyResult.rows.map((row) => {
                                const pct = (value: number | null) => value === null ? "—" : (value * 100).toFixed(1) + "%";
                                return <tr key={objectPolicyLabel(row.policy)} data-testid="object-policy-candidate">
                                  <td>{objectPolicyLabel(row.policy)}</td>
                                  <td>{pct(row.macro_purity)}</td>
                                  <td>{pct(row.buckets["<32"]?.purity ?? null)}</td>
                                  <td>{pct(row.buckets["32-96"]?.purity ?? null)}</td>
                                  <td>{pct(row.buckets["96-224"]?.purity ?? null)}</td>
                                  <td>{pct(row.buckets[">224"]?.purity ?? null)}</td>
                                </tr>;
                              })}
                            </tbody>
                          </table>
                        </div>
                        <div className="field-note">Higher macro kNN purity is better. This weak-label tuning signal is not a model-quality guarantee.</div>
                        <button
                          className="secondary-action"
                          type="button"
                          data-testid="object-policy-apply-best"
                          onClick={() => {
                            setObjectPolicy(visibleObjectPolicyResult.best_policy);
                            setNotice("Applied recommended policy " + objectPolicyLabel(visibleObjectPolicyResult.best_policy) + ". Run analysis to use it.");
                          }}
                        >
                          Apply recommended {objectPolicyLabel(visibleObjectPolicyResult.best_policy)}
                        </button>
                        <label className="field-label" htmlFor="object-policy-profile-name">Profile name</label>
                        <input
                          id="object-policy-profile-name"
                          aria-label="Object policy profile name"
                          value={objectPolicyProfileName}
                          maxLength={128}
                          onChange={(event) => setObjectPolicyProfileName(event.target.value)}
                        />
                        <button
                          className="secondary-action"
                          type="button"
                          data-testid="object-policy-save-profile"
                          disabled={!objectPolicyProfileName.trim()}
                          onClick={saveObjectPolicyProfile}
                        >
                          Save recommended policy as profile
                        </button>
                      </div>
                    )}
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
                  disabled={busy || objectPolicyTuning}
                />
                <div className="field-note">A dataset classes.txt takes priority. Enter comma-separated names only when the dataset has no class file.</div>
              </>
            )}
          </section>

          <section className="control-section">
            <div className="section-heading"><span className="step-number">01</span><h2>Image folders</h2></div>
            <button className="host-picker" type="button" onClick={chooseFolder} disabled={busy || objectPolicyTuning}>
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
                disabled={busy || objectPolicyTuning}
              />
              <button type="button" className="add-folder" onClick={() => addFolder()} aria-label="Add folder" disabled={busy || objectPolicyTuning}>+</button>
            </div>
            {folders.length > 0 && (
              <ul className="folder-list" aria-label="Selected folders">
                {folders.map((folder) => (
                  <li key={folder} title={folder}>
                    <span className="folder-chip-dot" />
                    <span className="folder-chip-name">{folder.split(/[\\/]/).filter(Boolean).at(-1) || folder}</span>
                    <button type="button" aria-label={"Remove " + folder} onClick={() => setFolders((current) => current.filter((item) => item !== folder))} disabled={busy || objectPolicyTuning}>×</button>
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
                    disabled={loadingModels || busy || objectPolicyTuning}
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

          <button className="run-button" type="button" onClick={runAnalysis} disabled={busy || objectPolicyTuning || !connected || loadingModels || !selectedModelIds.length || !selectedMethods.length}>
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
                    clusterCenters={dimension === 3 && usesKMeansBlocks ? blockData?.centers : undefined}
                  />
                ) : <div className="empty-filter">No images match these filters.</div>}
                {dimension === 3 && usesKMeansBlocks && currentRecords.length > 0 && (
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
                    <button
                      type="button"
                      className="similar-start-button"
                      data-testid="open-diagnostics-reference"
                      onClick={() => openDiagnosticsReference(selectedRecord.index)}
                    >
                      Open diagnostics reference
                    </button>
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
                                    <button
                                      type="button"
                                      className="result-curation-button"
                                      data-testid={`curate-similar-${index}`}
                                      disabled={addingToCuration || !job?.run_id}
                                      onClick={() => void addRecordsToCuration([index], "similar", { [index]: distance })}
                                    >
                                      Add to Curation
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
                                  <button
                                    type="button"
                                    className="result-curation-button"
                                    data-testid={`curate-text-${index}`}
                                    disabled={addingToCuration || !job?.run_id}
                                    onClick={() => void addRecordsToCuration([index], "search", { [index]: distance })}
                                  >
                                    Add to Curation
                                  </button>
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
                                  <button
                                    type="button"
                                    className="result-curation-button"
                                    data-testid={`curate-duplicate-${left_index}-${right_index}`}
                                    disabled={addingToCuration || !job?.run_id}
                                    onClick={() => void addRecordsToCuration(
                                      [left_index, right_index],
                                      "duplicate",
                                      { [left_index]: distance, [right_index]: distance },
                                    )}
                                  >
                                    Add pair to Curation
                                  </button>
                                </article>
                              );
                            })}
                          </div>
                        ) : <p className="search-tool-note">No duplicate candidates under the current settings.</p>}
                        {duplicateResults.results.length > 0 && <button type="button" className="search-export-button" data-testid="duplicate-export" onClick={() => void exportDuplicateCandidates()}>Export candidate pairs CSV</button>}
                        {duplicateResults.results.length > 0 && (
                          <div className="duplicate-handoff-controls" data-testid="duplicate-handoff-controls">
                            <label>
                              <input
                                type="checkbox"
                                aria-label="Include previously reviewed images"
                                checked={duplicateIncludeHandled}
                                onChange={(event) => setDuplicateIncludeHandled(event.target.checked)}
                              />
                              <span>Include previously reviewed images</span>
                            </label>
                            <label>
                              <input
                                type="checkbox"
                                aria-label="Record as a new duplicate review round"
                                checked={duplicateNewRound}
                                onChange={(event) => setDuplicateNewRound(event.target.checked)}
                              />
                              <span>Record as a new round</span>
                            </label>
                            {!duplicateNewRound && duplicateRounds.length > 0 && (
                              <label>
                                <span>Round to continue</span>
                                <select aria-label="Duplicate review round to continue" value={duplicateRoundId} onChange={(event) => setDuplicateRoundId(event.target.value)}>
                                  {[...duplicateRounds].reverse().map((round) => (
                                    <option key={round.round_id} value={round.round_id}>{round.round_id} · {round.strategy || "—"}</option>
                                  ))}
                                </select>
                              </label>
                            )}
                            <button
                              type="button"
                              data-testid="duplicate-labeling-handoff"
                              onClick={() => void sendDuplicateCandidatesToLabeling()}
                              disabled={duplicateHandoffLoading || busy || (!duplicateNewRound && duplicateRounds.length > 0 && !duplicateRoundId)}
                            >
                              {duplicateHandoffLoading ? "Preparing review…" : "Send pairs to Labeling"}
                            </button>
                          </div>
                        )}
                        {duplicateHandoff && (
                          <p className="search-status" data-testid="duplicate-handoff-status" role="status">
                            Saved {duplicateHandoff.pair_count} pair(s) with {duplicateHandoff.sent} unique image(s) to the Labeling inbox · round {duplicateHandoff.round_id}
                            {duplicateHandoff.skipped_pairs ? ` · skipped ${duplicateHandoff.skipped_pairs} pair(s) already handled` : ""}
                            {duplicateHandoff.skipped_handled ? ` · ${duplicateHandoff.skipped_handled} previously handled image(s) excluded` : ""}
                          </p>
                        )}
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
                              <button
                                type="button"
                                className="result-curation-button"
                                data-testid={`curate-ranked-${index}`}
                                disabled={addingToCuration || !job?.run_id}
                                onClick={() => void addRecordsToCuration([index], rankingMetric, { [index]: score })}
                              >
                                Add to Curation
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
