import {
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  api,
  type EmbeddingPoint,
  type KMeansBlocks,
  type ComparisonObjectPolicy,
  type JobStatus,
  type ModelOption,
  type ProjectionMethod,
  type SimilarSearchResult,
  type VisualizeUnit,
  type VisualizationRun,
} from "./api";
import CompareDistributions from "./CompareDistributions";
const LazyScatterPlot2D = lazy(() => import("./ScatterPlot2D"));
const LazyScatterPlot3D = lazy(() => import("./ScatterPlot3D"));

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
  const [activeTool, setActiveTool] = useState<"visualize" | "compare">("visualize");
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
  const [showAnnotationOverlay, setShowAnnotationOverlay] = useState(true);
  const [similarChain, setSimilarChain] = useState<number[]>([]);
  const [similarK, setSimilarK] = useState(9);
  const [similarResults, setSimilarResults] = useState<SimilarSearchResult | null>(null);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [similarError, setSimilarError] = useState("");
  const similarRequest = useRef(0);
  const [blockCount, setBlockCount] = useState(12);
  const [selectedBlock, setSelectedBlock] = useState<number | null>(null);
  const [blockData, setBlockData] = useState<KMeansBlocks | null>(null);
  const [blockLoading, setBlockLoading] = useState(false);
  const blockRequest = useRef(0);
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
  const selectedViewPosition = selectedIndex === null ? -1 : selectedIndices.indexOf(selectedIndex);
  const usesClassifierBlocks = run?.mode === "classifier" && run.unit === "image";
  const canSearchSimilar = run?.mode === "classifier" && run.unit === "image";
  const similarQueryIndex = similarChain[similarChain.length - 1] ?? null;

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
    setSelectedIndices((current) => [...new Set([...current, ...additions])]);
    setSelectedIndex(additions[additions.length - 1]);
  }

  function clearSelection() {
    setSelectedIndices([]);
    setSelectedIndex(null);
  }

  function moveSelectedImage(direction: -1 | 1) {
    if (!selectedIndices.length) return;
    const current = selectedViewPosition >= 0 ? selectedViewPosition : 0;
    const next = Math.max(0, Math.min(selectedIndices.length - 1, current + direction));
    setSelectedIndex(selectedIndices[next]);
  }

  function startSimilarSearch(index: number) {
    setSimilarK(Math.min(9, Math.max(1, (run?.records.length || 1) - 1)));
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
        </nav>
        <div className={"connection " + (connected ? "connection-ready" : "connection-waiting")}>
          <span className="connection-dot" />
          {connected ? "Local worker connected" : "Connecting to local worker"}
        </div>
      </header>

      {activeTool === "compare" ? (
        <CompareDistributions models={models} loadingModels={loadingModels} connected={connected} />
      ) : <div className="workspace">
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
                            onClick={() => setSelectedIndex(index)}
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
                        <button type="button" aria-label="Previous selected image" data-testid="selected-image-previous" disabled={selectedViewPosition <= 0} onClick={() => moveSelectedImage(-1)}>◀</button>
                        <span>{selectedViewPosition + 1} / {selectedIndices.length} selected</span>
                        <button type="button" aria-label="Next selected image" data-testid="selected-image-next" disabled={selectedViewPosition >= selectedIndices.length - 1} onClick={() => moveSelectedImage(1)}>▶</button>
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
