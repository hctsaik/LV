import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import type {
  CoverageObjectView,
  CoverageObjectViewInput,
  CoverageRun,
  JobStatus,
  ModelOption,
  ProjectionMethod,
  VisualizeUnit,
} from "./api";
import { api } from "./api";
import ScatterPlot2D from "./ScatterPlot2D";
import ScatterPlot3D from "./ScatterPlot3D";

const coverageLabelColors = [
  "#e74c3c", "#f39c12", "#2ecc71", "#9b59b6", "#3498db", "#1abc9c",
  "#95a5a6", "#e84393", "#fdcb6e", "#34495e", "#0984e3", "#6c5ce7",
  "#badc58", "#576574",
];

const projectionOptions: Array<{ value: ProjectionMethod; label: string }> = [
  { value: "pca", label: "PCA" },
  { value: "tsne", label: "t-SNE" },
  { value: "umap", label: "UMAP" },
  { value: "lda", label: "LDA (supervised)" },
  { value: "sumap", label: "Supervised UMAP" },
];

function pause(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function labelColors(points: CoverageRun["points"]) {
  const labels = [...new Set(points.map((point) => point.label))].sort();
  return new Map(labels.map((label, index) => [
    label,
    coverageLabelColors[index % coverageLabelColors.length],
  ]));
}

export default function CoverageWorkspace({
  models,
  connected,
  loadingModels,
}: {
  models: ModelOption[];
  connected: boolean;
  loadingModels: boolean;
}) {
  const [folders, setFolders] = useState<string[]>([]);
  const [folderDraft, setFolderDraft] = useState("");
  const [modelId, setModelId] = useState("");
  const [unit, setUnit] = useState<VisualizeUnit>("image");
  const [method, setMethod] = useState<ProjectionMethod>("sumap");
  const [dimension, setDimension] = useState<2 | 3>(2);
  const [k, setK] = useState(10);
  const [blockK, setBlockK] = useState(8);
  const [classFilter, setClassFilter] = useState("");
  const [minShortPx, setMinShortPx] = useState(0);
  const [confidenceMin, setConfidenceMin] = useState(0);
  const [confidenceMax, setConfidenceMax] = useState(1);
  const [colorMode, setColorMode] = useState<"class" | "sparsity">("class");
  const [run, setRun] = useState<CoverageRun | null>(null);
  const [objectView, setObjectView] = useState<CoverageObjectView | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);
  const [busy, setBusy] = useState(false);
  const [projectionBusy, setProjectionBusy] = useState(false);
  const [sparsityBusy, setSparsityBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const projectionRequest = useRef(0);
  const sparsityRequest = useRef(0);
  const objectViewRequest = useRef(0);

  useEffect(() => {
    if (models.some((model) => model.model_id === modelId && model.available)) return;
    const preferred = models.find((model) => model.model_id === "dinov2_vits14" && model.available)
      || models.find((model) => model.available);
    if (preferred) setModelId(preferred.model_id);
  }, [models, modelId]);

  const visiblePoints = run?.unit === "object" ? (objectView?.points || []) : (run?.points || []);
  const visibleScores = run?.unit === "object" ? (objectView?.sparsity || []) : (run?.sparsity || []);
  const labels = useMemo(() => [...new Set(visiblePoints.map((point) => point.label))].sort(), [visiblePoints]);
  const classColors = useMemo(() => run ? labelColors(run.points) : new Map<string, string>(), [run]);
  const effectiveColorMode = labels.length > 1 ? colorMode : "sparsity";
  const selectedRecords = useMemo(() => {
    if (!run) return [];
    return selectedIndices
      .filter((index) => Number.isInteger(index) && run.records[index] !== undefined)
      .map((index) => run.records[index]);
  }, [run, selectedIndices]);

  function objectViewInput(overrides: Partial<CoverageObjectViewInput> = {}): CoverageObjectViewInput {
    return {
      class_label: classFilter || null,
      min_short_px: minShortPx,
      confidence_min: confidenceMin,
      confidence_max: confidenceMax,
      method,
      dimension,
      k,
      block_k: blockK,
      ...overrides,
    };
  }

  async function refreshObjectView(overrides: Partial<CoverageObjectViewInput> = {}) {
    if (!run || run.unit !== "object") return;
    const requestId = ++objectViewRequest.current;
    setProjectionBusy(true);
    setError("");
    try {
      const result = await api.coverageObjectView(run.run_id, objectViewInput(overrides));
      if (objectViewRequest.current !== requestId) return;
      setObjectView(result);
      setK(result.k);
      setSelectedIndices([]);
    } catch (reason) {
      if (objectViewRequest.current === requestId) {
        setError(reason instanceof Error ? reason.message : "Could not update the object coverage view.");
      }
    } finally {
      if (objectViewRequest.current === requestId) setProjectionBusy(false);
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
      setNotice("Native folder picker is not connected. Paste a local folder path below.");
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

  async function startAnalysis(event: FormEvent) {
    event.preventDefault();
    setError("");
    setNotice("");
    if (!connected) {
      setError("Connect to the local worker before starting coverage analysis.");
      return;
    }
    if (!folders.length) {
      setError("Choose at least one image folder.");
      return;
    }
    if (!modelId) {
      setError("Choose an installed feature model.");
      return;
    }

    setBusy(true);
    setRun(null);
    setObjectView(null);
    setJob(null);
    setSelectedIndices([]);
    try {
      const created = await api.createCoverageJob(folders, modelId, unit, method, dimension, k);
      let status: JobStatus | null = null;
      const deadline = Date.now() + 10 * 60_000;
      while (Date.now() < deadline) {
        status = await api.coverageJob(created.job_id);
        setJob(status);
        if (["completed", "failed", "cancelled"].includes(status.state)) break;
        await pause(250);
      }
      if (!status || status.state !== "completed" || !status.run_id) {
        if (status?.error?.message) throw new Error(status.error.message);
        if (status?.state === "cancelled") return;
        throw new Error("Coverage analysis did not finish. Retry after checking the local worker.");
      }
      const coverageRun = await api.coverageRun(status.run_id);
      let initialObjectView: CoverageObjectView | null = null;
      if (coverageRun.unit === "object") {
        initialObjectView = await api.coverageObjectView(coverageRun.run_id, {
          class_label: null,
          min_short_px: 0,
          confidence_min: 0,
          confidence_max: 1,
          method: coverageRun.requested_method,
          dimension: coverageRun.dimension,
          k: coverageRun.k,
          block_k: blockK,
        });
      }
      setRun(coverageRun);
      setObjectView(initialObjectView);
      setUnit(coverageRun.unit);
      setMethod(coverageRun.requested_method);
      setDimension(coverageRun.dimension);
      setK(initialObjectView?.k || coverageRun.k);
      setClassFilter("");
      setMinShortPx(0);
      setConfidenceMin(0);
      setConfidenceMax(1);
      setColorMode(coverageRun.unit === "object" ? "sparsity" : coverageRun.records.some((record) => record.label !== coverageRun.records[0]?.label)
        ? "class" : "sparsity");
      setNotice(`${coverageRun.records.length} ${coverageRun.unit === "object" ? "objects" : "images"} analyzed with ${coverageRun.model_id}.`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Coverage analysis failed.");
    } finally {
      setBusy(false);
    }
  }

  async function cancelAnalysis() {
    if (!job?.job_id || !["queued", "running", "cancelling"].includes(job.state)) return;
    try {
      await api.cancelCoverage(job.job_id);
      setJob((current) => current ? { ...current, state: "cancelling", message: "Cancellation requested" } : current);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not cancel coverage analysis.");
    }
  }

  async function changeProjection(nextMethod: ProjectionMethod, nextDimension = dimension) {
    setMethod(nextMethod);
    setDimension(nextDimension);
    if (!run || projectionBusy) return;
    if (run.unit === "object") {
      await refreshObjectView({ method: nextMethod, dimension: nextDimension });
      return;
    }
    const requestId = ++projectionRequest.current;
    setProjectionBusy(true);
    setError("");
    try {
      const result = await api.projectCoverage(run.run_id, nextMethod, nextDimension);
      if (projectionRequest.current !== requestId) return;
      setRun((current) => current ? {
        ...current,
        method: result.method,
        requested_method: result.requested_method,
        dimension: result.dimension,
        points: result.points,
        warnings: result.warnings,
      } : current);
    } catch (reason) {
      if (projectionRequest.current === requestId) {
        setError(reason instanceof Error ? reason.message : "Could not switch the coverage projection.");
      }
    } finally {
      if (projectionRequest.current === requestId) setProjectionBusy(false);
    }
  }

  async function changeK(nextK: number) {
    setK(nextK);
    if (!run || !Number.isFinite(nextK) || nextK < 1 || sparsityBusy) return;
    if (run.unit === "object") {
      await refreshObjectView({ k: nextK });
      return;
    }
    const requestId = ++sparsityRequest.current;
    setSparsityBusy(true);
    setError("");
    try {
      const result = await api.coverageSparsity(run.run_id, nextK);
      if (sparsityRequest.current !== requestId) return;
      setRun((current) => current ? {
        ...current,
        k: result.k,
        sparsity: result.scores,
        sparse_threshold: result.threshold,
        sparse_count: result.sparse_count,
        sparse_percentile: result.sparse_percentile,
      } : current);
      setK(result.k);
    } catch (reason) {
      if (sparsityRequest.current === requestId) {
        setError(reason instanceof Error ? reason.message : "Could not update the sparsity values.");
      }
    } finally {
      if (sparsityRequest.current === requestId) setSparsityBusy(false);
    }
  }

  function toggleSelection(index: number) {
    setSelectedIndices((current) => current.includes(index)
      ? current.filter((item) => item !== index)
      : [...current, index]);
  }

  function addSelection(indices: number[]) {
    const valid = indices.filter((index) => run?.records[index] !== undefined);
    if (!valid.length) return;
    setSelectedIndices([...new Set(valid)].sort((left, right) => left - right));
  }

  function clearSelection() {
    setSelectedIndices([]);
  }

  function selectBlock(blockId: number) {
    if (!objectView) return;
    const indices = objectView.points
      .filter((point) => point.block_id === blockId)
      .map((point) => point.index);
    if (indices.length) setSelectedIndices(indices);
  }

  const colorValues = effectiveColorMode === "sparsity" ? visibleScores : undefined;
  const pointSizes = effectiveColorMode === "class" && visibleScores.length ? (() => {
    const min = Math.min(...visibleScores);
    const max = Math.max(...visibleScores);
    const span = max - min;
    const base = dimension === 2 ? 6 : 4;
    const sizeRange = dimension === 2 ? 8 : 4;
    return visibleScores.map((score) => base + (span > 1e-12 ? sizeRange * (score - min) / span : 0));
  })() : undefined;

  return (
    <div className="workspace coverage-workspace">
      <aside className="control-panel">
        <div className="eyebrow">COMPLETENESS</div>
        <h1>Embedding coverage</h1>
        <p className="intro">Map one dataset, find sparse neighborhoods, and inspect the images behind each selected point.</p>

        <form onSubmit={(event) => void startAnalysis(event)}>
          <section className="control-section">
            <div className="section-heading"><span className="step-number">01</span><h2>Image folders</h2></div>
            <button className="host-picker" type="button" onClick={() => void chooseFolder()}>
              <span className="folder-icon" aria-hidden="true">▱</span>
              Choose folder from device
              <span className="picker-hint">Native</span>
            </button>
            <label className="field-label" htmlFor="coverage-folder-path">Coverage image folder</label>
            <div className="folder-input-row">
              <input
                id="coverage-folder-path"
                data-testid="coverage-folder-path"
                value={folderDraft}
                onChange={(event) => setFolderDraft(event.target.value)}
                onKeyDown={(event) => event.key === "Enter" && (event.preventDefault(), addFolder())}
                placeholder="Paste a local folder path"
                autoComplete="off"
              />
              <button type="button" className="add-folder" onClick={() => addFolder()} aria-label="Add coverage folder">+</button>
            </div>
            {folders.length > 0 && (
              <ul className="folder-list" aria-label="Coverage folders">
                {folders.map((folder) => (
                  <li key={folder} title={folder}>
                    <span className="folder-chip-dot" />
                    <span className="folder-chip-name">{folder.split(/[\\/]/).filter(Boolean).at(-1) || folder}</span>
                    <button type="button" aria-label={`Remove coverage folder ${folder}`} onClick={() => setFolders((current) => current.filter((item) => item !== folder))}>×</button>
                  </li>
                ))}
              </ul>
            )}
            <div className="field-note">Use a folder containing class folders and images, or paste a flat image folder.</div>
          </section>

          <section className="control-section">
            <div className="section-heading"><span className="step-number">02</span><h2>Analysis setup</h2></div>
            <label className="field-label" htmlFor="coverage-model">Feature model</label>
            <select
              id="coverage-model"
              data-testid="coverage-model"
              value={modelId}
              onChange={(event) => setModelId(event.target.value)}
              disabled={loadingModels || busy}
            >
              <option value="">Choose a model</option>
              {models.filter((model) => model.available).map((model) => (
                <option value={model.model_id} key={model.model_id}>{model.name}</option>
              ))}
            </select>
            <label className="field-label field-label-spaced" htmlFor="coverage-unit">Analysis unit</label>
            <select
              id="coverage-unit"
              data-testid="coverage-unit"
              value={unit}
              onChange={(event) => setUnit(event.target.value as VisualizeUnit)}
              disabled={busy}
            >
              <option value="image">Whole image</option>
              <option value="object">Object level (YOLO)</option>
            </select>
            <label className="field-label field-label-spaced" htmlFor="coverage-method">Projection method</label>
            <select
              id="coverage-method"
              data-testid="coverage-method"
              value={method}
              onChange={(event) => void changeProjection(event.target.value as ProjectionMethod)}
              disabled={projectionBusy || busy}
            >
              {projectionOptions.map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}
            </select>
            <div className="coverage-inline-controls">
              <div>
                <span className="field-label">Projection dimension</span>
                <div className="dimension-toggle" role="group" aria-label="Coverage projection dimension">
                  <button type="button" data-testid="coverage-dimension-2d" aria-pressed={dimension === 2} onClick={() => void changeProjection(method, 2)} disabled={projectionBusy || busy}>2D</button>
                  <button type="button" data-testid="coverage-dimension-3d" aria-pressed={dimension === 3} onClick={() => void changeProjection(method, 3)} disabled={projectionBusy || busy}>3D</button>
                </div>
              </div>
              <div>
                <label className="field-label" htmlFor="coverage-k">Sparsity k</label>
                <input
                  id="coverage-k"
                  data-testid="coverage-k"
                  type="number"
                  min={1}
                  max={run ? Math.max(1, (run.unit === "object" ? objectView?.records.length || run.records.length : run.records.length) - 1) : 1000}
                  value={k}
                  onChange={(event) => void changeK(event.target.valueAsNumber)}
                  disabled={sparsityBusy || projectionBusy || busy}
                />
              </div>
            </div>
            {run && labels.length > 1 && (
              <fieldset className="coverage-color-mode">
                <legend className="field-label">Point color</legend>
                <label><input type="radio" name="coverage-color-mode" value="class" checked={colorMode === "class"} onChange={() => setColorMode("class")} /> Class</label>
                <label><input type="radio" name="coverage-color-mode" value="sparsity" checked={colorMode === "sparsity"} onChange={() => setColorMode("sparsity")} /> Sparsity</label>
              </fieldset>
            )}
            <button className="run-button" type="submit" disabled={busy || !connected || loadingModels || !modelId || !folders.length}>
              {busy ? "Analyzing…" : "Run coverage analysis"}
            </button>
            {busy && <button className="cancel-button" type="button" onClick={() => void cancelAnalysis()}>Cancel analysis</button>}
            {!connected && <div className="field-note">Connect to the local worker to analyze images.</div>}
          </section>
        </form>
      </aside>

      <main className="results-area coverage-results">
        <div className="coverage-heading">
          <div>
            <div className="eyebrow">{run?.unit === "object" ? "OBJECT-LEVEL WORKSPACE" : "IMAGE-LEVEL WORKSPACE"}</div>
            <h2>Embedding coverage map</h2>
            <p>{run?.unit === "object"
              ? "Object sparsity uses cosine distance in the original feature space; projection only positions points on this map."
              : "Sparsity uses cosine distance in the original feature space; projection only positions points on this map."}</p>
          </div>
        </div>
        {error && <div className="error-banner" role="alert">{error}</div>}
        {notice && <div className="notice-banner" role="status">{notice}</div>}
        {job && <div className="job-status" data-testid="coverage-job-status" role="status">
          <span>{job.message}</span>
          {job.total > 0 && <span>{job.done}/{job.total} · {Math.round(job.percent)}%</span>}
        </div>}
        {run ? (
          <>
            {run.warnings.map((warning, index) => <div className="coverage-warning" role="status" key={`${warning}-${index}`}>{warning}</div>)}
            {objectView?.warnings.map((warning, index) => <div className="coverage-warning" role="status" key={`object-${warning}-${index}`}>{warning}</div>)}
            {run.unit === "object" && objectView && (
              <section className="coverage-object-panel" aria-label="Object coverage filters and blocks">
                <div className="coverage-object-panel-heading">
                  <div><h3>Object coverage by class</h3><p>Class, confidence and source-pixel filters are applied together to records, labels, embeddings and boxes.</p></div>
                </div>
                <div className="coverage-object-filters">
                  <div className="coverage-object-field">
                    <label className="field-label" htmlFor="coverage-object-class">Object class</label>
                    <select
                      id="coverage-object-class"
                      data-testid="coverage-object-class"
                      value={classFilter}
                      onChange={(event) => setClassFilter(event.target.value)}
                    >
                      <option value="">All classes</option>
                      {[...new Set(run.records.map((record) => record.label))].sort().map((label) => (
                        <option key={label} value={label}>{label}</option>
                      ))}
                    </select>
                  </div>
                  <div className="coverage-object-field">
                    <label className="field-label" htmlFor="coverage-min-short-px">Hide objects smaller than (px)</label>
                    <input
                      id="coverage-min-short-px"
                      data-testid="coverage-min-short-px"
                      type="number"
                      min={0}
                      max={4096}
                      value={minShortPx}
                      onChange={(event) => setMinShortPx(Math.max(0, Math.min(4096, event.target.valueAsNumber || 0)))}
                    />
                  </div>
                  {objectView.has_confidence && (
                    <>
                      <div className="coverage-object-field">
                        <label className="field-label" htmlFor="coverage-confidence-min">Confidence minimum</label>
                        <input
                          id="coverage-confidence-min"
                          data-testid="coverage-confidence-min"
                          type="number"
                          min={0}
                          max={1}
                          step={0.05}
                          value={confidenceMin}
                          onChange={(event) => setConfidenceMin(Math.max(0, Math.min(1, event.target.valueAsNumber || 0)))}
                        />
                      </div>
                      <div className="coverage-object-field">
                        <label className="field-label" htmlFor="coverage-confidence-max">Confidence maximum</label>
                        <input
                          id="coverage-confidence-max"
                          data-testid="coverage-confidence-max"
                          type="number"
                          min={0}
                          max={1}
                          step={0.05}
                          value={confidenceMax}
                          onChange={(event) => setConfidenceMax(Number.isNaN(event.target.valueAsNumber)
                            ? 1 : Math.max(0, Math.min(1, event.target.valueAsNumber)))}
                        />
                      </div>
                    </>
                  )}
                  <div className="coverage-object-field">
                    <label className="field-label" htmlFor="coverage-block-k">Object block count K</label>
                    <input
                      id="coverage-block-k"
                      data-testid="coverage-block-k"
                      type="number"
                      min={3}
                      max={20}
                      value={blockK}
                      onChange={(event) => setBlockK(Math.max(3, Math.min(20, event.target.valueAsNumber || 3)))}
                    />
                  </div>
                  <button
                    type="button"
                    data-testid="coverage-apply-object-filters"
                    className="apply-object-filters"
                    onClick={() => void refreshObjectView()}
                    disabled={projectionBusy || confidenceMin > confidenceMax}
                  >Apply object filters</button>
                </div>
                {confidenceMin > confidenceMax && <p className="coverage-warning" role="alert">Confidence minimum must not exceed its maximum.</p>}
                <div className="coverage-object-summary">
                  <h4>Class counts and sparse samples</h4>
                  <div className="coverage-object-table-wrap">
                    <table data-testid="coverage-class-summary">
                      <thead><tr><th>Class</th><th>Objects</th><th>Block analysis</th><th>Top 20% sparse</th><th>Max sparsity</th></tr></thead>
                      <tbody>
                        {objectView.class_summary.map((summary) => (
                          <tr key={summary.label} data-testid="coverage-class-summary-row" data-label={summary.label}>
                            <td>{summary.label}</td><td>{summary.count}</td>
                            <td>{summary.analyzable ? "Ready" : `Small class (${summary.count})`}</td>
                            <td>{summary.sparse_count}</td><td>{summary.max_sparsity.toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {classFilter && objectView.blocks.length > 0 ? (
                    <div className="coverage-object-blocks" data-testid="coverage-object-blocks">
                      <h4>{classFilter} blocks · K={objectView.block_k}</h4>
                      <div className="coverage-block-list">
                        {objectView.blocks.map((block) => (
                          <button
                            type="button"
                            key={block.block_id}
                            data-testid="coverage-object-block"
                            data-block-id={block.block_id}
                            onClick={() => selectBlock(block.block_id)}
                          >
                            <strong>Block {block.block_id + 1}</strong>
                            <span>{block.count} objects</span>
                            <small>Mean sparsity {block.mean_sparsity.toFixed(3)}</small>
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : classFilter ? (
                    <p data-testid="coverage-no-blocks">This class has too few objects for block analysis.</p>
                  ) : (
                    <p>Select a class to inspect its object blocks.</p>
                  )}
                </div>
              </section>
            )}
            <div className="coverage-metrics" aria-label="Coverage summary">
              <article><strong data-testid="coverage-image-count">{visiblePoints.length}</strong><span>{run.unit === "object" ? "Objects" : "Images"}</span></article>
              <article><strong data-testid="coverage-sparse-count">{run.unit === "object" ? objectView?.sparse_count || 0 : run.sparse_count}</strong><span>Top {run.unit === "object" ? 20 : 100 - run.sparse_percentile}% sparsity candidates</span></article>
              <article><strong>{run.model_id}</strong><span>Feature model</span></article>
              <article><strong data-testid="coverage-k-summary">{run.unit === "object" ? objectView?.k || 1 : run.k}</strong><span>Nearest neighbors (k)</span></article>
            </div>
            <p className="coverage-caveat">Sparse means relatively few nearby examples in this dataset. It is a review clue, not a measured real-world coverage guarantee.</p>
            <div className="coverage-plot-toolbar">
              <p>{dimension === 2
                ? "Drag across the 2D map to select images. Empty plot events keep the current selection; use Clear selection to clear it."
                : "3D supports point selection. Switch to 2D to box-select images; use Clear selection to clear the current selection."}</p>
              <button type="button" data-testid="coverage-clear-selection" className="clear-selection" onClick={clearSelection} disabled={!selectedIndices.length}>
                Clear selection ({selectedIndices.length})
              </button>
            </div>
            <div className="coverage-plot" data-testid="coverage-plot">
              {dimension === 2 ? (
                <ScatterPlot2D
                  points={visiblePoints}
                  selectedIndices={selectedIndices}
                  onToggleSelection={toggleSelection}
                  onAddSelection={addSelection}
                  viewKey={`${run.run_id}:${run.unit}:${objectView?.method || run.method}:2:${effectiveColorMode}:${classFilter}`}
                  testIdPrefix="coverage"
                  colorValues={colorValues}
                  labelPalette={coverageLabelColors}
                  pointSizes={pointSizes}
                  scoreValues={visibleScores}
                  scoreLabel="Sparsity"
                />
              ) : (
                <ScatterPlot3D
                  points={visiblePoints}
                  selectedIndices={selectedIndices}
                  onToggleSelection={toggleSelection}
                  viewKey={`${run.run_id}:${run.unit}:${objectView?.method || run.method}:3:${effectiveColorMode}:${classFilter}`}
                  colorValues={colorValues}
                  labelPalette={coverageLabelColors}
                  pointSizes={pointSizes}
                  scoreValues={visibleScores}
                  scoreLabel="Sparsity"
                  testIdPrefix="coverage"
                />
              )}
            </div>
            <div className="coverage-legend" aria-label="Coverage map legend">
              {effectiveColorMode === "class" ? labels.map((label) => (
                <span key={label} data-testid="coverage-class-legend-item" data-label={label}>
                  <i className="coverage-legend-swatch" style={{ background: classColors.get(label) || coverageLabelColors[0] }} />{label || "(unlabeled)"}
                </span>
              )) : <span><i className="coverage-legend-gradient" />Sparse neighborhoods appear brighter; hover a point to read its score.</span>}
            </div>
            <section className="coverage-selection" aria-label={run.unit === "object" ? "Selected objects" : "Selected images"}>
              <div className="coverage-selection-heading">
              <div><h3>{run.unit === "object" ? "Selected objects" : "Selected images"}</h3><span data-testid="coverage-selected-count">{selectedRecords.length} selected</span></div>
              </div>
              {selectedRecords.length ? (
                <div className="coverage-gallery">
                  {selectedRecords.map((record) => (
                    <article className="coverage-image-card" key={record.index} data-testid="coverage-selected-image" data-index={record.index}>
                      <img src={api.coverageThumbnailUrl(run.run_id, record.index)} alt={record.filename} loading="lazy" />
                      <strong>{record.filename}</strong>
                      <span className="coverage-image-label" style={{ background: classColors.get(record.label) || coverageLabelColors[0] }}>{record.label || "(unlabeled)"}</span>
                      <small>{record.split}</small>
                      {record.obj_index != null && <small>Object {record.obj_index + 1} · short side {record.short_px ?? "?"} px</small>}
                    </article>
                  ))}
                </div>
              ) : <p className="coverage-empty-selection">No {run.unit === "object" ? "objects" : "images"} selected yet. Click a point or drag a box on the map.</p>}
            </section>
          </>
        ) : (
          <div className="empty-canvas coverage-empty-canvas">
            <div className="empty-illustration" aria-hidden="true"><span className="orbit orbit-one" /><span className="orbit orbit-two" /><i className="dot dot-one" /><i className="dot dot-two" /><i className="dot dot-three" /><i className="dot dot-four" /><b>CV</b></div>
            <h3>Start a separate coverage analysis</h3>
            <p>Choose an image dataset and local feature model. The analysis will run without requiring a prior Visualize run.</p>
            <div className="empty-facts"><span><b>01</b> Choose images</span><span><b>02</b> Extract features locally</span><span><b>03</b> Inspect sparse regions</span></div>
          </div>
        )}
        <footer className="canvas-footer"><span>VisualLatent · React Web</span><span>{run?.unit === "object" ? "Object-level coverage" : "Image-level coverage"}</span></footer>
      </main>
    </div>
  );
}
