import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import type {
  CoverageObjectView,
  CoverageObjectViewInput,
  CoverageCandidateRole,
  CoverageCandidateView,
  CoverageCandidateViewInput,
  CoverageCandidateWorkRecord,
  CoverageRun,
  JobStatus,
  ModelOption,
  ProjectionMethod,
  VisualizeUnit,
} from "./api";
import AttributeCompleteness from "./AttributeCompleteness";
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
  onCurationCountChange,
}: {
  models: ModelOption[];
  connected: boolean;
  loadingModels: boolean;
  onCurationCountChange?: (count: number) => void;
}) {
  const [folders, setFolders] = useState<string[]>([]);
  const [folderDraft, setFolderDraft] = useState("");
  const [candidateFolders, setCandidateFolders] = useState<string[]>([]);
  const [candidateFolderDraft, setCandidateFolderDraft] = useState("");
  const [candidateRole, setCandidateRole] = useState<CoverageCandidateRole>("pool");
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
  const [viewMode, setViewMode] = useState<"embedding" | "attribute">("embedding");
  const [objectView, setObjectView] = useState<CoverageObjectView | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [candidateJob, setCandidateJob] = useState<JobStatus | null>(null);
  const [candidateView, setCandidateView] = useState<CoverageCandidateView | null>(null);
  const [candidateReady, setCandidateReady] = useState(false);
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);
  const [selectedCandidateIndices, setSelectedCandidateIndices] = useState<number[]>([]);
  const [busy, setBusy] = useState(false);
  const [candidateBusy, setCandidateBusy] = useState(false);
  const [candidateViewBusy, setCandidateViewBusy] = useState(false);
  const [projectionBusy, setProjectionBusy] = useState(false);
  const [sparsityBusy, setSparsityBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const projectionRequest = useRef(0);
  const sparsityRequest = useRef(0);
  const objectViewRequest = useRef(0);
  const candidateViewRequest = useRef(0);

  useEffect(() => {
    if (models.some((model) => model.model_id === modelId && model.available)) return;
    const preferred = models.find((model) => model.model_id === "dinov2_vits14" && model.available)
      || models.find((model) => model.available);
    if (preferred) setModelId(preferred.model_id);
  }, [models, modelId]);

  const visiblePoints = run?.unit === "object" ? (objectView?.points || []) : (run?.points || []);
  const visibleScores = run?.unit === "object" ? (objectView?.sparsity || []) : (run?.sparsity || []);
  const plottedPoints = candidateView?.main_points || visiblePoints;
  const plottedScores = candidateView?.main_sparsity || visibleScores;
  const labels = useMemo(() => [...new Set(plottedPoints.map((point) => point.label))].sort(), [plottedPoints]);
  const classColors = useMemo(() => run ? labelColors(run.points) : new Map<string, string>(), [run]);
  const effectiveColorMode = labels.length > 1 ? colorMode : "sparsity";
  const selectedRecords = useMemo(() => {
    if (!run) return [];
    return selectedIndices
      .filter((index) => Number.isInteger(index) && run.records[index] !== undefined)
      .map((index) => run.records[index]);
  }, [run, selectedIndices]);
  const candidatePoints = candidateView?.candidate_points || [];
  const selectedCandidatePoints = useMemo(() => candidatePoints
    .filter((point) => selectedCandidateIndices.includes(point.index)),
  [candidatePoints, selectedCandidateIndices]);

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

  function candidateViewInput(overrides: Partial<CoverageCandidateViewInput> = {}): CoverageCandidateViewInput {
    return {
      role: candidateRole,
      class_label: classFilter || null,
      min_short_px: minShortPx,
      confidence_min: confidenceMin,
      confidence_max: confidenceMax,
      method,
      dimension,
      k,
      limit: 12,
      ...overrides,
    };
  }

  function invalidateCandidateView(clearSelection = false) {
    ++candidateViewRequest.current;
    setCandidateView(null);
    setCandidateViewBusy(false);
    if (clearSelection) setSelectedCandidateIndices([]);
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

  async function refreshCandidateView(
    overrides: Partial<CoverageCandidateViewInput> = {},
    allowNewCandidateSet = false,
  ) {
    if (!run || (!candidateReady && !allowNewCandidateSet)) return;
    const requestId = ++candidateViewRequest.current;
    const runId = run.run_id;
    setCandidateViewBusy(true);
    setError("");
    setCandidateView(null);
    try {
      const result = await api.coverageCandidateView(runId, candidateViewInput(overrides));
      if (candidateViewRequest.current !== requestId) return;
      setCandidateView(result);
      setSelectedCandidateIndices((current) => current.filter((index) =>
        result.candidate_points.some((point) => point.index === index),
      ));
    } catch (reason) {
      if (candidateViewRequest.current === requestId) {
        setError(reason instanceof Error ? reason.message : "Could not update the candidate comparison.");
      }
    } finally {
      if (candidateViewRequest.current === requestId) setCandidateViewBusy(false);
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

  function addCandidateFolder(value = candidateFolderDraft) {
    const path = value.trim();
    if (!path) return;
    setCandidateFolders((current) => current.includes(path) ? current : [...current, path]);
    setCandidateFolderDraft("");
    invalidateCandidateView(true);
    setCandidateReady(false);
    setNotice("");
  }

  async function chooseCandidateFolder() {
    setError("");
    if (!window.cimHost?.chooseFile) {
      setNotice("Native folder picker is not connected. Paste a local candidate folder path below.");
      return;
    }
    try {
      const result = await window.cimHost.chooseFile();
      if (result.canceled) return;
      const chosen = result.paths?.length ? result.paths : (result.path ? [result.path] : []);
      if (!chosen.length) throw new Error("The host did not return a folder path.");
      chosen.forEach((path) => addCandidateFolder(path));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The candidate folder picker failed.");
    }
  }

  async function startAnalysis(event: FormEvent) {
    event.preventDefault();
    setError("");
    setNotice("");
    if (candidateBusy) {
      setError("Wait for candidate analysis to finish before starting another coverage run.");
      return;
    }
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
    ++candidateViewRequest.current;
    setRun(null);
    setObjectView(null);
    setCandidateView(null);
    setCandidateReady(false);
    setCandidateViewBusy(false);
    setCandidateJob(null);
    setSelectedCandidateIndices([]);
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
      setViewMode("embedding");
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

  async function startCandidateAnalysis() {
    if (!run || !candidateFolders.length || !connected) return;
    setError("");
    setNotice("");
    invalidateCandidateView(true);
    setCandidateReady(false);
    setCandidateJob(null);
    setCandidateBusy(true);
    try {
      const created = await api.createCoverageCandidateJob(run.run_id, candidateFolders);
      let status: JobStatus | null = null;
      const deadline = Date.now() + 10 * 60_000;
      while (Date.now() < deadline) {
        status = await api.coverageCandidateJob(created.job_id);
        setCandidateJob(status);
        if (["completed", "failed", "cancelled"].includes(status.state)) break;
        await pause(250);
      }
      if (!status || status.state !== "completed") {
        if (status?.error?.message) throw new Error(status.error.message);
        if (status?.state === "cancelled") return;
        throw new Error("Candidate analysis did not finish. Retry after checking the local worker.");
      }
      setCandidateReady(true);
      setNotice(`Projected ${status.done} candidate ${run.unit === "object" ? "objects" : "images"} with ${run.model_id}.`);
      await refreshCandidateView({}, true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Candidate analysis failed.");
    } finally {
      setCandidateBusy(false);
    }
  }

  async function cancelCandidateAnalysis() {
    if (!candidateJob?.job_id || !["queued", "running", "cancelling"].includes(candidateJob.state)) return;
    try {
      await api.cancelCoverageCandidateJob(candidateJob.job_id);
      setCandidateJob((current) => current ? { ...current, state: "cancelling", message: "Cancellation requested" } : current);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not cancel candidate analysis.");
    }
  }

  async function changeProjection(nextMethod: ProjectionMethod, nextDimension = dimension) {
    setMethod(nextMethod);
    setDimension(nextDimension);
    if (!run || projectionBusy) return;
    if (run.unit === "object") {
      await Promise.all([
        refreshObjectView({ method: nextMethod, dimension: nextDimension }),
        refreshCandidateView({ method: nextMethod, dimension: nextDimension }),
      ]);
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
      await refreshCandidateView({ method: nextMethod, dimension: nextDimension });
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
      await Promise.all([refreshObjectView({ k: nextK }), refreshCandidateView({ k: nextK })]);
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
      await refreshCandidateView({ k: result.k });
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

  function toggleCandidateSelection(index: number) {
    setSelectedCandidateIndices((current) => current.includes(index)
      ? current.filter((item) => item !== index)
      : [...current, index]);
  }

  function addCandidateSelection(indices: number[]) {
    const valid = indices.filter((index) => candidatePoints.some((point) => point.index === index));
    if (valid.length) setSelectedCandidateIndices([...new Set(valid)].sort((left, right) => left - right));
  }

  async function addCandidatesToCuration(records: Array<Pick<CoverageCandidateWorkRecord, "index">>) {
    if (!run || !candidateView || !records.length) return;
    const indices = [...new Set(records.map((record) => record.index))];
    const source = candidateRole === "pool"
      ? "gap_filler"
      : classFilter ? "reference_class" : "reference";
    const scores = Object.fromEntries(indices.flatMap((index) => {
      const score = candidateView.work_scores[index];
      return score === undefined ? [] : [[index, score]];
    }));
    setError("");
    try {
      const result = await api.addCoverageCandidatesToCuration(run.run_id, indices, source, scores);
      onCurationCountChange?.(result.total);
      setNotice(`Added ${result.added} candidate(s) to curation; skipped ${result.skipped} duplicate(s).`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not add candidates to curation.");
    }
  }

  function selectBlock(blockId: number) {
    if (!objectView) return;
    const indices = objectView.points
      .filter((point) => point.block_id === blockId)
      .map((point) => point.index);
    if (indices.length) setSelectedIndices(indices);
  }

  const colorValues = effectiveColorMode === "sparsity" ? plottedScores : undefined;
  const pointSizes = effectiveColorMode === "class" && plottedScores.length ? (() => {
    const min = Math.min(...plottedScores);
    const max = Math.max(...plottedScores);
    const span = max - min;
    const base = dimension === 2 ? 6 : 4;
    const sizeRange = dimension === 2 ? 8 : 4;
    return plottedScores.map((score) => base + (span > 1e-12 ? sizeRange * (score - min) / span : 0));
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
            <button className="run-button" type="submit" disabled={busy || candidateBusy || !connected || loadingModels || !modelId || !folders.length}>
              {busy ? "Analyzing…" : "Run coverage analysis"}
            </button>
            {busy && <button className="cancel-button" type="button" onClick={() => void cancelAnalysis()}>Cancel analysis</button>}
            {!connected && <div className="field-note">Connect to the local worker to analyze images.</div>}
          </section>
        </form>

        <section className="control-section" aria-label="Coverage view">
          <div className="section-heading"><span className="step-number">04</span><h2>Coverage view</h2></div>
          <div className="coverage-view-switch" role="group" aria-label="Coverage view">
            <button type="button" data-testid="coverage-view-embedding" aria-pressed={viewMode === "embedding"}
              onClick={() => setViewMode("embedding")} disabled={!run}>Embedding map</button>
            <button type="button" data-testid="coverage-view-attributes" aria-pressed={viewMode === "attribute"}
              onClick={() => setViewMode("attribute")} disabled={!run || run.unit !== "image"}>Attribute grid</button>
          </div>
          {run?.unit === "object" && <div className="field-note">Attribute completeness uses whole images. Start an image-level run to enable this view.</div>}
        </section>

        {viewMode === "embedding" && <section className="control-section coverage-candidate-controls" aria-label="Candidate and reference folders">
          <div className="section-heading"><span className="step-number">03</span><h2>Candidate / reference</h2></div>
          <button className="host-picker" type="button" onClick={() => void chooseCandidateFolder()} disabled={candidateBusy}>
            <span className="folder-icon" aria-hidden="true">▱</span>
            Choose candidate folder
            <span className="picker-hint">Native</span>
          </button>
          <label className="field-label" htmlFor="coverage-candidate-folder-path">Candidate or reference folder</label>
          <div className="folder-input-row">
            <input
              id="coverage-candidate-folder-path"
              data-testid="coverage-candidate-folder-path"
              value={candidateFolderDraft}
              onChange={(event) => setCandidateFolderDraft(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && (event.preventDefault(), addCandidateFolder())}
              placeholder="Paste a local folder path"
              autoComplete="off"
              disabled={candidateBusy}
            />
            <button type="button" className="add-folder" onClick={() => addCandidateFolder()} aria-label="Add candidate folder" disabled={candidateBusy}>+</button>
          </div>
          {candidateFolders.length > 0 && (
            <ul className="folder-list" aria-label="Candidate folders">
              {candidateFolders.map((folder) => (
                <li key={folder} title={folder}>
                  <span className="folder-chip-dot" />
                  <span className="folder-chip-name">{folder.split(/[\\/]/).filter(Boolean).at(-1) || folder}</span>
                  <button type="button" aria-label={`Remove candidate folder ${folder}`} onClick={() => {
                    setCandidateFolders((current) => current.filter((item) => item !== folder));
                    invalidateCandidateView(true);
                    setCandidateReady(false);
                  }} disabled={candidateBusy}>×</button>
                </li>
              ))}
            </ul>
          )}
          <div className="field-note">Candidate vectors use the same local model and object crop policy as the main run.</div>
          <label className="field-label field-label-spaced" htmlFor="coverage-candidate-role">Candidate folder role</label>
          <select
            id="coverage-candidate-role"
            data-testid="coverage-candidate-role"
            value={candidateRole}
            onChange={(event) => {
              setCandidateRole(event.target.value as CoverageCandidateRole);
              invalidateCandidateView(true);
            }}
            disabled={candidateBusy}
          >
            <option value="pool">Mining pool · find gap fillers</option>
            <option value="reference">External reference · measure recall</option>
          </select>
          <button
            type="button"
            data-testid="coverage-project-candidates"
            className="run-button"
            onClick={() => void startCandidateAnalysis()}
            disabled={!run || !candidateFolders.length || !connected || busy || candidateBusy}
          >
            {candidateBusy ? "Projecting candidates…" : "Project candidate folders"}
          </button>
          {candidateBusy && <button className="cancel-button" type="button" onClick={() => void cancelCandidateAnalysis()}>Cancel candidate analysis</button>}
          {candidateReady && !candidateBusy && (
            <button
              type="button"
              data-testid="coverage-refresh-candidate-view"
              className="apply-object-filters"
              onClick={() => void refreshCandidateView()}
              disabled={candidateViewBusy || confidenceMin > confidenceMax}
            >Refresh comparison</button>
          )}
        </section>}
      </aside>

      <main className="results-area coverage-results">
        <div className="coverage-heading">
          <div>
            <div className="eyebrow">{run?.unit === "object" ? "OBJECT-LEVEL WORKSPACE" : "IMAGE-LEVEL WORKSPACE"}</div>
            <h2>{viewMode === "attribute" ? "Attribute completeness" : "Embedding coverage map"}</h2>
            <p>{viewMode === "attribute"
              ? "Review sample count and embedding diversity in each interpretable attribute cell."
              : run?.unit === "object"
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
        {candidateJob && <div className="job-status" data-testid="coverage-candidate-job-status" role="status">
          <span>{candidateJob.message}</span>
          {candidateJob.total > 0 && <span>{candidateJob.done}/{candidateJob.total} · {Math.round(candidateJob.percent)}%</span>}
        </div>}
        {run && viewMode === "attribute" && run.unit === "image" ? (
          <AttributeCompleteness run={run} />
        ) : run ? (
          <>
            {run.warnings.map((warning, index) => <div className="coverage-warning" role="status" key={`${warning}-${index}`}>{warning}</div>)}
            {objectView?.warnings.map((warning, index) => <div className="coverage-warning" role="status" key={`object-${warning}-${index}`}>{warning}</div>)}
            {candidateView?.warnings.map((warning, index) => <div className="coverage-warning" role="status" key={`candidate-${warning}-${index}`}>{warning}</div>)}
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
                      onChange={(event) => {
                        setClassFilter(event.target.value);
                        invalidateCandidateView(true);
                      }}
                    >
                      <option value="">All classes</option>
                      {[...new Set([
                        ...run.records.map((record) => record.label),
                        ...(candidateView?.candidate_points.map((point) => point.label) || []),
                        ...(candidateView?.class_gaps.map((gap) => gap.label) || []),
                      ])].sort().map((label) => (
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
                      onChange={(event) => {
                        setMinShortPx(Math.max(0, Math.min(4096, event.target.valueAsNumber || 0)));
                        invalidateCandidateView(true);
                      }}
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
                          onChange={(event) => {
                            setConfidenceMin(Math.max(0, Math.min(1, event.target.valueAsNumber || 0)));
                            invalidateCandidateView(true);
                          }}
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
                          onChange={(event) => {
                            setConfidenceMax(Number.isNaN(event.target.valueAsNumber)
                              ? 1 : Math.max(0, Math.min(1, event.target.valueAsNumber)));
                            invalidateCandidateView(true);
                          }}
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
                    onClick={() => {
                      void refreshObjectView();
                      void refreshCandidateView();
                    }}
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
            {candidateReady && (
              <section className="coverage-candidate-panel" aria-label="Candidate and reference comparison" data-testid="coverage-candidate-panel">
                <div className="coverage-object-panel-heading">
                  <div>
                    <h3>{candidateRole === "pool" ? "Candidate mining pool" : "External reference coverage"}</h3>
                    <p>{candidateRole === "pool"
                      ? "Orange diamonds are candidates ranked by distance from the main dataset; choose records to add to curation."
                      : "Dark diamonds are external reference records projected with the main set; uncovered records indicate reference gaps."}</p>
                  </div>
                  {candidateView && (
                    <div className="coverage-candidate-counts" data-testid="coverage-candidate-counts">
                      <span>{candidateView.main_count} main</span><span>{candidateView.candidate_count} candidates</span>
                    </div>
                  )}
                </div>
                {candidateViewBusy && <p className="field-note" role="status">Updating the joint projection and candidate ranking…</p>}
                {candidateView && candidateRole === "reference" && (
                  <div className="coverage-reference-metrics" data-testid="coverage-reference-metrics">
                    <article><strong data-testid="coverage-reference-recall">{candidateView.recall == null ? "Unavailable" : `${(candidateView.recall * 100).toFixed(1)}%`}</strong><span>Reference recall</span></article>
                    <article><strong data-testid="coverage-reference-uncovered">{candidateView.uncovered_count}</strong><span>Uncovered reference records</span></article>
                    <article><strong>{candidateView.radius == null ? "—" : candidateView.radius.toFixed(3)}</strong><span>Main-neighborhood radius</span></article>
                  </div>
                )}
                {candidateView && candidateRole === "reference" && candidateView.class_gaps.length > 0 && (
                  <div className="coverage-class-gaps">
                    <h4>Class alignment and reference gaps</h4>
                    <div className="coverage-object-table-wrap">
                      <table data-testid="coverage-class-gaps">
                        <thead><tr><th>Class</th><th>Main</th><th>Reference</th><th>Uncovered</th><th>Status</th></tr></thead>
                        <tbody>
                          {candidateView.class_gaps.map((gap) => (
                            <tr key={gap.label} data-testid="coverage-class-gap-row" data-label={gap.label}>
                              <td><button type="button" onClick={() => {
                                setClassFilter(gap.label);
                                invalidateCandidateView(true);
                                void refreshObjectView({ class_label: gap.label });
                                void refreshCandidateView({ class_label: gap.label });
                              }}>{gap.label}</button></td>
                              <td>{gap.main_count}</td><td>{gap.candidate_count}</td><td>{gap.uncovered_count}</td><td>{gap.state}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
                {candidateView && (
                  <div className="coverage-candidate-actions">
                    <button
                      type="button"
                      data-testid="coverage-add-selected-candidates"
                      onClick={() => void addCandidatesToCuration(selectedCandidatePoints)}
                      disabled={!selectedCandidatePoints.length || candidateViewBusy}
                    >Add selected candidates ({selectedCandidatePoints.length}) to curation</button>
                    <button
                      type="button"
                      data-testid="coverage-add-top-candidates"
                      onClick={() => void addCandidatesToCuration(candidateView.work_records)}
                      disabled={!candidateView.work_records.length || candidateViewBusy}
                    >Add top {candidateView.work_records.length} to curation</button>
                    <button
                      type="button"
                      data-testid="coverage-clear-candidate-selection"
                      onClick={() => setSelectedCandidateIndices([])}
                      disabled={!selectedCandidateIndices.length}
                    >Clear candidate selection ({selectedCandidateIndices.length})</button>
                  </div>
                )}
                {candidateView && (
                  <div className="coverage-candidate-worklists">
                    <section aria-label="Candidate worklist">
                      <h4>{candidateRole === "pool" ? "Highest-ranked gap fillers" : "Uncovered reference worklist"}</h4>
                      {candidateView.work_records.length ? (
                        <div className="coverage-gallery">
                          {candidateView.work_records.map((record) => (
                            <article className="coverage-image-card" key={record.index} data-testid="coverage-candidate-work-item" data-index={record.index}>
                              <img src={api.coverageCandidateThumbnailUrl(run.run_id, record.index)} alt={record.filename} loading="lazy" />
                              <strong>{record.filename}</strong>
                              <span className="coverage-image-label" style={{ background: classColors.get(record.label) || coverageLabelColors[0] }}>{record.label || "(unlabeled)"}</span>
                              <small>{record.split} · {candidateRole === "pool" ? "Gap score" : "Distance to main"} {record.score.toFixed(3)}</small>
                              {record.provisional_label && <small>Nearest main class: {record.provisional_label}</small>}
                            </article>
                          ))}
                        </div>
                      ) : <p className="coverage-empty-selection">No candidate records need review under these filters.</p>}
                    </section>
                    <section aria-label="Selected candidate records">
                      <h4>Selected candidate records <span data-testid="coverage-selected-candidate-count">{selectedCandidatePoints.length} selected</span></h4>
                      {selectedCandidatePoints.length ? (
                        <div className="coverage-gallery">
                          {selectedCandidatePoints.map((record) => (
                            <article className="coverage-image-card" key={record.index} data-testid="coverage-selected-candidate" data-index={record.index}>
                              <img src={api.coverageCandidateThumbnailUrl(run.run_id, record.index)} alt={record.filename} loading="lazy" />
                              <strong>{record.filename}</strong>
                              <span className="coverage-image-label" style={{ background: classColors.get(record.label) || coverageLabelColors[0] }}>{record.label || "(unlabeled)"}</span>
                              <small>{record.split}{record.obj_index == null ? "" : ` · Object ${record.obj_index + 1}`}</small>
                              {record.provisional_label && <small>Nearest main class: {record.provisional_label}</small>}
                            </article>
                          ))}
                        </div>
                      ) : <p className="coverage-empty-selection">Select a candidate point on the joint map or from the accessible point list.</p>}
                    </section>
                  </div>
                )}
              </section>
            )}
            <div className="coverage-metrics" aria-label="Coverage summary">
              <article><strong data-testid="coverage-image-count">{plottedPoints.length}</strong><span>{run.unit === "object" ? "Objects" : "Images"}</span></article>
              <article><strong data-testid="coverage-sparse-count">{candidateView?.main_sparse_count ?? (run.unit === "object" ? objectView?.sparse_count || 0 : run.sparse_count)}</strong><span>Top {run.unit === "object" ? 20 : 100 - run.sparse_percentile}% sparsity candidates</span></article>
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
                  points={plottedPoints}
                  selectedIndices={selectedIndices}
                  onToggleSelection={toggleSelection}
                  onAddSelection={addSelection}
                  candidatePoints={candidatePoints}
                  selectedCandidateIndices={selectedCandidateIndices}
                  onToggleCandidateSelection={toggleCandidateSelection}
                  onAddCandidateSelection={addCandidateSelection}
                  viewKey={`${run.run_id}:${run.unit}:${objectView?.method || run.method}:2:${effectiveColorMode}:${classFilter}`}
                  testIdPrefix="coverage"
                  colorValues={colorValues}
                  labelPalette={coverageLabelColors}
                  pointSizes={pointSizes}
                  scoreValues={plottedScores}
                  scoreLabel="Sparsity"
                />
              ) : (
                <ScatterPlot3D
                  points={plottedPoints}
                  selectedIndices={selectedIndices}
                  onToggleSelection={toggleSelection}
                  candidatePoints={candidatePoints}
                  selectedCandidateIndices={selectedCandidateIndices}
                  onToggleCandidateSelection={toggleCandidateSelection}
                  viewKey={`${run.run_id}:${run.unit}:${objectView?.method || run.method}:3:${effectiveColorMode}:${classFilter}`}
                  colorValues={colorValues}
                  labelPalette={coverageLabelColors}
                  pointSizes={pointSizes}
                  scoreValues={plottedScores}
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
        <footer className="canvas-footer"><span>VisualLatent · React Web</span><span>{viewMode === "attribute" ? "Attribute completeness" : run?.unit === "object" ? "Object-level coverage" : "Image-level coverage"}</span></footer>
      </main>
    </div>
  );
}
