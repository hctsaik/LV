import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import {
  api,
  type EmbeddingPoint,
  type JobStatus,
  type ModelOption,
  type ProjectionMethod,
  type VisualizationRun,
} from "./api";
import CompareDistributions from "./CompareDistributions";
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

function ScatterPlot2D({ points, selectedIndex, onSelect }: {
  points: EmbeddingPoint[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
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
          const active = selectedIndex === point.index;
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
              onClick={() => onSelect(point.index)}
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

function ScatterPlot({ points, selectedIndex, onSelect, dimension, viewKey }: {
  points: EmbeddingPoint[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
  dimension: 2 | 3;
  viewKey: string;
}) {
  if (dimension === 3) {
    return (
      <Suspense fallback={<div className="plot-frame plot-loading" role="status">Loading 3D plot…</div>}>
        <LazyScatterPlot3D points={points} selectedIndex={selectedIndex} onSelect={onSelect} viewKey={viewKey} />
      </Suspense>
    );
  }
  return <ScatterPlot2D points={points} selectedIndex={selectedIndex} onSelect={onSelect} />;
}

export default function App() {
  const [activeTool, setActiveTool] = useState<"visualize" | "compare">("visualize");
  const [models, setModels] = useState<ModelOption[]>([]);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  const [selectedMethods, setSelectedMethods] = useState<ProjectionMethod[]>(["sumap"]);
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
    setSelectedIndex(null);
    setLabelFilter("all");
    setSplitFilter("all");
    setDimension(2);
    setBusy(true);
    setJob(null);
    try {
      const created = await api.createJob(selectedFolders, selectedModelIds, selectedMethods);
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
              <p>{run && currentView ? run.records.length.toLocaleString() + " images · " + activeModelId + " · " + currentView.method.toUpperCase() : "Choose a folder and run an analysis to see your collection."}</p>
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
                  <div className="plot-title"><span className="chart-glyph" aria-hidden="true">⊙</span><div><strong>Image embeddings</strong><small>Click a point to inspect its source</small></div></div>
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
                    selectedIndex={selectedIndex}
                    onSelect={setSelectedIndex}
                    dimension={dimension}
                    viewKey={activeModelId + "-" + currentView.method + "-" + splitFilter + "-" + labelFilter}
                  />
                ) : <div className="empty-filter">No images match these filters.</div>}
                <div className="plot-legend">
                  {labels.map((label) => <span key={label}><i style={{ background: colorMap(currentView.points).get(label) || labelColors[0] }} />{label}</span>)}
                  <span className="legend-separator" />
                  <small>● train　■ test　◆ valid</small>
                  {selectedIndex !== null && <button type="button" className="clear-selection" onClick={() => setSelectedIndex(null)}>Clear selection</button>}
                </div>
              </section>

              <aside className="detail-card">
                <div className="detail-card-heading"><div><div className="eyebrow">IMAGE DETAIL</div><h3>{selectedRecord ? "Selected image" : "Select a point"}</h3></div><span className="detail-count">{selectedRecord ? "#" + (selectedRecord.index + 1) : "—"}</span></div>
                {selectedRecord && run ? (
                  <>
                    <div className="image-preview"><img data-testid="selected-image" src={"/api/v1/runs/" + encodeURIComponent(job?.run_id || "") + "/images/" + selectedRecord.index} alt={selectedRecord.filename} /></div>
                    <div className="image-name" data-testid="selected-image-name" title={selectedRecord.filename}>{selectedRecord.filename}</div>
                    <div className="metadata-list">
                      <div><span>Class</span><strong>{selectedRecord.label}</strong></div>
                      <div><span>Split</span><strong>{selectedRecord.split}</strong></div>
                      <div><span>Model</span><strong>{activeModelId}</strong></div>
                      <div><span>Method</span><strong>{currentView.method.toUpperCase()}</strong></div>
                    </div>
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
