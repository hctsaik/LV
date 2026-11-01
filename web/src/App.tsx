import { useEffect, useMemo, useState } from "react";
import { api, type EmbeddingPoint, type JobStatus, type ModelOption, type VisualizationRun } from "./api";

const labelColors = ["#4976d0", "#e08144", "#38a27c", "#9a6cc3", "#c95567", "#718096"];

function pause(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function ScatterPlot({ points, selectedIndex, onSelect }: {
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
  const labels = [...new Set(points.map((point) => point.label))].sort();
  const colorByLabel = new Map(labels.map((label, index) => [label, labelColors[index % labelColors.length]]));

  return (
    <div className="plot-frame" data-testid="embedding-plot">
      <svg className="scatter" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Image embedding scatter plot">
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="axis" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="axis" />
        <text x={width / 2} y={height - 8} className="axis-label">Component 1</text>
        <text x={15} y={height / 2} className="axis-label vertical-label">Component 2</text>
        {points.map((point) => {
          const x = padding + ((point.x - xMin) / xSpan) * (width - padding * 2);
          const y = height - padding - ((point.y - yMin) / ySpan) * (height - padding * 2);
          const color = colorByLabel.get(point.label) || labelColors[0];
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
              aria-label={`Select ${point.filename}, ${point.label}, ${point.split}`}
              data-testid="embedding-point"
              data-index={point.index}
              className={`point ${active ? "point-active" : ""}`}
              transform={`translate(${x} ${y})`}
              onClick={() => onSelect(point.index)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") onSelect(point.index);
              }}
            >
              <title>{`${point.filename} · ${point.label} · ${point.split}`}</title>
              {active && <circle r={12} className="point-selection-ring" />}
              {shape}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export default function App() {
  const [models, setModels] = useState<ModelOption[]>([]);
  const [modelId, setModelId] = useState("");
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
        setModelId((current) => current || response.items[0]?.model_id || "");
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "The local worker is unavailable.");
      })
      .finally(() => active && setLoadingModels(false));
    return () => { active = false; };
  }, []);

  const currentRecords = useMemo(() => {
    if (!run) return [];
    return run.points.filter((point) =>
      (labelFilter === "all" || point.label === labelFilter) &&
      (splitFilter === "all" || point.split === splitFilter),
    );
  }, [run, labelFilter, splitFilter]);
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
      const path = result.path || result.paths?.[0];
      if (!path) throw new Error("The host did not return a folder path.");
      addFolder(path);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The folder picker failed.");
    }
  }

  async function runAnalysis() {
    const selectedFolders = [...folders];
    const draft = folderDraft.trim();
    if (draft && !selectedFolders.includes(draft)) selectedFolders.push(draft);
    if (!selectedFolders.length) {
      setError("Choose at least one image folder.");
      return;
    }
    if (!modelId) {
      setError("Select an installed model before running.");
      return;
    }
    setFolders(selectedFolders);
    setFolderDraft("");
    setError("");
    setNotice("");
    setRun(null);
    setSelectedIndex(null);
    setLabelFilter("all");
    setSplitFilter("all");
    setBusy(true);
    setJob(null);
    try {
      const created = await api.createJob(selectedFolders, modelId, "pca");
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
        <div className={`connection ${connected ? "connection-ready" : "connection-waiting"}`}>
          <span className="connection-dot" />
          {connected ? "Local worker connected" : "Connecting to local worker"}
        </div>
      </header>

      <div className="workspace">
        <aside className="control-panel">
          <div className="eyebrow">VISUALIZE EMBEDDINGS</div>
          <h1>Explore your image space</h1>
          <p className="intro">Compare how your local image collections group together using the model already installed on this device.</p>

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
                    <button type="button" aria-label={`Remove ${folder}`} onClick={() => setFolders((current) => current.filter((item) => item !== folder))}>×</button>
                  </li>
                ))}
              </ul>
            )}
            <div className="field-note">Folders stay on this device. LV writes analysis cache to its app cache, not into your dataset.</div>
          </section>

          <section className="control-section model-section">
            <div className="section-heading"><span className="step-number">02</span><h2>Analysis setup</h2></div>
            <label className="field-label" htmlFor="model-select">Feature model</label>
            <select id="model-select" value={modelId} onChange={(event) => setModelId(event.target.value)} disabled={loadingModels || models.length === 0}>
              {models.length === 0 && <option value="">No installed models found</option>}
              {models.map((model) => <option key={model.model_id} value={model.model_id}>{model.name}</option>)}
            </select>
            {models.length === 0 && !loadingModels && <div className="model-empty">Install a supported LV model to enable analysis.</div>}
            <label className="field-label field-label-spaced" htmlFor="projection-select">Projection</label>
            <select id="projection-select" defaultValue="pca" disabled>
              <option value="pca">PCA · 2D</option>
            </select>
            <div className="field-note">Projection is for visual inspection. Distances in 2D do not replace the original embedding space.</div>
          </section>

          {error && <div className="alert alert-error" role="alert">{error}</div>}
          {notice && <div className="alert alert-info" role="status">{notice}</div>}

          <button className="run-button" type="button" onClick={runAnalysis} disabled={busy || !connected || loadingModels || !modelId}>
            <span aria-hidden="true">{busy ? "◌" : "▶"}</span>
            {busy ? "Analysis running" : "Run analysis"}
          </button>
          {busy && (
            <button className="cancel-button" type="button" onClick={cancelAnalysis}>Cancel analysis</button>
          )}
          <div className="privacy-note"><span aria-hidden="true">⌁</span> Local processing · no dataset upload</div>
        </aside>

        <main className="results-area">
          <div className="results-header">
            <div>
              <div className="eyebrow">ANALYSIS CANVAS</div>
              <h2>{run ? "Embedding distribution" : "Your workspace is ready"}</h2>
              <p>{run ? `${run.records.length.toLocaleString()} images · ${run.model_id} · PCA` : "Choose a folder and run an analysis to see your collection."}</p>
            </div>
            {run && <div className="run-badge"><span className="badge-dot" /> Result ready</div>}
          </div>

          {busy && (
            <div className="progress-card" data-testid="job-status" role="status">
              <div className="progress-row"><span>{job?.message || "Preparing analysis"}</span><strong>{Math.round(job?.percent || 0)}%</strong></div>
              <div className="progress-track"><div className="progress-fill" style={{ width: `${Math.max(3, job?.percent || 0)}%` }} /></div>
              <div className="progress-foot">{job?.done || 0} / {job?.total || "—"} images</div>
            </div>
          )}
          {!busy && job?.state === "completed" && <div className="completion-label" data-testid="job-status"><span className="completion-check">✓</span> Completed</div>}
          {!busy && job?.state === "cancelled" && <div className="completion-label" data-testid="job-status">Cancelled</div>}

          {run ? (
            <div className="result-grid">
              <section className="plot-card">
                <div className="plot-toolbar">
                  <div className="plot-title"><span className="chart-glyph" aria-hidden="true">⊙</span><div><strong>Image embeddings</strong><small>Click a point to inspect its source</small></div></div>
                  <div className="filters">
                    <label><span>Class</span><select aria-label="Filter by class" value={labelFilter} onChange={(event) => setLabelFilter(event.target.value)}><option value="all">All classes</option>{labels.map((label) => <option key={label}>{label}</option>)}</select></label>
                    <label><span>Split</span><select aria-label="Filter by split" value={splitFilter} onChange={(event) => setSplitFilter(event.target.value)}><option value="all">All splits</option>{splits.map((split) => <option key={split}>{split}</option>)}</select></label>
                  </div>
                </div>
                {currentRecords.length ? (
                  <ScatterPlot points={currentRecords} selectedIndex={selectedIndex} onSelect={setSelectedIndex} />
                ) : <div className="empty-filter">No images match these filters.</div>}
                <div className="plot-legend">
                  {labels.map((label, index) => <span key={label}><i style={{ background: labelColors[index % labelColors.length] }} />{label}</span>)}
                  <span className="legend-separator" />
                  <small>● train　■ test　◆ valid</small>
                </div>
              </section>

              <aside className="detail-card">
                <div className="detail-card-heading"><div><div className="eyebrow">IMAGE DETAIL</div><h3>{selectedRecord ? "Selected image" : "Select a point"}</h3></div><span className="detail-count">{selectedRecord ? `#${selectedRecord.index + 1}` : "—"}</span></div>
                {selectedRecord && run ? (
                  <>
                    <div className="image-preview"><img data-testid="selected-image" src={`/api/v1/runs/${encodeURIComponent((job?.run_id) || "")}/images/${selectedRecord.index}`} alt={selectedRecord.filename} /></div>
                    <div className="image-name" data-testid="selected-image-name" title={selectedRecord.filename}>{selectedRecord.filename}</div>
                    <div className="metadata-list">
                      <div><span>Class</span><strong>{selectedRecord.label}</strong></div>
                      <div><span>Split</span><strong>{selectedRecord.split}</strong></div>
                      <div><span>Model</span><strong>{run.model_id}</strong></div>
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
    </div>
  );
}
