import { lazy, Suspense, useMemo, useRef, useState } from "react";
import {
  api,
  type ComparePoint,
  type CompareRecord,
  type CompareRun,
  type CompareView,
  type EmbeddingPoint,
  type HostFolderResult,
  type JobStatus,
  type ModelOption,
  type ProjectionMethod,
} from "./api";

const LazyScatterPlot3D = lazy(() => import("./ScatterPlot3D"));
const groupColors = { a: "#3498db", b: "#e74c3c" } as const;
const methodOptions: { value: ProjectionMethod; label: string }[] = [
  { value: "pca", label: "PCA" },
  { value: "tsne", label: "t-SNE" },
  { value: "umap", label: "UMAP" },
  { value: "lda", label: "LDA (supervised)" },
  { value: "sumap", label: "Supervised UMAP" },
];
type GroupKey = "a" | "b";
type Selection = Record<GroupKey, number[]>;
type DisplayPoint = ComparePoint & { filename: string };

function pause(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function emptySelection(): Selection {
  return { a: [], b: [] };
}

function pointsFor(view: CompareView | null, run: CompareRun | null): DisplayPoint[] {
  if (!view || !run) return [];
  return view.points.map((point) => ({
    ...point,
    filename: (point.group === "a" ? run.records_a : run.records_b)[point.index]?.filename || `Image ${point.index + 1}`,
  }));
}

function ComparePlot2D({
  points,
  selected,
  onSelect,
  onBoxSelect,
}: {
  points: DisplayPoint[];
  selected: Selection;
  onSelect: (group: GroupKey, index: number) => void;
  onBoxSelect: (selection: Selection) => void;
}) {
  const width = 860;
  const height = 430;
  const padding = 44;
  const dragStart = useRef<{ x: number; y: number; pointerId: number } | null>(null);
  const [drag, setDrag] = useState<{ x: number; y: number; currentX: number; currentY: number } | null>(null);
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const xMin = Math.min(...xs, 0);
  const xMax = Math.max(...xs, 0);
  const yMin = Math.min(...ys, 0);
  const yMax = Math.max(...ys, 0);
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1;
  const position = (point: DisplayPoint) => ({
    x: padding + ((point.x - xMin) / xSpan) * (width - padding * 2),
    y: height - padding - ((point.y - yMin) / ySpan) * (height - padding * 2),
  });

  function svgPosition(event: React.PointerEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * width,
      y: ((event.clientY - rect.top) / rect.height) * height,
    };
  }

  function startBox(event: React.PointerEvent<SVGSVGElement>) {
    if (event.button !== 0 || (event.target as Element).closest?.("[data-compare-point]")) return;
    const start = svgPosition(event);
    dragStart.current = { ...start, pointerId: event.pointerId };
    event.currentTarget.setPointerCapture(event.pointerId);
    setDrag({ ...start, currentX: start.x, currentY: start.y });
  }

  function moveBox(event: React.PointerEvent<SVGSVGElement>) {
    if (!dragStart.current || dragStart.current.pointerId !== event.pointerId) return;
    const current = svgPosition(event);
    setDrag({ ...dragStart.current, currentX: current.x, currentY: current.y });
  }

  function finishBox(event: React.PointerEvent<SVGSVGElement>) {
    const start = dragStart.current;
    if (!start || start.pointerId !== event.pointerId) return;
    const end = svgPosition(event);
    dragStart.current = null;
    setDrag(null);
    if (Math.abs(start.x - end.x) < 5 || Math.abs(start.y - end.y) < 5) return;
    const bounds = {
      left: Math.min(start.x, end.x), right: Math.max(start.x, end.x),
      top: Math.min(start.y, end.y), bottom: Math.max(start.y, end.y),
    };
    const next = emptySelection();
    for (const point of points) {
      const p = position(point);
      if (p.x >= bounds.left && p.x <= bounds.right && p.y >= bounds.top && p.y <= bounds.bottom) {
        next[point.group].push(point.index);
      }
    }
    onBoxSelect(next);
  }

  const selectionRect = drag ? {
    x: Math.min(drag.x, drag.currentX),
    y: Math.min(drag.y, drag.currentY),
    width: Math.abs(drag.currentX - drag.x),
    height: Math.abs(drag.currentY - drag.y),
  } : null;

  return (
    <div className="plot-frame" data-testid="compare-plot">
      <svg
        className="compare-scatter"
        viewBox={`0 0 ${width} ${height}`}
        role="group"
        aria-label="A/B image embedding scatter plot. Drag across an empty area to box-select points."
        onPointerDown={startBox}
        onPointerMove={moveBox}
        onPointerUp={finishBox}
        onPointerCancel={finishBox}
      >
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="axis" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="axis" />
        <text x={width / 2} y={height - 8} className="axis-label">Component 1</text>
        <text x={15} y={height / 2} className="axis-label vertical-label">Component 2</text>
        {points.map((point) => {
          const p = position(point);
          const active = selected[point.group].includes(point.index);
          return (
            <g
              key={`${point.group}-${point.index}`}
              role="button"
              tabIndex={0}
              data-testid="compare-point"
              data-compare-point="true"
              aria-label={`Select Folder ${point.group.toUpperCase()} image ${point.filename}`}
              aria-pressed={active}
              className="compare-point"
              transform={`translate(${p.x} ${p.y})`}
              onClick={() => onSelect(point.group, point.index)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") onSelect(point.group, point.index);
              }}
            >
              <circle r={active ? 12 : 9} fill={groupColors[point.group]} opacity="0.2" />
              <circle r={active ? 7 : 5.5} fill={groupColors[point.group]} />
              <title>{`Folder ${point.group.toUpperCase()} · ${point.filename}`}</title>
            </g>
          );
        })}
        {selectionRect && <rect className="compare-selection-box" {...selectionRect} />}
      </svg>
      <div className="compare-legend" aria-label="Plot legend">
        <span><i className="group-a" />Folder A</span>
        <span><i className="group-b" />Folder B</span>
        <small>Click a point or drag a box to inspect images.</small>
      </div>
    </div>
  );
}

function FolderGroup({
  group,
  folders,
  draft,
  setDraft,
  addPaths,
  removeFolder,
  disabled,
  onNotice,
}: {
  group: GroupKey;
  folders: string[];
  draft: string;
  setDraft: (value: string) => void;
  addPaths: (paths: string[]) => void;
  removeFolder: (index: number) => void;
  disabled: boolean;
  onNotice: (message: string) => void;
}) {
  const letter = group.toUpperCase();
  function addDraft() {
    const path = draft.trim();
    if (path) addPaths([path]);
  }
  async function chooseFolders() {
    const picker = window.cimHost?.chooseFile;
    if (!picker) {
      onNotice("Native folder picker is not connected. Enter a local folder path below.");
      return;
    }
    try {
      const result: HostFolderResult = await picker();
      if (result.canceled) return;
      const paths = result.paths?.length ? result.paths : result.path ? [result.path] : [];
      if (!paths.length) {
        onNotice("The host did not return a folder path. Enter a local folder path below.");
        return;
      }
      addPaths(paths);
    } catch {
      onNotice("The native folder picker failed. Enter a local folder path below.");
    }
  }
  return (
    <section className="control-section compare-folder-group" aria-label={`Folder ${letter}`}>
      <div className="section-heading"><span>Image group</span><strong>Folder {letter}</strong></div>
      <div className="compare-add-row">
        <input
          className="compare-path"
          aria-label={`Folder ${letter} path`}
          placeholder="Enter an image folder path"
          value={draft}
          disabled={disabled}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => { if (event.key === "Enter") addDraft(); }}
        />
        <button type="button" className="small-action" onClick={addDraft} disabled={disabled || !draft.trim()}>Add</button>
      </div>
      <button type="button" className="folder-picker-button" onClick={chooseFolders} disabled={disabled}>
        Choose folders from this device
      </button>
      {folders.length ? (
        <ul className="compare-folder-list">
          {folders.map((folder, index) => (
            <li key={`${folder}-${index}`} title={folder}>
              <span>{folder}</span>
              <button type="button" aria-label={`Remove Folder ${letter} ${index + 1}`} onClick={() => removeFolder(index)} disabled={disabled}>×</button>
            </li>
          ))}
        </ul>
      ) : <p className="control-hint">Add one or more folders. Images from this side are merged into one group.</p>}
    </section>
  );
}

export default function Compare({ models, connected }: { models: ModelOption[]; connected: boolean }) {
  const [foldersA, setFoldersA] = useState<string[]>([]);
  const [foldersB, setFoldersB] = useState<string[]>([]);
  const [draftA, setDraftA] = useState("");
  const [draftB, setDraftB] = useState("");
  const [modelId, setModelId] = useState("");
  const [job, setJob] = useState<JobStatus | null>(null);
  const [runId, setRunId] = useState("");
  const [run, setRun] = useState<CompareRun | null>(null);
  const [view, setView] = useState<CompareView | null>(null);
  const [method, setMethod] = useState<ProjectionMethod>("sumap");
  const [dimension, setDimension] = useState<2 | 3>(2);
  const [selection, setSelection] = useState<Selection>(emptySelection);
  const [busy, setBusy] = useState(false);
  const [projecting, setProjecting] = useState(false);
  const [error, setError] = useState("");
  const [projectionError, setProjectionError] = useState("");
  const [notice, setNotice] = useState("");

  const selectedModel = modelId || models[0]?.model_id || "";
  const displayedPoints = useMemo(() => pointsFor(view, run), [view, run]);
  const selectedTotal = selection.a.length + selection.b.length;
  const selectedImagesA = useMemo(
    () => selection.a.length ? selection.a : (run?.records_a.slice(0, 12).map((record) => record.index) || []),
    [run, selection.a],
  );
  const selectedImagesB = useMemo(
    () => selection.b.length ? selection.b : (run?.records_b.slice(0, 12).map((record) => record.index) || []),
    [run, selection.b],
  );
  const embeddingPoints = useMemo<EmbeddingPoint[]>(() => displayedPoints.map((point, index) => ({
    index,
    filename: point.filename,
    split: point.group.toUpperCase(),
    label: point.group.toUpperCase(),
    x: point.x,
    y: point.y,
    z: point.z,
  })), [displayedPoints]);
  const selected3DIndex = selection.a.length === 1 && selection.b.length === 0
    ? selection.a[0]
    : selection.b.length === 1 && selection.a.length === 0
      ? (run?.records_a.length || 0) + selection.b[0]
      : null;

  function resetResult() {
    setJob(null);
    setRunId("");
    setRun(null);
    setView(null);
    setSelection(emptySelection());
    setError("");
    setProjectionError("");
  }

  function addPaths(group: GroupKey, paths: string[]) {
    const cleaned = paths.map((path) => path.trim()).filter(Boolean);
    if (!cleaned.length) return;
    const setter = group === "a" ? setFoldersA : setFoldersB;
    setter((current) => {
      const next = [...current];
      for (const path of cleaned) if (!next.includes(path)) next.push(path);
      return next;
    });
    if (group === "a") setDraftA(""); else setDraftB("");
    resetResult();
    setNotice("");
  }

  function removeFolder(group: GroupKey, index: number) {
    const setter = group === "a" ? setFoldersA : setFoldersB;
    setter((current) => current.filter((_, item) => item !== index));
    resetResult();
  }

  async function requestProjection(runKey: string, nextMethod: ProjectionMethod, nextDimension: 2 | 3) {
    setProjecting(true);
    setProjectionError("");
    setView(null);
    setSelection(emptySelection());
    try {
      const result = await api.compareProjection(runKey, nextMethod, nextDimension);
      setView(result);
    } catch (reason) {
      setProjectionError(reason instanceof Error ? reason.message : "This projection is unavailable in the local runtime.");
    } finally {
      setProjecting(false);
    }
  }

  async function runComparison() {
    const effectiveA = [...foldersA, ...(draftA.trim() && !foldersA.includes(draftA.trim()) ? [draftA.trim()] : [])];
    const effectiveB = [...foldersB, ...(draftB.trim() && !foldersB.includes(draftB.trim()) ? [draftB.trim()] : [])];
    if (!effectiveA.length || !effectiveB.length) {
      setError("Add at least one folder to both Folder A and Folder B.");
      return;
    }
    if (!selectedModel) {
      setError("Install a supported LV feature model to run a comparison.");
      return;
    }
    setBusy(true);
    setError("");
    setProjectionError("");
    setNotice("");
    setRun(null);
    setRunId("");
    setView(null);
    setSelection(emptySelection());
    setJob(null);
    setMethod("sumap");
    setDimension(2);
    try {
      const created = await api.createCompareJob(effectiveA, effectiveB, selectedModel);
      let status: JobStatus;
      for (;;) {
        status = await api.compareJob(created.job_id);
        setJob(status);
        if (["completed", "failed", "cancelled"].includes(status.state)) break;
        await pause(350);
      }
      if (status.state === "failed") {
        throw new Error(status.error?.message || "The comparison failed. Check the local worker and retry.");
      }
      if (status.state === "cancelled" || !status.run_id) {
        setNotice("Comparison cancelled. Adjust the folders and run it again.");
        return;
      }
      setRunId(status.run_id);
      const result = await api.compareRun(status.run_id);
      setRun(result);
      await requestProjection(status.run_id, "sumap", 2);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The comparison could not be completed.");
    } finally {
      setBusy(false);
    }
  }

  async function cancelComparison() {
    if (!job?.job_id || !job.can_cancel) return;
    try {
      await api.cancelCompare(job.job_id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The cancellation request failed.");
    }
  }

  async function changeProjection(nextMethod: ProjectionMethod) {
    setMethod(nextMethod);
    if (runId) await requestProjection(runId, nextMethod, dimension);
  }

  async function changeDimension(nextDimension: 2 | 3) {
    setDimension(nextDimension);
    if (runId) await requestProjection(runId, method, nextDimension);
  }

  function selectPoint(group: GroupKey, index: number) {
    const next = emptySelection();
    next[group] = [index];
    setSelection(next);
  }

  function select3DPoint(globalIndex: number) {
    const point = displayedPoints[globalIndex];
    if (point) selectPoint(point.group, point.index);
  }

  function recordFor(group: GroupKey, index: number): CompareRecord | undefined {
    return (group === "a" ? run?.records_a : run?.records_b)?.[index];
  }

  function imageGallery(group: GroupKey, indices: number[]) {
    const name = group === "a" ? run?.folder_a : run?.folder_b;
    return (
      <section className={`compare-gallery compare-gallery-${group}`} aria-label={`Folder ${group.toUpperCase()} image gallery`}>
        <div className="compare-gallery-heading">
          <strong>Folder {group.toUpperCase()}</strong>
          <span>{name} · {indices.length} {selection[group].length ? "selected" : "preview"}</span>
        </div>
        <div className="compare-gallery-grid">
          {indices.slice(0, 60).map((index) => {
            const record = recordFor(group, index);
            if (!record || !runId) return null;
            return (
              <figure key={index}>
                <img
                  data-testid={`compare-image-${group}-${index}`}
                  src={api.compareImageUrl(runId, group, index)}
                  alt={record.filename}
                  loading="lazy"
                />
                <figcaption title={record.filename}>{record.filename}</figcaption>
              </figure>
            );
          })}
        </div>
      </section>
    );
  }

  const projectionLabel = methodOptions.find((option) => option.value === method)?.label || method;
  return (
    <div className="compare-workspace">
      <aside className="compare-controls">
        <div className="eyebrow">DISTRIBUTION DIAGNOSTICS</div>
        <h1>Compare Distributions</h1>
        <p className="compare-intro">Compare two image groups with the same local feature model.</p>
        <FolderGroup group="a" folders={foldersA} draft={draftA} setDraft={setDraftA}
          addPaths={(paths) => addPaths("a", paths)} removeFolder={(index) => removeFolder("a", index)} disabled={busy} onNotice={setNotice} />
        <FolderGroup group="b" folders={foldersB} draft={draftB} setDraft={setDraftB}
          addPaths={(paths) => addPaths("b", paths)} removeFolder={(index) => removeFolder("b", index)} disabled={busy} onNotice={setNotice} />

        <section className="control-section">
          <div className="section-heading"><span>Feature extraction</span><strong>Model</strong></div>
          <select aria-label="Compare model" className="compare-select" value={selectedModel}
            onChange={(event) => { setModelId(event.target.value); resetResult(); }} disabled={busy || !models.length}>
            {!models.length && <option value="">No supported model installed</option>}
            {models.map((model) => <option key={model.model_id} value={model.model_id}>{model.name}</option>)}
          </select>
          {!models.length && <p className="control-hint">Install a local LV feature model before running this tool.</p>}
        </section>
        <div className="compare-scope-note">
          <strong>Whole-image mode</strong>
          <span>This React case measures one A/B centroid drift score. Object-by-class comparison is tracked separately.</span>
        </div>
        <button className="run-button compare-run-button" type="button" onClick={runComparison}
          disabled={busy || projecting || !connected || !models.length}>
          {busy ? "Comparing…" : "▶ Run comparison"}
        </button>
        {busy && job?.can_cancel && <button type="button" className="cancel-button" onClick={cancelComparison}>Cancel comparison</button>}
        {busy && (
          <div className="progress-card" data-testid="compare-job-status" role="status">
            <div className="progress-row"><span>{job?.message || "Preparing comparison"}</span><strong>{Math.round(job?.percent || 0)}%</strong></div>
            <div className="progress-track"><div className="progress-fill" style={{ width: Math.max(3, job?.percent || 0) + "%" }} /></div>
            <div className="progress-foot">{job?.done || 0} / {job?.total || "—"} image embeddings</div>
          </div>
        )}
        {error && <div className="error-banner" role="alert">{error}</div>}
        {notice && <div className="notice-banner" role="status">{notice}</div>}
        {!connected && <p className="control-hint">Waiting for the local worker connection.</p>}
      </aside>

      <main className="compare-results">
        {run ? (
          <>
            <div className="results-header">
              <div>
                <div className="eyebrow">WHOLE-IMAGE COMPARISON</div>
                <h2>{run.folder_a} <span className="versus">vs</span> {run.folder_b}</h2>
                <p>Centroid cosine drift: larger values mean the two groups differ more in the selected feature space.</p>
              </div>
            </div>
            <section className="drift-card" aria-label="Overall image drift">
              <div><span>OVERALL DRIFT</span><strong data-testid="compare-drift">{run.drift.toFixed(4)}</strong><small>normalized centroid cosine distance</small></div>
              <div className="drift-count"><span>Folder A</span><strong>{run.records_a.length}</strong><small>readable images</small></div>
              <div className="drift-count"><span>Folder B</span><strong>{run.records_b.length}</strong><small>readable images</small></div>
              <div className="drift-model"><span>FEATURE MODEL</span><strong>{run.model_id}</strong></div>
            </section>
            {run.warnings.length > 0 && <div className="compare-warnings" role="status">{run.warnings.map((warning, index) => <p key={index}>{warning}</p>)}</div>}
            <section className="compare-card">
              <div className="compare-card-head">
                <div><strong>Joint image projection</strong><small>Projection distance is for visual exploration; use the drift score above for the numeric comparison.</small></div>
                <div className="compare-projection-controls">
                  <label><span>Method</span><select aria-label="Compare projection method" value={method}
                    onChange={(event) => void changeProjection(event.target.value as ProjectionMethod)} disabled={projecting}>
                    {methodOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                  </select></label>
                  <div className="dimension-switch" role="group" aria-label="Compare projection dimension">
                    <button type="button" aria-pressed={dimension === 2} onClick={() => void changeDimension(2)} disabled={projecting}>2D</button>
                    <button type="button" aria-pressed={dimension === 3} onClick={() => void changeDimension(3)} disabled={projecting}>3D</button>
                  </div>
                </div>
              </div>
              {projecting && <div className="plot-loading" role="status">Projecting image features…</div>}
              {!projecting && projectionError && <div className="projection-error" role="alert">{projectionError} Select another projection method, such as PCA.</div>}
              {!projecting && view && displayedPoints.length > 0 && (
                dimension === 2 ? (
                  <ComparePlot2D points={displayedPoints} selected={selection} onSelect={selectPoint} onBoxSelect={setSelection} />
                ) : (
                  <Suspense fallback={<div className="plot-frame plot-loading" role="status">Loading 3D plot…</div>}>
                    <LazyScatterPlot3D points={embeddingPoints} selectedIndex={selected3DIndex}
                      onSelect={select3DPoint} viewKey={`${runId}-${method}-${dimension}`} />
                  </Suspense>
                )
              )}
              <div className="compare-selection-summary">
                <span>{selectedTotal ? `Selected ${selection.a.length} from A · ${selection.b.length} from B` : "No selection · showing the first 12 images from each group"}</span>
                {selectedTotal > 0 && <button type="button" className="clear-selection" onClick={() => setSelection(emptySelection())}>Clear selection</button>}
              </div>
            </section>
            <div className="compare-galleries">
              {imageGallery("a", selectedImagesA)}
              {imageGallery("b", selectedImagesB)}
            </div>
            <footer className="canvas-footer"><span>VisualLatent · React Web</span><span>{projectionLabel} · {dimension}D</span></footer>
          </>
        ) : (
          <div className="compare-empty">
            <div className="compare-empty-mark" aria-hidden="true"><i /><i /><i /><i /></div>
            <strong>Compare two image groups</strong>
            <span>Add one or more folders to each side, choose a feature model, then run the local analysis.</span>
            <small>Images are embedded locally. The comparison score is computed from the A/B embedding centroids.</small>
          </div>
        )}
      </main>
    </div>
  );
}
