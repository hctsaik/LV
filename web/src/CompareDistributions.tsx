import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type ComparisonPoint,
  type ComparisonObjectPolicy,
  type ComparisonProjection,
  type ComparisonRecord,
  type ComparisonRun,
  type DriftRow,
  type JobStatus,
  type ModelOption,
  type ProjectionMethod,
} from "./api";

const LazyCompareScatter3D = lazy(() => import("./CompareScatter3D"));
const policyResolutions = [168, 196, 224, 280, 336, 392, 448] as const;
const defaultObjectPolicy: ComparisonObjectPolicy = { pad: 0.12, target_res: 224, head: "cls" };
const objectSizeBuckets = ["<32", "32-96", "96-224", ">224"] as const;
const methods: Array<{ value: ProjectionMethod; label: string }> = [
  { value: "pca", label: "PCA" },
  { value: "tsne", label: "t-SNE" },
  { value: "umap", label: "UMAP" },
  { value: "lda", label: "LDA (supervised)" },
  { value: "sumap", label: "Supervised UMAP" },
];
type Side = "a" | "b";
type PointRef = { group: "A" | "B"; index: number };
const width = 860;
const height = 480;
const padding = 42;

function pointKey(point: PointRef) {
  return point.group + ":" + point.index;
}

function chartCoordinates(points: ComparisonPoint[]) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const xMin = Math.min(...xs, 0);
  const xMax = Math.max(...xs, 0);
  const yMin = Math.min(...ys, 0);
  const yMax = Math.max(...ys, 0);
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1;
  return new Map(points.map((point) => [pointKey(point), {
    x: padding + ((point.x - xMin) / xSpan) * (width - padding * 2),
    y: height - padding - ((point.y - yMin) / ySpan) * (height - padding * 2),
  }]));
}

function CompareScatter2D({
  points,
  selected,
  onSelect,
  onSelectMany,
}: {
  points: ComparisonPoint[];
  selected: string[];
  onSelect: (point: ComparisonPoint, additive: boolean) => void;
  onSelectMany: (points: ComparisonPoint[], additive: boolean) => void;
}) {
  const coordinates = useMemo(() => chartCoordinates(points), [points]);
  const origin = useRef<{
    x: number;
    y: number;
    startX: number;
    startY: number;
    additive: boolean;
    point: PointRef | null;
  } | null>(null);
  const dragged = useRef(false);
  const [box, setBox] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  function toViewBox(svg: SVGSVGElement, clientX: number, clientY: number) {
    const rect = svg.getBoundingClientRect();
    return {
      x: ((clientX - rect.left) / rect.width) * width,
      y: ((clientY - rect.top) / rect.height) * height,
    };
  }

  function onPointerDown(event: React.PointerEvent<SVGSVGElement>) {
    if (event.button !== 0) return;
    const p = toViewBox(event.currentTarget, event.clientX, event.clientY);
    const target = (event.target as Element).closest<SVGGElement>("[data-testid='comparison-point']");
    const group = target?.dataset.group;
    const index = Number(target?.dataset.index);
    const point = (group === "A" || group === "B") && Number.isInteger(index)
      ? { group, index } as PointRef
      : null;
    origin.current = {
      ...p,
      startX: event.clientX,
      startY: event.clientY,
      additive: event.ctrlKey || event.metaKey,
      point,
    };
    dragged.current = false;
  }

  function onPointerMove(event: React.PointerEvent<SVGSVGElement>) {
    if (!origin.current) return;
    if (Math.hypot(event.clientX - origin.current.startX, event.clientY - origin.current.startY) > 7) {
      dragged.current = true;
      try {
        event.currentTarget.setPointerCapture(event.pointerId);
      } catch {
        // The pointer may have left the document before the drag threshold.
      }
    }
    if (!dragged.current) return;
    const p = toViewBox(event.currentTarget, event.clientX, event.clientY);
    setBox({
      x: Math.min(origin.current.x, p.x),
      y: Math.min(origin.current.y, p.y),
      w: Math.abs(p.x - origin.current.x),
      h: Math.abs(p.y - origin.current.y),
    });
  }

  function onPointerUp(event: React.PointerEvent<SVGSVGElement>) {
    if (!origin.current) return;
    const start = origin.current;
    if (dragged.current) {
      const p = toViewBox(event.currentTarget, event.clientX, event.clientY);
      const selection = {
        x: Math.min(origin.current.x, p.x),
        y: Math.min(origin.current.y, p.y),
        w: Math.abs(p.x - origin.current.x),
        h: Math.abs(p.y - origin.current.y),
      };
      const picked = points.filter((point) => {
        const at = coordinates.get(pointKey(point));
        return at && at.x >= selection.x && at.x <= selection.x + selection.w
          && at.y >= selection.y && at.y <= selection.y + selection.h;
      });
      if (picked.length) onSelectMany(picked, start.additive);
    } else if (start.point) {
      const point = points.find((candidate) => candidate.group === start.point?.group && candidate.index === start.point.index);
      if (point) onSelect(point, start.additive);
    }
    origin.current = null;
    setBox(null);
    window.setTimeout(() => { dragged.current = false; }, 0);
  }

  function onPointerCancel() {
    origin.current = null;
    dragged.current = false;
    setBox(null);
  }

  return (
    <div className="plot-frame" data-testid="comparison-plot-2d">
      <svg
        className="scatter compare-scatter"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Folder A and Folder B embedding comparison scatter plot"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerCancel}
      >
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="axis" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="axis" />
        <text x={width / 2} y={height - 8} className="axis-label">Component 1</text>
        <text x={15} y={height / 2} className="axis-label vertical-label">Component 2</text>
        {points.map((point) => {
          const at = coordinates.get(pointKey(point));
          if (!at) return null;
          const active = selected.includes(pointKey(point));
          const color = point.group === "A" ? "#3498db" : "#e74c3c";
          return (
            <g
              key={pointKey(point)}
              role="button"
              tabIndex={0}
              aria-label={`Select ${point.filename} from group ${point.group}`}
              data-testid="comparison-point"
              data-group={point.group}
              data-index={point.index}
              className={"point " + (active ? "point-active" : "")}
              transform={`translate(${at.x} ${at.y})`}
              onClick={(event) => {
                event.stopPropagation();
                if (!dragged.current) onSelect(point, event.ctrlKey || event.metaKey);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") onSelect(point, event.ctrlKey || event.metaKey);
              }}
            >
              <title>{`${point.filename} · ${point.group} · ${point.label}`}</title>
              {active && <circle r={12} className="point-selection-ring" />}
              {point.group === "A" ? <circle r={7} fill={color} /> : <rect x={-6} y={-6} width={12} height={12} rx={2} fill={color} />}
            </g>
          );
        })}
        {box && <rect x={box.x} y={box.y} width={box.w} height={box.h} className="selection-box" />}
      </svg>
      <div className="chart-hint">Drag to select a region · Ctrl/⌘-click to add points</div>
    </div>
  );
}

function FolderList({
  side,
  folders,
  draft,
  setDraft,
  add,
  remove,
  choose,
}: {
  side: Side;
  folders: string[];
  draft: string;
  setDraft: (value: string) => void;
  add: () => void;
  remove: (path: string) => void;
  choose: () => void;
}) {
  const title = side === "a" ? "Folder A" : "Folder B";
  return (
    <section className="compare-folder-section">
      <div className="section-heading"><span className="step-number">{side === "a" ? "01" : "02"}</span><h2>{title}</h2></div>
      <button className="host-picker" type="button" onClick={choose}>
        <span className="folder-icon" aria-hidden="true">▱</span>Choose {title} from device<span className="picker-hint">Native</span>
      </button>
      <label className="field-label" htmlFor={`compare-${side}-folder`}>Image folder</label>
      <div className="folder-input-row">
        <input
          id={`compare-${side}-folder`}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && add()}
          placeholder="Paste a local folder path"
          autoComplete="off"
        />
        <button type="button" className="add-folder" onClick={add} aria-label={`Add ${title}`}>+</button>
      </div>
      {folders.length > 0 && (
        <ul className="folder-list" aria-label={`${title} folders`}>
          {folders.map((folder) => (
            <li key={folder} title={folder}>
              <span className="folder-chip-dot" />
              <span className="folder-chip-name">{folder.split(/[\\/]/).filter(Boolean).at(-1) || folder}</span>
              <button type="button" aria-label={`Remove ${folder}`} onClick={() => remove(folder)}>×</button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function selectedRecord(
  point: PointRef | null,
  run: ComparisonRun,
): { record: ComparisonRecord; side: Side } | null {
  if (!point) return null;
  const side: Side = point.group === "A" ? "a" : "b";
  const record = (side === "a" ? run.records_a : run.records_b)[point.index];
  return record ? { record, side } : null;
}

function uniqueRecords(points: ComparisonPoint[], run: ComparisonRun, side: Side) {
  const group = side === "a" ? "A" : "B";
  const relevant = points.filter((point) => point.group === group);
  const source = side === "a" ? run.records_a : run.records_b;
  const seen = new Set<string>();
  return relevant.flatMap((point) => {
    const record = source[point.index];
    if (!record || seen.has(record.source_id)) return [];
    seen.add(record.source_id);
    return [{ point, record }];
  }).slice(0, 60);
}

function ImageCard({
  runId,
  side,
  record,
  point,
  onSelect,
}: {
  runId: string;
  side: Side;
  record: ComparisonRecord;
  point: ComparisonPoint;
  onSelect: () => void;
}) {
  const bbox = record.bbox;
  return (
    <button className="compare-image-card" type="button" onClick={onSelect} title={record.filename}>
      <span className="compare-image-canvas">
        <img src={api.comparisonImageUrl(runId, side, record.index)} alt={record.filename} loading="lazy" />
        {bbox && <i style={{ left: `${(bbox[0] - bbox[2] / 2) * 100}%`, top: `${(bbox[1] - bbox[3] / 2) * 100}%`, width: `${bbox[2] * 100}%`, height: `${bbox[3] * 100}%` }} />}
      </span>
      <strong>{record.filename}</strong>
      <small>{record.label}{record.score === null ? "" : ` · conf ${record.score.toFixed(2)}`}</small>
      <span className="sr-only">At {point.x.toFixed(2)}, {point.y.toFixed(2)}</span>
    </button>
  );
}

export default function CompareDistributions({
  models,
  loadingModels,
  connected,
}: {
  models: ModelOption[];
  loadingModels: boolean;
  connected: boolean;
}) {
  const [foldersA, setFoldersA] = useState<string[]>([]);
  const [foldersB, setFoldersB] = useState<string[]>([]);
  const [draftA, setDraftA] = useState("");
  const [draftB, setDraftB] = useState("");
  const [modelId, setModelId] = useState("");
  const [unit, setUnit] = useState<"object" | "whole">("object");
  const [classNamesText, setClassNamesText] = useState("");
  const [policyDraft, setPolicyDraft] = useState<ComparisonObjectPolicy>(defaultObjectPolicy);
  const [appliedObjectPolicy, setAppliedObjectPolicy] = useState<ComparisonObjectPolicy | null>(null);
  const [busy, setBusy] = useState(false);
  const [loadingProjection, setLoadingProjection] = useState(false);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [runId, setRunId] = useState("");
  const [run, setRun] = useState<ComparisonRun | null>(null);
  const [projection, setProjection] = useState<ComparisonProjection | null>(null);
  const [selectedLabel, setSelectedLabel] = useState("");
  const [method, setMethod] = useState<ProjectionMethod>("sumap");
  const [dimension, setDimension] = useState<2 | 3>(2);
  const [minConfidence, setMinConfidence] = useState(0);
  const [maxConfidence, setMaxConfidence] = useState(1);
  const [minShortPx, setMinShortPx] = useState(0);
  const [selectedPoints, setSelectedPoints] = useState<PointRef[]>([]);
  const [focus, setFocus] = useState<PointRef | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  function invalidateResult() {
    setRun(null);
    setRunId("");
    setProjection(null);
    setSelectedLabel("");
    setSelectedPoints([]);
    setFocus(null);
    setJob(null);
    setNotice("");
    setError("");
  }

  function applyManualObjectPolicy() {
    const policy = { ...policyDraft, pad: Number(policyDraft.pad.toFixed(2)) };
    setAppliedObjectPolicy(policy);
    invalidateResult();
    setNotice("Manual object settings applied. Run the comparison again to use them.");
  }

  function restoreAutomaticObjectPolicy() {
    setAppliedObjectPolicy(null);
    invalidateResult();
    setNotice("Automatic object tuning restored. Run the comparison again to retune.");
  }

  useEffect(() => {
    if (!modelId && models.length) {
      const preferred = models.find((model) => model.model_id === "dinov2_vits14") || models[0];
      setModelId(preferred.model_id);
    }
    if (modelId && !models.some((model) => model.model_id === modelId) && models.length) {
      setModelId(models[0].model_id);
    }
  }, [models, modelId]);

  const rows = run?.rows ?? [];
  const availableConfidence = Boolean(run?.unit === "object" && [...(run.records_a ?? []), ...(run.records_b ?? [])].some((record) => record.score !== null));

  useEffect(() => {
    if (!run || !runId || !selectedLabel) {
      setProjection(null);
      return;
    }
    let active = true;
    setLoadingProjection(true);
    setError("");
    api.projectComparison(runId, selectedLabel, method, dimension, {
      min_confidence: minConfidence,
      max_confidence: maxConfidence,
      min_short_px: minShortPx,
    }).then((value) => {
      if (!active) return;
      setProjection(value);
      setNotice(value.warnings.join(" "));
    }).catch((reason: unknown) => {
      if (active) {
        setProjection(null);
        setError(reason instanceof Error ? reason.message : "The selected comparison view could not be projected.");
      }
    }).finally(() => active && setLoadingProjection(false));
    return () => { active = false; };
  }, [run, runId, selectedLabel, method, dimension, minConfidence, maxConfidence, minShortPx]);

  function addFolder(side: Side) {
    const draft = side === "a" ? draftA : draftB;
    const path = draft.trim();
    if (!path) return;
    invalidateResult();
    if (side === "a") {
      setFoldersA((current) => current.includes(path) ? current : [...current, path]);
      setDraftA("");
    } else {
      setFoldersB((current) => current.includes(path) ? current : [...current, path]);
      setDraftB("");
    }
  }

  function addChosenFolder(side: Side, path: string) {
    if (side === "a") setFoldersA((current) => current.includes(path) ? current : [...current, path]);
    else setFoldersB((current) => current.includes(path) ? current : [...current, path]);
    invalidateResult();
  }

  async function chooseFolder(side: Side) {
    setError("");
    if (!window.cimHost?.chooseFile) {
      setNotice("Native folder picker is not connected. Enter a local folder path below.");
      return;
    }
    try {
      const result = await window.cimHost.chooseFile();
      if (result.canceled) return;
      const chosen = result.paths?.length ? result.paths : result.path ? [result.path] : [];
      if (!chosen.length) throw new Error("The host did not return a folder path.");
      chosen.forEach((path) => addChosenFolder(side, path));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The folder picker failed.");
    }
  }

  async function runComparison() {
    const selectedA = [...foldersA];
    const selectedB = [...foldersB];
    if (draftA.trim() && !selectedA.includes(draftA.trim())) selectedA.push(draftA.trim());
    if (draftB.trim() && !selectedB.includes(draftB.trim())) selectedB.push(draftB.trim());
    if (!selectedA.length || !selectedB.length) {
      setError("Add at least one folder to both Folder A and Folder B.");
      return;
    }
    if (!modelId) {
      setError("Install a supported LV model to enable comparison.");
      return;
    }
    setError("");
    setNotice("");
    setRun(null);
    setRunId("");
    setProjection(null);
    setSelectedPoints([]);
    setFocus(null);
    setBusy(true);
    setJob(null);
    try {
      const created = await api.createComparisonJob({
        folders_a: selectedA,
        folders_b: selectedB,
        model_id: modelId,
        unit,
        class_names: classNamesText.split(/[\n,，]/).map((name) => name.trim()).filter(Boolean),
        ...(unit === "object" && appliedObjectPolicy ? { object_policy: appliedObjectPolicy } : {}),
      });
      setJob({
        job_id: created.job_id, state: "queued", phase: "queued", done: 0, total: 0,
        percent: 0, message: "Waiting to start", run_id: null, error: null, can_cancel: true,
      });
      let latest: JobStatus | null = null;
      for (let attempt = 0; attempt < 18_000; attempt += 1) {
        latest = await api.comparisonJob(created.job_id);
        setJob(latest);
        if (["completed", "failed", "cancelled"].includes(latest.state)) break;
        await new Promise((resolve) => window.setTimeout(resolve, 350));
      }
      if (!latest || !["completed", "failed", "cancelled"].includes(latest.state)) {
        throw new Error("The comparison is taking too long. Cancel it and retry with a smaller dataset.");
      }
      if (latest.state === "failed") throw new Error(latest.error?.message || "The comparison failed.");
      if (latest.state === "cancelled") {
        setNotice("Comparison cancelled. Update the folders or settings and try again.");
        return;
      }
      if (!latest.run_id) throw new Error("The worker completed without returning a comparison result.");
      const value = await api.comparison(latest.run_id);
      setRun(value);
      setRunId(latest.run_id);
      if (value.policy) setPolicyDraft(value.policy);
      setSelectedLabel(value.rows[0]?.label ?? "");
      setMethod("sumap");
      setDimension(2);
      setMinConfidence(0);
      setMaxConfidence(1);
      setMinShortPx(0);
      setNotice(value.warnings.join(" "));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The comparison failed.");
    } finally {
      setBusy(false);
    }
  }

  async function cancelComparison() {
    if (!job?.job_id) return;
    try {
      await api.cancelComparison(job.job_id);
      setJob((current) => current ? { ...current, state: "cancelling", message: "Cancellation requested" } : current);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not cancel the comparison.");
    }
  }

  function selectPoint(point: ComparisonPoint, additive: boolean) {
    const ref = { group: point.group, index: point.index };
    setFocus(ref);
    setSelectedPoints((current) => {
      if (!additive) return [ref];
      const key = pointKey(ref);
      return current.some((item) => pointKey(item) === key)
        ? current.filter((item) => pointKey(item) !== key)
        : [...current, ref];
    });
  }

  function selectMany(points: ComparisonPoint[], additive: boolean) {
    const refs = points.map(({ group, index }) => ({ group, index }));
    setFocus(refs[0] ?? null);
    setSelectedPoints((current) => {
      if (!additive) return refs;
      const merged = new Map(current.map((item) => [pointKey(item), item]));
      refs.forEach((item) => merged.set(pointKey(item), item));
      return [...merged.values()];
    });
  }

  const focused = run ? selectedRecord(focus, run) : null;
  const pointRefs = useMemo(() => selectedPoints.map(pointKey), [selectedPoints]);
  const selectedClasses = [...new Set(selectedPoints.map((point) => point.group))].length;
  const cardsA = run && projection
    ? uniqueRecords(selectedPoints.length ? projection.points.filter((point) => pointRefs.includes(pointKey(point))) : projection.points, run, "a")
    : [];
  const cardsB = run && projection
    ? uniqueRecords(selectedPoints.length ? projection.points.filter((point) => pointRefs.includes(pointKey(point))) : projection.points, run, "b")
    : [];

  return (
    <div className="compare-workspace">
      <aside className="compare-controls">
        <div className="eyebrow">COMPARE DISTRIBUTIONS</div>
        <h1>Compare two image groups</h1>
        <p className="intro">Measure how groups A and B differ in the local feature space. Images and embeddings stay on this device.</p>
        <FolderList side="a" folders={foldersA} draft={draftA} setDraft={setDraftA} add={() => addFolder("a")} remove={(path) => { setFoldersA((current) => current.filter((item) => item !== path)); invalidateResult(); }} choose={() => void chooseFolder("a")} />
        <FolderList side="b" folders={foldersB} draft={draftB} setDraft={setDraftB} add={() => addFolder("b")} remove={(path) => { setFoldersB((current) => current.filter((item) => item !== path)); invalidateResult(); }} choose={() => void chooseFolder("b")} />

        <section className="compare-config">
          <div className="section-heading"><span className="step-number">03</span><h2>Comparison setup</h2></div>
          <label className="field-label" htmlFor="compare-model">Feature model</label>
          <select id="compare-model" value={modelId} onChange={(event) => { invalidateResult(); setModelId(event.target.value); }} disabled={loadingModels || busy}>
            {models.map((model) => <option key={model.model_id} value={model.model_id}>{model.name}</option>)}
          </select>
          {!loadingModels && models.length === 0 && <div className="model-empty">Install a supported LV model to enable comparison.</div>}
          <label className="field-label field-label-spaced" htmlFor="compare-unit">Comparison unit</label>
          <select id="compare-unit" value={unit} onChange={(event) => { invalidateResult(); setUnit(event.target.value as "object" | "whole"); }} disabled={busy}>
            <option value="object">Object by class (YOLO / supported boxes)</option>
            <option value="whole">Whole image (no labels required)</option>
          </select>
          {unit === "object" ? (
            <>
              <p className="field-note">The local worker measures crop settings on the smaller group and applies the best policy to both sides.</p>
              <fieldset className="object-policy-editor" data-testid="object-policy-editor" disabled={busy}>
                <legend>Object embedding settings</legend>
                <div className="object-policy-grid">
                  <label className="field-label" htmlFor="compare-policy-pad">Padding</label>
                  <input id="compare-policy-pad" aria-label="Object padding" type="number" min="0" max="0.5" step="0.02" value={policyDraft.pad} onChange={(event) => setPolicyDraft((current) => ({ ...current, pad: Math.max(0, Math.min(0.5, Number(event.target.value))) }))} />
                  <label className="field-label" htmlFor="compare-policy-resolution">Resolution</label>
                  <select id="compare-policy-resolution" aria-label="Object resolution" value={policyDraft.target_res} onChange={(event) => setPolicyDraft((current) => ({ ...current, target_res: Number(event.target.value) as ComparisonObjectPolicy["target_res"] }))}>
                    {policyResolutions.map((resolution) => <option key={resolution} value={resolution}>{resolution}</option>)}
                  </select>
                  <label className="field-label" htmlFor="compare-policy-head">Embedding head</label>
                  <select id="compare-policy-head" aria-label="Object embedding head" value={policyDraft.head} onChange={(event) => setPolicyDraft((current) => ({ ...current, head: event.target.value as ComparisonObjectPolicy["head"] }))}>
                    <option value="cls">cls</option>
                    <option value="meanpool">meanpool</option>
                  </select>
                </div>
                <div className="policy-actions">
                  <button type="button" className="secondary-action" onClick={applyManualObjectPolicy} disabled={busy}>Apply manual settings</button>
                  {appliedObjectPolicy && <button type="button" className="text-action" onClick={restoreAutomaticObjectPolicy}>Use automatic tuning</button>}
                </div>
                <p className="field-note">{appliedObjectPolicy ? `Manual policy active: pad ${appliedObjectPolicy.pad} · ${appliedObjectPolicy.target_res} · ${appliedObjectPolicy.head}.` : "Automatic tuning runs on the smaller group; apply a manual policy here when needed."}</p>
              </fieldset>
              <label className="field-label field-label-spaced" htmlFor="compare-classes">Class names override (optional)</label>
              <textarea id="compare-classes" rows={3} value={classNamesText} onChange={(event) => { invalidateResult(); setClassNamesText(event.target.value); }} placeholder="One class name per line" disabled={busy} />
              <p className="field-note">Leave blank to use the classes.txt or data.yaml found beside each dataset.</p>
            </>
          ) : <p className="field-note">Whole-image mode compares the cosine distance between A and B embedding centroids.</p>}
        </section>

        {(error || notice) && <div className={"alert " + (error ? "alert-error" : "alert-info")} role={error ? "alert" : "status"}>{error || notice}</div>}
        <button className="run-button" type="button" onClick={() => void runComparison()} disabled={busy || !connected || loadingModels || !modelId}>
          <span aria-hidden="true">{busy ? "◌" : "▶"}</span>{busy ? "Comparison running" : "Run comparison"}
        </button>
        {busy && <button className="cancel-button" type="button" onClick={() => void cancelComparison()}>Cancel comparison</button>}
        <div className="privacy-note"><span aria-hidden="true">⌁</span> Local processing · no dataset upload</div>
      </aside>

      <main className="compare-results">
        <div className="results-header">
          <div><div className="eyebrow">A / B COMPARISON</div><h2>{run ? `${run.name_a} vs ${run.name_b}` : "Your comparison workspace is ready"}</h2><p>{run ? `${run.records_a.length.toLocaleString()} A records · ${run.records_b.length.toLocaleString()} B records · ${run.model}` : "Choose one or more folders for each group, then run a local comparison."}</p></div>
          {run && <div className="run-badge"><span className="badge-dot" /> Result ready</div>}
        </div>
        {busy && <div className="progress-card" data-testid="compare-job-status" role="status"><div className="progress-row"><span>{job?.message || "Preparing comparison"}</span><strong>{Math.round(job?.percent || 0)}%</strong></div><div className="progress-track"><div className="progress-fill" style={{ width: Math.max(3, job?.percent || 0) + "%" }} /></div><div className="progress-foot">{job?.done || 0} / {job?.total || "—"} images or objects</div></div>}
        {!busy && job?.state === "completed" && <div className="completion-label" data-testid="compare-job-status">✓ Completed</div>}
        {!busy && job?.state === "cancelled" && <div className="completion-label" data-testid="compare-job-status">Cancelled</div>}

        {run ? (
          <>
            {run.a_only && Object.keys(run.a_only).length > 0 || run.b_only && Object.keys(run.b_only).length > 0 ? (
              <div className="alert alert-warning" role="status">
                Classes only in {run.name_a}: {Object.entries(run.a_only).map(([label, count]) => `${label} (${count})`).join(", ") || "—"}
                {" · "}only in {run.name_b}: {Object.entries(run.b_only).map(([label, count]) => `${label} (${count})`).join(", ") || "—"}
              </div>
            ) : null}
            {run.policy && <div className="policy-summary" data-testid="compare-policy">{run.policy_source === "manual" ? "Manual object policy" : "Automatic object policy"}: pad {run.policy.pad} · resolution {run.policy.target_res} · {run.policy.head} head</div>}
            {run.unit === "object" && run.autotune_rows.length > 0 && <details className="autotune-details" data-testid="autotune-results"><summary>Automatic tuning candidates ({run.autotune_rows.length})</summary><div className="drift-table-wrap"><table className="drift-table"><thead><tr><th>Policy</th><th>Overall</th>{objectSizeBuckets.map((bucket) => <th key={bucket}>{bucket}px</th>)}</tr></thead><tbody>{run.autotune_rows.map((candidate) => <tr key={`${candidate.policy.pad}-${candidate.policy.target_res}-${candidate.policy.head}`}><td>pad{Math.round(candidate.policy.pad * 100)}_r{candidate.policy.target_res}_{candidate.policy.head}</td><td>{candidate.macro_purity === null ? "—" : `${(candidate.macro_purity * 100).toFixed(1)}%`}</td>{objectSizeBuckets.map((bucket) => { const score = candidate.buckets[bucket]?.purity; return <td key={bucket}>{score === null || score === undefined ? "—" : `${(score * 100).toFixed(1)}%`}</td>; })}</tr>)}</tbody></table></div></details>}
            <section className="drift-card">
              <div className="section-heading"><span className="step-number">04</span><h2>{run.unit === "whole" ? "Overall image drift" : "Drift by shared class"}</h2></div>
              <p className="field-note">Centroid cosine distance. Higher means the selected groups differ more; the projection below is for visual inspection only.</p>
              <div className="drift-table-wrap"><table className="drift-table"><thead><tr><th>{unit === "whole" ? "Comparison" : "Class"}</th><th>{run.name_a}</th><th>{run.name_b}</th><th>Drift ↑</th></tr></thead><tbody>
                {rows.map((row: DriftRow) => <tr key={row.label} className={selectedLabel === row.label ? "selected-row" : ""}>
                  <td><button type="button" className="class-select" onClick={() => { setSelectedLabel(row.label); setSelectedPoints([]); setFocus(null); }}>{row.label}</button></td>
                  <td>{row.a_count}</td><td>{row.b_count}</td><td><strong>{row.drift.toFixed(4)}</strong><span className="drift-track"><i style={{ width: `${Math.min(100, (row.drift / Math.max(0.05, ...rows.map((item) => item.drift))) * 100)}%` }} /></span></td>
                </tr>)}
              </tbody></table></div>
            </section>

            <section className="comparison-plot-card">
              <div className="plot-toolbar">
                <div className="plot-title"><span className="chart-glyph" aria-hidden="true">⊙</span><div><strong>A vs B embedding projection</strong><small>Click points or drag a rectangle to inspect corresponding records</small></div></div>
                <div className="filters">
                  <label><span>Class</span><select aria-label="Comparison class" value={selectedLabel} onChange={(event) => { setSelectedLabel(event.target.value); setSelectedPoints([]); setFocus(null); }}>{rows.map((row) => <option key={row.label}>{row.label}</option>)}</select></label>
                  <label><span>Method</span><select aria-label="Comparison projection" value={method} onChange={(event) => { setMethod(event.target.value as ProjectionMethod); setSelectedPoints([]); setFocus(null); }}>{methods.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
                </div>
              </div>
              <div className="plot-subtoolbar">
                <div className="dimension-switch" role="group" aria-label="Comparison projection dimension"><button type="button" aria-pressed={dimension === 2} onClick={() => { setDimension(2); setSelectedPoints([]); setFocus(null); }}>2D</button><button type="button" aria-pressed={dimension === 3} onClick={() => { setDimension(3); setSelectedPoints([]); setFocus(null); }}>3D</button></div>
                {availableConfidence && <div className="object-filters"><label>Confidence {minConfidence.toFixed(2)}–{maxConfidence.toFixed(2)}<div><input aria-label="Minimum confidence" type="range" min="0" max="0.95" step="0.05" value={minConfidence} onChange={(event) => { setMinConfidence(Math.min(Number(event.target.value), maxConfidence)); setSelectedPoints([]); setFocus(null); }} /><input aria-label="Maximum confidence" type="range" min="0.05" max="1" step="0.05" value={maxConfidence} onChange={(event) => { setMaxConfidence(Math.max(Number(event.target.value), minConfidence)); setSelectedPoints([]); setFocus(null); }} /></div></label><label>Hide objects with source short edge below {minShortPx}px<input aria-label="Minimum object short edge" type="range" min="0" max="64" step="4" value={minShortPx} onChange={(event) => { setMinShortPx(Number(event.target.value)); setSelectedPoints([]); setFocus(null); }} /></label></div>}
                {selectedPoints.length > 0 && <button type="button" className="clear-selection" onClick={() => { setSelectedPoints([]); setFocus(null); }}>Clear selection ({selectedPoints.length})</button>}
              </div>
              {loadingProjection ? <div className="plot-frame plot-loading" role="status">Computing {method.toUpperCase()} projection…</div>
                : projection?.points.length ? dimension === 3
                  ? <Suspense fallback={<div className="plot-frame plot-loading" role="status">Loading 3D plot…</div>}><LazyCompareScatter3D points={projection.points} selected={pointRefs} onSelect={(point) => selectPoint(point, false)} viewKey={`${runId}-${selectedLabel}-${projection.method}-${dimension}-${minConfidence}-${maxConfidence}-${minShortPx}`} /></Suspense>
                  : <CompareScatter2D points={projection.points} selected={pointRefs} onSelect={selectPoint} onSelectMany={selectMany} />
                : <div className="plot-frame plot-loading">No points match the selected class and filters.</div>}
              <div className="compare-legend"><span><i className="legend-a" />{run.name_a}</span><span><i className="legend-b" />{run.name_b}</span><small>Centroid drift comes from the original embeddings; projected distance is not a metric.</small></div>
            </section>

            <section className="compare-galleries">
              <div className="gallery-heading"><h2>{selectedPoints.length ? `Selected records (${selectedPoints.length})` : "Sample records"}</h2><span>{selectedClasses > 1 ? "A and B" : selectedPoints[0]?.group === "A" ? "Folder A" : selectedPoints[0]?.group === "B" ? "Folder B" : "Both groups"}</span></div>
              <div className="compare-gallery-columns">
                {(["a", "b"] as const).map((side) => {
                  const cards = side === "a" ? cardsA : cardsB;
                  const name = side === "a" ? run.name_a : run.name_b;
                  const group = side === "a" ? "A" : "B";
                  return <div className={`compare-gallery ${side}`} key={side}><h3>{name} <small>{cards.length} images</small></h3><div className="compare-gallery-grid">{cards.map(({ point, record }) => <ImageCard key={record.source_id} runId={runId} side={side} record={record} point={point} onSelect={() => selectPoint(point, false)} />)}</div>{!cards.length && <p className="field-note">No selected images from this group.</p>}</div>;
                })}
              </div>
            </section>
            {focused && <aside className="compare-focus" data-testid="comparison-focused-record"><strong>{focused.record.filename}</strong><span>Group {focused.side.toUpperCase()} · {focused.record.label}</span>{focused.record.bbox && <span>Box {focused.record.bbox.map((value) => value.toFixed(3)).join(", ")}</span>}</aside>}
          </>
        ) : (
          <div className="empty-canvas compare-empty"><div className="empty-illustration" aria-hidden="true"><span className="orbit orbit-one" /><span className="orbit orbit-two" /><i className="dot dot-one" /><i className="dot dot-two" /><i className="dot dot-three" /><i className="dot dot-four" /><b>A/B</b></div><h3>Measure two groups locally</h3><p>Compare by class for labeled detection sets or choose whole-image mode for ordinary image folders.</p><div className="empty-facts"><span><b>01</b> Choose A and B</span><span><b>02</b> Select a unit</span><span><b>03</b> Review drift and samples</span></div></div>
        )}
      </main>
    </div>
  );
}
