import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type CompletenessAxis,
  type CompletenessFrequency,
  type CoverageCompletenessCandidateResult,
  type CoverageCompletenessCell,
  type CoverageCompletenessInput,
  type CoverageCompletenessResult,
  type CoverageRun,
} from "./api";

const axisOptions: Array<{ value: CompletenessAxis; label: string }> = [
  { value: "label", label: "類別" },
  { value: "split", label: "資料集 split" },
  { value: "brightness", label: "亮度" },
  { value: "contrast", label: "對比" },
  { value: "sharpness", label: "銳利度" },
  { value: "aspect", label: "長寬比" },
];

const frequencyOptions: CompletenessFrequency[] = ["高", "中", "低", "不適用"];
const stateColors: Record<CoverageCompletenessCell["state"], string> = {
  "空": "#c0392b",
  "缺": "#e74c3c",
  "偏缺": "#f39c12",
  "健康": "#2ecc71",
  "假完整": "#9b59b6",
  "過多": "#3498db",
  "不適用": "#cbd2dc",
};

function cellKey(x: number, y: number) {
  return `${x}:${y}`;
}

function makeInput(
  axisX: CompletenessAxis,
  axisY: CompletenessAxis,
  bins: number,
  tAbs: number,
  dStar: number,
  frequencies: Record<string, CompletenessFrequency>,
): CoverageCompletenessInput {
  return {
    axis_x: axisX,
    axis_y: axisY,
    bins,
    t_abs: tAbs,
    d_star: dStar,
    frequencies: Object.entries(frequencies).map(([key, frequency]) => {
      const [x, y] = key.split(":").map(Number);
      return { x, y, frequency };
    }),
  };
}

export default function AttributeCompleteness({ run }: { run: CoverageRun }) {
  const [axisX, setAxisX] = useState<CompletenessAxis>("label");
  const [axisY, setAxisY] = useState<CompletenessAxis>("brightness");
  const [bins, setBins] = useState(3);
  const [target, setTarget] = useState(10);
  const [dStar, setDStar] = useState(0.6);
  const [grid, setGrid] = useState<CoverageCompletenessResult | null>(null);
  const [frequencies, setFrequencies] = useState<Record<string, CompletenessFrequency>>({});
  const [calibrationDirty, setCalibrationDirty] = useState(false);
  const [activeKey, setActiveKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [candidateFolders, setCandidateFolders] = useState<string[]>([]);
  const [candidateFolderDraft, setCandidateFolderDraft] = useState("");
  const [candidateBusy, setCandidateBusy] = useState(false);
  const [candidateResult, setCandidateResult] = useState<CoverageCompletenessCandidateResult | null>(null);
  const [exportBusy, setExportBusy] = useState(false);
  const requestSequence = useRef(0);
  const candidateSequence = useRef(0);

  async function calculate(nextFrequencies: Record<string, CompletenessFrequency>) {
    const requestId = ++requestSequence.current;
    setBusy(true);
    setError("");
    try {
      const result = await api.coverageCompleteness(run.run_id, makeInput(
        axisX, axisY, bins, target, dStar, nextFrequencies,
      ));
      if (requestSequence.current !== requestId) return;
      setGrid(result);
      setFrequencies(Object.fromEntries(result.frequency_classes.map((item) => [
        cellKey(item.x, item.y), item.frequency,
      ])));
      setCalibrationDirty(false);
      setActiveKey((current) => result.cells.some((cell) => cellKey(cell.x, cell.y) === current)
        ? current
        : cellKey(result.cells.find((cell) => cell.n > 0)?.x ?? 0, result.cells.find((cell) => cell.n > 0)?.y ?? 0));
    } catch (reason) {
      if (requestSequence.current === requestId) {
        setError(reason instanceof Error ? reason.message : "Could not calculate the attribute grid.");
      }
    } finally {
      if (requestSequence.current === requestId) setBusy(false);
    }
  }

  useEffect(() => {
    let active = true;
    const requestId = ++requestSequence.current;
    setGrid(null);
    setFrequencies({});
    setCalibrationDirty(false);
    setCandidateResult(null);
    setError("");
    setBusy(true);
    api.coverageCompleteness(run.run_id, {
      axis_x: axisX, axis_y: axisY, bins, t_abs: target, d_star: dStar, frequencies: [],
    }).then((result) => {
      if (!active || requestSequence.current !== requestId) return;
      setGrid(result);
      setActiveKey(cellKey(result.cells.find((cell) => cell.n > 0)?.x ?? 0,
        result.cells.find((cell) => cell.n > 0)?.y ?? 0));
    }).catch((reason: unknown) => {
      if (active && requestSequence.current === requestId) {
        setError(reason instanceof Error ? reason.message : "Could not calculate the attribute grid.");
      }
    }).finally(() => {
      if (active && requestSequence.current === requestId) setBusy(false);
    });
    return () => {
      active = false;
      requestSequence.current += 1;
    };
  }, [run.run_id]);

  const activeCell = useMemo(() => grid?.cells.find((cell) => cellKey(cell.x, cell.y) === activeKey) || null,
    [activeKey, grid]);
  const activeRecords = useMemo(() => activeCell
    ? activeCell.sample_indices.flatMap((index) => run.records[index] ? [{ index, record: run.records[index] }] : [])
    : [], [activeCell, run.records]);
  const missingCount = grid
    ? (grid.health.counts["空"] || 0) + (grid.health.counts["缺"] || 0)
    : 0;
  const axisLabel = (axis: CompletenessAxis) => axisOptions.find((option) => option.value === axis)?.label || axis;

  function resetGrid(next: Partial<{ axisX: CompletenessAxis; axisY: CompletenessAxis; bins: number }>) {
    ++requestSequence.current;
    setBusy(false);
    setGrid(null);
    setActiveKey("");
    ++candidateSequence.current;
    setCandidateBusy(false);
    setCandidateResult(null);
    setFrequencies({});
    setCalibrationDirty(false);
    if (next.axisX) setAxisX(next.axisX);
    if (next.axisY) setAxisY(next.axisY);
    if (next.bins) setBins(next.bins);
  }

  function changeFrequency(key: string, frequency: CompletenessFrequency) {
    setFrequencies((current) => {
      const next = { ...current };
      if (frequency === "中") delete next[key];
      else next[key] = frequency;
      return next;
    });
    setCalibrationDirty(true);
  }

  async function chooseCandidateFolder() {
    setError("");
    if (!window.cimHost?.chooseFile) {
      setNotice("The Native folder picker is not connected. Paste a local folder path below.");
      return;
    }
    try {
      const result = await window.cimHost.chooseFile();
      if (result.canceled) return;
      const paths = result.paths?.length ? result.paths : (result.path ? [result.path] : []);
      if (result.grant_id && paths.length !== 1) {
        throw new Error("Choose one folder at a time so NativeApp can scope its permission.");
      }
      if (result.grant_id) api.rememberFolderGrant(paths[0], result.grant_id);
      paths.forEach(addCandidateFolder);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The candidate folder picker failed.");
    }
  }

  function addCandidateFolder(value = candidateFolderDraft) {
    const path = value.trim();
    if (!path) return;
    setCandidateFolders((current) => current.includes(path) ? current : [...current, path]);
    setCandidateFolderDraft("");
    ++candidateSequence.current;
    setCandidateResult(null);
    setNotice("");
  }

  async function mineCandidates() {
    if (!grid || !activeCell || !candidateFolders.length) return;
    const requestId = ++candidateSequence.current;
    setCandidateBusy(true);
    setError("");
    setCandidateResult(null);
    try {
      const result = await api.mineCoverageCompletenessCandidates(run.run_id, {
        folders: candidateFolders,
        folderGrantIds: api.folderGrantIds(candidateFolders),
        axis_x: axisX,
        axis_y: axisY,
        bins,
        x: activeCell.x,
        y: activeCell.y,
      });
      if (candidateSequence.current !== requestId) return;
      setCandidateResult(result);
      if (!result.total) setError("The candidate pool did not contain any readable images.");
    } catch (reason) {
      if (candidateSequence.current === requestId) {
        setError(reason instanceof Error ? reason.message : "Could not retrieve cell candidates.");
      }
    } finally {
      if (candidateSequence.current === requestId) setCandidateBusy(false);
    }
  }

  async function exportCandidates() {
    if (!candidateResult?.total) return;
    setExportBusy(true);
    try {
      const blob = await api.coverageCompletenessCandidatesCsv(run.run_id);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "cell_candidates.csv";
      link.click();
      URL.revokeObjectURL(url);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not export the candidate list.");
    } finally {
      setExportBusy(false);
    }
  }

  return (
    <section className="attribute-completeness" aria-label="Attribute completeness grid">
      <div className="attribute-heading">
        <div>
          <h3>Attribute completeness grid</h3>
          <p>One dataset, split into interpretable attribute cells. Embedding diversity flags full cells padded with near-duplicates.</p>
        </div>
        <button type="button" data-testid="completeness-recalculate" onClick={() => void calculate(frequencies)} disabled={busy}>
          {busy ? "Calculating…" : "Recalculate grid"}
        </button>
      </div>

      <div className="attribute-controls">
        <label>Horizontal axis
          <select data-testid="completeness-axis-x" value={axisX} onChange={(event) => resetGrid({ axisX: event.target.value as CompletenessAxis })}>
            {axisOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </label>
        <label>Vertical axis
          <select data-testid="completeness-axis-y" value={axisY} onChange={(event) => resetGrid({ axisY: event.target.value as CompletenessAxis })}>
            {axisOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </label>
        <label>Continuous bins
          <input data-testid="completeness-bins" type="number" min={2} max={8} value={bins}
            onChange={(event) => resetGrid({ bins: Math.min(8, Math.max(2, event.target.valueAsNumber || 2)) })} />
        </label>
        <label>Minimum samples per cell
          <input data-testid="completeness-target" type="number" min={1} value={target}
            onChange={(event) => { ++requestSequence.current; setBusy(false); setGrid(null); setTarget(Math.max(1, event.target.valueAsNumber || 1)); }} />
        </label>
        <label>Near-duplicate threshold
          <select data-testid="completeness-dstar" value={dStar} onChange={(event) => { ++requestSequence.current; setBusy(false); setGrid(null); setDStar(Number(event.target.value)); }}>
            <option value={0.45}>Relaxed (0.45)</option>
            <option value={0.6}>Standard (0.60)</option>
            <option value={0.75}>Strict (0.75)</option>
          </select>
        </label>
      </div>

      {error && <div className="error-banner" role="alert">{error}</div>}
      {notice && <div className="notice-banner" role="status">{notice}</div>}
      {grid && (
        <>
          <div className="coverage-metrics" aria-label="Attribute completeness summary">
            <article><strong data-testid="completeness-health">{grid.health.coverage_health.toFixed(0)}%</strong><span>Coverage Health</span></article>
            <article><strong data-testid="completeness-missing-count">{missingCount}/{grid.cells.length}</strong><span>Empty / missing cells</span></article>
            <article><strong data-testid="completeness-gini">{grid.health.gini.toFixed(2)}</strong><span>Count Gini · 0 is balanced</span></article>
            <article><strong data-testid="completeness-fake-ratio">{(grid.health.fake_ratio * 100).toFixed(0)}%</strong><span>Fake-complete cells</span></article>
          </div>
          {!grid.calibrated && <p className="attribute-uncalibrated" data-testid="completeness-uncalibrated">Target counts use the uniform assumption. Rare combinations may appear missing until you calibrate their frequency prior.</p>}

          <div className="attribute-grid-scroll" role="group" aria-label={`${axisLabel(axisX)} by ${axisLabel(axisY)} heatmap`}>
            <div className="attribute-grid" style={{ gridTemplateColumns: `minmax(110px, 150px) repeat(${grid.labels_x.length}, minmax(88px, 1fr))` }}>
              <div className="attribute-grid-corner">{axisLabel(axisY)} ↓ / {axisLabel(axisX)} →</div>
              {grid.labels_x.map((label, index) => <div className="attribute-grid-header" key={`x-${index}`}>{label}</div>)}
              {grid.labels_y.flatMap((labelY, y) => [
                <div className="attribute-grid-row-label" key={`y-${y}`}>{labelY}</div>,
                ...grid.labels_x.map((_, x) => {
                  const cell = grid.cells.find((item) => item.x === x && item.y === y);
                  if (!cell) return <div key={`missing-${x}-${y}`} />;
                  const key = cellKey(cell.x, cell.y);
                  return (
                    <button
                      type="button"
                      key={key}
                      data-testid="coverage-completeness-cell"
                      data-index={key}
                      data-count={cell.n}
                      data-state={cell.state}
                      aria-label={`${cell.state}: ${cell.x_label} × ${cell.y_label}, ${cell.n} / ${cell.t || "not applicable"}`}
                      aria-pressed={activeKey === key}
                      onClick={() => { setActiveKey(key); ++candidateSequence.current; setCandidateBusy(false); setCandidateResult(null); }}
                      style={{ backgroundColor: stateColors[cell.state] }}
                    >
                      <strong>{cell.state}</strong>
                      <span>n={cell.n} / t={cell.t.toFixed(1)}</span>
                      <small>d={cell.d.toFixed(2)}</small>
                    </button>
                  );
                }),
              ])}
            </div>
          </div>
          <div className="attribute-legend" aria-label="Completeness state legend">
            {Object.entries(stateColors).map(([state, color]) => <span key={state}><i style={{ backgroundColor: color }} />{state}</span>)}
          </div>

          <div className="attribute-columns">
            <section className="attribute-card" aria-label="Missing cell worklist">
              <h4>Largest gaps</h4>
              {grid.health.top_gaps.length ? grid.health.top_gaps.slice(0, 30).map((cell) => (
                <button type="button" className="attribute-gap" key={cellKey(cell.x, cell.y)} data-testid="coverage-completeness-gap"
                  onClick={() => { setActiveKey(cellKey(cell.x, cell.y)); ++candidateSequence.current; setCandidateBusy(false); setCandidateResult(null); }}>
                  <span>{cell.x_label} × {cell.y_label}</span>
                  <small>{cell.state} · n={cell.n}/t={cell.t.toFixed(1)} · gap {cell.shortfall?.toFixed(1)}</small>
                </button>
              )) : <p className="coverage-empty-selection">No underfilled cells.</p>}
            </section>

            <section className="attribute-card" aria-label="Cell samples">
              <h4>Images in selected cell</h4>
              {activeCell ? <p className="attribute-cell-summary">{activeCell.x_label} × {activeCell.y_label} · {activeCell.state} · n={activeCell.n}/t={activeCell.t.toFixed(1)} · diversity d={activeCell.d.toFixed(2)}</p> : null}
              {activeCell?.state === "假完整" && <p className="attribute-fake-note">Full by count, low by diversity; inspect duplicates before adding more images.</p>}
              {activeRecords.length ? (
                <div className="coverage-gallery attribute-sample-gallery">
                  {activeRecords.map(({ index, record }) => (
                    <article className="coverage-image-card" key={index} data-testid="coverage-completeness-sample" data-index={index}>
                      <img src={api.coverageThumbnailUrl(run.run_id, index)} alt={record.filename} loading="lazy" />
                      <strong>{record.filename}</strong>
                      <span className="coverage-image-label">{record.label || "(unlabeled)"}</span>
                      <small>{record.split}</small>
                    </article>
                  ))}
                  {activeCell && activeCell.n > activeCell.sample_indices.length && <p className="coverage-empty-selection">Showing first {activeCell.sample_indices.length} of {activeCell.n} images.</p>}
                </div>
              ) : <p className="coverage-empty-selection">This cell has no images.</p>}
            </section>
          </div>

          <section className="attribute-card attribute-calibration" aria-label="Frequency prior calibration">
            <div className="attribute-card-heading">
              <div><h4>Real-world frequency prior</h4><p>High doubles the per-cell target, medium keeps it, low scales it to 0.4, and not applicable removes an impossible combination from the denominator. Unedited cells default to medium.</p></div>
              <button type="button" data-testid="completeness-apply-calibration" onClick={() => void calculate(frequencies)} disabled={busy || !calibrationDirty}>
                {busy ? "Applying…" : "Apply frequency calibration"}
              </button>
            </div>
            <div className="attribute-table-wrap">
              <table>
                <thead><tr><th>Cell</th><th>Images</th><th>Frequency prior</th></tr></thead>
                <tbody>
                  {grid.cells.map((cell) => {
                    const key = cellKey(cell.x, cell.y);
                    return <tr key={key} data-testid="completeness-frequency-row" data-index={key}>
                      <td>{cell.x_label} × {cell.y_label}</td>
                      <td>{cell.n}</td>
                      <td><select aria-label={`Frequency prior ${cell.x_label} × ${cell.y_label}`}
                        data-testid={`completeness-frequency-${cell.x}-${cell.y}`}
                        value={frequencies[key] || "中"}
                        onChange={(event) => changeFrequency(key, event.target.value as CompletenessFrequency)}>
                        {frequencyOptions.map((frequency) => <option value={frequency} key={frequency}>{frequency}</option>)}
                      </select></td>
                    </tr>;
                  })}
                </tbody>
              </table>
            </div>
          </section>

          <section className="attribute-card attribute-mining" aria-label="Cell candidate pool">
            <h4>Retrieve similar images for this cell</h4>
            <p>Candidate images are ranked by cosine distance to the selected cell centroid. Empty cells use the selected class as a query seed.</p>
            <button type="button" className="host-picker" onClick={() => void chooseCandidateFolder()}>
              <span className="folder-icon" aria-hidden="true">▱</span>Choose candidate folder<span className="picker-hint">Native</span>
            </button>
            <label className="field-label" htmlFor="completeness-candidate-folder-path">Candidate pool folder</label>
            <div className="folder-input-row">
              <input id="completeness-candidate-folder-path" data-testid="completeness-candidate-folder-path"
                value={candidateFolderDraft} onChange={(event) => setCandidateFolderDraft(event.target.value)}
                onKeyDown={(event) => event.key === "Enter" && (event.preventDefault(), addCandidateFolder())}
                placeholder="Paste a local folder path" autoComplete="off" />
              <button type="button" className="add-folder" data-testid="completeness-add-candidate-folder" onClick={() => addCandidateFolder()} aria-label="Add candidate pool folder">+</button>
            </div>
            {candidateFolders.length > 0 && <ul className="folder-list" aria-label="Completeness candidate folders">
              {candidateFolders.map((folder) => <li key={folder} title={folder}>
                <span className="folder-chip-dot" /><span className="folder-chip-name">{folder.split(/[\\/]/).filter(Boolean).at(-1) || folder}</span>
                <button type="button" aria-label={`Remove candidate folder ${folder}`} onClick={() => {
                  setCandidateFolders((current) => current.filter((item) => item !== folder));
                  ++candidateSequence.current;
                  setCandidateBusy(false);
                  setCandidateResult(null);
                }}>×</button>
              </li>)}
            </ul>}
            <button type="button" data-testid="completeness-mine-candidates" className="run-button"
              onClick={() => void mineCandidates()} disabled={!activeCell || !candidateFolders.length || candidateBusy}>
              {candidateBusy ? "Retrieving candidates…" : "Retrieve candidates for selected cell"}
            </button>
            {candidateResult && <>
              <div className="attribute-card-heading"><p data-testid="completeness-candidate-count">{candidateResult.total} candidates · nearest first</p>
                <button type="button" data-testid="completeness-export-candidates" onClick={() => void exportCandidates()} disabled={exportBusy}>
                  {exportBusy ? "Preparing CSV…" : "Export candidate CSV"}
                </button>
              </div>
              {candidateResult.items.length ? <div className="coverage-gallery attribute-sample-gallery">
                {candidateResult.items.map((item) => <article className="coverage-image-card" key={item.index} data-testid="completeness-candidate" data-index={item.index}>
                  <img src={api.coverageCompletenessCandidateThumbnailUrl(run.run_id, item.index)} alt={item.filename} loading="lazy" />
                  <strong>{item.filename}</strong>
                  <span className="coverage-image-label">{item.label || "(unlabeled)"}</span>
                  <small>{item.split} · cosine distance {item.distance.toFixed(3)}</small>
                </article>)}
              </div> : <p className="coverage-empty-selection">The candidate pool contains no readable images.</p>}
            </>}
          </section>
        </>
      )}
      {!grid && !error && <div className="job-status" role="status">{busy ? "Calculating attribute cells…" : "Choose the axes and recalculate."}</div>}
      <footer className="attribute-provenance">Run {run.run_id.slice(0, 8)} · {run.records.length} source images · {run.model_id} feature model</footer>
    </section>
  );
}
