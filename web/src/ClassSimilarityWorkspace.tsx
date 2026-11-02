import { useMemo, useState } from "react";
import {
  api,
  type ClassSimilarityBoxSummary,
  type ClassSimilarityItem,
  type ClassSimilarityMetric,
  type ClassSimilarityResult,
  type VisualizeUnit,
  type RunRecord,
} from "./api";

type Props = {
  runId: string;
  modelId: string;
  unit: VisualizeUnit;
  records: RunRecord[];
  onReturn: () => void;
  onOpenRecord: (index: number) => void;
};

const terminalStates = new Set(["completed", "failed", "cancelled"]);

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function BoxPlot({ rows, threshold = null }: { rows: ClassSimilarityBoxSummary[]; threshold?: number | null }) {
  const scored = rows.filter((row) => row.q1 !== undefined && row.q3 !== undefined);
  if (!scored.length) return <p className="class-similarity-empty">No records have a score for this measure.</p>;
  const min = Math.min(...scored.map((row) => row.whisker_low ?? row.q1 ?? 0));
  const max = Math.max(...scored.map((row) => row.whisker_high ?? row.q3 ?? 0));
  const span = max - min || 1;
  const left = 150;
  const right = 760;
  const x = (value: number) => left + ((value - min) / span) * (right - left);
  const height = Math.max(68, rows.length * 46 + 24);

  return (
    <svg className="class-similarity-boxplot" viewBox={`0 0 800 ${height}`} role="img" aria-label="Per-class similarity box plot">
      {threshold !== null && threshold >= min && threshold <= max && <line x1={x(threshold)} x2={x(threshold)} y1="12" y2={height - 16} className="class-similarity-threshold-line" />}
      {rows.map((row, index) => {
        const y = 34 + index * 46;
        if (row.q1 === undefined || row.q3 === undefined || row.median === undefined) {
          return <g key={row.label}><text x="8" y={y + 4} className="class-similarity-axis-label">{row.label}</text><text x={left} y={y + 4} className="class-similarity-unscored">No scored records · {row.n_unscored} unscored</text></g>;
        }
        const low = row.whisker_low ?? row.q1;
        const high = row.whisker_high ?? row.q3;
        return (
          <g key={row.label}>
            <text x="8" y={y + 4} className="class-similarity-axis-label">{row.label}</text>
            <line x1={x(low)} x2={x(high)} y1={y} y2={y} className="class-similarity-whisker" />
            <line x1={x(low)} x2={x(low)} y1={y - 6} y2={y + 6} className="class-similarity-whisker-cap" />
            <line x1={x(high)} x2={x(high)} y1={y - 6} y2={y + 6} className="class-similarity-whisker-cap" />
            <rect x={x(row.q1)} y={y - 11} width={Math.max(1, x(row.q3) - x(row.q1))} height="22" rx="3" className="class-similarity-box" />
            <line x1={x(row.median)} x2={x(row.median)} y1={y - 12} y2={y + 12} className="class-similarity-median" />
            <text x="790" y={y + 4} textAnchor="end" className="class-similarity-count">n={row.n_scored}/{row.n_total}</text>
          </g>
        );
      })}
      <text x={left} y={height - 4} className="class-similarity-axis-value">{min.toFixed(3)}</text>
      <text x={right} y={height - 4} textAnchor="end" className="class-similarity-axis-value">{max.toFixed(3)}</text>
    </svg>
  );
}

function metricValue(item: ClassSimilarityItem, metric: ClassSimilarityMetric) {
  return item[metric];
}

export default function ClassSimilarityWorkspace({ runId, modelId, unit, records, onReturn, onOpenRecord }: Props) {
  const [k, setK] = useState(5);
  const [analysisSplit, setAnalysisSplit] = useState("all");
  const [metric, setMetric] = useState<ClassSimilarityMetric>("margin");
  const [jobId, setJobId] = useState("");
  const [jobState, setJobState] = useState("");
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<ClassSimilarityResult | null>(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<ClassSimilarityItem | null>(null);
  const [visibleRows, setVisibleRows] = useState(40);
  const [busy, setBusy] = useState(false);
  const [threshold, setThreshold] = useState(0);
  const [candidateMode, setCandidateMode] = useState<"threshold" | "none" | "all_eligible">("threshold");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [cartBusy, setCartBusy] = useState(false);
  const [cartMessage, setCartMessage] = useState("");

  const splits = useMemo(() => [...new Set(records.map((record) => record.split).filter(Boolean))].sort(), [records]);
  const rows = useMemo(() => {
    if (!result) return [];
    return [...result.items].sort((a, b) => {
      const av = metricValue(a, metric);
      const bv = metricValue(b, metric);
      if (av === null && bv === null) return a.index - b.index;
      if (av === null) return 1;
      if (bv === null) return -1;
      return av - bv || a.index - b.index;
    });
  }, [result, metric]);
  const eligibleCount = result?.items.filter((item) => item.margin !== null).length || 0;
  const candidateIds = useMemo(() => {
    if (!result || candidateMode === "none") return [];
    return result.items.filter((item) => item.margin !== null && (
      candidateMode === "all_eligible" || item.margin < threshold
    )).map((item) => item.record_id);
  }, [result, candidateMode, threshold]);

  async function calculate() {
    if (busy) return;
    setBusy(true);
    setError("");
    setResult(null);
    setSelected(null);
    setSelectedIds([]);
    setCartMessage("");
    setVisibleRows(40);
    try {
      const created = await api.createClassSimilarityJob(runId, { model_id: modelId, k, analysis_split: analysisSplit });
      setJobId(created.job_id);
      setJobState(created.state);
      for (;;) {
        const status = await api.classSimilarityJob(created.job_id);
        setJobState(status.state);
        setProgress(status.percent);
        if (terminalStates.has(status.state)) {
          if (status.state === "completed" && status.result) setResult(status.result);
          else if (status.state === "failed") setError(status.error?.message || "Class-similarity analysis failed.");
          break;
        }
        await delay(160);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Class-similarity analysis failed.");
    } finally {
      setBusy(false);
    }
  }

  async function cancel() {
    if (!jobId || !busy) return;
    try {
      await api.cancelClassSimilarityJob(jobId);
      setJobState("cancelling");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not cancel this analysis.");
    }
  }

  async function addSelectedToCuration() {
    if (!result || !selectedIds.length || cartBusy) return;
    setCartBusy(true);
    setCartMessage("");
    try {
      const selectedSet = new Set(selectedIds);
      const indices = result.items.filter((item) => selectedSet.has(item.record_id)).map((item) => item.index);
      const reason = candidateMode === "threshold"
        ? `類別相似度候選：margin < ${threshold}`
        : candidateMode === "all_eligible" ? "類別相似度：人工選取全部可計分記錄" : "類別相似度：人工選取記錄";
      const response = await api.addCuration(runId, indices, "類別相似度", reason);
      setCartMessage(`Added ${response.added}; skipped ${response.skipped} existing item(s).`);
    } catch (reason) {
      setCartMessage(reason instanceof Error ? reason.message : "Could not add the selected records to Curation.");
    } finally {
      setCartBusy(false);
    }
  }

  const metricLabels: Record<ClassSimilarityMetric, string> = {
    margin: "Same-class minus rival similarity",
    same: "Same-class similarity",
    other: "Closest rival-class similarity",
  };

  return (
    <main className="class-similarity-workspace" data-testid="class-similarity-workspace" aria-labelledby="class-similarity-title">
      <header className="class-similarity-header">
        <div>
          <div className="eyebrow">VISUALIZE · REVIEW EVIDENCE</div>
          <h1 id="class-similarity-title">類別相似度分布</h1>
          <p>以目前 Visualize run 的原始高維特徵計算同類與最相似異類鄰居。負 margin 是優先檢查的線索，不是自動改標判定。</p>
        </div>
        <button type="button" className="secondary-action" onClick={onReturn}>Return to Visualize</button>
      </header>

      <section className="class-similarity-controls" aria-label="Class similarity settings">
        <label><span>Model</span><input aria-label="Class similarity model" value={modelId} readOnly /></label>
        <label><span>Record type</span><input aria-label="Class similarity record type" value={unit === "object" ? "Objects" : "Images"} readOnly /></label>
        <label><span>Analysis split</span>
          <select aria-label="Class similarity split" value={analysisSplit} onChange={(event) => setAnalysisSplit(event.target.value)} disabled={busy}>
            <option value="all">All splits</option>
            {splits.map((split) => <option key={split} value={split}>{split}</option>)}
          </select>
        </label>
        <label><span>Top-K per class</span>
          <input aria-label="Class similarity K" type="number" min={1} max={20} value={k} disabled={busy} onChange={(event) => setK(Math.max(1, Math.min(20, Number(event.target.value) || 1)))} />
        </label>
        <button type="button" className="primary-action" onClick={() => void calculate()} disabled={busy || !records.length}>Calculate class similarity</button>
        {busy && <button type="button" className="secondary-action" onClick={() => void cancel()}>Cancel</button>}
      </section>

      {busy && <div className="class-similarity-progress" role="status">Calculating local feature similarities · {progress}%</div>}
      {jobState === "cancelled" && <div className="class-similarity-progress" role="status">Analysis cancelled.</div>}
      {error && <div className="class-similarity-error" role="alert">{error}</div>}

      {result && (
        <>
          <section className="class-similarity-summary" data-testid="class-similarity-summary" aria-label="Analysis summary">
            <span><strong>{result.summary.valid_count.toLocaleString()}</strong> eligible</span>
            <span><strong>{result.summary.excluded_count.toLocaleString()}</strong> excluded</span>
            <span><strong>{result.summary.input_count.toLocaleString()}</strong> analyzed</span>
            <span>{result.model_id} · K={result.k} · {result.unit}</span>
          </section>

          <section className="class-similarity-chart-panel" aria-labelledby="class-similarity-chart-title">
            <div className="class-similarity-panel-heading">
              <div><div className="eyebrow">PER-CLASS DISTRIBUTION</div><h2 id="class-similarity-chart-title">Similarity box plot</h2></div>
              <label><span>Measure</span><select aria-label="Class similarity measure" value={metric} onChange={(event) => setMetric(event.target.value as ClassSimilarityMetric)}>
                {(Object.keys(metricLabels) as ClassSimilarityMetric[]).map((key) => <option value={key} key={key}>{metricLabels[key]}</option>)}
              </select></label>
            </div>
            <BoxPlot rows={result.summaries[metric]} threshold={metric === "margin" && candidateMode === "threshold" ? threshold : null} />
          </section>

          <section className="class-similarity-threshold-panel" aria-label="Threshold candidate preview">
            <div className="class-similarity-panel-heading"><div><div className="eyebrow">HUMAN REVIEW QUEUE</div><h2>Threshold preview</h2></div><span>Precision / Recall unavailable without a verified review set.</span></div>
            <div className="class-similarity-threshold-controls">
              <label><span>Margin threshold</span><input aria-label="Class similarity threshold" type="number" min={-2} max={2} step="any" value={threshold} disabled={candidateMode !== "threshold"} onChange={(event) => { const value = Number(event.target.value); if (Number.isFinite(value) && value >= -2 && value <= 2) setThreshold(value); }} /></label>
              <input aria-label="Class similarity threshold slider" type="range" min={-2} max={2} step={0.01} value={threshold} disabled={candidateMode !== "threshold"} onChange={(event) => setThreshold(Number(event.target.value))} />
              <button type="button" className={candidateMode === "none" ? "active" : ""} onClick={() => setCandidateMode("none")}>No candidates</button>
              <button type="button" className={candidateMode === "all_eligible" ? "active" : ""} onClick={() => setCandidateMode("all_eligible")}>All scored</button>
              <button type="button" className="secondary-action" onClick={() => { setCandidateMode("threshold"); setThreshold(0); }}>Reset threshold</button>
            </div>
            <p className="class-similarity-candidate-count" data-testid="class-similarity-candidate-count">
              {candidateIds.length.toLocaleString()} candidates · {eligibleCount ? (candidateIds.length / eligibleCount * 100).toFixed(1) : "0.0"}% of scored records · {result.items.length - eligibleCount} unscored
            </p>
            {metric !== "margin" && <p className="class-similarity-empty">Candidates still use margin &lt; threshold; the selected plot shows {metricLabels[metric]}.</p>}
            <div className="class-similarity-selection-actions">
              <button type="button" className="secondary-action" onClick={() => setSelectedIds(candidateIds)}>Select current candidates ({candidateIds.length})</button>
              <button type="button" className="secondary-action" onClick={() => setSelectedIds([])} disabled={!selectedIds.length}>Clear selection ({selectedIds.length})</button>
              <button type="button" className="primary-action" onClick={() => void addSelectedToCuration()} disabled={!selectedIds.length || cartBusy}>{cartBusy ? "Adding…" : "Add selected to Curation"}</button>
            </div>
            {cartMessage && <div className="class-similarity-progress" role="status">{cartMessage}</div>}
          </section>

          <div className="class-similarity-results-grid">
            <section className="class-similarity-table-panel" aria-labelledby="class-similarity-table-title">
              <div className="class-similarity-panel-heading"><div><div className="eyebrow">RECORDS</div><h2 id="class-similarity-table-title">Ranked evidence</h2></div><span>{rows.length.toLocaleString()} rows</span></div>
              <div className="class-similarity-table-scroll">
                <table>
                  <thead><tr><th>Image / object</th><th>Class</th><th>Same</th><th>Rival</th><th>Margin</th><th>Status</th></tr></thead>
                  <tbody>{rows.slice(0, visibleRows).map((item) => (
                    <tr key={item.record_id} data-testid="class-similarity-row" data-index={item.index} className={selected?.record_id === item.record_id ? "selected" : ""}>
                      <td><input type="checkbox" aria-label={`Select ${item.filename}`} checked={selectedIds.includes(item.record_id)} onChange={(event) => setSelectedIds((current) => event.target.checked ? (current.includes(item.record_id) ? current : [...current, item.record_id]) : current.filter((id) => id !== item.record_id))} /><button type="button" className="class-similarity-record-link" onClick={() => setSelected(item)}>{item.filename}{unit === "object" && item.obj_index !== null ? ` · object ${item.obj_index + 1}` : ""}</button></td>
                      <td>{item.label || "—"}</td>
                      <td>{item.same === null ? "—" : item.same.toFixed(4)}</td>
                      <td>{item.other === null ? "—" : item.other.toFixed(4)}</td>
                      <td>{item.margin === null ? "—" : item.margin.toFixed(4)}</td>
                      <td>{item.status}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
              {visibleRows < rows.length && <button type="button" className="secondary-action" onClick={() => setVisibleRows((count) => count + 40)}>Show more ({rows.length - visibleRows} remaining)</button>}
            </section>

            <aside className="class-similarity-evidence" aria-label="Selected similarity evidence">
              {selected ? (
                <>
                  <div className="eyebrow">SELECTED RECORD</div>
                  <h2>{selected.filename}</h2>
                  <img className="class-similarity-preview" data-testid="class-similarity-preview" src={`/api/v1/runs/${encodeURIComponent(runId)}/images/${selected.index}`} alt={selected.filename} />
                  <p>{selected.label || "Unlabeled"} · {selected.split || "No split"} · {selected.status}</p>
                  {selected.rival_class && <p>Closest rival class: <strong>{selected.rival_class}</strong></p>}
                  <button type="button" className="secondary-action" onClick={() => onOpenRecord(selected.index)}>Open record in Visualize</button>
                  <div className="class-similarity-neighbors">
                    <h3>Same-class neighbors</h3>
                    {selected.same_neighbors.map((neighbor) => <button type="button" className="class-similarity-neighbor" key={`${neighbor.index}-same`} onClick={() => onOpenRecord(neighbor.index)}><img src={`/api/v1/runs/${encodeURIComponent(runId)}/images/${neighbor.index}/thumbnail`} alt="" /><span>{neighbor.filename}<small>{neighbor.label} · cosine {neighbor.cosine.toFixed(4)}</small></span></button>)}
                    <h3>Rival-class neighbors</h3>
                    {selected.rival_neighbors.map((neighbor) => <button type="button" className="class-similarity-neighbor" key={`${neighbor.index}-rival`} onClick={() => onOpenRecord(neighbor.index)}><img src={`/api/v1/runs/${encodeURIComponent(runId)}/images/${neighbor.index}/thumbnail`} alt="" /><span>{neighbor.filename}<small>{neighbor.label} · cosine {neighbor.cosine.toFixed(4)}</small></span></button>)}
                  </div>
                </>
              ) : <p>Select a row to inspect its original image and Top-K neighbor evidence.</p>}
            </aside>
          </div>
        </>
      )}
    </main>
  );
}
