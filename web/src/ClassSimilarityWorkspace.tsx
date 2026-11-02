import { useMemo, useState } from "react";
import {
  api,
  type ClassSimilarityBoxSummary,
  type ClassSimilarityItem,
  type ClassSimilarityMetric,
  type ClassSimilarityResult,
  type ClassSimilarityEvaluationPreview,
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

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function PrecisionRecallPlot({ points }: { points: ClassSimilarityEvaluationPreview["precision_recall_curve"] }) {
  if (points.length < 2) return <p className="class-similarity-empty">A tuning review set with at least one reviewed incorrect and one reviewed correct item is needed for a PR curve.</p>;
  const xy = points.map((point) => `${32 + point.recall * 260},${180 - point.precision * 150}`).join(" ");
  return <svg className="class-similarity-pr-plot" viewBox="0 0 320 210" role="img" aria-label="Precision recall curve">
    <line x1="32" y1="30" x2="32" y2="180" className="class-similarity-pr-axis" />
    <line x1="32" y1="180" x2="292" y2="180" className="class-similarity-pr-axis" />
    <polyline points={xy} className="class-similarity-pr-line" />
    {points.map((point, index) => <circle key={`${point.threshold}-${index}`} cx={32 + point.recall * 260} cy={180 - point.precision * 150} r="3" className="class-similarity-pr-point" />)}
    <text x="150" y="204" className="class-similarity-pr-label">Recall</text>
    <text x="4" y="22" className="class-similarity-pr-label">Precision</text>
  </svg>;
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
  const [evaluationScope, setEvaluationScope] = useState<"all" | "selected">("all");
  const [evaluationPreview, setEvaluationPreview] = useState<ClassSimilarityEvaluationPreview | null>(null);
  const [evaluationBusy, setEvaluationBusy] = useState(false);
  const [evaluationMessage, setEvaluationMessage] = useState("");
  const [reportBusy, setReportBusy] = useState(false);

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

  const currentPolicy: ClassSimilarityEvaluationPreview["policy"] = {
    mode: candidateMode,
    threshold: candidateMode === "threshold" ? threshold : null,
    comparator: "lt",
    score: "margin",
  };
  const evaluationPreviewIsCurrent = evaluationPreview?.policy.mode === currentPolicy.mode
    && evaluationPreview.policy.threshold === currentPolicy.threshold;

  async function calculate() {
    if (busy) return;
    setBusy(true);
    setError("");
    setResult(null);
    setSelected(null);
    setSelectedIds([]);
    setCartMessage("");
    setEvaluationPreview(null);
    setEvaluationMessage("");
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

  async function downloadEvaluationTemplate() {
    if (!jobId || evaluationBusy) return;
    setEvaluationBusy(true);
    setEvaluationMessage("");
    try {
      const blob = await api.classSimilarityEvaluationTemplate(jobId, {
        scope: evaluationScope,
        selected_ids: evaluationScope === "selected" ? selectedIds : [],
      });
      saveBlob(blob, "class_similarity_evaluation.json");
      setEvaluationMessage("Template downloaded. Review verdicts in the JSON, then upload it here.");
    } catch (reason) {
      setEvaluationMessage(reason instanceof Error ? reason.message : "Could not create the review template.");
    } finally {
      setEvaluationBusy(false);
    }
  }

  async function importEvaluationFile(file: File | undefined) {
    if (!file || !jobId || evaluationBusy) return;
    setEvaluationBusy(true);
    setEvaluationMessage("");
    try {
      const manifest = JSON.parse(await file.text()) as Record<string, unknown>;
      const preview = await api.importClassSimilarityEvaluation(jobId, manifest, currentPolicy);
      setEvaluationPreview(preview);
      setEvaluationMessage("Review set validated against this Visualize source snapshot.");
    } catch (reason) {
      setEvaluationPreview(null);
      setEvaluationMessage(reason instanceof Error ? reason.message : "Could not validate this review JSON.");
    } finally {
      setEvaluationBusy(false);
    }
  }

  async function previewEvaluation() {
    if (!jobId || !evaluationPreview?.evaluation || evaluationBusy) return;
    setEvaluationBusy(true);
    setEvaluationMessage("");
    try {
      setEvaluationPreview(await api.previewClassSimilarityEvaluation(jobId, currentPolicy));
    } catch (reason) {
      setEvaluationMessage(reason instanceof Error ? reason.message : "Could not score the current threshold.");
    } finally {
      setEvaluationBusy(false);
    }
  }

  async function exportReport(scope: "all" | "candidates" | "selected", format: "csv" | "json") {
    if (!jobId || reportBusy || (scope === "selected" && !selectedIds.length)) return;
    setReportBusy(true);
    setEvaluationMessage("");
    try {
      const blob = await api.classSimilarityReport(jobId, {
        scope,
        selected_ids: scope === "selected" ? selectedIds : [],
        policy: currentPolicy,
        format,
      });
      saveBlob(blob, `class_similarity_${scope}.${format}`);
    } catch (reason) {
      setEvaluationMessage(reason instanceof Error ? reason.message : "Could not export class-similarity evidence.");
    } finally {
      setReportBusy(false);
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

          <section className="class-similarity-evaluation" aria-labelledby="class-similarity-evaluation-title">
            <div className="class-similarity-panel-heading">
              <div><div className="eyebrow">VERIFIED HUMAN REVIEW</div><h2 id="class-similarity-evaluation-title">Evaluation and evidence export</h2></div>
              <span>Review verdicts stay attached to this source snapshot.</span>
            </div>
            <p>Download a frozen JSON template, label each item as incorrect, correct, unresolved, or unreviewed, then upload it to measure review coverage and threshold Precision / Recall.</p>
            <div className="class-similarity-evaluation-controls">
              <label><span>Template population</span><select aria-label="Evaluation template scope" value={evaluationScope} onChange={(event) => setEvaluationScope(event.target.value as "all" | "selected")}>
                <option value="all">All eligible records</option><option value="selected">Selected records ({selectedIds.length})</option>
              </select></label>
              <button type="button" className="secondary-action" onClick={() => void downloadEvaluationTemplate()} disabled={evaluationBusy || (evaluationScope === "selected" && !selectedIds.length)}>Download evaluation template</button>
              <label className="class-similarity-file-input"><span>Upload reviewed JSON</span><input aria-label="Upload class similarity evaluation JSON" type="file" accept="application/json,.json" disabled={evaluationBusy} onChange={(event) => { void importEvaluationFile(event.target.files?.[0]); event.currentTarget.value = ""; }} /></label>
              {evaluationPreview?.evaluation && <button type="button" className="secondary-action" onClick={() => void previewEvaluation()} disabled={evaluationBusy}>{evaluationBusy ? "Calculating…" : "Evaluate current threshold"}</button>}
            </div>
            {evaluationMessage && <div className="class-similarity-progress" role="status">{evaluationMessage}</div>}
            {evaluationPreview?.evaluation && <div className="class-similarity-evaluation-results" data-testid="class-similarity-evaluation-results">
              <div className="class-similarity-evaluation-metrics">
                <span><strong>{evaluationPreview.evaluation.pending_n}</strong> pending / {evaluationPreview.evaluation.frozen_ids.length} frozen</span>
                <span><strong>{evaluationPreview.summary.candidate_n}</strong> candidates / {evaluationPreview.summary.eligible_n} scored</span>
                <span><strong>{evaluationPreview.counts?.coverage === null || evaluationPreview.counts?.coverage === undefined ? "—" : `${(evaluationPreview.counts.coverage * 100).toFixed(1)}%`}</strong> review coverage</span>
                <span><strong>{evaluationPreview.counts?.precision === null || evaluationPreview.counts?.precision === undefined ? "—" : evaluationPreview.counts.precision.toFixed(3)}</strong> precision</span>
                <span><strong>{evaluationPreview.counts?.recall === null || evaluationPreview.counts?.recall === undefined ? "—" : evaluationPreview.counts.recall.toFixed(3)}</strong> recall</span>
              </div>
              {evaluationPreview.counts?.status === "ready" ? <p>TP {evaluationPreview.counts.tp} · FP {evaluationPreview.counts.fp} · FN {evaluationPreview.counts.fn} · TN {evaluationPreview.counts.tn} · {evaluationPreview.evaluation.purpose}</p> : <p className="class-similarity-empty">Precision / Recall unavailable: {evaluationPreview.counts?.unavailable_reason || "review enough frozen items first"}.</p>}
              {evaluationPreview.evaluation.purpose === "tuning" && <PrecisionRecallPlot points={evaluationPreview.precision_recall_curve} />}
              {!evaluationPreviewIsCurrent && <p className="class-similarity-empty">The threshold changed. Evaluate again to update these metrics.</p>}
            </div>}
            <div className="class-similarity-export-controls" aria-label="Export class similarity evidence">
              <span>Export current analysis</span>
              {(["all", "candidates", "selected"] as const).map((scope) => <div key={scope}>
                <strong>{scope === "all" ? "All" : scope === "candidates" ? `Candidates (${candidateIds.length})` : `Selected (${selectedIds.length})`}</strong>
                <button type="button" className="secondary-action" onClick={() => void exportReport(scope, "csv")} disabled={reportBusy || (scope === "selected" && !selectedIds.length)}>CSV</button>
                <button type="button" className="secondary-action" onClick={() => void exportReport(scope, "json")} disabled={reportBusy || (scope === "selected" && !selectedIds.length)}>JSON</button>
              </div>)}
            </div>
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
