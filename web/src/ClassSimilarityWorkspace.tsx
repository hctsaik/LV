import { useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import {
  api,
  type ClassSimilarityBoxSummary,
  type ClassSimilarityItem,
  type ClassSimilarityMetric,
  type ClassSimilarityResult,
  type ClassSimilarityEvaluationPreview,
  type ClassSimilarityScatterItem,
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

function BoxPlot({ rows, threshold = null, domain }: { rows: ClassSimilarityBoxSummary[]; threshold?: number | null; domain?: [number, number] }) {
  const scored = rows.filter((row) => row.q1 !== undefined && row.q3 !== undefined);
  if (!scored.length) return <p className="class-similarity-empty">No records have a score for this measure.</p>;
  const min = domain?.[0] ?? Math.min(...scored.map((row) => row.whisker_low ?? row.q1 ?? 0));
  const max = domain?.[1] ?? Math.max(...scored.map((row) => row.whisker_high ?? row.q3 ?? 0));
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

function ScatterPlot({
  items, labels, metric, domain, threshold, candidateIds, selectedIds,
  onInspect, onBoxSelect,
}: {
  items: ClassSimilarityScatterItem[];
  labels: string[];
  metric: ClassSimilarityMetric;
  domain: [number, number];
  threshold: number | null;
  candidateIds: Set<string>;
  selectedIds: string[];
  onInspect: (recordId: string) => void;
  onBoxSelect: (recordIds: string[]) => void;
}) {
  const width = 800;
  const height = 390;
  const left = 72;
  const right = 782;
  const top = 18;
  const bottom = 342;
  const span = domain[1] - domain[0] || 1;
  const step = labels.length > 1 ? (right - left) / (labels.length - 1) : 0;
  const xAt = (item: ClassSimilarityScatterItem) => Math.max(left, Math.min(right, left + labels.indexOf(item.label || "") * step + item.jitter * Math.max(step, 48)));
  const yAt = (value: number) => bottom - ((value - domain[0]) / span) * (bottom - top);
  const points = items.filter((item) => labels.includes(item.label || ""));
  const startRef = useRef<{ x: number; y: number } | null>(null);
  const draggedRef = useRef(false);
  const [drag, setDrag] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);

  function point(event: ReactPointerEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    return { x: (event.clientX - rect.left) * width / rect.width, y: (event.clientY - rect.top) * height / rect.height };
  }

  function pointerDown(event: ReactPointerEvent<SVGSVGElement>) {
    if (event.button !== 0) return;
    const p = point(event);
    startRef.current = p;
    draggedRef.current = false;
    setDrag({ x1: p.x, y1: p.y, x2: p.x, y2: p.y });
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function pointerMove(event: ReactPointerEvent<SVGSVGElement>) {
    if (!startRef.current) return;
    const p = point(event);
    if (Math.abs(p.x - startRef.current.x) + Math.abs(p.y - startRef.current.y) > 5) draggedRef.current = true;
    setDrag({ x1: startRef.current.x, y1: startRef.current.y, x2: p.x, y2: p.y });
  }

  function pointerUp() {
    if (!startRef.current || !drag) return;
    if (draggedRef.current) {
      const x1 = Math.min(drag.x1, drag.x2), x2 = Math.max(drag.x1, drag.x2);
      const y1 = Math.min(drag.y1, drag.y2), y2 = Math.max(drag.y1, drag.y2);
      const selected = points.filter((item) => {
        const x = xAt(item), y = yAt(item.value);
        return x >= x1 && x <= x2 && y >= y1 && y <= y2;
      }).map((item) => item.record_id);
      if (selected.length) onBoxSelect(selected);
      window.setTimeout(() => { draggedRef.current = false; }, 0);
    }
    startRef.current = null;
    setDrag(null);
  }

  const selected = new Set(selectedIds);
  return <svg className="class-similarity-scatter" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${metric} distribution scatter plot`}
    onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp} onPointerCancel={pointerUp}>
    {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
      const value = domain[0] + span * fraction;
      const y = yAt(value);
      return <g key={fraction}><line x1={left} x2={right} y1={y} y2={y} className="class-similarity-scatter-grid" /><text x={left - 8} y={y + 4} textAnchor="end" className="class-similarity-scatter-axis">{value.toFixed(2)}</text></g>;
    })}
    {metric === "margin" && domain[0] <= 0 && domain[1] >= 0 && <line x1={left} x2={right} y1={yAt(0)} y2={yAt(0)} className="class-similarity-scatter-zero" />}
    {threshold !== null && threshold >= domain[0] && threshold <= domain[1] && <line x1={left} x2={right} y1={yAt(threshold)} y2={yAt(threshold)} className="class-similarity-scatter-threshold" />}
    {labels.map((label, index) => <g key={label}><line x1={labels.length === 1 ? (left + right) / 2 : left + index * step} x2={labels.length === 1 ? (left + right) / 2 : left + index * step} y1={top} y2={bottom} className="class-similarity-scatter-class-guide" /><text x={labels.length === 1 ? (left + right) / 2 : left + index * step} y={bottom + 24} textAnchor="middle" className="class-similarity-scatter-axis">{label.length > 18 ? `${label.slice(0, 16)}…` : label}</text></g>)}
    {points.map((item) => <circle key={item.record_id} cx={xAt(item)} cy={yAt(item.value)} r={selected.has(item.record_id) ? 5.5 : 4} className={candidateIds.has(item.record_id) ? "class-similarity-scatter-point candidate" : "class-similarity-scatter-point"} data-record-id={item.record_id} onClick={(event) => { event.stopPropagation(); if (!draggedRef.current) onInspect(item.record_id); }}><title>{`${item.label || "Unclassified"} · ${item.value.toFixed(4)} · ${item.record_id}`}</title></circle>)}
    {draggedRef.current && drag && <rect x={Math.min(drag.x1, drag.x2)} y={Math.min(drag.y1, drag.y2)} width={Math.abs(drag.x2 - drag.x1)} height={Math.abs(drag.y2 - drag.y1)} className="class-similarity-scatter-selection" />}
    <text x={(left + right) / 2} y={height - 4} textAnchor="middle" className="class-similarity-scatter-axis">Drag a box to add points to selection · click a point to inspect</text>
  </svg>;
}

function WorkloadCurve({ points }: { points: Array<{ candidate_n: number; mode: string; threshold: number | null }> }) {
  if (points.length < 2) return null;
  const width = 760, height = 170, left = 36, right = 744, top = 16, bottom = 140;
  const max = Math.max(1, ...points.map((point) => point.candidate_n));
  const coords = points.map((point, index) => `${left + index * (right - left) / (points.length - 1)},${bottom - point.candidate_n / max * (bottom - top)}`).join(" ");
  return <svg className="class-similarity-workload-plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Workload curve">
    {[0, 0.5, 1].map((fraction) => { const y = bottom - fraction * (bottom - top); return <g key={fraction}><line x1={left} x2={right} y1={y} y2={y} className="class-similarity-scatter-grid" /><text x={left - 7} y={y + 4} textAnchor="end" className="class-similarity-scatter-axis">{Math.round(max * fraction)}</text></g>; })}
    <polyline points={coords} className="class-similarity-workload-line" />
    <text x={(left + right) / 2} y={height - 4} textAnchor="middle" className="class-similarity-scatter-axis">Threshold rank · {points.length - 2} distinct scored margins</text>
  </svg>;
}

export default function ClassSimilarityWorkspace({ runId, modelId, unit, records, onReturn, onOpenRecord }: Props) {
  const [k, setK] = useState(5);
  const [analysisSplit, setAnalysisSplit] = useState("all");
  const [metric, setMetric] = useState<ClassSimilarityMetric>("margin");
  const [visibleClasses, setVisibleClasses] = useState<string[]>([]);
  const [axisMode, setAxisMode] = useState<"full" | "data">("full");
  const [onlyCandidates, setOnlyCandidates] = useState(false);
  const [onlyNegative, setOnlyNegative] = useState(false);
  const [confusionFilter, setConfusionFilter] = useState<"tp" | "fp" | "fn" | "tn" | null>(null);
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
  const [evaluationPurpose, setEvaluationPurpose] = useState<"tuning" | "heldout">("tuning");
  const [evaluationPreview, setEvaluationPreview] = useState<ClassSimilarityEvaluationPreview | null>(null);
  const [evaluationBusy, setEvaluationBusy] = useState(false);
  const [evaluationMessage, setEvaluationMessage] = useState("");
  const [reportBusy, setReportBusy] = useState(false);

  const candidateIds = useMemo(() => {
    if (!result || candidateMode === "none") return [];
    return result.items.filter((item) => item.margin !== null && (
      candidateMode === "all_eligible" || item.margin < threshold
    )).map((item) => item.record_id);
  }, [result, candidateMode, threshold]);
  const splits = useMemo(() => [...new Set(records.map((record) => record.split).filter(Boolean))].sort(), [records]);
  const classes = useMemo(() => [...new Set((result?.items || []).map((item) => item.label).filter((label): label is string => Boolean(label)))].sort(), [result]);
  const chartClasses = visibleClasses.length ? visibleClasses : classes;
  const candidateIdSet = useMemo(() => new Set(candidateIds), [candidateIds]);
  const itemById = useMemo(() => new Map((result?.items || []).map((item) => [item.record_id, item])), [result]);
  const visibleMetricValues = (result?.items || []).filter((item) => chartClasses.includes(item.label || "")).map((item) => item[metric]).filter((value): value is number => value !== null && Number.isFinite(value));
  const plotDomain = useMemo<[number, number]>(() => {
    if (axisMode === "full") return metric === "margin" ? [-2, 2] : [-1, 1];
    if (!visibleMetricValues.length) return metric === "margin" ? [-2, 2] : [-1, 1];
    const lo = Math.min(...visibleMetricValues), hi = Math.max(...visibleMetricValues);
    const pad = Math.max((hi - lo) * 0.05, 1e-4);
    return [lo - pad, hi + pad];
  }, [axisMode, metric, result, visibleClasses]);
  const workloadCurve = useMemo(() => {
    const margins = [...new Set((result?.items || []).map((item) => item.margin).filter((value): value is number => value !== null && Number.isFinite(value)))].sort((a, b) => a - b);
    const eligible = result?.items.filter((item) => item.margin !== null && Number.isFinite(item.margin)).length || 0;
    return [{ candidate_n: 0, mode: "none", threshold: null }, ...margins.map((value, index) => ({
      candidate_n: index + 1,
      mode: "threshold", threshold: value,
    })), { candidate_n: eligible, mode: "all_eligible", threshold: null }];
  }, [result]);
  const perClassSummary = useMemo(() => {
    const summary = new Map<string, { label: string; total: number; scored: number; candidates: number }>();
    for (const item of result?.items || []) {
      const label = item.label || "未分類／標籤無效";
      const row = summary.get(label) || { label, total: 0, scored: 0, candidates: 0 };
      row.total += 1;
      if (item.margin !== null) row.scored += 1;
      if (candidateIdSet.has(item.record_id)) row.candidates += 1;
      summary.set(label, row);
    }
    return [...summary.values()].sort((a, b) => a.label.localeCompare(b.label));
  }, [result, candidateIdSet]);
  const rows = useMemo(() => {
    if (!result) return [];
    let filtered = [...result.items];
    if (onlyCandidates) filtered = filtered.filter((item) => candidateIdSet.has(item.record_id));
    if (onlyNegative) filtered = filtered.filter((item) => item.margin !== null && item.margin < 0);
    if (confusionFilter && evaluationPreview?.counts?.confusion_ids) {
      const ids = new Set(evaluationPreview.counts.confusion_ids[confusionFilter] || []);
      filtered = filtered.filter((item) => ids.has(item.record_id));
    }
    return filtered.sort((a, b) => {
      const av = a.margin;
      const bv = b.margin;
      if (av === null && bv === null) return a.index - b.index;
      if (av === null) return 1;
      if (bv === null) return -1;
      return av - bv || a.index - b.index;
    });
  }, [result, onlyCandidates, onlyNegative, confusionFilter, evaluationPreview, candidateIdSet]);
  const eligibleCount = result?.items.filter((item) => item.margin !== null).length || 0;
  const currentPolicy: ClassSimilarityEvaluationPreview["policy"] = {
    mode: candidateMode,
    threshold: candidateMode === "threshold" ? threshold : null,
    comparator: "lt",
    score: "margin",
  };
  const evaluationPreviewIsCurrent = evaluationPreview?.policy.mode === currentPolicy.mode
    && evaluationPreview.policy.threshold === currentPolicy.threshold;
  const heldoutLocked = evaluationPreview?.evaluation?.purpose === "heldout"
    && evaluationPreview.frozen_policy !== null;

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
          if (status.state === "completed" && status.result) {
            setResult(status.result);
            const nextClasses = [...new Set(status.result.items.map((item) => item.label).filter((label): label is string => Boolean(label)))].sort();
            setVisibleClasses(nextClasses.slice(0, 10));
          }
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
        purpose: evaluationPurpose,
        selected_ids: evaluationScope === "selected" ? selectedIds : [],
      });
      saveBlob(blob, "class_similarity_evaluation.json");
      setEvaluationMessage(`${evaluationPurpose} template downloaded. Review verdicts in the JSON, then upload it here.`);
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
      setEvaluationPurpose(preview.evaluation?.purpose || "tuning");
      if (preview.evaluation?.purpose === "heldout") {
        setCandidateMode(preview.policy.mode);
        if (preview.policy.threshold !== null) setThreshold(preview.policy.threshold);
      }
      setEvaluationMessage(preview.evaluation?.purpose === "heldout"
        ? "Heldout verdicts validated against the saved operating point and disjoint source groups."
        : "Tuning review set validated against this Visualize source snapshot.");
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

  async function saveOperatingPoint() {
    if (!jobId || !evaluationPreview?.evaluation || evaluationPreview.evaluation.purpose !== "tuning" || evaluationBusy) return;
    setEvaluationBusy(true);
    setEvaluationMessage("");
    try {
      const saved = await api.saveClassSimilarityOperatingPoint(jobId, currentPolicy);
      setEvaluationPreview(saved);
      setEvaluationMessage("Current tuning policy saved as the operating point for a disjoint heldout review.");
    } catch (reason) {
      setEvaluationMessage(reason instanceof Error ? reason.message : "Could not save this operating point.");
    } finally {
      setEvaluationBusy(false);
    }
  }

  async function restoreOperatingPoint() {
    const frozen = evaluationPreview?.frozen_policy;
    if (!jobId || !frozen || evaluationBusy) return;
    setEvaluationBusy(true);
    setEvaluationMessage("");
    try {
      const policy = frozen.policy;
      setCandidateMode(policy.mode);
      if (policy.threshold !== null) setThreshold(policy.threshold);
      setEvaluationPreview(await api.previewClassSimilarityEvaluation(jobId, policy));
      setEvaluationMessage("Saved operating point restored.");
    } catch (reason) {
      setEvaluationMessage(reason instanceof Error ? reason.message : "Could not restore this operating point.");
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
            <div className="class-similarity-chart-controls">
              <label><span>Visible classes · up to 10</span><select aria-label="Visible classes" multiple size={Math.max(2, Math.min(6, classes.length))} value={visibleClasses} onChange={(event) => {
                const next = Array.from(event.currentTarget.selectedOptions).map((option) => option.value);
                setVisibleClasses(next.slice(0, 10));
              }}>{classes.map((label) => <option key={label} value={label}>{label}</option>)}</select></label>
              <label><span>Y-axis range</span><select aria-label="Class similarity axis range" value={axisMode} onChange={(event) => setAxisMode(event.target.value as "full" | "data")}>
                <option value="full">Full metric range</option><option value="data">Visible data range</option>
              </select></label>
              <button type="button" className="secondary-action" onClick={() => setAxisMode("full")}>Reset full range</button>
            </div>
            <p className="class-similarity-empty">{visibleClasses.length ? `Showing ${visibleClasses.length} class(es); analysis and candidate counts still use all records.` : "No class filter is selected; showing every class."}</p>
            <BoxPlot rows={result.summaries[metric].filter((row) => chartClasses.includes(row.label))} domain={plotDomain} threshold={metric === "margin" && candidateMode === "threshold" ? threshold : null} />
            <ScatterPlot
              items={result.scatter_items[metric].filter((item) => chartClasses.includes(item.label || ""))}
              labels={chartClasses}
              metric={metric}
              domain={plotDomain}
              threshold={metric === "margin" && candidateMode === "threshold" ? threshold : null}
              candidateIds={candidateIdSet}
              selectedIds={selectedIds}
              onInspect={(recordId) => setSelected(itemById.get(recordId) || null)}
              onBoxSelect={(recordIds) => setSelectedIds((current) => [...new Set([...current, ...recordIds])])}
            />
            <details className="class-similarity-class-summary">
              <summary>Per-class candidate summary</summary>
              <table><thead><tr><th>Class</th><th>Records</th><th>Scored</th><th>Candidates</th><th>Candidate ratio</th></tr></thead>
                <tbody>{perClassSummary.map((row) => <tr key={row.label}><td>{row.label}</td><td>{row.total}</td><td>{row.scored}</td><td>{row.candidates}</td><td>{row.scored ? `${(row.candidates / row.scored * 100).toFixed(1)}%` : "—"}</td></tr>)}</tbody>
              </table>
            </details>
          </section>

          <section className="class-similarity-threshold-panel" aria-label="Threshold candidate preview">
            <div className="class-similarity-panel-heading"><div><div className="eyebrow">HUMAN REVIEW QUEUE</div><h2>Threshold preview</h2></div><span>{heldoutLocked ? "Heldout threshold is locked to the saved operating point." : "Precision / Recall unavailable without a verified review set."}</span></div>
            <div className="class-similarity-threshold-controls">
              <label><span>Margin threshold</span><input aria-label="Class similarity threshold" type="number" min={-2} max={2} step="any" value={threshold} disabled={candidateMode !== "threshold" || heldoutLocked} onChange={(event) => { const value = Number(event.target.value); if (Number.isFinite(value) && value >= -2 && value <= 2) setThreshold(value); }} /></label>
              <input aria-label="Class similarity threshold slider" type="range" min={-2} max={2} step={0.01} value={threshold} disabled={candidateMode !== "threshold" || heldoutLocked} onChange={(event) => setThreshold(Number(event.target.value))} />
              <button type="button" className={candidateMode === "none" ? "active" : ""} onClick={() => setCandidateMode("none")} disabled={heldoutLocked}>No candidates</button>
              <button type="button" className={candidateMode === "all_eligible" ? "active" : ""} onClick={() => setCandidateMode("all_eligible")} disabled={heldoutLocked}>All scored</button>
              <button type="button" className="secondary-action" onClick={() => { setCandidateMode("threshold"); setThreshold(0); }} disabled={heldoutLocked}>Reset threshold</button>
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

          <section className="class-similarity-workload-panel" aria-labelledby="class-similarity-workload-title">
            <div className="class-similarity-panel-heading"><div><div className="eyebrow">REVIEW VOLUME</div><h2 id="class-similarity-workload-title">Workload curve · threshold to candidate count</h2></div><span>{eligibleCount.toLocaleString()} scored records</span></div>
            <WorkloadCurve points={workloadCurve} />
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
              <label><span>Review purpose</span><select aria-label="Evaluation purpose" value={evaluationPurpose} onChange={(event) => setEvaluationPurpose(event.target.value as "tuning" | "heldout")}>
                <option value="tuning">Tune threshold</option><option value="heldout">Heldout verification</option>
              </select></label>
              <button type="button" className="secondary-action" onClick={() => void downloadEvaluationTemplate()} disabled={evaluationBusy || (evaluationScope === "selected" && !selectedIds.length)}>Download evaluation template</button>
              <label className="class-similarity-file-input"><span>Upload reviewed JSON</span><input aria-label="Upload class similarity evaluation JSON" type="file" accept="application/json,.json" disabled={evaluationBusy} onChange={(event) => { void importEvaluationFile(event.target.files?.[0]); event.currentTarget.value = ""; }} /></label>
              {evaluationPreview?.evaluation && <button type="button" className="secondary-action" onClick={() => void previewEvaluation()} disabled={evaluationBusy}>{evaluationBusy ? "Calculating…" : "Evaluate current threshold"}</button>}
              {evaluationPreview?.evaluation?.purpose === "tuning" && <button type="button" className="secondary-action" onClick={() => void saveOperatingPoint()} disabled={evaluationBusy || evaluationPreview.counts?.status !== "ready"}>Save operating point</button>}
              {evaluationPreview?.frozen_policy && <button type="button" className="secondary-action" onClick={() => void restoreOperatingPoint()} disabled={evaluationBusy}>Restore saved operating point</button>}
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
              {evaluationPreview.counts?.status === "ready" ? <div className="class-similarity-confusion-drill" aria-label="Filter records by confusion category">{(["tp", "fp", "fn", "tn"] as const).map((key) => <button type="button" className={confusionFilter === key ? "active" : ""} key={key} onClick={() => setConfusionFilter((current) => current === key ? null : key)}>{key.toUpperCase()} {evaluationPreview.counts?.[key]}</button>)}<span>{evaluationPreview.evaluation.purpose}</span></div> : <p className="class-similarity-empty">Precision / Recall unavailable: {evaluationPreview.counts?.unavailable_reason || "review enough frozen items first"}.</p>}
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
              <div className="class-similarity-table-controls">
                <label><input type="checkbox" checked={onlyCandidates} onChange={(event) => setOnlyCandidates(event.target.checked)} /><span>Only candidates</span></label>
                <label><input type="checkbox" checked={onlyNegative} onChange={(event) => setOnlyNegative(event.target.checked)} /><span>Only margin &lt; 0</span></label>
                {confusionFilter && <button type="button" className="secondary-action" onClick={() => setConfusionFilter(null)}>Clear {confusionFilter.toUpperCase()} filter</button>}
              </div>
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
