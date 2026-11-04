import { useMemo, useState } from "react";
import {
  api,
  type LabelQualityResult,
  type RunRecord,
  type VisualizeUnit,
} from "./api";

type Props = {
  runId: string;
  modelId: string;
  unit: VisualizeUnit;
  records: RunRecord[];
  onReturn: () => void;
  onOpenRecord: (index: number) => void;
  onAddSelection: (indices: number[]) => void;
};

export default function LabelQualityWorkspace({
  runId,
  modelId,
  unit,
  records,
  onReturn,
  onOpenRecord,
  onAddSelection,
}: Props) {
  const [k, setK] = useState(10);
  const [result, setResult] = useState<LabelQualityResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const classesPresent = new Set(records.map((record) => record.label).filter(Boolean));
  const ready = records.length >= 3 && classesPresent.size >= 2;

  const separation = useMemo(() => {
    if (!result) return [];
    return result.classes.map((label, row) => {
      const values = result.confusion[row] || [];
      let rivalIndex = -1;
      let rivalShare = -1;
      values.forEach((value, column) => {
        if (column !== row && value > rivalShare) {
          rivalIndex = column;
          rivalShare = value;
        }
      });
      return {
        label,
        separation: values[row] || 0,
        count: result.counts[row] || 0,
        rival: rivalIndex >= 0 ? result.classes[rivalIndex] : "",
        rivalShare: Math.max(0, rivalShare),
      };
    }).sort((left, right) => left.separation - right.separation).slice(0, 5);
  }, [result]);

  const confusionDirections = useMemo(() => {
    if (!result) return [];
    const directions: Array<{ from: string; to: string; share: number }> = [];
    result.classes.forEach((from, row) => result.classes.forEach((to, column) => {
      if (row !== column && (result.confusion[row]?.[column] || 0) > 0) {
        directions.push({ from, to, share: result.confusion[row][column] });
      }
    }));
    return directions.sort((left, right) => right.share - left.share).slice(0, 8);
  }, [result]);

  async function calculate() {
    setLoading(true);
    setError("");
    setMessage("");
    try {
      setResult(await api.createLabelQualityResult(runId, { model_id: modelId, k }));
    } catch (reason) {
      setResult(null);
      setError(reason instanceof Error ? reason.message : "Label-quality analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  function selectSuspected() {
    if (!result?.candidates.length) return;
    const indices = result.candidates.map((item) => item.index);
    onAddSelection(indices);
    setMessage(`Added ${indices.length} suspected ${unit === "object" ? "object(s)" : "record(s)"} to Visualize selection.`);
  }

  return (
    <main className="label-quality-workspace" data-testid="label-quality-workspace" aria-labelledby="label-quality-title">
      <header className="label-quality-header">
        <div>
          <div className="eyebrow">VISUALIZE · LABEL REVIEW</div>
          <h1 id="label-quality-title">類別混淆與疑似標錯</h1>
          <p>使用目前 run 的原始高維 embedding 計算 k 近鄰。這是人工覆核線索，不代表自動改標或瑕疵定位。</p>
        </div>
        <button type="button" className="secondary-action" onClick={onReturn}>Return to Visualize</button>
      </header>

      <section className="label-quality-controls" aria-label="Label quality settings">
        <label><span>Model</span><input aria-label="Label quality model" value={modelId} readOnly /></label>
        <label><span>Record type</span><input aria-label="Label quality record type" value={unit === "object" ? "Objects" : "Images"} readOnly /></label>
        <label><span>Neighbors per record</span><input aria-label="Label quality K" type="number" min={1} max={20} value={k} onChange={(event) => setK(Math.max(1, Math.min(20, Number(event.target.value) || 1)))} disabled={loading} /></label>
        <button type="button" className="primary-action" onClick={() => void calculate()} disabled={!ready || loading} data-testid="label-quality-calculate">
          {loading ? "Calculating…" : result ? "Recalculate" : "Calculate label quality"}
        </button>
      </section>

      {!ready && <p className="label-quality-notice" data-testid="label-quality-insufficient">At least three records from two classes are required.</p>}
      {error && <div className="label-quality-error" role="alert">{error}</div>}
      {message && <div className="label-quality-notice" role="status">{message}</div>}

      {result && <>
        <section className="label-quality-summary" data-testid="label-quality-summary" aria-label="Label quality summary">
          <span><strong>{result.record_count}</strong> records</span>
          <span><strong>{result.classes.length}</strong> classes</span>
          <span><strong>{result.mismatch_count}</strong> suspected label issue(s)</span>
          <span>{result.model_id} · K={result.k} · {result.unit}</span>
        </section>

        <div className="label-quality-results-grid">
          <section className="label-quality-matrix-panel" aria-labelledby="label-quality-matrix-title">
            <div className="label-quality-panel-heading">
              <div><div className="eyebrow">FULL-D NEIGHBOR DISTRIBUTION</div><h2 id="label-quality-matrix-title">類別混淆／分離度</h2></div>
              <span>Each row is the current label; columns show neighbor labels.</span>
            </div>
            <div className="label-quality-table-scroll">
              <table data-testid="label-quality-confusion-matrix">
                <thead><tr><th>Current label</th>{result.classes.map((label) => <th key={label}>{label}</th>)}</tr></thead>
                <tbody>{result.classes.map((label, row) => <tr key={label}>
                  <th scope="row">{label} <small>{result.counts[row]} records</small></th>
                  {result.classes.map((neighborLabel, column) => {
                    const value = result.confusion[row]?.[column] || 0;
                    const alpha = 0.08 + value * 0.72;
                    return <td key={neighborLabel} title={`${label} neighbors labeled ${neighborLabel}: ${(value * 100).toFixed(1)}%`} style={{ backgroundColor: `rgba(73, 118, 208, ${alpha})` }} data-diagonal={row === column}>
                      {(value * 100).toFixed(1)}%
                    </td>;
                  })}
                </tr>)}</tbody>
              </table>
            </div>
          </section>

          <section className="label-quality-confusion-panel" aria-label="Most confused classes">
            <div className="label-quality-panel-heading"><div><div className="eyebrow">CLASS SEPARATION</div><h2>最難分離的類別</h2></div></div>
            <ul data-testid="label-quality-separation-list">
              {separation.map((item) => <li key={item.label}>
                <span><strong>{item.label}</strong> · {item.count} records</span>
                <span>separation {(item.separation * 100).toFixed(1)}% · most confused with {item.rival} ({(item.rivalShare * 100).toFixed(1)}%)</span>
              </li>)}
            </ul>
            <h3>最常混淆方向</h3>
            <ol data-testid="label-quality-confusion-directions">
              {confusionDirections.map((item) => <li key={`${item.from}-${item.to}`}><strong>{item.from} → {item.to}</strong>: {(item.share * 100).toFixed(1)}%</li>)}
            </ol>
          </section>
        </div>

        <section className="label-quality-candidates" aria-labelledby="label-quality-candidates-title">
          <div className="label-quality-panel-heading">
            <div><div className="eyebrow">HUMAN REVIEW CANDIDATES</div><h2 id="label-quality-candidates-title">疑似標錯</h2></div>
            <span>Showing up to {result.candidates.length} of {result.mismatch_count} records where neighbor-majority differs from the current label.</span>
          </div>
          {result.candidates.length ? <>
            <button type="button" className="primary-action" onClick={selectSuspected} data-testid="label-quality-select-candidates">
              Add top {result.candidates.length} suspected {unit === "object" ? "objects" : "records"} to Visualize selection
            </button>
            <div className="label-quality-candidate-grid" data-testid="label-quality-candidate-grid">
              {result.candidates.map((item) => <article className="label-quality-candidate" data-testid="label-quality-candidate" key={`${item.index}-${item.own}-${item.suggested}`}>
                <img src={`/api/v1/runs/${encodeURIComponent(runId)}/images/${item.index}/thumbnail`} alt={item.filename} loading="lazy" />
                <div><strong>{item.own} → {item.suggested}</strong><span>{item.filename}</span><small>Suggested class in {(item.suggested_frac * 100).toFixed(0)}% of neighbors · current class {(item.own_frac * 100).toFixed(0)}%</small></div>
                <button type="button" className="secondary-action" onClick={() => onOpenRecord(item.index)}>Open in Visualize</button>
              </article>)}
            </div>
          </> : <p className="label-quality-notice" data-testid="label-quality-no-candidates">No record currently has a different neighbor-majority class.</p>}
        </section>
      </>}
    </main>
  );
}
