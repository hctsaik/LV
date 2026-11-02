import { useCallback, useEffect, useMemo, useState } from "react";
import { api, ActiveLearningRound, CurationHandoffPreview, CurationItem, CurationSubsetOptions, CurationSubsetPreview, LabelingHandoff, LearningCurveAdvice, ReadbackAuditResponse, VisualizationRun } from "./api";

type Props = {
  onCountChange: (count: number) => void;
  currentRunId: string | null;
  onRunUpdated: (run: VisualizationRun) => void;
};

const downloadName: Record<"csv" | "json" | "zip", string> = {
  csv: "curation_cart.csv",
  json: "curation_cart.json",
  zip: "curation_cart_images.zip",
};

export default function CurationWorkspace({ onCountChange, currentRunId, onRunUpdated }: Props) {
  const [items, setItems] = useState<CurationItem[]>([]);
  const [sourceFilter, setSourceFilter] = useState("all");
  const [reasonDraft, setReasonDraft] = useState("");
  const [subsetDestination, setSubsetDestination] = useState("");
  const [subsetMode, setSubsetMode] = useState<CurationSubsetOptions["mode"]>("copy");
  const [subsetOnExists, setSubsetOnExists] = useState<CurationSubsetOptions["on_exists"]>("skip");
  const [subsetPreview, setSubsetPreview] = useState<{ key: string; data: CurationSubsetPreview } | null>(null);
  const [handoffPreview, setHandoffPreview] = useState<CurationHandoffPreview | null>(null);
  const [includeHandled, setIncludeHandled] = useState(false);
  const [newRound, setNewRound] = useState(true);
  const [selectedRoundId, setSelectedRoundId] = useState("");
  const [rounds, setRounds] = useState<ActiveLearningRound[]>([]);
  const [handoffs, setHandoffs] = useState<LabelingHandoff[]>([]);
  const [readbackAudit, setReadbackAudit] = useState<ReadbackAuditResponse | null>(null);
  const [readbackBusyId, setReadbackBusyId] = useState("");
  const [curveMetric, setCurveMetric] = useState("");
  const [curveAdvice, setCurveAdvice] = useState<LearningCurveAdvice["advice"] | null>(null);
  const [metricRoundId, setMetricRoundId] = useState("");
  const [metricName, setMetricName] = useState("");
  const [metricValue, setMetricValue] = useState("0");
  const [probeBusyRoundId, setProbeBusyRoundId] = useState("");
  const [clearConfirmation, setClearConfirmation] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const refresh = useCallback(async () => {
    setError("");
    const response = await api.curationItems();
    setItems(response.items);
    onCountChange(response.total);
    return response;
  }, [onCountChange]);

  useEffect(() => {
    let active = true;
    api.curationItems().then((response) => {
      if (!active) return;
      setItems(response.items);
      onCountChange(response.total);
    }).catch((reason: unknown) => {
      if (active) setError(reason instanceof Error ? reason.message : "Could not load the curation list.");
    }).finally(() => active && setLoading(false));
    return () => { active = false; };
  }, [onCountChange]);

  const sources = useMemo(() => [...new Set(items.map((item) => item.source || "manual"))].sort(), [items]);
  const visibleItems = useMemo(
    () => sourceFilter === "all" ? items : items.filter((item) => item.source === sourceFilter),
    [items, sourceFilter],
  );
  const subsetRequest = useMemo<CurationSubsetOptions>(() => ({
    source: sourceFilter,
    destination: subsetDestination.trim(),
    mode: subsetMode,
    on_exists: subsetOnExists,
  }), [sourceFilter, subsetDestination, subsetMode, subsetOnExists]);
  const subsetKey = useMemo(() => JSON.stringify({
    ...subsetRequest,
    itemIds: visibleItems.map((item) => item.id),
  }), [subsetRequest, visibleItems]);
  const currentSubsetPreview = subsetPreview?.key === subsetKey ? subsetPreview.data : null;
  const visibleItemIds = useMemo(() => visibleItems.map((item) => item.id), [visibleItems]);
  const visibleItemKey = useMemo(() => JSON.stringify(visibleItemIds), [visibleItemIds]);
  const metricNames = useMemo(() => [...new Set(
    rounds.flatMap((round) => round.metrics.map((metric) => metric.name).filter(Boolean)),
  )].sort(), [rounds]);
  const activeCurveMetric = metricNames.includes(curveMetric) ? curveMetric : metricNames[0] || "";
  const curvePoints = useMemo(() => {
    let cumulative = 0;
    return [...rounds].sort((a, b) => a.ts.localeCompare(b.ts)).flatMap((round) => {
      cumulative += round.n_labeled || 0;
      const metric = [...round.metrics].reverse().find((row) => row.name === activeCurveMetric && row.value !== null);
      return metric ? [{ x: cumulative, y: Number(metric.value), roundId: round.round_id }] : [];
    });
  }, [rounds, activeCurveMetric]);

  useEffect(() => {
    if (!visibleItemIds.length) {
      setHandoffPreview(null);
      return;
    }
    let active = true;
    api.previewCurationHandoff(visibleItemIds, includeHandled)
      .then((response) => active && setHandoffPreview(response))
      .catch((reason: unknown) => active && setError(reason instanceof Error ? reason.message : "Could not check the Labeling send set."));
    return () => { active = false; };
  }, [visibleItemKey, includeHandled]);

  useEffect(() => {
    let active = true;
    api.activeLearningRounds().then((response) => {
      if (!active) return;
      setRounds(response.rounds);
      const latestRound = response.rounds[response.rounds.length - 1];
      if (latestRound) {
        setSelectedRoundId((current) => current || latestRound.round_id);
        setMetricRoundId((current) => current || latestRound.round_id);
      }
    }).catch((reason: unknown) => {
      if (active) setError(reason instanceof Error ? reason.message : "Could not load the active-learning rounds.");
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!activeCurveMetric) {
      setCurveAdvice(null);
      return;
    }
    let active = true;
    api.activeLearningCurveAdvice(activeCurveMetric)
      .then((response) => active && setCurveAdvice(response.advice))
      .catch((reason: unknown) => active && setError(reason instanceof Error ? reason.message : "Could not calculate the learning-curve advice."));
    return () => { active = false; };
  }, [activeCurveMetric, rounds]);

  useEffect(() => {
    let active = true;
    Promise.all([api.labelingHandoffs(), api.readbackAudit()]).then(([pending, audit]) => {
      if (!active) return;
      setHandoffs(pending.items);
      setReadbackAudit(audit);
    }).catch((reason: unknown) => {
      if (active) setError(reason instanceof Error ? reason.message : "Could not load Labeling read-back records.");
    });
    return () => { active = false; };
  }, []);

  async function removeItem(itemId: string) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const response = await api.removeCurationItem(itemId);
      setItems(response.items);
      onCountChange(response.total);
      setNotice("Removed 1 item from the curation list.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not remove the curation item.");
    } finally {
      setBusy(false);
    }
  }

  async function applyReason() {
    const reason = reasonDraft.trim();
    if (!reason || !visibleItems.length) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const response = await api.setCurationReason(visibleItems.map((item) => item.id), reason);
      setItems(response.items);
      onCountChange(response.total);
      setNotice(`Added the reason to ${response.updated} item(s).`);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not update the curation reasons.");
    } finally {
      setBusy(false);
    }
  }

  async function clearItems() {
    if (!clearConfirmation) {
      setClearConfirmation(true);
      return;
    }
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const response = await api.clearCuration();
      setItems(response.items);
      onCountChange(response.total);
      setClearConfirmation(false);
      setSourceFilter("all");
      setNotice(`Cleared ${response.cleared} item(s) from the curation list.`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not clear the curation list.");
    } finally {
      setBusy(false);
    }
  }

  async function exportItems(format: "csv" | "json" | "zip") {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const blob = await api.curationExport(format);
      const href = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = downloadName[format];
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(href);
      setNotice(`Exported ${items.length} item(s) as ${format.toUpperCase()}.`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not export the curation list.");
    } finally {
      setBusy(false);
    }
  }

  async function previewSubset() {
    setBusy(true);
    setError("");
    setNotice("");
    setSubsetPreview(null);
    try {
      const response = await api.curationSubsetPreview(subsetRequest);
      setSubsetPreview({ key: subsetKey, data: response });
      setNotice("Export preview is ready. Review missing files and duplicate counts before exporting.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not preview the YOLO subset export.");
    } finally {
      setBusy(false);
    }
  }

  async function exportSubset() {
    if (!currentSubsetPreview) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const response = await api.curationSubsetExport({
        ...subsetRequest,
        preview_token: currentSubsetPreview.preview_token,
      });
      setSubsetPreview(null);
      setNotice(
        `${response.partial ? "Export completed with unresolved items" : "YOLO subset exported"}: `
        + `${response.exported} image(s), ${response.deduplicated} duplicate(s), `
        + `${response.unresolved} unresolved, ${response.skipped} skipped, `
        + `${response.annotations_preserved} original-format annotation(s) preserved. `
        + response.training_warning,
      );
    } catch (reason) {
      setSubsetPreview(null);
      setError(reason instanceof Error ? reason.message : "Could not export the YOLO subset.");
    } finally {
      setBusy(false);
    }
  }

  async function refreshRounds() {
    const response = await api.activeLearningRounds();
    setRounds(response.rounds);
    const latestRound = response.rounds[response.rounds.length - 1];
    if (latestRound) {
      setSelectedRoundId((current) => current || latestRound.round_id);
      setMetricRoundId((current) => current || latestRound.round_id);
    }
    return response.rounds;
  }

  async function refreshHandoffs() {
    const response = await api.labelingHandoffs();
    setHandoffs(response.items);
    return response.items;
  }

  async function refreshReadbackAudit() {
    const response = await api.readbackAudit();
    setReadbackAudit(response);
    return response;
  }

  async function sendToLabeling() {
    if (!visibleItemIds.length || !handoffPreview?.sendable_images) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const response = await api.sendCurationToLabeling({
        item_ids: visibleItemIds,
        include_handled: includeHandled,
        new_round: newRound,
        round_id: newRound ? null : selectedRoundId || null,
      });
      const [refreshedRounds] = await Promise.all([refreshRounds(), refreshHandoffs()]);
      if (response.created_round) {
        const latestRound = refreshedRounds[refreshedRounds.length - 1];
        if (latestRound) {
          setSelectedRoundId(latestRound.round_id);
          setMetricRoundId(latestRound.round_id);
        }
      }
      setHandoffPreview(await api.previewCurationHandoff(visibleItemIds, includeHandled));
      setNotice(
        `Saved ${response.sent} image(s) to the local Labeling inbox · `
        + `${response.deduplicated} duplicate(s) merged · round ${response.round_id}`
        + (response.skipped_handled ? ` · skipped ${response.skipped_handled} previously handled` : ""),
      );
      setIncludeHandled(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not send the curation items to Labeling.");
    } finally {
      setBusy(false);
    }
  }

  async function addRoundMetric() {
    if (!metricRoundId || !metricName.trim()) return;
    const value = Number(metricValue);
    if (!Number.isFinite(value)) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await api.attachRoundMetric(metricRoundId, { name: metricName.trim(), value, source: "manual" });
      await refreshRounds();
      setNotice(`Added ${metricName.trim()} to round ${metricRoundId}.`);
      setMetricName("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not add the round metric.");
    } finally {
      setBusy(false);
    }
  }

  async function runProbe(roundId: string) {
    if (!currentRunId) return;
    setProbeBusyRoundId(roundId);
    setError("");
    setNotice("");
    try {
      const response = await api.runRoundProbe(roundId, currentRunId);
      await refreshRounds();
      setNotice(
        `Probe complete (${response.model_id}): probe_acc=${response.acc.toFixed(3)} · `
        + `macro-F1=${response.macro_f1.toFixed(3)} · train ${response.n_train} / test ${response.n_test} `
        + `· ${response.classes.length} classes.`,
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Probe unavailable for this run.");
    } finally {
      setProbeBusyRoundId("");
    }
  }

  async function applyReadback(handoff: LabelingHandoff) {
    if (!currentRunId || !handoff.n_annotated || handoff.status === "read_back") return;
    setReadbackBusyId(handoff.handoff_id);
    setError("");
    setNotice("");
    try {
      const response = await api.applyLabelingReadback(handoff.handoff_id, currentRunId);
      if (response.n_changed) onRunUpdated(await api.run(currentRunId));
      await Promise.all([refreshHandoffs(), refreshReadbackAudit(), refreshRounds()]);
      setNotice(
        `Applied ${response.n_labeled} labeled image(s) from ${handoff.handoff_id}; `
        + `${response.n_changed} label change(s)${response.complete ? " · batch complete" : " · partial read-back"}`
        + `${response.annotator ? ` · annotator ${response.annotator}` : ""}`
        + (response.orphan_round ? " · no matching round; this handoff remains outside the round timeline" : ""),
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not apply Labeling read-back.");
    } finally {
      setReadbackBusyId("");
    }
  }

  async function exportReadbackAudit() {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const blob = await api.readbackAuditCsv();
      const href = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = "labeling_readback_corrections.csv";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(href);
      setNotice("Exported the Labeling correction audit as CSV.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not export the read-back audit.");
    } finally {
      setBusy(false);
    }
  }

  async function retryLoad() {
    setLoading(true);
    setError("");
    try {
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load the curation list.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="curation-workspace" data-testid="curation-workspace">
      <header className="curation-header">
        <div>
          <div className="eyebrow">CURATED IMAGE SET</div>
          <h1>My curation list</h1>
          <p>{items.length} item(s) · saved locally across analysis runs and worker restarts</p>
        </div>
        <div className="curation-export-actions" aria-label="Export curation list">
          <button type="button" data-testid="curation-export-csv" disabled={busy || !items.length} onClick={() => void exportItems("csv")}>Export CSV</button>
          <button type="button" data-testid="curation-export-json" disabled={busy || !items.length} onClick={() => void exportItems("json")}>Export JSON</button>
          <button type="button" data-testid="curation-export-zip" disabled={busy || !items.length} onClick={() => void exportItems("zip")}>Export ZIP</button>
        </div>
      </header>

      {error && <div className="alert alert-error" role="alert">{error}<button type="button" onClick={() => void retryLoad()}>Retry</button></div>}
      {notice && <div className="alert alert-info" role="status">{notice}</div>}

      <section className="curation-controls" aria-label="Curation list controls">
        <label>
          <span>Filter by source</span>
          <select aria-label="Filter curation by source" value={sourceFilter} onChange={(event) => setSourceFilter(event.target.value)}>
            <option value="all">All sources ({items.length})</option>
            {sources.map((source) => <option key={source} value={source}>{source} ({items.filter((item) => item.source === source).length})</option>)}
          </select>
        </label>
        <label className="curation-reason-control">
          <span>Reason for visible items</span>
          <input aria-label="Reason for visible curation items" maxLength={2000} value={reasonDraft} onChange={(event) => setReasonDraft(event.target.value)} placeholder="Add a review note to this filtered group" />
        </label>
        <button type="button" disabled={busy || !visibleItems.length || !reasonDraft.trim()} onClick={() => void applyReason()}>Apply reason</button>
        {clearConfirmation ? (
          <div className="curation-clear-confirm" role="group" aria-label="Confirm clearing curation list">
            <span>Remove all {items.length} items?</span>
            <button type="button" disabled={busy} onClick={() => void clearItems()}>Confirm clear list</button>
            <button type="button" disabled={busy} onClick={() => setClearConfirmation(false)}>Cancel</button>
          </div>
        ) : (
          <button type="button" className="curation-clear" disabled={busy || !items.length} onClick={() => void clearItems()}>Clear list</button>
        )}
      </section>

      <section className="curation-subset-panel" aria-label="Send curation items to Labeling" data-testid="curation-labeling-handoff">
        <div className="curation-subset-heading">
          <div>
            <h2>Send to Labeling</h2>
            <p>Creates a durable local handoff. Previously sent or labeled images are skipped unless you explicitly include them again.</p>
          </div>
          <span>{visibleItems.length} cart item(s)</span>
        </div>
        <div className="curation-subset-controls">
          <label className="curation-checkbox-control">
            <input
              type="checkbox"
              aria-label="Include previously handled samples"
              checked={includeHandled}
              onChange={(event) => setIncludeHandled(event.target.checked)}
            />
            <span>Include previously sent or labeled samples</span>
          </label>
          <label className="curation-checkbox-control">
            <input
              type="checkbox"
              aria-label="Record as a new round"
              checked={newRound}
              onChange={(event) => setNewRound(event.target.checked)}
            />
            <span>Record as a new active-learning round</span>
          </label>
          {!newRound && rounds.length > 0 && (
            <label>
              <span>Round to continue</span>
              <select aria-label="Round to continue" value={selectedRoundId} onChange={(event) => setSelectedRoundId(event.target.value)}>
                {[...rounds].reverse().map((round) => (
                  <option key={round.round_id} value={round.round_id}>{round.round_id} · {round.strategy || "—"}</option>
                ))}
              </select>
            </label>
          )}
          <div className="curation-subset-buttons">
            <button type="button" className="primary" data-testid="curation-send-labeling" disabled={busy || !visibleItemIds.length || !handoffPreview?.sendable_images || (!newRound && rounds.length > 0 && !selectedRoundId)} onClick={() => void sendToLabeling()}>
              Send {handoffPreview ? `(${handoffPreview.sendable_images})` : ""} to Labeling
            </button>
          </div>
        </div>
        {handoffPreview && (
          <div className="curation-subset-preview" data-testid="curation-labeling-preview" role="status">
            <strong>{handoffPreview.sendable_images} sendable · {handoffPreview.new_images} new · {handoffPreview.handled_images} already handled</strong>
            <span>{handoffPreview.unique_images} unique source image(s) · {handoffPreview.deduplicated} duplicate(s) merged · {handoffPreview.unavailable} unavailable</span>
            {handoffPreview.skipped_handled > 0 && <span>Default send skips {handoffPreview.skipped_handled} already handled image(s). The checkbox above is the explicit override.</span>}
          </div>
        )}
      </section>

      <section className="curation-subset-panel labeling-readback" aria-label="Labeling feedback and read-back" data-testid="labeling-readback-panel">
        <div className="curation-subset-heading">
          <div>
            <h2>Labeling feedback</h2>
            <p>Refresh after the annotator exports results. Read-back matches images by SHA-256, records corrections locally, and never edits the source dataset.</p>
          </div>
          <button type="button" data-testid="refresh-labeling-inbox" className="secondary-action" disabled={busy || !!readbackBusyId} onClick={() => void Promise.all([refreshHandoffs(), refreshReadbackAudit()])}>Refresh inbox</button>
        </div>
        {!currentRunId && <div className="labeling-readback-warning" role="status">Load the matching image-level classifier dataset in Visualize before applying labels.</div>}
        {handoffs.length === 0 ? (
          <div className="curation-empty labeling-readback-empty" data-testid="labeling-inbox-empty">No local Labeling handoffs yet.</div>
        ) : (
          <div className="labeling-handoff-list">
            {handoffs.map((handoff) => (
              <article className="labeling-handoff-row" key={handoff.handoff_id} data-testid="labeling-handoff-row">
                <div>
                  <strong>{handoff.source} · {handoff.task}</strong>
                  <span>{handoff.created_at} · {handoff.n_annotated ?? 0}/{handoff.n_total ?? 0} labeled · {handoff.status}</span>
                  <code>{handoff.handoff_id}</code>
                </div>
                <button
                  type="button"
                  data-testid={`labeling-readback-${handoff.handoff_id}`}
                  disabled={!currentRunId || !handoff.n_annotated || handoff.status === "read_back" || !!readbackBusyId}
                  onClick={() => void applyReadback(handoff)}
                >
                  {readbackBusyId === handoff.handoff_id ? "Applying…" : handoff.status === "read_back" ? "Read back applied" : `Apply read-back (${handoff.n_annotated ?? 0})`}
                </button>
              </article>
            ))}
          </div>
        )}
        {readbackAudit && (
          <div className="labeling-readback-audit" data-testid="labeling-readback-audit">
            <div className="labeling-readback-audit-heading">
              <strong>{readbackAudit.summary.n_images} corrected image(s) · {readbackAudit.summary.n_events} audit event(s)</strong>
              <button type="button" disabled={busy || !readbackAudit.summary.n_events} onClick={() => void exportReadbackAudit()}>Export correction CSV</button>
            </div>
            {readbackAudit.events.length > 0 && (
              <div className="labeling-readback-audit-table-wrap">
                <table className="labeling-readback-audit-table">
                  <thead><tr><th>Image</th><th>Old label</th><th>New label</th><th>Annotator</th><th>Round</th></tr></thead>
                  <tbody>
                    {[...readbackAudit.events].reverse().slice(0, 20).map((event, index) => (
                      <tr key={`${event.ts}-${event.sha256}-${index}`}>
                        <td title={event.sha256}>{event.filename}</td><td>{event.old_label || "—"}</td>
                        <td>{event.new_label}</td><td>{event.annotator || "—"}</td><td>{event.round_id || "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </section>

      <section className="curation-subset-panel" aria-label="Export as YOLO subset">
        <div className="curation-subset-heading">
          <div>
            <h2>Export as YOLO subset</h2>
            <p>Uses the current source filter. Object selections export their original image and full annotations.</p>
          </div>
          <span>{visibleItems.length} selected item(s)</span>
        </div>
        <div className="curation-subset-controls">
          <label className="curation-subset-destination">
            <span>Destination folder</span>
            <input
              aria-label="Subset destination folder"
              data-testid="curation-subset-destination"
              value={subsetDestination}
              onChange={(event) => setSubsetDestination(event.target.value)}
              placeholder="C:\\output\\my_subset"
            />
          </label>
          <label>
            <span>Image handling</span>
            <select aria-label="Subset image handling" value={subsetMode} onChange={(event) => setSubsetMode(event.target.value as CurationSubsetOptions["mode"])}>
              <option value="copy">Copy images (recommended)</option>
              <option value="symlink">Create symlinks</option>
              <option value="manifest-only">Manifest only</option>
            </select>
          </label>
          <label>
            <span>Existing destination files</span>
            <select aria-label="Subset collision policy" value={subsetOnExists} onChange={(event) => setSubsetOnExists(event.target.value as CurationSubsetOptions["on_exists"])}>
              <option value="skip">Skip</option>
              <option value="rename">Rename with a number</option>
              <option value="overwrite">Overwrite</option>
            </select>
          </label>
          <div className="curation-subset-buttons">
            <button type="button" data-testid="curation-subset-preview-button" disabled={busy || !visibleItems.length || !subsetRequest.destination} onClick={() => void previewSubset()}>Preview export</button>
            <button type="button" className="primary" data-testid="curation-subset-export-button" disabled={busy || !currentSubsetPreview || !visibleItems.length} onClick={() => void exportSubset()}>Export subset</button>
          </div>
        </div>
        {currentSubsetPreview && (
          <div className="curation-subset-preview" data-testid="curation-subset-preview" role="status">
            <strong>Preflight: {currentSubsetPreview.available} available · {currentSubsetPreview.unique_images} unique · {currentSubsetPreview.deduplicated} duplicate(s) · {currentSubsetPreview.missing} missing</strong>
            <span>{currentSubsetPreview.label_files} YOLO label file(s) · {currentSubsetPreview.object_items} object selection(s) · separate destination confirmed</span>
            {currentSubsetPreview.missing_filenames.length > 0 && (
              <span>Missing: {currentSubsetPreview.missing_filenames.join(", ")}</span>
            )}
            <span className="curation-subset-warning">This is a deliberately skewed diagnostic subset. Evaluate with an independent hold-out set.</span>
          </div>
        )}
      </section>

      {loading ? <div className="curation-empty" role="status">Loading the saved curation list…</div> : visibleItems.length === 0 ? (
        <div className="curation-empty" data-testid="curation-empty">
          {items.length ? "No items match this source filter." : "The curation list is empty. Select points in Visualize and add them here."}
        </div>
      ) : (
        <section className="curation-grid" aria-label="Curated images">
          {visibleItems.map((item) => (
            <article className="curation-item" key={item.id} data-testid="curation-item" data-item-id={item.id}>
              <img src={api.curationThumbnailUrl(item.id)} alt={item.filename} loading="lazy" />
              <div className="curation-item-content">
                <div className="curation-item-heading"><strong title={item.filename}>{item.filename}</strong><button type="button" aria-label={`Remove ${item.filename} from curation`} disabled={busy} onClick={() => void removeItem(item.id)}>Remove</button></div>
                <div className="curation-item-meta">{item.label || "Unlabelled"} · {item.split || "Unsplit"} · {item.source || "manual"}{item.level === "object" ? ` · object ${(item.obj_index ?? 0) + 1}` : ""}{item.score === null ? "" : ` · ${item.score.toFixed(4)}`}</div>
                <div className="curation-item-sha" title={item.sha256}>SHA-256 {item.sha256}</div>
                <p className="curation-item-reason">{item.reason || "No reason recorded"}</p>
              </div>
            </article>
          ))}
        </section>
      )}

      <section className="active-learning-timeline" aria-label="Active-learning rounds" data-testid="active-learning-timeline">
        <header className="active-learning-heading">
          <div>
            <div className="eyebrow">LABELING FEEDBACK</div>
            <h2>Active-learning rounds</h2>
            <p>Durable local history of sent batches, read-back counts, and manually recorded metrics.</p>
          </div>
          <button type="button" className="secondary-action" disabled={busy} onClick={() => void refreshRounds()}>Refresh rounds</button>
        </header>
        {rounds.length === 0 ? (
          <div className="curation-empty" data-testid="active-learning-empty">No rounds yet. Send a curation batch to Labeling to start one.</div>
        ) : (
          <>
            <form className="active-learning-metric-form" onSubmit={(event) => { event.preventDefault(); void addRoundMetric(); }}>
              <label>
                <span>Round</span>
                <select aria-label="Metric round" value={metricRoundId} onChange={(event) => setMetricRoundId(event.target.value)}>
                  {[...rounds].reverse().map((round) => <option key={round.round_id} value={round.round_id}>{round.round_id}</option>)}
                </select>
              </label>
              <label>
                <span>Metric name</span>
                <input aria-label="Metric name" value={metricName} maxLength={128} onChange={(event) => setMetricName(event.target.value)} placeholder="e.g. accuracy" />
              </label>
              <label>
                <span>Value</span>
                <input aria-label="Metric value" type="number" step="any" value={metricValue} onChange={(event) => setMetricValue(event.target.value)} />
              </label>
              <button type="submit" disabled={busy || !metricRoundId || !metricName.trim() || !Number.isFinite(Number(metricValue))}>Add metric</button>
            </form>
            {metricNames.length > 0 && (
              <div className="active-learning-curve" data-testid="active-learning-curve">
                <div className="active-learning-curve-heading">
                  <strong>Learning curve · cumulative labeled images</strong>
                  <label><span>Metric</span><select aria-label="Learning curve metric" value={activeCurveMetric} onChange={(event) => setCurveMetric(event.target.value)}>{metricNames.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
                </div>
                {curvePoints.length < 2 ? <p>Record this metric in at least two rounds to draw a curve.</p> : (() => {
                  const width = 720;
                  const height = 190;
                  const pad = 28;
                  const xs = curvePoints.map((point) => point.x);
                  const ys = curvePoints.map((point) => point.y);
                  const xMin = Math.min(...xs);
                  const xMax = Math.max(...xs);
                  const yMin = Math.min(...ys);
                  const yMax = Math.max(...ys);
                  const xSpan = xMax - xMin || 1;
                  const ySpan = yMax - yMin || 1;
                  const coords = curvePoints.map((point) => ({
                    ...point,
                    px: pad + ((point.x - xMin) / xSpan) * (width - pad * 2),
                    py: height - pad - ((point.y - yMin) / ySpan) * (height - pad * 2),
                  }));
                  return <svg className="active-learning-curve-chart" data-testid="active-learning-curve-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${activeCurveMetric} by cumulative labeled images`}>
                    <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} />
                    <line x1={pad} y1={pad} x2={pad} y2={height - pad} />
                    <polyline points={coords.map((point) => `${point.px},${point.py}`).join(" ")} />
                    {coords.map((point) => <circle key={point.roundId} cx={point.px} cy={point.py} r="5"><title>{`${point.roundId}: ${point.x} labeled · ${activeCurveMetric}=${point.y}`}</title></circle>)}
                    <text x={width / 2} y={height - 3}>Cumulative labeled images</text>
                    <text x={13} y={height / 2} transform={`rotate(-90 13 ${height / 2})`}>{activeCurveMetric}</text>
                  </svg>;
                })()}
                {curveAdvice && <p className="active-learning-advice" data-testid="active-learning-advice"><strong>停止建議：</strong>{curveAdvice.reason}</p>}
                <p className="active-learning-curve-note">Probe metrics are directional only. Review an independent hold-out set; LV never stops a workflow automatically. You decide when to stop.</p>
              </div>
            )}
            <p className="active-learning-discipline">Probe results need enough labeled examples in every class. A low-sample warning does not add any metric to the ledger. Plateau advice is informational and never stops a workflow automatically.</p>
            <div className="active-learning-cards">
              {[...rounds].reverse().map((round) => (
                <article className="active-learning-card" key={round.round_id} data-testid="active-learning-round">
                  <div><strong>{round.round_id}</strong><span>Strategy: {round.strategy || "—"}</span></div>
                  <p>Sent {round.n_sent} · Read back {round.n_labeled} · Changed {round.n_changed}</p>
                  {round.metrics.length > 0 && <p>Metrics: {round.metrics.map((metric) => `${metric.name}=${metric.value ?? "—"}`).join(" · ")}</p>}
                  <button type="button" data-testid={`active-learning-probe-${round.round_id}`} disabled={!currentRunId || !!probeBusyRoundId} onClick={() => void runProbe(round.round_id)}>
                    {probeBusyRoundId === round.round_id ? "Running probe…" : "Run probe evaluation"}
                  </button>
                </article>
              ))}
            </div>
          </>
        )}
      </section>
    </main>
  );
}
