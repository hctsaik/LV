import { useCallback, useEffect, useMemo, useState } from "react";
import { api, CurationItem } from "./api";

type Props = { onCountChange: (count: number) => void };

const downloadName: Record<"csv" | "json" | "zip", string> = {
  csv: "curation_cart.csv",
  json: "curation_cart.json",
  zip: "curation_cart_images.zip",
};

export default function CurationWorkspace({ onCountChange }: Props) {
  const [items, setItems] = useState<CurationItem[]>([]);
  const [sourceFilter, setSourceFilter] = useState("all");
  const [reasonDraft, setReasonDraft] = useState("");
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
    </main>
  );
}
