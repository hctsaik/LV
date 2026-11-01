import { useEffect, useMemo, useRef, useState } from "react";
import { api, type FalsePositiveCase, type FalsePositiveRegion } from "./api";

type Verdict = FalsePositiveRegion["review"]["verdict"];
type ReviewDraft = { verdict: Verdict; reviewer: string; reason: string };

const verdicts: Array<{ value: Verdict; label: string }> = [
  { value: "pending", label: "待覆核" },
  { value: "normal", label: "確認此區域正常" },
  { value: "defect", label: "確認此區域有瑕疵" },
  { value: "disputed", label: "規格有爭議" },
  { value: "insufficient", label: "資訊不足" },
];

type Props = {
  caseData: FalsePositiveCase;
  onCaseUpdated: (next: FalsePositiveCase) => void;
};

function errorMessage(reason: unknown) {
  return reason instanceof Error ? reason.message : "覆核操作失敗。請重新載入案件後再試。";
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function countReviews(rows: FalsePositiveCase["candidates"]) {
  const active = rows.filter((row) => !row.withdrawn);
  const reviewed = active.filter((row) => row.review.verdict !== "pending").length;
  return { total: active.length, reviewed, pending: active.length - reviewed };
}

export default function FalsePositiveReviewBatches({ caseData, onCaseUpdated }: Props) {
  const [batchId, setBatchId] = useState(caseData.batches.at(-1)?.id || "");
  const [drafts, setDrafts] = useState<Record<string, ReviewDraft>>({});
  const [dirtyDraftIds, setDirtyDraftIds] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const returnFile = useRef<HTMLInputElement>(null);
  const counts = useMemo(() => countReviews(caseData.candidates), [caseData.candidates]);
  const activeBatch = caseData.batches.find((batch) => batch.id === batchId)
    || caseData.batches[caseData.batches.length - 1]
    || null;

  useEffect(() => {
    if (!batchId && caseData.batches.length) setBatchId(caseData.batches.at(-1)?.id || "");
    setDrafts((current) => {
      const next = { ...current };
      for (const candidate of caseData.candidates) {
        if (dirtyDraftIds.has(candidate.id)) continue;
        next[candidate.id] = {
          verdict: candidate.review.verdict,
          reviewer: candidate.review.reviewer,
          reason: candidate.review.reason,
        };
      }
      return next;
    });
  }, [caseData, dirtyDraftIds]);

  function updateDraft(candidateId: string, draft: ReviewDraft) {
    setDrafts((current) => ({ ...current, [candidateId]: draft }));
    setDirtyDraftIds((current) => new Set(current).add(candidateId));
  }

  async function createBatch() {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const response = await api.prepareFalsePositiveReviewBatch(caseData.id, caseData.revision);
      onCaseUpdated(response.case);
      setBatchId(response.batch_id);
      setNotice("覆核批次已保存；這是 review-only handoff，不是訓練匯入。");
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  async function reviewCandidate(candidateId: string) {
    const draft = drafts[candidateId];
    if (!draft) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const updated = await api.reviewFalsePositiveCandidate(caseData.id, candidateId, caseData.revision, draft);
      setDirtyDraftIds((current) => {
        const next = new Set(current);
        next.delete(candidateId);
        return next;
      });
      onCaseUpdated(updated);
      setNotice("此候選的區域覆核已保存；訓練就緒影像仍為 0。");
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  async function exportBatch(format: "json" | "zip") {
    if (!activeBatch) return;
    setBusy(true);
    setError("");
    try {
      const blob = format === "json"
        ? await api.falsePositiveReviewBatchManifest(caseData.id, activeBatch.id)
        : await api.falsePositiveReviewBatchZip(caseData.id, activeBatch.id);
      downloadBlob(blob, `${activeBatch.id}.${format}`);
      setNotice(format === "json" ? "覆核清單已匯出。" : "含圖片的 review-only ZIP 已匯出。");
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  async function returnBatch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!activeBatch) return;
    const file = returnFile.current?.files?.[0];
    if (!file) {
      setError("請先選擇回傳的 JSON 覆核清單。");
      return;
    }
    setBusy(true);
    setError("");
    setNotice("");
    try {
      let manifest: unknown;
      try {
        manifest = JSON.parse(await file.text());
      } catch {
        throw new Error("回傳檔不是有效的 JSON。");
      }
      const response = await api.applyFalsePositiveReviewReturn(
        caseData.id,
        activeBatch.id,
        caseData.revision,
        manifest,
      );
      onCaseUpdated(response.case);
      if (returnFile.current) returnFile.current.value = "";
      setNotice(`已讀回 ${response.applied} 個區域覆核；訓練就緒影像：0。`);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="fp-card fp-batch-card" aria-labelledby="fp-batch-title" data-testid="fp-review-batches">
      <div className="fp-card-heading">
        <div><div className="eyebrow">REVIEW ONLY</div><h3 id="fp-batch-title">覆核批次</h3></div>
        <span>候選 {counts.total} 個區域 · 已覆核 {counts.reviewed} · 待覆核 {counts.pending}</span>
      </div>
      <p className="fp-boundary-note">資料夾標籤與相似度都不是人工判決。覆核數以區域計，獨立影像數另計；讀回不會更新訓練標籤或模型成績。</p>
      {error && <div className="alert alert-error" role="alert" data-testid="fp-batch-error">{error}</div>}
      {notice && <div className="alert alert-info" role="status" data-testid="fp-batch-notice">{notice}</div>}

      <div className="fp-batch-summary">
        <span>候選區域：{counts.total}</span>
        <span>候選影像數依 SHA-256 去重</span>
        <strong>訓練就緒影像：0</strong>
        <button type="button" className="run-button" data-testid="fp-create-batch" disabled={busy || counts.total === 0} onClick={() => void createBatch()}>
          {busy ? "正在保存…" : "建立／連結覆核回合"}
        </button>
      </div>

      {activeBatch && <div className="fp-batch-handoff">
        <label>覆核批次
          <select aria-label="覆核批次" data-testid="fp-batch-select" value={activeBatch.id} disabled={busy} onChange={(event) => setBatchId(event.target.value)}>
            {caseData.batches.slice().reverse().map((batch) => <option key={batch.id} value={batch.id}>{batch.id} · round {batch.round_id}</option>)}
          </select>
        </label>
        <div className="fp-batch-meta">唯一影像 {activeBatch.unique_images} 張 · 建立 revision {activeBatch.case_revision} · {activeBatch.training_export ? "training export" : "review-only"}</div>
        <div className="fp-batch-actions">
          <button type="button" className="fp-secondary-button" data-testid="fp-download-batch-json" disabled={busy} onClick={() => void exportBatch("json")}>下載覆核清單 JSON</button>
          <button type="button" className="fp-secondary-button" data-testid="fp-download-batch-zip" disabled={busy} onClick={() => void exportBatch("zip")}>下載含圖片覆核 ZIP</button>
        </div>
        <form className="fp-batch-return" onSubmit={(event) => void returnBatch(event)}>
          <label>回傳已覆核清單 JSON
            <input ref={returnFile} aria-label="回傳已覆核清單 JSON" type="file" accept="application/json,.json" data-testid="fp-review-return-file" />
          </label>
          <button type="submit" className="run-button" data-testid="fp-apply-review-return" disabled={busy}>驗證並讀回覆核</button>
        </form>
      </div>}

      <div className="fp-candidate-review-list">
        {caseData.candidates.filter((candidate) => !candidate.withdrawn).map((candidate) => {
          const draft = drafts[candidate.id] || {
            verdict: candidate.review.verdict,
            reviewer: candidate.review.reviewer,
            reason: candidate.review.reason,
          };
          const [x1, y1, x2, y2] = candidate.roi;
          return <article className="fp-candidate-review" key={candidate.id} data-testid={`fp-candidate-review-${candidate.id}`}>
            <div
              className="fp-candidate-image-frame"
              style={{ aspectRatio: `${candidate.source.size[0]} / ${candidate.source.size[1]}` }}
            >
              <img src={api.falsePositiveCandidateImageUrl(caseData.id, candidate.id)} alt={`待覆核候選 ${candidate.source.name}`} loading="lazy" />
              <span
                className="fp-candidate-roi-overlay"
                data-testid={`fp-candidate-roi-${candidate.id}`}
                title="候選區域 ROI"
                aria-hidden="true"
                style={{
                  left: `${x1 * 100}%`,
                  top: `${y1 * 100}%`,
                  width: `${(x2 - x1) * 100}%`,
                  height: `${(y2 - y1) * 100}%`,
                }}
              />
            </div>
            <div className="fp-candidate-review-body">
              <strong>{candidate.source.name} · {candidate.reference_category}</strong>
              <small>{candidate.source.sha256.slice(0, 16)}… · ROI {candidate.roi.map((value) => value.toFixed(3)).join(", ")}</small>
              <small>{candidate.reason} · 相似度線索 {candidate.similarity.toFixed(3)}</small>
              <label>覆核結果
                <select aria-label={`候選 ${candidate.source.name} 覆核結果`} value={draft.verdict} disabled={busy} onChange={(event) => updateDraft(candidate.id, { ...draft, verdict: event.target.value as Verdict })}>
                  {verdicts.map((verdict) => <option key={verdict.value} value={verdict.value}>{verdict.label}</option>)}
                </select>
              </label>
              <label>覆核者
                <input aria-label={`候選 ${candidate.source.name} 覆核者`} value={draft.reviewer} disabled={busy} onChange={(event) => updateDraft(candidate.id, { ...draft, reviewer: event.target.value })} />
              </label>
              <label>判斷依據
                <textarea aria-label={`候選 ${candidate.source.name} 判斷依據`} rows={2} value={draft.reason} disabled={busy} onChange={(event) => updateDraft(candidate.id, { ...draft, reason: event.target.value })} />
              </label>
              <button type="button" className="fp-secondary-button" disabled={busy} onClick={() => void reviewCandidate(candidate.id)}>儲存此候選覆核</button>
            </div>
          </article>;
        })}
      </div>
    </section>
  );
}
