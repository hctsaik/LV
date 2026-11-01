import { useEffect, useState } from "react";
import { api, type FalsePositiveCase, type FalsePositiveClaim } from "./api";

const claimKinds: Array<{ value: FalsePositiveClaim["kind"]; label: string }> = [
  { value: "fact", label: "確認事實" },
  { value: "hypothesis", label: "待驗證假設" },
  { value: "action", label: "下一步" },
];

type Props = {
  caseData: FalsePositiveCase;
  onCaseUpdated: (next: FalsePositiveCase) => void;
};

function claimLabel(kind: FalsePositiveClaim["kind"]) {
  return claimKinds.find((item) => item.value === kind)?.label || "說明";
}

function evidenceLabel(row: Record<string, unknown>) {
  return [String(row.id || "未知 ID"), String(row.role || "證據"), String(row.text || "")]
    .filter(Boolean)
    .join(" · ");
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export default function FalsePositiveCaseReport({ caseData, onCaseUpdated }: Props) {
  const [kind, setKind] = useState<FalsePositiveClaim["kind"]>("fact");
  const [text, setText] = useState("");
  const [refs, setRefs] = useState<string[]>([]);
  const [editingId, setEditingId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    setRefs(caseData.evidence.map((row) => String(row.id || "")).filter(Boolean).slice(0, 1));
  }, [caseData.id]);

  async function saveClaim(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const response = await api.upsertFalsePositiveClaim(caseData.id, caseData.revision, {
        claim_id: editingId || undefined,
        kind,
        text,
        evidence_ids: refs,
      });
      onCaseUpdated(response.case);
      setEditingId("");
      setKind("fact");
      setText("");
      setRefs(response.case.evidence.map((row) => String(row.id || "")).filter(Boolean).slice(0, 1));
      setNotice("說明已保存並記錄引用證據；內容由工程師撰寫，不是系統自動診斷。");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "說明保存失敗。請重新載入案件後再試。 ");
    } finally {
      setBusy(false);
    }
  }

  async function updateReplyStatus(status: "draft" | "replied") {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const response = await api.setFalsePositiveReplyStatus(caseData.id, caseData.revision, status);
      onCaseUpdated(response.case);
      setNotice(status === "replied" ? "已記錄回覆狀態；根因仍獨立顯示。" : "已將回覆狀態改回草稿。 ");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "回覆狀態保存失敗。請重新載入案件後再試。");
    } finally {
      setBusy(false);
    }
  }

  async function exportReport(format: "md" | "json") {
    setBusy(true);
    setError("");
    try {
      const blob = await api.falsePositiveCaseReport(caseData.id, format);
      downloadBlob(blob, `${caseData.id}.${format}`);
      setNotice(format === "md" ? "Markdown 案件說明已匯出。" : "JSON 案件快照已匯出。");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "報告匯出失敗。");
    } finally {
      setBusy(false);
    }
  }

  function editClaim(claim: FalsePositiveClaim) {
    setEditingId(claim.id);
    setKind(claim.kind);
    setText(claim.text);
    setRefs([...claim.refs]);
    setError("");
    setNotice("");
  }

  function cancelEdit() {
    setEditingId("");
    setKind("fact");
    setText("");
    setRefs(caseData.evidence.map((row) => String(row.id || "")).filter(Boolean).slice(0, 1));
  }

  return (
    <section className="fp-card fp-report-card" aria-labelledby="fp-report-title" data-testid="fp-case-report">
      <div className="fp-card-heading">
        <div><div className="eyebrow">CITED CASE REPORT</div><h3 id="fp-report-title">說明與回覆</h3></div>
        <span>根因：{caseData.cause_status === "hypothesis" ? "有待驗證假設" : "未知"}</span>
      </div>
      <p className="fp-boundary-note">分開記錄已確認事實、待驗證假設與下一步；每段都要引用案件證據。工程師說明不是系統確診，外觀相似度也不能證明根因。</p>

      {error && <div className="alert alert-error" role="alert" data-testid="fp-report-error">{error}</div>}
      {notice && <div className="alert alert-info" role="status" data-testid="fp-report-notice">{notice}</div>}

      <div className="fp-report-status-row">
        <label>使用者回覆狀態
          <select
            aria-label="使用者回覆狀態"
            data-testid="fp-reply-status"
            value={caseData.reply_status}
            disabled={busy}
            onChange={(event) => void updateReplyStatus(event.target.value as "draft" | "replied")}
          >
            <option value="draft">草稿</option>
            <option value="replied">已回覆使用者</option>
          </select>
        </label>
        <div className="fp-report-state-summary">
          <span>根因狀態：{caseData.cause_status === "hypothesis" ? "假設待驗證" : "未知"}</span>
          <span>改善評估：{caseData.improvement_status}</span>
        </div>
        <div className="fp-report-downloads">
          <button type="button" className="fp-secondary-button" data-testid="fp-download-report-md" disabled={busy} onClick={() => void exportReport("md")}>下載 Markdown</button>
          <button type="button" className="fp-secondary-button" data-testid="fp-download-report-json" disabled={busy} onClick={() => void exportReport("json")}>下載 JSON 快照</button>
        </div>
      </div>

      <div className="fp-claim-list" aria-label="案件說明清單">
        {caseData.claims.length === 0 ? <p className="fp-report-empty" data-testid="fp-claims-empty">尚未記錄說明；你可以先引用報案資料或已保存的案件證據。</p>
          : caseData.claims.map((claim) => <article className="fp-claim-item" key={claim.id} data-testid={`fp-claim-${claim.id}`}>
            <div>
              <strong>{claimLabel(claim.kind)}</strong>
              {claim.needs_review && <span className="fp-claim-review-warning" role="status">引用內容變更，請重新核對{claim.review_reason ? `：${claim.review_reason}` : "。"}</span>}
            </div>
            <p>{claim.text}</p>
            <small>引用證據：{claim.refs.join("、")}</small>
            <button type="button" className="fp-secondary-button" disabled={busy} data-testid={`fp-edit-claim-${claim.id}`} onClick={() => editClaim(claim)}>編輯此說明</button>
          </article>)}
      </div>

      <form className="fp-claim-form" onSubmit={(event) => void saveClaim(event)}>
        <h4>{editingId ? "編輯案件說明" : "新增案件說明"}</h4>
        <label>說明類型
          <select aria-label="說明類型" data-testid="fp-claim-kind" value={kind} disabled={busy} onChange={(event) => setKind(event.target.value as FalsePositiveClaim["kind"])}>
            {claimKinds.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
          </select>
        </label>
        <label>說明內容
          <textarea aria-label="說明內容" data-testid="fp-claim-text" rows={4} maxLength={10000} value={text} disabled={busy} onChange={(event) => setText(event.target.value)} />
        </label>
        <label>引用案件證據（可多選）
          <select
            multiple
            size={Math.min(Math.max(caseData.evidence.length, 2), 7)}
            aria-label="引用案件證據"
            data-testid="fp-claim-evidence"
            value={refs}
            disabled={busy || caseData.evidence.length === 0}
            onChange={(event) => setRefs(Array.from(event.currentTarget.selectedOptions).map((option) => option.value))}
          >
            {caseData.evidence.map((row, index) => {
              const evidenceId = String(row.id || "");
              return <option key={evidenceId || index} value={evidenceId}>{evidenceLabel(row)}</option>;
            })}
          </select>
        </label>
        <div className="fp-claim-actions">
          <button type="submit" className="run-button" data-testid="fp-save-claim" disabled={busy || !caseData.evidence.length}>{busy ? "正在保存…" : editingId ? "儲存說明修改" : "儲存說明"}</button>
          {editingId && <button type="button" className="fp-secondary-button" disabled={busy} onClick={cancelEdit}>取消編輯</button>}
        </div>
      </form>
    </section>
  );
}
