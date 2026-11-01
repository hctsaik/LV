import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type FalsePositiveCase,
  type FalsePositiveRegion,
  type FalsePositiveSource,
} from "./api";

const verdicts: Array<{ value: FalsePositiveRegion["review"]["verdict"]; label: string }> = [
  { value: "pending", label: "待覆核" },
  { value: "normal", label: "確認此區域正常" },
  { value: "defect", label: "確認此區域有瑕疵" },
  { value: "disputed", label: "規格有爭議" },
  { value: "insufficient", label: "資訊不足" },
];

const verdictLabels = new Map(verdicts.map((item) => [item.value, item.label]));

type ReviewDraft = {
  verdict: FalsePositiveRegion["review"]["verdict"];
  reviewer: string;
  reason: string;
};

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "案件操作失敗。";
}

async function fileToBase64(file: File): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return window.btoa(binary);
}

function sourceKindLabel(kind: FalsePositiveSource["kind"]) {
  return kind === "screenshot" ? "截圖" : "原圖";
}

function shortHash(value: string) {
  return value.slice(0, 12) + "…" + value.slice(-6);
}

function updateReviewDraft(
  drafts: Record<string, ReviewDraft>,
  item: FalsePositiveRegion,
): ReviewDraft {
  return drafts[item.id] || {
    verdict: item.review.verdict,
    reviewer: item.review.reviewer,
    reason: item.review.reason,
  };
}

export default function FalsePositiveCaseWorkspace() {
  const [cases, setCases] = useState<FalsePositiveCase[]>([]);
  const [selectedCase, setSelectedCase] = useState<FalsePositiveCase | null>(null);
  const [selectedSourceId, setSelectedSourceId] = useState("");
  const [selectedRegionId, setSelectedRegionId] = useState("");
  const [reviewDrafts, setReviewDrafts] = useState<Record<string, ReviewDraft>>({});
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [createSourceKind, setCreateSourceKind] = useState<FalsePositiveSource["kind"]>("screenshot");
  const [newSourceKind, setNewSourceKind] = useState<FalsePositiveSource["kind"]>("original");
  const [newTitle, setNewTitle] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [newModel, setNewModel] = useState("");
  const [newThreshold, setNewThreshold] = useState("");
  const [newRawOutput, setNewRawOutput] = useState("");
  const [regionLabel, setRegionLabel] = useState("");
  const [regionScore, setRegionScore] = useState("");
  const [regionRoi, setRegionRoi] = useState<[string, string, string, string]>(["0", "0", "1", "1"]);
  const createFileRef = useRef<HTMLInputElement>(null);
  const addSourceFileRef = useRef<HTMLInputElement>(null);
  const predictionFileRef = useRef<HTMLInputElement>(null);

  const selectedSource = useMemo(
    () => selectedCase?.sources.find((source) => source.id === selectedSourceId) || selectedCase?.sources[0] || null,
    [selectedCase, selectedSourceId],
  );
  const selectedRegionIndex = selectedCase?.items.findIndex((item) => item.id === selectedRegionId) ?? -1;
  const selectedRegion = selectedRegionIndex >= 0 ? selectedCase?.items[selectedRegionIndex] || null : null;
  const previewRegion = selectedRegion && selectedSource?.id === selectedRegion.source_id
    ? selectedRegion
    : null;
  const selectedReviewDraft = selectedRegion
    ? updateReviewDraft(reviewDrafts, selectedRegion)
    : null;

  async function refreshCases(selectId?: string) {
    const response = await api.falsePositiveCases();
    setCases(response.items);
    if (selectId) {
      const latest = await api.falsePositiveCase(selectId);
      setSelectedCase(latest);
      setSelectedSourceId(latest.sources[0]?.id || "");
      setSelectedRegionId(latest.items[0]?.id || "");
    }
    return response.items;
  }

  useEffect(() => {
    let current = true;
    void api.falsePositiveCases()
      .then((response) => { if (current) setCases(response.items); })
      .catch((reason: unknown) => { if (current) setError(errorMessage(reason)); })
      .finally(() => { if (current) setLoading(false); });
    return () => { current = false; };
  }, []);

  useEffect(() => {
    if (!selectedCase) return;
    setSelectedSourceId((current) => selectedCase.sources.some((source) => source.id === current)
      ? current : selectedCase.sources[0]?.id || "");
    setSelectedRegionId((current) => selectedCase.items.some((item) => item.id === current)
      ? current : selectedCase.items[0]?.id || "");
    setReviewDrafts((current) => {
      const next = { ...current };
      for (const item of selectedCase.items) {
        if (!next[item.id]) {
          next[item.id] = {
            verdict: item.review.verdict,
            reviewer: item.review.reviewer,
            reason: item.review.reason,
          };
        }
      }
      return next;
    });
  }, [selectedCase]);

  async function createCase(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const file = createFileRef.current?.files?.[0];
    if (!file) {
      setError("請先選擇報案圖片或截圖。");
      return;
    }
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const created = await api.createFalsePositiveCase({
        source_name: file.name,
        source_kind: createSourceKind,
        source_base64: await fileToBase64(file),
        title: newTitle,
        description: newDescription,
        external_model_id: newModel,
        threshold: newThreshold,
        raw_output: newRawOutput,
      });
      setSelectedCase(created);
      setSelectedSourceId(created.sources[0]?.id || "");
      setSelectedRegionId("");
      setNewTitle("");
      setNewDescription("");
      setNewModel("");
      setNewThreshold("");
      setNewRawOutput("");
      if (createFileRef.current) createFileRef.current.value = "";
      await refreshCases();
      setNotice("案件已保存；外部模型尚未重播。");
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  async function openCase(caseId: string) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const opened = await api.falsePositiveCase(caseId);
      setSelectedCase(opened);
      setSelectedSourceId(opened.sources[0]?.id || "");
      setSelectedRegionId(opened.items[0]?.id || "");
      setReviewDrafts(Object.fromEntries(opened.items.map((item) => [item.id, {
        verdict: item.review.verdict,
        reviewer: item.review.reviewer,
        reason: item.review.reason,
      }])));
      setNotice(`已讀回案件版本 ${opened.revision}。`);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  async function reloadCase() {
    if (!selectedCase) return;
    setBusy(true);
    setError("");
    try {
      const latest = await api.falsePositiveCase(selectedCase.id);
      setSelectedCase(latest);
      setReviewDrafts((current) => {
        const next = { ...current };
        for (const item of latest.items) {
          next[item.id] = {
            verdict: item.review.verdict,
            reviewer: item.review.reviewer,
            reason: item.review.reason,
          };
        }
        return next;
      });
      await refreshCases();
      setNotice(`已重新載入案件版本 ${latest.revision}。`);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  async function addSource(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedCase) return;
    const file = addSourceFileRef.current?.files?.[0];
    if (!file) {
      setError("請選擇要補入的原圖或截圖。");
      return;
    }
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const updated = await api.addFalsePositiveSource(selectedCase.id, selectedCase.revision, {
        source_name: file.name,
        source_kind: newSourceKind,
        source_base64: await fileToBase64(file),
      });
      setSelectedCase(updated);
      const added = updated.sources[updated.sources.length - 1];
      setSelectedSourceId(added?.id || "");
      if (addSourceFileRef.current) addSourceFileRef.current.value = "";
      await refreshCases();
      setNotice(`${sourceKindLabel(newSourceKind)}已加入案件；原來源仍保留。`);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  async function addRegion(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedCase || !selectedSource) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const roi = regionRoi.map((value) => Number(value));
      const scoreText = regionScore.trim();
      const score = scoreText === "" ? null : Number(scoreText);
      if (scoreText && !Number.isFinite(score)) {
        throw new Error("預測分數必須是有限數值；未知時請留空。");
      }
      const response = await api.addFalsePositiveRegion(selectedCase.id, selectedCase.revision, {
        source_id: selectedSource.id,
        roi,
        label: regionLabel,
        score,
      });
      setSelectedCase(response.case);
      setSelectedRegionId(response.item_id);
      await refreshCases();
      setRegionLabel("");
      setRegionScore("");
      setNotice("爭議區域已加入，狀態為待覆核。");
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  async function importPredictions(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedCase || !selectedSource) return;
    const file = predictionFileRef.current?.files?.[0];
    if (!file) {
      setError("請選擇六欄 YOLO 預測檔。");
      return;
    }
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const response = await api.importFalsePositiveYolo(selectedCase.id, selectedCase.revision, {
        source_id: selectedSource.id,
        filename: file.name,
        content: await file.text(),
      });
      setSelectedCase(response.case);
      setSelectedRegionId(response.case.items[response.case.items.length - response.imported]?.id || "");
      if (predictionFileRef.current) predictionFileRef.current.value = "";
      await refreshCases();
      setNotice(`已匯入 ${response.imported} 個預測區域；全部保持待覆核。`);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  async function saveReview() {
    if (!selectedCase || !selectedRegion || !selectedReviewDraft) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const updated = await api.reviewFalsePositiveRegion(
        selectedCase.id,
        selectedRegion.id,
        selectedCase.revision,
        selectedReviewDraft,
      );
      setSelectedCase(updated);
      setReviewDrafts((current) => ({
        ...current,
        [selectedRegion.id]: { ...selectedReviewDraft },
      }));
      await refreshCases();
      setNotice("此區域覆核已保存；其他區域與整圖標籤不會隨之改變。");
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  function showNewCaseForm() {
    setSelectedCase(null);
    setSelectedSourceId("");
    setSelectedRegionId("");
    setError("");
    setNotice("");
  }

  return (
    <main className="fp-case-workspace" aria-labelledby="fp-case-title">
      <aside className="fp-case-sidebar">
        <div className="eyebrow">FALSE-POSITIVE CASES</div>
        <h1 id="fp-case-title">誤逮案件</h1>
        <p>保存外部報案來源，逐區域覆核並保留每次案件版本。</p>
        <button type="button" className="fp-new-case-button" onClick={showNewCaseForm}>＋ 建立案件</button>
        <div className="fp-case-list-heading">案件列表 <span>{cases.length}</span></div>
        {loading ? <p className="fp-muted">載入案件…</p> : cases.length === 0 ? (
          <p className="fp-muted">尚無已保存案件。</p>
        ) : <div className="fp-case-list" aria-label="案件導航">
          {cases.map((item) => (
            <button
              type="button"
              key={item.id}
              data-testid={`fp-open-case-${item.id}`}
              aria-current={selectedCase?.id === item.id ? "page" : undefined}
              className={selectedCase?.id === item.id ? "active" : ""}
              onClick={() => void openCase(item.id)}
              disabled={busy}
            >
              <strong>{item.title}</strong>
              <span>{item.id} · v{item.revision}</span>
              <small>{item.sources.length} 個來源 · {item.items.length} 個區域</small>
            </button>
          ))}
        </div>}
        <div className="fp-privacy-note">本機保存 · 來源影像使用 SHA-256 身分 · 不修改原始資料集</div>
      </aside>

      <section className="fp-case-main">
        {error && <div className="alert alert-error" data-testid="fp-case-error" role="alert">{error}</div>}
        {notice && <div className="alert alert-info" data-testid="fp-case-notice" role="status">{notice}</div>}

        {!selectedCase ? (
          <section className="fp-card fp-create-card">
            <div className="eyebrow">NEW INVESTIGATION</div>
            <h2>建立誤逮案件</h2>
            <p>先保存可取得的截圖或原圖；來源可在建案後補入。建立案件不會自動重播外部模型。</p>
            <form className="fp-form" onSubmit={(event) => void createCase(event)}>
              <label>案件名稱（可留空）
                <input aria-label="案件名稱（可留空）" value={newTitle} onChange={(event) => setNewTitle(event.target.value)} maxLength={240} />
              </label>
              <label>建立時的影像類型
                <select aria-label="新案件來源類型" value={createSourceKind} onChange={(event) => setCreateSourceKind(event.target.value as FalsePositiveSource["kind"])}>
                  <option value="screenshot">截圖</option>
                  <option value="original">原圖</option>
                </select>
              </label>
              <label>報案圖片或截圖
                <input ref={createFileRef} aria-label="建立案件圖片" type="file" accept="image/png,image/jpeg,image/bmp,image/tiff,image/webp" />
              </label>
              <label className="fp-form-wide">使用者描述／報案理由
                <textarea aria-label="使用者描述／報案理由" value={newDescription} onChange={(event) => setNewDescription(event.target.value)} rows={3} maxLength={10000} />
              </label>
              <label>外部模型版本（未知可留空）
                <input aria-label="外部模型版本" value={newModel} onChange={(event) => setNewModel(event.target.value)} maxLength={240} />
              </label>
              <label>預測門檻（未知可留空）
                <input aria-label="預測門檻" value={newThreshold} onChange={(event) => setNewThreshold(event.target.value)} maxLength={120} />
              </label>
              <label className="fp-form-wide">外部模型原始預測／輸出（可留空）
                <textarea aria-label="外部模型原始預測／輸出" value={newRawOutput} onChange={(event) => setNewRawOutput(event.target.value)} rows={3} maxLength={20000} />
              </label>
              <button className="run-button fp-form-wide" type="submit" disabled={busy}>{busy ? "正在保存…" : "建立案件"}</button>
            </form>
            <p className="fp-boundary-note">截圖含有文字、框線或縮放時，會保留為獨立來源，不會當成外部模型的原始輸入。</p>
          </section>
        ) : (
          <>
            <header className="fp-detail-header">
              <div>
                <div className="eyebrow">CASE {selectedCase.id}</div>
                <h2 data-testid="fp-case-title-readback">{selectedCase.title}</h2>
                <p>版本 {selectedCase.revision} · 外部模型尚未重播 · 逐區域記錄，不產生整圖標籤</p>
              </div>
              <button type="button" className="fp-secondary-button" onClick={() => void reloadCase()} disabled={busy}>重新載入案件</button>
            </header>

            <div className="fp-content-grid">
              <section className="fp-card fp-source-card">
                <div className="fp-card-heading">
                  <div><div className="eyebrow">IMAGE SOURCES</div><h3>案件影像來源</h3></div>
                  <span>{selectedCase.sources.length} 個來源</span>
                </div>
                <div className="fp-source-nav" aria-label="影像來源導航">
                  {selectedCase.sources.map((source) => (
                    <button
                      type="button"
                      key={source.id}
                      data-testid={`fp-source-${source.id}`}
                      aria-pressed={selectedSource?.id === source.id}
                      onClick={() => setSelectedSourceId(source.id)}
                    >
                      <strong>{sourceKindLabel(source.kind)} · {source.name}</strong>
                      <small>{shortHash(source.sha256)}</small>
                    </button>
                  ))}
                </div>
                {selectedSource && <>
                  <figure className="fp-source-preview">
                    <div className="fp-image-stage">
                      <img
                        data-testid="fp-source-image"
                        src={api.falsePositiveSourceImageUrl(selectedCase.id, selectedSource.id)}
                        alt={`${sourceKindLabel(selectedSource.kind)} ${selectedSource.name}`}
                      />
                      {previewRegion && <svg
                        data-testid="fp-selected-region-roi"
                        aria-label="目前爭議區域 ROI"
                        viewBox="0 0 1 1"
                        preserveAspectRatio="none"
                      >
                        <rect
                          x={previewRegion.roi[0]}
                          y={previewRegion.roi[1]}
                          width={previewRegion.roi[2] - previewRegion.roi[0]}
                          height={previewRegion.roi[3] - previewRegion.roi[1]}
                        />
                      </svg>}
                    </div>
                  </figure>
                  <dl className="fp-source-metadata">
                    <div><dt>來源類型</dt><dd>{sourceKindLabel(selectedSource.kind)}</dd></div>
                    <div><dt>影像尺寸</dt><dd>{selectedSource.size[0]} × {selectedSource.size[1]}</dd></div>
                    <div><dt>SHA-256</dt><dd title={selectedSource.sha256}>{shortHash(selectedSource.sha256)}</dd></div>
                  </dl>
                </>}
                <form className="fp-add-source-form" onSubmit={(event) => void addSource(event)}>
                  <label>補入來源類型
                    <select aria-label="補入來源類型" value={newSourceKind} onChange={(event) => setNewSourceKind(event.target.value as FalsePositiveSource["kind"])}>
                      <option value="original">原圖</option>
                      <option value="screenshot">截圖</option>
                    </select>
                  </label>
                  <label>補入原圖或截圖
                    <input ref={addSourceFileRef} aria-label="補入原圖或截圖" type="file" accept="image/png,image/jpeg,image/bmp,image/tiff,image/webp" />
                  </label>
                  <button type="submit" className="fp-secondary-button" disabled={busy}>加入此案件</button>
                </form>
              </section>

              <section className="fp-card fp-region-card">
                <div className="fp-card-heading">
                  <div><div className="eyebrow">REGION REVIEW</div><h3>案件與區域</h3></div>
                  <span data-testid="fp-region-count">{selectedCase.items.length} 個區域</span>
                </div>
                <p className="fp-boundary-note">每個 ROI 獨立記錄 normal／defect／disputed／insufficient；缺少預測或標註不會自動成為正常。</p>
                <form className="fp-prediction-import" onSubmit={(event) => void importPredictions(event)}>
                  <label>匯入來源影像的六欄 YOLO 預測（class cx cy width height confidence）
                    <input ref={predictionFileRef} data-testid="fp-yolo-file" aria-label="YOLO 預測檔" type="file" accept=".txt,text/plain" />
                  </label>
                  <button type="submit" className="fp-secondary-button" disabled={busy || !selectedSource}>匯入 YOLO 預測</button>
                </form>

                <form className="fp-add-region-form" onSubmit={(event) => void addRegion(event)}>
                  <h4>手動加入爭議區域</h4>
                  <label>預測類別／爭議名稱
                    <input aria-label="預測類別／爭議名稱" value={regionLabel} onChange={(event) => setRegionLabel(event.target.value)} maxLength={256} />
                  </label>
                  <label>預測分數（未知可留空）
                    <input aria-label="預測分數（未知可留空）" inputMode="decimal" value={regionScore} onChange={(event) => setRegionScore(event.target.value)} />
                  </label>
                  <div className="fp-roi-grid" aria-label="Normalized ROI">
                    {regionRoi.map((value, index) => (
                      <label key={index}>ROI {(["x1", "y1", "x2", "y2"] as const)[index]}
                        <input aria-label={`ROI ${(["x1", "y1", "x2", "y2"] as const)[index]}`} inputMode="decimal" value={value} onChange={(event) => setRegionRoi((current) => current.map((part, position) => position === index ? event.target.value : part) as [string, string, string, string])} />
                      </label>
                    ))}
                  </div>
                  <button type="submit" className="fp-secondary-button" disabled={busy || !selectedSource}>加入爭議區域</button>
                </form>

                {selectedCase.items.length === 0 ? <div className="fp-empty-regions">尚無爭議區域。可以手動新增，或匯入 YOLO 預測檔。</div> : <>
                  <div className="fp-region-nav" aria-label="區域導航">
                    <button type="button" className="fp-secondary-button" disabled={selectedRegionIndex <= 0} onClick={() => setSelectedRegionId(selectedCase.items[selectedRegionIndex - 1]?.id || "")}>上一區域</button>
                    <span data-testid="fp-region-position">{Math.max(selectedRegionIndex + 1, 1)} / {selectedCase.items.length}</span>
                    <button type="button" className="fp-secondary-button" disabled={selectedRegionIndex < 0 || selectedRegionIndex >= selectedCase.items.length - 1} onClick={() => setSelectedRegionId(selectedCase.items[selectedRegionIndex + 1]?.id || "")}>下一區域</button>
                  </div>
                  <div className="fp-region-tabs" aria-label="選擇案件區域">
                    {selectedCase.items.map((item, index) => (
                      <button
                        type="button"
                        key={item.id}
                        data-testid={`fp-region-tab-${item.id}`}
                        aria-current={item.id === selectedRegionId ? "step" : undefined}
                        onClick={() => setSelectedRegionId(item.id)}
                      >
                        <strong>區域 {index + 1} · {item.label || "未命名"}</strong>
                        <small>{verdictLabels.get(item.review.verdict)}</small>
                      </button>
                    ))}
                  </div>
                  {selectedRegion && selectedReviewDraft && <section className="fp-review-detail" data-testid={`fp-region-review-${selectedRegionIndex}`}>
                    <div className="fp-review-heading">
                      <div><strong>區域 {selectedRegionIndex + 1} · {selectedRegion.label || "未命名"}</strong><small>{selectedRegion.origin === "yolo_prediction_file" ? "外部 YOLO 預測" : "手動建立"}</small></div>
                      <span>{selectedRegion.score === null ? "分數未知" : `分數 ${selectedRegion.score}`}</span>
                    </div>
                    <dl className="fp-region-metadata">
                      <div><dt>來源</dt><dd>{selectedCase.sources.find((source) => source.id === selectedRegion.source_id)?.name || "來源不存在"}</dd></div>
                      <div><dt>Normalized ROI (x1, y1, x2, y2)</dt><dd>{selectedRegion.roi.map((value) => value.toFixed(4)).join(", ")}</dd></div>
                      <div><dt>預測 ID</dt><dd>{selectedRegion.prediction_id || "—"}</dd></div>
                    </dl>
                    <label>覆核結果
                      <select
                        aria-label="覆核結果"
                        data-testid="fp-review-verdict"
                        value={selectedReviewDraft.verdict}
                        onChange={(event) => setReviewDrafts((current) => ({
                          ...current,
                          [selectedRegion.id]: { ...selectedReviewDraft, verdict: event.target.value as ReviewDraft["verdict"] },
                        }))}
                      >
                        {verdicts.map((verdict) => <option key={verdict.value} value={verdict.value}>{verdict.label}</option>)}
                      </select>
                    </label>
                    <label>覆核者
                      <input aria-label="覆核者" data-testid="fp-reviewer" value={selectedReviewDraft.reviewer} onChange={(event) => setReviewDrafts((current) => ({
                        ...current,
                        [selectedRegion.id]: { ...selectedReviewDraft, reviewer: event.target.value },
                      }))} maxLength={256} />
                    </label>
                    <label>判斷依據
                      <textarea aria-label="判斷依據" data-testid="fp-review-reason" value={selectedReviewDraft.reason} onChange={(event) => setReviewDrafts((current) => ({
                        ...current,
                        [selectedRegion.id]: { ...selectedReviewDraft, reason: event.target.value },
                      }))} rows={3} maxLength={4000} />
                    </label>
                    <button type="button" className="run-button" data-testid="fp-save-review" onClick={() => void saveReview()} disabled={busy}>儲存此區域覆核</button>
                  </section>}
                </>}
              </section>
            </div>
          </>
        )}
      </section>
    </main>
  );
}
