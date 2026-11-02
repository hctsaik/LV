import { useEffect, useState, type FormEvent } from "react";
import { api, type FewshotCandidate, type FewshotSampleBank, type FewshotScan, type ModelOption } from "./api";

type Props = { models: ModelOption[]; connected: boolean };

function splitLines(value: string) {
  return value.split(/\r?\n|,/).map((item) => item.trim()).filter(Boolean);
}

function pause(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export default function FewshotSampleBankWorkspace({ models, connected }: Props) {
  const availableModels = models.filter((model) => model.available);
  const [sampleFolders, setSampleFolders] = useState("");
  const [modelId, setModelId] = useState("");
  const [objectSource, setObjectSource] = useState<"yolo" | "whole_image">("yolo");
  const [classNames, setClassNames] = useState("");
  const [bankIdInput, setBankIdInput] = useState("");
  const [bank, setBank] = useState<FewshotSampleBank | null>(null);
  const [working, setWorking] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [targetFolders, setTargetFolders] = useState("");
  const [topK, setTopK] = useState(100);
  const [minProposalConfidence, setMinProposalConfidence] = useState(0);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.3);
  const [scan, setScan] = useState<FewshotScan | null>(null);
  const [scanWorking, setScanWorking] = useState(false);
  const [scanProgress, setScanProgress] = useState("");
  const [scanError, setScanError] = useState("");
  const [scanNotice, setScanNotice] = useState("");
  const [addError, setAddError] = useState("");
  const [accepted, setAccepted] = useState<Record<string, boolean>>({});
  const [classOverrides, setClassOverrides] = useState<Record<string, string>>({});
  const [candidateLimit, setCandidateLimit] = useState(40);
  const [adding, setAdding] = useState(false);

  useEffect(() => {
    if (!modelId && availableModels.length) {
      const preferred = availableModels.find((item) => item.model_id === "dinov2_vits14") || availableModels[0];
      setModelId(preferred.model_id);
    }
  }, [availableModels, modelId]);

  async function build(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const folders = splitLines(sampleFolders);
    if (!folders.length || !modelId) {
      setError("請輸入至少一個樣本資料夾並選擇特徵模型。");
      return;
    }
    setWorking(true);
    setError("");
    setNotice("");
    setProgress("排入樣本集建立工作…");
    setBank(null);
    setScan(null);
    try {
      const created = await api.createFewshotSampleBankJob({
        sample_folders: folders,
        model_id: modelId,
        object_source: objectSource,
        class_names: splitLines(classNames),
      });
      let job = await api.fewshotSampleBankJob(created.job_id);
      while (["queued", "running", "cancelling"].includes(job.state)) {
        setProgress(`${job.message}${job.total ? ` · ${job.percent.toFixed(0)}%` : ""}`);
        await pause(400);
        job = await api.fewshotSampleBankJob(created.job_id);
      }
      if (job.state !== "completed" || !job.result) {
        throw new Error(job.error?.message || job.message || "建立樣本集失敗。");
      }
      setBank(job.result);
      setBankIdInput(job.result.sample_bank_id);
      setNotice("樣本集已保存至 LV_CACHE_DIR，可用樣本集 ID 重新讀回。這個步驟沒有建立 anomaly model。");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "建立樣本集失敗。");
    } finally {
      setWorking(false);
      setProgress("");
    }
  }

  async function reloadBank() {
    const id = bankIdInput.trim();
    if (!id) {
      setError("請輸入樣本集 ID。");
      return;
    }
    setWorking(true);
    setError("");
    setNotice("");
    try {
      const response = await api.getFewshotSampleBank(id);
      setBank(response.sample_bank);
      setModelId(response.sample_bank.model_id);
      setObjectSource(response.sample_bank.object_source);
      setScan(null);
      setNotice("已從 LV_CACHE_DIR 重新讀回樣本集摘要。");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "無法讀回樣本集。");
      setBank(null);
    } finally {
      setWorking(false);
    }
  }

  async function scanTargets(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const folders = splitLines(targetFolders);
    if (!bank || !folders.length) {
      setScanError("請先讀回樣本集並輸入至少一個目標資料夾。");
      return;
    }
    setScanWorking(true);
    setScanError("");
    setScanNotice("");
    setScan(null);
    setAccepted({});
    setClassOverrides({});
    setCandidateLimit(40);
    setScanProgress("排入海掃工作…");
    try {
      const created = await api.createFewshotScanJob({
        sample_bank_id: bank.sample_bank_id,
        target_folders: folders,
        k: topK,
        min_proposal_conf: minProposalConfidence,
      });
      let job = await api.fewshotScanJob(created.job_id);
      while (["queued", "running", "cancelling"].includes(job.state)) {
        setScanProgress(`${job.message}${job.total ? ` · ${job.percent.toFixed(0)}%` : ""}`);
        await pause(400);
        job = await api.fewshotScanJob(created.job_id);
      }
      if (job.state !== "completed" || !job.result) {
        throw new Error(job.error?.message || job.message || "海掃失敗。");
      }
      setScan(job.result);
      setScanNotice(`已評分 ${job.result.objects_scored} 個候選；請逐項確認建議類別與框後再採納。`);
    } catch (reason) {
      setScanError(reason instanceof Error ? reason.message : "海掃失敗。");
    } finally {
      setScanWorking(false);
      setScanProgress("");
    }
  }

  function isAccepted(candidate: FewshotCandidate) {
    return accepted[candidate.item_id] ?? true;
  }

  const eligibleCandidates = scan?.candidates.filter(
    (candidate) => candidate.similarity >= similarityThreshold,
  ) || [];
  const selectedCandidates = eligibleCandidates.filter(isAccepted);

  async function addCandidates() {
    if (!bank || !scan || !selectedCandidates.length) return;
    setAdding(true);
    setAddError("");
    setScanNotice("");
    try {
      const result = await api.addFewshotCandidates({
        sample_bank_id: bank.sample_bank_id,
        scan_id: scan.scan_id,
        item_ids: selectedCandidates.map((candidate) => candidate.item_id),
        theta: similarityThreshold,
        class_overrides: Object.fromEntries(selectedCandidates.flatMap((candidate) => {
          const label = classOverrides[candidate.item_id];
          return label && label !== candidate.suggested_class ? [[candidate.item_id, label]] : [];
        })),
      });
      setBank(result.sample_bank);
      setAccepted((current) => ({
        ...current,
        ...Object.fromEntries(selectedCandidates.map((candidate) => [candidate.item_id, false])),
      }));
      setScanNotice(`已加入 ${result.added} 顆到樣本集；略過 ${result.skipped} 顆。後續海掃會使用更新後的 bank。`);
    } catch (reason) {
      setAddError(reason instanceof Error ? reason.message : "加入樣本集失敗。");
    } finally {
      setAdding(false);
    }
  }

  return (
    <main className="anomaly-workspace" aria-labelledby="fewshot-title" data-testid="fewshot-sample-workspace">
      <section className="anomaly-card">
        <div className="eyebrow">FEW-SHOT SEARCH · STEP 1</div>
        <h1 id="fewshot-title">以樣搜樣：建立樣本集</h1>
        <p>選擇小樣本資料夾並建立本機特徵 bank。這個流程直接使用特徵模型，不必先建立瑕疵偵測模型。</p>
        <form className="anomaly-build-form" onSubmit={(event) => void build(event)}>
          <label className="anomaly-wide">樣本資料夾（每行一個本機路徑）
            <textarea aria-label="以樣搜樣樣本資料夾" data-testid="fewshot-sample-folders" rows={3}
              placeholder={'C:\\datasets\\samples\\set-a'} value={sampleFolders} disabled={working}
              onChange={(event) => setSampleFolders(event.target.value)} />
            <small>YOLO 模式使用影像與標註框；整張影像模式不讀取 labels。</small>
          </label>
          <label>特徵模型
            <select aria-label="以樣搜樣特徵模型" data-testid="fewshot-model" value={modelId}
              disabled={working || availableModels.length === 0}
              onChange={(event) => setModelId(event.target.value)}>
              {availableModels.map((model) => <option key={model.model_id} value={model.model_id}>{model.name}</option>)}
            </select>
          </label>
          <label>樣本來源
            <select aria-label="以樣搜樣樣本來源" data-testid="fewshot-object-source" value={objectSource}
              disabled={working} onChange={(event) => setObjectSource(event.target.value as "yolo" | "whole_image")}>
              <option value="yolo">物件（YOLO 框）</option>
              <option value="whole_image">整張影像</option>
            </select>
          </label>
          <label>類別名稱（選填，逗號或換行分隔）
            <input aria-label="以樣搜樣類別名稱" data-testid="fewshot-class-names" value={classNames}
              disabled={working || objectSource === "whole_image"}
              onChange={(event) => setClassNames(event.target.value)} />
          </label>
          <button type="submit" className="run-button anomaly-wide" data-testid="fewshot-build-bank"
            disabled={working || !connected || !modelId || !splitLines(sampleFolders).length}>
            {working ? "正在處理…" : "建立並保存樣本集"}
          </button>
        </form>
        {progress && <div className="anomaly-progress" role="status" data-testid="fewshot-progress">{progress}</div>}
        {error && <div className="coverage-error" role="alert" data-testid="fewshot-error">{error}</div>}
        {notice && <div className="coverage-notice" role="status" data-testid="fewshot-notice">{notice}</div>}
      </section>

      <section className="anomaly-card" aria-labelledby="fewshot-reopen-title">
        <div className="eyebrow">LOCAL PERSISTENCE</div>
        <h2 id="fewshot-reopen-title">讀回已保存樣本集</h2>
        <p>樣本集資料留在應用程式的 <code>LV_CACHE_DIR</code>。前端只保存不透明 ID，不取得磁碟路徑或樣本來源路徑。</p>
        <div className="anomaly-persist-controls">
          <label>樣本集 ID
            <input aria-label="已保存樣本集 ID" data-testid="fewshot-bank-id" value={bankIdInput}
              disabled={working} onChange={(event) => setBankIdInput(event.target.value)} />
          </label>
          <button type="button" className="secondary-action" data-testid="fewshot-reload-bank"
            disabled={working || !connected || !bankIdInput.trim()} onClick={() => void reloadBank()}>
            從本機讀回
          </button>
        </div>
        {bank && <div className="anomaly-model-status" data-testid="fewshot-bank-summary">
          <strong>樣本集已{notice.startsWith("已從") ? "讀回" : "建立並保存"}</strong>
          <span>{bank.sample_count} 個樣本 · {bank.object_source === "whole_image" ? "整張影像語義" : "YOLO 物件語義"} · {bank.model_id} · {bank.target_res}px</span>
          <span>特徵維度 D={bank.feature_dimension} · {Object.entries(bank.class_counts).map(([label, count]) => `${label} ${count}`).join("、")}</span>
          <small>樣本集 ID {bank.sample_bank_id}</small>
        </div>}
      </section>

      <section className="anomaly-card" aria-labelledby="fewshot-scan-title" data-testid="fewshot-scan-workspace">
        <div className="eyebrow">FEW-SHOT SEARCH · STEP 2</div>
        <h2 id="fewshot-scan-title">海掃候選並確認</h2>
        <p>以樣本集綁定的模型與整圖／YOLO 語義搜尋目標資料夾。勾選代表採納；可調整建議類別，預覽框只用於人工覆核。</p>
        <form className="anomaly-build-form" onSubmit={(event) => void scanTargets(event)}>
          <label className="anomaly-wide">目標資料夾（每行一個本機路徑）
            <textarea aria-label="以樣搜樣目標資料夾" data-testid="fewshot-target-folders" rows={3}
              placeholder={'C:\\datasets\\to-review'} value={targetFolders} disabled={scanWorking || adding}
              onChange={(event) => setTargetFolders(event.target.value)} />
          </label>
          <label>前 K 個候選
            <input aria-label="以樣搜樣前 K 個" data-testid="fewshot-top-k" type="number" min={10} max={500} step={10}
              value={topK} disabled={scanWorking || adding} onChange={(event) => setTopK(Number(event.target.value))} />
          </label>
          <label>粗框信心預篩
            <input aria-label="以樣搜樣粗框信心" data-testid="fewshot-min-confidence" type="number" min={0} max={1} step={0.05}
              value={minProposalConfidence} disabled={scanWorking || adding}
              onChange={(event) => setMinProposalConfidence(Number(event.target.value))} />
          </label>
          <label>相似度門檻
            <input aria-label="以樣搜樣相似度門檻" data-testid="fewshot-theta" type="number" min={0} max={1} step={0.05}
              value={similarityThreshold} disabled={scanWorking || adding}
              onChange={(event) => setSimilarityThreshold(Number(event.target.value))} />
          </label>
          <button type="submit" className="run-button anomaly-wide" data-testid="fewshot-scan"
            disabled={scanWorking || adding || working || !connected || !bank || !splitLines(targetFolders).length}>
            {scanWorking ? "正在海掃…" : "海掃目標資料夾"}
          </button>
        </form>
        {scanProgress && <div className="anomaly-progress" role="status" data-testid="fewshot-scan-progress">{scanProgress}</div>}
        {scanError && <div className="coverage-error" role="alert" data-testid="fewshot-scan-error">{scanError}</div>}
        {addError && <div className="coverage-error" role="alert" data-testid="fewshot-add-error">{addError}</div>}
        {scanNotice && <div className="coverage-notice" role="status" data-testid="fewshot-scan-notice">{scanNotice}</div>}
        {scan && <>
          <div className="anomaly-model-status" data-testid="fewshot-scan-summary">
            <strong>{scan.done ? "以樣搜樣掃描完成" : "以樣搜樣掃描未完成"}</strong>
            <span>候選 {scan.candidate_count} · 已評分 {scan.objects_scored} 個 · 相似度 ≥ {similarityThreshold.toFixed(2)} 顯示 {eligibleCandidates.length} 個</span>
            <small>Scan {scan.scan_id} · bank {scan.sample_bank_id}</small>
          </div>
          {eligibleCandidates.length > 0 && <>
            <div className="anomaly-review-actions">
              <span>{selectedCandidates.length} 顆已勾選 · 未勾選項目不會加入 bank</span>
              <button type="button" className="run-button" data-testid="fewshot-add-selected"
                disabled={adding || working || scanWorking || !connected || !selectedCandidates.length}
                onClick={() => void addCandidates()}>
                {adding ? "正在重新擷取特徵…" : `把已採納的加入樣本集 (${selectedCandidates.length})`}
              </button>
            </div>
            <div className="anomaly-object-list" data-testid="fewshot-candidate-list">
              {eligibleCandidates.slice(0, candidateLimit).map((candidate) => {
                const preview = api.fewshotCandidatePreviewUrl(scan.scan_id, candidate.index);
                const classOptions = Object.keys(bank?.class_counts || {});
                return <article className="anomaly-object" key={candidate.item_id} data-testid="fewshot-candidate"
                  data-item-id={candidate.item_id}>
                  <img className="anomaly-object-preview" src={preview} alt={`${candidate.filename} 候選框預覽`}
                    loading="lazy" data-testid="fewshot-candidate-preview" />
                  <div className="anomaly-object-details">
                    <strong>{candidate.filename} · 物件 {candidate.obj_index + 1}</strong>
                    <span>建議 {candidate.suggested_class} · 相似度 {candidate.similarity.toFixed(2)}</span>
                    <span>框 [ {candidate.bbox.map((value) => value.toFixed(3)).join(", ")} ] · {candidate.reason}</span>
                    <label className="anomaly-curation-choice">
                      <input type="checkbox" aria-label={`採納 ${candidate.filename} ${candidate.suggested_class}`}
                        checked={isAccepted(candidate)} disabled={adding}
                        onChange={(event) => setAccepted((current) => ({ ...current, [candidate.item_id]: event.target.checked }))} />
                      採納此候選
                    </label>
                    <label>確認類別
                      <select aria-label={`候選類別 ${candidate.filename}`} value={classOverrides[candidate.item_id] || candidate.suggested_class}
                        disabled={adding} onChange={(event) => setClassOverrides((current) => ({ ...current, [candidate.item_id]: event.target.value }))}>
                        {classOptions.map((label) => <option key={label} value={label}>{label}</option>)}
                      </select>
                    </label>
                  </div>
                </article>;
              })}
            </div>
            {candidateLimit < eligibleCandidates.length && <button type="button" className="secondary-action"
              data-testid="fewshot-load-more" onClick={() => setCandidateLimit((current) => Math.min(current + 40, eligibleCandidates.length))}>
              載入更多候選（+40，尚有 {eligibleCandidates.length - candidateLimit}）
            </button>}
          </>}
          {eligibleCandidates.length === 0 && <p data-testid="fewshot-no-hits">沒有候選達門檻；可調低相似度或增加樣本。</p>}
        </>}
      </section>

      <section className="anomaly-card">
        <div className="eyebrow">MIGRATION SCOPE</div>
        <h2>接下來的 React 步驟</h2>
        <p>React 已支援建立／讀回樣本集、海掃、相似度篩選、候選確認與改類、回加特徵 bank。YOLO＋CSV 匯出與監看佇列仍待移植；此處只表示搜尋切片已可用，不代表整個「以樣搜樣」已完成。</p>
      </section>
    </main>
  );
}
