import { useEffect, useState, type FormEvent } from "react";
import {
  api,
  type FewshotCandidate,
  type FewshotMonitorQueueItem,
  type FewshotMonitorStatus,
  type FewshotSampleBank,
  type FewshotScan,
  type ModelOption,
} from "./api";

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
  const [exportDirectory, setExportDirectory] = useState("");
  const [copyImages, setCopyImages] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState("");
  const [exportNotice, setExportNotice] = useState("");
  const [monitorDirectory, setMonitorDirectory] = useState("");
  const [monitorK, setMonitorK] = useState(100);
  const [monitorWorkspaceId, setMonitorWorkspaceId] = useState("");
  const [monitorStatus, setMonitorStatus] = useState<FewshotMonitorStatus | null>(null);
  const [monitorWorking, setMonitorWorking] = useState(false);
  const [monitorProgress, setMonitorProgress] = useState("");
  const [monitorError, setMonitorError] = useState("");
  const [monitorNotice, setMonitorNotice] = useState("");
  const [monitorLabels, setMonitorLabels] = useState<Record<string, string>>({});
  const [monitorQueueLimit, setMonitorQueueLimit] = useState(30);

  useEffect(() => {
    if (!modelId && availableModels.length) {
      const preferred = availableModels.find((item) => item.model_id === "dinov2_vits14") || availableModels[0];
      setModelId(preferred.model_id);
    }
  }, [availableModels, modelId]);

  async function chooseFewshotFolder(target: "sample" | "scan" | "export") {
    const reportError = target === "sample" ? setError : target === "scan" ? setScanError : setExportError;
    const reportNotice = target === "sample" ? setNotice : target === "scan" ? setScanNotice : setExportNotice;
    reportError("");
    reportNotice("");
    if (!window.cimHost?.chooseFile) {
      reportNotice("NativeApp 資料夾選擇器尚未連線；可在欄位中輸入本機資料夾路徑。");
      return;
    }
    try {
      const result = await window.cimHost.chooseFile();
      if (result.canceled) return;
      const chosen = result.paths?.length ? result.paths : (result.path ? [result.path] : []);
      if (chosen.length !== 1) throw new Error("請一次選擇一個資料夾，才能綁定 NativeApp 權限。");
      const path = chosen[0];
      if (!path) throw new Error("NativeApp 沒有回傳資料夾路徑。");
      if (target === "export") {
        setExportDirectory(path);
      } else {
        const current = target === "sample" ? sampleFolders : targetFolders;
        const paths = splitLines(current);
        if (!paths.includes(path) && paths.length >= 32) {
          throw new Error("最多可選擇 32 個資料夾。");
        }
        const update = target === "sample" ? setSampleFolders : setTargetFolders;
        update((value) => [...new Set([...splitLines(value), path])].join("\n"));
      }
      if (result.grant_id) api.rememberFolderGrant(path, result.grant_id);
      reportNotice("已選擇 NativeApp 資料夾；送出時會驗證對應的本機權限。");
    } catch (reason) {
      reportError(reason instanceof Error ? reason.message : "NativeApp 資料夾選擇失敗。");
    }
  }

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
        folder_grant_ids: api.folderGrantIds(folders),
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
        folder_grant_ids: api.folderGrantIds(folders),
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

  async function exportCandidates() {
    if (!bank || !scan || !exportDirectory.trim()) return;
    setExporting(true);
    setExportError("");
    setExportNotice("");
    try {
      const result = await api.exportFewshotCandidates({
        sample_bank_id: bank.sample_bank_id,
        scan_id: scan.scan_id,
        accepted_item_ids: selectedCandidates.map((candidate) => candidate.item_id),
        theta: similarityThreshold,
        class_overrides: Object.fromEntries(selectedCandidates.flatMap((candidate) => {
          const label = classOverrides[candidate.item_id];
          return label && label !== candidate.suggested_class ? [[candidate.item_id, label]] : [];
        })),
        output_dir: exportDirectory.trim(),
        output_folder_grant_id: api.folderGrantIds([exportDirectory.trim()])[0] ?? null,
        copy_images: copyImages,
      });
      setExportNotice(
        `已匯出 ${result.objects} 個 YOLO 標註、CSV ${result.csv_rows} 列`
        + (copyImages ? `、影像 ${result.images_copied} 張` : "")
        + ` → ${result.out_dir}`,
      );
    } catch (reason) {
      setExportError(reason instanceof Error ? reason.message : "匯出失敗。請確認輸出資料夾安全且可寫入。");
    } finally {
      setExporting(false);
    }
  }

  async function initializeMonitor() {
    if (!bank || !splitLines(targetFolders).length || !monitorDirectory.trim()) {
      setMonitorError("請先建立樣本集、選擇目標資料夾並指定獨立的監看工作區。");
      return null;
    }
    const initialized = await api.createFewshotMonitorWorkspace({
      sample_bank_id: bank.sample_bank_id,
      target_folders: splitLines(targetFolders),
      workspace_dir: monitorDirectory.trim(),
      k: monitorK,
    });
    setMonitorWorkspaceId(initialized.monitor_workspace_id);
    setMonitorStatus(initialized);
    setMonitorQueueLimit(30);
    return initialized.monitor_workspace_id;
  }

  async function initializeMonitorAction() {
    setMonitorWorking(true);
    setMonitorError("");
    setMonitorNotice("");
    try {
      const workspaceId = await initializeMonitor();
      if (workspaceId) setMonitorNotice("監看工作區已初始化；設定和待標佇列保存在指定資料夾。");
    } catch (reason) {
      setMonitorError(reason instanceof Error ? reason.message : "無法初始化監看工作區。");
    } finally {
      setMonitorWorking(false);
    }
  }

  async function runMonitorScan() {
    setMonitorWorking(true);
    setMonitorError("");
    setMonitorNotice("");
    setMonitorProgress("初始化監看設定…");
    try {
      const workspaceId = await initializeMonitor();
      if (!workspaceId) return;
      const created = await api.createFewshotMonitorScanJob(workspaceId);
      let job = await api.fewshotMonitorScanJob(created.job_id);
      while (["queued", "running", "cancelling"].includes(job.state)) {
        setMonitorProgress(`${job.message}${job.total ? ` · ${job.percent.toFixed(0)}%` : ""}`);
        await pause(400);
        job = await api.fewshotMonitorScanJob(created.job_id);
      }
      if (job.state !== "completed" || !job.result) {
        throw new Error(job.error?.message || job.message || "監看掃描失敗。");
      }
      const status = await api.fewshotMonitorStatus(workspaceId);
      setMonitorStatus(status);
      setMonitorNotice(`監看掃描完成 · 新增 ${job.result.new} 張 · 評分 ${job.result.objects_scored} 個 · 佇列 ${status.queue_len}`);
    } catch (reason) {
      setMonitorError(reason instanceof Error ? reason.message : "監看掃描失敗。");
    } finally {
      setMonitorWorking(false);
      setMonitorProgress("");
    }
  }

  async function labelMonitorItem(item: FewshotMonitorQueueItem, decision: "defect" | "skip") {
    if (!monitorWorkspaceId) return;
    setMonitorError("");
    try {
      const label = monitorLabels[item.item_id] || item.suggested_class;
      await api.labelFewshotMonitorItem(monitorWorkspaceId, {
        item_id: item.item_id,
        decision,
        label,
      });
      const status = await api.fewshotMonitorStatus(monitorWorkspaceId);
      setMonitorStatus(status);
      setMonitorNotice(decision === "defect"
        ? `已採納 ${label}；下次監看掃描會將此項移出待標佇列。`
        : "已略過；此項會保留在後續待標佇列。");
    } catch (reason) {
      setMonitorError(reason instanceof Error ? reason.message : "無法更新監看標記。");
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
          <button type="button" className="secondary-action anomaly-wide" disabled={working}
            onClick={() => void chooseFewshotFolder("sample")}>從 NativeApp 選樣本資料夾</button>
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
              placeholder={'C:\\datasets\\to-review'} value={targetFolders} disabled={scanWorking || adding || exporting}
              onChange={(event) => setTargetFolders(event.target.value)} />
          </label>
          <button type="button" className="secondary-action anomaly-wide" disabled={scanWorking || adding || exporting}
            onClick={() => void chooseFewshotFolder("scan")}>從 NativeApp 選目標資料夾</button>
          <label>前 K 個候選
            <input aria-label="以樣搜樣前 K 個" data-testid="fewshot-top-k" type="number" min={10} max={500} step={10}
              value={topK} disabled={scanWorking || adding || exporting} onChange={(event) => setTopK(Number(event.target.value))} />
          </label>
          <label>粗框信心預篩
            <input aria-label="以樣搜樣粗框信心" data-testid="fewshot-min-confidence" type="number" min={0} max={1} step={0.05}
              value={minProposalConfidence} disabled={scanWorking || adding || exporting}
              onChange={(event) => setMinProposalConfidence(Number(event.target.value))} />
          </label>
          <label>相似度門檻
            <input aria-label="以樣搜樣相似度門檻" data-testid="fewshot-theta" type="number" min={0} max={1} step={0.05}
              value={similarityThreshold} disabled={scanWorking || adding || exporting}
              onChange={(event) => setSimilarityThreshold(Number(event.target.value))} />
          </label>
          <button type="submit" className="run-button anomaly-wide" data-testid="fewshot-scan"
            disabled={scanWorking || adding || exporting || working || !connected || !bank || !splitLines(targetFolders).length}>
            {scanWorking ? "正在海掃…" : "海掃目標資料夾"}
          </button>
        </form>
        {scanProgress && <div className="anomaly-progress" role="status" data-testid="fewshot-scan-progress">{scanProgress}</div>}
        {scanError && <div className="coverage-error" role="alert" data-testid="fewshot-scan-error">{scanError}</div>}
        {addError && <div className="coverage-error" role="alert" data-testid="fewshot-add-error">{addError}</div>}
        {exportError && <div className="coverage-error" role="alert" data-testid="fewshot-export-error">{exportError}</div>}
        {scanNotice && <div className="coverage-notice" role="status" data-testid="fewshot-scan-notice">{scanNotice}</div>}
        {exportNotice && <div className="coverage-notice" role="status" data-testid="fewshot-export-notice">{exportNotice}</div>}
        {scan && <>
          <div className="anomaly-model-status" data-testid="fewshot-scan-summary">
            <strong>{scan.done ? "以樣搜樣掃描完成" : "以樣搜樣掃描未完成"}</strong>
            <span>候選 {scan.candidate_count} · 已評分 {scan.objects_scored} 個 · 相似度 ≥ {similarityThreshold.toFixed(2)} 顯示 {eligibleCandidates.length} 個</span>
            <small>Scan {scan.scan_id} · bank {scan.sample_bank_id}</small>
          </div>
          <div className="anomaly-review-actions" aria-label="以樣搜樣匯出" data-testid="fewshot-export-workspace">
            <label>YOLO＋CSV 輸出資料夾
              <input aria-label="以樣搜樣輸出資料夾" data-testid="fewshot-export-directory" value={exportDirectory}
                disabled={exporting || scanWorking || adding}
                onChange={(event) => setExportDirectory(event.target.value)} />
            </label>
            <button type="button" className="secondary-action" disabled={exporting || scanWorking || adding}
              onClick={() => void chooseFewshotFolder("export")}>選擇輸出資料夾</button>
            <label className="anomaly-curation-choice">
              <input type="checkbox" data-testid="fewshot-export-copy-images" checked={copyImages}
                disabled={exporting || scanWorking || adding}
                onChange={(event) => setCopyImages(event.target.checked)} />
              一併複製影像，輸出成可直接訓練的 YOLO 資料集
            </label>
            <button type="button" className="run-button" data-testid="fewshot-export"
              disabled={exporting || scanWorking || adding || working || !connected || !exportDirectory.trim()}
              onClick={() => void exportCandidates()}>
              {exporting ? "正在輸出…" : "匯出 YOLO＋CSV"}
            </button>
          </div>
          {eligibleCandidates.length > 0 && <>
            <div className="anomaly-review-actions">
              <span>{selectedCandidates.length} 顆已勾選 · 未勾選項目不會加入 bank</span>
              <button type="button" className="run-button" data-testid="fewshot-add-selected"
                disabled={adding || exporting || working || scanWorking || !connected || !selectedCandidates.length}
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
                        checked={isAccepted(candidate)} disabled={adding || exporting}
                        onChange={(event) => setAccepted((current) => ({ ...current, [candidate.item_id]: event.target.checked }))} />
                      採納此候選
                    </label>
                    <label>確認類別
                      <select aria-label={`候選類別 ${candidate.filename}`} value={classOverrides[candidate.item_id] || candidate.suggested_class}
                        disabled={adding || exporting} onChange={(event) => setClassOverrides((current) => ({ ...current, [candidate.item_id]: event.target.value }))}>
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

      <section className="anomaly-card" aria-labelledby="fewshot-monitor-title" data-testid="fewshot-monitor-workspace">
        <div className="eyebrow">FEW-SHOT SEARCH · MONITOR</div>
        <h2 id="fewshot-monitor-title">監看新影像與待標佇列</h2>
        <p>監看設定、增量掃描和人工標記沿用本機 `al_service` 工作區；此流程使用目前樣本集，不需建立 anomaly model。</p>
        <div className="anomaly-build-form">
          <label className="anomaly-wide">監看工作區（設定、佇列與人工標記儲存位置）
            <input aria-label="以樣搜樣監看工作區" data-testid="fewshot-monitor-directory" value={monitorDirectory}
              disabled={monitorWorking} onChange={(event) => setMonitorDirectory(event.target.value)} />
          </label>
          <label>每次保留前 K 個待標項目
            <input aria-label="以樣搜樣監看前 K 個" data-testid="fewshot-monitor-top-k" type="number" min={10} max={500} step={10}
              value={monitorK} disabled={monitorWorking}
              onChange={(event) => setMonitorK(Number(event.target.value))} />
          </label>
          <div className="anomaly-review-actions anomaly-wide">
            <button type="button" className="secondary-action" data-testid="fewshot-monitor-init"
              disabled={monitorWorking || !connected || !bank || !splitLines(targetFolders).length || !monitorDirectory.trim()}
              onClick={() => void initializeMonitorAction()}>
              初始化監看
            </button>
            <button type="button" className="run-button" data-testid="fewshot-monitor-scan"
              disabled={monitorWorking || !connected || !bank || !splitLines(targetFolders).length || !monitorDirectory.trim()}
              onClick={() => void runMonitorScan()}>
              {monitorWorking ? "正在監看掃描…" : "立即掃描一次"}
            </button>
            {monitorWorkspaceId && <a className="secondary-action" data-testid="fewshot-monitor-export-profile"
              href={api.fewshotMonitorProfileUrl(monitorWorkspaceId)} download>
              匯出 profile.yaml
            </a>}
          </div>
        </div>
        {monitorProgress && <div className="anomaly-progress" role="status" data-testid="fewshot-monitor-progress">{monitorProgress}</div>}
        {monitorError && <div className="coverage-error" role="alert" data-testid="fewshot-monitor-error">{monitorError}</div>}
        {monitorNotice && <div className="coverage-notice" role="status" data-testid="fewshot-monitor-notice">{monitorNotice}</div>}
        {monitorStatus && <>
          <div className="anomaly-model-status" data-testid="fewshot-monitor-summary">
            <strong>{monitorStatus.last_run ? "監看掃描已執行" : "監看已初始化，尚未執行"}</strong>
            <span>待標佇列 {monitorStatus.queue_len} 項
              {monitorStatus.last_run ? ` · 上次新增 ${monitorStatus.last_run.new || 0} 張 · 評分 ${monitorStatus.last_run.objects_scored || 0} 個` : ""}</span>
            {monitorStatus.last_run?.run_id && <small>Run {monitorStatus.last_run.run_id}</small>}
          </div>
          {monitorStatus.items.length > 0 && <div className="anomaly-object-list" data-testid="fewshot-monitor-queue">
            {monitorStatus.items.slice(0, monitorQueueLimit).map((item) => (
              <article className="anomaly-object" key={`${item.item_id}:${item.index}`} data-testid="fewshot-monitor-queue-item">
                <img className="anomaly-object-preview" src={api.fewshotMonitorPreviewUrl(monitorStatus.monitor_workspace_id, item.index)}
                  alt={`${item.filename} 監看佇列預覽`} loading="lazy" data-testid="fewshot-monitor-preview" />
                <div className="anomaly-object-details">
                  <strong>{item.filename} · 物件 {item.obj_index + 1}</strong>
                  <span>建議 {item.suggested_class} · 相似度 {item.similarity.toFixed(2)}</span>
                  <span>人工狀態：{item.human_decision === "defect"
                    ? `已採納 ${item.human_label}`
                    : item.human_decision === "skip" ? "已略過（下輪仍保留）" : "待確認"}</span>
                  <label>確認類別
                    <select aria-label={`監看候選類別 ${item.filename}`}
                      value={monitorLabels[item.item_id] || item.human_label || item.suggested_class}
                      disabled={monitorWorking}
                      onChange={(event) => setMonitorLabels((current) => ({ ...current, [item.item_id]: event.target.value }))}>
                      {Object.keys(bank?.class_counts || {}).map((label) => <option key={label} value={label}>{label}</option>)}
                    </select>
                  </label>
                  <div className="anomaly-review-actions">
                    <button type="button" className="run-button" data-testid={`fewshot-monitor-accept-${item.index}`}
                      disabled={monitorWorking || !connected}
                      onClick={() => void labelMonitorItem(item, "defect")}>採納</button>
                    <button type="button" className="secondary-action" data-testid={`fewshot-monitor-skip-${item.index}`}
                      disabled={monitorWorking || !connected}
                      onClick={() => void labelMonitorItem(item, "skip")}>略過</button>
                  </div>
                </div>
              </article>
            ))}
            {monitorQueueLimit < monitorStatus.items.length && <button type="button" className="secondary-action"
              data-testid="fewshot-monitor-load-more"
              onClick={() => setMonitorQueueLimit((current) => Math.min(current + 30, monitorStatus.items.length))}>
              載入更多待標項目（+30）
            </button>}
          </div>}
          {monitorStatus.queue_len === 0 && <p data-testid="fewshot-monitor-empty">佇列空，尚無待標項目。</p>}
        </>}
      </section>

      <section className="anomaly-card">
        <div className="eyebrow">MIGRATION SCOPE</div>
        <h2>接下來的 React 步驟</h2>
        <p>React 已支援建立／讀回樣本集、海掃、相似度篩選、候選確認與改類、回加特徵 bank、YOLO＋CSV 匯出，以及本機監看工作區與待標佇列。真實模型 parity 與 NativeApp folder grant 尚待驗證；此處只表示以樣搜樣操作流程已有 fixture 覆蓋，不代表整個 LV React 遷移已完成。</p>
      </section>
    </main>
  );
}
