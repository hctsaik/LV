import { useEffect, useState, type FormEvent } from "react";
import { api, type FewshotSampleBank, type ModelOption } from "./api";

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
      setNotice("已從 LV_CACHE_DIR 重新讀回樣本集摘要。");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "無法讀回樣本集。");
      setBank(null);
    } finally {
      setWorking(false);
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

      <section className="anomaly-card">
        <div className="eyebrow">MIGRATION SCOPE</div>
        <h2>接下來的 React 步驟</h2>
        <p>本切片只完成建立與保存樣本集。海掃候選、人工確認／改類、候選回加與 YOLO＋CSV 匯出仍待移植；目前尚不能把「以樣搜樣」整體標記為完成。</p>
      </section>
    </main>
  );
}
