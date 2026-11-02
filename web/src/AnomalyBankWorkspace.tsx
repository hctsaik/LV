import { useEffect, useMemo, useState } from "react";
import { api, type AnomalyModel, type ModelOption } from "./api";

type Props = { models: ModelOption[]; connected: boolean };

function modelLabel(model: AnomalyModel) {
  const resolution = model.meta.score_mode === "patch" ? `${model.meta.target_res}px patches` : "object score";
  return `${model.model_id} · ${model.meta.model} · ${resolution}`;
}

function splitLines(value: string) {
  return value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
}

function pause(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export default function AnomalyBankWorkspace({ models, connected }: Props) {
  const [foldersText, setFoldersText] = useState("");
  const [modelId, setModelId] = useState("");
  const [scoreMode, setScoreMode] = useState<"patch" | "object">("patch");
  const [targetRes, setTargetRes] = useState(224);
  const [sampleN, setSampleN] = useState(64);
  const [semantic, setSemantic] = useState<"object" | "defect">("object");
  const [nMin, setNMin] = useState(8);
  const [objectSource, setObjectSource] = useState<"yolo" | "whole_image">("yolo");
  const [classNames, setClassNames] = useState("");
  const [savedModels, setSavedModels] = useState<AnomalyModel[]>([]);
  const [selectedSavedId, setSelectedSavedId] = useState("");
  const [activeModel, setActiveModel] = useState<AnomalyModel | null>(null);
  const [working, setWorking] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const availableModels = useMemo(() => models.filter((item) => item.available), [models]);

  async function refreshModels() {
    const response = await api.anomalyModels();
    setSavedModels(response.items);
    setSelectedSavedId((current) => current || response.items[0]?.model_id || "");
    const current = response.items.find((item) => item.active);
    if (current) setActiveModel(current);
  }

  useEffect(() => {
    if (!modelId && availableModels[0]) setModelId(availableModels[0].model_id);
  }, [availableModels, modelId]);

  useEffect(() => {
    void refreshModels().catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : "無法讀取已保存的 Normal Bank。");
    });
  }, []);

  async function build(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const folders = splitLines(foldersText);
    if (!folders.length || !modelId) {
      setError("請輸入至少一個訓練資料夾並選擇模型。");
      return;
    }
    setWorking(true);
    setError("");
    setNotice("");
    setProgress("排入建模工作…");
    try {
      const created = await api.createAnomalyModelJob({
        train_folders: folders,
        model_id: modelId,
        score_mode: scoreMode,
        target_res: targetRes,
        sample_n: sampleN,
        label_semantic: semantic,
        n_min: nMin,
        object_source: objectSource,
        class_names: splitLines(classNames.replaceAll(",", "\n")),
      });
      let job = await api.anomalyJob(created.job_id);
      while (job.state === "queued" || job.state === "running" || job.state === "cancelling") {
        setProgress(`${job.message}${job.total ? ` · ${job.percent.toFixed(0)}%` : ""}`);
        await pause(500);
        job = await api.anomalyJob(created.job_id);
      }
      if (job.state !== "completed" || !job.result) {
        throw new Error(job.error?.message || job.message || "建立 Normal Bank 失敗。");
      }
      setActiveModel(job.result);
      setNotice("模型已建立於本機 worker 記憶體。按「保存模型」後，才能在新 worker 程序重新載入。後續套用偵測流程尚未接入此畫面。 ");
      await refreshModels();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "建立 Normal Bank 失敗。");
    } finally {
      setWorking(false);
      setProgress("");
    }
  }

  async function save() {
    if (!activeModel) return;
    setWorking(true);
    setError("");
    setNotice("");
    try {
      const result = await api.saveAnomalyModel(activeModel.model_id);
      setActiveModel(result.model);
      setNotice("Normal Bank、投影器、few-shot 標記、分類頭與 manifest 已保存至 LV_CACHE_DIR。");
      await refreshModels();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "保存 Normal Bank 失敗。");
    } finally {
      setWorking(false);
    }
  }

  async function load() {
    if (!selectedSavedId) return;
    setWorking(true);
    setError("");
    setNotice("");
    try {
      const result = await api.loadAnomalyModel(selectedSavedId);
      setActiveModel(result.model);
      setNotice("已從 LV_CACHE_DIR 載入模型；建模時的模型、解析度、分數依據與語義已鎖定。");
      await refreshModels();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "載入 Normal Bank 失敗。");
    } finally {
      setWorking(false);
    }
  }

  return (
    <main className="anomaly-workspace" aria-labelledby="anomaly-title" data-testid="anomaly-bank-workspace">
      <section className="anomaly-card">
        <div className="eyebrow">ANOMALY DETECTION · LV-WEB-06</div>
        <h1 id="anomaly-title">瑕疵偵測 Normal Bank</h1>
        <p>選訓練／參考資料夾並建立模型。模型檔只會寫入本機 <code>LV_CACHE_DIR</code>，不會寫回資料集。</p>
        {!connected && <div className="alert alert-error" role="alert">Local worker 尚未連線。</div>}
        {error && <div className="alert alert-error" role="alert" data-testid="anomaly-error">{error}</div>}
        {notice && <div className="alert alert-info" role="status" data-testid="anomaly-notice">{notice}</div>}
        {progress && <div className="anomaly-progress" role="status" aria-live="polite" data-testid="anomaly-progress">{progress}</div>}

        <form className="anomaly-build-form" onSubmit={(event) => void build(event)}>
          <label className="anomaly-wide">訓練／參考資料夾（每行一個本機路徑）
            <textarea aria-label="瑕疵偵測訓練資料夾" data-testid="anomaly-train-folders" rows={3} value={foldersText} disabled={working} onChange={(event) => setFoldersText(event.target.value)} placeholder="C:\\data\\product\\train" />
            <small>YOLO 模式需有 images/ 與 labels/；若資料沒有框，改選整張影像。NativeApp 資料夾權限橋接尚待接線。</small>
          </label>
          <label>特徵模型
            <select aria-label="瑕疵偵測模型" data-testid="anomaly-model" value={modelId} disabled={working || availableModels.length === 0} onChange={(event) => setModelId(event.target.value)}>
              {availableModels.map((model) => <option key={model.model_id} value={model.model_id}>{model.name}</option>)}
            </select>
          </label>
          <label>分數依據
            <select aria-label="瑕疵分數依據" data-testid="anomaly-score-mode" value={scoreMode} disabled={working} onChange={(event) => setScoreMode(event.target.value as "patch" | "object")}>
              <option value="patch">Normal patch bank</option>
              <option value="object">Object embedding density</option>
            </select>
          </label>
          <label>Patch 解析度
            <select aria-label="Normal Bank 解析度" data-testid="anomaly-resolution" value={targetRes} disabled={working || scoreMode === "object"} onChange={(event) => setTargetRes(Number(event.target.value))}>
              {[224, 336, 448].map((value) => <option key={value} value={value}>{value} × {value}</option>)}
            </select>
          </label>
          <label>物件來源
            <select aria-label="瑕疵偵測物件來源" data-testid="anomaly-object-source" value={objectSource} disabled={working} onChange={(event) => setObjectSource(event.target.value as "yolo" | "whole_image")}>
              <option value="yolo">YOLO labels</option>
              <option value="whole_image">整張影像</option>
            </select>
          </label>
          <label>標籤語義
            <select aria-label="瑕疵標籤語義" data-testid="anomaly-label-semantic" value={semantic} disabled={working} onChange={(event) => setSemantic(event.target.value as "object" | "defect")}>
              <option value="object">物件類別（不訓練分類頭）</option>
              <option value="defect">瑕疵類別（資料達門檻時訓練）</option>
            </select>
          </label>
          <label>分類頭最小樣本數
            <input aria-label="分類頭最小樣本數" type="number" min={2} max={10000} value={nMin} disabled={working || semantic !== "defect"} onChange={(event) => setNMin(Number(event.target.value))} />
          </label>
          <label>聚類抽樣上限
            <input aria-label="聚類抽樣上限" type="number" min={1} max={10000} value={sampleN} disabled={working} onChange={(event) => setSampleN(Number(event.target.value))} />
          </label>
          <label className="anomaly-wide">手動類別名稱（選填，逗號或換行分隔）
            <input aria-label="瑕疵偵測手動類別" value={classNames} disabled={working} onChange={(event) => setClassNames(event.target.value)} />
          </label>
          <button type="submit" className="run-button anomaly-wide" data-testid="anomaly-build" disabled={working || !connected || !modelId || !splitLines(foldersText).length}>
            {working ? "正在處理…" : "建立 Normal Bank"}
          </button>
        </form>
      </section>

      <section className="anomaly-card" aria-labelledby="anomaly-persist-title">
        <div className="eyebrow">SAVE / REOPEN</div>
        <h2 id="anomaly-persist-title">模型保存與載入</h2>
        <p>保存內容包含 patch vectors（如有）、固定投影器、few-shot 標記、可選分類頭和 manifest。新 worker 啟動後可從相同的 <code>LV_CACHE_DIR</code> 載入。</p>
        <div className="anomaly-persist-controls">
          <label>已保存模型
            <select aria-label="已保存瑕疵偵測模型" data-testid="anomaly-saved-model" value={selectedSavedId} disabled={working || savedModels.filter((item) => item.saved).length === 0} onChange={(event) => setSelectedSavedId(event.target.value)}>
              {savedModels.filter((item) => item.saved).length === 0 && <option value="">沒有已保存模型</option>}
              {savedModels.filter((item) => item.saved).map((item) => <option key={item.model_id} value={item.model_id}>{modelLabel(item)}</option>)}
            </select>
          </label>
          <button type="button" className="secondary-action" data-testid="anomaly-load" disabled={working || !selectedSavedId || !connected} onClick={() => void load()}>載入模型</button>
          <button type="button" className="run-button" data-testid="anomaly-save" disabled={working || !activeModel || activeModel.saved || !connected} onClick={() => void save()}>{activeModel?.saved ? "模型已保存" : "保存模型"}</button>
        </div>
        {activeModel && <div className="anomaly-model-status" data-testid="anomaly-model-status">
          <strong>{activeModel.source === "loaded" ? "模型已載入" : activeModel.saved ? "模型已保存" : "模型已建立"}</strong>
          <span>{activeModel.meta.model} · {activeModel.meta.score_mode} · {activeModel.meta.target_res} · {activeModel.summary.n_objects} 個物件</span>
          <span>{activeModel.summary.head_status}{activeModel.summary.eligible_classes.length ? ` · ${activeModel.summary.eligible_classes.join(", ")}` : ""}</span>
          <span>Normal Bank {activeModel.summary.has_normal_bank ? "已建立" : "未建立（object score）"} · embedding D={activeModel.meta.obj_dim ?? "—"}</span>
          <small>模型 ID {activeModel.model_id}</small>
        </div>}
      </section>
      <p className="anomaly-boundary" data-testid="anomaly-boundary">目前完成建模與 Normal Bank 保存／載入。套用偵測、風險排序／熱圖、人工標註交接和 Streamlit 數值 parity 尚未完成。</p>
    </main>
  );
}
