import { useEffect, useMemo, useState } from "react";
import { api, type AnomalyModel, type AnomalyObject, type AnomalyPrelabelPreview, type AnomalyRun, type ModelOption } from "./api";

type Props = {
  models: ModelOption[];
  connected: boolean;
  onCurationCountChange: (count: number) => void;
};

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

export default function AnomalyBankWorkspace({ models, connected, onCurationCountChange }: Props) {
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
  const [targetFoldersText, setTargetFoldersText] = useState("");
  const [contamination, setContamination] = useState(0.05);
  const [scoreWorking, setScoreWorking] = useState(false);
  const [run, setRun] = useState<AnomalyRun | null>(null);
  const [objects, setObjects] = useState<AnomalyObject[]>([]);
  const [objectOffset, setObjectOffset] = useState(0);
  const [objectTotal, setObjectTotal] = useState(0);
  const [filterVerdict, setFilterVerdict] = useState<"" | "good" | "bad">("");
  const [filterClass, setFilterClass] = useState("");
  const [filterMin, setFilterMin] = useState("");
  const [filterMax, setFilterMax] = useState("");
  const [draftConfirmed, setDraftConfirmed] = useState<Record<string, "good" | "bad" | "">>({});
  const [heatmapIndex, setHeatmapIndex] = useState<number | null>(null);
  const [selectedCuration, setSelectedCuration] = useState<Set<number>>(() => new Set());
  const [curationWorking, setCurationWorking] = useState(false);
  const [prelabelMinConf, setPrelabelMinConf] = useState(0.5);
  const [prelabelClasses, setPrelabelClasses] = useState<string[]>([]);
  const [prelabelOutputDir, setPrelabelOutputDir] = useState("");
  const [prelabelPreview, setPrelabelPreview] = useState<AnomalyPrelabelPreview | null>(null);
  const [prelabelOffset, setPrelabelOffset] = useState(0);
  const [prelabelWorking, setPrelabelWorking] = useState(false);

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
      setRun(null);
      setObjects([]);
      setPrelabelPreview(null);
      setPrelabelClasses(job.result.summary.eligible_classes);
      setDraftConfirmed({});
      setSelectedCuration(new Set());
      setNotice("模型已建立於本機 worker 記憶體。按「保存模型」後，才能在新 worker 程序重新載入。");
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
      setRun(null);
      setObjects([]);
      setPrelabelPreview(null);
      setPrelabelClasses(result.model.summary.eligible_classes);
      setDraftConfirmed({});
      setSelectedCuration(new Set());
      setNotice("已從 LV_CACHE_DIR 載入模型；建模時的模型、解析度、分數依據與語義已鎖定。");
      await refreshModels();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "載入 Normal Bank 失敗。");
    } finally {
      setWorking(false);
    }
  }

  async function loadObjects(nextOffset = 0, runId = run?.run_id) {
    if (!runId) return;
    const page = await api.anomalyRunObjects(runId, {
      offset: nextOffset,
      limit: 40,
      verdict: filterVerdict || undefined,
      class_label: filterClass || undefined,
      score_min: filterMin === "" ? undefined : Number(filterMin),
      score_max: filterMax === "" ? undefined : Number(filterMax),
    });
    setObjects(page.items);
    setObjectOffset(page.offset);
    setObjectTotal(page.total);
  }

  async function scoreTargets(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!activeModel) return;
    const folders = splitLines(targetFoldersText);
    if (!folders.length) {
      setError("請輸入至少一個目標資料夾。");
      return;
    }
    setScoreWorking(true);
    setError("");
    setNotice("");
    setRun(null);
    setObjects([]);
    setPrelabelPreview(null);
    setSelectedCuration(new Set());
    setProgress("排入套用工作…");
    try {
      const created = await api.createAnomalyScoreJob({
        target_folders: folders,
        contamination,
        class_names: splitLines(classNames.replaceAll(",", "\n")),
      });
      let job = await api.anomalyScoreJob(created.job_id);
      while (job.state === "queued" || job.state === "running" || job.state === "cancelling") {
        setProgress(`${job.message}${job.total ? ` · ${job.percent.toFixed(0)}%` : ""}`);
        await pause(500);
        job = await api.anomalyScoreJob(created.job_id);
      }
      if (job.state !== "completed" || !job.result) {
        throw new Error(job.error?.message || job.message || "套用瑕疵偵測模型失敗。");
      }
      setRun(job.result);
      setPrelabelClasses(activeModel.summary.eligible_classes);
      setPrelabelPreview(null);
      setDraftConfirmed({});
      setHeatmapIndex(null);
      await loadObjects(0, job.result.run_id);
      setNotice("已用目前載入的模型評分目標資料夾。此結果只保存於目前 worker；重算門檻不會重建 Normal Bank。");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "套用瑕疵偵測模型失敗。");
    } finally {
      setScoreWorking(false);
      setProgress("");
    }
  }

  async function reapply() {
    if (!run) return;
    setWorking(true);
    setError("");
    setNotice("");
    try {
      const updated = await api.reapplyAnomalyRun(run.run_id, {
        expected_revision: run.revision,
        contamination,
        confirmed: draftConfirmed,
      });
      setRun(updated);
      setPrelabelPreview(null);
      setDraftConfirmed({});
      setSelectedCuration(new Set());
      await loadObjects(0, updated.run_id);
      setNotice("已按確認標籤與污染率重新計算門檻和排序；Normal Bank 未變更。");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "重算瑕疵標籤失敗。");
    } finally {
      setWorking(false);
    }
  }

  async function addSelectedToCuration() {
    if (!run || selectedCuration.size === 0) return;
    setCurationWorking(true);
    setError("");
    setNotice("");
    try {
      const result = await api.addAnomalyCuration(run.run_id, {
        expected_revision: run.revision,
        indices: [...selectedCuration],
      });
      setSelectedCuration(new Set());
      onCurationCountChange(result.total);
      setNotice(`已加入策展清單 ${result.added} 個原圖候選；重複略過 ${result.skipped} 個。可到 Curation 匯出原始影像。`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "加入策展清單失敗。");
    } finally {
      setCurationWorking(false);
    }
  }

  async function previewPrelabels(offset = 0) {
    if (!run) return;
    setPrelabelWorking(true);
    setError("");
    setNotice("");
    try {
      const result = await api.previewAnomalyPrelabels(run.run_id, {
        expected_revision: run.revision,
        min_conf: prelabelMinConf,
        target_classes: prelabelClasses,
        offset,
        limit: 40,
      });
      setPrelabelPreview(result);
      setPrelabelOffset(result.offset);
    } catch (reason) {
      setPrelabelPreview(null);
      setError(reason instanceof Error ? reason.message : "計算瑕疵預標失敗。");
    } finally {
      setPrelabelWorking(false);
    }
  }

  async function exportPrelabels() {
    if (!run || !prelabelPreview || !prelabelOutputDir.trim()) return;
    setPrelabelWorking(true);
    setError("");
    setNotice("");
    try {
      const result = await api.exportAnomalyPrelabels(run.run_id, {
        expected_revision: run.revision,
        min_conf: prelabelMinConf,
        target_classes: prelabelClasses,
        output_dir: prelabelOutputDir.trim(),
      });
      setNotice(`預標匯出完成：接受 ${result.accepted} 個物件，寫入 ${result.written} 個標註檔。來源資料未修改。`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "預標匯出失敗。");
    } finally {
      setPrelabelWorking(false);
    }
  }

  const canPrelabel = Boolean(run && activeModel?.summary.has_head && activeModel.model_id === run.model_bundle_id);

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
          <button type="submit" className="run-button anomaly-wide" data-testid="anomaly-build" disabled={working || scoreWorking || !connected || !modelId || !splitLines(foldersText).length}>
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
            <select aria-label="已保存瑕疵偵測模型" data-testid="anomaly-saved-model" value={selectedSavedId} disabled={working || scoreWorking || savedModels.filter((item) => item.saved).length === 0} onChange={(event) => setSelectedSavedId(event.target.value)}>
              {savedModels.filter((item) => item.saved).length === 0 && <option value="">沒有已保存模型</option>}
              {savedModels.filter((item) => item.saved).map((item) => <option key={item.model_id} value={item.model_id}>{modelLabel(item)}</option>)}
            </select>
          </label>
          <button type="button" className="secondary-action" data-testid="anomaly-load" disabled={working || scoreWorking || !selectedSavedId || !connected} onClick={() => void load()}>載入模型</button>
          <button type="button" className="run-button" data-testid="anomaly-save" disabled={working || scoreWorking || !activeModel || activeModel.saved || !connected} onClick={() => void save()}>{activeModel?.saved ? "模型已保存" : "保存模型"}</button>
        </div>
        {activeModel && <div className="anomaly-model-status" data-testid="anomaly-model-status">
          <strong>{activeModel.source === "loaded" ? "模型已載入" : activeModel.saved ? "模型已保存" : "模型已建立"}</strong>
          <span>{activeModel.meta.model} · {activeModel.meta.score_mode} · {activeModel.meta.target_res} · {activeModel.summary.n_objects} 個物件</span>
          <span>{activeModel.summary.head_status}{activeModel.summary.eligible_classes.length ? ` · ${activeModel.summary.eligible_classes.join(", ")}` : ""}</span>
          <span>Normal Bank {activeModel.summary.has_normal_bank ? "已建立" : "未建立（object score）"} · embedding D={activeModel.meta.obj_dim ?? "—"}</span>
          <small>模型 ID {activeModel.model_id}</small>
        </div>}
      </section>

      <section className="anomaly-card" aria-labelledby="anomaly-score-title" data-testid="anomaly-score-workspace">
        <div className="eyebrow">APPLY · RANK · REVIEW</div>
        <h2 id="anomaly-score-title">套用模型與風險排序</h2>
        <p>以目前載入模型掃描目標資料夾。重新套用標註只會重算分類門檻與排序，不會重建 Normal Bank 或重跑特徵。</p>
        <form className="anomaly-score-form" onSubmit={(event) => void scoreTargets(event)}>
          <label className="anomaly-wide">目標資料夾（每行一個本機路徑）
            <textarea aria-label="瑕疵偵測目標資料夾" data-testid="anomaly-target-folders" rows={3} value={targetFoldersText} disabled={scoreWorking || working} onChange={(event) => setTargetFoldersText(event.target.value)} placeholder="C:\\data\\product\\test" />
            <small>模型的特徵模型、分數依據、解析度與物件來源固定沿用建模設定。NativeApp 資料夾權限橋接尚待接線。</small>
          </label>
          <label>預估異常比例
            <input aria-label="瑕疵偵測污染率" data-testid="anomaly-contamination" type="number" min={0} max={1} step={0.01} value={contamination} disabled={working || scoreWorking} onChange={(event) => setContamination(Number(event.target.value))} />
          </label>
          <button type="submit" className="run-button anomaly-wide" data-testid="anomaly-score" disabled={working || scoreWorking || !connected || !activeModel || !splitLines(targetFoldersText).length}>
            {scoreWorking ? "正在評分…" : "套用目前模型"}
          </button>
        </form>
        {run && <div className="anomaly-run-summary" data-testid="anomaly-run-summary">
          <div><strong>{run.n_objects}</strong><span>個物件</span></div>
          <div><strong>{run.verdict_counts.bad}</strong><span>異常候選</span></div>
          <div><strong>{run.threshold.toFixed(5)}</strong><span>目前門檻</span></div>
          <div><strong>{run.auroc === null ? "—" : run.auroc.toFixed(3)}</strong><span>確認標籤 AUROC</span></div>
          <small>結果 {run.run_id.slice(0, 12)} · bank SHA-256 {run.model_bank_sha256.slice(0, 16)}… · revision {run.revision} · 已對齊 few-shot {run.fewshot_aligned} 筆</small>
        </div>}
      </section>

      {run && <section className="anomaly-card" aria-labelledby="anomaly-prelabel-title" data-testid="anomaly-prelabel-workspace">
        <div className="eyebrow">M11 · PRELABEL</div>
        <h2 id="anomaly-prelabel-title">瑕疵預標與安全 YOLO 匯出</h2>
        <p>分類頭只提供預標建議；檢查預覽後可匯出到來源資料夾以外的位置，人工仍需確認標註。</p>
        {!canPrelabel ? <div className="alert alert-info" role="status" data-testid="anomaly-prelabel-unavailable">
          {activeModel?.model_id !== run.model_bundle_id
            ? "目前載入的模型與這次偵測結果不同，請重新套用模型。"
            : "此模型沒有瑕疵分類頭，無法分辨瑕疵類別，因此不提供預標匯出。請以瑕疵類別語義建立並載入分類頭模型。"}
        </div> : <>
          <div className="anomaly-prelabel-controls">
            <label>最低信心 <strong>{prelabelMinConf.toFixed(2)}</strong>
              <input aria-label="預標最低信心" data-testid="anomaly-prelabel-min-conf" type="range" min={0} max={1} step={0.05} value={prelabelMinConf} disabled={prelabelWorking} onChange={(event) => { setPrelabelMinConf(Number(event.target.value)); setPrelabelPreview(null); }} />
            </label>
            <fieldset className="anomaly-prelabel-classes">
              <legend>接受的目標類別</legend>
              {activeModel?.summary.eligible_classes.map((label) => <label key={label}>
                <input type="checkbox" aria-label={`預標類別 ${label}`} checked={prelabelClasses.includes(label)} disabled={prelabelWorking} onChange={(event) => {
                  setPrelabelClasses((current) => event.target.checked
                    ? [...current, label]
                    : current.filter((item) => item !== label));
                  setPrelabelPreview(null);
                }} />
                {label}
              </label>)}
              {activeModel?.summary.eligible_classes.length === 0 && <small>模型沒有可用的瑕疵類別。</small>}
            </fieldset>
            <button type="button" className="run-button" data-testid="anomaly-prelabel-preview" disabled={prelabelWorking || !connected} onClick={() => void previewPrelabels(0)}>
              {prelabelWorking ? "正在計算…" : "產生預標預覽"}
            </button>
          </div>
          {prelabelPreview && <>
            <div className="anomaly-prelabel-summary" role="status" data-testid="anomaly-prelabel-summary">
              接受 {prelabelPreview.accepted} / {prelabelPreview.total} 個物件 · 分頁 {prelabelPreview.total ? `${prelabelOffset + 1}–${Math.min(prelabelOffset + prelabelPreview.items.length, prelabelPreview.total)}` : "0 筆"}
            </div>
            <div className="anomaly-prelabel-list" data-testid="anomaly-prelabel-list">
              {prelabelPreview.items.map((item) => <article key={item.index} className="anomaly-prelabel-item" data-testid={`anomaly-prelabel-item-${item.index}`}>
                <img className="anomaly-object-preview" src={api.anomalyObjectImageUrl(run.run_id, item.index, "preview")} alt={`${item.filename} 預標物件預覽`} loading="lazy" />
                <div>
                  <strong>{item.filename} · 物件 {item.obj_index + 1}</strong>
                  <span>原標籤 {item.label || "未命名"} · 異常分數 {item.score.toFixed(5)}</span>
                  <span>預測 {item.pred} · 信心 {item.confidence.toFixed(3)} · {item.accept ? "接受" : "不接受"}</span>
                  <span>框 [ {item.bbox.map((value) => value.toFixed(3)).join(", ")} ]</span>
                </div>
              </article>)}
              {prelabelPreview.items.length === 0 && <p className="anomaly-empty-results">目前沒有物件可預覽。</p>}
            </div>
            <div className="anomaly-pagination">
              <button type="button" className="secondary-action" disabled={prelabelWorking || prelabelOffset <= 0} onClick={() => void previewPrelabels(Math.max(0, prelabelOffset - 40))}>上一頁</button>
              <span>{prelabelPreview.total ? `${prelabelOffset + 1}–${Math.min(prelabelOffset + prelabelPreview.items.length, prelabelPreview.total)} / ${prelabelPreview.total}` : "0 筆"}</span>
              <button type="button" className="secondary-action" disabled={prelabelWorking || prelabelOffset + prelabelPreview.items.length >= prelabelPreview.total} onClick={() => void previewPrelabels(prelabelOffset + 40)}>下一頁</button>
            </div>
            <div className="anomaly-prelabel-export">
              <label>YOLO 標註輸出資料夾（須位於來源資料夾之外）
                <input aria-label="預標輸出資料夾" data-testid="anomaly-prelabel-output-dir" value={prelabelOutputDir} disabled={prelabelWorking} onChange={(event) => setPrelabelOutputDir(event.target.value)} placeholder="C:\\data\\prelabel-output" />
              </label>
              <button type="button" className="run-button" data-testid="anomaly-prelabel-export" disabled={prelabelWorking || !prelabelOutputDir.trim() || !connected} onClick={() => void exportPrelabels()}>
                {prelabelWorking ? "正在匯出…" : `匯出接受的預標（${prelabelPreview.accepted}）`}
              </button>
            </div>
          </>}
        </>}
      </section>}

      {run && <section className="anomaly-card" aria-labelledby="anomaly-review-title" data-testid="anomaly-review-workspace">
        <div className="eyebrow">SCORE REVIEW</div>
        <h2 id="anomaly-review-title">分數清單與確認標註</h2>
        <form className="anomaly-run-filters" onSubmit={(event) => { event.preventDefault(); void loadObjects(0).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "讀取評分清單失敗。")); }}>
          <label>結果
            <select aria-label="瑕疵結果篩選" value={filterVerdict} onChange={(event) => setFilterVerdict(event.target.value as "" | "good" | "bad")}>
              <option value="">全部</option><option value="bad">異常候選</option><option value="good">正常候選</option>
            </select>
          </label>
          <label>類別
            <select aria-label="瑕疵類別篩選" value={filterClass} onChange={(event) => setFilterClass(event.target.value)}>
              <option value="">全部類別</option>
              {run.class_labels.map((label) => <option key={label} value={label}>{label}</option>)}
            </select>
          </label>
          <label>分數至少
            <input aria-label="瑕疵最低分數" type="number" step="any" value={filterMin} onChange={(event) => setFilterMin(event.target.value)} />
          </label>
          <label>分數至多
            <input aria-label="瑕疵最高分數" type="number" step="any" value={filterMax} onChange={(event) => setFilterMax(event.target.value)} />
          </label>
          <button type="submit" className="secondary-action">套用篩選</button>
        </form>
        <div className="anomaly-review-actions">
          <span>依異常分數由高至低排列 · 篩選結果 {objectTotal} 筆。確認標籤用來重估門檻，單筆結果仍依分數與門檻分類。</span>
          <div className="anomaly-review-buttons">
            <button type="button" className="secondary-action" data-testid="anomaly-add-selected-curation" disabled={working || scoreWorking || curationWorking || selectedCuration.size === 0} onClick={() => void addSelectedToCuration()}>
              {curationWorking ? "正在加入…" : `加入策展清單（${selectedCuration.size}）`}
            </button>
            <button type="button" className="run-button" data-testid="anomaly-reapply" disabled={working || scoreWorking || Object.keys(draftConfirmed).length === 0} onClick={() => void reapply()}>
              套用標註並重算門檻{Object.keys(draftConfirmed).length ? `（${Object.keys(draftConfirmed).length} 筆變更）` : ""}
            </button>
          </div>
        </div>
        <div className="anomaly-object-list">
          {objects.map((item) => {
            const confirmation = draftConfirmed[String(item.index)] ?? item.confirmed;
            return <article key={item.index} className="anomaly-object" data-testid={`anomaly-object-${item.index}`}>
              <img className="anomaly-object-preview" src={api.anomalyObjectImageUrl(run.run_id, item.index, "preview")} alt={`${item.filename} 物件裁切`} loading="lazy" />
              <div className="anomaly-object-details">
                <strong>{item.filename}</strong>
                <span>{item.label || "未命名類別"} · 物件 {item.obj_index + 1} · {item.verdict === "bad" ? "異常候選" : "正常候選"}</span>
                <span>異常分數 {item.score.toFixed(6)} · 框 [ {item.bbox.map((value) => value.toFixed(3)).join(", ")} ]</span>
                {item.verdict === "bad" && <label className="anomaly-curation-choice">
                  <input
                    type="checkbox"
                    aria-label={`選取 ${item.filename} 物件 ${item.obj_index + 1} 加入策展清單`}
                    checked={selectedCuration.has(item.index)}
                    disabled={working || scoreWorking || curationWorking}
                    onChange={(event) => setSelectedCuration((current) => {
                      const next = new Set(current);
                      if (event.target.checked) next.add(item.index);
                      else next.delete(item.index);
                      return next;
                    })}
                  />
                  選取異常候選加入策展清單
                </label>}
                <label>人工確認
                  <select aria-label={`${item.filename} 人工確認`} value={confirmation} disabled={working || scoreWorking} onChange={(event) => setDraftConfirmed((current) => ({ ...current, [String(item.index)]: event.target.value as "good" | "bad" | "" }))}>
                    <option value="">未確認</option><option value="good">正常</option><option value="bad">異常</option>
                  </select>
                </label>
                {run.score_mode === "patch" && <button type="button" className="secondary-action" aria-expanded={heatmapIndex === item.index} onClick={() => setHeatmapIndex((current) => current === item.index ? null : item.index)}>
                  {heatmapIndex === item.index ? "隱藏 Patch 熱圖" : "顯示 Patch 熱圖"}
                </button>}
                {heatmapIndex === item.index && <img className="anomaly-object-heatmap" src={api.anomalyObjectImageUrl(run.run_id, item.index, "heatmap")} alt={`${item.filename} Patch 異常熱圖`} />}
              </div>
            </article>;
          })}
          {objects.length === 0 && <p className="anomaly-empty-results">目前篩選沒有符合的物件。</p>}
        </div>
        <div className="anomaly-pagination">
          <button type="button" className="secondary-action" disabled={objectOffset <= 0} onClick={() => void loadObjects(Math.max(0, objectOffset - 40)).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "讀取評分清單失敗。"))}>上一頁</button>
          <span>{objectTotal ? `${objectOffset + 1}–${Math.min(objectOffset + objects.length, objectTotal)} / ${objectTotal}` : "0 筆"}</span>
          <button type="button" className="secondary-action" disabled={objectOffset + objects.length >= objectTotal} onClick={() => void loadObjects(objectOffset + 40).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "讀取評分清單失敗。"))}>下一頁</button>
        </div>
      </section>}
      <p className="anomaly-boundary" data-testid="anomaly-boundary">目前包含建模、Normal Bank 保存／載入、目標資料夾評分、異常候選加入策展清單與原圖匯出。評分結果跨 worker 保存、完整 Streamlit 數值 parity 與 NativeApp 資料夾權限仍未驗收；加入清單的候選仍需人工覆核。</p>
    </main>
  );
}
