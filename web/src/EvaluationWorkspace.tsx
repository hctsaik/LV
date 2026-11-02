import { useEffect, useState } from "react";
import { api, type ActiveLearningRound, type DetectionEvaluation } from "./api";

export default function EvaluationWorkspace() {
  const [folders, setFolders] = useState<string[]>([]);
  const [folderDraft, setFolderDraft] = useState("");
  const [predictionFile, setPredictionFile] = useState<File | null>(null);
  const [consensusFile, setConsensusFile] = useState<File | null>(null);
  const [iouThresh, setIouThresh] = useState(0.5);
  const [confThresh, setConfThresh] = useState(0);
  const [classAware, setClassAware] = useState(true);
  const [result, setResult] = useState<DetectionEvaluation | null>(null);
  const [rounds, setRounds] = useState<ActiveLearningRound[]>([]);
  const [selectedRoundId, setSelectedRoundId] = useState("");
  const [busy, setBusy] = useState(false);
  const [attaching, setAttaching] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [roundLoadError, setRoundLoadError] = useState("");
  const [attachedRoundId, setAttachedRoundId] = useState("");

  useEffect(() => {
    let active = true;
    api.activeLearningRounds().then((response) => {
      if (!active) return;
      setRounds(response.rounds);
      const latest = response.rounds[response.rounds.length - 1];
      if (latest) setSelectedRoundId(latest.round_id);
    }).catch((reason: unknown) => {
      if (active) setRoundLoadError(reason instanceof Error ? reason.message : "Could not load active-learning rounds.");
    });
    return () => { active = false; };
  }, []);

  function addFolder(path: string, grantId?: string) {
    const normalized = path.trim();
    if (!normalized) return;
    if (grantId) api.rememberFolderGrant(normalized, grantId);
    setResult(null);
    setAttachedRoundId("");
    setFolders((current) => current.includes(normalized) ? current : [...current, normalized]);
    setFolderDraft("");
  }

  async function chooseFolder() {
    setError("");
    setNotice("");
    if (!window.cimHost?.chooseFile) {
      setNotice("Native folder picker is unavailable in browser mode. Enter the local dataset path below.");
      return;
    }
    try {
      const selected = await window.cimHost.chooseFile();
      if (selected.canceled) return;
      const paths = selected.paths?.length ? selected.paths : selected.path ? [selected.path] : [];
      if (!paths.length) throw new Error("The native host did not return a folder path.");
      if (selected.grant_id && paths.length !== 1) {
        throw new Error("Choose one folder at a time so NativeApp can scope its permission.");
      }
      paths.forEach((path) => addFolder(path, selected.grant_id));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not choose the dataset folder.");
    }
  }

  async function runEvaluation() {
    setError("");
    setNotice("");
    if (!folders.length) {
      setError("Select at least one dataset folder.");
      return;
    }
    if (!predictionFile) {
      setError("Choose predictions.csv before running the evaluation.");
      return;
    }
    setBusy(true);
    setResult(null);
    setAttachedRoundId("");
    try {
      const evaluated = await api.evaluateDetections({
        folders,
        folderGrantIds: api.folderGrantIds(folders),
        predictionCsv: await predictionFile.text(),
        consensusCsv: consensusFile ? await consensusFile.text() : null,
        iouThresh,
        confThresh,
        classAware,
      });
      setResult(evaluated);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The detection evaluation failed.");
    } finally {
      setBusy(false);
    }
  }

  async function runDemo() {
    setError("");
    setNotice("");
    setBusy(true);
    setResult(null);
    setAttachedRoundId("");
    try {
      setResult(await api.demoDetectionEvaluation());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The sample evaluation failed.");
    } finally {
      setBusy(false);
    }
  }

  async function attachEvaluationToRound() {
    if (!result || result.demo || !selectedRoundId) return;
    setError("");
    setNotice("");
    setAttaching(true);
    try {
      const response = await api.attachEvaluationToRound(selectedRoundId, result.evaluation_id);
      setRounds(response.rounds);
      setAttachedRoundId(response.round_id);
      setNotice(response.attached
        ? `評估摘要已加入回合 ${response.round_id}。`
        : `這筆評估已在回合 ${response.round_id}，沒有重複新增。`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not add this evaluation to the round.");
    } finally {
      setAttaching(false);
    }
  }

  const existingEvaluationRound = result
    ? rounds.find((round) => round.evaluations.some((item) => item.evaluation_id === result.evaluation_id))
    : undefined;

  return (
    <main className="evaluation-page" aria-labelledby="evaluation-title" data-testid="evaluation-workspace">
      <header className="evaluation-header">
        <div>
          <div className="eyebrow">DETECTION EVALUATION</div>
          <h1 id="evaluation-title">在共識子集上評估偵測結果</h1>
          <p>匯入 YOLO ground truth、模型 predictions 與可選的 consensus set，依 IoU 計算逐類 recall、漏抓與混淆。</p>
        </div>
        <button className="secondary-action evaluation-demo" type="button" data-testid="run-evaluation-demo" disabled={busy} onClick={() => void runDemo()}>
          ✨ 用範例資料試跑
        </button>
      </header>

      <section className="evaluation-inputs" aria-label="Evaluation inputs">
        <div className="evaluation-card">
          <div className="section-heading"><span className="step-number">01</span><h2>資料集資料夾</h2></div>
          <p className="field-note">每個資料夾需包含 images/ 和 labels/；多個資料夾的影像檔名必須唯一。</p>
          <button className="host-picker" type="button" data-testid="evaluation-native-folder-picker" onClick={() => void chooseFolder()}>
            <span className="folder-icon" aria-hidden="true">⌕</span>
            <span>選擇資料夾</span>
          </button>
          <label className="field-label" htmlFor="evaluation-folder-path">或輸入資料夾路徑</label>
          <div className="folder-input-row">
            <input
              id="evaluation-folder-path"
              aria-label="Dataset folder path"
              value={folderDraft}
              onChange={(event) => setFolderDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  addFolder(folderDraft);
                }
              }}
              placeholder="D:\\dataset"
            />
            <button type="button" className="add-folder" aria-label="Add dataset folder" onClick={() => addFolder(folderDraft)}>+</button>
          </div>
          <ul className="folder-list" aria-label="Selected evaluation folders" data-testid="evaluation-folder-list">
            {folders.map((path) => (
              <li key={path} title={path}>
                <span className="folder-chip-dot" />
                <span className="folder-chip-name">{path}</span>
                <button type="button" aria-label={`Remove ${path}`} onClick={() => { setResult(null); setFolders((current) => current.filter((item) => item !== path)); }}>×</button>
              </li>
            ))}
          </ul>
        </div>

        <div className="evaluation-card">
          <div className="section-heading"><span className="step-number">02</span><h2>匯入 CSV</h2></div>
          <label className="field-label" htmlFor="evaluation-predictions-file">模型預測 predictions.csv</label>
          <input
            id="evaluation-predictions-file"
            aria-label="predictions.csv"
            data-testid="evaluation-predictions-file"
            type="file"
            accept=".csv,text/csv"
            onChange={(event) => { setResult(null); setPredictionFile(event.target.files?.[0] || null); }}
          />
          <p className="field-note">欄位：filename,class,cx,cy,w,h[,score]；座標使用 0 到 1 的正規化 YOLO 格式。</p>
          <label className="field-label" htmlFor="evaluation-consensus-file">共識子集 consensus_set.csv（可選）</label>
          <input
            id="evaluation-consensus-file"
            aria-label="consensus_set.csv"
            data-testid="evaluation-consensus-file"
            type="file"
            accept=".csv,text/csv"
            onChange={(event) => { setResult(null); setConsensusFile(event.target.files?.[0] || null); }}
          />
          <p className="field-note">支援影像層級或框層級共識；灰帶 GT 會獨立計數，不會算成漏抓。</p>
        </div>

        <div className="evaluation-card">
          <div className="section-heading"><span className="step-number">03</span><h2>比對條件</h2></div>
          <label className="evaluation-slider-label" htmlFor="evaluation-iou-threshold">
            <span>IoU 門檻</span><output>{iouThresh.toFixed(2)}</output>
          </label>
          <input id="evaluation-iou-threshold" aria-label="IoU threshold" type="range" min="0.1" max="0.9" step="0.05" value={iouThresh} onChange={(event) => { setResult(null); setIouThresh(Number(event.target.value)); }} />
          <label className="evaluation-slider-label" htmlFor="evaluation-confidence-threshold">
            <span>信心門檻</span><output>{confThresh.toFixed(2)}</output>
          </label>
          <input id="evaluation-confidence-threshold" aria-label="Confidence threshold" type="range" min="0" max="1" step="0.05" value={confThresh} onChange={(event) => { setResult(null); setConfThresh(Number(event.target.value)); }} />
          <label className="evaluation-checkbox">
            <input type="checkbox" aria-label="Class-aware matching" checked={classAware} onChange={(event) => { setResult(null); setClassAware(event.target.checked); }} />
            <span>類別需相符（class-aware）</span>
          </label>
          <button className="run-button" type="button" data-testid="run-detection-evaluation" disabled={busy} onClick={() => void runEvaluation()}>
            {busy ? "評估中…" : "▶ 評估"}
          </button>
        </div>
      </section>

      {error && <div className="evaluation-error" role="alert">{error}</div>}
      {notice && <div className="evaluation-notice" role="status">{notice}</div>}

      {result && <section className="evaluation-results" aria-label="Evaluation results" data-testid="evaluation-results">
        <div className="evaluation-result-heading">
          <div>
            <div className="eyebrow">RESULTS</div>
            <h2>偵測評估結果</h2>
            <p>共分析 {result.image_count} 張含 GT 的影像；prediction CSV 對應 {result.prediction_image_count} 個檔名。</p>
          </div>
          <a className="secondary-action evaluation-export" href={api.evaluationEscapesCsvUrl(result.evaluation_id)} download="escapes.csv" data-testid="download-evaluation-escapes">
            ⬇ 匯出漏抓清單 CSV
          </a>
        </div>

        {result.demo && <div className="evaluation-notice" role="status">範例資料：預測刻意漏掉一張影像，並把另一張標為灰帶。這是示範資料，不代表模型預測結果。</div>}
        {result.warnings.includes("NO_PREDICTIONS_MATCHED") && <div className="evaluation-notice" role="status">預測 CSV 沒對到任何影像；請檢查 filename 欄是否為影像檔名。</div>}
        {result.warnings.includes("CLASS_NAMESPACE_MISMATCH") && <div className="evaluation-notice" role="status">預測與 GT 的類別名稱沒有交集；class-aware 比對會讓 recall 看似全漏。請統一類別名稱或關閉 class-aware。</div>}
        {result.warnings.includes("CONSENSUS_NOT_PROVIDED") && <div className="evaluation-notice" role="status">未提供共識子集；目前 recall 僅供參考，建議先準備 consensus_set.csv。</div>}

        <div className="evaluation-metrics" aria-label="Overall metrics">
          <article><span>整體 recall</span><strong data-testid="evaluation-overall-recall">{(result.overall.recall * 100).toFixed(1)}%</strong><small>分母為共識 GT</small></article>
          <article><span>整體 precision</span><strong data-testid="evaluation-overall-precision">{(result.overall.precision * 100).toFixed(1)}%</strong></article>
          <article><span>漏抓 FN</span><strong data-testid="evaluation-escape-count">{result.overall.fn}</strong></article>
          <article><span>誤報 FP</span><strong data-testid="evaluation-false-positive-count">{result.overall.fp}</strong></article>
        </div>
        {result.used_consensus && <p className="evaluation-consensus-note">共識 GT {result.consensus_gt_count} · 灰帶 GT {result.gray_gt_count}（灰帶不計入 recall）。</p>}

        <section className="evaluation-round-attachment" aria-label="Add evaluation to an active-learning round" data-testid="evaluation-round-attachment">
          <div>
            <h3>加入回合報告</h3>
            <p>保存 recall、precision、FN／FP、逐類指標與本次比對條件；不保存資料夾路徑或影像。</p>
          </div>
          {result.demo ? (
            <p role="status" data-testid="sample-evaluation-not-attachable">範例結果不能加入正式回合。</p>
          ) : roundLoadError ? (
            <p role="alert">無法載入回合：{roundLoadError}</p>
          ) : existingEvaluationRound || attachedRoundId ? (
            <p role="status" data-testid="evaluation-round-attached">此評估已加入回合 {attachedRoundId || existingEvaluationRound?.round_id}。</p>
          ) : rounds.length === 0 ? (
            <p role="status">目前沒有可加入的回合。請先在 Curation 送出標註批次建立回合。</p>
          ) : (
            <div className="evaluation-round-controls">
              <label htmlFor="evaluation-round-select">回合</label>
              <select
                id="evaluation-round-select"
                aria-label="Round to attach evaluation"
                data-testid="evaluation-round-select"
                value={selectedRoundId}
                onChange={(event) => setSelectedRoundId(event.target.value)}
              >
                {[...rounds].reverse().map((round) => (
                  <option key={round.round_id} value={round.round_id}>{round.round_id} · {round.strategy || "—"}</option>
                ))}
              </select>
              <button
                type="button"
                className="primary-action"
                data-testid="attach-evaluation-to-round"
                disabled={attaching || !selectedRoundId}
                onClick={() => void attachEvaluationToRound()}
              >
                {attaching ? "加入中…" : "加入回合報告"}
              </button>
            </div>
          )}
        </section>

        {result.per_class.length > 0 && <section className="evaluation-table-section" aria-label="Per-class metrics">
          <h3>逐型態（FN 多的在前）</h3>
          <div className="evaluation-table-scroll"><table data-testid="evaluation-per-class-table">
            <thead><tr><th>型態</th><th>n_GT</th><th>recall</th><th>precision</th><th>TP</th><th>FN</th><th>FP</th></tr></thead>
            <tbody>{result.per_class.map((item) => <tr key={item.class_name}>
              <th scope="row">{item.class_name}</th><td>{item.n_gt}</td><td>{(item.recall * 100).toFixed(0)}%</td><td>{(item.precision * 100).toFixed(0)}%</td><td>{item.tp}</td><td>{item.fn}</td><td>{item.fp}</td>
            </tr>)}</tbody>
          </table></div>
        </section>}

        {result.escape_count > 0 && <section className="evaluation-escape-section" aria-label="Missed detections">
          <h3>漏抓畫廊（escape，共 {result.escape_count}）</h3>
          <div className="evaluation-escape-grid">{result.escapes.map((item) => <figure key={`${item.filename}-${item.index}`}>
            <img src={api.evaluationEscapeThumbnailUrl(result.evaluation_id, item.index)} alt={`${item.filename} 的 ${item.class_name} 漏抓區域`} loading="lazy" />
            <figcaption><strong>{item.class_name}</strong><span>{item.filename}</span></figcaption>
          </figure>)}</div>
          {result.escape_count > result.escapes.length && <p className="field-note">畫廊顯示前 {result.escapes.length} 個漏抓；CSV 包含全部 {result.escape_count} 個。</p>}
        </section>}

        {result.confusion.length > 0 && <section className="evaluation-confusion-section" aria-label="Class confusion">
          <h3>最常見的類別混淆（GT → 預測）</h3>
          <ul>{result.confusion.map((item) => <li key={`${item.ground_truth}-${item.predicted}`}>
            {item.ground_truth} → {item.predicted}：{item.count}
          </li>)}</ul>
        </section>}
      </section>}
    </main>
  );
}
