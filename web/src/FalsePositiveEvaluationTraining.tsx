import React, { useEffect, useMemo, useState } from "react";
import {
  api,
  type FalsePositiveCase,
  type FalsePositiveEvaluation,
  type FalsePositiveExternalProfile,
  type FalsePositiveTrainingExport,
} from "./api";

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "評估或訓練交接操作失敗。";
}

type Props = {
  caseData: FalsePositiveCase;
  onCaseUpdated: (next: FalsePositiveCase) => void;
};

type TrainingRow = {
  id: string;
  sha256: string;
  sourceName: string;
  sourceKind: string;
  roi: [number, number, number, number];
  verdict: string;
};

function formatRate(value: number | null) {
  return value === null ? "—" : String((value * 100).toFixed(1)) + "%";
}

function downloadJson(value: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

const manifestExample = JSON.stringify({
  schema_version: 1,
  purpose: "independent",
  independent_declared: true,
  split_basis: "不同批次，未用於訓練／選樣／調門檻",
  items: [
    { id: "normal-1", path: "normal.png", truth: "normal", verified: true, group_id: "lot-1" },
    { id: "defect-1", path: "defect.png", truth: "defect", verified: true, group_id: "lot-2" },
  ],
}, null, 2);

export default function FalsePositiveEvaluationTraining({ caseData, onCaseUpdated }: Props) {
  const [profiles, setProfiles] = useState<FalsePositiveExternalProfile[]>([]);
  const [profileRefresh, setProfileRefresh] = useState(0);
  const [baselineId, setBaselineId] = useState("");
  const [candidateId, setCandidateId] = useState("");
  const [baselineRevision, setBaselineRevision] = useState("1");
  const [candidateRevision, setCandidateRevision] = useState("1");
  const [manifestText, setManifestText] = useState("");
  const [baseDir, setBaseDir] = useState("");
  const [minSupport, setMinSupport] = useState("10");
  const [maxMissIncrease, setMaxMissIncrease] = useState("0");
  const [minFpReduction, setMinFpReduction] = useState("0");
  const [classNamesText, setClassNamesText] = useState("crack");
  const [selectedImages, setSelectedImages] = useState<string[]>([]);
  const [classAssignments, setClassAssignments] = useState<Record<string, string>>({});
  const [reviewer, setReviewer] = useState("");
  const [reason, setReason] = useState("");
  const [confirmedComplete, setConfirmedComplete] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const rows = useMemo<TrainingRow[]>(() => {
    const sources = new Map(caseData.sources.map((source) => [source.id, source]));
    const items = caseData.items
      .filter((item) => !item.withdrawn)
      .flatMap((item) => {
        const source = sources.get(item.source_id);
        return source ? [{
          id: "item:" + item.id,
          sha256: source.sha256,
          sourceName: source.name,
          sourceKind: source.kind,
          roi: item.roi,
          verdict: item.review.verdict,
        }] : [];
      });
    const candidates = caseData.candidates
      .filter((item) => !item.withdrawn)
      .map((item) => ({
        id: "candidate:" + item.id,
        sha256: item.source.sha256,
        sourceName: item.source.name,
        sourceKind: item.source.kind,
        roi: item.roi,
        verdict: item.review?.verdict || "pending",
      }));
    return [...items, ...candidates];
  }, [caseData.candidates, caseData.items, caseData.sources]);

  const images = useMemo(() => {
    const byHash = new Map<string, TrainingRow>();
    rows.forEach((row) => { if (!byHash.has(row.sha256)) byHash.set(row.sha256, row); });
    return [...byHash.entries()].map(([sha256, row]) => ({
      sha256, name: row.sourceName, kind: row.sourceKind,
    }));
  }, [rows]);

  const classNames = useMemo(
    () => classNamesText.split(",").map((name) => name.trim()).filter(Boolean),
    [classNamesText],
  );

  useEffect(() => {
    let current = true;
    void api.externalModelProfiles().then(({ items }) => {
      if (!current) return;
      setProfiles(items);
      if (items[0]) {
        setBaselineId(items[0].id);
        setCandidateId(items[0].id);
        setBaselineRevision(String(items[0].latest_revision));
        setCandidateRevision(String(items[0].latest_revision));
        const names = Array.isArray(items[0].config.class_names)
          ? items[0].config.class_names as string[]
          : [];
        const normal = Array.isArray(items[0].config.normal_classes)
          ? items[0].config.normal_classes as string[]
          : [];
        const defects = names.filter((name) => !normal.includes(name));
        if (defects.length) setClassNamesText(defects.join(", "));
      }
    }).catch((reasonValue: unknown) => {
      if (current) setError(errorMessage(reasonValue));
    });
    return () => { current = false; };
  }, [caseData.id, profileRefresh]);

  function setProfile(which: "baseline" | "candidate", id: string) {
    const profile = profiles.find((item) => item.id === id);
    if (which === "baseline") {
      setBaselineId(id);
      setBaselineRevision(String(profile?.latest_revision || 1));
    } else {
      setCandidateId(id);
      setCandidateRevision(String(profile?.latest_revision || 1));
    }
  }

  async function loadManifestFile(file?: File) {
    if (!file) return;
    try {
      setManifestText(await file.text());
      setError("");
    } catch (reasonValue) {
      setError(errorMessage(reasonValue));
    }
  }

  async function runEvaluation(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const parsed: unknown = JSON.parse(manifestText);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("評估清單必須是 JSON 物件。");
      }
      const response = await api.evaluateFalsePositiveCase(caseData.id, {
        expected_revision: caseData.revision,
        baseline_profile_id: baselineId,
        baseline_profile_revision: Number(baselineRevision),
        candidate_profile_id: candidateId,
        candidate_profile_revision: Number(candidateRevision),
        manifest: parsed as Record<string, unknown>,
        base_dir: baseDir.trim() || undefined,
        min_support: Number(minSupport),
        max_miss_increase: Number(maxMissIncrease) / 100,
        min_fp_reduction: Number(minFpReduction) / 100,
      });
      onCaseUpdated(response.case);
      setNotice("評估已保存：" + response.evaluation.gate.status + " · " + response.evaluation.id);
    } catch (reasonValue) {
      setError(errorMessage(reasonValue));
    } finally {
      setBusy(false);
    }
  }

  async function createTrainingExport(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const assignments = Object.fromEntries(
        Object.entries(classAssignments).filter(([, value]) => value),
      );
      const response = await api.exportFalsePositiveTraining(caseData.id, {
        expected_revision: caseData.revision,
        complete_images: selectedImages,
        class_assignments: assignments,
        class_names: classNames,
        reviewer,
        reason,
        confirmed_complete: confirmedComplete,
      });
      onCaseUpdated(response.case);
      const blob = await api.falsePositiveTrainingZip(caseData.id, response.training_export.id);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = response.training_export.id + ".zip";
      anchor.click();
      URL.revokeObjectURL(url);
      setNotice("訓練交接已保存並下載：" + response.training_export.n_images + " 張影像 · "
        + response.training_export.n_boxes + " 個框");
    } catch (reasonValue) {
      setError(errorMessage(reasonValue));
    } finally {
      setBusy(false);
    }
  }

  async function downloadTraining(record: FalsePositiveTrainingExport) {
    setBusy(true);
    setError("");
    try {
      const blob = await api.falsePositiveTrainingZip(caseData.id, record.id);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = record.id + ".zip";
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (reasonValue) {
      setError(errorMessage(reasonValue));
    } finally {
      setBusy(false);
    }
  }

  const latestEvaluation = caseData.evaluations[caseData.evaluations.length - 1];
  const baselineProfile = profiles.find((profile) => profile.id === baselineId);
  const candidateProfile = profiles.find((profile) => profile.id === candidateId);
  const baselineRevisions = baselineProfile
    ? Array.from({ length: baselineProfile.latest_revision }, (_, index) => index + 1).reverse()
    : [1];
  const candidateRevisions = candidateProfile
    ? Array.from({ length: candidateProfile.latest_revision }, (_, index) => index + 1).reverse()
    : [1];

  return (
    <section className="fp-card fp-eval-training" aria-labelledby="fp-eval-training-title" data-testid="fp-evaluation-training">
      <div className="fp-card-heading">
        <div><div className="eyebrow">MODEL CHANGE EVIDENCE</div><h3 id="fp-eval-training-title">固定資料集改善評估與訓練交接</h3></div>
        <div className="fp-ext-downloads">
          <span>{caseData.evaluations.length} 次評估 · {caseData.training_exports.length} 個訓練包</span>
          <button type="button" className="fp-secondary-button" disabled={busy} onClick={() => setProfileRefresh((value) => value + 1)}>重新載入模型設定</button>
        </div>
      </div>
      <p className="fp-boundary-note">
        前後模型使用相同影像與真值。儲存新的模型設定後可重新載入。只有獨立資料且支持數足夠時才判定門檻；通過只代表此固定資料集符合條件，不代表自動部署許可。
      </p>
      {error && <div className="alert alert-error" role="alert" data-testid="fp-evaluation-training-error">{error}</div>}
      {notice && <div className="alert alert-info" role="status">{notice}</div>}

      <form className="fp-ext-run-form" onSubmit={(event) => void runEvaluation(event)}>
        <h4>固定資料集：改善前後模型成對評估</h4>
        <label>改善前模型設定
          <select aria-label="評估改善前模型" value={baselineId} disabled={busy || profiles.length === 0} onChange={(event) => setProfile("baseline", event.target.value)}>
            {profiles.map((profile) => <option key={profile.id} value={profile.id}>{profile.name}</option>)}
          </select>
        </label>
        <label>改善前設定版本
          <select aria-label="評估改善前版本" value={baselineRevision} disabled={busy || !baselineId} onChange={(event) => setBaselineRevision(event.target.value)}>
            {baselineRevisions.map((revision) => <option key={revision} value={revision}>v{revision}</option>)}
          </select>
        </label>
        <label>改善後模型設定
          <select aria-label="評估改善後模型" value={candidateId} disabled={busy || profiles.length === 0} onChange={(event) => setProfile("candidate", event.target.value)}>
            {profiles.map((profile) => <option key={profile.id} value={profile.id}>{profile.name}</option>)}
          </select>
        </label>
        <label>改善後設定版本
          <select aria-label="評估改善後版本" value={candidateRevision} disabled={busy || !candidateId} onChange={(event) => setCandidateRevision(event.target.value)}>
            {candidateRevisions.map((revision) => <option key={revision} value={revision}>v{revision}</option>)}
          </select>
        </label>
        <label className="fp-ext-wide">評估清單 JSON
          <input aria-label="評估 JSON 檔案" type="file" accept=".json,application/json" disabled={busy} onChange={(event) => void loadManifestFile(event.target.files?.[0])} />
          <textarea aria-label="評估清單 JSON 內容" rows={8} value={manifestText} disabled={busy} onChange={(event) => setManifestText(event.target.value)} placeholder={manifestExample} />
          <small>圖片可使用絕對路徑，或以基準資料夾為相對路徑；每張圖須標明真值與 verified=true。</small>
        </label>
        <label className="fp-ext-wide">相對影像路徑的基準資料夾
          <input aria-label="評估資料基準資料夾" value={baseDir} disabled={busy} onChange={(event) => setBaseDir(event.target.value)} placeholder="本機資料夾路徑；絕對圖片路徑可留空" />
        </label>
        <div className="fp-ext-tolerance-grid">
          <label>正常／瑕疵各自最低支持數
            <input aria-label="評估最低支持數" type="number" min={1} max={5000} value={minSupport} disabled={busy} onChange={(event) => setMinSupport(event.target.value)} />
          </label>
          <label>容許漏檢率增加（百分點）
            <input aria-label="容許漏檢率增加" type="number" min={0} max={100} step={0.1} value={maxMissIncrease} disabled={busy} onChange={(event) => setMaxMissIncrease(event.target.value)} />
          </label>
          <label>要求誤逮率下降（百分點）
            <input aria-label="要求誤逮率下降" type="number" min={0} max={100} step={0.1} value={minFpReduction} disabled={busy} onChange={(event) => setMinFpReduction(event.target.value)} />
          </label>
          <button type="submit" className="run-button" data-testid="fp-run-paired-evaluation" disabled={busy || !baselineId || !candidateId || !manifestText.trim()}>
            {busy ? "正在逐圖執行兩個模型…" : "執行前後模型評估"}
          </button>
        </div>
      </form>

      {latestEvaluation && <section className="fp-ext-results" data-testid="fp-paired-evaluation-result">
        <div className={"fp-ext-parity " + (latestEvaluation.gate.status === "passed" ? "fp-ext-parity-matched" : "")}>
          <strong>最近評估：{latestEvaluation.gate.status}</strong>
          <span>{latestEvaluation.gate.reasons.join("；")}</span>
          <span>{latestEvaluation.baseline_profile.name} v{latestEvaluation.baseline_profile.revision} → {latestEvaluation.candidate_profile.name} v{latestEvaluation.candidate_profile.revision} · 資料集 {latestEvaluation.dataset_fingerprint.slice(0, 16)}</span>
        </div>
        <div className="fp-ext-table-wrap">
          <table className="fp-ext-table">
            <thead><tr><th>模型</th><th>正常圖</th><th>瑕疵圖</th><th>誤逮</th><th>誤逮率</th><th>漏檢</th><th>漏檢率</th><th>拒判</th></tr></thead>
            <tbody>
              {[["改善前", latestEvaluation.baseline], ["改善後", latestEvaluation.candidate]].map(([label, score]) => {
                const item = score as FalsePositiveEvaluation["baseline"];
                return <tr key={label as string}><td>{label as string}</td><td>{item.normal_images}</td><td>{item.defect_images}</td><td>{item.false_alarms}</td><td>{formatRate(item.false_alarm_rate)}</td><td>{item.misses}</td><td>{formatRate(item.miss_rate)}</td><td>{item.abstentions}</td></tr>;
              })}
            </tbody>
          </table>
        </div>
        {latestEvaluation.baseline.detection && latestEvaluation.candidate.detection && <p className="fp-ext-boundary">框層級：改善前 TP/FP/FN {latestEvaluation.baseline.detection.tp}/{latestEvaluation.baseline.detection.fp}/{latestEvaluation.baseline.detection.fn} · 改善後 {latestEvaluation.candidate.detection.tp}/{latestEvaluation.candidate.detection.fp}/{latestEvaluation.candidate.detection.fn}</p>}
        <details><summary>群組支持數與逐圖結果</summary><pre>{JSON.stringify({ baseline_groups: latestEvaluation.baseline.groups, candidate_groups: latestEvaluation.candidate.groups, per_image: latestEvaluation.per_image }, null, 2)}</pre></details>
        <p className="fp-ext-boundary">{latestEvaluation.limitation}</p>
        <button type="button" className="fp-secondary-button" onClick={() => downloadJson(latestEvaluation, latestEvaluation.id + ".json")}>下載評估 JSON</button>
      </section>}

      <form className="fp-ext-run-form" onSubmit={(event) => void createTrainingExport(event)}>
        <h4>人工完整覆核後的 YOLO 訓練交接</h4>
        <p className="fp-ext-boundary fp-ext-wide">此處只建立交接 ZIP，不修改來源影像，也不啟動外部重訓。正常負樣本只接受整張原圖的正常覆核。</p>
        {rows.length === 0 ? <p className="fp-ext-boundary fp-ext-wide">先建立並逐項覆核爭議區域或搜尋候選。</p> : <>
          <label className="fp-ext-wide">訓練瑕疵類別順序（逗號分隔）
            <input aria-label="訓練瑕疵類別" value={classNamesText} disabled={busy} onChange={(event) => setClassNamesText(event.target.value)} />
          </label>
          <div className="fp-ext-wide fp-ext-training-images">
            <strong>已確認整張標註完整的影像</strong>
            {images.map((image) => <label key={image.sha256}>
              <input type="checkbox" checked={selectedImages.includes(image.sha256)} disabled={busy} onChange={(event) => setSelectedImages((current) => event.target.checked ? [...current, image.sha256] : current.filter((sha) => sha !== image.sha256))} />
              {image.kind === "screenshot" ? "截圖（不可匯出）" : image.kind === "reference" ? "參考圖" : "原圖"} · {image.name} · {image.sha256.slice(0, 12)}
            </label>)}
          </div>
          <div className="fp-ext-wide fp-ext-table-wrap">
            <table className="fp-ext-table"><thead><tr><th>區域</th><th>影像</th><th>範圍</th><th>覆核</th><th>人工訓練類別</th></tr></thead>
              <tbody>{rows.map((row) => <tr key={row.id}>
                <td>{row.id}</td><td>{row.sourceName}</td><td>{row.roi.map((value) => value.toFixed(3)).join(", ")}</td><td>{row.verdict}</td>
                <td>{row.verdict === "defect" ? <select aria-label={"訓練類別 " + row.id} value={classAssignments[row.id] || ""} disabled={busy || classNames.length === 0} onChange={(event) => setClassAssignments((current) => ({ ...current, [row.id]: event.target.value }))}>
                  <option value="">請人工指定</option>{classNames.map((name) => <option key={name} value={name}>{name}</option>)}
                </select> : "—"}</td>
              </tr>)}</tbody>
            </table>
          </div>
          <label>完整標註確認者
            <input aria-label="訓練覆核者" value={reviewer} maxLength={256} disabled={busy} onChange={(event) => setReviewer(event.target.value)} />
          </label>
          <label>完整檢查與類別確認依據
            <input aria-label="完整標註依據" value={reason} maxLength={4000} disabled={busy} onChange={(event) => setReason(event.target.value)} />
          </label>
          <label className="fp-ext-wide fp-ext-confirmation">
            <input aria-label="確認影像標註完整" type="checkbox" checked={confirmedComplete} disabled={busy} onChange={(event) => setConfirmedComplete(event.target.checked)} />
            我已檢查選取影像的全部內容，真瑕疵框無遺漏，且影像是可用的原始訓練內容
          </label>
          <button type="submit" className="run-button fp-ext-wide" data-testid="fp-create-training-export" disabled={busy || rows.length === 0 || !confirmedComplete || selectedImages.length === 0 || !reviewer.trim() || !reason.trim() || classNames.length === 0}>
            {busy ? "正在驗證並建立 ZIP…" : "建立訓練交接 ZIP"}
          </button>
        </>}
      </form>
      {caseData.training_exports.length > 0 && <section className="fp-ext-results">
        <h4>訓練交接紀錄</h4>
        <div className="fp-ext-table-wrap"><table className="fp-ext-table"><thead><tr><th>紀錄</th><th>影像</th><th>瑕疵框</th><th>正常負樣本</th><th>確認者</th><th>建立時間</th><th></th></tr></thead>
          <tbody>{caseData.training_exports.slice().reverse().map((record) => <tr key={record.id}>
            <td>{record.id}</td><td>{record.n_images}</td><td>{record.n_boxes}</td><td>{record.n_negative_images}</td><td>{record.reviewer}</td><td>{record.at.slice(0, 19)}</td>
            <td><button type="button" className="fp-secondary-button" disabled={busy} onClick={() => void downloadTraining(record)}>下載 ZIP</button></td>
          </tr>)}</tbody>
        </table></div>
      </section>}
    </section>
  );
}
