import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type FalsePositiveCase,
  type FalsePositiveExternalProfile,
  type FalsePositiveExternalRun,
  type FalsePositiveSource,
} from "./api";

type Props = {
  caseData: FalsePositiveCase;
  selectedSourceId: string;
  onCaseUpdated: (next: FalsePositiveCase) => void;
};

type OutputFormat = "classification" | "yolo_v8" | "yolo_v5" | "xyxy";

const ADVANCED_DEFAULTS = {
  interpolation: "bilinear",
  color: "RGB",
  input_layout: "NCHW",
  scale: 1 / 255,
  mean: [0, 0, 0],
  std: [1, 1, 1],
  pad_value: 114,
  score_activation: "none",
  coordinates: "pixels",
  nms_iou: 0.45,
  class_agnostic_nms: false,
  max_detections: 300,
  input_name: "",
  output_name: "",
};

const MAIN_CONFIG_KEYS = new Set([
  "model_path", "model_version", "output_format", "class_names", "normal_classes",
  "input_size", "resize", "threshold",
]);

function splitNames(value: string) {
  return [...new Set(value.split(",").map((name) => name.trim()).filter(Boolean))];
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "外部模型操作失敗。";
}

function shortHash(value: string) {
  return value.slice(0, 16) + "…" + value.slice(-8);
}

function modelConfig(input: {
  modelVersion: string;
  format: OutputFormat;
  classNames: string;
  normalClasses: string;
  width: string;
  height: string;
  resize: "letterbox" | "stretch";
  threshold: string;
  advanced: string;
}) {
  const advanced: unknown = JSON.parse(input.advanced);
  if (!advanced || typeof advanced !== "object" || Array.isArray(advanced)) {
    throw new Error("進階前後處理設定必須是 JSON 物件。");
  }
  const extras = advanced as Record<string, unknown>;
  if (Object.keys(extras).some((key) => MAIN_CONFIG_KEYS.has(key))) {
    throw new Error("進階 JSON 不得重複設定模型版本、輸出格式、類別、尺寸、縮放或門檻。");
  }
  const width = Number(input.width);
  const height = Number(input.height);
  const threshold = Number(input.threshold);
  if (!Number.isInteger(width) || !Number.isInteger(height) || !Number.isFinite(threshold)) {
    throw new Error("輸入寬高和判定門檻必須是數值。");
  }
  return {
    ...extras,
    model_version: input.modelVersion.trim(),
    output_format: input.format,
    class_names: splitNames(input.classNames),
    normal_classes: splitNames(input.normalClasses),
    input_size: [width, height],
    resize: input.resize,
    threshold,
  };
}

function applyProfile(profile: FalsePositiveExternalProfile) {
  const config = profile.config;
  const known = new Set([...MAIN_CONFIG_KEYS, ...Object.keys(ADVANCED_DEFAULTS)]);
  const advanced = Object.fromEntries(Object.entries(config).filter(([key]) => !known.has(key)));
  return {
    name: profile.name,
    modelVersion: String(config.model_version || ""),
    format: (config.output_format || "classification") as OutputFormat,
    classNames: Array.isArray(config.class_names) ? config.class_names.join(", ") : "",
    normalClasses: Array.isArray(config.normal_classes) ? config.normal_classes.join(", ") : "",
    width: String(Array.isArray(config.input_size) ? config.input_size[0] : 224),
    height: String(Array.isArray(config.input_size) ? config.input_size[1] : 224),
    resize: (config.resize || "letterbox") as "letterbox" | "stretch",
    threshold: String(config.threshold ?? 0.25),
    advanced: JSON.stringify({ ...ADVANCED_DEFAULTS, ...Object.fromEntries(
      Object.entries(config).filter(([key]) => Object.prototype.hasOwnProperty.call(ADVANCED_DEFAULTS, key)),
    ), ...advanced }, null, 2),
  };
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function parityLabel(run: FalsePositiveExternalRun) {
  if (run.parity.status === "matched") return "完整歷史輸出相符";
  if (run.parity.status === "mismatched") return "完整歷史輸出不相符";
  return "歷史 parity 不可用";
}

function detectionStyle(box: [number, number, number, number]) {
  return {
    left: `${box[0] * 100}%`, top: `${box[1] * 100}%`,
    width: `${(box[2] - box[0]) * 100}%`, height: `${(box[3] - box[1]) * 100}%`,
  };
}

export default function FalsePositiveExternalModels({ caseData, selectedSourceId, onCaseUpdated }: Props) {
  const [profiles, setProfiles] = useState<FalsePositiveExternalProfile[]>([]);
  const [selectedProfileId, setSelectedProfileId] = useState("");
  const [sourceId, setSourceId] = useState(selectedSourceId);
  const [profileName, setProfileName] = useState("外部模型");
  const [modelVersion, setModelVersion] = useState("");
  const [format, setFormat] = useState<OutputFormat>("classification");
  const [classNames, setClassNames] = useState("good, crack");
  const [normalClasses, setNormalClasses] = useState("good");
  const [width, setWidth] = useState("224");
  const [height, setHeight] = useState("224");
  const [resize, setResize] = useState<"letterbox" | "stretch">("letterbox");
  const [threshold, setThreshold] = useState("0.25");
  const [advanced, setAdvanced] = useState(JSON.stringify(ADVANCED_DEFAULTS, null, 2));
  const [historicalJson, setHistoricalJson] = useState("");
  const [scoreTolerance, setScoreTolerance] = useState("0.001");
  const [boxTolerance, setBoxTolerance] = useState("0.001");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const modelFileRef = useRef<HTMLInputElement>(null);
  const referenceFileRef = useRef<HTMLInputElement>(null);

  const selectedProfile = profiles.find((profile) => profile.id === selectedProfileId) || null;
  const selectedSource = caseData.sources.find((source) => source.id === sourceId) || null;
  const activeRun = useMemo(
    () => caseData.external_runs.find((run) => run.id === selectedRunId)
      || caseData.external_runs[caseData.external_runs.length - 1]
      || null,
    [caseData.external_runs, selectedRunId],
  );
  const activeRunSource = activeRun
    ? caseData.sources.find((source) => source.id === activeRun.source_id) || null
    : null;

  useEffect(() => {
    setSourceId(selectedSourceId || caseData.sources[0]?.id || "");
  }, [caseData.id, caseData.sources, selectedSourceId]);

  useEffect(() => {
    let current = true;
    void api.externalModelProfiles()
      .then(({ items }) => {
        if (!current) return;
        setProfiles(items);
        if (!selectedProfileId && items[0]) {
          setSelectedProfileId(items[0].id);
          const defaults = applyProfile(items[0]);
          setProfileName(defaults.name);
          setModelVersion(defaults.modelVersion);
          setFormat(defaults.format);
          setClassNames(defaults.classNames);
          setNormalClasses(defaults.normalClasses);
          setWidth(defaults.width);
          setHeight(defaults.height);
          setResize(defaults.resize);
          setThreshold(defaults.threshold);
          setAdvanced(defaults.advanced);
        }
      })
      .catch((reason: unknown) => { if (current) setError(errorMessage(reason)); });
    return () => { current = false; };
  }, [caseData.id]);

  useEffect(() => {
    if (caseData.external_runs.length && !caseData.external_runs.some((run) => run.id === selectedRunId)) {
      setSelectedRunId(caseData.external_runs[caseData.external_runs.length - 1].id);
    }
  }, [caseData.external_runs, selectedRunId]);

  function selectProfile(profileId: string) {
    setSelectedProfileId(profileId);
    const profile = profiles.find((item) => item.id === profileId);
    if (!profile) return;
    const defaults = applyProfile(profile);
    setProfileName(defaults.name);
    setModelVersion(defaults.modelVersion);
    setFormat(defaults.format);
    setClassNames(defaults.classNames);
    setNormalClasses(defaults.normalClasses);
    setWidth(defaults.width);
    setHeight(defaults.height);
    setResize(defaults.resize);
    setThreshold(defaults.threshold);
    setAdvanced(defaults.advanced);
  }

  async function saveProfile(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const config = modelConfig({ modelVersion, format, classNames, normalClasses, width, height, resize, threshold, advanced });
      const file = modelFileRef.current?.files?.[0];
      if (!selectedProfile && !file) throw new Error("請先選擇單檔 ONNX 模型。");
      const asset = file ? await api.uploadExternalModelAsset(file) : null;
      const profile = selectedProfile
        ? await api.reviseExternalModelProfile(selectedProfile.id, {
          expected_revision: selectedProfile.revision,
          name: profileName,
          asset_sha256: asset?.sha256,
          config,
        })
        : await api.createExternalModelProfile({
          name: profileName,
          asset_sha256: asset!.sha256,
          config,
        });
      setProfiles((current) => [profile, ...current.filter((item) => item.id !== profile.id)]);
      setSelectedProfileId(profile.id);
      if (modelFileRef.current) modelFileRef.current.value = "";
      setNotice(`模型設定 v${profile.revision} 已保存；儲存設定不會執行推論。`);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  async function runModel(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedProfile || !selectedSource) {
      setError("請先選擇模型設定與案件影像來源。");
      return;
    }
    setBusy(true);
    setError("");
    setNotice("");
    try {
      let reference: Record<string, unknown> | undefined;
      if (historicalJson.trim()) {
        const parsed: unknown = JSON.parse(historicalJson);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
          throw new Error("歷史部署紀錄必須是 JSON 物件。");
        }
        reference = parsed as Record<string, unknown>;
      }
      const response = await api.runExternalModel(caseData.id, {
        expected_revision: caseData.revision,
        source_id: selectedSource.id,
        profile_id: selectedProfile.id,
        profile_revision: selectedProfile.revision,
        reference,
        score_atol: Number(scoreTolerance),
        box_atol: Number(boxTolerance),
      });
      onCaseUpdated(response.case);
      setSelectedRunId(response.run.id);
      setNotice("推論及完整輸出比對已保存為案件證據；結果不會自動寫成區域標籤。");
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  async function exportRun(formatValue: "json" | "zip") {
    if (!activeRun) return;
    setBusy(true);
    setError("");
    try {
      const blob = formatValue === "json"
        ? await api.externalModelEvidence(caseData.id, activeRun.id)
        : await api.externalModelEvidenceZip(caseData.id, activeRun.id);
      downloadBlob(blob, `${activeRun.id}.${formatValue}`);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  const profileConfig = selectedProfile?.config;
  const profileConfigForForm = selectedProfile && profileName === selectedProfile.name
    && modelVersion === String(profileConfig?.model_version || "");

  return (
    <section className="fp-card fp-external-card" aria-labelledby="fp-external-title" data-testid="fp-external-models">
      <div className="fp-card-heading">
        <div><div className="eyebrow">EXTERNAL ONNX MODEL</div><h3 id="fp-external-title">外部模型推論與歷史比對</h3></div>
        <span>{caseData.external_runs.length} 筆推論紀錄</span>
      </div>
      <p className="fp-boundary-note">只執行明確設定的單張 ONNX 影像輸入契約。匯入模型不會推論；保存設定後仍需明確按下執行。截圖可做本次推論，但不能證明當時部署結果已重現。</p>
      {error && <div className="alert alert-error" role="alert" data-testid="fp-external-error">{error}</div>}
      {notice && <div className="alert alert-info" role="status" data-testid="fp-external-notice">{notice}</div>}

      <form className="fp-ext-profile-form" onSubmit={(event) => void saveProfile(event)}>
        <h4>模型設定</h4>
        <label>設定名稱
          <input aria-label="外部模型設定名稱" value={profileName} maxLength={240} disabled={busy} onChange={(event) => setProfileName(event.target.value)} />
        </label>
        <label>保存的模型版本名稱
          <input aria-label="保存的模型版本名稱" value={modelVersion} maxLength={240} disabled={busy} onChange={(event) => setModelVersion(event.target.value)} />
        </label>
        <label>ONNX 模型檔（最多 1 GiB）
          <input ref={modelFileRef} aria-label="ONNX 模型檔" type="file" accept=".onnx,application/onnx" disabled={busy} />
          <small>選擇新檔會用內容 SHA-256 建立新模型身分；留空則只保存目前模型的新設定版本。</small>
        </label>
        <label>輸出契約
          <select aria-label="外部模型輸出契約" value={format} disabled={busy} onChange={(event) => setFormat(event.target.value as OutputFormat)}>
            <option value="classification">分類 [1,C] / [C]</option>
            <option value="yolo_v8">YOLO 類別分數 [1,4+C,N]</option>
            <option value="yolo_v5">YOLO objectness [1,N,5+C]</option>
            <option value="xyxy">已後處理框 [1,N,6] / [N,6]</option>
          </select>
        </label>
        <label>模型類別順序（逗號分隔）
          <input aria-label="外部模型類別順序" value={classNames} disabled={busy} onChange={(event) => setClassNames(event.target.value)} />
        </label>
        <label>正常類別（逗號分隔）
          <input aria-label="外部模型正常類別" value={normalClasses} disabled={busy} onChange={(event) => setNormalClasses(event.target.value)} />
        </label>
        <label>輸入寬度
          <input aria-label="外部模型輸入寬度" type="number" min={8} max={4096} value={width} disabled={busy} onChange={(event) => setWidth(event.target.value)} />
        </label>
        <label>輸入高度
          <input aria-label="外部模型輸入高度" type="number" min={8} max={4096} value={height} disabled={busy} onChange={(event) => setHeight(event.target.value)} />
        </label>
        <label>縮放方式
          <select aria-label="外部模型縮放方式" value={resize} disabled={busy} onChange={(event) => setResize(event.target.value as "letterbox" | "stretch")}>
            <option value="letterbox">保留比例並補邊</option>
            <option value="stretch">縮放至指定寬高</option>
          </select>
        </label>
        <label>判定門檻
          <input aria-label="外部模型判定門檻" type="number" min={0} max={1} step="0.01" value={threshold} disabled={busy} onChange={(event) => setThreshold(event.target.value)} />
        </label>
        <label className="fp-ext-wide">前後處理 JSON
          <textarea aria-label="外部模型前後處理 JSON" rows={9} value={advanced} disabled={busy} onChange={(event) => setAdvanced(event.target.value)} />
          <small>可設定 RGB/BGR、NCHW/NHWC、scale/mean/std、score activation、座標單位、NMS、max detections 與 input/output 名稱。不得填本機模型路徑。</small>
        </label>
        <div className="fp-ext-profile-picker">
          <label>已保存模型設定
            <select aria-label="已保存外部模型設定" value={selectedProfileId} disabled={busy || profiles.length === 0} onChange={(event) => selectProfile(event.target.value)}>
              {profiles.length === 0 && <option value="">尚無模型設定</option>}
              {profiles.map((profile) => <option key={profile.id} value={profile.id}>{profile.name} · v{profile.revision} · {shortHash(profile.asset.sha256)}</option>)}
            </select>
          </label>
          <button type="submit" className="run-button" data-testid="fp-save-external-profile" disabled={busy}>
            {busy ? "正在保存…" : selectedProfile ? "保存新設定版本" : "匯入並保存模型設定"}
          </button>
        </div>
        {selectedProfile && <div className="fp-ext-profile-meta">
          <span>{selectedProfile.asset.name} · {(selectedProfile.asset.size / (1024 * 1024)).toFixed(1)} MiB</span>
          <span>模型 SHA-256 {shortHash(selectedProfile.asset.sha256)}</span>
          {profileConfigForForm && <span>設定修訂會保留舊版，可供日後前後模型對照。</span>}
        </div>}
      </form>

      <form className="fp-ext-run-form" onSubmit={(event) => void runModel(event)}>
        <h4>執行與歷史判定重播</h4>
        <label>推論影像
          <select aria-label="外部模型推論影像" value={sourceId} disabled={busy || caseData.sources.length === 0} onChange={(event) => setSourceId(event.target.value)}>
            {caseData.sources.map((source: FalsePositiveSource) => <option key={source.id} value={source.id}>{source.kind === "original" ? "原圖" : source.kind === "screenshot" ? "截圖" : "參考圖"} · {source.name}</option>)}
          </select>
        </label>
        <label>歷史部署輸出 JSON（可留空）
          <input ref={referenceFileRef} aria-label="歷史部署輸出 JSON 檔" type="file" accept=".json,application/json" disabled={busy} onChange={(event) => {
            const file = event.currentTarget.files?.[0];
            if (file) void file.text().then(setHistoricalJson).catch((reason: unknown) => setError(errorMessage(reason)));
          }} />
          <textarea aria-label="歷史部署輸出 JSON 文字" rows={5} value={historicalJson} disabled={busy} onChange={(event) => setHistoricalJson(event.target.value)} placeholder={'需包含 origin="deployment"、recorded_at、image_sha256、contract_id 和完整 prediction'} />
        </label>
        <div className="fp-ext-tolerance-grid">
          <label>分數容許誤差
            <input aria-label="歷史分數容許誤差" type="number" min={0} max={0.1} step="0.001" value={scoreTolerance} disabled={busy} onChange={(event) => setScoreTolerance(event.target.value)} />
          </label>
          <label>框座標容許誤差（normalized）
            <input aria-label="歷史框座標容許誤差" type="number" min={0} max={0.1} step="0.001" value={boxTolerance} disabled={busy} onChange={(event) => setBoxTolerance(event.target.value)} />
          </label>
          <button type="submit" className="run-button" data-testid="fp-run-external-model" disabled={busy || !selectedProfile || !selectedSource}>
            {busy ? "正在執行…" : "執行外部模型並比對"}
          </button>
        </div>
      </form>

      {caseData.external_runs.length > 0 && <section className="fp-ext-results" data-testid="fp-external-results">
        <div className="fp-ext-results-heading">
          <label>外部推論紀錄
            <select aria-label="外部推論紀錄" value={activeRun?.id || ""} disabled={busy} onChange={(event) => setSelectedRunId(event.target.value)}>
              {caseData.external_runs.slice().reverse().map((run) => <option key={run.id} value={run.id}>{run.at.slice(0, 19)} · {run.profile_name} · {run.id}</option>)}
            </select>
          </label>
          <div className="fp-ext-downloads">
            <button type="button" className="fp-secondary-button" disabled={busy || !activeRun} onClick={() => void exportRun("json")}>下載 JSON 證據</button>
            <button type="button" className="fp-secondary-button" disabled={busy || !activeRun} onClick={() => void exportRun("zip")}>下載完整推論 ZIP</button>
          </div>
        </div>
        {activeRun && <>
          <div className={`fp-ext-parity fp-ext-parity-${activeRun.parity.status}`} data-testid="fp-external-parity">
            <strong>{parityLabel(activeRun)}</strong>
            <span>{activeRun.parity.reasons.join("；")}</span>
          </div>
          <dl className="fp-ext-metadata">
            <div><dt>模型／設定</dt><dd>{activeRun.profile_name} · v{activeRun.profile_revision}</dd></div>
            <div><dt>模型 SHA-256</dt><dd title={activeRun.profile_asset_sha256}>{shortHash(activeRun.profile_asset_sha256)}</dd></div>
            <div><dt>契約 ID</dt><dd title={activeRun.contract_id}>{shortHash(activeRun.contract_id)}</dd></div>
            <div><dt>執行環境</dt><dd>{activeRun.runtime.name} {activeRun.runtime.version} · {activeRun.runtime.provider}</dd></div>
            <div><dt>輸入 tensor</dt><dd>{activeRun.input_shape.join(" × ")} · {shortHash(activeRun.input_sha256)}</dd></div>
            <div><dt>輸出 tensor</dt><dd>{activeRun.output_shape.join(" × ")} · {shortHash(activeRun.output_sha256)}</dd></div>
          </dl>
          {activeRun.parity.status !== "matched" && <p className="fp-ext-boundary">本機新推論不會自我認證為當時部署結果；只有原圖 hash、部署來源聲明、記錄時間、契約及完整輸出相符時才標示相符。</p>}
          <div className="fp-ext-result-grid">
            <figure>
              <div className="fp-ext-image-stage" style={{ aspectRatio: activeRunSource ? `${activeRunSource.size[0]} / ${activeRunSource.size[1]}` : "4 / 3" }}>
                <img src={api.falsePositiveSourceImageUrl(caseData.id, activeRun.source_id)} alt="外部模型原始輸入" />
                {activeRun.prediction.detections.map((detection, index) => <span key={`${detection.output_index ?? index}-${index}`} className="fp-ext-detection-box" style={detectionStyle(detection.box)}>
                  <small>{detection.class} {detection.score.toFixed(3)}</small>
                </span>)}
              </div>
              <figcaption>原始來源上的最終輸出框</figcaption>
            </figure>
            <figure>
              <img className="fp-ext-input-preview" src={api.externalModelInputPreviewUrl(caseData.id, activeRun.id)} alt="送入 ONNX 的前處理影像" />
              <figcaption>實際模型輸入預覽</figcaption>
            </figure>
          </div>
          {activeRun.prediction.task === "classification" ? <div className="fp-ext-classification" data-testid="fp-external-classification">
            <strong>{activeRun.prediction.class} · {activeRun.prediction.score?.toFixed(5)} · {activeRun.prediction.decision === "abstain" ? "拒判" : activeRun.prediction.alarm ? "警報" : "未警報"}</strong>
            <div>{Object.entries(activeRun.prediction.scores || {}).map(([label, score]) => <span key={label}>{label}: {score.toFixed(5)}</span>)}</div>
          </div> : <>
            <p className="fp-ext-detection-summary" data-testid="fp-ext-detection-summary">最終輸出 {activeRun.prediction.detections.length} 個框 · {activeRun.prediction.alarm ? "警報" : "未警報"}</p>
            {!activeRun.prediction.raw_candidates_available && <p className="fp-boundary-note">此模型輸出已包含後處理；沒有 NMS 前候選分數，不會重複執行 NMS。</p>}
            <div className="fp-ext-table-wrap"><table className="fp-ext-table"><thead><tr><th>類別</th><th>分數</th><th>Normalized box</th><th>輸出索引</th></tr></thead><tbody>
              {activeRun.prediction.detections.map((row, index) => <tr key={`${row.output_index ?? index}-${index}`}><td>{row.class}</td><td>{row.score.toFixed(5)}</td><td>{row.box.map((value) => value.toFixed(5)).join(", ")}</td><td>{row.output_index ?? "—"}</td></tr>)}
              {activeRun.prediction.detections.length === 0 && <tr><td colSpan={4}>沒有符合門檻的最終框。</td></tr>}
            </tbody></table></div>
            {activeRun.prediction.candidate_count !== undefined && <p className="fp-ext-boundary">保留 {activeRun.prediction.candidate_count} 個排序候選；模型輸出 JSON {activeRun.prediction.candidate_json_truncated ? "已截斷" : "完整"}。</p>}
          </>}
        </>}
      </section>}
    </section>
  );
}
