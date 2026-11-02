import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { api, type GroupDiffPca, type GroupDiffRegion, type GroupDiffRun, type JobStatus, type ModelOption } from "./api";

type Side = "good" | "bad";

function folderName(path: string) {
  return path.split(/[\\/]/).filter(Boolean).at(-1) || path;
}

function FolderList({
  side, title, folders, draft, setDraft, add, remove, choose,
}: {
  side: Side; title: string; folders: string[]; draft: string;
  setDraft: (value: string) => void; add: () => void; remove: (path: string) => void; choose: () => void;
}) {
  return (
    <section className="compare-folder-section">
      <div className="section-heading"><span className="step-number">{side === "good" ? "01" : "02"}</span><h2>{title}</h2></div>
      <button className="host-picker" type="button" onClick={choose}>
        <span className="folder-icon" aria-hidden="true">▱</span>從本機選擇 {title} 資料夾<span className="picker-hint">Native</span>
      </button>
      <label className="field-label" htmlFor={`groupdiff-${side}-folder`}>資料夾路徑</label>
      <div className="folder-input-row">
        <input id={`groupdiff-${side}-folder`} value={draft} onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && add()} placeholder="貼上本機影像資料夾路徑" autoComplete="off" />
        <button type="button" className="add-folder" onClick={add} aria-label={`Add ${title} folder`}>+</button>
      </div>
      {folders.length > 0 && <ul className="folder-list" aria-label={`${title} folders`}>
        {folders.map((folder) => <li key={folder} title={folder}><span className="folder-chip-dot" />
          <span className="folder-chip-name">{folderName(folder)}</span>
          <button type="button" aria-label={`Remove ${folder}`} onClick={() => remove(folder)}>×</button>
        </li>)}
      </ul>}
    </section>
  );
}

function GroupPca({ pca }: { pca: GroupDiffPca }) {
  const all = [...pca.xy_good, ...pca.xy_bad];
  if (!all.length) return <p className="field-note">PCA 沒有可繪製的樣本。</p>;
  const xs = all.map((point) => point[0]);
  const ys = all.map((point) => point[1]);
  const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xSpan = xMax - xMin || 1, ySpan = yMax - yMin || 1;
  const at = (point: number[]) => ({
    x: 36 + ((point[0] - xMin) / xSpan) * 588,
    y: 344 - ((point[1] - yMin) / ySpan) * 302,
  });
  return <div className="groupdiff-pca" data-testid="groupdiff-pca">
    <svg viewBox="0 0 660 380" role="img" aria-label="Good and Bad non-supervised PCA projection">
      <line x1="36" y1="344" x2="624" y2="344" /><line x1="36" y1="42" x2="36" y2="344" />
      {pca.xy_good.map((point, index) => { const pos = at(point); return <circle key={`g-${index}`} cx={pos.x} cy={pos.y} r="5" className="groupdiff-pca-good"><title>{`Good ${index + 1}`}</title></circle>; })}
      {pca.xy_bad.map((point, index) => { const pos = at(point); return <rect key={`b-${index}`} x={pos.x - 4.5} y={pos.y - 4.5} width="9" height="9" rx="1" className="groupdiff-pca-bad"><title>{`Bad ${index + 1}`}</title></rect>; })}
    </svg>
    <div className="groupdiff-pca-legend"><span><i className="groupdiff-pca-good" />Good</span><span><i className="groupdiff-pca-bad" />Bad</span>
      <small>PC1 {(pca.evr[0] * 100).toFixed(1)}% · PC2 {(pca.evr[1] * 100).toFixed(1)}%</small></div>
  </div>;
}

function RegionGallery({ runId, region, side }: { runId: string; region: GroupDiffRegion; side: Side }) {
  const count = side === "good" ? region.good_count : region.bad_count;
  const label = side === "good" ? "Good（最典型長相）" : "Bad（差異最明顯）";
  return <div className={`groupdiff-gallery ${side}`}>
    <h3>{label}</h3>
    {count ? <div className="groupdiff-gallery-grid">{Array.from({ length: Math.min(count, 3) }, (_, index) =>
      <img key={index} data-testid={`groupdiff-${side}-region-image`} loading="lazy"
        src={api.groupDiffRegionImageUrl(runId, region.rank, side, index)} alt={`${side} ${index + 1} · ${region.text}`} />,
    )}</div> : <p className="field-note">代表圖已無法讀取。</p>}
  </div>;
}

export default function GroupDiffWorkspace({
  models, loadingModels, connected,
}: { models: ModelOption[]; loadingModels: boolean; connected: boolean }) {
  const [goodFolders, setGoodFolders] = useState<string[]>([]);
  const [badFolders, setBadFolders] = useState<string[]>([]);
  const [goodDraft, setGoodDraft] = useState("");
  const [badDraft, setBadDraft] = useState("");
  const [modelId, setModelId] = useState("");
  const [nPerm, setNPerm] = useState<100 | 200 | 500>(200);
  const [alpha, setAlpha] = useState<0.01 | 0.05 | 0.1>(0.05);
  const [topK, setTopK] = useState(5);
  const [seed, setSeed] = useState(0);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [run, setRun] = useState<GroupDiffRun | null>(null);
  const [runSignature, setRunSignature] = useState("");
  const [regionRank, setRegionRank] = useState(1);
  const [baseChoice, setBaseChoice] = useState("auto-bad");
  const [overlayAlpha, setOverlayAlpha] = useState(0.55);
  const [displayWidth, setDisplayWidth] = useState(640);
  const [outputDir, setOutputDir] = useState("");
  const [pca, setPca] = useState<GroupDiffPca | null>(null);
  const [exportBusy, setExportBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const dinoModels = useMemo(() => models.filter((model) => model.model_id.startsWith("dinov2_vit")), [models]);

  useEffect(() => {
    if (dinoModels.some((model) => model.model_id === modelId && model.available)) return;
    const preferred = dinoModels.find((model) => model.model_id === "dinov2_vits14" && model.available)
      || dinoModels.find((model) => model.available);
    if (preferred) setModelId(preferred.model_id);
  }, [dinoModels, modelId]);

  const effectiveGoodFolders = goodDraft.trim() && !goodFolders.includes(goodDraft.trim()) ? [...goodFolders, goodDraft.trim()] : goodFolders;
  const effectiveBadFolders = badDraft.trim() && !badFolders.includes(badDraft.trim()) ? [...badFolders, badDraft.trim()] : badFolders;
  const currentSignature = JSON.stringify({ goodFolders: effectiveGoodFolders, badFolders: effectiveBadFolders, goodDraft: "", badDraft: "", modelId, nPerm, alpha, topK, seed });
  const stale = Boolean(run && runSignature !== currentSignature);
  const selectedRegion = run?.regions.find((region) => region.rank === regionRank) ?? run?.regions[0] ?? null;
  const base = useMemo(() => {
    if (!run) return null;
    if (baseChoice === "auto-good") return { side: "good" as const, index: run.base_good_index };
    if (baseChoice === "auto-bad") return { side: "bad" as const, index: run.base_bad_index };
    const [side, index] = baseChoice.split(":");
    const parsed = Number(index);
    if (!Number.isInteger(parsed)) return null;
    if (side === "good") return { side: "good" as const, index: parsed };
    if (side === "bad") return { side: "bad" as const, index: parsed };
    return null;
  }, [baseChoice, run]);

  useEffect(() => {
    if (!run) { setPca(null); return; }
    let active = true;
    api.groupDiffPca(run.run_id).then((value) => { if (active) setPca(value); })
      .catch(() => { if (active) setPca(null); });
    return () => { active = false; };
  }, [run]);

  function addFolder(side: Side) {
    const draft = side === "good" ? goodDraft : badDraft;
    const path = draft.trim();
    if (!path) return;
    if (side === "good") { setGoodFolders((current) => current.includes(path) ? current : [...current, path]); setGoodDraft(""); }
    else { setBadFolders((current) => current.includes(path) ? current : [...current, path]); setBadDraft(""); }
  }

  async function chooseFolder(side: Side) {
    setError("");
    if (!window.cimHost?.chooseFile) { setNotice("Native folder picker 尚未連線，請直接輸入本機資料夾路徑。"); return; }
    try {
      const result = await window.cimHost.chooseFile();
      if (result.canceled) return;
      const paths = result.paths?.length ? result.paths : result.path ? [result.path] : [];
      if (!paths.length) throw new Error("Native host 沒有回傳資料夾路徑。");
      if (side === "good") setGoodFolders((current) => [...new Set([...current, ...paths])]);
      else setBadFolders((current) => [...new Set([...current, ...paths])]);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "無法選擇本機資料夾。"); }
  }

  async function chooseOutputFolder() {
    setError("");
    if (!window.cimHost?.chooseFile) { setNotice("Native folder picker 尚未連線，請直接輸入輸出路徑。"); return; }
    try {
      const result = await window.cimHost.chooseFile();
      if (result.canceled) return;
      const path = result.paths?.[0] || result.path || "";
      if (!path) throw new Error("Native host 沒有回傳輸出資料夾。");
      setOutputDir(path);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "無法選擇輸出資料夾。"); }
  }

  async function runAnalysis() {
    const good = [...goodFolders];
    const bad = [...badFolders];
    if (goodDraft.trim() && !good.includes(goodDraft.trim())) good.push(goodDraft.trim());
    if (badDraft.trim() && !bad.includes(badDraft.trim())) bad.push(badDraft.trim());
    if (!good.length || !bad.length) { setError("Good 與 Bad 各至少選擇一個資料夾。"); return; }
    if (!modelId) { setError("請安裝可用的 DINOv2 模型後再分析。"); return; }
    setError(""); setNotice(""); setRun(null); setPca(null); setBusy(true); setJob(null);
    const signature = JSON.stringify({ goodFolders: good, badFolders: bad, goodDraft: "", badDraft: "", modelId, nPerm, alpha, topK, seed });
    try {
      const created = await api.createGroupDiffJob({ good_dirs: good, bad_dirs: bad, model_id: modelId, n_perm: nPerm, alpha, top_k: topK, seed });
      let latest: JobStatus | null = null;
      for (let attempt = 0; attempt < 18_000; attempt += 1) {
        latest = await api.groupDiffJob(created.job_id);
        setJob(latest);
        if (["completed", "failed", "cancelled"].includes(latest.state)) break;
        await new Promise((resolve) => window.setTimeout(resolve, 350));
      }
      if (!latest || !["completed", "failed", "cancelled"].includes(latest.state)) throw new Error("分析時間過長；請確認本機模型和影像資料後再重試。");
      if (latest.state === "failed") throw new Error(latest.error?.message || "群組差異分析失敗。");
      if (latest.state !== "completed" || !latest.run_id) throw new Error("分析工作未回傳有效結果。");
      const result = await api.groupDiffRun(latest.run_id);
      setRun(result); setRunSignature(signature); setRegionRank(result.regions[0]?.rank ?? 1);
      setBaseChoice("auto-bad");
      if (result.warn_msgs.length) setNotice(result.warn_msgs.join(" "));
    } catch (reason) { setError(reason instanceof Error ? reason.message : "群組差異分析失敗。"); }
    finally { setBusy(false); }
  }

  async function exportReport() {
    if (!run || !outputDir.trim()) { setError("請選擇報告輸出資料夾。"); return; }
    setError(""); setNotice(""); setExportBusy(true);
    try {
      const result = await api.exportGroupDiff(run.run_id, outputDir.trim());
      setNotice(`已匯出 ${result.count} 個檔案：${result.files.join("、")} → ${outputDir.trim()}`);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "匯出差異分析報告失敗。"); }
    finally { setExportBusy(false); }
  }

  return (
    <div className="compare-workspace groupdiff-workspace">
      <aside className="compare-controls">
        <div className="eyebrow">GOOD / BAD PATCH DIFFERENCE</div>
        <h1>Good／Bad 差異探索</h1>
        <p className="intro">逐 patch 比較兩群影像，找出穩定差異的位置。這是製程線索；兩群需來自大致對齊的拍攝視角。</p>
        <details className="groupdiff-help"><summary>適用情況與解讀方式</summary>
          <p>這裡比較 Good 群與 Bad 群在相同 patch 位置的特徵分布；Compare 比的是整體分布，瑕疵偵測則是單張影像相對正常群。</p>
          <p>拍攝站點和視角需大致一致。統計未顯著時會明示 null，熱圖只供參考；區域穩定度表示 Bad 群有多少影像在該位置與 Good 群明顯不同。</p>
        </details>
        <FolderList side="good" title="Good（良品）" folders={goodFolders} draft={goodDraft} setDraft={setGoodDraft}
          add={() => addFolder("good")} remove={(path) => setGoodFolders((current) => current.filter((item) => item !== path))} choose={() => void chooseFolder("good")} />
        <FolderList side="bad" title="Bad（不良品）" folders={badFolders} draft={badDraft} setDraft={setBadDraft}
          add={() => addFolder("bad")} remove={(path) => setBadFolders((current) => current.filter((item) => item !== path))} choose={() => void chooseFolder("bad")} />

        <section className="compare-config groupdiff-config">
          <div className="section-heading"><span className="step-number">03</span><h2>分析設定</h2></div>
          <label className="field-label" htmlFor="groupdiff-model">特徵模型</label>
          <select id="groupdiff-model" aria-label="特徵模型" value={modelId} onChange={(event) => setModelId(event.target.value)} disabled={busy || loadingModels}>
            {dinoModels.map((model) => <option key={model.model_id} value={model.model_id}>{model.name}</option>)}
          </select>
          {!loadingModels && !dinoModels.length && <div className="model-empty">請先安裝 DINOv2 模型以進行 patch 差異分析。</div>}
          <details className="groupdiff-advanced"><summary>進階統計參數</summary>
            <label className="field-label" htmlFor="groupdiff-top-k">Top-K 差異區域</label>
            <input id="groupdiff-top-k" aria-label="Top-K 差異區域" type="number" min="1" max="10" value={topK} onChange={(event) => setTopK(Math.max(1, Math.min(10, Number(event.target.value) || 1)))} disabled={busy} />
            <label className="field-label" htmlFor="groupdiff-permutations">隨機重分組次數（null 校準）</label>
            <select id="groupdiff-permutations" aria-label="隨機重分組次數" value={nPerm} onChange={(event) => setNPerm(Number(event.target.value) as 100 | 200 | 500)} disabled={busy}>
              {[100, 200, 500].map((count) => <option key={count} value={count}>{count}</option>)}
            </select>
            <label className="field-label" htmlFor="groupdiff-alpha">顯著水準 α</label>
            <select id="groupdiff-alpha" aria-label="顯著水準" value={alpha} onChange={(event) => setAlpha(Number(event.target.value) as 0.01 | 0.05 | 0.1)} disabled={busy}>
              {[0.01, 0.05, 0.1].map((value) => <option key={value} value={value}>{value.toFixed(2)}</option>)}
            </select>
            <label className="field-label" htmlFor="groupdiff-seed">Random seed</label>
            <input id="groupdiff-seed" aria-label="Random seed" type="number" min="0" max="9999" value={seed} onChange={(event) => setSeed(Math.max(0, Math.min(9999, Number(event.target.value) || 0)))} disabled={busy} />
          </details>
        </section>

        {(error || notice) && <div className={`alert ${error ? "alert-error" : "alert-info"}`} role={error ? "alert" : "status"}>{error || notice}</div>}
        <button className="run-button" type="button" data-testid="groupdiff-run" onClick={() => void runAnalysis()}
          disabled={busy || !connected || loadingModels || !modelId || !goodFolders.length && !goodDraft.trim() || !badFolders.length && !badDraft.trim()}>
          <span aria-hidden="true">{busy ? "◌" : "▶"}</span>{busy ? "正在分析差異" : "🔬 分析差異"}
        </button>
        <div className="privacy-note"><span aria-hidden="true">⌁</span> 本機處理 · 不上傳影像</div>
      </aside>

      <main className="compare-results groupdiff-results">
        <div className="results-header"><div><div className="eyebrow">PATCH-ALIGNED GROUP ANALYSIS</div>
          <h2>{run ? "差異分析結果" : "開始 Good／Bad 群組分析"}</h2>
          <p>{run ? `Good ${run.n_good} 張 · Bad ${run.n_bad} 張 · ${run.meta.model} · ${run.grid[0]}×${run.grid[1]} patch` : "各選一組已拍攝對齊的良品與不良品資料夾，再開始本機分析。"}</p></div>
          {run && <div className="run-badge"><span className="badge-dot" />結果可檢視</div>}
        </div>
        {busy && <div className="progress-card" data-testid="groupdiff-job-status" role="status">
          <div className="progress-row"><span>{job?.message || "準備差異分析"}</span><strong>{Math.round(job?.percent || 0)}%</strong></div>
          <div className="progress-track"><div className="progress-fill" style={{ width: Math.max(3, job?.percent || 0) + "%" }} /></div>
        </div>}
        {!busy && job?.state === "completed" && <div className="completion-label" data-testid="groupdiff-job-status">✓ 分析完成</div>}
        {stale && <div className="alert alert-warning" role="status">設定已變更；下方仍是上次分析結果，請重新執行以套用新設定。</div>}

        {run ? <>
          {run.verdict ? <div className="alert alert-success groupdiff-verdict" data-testid="groupdiff-verdict" role="status">
            ✅ 兩群存在穩定的局部差異（p={run.p_global.toFixed(4)}；Good {run.n_good} 張 vs Bad {run.n_bad} 張）
          </div> : <div className="alert alert-warning groupdiff-verdict" data-testid="groupdiff-verdict" role="status">
            ⚠ 未發現穩定差異——目前兩群在局部特徵上無法可靠區分（p={run.p_global.toFixed(4)}；Good {run.n_good} 張 vs Bad {run.n_bad} 張）。不建議把下方熱圖當製程線索。
          </div>}
          {run.warn_msgs.map((warning) => <div className="alert alert-warning" role="status" key={warning}>⚠ {warning}</div>)}
          <details className="groupdiff-heat-card" data-testid="groupdiff-heatmap-details" open={run.verdict}>
            <summary>{run.verdict ? "差異熱圖" : "仍要看熱圖（統計未達顯著，僅供參考）"}</summary>
            <div className="groupdiff-heat-controls">
              <label>熱圖底圖<select aria-label="熱圖底圖" value={baseChoice} onChange={(event) => setBaseChoice(event.target.value)}>
                <option value="auto-bad">自動 Bad 代表圖</option><option value="auto-good">自動 Good 代表圖</option>
                {run.bases_bad.map((name, index) => <option key={`b-${index}`} value={`bad:${index}`}>Bad｜{name}</option>)}
                {run.bases_good.map((name, index) => <option key={`g-${index}`} value={`good:${index}`}>Good｜{name}</option>)}
              </select></label>
              <label>熱力圖濃度 {overlayAlpha.toFixed(2)}<input aria-label="熱力圖濃度" type="range" min="0" max="1" step="0.05" value={overlayAlpha} onChange={(event) => setOverlayAlpha(Number(event.target.value))} /></label>
              <label>顯示寬度 {displayWidth}px<input aria-label="顯示寬度" type="range" min="320" max="1200" step="20" value={displayWidth} onChange={(event) => setDisplayWidth(Number(event.target.value))} /></label>
            </div>
            {base && <div className="groupdiff-heat-images" style={{ "--groupdiff-width": `${displayWidth}px` } as CSSProperties}>
              <figure><img src={api.groupDiffBaseUrl(run.run_id, base.side, base.index, false, overlayAlpha)} alt={`${base.side} 原始底圖`} />
                <figcaption>原圖：{(base.side === "good" ? run.bases_good : run.bases_bad)[base.index]}</figcaption></figure>
              <figure><img data-testid="groupdiff-heatmap" src={api.groupDiffBaseUrl(run.run_id, base.side, base.index, true, overlayAlpha)} alt="群組 patch 差異熱圖" />
                <figcaption>越紅代表兩群在該位置的系統性差異越大。</figcaption></figure>
            </div>}
          </details>
          {selectedRegion ? <section className="drift-card groupdiff-region-card" data-testid="groupdiff-top-region">
            <div className="section-heading"><span className="step-number">04</span><h2>Top 差異區域</h2></div>
            <label className="field-label" htmlFor="groupdiff-region">選擇區域</label>
            <select id="groupdiff-region" aria-label="選擇差異區域" value={selectedRegion.rank} onChange={(event) => setRegionRank(Number(event.target.value))}>
              {run.regions.map((region) => <option key={region.rank} value={region.rank}>第 {region.rank} 名 · z={region.peak_z.toFixed(2)} · Bad {region.pct}%</option>)}
            </select>
            <p className="field-note"><strong>{selectedRegion.text}</strong>（峰值 z={selectedRegion.peak_z.toFixed(2)}，顯著門檻 {run.threshold_z.toFixed(2)}；穩定度 {selectedRegion.pct}%）</p>
            <div className="groupdiff-region-grid"><RegionGallery runId={run.run_id} region={selectedRegion} side="good" /><RegionGallery runId={run.run_id} region={selectedRegion} side="bad" /></div>
          </section> : run.verdict && <div className="alert alert-info">統計顯著，但沒有連通區域超過門檻。可查看上方熱圖。</div>}
          <details className="groupdiff-pca-card"><summary>📎 整體旁證（PCA，非監督）</summary>
            {pca ? <GroupPca pca={pca} /> : <p className="field-note">樣本不足或無法建立 PCA 旁證。</p>}
            <p className="field-note">PCA 軸不使用 Good／Bad 標籤，避免監督投影製造假分離；此圖只作整體旁證。</p>
          </details>
          <section className="drift-card groupdiff-export-card">
            <div className="section-heading"><span className="step-number">05</span><h2>匯出差異報告</h2></div>
            <div className="groupdiff-export-row"><input aria-label="報告輸出資料夾" value={outputDir} onChange={(event) => setOutputDir(event.target.value)} placeholder="選擇另一個輸出資料夾" />
              <button type="button" className="host-picker" onClick={() => void chooseOutputFolder()}>📁 瀏覽</button>
              <button type="button" className="secondary-action" onClick={() => void exportReport()} disabled={exportBusy || !outputDir.trim()}>{exportBusy ? "匯出中…" : "📦 匯出報告"}</button>
            </div>
            <p className="field-note">匯出熱圖、Top 區域 CSV、Good／Bad 對照圖及統計摘要。輸出路徑不能位於來源資料夾內。</p>
          </section>
        </> : <div className="empty-canvas compare-empty"><div className="empty-illustration" aria-hidden="true"><span className="orbit orbit-one" /><span className="orbit orbit-two" /><i className="dot dot-one" /><i className="dot dot-two" /><i className="dot dot-three" /><i className="dot dot-four" /><b>G/B</b></div>
          <h3>比較兩群對齊影像</h3><p>統計不顯著時會如實顯示 null 結果，不會把噪聲包裝成製程線索。</p></div>}
      </main>
    </div>
  );
}
