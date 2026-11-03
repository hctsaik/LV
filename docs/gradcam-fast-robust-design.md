# Grad-CAM「快速探索／穩健檢查」詳細設計

> 狀態：Proposed，待工程實作與真實資料校準  
> 日期：2026-08-30  
> 範圍：LV 單張分類理由（DINOv2 + differentiable classifier head）  
> 決策：v1 僅提供「快速探索」與「穩健檢查」，不推出「精準版」或「深度稽核」

## 1. Executive decision

下一版 GUI 在「分類理由（Grad-CAM）」內增加兩個分析層級：

1. **快速探索（單次歸因）**：沿用目前 signed contrastive token Grad-CAM，提供低成本的單張診斷與 quick occlusion smoke check。
2. **穩健檢查（重複驗證，推薦用於調模）**：固定同一 target/reference，在 7 個小幅擾動上重算 CAM，以 raw signed median 形成 consensus map，再以約 22 個遮蔽 forward 驗證 deletion faithfulness。

「穩健」只表示歸因在已定義的擾動與遮蔽試驗中較一致，不代表：

- 模型分類更準；
- CAM 空間解析度變高；
- 找到 defect segmentation；
- 得到完整因果解釋；
- 通過整個資料集或其他機台的泛化驗證。

GUI 禁止使用「快速／精準」、「普通／高品質」或任何暗示較慢必然較正確的命名。

## 2. 為什麼這是甜蜜點

目前單次 CAM 已能快速提出假說，但一張視覺上漂亮的 saliency map 不足以證明它依賴模型學到的內容；研究亦指出只靠視覺檢查可能誤導，應加入 model/data sensitivity 或 perturbation checks。參考：

- [Grad-CAM, ICCV 2017](https://arxiv.org/abs/1610.02391)：Grad-CAM 本質是 coarse、gradient-weighted localization。
- [SmoothGrad, 2017](https://arxiv.org/abs/1706.03825)：對輸入加入小擾動並聚合，可降低單次 gradient map 的視覺噪聲。
- [Sanity Checks for Saliency Maps, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html)：視覺吸引力不等於方法真的依賴模型或資料。
- [RISE, BMVC 2018](https://arxiv.org/abs/1806.07421)：以遮蔽輸入並觀察輸出變化，是可操作的 empirical faithfulness 檢查。
- [Transformer Interpretability Beyond Attention Visualization, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html)：Transformer 的 skip connection、attention 與跨層 relevance propagation 有額外困難，不能把單層熱圖描述成完整決策路徑。

甜蜜點不是把所有研究方法都塞進單張分析，而是補上兩個目前最缺的問題：

1. **穩定性**：小幅、應保持語意的改變後，分類與熱區是否仍一致？
2. **局部忠實度**：遮掉 consensus 熱區，固定的 target-vs-reference margin 是否真的比 random/control 降得更多？

本版刻意不預設 exhaustive per-patch occlusion、Integrated Gradients、Transformer LRP、多層 CAM 平均或 20 次以上 random trial。這些方法成本高、解讀契約不同，而且在分類頭不穩定時邊際效益很低。

## 3. Multi-agent 審查與裁決

本設計由三個獨立角色並行審查，再由主設計整合：

| 角色 | 主要主張 | 最終採納 |
| --- | --- | --- |
| XAI 方法審查 | 固定 clean target/reference、signed raw median、擾動穩定度、multi-control deletion | 採納核心方法；將 9 個 CAM 降為 7 個以控制成本 |
| 產品／GUI 審查 | 不可稱精準版；low-margin/OOF gating；清楚顯示進度、狀態與證據層級 | 採納命名、gating、布局；不採百次級 random 作為預設 |
| 工程／效能審查 | shared hook concurrency、stale result、cache duplication 與 OOM degrade 必須先修 | 列為上線前 P0 hard prerequisite |

主要分歧與裁決：

- 產品審查提出每個 fraction/baseline 至少 20 次 random，科學上較保守，但會把單張預設推到百次級 forward；改為 3 次 shape-matched random，並明確標為工程穩定度檢查，不宣稱統計顯著。
- XAI 審查提出 9 個 CAM；本版採 7 個 deterministic variants，在空間邊界、亮度與成本間取中間值。
- 快速模式保留目前一次 CAM + 三組遮蔽，維持既有使用者延遲與向後相容；其結果只能叫 quick check，不得顯示「已證實」。

## 4. 產品目標與非目標

### 4.1 Goals

- 讓工程師能在 GUI 選擇低成本探索或較可靠的重複驗證。
- 讓穩健模式回答「這張 map 是否穩定、是否對固定 pair margin 有局部忠實度」。
- 把 classifier quality、decision clarity、map stability、faithfulness 分成不同證據層，避免一個綠燈蓋掉其他失敗。
- 所有設定、seed、threshold、provenance 與 partial/stale 狀態可匯出、可重播。
- 單張按需執行，不自動寫 dataset、embedding cache、anomaly cache 或 Normal Bank。

### 4.2 Non-goals

- 不用更多 Grad-CAM 計算修正分類錯誤。
- 不自動提高輸入解析度。
- 不把 upsampling 當成更高定位解析度。
- 不把 Grad-CAM、DINO Attention 與 Anomaly Heatmap 融成一個分數。
- 不在 v1 提供假的即時取消；若仍為 blocking call，只顯示進度。
- 不把 sample-stratified OOF 稱為 grouped holdout 或 external validation。

## 5. 目前 baseline

現行 `scripts/gradcam_explain.py` 已具備：

- clean input 的 top-1 與 strongest rival；
- scalar target 為 `logit(target) - logit(reference)`；
- last-block token Grad-CAM；
- 保留正負 raw CAM，不以 ReLU/min-shift 製造紅色證據；
- signed symmetric display；
- 10% top/bottom/random 各一組的 quick occlusion；
- classifier probability margin、sample-level OOF 與改善提示；
- head/checkpoint/preprocess/scope provenance preflight。

現行 quick faithfulness 的限制：只有一個 random control，且 mean-color patch replacement 可能產生 out-of-distribution 方塊。因此 `supported` 只能理解成 smoke check，不能稱穩健驗證。

## 6. GUI information architecture

### 6.1 控制項順序

在現有「模式／分析單位」之後、target 選擇之前新增：

```text
檢查層級
● 快速探索（單次歸因）
○ 穩健檢查（重複驗證）
```

固定副標：

> 穩健檢查會重複計算穩定性與遮蔽證據；運算較多，但不會讓分類器更準，也不會增加 CAM 的 patch 解析度。

預設為「快速探索」，以保持目前延遲與操作習慣。快速結果完成後提供次要 CTA：

> 對同一決策執行穩健檢查

### 6.2 解釋問題

預設：

- target = canonical clean input 的實際 top-1；
- reference = clean input 的 runner-up；
- 所有擾動、遮蔽與 aggregation 固定同一 pair。

進階選項「自訂類別比較」才顯示 target/reference 兩個下拉選單：

- target 與 reference 不得相同；
- target 非實際 top-1 時，控制區、結果區與 JSON 均標 `counterfactual=true`；
- 文案固定為「反事實比較」，不能寫「模型為何如此判斷」。

### 6.3 執行前工作量摘要

快速探索：

```text
1 次 gradient attribution + 3 個遮蔽對照
用途：提出假說；尚未驗證擾動穩定性
```

穩健檢查：

```text
7 次 attribution + 最多約 22 個遮蔽 forward
預估工作量約為目前快速模式的 6–8 倍；實際時間依模型、解析度與硬體而異
```

不得在尚未建立 benchmark matrix 前顯示固定秒數。

### 6.4 主按鈕

- 快速：「產生單次歸因」
- 穩健：「執行穩健檢查」

切圖、調 opacity、開關 grid、展開 metadata 均不得自動重跑。

## 7. 共用數學契約

### 7.1 Canonical prediction

對 canonical classifier input 計算 logits `z` 與 probabilities `p`：

```text
target = argmax(z)
reference = argmax(z excluding target)
score = z[target] - z[reference]
```

若使用者指定 pair，改用指定 target/reference，但 canonical prediction 仍完整保存。

所有 variants 固定 canonical target/reference。即使某個 variant 的 top-1 翻轉，也不能偷偷改 reference；翻轉本身就是 stability evidence。

### 7.2 Signed CAM

每次 run 保留未正規化 `raw_signed_cam`。聚合前禁止對每張 map 各自 max-normalize，避免把極弱噪聲放大成滿幅紅藍。

Display normalization：

```text
scale = q99(abs(map))
if scale <= epsilon: fallback max(abs(map))
display = clip(map / scale, -1, 1)
```

匯出需保存 raw map、display scale 與 epsilon。紅色只表示支持 target 相對 reference；藍色只表示支持 reference 相對 target；透明為近中性。

## 8. 快速探索規格

### 8.1 計算

- 1× canonical forward + backward。
- last-block token Grad-CAM。
- 10% patches：top positive、strongest negative、1× shape/count-matched random。
- target/reference 固定。
- 遮蔽預設改為 canonical RGB 的 blurred replacement，邊緣 feather 1–2 px；若 blur 失敗才使用現行 per-channel mean，並在 metadata 記錄 fallback。

### 8.2 結論層級

快速模式最高只能顯示：

> 單次歸因已完成；quick faithfulness 支持／不足，尚未驗證擾動穩定性。

不得顯示「可信」、「可靠」、「已證實」或綠色最終通過。

### 8.3 用途

- 快速逐張瀏覽；
- 找可能的 shortcut 假說；
- 決定哪些樣本值得升級跑穩健檢查。

## 9. 穩健檢查規格

### 9.1 Variant set：N=7

所有擾動在 canonical resize/crop 完成後、ImageNet normalization 前施作，固定 seed，尺寸不變：

| Variant | 參數 |
| --- | --- |
| identity | canonical input |
| x-minus | 水平平移 `-0.25 × patch_size` |
| x-plus | 水平平移 `+0.25 × patch_size` |
| y-minus | 垂直平移 `-0.25 × patch_size` |
| y-plus | 垂直平移 `+0.25 × patch_size` |
| brightness-minus | RGB brightness `-3%` |
| brightness-plus | RGB brightness `+3%` |

空間平移使用 reflect padding；CAM inverse-warp 回 canonical grid，只在共同有效區計算 metrics。

v1 不預設 rotation、scale/crop jitter 或大幅 brightness：對極小 poke，這些操作可能真的移除或改變 defect，不能安全假設 label-preserving。Gaussian noise、contrast 與 blur variants 留待產品別校準後加入 profile。

### 9.2 Aggregation

- 每次保留 aligned raw signed CAM。
- `consensus_raw = pixelwise median(aligned_raw_maps)`。
- `uncertainty = 1.4826 × MAD`。
- 每格計算 `sign_agreement`。
- Display 只對 consensus 做一次 symmetric normalization。
- `sign_agreement < 0.75` 的格子降低 overlay opacity；不得把不一致格子畫成同等確定。
- 同時保存 clean map、consensus map 與 worst-agreement variant。

若所有 run 沒有 positive mass，仍可顯示藍色 reference evidence，但必須明示「沒有 target 正向證據」，不得製造紅色熱點。

### 9.3 Stability metrics

| Metric | 定義 |
| --- | --- |
| `target_win_rate` | variant top-1 等於 canonical target 的比例 |
| `fixed_margin_positive_rate` | 固定 pair margin > 0 的比例 |
| `margin_median/iqr/min` | 7 個 fixed-pair logit margins |
| `median_signed_spearman` | aligned raw map 對 clean map 的 signed rank correlation 中位數 |
| `top_positive_jaccard` | top-k positive cells 的 Jaccard；`k=max(4, ceil(0.10P))` |
| `salient_sign_agreement` | consensus `|CAM|` top 20% cells 的平均同號比例 |
| `mass_iqr_ratio` | `IQR(L1_abs_mass) / median(L1_abs_mass)` |

若某 run 的 positive cells 少於 k，`top_positive_jaccard` 標 `insufficient`，不可用零或負 cell 補滿。

### 9.4 Provisional stability gates

以下是工程門檻，不是統計顯著性，必須寫入 metadata 並在真實驗證集校準：

- `target_win_rate >= 6/7`
- `fixed_margin_positive_rate >= 6/7`
- `median_signed_spearman >= 0.60`
- `top_positive_jaccard >= 0.40`
- `salient_sign_agreement >= 0.75`

N=7 不足以給 p-value 或信賴區間，UI 禁止使用「統計顯著」。

### 9.5 Robust deletion：最多約 22 forwards

Primary baseline：blurred canonical RGB + feather edge。

| 組別 | 數量 |
| --- | ---: |
| Top-positive cumulative deletion：5%、10%、20%、30% | 4 |
| 每個 fraction 的 shape-matched random control ×3 | 12 |
| Strongest-negative control：10%、20% | 2 |
| Alternate baseline sensitivity：10% top + random ×3 | 4 |
| 合計 | 22 |

Alternate baseline 使用 per-channel global mean。Random mask 優先平移 top-mask 的形狀到非重疊位置，而不是抽散亂 cell，以降低 cluster geometry confound。

所有 variants 使用 canonical fixed target/reference margin，報告：

- top deletion curve；
- random mean/range；
- absolute AOPC；
- `top_AOPC - random_AOPC`；
- 10% top 的 empirical random rank；
- negative deletion margin change；
- 兩種 baseline 的方向一致性。

若 canonical margin 接近 0，只報 absolute margin drop，不報除法型 normalized drop。

### 9.6 Faithfulness gate

穩健遮蔽最多只能標 `heuristic_supported`，條件為：

- top 10% absolute margin drop > 0；
- top 10% drop 大於 3/3 random；
- top AOPC 大於 random mean AOPC；
- negative deletion 不呈現與 top 相同或更強的下降；
- alternate baseline 結論方向一致。

這表示「遮蔽結果與 CAM 排名一致」，不表示 CAM 落在真實 defect。

### 9.7 Adaptive early stop

若 7 個 variants 完成後符合任一條件：

- `target_win_rate < 4/7`；
- `fixed_margin_positive_rate < 4/7`；
- raw CAM 有效 run 少於 4；

則跳過 22-forward robust deletion，狀態為 `classifier_or_attribution_unstable`。結果仍保留 clean/consensus/metrics，並提供明確的次要操作「仍以研究用途執行遮蔽稽核」，不自動浪費運算。

## 10. Quality gating 與結論優先序

### 10.1 Hard block

- head/checkpoint/preprocess/scope provenance 缺失或不相容；
- 少於兩個 classes；
- target = reference；
- Normal/Unknown cascade target；
- 圖片不可讀；
- model 不支援 `forward_features` 或 token hook；
- token/grid mismatch；
- non-finite 或 zero gradient；
- 空間 variant 無法 inverse-align。

Hard block 不得以 Attention 或 Anomaly Heatmap 靜默替代 Grad-CAM。

### 10.2 Soft gates

- clean probability margin < 5%；
- OOF unavailable；
- OOF balanced accuracy < 60%；
- sample OOF prediction 與 full-fit top-1 不同；
- validation 無可追溯 provenance；
- 沒有 positive mass；
- stability 或 faithfulness 不足。

OOF 門檻為 provisional。sample-stratified OOF 必須標示 near-duplicate/group leakage 限制；若有 grouped holdout 或 external validation，應優先顯示其結果與類型。

低 OOF 或 sample OOF mismatch 時，穩健按鈕預設為次要操作；使用者勾選：

> 仍以研究用途執行；結果不可作模型改善依據

後才可執行。

### 10.3 最終狀態 precedence

結果由上而下判定，較下層成功不得覆蓋較上層失敗：

```text
incompatible_or_failed
  > classifier_validation_insufficient
  > decision_ambiguous
  > attribution_unstable
  > faithfulness_insufficient
  > local_evidence_heuristic_supported
```

即使最後一項成立，固定附註：

> 只支持此模型、此輸入與此 target/reference 的局部歸因；不代表因果真相、defect segmentation 或其他資料的泛化。

## 11. 結果區布局

由上到下固定：

1. **決策摘要**：top-1、runner-up、probability gap、logit margin、validation 類型。
2. **三欄影像**：原圖脈絡、模型實際輸入、signed overlay。
3. **比較問題卡**：實際決策或反事實比較、target/reference probabilities。
4. **四張 evidence cards**：分類頭泛化、決策清楚度、歸因穩定性、遮蔽忠實度。
5. **穩健模式比較**：clean map、consensus map、worst variant；附 Spearman/Jaccard/sign agreement。
6. **Deletion chart**：top curve、random range、negative controls、alternate baseline。
7. **保守結論與下一步**：依 precedence 產生，不以單一綠燈蓋掉上層問題。
8. **Method/provenance expander**：模型、checkpoint、head/input hash、scope、crop/pad、resolution、pooling、seed、profile 與完整 thresholds。

快速結果提供 CTA「對此結果執行穩健檢查」。Grad-CAM 結果另可提供次要 CTA：

> 你的問題是「哪裡不像正常」？前往 Anomaly Heatmap

該 CTA 必須使用現有 Normal Bank build/apply/inspect 引導；兩種圖保持獨立 legend、scale 與 metadata，不融合分數。

## 12. Progress、取消與 stale state

### 12.1 同步 MVP

穩健檢查以工作單元顯示單調遞增進度：

```text
1/4 Canonical prediction 與 clean CAM
2/4 Stability variants 3/7
3/4 Occlusion variants 8/22
4/4 Aggregate、metrics 與 export metadata
```

若仍在單一 blocking Streamlit run 中，不顯示假的「停止」按鈕。

### 12.2 Async phase

完成 background job registry 後才加入：

```text
queued / running / cancel_requested / partial / completed / failed / stale
```

取消只能在 variant/microbatch 邊界 cooperative 生效。Partial 結果可下載但永不顯示通過；需記錄完成／未完成項與 cancel timestamp。

### 12.3 Result signature

Result key 至少包含：

- source SHA-256 / data generation；
- head SHA-256（coef、intercept、classes、temperature、recipe）；
- checkpoint SHA-256；
- preprocess/scope/crop/pad/pooling/target resolution；
- canonical target/reference；
- profile id + profile version；
- variant spec、seed、faithfulness config；
- algorithm/schema version。

任一欄位改變，舊結果必須隱藏或明確標 stale；不能只 warning 後繼續顯示舊 overlay。Opacity、grid、折疊狀態不屬於 inference signature。

## 13. API 與 schema v3

建議新增 immutable config／result types：

```python
@dataclass(frozen=True)
class GradCAMProfile:
    profile_id: Literal["quick_v1", "robust_v1"]
    cam_variants: tuple[PerturbationSpec, ...]
    faithfulness: FaithfulnessConfig
    thresholds: StabilityThresholds
    seed: int

@dataclass(frozen=True)
class GradCAMTargetSpec:
    mode: Literal[
        "top1_vs_runner_up",
        "class_vs_strongest",
        "class_vs_class",
    ]
    target_class: str | None
    reference_class: str | None

@dataclass
class StabilityResult:
    status: str
    target_win_rate: float
    fixed_margin_positive_rate: float
    median_signed_spearman: float | None
    top_positive_jaccard: float | None
    salient_sign_agreement: float | None
    variant_records: list[dict]

@dataclass
class GradCAMResultV3:
    prediction: PredictionSummary
    target: ContrastiveTarget
    profile: GradCAMProfileSummary
    stability: StabilityResult | None
    faithfulness: FaithfulnessResult
    head_audit: HeadAuditSummary
    provenance: Provenance
    timing: TimingBreakdown
    raw_patch_scores: np.ndarray
    display_patch_scores: np.ndarray
    consensus_raw: np.ndarray | None
    uncertainty_map: np.ndarray | None
```

Runtime model/tensor handle 不得進 export schema。Export envelope：

```json
{
  "schema_version": 3,
  "kind": "token_gradcam",
  "status": "completed|partial|stale|failed",
  "profile": {},
  "provenance": {},
  "geometry": {},
  "prediction": {},
  "target": {},
  "quality": {},
  "raw_patch_scores": [[]],
  "display_normalization": {},
  "warnings": []
}
```

16×16、24×24、32×32 raw grids 可直接放 JSON；per-variant grids 可放手動下載的 NPZ/evidence package。

## 14. 匯出

快速探索保留：

- overlay PNG；
- metadata JSON，增加完整 raw signed grid、target/reference、probabilities、margins、profile、gate 與未執行檢查。

穩健檢查新增手動下載 `evidence package`：

```text
summary.json
quick_overlay.png
consensus_overlay.png
occlusion_checks.csv
stability_checks.csv
raw_grids.npz
```

不可稱「認證報告」。預設不包含輸入影像 pixels；若未來允許，需使用者另行勾選。所有匯出均由使用者觸發，不寫入 dataset 或既有 LV cache。

## 15. 工程安全 P0：穩健模式上線前必做

### 15.1 Per-model lease

目前 shared explainer model 會掛 temporary hooks，但 Grad-CAM 全段未被 model lock 保護。多 session 可能發生 hook/activation 串線。

必須用 per-model `RLock` 包住：

```text
hook install
→ canonical/variant forward
→ backward/autograd.grad
→ hook remove + graph release
→ faithfulness forward chunks
→ lease release
```

同一 GPU model 預設 concurrency=1；CPU 同一 model 亦先 serialize。若要 parallel，只能受 RAM budget 控制地建立獨立 model instance。

### 15.2 Strong hashes 與 stale-hide

- checkpoint 由 filename/size/mtime 升級為 content SHA-256，並以 path/size/mtime_ns memoize。
- head hash 包含 coef、intercept、classes、temperature、l2norm 與完整 recipe。
- source 使用 content SHA-256 或 caller data-generation id。
- Signature mismatch 必須 hide/return，不再只 warning 後顯示舊圖。

### 15.3 Autograd memory

- explanation-only model 不執行 optimizer/update；
- 改用 `torch.autograd.grad(score, captured_activation)`，避免 `score.backward()` 累積全部 parameter grad buffers；
- 每個 variant 後立即移除 hooks、物化 CPU map、釋放 graph；
- 測試完成與例外後所有 model parameter `.grad is None`。

### 15.4 Faithfulness OOM degrade

遮蔽 batch 發生 OOM：

```text
configured microbatch
→ retry microbatch=1
→ faithfulness unavailable(resource_exhausted)
```

核心 CAM 已成功時仍返回 CAM，但不得把 unavailable 當 supported。不可偷偷降解析度，因為那會破壞 classifier provenance。

### 15.5 Model cache

Explainer cache key 不包含 `pooling/head`；同 model/checkpoint/device/dtype 的 cls/meanpool 共用同一份 DINO 權重。Pooling 屬於 request。預設 device LRU max=1；evict 後才執行 device cache cleanup。

## 16. Resource budget

若一次 forward 記為 `F`、backward 約 `1.5–2F`：

| Profile | 理論工作量 | 相對目前 quick |
| --- | ---: | ---: |
| Plain CAM | `1F + 1B` | 約 0.4–0.5× |
| Quick v1 | `1F + 1B + 3F` | 1× |
| Robust v1 | `7(F+B) + up to 22F` | 約 6–8× |

GPU batching 只能降低 wall time，不能降低 FLOPs。Occlusion 使用可配置 microbatch，預設 CPU=4、CUDA=8，OOM 時降至 1。

初步本機 x64 CPU、vits14/224 量測（2026-08-30，僅供 benchmark 起點，不是 SLA）：

- cold model load：約 6.96–9.72 s；
- warm Grad-CAM core：約 0.64 s；
- warm core + 3-variant quick faithfulness：約 0.81–0.87 s。

正式 UI 不顯示由這組數字推導的固定時間；需完成目標 Snapdragon/CUDA matrix 後才定 SLA。

### 16.1 Resolution guard

Grad-CAM 必須使用 head provenance 的 resolution。GUI 不得只為了更細熱圖把 224 改成 336/448；若要新 resolution，必須重算 embedding、重訓並驗證 head。

- 224 / patch14：16×16 = 256 tokens。
- 336 / patch14：24×24 = 576 tokens。
- 448 / patch14：32×32 = 1024 tokens。

Attention memory 隨 token pair 接近平方增長。Robust preflight 必須按 model/device/token/microbatch 估算 work budget；超限時明確建議 Quick 或建立相容的低解析度 head，不可 silent fallback。

## 17. Error model

Typed error codes：

```text
INVALID_INPUT
MISSING_CHECKPOINT
INCOMPATIBLE_HEAD
UNSUPPORTED_MODEL
RESOURCE_LIMIT
NUMERICAL_ERROR
CANCELLED
INTERNAL
```

每個 error 含 `retryable` 與使用者可操作建議。只有 audit resource failure 可降級保留核心 CAM；核心 forward/backward、provenance 或 gradient 失敗不可出圖。

## 18. Test plan

### 18.1 Unit

- Quick/Robust profile defaults、hash 與 schema round-trip。
- Canonical target/reference 在所有 variants 固定。
- target/reference tie 使用 stable class-index rule。
- Raw median aggregation 不先各自 normalize。
- MAD、sign agreement、Spearman、top-k Jaccard 的 normal/uniform/insufficient cases。
- 空間 shift inverse alignment 與 non-square grid orientation。
- All-negative CAM 不產生紅色 evidence。
- Variant pixel range、fixed seed、deterministic replay。
- Occlusion top/random/negative 不重疊；shape-matched random deterministic。
- Deletion curve/AOPC、alternate baseline direction、near-zero margin 不作除法。
- Robust early stop 與 research override。
- 所有 success/exception/OOM 後 hooks 清空、parameter grads 為 None。
- Batch OOM → microbatch1 → typed degrade。
- Source/head/checkpoint/profile 改變使結果 stale；opacity/grid 不 stale。

### 18.2 Concurrency

- TinyViT 兩 thread + barrier：結果各自等於 serial，無 hook 洩漏或死鎖。
- 同一 model concurrency=1；不同 model lease 可獨立。
- 快速切圖、改 target、重訓 head 後 late result 不得覆蓋新 request。
- Cache eviction 不留下 dangling hook/tensor。

### 18.3 Integration

- 真 DINO cls/meanpool logits 與既有 `predict_head` parity。
- cls/meanpool 共用同一 model instance。
- 224 square 與 keep-aspect non-square。
- CPU deterministic；CUDA 以 map sign/rank tolerance 驗證。
- Result cache hit 不執行 forward。
- Faithfulness OOM 仍保留 CAM 且明確 unavailable。

### 18.4 E2E

- GUI 預設 Quick，且沒有「較慢＝更準」文案。
- Robust 顯示工作量、進度、四層 evidence 與 consensus map。
- Low OOF 保留研究用途限制提示；不升級為可靠模型結論，也不要求重複確認後才可查看證據。
- Source/head/target/profile 改變後舊圖消失。
- 兩 browser sessions 不串圖。
- Export package 可 round-trip，含 hashes、seed、threshold、status。
- Synthetic planted cue：top deletion 應高於 random controls。
- MVTec `poke/012` 只驗證 low-margin/OOF mismatch 的保守文案，不以 mask overlap 當 Grad-CAM 正確性標準。
- 持續驗證不寫 dataset、embedding cache、anomaly cache 或 Normal Bank。

## 19. Benchmark matrix

| Dimension | Values |
| --- | --- |
| Model | vits14、vitb14 |
| Device | x64 CPU、Snapdragon X、CUDA（若有） |
| Resolution | 224、336、448 |
| Geometry | square、極端 non-square、object crop |
| Pooling | cls、meanpool |
| Profile | quick、robust、robust early-stop |
| Cache | cold load、warm、result-cache hit |
| Concurrency | 1、2、4 sessions |
| Failure | audit OOM、cancel、stale completion |

收集：p50/p95、load/preprocess/forward/backward/audit/render timing、peak RSS、CUDA allocated/reserved、forward count、cache hit、OOM、cancel latency、prediction parity、map rank/sign parity。

CI 初始採相對 gate：warm latency 或 peak memory 相對 baseline 回歸 >15% 則 fail；產品絕對 SLA 待 Snapdragon 實機校準。

## 20. Implementation phases

### P0 — Correctness prerequisites（2.5–3.5 人日）

- per-model lease + concurrency tests；
- source/head/checkpoint strong hash；
- stale mismatch hide；
- model cache 去除 pooling duplication；
- `autograd.grad` 與 hook cleanup；
- faithfulness OOM microbatch/degrade。

完成後才可稱「單機、多 session 安全的同步 MVP」。

### P1 — Profiles、schema v3、Robust engine（2–3 人日）

- profile/config dataclasses；
- deterministic perturbations/inverse alignment；
- median/MAD/stability metrics；
- robust deletion/AOPC；
- typed results/errors；
- evidence package。

### P2 — GUI 與同步 UX（1.5–2.5 人日）

- level selector、work estimate、gating；
- progress、result cards、charts；
- quick → robust upgrade；
- Normal Bank CTA；
- focused unit/integration/E2E。

### P3 — Async cancellation 與 session result LRU（2–3 人日）

- Future/job registry + generation id；
- cooperative cancellation；
- partial/stale lifecycle；
- byte-bounded per-session result LRU。

### P4 — Benchmark/calibration/release gate（2–3 人日，不含實機排程）

- x64/Snapdragon/CUDA matrix；
- Capsule 與其他產品 grouped validation；
- threshold calibration；
- real E2E、docs 與 release evidence。

P0+P1+P2 的同步可用版本約 6–9 人日；再加入 async cancellation 與完整硬體／發布 benchmark，總預算約 10–15 人日。若只維持目前單使用者研究工具，最低安全修補約 3 人日。建議先核准同步版本，再以 Robust UX 實測決定是否投入 async。

## 21. Acceptance criteria

1. GUI 只有「快速探索（單次歸因）」與「穩健檢查（重複驗證）」，沒有精準／高品質暗示。
2. 預設 target/reference 等於 canonical top-1/runner-up；所有 variants 固定同一 pair。
3. 非 top-1 target 的 UI、result、JSON 均為 counterfactual。
4. Quick 只顯示探索性／quick-check 狀態，不得顯示最終可靠通過。
5. Robust 實際執行 7 個指定 variants；設定、seed 與完成數寫入 metadata。
6. Aggregation 使用 aligned raw signed median；禁止 per-map normalize 後平均。
7. Stability metrics 對 insufficient positive evidence 不偽造數字。
8. Robust deletion 符合指定 fractions、random、negative 與 alternate baseline；減少工作量時必須 partial/degraded，不得仍標 Robust complete。
9. Low-margin、low-OOF 與 sample mismatch 的上層狀態不得被 faithfulness/stability 成功覆蓋。
10. Provenance mismatch、target=reference、zero/nonfinite gradient 等 hard block 不出圖。
11. Source/head/checkpoint/preprocess/target/profile/config 變更後舊結果隱藏；display-only 控制不重跑。
12. 同一 shared model 的 hooks/forward/backward/audit 受 per-model lease 保護；concurrency test 無串圖。
13. 每個 variant/例外後 hook 移除，model parameter grads 為 None。
14. Audit OOM 可保留核心 CAM，但 faithfulness 必須 unavailable；核心 OOM 不出圖。
15. Robust 過重時只能明確拒絕、early-stop 或由使用者改選 Quick；不可 silent 降解析度或少跑後冒充完成。
16. Result schema v3 包含 raw grid、full provenance、profile、thresholds、seed、timing、status，且可 round-trip。
17. Progress 單調遞增；blocking MVP 不顯示假的取消按鈕。
18. Signed legend 固定：紅支持 target、藍支持 reference、透明中性；紅色不得稱異常或 defect。
19. Normal/Unknown cascade 不做 Grad-CAM；Grad-CAM/Attention/Anomaly Heatmap 不融合或共用 scale。
20. 所有執行與匯出不修改來源資料、既有 LV cache、anomaly cache 或 Normal Bank。

## 22. Open questions requiring real-data calibration

- Capsule 以外產品是否允許相同的 ±0.25 patch shift 與 ±3% brightness？
- 0.60 Spearman、0.40 top-k Jaccard、0.75 sign agreement 是否能區分可用與不可用案例？
- Blur replacement 對各產品是否比 mean-color 更少造成 OOD response？
- 3 個 random controls 是否足以支援工程決策，哪些場景需要離線 20+ trials？
- 336/448 recipe 是否真的改善小 defect classifier，而不只是增加運算？
- Snapdragon X 的 acceptable p95、RAM ceiling 與可接受 queue wait 是多少？
- 何時需要從 Grad-CAM 升級到 segmentation、Transformer relevance propagation 或專門的 defect localization model？

在以上問題校準前，所有 threshold 必須標 `provisional`，不得寫成產品保證。
