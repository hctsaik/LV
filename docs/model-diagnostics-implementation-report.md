# Model Diagnostics / Grad-CAM 實作與驗收報告

日期：2026-08-31  
OpenSpec change：`add-explainability-model-diagnostics`

## 結論先行

這個 capsule case 的主要問題不是單純把 Grad-CAM 色盤或解析度調錯，而是分類 head 對 `crack` 的決策證據很弱。既有六類探索結果顯示 OOF accuracy 約 49.2%、balanced accuracy 約 50.1%、`crack` recall 約 21.7%；`crack/001.png` 在 OOF 被判為 `squeeze`。因此熱圖即使數值正確，也只能解釋「目前這個分類 head 如何做出決策」，不能保證會聚焦在裂痕，更不能當作裂痕 mask。

本次已完成 P0 與 P1 的可交付部分：版本化 context/run/render 契約、Visualize → Model Diagnostics handoff、弱 head 限制提示、current/OOF/target/reference 分離、canonical signed Grad-CAM、blue-white-red 語意、stale generation protection、no-cache-write E2E，以及 224/336/448 的明確解析度選項。精確解析度 head 已支援「使用者明確選取時以前景建立 session head」；完整的跨解析度 benchmark、grouped OOF、learning curve、mask localization、MIL/segmentation 仍刻意保留為後續 phase。

## 已落地的行為

### 1. 版本化 reference-only context

- `ExplainContextV1` 只傳遞 dataset/source SHA-256、stable record key、scope、bbox/crop recipe、model/head/Normal Bank provenance 與 generation。
- 不把 PIL image、tensor、embedding、sklearn model 或 bank vector 放入 handoff。
- whole-image key 以 source SHA-256 為主；object key 再加入 object index、normalized bbox 與 crop policy。
- context 進入 session 前經 `context_contract()` canonicalize；`context_id` 使用 canonical JSON，排除 `created_at` 與 handoff array index。
- record resolver 依 source path/SHA、object index、bbox、record key 逐項比對，array index 只作 legacy fallback。

### 2. 分類證據與可靠性 warning

- 畫面分開呈現 ground truth、current top-1、runner-up、current margin、held-out OOF prediction 與 calibration status。
- head audit 提供 split、sample/fold 數、balanced accuracy、macro-F1、target precision/recall/F1/support，並明示 sample-level OOF 不是 grouped holdout。
- 若 head validation 不足，畫面保留 persistent warning；使用者仍可直接檢視分類器證據，但結果不會升級成可靠或定位結論。
- target 不等於 current top-1 時標成 counterfactual，並重新計算 target/reference 語意，不沿用舊 margin。

### 3. canonical signed Grad-CAM

- 產品主路徑是 channel-weighted signed Grad-CAM（target logit − reference logit margin，不做 ReLU）。
- token gradient-times-activation 保留在 Advanced controls，名稱明確標示為 experimental，不稱作 standard Grad-CAM 或「建議聚焦」。
- raw signed patch scores、display scores、target-positive display 分離；display-only 操作不改變 compute identity。
- signed legend 固定為：

  > Blue = lowers the target-reference margin; white = neutral; red = raises the target-reference margin.

  同時明示這是分類 margin evidence，不是 segmentation/anomaly mask。

- 對 negative/zero/positive、flat map、CLS/mean-pool、target/reference margin 有 regression tests；zero/flat 不製造假熱點。

### 4. Model Diagnostics workspace

- 新增頂層 `Model Diagnostics` route；沒有 handoff context 時是 actionable empty state，不會自動載入模型或啟動 inference。
- Visualize 只提供 reference-only CTA；handoff 不複製 source image、embedding 或 cache。
- 顯示 dataset fingerprint、image、model、scope、record key、Context ID、head/mask/Normal Bank/result status。
- Advanced analysis controls：
  - `224×224 → 16×16 patch`
  - `336×336 → 24×24 patch`
  - `448×448 → 32×32 patch`
- 在 workspace 選擇 224、336 或 448 只會更新待分析設定，不會觸發計算。按一次唯一的主操作（例如 `建立 448×448 模型並分析此圖`）才以前景進度狀態產生完整資料集的對應 embeddings、訓練 frozen-backbone exact head，並接續產生 Grad-CAM；不會靜默沿用 224 head，也不需要第二個熱圖或研究用途確認按鈕。
- 若該解析度建立失敗，該 head 保持 unavailable，不產生誤導性的 Grad-CAM。既有 224 head 只作 224 baseline；只有沒有完整 session 資料的 inspector 才會停用非匹配解析度，不以 spatial probe 冒充 exact head。
- opacity、patch grid 與 display mode 只重畫既有 result；compute key 不包含這些 render-only 欄位。

### 5. stale 與來源完整性

- Visualize generation 或 source SHA 改變時，頁面顯示 `Context stale` 與 stale fields，禁止舊 result 發布。
- fresh handoff 會產生新的 Context ID；舊 heatmap 與 head 不復活。
- dedicated E2E 對 source dataset 與既有 cache 做 fingerprint 前後比對，diagnostics 不寫入來源資料、既有 embedding cache 或 persistent explanation artifact。

## 五情境 multi-agent 驗收

固定五情境：

1. route empty / no-auto-run
2. Visualize handoff identity
3. weak-head warning / prediction semantics
4. signed Grad-CAM blue-white-red、render-only、no-cache
5. stale generation、fresh handoff、舊 result/auth 不復活

Round 1 平均未達門檻（UX 93.6、contracts 93.8），所以沒有停下來；修正了 resolver、per-class audit、研究限制提示、stale contract、重複 CTA 與 E2E remount 問題。

Round 2 仍有 E2E remount/locator 缺口，所以繼續修正；Round 3 已有代理人達 98.0，但另一份獨立 review 仍指出驗收深度不足，因此再做 Round 4。

Round 4 在最新程式與最新 E2E 上的五情境分數：

| reviewer | 五情境分數 | 平均 |
|---|---:|---:|
| UX reviewer | 100, 100, 99, 99, 99 | **99.4** |
| contracts reviewer | 100, 98, 98, 95, 97 | **97.6** |
| pipeline reviewer | 100, 98, 90, 98, 98 | **96.8** |

pipeline reviewer 的 90 分只扣在「沒有真正低品質 head fixture 的瀏覽器分支」；它不否定弱 head 限制本身，因為該限制已由 page/unit tests 與 OOF per-class metrics 覆蓋。三份 Round 4 平均均大於 95，達到停下條件。

## 驗證證據

```text
focused explainability/unit suite: 65 passed in 12.98s
full non-E2E suite: 767 passed in 116.99s
dedicated diagnostics Playwright E2E (including 336→448 exact-head builds): 1 passed in 48.73s
legacy explainability Playwright E2E (including 336 exact-head build): 1 passed in 68.45s
OpenSpec strict validation: 1 passed, 0 failed
```

### Latest guided-workspace validation

The user-facing flow was reorganized after the baseline above. The latest
regression run covers the selected-image → resolution selection → matching
head preparation → heatmap → interpretation path:

```text
focused page/contracts/UI tests: 32 passed in 4.50s
full non-E2E suite (excluding tests/e2e): 770 passed in 138.55s
Model Diagnostics Playwright E2E (one-click 224/336/448): 1 passed in 91.69s
legacy inline explainability Playwright E2E: 1 passed in 61.40s
OpenSpec strict validation: 1 passed, 0 failed
```

The diagnostics E2E proves that the selected-image handoff reaches the
workspace, that the 224/336/448 selector is visible before any head exists,
that changing to 336 exposes a combined action without training as a selector
side effect, and that one action at each of 224/336/448 performs exact-head
preparation plus heatmap generation without a redundant acknowledgement button.
It also verifies that the signed legend plus the render-only Compute key survive
a grid rerender. The
legacy inline E2E continues to cover the 336×336 exact-head path. These
336/448 heads are foreground, ephemeral research models; the P3 dataset-level benchmark,
grouped OOF, learning curves, and batch localization remain deferred.

Dedicated E2E 另外驗證：

- empty route 沒有 inference/OOF/faithfulness 自動執行；
- 選取前沒有 handoff CTA，選取後才建立 reference context；
- canonical signed legend 實際出現在瀏覽器；
- patch grid rerender 後 `Compute key` 不變；
- cache fingerprint 不變；
- stale/current compatibility 與舊 overlay 不復活則由 context/unit regression gate 覆蓋。

## 尚未宣稱完成的工作

這些項目仍保留在 OpenSpec 未勾選狀態，避免把研究方向誤報成已驗證功能：

- 六類 capsule dataset 的固定 fingerprint、完整 machine-readable evidence report；
- grouped OOF/holdout、learning curves（25/50/75/100%）、confidence intervals 與 leakage grouping；
- 對固定 dataset snapshot 完成 224/336/448 各自重新產生全資料 embeddings、各自訓練與驗證的 benchmark exact heads（目前已完成 session 內明確選取時的 336/448 exact-head build，尚未完成固定資料集 benchmark）；
- mask resolver、patch-level localization、faithfulness batch metrics；
- background jobs、atomic persistent explanation artifacts；
- patch classifier、MIL、segmentation 與 crack boundary/measurement。

因此對 `crack/001.png` 最準確的產品訊息仍是：「目前分類 head 對此樣本的判斷與歸因證據不可靠；Grad-CAM 沒有找到可被驗證的 crack localization。」下一階段應先用 learning curve 與 grouped validation 判斷資料量/資料分組問題，再決定是否投入 patch/MIL/segmentation，而不是繼續用更鮮豔的熱圖替代模型能力。
