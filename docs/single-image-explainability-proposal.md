# VisualLatent 單張影像可解釋性檢視：產品與架構建議

> 狀態：已實作（共用可解釋性面板；Attention／Grad-CAM／Anomaly Heatmap 依能力啟用）  
> 日期：2026-08-29  
> 範圍：LV 內的單張影像檢視；不修改 YOLO 偵測器或既有批次流程。

## 結論

建議分兩階段實作：

1. Phase 1：DINO Attention，回答「DINO 看這張圖時主要關注哪裡？」
2. Phase 2：Token Grad-CAM，只在存在明確、可微分且可追溯的 target 時，回答「為什麼模型判定為這個 defect？」

兩者不可混稱為同一種 heatmap。Phase 1 可獨立、離線、按需運作；Phase 2 必須先補足分類頭與前處理的可追溯性契約。

目前三種方法已由同一個單張影像面板承載。面板會依入口明確傳入的模型、
分類 head、資料範圍與 patch Normal Bank 狀態，顯示「可用」、「可先建立」或
「不可用」及其原因；不可用時不會把另一種方法的熱圖冒充成替代品。

## 功能應放在哪裡

功能主體應放在 LV 的既有單張圖片 viewer，不是 YOLO viewer 或 YOLO 偵測器。

- Visualize 的單張圖片 viewer 與瑕疵偵測的單張 inspector 共用同一個「可解釋性」面板，
  並各自帶入目前影像／物件與其能力狀態。
- 面板中的三種方法保留各自語意；入口只提供目前情境真正可用或可準備的操作。
- 使用者可選擇分析原圖，或目前 YOLO bbox 的物件 crop。
- YOLO 只提供物件位置與 crop 脈絡；本功能不更動 YOLO 模型、標註 viewer、推論或匯出流程。

這樣 DINO Attention 可同時用於整張圖與物件圖，也不會被綁死在 anomaly 功能中。

## 現況與事實基礎

目前程式的行為：

- scripts/models.py 的 Dinov2Extractor 輸出 CLS embedding，或將 patch token mean pool 成單一向量；公開呼叫路徑使用 torch.no_grad()。
- scripts/dinov2_hub/dinov2/layers/attention.py 使用 PyTorch SDPA（或可選 xFormers），只回傳 attention 後的 feature，不回傳 attention weights。
- scripts/anomaly_score.py 的 patch map 是 patch 到 Normal Bank 最近鄰的 cosine distance。
- scripts/dino_head.py 是 sklearn LogisticRegression，吃 NumPy embedding，沒有可供 autograd 追蹤的分類流程。

目前本機的 dinov2_vits14 為 ViT-S/14：12 個 blocks、6 heads、patch size 14、無 register token。224×224 輸入會得到 16×16 patch grid（256 個 patch，加上 1 個 CLS token）。實作仍須支援 register-token 模型與非方形輸入。

## 三種 heatmap 的語意

| 類型 | 它回答什麼 | 基礎計算 | 不代表什麼 |
| --- | --- | --- | --- |
| DINO Attention | DINO 的 CLS token 對哪些影像 patch 分配較多 attention？ | self-attention 的 CLS query → patch keys | defect、異常機率、因果證據、分類理由 |
| Anomaly Heatmap | 哪些 patch 最不像目前的 Normal Bank？ | patch-to-bank 最近鄰 cosine distance | DINO attention、已驗證瑕疵、類別證據 |
| Grad-CAM | 為何一個指定、可微分的 target 對某個類別／相似度得分較高？ | target 對中間 token activation 的梯度歸因 | 沒有 target 時的通用瑕疵圖 |

必須保留的產品語言：

- DINO Attention：DINO 關注區域。
- Anomaly Heatmap：相對於 Normal Bank 的不相似區域。
- Grad-CAM：指定 target 的類別證據。

不得把 Attention 的亮區稱為瑕疵，也不得把 anomaly map 稱為模型看哪裡。

## 產品範圍與優先順序

### Phase 1 MVP：最後一層 CLS Attention

首版只提供：

- 本機 DINOv2 模型。
- 單張圖片、使用者按鈕觸發。
- 最後一層的 CLS → patch attention。
- 所有 head 的平均值作為預設結果。
- 原圖、模型輸入圖、疊圖、patch grid 與完整 metadata。

選最後一層、head 平均的原因：

- 最容易對使用者說明，且結果穩定、成本低。
- 不會誘導使用者挑選某個「看起來最漂亮」的 head。
- 只需要計算 CLS query 的一列，不必保存完整 token-to-token matrix。

### Phase 1 進階選項

- 單一 head：可供研發診斷，不作預設，也不暗示某 head 更正確。
- Attention rollout：列為 experimental；只限單張、224 輸入，且須顯示較慢與較耗記憶體的提示。

Rollout 不是首版預設。它必須跨層保留完整 token-to-token attention，加入 residual 後再做正規化與乘積；這是一種啟發式聚合，不是更高等級的因果解釋。

### Phase 2：Token Grad-CAM

只有在下列其中一項成立時才開放：

- 有可微分的 defect classifier。
- 現有 sklearn head 已被轉為已驗證的 frozen PyTorch adapter。
- 有已定義、可微分、具 provenance 的 similarity target。

不應直接為以下結果產生 Grad-CAM：

- 正常：這是 Normal Bank gate 的結果，不是分類頭的一個類別。
- Unknown：這是 anomaly 分數與低分類信心共同形成的 cascade 結果。
- 任意未定義的異常程度：現有 bank／nearest-neighbour 管線不是可微分 target。

## UI/UX 草案

### 入口與互動

1. 使用者在 LV Visualize viewer 選一張圖或一個物件。
2. 選擇分析範圍：原圖或物件 crop。
3. 選擇模式並按「分析此圖」。
4. 運算完成後可調 opacity、格線和顯示尺度；這些純顯示控制不應重新跑模型。

不得在使用者每次切換圖片時自動執行 DINO 推論。

### 模式與可用性

| 模式 | 啟用條件 | 顯示內容 |
| --- | --- | --- |
| DINO Attention | 已選本機 DINOv2 checkpoint | CLS attention、輸入圖、overlay、grid |
| Anomaly Heatmap | 目前結果有 patch Normal Bank | 現有 patch 不相似圖與 bank 說明 |
| 分類理由（Grad-CAM） | 相容 differentiable target 可用 | 實際 top-1／runner-up、對比 signed token Grad-CAM、OOF、遮蔽忠實度、head provenance |

當使用者選擇 Anomaly Heatmap、但目前情境沒有 patch Normal Bank 時，不應只顯示紅色錯誤。
面板需依 session 狀態提供下一步：尚無 patch 模型則前往「瑕疵偵測①建模」；已有 patch
模型但尚未套用則前往「②套用偵測」；已有結果則前往②選擇對應圖片。引導需說明正常參考
資料、patch 分數模式、無 YOLO labels 時使用整張影像，以及熱圖只能在相容的瑕疵偵測結果
inspector 中查看，不能讓 Visualize 靜默借用其他情境的 bank。

### 控制項與標示

- 方法：最後一層＋head 平均；進階模式可選單一 head／rollout。
- 圖像：原圖、模型實際輸入圖、overlay、patch grid 開關。
- 顯示：opacity、顯示 scale、raw score 摘要。
- Metadata：模型、checkpoint、輸入尺寸、patch size、grid、crop/pad、resize policy、耗時。
- 匯出：手動下載 PNG 加 JSON metadata；不要自動落盤。

### 必須顯示的警告

- Attention 不等於 defect，也不等於模型的因果理由。
- Anomaly map 受 Normal Bank、前處理與參考樣本品質影響。
- Grad-CAM 使用 signed 藍→透明→紅色盤：紅＝支持 target 相對 reference，藍＝支持
  reference 相對 target，透明＝近中性；紅色不代表異常嚴重度。
- 每張圖的自動色階僅供該圖閱讀，不能直接跨圖比較亮度。
- patch grid 是低解析度空間資訊，邊界不能被當成 pixel-accurate segmentation。
- 預設 target 必須是模型實際 top-1；手選非 top-1 時要標示為反事實假說，不能宣稱是
  「模型為何如此判斷」。
- Grad-CAM 僅解釋 target 相對最強競爭類別的局部 margin，不代表整體品質或機台判定。

### 可用於模型改善的證據門檻

熱圖本身不足以決定改善方向。介面需同時呈現以下三層證據：

1. **決策清楚度**：實際 top-1、runner-up、兩者 probability margin。預設暫定 margin
   小於 5% 視為近似平手，提示先改善邊界樣本、標註與類別定義。
2. **分類頭泛化**：session head 建立時執行 sample-level stratified OOF，顯示 balanced
   accuracy、macro-F1 與此圖的 OOF prediction。它不是 grouped holdout；同批次或近重複
   仍可能洩漏，正式評估應補 acquisition-group split。
3. **歸因忠實度**：遮蔽最紅 10% patches，並與隨機及最藍／最低分 patches 比較
   target-vs-reference logit margin drop。只有 top-red drop 較大時，才標示為局部 supported。

若任一層偏弱，UI 應明確顯示「不要依單張熱點調模型」。若三者足夠穩定，而熱區落在文字、
背景或膠囊邊界等非業務 cue，才建議以 ROI、反例、資料增強或去除捷徑作為改善方向。

## Phase 1 技術設計

### 模組邊界

建議新增框架無關的模組，而非將邏輯塞入 app.py：

    scripts/dino_explain.py
      - ExplainRequest / AttentionResult
      - attention_map(...)

    scripts/explanation_render.py
      - render_attention_overlay(...)
      - render_grid(...)

    scripts/app.py
      - 單張 viewer 的按鈕、session state、結果呈現

Phase 2 再新增：

    scripts/head_adapter.py
      - sklearn LogisticRegression → frozen torch adapter

    scripts/gradcam_explain.py
      - DifferentiableTarget
      - token_gradcam_map(...)

現有可重用的部分：

- ImagePreprocessor、DINO checkpoint loader、DINO transform。
- patch_features 的 grid 計算慣例。
- Pillow／NumPy heatmap 疊圖機制。
- Visualize 的 single-image viewer 與 Anomaly inspector 的選圖互動。

不可重用為 Attention 的部分：

- Normal Bank、score_object()、anomaly cache。
- sklearn head 的 NumPy inference。
- 既有 anomaly renderer 的「紅＝不正常」語意與 per-image min/max 色階。

### Attention weights 的取得方式

不要修改 vendored attention.py，也不要藉由停用 xFormers 改變全域推論 backend。

建議在單張 request 期間，對選定 block 的 attn.qkv 掛 temporary forward hook：

1. hook 取得真正 forward 所產生的 qkv tensor。
2. 取 CLS 的 query 與所有 key，計算 softmax(QKᵀ / √head_dim)。
3. 排除 token 順序中的 CLS 與 register tokens。
4. 將 patch 部分 reshape 為 (grid_h, grid_w)。
5. 在 finally 移除 hook。

token layout 必須視為：

    [CLS] + [R 個 register tokens] + [H/patch × W/patch 個 patch tokens]

目前 R 為 0，但實作不可把 patch 起點硬編碼成 1。共享模型時須以 lock 或 request-local instance 保護 temporary hook，避免並行 request 彼此污染。

### 建議的資料契約

    ExplainRequest
      source_image / source_id
      scope: whole_image | object_crop
      bbox + crop_policy（若適用）
      model_id + checkpoint identity
      preprocessing recipe
      method: last_layer_head_mean | single_head | rollout

    AttentionResult
      raw_patch_scores
      display_patch_scores
      grid, patch_size, token accounting
      model/checkpoint/preprocess provenance
      input/original geometry
      timing, warnings, schema_version

raw attention 與供畫圖的正規化 map 必須分開保存。CLS row 的 patch mass 可能小於 1，因為 attention 也會分給 CLS 本身及未來可能的 register tokens；亮度圖不能被誤解為每個 patch 的獨立 defect probability。

### 前處理與幾何一致性

同一張 map 必須清楚記錄並呈現：

- 原圖尺寸。
- crop bbox 與 pad policy。
- 模型輸入尺寸。
- 是否 keep-aspect。
- patch grid。

全圖既有流程可能拉成 224×224；物件流程則通常保留長寬比並吸附至 14 的倍數。解釋頁若要聲稱對應既有某次 embedding／head 判定，就必須使用同一份前處理 recipe。預設應把模型實際輸入圖與 heatmap 並排，避免使用者把拉伸後的 map 誤看為原始影像座標。

## Phase 2 技術決策：sklearn Head 與 Grad-CAM

現有 sklearn LogisticRegression 不能直接反向傳播，但不代表一定要重訓 head。

推薦順序：

1. 先做 PyTorch adapter：把 coef_、intercept_、L2 normalization 與 temperature 搬成 frozen PyTorch module。
2. 以 binary 與 multiclass 測試驗證 adapter 的 logits、probability、prediction 對既有 predict_head() 一致。
3. 只在相容性驗證成功後，以 `target logit - strongest-rival logit` 作為 Grad-CAM target；
   不使用 raw target logit，避免多類共同特徵被誤解為該類的判別理由。
4. 未來若需要 end-to-end fine-tuning，再新增原生 PyTorch head；不要靜默替換舊 sklearn artifact。

對 ViT 的 Grad-CAM 必須採 token-aware 設計。不能直接把 CNN 的最後 feature map 做法照搬，也不應只取最後輸出的 patch tokens：CLS target 對那些 tokens 的梯度可能為零。應在經過驗證的 transformer activation（例如最後 block 的 pre-attention / norm token tensor）取 activation 和 gradient，排除 CLS/register token 後再 reshape 成 patch grid。

### Phase 2 的 provenance 前置條件

現有 head artifact 只有 sklearn model、classes、l2norm、dim，以及少量 app 補充欄位；不足以保證可重現原分類判定。

新的可解釋 head metadata 至少要保存：

    model_id, checkpoint hash, embedding_dim,
    input scope, crop/pad policy, target_res, keep_aspect,
    pooling mode, image normalization,
    classes, l2norm, temperature,
    head artifact hash, schema version, build timestamp

舊 joblib 若缺少這些資訊，最安全的策略是：仍可照舊分類，但不開 Grad-CAM，或以明確標示為 legacy/unverified 的方式要求使用者重新建立相容 head。

## CPU-only 與離線策略

Phase 1 在 CPU-only、16 GB RAM 上可行，條件是：

- 單張、按需、batch size 1。
- 預設 dinov2_vits14、224 輸入、最後一層 head mean。
- DINO model 僅 lazy-load 並在記憶體保留一個 active explainer。
- 結果只存 session memory，不寫進既有 .lv_cache、embedding cache 或 anomaly cache。

風險與限制：

- vitb14 權重與 activation 較大；不應是 CPU-only 預設。
- attention 成本隨 patch token 數平方成長；336、448 以上須設解析度 guard。
- rollout 需跨層完整 attention matrix；只能作有明確上限的實驗功能。
- Grad-CAM 需要 autograd，RAM／延遲高於 Attention。
- 本機的 x64 CPU 數據不能承諾 Snapdragon X 的表現；Snapdragon 實機 benchmark 必須成為驗收條件。

功能可全離線：DINO checkpoint 已在本機，attention 計算、PyTorch 推論與 Pillow rendering 均不需外部服務。

## 驗收條件與測試策略

### Phase 1

- 選圖不會自動推論，只有「分析此圖」才運算。
- 224×224 vits14 產生 16×16 map；非方圖的 (grid_h, grid_w) 正確且不轉置。
- register-token slicing、single-head、head-mean、有限值、row sum 與 deterministic eval 有單元測試。
- hook 前後既有 CLS embedding 不變；例外後 hook 必定移除。
- 無 DINO、壞圖、不相容模型、超出解析度／記憶體限制時，顯示明確錯誤或停用，不靜默降級。
- 回歸驗證既有 embedding、cache、檢索、UMAP、anomaly pipeline 均未改變。

### Phase 2

- sklearn adapter 在 binary/multiclass、L2 normalization、temperature 下與現有 head 推論一致。
- 指定 target 的 gradient 非零、map shape 正確、token slicing 正確。
- signed map 不得以 ReLU fallback 或 min-shift 把全負證據製造成紅色正證據。
- head/checkpoint/preprocess/crop recipe 不相容時拒絕產圖。
- 正常、Unknown、缺 target、gradient 為零、壞圖、OOM 等狀況均有明確 UI 行為。
- UI 顯示實際 top-1、runner-up、margin、OOF、遮蔽檢查與 head provenance，避免把 CAM
  解讀為全域 anomaly 結論，或在證據不穩定時提供錯誤的調模方向。

## 尚待產品決策

1. 首版預設分析單位要以原圖還是 YOLO 物件 crop 為主？建議兩者都支援，但使用目前 viewer 的範圍為預設。
2. 是否在首版露出 rollout？建議不露出，待實機 benchmark 與真實使用需求確認。
3. 是否需要 PNG＋JSON 手動匯出作為稽核紀錄？
4. Phase 2 是否先支援現有 sklearn head 的 PyTorch adapter？建議是，但絕不取代既有 head。
5. 是否接受 legacy head 沒有完整 provenance 時停用 Grad-CAM？建議接受，避免產生看似合理但不對應原判定的 heatmap。
6. Snapdragon X 的可接受等待時間、最大輸入解析度與記憶體上限是多少？這將決定 rollout 與 Grad-CAM 的實際開放範圍。

## 不應承諾的事

- DINO Attention 不是 defect segmentation，也不是因果解釋。
- Anomaly Heatmap 不是 attention，也不是 defect ground truth。
- Grad-CAM 不會自動解釋 Normal Bank gate、Unknown 或沒有明確 target 的結果。
- 在未完成 Snapdragon 實測前，不應承諾固定秒數或所有模型／解析度都能流暢執行。
- 任何 heatmap 都應作為人工判讀與資料診斷輔助，而非自動放行／攔截的唯一依據。
