# PRD:M14 以樣搜樣特徵器解耦(免整包建模)

素材:[FEWSHOT_EXTRACTOR_DECOUPLE_PLAN.md](../FEWSHOT_EXTRACTOR_DECOUPLE_PLAN.md)(多 agent 討論 + code 驗證)。
定位:M13「以樣搜樣」完成後的**再架構增量**——把「先去『瑕疵偵測』建/存整包模型」的前置稅拆掉。

## 三行自問
- **誰會用**:要用小樣本海撈大資料做 YOLO 預標的日常使用者(few-shot 主路徑,手上只有 query + haystack,**沒有** anomaly「正常參考批」)。
- **不做會怎樣**:主路徑被迫繞道一個語義不符的「瑕疵偵測」功能、建一份 retrieve 永不使用的 anomaly bank、
  多 2 次工具切換 + 多 2 個資料夾;海撈時每顆物件白算一份 anomaly 分數(大資料算力浪費)。分散感勸退。
- **成功的可觀察判準**(直接餵 /pm 當 AC 來源):
  1. **從沒有任何 anomaly 模型的乾淨狀態**,使用者能「選樣本夾 → 建樣本集 → 選目標夾 → 海掃出結果」全程可用,**中途不需進『瑕疵偵測』**。
  2. 解耦後**相似物件仍被排到佇列前面**(檢索正確性不變)。
  3. **換特徵器(model 名 / object_source)後續跑,不吃到舊 shard 的過期結果**(不 silent-stale)。
  4. al_batch 既有 26 測、sample_bank 13 測、al_service 10 測、fewshot E2E 全部**無回歸**。

## 目標與成功指標
- 到第一次海掃的操作成本:**14 步 / 3 夾 / 2 切換 → 5 步 / 1 夾 / 0 切換**。
- retrieve 海撈不再計算/依賴 anomaly bank(移除死算 + 移除 bank 硬依賴)。
- 樣本集自描述(meta 帶 model + target_res + object_source)= 特徵器身分的唯一 artifact。

## 範圍(MoSCoW)
- **Must**
  - E1 `sample_bank` meta 加 `object_source`(build 記錄、load 回填;舊集缺 → 預設 "yolo" 相容)。
  - E2 `al_batch` retrieve **免 bank/projection**:objective-guarded lite 載入(只需 model 名 + target_res),
    `_score_batch` retrieve **不算 anomaly 分數**,`_model_version`(或 resume identity)retrieve 改綁**特徵器身分**。
  - E3 `al_service.run_once` retrieve 同步走免 model_dir 路徑(特徵器身分取自 sample_bank)。
  - B1 第 9 工具 ①:內建預設 `dinov2_vits14` 特徵器 + object_source inline 開關 + **拿掉「需先建模」死路閘**;
    build/scan/加入樣本集/監看一律改讀**樣本集自描述**,不再依賴 `anomaly_model._dir`。
- **Should**
  - B2 進階者可在 `st.expander` 換 DINOv2 變體(仍鎖成樣本集屬性,build↔scan 一致)。
  - B3 導流回①訓 head 時**沿用同一 model 選擇**的銜接提示(守紅線 2:避免 head 訓在不同空間)。
- **Won't(本輪)**
  - 全域共用特徵器架構(視角 C:橫掃工具 1/2/8/9 + 監看,回歸成本過高,性價比最差)。
  - D 墊檔(建 retrieve 不用的死 projection = 技術債,與 E 目標衝突)。
  - 跨 DINOv2 變體 embedding 比對(維持不可比、明確擋)。

## 模組分解(交 /architect;皆 Tier B — 有 I/O / 契約 / GUI)
| # | Module | 一句話職責 | Tier | 無回歸鐵則 |
|---|--------|-----------|------|-----------|
| E1 | `sample_bank` 擴 | meta 加 object_source(自描述 + 舊集相容回填) | B | 既有 13 測 |
| E2 | `al_batch` retrieve 免 bank | retrieve 走特徵器 lite 載入、免算 anomaly、model_version 綁特徵器身分 | B | **既有 26 測(objective-guarded,不碰他者)** |
| E3 | `al_service` retrieve | run_once retrieve 免 model_dir、特徵器身分取自 sample_bank | B | 既有 10 測 |
| B1 | 第 9 工具 GUI | ① 預設特徵器 + object_source 開關 + 去死路閘;全流程改讀樣本集自描述 | B(真實 E2E) | 既有 fewshot E2E |

**拆模組品質自檢**:E1/E2/E3 各自對外介面明確(sample_bank schema / run_batched retrieve 參數 / run_once),
相依顯式(B1→E1,E2→E1,E3→E1、E2)不成環;/pm 可各自寫紅測試。E2 是唯一高風險點(動 26 測契約),
必須 objective-guarded 加法,先跑 `verify/gate.py --snapshot` 重建 baseline 再放行 /pg。

## 使用者驗收(銜接 User)
乾淨環境(無 anomaly 模型)打開 app → 點以樣搜樣 → 選樣本夾 → 建樣本集(一下就成)→ 選目標夾 → 海掃 →
看到相似物件排在前面 → ③ 確認/匯出。全程沒被要求去「瑕疵偵測」。

## 給架構師的開放問題
1. **retrieve lite 載入的介面形狀**:是給 `run_batched` 一個新參數(如 `extractor_id={model,target_res,object_source}`
   或 `sample_bank_dir`),還是讓 `load_frozen_model` 長一條「meta-only」分支?哪個對 26 測衝擊最小?
2. **`model_version` vs `resume identity`**:retrieve 的續跑身分該改 `_model_version`(影響 shard 命名語義),
   還是只擴 `_resume_identity` 的 ref_key/sig 納入特徵器身分?哪個更不動既有 novelty/similar 語義?
3. **object_source schema 遷移**:舊 sample_bank(無 object_source)load 時回填 "yolo" 是否安全?要不要 bump `_SCHEMA`?
4. **B3 導流一致性**的最小承接機制(session 記住特徵器選擇 → ①建模預填同 model),避免過度設計。

## 本輪 appetite
**≤4 模組(E1/E2/E3/B1)+ GUI,一輪做完**;B2/B3 為 Should,時間不足則進候選。
這是**修不可用/分散(UX 稅)+ 拆真實依賴**的輪次,核心 Must 應做到真能用(不受純「新增能力」appetite 限制),
但不擴張到 Won't 的全域架構。
