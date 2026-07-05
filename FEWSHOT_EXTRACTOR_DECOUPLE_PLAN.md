# 以樣搜樣「特徵器解耦」規劃(B+E)— 設計素材(非契約)

> 由多 agent 討論(2026-07-06,`wf_2da7ae02-78d`)+ 實地 code 驗證聚合而成。
> 目的:消除「以樣搜樣必須先到『瑕疵偵測』①建/存整包模型再回來選 dataset」的分散流程。
> 這是 M13 完成後的**再架構增量**,走完整 U-Net。本檔是給 `/po` 收斂 PRD 的素材,**不是契約**。

## 1. 問題(使用者原話)
「為什麼以樣搜樣要去先參考另一個功能、又要另一個功能存 model,然後再回到這個功能去選 dataset,流程很複雜跟很分散。」

## 2. 診斷(一句話)
**這是抽象副作用,不是真實技術依賴。** 以樣搜樣的檢索(retrieve)在程式層面只需要「一個特徵器身分」
——DINOv2 **model 名** + **target_res** + **object_source**;瑕疵偵測建模產出的 anomaly bank / coreset /
projection / head 在 retrieve 路徑上**零消費**。逼使用者先建整包模型,只是因為 M13 GUI 選擇復用 `anomaly_model`
當載體,而 `load_frozen_model` 又硬性要求 bank/projection 存在才肯載。

## 3. 實地 code 驗證(四點全成立)
1. **`al_batch.load_frozen_model`(scripts/al_batch.py:77-89)硬要 `bank.npz`(patch 模式)或
   `projection.good_obj_emb`(object 模式)**,缺就 `raise`。→ 這是「非得先建整包模型」的根源。
2. **retrieve 仍無條件算 anomaly 分數卻不拿來排序**:`_score_batch`(:140-153)每批先算 `scores`
   (patch 走 `score_object` 對 `fm["bank"]`;object 走 `compute_outlier_scores`),retrieve 真正的排序
   在(:165-170)用 `multi_ref_similarity` 的 `best_sim`。→ 算力白花,且**正因要算 scores 才需要 bank**。
3. **`_model_version`(:23-32)hash `bank.npz`/`head.joblib`/`meta.json`/`projection.npz`** 當續跑身分。
   → 拿掉 bank 後,retrieve 的續跑身分**必須重定義**(否則換特徵器重掃可能默默沿用舊 shard = stale,無報錯)。
4. **`sample_bank` 的 meta 只存 `model`+`target_res`+`provenance`,沒存 `object_source`**
   (scripts/sample_bank.py:49-51;現由 `load_bank(mdir).meta` 取得)。→ E 要自足,object_source 必須
   進樣本集 schema,成為樣本集的自描述屬性。

## 4. 實測代價(現狀)
到第一次海掃:**14 步、3 個資料夾(含一個對 few-shot 語義上不存在的「正常參考夾」)、2 次強制工具切換**,
外加海掃時每顆物件一份用不到的 patch 評分。

## 5. 推薦方案:B(零設定預設特徵器 UX)構建在 E(引擎級 retrieve 自足)之上
- **E(引擎)**:讓樣本集 `meta.json` 成為特徵器身分的唯一 artifact;retrieve 短路掉 `load_frozen_model`
  的 bank/projection 要求,只需 model 名 + target_res(+ object_source);順手 guard 掉 retrieve 的死算 anomaly 分數。
- **B(UX)**:以樣搜樣 ① 內建預設 `dinov2_vits14` 特徵器(進階者在 expander 換);object_source inline 開關;
  **拿掉「需先到瑕疵偵測建模」的死路閘**。few-shot 本質就是「還沒 head」的場景,主路徑不該出現「模型」概念。
- **D(可選墊檔,非目標)**:若要極速止痛,可先在①內嵌「一鍵建特徵器」捷徑;但會留一個 retrieve 永不用的
  死 projection = 欠 E 的技術債。**本輪目標是 B+E 一次拆對,不做 D。**

### 重設計後 ① 使用者流程(目標)
1. 點「🎯 以樣搜樣」(不再撞死路)。
2. 選 few-shot 樣本資料夾。
3. (可選)無 YOLO 框 → inline 切「整張影像」;特徵器預設 vits14,進階 expander 可換。
4. 按「▶ 建立樣本集」——第一下就成,樣本集 meta 自帶特徵器身分。
5. 點「② 海掃」→ 選目標夾 → 海掃。
→ **5 步、1 夾、0 切換**(現狀 14/3/2)。

## 6. 不可為簡化拿掉的正確性紅線
1. **樣本集 build ↔ 海掃共用同一 model 名**(跨 DINOv2 變體 embedding 不可比)。守衛
   `sample_bank.assert_model_compatible` 已獨立於 anomaly model,保留。
2. **以樣搜樣 ↔ 導流回①訓 head 共用單一 model 選擇**——否則 head 悄悄訓在不同 embedding 空間 = silent-wrong。
   解耦後要用「共用的特徵器選擇」承接,不能退回各處自由亂選。
3. **object_source 一致**(樣本集內部 + 樣本集↔海掃)——不一致 → 相似度默默校爛(soft silent-wrong)。
   把 object_source 鎖成樣本集屬性。
4. **checkpoint 續跑身分 `model_version` 重定義**——retrieve 改綁「特徵器身分(model 名 + target_res +
   object_source)」而非 bank 檔雜湊。**這是解耦唯一要動腦的正確性點。**

### 可安全砍
Normal bank/coreset、head、projection、`target_res` 可調旋鈕(物件級恆 224,調了沒反應=UX 陷阱)、
強制落盤 `_dir`、retrieve 每顆物件的死算 patch 評分。

## 7. 模組分解(草案,交 /po 定 Tier/MoSCoW)
| 模組 | 加法 | Tier | 無回歸鐵則 |
|------|------|------|-----------|
| `sample_bank` 擴 | meta 加 `object_source`;build 記錄之;load 回填(舊集缺 → 預設 "yolo" 相容) | B | 既有 13 測無回歸 |
| `al_batch` retrieve 免 bank | `run_batched(objective="retrieve")` 走「特徵器 lite 載入」免 bank/projection;`_score_batch` retrieve guard 掉 anomaly 分數;`_model_version` retrieve 改綁特徵器身分 | B | **既有 26 測無回歸鐵則**;objective-guarded 加法,不碰其他 objective |
| `al_service` 監看 | run_once retrieve 同步走免 model_dir 路徑(從 sample_bank 取特徵器身分) | B | 既有 10 測無回歸 |
| 第 9 工具 GUI | ① 內建預設特徵器 + object_source 開關 + 拿掉死路閘;build/scan 改讀樣本集自描述而非 anomaly_model | B(真實 E2E) | 既有 fewshot E2E 無回歸 |

## 8. 一個要盯的風險
**`model_version` 續跑身分重定義**——解耦後唯一會靜默出錯的點。E2E 必須專門加一條「換特徵器後續跑不吃 stale shard」。
(`ref_key` 那半已正確涵蓋樣本集向量,不用動;要動腦的只有 `model_version`。)

## 9. 驗收方向(交 /pm 落 AC)
- 單元:al_batch retrieve **無 bank/projection 也能跑**(給只有 meta 的 model 或直接給特徵器身分);
  換 model 名/object_source → `model_version`(或 resume identity)改變 → 不吃舊 shard。sample_bank meta 帶 object_source 往返。
- 真實 E2E:(a) **無任何 anomaly model 存在時,選樣本→海掃出結果全程可用**(證死路拆掉);
  (b) 相似物件真的排前面(證解耦後檢索不變);(c) 換特徵器續跑不吃 stale shard。
