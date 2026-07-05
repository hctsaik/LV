# M14 設計:以樣搜樣特徵器解耦(免整包建模,B+E)

PRD:[2_PO_PRD/fewshot_extractor_decouple_prd.md](../2_PO_PRD/fewshot_extractor_decouple_prd.md);
素材:[FEWSHOT_EXTRACTOR_DECOUPLE_PLAN.md](../FEWSHOT_EXTRACTOR_DECOUPLE_PLAN.md)。

## 目的(一句話)
讓「以樣搜樣」的 retrieve 只依賴**特徵器身分(model 名 + target_res + object_source)**,
免去 anomaly bank/projection,使用者不必先到『瑕疵偵測』建整包模型。

## 核心資料流(解耦後)
```
樣本夾 ──build──▶ sample_bank{vectors,labels,provenance, model,target_res,object_source}  ← 特徵器身分的唯一 artifact
                        │
目標夾 ──scan(retrieve)─┤ run_batched(objective="retrieve",
                        │   feature_extractor={model,target_res}, object_source=<bank.object_source>,
                        │   ref_vectors=bank.vectors, ref_labels=bank.labels)  ← 不再需要 model_dir/bank
                        ▼
                    佇列(best_class + best_sim 排序)
```

---

## E1 — `sample_bank` 自描述(加 object_source)
**檔案**:scripts/sample_bank.py。**Tier B**。既有 13 測無回歸。

### I/O 契約(加法)
- `build_sample_bank(...)` 回傳 dict 增加鍵 `"object_source": str`(值=傳入的 object_source,預設 "yolo")。
- `save_sample_bank(bank_dir, bank)`:`meta.json` 增寫 `"object_source"`(缺 → "yolo")。
- `load_sample_bank(bank_dir)` 回傳 dict 增加鍵 `"object_source"`;**舊集(meta 無此欄)→ 回填 `"yolo"`**(向後相容,不 raise、不 bump schema)。
- `append_sample` / `assert_model_compatible`:**簽名與行為不變**(object_source 不進相容檢查——它由自描述使用保證一致)。

### Acceptance(單元;tests/test_sample_bank.py)
- **AC-OS-1**:`build_sample_bank(..., object_source="whole_image")` 回傳 dict 的 `bank["object_source"]=="whole_image"`。
- **AC-OS-2**:`save`→`load` 往返後 `load(...)["object_source"]` 等於存入值(如 "whole_image")。
- **AC-OS-3**(向後相容):手寫一份**不含 object_source** 的舊 meta.json + emb.npz → `load_sample_bank` 回 `object_source=="yolo"`(不 raise)。
- **AC-OS-4**(不變量):既有 13 測全綠(build/save/load/append/compat/training_head_ready 皆無回歸)。

---

## E2 — `al_batch` retrieve 免 bank(核心,最高風險)
**檔案**:scripts/al_batch.py。**Tier B**。**既有 26 測無回歸鐵則**;objective-guarded 加法,**絕不碰 novelty/uncertain/confusion/similar 路徑**。

### I/O 契約(加法)
1. **新純函式** `_extractor_version(model, target_res, object_source) -> str`:
   `sha256(f"{model}|{int(target_res)}|{object_source}")` 前 16 碼。用於 retrieve 的續跑身分(取代 bank 檔雜湊)。
2. **新函式** `load_feature_extractor(model, target_res, object_source="yolo") -> dict`:回一個「lite fm」:
   `{"meta":{"model":model,"target_res":int(target_res),"object_source":object_source}, "model":str(model),
   "target_res":int(target_res), "score_mode":"object", "bank":None, "ref":None, "head":None,
   "model_version":_extractor_version(model,target_res,object_source)}`。**不讀任何檔、不需 bank/projection。**
3. `run_batched` 新增 keyword `feature_extractor: dict|None = None`(含 "model"、"target_res");`model_dir` 改為 **可選**
   (`model_dir=None` 預設)。載入分流:
   - `objective=="retrieve"` 且 `feature_extractor` 給定 → `fm = load_feature_extractor(feature_extractor["model"],
     feature_extractor["target_res"], object_source)`(**免 bank**)。
   - 否則(含所有既有 objective、及 retrieve 但未給 feature_extractor 的舊呼叫)→ `fm = load_frozen_model(model_dir)`(原路,向後相容)。
   - `feature_extractor is None and model_dir is None` → `raise ValueError`(明確錯,不隱性崩)。
4. `_score_batch` 對 `objective=="retrieve"` **短路**:**不算 anomaly 分數、不碰 bank/ref**——只
   `oe=_object_embeddings(meta, fm["model"], embed_fn, cache_path=obj_cache)` → `multi_ref_similarity` →
   回 `(np.zeros(n,float32), None, {"best_sim","best_class"})`。其他 objective 分支**原封不動**。

> 續跑身分:`fm["model_version"]` 對 retrieve 已來自 `_extractor_version`(綁 model+res+object_source),
> 流進既有 `_resume_identity`(不改該函式);疊加既有 retrieve `ref_key`(樣本集雜湊)→
> **換特徵器或換樣本集都會令續跑身分改變 → 不吃 stale shard**。

### Acceptance(單元;tests/test_al_batch.py)
- **AC-RL-1**(免 bank 跑得動):建一個**只有 model 名、無 bank.npz/projection** 的情境——
  `run_batched(image_paths, model_dir=None, feature_extractor={"model":M,"target_res":224},
  objective="retrieve", ref_vectors=R, ref_labels=L, embed_fn=<注入>, checkpoint_dir=ck, object_source="yolo")`
  → 正常回傳、`topk_records` 每筆帶 `suggested_class`+`similarity`,**全程未觸及 load_frozen_model / bank**。
- **AC-RL-2**(不算 anomaly 分數):同上結果每筆 `anomaly_score==0.0`(retrieve 不再算異常;排序仍由 best_sim 決定,
  相似物件排前)。
- **AC-RL-3**(model_version 綁特徵器身分):`_extractor_version("dinov2_vits14",224,"yolo") != _extractor_version("dinov2_vitb14",224,"yolo")`
  且 `!= _extractor_version("dinov2_vits14",224,"whole_image")`;同輸入穩定相等。
- **AC-RL-4**(換特徵器不吃 stale shard):同一 checkpoint_dir 先用 `feature_extractor A` 跑一輪、再用
  `feature_extractor B`(不同 model 名)跑,`on_identity_mismatch="restart"` → 續跑身分不同 → 清舊 shard 重跑
  (斷言:B 輪的結果不含 A 特徵器算出的舊項/或 state.resume_identity 改變)。
- **AC-RL-5**(缺兩者報錯):`run_batched(..., objective="retrieve", model_dir=None, feature_extractor=None,...)` → `ValueError`。
- **AC-RL-6**(不變量,反向稽查):`feature_extractor=None` 的既有呼叫路徑行為不變 → **既有 26 測全綠**
  (novelty/uncertain/confusion/similar/retrieve-with-model_dir 皆無回歸)。

---

## E3 — `al_service` 監看 retrieve 免 model_dir
**檔案**:scripts/al_service.py(+ 最小 scripts/al_workspace.py)。**Tier B**。既有 al_service 10 / al_workspace 14 測無回歸。

### I/O 契約(加法)
1. `al_workspace.load_profile`:`model_dir` 必填檢查**放寬**——`objective=="retrieve"` 時 model_dir 可空
   (特徵器身分改由 sample_bank 提供);其他 objective 維持必填。`default_profile`/`save`/`init` 不變。
2. `al_service.run_once`:`objective=="retrieve"` 時**不呼叫 `load_frozen_model`**——
   `bank=sample_bank.load_sample_bank(profile["sample_bank_dir"])` → `object_source=bank["object_source"]`、
   `feature_extractor={"model":bank["model"],"target_res":bank["target_res"]}`、`ref_vectors/ref_labels=bank`;
   `run_batched(..., model_dir=None, feature_extractor=feature_extractor, object_source=object_source, ...)`。
   其他 objective 路徑(先 `fm=load_frozen_model` 取 object_source)**原封不動**。

### Acceptance(單元;tests/test_al_service.py)
- **AC-SVC-DEC-1**(監看免 anomaly 模型):`init_workspace(objective="retrieve", sample_bank_dir=<存好樣本集>, model_dir="")`
  → `load_profile` 不因 model_dir 空而 raise;`run_once`(注入 embed_fn;樣本集自帶 model/res/object_source)→
  `status=="ok"`、佇列帶 `suggested_class`、**全程未載入任何 anomaly bank**。
- **AC-SVC-DEC-2**(不變量):既有 AC-SVC-RET 與其餘 al_service 測全綠;al_workspace 14 測全綠(retrieve 以外 model_dir 仍必填)。

---

## B1 — 第 9 工具 GUI:預設特徵器 + 去死路閘
**檔案**:scripts/app.py。**Tier B(真實 E2E)**。既有 fewshot E2E 無回歸。

### 行為契約
1. `_fewshot_search_ui`:**移除**「需先到瑕疵偵測建模」的死路 `return`。改為:若 `anomaly_model._dir` 不存在,
   仍可進三步流程(特徵器用預設)。
2. `_fewshot_step_samples`(①):
   - 預設特徵器 `dinov2_vits14`、target_res 224;進階者在 `st.expander("進階:特徵器")` 可換 DINOv2 變體(Should B2)。
   - object_source **inline 開關**:「物件(YOLO 框)/ 整張影像」→ 決定 build 的 object_source(寫進樣本集自描述)。
   - `_fewshot_build_bank_execute`:model/target_res/object_source **改讀本步 UI 選擇**(不再讀 `anomaly_model.meta`);
     若使用者確實載了 anomaly 模型,expander 可「沿用已載模型的特徵器」(選項,非強制)。
3. `_fewshot_scan_execute` / `_fewshot_add_to_bank` / 監看(④):特徵器身分一律**改讀樣本集自描述**
   (`load_sample_bank` 的 model/target_res/object_source),`run_batched`/`run_once` 走 E2/E3 的 feature_extractor 路徑,
   `model_dir` 不再必要。
4. 導流訓 head(Should B3):把①選的特徵器 model 名記進 session;使用者若點去『瑕疵偵測』建模,預填同 model
   +一行 caption 提醒「用同一特徵器,確認才能導流訓頭一致」。守紅線 2(避免 head 訓在不同空間)。

### Acceptance(真實 E2E;tests/e2e/test_fewshot_decouple_e2e.py)
- **AC-D1**(死路拆掉,旗艦):**乾淨 session、從未建任何 anomaly 模型** → 進「以樣搜樣」→ 選樣本夾 → 建樣本集
  → 選目標夾 → 海掃 → 佇列出現相似物件(帶建議類別+相似度)。**全程不進『瑕疵偵測』、無 stException**。
- **AC-D2**(檢索正確性不變):AC-D1 的佇列中,與樣本同類的物件真的排在前面(相似度 ≥ θ 命中非空、含樣本類名)。
- **AC-D3**(換特徵器續跑不吃 stale;可與 E2 AC-RL-4 單元互補):E2E 層——①用預設變體建樣本集+掃一輪→改 object_source
  重建樣本集+重掃 → 結果依新 object_source(不沿用舊 shard)。〔若 E2E 成本過高,以 AC-RL-4 單元覆蓋為主,此條可標可選〕

### 邊界與錯誤處理
- 樣本夾無 YOLO 框且 object_source=物件 → 明確提示「此夾無 labels,請切『整張影像』或換夾」(不靜默退化)。
- 樣本集與海掃特徵器不符(理論上不會,因自描述)→ `assert_model_compatible` 仍守 model/target_res,raise 友善訊息。

## 相依(無環)
B1 → E1、E2、E3;E2 → E1(讀 object_source 概念,實際值由呼叫端傳);E3 → E1、E2。
拆分可各自驗收:E1 純 schema、E2 純引擎(注入 embed_fn)、E3 服務編排、B1 GUI E2E。

## 非目標(守住,不擴張)
不做全域共用特徵器(C)、不做 D 墊檔、不動其他 objective 的評分/續跑語義、不改 similarity/retrieval_export/prelabel 契約。
