# 08 — al_batch 技術設計(分批可續跑主動學習選樣引擎)

對應 PRD:[2_PO_PRD/al_scale_offline_prd.md](../2_PO_PRD/al_scale_offline_prd.md) · 里程碑 **M9** · Tier B
設計素材(非契約):[AL_SCALE_AND_OFFLINE_SERVICE_PLAN.md](../AL_SCALE_AND_OFFLINE_SERVICE_PLAN.md) §6.1

> 本設計經 workflow 對既有程式碼逐一紮根 + 對抗式挑批次正確性漏洞後定案。以下**設計決策**多是為了避開
> 已查證的 silent-wrong 陷阱;每條標了根據的 `file:line`。**PG 不得為省事違反這些決策**(尤其 §2.6「禁止事項」)。

## 1. 目的(Purpose)

在**凍結模型**下,把「發現物件 → embedding → 對凍結模型評分」**分批**跑完大資料(數萬~十萬物件),
每批落一個**不可變 checkpoint**,中斷後可**續跑不重算**,並在**任何時刻**給得出「已處理併集」的
(暫定)priority Top-K。**GUI 與離線服務共用本引擎**(憲法 C1)。

## 2. I/O 契約

### 2.0 憲法對映(本模組承擔哪幾條)

- **C1**(一引擎兩前端):本模組是唯一引擎,GUI/服務都呼叫 `run_batched`。
- **C4**(凍結模型):`model_dir` 是唯讀輸入,**永不** `train_head`/`build_memory_bank`/改模型。
- **C6**(不寫使用者資料夾):`checkpoint_dir` 開跑前先過 `assert_safe_bank_dir`;embedding 快取進 checkpoint 內。
- **C8**(分批正確性):批次迴圈**只做 per-item**;所有全域運算(_minmax 正規化、門檻、分群、ranking)只在
  「已處理併集」上、於 Top-K 合併時或 `done=True` 時跑一次。**單批內嚴禁全域運算。**

### 2.1 主入口 `run_batched`(放 `scripts/al_batch.py`)

```python
def run_batched(
    image_paths,                      # Sequence[Path]（來源影像;非物件）
    *,
    model_dir,                        # Path：GUI 一鍵存出的凍結模型目錄
    checkpoint_dir,                   # Path：工作/續跑目錄（必給;不得 None）
    objective: str = "novelty",       # "novelty" | "uncertain" | "confusion"
    k: int = 100,
    batch_size: int = 1000,           # 以「影像」為單位切批
    object_source: str = "yolo",      # "yolo" | "whole_image"
    class_names=None,                 # yolo 模式的類別名
    dataset_dirs=(),                  # Sequence[Path]：來源資料夾（給 assert_safe_bank_dir 重疊防呆）
    embed_fn=None, extractor=None,    # 測試注入（object / patch 兩路徑），確定性
    progress=None,                    # Callable[[dict], None]：每批回報（見 §2.5）
    resume: bool = True,
    on_identity_mismatch: str = "error",  # "error" | "restart"
    max_batches=None,                 # 可選:本次最多跑幾批就返回（讓 GUI 分次 rerun 不阻塞;None=跑到完）
) -> dict:                            # 回傳見 §2.4
```

### 2.2 凍結模型載入 `load_frozen_model`

```python
def load_frozen_model(model_dir) -> dict:
    """讀 GUI 存出的模型目錄，回傳批次評分需要的凍結物件。
    回傳 {"meta": dict, "model": str, "target_res": int, "score_mode": str,
          "bank": MemoryBank | None,       # patch 模式;object 模式為 None
          "ref": np.ndarray | None,        # object 模式的 good_obj_emb 參考;patch 模式為 None
          "head": dict | None,             # 有 head.joblib 才有
          "model_version": str}            # 見 §2.7
    """
```

- 用既有 `anomaly_bank_store.load_bank(model_dir)`(缺 `meta.json` 會 raise)→ 取 `meta.model/target_res/score_mode`。
  [根據 anomaly_bank_store.py:62-85]
- `check_compat(meta, model=meta.model, target_res=meta.target_res)` 自我一致性檢查(既有恆檢 model+target_res)。[anomaly_bank_store.py:88-100]
- `score_mode=="patch"` → 用 `bank.npz` 的 `vectors` 建 `MemoryBank`(凍結)。[anomaly_score.py:21-70]
- `score_mode=="object"` → `bank` 為 None;由 `projection.npz` 的 `good_obj_emb` 當 `ref`(external_ref)。
- `head = load_head(model_dir/"head.joblib")` 若該檔存在。[dino_head.py:120-122]

### 2.3 目標(objective)→ 所需凍結產物 + 每項 priority 成分(**開跑前驗證,缺就 raise**)

> **重大更正**(對抗 critique high #2):PRD/plan 原寫 objective `confusion`「需 labels」是**錯的**——既有
> `active_loop.confusion_targeted_priority(anomaly_scores, head_proba, ...)` 其實吃 **head_proba(entropy),不吃 labels**,
> 其 `anomaly_threshold` 參數是死碼。[active_loop.py:31-43] 為讓 al_batch **零 M10/workspace 相依**,本設計把三個
> objective 全部定義成「只吃凍結 bank/head、不吃 labels」:

| objective | 需要的凍結產物 | priority 向量算法(全 per-item 或全域正規化,**皆非門檻依賴**) | 缺產物時 |
|---|---|---|---|
| `novelty` | bank(patch)或 ref(object) | `_minmax(anomaly_scores)`(= `priority_score(scores, w_novelty=1, w_boundary=0, w_disagreement=0)`)[active_learning.py:35-54] | — |
| `uncertain` | bank/ref **且** head | `w_novelty*_minmax(novelty) + w_boundary*boundary_score(head_proba) + w_entropy*entropy_score(head_proba)`;**boundary/entropy 皆 per-item 絕對值**[active_learning.py:25-32;active_loop.py:22-28] | 無 head → raise `ValueError("objective 'uncertain' 需要含分類頭的模型")` |
| `confusion` | bank/ref **且** head | `confusion_targeted_priority(anomaly_scores, head_proba)` = `_minmax(novelty)+entropy_score(head_proba)`[active_loop.py:31-43] | 無 head → raise 同上 |

> **刻意不用 disagreement 項**(critique high #5):`priority_score` 的 disagreement 需要全域 `anomaly_threshold`
> (`gate_threshold` 是全域 quantile [anomaly_tool.py:265-284]),凍結 meta **沒存門檻**。單批算門檻違反 C8。
> 故 M9 一律 `w_disagreement=0`,uncertain 改用 boundary+entropy(皆 per-item 絕對值,批次安全)。
> (未來若要 disagreement:上游 model save 於 build 時把絕對門檻寫進 meta——那是對 model-save 擁有者的 reverse gate,不在 M9。)

### 2.4 `run_batched` 回傳 dict(釘死鍵)

```python
{
  "topk": [item_id, ...],            # 長度 = min(k, objects_scored);priority 由高到低
  "topk_records": [                  # 與 topk 同序、同長
    {"item_id": str, "image_path": str, "bbox": [cx,cy,w,h], "obj_index": int,
     "label": str, "score": float, "priority": float, "reason": str},
    ...],
  "objective": str, "k": int,
  "images_total": int, "images_processed": int, "objects_scored": int,
  "done": bool,                      # 全部批次完成
  "provisional": bool,               # == not done
  "diversity_applied": bool,         # 見 §3.4
  "model_version": str,
  "skipped": [str, ...],             # 不可讀影像（partition_readable 剔除）
}
```

- **`item_id`(critique high #7,rescore-in-place 的關鍵)**:內容定址、**排除任何模型/分數相依**,格式
  `f"{img_key}:{bx}:{class_name}:{obj_index}"`,其中
  - `img_key = _file_sha256(image_path)[:16]`(既有串流雜湊 [export_subset.py:102-107])——**不可用 stem**
    (patch/object 快取都用 stem 會跨資料夾撞名 [patch_features.py:80, object_eval.py:193])。
  - `bx = ",".join(f"{v:.3f}" for v in bbox)`(量化,對齊 `confirmed_to_fewshot` 的 (stem,bbox,label) 內容鍵精神 [anomaly_bank_store.py:118-123])。
  - 同一物件在**不同 model_dir** 重評分 → **同一 item_id**(id 不含 model)→ 服務可 diff 舊新、原地更新(憲法 C4/C1)。
- **`reason`(Should,人話理由)**:由 priority 成分挑主因的中文模板,例:
  `novelty` 高 → 「很不像正常樣本(異常 {score:.2f})」;`entropy` 高 → 「模型對這張拿不準」;
  `boundary` 高 → 「介於兩類邊界」。**純由已算成分產生,不新增運算。**

### 2.5 `progress` 回呼(每批一次)

```python
progress({"images_processed": int, "images_total": int, "objects_scored": int,
          "provisional_topk": [item_id, ...], "done": bool})
```
- GUI 用它畫「暫定·已處理 x%」進度 + 暫定 Top-K(PRD Must「邊跑邊看」)。

### 2.6 內部批次評分(**per-item,直接呼叫子函式,不呼叫 run_pipeline**)

> **禁止事項(critique + C8 風險彙整,PG 嚴禁違反)**:
> 1. **不得在批次迴圈呼叫 `run_pipeline`**——它每次無條件跑 `cluster_objects`(全域 HDBSCAN)+ normal_set +
>    `classify`(全域 quantile 門檻 + 全 ranking),連給 external_bank 也照跑 [anomaly_tool.py:87-92,163]。
>    每批呼叫 = 三個全域運算跑在部分資料上 = 直接違反 C8,且空批會 KeyError(N==0 回傳缺 `scores` 鍵 [anomaly_tool.py:69-73])。
> 2. **不得用 `candidates_in_reference=True`**(全域 leave-one-out,reference=批次自身)[anomaly_tool.py:138-145]——非批次可合併。
> 3. **不得每批重建 bank / 分群 / 算門檻 / 算 ranking**。
> 4. **不得傳 `cache_dir=None`**(會關掉兩層 .lv_cache,每次續跑重算 embedding [anomaly_tool.py:84, patch_features.py:71-73])。

每批(一段影像)做:
1. `good, bad = safe_io.partition_readable(chunk)`(全解碼;bad 併入 `skipped`)[safe_io.py:66-77]。
2. `meta = discover_yolo_objects(good, class_names)`(yolo)或 `discover_whole_images(good)`(whole_image)[interaction.py:1173/1203]。
3. **patch 路徑**(score_mode=="patch"):`all_pf = embed_objects_patch(meta, model, target_res=meta_res, cache_dir=<§2.8 快取>, extractor=extractor)`;逐 i `score = score_object(all_pf[i]["feats"], all_pf[i]["grid"], bank)[0]`(對凍結 bank,per-item **精確**)[patch_features.py:63; anomaly_score.py:76]。
   **object 路徑**(score_mode=="object"):`obj_emb = <embed_objects 或注入 embed_fn>`;`scores = compute_outlier_scores(obj_emb, ref, k=min(5,len(ref)), candidates_in_reference=False)`(對凍結 ref,per-item 精確)[anomaly_tool.py:126-132]。
4. **有 head 時**額外算 `head_proba = predict_head(head, obj_emb)[2]`(per-item;需 `obj_emb.shape[1]==head["dim"]` 否則 raise [dino_head.py:75-88])。patch 模式的 object embedding 另走 `embed_objects`(增量快取)取得 head 用的物件級向量。
5. 落 shard(§2.8),存**每項原始成分**(anomaly_score、head_proba;有需要 diversity 的 obj_emb f16),**不存正規化後的 priority**(_minmax 是全域,見 §3.3)。

### 2.7 `model_version`(凍結模型內容雜湊,critique gap #8)

- 對 model_dir 內**存在的承重檔**(`meta.json`、`bank.npz?`、`projection.npz?`、`head.joblib?`)各算
  `export_subset._file_sha256`,按檔名排序,把 `"name:digest"` 清單再 sha256(同 app.py:484 / interaction.py 既有 pattern)。
- **排除 `manifest.json`**(非原子寫 + 帶 `saved_at=time.time()` 不確定 [app.py:1411-1419])。
- **開跑時算一次**,存入 state.json;續跑時重算,**不符則 refuse**(見 §4)。誠實聲明:`np.savez` 會嵌 mtime,
  故此雜湊是「磁碟上有沒有變」的身分(C4 失效偵測),**不是語義等價**判斷。

### 2.8 Checkpoint 目錄佈局(重用既有原子寫 helper,不另造持久層)

```
<checkpoint_dir>/                         # 開跑前先 assert_safe_bank_dir(checkpoint_dir, selected_folders=dataset_dirs)
  cache_<model>_<res>/                     # .lv_cache embedding 快取（依凍結 meta 的 model/res 隔離，避免跨設定撞 stem）
  shard_<NNNNNN>.npz                       # 每批一個「不可變」檔;_atomic_npz 寫;NNNNNN = 確定性 batch_index（非遞增計數）
                                           #   arrays: item_id(str[]), anomaly_score(f32[]), [head_proba(f32[N,C])], [obj_emb(f16[N,D])]
  state.json                               # 續跑清單 + 完成 sentinel;_atomic_text 寫;每批「最後」才更新（鏡射 bank_store 的 meta.json-last 不變式）
  finalize.json                            # 僅 done=True 原子寫:全域門檻 + 最終 Top-K item_ids
```
- 原子寫直接 import `anomaly_bank_store._atomic_text` / `_atomic_npz`(temp 同目錄 + `os.replace`,同檔案系統原子)[anomaly_bank_store.py:185-194]。
- `state.json` 內容:`schema_version`、`resume_identity`(§4)、`model_version`、`objective`、`k`、`batch_size`、
  `object_source`、`images_total`、`images_processed`、`objects_scored`、`committed_batches`(list[int])、`done`、`ts`。

## 3. 資料流(Data Flow)

1. **開跑**:`assert_safe_bank_dir` → `load_frozen_model`(算 model_version)→ 驗 objective 產物(§2.3)→
   把 `image_paths` 做**確定性排序**(依 str)→ 算 `resume_identity`(§4)→ 讀/建 state.json。
2. **批次迴圈**:對每個未 committed 的 `batch_index`(切自排序後 image_paths):§2.6 評分 → `_atomic_npz` 寫 shard →
   `_atomic_text` 更新 state.json(committed_batches += index,**最後寫**)→ 呼叫 `progress`(附暫定 Top-K)。
   `max_batches` 到就返回(GUI 分次 rerun);否則跑到全 committed → `done=True` → 寫 finalize.json。
3. **Top-K 合併(暫定或最終)**:讀所有 committed shards → **依 item_id 排序**(canonical,消 tie-break 不確定,§3.1)→
   在併集上算一次 priority(§3.3)→ `select_for_labeling`(§3.4)→ 取前 k。
4. `read_checkpoint(checkpoint_dir, k=?)`:同 3,不再跑批次(GUI 開頁即讀暫定 Top-K)。

### 3.1 Top-K 確定性(critique high #1)

合併時**先依 `item_id` 字典序排序**再 ranking → 陣列位置成為 item 身分的確定函式 → `classify`/`select_for_labeling`
的「以位置 tie-break」等價於「以 item_id tie-break」→ 同輸入同模型,**跨前端/跨續跑/跨 batch_size 得到位元相同的 Top-K**。
**嚴禁**用 `glob shard_*.npz` 決定順序;一律從 state.json 的 committed 清單以固定排序組裝。

### 3.2 續跑精確性(critique high #6)

- 提交單位 = 一批:shard 原子寫**後**才原子更新 state.json(meta-last sentinel)。crash 在兩者之間 → 該 batch 未 committed →
  續跑**重跑該批**(凍結 bank 下確定性 → 同 bytes),`.lv_cache` 讓 embedding 不重算(embed_objects flush-200 原子 [object_eval.py:211-256];embed_objects_patch 逐物件 npz [patch_features.py:79-118])。
- shard 以**確定性 batch_index** 命名 → 重跑覆寫同檔、冪等,不會雙計/漏批。合併只讀 committed 批。

### 3.3 全域正規化只在合併時(critique high #3 + C8)

- shard 存**原始 anomaly_score / head_proba**,**不存** `_minmax` 後的 priority(`_minmax` 是全 array 的 min-max [active_learning.py:17-22,41])。
- 每次 Top-K 在「已處理併集」上算一次 priority(O(N),便宜)。
- **誠實聲明(寫進 GUI 文案 & AC)**:`novelty` 的 Top-K **成員**等於「原始分數的 Top-K」(_minmax 單調,不改序)→ 暫定即真;
  `uncertain/confusion` 混了全域正規化 novelty 與絕對 entropy/boundary,**併集變大時已處理項的相對序可能變動** →
  暫定 Top-K 對這兩個 objective 是**移動標的**,只有 `done=True` 才最終。回傳 `provisional` 旗標據此。

### 3.4 多樣性(diversity,critique high #3 / med #1)

- **暫定(done=False)且無 head**:`select_for_labeling(priority, k, cluster_labels=None)` = 純 priority(`diversity_applied=False`)。
  誠實——此時沒有全域分群可用,且分群是 N 依賴、label 不穩的全域 HDBSCAN [bootstrap_cluster.py:18-43],不可每批跑(C8)。
- **有 head(任何時刻)**:`cluster_labels = predict_head(head, obj_emb)[0]`(per-item 預測類別,批次安全、確定 [dino_head.py:75-88]),
  `select_for_labeling` 套每群上限 `max_per_cluster=2` → `diversity_applied=True`。(對映 app.py:1805 既有作法。)
- **M9 範圍決策**:bank-only(無 head)**不做** HDBSCAN 多樣性(需額外 object-embedding pass + 重 HDBSCAN,成本高),
  `diversity_applied=False` 並文件揭露。bank-only 全域 HDBSCAN 多樣性列 M9 Could / 候選,非阻擋。

## 4. 邊界條件與錯誤處理

| 情況 | 預期行為 |
|---|---|
| `checkpoint_dir` 落在來源資料夾內/含 images\|labels | `assert_safe_bank_dir` raise `ValueError`(C6)[anomaly_bank_store.py:156-181] |
| `cache_dir`/`checkpoint_dir` 為 None | `run_batched` raise `ValueError("checkpoint_dir 必填")`(不得關快取) |
| objective 需 head 但模型無 head.joblib | 開跑前 raise `ValueError`(§2.3),**不先 embedding** |
| object 模式但無 `projection.npz`/good_obj_emb | raise `ValueError`(novelty 在 object 模式需 ref)[critique med] |
| `object_source` 非法值 | raise `ValueError("object_source must be 'yolo' or 'whole_image'")` |
| `image_paths=[]` | `images_total=0`,`topk=[]`,`done=True`,不崩潰 |
| 某批 0 物件(yolo 無框) | 寫**空 shard**、仍 commit 該 batch_index(不再重掃)、`objects_scored` 不增、無 KeyError [critique med #4] |
| 含不可讀影像 | `partition_readable` 前置剔除 → 併入 `skipped`;索引對齊不受影響 |
| 續跑 `resume_identity` 不符(image_paths/object_source/class_names/batch_size/objective/model_version 任一變) | `on_identity_mismatch="error"` → raise;`="restart"` → 清 checkpoint 重跑。**batch_size 變一律拒絕**(shard 邊界移動)[critique high #4] |
| `state.json` 缺/毀(torn) | 視為全新開始(重掃),不採信半寫狀態 |
| crash 在 shard 寫後、state 更新前 | 該批未 committed → 續跑重跑該批(確定性,`.lv_cache` 不重算 embedding),冪等覆寫,無雙計 [critique high #6] |
| 只重寫 `manifest.json` | `model_version` **不變**(承重雜湊排除 manifest)→ 續跑照常 [critique low] |
| 重寫 `bank.npz`/`head.joblib` | `model_version` 變 → 續跑依 `on_identity_mismatch` 處置 |
| 兩前端不同 target_res 共用同 checkpoint_dir | cache 依 `cache_<model>_<res>/` 隔離,不撞;文件警告兩前端勿共用不同 model/res 的目錄 [critique med #3] |
| `k > objects_scored` | `topk` 長度 = objects_scored(`select_for_labeling` 夾 k [active_learning.py:63]) |

## 5. Acceptance Criteria(釘死;給 /pm。皆用**假 embedder/extractor + tmp 目錄**,不觸 DINOv2/真影像磁碟外的東西)

> 驗收層級:**單元 + 真實檔案系統整合測試**(Tier B 無 GUI → 不需 Playwright)。
> `model_dir` 由測試用既有 `anomaly_bank_store.save_bank`/`save_head` 造小凍結模型(patch bank 用少數合成向量)。
> 確定性:注入 `embed_fn`/`extractor`(回確定性向量,如影像雜湊→固定 embedding),`score_object`/`compute_outlier_scores` 為純函式。

**分批正確性 / 續跑**
- **AC1(分批 == 一次跑完,精確)**:同一組 N=40 影像(合成,novelty objective)、同一凍結 bank,
  `run_batched(paths, batch_size=7)` 的 `topk`(item_id 清單,含順序)**逐一等於** `batch_size=40`(單批)的 `topk`。
- **AC2(續跑不重算)**:`embed_fn` 內部計數呼叫次數。跑到第 2 批後以例外中斷;`resume=True` 重呼叫 →
  **已 committed 批的影像其 embed_fn 呼叫次數增量為 0**(全靠 shard/.lv_cache);最終 `topk` == 未中斷結果。
- **AC3(暫定 Top-K = 已處理併集的真 Top-K,novelty)**:用 `max_batches=1` 跑一批 → `progress` 收到的
  `provisional_topk` **等於**「僅該批已處理物件、依原始 anomaly_score 由大到小」的前 k;`done=False`、`provisional=True`。
- **AC4(Top-K 跨 batch_size 確定)**:構造多個**分數完全相同**的物件,`batch_size=3` 與 `batch_size=50` 兩次跑,
  最終 `topk`(item_id 清單)**完全相同**(證明 item_id canonical 排序消 tie-break,非陣列位置)。
- **AC5(續跑身分漂移拒絕)**:先跑一半;改 `batch_size`(或動 `bank.npz` 使 model_version 變)後 `resume=True` →
  `on_identity_mismatch="error"` 時 raise;`="restart"` 時 checkpoint 被清、從 0 重跑(不靜默合併)。

**契約 / 目標 / 身分**
- **AC6(objective 產物驗證,開跑前)**:`objective="uncertain"` 給**無 head** 的 model_dir → raise `ValueError`,
  且 `embed_fn` 呼叫次數 == 0(在任何 embedding 前就擋)。
- **AC7(item_id rescore 不變 + 不撞名)**:(a) 同一物件、兩個**不同** model_dir → **同一 item_id**;
  (b) 兩張**不同內容**但**同 stem**(不同資料夾)的影像 → **不同 item_id**(證明用 `_file_sha256` 非 stem)。
- **AC8(不寫來源資料夾,C6)**:對來源資料夾跑完,來源資料夾的檔案清單 + 各檔 mtime 快照**零變化**;
  所有產出都在 `checkpoint_dir` 下。
- **AC9(安全目錄)**:`checkpoint_dir` 指到來源資料夾(或其含 images/labels 的路徑)→ `assert_safe_bank_dir` raise。
- **AC10(空批 / 零物件不崩)**:yolo 模式、某批影像**無 label 框** → 該批寫空 shard、commit、`objects_scored` 不增、
  無 KeyError;整體 `done=True` 且對有框批的 `topk` 正確。
- **AC11(crash 冪等)**:monkeypatch 使「shard 寫入後、state 更新前」丟例外;續跑後該批**只被計一次**
  (item_id 去重),最終 `objects_scored` 與 `topk` == 未中斷結果。
- **AC12(model_version 排除 manifest)**:只改寫 `manifest.json`(改 saved_at)→ `load_frozen_model` 的
  `model_version` **不變**;改寫 `bank.npz` → `model_version` **改變**。
- **AC13(novelty 暫定即真;文件旗標)**:novelty objective 下,任一中間點的 `provisional_topk` 成員集合
  **等於**「已處理併集依原始分數的 Top-k 成員集合」(_minmax 單調不改成員);回傳 `provisional==(not done)`。
- **AC14(多樣性:有 head 套群上限)**:凍結模型**含 head**、造 head 對某類別過度自信使多物件同預測類 →
  最終 `topk` 中同一 head 預測類別的數量 **≤ `max_per_cluster`(=2)**,且 `diversity_applied==True`;
  無 head 時 `diversity_applied==False` 且 Top-K == 純 priority 前 k。

**回傳形狀**
- **AC15(回傳鍵齊全 + reason 非空)**:`run_batched` 回傳含 §2.4 所有鍵;`topk_records[i]` 含
  `item_id/image_path/bbox/obj_index/score/priority/reason`,且 `reason` 為非空中文字串。

## 6. 給 /pm 的提示

- 單元/整合測試 `tests/test_al_batch.py`;E2E 不適用(無 GUI,M9-GUI 接線另以 Playwright 驗,不在本模組)。
- **凍結模型 fixture**:用 `anomaly_bank_store.save_bank(model_dir, vectors=<少數合成 f32 向量>, meta={model,target_res,score_mode,...})`
  造 patch bank;`dino_head.save_head` 造小 head(≥2 類)驗 uncertain/AC14;object 模式用 `save_bank(vectors=None)+projection` 造 ref。
- **確定性**:全程注入 `embed_fn`(object)與 `extractor`(patch)回「影像內容→固定向量」;
  bank/ref 用同一映射的向量,使分數可預期、可構造 tie(AC4)、可構造過度自信 head(AC14)。
- **假影像**:造少量小 jpg(可用既有測試 helper);AC7(b) 需兩張同 stem、不同內容於不同子資料夾。
- **中斷模擬**:AC2/AC11 用 monkeypatch 在指定批/指定寫入點丟例外,再以 `resume=True` 重呼叫同參數。
- **契約鎖點**:`item_id` 內容組成(AC7)、分批==一次跑(AC1)、暫定=已處理真 Top-K(AC3/AC13)、
  身分漂移拒絕(AC5)是防 silent-wrong 的核心,務必保留。
- 開跑前放行 PG 記得 `python verify/gate.py --snapshot` 重建 baseline。
