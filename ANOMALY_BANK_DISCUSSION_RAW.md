# 瑕疵偵測 — Memory Bank 持久化 + 跨資料夾投影:多 Agent 三輪設計討論(完整原始記錄)

> 2026-06-27 由 multi-agent workflow 產出(2 摸清現況 + 4 提案 × 對抗批判 × 修正 + 1 綜合)。
> 狀態:**設計討論記錄,尚未動工**。精煉定案見 ANOMALY_BANK_DESIGN_NOTES.md。


---
# 一、摸清現況(讀碼事實)


## 〔Memory bank 機制:怎麼建、內含什麼、與模型/解析度/維度耦合、patch 特徵快取、能否序列化存硬碟(讀 anomaly_tool.py / anomaly_score.py / patch_features.py / object_eval.py / interaction.py / app.py)〕

**findings(事實)**
- 【memory bank 怎麼建】只在 score_mode=='patch' 且有 normal_set 時才建(anomaly_tool.py:109-118)。流程:embed_objects_patch 算出每物件的 patch 特徵 all_pf(每個 = {'feats':(P,384或768) L2正規化, 'grid':(gh,gw)})→ 取 normal_set 內物件 → np.vstack([all_pf[i]['feats'] for i in normal_set]) 把這些物件的所有 patch 疊成一個 (sum_P, D) 大矩陣 → 傳給 build_memory_bank。bank 只用『正常集』的 patch,不含全部物件。
- 【normal_set 怎麼來】anomaly_tool.py:89-93。優先序:若 confirmed 有標記 good(few-shot)→ normal_set = 那些 good 索引;否則 = 分群(cluster_objects)判為 normal_mask 且非 bad 的物件索引。若 normal_set 為空 → 根本不建 bank(bank=None),改走無監督 leave-one-out 物件級離群(compute_outlier_scores),不經過 patch memory bank。
- 【bank 內含什麼】MemoryBank(anomaly_score.py:21-44) 是不透明 handle,持有:self.vectors=np.ascontiguousarray(正規化後正常 patch 向量, float32, 形狀 (size, D));self.size;self._exact(size<=_EXACT_MAX=8192 用精確 matmul,否則 self._index=build_nn_index(vectors) 建 hnswlib cosine HNSW)。build_memory_bank(anomaly_score.py:47-57):_l2n(再次 L2 正規化)→ 若列數 > budget(預設 100000)用 seed=42 的 rng.choice 無放回隨機抽樣到 budget(這就是 coreset,純隨機子抽樣,不是 greedy coreset)→ MemoryBank。
- 【評分如何用 bank】score_object(anomaly_score.py:60-76):對單一物件的 query patch feats 做 _l2n → bank.nn_dist 算每個 query patch 到 bank 的最近鄰 cosine 距離 ∈[0,2](精確路徑:1 - (q@vectors.T).max;近似路徑:hnswlib knn_query k=1,set_ef(128))→ reshape 成 (gh,gw) patch 熱力圖 → object_score = 最高 ceil(q*P) 個 patch 距離之平均(q=0.01,至少 1 個)。run_pipeline 逐物件呼叫得 scores[i]。
- 【與模型耦合】bank 向量維度 D 完全由 DINOv2 變體決定:patch_features._DINOV2_DIM = {vits14:384, vitb14:768, vitl14:1024, vitg14:1536}(壞圖 fallback 與 model_dim 都用它)。embed_objects_patch 預設 model='dinov2_vits14'。換模型 → patch 特徵維度不同 → bank 維度不同 → 不可跨模型混用(向量 @ 不同維度會炸)。app.py 註解明說『換模型用各自獨立快取』。
- 【與解析度耦合】target_res 不改維度 D,但改每物件的 patch 數 P 與 grid(gh,gw):_real_extractor 回 (t.shape[2]//14, t.shape[3]//14)(patch_features.py:47)。224→16×16、336→24×24、448→32×32、560→40×40(app.py:1225-1226 _RES)。解析度越高 → 每物件 patch 越多 → bank 越大、越慢。bank 本身只存 patch 向量,不記 target_res;但用不同 res 算出的 query patch 與用某 res 建的 bank 比對,分佈/grid 會不一致(同物件不同 res 的 patch 特徵不可比)。
- 【與『分數依據』耦合】score_mode=='object'(物件級)走的是 obj_emb(整物件一條向量)+ compute_outlier_scores,完全不建 MemoryBank、不碰 patch_features。MemoryBank 只屬於 patch 路徑。
- 【patch 特徵快取機制】embed_objects_patch(patch_features.py:63-118):cache_dir 給定時逐物件一個 .npz,檔名 key=f'{image_stem}__{obj_index}'(patch_features.py:80)。存的是 feats=float16(壓縮)+ grid=int32(np.savez,patch_features.py:114-115)。讀回時 d['feats'].astype(float32)。快取命中直接跳過模型。cache_dir 由 app.py:1249 給:dataset_cache_dir(roots[0], f'anomaly_patch_{score_mode}_{model}_r{target_res}') → object_eval.dataset_cache_dir 算 .lv_cache/<foldername>_<sha1[:10]>/anomaly_patch_..._r... ;快取 key 已把 model 與 res 編進路徑,所以不同 model/res 自然分流不互污染。LV_CACHE_DIR 環境變數可覆寫根目錄。
- 【另一層快取:物件級 obj_emb】run_pipeline 另有 _obj_cache = cache_dir.parent / f'objemb_{model}.npz'(anomaly_tool.py:77,跨 score_mode/res 共用),走 object_eval.embed_objects 的增量 checkpoint npz(keys/emb/short_px,每 200 個 flush + atomic replace)。這是整物件向量(用於散點圖 SVD 投影與物件級評分),不是 patch、不是 bank。
- 【bank 目前不持久化、無序列化路徑】全 repo 沒有任何把 MemoryBank 存硬碟/讀回的程式碼。run_pipeline 把 bank 物件放進回傳 dict['bank'],app.py 把整個 result 塞進 session_state['anomaly_result'](記憶體,跨 rerun 在,跨 process 不在)。MemoryBank 沒有 to/from disk、沒有 __getstate__,但其狀態本質是可序列化的:self.vectors 是 float32 ndarray(可 np.save/npz)。唯一麻煩是大 bank 的 self._index(hnswlib.Index)不能直接被 numpy 存——但 hnswlib 本身有 save_index/load_index,且 index 可由 vectors 重建(__init__ size>8192 時自動 build_nn_index),所以序列化只需存 vectors + 重建 index 即可,不必存 hnswlib 物件。
- 【confirmed good/bad few-shot 也未持久化】st.session_state['anomaly_confirmed']={物件索引:'good'/'bad'}(app.py:1144 等)。這個索引是『當前 result 的 meta 順序索引』,綁定當次資料夾/掃描順序,跨資料夾無意義——若要存硬碟重用 few-shot,需改存穩定身份(image_stem+obj_index 或裁切後的 embedding/patch feats),不能存 int 索引。

**constraints(約束)**
- bank 向量維度 D 鎖死於 DINOv2 變體(vits14=384/vitb14=768/...),跨模型不可混;app.py 已強制不同 model 用不同快取路徑。
- 同物件用不同 target_res 算出的 patch 特徵不可互比(P、grid 不同);要『把新資料投影回舊 bank 分佈』必須用建 bank 時相同的 model + target_res 來算新資料的 patch/物件特徵。
- bank 的 self.vectors 已 L2 正規化(build_memory_bank 內 _l2n;MemoryBank.nn_dist 用 cosine)→ 存硬碟要連同『已正規化』語義一起記,讀回後 query 端也要 _l2n(現有 score_object 已做)。
- 散點圖投影是『對當前 result 的 obj_emb 即時 np.linalg.svd』(app.py:1301-1304),每次資料夾各自 fit,無跨資料夾共用的投影基底——要做需求(3)『新點映射回舊 good 分佈』必須改成:存舊資料的 SVD 基底(或舊 obj_emb 的 mean+Vt),對新 obj_emb 套用同一 (x-mean)@Vt[:2] 投影,不能再各自 svd。
- 絕不寫使用者資料集:所有快取一律進 .lv_cache(或 LV_CACHE_DIR / User 明確指定硬碟位置);bank/few-shot 持久化也須遵守。
- _EXACT_MAX=8192:bank patch 數 ≤8192 走精確 matmul、>8192 走 hnswlib 近似;『非常大量 good』會輕易超過(每物件數百 patch × 多物件),落到 hnswlib 路徑且最終受 budget=100000 上限隨機抽樣。

**reusable(可復用)**
- build_memory_bank(normal_patch_feats, budget, seed) → MemoryBank(scripts/anomaly_score.py:47):已是把任意 (M,D) patch 矩陣建成 bank 的純函式,持久化只需在外層存/讀那個 (M,D) 矩陣再 build_memory_bank 重建。
- MemoryBank.vectors 屬性(scripts/anomaly_score.py:25):序列化的唯一必要資料(float32, 已正規化),np.savez 即可存。
- embed_objects_patch(meta, model, target_res, cache_dir, ...) (scripts/patch_features.py:63):已內建逐物件 .npz float16 快取 + 壞圖補零保索引;可直接拿來對『新資料夾』算 patch 特徵再丟進舊 bank 評分,且快取自動分流 model/res。
- object_eval.dataset_cache_dir(folder, kind) (scripts/object_eval.py:44):.lv_cache 雜湊路徑產生器(LV_CACHE_DIR 可覆寫),app 與 CLI 共用——任何新的 bank/few-shot 持久化檔都應走它取得不污染使用者資料夾的落點。
- object_eval.embed_objects(...) 的增量 checkpoint + atomic replace npz 模式(keys/emb/short_px,每 200 flush,tmp.replace)(scripts/object_eval.py:188-262):是現成的『大量資料安全增量寫 npz』範式,可照抄做 bank 持久化。
- object_eval 的 profile store 範式:dataset_fingerprint / save_profile / match_profile / load_profiles + object_policy_profiles.json(scripts/object_eval.py:116-184):現成的『把每資料集設定/指紋存 JSON 並用指紋找最近舊設定』模式,可作為『存舊 bank 元資料 + 用指紋把新資料夾配到舊 bank』的直接模板。
- interaction.build_nn_index(emb) (scripts/interaction.py:42):cosine HNSW 建構;MemoryBank 大 bank 與一般 NN 查詢共用,讀回 vectors 後重建 index 即用此函式。

**gaps(缺口)**
- 完全沒有 bank 序列化/反序列化的程式碼或檔案格式約定——需求(1)(2)的『存硬碟 / 讀回 bank』要從零設計(建議:存 vectors npz + model/target_res/budget/維度 等 metadata,讀回後 build_memory_bank 重建;大 bank 不必存 hnswlib 物件)。
- confirmed good/bad 用『當次 result 的整數索引』存 session,跨資料夾/重掃即失效——要把 few-shot 存硬碟重用,需重新設計穩定身份(image_stem+obj_index 或直接存其 patch/obj 特徵),這層映射目前不存在。
- 散點圖投影目前每資料夾各自 np.linalg.svd,沒有『把新 obj_emb 投影進舊基底』的程式碼——需求(3)的核心(新點落在舊 good 密集區判正常)需要新增『存舊投影基底(mean+Vt 或舊 obj_emb)+ 對新資料套同一變換』的邏輯。
- 沒有讀回 bank 後對『新資料夾』走評分的入口:run_pipeline 永遠當場從 normal_set 重建 bank,無法傳入既有 bank;要支援需求(2)需新增一條『bank 由外部提供、跳過建 bank 直接 score_object』的路徑(目前 run_pipeline 簽章與分支不支援)。
- budget=100000 的 coreset 是純隨機抽樣(seed=42),沒有 representativeness 保證;『非常大量 good』時 bank 內容受隨機抽樣影響,且兩次建 bank 只要輸入 patch 集相同 seed 相同才可重現——若需『穩定可存可比的 bank』要確認/固定這層隨機性語義。

## 〔瑕疵偵測(第8工具)— 散點投影/memory bank 持久化/few-shot good-bad 標記/資料夾選取與執行流程的事實盤點〕

**findings(事實)**
- 散點圖投影(SVD)的真實做法【app.py:1298-1306】:對「當前 result 的 obj_emb」即時做 PCA-via-SVD。先 `c = emb - emb.mean(axis=0)` 中心化,再 `_u,_s,_vt = np.linalg.svd(c, full_matrices=False)`,座標 `coords = _u[:,:2]*_s[:2]`(等於取前兩主成分的投影分數)。LinAlgError 時退回 `c[:,:2]`(原始前兩維)。**每次執行偵測都重新 fit,完全不快取、不存盤、不跨資料夾**。`_vt`(主成分方向/載荷矩陣)被算出但**從未被保存或重用**——這是把『新資料 transform 進既有空間』缺的關鍵物件。
- 目前『沒有』任何把新資料投影進既有空間的基礎:`_anomaly_ui` 的 SVD 是 self-contained per-result,沒有 fit/transform 分離。`result` dict 不含 `_vt`/`mean`/任何投影 basis。session_state 也沒存。要做 User 需求(2)(把新資料夾投影進既有 memory bank 分佈)目前**零基礎**,需新增:把舊資料的 `mean` 與 `_vt[:2]` 存盤 → 新資料 `(emb_new - mean_old) @ _vt[:2].T`。
- 既有『跨資料集投影到同一座標系』的最近 prior art 在『嵌入覆蓋圖/覆蓋』工具,而**非**瑕疵偵測:`_cov_projection`【app.py:4600-4632】把 `dataset_emb` 與 `cand_emb` 用 `np.vstack` 合併後一起 fit(PCA/UMAP/TSNE),兩者落在同一座標系。注意它是『合併重新 fit』(comment 明說『不用近似 transform』),**不是** transform 新點進舊 basis。語義上接近 User 要的『新點映射回舊分佈』,但實作是 re-fit-together,且這套是覆蓋圖工具不是瑕疵偵測。`interaction.reference_coverage(emb_a, emb_b, radius)` 與 `rank_gap_fillers(cand, dataset)` 提供『以外部參照 B 量 A 覆蓋/缺口』的純函式,可複用於『新資料相對舊 good 分佈的密度/離群』。
- few-shot good/bad 標記的『產生』:全靠散點圖框選 → 三個按鈕。`_anomaly_mark(records, indices, verdict)`【app.py:1140-1148】把全域物件 index 寫進 `st.session_state['anomaly_confirmed'][i] = 'good'|'bad'`。另有 `_anomaly_autoseed_normal(n=12)`【1151-1159】把 ranking 末端(最不可疑)n 個自動標 good。`_anomaly_clear_confirmed`【1162】清空。**僅存 session_state,未持久化到硬碟**(User 已知),換資料夾/重啟即失;index 是全域物件索引,只在同資料集同一次 discover 順序下穩定。
- few-shot good/bad 標記的『使用』全在 `run_pipeline`【anomaly_tool.py:85-123】:`confirmed` 過濾出 `good_idx`/`bad_idx`。正常集(memory bank 來源)優先序:**有確認 good → 只用這些 good few-shot 當 normal_set**;否則用分群 `normal_mask` 且排除 bad。三條評分路徑:(a) normal_set 空 → 無監督 leave-one-out kNN 離群(`compute_outlier_scores(obj_emb,obj_emb)`);(b) score_mode=='object' → 物件級 emb 對 normal_set kNN;(c) patch(預設)→ `embed_objects_patch` 取 patch 特徵 → `build_memory_bank(np.vstack(normal_set 的 patch feats))` → 逐物件 `score_object`。bad 標記只進 `classify`【anomaly_classify.py】:good+bad 都有→Youden's J 門檻+算 AUROC;只有 bad→門檻設在最低 bad 分數下;無確認→`(1-contamination)` 百分位。
- MemoryBank 完全『無持久化』:`anomaly_score.py` grep `save|load|persist|npz|pickle` 零命中。`MemoryBank.__init__(vectors)` 只持有正規化向量+(>8192 時)即時建 hnswlib index。每次 `run_pipeline` 都 `build_memory_bank(...)` 從頭重建(PRD【2_PO_PRD:16】明寫『重跑只重建 memory bank,不重算 DINOv2 特徵』——意即 bank 本身不存)。`bank` 物件只塞進 `result['bank']` 活在 session_state,供右欄 heatmap 重算用;**從不落盤,無法跨 session/跨資料夾載入**。
- **唯一**落盤的快取是 DINOv2 特徵,不是 bank:`run_pipeline(cache_dir=...)` 經 `embed_objects_patch`【patch_features.py:63-118】逐物件存 `.npz`(float16 patch feats + grid);`_object_embeddings` 另存 `objemb_{model}.npz`【anomaly_tool.py:74-80】。路徑由 `dataset_cache_dir(folder, kind)`【object_eval.py:44-50】=`<LV_CACHE_DIR 或 repo/.lv_cache>/<foldername>_<sha1(abspath)[:10]>/<kind>`,kind=`anomaly_patch_{score_mode}_{model}_r{res}`【app.py:1249】。**綁單一資料夾絕對路徑雜湊,換資料夾就換目錄,天然無跨資料夾共用機制**。守 no-dataset-writes:一律寫 .lv_cache。
- GUI 資料夾選取與執行流程:側欄 `_folder_picker_list('anomaly_folder')`【app.py:201-217】= 📁原生對話框 + 文字加入 + ×移除的『多資料夾清單』(可加多個,程式對每個 `list_images`+取首個 `classes_for`)。設定 widget:DINOv2 模型、流程(two_stage/one_stage)、分數依據(patch/object)、解析度(224/336/448/560)、2-stage 抽樣數。按『▶ 執行偵測』(`anomaly_run`)或主畫面『🔁 重新偵測』(`_anomaly_rerun` 旗標)都走同一 run 區塊【1239-1265】:`cache=dataset_cache_dir(roots[0], ...)` → `run_pipeline(...)` → 把 `_image_paths/_class_names/_cache/_model/_target_res` 塞回 result 存 `anomaly_result` → 收合側欄。**沒有任何『存 bank 到硬碟』『從硬碟讀 bank』『選既有 bank 對新資料夾投影』的 widget 或程式路徑**。
- 右欄看圖/heatmap【app.py:1431-1492】:分數 slider(`anomaly_heat_filter`)+類別 multiselect(`anomaly_class_filter`)與左欄散點連動(變淡不移除)。選一物件→若 `result['bank']` 存在(patch 模式且有 normal_set)就對該物件重抽 patch 特徵 `score_object(...,result['bank'])` 取 pmap 畫 `render_heatmap`;否則只顯示 crop。**heatmap 依賴 in-memory `result['bank']`,bank 一旦來自『未持久化的本次執行』,跨 session 不存在**。

**constraints(約束)**
- 鐵則:絕不寫使用者資料集;所有快取一律進 `.lv_cache`(或 LV_CACHE_DIR / User 明確指定的硬碟位置)。要新增『存 bank 到硬碟』必須讓 User 明確指定路徑或進 .lv_cache,不可寫回資料夾(memory [[no-dataset-writes]])。
- U-Net 規格疊加紀律:實作只進 `scripts/`、測試只進 `tests/`、設計只出 Markdown(`3_Architect_Design/`)+AC。新功能(bank 持久化、新資料投影進舊空間)應走 /user→/po→/architect→/pm→/pg,GUI/IO/外部編解碼模組屬 Tier B,done=單元綠 AND 真實 E2E 綠(`tests/e2e/test_<module>_e2e.py` 標 @pytest.mark.e2e)。
- PG 嚴禁改 tests/conftest.py/契約檔/基礎設施;機器判綠跑 `python verify/gate.py <module>`。MemoryBank 是 Tier A 純邏輯(anomaly_score),但加 save/load IO 後會自動升 Tier B。
- 現有 SVD 投影是『每 result 獨立 fit』,座標僅供畫圖(comment 與 _cov_projection 一致:『稀疏度永遠不從這些座標讀』)。若要做『新點映射回舊分佈』,必須保存舊的 `mean` + `_vt[:2]` 主成分 basis 才能 transform,不能沿用現有 self-contained SVD。
- few-shot 標記用『全域物件 index』定位,只在同資料集同一次 `discover_yolo_objects` 順序下穩定;跨資料夾或檔案集變動即失效。要持久化 few-shot 必須改用『內容定址(image_path+bbox+obj_index 或 sha256)』而非裸 index(對齊既有 `match_shas_to_indices`/object-export-crop-bug 教訓)。

**reusable(可復用)**
- `scripts/anomaly_score.py`:`MemoryBank(vectors)` 持有 L2 正規化向量,只差 save/load(np.savez vectors → 重建即可,因 hnswlib index 是 lazy 重建);`build_memory_bank(normal_patch_feats, budget=100000)` 已含 coreset 下採樣——存 bank = 存它的 `self.vectors`。
- `scripts/object_eval.py:dataset_cache_dir(folder, kind)` + `app.py:_dataset_cache_dir`(同雜湊規則)= 既有 .lv_cache 路徑慣例;`patch_features.embed_objects_patch(cache_dir=...)` 已示範逐物件 .npz 落盤模式,可照抄做 bank 落盤。
- `scripts/interaction.py`:`reference_coverage(emb_a, emb_b, radius)`(以 B 當外部參照量 A 覆蓋/recall)、`rank_gap_fillers(cand, dataset)`(新資料相對舊資料的離群/缺口分數)、`compute_outlier_scores(candidates, reference, k)`——三者正是『新資料 vs 舊 good 分佈密度/離群』的現成純函式,可直接服務 User 需求(3)。
- `app.py:_cov_projection`【4600-4632】=『把舊資料+新候選 vstack 後 fit 到同一座標系』的現成範式(含 session 快取),User 需求(2)的『同一座標系顯示新舊』可複用此模式(惟它是 re-fit-together 非 transform-into-fixed-basis)。
- `interaction.match_shas_to_indices(shas, sha_to_index)`、`snapshots_to_csv`、`curation_log_csv`:既有『內容定址 + CSV 持久化 + 回載對齊 index』樣板,適合做 few-shot good/bad 的硬碟持久化(避免裸 index 失效)。
- `scripts/anomaly_tool.run_pipeline(confirmed=..., cache_dir=...)` 已是邏輯總入口且無 streamlit;新增『載入既有 bank』可在此加一個 `external_bank` 參數注入,不必改 GUI 結構。`embed_fn`/`extractor` 注入點已存在(測試友好)。

**gaps(缺口)**
- 完全沒有 memory bank 持久化:無 save/load API、無存盤格式、無『硬碟既有 bank』的選取 widget。User 需求(1)(建 bank→存硬碟)與(2)(從硬碟讀 bank)目前 0 實作。
- 完全沒有 fit/transform 分離:現有 SVD 不保存 basis,無法把新資料夾 transform 進既有(舊資料)空間。User 需求(2)(3)『新點映射回舊分佈』缺核心機制——需新增保存 `mean+_vt` 並對新 emb 投影。
- few-shot good/bad 標記未持久化(只在 session_state,且用裸全域 index),無法隨 bank 一起存硬碟、跨 session/資料夾沿用。User 需求(1)『硬碟也要能存標記的 good/bad few-shot』缺實作,且需先解決 index→內容定址。
- 沒有『跨資料夾』的執行流程:`dataset_cache_dir` 以單資料夾絕對路徑雜湊綁定,run 流程一次只服務當前 folders;沒有『舊資料夾建/存 bank、新資料夾載入該 bank 並對照』的雙資料夾 session 概念。
- MemoryBank.vectors 是『coreset 下採樣後的正常 patch 特徵』,不含每物件 obj_emb;但散點投影用的是 `obj_emb`(物件級),兩者不同層級。User 要的『新資料投影回舊 good 密集區』需釐清是用 obj_emb 分佈(畫散點用)還是 patch bank(評分用)——目前散點與評分用不同表徵,跨資料夾投影要先決定對齊哪一個。
- 未驗證:本盤點為靜態讀碼,未實跑 app 或 E2E 確認 SVD/bank 行為;`result['bank']` 在無 normal_set / object / LOO 模式下為 None(anomaly_tool 只在 patch 路徑建 bank),跨模式的 bank 可用性需實測。


---
# 二、Round1 提案(4 面向)


## 提案:資料/序列化(memory bank coreset + few-shot good/bad 標記的硬碟持久化、版本相容鍵、過期偵測、穩定身份)

# 設計:Memory Bank 與 Few-shot 標記的硬碟持久化(資料/序列化面向)

目標:支援 User 三需求 ——(1)第一次建 bank 並可存硬碟(含 few-shot good/bad)、(2)第二次從硬碟讀回 bank 對「新資料夾」評分、(3)新資料映射回舊 good 分佈。本設計只負責「資料怎麼存、怎麼讀、怎麼驗相容、怎麼判過期、few-shot 怎麼用穩定身份」。投影/評分入口/GUI 的細節由其他面向設計者補,本文僅定義它們依賴的**契約資料結構**。

核心原則(與既有碼一致,務必遵守):
- **絕不寫使用者資料集**:預設落點走 `object_eval.dataset_cache_dir(folder, kind)`(`.lv_cache`,`LV_CACHE_DIR` 可覆寫);User 要存到別的硬碟位置時,必須由 User 明確指定一個目錄路徑,程式只寫該目錄,絕不回寫資料夾。
- **復用既有純函式**:`build_memory_bank(vectors)` 已是「(M,D) → MemoryBank」純函式,且 hnswlib index 是 lazy 重建 → **存 bank = 存 `MemoryBank.vectors`(已 L2 正規化,float32)**,讀回後呼叫 `MemoryBank(vectors)` 即重建,**不存 hnswlib 物件**。
- **新增 IO 不改既有契約檔**:`run_pipeline` 既有簽章不動,以「新增可選參數」方式注入(`external_bank=` / 新 module 提供 save/load)。

---

## 資料模型

一個「存盤的 bank」是一個**目錄**(稱 `bank_dir`),內含三個檔(分檔而非單一大檔,理由:vectors 可能上百 MB 要 mmap/壓縮,metadata 要人可讀可 diff,few-shot 要可單獨增量更新)。

```
<bank_dir>/                       # User 指定路徑 或 dataset_cache_dir(src_folder, "anomaly_bank")
├── bank.npz                      # coreset 向量(大檔,二進位)
├── bank_meta.json               # 版本相容鍵 + 來源指紋 + 投影 basis 參照(人可讀)
└── fewshot.json                 # few-shot good/bad 標記(內容定址,人可讀可手改)
```

### 1) `bank.npz`(coreset 向量 + 投影 basis)
用 `np.savez_compressed`(對齊既有 `embed_objects` 的 npz 慣例;coreset 已被 budget 上限到 ≤100k×D,壓縮後可控)。
```python
np.savez_compressed(bank_path,
    vectors = bank.vectors.astype(np.float16),   # (M, D) coreset,已 L2 正規化;存 fp16 省一半(讀回 .astype(fp32) 再 _l2n)
    proj_mean = mean.astype(np.float32),         # (D_obj,) 舊 obj_emb 的均值(投影中心化用)
    proj_vt   = vt2.astype(np.float32),          # (2, D_obj) 舊 obj_emb SVD 前兩主成分(投影 basis)
    good_obj_emb = good_emb.astype(np.float16),  # (G, D_obj) 舊 good few-shot 的物件級向量(畫舊分佈/算密度用,可空)
)
```
**為何 bank 向量(patch 級)與投影 basis(obj 級)同檔**:它們語義上同屬「這個 bank 快照」,且必須一起版本綁定(同 model/res 才可比)。但維度不同 —— `vectors` 是 patch 級 `D`(vits14=384…),`proj_mean/proj_vt/good_obj_emb` 是 obj 級 `D_obj`(物件整體一條向量,維度同 patch D 但語義不同層級)。`bank_meta.json` 明記兩者,讀回端不可混用。
> **缺口補上**:現有散點圖每 result 即時 `np.linalg.svd`、`_vt` 算了從不存。本設計在「建 bank 時順手把 `mean` 與 `_vt[:2]` 存進 bank.npz」,這就是需求(3)「新點映射回舊 basis」缺的關鍵物件。fp16 對 basis 方向足夠(投影只取前 2 維畫圖,非精確距離)。

### 2) `bank_meta.json`(版本相容鍵 + 過期偵測)
人可讀、可 diff、可手審。**這是相容性的單一真相**。
```json
{
  "schema_version": 1,
  "model": "dinov2_vits14",
  "target_res": 224,
  "score_mode": "patch",
  "patch_dim": 384,
  "obj_dim": 384,
  "patch_grid": [16, 16],
  "budget": 100000,
  "coreset_seed": 42,
  "coreset_size": 98231,
  "l2_normalized": true,
  "source": {
    "folder_name": "good_train",
    "folder_sha1": "a1b2c3d4e5",
    "fingerprint": { "n_objects": 12044, "n_classes": 1, "size_q": [...], "ar_q": [...] }
  },
  "fewshot": { "n_good": 18, "n_bad": 5 },
  "created": "2026-06-27T10:00:00",
  "tool_version": "anomaly@<git_short_sha 或 常數>"
}
```
**相容性硬鍵(讀回時必須完全相等,否則拒絕)**:`schema_version`、`model`、`patch_dim`、`target_res`、`patch_grid`、`l2_normalized`、`score_mode`。理由(都已在現況確認):
- `model` 不同 → patch 維度 D 不同 → `vectors @ q.T` 直接炸 → **硬擋**。
- `target_res` 不同 → 每物件 P、grid 不同,同物件不同 res 的 patch 特徵**不可互比** → **硬擋**(即使 D 相同)。
- `patch_dim` / `patch_grid` 當作 `model`/`res` 的冗餘自校驗(double-check,避免 metadata 與向量漂移)。
- `l2_normalized=true` 記錄語義:`vectors` 已正規化,query 端 `score_object` 已 `_l2n`,讀回後**不可二次破壞**。

**相容性軟鍵(僅警告,不擋)**:`coreset_seed`、`budget`、`fingerprint`(來源指紋)、`tool_version`。

**過期偵測(stale)**:用既有 `object_eval.dataset_fingerprint` + `fingerprint_similarity`。
- 對「現在新選的資料夾」算 fingerprint,與 `meta.source.fingerprint` 比;`fingerprint_similarity < 0.6`(經驗門檻,可調)→ GUI 顯示「⚠ 新資料分佈與舊 bank 差異大,映射結果僅供參考」。**不擋**(需求(2)本來就是拿舊 bank 看新資料,差異大正是要看的訊號)。
- 若 `bank_dir` 是 `.lv_cache` 內、且 source folder 仍在原路徑 → 額外比 `folder_sha1` 與最新檔案 mtime,提示「來源資料夾已變動,bank 可能過期,可重建」。

### 3) `fewshot.json`(穩定身份 — 解決裸 index 失效)
**問題(現況已確認)**:few-shot 現存 `session_state["anomaly_confirmed"][global_int_index]`,只在「同資料集同一次 `discover_yolo_objects` 順序」下穩定,跨資料夾/重掃即失效 → 不能直接序列化裸 index。
**解法**:改用**內容定址**。穩定身份鍵 = patch 快取已在用的 `f"{Path(image_path).stem}__{obj_index}"`(見 `patch_features.py:80`),再帶 `bbox` + `class_id` 做碰撞自檢。
```json
{
  "schema_version": 1,
  "items": [
    { "key": "IMG_0007__2", "verdict": "good",
      "image_path": "D:/ds/good/images/IMG_0007.jpg",
      "image_stem": "IMG_0007", "obj_index": 2,
      "bbox": [0.51, 0.33, 0.12, 0.20], "class_id": 0, "label": "screw" },
    { "key": "IMG_0042__0", "verdict": "bad", ... }
  ]
}
```
- **存(序列化)**:把 session 的 `anomaly_confirmed{int_index: 'good'|'bad'}` 透過**當次 result 的 `records[i]`**(已含 `image_path`/`obj_index`/`bbox`/`label`)反查出穩定身份再寫檔。`records` 已備齊欄位(`anomaly_tool.py:128-134`),零額外計算。
- **讀回(反序列化 → 對齊 index)**:對新資料夾 `discover_yolo_objects` 後,建 `key → new_index` 映射(`f"{stem}__{obj_index}"`),用 `fewshot.json` 的 key 配對。配上 → 警告若 `bbox`/`class_id` 不符(同名檔但內容變了 → skip 該標記並提示);配不上(該物件不在新資料夾)→ 預設略過(few-shot good/bad 的語義是「舊資料的人工標記」,本就主要服務建 bank 那次;跨資料夾沿用是 bonus,不該硬塞)。
- 復用既有 `interaction.match_shas_to_indices` 的「內容定址 → 回載對齊 index」樣板精神(對齊 object-export-crop-bug 教訓:**永遠別存裸 index**)。

> **few-shot 為何也存 good 的 obj_emb 進 bank.npz?** 需求(1)說「硬碟也要能存標記的 good/bad」。`fewshot.json` 存「身份 + verdict」(輕、人可讀、可手改);`bank.npz/good_obj_emb` 另存舊 good 的**特徵向量**,使需求(3)「畫舊 good 密集區、新點落點對照」即使**舊資料夾已不在線上**也能畫(否則得重新載入舊資料夾跑 DINOv2)。兩者互補:json 是「決策」、npz 是「特徵快取」。

---

## UX流程(資料層視角 — 只描述存/讀觸發點與資料流,不設計像素)

**情境 A:第一次(建 bank + 存硬碟)**
1. User 選舊資料夾 → 設定 model/res/score_mode=patch → 在散點圖框選標 good/bad(現有 `_anomaly_mark`)→ 執行偵測。
2. `run_pipeline` 照舊建出 `result{bank, obj_emb, normal_set, records, ...}`。
3. 新增「💾 存 memory bank」按鈕 → 呼叫新 module `bank_store.save_bank(bank_dir, result, confirmed, meta_kv)`:
   - `bank.vectors` → `bank.npz:vectors`(fp16);
   - 即時對 `obj_emb` 算 `mean`/`svd`,存 `proj_mean`/`proj_vt`;`obj_emb[good_idx]` → `good_obj_emb`;
   - 從 `result['_model'/'_target_res'/'score_mode']` + patch grid 寫 `bank_meta.json`;
   - 從 `confirmed` + `records` 反查穩定身份寫 `fewshot.json`。
   - `bank_dir` 預設 `dataset_cache_dir(src_folder, "anomaly_bank")`;User 可在 widget 指定其他硬碟目錄(只寫該目錄)。
4. **DINOv2 patch 特徵不重算**(已快取在 `.lv_cache/..._anomaly_patch_..._r...`),存 bank 只是序列化已有向量 → 秒級。

**情境 B:第二次(讀回 bank → 對新資料夾評分 + 投影)**
1. User 選新資料夾 + 新增「📂 載入 memory bank」widget(選 `bank_dir`)。
2. `bank_store.load_bank(bank_dir)` → 回 `LoadedBank{bank: MemoryBank, meta: dict, proj_mean, proj_vt, good_obj_emb, fewshot: list}`:
   - 讀 `bank.npz:vectors` → `MemoryBank(vectors)`(內部依 size 自動選 exact/hnswlib)。
   - 讀 `bank_meta.json`,**先做相容性硬鍵檢查**:用新資料夾當前選的 model/res 與 meta 比;不符 → **拒載 + 明確錯誤**(「此 bank 用 dinov2_vitb14@336 建,當前設定 vits14@224,維度/grid 不相容」)→ 引導 User 切回相容設定或重建。
   - 算新資料夾 fingerprint 比 `meta.source.fingerprint` → stale 警告。
3. 評分:走「external_bank 注入」路徑(見下「投影」與 U-Net 模組):對新資料夾 `embed_objects_patch`(自動命中/新建 `.lv_cache` 快取,**用 meta 鎖定的 model/res**)→ 逐物件 `score_object(..., loaded.bank)`。**不重建 bank**。
4. 投影:對新 `obj_emb` 套 `(emb_new - proj_mean) @ proj_vt.T` → 與 `good_obj_emb`(舊 good)同座標系畫散點。
5. few-shot:`fewshot.json` 經內容定址對齊到新資料夾 index(配不上的略過),回填 session 供 GUI 顯示「這些是舊標記」。

---

## 投影(資料層契約 — fit/transform 分離)

需求(3)的核心:**新點映射回舊 good 分佈**。現況是「每 result 各自 `np.linalg.svd`,basis 從不存」→ 零基礎。本設計把投影 basis 變成 **bank 的一部分資料**:
- **fit(建 bank 時做一次)**:`c = obj_emb - obj_emb.mean(0)`;`_,_,vt = np.linalg.svd(c, full_matrices=False)`;存 `proj_mean = obj_emb.mean(0)`、`proj_vt = vt[:2]`(對齊 `app.py:1298-1306` 的既有算法,只是改成「存下來」)。
- **transform(讀回對新資料做)**:`coords_new = (obj_emb_new - proj_mean) @ proj_vt.T` → `(N_new, 2)`。**不再對新資料各自 svd**,確保新舊同座標系。
- 舊 good 的座標:`coords_good = (good_obj_emb - proj_mean) @ proj_vt.T`(從 bank.npz 直接得,不需舊資料夾在線)。
- **與 `_cov_projection` 的差異**:既有覆蓋圖工具是「新舊 vstack 後 re-fit-together」;本設計刻意採 **transform-into-fixed-basis**(舊 basis 固定),才符合「映射**回**舊分佈」語義(re-fit 會讓舊點也跟著動,失去「舊分佈是固定參照」的意義)。
- **密度/離群量化(可選增強)**:除了畫散點,可復用 `interaction.reference_coverage(emb_new, good_obj_emb, radius)` / `rank_gap_fillers(emb_new, good_obj_emb)` 給「新點落在舊 good 密集區 vs 離群」的數值分數(純函式,現成)。
- **LinAlgError 後備**:`proj_vt` 退回 `eye(2, D_obj)`(取前兩維),與既有 `c[:,:2]` 後備等價,確保 transform 不炸。
- **DINOv2 變體 → 維度查表復用**:`patch_features` 已有 `_DINOV2_DIM`/`model_dim(model)`,序列化端的 `patch_dim`/`obj_dim` 一律從它取,單一真相,避免硬編碼。

---

## 風險

| 風險 | 說明 | 緩解 |
|------|------|------|
| **跨模型維度炸裂** | 用 vits14 建的 bank 拿 vitb14 query → `q@vectors.T` 形狀不符直接 crash | 讀回**硬鍵檢查** `model`/`patch_dim`,不符即拒載 + 明確訊息(絕不讓它跑到 matmul) |
| **跨解析度語義錯但不報錯** | D 相同(同 model)、res 不同時 matmul **不會炸**,但 patch 數/grid/分佈不可比 → 分數靜默錯 | 硬鍵也含 `target_res`/`patch_grid`,res 不符即拒載(這是最危險的 silent-wrong,優先擋) |
| **fp16 存向量的數值損失** | vectors 存 fp16 省空間,讀回 `.astype(fp32)` 後 cosine 距離有微小漂移 | 讀回後**重跑 `_l2n`** 復正規化;cosine 對 fp16 容忍度高(AnomalyDINO 距離本就粗粒度);如要保守可提供 `dtype=fp32` 選項旗標寫進 meta |
| **few-shot 內容定址碰撞** | 不同資料夾有同名 `IMG_0007.jpg` → key `IMG_0007__2` 撞 | 帶 `bbox`+`class_id` 自檢,不符即 skip + 警告;key 只當「候選」非「保證」 |
| **coreset 隨機性不可重現** | budget 隨機抽樣(seed=42),若兩次輸入 patch 集不同則 bank 內容不同 | 存的是**已抽樣後的 vectors 本體**(非重抽),讀回 100% 可重現;`coreset_seed`/`coreset_size` 記進 meta 供追溯 |
| **bank.npz 過大** | 100k×384 fp16 ≈ 75 MB;多 bank 累積佔硬碟 | `np.savez_compressed` + fp16;`.lv_cache` 落點受既有快取清理涵蓋;User 指定路徑時由 User 自管 |
| **寫到使用者資料夾** | User 指定路徑時手滑指到資料集 | `save_bank` 預設走 `dataset_cache_dir`;User 自訂路徑時**不做任何 dataset 偵測寫入**,只寫所給目錄;文件明示鐵則 |
| **bank_dir 半寫損毀** | 寫到一半中斷 → 殘檔 | 照抄 `embed_objects` 的 **atomic replace**(寫 `*.tmp` 再 `os.replace`);`load_bank` 對缺檔/壞檔 `try/except` 回明確「bank 損毀請重建」(不靜默吞錯成空 bank) |
| **proj_vt 與 vectors 維度層級混淆** | patch D 與 obj D 雖數值相同但語義不同,誤用 | meta 分記 `patch_dim`/`obj_dim`;`good_obj_emb`/`proj_*` 僅供投影、`vectors` 僅供評分,API 分開回傳不混 |
| **stale 誤擋** | 需求(2)本就是拿舊 bank 看「不一樣」的新資料,若把 fingerprint 差異當錯誤會擋掉正常用例 | fingerprint **只警告不擋**;唯一硬擋的是 model/res/dim/grid 不相容(那是真會出錯) |

---

## U-Net模組

新增**一個 Tier B 模組**(有 IO,自動升 B;done = 單元綠 AND 真實 E2E 綠),設計只出 Markdown + AC,實作進 `scripts/`、測試進 `tests/`。

**模組名:`anomaly_bank_store`**(`scripts/anomaly_bank_store.py` / `tests/test_anomaly_bank_store.py` / `3_Architect_Design/0X_anomaly_bank_store.md` / `4_PM_Feedback/anomaly_bank_store.md`)

純資料/序列化 API(無 streamlit,測試友好):
```python
def save_bank(bank_dir, *, vectors, proj_mean, proj_vt, good_obj_emb,
              meta: dict, fewshot: list[dict]) -> Path        # atomic 寫三檔
def load_bank(bank_dir) -> LoadedBank                          # 讀回 + 結構檢查(不含相容性比對)
def check_compat(meta_on_disk, *, model, target_res, patch_dim, patch_grid) -> CompatResult
                                                              # 回 ok/拒載原因(硬鍵)
def check_stale(meta_on_disk, new_fingerprint) -> float        # 回 similarity(GUI 決定是否警告)
def confirmed_to_fewshot(confirmed, records) -> list[dict]     # session int-index → 穩定身份
def fewshot_to_confirmed(fewshot, new_meta) -> dict            # 穩定身份 → 新資料夾 int-index(內容定址對齊)
```

**對既有碼的最小侵入(由其他面向設計者承接,本文只定契約)**:
- `anomaly_score.MemoryBank`:**不改**(`vectors` 屬性已是唯一序列化必要資料;lazy hnswlib 自動重建)。若 PO 要把 save/load 掛在 MemoryBank 上,則 `anomaly_score` 由 Tier A 升 Tier B —— **建議反而獨立成 `anomaly_bank_store` 模組**,讓 `anomaly_score` 維持純邏輯 Tier A。
- `anomaly_tool.run_pipeline`:新增可選參數 `external_bank: MemoryBank | None = None` —— 給定時**跳過 normal_set 建 bank**,直接用它走 `score_object`(需求(2)入口)。簽章只「加可選參數」,不破壞既有呼叫。
- `scripts/app.py`:加「存 bank / 載入 bank / 指定路徑」widget,並把投影改用 `bank_store` 回傳的 `proj_mean/proj_vt`(GUI 面向設計者負責)。

**AC 重點(資料層,供 /pm 寫先紅測試)**:
- AC1:`save_bank` → `load_bank` round-trip,`vectors` cosine 距離與原 bank 一致(容忍 fp16 誤差 < 1e-3)。
- AC2:`check_compat` 對 model 不符 / res 不符 / dim 不符 / grid 不符**各自**回拒載(四條負例)。
- AC3:`confirmed_to_fewshot` → `fewshot_to_confirmed` round-trip:同資料夾 index 完全還原;改檔名順序後仍靠內容定址正確對齊。
- AC4:`fewshot_to_confirmed` 遇 `bbox`/`class_id` 不符的同名 key → skip 該項並可被觀測(不靜默吞)。
- AC5:`load_bank` 遇缺檔/壞 npz → 拋明確例外(不回空 bank)。
- AC6:投影 transform — 對「建 bank 時的同一批 obj_emb」套 `(x-mean)@vt.T` 必須等於建 bank 當下的 `coords`(fit/transform 自洽)。
- AC7(E2E,標 `@pytest.mark.e2e`):存 bank → 切新資料夾載入 → heatmap/散點真的用舊 basis 畫出新點落點(行為斷言,非 element 存在)。

**Reverse gate 注意**:本設計假設 `result` 已含 `obj_emb`(投影 basis 來源)且 patch 模式才有 `bank`。若 PO 要支援 `score_mode=='object'`(無 patch bank)也能存「投影 + few-shot」,需回 /po 釐清「object 模式存的是 obj_emb-only bank」這個變體 —— 屬需求澄清,不在本資料層自行擴張。


## 提案:UX/流程

這份設計只負責 **UX/流程面**:在 `_anomaly_ui` 怎麼把「第一次建 bank →(可選)存硬碟」與「第二次讀 bank → 對新資料夾投影」兩條路徑,以最小侵入、與既有「執行偵測」流程相容的方式呈現出來。投影/評分演算法細節由架構面同伴負責,本文只定義 UX 觸點與必要的資料契約。

---

## 資料模型

UX 只要求落盤產物能支撐「狀態提示」與「投影回舊分佈」。所有檔案一律進 `.lv_cache`(走 `object_eval.dataset_cache_dir`),或 User 在 widget 明確指定的硬碟資料夾;**絕不寫回資料集**。

### 1. Bank 套件(一個資料夾 = 一個可攜 bank,鏡像既有 profile store 範式)

```
<bank_dir>/                       # User 指定路徑 或 .lv_cache/<folder>_<sha>/banks/<name>/
  manifest.json                   # 給 UX 顯示與相容性檢查(人可讀)
  vectors.npz                     # bank patch coreset: vectors=(M,D) float16 + grid 慣例
  projection.npz                  # 投影基底: mean=(D,) + components=(2,D)  (= 舊 obj_emb 的 _vt[:2])
  fewshot.json                    # few-shot 標記(內容定址,非裸 index)
```

`manifest.json`(UX 用來組「狀態膠囊」與擋掉不相容讀入,欄位刻意對齊 `save_profile`):

```json
{
  "name": "line-A-2026Q2",
  "created": "2026-06-27T10:00:00",
  "model": "dinov2_vits14",            // ← 相容性硬鎖:跨模型不可讀
  "target_res": 224,                   // ← 相容性硬鎖:跨解析度 patch 不可比
  "dim": 384,
  "score_mode": "patch",
  "budget": 100000,
  "n_normal_objects": 1240,            // 建 bank 用了幾個正常物件
  "n_patches": 100000,                 // coreset 後 bank 大小(顯示給 User 看規模)
  "n_good": 18, "n_bad": 3,            // few-shot 數量
  "source_fingerprint": { ... },       // object_eval.dataset_fingerprint(舊資料夾)
  "source_folders": ["D:/data/line-A"] // 僅顯示用,不依賴它存在
}
```

> 復用點:`vectors.npz` 只存 `MemoryBank.vectors`(已 L2 正規化,float16 壓縮即可,讀回 `build_memory_bank` 重建,hnswlib index lazy 重建)。`projection.npz` 存的 `mean`+`components` 就是現有 SVD 算出但被丟掉的 `c.mean(axis=0)` 與 `_vt[:2]`。

### 2. few-shot 內容定址(`fewshot.json`)

現況 `anomaly_confirmed` 是 `{裸全域 index: 'good'|'bad'}`,跨資料夾/重掃即失效。落盤時轉成**穩定身份**(對齊 object-export-crop-bug 教訓):

```json
{ "marks": [
  {"image_stem": "img_0007", "obj_index": 2, "bbox": [..], "verdict": "good"},
  {"image_stem": "img_0102", "obj_index": 0, "bbox": [..], "verdict": "bad"}
]}
```

讀回時用 `(image_stem, obj_index)` 對齊到當前 result 的 records(records 已帶 `image_path`/`obj_index`/`bbox`),映射回裸 index 後塞進 `anomaly_confirmed` —— 對下游 `run_pipeline(confirmed=...)` **完全透明**,不需改評分邏輯。對齊不到的(新資料夾沒這張圖)就略過並在膠囊提示「few-shot 對齊 12/18」。

### 3. Session 狀態(新增 key,全部可從上述檔案重建,不持久化於 session)

| key | 內容 | 用途 |
|-----|------|------|
| `anomaly_loaded_bank` | `{dir, manifest, vectors, projection}` 或 None | 表示「目前掛載了一個外部 bank」=進入第二條路徑 |
| `anomaly_bank_save_dir` | 上次存檔路徑(記憶,免重打) | 存檔 widget 預設值 |

> 既有 `anomaly_result` / `anomaly_confirmed` 不改語義;`anomaly_loaded_bank` 是**唯一**新的模式開關。

---

## UX流程

核心決策:**不新增第二顆主按鈕、不切 tab**。沿用唯一的「▶ 執行偵測」當動作收斂點,bank 的「建/存/讀」用一個**側欄 expander**承載,並用「目前是否掛載外部 bank」這個狀態,讓同一顆「執行偵測」表現出兩種行為。理由:大量 good/少數 bad 的 User 心智是「我就是要跑偵測」,bank 是加速與對照的**修飾語**,不是平行功能 —— 把它做成 mode 切換會增加誤操作面。

### 側欄版面(插在現有「解析度 / 2-stage 抽樣數」與「▶ 執行偵測」之間)

```
### 🔧 瑕疵偵測設定
  [📁 資料夾清單 …]   [模型] [流程] [分數依據] [解析度] [抽樣數]   ← 既有,不動

  ▼ 💾 Memory Bank（建一次、重複用）           ← 新增 expander，預設收合
    狀態膠囊：
      ◦ 未掛載 → :gray[未掛載 bank。執行偵測會「當場用此資料夾建 bank」（現有行為）。]
      ◦ 已掛載 → ✅ 已掛載「line-A-2026Q2」· vits14@224 · 100k patch · few-shot 18好/3壞
                  [✕ 卸載]
    分隔線
    第一次（建好 bank 後存硬碟）：
      [💾 存目前 bank 到硬碟…]   ← 僅在 anomaly_result 有 bank 時 enabled
        └ 展開：名稱輸入 + 路徑（📁 picker，預設 .lv_cache/…/banks/<name>，可改硬碟）+ [確認存檔]
    第二次（讀既有 bank 對照新資料）：
      [📂 讀入硬碟上的 bank…]
        └ 展開：📁 picker 選 bank 資料夾 → 顯示 manifest 摘要 + 相容性檢查 → [掛載]

  [▶ 執行偵測]   ← 唯一主按鈕；文案隨狀態變
```

### 路徑 A —— 第一次:建 bank →(可選)存硬碟

1. User 照舊選資料夾、跑 **2-stage**:執行偵測 → 框選正常 →「✅ 框選標為正常範例」(或自動種子)→ 再執行偵測。此時 `result['bank']` 已是乾淨 few-shot bank(現況已支援,**零改動**)。
2. expander 內「💾 存目前 bank 到硬碟」由 disabled 變 enabled(條件:`result.get('bank') is not None`)。點開填名稱、選路徑,**預設落點 = `.lv_cache/<folder>_<sha>/banks/<name>/`**(安全預設),User 想存到外接硬碟可用 📁 改路徑。
3. 確認存檔 → 寫 4 個檔(vectors / projection / fewshot / manifest)→ `st.toast("已存 bank：line-A · 100k patch · 18 好/3 壞")`。projection 基底在存檔當下從當前 result 的 `obj_emb` 算 `mean`+`_vt[:2]`(就是現在散點圖那段 SVD,只是把結果存下來)。

> 大量 good 體感:User 在「非常多良品」資料夾框選一團正常即可;不需要逐張標。存檔把這份「正常分佈 + few-shot」凍結成可攜資產,下次不必重標、不必重算 DINOv2(patch 特徵仍走既有 `.lv_cache` 快取)。

### 路徑 B —— 第二次:讀 bank → 對新資料夾投影

1. 換新資料夾(folder 清單換成新資料)。展開「📂 讀入硬碟上的 bank」→ 📁 選 bank 資料夾。
2. 立即讀 `manifest.json` 顯示摘要,並做**相容性檢查**(純 UX gate,擋在掛載前):
   - `manifest.model == 側欄選的 model` 且 `manifest.target_res == 側欄 target_res` → ✅ 可掛載。
   - 不符 → 不給掛載,顯示明確錯誤:「此 bank 為 vits14@224,當前設定為 vitb14@336。維度/patch 不可比 —— 請把模型/解析度改回 vits14@224 再掛載」並提供 [一鍵同步設定] 把側欄 model/res 改成 manifest 的值。
3. 點 [掛載] → 寫入 `anomaly_loaded_bank`,膠囊變綠;few-shot marks 立即用內容定址映射到新資料夾的 records(在下一次執行偵測時生效),提示「few-shot 對齊 12/18(6 個在新資料夾找不到對應物件)」。
4. 按「▶ 執行偵測」—— 因為 `anomaly_loaded_bank` 存在,走**外部 bank 路徑**:對新資料夾算 patch 特徵(走 `embed_objects_patch`,快取自動分流到新資料夾的 `.lv_cache`),但**不重建 bank**,直接用掛載的 bank `score_object`。同時把新 `obj_emb` 用掛載的 `projection`(mean+components)投影 → 散點圖呈現「新點疊在舊分佈上」。

### 「執行偵測」文案隨狀態變(降低模式混淆)

| 狀態 | 按鈕文案 | 行為 |
|------|----------|------|
| 未掛載 bank | `▶ 執行偵測` | 現況:當場建 bank(或無監督) |
| 已掛載 bank | `▶ 對照已掛載 bank 偵測` | 用外部 bank 評分 + 投影回舊分佈 |

並在主畫面結果列加一行狀態:`對照 bank「line-A」· 新資料 320 物件投影至舊 good 分佈`。

### 右欄 heatmap 連動

右欄 heatmap 已依賴 `result['bank']`。外部 bank 路徑要確保 `result['bank']` 填的是**掛載的 bank**(由架構面在 `run_pipeline` 回傳),這樣 heatmap「點一物件→`score_object(...,bank)`→疊熱力圖」**完全沿用既有碼**,無需 UX 改動。

---

## 投影

這是需求(3)「新點映射回舊分佈」的 UX 落點,**不改散點圖元件本身,只換座標來源**:

- **未掛載 bank(路徑 A 與現況)**:座標仍是「對當前 result 的 obj_emb 即時 SVD」(app.py:1298-1306,不動)。
- **已掛載 bank(路徑 B)**:座標改為 `coords = (emb_new - manifest.mean) @ manifest.components.T`(用舊基底 transform,不再各自 svd)。散點圖再**疊一層底圖**:把舊 bank 的 good 分佈(存 projection 時順手存舊 obj_emb 的 2D 座標,或存少量代表點)畫成淺灰半透明背景點,新資料點實色上色 —— User 一眼看到「新點落在舊 good 密集區=正常 / 落在離群處=可疑」。
- 散點圖 caption 隨之變:`座標 = 投影至「line-A」舊 good 分佈(灰=舊正常,彩=新資料異常分數)`。

> UX 契約:架構面需在 result 提供 `coords2d`(已是現有但恆 None 的欄位 —— 正好拿來裝投影座標)與 `_bank_bg_coords`(舊分佈底圖點)。本設計只要求「掛載時座標來源切到 projection 基底 + 疊舊分佈底圖」,演算法由架構面定。

---

## 風險

1. **模式隱形切換造成誤判**:User 掛了 bank 卻沒注意,以為還是「當場建 bank」。緩解:① 膠囊綠色高可見;② 主按鈕文案變「對照已掛載 bank 偵測」;③ 結果列明寫「對照 bank「X」」。三處冗餘提示。
2. **相容性陷阱(最大風險)**:bank 是 vits14@224,User 側欄改成 vitb14 → 維度不合直接炸。緩解:**掛載前硬 gate**(model+res 必須相符才給掛載),並提供「一鍵同步設定」。掛載後若 User 又改了側欄 model/res → 偵測前再檢一次,不符就紅字擋住執行,不讓它跑到崩。
3. **few-shot 對齊掉點**:新資料夾不含舊 few-shot 那些圖 → 對齊不到。這是正常的(本來就是不同資料),用「對齊 12/18」如實顯示,不視為錯誤;但要避免「對齊 0/18」靜默 —— 0 時提示「few-shot 全數無法對齊新資料夾,bank 的正常分佈仍可用,但 few-shot 門檻校準將失效」。
4. **存檔路徑寫到資料集**:User 用 📁 選錯把 bank 存進資料集資料夾。緩解:預設落點 `.lv_cache`(安全),且若偵測到目標路徑 == 任一已選 folder 或其子目錄 → 紅字擋下「不可存入資料集資料夾」(對齊 no-dataset-writes)。
5. **「非常大量 good」存檔規模**:coreset budget=100000 patch × float16 ≈ 數十 MB,可接受;但 projection 底圖若存全部舊 obj_emb 座標會膨脹。緩解:底圖最多存 N 個代表點(如隨機 2000)或只存 mean+components,底圖點由「存檔時的舊資料 2D 座標」抽樣存。
6. **掛載 bank 後 normal_set 語義衝突**:外部 bank 路徑下不該再從新資料夾的 confirmed/分群重建 normal_set。UX 契約:掛載時「正常範例」UI 應切為唯讀提示「正常參考來自掛載的 bank」,避免 User 在新資料夾又框選正常造成兩套正常集打架(實際分支由架構面在 `run_pipeline` 處理)。
7. **expander 預設收合導致發現性低**:第一次用的 User 可能找不到「存 bank」。緩解:當 `result['bank']` 首次出現且未存過,在主畫面結果列加一次性 `st.info("可把這份正常參考存成 bank 重複用 → 側欄『💾 Memory Bank』")`。

---

## U-Net模組

本 UX 設計觸及的功能屬 **Tier B**(GUI + IO + 外部編解碼),done = 單元綠 AND 真實 E2E 綠。建議模組切分(供 PO 拆 module,本文不寫實作):

| module | 歸屬 | UX 相關 AC 種子(待 PM 轉測試) |
|--------|------|-------------------------------|
| `anomaly_bank_store`(新,Tier B) | bank 序列化/反序列化 + manifest + few-shot 內容定址讀寫 | 存→讀 round-trip 後 `MemoryBank.vectors` 數值一致;manifest 欄位齊;few-shot 映射回 records 正確;絕不寫入資料集路徑(寫到 .lv_cache/指定硬碟) |
| `anomaly_tool`(既有,擴充) | `run_pipeline` 加 `external_bank` 注入點(line 109-123 分支) + projection 基底產出/回傳 `coords2d` | 傳 external_bank 時不重建 bank、用它評分;掛載時 `coords2d` = 投影至舊基底;normal_set 衝突由此處理 |
| `app._anomaly_ui`(既有 GUI,擴充) | 側欄 expander、狀態膠囊、存/讀 widget、按鈕文案隨狀態、散點圖座標來源切換、相容性 gate | **E2E**:第一次建+存 bank → 換資料夾讀入 → 執行 → 散點圖座標來自舊基底(新點疊舊分佈)、heatmap 仍可用、膠囊顯示已掛載、相容性不符時擋住執行 |

紀律提醒:本文只是 UX/流程**設計輸入**,需走 /user→/po→/architect→/pm→/pg。`run_pipeline` 的 `external_bank` 與 projection 演算法、`MemoryBank` save/load 的精確格式屬架構/PG 層,本設計只鎖定「UX 觸點 + 必要資料契約(manifest 欄位、few-shot 內容定址、coords2d 來源切換)」。PG 嚴禁為過綠改 tests/契約。

關鍵相容點(與既有碼直接接合,降低落地風險):
- 唯一新主路徑開關 = `anomaly_loaded_bank` session key;未掛載時**所有現有行為完全不變**。
- few-shot 讀回後映射成裸 index 塞 `anomaly_confirmed` → 下游 `run_pipeline(confirmed=...)` 透明。
- bank 存讀檔案格式鏡像 `object_eval` 的 profile store(`save_profile`/`load_profiles`/`dataset_fingerprint`)+ `embed_objects_patch` 的 npz 落盤範式,不另起爐灶。
- 投影座標塞進現有恆 None 的 `result['coords2d']`,heatmap 沿用 `result['bank']`,兩者皆既有欄位。

## 提案:投影/視覺化:跨資料夾 fit/transform 投影器 + 「新資料映射回舊分佈」視覺

本設計只負責「投影/視覺化」面向(跨資料夾把新資料投影回舊資料分佈,並讓 User 直覺看出新點落在 good 密集區還是離群)。**刻意不重複設計** bank 序列化格式 / few-shot 內容定址 / run_pipeline 注入既有 bank 等(那是「持久化/評分」面向的職責);本設計只**消費**他們存的 metadata 並**新增**一個獨立、可 persist 的投影器層。兩面向的銜接點明確標於下方。

---

## 資料模型

核心新增物件是 **`Projector`** —— 一個「以舊資料 fit、可對新資料 transform、可存可讀」的固定投影基底。它與 bank **分開存**(投影用 `obj_emb` 物件級向量,bank 用 patch 向量,兩者層級不同,硬綁會互相污染),但**共用同一份硬碟 metadata** 以保證 model/target_res 對齊。

### 1. `ProjectorBasis`(純資料,可序列化)— `scripts/anomaly_project.py`(新模組)
線性(PCA-via-SVD)投影器的全部狀態,刻意只存「能 transform 新點」的最小集合:

```
ProjectorBasis = {
  "schema": "lv-anomaly-projector/1",
  "method": "pca",            # v1 只支援線性可 transform 的 PCA;umap/tsne 見「投影」段
  "model": "dinov2_vits14",   # ← 必須與建 bank 的 model 一致(維度鎖死)
  "target_res": 224,          # ← 投影本身不依賴 res,但寫入做相容性檢查
  "dim_in": 384,              # 原始 obj_emb 維度 D(= DINOv2 變體維度)
  "dim_out": 2,               # 投影後維度(散點固定 2)
  "l2norm": true,             # fit 前是否 L2 正規化(與既有 _l2norm 幾何一致)
  "mean":  float32[D],        # 中心化向量(fit 時的舊資料均值)
  "components": float32[dim_out, D],  # = _vt[:dim_out](主成分方向/載荷)
  # ↓ 視覺化用的「舊分佈摘要」,讓新點能立刻有灰底密度可比,毋須重存全部舊 emb
  "ref_coords": float32[N_ref, 2],    # 舊資料(coreset 後)投影後座標 → 灰底散點
  "ref_score":  float32[N_ref] | null,# 舊資料異常分數(可選,用於灰底著色/門檻)
  "ref_good_mask": bool[N_ref] | null,# 舊資料中哪些是 confirmed good(畫密度核心)
  "fingerprint": {...},       # object_eval.dataset_fingerprint(舊資料)→ 自動配對提示
  "created": "2026-06-27T..",
  "source_folder": "....",    # 僅供 UI 顯示「這個投影器來自哪個舊資料夾」
}
```

落盤:`mean/components/ref_coords/ref_score/ref_good_mask` 進一個 `projector.npz`(`np.savez`,沿用 `object_eval.embed_objects` 的 atomic `tmp.replace` 範式);其餘 scalar/dict 進**同目錄** `projector.json`。兩者同放於 bank 所在目錄(由「持久化」面向決定的硬碟落點;若該面向用 `dataset_cache_dir(old_folder, "anomaly_bank_<model>_r<res>")`,投影器就放同一 dir,天然與 bank 同進退、同 model/res 分流)。

> **與「持久化」面向的銜接契約(必須對齊,否則 reverse-gate 回 PO)**:`projector.json.model` / `target_res` / `dim_in` 必須等於 bank metadata 的同名欄位。讀回時若不符 → UI 報錯「投影器與 bank 的模型/解析度不符,無法套用」,不靜默繼續(維度不符會炸,且分佈不可比)。

### 2. `ProjectionResult`(in-memory,塞進 `result`)
新資料 transform 後,在 session 內帶著:
```
result["_projector"] = ProjectorBasis        # 載入的舊投影器(None=同舊行為,自己 fit)
result["_proj_new"]  = float32[N_new, 2]      # 新 obj_emb 投影後座標
result["_proj_nn"]   = { "ref_idx": int[N_new],   # 每個新點最近的舊 ref 點 index
                          "dist":    float32[N_new], # 到最近舊 ref 的距離(原始 D 維 cosine,非 2D)
                          "in_dense": bool[N_new] }  # 是否落在舊 good 密集區(見「投影」段定義)
```
注意 `_proj_nn.dist` 是在**原始 D 維空間**算的(投影後 2D 距離會失真),只有畫圖座標用 2D。

### 3. session_state 鍵(新增,全部前綴 `anomaly_proj_` 避免撞既有鍵)
- `anomaly_proj_path`:User 選的投影器目錄(None=不啟用跨資料夾,走原本 per-result SVD)。
- `anomaly_proj_loaded`:已載入的 `ProjectorBasis`(避免每次 rerun 重讀 npz)。
- `anomaly_proj_show_ref` / `anomaly_proj_show_links`:灰底密度、最近鄰連線開關。

---

## UX流程

分兩個情境,**完全不破壞現有單資料夾流程**(不選投影器 = 行為與今日 100% 相同)。

### 情境 A:第一次(建並存投影器)
1. User 照舊選舊資料夾 → 設定 model/res → ▶ 執行偵測(既有流程不變)。
2. 偵測完成後,**左欄散點圖下方**新增一顆按鈕 **「💾 存成投影器(供之後比對新資料)」**。
   - 按下 → 用「持久化」面向決定的硬碟落點,呼叫 `anomaly_project.fit_projector(result, …)` 把 `mean/components/ref_coords/…` 寫成 `projector.npz`+`projector.json`。
   - `ref_good_mask` 直接讀 `session_state["anomaly_confirmed"]`(哪些是 good);`ref_score` 讀 `result["scores"]`。
   - 成功 toast:「已存投影器:<source_folder> · 384 維 · model=vits14 · res=224」。
3. (bank 是否同時存盤,由「持久化」面向的 widget 處理;投影器存盤是**獨立**動作,User 可只存投影器、之後另算 bank,反之亦然——解耦避免一個壞掉拖累另一個。)

### 情境 B:第二次(載入投影器、把新資料投影回舊分佈)
1. **側欄**在既有資料夾清單下方新增一個 **`_folder_picker_single('anomaly_proj_path')`**(原生對話框選「投影器目錄」)。選定後即時讀 `projector.json` 顯示一行摘要:「來源:<old folder> · N_ref=… · model/res=…」。
   - 若 `dataset_fingerprint(新資料夾)` 與 `projector.json.fingerprint` 相似度低 → 顯示**橘色提示**「新舊資料分佈差異大,投影對照僅供參考」(沿用 `object_eval.fingerprint_similarity`,不擋執行)。
2. User 選**新**資料夾 → ▶ 執行偵測。run 區塊偵測到 `anomaly_proj_path` 已設 → 偵測算完新 `obj_emb` 後,額外:
   - **相容性檢查**:新 `obj_emb` 維度 == `dim_in`?model/res match?不符 → `st.error` 並**不**畫跨資料夾圖(退回 per-result SVD,功能不全失)。
   - `transform_new(projector, new_obj_emb)` → 填 `result["_proj_new"]`、`result["_proj_nn"]`。
3. **左欄散點圖切到「跨資料夾對照」呈現**(下段「投影」詳述視覺):灰底舊分佈 + 高亮新點 + 最近鄰連線。圖上方一行 caption:「新點落在舊正常密集區 = 正常;落在離群處 = 可疑。新資料共 N,其中 落在密集區 X、離群 Y。」
4. 既有的「分數/類別篩選變淡」「框選→標 good/bad/購物車」**照常作用在新點上**(customdata 仍是新資料全域 index),User 可框選離群新點直接加購物車。
5. 右欄 heatmap:**不變**(依賴 in-memory bank,由「持久化」面向決定 bank 來源;投影器不碰右欄)。

### 一個關鍵 UX 決策:新點上色語義切換
- **單資料夾模式**:新點延用 Turbo 異常分數上色(同今日)。
- **跨資料夾模式**:新點改用**二元語義**——落在舊 good 密集區=綠、離群=紅、中間=橘(由 `in_dense` + `dist` 分桶)。舊 ref 點固定灰底半透明當「地圖」。這樣 User 一眼看「新的紅點 = 從舊分佈看出來的可疑」,不必再讀 colorbar 數值。兩種模式互斥、由「是否載入投影器」自動切,**不**增加 User 要懂的設定。

---

## 投影

### Fit(舊資料,情境 A)— `anomaly_project.fit_projector(emb, *, l2norm=True, dim_out=2)`
**完全沿用 app.py:1298-1306 既有 SVD 數學**,只是把丟掉的 `_vt`/`mean` 留下來:
```
emb = obj_emb (N, D)
if l2norm: emb = _l2norm(emb)          # 與既有 _cov 投影 cosine 幾何一致
mean = emb.mean(0, keepdims=True)
c = emb - mean
_u, _s, _vt = np.linalg.svd(c, full_matrices=False)
components = _vt[:dim_out]              # (dim_out, D) ← 今日被算出但丟棄的關鍵物件
ref_coords = (_u[:,:dim_out] * _s[:dim_out])   # 舊資料座標,與今日散點完全相同
```
- 為控制 `projector.npz` 大小,`ref_coords/ref_score/ref_good_mask` 在 N_ref 很大時做**確定性下採樣**(seed=42 的 `rng.choice`,保留全部 good + 抽樣 normal,上限如 5000 點),純為畫灰底;`components/mean` 不抽樣(transform 需全量)。

### Transform(新資料,情境 B)— `anomaly_project.transform_new(basis, emb_new)`
```
if basis.l2norm: emb_new = _l2norm(emb_new)
coords_new = (emb_new - basis.mean) @ basis.components.T   # (N_new, dim_out)
```
這是**真正的 transform-into-fixed-basis**(與 `_cov_projection` 的「re-fit-together」不同):新點被投到**舊資料定義的同一座標軸**,所以「新點 (x,y) 落在舊 good 群附近」才有意義。re-fit-together 會讓新資料反過來扭動座標軸,破壞「映射回舊分佈」的語義,故此處**不**用 `_cov_projection`。

### 「落在 good 密集區 vs 離群」的判定(視覺標示的數值依據)
不靠 2D 座標(會失真),用**原始 D 維**最近鄰,復用 `interaction.compute_outlier_scores`:
```
# reference = 舊 good 的 obj_emb(若投影器存了 ref 的原始向量;見下「需存什麼」)
dist = compute_outlier_scores(emb_new, ref_good_emb, k=min(5, len(ref_good)))
in_dense = dist <= tau          # tau = 舊 good 自身 kNN 距離的高分位(如 95%),fit 時一併存
```
- **需多存一份**:要在 D 維算最近鄰,投影器得存 `ref_good_emb`(舊 good 的原始 obj_emb,float16 壓縮)而非只有 2D 座標。這是 `ProjectorBasis` 的可選欄位 `ref_good_emb: float16[N_good, D]`;沒有 confirmed good 時退而用全部 ref(無監督密度)。`tau` 同步存於 json。
- 退路:若 User 不想存原始向量(隱私/體積),`in_dense` 改用 **2D 座標**的最近鄰近似(`interaction.reference_coverage(coords_new, ref_coords, radius)`),標示為「近似」。預設存 D 維(準),json `dense_metric: "Dspace"|"2D"` 記哪種。

### umap/tsne 的處理(誠實聲明)
v1 **只支援 PCA**,因為只有線性投影能 persist basis 並對新點做**真 transform**。umap/tsne 沒有穩定 out-of-sample transform(`_cov_projection` 也是靠 re-fit-together 迴避這點),硬做近似 transform 會誤導「映射回舊分佈」。若 User 要 umap 視覺,UI 提供**第二個按鈕「umap 合併重投影(舊+新一起,座標會變動)」**,直接複用既有 `_cov_projection(ref_emb, new_emb, "umap", 2, …)`——明確標註「這是合併重擬合,不是把新點投進固定舊空間」,語義誠實分開,不混為一談。

### 視覺(plotly,改造既有左欄 `go.Figure`,不另開圖區)
跨資料夾模式時,左欄圖**疊三層 trace**(由底到頂):
1. **舊 ref 灰底密度**:`Scattergl(ref_coords, marker grey rgba(150,150,150,0.18), size 5)`;good 點稍深(rgba 0.3)當「正常核心」。開關 `anomaly_proj_show_ref`。可選 `Histogram2dContour` 畫密度等高線當底圖(舊 good 越密的地方等高線越密),讓「密集區」一眼可見。
2. **新點**:`Scattergl(coords_new, marker color = 綠/橘/紅 by in_dense+dist, size 8, line width 1 white)`;customdata=新資料全域 index(維持框選/標記相容)。hover 顯示「檔名 · D維距最近舊 good=… · 判定」。
3. **最近鄰連線**(開關 `anomaly_proj_show_links`,預設**離群點才連**,避免線太多):對每個 `in_dense=False` 的新點,畫一條淡線到它最近的舊 ref 點(`_proj_nn.ref_idx`),讓 User 看「這個可疑新點最像舊資料裡的哪一個」。用單一 `Scattergl(mode="lines")` 把所有線段(NaN 分隔)一次畫完,效能 OK。

色階圖例改成離散三色(正常/邊緣/離群)取代 Turbo colorbar;hover 仍給原始 D 維距離數字,兼顾直覺與可量化。

---

## 風險

1. **層級錯配(最大風險)**:散點投影用 `obj_emb`(物件級),評分 bank 用 patch 向量。User 直覺「新點落在 good 密集區」是看 `obj_emb` 投影,但「可疑」的權威來源是 patch bank 分數——**兩者可能不一致**(obj_emb 看正常、patch 看出細微瑕疵,反之亦然)。緩解:UI 明說「散點=物件整體相似度地圖,細微瑕疵以右欄 heatmap/分數為準」,且新點的紅/綠**同時參考 `_proj_nn.dist`(obj 級)與 `result["scores"]`(若該新資料也跑了 patch bank)**,兩者衝突時上色取較保守(較可疑)者並在 hover 標注。**不**讓散點顏色假裝是最終判定。
2. **線性 PCA 只解釋前 2 主成分**:obj_emb 在 2D 投影可能把離群點壓到 good 群里(假陰)。緩解:`in_dense` 用 **D 維** kNN 判定(不信任 2D 座標),2D 只負責「畫在哪」;hover 給 D 維真實距離。風險殘留:極端 case 點畫得近但實際遠 → 連線+hover 數字補救。
3. **跨資料夾分佈漂移**:新資料夾與舊資料若本質不同(不同產線/光照),投到舊基底整團偏移,所有新點都「離群」→ 失去鑑別力。緩解:fingerprint 相似度低時橘字預警;提供「umap 合併重投影」作為 sanity-check 第二視角。
4. **體積/隱私**:存 `ref_good_emb`(D 維原始向量)讓投影器檔變大、且把舊資料特徵帶出資料夾。緩解:float16 + 下採樣;提供「只存 2D 近似」開關;一律寫 `.lv_cache` 或 User 指定路徑,**絕不**寫回資料夾(守 no-dataset-writes)。
5. **無 obj_emb 的模式**:object 模式有 obj_emb,但 LOO/無 normal_set 路徑仍有 obj_emb(散點本來就用它),故投影器在所有模式都可 fit;唯 patch bank 可能為 None(右欄 heatmap 不可用),投影器**獨立於 bank**,不受影響。已在資料模型解耦。
6. **session 一致性**:投影器載入後若 User 中途改 model/res 再跑,維度會不符 → 相容性檢查擋下並退回 per-result SVD,不靜默炸。
7. **E2E 真實行為**(Tier B 要求):必須斷言「故意放一個已知離群的新物件 → 它在跨資料夾圖上被標紅/落在離群桶、且 `in_dense=False`」,以及「新點座標 = `(emb-mean)@components.T`(transform 正確、與 fit 的舊點同基底)」。element 存在不算。

---

## U-Net模組

依規格疊加紀律,本面向落為**一個新純邏輯模組 + 一段 GUI 整合**,與「持久化」面向共享 metadata 契約:

| module | Tier | 產出 | 內容 | done 條件 |
|---|---|---|---|---|
| `anomaly_project` | 先 A(純函式),加 npz/json IO 後**自動升 B** | `scripts/anomaly_project.py` | `fit_projector(emb, good_mask, score, *, l2norm, dim_out)` → `ProjectorBasis`;`transform_new(basis, emb_new)` → coords;`classify_new(basis, emb_new)` → `_proj_nn`(用 `compute_outlier_scores`);`save_projector(basis, dir)` / `load_projector(dir)`(npz+json,atomic replace);`check_compat(basis, model, res, dim)` | `python verify/gate.py anomaly_project` 印 GREEN(單元:fit→save→load 往返一致、transform 對新點等於手算 `(x-mean)@Vt.T`、已知離群點 `in_dense=False`、維度不符 raise) |
| `anomaly_ui`(既有,擴充) | B | 改 `scripts/app.py` `_anomaly_ui` + 側欄 picker + run 區塊 | 加投影器 picker、存/載按鈕、跨資料夾三層散點 trace、二元上色、相容性檢查與退路 | 單元綠 **AND** `tests/e2e/test_anomaly_project_e2e.py`(@e2e):情境 A 存投影器→情境 B 載入+新資料夾→斷言「已知離群新物件真的被標紅/排到離群、連線指向最近舊點」 |

**設計階段只出本 Markdown 規格 + 上述 AC**;`anomaly_project.py` / app.py 改動進 PG 階段。

**跨面向契約(/architect 需與「持久化」面向對齊,寫進各自 `3_Architect_Design/*.md` 的「共享契約」段)**:
- C1:投影器與 bank 共用 `{model, target_res, dim_in}` metadata,值必須一致;讀回時 `check_compat` 強制。
- C2:投影器存盤落點 = bank 落點同目錄(`dataset_cache_dir(old_folder, kind)` 或 User 指定路徑),由「持久化」面向統一決定,投影器只接收 `dir` 參數,不自己決定路徑策略。
- C3:few-shot good/bad 的**內容定址身份**(image_stem+obj_index 或 sha)由「持久化」面向設計;投影器只在 fit 時消費「當次 result 的 `anomaly_confirmed` 整數 index → good_mask」,**不**負責跨 session 重對齊 few-shot(那是另一面向的事)。若該面向尚未就緒,投影器 fit 仍可運作(good_mask 缺 → 退無監督全量 ref 密度)。
- C4(reverse-gate 觸發點):若「持久化」面向決定的 bank metadata 不含 `dim_in`/`model`/`target_res`,投影器無法做相容性檢查 → 停手回報 `/architect`→`/po`,在 ROADMAP 決策日誌記一行。

## 提案:整合 / 效能 / 風險(bank 持久化 + 新資料投影進舊分佈)

本設計只出規格與 AC，不寫實作。聚焦三件事：(A) 讓「新資料夾用既有 bank 評分」能接進現有 `run_pipeline`/`score_object`，新增「bank 外部提供、跳過建 bank」路徑；(B) 大量 good 的效能與 coreset budget；(C) 邊界/風險（model/res/維度不符、空 bank、跨資料夾類別不一致、bank 過期）與 U-Net 模組分解、測試策略。

---

## 資料模型

核心原則：bank 物件本身（`MemoryBank.vectors`）是不透明的正規化 patch 矩陣，**單獨存它不足以安全重用**——重用的正確性完全取決於「算新資料的 patch 特徵時，用了與建 bank 時相同的 model + target_res」。因此持久化的單位不是「bank」而是「bank profile = vectors + 一份鎖死語義的 metadata + 投影基底 + few-shot 內容定址清單」。

### 1. 落盤格式：`AnomalyBankProfile`(一個目錄,不是單檔)

存放位置（守 no-dataset-writes）：預設 `dataset_cache_dir(source_folder, "anomaly_bank")`（沿用 `object_eval.dataset_cache_dir` 雜湊規則，落在 `.lv_cache/<folder>_<sha1>/anomaly_bank/<bank_id>/`）；若 User 在 GUI 明確指定硬碟路徑（需求(1)「可選存到硬碟以加速」），則寫該路徑。**絕不寫回來源資料夾**。

一個 profile 目錄含：

| 檔案 | 內容 | 對應既有範式 |
|------|------|------------|
| `bank.npz` | `vectors`(float16 壓縮存、讀回 `astype(float32)`，與 `patch_features` 一致)；`size:int` | `patch_features` 的 float16 npz |
| `meta.json` | 鎖死語義欄位(見下) | `object_eval` 的 `object_policy_profiles.json` profile store |
| `projection.npz` | `proj_mean`(D,)、`proj_vt`(2,D) — 物件級 obj_emb 的 SVD 基底(需求(3)用) | 從 `app.py:1298-1306` 的 `_vt`/mean 抽出保存 |
| `fewshot.csv` | few-shot good/bad 的**內容定址**(非裸 index) | `interaction.curation_log_csv` / `match_shas_to_indices` |

`meta.json` 的鎖死欄位(讀回時逐一校驗，任一不符即拒用 → 見風險段)：
```
{
  "format_version": 1,
  "bank_id": "<uuid 或 source 指紋>",
  "model": "dinov2_vits14",           # 維度 D 的唯一決定者
  "model_dim": 384,                    # = patch_features.model_dim(model)，冗餘自證
  "target_res": 224,                   # 決定 P/grid，跨 res 不可比
  "budget": 100000, "seed": 42,        # coreset 隨機性語義(可重現性)
  "score_mode": "patch",              # bank 只屬 patch 路徑
  "n_normal_objects": <int>,           # 建 bank 用了幾個 good 物件
  "vectors_rows": <int>,               # 抽樣後實際列數(可能 < n_normal_objects*P)
  "source_fingerprint": "<dataset_fingerprint>",  # 來源資料集指紋(對齊既有 match_profile)
  "created_at": "...", "tool_version": "..."
}
```

`fewshot.csv` 每列：`image_path_rel, image_stem, obj_index, bbox_xywh, verdict(good|bad), sha256(可選)`。用內容定址而非裸 index，因為「confirmed 用當次 result 的整數索引」跨資料夾/重掃即失效（既有教訓）。讀回時用 `interaction.match_shas_to_indices` 風格把內容對回「當前 discover 順序」的 index。

### 2. 記憶體模型:`run_pipeline` 新增 `external_bank` 注入點

`run_pipeline` 現況永遠 `build_memory_bank(np.vstack(normal_set 的 patch feats))`（anomaly_tool.py:118）當場重建。新增一條路徑：

```
run_pipeline(..., external_bank: ExternalBank | None = None)
```

`ExternalBank` 是輕量 dataclass（非 streamlit）：`bank: MemoryBank`(重建好的)、`meta: dict`(= meta.json)、`projection: tuple[mean, vt] | None`。當 `external_bank is not None` 且 `score_mode=="patch"`：
- **跳過 normal_set 計算與 `build_memory_bank`**，直接用 `external_bank.bank`；
- 仍對「新資料夾」跑 `embed_objects_patch`（產 query patch 特徵，快取自動分流 model/res）；
- 逐物件 `score_object(all_pf[i]["feats"], grid, external_bank.bank)`。
這是最小侵入：`score_object`/`MemoryBank` 完全不動，只在 `run_pipeline` 的 else 分支前加一個 `if external_bank` 短路。

few-shot bad 仍可用（從 `fewshot.csv` 對回當前 index → 進 `classify` 算門檻/AUROC）；good few-shot 在「外部 bank」模式不再用於建 bank（bank 已外部提供），只供投影著色與門檻。

---

## UX流程

GUI 在側欄 `_anomaly_ui` 既有設定區下方加一個 **「Memory Bank（跨資料夾）」expander**，三個動作對應三個需求；維持「執行邏輯一律委派 run_pipeline / 新 save/load 純函式」的紀律，widget 不含演算法。

**需求(1) 建 bank + 存硬碟（第一次，大量 good）**
1. 照常選資料夾 → 標 good few-shot（或自動 seed normal）→ ▶ 執行偵測 → 得 `result`(含 `bank`、`obj_emb`)。
2. expander 出現「💾 儲存此 Memory Bank」：可選「存進 .lv_cache（預設）」或「另存到指定硬碟資料夾」（原生資料夾對話框，沿用 `_folder_picker` 慣例）。
3. 按下 → 呼叫新純函式 `save_bank_profile(result, dest)`：寫 `bank.npz`(從 `result['bank'].vectors`)、`meta.json`(從 `result['_model']/_target_res` 等)、`projection.npz`(把 app.py:1298-1306 算出的 `mean+_vt[:2]` 也存回 result 再寫盤)、`fewshot.csv`(從 `session_state['anomaly_confirmed']` + records 轉內容定址)。
4. 顯示「已存：<路徑>、<vectors_rows> patch、model=… res=…」。

**需求(2) 讀 bank + 對新資料夾投影評分（第二次）**
1. expander「📂 載入既有 Memory Bank」：列出 .lv_cache 下既有 profiles（用 `match_profile`/指紋排序，顯示 model/res/rows/來源資料夾名）+「瀏覽硬碟」。
2. User 選一個 profile → 載入成 `ExternalBank`（`load_bank_profile(path)` → `build_memory_bank` 不需要，直接 `MemoryBank(vectors)` 重建；大 bank 不存 hnswlib，讀回 lazy 重建）。
3. User 選「新資料夾」→ ▶ 執行偵測（重新偵測旗標也走同路徑）→ run 區塊偵測到 `external_bank` 已載入 → `run_pipeline(..., external_bank=eb, model=eb.meta['model'], target_res=eb.meta['target_res'])`。**model/res 由 bank 鎖死，GUI 對應 widget 在此模式下 disable 並顯示「沿用 bank 設定」**（防 model/res 不符炸維度）。
4. 載入時先跑相容性檢查（見風險段）；不相容 → 紅字擋下、不執行。

**需求(3) 新點映射回舊分佈（可視化）**
- 左欄散點圖改為：若有 `external_bank.projection`，新資料的 `obj_emb` 用 `coords = (emb_new - proj_mean) @ proj_vt.T`（固定基底 transform，**不再各自 svd**）；同畫面以淡色背景畫舊 good 的座標（從 profile 一併存少量舊 good obj_emb 或其 2D 座標，量大時只存 2D 座標雲）。
- 新點落在舊 good 密集區=正常、離群處=可疑，與 `score_object` 的 bank 距離互為佐證（兩個獨立信號：投影是物件級 obj_emb、評分是 patch bank，UX 文案需明說兩者層級不同，避免誤解）。
- 可選複用 `interaction.reference_coverage` / `rank_gap_fillers`：用「舊 good obj_emb 當參照」量新資料覆蓋/離群，作為散點之外的數字指標。

---

## 投影

這是需求(3)的核心機制，也是現況零基礎處（`_vt` 算了從未保存）。

**Fit（建 bank 時，舊資料）**：沿用 app.py:1298-1306 的 `c = emb - emb.mean(0); _u,_s,_vt = svd(c)`。**新增保存** `proj_mean = emb.mean(0)`(D,) 與 `proj_vt = _vt[:2]`(2,D) 進 `projection.npz`。為畫舊分佈背景，再存舊 good 的 2D 座標 `old_good_coords = (_u[:,:2]*_s[:2])[good_mask]`（量大時抽樣存 ≤2000 點即可，純為背景密度雲）。

**Transform（讀 bank 後，新資料）**：`coords_new = (emb_new - proj_mean) @ proj_vt.T`。與舊基底同一座標系，新舊可直接疊圖。LinAlgError/維度不符的退路：若 `emb_new` 維度 ≠ `proj_mean` 維度（model 不符）→ 直接擋（風險段已攔），不退回 `c[:,:2]`（會誤導）。

**關鍵正確性約束**：投影基底是對「物件級 obj_emb」算的，而 bank 是「patch 級」。兩者維度同為 D（同 model）但語義層級不同。需求(3)的散點用投影、評分用 bank，**設計上明確分離、不混用**；profile 同時存兩者，讀回時各走各的。obj_emb 也是 model 決定維度，故 model 不符時投影同樣失效，與 bank 共用同一道 model 校驗即可。

**為何不用 _cov_projection 的 re-fit-together**：`_cov_projection`(app.py:4600-4632)是 vstack 後一起 fit，每次新資料進來舊座標都會變，無法「映射回固定的舊分佈」。需求(3)明確要「固定舊基底、新點 transform 進去」，故採 fit/transform 分離而非 re-fit-together。

---

## 風險

按嚴重度排序，每條給「偵測點 + 處置」：

1. **model 不符 → 維度炸（最高危）**。bank D 鎖死於 DINOv2 變體（vits14=384/vitb14=768/…），`q @ vectors.T` 維度不符直接 ValueError。**處置**：載入時 `meta.json.model_dim == patch_features.model_dim(當前 model)` 硬校驗 + GUI 在 external_bank 模式 disable model widget、強制沿用 `meta['model']`。校驗失敗 → 紅字擋下，絕不執行。

2. **target_res 不符 → 靜默錯誤（次高危，比 model 更陰險）**。res 不改 D，**不會炸**，但同物件不同 res 的 patch 特徵不可比（P/grid 不同），分數全錯卻無例外。**處置**：同樣硬校驗 `meta['target_res']`，external_bank 模式 disable res widget。這條最需要測試守（單元測「res 不符必須 raise/拒用」）。

3. **空 bank / vectors 0 列**。`build_memory_bank` 對空輸入已 `raise ValueError("empty memory bank")`。讀回時若 `bank.npz` 的 vectors 0 列 → 同樣拒用、提示「此 profile 無正常樣本，請重建」。需求(1)若 User 沒標任何 good 且分群 0 群 → 走的是 LOO 路徑根本沒 bank，存盤動作要 disable（無 bank 可存）。

4. **跨資料夾類別不一致**。舊 bank 由某些類別的 good 建立，新資料夾可能含舊 bank 沒見過的類別 → 這些物件對 bank 距離天然偏高（並非真瑕疵，是 OOD）。**處置**：不擋（這正是異常偵測想抓的），但 UX 標註「新類別=分佈外，高分未必是缺陷」；複用既有「物件類別篩選」(991afdd) 讓 User 按類別過濾，並在投影圖上以類別著色。`class_names` 不一致不影響評分維度（評分只看特徵），只影響語義解讀。

5. **bank 過期 / 來源漂移**。`meta.json.source_fingerprint`(= `dataset_fingerprint`) 記建 bank 時的來源；若 User 重建來源資料夾內容變了，舊 profile 仍可用但代表舊分佈。**處置**：載入時顯示 `created_at` 與來源指紋；提供「以當前資料重建 bank」按鈕。不自動失效（過期是 feature，User 要的就是「映射回舊分佈」）。

6. **效能：大量 good 的 coreset budget**。`build_memory_bank` budget=100000 純隨機抽樣(seed=42)，每物件數百 patch × 大量 good 會輕易 >100000 → 落 hnswlib 近似 + 隨機抽樣。風險：(a) 隨機 coreset 無 representativeness 保證，稀有正常 pattern 可能被抽掉 → 偽高分；(b) >8192 走 hnswlib，`set_ef(128)` 是延遲/召回折衷。**處置（保守、不過度設計）**：(i) budget 與 seed 寫進 meta.json，確保「同輸入同 seed 可重現」這層語義可存可比；(ii) **此版不換 greedy coreset**（appetite 控制，純隨機已是既有行為），但在 PRD/設計留 backlog：未來可換 greedy/k-center coreset 提升 representativeness；(iii) GUI 顯示 `vectors_rows` 與是否觸發抽樣，讓 User 知情。存盤存的是**抽樣後**的 vectors（即實際用於評分的那份），保證「存的 = 跑的」，避免讀回重抽樣導致分數漂移。

7. **bank.npz 用 float16 存的精度損失**。patch 特徵已 L2 正規化，float16 對 cosine 距離影響極小（既有 `patch_features` 已這樣存 query 端），可接受；但需測「存→讀→評分」與「不存直接評分」分數差 < 容忍閾值（如 1e-2），守住「存盤不改語義」。

8. **few-shot 內容定址回載失敗**。新資料夾不含舊 few-shot 的圖（正常，跨資料夾）→ `match_shas_to_indices` 命中 0 個 → few-shot 在新資料夾自然不生效（bad 門檻退回無監督百分位）。**處置**：不報錯，提示「few-shot 來自舊資料夾，未在當前資料命中」。

9. **hnswlib 不持久化**。只存 `vectors`，hnswlib index lazy 重建（讀回 `MemoryBank(vectors)` 時 size>8192 自動 `build_nn_index`）。風險：大 bank 每次載入要重建 index（秒級）。**處置**：可接受（一次性、且比重算 DINOv2 快數量級）；不存 hnswlib 二進位（版本綁定、不可攜）。

10. **並發/快取一致性**。`save_bank_profile` 寫多檔，中途失敗 → 半截 profile。**處置**：照抄 `object_eval` 的 atomic replace 範式（寫 tmp 目錄 → `os.replace`），`meta.json` 最後寫（其存在=profile 完整的 sentinel）；`load` 先檢 `meta.json` 存在且 `format_version` 相容。

---

## U-Net模組

依規格疊加紀律：實作進 `scripts/`、測試進 `tests/`、設計只出 `3_Architect_Design/0X_<module>.md` + AC；新增 IO 的模組一律 **Tier B**（done = 單元綠 AND 真實 E2E 綠）。建議拆 4 個 module（同名對齊 design/test/feedback/scripts）：

| module | 角色 / Tier | scripts 落點 | 主要 AC（先紅測試方向） | 整合點 |
|--------|------------|------------|----------------------|--------|
| `anomaly_bank_store` | 持久化(IO) **Tier B** | 新 `scripts/anomaly_bank_store.py`:`save_bank_profile(result, dest)` / `load_bank_profile(path)` / `list_bank_profiles(.lv_cache)` | AC: 存→讀 round-trip vectors bit 級一致(float16 容忍)；meta.json 鎖死欄位齊全；空 bank 拒存；atomic（中途 kill 不留半截）；只寫 .lv_cache 或指定路徑、**不寫來源資料夾**(對齊 no-dataset-writes 測試) | 讀 `result['bank'].vectors`、`_model/_target_res`；回 `ExternalBank` |
| `anomaly_external_bank` | 評分整合(純邏輯) **Tier A→B** | 改 `scripts/anomaly_tool.py`:`run_pipeline(..., external_bank=)` 新分支 | AC: external_bank 給定 → **不呼叫 build_memory_bank**(用 mock/spy 斷言)；用外部 bank 對新 meta 逐物件 score_object；good few-shot 在此模式不建 bank；model/res 取自 bank meta；存→讀 bank 後分數 ≈ 原地建 bank 分數(<1e-2) | 改 anomaly_tool.py:109 前加 `if external_bank` 短路；`score_object`/`MemoryBank` 不動 |
| `anomaly_projection` | 投影(純邏輯) **Tier A** | 新 `scripts/anomaly_projection.py`:`fit_projection(emb)`→(mean,vt)；`transform(emb_new, mean, vt)`→coords | AC: fit 結果與 app.py:1298-1306 同（同 mean、vt[:2]）；transform 對「同一批資料」≈ 原 self-fit 座標(符號可差，用 \|cos\| 或 procrustes 容忍)；維度不符 raise；空輸入處置 | 從 app.py SVD 抽純函式；存進 projection.npz；app 散點改呼叫它 |
| `anomaly_bank_ui` | GUI 整合 **Tier B** | 改 `scripts/app.py` `_anomaly_ui`:存/載/投影 widget + run 區塊接 external_bank | AC(E2E,`tests/e2e/test_anomaly_bank_ui_e2e.py` 標 @e2e)：存 bank→檔案真的在 .lv_cache；載入既有 bank→新資料夾真的用它評分(分數欄非空、heatmap 真的疊)；model/res widget 在 external 模式 disable；**新點投影真的落在舊 good 雲上同一座標系**(取兩個已知 good/bad 點斷言落位)；不相容 bank 載入被紅字擋下 | 委派上面三模組;widget 不含演算法 |

**測試策略要點**：
- `anomaly_external_bank` 用 **spy/mock 斷言「build_memory_bank 沒被呼叫」**（這是「跳過建 bank」路徑的核心行為證據，非代理）。
- 「存→讀→評分」分數一致性測試（module 2 + module 1 整合）是守住「持久化不改語義」的關鍵紅線。
- res 不符**必須 raise**、model 維度不符**必須 raise** 各一條獨立測試（風險 1、2 是最高危且最易靜默）。
- E2E 真實行為斷言（CLAUDE.md 鐵則）：不可只測「element 存在」，要測「已知瑕疵物件用舊 bank 真的被排到前面」「新點真的投影到舊座標系正確象限」「heatmap 真的疊在缺陷位置」。
- PG 開工前 PM 先 `python verify/gate.py --snapshot` 重建 baseline；每 module done 跑 `python verify/gate.py <module>` 印 GREEN，GUI/IO module 另需 E2E 綠。

**相依與順序（給 PO 排 ROADMAP）**：`anomaly_projection`(Tier A,無相依,先做) → `anomaly_external_bank`(改 run_pipeline,依 MemoryBank 既有) → `anomaly_bank_store`(依前兩者的產物格式) → `anomaly_bank_ui`(整合全部,最後)。前三個純邏輯/IO 可先各自綠，UI 的 E2E 收尾。reverse gate 提醒：若 PG 發現 obj_emb 與 patch bank 維度語義在投影上無法調和（需求(3)該對齊哪個表徵），停手回報 /architect。


---
# 三、Round2 對抗批判


## 批判:資料/序列化(memory bank coreset + few-shot good/bad 標記的硬碟持久化、版本相容鍵、過期偵測、穩定身份) — verdict: revise — 核心 round-trip/投影 basis 持久化方向正確且接合點扎實,但 `patch_grid` 硬鍵是與實作(可變 grid)直接矛盾的虛構欄位必須移除、fp16 容忍與 fewshot 格式需與其他三提案收斂、obj_emb 維度檢查與 AC6 符號翻轉需補正,故不可直接 keep。

## 已實證確認(讀過原始碼)

整體方向正確、與既有碼接合點大致成立:`MemoryBank.vectors` 確為唯一序列化必要資料(anomaly_score.py:24-31,hnswlib lazy 重建✓);`build_memory_bank(vectors)` 是純函式✓;`run_pipeline` 簽章可加可選參數而不破壞既有呼叫✓;`obj_emb` 永遠算(anomaly_tool.py:78,投影 basis 來源無虞✓);patch 快取 key `f"{stem}__{obj_index}"` 確在 patch_features.py:80✓;records 已含 `image_path/obj_index/bbox/label`(anomaly_tool.py:129-134✓);`dataset_cache_dir`/`dataset_fingerprint`/`fingerprint_similarity` 簽章與引用全部正確✓;app.py:1301-1306 的 SVD 投影確實算了 `_vt` 卻丟棄✓。這些是這份提案最扎實的部分。

## 致命漏洞

### 1. `patch_grid: [16,16]` 是虛構的硬鍵 —— 會誤擋一切(最高危,設計即錯)
提案把 `patch_grid` 列為**相容性硬鍵**(讀回時必須完全相等否則拒載),並在風險表寫「res 不符即拒載」。但實際上 grid **不是固定值,而是每物件可變**:
- `extract_patch_grid` 用 `ImagePreprocessor(size=target_res, keep_aspect=True)`(patch_features.py:56),走 `_resize_keep_ar_mult14`(models.py:20-31):長邊→target、**兩邊各自 snap 到 14 的倍數**,所以 `grid=(t.shape[2]//14, t.shape[3]//14)` **隨每個 crop 的長寬比變動**(patch_features.py:47)。
- 一個 bank 是由「多個不同長寬比物件的 patch」`np.vstack` 而成(anomaly_tool.py:118),**根本沒有單一 grid 可寫進 meta**。寫 `[16,16]` 是憑空假設正方形 224 輸入,與實作矛盾。
- 後果:① 寫入時無從填這個欄位(只能瞎填);② 讀回比對若真用它當硬鍵,任何非正方形資料集都會「grid 不符 → 拒載」,把正常用例全擋掉。
- **正解**:bank 是 flat (M,D) patch 矩陣,`nn_dist` 只做 `q @ vectors.T`(anomaly_score.py:39),**grid-agnostic**;每物件的 grid 在 query 端各自帶(score_object 用 `all_pf[i]["grid"]`)。所以 `patch_grid` 既不可填、也對 bank 正確性**無關**,應整條移除。真正需要硬擋的只有 `model`(決定 D)與 `target_res`(決定 patch 取樣尺度/分佈可比性)。提案自己在「投影」段也只用 `patch_dim`,前後不一致。

### 2. `target_res` 不符「不會炸但靜默錯」的論述,因 grid 可變而更微妙
提案說「res 不同 → grid 不同 → 不可比」。實際上 res 不同時 D 相同(同 model),matmul 確實不炸(風險表這點對)。但「不可比」的真正原因不是 grid 數字,而是**patch 對應的物理尺度不同**(同一物件在 224 vs 336 下,單一 patch 涵蓋的像素面積不同 → 特徵分佈位移)。硬擋 `target_res` 的結論對,但**理由要改正**(不是 grid 形狀,grid 本來就因物件而異)。否則 PM 寫測試會去斷言「grid 相等」這種對可變值的錯誤約束。

### 3. `proj_dim`/`obj_dim` 與「good_obj_emb 存進 bank.npz」的層級宣稱可疑
提案宣稱 `proj_mean/proj_vt/good_obj_emb` 是「obj 級 `D_obj`,維度同 patch D 但語義不同層級」。需查證:obj_emb 來自 `embed_objects`(object_eval,cls/meanpool head,D=384),patch feats 來自 `forward_features["x_norm_patchtokens"]`(D=384)。兩者數值維度同為 384(同 model)屬巧合,**語義確實不同**。提案此處論述正確,但把兩種語義不同、僅維度巧合相同的向量塞進**同一個 bank.npz** 是埋雷:一旦未來 obj head 改變(如 meanpool→cls)D_obj 仍 384 但語義變、或 patch 模型升級 D 變但 obj 不變,同檔 fp16 混存會讓「讀回端不可混用」這條紀律極脆弱。**建議拆檔**(projection.npz 獨立,與另一份「投影/視覺化」提案的 `ProjectorBasis` 解耦設計一致 —— 該提案明確主張投影器與 bank 分開存,本提案卻反向綁同檔,兩提案直接矛盾)。

### 4. AC1 容忍度自相矛盾
資料模型段說 vectors 存 **fp16**;AC1 要求「round-trip cosine 距離一致,容忍 fp16 誤差 **< 1e-3**」;但風險表又說 fp16 對 cosine「容忍度高」。fp16 的相對精度約 1e-3,L2 正規化後逐元素誤差可累積到 cosine 距離 **> 1e-3** 的情況存在(尤其高維 384)。另一份「整合/效能」提案把同一容忍寫成 **< 1e-2**。同一專案兩份提案對「存→讀分數一致」給出差 10 倍的閾值 —— PM 無所適從。**需釘死單一容忍值並用實測校準**(建議先實測 fp16 round-trip 的實際 cosine 漂移再定,不要拍腦袋 1e-3)。

### 5. fewshot 內容定址的 `obj_index` 穩定性未驗證,且碰撞自檢仍不足
`obj_index` 來自 `discover_yolo_objects` 的 `enumerate(parse_yolo_boxes_conf(...))`(interaction.py:1192),即「該圖 YOLO label 檔內的框順序」。提案 key=`stem__obj_index` 的穩定性**完全依賴 label 檔行序不變**。若 User 重新標註/重存 label,框順序可能變 → 同 key 對到不同物件。提案用 `bbox`+`class_id` 自檢(好),但:① bbox 是 float,需定義比對容忍(完全相等比對在重存 label 後常因浮點格式失敗);② 提案沒說明自檢失敗時除了 skip 外是否嘗試「用 bbox 反查正確 obj_index」(更穩健)。AC4 只測「不符就 skip」,沒測「行序變動但 bbox 仍在 → 應能救回」。**建議內容定址主鍵改用 bbox(量化後)而非 obj_index 序號**,obj_index 僅當 tie-break。

### 6. `embed_objects_patch` 的 cache_dir 與 bank_dir 是兩回事 —— 情境 B 的快取分流宣稱有歧義
提案情境 B 說「對新資料夾 `embed_objects_patch`(自動命中/新建 .lv_cache 快取,用 meta 鎖定的 model/res)」。但 `embed_objects_patch` 的 `cache_dir` 是**逐物件 npz 的目錄**(patch_features.py:81 `cache_dir / f"{key}.npz"`),且 app.py:1249 的 cache key 含 `score_mode_{model}_r{target_res}`。情境 B 用「meta 鎖定的 model/res」算新資料夾的 patch 特徵時,**這個 cache_dir 必須由新資料夾 + meta 的 model/res 組出**,不是 bank_dir。提案沒把這條接線講清楚,PG 容易誤把 bank_dir 當 patch cache_dir。**需明確:bank_dir(序列化產物)與新資料夾的 patch cache_dir(逐物件特徵)是兩個獨立路徑**,且後者的 model/res 必須與 bank meta 一致(否則算出不可比特徵卻不報錯 —— 這才是真正的 silent-wrong 入口,比 res 硬鍵更隱蔽)。

### 7. `external_bank` 注入後 `obj_emb` 仍走 `embed_objects`(obj head),與 bank 的 patch 特徵走不同 extractor 路徑 —— res/model 鎖定要鎖兩處
情境 B 評分用 patch bank,但投影用 obj_emb,而 obj_emb 由 `_object_embeddings`→`embed_objects`(anomaly_tool.py:17-19)算,走的是 **object_eval 的 DEFAULT_POLICY**(pad0.12/r224/cls),**與 patch 路徑的 model/res 是不同參數系統**。提案把「model/res 鎖定」只講在 patch 端,沒說 obj_emb 那條也得用相容設定。若 User 切了 model,obj_emb 維度也變,投影 `(emb_new - proj_mean)` 直接維度不符。硬鍵檢查必須涵蓋「obj_emb 維度 == proj_mean 維度」這條(提案投影段有提 LinAlgError 後備,但維度不符不是 LinAlgError,是 broadcasting ValueError,後備 `eye(2,D_obj)` 救不了)。

## 與其他提案的矛盾(供 PO 收斂)
- **vs「投影/視覺化」提案**:對方主張 `Projector` 與 bank **分開存**(理由:obj 級 vs patch 級硬綁互相污染);本提案主張同檔 bank.npz。直接衝突,見漏洞#3。對方設計較乾淨。
- **vs「整合/效能」提案**:fp16 容忍 1e-2 vs 本提案 1e-3(漏洞#4);對方用 `fewshot.csv`+`match_shas_to_indices`,本提案用 `fewshot.json`+自寫對齊。四份提案各自定義 fewshot 格式(json/csv/marks),**PO 必須在拆 module 時統一,否則四個 module 各寫一套序列化**。

## 較小問題
- meta 的 `created`/`tool_version` 用 git_short_sha:本 repo 是 git,但 bank 可能存到 User 硬碟離開 repo,sha 對讀回端無意義,僅追溯用 —— 標清楚別當相容鍵(提案已歸軟鍵,OK)。
- AC6「transform 等於建 bank 當下 coords」:建 bank 當下 coords 是 `_u[:,:2]*_s[:2]`(app.py:1304),而 transform 是 `(x-mean)@vt.T`。兩者數學上等價(`U S = X_centered Vt.T` 對前 2 維),但**符號可能整體翻轉**(SVD 符號不定),AC6 直接斷言相等會偽紅。需用 `|cos|` 或 procrustes 容忍(另一提案 anomaly_projection 的 AC 有正確指出這點,本提案漏)。

## 批判:UX/流程 — verdict: revise — UX 骨架可用且最小侵入正確,但「2-stage 抽樣」措辭虛構、bank=None 主線未處理、底圖 trace 框選錯位、heatmap model 鎖死遺漏、以及與另兩份提案的檔名/欄位/fewshot 格式三方矛盾必須先收斂,才能進 PM 寫測試。

已實際讀過 anomaly_tool.py / anomaly_score.py / app.py(_anomaly_ui 1178-1492、_folder_picker* 134-217)/ object_eval.py。整體 UX 方向正確、最小侵入哲學對,但有數個與**實際既有碼**衝突或誇大之處,且與同批「持久化/投影」提案有未收斂的契約矛盾。

## A. 與既有碼直接衝突 / 事實錯誤(必修)

1. **「2-stage:執行偵測 → 框選正常 → 再執行」的敘述把不存在的機制當前提。** `run_pipeline(mode, sample_n)` 兩參數**收了但函式體完全沒用到**(anomaly_tool.py:45-49 之後全文無 `mode`/`sample_n`)。現況的「2-stage」純粹是 GUI 迴圈(標 good → `_anomaly_trigger_rerun` → 同一條 run 路徑,confirmed 改變),**不是抽樣階段**。路徑 A 步驟 1 寫「跑 2-stage:執行偵測→框選正常→再執行偵測,此時 result['bank'] 已是乾淨 few-shot bank(現況已支援,零改動)」語意上 OK,但用「2-stage」這個詞會誤導 PM 以為有抽樣子流程要驗。**修正措辭**,否則 AC 會去測不存在的抽樣。

2. **`result['bank']` 在「無 good few-shot」時是 None,但散點/存盤前提沒處理這條主線。** run_pipeline 邏輯(anomaly_tool.py:90-118):只有 `good_idx`(=confirmed good)或分群找到 normal_mask 才建 bank;**大量 good 但 User 還沒框選任何 good 的第一次跑 = 走 LOO 無監督路徑,`bank=None`**(line 95-103)。提案路徑 A「在非常多良品框選一團正常即可」對,但「💾 存目前 bank」的 enable 條件 `result.get('bank') is not None` 會讓**第一次跑完(還沒標 good)時存盤鈕是 disabled**,User 體感是「我跑完了為什麼不能存」。需在 UX 明說「**必須先標≥1 個正常範例(或自動種子)讓 bank 生成**才可存」,否則風險 7 的「一次性 info」會在 bank=None 時根本不觸發,發現性更糟。

3. **散點底圖「疊舊分佈灰點」與既有散點 trace 結構耦合度被低估。** 現況散點(app.py:1322-1335)是**兩條 trace(_dim 變淡 + _hot 實色)+ customdata=全域 index 供框選**,且框選經 `selection_points_to_indices` → `_passes` 過濾。提案要「再疊一層舊 ref 背景點」,但 plotly 框選事件回傳的 point index 會**跨所有 trace 連號**;若舊 ref 點也是可框選 trace,`selection_points_to_indices`/customdata 對齊會錯位(舊點被框進來 → customdata 指向新資料 index 造成張冠李戴)。必須明確規定**舊 ref 背景 trace 不可被套索選取**(設 `hoverinfo`/不放 customdata 或用獨立不可選 trace),這點提案只說「疊一層底圖」未鎖死,是 E2E 會炸的隱形坑。

4. **「coords2d 是現有恆 None 欄位,正好拿來裝投影座標」— 但散點圖目前根本不讀 coords2d。** 確認 result['coords2d'] 永遠 None(anomaly_tool.py:64,139),且 `_anomaly_ui` 散點**完全沒讀它**,是每次即時 `np.linalg.svd`(app.py:1300-1306)。所以「塞進 coords2d 就生效」是假的——必須**同時改 app.py 散點分支去讀 coords2d**(掛載時)。提案在「投影」段確實要求改座標來源,但 U-Net 模組表把這寫成「架構面在 run_pipeline 回傳 coords2d」就夠,**漏掉 GUI 端必須加 if-掛載-讀-coords2d-else-現況-svd 的分支**,這是 UX 提案自己該鎖的觸點卻甩給架構面。

## B. 與鐵則 / no-dataset-writes

5. **路徑寫入防呆「目標路徑 == 任一已選 folder 或其子目錄即擋」不夠。** `dataset_cache_dir` 落點是 `.lv_cache/<name>_<sha>/...`(object_eval.py:44-50),而 User 選的「指定硬碟資料夾」可能是**資料集的父目錄**(images/labels 在子層)。風險 4 只擋「== folder 或子目錄」,擋不住「選到資料集的父層」→ bank 寫進去後與資料集混居仍違反精神。建議改為**正向白名單**:只允許 .lv_cache 或「不含 images//labels/ 子目錄、且非任一已選 folder 之祖先/後代」的路徑。

6. **「指定硬碟資料夾」缺對應 picker 原語。** 既有只有 `_folder_picker_list`(多資料夾清單,app.py:201)與 `tkinter.filedialog.askdirectory`(134/154/170)。提案多處畫「📁 picker 選 bank 資料夾(單一)」「📁 改硬碟路徑」,但**沒有 `_folder_picker_single`**(另一份投影提案直接引用 `_folder_picker_single('anomaly_proj_path')` 當既有物,實際不存在)。UX 要嘛復用 askdirectory 包一顆單選鈕,要嘛新增原語——必須點明這是**新 widget 要做**,否則 PG 會以為有現成的。

## C. 跨提案矛盾(PO 收斂時必爆)

7. **檔名/欄位三份提案各說各話,UX 這份又是第三套。** bank 目錄結構:資料/序列化版=`bank.npz/bank_meta.json/fewshot.json`;整合版=`bank.npz/meta.json/projection.npz/fewshot.csv`;**本 UX 版=`vectors.npz/manifest.json/projection.npz/fewshot.json`**。fewshot 還有 json vs csv 之爭。manifest 欄位名(`dim` vs `patch_dim`/`obj_dim`、`n_patches` vs `vectors_rows`、`source_fingerprint` vs `source.fingerprint`)全不一致。UX 提案宣稱「欄位刻意對齊 save_profile」,但 `save_profile`(object_eval.py:164)的簽章是 `(name, fingerprint, policy, metrics, ...)`,**根本沒有 model/target_res/budget 這些欄位**,所謂「對齊」是空話。這是 PO 拆 module 前**必須三方收斂成單一契約**的單點(CLAUDE.md:粒度轉換點唯一無紅綠保護處)。

8. **projection 同時要存「2D 座標雲」又要「mean+components」,但 D 維離群判定需求未被本 UX 版承接。** 投影提案明確要存 `ref_good_emb`(D 維)在**原始空間**算 in_dense(2D 會失真);本 UX 版只說「存少量代表點 2D 座標當灰底」。若最終採 2D 近似,「新點落在 good 密集區=正常」的判定會在 PCA 前 2 主成分壓扁時假陰——這是投影提案風險 2 點名的最大坑,UX 版用 2D 底圖**靜默選了較不準的那條**而沒聲明。需與投影面向對齊:**底圖用 2D、但紅/綠判定用 D 維**(否則散點顏色會騙人)。

## D. 漏洞 / 未爆但會痛

9. **掛載外部 bank 後,右欄 heatmap 的索引重算路徑會錯。** heatmap(app.py:1477-1484)是用 `result['_image_paths']` 重跑 `discover_yolo_objects` 再取 `meta[pick]` → `embed_objects_patch([meta[pick]], _model, _target_res, _cache)`。外部 bank 模式下 `_model/_target_res/_cache` 必須**被 bank 的 meta 鎖死值覆寫**(不是 GUI 當前 widget 值),否則 User 掛了 vits14@224 的 bank、widget 卻停在 vitb14 → heatmap 重算 patch 用錯 model → `score_object(...,外部bank)` 維度炸。提案「右欄 heatmap 完全沿用既有碼、無需 UX 改動」是**錯的**:必須保證 result['_model'/'_target_res'] 寫的是 bank 鎖死值。這條要明確列為 UX 觸點。

10. **「一鍵同步設定」改 selectbox 的 session key 會踩 Streamlit 慣例。** 想把側欄 model/res 改成 manifest 值,得寫 `st.session_state['anomaly_model_sel']=...`,但該 key 已綁 `st.selectbox`;在 widget 實例化後同回合改其 key 會被 Streamlit 警告/拒絕(需在 widget 建立前改 + rerun)。可行但**有時序坑**,提案當成「提供一鍵」一筆帶過,PG 會卡。

11. **few-shot 內容定址用 `(image_stem, obj_index)` 作鍵,跨資料夾同名 stem 會誤配。** 資料/序列化版有帶 bbox+class_id 自檢,本 UX 版 fewshot.json 雖列了 bbox 欄位但**讀回對齊只說「用 (image_stem,obj_index) 對到 records」**,沒講碰撞自檢。大量 good 場景跨產線常有 `img_0001` 撞名 → 靜默配錯成「舊標記貼到不相干新物件」。需補 bbox 容差自檢(對齊另兩份提案),否則 few-shot 顯示會錯。

## 可取之處(keep 的理由)
- 「不新增第二主按鈕、用掛載狀態讓單一『執行偵測』表現雙行為」的決策正確,確實降低誤操作面,且與既有單一 run 路徑(app.py:1238 `run or pop(_anomaly_rerun)`)天然相容。
- 三處冗餘提示(膠囊/按鈕文案/結果列)防「模式隱形切換」是務實的。
- few-shot 內容定址、預設落點 .lv_cache、相容性硬 gate(model+res)方向都對。

## 批判:投影/視覺化:跨資料夾 fit/transform 投影器 + 「新資料映射回舊分佈」視覺 — verdict: revise — 投影 fit/transform 與固定舊基底的核心洞見正確且有價值,但 L2norm 幾何自相矛盾(打臉「與今日散點相同」的 AC)、ref_idx/in_dense 的函式復用不成立、與「持久化」面向重複搶投影 basis 序列化、獨立存投影器的落點契約真空,四處須先收斂才能交 PM 寫測試。

## 已讀過實際碼,逐條對證

### 漏洞 1(最嚴重,直接打臉「與今日散點完全相同」的核心賣點)— L2norm 幾何不一致
- 既有散點 `app.py:1298-1306` 對 **raw obj_emb** 做 SVD:`c = emb - emb.mean(0); svd(c)`,**沒有 `_l2norm`**。而 `embed_objects`(object_eval.py:250/260)回傳的是**未正規化的 raw DINOv2 向量**(我已逐行確認,return 前無 `_l2n`)。
- 本提案 `fit_projector(emb, *, l2norm=True)` 預設先 `_l2norm` 再 SVD,並自稱「完全沿用 app.py:1298-1306 既有 SVD 數學」「ref_coords 與今日散點完全相同」。**這兩句互相矛盾**:加了 L2norm 後座標就**不等於**今日散點(球面化會改變主成分方向與點位)。
- 後果:User 在情境 A「存投影器」當下看到的散點(若仍走舊路徑=raw),與投影器內 `ref_coords`(=l2norm 後)**不是同一張圖**;情境 B 新點用 l2norm basis transform,User 拿來和「記憶中那張 raw 散點」對照 → 落點語義錯位。**這正是 AC「ref_coords 與今日散點相同」會直接紅的點**,PG 無法在不改既有 app 散點(改成 l2norm)或不改 fit 預設(改成 l2norm=False)之間二選一而不破壞另一面。屬設計層必須先定的契約,不是 PG 自由度。建議:`l2norm` 預設值必須與「最終散點要走哪種幾何」綁定,並在設計明寫「同時改 app.py 散點為 l2norm」或「fit 預設 l2norm=False 對齊現況」,二擇一寫死,不可含糊。

### 漏洞 2 — `compute_outlier_scores` 拿不到 `ref_idx`,連線功能無實作基礎
- 提案 `_proj_nn = {ref_idx, dist, in_dense}`,連線 trace(第 3 層)要「新點 → 最近舊 ref 點」的 index。但提案指定用 `interaction.compute_outlier_scores` 算 in_dense。我讀了該函式(interaction.py:149-154):`dist, _ = nn.kneighbors(...)` —— **丟掉了 indices(`_`)**,且回傳的是 **k 鄰平均距離**,不是「到最近單一鄰居」的 index。
- 所以 `ref_idx`(連線端點)**沒有現成來源**;且 `compute_outlier_scores` 是 mean-of-k,不適合當「最近鄰連線」的單一最近點。要嘛新寫 kNN 取 index(那就不是「復用 `compute_outlier_scores`」),要嘛連線退化。提案把它說成「復用既有純函式」是不準確的——這條要新函式。屬可行但需誠實標為新增,別當零成本復用。

### 漏洞 3 — `in_dense` 的 reference 與 score 路徑層級錯配且 tau 來源未定
- `in_dense = compute_outlier_scores(emb_new, ref_good_emb) <= tau`,tau=「舊 good 自身 kNN 距離 95 分位」。但 `compute_outlier_scores` 用 sklearn `metric="cosine"`,而提案前面 fit 又可能對 emb 做 `_l2norm`(cosine 幾何下 L2norm 是冗餘但不衝突)——**前後度量需一致**否則 tau 不可比。更關鍵:tau 在 fit 時對「舊 good 對舊 good」算(含自比 → 要 `candidates_in_reference=True` 去掉自身),提案文字沒提這個 flag,漏了會讓 tau 偏低(自身距離=0 拉低分位)→ 新點普遍被判離群。這條在實作會默默錯。

### 漏洞 4 — `_folder_picker_single` / picker 慣例不存在
- 提案兩處寫 `_folder_picker_single('anomaly_proj_path')`「沿用 `_folder_picker` 慣例」。實際 app.py 只有 `_pick_folder`(130)、`_pick_file`(140)、`_folder_picker_list`(201)等,**無 `_folder_picker_single` 也無 `_folder_picker`**。要選「單一目錄」得用 `_pick_folder(session_key)` + 自繪 text_input(`_pick_folder` 把路徑寫進 session_key)。不是致命,但「沿用既有慣例」名稱對不上,PG 會找不到函式。需更正為實際 helper。

### 漏洞 5 — 與「持久化」面向的落點契約有循環依賴 / 職責真空
- C2 說「投影器存盤落點由持久化面向統一決定,投影器只接收 `dir`」。但情境 A 的 UX(本提案自己畫的)是「左欄散點下方一顆『存成投影器』按鈕」,**獨立於 bank 存檔**(提案明寫「解耦,可只存投影器」)。若 User 只存投影器、還沒存 bank,「bank 落點同目錄」這個 dir **還不存在**——那投影器的 dir 由誰決定?提案說「不自己決定路徑策略」,但此情境下沒有 bank 面向給 dir → **落點真空**。要嘛投影器自己也要能走 `dataset_cache_dir(old_folder, kind)`(那就違反 C2 的「不自己決定」),要嘛強制「存投影器必須先有 bank dir」(那又違反「解耦可獨立存」的 UX 賣點)。**內部自相矛盾**,需收斂。

### 漏洞 6 — 「umap 第二按鈕」與既有 `_cov_projection` 的 token 契約沒對齊
- 提案說 umap 走「直接複用既有 `_cov_projection(ref_emb, new_emb, "umap", 2, …)`」。但我讀了 `_cov_projection`(app.py:4600)簽章是 `(dataset_emb, cand_emb, method, dim, token, cand_token)` —— **需要 token / cand_token 做快取鍵分流**(避免整圖/物件級互撞)。提案的呼叫省略了 token,且 `_cov_projection` 內部對 combined 強制 `_l2norm`(4617),又與本提案 PCA 路徑的 l2norm 開關語義打架(一邊可關、一邊強制)。「直接複用」沒這麼直接。

### 與既有碼/鐵則的正向確認(這些是對的,給 keep 依據)
- transform-into-fixed-basis vs `_cov_projection` 的 re-fit-together 區別**正確且重要**:`_cov_projection`(4616 vstack 後一起 fit)確實會讓舊座標跟著新資料動,不符「映射回固定舊分佈」。這是本提案最有價值的洞見,且 LinAlgError 退路 `c[:,:2]` 與既有 1305-1306 一致。
- fp16 + 確定性下採樣(seed=42)畫灰底、components/mean 不抽樣 → 合理。
- 投影獨立於 bank、LOO 模式也有 obj_emb(我確認 run_pipeline 各分支都產 obj_emb,78-83)→ 風險 5 結論正確。
- coords2d 欄位現況恆 None(anomaly_tool.py:139)可借用 → 但本提案改用 `result["_proj_new"]` 而非塞 coords2d,與「持久化」面向(它要塞 coords2d)**潛在欄位之爭**,需 PO 仲裁誰用 coords2d。

### 與其他提案的矛盾
- 「持久化」面向把投影 basis 存進 `bank.npz`(proj_mean/proj_vt/good_obj_emb),本提案另立 `projector.npz`+`projector.json` **獨立檔**。兩者都聲稱自己負責投影 basis 的序列化 → **重複/衝突**。PO 必須砍一個:要嘛投影 basis 歸 bank.npz(持久化面向),本提案只做「視覺化消費」不自己存檔;要嘛獨立 projector 檔,持久化面向別存 proj_*。現狀是兩份設計都搶這塊,落地會撞。本提案自稱「刻意不重複設計序列化」,但 `projector.npz`/`save_projector`/`load_projector` 明明就是在設計另一套序列化,**言行不一**。

## 批判:整合 / 效能 / 風險(bank 持久化 + 新資料投影進舊分佈) — verdict: revise — 方向與既有碼貼合度最高、external_bank 短路設計正確,但「obj_emb 投影未 l2norm 的幾何」「patch 快取 cache_dir 必須用 bank meta 的 res 組名(否則 silent-wrong)」「fewshot 內容定址鍵碰撞防護」「normal_set in-pipeline 旁路點」四處是真漏洞且未進 AC,且 fewshot.csv/底圖只存 2D 與跨提案矛盾,須由 PO 收斂格式後補 AC 才可放行。

已實際讀過 anomaly_tool.py / anomaly_score.py / patch_features.py / object_eval.py / interaction.py / app.py 相關段。整體方向正確且與既有碼貼合度最高(external_bank 短路、float16 npz、atomic replace、model/res 硬校驗都對),但有數個**load-bearing 的事實錯誤與漏洞**會讓 PG 落地時踩雷或產生 silent-wrong。

## A. 事實錯誤(直接影響正確性)

1. **obj_emb 投影幾何寫錯 —— 既有 anomaly 散點「不」做 L2 正規化。** 本文「投影」段照抄 app.py:1298-1306 是對的,但**那段散點對 raw `c = emb - emb.mean()` 做 SVD,完全沒有 `_l2norm`**(app.py:1301-1304 我已逐行讀過)。`embed_objects`(object_eval.py:250)存的也是 `embed_fn(crop)` 原始向量,**未正規化**(`_l2n` 在 object_eval:265 存在但 embed_objects 不呼叫)。本文沒提 l2norm(這點比同場「投影/視覺化」提案誠實,那份硬加了 `l2norm=true` 並宣稱「與 _cov cosine 幾何一致」是錯的)。**但你引用 `interaction.reference_coverage`/`compute_outlier_scores` 來量「新點落舊 good 密集區」時用的是 cosine metric** —— 投影座標來自 raw-SVD、密度判定來自 cosine-kNN,**兩套幾何不一致**。AC 必須鎖死「投影 transform 用 raw(不 l2norm),與現行散點逐點一致」,否則 PG 會自行決定要不要正規化 → 與既有散點不可疊。這條沒寫進 AC 是真漏洞。

2. **`match_shas_to_indices` 被誤當可復用範式。** 我讀過 interaction.py:179-193:它簽章是 `(shas, sha_to_index)`,純粹 dict 查表,**鍵是 sha256**。你的 `fewshot.csv` 用 `(image_stem, obj_index)` 內容定址 —— 那是**另一個鍵**,`match_shas_to_indices` 幫不上,頂多「風格」相似。文中「讀回時用 `interaction.match_shas_to_indices` 風格把內容對回」會誤導 PG 以為有現成函式可呼叫。實際要新寫 `{stem__objidx → new_index}` 映射(這倒簡單,但別假裝復用)。同理 `curation_log_csv`(interaction:160)寫的是 ts/reason/filenames/sha256s,**欄位與你的 fewshot.csv 完全不同**,「對應既有範式」這欄是裝飾性的,別讓 PM 以為照抄即可。

3. **few-shot 內容定址鍵會碰撞且本文未防。** `discover_yolo_objects`(interaction:1190-1199)的 `obj_index` 是**單張圖內 box 的 enumerate 序號 k**,不是全域唯一。鍵 `{stem}__{obj_index}` 在「不同資料夾有同名 IMG_0001.jpg」或「同圖 label 改了 box 順序」時會錯配。同場「資料/序列化」提案明確加了 `bbox+class_id` 自檢,**你的 fewshot.csv 雖有 bbox 欄但風險段第 8 條只講「命中 0 個」,沒講「同名誤命中」**。需補一條 AC:同名 stem 但 bbox/class_id 不符 → skip 且可觀測(不可靜默回填錯 index)。

## B. 整合漏洞(會 silent-wrong)

4. **external_bank 模式的 patch 快取目錄會吃到錯 res 的 stale npz。** 我讀過 patch_features.py:80-81 —— 快取鍵 `{stem}__{obj_index}.npz` **不含 model/res**;分流完全靠 `cache_dir` 名(app.py:1249 = `anomaly_patch_{score_mode}_{model}_r{target_res}`)。你的需求(2)「對新資料夾跑 embed_objects_patch、快取自動分流 model/res」**只有在 cache_dir 用 bank.meta 的 model/res 組名時才成立**。若 GUI widget 被 User 改過、或 PG 用 widget 值組 cache_dir,就會把 bank(vits14@224)的 query 寫到/讀自錯 res 的目錄 → matmul 不炸但分數全錯(正是你風險 2 點名的最陰險 silent-wrong,卻沒連到這個快取路徑)。**必須加 AC**:external_bank 模式下 cache_dir 名一律由 `meta['model']/meta['target_res']` 決定,GUI widget 值不參與組名。這是本提案最關鍵的落地缺口。

5. **`run_pipeline` 的 obj_emb 快取(objemb_{model}.npz)只 key by model,不含 res。** anomaly_tool.py:77 我已讀:`objemb_{model}.npz` 跨 score_mode/res 共用。投影 basis 來自 obj_emb;對「物件級 obj_emb」res 不影響維度但 crop→resize 流程相同,所以共用尚可。但你風險段把 res 不符列為次高危(對 patch 對),**卻沒說明 obj_emb 快取的 res 無關性是有意設計** —— PG 會困惑「為何 patch 要 res 分流、obj 不用」。設計要明寫:投影 basis 對 res 不敏感(obj_emb 整體向量),patch 評分對 res 敏感,兩者快取策略不同是刻意。

6. **external_bank + few-shot good 的 normal_set 衝突未在本文解。** 你說「good few-shot 在外部 bank 模式不用於建 bank,只供投影著色與門檻」。但 anomaly_tool.py:87-93 的 `normal_set` 決策(有 good_idx 就用 good_idx,否則用 normal_mask)**在你的 if external_bank 短路前就跑了**。若 PG 只在 build_memory_bank 前短路、沒短路 normal_set 計算,新資料夾的 confirmed/分群仍會算 normal_set(浪費且語義混亂)。同場 UX 提案的風險 6 點到了「掛載後正常範例 UI 唯讀」,你的版本漏了這個 in-pipeline 分支點。AC「good few-shot 不建 bank」不夠,要加「external_bank 模式 normal_set 計算被旁路或忽略」。

## C. 與其他提案的矛盾

7. **三份提案的落盤格式不一致,PO 必須收斂。** 「資料/序列化」用 `bank.npz/bank_meta.json/fewshot.json`(json);本文用 `bank.npz/meta.json/projection.npz/fewshot.csv`(csv)；UX 用 `vectors.npz/manifest.json/projection.npz/fewshot.json`。三套檔名+格式打架。你的 `fewshot.csv` vs 他們的 `fewshot.json`:csv 對 bbox(4 float)/可選 sha 的表達不如 json 自然,且既有可比範式 `object_policy_profiles.json` 是 json。**建議改 json 對齊另兩份**,reject 你的 csv 選擇(理由:bbox 陣列、跨資料夾可選欄位用 json 較穩,且既有 profile store 就是 json)。這是跨提案必須由 PO 拍板的單點。

8. **「投影底圖」存什麼,三份各說各話。** 你存 `old_good_coords`(2D,抽樣≤2000);「資料/序列化」存 `good_obj_emb`(D 維 fp16);「投影/視覺化」兩者都要(D 維算 in_dense + 2D 畫底圖)。**只存 2D 座標 → 無法在 D 維算「新點落舊 good 密集區」**(2D 距離失真,你風險 2 自己也承認線性 PCA 會壓離群點)。你的需求(3)「密集區 vs 離群」判定**需要 D 維 good_obj_emb**,只存 2D 是不夠的。本文存 2D 與你自己引用 `reference_coverage`(cosine,D 維)矛盾。revise:至少存舊 good 的 D 維向量(fp16),否則密集區判定降級成 2D 近似要明講。

## D. 次要 / 可接受

- AC 表「vectors bit 級一致(float16 容忍)」措辭矛盾(bit 級 vs 容忍),改「cosine 距離差 < 1e-3」。
- 「`build_memory_bank` budget 純隨機、此版不換 greedy」appetite 控制合理,keep;但 seed=42 寫死(anomaly_score:48),meta 記 seed 對「重抽」無意義(你存的是抽樣後 vectors,讀回不重抽)—— 記 seed 只為追溯,文中已說對,OK。
- atomic「meta.json 最後寫當 sentinel」對,但你有 4 個檔(bank/meta/projection/fewshot),要明寫**順序**:vectors+projection+fewshot 先,meta 最後 os.replace,load 先檢 meta 存在。否則 meta 在但 projection 半截仍會炸。
- spy/mock 斷言「build_memory_bank 沒被呼叫」是好 AC,keep。


---
# 四、Round3 回應修正


## 修正:資料/序列化(memory bank coreset + few-shot 標記的硬碟持久化、版本相容鍵、過期偵測、穩定身份)


# 修正後最終立場(資料/序列化面向)

我讀過 `patch_features.py`、`anomaly_score.py`、`anomaly_tool.py`、`models.py`、`object_eval.py`、`interaction.py`、`app.py:1240-1320` 後,**接受批判中的全部 7 條致命/實質問題與全部較小問題**。其中 #1、#5、#7 我原設計是真的錯,不是措辭問題;以下逐條給「改/守」與理由,最後列承認的取捨。

## 接受並修正(設計錯誤)

### #1 移除 `patch_grid` 硬鍵 —— 我錯了,且錯得最危險
已實證:`_resize_keep_ar_mult14`(models.py:28-29)對 `nw`、`nh` **各自** snap 到 14 的倍數,grid `=(t.shape[2]//14, t.shape[3]//14)`(patch_features.py:47)**逐物件變動**;bank 由多個不同長寬比物件 `np.vstack`(anomaly_tool.py:118),**根本沒有單一 grid 可寫**。我寫 `[16,16]` 是憑空假設正方形,且若當硬鍵會把所有非正方形資料集全擋掉(誤擋正常用例)。
**改**:`bank_meta.json` **整條移除 `patch_grid`**;`save_bank`/`check_compat` 都不碰它。bank 是 flat `(M,D)`,`nn_dist` 只做 `q @ vectors.T`(anomaly_score.py:39)**grid-agnostic**;每物件 grid 在 query 端由 `score_object` 各自帶(`all_pf[i]["grid"]`),與 bank 相容性無關。`patch_grid` 連「冗餘自校驗」的資格都沒有(它不是常數)。

### #2 `target_res` 硬擋「守」,但理由改正
**守**硬擋 `target_res`,但理由不是「grid 形狀不同」(grid 本來就因物件而異),而是**patch 對應的物理像素尺度不同 → 特徵分佈位移 → 距離靜默錯**(silent-wrong,比 crash 更危險)。這條要明確寫進設計,否則 PM 會去斷言「grid 相等」這種對可變值的錯誤約束(AC2 必須只測 model/target_res/dim 不符,**不得**含 grid)。

### #3 拆檔 —— `projection.npz` 與 `bank.npz` 分離
已實證:obj_emb 走 `embed_objects` 的 `DEFAULT_POLICY={pad:0.12, r224, head:"cls"}`(object_eval.py:65),patch 走 `head="meanpool"`(patch_features.py:40)。兩者 D=384 僅在 vits14 下**巧合**相同(object_eval.py:56 註解自證「All heads stay D=384」),語義是不同 extractor/head。把語義不同、僅維度巧合的向量塞同檔是埋雷,且與「投影/視覺化」提案的 `ProjectorBasis` 分離主張正面衝突 —— 對方較乾淨。
**改**:`<bank_dir>/` 內拆成 `bank.npz`(只放 patch `vectors`)+ `projection.npz`(`proj_mean`/`proj_vt`/`good_obj_emb`,obj 級)+ `bank_meta.json` + `fewshot.json`。meta 分記 `patch_dim` 與 `obj_dim` 兩個獨立欄位;讀回 API 分開回傳,從型別上杜絕混用。**這也讓 object 模式(無 patch bank)可只存 projection,不需假 bank**(順帶回應我原 Reverse gate 的疑問)。

### #4 容忍度釘死一個值,且實測校準(不拍腦袋)
我原本 fp16 + AC1<1e-3 與風險表「容忍度高」自相矛盾,且與另兩份提案的 1e-2 差 10 倍。
**改**:① 序列化 dtype 與容忍值由**實測**決定,不預設。PM 寫 AC1 前,先跑一支微基準:對真實 384 維 L2 正規化向量做 round-trip,量 fp16 的實際逐物件 `nn_dist` 漂移。② **預設改 fp32 存 vectors**(coreset 已被 budget≤100k×384 上限,fp32≈150MB 仍可 `savez_compressed`;正確性 > 省一半空間,因為這是評分數值的單一真相);fp16 降級為 meta 內可選旗標 `vectors_dtype`,選了就放寬容忍。③ AC1 釘**單一**容忍值(預設 fp32 → `< 1e-5`;選 fp16 → 用實測上界,不是猜的 1e-3),全專案統一,寫進 ROADMAP 決策日誌讓四個 module 不各寫一套。

### #5 few-shot 內容定址主鍵改 bbox,obj_index 降為 tie-break
已實證:`obj_index=k` 來自 `enumerate(parse_yolo_boxes_conf(...))`(interaction.py:1192),**純 label 檔行序**,重標註/重存即可能變。我原 key=`stem__obj_index` 穩定性完全押在行序不變上。
**改**:① 主鍵=`(image_stem, 量化後 bbox)`,`obj_index` 僅 tie-break(同圖多個近似 bbox 時)。② bbox 比對定義**量化容忍**(如四捨五入到 1e-3 的歸一化座標再比,避免重存 label 的浮點格式差異造成完全相等比對失敗)。③ 對齊邏輯升級為「先用量化 bbox 反查正確 `new_index`」而非只 skip;真配不上才 skip + 可觀測。AC3/AC4 補一條:**行序被打亂但 bbox 仍在 → 必須救回對齊**(不是只測「不符就 skip」)。④ 復用 `interaction.match_shas_to_indices` 的內容定址樣板精神(已確認在 interaction.py:179),不自寫一套。

### #6 明確切開 bank_dir 與 patch cache_dir 兩條獨立路徑
已實證:`embed_objects_patch` 的 `cache_dir` 是**逐物件 npz 目錄**(patch_features.py:81),app.py:1249 的 cache key 含 `anomaly_patch_{score_mode}_{model}_r{target_res}`。情境 B 算新資料夾 patch 特徵時,cache_dir 必須由**新資料夾 + meta 鎖定的 model/res** 組出,**不是 bank_dir**。
**改**:設計明文寫死 ——「序列化產物路徑(bank_dir)」與「新資料夾逐物件 patch 快取路徑(`dataset_cache_dir(new_folder, f"anomaly_patch_patch_{meta.model}_r{meta.target_res}")`)」是**兩個獨立路徑**;後者的 model/res **必須**用 meta 的值而非當前 GUI widget 值,否則算出不可比特徵卻不報錯(這是比 res 硬鍵更隱蔽的真正 silent-wrong 入口)。E2E AC 要斷言:載入 bank 後,新資料夾的 patch 快取 key 確實帶 meta 的 model/res。

### #7 硬鍵涵蓋 obj_emb 維度 / model 鎖兩處
已實證:obj_emb 維度由 model 決定(`embed_objects` 走 model 的 backbone,head 固定 cls)。投影 `(emb_new - proj_mean)` 維度不符時是 **broadcasting ValueError**,不是 LinAlgError,我原本的 `eye(2,D_obj)` 後備**救不了**。
**改**:① 硬鍵把 `model` 當「同時鎖 patch 與 obj 兩條 extractor」的單一鍵(D 與 D_obj 都由它定);`check_compat` 額外驗 `obj_emb_new.shape[1] == proj_mean.shape[0]`,不符即**拒投影 + 明確訊息**,絕不讓它跑到 broadcasting。② 設計講清楚 obj_emb 那條也鎖 model(res 由 DEFAULT_POLICY 固定 r224,不隨 GUI patch res 變,這點要在文件點明避免誤解)。

## 接受(較小問題)
- **AC6 符號翻轉**:SVD 符號不定,建 bank 當下 coords=`_u[:,:2]*_s[:2]`(app.py:1304 已證),transform=`(x-mean)@vt.T` 數學等價但**整體可能翻轉**。AC6 改用 `|cos|` 對齊或 procrustes 容忍,直接斷言相等會偽紅 —— 對齊 anomaly_projection 提案的正確做法。
- `created`/`tool_version`/`folder_sha1`:維持軟鍵(bank 可能離開 repo,sha 對讀回端無意義),設計標清楚「僅追溯,絕非相容鍵」。

## 守住(批判未動搖的核心)
- 存 bank = 存 `MemoryBank.vectors`、讀回 `MemoryBank(vectors)` 重建、**不存 hnswlib**(已證 anomaly_score.py:29-31 lazy 重建)。
- `run_pipeline` 加可選 `external_bank=` 不破壞既有簽章(已證 anomaly_tool.py:45-49)。
- 投影採 **transform-into-fixed-basis**(非 re-fit-together),才符合「映射**回**舊分佈」語義;這是與 `_cov_projection` 的刻意差異,守。
- fingerprint **只警告不擋**(需求(2)本就是拿舊 bank 看不一樣的新資料);唯一硬擋 model/target_res/dim 不相容。
- atomic replace(`*.tmp`→`os.replace`)、壞檔拋明確例外不靜默回空 bank。
- 預設落點 `dataset_cache_dir`、User 自訂路徑只寫該目錄、絕不回寫資料集 —— 鐵則不變。

## 承認的取捨
1. **fp32 預設犧牲空間換正確性**:bank 體積翻倍(~150MB/bank),但避免「存→讀分數漂移」的爭議與跨提案閾值打架。空間靠 `savez_compressed` + User 自管硬碟位置緩解。能接受。
2. **拆三檔/四檔提高 IO 複雜度**:多一次開檔、多一份 meta 維度對賬,但換來 obj/patch 層級從型別上隔離、object 模式可只存 projection。值得。
3. **bbox 主鍵需定量化容忍**:引入一個經驗門檻(歸一化座標四捨五入位數),是新的可調參數面;但比押注 label 行序穩定可靠得多。門檻寫進 meta 供追溯。
4. **跨資料夾 few-shot 沿用仍是 bonus 不是保證**:即便改用 bbox 主鍵,物件不在新資料夾就是配不上 —— 這是語義事實,不強塞。few-shot 主要服務「建 bank 那次」,沿用是加分。
5. **與其他提案的收斂交給 PO**:fewshot 格式(json/csv)、容忍值、投影器是否獨立存,四份提案不一致 —— 我讓步採「projection 獨立存 + 容忍實測 + 內容定址」與另兩份對齊,但最終單一真相須由 PO 在拆 module 時釘死並記決策日誌,否則四 module 各寫一套序列化。


## 修正:UX/流程

已逐條核對既有碼(anomaly_tool.py:45-141、anomaly_score.py、app.py:1178-1492 與 130-217、object_eval.py:44-184、interaction.py:694-719/1173)後的最終立場。批判命中率極高,11 點裡 10 點屬實必修,只有 1 點機制描述要校正(但補救方向對)。以下「接受/校正/堅持」逐條給理由,並承認取捨。

---

## A. 與既有碼衝突 / 事實錯誤 —— 全部接受(措辭與觸點都要改)

**A1(措辭誤導)接受。** 確認 `run_pipeline` 的 `mode`/`sample_n` 收了不用(45-49 後全文無引用),現況「2-stage」是 GUI 迴圈(標 good → `_anomaly_trigger_rerun` → 同條 run 路徑,只是 `confirmed` 變了),不是抽樣子流程。**修正**:設計全文把「跑 2-stage 抽樣」改寫為「**標記正常範例 → 重跑(confirmed 改變使 normal_set 切到乾淨 few-shot)**」,並在交棒 PM 的 AC 種子明寫「**不得**為此功能新增/驗證抽樣階段」。取捨:沿用既有詞彙原想降溝通成本,反而會讓 PM 去測不存在的機制,得失不償。

**A2(存盤鈕 enable 條件會誤導)接受,且這是路徑 A 的主線洞,不是邊角。** 確認 `result['bank']` 僅在 `good_idx` 或分群 `normal_mask` 命中時才建(anomaly_tool.py:90-118);**大量 good 但 User 第一次跑、尚未框選任何 good = 走 LOO 無監督,bank=None**(95-103)。所以 `result.get('bank') is not None` 當存盤 enable 條件,會讓「我明明跑完了」的 User 看到 disabled 鈕。**修正**:(a) 存盤 expander 在 bank 為 None 時顯示明確引導「**先框選一團正常範例並重跑(或按『自動把最不可疑的 12 個標為正常』種子鈕,app.py:1406 已存在)讓 bank 生成**,才能存」;(b) 把風險 7 的一次性 `st.info` 觸發點從「bank 首次出現」改成「**confirmed good 首次出現**」更前面,否則 bank=None 時它根本不觸發、發現性反而最差。承認:這條我原設計把「框選一團即可」講得太輕,沒接上「框選≠bank 已生成,中間要重跑一次」的事實。

**A3(疊舊分佈底圖會與框選對齊衝突)接受,但機制要校正——校正後反而更好辦。** 批判說「plotly 框選回傳的 point index 跨 trace 連號 → 舊點被框會張冠李戴」。實測 `selection_points_to_indices`(interaction.py:704-718)**讀的是 `customdata` 不是 point index**;當 `customdata is None` 直接 `continue` 跳過(706-707)。所以真正的危險不是「index 連號」,而是「**舊 ref 背景 trace 若帶了會被誤解的 customdata**」。**修正(鎖死)**:舊 ref 背景 trace **一律不放 customdata、設 `hoverinfo='skip'`**(或獨立到不可套索的 trace)。如此即使被套索掃到也會在 704-718 被安全跳過,不會污染 `sel_idx`。這是 E2E 會炸的隱形坑,提案原本只說「疊一層底圖」沒鎖,確實是我的疏漏。校正後的好處:補救比批判想像的更簡單(不需動 `selection_points_to_indices`,只要 ref trace 不帶 customdata)。

**A4(coords2d 塞了不會生效)接受,這是我甩鍋給架構面的觸點,收回。** 確認 `result['coords2d']` 恆 None(64/139),散點圖**完全不讀它**,每次即時 `np.linalg.svd`(app.py:1300-1306)。「塞 coords2d 就生效」是假的。**修正**:把「**GUI 散點分支必須加 `if 掛載→讀 result['coords2d'] else→現況即時 SVD`**」明列為 `app._anomaly_ui` 的 **UX 觸點**(原文只在「投影」段含糊帶過,U-Net 模組表卻把它寫成「架構面回傳 coords2d 就夠」)。這是 UX 提案自己該鎖的座標來源切換,不該推給架構面。

## B. no-dataset-writes —— 接受(防呆要改成正向白名單 + 點明 picker 是新 widget)

**B5(路徑防呆不足)接受。** `dataset_cache_dir` 落 `.lv_cache/<name>_<sha>/`(object_eval.py:44-50);User 選的硬碟路徑可能是**資料集父層**(images/labels 在子層),只擋「== folder 或子目錄」擋不住父層混居。**修正**:改**正向白名單**——只允許 (a) `.lv_cache`(含 `LV_CACHE_DIR`),或 (b) 一個「**不含 images//labels/ 子目錄、且非任一已選 folder 之祖先或後代**」的乾淨硬碟資料夾;不符紅字擋存。對齊 no-dataset-writes 記憶。

**B6(缺單選 picker 原語)接受。** Grep 確認 `_folder_picker_single` **不存在**(另一份投影提案把它當既有物引用是錯的)。既有只有 `_folder_picker_list`(多選清單,201)與 `filedialog.askdirectory` 包裝(`_pick_folder` 130 等)。**修正**:設計明寫「bank 存/讀的『選單一資料夾』需**新增 widget**(復用 `askdirectory` 包一顆單選鈕,鏡像 `_pick_folder` 範式),不是現成的」,免得 PG 以為有現貨。

## C. 跨提案矛盾 —— 接受,且承認這是最該先收斂的單點(交還 PO)

**C7(三份提案檔名/欄位各說各話,且「對齊 save_profile」是空話)接受,這是最重的一條。** 確認 `save_profile(name, fingerprint, policy, metrics, ...)`(object_eval.py:164)**根本沒有 model/target_res/budget/n_patches** 欄位,我宣稱「欄位刻意對齊 save_profile」是空話——它對齊的只有「profile store 的**範式**(json + load/save/match 三件套)」,不是欄位。**修正立場**:(a) 收回「對齊 save_profile 欄位」措辭,改為「**鏡像 profile store 的存取範式**,欄位另立(瑕疵 bank 本就需要 model/res/budget 這些 profile 沒有的硬鎖欄位)」;(b) 明確標註「`vectors.npz`/`manifest.json`/`fewshot.json` 的精確檔名與欄位名,屬**架構/PG 層契約**,UX 不單方面定」。依 CLAUDE.md「粒度轉換點是唯一無紅綠保護的單點」,**三份提案的單一契約收斂應在 PO 拆 module 前完成**;UX 只鎖**語意觸點**(manifest 要能支撐狀態膠囊顯示 + model/res 相容硬鎖 + few-shot 內容定址),不鎖鍵名拼寫。取捨:我原想用「對齊既有範式」降落地風險,但用詞過頭變成假承諾,反而製造矛盾。

**C8(2D 底圖會在 PCA 壓扁時假陰,本版靜默選了較不準那條)接受。** 投影提案要在**原始 D 維**算 in_dense;本 UX 版只說「存少量 2D 代表點當灰底」,等於靜默用 2D 近似做「落在 good 密集區=正常」判定,前 2 主成分壓扁會假陰(投影提案風險 2 點名)。**修正並聲明取捨**:**底圖視覺用 2D(散點本就只能畫 2D),但紅/綠(正常/可疑)的判定來源用 D 維**(即顏色=來自 D 維 `score_object`/bank 距離的異常分數,不是「2D 上離群與否」)。如此散點顏色不會騙人,2D 只負責「擺位置」、不負責「定生死」。這點與投影面向對齊後寫進 UX 契約。

## D. 漏洞 —— 接受(heatmap 鎖死值、同步設定時序、few-shot 撞名)

**D9(掛載後 heatmap 用錯 model/res 會維度炸)接受,「heatmap 無需 UX 改動」是錯的。** 確認右欄 heatmap(app.py:1477-1484)用 `result['_model']/_target_res/_cache` 重跑 `embed_objects_patch` 再 `score_object(...,result['bank'])`。外部 bank 模式下,若這三者仍取 GUI 當前 widget 值、而 widget 停在與 bank 不符的 model → patch 維度與 bank 不合直接炸。**修正**:UX 契約明列「掛載外部 bank 時,`result['_model']/_target_res'` 必須寫 **bank manifest 的鎖死值**(非 widget 現值),`result['bank']` 填掛載的 bank」。如此 heatmap **碼確實沿用**,但前提是這幾個欄位被鎖死值覆寫——這是 UX 觸點,收回「完全無需改動」。

**D10(一鍵同步設定踩 Streamlit widget-key 時序)接受。** 想把側欄 model/res 改成 manifest 值要寫 `st.session_state['anomaly_model_sel']=...`,但該 key 已綁 `st.selectbox`(app.py:1213),widget 實例化後同回合改其 key 會被 Streamlit 拒/警告。**修正**:「一鍵同步」改為**在 widget 建立前的回合開頭設值 + `st.rerun()`**(callback/pending-pattern),設計明標此時序坑,不再「一筆帶過」。

**D11(few-shot 內容定址撞名誤配)接受。** 本版 fewshot.json 雖列 bbox,但讀回對齊只用 `(image_stem, obj_index)`;跨產線常見 `img_0001` 撞名 → 靜默把舊標記貼到不相干新物件。**修正**:對齊另兩份,讀回時**加 bbox 容差自檢(IoU/中心距閾值)**,不過就視為「對不到」計入「對齊 12/18」而非靜默配錯;`(stem,obj_index,bbox)` 三鍵一致才算命中。

---

## 我堅持不改的(給理由)

- **單一「執行偵測」+ 掛載狀態驅動雙行為,不新增第二主按鈕、不切 tab。** 批判也認同(keep)。理由:與既有唯一 run 路徑(app.py:1238 `run or pop('_anomaly_rerun')`)天然相容;大量 good/少數 bad 的 User 心智是「我就是要跑偵測」,bank 是修飾語不是平行功能。`anomaly_loaded_bank` 維持**唯一**新模式開關,未掛載時所有現有行為**零改動**——這是最小侵入的核心,守住。
- **三處冗餘提示(綠膠囊 / 按鈕文案變「對照已掛載 bank 偵測」/ 結果列明寫「對照 bank X」)防隱形模式切換。** 務實,保留。
- **few-shot 內容定址、預設落點 .lv_cache、相容性硬 gate(model+res 必相符才給掛載)** 方向不變(批判也認同),只是把 D11/B5 的自檢補強。

## 承認的整體取捨

1. 我原設計把「**正確的 UX 哲學**」和「**對既有碼的具體承諾**」混在一起寫,導致多處把「該由本提案鎖死的 UX 觸點」(A3 ref trace 不帶 customdata、A4 散點讀 coords2d 分支、D9 heatmap 鎖死值)甩給架構面。修正後這三個觸點明確回歸 `app._anomaly_ui` 的 UX 責任範圍。
2. 「對齊既有範式」是對的,但我用詞膨脹成「對齊 save_profile 欄位」這種可被一眼證偽的假承諾。改為「鏡像存取範式、欄位另立、鍵名拼寫交架構/PG 收斂」。
3. **D 維判定 vs 2D 視覺**的分工沒在原文聲明,等於默選了較不準的純 2D 路徑。現明確聲明:2D 只擺位、顏色(生死)來自 D 維。
4. 路徑 A 的「框選≠bank 已生成、中間要重跑」這個事實沒接好,讓存盤鈕 enable 邏輯與一次性提示都建立在錯誤前提上。已補正引導文案與觸發點。

紀律邊界不變:本文仍只是 UX/流程**設計輸入**,須走 /user→/po→/architect→/pm→/pg;檔案精確格式與 `run_pipeline` 的 `external_bank` 注入點屬架構/PG 層。三份提案的**單一契約收斂**請 PO 在拆 module 前於 ROADMAP 決策日誌記一行後定案——這是整條管線唯一無紅綠保護的收斂點,不該由任一 UX/投影/序列化提案單方面定。

## 修正:投影/視覺化:跨資料夾 fit/transform 投影器 + 「新資料映射回舊分佈」視覺

## 修正後最終立場(逐條回應批判,先認對的)

批判逐條成立 —— 我把六個漏洞都對著實碼重新查證過(app.py:1298-1306、1301 無 _l2norm;interaction.py:131-154 丟 index 且回 mean-of-k;object_eval.py:250 回 raw 向量;app.py 只有 `_pick_folder`/`_pick_folder_into_text` 無 `_folder_picker*`;`_cov_projection(...,token,cand_token)` 且 4617 強制 l2norm;anomaly_tool.py:78 obj_emb 在 if/elif/else 之前產出、各分支都有)。下面是哪裡改、哪裡守、為什麼。

---

### 漏洞 1(L2norm 幾何不一致)— **接受,改設計、寫死契約**
我犯了「兩句互相矛盾」的錯:不可能同時「`l2norm=True` 預設」又「ref_coords 與今日散點完全相同」。今日散點對 **raw** obj_emb 做 SVD(1301 確認無正規化),而 object_eval.embed_objects(250)回 raw DINOv2。

**決議(寫死,非 PG 自由度):`fit_projector` 預設 `l2norm=False`,對齊現況 raw 幾何。**
- 理由:選 raw 的成本最低且零附帶傷害 —— 不動既有 app 散點、ref_coords 真的等於 User 在情境 A 當下看到的圖、AC「ref_coords == 今日散點」可直接綠。
- 取捨(誠實承認):raw DINOv2 向量範數有量級差異,理論上 cosine/球面幾何對「相似度」更乾淨。但本設計的核心賣點是「新點落回 User 記得的那張舊圖」,**幾何一致性 > 幾何最優**。改 app 散點成 l2norm 會是個跨功能的視覺破壞性變更,不該由這個 feature 夾帶。
- AC 修正:刪掉模稜兩可的「完全沿用」字眼,改為兩條可機器驗的 AC:(a)`fit_projector(emb, l2norm=False)` 的 ref_coords 對同一 emb 等於 app.py:1301-1304 的 coords(數值 allclose);(b)`l2norm` 參數仍保留但**設計層鎖死預設 False**,若未來要切 l2norm,必須同一個 PR 連 app 散點一起改 —— 寫進設計「禁止只改一邊」。

### 漏洞 2(`compute_outlier_scores` 拿不到 ref_idx)— **接受,誠實標為新增函式**
我把連線端點的來源講成「零成本復用」是不準確的。`compute_outlier_scores`(151)`dist, _ = nn.kneighbors(...)` 丟掉 indices,且回 **mean-of-k**,給不了「最近單一鄰」的 index。

**決議:`anomaly_project` 新增一個小函式 `nearest_ref(emb_new, ref_emb) -> (idx, dist)`**(NearestNeighbors metric=cosine, n_neighbors=1, 回 indices)。誠實標為**新增**,不是復用。`in_dense` 的密度判定仍用 `compute_outlier_scores`(mean-of-k 適合密度);連線端點用新的 `nearest_ref`(單一最近)。兩者職責不同、本就該是兩個 call。成本:約 5 行 + 一條單元 AC(已知離群新點的 ref_idx 指向人工擺好的最近舊點)。

### 漏洞 3(in_dense 度量/tau 自比)— **接受,補 flag 與度量對齊契約**
這條會「默默錯」,最該補。決議三點寫進設計:
1. **度量統一 cosine**:`compute_outlier_scores` 是 sklearn cosine;故 `in_dense` 路徑與 fit 的幾何都走 cosine,且因漏洞 1 已定 `l2norm=False`,raw 向量在 cosine 下範數被正規化掉、tau 可比。明寫「dense_metric 固定 cosine,不受 l2norm 開關影響」。
2. **tau 必須帶 `candidates_in_reference=True`**:fit 時對「舊 good 對舊 good」算 kNN 求 95 分位,**必須去自身**(否則 self-dist=0 拉低分位 → 新點普遍誤判離群)。這條進 AC:tau 用 `candidates_in_reference=True` 算,且有一個回歸測試斷言「不帶 flag 時 tau 明顯偏低」以鎖死。
3. tau 與 dense_metric 一起存 json。

### 漏洞 4(picker 名稱對不上)— **接受,更正為實際 helper**
無 `_folder_picker_single`。**改為 `_pick_folder('anomaly_proj_path')`(把目錄寫進 session_key)+ 自繪 text_input 顯示/編輯路徑**,與既有 `_pick_folder_into_text`(163)的慣例一致。純文字更正,不影響架構。

### 漏洞 5(落點循環依賴/職責真空)— **接受,這是真矛盾,收斂如下**
我同時主張「投影器存盤獨立於 bank、可只存投影器」又「落點由持久化面向給 dir(C2)」,當 User 只存投影器、bank dir 還不存在 → 真空。

**決議:投影器的落點策略改由本面向自負,不再依賴持久化面向給 dir。**
- 落點 = `object_eval.dataset_cache_dir(old_folder, "anomaly_proj_<model>_r<res>")`(沿用既有 cache 慣例,天然 model/res 分流、守 no-dataset-writes)。
- **撤回 C2 的「投影器不自己決定路徑」**。改為弱契約 C2':**若**持久化面向也落在 `dataset_cache_dir(old_folder, ...)` 同 parent,投影器與 bank 自然同 parent dir、同進退;但投影器**不要求** bank 先存在。這樣「解耦可獨立存」的 UX 賣點得以保留,矛盾消除。
- 取捨:投影器因此「知道一點路徑策略」(違反我原本想要的純消費者潔癖),但這是消除真空的必要代價,且 `dataset_cache_dir` 本就是全 app 共用的既有公約,不算另起爐灶。

### 漏洞 6(umap 第二按鈕的 token 契約)— **接受,補 token 並標註語義差異**
`_cov_projection` 簽章需 `(token, cand_token)`(4600)且內部強制 `_l2norm`(4617),與 PCA 路徑的 l2norm 開關打架。

**決議:umap 第二按鈕呼叫改為完整簽章** `_cov_projection(ref_emb, new_emb, "umap", 2, token="anomaly_obj", cand_token="anomaly_new")`,明確傳 token 避免與整圖/物件級快取互撞。並在按鈕旁註明:「此路徑強制 l2norm 且為**合併重擬合**(舊座標會動),與上面 PCA『投進固定舊基底』語義不同」。撤回「直接複用、這麼直接」的輕率措辭。

---

### 守住不改的(給 keep 依據,批判也認同)
- **transform-into-fixed-basis vs re-fit-together 的區別**:這是本設計最有價值且批判明確背書的洞見。`_cov_projection`(4616 vstack 後一起 fit)會讓舊座標跟著新資料動,破壞「映射回固定舊分佈」語義。PCA persist basis + `(emb_new-mean)@components.T` 是唯一能對新點做**真 transform** 的路。**守。**
- **v1 只支援 PCA**:umap/tsne 無穩定 out-of-sample transform,硬做近似會誤導。守(umap 只當「合併重投影」第二視角,語義誠實分開)。
- **in_dense 用 D 維 kNN 而非 2D 座標**:2D 投影會把離群點壓進 good 群(假陰),D 維判定 + 2D 只負責畫在哪 + hover 給 D 維真實距離。守。
- **fp16 + seed=42 確定性下採樣畫灰底、components/mean 不抽樣**。守。
- **投影器獨立於 bank、LOO 模式也有 obj_emb**(anomaly_tool.py:78 已驗,各分支都產 obj_emb)→ 風險 5 結論成立。守。

---

### 漏洞 7 + coords2d 之爭 —— 跨面向衝突,**升 reverse-gate 回 PO 仲裁**
我查了磁碟:`3_Architect_Design/` 只有 01-06,**持久化/投影器設計都還沒落盤**,兩份都還是提案。批判正確:我自稱「刻意不重複設計序列化」卻又設計了 `projector.npz`/`save_projector`/`load_projector` —— **言行不一**。而持久化面向把 proj_mean/proj_vt 塞進 `bank.npz`,兩份搶同一塊序列化 + 搶 `coords2d` 欄位(anomaly_tool.py:139 現恆 None)。

**這不是我能單方在本面向決掉的,落地會撞。決議:停手,reverse-gate `/architect → /po`,在 ROADMAP 決策日誌記一行,請 PO 仲裁兩件事:**
1. **投影 basis 的序列化歸誰**:我的建議是 **歸本面向的獨立 `projector.json`+`projector.npz`,持久化面向不存 proj_***。理由:投影用 obj_emb(物件級)、bank 用 patch(patch 級),硬綁進 bank.npz 會混兩個層級、且「只存投影器不存 bank」的 UX(漏洞 5 已保住)要求兩者能獨立落盤。但我**承認這是建議不是裁決**,接受 PO 反向裁定(若 PO 判歸 bank.npz,我就退成純消費者、刪掉 save/load_projector,信守原本的「不重複序列化」承諾 —— 那反而讓我的言行一致)。
2. **coords2d 欄位**:建議本面向用 `result["_proj_new"]` 不碰 coords2d,把 coords2d 讓給持久化面向;若 PO 判給我,改塞 coords2d。任一裁定我都能接,只要單一所有者明確。

**承認的整體取捨**:本次批判讓我從「自稱純視覺化消費者」退守到「必須承擔投影器落點策略(漏洞5)、必須新增 nearest_ref 函式(漏洞2)、且序列化歸屬要 PO 裁(漏洞7)」—— 我原本的「潔癖式零成本消費者」定位部分破產。但核心技術主張(固定基底真 transform、D 維判定、PCA-only、l2norm=False 對齊現況)經查證全部站得住,且把 l2norm 從含糊改成寫死、把 tau 自比 flag 補上,設計反而更可機器驗、更不會 false-green。

## 修正:整合 / 效能 / 風險(bank 持久化 + 新資料投影進舊分佈)

## 回應批判:逐條認賬與最終立場

我已逐行重讀 anomaly_tool.py / anomaly_score.py / patch_features.py / object_eval.py / interaction.py / app.py 相關段,**批判的每一條 load-bearing 事實都成立**,且其中 A1、A3、B4、B6 是會讓 PG 落地時產生 silent-wrong 的真漏洞,不是措辭問題。原提案方向(external_bank 短路、float16 npz、atomic replace、model/res 硬校驗、fit/transform 分離 vs re-fit-together)我維持;但下列七處必須改進設計與 AC,否則不該放行 PG。

---

### A. 事實錯誤 —— 全部接受,改設計+補 AC

**A1(投影幾何:不可加 l2norm)—— 接受,這是最該補的 AC。**
已查 app.py:1301-1304:`c = emb - emb.mean(0)` 後直接 `np.linalg.svd(c)`,**全程無 `_l2n`**;`embed_objects`(object_eval:250)存 `embed_fn(crop)` 原始向量(`_l2n` 在 :265 只被 `score_embeddings` 用,embed_objects 不碰)。原提案沒寫 l2norm 是「沒寫錯」但「沒鎖死」——這正是漏洞。**最終立場:`fit_projection`/`transform` 一律對 raw obj_emb(不 l2norm)操作,AC 鎖死「transform 對當批資料的座標 == 現行散點 self-fit 座標(符號/正負軸可差,用 |cos| 或 procrustes 容忍)」**。
但批判點出一個我原文自相矛盾處必須處理:我引用 `interaction.reference_coverage`/`compute_outlier_scores` 來量「新點落舊 good 密集區」——那是 cosine metric(D 維 kNN)。**投影座標(raw-SVD)與密度判定(cosine)兩套幾何不一致**。取捨後的決策:**需求(3)的「視覺落點」與「密集度數字指標」分屬兩個獨立信號,不強行統一幾何**——散點用 raw-SVD 投影(與既有散點逐點可疊,這是硬約束),密度/覆蓋指標用 cosine kNN(與 bank 評分同族)。設計文案必須明說「兩套幾何刻意不同、各自的意義」,AC 分別守:投影座標一致性測 raw-SVD,覆蓋指標測 cosine。這承認一個取捨:散點上「看起來離群」與「cosine 覆蓋數字離群」可能不完全對齊,UX 文案要誠實標註此為「兩個視角」而非「一個真相」。

**A2(`match_shas_to_indices`/`curation_log_csv` 不可復用)—— 接受,刪除誤導性「復用」措辭。**
interaction.py:179-193 簽章 `(shas, sha_to_index)` 是純 dict 查表、鍵為 sha256;curation_log_csv header 是 `ts/reason/n/filenames/sha256s`。與我的 fewshot 內容定址(`stem+obj_index+bbox`)是**不同的鍵與 schema**。**最終立場:設計改寫為「新寫一個 `match_fewshot_to_indices(fewshot_rows, meta)` 純函式,語義上『風格』類似 match_shas(graceful skip、order-preserved、去重),但不是呼叫它」**。移除「對應既有範式」這欄的裝飾性宣稱,避免 PM 以為照抄。這是把「靈感來源」誠實降級為「靈感來源」,不假裝有現成函式。

**A3(內容定址鍵碰撞)—— 接受,補一條 AC。**
`discover_yolo_objects`(interaction:1192)的 obj_index = 單圖內 box 的 `enumerate` 序號 k,非全域唯一;跨資料夾同名 `IMG_0001.jpg` 或 box 順序變動會誤配。**最終立場:fewshot 回載的對齊鍵不只 `{stem, obj_index}`,必須帶 `bbox(4 float)` 與 `class_id` 做自檢**;新增 AC:「同名 stem 但 bbox(IoU 容忍)或 class_id 不符 → skip 且可觀測(回報未命中數),不可靜默回填錯 index」。這與另兩份提案的 bbox+class_id 自檢收斂一致。

---

### B. 整合漏洞 —— 全部接受,B4/B6 是最關鍵落地缺口

**B4(patch 快取目錄吃到錯 res 的 stale npz)—— 接受,這是本提案最關鍵的補強。**
patch_features:80 快取鍵 `{stem}__{obj_index}.npz` **不含 model/res**;分流全靠 `cache_dir` 名(app.py 組 `anomaly_patch_{score_mode}_{model}_r{target_res}`)。我原文「快取自動分流 model/res」**只在 cache_dir 用 bank.meta 的 model/res 組名時才成立**。**最終立場:新增硬 AC ——「external_bank 模式下,`cache_dir` 名一律由 `meta['model']`/`meta['target_res']` 決定,GUI widget 值絕不參與組名」**。這比「disable widget」更根本:即使 widget 被 disable,只要組 cache_dir 的程式碼讀的是 widget 變數而非 meta,就會 silent-wrong。AC 要 spy `embed_objects_patch` 收到的 `cache_dir` 參數,斷言其路徑含 meta 的 model/res。這條接上了我風險段第 2 點(res silent-wrong)原本懸空的快取路徑。

**B5(objemb 快取只 key by model 不含 res)—— 接受,改為「明寫刻意」。**
anomaly_tool:77 `objemb_{model}.npz` 跨 score_mode/res 共用(註解明載)。**最終立場:設計明寫「投影 basis 來自 obj_emb(整體向量,crop→resize 流程相同),對 res 不敏感 → 快取不需 res 分流;patch 評分對 res 敏感 → 必須 res 分流。兩者快取策略不同是刻意設計,非疏漏」**。消除 PG「為何 patch 分流、obj 不分流」的困惑。取捨:若未來 obj_emb 的 resize 行為改成 res 相關,這個假設會破,設計留一行 backlog 註記。

**B6(external_bank 模式 normal_set 仍被計算)—— 接受,補旁路 AC。**
anomaly_tool:87-93 的 normal_set 決策在 :118 build_memory_bank **之前**就跑;只在 :118 短路會留下「新資料夾仍算 confirmed/分群 normal_set」的浪費與語義混亂。**最終立場:`if external_bank` 短路點要前移到 normal_set 之前,或明確「external_bank 模式 normal_set 計算被旁路/忽略」**;新增 AC:「external_bank 模式下,normal_set 不參與評分(spy 斷言不進入 normal_set-based 分支),且 good few-shot 不建 bank」。原「good few-shot 不建 bank」這條 AC 不足以涵蓋 in-pipeline 分支點,補強。對應 UX 提案「掛載後正常範例 UI 唯讀」,我這版補上 pipeline 內部的對應旁路。

---

### C. 跨提案矛盾 —— 接受 PO 收斂方向,放棄我的 csv

**C7(落盤格式三份打架)—— 接受,改 json,reject 我自己的 csv。**
既有可比範式 `object_policy_profiles.json` 是 json;bbox(4 float 陣列)、跨資料夾可選欄位(sha 可選)用 json 較穩,csv 表達 bbox 陣列彆扭。**最終立場:fewshot 改 `fewshot.json`,向另兩份提案收斂**。檔名與另兩份仍有歧異(我:`bank.npz/meta.json/projection.npz/fewshot.json`;序列化提案:`bank_meta.json`;UX:`vectors.npz/manifest.json`)——**這是必須由 PO 在 ROADMAP 拍板的單點,我不單方面定檔名**,只表態:格式全 json(除向量 npz)、meta 用 `meta.json`(對齊 object_eval profile store 慣例)。承認取捨:放棄 csv 的「人眼可 grep」便利,換 schema 一致性與跨提案收斂。

**C8(投影底圖存 2D 還是 D 維)—— 接受,改存 D 維 fp16。**
我原文存 `old_good_coords`(2D ≤2000)與我自己引用 `reference_coverage`(cosine,D 維)矛盾:**只存 2D 無法在 D 維算「新點落舊 good 密集區」**,2D 距離失真(線性 PCA 壓離群點,我風險 2 自己也承認)。**最終立場:profile 存舊 good 的 D 維 obj_emb(fp16,抽樣上限可設,如 ≤2000 物件),底圖 2D 座標由讀回後用 `transform` 即時算**(已有固定 basis,免另存 2D)。密集區判定走 D 維 cosine,視覺底圖走 transform 後的 2D,兩者來源同一份 D 維資料,不再各說各話。取捨:profile 體積增加(D 維 fp16 vs 2D),但這是需求(3)「密集 vs 離群」正確性的必要成本;抽樣上限控制體積。

---

### D. 次要 —— 接受

- 「vectors bit 級一致(float16 容忍)」措辭矛盾 → 改 **「存→讀→評分 cosine 距離差 < 1e-3」**(行為一致性,非 bit 一致)。
- seed=42 寫死、存的是抽樣後 vectors、meta 記 seed 只為追溯不為重抽 —— 批判確認我原文說對,keep。
- **atomic 寫盤順序明寫**:`bank.npz` + `projection.npz` + `fewshot.json` 先寫進 tmp 目錄 → 全部就緒後 **`meta.json` 最後 `os.replace`** 當完整性 sentinel;`load` 先檢 `meta.json` 存在且 `format_version` 相容才讀其餘。否則「meta 在但 projection 半截」仍會炸。補進 anomaly_bank_store 的 AC。
- spy/mock 斷言「build_memory_bank 沒被呼叫」—— keep,這是「跳過建 bank」的核心行為證據。

---

### 守住不改的(給理由)

1. **external_bank 短路 + score_object/MemoryBank 不動** —— 最小侵入,批判未質疑此方向,且讀碼確認 :118 前加分支可行。keep。
2. **fit/transform 分離,不用 _cov 的 re-fit-together** —— 需求(3)明確要「固定舊基底、新點映射進去」,re-fit 每次舊座標都變,做不到。批判未反對。keep。
3. **此版不換 greedy coreset(純隨機 + budget/seed 寫 meta)** —— appetite 控制,純隨機是既有行為,留 backlog。批判確認合理。keep。
4. **bank D 鎖死於 model、model/res 硬校驗 + 紅字擋下** —— 批判點名這是對的核心,且 B4 的補強讓「res silent-wrong」從「靠 disable widget」升級為「靠 cache_dir 組名來源」雙保險。keep 並強化。

---

### 對 U-Net 模組與 PO 的影響(更新)

- `anomaly_projection`(Tier A,先做):AC 補「raw obj_emb 不 l2norm、與 app.py:1301-1304 self-fit 座標一致」。
- `anomaly_external_bank`:AC 補「normal_set 旁路(B6)」「cache_dir 組名由 meta 決定(B4)」「存讀評分 cosine 差<1e-3」。
- `anomaly_bank_store`:格式改 json(C7)、存 D 維 good fp16(C8)、atomic meta-last 順序(D)、fewshot 帶 bbox+class_id 自檢(A3)、新寫 `match_fewshot_to_indices`(A2,非復用)。
- `anomaly_bank_ui`:E2E 斷言不變,但加「同名跨資料夾 fewshot 誤命中被 skip 且可觀測」。
- **交 PO 拍板的跨提案單點**:profile 目錄結構與檔名統一(三份歧異),我表態全 json + D 維底圖,但檔名由 PO 收斂。

**承認的總取捨**:這次修正讓 profile 變大(D 維 good)、設計更囉嗦(兩套幾何要明說、快取分流策略要明寫),但換來的是消除四個 silent-wrong(res 快取污染、normal_set 殘留、投影加錯 l2norm、fewshot 同名誤命中)——這些正是「分數全錯卻無例外」的最危險類別,值得用設計複雜度換正確性可觀測性。reverse gate 立場不變:若 PG 落地發現 obj_emb(投影)與 patch bank(評分)維度語義在某 model 變體下無法調和,停手回報 /architect。


---
# 五、綜合收斂(首席架構師)

## 推薦整體設計:跨資料夾 Memory Bank 持久化 + 投影回舊分佈

### 一句話定調
新增「**bank profile = 一個目錄**」可攜資產,讓 User 第一次在大量 good 資料夾建好 memory bank + 投影基底 + few-shot,凍結到硬碟;第二次選新資料夾把它讀回,用**固定舊基底 transform** 新點、用**外部 bank 評分**,一眼看出新點落在舊 good 密集區(正常)還是離群(可疑)。所有實作走既有純函式 + 既有 `.lv_cache` 慣例,`run_pipeline` 只加可選參數、`MemoryBank` 不動。

我已逐行核對 `anomaly_tool.py / anomaly_score.py / patch_features.py / object_eval.py / interaction.py / app.py`,三輪辯論中的 load-bearing 事實全部成立,以下是收斂後的最終裁決(含我對四個跨提案爭點的拍板)。

---

### A. 跨資料夾投影:最終做法與理由(這是需求(3)的核心,也是現況零基礎處)

**做法:transform-into-fixed-basis(PCA via SVD),`l2norm=False`,投影用 obj_emb(物件級)。**

1. **Fit(建 bank 那次,對舊資料 obj_emb 算一次)**:完全沿用 `app.py:1301-1304` 的數學 —— `c = emb - emb.mean(0)`;`_u,_s,_vt = svd(c)`;`components = _vt[:2]`、`mean = emb.mean(0)`、`ref_coords = _u[:,:2]*_s[:2]`。差別只是**把今天算了就丟的 `_vt`/`mean` 存下來**。
2. **Transform(讀回那次,對新資料 obj_emb)**:`coords_new = (emb_new - mean) @ components.T` → `(N_new, 2)`。**新點被投到舊資料定義的同一座標軸**,不再各自 svd,這才是「映射**回**舊分佈」。
3. **為何不用 `_cov_projection`**:它是 vstack 後 re-fit-together(app.py:4616),每次新資料進來舊座標會跟著動,無法「對照固定的舊分佈」。這是經三輪辯論一致背書的核心洞見,守住。umap/tsne 沒有穩定 out-of-sample transform → **v1 只支援 PCA**;若 User 要 umap 視角,額外提供「合併重投影」第二按鈕(明確標註「座標會動、非投進固定基底」),呼叫 `_cov_projection(ref, new, "umap", 2, token="anomaly_obj", cand_token="anomaly_new")` 完整簽章。

**裁決 1 —— `l2norm=False`(寫死,非 PG 自由度)**:`app.py:1301` 對 raw obj_emb 做 SVD,`embed_objects`(object_eval.py:190)回 raw 向量(無 `_l2n`)。「投影/視覺化」提案原本預設 `l2norm=True` 並宣稱「與今日散點完全相同」是**自相矛盾**(加了 l2norm 座標就不同)。最終取「raw 對齊現況」:成本最低、ref_coords 真的等於 User 在情境 A 看到的圖、AC 可機器驗「transform 對同批資料 == 現行散點 self-fit(符號可差,用 |cos|/procrustes 容忍)」。若未來要切 l2norm,**必須同一 PR 連 app 散點一起改**,設計禁止只改一邊。

**裁決 2 —— 密度判定(密集區 vs 離群)用 D 維,不用 2D 座標**:2D 投影會把離群點壓進 good 群(假陰)。`in_dense` 用**原始 D 維** kNN 算,2D 只負責「畫在哪」。承認此處有**兩套幾何刻意並存**:投影座標走 raw-SVD(與既有散點逐點可疊),密度/覆蓋指標走 cosine kNN(`compute_outlier_scores`,與 bank 評分同族)。UX 文案誠實標註「散點位置」與「離群數字」是兩個視角而非一個真相;散點顏色(紅/綠生死)來源用 D 維異常分數,不用 2D 上的離群與否,顏色才不會騙人。

**裁決 3 —— 連線端點需新寫函式**:`compute_outlier_scores` 丟掉 indices(interaction.py:151 `dist, _`)且回 mean-of-k,給不了「最近單一鄰」的 index。最近鄰連線(可疑新點→最像哪個舊點)需 `anomaly_project` 內**新增** `nearest_ref(emb_new, ref_emb)->(idx,dist)`(NearestNeighbors metric=cosine, k=1),誠實標為新增、非復用。tau(密集區門檻)在 fit 時對「舊 good 對舊 good」算 kNN 95 分位,**必須帶 `candidates_in_reference=True` 去自身**(否則 self-dist=0 拉低分位 → 新點普遍誤判離群),這條進 AC。

---

### B. bank + few-shot 硬碟格式 / 位置 / 相容鍵的最終決定

**裁決 4 —— 一個 bank profile = 一個目錄,四個檔,全 JSON(向量除外),meta 最後寫當 sentinel。**

```
<bank_dir>/                  # 預設 dataset_cache_dir(src_folder, "anomaly_bank/<name>")
  bank.npz                   # patch coreset 向量(評分用)
  projection.npz             # 投影基底 + 舊 good 特徵(投影/密度用)
  fewshot.json               # few-shot good/bad 標記(內容定址)
  meta.json                  # 相容鍵 + 來源指紋(最後寫,存在=profile 完整)
```

收斂三提案的格式打架(`bank.npz/meta.json/projection.npz/fewshot.{json,csv}` vs `vectors.npz/manifest.json` vs `bank_meta.json`):**全 JSON(對齊既有 `object_policy_profiles.json` profile store 慣例),fewshot 用 json 不用 csv**(bbox 四 float 陣列 + 可選欄位 json 較自然),meta 檔名 `meta.json`,向量檔 `bank.npz`、投影檔 `projection.npz`。

**`bank.npz`(patch 級,評分用)**:`vectors` = `MemoryBank.vectors`(已 L2 正規化)。**裁決 5 —— 預設 fp32 存**(正確性 > 省空間;coreset 已被 budget≤100k×D 上限,`savez_compressed` 後可控);`vectors_dtype` 寫進 meta,fp16 降為可選旗標。讀回 `MemoryBank(vectors)` 重建,**不存 hnswlib**(size>8192 時 lazy 重建)。空 bank(0 列)→ 拒存。

**`projection.npz`(obj 級,投影/密度用)**:`proj_mean (D,)`、`proj_components (2,D)`、`good_obj_emb (G,D) fp16`(舊 good 的 D 維特徵,密度判定必需;抽樣上限 ≤2000)、`tau (float)`、`ref_coords_2d` 抽樣(畫灰底,可由 transform 即時算或抽樣存)。

**裁決 6 —— 移除 `patch_grid` 硬鍵(最高危的設計錯)**:`_resize_keep_ar_mult14`(models.py)對 nw/nh **各自** snap 到 14 倍數,grid 逐物件變動,bank 由多長寬比物件 vstack 而成,**根本沒有單一 grid 可寫**。當硬鍵會把所有非正方形資料集全擋掉。bank 是 flat (M,D),`nn_dist` 只做 `q@vectors.T` 是 grid-agnostic;grid 在 query 端各自帶。整條移除。

**裁決 7 —— 拆 bank.npz 與 projection.npz(不同檔)**:patch 走 head=meanpool、obj 走 head=cls,維度同 384 僅 vits14 巧合,語義不同層級,同檔混存埋雷。拆檔後 object 模式(無 patch bank)可只存 projection,不需假 bank。meta 分記 `patch_dim` 與 `obj_dim`。

**`meta.json` 相容鍵**:
- **硬鍵(讀回必須相等,否則拒載 + 明確訊息,絕不跑到 matmul/broadcasting)**:`schema_version`、`model`、`target_res`、`patch_dim`、`obj_dim`。理由:`model` 決定 patch D 與 obj D_obj(`q@vectors.T` / `(emb-mean)` 維度不符直接炸);`target_res` 不改 D 但改 patch 物理像素尺度 → 特徵分佈位移 → 距離**靜默錯**(silent-wrong,比 crash 更危險,優先擋)。`check_compat` 額外驗 `obj_emb_new.shape[1]==proj_mean.shape[0]`(維度不符是 broadcasting ValueError,非 LinAlgError,`eye(2,D)` 後備救不了)。
- **軟鍵(僅警告/追溯,絕非相容鍵)**:`budget`、`coreset_seed`、`source_fingerprint`、`created`、`tool_version`、`folder_sha1`(bank 可能離開 repo,sha 對讀回端無意義)。
- **過期偵測**:用 `dataset_fingerprint` + `fingerprint_similarity` 比新資料夾 vs `meta.source_fingerprint`;低於門檻**只警告不擋**(需求(2)本就是拿舊 bank 看不一樣的新資料,差異大正是要看的訊號)。

**裁決 8 —— few-shot 內容定址主鍵 = `(image_stem, 量化 bbox, class_id)`,obj_index 僅 tie-break**:`obj_index` 來自 YOLO label 行序(interaction.py:1192),重標註即可能變。bbox 比對定**量化容忍**(歸一化座標 IoU 或四捨五入位數);對齊邏輯「先用量化 bbox 反查正確 new_index」而非只 skip;同名 stem 但 bbox/class_id 不符 → skip 且可觀測(回報未命中數,不靜默回填錯 index)。新寫 `match_fewshot_to_indices`(語義借 `match_shas_to_indices` 風格,**非呼叫它**——鍵不同)。

**裁決 9 —— 容忍值釘單一數**:預設 fp32 → 存讀評分 cosine 距離差 `< 1e-5`;選 fp16 → 用實測上界(PM 寫 AC 前先跑微基準量真實漂移,不拍腦袋)。寫進 ROADMAP 決策日誌讓四模組不各寫一套。

**位置(鐵則)**:預設落點 `dataset_cache_dir(src_folder, "anomaly_bank/<name>")`(`.lv_cache`,`LV_CACHE_DIR` 可覆寫);User 可在 widget 明確指定硬碟目錄。**正向白名單防呆**:只允許 (a) `.lv_cache`,或 (b) 不含 `images/`、`labels/` 子目錄且非任一已選 folder 之祖先/後代的乾淨目錄;不符紅字擋存。絕不回寫資料集。原子寫:bank/projection/fewshot 先寫 tmp → 全就緒後 `meta.json` 最後 `os.replace`;load 先檢 meta 存在且 schema 相容;壞檔/缺檔拋明確例外不靜默回空 bank。

**裁決 10 —— external_bank 注入 + 兩條快取路徑切開**:`run_pipeline(..., external_bank=None)` 新增可選參數,給定且 patch 模式時**短路點前移到 normal_set 計算之前**(不只在 build_memory_bank 前短路),直接用外部 bank 走 `score_object`。**新資料夾 patch cache_dir 名一律由 `meta['model']/meta['target_res']` 決定,GUI widget 值絕不參與組名**(否則寫/讀錯 res 的 stale npz → 分數全錯不報錯,這是最隱蔽的 silent-wrong)。掛載 bank 時 `result['_model']/_target_res'/_bank` 寫 bank 鎖死值,右欄 heatmap 才不會用 widget 現值重算 patch 而維度炸。

---

### C. MVP(最小可用、可獨立出貨)

**MVP = 需求(1)存 + 需求(2)讀評分 + 需求(3)投影,但砍掉所有 nice-to-have:**
- **保留**:bank.npz(fp32)+ projection.npz + meta.json 存讀;`external_bank` 評分路徑;固定基底 transform 畫「灰底舊分佈 + 彩色新點」;model/res 硬鍵擋載;預設落點 .lv_cache + 白名單防呆。
- **MVP 砍掉(進 backlog)**:fewshot.json 跨資料夾沿用(MVP few-shot 仍只服務建 bank 那次,session 內);最近鄰連線視覺;umap 第二按鈕;greedy coreset(維持純隨機 budget=100000);fingerprint stale 警告(MVP 先不算)。
- **MVP 的單一新主開關** = `anomaly_loaded_bank` session key;**未掛載時所有現有行為零改動**。不新增第二主按鈕、不切 tab,沿用唯一「▶ 執行偵測」,掛載狀態驅動雙行為(膠囊綠 + 按鈕文案變「對照已掛載 bank 偵測」+ 結果列冗餘提示三處防隱形切換)。

---

### D. 里程碑

- **M0(契約收斂,PO)**:在 ROADMAP 決策日誌釘死 profile 目錄結構/檔名/相容鍵/fewshot 格式/容忍值(裁決 4-9),這是整條管線唯一無紅綠保護的單點。`verify/gate.py --snapshot` 重建 baseline。
- **M1(`anomaly_project`,Tier A→B)**:fit/transform/nearest_ref/save/load 純函式綠。AC:transform 對同批 == 現散點(|cos|);維度不符 raise;tau 帶 去自身 flag;l2norm=False 鎖死。
- **M2(`anomaly_bank_store`,Tier B)**:存讀 round-trip vectors cosine 一致;meta 硬鍵齊;atomic meta-last;空 bank 拒存;只寫 .lv_cache/指定路徑不寫資料集;fewshot bbox 自檢。
- **M3(`anomaly_external_bank`,擴 anomaly_tool,Tier B)**:external_bank 給定 → spy 斷言 build_memory_bank 未被呼叫、normal_set 旁路;cache_dir 組名由 meta 決定(spy 參數);存讀評分 cosine 差 < 1e-5。
- **M4(`anomaly_bank_ui`,擴 app.py,Tier B,E2E 收尾)**:存 bank→換資料夾載入→新點真的投影到舊基底正確象限、已知離群新物件真被標紅/排前、heatmap 仍可疊、不相容 bank 紅字擋下。`tests/e2e/test_anomaly_bank_ui_e2e.py` 標 @e2e,行為斷言非 element 存在。
- **相依順序**:M1 → M3 → M2 → M4(M2 依前兩者產物格式)。

### 守住的鐵則
絕不寫使用者資料集(.lv_cache 或 User 指定路徑 + 白名單);U-Net 規格疊加(設計只出 Markdown+AC、實作進 scripts/、測試進 tests/);PG 嚴禁改 tests/契約/基礎設施;IO 模組 Tier B,done=單元綠 AND 真實 E2E 綠;reverse gate:若 PG 落地發現 obj_emb(投影)與 patch bank(評分)維度語義在某 model 變體無法調和,停手回報 /architect。


## U-Net 模組分解

- **[Tier A → 自動升 B(加 npz/json IO 後)] anomaly_project** — 跨資料夾投影器純邏輯。fit_projector(obj_emb, good_mask, *, l2norm=False, dim_out=2)→ProjectorBasis(mean/components=_vt[:2]/ref_coords/good_obj_emb D維fp16/tau);transform_new(basis, emb_new)→coords=(emb_new-mean)@components.T(固定舊基底真 transform,不 re-fit);nearest_ref(emb_new, ref_emb)→(idx,dist)(新增,連線端點,compute_outlier_scores 給不了 index);classify_dense(emb_new, good_obj_emb, tau)→in_dense(D維 cosine kNN,tau 帶 candidates_in_reference=True 去自身);save_projector/load_projector(projection.npz+json,atomic);check_compat(維度/model/res 不符 raise)。l2norm=False 鎖死對齊 app.py:1301 現況散點。
- **[Tier B(IO)] anomaly_bank_store** — bank profile 目錄序列化/反序列化。save_bank(bank_dir,*,vectors,projection_arrays,meta,fewshot)→atomic 寫 bank.npz(預設 fp32)/projection.npz/fewshot.json,meta.json 最後寫當完整性 sentinel;load_bank(bank_dir)→LoadedBank{MemoryBank, meta, proj_*, good_obj_emb, fewshot}(壞檔/缺檔拋明確例外不回空 bank);check_compat(meta, model, target_res, patch_dim, obj_dim)→硬鍵(model/res/dim)不符即拒載;check_stale(meta, new_fingerprint)→similarity(只警告不擋);confirmed_to_fewshot(confirmed, records)→內容定址(stem+量化bbox+class_id);match_fewshot_to_indices(fewshot, new_meta)→對齊(bbox反查救回行序變動,自檢不符 skip 且可觀測;新寫非復用 match_shas)。落點預設 dataset_cache_dir + 正向白名單防呆,絕不寫資料集。
- **[Tier A→B(擴既有 anomaly_tool.run_pipeline)] anomaly_external_bank** — run_pipeline 加可選 external_bank 參數。給定且 patch 模式時:短路點前移到 normal_set 計算之前(spy 斷言 build_memory_bank 未被呼叫、normal_set 旁路),直接用外部 bank 逐物件 score_object;新資料夾 patch cache_dir 名一律由 meta['model']/meta['target_res'] 決定(非 GUI widget 值,擋 res silent-wrong);bad few-shot 仍進 classify 算門檻/AUROC,good few-shot 不建 bank;obj_emb 仍算供投影。簽章只加可選參數不破壞既有呼叫;MemoryBank/score_object 完全不動。AC:存讀評分 cosine 差 < 1e-5。
- **[Tier B(GUI,E2E 收尾)] anomaly_bank_ui** — scripts/app.py _anomaly_ui 擴充。側欄 expander(狀態膠囊/存 bank/讀 bank,新增單選資料夾 widget=_pick_folder 包一顆鈕,非不存在的 _folder_picker_single);唯一新模式開關 anomaly_loaded_bank,未掛載零改動;掛載 model/res 硬 gate + 一鍵同步設定(widget 建立前設值 + rerun 避時序坑);散點分支加 if 掛載→讀投影座標 else→現況即時 SVD;跨資料夾三層 trace(舊 ref 灰底底圖一律不帶 customdata+hoverinfo=skip 避框選錯位、彩色新點、可選離群連線),顏色生死來自 D維 in_dense 非 2D;掛載時 result['_model']/_target_res'/_bank 寫 bank 鎖死值使右欄 heatmap 沿用既有碼不炸;按鈕文案+結果列冗餘提示防隱形切換。E2E 行為斷言非 element 存在。


## 待拍板的開放決策點(原始)

1. 檔名/欄位最終拼寫(bank.npz/projection.npz/meta.json/fewshot.json 與各欄位名)需 PO 在拆 module 前於 ROADMAP 決策日誌一行釘死 —— 這是三提案歧異、且 CLAUDE.md 明示的唯一無紅綠保護收斂單點;我已表態全 JSON + 拆 bank/projection 雙檔 + meta.json 慣例,但須 PO 拍板。
2. vectors 預設 dtype:我裁 fp32(正確性優先,cosine 差<1e-5),fp16 降為可選旗標。是否接受 bank 體積翻倍(~150MB/100k×384)換取無「存讀分數漂移」爭議?或 PO 判定可接受 fp16 + 實測容忍值(需 PM 先跑微基準)?
3. 投影底圖存舊 good 的 D 維特徵(good_obj_emb fp16)會把舊資料特徵帶離原資料夾(隱私/體積)。是否需提供『只存 2D 近似(密集區判定降級、明確標註不準)』的 opt-out 開關?還是一律存 D 維(密集區判定才準)?
4. few-shot 跨資料夾沿用是否進 MVP?我裁 MVP 砍掉(few-shot 仍只服務建 bank 那次,session 內),內容定址持久化進 backlog。若 User 第二次資料夾就想看舊標記對齊,需提到 MVP。
5. object 模式(score_mode=='object',無 patch bank)是否也要支援存『投影 + few-shot only』的 projection-only profile(無 bank.npz)?拆檔設計已可支援,但需 PO 確認此變體在範圍內。
6. umap 合併重投影第二按鈕(re-fit-together 第二視角)與 greedy/k-center coreset(取代純隨機 budget=100000)是否進範圍?我裁兩者皆 backlog,但若 User 在意 coreset representativeness 或 umap 視覺,需提前。
7. stale 過期警告(fingerprint_similarity 門檻)的具體閾值(0.6?)與是否進 MVP:我裁 MVP 先不算 stale,只在 backlog。若 User 需要『新舊差異大』的明確提示,需提到 MVP。