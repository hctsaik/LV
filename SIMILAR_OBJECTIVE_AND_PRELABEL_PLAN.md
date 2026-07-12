# 功能建議書:「找長得像指定物件」選樣目標 + 「看過類別 AnomalyDINO 預標」

> 日期:2026-07-05。狀態:**未動任何程式**。
> **Feature B(預標)已由使用者拍板要做(2026-07-05)** → 正式需求見
> [1_user_needs/prelabel_seen_class.md](1_user_needs/prelabel_seen_class.md),
> ROADMAP 掛為 **M11 候選**;Feature A(找相似)仍為**建議、未拍板**。
> 目的:讓較小的 AI model(或人)能按此文件**逐步、小步**完成開發。
> 每個 Task 都刻意切到「一次改一個函式、有明確驗收指令」的粒度。
> 走 U-Net 流程時:本文件是 User/PO 素材;正式開發前仍應過 `/po → /architect → /pm → /pg`。

---

## 0. 現況盤點(先讀這段,別假設)

### 0.1 「選樣目標」目前出現在 3 個地方

| 位置 | 函式 | UI 選項 key | 實際引擎 |
|------|------|------------|----------|
| ③挑樣 2×2 矩陣 | `_anomaly_tab_sample` + `_anomaly_priority_for`(`scripts/app.py`) | `novelty / confusion / balanced / pure` | in-memory:`active_learning.priority_score` 權重組合、`active_loop.confusion_targeted_priority` |
| M9 大資料分批掃描 | `_anomaly_batch_section` → `_anomaly_batch_run`(`scripts/app.py`) | `novelty / balanced / confusion / pure` | `al_batch.run_batched(objective=...)` |
| M10 持續監看 | `_anomaly_watch_section` → `_anomaly_watch_init`(`scripts/app.py`) | 同上 | 寫進 `profile.yaml` → `al_service.run_once` → `al_batch` |

### 0.2 引擎端 objective 的真相(`scripts/al_batch.py`)

- `run_batched` **只接受** `("novelty", "uncertain", "confusion")`,其它值直接
  `raise ValueError`(見 `run_batched` 開頭的驗證)。
- `_priority(objective, score, proba)` 語義:
  - `novelty` = `_minmax(score)`(**等於 GUI 的「純 novelty」**,不是 GUI「抓沒看過的異常」的 (1,.4,.4) 加權)
  - `uncertain` = novelty + boundary + entropy(**等於 GUI 的「三訊號均衡」**)
  - `confusion` = `confusion_targeted_priority`(novelty + entropy)
- checkpoint shard(`shard_*.npz`)欄位:`item_id / anomaly_score / image_path / bbox /
  obj_index / label / head_proba`。**沒有物件 embedding** —— 這是新目標「找相似」要動的點。
- C8 憲法:**任何全域運算(minmax/threshold/HDBSCAN)只准在 `_merge_and_topk` 合併時做**,
  per-batch 只准 per-item 運算。

### 0.3 模型目錄(bank profile)存了什麼(`scripts/anomaly_bank_store.py`)

`meta.json` + `bank.npz(vectors)` + `projection.npz` + `fewshot` + `head.joblib`。
**vectors 沒有 provenance**(不知道每支向量來自哪張圖哪個 bbox)→ 跨 session 載入模型後,
「挑一個建模時的特定物件」目前**做不到**(沒縮圖可挑)。這決定了 Feature A 的分期。

### 0.4 「預標」現況:**不存在**

- 全 repo `scripts/` 無任何 prelabel/預標實作;M5 設計明寫「非自動標註」「不做全自動標註」。
- 最接近的既有零件:
  - `dino_head.gated_predict` → 每物件判 `正常 / <類別名> / Unknown`(③ 的徽章就是它,
    見 `app.py::_anomaly_queue_labels`)。**預測已存在,缺的是「落成標註檔 + 批次接受」**。
  - M10 `al_workspace.append_label / labels.jsonl`(佇列單筆人工標)。
  - `interaction.zip_selected_images`(購物車匯出,只有圖沒有標)。
- 憲法 C6([[no-dataset-writes]]):**絕不寫使用者資料集資料夾**。預標輸出只能
  (a) 進 `.lv_cache` 暫存,(b) 匯出到使用者**另選**的輸出資料夾,或 (c) download_button。

---

## Task 0(先修 bug,擋在所有新功能前面):M9/M10 objective 名稱不匹配

**症狀**:M9「大資料分批掃描」或 M10「監看」選「三訊號均衡」或「純 novelty」→
GUI 把 `"balanced"` / `"pure"` 原樣傳給 `al_batch.run_batched` → `ValueError:
objective must be 'novelty' / 'uncertain' / 'confusion'` → 使用者看到「分批掃描失敗:…」。
(現有 E2E 只測了 novelty,所以沒炸。)

**修法(建議,最小改動)**:在 `scripts/app.py` 加一個模組層映射,兩處呼叫點套用:

```python
# GUI 選樣目標 → al_batch 引擎 objective(引擎的 novelty 本義就是純 minmax)
_AL_ENGINE_OBJ = {"novelty": "novelty", "pure": "novelty",
                  "balanced": "uncertain", "confusion": "confusion"}
```

- `_anomaly_batch_run` 開頭:`objective = _AL_ENGINE_OBJ.get(objective, objective)`
- `_anomaly_watch_init`:寫 profile 前同樣映射(profile.yaml 內存**引擎值**,
  離線服務才不用懂 GUI 詞彙)。
- UI 誠實化:M9/M10 的 `_OBJ` 選單裡「抓沒看過的異常」與「純 novelty」在引擎端**同義**
  (引擎沒有 .4 加權版),二選一:(a) M9/M10 選單合併成 3 項(建議),
  (b) 保留 4 項但 caption 註明兩者等價。**選 (a)**:`_OBJ` 只留
  `novelty(純 novelty:抓沒看過的異常)/ balanced / confusion`。
- ③挑樣 的 2×2 **不動**(它走 in-memory 權重,語義本來就不同)。

**驗收**:
1. 新增單元測試不可行(app.py 是 GUI),用 E2E:在
   `tests/e2e/test_al_batch_gui_e2e.py` 加一測:選「三訊號均衡」按掃描(用含 head 的模型
   fixture),斷言頁面**出現「掃描完成」而非「分批掃描失敗」**。
2. `pytest tests/e2e/test_al_batch_gui_e2e.py -m e2e -v` 全綠。
3. 手驗:M10 選 balanced → 初始化 → 掃描 → `profile.yaml` 內 `objective: uncertain`。

---

## Feature A:「長得像指定物件」選樣目標(找相似)

### A.0 使用者故事與 UX 決策

> 「我在結果牆看到一顆特別的物件(或建模時就認識某顆),我要把整批資料裡
> **長得像它的**全部撈到佇列最前面。」

**參考物件從哪來?——分期決策(關鍵)**:

| 期 | 參考來源 | 為什麼 |
|----|---------|--------|
| A1(先做) | 從 **② 套用結果的物件牆挑一顆**(「以此為參考找相似」按鈕) | `result["obj_emb"]` 已在 session,零持久化改動,立刻可用 |
| A2 | 建模時的物件:存模型時**順手寫 provenance**(`ref_meta.json`:每支 vector 的 image_path+bbox+label)→ 載入模型後可出縮圖挑選牆 | 解「跨 session 挑建模物件」;只動 `.lv_cache` 內的模型目錄,合憲 |
| A3 | M9/M10 引擎支援 `objective="similar"` | 大資料批次跑;需動 shard 格式與 run 識別 |

**相似度定義(釘死)**:cosine similarity。參考向量 `r`、物件 embedding `e_i`(同一
DINOv2 模型、同 target_res、同 L2-normalize 流程——直接重用 `_object_embeddings` 的輸出):
`sim_i = (e_i·r)/(‖e_i‖‖r‖)`。優先分數 = `_minmax(sim_i)`(**降冪=最像在前**)。
多參考(進階,可不做):`sim_i = max_j cosine(e_i, r_j)`。

### A.1 Task:③挑樣 加「🔎 找相似」模式(in-memory,Tier B-GUI)

1. **新純函式**(放 `scripts/active_learning.py`,Tier A 可單測):
   ```python
   def similarity_priority(obj_emb, ref_vec) -> np.ndarray:
       """cosine(e_i, ref) 經 _minmax;空輸入回 zeros(0);ref 全零 raise ValueError。"""
   ```
   釘死 AC:`similarity_priority([[1,0],[0,1],[.9,.1]], [1,0])` 的 argsort 降冪 = `[0,2,1]`;
   輸出範圍 [0,1];`len==0` 回 shape (0,)。
2. **UI**:`_anomaly_tab_sample` 的 2×2 之下加一個 expander「🔎 找長得像指定物件的」:
   - 參考挑選:`st.selectbox` 或縮圖牆(重用 ③ 大圖牆的 `crop_bbox` 渲染)從
     `result["records"]` 挑 index;存 `st.session_state["anomaly_sim_ref_idx"]`。
   - 也提供「從②物件詳情按『以此找相似』」捷徑:在 `_anomaly_render_inspector`
     選中物件處加一顆 button,寫同一個 session key 並把 ③ 切到找相似模式。
   - 排序:`pri = similarity_priority(obj_emb, obj_emb[ref_idx])` →
     沿用 `select_for_labeling(pri, k, cluster_labels, max_per_cluster)` 出佇列與購物車。
   - **防呆**:`obj_emb is None`(patch-only 模式可能沒物件 embedding)→
     顯示「此模型/結果無物件 embedding,無法找相似」,不炸。
3. **驗收**:單元 `tests/test_<module>.py`(similarity_priority 的 AC)+ E2E 一條:
   合成資料(兩群明顯不同的物件)→ 挑 A 群一顆當參考 → 斷言佇列前 K 全是 A 群。

### A.2 Task:存模型時寫 reference provenance(解跨 session 挑建模物件)

1. `anomaly_bank_store.save_bank` 加 optional 參數 `ref_meta: list[dict] | None`
   (每項 `{image_path, bbox, label}`,與 vectors 同序同長)。寫成
   `ref_meta.json`(用既有 `_atomic_text`);`load_bank` 讀回(缺檔回 None——
   **舊模型目錄相容,不 raise**)。
2. `app.py::_anomaly_save_model` 在建模 result 可得 records 時傳入。
3. UI:①載入模型後,③ 的找相似 expander 若 `ref_meta` 存在 → 多一個
   「參考來源:建模物件」tab,縮圖用 `crop_bbox(image_path, bbox)` 現場裁
   (檔案可能已被移走 → 裁失敗顯示灰框佔位,不炸)。
4. 釘死 AC:save→load 往返 `ref_meta` 逐欄相等;舊目錄(無 ref_meta.json)load 回
   `None` 且其它欄位照常;bbox 裁圖失敗路徑有 fallback。

### A.3 Task:al_batch 支援 `objective="similar"`(M9/M10 批次版)

**動引擎前先讀 `3_Architect_Design/08_al_batch.md` 與 C8。** 改動點:

1. `run_batched` 簽名加 `ref_vector: np.ndarray | None = None`;
   `objective` 白名單加 `"similar"`;`similar` 而 `ref_vector is None` → `ValueError`。
2. **run 識別**:`_run_key(...)` 的組成加 `sha256(ref_vector.tobytes())[:16]`
   (換參考物件=另一個 run,不能沿用舊 checkpoint;`objective` 本來就在 key 裡)。
3. **per-batch**(C8-safe,只做 per-item):`_score_batch` 之後、寫 shard 前,
   `similar` 時算 `ref_sim = cosine(obj_emb_i, ref_vector)`(obj_emb 在 object 模式
   已經算了;patch 模式也要 `_object_embeddings` 一次——成本註記在 docstring)。
   shard 多存一欄 `ref_sim`(float32)。
4. **merge 端**(全域運算唯一合法處):`_priority` 加分支
   `similar → _minmax(concat(ref_sim))`;`_reason` 加
   `f"和參考物件相似度 {sim:.2f}"`。
5. **相容**:舊 shard 沒 `ref_sim` 欄 + objective=similar → 視為 identity mismatch
   → 走既有 `on_identity_mismatch` 路徑(其實 run_key 已擋掉,這是雙保險)。
6. GUI:M9 `_OBJ` 加「長得像指定物件」;選它時出參考挑選器(A1/A2 的來源);
   `_anomaly_batch_run` 把 ref 向量傳進 `run_batched`。
7. M10 profile:`profile.yaml` 加 optional `reference_vector_file`(存 `.npy` 於
   workspace 內,`init_workspace` 寫入);`al_service.run_once` 讀檔傳給 run_batched。
8. 釘死 AC(給 /pm):
   - 合成 3 物件、參考=物件0 → topk 順序 = 相似度降冪,`reason` 含「相似度」。
   - 換 ref_vector 續跑 → 舊 checkpoint 不被沿用(識別不符行為與既有 AC 同)。
   - `objective="similar"` 無 ref_vector → ValueError,訊息含 "ref_vector"。
   - 兩批分開跑 vs 一次跑完,topk **完全一致**(C8 批次正確性,對齊既有 AC 寫法)。

**建議順序:A1 → (Task 0) → A3 → A2 可與 A3 平行。** A1 自己就有完整使用者價值。

---

## Feature B:看過類別的 AnomalyDINO 預標(pre-label)

### B.0 誠實定位

> **狀態:使用者已拍板要做(2026-07-05)。** 正式需求:
> [1_user_needs/prelabel_seen_class.md](1_user_needs/prelabel_seen_class.md);
> 開發前走 U-Net 正式流程:`/po` 收斂 PRD(可直接取材本章)→ `/architect` → `/pm` → `/pg`。

你之前的印象沒錯——我們**討論過**,但 M5 拍板「不做全自動標註」,所以**目前沒有這功能**。
現在要補的是「**預**標」:模型先填,**人仍是最終裁決**,和 M5 的精神不衝突。

今天最接近的用法(不改程式就能做到的):
①建模時用**瑕疵類別**標籤(每類 ≥ N_min)→ 模型含分類頭 → ③挑樣 的佇列徽章
會顯示每顆物件的 `正常/<類別>/Unknown` 判定 → 人眼參考它手標。
缺的只是「把這些判定**寫出去成標註檔**+**批次接受**」。

### B.1 範圍決策(v1 只做分類預標,不做偵測預標)

| | v1(做) | v2(現在不做,記著) |
|--|--------|--------------------|
| 輸入 | **已有 bbox** 的物件(YOLO 標籤已存在,類別可能是佔位/錯的) | 無標籤影像 |
| 預標內容 | 每個 bbox 的**類別**(head gated_predict:正常/類別/Unknown) | bbox 本身也要生(patch 熱力圖 pmap → 閾值 → 連通域 → 外接框),工程大、雜訊多 |
| 輸出 | 修訂版 YOLO labels(**匯出到使用者另選資料夾**,C6) | 同左 |

### B.2 Task:純邏輯核心 `scripts/prelabel.py`(Tier A,先做)

```python
def prelabel_records(records, obj_emb, scores, head, *, class_names,
                     anomaly_threshold, min_conf=0.5) -> list[dict]:
    """每物件回 {item: i, pred: 'normal'|'unknown'|<class>, conf: float,
    accept: bool}。accept = pred 不是 unknown 且 conf >= min_conf。
    直接包 dino_head.gated_predict,不重新發明。"""

def to_yolo_lines(records, decisions, *, class_names) -> dict[str, list[str]]:
    """依 image_path 分組,回 {image_path: ['<cls_id> cx cy w h', ...]}。
    僅含 accept 的物件;pred=='normal' 的物件輸出時**保留原標籤或剔除**
    ——釘死:剔除(正常=非瑕疵,不進瑕疵標註檔),並在 summary 記數。"""

def export_prelabels(lines_by_image, out_dir, *, copy_images=False) -> dict:
    """寫 out_dir/labels/*.txt(+選配 copy images/)。out_dir 必須通過
    assert_safe 檢查:**不得等於或位於任何 watch/dataset 資料夾內**(C6)。
    原子寫(重用 anomaly_bank_store._atomic_text)。回 {written, skipped}。"""
```

釘死 AC(給 /pm 直接抄):
- 3 物件(head 判 A 類 conf .9 / Unknown / 正常),`min_conf=0.5` →
  `accept == [True, False, False]`,to_yolo_lines 只含第 1 顆,cls_id 對 `class_names.index`。
- bbox 轉 YOLO 格式往返誤差 < 1e-6(cx,cy,w,h 均在 [0,1])。
- `export_prelabels(out_dir=某 watch 資料夾內路徑)` → raise(訊息含「資料集」或 C6 字樣)。
- `head=None` → raise ValueError(訊息含「分類頭」),不靜默退化。

### B.3 Task:GUI 接線(Tier B)

位置:③挑樣 佇列/購物車區加 expander「🏷️ 預標(分類頭代填,人工最終確認)」:
1. 參數:`min_conf` slider(預設 0.5)、目標類別 multiselect(預設全部;
   **這就是「某一個看過類別」**——只接受選中類別的預標)。
2. 預覽表:pred/conf/accept 三欄 + 縮圖,**先看再匯**;被 reject 的顯示原因(Unknown/低信心)。
3. 匯出:`st.text_input` 選輸出資料夾 + 「⬇ 匯出預標 YOLO」按鈕(呼叫 export_prelabels);
   或 download_button 打包 zip。**絕不提供「直接寫回來源資料夾」選項。**
4. M10 佇列的單筆標註按鈕旁,順手顯示 head 建議(`建議:<類別> (0.87)`)+
   「採納建議」快捷鍵——這是預標在監看流程的形態(寫進 labels.jsonl,
   `source: "prelabel_accepted"` 供追溯)。
5. E2E AC:合成兩類物件建含 head 模型 → ② 套用 → 預標 expander 出現、
   預覽表 accept 數正確 → 匯出到 tmp 資料夾 → **實際讀回 labels/*.txt 斷言行數與 cls_id**
   (真實行為,不是 element 存在)。

### B.4 v2 備忘(現在不做)

無標籤影像的偵測預標:`anomaly_score.score_object` 已回 pmap(③熱力圖在用)→
`pmap > percentile 閾值` → `scipy.ndimage.label` 連通域 → 外接框 → 每框過 B.2 分類。
風險:框品質低、參數敏感;若要做,先用 M8 whole_image 佇列人工框一輪估 baseline。

---

## 建議執行順序(給小模型的路線圖)

> 2026-07-05 調整:使用者拍板 **Feature B 優先**(已入 ROADMAP=M11 候選);
> Feature A 未拍板,排在 B 之後、由 PO 屆時再決定要不要開輪。

```
Task 0   (bug 修復,~1 函式 + 1 E2E)          ← 先做,10 分鐘等級
B.2      (prelabel.py 純邏輯,Tier A)          ← M11 主體,已拍板
B.3      (預標 GUI + E2E)                      ← M11 主體,已拍板
A1       (③找相似 in-memory)                  ← 未拍板,候選
A3       (al_batch similar + M9/M10 接線)      ← 未拍板;動引擎,走完整 U-Net 五層
A2       (ref_meta provenance)                 ← 未拍板;獨立,隨時可插
```

每個 Task 完成的客觀判準:新/改模組跑 `python verify/gate.py <module>` 印 GREEN;
GUI 相關另加對應 E2E 綠(`pytest tests/e2e/test_<module>_e2e.py -m e2e -v`)。
**動 al_batch/al_workspace/al_service 前必讀**:`3_Architect_Design/08~10_*.md` 的 AC,
改完先 `python verify/gate.py --snapshot`?——**不對,snapshot 是 PM 改契約後才跑**;
PG 只實作。契約要變(如 run_batched 簽名加參數)→ 走 /architect 增修設計 → /pm 補測 →
snapshot → 才放 /pg。

## 憲法紅線(小模型最容易踩,抄在這)

- C6:任何輸出**不得寫進使用者資料集/監看資料夾**;快取進 `.lv_cache`,
  匯出只到使用者另選目錄且要 assert 防呆。
- C8:al_batch 的 per-batch 階段禁止 minmax/threshold/全域統計;只有
  `_merge_and_topk` 能做。找相似的 cosine 是 per-item,合法;minmax 留在 merge。
- C4:凍結模型;找相似/預標都**不得觸發任何 retrain**。
- 測試基礎設施(conftest/fixtures)PM 擁有;PG 不碰 `tests/`、`verify/`、`3_/4_` 契約。
