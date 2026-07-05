# 功能分類建議:「🎯 以樣搜樣」— 小樣本海撈 + YOLO 預標 + 人工確認

> 日期:2026-07-05。狀態:**規格已與使用者兩輪問答定案,未動任何程式**。等待開發指令。
> 使用者拍板的核心工作流(自述「非常主要的工作」):
> **用 sample 小量資料 → 海掃大量資料 → 找出我要的 → YOLO 預標 → 交人確認。**
> 目的:讓後續 AI(或人)能按此文件小步實作。開發走 U-Net 流程(/po→/architect→/pm→/pg),
> 每個 Task 有 gate/E2E 客觀綠燈。

## 0. 已拍板決策(2026-07-05,兩輪問答;後續 AI 不得擅改)

| 問題 | 拍板 |
|------|------|
| 大資料有無框? | **有「低信心粗框」可參考**:大資料夾 `labels/*.txt` 為**六欄 YOLO(cls cx cy w h conf)**(偵測器輸出)。既有 `interaction.parse_yolo_boxes_conf` 已支援讀取,零改動。 |
| 粗框的類別可信嗎? | **不可信,只用框位置**。粗框只當「這裡有個東西」的候選;**類別一律由「最像哪類樣本」決定**,完全忽略粗框 cls。 |
| 樣本規模 | **每類 5~10 張、約 4 類**(總計 ~20-40 顆樣本)。 |
| 離線監看版 | **要,v1 一起做**(資料夾會持續進新圖;復用 M10 架構)。 |
| 匯出形式 | **YOLO labels + 影像清單 CSV 都要**。 |
| 「像某一類」教訓(M12) | 檢索類別語義一律錨在**樣本集**(使用者策展),不吃掃描端標籤;與預標(head)分工不重疊。 |

---

## 1. 為什麼是「新功能分類」,而不是塞進瑕疵偵測

方向相反、但引擎共用:

| | 瑕疵偵測(第 8 工具) | 以樣搜樣(新,第 9 工具) |
|---|---|---|
| 問題 | 「哪些**不像正常**?」(離群=可疑) | 「哪些**像我的樣本**?」(相近=要的) |
| 參考 | 正常樣本 bank | **想找的東西**的小樣本集 |
| 排序 | 異常分數降冪 | 相似度降冪 |
| 產出 | 可疑佇列 | **YOLO 預標 + CSV**(等人確認) |

塞進瑕疵偵測 wizard 會重演「找相似 vs 分種類」的混淆(前車之鑑:M12「像某一類」因此被砍)。
**定案:獨立第 9 工具「🎯 以樣搜樣」,三步 wizard:①放樣本 → ②海掃 → ③確認+匯出;外加監看模式。**

### 與既有功能的分工(必守)

- **找同款(M12)**:②結果內、臨時、單顆範本的互動探索。以樣搜樣=「正式工作流」版:多顆樣本、掃別的大資料夾、產出預標。
- **預標(M11)**:模型**已會分種類**(head)時的閉集代填。以樣搜樣=**還沒有 head / 新類別 / 少樣本**時的預標手段(相似度→樣本類別)。
- **閉環定位**:以樣搜樣 = bootstrapping。確認標註累積到「≥2 類 × 每類 ≥8」→ GUI 提示「🎓 可以訓分種類模型了 → 瑕疵偵測①」,之後改用 M11(更準)。

---

## 2. 三步設計(含拍板細節)

### ① 樣本集(few-shot bank)

- 樣本來源:小 YOLO 資料夾(images/+labels/,每框=一顆樣本,label=想找的類別;**約 4 類、每類 5~10 張**)。
  無標樣本資料夾(整張=樣本,單一類、類名使用者填)也支援,但主路是 YOLO。
- **持久化**:`.lv_cache/<hash>/sample_bank/` —— `emb.npz`(N×384)+ `meta.json`
  (每顆 image_path/bbox/label + 綁定凍結模型名/res)。~40 顆樣本 ≈ 60KB。
  > 與 anomaly bank 不同:anomaly bank 是 patch coreset **無 provenance**(M12 討論確認不可挪用);
  > sample_bank 從第一天帶 provenance。
- **增量**:③確認過的物件一鍵「➕ 加入樣本集」→ 下輪更準(每類樣本少,滾大很重要)。
- 樣本集**綁凍結模型**;meta 記模型,不符 → 明確 raise + 提示用 provenance 重 embed(原圖缺 → 明確提示)。

### ② 海掃(大資料、可續跑、粗框 proposal)

- **物件來源(拍板)**:大資料夾 `labels/*.txt` = 六欄 YOLO 低信心粗框 → **逐粗框**裁切、embed、比對。
  - 粗框 **cls 忽略**;`conf` 欄可選「信心預篩」滑桿(例:略過 conf < 0.05 的雜框;預設不篩)。
  - 完全無 labels 的資料夾 → 退回**整張影像級**檢索(引擎已支援 `object_source="whole_image"`)。
- **相似度(釘死)**:multi-exemplar —— `sim(x, 類c) = max_j cosine(x, 樣本_j∈c)`
  (**max 不取平均**:每類才 5~10 顆、可能含次型,centroid 會糊;max=「像任何一顆就算像」)。
  每物件輸出 `(best_class, best_sim)` = 最像的類 + 該相似度。
- **引擎**:復用 `al_batch`(分批、checkpoint、續跑、C8)。加法擴充:
  `ref_vector` → `ref_vectors (M,D) + ref_labels (M,)`;shard 多存 `best_sim`/`best_class`(per-item,C8 合法);
  merge 端 minmax 排序;run 識別含**樣本集 hash**(樣本變=另一 run;對齊 M12 ref_key 作法)。
- **門檻 θ**:`best_sim ≥ θ` 才進「找到」佇列;GUI 滑桿 + 分數分佈直方圖。
  **誠實界定**:cosine 未校準,θ 是操作參數非機率;預設保守。

### ③ 確認 + 匯出(拍板:兩種都出)

- 佇列:`best_sim` 降冪;縮圖(粗框裁切)+ **建議類別 + 相似度**;每顆:✅ 收 / ✏ 改類別(4 類下拉)/ ⏭ 略過
  (樣式復用 M10 監看佇列)。
- **匯出 A:YOLO labels** → 使用者另選資料夾。**框沿用粗框幾何**(五欄,去掉 conf),類別=確認後的樣本類別。
  復用 `prelabel.to_yolo_lines` + `export_prelabels` + `assert_safe_prelabel_dir`(C6);classes.txt=樣本類別空間。
- **匯出 B:影像清單 CSV** → 同輸出資料夾 `retrieval_report.csv`,欄位(釘死):
  `image_path, obj_index, cx, cy, w, h, proposal_conf, suggested_class, similarity, decision(accepted/relabeled/skipped/pending), final_class`。
- 回饋:確認數達「≥2 類 × 每類 ≥8」→ 顯示訓頭導流提示。

### ④ 監看模式(拍板:v1 就做)

- 復用 M10 全套:workspace(profile.yaml / state.json / queue.jsonl / labels.jsonl / 鎖 / 增量掃描)。
- profile 擴充:`sample_bank_dir`(取代單一 reference_vector_file 的角色;M12b 的 reference.npy 機制是單參考版前例)。
- `run_once`:增量掃新圖 → 逐粗框比對樣本集 → 更新佇列;GUI 佇列消費同 M10(✅/✏/⏭ → labels.jsonl)。
- **匯出設定可攜**:profile.yaml 可匯出給離線服務跑(同 M10 UX)。

---

## 3. 模組分解(交 /po 收斂;Tier 建議)

| # | Module | 一句話職責 | Tier | 復用 |
|---|--------|-----------|------|------|
| 13a | `similarity` 擴充 | `multi_ref_similarity(X, refs, ref_labels) -> (best_label, best_sim)`(per-class max-cosine) | **A** | cosine_similarity_to_ref |
| 13b | `sample_bank` | 樣本集 build/save/load/append(emb+provenance,.lv_cache,原子寫,模型綁定驗證) | **B** | discover_yolo_objects / _object_embeddings / _atomic_* |
| 08 擴 | `al_batch` 多參考 | `ref_vectors+ref_labels`、shard 存 best_sim/best_class、run 識別含樣本 hash、conf 預篩參數 | **B** | 既有 21 測**無回歸鐵則** |
| 13c | `retrieval_export` | decisions → YOLO 行(沿用粗框幾何)+ `retrieval_report.csv`(欄位見 §2③) | **A** | prelabel.to_yolo_lines / export_prelabels |
| 09/10 擴 | 監看 profile | `sample_bank_dir` 欄位 + run_once 讀樣本集跑多參考 | **B** | M12b reference 機制、al_workspace/al_service(14+9 測無回歸) |
| — | 第 9 工具 GUI | 三步 wizard + θ 滑桿/直方圖 + 佇列(收/改類/略過)+ 雙匯出 + 加入樣本集 + 訓頭提示 + 監看區 | **B**(真實 E2E) | M10 佇列、M11 匯出、M9 進度條模式(pending flag→主體執行,progress 才會動) |

## 4. 釘死 AC 示例(給 /pm 直接取材)

- **13a**:`multi_ref_similarity([[1,0]], refs=[[1,0],[0,1]], ref_labels=["a","b"])` → `("a", 1.0±1e-6)`;
  同類樣本 `[1,0],[0,1]`(label 都 "a")、query `[0,1]` → `("a", 1.0)`(**max** 非平均);
  空 refs / 長度不符 → ValueError;N==0 → 空輸出。
- **13b**:save→load 往返逐欄相等;模型不符 → raise(訊息含模型名);append 後 N+1 且原子;
  舊/缺目錄 → 明確錯誤。
- **08 擴**:分批 vs 一次跑 topk 一致(照抄既有 C8 AC);換樣本集 → 舊 checkpoint 不沿用;
  六欄 conf 預篩 `min_conf=0.5` 時 conf<0.5 的粗框不進評分;既有 21 測全綠。
- **13c**:匯出 YOLO 行 = 粗框幾何(誤差<1e-6)+ 確認類別;CSV 欄位/行數與 decisions 一致;
  C6:輸出目錄在來源內 → raise;來源資料夾前後檔案集合不變(照抄 M11 AC-E4 樣式)。
- **E2E(真實行為)**:樣本=紅塊類+藍塊類(每類 5 顆)→ 海掃「紅/藍/綠塊混合、六欄粗框(cls 全錯)」
  大資料夾 → 佇列前段紅/藍佔多數且 `suggested_class` **正確(≠粗框 cls)** → ✅收幾顆+✏改一顆 →
  匯出 → 讀回 labels/*.txt 驗框幾何與類別、讀回 CSV 驗 decision 欄、來源零寫入。
- **監看 E2E**:init(profile 含 sample_bank_dir)→ 丟新圖 → run_once → 佇列出現新命中;
  標註後下輪移出(照抄 M10 AC-W3 樣式)。

## 5. Won't(v1 明確不做)

- ~~框生成~~ → **不需要了**(粗框即 proposal,拍板)。僅「完全無框且無標」資料退整張級。
- 粗框幾何的自動精修(refine/NMS):v1 框原樣沿用,人確認時修;粗框若重疊過多,誠實顯示。
- 相似度校準成機率(θ=操作參數)。
- 跨模型樣本集自動遷移(不符就 raise + 提示重建)。
- 全自動接受(人是最終裁決,同 M11 底線)。

## 6. Task 順序(給後續 AI;每步 gate 綠才下一步)

```
Task 1  similarity.multi_ref_similarity(Tier A)                  → gate similarity 綠
Task 2  sample_bank(build/save/load/append + 模型綁定)          → gate sample_bank 綠
Task 3  al_batch 多參考 + conf 預篩(加法;21 測無回歸)          → gate al_batch 綠
Task 4  retrieval_export(YOLO 沿用粗框幾何 + CSV)               → gate retrieval_export 綠
Task 5  第 9 工具 GUI wizard(①②③ + θ + 佇列 + 雙匯出)          → 真實 E2E 綠
Task 6  監看模式(09/10 profile 擴充 + GUI 監看區)               → gate 09/10 無回歸 + 監看 E2E 綠
Task 7  (Should)加入樣本集迴圈 + 訓頭導流提示                   → E2E 綠
```

## 7. 憲法紅線 + 既有教訓(後續 AI 必讀)

- **C6**:樣本集/checkpoint 進 `.lv_cache`;匯出只到另選資料夾;**來源零寫入**(E2E 檔案集合比對驗證)。
- **C8**:per-batch 只准 per-item(max-cosine 合法);全域 minmax/排序只在 merge。
- **C4**:凍結模型;「訓頭提示」只導流,不自動訓。
- **進度條教訓(2026-07-05)**:長時操作**不可**在 on_click callback 裡跑(st.progress 不串流)——
  一律「callback 設 pending flag → 主體執行」;batch_size 用 20(每 20 張更新)。
- **術語教訓**:UI 全人話(不寫 novelty/head/embedding/N_min);「類」字只給樣本類別,不與瑕疵種類混用。
- U-Net:PM 先紅測試 + `gate.py --snapshot` → PG 實作;PG 不碰 `tests/`、`verify/`、`3_/4_`。
- E2E 前清殘留 streamlit server;E2E 交 harness(`tests/e2e/conftest.py`),不進 PG 自主修綠迴圈。
