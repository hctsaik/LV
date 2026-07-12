# PRD:以樣搜樣(小樣本海撈 + YOLO 預標 + 人工確認)(M13 / 第 9 工具)

> 來源需求:[1_user_needs/fewshot_search_prelabel.md](../1_user_needs/fewshot_search_prelabel.md)。
> 規格素材(§0 拍板決策表不得擅改):[FEWSHOT_SEARCH_AND_PRELABEL_PLAN.md](../FEWSHOT_SEARCH_AND_PRELABEL_PLAN.md)。

## 三行自問(缺一不放行)

- **誰會用**:資料集管理者/品管工程師,手上有少量樣本(**4 類 × 每類 5~10 張**),要從大量原始資料
  (帶**低信心六欄 YOLO 粗框**的偵測器輸出)裡海撈同類、先標好、交人確認。這是他自述**最主要的日常工作**。
- **不做會怎樣**:只能人眼翻幾千張、找到還得逐顆手標,一天做不完幾百張;之前確認過的沒有累積效果。
- **成功的可觀察判準**(→ 餵 `/pm`):
  1. 丟樣本(小 YOLO 資料夾)→ 對大資料夾海掃 → 佇列**依相似度降冪**,前段**大多真的是要的**;
     每顆顯示**建議類別(來自樣本比對,非粗框 cls)+ 相似度**。
  2. 確認(✅收/✏改類/⏭略過)→ 匯出 **YOLO labels**(框沿用粗框幾何、類別=確認結果)**+ retrieval_report.csv**
     到使用者**另選**資料夾;讀回檔案正確;**來源資料夾一個位元組未變**(C6)。
  3. 大資料掃到一半可**中斷、續跑**(接著上次)。
  4. 確認過的物件可**加回樣本集**;累積到「≥2 類 × 每類 ≥8」→ 提示可去瑕疵偵測①訓分種類。
  5. **監看版**:指定會持續進新圖的資料夾 → 背景服務自動掃新圖 → 早上打開看結果。

## 目標與成功指標(可衡量)

| 指標 | 目標 |
|------|------|
| 檢索正確性 | 合成兩群 + 錯 cls 粗框:佇列前段建議類別**正確且 ≠ 粗框 cls**(釘死於 E2E) |
| 相似度定義 | multi-exemplar **max**-cosine over 同類樣本(不取平均);釘死於 13a AC |
| 分批正確性 | 分批==一次跑 topk 一致(C8);換樣本集→不沿用舊 checkpoint |
| C6 安全 | 匯出只到另選目錄;來源資料夾前後檔案集合不變 |
| 匯出 | YOLO labels(五欄沿用粗框幾何)+ CSV(欄位釘死)雙出 |
| 既有無回歸 | al_batch 21 / al_workspace 14 / al_service 9 測全綠 |

## 範圍(MoSCoW)

- **Must**
  - 樣本集(few-shot bank,帶 provenance,.lv_cache)+ 海掃(粗框 proposal,max-cosine,可續跑)
    + 確認佇列(收/改類/略過)+ **雙匯出**(YOLO 沿用粗框幾何 + CSV)。
  - **監看模式**(復用 M10:workspace/增量/佇列/匯出設定,profile 加 sample_bank_dir)。
  - 獨立第 9 工具三步 wizard(不塞進瑕疵偵測)。
- **Should**:確認物件「➕ 加回樣本集」迴圈;累積夠→訓頭導流提示;conf 預篩滑桿。
- **Could**:完全無標資料的整張影像級檢索(退化路)。
- **Won't(v1)**:粗框自動修框/NMS(框原樣沿用、人確認時修);相似度校準成機率;跨模型樣本集自動遷移;
  全自動接受(人是最終裁決)。

## 模組分解(交 `/architect`;Tier)

| # | Module | 一句話職責 | Tier | 相依 |
|---|--------|-----------|------|------|
| 13a | `similarity` 擴充 | 對多樣本(每類多顆)算 per-class **max**-cosine,回 (best_label, best_sim) | **A** | numpy;既有 cosine_similarity_to_ref |
| 13b | `sample_bank` | few-shot 樣本集 build/save/load/append(emb+provenance+模型綁定,原子寫) | **B** | discover_yolo_objects / _object_embeddings / anomaly_bank_store._atomic_* |
| 08 | `al_batch` 擴充 | `ref_vectors+ref_labels` 多參考 + shard 存 best_sim/best_class + run 識別含樣本 hash + conf 預篩 | **B** | 既有 08(21 測無回歸鐵則) |
| 13c | `retrieval_export` | decisions → YOLO 行(沿用粗框幾何)+ retrieval_report.csv(欄位釘死) | **A** | prelabel.to_yolo_lines/export_prelabels/assert_safe_prelabel_dir |
| 09/10 | 監看擴充 | profile 加 `sample_bank_dir`;run_once 讀樣本集跑多參考 | **B** | M12b reference 機制;al_workspace/al_service(14/9 測無回歸) |
| — | 第 9 工具 GUI | 三步 wizard + θ 滑桿/直方圖 + 佇列 + 雙匯出 + 加入樣本集 + 訓頭提示 + 監看區 | **B** | M10 佇列/M11 匯出/M9 進度條 pending-flag 模式 |

### 拆模組品質自檢
- 13a 內聚(多樣本→per-class max-cosine),契約=1 純函式,可獨立驗收(小 2D 向量)。✅
- 13b 內聚(樣本集持久化),對外 build/save/load/append + 明確 schema,不成環。✅
- 08 擴為**加法**(新可選參數 + 新 objective 分支),既有 21 測**全綠**是紅線。
- 13c 薄層(組合 prelabel 匯出 + CSV),可獨立驗收。✅

## 「done」定義(分層)
- 純邏輯(13a/13c):`gate.py <module>` 印 GREEN。
- I/O(13b)/引擎擴(08)/服務擴(09/10):gate GREEN(單元+真實 tmp 檔案系統/合成資料)。
- GUI(第 9 工具/監看):done = 相關 gate 綠 **AND** 真實 Playwright E2E 綠。

## 使用者驗收
- 丟 4 類×5~10 張樣本、海掃帶粗框大資料夾 → 佇列前段大多是我要的、建議類別對;
  收/改/略過 → 匯出 YOLO+CSV 到另選夾、來源沒被動;中斷續跑;資料夾長大時監看自動撈。

## 給架構師的開放問題
1. 第 9 工具在 tool_switch 的命名/圖示(「🎯 以樣搜樣」);與第 8「瑕疵偵測」的導覽關係。
2. 13a 回傳結構:每物件 `(best_label, best_sim)` 是否也要回 per-class 全分數(供直方圖/除錯)?
3. 08 的 `ref_vectors/ref_labels` 與既有單一 `ref_vector`(M12b)如何相容(建議:單顆=M=1 的特例)。
4. 監看 profile `sample_bank_dir` 的可攜性(離線服務要能只憑 profile+sample_bank 目錄跑)。

## 本輪 appetite
**Task 1→7 分段做,每段 gate/E2E 綠才進下一段**(先 13a 純函式,最後 GUI+監看)。
達里程碑即停輪;Should/Could 未完進候選。
