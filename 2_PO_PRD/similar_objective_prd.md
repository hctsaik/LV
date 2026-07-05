# PRD:找相似選樣目標(長得像指定物件)(M12)

> 來源:使用者拍板(2026-07-05)「A1 + A3 批次版」。設計素材:
> [SIMILAR_OBJECTIVE_AND_PRELABEL_PLAN.md](../SIMILAR_OBJECTIVE_AND_PRELABEL_PLAN.md) Feature A。

## 三行自問

- **誰會用**:在結果牆看到一顆特別物件(或建模時就認識某顆),想把整批**長得像它的**全撈到佇列最前面的品管工程師。
- **不做會怎樣**:現有選樣只有 novelty/uncertain/confusion —— 想「找同一種東西/同一個缺陷樣態」時無從下手,只能一顆顆翻。
- **成功的可觀察判準**(→ `/pm` AC 來源):
  1. 挑一顆參考物件 → 佇列**依 cosine 相似度降冪**,最像參考的排最前(合成兩群資料:挑 A 群一顆 → 前 K 全 A 群)。
  2. 大資料**分批**跑 objective=similar 與一次跑完,Top-K **完全一致**(C8 批次正確性)。
  3. **換參考物件** → 不沿用舊 checkpoint(重算);離線監看服務可吃「參考向量」設定跑。

## 範圍(MoSCoW)

- **Must**
  - **A1(③ 互動)**:③ 挑樣加「🔎 找相似」—— 從**②套用結果**挑一顆物件當參考 → cosine 相似度排序佇列。
  - **A3(批次)**:`al_batch` 加 `objective="similar"` + `ref_vector`;M9 分批掃描選單加「找相似」+ 參考挑選;
    M10 監看 profile 可帶 `reference_vector_file`,`run_once` 讀來跑。
  - 純函式 `similarity_priority`(cosine→minmax,降冪=最像)為兩者共用核心。
- **Should**:②物件詳情加「以此找相似」捷徑(寫參考 → ③ 切找相似模式)。
- **Could**:多參考(max cosine)。
- **Won't(v1)**:跨模型比對(參考向量與目標須同一凍結模型 embedding 空間);A2「建模物件 provenance」
  (ref_meta.json)本輪不做,參考一律來自**當前②結果**(A2 進候選)。

## 模組分解(交 `/architect`)

| # | Module | 一句話職責 | Tier | 相依 |
|---|--------|-----------|------|------|
| 12 | `similarity` | 對參考向量算每物件 cosine 相似並正規化成優先分數(降冪=最像) | **A**(純陣列、無 I/O) | numpy |
| 08 | `al_batch`(**改**) | 既有分批引擎**加** objective=`similar` + `ref_vector`(per-item cosine 存 shard、merge 端 minmax) | **B** | 復用模組 12;既有 08 契約 |
| — | M12a GUI | ③「🔎 找相似」:參考挑選 + similarity_priority 排序佇列 | B(E2E) | 12、既有 wizard |
| — | M12b GUI | M9 選單加「找相似」+ 參考挑選;M10 profile `reference_vector_file` + run_once 傳入 | B(E2E) | 08(改)、09/10 |

> 09 `al_workspace`/10 `al_service` 只需**加一個可選 profile 欄位 + 傳遞**,不改核心契約(小改;若架構師判定需獨立設計再拆)。

### 拆模組品質自檢
- 12 內聚:一句話(參考向量→相似優先分數),無「以及」。✅ 契約=1 純函式 + 1 輔助;相依只 numpy,不成環。✅
  可獨立驗收:PM 用小 2D 向量即可寫紅測試(argsort 釘死)。✅
- 08 改動**加法為主**(新 objective 分支 + 新可選參數),既有 17 測須**全數保持綠**(回歸鐵則)。

## 「done」定義
- 12:`gate.py similarity` GREEN。
- 08(改):`gate.py al_batch` GREEN(既有 17 + 新增 similar AC,**無回歸**)。
- M12a/M12b GUI:對應真實 E2E 綠(③ 找相似前 K 全同群;M9 similar 掃出佇列;換參考不沿用 checkpoint)。

## 給架構師的開放問題
1. **相似度定義**:cosine(L2 normalize 後內積);參考向量 r 全零 → ValueError。優先=`_minmax(cosine)`(降冪)。請釘死。
2. **run 識別**:`objective="similar"` 的 checkpoint 身分須含 `sha256(ref_vector)`(換參考=另一 run)。請確認納入 `_resume_identity`。
3. **patch-only 模型**:similar 需**物件級 embedding** 算 cosine;patch 模式也要 `_object_embeddings` 一次(成本註記)。
   若某型態無物件 embedding → 明確報錯,不硬跑。
4. **C8**:cosine 是 per-item(合法);全域 minmax **只在 `_merge_and_topk`**。shard 多存 `ref_sim` 欄;
   舊 objective 的 shard 無此欄 → 靠 run 識別(objective 在身分內)天然隔離。

## 本輪 appetite
**A1(模組 12 + ③ GUI)先做完做綠 → commit → 再做 A3(08 改 + M9/M10 GUI)。** 兩段各自 gate+E2E 綠才進下一段。
