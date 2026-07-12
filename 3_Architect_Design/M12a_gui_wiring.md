# 設計:M12a GUI 接線(③ 找相似,`scripts/app.py`)

> 對應 PRD:[2_PO_PRD/similar_objective_prd.md](../2_PO_PRD/similar_objective_prd.md);純邏輯:[12_similarity.md](12_similarity.md)。
> Tier B(GUI → 真實 E2E)。done = 模組 12 gate 綠 AND 真實 Playwright E2E 綠。**只出契約 + 可觀察 AC。**

## 1. 目的
在 ③ 挑樣加「🔎 找相似」:從**②套用結果**挑一顆物件當參考 → 整批依 cosine 相似度降冪排成佇列,
把長得最像參考的排到最前面送人工標。

## 2. 位置與資料來源
- ③ 挑樣送人工標 tab 內,2×2 取樣矩陣**之後**(預標 expander 附近)新增 expander「🔎 找相似(長得像指定物件)」。
- 讀:`anomaly_apply_result` 的 `records`、`obj_emb`。參考 = `obj_emb[ref_idx]`,`ref_idx` 由使用者挑。
- 排序:`similarity.similarity_priority(obj_emb, obj_emb[ref_idx])` → `select_for_labeling(pri, k)`;
  **佇列排除參考自身**(參考是 query 不是結果)。

## 3. Widget key 契約(PG 必須實作)
| key | 元件 | 說明 |
|-----|------|------|
| `anomaly_sim_ref_idx` | number_input / selectbox(0..N-1) | 參考物件索引;旁顯示其縮圖 + 類別(label) |
| `anomaly_sim_k` | slider | 取幾個最像的送標註 |
| `anomaly_sim_ref` | container | 參考物件顯示(縮圖 + 「參考:<label>」) |
| `anomaly_sim_queue` | container | 相似佇列(縮圖牆;每卡含 label + 檔名;**不含參考自身**) |

DOM 穩定標記:參考容器內含文字「參考:」;佇列容器渲染 ≥1 個 `stImage`。

## 4. 邊界與錯誤(GUI 友善)
| 情境 | 行為 |
|------|------|
| `obj_emb is None`(結果無物件 embedding) | `st.info("此結果無物件 embedding,無法找相似。")`,不 crash |
| 尚無 ② 結果 | `st.info("請先②套用偵測產生結果。")` |
| 物件數 < 2 | 提示「至少需 2 個物件才能找相似」 |

## 5. Acceptance Criteria(真實行為;給 `/pm` 落 `@pytest.mark.e2e`)
> 用 `yolo_defect_at_nmin`(scratch 紅塊左上 ×8 + stain 藍塊左下 ×8,兩群視覺明顯可分)。
> 建模語義可用「物件類別」(找相似**不需 head**);② 套用同 root → 結果含 obj_emb + 每物件 label。

- **AC-S1(相似佇列同群 + 排除自身)**:③ 找相似設 `ref_idx=0` → 讀參考容器的「參考:<L>」得標籤 L
  (scratch 或 stain)→ 佇列渲染 ≥1 縮圖,且**佇列首項(最像)的 label == L**
  (真實 DINOv2:同群 cosine > 跨群);佇列**不含 ref_idx=0 自身**。
- **AC-S2(切換參考改變佇列)**:把 `ref_idx` 改到一個**另一群**的物件(label≠L)→ 佇列首項 label 變成新群
  (相似排序真的跟著參考走,非固定)。
- **AC-S3(無 embedding 友善)**:(可選,若易構造)無 obj_emb 的結果 → 顯示「無物件 embedding」提示、無 stException。

> 真實行為鐵則:AC-S1/S2 斷言「最像參考的**真的**是同群物件」,非只查 element 存在。

---
## 增補(UX 精修):參考依據切換(③/M9/M10 一致)
- 新 widget:`anomaly_sim_ref_mode`(radio「類別 / 具體物件」)。
  - 類別:`anomaly_sim_ref_class`(selectbox 類別名)→ ref = `class_centroid(obj_emb, labels, class)`。
  - 具體物件:`anomaly_sim_ref_idx` 改成 **selectbox 顯示「第N · <label> · <檔名>」**(不再裸索引)+ 縮圖預覽。
- M9/M10 同步:`anomaly_batch_ref_mode`/`anomaly_batch_ref_class`、`anomaly_watch_ref_mode`/`anomaly_watch_ref_class`。
- E2E 改以「選類別」為主驗收(E2E 已知所選類別,免解析):選類別 scratch → 佇列多數 scratch;切 stain → 翻 stain。

---
## 設計修正(2026-07-05,多 agent 討論 + 使用者拍板;取代上方「參考依據切換」增補)
- **移除「像某一類」**:class_centroid 吃②的 label 是 silent-wrong(②常無標→退化全體平均)且與 M11 預標重疊。
  「找某已知瑕疵種類」一律導流「🏷️ 預標」。`similarity.class_centroid` 保留為純函式(已測),GUI 不再使用。
- 「找相似」改名「**🔎 找同款**」= 純 by-example:挑**一顆**當範本 → cosine 排序。
- **參考來源顯性切換**(>1 來源才出 radio):`anomaly_{prefix}_ref_src` =「② 這次掃描結果」(anomaly_apply_result)/
  「① 建模已知範例」(anomaly_train_result,同 session 免持久化;載入舊模型則無①)。
- widget 契約(新):`anomaly_sim_ref_src` / `anomaly_sim_ref_idx`(selectbox「第N·label·檔名」+ 縮圖)/
  `anomaly_sim_queue`。佇列排除範本自身**僅當參考來自②**(①範本不在搜尋集,不排除)。
- E2E 改驗 by-example:挑 scratch 物件 → 佇列 scratch 多數;換挑 stain 物件 → 翻 stain
  (`test_by_example_queue_follows_reference`)。M9 同步:`anomaly_batch_ref_idx` selectbox 挑範本(g8)。
