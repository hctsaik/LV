# 從 VIX 移植到 LV(visuallatent)的前 6 名 GUI 互動功能

> 來源:對 `C:\code\claude\VIX` 的多 agent 調查 + 對立辯論(UX 派 vs 工程派)的收斂結果。
> 目標:把 VIX 那種「可選取、可點看圖、可搜相似」的便利互動,移植進 LV 這個 embedding 分析工具。

---

## 背景:一個關鍵前提

VIX 的精緻 GUI **不是 Streamlit,而是 FiftyOne App + 自訂 plugin**(`src/vix/plugins/vix_review/__init__.py`)。
但互動背後的**邏輯**都在框架無關的純函式裡(`core/scorer.py`、`core/triage.py`、`core/explain.py`、`core/weakness_report.py`)。

**因此移植策略是:純 `core/` 函式可直接搬;FiftyOne 的 operator/panel 機制要用 Streamlit 元件重做**
(`st.plotly_chart(on_select=...)`、`st.dataframe(on_select=...)`、`st.image`、`st.toast`)。

LV 已經具備的「水電管路」:
- 原生 Tk 資料夾選取(`app.py:31`)
- `st.session_state` 跨 rerun 保存結果(`app.py:331`)
- embeddings 磁碟快取(`app.py:301`)
- Plotly 散點 + Model/Method/Split 下拉(`app.py:358`)
- standalone HTML / JSON 下載(`app.py:363`)

缺的是**互動模型**:目前散點是「死路型檢視器」——只能看,不能操作。下面 6 項就是補上這一塊。

---

## 評分標準(1–5)

| 維度 | 說明 |
|---|---|
| **User Value** | 對使用者的價值 |
| **Port Effort** | 移植難易(5 = Streamlit 內建即可,1 = 需脆弱第三方/對抗 rerun 模型) |
| **Mission Fit** | 與「embedding 分析工具」的契合度(5 = 核心,1 = 屬於 labeling/稽核產品) |
| **Risk-safe** | 維護風險(5 = 安全,1 = 長期負債) |

---

## 前 6 名功能

### #1 — 串流式進度條(先做,這是 bug)

- **為何要搬:** 使用者實際踩到——以為 app 當機。根因:`_n_steps = len(models)*4`(`app.py:248`)讓進度條只在粗粒度階段邊界更新,而抽取是同步的 `np.stack([embed_fn(p) for p in tqdm(...)])`(`_utils.py:70-72`),整批跑完才動。
- **評分:** Value 5 · Effort 4 · Fit 5 · Risk 4 → **✅ 先做**
- **要搬的 VIX 函式:** 無(這是 LV 自己的 bug)
- **Streamlit 實作草稿(~1–2 小時):**
  1. 在 `_utils.py` 的 `extract_embeddings` 加一個可選參數 `progress_cb: Callable[[int,int],None]`。
  2. 在 per-image 迴圈內呼叫 `progress_cb(i, n)`。
  3. `app.py` 用 `st.status("Extracting…", expanded=True)` 包一個 `st.progress`。
  4. 傳入閉包:`bar.progress(i/n, text=f"{i}/{n} images — {model_name}")`。
  5. 把原本 PCA/t-SNE/UMAP 的 4 步粗粒度進度放進同一個 `st.status`,讓畫面持續有動靜。

---

### #2 — 點選點/列 → 看真實影像(全場 CP 值最高)

- **為何要搬:** 一個離群點在你看到「它其實是張模糊重複圖」前毫無意義。LV 每個散點**已經帶著** `records[i]["path"]`(`visualize_embeddings.py:114`),離 `st.image` 只差一行。VIX 為此要蓋整個 FiftyOne sample resolver,LV 幾乎免費。
- **評分:** Value 5 · Effort 4 · Fit 5 · Risk 4 → **✅ 採用**
- **要搬的 VIX 概念:** `ctx.ops.open_sample`(`__init__.py:559-568`)的「點 → 看圖」行為
- **Streamlit 實作草稿(~2–4 小時):**
  1. 重用 #3 的 `on_select` 選取串流(單點點選 = 1 個點的選取)。
  2. 取 `records[idx]["path"]`(已是 `Path`)。
  3. 用 `@st.dialog("Selected image")` modal 或 `st.expander` 顯示 `st.image(path, caption=path.name)`。
  4. **進階(FiftyOne 風格):** detector 模式下用 PIL `ImageDraw` 把該圖的 YOLO 框疊上去 → 「看圖 + 看標了什麼」。
  5. 加 prev/next 按鈕走訪目前選取集。

---

### #3 — Lasso / 框選散點 → 對選取操作(核心升級)

- **為何要搬:** 散點是整個 app,目前卻是惰性的(`app.py:358` 無 `on_select`)。選取是把「一張點的圖」變成「這些點,做點什麼」。**這正是你在 FiftyOne 覺得方便的那個功能。**
- **誠實提醒:** FiftyOne 的套索是即時的;Streamlit 是「每次選取觸發一次伺服器 rerun」,功能等價但手感略沒那麼絲滑。另外 FiftyOne 框選的是**物件**,LV 框選的是**整張圖**(見文末「層級」說明)。
- **評分:** Value 5 · Effort 3 · Fit 5 · Risk 3 → **✅ 採用**
- **要搬的 VIX 概念:** `ComputeVisualization` + `ctx.selected` 把選取接到動作(`__init__.py:985-1005`)
- **Streamlit 實作草稿(~0.5–1 天):**
  1. `st.plotly_chart` 加 `key="scatter"`、`on_select="rerun"`、`selection_mode=["box","lasso"]`(原生 Streamlit ≥1.35,**零第三方依賴**)。
  2. 建 trace 時把 record index 放進 `customdata`,方便把選取的 `pointIndex` 對應回 `records`。
  3. 讀 `st.session_state["scatter"]["selection"]["points"]`,組出選取子集。
  4. 用 `st.dataframe` 列出選取(filename / label / split)。
  5. 在選取上方放三顆按鈕:**匯出子集**(→ #6)、**找相似**(→ #4)、**看圖**(→ #2)。
  6. `st.toast(f"{len(sel)} points selected")` 回饋。

---

### #4 — Find-similar(以圖搜圖 / query-by-example)

- **為何要搬:** 閉環——點一個離群點 →「找出所有像它的」→ 就是你的問題群,直接匯出。embedding 矩陣已在記憶體,`sklearn` cosine-NN 一行搞定。VIX 那 200 行 operator glue(`__init__.py:876-1096`)在 LV 縮成十幾行。
- **層級(回答你的問題):** 目前 LV 的 embedding 是**整張 image** → 這版 find-similar 是「**整張圖**相似」。
  **進階(物件層級,VIX 的做法):** detector 模式裁切每個 YOLO 框 → 各自 DINO embed → 建 patch 索引 → 變成「**相似物件**」。對應 VIX `BuildSimilarity`/`FindSimilar`(`__init__.py:877`, `1073`)。
- **評分:** Value 4 · Effort 4 · Fit 5 · Risk 4 → **✅ 採用(先做整圖版,物件版列為後續)**
- **要搬的 VIX 函式:** `scorer.cosine_knn_distance`(`scorer.py:24`)的 NN 概念;Streamlit 端用 `sklearn.neighbors.NearestNeighbors(metric="cosine")`
- **Streamlit 實作草稿(~3–5 小時):**
  1. 任一選取點 → 取其**原始**(投影前)embedding 當 query。
  2. `NearestNeighbors` 對完整 embedding 矩陣 fit 一次,用 `@st.cache_resource` 快取(以矩陣為 key)。
  3. `kneighbors(query, n_neighbors=20)` → indices → `records`。
  4. 用 #2 的縮圖牆(`st.columns` + `st.image`)顯示結果 + 「匯出此群」按鈕。

---

### #5 — 離群排序(outlier sort,免標籤;由 VIX「風險佇列」重新框定)

- **為何要搬:** LV 沒有標籤、沒有順序、沒有「先看這裡」。用平均 kNN cosine 距離排出「最怪的點優先看」,讓使用者看 20 個而不是 2000 個。
- **關鍵——誠實命名:** **叫「離群度 / outlier-ness」,不要叫「風險 / 錯誤」。** 沒有 golden/confidence 脈絡時,把它當成「錯誤裁決」會誤導。VIX 自己也花整段警告使用者「風險不是機率」(`__init__.py:613-615`)——把這份誠實一起抄過來。
- **評分:** Value 4 · Effort 3 · Fit 4 · Risk 3 → **✅ 採用(限誠實命名)**
- **要搬的 VIX 函式:** `scorer.cosine_knn_distance`(`scorer.py:24`,~12 行 NumPy,可逐字搬);`triage.review_queue` 的結構(`triage.py:28`)但**移除 confidence/golden 依賴**,只留 novelty
- **Streamlit 實作草稿(~1 天):**
  1. 讓使用者選一個 split/folder 當**參考集(reference)**,其餘為候選。
  2. 對每個候選點算「到參考集的平均 kNN 距離」= 離群度。
  3. 用 `st.dataframe(on_select="rerun")` 顯示排序表(距離高→低)。
  4. 選一列 → 驅動 #2 的看圖 + 在散點上高亮該點。

---

### #6 — 旗標 / 匯出選取子集 + 誠實三態回饋

- **為何要搬:** 選取的自然下一步——把找到的問題群變成可追蹤的清單,而不是截圖。同時補上 LV 目前缺的「誠實三態」(空 / 完成 / 壞掉)與每個動作的 `st.toast` 回饋,讓上面每個功能都「可信」。
- **界線(避免越界):** **只當「匯出選取」**(zip/CSV 路徑清單)。一旦長成 `vixq:label_suspect` 那種修補待辦清單(`__init__.py:686-873`),就越界成 labeling QA 工具——不要。
- **評分:** Value 3 · Effort 3 · Fit 3 · Risk 3 → **🟡 採用(限「匯出選取」範圍)**
- **要搬的 VIX 概念:** 誠實 empty/clear/broken 三態渲染(`__init__.py:604-612`)、每條路徑都 toast(避免「按了沒反應」)
- **Streamlit 實作草稿(~3–5 小時):**
  1. 選取子集 → 「匯出」按鈕:打包成 zip 或寫出 CSV(檔名/路徑/label/split)。
  2. 每個動作 `st.toast`;成功/空/錯誤分別用 `st.success`/`st.info`/`st.warning`,絕不留空白 widget。
  3. (選用)在 session_state 維護一個「已旗標」集合,可累加、可清除。

---

## 建議落地順序

1. **🔧 #1 串流進度**(bug,先修,~20 行)——「看起來當機」會毒化對其他一切的信任。
2. **🎯 #2 點圖 + #3 lasso/框選 + #6 的 toast/三態**——把惰性散點變成真正的探索面;原生 selection、零脆弱依賴、零新資料模型。**這三個一起就完整複製了你在 FiftyOne 覺得方便的「框選 + 看圖」。**
3. **➕ #4 find-similar(整圖版)+ #5 離群排序**——幾乎免費,因為 embedding 已在記憶體。
4. **🔮 後續:** #4 的**物件層級**版(裁切 YOLO 框 → patch 索引),完整對齊 FiftyOne 的物件級體驗。

---

## 刻意婉拒(避免淪為劣化版 FiftyOne)

| 功能 | 為何不搬原版 | 便宜替代 |
|---|---|---|
| 可點混淆矩陣 | LV 無 predictions/eval,等於畫沒資料的圖 | 讓現有 coverage-gap 散點(`app.py:601-640`)可點,折進 #3 |
| 解釋下鑽(why flagged) | 原版需 calibration 的 `conf_thr/dist_thr`,LV 沒有 | 點圖時順手顯示 kNN 距離 + 最近鄰縮圖 |
| Snapshot / 稽核日誌 | VIX 的脊椎是為了讓**覆核決策**可信;LV 不做決策,沒東西可稽核 | 把 run-config(模型/資料夾/投影參數/時間戳)dump 進現有 JSON |

**共同陷阱:** 這些功能的 pure core 都仍依賴 LV 不產生的 golden/confidence/eval 輸入。採用它們 = 在分析工具底下偷偷蓋 VIX 的 triage/稽核產品。

---

## 附:embedding 層級對照(整張圖 vs 物件)

| | 層級 | 證據 |
|---|---|---|
| **LV(現在)** | **整張 image** | `models.py:24-30` 把整張圖 resize 224×224;ResNet 全域池化、DINOv2 取 CLS → 一張圖一個向量一個點 |
| **VIX** | **物件(每個框)** | `__init__.py:952`「plot/lasso are about objects, not whole scenes」;對每個 YOLO 框裁切算 DINO embedding(patches_field);`FindSimilar` 用 `to_patches(...).sort_by_similarity`(`:1073`) |

要在 LV 做到 VIX 的物件級體驗:detector 模式下裁切每個 bbox → 各自 embed → 建 patch 索引。屬 #4 的進階路線。
