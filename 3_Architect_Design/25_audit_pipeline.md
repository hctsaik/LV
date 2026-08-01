# 25 — audit_pipeline 技術設計(資料夾+CSV → 體檢報告包 + 匯出)+ GUI 接線

對應 PRD:[2_PO_PRD/dataset_audit_prd.md](../2_PO_PRD/dataset_audit_prd.md) · 里程碑 M20 · **Tier B**;GUI 接線亦 Tier B(`scripts/app.py` 第 11 工具)

## 1. 目的

一顆鈕把六訊號跑成一份可匯出的量化報告。**全部訊號復用既有函式**:
`update_manifest`(sha256/phash 增量)、`find_duplicate_pairs_phash/embedding`、
`compute_outlier_scores`、`discover_images(_classifier)`、`partition_readable`、
`_utils.load_model`(全圖 embedding,可注入)。回答 PRD 開放問題:
- **(Q1 佈局)** 一級子目錄名(lower)∈ {train,val,valid,test} → 各為一個 split 夾;
  否則 root 自己是單一 split。每個 split 夾內:`folder_has_annotations` 或
  images/+labels/ → `discover_images`;否則 `discover_images_classifier(allow_flat=True)`。
- **(Q2/Q3 門檻)** phash 漢明 ≤4、emb cosine ≤0.05、outlier k=5(全部沿既有預設),
  method line 帶出。
- **(Q4 sha 來源)** `update_manifest` 增量計算(**只取值、不 write**,不動 viz 的
  embedding_refs 簿記);第二次跑 (size,mtime) 未變零重算。
- **(Q5 HTML)** 沿體檢卡 f-string 自含式 HTML 手法。
- **(Q6 鍵空間)** 23 的 group_counts max_groups=20 + "(其他)"。
- **(Q7 成本)** embedding 自建**內容定址快取**(sha 為鍵)+ `use_embedding=False`
  開關(略過 emb 近重複與離群,節標「未計算」)。

## 2. I/O 契約

### `scripts/audit_pipeline.py`

依賴(顯式):`visualize_embeddings.discover_images/discover_images_classifier`、
`label_formats.folder_has_annotations`、`safe_io.partition_readable`、
`manifest.update_manifest`、`interaction.find_duplicate_pairs_phash/
find_duplicate_pairs_embedding/compute_outlier_scores`、`object_eval.dataset_cache_dir`、
`meta_join`(23)、`audit_report`(24)、numpy。不 import app.py。

```python
def run_audit(root, *, model: str = "dinov2_vits14", metadata_csv=None,
              embed_fn=None,            # 注入(path→1D ndarray);None→_utils.load_model(model)
              use_embedding: bool = True,
              phash_max_hamming: int = 4, emb_max_distance: float = 0.05,
              outlier_k: int = 5, top_n: int = 10, max_pairs: int = 200,
              cache_root=None,          # 測試注入;None→dataset_cache_dir(root, f"audit_{model}")
              progress=None) -> dict
# 步驟:
# 1) root 不存在 → ValueError(含路徑)。佈局辨識(Q1)→ split 夾清單。
# 2) 逐 split 夾 discover → records{path,split,label};合計 0 張 → ValueError。
# 3) partition_readable → n_unreadable;可讀 <2 → ValueError。
# 4) 逐 split 夾 update_manifest(夾, 該夾可讀 records) → sha256/phash(不寫檔)。
# 5) use_embedding:快取 <cache_root>/emb_cache.npz(keys=sha、vecs=float32 L2)→
#    缺的逐張 embed_fn(進度回報),回寫快取。同 sha 只算一次。
# 6) 訊號:phash_pairs(全域)、leak_pairs(cross_split_only=True)、
#    emb_pairs、outlier_scores(candidates_in_reference=True)。
# 7) metadata_csv → 23:load_metadata_csv + join_metadata(sha_of=idx→sha)+
#    每個 fields 欄位 group_counts。CSV 相關 ValueError 直接上拋(不吞)。
# 8) 24 build_report。
# 回:{"report": <24 輸出>, "records": [{path,split,label,sha}...],
#     "n_images": int, "n_unreadable": int, "root": str,
#     "meta_info": 23 join 輸出|None}
# progress:單調不減、終值 ≥0.99、text 非空。

def export_audit(result, out_dir) -> dict
# 寫 out_dir/(不存在則建;冪等覆寫):
#   report.html — 自含式:標題+產生時間+每節(title/value/text/method+items 表,
#                 items 超過 50 列截斷並註明);utf-8。
#   report.json — {"sections","totals","root","params?"} ensure_ascii=False。
#   issues.csv  — utf-8-sig;欄:issue_type,path,split,label,detail;
#                 列=exact_dup 每張額外複本、leakage 每對(兩路徑併 detail)、
#                 label_conflict 每張、outliers 每張(detail=score)、
#                 metadata 未匹配影像每張(issue_type=meta_unmatched)。
# out_dir == root 或在 root 內 → ValueError(絕不寫來源)。
# 回 {"out_dir": str, "files": [str]}
```

### GUI 接線(`scripts/app.py`,第 11 工具「🩺 資料體檢報告」;keys 一律 `adt_`)

- 工具列 options 加「🩺 資料體檢」(排「🧪 差異探索」後);功能地圖補一行
  (整資料集量化體檢,與單張「體檢卡」呼應)。
- `_dataset_audit_ui()`(主畫面,不進 sidebar):
  - 標題 + ❓ popover(定位/六訊號/CSV 格式說明——**不得**含成功 banner 字串)。
  - 輸入:可累加、去重、逐項移除的資料夾清單(`adt_roots`);
    metadata CSV [輸入框|📁](`adt_csv`/`adt_browse_csv`,選填,caption 說明鍵欄);
    Model selectbox(`adt_model`,預設 dinov2_vits14)+
    checkbox「含 embedding 訊號(離群/語意近重複)」(`adt_use_emb`,預設開)。
  - 「🩺 產生體檢報告」(`adt_run`,primary,任一 root 非資料夾時 disabled)→
    st.progress + spinner(首次載模型提示慣例)→ 結果存
    `st.session_state["adt_result"]`(簽章=root/csv/model/use_emb;變更顯提醒)。
  - 完成 banner(唯一訊號):`st.success("✅ 體檢報告完成(共 N 張,見下方各節)")`;
    失敗:`st.error(f"無法體檢:{e}")`。
  - 報告呈現:totals 列(st.metric ×3:影像數/壞檔/split 數)→ 逐節
    `st.metric(title, value 或 "—")` + text + `st.caption(method)`(灰字)+
    items 進 expander(前 50 列 dataframe;Should-6:離群/重複節放縮圖列
    `st.image(width=96)` 前 8 張;Should-7:「➕ 全部加入策展購物車」鈕
    (`adt_cart_{key}`),復用既有 cart append 慣例)。
  - 匯出:`adt_out_dir`/`adt_browse_out` + 「📦 匯出報告」(`adt_export`)→
    export_audit → st.success(檔案數+路徑)。

## 3. 邊界條件與錯誤處理

| 情況 | 行為 |
|---|---|
| root 不存在 / 0 張 / 可讀 <2 | ValueError(含路徑/數量) |
| 單一 split | coverage 節「不適用」(24 處理),其餘照常 |
| 平鋪無標籤 | label 全 "" → conflict 0、coverage 不適用 |
| use_embedding=False | 不載模型、不呼叫 embed_fn;emb 兩節「未計算」 |
| CSV 錯誤(空/無鍵欄) | ValueError 上拋 → GUI st.error(不產半份報告) |
| pairs 達 max_pairs | 24 的 text 註明顯示上限(no silent caps) |
| out_dir 在 root 內 | ValueError |
| 匯出時 items 空 | issues.csv 只有表頭(合法) |

## 4. Acceptance Criteria

**單元/整合(`tests/test_audit_pipeline.py`;tmp 檔案樹 + 注入 embed_fn,免真模型)**

共用 fixture `_mk_tree(root)`(數字全手算;28×28 px 小圖,每張獨立 rng 紋理):
```
train/classA: base 8 張 + img0 複本 ×3(dup3)+ outlier.png(純色,embed_fn 給遠向量)
train/classB: base 6 張 + conflictcopy.png(= train/classA/img1 複本)
train/classC: base 2 張
val/classA:   base 3 張 + leakcopy.png(= train/classA/img2 複本)+ corrupt.png(文字寫入)
```
→ 手算:n_unreadable==1;exact extras==5(3+1+1);leak_pairs==1;
conflict groups==1;val missing==["classB","classC"];可讀 n_images==8+3+1+6+1+2+3+1=25。
注入 embed_fn:讀像素均值決定——outlier.png(純色)回遠端向量,其餘回基底+小雜訊
(內容 seed 決定論);並記呼叫次數。

- **AC1(植入數字全對)**:run_audit → report sections:exact_dup value==5、
  leakage==1、label_conflict==1、class_coverage value==2 且 val missing==
  ["classB","classC"]、unreadable==1;totals n_images==25。
- **AC2(metadata)**:CSV(filename,tool;T1/T2 分佈手算;5 列 bogus;2 張圖不列入)
  → metadata 節:unmatched_row_idx 長 5、n_unmatched_images==2、
  tool 分組數字==手算。
- **AC3(無 CSV)**:同 AC1 且無 metadata 節。
- **AC4(關 embedding)**:use_embedding=False → embed_fn 呼叫 0 次;
  outliers/emb_near_dup 節 value None、text 含「未計算」;其餘同 AC1。
- **AC5(embedding 快取)**:同參數重跑 → 第二輪 embed_fn 呼叫 0 次(sha 內容定址);
  首輪呼叫次數 == 不同 sha 數(複本不重算)。
- **AC6(匯出往返+零寫入)**:export_audit → 三檔存在非空;report.json 讀回
  exact_dup value==5;issues.csv 列數 == 各節 items 總和 +1(表頭);
  **root rglob+size 前後全等**;export 到 root 內 → ValueError。
- **AC7(單 split/平鋪)**:只有 class 夾無 train/val 的 root → 跑通,
  coverage 節 text 含「不適用」;平鋪無標籤資料夾 → conflict value==0。
- **AC8(防呆+進度)**:root 不存在 ValueError 含路徑;單張資料夾 ValueError;
  progress 單調不減、終值 ≥0.99、text 非空。
- **AC9(離群命中)**:outliers items 第一名 path 以 "outlier" 結尾(注入 embed_fn
  的遠向量真的被 compute_outlier_scores 排最前)。

**GUI / 整合(真實 E2E,`tests/e2e/test_dataset_audit_e2e.py`,`@pytest.mark.e2e`,真實 DINOv2)**

fixture 同構(224px 紋理圖版本;outlier=純色灰圖;CSV 同 AC2 規格)。

- **AC-G1(端到端數字全對,檔案級)**:GUI 選 root+CSV → 產生報告 → 完成 banner →
  匯出 → report.json:exact extras==5、leakage==1、conflict==1、unreadable==1、
  val missing==["classB","classC"]、metadata unmatched rows==5;
  頁面 innerText 含「體檢報告完成」。
- **AC-G2(真實 DINOv2 行為斷言)**:outliers items **前 10 含植入的純色離群圖**
  (以 report.json 斷言,防 DOM 假綠);emb_near_dup 節存在且 method line 非空
  (數值不釘死——真實特徵下近重複數會浮動,誠實不硬斷言)。
- **AC-G3(來源零寫入)**:全程 root rglob+size+mtime 前後全等。

## 5. 給 /pm 的提示

- 單元注入 embed_fn 契約=`embed_fn(path)->1D ndarray`(與 `_utils.load_model` 回傳
  的 callable 同形)。
- E2E 前清殘留 streamlit server(skill 鐵則);完成訊號「體檢報告完成」必須唯一
  (❓ popover/說明文字禁用該字串)。
- AC5 的「同 sha 不重算」:5 張複本共 4 個獨特內容被複製…注意 fixture 的獨特 sha 數
  = 25 − 5(複本)= 20,首輪 embed_fn 呼叫次數應 == 20。
