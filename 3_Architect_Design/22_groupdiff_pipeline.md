# 22 — groupdiff_pipeline 技術設計(雙資料夾 → 群差異結果包 + 匯出)+ GUI 接線

對應 PRD:[2_PO_PRD/goodbad_patch_diff_prd.md](../2_PO_PRD/goodbad_patch_diff_prd.md) · 里程碑 M19 · **Tier B(檔案 I/O + 整合)**;GUI 接線亦 Tier B(落 `scripts/app.py`)

## 1. 目的(Purpose)

把 M19 串成一次可跑完的分析:兩個影像資料夾 → 整張影像記錄(復用 M8)→
DINOv2 patch 特徵(復用 `patch_features`,含 `.lv_cache` 快取)→ grid 對齊檢查 →
20(統計)→ 21(區域/代表)→ **單一結果包 dict**;外加**匯出報告**(熱圖 PNG /
regions.csv / 對照圖 / summary.json)。GUI 只做薄呈現,一切可斷言邏輯下沉本模組。

回答 PRD 開放問題:
- **(Q2 grid 一致性)** 沿用既有 `keep_aspect=True` 路徑(=`embed_objects_patch` 原樣復用,
  含快取格式):同長寬比 → 同 grid → 位置可比;**混 grid 走多數決**,非多數 grid 的圖
  跳過並警示(對齊性警示=PRD Should-10,一起落)。不改 `patch_features`(零回歸)。
- **(Q5 熱圖底圖)** Good/Bad 各選一張代表(Good=全 patch 平均 `dist_good` 最小=最典型;
  Bad=區域貢獻總和最大),GUI 可切換(Should-11)。
- **(Q6 記憶體)** 500+500 × 256 patch × 384 維 fp32 ≈ 400MB,一次載入可行;
  堆疊用 float32,統計內部升 float64(20 的契約)。
- **(Q7 散點旁證)** global embedding = patch 特徵逐 patch 平均(meanpool,免再過模型),
  PCA 由 numpy SVD 下沉本模組(`pca2d`),GUI 畫散點;**不用群標籤做監督投影**(PRD Should-9)。

## 2. I/O 契約

### 2.1 `scripts/groupdiff_pipeline.py`

依賴(顯式):`interaction.discover_whole_images`、`patch_features.embed_objects_patch`、
`safe_io.partition_readable`、`object_eval.list_images` / `dataset_cache_dir`、
`group_patch_stats`(20)、`diff_regions`(21)、`anomaly_heatmap.render_heatmap`(匯出)、
numpy / PIL。不 import `app.py`。

```python
def run_groupdiff(
    good_dir, bad_dir, *,
    model: str = "dinov2_vits14", target_res: int = 224,
    n_perm: int = 200, alpha: float = 0.05, stab_q: float = 0.95,
    top_k: int = 5, n_reps: int = 3, seed: int = 0,
    extractor=None,        # 測試注入(簽名同 patch_features;None → 真實 DINOv2)
    cache_root=None,       # 測試注入;None → dataset_cache_dir(dir, f"groupdiff_{model}_{target_res}")
    progress=None,         # callable(frac: float, text: str);frac 單調不減、終值 ≥ 0.99
) -> dict
```

步驟(釘死順序):
1. 兩夾各:`list_images` → `partition_readable` → `discover_whole_images(label="good"/"bad")`。
   可讀影像任一夾 < 2 → ValueError(訊息含該資料夾路徑與張數)。
2. 各夾 `embed_objects_patch(..., cache_dir=<cache_root>/good|bad)`(快取沿既有
   `stem__0.npz` 慣例;progress 轉發)。
3. **grid 多數決**:兩夾合併統計 `grid` 眾數 → 非眾數 grid 的記錄剔除(壞圖 (1,1)
   fallback 自然被剔)。任一張被剔 → `alignment_warning=True` + warn_msgs 記
   「N 張影像因尺寸/長寬比不一致被跳過」。剔除後任一群 < 2 → ValueError。
   另:兩夾原圖長寬比 max/min 差 > 5% 也置 `alignment_warning=True`(不剔圖,只警示)。
4. 堆疊 (Ng,P,D)/(Nb,P,D) → `group_diff_stats`(20)。
5. `z`/`stability` reshape (gh,gw) → `extract_regions`(21,`z_thresh=threshold_z`)→
   每區 `pick_representatives` + `region_summary`,把代表**索引換成原圖路徑**。
6. 組 result dict(鍵名即契約):

```python
{"grid": (gh, gw), "score_map": (gh,gw), "z_map": (gh,gw), "stability_map": (gh,gw),
 "verdict": bool, "p_global": float, "threshold_z": float,
 "regions": [{**21 的區域鍵, "text": str, "pct": int,
              "good_paths": [str], "bad_paths": [str],       # 代表圖(原圖路徑)
              "bbox_px": (x0, y0, x1, y1)}],                 # 相對座標 0-1(乘原圖寬高即像素)
 "good_paths": [str], "bad_paths": [str],                    # 有效樣本(=堆疊順序)
 "base_good": str, "base_bad": str,                          # 熱圖底圖代表(§1 Q5)
 "gmean_good": (Ng, D), "gmean_bad": (Nb, D),                # meanpool global(散點旁證用)
 "n_good": int, "n_bad": int, "skipped": {"unreadable": int, "grid": int},
 "alignment_warning": bool, "warn_msgs": [str],
 "meta": {"model", "target_res", "n_perm", "alpha", "stab_q", "seed", "top_k"}}
```

`bbox_px`:`bbox_grid=(r0,c0,r1,c1)` → `(c0/gw, r0/gh, c1/gw, r1/gh)`(0-1 相對;
keep_aspect 縮放為等比,相對座標直接映射原圖)。

```python
def region_compare_images(result, region, *, n_each=3, margin=0.5) -> dict
# {"good": [PIL.Image], "bad": [PIL.Image]}——代表圖依 bbox_px 裁「區域+margin 倍
# 上下文」放大圖,紅框標出區域本體。壞路徑跳過(safe_open_image None → 略)。

def pca2d(gmean_good, gmean_bad) -> dict
# {"xy_good": (Ng,2), "xy_bad": (Nb,2), "evr": (2,)}——合併後中心化 SVD 取前 2 主成分
# (非監督;不吃群標籤)。樣本 < 3 → ValueError。

def export_report(result, out_dir) -> dict
# 寫 out_dir/(不存在則建;冪等覆寫):
#   heatmap_good.png / heatmap_bad.png  — render_heatmap(z_map, 底圖, vmin=0,
#       vmax=max(z_map.max(), threshold_z));verdict=False 時檔名加後綴 "_未達顯著"
#   regions.csv — 欄位:rank,peak_z,mean_z,stability_pct,text,
#       bbox_px_x0,bbox_px_y0,bbox_px_x1,bbox_px_y1(0-1 相對座標,utf-8-sig)
#   region_{rank:02d}.png — 該區 good/bad 對照圖橫排(region_compare_images 拼接)
#   summary.json — verdict/p_global/threshold_z/n_good/n_bad/skipped/warn_msgs/meta
# out_dir 位於 good_dir 或 bad_dir 內(含相等)→ ValueError(絕不寫來源)。
# 回 {"out_dir": str, "files": [str]}(files=實際寫出的檔案)。
```

### 2.2 GUI 接線(`scripts/app.py`,新第 10 工具)

- 工具列 `st.segmented_control` options 加 **「🧪 差異探索」**(排「🎯 以樣搜樣」後);
  「✨ 功能地圖」popover 補一行(比兩群局部差異、找製程線索)。
- 新 `_groupdiff_ui()`;所有 key 前綴 **`gpd_`**:
  - 標題「🧪 Good/Bad 差異探索」+ ❓ `st.popover`:一句定位(**群對群**、與瑕疵偵測
    「單張 vs 正常群」不同)、前提(拍攝對齊)、誠實界線(無穩定差異就明講)。
  - 輸入:Good/Bad 兩列 [輸入框|📁] 同列慣例(仿 Compare `browse_a/b`;
    keys `gpd_good_dir`/`gpd_bad_dir` + `gpd_browse_good`/`gpd_browse_bad`)。
    模型 selectbox(既有模型清單 helper,預設 `dinov2_vits14`)。
    進階 popover:top_k / n_perm / alpha / seed(預設同 §2.1)。
  - 「🔬 分析差異」按鈕 → `run_groupdiff`(`st.progress` 接 progress;spinner 提示
    「首次先載入模型(約 10~30 秒)」慣例)→ 結果 + 參數簽章存
    `st.session_state["gpd_result"]`(兩夾路徑+model+參數變 → 結果失效重跑)。
  - 結果呈現(依 verdict 分流,**G2 的 UI 端**):
    - True → `st.success`:「✅ 兩群存在穩定的局部差異(p=…)」;
    - False → `st.warning`:「⚠ 未發現穩定差異——目前兩群在局部特徵上無法可靠區分
      (p=…)。不建議把下方熱圖當製程線索。」,熱圖收進**預設收合** expander。
    - `alignment_warning` → `st.warning(warn_msgs)`。
  - 熱圖:底圖 `st.segmented_control`(「Good 代表 / Bad 代表」,key `gpd_base`)→
    `render_heatmap(z_map, 底圖, vmin=0, vmax=…)` `st.image`。
  - Top-K 區域:`st.selectbox`(「第 n 名 · z=… · Bad xx%」,key `gpd_region`)→
    區域詳情:白話句 + `region_compare_images` 兩欄(左 Good 代表 / 右 Bad 代表,紅框)。
  - 旁證 expander「📎 整體旁證(PCA,非監督)」:`pca2d` → plotly 散點(good/bad 兩色)
    + caption 說明「僅供旁證;軸不吃 Good/Bad 標籤」。
  - 匯出:輸出資料夾 [輸入框|📁](key `gpd_out_dir`)+「📦 匯出報告」→
    `export_report` → `st.success(檔案數 + 路徑)`。

## 3. 資料流

```
good_dir ─ list_images → partition_readable → discover_whole_images ┐
bad_dir  ─ 同上 ─────────────────────────────────────────────────────┤
   embed_objects_patch(.lv_cache 快取) → grid 多數決 → 堆疊 (N,P,D)
   → 20 group_diff_stats → 21 extract_regions/pick_representatives/region_summary
   → result dict → GUI 呈現 / export_report(out_dir;絕不寫來源)
```

## 4. 邊界條件與錯誤處理

| 情況 | 預期行為 |
|---|---|
| 資料夾不存在 / 可讀 < 2 | ValueError(訊息含路徑) |
| 壞圖(safe_io 補零 (1,1) grid) | grid 多數決自然剔除,計入 skipped |
| 兩夾 grid 眾數不同(整批不相容) | 以合併眾數為準;若剔除後任一群 < 2 → ValueError |
| verdict=False | result 照回(regions 依 threshold_z 自然趨空);GUI 走警示分流 |
| top_k=0 / regions 空 | GUI 顯示「無達標區域」,匯出仍寫 heatmap/summary |
| out_dir 在來源內(或等於) | ValueError |
| 匯出時代表圖已被刪 | region_compare_images 跳過該張;全滅則該區 png 不寫、files 如實 |
| 進度回呼拋例外 | 不吞(往上炸,GUI 自己保證回呼安全) |

## 5. Acceptance Criteria

**單元/整合(`tests/test_groupdiff_pipeline.py`;tmp 檔案樹 + 注入 extractor,免真模型)**

共用 fixture:`_mkimgs(dir, n, size=(56,56))` 產小 PNG;注入 extractor 依「影像內容
簽名」回傳決定論特徵(grid 4×4、D=16):Good 基底 + 雜訊;檔名含 "bad" 者在
patch 5 加位移(與 20 的 AC fixture 同手法);並記錄呼叫次數。

- **AC1(端到端合成)**:8 good + 8 bad → `verdict is True`;rank1 區域 `peak_idx==5`;
  `z_map.shape==(4,4)`;`regions[0]["text"]` 含 "%";`good_paths`/`bad_paths` 長度 8/8;
  `bbox_px` 全在 [0,1]。
- **AC2(null 誠實)**:兩夾都放 good 樣式 → `verdict is False`。
- **AC3(快取)**:AC1 同參數重跑 → extractor 呼叫次數第二輪 == 0(`.npz` 命中),
  結果 verdict 不變。
- **AC4(grid 不一致剔除+警示)**:摻 1 張讓注入 extractor 回 (2,2) grid 的圖 →
  該圖被剔、`skipped["grid"]==1`、`alignment_warning is True`、warn_msgs 非空,
  其餘照常出結果。
- **AC5(防呆)**:不存在資料夾 → ValueError 含路徑;單張資料夾 → ValueError;
  `export_report(result, good_dir 內路徑)` → ValueError。
- **AC6(匯出往返 + 零寫入)**:export_report → heatmap_good.png / heatmap_bad.png /
  regions.csv / region_01.png / summary.json 存在且非空;csv 行數 == len(regions)+1、
  首欄 rank 遞增;summary.json 讀回 verdict/p_global 與 result 一致;
  **good_dir/bad_dir 檔案清單(rglob)前後完全相同**。
- **AC7(進度)**:progress 收到的 frac 單調不減且終值 ≥ 0.99,text 非空。
- **AC8(旁證)**:`pca2d` 輸出形狀 (Ng,2)/(Nb,2)、evr 遞減;樣本 <3 → ValueError。
- **AC9(對照圖)**:`region_compare_images` 回 good/bad 各 ≤ n_each 張 PIL、尺寸 >0;
  刪掉一張代表圖檔後呼叫 → 該張被跳過不炸。

**GUI / 整合(真實 E2E,`tests/e2e/test_groupdiff_e2e.py`,`@pytest.mark.e2e`,真實 DINOv2)**

合成影像 fixture(**回答「能不能產生影像測試」——能,答案已知才可客觀斷言**):
224×224 灰底 + 每張獨立 seed 紋理;Bad 每張在**固定區域**(像素 (56,56)-(112,112))
畫暗色瑕疵塊;good_dir 16 張、bad_dir 12 張、另留 8 張 good 做 null 對照夾。

- **AC-G1(植入命中,端到端)**:GUI 選兩夾 → 分析 → `st.success` 含「穩定的局部差異」;
  匯出到 tmp → regions.csv rank1 的 `bbox_px` 中心落在植入區域內(±1 patch 容差);
  region_01.png 存在非空。
- **AC-G2(null 誠實,端到端)**:good 前 8 張 vs 後 8 張(同源兩夾)→ 分析 →
  頁面出現「未發現穩定差異」警示文案;成功文案**不得**出現。
- **AC-G3(來源零寫入)**:AC-G1/G2 全程後,兩個來源資料夾檔案清單與 mtime 前後不變。

## 6. 給 /pm 的提示

- 注入 extractor 的簽名 = `patch_features` 的 extractor 契約:`extractor(proc_pil) ->
  (feats (P,D), (gh,gw))`;它進 `embed_objects_patch` 後會被逐物件呼叫。
  「依檔名 bad 決定位移」在 extractor 內做不到(它只看到 PIL)——改成**依影像內容**
  (fixture 給 good/bad 畫不同像素簽名,如 bad 左上角 4px 純黑),extractor 讀該角
  平均亮度分流。AC3 計數用閉包 list。
- E2E 沿用 `tests/e2e/conftest.py` harness;**跑前先清殘留 streamlit server**
  (`.claude/skills/visuallatent-e2e` 鐵則);成功訊號選唯一文案(避免說明文字誤觸,
  [[streamlit-e2e-pitfalls]]);兩次「分析」之間等 spinner/進度消失而非 wait_idle。
- AC-G1 的「±1 patch 容差」:植入區域相對座標 (0.25,0.25)-(0.5,0.5),grid 16×16 時
  1 patch = 1/16 ≈ 0.0625;斷言 rank1 bbox_px 中心 ∈ (0.25−0.0625, 0.5+0.0625) 兩軸。
