# M22 GUI 接線 — 🧭 晶圓地圖(第 12 工具,Tier B)

> PRD [2_PO_PRD/wafer_spatial_signature_prd.md](../2_PO_PRD/wafer_spatial_signature_prd.md)。
> 組裝 29 `wafer_grid` / 30 `spatial_sig` + M20 `meta_join`。獨立工具分頁
> (掛法定案=PRD 開放問題 5:它吃②結果但視覺重心完全不同,獨立分頁;
> `tool_switch` 清單加「🧭 晶圓地圖」,列「🩺 資料體檢」之後)。
> app.py 只做讀取/組裝/顯示,判定與整形零複製(單一真相源=引擎)。

## 1. 目的

把瑕疵偵測②的判定攤回晶圓地圖,給每片誠實的空間簽名結論,支援 lot 並排與匯出;
缺欄位時逐層明講(G6),絕不畫偽圖。

## 2. 介面配置與 keys(契約)

主畫面(沿 M19/M20 慣例,不放側欄):

| 元件 | key | 行為 |
|---|---|---|
| CSV 路徑 | `wmap_csv` | text_input;🩺 已填 `adt_csv` 時預填同值(僅預填,不綁定) |
| 欄位對應 | `wmap_wafer_col` / `wmap_x_col` / `wmap_y_col` / `wmap_lot_col` | 4 個 selectbox,選項=CSV 欄名+`"(無)"`;預設=`guess_fields` 命中值,沒中=`"(無)"` |
| 同屬一片勾選 | `wmap_single_wafer` | checkbox「這批影像同屬一片晶圓(我確認)」;**僅 wafer 欄=(無) 時渲染**(G6:使用者斷言,工具不猜) |
| 產生鈕 | `wmap_go` | 前置齊備才 enabled |
| 對不上摘要 | `wmap_gaps` | container:unmatched/missing_coord/missing_wafer 計數與前幾例 |
| 每片區塊 | `wmap_wafer_{i}` | container:plotly 地圖 + 結論行 + method expander |
| 匯出目錄 | `wmap_export_dir` + `wmap_export_btn` | 另選目錄;PNG(29 `wafer_map_png`,PIL 自繪,零新依賴)+ 結論 CSV |

## 3. 資料流與 G6 誠實降級鏈(語彙釘死,E2E 錨字)

依序檢查,**命中即顯示該層訊息+停止渲染後續控件**(不 crash、不畫偽圖):

1. 無 `anomaly_apply_result` → `st.info` 含 **`"請先在瑕疵偵測②執行套用"`**。
2. `wmap_csv` 空或檔案不存在 → info 含 **`"無 metadata CSV，無法取得座標"`**。
3. CSV 讀入(`meta_join.load_metadata_csv`)後渲染欄位對應;
   x 或 y 欄=(無) → warning 含 **`"缺座標欄位，無法做空間分析"`**,產生鈕 disabled。
4. wafer 欄=(無) 且未勾同屬一片 → warning 含 **`"無法分片"`**,產生鈕 disabled
   (勾了 → 走 `assume_single_wafer=True`)。
5. lot 欄=(無) → 正常執行,lot 並排區顯示 caption 含 **`"無 lot 欄位"`**(功能收起)。

產生:②`records` → per-item {item_id,path,flag=verdict=="bad",score} →
`meta_join.join_metadata(rows, paths, sha_of)` 組 `path→row` →
29 `build_wafer_grids` → 每片 30 `spatial_signature`(預設參數)→ 渲染:

- **對不上摘要(G4)**:`unmatched/missing_coord/missing_wafer` 各顯示
  `對不到列 X 張`/`缺座標 Y 張`/`缺 wafer 值 Z 張`(0 也顯示,語彙固定)。
- **每片**:plotly scatter(die 方塊記號;flag=1 紅 `#dc3c3c`、0 藍 `#78a0dc`;
  hover=檔名+score;`yaxis autorange reversed` 對齊影像座標)+ 結論行白話:
  `clustered→「聚一團(可信)」`、`edge→「偏邊緣(可信)」`、
  `linear→「線狀(可信)」`、`none→「看不出模式」+reasons`;
  method line 收「詳細(統計)」expander(含 exploratory)。
- **lot 並排**:依 lot 分組,每列最多 4 片 `st.columns`;每 lot 最多顯示 12 片,
  超過顯示 `「共 N 片，顯示前 12 片」`(no silent caps)。
- **Should(點 die 看縮圖)**:plotly `on_select="rerun"` 取選點 → 地圖下方顯示
  該 die 的原圖縮圖(`safe_open_image`)+檔名。E2E 不強制驗(box select WebGL
  風險已知),單元不可達 → 以手動驗收;若實作成本爆表可降候選(記決策日誌)。

匯出:`wafer_map_png` 逐片存 `wmap_<wafer_id>.png` + `wafer_signatures.csv`
(欄:wafer_id,lot,n,k,verdict,p_clustered,p_edge,p_linear,reasons)到另選目錄;
成功訊息含 **`"已匯出"`** 與檔數。零寫來源資料夾([[no-dataset-writes]])。

## 4. 邊界條件與錯誤處理

- CSV 無任何列/欄 → `load_metadata_csv` 的 ValueError → `st.error` 訊息本文。
- 全部對不上(wafers 空)→ warning 含 `"沒有任何影像對得上座標"`,不渲染地圖區。
- 引擎 ValueError → `st.error` 顯示訊息本文(不吞錯、不裸 traceback)。
- 匯出目錄=來源資料夾(或其子目錄)→ 拒絕並提示另選(沿 prelabel C6 慣例)。

## 5. Acceptance Criteria(E2E,`tests/e2e/test_m22_wafer_map_e2e.py`,@pytest.mark.e2e)

真實 Streamlit+Playwright+真 DINOv2。專用 fixture:50 張 YOLO 影像
(42 normal+**8 defect**,兩片:W1=17 normal+8 defect、W2=25 normal)+ 植入 CSV:
W1 的 8 張 defect 影像坐**遠圈座標**(半徑 10 的圓環;normal 聚中心 ±2)——
統計穩健性定案:8 個遠圈缺陷讓 edge 的 MC p 觸底(5 個角落缺陷 p≈0.003 偏臨界,
會做 flaky);W2 全 normal(k=0 → 缺陷太少 → 看不出模式,deterministic)。
lot 欄:兩片同 lot `L1`。①建模→②套用→autoseed+就地重評(沿 M21 先例,
確保 defect 全 flag)→ 切「🧭 晶圓地圖」。**契約:每片地圖 flag=1(紅)為
第一個 trace**(E2E 據此數紅點)。

- **AC-E1(G3+G1 主流程)**:填 CSV → 欄位對應自動命中(selectbox 顯示猜中欄名)
  → 產生 → W1 區塊含 `"偏邊緣"`;W2 含 `"看不出模式"`;地圖 SVG 存在且
  W1 紅點數==8(plotly 第一 trace 點數,與②判定一致)。
- **AC-E2(G4 對不上明講)**:CSV 故意缺 2 張影像的列+1 列缺 y →
  `wmap_gaps` 內含 `"對不到列 2"` 與 `"缺座標 1"`。
- **AC-E3(G6 降級鏈逐層)**:清空 CSV 欄位 → `"無 metadata CSV"`;
  給只有 filename 欄的 CSV → `"缺座標欄位"` 且產生鈕 disabled;
  wafer 欄選 `(無)` 未勾確認 → `"無法分片"` 且 disabled;勾「同屬一片」→
  enabled 且產生後 wafer_id 顯示 `(單片)`。全程 `_no_exception`。
- **AC-E4(G5 lot 並排+匯出)**:lot 標題 `L1` 出現且 W1/W2 兩片區塊皆渲染;
  匯出到 tmp 目錄 → 檔案恰為 {wmap_W1.png, wmap_W2.png, wafer_signatures.csv},
  CSV 讀回 2 列、W1 列 verdict=="edge"、W2 列 verdict=="none";
  來源資料夾檔案清單+bytes 前後 ==(零寫入)。
- **AC-E5(可達性)**:以上全在「🧭 晶圓地圖」工具分頁完成;工具切換鈕文字
  比對沿 pitfalls #8(emoji 拆節點:`get_by_text("晶圓地圖", exact=False)`
  範圍限 `.st-key-tool_switch`)。

E2E 陷阱備忘適用:錨字唯一(「偏邊緣」「看不出模式」「無法分片」等不得出現在
說明文字);expander 互動要 summary 真開(pitfalls #12);成功訊號等收斂再斷言。
