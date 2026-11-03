# 類別相似度／Threshold設計：第三輪多Agent Review

日期：2026-09-14。對象：[設計與驗收契約r3](class-similarity-review-implementation-plan.md)。

## 結論

功能方向合理，原版仍有影響數字、圖片身份與實際操作的缺口，不能僅因已有24項AC就稱為完整。此次以三個獨立agent審查數學／真值、產品／UX、程式整合，由主agent核對本地程式並將必要修正寫回原章節。修正後共有AC01–34、T0–T10，沒有實作功能或宣稱驗收通過。

目前文件可用於建立正式OpenSpec與分段實作；不需要再加入大型分析功能才開始。實際性能、連續拖動事件及真實資料效果仍須依契約測試，設計review不能取代這些證據。

## Findings及處置

P1表示不修會可能使用錯誤資料／版本或產生誤導的評估；P2表示可用性、可重算性或邊界契約不足。表內「已修」只表示設計已修，程式尚未實作。

| ID | 優先 | 原缺口／具體風險 | 已修的契約 | 對應AC |
|---|---|---|---|---|
| R01 | P1 | heldout只鎖t，仍可換K／模型挑最好結果；可重用tuning資料 | 凍結完整操作點、同U/analysis_key/score protocol，查ID及已知來源群交集；來源不足不宣稱獨立驗證 | 24、34 |
| R02 | P1 | 同evaluation ID/version改verdict可能沿用舊PR | truth_digest依內容，evaluation_key含用途／操作點；明定manifest型別／集合／空E0 | 19、20、28 |
| R03 | P1 | 拖動尚未提交便加入／下载，可能使用上個門檻 | draft/committed分離；action驗preview_key及seq；未一致停用action，舊回覆不覆蓋新值 | 16、18、25 |
| R04 | P1 | local勾選、全域選取與selected scope混淆，跨頁漏資料 | inspect/local/global分離，selected固定本面板local∩U，跨頁保留；候選scope另列 | 7、8、17、26 |
| R05 | P1 | 復用其他穩定ID改變身份語意；同名圖可能碰撞crop快取 | 本功能path＋原ROI的ID；invalid列有獨立暫存ID；同crop多來源整份分析拒絕 | 3、9、29 |
| R06 | P1 | 只檢Run token漏掉即時readback；adapter重讀來源又蓋掉人工修標 | 兩條readback路徑失效＋操作前fingerprint；effective label優先；flat fallback不當真標 | 6、22、30 |
| R07 | P2 | 子集報告無法重建未選資料／未顯示類別的統計 | 全U score_index含label/status，完整E0 evaluation_items；單一schema2與明確scope/API | 11、23、31 |
| R08 | P1 | 人工模板只有hash ID難填；未完成一筆時不知道找誰 | 模板含原圖／ROI及unreviewed；凍結E0入口、版本與待確認清單可開證據 | 19、27 |
| R09 | P2 | PR數字看不到FN圖片；窄分布在完整軸上看不清 | TP/FP/FN/TN表格下鑽與box顯示縮放，皆不改母體／分數 | 27、33 |
| R10 | P2 | 有限大向量norm溢位成假0；同分類名可能長成N×C | scale-safe normalize；ties只保存前10類和完整count | 4、32 |
| R11 | P2 | AC只禁N×N持久化，容許先分配再刪；拖動可能重跑舊混淆 | 生產流程任一時刻不超block；驗block shape、call count、峰值記憶體 | 12、16、18 |
| R12 | P1 | 同路徑新內容、或Run後改圖再分析，用新hash替舊raw背書 | snapshot綁embedding generation，分析只核對；未驗來源無PR、label_origin/stale gate、保存／恢復相容操作點 | 9、24、34 |

主agent另外統一core輸入的eligibility/invalid_reasons、items.label、invalid rows對齊與新舊報告schema，避免後續AI需要自己補資料契約。

## 本地證據

- `scripts/app.py::_render_label_quality`：舊guard與昂貴混淆計算需置於新模式分派之後。
- `scripts/app.py::_render_readback_section`、`_apply_readback_overlay`：即時及Run期間兩條標籤變更路徑。
- `scripts/visualize_embeddings.py::discover_images_classifier`：`allow_flat`可產生探索用folder label；不能當已確認單類標籤。
- `scripts/explain_contracts.py::stable_record_key`：其身份定義與本功能不同，不直接復用。
- `scripts/app.py::_crop_and_embed_objects`、`_add_to_export`：crop path與購物車去重使上游同名來源碰撞必須檢查。
- `scripts/cost_curve.py::cost_curve`：異常分數`>=`門檻，不等於誤標margin的`<`，保留原API。

以上是本次讀到的程式狀態。工作區已有其他未提交修改；實作前須再核對符號，不依歷史行號直接改檔。

## 應加入與暫不加入

本輪只增加三項直接補足現有流程的小功能：

1. 點TP／FP／FN／TN查看對應圖片，尤其讓人看到門檻漏掉的誤標。
2. 窄分布的顯示縮放／重設，不重算或重新正規化分數。
3. 可辨識圖片／ROI的評估模板與未完成清單，讓人工真值可以實際建立和往返。

Top-N人工預算仍是後續優先項；每類獨立門檻、自動最佳F1／成本、ANN、另一套熱圖與審查資料庫仍延後。這些不應混入本輪必修缺口。

## 驗收覆蓋與交接要求

| 面向 | 主要AC | 實作時須提供的證據 |
|---|---|---|
| 數學及例外 | 01–05、15、20、32 | 手算fixture、獨立oracle、真實布林membership |
| 身份、來源、失效 | 06、08–10、22、29、30、34 | 兩ROI／同名來源／即時改標／原檔內容變更案例 |
| GUI與候選操作 | 07、16–18、25–27、33 | 真實Streamlit操作、pointer-held、延遲回覆race、跨頁選取 |
| 評估與匯出 | 11、19–24、28、31、34 | 匯入/下載往返、獨立重算每類統計與PR、heldout拒絕案例 |
| 資源與回歸 | 12–14 | 最大block、呼叫次數、記憶體／延遲測量、image/object及原功能回歸 |

測試phase建立具名AC對照，實作phase不得為了綠燈放寬測試。任一必要AC未跑或未通過，報告明列未完成，不以「UI有按鈕」或「檔案存在」代替行為驗證。

本輪只做文件結構與小型手算規格檢查；沒有跑產品E2E、模型性能測試，也沒有實際PR成績。必要修正已回寫設計，是否實作完成仍以T0–T10與AC01–34證據為準。

二次交叉檢查確認數學／UX原缺口已落入契約；整合review再指出的invalid匯出範圍及generation快照問題也已統一到原章節和AC31/34。未新增產品程式碼。
