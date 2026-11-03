# YOLO 案件診斷整合驗證

2026-09-06。對應 OpenSpec `add-yolo-gradcam-case-diagnostics`。此文件區分已實測、仍待整合驗證與實際部署驗收，不以 mock 或原型取代使用者流程驗收。

本版實作與固定案例整合驗收已完成。最後 Python 編譯、`git diff --check` 與 OpenSpec strict validation 均通過；各模型與瀏覽器故事的實測範圍列於下方。

## 已完成的驗證

| 範圍 | 證據 |
|---|---|
| U01–U12 操作原型 | `docs/prototypes/investigation-workflow.html`，Playwright 12／12 故事通過：收案、缺項、檢查、區域修正、回覆及返回 |
| 既有 DINO | 6 個核心測試模組 68 passed；V1 contract、margin、head、Grad-CAM、頁面與三種 resolution。瀏覽器另列，不合併宣稱 |
| 案件與搜尋基礎 | `test_fp_case_page.py`、`test_fp_cases.py`、`test_fp_case_search.py`：19 passed |
| 模型庫與工作 | 模型版本／內容資產／工作進度／取消／發布衝突，12 backend passed；3 model UI AppTest passed，含實際 ONNX subprocess 和雙向 ONNX/PT 配對 |
| 真實 YOLO adapter | 原版 v5、v8、11：15 passed；實際固定版本 Detect 前向／反向／ONNX 匯出數值比較，不使用伪造模型替代 adapter |
| YOLO 負例 | 3 passed；6 skipped 是不適用該版本的分支。零梯度、不支援 end2end head、原版 v5 模組邊界 |
| 案件歸因橋接 | 初版 5 passed，含真實 v8 原圖推論→選框→CAM→保存→重開、錯誤來源與過期區域隔離 |
| 原圖實驗與回覆 | source-first＋既有實驗回歸 13 passed；資料／回覆／UI 13 passed，含熱圖完整性及越界拒絕 |
| 完整工作區備份 | 備份＋模型 UI 16 passed；最後還原內容身分檢查後備份 12 passed。驗證新工作區、不覆寫、相對位置／工作證據／SQLite 歷史 |
| 工作區整合 | `test_investigation_workspace.py`：16 passed，涵蓋來源身分、批次子工作獨立重試、部分歷史紀錄比對、評估發布與獨立還原工作區重新開啟 |
| 圖形圈選 | `test_region_picker_page.py`：1 passed；實際瀏覽器亦確認拖曳後左側 10% 座標帶入表單並保存，修正原先畫面圈選與儲存座標不一致 |
| 來源修復與回覆 | `test_fp_cases.py`＋`test_investigation_materials.py`：25 passed，包含遺失／損壞來源補回、不同檔案拒絕且保留案件身分，以及回覆證據檢查 |
| 新案切換 | `test_fp_case_page.py` 最後一輪 3 passed；舊案件網址進入後建立新案，案件選單與網址均切換至新案，避免在旧案誤送推論 |
| CAM 搜尋上下文與實驗呼叫 | `test_cam_search_context_page.py`＋`test_source_first_experiment.py`：5 passed，25.64 秒；換 query 清除舊 CAM context、真 ONNX durable experiment wrapper 修正回歸。外部實驗 GUI focused 重跑結果另列於下。 |
| Faithfulness 批次 | `test_faithfulness_batch.py`：2 passed，0.49 秒；替代框／消失框不替代固定 scalar 分數，random control 差異保留支持數／標準差，缺控制為 inconclusive，批次上限 200。 |
| 最後搜尋接線 | `test_fp_attribution_lineage.py`：7 passed，36.70 秒；Diagnostics LV 輔助搜尋固定目前區域並清除 CAM context，同 ROI 第二次 CAM 更新目前引用且保留歷史，重試不重複新增。 |
| 資源預算 | `test_investigation_resource_limits.py`：2 passed，1.63 秒；以刻意小額 1 MB／0.2 秒預算終止真 subprocess，確認限制理由、worker 已終止、無完成資產與假成功；不耗盡主機資源，非 GUI 壓力驗收。 |
| 獨立 Diagnostics 瀏覽器 | `test_explainability_diagnostics_e2e.py` 已實際通過 224／336／448 三解析度；legacy inspector 最後通過結果另列。 |

各批次有重疊，不相加為獨立測試總數。AppTest 啟動預設 3 秒不足以涵蓋此主機首次載入，案件測試明確用 30 秒。瀏覽器伺服器等待上限可由 `LV_E2E_STARTUP_TIMEOUT` 指定，預設 180 秒；調整等待時間不放寬功能斷言。

整合期間發現同機其他專案的 pytest 清理共用 `%TEMP%/pytest-of-*`，使仍在執行的測試來源／案件目錄消失。各瀏覽器驗收改用獨立 `--basetemp=.pytest-<suite>-browser-validation`；受清理干擾的測試不列通過，也不以此推論產品故障。

## 整合驗收追蹤

`test_investigation_materials_e2e.py` 的第一個實際瀏覽器故事已通過：混合 ZIP 上傳與配對／錯誤預覽→截圖建案→拖曳圈選並保存座標→選擇上傳的參考資料集→保存回覆→下載包含 data URI 圖片且不含伺服器路徑的 HTML。驗收修正了圈選座標未帶入表單，以及下載觸發重跑造成取消的問題。

`test_yolo_investigation_e2e.py` 第一個實際瀏覽器故事已通過：PT 上傳與檢查→單圖試跑→由結果建案→指定框→正負 CAM→調透明度不增加計算→保存與完整 ZIP→自動切到新建第二案→重用模型且不重複檢查→服務重啟後保留 compute ID。驗收修正了建立新案後可能仍停留舊案的切換問題。

同檔另兩個故事也各自通過。ONNX-only 故事完成上傳、檢查、試跑轉案與選框，顯示需要配套 PT 的提示及對照實驗入口，未提供 CAM 執行按鈕；此故事沒有執行擾動實驗。配套 PT 故事完成 ONNX→補 PT→設定版本 2→指定該版本重播→CAM，保存 `per_case_matched`，仍保留歷史一致性未知與版本 1 原結果，最後拒絕損壞 checkpoint 且保留案件證據。三個故事分次通過，沒有宣稱最後一次全檔三項同跑通過。原始範圍與截圖見 [YOLO 驗證明細](validation/yolo-attribution-validation.md)。

| 真 GUI 證據 | 通過範圍／保留位置 |
|---|---|
| YOLO PT | `.pytest-yolo-browser-gate2`；第一故事通過，後续不同故事失敗，不將整個舊批次列為全綠。 |
| YOLO ONNX-only | `.pytest-yolo-onnx-gate`；ONNX-only 通過，同批 paired 失敗後另修正重跑。 |
| YOLO paired | `.pytest-yolo-pair-final`；1 passed、2 deselected，515.30 秒，exit 0。 |
| 既有 FP 真 DINO | `.pytest-fp-browser-validation/test_real_case_to_comparison_r0`；案件→實際搜尋→引用說明→覆核 ZIP 回傳→重啟與來源不可變性通過。此路徑不能取代 legacy inspector 頁驗收。 |
| recovery 四故事 | `.pytest-fp-browser-validation`，`test_investigation_recovery_e2e.py`；以既有案件／結果 fixture 建立起點，實際 GUI 完成錯圖／正確來源修復、逐區修正／撤回／恢復及工時、零偵測仍可人工圈區、零梯度警告且不重算，四項均通過。 |

該 FP 整批結果為 **5 passed、3 failed，681.85 秒**：通過的是 oldFP 一故事及 recovery 四故事；未通過的是外部實驗與 lifecycle 兩故事，不能列為整批完成。

其後三個未通過故事均於 focused runs 完成：`.pytest-fp-lifecycle-validation-r2` 的 seeded 排隊取消→真缺失權重檢查失敗→重試 GUI 通過；`.pytest-fp-external-lifecycle-validation-r3` 的外部與模型歷史故事 **2 passed，312.69 秒**。外部故事走完歷史重播／ZIP、持久擾動實驗、訓練 ZIP、固定評估門檻、報告／來源不可變性與重啟；模型歷史故事檢查 seeded 同資產的站點／門檻兩版本、封存→舊版本→恢復，原案件版本 1 快照保持。原四模組的八個故事因此分次全部通過；不是單一最後整批八項重跑，也不是實際生產模型跨站推論驗收。

U05 十圖混合匯入的獨立實際瀏覽器故事已通過：不同目錄的同名圖片、類別對應、逐檔人工標註角色覆寫、預測分數、已提供零框／未提供紀錄、損壞檔與路徑越界拒絕，保存後核對資料庫內容。逐張看圖確認的 focused AppTest 亦已通過：未確認不可建清單，保存後可建立，撤回確認使原清單失效。

U12 的訓練匯出與照片評估已拆成兩個獨立實際瀏覽器故事，兩項皆通過（2 passed，156.90 秒；證據位於 `.pytest-materials-browser-validation-11`）。訓練故事由 UI 建案、精確座標新增區域、人工覆核、類別與整張完整性確認，到 ZIP 實際下載並驗證類別檔。評估故事以既有案件與可控模型作 fixture，其餘資料集上傳、人工標註角色、逐張看圖確認、建立清單、持久工作執行與結果加入案件皆由一般 UI 完成；未以此宣稱真實部署模型已驗收。

覆核 ZIP 新增獨立 `provenance.json`：保留本次區域／來源／覆核快照及所引用的搜尋→模型歸因證據，修正後的舊引用仍標示為歷史選樣關係。可攜覆核清單不含本機來源路徑，既有版本與回傳身分契約保持相容。focused 測試 3 passed，包含傳遞引用、無關證據排除、路徑排除、快照不可被後續修改改寫，以及回傳覆核相容性。

U14 的真實雙 Streamlit AppTest 與既有案件頁測試共 4 passed（19.49 秒）：tab A 先保存，tab B 的過期報案草稿被拒存且不增加版本；明確開啟最新版本後保留 B 草稿，合併保存，再次本分頁保存亦可完成。證據為 `test_case_ui_concurrency.py` 與 `test_fp_case_page.py`。這是報案資訊表單的雙 session 驗證，沒有把其他表單或模型封存 GUI 一併標為通過。

U10 CAM 交付與 U14 真 Chromium 雙分頁另完成 **2 passed，73.65 秒**（`.pytest-materials-browser-validation-13`）。CAM 故事以已有 checksum 與目標身分的已保存結果作 fixture，實際選證據、指定反駁關係、編輯保存、檢視 iframe 熱圖、下載 HTML 與工程附錄；核對嵌入圖片 bytes、證據 ID／SHA、排除本機路徑與未選證據。成功截圖：[CAM 圖文交付](validation/yolo-case-cam-reply-browser.png)。本故事未重新執行模型。

雙分頁故事以同一既有案件為起點：B 草稿失焦但不提交，A 先保存，B 過期提交被拒且資料庫版本不變、草稿保留；B 明確開啟最新版看見 A 內容，再合併保存且版本只增加一次。此為真瀏覽器驗證，不以 AppTest 代替；衝突範圍仍限定報案資訊表單。另 `test_case_ui_concurrency.py` 最後兩項 AppTest 通過（16.95 秒），補驗本頁背景結果發布可見且保留草稿，其他分頁發布仍需明確開最新版本。

U11 的無路徑上傳→真 DINO 搜尋→CAM 關聯候選交接已完成單一整鏈故事：`test_uploaded_references_cam_to_real_dino_handoff` **1 passed，132.90 秒**，證據位於 `.pytest-materials-browser-validation-18`。以已保存且具 checksum／目標身分的 CAM 作 fixture，實際 GUI 各上傳一張正常／瑕疵參考圖，核對資料集內容與兩個資料夾欄均空白；由 CAM 的搜尋按鈕建立正確上下文，真實執行 `dinov2_vits14`，確認模型 checkpoint 指紋、兩類結果及 CAM 引用。候選經人工覆核後，實際下載覆核 ZIP 與訓練 ZIP，兩者皆含對應搜尋／CAM 證據鏈；訓練來源依上傳原檔名選取，另核對人工類別檔。此故事沒有重新執行 Grad-CAM，也不以相似度自動指定真值。成功截圖：[上傳參考到交接](validation/yolo-case-cam-handoff-browser.png)。

整鏈驗收修正兩個產品缺陷：CAM 搜尋按鈕原本在主選單建立後直接修改 `tool_switch`，改成有錯誤處理的 callback，失敗保留在診斷頁；另外上傳參考圖經搜尋成為候選後原檔名被內容雜湊取代，現在搜尋結果、候選與交接保留原名，內容指紋與儲存位置不變。focused AppTest 2 passed（12.11 秒）涵蓋真主選單元件下的導航與另一分頁撤回目標後拒絕導航；原檔名從搜尋到兩種交接的 focused 測試 1 passed（3.20 秒）。

## 驗證範圍

| 項目 | 最近已知結果與下一步 |
|---|---|
| legacy DINO inspector | **已完成**：r6 最後一輪 1 passed，353.08 秒，exit 0。實際走完 Visualize／Attention、336 exact-resolution head、336→24×24 Grad-CAM、Normal Bank 導頁。`.pytest-dino-browser-validation-r6/test_explainability_attention_0/legacy_stage.txt` 為 passed，來源及快取前後指紋相同；持久證據：[成功截圖](validation/dino-legacy-gradcam-browser.png)、[指紋核對](validation/dino-legacy-immutability.json)。與先前 standalone Diagnostics 224／336／448 合併構成兩入口回歸。 |
| 未宣稱的現場／全組合驗證 | 兩現場不同前處理的生產推論、所有工作階段的中途取消與 GUI 資源壓力修復，沒有以目前固定案例或後端測試代替。無路徑參考上傳→真搜尋→CAM 引用交接，以及 CAM→對外回覆接線均已由上列故事實際通過。 |

各 U 編號的完整要求與實測範圍見 [產品驗收矩陣](lv-investigation-product-review.md)。固定案例的軟體接線驗收與生產現場效果分別記錄，不互相替代。

工作子程序預設各使用 1 個 OMP／MKL／OpenBLAS thread，尊重既有環境設定，不改 Streamlit 主程序的全域計算設定。`test_investigation_jobs.py` 最後一輪 7 passed，涵蓋此環境設定與工作生命週期。

## 真實部署驗收

對照實驗的方向顯示另經真 ONNX 與 AppTest 驗證：修改後分數增加顯示「分數上升」，控制圖分數相同顯示「分數未變」，案件原因仍保持未知。`test_source_first_experiment.py` 最後一輪 5 passed，16.63 秒，包含持久工作呼叫與來源身分回歸。

測試使用可控 ONNX 與固定套件建立的真实 YOLO 架構 checkpoint，驗證執行契約與數值行為。尚未取得使用者的實際部署權重、原圖、現場前處理和報警規則，因此沒有宣稱真實案件根因已確認、所有 YOLO 變種相容，或已測得節省工時百分比。

工時測量欄位與固定案例比較協議已在案件頁與操作指南提供。部署方可把實際案例跑完後補記首次設定／每案時間、重複輸入、目標重作及證據退回。

操作與邊界見 [YOLO 案件診斷指南](yolo-case-diagnostics.md)。
