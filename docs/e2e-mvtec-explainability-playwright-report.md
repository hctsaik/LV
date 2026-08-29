# LV 真實 MVTec Explainability E2E 報告

執行日期：2026-08-29  
測試工具：Playwright Python + pytest-playwright + Chromium
測試入口：[tests/e2e/test_explainability_mvtec_e2e.py](C:/code/claude/LV/visuallatent/tests/e2e/test_explainability_mvtec_e2e.py:88)

## 1. 測試資料

使用者提供的帶底線路徑不存在；實際使用的 MVTec `can/test_public/bad` 目錄為：

```text
C:\code\dataset\MVTec2\ad2_test_public_gold_v1\mvtec_ad_2\can\test_public\bad
```

測試只複製以下 5 張影像到 pytest 暫存目錄，沒有修改原始資料集：

| Case |
|---|
| `000_regular.png` |
| `000_underexposed.png` |
| `000_overexposed.png` |
| `000_shift_1.png` |
| `001_regular.png` |

### 後續 Grad-CAM E2E 核准資料集（研究用途）

真實 Grad-CAM E2E 使用下列本機 MVTec AD 資料夾：

```text
C:\code\dataset\MVTec\capsule\test
```

它的 `good`、`crack`、`scratch` 等子資料夾可直接作為
`根目錄／類別／影像` 的 classifier 輸入。測試會從每個類別複製少量影像到
pytest 暫存資料夾、建立當次 session classifier head，並對 `scratch` target
做 Grad-CAM；絕不寫入上述原始資料夾。資料僅供個人研究，保留 MVTec AD
的 CC BY-NC-SA 授權與來源註記。

## 2. Playwright 實際操作流程

每張影像使用獨立 browser context，避免 Streamlit rerun 留下舊 callback：

1. 開啟 LV Visualize。
2. 選擇 `Image Classifier`。
3. 加入只含上述 5 張圖的暫存資料夾。
4. 將 projection method 設為 `PCA`，按 `Run`。
5. 用 Playwright 選取指定檔名的縮圖。
6. 確認 viewer 顯示正確檔名及原圖。
7. 展開可解釋性區塊，按 `分析此圖`。
8. 確認分析完成、PNG 與 JSON 下載控制項存在。
9. 確認 viewer 有 4 張圖：原圖、模型輸入、attention overlay、viewer 顯示圖。
10. 確認沒有 Streamlit exception。
11. 比對 cache 與原始 MVTec 目錄 fingerprint，確認單張分析沒有寫入。

目前共用面板會在上述單張圖詳情中列出三種能力狀態：DINO Attention、分類理由
（Grad-CAM）與 Anomaly Heatmap。Visualize 會明確標示 Anomaly Heatmap 需要
瑕疵偵測 inspector 的 patch Normal Bank；分類資料則會在尚未有 head 時提示可先
建立 session head，不會將另一種熱圖當成替代品。

## 3. 評分規則

每案滿分 100 分：

| 項目 | 分數 |
|---|---:|
| 正確選取真實影像並在 viewer 顯示 | 20 |
| DINO Attention 分析完成，PNG/JSON 匯出存在 | 30 |
| 原圖、模型輸入、overlay、viewer 圖像均顯示 | 20 |
| 無 Streamlit exception | 15 |
| cache 與原始資料 fingerprint 不變 | 15 |

停止條件：5 案平均分數必須 `> 90`。

## 4. 結果

| Case | 分數 | 選圖 screenshot | 分析完成 screenshot |
|---|---:|---|---|
| `000_regular.png` | **100/100** | [PNG](C:/code/claude/LV/visuallatent/scratch/playwright/mvtec-explainability-20260829-report-screens/cases/000_regular/01-selected.png) | [PNG](C:/code/claude/LV/visuallatent/scratch/playwright/mvtec-explainability-20260829-report-screens/cases/000_regular/02-analyzed.png) |
| `000_underexposed.png` | **100/100** | [PNG](C:/code/claude/LV/visuallatent/scratch/playwright/mvtec-explainability-20260829-report-screens/cases/000_underexposed/01-selected.png) | [PNG](C:/code/claude/LV/visuallatent/scratch/playwright/mvtec-explainability-20260829-report-screens/cases/000_underexposed/02-analyzed.png) |
| `000_overexposed.png` | **100/100** | [PNG](C:/code/claude/LV/visuallatent/scratch/playwright/mvtec-explainability-20260829-report-screens/cases/000_overexposed/01-selected.png) | [PNG](C:/code/claude/LV/visuallatent/scratch/playwright/mvtec-explainability-20260829-report-screens/cases/000_overexposed/02-analyzed.png) |
| `000_shift_1.png` | **100/100** | [PNG](C:/code/claude/LV/visuallatent/scratch/playwright/mvtec-explainability-20260829-report-screens/cases/000_shift_1/01-selected.png) | [PNG](C:/code/claude/LV/visuallatent/scratch/playwright/mvtec-explainability-20260829-report-screens/cases/000_shift_1/02-analyzed.png) |
| `001_regular.png` | **100/100** | [PNG](C:/code/claude/LV/visuallatent/scratch/playwright/mvtec-explainability-20260829-report-screens/cases/001_regular/01-selected.png) | [PNG](C:/code/claude/LV/visuallatent/scratch/playwright/mvtec-explainability-20260829-report-screens/cases/001_regular/02-analyzed.png) |

**平均：100.00/100（通過 `> 90` 停止條件）**

### Capsule 真實 Grad-CAM 結果

2026-08-29 以 Capsule 資料的 `good`、`scratch`、`crack` 各 6 張暫存副本，
選取 `scratch/000.png`，建立 session head 並明確選擇 `scratch` target。

| 項目 | 結果 |
|---|---|
| 真實 MVTec Playwright 回歸 | **2 passed in 208.93s**（AD2 Attention 139.56s；Capsule Grad-CAM 69.21s） |
| 原始 `C:\code\dataset\MVTec\capsule\test` | fingerprint 不變 |
| 隔離 LV cache（Visualize 後 → Grad-CAM 後） | fingerprint 不變 |
| UI 證據 | [Grad-CAM screenshot](C:/code/claude/LV/visuallatent/scratch/playwright/mvtec-explainability-regression-20260829-183705/cases/capsule-gradcam/01-gradcam-scratch.png) |
| 機器可讀證據 | [result.json](C:/code/claude/LV/visuallatent/scratch/playwright/mvtec-explainability-regression-20260829-183705/cases/capsule-gradcam/result.json)、[JUnit XML](C:/code/claude/LV/visuallatent/scratch/playwright/mvtec-explainability-regression-20260829-183705/junit.xml) |

此結果驗證真正的 MVTec 影像能經由 Visualize → session head → 指定 `scratch`
target → Token Grad-CAM 完整運作；不是以 mask 衡量定位準確率。

由 Playwright 產生的機器可讀證據：

- [case-results.json](C:/code/claude/LV/visuallatent/scratch/playwright/mvtec-explainability-20260829-report-screens/cases/case-results.json)
- [JUnit XML](C:/code/claude/LV/visuallatent/scratch/playwright/mvtec-explainability-20260829-report-screens/junit.xml)
- [pytest stdout](C:/code/claude/LV/visuallatent/scratch/playwright/mvtec-explainability-20260829-report-screens/stdout.log)

JUnit 結果：`1 passed in 118.37s`。

## 5. 重現命令

```powershell
$env:LV_RUN_REAL_MVTEC_E2E = '1'
$env:LV_E2E_ARTIFACT_DIR = (Resolve-Path 'scratch/playwright/mvtec-explainability-20260829-report-screens/cases').Path
.\.venv\Scripts\python.exe -m pytest -q -s -p no:cacheprovider `
  tests\e2e\test_explainability_mvtec_e2e.py `
  --output=scratch\playwright\mvtec-explainability-20260829-report-screens\pytest-output `
  --junit-xml=scratch\playwright\mvtec-explainability-20260829-report-screens\junit.xml
```

Capsule Grad-CAM（opt-in，原始資料只讀）：

```powershell
$env:LV_RUN_REAL_MVTEC_GRADCAM_E2E = '1'
$env:LV_MVTEC_CAPSULE_TEST_DIR = 'C:\code\dataset\MVTec\capsule\test'
$env:LV_E2E_ARTIFACT_DIR = (Resolve-Path 'scratch/playwright/mvtec-explainability-regression-20260829-183705/cases').Path
.\.venv\Scripts\python.exe -m pytest -q -s -p no:cacheprovider `
  tests\e2e\test_explainability_mvtec_e2e.py::test_real_mvtec_capsule_gradcam `
  --output=scratch\playwright\mvtec-explainability-regression-20260829-183705\pytest-output `
  --junit-xml=scratch\playwright\mvtec-explainability-regression-20260829-183705\junit.xml
```

測試程式也支援 `LV_E2E_RECORD_TRACE=1` 和 `LV_E2E_RECORD_VIDEO=1`。本次正式報告使用 screenshot-only，以避免 CPU-only 環境的錄影編碼干擾測試時序；Playwright 的 screenshot、JUnit 與逐案 JSON 已保留完整證據。

## 6. 限制與解讀

- 這是 LV Visualize 的真實 browser E2E，不是 YOLO viewer 流程。
- 真實 `bad` 資料夾沒有可微分 classifier head，因此本輪驗證的是 DINO Attention、匯出、UI 與不寫入保證；Grad-CAM disabled warning 屬預期行為。
- Capsule Grad-CAM E2E 的 session head 由當次載入影像建立；它驗證真實影像、可微 target、對比 signed Grad-CAM、UI／匯出與零原始資料寫入。新版 UI 另顯示 sample-level OOF 與遮蔽忠實度，但 sample-level OOF 仍不是 acquisition-group holdout，不能當成正式泛化評估。
- MVTec pixel mask 目前只供熱圖的目視對照；未定義量化指標前，不宣稱 segmentation IoU 或定位準確率。
- Grad-CAM 預設解釋實際 top-1 相對最強 runner-up 的分類證據；手選其他 target 只標成反事實假說。它不等同通用瑕疵圖、像素級分割或機台判定，且低 margin、低 OOF 或遮蔽不支持時不應拿來指導調模。
- CPU-only 執行會觸發既有 xFormers fallback warning，但沒有造成測試失敗。
