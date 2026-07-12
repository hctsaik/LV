# PRD:看過類別的 AnomalyDINO 預標(M11)

> 來源需求:[1_user_needs/prelabel_seen_class.md](../1_user_needs/prelabel_seen_class.md)
> 設計素材:[SIMILAR_OBJECTIVE_AND_PRELABEL_PLAN.md](../SIMILAR_OBJECTIVE_AND_PRELABEL_PLAN.md) Feature B。
> 前置維護修復 **Task 0(objective 名稱映射 bug)已於 2026-07-05 修復並 E2E 綠**(不屬本 PRD 範圍)。

## 三行自問(缺一不放行)

- **誰會用**:已經用「瑕疵類別」標籤(每類 ≥ N_min)建過含分類頭模型的品管工程師,
  面對持續進來的新資料(bbox 已存在、類別待填/待修)。
- **不做會怎樣**:模型明明已認得幾種瑕疵(③徽章看得到判定),但使用者仍得逐顆手標,
  重複標同一類數百次,半小時標註預算被「模型其實會的」吃光,難例反而沒空看。
- **成功的可觀察判準**(→ 餵 `/pm` 當 AC 來源):
  1. 對含 head 的模型 + 一個目標資料夾跑預標 → 畫面出現**預覽清單**,每個物件顯示
     `pred(正常/類別名/Unknown)· conf · 收不收(accept)`;accept 數 == gated_predict 判為
     「非正常且非 Unknown、conf ≥ min_conf、且屬使用者選定類別」的物件數。
  2. 按匯出 → 使用者**另選**的輸出資料夾出現 `labels/*.txt`,讀回的行數與 cls_id 正確
     (只含 accept 且非正常的物件;cls_id == class_names.index(該類))。
  3. 把輸出目錄指到**來源/監看資料夾內** → 明確報錯(C6),不寫。
  4. 跑完預標 + 匯出後,**來源資料夾一個位元組未變**。

## 目標與成功指標(可衡量)

| 指標 | 目標值 |
|------|--------|
| 預覽 accept 判定正確性 | 與 `gated_predict` × min_conf × 類別過濾**逐物件一致**(釘死於 AC) |
| 匯出 YOLO 格式 | cx,cy,w,h ∈ [0,1];往返誤差 < 1e-6;cls_id 對 class_names 索引 |
| C6 安全 | 輸出目錄 == 或位於任何來源/監看資料夾內 → raise;來源資料夾 hash 不變 |
| 誠實留白 | Unknown / conf < min_conf 的物件 accept=False,不進標註檔 |
| 無 head 防呆 | 無分類頭模型 → 明確中文提示(不靜默、不亂猜) |

## 範圍(MoSCoW)

- **Must**
  - 純邏輯:`prelabel_records`(每物件 pred/conf/accept)+ `to_yolo_lines`(accept 且非正常 → YOLO 行)
    + `export_prelabels`(寫使用者另選目錄,C6 assert + 原子寫)。
  - GUI:③ 或 ② 內「🏷️ 預標」區塊 —— `min_conf` slider + **目標類別 multiselect** + 預覽表
    (pred/conf/accept + 縮圖)+ **匯出到另選資料夾**按鈕。**絕不提供寫回來源**選項。
- **Should**
  - M10 監看佇列每項旁顯示 head 建議(`建議:<類別> (0.87)`)+「採納建議」→ append_label
    (`source:"prelabel_accepted"` 供追溯)。
- **Could**
  - 匯出打包成 zip 供 download_button(免填輸出路徑)。
- **Won't(v1,明列以免膨脹)**
  - **無標籤影像的偵測預標**(bbox 生成:pmap→連通域→外接框)= v2 備忘,本輪不做。
  - 全自動標註(人工最終確認是底線,非負擔)。
  - 跨機、排程自動跑(跟現有掃描/監看流程走)。

## 模組分解(交 `/architect`)

| # | Module | 一句話職責 | Tier | 相依 |
|---|--------|-----------|------|------|
| 11 | `prelabel` | 把凍結 head 的閘控判定轉成「可接受的預標決定 + YOLO 標註檔」並安全匯出到使用者另選目錄 | **B**(檔案 I/O + C6 安全;**無 GUI/非同步 → gate=單元+真實檔案系統測試**,對齊 08/09/10) | `dino_head.gated_predict`、`anomaly_bank_store._atomic_text`、C6 安全目錄檢查 |
| — | M11 GUI 接線 | ③/② 預標區塊(min_conf + 類別過濾 + 預覽表 + 匯出)+ M10 佇列建議按鈕 | **B**(GUI → 真實 Playwright E2E) | 模組 11、既有 `_anomaly_*` wizard |

### 拆模組品質自檢

- **內聚**:模組 11 一句話講得完(「閘控判定 → 安全的 YOLO 預標檔」),無「以及/還有」。✅
- **契約耦合**:對外就 3 個純函式 + 明確資料結構(records/decisions/lines_by_image);
  相依顯式(gated_predict、原子寫、安全檢查),不成環。✅
- **可獨立驗收**:`/pm` 只憑模組 11 的 AC(用假 head/合成 records + tmp_path)即可寫紅測試,
  不依賴 GUI。✅ GUI 接線另走真實 E2E(見「done 定義」)。

## 「done」定義(分層,防 false-green)

- 模組 11:`python verify/gate.py prelabel` 印 GREEN(單元 + 真實 tmp_path 寫檔/往返/C6 拒寫)。
- M11 GUI 接線:done = 模組 11 gate 綠 **AND** 真實 E2E 綠
  (`tests/e2e/test_prelabel_gui_e2e.py -m e2e`:含 head 模型 → 預標 → 預覽 accept 數正確 →
  匯出到 tmp 目錄 → **實際讀回 labels/*.txt 斷言行數/cls_id**;來源資料夾 hash 不變)。

## 使用者驗收(從使用者角度)

- 我對一批新資料跑預標,**大部分模型認得的都已填好類別**,我一眼掃過、按匯出。
- 匯出的標註檔放進我指定的資料夾,訓練工具**不用轉檔**直接吃;**來源資料夾沒被動**。
- 沒把握的它標「不確定」留給我,而不是亂填錯的來浪費我時間。

## 給架構師的開放問題

1. **正常物件的處理**:head 判「正常」的物件,匯出時應「剔除」(不進瑕疵標註檔)還是「保留原標籤」?
   建議**剔除**(正常=非瑕疵),並在 summary 記剔除數。請在 AC 釘死。
2. **conf 定義**:用 `gated_predict` 回傳的類別機率最大值當 conf?Unknown 的 conf 如何定義
   (建議 Unknown 一律 accept=False,conf 記其類別機率但不影響 accept)。請釘死。
3. **C6 安全檢查粒度**:「輸出目錄位於來源/監看資料夾內」如何判定(路徑包含關係 + 反向)?
   建議重用既有 `assert_safe_*` 家族的判定風格。
4. **多來源類別對齊**:目標資料夾的 `class_names` 與模型 head 訓練時的類別若不一致,cls_id 以誰為準?
   建議**以模型 head 的類別為準**(預標是模型在說話),並在對不上時明確報錯。

## 本輪 appetite

**≤1 模組(11 prelabel)+ GUI 接線,一輪做完。** 達上限即停輪;Should/Could 未完進候選。
Feature A(找相似)不在本輪(未拍板,候選)。
