# PRD:Dataset Audit 報告化 + 製程 metadata 最小關聯 — M20

- 需求來源:[1_user_needs/dataset_audit_metadata.md](../1_user_needs/dataset_audit_metadata.md)
- 上游依據:[5_active_learning_product_review_2026-07-12/08_平台概念差距分析.md](../5_active_learning_product_review_2026-07-12/08_平台概念差距分析.md)
  的優先序裁決(P0-A「最小 metadata=Audit/Registry 共同前置」→ P0-B「可重現 Dataset Audit」)
  與「單機尺度最小落地形態」表前兩列。
- 立案:2026-07-19;狀態:PRD 完成,放行 `/architect`

## 三行自問

1. **誰會用**:要對主管交代「資料可不可靠」的製程分析工程師;手上有影像資料夾 +
   一張「檔名 ↔ 機台/配方/批號」的 CSV。
2. **不做會怎樣**:六種品質訊號散在五六個畫面,每次交代要截六張圖手工拼;
   「哪台機的資料髒」這類製程端問題**永遠答不了**;數字沒有出處,被追問就心虛。
3. **成功的可觀察判準**(餵 `/pm` 當 AC 來源):
   - (a) **植入數字全對**:合成資料集植入已知數量的問題(3 張重複、1 組跨 split 洩漏、
     1 組同圖異標、1 個未覆蓋類別、離群圖)→ 報告數字**逐項等於植入數**,不多不少。
   - (b) **metadata 切片**:丟 CSV(含 tool 欄)→ 報告出現依 tool 分組段落;
     故意 5 列對不上 → 報告明寫未匹配數 5。
   - (c) **一鈕到匯出**:選資料夾 → 按鈕 →(進度)→ 畫面報告 → 匯出檔案存在且非空;
     每個數字帶「方法+門檻」一行;無 CSV 時一切照常(僅無分組段落)。

## 目標與成功指標(可衡量)

- **G1 植入精確性**:六類植入問題,報告數字 == 植入數(單元合成 + 真實 E2E 皆驗)。
- **G2 metadata 往返**:CSV 匹配率正確(含未匹配計數);分組統計(per-tool 張數/
  重複數/驗證集覆蓋)與手算一致。
- **G3 可稽核**:報告每個指標帶 method line(方法名 + 門檻值);這是 08 文件
  「Audit 數字要帶校準」差異化的落實。
- **G4 匯出**:HTML(可貼報告)+ JSON(機器可讀)+ CSV(問題清單);
  絕不寫來源資料夾([[no-dataset-writes]],輸出到使用者另選資料夾)。
- **G5 復用**:訊號計算復用既有函式(phash/embedding 重複、洩漏、離群、類別發現、
  manifest sha256),不重造演算法;embedding 走既有快取。

## 範圍(MoSCoW)

### Must
1. **輸入**:單一資料夾(自動辨識:第一層子目錄名含 train/val/test → 視為 split;
   classifier layout 父目錄名=類別;YOLO labels/ 存在 → 類別取自標註;平鋪=無類別)。
2. **六訊號彙總**(全部復用既有能力):
   - 完全重複(內容 hash)與近重複(phash/embedding),含跨 split 洩漏獨立計數;
   - 同內容異標(標籤衝突);
   - 類別分布(per-class 張數)+ **split 覆蓋缺口**(val/test 缺哪些類);
   - 離群(embedding 離群分數 Top-N 清單);
   - 壞檔數(不可解碼,復用 safe_io);
   - (給了 CSV 時)per-metadata-欄位分組:張數/重複/覆蓋。
3. **metadata CSV 關聯**:欄位自由;以**檔名**(必)與 **sha256**(可,若 CSV 提供)對照;
   輸出匹配率與未匹配清單;對不上**明講不默略**。
4. **報告產出**:畫面呈現(每節:數字+白話一句+method line)+ 匯出 HTML/JSON/CSV。
5. **進度回報**;**新工具分頁**「🩺 資料體檢報告」(與單張「體檢卡」語義呼應:
   體檢卡=一張圖,本工具=整個資料集)。

### Should
6. 離群/重複的縮圖預覽(點清單看圖)。
7. 報告內「問題影像清單」一鍵加入策展購物車(復用既有 cart)。

### Could(達 appetite 即進候選)
8. class overlap(cross_class_nn_pairs 數量化)。
9. 兩資料夾距離(training-production distance)——先單資料集。
10. 報告快照比較(這次 vs 上次)。

### Won't(v1,明砍;理由記決策日誌)
- 自動修復(刪重複/改標)——使用者明言只要看清楚。
- MES/資料庫即時接線——CSV 手動丟。
- severity/owner/waiver 簽核流——平台尺度(08 文件 C 類)。
- 排程自動跑。
- S1/S2/S3 全域語言改版、AL Promotion Gate、特徵版本治理、儲存 GC——
  **另輪立案**(同屬 08 文件 B 類,但不塞進本輪 appetite)。

## 模組分解(交給 `/architect`;新號從 23 起,身分以名稱為準)

| # | module | Tier | 一句話職責 | 對外相依(顯式) |
|---|--------|------|-----------|----------------|
| 23 | `meta_join` | B | metadata CSV ↔ 影像清單對照(檔名/sha256 雙鍵)→ 每影像欄位 dict + 匹配統計 + 分組索引 | csv/stdlib;吃影像路徑清單與(可選)sha256 對照表 |
| 24 | `audit_report` | A | 吃各訊號的**原始結果陣列/清單** → 量化指標 + 分組統計 + method line + 白話句的報告資料結構(JSON-able,無 I/O) | 只吃 python 基本型別/numpy |
| 25 | `audit_pipeline` | B | 資料夾(+可選 CSV)→ 佈局辨識 → 呼叫既有訊號函式 → 23/24 → 結果包 + 匯出 HTML/JSON/CSV | 既有 phash/embedding/離群/類別發現/manifest/safe_io + 23、24 |
| — | GUI 接線 | B | 第 11 工具「🩺 資料體檢報告」:輸入/進度/報告呈現/匯出(落 `scripts/app.py`) | 25 |

拆模組自檢:
- **內聚**:23=「對照與分組」、24=「彙總成報告結構」、25=「串訊號與落地」;各一句話。
- **契約耦合**:24 純資料進出可獨立紅測;23 只有 CSV 讀檔一點 I/O;25 顯式列依賴;無環。
- **可獨立驗收**:23/24 合成輸入即可紅測(植入數字手算);25 tmp 檔案樹 + 注入 embedding
  極小化真模型依賴;GUI 走真實 E2E(G1/G2 植入數字斷言)。

## 使用者驗收(銜接 User 文件「成功」)

| User 成功描述 | 驗收對應 |
|---|---|
| 一顆鈕 → 量化報告,不用截六張圖 | G1/G3(E2E 真實流程) |
| 植入的問題都抓到、數字對得上 | G1(單元+E2E,逐項 ==) |
| CSV → 依機台切開看;對不上明講 | G2(單元+E2E) |
| 匯出貼簡報 | G4(檔案存在非空、HTML 含關鍵數字) |
| 每個數字有出處 | G3(method line 斷言) |
| 沒 CSV 也能用 | G2 反向(E2E 無 CSV 路徑) |

## 給架構師的開放問題

1. 佈局辨識規則的精確定義(train/val/test 判定、classifier vs YOLO vs 平鋪的優先序、
   混合情況怎麼辦)。
2. 近重複門檻(phash 距離/embedding cosine)沿用哪個既有值;method line 的格式。
3. 離群用哪條既有函式與 N 的預設;要不要 per-class。
4. sha256 來源:重算 vs 讀 manifest(manifest 不存在時?)。
5. HTML 匯出的實作形態(體檢卡已有 HTML 匯出樣板,沿用其手法?)。
6. 分組統計的鍵空間控制(CSV 欄位很多/值很碎時,報告怎麼不爆炸——上限與「其他」桶?)。
7. embedding 訊號對大資料夾的成本控制(抽樣?沿用快取?可跳過開關?)。

## 本輪 appetite

**新增能力輪:≤3 模組 + GUI 接線,一輪做完。** Could(8–10)與 Won't 中的「另輪立案」
四項全部進候選。組織性決策(Business Risk 擁有者、Platform/Scenario Owner)**不可程式化**,
不在本輪也不在任何 PG 輪——屬使用者決策,ROADMAP 決策日誌記錄即可。
