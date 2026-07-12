# 2026-07 維護後續 Punch List(單一資料夾語義 / 徽章 / 監督UMAP 預設)

> **來源**:commit `f062cb6`(單一資料夾語義 + 類別｜檔名徽章 + 監督UMAP 唯一預設)提交前的
> 對抗式複查(19-agent workflow,15 raw → 9 confirmed findings)+ 3 輪完整 E2E 驗證。
> 這些項目已依使用者決策「先提交、記錄為已知限制」入庫(見 ROADMAP 決策日誌 2026-07-05 則、
> commit `f062cb6` 訊息);本文件是**擇期修復的執行計畫**,照優先序做即可。
> 每項完成後:更新本文件狀態欄 + ROADMAP 決策日誌 append 一行。

---

## 通用開發協定(每一項都適用)

1. **流程歸屬**:除 D1 外全部屬**維護迴圈**(只動 `scripts/app.py` + `tests/e2e/`,不受 appetite
   約束、不起新 U-Net 輪)。**D1 動測試基礎設施(PM 擁有)→ 必須走 reverse gate 交 `/pm`**,
   禁止在 PG/維護模式就地改 `tests/conftest.py`、`tests/e2e/conftest.py`、fixtures。
2. **done 定義**(本清單全是 GUI/整合類):單元 gate 綠 **AND** 真實 E2E 綠。
   - 單元:`python verify/gate.py <module>`(印 `GREEN` 才算)
   - E2E:`python -m pytest tests/e2e/<檔> -m e2e -q`
   - **鐵則**:禁止為過綠而 skip/xfail/放寬斷言/改契約。
3. **跑 E2E 前必清殘留 Streamlit server**(最常見假失敗來源,詳見
   `.claude/skills/visuallatent-e2e/SKILL.md`):
   ```powershell
   Get-CimInstance Win32_Process -Filter "name='python.exe'" |
     Where-Object { $_.CommandLine -like '*streamlit*' } |
     ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
   ```
4. **已知假紅判別**(先排除環境,再懷疑程式):
   - 全套 non-e2e 中 `test_scenarios_cov.py` s07/s10 出現
     `X has 16 features, but NearestNeighbors is expecting 384` = **既有跨測試污染(D1)**,
     單獨跑該檔綠即非回歸(2026-07-04 已與 HEAD 對照證實)。
   - E2E 全套跑到後段連鎖逾時(`test_gui_flows` 共用 page 的骨牌效應)= 長 session 資源
     衰退;**單獨重跑該檔**判別,乾淨過=假紅。
5. 涉及行為改變的項目(A、B1)在 ROADMAP 決策日誌**明寫規格修訂**,免得未來被誤判為回歸。

---

## 優先序總覽

| 順位 | 編號 | 一句話 | 主要檔案 | 幅度 | 狀態 |
|------|------|--------|----------|------|------|
| 1 | B1 | PCA 回到預設投影(恢復無監督交叉校驗) | app.py + test_gui_flows.py | 極小 | ⬜ |
| 2 | A | 選資料集根目錄→自動偵測 split→一鍵展開(一招解 A1–A5) | app.py + 多個 E2E | 中 | ⬜ |
| 3 | A3b | `_VIZ_SYMBOLS` 補 `"val"` key(一行) | app.py | 一行 | ⬜ |
| 4 | C1 | 物件級徽章加 `#obj序號` 消除碰撞 | app.py + E2E 斷言 | 小 | ⬜ |
| 5 | A6 | 取代多資料夾清單前 toast + 一鍵復原 | app.py + E2E | 小 | ⬜ |
| 6 | B2 | 「標籤分歧」連線在監督投影上的 help 文案重框架 | app.py | 文案 | ⬜ |
| 7 | C2 | test_s04 離群排序斷言誠實化(表格加 score 欄) | app.py + test_scenarios_r1.py | 小 | ⬜ |
| 8 | D1 | cov 測試 16-vs-384 快取污染隔離(**走 /pm**) | tests 基礎設施 | 中 | ⬜ |
| — | D2 | E2E 長 session 韌性(可選,現狀可接受) | — | — | ⬜ |

---

## B1 — PCA 回到預設投影方法

**問題**:`_DEFAULT_METHODS = ["監督UMAP"]`(app.py:348)後,有標籤資料(≥2 類、≥4 樣本)
只算監督投影;「PCA 保底」(app.py:4181 `if not proj:`)只在**完全沒算出任何投影**時觸發,
正常情況永遠不會跑。監督UMAP 用標籤硬拉開類別,「看起來分得很好」≠ 真的可分——使用者
失去無監督誠實視圖的交叉校驗,除非自己記得手動加選。

**做法**:預設改 `["PCA", "監督UMAP"]`。PCA 計算量毫秒級,不拖慢 Run;t-SNE/UMAP(慢)不加回。
**預設顯示的投影維持監督UMAP 不變**(下拉預設值邏輯在 app.py:4240 附近,確認加入 PCA 後
預設 index 仍指向監督UMAP)。

**開發步驟**:
1. 改 app.py:348 一行。
2. 檢查 app.py:4240-4243 投影下拉的預設選擇邏輯,確保預設顯示仍為監督UMAP。
3. E2E 簡化:`test_b`(test_gui_flows.py ~147-150)f062cb6 加的「手動點開下拉勾 PCA」三行
   **移除**(PCA 已是預設,再手動勾反而測不到預設行為);`test_f` 的 PCA↔監督UMAP 切換不動。

**驗收(AC)**:
- AC1:有標籤資料集 Run 後,投影下拉同時提供 PCA 與 監督UMAP,預設顯示為監督UMAP。
- AC2:切到 PCA 座標非空、散點正常渲染(選取跨投影存活 — 既有 test_f 覆蓋)。
- AC3:無標籤/單一類別資料 Run 不空結果(監督跳過 → PCA 為預設方法直接算出,保底不再是唯一防線)。
- 綠燈:`pytest tests/e2e/test_gui_flows.py -m e2e` 全綠(含移除手動勾選後的 test_b/test_f)。

**ROADMAP 日誌措辭建議**:「預設投影修訂為 PCA+監督UMAP:監督排版仍是預設顯示,PCA 恢復為
恆算的無監督交叉校驗視圖(修 f062cb6 已知限制 c)」。

---

## A — 選資料集根目錄 → 自動偵測 split → 一鍵展開(核心項)

**根因**:split = 載入資料夾的 basename(visualize_embeddings.py:39/58),UI 現在最多持有
1 個資料夾 → 所有需要 ≥2 split 的功能一次全斷:

- A1 跨 split 洩漏掃描 UI 不可達(`test_p_duplicate_leakage_scan` 已 skip)
- A2 **train/val/test 資料集只能載一個 split,其餘靜默消失、無警告**;指向根目錄直接
  "No images found"(root 的 train/val 子目錄被 `discover_images_classifier` 當成類別名)
- A3 多 split 疊圖散點(per-split symbol)、Split 下拉全成死 UI
- A4 YOLO 偵測資料集 train/、val/ 兩資料夾無法合併
- A5 覆蓋度熱圖「資料集(split)」軸永遠單欄

**設計原則**:保留 2026-07「單一**選取動作**」語義——使用者仍然只選一個東西;選到的是
**資料集根目錄**時,偵測並提示展開。程式化多資料夾路徑(`viz_folder_list` 為 list)整條
下游都還活著(demo 按鈕 app.py:3348 已證明),只是 UI 進不去。

**偵測啟發式(建議)**:選取的資料夾本身依當前模式 discover 不到任何 records,**且**其
直接子目錄中存在名稱屬於 `{train, val, valid, test}` 的資料夾、其中至少一個能 discover 出
records(分類器=子目錄下還有一層類別資料夾;偵測器=有 images/+labels/)→ 顯示
`st.info("偵測到 train/val 子資料夾")` + 按鈕「⊞ 一起載入 N 個 split」→ 按下把
`viz_folder_list` 設為 `[root/train, root/val, ...]` 並 rerun。

**開發步驟**:
1. app.py `_visualize_embeddings_ui` 資料夾區塊(~3790-3808)後加偵測與展開提示
   (只在 viz 工具做;cov/quiz/gray 等單資料夾工具不動)。
2. 展開後的多項清單沿用既有 render 迴圈(✕ 移除鈕從死 UI 變回活的)。
3. 輸入路徑/📁 的「取代」語義**不變**(取代整個清單,配合 A6 的 toast+復原)。
4. E2E:
   - **新增** `test_root_split_expansion`:建 root/{train,val}/class/*.jpg → 貼 root 路徑 →
     斷言提示出現 → 點展開 → Run → Split 下拉含 train 與 val、兩 split records 都在。
   - **`test_p_duplicate_leakage_scan` 解除 skip**:改走「貼 root → 一鍵展開」載入
     train+val,原斷言(byte-copy 跨 split 對被掃出)全數恢復,不放寬。
   - `test_s07_unicode_paths_round_trip` 建議改回雙 split(經展開路徑),恢復
     `viz_split_select` 切 val 的原始斷言;`test_s01`(冷跑效能鏈)可維持單 train 不動。
5. 決策日誌明寫規格修訂:「單一資料夾語義修訂=單一『選取』動作 + 資料集根目錄自動展開」。

**驗收(AC)**:
- AC1:選含 train/+val/ 的 root → 出現展開提示;按下後 `viz_folder_list == [root/train, root/val]`,
  Run 後兩 split 的 records 都在(Split 下拉= All/train/val)。
- AC2:選單一 split 資料夾(train/class/*.jpg)→ **不**跳提示,行為與現狀完全相同。
- AC3:選完全無效的資料夾 → 維持現有 "No images found" 錯誤,不誤觸發展開提示。
- AC4:展開後散點圖 train/val 以不同 symbol 呈現(需先做 A3b,否則 val 落到預設圓形)。
- AC5:跨 split 洩漏掃描恢復:test_p 解除 skip 後全斷言綠(勾「僅跨 split」掃出 byte-copy 對)。
- AC6:展開狀態下輸入新路徑=取代整個清單(單一選取語義不破壞),配合 A6 有 toast+復原。
- 綠燈:新增 E2E + test_p + gui_flows 全套 + scenarios_r1 全綠;完整 e2e 套件 0 failed。

**風險備註**:偵測啟發式若太寬會騷擾(每次選錯資料夾都跳提示)——限定「名稱屬於
{train,val,valid,test} 且能 discover 出 records」兩條件都滿足才提示,可控。

---

## A3b — `_VIZ_SYMBOLS` 補 `"val"` key(一行,順手修)

**問題**:`_VIZ_SYMBOLS = {"train":"circle","test":"square","valid":"diamond"}`(app.py:343)
key 是 `"valid"`,但 demo(app.py:3348)與現實資料集慣例是 `"val"` → val split 落到預設
circle,與 train 形狀無差別,per-split symbol 功能在最常見命名下失效。

**做法**:加 `"val": "diamond"`(與 "valid" 同形狀)。
**驗收**:demo 一鍵載入(train+val)後散點圖兩 split 形狀不同;做完 A 後其 AC4 直接覆蓋。
無需獨立 E2E,併入 A 的展開測試斷言即可。

---

## C1 — 物件級徽章加 `#obj序號` 消除碰撞

**問題**:`_rec_fname`(app.py:288)回傳 `Path(image_path or path).name`。物件級記錄
path=`.lv_cache` 裁切圖(`{stem}__objN.jpg`)、image_path=原圖 → 同一張原圖切出的多個
crop 徽章**完全相同**(`chair｜img001.jpg` ×3),而舊的 `#idx`/`第n` 已移除(grid 只剩
hover tooltip `help="#{i}"`)。dup view(app.py:2659)同 split byte-copy 對兩側徽章也相同。

**做法**:物件級記錄(判別式:`image_path` 存在且 ≠ `path`,同 app.py:2201 `is_obj_rec`
的既有判別)徽章補物件序號:`chair｜img001.jpg#3`。序號來源二選一:
(a) record 若帶 `obj_index` 直接用;(b) 從裁切檔名 `__objN` 後綴 parse(來源 app.py:5655
命名規則)。建議 (a) 優先、(b) 為 fallback。實作位置:改 `_rec_fname` 本體(或新增
`_rec_badge` 包一層),30 個呼叫點全站生效,整圖級/分類器記錄無 `__objN`/`obj_index`
→ 輸出不變。

**驗收(AC)**:
- AC1:物件偵測模式,同一張原圖的兩個 crop,grid/相似/文搜/dup 牆上徽章互不相同(含 #N)。
- AC2:分類器與整張影像(M8)模式徽章**不變**(無 # 後綴)——迴歸保護。
- AC3:dup view 同 split byte-copy 對兩側可由徽章區分。
- AC4:E2E 既有解析不破:`test_covfilter_classleak` 取 `split("｜")[0]` 的類別解析不受影響
  (#N 在「｜」後段);`test_objcov` s6 的 `rsplit("d=",1)` 解析不受影響。
- 綠燈:anomaly 全套 E2E + covfilter/objcov AppTest + gui_flows 全綠。

---

## A6 — 取代多資料夾清單前 toast + 一鍵復原

**問題**:demo(或 A 展開後)清單有 2+ 資料夾時,輸入任何路徑會**靜默清掉全部**;
UI 每列還有 ✕ 移除鈕,暗示「可加可減」的心智模型,與取代語義矛盾。

**做法**:`_folder_add_cb` / `_pick_folder_set` 在「即將取代的清單有 ≥2 項」時:
先把舊清單存進 session snapshot(沿用 mode-switch 的 `_viz_mode_snapshot` 復原 idiom,
app.py:3369 附近)→ 取代 → `st.toast("已取代原本的 N 個資料夾")` + 顯示「↩ 復原」鈕。
Streamlit 無原生 modal confirm,不要硬做阻擋式確認。

**驗收(AC)**:
- AC1:demo 載入(2 資料夾)→ 輸入自己的路徑 → 清單變 `[輸入值]`、toast 出現、
  「↩ 復原」按下後清單回到原 2 項。
- AC2:清單只有 0/1 項時取代**不**跳 toast/復原(平常操作零騷擾)。
- 綠燈:新增 1 條 E2E(demo → 取代 → 復原)+ gui_flows 既有全綠。

---

## B2 — 「標籤分歧」連線在監督投影上的 help 文案重框架

**問題**:連線 pairs 是 **raw 空間**最近鄰異類對(`cross_class_nn_pairs`,
interaction.py:1243 cosine NN),畫在被標籤拉開的監督UMAP 座標上 → 長線橫跨全圖,
視覺上說「離很遠」,與 help 文案「你眼睛看到的爭議」(app.py:4254-4256)承諾相反。
著色訊號(raw 空間計算)不受投影影響,仍正確。

**做法(便宜方案,推薦)**:改 help/caption 文案,把長線**重新框架成訊號**:
「在監督投影上,橫跨兩群的長線=該點對在原始 embedding 空間是最近鄰、卻被標成不同類
——線越長越值得複查標註」。語義即自洽,長線反而成為 mislabel 雷達。
(徹底方案=監督投影時隱藏連線並提示切 PCA 看,成本高、必要性低,不建議。)

**驗收**:純文案,無行為改變。人工審閱文案 + `_no_exception` 級 E2E 即可;
不得順手改動連線計算或渲染邏輯(那不在本項範圍)。

---

## C2 — test_s04 離群排序斷言誠實化

**問題**:`test_s04`(test_scenarios_r1.py:308)現斷言 `"｜" in cards.first.inner_text()`
——「｜」是每張卡的結構常數,排序壞掉照樣綠(空洞)。注意:舊斷言 `"第1"` 同樣是
位置性的、從未驗過值排序,**這是既有缺口非 f062cb6 新引入**;codebase 已有誠實 pattern
(同檔 ~275 行相似度面板 `ranked == sorted(ranked)`)。

**做法**:兩步。
1. app 端:離群度/標籤分歧排序時,「詳細表格」expander(app.py:2398 附近的 dataframe)
   **加 score 欄**(顯示當前排序準則的分數)。
2. 測試端:test_s04 讀表格 score 欄,斷言嚴格遞減(`vals == sorted(vals, reverse=True)`)。

**驗收(AC)**:
- AC1:排序=離群度時表格含 score 欄且由大到小;切「檔名」排序時無 score 欄(或欄值合理)。
- AC2:test_s04 新斷言對「打斷排序」的 mutation 會紅(自查:暫時把 argsort 的 `[::-1]`
  拿掉跑一次應紅,改回後綠——**只在本機自查,不留在程式碼**)。
- 綠燈:scenarios_r1 全綠。

---

## D1 — cov 測試 16-vs-384 快取污染隔離(⚠ 走 /pm,不可就地改)

**問題**(既有,與 f062cb6 無關,2026-07-04 已與 HEAD 對照證實):
`test_covfilter_classleak` / `test_objcov_*` / `test_scenarios_cmpc` 用 **dim=16 假 embedding**
preseed 進共享 `.lv_cache`;全套 non-e2e 一起跑時洩漏進 `test_scenarios_cov.py` s07/s10 的
**真實 384 維 DINOv2** 路徑 → `ValueError: X has 16 features...`。單跑該檔綠。
測試作者已在 test_covfilter_classleak.py:61-63 記載此失敗模式。

**做法**:測試基礎設施屬 **PM 擁有** → 走 reverse gate 記錄「/維護 →(測試隔離缺陷)→ /pm」,
由 /pm 設計隔離。建議給 /pm 的方向:
- 所有 dim=16 preseed 的測試統一 monkeypatch `LV_CACHE_DIR` 到 `tmp_path`(隔離寫入);
- **注意既有相依**:e2e conftest 註記 test_b/test_t 刻意驗證真實 `.lv_cache` 內容
  (`manifest_path_for`/`ref_path_for` 無 env 直讀)——隔離範圍只限假 embedding 測試,
  勿全域隔離打破這些誠實斷言;
- 替代案:cov_token 命名空間隔離(快取 key 帶 token)。

**驗收(AC)**:
- AC1:`pytest tests/ -m "not e2e"` 全套**連續跑兩輪**皆 0 failed(含 s07/s10)。
- AC2:單跑 `test_scenarios_cov.py`、單跑各 dim=16 測試檔,行為不變全綠。
- AC3:e2e 的 test_b/test_t(真實 `.lv_cache` 斷言)不受隔離影響,照常綠。

---

## D2 — E2E 長 session 韌性(可選,不排程)

第一輪全套曾見 `test_gui_flows` 共用 module-scoped page 的骨牌效應(test_b 一倒、後面 17 個
連鎖逾時),以及尾段單測假紅;皆單獨重跑即綠。已寫入 skill 的判別流程,現狀可接受。
若未來想強化:test_b 散點等待 deadline 放寬(150s→240s)、或把 flow_page 鏈拆段。
**不建議**改成 per-test server(session-scoped server 是刻意的效能取捨)。

---

## 收尾檢查清單(全部做完後)

- [ ] `python verify/gate.py` 相關模組全 GREEN
- [ ] 完整 `pytest tests/e2e/ -m e2e` 一輪 0 failed(先清殘留 server)
- [ ] `pytest tests/ -m "not e2e"` 0 failed(D1 完成後 s07/s10 應穩定綠)
- [ ] `test_p_duplicate_leakage_scan` 已解除 skip 且綠(A 完成的標誌)
- [ ] ROADMAP 決策日誌:每項一行 + 「單一資料夾語義修訂」規格明文
- [ ] commit 訊息比照 f062cb6 風格:做了什麼/為什麼/驗證方法/殘餘限制誠實列出
