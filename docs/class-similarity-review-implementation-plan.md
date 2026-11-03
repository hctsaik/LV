# LV 類別相似度分布與覆核候選：多 Agent 分析及實作交接

日期：2026-09-14。狀態：**設計建議，尚未實作／驗收**。

版本：設計r3。同日第三輪由三個agent獨立審查，再由主agent整合修正。本文各章節為目前唯一實作契約，沒有需要下游自行判斷的新舊覆寫規則。工作單T0–T10；未提供人工真值時可用候選預覽，不得生成虛構PR。審查發現與修正對照見 [本輪審查紀錄](class-similarity-review-design-audit.md)。

目的：讓使用者從各類別的 box plot 找到值得檢查的圖片／物件，再檢視鄰居證據、選取與匯出。本文鎖定第一版規則，供能力較弱的 AI 分階段執行；不是已完成的功能報告。

## 1. 決策摘要

適合加入 LV，定位為 Visualize 既有「標錯偵測／類別混淆」面板的擴充，名稱建議「類別相似度分布」。

- 每類一個 box，每個散點是一筆 image 或 object record。
- 預設顯示「同類相似度 − 最接近異類相似度」，可切換兩個相似度本身。
- 由使用者按「計算」啟動，使用當前模型的原始高維 embedding。
- 點／框選散點或勾選表格 → 原圖及鄰居證據 → 同步既有選取／加入購物車 → 匯出證據。
- Threshold滑桿以margin門檻預覽候選及工作量；固定人工評估集具備後，再顯示覆核Precision／Recall，詳見§16–§22。
- 不依分數自動改標；負分只是優先檢查的線索。
- 第一版不新增頂層工具、不訓練模型、不建立另一套審查系統。

**修正前面討論：不能接舊灰帶覆核入口。** 程式已將該工具隱藏並重導至 Visualize。現行產品方向是選取／購物車 → Labeling → 標註回饋。第一版以候選與證據匯出收斂；物件原圖與 ROI 的 Labeling 傳遞需另行完成契約，不能宣稱既有出口已完整支援。

## 2. 多 Agent 審查及整合裁決

本次由主 agent 整合三個獨立唯讀分析；沒有要求子 agent 修改程式。

| 審查分工 | 發現／應考慮事項 | 本文裁決 |
|---|---|---|
| 產品／UX | 現有標籤健檢已提供混淆熱圖與疑似標錯；灰帶指南與現行入口不一致 | 擴充既有面板，不重建混淆熱圖或復活灰帶 |
| 數學／統計 | 原始 cosine、min-max 優先分数、鄰居異類比例是不同量 | 新純函式明確計算 Top-K mean；不用舊優先分數畫圖 |
| 整合／測試 | index、crop path、原圖和物件身份不同；現行 Labeling adapter 不能假定保留 ROI | 保留 record ID、原 index、原圖／bbox；獨立匯出完整證據 |
| 主 agent | 全配對、全類別圖、ANN、覆核資料庫同時做會讓弱 AI 失控 | 限制第一版 exact 分塊計算 ≤5,000 筆，其他分期 |

上表的限制值與預設值是本次設計選擇，不是經效能測試證明的產品能力。

## 3. 已查證的本地整合點

以函式名稱定位；行號可能隨其他工作變動。實作前重新閱讀實際檔案，不能以歷史文件取代程式證據。

| 路徑／符號 | 已存在的能力 | 使用方式／限制 |
|---|---|---|
| `scripts/app.py::_render_label_quality` | 混淆熱圖、疑似標錯、寫入 `viz_selection` | 增加分布檢視，保留舊功能 |
| `scripts/app.py::_visualize_embeddings_ui` | image/object 共用尾端呼叫上述面板 | 兩種模式都要驗收；勿只支援 detector |
| `viz_records`、`viz_raw_embeddings[model]` | record 列及降維前向量 | row 必須一一對應；不用 `viz_embeddings` 的投影座標 |
| `scripts/similarity.py::cosine_similarity_to_ref` | raw cosine | 可參考正規化，但新模組自行加嚴無效輸入檢查 |
| `similarity_priority` | 每次參考查詢內 min-max 到 0–1 | **不可當原始 cosine**；不同查詢結果也不能直接混畫 |
| `multi_ref_similarity` | 每類最大 cosine | 不等於本設計逐類 Top-K 平均，不直接代用 |
| `scripts/interaction.py::compute_label_disagreement` | 鄰居異類比例，按 index 排除自己 | 參考身份排除；保留既有分數語意 |
| `find_similar_indices` | HNSW 檢索，回傳 cosine distance | 新 exact 統計不用它；distance 和 similarity 不能混淆 |
| `scripts/object_eval.py::neighbor_confusion` | 既有混淆統計，含完整 N×N 運算 | 不直接拿來當新分布計算 backend |
| `scripts/app.py::_render_select_view` | 現行選取、縮圖與送標入口 | 新圖同步原始record index；不另造全域選取集合，本面板暫存勾選依§7.2 |
| `_add_to_export`、`_export_entry` | 購物車，物件保存 `image_path/bbox/obj_index` | 以 crop path 去重；score 只存四位小數；重加不更新證據 |
| `scripts/labeling_handoff.py` | Labeling 出口 | 不能假定 crop 自動還原為原圖 ROI；第一版不修改其契約 |
| `tests/e2e/conftest.py` | 真實 Streamlit + Playwright harness | 重用；不要為了新功能改通用 harness |

`docs/usage_guide.md` 仍有舊灰帶流程，實作文件更新時只修本功能相關描述。現有 UI 的「全 384 維」文案也不適用所有模型，改以「原始高維特徵」及實際 D 呈現。

`_render_label_quality` 現有「少於3筆或2類就return」只適用舊分析。新增模式分派須放在此guard之前，讓第一版能呈現singleton／單類的有效same分數及缺值原因。新state加入 `_MODE_CLEAR_KEYS`。兩條label更新路徑都必須使新結果過期：Run期間的 `_apply_readback_overlay`，以及 `_render_readback_section` 直接呼叫 `LH.apply_readback` 的即時讀回；不能只靠不一定改變的 `viz_data_token`。

工作區已有多項未提交變更。實作 AI 必須先看 `git status --short`，保留原有工作，不 reset、clean 或覆蓋無關修改。

## 4. 第一版範圍與成功條件

### 必做

1. 已有單一標籤的 image / object records；重用已抽好的同一模型 embedding。
2. 三個分數的各類 box plot、散點、完整分頁表格。
3. 原图／ROI 及同類、最接近異類 Top-K 鄰居證據。
4. 明確選取、加入現有購物車、CSV + JSON 證據下載。
5. 資料無效、樣本不足、過期事件、效能上限、身份追溯與離線行為驗收。

### 後續再做

- 每對類別的全配對 box plot、另一套熱圖。
- >5,000 筆的分層抽樣統計／ANN 加速與 recall 校驗。
- 無標籤分群、多標籤分類、自動刪重、自動改標／合併類別。
- 微調 embedding、不同模型的分數混合、active learning 自動排程。
- 新審查資料庫、雙簽或裁決流程。
- 將新完整證據與原圖 ROI 原生送進 Labeling、回讀後自動重算。

產品試用時記錄：前 20 筆候選中有多少筆被人認為值得處理、檢查耗時、誤標／合理少見樣态／分類規則問題各有多少。這是試用指標，不設定沒有基準的「準確率達 90%」。

## 5. 分析母體與身份契約

### 5.1 分析與顯示分開

- 預設分析按下計算時 `viz_records` 的所有 rows；新增明確「分析 split」設定，預設全部。
- 選定 split 後，query 和鄰居 reference 都取自該固定母體。這是資料探索，不是模型泛化成績。
- 主散點頁的 split／類別顯示篩選不暗中改此母體；新面板顯示「分析母體：…」。
- 計算後的類別顯示選擇、指標切換、表格排序與選取不重算分數。
- 模型、K、分析 split、資料／標籤／embedding 改變，顯示結果過期並禁止舊結果選取和匯出，需重新計算。
- 不混用模型、層、pooling、解析度或 crop policy；同 D 不代表同 embedding 空間。

### 5.2 record 身份

新增 adapter 產生 `record_id`，並保留 `original_index`（在當前 `viz_records` 的位置）。

- 本功能明確使用下述canonical JSON的SHA-256作ID，不直接復用其他模組名稱相似的ID。`explain_contracts.stable_record_key` 採內容hash及其他bbox/policy規則，不能代用。
- image 身份欄位：`level=image`、正規化絕對來源路徑。
- object 身份欄位：`level=object`、正規化 `image_path`、原樣 bbox、`obj_index`。
- 路徑在 Windows 以一致的 resolve/normcase 規則處理。bbox 須為有限數值，保留原座標系、不做不明單位轉換；在 manifest 記其既有座標契約。
- ID 不含 label、分數或排序位置；label 變動由分析 key 處理。裁切策略改變也由 recipe／run key 處理。
- object provenance 不完整時列為無效，不能退化成只用原圖路徑把多顆物件合併。
- 缺provenance而無法建立正式ID時，adapter給 `invalid:<data_token>:<original_index>` 暫存ID及原因，以維持列對齊；不允許加入候選／購物車／人工評估。其詳細items／CSV只在all診斷匯出；所有scope的全U最小score_index仍保留該診斷列，以重算總數，不將它當可信來源身份。
- 同一身份出現多列時停止並報重複來源衝突；不同身份卻 embedding 相同仍保留，不自動判定重複圖。
- 另檢查同一crop path是否對應不同原圖／ROI身份。現有 `_crop_and_embed_objects` 的stem命名可能令不同來源的同名圖碰撞；一旦發現，整份新分析拒絕並列出衝突，不能因record ID不同就信任可能已錯配的embedding。修復上游crop/cache命名並重新抽取是另一個明確任務，不在本功能靜默修復。
- 散點 `customdata` 至少带 `analysis_key, record_id, original_index`。不能用分組後的 point index 找原圖。

### 5.3 analysis key 與快取

`analysis_key = SHA256(canonical JSON metadata + normalized valid embedding bytes)`。

Metadata至少含：schema/method version、當前完整data token、按輸入順序的IDs/labels/splits、source_snapshot key、有效與排除狀態、模型與可取得的權重身份、embedding recipe（level、D、resolution、pooling、crop policy）、K、analysis split、數值方法版本。source_snapshot必須綁定embedding生成時的generation context；按分析時只核對，不能用當下新圖片hash替舊raw補造證明。

Run/extraction接線須保留同generation的來源內容hash／recipe／rows對齊（抽取前後hash不一致則該generation失敗；cache必須已有可驗來源版本才重用）。可核對image模式viz_manifest是否確實與該generation相綁；object的空manifest不能視為已驗證。既有raw缺此證據先標source_unverified，需透過既有抽取pipeline產生可驗新generation才能解除，不由本分析模組另載模型或盲信現在的crop檔案。

快取第一版只存 session，且只留最新一次結果。未知模型／recipe metadata 標記 unknown，不編造權重 hash。embedding bytes 納入 digest，避免同名模型或同一批路徑換向量後誤用；digest 可在載入／變更時計算一次，不在每次選圖時重算。

選取事件使用的 key 必須等於當前非過期 analysis key，且 original index 指向相同 record ID。任一不符即拒絕，要求重選。

每次面板進入及選取／匯出前，比較当前records的ID、有效label、label_status與scope fingerprint；不能只在Run時驗一次。embedding digest在向量載入／替換時更新；外部圖片內容不靠同路徑假定不變，見§18的來源snapshot。分析狀態固定為empty/computing/ready/stale/error；失敗或過期禁止action，新結果完整成功後才切ready。

## 6. 明確數學規格：弱 AI 不得自行替換

### 6.1 輸入

`E` 為可轉為浮點的 `(N,D)` 二維矩陣，D>0；有限性逐列檢查。K 預設 5，可選整數 1–20（拒絕 bool）。

外層維度、record 長度不符、K 非法為整體錯誤，不能截斷資料湊長度。

逐 row：缺標籤、NaN/Inf、L2 norm ≤1e-12 者列入 exclusions，不作 query 或 reference，不填 0。缺標籤使用 loader 的缺標記契約，不把名為 `Unknown` 的合法類別憑字面刪掉。多標籤不任挑第一類，列為第一版不支援。

現有records未必有獨立的缺標狀態：detector流程會出現`unknown`／`mix`，classifier卻可以有合法同名資料夾。adapter必須新增分析用 `label_status=single|missing|multi|unresolved`：classifier從分類資料夾確認標籤；detector依annotation來源及物件index確認該分析單位是否有單一標籤。整圖包含多類時為multi，已拆出的單一ROI可為single。無足夠來源證據則unresolved並排除、提示原因，不能只看字串猜。此狀態只存分析adapter，不直接改原records標籤；T0查明實際loader來源並把適配規則及同名folder測例寫進正式契約。

標籤語意優先序：有已確認讀回覆蓋時以目前record label為有效label，來源標註只用來證明原始粒度，不把它讀回覆蓋新label。classifier `allow_flat=True` 將平鋪資料夾名当label只是探索fallback，預設為unresolved；真正class子資料夾（含字面unknown/mix）才是single。必要的loader provenance欄位可加在新adapter/context，不改動其他功能的label解讀。完整保留 `source_label`、`effective_label`、`label_origin` 於snapshot；本文後續的original_label是被評估當下effective_label，不一定是磁碟最初label。

計算與正規化採float64，cosine clip到[-1,1]處理浮點誤差；測試數值tolerance 1e-6，但候選membership與同分比較不套此容忍度。有限大向量也可能讓直接norm溢位，採scale-safe正規化：a=max(abs(row))，a=0排除；v=row/a、b=norm(v)，以a≤1e-12/b判斷過小，否則u=v/b。正規化後再驗finite，不把overflow變成假的零向量。

### 6.2 定義

有效向量 `u_i = E_i / ||E_i||₂`，`sim(i,j) = clip(u_i · u_j, -1, 1)`。

对每个 query i 和候選類別 c：

1. `J(i,c) = {j : label_j=c 且 record_id_j != record_id_i}`。
2. 排序以 `(-sim(i,j), record_id_j)`，同分以 ID 字典序穩定決定。
3. `k(i,c) = min(K, |J(i,c)|)`。
4. `S(i,c) = 前 k(i,c) 個 sim 的平均`；無候選則 null。

定義三個輸出：

- `same(i) = S(i, label_i)`。
- `rival(i) = argmax_c S(i,c), c != label_i`；同分用 class key 字典序決定。
- `other(i) = S(i, rival(i))`。
- `margin(i) = same(i) - other(i)`，兩者皆有效才計算，範圍 [-2,2]。

**異類算法是「每個異類各自取 Top-K 平均，再挑最高的一類」**。不是所有異類混在一起取 Top-K，也不是每類 centroid 或 max 單張。`rival` 只是最接近的異類，不命名為模型預測或建議改標。

保留 same / rival 的完整 Top-K IDs、原 index、cosine、有效 K，作為可重算分數的證據。負 cosine 是合法值，不直接裁成 0。

同分判斷用實際float64計算值，不用顯示小數；另存 `tied_rival_count` 及按class key排序的前10個 `tied_rival_classes`（包含選中的rival）。超過10類顯示總數與「僅列前10類」，不持久保存N×C個同分類名。超過一類時UI註明「多個異類同分」，不能讓字典序裁決看成唯一混淆證據。

### 6.3 邊界與解讀

| 情況 | 規定 |
|---|---|
| 類別只有自己一筆 | same=null；other 可算；margin=null；顯示「無同類參考」 |
| 整批只有一類 | same 可算；other/margin=null；顯示「無異類參考」 |
| 全部無效／空資料 | N=0才items=[]；全無效仍保留N筆items/status，可計分集合與box為空 |
| 某類可用候選筆數小於 K（同類已排自己） | 用有效 K，顯示 K 不足；不可補零／重複補樣本 |
| 所有相似度相同 | box 可退化成線；margin=0 合法，不除以 IQR |
| margin<0 | 較接近某異類鄰居，優先檢查，不是誤標結論 |
| 同類相似度高但 margin 近0 | 同類與異類都像；不是分類品質高的證明 |

逐類計分減少混合鄰居投票被大類主導，但**不消除類別不平衡**：大類有更多機會找到近鄰，小類有效 K 較少。顯示類別數及有效 K，不聲稱公平校正完成。

近重複圖、同一原圖的多個 ROI、同一批次相似背景都可能讓分數偏高。第一版只排除自己，保留其他樣本並揭露來源；不把 0.999 當成刪除依據。跨來源／重複群排除是後續獨立設定。

Box 是描述性統計，不是信賴區間；樣本共享鄰居、配對也不獨立，不做未設計的 p-value／顯著性宣稱。DINO 未表現出的細小缺陷，不能靠 box plot 補出來。

## 7. 圖表、證據与操作契約

### 7.1 Box plot

- 預設 margin；另外兩個選項為 same、other，名稱包含「Top-K 平均 cosine」。
- same/other 軸 [-1,1]，margin 軸 [-2,2]，margin 畫 y=0 參考線。
- 上述為預設完整範圍；提供純顯示的「符合資料範圍」及「重設完整範圍」。窄分布如±0.015必須能放大查看。縮放不改分數、box摘要、門檻或候選；門檻在視窗外時提示其實際值，不能以看不到門檻線推論全選。此控制不是重新正規化分數。
- 每類 box 使用該指標全部有效 rows；不得使用散點抽樣結果計算四分位。
- Q1/Q2/Q3 使用 NumPy linear percentile 定義；鬚端為落在 `[Q1−1.5 IQR, Q3+1.5 IQR]` 內最小／最大實際值。將已算出的統計交给 Plotly，避免前後端 quartile 算法不同。
- outlier 是圖表術語，不自動加入「疑似誤標」。
- 類別順序固定 class key；使用者可選顯示類別，單頁最多 10 類，不改分析母體。
- 每類顯示 `有效n/總n`、無法計分原因、same/rival 有效 K 的範圍。

### 7.2 散點與表格

- 散點候選只含當前指標finite的rows。每類≤200筆時全部顯示；超過時先保留其中有效margin最低的20筆，不足20按當前指標升序、同分ID補足；再從未保留ID依 `SHA256(analysis_key+record_id)` 取到200。當前指標null者絕不畫點。
- UI 清楚寫「box 使用全量；散點顯示 x/n」；抽樣只為繪圖，不影響分數或表格。
- 水平 jitter 也用固定 hash，rerun 不跳動。
- 用獨立 scatter trace 承載選取事件；box 的中位線／鬚端不是 record，點到不能任選一張圖。
- 完整分頁表格每頁50筆，預設有效 margin 升序、同分 record ID；null 排後。提供 checkbox，作可完整操作的替代入口。
- 表格含來源、label、same、rival、other、margin、有效 K、status；支持只看 margin<0（預設不隱藏其他 rows）。
- 點選只開證據；「同步至 LV 選取」按鈕才合併目前選中的原 index，不暗中清除舊選取。同步前驗 token 和 key；重複按不重複加入。
- 清除本功能選取要使舊 Plotly 事件失效，不讓 rerun 復活已取消的樣本。

選取狀態釘死：`local_selected_ids` 是本面板checkbox／框選暫存，跨表格分頁／類別篩選保留；點一下只變inspect_id，框選才合併到local集合。`viz_selection`是已同步的全域選取。`scope=selected` 固定匯出local集合中屬於當前U的有效身份rows，不自動併入全域選取；按鈕寫「下載本面板已勾選」。同步按鈕將local集合合併至全域，沒有反向自動灌入。原全域位於分析split外的選取保留，但不出現在本面板下載。無margin但身份有效的row可人工勾選，score仍null；無效身份row不進local或global，匯出依§5.2區分詳細資料與最小統計索引。

### 7.3 圖片證據

顯示 query 的現標、same/other/margin 與理由；左右兩組分別為同類及 rival Top-K。每張鄰居顯示 label、cosine、來源；顯示順序與計分一致。

object 必須顯示 crop，並能看來源原圖上的該 bbox；同張原圖的兩個 ROI 不可混淆。缺圖時保留數值與身份、顯示缺檔，不隨便代用另一張；來源內容改變時依資料更新流程標結果過期。

數值顯示四位小數，但排序和 JSON 保存完整計算精度。不得以畫面顯示的四捨五入數反算 mean。

## 8. 效能邊界

- 第一版分析母體最多5,000列（包含無效列）；超過即在運算前提示縮小 analysis split／資料範圍，不偷偷截斷前5,000列。
- exact計算採query block（建議128）×reference block（建議512），按候選類別維護top-K；生產算法的任一時刻都不能建立超出設定block的全量配對矩陣，不能先N×N再刪除。小型oracle測試才可用完整矩陣。
- 時間仍約 O(N²D)；分塊只降低記憶體，不可寫成已解決平方時間問題。
- 每個 query block 完成後輸出 same/rival 鄰居；僅保存 O(NK) 最終證據，不把每筆所有類別的完整鄰居長存。
- 穩定 tie 需要在候選合併時處理 ID 次序，不能用不穩定的 argpartition 後只對已選集合排序就宣稱同分一致。
- 切 tab 不重算；不要因 Streamlit 的隱藏 tab 仍執行而連帶觸發昂貴旧混淆統計。可用持久化 segmented selector 分派 panel，保留原功能可到達。
- 顯示進度；新計算完成前不發布半份成功結果。第一版不承諾取消按鈕或背景服務。
- 效能驗收記錄實際 N、D、類別數、K、耗時、峰值記憶體與硬體；未測不得宣稱秒級。

## 9. 候選、購物車與匯出契約

### 9.1 購物車

復用 `_add_to_export`，source=`class_similarity`，score 可傳 margin，reason 說明 method / analysis_key / rival。保留負值。

既有購物車 score 四位小數且按 path 去重，已存在的項目不更新。因此它只是候選容器，**完整證據由本功能 JSON 保存**。顯示新增／略過數，不能聲稱重加後證據已更新。

不呼叫 `_viz_send_to_gray`。不因使用者看圖或加入候選就自動發送 Labeling。本計畫不改 Labeling schema，不宣稱 object 的原圖 ROI 已端到端傳遞。

### 9.2 檔案格式

唯一下載scope為 `all | candidates | selected`：all是全量U診斷報告；candidates是已提交threshold的候選；selected是§7.2本面板勾選。基礎階段未接threshold時candidates停用；不用另一種schema過渡。下載不寫回資料集。

CSV 每列一筆 query，至少：`analysis_key, record_id, original_index, level, path, image_path, bbox_json, obj_index, label, split, same, other, rival_class, margin, same_k, rival_k, status`。無效值為空欄，不是0。

JSON 包含：

```text
schema_version: 2
method_version: class_topk_margin_v1
analysis_key
created_at_utc
scope: all | candidates | selected
config: {K, analysis_split, model, embedding_recipe, exact: true}
summary: {input_count, valid_count, excluded_count, per_class_counts}
records: 身份／來源索引表（query與其鄰居都可解析，含object provenance）
items: [{record_id, same, other, rival_class, margin, same_k, rival_k,
         status, tied_rival_count, tied_rival_classes, same_neighbors:[{record_id, cosine}],
         rival_neighbors:[{record_id, cosine}]}]
exclusions: [{record_id或original_index, reason}]
policy / preview_key / candidate_summary / evaluation
score_index / evaluation_items  # 完整定義見§21
```

JSON null 表達不能計分；使用嚴格 JSON，不輸出 NaN/Infinity。選中匯出需連帶放入鄰居 records 索引，但不把鄰居假列為使用者選中的 query。

中文及特殊字元需可往返；CSV 檔名／標籤等使用既有安全匯出慣例。缺失 provenance 必須明確報告，不能從 filename 猜原圖。瀏覽器直接下載，不新增外部服務。

## 10. 建議模組與輸出契約

以下名稱都是**建議新增**，不是已存在 API。

| 模組 | 職責 | 不允許混入 |
|---|---|---|
| `scripts/class_similarity.py` | 驗證向量、分塊 exact Top-K、分數、box summary | Streamlit、磁碟寫入、模型載入 |
| `scripts/class_similarity_records.py` | record adapter、穩定 ID、analysis key、選取驗證 | 偷改 label、裁切策略 |
| `scripts/class_similarity_export.py` | CSV／JSON bytes、鄰居身份索引 | 自動寫資料集／送標 |
| `scripts/class_similarity_ui.py` | 控制項、圖表、表格、證據；回傳明確 action | 自行重建購物車／Labeling |
| `scripts/app.py` 小幅接線 | 傳入 records/raw/context，消費選取和加購物車 action | 放入整套 O(N²) 計算 |

建議核心介面：

```text
analyze(embeddings, labels, record_ids, *, eligibility, invalid_reasons,
        k=5, query_block=128,
        reference_block=512, progress=None) -> AnalysisResult

AnalysisResult.items 依輸入顺序對齊（含無效rows及status），每項包含：
record_id, input_index, label, same, other, margin, rival_class,
same_k, rival_k, same_neighbors, rival_neighbors, tied_rival_count, tied_rival_classes,
status, reasons

prepare_records(viz_records, raw, context) -> PreparedInput
build_box_summary(items, metric) -> per-class summary
validate_selection(event, result, current_context) -> 原始indices或明確拒絕
export_report(result, records, *, scope, local_selected_ids,
              committed_preview, evaluation=None) -> (csv_bytes, json_bytes)
render_panel(...) -> action{type, ids, analysis_key, preview_key?, request_seq?}
```

核心 input_index 是傳入分析矩陣的位置；adapter 要保存它到 original_index 的映射。禁止以有效資料壓縮後的位置寫 `viz_selection`。

AnalysisResult.items的tied_rival_classes遵循前10類上限；無rival時為空list、count=0。

PreparedInput保留scope內全部rows及embeddings/labels/IDs/original_index。`eligibility`與`invalid_reasons`是與rows等長的adapter結果，讓缺provenance、unresolved/multi標籤不會因核心只收到字串而意外進reference。核心再檢查向量，合併reasons，不壓縮items；status使用ok/invalid/unscorable，具體原因用reasons list，可同時有insufficient_same與no_other_class。build_box_summary直接依items.label分組。

基礎階段尚未匯入evaluation／尚未有threshold時，其report欄位為null與明確not_configured狀態，不捏造候選或truth。T7/T8只啟用能力，不更换scope與schema。

## 11. 驗收案例與分層測試

### 11.1 可手算數值 fixture（K=1）

向量如下，皆為單位向量；label 與 ID 固定：

| ID | label | 向量 |
|---|---|---|
| a1 | A | (1,0) |
| a2 | A | (0.8,0.6) |
| b1 | B | (0,1) |
| b2 | B | (0.6,0.8) |

預期 `(same, other, margin)`：a1=(0.8,0.6,0.2)，a2=(0.8,0.96,-0.16)，b1=(0.8,0.6,0.2)，b2=(0.8,0.96,-0.16)。負 margin 不代表 fixture 標籤錯，測的是算法。

再加 c1/C=(−1,0)，對 a1 的 rival 仍為B；c1 的 same/margin 為null。單獨拿 c1 分析則三分數均null。零向量與 NaN row 要記排除原因。

額外固定測例：A兩筆(1,0)、B兩筆(−1,0)，K=1時所有margin=2；全向量改成(1,0)時所有margin=0。K=10且每類2筆時same_k=1、rival_k=2。通用box helper對[0,1,2,3,100]應得Q1=1、median=2、Q3=3、鬚端0與3（此fixture只測摘要，不當合法cosine資料）。

### 11.2 必須覆蓋的 AC

| AC | 驗收內容 |
|---|---|
| AC01 | 手算 fixture 數值、鄰居、有效K與mean一致；K=2與逐類算法也有獨立測例 |
| AC02 | 小型暴力雙迴圈 oracle 與分塊算法相同；多種 block size 結果相同 |
| AC03 | identity 排除自己；不同ID的相同向量保留；同分穩定、改列順序後ID結果一致 |
| AC04 | 零向量／NaN／Inf／缺標／多標／長度錯／非法K／空資料／singleton／單類處理符合規格 |
| AC05 | box全量統計與散點抽樣分離；null不當0；固定quartile與whisker正確 |
| AC06 | 改顯示類別／排序／指標不改分數；改K／label／raw／recipe／split使舊結果失效 |
| AC07 | 圖表點與表格選同ID；box摘要不選圖；取消選取不復活 |
| AC08 | 排除無效row、排序、篩選後仍能對到原index；過期事件不能加入新資料的選取／購物車 |
| AC09 | 同原圖兩ROI在證據與匯出仍是兩物件；crop／原圖／bbox一致；缺圖顯示明確 |
| AC10 | 新增／重加購物車的行為及provenance正確，不宣稱已更新既有證據 |
| AC11 | CSV/JSON可解析，中文往返、null合法、鄰居ID可解析，mean可由JSON重算 |
| AC12 | >5,000列先拒絕；配對矩陣任一時刻不超設定block，含中間暫存；記錄最大block shape及峰值記憶體；未按計算及拖threshold時analyze/舊neighbor_confusion call count不增加 |
| AC13 | 舊混淆／疑似標錯、框選／取消、相似搜尋、購物車、現行送標入口仍可用 |
| AC14 | 真實離線 DINO image及object流程至少各一例，來源資料的內容／目錄清單前後不變 |

數值測試用手算與獨立 oracle，避免只把 implementation 抄進測試。真实 DINO 不鎖定任意生成圖片必須得到某個 cosine；以輸出的 embedding 重算期望值並驗證 UI／證據身份。另用確定性向量 fixture 驗證低 margin 排序。

## 12. 給較弱 AI 的分段工作單

一次只做一個任務；完成條件要附輸出／測試證據，不能只打勾。遵守 repo 現行角色與 OpenSpec 規範；角色啟用時由對應測試階段建立契約，不讓實作階段竄改測試。

- [ ] **T0 現況與正式契約**：重讀本文、適用的 AGENTS/CLAUDE、現有函式與 git diff；建立本功能 OpenSpec proposal/design/spec/tasks 和必要的設計／AC對照，不把本文直接當已完成的變更。輸出實際接線點、現有API差異、檔案範圍。不得改應用程式。
- [ ] **T1 身份及數值契約測試**：測試階段新增 `tests/test_class_similarity.py`、`tests/test_class_similarity_records.py`，先涵蓋AC01–04/06/08。準備獨立oracle及上述fixture。先紅是此階段預期，不假稱完成實作。
- [ ] **T2 純邏輯實作**：只新增 core/records 模組；通過T1測試及分塊一致性。不得先改 `app.py`。提供介面、邊界結果和資源估算。
- [ ] **T3 匯出契約與實作**：新增 export 模組及測試，覆蓋AC09–11的純資料部分。範例JSON需讀回、由鄰居重算mean，不新增Labeling協定。
- [ ] **T4 唯讀UI**：新增ui模組並小幅接入 `_render_label_quality`；計算按鈕、狀態、box/散點/表格與證據。不接外送。測試阶段先定行為契約，實作完成以真實瀏覽器驗AC05/07/09/12。
- [ ] **T5 選取／購物車／下載**：接現有 `viz_selection` 和 `_add_to_export`，驗身份、清除、過期、重加與完整匯出；不呼叫灰帶。不得把專屬JSON證據縮成購物車單score。
- [ ] **T6 回歸與文件**：真實 image/object E2E、AC13/14、既有受影響測試；更新本功能使用說明、限制及OpenSpec tasks。保留性能與畫面操作證據。只有此時才能宣布第一版完成。

每步回報固定包含：改動檔案、對應AC、執行命令、通過/失敗/未執行、剩餘問題。遇到API不符先列出差異與最小修正，不自行放寬AC、不skip/xfail、不改通用測試harness來製造綠燈。

預計檢查命令（新檔尚未存在，建立後才執行；解譯器使用repo有效環境）：

```powershell
python -m pytest tests/test_class_similarity.py tests/test_class_similarity_records.py tests/test_class_similarity_export.py -q
python -m pytest tests/test_interaction.py tests/test_similarity.py tests/test_visualize_embeddings.py -q
python -m pytest tests/e2e/test_class_similarity_e2e.py -m e2e -v
python -m pytest tests/e2e/test_export_subset_ui.py tests/e2e/test_similar_gui_e2e.py -m e2e -v
```

依正式模組註冊完成 repo 的 `verify/gate.py` 及設計／測試對照流程；不要為新模組捏造已GREEN的輸出。測試時間不足或環境缺模型時誠實標未執行，不能以UI元素存在替代真實行為。

E2E需明確指定本測試的cache與output位置，不能假定共用harness已隔離全部cache/log。來源不寫入驗收比較來源目錄清單及檔案內容hash，不以會受讀取影響的atime判斷。

最小允許接線：app.py的panel分派、Run context、兩條readback失效、mode切換/restore generation及selection/cart actions。provenance若無法由adapter取得，可在visualize_embeddings.py／interaction.py新增sidecar回傳契約，但不改現有label值與列順序；T0先列其精確範圍與測試。E2E可參考現有 `explain_app_server` 的cache/log隔離方式；不為修新功能改共用harness。

## 13. 可直接貼給實作 AI 的提示詞

> 請閱讀 `docs/class-similarity-review-implementation-plan.md`，依T0–T10順序實作LV類別相似度分布及Threshold擴充。先完成尚未完成的第一項，不要同時大改多個階段。遵守本文公式、身份、過期、匯出與AC契約，並遵守repo現行角色與OpenSpec流程。保留既有未提交修改。重用當前原始embedding；不得用2D座標、min-max優先分數、class centroid或每類max取代Top-K mean。第一版不復活灰帶、不自動改標、不修改Labeling協定、不寫回來源資料集。Threshold只做候選預覽，明確操作才加入選取；PR必須按§18固定評估集與人工真值计算，unknown不當負例。每項完成後附實際檢查證據；測試失敗時修實作，不放寬斷言。若API與本文不符，指出確切函式和差異，再提出最小調整。此文件是設計，不代表任何功能已完成。

## 14. 第一版之後仍需決策的問題

這些問題不阻擋本文MVP，不留給實作AI自行擴充：

1. 大資料以全量ANN或分層reference抽樣？需量測與exact結果差异，報告抽樣母體。
2. 是否排除同來源圖／連拍群／近重複群？需定義群ID及不足K行為。
3. 是否有跨split獨立reference模式？需和資料洩漏檢查、模型評估清楚區分。
4. 是否把同類相似度低與margin低做雙軸檢查？需由試用案例驗證價值。
5. 如何將證據、原圖ROI、裁決與label version送到Labeling並回讀？需完整lineage端到端契約。
6. 哪些類別差異DINO無法反映？由User提供已確認同類／異類對，評估ROI/解析度/專用模型，再決定是否擴充。

## 15. 本次交付邊界

本次完成多agent唯讀調查及這份規劃文件；未修改應用程式、未建立正式OpenSpec變更、未執行功能或性能測試。本文所列數值、cap與介面是待實作契約。第一版任務仍全部未勾選。

## 16. 第二輪多 Agent 整合結論：Threshold 是同一面板的操作層

本輪再次分派三個agent，分別審查數學／真值契約、UX與相近功能、現有程式復用點。以下是主agent整合後的設計決定，不是新增模型評估結果。

| 問題 | 決定 |
|---|---|
| 與box plot的關係 | box呈現固定分布；threshold決定從分布挑哪些資料；PR評估這個挑選規則 |
| 放在哪裡 | 既有「類別混淆 → 類別相似度分布」，不新增頂層工具 |
| threshold對哪個分數 | 僅margin，固定 `margin < t`，不用任意多條件組合 |
| 能否立即知道PR | 只有具人工確認結果的固定評估集才行；無真值也能預覽候選與工作量 |
| threshold是否修改模型 | 不修改模型、異常門檻、DINO head或原標籤 |
| 要不要同時做自動最佳門檻 | 延後；先讓人看明確數值與證據，再決定操作點 |

### 建議面板順序

```text
既有標錯偵測／類別混淆
└─ 類別相似度分布
   ├─ 分析母體、模型、K、計算狀態
   ├─ Threshold滑桿＋數字輸入＋重設／全部／無候選
   ├─ 候選數、占比、缺分數數；有真值時加Precision／Recall
   ├─ Box＋散點＋門檻線
   ├─ 工作量曲線；有真值時加PR曲線及當前操作點
   ├─ 每類候選摘要、候選／完整表格
   └─ 鄰居證據 → 加入LV選取／購物車／下載
```

對使用者的說明：「這個門檻決定哪些資料先請人檢查。Precision／Recall衡量找出誤標的能力，不是分類模型本身的準確度。」

## 17. Threshold 與互動狀態契約

### 17.1 篩選規則及端點

在固定分析母體U中，`eligible = {i: margin_i有限}`；
`candidate(t) = {i∈eligible: margin_i < t}`。

- 預設t=0；slider範圍[-2,2]、step=0.01，旁邊數字輸入可輸入更精確有限值（第一版同樣限制[-2,2]）。兩者同步到唯一threshold state，不能各有一套值。
- canonical threshold保留數字輸入完整精度。例如t=0.123456，slider拇指可近似顯示0.12，但rerun不能用刻度值覆寫canonical值；只有新的使用者拖曳事件才提交0.01刻度值。門檻線、mask、PR及下載始終使用canonical值。
- 完整精度比較；`margin==t` 不入選；同分資料整組進出，不依ID拆分。
- 快捷模式 `none` 與 `all_eligible` 明確處理空／全選候選。t=2仍不包含margin=2，不能假稱已選全部。
- 改slider或數字輸入即回到`threshold`模式；「恢復預設」回到t=0。模式為none/all時不畫誤導性的有限門檻線。
- margin=null永遠不進候選，獨立列「無法計分」，其same/other圖仍可按既有規格檢視。

### 17.2 固定分布與完整計數

- threshold不改box四分位、不改reference pool、不改已計好的same/rival鄰居。
- margin圖畫門檻線；same/other圖不畫margin線，候選顏色仍按margin，caption明示。
- counts使用全部分析資料，不使用每類200點的繪圖子集。
- 主卡片：候選數／eligible數、比例、分析總數、無margin數。eligible=0時比例null，不能除以0。
- 每類摘要：總n、eligible n、候選n、候選／eligible比例、無margin n。預設class key排序。
- 類別顯示篩選不改candidate集合；操作文案明寫「將全部候選加入LV選取（含未顯示類別）」。表格可切「僅候選」，不暗中縮小母體。

### 17.3 候選與選取分開

滑動只更新預覽，不改`viz_selection`、購物車、標籤或外部系統。點資料點仍只開證據。

只有按「將全部候選加入LV選取」才驗analysis_key、合併原始index；重複按是冪等操作。放寬／收緊threshold後，既有選取保留，另顯示已選取中有多少不符合目前門檻。`none`只清候選預覽，不等於清LV選取。

下載scope一律依§9的 `all | candidates | selected`；候選、local已勾選及全域選取不能混稱。全候選按鈕直接合併全域，不暗中修改local集合；要下載該批就用candidates範圍。

### 17.4 拖動時更新與效能

目標是拖動中看見候選數、門檻線與可用的PR更新。embedding／近鄰結果預先算好，threshold操作只做mask、counts及marker更新；圖片只更新可見頁，不為每個滑動事件載入全批縮圖。

- 計分cache key不含threshold；另建 `preview_key = hash(analysis_key, policy_mode, threshold, evaluation_key)`。class顯示篩選不改統計key。
- 原生Streamlit slider是否於拖動中發送值，必須用當前本地版本實測，不能從API名字假定。瀏覽器驗收要在pointer仍按下且移到中間值時觀察數字，再release。
- 若只在放開後更新，需標記「放開更新」並回報連續預覽驗收未完成；不能當成已滿足拖動中預覽。先保留可工作的原生原型，再以最小本地component補preview事件，不引入遠端服務。
- 連續預覽可使用約200ms節流，放開後一定提交最後值；實測記錄延遲。前端預覽與Python都要對同一完整分數集合套同一個`<`規則，最終選取／下載以服務端驗證結果為準。
- 新request附analysis_key及遞增request_seq；舊request晚回不能覆蓋新值。切run／mode要清threshold widget generation及新state。
- state分成draft_policy（拖動暫值）及committed_preview（最新伺服端確認的policy/key/seq/mask）。draft與committed未一致時停用「加入全部候選」、保存操作點及各scope下載，顯示更新中；不得退回上一版本。action同時帶analysis_key、preview_key、request_seq，服務端核對全部及目前draft，任一不符拒絕並要求等候最新值。前端預覽可以先畫，但不能先產生不可追溯的下載。
- 不為了「即時」每次跑整個原有N×N混淆分析；沿用§8的panel分派。

## 18. Precision／Recall 的真值與評估集

### 18.1 正例是「原標籤經人工確認錯誤」

此處不是YOLO物件召回，也不是DINO head分類PR。判定正例：這筆資料在被評估的label snapshot中的原標籤錯誤。

`verdict = incorrect | correct | unresolved | unreviewed`。

- 合理的少見外觀且原標正確屬correct。
- 類別界線仍有爭議屬unresolved，不硬轉incorrect。
- 原標籤、rival、margin、加入購物車、系統建議，都不是人工verdict。
- 不以User只修改了哪些label，推論所有未修改者都是correct。

第一版擴充使用本地評估JSON匯入／模板下載，不新增覆核資料庫，不冒用既有Labeling回讀格式。匯入不改原資料／label。

起步流程必須可操作：「建立評估模板」明確選範圍為目前U全部或本面板勾選，按建立時凍結E0；使用勾選範圍須記為人工選樣，不能宣稱隨機或代表全資料。模板附每筆來源path/image_path/bbox/obj_index、原label及預設unreviewed，讓人可辨識圖／ROI，不只給hash ID。提供相同版本的重新下載。匯入後呈現verdict與待確認清單，點選可開證據；僅看過圖片不會自動變correct。評估集合凍結後不因勾選／門檻改變；修改E0需明確建立新evaluation版本。

```text
schema_version: 1
evaluation_id / evaluation_version
purpose: tuning | heldout
snapshot_key: 原資料身份與原label快照（不含K/threshold）
selection_method: 人工描述這批ID如何決定
frozen_ids: 事先固定的待評估record IDs
reviews: [{record_id, original_label, verdict,
           review_source, review_version}]
record_catalog: [{record_id, path, image_path, bbox, obj_index,
                  original_label, source_group_id}]
```

snapshot_key由adapter生成，含來源身份／資料版本及每筆原label。來源版本不足時標明限制、僅同session使用，不憑空建立跨session可信版本。K或模型改變可沿用同一份有效真值，但要建立新的analysis/evaluation result key；label或來源身份改變不能沿用舊真值。

來源snapshot包含每個原圖（object按原圖去重）的內容SHA-256、有效label、已知annotation／類別映射版本及label_origin。像素hash依§5.3綁embedding generation，label快照綁當前effective label。hash只讀計算、顯示進度，不在滑桿每次更新時重hash。開圖片證據及匯入／下載前核對涉及的來源內容；Run後且分析前已改圖也必須stale，不能把舊向量配新hash。缺生成證據時標source_unverified，數值探索及報告可保留，但圖片caption明示「與embedding的一致性未驗證」，不稱為已驗圖片證據；E用於任何PR評估的rows必須均source_verified，否則status=source_unverified、PR不可用，並提示重新Run產生可驗generation。

重複／衝突review ID、ID不屬於母體、原label不符、snapshot不符、verdict非法時拒絕整份匯入並列出原因，不last-write-wins或模糊比對檔名。frozen_ids中沒review的ID算unreviewed，不能當correct。

frozen_ids必須為無重複的非空字串ID序列；重複拒絕，不默默set去重。reviews的ID必須是frozen_ids子集；schema_version、purpose及欄位型別驗證，禁止bool代替數字版本。空E0允許預覽空模板，但coverage=null、PR unavailable。catalog的來源欄位用於人工辨識，匯入時仍依snapshot驗證，不信任任意手改路徑。

`truth_digest=SHA256(canonical JSON(snapshot_key, sorted frozen_ids, 按ID排序的original_label/verdict/review_source/review_version))`，缺review先補unreviewed參與digest。`evaluation_key=SHA256(analysis_key, truth_digest, purpose, frozen_policy_id或null)`。不能只用evaluation_id/version作cache key；同ID/version改verdict必須更新PR/曲線/下載，僅重排reviews順序不改digest。任一evaluation_key變更使committed_preview及未完成request過期。

### 18.2 固定評估範圍

先固定E0=`frozen_ids`，再定E=`E0∩eligible`。E0中不能計分者單獨計數及列出原因。選threshold前就確定E0，不依目前候選動態改名單。

只有E非空且E全部具有correct/incorrect二元verdict，才顯示主PR；否則顯示覆核完成度，PR為「尚不可用」。unresolved/unreviewed不能靜默刪除後假稱E完整。E0不能計分者不納入PR分母，但畫面必須同時揭露其數量、已知誤標數與`|E|/|E0|`可計分coverage。

指標名稱固定為「此評估集可計分資料上的Precision／Recall」。同時顯示E0/E大小、已知誤標數、unknown數、可計分coverage及E占分析母體比例。不要宣稱代表整包資料的recall；若有不可計分的誤標，不能因E的recall=100%便宣稱全部誤標已找出。

顯示類別篩選不改E。即使E是事先人為挑選的困難案例，數字仍只能描述這批案例，不能推廣成整體品質。

### 18.3 計數及零分母

在E中，flag表示符合當前candidate規則：

| | 人工incorrect | 人工correct |
|---|---|---|
| flag=True | TP | FP |
| flag=False | FN | TN |

`Precision=TP/(TP+FP)`；`Recall=TP/(TP+FN)`。

分母0為null，UI顯示「無候選」或「評估集無已知誤標」等原因，不填0或100%。只有E完整且非空時顯示TP/FP/FN/TN；未就緒時counts為null並呈現完成度及原因。主候選數涵蓋U，TP+FP只涵蓋E，两者不一定相同，卡片清楚標母體。

尚未完整覆核E時不顯示PR曲線；仍可正常使用U的候選預覽、工作量曲線及評估完成度。

有完整E時，表格附verdict與TP/FP/FN/TN分組；點計數卡片只篩表格，可直接檢查被門檻漏掉的FN並開原圖／鄰居。E外列顯示「未納入評估」，不當TN；切TP等檢視不改U/E、分數、candidate或主指標。E未完整時只呈現verdict與待確認狀態，不先假算TP類別。

### 18.4 選門檻與驗證分開

反覆拖動挑出的最佳表現只能標為tuning結果。第一版heldout限定在同一固定U及同一analysis_key中的另一份人工覆核子集，屬固定資料集內的覆核策略檢查，不是全新資料的泛化驗證。query/reference規則不變，不另增跨split模型驗證能力。

T9增加「保存目前操作點」「恢復已保存操作點」及未保存變更提示，僅保存已committed的預览。保存內容為 `frozen_policy={policy_id, created_at, policy, analysis_key, score_protocol_key, tuning_evaluation_key, tuning_ids, tuning_group_ids}`；score_protocol_key含method版本、K、model/weights、recipe、比較方向、固定U/reference規則。存session及報告，不部署模型設定。

heldout需當前analysis_key與已保存值相同、調整用evaluation已完成、E0與tuning_ids無交集，且已知來源群不重疊。object至少按canonical image_path作source_group；image至少按canonical path，已有連拍／批次群ID則一起檢查，缺群metadata明示未驗證。任一不符禁止heldout指標，仍可回tuning探索。恢復操作點前驗相容性，不能跨K／model／scope默套同一t。

具體UI規則：purpose=heldout時隱藏PR曲線，停用門檻滑桿／數字／快捷模式，P/R卡片固定到已保存policy。沒有保存policy時要求先從tuning確定操作點。按「轉為tuning」後才解鎖滑桿及曲線，並記錄用途變更；不能把通用互動PR介面直接套到heldout。

heldout期間K／模型／recipe／分析母體也不可調後繼續沿用heldout名稱。全域設定若改變，舊結果變stale並停用heldout；要評估新設定先重新tuning及保存。轉為tuning後看過的heldout ID記入該操作點的tuning集合，不可在同一session再切回來冒充未使用的heldout。

第一版不自动找最佳F1／成本點，不做沒有抽樣方案的信賴區間。少數類小樣本直接顯示support，不聲稱統計可靠。

## 19. 應一起加入的相近功能與分期

| 功能 | 價值 | 本輪決定 |
|---|---|---|
| Threshold滑桿＋精確輸入＋重設 | 看門檻影響哪些資料 | T7必做 |
| 每類候選n／比例＋缺分數n | 避免總量掩蓋某類被大量挑出 | T7必做 |
| 檢查量曲線（threshold → candidate count） | 不需要真值，就能估計人工工作量 | T7必做 |
| 人工評估集匯入與完整度 | 讓PR有可追溯依據 | T8必做；無真值時正確顯示不可用 |
| TP/FP/FN/TN、P/R與PR曲線當前點 | 找到人工檢查成本與漏掉誤標的取捨 | T8–T9必做，有效E才呈現 |
| 保存操作點及門檻證據 | 重現數字、避免記不清用過哪個門檻 | T9必做；匯出不等於部署設定 |
| TP／FP／FN／TN圖片下鑽 | 直接檢查找錯與漏掉的資料 | T8–T9必做，沿用表格／證據面板 |
| Box顯示範圍縮放／重設 | 真實分布很窄時仍可辨識 | T4必做，純顯示不改計分 |
| 評估模板來源對照與未完成清單 | 讓User建立真值並找到漏填項 | T8必做，不新增審查資料庫 |
| 固定人工預算Top-N／Precision@N | 「今天只能看100筆」很實用 | 後續優先；Top-N會以ID拆同分，不等同單一threshold |
| 累積找到問題／檢查量曲線 | 比較同樣檢查量能找到多少問題 | 後續；需要完整評估集與同分規則 |
| 各類別獨立門檻 | 可能照顧不同分布 | 延後，避免小類过擬合及策略難解讀 |
| same對margin雙軸散點 | 分辨類內異常與跨類混淆 | 延後，用真實審查案例驗證再加 |
| 自動最佳F1、成本／收益曲線 | 可提供策略建議 | 延後，需有成本定義、評估資料與驗證流程 |
| 分類模型confidence門檻試算 | 評估分類預測的P/R | 屬不同語意，留在模型診斷功能規劃 |

### 曲線定義

工作量曲線涵蓋U的全部eligible。依唯一margin排序，同分整組；以每個值的`nextafter(value,+∞)`建立剛納入該組的切點，另含none端點。當最大值為2時，對應切點畫為「全部可計分」，不得把數值2當包含等號。避免只以0.01步長掃描，漏掉很接近的不同分數。

PR曲線使用E，候選比較仍為`margin<t`。顯示各實際操作點，不做平滑擬合；precision不存在的點不偽填成1。當前slider值即使不在預算切點上，仍用同一predicate直接計數，標到正確的(P,R)位置。沒有真值時不画假PR。

PR掃描集合明確使用E的唯一margin之nextafter切點，含none/all端點，同分整組；每點都可由`<`重算驗證。P或R任一為null就不畫該座標。E全correct時recall無定義，不畫整條PR曲線，保留counts及原因。heldout不顯示此曲線。

T7/T8預計純函式模組 `scripts/class_similarity_threshold.py`：

```text
candidate_mask(items, policy={mode, threshold, comparator:"lt"})
summarize_candidates(items, mask)
workload_curve(items)
validate_evaluation_manifest(manifest, snapshot, items)
evaluate_threshold(items, mask, evaluation) -> counts/metrics/coverage/status
precision_recall_curve(items, evaluation)
```

所有UI、曲線、匯出都呼叫這個相同predicate／計數來源，不各抄一份可能等號不同的規則。

## 20. 既有類似功能：能参考，不能混用

| 已查證模組／位置 | 現有用途 | 新功能的界線 |
|---|---|---|
| `scripts/cost_curve.py::cost_curve` | 已確認正常／瑕疵、threshold掃描FN/FP/工作量/成本；`score>=t` | 正例與比較方向都不同；不直接代替誤標margin評估 |
| `scripts/app.py`成本工作點區段（搜尋`rcost_curve`） | 現有只試算不套用的曲線展示 | 可沿用視覺呈現方式，勿修改其門檻設定 |
| `scripts/evaluation.py::match_image` | 物件框IoU配對及偵測TP/FN/FP | 與「是否誤標」的二元真值不同，不复用IoU計數 |
| `scripts/dino_head.py`分類評估 | 類別預測的precision/recall | 不能把這些分數顯示成覆核策略PR |
| `scripts/app.py`以樣搜樣`fewshot_theta` | 相似度≥門檻的檢索篩選 | 本功能是margin<門檻，名稱和方向分開 |

不要簡單把margin取負再丟`cost_curve`：`-margin>=-t`對應margin≤t，會改掉等號規則。保留舊API及測試，新純函式釘死語意。

## 21. Threshold 匯出與驗收

唯一JSON分析報告schema為§9的2；§18人工評估匯入是另一份schema1文件。沒有舊產物需要遷移。下列為schema2其餘欄位，基礎階段也保留欄位與not_configured狀態。

報告新增：

```text
policy: {mode: threshold|none|all_eligible, threshold: number|null,
         comparator: lt, score: margin}
preview_key
candidate_ids / selected_ids / visible_class_filter
candidate_summary: {eligible_n, candidate_n, unscorable_n, per_class}
evaluation: {evaluation_id, evaluation_version, purpose, truth_digest,
             frozen_ids, scored_ids, coverage, status,
             tp, fp, fn, tn, precision, recall, unavailable_reason}
score_index: [{record_id, label, label_status, status, reasons, margin}]  # 全部U，無分數保留null
evaluation_items: [{record_id, margin, original_label, verdict,
                    review_source, review_version}]  # 完整E0
frozen_policy: object|null  # §18.4完整操作點，非只有threshold數字
```

具有效評估時把用到的verdict及來源版本帶入報告，方便重算；status未就緒時counts/PR為null並附原因。CSV新增policy_mode、threshold、comparator、is_candidate欄位；下載all/candidates/selected都保存scope，候選ID以全部母體規則產生，不能只取螢幕已畫出的200點。

三種JSON scope都保存全U最小score_index及完整E0 evaluation_items，才能在匯出候選子集時重建未選資料造成的FN/TN。scope僅決定詳細query／鄰居items，不裁去這兩份統計索引；下載說明明示「含全批ID／分數與評估索引，不附全批影像」。未匯入人工評估時evaluation_items為空，仍可重算全批候選。

score_index帶label/status以重算所有類別總數／eligible／候選／無分數，即使該類完全沒出現在所選query或鄰居也不能遺失。不能分到有效類別的無效列獨立列「未分類／標籤無效」bucket，使每類加未分類總數等於U大小；此bucket不是參與分析的類別。

### 手算fixture

固定E0=E，四筆：

| ID | margin | verdict |
|---|---|---|
| r1 | -0.5 | incorrect |
| r2 | -0.1 | correct |
| r3 | 0.1 | incorrect |
| r4 | 0.4 | correct |

t=0.1：r3不入選，TP=1/FP=1/FN=1/TN=1，P=0.5/R=0.5。
t=0.2：TP=2/FP=1/FN=0/TN=1，P=2/3/R=1。
none：TP=FP=0/FN=2/TN=2，P=null/R=0。
all_eligible：TP=2/FP=2/FN=TN=0，P=0.5/R=1。

在U另加r5=(-0.2,unreviewed)，若r5不在事先固定E0，主候選增加但E的PR不變；若r5屬E0，主PR改為不可用。再加r6=(null,incorrect)到E0，r6不可計分，需揭露其誤標及coverage，不能把r6算TN或默默刪除。獨立測全correct集合時recall=null。

| AC | 新增驗收內容 |
|---|---|
| AC15 | `<`、t相等、重複分數、±2、none/all、null，候選筆數符合手算 |
| AC16 | slider／數字輸入單一state；精確輸入0.123456經rerun/切圖/下載不被刻度覆寫；拖動不重算embedding/鄰居、不改box、不改選取/購物車 |
| AC17 | counts來自全量；各類加總等於總量；顯示filter不改母體；候選及選取正確區分 |
| AC18 | pointer按住中間值時更新preview；最終值提交；舊request不能覆蓋新request，失敗需誠實標未達連續預覽 |
| AC19 | 人工manifest身份／label/version驗證；unknown不當correct；固定E與coverage規則成立 |
| AC20 | 手算TP/FP/FN/TN/P/R及零分母成立；無人工真值不顯示假PR |
| AC21 | 工作量／PR曲線與直接predicate一致；同分整組；當前點匹配；P不假設單調 |
| AC22 | 修改label使真值過期；修改K使分數/PR重算但不偽造新review；threshold不使embedding重算 |
| AC23 | 匯出讀回可重建候選與PR；candidate範圍含未畫出的有效資料；scope與threshold精度正確 |
| AC24 | heldout凍結完整score protocol和analysis_key；K/model/U變更、tuning ID或同原圖ROI重疊即拒絕；原始模型門檻／Labeling／來源檔案不被threshold預覽修改 |

### 第三輪補齊的驗收（與原AC共同有效）

| AC | 固定反例與通過條件 | 任務 |
|---|---|---|
| AC25 | t從0.1拖至0.4，故意延遲及逆序回覆；尚未commit時加入/保存/下載停用，舊action拒絕；commit後集合與最後完整精度值一致 | T7/T9 |
| AC26 | 跨兩表格頁勾選、換類別/排序再回來不丟ID；local清除與全域清除互不冒用；selected下載只含local∩U，不混入其他split全域選取 | T4/T5/T7 |
| AC27 | 同原圖兩ROI的模板有可辨識來源，預設unreviewed；改verdict匯入可定位剩餘未確認列；點TP/FP/FN/TN各回到四筆fixture的正確ID且主指標不變 | T8/T9 |
| AC28 | evaluation_id/version不變而改verdict，digest/PR/曲線/JSON都更新；只重排reviews不改digest；重複frozen_id、review超出E0、非法schema/purpose拒絕，空E0為合法不可計算 | T8 |
| AC29 | 同內容不同路徑ID不同、crop policy改變ID不變；兩個缺provenance rows各有不同暫存ID且不能進評估；同crop path多來源直接integrity error，不輸出正常分數 | T1/T2 |
| AC30 | 不重新Run直接套Labeling讀回使舊結果過期；flat fallback不當single，合法unknown/mix資料夾保留；人工A→B以目前B計分不被原資料夾覆寫；mode restore舊事件無效 | T1/T5/T6 |
| AC31 | 三類selected匯出只含A query且C不在鄰居，另含一筆invalid來源；三種scope都能從全U score_index重算所有類別總数/eligible/候選/無分數及全局PR，invalid詳細items/CSV僅all包含 | T3/T9 |
| AC32 | [1e308,1e308]與[1,1] cosine=1，不溢位成0；微小向量照規定排除；>10個同分rival只存前10類及完整count，block變化不破壞結果契約 | T1/T2 |
| AC33 | margin集中±0.015可縮放/重設；box統計/threshold/candidate/PR/export皆不變，視窗外門檻清楚提示 | T4/T7 |
| AC34 | Run→改原圖→按計算即stale，不能新hash配舊raw；同路徑來源或有效label改變後舊evaluation/操作點失效；未驗generation不顯示任何PR；保存/恢復t=0.123456不失精度；tuning/heldout轉換及來源群交集檢查有證據 | T0/T6/T8/T9/T10 |

驗收清單總計AC01–34。數值相等、mask membership、身份及來源hash驗證要有真正行為assertion，不以「元素存在／JSON有key」代替。UI事件驗收以延遲可控fixture測race，並至少一次真實Streamlit pointer-held操作。效能輸出必須附N/D/C/K、block尺寸、cache命中、計算呼叫次數及預覽延遲；沒有實測數值不宣稱即時性能已達成。

## 22. 延伸工作單：先工作量，再真值PR

T0正式契約需一併納入本次擴充。T0–T6完成基礎box流程後再依序做：

- [ ] **T7 Threshold候選預覽**：建立新純函式與AC15–18/25/26/33測試契約，接slider／精確輸入、每類摘要、工作量曲線、候選預覽與加入選取。先完成原生UI原型，再依pointer-held驗收決定是否需要本地component；補draft/commit與action版本核對。不能用放開更新冒充連續預覽。
- [ ] **T8 人工評估與計數**：可辨識來源的模板及待確認清單／JSON匯入、snapshot/truth digest及完整度驗證、TP/FP/FN/TN/P/R；完成AC19/20/22/27/28/34對應部分。使用人工fixture，沒有User真值也能驗證不可用狀態。
- [ ] **T9 曲線及可重現報告**：PR曲線當前點、TP/FP/FN/TN下鑽、保存/恢復相容操作點、schema2報告、tuning/heldout用途；完成AC21/23–25/27/31/34。不偷偷加入最佳門檻搜索。
- [ ] **T10 完整回歸與交接**：真實瀏覽器驗完整拖動、提交race與最終值一致性、來源不寫入、原功能回歸；更新使用說明和正式OpenSpec任務。對AC01–34逐項列具名測試及證據，完成下表T0–T6補充，報效能實測及剩餘限制。

T0–T6必須吸收r3補充，不能等T10才回頭發現：T0鎖定唯一schema/API及上游provenance/crop衝突範圍；T1/T2覆蓋AC29/30/32及eligibility傳遞；T3覆蓋AC31資料層；T4覆蓋AC26/33；T5/T6覆蓋AC26/30及scope/身份接線。每項依對應角色先定契約測試，再實作。

預計新增測試：`tests/test_class_similarity_threshold.py`、`tests/test_class_similarity_evaluation.py`、`tests/e2e/test_class_similarity_threshold_e2e.py`。核心測試和真實UI測試分開；新增component時前後端predicate亦須以同一fixture交叉驗證。

本次仍只更新設計，T7–T10未執行。沒有真值資料時，功能完成不代表已證明實際precision／recall；那需要後續具代表性的人工評估與獨立驗證。
