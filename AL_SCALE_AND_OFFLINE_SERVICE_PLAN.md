# Active Learning 大資料可用性 + 離線監看服務 — 產品規劃書(PO/User 草案)

> **性質**:這是「提案 + 開發規劃」文件,**尚未實作任何東西**。依本 repo 的 U-Net 協定,
> 這是新能力 → 必須從 `/user` → `/po` 起新輪;本文件就是給那兩個階段的**預聚合草稿**,
> 以及給後續 `/architect` `/pm` `/pg` 的執行藍圖。
> **設計給「較弱的 AI」執行**:每個模組的契約、步驟、驗收都寫到可照抄的粒度;
> 引用的既有函式**全部已核實存在**(2026-07-05 對 `scripts/` 實際 grep 確認),不是想像。
> 執行順序見 §10「執行手冊」。

---

## 0. 一句話總結

兩個問題(GUI 跑不動大資料、想要離線自動選樣服務)其實是**同一個根因**:
**重運算被綁死在 Streamlit 互動 session 裡**。解法是一體的:

> **把「掃描→embedding→評分→選樣」抽成一個可分批、可斷點續跑的無頭引擎;
> GUI 和離線服務是它的兩個前端;「已存模型」是兩者之間唯一的契約;
> 「標註佇列」是產品的核心產出物(不是散點圖)。**

---

## 1. 問題定義(User 的痛,PO 的話)

### 問題 1:資料量大時 GUI 不可用
- 現況:瑕疵偵測 wizard(①建模→②套用→③選樣)在 Streamlit session 內同步跑完整條
  pipeline。幾萬個物件時:embedding 要跑數十分鐘~數小時、UMAP 投影變慢、
  散點圖/縮圖牆渲染卡頓;中途關掉 app 全部重來(embedding 有 `.lv_cache` 快取,
  但**編排狀態沒有**)。
- 使用者提的兩個方向都對,但**各解一半**:
  - 「分批慢慢跑、最後合併」→ 解**計算**可擴展性
  - 「先設定想看的結果、只保留條件內的」→ 解**顯示**可擴展性
  - **兩者都要**,缺一不可(只分批,最後還是渲染 5 萬點散點圖照樣卡死;
    只設條件,第一次評分還是得算完全部 embedding)。

### 問題 2:離線 Active Learning 監看服務
- 想要:設定一個(會持續有新圖進來的)資料夾 → 服務自動按 AL 目標找出
  **Top-K 最值得人花時間標註/澄清的 case** → 使用者定期來看佇列、標註。
- 三個待答問題:GUI 上怎麼設定?服務跟 GUI 怎麼連動?使用者最後在哪裡標?

### 為什麼要一起做
分批引擎(問題 1)就是離線服務(問題 2)的心臟。先做引擎,GUI 和服務各自接上,
避免做兩套。

---

## 2. User 視角(給 `/user` 階段謄錄用)

### Persona
產線品管工程師/資料集管理者。不懂 ML 內部,懂「哪些圖有問題」。時間寶貴,
一天願意花在標註上的時間 ≤ 30 分鐘。資料夾每天自動進新圖(相機/產線落檔)。

### User Stories(附使用者語言的驗收)

| # | 故事 | 使用者驗收語句 |
|---|------|----------------|
| U1 | 我有 5~10 萬張圖,想「掛著讓它跑」,中途關電腦明天繼續,不要從頭來 | 「重開 app 後按繼續,進度從上次的地方接著走」 |
| U2 | 跑到一半我就想先看目前最可疑的,不想等全部跑完 | 「跑到 30% 就能看暫定 Top-K,且畫面明白標示『暫定,基於已處理 30%』」 |
| U3 | 資料量大時我不需要看每一個點,我只要今天該看的 K 張 | 「我設 K=50,畫面直接給我 50 張排好的佇列 + 一張分數分佈圖,不用等散點圖」 |
| U4 | 資料夾每天進 2000 張新圖,我每天早上只想看最值得看的 50 張 | 「服務半夜自己跑,早上打開 GUI,佇列裡就是新的 50 張,舊的、我標過的不再出現」 |
| U5 | 我要知道「為什麼選這張給我」 | 「每張卡片寫得出人話理由:很不像正常樣本/模型拿不準/跟我之前標的矛盾」 |
| U6 | 我標完的要有用:下次選樣要變聰明 | 「我確認過的正常樣本會進模型;我按下『更新模型』後,之後的佇列用新模型算」 |
| U7 | 絕對不能動到我的資料夾 | 「服務對監看資料夾只讀不寫,所有產出放別的地方」 |
| U8 | 服務掛了我要知道 | 「GUI 打開時如果服務太久沒跑,會顯眼提示上次成功時間」 |

### 明確不要(User 說了算的邊界)
- 不需要即時(分鐘級延遲即可,一天跑幾次就夠)
- 不需要多人同時標(單人使用,v1)
- 不需要 email/通知推播(打開 GUI 看得到狀態就好)

---

## 3. PO 視角:產品原則(8 條關鍵決策,執行時不可偷改)

> 這 8 條是整個功能的「憲法」。後續任何階段(architect/pm/pg)發現與它矛盾,
> 走 reverse gate 回報,不准就地繞過。

1. **一個引擎、兩個前端**:分批評分引擎只有一份(純函式模組);GUI 內嵌跑它,
   CLI 服務也跑它。禁止 GUI 和服務各寫一套 pipeline。
2. **佇列是核心產出物**:產品輸出=「排好序、附理由、可消費的標註佇列」。
   散點圖降級為(可選的)探索視圖。
3. **顯示預算三模式**:標註模式(免投影、只渲染佇列+分數直方圖)/
   探索模式(抽樣散點,上限例如 5000 點,永遠包含 Top-K 與已確認項)/
   完整模式(小資料,=現況)。依資料量自動建議,使用者可改。
4. **「已存模型」是 GUI↔服務的唯一契約**:服務只吃 GUI 一鍵存出的模型目錄
   (`_anomaly_save_model` 已存在,存 bank+head+meta)。**顯式版本升級**:
   使用者在 GUI 重存模型,服務下一輪才用新版;服務**絕不**自己重訓模型。
   佇列項目都蓋模型版本章。
5. **單寫者原則**:服務寫 `queue`/`state`,GUI 只 append 寫 `labels`;
   服務下一輪合併 labels(標過的移出待辦)。兩邊永不寫同一個檔 → 免鎖檔地獄。
6. **絕不寫使用者資料夾**(既有鐵則 [[no-dataset-writes]]):工作區(workspace)
   放 `.lv_cache/al_workspace/<profile>/` 或使用者另指定的**非資料集**目錄;
   沿用 `anomaly_bank_store.assert_safe_bank_dir` 的防呆模式。
7. **run-once + 外部排程,不做常駐 daemon**(v1):CLI 提供 `run`(跑一輪就退出)。
   排程交給 Windows 工作排程器/使用者手動/未來平台 engine。
   GUI 提供「▶ 立即掃描一次」(in-process 跑,重用分批引擎,不擋 UI 進度可見)。
8. **分批正確性原則**(技術憲法):**凍結模型下,逐項評分是獨立的 → 分批+合併
   數學上安全**;任何全域運算(bootstrap 分群、UMAP 投影、門檻校準)只能
   (a) 對抽樣跑、或 (b) 全部批次完成後跑。執行 AI 必須理解這條,否則會做出
   「分批分群」這種錯誤設計。

### MoSCoW(feature 級)

- **Must**
  - M9:分批可續跑引擎 + GUI 三模式顯示 + 暫定 Top-K
  - M10:profile 設定(GUI)+ CLI run-once 服務 + 佇列持久化 + Step③ 消費佇列
    + labels 回寫 + 增量掃描(mtime/size)
- **Should**
  - 佇列項目「人話理由」(score breakdown → 文字模板)
  - 服務健康狀態呈現(上次跑、處理量、模型版本不符警告)
  - GUI「複製排程指令」框(Task Scheduler 一行命令)
- **Could**
  - 多 profile 並存(建議做,成本低:workspace 本來就 per-profile 目錄)
  - 確認正常 → 一鍵擴 bank → 重存模型的引導動線(串既有 M5 confirm 流程)
  - 分數分佈漂移警示(新資料分佈 vs bank 建立時)
- **Won't(v1 明確不做)**
  - 常駐 daemon / 檔案系統 watch(用輪詢掃描)
  - 服務自動重訓/自動更新模型
  - 多人並行標註、權限
  - CIM 平台(nativeApp)整合:engine 排程 LV 服務、佇列餵給 Labeling plugin
    (X-AnyLabeling)——**留鉤子不實作**;`scripts/labeling_handoff.py` 已存在,
    architect 設計佇列 schema 時讀一下它,保持欄位可對接
  - email/推播通知

### Appetite
- M9:1 輪(1 個核心模組 + GUI 接線)
- M10:1 輪(2 個模組 + GUI 接線)
- 兩輪分開驗收,M9 綠了才開 M10(M10 依賴 M9 的引擎)。

### 規模假設(誠實的數學,architect 給 AC 時用)
- DINOv2 vits14 CPU 上物件 crop embedding 約 5~20 obj/s → 5 萬物件首跑約 1~3 小時
  → **首跑必須可斷點續跑(U1),之後增量只算新圖(U4)**。
- 評分(對 bank NN)與選樣(priority + diversity)相對便宜(秒~分鐘級)。
- UMAP 對 5 萬點要數分鐘且記憶體大 → 佐證「標註模式免投影」。

---

## 4. 總體架構藍圖

```
                ┌─────────────────────────────────────────────┐
                │              al_batch(M9,核心引擎)          │
                │  discover(yolo/whole_image) → embed(快取)    │
                │  → score(vs 已存模型) → priority → Top-K     │
                │  特性:分批、checkpoint、可續跑、progress cb   │
                └───────────────┬─────────────────────────────┘
          in-process 呼叫        │                 CLI 呼叫
   ┌────────────────────────┐   │   ┌─────────────────────────────┐
   │  GUI(scripts/app.py)  │◄──┴──►│  al_service(M10,CLI)        │
   │  wizard ①②③ + ④監看   │       │  run-once:增量掃描→引擎→佇列 │
   └───────────┬────────────┘       └──────────────┬──────────────┘
               │ 讀 queue / append labels           │ 寫 queue/state,合併 labels
               ▼                                    ▼
        ┌──────────────────────────────────────────────────┐
        │   al_workspace(M10,工作區,.lv_cache 或自訂)     │
        │   profile.yaml / state.json / queue.jsonl         │
        │   labels.jsonl / runs/*.json / lock               │
        └──────────────────────────────────────────────────┘
               ▲
               │ 只讀
        ┌──────┴───────────┐
        │ 已存模型目錄       │  ← GUI「存模型」產生(既有 _anomaly_save_model)
        │ bank+head+meta    │     服務只讀;版本=內容 hash
        └──────────────────┘
```

### 資料流(每日循環,U4)
1. 半夜:排程器跑 `al_service run --profile <dir>` → 增量掃描發現 2000 張新圖
   → 分批 embed(進 `.lv_cache`)→ 用已存模型評分 → priority + diversity 選樣
   → 合併昨天的 labels(標過的移出)→ 重寫 queue.jsonl(atomic)→ 寫 run summary → 退出
2. 早上:使用者開 GUI → wizard ④ 顯示服務狀態(上次跑 03:12,新增 2000、佇列 50)
   → Step③ 切「監看佇列」→ 逐張看縮圖+理由 → 確認正常/瑕疵/略過(append labels.jsonl)
3. 累積夠多確認後:GUI 引導「把確認正常的擴進 bank → 重存模型」→ 服務下輪用新版

---

## 5. 里程碑與模組拆解

### M9 — 大資料 GUI 可用性(分批引擎 + 顯示預算)

| 模組 | Tier | 內容 |
|------|------|------|
| `al_batch` | **A**(純邏輯,無 GUI) | 分批編排引擎:批次迭代、checkpoint、續跑、部分 Top-K 合併 |
| GUI 接線(app.py) | **B**(E2E 必須) | 三模式顯示、進度+暫定 Top-K、續跑按鈕、抽樣散點 |

### M10 — 離線 AL 監看服務

| 模組 | Tier | 內容 |
|------|------|------|
| `al_workspace` | **A** | workspace/schema:profile、state、queue、labels、runs 的讀寫與驗證 |
| `al_service` | **B**(有 I/O + CLI) | run-once 編排:增量掃描→al_batch→佇列物化→labels 合併→lock→summary |
| GUI 接線(app.py) | **B**(E2E 必須) | wizard ④「監看」分頁(profile CRUD+狀態+立即掃描)、Step③ 佇列來源切換+標註動作 |

**相依**:al_service → al_batch + al_workspace;GUI → 全部。無環。

---

## 6. 模組契約草案(給 `/architect` 的起點,可調整但不可偷工)

> 以下簽名是**草案**,architect 有最終決定權;但「必須重用的既有函式」清單是硬的。

### 6.1 `al_batch`(M9,Tier A)

**必須重用(已核實存在)**:
- `anomaly_tool.run_pipeline(...)` — 已有 `progress` callback、`external_bank`/`external_ref`
  注入、`object_source="yolo"|"whole_image"`、`cache_dir`、`embed_fn`(測試可注入假 embedder)
- `active_learning.priority_score(anomaly_scores, *, head_proba=None, anomaly_threshold=None, ...)`
- `active_learning.select_for_labeling(priority, *, k, cluster_labels=None, ...)`
- `active_loop.confusion_targeted_priority(anomaly_scores, head_proba, *, ...)`(澄清型目標用)
- `dino_head.load_head` / `dino_head.gated_predict`
- `anomaly_bank_store.load_bank` / `check_compat`
- `safe_io.partition_readable`(壞檔過濾,run_pipeline 內已用)

**新增介面(草案)**:
```python
def run_batched(image_paths, *, model_dir, objective: str, k: int,
                batch_size: int = 1000, checkpoint_dir, object_source: str,
                class_names=None, progress=None, resume: bool = True) -> dict:
    """分批跑 discover→embed→score,逐批寫 checkpoint;回傳
    {"scores": ..., "meta": ..., "topk": ..., "done": bool, "processed": int, "total": int}。
    resume=True 且 checkpoint 存在 → 從斷點接續。
    objective ∈ {"novelty"(bank-only), "uncertain"(需 head), "confusion"(需 labels)}
    → 內部映射到 priority_score / confusion_targeted_priority 的參數組合。"""
```
**設計要點(憲法第 8 條的落實)**:
- 逐批只做「discover+embed+對凍結模型評分」;priority/diversity/Top-K 在
  **已完成批次的併集**上重算(便宜,O(N) ~ O(N log N)),所以任何時刻都能給
  「暫定 Top-K(基於已處理 x%)」。
- checkpoint 內容:已處理 path 清單(或 index 游標)+ 逐項 score 的增量檔
  (`.npz`/`.jsonl` 分批檔,合併時 concat)。checkpoint 放 caller 指定目錄
  (GUI 用 `.lv_cache/...`,服務用 workspace),**不放資料集**。
- 寫檔一律 temp+rename(照抄 `anomaly_bank_store._atomic_text/_atomic_npz` 的模式)。

**驗收(AC 草案,PM 落測試)**:
- AC1 正確性:同一資料(≤ 數百張合成圖,fake embedder)分批(batch_size=7)與
  一次跑完,最終 scores/Top-K **完全一致**(排序穩定,tie-break 用 path)。
- AC2 續跑:跑到第 2 批後模擬中斷(丟例外/直接殺),`resume=True` 重呼叫 →
  已處理批次**不重算**(以 fake embedder 的呼叫計數斷言),最終結果與不中斷一致。
- AC3 暫定 Top-K:progress callback 每批收到 `{processed,total,topk_partial}`,
  且部分 Top-K 是「已處理併集」的正確 Top-K。
- AC4 目標切換:三種 objective 在合成資料上選出可預期的不同集合
  (novelty 選離群、uncertain 選 head 機率接近均勻者——合成資料可構造)。
- AC5 無資料集寫入:跑完後對資料集資料夾做 mtime/檔案清單快照比對,零變化。

### 6.2 M9 GUI 接線(app.py,Tier B)

- 入口:wizard ② 套用偵測時,若目標物件數 > 門檻(建議 5000,常數可調)→
  預設切「標註模式」,提示文案講清楚差異;segmented_control 或 radio 三模式。
- 標註模式渲染:分數直方圖(`st.plotly_chart`,O(bins))+ Top-K 佇列(沿用 Step③
  的卡片元件)+「載入更多」。**不算投影、不畫散點**。
- 探索模式:抽樣散點(分層抽樣 ≤5000 點 + 強制包含 Top-K/已確認),
  圖上標註「顯示 5000/48213(抽樣)」。
- 進度:`run_batched` 的 progress → `st.progress` + 暫定 Top-K 區塊,標示
  「暫定 · 已處理 x%」;中斷後重進頁面出現「⏵ 繼續上次(x%)」鈕。
- **Streamlit 效能鐵則**(照抄平台規範):禁止 rerun 內全量 `json.loads`/`exists()`
  迴圈;列表分頁;禁 loop 內 `list.index()`。

**E2E(AC 草案)**:
- 合成 ~30 物件 + 門檻暫調成 10(測試經 env/設定注入,不 hardcode 改門檻):
  進入標註模式 → 佇列與直方圖出現、散點不渲染;切探索模式 → 抽樣散點出現。
- 暫定 Top-K:batch_size 調小,斷言進度途中佇列區已有卡片且帶「暫定」字樣。
- 續跑:E2E 較難模擬殺程序 → 單元層 AC2 已蓋,E2E 只驗「繼續」鈕存在與可按。

### 6.3 `al_workspace`(M10,Tier A)

**Workspace 目錄佈局(草案)**:
```
<workspace>/                     # .lv_cache/al_workspace/<name>/ 或使用者自訂(需過 assert_safe 檢查)
  profile.yaml                   # 設定(人可讀可手改)
  state.json                     # 增量掃描狀態:{path: {mtime, size}}, last_run, schema_version
  queue.jsonl                    # 佇列(服務寫,GUI 只讀)
  labels.jsonl                   # 人工決定(GUI append,服務只讀+下輪合併)
  runs/<ts>.json                 # 每輪摘要:new/scored/queued/duration/model_version
  lock                           # 服務執行鎖(含 PID + 開始時間;過期鎖可搶)
```

**profile.yaml(草案)**:
```yaml
schema_version: 1
name: line3-pills
watch_folders: ["D:/data/line3/incoming"]     # 只讀
model_dir: "D:/lv_models/pill_v3"             # GUI 一鍵存出的模型目錄
object_source: whole_image                     # 或 yolo
objective: novelty                             # novelty | uncertain | confusion
k: 50
score_mode: patch                              # 對齊既有 run_pipeline 參數
target_res: 224
batch_size: 1000
filters: {min_score: null, classes: []}        # Could
```

**queue.jsonl 單筆(草案;欄位命名時參考 `labeling_handoff.py` 以利未來對接)**:
```json
{"id": "<sha1(path|bbox|model_version)>", "image_path": "...", "bbox": null,
 "obj_index": 0, "score": 0.87,
 "breakdown": {"novelty": 0.9, "boundary": 0.3, "disagreement": 0.1},
 "reason": "非常不像正常樣本(novelty 0.90)",
 "model_version": "<hash>", "run_id": "20260705T031201",
 "status": "pending", "created_at": "..."}
```
**labels.jsonl 單筆**:`{"id": "...", "decision": "good|defect|skip", "label": "刮痕",
"decided_at": "...", "model_version_at_decision": "..."}`

**介面(草案)**:load/save profile(schema 驗證+預設值)、read/append labels、
atomic rewrite queue、state diff(`scan_incremental(folders, state) -> (new, changed, deleted)`)、
`model_version(model_dir) -> str`(bank/head/meta 檔案內容 hash;`model_manifest.py`
已存在,architect 先讀它決定重用或擴充)、lock 取得/釋放/過期搶佔。

**AC 草案**:
- AC1 round-trip:profile/queue/labels 寫→讀 完全一致;壞 YAML/缺欄位 → 明確錯誤。
- AC2 增量掃描:新增/修改(mtime 變)/刪除 檔案各自被正確分類;第二次掃描零變化 → 空 diff。
- AC3 原子性:rewrite queue 途中模擬失敗(monkeypatch rename 前丟例外)→ 舊檔完好。
- AC4 安全:workspace 指到 watch_folder 內 → 拒絕(重用 `assert_safe_bank_dir` 模式)。
- AC5 lock:第二個取鎖者失敗;PID 已死的過期鎖可搶。

### 6.4 `al_service`(M10,Tier B)

**CLI(草案)**:
```
python scripts/al_service.py run    --profile <workspace_dir>   # 跑一輪就退出,exit code 0/非0
python scripts/al_service.py status --profile <workspace_dir>   # 印上次 run 摘要(給人/排程健檢)
python scripts/al_service.py init   --profile <workspace_dir> --model-dir ... --watch ... [--k 50 ...]
```
**run 的流程**:取鎖 → 載 profile → `check_compat` 驗模型 → 增量掃描 → 對新/變更檔跑
`al_batch.run_batched`(checkpoint 在 workspace)→ 讀 labels 合併(labeled/skip 移出
pending;deleted 檔標 stale)→ 重算全體 pending 的 priority → Top-K 物化 queue.jsonl
(atomic)→ 寫 runs/<ts>.json → 釋放鎖。

**AC 草案(全部用合成資料 + fake embedder,無真模型)**:
- AC1 首輪:空 state + 20 張合成圖(其中 3 張構造成離群)→ queue 有 K 筆、
  離群 3 張在最前、每筆有 reason 與 model_version。
- AC2 增量:再加 10 張新圖 → 第二輪只 embed 10 張(fake embedder 計數斷言),
  queue 重新排序正確。
- AC3 閉環:對 queue 前 2 筆 append labels(good/defect)→ 第三輪這 2 筆不再是
  pending;runs 摘要 counters 正確。
- AC4 模型版本:換一個 model_dir(不同內容)→ 新 queue 項目 model_version 改變;
  舊 pending 項目標示版本過期(策略由 architect 定:重評分或標 stale——建議重評分,
  因為 embedding 有快取、重評分便宜)。
- AC5 壞輸入:watch 資料夾不存在/模型目錄缺檔 → 非零 exit + 人話錯誤訊息 +
  不留半寫檔案;執行中檔案被外部寫入(不可讀)→ 跳過並記入 summary,下輪重試。
- AC6 無資料集寫入:同 al_batch AC5。
- AC7 併發:run 進行中再起一個 run → 立即失敗退出,exit code 明確。

### 6.5 M10 GUI 接線(app.py,Tier B)

**設定 UX(wizard 新第 ④ 段「🛰 持續監看」)**:
- Profile 清單(掃 `.lv_cache/al_workspace/*/profile.yaml` + 記住使用者自訂路徑)
  → 選取/新建/編輯。新建表單:名稱、監看資料夾(沿用單一資料夾選取器)、
  模型目錄(預設=目前 `anomaly_model_dir`)、目標 preset 三選一(**用人話**:
  「抓沒看過的異常」novelty /「模型拿不準的」uncertain /「跟已標註矛盾的」confusion,
  後兩者在無 head/無標註時反灰並說明原因)、K、object_source。
- 狀態卡:上次成功時間(過久→顯眼警告,U8)、上輪 new/scored/queued、
  目前模型版本 vs profile 模型版本(不符→黃色警告「服務用的是舊模型,重存後生效」)。
- 動作:「▶ 立即掃描一次」(in-process 呼叫 al_service 的 run 函式,重用 M9 進度 UI;
  **不是** spawn 子程序,v1 求簡)+「📋 複製排程指令」(顯示
  `python scripts/al_service.py run --profile ...` 與 Task Scheduler 設定簡述)。

**消費 UX(Step③ 擴充)**:
- 佇列來源切換:「本次 Run / 監看佇列(選 profile)」。
- 監看佇列卡片:縮圖(經既有 crop/cache 機制)+ reason + breakdown +
  三鍵:✅ 確認正常 / 🏷 瑕疵(選類別) / ⏭ 略過 → append `labels.jsonl` 並即時
  從畫面移除(樂觀更新);分頁(PAGE_SIZE=50)。
- 確認正常的項目 → 提示「已累積 N 個確認,可擴充 Normal Bank 並重存模型」
  連到既有 ①(M5 confirm→bank 流程)。

**E2E(AC 草案;重用 tests/e2e harness + 合成資料 + 假 embedder 環境)**:
- 建 profile(UI 表單)→「立即掃描一次」→ 狀態卡數字出現 → Step③ 切監看佇列
  → 卡片有 reason → 點「確認正常」→ labels.jsonl 多一行(檔案層斷言)→
  再掃描一次 → 該卡片消失。
- 模型版本不符警告:改存新模型 → 警告出現。

---

## 7. 需要 User/PO 拍板的開放問題(執行前先答)

| # | 問題 | 建議預設 |
|---|------|----------|
| Q1 | K 預設值? | 50(一次坐下來標得完的量) |
| Q2 | 多 profile 並存 v1 就做? | 是(成本低,workspace 天然隔離) |
| Q3 | 版本過期的 pending 項目:重評分或標 stale? | 重評分(embedding 有快取,便宜且誠實) |
| Q4 | 「標註模式」自動切換門檻 N? | 5000 物件(可在進階設定調) |
| Q5 | 服務跑的機器=看 GUI 的機器? | v1 假設同一台(workspace 本機路徑);跨機留待平台整合 |
| Q6 | 監看資料夾 = 建模用資料夾時要擋嗎? | 警告不阻擋(有人就是要「全量巡檢自己」) |
| Q7 | deleted 檔案的佇列項目? | 標 stale 並隱藏,不刪(保留審計) |

---

## 8. 風險與對策(PO 誠實清單)

| 風險 | 對策 |
|------|------|
| 首跑數小時,使用者以為當機 | U2 暫定 Top-K + 進度 + 預估剩餘時間(粗略即可) |
| mtime 在網路磁碟不可靠 | state 用 mtime+size 雙鍵;文件註明網路磁碟限制;Could: 可選 hash 模式 |
| 檔案寫到一半被掃到 | `partition_readable` 已擋壞檔;跳過+下輪重試(al_service AC5) |
| 佇列與目前 GUI 模型不同版 | 版本章+警告(6.5);不自動遷移 |
| 弱 AI 把全域運算塞進批次 | 憲法第 8 條 + al_batch AC1(分批=一次跑完 全等)機器擋 |
| 弱 AI 讓 GUI 長迴圈卡 rerun | E2E + Streamlit 效能鐵則寫進設計文件 AC |
| Windows 排程環境(venv、WDAC) | CLI 用 `py -3.11` 絕對路徑寫在「複製排程指令」裡;服務只跑 python(WDAC 白名單內) |

---

## 9. 與既有資產的關係(執行 AI 必讀:先讀這些,嚴禁重造)

| 既有資產 | 在本功能的角色 |
|----------|----------------|
| `anomaly_tool.run_pipeline`(progress/external_bank/object_source/cache_dir/embed_fn) | al_batch 的內核;分批=在它外面編排,或 architect 決定抽它的內部階段 |
| `active_learning.priority_score` / `select_for_labeling` | 目標函數與多樣性選樣,直接用 |
| `active_loop.confusion_targeted_priority` 等 | confusion 目標,直接用 |
| `anomaly_bank_store.load_bank/check_compat/assert_safe_bank_dir/_atomic_*` | 模型載入、相容檢查、安全目錄、原子寫模式 |
| `dino_head.load_head/gated_predict` | uncertain/confusion 目標需要的 head 推論 |
| `app.py _anomaly_save_model/_anomaly_load_model`(~1384/1426) | GUI↔服務模型契約的現成實作 |
| `model_manifest.py` | 模型版本 hash 的候選基礎(architect 先讀再決定) |
| `labeling_handoff.py` | 佇列 schema 命名對齊(未來平台 Labeling 對接) |
| `safe_io.partition_readable` | 壞檔/半寫檔防禦 |
| `.lv_cache` embedding 快取 + `patch_features.embed_objects_patch` | 增量的根基:同圖不重算 |
| tests/e2e harness、fake embedder(tests/conftest.py 384-d fake) | 全部測試不碰真模型 |

---

## 10. 執行手冊(給較弱的 AI 的操作順序)

> 本 repo 的 U-Net 協定就是為「弱執行者」設計的護欄。**照走,不要跳階段。**

1. **`/user`**:把本文件 §2 謄成 `1_user_needs/al_scale_offline.md`(User 口吻,
   不含技術)。兩個 feature 可分開兩份(M9/M10)或一份標明兩期。
2. **`/po`**:把 §3/§5/§7(拍板後)收斂成 `2_PO_PRD/al_scale_offline_prd.md`
   + 更新 `ROADMAP.md`(M9/M10 里程碑、模組表、appetite、決策日誌記「起新輪」)。
   §3 的 8 條產品原則原文抄進 PRD 的「不可違反約束」。
3. **`/architect`**(每模組一份):`3_Architect_Design/08_al_batch.md`、
   `09_al_workspace.md`、`10_al_service.md`(編號接續既有 07)。以 §6 為起點,
   讀 §9 清單裡的原始碼後把簽名定案、AC 給到可測數值。GUI 接線設計併入
   對應模組文件或獨立短文件。**不寫實作**。
4. **`/pm`**:先紅的驗收測試 → `tests/test_al_batch.py`、`tests/test_al_workspace.py`、
   `tests/test_al_service.py`(每測試註解 `# ACn`);E2E →
   `tests/e2e/test_al_scale_e2e.py`、`tests/e2e/test_al_service_e2e.py`(標
   `@pytest.mark.e2e`,重用既有 harness);寫 `4_PM_Feedback/<module>.md` 對應表;
   跑 `python verify/gate.py --snapshot`。
5. **`/pg`**(每模組):實作進 `scripts/al_batch.py` 等;`python verify/gate.py
   al_batch` 印 GREEN 才算過;GUI 部分做完交 E2E。**嚴禁**:改 tests/、改設計、
   skip/xfail、吞例外換綠(鐵則 2)。
6. **E2E 執行紀律**:跑前清殘留 Streamlit server(見
   `.claude/skills/visuallatent-e2e/SKILL.md`);全套偶發紅先單獨重跑判別
   (已知長 session 假紅模式,見 `MAINT_2026-07_FOLLOWUP_PUNCHLIST.md` §通用協定)。
7. **每模組完成**:ROADMAP 狀態更新 + 決策日誌一行;全部完成後跑一次完整
   `pytest tests/e2e/ -m e2e` 0 failed 才宣告里程碑 ✅。

### 給執行 AI 的「常見錯誤預防」清單
- ❌ 在批次內跑 UMAP/分群/門檻校準(違反憲法 8)
- ❌ GUI 直接 spawn 常駐子程序管理服務(違反憲法 7)
- ❌ 服務自己重訓 head 或改 bank(違反憲法 4)
- ❌ queue 與 labels 寫同一檔(違反憲法 5)
- ❌ workspace/checkpoint 放進使用者資料夾(違反憲法 6;有 AC 機器擋)
- ❌ 重寫 embedding/評分邏輯(§9 全有現成的)
- ❌ Streamlit rerun 迴圈裡做全量 I/O、或 `time.sleep`+`st.rerun` polling
- ❌ 測試用真 DINOv2(一律 fake embedder;真模型只在人工/E2E smoke 驗證)

---

## 11. 驗收總覽(全部做完的 Definition of Done)

- [ ] `verify/gate.py al_batch / al_workspace / al_service` 全 GREEN
- [ ] 新增 E2E 全綠 + 既有完整 `pytest tests/e2e/ -m e2e` 0 failed
- [ ] U1~U8 每條 user story 可在真實 app 演示(人工 golden path 走一次)
- [ ] 真實資料 smoke:一個真資料夾(≥1k 物件)首跑→中斷→續跑→佇列→標 3 筆→
      第二輪佇列反映標註(此項人工,不進 CI)
- [ ] ROADMAP:M9/M10 ✅ + 決策日誌;開放問題 Q1~Q7 的拍板記錄在 PRD
- [ ] 文件:README 或 docs 增「離線監看服務」使用說明(含排程設定範例)
