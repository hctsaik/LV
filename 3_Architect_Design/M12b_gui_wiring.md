# 設計:M12b GUI + 服務接線(M9/M10 找相似,`scripts/app.py` + al_service/al_workspace)

> 對應 PRD:[2_PO_PRD/similar_objective_prd.md](../2_PO_PRD/similar_objective_prd.md);引擎契約:[08_al_batch.md](08_al_batch.md)「增補(M12b/A3)」。
> Tier B。done = `gate.py al_batch`(含 similar,無回歸)+ `gate.py al_service`(含 similar run_once)+ M9 similar 真實 E2E 綠。

## 1. 目的
把「找相似」延伸到**大資料分批(M9)**與**離線監看服務(M10)**:挑一顆參考物件 → 整批/持續掃描
把最像它的排到佇列最前面。**參考一律來自當前②套用結果**(A2 建模物件 provenance 本輪不做)。

## 2. 服務層契約(al_workspace 09 / al_service 10,加法擴充)
- `al_workspace.default_profile(...)`:profile 加可選欄位 `reference_vector_file: str | None`(workspace 內相對路徑,如 `reference.npy`)。
- `al_service.init_workspace(..., reference_vector=None)`:`reference_vector`(np.ndarray)非 None → 存
  `<ws>/reference.npy`(原子)並把 `reference_vector_file="reference.npy"` 寫進 profile。
- `al_service.run_once(...)`:`profile.objective=="similar"` 時,載入 `<ws>/<reference_vector_file>`
  → 傳 `ref_vector=` 給 `al_batch.run_batched`;缺檔或 objective=similar 但無 reference_vector_file → 明確報錯。
- **既有 novelty 等路徑不變**(reference_vector_file 為 None → 不碰)。

## 3. GUI 契約(app.py;參考皆讀 `st.session_state["anomaly_apply_result"]`)
- **M9 `_anomaly_batch_section`**:選樣目標選單加 **「🔎 找相似」**(key 值 `similar`)。選它時:
  - 讀 `anomaly_apply_result.obj_emb/records`;無結果 → 提示「請先②套用偵測,挑一顆參考物件」+ 掃描鈕 disabled。
  - `st.number_input("參考物件索引")`(key=`anomaly_batch_ref_idx`)→ ref_vector = obj_emb[idx]。
  - `_anomaly_batch_run` 對 similar 傳 `ref_vector=obj_emb[idx]` 給 `run_batched`(objective 引擎值 `"similar"`);
    checkpoint 目錄名含 similar(換參考=另一 run 由引擎身分處理)。
- **M10 `_anomaly_watch_section`**:選樣目標加「🔎 找相似」;選它時同樣挑參考索引(key=`anomaly_watch_ref_idx`);
  `_anomaly_watch_init` 對 similar 把 `reference_vector=obj_emb[idx]` 傳 `init_workspace`(存 reference.npy + profile 欄位)。
- `_AL_ENGINE_OBJ` 加 `"similar": "similar"`(GUI 詞彙→引擎;similar 兩端同名)。

## 4. 邊界(GUI 友善)
| 情境 | 行為 |
|------|------|
| 選「找相似」但無②結果 / 無 obj_emb | 提示「請先②套用偵測產生參考物件」;掃描/初始化 disabled |
| 參考索引超出範圍 | number_input min/max 夾住(0..N-1) |
| similar 但服務缺 reference.npy | run_once 報明確錯(GUI 顯示 st.error,不 crash) |

## 5. Acceptance Criteria
### 引擎/服務(單元,gate 收)
- al_batch:AC-SIM1~SIM5(見 08 增補)—— `tests/test_al_batch.py`。
- al_service:**AC-SVC-SIM**:`init_workspace(reference_vector=v, objective="similar")` → `<ws>/reference.npy` 存在、
  profile `reference_vector_file=="reference.npy"`;`run_once` 讀它跑 similar → 佇列 `id` 對應的 topk 以參考同群為主
  (注入式 embed_fn;ref=E2 → X 群在前)。`tests/test_al_service.py`。

### M9 GUI(真實 E2E,`tests/e2e/test_al_batch_gui_e2e.py` 追加)
- **AC-BSIM(M9 分批找相似)**:含兩群資料 → ②套用(產生 obj_emb)→ M9 分批掃描選「🔎 找相似」+
  參考索引=0 → 掃描 → **標註佇列**渲染;佇列項多數與參考同群(讀 reason「相似度」+ label 多數決)。
  真實行為:最像參考的**真的**同群,非 element 存在。

> M10 GUI 的 similar 以 **AC-SVC-SIM(al_service 單元)** 覆蓋核心(profile 存取 + run_once 排序),
> GUI 接線沿用既有 watch E2E 骨架;不另立完整 M10-similar Playwright E2E(避免重複 + 控成本),
> 但 GUI 仍須實作 §3 的 watch 參考挑選 + init 傳 reference_vector。
