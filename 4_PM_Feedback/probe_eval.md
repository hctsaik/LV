# 12 probe_eval — AC ↔ 測試對應表(回饋契約)

設計:[3_Architect_Design/12_probe_eval.md](../3_Architect_Design/12_probe_eval.md) · 里程碑 M11 · Tier B(核心純函式 A 級品質)
單元測試:`tests/test_probe_eval.py`(gate:`python verify/gate.py probe_eval`)
E2E:`tests/e2e/test_probe_eval_e2e.py`(`@pytest.mark.e2e`,**不進 gate**)

> 實作註記:既有 `active_loop.stratified_pool_eval_split(labels, *, eval_frac, seed)` 的語義是
> 「pool / eval 切分」、參數是 `eval_frac`,與本設計要的 `test_size` 不合 → 依設計 §2.1 授權
> 自帶分層切分(用 sklearn `train_test_split(stratify=…)`,決定論)。**不改既有函式**。

## 單元 AC ↔ 測試

| AC | 內容 | 測試 | 初跑 |
|----|------|------|------|
| AC1 | 可分資料 → acc==1.0、macro_f1==1.0、n_train+n_test==40、n_test==10 | `test_ac1_separable_acc_perfect` | 🔴 |
| AC2 | 分層:3:1 的 80 點 → test 集內比例仍 3:1(±1 張) | `test_ac2_stratified_test_ratio` | 🔴 |
| AC3 | 決定論:同輸入重呼 → dict 逐鍵相等 | `test_ac3_deterministic` | 🔴 |
| AC4 | 類不足:某類 3 張(min_per_class=4)→ ValueError,訊息含 `insufficient_class` 與類名 | `test_ac4_insufficient_class_raises` | 🔴 |
| AC5 | plateau 判讀釘死:slope==0.002 → plateau 且 reason 含 "0.002";2 點 → insufficient;slope==0.1 → improving | `test_ac5_plateau_pinned`<br>`test_ac5_insufficient_points`<br>`test_ac5_improving` | 🔴 |
| AC6 | **不 import streamlit**:乾淨子行程 import 後 `streamlit not in sys.modules` | `test_ac6_no_streamlit_import` | 🔴 |

## 推導測試(設計未明列,PM 自加施壓)

| 測試 | 施壓的不變量 |
|------|-------------|
| `test_metamorphic_relabel_invariance` | 保序一致改名(A→cat、B→dog)→ acc / macro_f1 / 切分數不變 → 逼出「把類名混進度量」「分層依名字順序而非群組」 |
| `test_metamorphic_plateau_vertical_shift` | ys 整體平移常數 → slope / status 不變 → 逼出「用絕對值而非**增量**判 plateau」 |

## GUI / 整合 E2E AC(Tier B done 要件)

| AC | 內容 | 測試 |
|----|------|------|
| AC-G1 | 一鍵入帳:回合卡按「🧪 跑代理評估」→ 卡上出現 `probe_acc`,`output/rounds.jsonl` 出現 `kind=="metric"` 且 `source=="probe"` 的行 | `test_probe_attaches_metric_e2e` |
| AC-G2 | 曲線 + 建議:多個回合各有 probe_acc → 折線圖渲染 + 停止建議 caption(**附數字**) | `test_curve_and_advice_render_e2e` |
| AC-G3 | 類不足**誠實**:單類資料 → `st.warning` 含類名,不入帳、不崩潰 | `test_insufficient_class_warns_e2e` |

> **AC-G2 的設計不一致(PM 記錄,已與 architect 對齊)**:設計 §5 AC-G2 寫「兩個回合 → 曲線兩點
> + 建議(improving/plateau 其一)」,但 §2.1 的 `plateau_advice(min_points=3)` 與 AC5 明定「2 點 →
> **insufficient**」。二者互斥。裁決:**core 契約(AC5)為準**,E2E 以 **3 個回合**驗
> improving/plateau 附數字;曲線本身 2 點即可畫。`insufficient` 也一樣附數字(還差幾點),
> 故 AC-G2「附數字」的產品意圖在任何點數下都成立。

## done 判準

- **單元綠**:`python verify/gate.py probe_eval` 印 `GREEN`。
- **AND 真實 E2E 綠**:AC-G1~G3。
- **產品紀律(不可違約)**:「**不做自動停**」——plateau 只給建議,任何把它變成自動行為的實作
  都算違約。UI 必須常駐「代理指標僅供方向,最終以外部模型驗證為準;是否停止由你決定」。
