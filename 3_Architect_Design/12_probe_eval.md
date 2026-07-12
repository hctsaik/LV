# 12 — probe_eval 技術設計(linear probe 回合代理評估 + 報酬遞減視圖)

對應 PRD:[2_PO_PRD/al_loop_infrastructure_prd.md](../2_PO_PRD/al_loop_infrastructure_prd.md) · 里程碑 M11 · Tier B(核心純函式 A 級品質)

## 1. 目的(Purpose)

回讀完 30 秒內回答「這批標註值不值」:用 **frozen embedding + LogisticRegression**(dino_head 同款配方,
泛化到任何有 label 的記錄集)訓一個代理 probe,把分數掛進 09 的回合帳本,長出「指標 vs 累積標註量」
的報酬遞減視圖與**附理由的**停止建議。誠實界線:probe 是方向性代理,不是最終模型成績(UI 常駐標示)。

## 2. I/O 契約

### 2.1 新模組 `scripts/probe_eval.py`(依賴 numpy + sklearn;**不 import streamlit**)

```python
def train_probe(
    emb: "np.ndarray",            # (N, D) frozen embedding(呼叫端已算好/已快取)
    labels: list[str],            # (N,) 類別字串;至少 2 類、每類至少 min_per_class
    *,
    test_size: float = 0.25,
    min_per_class: int = 4,       # 任一類 < 此值 → 拋 ValueError("insufficient_class:<類名>")
    seed: int = 42,
) -> dict
    # 分層切分(重用 M5 active_loop 的分層切分 helper;若簽名不合,本模組自帶
    # stratified_split(labels, test_size, seed) 純函式,決定論)。
    # LogisticRegression(max_iter=1000, random_state=seed) fit train → 在 test 上算:
    # {"acc": float, "macro_f1": float, "per_class": {類: {"n": int, "recall": float}},
    #  "n_train": int, "n_test": int, "classes": [str...]}
    # 全決定論(同輸入同輸出)。

def plateau_advice(xs: list[int], ys: list[float], *, min_points: int = 3,
                   eps_per_100: float = 0.005) -> dict
    # 對 learning_curve 輸出(09)判讀報酬遞減:
    # 取最後兩點斜率 slope = (y[-1]-y[-2]) / max(x[-1]-x[-2], 1) * 100(每 100 張標註的指標增量)。
    # 回 {"status": "insufficient" | "improving" | "plateau",
    #     "slope_per_100": float | None, "reason": str(中文、含數字,給 UI 直接顯示)}
    # len<min_points → insufficient;slope < eps_per_100 → plateau;否則 improving。純函式。
```

### 2.2 GUI 接線(Tier B,`scripts/app.py` 📥 標註回饋「🔁 回合」視圖內)

- 回合卡加按鈕「🧪 跑代理評估」:
  - 資料來源 = **目前載入工具的 records+embedding**(viz classifier 記錄或瑕疵偵測物件記錄;
    embedding 沿用各工具已快取者,不重算);labels 取 `r["label"]`,`unknown`/空標籤的記錄剔除。
  - 成功 → `attach_metric(round_id, "probe_acc", acc, source="probe")` +
    `attach_metric(round_id, "probe_macro_f1", ...)`;卡上即時顯示,曲線多一點。
  - `insufficient_class` → st.warning 顯示是哪一類不足(不靜默)。
- 曲線區(09 已畫)下方加「停止建議」caption:`plateau_advice` 的 reason 原文顯示,
  並固定附註「代理指標僅供方向,最終以外部模型驗證為準;是否停止由你決定」。**不做自動停**。

## 3. 資料流

回讀套用(09)→ 使用者按「跑代理評估」→ 現有 embedding+labels → train_probe →
attach_metric → learning_curve 多一點 → plateau_advice 給附數字的建議。

## 4. 邊界條件與錯誤處理

| 情況 | 預期行為 |
|---|---|
| 只有 1 類 / 某類 < min_per_class | ValueError("insufficient_class:<類名>"),UI 轉 warning |
| N 很小但滿足 min_per_class | 照跑(小樣本本來就抖,由使用者判讀;不加隱藏門檻) |
| labels 含 "unknown"/"" | GUI 端先剔除再呼叫(核心函式不做剔除——單一職責) |
| 曲線點數 < 3 | plateau_advice 回 insufficient(reason 說明還差幾點) |
| x 兩點相同(同標註量) | slope 分母 clamp 1,不除零 |
| 重跑同回合 | metric 追加(09 語義:曲線取最後一筆),歷史保留 |

## 5. Acceptance Criteria

**單元(`tests/test_probe_eval.py`;合成可分 embedding)**

- **AC1(可分資料 acc==1.0)**:兩類各 20 點、線性可分(如 x 軸 ±10 的高斯)→ acc==1.0 且
  macro_f1==1.0;n_train+n_test==40 且 n_test==10(test_size=0.25)。
- **AC2(分層)**:類比 3:1 的 80 點 → test 集內兩類比例仍為 3:1(±1 張)。
- **AC3(決定論)**:同輸入重呼 → dict 逐鍵相等。
- **AC4(類不足)**:某類僅 3 張(min_per_class=4)→ 拋 ValueError 且訊息含該類名。
- **AC5(plateau 判讀釘死)**:xs=[100,200,300], ys=[.70,.80,.802] → status=="plateau" 且
  `abs(slope_per_100-0.2/100*100…)`——精確:slope=(0.802-0.80)/100*100=0.002 < 0.005 ✓,
  reason 內含 "0.002"(格式 %.3f)。xs=[100,200], ys 任意 → 需 min_points=3 → "insufficient"。
  ys=[.70,.80,.90] → "improving"(slope=0.1)。
- **AC6(不 import streamlit)**:乾淨子行程 import probe_eval 後 `"streamlit" not in sys.modules`。

**GUI / 整合(真實 E2E,`tests/e2e/test_probe_eval_e2e.py`)**

- **AC-G1(一鍵入帳)**:載入合成雙類資料集並 Run → 回合卡按「跑代理評估」→ 卡上出現
  `probe_acc` badge,`output/rounds.jsonl` 出現 kind=="metric" 且 source=="probe" 的行。
- **AC-G2(曲線+建議)**:兩個回合各有 probe_acc → 曲線兩點 + 停止建議 caption 渲染
  (improving/plateau 其一,附數字)。
- **AC-G3(類不足誠實)**:單類資料 → st.warning 含類名,不入帳、不崩潰。

## 6. 給 /pm 的提示

- AC5 的三組數字先手算寫死(上面已列),不要讓實作反推。
- AC1 的可分 fixture 與 dino_head 既有測試同手法,可參考 `tests/test_dino_head.py`(若在)。
- E2E 依賴 09 已落地(回合卡存在);排程上 12 一定在 09 之後。
- 「不做自動停」是產品紀律:任何把 plateau 變成自動行為的實作都算違約。
