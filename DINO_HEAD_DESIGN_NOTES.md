# DINO Classification Head + 閘控級聯 — 設計與實作

> **狀態:✅ 已實作(2026-06-28)。** 對應使用者提供的「DINO 工業瑕疵落地架構」中的 **Classification Head**
> 角色 + **Gated Cascade(閘控級聯)**。單元 gate 綠 + 真實 E2E 綠。

## 角色定位(摘自架構文件)
- **Memory Bank(Normal Bank)** 問:這個樣本是否偏離我已知的正常世界?(開集、攔未知)→ 既有 `anomaly_score`。
- **Classification Head** 問:若它屬於已知類別,最像哪一種瑕疵?(閉集、分已知、推論快)→ 本功能 `dino_head`。
- 成熟系統**不是替換,是雙軌**:Head 處理已知類別高速分類,Normal Bank 持續守未知。

## 閘控級聯(本實作的核心安全結構)
closed-set head 只會從已知類別選答案,對未知瑕疵會自信誤分。故用 Normal Bank 異常分數**先守門**:
```
物件 → Normal Bank 異常分數
  ├─ 異常低(< 門檻)        → 「正常」(benign 分流,不送 head)
  └─ 異常高               → Classification Head
                              ├─ head 有把握(conf ≥ min_conf) → 已知瑕疵類別
                              └─ head 沒把握                  → 「Unknown」(送人工 / active learning)
```
從**結構上**防止未知瑕疵被閉集 head 自信塞進已知 bucket(比最後才 decision-fusion 更安全)。

## 模組 `scripts/dino_head.py`(凍結 DINOv2 特徵 + linear head)
- `train_head(obj_emb, labels, *, C, seed, l2norm)` → LogisticRegression head(≥2 類;class_weight='balanced' 因瑕疵不均衡)。回 head dict{model, classes, l2norm, dim}。
- `predict_head(head, obj_emb)` → (pred_labels, confidence, proba)。維度不符明確 raise。
- `gated_predict(head, obj_emb, anomaly_scores, *, anomaly_threshold, min_conf)` → 每物件閘控標籤(正常/已知類別/Unknown)。
- `save_head`/`load_head`(joblib,atomic,自建落點目錄)。
凍結特徵 → 訓練快、推論快、易部署(joblib;後續可 ONNX 化)。

## GUI(`app._anomaly_ui` 結果區的「🏷 瑕疵分類頭」expander)
偵測後(有 obj_emb + label,≥2 類)出現:訓練/重訓分類頭 → 顯示**閘控分類結果**計數(用 Normal Bank 的
異常門檻 `result['threshold']` 守門 + 信心 slider)→ 存/讀分類頭(部署)。

## 驗證
- 單元 gate `dino_head` 6 綠(train/predict 可分準確、信心∈[0,1]、<2 類 raise、save/load round-trip、閘控級聯三路、維度不符 raise)。
- E2E `test_dino_head_e2e` 綠(偵測→訓練頭→閘控分類結果出現→存頭寫出 .joblib)。

## 未進首版(對齊架構文件 roadmap)
Defect Prototype Bank(few-shot 相似檢索)、完整 active learning priority score(novelty+disagreement+boundary+
diversity+temporal+risk)、drift 偵測、per-condition bank 自動路由、評估指標(escape/over-kill/unknown queue)、
時序差分、ONNX 匯出。head 目前用資料夾的 YOLO 類別當已知類別;真正的「瑕疵種類」標註體系待 active learning 累積。
