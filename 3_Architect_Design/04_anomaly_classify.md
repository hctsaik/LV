# 設計 04:anomaly_classify(Tier A,純函式)

## Purpose
由物件異常分數(可選少量已確認標籤)決定門檻,將整批物件分「好/壞」並排序;有標籤時回報 AUROC。

## 復用
- AUROC:`sklearn.metrics.roc_auc_score`。門檻最佳化用 Youden's J(`roc_curve` 的 `tpr-fpr` 最大點)。

## I/O 契約
```
ClassifyResult = {
  "threshold": float,
  "labels":    np.ndarray,   # (N,) dtype=object/<U4,值 ∈ {"good","bad"}
  "ranking":   list[int],    # 全體索引依分數「由高到低」;同分依索引升冪 tie-break
  "auroc":     float | None, # 有 confirmed 雙類時提供,否則 None
  "n_bad":     int,
}

def classify(
    scores: np.ndarray,                 # (N,) 異常分數,越高越可疑
    *, confirmed: dict[int, str] | None = None,  # {idx: "good"|"bad"}
    contamination: float = 0.05,        # 無標籤時:最高的這個比例判 bad
) -> ClassifyResult
```

## Data Flow(門檻三條路)
1. **有 confirmed 且含 good 與 bad**:在 confirmed 子集上用 Youden's J 取最佳門檻 `t*`(`roc_curve` 上 `tpr-fpr` 最大的 threshold);`auroc = roc_auc_score(confirmed)`。
2. **只有 bad(無 good)**:`threshold = min(confirmed_bad_scores) - eps`(eps=1e-9),確保所有已確認 bad 被判 bad;`auroc=None`。
3. **無 confirmed**:`threshold = quantile(scores, 1 - contamination)`;最高的 `contamination` 比例判 bad;`auroc=None`。
- `labels[i] = "bad" if scores[i] >= threshold else "good"`。`ranking = argsort(-scores, kind="stable")` 再以索引穩定 tie-break。`n_bad = (labels=="bad").sum()`。

## 邊界與錯誤處理
- `N==0`:`labels=[]`、`ranking=[]`、`threshold=+inf`、`auroc=None`、`n_bad=0`,不丟例外。
- confirmed 索引越界 / 值非 good|bad → 略過該筆(不丟例外);全略過則退化到「無 confirmed」路。
- 全部同分:路 3 下 `quantile` 退化 → 用 `> threshold` 會全 good;契約規定用 `>=` 故「分數==threshold」者為 bad(避免 contamination=1.0 卻 0 bad 的矛盾)。
- `contamination` 夾到 [0,1]。

## Acceptance Criteria(釘死數值)
- **AC1**(可分+雙標籤):`scores`=正常 40 個 ∈[0,0.2]、壞 10 個 ∈[0.8,1.0];`confirmed` 給其中 5 正常("good")+ 5 壞("bad")→ `0.2 < threshold < 0.8`、全部 50 個 `labels` 完全正確切開、`auroc==1.0`。
- **AC2**(無標籤百分位,精確計數):`scores=arange(100)`、`contamination=0.1` → `threshold==quantile(arange(100),0.9)`、`n_bad==10`、且被判 bad 的恰是分數最高的 10 個(idx 90..99)。
- **AC3**(排序):`ranking[0]==argmax(scores)`;`ranking` 對應分數嚴格非遞增;同分時索引升冪。
- **AC4**(只給 bad):`scores` 任意,`confirmed={j:"bad"}` → `threshold <= scores[j]`,故 `labels[j]=="bad"`,且所有 `scores>=threshold` 者皆 bad。
- **AC5**(AUROC 對齊):一組 `scores` 與 `confirmed`(含已知可算 AUROC 的子集,如 4 good/4 bad 設計成 AUROC=0.75)→ `round(auroc,3)==0.75`(與 `roc_auc_score` 一致)。
- **AC6**(退化):`N==0` 與「confirmed 全越界」皆不丟例外,結果型別/形狀正確。
