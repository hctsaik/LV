# 24 — audit_report 技術設計(訊號原始結果 → 量化報告資料結構)

對應 PRD:[2_PO_PRD/dataset_audit_prd.md](../2_PO_PRD/dataset_audit_prd.md) · 里程碑 M20 · **Tier A(純函式,無 I/O)**

## 1. 目的

把 25 收集到的各訊號**原始結果**(配對清單、sha、標籤、分數陣列)彙總成一份
JSON-able 報告結構:每節=**數字 + 白話一句 + method line(怎麼算+門檻)**。
method line 是 G3「數字要帶出處」的落實;本模組**不下總判詞**(v1 無校準基準,
誠實地只給數字與出處,不說「資料集健康/不健康」)。

## 2. I/O 契約

### `scripts/audit_report.py`(依賴:numpy;禁 I/O、禁 import 其他專案模組)

```python
def build_report(inputs: dict) -> dict
```

`inputs` 鍵(由 25 準備;缺鍵 → KeyError 自然拋,不吞):

| 鍵 | 型別 | 意義 |
|---|---|---|
| `paths` | [str] | 可讀影像(索引=全文件 idx) |
| `splits` | [str] | 每影像 split 名 |
| `labels` | [str] | 每影像類別("" = 無標籤) |
| `shas` | [str] | 每影像 sha256 |
| `n_unreadable` | int | 壞檔數 |
| `phash_pairs` | [(i,j,d)] | 近重複候選(不分 split) |
| `leak_pairs` | [(i,j,d)] | 跨 split 配對 |
| `emb_pairs` | [(i,j,dist)] \| None | embedding 近重複(未算=None) |
| `outlier_scores` | ndarray \| None | 逐影像離群分數(未算=None) |
| `meta` | dict \| None | 23 `join_metadata` 輸出 |
| `meta_groups` | {field: group_counts 輸出} \| None | 23 分組 |
| `params` | dict | model / phash_max_hamming / emb_max_distance / outlier_k / top_n / max_pairs / use_embedding |

回傳:

```python
{"sections": [ {"key", "title", "value",        # 主數字(int/float/None=不適用)
                "text",                          # 白話一句(含數字)
                "method",                        # 出處行(方法+門檻,非空)
                "items": [...],                  # 該節問題清單(可空)
               }, ...],                          # 順序固定,見下
 "totals": {"n_images", "n_unreadable", "split_counts": {split: n}}}
```

sections **順序固定**:`overview`、`unreadable`、`exact_dup`、`near_dup`、`leakage`、
`label_conflict`、`class_coverage`、`outliers`,最後(meta 給定時)`metadata`。

### 各節計算定義(釘死)

- **exact_dup**:sha 分組取 len>1 → `value = Σ(len−1)`(額外複本張數,**全域計,
  含跨 split 與異標的複本**——它們同時也各自計入 leakage/conflict,白話句明講);
  items=每組 {sha, paths}。
- **near_dup**:`phash_pairs` **濾掉同 sha 的對**(那是 exact)→ value=剩餘對數;
  items=每對 {path_i, path_j, hamming}。
- **leakage**:value=len(leak_pairs);items 每對含 (split_i, split_j)。
- **label_conflict**:同 sha 且 label 種類(非空)>1 → value=組數;items=每組
  {sha, labels, paths}。
- **class_coverage**:per (split,label) 張數(label=="" 不計);基準 split=張數最多者;
  value=其他 split 缺類總數;items=每 split {split, missing:[label...](排序)}。
  全部影像無標籤或僅單一 split → value=None、text="不適用(…原因)"。
- **outliers**:分數降冪取前 `top_n`;value=top_n 實際張數;items={path, score}。
  `outlier_scores is None` → value=None、text 明寫「未計算(已略過 embedding)」。
- **near-dup(embedding)**:併入 `near_dup` 節 items?否——**獨立鍵 `emb_near_dup`**
  插在 `near_dup` 之後(emb_pairs=None 時 value=None「未計算」)。
  ⚠ 修正 sections 順序為:…`near_dup`、`emb_near_dup`、`leakage`…
- **metadata**:value=meta["n_matched_images"];text 含匹配率 % 與
  `n_unmatched_images`、`len(unmatched_row_idx)`;items=每欄位的分組表
  {field, groups: {value: n}, n_missing}。
- **截斷誠實(no silent caps)**:phash/emb/leak 對數若==params["max_pairs"],
  該節 text 追加「(已達顯示上限 {max_pairs} 對,實際可能更多)」。

### method line(格式釘死,AC 逐字驗)

- exact_dup:`方法:SHA-256 內容雜湊完全相同`
- near_dup:`方法:perceptual hash(8×8)漢明距離 ≤ {phash_max_hamming},排除內容完全相同者`
- emb_near_dup:`方法:embedding cosine 距離 ≤ {emb_max_distance}(模型 {model})`
- leakage:`方法:跨 split 的 perceptual hash 配對(漢明距離 ≤ {phash_max_hamming})`
- label_conflict:`方法:SHA-256 相同但類別標註不同`
- class_coverage:`方法:各 split 的類別集合與最大 split 比對`
- outliers:`方法:embedding k={outlier_k} 近鄰平均 cosine 距離(模型 {model}),取前 {top_n}`
- metadata:`方法:CSV 以 sha256/檔名 對照(未匹配明列)`
- unreadable:`方法:完整解碼失敗(safe_io)`

## 3. 邊界條件

| 情況 | 行為 |
|---|---|
| 空 pairs / 空 groups | value=0、items=[](誠實空,不是 None) |
| labels 全 "" | conflict value=0;coverage value=None 不適用 |
| 單一 split | coverage value=None 不適用 |
| value=None 的節 | text 必含「不適用」或「未計算」字樣 |
| shas 長度與 paths 不一致 | ValueError |

## 4. Acceptance Criteria(`tests/test_audit_report.py`;全合成手算)

- **AC1(exact 手算)**:6 圖 shas=[a,a,a,b,c,c] → value==3(2+1);2 組,組員正確。
- **AC2(near 濾 exact)**:phash_pairs=[(0,1,0),(3,4,2)],sha 0==1 → value==1(只 3-4)。
- **AC3(leakage)**:leak_pairs 2 對 → value==2;items 帶 (split_i,split_j)。
- **AC4(conflict)**:同 sha 標 A/B → 1 組;同 sha 同標 → 0 組;label=="" 不觸發。
- **AC5(coverage)**:train{A:3,B:2,C:1} val{A:2} → 基準 train、value==2、
  val missing==["classB","classC"] 排序;單 split → value None + text 含「不適用」。
- **AC6(outliers)**:scores=[.1,.9,.5,.7], top_n=2 → items==[idx1, idx3] 順序;
  None → value None + text 含「未計算」。
- **AC7(metadata 節)**:meta/meta_groups 給定 → 匹配率/未匹配數/分組數字正確;
  meta=None → 無 metadata 節。
- **AC8(結構契約)**:sections 鍵順序 == 固定序;每節 method 非空字串;
  overview value==n_images。
- **AC9(白話句含數字)**:exact_dup text 含 str(value);截斷情境(len==max_pairs)
  text 含「顯示上限」。
- **AC10(防呆)**:shas 與 paths 長度不一 → ValueError。

## 5. 給 /pm 的提示

- method line 逐字比對(它是 G3 的契約,不是裝飾)。
- AC5 的缺類清單排序是**字典序**,fixture 用 classB/classC 這種可排序名。
