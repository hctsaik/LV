# 設計:11 `prelabel`(看過類別的分類預標 → 安全 YOLO 標註檔)

> 對應 PRD:[2_PO_PRD/prelabel_prd.md](../2_PO_PRD/prelabel_prd.md)。Tier B(檔案 I/O + C6 安全;
> 無 GUI/非同步 → gate=單元 + 真實 tmp_path 檔案系統測試,對齊 08/09/10)。
> **本檔只出契約 + AC,不寫實作。**

## 1. 目的(Purpose)

把凍結分類頭的閘控判定(`gated_predict`)轉成「可被接受的預標決定」,再輸出成 YOLO 標註檔
**安全地**寫到使用者另選目錄(絕不碰來源資料集,C6)。人仍是最終裁決:模型只填、只建議。

## 2. I/O 契約

模組檔:`scripts/prelabel.py`。**復用** `dino_head.gated_predict` / `predict_head`、
`anomaly_bank_store._atomic_text`;**不重新發明**閘控/原子寫。

### 2.1 `prelabel_records`

```python
def prelabel_records(records, obj_emb, scores, head, *, anomaly_threshold,
                     min_conf=0.5, target_classes=None) -> list[dict]:
    """每物件回 {"item": i, "pred": str, "conf": float, "accept": bool}。
    - pred = gated_predict(head, obj_emb, scores, anomaly_threshold=, min_conf=)[i]
             ∈ {"正常", "Unknown", <已知類別名>}(min_conf 已內含於閘控)。
    - conf = predict_head(head, obj_emb)[1][i](最大類別機率;正常/Unknown 仍給值,僅供顯示)。
    - accept = (pred 為已知類別,即 pred ∉ {"正常","Unknown"})
               AND (target_classes is None OR pred in target_classes)。
    - records 僅供長度對齊/追溯;決策只依 obj_emb/scores/head。
    """
```

- `records`:`[{image_path, label, class_id, bbox=(cx,cy,w,h), obj_index, ...}, …]`(既有 pipeline 記錄;
  `bbox` 已是正規化 YOLO)。
- `obj_emb`:形狀 `(N, dim)`;`scores`:長度 `N` 異常分數;三者長度須一致。
- `head`:`dino_head` 的頭 dict(含 `model`/`classes`/`dim`)。
- `target_classes`:`list[str] | None`;None=接受全部已知類別。

### 2.2 `to_yolo_lines`

```python
def to_yolo_lines(records, decisions, *, class_names) -> dict:
    """依 image_path 分組回 {image_path: ["<cls_id> cx cy w h", …]}。
    - 只納入 decision["accept"] 為 True 且 pred ∉ {"正常","Unknown"} 的物件(雙保險)。
    - cls_id = class_names.index(pred)(**以 head 類別空間為準**,見 PRD 開放問題#4);
      pred 不在 class_names → ValueError(訊息含該 pred)。
    - 座標取 records[decision["item"]]["bbox"]=(cx,cy,w,h),格式化為 6 位小數。
    - 無任何 accept 物件的影像不出現在回傳 dict(不寫空檔)。
    """
```

行格式(釘死):`f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"`(單一空格分隔,無行尾空白)。

### 2.3 `export_prelabels`

```python
def export_prelabels(lines_by_image, out_dir, *, class_names=None, source_dirs=()) -> dict:
    """把 to_yolo_lines 的結果寫到 out_dir(原子),回 {"written": int, "objects": int, "out_dir": str}。
    - 安全(C6,見 §4):out_dir 不得等於/位於/包含任一 source_dirs;不得含 images/ 子目錄。違反→ValueError。
    - 寫 out_dir/labels/<Path(image_path).stem>.txt(內容=該圖所有行,以 \\n 連接 + 末尾換行),原子寫。
    - class_names 提供時,另寫 out_dir/classes.txt(每行一類,末尾換行)= 輸出標籤空間對照。
    - written=寫出的 .txt 檔數;objects=總標註行數。
    """
```

### 2.4 C6 安全檢查(公開輔助)

```python
def assert_safe_prelabel_dir(out_dir, source_dirs=()) -> None:
    """out_dir 含 images/ 子目錄 → raise(像資料集);out_dir 與任一 source_dir 有
    等於/祖先/後代關係 → raise。允許已含 labels/ 的目錄(支援重複匯出——這是與
    assert_safe_bank_dir 的關鍵差異:後者見 labels/ 即誤判為資料集)。"""
```

## 3. 資料流(Data Flow)

```
② 套用結果(records, obj_emb, scores) + ① 模型 head + 使用者(min_conf, target_classes)
   → prelabel_records ─→ decisions[{item,pred,conf,accept}]
   → (GUI 預覽表:pred/conf/accept + 縮圖;人看)
   → to_yolo_lines(records, decisions, class_names=head.classes) ─→ {image_path: [lines]}
   → export_prelabels(…, out_dir=使用者另選, source_dirs=來源/監看) ─→ out_dir/labels/*.txt (+classes.txt)
```

- 決策全 per-object,無跨物件全域運算(與 C8 無關,但天然可分批)。
- 標籤空間以 **head.classes** 為準;GUI 傳 `class_names=head["classes"]`,並把同一份寫進 classes.txt。

## 4. 邊界條件與錯誤處理

| 情境 | 行為 |
|------|------|
| `head is None` | `prelabel_records` raise `ValueError`(訊息含「分類頭」),**不靜默降級/亂猜** |
| `len(obj_emb) != len(records)` 或 `!= len(scores)` | `prelabel_records` raise `ValueError`(長度不符) |
| `records == []` | `prelabel_records` 回 `[]`;`to_yolo_lines` 回 `{}` |
| pred 為「正常」或「Unknown」 | 不 accept、不輸出 YOLO 行(留白給人) |
| `target_classes` 不含某已知 pred | 該物件 accept=False(留給人) |
| pred 不在 `class_names` | `to_yolo_lines` raise `ValueError`(訊息含該 pred) |
| out_dir 位於/等於/包含來源 | `export_prelabels` raise `ValueError`(C6) |
| out_dir 含 `images/` | `export_prelabels` raise `ValueError`(像資料集) |
| out_dir 已含 `labels/`(重複匯出) | **允許**(覆寫同名 .txt,原子) |
| `lines_by_image == {}` | 寫 0 檔(仍可寫 classes.txt),`written=0`,不 crash |

## 5. Acceptance Criteria(釘死;給 `/pm`)

> 需要 head 的 AC(AC1–AC6)用 `dino_head.train_head` 在**線性可分**的小 2D embedding 上訓真頭
> → 判定確定;閘控走向用「分數 vs 門檻」與極端 `min_conf` 強制(不依賴 head 機率精確值)。
> `to_yolo_lines`/`export_prelabels`(AC7–AC16)為純資料/檔案,無需 head。

**prelabel_records**
- **AC1(正常分流)**:2 物件,`scores=[0.10, 0.90]`,`anomaly_threshold=0.5`,`min_conf=0.0`,
  `target_classes=None`,真頭(2 類 scratch/stain)。→ `r[0]["pred"]=="正常"` 且 `r[0]["accept"] is False`;
  `r[1]["pred"] in {"scratch","stain"}` 且 `r[1]["accept"] is True`。
- **AC2(Unknown 留白)**:同 AC1 但 `min_conf=1.1`(不可能達到)。→ `r[1]["pred"]=="Unknown"` 且
  `r[1]["accept"] is False`(分數高於門檻但沒把握 → 留白)。
- **AC3(target_classes 過濾)**:令物件 1 的真頭預測為 `"scratch"`(可分資料保證),
  `scores=[0.9]`,`min_conf=0.0`。`target_classes=["stain"]` → `accept is False`;
  `target_classes=["scratch"]` → `accept is True`。
- **AC4(head=None)**:`head=None` → `ValueError`,訊息含「分類頭」。
- **AC5(長度不符)**:`len(obj_emb)=3` 但 `len(records)=2` → `ValueError`。
- **AC6(空輸入)**:`records=[], obj_emb=[], scores=[]` → 回 `[]`。

**to_yolo_lines**
- **AC7(只含 accept + 格式)**:records 2 物件同圖 `/d/img_a.jpg`,
  bbox=`(0.5,0.5,0.4,0.6)`(item0)、`(0.25,0.75,0.2,0.2)`(item1);
  decisions=`[{item:0,pred:"scratch",accept:True},{item:1,pred:"stain",accept:False}]`,
  `class_names=["scratch","stain"]`。→ `{"/d/img_a.jpg": ["0 0.500000 0.500000 0.400000 0.600000"]}`
  (只 item0;cls_id=0=index("scratch"))。
- **AC8(往返誤差)**:解析 AC7 輸出行的 4 個座標 → 與原 bbox 各差 `< 1e-6`。
- **AC9(pred 不在 class_names)**:decision `pred="ghost"`(accept=True),`class_names=["scratch","stain"]`
  → `ValueError`,訊息含 `"ghost"`。
- **AC10(正常/Unknown 不輸出)**:decision `{item:0,pred:"正常",accept:True}`(即使 accept 被誤設 True)
  → 該物件**不出現**在輸出(雙保險過濾)。
- **AC11(多圖分組)**:兩張圖各 1 accept 物件 → 回傳 dict 有 2 個 key,各 1 行。

**export_prelabels**
- **AC12(寫檔 + 讀回)**:`lines_by_image={"/x/img_a.jpg":["0 0.5 0.5 0.4 0.6"]}`,`out_dir=tmp/out`
  → `tmp/out/labels/img_a.txt` 存在,內容 `== "0 0.5 0.5 0.4 0.6\n"`;回傳 `written==1, objects==1`。
- **AC13(C6 拒寫來源內)**:`out_dir` 設為某 `source_dirs` 內(或等於)→ `ValueError`(訊息含「來源」或「C6」)。
- **AC14(C6 拒寫含 images/)**:`out_dir` 下先建 `images/` 子目錄 → `export_prelabels` raise `ValueError`。
- **AC15(重複匯出允許)**:先匯出一次(產生 `labels/`),同 out_dir 再匯出 → **不 raise**,覆寫成功。
- **AC16(classes.txt + 空輸入)**:`class_names=["scratch","stain"]`,`lines_by_image={}` →
  `out_dir/classes.txt == "scratch\nstain\n"`,`written==0`,不 crash。

**衍生/不變量測試(PM 自加,設計未明列)**
- **AC17(來源零寫入)**:對某來源資料集 `src` 做完整 `prelabel_records→to_yolo_lines→export_prelabels`
  (out_dir 在 tmp),`src` 內**檔案集合與各檔 mtime/大小不變**(C6 真實不變量,非僅靠 assert)。
- **AC18(accept 冪等)**:同輸入呼叫兩次 `prelabel_records`,decisions **逐欄相等**(純函式)。

## 6. 給下游(PM/PG)的備註

- `assert_safe_prelabel_dir` 與既有 `assert_safe_bank_dir` **不可混用**(後者見 `labels/` 會誤擋重複匯出)。
- GUI 傳 `class_names=head["classes"]`(標籤空間以模型為準);classes.txt 同源寫出供訓練工具對照。
- 效能非本模組 AC 重點(決策是 per-object 輕量);真實行為驗收在 GUI E2E(見 `M11_gui_wiring.md`)。
