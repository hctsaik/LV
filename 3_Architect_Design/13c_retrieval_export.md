# 設計:13c `retrieval_export`(以樣搜樣確認 → YOLO 預標 + CSV)

> 對應 PRD:[2_PO_PRD/fewshot_search_prd.md](../2_PO_PRD/fewshot_search_prd.md)(M13 Task4)。Tier A(薄層)。
> **復用** `prelabel.to_yolo_lines/export_prelabels/assert_safe_prelabel_dir`(C6);只多加 CSV 報表。

## 1. 目的
把海掃佇列 + 使用者決定,匯成兩份到**另選資料夾**:(A)YOLO labels(沿用**粗框幾何**、類別=確認後),
(B)retrieval_report.csv(誰被撈到、建議什麼、我怎麼決定)。**絕不寫來源**。

## 2. 資料結構
- `records`:海掃 topk_records —— 含 `image_path, obj_index, bbox(cx,cy,w,h), score(粗框 conf,可 None),
  suggested_class, similarity`。
- `decisions`:對齊的決定清單,每項 `{"item": i(索引 records), "decision": "accepted"|"relabeled"|"skipped"|"pending", "final_class": str|None}`。
  - accepted:採納 suggested_class(final_class=suggested_class);relabeled:改成 final_class;
    skipped/pending:不進 YOLO(仍進 CSV)。

## 3. I/O 契約(`scripts/retrieval_export.py`)
```python
def retrieval_report_csv(records, decisions) -> str:
    """回 CSV 字串。表頭(釘死順序):
    image_path,obj_index,cx,cy,w,h,proposal_conf,suggested_class,similarity,decision,final_class
    每列一個 record(依 records 順序);proposal_conf 為 None → 空;final_class skip/pending → 空。"""

def export_retrieval(records, decisions, out_dir, *, class_names, source_dirs=()) -> dict:
    """匯出:out_dir/labels/*.txt(accept/relabel 的 final_class,沿用 record bbox 幾何)+
    out_dir/classes.txt + out_dir/retrieval_report.csv(所有 record)。C6 安全(assert_safe_prelabel_dir)。
    回 {"written": labels 檔數, "objects": YOLO 行數, "csv_rows": len(records), "out_dir": str}。"""
```

## 4. 資料流 / 邊界
- YOLO:decision∈{accepted,relabeled} → prelabel decisions `{item, pred=final_class, accept=True}`
  → `prelabel.to_yolo_lines(records, ., class_names)` → `prelabel.export_prelabels(., out_dir, class_names, source_dirs)`。
- CSV:所有 records 一列;用 `csv` 模組正確跳脫(路徑可能含逗號)。原子寫(`anomaly_bank_store._atomic_text`)。
- final_class 不在 class_names → 由 prelabel.to_yolo_lines raise(沿用既有契約)。
- out_dir 位於來源內 / 含 images/ → raise(prelabel C6)。

## 5. Acceptance Criteria(釘死;給 `/pm`)
> records 用手造 dict(不需真掃描)。class_names=["cat","dog"]。
- **AC-CE1(CSV 表頭與值)**:2 records(img_a obj0 bbox(0.5,0.5,0.4,0.6) conf0.9 suggested "cat" sim0.8;
  img_b obj0 conf None suggested "dog" sim0.3)+ decisions[accepted, skipped] →
  CSV 首行==表頭;第 2 行 decision==accepted、final_class==cat、proposal_conf==0.9;
  第 3 行 decision==skipped、final_class 空、proposal_conf 空。
- **AC-CE2(YOLO 只含 accept/relabel + 沿用粗框幾何)**:上例 → labels 只 img_a(cat=cls0)、行=
  `0 0.500000 0.500000 0.400000 0.600000`;img_b(skipped)無檔。
- **AC-CE3(relabel)**:decision relabeled、final_class "dog" → YOLO cls==1(dog),用 record bbox。
- **AC-CE4(export 雙出 + 讀回)**:export → out_dir/labels/img_a.txt + retrieval_report.csv + classes.txt 都在;
  回 {written:1, objects:1, csv_rows:2};讀回 CSV 行數==3(表頭+2)。
- **AC-CE5(C6)**:out_dir 在 source_dirs 內 → raise;來源零寫入。
- **AC-CE6(final_class 不在 class_names)**:relabel 成 "bird" 但 class_names 無 → ValueError(含 bird)。
