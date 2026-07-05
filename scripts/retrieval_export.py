"""13c retrieval_export:以樣搜樣確認 → YOLO 預標(沿用粗框幾何、類別=確認後)+ retrieval_report.csv。

設計:3_Architect_Design/13c_retrieval_export.md(M13)。復用 prelabel.to_yolo_lines/export_prelabels
(C6 守門);只多加 CSV 報表。decisions 每項 {item, decision, final_class}。
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

_CSV_HEADER = ["image_path", "obj_index", "cx", "cy", "w", "h", "proposal_conf",
               "suggested_class", "similarity", "decision", "final_class"]


def retrieval_report_csv(records, decisions) -> str:
    """回 CSV 字串(表頭固定順序;每列一個 record)。proposal_conf None → 空;skip/pending final_class → 空。"""
    dec_by_item = {int(d["item"]): d for d in decisions}
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(_CSV_HEADER)
    for i, r in enumerate(records):
        d = dec_by_item.get(i, {"decision": "pending", "final_class": None})
        cx, cy, bw, bh = r["bbox"]
        conf = r.get("score")
        w.writerow([r.get("image_path", ""), r.get("obj_index", ""), cx, cy, bw, bh,
                    "" if conf is None else conf, r.get("suggested_class", ""),
                    r.get("similarity", ""), d.get("decision", "pending"),
                    d.get("final_class") or ""])
    return buf.getvalue()


def export_retrieval(records, decisions, out_dir, *, class_names, source_dirs=(),
                     copy_images=False) -> dict:
    """匯出 YOLO labels(accept/relabel 的 final_class,沿用 record bbox)+ classes.txt + retrieval_report.csv
    到 out_dir(C6 安全)。回 {written, objects, csv_rows, out_dir, images_copied}。
    copy_images=True(M14c):另把每個有標註物件的來源影像複製到 out_dir/images/(成 standalone YOLO 資料集;
    只讀來源、只寫 out_dir;允許 out_dir 已含 images/ 以支援重複匯出,但來源關係檢查仍守)。"""
    import prelabel
    from anomaly_bank_store import _atomic_text
    accepted = [int(d["item"]) for d in decisions
                if d.get("decision") in ("accepted", "relabeled") and d.get("final_class")]
    pl = [{"item": i, "pred": str(next(d["final_class"] for d in decisions if int(d["item"]) == i)),
           "accept": True} for i in accepted]
    lines = prelabel.to_yolo_lines(records, pl, class_names=class_names)
    res = prelabel.export_prelabels(lines, out_dir, class_names=class_names,
                                    source_dirs=source_dirs, allow_images=copy_images)
    _atomic_text(Path(out_dir) / "retrieval_report.csv", retrieval_report_csv(records, decisions))
    images_copied = 0
    if copy_images:                       # 複製有標註物件的來源影像 → out_dir/images/(去重、只讀來源)
        import shutil
        img_dir = Path(out_dir) / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        seen = set()
        for i in accepted:
            src = records[i].get("image_path")
            if not src or src in seen:
                continue
            seen.add(src)
            sp = Path(src)
            if sp.exists():
                shutil.copy2(sp, img_dir / sp.name)
                images_copied += 1
    return {"written": res["written"], "objects": res["objects"],
            "csv_rows": len(records), "out_dir": str(out_dir), "images_copied": images_copied}
