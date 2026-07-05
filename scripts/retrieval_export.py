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


def export_retrieval(records, decisions, out_dir, *, class_names, source_dirs=()) -> dict:
    """匯出 YOLO labels(accept/relabel 的 final_class,沿用 record bbox)+ classes.txt + retrieval_report.csv
    到 out_dir(C6 安全)。回 {written, objects, csv_rows, out_dir}。"""
    import prelabel
    from anomaly_bank_store import _atomic_text
    pl = [{"item": int(d["item"]), "pred": str(d["final_class"]), "accept": True}
          for d in decisions
          if d.get("decision") in ("accepted", "relabeled") and d.get("final_class")]
    lines = prelabel.to_yolo_lines(records, pl, class_names=class_names)
    res = prelabel.export_prelabels(lines, out_dir, class_names=class_names, source_dirs=source_dirs)
    _atomic_text(Path(out_dir) / "retrieval_report.csv", retrieval_report_csv(records, decisions))
    return {"written": res["written"], "objects": res["objects"],
            "csv_rows": len(records), "out_dir": str(out_dir)}
