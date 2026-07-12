"""讀回落地層(設計 3_Architect_Design/14_readback_store.md)。

以影像 **sha256** 為鍵,把「標註讀回的修正」落成 append-only JSONL 變更日誌(不可變歷史、
可稽核)與由它折疊出的覆蓋層(sha→最新修正標籤)。存 `.lv_cache`,**絕不寫使用者資料夾**
(鐵則見記憶 no-dataset-writes)。Visualize Run 建完 records 後套用覆蓋層,讓修正跨 Run/
重啟存活——閉合主動學習迴圈的「回讀」段。

**依賴契約:pure stdlib**(不 import numpy/streamlit/PIL;AC10 在乾淨子行程驗)。
env(LV_CACHE_DIR)於 call-time 讀,不在 import 期快取路徑。
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

_REPO = Path(__file__).resolve().parent.parent


def changelog_path() -> Path:
    """<LV_CACHE_DIR 或 <repo>/.lv_cache>/readback_changelog.jsonl(跨資料集全域)。
    call-time 讀 env → 與其他快取同根,且測試可 monkeypatch。"""
    root = Path(os.environ.get("LV_CACHE_DIR") or (_REPO / ".lv_cache"))
    return root / "readback_changelog.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_corrections(changes: list[dict], *, batch_id: str = "",
                       round_id: str = "", annotator: str = "") -> int:
    """逐筆 append 修正事件;缺 sha256 或 new_label 的跳過(不製造無鍵/空修正)。
    自動補 ts;目錄不存在先建。回實際寫入筆數。"""
    rows = []
    for c in changes or []:
        sha = c.get("sha256")
        new_label = c.get("new_label")
        if not sha or new_label is None or new_label == "":
            continue
        rows.append({
            "ts": _utc_now(),
            "sha256": sha,
            "old_label": c.get("old_label", ""),
            "new_label": new_label,
            "filename": c.get("filename", ""),
            "reason": c.get("reason", ""),
            "batch_id": batch_id,
            "round_id": round_id,
            "annotator": annotator or c.get("annotator", ""),
        })
    if not rows:
        return 0
    p = changelog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _iter_valid_events():
    """逐行 yield 合法事件(dict 有 sha256 與 new_label);壞行/缺鍵不 yield。"""
    p = changelog_path()
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(row, dict) or not row.get("sha256") or row.get("new_label") in (None, ""):
            continue
        yield row


def load_overlay(shas: Iterable[str] | None = None) -> dict[str, str]:
    """折疊變更日誌 → {sha: 最新 new_label}(依 append 順序,後到覆蓋先到=最新勝)。
    空/檔不存在 → {};壞行跳過;給 shas 只回命中的。"""
    overlay: dict[str, str] = {}
    for row in _iter_valid_events():
        overlay[row["sha256"]] = row["new_label"]
    if shas is not None:
        want = set(shas)
        overlay = {s: lbl for s, lbl in overlay.items() if s in want}
    return overlay


def apply_overlay(records: list[dict], sha_of: Callable[[dict], str | None]) -> int:
    """對每筆 record:sha=sha_of(rec);命中覆蓋層且覆蓋值 != 現 label 時就地設
    record["label"]=覆蓋值。回實際覆蓋筆數。sha_of 回 None → 該筆跳過。"""
    overlay = load_overlay()
    if not overlay:
        return 0
    n = 0
    for rec in records:
        try:
            sha = sha_of(rec)
        except Exception:  # noqa: BLE001 呼叫端 sha 計算失敗 → 該筆跳過,不炸
            sha = None
        if not sha:
            continue
        new_label = overlay.get(sha)
        if new_label is not None and new_label != rec.get("label"):
            rec["label"] = new_label
            n += 1
    return n


def load_changelog() -> list[dict]:
    """全部合法事件、依 append 順序(供稽核視圖 / 匯出);壞行跳過。"""
    return list(_iter_valid_events())


def corrections_summary() -> dict:
    """{"n_events": 合法事件數, "n_images": len(overlay), "by_new_label": {label: 圖數}}。"""
    events = load_changelog()
    overlay = load_overlay()
    by_label: dict[str, int] = {}
    for sha, lbl in overlay.items():
        by_label[lbl] = by_label.get(lbl, 0) + 1
    return {"n_events": len(events), "n_images": len(overlay), "by_new_label": by_label}


def export_version(out_dir: Path, version_id: str) -> Path:
    """(Should)把目前變更日誌收成一份帶版本 id 的修正清單 CSV,寫到
    out_dir/readback_version_<version_id>.csv。純讀日誌 + 寫指定 out_dir,不碰使用者資料夾。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"readback_version_{version_id}.csv"
    fields = ["sha256", "filename", "old_label", "new_label", "annotator", "ts"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in load_changelog():
            w.writerow({k: e.get(k, "") for k in fields})
    return out
