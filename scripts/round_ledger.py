"""回合帳本(設計 3_Architect_Design/09_round_ledger.md)。

主動學習迴圈的持久記憶:每一輪的選樣策略、送標批次、回讀結果、(可掛的)模型指標,
以 append-only JSONL 記錄,round 狀態由事件折疊而成。是停止判準與學習曲線的資料底座。

**依賴契約:pure stdlib**(不 import numpy/streamlit;AC7 在乾淨子行程驗)。
env(LV_OUTPUT_DIR)於 call-time 讀。
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def ledger_path() -> Path:
    """<LV_OUTPUT_DIR 或 <repo>/output>/rounds.jsonl(與 usage_log/curation_log 同目錄慣例)。"""
    root = Path(os.environ.get("LV_OUTPUT_DIR") or (_REPO / "output"))
    return root / "rounds.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(event: dict) -> None:
    p = ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    row = dict(event)
    row.setdefault("ts", _utc_now())
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def start_round(strategy: str, params: dict, tool: str, note: str = "") -> str:
    """開一個回合。round_id = "r"+UTC%Y%m%d%H%M%S+"-"+uuid4().hex[:6]
    (uuid 尾提供足夠熵:同秒大量連呼不撞)。回 round_id。"""
    ts = _utc_now()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    round_id = f"r{stamp}-{uuid.uuid4().hex[:6]}"
    _append({"kind": "start", "round_id": round_id, "ts": ts,
             "strategy": strategy, "params": params or {}, "tool": tool, "note": note})
    return round_id


def attach_batch(round_id: str, batch_id: str, n_sent: int) -> None:
    _append({"kind": "batch", "round_id": round_id,
             "batch_id": batch_id, "n_sent": int(n_sent)})


def record_readback(round_id: str, summary: dict) -> None:
    row = {"kind": "readback", "round_id": round_id}
    row.update(summary)
    _append(row)


def attach_metric(round_id: str, name: str, value: float,
                  source: str, note: str = "") -> None:
    _append({"kind": "metric", "round_id": round_id, "name": name,
             "value": value, "source": source, "note": note})


def _iter_valid_events():
    """逐行 yield 合法事件(dict 有 kind 與 round_id);壞行/缺鍵不 yield。"""
    p = ledger_path()
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
        if not isinstance(row, dict) or not row.get("kind") or not row.get("round_id"):
            continue
        yield row


def load_rounds() -> list[dict]:
    """折疊成**乾淨的** round 物件 list(依 start ts 排序;空/檔不存在 → [])。
    掛在不存在 round_id 上的事件不進 list(由 orphan_events() 另取)。"""
    rounds: dict[str, dict] = {}
    pending: list[dict] = []  # 非 start 事件,先收著等 start 建好
    for row in _iter_valid_events():
        rid = row["round_id"]
        if row["kind"] == "start":
            rounds.setdefault(rid, {
                "round_id": rid, "ts": row.get("ts", ""),
                "strategy": row.get("strategy", ""), "params": row.get("params", {}),
                "tool": row.get("tool", ""), "note": row.get("note", ""),
                "batches": [], "n_sent": 0,
                "readbacks": [], "n_labeled": 0, "n_changed": 0,
                "metrics": [],
            })
        else:
            pending.append(row)
    for row in pending:
        r = rounds.get(row["round_id"])
        if r is None:
            continue  # 孤兒 → 由 orphan_events() 提供
        kind = row["kind"]
        if kind == "batch":
            r["batches"].append({"batch_id": row.get("batch_id", ""),
                                 "n_sent": int(row.get("n_sent", 0) or 0)})
            r["n_sent"] += int(row.get("n_sent", 0) or 0)
        elif kind == "readback":
            summary = {k: v for k, v in row.items()
                       if k not in ("kind", "round_id", "ts")}
            r["readbacks"].append(summary)
            r["n_labeled"] += int(row.get("n_labeled", 0) or 0)
            r["n_changed"] += int(row.get("n_changed", 0) or 0)
        elif kind == "metric":
            r["metrics"].append({"name": row.get("name", ""),
                                 "value": row.get("value"),
                                 "source": row.get("source", ""),
                                 "ts": row.get("ts", "")})
    return sorted(rounds.values(), key=lambda r: r["ts"])


def orphan_events() -> list[dict]:
    """掛在不存在 round_id 上的事件(batch/readback/metric);正常情況為 []。"""
    starts = {row["round_id"] for row in _iter_valid_events()
              if row["kind"] == "start"}
    return [row for row in _iter_valid_events()
            if row["kind"] != "start" and row["round_id"] not in starts]


def learning_curve(rounds: list[dict], metric_name: str) -> tuple[list[int], list[float]]:
    """x = 依回合序的**累積 n_labeled**(含無此 metric 的回合);y = 該回合此 metric
    的**最後一筆** value;無此 metric 的回合不產生點(但仍計入累積)。空輸入 → ([], [])。"""
    xs: list[int] = []
    ys: list[float] = []
    cum = 0
    for r in rounds:
        cum += int(r.get("n_labeled", 0) or 0)
        vals = [m["value"] for m in r.get("metrics", []) if m.get("name") == metric_name]
        if vals:
            xs.append(cum)
            ys.append(vals[-1])  # last-wins
    return xs, ys
