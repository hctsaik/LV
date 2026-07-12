"""全域樣本池記帳(設計 3_Architect_Design/08_pool_registry.md)。

以影像 **sha256** 為身分鍵的 append-only JSONL 狀態機:送標/已標/排除 狀態跨工具、
跨回合、跨 app 重啟存活。送標入口在送出前查帳,預設不重複送標。

**依賴契約:pure stdlib**(不 import numpy/streamlit/PIL —— 送標端與外部宿主會在最小環境
呼叫;AC8 會在乾淨子行程驗證 numpy 未被拉入)。env 於 call-time 讀,不在 import 期快取路徑。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_REPO = Path(__file__).resolve().parent.parent

# 狀態折疊優先序(高蓋低);"reset" 特殊處理(清除)。
_PRIORITY = {"sent": 1, "excluded": 2, "labeled": 3}


def registry_path() -> Path:
    """<LV_CACHE_DIR 或 <repo>/.lv_cache>/pool_registry.jsonl(跨資料集全域)。
    call-time 讀 env → 與其他快取同根,且測試可 monkeypatch。"""
    root = Path(os.environ.get("LV_CACHE_DIR") or (_REPO / ".lv_cache"))
    return root / "pool_registry.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_events(events: Iterable[dict]) -> None:
    """append 寫入 registry;每筆自動補 ts(若缺);目錄不存在先建。"""
    events = list(events)
    if not events:
        return
    p = registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for e in events:
            row = dict(e)
            row.setdefault("ts", _utc_now())
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _iter_valid_events():
    """逐行 yield 合法事件(dict 有 sha256 與 event);壞行不 yield。"""
    p = registry_path()
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
        if not isinstance(row, dict) or not row.get("sha256") or not row.get("event"):
            continue
        yield row


def load_states(shas: Iterable[str] | None = None) -> dict[str, dict]:
    """折疊事件流 → **乾淨** {sha: {"state","batch_id","tool","ts"}}。
    鍵一律是 sha(不混診斷欄位;空 registry / 檔不存在 → {})。
    優先序 labeled>excluded>sent、同優先序後到蓋先到;"reset" 清回 unseen。"""
    # 每 sha 保留 (priority, state_dict)。reset → 移除。
    folded: dict[str, tuple[int, dict]] = {}
    for row in _iter_valid_events():
        sha = row["sha256"]
        ev = row["event"]
        if ev == "reset":
            folded.pop(sha, None)
            continue
        pri = _PRIORITY.get(ev)
        if pri is None:  # 未知 event 值 → 跳過(不崩潰、不折疊)
            continue
        cur = folded.get(sha)
        if cur is None or pri >= cur[0]:  # 高優先或同優先(後到)蓋掉;低優先不降級
            folded[sha] = (pri, {
                "state": ev,
                "batch_id": row.get("batch_id", ""),
                "tool": row.get("tool", ""),
                "ts": row.get("ts", ""),
            })
    states = {sha: sd for sha, (_pri, sd) in folded.items()}
    if shas is not None:
        want = set(shas)
        states = {s: sd for s, sd in states.items() if s in want}
    return states


def corrupt_line_count() -> int:
    """掃 registry,回無法解析 / 缺必要鍵(sha256 或 event)的行數;檔不存在回 0。"""
    p = registry_path()
    if not p.exists():
        return 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            n += 1
            continue
        if not isinstance(row, dict) or not row.get("sha256") or not row.get("event"):
            n += 1
    return n


def partition_new(shas: list[str]) -> tuple[list[str], dict[str, dict]]:
    """(從未見過的 sha 順序保留, {已有狀態的 sha: 乾淨 state dict})。
    空輸入短路(不讀檔)。"""
    if not shas:
        return [], {}
    states = load_states(shas)
    new = [s for s in shas if s not in states]
    have = {s: states[s] for s in shas if s in states}
    return new, have


# ── 便捷包裝(都走 record_events)──────────────────────────────────────────
def mark_sent(shas: list[str], batch_id: str, tool: str,
              paths: dict[str, str] | None = None) -> None:
    paths = paths or {}
    record_events([{"sha256": s, "event": "sent", "batch_id": batch_id,
                    "tool": tool, "path_hint": paths.get(s, "")} for s in shas])


def mark_labeled(shas: list[str], batch_id: str) -> None:
    record_events([{"sha256": s, "event": "labeled", "batch_id": batch_id}
                   for s in shas])


def mark_excluded(shas: list[str], tool: str, reason: str = "") -> None:
    record_events([{"sha256": s, "event": "excluded", "tool": tool,
                    "reason": reason} for s in shas])
