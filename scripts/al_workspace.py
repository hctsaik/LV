"""al_workspace:離線監看服務的工作區持久化 + 增量掃描 + 執行鎖(設計 3_Architect_Design/09_al_workspace.md)。

純持久化;無選樣邏輯(al_batch)、無編排(al_service)。承擔憲法:
- C5 單寫者:服務寫 queue/state/runs;GUI 只 append labels。兩邊永不寫同一檔。
- C6 不寫來源:workspace 開跑前過 assert_safe_workspace;掃描只讀來源。
- C8 不做全域運算(這裡本就只有 I/O)。

重用既有原子寫(anomaly_bank_store._atomic_text)+ 安全目錄(assert_safe_bank_dir);profile 用 YAML(可攜設定)。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from anomaly_bank_store import _atomic_text, assert_safe_bank_dir

_SCHEMA = 1
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
_DEFAULTS = {
    "objective": "novelty", "k": 100, "score_mode": "patch",
    "target_res": 224, "batch_size": 20,   # 小批次 → 監看/服務掃描每 20 張更新一次進度(可續跑更細)
    "reference_vector_file": None,          # similar:workspace 內參考向量 .npy 相對路徑(否則 None)
    "filters": {"min_score": None, "classes": []},
}


# ── 設定(profile;可攜、人可編)────────────────────────────────────────
def default_profile(*, name, watch_folders, model_dir, **over) -> dict:
    p = {"schema_version": _SCHEMA, "name": str(name),
         "watch_folders": [str(f) for f in watch_folders],
         "model_dir": str(model_dir), **{k: (dict(v) if isinstance(v, dict) else v)
                                         for k, v in _DEFAULTS.items()}}
    p.update(over)
    return p


def save_profile(workspace_dir, profile: dict) -> None:
    import yaml
    d = Path(workspace_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = dict(profile)
    p.setdefault("schema_version", _SCHEMA)
    _atomic_text(d / "profile.yaml",
                 yaml.safe_dump(p, allow_unicode=True, sort_keys=False))


def load_profile(workspace_dir) -> dict:
    import yaml
    f = Path(workspace_dir) / "profile.yaml"
    if not f.exists():
        raise FileNotFoundError(f"缺 profile.yaml(工作區未初始化):{f}")
    data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    sv = data.get("schema_version")
    if sv != _SCHEMA:
        raise ValueError(f"profile schema_version {sv} 不支援(本版需 {_SCHEMA})")
    for req in ("name", "watch_folders", "model_dir"):
        if not data.get(req):
            raise ValueError(f"profile 缺必填欄位:{req}")
    for k, v in _DEFAULTS.items():
        data.setdefault(k, dict(v) if isinstance(v, dict) else v)
    return data


# ── 增量掃描狀態 ────────────────────────────────────────────────────────
def _is_image(p: Path) -> bool:
    return p.suffix.lower() in _IMG_EXTS


def scan_incremental(folders, state: dict) -> dict:
    """依 (mtime,size) 雙鍵比對來源 vs state。只讀來源(C6)。
    回 {new:[Path], changed:[Path], deleted:[str], state:<更新後>}。"""
    prev = dict((state or {}).get("files", {}))
    cur: dict = {}
    new, changed = [], []
    for folder in folders:
        root = Path(folder)
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not (p.is_file() and _is_image(p)):
                continue
            st = p.stat()
            key = str(p.resolve())
            cur[key] = {"mtime": st.st_mtime, "size": st.st_size}
            old = prev.get(key)
            if old is None:
                new.append(p)
            elif old.get("mtime") != st.st_mtime or old.get("size") != st.st_size:
                changed.append(p)
    deleted = [k for k in prev if k not in cur]
    new_state = {"schema_version": _SCHEMA, "files": cur,
                 "last_run": (state or {}).get("last_run")}
    return {"new": new, "changed": changed, "deleted": deleted, "state": new_state}


def load_state(workspace_dir) -> dict:
    f = Path(workspace_dir) / "state.json"
    if not f.exists():
        return {"schema_version": _SCHEMA, "files": {}}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {"schema_version": _SCHEMA, "files": {}}   # 壞/torn → 視為全新


def save_state(workspace_dir, state: dict) -> None:
    d = Path(workspace_dir)
    d.mkdir(parents=True, exist_ok=True)
    _atomic_text(d / "state.json", json.dumps(state, ensure_ascii=False))


# ── 佇列(服務原子重寫、GUI 只讀)────────────────────────────────────────
def write_queue(workspace_dir, items: list) -> None:
    d = Path(workspace_dir)
    d.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    _atomic_text(d / "queue.jsonl", text)


def read_queue(workspace_dir) -> list:
    f = Path(workspace_dir) / "queue.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue   # 壞行跳過(不整份失敗)
    return out


# ── 標註(GUI append、服務讀+合併)──────────────────────────────────────
def append_label(workspace_dir, label: dict) -> None:
    d = Path(workspace_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "labels.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(label, ensure_ascii=False) + "\n")


def read_labels(workspace_dir) -> list:
    f = Path(workspace_dir) / "labels.jsonl"
    if not f.exists():
        return []
    seen: dict = {}
    order: list = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = rec.get("id")
        if rid not in seen:
            order.append(rid)
        seen[rid] = rec   # 後覆前
    return [seen[rid] for rid in order]


def merge_labels_into_pending(queue_items: list, labels: list, *, deleted_ids=()) -> list:
    """純函式:把已標(good/defect)或 deleted 的 item 從 pending 移出;skip 留下。不碰檔案。"""
    decided = {l.get("id") for l in labels if l.get("decision") in ("good", "defect")}
    dele = {str(x) for x in deleted_ids}
    return [it for it in queue_items
            if it.get("id") not in decided and str(it.get("id")) not in dele]


# ── 每輪摘要 ────────────────────────────────────────────────────────────
def write_run_summary(workspace_dir, summary: dict, *, ts: str) -> Path:
    d = Path(workspace_dir) / "runs"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{ts}.json"
    _atomic_text(f, json.dumps(summary, ensure_ascii=False, indent=2))
    return f


# ── 執行鎖(單一服務實例;時間過期搶佔,pid 記錄供 release-own)──────────────
def acquire_lock(workspace_dir, *, pid: int, started: float,
                 stale_after: float = 3600.0) -> bool:
    d = Path(workspace_dir)
    d.mkdir(parents=True, exist_ok=True)
    lk = d / "lock"
    if lk.exists():
        try:
            info = json.loads(lk.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            info = None
        if info is not None:
            fresh = (float(started) - float(info.get("started", 0.0))) < stale_after
            if fresh and int(info.get("pid", -1)) != int(pid):
                return False   # 新鮮且他人持有 → 第二實例不啟動
    _atomic_text(lk, json.dumps({"pid": int(pid), "started": float(started)}))
    return True


def release_lock(workspace_dir, *, pid: int) -> None:
    lk = Path(workspace_dir) / "lock"
    if not lk.exists():
        return
    try:
        info = json.loads(lk.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if int(info.get("pid", -1)) == int(pid):   # 只移本 PID 的鎖
        try:
            lk.unlink()
        except FileNotFoundError:
            pass


# ── 安全目錄(C6)──────────────────────────────────────────────────────
def assert_safe_workspace(workspace_dir, watch_folders=()) -> None:
    assert_safe_bank_dir(workspace_dir, selected_folders=[Path(f) for f in watch_folders])
