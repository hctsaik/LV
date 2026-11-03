"""Persistent, review-only false-positive investigations; never writes training labels."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

VERDICTS = {"pending": "待覆核", "normal": "確認此區域正常", "defect": "確認有瑕疵",
            "disputed": "規格有爭議", "insufficient": "資訊不足"}
KINDS = {"fact": "確認事實", "hypothesis": "待驗證假設", "action": "下一步"}
FULL_ROI = [0.0, 0.0, 1.0, 1.0]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def roi_checked(roi):
    if len(roi) != 4:
        raise ValueError("區域需要 x1、y1、x2、y2 四個座標")
    r = [float(x) for x in roi]
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in r) or r[0] >= r[2] or r[1] >= r[3]:
        raise ValueError("區域座標須介於 0 與 1，且右下角須大於左上角")
    return r


def region_key(sha, roi):
    return digest(_json([sha, roi_checked(roi)]).encode())[:24]


def active_rows(rows):
    return [r for r in rows if not r.get("withdrawn")]


def revise_region(case, item_id, *, roi=None, label=None, withdraw=False, reason="", candidate=False):
    """Correct a region without rewriting old evidence or applying old reviews to new geometry."""
    if not str(reason).strip():
        raise ValueError("請填寫修正／撤回原因")
    rows = case["candidates" if candidate else "items"]
    item = next((r for r in rows if r["id"] == item_id), None)
    if item is None:
        raise ValueError("找不到區域")
    replacement = roi_checked(roi) if roi is not None else item["roi"]
    new_label = str(label) if label is not None else item.get("label", "")
    if candidate and replacement != item["roi"]:
        raise ValueError("對照候選的範圍屬於原搜尋證據；請撤回後從新的搜尋結果加入")
    for other in active_rows(rows):
        if other["id"] != item_id and other.get("source_id",other.get("source",{}).get("id")) == item.get("source_id",item.get("source",{}).get("id")) and other["roi"] == replacement and other.get("label","") == new_label and not withdraw:
            raise ValueError("相同區域與類別已存在")
    before = copy.deepcopy(item)
    changed = replacement != item["roi"] or new_label != item.get("label", "")
    case.setdefault("region_history", []).append({"item": before, "candidate": candidate,
                                                  "at": utc_now(), "reason": str(reason).strip()})
    item.update(roi=replacement, label=new_label, withdrawn=bool(withdraw))
    if changed or bool(before.get("withdrawn")) != bool(withdraw):
        item["review"] = _review("pending", "", "")
        item.pop("external_target", None)
    affected = {e["id"] for e in case["evidence"] if e["details"].get("item_id") == item_id
                or e["details"].get("id") == item_id or e["details"].get("query_item_id") == item_id}
    for claim in case["claims"]:
        if affected.intersection(claim["refs"]):
            claim["needs_review"] = True
            claim["review_reason"] = "引用區域已修正／撤回，請重新核對"
    item["evidence_id"] = _evidence(case, "region_revision", "區域修正／撤回", item_id=item_id,
                                    before=before, after=copy.deepcopy(item), reason=str(reason).strip())
    return item


def _evidence(case, role, text, **details):
    row = {"id": "e_" + uuid.uuid4().hex[:12], "role": role, "text": text,
           "at": utc_now(), "details": details}
    case["evidence"].append(row)
    return row["id"]


def _review(verdict, reviewer, reason):
    if not isinstance(verdict, str) or verdict not in VERDICTS:
        raise ValueError("未知的覆核結果")
    if not isinstance(reviewer, str) or not isinstance(reason, str):
        raise ValueError("覆核者與判斷依據須為文字")
    if verdict != "pending" and (not reviewer.strip() or not reason.strip()):
        raise ValueError("覆核需要填寫覆核者與判斷依據")
    return {"verdict": verdict, "reviewer": reviewer.strip(), "reason": reason.strip(), "at": utc_now()}


class CaseStore:
    def __init__(self, root=None):
        # Optional UI-session hooks. Plain worker/CLI stores retain database-only locking.
        self.on_load = None
        self.before_save = None
        self.after_save = None
        default = Path(os.environ.get("LV_OUTPUT_DIR") or Path(__file__).resolve().parents[1] / "output") / "fp_cases"
        self.root = Path(root or os.environ.get("LV_CASES_DIR") or default).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "cases.sqlite3"
        with self._connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, revision INTEGER, body TEXT NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS history (id TEXT, revision INTEGER, body TEXT NOT NULL, PRIMARY KEY(id,revision))")

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.db, timeout=15)
        try:
            with con:
                yield con
        finally:
            con.close()

    def ingest(self, data: bytes, name: str, *, kind="original", source_path=""):
        from PIL import Image
        if kind not in {"original", "screenshot", "reference"}:
            raise ValueError("未知影像來源")
        if not data or len(data) > 50 * 1024 * 1024:
            raise ValueError("圖片必須介於 1 byte 與 50 MB")
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            size = list(image.size)
            fmt = image.format
        ext = {"JPEG": ".jpg", "PNG": ".png", "BMP": ".bmp", "TIFF": ".tif", "WEBP": ".webp"}.get(fmt)
        if ext is None:
            raise ValueError("支援 PNG、JPEG、BMP、TIFF、WebP 圖片")
        sha = digest(data)
        assets = self.root / "assets"
        assets.mkdir(exist_ok=True)
        path = assets / (sha + ext)
        if not path.exists():
            tmp = assets / (uuid.uuid4().hex + ".tmp")
            try:
                tmp.write_bytes(data)
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        return {"id": "s_" + uuid.uuid4().hex[:12], "sha256": sha,
                "asset": path.name, "name": Path(name).name, "kind": kind,
                "source_path": str(source_path), "size": size, "added_at": utc_now()}

    def restore_source(self, source, data):
        """Repair the original bytes without changing source identity or prior evidence."""
        name = source["asset"]
        if Path(name).name != name or digest(data) != source["sha256"]:
            raise ValueError("補回圖片必須與案件原始檔案完全一致；檔案指紋不符")
        from PIL import Image
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if list(image.size) != source["size"]:
                raise ValueError("圖片尺寸與案件來源不符")
        assets = self.root / "assets"
        assets.mkdir(exist_ok=True)
        temporary = assets / (uuid.uuid4().hex + ".tmp")
        try:
            temporary.write_bytes(data)
            os.replace(temporary, assets / name)
        finally:
            temporary.unlink(missing_ok=True)
        return self.image_path(source)

    def image_path(self, source):
        name = source["asset"]
        if Path(name).name != name:
            raise ValueError("無效的案件影像位置")
        path = self.root / "assets" / name
        if digest(path.read_bytes()) != source["sha256"]:
            raise ValueError("案件影像已變更，請重新補入來源")
        return path

    def create(self, source, title="", description="", model_id="", threshold="", raw_output=""):
        case = {"schema_version": 1, "id": "case_" + uuid.uuid4().hex[:16], "revision": 0,
                "created_at": utc_now(), "updated_at": utc_now(), "title": title.strip() or source["name"],
                "description": description, "sources": [source], "items": [], "candidates": [],
                "external": {"model_id": model_id.strip(), "threshold": threshold, "raw_output": raw_output,
                             "replay_status": "not_replayed"},
                "evidence": [], "claims": [], "batches": [], "searches": [],
                "reply_status": "draft", "cause_status": "unknown", "improvement_status": "not_evaluated",
                "effort_minutes": 0.0, "effort_log": []}
        _evidence(case, "reported_output", "報案敘述與外部模型輸出（尚未重播）",
                  source_id=source["id"], description=description, **case["external"])
        return self.save(case)

    def save(self, case):
        if self.before_save is not None:
            self.before_save(case)
        value = copy.deepcopy(case)
        old = value["revision"]
        value["revision"] = old + 1
        value["updated_at"] = utc_now()
        body = _json(value)
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            if old == 0:
                con.execute("INSERT INTO cases VALUES (?,?,?)", (value["id"], 1, body))
            else:
                cur = con.execute("UPDATE cases SET revision=?,body=? WHERE id=? AND revision=?",
                                  (old + 1, body, value["id"], old))
                if cur.rowcount != 1:
                    raise ValueError("案件已在另一個視窗更新，請重新載入後再儲存")
            con.execute("INSERT INTO history VALUES (?,?,?)", (value["id"], old + 1, body))
        if self.after_save is not None:
            self.after_save(value)
        return value

    def load(self, case_id):
        with self._connect() as con:
            row = con.execute("SELECT body FROM cases WHERE id=?", (case_id,)).fetchone()
        if row is None:
            raise ValueError("找不到案件")
        value = json.loads(row[0])
        return self.on_load(value) if self.on_load is not None else value

    def list(self):
        with self._connect() as con:
            rows = con.execute("SELECT body FROM cases ORDER BY rowid DESC").fetchall()
        return [json.loads(row[0]) for row in rows]

    def duplicates(self, sha):
        return [c for c in self.list() if any(s["sha256"] == sha for s in c["sources"])]


def add_source(case, source):
    if any(s["sha256"] == source["sha256"] and s["kind"] == source["kind"] for s in case["sources"]):
        raise ValueError("此影像來源已在案件中")
    case["sources"].append(source)
    _evidence(case, "source", "補入影像來源", **source)


def update_context(case, description, model_id, threshold, raw_output):
    case["description"] = description
    case["external"].update(model_id=model_id.strip(), threshold=threshold, raw_output=raw_output)
    _evidence(case, "reported_output", "更新報案資訊（尚未重播）", description=description, **case["external"])


def add_item(case, source_id, roi, label="", score=None, prediction_id="", origin="manual"):
    source = next((s for s in case["sources"] if s["id"] == source_id), None)
    if source is None:
        raise ValueError("找不到指定影像")
    roi = roi_checked(roi)
    if score is not None and not math.isfinite(float(score)):
        raise ValueError("分數必須為有限數值")
    if any(i["source_id"] == source_id and i["roi"] == roi and i["label"] == label for i in active_rows(case["items"])):
        raise ValueError("同一來源、區域與類別已存在")
    item = {"id": "i_" + uuid.uuid4().hex[:12], "source_id": source_id, "roi": roi,
            "label": str(label), "score": None if score is None else float(score),
            "prediction_id": str(prediction_id), "origin": origin, "review": _review("pending", "", "")}
    item["evidence_id"] = _evidence(case, "reported_region", "爭議區域／預測紀錄（未覆核）", **copy.deepcopy(item))
    case["items"].append(item)
    return item


def import_yolo(case, source_id, text):
    """Strict six-column cxcywh/conf input; validate all rows before mutation."""
    staged = copy.deepcopy(case)
    n = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        cells = line.split()
        if len(cells) != 6:
            raise ValueError("YOLO 預測每行須為 class cx cy width height confidence 六欄")
        cls, cx, cy, w, h, score = map(float, cells)
        if not all(math.isfinite(x) for x in (cls, cx, cy, w, h, score)) or cls < 0 or not cls.is_integer() or not 0 <= score <= 1:
            raise ValueError("YOLO 類別／分數格式不正確")
        add_item(staged, source_id, [cx-w/2, cy-h/2, cx+w/2, cy+h/2], str(int(cls)), score,
                 prediction_id=str(n), origin="yolo_prediction_file")
        n += 1
    if not n:
        raise ValueError("預測檔沒有有效列")
    case.clear()
    case.update(staged)
    return n


def review_item(case, item_id, verdict, reviewer, reason, *, candidate=False):
    collection = case["candidates" if candidate else "items"]
    item = next((i for i in collection if i["id"] == item_id), None)
    if item is None:
        raise ValueError("找不到覆核項目")
    if item.get("withdrawn"):
        raise ValueError("此區域已撤回，請先恢復再覆核")
    item["review"] = _review(verdict, reviewer, reason)
    _evidence(case, "human_review", "人工逐區域覆核", item_id=item_id,
              candidate=candidate, roi=item["roi"], **item["review"])


def add_claim(case, kind, text, refs, claim_id=None):
    if kind not in KINDS or not text.strip():
        raise ValueError("請填寫說明內容")
    refs = list(dict.fromkeys(refs))
    known = {e["id"] for e in case["evidence"]}
    if not refs or not set(refs) <= known:
        raise ValueError("請引用本案件中至少一筆有效證據")
    claim = {"id": claim_id or "c_" + uuid.uuid4().hex[:12], "kind": kind,
             "text": text.strip(), "refs": refs, "at": utc_now(), "authorship": "engineer"}
    if claim_id:
        index = next((j for j, c in enumerate(case["claims"]) if c["id"] == claim_id), None)
        if index is None:
            raise ValueError("找不到原說明")
        case["claims"][index] = claim
    else:
        case["claims"].append(claim)
    case["cause_status"] = "hypothesis" if any(c["kind"] == "hypothesis" for c in case["claims"]) else "unknown"


def record_effort(case, minutes, *, mode="add", note=""):
    minutes = float(minutes)
    if not math.isfinite(minutes) or minutes < 0 or mode not in {"add", "set"}:
        raise ValueError("工時必須為非負分鐘數")
    case["effort_minutes"] = round(minutes + (case["effort_minutes"] if mode == "add" else 0), 3)
    case["effort_log"].append({"at": utc_now(), "minutes": minutes, "mode": mode, "note": note})


def add_candidate(case, result, store):
    source = store.ingest(Path(result["path"]).read_bytes(), result.get("name") or Path(result["path"]).name,
                          kind="reference", source_path=result.get("source_path", result["path"]))
    source["original_kind"] = result.get("source_kind", "reference")
    if source["sha256"] != result["sha256"]:
        raise ValueError("對照影像已變更，請重新搜尋")
    roi = roi_checked(result["roi"])
    key = region_key(source["sha256"], roi)
    if any(c["id"] == key for c in case["candidates"]):
        return False
    candidate = {"id": key, "source": source, "roi": roi,
                 "reason": result["reason"], "reference_category": result["category"],
                 "evidence_id": result["evidence_id"], "review": _review("pending", "", "")}
    if candidate["evidence_id"] not in {e["id"] for e in case["evidence"]}:
        raise ValueError("對照證據不属于此案件")
    case["candidates"].append(candidate)
    return True


def review_counts(rows):
    rows = active_rows(rows)
    return {"total_regions": len(rows), "reviewed_regions": sum(r["review"]["verdict"] != "pending" for r in rows),
            "pending_regions": sum(r["review"]["verdict"] == "pending" for r in rows),
            "normal_regions": sum(r["review"]["verdict"] == "normal" for r in rows),
            "defect_regions": sum(r["review"]["verdict"] == "defect" for r in rows), "training_ready_images": 0}


def prepare_batch(case):
    if not active_rows(case["candidates"]):
        raise ValueError("先加入待覆核候選")
    identities = sorted(c["id"] for c in active_rows(case["candidates"]))
    batch_id = "fpb_" + digest(_json([case["id"], identities]).encode())[:20]
    if not any(b["id"] == batch_id for b in case["batches"]):
        case["batches"].append({"id": batch_id, "candidate_ids": identities, "round_id": None,
                                "at": utc_now(), "case_revision": case["revision"]})
    return batch_id


def link_round(case, batch_id):
    """Recover by stable batch key after interrupted save; no label metrics/readback."""
    import round_ledger as ledger
    batch = next(b for b in case["batches"] if b["id"] == batch_id)
    if batch["round_id"]:
        return batch["round_id"]
    old = next((r for r in ledger.load_rounds() if r.get("params", {}).get("fp_batch_id") == batch_id), None)
    round_id = old["round_id"] if old else ledger.start_round(
        "case_fp_review", {"case_id": case["id"], "fp_batch_id": batch_id,
                           "review_unit": "region", "n_regions": len(batch["candidate_ids"]),
                           "training_export": False}, "誤逮案件", "僅送覆核；區域覆核不計入已標註影像")
    # Legacy n_sent is image count; region count stays in explicit manifest/params.
    if not old or not any(b["batch_id"] == batch_id for b in old.get("batches", [])):
        rows = [c for c in case["candidates"] if c["id"] in batch["candidate_ids"]]
        ledger.attach_batch(round_id, batch_id, len({c["source"]["sha256"] for c in rows}))
    batch["round_id"] = round_id
    return round_id


def review_manifest(case, batch_id):
    batch = next(b for b in case["batches"] if b["id"] == batch_id)
    rows = [copy.deepcopy(c) for c in case["candidates"] if c["id"] in batch["candidate_ids"]]
    return {"schema_version": 1, "purpose": "review_only", "case_id": case["id"], "batch_id": batch_id,
            "round_id": batch["round_id"], "case_revision": case["revision"], "unit": "region",
            "training_export": False, "counts": review_counts(rows), "items": rows,
            "external_model": case["external"], "instructions": "逐區域覆核；未覆核不是正常，不可直接轉成訓練標籤。"}


def review_provenance(case, batch_id):
    """Immutable export snapshot; evidence explains selection, not review truth."""
    from fp_evidence_lineage import region_provenance
    manifest = review_manifest(case, batch_id)
    rows = [{**row, "id": "candidate:"+row["id"]} for row in manifest["items"]]
    lineage = region_provenance(case, rows)
    return {**lineage, "purpose": "review_only", "case_id": case["id"],
            "case_revision": case["revision"], "batch_id": batch_id,
            "round_id": manifest["round_id"], "exported_at": utc_now()}


def apply_review_return(case, manifest):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("回傳檔須為版本 1 的覆核清單物件")
    if manifest.get("purpose") != "review_only" or manifest.get("case_id") != case["id"]:
        raise ValueError("回傳檔不屬於此案件或不是覆核清單")
    batch = next((b for b in case["batches"] if b["id"] == manifest.get("batch_id")), None)
    if batch is None or manifest.get("round_id") != batch["round_id"]:
        raise ValueError("回傳批次／回合不相符")
    if manifest.get("case_revision") != case["revision"]:
        raise ValueError("回傳檔版本已過期，請重新下載清單並合併覆核結果")
    updates = manifest.get("items")
    if not isinstance(updates, list) or not updates:
        raise ValueError("回傳檔沒有覆核項目")
    staged = copy.deepcopy(case)
    known = {c["id"]: c for c in staged["candidates"]}
    seen = set()
    for row in updates:
        if not isinstance(row, dict) or not isinstance(row.get("source"), dict) or not isinstance(row.get("review"), dict):
            raise ValueError("回傳項目缺少來源或覆核欄位")
        key = row.get("id")
        if not isinstance(key, str) or key in seen or key not in batch["candidate_ids"]:
            raise ValueError("回傳含重複或不屬於批次的區域")
        seen.add(key)
        original = known[key]
        if original.get("withdrawn"):
            raise ValueError("回傳區域已撤回，請重新建立覆核批次")
        if row.get("source", {}).get("sha256") != original["source"]["sha256"] or row.get("roi") != original["roi"]:
            raise ValueError("回傳影像／區域身分不相符")
        r = row.get("review", {})
        checked = _review(r.get("verdict"), r.get("reviewer", ""), r.get("reason", ""))
        if checked["verdict"] == "pending":
            continue
        review_item(staged, key, checked["verdict"], checked["reviewer"], checked["reason"], candidate=True)
    case.clear()
    case.update(staged)


def report_markdown(case):
    # Plain Markdown is downloadable; escape source text used in headings/table-like lines.
    def plain(value):
        return str(value).replace("<", "&lt;").replace(">", "&gt;")
    external_note = (f"外部模型已有 {len(case.get('external_runs', []))} 次本機推論；是否重現歷史判定須逐筆核對。" if case.get("external_runs") else "外部模型尚未重播。")
    evaluation_note = "請見各次固定資料集評估，結論限定模型版本與資料集" if case.get("evaluations") else case["improvement_status"]
    lines = [f"# 誤逮案件：{plain(case['title'])}", "", f"案件：{case['id']}／版本 {case['revision']}",
             f"報案記錄模型：{plain(case['external']['model_id'] or '未知')}；門檻：{plain(case['external']['threshold'] or '未知')}",
             f"回覆：{case['reply_status']}；原因：{case['cause_status']}；改善：{evaluation_note}",
             f"記錄調查時間：{case['effort_minutes']:.1f} 分鐘", "",
             external_note + "LV 對照是輔助線索，不能證明外部模型採用何種特徵；人工撰寫的說明仍需覆核。", "",
             "## 報案", plain(case["description"]), "", "## 影像來源"]
    for s in case["sources"]:
        lines.append(f"- {plain(s['name'])} ({s['kind']}) sha256={s['sha256']}")
    if any(s["kind"] == "screenshot" for s in case["sources"]):
        lines.append("截圖含縮放、文字或框線，不能視為當時正式輸入。")
    lines.extend(["", "## 逐區域覆核"])
    for i in case["items"]:
        r = i["review"]
        lines.append(f"- {i['id']}／來源 {i['source_id']}／ROI {i['roi']}／類別 {plain(i['label'])}／分數 {i['score']}："
                     f"{VERDICTS[r['verdict']]}；{plain(r['reviewer'])}；{plain(r['reason'])}")
    for kind, label in KINDS.items():
        lines.extend(["", f"## {label}"])
        rows = [c for c in case["claims"] if c["kind"] == kind]
        lines.extend([f"- {plain(c['text'])} 【引用：{', '.join(c['refs'])}】" for c in rows] or ["尚未記錄。"])
    if case.get("external_runs"):
        from fp_investigation import current_run
        lines.extend(["", "## 外部模型推論與歷史重播"])
        for run in case["external_runs"]:
            lines.append(f"- {run['id']}／模型 {plain(run['contract']['model_version'])}／契約 {run['contract_id']}／"
                         f"歷史比對 {run['parity']['status']}／目前脈絡 {'有效' if current_run(case, run) else '已過期'}："
                         f"{plain('；'.join(run['parity']['reasons']))} 【{run['evidence_id']}】")
    if case.get("experiments"):
        lines.extend(["", "## 外部模型對照實驗"])
        for experiment in case["experiments"]:
            lines.append(f"- {experiment['id']}／{experiment['mode']}／固定類別 {plain(experiment['target_class'])}："
                         f"{plain(experiment['limitation'])} 【{experiment['evidence_id']}】")
    if case.get("evaluations"):
        lines.extend(["", "## 固定資料集改善評估"])
        for result in case["evaluations"]:
            lines.append(f"- {result['id']}／{result['manifest']['purpose']}／資料 {result['dataset_fingerprint']}／"
                         f"驗收 {result['gate']['status']}：{plain('；'.join(result['gate']['reasons']))} 【{result['evidence_id']}】")
            lines.append("    " + plain(_json({"baseline": result["baseline"], "candidate": result["candidate"], "gate": result["gate"]})))
    if case.get("training_exports"):
        lines.extend(["", "## 訓練資料交接"])
        for result in case["training_exports"]:
            lines.append(f"- {result['id']}：{result['n_images']} 張影像／{result['n_boxes']} 個真瑕疵框／"
                         f"{result['n_negative_images']} 張整圖正常負樣本；確認者 {plain(result['reviewer'])} 【{result['evidence_id']}】")
    lines.extend(["", "## 證據紀錄（保留歷史，請核對時間與區域）"])
    for e in case["evidence"]:
        lines.extend([f"- {e['id']} [{e['role']}] {e['at']}：{plain(e['text'])}",
                      "    " + plain(_json(e["details"]))])
    lines.extend(["", "## 待覆核批次", _json(review_counts(case["candidates"])),
                  "此處只計覆核區域；訓練交接與模型改善須分別參照對應證據。"])
    for b in case["batches"]:
        lines.append(f"- {b['id']} → round {b['round_id'] or '尚未連結'}")
    return "\n".join(lines) + "\n"
