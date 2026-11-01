"""Cited engineer-written case claims and path-safe report snapshots."""
from __future__ import annotations

import copy
import json
import re
import uuid

from fp_case_store import public_case, utc_now


CLAIM_KINDS = {
    "fact": "確認事實",
    "hypothesis": "待驗證假設",
    "action": "下一步",
}


def add_claim_evidence(case: dict, claim: dict, *, event: str) -> str:
    evidence_id = "e_" + uuid.uuid4().hex[:12]
    case.setdefault("evidence", []).append({
        "id": evidence_id,
        "role": "engineer_claim",
        "text": f"工程師說明（{CLAIM_KINDS[claim['kind']]}；{event}）",
        "at": utc_now(),
        "claim_id": claim["id"],
        "kind": claim["kind"],
        "claim_text": claim["text"],
        "refs": list(claim["refs"]),
        "authorship": "engineer",
    })
    return evidence_id


def upsert_claim(case: dict, kind: str, text: str, evidence_ids: list[str], claim_id: str = "") -> dict:
    if kind not in CLAIM_KINDS:
        raise ValueError("說明類型必須是確認事實、待驗證假設或下一步")
    normalized_text = str(text or "").strip()
    if not normalized_text:
        raise ValueError("請填寫說明內容")
    if len(normalized_text) > 10000:
        raise ValueError("說明內容不可超過 10,000 個字元")
    refs = list(dict.fromkeys(str(value) for value in evidence_ids if str(value)))
    known = {str(row.get("id")) for row in case.get("evidence", [])}
    if not refs or not set(refs) <= known:
        raise ValueError("請引用本案件中至少一筆有效證據")

    claims = case.setdefault("claims", [])
    previous = None
    if claim_id:
        previous = next((row for row in claims if row.get("id") == claim_id), None)
        if previous is None:
            raise ValueError("找不到要編輯的說明")
    at = utc_now()
    claim = {
        "id": claim_id or "c_" + uuid.uuid4().hex[:12],
        "kind": kind,
        "text": normalized_text,
        "refs": refs,
        "at": previous.get("at", at) if previous else at,
        "updated_at": at,
        "authorship": "engineer",
    }
    if previous:
        index = claims.index(previous)
        claims[index] = claim
    else:
        claims.append(claim)
    claim["evidence_id"] = add_claim_evidence(case, claim, event="編輯" if previous else "新增")
    case["cause_status"] = "hypothesis" if any(row["kind"] == "hypothesis" for row in claims) else "unknown"
    return claim


def set_reply_status(case: dict, status: str) -> bool:
    if status not in {"draft", "replied"}:
        raise ValueError("回覆狀態必須是草稿或已回覆")
    previous = case.get("reply_status", "draft")
    if previous == status:
        return False
    case["reply_status"] = status
    case.setdefault("evidence", []).append({
        "id": "e_" + uuid.uuid4().hex[:12],
        "role": "customer_reply_status",
        "text": "案件回覆狀態更新",
        "at": utc_now(),
        "previous_status": previous,
        "status": status,
    })
    return True


def report_snapshot(case: dict) -> dict:
    snapshot = public_case(copy.deepcopy(case))
    snapshot["report_version"] = 1
    snapshot["exported_at"] = utc_now()
    snapshot["claim_kinds"] = dict(CLAIM_KINDS)
    return snapshot


def _escape_markdown(value: object) -> str:
    text = str(value or "")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([\\`*_{}\[\]()#+.!|])", r"\\\1", text)


def report_markdown(case: dict) -> str:
    """Render the current revision with explicit claims, citations and evidence."""
    lines = [
        f"# 誤逮案件：{_escape_markdown(case.get('title', ''))}",
        "",
        f"案件：`{_escape_markdown(case.get('id', ''))}`／版本 {int(case.get('revision', 0))}",
        f"回覆狀態：{case.get('reply_status', 'draft')}；根因狀態：{case.get('cause_status', 'unknown')}；",
        f"改善狀態：{case.get('improvement_status', 'not_evaluated')}；記錄調查時間：{float(case.get('effort_minutes', 0)):.1f} 分鐘。",
        "",
        "此報告由工程師整理。LV 外觀相似度是輔助線索，不代表外部模型信心或根因；歷史紀錄不會自動套用到本案。",
        "",
        "## 報案資訊",
        "",
        _escape_markdown(case.get("description", "")) or "尚無報案描述。",
        "",
        f"報案模型：{_escape_markdown(case.get('external', {}).get('model_id') or '未知')}；",
        f"門檻：{_escape_markdown(case.get('external', {}).get('threshold') or '未知')}；外部模型尚未重播。",
        "",
        "## 影像來源",
        "",
    ]
    for source in case.get("sources", []):
        lines.append(
            f"- {_escape_markdown(source.get('name', 'image'))} "
            f"({source.get('kind', 'unknown')}) SHA-256 `{source.get('sha256', '')}`"
        )
    lines.extend(["", "## 區域覆核", ""])
    verdict_labels = {
        "pending": "待覆核", "normal": "確認此區域正常", "defect": "確認有瑕疵",
        "disputed": "規格有爭議", "insufficient": "資訊不足",
    }
    for item in case.get("items", []):
        review = item.get("review", {})
        lines.append(
            f"- `{item.get('id', '')}`／來源 `{item.get('source_id', '')}`／"
            f"ROI `{json.dumps(item.get('roi', []), ensure_ascii=False)}`／"
            f"類別 {_escape_markdown(item.get('label', ''))}／"
            f"覆核 {verdict_labels.get(review.get('verdict'), '待覆核')}；"
            f"覆核者 {_escape_markdown(review.get('reviewer', ''))}；"
            f"依據 {_escape_markdown(review.get('reason', ''))}"
        )
    for kind, label in CLAIM_KINDS.items():
        lines.extend(["", f"## {label}", ""])
        rows = [claim for claim in case.get("claims", []) if claim.get("kind") == kind]
        if not rows:
            lines.append("尚未記錄。")
        for claim in rows:
            lines.append(f"說明 ID：`{claim.get('id', '')}`")
            if claim.get("needs_review"):
                lines.append("> 引用區域已變更，請重新核對此說明。")
            lines.append(_escape_markdown(claim.get("text", "")))
            lines.append("")
            lines.append("引用證據：" + ", ".join(f"`{value}`" for value in claim.get("refs", [])))
    lines.extend(["", "## 證據紀錄", ""])
    for evidence in case.get("evidence", []):
        lines.append(
            f"- `{evidence.get('id', '')}` [{_escape_markdown(evidence.get('role', ''))}] "
            f"{_escape_markdown(evidence.get('at', ''))}：{_escape_markdown(evidence.get('text', ''))}"
        )
        details = {key: value for key, value in evidence.items() if key not in {"id", "role", "at", "text"}}
        if details:
            lines.append("    " + _escape_markdown(json.dumps(details, ensure_ascii=False, sort_keys=True)))
    return "\n".join(lines).rstrip() + "\n"


def report_json(case: dict) -> str:
    return json.dumps(report_snapshot(case), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
