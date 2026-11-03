"""Portable, selected-region evidence lineage shared by review and training handoffs."""
from __future__ import annotations
import copy
import hashlib
import json


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _pick(record, keys):
    return {key: copy.deepcopy(record[key]) for key in keys if key in record}


def region_provenance(case, rows):
    """Portable evidence closure for selected regions, never unrelated case assets."""
    evidence = {e["id"]: e for e in case["evidence"]}
    regions, pending = [], []
    for row in rows:
        raw_id = row["id"].split(":", 1)[1]
        refs = {row["evidence_id"]} if row.get("evidence_id") else set()
        if row["id"].startswith("item:"):
            refs.update(e["id"] for e in evidence.values()
                        if e["role"] in {"attribution_region", "model_attribution"}
                        and e["details"].get("item_id") == raw_id
                        and (e["role"] == "model_attribution" or e["details"].get("roi") == row["roi"]))
        pending.extend(refs)
        regions.append({"region_id": row["id"], "source_sha256": row["source"]["sha256"],
                        "source": _pick(row["source"], ("id", "name", "sha256", "size", "kind", "original_kind")),
                        "roi": copy.deepcopy(row["roi"]), "review": copy.deepcopy(row["review"]),
                        "evidence_ids": sorted(refs), "association_semantics": "historical_selection"})
    selected, search_ids, attribution_ids = {}, set(), set()
    detail_keys = ("item_id", "query_item_id", "query_sha256", "query_roi", "roi", "search_id",
                   "attribution_evidence_id", "attribution_id", "profile_id", "profile_revision",
                   "target_identity", "sha256", "category", "source_kind", "label_status")
    while pending:
        ref = pending.pop()
        if ref in selected:
            continue
        if ref not in evidence:
            raise ValueError("交接來源證據不存在，請重新核對候選與案件")
        e = evidence[ref]
        details = e["details"]
        selected[ref] = {**_pick(e, ("id", "role", "at")), "details": _pick(details, detail_keys)}
        if details.get("attribution_evidence_id"):
            pending.append(details["attribution_evidence_id"])
        if details.get("search_id"):
            search_ids.add(details["search_id"])
        if details.get("attribution_id"):
            attribution_ids.add(details["attribution_id"])
        # Region correction retains its prior provenance, but does not copy old reviews forward.
        before = details.get("before", {})
        if before.get("evidence_id"):
            pending.append(before["evidence_id"])
            selected[ref]["details"]["previous_evidence_id"] = before["evidence_id"]
    searches = []
    for search in case.get("searches", []):
        if search["id"] in search_ids:
            record = _pick(search, ("id", "query_item_id", "query_sha256", "query_roi", "candidate_scope", "attribution_evidence_id"))
            record["recipe_sha256"] = digest(json.dumps(search.get("recipe", {}), sort_keys=True, ensure_ascii=False).encode())
            searches.append(record)
    attributions = []
    for cam in case.get("attributions", []):
        if cam["id"] not in attribution_ids:
            continue
        record = _pick(cam, ("id", "evidence_id", "compute_id", "item_id", "profile_id", "profile_revision",
                             "target_identity", "method", "family", "overlay_sha256"))
        record["target"] = _pick(cam.get("target", {}), ("model_sha256", "input_sha256", "output_index", "class_index", "class_name", "box", "scalar", "schema_version"))
        record["correspondence_status"] = cam.get("correspondence", {}).get("status", "unavailable")
        record["historical_parity_status"] = cam.get("historical_parity", {}).get("status", "unavailable")
        original = next((i for i in case["items"] if i["id"] == cam.get("item_id")), None)
        from fp_attribution import target_identity
        record["target_current"] = bool(original and not original.get("withdrawn") and cam.get("target_identity")
                                        and target_identity(case, original) == cam["target_identity"])
        attributions.append(record)
    return {"schema_version": 1, "regions": regions, "evidence": sorted(selected.values(), key=lambda e: e["id"]),
            "searches": searches, "attributions": attributions,
            "meaning": "Review/training handoff selection provenance only; verdicts and training labels require explicit human review, not CAM or similarity."}

