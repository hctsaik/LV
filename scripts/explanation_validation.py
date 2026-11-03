"""Conservative sample accounting and paired, quality-aware effort observations."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import re


ROLES = ("normal", "true_defect", "false_positive")


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _cohort_contrasts(rows):
    # First deduplicate native targets, then average within image and declared
    # group. Repeating one target must not increase its weight in either mean.
    targets = defaultdict(set)
    for row in rows:
        if not row.get("target_key"):
            continue
        for contrast in row.get("contrasts", []):
            value, fraction = contrast.get("excess_drop"), contrast.get("fraction")
            if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in (value, fraction)):
                continue
            if not 0 < fraction < 1 or contrast.get("fill_method") not in ("constant", "blur"):
                continue
            key = (row["role"], fraction, contrast["fill_method"], row["group_id"], row["source_sha256"], row["target_key"])
            targets[key].add(float(value))
    images = defaultdict(list)
    ambiguous = 0
    for key, values in targets.items():
        if len(values) != 1:
            ambiguous += 1
            continue
        images[key[:-1]].append(next(iter(values)))
    groups = defaultdict(list)
    for key, values in images.items():
        groups[key[:-1]].append(sum(values) / len(values))
    outcomes = defaultdict(list)
    for key, values in groups.items():
        outcomes[key[:-1]].append({"group_id": key[-1], "mean_excess_drop": sum(values) / len(values), "unique_images": len(values)})
    return {"rows": [{"role": key[0], "fraction": key[1], "fill_method": key[2],
                       "unique_images": sum(k[:3] == key for k in images), "unique_groups": len(values),
                       "mean_group_excess_drop": sum(v["mean_excess_drop"] for v in values) / len(values),
                       "group_means": values} for key, values in sorted(outcomes.items())],
            "ambiguous_target_contrasts": ambiguous,
            "scope": "Descriptive fixed-score contrasts, equally weighted by declared group after target deduplication and image averaging; no causal or population-level conclusion."}


def cohort_summary(records):
    """Input rows: source_sha256, group_id, role, reviewer, reason, verified,
    identity {model_sha256,runtime_id,head_path,method,protocol}, evidence_id,
    status. Group IDs are operator declarations, not inferred independence.
    """
    records = list(records)
    if len(records) > 200:
        raise ValueError("At most 200 cohort targets")
    measured = [r for r in records if r.get("status") == "measured" and _sha(r.get("source_sha256"))]
    metadata = defaultdict(set)
    for r in measured:
        metadata[r["source_sha256"]].add((r.get("group_id", ""), r.get("role", "")))
    conflicts = sorted(sha for sha, values in metadata.items() if len(values) > 1)
    buckets = {}
    excluded = []
    for r in measured:
        identity = r.get("identity") or {}
        missing = [k for k in ("runtime_id", "head_path", "method", "protocol") if not identity.get(k)]
        reason = None
        if r["source_sha256"] in conflicts:
            reason = "conflicting image group or role"
        elif r.get("verified") is not True or r.get("role") not in ROLES or not all(str(r.get(k) or "").strip() for k in ("group_id", "reviewer", "reason", "evidence_id")):
            reason = "unverified cohort metadata"
        elif missing or not _sha(identity.get("model_sha256")):
            reason = "incomplete execution identity"
        if reason:
            excluded.append({"evidence_id": r.get("evidence_id"), "source_sha256": r["source_sha256"], "reason": reason})
            continue
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(canonical.encode()).hexdigest()
        bucket = buckets.setdefault(key, {"identity": identity, "records": []})
        bucket["records"].append(r)
    partitions = []
    for key, bucket in sorted(buckets.items()):
        rows = bucket["records"]
        by_role = {role: {"unique_images": len({r["source_sha256"] for r in rows if r["role"] == role}),
                          "unique_groups": len({r["group_id"] for r in rows if r["role"] == role})} for role in ROLES}
        partitions.append({"identity_key": key, "identity": bucket["identity"], "measured_targets": len(rows),
                           "unique_images": len({r["source_sha256"] for r in rows}),
                           "unique_groups": len({r["group_id"] for r in rows}), "roles": by_role,
                           "contrasts": _cohort_contrasts(rows),
                           "members": [{k: r.get(k) for k in ("source_sha256", "group_id", "role", "target_key", "evidence_id")} for r in rows],
                           "evidence_ids": sorted({r["evidence_id"] for r in rows})})
    eligible = [r for b in buckets.values() for r in b["records"]]
    return {"schema_version": 1, "status": "observed" if eligible else "inconclusive",
            "requested_targets": len(records), "measured_targets": len(measured),
            "unique_images": len({r["source_sha256"] for r in measured}),
            "verified_images": len({r["source_sha256"] for r in eligible}),
            "unique_groups": len({r["group_id"] for r in eligible}), "partitions": partitions,
            "conflicting_images": conflicts, "excluded": excluded,
            "unmeasured_targets": len(records) - len(measured),
            "evidence_ids": sorted({r["evidence_id"] for r in eligible}),
            "scope": "Groups are human-declared; repeated boxes/runs are not independent images. Different execution protocols are not pooled. No population efficacy claim."}


def compare_effort(records):
    """Compare only uniquely matched, completed measurements with independent review."""
    grouped, excluded = defaultdict(list), []
    required = ("study_id", "pair_id", "case_group", "reviewer", "quality_reviewer")
    numeric = ("active_minutes", "waiting_minutes", "setup_minutes", "quality_errors", "unsupported_claims", "repeated_inputs", "wrong_target_rework", "evidence_returns")
    for index, record in enumerate(records):
        why = None
        if record.get("withdrawn"):
            why = "measurement withdrawn with audit history retained"
        elif record.get("schema_version") != 2 or any(not str(record.get(k) or "").strip() for k in required):
            why = "missing study, pairing or reviewer metadata"
        elif record.get("condition") not in ("原有流程", "LV 查案流程") or record.get("completed") is not True or record.get("quality_reviewed") is not True:
            why = "incomplete workflow or quality review"
        elif str(record["quality_reviewer"]).strip().casefold() == str(record["reviewer"]).strip().casefold():
            why = "quality review is not independent"
        elif isinstance(record.get("order"), bool) or record.get("order") not in (1, 2):
            why = "missing workflow order"
        else:
            for key in numeric:
                value = record.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    why = "invalid or missing measurement"
                    break
                if key not in ("active_minutes", "waiting_minutes", "setup_minutes") and value != int(value):
                    why = "error counts must be integers"
                    break
        if why:
            excluded.append({"index": index, "reason": why})
            continue
        key = tuple(str(record[k]).strip() for k in ("study_id", "pair_id", "case_group", "reviewer"))
        grouped[key].append((index, record))
    pairs = []
    for key, entries in grouped.items():
        baseline = [r for _, r in entries if r["condition"] == "原有流程"]
        lv = [r for _, r in entries if r["condition"] == "LV 查案流程"]
        if len(baseline) != 1 or len(lv) != 1 or baseline[0]["order"] == lv[0]["order"]:
            excluded.extend({"index": i, "reason": "unmatched, duplicate or ambiguous workflow order"} for i, _ in entries)
            continue
        a, b = baseline[0], lv[0]
        pairs.append({"study_id": key[0], "pair_id": key[1], "case_group": key[2], "reviewer": key[3],
                      "lv_order": b["order"], "active_minutes_difference": a["active_minutes"] - b["active_minutes"],
                      "waiting_minutes_difference": a["waiting_minutes"] - b["waiting_minutes"],
                      "baseline": {k: a[k] for k in numeric}, "lv": {k: b[k] for k in numeric},
                      "quality_not_worse": all(b[k] <= a[k] for k in ("quality_errors", "unsupported_claims", "wrong_target_rework", "evidence_returns"))})
    return {"schema_version": 1, "status": "observed" if pairs else "insufficient_data", "pairs": pairs,
            "matched_pairs": len(pairs), "excluded": excluded,
            "mean_active_minutes_difference": sum(r["active_minutes_difference"] for r in pairs) / len(pairs) if pairs else None,
            "quality_not_worse_pairs": sum(r["quality_not_worse"] for r in pairs),
            "order_balanced": bool(pairs) and {r["lv_order"] for r in pairs} == {1, 2},
            "scope": "Observed matched-case time differences only. Setup, active time, waiting, quality and order are separate; no unmeasured effort-savings claim."}
