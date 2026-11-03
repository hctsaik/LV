"""P0 contract tests: identity, canonical payloads, and stale protection."""
from __future__ import annotations

import json

import pytest

from explain_context import (
    check_compatibility,
    late_result_is_current,
    make_explain_context,
    record_key_for,
    resolve_record,
    validate_late_result,
)
from explain_contracts import (
    CANONICAL_SIGNED_GRADCAM,
    MAX_CONTEXT_PAYLOAD_BYTES,
    ExplainContextV1,
    ExplainResultV1,
    ExplainRunSpecV1,
    RenderSpecV1,
    ContractError,
    canonical_json,
    compute_key,
    digest_checkpoint,
    digest_classifier_head,
    digest_normal_bank,
    identity_status,
    mark_legacy_identity,
    stable_record_key,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def context(**updates):
    payload = {
        "context_id": "ctx-1",
        "origin_generation": 7,
        "dataset_fingerprint": "dataset-1",
        "source_sha256": SHA_A,
        "source_dimensions": (1000, 1000),
        "model_id": "dinov2_vits14",
        "scope": "whole_image",
        "head_ref": "head-1",
        "embedding_recipe": {"target_res": 224, "pooling": "cls"},
    }
    payload.update(updates)
    return ExplainContextV1(**payload)


def test_context_round_trip_is_canonical_and_reference_only():
    value = context()
    encoded = value.to_json()
    assert encoded == canonical_json(json.loads(encoded))
    assert ExplainContextV1.from_json(encoded) == value
    assert value.payload_bytes < MAX_CONTEXT_PAYLOAD_BYTES
    assert '"model"' not in encoded
    assert '"image"' not in encoded


def test_context_rejects_binary_or_oversized_payload():
    with pytest.raises(ContractError):
        context(embedding_recipe={"bytes": b"pixels"})
    with pytest.raises(ContractError):
        context(embedding_recipe={"note": "x" * MAX_CONTEXT_PAYLOAD_BYTES})


def test_whole_image_and_object_keys_are_stable_and_geometry_sensitive():
    assert stable_record_key(SHA_A) == stable_record_key(SHA_A)
    assert stable_record_key(SHA_A) != stable_record_key(SHA_B)
    first = stable_record_key(SHA_A, object_index=1, bbox_normalized=(0.1, 0.2, 0.5, 0.8), crop_policy="letterbox")
    same = stable_record_key(SHA_A, object_index=1, bbox_normalized=(0.1, 0.2, 0.5, 0.8), crop_policy="letterbox")
    changed = stable_record_key(SHA_A, object_index=1, bbox_normalized=(0.1, 0.2, 0.6, 0.8), crop_policy="letterbox")
    assert first == same
    assert first != changed


def test_record_resolution_survives_reordering():
    records = [
        {"source_sha256": SHA_B, "label": "b"},
        {"source_sha256": SHA_A, "label": "a"},
    ]
    key = record_key_for(records[1])
    assert resolve_record(list(reversed(records)), key)["label"] == "a"


def test_content_digests_are_sensitive_to_behavior_and_do_not_use_mtime():
    assert digest_checkpoint(b"checkpoint-a") != digest_checkpoint(b"checkpoint-b")
    head = {"weight": [1, 2], "classes": ["good", "crack"], "temperature": 1.0, "l2norm": True}
    assert digest_classifier_head(head) != digest_classifier_head({**head, "temperature": 2.0})
    assert digest_normal_bank({"vectors": [[1.0, 0.0]], "model": "m"}) != digest_normal_bank({"vectors": [[0.0, 1.0]], "model": "m"})
    assert identity_status({"path": "old.pt", "size": 1, "mtime": 2}) == "legacy"
    marked = mark_legacy_identity({"path": "old.pt", "size": 1, "mtime": 2})
    assert marked["reusable"] is False


def test_compute_key_changes_for_run_inputs_but_not_render_settings():
    value = context()
    run = ExplainRunSpecV1(attribution_method=CANONICAL_SIGNED_GRADCAM, head_digest="c" * 64)
    key = compute_key(value, run)
    assert key == compute_key(value, run)
    assert key == compute_key(context(source_path="C:/different/path.png", context_id="another", created_at="later"), run)
    assert key != compute_key(value, ExplainRunSpecV1(attribution_method="experimental_token_wise_attribution", head_digest="c" * 64))
    assert key == compute_key(value, run)
    assert RenderSpecV1(opacity=0.45).to_json() != RenderSpecV1(opacity=0.90).to_json()


def test_stale_and_incompatible_precedence_and_late_generation_guard():
    old = context()
    changed = context(origin_generation=8, source_sha256=SHA_B, model_id="other")
    result = check_compatibility(old, changed)
    assert result.status == "incompatible"
    assert "model_id" in result.incompatible_fields
    assert "origin_generation" in result.stale_fields
    late = ExplainResultV1(
        context_id="ctx-1",
        requested_generation=7,
        source_sha256=SHA_A,
        run_key="run-1",
    )
    check = validate_late_result(late, changed, current_run_key="run-2")
    assert check.status == "stale"
    assert {"requested_generation", "source_sha256", "run_key"}.issubset(check.stale_fields)
    assert not late_result_is_current(late, changed, current_run_key="run-2")


def test_make_context_derives_object_identity():
    value = make_explain_context(source_sha256=SHA_A, object_index=2, bbox_normalized=(0, 0, 1, 1), crop_policy="tight")
    assert value.record_key == stable_record_key(SHA_A, object_index=2, bbox_normalized=(0, 0, 1, 1), crop_policy="tight")
