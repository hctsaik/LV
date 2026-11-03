import copy

from fp_attribution import cohort_record
from explanation_validation import cohort_summary


def fixture_report():
    return {"id": "cam-one", "status": "ok", "source_sha256": "a" * 64, "method": "positive",
        "target": {"model_sha256": "b" * 64, "native_candidate_index": 5, "class_index": 0, "class_name": "scratch", "scalar": "confidence"},
        "execution": {"runtime_id": "modern-8.4.142", "head_path": "one_to_many"},
        "layer": {"path": "model.23.input[0]", "level": 0, "stride_xy": [8, 8], "shape": [1, 64, 8, 8],
                  "candidate_offset": 0, "candidate_level_counts": [64, 16, 4]},
        "contract": {"input_size": [64, 64], "resize": "letterbox", "color": "RGB", "scale": 1/255,
                     "mean": [0, 0, 0], "std": [1, 1, 1], "pad_value": 114, "model_path": "server-only"},
        "faithfulness": {"status": "measured", "protocol": {"protocol_version": 2, "fill_methods": ["constant", "blur"]},
                         "assessment": {"comparisons": []}}}


def child():
    return {"id": "job-one", "payload": {"cohort": {"group_id": "lot-a", "role": "false_positive",
        "reviewer": "engineer", "reason": "checked original", "verified": True}}}


def test_two_images_native_offsets_share_partition_but_not_target_key():
    one, two = fixture_report(), fixture_report()
    two.update(id="cam-two", source_sha256="c" * 64)
    two["target"]["native_candidate_index"] = 78
    two["layer"]["candidate_offset"] = 64
    two["layer"]["candidate_level_counts"] = [128, 32, 8]
    rows = [cohort_record(child(), report, "succeeded") for report in (one, two)]
    assert rows[0]["identity"] == rows[1]["identity"]
    assert rows[0]["target_key"] != rows[1]["target_key"]
    assert "model_path" not in rows[0]["identity"]["preprocessing"]
    result = cohort_summary(rows)
    assert result["unique_images"] == 2 and len(result["partitions"]) == 1


def test_scalar_preprocessing_and_layer_changes_split_partitions():
    original = fixture_report()
    variants = [original]
    for field in ("scalar", "preprocessing", "layer", "class"):
        report = copy.deepcopy(original)
        if field == "scalar": report["target"]["scalar"] = "class"
        elif field == "preprocessing": report["contract"]["color"] = "BGR"
        elif field == "layer": report["layer"]["level"] = 1
        else: report["target"].update(class_index=1, class_name="dent")
        variants.append(report)
    result = cohort_summary([cohort_record(child(), report, "succeeded") for report in variants])
    assert len(result["partitions"]) == 5
    assert result["unique_images"] == 1


def test_successful_job_without_measurement_does_not_increase_measured_support():
    report = fixture_report()
    report["status"] = "zero_gradient"
    assert cohort_record(child(), report, "succeeded")["status"] == "unavailable"
    report["status"] = "ok"
    report["faithfulness"]["status"] = "unavailable"
    assert cohort_record(child(), report, "succeeded")["status"] == "unavailable"
    report = fixture_report()
    report.pop("execution")
    result = cohort_summary([cohort_record(child(), report, "succeeded")])
    assert result["verified_images"] == 0
    assert result["excluded"][0]["reason"] == "incomplete execution identity"
    report = fixture_report()
    report["target"].pop("class_name")
    report["faithfulness"]["assessment"]["comparisons"] = [{"excess_drop": .5}]
    assert cohort_record(child(), report, "succeeded")["contrasts"] == []
