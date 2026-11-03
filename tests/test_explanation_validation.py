import copy

from explanation_validation import cohort_summary, compare_effort


def cohort(sha="a",group="lot1",role="false_positive"):
    return {"status":"measured","source_sha256":sha*64,"group_id":group,"role":role,"verified":True,
            "reviewer":"Alice","reason":"inspected source","evidence_id":"e1",
            "identity":{"model_sha256":"b"*64,"runtime_id":"modern-8.4.142","head_path":"one2one","method":"positive","protocol":{"version":2}}}


def test_duplicate_boxes_are_not_independent_and_protocols_are_partitioned():
    a=cohort();b=cohort("c","lot2","normal");c=cohort("d","lot2","true_defect")
    c["identity"]["head_path"]="one2many"
    result=cohort_summary([a,copy.deepcopy(a),b,c])
    assert result["measured_targets"]==4 and result["unique_images"]==3 and result["unique_groups"]==2
    assert len(result["partitions"])==2


def test_conflicting_or_unreviewed_metadata_cannot_support_cohort():
    a=cohort();b=cohort(role="normal");c=cohort("c");c["verified"]=False
    r=cohort_summary([a,b,c,{"status":"failed"}])
    assert r["status"]=="inconclusive" and r["verified_images"]==0 and r["unique_images"]==2
    assert r["conflicting_images"]==["a"*64] and r["unmeasured_targets"]==1


def test_group_contrasts_do_not_weight_repeated_targets_or_large_groups_more():
    a=cohort();a.update(target_key="target1",contrasts=[{"fraction":.1,"fill_method":"constant","excess_drop":.8}])
    b=copy.deepcopy(a);b["target_key"]="target2";b["contrasts"][0]["excess_drop"]=.4
    c=cohort("c","lot2");c.update(target_key="target3",contrasts=[{"fraction":.1,"fill_method":"constant","excess_drop":0.}])
    result=cohort_summary([a,a,a,b,c])["partitions"][0]["contrasts"]["rows"][0]
    assert abs(result["mean_group_excess_drop"]-.3)<1e-8
    assert result["unique_images"]==2 and result["unique_groups"]==2
    changed=copy.deepcopy(a);changed["contrasts"][0]["excess_drop"]=.1
    r=cohort_summary([a,changed])["partitions"][0]["contrasts"]
    assert r["ambiguous_target_contrasts"]==1 and not r["rows"]


def effort(condition="原有流程",minutes=20,order=1):
    return {"schema_version":2,"study_id":"study","pair_id":"pair","case_group":"case","reviewer":"A",
            "quality_reviewer":"B","completed":True,"quality_reviewed":True,"condition":condition,"order":order,
            "active_minutes":minutes,"waiting_minutes":3,"setup_minutes":10,"quality_errors":0,
            "unsupported_claims":0,"repeated_inputs":2,"wrong_target_rework":0,"evidence_returns":0}


def test_time_improvement_is_reported_with_worse_quality_not_hidden():
    a=effort();b=effort("LV 查案流程",10,2);b["quality_errors"]=1
    r=compare_effort([a,b])
    assert r["mean_active_minutes_difference"]==10 and r["quality_not_worse_pairs"]==0
    assert not r["order_balanced"]


def test_legacy_unmatched_duplicate_and_self_review_do_not_estimate_savings():
    assert compare_effort([{"condition":"原有流程","case_minutes":100}])["mean_active_minutes_difference"] is None
    a=effort();b=effort("LV 查案流程",10,2)
    assert compare_effort([a,a,b])["matched_pairs"]==0
    withdrawn=copy.deepcopy(a);withdrawn["withdrawn"]=True
    assert compare_effort([a,withdrawn,b])["matched_pairs"]==1
    b["quality_reviewer"]=" A "
    assert compare_effort([a,b])["matched_pairs"]==0
    b["quality_reviewer"]="B";b["pair_id"]="different"
    assert compare_effort([a,b])["matched_pairs"]==0
    b["pair_id"]="pair";a["order"]=True
    assert compare_effort([a,b])["matched_pairs"]==0
