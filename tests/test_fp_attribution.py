"""Case bridge tests use actual ONNX and pinned YOLO execution, no engine stubs."""
import copy
from pathlib import Path

import pytest

from fp_attribution import (infer_task, gradcam_task, attach_result, select_detection,
                            target_identity, _match_target, promote_trial)
from fp_cases import CaseStore, revise_region
from fp_external_runtime import DEFAULT_CONFIG
from fp_onnx_demo import make_demo_model
from model_profiles import ModelRegistry
from test_yolo_attribution import picture


class Context:
    def __init__(self, root, name):
        self.workspace_root = Path(root)
        self.artifact_dir = Path(root)/"test-jobs"/name
        self.artifact_dir.mkdir(parents=True)
        self.progress_events = []

    def progress(self, value, message):
        self.progress_events.append((value,message))

    def check_cancelled(self):
        return None


def resolved(value, context):
    if isinstance(value,dict):
        return {k:resolved(v,context) for k,v in value.items()}
    if isinstance(value,list):
        return [resolved(v,context) for v in value]
    if isinstance(value,str) and value.startswith("@artifact/"):
        return str(context.artifact_dir/value[len("@artifact/"):])
    return value


@pytest.fixture
def onnx_case(tmp_path):
    store = CaseStore(tmp_path/"workspace")
    source = store.ingest(picture(),"report.png")
    case = store.create(source,"source first")
    path = make_demo_model(tmp_path/"model.onnx",detection=True)
    registry = ModelRegistry(store.root)
    asset = registry.ingest(path.read_bytes(),path.name)
    config = {**DEFAULT_CONFIG,"input_size":[32,32],"resize":"stretch","class_names":["scratch"],"output_format":"xyxy","threshold":.1}
    profile = registry.create("Station A",deployment_asset=asset,runtime_config=config,metadata={"machine_rules":"unknown PLC"})
    return store,case,source,profile


def execute_infer(store,source,profile,name="infer"):
    context=Context(store.root,name)
    result=infer_task({"source":source,"profile_id":profile["id"],"profile_revision":profile["revision"]},context)
    return resolved(result,context),context


def test_source_first_infer_attach_selection_and_idempotence(onnx_case):
    store,case,source,profile=onnx_case
    assert case["items"] == []
    result,context=execute_infer(store,source,profile)
    assert result["run"]["prediction"]["detections"]
    assert result["run"]["parity"]["status"] == "unavailable"
    assert result["run"]["decision_scope"]["machine_decision"] == "unverified"
    assert Path(result["run"]["artifact_dir"],"tensors.npz").exists()
    attached=attach_result(store,case["id"],result)
    assert attached["items"] == []
    again=attach_result(store,case["id"],result)
    assert again["revision"] == attached["revision"]
    run=attached["external_runs"][0]
    item=select_detection(attached,run,run["prediction"]["detections"][0])
    assert select_detection(attached,run,run["prediction"]["detections"][0])["id"] == item["id"]
    assert len(attached["items"]) == 1
    changed=copy.deepcopy(run["prediction"]["detections"][0]); changed["score"] += .01
    with pytest.raises(ValueError,match="不屬於"):
        select_detection(attached,run,changed)
    assert context.progress_events[-1][0] == .95


def test_trial_promotes_without_reinference_or_new_source(onnx_case):
    store,_,source,profile=onnx_case
    result,_=execute_infer(store,source,profile)
    case=promote_trial(store,result,title="From trial")
    assert case["sources"][0]["sha256"] == source["sha256"]
    assert case["external_runs"][0]["id"] == result["run"]["id"]


def test_valid_zero_detections_do_not_require_fabricated_roi(onnx_case):
    store,case,source,profile=onnx_case
    registry=ModelRegistry(store.root)
    config={**profile["runtime_config"],"threshold":1.0}
    profile=registry.revise(profile["id"],runtime_config=config)
    result,_=execute_infer(store,source,profile)
    assert result["run"]["prediction"]["detections"] == []
    saved=attach_result(store,case["id"],result)
    assert saved["items"] == [] and saved["external_runs"]
    assert saved["cause_status"] == "unknown"


def test_onnx_only_case_rejects_gradcam_with_actionable_reason(onnx_case):
    store,case,source,profile=onnx_case
    result,_=execute_infer(store,source,profile)
    case=attach_result(store,case["id"],result)
    run=case["external_runs"][0]
    item=select_detection(case,run,run["prediction"]["detections"][0])
    case=store.save(case)
    with pytest.raises(ValueError,match="PyTorch"):
        gradcam_task({"case_id":case["id"],"item_id":item["id"],"target_identity":target_identity(case,item)},Context(store.root,"unavailable"))


def test_cross_backend_matching_uses_geometry_not_numeric_index():
    a={"class":"scratch","score":.8,"box":[.1,.1,.3,.3],"output_index":9}
    b={"class":"scratch","score":.7,"box":[.6,.6,.8,.8],"output_index":2}
    deployment={"contract_id":"deployment","prediction":{"task":"detection","detections":[a,b]}}
    target={**a,"output_index":88}
    gradient={"contract_id":"gradient","prediction":{"task":"detection","detections":[{**b,"output_index":42},target]}}
    mapped,record=_match_target(deployment,gradient,a)
    assert mapped["output_index"] == 88
    assert record["status"] == "per_case_matched"
    assert "history" not in record
    ambiguous=copy.deepcopy(gradient)
    ambiguous["prediction"]["detections"].append({**target,"output_index":99})
    with pytest.raises(ValueError,match="唯一對應"):
        _match_target(deployment,ambiguous,a)


def test_native_match_requires_real_input_and_output_identities():
    row={"class":"scratch","score":.8,"box":[.1,.1,.3,.3],"output_index":9}
    run={"contract_id":"same","prediction":{"task":"detection","detections":[row,{**row,"output_index":10}]}}
    with pytest.raises(ValueError):
        _match_target(run,copy.deepcopy(run),row)
    complete={**run,"input_sha256":"a"*64,"output_sha256":"b"*64}
    mapped,record=_match_target(complete,copy.deepcopy(complete),row)
    assert mapped["output_index"] == 9 and record["status"] == "native_matched"
    changed={**complete,"output_sha256":"c"*64}
    with pytest.raises(ValueError):
        _match_target(complete,changed,row)


def test_changed_target_rejects_attach_preserves_result(onnx_case):
    store,case,source,profile=onnx_case
    result,_=execute_infer(store,source,profile)
    case=attach_result(store,case["id"],result)
    run=case["external_runs"][0]
    item=select_detection(case,run,run["prediction"]["detections"][0])
    old=target_identity(case,item)
    case=store.save(case)
    artifact=store.root/"saved-cam"; artifact.mkdir()
    (artifact/"maps.npz").write_bytes(b"completed artifact remains")
    pending={"source":source,"attribution":{"id":"cam_test","item_id":item["id"],"target_identity":old,"artifact_dir":str(artifact)}}
    revise_region(case,item["id"],roi=[.1,.1,.8,.8],reason="correct box")
    store.save(case)
    assert target_identity(case,case["items"][0]) != old
    with pytest.raises(ValueError,match="區域已改變"):
        attach_result(store,case["id"],pending)
    assert (artifact/"maps.npz").exists()


def test_source_corruption_and_wrong_case_are_rejected(onnx_case):
    store,case,source,profile=onnx_case
    result,_=execute_infer(store,source,profile)
    wrong=copy.deepcopy(result); wrong["source"]["id"]="unrelated"
    with pytest.raises(ValueError,match="不屬於"):
        attach_result(store,case["id"],wrong)
    store.image_path(source).write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="影像已變更"):
        execute_infer(store,source,profile,"corrupt")


@pytest.fixture(scope="module")
def v8_file(tmp_path_factory):
    import torch
    from ultralytics.nn.tasks import DetectionModel
    torch.set_num_threads(1); torch.manual_seed(34)
    model=DetectionModel("yolov8n.yaml",ch=3,nc=2,verbose=False).eval()
    model.names={0:"scratch",1:"dent"}
    path=tmp_path_factory.mktemp("bridge-v8")/"model.pt"
    torch.save({"model":model},path)
    return path


def test_actual_yolo_case_gradcam_bridge_and_stale_guard(tmp_path,v8_file):
    from yolo_attribution import inspect_model
    store=CaseStore(tmp_path/"workspace")
    source=store.ingest(picture(),"original.png")
    case=store.create(source,"YOLO real bridge")
    registry=ModelRegistry(store.root)
    asset=registry.ingest(v8_file.read_bytes(),"original.pt")
    info=inspect_model(v8_file,trusted=True)
    config={**info["default_config"],"input_size":[64,64],"threshold":0.0,"max_detections":10}
    profile=registry.create("YOLOv8 station",deployment_asset=asset,runtime_config=config,metadata={"family":"yolov8"})
    inferred,_=execute_infer(store,source,profile)
    case=attach_result(store,case["id"],inferred)
    run=case["external_runs"][0]
    item=select_detection(case,run,run["prediction"]["detections"][0])
    case=store.save(case)
    payload={"case_id":case["id"],"item_id":item["id"],"target_identity":target_identity(case,item),"method":"signed","fractions":[]}
    context=Context(store.root,"cam")
    result=resolved(gradcam_task(payload,context),context)
    attr=result["attribution"]
    assert attr["target"]["output_index"] == item["external_target"]["output_index"]
    assert attr["status"] == "ok"
    assert attr["spec"]["profile_revision"] == str(profile["revision"])
    assert attr["spec"]["case_context"]["case_id"] == case["id"]
    assert Path(attr["artifact_dir"],"maps.npz").exists()
    assert attr["historical_parity"]["status"] == "unavailable"
    saved=attach_result(store,case["id"],result)
    assert saved["attributions"][0]["evidence_id"]
    assert CaseStore(store.root).load(case["id"])["attributions"][0]["compute_id"] == attr["compute_id"]
    revise_region(saved,item["id"],withdraw=True,reason="wrong target")
    store.save(saved)
    with pytest.raises(ValueError,match="已改變"):
        gradcam_task(payload,Context(store.root,"stale"))
