"""al_service:離線監看選樣服務(CLI run-once 編排)——設計 3_Architect_Design/10_al_service.md。

run-once + 外部排程,不做常駐 daemon(憲法 C7)。只編排:持久化委派 09 al_workspace、選樣委派 08 al_batch、
不自寫檔、不自算分數。每輪:load_profile → 鎖 → 驗模型 → 增量掃描 → al_batch 全量評分(.lv_cache 讓只有新圖真 embed)
→ 合併人工標註(移出已標/deleted)→ 物化 Top-K 佇列 → 寫摘要 → 放鎖。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _run_id(now: float) -> str:
    """由給定 now(float)產生確定性 run_id;不取「當前時間」(可注入、測試確定)。"""
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime(float(now)))


def init_workspace(workspace_dir, *, name, watch_folders, model_dir, **over) -> dict:
    import os

    import numpy as np

    from al_workspace import assert_safe_workspace, default_profile, save_profile
    assert_safe_workspace(workspace_dir, watch_folders)
    ref = over.pop("reference_vector", None)     # similar:存成 reference.npy,profile 只留檔名(可攜)
    prof = default_profile(name=name, watch_folders=watch_folders,
                           model_dir=model_dir, **over)
    if ref is not None:
        wsd = Path(workspace_dir)
        wsd.mkdir(parents=True, exist_ok=True)
        tmp = wsd / "_tmp_reference.npy"
        np.save(tmp, np.asarray(ref, dtype=np.float32).ravel())
        os.replace(tmp, wsd / "reference.npy")   # atomic
        prof["reference_vector_file"] = "reference.npy"
    save_profile(workspace_dir, prof)
    return prof


def run_once(workspace_dir, *, embed_fn=None, extractor=None,
             max_batches=None, now=None, progress=None) -> dict:
    import al_batch
    import al_workspace as ws
    from object_eval import classes_for

    now = time.time() if now is None else float(now)
    run_id = _run_id(now)
    wsd = Path(workspace_dir)

    # 1. 設定(在取鎖前;缺/壞 → error,不建鎖/不寫檔)
    try:
        profile = ws.load_profile(wsd)
    except Exception as e:
        return {"status": "error", "reason": f"設定讀取失敗:{e}", "run_id": run_id}
    watch_folders = [Path(f) for f in profile["watch_folders"]]

    # 1b. 安全目錄(工作區不可落來源內)
    try:
        ws.assert_safe_workspace(wsd, watch_folders)
    except Exception as e:
        return {"status": "error", "reason": f"工作區不安全:{e}", "run_id": run_id}

    # 2. 鎖(單實例;C7)
    pid = os.getpid()
    if not ws.acquire_lock(wsd, pid=pid, started=now):
        return {"status": "skipped", "reason": "另一實例執行中", "run_id": run_id}

    try:
        # 3. 驗凍結模型 + 取 object_source(從磁碟)
        fm = al_batch.load_frozen_model(profile["model_dir"])
        object_source = fm["meta"].get("object_source", "yolo")

        # 4. 增量掃描
        state = ws.load_state(wsd)
        scan = ws.scan_incremental(watch_folders, state)
        all_paths = [Path(k) for k in scan["state"]["files"]]   # 當前全部來源影像

        # 5. 選樣(委派 08;當前全部影像,.lv_cache 讓只有新圖真 embed)
        labels = ws.read_labels(wsd)
        class_names = None
        for r in watch_folders:
            class_names = class_names or classes_for(r)
        ref_vec = None
        ref_vectors = ref_labels = None
        if profile["objective"] == "similar":       # similar:載入可攜參考向量傳給引擎
            import numpy as np
            rvf = profile.get("reference_vector_file")
            if not rvf or not (wsd / rvf).exists():
                raise ValueError("objective 'similar' 需要參考向量,但 workspace 缺 reference 檔")
            ref_vec = np.load(wsd / rvf)
        elif profile["objective"] == "retrieve":    # retrieve(M13 以樣搜樣):載入樣本集傳給引擎
            import sample_bank
            sbd = profile.get("sample_bank_dir")
            if not sbd:
                raise ValueError("objective 'retrieve' 需要 sample_bank_dir(樣本集目錄)")
            _bank = sample_bank.load_sample_bank(sbd)
            ref_vectors, ref_labels = _bank["vectors"], list(_bank["labels"])
        result = al_batch.run_batched(
            all_paths, model_dir=profile["model_dir"],
            checkpoint_dir=wsd / "al_batch_ck",
            objective=profile["objective"], k=int(profile["k"]) + len(labels),
            batch_size=int(profile["batch_size"]), object_source=object_source,
            class_names=class_names, dataset_dirs=watch_folders,
            embed_fn=embed_fn, extractor=extractor, progress=progress,
            resume=True, on_identity_mismatch="restart", max_batches=max_batches,
            ref_vector=ref_vec, ref_vectors=ref_vectors, ref_labels=ref_labels)

        # 6. 正規化 al_batch item_id → 佇列契約的 id(設計 §2.4),再合併標註 → 取前 k 個未標的
        records = [{"id": r["item_id"],
                    **{k: v for k, v in r.items() if k != "item_id"}}
                   for r in (result.get("topk_records") or [])]
        pending = ws.merge_labels_into_pending(
            records, labels, deleted_ids=scan["deleted"])
        pending = pending[:int(profile["k"])]

        # 7. 蓋章物化佇列
        mv = result.get("model_version")
        queued = [{**rec, "model_version": mv, "run_id": run_id,
                   "status": "pending", "created_at": now} for rec in pending]
        ws.write_queue(wsd, queued)

        # 8. 落狀態 + 摘要
        ws.save_state(wsd, scan["state"])
        summary = {"run_id": run_id, "at": now,
                   "new": len(scan["new"]), "changed": len(scan["changed"]),
                   "deleted": len(scan["deleted"]),
                   "objects_scored": int(result.get("objects_scored") or 0),
                   "queue_len": len(queued), "model_version": mv,
                   "objective": profile["objective"], "done": bool(result.get("done"))}
        sp = ws.write_run_summary(wsd, summary, ts=run_id)

        return {"status": "ok", "reason": "", "new": len(scan["new"]),
                "changed": len(scan["changed"]), "deleted": len(scan["deleted"]),
                "objects_scored": summary["objects_scored"], "queue_len": len(queued),
                "model_version": mv, "run_id": run_id, "summary_path": str(sp)}
    except Exception as e:
        return {"status": "error", "reason": str(e), "run_id": run_id}
    finally:
        ws.release_lock(wsd, pid=pid)


def status(workspace_dir) -> dict:
    from al_workspace import read_queue
    wsd = Path(workspace_dir)
    runs = sorted((wsd / "runs").glob("*.json")) if (wsd / "runs").exists() else []
    last = None
    if runs:
        try:
            last = json.loads(runs[-1].read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            last = None
    return {"last_run": last, "queue_len": len(read_queue(wsd)),
            "locked": (wsd / "lock").exists()}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="al_service")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run"); pr.add_argument("--profile", required=True)
    pst = sub.add_parser("status"); pst.add_argument("--profile", required=True)
    pin = sub.add_parser("init")
    pin.add_argument("--profile", required=True)
    pin.add_argument("--model-dir", required=True)
    pin.add_argument("--watch", action="append", required=True)
    pin.add_argument("--k", type=int, default=100)
    pin.add_argument("--objective", default="novelty")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "run":
            res = run_once(args.profile)
            print(json.dumps(res, ensure_ascii=False))
            return 0 if res.get("status") in ("ok", "skipped") else 1
        if args.cmd == "status":
            print(json.dumps(status(args.profile), ensure_ascii=False))
            return 0
        if args.cmd == "init":
            init_workspace(args.profile, name=Path(args.profile).name,
                           watch_folders=args.watch, model_dir=args.model_dir,
                           k=args.k, objective=args.objective)
            print(f"initialized workspace: {args.profile}")
            return 0
    except Exception as e:
        print(f"error: {e}")
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
