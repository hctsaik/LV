"""偵測結果對齊與比較輸入整形(M21 模組 28,Tier A)。

- `pair_runs`:兩份 {item_id: 數值} → 依 item 身分對齊成同序配對陣列;
  集合差與非有限值明講(丟棄可見,不靜默)。
- `compare_inputs`:兩份偵測 records + 人工確認 → per-item 0/1 錯誤 dict
  (錯誤定義鎖在這裡,GUI 零複製;G1 方向性由本模組+paired_compare 純函式鏈驗證)。

item_id 約定(全 M21 統一):f"{path}#{obj_index}"——物件框來自標註檔,
同資料夾跨配置穩定,可跨 run 對齊。
設計:3_Architect_Design/28_run_pairing.md(含 2026-07-19 反向閘門修訂)。
"""
from __future__ import annotations

import math
from typing import Mapping


def _as_float(key, value) -> float:
    if not isinstance(key, str):
        raise ValueError(f"item_id 必須是 str:{key!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"item {key!r} 的值不可轉 float:{value!r}") from e


def pair_runs(run_a: Mapping[str, float], run_b: Mapping[str, float]) -> dict:
    fa = {k: _as_float(k, v) for k, v in run_a.items()}
    fb = {k: _as_float(k, v) for k, v in run_b.items()}
    common = set(fa) & set(fb)
    dropped = sorted(k for k in common
                     if not (math.isfinite(fa[k]) and math.isfinite(fb[k])))
    ids = sorted(common - set(dropped))
    return {
        "ids": ids,
        "a": [fa[k] for k in ids],
        "b": [fb[k] for k in ids],
        "only_a": sorted(set(fa) - set(fb)),
        "only_b": sorted(set(fb) - set(fa)),
        "dropped": dropped,
        "n_common": len(ids),
    }


_REQUIRED_RECORD_KEYS = ("path", "obj_index", "verdict")


def _item_id(record: dict) -> str:
    return f"{record['path']}#{record['obj_index']}"


def _decisions(records: list) -> dict:
    """record → {item_id: 判異常 0/1}(verdict=="bad" 即判異常)。"""
    for r in records:
        for k in _REQUIRED_RECORD_KEYS:
            if k not in r:
                raise ValueError(f"record 缺必要鍵 {k!r}:{r!r}")
    return {_item_id(r): (1 if r["verdict"] == "bad" else 0) for r in records}


def compare_inputs(
    records_a: list, records_b: list,
    confirmed_ref: Mapping[int, str],
    records_ref: list,
) -> dict:
    dec_a = _decisions(records_a)
    dec_b = _decisions(records_b)
    dec_ref = _decisions(records_ref)   # 同時完成 records_ref 的鍵檢查
    del dec_ref

    labels: dict[str, int] = {}
    for idx, v in confirmed_ref.items():
        if v not in ("good", "bad"):
            raise ValueError(f"confirmed 值僅允許 'good'/'bad':{v!r}")
        i = int(idx)
        if not (0 <= i < len(records_ref)):
            continue                      # 殘留索引(套用結果已換)→ 跳過不猜
        labels[_item_id(records_ref[i])] = 1 if v == "bad" else 0

    run_a_err = {rid: int(dec_a[rid] != lab)
                 for rid, lab in labels.items() if rid in dec_a}
    run_b_err = {rid: int(dec_b[rid] != lab)
                 for rid, lab in labels.items() if rid in dec_b}
    return {"run_a_err": run_a_err, "run_b_err": run_b_err,
            "n_labeled": len(labels)}
