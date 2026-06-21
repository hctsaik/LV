"""即時物件過濾驗收（confidence + 裁切可靠度）。

契約：
  • 資料層：`parse_yolo_boxes_conf` 讀可選第 6 欄 conf（6 欄預測→score；5 欄 GT→None）；
    `discover_yolo_objects` 把 score 帶到每個物件。
  • 過濾在「渲染時」做（不在 _crop_and_embed_objects 內）：純函式 `_obj_keep` /
    `_obj_view_indices` 依共用 session key `obj_conf_range`（信心區間，雙邊）與 `obj_min_short_px`
    （源短邊）過濾；皆 0 = 不過濾；score=None（GT）不受信心門檻影響。
  • 這樣 Visualize/Compare（Run-gated、存整批結果）也能即時重濾，且調低門檻物件會回來。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from streamlit.testing.v1 import AppTest  # noqa: E402


# ── 資料層：6 欄 conf 解析 ───────────────────────────────────────────────
def test_parse_yolo_boxes_conf_reads_optional_6th_col(tmp_path):
    from interaction import parse_yolo_boxes_conf
    f6 = tmp_path / "pred.txt"
    f6.write_text("0 0.5 0.5 0.2 0.2 0.87\n3 0.1 0.1 0.05 0.05 0.12\n", encoding="utf-8")
    rows = parse_yolo_boxes_conf(f6)
    assert [r[0] for r in rows] == [0, 3]
    assert rows[0][5] == pytest.approx(0.87)
    assert rows[1][5] == pytest.approx(0.12)
    # 5 欄 GT → score None（不會誤把座標當 conf）
    f5 = tmp_path / "gt.txt"
    f5.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert parse_yolo_boxes_conf(f5)[0][5] is None


def test_discover_yolo_objects_carries_score(tmp_path):
    from interaction import discover_yolo_objects
    root = tmp_path / "train"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    Image.fromarray(np.zeros((64, 64, 3), "uint8")).save(root / "images" / "a.png")
    (root / "labels" / "a.txt").write_text(
        "0 0.5 0.5 0.4 0.4 0.9\n1 0.2 0.2 0.1 0.1 0.2\n", encoding="utf-8")
    objs = discover_yolo_objects([root / "images" / "a.png"], ["x", "y"])
    assert [o["score"] for o in objs] == [pytest.approx(0.9), pytest.approx(0.2)]


# ── 過濾邏輯：_obj_keep / _obj_view_indices（讀 session → from_string 驅動）──
def test_obj_keep_filters_by_conf_and_size():
    script = (
        "import sys\n"
        f"sys.path.insert(0, r'{SCRIPTS}')\n"
        "import streamlit as st, app\n"
        "recs = [\n"
        "  {'score': 0.9,  'short_px': 50},\n"   # #0 高信心、夠大
        "  {'score': 0.1,  'short_px': 50},\n"   # #1 低信心
        "  {'score': 0.9,  'short_px': 5},\n"    # #2 夠信心、太小
        "  {'score': None, 'short_px': 50},\n"   # #3 無 score（GT）
        "]\n"
        "st.session_state['obj_min_short_px'] = 0\n"
        "st.session_state['_k0'] = app._obj_view_indices(recs)\n"
        "st.session_state['_hasconf'] = app._obj_has_conf(recs)\n"
        # 區間 [0.5, 1.0] → 丟低信心 #1（0.1）；#3 score=None 不受影響
        "st.session_state['obj_conf_range'] = (0.5, 1.0)\n"
        "st.session_state['_klo'] = app._obj_view_indices(recs)\n"
        # 區間 [0.0, 0.5] → 反過來丟高信心 #0/#2（0.9）；留 #1（0.1）、#3（None）
        "st.session_state['obj_conf_range'] = (0.0, 0.5)\n"
        "st.session_state['_khi'] = app._obj_view_indices(recs)\n"
        # 中段 [0.3, 0.95] + 源短邊 10：#1 出區間、#2 在區間但太小 → 只剩 #0、#3
        "st.session_state['obj_conf_range'] = (0.3, 0.95)\n"
        "st.session_state['obj_min_short_px'] = 10\n"
        "st.session_state['_kmid'] = app._obj_view_indices(recs)\n"
        # 全無 score 的資料（GT）→ _obj_has_conf False（信心滑桿不顯示）
        "st.session_state['_hasconf_gt'] = app._obj_has_conf("
        "  [{'score': None, 'short_px': 9}])\n"
    )
    at = AppTest.from_string(script, default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    ss = at.session_state
    print("k0", ss["_k0"], "| klo", ss["_klo"], "| khi", ss["_khi"],
          "| kmid", ss["_kmid"], "| hasconf", ss["_hasconf"], ss["_hasconf_gt"])
    assert ss["_k0"] == [0, 1, 2, 3], "無區間應全留"
    assert ss["_hasconf"] is True, "有 score 的資料應偵測為 has_conf"
    assert ss["_klo"] == [0, 2, 3], "區間 [0.5,1] 應丟低信心 #1（None 不受影響）"
    assert ss["_khi"] == [1, 3], "區間 [0,0.5] 應反過來丟高信心 #0/#2（證明雙邊）"
    assert ss["_kmid"] == [0, 3], "[0.3,0.95]+源短邊10 應只留 #0、#3"
    assert ss["_hasconf_gt"] is False, "全 None（GT）應 has_conf=False → 不顯示信心滑桿"


def test_filter_controls_show_conf_slider_only_when_scored():
    # 信心滑桿（「信心區間」）只在資料含 score 時出現；全 GT（score=None）只剩源短邊。
    # 四個工具共用 _obj_filter_controls，故此處即代表 Visualize/coverage/Compare/objcov 一致。
    script = (
        "import sys\n"
        f"sys.path.insert(0, r'{SCRIPTS}')\n"
        "import streamlit as st, app\n"
        "app._obj_filter_controls('scored', [{'score': 0.5, 'short_px': 40},"
        "                                    {'score': 0.9, 'short_px': 40}])\n"
        "app._obj_filter_controls('gtonly', [{'score': None, 'short_px': 40}])\n"
    )
    at = AppTest.from_string(script, default_timeout=60)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    labels = [s.label for s in at.slider]
    print("slider labels:", labels)
    conf = [l for l in labels if "信心區間" in (l or "")]
    assert len(conf) == 1, "有 score 的那組應顯示一個『信心區間』滑桿（GT 那組不該）"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
