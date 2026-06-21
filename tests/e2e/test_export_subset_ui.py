"""E2E（AppTest）：『匯出子集』頁面 —— 從購物車(viz_export_list)讀影像 →
按匯出 → 在新資料夾產出 YOLO 子資料集。驗證 UI 接線 + export_subset 整合 + dst 硬擋。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPTS = REPO / "scripts"
COCO8 = REPO / "demo" / "coco8" / "train"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (COCO8 / "images").exists(), reason="demo/coco8 not present")


def _harness() -> str:
    return (
        "import sys\n"
        f"sys.path.insert(0, r'{SCRIPTS}')\n"
        "import streamlit as st, app\n"
        "from pathlib import Path\n"
        f"imgs = sorted((Path(r'{COCO8}')/'images').glob('*.jpg'))[:3]\n"
        "if 'viz_export_list' not in st.session_state:\n"
        "    st.session_state['viz_export_list'] = {\n"
        "        str(p): {'path': str(p), 'filename': p.name, 'label': 'person',\n"
        "                 'split': 'train', 'sha256': None, 'source': 'visualize',\n"
        "                 'score': 0.5, 'reason': 'test'} for p in imgs}\n"
        "app._export_subset_ui()\n"
    )


def test_export_subset_page_renders_and_exports(tmp_path):
    at = AppTest.from_string(_harness(), default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    # 頁面有渲染購物車張數
    txt = " ".join(str(m.value) for m in (*at.markdown, *at.caption, *at.subheader))
    assert "購物車共" in txt and ("Export" in txt or "匯出" in txt)

    # 設目的地 → 按匯出
    dst = tmp_path / "subset"
    at.text_input(key="exp_dst").set_value(str(dst)).run()
    at.button(key="exp_run").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]

    # 產出 YOLO 子資料集
    assert (dst / "images").exists(), "images/ 未產出"
    assert (dst / "classes.txt").exists(), "classes.txt 未產出"
    assert (dst / "data.yaml").exists(), "data.yaml 未產出"
    assert (dst / "manifest.csv").exists(), "manifest.csv 未產出"
    copied = list((dst / "images").rglob("*.jpg"))
    print("exported images:", len(copied), "→", dst)
    assert len(copied) >= 1, "沒有影像被複製出去"


def test_export_subset_dst_containment_blocks(tmp_path):
    # 目的地落在來源資料集內 → export_subset 應 raise，UI 顯示被擋下、不寫檔
    at = AppTest.from_string(_harness(), default_timeout=120)
    at.run()
    bad_dst = COCO8 / "subset_inside"   # 在 coco8 來源內
    at.text_input(key="exp_dst").set_value(str(bad_dst)).run()
    at.button(key="exp_run").click().run()
    assert not at.exception, "UI 應捕捉 ValueError 而非崩潰"
    errs = " ".join(str(e.value) for e in at.error)
    print("error shown:", errs[:120])
    assert "擋下" in errs, "未顯示 dst 被擋下的錯誤"
    assert not bad_dst.exists(), "被擋下卻仍寫了檔"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
