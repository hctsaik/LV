from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import tkinter as tk
import uuid
from pathlib import Path
from tkinter import filedialog

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent))

from _utils import (
    available_models,
    extract_embeddings,
    load_model,
    load_text_encoder,
    supports_text_query,
    yaml_class_names,
)
from interaction import (  # noqa: F401  (parse_folder_paths re-exported for tests)
    CAUSE_H0,
    CAUSE_H1,
    CAUSE_H5,
    attribute_escape,
    build_nn_index,
    candidates_to_quiz_records,
    annotation_boxes_for_image,
    crop_bbox,
    folder_has_annotations,
    curation_log_csv,
    diagnose_root_cause,
    diagnose_sparse_points,
    discover_yolo_objects,
    has_seg_like_labels,
    match_shas_to_indices,
    compute_label_disagreement,
    compute_outlier_scores,
    cross_class_nn_pairs,
    draw_yolo_boxes,
    ensure_thumbnails,
    find_duplicate_pairs_embedding,
    find_duplicate_pairs_phash,
    farthest_point_sampling,
    find_similar_indices,
    find_similar_to_vector,
    gray_decision_csv,
    load_scores_csv,
    nearest_anchor,
    nearest_labels,
    rank_gap_fillers,
    reference_coverage,
    select_gray_zone,
    make_thumbnail,
    neighbor_hit_density,
    neighbor_label_entropy,
    parse_folder_paths,
    sparsity_scores,
    records_to_csv,
    selection_points_to_indices,
    snapshots_to_csv,
    spatial_order,
    thumbnail_path_for,
    yolo_label_path_for,
    zip_selected_images,
)
from manifest import rel_key, set_embedding_refs, update_manifest, write_manifest
import labeling_handoff as LH  # unified LV → Labeling hand-over (framework-free)
from completeness import (
    STATE_EMPTY,
    STATE_FAKE,
    STATE_HEALTHY,
    STATE_LOW,
    STATE_MISSING,
    STATE_OVER,
    build_completeness,
    bucketize,
    categorical_buckets,
    image_stats,
)
from quiz import (
    build_quiz,
    fleiss_kappa,
    geometric_skin,
    score_quiz,
)
# compare_distributions imports umap (~22s) + torch + clean_fid + lpips at module
# load — the real reason the shell was slow. Defer it via thin lazy wrappers so it
# only costs time when the user actually opens "Compare Distributions".
def _cmp():
    import compare_distributions as _m
    return _m


def build_projection_figure(*a, **k): return _cmp().build_projection_figure(*a, **k)
def compute_fid(*a, **k): return _cmp().compute_fid(*a, **k)
def compute_inception_score(*a, **k): return _cmp().compute_inception_score(*a, **k)
def compute_kid(*a, **k): return _cmp().compute_kid(*a, **k)
def compute_lpips_score(*a, **k): return _cmp().compute_lpips_score(*a, **k)
def compute_psnr_score(*a, **k): return _cmp().compute_psnr_score(*a, **k)
def compute_ssim_score(*a, **k): return _cmp().compute_ssim_score(*a, **k)
def get_image_paths(*a, **k): return _cmp().get_image_paths(*a, **k)
from visualize_embeddings import build_plotly_figure, discover_images, discover_images_classifier
from safe_io import safe_open_image, safe_read_text, partition_readable


# umap-learn costs ~22s to import (numba JIT) — by far LV's biggest startup cost.
# Load it (and umap_ref, which imports it) LAZILY so the UI shell renders instantly;
# they materialise only when the user actually runs a UMAP projection.
def _umap():
    import umap
    return umap


def stable_umap(*args, **kwargs):
    from umap_ref import stable_umap as _f
    return _f(*args, **kwargs)


def ref_path_for(*args, **kwargs):
    from umap_ref import ref_path_for as _f
    return _f(*args, **kwargs)

_DEFAULT_MODEL = "dinov2_vits14"


def _default_model_index(models: list[str]) -> int:
    """全 app 模型選單的預設:dinov2_vits14(快、輕);不在清單時退回第 0 個。"""
    return models.index(_DEFAULT_MODEL) if _DEFAULT_MODEL in models else 0


def _pick_folder(session_key: str) -> None:
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    path = filedialog.askdirectory(title="選擇資料夾")
    root.destroy()
    if path:
        st.session_state[session_key] = path


def _pick_file(session_key: str, title: str = "選擇檔案", filetypes: list | None = None) -> None:
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes or [("All files", "*.*")])
    root.destroy()
    if path:
        st.session_state[session_key] = path


def _pick_folder_set(list_key: str) -> None:
    """原生資料夾對話框 → 單一資料夾語義:選新的直接取代舊的。
    session 值維持 list 型別(E2E harness 與 render 端都以 list 迭代)。"""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    path = filedialog.askdirectory(title="選擇資料夾")
    root.destroy()
    if path:
        st.session_state[list_key] = [path]


def _pick_folder_into_text(text_key: str, replace: bool = False) -> None:
    """Open the native folder dialog and append the chosen path (one per
    line) to a text-area's value — for the tools that take a pasted path.
    replace=True → 單一資料夾語義:直接取代整個內容(給後端只吃一個資料夾的工具用)。
    Runs as an on_click callback so the value is set before the rerun."""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    path = filedialog.askdirectory(title="選擇資料夾")
    root.destroy()
    if path:
        if replace:
            st.session_state[text_key] = path
            return
        cur = st.session_state.get(text_key, "").rstrip()
        lines = [ln.strip() for ln in cur.splitlines() if ln.strip()]
        if path not in lines:
            lines.append(path)
        st.session_state[text_key] = "\n".join(lines)


def _class_source_expander(prefix: str) -> None:
    """共用「類別來源」expander（與 Visualize 同款 UI）：📄 手動選 classes.txt，
    或逗號手動輸入；皆空＝自動偵測資料集旁的 classes.txt。
    由 _manual_class_names(prefix) 讀取；各工具用自己的 prefix，互不影響。"""
    with st.expander("類別來源（預設自動偵測 classes.txt）"):
        cc1, cc2 = st.columns([4, 1])
        classes_path = st.session_state.get(f"{prefix}_classes_file", "")
        cc1.caption("classes.txt")
        cc1.text(Path(classes_path).name if classes_path else "（自動偵測或手動輸入）")
        if cc2.button("📄", key=f"{prefix}_browse_classes", use_container_width=True,
                      help="選擇 classes.txt"):
            _pick_file(f"{prefix}_classes_file", title="選擇 classes.txt",
                       filetypes=[("Text", "*.txt"), ("All files", "*.*")])
            st.rerun()
        if classes_path:
            if st.button("✕ 清除", key=f"{prefix}_clear_classes", use_container_width=True):
                del st.session_state[f"{prefix}_classes_file"]
                st.rerun()
        st.text_input("Class names — 手動輸入（classes.txt 未選擇時使用）",
                      key=f"{prefix}_class_names_input",
                      placeholder="apple,banana,orange",
                      help="逗號分隔；留空＝自動偵測。")


def _manual_class_names(prefix: str) -> list[str] | None:
    """共用「類別來源」的手動指定值：選檔 > 逗號輸入；皆空回 None（呼叫端自動偵測）。"""
    classes_path = (st.session_state.get(f"{prefix}_classes_file") or "").strip()
    if classes_path and Path(classes_path).exists():
        lines = [ln.strip() for ln in safe_read_text(classes_path).splitlines() if ln.strip()]
        if lines:
            return lines
    manual = (st.session_state.get(f"{prefix}_class_names_input") or "").strip()
    if manual:
        return [c.strip() for c in manual.split(",") if c.strip()]
    return None


def _folder_add_cb(list_key: str, input_key: str) -> None:
    """Set the typed/pasted path as THE folder (single-folder semantics —
    a new pick replaces the old one), then clear the field. The session value
    stays a list so render/E2E contracts are unchanged."""
    v = (st.session_state.get(input_key) or "").strip()
    if v:
        st.session_state[list_key] = [v]
    st.session_state[input_key] = ""


def _folder_add_input(list_key: str, *, help: str | None = None) -> None:
    """Small single-line '輸入路徑 → Enter' field — the only Playwright/headless-
    driveable folder input (the 📁 picker is a native dialog). Single-folder
    semantics: entering a path replaces the previous selection."""
    st.text_input("或輸入路徑後 Enter（取代目前選擇）", key=f"{list_key}_add",
                  on_change=_folder_add_cb, args=(list_key, f"{list_key}_add"),
                  placeholder="例：demo/coco8/train", label_visibility="collapsed",
                  help=help)


def _folder_picker_list(list_key: str, *, add_help: str | None = None) -> list:
    """[輸入路徑|📁] 同列(輸入框左、📁 native picker 右)+ current folder (× remover)。
    Returns paths (list 型別,單一資料夾語義:選新的取代舊的)。"""
    st.session_state.setdefault(list_key, [])
    _fc1, _fc2 = st.columns([5, 1], vertical_alignment="bottom")
    with _fc1:
        _folder_add_input(list_key, help=add_help)  # headless/E2E-driveable 文字加入（📁 為原生對話框）
    with _fc2:
        if st.button("📁", use_container_width=True, key=f"add_{list_key}",
                     help="開啟系統的『選擇資料夾』視窗；也可直接在左邊貼上路徑。"):
            _pick_folder_set(list_key)
            st.rerun()
    for i, folder in enumerate(st.session_state[list_key]):
        c1, c2 = st.columns([5, 1])
        c1.caption(folder)
        if c2.button("✕", key=f"rm_{list_key}_{i}"):
            st.session_state[list_key].pop(i)
            st.rerun()
    if not st.session_state[list_key]:
        st.caption(":gray[尚未選擇資料夾]")
    return list(st.session_state[list_key])


def _picked_paths_display(text_key: str) -> None:
    """唯讀顯示已用 📁 選入的資料夾（session 換行字串）＋清空鈕；取代可貼路徑的輸入框。"""
    lines = [l for l in (st.session_state.get(text_key) or "").splitlines() if l.strip()]
    if not lines:
        st.caption(":gray[尚未選擇資料夾（按上方 📁）]")
        return
    for l in lines:
        st.caption(f"• {l}")
    if st.button("清空", key=f"{text_key}_clear", use_container_width=True):
        st.session_state[text_key] = ""
        st.rerun()


# ── sidebar 執行期收合（Streamlit 1.58 無 runtime API → CSS 注入）──
_SIDEBAR_COLLAPSE_CSS = (
    "<style>[data-testid='stSidebar']{transform:translateX(-110%)!important;"
    "width:0!important;min-width:0!important;visibility:hidden!important;"
    "transition:transform .2s ease}"
    "[data-testid='stSidebarCollapseButton']{display:none!important}</style>"
)


def _collapse_sidebar() -> None:
    st.session_state["_sidebar_collapsed"] = True


def _expand_sidebar() -> None:
    st.session_state["_sidebar_collapsed"] = False


# 14 perceptually distinct hues — first 7 kept stable; the rest let ≥10-class
# datasets (e.g. detection object-level) get a unique colour per class.
_VIZ_COLORS = ["#e74c3c", "#f39c12", "#2ecc71", "#9b59b6", "#3498db", "#1abc9c",
               "#95a5a6", "#e84393", "#fdcb6e", "#34495e", "#0984e3", "#6c5ce7",
               "#badc58", "#576574"]


def _text_on(hex_color: str) -> str:
    """Readable text colour (dark/light) for a given background hex."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#111" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#fff"


def _turbo_hex(t: float) -> str:
    """Turbo colorscale 取樣 → hex(與異常散點圖同款分數色;t∈[0,1])。"""
    from plotly.colors import sample_colorscale
    t = max(0.0, min(1.0, float(t)))
    s = sample_colorscale("Turbo", [t])[0]            # 形如 'rgb(48, 18, 59)'
    r, g, b = (int(float(v)) for v in s[s.find("(") + 1:s.find(")")].split(","))
    return "#%02x%02x%02x" % (r, g, b)


def _viz_color_map(records: list[dict]) -> dict:
    """Stable class→colour map (matches the scatter's sorted-label ordering)."""
    classes = sorted({r.get("label", "") for r in records})
    return {c: _VIZ_COLORS[j % len(_VIZ_COLORS)] for j, c in enumerate(classes)}


def _rec_fname(rec: dict) -> str:
    """徽章/caption 顯示用的原圖 basename——物件級記錄 path 是 .lv_cache 裁切圖,
    image_path 才是使用者資料夾裡的原圖;整圖級只有 path(即原圖)。"""
    return Path(rec.get("image_path") or rec.get("path", "")).name


def _hex_rgb(hexc: str) -> list[int]:
    h = hexc.lstrip("#")
    return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]


def _viz_kmeans_blocks(coords, indices, k: int, cache_key: str):
    """KMeans-partition the shown 3D points into k numbered blocks → lets the
    user pick a block number in 3D and box-select it in a native 2D sub-scatter.
    Returns (blocks {global_idx: block_id}, centers (k,3)). Cached per cache_key+k."""
    ck = f"_viz_blocks_{cache_key}_{k}"
    cached = st.session_state.get(ck)
    if cached and cached.get("n") == len(indices):
        return cached["blocks"], cached["centers"]
    if not indices:
        return {}, np.zeros((0, 3))
    pts = np.array([list(coords[i][:3]) for i in indices], dtype=float)
    if pts.shape[1] < 3:
        pts = np.hstack([pts, np.zeros((len(pts), 3 - pts.shape[1]))])
    from sklearn.cluster import KMeans
    kk = max(1, min(k, len(indices)))
    km = KMeans(n_clusters=kk, n_init=4, random_state=42).fit(pts)
    blocks = {int(i): int(lab) for i, lab in zip(indices, km.labels_)}
    centers = km.cluster_centers_
    st.session_state[ck] = {"n": len(indices), "blocks": blocks, "centers": centers}
    return blocks, centers


def _viz_deck_thumbs(records: list[dict], data_token: str) -> list[str]:
    """Small base64 thumbnails (one per record) for the deck.gl 3D hover preview.
    Encoded once per data_token and cached in session (≈1KB each)."""
    import base64
    import io
    ck = f"_viz_deck_thumbs_{data_token[:8]}"
    cached = st.session_state.get(ck)
    if cached is not None and len(cached) == len(records):
        return cached
    out: list[str] = []
    for r in records:
        uri = ""
        src = _thumb_or_none(Path(r["path"])) or str(r["path"])
        im = safe_open_image(src)            # 壞檔/格式錯回 None → 略過該縮圖
        if im is not None:
            try:
                im.thumbnail((56, 56))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=70)
                uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            except (OSError, ValueError):
                uri = ""
        out.append(uri)
    st.session_state[ck] = out
    return out
_VIZ_SYMBOLS = {"train": "circle", "test": "square", "valid": "diamond"}
_METHOD_KEY = {"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap",
               "LDA(監督)": "lda", "監督UMAP": "sumap"}
# 預設只算監督UMAP(用標籤排版,最易看出分群/挑標錯);無標籤/單一類別時
# 監督投影會被跳過,由「未算出任何投影 → PCA 保底」接手,不會空結果。
_DEFAULT_METHODS = ["監督UMAP"]
_SUPERVISED_METHODS = {"lda", "sumap"}

# ── Remember last selections across Streamlit restarts ───────────────────────
# Sidebar choices are written to a small JSON and reloaded on the next launch,
# so you don't re-pick mode/folders/models/methods every time.
_PERSIST_KEYS = ("viz_mode", "viz_granularity", "viz_methods", "viz_models_sel",
                 "viz_object_policy", "viz_folder_list", "viz_umap_ref")


def _ui_state_path() -> Path:
    import os
    return Path(os.environ.get("LV_UI_STATE")
                or (Path(__file__).resolve().parent.parent / ".lv_ui_state.json"))


def _load_ui_state_once() -> None:
    """Seed session_state from the saved file ONCE per session (before widgets),
    validating against currently-available options so a stale choice can't crash."""
    if st.session_state.get("_ui_state_loaded"):
        return
    st.session_state["_ui_state_loaded"] = True
    import json
    try:
        data = json.loads(_ui_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):  # 合法 JSON 但非物件([]/null/數字)→ 降級不崩(B1)
        return
    method_opts = set(_METHOD_KEY)
    try:
        model_opts = set(available_models())
    except Exception:
        model_opts = set()
    for k, v in data.items():
        if k not in _PERSIST_KEYS:
            continue
        if k == "viz_methods" and isinstance(v, list):
            v = [m for m in v if m in method_opts] or _DEFAULT_METHODS
        elif k == "viz_models_sel" and isinstance(v, list):
            v = [m for m in v if m in model_opts]
            if not v:
                continue
        elif k == "viz_folder_list" and isinstance(v, list):
            v = [f for f in v if Path(f).exists()]
        st.session_state.setdefault(k, v)
    # keep the mode/granularity change-detectors in sync so the restored folder
    # list isn't wiped by the "mode changed → clear" guard on first load
    if "viz_mode" in st.session_state:
        st.session_state.setdefault("_viz_mode_prev", st.session_state["viz_mode"])
    if "viz_granularity" in st.session_state:
        st.session_state.setdefault("_viz_gran_prev", st.session_state["viz_granularity"])


def _save_ui_state() -> None:
    import json
    data = {k: st.session_state.get(k) for k in _PERSIST_KEYS if k in st.session_state}
    try:
        _ui_state_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass

_GRID_BATCH = 60          # cards appended per「載入更多」click
_GRID_CAP = 240           # DOM ceiling agreed in the UX review
_DEFAULT_TOP_OUTLIERS = 50
_SCATTERGL_THRESHOLD = 2000   # 點數超過此值，2D 散點改 WebGL（render 體質 #2）

_USAGE_LOG = Path(__file__).parent.parent / "output" / "usage_log.jsonl"


def _fmt_classes(names: list[str], max_show: int = 10) -> str:
    """Class list for banners — truncated so an 80-class dataset doesn't
    blow up the zero-scroll layout budget."""
    shown = ", ".join(names[:max_show])
    if len(names) > max_show:
        shown += f", …（共 {len(names)} 個）"
    return shown


def _legend_toggle_buttons() -> list[dict]:
    """Plotly client-side 全選/全不選 buttons for the legend.

    These restyle trace visibility in the browser WITHOUT a Streamlit
    rerun, so they never reset the chart's box/lasso selection (a
    Streamlit-side button would re-send the figure and drop the selection
    — the figure-reset trap). "全不選" sets every trace to "legendonly"
    (hidden but still clickable in the legend); "全選類別" brings them back.
    """
    return [dict(
        type="buttons", direction="right",
        x=0.0, y=1.06, xanchor="left", yanchor="bottom",
        pad=dict(t=0, r=0), showactive=False,
        bgcolor="#f0f0f0", bordercolor="#ccc", font=dict(size=11),
        buttons=[
            dict(label="全選類別", method="restyle", args=[{"visible": True}]),
            dict(label="全不選", method="restyle", args=[{"visible": "legendonly"}]),
        ],
    )]


def read_classes_txt(folder: Path) -> list[str] | None:
    """Return class names from <folder-parent>/classes.txt；缺席/空檔時退
    <folder-parent>/data.yaml 的 `names:`(Roboflow/Ultralytics 匯出常只有 data.yaml)。"""
    classes_file = folder.parent / "classes.txt"
    if classes_file.exists():
        lines = [ln.strip() for ln in safe_read_text(classes_file).splitlines() if ln.strip()]
        if lines:
            return lines
    return yaml_class_names(folder.parent / "data.yaml")


def _seg_format_hint(image_paths) -> str:
    """「labels/ 找不到任何 bbox」的診斷尾註:抽樣發現 seg/OBB 行(>6 欄)時點名格式不符,
    避免使用者以為自己沒標註。"""
    if has_seg_like_labels(image_paths):
        return ("\n\n⚠ 偵測到 labels/ 是 **segmentation／OBB 格式**（每行 >6 欄）——"
                "目前只支援偵測框格式（class cx cy w h），請先轉成偵測框再使用。")
    return ""


def _classes_txt_nested(folder: Path) -> list[str] | None:
    """容忍巢狀佈局的 classes.txt 解析（如 …/[Small]/test → indoor/classes.txt）：
    依序探 folder.parent/、folder/、folder.parent.parent/ 的 classes.txt。
    全 app 的「資料夾→類別名」統一走這裡，避免各工具解析深度不一致。"""
    for f in (folder, folder / "_", folder.parent):
        names = read_classes_txt(f)
        if names:
            return names
    return None


# ── 壞檔防呆:統一前置過濾 + 「已略過 N 個壞檔」提示 ─────────────────────────
_SKIP_WARN = ("⚠️ 已略過 {n} 個無法讀取的檔案"
              "（壞檔/格式錯誤，已跳過，不影響其餘）")


def _partition_cache_key(image_paths) -> str:
    """以「資料夾集合 + 每檔 mtime」為鍵；同一輪載入不重複解碼整個資料夾。
    mtime 變了（換檔/覆寫）就重算，壞→好或好→壞都會反映。"""
    sig: list[str] = []
    for p in image_paths:
        p = Path(p)
        try:
            sig.append(f"{p}|{p.stat().st_mtime_ns}")
        except OSError:
            sig.append(f"{p}|missing")
    return hashlib.md5("\n".join(sorted(sig)).encode("utf-8")).hexdigest()


def _partition_image_paths(image_paths, scope: str) -> tuple[list[Path], list[Path]]:
    """把影像清單分成 (可讀, 壞檔)，依 scope+mtime 在 session_state 內快取，
    避免每次 rerun 都重解碼整個資料夾。回傳 (good, bad)。"""
    image_paths = [Path(p) for p in image_paths]
    key = f"_bad_partition_{scope}"
    ck = _partition_cache_key(image_paths)
    cached = st.session_state.get(key)
    if cached is not None and cached.get("ck") == ck:
        return list(cached["good"]), list(cached["bad"])
    good, bad = partition_readable(image_paths)
    st.session_state[key] = {"ck": ck, "good": good, "bad": bad}
    return good, bad


def _warn_skipped(bad) -> None:
    """壞檔非空時，用全 app 統一字串顯示「已略過」提示（E2E 斷言此字串）。"""
    if bad:
        st.warning(_SKIP_WARN.format(n=len(bad)))


def _filter_records_readable(records: list[dict], scope: str) -> list[dict]:
    """以 record 的來源影像（物件級用 image_path、整圖用 path）做前置壞檔過濾。
    壞檔連同其對應 label 自然被排除（下游一律以路徑為鍵），並顯示統一略過提示。"""
    def _src(r):
        return r.get("image_path") or r.get("path")
    paths = [_src(r) for r in records if _src(r)]
    if not paths:
        return records
    good, bad = _partition_image_paths(paths, scope)
    _warn_skipped(bad)
    if not bad:
        return records
    bad_set = {str(Path(p)) for p in bad}
    return [r for r in records if str(Path(_src(r))) not in bad_set]


def _rows_to_csv(header: list[str], rows) -> str:
    """以標準 csv 模組序列化（正確處理逗號／引號／換行）——避免手寫 f-string CSV
    在檔名或類別名含逗號/引號時損毀。"""
    import csv as _csv
    import io as _io
    buf = _io.StringIO()
    w = _csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()



_DISAGREE_SCALE = [[0.0, "#cfd8dc"], [0.5, "#ff9800"], [1.0, "#d32f2f"]]


def _viz_cross_pairs(model: str, token: str, records: list[dict]) -> list[tuple[int, int]]:
    """『最近鄰卻異類』點對（在原始高維 cosine 空間，與分歧度同源），以
    (token, model) 快取避免每次 rerun 重算。"""
    key = f"{token}|{model}"
    cache = st.session_state.get("_viz_pairs")
    if cache and cache.get("key") == key:
        return cache["pairs"]
    raw = st.session_state.get("viz_raw_embeddings", {}).get(model)
    pairs = ([] if raw is None
             else cross_class_nn_pairs(raw, [r["label"] for r in records],
                                       k=1, max_pairs=300))
    st.session_state["_viz_pairs"] = {"key": key, "pairs": pairs}
    return pairs


def _viz_send_to_gray(indices: list[int], model: str) -> None:
    """把散點框選的爭議點直接送進『灰帶覆核』佇列（與 _cov_send_to_quiz 對稱的
    handoff）：寫入灰帶 session keys、依分歧度排序、每類取最明確者當錨例，切頁。"""
    records = st.session_state.get("viz_records")
    raw = st.session_state.get("viz_raw_embeddings", {}).get(model)
    if not records or raw is None or not indices:
        return
    labels = [r["label"] for r in records]
    dis = st.session_state.get("viz_label_disagreement", {}).get(model)

    def _d(i):
        return float(dis[i]) if dis is not None else 0.0
    queue = sorted(dict.fromkeys(int(i) for i in indices), key=_d, reverse=True)
    anchors: dict[str, int | None] = {}
    for c in sorted(set(labels)):
        cand = [i for i in range(len(records)) if labels[i] == c]
        anchors[c] = (min(cand, key=_d) if cand else None)
    st.session_state["gray_records"] = records
    st.session_state["gray_emb"] = raw
    st.session_state["gray_queue"] = queue
    st.session_state["gray_anchors"] = anchors
    st.session_state["gray_dis"] = dis
    st.session_state["gray_disp"] = {}
    st.session_state.pop("gray_mode", None)
    st.session_state.pop("gray_pos", None)
    st.session_state["gray_inbound"] = True
    st.session_state["tool_switch"] = "灰帶覆核"
    # 此鈕在右欄 fragment 內：callback 只重跑 fragment，需設旗標讓 fragment 跳出做
    # app 範圍 rerun 才會真的換到灰帶覆核分頁（與購物車分流鈕一致）
    st.session_state["_cart_app_rerun"] = True
    _log_usage("viz_send_to_gray", n=len(queue))


def _build_viz_figure(
    records: list[dict],
    coords: np.ndarray,
    indices: list[int],
    model_name: str,
    method_label: str,
    dim: int = 2,
    highlight: list[int] | None = None,
    color_by: str = "class",
    disagreement: np.ndarray | None = None,
    pairs: list[tuple[int, int]] | None = None,
) -> go.Figure:
    """Simple scatter for the selected model/method/split combination.

    Each point carries its GLOBAL record index in customdata so box/lasso
    selections map back to records regardless of trace/split filtering.

    ``color_by="disagreement"`` recolours points by k-NN label disagreement
    (gray→red) instead of class, and draws thin lines between "closest
    neighbour but different class" ``pairs`` — the conflicts the eye catches.
    The disagreement spec stays selection-independent, so box/lasso never
    resets.
    """
    use_3d = dim == 3 and coords.shape[1] >= 3
    # render 體質（重評 #2）：SVG Scatter 約 5千–1萬點就卡。點數過門檻才換
    # WebGL Scattergl（撐到十萬級）；小資料集維持 SVG 保留逐點點擊互動。
    scatter2d = go.Scattergl if len(indices) > _SCATTERGL_THRESHOLD else go.Scatter
    show_disagree = color_by == "disagreement" and disagreement is not None
    traces = []

    if show_disagree:
        # ① 相鄰異類連線（靜態，不依賴選取）
        if pairs:
            lx: list[float | None] = []
            ly: list[float | None] = []
            lz: list[float | None] = []
            iset = set(indices)
            for i, j in pairs:
                if i not in iset or j not in iset:
                    continue
                lx += [coords[i, 0], coords[j, 0], None]
                ly += [coords[i, 1], coords[j, 1], None]
                lz += [coords[i, 2] if use_3d else 0,
                       coords[j, 2] if use_3d else 0, None]
            if lx:
                line = dict(mode="lines", name="相鄰異類", legendgroup="pairs",
                            line=dict(width=1, color="rgba(214,39,40,0.35)"),
                            hoverinfo="skip")
                traces.append(go.Scatter3d(x=lx, y=ly, z=lz, **line) if use_3d
                              else go.Scatter(x=lx, y=ly, **line))
        # ② 點以分歧度著色（紅＝鄰居都異類）
        vals = [float(disagreement[i]) for i in indices]
        common = dict(
            mode="markers", name="標籤分歧", showlegend=False,
            marker=dict(color=vals, colorscale=_DISAGREE_SCALE, cmin=0.0, cmax=1.0,
                        showscale=True, colorbar=dict(title="分歧"),
                        size=4 if use_3d else 7, opacity=0.85),
            text=[records[i]["path"].name for i in indices],
            customdata=[[i] for i in indices],
            hovertemplate="%{text}<br>分歧=%{marker.color:.2f}"
                          "<br>#%{customdata[0]}<extra></extra>",
        )
        if use_3d:
            traces.append(go.Scatter3d(
                x=[coords[i, 0] for i in indices], y=[coords[i, 1] for i in indices],
                z=[coords[i, 2] for i in indices], **common))
        else:
            traces.append(scatter2d(
                x=[coords[i, 0] for i in indices],
                y=[coords[i, 1] for i in indices], **common))
    else:
        labels = sorted({records[i]["label"] for i in indices})
        splits = sorted({records[i]["split"] for i in indices})
        color_map = {lbl: _VIZ_COLORS[j % len(_VIZ_COLORS)]
                     for j, lbl in enumerate(labels)}
        for label in labels:
            for split in splits:
                idx = [i for i in indices if records[i]["label"] == label
                       and records[i]["split"] == split]
                if not idx:
                    continue
                common = dict(
                    mode="markers",
                    name=f"{label} ({split})",
                    legendgroup=label,
                    marker=dict(
                        color=color_map[label],
                        symbol=_VIZ_SYMBOLS.get(split, "circle"),
                        size=4 if use_3d else 7,
                        opacity=0.8,
                    ),
                    text=[records[i]["path"].name for i in idx],
                    customdata=[[i] for i in idx],
                    hovertemplate="%{text}<br>Label: " + label + "<br>Split: " + split
                                  + "<br>#%{customdata[0]}<extra></extra>",
                )
                if use_3d:
                    traces.append(go.Scatter3d(
                        x=[coords[i, 0] for i in idx],
                        y=[coords[i, 1] for i in idx],
                        z=[coords[i, 2] for i in idx],
                        **common,
                    ))
                else:
                    traces.append(scatter2d(
                        x=[coords[i, 0] for i in idx],
                        y=[coords[i, 1] for i in idx],
                        **common,
                    ))

    # ring overlay marking the active/highlighted record(s)
    hs = [i for i in (highlight or []) if i in set(indices)]
    if hs:
        ring = dict(
            mode="markers", name="● active", showlegend=False, hoverinfo="skip",
            marker=dict(size=16, color="rgba(0,0,0,0)",
                        line=dict(width=3, color="#222222")),
        )
        if use_3d:
            traces.append(go.Scatter3d(
                x=[coords[i, 0] for i in hs], y=[coords[i, 1] for i in hs],
                z=[coords[i, 2] for i in hs], **ring))
        else:
            traces.append(scatter2d(
                x=[coords[i, 0] for i in hs], y=[coords[i, 1] for i in hs], **ring))

    fig = go.Figure(data=traces)
    # 620px：layout 評審 R2 拍板的散點高度（填滿左欄、消死白）；
    # plotly 預設邊距很肥，壓到貼齊容器。t 留 40 給 全選/全不選 按鈕。
    # 不放圖內標題：model · method 已在正上方 Model/Method 下拉重複顯示，
    # 圖內置中長標題會壓到左上的 全選/全不選 按鈕（排版重疊）。
    layout = dict(height=620,
                  margin=dict(l=10, r=10, t=40, b=10),
                  legend=dict(title="Class (Split)", groupclick="toggleitem"))
    if show_disagree:
        layout["legend"] = dict(title="", orientation="h", y=1.02, yanchor="bottom")
        layout["dragmode"] = "select"  # 分歧檢視＝拖曳即框選紅點，方便整群送覆核
    else:  # 全選/全不選 只在類別圖例下有意義
        layout["updatemenus"] = _legend_toggle_buttons()
    if use_3d:
        layout["scene"] = dict(xaxis_title="C1", yaxis_title="C2", zaxis_title="C3")
    else:
        layout.update(xaxis_title="Component 1", yaxis_title="Component 2")
    fig.update_layout(**layout)
    return fig


def _build_cmp_figure(
    paths_a: list[Path],
    paths_b: list[Path],
    proj: np.ndarray,
    name_a: str,
    name_b: str,
    dim: int = 2,
) -> go.Figure:
    """Simple scatter showing two groups in the selected projection."""
    n_a = len(paths_a)
    use_3d = dim == 3 and proj.shape[1] >= 3

    def _trace(data, names, color, label, base_idx):
        common = dict(
            mode="markers", name=label,
            marker=dict(color=color, size=4 if use_3d else 6, opacity=0.7),
            text=names,
            # global index over paths_a + paths_b → click/lasso 可對回影像
            customdata=[[base_idx + j] for j in range(len(names))],
            hovertemplate="%{text}<br>Group: " + label
                          + "<br>#%{customdata[0]}<extra></extra>",
        )
        if use_3d:
            return go.Scatter3d(x=data[:, 0].tolist(), y=data[:, 1].tolist(), z=data[:, 2].tolist(), **common)
        return go.Scatter(x=data[:, 0].tolist(), y=data[:, 1].tolist(), **common)

    fig = go.Figure(data=[
        _trace(proj[:n_a], [p.name for p in paths_a], "#3498db", name_a, 0),
        _trace(proj[n_a:], [p.name for p in paths_b], "#e74c3c", name_b, n_a),
    ])
    layout = dict(legend=dict(title="Group"), height=560,
                  margin=dict(l=10, r=10, t=44, b=10),
                  updatemenus=_legend_toggle_buttons())
    if use_3d:
        layout["scene"] = dict(xaxis_title="C1", yaxis_title="C2", zaxis_title="C3")
    else:
        layout.update(xaxis_title="Component 1", yaxis_title="Component 2")
    fig.update_layout(**layout)
    return fig


# ── interaction helpers & callbacks ──────────────────────────────────────

def _log_usage(event: str, **fields) -> None:
    """Anonymous local usage log（UX 評審 W8 的留存指標）."""
    try:
        _USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _USAGE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **fields}) + "\n")
    except OSError:
        pass


_CURATION_LOG = Path(__file__).parent.parent / "output" / "curation_log.jsonl"


def _load_curation_log() -> list[dict]:
    """Read the on-disk curation log (most recent first). Survives restart."""
    if not _CURATION_LOG.exists():
        return []
    out = []
    for line in _CURATION_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(reversed(out))


def _append_curation_entry(records: list[dict], indices: list[int],
                           reason: str) -> None:
    """Append one selection + reason to the disk log (策展時間維度 #1)."""
    man = st.session_state.get("viz_manifest", {})
    items = []
    for i in indices:
        p = Path(records[i]["path"])
        entry = man.get(str(p.resolve()), {})
        items.append({"sha256": entry.get("sha256", ""), "filename": p.name,
                      "label": records[i].get("label", ""),
                      "split": records[i].get("split", "")})
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "reason": reason.strip(),
           "n": len(indices), "items": items}
    try:
        _CURATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _CURATION_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        st.toast(f"已記錄此選取（{len(indices)} 張）＋理由", icon="📝")
        _log_usage("curation_log", n=len(indices))
    except OSError as exc:
        st.warning(f"寫入策展日誌失敗：{exc}")


def _curation_reselect(records: list[dict], shas: list[str]) -> None:
    """Re-select a logged selection by content hash（回到上週的選取）."""
    man = st.session_state.get("viz_manifest", {})
    sha_to_index = {}
    for i, r in enumerate(records):
        e = man.get(str(Path(r["path"]).resolve()))
        if e and e.get("sha256"):
            sha_to_index[e["sha256"]] = i
    idxs = match_shas_to_indices(shas, sha_to_index)
    if not idxs:
        st.toast("此日誌的影像不在目前資料集中（可能是別的 Run）。", icon="⚠")
        return
    st.session_state["viz_selection"] = {
        "token": st.session_state.get("viz_data_token"), "indices": idxs}
    st.session_state["viz_grid_limit"] = _GRID_BATCH
    st.session_state["viz_active_image"] = None
    st.toast(f"已重選 {len(idxs)} 張（日誌回放）", icon="↩")


def _set_active_image(idx: int | None, ctx: list[int] | None = None) -> None:
    st.session_state["viz_active_image"] = idx
    if ctx is not None:
        st.session_state["viz_viewer_ctx"] = ctx


def _start_query(idx: int) -> None:
    st.session_state["viz_query_chain"] = [idx]
    st.session_state["viz_panel_view"] = "相似"


def _chain_query(idx: int) -> None:
    chain = st.session_state.setdefault("viz_query_chain", [])
    if not chain or chain[-1] != idx:
        chain.append(idx)


def _truncate_chain(pos: int) -> None:
    chain = st.session_state.get("viz_query_chain", [])
    st.session_state["viz_query_chain"] = chain[: pos + 1]


def _close_query() -> None:
    st.session_state["viz_query_chain"] = []


def _clear_selection(scatter_key: str) -> None:
    """Explicit clear — the ONLY path that empties the selection.

    Also drops the plotly widget's stored event so a stale non-empty event
    cannot resurrect the cleared selection on the next full rerun.
    """
    st.session_state["viz_selection"] = {
        "token": st.session_state.get("viz_data_token"), "indices": []
    }
    st.session_state.pop(scatter_key, None)
    # bump the nonce so the scatter widget remounts fresh (drops the browser-side box)
    st.session_state["_viz_clear_nonce"] = st.session_state.get("_viz_clear_nonce", 0) + 1
    st.session_state["viz_active_image"] = None
    st.session_state["viz_viewer_ctx"] = []


def _cov_clear_selection(scatter_key: str) -> None:
    """完整度熱力圖的『取消框選』——對齊 Visualize 的 _clear_selection：
    清空選取、丟掉 plotly widget 的舊事件、bump nonce 讓散點重新掛載
    （否則同一個 key 會把舊框選再回報回來，按了等於沒清）。"""
    cs = st.session_state.get("cov_sel") or {}
    st.session_state["cov_sel"] = {"token": cs.get("token", ""), "indices": [], "cand": []}
    st.session_state.pop(scatter_key, None)
    st.session_state["_cov_clear_nonce"] = st.session_state.get("_cov_clear_nonce", 0) + 1
    st.session_state["viz_grid_limit"] = _GRID_BATCH


def _load_more() -> None:
    cur = st.session_state.get("viz_grid_limit", _GRID_BATCH)
    st.session_state["viz_grid_limit"] = min(cur + _GRID_BATCH, _GRID_CAP)


def _export_entry(records: list[dict], i: int, source: str = "manual",
                  score: float | None = None, reason: str = "") -> dict:
    r = records[i]
    p = Path(r["path"])
    man = st.session_state.get("viz_manifest", {}).get(str(p.resolve()))
    snap = {"index": i, "filename": p.name, "path": str(p),
            "label": r.get("label", ""), "split": r.get("split", ""),
            "sha256": man.get("sha256") if man else None,
            "source": source,
            "score": (round(float(score), 4) if score is not None else None),
            "reason": reason}
    # 物件級紀錄：path 是裁切快取圖（供檢視/相似度），但匯出要回到「原圖＋標記」。
    # 額外記下原圖路徑、bbox、框 index，並標記 level=object；path 維持裁切圖不動。
    ip = r.get("image_path")
    if ip is not None and str(ip) != str(p):
        snap["level"] = "object"
        snap["image_path"] = str(ip)
        snap["bbox"] = list(r.get("bbox")) if r.get("bbox") is not None else None
        snap["obj_index"] = r.get("obj_index")
    return snap


def _add_to_export(records: list[dict], indices: list[int], source: str = "manual",
                   scores: dict | None = None, reason: str = "") -> tuple[int, int]:
    """Add records to the策展購物車 (keyed by image path), tagging each with
    its provenance (source / score / reason). → (added, skipped)."""
    elist = st.session_state.setdefault("viz_export_list", {})
    added = 0
    for i in indices:
        key = str(records[i]["path"])
        if key not in elist:
            sc = scores.get(i) if isinstance(scores, dict) else None
            elist[key] = _export_entry(records, i, source=source, score=sc,
                                       reason=reason)
            added += 1
    return added, len(indices) - added


def _batch_add(records: list[dict], indices: list[int], source: str = "manual",
               scores: dict | None = None) -> None:
    added, skipped = _add_to_export(records, indices, source=source, scores=scores)
    msg = f"已加入清單 {added} 張" + (f"（略過 {skipped} 張重複）" if skipped else "")
    st.toast(msg, icon="🛒")
    _log_usage("export_list_add", n=added, source=source)


def _add_one(records: list[dict], idx: int, source: str = "manual",
             score: float | None = None) -> None:
    _add_to_export(records, [idx], source=source,
                   scores=({idx: score} if score is not None else None))


def _remove_from_export(path_key: str) -> None:
    st.session_state.get("viz_export_list", {}).pop(path_key, None)


def _clear_export_list() -> None:
    st.session_state["viz_export_list"] = {}
    st.session_state["viz_clear_list_confirm"] = False


def _cart_snapshots(source_filter: str | None = None) -> list[dict]:
    snaps = list(st.session_state.get("viz_export_list", {}).values())
    if source_filter and source_filter != "全部":
        snaps = [s for s in snaps if s.get("source", "") == source_filter]
    return snaps


def _cart_pseudo_records(snaps: list[dict]) -> tuple[list[dict], list[float]]:
    """購物車快照 → (pseudo_records, scores)。score 缺則 0。"""
    recs = [{"path": Path(s["path"]), "label": s.get("label", ""),
             "split": s.get("split", "")} for s in snaps]
    scores = [float(s["score"]) if s.get("score") is not None else 0.0 for s in snaps]
    return recs, scores


def _cart_to_quiz(snaps: list[dict]) -> None:
    """購物車 → 組考卷 一鍵 handoff（與 _cov_send_to_quiz 對稱）。"""
    recs, scores = _cart_pseudo_records(snaps)
    if len(recs) < 4:
        st.toast("購物車至少要 4 張才能出考卷（考卷有 4 種題型）。", icon="⚠")
        return
    st.session_state["quiz_records"] = recs
    st.session_state["quiz_disagreement"] = np.asarray(scores, dtype=float)
    st.session_state["quiz_class_opts"] = sorted({r["label"] for r in recs if r["label"]})
    st.session_state["quiz_inbound"] = True
    for k in ("quiz_spec", "quiz_answers", "quiz_pos"):
        st.session_state.pop(k, None)
    st.session_state["tool_switch"] = "組考卷"
    st.session_state["_cart_app_rerun"] = True
    _log_usage("cart_to_quiz", n=len(recs))


def _cart_to_gray(snaps: list[dict], model: str) -> None:
    """購物車 → 灰帶覆核 一鍵 handoff。購物車不帶 embedding，故以指定模型即時
    重算清單影像特徵，供錨例（nearest_anchor）比對。"""
    recs, scores = _cart_pseudo_records(snaps)
    if not recs:
        return
    embed_fn = load_model(model)
    with st.spinner(f"擷取清單特徵（{len(recs)} 張）以建立覆核佇列…"):
        emb = extract_embeddings([r["path"] for r in recs], embed_fn)
    labels = [r["label"] for r in recs]
    order = sorted(range(len(recs)), key=lambda i: scores[i], reverse=True)
    anchors: dict[str, int | None] = {}
    for c in sorted(set(labels)):
        cand = [i for i in range(len(recs)) if labels[i] == c]
        anchors[c] = (min(cand, key=lambda i: scores[i]) if cand else None)
    st.session_state["gray_records"] = recs
    st.session_state["gray_emb"] = emb
    st.session_state["gray_queue"] = order
    st.session_state["gray_anchors"] = anchors
    st.session_state["gray_dis"] = np.asarray(scores, dtype=float)
    st.session_state["gray_disp"] = {}
    st.session_state.pop("gray_mode", None)
    st.session_state.pop("gray_pos", None)
    st.session_state["gray_inbound"] = True
    st.session_state["tool_switch"] = "灰帶覆核"
    st.session_state["_cart_app_rerun"] = True
    _log_usage("cart_to_gray", n=len(recs))


def _export_items_from_cart(snaps: list[dict]) -> list:
    """購物車快照 → export_subset.ExportItem 清單。
    image-level 用 path；物件級（level=object）改用原圖 image_path＋obj_index，
    匯出成「原圖＋整份標記」。label 走 yolo_label_path_for；類別名走
    _classes_txt_nested（給 class_remap）。"""
    import export_subset as _es
    from interaction import yolo_label_path_for
    items = []
    for s in snaps:
        # 物件級快照：image_path 指向原圖（path 是裁切圖，僅供檢視）。匯出要的是
        # 原圖＋整份標記＋被挑中的框 index，類別名也要從「原圖的資料集」推。
        is_obj = s.get("level") == "object" and s.get("image_path")
        ip = Path(s["image_path"]) if is_obj else Path(s["path"])
        lp = yolo_label_path_for(ip)
        names = (_classes_txt_nested(ip.parent.parent)
                 or _classes_txt_nested(ip.parent) or [])
        obj_index = s.get("obj_index")
        items.append(_es.ExportItem(
            image_path=ip,
            label_path=(lp if lp and lp.exists() else None),
            class_names=list(names),
            split=(s.get("split") or None),
            source_tool=s.get("source", ""),
            source_tag=s.get("source", ""),
            reason=s.get("reason", "") or "",
            level=("object" if is_obj else "image"),
            object_ids=([int(obj_index)] if is_obj and obj_index is not None
                        else None),
            # 物件級的 sha 必須是「原圖」的；傳 None 讓 export_subset 從原圖
            # bytes 重算，絕不沿用裁切圖的 sha 當原圖身分。
            sha256=(None if is_obj else s.get("sha256")),
        ))
    return items


def _export_subset_ui() -> None:
    """📦 匯出子集：把跨工具加入『策展購物車／匯出清單』的影像，收成一個 YOLO 子資料集
    （寫到使用者指定的新資料夾，絕不碰原資料集；dst-containment 由 export_subset 硬擋）。"""
    import export_subset as _es
    from collections import Counter

    st.subheader("📦 匯出（Export）")
    st.caption("把你在各工具加入『策展購物車／匯出清單』的影像，收成一個可用的子資料集。"
               "**寫到你指定的新資料夾，絕不碰原資料集。**")

    snaps = _cart_snapshots()
    if not snaps:
        st.info("購物車目前是空的。到 Visualize 選取面板、完整度、物件覆蓋、灰帶覆核…，"
                "用「加入清單／加入購物車」挑出影像，再回來這裡匯出。")
        return

    by_src = Counter(s.get("source", "?") for s in snaps)
    st.markdown(f"購物車共 **{len(snaps)}** 張 · 跨 {len(by_src)} 個來源："
                + "　".join(f"`{k}×{v}`" for k, v in sorted(by_src.items())))

    # 壞檔防呆:購物車內若有無法讀取的影像，匯出前先提示（不影響其餘可匯出者）
    _, _exp_bad = _partition_image_paths([s["path"] for s in snaps if s.get("path")], "export")
    _warn_skipped(_exp_bad)

    pick = st.selectbox("依來源篩選", ["全部"] + sorted(by_src), key="exp_src_filter")
    shown = _cart_snapshots(None if pick == "全部" else pick)
    with st.container(height=480):
        cols = st.columns(6)
        for j, s in enumerate(shown[:60]):
            with cols[j % 6]:
                t = _thumb_or_none(Path(s["path"]))
                if t is not None:
                    st.image(t, use_container_width=True,
                             caption=f'{s.get("source", "")}·{s.get("label", "") or "?"}')
                else:
                    st.caption("⚠ 缺檔")

    st.divider()
    st.markdown("### 匯出成資料夾（YOLO 子資料集）")
    _dc1, _dc2 = st.columns([5, 1], vertical_alignment="bottom")
    with _dc1:
        dst_str = st.text_input("目的地資料夾（新的／空的；若落在來源資料集內會被擋下）",
                                key="exp_dst", placeholder=r"C:\out\my_subset")
    with _dc2:
        # on_click 回呼在 rerun 開始、widget 實例化「之前」就跑完，因此可以安全寫回
        # 同名 widget key（exp_dst）；用 if st.button()+st.rerun() 反而會在本輪
        # text_input 已實例化後才寫，Streamlit 會丟例外。
        st.button("📁 選資料夾", key="exp_dst_pick", use_container_width=True,
                  on_click=_pick_folder, args=("exp_dst",),
                  help="開啟系統的『選擇資料夾』視窗；也可直接在左邊貼上路徑。")
    _MODE_LBL = {"copy": "複製檔案（推薦）", "symlink": "建捷徑（省空間・進階）",
                 "manifest-only": "只出清單（不搬影像）"}
    _EXIST_LBL = {"skip": "略過同名（推薦）", "rename": "改名加序號", "overwrite": "覆寫"}
    c1, c2 = st.columns(2)
    mode = c1.selectbox("怎麼搬影像", ["copy", "symlink", "manifest-only"], key="exp_mode",
                        format_func=lambda m: _MODE_LBL[m],
                        help="複製檔案＝把影像實際複製一份到新資料夾，最常用最安全（會佔空間）。"
                             "建捷徑＝只放指向原檔的捷徑，省空間但原檔搬走就失效、Windows 常因權限失敗。"
                             "只出清單＝不搬影像，只寫一份 CSV 列出你選了哪些。")
    on_exists = c2.selectbox("目的地已有同名檔時", ["skip", "rename", "overwrite"],
                             key="exp_onexists", format_func=lambda x: _EXIST_LBL[x],
                             help="預設略過已存在的同名檔。")

    items = _export_items_from_cart(shown)
    uniq = len({(it.sha256 or str(it.image_path.resolve())) for it in items})
    n_lab = sum(1 for it in items if it.label_path)
    st.caption(f":gray[將處理 {len(items)} 筆 → sha256 去重後約 **{uniq}** 張影像、{n_lab} 個 label 檔。"
               "輸出含 images/＋labels/＋classes.txt＋data.yaml＋manifest.csv＋lineage.json;"
               "非 YOLO-txt 標註（COCO／VOC XML／LabelMe／NDJSON）會**原格式保留**"
               "（sidecar 隨影像、資料集級檔過濾成只含匯出影像）。]")
    st.caption(":orange[⚠ 這是你刻意挑出的偏斜子集（多為難／稀疏／離群樣本），非隨機抽樣；"
               "直接拿去訓練或當 benchmark 會誤導，評估請用獨立 hold-out。]")

    if st.button(f"▶ 匯出到「{dst_str.strip() or '…'}」", type="primary",
                 disabled=not dst_str.strip(), key="exp_run", use_container_width=True):
        try:
            rep = _es.export_subset(items, Path(dst_str.strip()),
                                    mode=mode, layout="yolo", on_exists=on_exists)
        except ValueError as e:
            st.error(f"❌ 匯出被擋下：{e}")
            return
        st.success(f"✅ 完成 → `{rep.dst}`")
        m = st.columns(5)
        m[0].metric("已匯出", rep.exported)
        m[1].metric("略過", len(rep.skipped))
        m[2].metric("去重", rep.deduped)
        m[3].metric("未解析", len(rep.unresolved))
        m[4].metric("原格式標註", rep.annotations_preserved,
                    help="COCO/VOC/LabelMe/NDJSON 標註被一併保留的影像數"
                         "(YOLO txt 不計,走 labels/)。")
        if rep.warnings:
            with st.expander(f"⚠ 警告（{len(rep.warnings)}）"):
                for w in rep.warnings[:50]:
                    st.caption(f"· {w}")
        if rep.errors:
            st.error("錯誤：" + "；".join(map(str, rep.errors[:10])))
        _log_usage("export_subset", n=rep.exported, source="yolo")


def _anomaly_add_to_cart(records: list[dict], indices: list[int]) -> None:
    """把選定的物件(原圖)加入策展購物車,來源標 anomaly,分數帶上。"""
    scores = {i: records[i].get("score") for i in indices}
    _add_to_export(records, indices, source="anomaly", scores=scores)


def _anomaly_mark(records: list[dict], indices: list[int], verdict: str,
                  context: str = "apply") -> None:
    """把框選的物件標為正常(good)/瑕疵(bad)範例 → 寫入 anomaly_confirmed_{context}。
    ⚠ confirmed 依 index 空間命名空間化:context='build' 對『訓練資料夾』索引(①建乾淨 few-shot bank);
    context='apply' 對『目標資料夾』索引(②就地重評)。兩者**絕不可共用同一槽** —— 否則①的標記殘留到②
    會被當成目標資料夾完全不同物件的標記,把 classify 門檻釘死、AUROC 假成 0(silent-wrong)。"""
    conf = st.session_state.setdefault(f"anomaly_confirmed_{context}", {})
    for i in indices:
        conf[int(i)] = verdict
    kind = "正常" if verdict == "good" else "瑕疵"
    st.toast(f"已標記 {len(indices)} 個為{kind}範例。", icon="✅")


def _anomaly_autoseed_normal(n: int = 12, context: str = "apply") -> None:
    """免框選的快速 few-shot 種子:把『最不可疑』的 n 個物件標為正常範例(用②套用結果;index 空間=apply)。"""
    res = st.session_state.get("anomaly_apply_result") or {}
    ranking = list(res.get("ranking") or [])
    seed = ranking[-n:] if len(ranking) >= n else ranking
    conf = st.session_state.setdefault(f"anomaly_confirmed_{context}", {})
    for i in seed:
        conf[int(i)] = "good"
    st.toast(f"已把最不可疑的 {len(seed)} 個標為正常範例。", icon="✅")


def _anomaly_clear_confirmed(context: str = "apply") -> None:
    st.session_state[f"anomaly_confirmed_{context}"] = {}


def _anomaly_clear_sel(scatter_key: str) -> None:
    """取消框選(對齊 Visualize):丟掉 plotly widget 的舊事件 + bump nonce 讓散點重新掛載,
    否則同一個 key 會把舊框選再回報回來,按了等於沒清。"""
    st.session_state.pop(scatter_key, None)
    st.session_state["_anomaly_clear_nonce"] = st.session_state.get("_anomaly_clear_nonce", 0) + 1


def _anomaly_trigger_rerun() -> None:
    """②就地重評:設旗標,下一輪用最新 confirmed 重算分數(不動模型;正式重建走①)。"""
    st.session_state["_anomaly_reapply"] = True


def _anomaly_bank_default_dir(folder: str, name: str = "anomaly_model") -> str:
    """模型暫存目錄預設落點 .lv_cache/<dataset>_<sha>/<name>(守 no-dataset-writes)。"""
    if not folder:
        return ""
    from object_eval import dataset_cache_dir
    return str(dataset_cache_dir(Path(folder), name))


# ── M7 wizard:唯一模型槽 anomaly_model(① built / 📂 loaded 都寫這;②③ 只讀這)──
def _anomaly_build_request() -> None:
    """① 建模鈕(on_click):只設 flag;實際建模在主體 `_anomaly_build_model` 跑
    ——放 on_click callback 的 st.progress 不會即時串流(卡到跑完才更新)。"""
    st.session_state["anomaly_build_pending"] = True


def _anomaly_apply_request() -> None:
    """② 套用鈕(on_click):只設 flag;實際套用在主體跑(progress 即時串流)。"""
    st.session_state["anomaly_apply_pending"] = True


def _anomaly_watch_scan_request() -> None:
    """▶ 立即掃描鈕(on_click):只設 flag;實際掃描在主體跑(progress 即時串流)。"""
    st.session_state["anomaly_watch_scan_pending"] = True


def _anomaly_build_model() -> None:
    """①:用 train_folder 跑 run_pipeline(confirmed)→ 凍 bank/projection/fewshot;
    語義=defect 且解鎖 → 一起訓 head。**語義硬守衛**:非 defect 一律不訓 head(修 silent-wrong)。
    結尾清掉舊 anomaly_apply_result(模型換了、舊結果 stale)。"""
    import time as _time

    import numpy as _np

    from anomaly_bank_store import confirmed_to_fewshot
    from anomaly_project import fit_projector
    from anomaly_tool import head_unlock_state, run_pipeline
    from dino_head import train_head
    from object_eval import classes_for, dataset_cache_dir, list_images
    st.session_state.pop("_anomaly_model_err", None)
    st.session_state.pop("_anomaly_model_saved", None)
    folders = list(st.session_state.get("anomaly_train_folder") or [])
    if not folders:
        st.session_state["_anomaly_model_err"] = "請先在①選訓練/參考資料夾。"
        return
    model = st.session_state.get("anomaly_model_sel", "dinov2_vits14")
    score_mode = st.session_state.get("anomaly_score_mode", "patch")
    target_res = int(st.session_state.get("anomaly_res", 224)) if score_mode == "patch" else 224
    mode = st.session_state.get("anomaly_mode", "two_stage")
    sample_n = int(st.session_state.get("anomaly_sample_n", 64))
    semantic = st.session_state.get("anomaly_train_semantic", "object")
    n_min = int(st.session_state.get("anomaly_n_min", 8))
    object_source = st.session_state.get("anomaly_object_source", "yolo")
    try:
        roots = [Path(f) for f in folders]
        image_paths: list[Path] = []
        class_names = _manual_class_names("anomaly")  # 手動指定 > 各 root 自動偵測
        for r in roots:
            image_paths.extend(list_images(r))
            class_names = class_names or classes_for(r)
        if not image_paths:
            st.session_state["_anomaly_model_err"] = "資料夾內找不到影像(需 images/ + labels/)。"
            return
        cache = dataset_cache_dir(roots[0], f"anomaly_patch_{score_mode}_{model}_r{target_res}")
        _bar = st.progress(0.0, text=f"準備中…首次會先載入 {model} 模型(約 10~30 秒),再掃描 {len(image_paths)} 張圖")
        _conf_build = st.session_state.get("anomaly_confirmed_build", {})  # ①只用訓練資料夾索引空間的標記
        with st.spinner(f"建模中…首次載入 {model} 模型較久(約 10~30 秒),請稍候"):
            res = run_pipeline(
                image_paths, class_names, mode=mode, score_mode=score_mode,
                sample_n=sample_n, model=model, target_res=target_res,
                confirmed=_conf_build, object_source=object_source,
                cache_dir=cache, progress=lambda f, t: _bar.progress(f, text=t))
        _bar.empty()
        if not res.get("records") or res.get("obj_emb") is None:
            if object_source == "yolo":
                st.session_state["_anomaly_model_err"] = (
                    "找不到 YOLO 物件(資料夾需含 images/ 與 labels/,labels 為 YOLO .txt)。"
                    "若這批圖沒有標註框,可改用『物件來源 → 整張影像』。")
                st.session_state["_anomaly_offer_whole_image"] = True
            else:
                st.session_state["_anomaly_model_err"] = "資料夾內找不到可讀影像。"
            return
        st.session_state.pop("_anomaly_offer_whole_image", None)
        # 凍結投影器(跨資料夾投影回此分佈 + 灰底)
        obj_emb = _np.asarray(res["obj_emb"], dtype=_np.float32)
        normal_set = res.get("normal_set") or []
        good_mask = _np.zeros(len(obj_emb), dtype=bool)
        if normal_set:
            good_mask[_np.asarray(normal_set, dtype=int)] = True
        projection = fit_projector(obj_emb, good_mask if good_mask.any() else None)
        bank = res.get("bank")
        bank_vectors = getattr(bank, "vectors", None)
        # 守衛:patch 模式卻沒建出 bank(殘餘邊界:參考全標瑕疵→無正常可建)→ 明確報錯,
        # 不可靜默存 bank_vectors=None 假顯示「模型已建立」再悄悄退化成物件級(silent-wrong)。
        if score_mode == "patch" and bank_vectors is None:
            st.session_state["_anomaly_model_err"] = (
                "patch 模式建不出 Normal Bank(參考可能全被標為瑕疵,或無正常物件)。"
                "請確認參考資料夾多數為正常,或側欄『分數依據』改用『物件級』。")
            return
        fewshot = confirmed_to_fewshot(_conf_build, res["records"]) if _conf_build else None
        # 語義硬守衛:只有 defect 且解鎖 ≥2 類各達 N_min 才訓 head;object → 永不訓(語義安全鎖)
        head = None
        labels = [r.get("label", "") for r in res["records"]]
        unlock = head_unlock_state(label_semantic=semantic, labels=labels, n_min=n_min)
        if semantic == "defect" and unlock["unlocked"]:
            elig = set(unlock["eligible_classes"])
            keep = [i for i, l in enumerate(labels) if l in elig]
            head = train_head(obj_emb[keep], [labels[i] for i in keep])
            head["_model_name"] = model
            head["_eligible"] = sorted(elig)
        meta = {
            "model": model, "target_res": target_res, "score_mode": score_mode,
            "object_source": object_source,
            "patch_dim": int(bank_vectors.shape[1]) if bank_vectors is not None else None,
            "obj_dim": int(obj_emb.shape[1]), "n_objects": len(res["records"]),
        }
        st.session_state["anomaly_model"] = {
            "source": "built", "ref_folder": str(folders[0]), "_built_at": _time.time(),
            "meta": meta, "label_semantic": semantic,
            "bank_vectors": (_np.asarray(bank_vectors, dtype=_np.float32)
                             if bank_vectors is not None else None),
            "projection": projection, "fewshot": fewshot, "head": head, "_dir": None,
        }
        st.session_state["anomaly_train_result"] = res     # ①框選標 good 用
        st.session_state.pop("anomaly_apply_result", None)  # 模型換了 → 舊套用結果 stale
        _log_usage("anomaly_build", n=len(res["records"]), source=score_mode,
                   semantic=semantic, head=bool(head))
    except Exception as exc:
        st.session_state["_anomaly_model_err"] = f"建模失敗:{exc}"


def _anomaly_apply_model() -> None:
    """②:用 target_folder 跑 run_pipeline,external_bank/external_ref 取自模型,
    model/target_res/score_mode **鎖 model.meta**(非側欄現值,擋 silent-wrong)→ anomaly_apply_result。"""
    import numpy as _np

    from anomaly_score import MemoryBank
    from anomaly_tool import run_pipeline
    from object_eval import classes_for, dataset_cache_dir, list_images
    st.session_state.pop("_anomaly_apply_err", None)
    model = st.session_state.get("anomaly_model")
    if not model:
        st.session_state["_anomaly_apply_err"] = "請先到①建模或載入模型。"
        return
    folders = list(st.session_state.get("anomaly_target_folder") or [])
    if not folders:
        st.session_state["_anomaly_apply_err"] = "請先在②選異常目標資料夾。"
        return
    _m = model["meta"]
    m_model = _m.get("model") or "dinov2_vits14"
    m_res = int(_m.get("target_res") or 224)
    object_source = _m.get("object_source", "yolo")     # 鎖模型的物件來源(②沿用①的決定)
    try:
        roots = [Path(f) for f in folders]
        image_paths: list[Path] = []
        class_names = _manual_class_names("anomaly")  # 手動指定 > 各 root 自動偵測
        for r in roots:
            image_paths.extend(list_images(r))
            class_names = class_names or classes_for(r)
        if not image_paths:
            st.session_state["_anomaly_apply_err"] = "目標資料夾內找不到影像(需 images/ + labels/)。"
            return
        # 鎖模型的評分管線:有 patch bank → patch 級;無 patch bank 但有投影 → object 級對舊 good 離群。
        ext_bank = ext_ref = None
        score_mode = _m.get("score_mode") or "patch"
        if model.get("bank_vectors") is not None:
            ext_bank = MemoryBank(_np.asarray(model["bank_vectors"], dtype=_np.float32))
            score_mode = "patch"
        elif model.get("projection") is not None:
            ext_ref = _np.asarray(model["projection"]["good_obj_emb"], dtype=_np.float32)
            score_mode = "object"
        cache = dataset_cache_dir(roots[0], f"anomaly_patch_{score_mode}_{m_model}_r{m_res}")
        _bar = st.progress(0.0, text=f"準備中…首次會先載入模型(約 10~30 秒),再掃描 {len(image_paths)} 張圖")
        with st.spinner("套用偵測中…首次會先載入模型(約 10~30 秒),請稍候"):
            result = run_pipeline(
                image_paths, class_names, mode="one_stage", score_mode=score_mode,
                sample_n=64, model=m_model, target_res=m_res, object_source=object_source,
                confirmed=st.session_state.get("anomaly_confirmed_apply", {}),  # ②只用目標資料夾索引空間
                cache_dir=cache, external_bank=ext_bank, external_ref=ext_ref,
                progress=lambda f, t: _bar.progress(f, text=t))
        _bar.empty()
        result["_image_paths"] = [str(p) for p in image_paths]
        result["_class_names"] = list(class_names) if class_names else None
        result["_cache"] = str(cache)
        result["_model"] = m_model
        result["_target_res"] = m_res
        result["_model_built_at"] = model.get("_built_at")
        result["_loaded_bank"] = True
        # 跨資料夾沿用 few-shot:把模型的標記用內容定址對齊到新資料夾的物件索引
        if model.get("fewshot"):
            from anomaly_bank_store import match_fewshot_to_indices
            _aligned, _un = match_fewshot_to_indices(model["fewshot"], result["records"])
            result["_fewshot_aligned"] = _aligned
            result["_fewshot_unmatched"] = int(_un)
        st.session_state["anomaly_apply_result"] = result
        _collapse_sidebar()
        _log_usage("anomaly_apply", n=len(result["records"]), source=score_mode)
    except Exception as exc:
        st.session_state["_anomaly_apply_err"] = f"套用失敗:{exc}"


def _anomaly_save_model(model_dir: str, folders: list) -> None:
    """(2)一鍵存模型:save_bank(bank+projection+fewshot+meta+label_semantic)+ save_head + manifest.json,
    全進同一目錄。白名單防呆:絕不寫使用者資料集。"""
    import json as _json
    import time as _time

    from anomaly_bank_store import assert_safe_bank_dir, save_bank
    from dino_head import save_head
    st.session_state.pop("_anomaly_model_err", None)
    st.session_state.pop("_anomaly_model_saved", None)
    model = st.session_state.get("anomaly_model")
    if not model:
        st.session_state["_anomaly_model_err"] = "請先建模或載入模型,再存。"
        return
    if not model_dir:
        st.session_state["_anomaly_model_err"] = "請填模型暫存目錄。"
        return
    try:
        assert_safe_bank_dir(model_dir, folders)
        meta = dict(model["meta"])
        meta["label_semantic"] = model.get("label_semantic", "object")  # 多帶語義欄(設計准許)
        save_bank(model_dir, vectors=model.get("bank_vectors"),
                  projection=model.get("projection"), meta=meta,
                  fewshot=model.get("fewshot"))
        head = model.get("head")
        if head:
            save_head(str(Path(model_dir) / "head.joblib"), head)
        manifest = {
            "model": meta.get("model"), "target_res": meta.get("target_res"),
            "score_mode": meta.get("score_mode"), "label_semantic": meta.get("label_semantic"),
            "eligible_classes": (head or {}).get("_eligible") or [],
            "has_head": bool(head), "built_at": model.get("_built_at"),
            "saved_at": _time.time(),
        }
        (Path(model_dir) / "manifest.json").write_text(
            _json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        model["_dir"] = str(model_dir)
        st.session_state["_anomaly_model_saved"] = str(model_dir)
    except Exception as exc:
        st.session_state["_anomaly_model_err"] = f"存檔失敗:{exc}"


def _anomaly_load_model(model_dir: str) -> None:
    """📂(2)一鍵載回模型:load_bank + load_head(若有)+ 讀 manifest → 填 anomaly_model(source='loaded')。
    舊 bank 無 label_semantic → 預設 'object'(安全側),①顯示可重宣告。清掉舊套用結果逼②重跑。"""
    import json as _json

    from anomaly_bank_store import load_bank
    from dino_head import load_head
    st.session_state.pop("_anomaly_model_err", None)
    try:
        b = load_bank(model_dir)
        _m = b.get("meta", {})
        if not _m.get("model") or _m.get("target_res") in (None, ""):
            raise ValueError("模型 meta 缺 model/target_res,無法安全載入(會回退到當前設定 → 分數錯)")
        _v = b.get("vectors")
        if _v is not None and _m.get("patch_dim") not in (None, int(_v.shape[1])):
            raise ValueError(f"bank.npz 維度 {int(_v.shape[1])} 與 meta.patch_dim {_m.get('patch_dim')} 不符")
        head = None
        _hp = Path(model_dir) / "head.joblib"
        if _hp.exists():
            head = load_head(str(_hp))
        manifest = {}
        _mp = Path(model_dir) / "manifest.json"
        if _mp.exists():
            manifest = _json.loads(_mp.read_text(encoding="utf-8"))
        semantic = (_m.get("label_semantic") or manifest.get("label_semantic") or "object")
        built_at = manifest.get("built_at")
        st.session_state["anomaly_model"] = {
            "source": "loaded", "ref_folder": "", "_built_at": built_at,
            "meta": {k: _m.get(k) for k in
                     ("model", "target_res", "score_mode", "object_source",
                      "patch_dim", "obj_dim", "n_objects")},
            "label_semantic": semantic,
            "bank_vectors": b.get("vectors"), "projection": b.get("projection"),
            "fewshot": b.get("fewshot"), "head": head, "_dir": str(model_dir),
        }
        st.session_state.pop("anomaly_apply_result", None)
        st.session_state.pop("anomaly_train_result", None)
    except Exception as exc:                      # 壞檔/不完整/缺 meta/維度不符 → 明確訊息,不靜默
        st.session_state["_anomaly_model_err"] = f"載入失敗:{exc}"


def _anomaly_pick_qmode(mode_id: str) -> None:
    """③ 2×2 取樣矩陣:點某一格 → 設取樣模式(僅設 anomaly_q_mode)。"""
    st.session_state["anomaly_q_mode"] = mode_id


def _anomaly_priority_for(mode_id, scores, proba, threshold):
    """③:依取樣模式回優先分數陣列。
    novelty→(1,.4,.4);balanced→(1,1,1);pure→(1,0,0);confusion→弱類定向(proba=None 退純 novelty)。"""
    from active_learning import priority_score
    if mode_id == "confusion":
        from active_loop import confusion_targeted_priority
        return confusion_targeted_priority(scores, proba, w_novelty=1.0, w_entropy=1.0)
    _w = {"novelty": (1.0, 0.4, 0.4), "balanced": (1.0, 1.0, 1.0),
          "pure": (1.0, 0.0, 0.0)}.get(mode_id, (1.0, 0.4, 0.4))
    return priority_score(scores, head_proba=proba, anomaly_threshold=float(threshold),
                          w_novelty=_w[0], w_boundary=_w[1], w_disagreement=_w[2])


def _anomaly_queue_labels(sel, records, scores, head, gthr, min_conf, obj_emb):
    """③:每個選中 idx → {label:類別名, verdict:閘控判定, score:異常分數}。
    有 head → gated_predict(正常/類別/Unknown);無 head → record.verdict 對映(可疑/正常)。"""
    import numpy as _np
    out = {}
    gated = None
    if head is not None and obj_emb is not None and len(sel):
        from dino_head import gated_predict
        try:
            _g = gated_predict(head, _np.asarray(obj_emb, dtype=float), scores,
                               anomaly_threshold=float(gthr), min_conf=float(min_conf))
            gated = {int(i): _g[int(i)] for i in sel}
        except Exception as _exc:
            gated = None
            st.warning(f"⚠️ 瑕疵種類判定失敗,本次改用「可疑 / 正常」二分;分種類功能暫不作用。"
                       f"({type(_exc).__name__}: {_exc})")
    for i in sel:
        r = records[int(i)]
        if gated is not None:
            verdict = gated.get(int(i), "—")
        else:
            verdict = "可疑" if r.get("verdict") == "bad" else "正常"
        out[int(i)] = {"label": r.get("label") or "—", "verdict": verdict,
                       "score": float(scores[int(i)])}
    return out


def _anomaly_sidebar_settings() -> None:
    """sidebar 共用設定(跨 tab):model / score_mode / res / mode / sample_n / N_min。
    只作用於『下次①建模』;②③ 一律讀 anomaly_model.meta 鎖死的值,不吃側欄現值。"""
    from _utils import available_models
    with st.sidebar:
        st.markdown("### 🔧 瑕疵偵測設定(作用於下次①建模)")
        _dino = [m for m in available_models() if m.startswith("dinov2")] or ["dinov2_vits14"]
        st.selectbox(
            "DINOv2 模型", _dino,
            index=(_dino.index("dinov2_vits14") if "dinov2_vits14" in _dino else 0),
            key="anomaly_model_sel",
            help="vits14=快、輕(D=384);vitb14=較準、較重(D=768)。換模型用各自獨立快取。")
        _class_source_expander("anomaly")  # YOLO 類別名:手動指定 > 自動偵測(容忍巢狀)
        st.radio("流程", ["two_stage", "one_stage"], key="anomaly_mode",
                 format_func=lambda m: {"two_stage": "2-stage(抽樣→確認→整批)",
                                        "one_stage": "1-stage(直接整批)"}[m],
                 help="2-stage:先看分布、框選確認少數正常 → 對照模式抓細微瑕疵。"
                      "1-stage:直接整批(沒確認時=無監督 leave-one-out,找少見/離群類別)。")
        _score_mode = st.radio("分數依據", ["patch", "object"], key="anomaly_score_mode",
                               format_func=lambda m: {"patch": "patch 級(抓細微瑕疵,較準)",
                                                      "object": "物件級(快,粗)"}[m])
        # DINOv2 patch size 固定 14px;真正可調的是「物件裁切縮放到的解析度」=每物件 patch 細緻度。
        _RES = {224: "224(16×16 patch,快)", 336: "336(24×24,較細)",
                448: "448(32×32,更細)", 560: "560(40×40,最細,慢)"}
        if _score_mode == "patch":
            st.selectbox("解析度(每物件 patch 細緻度)", list(_RES),
                         format_func=lambda r: _RES[r], key="anomaly_res",
                         help="DINOv2 patch=14px 固定;這裡調物件裁切的縮放解析度,"
                              "等效於 patch 細緻度。瑕疵越小選越高。")
        if st.session_state.get("anomaly_mode", "two_stage") == "two_stage":
            st.slider("2-stage 抽樣數", 16, 256, 64, key="anomaly_sample_n")


def _anomaly_switch_to_whole_image() -> None:
    """AC-G5:YOLO 模式偵測不到物件時,一鍵切『整張影像』並重新建模。
    在 callback 內改 widget-keyed session_state 是允許的(渲染前)。"""
    st.session_state["anomaly_object_source"] = "whole_image"
    st.session_state.pop("_anomaly_offer_whole_image", None)
    st.session_state.pop("_anomaly_model_err", None)
    _anomaly_build_model()


def _anomaly_tab_build() -> None:
    """① 建模 / 載入模型:選訓練/參考資料夾 → (1)建模(凍 bank/projection/fewshot/head)
    → (2)一鍵存模型 / 📂 一鍵載回。label 語義白話 radio(object→不訓 head)。"""
    from anomaly_tool import head_unlock_state, label_semantic_hint
    st.markdown("#### ① 建模 / 載入模型")
    st.caption("用一批**多數良品**當『正常參考』建模(凍結 Normal Bank + 投影器);"
               "或 📂 載回先前存好的模型,直接到②套用。")

    # 1a. 訓練/參考資料夾
    st.markdown("**(1) 選訓練/參考資料夾**")
    train_folders = _folder_picker_list("anomaly_train_folder",
                                        add_help="YOLO 偵測資料集:含 images/ 與 labels/")
    # 換訓練資料夾 → 清 build 索引空間的殘留標記/篩選(殘留索引對到新資料夾不同物件 = silent-wrong)
    _tsig = str(hash(tuple(sorted(str(f) for f in (train_folders or [])))))
    if st.session_state.get("_anomaly_train_sig") != _tsig:
        st.session_state["_anomaly_train_sig"] = _tsig
        for _k in ("anomaly_confirmed_build", "anomaly_heat_filter_build", "anomaly_class_filter_build"):
            st.session_state.pop(_k, None)

    # 1b. 物件來源:逐 YOLO 框 vs 整張影像(無需 labels/)
    st.markdown("**(2) 物件來源**")
    st.session_state.setdefault("anomaly_object_source", "yolo")
    _osrc = st.radio("這個資料夾要怎麼當『偵測對象』?", ["yolo", "whole_image"],
                     key="anomaly_object_source",
                     format_func=lambda s: {
                         "yolo": "YOLO 物件(讀 labels/,逐物件偵測)",
                         "whole_image": "整張影像(無需 labels/,每張圖當一個對象)",
                     }[s])

    # 訓練結果(後面「框選確認」區塊也用)→ 須定義在 yolo 條件外,否則 whole_image 模式 UnboundLocalError
    _tr = st.session_state.get("anomaly_train_result") or {}
    _recs = _tr.get("records") or []

    # 1c. label 語義:只有 YOLO 模式才問(整張影像恆 1 類、不訓 head → 語義無意義)
    if _osrc == "yolo":
        st.markdown("**(3) 你的 YOLO 標籤名稱代表什麼?**")
        _hint = label_semantic_hint(_recs) if _recs else {"suggested": "unknown", "hint": ""}
        if "anomaly_train_semantic" not in st.session_state:
            st.session_state["anomaly_train_semantic"] = (
                _hint["suggested"] if _hint["suggested"] in ("object", "defect") else "object")
        st.radio("標籤名稱(classes.txt)指的是『物件本身』還是『缺陷本身』?", ["object", "defect"],
                 key="anomaly_train_semantic",
                 format_func=lambda s: {
                     "object": "物件類別(門 / 窗 / 螺絲…)— 名稱指「東西本身」→ 只找異常,不分瑕疵種類",
                     "defect": "瑕疵類別(刮傷 / 污漬 / 裂痕…)— 名稱指「缺陷本身」→ 額外解鎖:自動分辨缺陷類型",
                 }[s])
        # 每 rerun 同步寫 active_semantic(供任何硬守衛讀;與 build callback 一致)
        st.session_state["anomaly_active_semantic"] = st.session_state["anomaly_train_semantic"]
        if st.session_state["anomaly_train_semantic"] == "defect" and _hint["suggested"] == "object":
            st.warning("⚠ " + _hint["hint"] + " 確定是瑕疵類別嗎?")
        with st.expander("⚙ 進階(每種瑕疵最少幾張才拿來訓練分種類)"):
            st.slider("每種瑕疵最少幾張才拿來訓練分種類(暫定 8;尚未完整驗證)", 2, 30, 8,
                      key="anomaly_n_min")
    else:
        st.session_state["anomaly_active_semantic"] = "object"
        st.caption("整張影像模式:每張圖一個對象、只做異常偵測(不分種類、不用先框標註)。")

    # 1c. 建模按鈕
    st.button("▶ (1) 建立模型", key="anomaly_build_btn", type="primary",
              use_container_width=True, disabled=not train_folders,
              on_click=_anomaly_build_request)
    if st.session_state.pop("anomaly_build_pending", False):   # 主體執行 → progress 即時串流
        _anomaly_build_model()
    if st.session_state.get("_anomaly_model_err"):
        st.error(st.session_state["_anomaly_model_err"])
        if st.session_state.get("_anomaly_offer_whole_image"):
            st.button("→ 改用『整張影像』重新建模(無需 labels/)",
                      key="anomaly_switch_whole_image", type="primary",
                      on_click=_anomaly_switch_to_whole_image)

    model = st.session_state.get("anomaly_model")
    if model:
        _m = model["meta"]
        _src = "已建立" if model.get("source") == "built" else "已載入"
        _head_txt = "能分辨瑕疵種類" if model.get("head") else "只做異常偵測(不分種類)"
        _sem_txt = {"object": "物件類別", "defect": "瑕疵類別"}.get(
            model.get("label_semantic", "object"), model.get("label_semantic"))
        _osrc_txt = {"yolo": "YOLO 物件", "whole_image": "整張影像"}.get(
            _m.get("object_source", "yolo"), "YOLO 物件")
        st.success(f"✅ **模型{_src}** · 來源={_osrc_txt} · {_m.get('model')}@{_m.get('target_res')} · "
                   f"{_m.get('score_mode')} · {_m.get('n_objects')} 物件 · 語義={_sem_txt} · {_head_txt} "
                   "→ 切到 **②套用偵測**")
        if model.get("source") == "loaded":
            st.caption("此模型由 📂 載入;若舊 bank 未記語義,上方 radio 可重新宣告"
                       "(僅影響本 session 顯示,不改已存模型)。")

    # 1d. 統一模型暫存目錄:一鍵存 + 📂 一鍵載
    st.divider()
    st.markdown("**(2) 模型暫存目錄(一鍵存 bank + classifier + manifest;一鍵載回)**")
    _def_dir = _anomaly_bank_default_dir(train_folders[0] if train_folders else "")
    # setdefault(非 value=)初始化:避免 picker 在 callback 寫 session_state 時與 value= 衝突跳警告/例外
    st.session_state.setdefault("anomaly_model_dir", _def_dir)
    st.caption("模型暫存目錄(預設 .lv_cache,不寫你的資料集;可直接打路徑,或按 📁 用原生對話框選)")
    _mdc1, _mdc2 = st.columns([5, 1])
    _mdc1.text_input("模型暫存目錄", key="anomaly_model_dir", label_visibility="collapsed",
                     help="存:bank.npz + projection.npz + fewshot.json + head.joblib + manifest.json 全進此目錄。"
                          "載:一鍵把它們讀回成模型。")
    _mdc2.button("📁 選", key="anomaly_pick_model_dir", use_container_width=True,
                 on_click=_pick_folder, args=("anomaly_model_dir",),
                 help="開原生資料夾對話框選一個目錄(會覆蓋左邊路徑)")
    _mdir = st.session_state.get("anomaly_model_dir") or ""
    _sc1, _sc2 = st.columns(2)
    _sc1.button("💾 存模型", key="anomaly_save_model_btn", disabled=not (model and _mdir),
                use_container_width=True,
                on_click=_anomaly_save_model, args=(_mdir, train_folders))
    _sc2.button("📂 載入模型", key="anomaly_load_model_btn", disabled=not _mdir,
                use_container_width=True,
                on_click=_anomaly_load_model, args=(_mdir,))
    if st.session_state.get("_anomaly_model_saved"):
        st.success(f"✅ 已存:`{st.session_state['_anomaly_model_saved']}` —— "
                   "下次在此填同路徑按「📂 載入模型」即可在新資料夾重用。")

    # 1e. (進階)在訓練資料上框選確認 good/bad → 影響下次建模的乾淨 few-shot bank
    if _recs:
        with st.expander("🔧 (進階)在訓練資料上框選確認正常/瑕疵範例(影響下次建模的乾淨 bank)"):
            st.caption("框選你**確定正常**的點 →「✅ 框選標為正常範例」→ 再按「(1) 建立模型」即用乾淨 few-shot bank。")
            _anomaly_render_scatter(_tr, context="build")
            _conf = st.session_state.get("anomaly_confirmed_build", {})
            _ng = sum(1 for v in _conf.values() if v == "good")
            _nb = sum(1 for v in _conf.values() if v == "bad")
            if _conf:
                _cc1, _cc2 = st.columns([3, 1])
                _cc1.caption(f"📌 已標記正常 **{_ng}** · 瑕疵 **{_nb}** —— 按「(1) 建立模型」用乾淨 few-shot bank 重建。")
                _cc2.button("清除標記", key="anomaly_clear_confirmed_build",
                            on_click=_anomaly_clear_confirmed, args=("build",), use_container_width=True)


# GUI 選樣目標詞彙 → al_batch 引擎 objective。引擎 "novelty"=純 minmax(=舊 GUI「純 novelty」),
# "uncertain"=novelty+boundary+entropy(=GUI「三訊號均衡/balanced」);"pure" 舊鍵相容映到 novelty。
# (Task0:GUI 曾把 balanced/pure 原樣傳 run_batched → 引擎只收 novelty/uncertain/confusion → ValueError。)
_AL_ENGINE_OBJ = {"novelty": "novelty", "pure": "novelty",
                  "balanced": "uncertain", "confusion": "confusion", "similar": "similar"}


def _anomaly_batch_request(objective: str, k: int, resume: bool) -> None:
    """② 分批掃描按鈕(on_click):只記下請求;實際掃描在主體 `_anomaly_batch_execute` 跑。
    放 on_click callback 裡的 `st.progress` **不會即時串流**(會卡到整批掃完才一次更新),
    故改「callback 設 flag → 主體執行」讓進度條真的會動。"""
    st.session_state.pop("anomaly_batch_err", None)
    st.session_state["anomaly_batch_pending"] = (objective, int(k), bool(resume))


def _anomaly_batch_execute() -> None:
    """② 大資料分批掃描(**在主體執行 → progress 即時串流**)。用①存出的凍結模型跑 al_batch.run_batched。
    - 未存模型 → **自動存**(使用者拍板)後再掃(al_batch 讀磁碟目錄,非 in-memory 槽)。
    - object_source **從磁碟 meta.json 讀**(避免 in-memory 載入模型掉此鍵 → whole_image 被默默當 yolo)。
    - checkpoint_dir 走 dataset_cache_dir(.lv_cache)守 no-dataset-writes;結果落 anomaly_batch_result。"""
    import al_batch
    from anomaly_bank_store import load_bank
    from object_eval import classes_for, dataset_cache_dir, list_images
    objective, k, resume = st.session_state.pop("anomaly_batch_pending")
    objective = _AL_ENGINE_OBJ.get(objective, objective)   # GUI 詞彙 → 引擎 objective(Task0)
    ref_vector = None
    if objective == "similar":                             # M12b:參考向量由挑選器存進 session
        ref_vector = st.session_state.get("anomaly_batch_ref_vec")
        if ref_vector is None:
            st.session_state["anomaly_batch_err"] = "「找相似」需先②套用偵測並挑一個參考。"
            return
        ref_vector = np.asarray(ref_vector, dtype=float)
    model = st.session_state.get("anomaly_model")
    folders = list(st.session_state.get("anomaly_target_folder") or [])
    if not model or not folders:
        st.session_state["anomaly_batch_err"] = "請先①建模 / 載入模型,並在②選目標資料夾。"
        return
    try:
        model_dir = model.get("_dir")
        if not model_dir:  # 未存 → 自動存到模型暫存目錄
            _mdir = (st.session_state.get("anomaly_model_dir")
                     or _anomaly_bank_default_dir(str(folders[0])))
            _train = [Path(f) for f in (st.session_state.get("anomaly_train_folder") or folders)]
            _anomaly_save_model(_mdir, _train)
            model_dir = model.get("_dir")
            if not model_dir:
                st.session_state["anomaly_batch_err"] = (
                    st.session_state.get("_anomaly_model_err") or "自動存模型失敗,無法分批掃描。")
                return
            st.session_state["anomaly_batch_autosaved"] = str(model_dir)   # 給 UX 提示「已自動存」
        object_source = load_bank(model_dir).get("meta", {}).get("object_source", "yolo")
        roots = [Path(f) for f in folders]
        image_paths: list[Path] = []
        class_names = None
        for r in roots:
            image_paths.extend(list_images(r))
            class_names = class_names or classes_for(r)
        if not image_paths:
            st.session_state["anomaly_batch_err"] = "目標資料夾內找不到影像。"
            return
        _ck_name = f"al_batch_{objective}"
        if objective == "similar":                         # 每個參考向量各自 checkpoint(可獨立續跑)
            import hashlib as _hl
            _ck_name += "_" + _hl.sha256(ref_vector.astype("float32").tobytes()).hexdigest()[:8]
        ck = dataset_cache_dir(roots[0], _ck_name)
        _bar = st.progress(0.0, text=f"準備中…首次會先載入模型(約 10~30 秒),再開始掃描 {len(image_paths)} 張圖"
                                     "(之後每 20 張更新一次)")

        def _cb(d):
            _tot = max(int(d.get("images_total") or 1), 1)
            _proc = int(d.get("images_processed") or 0)
            _frac = min(max(_proc / _tot, 0.0), 1.0)
            _bar.progress(_frac, text=f"分批掃描 {_proc}/{_tot}({_frac * 100:.0f}%)"
                                      f" · 暫定挑出 {len(d.get('provisional_topk') or [])} 個")

        with st.spinner("分批掃描中…首次會先載入模型(約 10~30 秒),之後每 20 張更新進度,請稍候"):
            res = al_batch.run_batched(
                image_paths, model_dir=model_dir, checkpoint_dir=ck,
                objective=objective, k=int(k), object_source=object_source,
                class_names=class_names, dataset_dirs=roots, progress=_cb, resume=resume,
                ref_vector=ref_vector, batch_size=20, on_identity_mismatch="restart")
        _bar.empty()
        st.session_state["anomaly_batch_result"] = res
        _log_usage("anomaly_batch_scan", n=res.get("objects_scored"),
                   objective=objective, done=res.get("done"))
    except Exception as exc:
        st.session_state["anomaly_batch_err"] = f"分批掃描失敗:{exc}"


def _anomaly_batch_render_queue(res: dict) -> None:
    """標註模式:Top-K 縮圖佇列(用 al_batch 自帶 reason)+ 分數分佈 + 分頁 + 購物車。**不渲染散點**(scale-safe)。"""
    from interaction import crop_bbox
    recs = res.get("topk_records") or []
    done = bool(res.get("done"))
    n = int(res.get("objects_scored") or 0)
    _warn_skipped(res.get("skipped") or [])
    st.markdown(f"{'✅ 分批掃描完成' if done else '⏳ 暫定(掃描中)'} · 已評分 **{n}** 物件 · "
                f"佇列 Top-**{len(recs)}**(目標:{res.get('objective')}"
                f"{' · 已套用多樣性' if res.get('diversity_applied') else ''})")
    if not recs:
        st.info("佇列空(此目標/資料夾未選出物件)。")
        return
    st.caption("分數分佈(佇列各物件的異常分數;標註模式不畫全量散點):")
    st.bar_chart([float(r.get("score", 0.0)) for r in recs], height=140)
    _c1, _c2 = st.columns(2)
    _cols = _c1.slider("每列張數", 2, 6, 3, key="anomaly_batch_cols")
    _th = _c2.slider("縮圖高度(px)", 120, 400, 200, 10, key="anomaly_batch_th")
    _limit = int(st.session_state.get("anomaly_batch_limit", 60))
    _shown = recs[:_limit]
    with st.container(key="anomaly_batch_queue"):
        _wall = st.columns(_cols)
        for _j, _r in enumerate(_shown):
            with _wall[_j % _cols]:
                _im = safe_open_image(_r["image_path"])
                if _im is None:
                    st.caption("⚠ 缺圖")
                else:
                    _crop = crop_bbox(_im, *_r["bbox"], pad=0.1)
                    _cw, _ch = _crop.size
                    _w = max(1, int(_cw * _th / max(1, _ch)))
                    st.image(_crop.resize((_w, int(_th))))
                st.markdown(
                    f"<div style='text-align:center;font-size:0.8em;line-height:1.4'>"
                    f"<b>{(_r.get('label') or '?')}｜{Path(_r['image_path']).name}</b><br>"
                    f"<span style='color:#555'>{_r.get('reason', '')}</span><br>"
                    f"異常 {float(_r.get('score', 0.0)):.2f}</div>", unsafe_allow_html=True)
    if len(recs) > _limit:
        st.button(f"載入更多(+60,共 {len(recs)})", key="anomaly_batch_more",
                  use_container_width=True, on_click=_anomaly_batch_load_more, args=(len(recs),))
    _cart_recs = [{**r, "path": r["image_path"]} for r in recs]   # 注入 path 給既有購物車(以 path 去重)
    st.button(f"🛒 把佇列 {len(recs)} 個加入購物車(送標註)", key="anomaly_batch_cart",
              use_container_width=True, on_click=_anomaly_add_to_cart,
              args=(_cart_recs, list(range(len(_cart_recs)))))


def _anomaly_batch_load_more(total: int) -> None:
    st.session_state["anomaly_batch_limit"] = min(
        int(st.session_state.get("anomaly_batch_limit", 60)) + 60, int(total))


def _anomaly_batch_section(model: dict, target_folders: list) -> None:
    """② 內的「大資料分批掃描」區塊:目標選單 + K + 掃描/續跑 + 標註佇列(不出散點)。"""
    st.divider()
    st.markdown("**⚡ 大資料分批掃描**(物件很多時用這條:可續跑、只給 Top-K 佇列不卡)")
    has_head = bool(model.get("head"))
    # 人話選樣目標(不寫 novelty/head/Top-K 術語)。引擎端 novelty≡pure,不列同義假選項(Task0)。+ 找相似。
    _OBJ = {"novelty": "抓沒看過的異常", "balanced": "又異常又難分種類(需會分種類的模型)",
            "confusion": "最難分辨是哪一種(需會分種類的模型)", "similar": "🔎 找同款(長得像你挑的那一顆)"}
    obj = st.segmented_control("選樣目標", list(_OBJ), format_func=lambda o: _OBJ[o],
                               key="anomaly_batch_objective", default="novelty") or "novelty"
    _needs_head = obj in ("balanced", "confusion")
    if _needs_head and not has_head:
        st.warning("這個目標需要**會分辨瑕疵種類**的模型,但目前這個模型只會找異常、不會分種類。"
                   "請改用『抓沒看過的異常』;或回到①,建模時**標明每個瑕疵是哪一種**"
                   "(每種至少 8 張範例),就能建出會分種類的模型。")
    _sim_ok = True
    if obj == "similar":                              # M12:by-example,參考來源②/①(共用挑選器)
        _sim_ok = _anomaly_pick_reference("batch") is not None
    k = st.slider("要挑前幾個送標註", 10, 500, 100, 10, key="anomaly_batch_k")
    _disabled = ((not target_folders) or (_needs_head and not has_head)
                 or (obj == "similar" and not _sim_ok))
    _b1, _b2 = st.columns(2)
    _b1.button("▶ 大資料分批掃描", key="anomaly_batch_scan_btn", type="primary",
               use_container_width=True, disabled=_disabled,
               on_click=_anomaly_batch_request, args=(obj, int(k), True))
    _b2.button("▶ 繼續上次", key="anomaly_batch_resume_btn", use_container_width=True,
               disabled=_disabled, on_click=_anomaly_batch_request, args=(obj, int(k), True))
    # 掃描在主體跑(非 callback)→ st.progress 才會即時串流更新
    if st.session_state.get("anomaly_batch_pending"):
        _anomaly_batch_execute()
    if st.session_state.get("anomaly_batch_err"):
        st.error(st.session_state["anomaly_batch_err"])
    if st.session_state.get("anomaly_batch_autosaved"):
        st.caption(f"⚡ 已自動存模型至 `{st.session_state['anomaly_batch_autosaved']}` 後開始掃描"
                   "(al_batch 讀磁碟凍結模型)。")
    res = st.session_state.get("anomaly_batch_result")
    if res:
        _anomaly_batch_render_queue(res)


def _anomaly_watch_ensure_model(model, target_folders):
    """確保凍結模型已存到磁碟(未存自動存);回 model_dir 或 None(失敗設 anomaly_watch_err)。"""
    model_dir = model.get("_dir")
    if not model_dir:
        _mdir = (st.session_state.get("anomaly_model_dir")
                 or _anomaly_bank_default_dir(str(target_folders[0])))
        _train = [Path(f) for f in (st.session_state.get("anomaly_train_folder") or target_folders)]
        _anomaly_save_model(_mdir, _train)
        model_dir = model.get("_dir")
        if not model_dir:
            st.session_state["anomaly_watch_err"] = (
                st.session_state.get("_anomaly_model_err") or "自動存模型失敗。")
    return model_dir


def _anomaly_watch_init(objective: str, k: int) -> None:
    """🆕 初始化監看工作區(on_click):未存模型自動存 → al_service.init_workspace 寫 profile.yaml。"""
    import al_service
    st.session_state.pop("anomaly_watch_err", None)
    model = st.session_state.get("anomaly_model")
    target = list(st.session_state.get("anomaly_target_folder") or [])
    ws = (st.session_state.get("anomaly_watch_ws") or "").strip()
    if not (model and target and ws):
        st.session_state["anomaly_watch_err"] = "請先①建模、②選目標資料夾、填工作區目錄。"
        return
    try:
        model_dir = _anomaly_watch_ensure_model(model, [Path(f) for f in target])
        if not model_dir:
            return
        objective = _AL_ENGINE_OBJ.get(objective, objective)   # GUI 詞彙 → 引擎 objective(Task0)
        _init_kw = {}
        if objective == "similar":                             # M12b:參考向量由挑選器存進 session(存 workspace 可攜)
            _rv = st.session_state.get("anomaly_watch_ref_vec")
            if _rv is None:
                st.session_state["anomaly_watch_err"] = "「找相似」監看需先②套用偵測並挑一個參考。"
                return
            _init_kw["reference_vector"] = np.asarray(_rv, dtype=float)
        al_service.init_workspace(ws, name=Path(target[0]).name,
                                  watch_folders=[str(t) for t in target],
                                  model_dir=str(model_dir), objective=objective, k=int(k),
                                  **_init_kw)
        st.session_state["anomaly_watch_inited"] = ws
    except Exception as exc:
        st.session_state["anomaly_watch_err"] = f"初始化監看失敗:{exc}"


def _anomaly_watch_scan() -> None:
    """▶ 立即掃描一次(on_click):profile 缺 → 先 init;跑 al_service.run_once(阻塞+進度)。"""
    import al_service
    st.session_state.pop("anomaly_watch_err", None)
    model = st.session_state.get("anomaly_model")
    target = list(st.session_state.get("anomaly_target_folder") or [])
    ws = (st.session_state.get("anomaly_watch_ws") or "").strip()
    if not (model and target and ws):
        st.session_state["anomaly_watch_err"] = "請先①建模、②選目標資料夾、填工作區目錄。"
        return
    if not (Path(ws) / "profile.yaml").exists():
        _anomaly_watch_init(st.session_state.get("anomaly_watch_objective", "novelty"),
                            int(st.session_state.get("anomaly_watch_k", 100)))
        if st.session_state.get("anomaly_watch_err"):
            return
    try:
        _bar = st.progress(0.0, text="準備中…首次會先載入模型(約 10~30 秒),再開始掃描")

        def _cb(d):
            _tot = max(int(d.get("images_total") or 1), 1)
            _bar.progress(min(max(int(d.get("images_processed") or 0) / _tot, 0.0), 1.0),
                          text=f"監看掃描 {d.get('images_processed')}/{_tot}")

        with st.spinner("監看掃描中…首次會先載入模型(約 10~30 秒),請稍候"):
            res = al_service.run_once(ws, progress=_cb)
        _bar.empty()
        st.session_state["anomaly_watch_last"] = res
        if res.get("status") == "error":
            st.session_state["anomaly_watch_err"] = f"監看掃描失敗:{res.get('reason')}"
    except Exception as exc:
        st.session_state["anomaly_watch_err"] = f"監看掃描失敗:{exc}"


def _anomaly_watch_label(ws: str, item_id: str, decision: str, label: str) -> None:
    import time as _t

    import al_workspace
    al_workspace.append_label(ws, {"id": item_id, "decision": decision, "label": label,
                                   "decided_at": _t.time()})
    st.toast({"good": "已標:正常", "defect": "已標:瑕疵", "skip": "已略過"}.get(decision, "已標"),
             icon="✅")


def _anomaly_watch_render_queue(ws: str, items: list) -> None:
    from interaction import crop_bbox
    st.markdown(f"**監看佇列**(共 {len(items)};✅正常 / 🏷瑕疵 / ⏭略過 → 下輪自動移出)")
    if not items:
        st.info("佇列空(尚無待標,或都標完了)。")
        return
    _limit = int(st.session_state.get("anomaly_watch_limit", 30))
    with st.container(key="anomaly_watch_queue"):
        _cols = st.columns(3)
        for _j, _it in enumerate(items[:_limit]):
            with _cols[_j % 3]:
                _im = safe_open_image(_it.get("image_path"))
                if _im is None:
                    st.caption("⚠ 缺圖")
                else:
                    _crop = crop_bbox(_im, *_it.get("bbox", [0.5, 0.5, 1.0, 1.0]), pad=0.1)
                    _cw, _ch = _crop.size
                    st.image(_crop.resize((160, max(1, int(160 * _ch / max(1, _cw))))))
                st.markdown(f"<div style='font-size:0.8em;color:#555;line-height:1.3'>"
                            f"{_it.get('reason', '')}<br>異常 {float(_it.get('score', 0)):.2f}</div>",
                            unsafe_allow_html=True)
                _id = str(_it.get("id", ""))
                _lb = _it.get("label", "")
                _g, _d, _s = st.columns(3)
                _g.button("✅ 正常", key=f"anomaly_watch_good_{_j}",
                          on_click=_anomaly_watch_label, args=(ws, _id, "good", _lb))
                _d.button("🏷 瑕疵", key=f"anomaly_watch_defect_{_j}",
                          on_click=_anomaly_watch_label, args=(ws, _id, "defect", _lb))
                _s.button("⏭ 略過", key=f"anomaly_watch_skip_{_j}",
                          on_click=_anomaly_watch_label, args=(ws, _id, "skip", _lb))
    if len(items) > _limit:
        st.button(f"載入更多(+30,共 {len(items)})", key="anomaly_watch_more",
                  use_container_width=True, on_click=lambda: st.session_state.update(
                      anomaly_watch_limit=min(_limit + 30, len(items))))


def _anomaly_watch_section(model: dict, target_folders: list) -> None:
    """② 內「🛰 持續監看服務」:工作區設定 + 初始化/匯出 + 立即掃描 + 狀態卡 + 佇列消費。"""
    import al_service
    import al_workspace
    st.divider()
    st.markdown("**🛰 持續監看服務**(設定一個資料夾 → 自動選出最該標的;可**匯出設定**給離線服務跑)")
    _def_ws = (_anomaly_bank_default_dir(str(target_folders[0]), "al_watch")
               if target_folders else "")
    # ws 空且有目標 → 補預設(不可只 setdefault:watch 區塊可能在目標未加時先渲染 → 鎖住空字串)
    if not st.session_state.get("anomaly_watch_ws") and _def_ws:
        st.session_state["anomaly_watch_ws"] = _def_ws
    st.text_input("工作區目錄(.lv_cache;存 profile / 佇列 / 標註)", key="anomaly_watch_ws",
                  label_visibility="collapsed",
                  help="設定/佇列/標註都存這;profile.yaml 可匯出給離線服務。")
    _OBJ = {"novelty": "抓沒看過的異常", "balanced": "又異常又難分種類(需會分種類的模型)",
            "confusion": "最難分辨是哪一種(需會分種類的模型)", "similar": "🔎 找同款(長得像你挑的那一顆)"}
    obj = st.segmented_control("選樣目標", list(_OBJ), format_func=lambda o: _OBJ[o],
                               key="anomaly_watch_objective", default="novelty") or "novelty"
    # watch 也要「會分種類」閘(否則選前兩者但模型不會分種類 → 掃描時 raise;與批次一致,Task0)
    _watch_needs_head = obj in ("balanced", "confusion")
    if _watch_needs_head and not bool(model.get("head")):
        st.warning("這個目標需要**會分辨瑕疵種類**的模型,但目前這個模型只會找異常、不會分種類。"
                   "請改用『抓沒看過的異常』;或回到①,建模時**標明每個瑕疵是哪一種**(每種至少 8 張範例)。")
    _wsim_ok = True
    if obj == "similar":                              # M12:by-example,參考來源②/①(存進 profile 給服務)
        _wsim_ok = _anomaly_pick_reference("watch") is not None
    k = st.slider("每次要挑前幾個", 10, 500, 100, 10, key="anomaly_watch_k")
    ws = (st.session_state.get("anomaly_watch_ws") or "").strip()
    _enabled = (bool(target_folders and ws)
                and not (_watch_needs_head and not bool(model.get("head")))
                and not (obj == "similar" and not _wsim_ok))
    _c1, _c2, _c3 = st.columns(3)
    _c1.button("🆕 初始化監看", key="anomaly_watch_init_btn", use_container_width=True,
               disabled=not _enabled, on_click=_anomaly_watch_init, args=(obj, int(k)))
    _c2.button("▶ 立即掃描一次", key="anomaly_watch_scan_btn", type="primary",
               use_container_width=True, disabled=not _enabled,
               on_click=_anomaly_watch_scan_request)
    _prof = (Path(ws) / "profile.yaml") if ws else None
    with _c3:
        if _prof and _prof.exists():
            st.download_button("📤 匯出設定", key="anomaly_watch_export_btn",
                               data=_prof.read_bytes(),
                               file_name=f"al_profile_{Path(ws).name}.yaml",
                               mime="text/yaml", use_container_width=True)
        else:
            st.button("📤 匯出設定", key="anomaly_watch_export_btn", disabled=True,
                      use_container_width=True, help="先「初始化監看」才有設定可匯出。")
    if st.session_state.pop("anomaly_watch_scan_pending", False):   # 主體執行 → progress 即時串流
        _anomaly_watch_scan()
    if st.session_state.get("anomaly_watch_err"):
        st.error(st.session_state["anomaly_watch_err"])
    _last_res = st.session_state.get("anomaly_watch_last")
    if _last_res and _last_res.get("status") == "ok":
        st.success(f"✅ 監看掃描完成 · 新增 {_last_res.get('new')} · "
                   f"評分 {_last_res.get('objects_scored')} · 佇列 {_last_res.get('queue_len')}")
    with st.container(key="anomaly_watch_status"):
        if ws and (Path(ws) / "profile.yaml").exists():
            try:
                _stat = al_service.status(ws)
            except Exception:
                _stat = {"last_run": None, "queue_len": 0}
            _last = _stat.get("last_run")
            if _last:
                st.caption(f"✅ 上次掃描 {_last.get('run_id')} · 新增 {_last.get('new')} · "
                           f"評分 {_last.get('objects_scored')} · 佇列 {_last.get('queue_len')}")
            else:
                st.caption("🛰 已初始化,尚未跑過掃描 —— 按「▶ 立即掃描一次」。")
        else:
            st.caption("🛰 尚未初始化監看(填工作區目錄 → 按「🆕 初始化監看」)。")
    if ws and (Path(ws) / "queue.jsonl").exists():
        _anomaly_watch_render_queue(ws, al_workspace.read_queue(ws))


def _anomaly_tab_apply() -> None:
    """② 套用偵測:選異常目標資料夾 → 用①鎖定的模型 run_pipeline → 結果概覽 + 散點 + 排序看圖 + 購物車匯出。"""
    st.markdown("#### ② 套用偵測")
    model = st.session_state.get("anomaly_model")
    if not model:
        st.info("請先到 **①建模 / 載入模型** 建立或載入一個模型,再回來套用偵測。")
        return
    _m = model["meta"]
    _osrc_txt = {"yolo": "YOLO 物件", "whole_image": "整張影像"}.get(
        _m.get("object_source", "yolo"), "YOLO 物件")
    st.caption(f":violet[🔒 用模型鎖定的 **{_m.get('model')}@{_m.get('target_res')}** · {_m.get('score_mode')} "
               f"· 來源={_osrc_txt} 評分(非側欄現值;側欄改 model/res 只作用於下次①建模)。]")

    st.markdown("**(1) 選異常目標資料夾**")
    target_folders = _folder_picker_list("anomaly_target_folder",
                                         add_help="要偵測異常的 YOLO 資料夾:含 images/ 與 labels/")
    # 換目標資料夾 → 清 apply 索引空間的殘留標記/篩選(避免殘留索引污染就地重評/門檻)
    _gsig = str(hash(tuple(sorted(str(f) for f in (target_folders or [])))))
    if st.session_state.get("_anomaly_target_sig") != _gsig:
        st.session_state["_anomaly_target_sig"] = _gsig
        for _k in ("anomaly_confirmed_apply", "anomaly_heat_filter_apply", "anomaly_class_filter_apply"):
            st.session_state.pop(_k, None)
    st.button("▶ (2) 套用偵測", key="anomaly_apply_btn", type="primary",
              use_container_width=True, disabled=not target_folders,
              on_click=_anomaly_apply_request)
    if st.session_state.pop("anomaly_apply_pending", False):   # 主體執行 → progress 即時串流
        _anomaly_apply_model()
    if st.session_state.get("_anomaly_apply_err"):
        st.error(st.session_state["_anomaly_apply_err"])

    _anomaly_batch_section(model, target_folders)   # M9:大資料分批掃描(標註佇列,scale-safe)
    _anomaly_watch_section(model, target_folders)    # M10:持續監看服務(設定/匯出/掃描/佇列消費)

    result = st.session_state.get("anomaly_apply_result")
    if not result:
        st.info("選好目標資料夾後按「(2) 套用偵測」,結果(散點 / 排序 / 熱力圖)會顯示在這裡。")
        return
    _warn_skipped(result.get("skipped") or [])
    if not result.get("records"):
        st.error("找不到 YOLO 標註:目標資料夾需含 **images/** 與 **labels/**(YOLO .txt)。請確認後重選。")
        return

    records = result["records"]
    scores = np.asarray(result["scores"], dtype=float)
    n_bad = result["n_bad"]
    auroc = result.get("auroc")
    msg = f"共 **{len(records)}** 個物件 · 判為可疑(bad)**{n_bad}** 個 · 門檻 {result['threshold']:.3f}"
    if auroc is not None:
        msg += f" · 對照確認標籤 AUROC **{auroc:.3f}**"
    st.markdown(msg)
    _al = result.get("_fewshot_aligned") or {}
    _extra = (f" · 對齊 {len(_al)} 個模型 few-shot 標記(對不到 {result.get('_fewshot_unmatched', 0)})"
              if (_al or result.get("_fewshot_unmatched")) else "")
    st.caption(f":violet[🔗 對照模型;目標資料已投影回模型分佈(灰底=參考資料)]{_extra}")
    if n_bad == len(records) and not st.session_state.get("anomaly_confirmed_apply"):
        st.warning("⚠ 全部被判為可疑 = 門檻無法校準。可在下方散點**框選你確定正常的那一團點 →「✅ 框選標為正常範例」**"
                   "→ 按「🔁 就地重評」用乾淨參考重算(不動模型;正式重建走①)。")

    left, right = st.columns([3, 2], gap="medium")
    with left:
        _anomaly_render_scatter(result, context="apply")
        _outliers = [i for i in range(len(records)) if records[i]["verdict"] == "bad"]
        st.button(f"🛒 把判為可疑(bad)的 {len(_outliers)} 個加入購物車",
                  key="anomaly_select_outliers", use_container_width=True,
                  disabled=not _outliers,
                  on_click=_anomaly_add_to_cart, args=(records, _outliers))
        st.button("✅ 自動把最不可疑的 12 個標為正常範例(few-shot 種子)",
                  key="anomaly_autoseed_normal", use_container_width=True,
                  on_click=_anomaly_autoseed_normal, args=(12, "apply"))
        _conf = st.session_state.get("anomaly_confirmed_apply", {})
        _ng = sum(1 for v in _conf.values() if v == "good")
        _nb = sum(1 for v in _conf.values() if v == "bad")
        if _conf:
            sc1, sc2 = st.columns([3, 1])
            sc1.caption(f"📌 已標記正常 **{_ng}** · 瑕疵 **{_nb}** —— 按「🔁 就地重評」用乾淨 few-shot bank 重算。")
            sc2.button("清除標記", key="anomaly_clear_confirmed",
                       on_click=_anomaly_clear_confirmed, args=("apply",), use_container_width=True)
        cart_n = len(_cart_snapshots("anomaly"))
        with st.container(key="anomaly_cart_count"):
            st.caption(f"購物車(瑕疵)目前 {cart_n} 個物件(原圖)。")
        cart = _cart_snapshots("anomaly")
        recs = [{"path": s["path"], "split": s.get("split", ""),
                 "label": s.get("label", "")} for s in cart]
        from interaction import zip_selected_images
        st.download_button(
            "⬇ 匯出選取原圖 ZIP", key="anomaly_export",
            data=(zip_selected_images(recs, list(range(len(recs)))) if recs else b""),
            file_name="anomaly_subset.zip", mime="application/zip",
            disabled=not recs, use_container_width=True)
    with right:
        _anomaly_render_inspector(result, scores)


def _anomaly_prelabel_export(out_dir, lines_by_image, class_names, source_dirs) -> None:
    """on_click:匯出預標到 out_dir(prelabel.export_prelabels 內建 C6 安全檢查)。結果/錯誤落 session。"""
    import prelabel
    st.session_state.pop("anomaly_prelabel_err", None)
    st.session_state.pop("anomaly_prelabel_done", None)
    try:
        st.session_state["anomaly_prelabel_done"] = prelabel.export_prelabels(
            lines_by_image, out_dir, class_names=class_names, source_dirs=source_dirs)
    except Exception as exc:
        st.session_state["anomaly_prelabel_err"] = f"匯出失敗(安全檢查):{exc}"


def _anomaly_prelabel_preview_grid(records, decisions) -> None:
    """預覽:接受的預標縮圖牆(pred·信心·✅收)。無接受 → 提示調參。"""
    shown = [d for d in decisions if d["accept"]][:12]
    if not shown:
        st.caption("目前沒有可接受的預標 —— 調低信心門檻,或多選目標類別。")
        return
    cols = st.columns(4)
    for j, d in enumerate(shown):
        rec = records[d["item"]]
        with cols[j % 4]:
            _im = safe_open_image(rec["image_path"])
            if _im is None:
                st.caption("⚠ 缺圖")
            else:
                st.image(crop_bbox(_im, *rec["bbox"], pad=0.1), use_container_width=True)
            st.caption(f"{d['pred']} · 信心 {d['conf']:.2f} · ✅收")


def _anomaly_prelabel_section(model, result, scores, threshold, obj_emb) -> None:
    """③「🏷️ 預標(分類頭代填,人工最終確認)」:min_conf + 類別過濾 + 預覽 + 匯出到另選目錄(C6)。
    無 head → 友善提示、不出匯出鈕;patch-only 無 obj_emb → 提示不支援(不硬跑)。"""
    import prelabel
    st.divider()
    with st.expander("🏷️ 預標(模型代填瑕疵種類,人工最終確認)", expanded=False):
        head = model.get("head")
        if not head:
            st.info("這個模型**不會分辨瑕疵種類**,無法預標。請回到①,建模時**標明每個瑕疵是哪一種**"
                    "(每種至少 8 張範例)重建。")
            return
        records = result.get("records") or []
        if obj_emb is None or not records:
            st.info("此模型型態或結果無物件 embedding,無法預標(需②套用偵測產生含 embedding 的結果)。")
            return
        classes = list(head.get("classes") or [])
        st.caption("模型對每個物件代填類別;**你是最終裁決**。沒把握的(Unknown / 低於信心門檻)一律留白,不亂猜。")
        min_conf = st.slider("信心門檻(低於此 → 留白給你手標)", 0.0, 1.0, 0.5, 0.05,
                             key="anomaly_prelabel_min_conf")
        target = st.multiselect("只預標這些類別(其餘留給你手標)", classes, default=classes,
                                key="anomaly_prelabel_classes")
        decisions = prelabel.prelabel_records(
            records, np.asarray(obj_emb, dtype=float), scores, head,
            anomaly_threshold=float(threshold), min_conf=float(min_conf),
            target_classes=list(target))
        n_accept = sum(1 for d in decisions if d["accept"])
        with st.container(key="anomaly_prelabel_preview"):
            st.markdown(f"**接受 {n_accept}** / 共 {len(decisions)} 個物件"
                        "(其餘 = 正常 / Unknown / 未選類別 → 留白)")
            _anomaly_prelabel_preview_grid(records, decisions)
        _tf = st.session_state.get("anomaly_target_folder") or []
        _def_out = _anomaly_bank_default_dir(str(_tf[0]), "prelabel_out") if _tf else ""
        if not st.session_state.get("anomaly_prelabel_out_dir") and _def_out:
            st.session_state["anomaly_prelabel_out_dir"] = _def_out
        st.text_input("輸出資料夾(另存 YOLO labels;來源資料集一律不動)",
                      key="anomaly_prelabel_out_dir",
                      help="預標寫這;C6:不得等於/位於來源資料集內。")
        out_dir = (st.session_state.get("anomaly_prelabel_out_dir") or "").strip()
        lines_by_image = prelabel.to_yolo_lines(records, decisions, class_names=classes)
        st.button("⬇ 匯出預標 YOLO", key="anomaly_prelabel_export_btn", type="primary",
                  disabled=not out_dir, use_container_width=True,
                  on_click=_anomaly_prelabel_export,
                  args=(out_dir, lines_by_image, classes, [str(f) for f in _tf]))
        if st.session_state.get("anomaly_prelabel_err"):
            st.error(st.session_state["anomaly_prelabel_err"])
        _done = st.session_state.get("anomaly_prelabel_done")
        if _done:
            st.success(f"✅ 預標匯出完成 · 寫 {_done['written']} 檔 · "
                       f"{_done['objects']} 個標註 → `{_done['out_dir']}`")


def _anomaly_pick_reference(prefix: str):
    """共用參考挑選器(③/M9/M10):選**參考來源**(② 這次掃描結果 / ① 建模已知範例)→ 挑**一顆**物件當範本
    → 回 ref_vector 並存 session[anomaly_{prefix}_ref_vec]。這是『找長得像這一顆』的 by-example;
    想找某種**已知瑕疵種類**請用『🏷️ 預標』(分類頭更準)。① 來源只在建模那一輪有(載入舊模型則只有②)。"""
    _srcs = []
    _ar = st.session_state.get("anomaly_apply_result") or {}
    if _ar.get("obj_emb") is not None and _ar.get("records"):
        _srcs.append(("apply", "② 這次掃描結果", _ar["records"], _ar["obj_emb"]))
    _tr = st.session_state.get("anomaly_train_result") or {}
    if _tr.get("obj_emb") is not None and _tr.get("records"):
        _srcs.append(("train", "① 建模已知範例", _tr["records"], _tr["obj_emb"]))
    if not _srcs:
        st.session_state[f"anomaly_{prefix}_ref_vec"] = None
        st.info("沒有可當範本的物件(先按②套用偵測;建模那一輪也會留下①的範例)。")
        return None
    _names = {s[0]: s[1] for s in _srcs}
    src = (st.radio("參考來源", list(_names), horizontal=True, key=f"anomaly_{prefix}_ref_src",
                    format_func=lambda k: _names[k]) if len(_srcs) > 1 else _srcs[0][0])
    _sel = next(s for s in _srcs if s[0] == src)
    records, emb = _sel[2], np.asarray(_sel[3], dtype=float)
    labels = [str(r.get("label") or "—") for r in records]
    pick = int(st.selectbox("挑一顆當範本(找長得最像它的)", list(range(len(records))),
                            key=f"anomaly_{prefix}_ref_idx",
                            format_func=lambda i: f"第 {i} 個 · {labels[i]} · {_rec_fname(records[i])}"))
    rv = emb[pick]
    with st.container(key=f"anomaly_{prefix}_ref"):     # 縮圖讓你看到挑的是哪一顆
        _rr = records[pick]
        _im = safe_open_image(_rr["image_path"])
        if _im is not None:
            st.image(crop_bbox(_im, *_rr["bbox"], pad=0.1), width=160)
        st.caption(f"範本:{_rr.get('label') or '—'}｜{_rec_fname(_rr)}")
    st.session_state[f"anomaly_{prefix}_ref_vec"] = rv
    st.session_state[f"anomaly_{prefix}_ref_src_val"] = src
    return rv


def _anomaly_similar_section(result, obj_emb) -> None:
    """③「🔎 找同款」:挑**一顆**當範本(來自②掃描結果或①建模範例)→ 依相似度把長得最像的排到最前面。
    這是 by-example『找長得像這一顆』;想找某種**已知瑕疵種類**請用『🏷️ 預標』(分類頭更準)。"""
    import similarity
    st.divider()
    with st.expander("🔎 找同款(找長得像你挑的那一顆的)", expanded=False):
        records = result.get("records") or []
        if obj_emb is None or len(records) < 2:
            st.info("這個結果沒有足夠物件(先按②套用偵測產生 ≥2 個物件)。")
            return
        emb = np.asarray(obj_emb, dtype=float)
        st.caption("挑一顆當**範本**,整批就把長得最像它的排到最前面送你標註。"
                   "想找某種**已知瑕疵種類**?用上面的『🏷️ 預標』選類別會更準。")
        rv = _anomaly_pick_reference("sim")
        if rv is None:
            return
        k = st.slider("取幾個最像的送標註", 1, min(48, len(records) - 1),
                      min(12, len(records) - 1), key="anomaly_sim_k")
        # 只有『參考來自②(=搜尋集)』時才排除範本自身;來自①的範本不在②裡,不用排除
        _excl = (int(st.session_state.get("anomaly_sim_ref_idx", -1))
                 if st.session_state.get("anomaly_sim_ref_src_val") == "apply" else -1)
        pri = np.asarray(similarity.similarity_priority(emb, rv), dtype=float)
        order = [int(i) for i in np.argsort(-pri) if int(i) != _excl][:int(k)]
        with st.container(key="anomaly_sim_queue"):
            _cols = st.columns(4)
            for j, i in enumerate(order):
                r = records[i]
                with _cols[j % 4]:
                    _im = safe_open_image(r["image_path"])
                    if _im is not None:
                        st.image(crop_bbox(_im, *r["bbox"], pad=0.1), use_container_width=True)
                    st.caption(f"{r.get('label') or '—'}｜相似 {pri[i]:.2f}")
            st.button(f"🛒 把這 {len(order)} 個最像的加入購物車(送標註)", key="anomaly_sim_cart",
                      disabled=not order, use_container_width=True,
                      on_click=_anomaly_add_to_cart, args=(records, order))


def _anomaly_tab_sample() -> None:
    """③ 挑樣送人工標:2×2 取樣矩陣 master-detail → 點格出大圖牆 + 有意義標籤 → 加入購物車。
    4 模式排 2×2(對齊現 radio 順序):偏novelty / 弱類定向 / 三訊號均衡 / 純novelty。"""
    from collections import Counter

    from active_learning import select_for_labeling
    from interaction import crop_bbox
    st.markdown("#### ③ 挑樣送人工標")
    model = st.session_state.get("anomaly_model")
    result = st.session_state.get("anomaly_apply_result")
    if not model:
        st.info("請先到 **①建模 / 載入模型**,再到 **②套用偵測** 產生結果,才能挑樣。")
        return
    if not result or not result.get("records"):
        st.info("請先到 **②套用偵測** 對目標資料夾跑出結果,這裡才有物件可挑。")
        return

    records = result["records"]
    scores = np.asarray(result["scores"], dtype=float)
    obj_emb = result.get("obj_emb")
    threshold = float(result["threshold"])
    head = model.get("head")

    st.caption("優先分數 = **Novelty**(Normal Bank 異常)+ **近決策邊界** + **Model Disagreement**;"
               "選樣再加 **Cluster Diversity**。距離訊號比 softmax 可靠 → novelty 一定納入。")
    _kmax = max(1, min(48, len(records)))
    _alk = (st.slider("挑幾個送標註", 1, _kmax, min(12, _kmax), key="anomaly_q_k")
            if _kmax > 1 else 1)                        # 防呆:物件極少時 slider min==max 會爆
    _almpc = st.slider("每群上限(diversity)", 1, 5, 2, key="anomaly_q_mpc")

    # head proba / pred(diversity 分群 + 弱類定向用)
    _proba, _pred = None, None
    if head is not None and obj_emb is not None:
        from dino_head import predict_head
        try:
            _pred, _, _proba = predict_head(head, np.asarray(obj_emb, dtype=float))
        except Exception as _exc:
            _proba = _pred = None
            st.warning(f"⚠️ 瑕疵種類預測失敗,取樣改用分群、四格只留『純異常分數』;分種類功能暫不作用。"
                       f"({type(_exc).__name__}: {_exc})")
    _div = (_pred if _pred is not None else (result.get("cluster") or {}).get("labels"))

    # gated 門檻(供徽章/標籤的閘控判定)。沿用 gate_threshold(quantile;有 head 才實際拆 Unknown)。
    _min_conf = 0.5
    if head is not None and obj_emb is not None:
        from anomaly_tool import gate_threshold as _gtf
        _gt = _gtf(scores, contamination=0.05,                              # ③吃目標資料夾 index 空間
                   confirmed=st.session_state.get("anomaly_confirmed_apply"), mode="quantile")
        _gthr = float(_gt["threshold"]) if len(scores) else threshold
    else:
        _gthr = threshold

    # 2×2 模式(固定順序對齊舊 radio):偏novelty / 弱類定向 / 三訊號均衡 / 純novelty
    _modes = [("novelty", "偏異常(稀有/沒看過的更優先)"), ("confusion", "最難分辨是哪一種的"),
              ("balanced", "又異常又難分種類"), ("pure", "純異常分數")]
    # 偏novelty/弱類定向/均衡 的差異全靠 head 的「邊界 / 分歧 / 混淆」訊號;無 head 時這些訊號=0,
    # 三者與純 novelty 變成完全相同的排序 → 反灰它們,避免顯示「其實一模一樣」的假選項,只留純 novelty 可點。
    _HEAD_DEP = {"novelty", "confusion", "balanced"}
    # 反灰判據用 _proba(非 head 是否存在):模式差異全來自 head_proba 的 boundary/disagreement/entropy,
    # _proba 不可用(無 head、obj_emb 缺、predict_head 失敗)時四模式等價 → 比只看 head 更準。
    _no_signal = _proba is None
    st.session_state.setdefault("anomaly_q_mode", "novelty")
    if _no_signal and st.session_state["anomaly_q_mode"] in _HEAD_DEP:
        st.session_state["anomaly_q_mode"] = "pure"   # 無分類頭訊號 → 強制落在唯一有意義的格
    _active = st.session_state["anomaly_q_mode"]

    # 每格各算自己的 _sel + 成分拆解徽章(有 head→正常N·類別M·未知U;無 head→可疑N·正常M)
    def _sel_for(mode_id):
        _pri = _anomaly_priority_for(mode_id, scores, _proba, threshold)
        return select_for_labeling(_pri, k=_alk, cluster_labels=_div, max_per_cluster=_almpc)

    def _badge(sel):
        ql = _anomaly_queue_labels(sel, records, scores, head, _gthr, _min_conf, obj_emb)
        verdicts = [ql[i]["verdict"] for i in sel]
        # 依「實際 verdict 內容」拆,不靠 head 是否存在:head 存在但 gated_predict 退場(proba=None)時
        # verdict 仍是『可疑/正常』二分,若用 head-存在判斷會把『可疑』誤算成『類別』。
        if any(v == "可疑" for v in verdicts):
            c = Counter(verdicts)
            return f"可疑 {c.get('可疑', 0)}·正常 {c.get('正常', 0)}"
        n_norm = sum(1 for v in verdicts if v == "正常")
        n_unk = sum(1 for v in verdicts if v == "Unknown")
        n_cls = len(verdicts) - n_norm - n_unk
        return f"正常 {n_norm}·類別 {n_cls}·未知 {n_unk}"

    st.markdown("**取樣模式(點一格 → 右下出大圖牆)**")
    if _no_signal:
        st.caption("🔒 前三種要有**會分辨瑕疵種類**的模型(①用瑕疵種類標籤、每種至少 8 張建模)"
                   "才和『純異常分數』有差別;目前只用異常分數排序。")
    _sel_cache = {}
    r0 = st.columns(2)
    r1 = st.columns(2)
    _cells = [(r0[0], _modes[0]), (r0[1], _modes[1]), (r1[0], _modes[2]), (r1[1], _modes[3])]
    for _col, (mid, mname) in _cells:
        _disabled = _no_signal and mid in _HEAD_DEP
        with _col:
            if _disabled:
                st.button(f"{mname}\n\n🔒 需會分種類的模型", key=f"anomaly_qmode_{mid}",
                          use_container_width=True, disabled=True)
            else:
                _sel_cache[mid] = _sel_for(mid)
                st.button(f"{mname}\n\n{_badge(_sel_cache[mid])}", key=f"anomaly_qmode_{mid}",
                          use_container_width=True,
                          type=("primary" if mid == _active else "secondary"),
                          on_click=_anomaly_pick_qmode, args=(mid,))

    # detail:選中模式的大圖牆
    _sel = _sel_cache.get(_active, [])
    _active_name = dict(_modes).get(_active, _active)
    st.markdown(f"**{_active_name} — 取樣佇列(優先序,共 {len(_sel)} 個)**"
                + ("" if head is not None else " ｜ 只做異常偵測 → 只用異常分數排序,不分種類"))
    _gc1, _gc2 = st.columns(2)
    _cols = _gc1.slider("每列張數", 2, 6, 3, key="anomaly_q_cols")
    _th = _gc2.slider("縮圖高度(px)", 120, 400, 200, 10, key="anomaly_q_th")
    _labels = _anomaly_queue_labels(_sel, records, scores, head, _gthr, _min_conf, obj_emb)
    if _sel:
        _wall = st.columns(_cols)
        for _j, _i in enumerate(_sel):
            _r = records[_i]
            with _wall[_j % _cols]:
                _im = safe_open_image(_r["image_path"])
                if _im is None:
                    st.caption("⚠ 缺圖")
                else:
                    # 縮圖高度 slider 真正生效:把裁切縮到指定像素高(等比),不吃欄寬。
                    _crop = crop_bbox(_im, *_r["bbox"], pad=0.1)
                    _cw, _ch = _crop.size
                    _w = max(1, int(_cw * _th / max(1, _ch)))
                    st.image(_crop.resize((_w, int(_th))))
                _info = _labels[int(_i)]
                # 影像標籤三行:類別名 / 閘控判定(正常·類別·Unknown)/ 異常分數
                _vc = {"正常": "#1a7f37", "Unknown": "#9a6700"}.get(_info["verdict"], "#cc0000")
                st.markdown(
                    f"<div style='text-align:center;font-size:0.8em;line-height:1.45'>"
                    f"<b>{_info['label']}｜{_rec_fname(_r)}</b><br>"
                    f"<span style='color:{_vc};font-weight:600'>判定:{_info['verdict']}</span><br>"
                    f"異常 {_info['score']:.2f}</div>",
                    unsafe_allow_html=True)
    st.button(f"🛒 把「{_active_name}」佇列 {len(_sel)} 個加入購物車(送標註)",
              key="anomaly_q_cart", disabled=not _sel, use_container_width=True,
              on_click=_anomaly_add_to_cart, args=(records, _sel))

    # M11:分類頭預標(代填類別 → 匯出 YOLO 到另選目錄;人工最終確認)
    _anomaly_prelabel_section(model, result, scores, threshold, obj_emb)
    # M12a:找相似(挑參考物件 → cosine 相似度排序佇列)
    _anomaly_similar_section(result, obj_emb)


def _anomaly_render_scatter(result: dict, *, context: str) -> None:
    """散點圖框選(物件級 embedding 投影,異常分數上色)+ 標 good/bad + 框選預覽。
    context='build':標記影響下次①建模;context='apply':提供「🔁 就地重評」(用 confirmed 重算不動模型)。
    沿用既有散點/篩選連動/取消框選/預覽邏輯(只是搬進對的 tab)。"""
    import plotly.graph_objects as go

    from interaction import crop_bbox, selection_points_to_indices
    records = result["records"]
    scores = np.asarray(result["scores"], dtype=float)
    smin, smax = (float(scores.min()), float(scores.max())) if len(scores) else (0.0, 1.0)
    emb = np.asarray(result.get("obj_emb"), dtype=float) if result.get("obj_emb") is not None else np.zeros((0, 0))
    sel_idx: list[int] = []
    if emb.shape[0] >= 2 and emb.shape[1] >= 2:
        # 模型有投影器且維度相容 → 投到模型基底(映射回參考分佈);否則維持當資料夾即時 SVD。
        _mp = (st.session_state.get("anomaly_model") or {}).get("projection")
        _ref_bg = None
        if _mp is not None and emb.shape[1] == np.asarray(_mp["mean"]).shape[0]:
            from anomaly_project import transform_new
            coords = transform_new(_mp, emb)
            _ref_bg = np.asarray(_mp["ref_coords"], dtype=float)
        else:
            c = emb - emb.mean(axis=0, keepdims=True)
            try:
                _u, _s, _vt = np.linalg.svd(c, full_matrices=False)
                coords = (_u[:, :2] * _s[:2])
            except np.linalg.LinAlgError:
                coords = c[:, :2]
        _band = st.session_state.get(f"anomaly_heat_filter_{context}", (smin, smax))  # 依 context 隔離,免②滲①
        _lo, _hi = (_band if isinstance(_band, (tuple, list)) else (_band, smax))
        _cls = st.session_state.get(f"anomaly_class_filter_{context}") or []
        _passes = [(_lo <= scores[i] <= _hi)
                   and (not _cls or records[i].get("label", "") in _cls)
                   for i in range(len(records))]
        _txt = [f"{Path(records[i]['image_path']).name}"
                f"<br>score={records[i]['score']:.3f}·{records[i]['verdict']}"
                for i in range(len(records))]
        _dim = [i for i in range(len(records)) if not _passes[i]]
        _hot = [i for i in range(len(records)) if _passes[i]]
        fig = go.Figure()
        if _ref_bg is not None and len(_ref_bg):
            fig.add_trace(go.Scattergl(
                x=_ref_bg[:, 0], y=_ref_bg[:, 1], mode="markers",
                marker=dict(size=5, color="rgba(140,140,140,0.28)"),
                hoverinfo="skip", showlegend=False, name="參考分佈"))
        if _dim:
            fig.add_trace(go.Scattergl(
                x=coords[_dim, 0], y=coords[_dim, 1], mode="markers",
                marker=dict(size=6, color="rgba(150,150,150,0.25)"),
                customdata=_dim, text=[_txt[i] for i in _dim],
                hovertemplate="%{text}<extra></extra>", showlegend=False))
        fig.add_trace(go.Scattergl(
            x=coords[_hot, 0], y=coords[_hot, 1], mode="markers",
            marker=dict(size=7, color=[scores[i] for i in _hot], colorscale="Turbo",
                        cmin=smin, cmax=smax, showscale=True,
                        colorbar=dict(title="異常")),
            customdata=_hot, text=[_txt[i] for i in _hot],
            hovertemplate="%{text}<extra></extra>", showlegend=False))
        fig.update_layout(height=440, margin=dict(l=0, r=0, t=10, b=0),
                          dragmode="lasso")
        _nonce = st.session_state.get("_anomaly_clear_nonce", 0)
        _skey = f"anomaly_scatter_{context}_{_nonce}"
        _tb1, _tb2 = st.columns([5, 1])
        _tb1.caption("💡 在散點上拖曳框選/套索離群點 → 下方可加購物車或標正常/瑕疵。")
        _clear_slot = _tb2.empty()
        ev = st.plotly_chart(fig, key=_skey, on_select="rerun",
                             selection_mode=("box", "lasso"))
        if ev and getattr(ev, "selection", None):
            sel_idx = selection_points_to_indices(ev.selection.get("points", []))
            sel_idx = [i for i in sel_idx if _passes[i]]
        _clear_slot.button(
            f"✕ 取消框選({len(sel_idx)})" if sel_idx else "✕ 取消框選",
            key=f"anomaly_clear_sel_{context}", use_container_width=True,
            disabled=not sel_idx, on_click=_anomaly_clear_sel, args=(_skey,))
        st.caption(f"散點圖:點越紅越可疑。框選/套索離群點。已框選 {len(sel_idx)} 個。")
        st.button("🛒 加入購物車(框選)", key=f"anomaly_cart_selected_{context}",
                  disabled=not sel_idx, use_container_width=True,
                  on_click=_anomaly_add_to_cart, args=(records, sel_idx))
        mc1, mc2 = st.columns(2)
        mc1.button("✅ 框選標為正常範例", key=f"anomaly_mark_normal_{context}",
                   disabled=not sel_idx, use_container_width=True,
                   on_click=_anomaly_mark, args=(records, sel_idx, "good", context),
                   help="把你確定正常的點標起來 → 建乾淨 few-shot bank,專抓細微瑕疵。")
        mc2.button("🔴 框選標為瑕疵範例", key=f"anomaly_mark_bad_{context}",
                   disabled=not sel_idx, use_container_width=True,
                   on_click=_anomaly_mark, args=(records, sel_idx, "bad", context),
                   help="把已知瑕疵標起來校準門檻(會算出 AUROC)。")
        if context == "apply":
            # ② 就地重評:用最新 confirmed 重算分數(不動模型);正式重建走①(設計拍板)。
            st.button("🔁 就地重評(套用上面標記,不動模型)", key="anomaly_reapply_btn",
                      type="primary", use_container_width=True,
                      on_click=_anomaly_trigger_rerun,
                      help="用最新標記在目標資料上重算分數(不重建模型);正式重建請回①。")
        if sel_idx:
            _ph = st.slider("預覽高度(px)", 200, 1000, 400, 40,
                            key=f"anomaly_preview_h_{context}",
                            help="拖動調整下方框選預覽區的高度(看大圖)。")
            st.caption(f"框選 {len(sel_idx)} 個 — 預覽(看清楚再標;最多 24):")
            with st.container(height=_ph, key=f"anomaly_sel_preview_{context}"):
                _pc = st.columns(6)
                for _j, _i in enumerate(sel_idx[:24]):
                    _r = records[_i]
                    with _pc[_j % 6]:
                        _im = safe_open_image(_r["image_path"])
                        if _im is None:
                            st.caption("⚠ 缺圖")
                        else:
                            st.image(crop_bbox(_im, *_r["bbox"], pad=0.1),
                                     use_container_width=True)
                            _t = (0.5 if smax <= smin
                                  else (float(_r["score"]) - smin) / (smax - smin))
                            _bg = _turbo_hex(_t)          # 底色=與上方散點同款分數色
                            _lab = _r.get("label") or _r["verdict"]
                            st.markdown(
                                f"<div style='background:{_bg};color:{_text_on(_bg)};"
                                f"text-align:center;font-size:0.82em;line-height:1.6;"
                                f"font-weight:600;border-radius:6px;padding:2px 6px;"
                                f"margin-top:3px'>"
                                f"{_lab}｜{_rec_fname(_r)}<br>{_r['score']:.2f}</div>",
                                unsafe_allow_html=True)


def _anomaly_render_inspector(result: dict, scores) -> None:
    """右欄:🔥 熱度/類別篩選 + 排序清單 + 點選看圖(LOO/物件級→裁切圖;patch bank→熱力圖)。"""
    from PIL import Image

    from anomaly_heatmap import render_heatmap
    from anomaly_score import score_object
    from interaction import crop_bbox, discover_yolo_objects
    from object_eval import _adaptive_pad_px
    from patch_features import embed_objects_patch
    records = result["records"]
    ranking = result["ranking"]
    smin, smax = float(scores.min()), float(scores.max())
    st.markdown("**🔥 最可疑物件(篩選 + 點選看圖)**")
    # ⚠ 防呆要用「round 後」的值比較:smin/smax 都很小(如建模+套用同資料,分數很接近)時,
    # 原始 smax>smin 可能過關但 round(.,3) 後兩者都 = 0.0 → slider(0.0,0.0) 爆 StreamlitAPIException。
    _rmin, _rmax = round(smin, 3), round(smax, 3)
    if _rmax > _rmin:
        # 鍵加 _apply 後綴(inspector 只在②):與 ① build-scatter 的 band 隔離,免②滲①(散點全變淡選不到)
        lo, hi = st.slider("篩選:異常分數範圍 [低–高]", _rmin, _rmax, (_rmin, _rmax),
                           key="anomaly_heat_filter_apply",
                           help="雙邊範圍:只看分數落在區間內的物件。"
                                "拉高『低』界=濾掉低分(正常);拉低『高』界=排除最極端、只看中低段。")
    else:
        lo, hi = smin, smax
    labels_present = sorted({records[i].get("label", "") for i in ranking
                             if records[i].get("label", "")})
    cls_sel: list[str] = []
    if len(labels_present) > 1:
        cls_sel = st.multiselect(
            "篩選:只看類別", labels_present, default=[],
            key="anomaly_class_filter_apply",
            help="只看選定類別的物件(留空=全部)。針對某一類找漏 / 挑該類的異常。")
    shown = [i for i in ranking
             if lo <= scores[i] <= hi
             and (not cls_sel or records[i].get("label", "") in cls_sel)]
    st.caption(f"符合 {len(shown)} / {len(records)} 個(由最可疑排到最不可疑)")
    if shown:
        _sig = (round(float(lo), 6), round(float(hi), 6), tuple(sorted(cls_sel)))
        if (st.session_state.get("_anomaly_filt_sig") != _sig
                or st.session_state.get("anomaly_inspect") not in shown):
            st.session_state["anomaly_inspect"] = shown[0]
        st.session_state["_anomaly_filt_sig"] = _sig
        pick = st.selectbox(
            "選一個看圖", shown, key="anomaly_inspect",
            format_func=lambda i: f"{scores[i]:.3f} · {Path(records[i]['image_path']).name}"
            f" #{records[i]['obj_index']} · {records[i]['label']}")
        r = records[pick]
        img = safe_open_image(r["image_path"])
        try:
            if img is None:
                raise OSError("來源影像損壞或格式錯誤")
            iw, ih = img.size
            crop = crop_bbox(img, *r["bbox"],
                             pad_px=_adaptive_pad_px(r["bbox"], iw, ih, 0.15))
            if result.get("bank") is not None and result.get("_image_paths"):
                meta = discover_yolo_objects([Path(p) for p in result["_image_paths"]],
                                             result.get("_class_names"))
                pf = embed_objects_patch(
                    [meta[pick]], model=result.get("_model", "dinov2_vits14"),
                    target_res=result.get("_target_res", 224),
                    cache_dir=Path(result["_cache"]))[0]
                _, pmap = score_object(pf["feats"], pf["grid"], result["bank"])
                st.image(render_heatmap(pmap, crop), use_container_width=True,
                         caption=f"{r['label']}｜{_rec_fname(r)}"
                                 f" · {r['score']:.3f} · 紅=最不像正常")
            else:
                st.image(crop, use_container_width=True,
                         caption=f"{r['label']}｜{_rec_fname(r)} · {r['score']:.3f}"
                         " ·(無監督模式;確認正常範例後可看熱力圖)")
        except (OSError, Image.DecompressionBombError, IndexError) as e:
            st.caption(f":gray[無法顯示:{e}]")


def _anomaly_ui() -> None:
    """🔧 瑕疵偵測(AnomalyDINO 風格)— 引導式 wizard:
    ① 建模 / 載入 → ② 套用偵測 → ③ 挑樣送人工標。sidebar 留共用設定。
    跨 tab 狀態一律落 session_state;每個 tab function 開頭前置守門。"""
    # 標題旁放一個 ❓ 當 manual:平常不佔版面,點開才看說明。
    _hc1, _hc2 = st.columns([0.8, 0.2], vertical_alignment="bottom")
    _hc1.subheader("🔧 瑕疵偵測(Anomaly Detection)")
    with _hc2.popover("❓ 說明", use_container_width=True):
        st.markdown("**多數良品當『正常參考』**,把偏離的物件挑出來。基於凍結 DINOv2 patch 特徵 + "
                    "最近鄰距離(AnomalyDINO, WACV 2025);**不需瑕疵樣本訓練、不寫你的資料集**。")
        st.markdown("**引導三步**:① 用訓練/參考資料夾**建模**(或載入)→ ② 對異常目標資料夾**套用偵測** → "
                    "③ 2×2 取樣矩陣**挑最值得送人工標**的物件加入購物車匯出。")
        st.markdown("**沒確認正常時 = 無監督**:leave-one-out 把少見/離群類別排前面;**確認正常後 = 對照模式**:"
                    "只跟你確認的正常比,專抓細微瑕疵。")

    # ② 就地重評:用最新 confirmed 重算分數(不重建模型)。在 tab 渲染前先消費旗標。
    if st.session_state.pop("_anomaly_reapply", False):
        _anomaly_apply_model()

    _anomaly_sidebar_settings()
    # ⚠ 用 segmented_control(存 session_state)而非 st.tabs:st.tabs 的 active tab 是純前端狀態、
    # 沒進 session_state,任何整頁 rerun(加資料夾 / 按鈕 / slider)都會把它彈回第一步①;
    # segmented_control 有 key → 所選步驟跨 rerun 保留,不再「每次互動都跳回 Step-1」。
    _STEPS = ["① 建模 / 載入模型", "② 套用偵測", "③ 挑樣送人工標"]
    st.session_state.setdefault("anomaly_step", _STEPS[0])
    _sel = st.segmented_control("步驟", _STEPS, key="anomaly_step",
                                label_visibility="collapsed")
    # 點當前步驟會 deselect→None;保持上次步驟,避免空白(不重寫 widget key,避開 Streamlit gotcha)
    _step = _sel or st.session_state.get("_anomaly_step_shown") or _STEPS[0]
    st.session_state["_anomaly_step_shown"] = _step
    st.divider()
    if _step == _STEPS[0]:
        _anomaly_tab_build()
    elif _step == _STEPS[1]:
        _anomaly_tab_apply()
    else:
        _anomaly_tab_sample()


def _nn_index_for(model_name: str):
    """Lazy, per-session NN index over the raw embeddings (cleared at Run)."""
    store = st.session_state.setdefault("viz_nn_index", {})
    if model_name not in store:
        store[model_name] = build_nn_index(
            st.session_state["viz_raw_embeddings"][model_name]
        )
    return store[model_name]


def _current_selection() -> list[int]:
    """Selection indices, valid only for the current data token.

    Read straight from session_state（不是 fragment 參數）— fragment-local
    reruns must see post-callback state, not the args captured at call time.
    """
    sel = st.session_state.get("viz_selection") or {}
    if sel.get("token") != st.session_state.get("viz_data_token"):
        return []
    return list(sel.get("indices", []))


def _thumb_or_none(path: Path) -> str | None:
    try:
        return str(make_thumbnail(path))
    except (OSError, Image.DecompressionBombError):  # 壞圖/超大圖→無縮圖
        return None


# ── right-panel renderers（兩欄佈局的右欄）────────────────────────────────

@st.dialog("🔍 放大檢視", width="large")
def _zoom_image_dialog(path: Path, show_boxes: bool, class_names, caption: str) -> None:
    """全螢幕大圖檢視（沿用『顯示標註框』的當前狀態）。"""
    try:
        src = (draw_yolo_boxes(path, yolo_label_path_for(path), class_names)
               if (show_boxes and class_names) else str(path))
        if src is None:  # draw_yolo_boxes 遇壞圖回 None → 退回原圖路徑（仍可能壞）
            src = str(path)
        st.image(src, use_container_width=True)
    except OSError as exc:
        st.warning(f"無法讀取影像：{exc}")
    st.caption(caption)


def _render_viewer_slot(records: list[dict], ctx_default: list[int]) -> None:
    """Fixed-height, always-present viewer slot.

    Clicking a card only swaps the slot's content — the grid below never
    moves (zero layout shift), and the YOLO toggle keeps a fixed key so its
    state survives across images.
    """
    with st.container(border=True, key="viz_image_viewer"):
        idx = st.session_state.get("viz_active_image")
        if idx is None or not (0 <= idx < len(records)):
            st.caption("檢視槽 — 點選下方任一縮圖，在此檢視大圖與標註框，並可逐張加入匯出清單。")
            return
        r = records[idx]
        p = Path(r["path"])
        # 物件級記錄同時帶 crop(path)與來源整張圖(image_path)→ 可切換檢視
        image_path = r.get("image_path")
        is_obj_rec = bool(image_path) and str(image_path) != str(p)
        class_names = st.session_state.get("viz_classes")
        if class_names:
            st.session_state.setdefault("viz_img_boxes", True)  # 預設顯示標註框
        show_boxes = bool(class_names) and st.session_state.get("viz_img_boxes", False)
        ctx = st.session_state.get("viz_viewer_ctx") or list(ctx_default) or [idx]
        pos = ctx.index(idx) if idx in ctx else 0
        h1, h2, h3, h4 = st.columns([5, 1, 1, 1])
        h1.markdown(f"**{r['label']}｜{_rec_fname(r)}**（{r['split']}）"
                    f"· {pos + 1}/{len(ctx)} · #{idx}")
        h2.button("◀", key="viz_img_prev", disabled=pos <= 0,
                  on_click=_set_active_image, args=(ctx[max(pos - 1, 0)],))
        h3.button("▶", key="viz_img_next", disabled=pos >= len(ctx) - 1,
                  on_click=_set_active_image, args=(ctx[min(pos + 1, len(ctx) - 1)],))
        h4.button("✕", key="viz_img_close", on_click=_set_active_image, args=(None,))

        # 物件 / 原圖 切換(僅物件級記錄);原圖會畫上該圖所有 YOLO 框,看物件在場景中的脈絡
        if is_obj_rec:
            view = st.radio("檢視", ["物件", "原圖"], horizontal=True,
                            key="viz_view_mode", label_visibility="collapsed",
                            help="物件＝裁切的 crop;原圖＝此物件來源的整張影像。")
        else:
            view = "物件"
        disp = Path(image_path) if (is_obj_rec and view == "原圖") else p

        # 圖片用滿整個寬度（更大、不再被窄欄壓小）；控制列移到圖片下方
        if not disp.exists():
            st.warning(f"找不到檔案：{disp}")
        else:
            try:
                src = (draw_yolo_boxes(disp, yolo_label_path_for(disp), class_names)
                       if show_boxes else str(disp))
                if src is None:  # 壞圖 → draw 回 None；改顯示「已略過」而非崩潰
                    st.warning("⚠️ 已略過 1 個無法讀取的檔案"
                               "（壞檔/格式錯誤，已跳過，不影響其餘）")
                else:
                    st.image(src, use_container_width=True)
            except OSError as exc:
                st.warning(f"無法讀取影像：{exc}")
        a, b, c = st.columns(3)
        if class_names:
            a.toggle("顯示標註框", key="viz_img_boxes")
        if disp.exists() and b.button("🔍 放大檢視", key="viz_slot_zoom",
                                      use_container_width=True):
            _zoom_image_dialog(disp, show_boxes, class_names,
                               f"{r['label']}｜{_rec_fname(r)}（{view}）（{r['split']}）· #{idx}")
        elist = st.session_state.get("viz_export_list", {})
        if str(p) in elist:
            c.button("✓ 移除清單", key="viz_slot_remove", use_container_width=True,
                     on_click=_remove_from_export, args=(str(p),))
        else:
            c.button("⬇ 加入清單", key="viz_slot_add", use_container_width=True,
                     on_click=_add_one, args=(records, idx))
        d, e = st.columns(2)
        d.button("🔎 以此找相似", key="viz_slot_similar", use_container_width=True,
                 on_click=_start_query, args=(idx,))
        with e:
            _send_to_labeling_ui(
                records, [idx], source="viewer", task=LH.TASK_RELABEL,
                label="📤 送 Labeling", key=f"viz_slot_to_lbl_{idx}",
                original_labels={idx: r.get("label", "")})
        man = st.session_state.get("viz_manifest", {}).get(str(p.resolve()))
        if man:  # 資料合約可追溯性
            with st.popover("📄 Manifest", use_container_width=True):
                st.caption(f"sha256：`{man.get('sha256', '—')}`")
                st.caption(f"phash：`{man.get('phash') or '—'}`")
                st.caption(f"大小：{man.get('size', 0):,} bytes · "
                           f"檔案時間：{man.get('captured_at', '—')}")
                refs = man.get("embedding_refs", {})
                if refs:
                    st.caption("embedding refs：" +
                               "、".join(f"{m} → 列 {r}" for m, r in refs.items()))


def _render_grid(records: list[dict], shown: list[int]) -> None:
    elist = st.session_state.get("viz_export_list", {})
    cmap = _viz_color_map(records)
    css_rules: list[str] = []
    with st.container(height=560, key="viz_grid"):
        if not shown:
            st.info("在左圖以點選、框選（box）或套索（lasso）圈出資料點，縮圖會立即顯示在這裡。")
            return
        cols = st.columns(4)
        for j, i in enumerate(shown):
            with cols[j % 4]:
                p = Path(records[i]["path"])
                thumb = _thumb_or_none(p)
                if thumb is not None:
                    st.image(thumb, use_container_width=True)
                else:
                    st.warning("⚠ 檔案遺失")
                mark = "✓ " if str(p) in elist else ""
                label = records[i].get("label") or f"#{i}"
                st.button(f"{mark}{label}｜{_rec_fname(records[i])}", key=f"viz_card_{i}",
                          use_container_width=True, help=f"#{i}",
                          on_click=_set_active_image, args=(i, list(shown)))
                col = cmap.get(records[i].get("label", ""), "#9aa0a6")
                css_rules.append(
                    f".st-key-viz_card_{i} button{{background:{col}!important;"
                    f"color:{_text_on(col)}!important;border-color:{col}!important;}}")
    if css_rules:  # tint each caption box to its class colour (matches the dots)
        st.markdown("<style>" + "".join(css_rules) + "</style>",
                    unsafe_allow_html=True)


def _render_select_view(
    records: list[dict], coords: np.ndarray, model_name: str,
    selected_split: str, scatter_key: str,
) -> None:
    sel_indices = _current_selection()
    a1, a2, a3, a4 = st.columns([1.6, 1.2, 1.5, 1])
    a1.button("⬇ 批次加入清單", key="viz_add_btn", use_container_width=True,
              disabled=not sel_indices, on_click=_batch_add, args=(records, sel_indices))
    focus = st.session_state.get("viz_active_image")
    sim_target = focus if focus is not None else (sel_indices[0] if sel_indices else None)
    a2.button("🔎 找相似", key="viz_similar_btn", use_container_width=True,
              disabled=sim_target is None, on_click=_start_query, args=(sim_target,))
    # 注意：label_visibility="collapsed" 會連 help 問號一起藏掉（實測抓到），
    # 排序說明改掛在永遠可見的狀態行上。
    sort = a3.selectbox("排序", ["空間順序", "離群度", "標籤分歧", "檔名"],
                        key="viz_grid_sort", label_visibility="collapsed")
    a4.button("✕ 清除", key="viz_clear_btn", use_container_width=True,
              disabled=not sel_indices, on_click=_clear_selection, args=(scatter_key,))

    if sel_indices:
        _dis = st.session_state.get("viz_label_disagreement", {}).get(model_name)
        _out = st.session_state.get("viz_outlier_scores", {}).get(model_name)
        _scores = {}
        for i in sel_indices:
            s = {}
            if _dis is not None:
                s["disagreement"] = round(float(_dis[i]), 4)
            if _out is not None:
                s["outlier"] = round(float(_out[i]), 4)
            if s:
                _scores[str(i)] = s
        # 依排序語境推導任務：離群度排序＝多半是「這張對不對」(verify)，否則重標
        _sel_task = LH.TASK_VERIFY if sort == "離群度" else LH.TASK_RELABEL
        # 把框選這批送到 Labeling 標註（灰帶覆核工具已隱藏，移除其 handoff）
        _send_to_labeling_ui(
            records, sel_indices, source="selection", task=_sel_task,
            label="📤 送到 Labeling 標註", key="viz_sel_to_labeling",
            original_labels={i: records[i].get("label", "") for i in sel_indices},
            payload={"scores": _scores} if _scores else None,
            help="把框選的這批（分歧／離群／重複皆可）送到 Labeling 逐張標／改類別；"
                 "分歧／離群分數隨件帶過。標完在 Labeling 端「匯出 / 回傳」匯出即完成，不用回 LV。")

    outlier = st.session_state.get("viz_outlier_scores", {}).get(model_name)
    disagreement = st.session_state.get("viz_label_disagreement", {}).get(model_name)
    if sel_indices:
        if selected_split == "All":
            disp = list(sel_indices)
        else:
            disp = [i for i in sel_indices if records[i]["split"] == selected_split]
        if sort == "離群度" and outlier is not None:
            order = sorted(disp, key=lambda i: -float(outlier[i]))
        elif sort == "標籤分歧" and disagreement is not None:
            order = sorted(disp, key=lambda i: -float(disagreement[i]))
        elif sort == "檔名":
            order = sorted(disp, key=lambda i: _rec_fname(records[i]))
        else:
            order = spatial_order(coords, disp)
        limit = min(st.session_state.get("viz_grid_limit", _GRID_BATCH), _GRID_CAP)
        shown = order[:limit]
        status = f"已選取 {len(sel_indices)} 個點"
        if selected_split != "All" and len(disp) != len(sel_indices):
            status += f"（目前 split 顯示 {len(disp)}）"
        status += f" · 已載入 {len(shown)}/{len(disp)} · 排序：{sort}"
        if len(disp) > _GRID_CAP:
            status += f" · ⚠ 僅瀏覽前 {_GRID_CAP} 張，全部 {len(disp)} 筆仍可批次加入清單"
    else:
        # 未選取的預設視圖：排名分數前 N（離群度，或 F5 的標籤分歧）
        if sort == "標籤分歧" and disagreement is not None:
            scores, crit, note = disagreement, "標籤分歧", "僅鄰居標籤統計，非品質判定"
        else:
            scores, crit, note = outlier, "離群度", "僅幾何距離，非品質判定"
        if scores is not None and len(records) >= 3:
            order = [int(i) for i in np.argsort(scores)[::-1][:_DEFAULT_TOP_OUTLIERS]]
            shown = order
            status = f"未選取 · 預設顯示{crit}前 {len(shown)} 張（{note}）"
        else:
            order, shown = [], []
            status = "未選取"
    with st.container(key="viz_status_line"):
        st.caption(
            status,
            help="排序說明：空間順序＝縮圖位置模仿散點圖；離群度＝到鄰居的平均距離，"
                 "越高越「孤立」；標籤分歧＝k 近鄰中標籤不同的比例，越高越值得複查標註"
                 "（後兩者僅供排序參考，非品質判定）。卡片文字是「類別｜檔名」"
                 "（物件級顯示來源原圖檔名），順序即目前排序。",
        )

    _render_viewer_slot(records, shown)
    _render_grid(records, shown)
    if sel_indices and len(order) > len(shown):
        st.button(f"載入更多（+{_GRID_BATCH}）", key="viz_more_btn",
                  use_container_width=True, on_click=_load_more,
                  disabled=len(shown) >= _GRID_CAP)
    with st.expander("詳細表格"):
        df = pd.DataFrame([
            {"index": i, "filename": _rec_fname(records[i]),
             "label": records[i]["label"], "split": records[i]["split"]}
            for i in shown
        ])
        st.dataframe(df, key="viz_sel_table", hide_index=True,
                     use_container_width=True, height=220)

    _render_curation_log(records, sel_indices)


def _render_curation_log(records: list[dict], sel_indices: list[int]) -> None:
    """策展時間維度（重評 #1）：把選取＋判斷理由落盤，跨重啟保存、可回看、
    可一鍵重選、可匯出交接。回答『回到上週的選取＋為什麼這樣選』。"""
    with st.expander("📝 策展日誌（記錄選取＋理由，跨重啟保存）"):
        reason = st.text_input("這次選取的理由（為什麼選這批）", key="viz_cur_reason",
                               placeholder="例：疑似標錯的灰帶，待覆核")
        st.button(f"記錄目前選取（{len(sel_indices)} 張）＋理由", key="viz_cur_log",
                  use_container_width=True,
                  disabled=not sel_indices or not reason.strip(),
                  on_click=_append_curation_entry, args=(records, sel_indices, reason))
        entries = _load_curation_log()
        if not entries:
            st.caption("尚無紀錄。選取後填理由按上方按鈕即可留痕。")
            return
        st.caption(f"近期紀錄（共 {len(entries)} 筆，最新在上）：")
        with st.container(height=200):
            for k, e in enumerate(entries[:30]):
                shas = [it.get("sha256", "") for it in e.get("items", [])]
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{e.get('ts','')}** · {e.get('n',0)} 張 · "
                            f"{e.get('reason','')}")
                c2.button("↩ 重選", key=f"viz_cur_re_{k}", use_container_width=True,
                          on_click=_curation_reselect, args=(records, shas))
        st.download_button("⬇ 匯出策展日誌 CSV", data=curation_log_csv(entries),
                           file_name="curation_log.csv", mime="text/csv",
                           key="viz_cur_csv", use_container_width=True)


def _pivot_to_image_query(idx: int) -> None:
    """A text-search hit becomes the root of an image query chain."""
    st.session_state["viz_query_chain"] = [idx]
    st.session_state["viz_text_query"] = ""


def _encode_text_cached(model_name: str, q_text: str) -> np.ndarray:
    """Encode a text query; memoize the encoder and the last query vector."""
    memo = st.session_state.get("_viz_text_vec")
    if memo and memo[0] == model_name and memo[1] == q_text:
        return memo[2]
    enc_store = st.session_state.setdefault("viz_text_encoders", {})
    if model_name not in enc_store:
        with st.spinner("載入文字編碼器…"):
            enc_store[model_name] = load_text_encoder(model_name)
    vec = enc_store[model_name](q_text)
    st.session_state["_viz_text_vec"] = (model_name, q_text, vec)
    return vec


def _render_text_search(records: list[dict], model_name: str,
                        raw: np.ndarray, q_text: str) -> None:
    """F7: text-to-image search over the shared Chinese-CLIP space."""
    vec = _encode_text_cached(model_name, q_text)
    k = st.number_input(
        "k（回傳數量）", min_value=1, max_value=max(1, len(records)),
        value=min(9, len(records)), key="viz_text_k",
    )
    idxs, dists = find_similar_to_vector(raw, vec, k=int(k),
                                         nn_index=_nn_index_for(model_name))
    if not idxs:
        st.info("沒有可比對的影像。")
        return
    st.caption(f"「{q_text}」的前 {len(idxs)} 名 — cosine 距離越小越相符。")
    with st.container(height=560):
        cols = st.columns(3)
        for j, (i, d) in enumerate(zip(idxs, dists)):
            with cols[j % 3]:
                p = Path(records[i]["path"])
                thumb = _thumb_or_none(p)
                if thumb is not None:
                    st.image(thumb, use_container_width=True,
                             caption=f"{records[i].get('label') or '?'}"
                                     f"｜{_rec_fname(records[i])} · d={d:.4f}")
                else:
                    st.warning(f"缺檔：{_rec_fname(records[i])}")
                st.button("↻ 以此圖續查", key=f"viz_textpivot_{i}",
                          use_container_width=True,
                          on_click=_pivot_to_image_query, args=(i,))
                st.button("⬇ 加入清單", key=f"viz_textadd_{i}",
                          use_container_width=True,
                          on_click=_add_one, args=(records, i, "search", float(d)))
    st.download_button(
        "⬇ 匯出此結果 CSV", data=records_to_csv(records, idxs),
        file_name="text_search.csv", mime="text/csv", key="viz_export_textsearch",
    )


def _render_similar_view(records: list[dict], model_name: str) -> None:
    _render_viewer_slot(records, [])
    with st.container(key="viz_similar_panel"):
        chain = st.session_state.get("viz_query_chain", [])
        raw = st.session_state.get("viz_raw_embeddings", {}).get(model_name)
        # F7 以文搜圖 — 只在文字塔與影像塔同空間的模型（chinese-clip）開放
        if supports_text_query(model_name) and raw is not None and len(raw) > 0:
            q_text = st.text_input(
                "以文搜圖（繁／簡中文，英文次之）", key="viz_text_query",
                placeholder="例：斑馬、長頸鹿、夜間反光、zebra",
                help="繁體查詢會自動正規化為簡體再編碼（Chinese-CLIP 訓練語料以簡體為主）；英文可用但精度次之。",
            )
            if q_text.strip():
                _render_text_search(records, model_name, raw, q_text.strip())
                return
        elif raw is not None and any(supports_text_query(m) for m in
                                     st.session_state.get("viz_raw_embeddings", {})):
            st.caption("ℹ 將上方 Model 切換為 chinese-clip 模型即可使用以文搜圖。")
        if not chain:
            st.info("在「選取」面板選定影像後按「🔎 找相似」，或在檢視槽按「以此找相似」。")
            return
        if raw is None or len(raw) < 2:
            st.info("此模型沒有可用的原始特徵向量，無法找相似。")
            return
        q = chain[-1]
        if not (0 <= q < len(records)):
            st.info("查詢影像已失效，請重新選擇。")
            return
        chip_cols = st.columns(max(len(chain), 1))
        for ci, qi in enumerate(chain):
            chip_cols[ci].button(f"#{qi}", key=f"viz_chip_{ci}", use_container_width=True,
                                 on_click=_truncate_chain, args=(ci,))
        st.caption(f"查詢影像：{_rec_fname(records[q])} — cosine 距離越小越相似。")
        k = st.number_input(
            "k（回傳數量）", min_value=1, max_value=max(1, len(records) - 1),
            value=min(9, len(records) - 1), key="viz_similar_k",
        )
        idxs, dists = find_similar_indices(raw, q, k=int(k),
                                           nn_index=_nn_index_for(model_name))
        if not idxs:
            st.info("沒有其他影像可比對。")
            return
        with st.container(height=560):
            cols = st.columns(3)
            for j, (i, d) in enumerate(zip(idxs, dists)):
                with cols[j % 3]:
                    p = Path(records[i]["path"])
                    thumb = _thumb_or_none(p)
                    if thumb is not None:
                        st.image(thumb, use_container_width=True,
                                 caption=f"{records[i].get('label') or '?'}"
                                         f"｜{_rec_fname(records[i])} · d={d:.4f}")
                    else:
                        st.warning(f"缺檔：{_rec_fname(records[i])}")
                    st.button("↻ 以此為查詢", key=f"viz_requery_{i}", use_container_width=True,
                              on_click=_chain_query, args=(i,))
                    st.button("⬇ 加入清單", key=f"viz_simadd_{i}", use_container_width=True,
                              on_click=_add_one, args=(records, i, "similar", float(d)))
        st.download_button(
            "⬇ 匯出此相似群 CSV", data=records_to_csv(records, [q] + idxs),
            file_name="similar_group.csv", mime="text/csv", key="viz_export_similar",
        )
        st.button("✕ 關閉相似查詢", key="viz_similar_close", on_click=_close_query)


def _scan_duplicates(records: list[dict], model_name: str) -> None:
    """F4: scan for duplicate / leakage candidate pairs (runs in callback)."""
    method = st.session_state.get("viz_dup_method", "phash（嚴格重複）")
    cross = bool(st.session_state.get("viz_dup_cross", False))
    splits = [r["split"] for r in records]
    if method.startswith("phash"):
        thr = int(st.session_state.get("viz_dup_thr_ph", 4))
        pairs = find_duplicate_pairs_phash(
            st.session_state.get("viz_phashes", []),
            max_hamming=thr, splits=splits, cross_split_only=cross)
        kind = "phash"
    else:
        thr = float(st.session_state.get("viz_dup_thr_emb", 0.05))
        raw = st.session_state.get("viz_raw_embeddings", {}).get(model_name)
        pairs = [] if raw is None else find_duplicate_pairs_embedding(
            raw, max_distance=thr, splits=splits, cross_split_only=cross)
        kind = "embedding"
    st.session_state["viz_dup_result"] = {
        "token": st.session_state.get("viz_data_token"),
        "model": model_name, "kind": kind, "cross": cross, "pairs": pairs,
    }
    st.toast(f"掃描完成：{len(pairs)} 對候選", icon="🔍")
    _log_usage("dup_scan", kind=kind, cross=cross, n_pairs=len(pairs))


def _render_dup_view(records: list[dict], model_name: str) -> None:
    """F4: duplicate / train-val leakage review — pairs side by side,
    each reviewable in the viewer slot and exportable to the list."""
    _render_viewer_slot(records, [])
    with st.container(key="viz_dup_panel"):
        st.caption(
            "以 phash（位元近似）或 embedding（語意近似）找出疑似重複的影像對；"
            "勾「僅跨 split」即 train/val 洩漏候選。僅供人工複核，非自動判決。"
        )
        # 兩列排版：同列混用「有標籤」與「無標籤」控件會高度錯位
        r1a, r1b, r1c = st.columns([2.1, 1.2, 1])
        method = r1a.selectbox("方法", ["phash（嚴格重複）", "embedding（語意重複）"],
                               key="viz_dup_method", label_visibility="collapsed")
        r1b.toggle("僅跨 split", key="viz_dup_cross",
                   help="只列出跨資料夾的重複＝train/val 洩漏候選")
        r1c.button("🔍 掃描", key="viz_dup_scan", use_container_width=True,
                   on_click=_scan_duplicates, args=(records, model_name))
        if method.startswith("phash"):
            st.number_input(
                "指紋差異門檻（漢明距離 ≤）", min_value=0, max_value=16, value=4,
                key="viz_dup_thr_ph",
                help="每張圖會壓成 64 位元的感知指紋（dHash）；此值＝允許兩張圖指紋"
                     "不同的位元數。0＝幾乎位元級相同；預設 4 抓近似重複；"
                     "越大越寬鬆、誤報越多。",
            )
        else:
            st.number_input(
                "語意距離門檻（cosine ≤）", min_value=0.0, max_value=0.5, value=0.05,
                step=0.01, format="%.2f", key="viz_dup_thr_emb",
                help="兩張圖 embedding 的 cosine 距離上限；預設 0.05 抓改尺寸／"
                     "重新壓縮後內容仍相同的圖；越大越寬鬆。",
            )

        res = st.session_state.get("viz_dup_result")
        if (not res or res.get("token") != st.session_state.get("viz_data_token")
                or (res.get("kind") == "embedding" and res.get("model") != model_name)):
            st.info("設定方法與門檻後按「🔍 掃描」。")
            return
        pairs = res["pairs"]
        if not pairs:
            st.success("在目前條件下未發現重複候選。")
            return
        shown_pairs = pairs[:50]
        st.caption(
            f"找到 {len(pairs)} 對候選 · 顯示前 {len(shown_pairs)} 對（依距離排序）。"
            "右側 = 載入順序較後者。"
        )
        st.button("⬇ 將全部右側加入匯出清單", key="viz_dup_add_all",
                  use_container_width=True,
                  on_click=_batch_add, args=(records, [j for _, j, _ in pairs], "duplicate"))
        # 成對送 Labeling 覆核（keep/drop）；配對關係+距離隨 payload 帶過、不被攤平
        _dup_all = sorted({i for i, _j, _ in pairs} | {j for _i, j, _ in pairs})
        _send_to_labeling_ui(
            records, _dup_all, source="duplicate", task=LH.TASK_VERIFY,
            label="📤 送重複對到 Labeling 覆核", key="viz_dup_to_lbl",
            original_labels={k: records[k].get("label", "") for k in _dup_all},
            payload={"pairs": [[int(i), int(j), (d if isinstance(d, int) else float(d))]
                               for i, j, d in pairs]},
            help="把疑似重複／跨 split 洩漏『對』成對送到 Labeling 覆核（保留/丟棄）；"
                 "配對關係與距離隨件帶過，去重決策在 Labeling 端完成（不自動刪），不用回 LV。")
        with st.container(height=480, key="viz_dup_list"):
            for row, (i, j, d) in enumerate(shown_pairs):
                dd = f"{d}" if isinstance(d, int) else f"{d:.4f}"
                cc = st.columns([2, 2, 1.5])
                for side, idx_ in ((0, i), (1, j)):
                    with cc[side]:
                        p = Path(records[idx_]["path"])
                        thumb = _thumb_or_none(p)
                        if thumb is not None:
                            st.image(thumb, use_container_width=True)
                        else:
                            st.warning("⚠ 檔案遺失")
                        st.button(f"{records[idx_].get('label') or '?'}"
                                  f"｜{_rec_fname(records[idx_])}（{records[idx_]['split']}）",
                                  key=f"viz_dup_{row}_{side}", use_container_width=True,
                                  on_click=_set_active_image, args=(idx_, [i, j]))
                with cc[2]:
                    st.caption(f"d={dd}")
                    st.button("⬇ 右側入清單", key=f"viz_dup_addr_{row}",
                              use_container_width=True,
                              on_click=_add_one, args=(records, j, "duplicate"))


def _run_sampling(model_name: str, n: int, seed_from_list: bool) -> None:
    """F6: pick the N most diverse unlabeled images to label next."""
    raw = st.session_state.get("viz_raw_embeddings", {}).get(model_name)
    if raw is None:
        return
    records = st.session_state["viz_records"]
    seeds = None
    if seed_from_list:
        elist = st.session_state.get("viz_export_list", {})
        by_path = {str(Path(r["path"]).resolve()): i for i, r in enumerate(records)}
        seeds = [by_path[k] for k in elist if k in by_path] or None
    picks = farthest_point_sampling(raw, int(n), seed_indices=seeds)
    st.session_state["viz_sampling"] = {"token": st.session_state.get("viz_data_token"),
                                        "picks": picks, "seeded": bool(seeds)}
    st.toast(f"已選出 {len(picks)} 張多樣性樣本", icon="🎯")
    _log_usage("sampling", n=len(picks), seeded=bool(seeds))


def _render_sampling_view(records: list[dict], model_name: str) -> None:
    """F6 多樣性選樣 / 主動學習：farthest-point 從資料集挑最該標的 N 張。"""
    _render_viewer_slot(records, [])
    with st.container(key="viz_sampling_panel"):
        st.caption("用 farthest-point（k-center greedy）挑出彼此最不像、"
                   "最該優先標註的一批樣本。勾「避開匯出清單」＝把清單當已覆蓋，"
                   "只挑沒被涵蓋到的新樣本（主動學習）。")
        c1, c2 = st.columns([1, 2])
        n = c1.number_input("選幾張", min_value=1, max_value=min(200, len(records)),
                            value=min(12, len(records)), key="viz_sampling_n")
        seed = c2.toggle("避開匯出清單（主動學習）", key="viz_sampling_seed",
                         help="把目前匯出清單視為『已標/已覆蓋』，只挑離它最遠的新樣本。")
        st.button("🎯 挑選多樣性樣本", key="viz_sampling_btn", use_container_width=True,
                  on_click=_run_sampling, args=(model_name, n, seed))

        res = st.session_state.get("viz_sampling")
        if not res or res.get("token") != st.session_state.get("viz_data_token"):
            st.info("設定數量後按「挑選」。結果按多樣性排序（越前越獨特）。")
            return
        picks = res["picks"]
        if not picks:
            st.info("沒有可挑選的樣本（清單可能已覆蓋全部）。")
            return
        st.caption(f"選出 {len(picks)} 張"
                   + ("（已避開匯出清單）" if res["seeded"] else "")
                   + " · 多樣性排序，越前越該優先標")
        st.button("⬇ 全部加入匯出清單", key="viz_sampling_addall",
                  use_container_width=True,
                  on_click=_batch_add, args=(records, picks, "sampling"))
        _send_to_labeling_ui(
            records, list(picks), source="diversity", task=LH.TASK_FRESH,
            label="📤 送待標清單到 Labeling 標註", key="viz_sampling_to_lbl",
            help="主動學習：把最多樣的未標樣本送到 Labeling 從頭標註（fresh）；"
                 "標完在 Labeling 端「匯出 / 回傳」匯出即為新標籤，不用回 LV。")
        with st.container(height=520):
            cols = st.columns(3)
            for j, i in enumerate(picks):
                with cols[j % 3]:
                    p = Path(records[i]["path"])
                    thumb = _thumb_or_none(p)
                    if thumb:
                        st.image(thumb, use_container_width=True,
                                 caption=f"{records[i].get('label') or '?'}"
                                         f"｜{_rec_fname(records[i])}")
                    else:
                        st.warning("⚠ 缺檔")
                    st.button("看圖", key=f"viz_samp_view_{i}", use_container_width=True,
                              on_click=_set_active_image, args=(i, list(picks)))
        st.download_button(
            "⬇ 匯出待標清單 CSV", data=records_to_csv(records, picks),
            file_name="active_learning_picks.csv", mime="text/csv",
            key="viz_sampling_csv", use_container_width=True)


def _scores_for(path: Path) -> dict:
    """Walk up from an image to find a scores.csv (≤3 levels), cached per
    directory. Returns {filename: (score, threshold)} — empty if none."""
    cache = st.session_state.setdefault("viz_scores_cache", {})
    p = Path(path).parent
    for _ in range(4):
        if str(p) in cache:
            return cache[str(p)]
        csv_path = p / "scores.csv"
        if csv_path.exists():
            loaded = load_scores_csv(csv_path)
            cache[str(p)] = loaded
            return loaded
        if p.parent == p:
            break
        p = p.parent
    return {}


def _record_image_type(r: dict, type_from: str | None) -> str | None:
    """The record's image type, matching the calibration's grouping rule so
    the right per-type threshold is selected. ``class``/``label`` use the
    record's class label (object-mode label == the box class); ``parent``
    uses the image's parent directory name."""
    if type_from in ("class", "label"):
        return str(r.get("label") or "") or None
    if type_from == "parent":
        p = r.get("image_path") or r.get("path")
        return Path(p).parent.name if p else None
    return None


def _resolve_record_roi(r: dict):
    """(image_path, pixel_bbox (x0,y0,x1,y1), source_note) for the record's
    reported defect location, or None. Object-mode records carry a normalized
    YOLO bbox; a whole-image record falls back to its single YOLO label box."""
    from interaction import bbox_to_pixels, parse_yolo_boxes, yolo_label_path_for
    try:
        bbox_norm = r.get("bbox")  # object mode: (cx, cy, w, h) normalized
        if bbox_norm is not None:
            img_path = r.get("image_path") or r["path"]
            im = safe_open_image(img_path, mode=None)  # 只取 .size;壞檔 None → 略過
            if im is None:
                return None
            iw, ih = im.size
            return img_path, bbox_to_pixels(*bbox_norm, iw, ih), "YOLO 物件框"
        boxes = parse_yolo_boxes(yolo_label_path_for(Path(r["path"])))
        if len(boxes) == 1:  # one reported defect location → use it as the ROI
            im = safe_open_image(r["path"], mode=None)  # 只取 .size;壞檔 None → 略過
            if im is None:
                return None
            iw, ih = im.size
            _, cx, cy, w, h = boxes[0]
            return r["path"], bbox_to_pixels(cx, cy, w, h, iw, ih), "YOLO 標註框（單框）"
    except (OSError, ValueError, KeyError):
        return None
    return None


def _signal_level_for_record(r: dict) -> tuple[str, str, str | None]:
    """桶① physical-detectability for the active record, if a defect ROI is
    known. Returns (signal_level, source_note, image_type). No locatable ROI →
    UNKNOWN (never guess a 桶① verdict). ``image_type`` selects the matching
    per-type calibration."""
    from signal_strength import (
        SIGNAL_UNKNOWN, load_calibration, signal_level_for_image)
    itype = _record_image_type(r, (load_calibration() or {}).get("type_from"))
    roi = _resolve_record_roi(r)
    if roi is None:
        return SIGNAL_UNKNOWN, "無缺陷框，無法定位 ROI", itype
    img_path, px, src = roi
    return signal_level_for_image(img_path, px, image_type=itype), src, itype


def _render_health_card(records: list[dict], model_name: str) -> None:
    """Escape report card (defect-mechanisms §4): embedding-side diagnostics
    + decision-tree attribution for the active image. Reads an optional
    scores.csv to add the N4 gate; degrades honestly without it."""
    _render_viewer_slot(records, [])
    with st.container(key="viz_card_panel"):
        idx = st.session_state.get("viz_active_image")
        if idx is None or not (0 <= idx < len(records)):
            st.info("在「選取」面板點一張縮圖、或在散點上選一個點，"
                    "再回此頁產生該影像的體檢卡。")
            return
        raw = st.session_state.get("viz_raw_embeddings", {}).get(model_name)
        if raw is None or len(raw) < 2:
            st.info("此模型沒有可用特徵向量。"); return
        r = records[idx]
        p = Path(r["path"])
        labels = [rec["label"] for rec in records]
        outlier = st.session_state.get("viz_outlier_scores", {}).get(model_name)

        # 自動半徑 r = 訓練集 kNN 距離 P75（與 N2 閘一致；可調）
        radius = st.session_state.get("_viz_card_radius")
        if radius is None:
            from interaction import build_nn_index as _bni  # noqa
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(metric="cosine", n_neighbors=2).fit(raw)
            dist, _ = nn.kneighbors(raw)
            radius = float(np.percentile(dist[:, 1], 75))
            st.session_state["_viz_card_radius"] = radius

        density = neighbor_hit_density(raw, idx, radius)          # S2 覆蓋度
        label_entropy = neighbor_label_entropy(raw, labels, idx, k=20)

        scores = _scores_for(p)
        sc = scores.get(p.name)
        score_v = sc[0] if sc else None
        thr_v = sc[1] if sc else None

        # S3 模型不確定度：有 scores.csv 用分數 margin（真模型不確定度），
        # 否則退回鄰域標籤熵當代理（誠實標示）。
        if score_v is not None and thr_v is not None:
            margin = abs(score_v - thr_v) / (abs(thr_v) + 1e-9)
            s3 = float(max(0.0, 1.0 - min(margin, 1.0)))
            s3_src = "模型分數 margin"
        else:
            s3 = float(label_entropy)
            s3_src = "鄰域標籤熵（代理，未提供 scores.csv）"

        st.subheader(f"🩺 體檢卡 · {p.name}")
        st.caption(f"{r['label']}（{r['split']}）· #{idx}")

        # S1 概念歧義度：人類一致性（來自組考卷 / gauge R&R），由使用者提供
        s1_default = float(st.session_state.get("quiz_last_consistency", 0.9))
        s1 = st.slider("S1 此概念的人類一致性（來自組考卷 / gauge study）",
                       0.0, 1.0, s1_default, 0.01, key="viz_card_s1",
                       help="量『人』——專家對這類樣本判定有多一致。"
                            "低於門檻才會判 H2 定義歧義（補資料不收斂）。"
                            "跑過組考卷會自動帶入其自我一致率。")

        signal_level, sig_src, sig_type = _signal_level_for_record(r)
        diag = diagnose_root_cause(s1, density, s3, signal_level=signal_level)
        _sig_emoji = {"明顯": "🟢", "疑似": "🟡", "確無": "🔴",
                      "未知": "⚪"}.get(signal_level, "⚪")
        st.markdown(f"### 根因：{diag['cause']}")
        st.markdown(f"**補資料有效性：{diag['add_data']}**")
        st.caption(diag["action"])
        if diag.get("caveat"):
            st.warning(diag["caveat"], icon="⚠️")
        from signal_strength import effective_thresholds
        _thr = effective_thresholds(image_type=sig_type)
        if _thr["calibrated"]:
            _scope = (f"此類型「{sig_type}」" if _thr["source"] == "config:type"
                      else "全域")
            _cal = (f"門檻已校準（{_scope}：none={_thr['snr_none']}, "
                    f"obvious={_thr['snr_obvious']}"
                    + (f"，n={_thr['n']}" if _thr.get("n") else "") + "）")
        else:
            _cal = "門檻為預設、未校準——跑 calibrate_signal_gate.py --write 即自動套用"
        st.caption(f":gray[S0 訊號強度（桶①閘）：{_sig_emoji} {signal_level}"
                   f"（{sig_src}）。『確無』直接判 H0 物理天花板（補資料無效）；"
                   f"未量到 ROI 時為『未知』，不臆測。{_cal}]")

        # 三正交訊號
        c1, c2, c3 = st.columns(3)
        c1.metric("S1 人類一致性", f"{s1 * 100:.0f}%",
                  "歧義" if diag["s1_low"] else "清楚",
                  help="量人：低＝專家也喬不定＝定義問題。")
        c2.metric("S2 命中密度", density, "稀疏" if diag["s2_sparse"] else "密集",
                  help=f"量資料：半徑 {radius:.3f} 內訓練集相似鄰居數。")
        c3.metric("S3 模型不確定度", f"{s3:.2f}", "猶豫" if diag["s3_high"] else "篤定",
                  help=f"量模型：來源＝{s3_src}。")
        st.caption(f":gray[S3 來源：{s3_src}。三訊號需彼此獨立——"
                   "缺 S1（沒跑組考卷）時 H2 無法觸發，請補測人類一致性。]")

        # kNN 鄰居縮圖牆
        st.markdown("**最近鄰（它長得像誰）**")
        nbr_idx, nbr_d = find_similar_indices(raw, idx, k=6,
                                              nn_index=_nn_index_for(model_name))
        with st.container(height=520):
            cols = st.columns(3)
            for j, (ni, nd) in enumerate(zip(nbr_idx, nbr_d)):
                with cols[j % 3]:
                    thumb = _thumb_or_none(Path(records[ni]["path"]))
                    if thumb:
                        st.image(thumb, use_container_width=True,
                                 caption=f"{records[ni]['label']}"
                                         f"｜{_rec_fname(records[ni])} · d={nd:.3f}")
        # 匯出
        report = _health_card_report(p, r, idx, diag, s1, density, s3, s3_src,
                                     score_v, thr_v, radius, nbr_idx, nbr_d, records,
                                     _thr, sig_type)
        st.download_button("⬇ 匯出體檢卡 HTML", data=report,
                           file_name=f"healthcard_{p.stem}.html",
                           mime="text/html", key="viz_card_export",
                           use_container_width=True)


def _health_card_report(p, r, idx, diag, s1, density, s3, s3_src,
                        score_v, thr_v, radius, nbr_idx, nbr_d, records,
                        thr_info=None, sig_type=None) -> str:
    rows = "".join(
        f"<tr><td>#{ni}</td><td>{records[ni]['label']}</td>"
        f"<td>{records[ni]['split']}</td><td>{nd:.4f}</td></tr>"
        for ni, nd in zip(nbr_idx, nbr_d))
    score_line = (f"模型分數 {score_v:.3f}" +
                  (f"，閾值 {thr_v:.3f}" if thr_v is not None else "（無閾值）")
                  ) if score_v is not None else "未提供 scores.csv"
    caveat_html = (f'<p style="color:#b4232a">⚠️ {diag["caveat"]}</p>'
                   if diag.get("caveat") else "")
    thr_info = thr_info or {"snr_none": "?", "snr_obvious": "?",
                            "source": "default", "calibrated": False}
    _scope = ({"config:type": f"此類型「{sig_type}」", "config:global": "全域"}
              .get(thr_info["source"], "預設未校準"))
    gate_html = (f'<br>桶①閘門檻：<b>{_scope}</b>'
                 f'（none={thr_info["snr_none"]}, obvious={thr_info["snr_obvious"]}'
                 f'，來源 {thr_info["source"]}）')
    return f"""<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8">
<title>體檢卡 {p.name}</title><style>
body{{font-family:"Noto Sans TC",sans-serif;max-width:720px;margin:24px auto;color:#1a2433}}
h1{{font-size:20px}} .k{{color:#5a6b80}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #e3e8ef;padding:6px 10px;font-size:14px}}
.attr{{background:#eef2f7;border-radius:8px;padding:12px 16px;margin:12px 0}}</style></head><body>
<h1>🩺 Escape 體檢卡 · {p.name}</h1>
<p class="k">{r['label']}（{r['split']}）· #{idx} · {p}</p>
<div class="attr"><b>根因：{diag['cause']}</b><br>補資料有效性：<b>{diag['add_data']}</b>
<br>S0 訊號強度（桶①閘）：<b>{diag.get('signal_level') or '未量測'}</b>{gate_html}
<p>{diag['action']}</p>{caveat_html}</div>
<p>三正交訊號 —
S1 人類一致性 <b>{s1*100:.0f}%</b>（{'歧義' if diag['s1_low'] else '清楚'}）　·
S2 命中密度 <b>{density}</b>（半徑 {radius:.3f}，{'稀疏' if diag['s2_sparse'] else '密集'}）　·
S3 模型不確定度 <b>{s3:.2f}</b>（{'猶豫' if diag['s3_high'] else '篤定'}，來源 {s3_src}）</p>
<p>{score_line}</p>
<h3>最近鄰</h3><table><tr><th>#</th><th>label</th><th>split</th><th>cosine 距離</th></tr>
{rows}</table>
<p class="k">由 LV 產生。歸因僅含 N2/N3（與可選 N4）embedding 訊號；N0 品質、N1 定義仲裁需人工。</p>
</body></html>"""


# ── 桶①佔比 view + in-app recalibration ─────────────────────────────────

def _bucket1_record_metrics(records: list[dict]):
    """{image_type: [snr, …]} over records with a resolvable ROI, grouped by
    the active calibration's type_from (default 'label')."""
    from signal_strength import load_calibration, roi_background_metrics
    type_from = (load_calibration() or {}).get("type_from") or "label"
    groups: dict[str, list[float]] = {}
    for r in records:
        roi = _resolve_record_roi(r)
        if roi is None:
            continue
        img_path, px, _src = roi
        im = safe_open_image(img_path, mode="L")  # 灰階流程;壞檔 None → 跳過該筆
        if im is None:
            continue
        g = np.asarray(im, dtype=np.float64) / 255.0
        m = roi_background_metrics(g, px)
        if not m:
            continue
        groups.setdefault(_record_image_type(r, type_from) or "_global",
                          []).append(m["snr"])
    return groups, type_from


def _do_recalibrate_bucket1(records: list[dict]) -> None:
    """Recalibrate the 桶① gate from the loaded data and persist it (no CLI)."""
    from signal_strength import calibrate_thresholds, save_calibration
    with st.spinner("量測各框訊號強度、校正中…"):
        groups, type_from = _bucket1_record_metrics(records)
    if not groups:
        st.warning("目前資料沒有可定位 ROI 的標註框（需偵測/物件模式）。")
        return
    all_snr = np.concatenate([np.asarray(v) for v in groups.values()])
    g_cal = calibrate_thresholds(all_snr)
    if not g_cal["calibrated"]:
        st.warning(f"可量測框數 {g_cal['n']} < 10，樣本太少不足以校準。")
        return
    per_type = {}
    for t, v in groups.items():
        c = calibrate_thresholds(np.asarray(v))
        if c["calibrated"]:
            per_type[t] = {"snr_none": c["snr_none"],
                           "snr_obvious": c["snr_obvious"], "n": c["n"]}
    folders = st.session_state.get("viz_folder_list", [])
    save_calibration({**g_cal, "type_from": type_from, "per_type": per_type,
                      "dataset": " ; ".join(map(str, folders)) or "in-app"})
    st.session_state.pop("_bucket1_rows", None)  # recompute with new thresholds
    st.success(f"已從目前 {int(g_cal['n'])} 個框校準桶①門檻："
               f"全域 none={g_cal['snr_none']}/obvious={g_cal['snr_obvious']}、"
               f"逐類型 {len(per_type)} 組。重新計算佔比即套用。")


def _bucket1_proportions(records: list[dict]):
    """Per image type: 明顯/疑似/確無/未知 counts from the 桶① gate.
    Returns (rows, idx_map) where idx_map[(image_type, level)] = [record index],
    so the table can drill down to the actual images behind each count."""
    from collections import Counter
    from signal_strength import (
        SIGNAL_NONE, SIGNAL_OBVIOUS, SIGNAL_SUSPECT, SIGNAL_UNKNOWN,
        load_calibration)
    type_from = (load_calibration() or {}).get("type_from")
    per: dict[str, Counter] = {}
    idx_map: dict[tuple, list[int]] = {}
    for i, r in enumerate(records):
        level, _src, _it = _signal_level_for_record(r)
        it = _record_image_type(r, type_from) or "（未分型）"
        per.setdefault(it, Counter())[level] += 1
        idx_map.setdefault((it, level), []).append(i)
    rows = []
    for t, c in per.items():
        meas = c[SIGNAL_NONE] + c[SIGNAL_SUSPECT] + c[SIGNAL_OBVIOUS]
        rows.append({
            "影像類型": t, "可量測框": meas,
            "🔴確無(桶①)": c[SIGNAL_NONE], "🟡疑似": c[SIGNAL_SUSPECT],
            "🟢明顯": c[SIGNAL_OBVIOUS], "⚪未知": c[SIGNAL_UNKNOWN],
            "桶①佔比": f"{100 * c[SIGNAL_NONE] / meas:.0f}%" if meas else "—",
        })
    rows.sort(key=lambda x: -x["可量測框"])
    return rows, idx_map


def _render_bucket1_view(records: list[dict], model_name: str) -> None:
    """桶① physical-detectability proportions per image type — serves the
    『四桶比例』open question (the 桶① cell only)."""
    from signal_strength import effective_thresholds, load_calibration
    st.subheader("🧱 桶①佔比（物理可偵測性）")
    st.caption("對每筆有缺陷框的影像量訊號強度，統計各影像類型多少落在桶①"
               "（🔴確無＝訊號沒進資料、補資料無效）。直接回答『四桶比例』的桶①格；"
               "桶②③④需配合覆蓋/盲測重標，這裡只量得到桶①下界。")
    cfg = load_calibration()
    thr = effective_thresholds()
    if thr["calibrated"]:
        n_pt = len((cfg or {}).get("per_type") or {})
        st.caption(f":green[門檻已校準（{thr['source']}）：全域 none={thr['snr_none']}, "
                   f"obvious={thr['snr_obvious']}"
                   + (f"，{n_pt} 個逐類型" if n_pt else "") + "]")
    else:
        st.caption(":orange[門檻為預設、未校準——先按『重新校正』用目前資料校準，"
                   "桶①佔比才可信。]")
    b1, b2 = st.columns(2)
    if b1.button("🎯 從目前資料重新校正門檻", key="bucket1_recal",
                 use_container_width=True):
        _do_recalibrate_bucket1(records)
    if b2.button("📊 計算桶①佔比", key="bucket1_calc", type="primary",
                 use_container_width=True):
        with st.spinner("量測各框訊號強度中…"):
            _rows, _imap = _bucket1_proportions(records)
            st.session_state["_bucket1_rows"] = _rows
            st.session_state["_bucket1_idx"] = _imap
    rows = st.session_state.get("_bucket1_rows")
    if not rows:
        st.info("按「計算桶①佔比」開始；需偵測/物件模式（每筆有缺陷框）。")
        return
    tot_meas = sum(x["可量測框"] for x in rows)
    tot_none = sum(x["🔴確無(桶①)"] for x in rows)
    if tot_meas:
        st.metric("整體『看不見』佔比（物理天花板）", f"{100 * tot_none / tot_meas:.0f}%",
                  help="🔴看不見 ÷ 可量測。越高＝越多目標在影像裡根本看不到，"
                       "補同類資料救不了。門檻未校準前偏估。")
    st.caption(":gray[名詞：每個框量它的「訊號強度」=在影像裡看不看得見。"
               "桶①＝物理看不見（四桶之一：①物理 ②覆蓋 ③邊界 ④標註）。滑到表頭看說明。]")
    _bucket1_colcfg = {
        "影像類型": st.column_config.TextColumn(
            "影像類型", help="YOLO 類別／影像分型；未設定則為（未分型）。"),
        "可量測框": st.column_config.NumberColumn(
            "可量測框", help="量得到訊號強度的框數（＝分母，排除未量測）。"),
        "🔴確無(桶①)": st.column_config.NumberColumn(
            "🔴看不見(桶①)",
            help="訊號太弱、物理上看不見 → 補同類資料無效。"
                 "動作：改善拍攝（打光／對比／解析度）重拍，或排除。"),
        "🟡疑似": st.column_config.NumberColumn(
            "🟡邊界", help="訊號在臨界，可能看得到也可能不行。"
                          "動作：送體檢卡／加入清單。"),
        "🟢明顯": st.column_config.NumberColumn(
            "🟢清楚", help="訊號夠強、清楚可見 → 正常，不需處理。"),
        "⚪未知": st.column_config.NumberColumn(
            "⚪未量測", help="沒有可定位的框／ROI，量不到 → 不列入分母。"),
        "桶①佔比": st.column_config.TextColumn(
            "看不見佔比", help="🔴看不見 ÷ 可量測。越高＝資料品質天花板越低。"),
    }
    st.dataframe(rows, use_container_width=True, hide_index=True,
                 column_config=_bucket1_colcfg)
    st.caption(":gray[『看不見』是桶①下界，校準前偏估；可疑（邊界）框送『體檢卡』或"
               "『組考卷』再分清桶②(覆蓋)/桶③(邊界帶)。下方可下鑽看/選實際影像。]")

    # ── 下鑽:選某一格 → 看/選取那批實際影像 → 走右上既有出口 ──
    idx_map = st.session_state.get("_bucket1_idx", {})
    if idx_map:
        from signal_strength import (SIGNAL_NONE, SIGNAL_OBVIOUS,
                                     SIGNAL_SUSPECT, SIGNAL_UNKNOWN)
        _LV = {"🔴確無": SIGNAL_NONE, "🟡疑似": SIGNAL_SUSPECT,
               "🟢明顯": SIGNAL_OBVIOUS, "⚪未知": SIGNAL_UNKNOWN}
        st.markdown("**🔎 下鑽:看 / 選取某一格的實際影像**")
        d1, d2 = st.columns(2)
        sel_type = d1.selectbox("影像類型", [x["影像類型"] for x in rows],
                                key="bucket1_dd_type")
        sel_lv = d2.selectbox("訊號等級", list(_LV), key="bucket1_dd_lv")
        picks = idx_map.get((sel_type, _LV[sel_lv]), [])
        st.caption(f"{sel_type} · {sel_lv}：{len(picks)} 個")
        if picks:
            if st.button(f"🎯 選取這 {len(picks)} 個"
                         "（再用右上『送體檢卡／加入清單』）",
                         key="bucket1_pick", use_container_width=True):
                st.session_state["viz_selection"] = {
                    "token": st.session_state.get("viz_data_token"), "indices": picks}
                st.session_state["viz_grid_limit"] = _GRID_BATCH
                st.session_state["viz_active_image"] = None
                st.toast(f"已選取 {len(picks)} 個；切到『選取』分頁或用右上出口處理")
                st.rerun()
            _cap = 60
            st.caption(f"預覽（捲動看；最多顯示 {_cap}，共 {len(picks)}）。"
                       "要完整檢視/處理請按上方『選取這…個』。")
            with st.container(height=520):
                cols = st.columns(3)
                for j, i in enumerate(picks[:_cap]):
                    with cols[j % 3]:
                        th = _thumb_or_none(Path(records[i]["path"]))
                        if th is not None:
                            st.image(th, use_container_width=True)
                        st.caption(f"{records[i].get('label') or '?'}"
                                   f"｜{_rec_fname(records[i])}")


_SOURCE_LABEL = {
    "manual": "手選", "search": "以文搜圖", "similar": "以圖搜圖",
    "duplicate": "重複/洩漏", "sampling": "多樣性選樣", "outlier": "離群",
    "disagreement": "標籤分歧", "sparse": "稀疏盲區", "gap_filler": "補洞候選",
    "gray": "灰帶", "quiz": "考卷爭議", "quiz_disputed": "考卷低一致",
}


def _cart_set_reason(snaps: list[dict], reason: str) -> None:
    """為購物車裡（篩選後）這批寫上理由——snaps 是 elist 的活字典值，直接改即生效。"""
    for s in snaps:
        s["reason"] = reason
    st.toast(f"已為 {len(snaps)} 張加註理由", icon="📝")
    _log_usage("cart_reason", n=len(snaps))


def _render_export_view() -> None:
    elist = st.session_state.get("viz_export_list", {})
    st.caption(f"策展購物車 — 共 {len(elist)} 張（跨工具累積、跨 Run 保留；"
               "下方可一鍵分流到組考卷／灰帶覆核或匯出，不寫回資料集）")
    if not elist:
        st.info("購物車是空的。任何看得到縮圖的地方（選取／覆蓋圖／灰帶／相似／"
                "重複／選樣）都能「加入清單」，會帶來源標籤累積到這裡。")
        return
    all_snaps = list(elist.values())

    from collections import Counter
    cnt = Counter((s.get("source") or "manual") for s in all_snaps)
    breakdown = " · ".join(f"{_SOURCE_LABEL.get(k, k)} {v}" for k, v in cnt.most_common())
    st.caption(f":gray[來源組成：{breakdown}]")
    fcol, _sp = st.columns([1.5, 2])
    src = fcol.selectbox(
        "來源篩選", ["全部"] + [k for k, _ in cnt.most_common()],
        format_func=lambda k: ("全部" if k == "全部"
                               else f"{_SOURCE_LABEL.get(k, k)}（{cnt.get(k, 0)}）"),
        key="cart_src_filter", label_visibility="collapsed")
    snapshots = _cart_snapshots(src)
    pseudo_records = [
        {"path": Path(s["path"]), "label": s.get("label", ""),
         "split": s.get("split", "")} for s in snapshots]

    with st.container(height=500, key="viz_export_grid"):
        cols = st.columns(4)
        for j, s in enumerate(snapshots):
            with cols[j % 4]:
                thumb = _thumb_or_none(Path(s["path"]))
                cap = _SOURCE_LABEL.get(s.get("source", "manual"), s.get("source", ""))
                if s.get("score") is not None:
                    cap += f"·{s['score']:.2f}"
                cap = f"{s.get('label') or '?'}｜{_rec_fname(s)} · {cap}"
                if thumb is not None:
                    st.image(thumb, use_container_width=True, caption=cap)
                else:
                    st.warning(f"⚠ 缺檔 · {cap}")
                st.button(f"移除 #{s['index']}", key=f"viz_unlist_{j}",
                          use_container_width=True,
                          on_click=_remove_from_export, args=(s["path"],))

    # 組考卷／灰帶覆核工具已隱藏 → 移除購物車對它們的 handoff（出考卷／送灰帶覆核）。
    # 跨工具：把整車送到 Labeling 工具實際標註（單向交棒，標完在 Labeling 端匯出）
    _send_to_labeling_ui(
        pseudo_records, range(len(pseudo_records)), source="cart",
        task=LH.TASK_RELABEL,
        label="📤 送整車到 Labeling 標註", key="cart_to_labeling",
        original_labels={i: s.get("label", "") for i, s in enumerate(snapshots)},
        help="把整車影像送到 Labeling 工具逐張標/改類別；標完在 Labeling 端"
             "「匯出 / 回傳」匯出即完成，不用回 LV。")

    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇ 匯出 CSV（含來源/分數/sha256）", data=snapshots_to_csv(snapshots),
        file_name="curation_cart.csv", mime="text/csv",
        key="viz_export_csv", use_container_width=True)
    d2.download_button(
        "⬇ 匯出 ZIP", data=zip_selected_images(pseudo_records,
                                              list(range(len(pseudo_records)))),
        file_name="curation_cart_images.zip", mime="application/zip",
        key="viz_export_zip", use_container_width=True)

    with st.expander("📝 為這批加註理由（寫進 CSV 的 reason 欄）"):
        reason = st.text_input("理由", key="cart_reason_text",
                               placeholder="例：疑似 donut 與貝果混淆的一批",
                               label_visibility="collapsed")
        st.button("套用到目前篩選的這批", key="cart_reason_btn",
                  disabled=not reason.strip(),
                  on_click=_cart_set_reason, args=(snapshots, reason))
    c1, c2 = st.columns(2)
    confirm = c1.checkbox("確認清空", key="viz_clear_list_confirm")
    c2.button("🗑 清空清單", key="viz_clear_list_btn", disabled=not confirm,
              on_click=_clear_export_list)

    st.divider()
    _render_send_confirmation()


@st.fragment
def _render_right_panel(
    records: list[dict],
    coords: np.ndarray,
    model_name: str,
    selected_split: str,
    scatter_key: str,
) -> None:
    """The pinned right column: panel switcher + viewer slot + grid /
    find-similar / export list.

    Runs as a fragment so card clicks, chain queries and list edits rerun
    only this column — the scatter is never rebuilt, which also protects
    its box/lasso selection state. Selection itself is read from
    session_state inside（見 _current_selection）.
    """
    # 購物車「分流」鈕在 fragment 內，其 callback 只重跑 fragment、切不了主工具——
    # 收到旗標時跳出 fragment 作 app 範圍 rerun，讓外層 segmented_control 換頁。
    if st.session_state.pop("_cart_app_rerun", False):
        st.rerun(scope="app")
    st.session_state.setdefault("viz_panel_view", "選取")
    view = st.segmented_control(
        "面板", ["選取", "相似", "重複", "選樣", "體檢卡", "桶①佔比", "匯出清單"],
        key="viz_panel_view", label_visibility="collapsed",
    ) or "選取"

    if view == "選取":
        _render_select_view(records, coords, model_name, selected_split, scatter_key)
    elif view == "相似":
        _render_similar_view(records, model_name)
    elif view == "重複":
        _render_dup_view(records, model_name)
    elif view == "選樣":
        _render_sampling_view(records, model_name)
    elif view == "體檢卡":
        _render_health_card(records, model_name)
    elif view == "桶①佔比":
        _render_bucket1_view(records, model_name)
    else:
        _render_export_view()


_MODE_CLEAR_KEYS = (
    "viz_records", "viz_embeddings", "viz_raw_embeddings",
    "viz_data_token", "viz_nn_index", "viz_classes",
    "viz_selection", "viz_active_image", "viz_viewer_ctx",
    "viz_query_chain", "viz_outlier_scores", "viz_grid_limit",
    "viz_export_list", "viz_panel_view", "viz_manifest",
    "viz_phashes", "viz_label_disagreement", "viz_dup_result",
    "_bucket1_rows",
)

_DEMO_DIR = Path(__file__).parent.parent / "demo" / "coco8"


def _sample_root() -> Path:
    """Writable cache dir for generated sample datasets (under gitignored output/)."""
    import os
    base = os.environ.get("CIM_LOG_DIR") or str(Path(__file__).parent.parent / "output")
    p = Path(base) / "_samples"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _demo_classifier_dir() -> str:
    """imagenette demo if provisioned, else a generated tiny classifier set."""
    d = Path(__file__).parent.parent / "demo" / "imagenette" / "train"
    if d.exists():
        return str(d)
    from sample_data import ensure_classifier_sample
    return str(ensure_classifier_sample(_sample_root() / "classify"))


def _demo_compare_dirs() -> tuple[str, str]:
    """imagenette cassette_player/chainsaw if provisioned, else two synthetic sets."""
    base = Path(__file__).parent.parent / "demo" / "imagenette" / "train"
    a, b = base / "cassette_player", base / "chainsaw"
    if a.exists() and b.exists():
        return str(a), str(b)
    from sample_data import ensure_compare_sample
    sa, sb = ensure_compare_sample(_sample_root() / "compare")
    return str(sa), str(sb)


def _demo_detection_dir() -> str:
    """coco8 demo if provisioned, else a generated tiny YOLO detection set."""
    d = Path(__file__).parent.parent / "demo" / "coco8" / "train"
    if (d / "images").exists():
        return str(d)
    from sample_data import ensure_detection_sample
    return str(ensure_detection_sample(_sample_root() / "detect"))


def _load_demo() -> None:
    """快速開始：一鍵載入範例並自動執行（detector 模式）。coco8 沒提供時用合成迷你偵測集。"""
    coco = _DEMO_DIR / "train"
    folders = ([str(coco), str(_DEMO_DIR / "val")] if coco.exists()
               else [_demo_detection_dir()])
    st.session_state["viz_mode"] = "Object Detector"
    st.session_state["_viz_mode_prev"] = "Object Detector"
    st.session_state["viz_folder_list"] = folders
    st.session_state["_viz_autorun"] = True
    _log_usage("demo_load")


def _restore_mode_snapshot() -> None:
    """切模式誤觸的後悔藥：還原上一個模式的全部結果（可逆優於攔截）。"""
    snap = st.session_state.pop("_viz_mode_snapshot", None)
    if not snap:
        return
    st.session_state["viz_mode"] = snap["mode"]
    st.session_state["_viz_mode_prev"] = snap["mode"]
    for k, v in snap["state"].items():
        if v is None:
            st.session_state.pop(k, None)
        else:
            st.session_state[k] = v
    st.session_state["viz_folder_list"] = snap["folders"]


def _render_quick_start() -> None:
    """冷啟動空狀態：三步教學卡 + 一鍵 demo（取代一行英文提示的死白）。"""
    st.markdown("##### 快速開始")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, st.container(border=True):
        st.markdown("**① 選資料**")
        st.caption("在左側貼上圖片資料夾路徑，或先用下方範例試跑。")
    with c2, st.container(border=True):
        st.markdown("**② 跑分析**")
        st.caption("按 ▶ Run，自動萃取特徵並降維成散點圖。")
    with c3, st.container(border=True):
        st.markdown("**③ 探索**")
        st.caption("在散點圖框選任一群點，右欄立即顯示對應縮圖。"
                   "進階功能（以文搜圖、重複掃描…）見右上「✨ 功能地圖」。")
    mid = st.columns([2, 1.6, 2])[1]
    mid.button("▶ 一鍵體驗（coco8 範例）", key="viz_demo_btn", type="primary",
               use_container_width=True, on_click=_load_demo)


def _supervised_projection(embeddings, labels, mkey: str, n_comps: int):
    """Label-guided projection — LDA (max class separation) or supervised UMAP.
    A VISUAL AID (it uses the labels to lay points out), not proof of real
    separability. Returns an array, or None when not computable."""
    y = np.asarray(labels)
    classes = np.unique(y)
    if len(classes) < 2 or len(y) < 4:
        return None
    yi = np.searchsorted(classes, y)
    if mkey == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        nc = min(n_comps, len(classes) - 1)
        if nc < 1:
            return None
        return LinearDiscriminantAnalysis(n_components=nc).fit_transform(embeddings, yi)
    if mkey == "sumap":
        n_neighbors = min(15, max(2, len(embeddings) - 1))
        return _umap().UMAP(n_components=n_comps, n_neighbors=n_neighbors,
                            random_state=42).fit_transform(embeddings, y=yi)
    return None


def _project_object_embeddings(embeddings: np.ndarray, method_pairs,
                               labels=None) -> dict:
    """PCA / t-SNE / UMAP (+ optional supervised LDA / 監督UMAP, needs labels)
    projections for object-level embeddings. Plain UMAP (the stable reference
    frame is image-level only)."""
    n_samples = len(embeddings)
    n_comps = min(3, max(1, n_samples - 2))

    def _pad2d(a: np.ndarray) -> np.ndarray:
        return a if a.shape[1] >= 2 else np.hstack([a, np.zeros((len(a), 1))])

    proj: dict[str, np.ndarray] = {}
    for mkey, _mlabel in method_pairs:
        if mkey in _SUPERVISED_METHODS:
            if labels is None:
                continue
            arr = _supervised_projection(embeddings, labels, mkey, n_comps)
            if arr is None:
                continue
            proj[mkey] = _pad2d(arr)
            continue
        if mkey != "pca" and n_samples < 4:
            continue
        if mkey == "pca":
            arr = PCA(n_components=n_comps, random_state=42).fit_transform(embeddings)
        elif mkey == "tsne":
            perplexity = min(30, max(1, n_samples - 1))
            arr = TSNE(n_components=n_comps, random_state=42,
                       perplexity=perplexity).fit_transform(embeddings)
        else:
            n_neighbors = min(15, max(2, n_samples - 1))
            arr = _umap().UMAP(n_components=n_comps, n_neighbors=n_neighbors,
                               random_state=42).fit_transform(embeddings)
        proj[mkey] = _pad2d(arr)
    if not proj:
        proj["pca"] = _pad2d(
            PCA(n_components=n_comps, random_state=42).fit_transform(embeddings))
    return proj


def _fmt_eta(sec: float) -> str:
    sec = int(max(0, sec))
    return f"{sec}s" if sec < 60 else f"{sec // 60}m{sec % 60:02d}s"


_OBJ_RES_OPTS = [168, 196, 224, 280, 336, 392, 448]  # DINOv2 patch=14 → 需 14 的倍數


def _manual_policy_controls(policy: dict, key_prefix: str) -> dict:
    """手動微調 pad／解析度／head 三個控制項（在自動找到的設定上再調）。
    回傳調整後的 policy（尚未套用——由呼叫端的按鈕寫入 session）。"""
    pol = dict(policy or {"pad": 0.12, "target_res": 224, "head": "cls"})
    c1, c2, c3 = st.columns(3)
    pad = c1.number_input("外擴 pad", 0.0, 0.5, float(pol.get("pad", 0.12)), 0.02,
                          key=f"{key_prefix}_pad",
                          help="裁框時多留幾成背景脈絡（0＝貼框）。")
    _res = int(pol.get("target_res", 224))
    _opts = sorted(set(_OBJ_RES_OPTS) | {_res})
    res = c2.selectbox("解析度", _opts, index=_opts.index(_res), key=f"{key_prefix}_res",
                       help="DINOv2 patch=14，需 14 的倍數；越大越細、越慢。")
    _heads = ["cls", "meanpool"]
    _h = pol.get("head", "cls")
    head = c3.selectbox("head", _heads, index=_heads.index(_h) if _h in _heads else 0,
                        key=f"{key_prefix}_head",
                        help="cls＝整體向量；meanpool＝patch 平均（室內多物件常較佳）。")
    return {"pad": float(pad), "target_res": int(res), "head": head}


def _active_object_policy() -> dict:
    import object_eval as oe
    return st.session_state.get("viz_object_policy", oe.DEFAULT_POLICY)


def _obj_keep(rec) -> bool:
    """渲染時的物件過濾判斷：confidence 不在所選『信心區間』內、或源像素短邊太小
    （DINO 對極小框的嵌入不可靠）→ 丟。門檻取自共用 session key
    （obj_conf_range 預設 (0,1)＝全留；obj_min_short_px 預設 0＝不過濾）。
    score=None（GT/無信心）不受信心區間影響。"""
    rng = st.session_state.get("obj_conf_range")
    sc = rec.get("score")
    if rng is not None and sc is not None:
        lo, hi = rng
        if not (float(lo) <= float(sc) <= float(hi)):
            return False
    msp = int(st.session_state.get("obj_min_short_px", 0) or 0)
    if msp > 0 and int(rec.get("short_px", 1 << 30)) < msp:
        return False
    return True


def _obj_filter_tag() -> str:
    """目前物件過濾狀態的字串指紋。**必須併進散點 widget 的 key**——這些 on_select
    散點『點集一變就得換 key 重掛』才會反映新點集（同 split 的道理）；只改 figure、key
    不變，畫面會停在舊點集，拖過濾滑桿像沒反應。"""
    return (f"{st.session_state.get('obj_conf_range')}"
            f"_{st.session_state.get('obj_min_short_px', 0)}")


def _obj_view_indices(records, base_indices=None) -> list[int]:
    """套 _obj_keep 後保留的 index 清單（不動 records/coords → 即時、可回復）。
    給 Visualize/Compare 用：它們對『index 子集』畫散點。"""
    idx = range(len(records)) if base_indices is None else base_indices
    return [i for i in idx if _obj_keep(records[i])]


def _obj_filter_view(records, emb, token):
    """對 (records, emb) 直接套 _obj_keep（給 coverage/objcov 用：它們要對 emb 算
    sparsity）。回 (records', emb', token')；門檻併入 token 供下游快取正確失效。"""
    keep = _obj_view_indices(records)
    if len(keep) == len(records):
        return records, emb, token
    e2 = np.asarray(emb)[keep] if len(emb) else emb
    msp = int(st.session_state.get("obj_min_short_px", 0) or 0)
    rng = st.session_state.get("obj_conf_range") or (0.0, 1.0)
    return [records[i] for i in keep], e2, f"{token}|f{msp}_{rng[0]}_{rng[1]}"


def _obj_has_conf(records) -> bool:
    return bool(records) and any(r.get("score") is not None for r in records)


def _obj_filter_controls(prefix: str, records, *, size_hint: float | None = None) -> None:
    """右側即時過濾控制：**信心區間**（雙把手；資料含 score 才顯示）＋源短邊。共寫
    session key obj_conf_range / obj_min_short_px（只渲染一個工具/次，故各用各自 widget
    key、回填共用值）。放散點旁 → 改動即時重濾、不需重 Run；調寬區間物件會回來
    （渲染時過濾、不丟資料）。"""
    if _obj_has_conf(records):
        cur = tuple(st.session_state.get("obj_conf_range", (0.0, 1.0)))
        rng = st.slider("信心區間（只顯示 conf 落在此區間的物件）", 0.0, 1.0, cur, 0.05,
                        key=f"obj_conf_range_{prefix}",
                        help="拖兩端 → 只顯示偵測 confidence 落在此區間的物件（即時）。"
                             "全範圍 0.00–1.00＝全顯示。")
        if tuple(rng) != cur:
            st.session_state["obj_conf_range"] = tuple(rng)
    # size 滑桿只在「物件級」（record 帶 short_px）時顯示，整圖模式不顯示（避免空操作）
    if records and "short_px" in records[0]:
        cur_px = int(st.session_state.get("obj_min_short_px", 0) or 0)
        hint = f"（短邊中位 ≈ {int(round(size_hint))}px）" if size_hint else ""
        px = st.slider("隱藏小物件（源短邊 < px）", 0, 64, cur_px, 4,
                       key=f"obj_min_short_px_{prefix}",
                       help="源像素短邊小於此值就隱藏（DINO 對極小框不可靠）。0＝不過濾。" + hint)
        if px != cur_px:
            st.session_state["obj_min_short_px"] = px


def _render_object_policy_ui(folders: list[Path]) -> None:
    """物件 embedding 設定:資料指紋→建議設定檔(套用需確認)+ 🔬 自動找最佳設定。"""
    import object_eval as oe

    active = _active_object_policy()
    with st.expander(f"⚙ 物件 embedding 設定 — 目前 {oe.policy_tag(active)}"):
        # 資料指紋 + 設定檔比對(每組資料夾只算一次,存 session)
        ftoken = repr(sorted(str(f) for f in folders))
        cache = st.session_state.get("_viz_fp_cache")
        if not cache or cache.get("token") != ftoken:
            imgs: list[Path] = []
            for f in folders:
                imgs += oe.list_images(f)
            cnames = oe.classes_for(folders[0]) if folders else None
            fp = oe.dataset_fingerprint(imgs, cnames) if imgs else None
            prof, sim = oe.match_profile(fp) if fp else (None, 0.0)
            cache = {"token": ftoken, "imgs": [str(p) for p in imgs],
                     "cnames": cnames, "fp": fp, "prof": prof, "sim": sim}
            st.session_state["_viz_fp_cache"] = cache
        fp, prof, sim = cache["fp"], cache["prof"], cache["sim"]
        if fp:
            st.caption(f"資料指紋:{fp['n_objects']} 物件 / {fp['n_classes']} 類 · "
                       f"短邊 px 中位 ≈ {round(fp['size_q'][1])}")
        st.caption("套用設定後請重新 ▶ Run 才會生效。")

        if prof and sim >= 0.6 and prof["policy"] != active:
            st.info(f"這份資料跟設定檔『{prof['name']}』很像(相似度 {sim:.2f})。")
            if st.button(f"套用『{prof['name']}』的設定", key="viz_apply_prof",
                         use_container_width=True):
                st.session_state["viz_object_policy"] = prof["policy"]
                st.toast(f"已套用『{prof['name']}』:{oe.policy_tag(prof['policy'])}")
                st.rerun()

        if st.button("🔬 自動找最佳設定", key="viz_autotune_btn",
                     use_container_width=True,
                     help="對目前資料各試幾種 pad／解析度／head,依同類 kNN 純度排名。需幾分鐘。"):
            imgs = [Path(p) for p in cache["imgs"]]
            if not imgs:
                st.warning("先選含 images/+labels/ 的資料夾。")
            else:
                with st.status("自動找最佳設定中…", expanded=True) as _s:
                    _b = st.progress(0.0, text="量測…")
                    import time
                    _at0 = time.time()

                    def _p(ci, n, pol, od, ot):
                        frac = (ci + od / max(ot, 1)) / max(n, 1)
                        el = time.time() - _at0
                        eta = (el / frac - el) if frac > 0.02 else 0.0
                        txt = f"[{ci + 1}/{n}] {oe.policy_tag(pol)} · 物件 {od}/{ot}"
                        if eta > 1:
                            txt += f" · 剩約 {_fmt_eta(eta)}"
                        _b.progress(min(frac, 1.0), text=txt)

                    rows, best, fp2 = oe.run_autotune(
                        imgs, cache["cnames"],
                        cache_dir=_dataset_cache_dir(folders[0], "object_crops") / "_autotune",
                        progress=_p)
                    _s.update(label="完成", state="complete", expanded=False)
                st.session_state["viz_autotune"] = {
                    "token": ftoken, "rows": rows, "best": best, "fp": fp2}
                st.rerun()

        at = st.session_state.get("viz_autotune")
        if at and at.get("token") == ftoken and at.get("rows"):
            import pandas as pd

            def _pct(x):
                return None if x is None else round(x * 100, 1)

            data = []
            for r in at["rows"]:
                b = r["buckets"]
                data.append({"設定": oe.policy_tag(r["policy"]),
                             "總體%": _pct(r["macro_purity"]),
                             "<32": _pct(b["<32"]["purity"]),
                             "32-96": _pct(b["32-96"]["purity"]),
                             "96-224": _pct(b["96-224"]["purity"]),
                             ">224": _pct(b[">224"]["purity"])})
            st.markdown("**量測結果 — macro kNN 純度 %(越高越好)**")
            st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
            best = at["best"]
            st.success(f"建議設定:{oe.policy_tag(best)}")
            if st.button("✅ 套用建議設定", key="viz_apply_best", use_container_width=True):
                st.session_state["viz_object_policy"] = best
                st.toast(f"已套用建議:{oe.policy_tag(best)}")
                st.rerun()
            nm = st.text_input("設定檔名稱", value=(folders[0].parent.name or "profile"),
                               key="viz_prof_name")
            if st.button("💾 存成設定檔(之後類似資料自動建議)", key="viz_save_prof",
                         use_container_width=True):
                oe.save_profile(nm, at["fp"], best, at["rows"][0])
                st.session_state.pop("_viz_fp_cache", None)
                st.toast(f"已存設定檔『{nm}』")
                st.rerun()

        st.divider()
        st.caption("✏️ 手動微調（在自動找到的設定上再調細部）")
        _man = _manual_policy_controls(active, "vizman")
        if st.button("套用手動設定", key="viz_apply_manual", use_container_width=True):
            st.session_state["viz_object_policy"] = _man
            st.toast(f"已套用：{oe.policy_tag(_man)}（請重新 ▶ Run）")
            st.rerun()


def _render_label_quality(records, raw, model, data_token) -> None:
    """🩹 標錯偵測:每類別混淆/分離度 + 疑似標錯清單(現標→建議標)。
    全部在 384 維算(neighbor_confusion),不受 2D/3D 投影擠壓影響。"""
    import object_eval as oe

    labels = [r.get("label", "") for r in records]
    if len(records) < 3 or len(set(labels)) < 2:
        st.caption("需要 ≥3 物件、≥2 類別才能分析。")
        return
    ckey = f"_viz_conf_{data_token[:8]}_{model}"
    res = st.session_state.get(ckey)
    if not res:
        res = oe.neighbor_confusion(raw, labels, k=10)
        st.session_state[ckey] = res
    classes, conf = res["classes"], np.asarray(res["confusion"])
    counts, per = res["counts"], res["per_object"]
    C = len(classes)

    tab_conf, tab_bad = st.tabs(["🔀 類別混淆 / 分離度", "🩹 疑似標錯"])

    with tab_conf:
        z = (conf * 100).round(1)
        hm = go.Figure(go.Heatmap(
            z=z, x=classes, y=classes, colorscale="Blues", zmin=0, zmax=100,
            colorbar=dict(title="鄰居%"),
            hovertemplate="標為 %{y}・鄰居是 %{x}:%{z}%<extra></extra>"))
        hm.update_layout(height=380, margin=dict(l=10, r=10, t=34, b=10),
                         title="每列＝該類物件的 k 近鄰類別分布(對角=分離度,離對角=混淆)",
                         yaxis=dict(autorange="reversed"))
        st.plotly_chart(hm, use_container_width=True, key=f"conf_hm_{model}")

        order = sorted(range(C), key=lambda i: conf[i, i])
        st.markdown("**最難分離的類別(對角越低＝越糊)**")
        for i in order[:5]:
            row = conf[i].copy()
            row[i] = -1.0
            j = int(np.argmax(row))
            st.write(f"- **{classes[i]}** 分離度 {conf[i, i] * 100:.0f}%"
                     f"（{int(counts[i])} 個）· 最常混 → {classes[j]} {row[j] * 100:.0f}%")

        pairs = [(classes[i], classes[j], conf[i, j])
                 for i in range(C) for j in range(C) if i != j and conf[i, j] > 0]
        pairs.sort(key=lambda t: -t[2])
        st.markdown("**最常混淆的方向(標為 A 的鄰居常是 B)**")
        for a, b, v in pairs[:8]:
            st.write(f"- {a} → {b}:{v * 100:.0f}%")

    with tab_bad:
        cand = sorted((p for p in per if p["mismatch"]),
                      key=lambda p: p["suggested_frac"], reverse=True)
        st.caption(f"共 {len(cand)} 個物件的鄰居多數是別類(現標→建議標),越上面越可疑。")
        if not cand:
            st.success("沒有明顯的疑似標錯 🎉")
            return
        topn = cand[:30]
        if st.button(f"🎯 選取這 {len(topn)} 個可疑物件(再用右上『加入清單』)",
                     key=f"sel_bad_{model}", use_container_width=True):
            st.session_state["viz_selection"] = {
                "token": data_token, "indices": [p["idx"] for p in topn]}
            st.session_state["viz_grid_limit"] = _GRID_BATCH
            st.session_state["viz_active_image"] = None
            st.toast(f"已選取 {len(topn)} 個可疑物件")
            st.rerun()
        cols = st.columns(4)
        for _k, p in enumerate(topn):
            with cols[_k % 4]:
                th = _thumb_or_none(Path(records[p["idx"]]["path"]))
                if th is not None:
                    st.image(th, use_container_width=True)
                st.caption(f"**{p['own']} → {p['suggested']}** "
                           f"({p['suggested_frac'] * 100:.0f}%)")


def _visualize_embeddings_ui() -> None:
    _load_ui_state_once()  # restore last session's sidebar choices (before widgets)
    with st.sidebar:
        st.markdown("**① 資料**")
        mode = st.radio(
            "模式", ["Object Detector", "Image Classifier"],
            key="viz_mode", horizontal=True,
            captions=["YOLO 格式（images/ + labels/）", "依類別分子資料夾；平鋪無分類也可（整夾當一類）"],
        )
        # 分析單位：物件級（YOLO，預設）vs 整張影像 — 僅偵測資料集可選物件級。
        # 兩種粒度是不同的點集（且物件級用保長寬比裁切策略），請分開 Run，勿混看。
        if mode == "Object Detector":
            granularity = st.radio(
                "分析單位", ["物件級（YOLO）", "整張影像"],
                key="viz_granularity", horizontal=True,
                help="物件級＝讀 labels/ 每個框→裁出物件各自算 embedding（一個物件一個點，"
                     "保長寬比裁切、不擠壓）；整張影像＝每張圖一個點。兩者分開 Run。",
            )
        else:
            granularity = "整張影像"
        is_object_level = (mode == "Object Detector"
                           and granularity == "物件級（YOLO）")
        # 切粒度→清舊結果，避免散點圖混到另一種粒度的點。但「切模式」也會連帶改變
        # granularity（Classifier 強制整張影像、Detector 預設物件級），那種情況不在這裡清，
        # 留給下方切模式區塊先快照再清（否則 viz_records 會被偷走、無法留一鍵復原）。
        _mode_changing = st.session_state.get("_viz_mode_prev") != mode
        if (not _mode_changing
                and st.session_state.get("_viz_gran_prev") not in (None, granularity)):
            st.session_state.pop("viz_records", None)
        st.session_state["_viz_gran_prev"] = granularity

        # 切換模式時清除舊結果（先快照，留一鍵復原）
        if st.session_state.get("_viz_mode_prev") != mode:
            prev = st.session_state.get("_viz_mode_prev")
            st.session_state["_viz_mode_prev"] = mode
            if prev is not None and st.session_state.get("viz_records") is not None:
                st.session_state["_viz_mode_snapshot"] = {
                    "mode": prev,
                    "state": {k: st.session_state.get(k) for k in _MODE_CLEAR_KEYS},
                    "folders": list(st.session_state.get("viz_folder_list", [])),
                }
            for k in _MODE_CLEAR_KEYS:
                st.session_state.pop(k, None)
            st.session_state["viz_folder_list"] = []

        snap = st.session_state.get("_viz_mode_snapshot")
        if snap:
            st.warning(f"已切換模式，{snap['mode']} 的結果已清空。")
            st.button("↩ 復原上個模式的結果", key="viz_mode_undo",
                      use_container_width=True, on_click=_restore_mode_snapshot)

        if "viz_folder_list" not in st.session_state:
            st.session_state["viz_folder_list"] = []

        _vfc1, _vfc2 = st.columns([5, 1], vertical_alignment="bottom")
        with _vfc1:
            _folder_add_input("viz_folder_list")  # headless/E2E-driveable 文字加入（📁 為原生對話框）
        with _vfc2:
            if st.button("📁", use_container_width=True, key="add_viz_folder",
                         help="開啟系統的『選擇資料夾』視窗；也可直接在左邊貼上路徑。"):
                _pick_folder_set("viz_folder_list")
                st.rerun()

        for i, folder in enumerate(st.session_state["viz_folder_list"]):
            c1, c2 = st.columns([5, 1])
            c1.text(Path(folder).name)
            c1.caption(folder)
            if c2.button("✕", key=f"rm_viz_{i}"):
                st.session_state["viz_folder_list"].pop(i)
                st.rerun()

        if not st.session_state["viz_folder_list"]:
            st.caption("尚未選擇資料夾（按上方 📁 或輸入路徑）")

        if mode == "Object Detector":
            # 類別來源屬進階設定（預設自動偵測 classes.txt），收進 expander（G4）
            with st.expander("類別來源（預設自動偵測 classes.txt）"):
                cc1, cc2 = st.columns([4, 1])
                classes_path = st.session_state.get("viz_classes_file", "")
                cc1.caption("classes.txt")
                cc1.text(Path(classes_path).name if classes_path else "（自動偵測或手動輸入）")
                if cc2.button("📄", key="browse_classes", use_container_width=True,
                              help="選擇 classes.txt"):
                    _pick_file("viz_classes_file", title="選擇 classes.txt",
                               filetypes=[("Text", "*.txt"), ("All files", "*.*")])
                    st.rerun()
                if classes_path:
                    if st.button("✕ 清除", key="clear_classes", use_container_width=True):
                        del st.session_state["viz_classes_file"]
                        st.rerun()

                class_input = st.text_input(
                    "Class names — 手動輸入（classes.txt 未選擇時使用）",
                    value="apple,banana,orange", key="viz_class_names",
                )
        else:
            class_input = ""

        if is_object_level and st.session_state.get("viz_folder_list"):
            _render_object_policy_ui(
                [Path(f) for f in st.session_state["viz_folder_list"]])

        st.markdown("**② 模型**")
        all_models = available_models()
        if not all_models:
            st.error("models/ 內找不到模型檔，請放入 .pth 模型後重啟。")
            return
        _def_models = [m for m in all_models if m == _DEFAULT_MODEL] or all_models
        selected_models = st.multiselect(
            "模型", all_models, default=_def_models, key="viz_models_sel",
            label_visibility="collapsed",
            help="每個模型各算一份 embedding；chinese-clip 同時解鎖「以文搜圖」。"
                 "預設用較快的 dinov2_vits14；要更準可再加 dinov2_vitb14。",
        )

        st.markdown("**③ 投影方法**")
        selected_method_labels = st.multiselect(
            "投影方法", list(_METHOD_KEY), default=_DEFAULT_METHODS,
            key="viz_methods", label_visibility="collapsed",
            help="只勾選需要的投影可大幅縮短計算時間。LDA／監督UMAP 用類別標籤排版，"
                 "讓 2D 依類別分明——僅視覺輔助，不代表真實可分性。",
        )
        if "UMAP" in selected_method_labels:
            st.toggle(
                "固定 UMAP 參考系", key="viz_umap_ref",
                help="首跑擬合並凍結 UMAP 空間（存於 embeddings_<model>/umap_ref.pkl）；"
                     "之後新增的影像以 transform 投入同一座標系，舊點完全不動，"
                     "跨 Run 佈局可比較。注意：transform 的擺位是近似值，"
                     "資料大幅改變後按下方「↻ 重建參考系」再 Run。",
            )
            if st.session_state.get("viz_umap_ref"):
                st.button("↻ 重建參考系（下次 Run 重新擬合）",
                          key="viz_umap_rebuild_btn", use_container_width=True,
                          help="丟掉現有參考系，下次 Run 以目前全部資料重新擬合並覆寫。"
                               "資料大幅改變後才需要。",
                          on_click=lambda: st.session_state.__setitem__(
                              "_viz_umap_rebuild", True))
                if st.session_state.get("_viz_umap_rebuild"):
                    st.caption(":orange[↻ 已標記：下次 Run 將重建 UMAP 參考系]")

        st.markdown("**④ 執行**")
        n_folders = len(st.session_state.get("viz_folder_list", []))
        missing = []
        if n_folders == 0:
            missing.append("①資料夾")
        if not selected_models:
            missing.append("②模型")
        if not selected_method_labels:
            missing.append("③投影")
        st.caption(f"{n_folders} 資料夾 · {len(selected_models)} 模型 · "
                   f"{len(selected_method_labels)} 投影")
        if missing:
            st.caption(f":red[⚠ 缺：{'、'.join(missing)}]")
        # 注意：資料夾欄是 text_area，值要「失焦」才提交——若用它 gate
        # disabled，填完直接點 Run 會點到還沒解鎖的按鈕（點擊被吞）。
        # 所以只有即時提交的 multiselect 缺件才真正鎖按鈕；缺資料夾僅紅字
        # 提示，按下去由 Run 內的驗證錯誤接手。
        hard_missing = not selected_models or not selected_method_labels
        run = st.button("▶ Run", use_container_width=True, key="run_viz",
                        disabled=hard_missing,
                        type="secondary" if missing else "primary")

        _save_ui_state()  # remember sidebar choices for the next launch

    if st.session_state.pop("_viz_autorun", False):
        run = True
    if run:
        folders = [Path(f) for f in st.session_state.get("viz_folder_list", [])]
        if not folders:
            st.error("請先選擇資料夾。")
            return
        missing_dirs = [str(p) for p in folders if not p.exists()]
        if missing_dirs:
            st.error(f"資料夾不存在：{', '.join(missing_dirs)}")
            return
        _collapse_sidebar()

        if not selected_models:
            st.error("Select at least one model.")
            return
        if not selected_method_labels:
            st.error("請至少勾選一種投影方法。")
            return
        method_pairs = [(_METHOD_KEY[lbl], lbl) for lbl in selected_method_labels]

        class_names: list[str] | None = None
        if mode == "Object Detector":
            # images/ 或 JPEGImages/ 子資料夾(YOLO/VOC 佈局),或帶任一支援的
            # 標註來源(COCO JSON/VOC XML/LabelMe/NDJSON,平鋪佈局)擇一即可
            missing = [str(f) for f in folders
                       if not (f / "images").exists()
                       and not (f / "JPEGImages").exists()
                       and not folder_has_annotations(f)]
            if missing:
                st.error("以下資料夾缺 images/(或 JPEGImages/)子資料夾、也找不到"
                         "支援的標註（COCO JSON／VOC XML／LabelMe／NDJSON）: "
                         f"{', '.join(missing)}")
                return

            # classes.txt 優先級：手動選擇 > 自動偵測 > 文字輸入
            classes_path = st.session_state.get("viz_classes_file", "")
            if classes_path and Path(classes_path).exists():
                lines = [ln.strip() for ln in safe_read_text(classes_path).splitlines() if ln.strip()]
                class_names = lines
                st.success(f"使用選定的 classes.txt（{len(class_names)} 個類別）：{_fmt_classes(class_names)}")
            else:
                detected = _classes_txt_nested(folders[0])
                if detected is not None:
                    class_names = detected
                    st.session_state["viz_class_msg"] = (
                        f"Auto-detected {len(class_names)} classes: {_fmt_classes(class_names)}")
                else:
                    class_names = [c.strip() for c in class_input.split(",") if c.strip()]
                    if not class_names:
                        st.error("Enter at least one class name.")
                        return
                    st.info(f"Using manually entered classes: {_fmt_classes(class_names)}")

            records = discover_images(folders, class_names)
        else:
            records = discover_images_classifier(folders, allow_flat=True)
            if records:
                detected_classes = sorted({r["label"] for r in records})
                st.session_state["viz_class_msg"] = (
                    f"自動偵測到 {len(detected_classes)} 個類別：{_fmt_classes(detected_classes)}")

        if not records:
            st.error("No images found in the specified folders.")
            return

        # 壞檔防呆:把無法解碼的影像（連同其 label）先剔除，並顯示統一略過提示
        records = _filter_records_readable(records, "viz")
        if not records:
            st.error("No images found in the specified folders.")
            return

        empty_folders = [f.name for f in folders if not any(r["split"] == f.name for r in records)]
        if empty_folders:
            st.warning(f"No images found in folder(s): {', '.join(empty_folders)}")

        # ── 物件級（YOLO）路徑：每個 bbox 一個點 ──────────────────────────
        # 與整圖路徑分流：裁切+特徵走升級後的 _crop_and_embed_objects（記憶體裁切、
        # 保長寬比、自適應 pad），存好 session_state 後 rerun 進共用的散點渲染。
        if is_object_level:
            base_token = repr(sorted(str(f) for f in folders))
            active_policy = _active_object_policy()
            obj_records = None
            o_raw: dict[str, np.ndarray] = {}
            o_proj: dict[str, dict[str, np.ndarray]] = {}
            import time
            _t0 = time.time()
            _nm = max(len(selected_models), 1)
            with st.status(f"計算中（物件級 · {active_policy['head']}）…",
                           expanded=True) as _status:
                _obar = st.progress(0.0, text="裁切＋特徵…")

                def _mk_cb(mi, mname):
                    # crop = 前半 (0–50%)、特徵 = 後半 (50–100%);跨模型再均分
                    def _cb(done, total, phase):
                        sub = done / max(total, 1)
                        mfrac = 0.5 * sub if phase == "crop" else 0.5 + 0.5 * sub
                        frac = (mi + mfrac) / _nm
                        el = time.time() - _t0
                        eta = (el / frac - el) if frac > 0.03 else 0.0
                        lbl = "裁切" if phase == "crop" else "特徵"
                        txt = f"[{mname}] {lbl} {done}/{total}"
                        if eta > 1:
                            txt += f" · 剩約 {_fmt_eta(eta)}"
                        _obar.progress(min(frac, 1.0), text=txt)
                    return _cb

                for _mi, model_name in enumerate(selected_models):
                    orecs, oemb, _ = _crop_and_embed_objects(
                        records, model_name, class_names,
                        active_policy.get("pad", 0.12),
                        base_token=base_token, session_key="_viz_obj",
                        spinner=f"[{model_name}] 裁切＋特徵", policy=active_policy,
                        progress_cb=_mk_cb(_mi, model_name))
                    if not orecs:
                        st.error("此偵測資料集的 labels/ 找不到任何 bbox，無法做物件級。"
                                 "請改『整張影像』或先補標註。"
                                 + _seg_format_hint([r["path"] for r in records]))
                        return
                    if obj_records is None:
                        obj_records = orecs
                    o_raw[model_name] = oemb
                    o_proj[model_name] = _project_object_embeddings(
                        oemb, method_pairs, labels=[r["label"] for r in orecs])
                    _obar.progress((_mi + 1) / max(len(selected_models), 1),
                                   text=f"[{model_name}] 完成")
                _status.update(label="完成", state="complete", expanded=False)
            records = obj_records

            outlier_scores: dict[str, np.ndarray] = {}
            label_disagreement: dict[str, np.ndarray] = {}
            if len(records) >= 3:
                k_out = min(5, len(records) - 1)
                rec_labels = [r["label"] for r in records]
                for m, raw in o_raw.items():
                    outlier_scores[m] = compute_outlier_scores(
                        raw, raw, k=k_out, candidates_in_reference=True)
                    label_disagreement[m] = compute_label_disagreement(
                        raw, rec_labels, k=k_out)

            data_token = uuid.uuid4().hex
            st.session_state["viz_records"] = records
            st.session_state["viz_embeddings"] = o_proj
            st.session_state["viz_raw_embeddings"] = o_raw
            st.session_state["viz_manifest"] = {}            # objects: no per-image manifest
            st.session_state["viz_phashes"] = [None] * len(records)
            st.session_state["viz_outlier_scores"] = outlier_scores
            st.session_state["viz_label_disagreement"] = label_disagreement
            st.session_state.pop("viz_dup_result", None)
            st.session_state["viz_data_token"] = data_token
            st.session_state["viz_nn_index"] = {}
            st.session_state["viz_classes"] = class_names
            st.session_state["viz_selection"] = {"token": data_token, "indices": []}
            st.session_state["viz_active_image"] = None
            st.session_state["viz_viewer_ctx"] = []
            st.session_state["viz_query_chain"] = []
            st.session_state["viz_grid_limit"] = _GRID_BATCH
            st.session_state.pop("_viz_mode_snapshot", None)
            st.session_state["viz_unit"] = "物件"
            st.toast(f"完成：{len(records)} 物件 × {len(selected_models)} 模型", icon="✅")
            st.rerun()

        def _thumb_lookup(p: Path) -> Path | None:
            try:
                t = thumbnail_path_for(p)
                return t if t.exists() else None
            except OSError:
                return None

        embeddings_per_model: dict[str, dict[str, np.ndarray]] = {}
        raw_per_model: dict[str, np.ndarray] = {}
        manifest_by_folder: dict[Path, dict[str, dict]] = {}
        _n_steps = 2 + len(selected_models) * (1 + len(method_pairs))
        _step = 0
        with st.status("計算中…", expanded=True) as _status:
            _bar = st.progress(0.0, text="縮圖快取…")

            def _thumb_cb(done: int, total: int) -> None:
                _bar.progress(min(done / max(total, 1) / _n_steps, 1.0),
                              text=f"縮圖快取 {done}/{total}")

            ensure_thumbnails([r["path"] for r in records], progress_cb=_thumb_cb)
            _step += 1
            _bar.progress(_step / _n_steps, text="縮圖快取完成")

            # Manifest（F1 資料合約）：增量更新，未變更的檔案不重算 hash
            _m_done, _m_total = 0, len(records)
            for folder in folders:
                folder_records = [r for r in records if r["split"] == folder.name]
                if not folder_records:
                    continue

                def _mcb(done: int, total: int, _base=_m_done) -> None:
                    _bar.progress(
                        min((_step + (_base + done) / max(_m_total, 1)) / _n_steps, 1.0),
                        text=f"Manifest 更新 {_base + done}/{_m_total}",
                    )

                manifest_by_folder[folder] = update_manifest(
                    folder, folder_records,
                    thumb_lookup=_thumb_lookup, progress_cb=_mcb,
                )
                _m_done += len(folder_records)
            _step += 1
            _bar.progress(_step / _n_steps, text="Manifest 更新完成")

            # 影像內容雜湊（combined 順序）— 固定 UMAP 參考系的點身分證
            all_keys: list[str] = []
            for folder in folders:
                m_entries = manifest_by_folder.get(folder, {})
                all_keys += [m_entries[rel_key(folder, r["path"])]["sha256"]
                             for r in records if r["split"] == folder.name]

            for model_name in selected_models:
                embed_fn = load_model(model_name)
                all_embs = []
                for folder in folders:
                    folder_records = [r for r in records if r["split"] == folder.name]
                    folder_paths = [r["path"] for r in folder_records]
                    cache_path = _dataset_cache_dir(folder, f"embeddings_{model_name}") / "embeddings.npz"
                    if folder_paths:
                        def _cb(done: int, total: int, _m=model_name, _f=folder.name, _s=_step) -> None:
                            _bar.progress(
                                min((_s + done / max(total, 1)) / _n_steps, 1.0),
                                text=f"[{_m}] {_f}: 特徵擷取 {done}/{total}",
                            )
                        m_entries = manifest_by_folder.get(folder, {})
                        keys = [m_entries[rel_key(folder, p)]["sha256"]
                                for p in folder_paths]
                        all_embs.append(
                            extract_embeddings(folder_paths, embed_fn,
                                               cache_path=cache_path,
                                               progress_cb=_cb, cache_keys=keys)
                        )
                        set_embedding_refs(m_entries, folder, model_name, folder_paths)
                embeddings = np.vstack(all_embs)
                raw_per_model[model_name] = embeddings
                _step += 1
                _bar.progress(_step / _n_steps, text=f"[{model_name}] 特徵向量提取完成")

                n_samples = len(embeddings)
                n_comps = min(3, max(1, n_samples - 2))

                def _pad2d(a: np.ndarray) -> np.ndarray:
                    # n≤3 時投影只有 1 維 — 補零軸，散點圖永遠拿得到 y
                    return a if a.shape[1] >= 2 else np.hstack(
                        [a, np.zeros((len(a), 1))])

                proj: dict[str, np.ndarray] = {}
                for mkey, mlabel in method_pairs:
                    # t-SNE/UMAP 對極小樣本無定義（perplexity / n_neighbors
                    # 必須 < n）——誠實跳過，別讓整個 Run 帶著 traceback 倒地
                    if mkey != "pca" and n_samples < 4:
                        _step += 1
                        _bar.progress(_step / _n_steps,
                                      text=f"[{model_name}] {mlabel} 已跳過（樣本 < 4）")
                        continue
                    if mkey == "pca":
                        arr = PCA(n_components=n_comps, random_state=42).fit_transform(embeddings)
                    elif mkey == "tsne":
                        perplexity = min(30, max(1, n_samples - 1))
                        arr = TSNE(n_components=n_comps, random_state=42,
                                   perplexity=perplexity).fit_transform(embeddings)
                    elif mkey in _SUPERVISED_METHODS:
                        arr = _supervised_projection(
                            embeddings, [r["label"] for r in records], mkey, n_comps)
                        if arr is None:
                            _step += 1
                            _bar.progress(_step / _n_steps,
                                          text=f"[{model_name}] {mlabel} 跳過（需 ≥2 類別）")
                            continue
                    else:
                        n_neighbors = min(15, max(2, n_samples - 1))
                        if st.session_state.get("viz_umap_ref"):
                            arr, n_new, refitted = stable_umap(
                                embeddings, all_keys,
                                ref_path_for(folders[0], model_name),
                                n_comps, n_neighbors,
                                rebuild=bool(st.session_state.get("_viz_umap_rebuild")),
                            )
                            mlabel = (f"{mlabel}（重擬合參考系）" if refitted
                                      else f"{mlabel}（參考系沿用，+{n_new} 新點）")
                        else:
                            arr = _umap().UMAP(n_components=n_comps, n_neighbors=n_neighbors,
                                            random_state=42).fit_transform(embeddings)
                    proj[mkey] = _pad2d(arr)
                    _step += 1
                    _bar.progress(_step / _n_steps, text=f"[{model_name}] {mlabel} 完成")

                if not proj:  # 極小樣本且未勾 PCA → 以 PCA 保底，不留空結果
                    proj["pca"] = _pad2d(
                        PCA(n_components=n_comps, random_state=42).fit_transform(embeddings))
                embeddings_per_model[model_name] = proj

            # embedding_refs 填完才落盤 — manifest 是後續策展功能的唯一入口
            for folder, m_entries in manifest_by_folder.items():
                write_manifest(folder, m_entries)
            _status.update(label="完成", state="complete", expanded=False)

        # 離群度與標籤分歧自動算（UX 評審 W7 / F5）：Run 完即排序可用
        outlier_scores: dict[str, np.ndarray] = {}
        label_disagreement: dict[str, np.ndarray] = {}
        if len(records) >= 3:
            k_out = min(5, len(records) - 1)
            rec_labels = [r["label"] for r in records]
            for m, raw in raw_per_model.items():
                outlier_scores[m] = compute_outlier_scores(
                    raw, raw, k=k_out, candidates_in_reference=True)
                label_disagreement[m] = compute_label_disagreement(
                    raw, rec_labels, k=k_out)

        manifest_lookup: dict[str, dict] = {}
        for folder, m_entries in manifest_by_folder.items():
            for key, e in m_entries.items():
                manifest_lookup[str((folder / key).resolve())] = e
        # phash list aligned to records order — the F4 dup-scan input
        phashes = [
            manifest_lookup.get(str(Path(r["path"]).resolve()), {}).get("phash")
            for r in records
        ]

        st.session_state["viz_records"] = records
        st.session_state["viz_embeddings"] = embeddings_per_model
        st.session_state["viz_raw_embeddings"] = raw_per_model
        st.session_state["viz_manifest"] = manifest_lookup
        st.session_state["viz_phashes"] = phashes
        st.session_state["viz_outlier_scores"] = outlier_scores
        st.session_state["viz_label_disagreement"] = label_disagreement
        st.session_state.pop("viz_dup_result", None)
        st.session_state["viz_data_token"] = uuid.uuid4().hex
        st.session_state["viz_nn_index"] = {}
        st.session_state["viz_classes"] = class_names
        st.session_state["viz_selection"] = {
            "token": st.session_state["viz_data_token"], "indices": []
        }
        st.session_state["viz_active_image"] = None
        st.session_state["viz_viewer_ctx"] = []
        st.session_state["viz_query_chain"] = []
        st.session_state["viz_grid_limit"] = _GRID_BATCH
        st.session_state.pop("_viz_mode_snapshot", None)
        st.session_state.pop("_viz_umap_rebuild", None)
        st.session_state["viz_unit"] = "張影像"
        # 匯出清單以 image path 為鍵，跨 Run 仍有效 — 刻意不清
        st.toast(f"完成：{len(records)} 張影像 × {len(selected_models)} 模型", icon="✅")

    if "viz_records" not in st.session_state:
        _render_quick_start()
        return

    records = st.session_state["viz_records"]
    embeddings_per_model = st.session_state["viz_embeddings"]
    data_token = st.session_state.get("viz_data_token", "")

    model_names = list(embeddings_per_model.keys())
    unique_splits = sorted({r["split"] for r in records})

    col_plot, col_panel = st.columns([5, 3], gap="medium")

    with col_plot:
        # 常駐資料規模摘要（G6）：不靠 Run 當下的 banner/toast，rerun 後仍可見
        _unit = st.session_state.get("viz_unit", "張影像")
        st.caption(f"{len(records)} {_unit} · {len(model_names)} 模型 · "
                   f"{len(unique_splits)} 個 split")
        _cmsg = st.session_state.get("viz_class_msg")  # 常駐重現 Run 當下的類別偵測訊息(G6)
        if _cmsg:
            st.caption(_cmsg)
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1.4])
        _md = model_names.index("dinov2_vitb14") if "dinov2_vitb14" in model_names else 0
        selected_model = c1.selectbox("Model", model_names, index=_md,
                                      key="viz_model_select")
        method_labels = [lbl for lbl, key in _METHOD_KEY.items()
                         if key in embeddings_per_model[selected_model]]
        _xd = method_labels.index("監督UMAP") if "監督UMAP" in method_labels else 0
        selected_method = c2.selectbox("Method", method_labels, index=_xd,
                                       key="viz_method_select")
        selected_split = c3.selectbox("Split", ["All"] + unique_splits, key="viz_split_select")
        dim = 3 if c4.radio("維度", ["2D", "3D"], horizontal=True, key="viz_dim_radio") == "3D" else 2

        # ── 著色依據：類別（預設）/ 標籤分歧（紅＝鄰居都異類＋相鄰異類連線）──
        disagreement = st.session_state.get("viz_label_disagreement", {}).get(selected_model)
        color_by, pairs = "class", None
        if disagreement is not None:
            if st.radio("著色依據", ["類別", "標籤分歧"], horizontal=True,
                        key="viz_color_by",
                        help="標籤分歧＝點以 k 近鄰中異類比例著色（紅＝鄰居都異類），"
                             "並把『最近鄰卻異類』的點對連線——你眼睛看到的爭議，"
                             "工具直接幫你標出；框選後可一鍵送灰帶覆核。") == "標籤分歧":
                color_by = "disagreement"
                pairs = _viz_cross_pairs(selected_model, data_token, records)

        method_key = _METHOD_KEY[selected_method]
        coords = embeddings_per_model[selected_model][method_key]
        if method_key in _SUPERVISED_METHODS:
            st.caption(":orange[⚠ 監督式投影：用類別標籤排版，讓 2D 依類別分明——"
                       "僅視覺輔助/便於挑標錯；真實可分性請看『標錯偵測』的 kNN 純度。]")

        if selected_split == "All":
            indices = list(range(len(records)))
        else:
            indices = [i for i, r in enumerate(records) if r["split"] == selected_split]

        # 右側即時過濾：信心（資料含 score 才出現）／源短邊 → 直接濾 indices
        # （不動 records/coords → 即時生效、調低門檻物件即回復；物件級才有意義）。
        _obj_filter_controls("viz", records)
        indices = _obj_view_indices(records, indices)

        # ── 選取生命週期（UX 評審 W2）──
        # 選取只掛 data token：換 model/method/split/dim 一律保留，
        # 只有重新 Run（資料變更）才清空。
        sel_state = st.session_state.get("viz_selection") or {}
        if sel_state.get("token") != data_token:
            sel_state = {"token": data_token, "indices": []}

        # scatter widget key 帶 view 資訊：舊視圖的 widget 事件不可能滲入新視圖
        # （含 color_by：切換著色模式＝乾淨重掛，避免跨模式殘留選取狀態）
        # _clear_nonce 只在按「取消框選」時 +1 → key 改變 → 散點 widget 重新掛載、
        # 瀏覽器端的框真正被丟掉(否則同一個 key 會把舊選取再回報回來,等於沒清掉)。
        _clear_nonce = st.session_state.get("_viz_clear_nonce", 0)
        # 物件過濾會改變顯示點集 → 進 key（同 split 的道理），否則 widget 不重掛、停舊點集。
        scatter_key = (f"viz_scatter_{data_token[:8]}_{selected_model}"
                       f"_{method_key}_{selected_split}_{color_by}_{_clear_nonce}"
                       f"_{_obj_filter_tag()}")

        # NOTE: the 2D interactive chart must keep a STABLE figure spec across
        # reruns — mutating it (e.g. adding a highlight trace) makes Streamlit
        # reset the chart's selection state, silently dropping the user's
        # box/lasso selection. Highlight rings are therefore 3D-only.
        # 3D doesn't carry a box/lasso selection to protect, so it can safely
        # highlight the WHOLE current 2D selection (重評 #3：2D 選、3D 看) —
        # ring the selected batch plus the active image.
        active_idx = st.session_state.get("viz_active_image")
        if dim == 3:
            highlight = list(sel_state["indices"])
            if active_idx is not None and active_idx not in highlight:
                highlight.append(active_idx)
        else:
            highlight = []
        fig = _build_viz_figure(records, coords, indices, selected_model, selected_method,
                                dim=dim, highlight=highlight,
                                color_by=color_by, disagreement=disagreement, pairs=pairs)

        if dim == 3 and highlight:
            st.caption(f"3D 看：黑圈為目前選取的 {len(highlight)} 點（2D 選、3D 看）。")
        if color_by == "disagreement":
            st.caption(":gray[🔴 紅＝k 近鄰多為異類（最該複查標註）；紅線＝最近鄰卻異類的點對。"
                       "框選爭議點 → 下方一鍵送灰帶覆核。]")
        if dim == 2 and len(indices) > _SCATTERGL_THRESHOLD:
            st.caption(f"⚡ {len(indices)} 點：已切換 WebGL 加速渲染；框選/套索照常可用，"
                       "單點 hover 精度略降。")
        if not sel_state["indices"] and dim == 2:
            st.caption("💡 在圖上拖曳框選或套索圈點，右欄會立即顯示對應縮圖。")
        # plotly 工具列的按鈕無法回寫伺服器端選取狀態,故以右上角小「✕」圖示鈕貼著工具列。
        # 用 placeholder 佔位、等散點事件處理完(sel_state 定案)再填入,才不會永遠反灰。
        _clear_slot = st.columns([11, 1])[1].empty() if dim == 2 else None
        if dim == 2:
            st.markdown(
                "<style>.st-key-viz_clear_sel_plot button{min-height:0;height:34px;"
                "padding:0 .55rem;border-radius:6px;font-size:1rem;line-height:1;}</style>",
                unsafe_allow_html=True)
        with st.container(key="viz_scatter_wrap"):
            if dim == 2:
                event = st.plotly_chart(
                    fig, use_container_width=True, key=scatter_key,
                    on_select="rerun", selection_mode=("points", "box", "lasso"),
                )
                sel_points: list[dict] = []
                if event is not None:
                    sel_obj = event.get("selection") if hasattr(event, "get") else None
                    if sel_obj:
                        sel_points = list(sel_obj.get("points", []))
                new_indices = selection_points_to_indices(sel_points)
                # 單向資料流：只有「非空」的 widget 事件能改寫選取；
                # 清空只能走右欄的 ✕ 清除（_clear_selection）。
                if new_indices and new_indices != sel_state["indices"]:
                    sel_state = {"token": data_token, "indices": new_indices}
                    st.session_state["viz_grid_limit"] = _GRID_BATCH
                    st.session_state["viz_active_image"] = None
                    st.session_state["viz_viewer_ctx"] = []
                    st.toast(f"已選取 {len(new_indices)} 個點", icon="🎯")
            else:
                if False:  # deck.gl 引擎已移除（依需求）；3D 一律用 plotly + 區塊號碼
                    import deck3d
                    cmap = _viz_color_map(records)
                    with st.spinner("準備 3D 縮圖…"):
                        thumbs = _viz_deck_thumbs(records, data_token)
                    _z = coords.shape[1] > 2
                    pts = [{"index": int(i),
                            "position": [float(coords[i][0]), float(coords[i][1]),
                                         float(coords[i][2]) if _z else 0.0],
                            "color": _hex_rgb(cmap.get(records[i].get("label", ""), "#9aa0a6")),
                            "thumb": thumbs[i] if i < len(thumbs) else "",
                            "label": records[i].get("label", "")}
                           for i in indices]
                    picked = deck3d.render_deck3d(
                        pts, key=f"viz_deck3d_{data_token[:8]}_{selected_model}_{method_key}")
                    if picked:
                        merged = list(dict.fromkeys(
                            list(sel_state["indices"])
                            + [pi for pi in picked if 0 <= pi < len(records)]))
                        if merged != sel_state["indices"]:
                            sel_state = {"token": data_token, "indices": merged}
                            st.session_state["viz_grid_limit"] = _GRID_BATCH
                            st.session_state["viz_selection"] = sel_state
                            st.toast(f"3D 框選 {len(picked)} 點（共 {len(merged)}）", icon="🎯")
                    st.caption("ℹ deck.gl 3D：拖曳=旋轉、滾輪=縮放、**滑點=看縮圖**、"
                               "**Shift+拖框=選取**。點多時首次較久；不順可切回 plotly。")
                else:
                    # 3D 總覽(plotly)標上「區塊號碼」→ 選號碼進入該塊做原生 2D 框選
                    _ckey = f"{data_token[:8]}_{selected_model}_{method_key}_{selected_split}"
                    _K = st.slider("區塊數 K", 4, 30, 12, key="viz_3d_blocks_k",
                                   help="把 3D 點以 KMeans 分成 K 塊並標號;選一個號碼進該塊做 2D 框選。")
                    blocks, centers = _viz_kmeans_blocks(coords, indices, _K, _ckey)
                    if len(centers):
                        # scene annotations(layout)而非 trace → 不受圖例「全選/全不選」影響
                        fig.update_scenes(annotations=[
                            dict(x=float(centers[b, 0]), y=float(centers[b, 1]),
                                 z=float(centers[b, 2]), text=str(b), showarrow=False,
                                 font=dict(size=18, color="#111"),
                                 bgcolor="rgba(255,255,255,0.7)", borderpad=2)
                            for b in range(len(centers))])
                    st.plotly_chart(fig, use_container_width=True, key="viz_scatter_3d")
                    st.caption("旋轉看各區塊的號碼 → 下方選號碼進入該塊做 **2D 框選/套索**。")
                    from collections import Counter as _Counter
                    _cnt = _Counter(blocks.values())
                    _opts = [f"#{b}（{_cnt[b]}）" for b in range(len(centers)) if _cnt.get(b)]
                    if _opts:
                        _pick = st.selectbox("進入區塊 → 2D 框選", _opts,
                                             key="viz_3d_block_pick")
                        _bk = int(_pick.split("（")[0][1:])
                        _bidx = [i for i in indices if blocks.get(int(i)) == _bk]
                        if _bidx:
                            # 該塊單獨 PCA→2D,把塊內結構攤開(比沿用 3D 座標更好框選)
                            _raw = st.session_state.get(
                                "viz_raw_embeddings", {}).get(selected_model)
                            _cmap = _viz_color_map(records)
                            if _raw is not None and len(_bidx) > 2:
                                _xy = PCA(n_components=2, random_state=42).fit_transform(
                                    _raw[_bidx])
                            else:
                                _xy = np.array([[coords[i][0], coords[i][1]] for i in _bidx])
                            _bfig = go.Figure(go.Scatter(
                                x=_xy[:, 0].tolist(), y=_xy[:, 1].tolist(), mode="markers",
                                marker=dict(size=9, color=[
                                    _cmap.get(records[i].get("label", ""), "#9aa0a6")
                                    for i in _bidx]),
                                customdata=[[i] for i in _bidx],
                                text=[f"#{i} · {records[i].get('label', '')}" for i in _bidx],
                                hovertemplate="%{text}<extra></extra>"))
                            _bfig.update_layout(height=360, dragmode="select",
                                                margin=dict(l=10, r=10, t=10, b=10))
                            _bev = st.plotly_chart(
                                _bfig, use_container_width=True,
                                key=f"viz_block2d_{_ckey}_{_bk}",
                                on_select="rerun",
                                selection_mode=("points", "box", "lasso"))
                            _bpts = []
                            if _bev is not None:
                                _so = _bev.get("selection") if hasattr(_bev, "get") else None
                                if _so:
                                    _bpts = selection_points_to_indices(
                                        list(_so.get("points", [])))
                            if _bpts:
                                merged = list(dict.fromkeys(
                                    list(sel_state["indices"]) + _bpts))
                                if merged != sel_state["indices"]:
                                    sel_state = {"token": data_token, "indices": merged}
                                    st.session_state["viz_grid_limit"] = _GRID_BATCH
                                    st.session_state["viz_selection"] = sel_state
                                    st.toast(f"區塊#{_bk} 框選 {len(_bpts)} 點"
                                             f"（共 {len(merged)}）", icon="🎯")
                    with st.expander("或：依 # 編號加選"):
                        pc1, pc2 = st.columns([3, 1])
                        _ids = pc1.text_input(
                            "輸入 # 編號", key="viz_3d_pick_input",
                            placeholder="例：12, 45, 88", label_visibility="collapsed")
                        if pc2.button("➕ 加選", key="viz_3d_pick_btn",
                                      use_container_width=True):
                            try:
                                nums = [int(x) for x in _ids.replace("，", ",").split(",")
                                        if x.strip()]
                            except ValueError:
                                nums = []
                            valid = [n for n in nums if 0 <= n < len(records)]
                            if valid:
                                merged = list(dict.fromkeys(list(sel_state["indices"]) + valid))
                                sel_state = {"token": data_token, "indices": merged}
                                st.session_state["viz_grid_limit"] = _GRID_BATCH
                                st.session_state["viz_selection"] = sel_state
                                st.toast(f"已加選 {len(valid)} 點", icon="🎯")
                                st.rerun()
                            else:
                                st.warning("沒有有效的 # 編號。")
        st.session_state["viz_selection"] = sel_state

        if _clear_slot is not None:  # fill now that sel_state is final → correct enabled state
            _nsel = len(sel_state["indices"])
            _clear_slot.button(
                f"✕ {_nsel}" if _nsel else "✕",
                key="viz_clear_sel_plot", use_container_width=True,
                disabled=not _nsel, help="取消框選",
                on_click=_clear_selection, args=(scatter_key,))

        with st.expander("🩹 標錯偵測 / 類別混淆（全 384 維，不受 2D 擠壓影響）"):
            _raw = st.session_state.get("viz_raw_embeddings", {}).get(selected_model)
            if _raw is not None:
                _render_label_quality(records, _raw, selected_model, data_token)
            else:
                st.caption("此檢視需要原始高維特徵。")

        # 「送灰帶覆核」與「送 Labeling」兩個出口已移到右上角選取區並排（_render_select_view）

        dl_fig = build_plotly_figure(records, embeddings_per_model)
        st.download_button(
            "⬇ Download HTML (all views)",
            data=dl_fig.to_html(include_plotlyjs="cdn"),
            file_name="embeddings_visualization.html",
            mime="text/html",
        )

    with col_panel:
        _render_right_panel(records, coords, selected_model, selected_split, scatter_key)


def _set_cmp_active(idx: int | None, ctx: list[int] | None = None) -> None:
    st.session_state["cmp_active_image"] = idx
    if ctx is not None:
        st.session_state["cmp_viewer_ctx"] = ctx


@st.fragment
def _render_cmp_panel(cmp_paths: list[Path], cmp_groups: list[str]) -> None:
    """Compare 的 linked view 右欄：框選的影像縮圖 + 檢視槽（同 Visualize 的
    互動模型；fragment 隔離，點縮圖不重繪散點）。"""
    sel_state = st.session_state.get("cmp_selection") or {}
    sel = (sel_state.get("indices", [])
           if sel_state.get("token") == st.session_state.get("cmp_data_token") else [])
    with st.container(height=500, border=True, key="cmp_image_viewer"):
        idx = st.session_state.get("cmp_active_image")
        if idx is None or not (0 <= idx < len(cmp_paths)):
            st.caption("檢視槽 — 在左圖框選資料點後，點下方縮圖在此檢視大圖。")
        else:
            p = Path(cmp_paths[idx])
            ctx = st.session_state.get("cmp_viewer_ctx") or [idx]
            pos = ctx.index(idx) if idx in ctx else 0
            h1, h2, h3, h4 = st.columns([5, 1, 1, 1])
            h1.markdown(f"**{p.name}** — {cmp_groups[idx]} · {pos + 1}/{len(ctx)} · #{idx}")
            h2.button("◀", key="cmp_img_prev", disabled=pos <= 0,
                      on_click=_set_cmp_active, args=(ctx[max(pos - 1, 0)],))
            h3.button("▶", key="cmp_img_next", disabled=pos >= len(ctx) - 1,
                      on_click=_set_cmp_active, args=(ctx[min(pos + 1, len(ctx) - 1)],))
            h4.button("✕", key="cmp_img_close", on_click=_set_cmp_active, args=(None,))
            if p.exists():
                st.image(str(p), use_container_width=True)
            else:
                st.warning(f"找不到檔案：{p}")
    with st.container(height=560, key="cmp_grid"):
        if not sel:
            st.info("在左圖以點選、框選（box）或套索（lasso）圈出資料點，"
                    "對應影像會立即顯示在這裡。")
            return
        shown = sel[:60]
        st.caption(f"已選取 {len(sel)} 張" +
                   (f" · 顯示前 {len(shown)}" if len(sel) > len(shown) else ""))
        cols = st.columns(3)
        for j, i in enumerate(shown):
            with cols[j % 3]:
                p = Path(cmp_paths[i])
                thumb = _thumb_or_none(p)
                if thumb is not None:
                    st.image(thumb, use_container_width=True)
                else:
                    st.warning("⚠ 檔案遺失")
                st.button(f"{cmp_groups[i]}｜{Path(cmp_paths[i]).name}",
                          key=f"cmp_card_{i}", use_container_width=True,
                          on_click=_set_cmp_active, args=(i, list(shown)))


_CMP_DEMO_A = Path(__file__).parent.parent / "demo" / "imagenette" / "train" / "cassette_player"
_CMP_DEMO_B = Path(__file__).parent.parent / "demo" / "imagenette" / "train" / "chainsaw"
_CMP_CACHE_DIRS = ("embeddings_", "object_crops", ".thumbs")


def _load_cmp_demo() -> None:
    """一鍵填入兩個範例『直接含圖片』資料夾並自動跑（對齊其他工具的 demo）。"""
    a, b = _demo_compare_dirs()
    st.session_state["cmp_folder_a"] = a
    st.session_state["cmp_folder_b"] = b
    st.session_state["_cmp_autorun"] = True
    _log_usage("cmp_demo_load")


def _cmp_resolve_images(folder: Path) -> tuple[list[Path], str | None]:
    """抓資料夾裡的影像（YOLO 偵測資料夾通常放在 images/ 子夾）。先抓直接圖片,
    沒有就往下層遞迴找(略過 embeddings_/縮圖/crop 等快取夾),回傳 (paths, note)。"""
    flat = get_image_paths(folder)
    if flat:
        return flat, None
    rec: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in folder.rglob(ext):
            if any(part.startswith(_CMP_CACHE_DIRS) for part in p.parts):
                continue
            rec.append(p)
    rec = sorted(set(rec))
    if rec:
        return rec, f"從子資料夾找到 {len(rec)} 張圖（會讀對應的 labels/ 做物件級比較）。"
    return [], None


def _cmp_policy_token(model: str) -> str:
    return (f"{st.session_state.get('cmp_folder_a','')}|"
            f"{st.session_state.get('cmp_folder_b','')}|{model}")


def _cmp_autotune(paths_a, paths_b, model):
    """在兩個資料夾中**較小**的那個自動找最佳物件設定(pad/res/head),套用到兩邊。
    較小者＝物件較少→量測較快。回傳 best policy,並把排名表存進 session 供顯示。"""
    import object_eval as oe
    small = paths_a if len(paths_a) <= len(paths_b) else paths_b
    which = "A" if small is paths_a else "B"
    if not small:  # 不該發生(Run 前已擋空資料夾),保險用預設
        best = dict(oe.DEFAULT_POLICY)
        st.session_state["cmp_object_policy"] = best
        st.session_state["cmp_policy_token"] = _cmp_policy_token(model)
        return best
    cnames = _manual_class_names("cmp") or _cov_class_names([{"path": str(small[0])}])
    root = _cov_object_root([{"path": str(small[0])}])
    rows_at, best, fp = [], None, None
    with st.status(f"自動尋找最佳設定（用較小的 Folder {which}，{len(small)} 張）…",
                   expanded=True) as _s:
        _b = st.progress(0.0)

        def _p(ci, n, pol, od, ot):
            frac = (ci + od / max(ot, 1)) / max(n, 1)
            _b.progress(min(frac, 1.0),
                        text=f"[{ci+1}/{n}] {oe.policy_tag(pol)} · 物件 {od}/{ot}")

        try:
            rows_at, best, fp = oe.run_autotune(
                small, cnames, model=model,
                cache_dir=_dataset_cache_dir(root, "object_crops") / "_autotune",
                progress=_p)
        except Exception:
            rows_at, best, fp = [], None, None  # 無 labels/物件等 → 退預設,交下游友善錯誤
        if not best:  # 找不到物件(無 bbox)→ 用預設,讓下游顯示「找不到 bbox」友善訊息
            best = dict(oe.DEFAULT_POLICY)
        _s.update(label=f"物件設定：{oe.policy_tag(best)}（用較小的 Folder {which}）",
                  state="complete", expanded=False)
    st.session_state["cmp_autotune"] = {"rows": rows_at, "best": best,
                                        "fp": fp, "which": which}
    st.session_state["cmp_object_policy"] = best
    st.session_state["cmp_policy_token"] = _cmp_policy_token(model)
    return best


def _compute_compare_by_class(paths_a, paths_b, name_a, name_b, model) -> bool:
    """逐 YOLO 類別比 A vs B：**自動找物件設定（用較小資料夾）**→裁出物件→各自
    embedding→依類別比形心餘弦距離。成功寫進 session 回傳 True;失敗顯示錯誤回傳 False。"""
    # 設定一律自動尋找(不讓 user 選 pad);換資料夾/模型就自動重新量測,套用到 A／B 兩邊。
    if (st.session_state.get("cmp_object_policy") is None
            or st.session_state.get("cmp_policy_token") != _cmp_policy_token(model)):
        _cmp_autotune(paths_a, paths_b, model)
    policy = st.session_state["cmp_object_policy"]
    _pad = float(policy.get("pad", 0.12))
    recs_a = [{"path": str(p)} for p in paths_a]
    recs_b = [{"path": str(p)} for p in paths_b]
    _manual = _manual_class_names("cmp")  # 手動指定 > 各邊自動偵測 classes.txt
    oa, ea, _ = _crop_and_embed_objects(
        recs_a, model, _manual or _cov_class_names(recs_a), _pad,
        base_token="cmpA" + uuid.uuid4().hex, session_key="_cmp_objA",
        spinner="A：裁切物件", policy=policy)
    ob, eb, _ = _crop_and_embed_objects(
        recs_b, model, _manual or _cov_class_names(recs_b), _pad,
        base_token="cmpB" + uuid.uuid4().hex, session_key="_cmp_objB",
        spinner="B：裁切物件", policy=policy)
    if not oa or not ob:
        which = "A" if not oa else "B"
        st.error(f"Folder {which} 的 labels/ 找不到任何 bbox — 此功能需要偵測標註"
                 "（YOLO images/+labels/ 或 COCO 標註 JSON）。請選含標註的資料夾，"
                 "或切「整張影像（不分類別）」。"
                 + _seg_format_hint(paths_a if not oa else paths_b))
        return False
    la = [r.get("label", "") for r in oa]
    lb = [r.get("label", "") for r in ob]
    import collections as _co
    _ca, _cb = _co.Counter(la), _co.Counter(lb)
    # 整類差異：一邊有、另一邊完全沒有（逐類別交集漂移表看不到的最大差異，如 couch）
    a_only = {c: _ca[c] for c in sorted(set(la) - set(lb))}
    b_only = {c: _cb[c] for c in sorted(set(lb) - set(la))}
    classes = sorted(set(la) & set(lb))
    if not classes:
        _extra = ""
        if a_only or b_only:
            _extra = (f"（只在 A：{'、'.join(a_only) or '—'}；"
                      f"只在 B：{'、'.join(b_only) or '—'}）")
        st.error("A 與 B 沒有共同的 YOLO 類別,無法逐類別比較"
                 "（兩邊 classes.txt 的類別名需一致）。" + _extra)
        return False
    rows = []
    for c in classes:
        ia = [i for i, l in enumerate(la) if l == c]
        ib = [i for i, l in enumerate(lb) if l == c]
        ma, mb = ea[ia].mean(0), eb[ib].mean(0)
        ma = ma / (np.linalg.norm(ma) + 1e-9)
        mb = mb / (np.linalg.norm(mb) + 1e-9)
        drift = max(0.0, 1.0 - float(ma @ mb))
        rows.append({"類別": c, "A 數量": len(ia), "B 數量": len(ib),
                     "漂移↓": round(drift, 4)})
    rows.sort(key=lambda r: r["漂移↓"], reverse=True)
    st.session_state.update({
        "cmpc_oa": oa, "cmpc_ea": ea, "cmpc_ob": ob, "cmpc_eb": eb,
        "cmpc_la": la, "cmpc_lb": lb, "cmpc_rows": rows,
        "cmpc_aonly": a_only, "cmpc_bonly": b_only,
        "cmpc_names": (name_a, name_b), "cmpc_model": model,
        "cmpc_token": uuid.uuid4().hex, "cmp_result_kind": "byclass",
    })
    return True


_CMP_ALL_LABEL = "(全部影像)"


def _compute_compare_whole(root_a: Path, root_b: Path, paths_a, paths_b,
                           name_a, name_b, model) -> bool:
    """整張影像・不分類別:整圖 embedding → A/B 形心餘弦距離（整體漂移）。
    不需 labels/。結果塞進與 by-class 相同的 cmpc_* session 管線
    （單一類別 _CMP_ALL_LABEL），由 render 端以 cmp_result_kind 區分呈現。"""
    embed_fn = load_model(model)
    with st.status("計算整張影像 embedding…", expanded=True):
        bar = st.progress(0.0, text="A：特徵擷取…")
        ea = extract_embeddings(
            paths_a, embed_fn,
            cache_path=_dataset_cache_dir(root_a, f"embeddings_{model}") / "embeddings.npz",
            progress_cb=lambda d, t: bar.progress(
                min(d / max(t, 1) * 0.5, 0.5), text=f"A：特徵擷取 {d}/{t}"))
        eb = extract_embeddings(
            paths_b, embed_fn,
            cache_path=_dataset_cache_dir(root_b, f"embeddings_{model}") / "embeddings.npz",
            progress_cb=lambda d, t: bar.progress(
                min(0.5 + d / max(t, 1) * 0.5, 1.0), text=f"B：特徵擷取 {d}/{t}"))
    if ea is None or eb is None or not len(ea) or not len(eb):
        st.error("有一邊算不出 embedding（影像可能全部無法解碼）。")
        return False
    ma = ea.mean(0); ma = ma / (np.linalg.norm(ma) + 1e-9)
    mb = eb.mean(0); mb = mb / (np.linalg.norm(mb) + 1e-9)
    drift = max(0.0, 1.0 - float(ma @ mb))
    oa = [{"path": str(p), "label": _CMP_ALL_LABEL} for p in paths_a]
    ob = [{"path": str(p), "label": _CMP_ALL_LABEL} for p in paths_b]
    rows = [{"類別": _CMP_ALL_LABEL, "A 數量": len(oa), "B 數量": len(ob),
             "漂移↓": round(drift, 4)}]
    st.session_state.update({
        "cmpc_oa": oa, "cmpc_ea": ea, "cmpc_ob": ob, "cmpc_eb": eb,
        "cmpc_la": [_CMP_ALL_LABEL] * len(oa), "cmpc_lb": [_CMP_ALL_LABEL] * len(ob),
        "cmpc_rows": rows, "cmpc_aonly": {}, "cmpc_bonly": {},
        "cmpc_names": (name_a, name_b), "cmpc_model": model,
        "cmpc_token": uuid.uuid4().hex, "cmp_result_kind": "whole",
    })
    return True


_CMPC_BOFF = 10_000_000  # customdata sentinel: code >= OFFSET ⇒ group B


def _render_compare_by_class() -> None:
    rows = st.session_state.get("cmpc_rows") or []
    name_a, name_b = st.session_state.get("cmpc_names", ("A", "B"))
    _whole = st.session_state.get("cmp_result_kind") == "whole"
    _noun = "影像" if _whole else "物件"
    if _whole:
        st.subheader(f"整張影像比較：{name_a} vs {name_b}")
        st.caption("**整體漂移**＝A／B 全部影像 embedding 形心的餘弦距離，越大代表兩堆影像"
                   "整體長得越不一樣。分佈細節看下方散點（監督投影以 A／B 分組，會凸顯兩邊差異）。")
    else:
        st.subheader(f"逐類別比較：{name_a} vs {name_b}")
        _pol = st.session_state.get("cmp_object_policy")
        if _pol:
            import object_eval as oe
            _w = (st.session_state.get("cmp_autotune") or {}).get("which", "?")
            st.caption(f"物件設定：**{oe.policy_tag(_pol)}**（自動量測於較小的 Folder {_w}），A／B "
                       "兩邊一致。**漂移**＝該類別在 A／B 兩邊 embedding 形心的餘弦距離，越大代表"
                       "這類物件在兩份資料間長得越不一樣。")
    if not rows:
        st.warning("沒有可比較的共同類別。"); return
    # 整類差異（一邊有、另一邊完全沒有）——逐類別交集漂移表結構上看不到，這裡補上
    _aonly = st.session_state.get("cmpc_aonly") or {}
    _bonly = st.session_state.get("cmpc_bonly") or {}
    if _aonly or _bonly:
        _seg = []
        if _aonly:
            _seg.append(f"**只在 {name_a}**：" + "、".join(f"{c}({n})" for c, n in _aonly.items()))
        if _bonly:
            _seg.append(f"**只在 {name_b}**：" + "、".join(f"{c}({n})" for c, n in _bonly.items()))
        st.warning("⚠ 整類差異（一邊有、另一邊完全沒有——這是最大的分佈差異，"
                   "但逐類別漂移表只比共同類、結構上看不到）：　" + "　｜　".join(_seg))
    if not _whole:
        _maxd = max(0.05, max(r["漂移↓"] for r in rows))
        st.dataframe(
            pd.DataFrame(rows), use_container_width=True, hide_index=True,
            column_config={
                "類別": st.column_config.TextColumn("類別"),
                "A 數量": st.column_config.NumberColumn(f"{name_a} 數量"),
                "B 數量": st.column_config.NumberColumn(f"{name_b} 數量"),
                "漂移↓": st.column_config.ProgressColumn(
                    "類別漂移（越大＝越不同）", help="A／B 形心餘弦距離，已由大到小排序",
                    min_value=0.0, max_value=_maxd, format="%.3f"),
            })

    oa = st.session_state["cmpc_oa"]; ea = st.session_state["cmpc_ea"]
    ob = st.session_state["cmpc_ob"]; eb = st.session_state["cmpc_eb"]
    la, lb = st.session_state["cmpc_la"], st.session_state["cmpc_lb"]
    dc1, dc2, dc3 = st.columns([3, 1, 1])
    if _whole:
        pick = rows[0]["類別"]
        dc1.metric("整體漂移（越大＝越不同）", f"{rows[0]['漂移↓']:.4f}",
                   help=f"{name_a} {rows[0]['A 數量']} 張 vs {name_b} {rows[0]['B 數量']} 張"
                        "的 embedding 形心餘弦距離")
    else:
        pick = dc1.selectbox("進入某一類看 A vs B 散點（預設＝漂移最大）",
                             [r["類別"] for r in rows])
    _pm = dc2.selectbox("投影方法", list(_METHOD_KEY),
                        index=list(_METHOD_KEY).index("監督UMAP"), key="cmpc_proj",
                        help="散點降維法；非線性／監督法的距離僅供視覺，漂移以上表為準。"
                             "LDA／監督UMAP 以 A／B 分組做監督，凸顯兩邊差異。")
    _dim = 3 if dc3.radio("維度", ["2D", "3D"], horizontal=True, key="cmpc_dim") == "3D" else 2

    ia = [i for i, l in enumerate(la) if l == pick]
    ib = [i for i, l in enumerate(lb) if l == pick]
    if not _whole:
        # 即時過濾（信心／源短邊）：直接縮 ia/ib —— n_a/combined/coords/code/gallery 全由
        # ia/ib 衍生 → 自動一致；filter 狀態入投影快取 key，改門檻才重投影。
        # （整張影像模式無 conf／bbox 欄位 → 跳過物件過濾器。）
        _obj_filter_controls("cmpc", oa)
        ia = [i for i in ia if _obj_keep(oa[i])]
        ib = [i for i in ib if _obj_keep(ob[i])]
    n_a = len(ia)
    if not (ia or ib):
        st.info("目前的信心／大小門檻把這一類的物件全濾掉了——把門檻調低看看。")
        return
    combined = np.vstack([ea[ia], eb[ib]])
    _nc = 3 if _dim == 3 else 2
    _ff = (f"{int(st.session_state.get('obj_min_short_px', 0) or 0)}"
           f"_{st.session_state.get('obj_conf_range') or (0.0, 1.0)}")
    # 投影快取:UMAP/t-SNE 很慢,**每次框選都會 rerun**,不快取就會「畫面當掉」。
    _proj_key = f"{st.session_state.get('cmpc_token','')}_{pick}_{_pm}_{_dim}_{_ff}"
    _pcache = st.session_state.get("_cmpc_proj_cache") or {}
    if _pcache.get("key") == _proj_key:
        coords = _pcache["coords"]
    else:
        _mk = _METHOD_KEY[_pm]
        if len(combined) <= _nc + 1:
            coords = np.zeros((len(combined), _nc))
        elif _mk == "pca":
            coords = PCA(n_components=_nc, random_state=42).fit_transform(combined)
        else:
            with st.spinner(f"投影中（{_pm}）…"):
                if _mk == "tsne":
                    _pp = min(30, max(2, len(combined) - 1))
                    coords = TSNE(n_components=_nc, random_state=42,
                                  perplexity=_pp).fit_transform(combined)
                elif _mk == "umap":
                    _nn = min(15, max(2, len(combined) - 1))
                    coords = _umap().UMAP(n_components=_nc, n_neighbors=_nn,
                                          random_state=42).fit_transform(combined)
                else:  # 監督法(lda / sumap)：以 A／B 分組做監督
                    _groups = [0] * n_a + [1] * len(ib)
                    _sup = _supervised_projection(combined, _groups, _mk, _nc)
                    if _sup is None:
                        coords = PCA(n_components=_nc, random_state=42).fit_transform(combined)
                    else:
                        _sup = np.asarray(_sup)
                        if _sup.shape[1] < _nc:  # LDA 兩群→1 維,補 PCA 湊成 2D/3D
                            _padc = PCA(n_components=_nc - _sup.shape[1],
                                        random_state=42).fit_transform(combined)
                            coords = np.hstack([_sup, _padc])
                        else:
                            coords = _sup
        st.session_state["_cmpc_proj_cache"] = {"key": _proj_key, "coords": coords}

    # customdata sentinel so a flat selection index routes back to oa/ob
    code_a = [[ia[j]] for j in range(n_a)]
    code_b = [[ib[j] + _CMPC_BOFF] for j in range(len(ib))]
    all_codes = [c[0] for c in code_a] + [c[0] for c in code_b]
    txt_a = [f"{name_a} · {Path(oa[ia[j]]['path']).name}" for j in range(n_a)]
    txt_b = [f"{name_b} · {Path(ob[ib[j]]['path']).name}" for j in range(len(ib))]

    _tok = st.session_state.get("cmpc_token", "")[:8]
    _sel_tok = f"{_tok}_{pick}_{_pm}_{_dim}"
    cur = st.session_state.get("cmpc_sel") or {}
    if cur.get("token") != _sel_tok:
        cur = {"token": _sel_tok, "a": [], "b": []}
    _cn = st.session_state.get("_cmpc_clear_nonce", 0)
    sel_codes = None

    _nsel = len(cur.get("a", [])) + len(cur.get("b", []))
    tb1, tb2 = st.columns([5, 1])
    tb1.caption(f"💡 在散點上拖曳框選／套索圈點 → 下方顯示選到的{_noun}縮圖。")
    if tb2.button(f"✕ 取消框選（{_nsel}）" if _nsel else "✕ 取消框選",
                  key="cmpc_clear_top", use_container_width=True, disabled=not _nsel):
        st.session_state["_cmpc_clear_nonce"] = _cn + 1
        st.session_state["cmpc_sel"] = {"token": _sel_tok, "a": [], "b": []}
        st.rerun()

    if _dim == 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=coords[:n_a, 0].tolist(), y=coords[:n_a, 1].tolist(), mode="markers",
            name=name_a, marker=dict(color="#3498db", size=8, opacity=0.7),
            customdata=code_a, text=txt_a, hovertemplate="%{text}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=coords[n_a:, 0].tolist(), y=coords[n_a:, 1].tolist(), mode="markers",
            name=name_b, marker=dict(color="#e74c3c", size=8, opacity=0.7),
            customdata=code_b, text=txt_b, hovertemplate="%{text}<extra></extra>"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=34, b=10),
                          dragmode="select", legend=dict(orientation="h"),
                          updatemenus=_legend_toggle_buttons())
        ev = st.plotly_chart(
            fig, use_container_width=True, key=f"cmpc2d_{_sel_tok}_{_cn}_{_obj_filter_tag()}",
            on_select="rerun", selection_mode=("points", "box", "lasso"))
        if ev is not None:
            _so = ev.get("selection") if hasattr(ev, "get") else None
            if _so:
                sel_codes = selection_points_to_indices(list(_so.get("points", [])))
    else:
        _K = st.slider("區塊數 K", 4, 30, 12, key="cmpc_3d_k")
        _ck = f"cmpc3d_{_sel_tok}_{_K}"  # K 入 key → 改 K 不會沿用舊的分塊
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=coords[:n_a, 0].tolist(), y=coords[:n_a, 1].tolist(),
            z=coords[:n_a, 2].tolist(), mode="markers", name=name_a,
            marker=dict(color="#3498db", size=4, opacity=0.8),
            customdata=code_a, text=txt_a, hovertemplate="%{text}<extra></extra>"))
        fig.add_trace(go.Scatter3d(
            x=coords[n_a:, 0].tolist(), y=coords[n_a:, 1].tolist(),
            z=coords[n_a:, 2].tolist(), mode="markers", name=name_b,
            marker=dict(color="#e74c3c", size=4, opacity=0.8),
            customdata=code_b, text=txt_b, hovertemplate="%{text}<extra></extra>"))
        blocks, centers = _viz_kmeans_blocks(coords, list(range(len(coords))), _K, _ck)
        if len(centers):
            fig.update_scenes(annotations=[
                dict(x=float(centers[b, 0]), y=float(centers[b, 1]),
                     z=float(centers[b, 2]), text=str(b), showarrow=False,
                     font=dict(size=18, color="#111"),
                     bgcolor="rgba(255,255,255,0.7)", borderpad=2)
                for b in range(len(centers))])
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0),
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True, key=f"cmpc3dfig_{_ck}_{_cn}")
        st.caption("3D 旋轉看區塊號碼 → 下方選號碼進該塊做 **2D 框選/套索**。")
        from collections import Counter as _Counter
        _cnt = _Counter(blocks.values())
        _opts = [f"#{b}（{_cnt[b]}）" for b in range(len(centers)) if _cnt.get(b)]
        if _opts:
            _bp = st.selectbox("進入區塊 → 2D 框選", _opts, key=f"cmpc_block_pick_{_ck}")
            _bk = int(_bp.split("（")[0][1:])
            _bidx = [i for i in range(len(coords)) if blocks.get(int(i)) == _bk]
            if _bidx:
                _xy = (PCA(n_components=2, random_state=42).fit_transform(combined[_bidx])
                       if len(_bidx) > 2 else coords[_bidx][:, :2])
                _bfig = go.Figure(go.Scatter(
                    x=_xy[:, 0].tolist(), y=_xy[:, 1].tolist(), mode="markers",
                    marker=dict(size=9, color=["#3498db" if i < n_a else "#e74c3c"
                                               for i in _bidx]),
                    customdata=[[all_codes[i]] for i in _bidx],
                    text=[(txt_a[i] if i < n_a else txt_b[i - n_a]) for i in _bidx],
                    hovertemplate="%{text}<extra></extra>"))
                _bfig.update_layout(height=340, dragmode="select",
                                    margin=dict(l=10, r=10, t=10, b=10))
                _bev = st.plotly_chart(
                    _bfig, use_container_width=True,
                    key=f"cmpc_block2d_{_ck}_{_bk}_{_cn}",
                    on_select="rerun", selection_mode=("points", "box", "lasso"))
                if _bev is not None:
                    _bso = _bev.get("selection") if hasattr(_bev, "get") else None
                    if _bso:
                        sel_codes = selection_points_to_indices(list(_bso.get("points", [])))

    if sel_codes:
        sa = sorted({v for v in sel_codes if v < _CMPC_BOFF})
        sb = sorted({v - _CMPC_BOFF for v in sel_codes if v >= _CMPC_BOFF})
        if (sa, sb) != (cur["a"], cur["b"]):
            cur = {"token": _sel_tok, "a": sa, "b": sb}
            st.toast(f"框選 {len(sa) + len(sb)} 個{_noun}", icon="🎯")
    st.session_state["cmpc_sel"] = cur

    sa, sb = cur["a"], cur["b"]
    if sa or sb:
        st.caption(f"框選：{name_a} {len(sa)} 個 · {name_b} {len(sb)} 個（下方為選到的{_noun}；"
                   "用上方「✕ 取消框選」清除）")
        # 加入清單：物件級框選 → 收原圖（cart 為 image-level，同圖多框去重成一張）
        _picked: dict[str, dict] = {}
        for _objs, _sel in ((oa, sa), (ob, sb)):
            for _i in _sel:
                _o = _objs[_i]
                _ip = str(_o.get("image_path") or _o.get("path"))
                _picked.setdefault(_ip, {"path": _ip, "label": _o.get("label", ""),
                                         "split": _o.get("split", "")})
        _crecs = list(_picked.values())
        if st.button(f"🛒 把框選的{_noun}加入清單（{len(_crecs)} 張原圖）",
                     key="cmpc_add_cart", use_container_width=True):
            _batch_add(_crecs, list(range(len(_crecs))), source="compare")
    else:
        st.caption(f"未框選 → 顯示前 12 個樣本（{name_a}：{n_a} · {name_b}：{len(ib)}）。")

    def _grid(recs, ids):
        cols = st.columns(6)
        for j, i in enumerate(ids[:60]):
            cols[j % 6].image(str(recs[i]["path"]), use_container_width=True,
                              caption=f"{recs[i].get('label') or '?'}"
                                      f"｜{_rec_fname(recs[i])}")

    # 左右各包一個外框，框色＝散點上該資料集的點色（A 藍／B 紅），框上標資料集名，
    # 讓「左＝A、右＝B」一眼分群（先前兩欄無邊界，縮圖牆糊成一片）。
    _CA, _CB = "#3498db", "#e74c3c"
    st.markdown(
        "<style>"
        f".st-key-cmpc_grp_a{{border:2px solid {_CA};border-radius:8px;padding:6px 12px 10px;}}"
        f".st-key-cmpc_grp_b{{border:2px solid {_CB};border-radius:8px;padding:6px 12px 10px;}}"
        "</style>",
        unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1, st.container(key="cmpc_grp_a"):
        st.markdown(f"<b style='color:{_CA}'>{name_a}</b> ・ {pick}", unsafe_allow_html=True)
        _grid(oa, sa if sa else ia[:12])
    with g2, st.container(key="cmpc_grp_b"):
        st.markdown(f"<b style='color:{_CB}'>{name_b}</b> ・ {pick}", unsafe_allow_html=True)
        _grid(ob, sb if sb else ib[:12])


def _compare_distributions_ui() -> None:
    _UNIT_OBJ, _UNIT_WHOLE = "物件級・按類別（YOLO）", "整張影像（不分類別）"
    with st.sidebar:
        _is_whole = st.session_state.get("cmp_unit") == _UNIT_WHOLE
        _freq = "" if _is_whole else "（偵測資料夾，含 images/＋labels/）"
        # [輸入框|📁] 同列（全 app 一致樣式）。📁 用 on_click 回呼：rerun 開始、
        # widget 實例化前寫回同名 key（同 exp_dst 模式，見 app.py exp_dst 註解）。
        _fa1, _fa2 = st.columns([5, 1], vertical_alignment="bottom")
        with _fa1:
            st.text_input(f"Folder A{_freq}", key="cmp_folder_a",
                          placeholder=r"例：C:\data\setA")
        with _fa2:
            st.button("📁", key="browse_a", use_container_width=True,
                      on_click=_pick_folder, args=("cmp_folder_a",),
                      help="開啟系統的『選擇資料夾』視窗；也可直接在左邊貼上路徑。")
        folder_a = st.session_state.get("cmp_folder_a", "")

        _fb1, _fb2 = st.columns([5, 1], vertical_alignment="bottom")
        with _fb1:
            st.text_input(f"Folder B{_freq}", key="cmp_folder_b",
                          placeholder=r"例：C:\data\setB")
        with _fb2:
            st.button("📁", key="browse_b", use_container_width=True,
                      on_click=_pick_folder, args=("cmp_folder_b",),
                      help="開啟系統的『選擇資料夾』視窗；也可直接在左邊貼上路徑。")
        folder_b = st.session_state.get("cmp_folder_b", "")
        all_models = available_models()
        if not all_models:
            st.error("No .pth models found in ./models/. Add a model file and restart.")
            return
        selected_model = st.selectbox("Model", all_models, key="cmp_model_sel",
                                      index=_default_model_index(all_models))
        unit = st.radio("比較單位", [_UNIT_OBJ, _UNIT_WHOLE], key="cmp_unit",
                        captions=["讀 labels/ 裁物件，逐類別比漂移",
                                  "不需 labels/——平鋪／分類／YOLO 資料夾都可"])
        if unit == _UNIT_OBJ:
            st.caption("讀 labels/ 裁出每個物件，逐類別比 A↔B 漂移。物件設定"
                       "（pad／解析度／head）由系統**自動尋找**（用較小的資料夾），套用到 A／B 兩邊。")
            _class_source_expander("cmp")  # 手動指定套用到 A/B 兩邊(id→名一致)
        else:
            st.caption("整圖 embedding 比 A↔B **整體**分佈：一個整體漂移分數＋A/B 散點"
                       "（框選看縮圖）。適合沒有標註、只想看兩堆影像像不像。")
        run = st.button("▶ Run", use_container_width=True, key="run_cmp", type="primary")

        if unit == _UNIT_OBJ:
            _cmp_object_settings_expander()

    if st.session_state.pop("_cmp_autorun", False):
        run = True

    if run:
        path_a = Path(folder_a.strip()) if folder_a.strip() else None
        path_b = Path(folder_b.strip()) if folder_b.strip() else None

        if not path_a or not path_b:
            st.error("請在左側填入 Folder A 與 Folder B 兩個資料夾路徑"
                     "（或按下方「✨ 用範例資料試跑」）。")
            return
        if not path_a.exists():
            st.error(f"找不到 Folder A：{path_a}")
            return
        if not path_b.exists():
            st.error(f"找不到 Folder B：{path_b}")
            return
        _collapse_sidebar()

        paths_a, note_a = _cmp_resolve_images(path_a)
        paths_b, note_b = _cmp_resolve_images(path_b)

        # 壞檔防呆:兩邊各自前置過濾，壞檔（連同其 label）排除在逐類別比較外
        paths_a, _bad_a = _partition_image_paths(paths_a, "cmp_a")
        paths_b, _bad_b = _partition_image_paths(paths_b, "cmp_b")
        _warn_skipped(list(_bad_a) + list(_bad_b))

        _hint = ("\n物件級・按類別需要 **YOLO 偵測資料夾**（含 images/ 與 labels/）。"
                 "請選資料集根目錄（或其 images/ 子夾），例如 …/train、…/valid。"
                 if unit == _UNIT_OBJ else "")
        if not paths_a:
            st.error(f"Folder A 找不到影像：{path_a}" + _hint); return
        if not paths_b:
            st.error(f"Folder B 找不到影像：{path_b}" + _hint); return
        _ov = len(set(map(str, paths_a)) & set(map(str, paths_b)))
        if _ov:
            st.warning(f"Folder A 與 B 有 {_ov} 張重疊（A 可能是 B 的子集）；"
                       "分佈差異會偏低、參考價值低 — 建議選兩個不重疊的資料夾。")
        if note_a:
            st.info(f"Folder A：{note_a}")
        if note_b:
            st.info(f"Folder B：{note_b}")

        if unit == _UNIT_WHOLE:
            if _compute_compare_whole(path_a, path_b, paths_a, paths_b,
                                      path_a.name, path_b.name, selected_model):
                st.rerun()
            return
        if _compute_compare_by_class(paths_a, paths_b, path_a.name, path_b.name,
                                     selected_model):
            st.rerun()
        return

    if st.session_state.get("cmpc_rows"):
        _render_compare_by_class()
        return

    st.markdown("##### 快速開始")
    st.caption("比兩份資料像不像（train vs val、資料 v1 vs v2、真實 vs 生成…）。"
               "**物件級・按類別（YOLO）**：讀 labels/ 裁物件、逐類別比漂移；"
               "**整張影像（不分類別）**：不需標註，整圖 embedding 比整體分佈。"
               "和「完整度熱力圖」不同：熱力圖看單一資料集**內部**哪裡缺，"
               "這裡看 A、B 兩份資料像不像。")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, st.container(border=True):
        st.markdown("**① 選 Folder A／B**")
        st.caption("物件級需 YOLO 資料夾（含 images/ 與 labels/）；"
                   "整張影像模式任何含影像的資料夾都可。")
    with c2, st.container(border=True):
        st.markdown("**② 選模型與比較單位**")
        st.caption("DINOv2 embedding；物件級的 pad／解析度／head 由系統自動量測。")
    with c3, st.container(border=True):
        st.markdown("**③ Run**")
        st.caption("物件級出**逐類別漂移表**；整張影像出**整體漂移＋A/B 散點**，"
                   "框選看縮圖。")


def _cmp_object_settings_expander() -> None:
    with st.expander("🔬 物件設定（系統自動尋找，套用到 A／B 兩邊）"):
            import object_eval as oe
            _cur = st.session_state.get("cmp_object_policy")
            st.caption(f"目前設定：**{oe.policy_tag(_cur) if _cur else '尚未量測（首次 Run 會自動找）'}**。"
                       "用兩個資料夾中**較小**的那個探索 pad／解析度／head，再套到 A／B 兩邊。")
            _mdl = st.session_state.get("cmp_model_sel", "")
            if _mdl and not _mdl.startswith("dinov2"):
                st.info("自動調 head（cls／meanpool）僅對 DINOv2 有意義；其他模型主要比 pad／解析度。")
            if st.button("🔄 重新量測", key="cmp_retune", use_container_width=True):
                _fa = st.session_state.get("cmp_folder_a", "").strip()
                _fb = st.session_state.get("cmp_folder_b", "").strip()
                if not (_fa and _fb and Path(_fa).exists() and Path(_fb).exists()):
                    st.warning("先在上方選好 Folder A 與 B。")
                else:
                    _ipa, _ = _cmp_resolve_images(Path(_fa))
                    _ipb, _ = _cmp_resolve_images(Path(_fb))
                    st.session_state.pop("_cmp_objA", None)
                    st.session_state.pop("_cmp_objB", None)
                    # 設定變了 → 丟掉舊結果,逼使用者重新 Run(避免表格與設定不一致)
                    st.session_state.pop("cmpc_rows", None)
                    st.session_state.pop("cmp_result_kind", None)
                    _cmp_autotune(_ipa, _ipb, _mdl or "dinov2_vits14")
                    st.toast("設定已更新，請按 ▶ Run 重新比較。")
                    st.rerun()
            _at = st.session_state.get("cmp_autotune")
            if _at and _at.get("rows"):
                def _pct(x):
                    return None if x is None else round(x * 100, 1)
                _data = []
                for r in _at["rows"]:
                    b = r["buckets"]
                    _data.append({"設定": oe.policy_tag(r["policy"]),
                                  "總體%": _pct(r["macro_purity"]),
                                  "<32": _pct(b["<32"]["purity"]),
                                  "32-96": _pct(b["32-96"]["purity"]),
                                  "96-224": _pct(b["96-224"]["purity"]),
                                  ">224": _pct(b[">224"]["purity"])})
                st.caption(f"（用較小的 Folder {_at.get('which','?')} 量測）"
                           "macro kNN 純度 %（越高越好）")
                st.dataframe(pd.DataFrame(_data), hide_index=True, use_container_width=True)

            st.divider()
            st.caption("✏️ 手動微調（在自動找到的設定上再調細部）")
            _man = _manual_policy_controls(
                _cur or {"pad": 0.12, "target_res": 224, "head": "cls"}, "cmpman")
            if st.button("套用手動設定（清掉結果，請重新 Run）", key="cmp_apply_manual",
                         use_container_width=True):
                st.session_state["cmp_object_policy"] = _man
                st.session_state["cmp_policy_token"] = _cmp_policy_token(
                    _mdl or "dinov2_vits14")
                st.session_state.pop("_cmp_objA", None)
                st.session_state.pop("_cmp_objB", None)
                st.session_state.pop("cmpc_rows", None)
                st.session_state.pop("cmp_result_kind", None)
                st.toast(f"已套用：{oe.policy_tag(_man)}，請按 ▶ Run。")
                st.rerun()


_STATE_COLOR = {
    STATE_EMPTY: "#c0392b", STATE_MISSING: "#e74c3c", STATE_LOW: "#f39c12",
    STATE_HEALTHY: "#2ecc71", STATE_FAKE: "#9b59b6", STATE_OVER: "#3498db",
}
_STATE_Z = {  # 離散色階用的整數編碼
    STATE_EMPTY: 0, STATE_MISSING: 1, STATE_LOW: 2,
    STATE_HEALTHY: 3, STATE_FAKE: 4, STATE_OVER: 5,
}
# 可當軸的「自動計算屬性」（從影像算，不需事先標註）
_AUTO_AXES = ["brightness", "contrast", "sharpness", "aspect"]


def _completeness_axis_values(records: list[dict], axis: str):
    """回傳該軸的 (每筆 bucket index, bucket labels)。類別軸用 records 欄位，
    數值軸用快取的影像統計分桶。"""
    if axis in ("label", "split"):
        return categorical_buckets([r.get(axis, "") for r in records])
    stats = st.session_state.get("cmp_img_stats", [])
    vals = [s.get(axis, 0.0) for s in stats]
    n_bins = int(st.session_state.get("cov_bins", 3))
    return bucketize(vals, n_bins, method="quantile")


def _mine_cell_candidates(cell: dict, records: list[dict], emb: np.ndarray,
                          model: str) -> None:
    """(b) Fill a cell from the candidate pool: embed the pool (cached),
    query by the cell's centroid (or the cell's X-marginal for an empty
    cell), and stash the nearest pool images for review."""
    from completeness import cell_centroid, mine_candidates
    pool_folders = parse_folder_paths(st.session_state.get("cov_pool_text", ""))
    pool_folders = [p for p in pool_folders if p.exists()]
    if not pool_folders:
        st.warning("候選池資料夾不存在。"); return
    pool_records = discover_images_classifier(pool_folders)
    if not pool_records:  # 候選池常是未分類的平鋪資料夾
        pool_paths = []
        for f in pool_folders:
            pool_paths += [p for ext in ("*.jpg", "*.jpeg", "*.png")
                           for p in f.rglob(ext)]
        pool_records = [{"path": p, "split": p.parent.name, "label": ""}
                        for p in sorted(set(pool_paths))]
    if not pool_records:
        st.warning("候選池中找不到影像。"); return

    with st.spinner(f"擷取候選池特徵（{len(pool_records)} 張）…"):
        embed_fn = load_model(model)
        pool_paths = [r["path"] for r in pool_records]
        cache = _dataset_cache_dir(pool_folders[0], f"embeddings_{model}") / "embeddings.npz"
        pool_emb = extract_embeddings(pool_paths, embed_fn, cache_path=cache)

    # query：有樣本用格心；空格退回該 X 標籤（同類）的整體中心
    q = cell_centroid(emb, cell["indices"])
    if q is None:
        q = cell_centroid(emb, [i for i, r in enumerate(records)
                                if r.get("label") == cell["x_label"]])
    if q is None:
        st.warning("此格無可用查詢向量（空格且無同類樣本可當種子）。"); return
    idxs, dists = mine_candidates(pool_emb, q, k=12)
    st.session_state["cov_candidates"] = {
        "cell": (cell["x"], cell["y"]),
        "items": [{"path": str(pool_records[i]["path"]), "d": d}
                  for i, d in zip(idxs, dists)],
    }
    _log_usage("cov_mine", n=len(idxs))


def _render_cov_candidates(cell: dict, records: list[dict]) -> None:
    res = st.session_state.get("cov_candidates")
    if not res or res.get("cell") != (cell["x"], cell["y"]):
        return
    items = res["items"]
    if not items:
        st.info("候選池中沒有夠相似的候選。"); return
    st.caption(f"候選池相似候選（{len(items)} 張，距離小→大）——人工挑選後再進標註/資料集：")
    with st.container(height=480):
        cols = st.columns(3)
        for j, it in enumerate(items):
            with cols[j % 3]:
                thumb = _thumb_or_none(Path(it["path"]))
                if thumb:
                    st.image(thumb, use_container_width=True,
                             caption=f"d={it['d']:.3f}")
                else:
                    st.warning("⚠ 缺檔")
    csv = _rows_to_csv(["path", "distance"],
                       [[it["path"], f'{it["d"]:.6f}'] for it in items])
    st.download_button("⬇ 匯出候選清單 CSV", data=csv,
                       file_name="cell_candidates.csv", mime="text/csv",
                       key="cov_cand_csv", use_container_width=True)


_COV_DEMO_DIR = Path(__file__).parent.parent / "demo" / "imagenette" / "train"
_D_STAR_PRESET = {"寬鬆": 0.45, "標準": 0.6, "嚴格": 0.75}


def _load_cov_demo() -> None:
    st.session_state["cov_folder_list"] = [_demo_classifier_dir()]
    st.session_state["_cov_autorun"] = True
    _log_usage("cov_demo_load")


def _render_cov_quick_start() -> None:
    """冷啟動空狀態：三步卡 + 一鍵 demo（對齊 Visualize 的引導模式）。"""
    st.markdown("##### 快速開始")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, st.container(border=True):
        st.markdown("**① 貼資料夾**")
        st.caption("在左側貼上含「類別子資料夾」的影像資料夾路徑。")
    with c2, st.container(border=True):
        st.markdown("**② 選模型**")
        st.caption("用來算每格內影像的多樣性（偵測近重複充數）。")
    with c3, st.container(border=True):
        st.markdown("**③ 開始分析**")
        st.caption("跑完出現熱力圖；切格方式與達標標準可在圖上方即時調，免重跑。")
    mid = st.columns([2, 1.6, 2])[1]
    mid.button("✨ 用範例資料試跑（imagenette）", key="cov_demo_btn",
               type="primary", use_container_width=True,
               on_click=_load_cov_demo)


# ── 嵌入覆蓋圖（embedding-space coverage / gap-filling）─────────────────────
# 與屬性棋盤互補：密度一律在「原始高維 cosine 空間」用 kNN 距離算，2-D/3-D
# 投影只拿來「畫」高維找出的稀疏區，絕不拿來數密度（正是 completeness.py
# docstring 拒絕 UMAP 網格的理由）。
_COV_PROJ_LABELS = {"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap"}
_COV_SPARSE_PCT = 80   # 自參照百分位門檻：稀疏度落在前 (100-pct)% 視為盲區候選


def _pad_cols(arr: np.ndarray, d: int) -> np.ndarray:
    """確保投影至少有 d 欄（補零軸），2D/3D 散點永遠拿得到對應座標。"""
    arr = np.asarray(arr)
    if arr.shape[1] >= d:
        return arr
    return np.hstack([arr, np.zeros((len(arr), d - arr.shape[1]))])


def _l2norm(arr):
    """L2 正規化每列。投影前一律做 → 用 cosine 方向幾何（與稀疏度/配對一致），
    不被 DINOv2 向量『量級』差異（隨裁切大小/管線變動）把候選整團推到別處。"""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or len(arr) == 0:
        return arr
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)


def _cov_projection(dataset_emb, cand_emb, method, dim, token, cand_token):
    """資料集（+ 候選）合併擬合到同一座標系，兩者位置完全一致（不用近似
    transform）。以 (token, 候選 token, 投影法, 維度) 快取——token 由呼叫端
    依「整圖／物件級」傳入，避免兩種粒度互撞。稀疏度永遠不從這些座標讀——
    只拿來畫圖。回傳 (coords_dataset, coords_candidates|None)。
    """
    has_cand = cand_emb is not None and len(cand_emb) > 0
    key = "|".join([
        token,
        cand_token if has_cand else "-",
        method, str(dim),
    ])
    cache = st.session_state.get("_cov_proj_cache")
    if cache and cache.get("key") == key:
        return cache["coords_d"], cache["coords_c"]
    n_d = len(dataset_emb)
    combined = np.vstack([dataset_emb, cand_emb]) if has_cand else np.asarray(dataset_emb)
    combined = _l2norm(combined)   # 投影用 cosine 幾何，候選才不會整團被推到別處
    n = len(combined)
    n_comps = min(dim, max(1, n - 1))
    if method == "tsne" and n >= 4:
        arr = TSNE(n_components=n_comps, random_state=42,
                   perplexity=min(30, max(1, n - 1))).fit_transform(combined)
    elif method == "umap" and n >= 4:
        arr = _umap().UMAP(n_components=n_comps, n_neighbors=min(15, max(2, n - 1)),
                        random_state=42).fit_transform(combined)
    else:  # PCA，或極小樣本保底
        arr = PCA(n_components=n_comps, random_state=42).fit_transform(combined)
    arr = _pad_cols(np.asarray(arr), dim)
    coords_d, coords_c = arr[:n_d], (arr[n_d:] if has_cand else None)
    st.session_state["_cov_proj_cache"] = {
        "key": key, "coords_d": coords_d, "coords_c": coords_c}
    return coords_d, coords_c


def _cov_project5(emb, cand_emb, labels, cand_labels, mkey, dim, token, cand_token):
    """覆蓋圖投影，支援 5 法。lda/sumap 以類別標籤監督（候選用最近鄰暫定標籤），
    其餘走 _cov_projection。回傳 (coords_dataset, coords_candidates|None)。"""
    if mkey not in _SUPERVISED_METHODS:
        return _cov_projection(emb, cand_emb, mkey, dim, token, cand_token)
    has_c = cand_emb is not None and len(cand_emb) > 0
    key = "|".join([token, cand_token if has_c else "-", mkey, str(dim), "sup"])
    cache = st.session_state.get("_cov_proj_cache")
    if cache and cache.get("key") == key:
        return cache["coords_d"], cache["coords_c"]
    combined = np.vstack([emb, cand_emb]) if has_c else np.asarray(emb)
    combined = _l2norm(combined)   # cosine 幾何
    ylabels = list(labels) + (list(cand_labels) if has_c else [])
    # 監督UMAP spectral 初始化需 n_components+1 < N；樣本太少(如 3D+n=4)會崩 → 退 PCA
    sup = (_supervised_projection(combined, ylabels, mkey, dim)
           if len(combined) > dim + 1 else None)
    if sup is None:   # 類別 <2 / 樣本過少 → 退 PCA
        return _cov_projection(emb, cand_emb, "pca", dim, token, cand_token)
    arr = _pad_cols(np.asarray(sup), dim)
    cd, cc = arr[:len(emb)], (arr[len(emb):] if has_c else None)
    st.session_state["_cov_proj_cache"] = {"key": key, "coords_d": cd, "coords_c": cc}
    return cd, cc


def _cov_sparsity(emb, k, token):
    """高維稀疏度（到 k 近鄰平均 cosine 距離），以 (token, k) 快取。"""
    key = f"{token}|{k}"
    cache = st.session_state.get("_cov_sparsity")
    if cache and cache.get("key") == key:
        return cache["scores"]
    scores = sparsity_scores(emb, k=int(k))
    st.session_state["_cov_sparsity"] = {"key": key, "scores": scores}
    return scores


def _cov_radius(emb, token):
    """N2 命中密度用的半徑＝資料集 1-NN cosine 距離 P75（與體檢卡一致）。"""
    key = token
    cache = st.session_state.get("_cov_radius")
    if cache and cache.get("key") == key:
        return cache["radius"]
    from sklearn.neighbors import NearestNeighbors
    emb = np.asarray(emb)
    if len(emb) < 2:
        radius = 0.0
    else:
        nn = NearestNeighbors(metric="cosine", n_neighbors=2).fit(emb)
        dist, _ = nn.kneighbors(emb)
        radius = float(np.percentile(dist[:, 1], 75))
    st.session_state["_cov_radius"] = {"key": key, "radius": radius}
    return radius


def _cov_candidate_image_records(folders: list[Path]) -> list[dict]:
    """候選資料夾的影像清單：先試 類別子資料夾，否則平鋪掃描。"""
    recs = discover_images_classifier(folders)
    if not recs:
        paths: list[Path] = []
        for f in folders:
            paths += [p for ext in ("*.jpg", "*.jpeg", "*.png") for p in f.rglob(ext)]
        recs = [{"path": p, "split": p.parent.name, "label": ""}
                for p in sorted(set(paths))]
    return recs


def _cov_embed_candidates(model: str) -> None:
    """投影新候選資料夾進此空間：擷取（快取）特徵，存進 session。粒度跟隨
    目前『分析單位』——物件級時改裁候選資料夾的 YOLO 物件再各自算特徵。"""
    folders = [p for p in parse_folder_paths(
        st.session_state.get("cov_cand_text", "")) if p.exists()]
    if not folders:
        st.warning("候選資料夾不存在。"); return
    cand_records = _cov_candidate_image_records(folders)
    if not cand_records:
        st.warning("候選資料夾中找不到影像。"); return
    is_obj = st.session_state.get("cov_granularity") == "物件級（YOLO）"
    if is_obj:
        pad = float((st.session_state.get("objcov_policy") or {}).get("pad", 0.12))
        cnames = (_manual_class_names("cov")       # 手動指定 > 兩層解析(id→名對齊)
                  or _cov_class_names(cand_records)
                  or st.session_state.get("cov_obj_class_names"))
        obj_records, cand_emb, _ = _crop_and_embed_objects(
            cand_records, model, cnames, pad,
            base_token="cand|" + repr(sorted(str(f) for f in folders)),
            session_key="_cov_obj_cand", spinner="裁切候選物件")
        if not obj_records:
            st.warning("候選資料夾的 labels/ 找不到 bbox；請改『整張影像』或先補標註。")
            return
        cand_records = obj_records
    else:
        with st.spinner(f"擷取候選特徵（{len(cand_records)} 張）…"):
            embed_fn = load_model(model)
            cache = _dataset_cache_dir(folders[0], f"embeddings_{model}") / "embeddings.npz"
            cand_emb = extract_embeddings(
                [r["path"] for r in cand_records], embed_fn, cache_path=cache)
    st.session_state["cov_cand_records"] = cand_records
    st.session_state["cov_cand_emb"] = cand_emb
    st.session_state["cov_cand_token"] = uuid.uuid4().hex
    st.session_state.pop("_cov_proj_cache", None)
    unit = "物件" if is_obj else "張"
    st.toast(f"已投影 {len(cand_records)} {unit}候選進此空間", icon="🧭")
    _log_usage("cov_project_candidates", n=len(cand_records))


def _cov_send_to_quiz(quiz_records, scores, class_opts) -> None:
    """送補洞候選進組考卷：寫入 quiz 的 inbound session keys 並切到考卷頁。"""
    st.session_state["quiz_records"] = quiz_records
    st.session_state["quiz_disagreement"] = np.asarray(scores, dtype=float)
    st.session_state["quiz_class_opts"] = class_opts
    st.session_state["quiz_inbound"] = True
    for k in ("quiz_spec", "quiz_answers", "quiz_pos"):
        st.session_state.pop(k, None)
    st.session_state["tool_switch"] = "組考卷"
    _log_usage("cov_send_to_quiz", n=len(quiz_records))


def _build_cov_scatter(coords_d, sparsity, records, dim,
                       coords_c, cand_records, ranked_idx, ranked_scores,
                       color_mode="class"):
    """流形散點。color_mode='class'：顏色＝類別（與下方縮圖底色一致）、點大小＝稀疏度
    （大＝稀疏盲區）；color_mode='sparsity'：顏色＝高維稀疏度（Turbo）。候選為深色菱形。"""
    use_3d = dim == 3 and coords_d.shape[1] >= 3
    scatter = go.Scatter3d if use_3d else (
        go.Scattergl if len(coords_d) > _SCATTERGL_THRESHOLD else go.Scatter)

    def _xyz(coords):
        d = {"x": coords[:, 0].tolist(), "y": coords[:, 1].tolist()}
        if use_3d:
            d["z"] = coords[:, 2].tolist()
        return d

    _sp = np.asarray(sparsity, dtype=float)
    if color_mode == "class":
        cmap = _viz_color_map(records)
        _rng = float(_sp.max() - _sp.min()) if len(_sp) else 0.0
        _norm = (_sp - _sp.min()) / _rng if _rng > 1e-12 else np.zeros_like(_sp)
        _base = 4 if use_3d else 6
        _marker = dict(size=(_base + (4 if use_3d else 8) * _norm).tolist(),
                       color=[cmap.get(r.get("label", ""), "#9aa0a6") for r in records],
                       opacity=0.85)
    else:
        _marker = dict(size=4 if use_3d else 7, color=_sp.tolist(),
                       colorscale="Turbo", showscale=True,
                       colorbar=dict(title="稀疏度"), opacity=0.85)
    data = [scatter(
        **_xyz(coords_d), mode="markers", name="資料集", marker=_marker,
        customdata=[[i] for i in range(len(records))],  # 框選→record index→縮圖
        text=[f"{Path(r['path']).name}<br>{r.get('label', '')}（{r.get('split', '')}）"
              f"<br>稀疏度 {float(_sp[i]):.3f}" for i, r in enumerate(records)],
        hovertemplate="%{text}<extra></extra>",
    )]
    if coords_c is not None and cand_records:
        score_by_idx = dict(zip(ranked_idx, ranked_scores))
        cs = np.asarray([score_by_idx.get(i, 0.0) for i in range(len(cand_records))])
        rng = cs.max() - cs.min()
        norm = (cs - cs.min()) / rng if rng > 1e-12 else np.zeros_like(cs)
        base, span = (3, 3) if use_3d else (5, 5)   # 候選菱形小一點
        data.append(scatter(
            **_xyz(coords_c), mode="markers", name="候選（新資料夾）",
            marker=dict(size=(base + span * norm).tolist(), symbol="diamond",
                        color="#111111", line=dict(width=1, color="#ffffff"),
                        opacity=0.9),
            # 候選用 sentinel customdata（i + _CMPC_BOFF）→ 框選可分辨並看縮圖
            customdata=[[i + _CMPC_BOFF] for i in range(len(cand_records))],
            text=[f"{Path(r['path']).name}<br>補洞分數 {score_by_idx.get(i, 0.0):.3f}"
                  for i, r in enumerate(cand_records)],
            hovertemplate="%{text}<extra></extra>",
        ))
    fig = go.Figure(data=data)
    _title = ("嵌入特徵空間 · 顏色＝類別、點大小＝稀疏度（大＝盲區）"
              if color_mode == "class"
              else "嵌入特徵空間 · 顏色＝高維稀疏度（亮＝稀疏盲區）")
    layout = dict(height=560, margin=dict(l=10, r=10, t=34, b=10),
                  title=_title,
                  legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    if use_3d:
        layout["scene"] = dict(xaxis_title="C1", yaxis_title="C2", zaxis_title="C3")
    else:
        # 預設拖曳＝框選（不是縮放），讓「拉一群點看縮圖」一拖就中
        layout.update(xaxis_title="Component 1", yaxis_title="Component 2",
                      dragmode="select")
    fig.update_layout(**layout)
    return fig


def _lv_cache_dir() -> Path:
    """App 端快取根目錄——**不寫進 user dataset**,避免污染對方的 check-in。
    可用環境變數 LV_CACHE_DIR 覆寫;預設在本 repo 的 .lv_cache/（已 gitignore）。"""
    return Path(os.environ.get("LV_CACHE_DIR")
                or (Path(__file__).resolve().parent.parent / ".lv_cache"))


def _dataset_cache_dir(folder: Path, kind: str) -> Path:
    """某資料夾的 app 端快取目錄,以絕對路徑雜湊命名 → 同資料夾穩定共用、不同
    資料夾不衝突。kind 例:'object_crops'、'embeddings_<model>'。"""
    folder = Path(folder).resolve()
    key = hashlib.sha1(str(folder).encode("utf-8")).hexdigest()[:10]
    return _lv_cache_dir() / f"{folder.name}_{key}" / kind


def _cov_object_root(records: list[dict]) -> Path:
    """資料集根目錄（YOLO 為 <root>/images/x.jpg → <root>），物件 crop 寫這底下。"""
    img0 = Path(records[0]["path"])
    return img0.parent.parent if img0.parent.parent != img0.parent else img0.parent


def _cov_class_names(records: list[dict]) -> list[str] | None:
    """從 classes.txt 解析 YOLO 類別名（試資料集根與其上層）。"""
    if not records:
        return None
    root = _cov_object_root(records)
    for folder in (root, root.parent):
        names = read_classes_txt(folder)
        if names:
            return names
    return None


def _is_detection_dataset(records: list[dict], probe: int = 25) -> bool:
    """目前載入的影像是否帶偵測標註 → 可做物件級覆蓋。
    認 YOLO labels/*.txt,也認 COCO JSON / VOC XML / LabelMe / NDJSON
    (discover_yolo_objects 原生支援)。"""
    for r in records[:probe]:
        p = Path(r["path"])
        if yolo_label_path_for(p).exists():
            return True
        if annotation_boxes_for_image(p) is not None:
            return True
    return False


def _cov_class_src(records: list[dict]) -> Path | None:
    """類別名實際命中的檔案路徑（摘要卡顯示「類別來源」用）：classes.txt > data.yaml。"""
    root = _cov_object_root(records)
    for folder in (root, root.parent):
        cf = folder.parent / "classes.txt"
        if cf.exists() and [ln for ln in safe_read_text(cf).splitlines() if ln.strip()]:
            return cf
        yf = folder.parent / "data.yaml"
        if yaml_class_names(yf):
            return yf
    return None


def _cov_goto_reference() -> None:
    """縫合『缺口→補』：一鍵切到 嵌入覆蓋圖・覆蓋散點＋物件級＋參照分佈，待使用者載入候選。
    整類缺失補洞要在物件級才精準（候選帶 label、可按類聚合）。"""
    st.session_state["cov_view_mode"] = "嵌入覆蓋圖"
    st.session_state["cov_emb_submode"] = "覆蓋散點"
    st.session_state["cov_granularity"] = "物件級（YOLO）"
    st.session_state["cov_b_role"] = "參照分佈"
    st.toast("已切到『物件級・參照分佈』：請在右側載入候選資料夾（如 test）來量／補缺口",
             icon="🧭")


def _cov_summary_card(records: list[dict]) -> None:
    """資料集摘要卡——進完整度即見：張數／物件數／每類分佈／不均衡／整類缺失／類別來源。
    整類缺失是自指稀疏度的盲區，這裡直接點名並縫到『參照分佈』去量／補。"""
    import collections
    n_img = len(records)
    class_names = _manual_class_names("cov") or _cov_class_names(records)
    with st.container(border=True):
        if _is_detection_dataset(records) and class_names:
            cnt: collections.Counter = collections.Counter()
            for r in records:
                lp = yolo_label_path_for(Path(r["path"]))
                if not lp.exists():
                    continue
                for ln in safe_read_text(lp).splitlines():
                    ps = ln.split()
                    if ps:
                        try:
                            cnt[int(float(ps[0]))] += 1
                        except ValueError:
                            pass
            present = {i for i in cnt if cnt[i] > 0}
            missing = [class_names[i] for i in range(len(class_names))
                       if i not in present]
            pc = [cnt[i] for i in present]
            imbal = (max(pc) / max(min(pc), 1)) if pc else 0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("影像", n_img)
            c2.metric("物件", sum(cnt.values()))
            c3.metric("類別 出現/全集", f"{len(present)}/{len(class_names)}")
            c4.metric("最不均衡", f"{imbal:.0f}:1" if pc else "—")
            if missing:
                st.warning(
                    f"⚠ **整類缺失**：classes.txt 有 {len(class_names)} 類，"
                    f"本資料集 0 個的有 **{'、'.join(missing)}**。"
                    "自指稀疏度看不到這種洞——要量／補它，需用候選資料夾＋『參照分佈』。")
                st.button("➜ 用候選資料夾量這些缺口（切到參照分佈）",
                          key="cov_sum_goto_ref", on_click=_cov_goto_reference)
            with st.expander(f"每類物件數（出現 {len(present)} / 全集 {len(class_names)}）"):
                ser = {class_names[i]: int(cnt.get(i, 0))
                       for i in range(len(class_names))}
                st.bar_chart(pd.Series(ser, name="物件數"))
        else:
            labs = collections.Counter(r.get("label", "") for r in records)
            c1, c2 = st.columns(2)
            c1.metric("影像", n_img)
            c2.metric("類別", len(labs))
            with st.expander(f"每類影像數（{len(labs)} 類）"):
                st.bar_chart(pd.Series(dict(labs), name="影像數"))
        src = _cov_class_src(records)
        st.caption(
            f"類別來源：`{src}`" if src else
            "類別來源：未找到 classes.txt（類別以編號顯示）；可在側欄手動指定。")


def _crop_and_embed_objects(records, model, class_names, pad, *, base_token,
                            session_key="_cov_obj", crops_subdir="object_crops",
                            spinner="裁切物件", policy=None, progress_cb=None):
    """把 records 裡每個 YOLO bbox 各自算 embedding（一個物件一個點）。

    embedding 取自「原圖即時裁切」的記憶體影像（不讀回 q88 JPEG，避免雙重壓縮傷
    小物件）並以保長寬比、multiples-of-14 的方式 resize（不擠壓）；context 邊距隨
    物件大小自適應（clamp 到 8–64px）。crop JPEG 仍會落地，但只供 UI 縮圖用。
    以 (base_token, model, pad, n) 在 session 快取；crop 檔與 npz 皆落地、第二次
    極快。回傳 (object_records, object_emb, token)；object_records 的 path＝crop 檔、
    另帶 image_path / bbox / label(類別) / class_id。"""
    # policy = {pad, target_res, head}；換策略即換目錄/npz/seed，舊版快取自然作廢
    policy = policy or {}
    pad = float(policy.get("pad", pad))
    target_res = int(policy.get("target_res", 224))
    head = str(policy.get("head", "cls"))
    ptag = f"pad{int(round(pad * 100))}_r{target_res}_{head}_arv2"
    seed = repr((base_token, model, round(pad, 3), target_res, head, len(records)))
    # NOTE: 物件大小/信心過濾**不在這裡**做——改在各工具「渲染散點時」用 _obj_keep
    # 過濾（見 _obj_keep / _obj_view_indices）。理由：Visualize/Compare 把整批結果存進
    # session（Run-gated），若在此先濾，調低門檻時被丟掉的物件無法回來；渲染時過濾才能
    # 即時、可回復。本函式一律回傳「完整集合」（每個 record 帶 short_px 與 score）。
    cached = st.session_state.get(session_key)
    if cached and cached.get("seed") == seed:
        return cached["records"], cached["emb"], cached["token"]
    image_paths = [Path(r["path"]) for r in records]
    split_by = {str(Path(r["path"])): r.get("split", "") for r in records}
    meta = discover_yolo_objects(image_paths, class_names)
    if not meta:
        empty = {"seed": seed, "records": [], "emb": np.zeros((0, 1)), "token": ""}
        st.session_state[session_key] = empty
        return [], np.zeros((0, 1)), ""
    crops_dir = _dataset_cache_dir(_cov_object_root(records), crops_subdir) / ptag
    crops_dir.mkdir(parents=True, exist_ok=True)

    def _adaptive_pad_px(bbox, iw, ih) -> int:
        long_px = max(bbox[2] * iw, bbox[3] * ih)
        return int(min(64, max(8, round(pad * long_px))))

    # 1-image source cache: meta is image-then-box order, so consecutive objects
    # share an original — open it once.
    _src: dict[str, object] = {"ip": None, "img": None}

    def _open(ip):
        if str(ip) != _src["ip"]:
            _src["img"] = safe_open_image(ip)  # 壞圖/超大圖回 None → 下游跳過該物件
            _src["ip"] = str(ip)
        return _src["img"]

    _imwh: dict[str, tuple[int, int]] = {}

    def _img_wh(ip) -> tuple[int, int]:
        """源圖寬高 → 供算物件源像素短邊（裁切可靠度）；壞圖回 (0, 0)。"""
        k = str(ip)
        if k not in _imwh:
            _im = safe_open_image(ip, mode=None)
            _imwh[k] = _im.size if _im is not None else (0, 0)
        return _imwh[k]

    obj_records: list[dict] = []
    crop_paths: list[Path] = []
    spec_by_path: dict[Path, tuple] = {}  # crop_path -> (image_path, bbox)
    with st.spinner(f"{spinner}（{len(meta)} 個物件）…"):
        for _i, o in enumerate(meta):
            if progress_cb:
                progress_cb(_i + 1, len(meta), "crop")
            ip = o["image_path"]
            out = crops_dir / f"{ip.stem}__obj{o['obj_index']}.jpg"
            if not out.exists():
                img = _open(ip)
                if img is None:
                    continue
                try:
                    pad_px = _adaptive_pad_px(o["bbox"], *img.size)
                    crop_bbox(img, *o["bbox"], pad_px=pad_px).save(out, quality=88)
                except (OSError, ValueError):
                    continue
            crop_paths.append(out)
            spec_by_path[out] = (ip, o["bbox"])
            _iw, _ih = _img_wh(ip)
            obj_records.append({
                "path": out, "image_path": ip, "split": split_by.get(str(ip), ""),
                "label": o["label"], "class_id": o["class_id"],
                "bbox": o["bbox"], "obj_index": o["obj_index"],
                "short_px": int(round(min(o["bbox"][2] * _iw, o["bbox"][3] * _ih))),
                "score": o.get("score"),   # 6 欄 label 的 conf；GT(5 欄)為 None
            })
        # Embed from the in-memory crop of the FULL-RES original (never the JPEG).
        embed_fn = load_model(model, keep_aspect=True,
                              target_res=target_res, head=head)

        def _embed(crop_path):
            ip, bbox = spec_by_path[crop_path]
            img = _open(ip)
            if img is None:
                return np.zeros(384, dtype=np.float32)
            crop = crop_bbox(img, *bbox, pad_px=_adaptive_pad_px(bbox, *img.size))
            return embed_fn(crop)

        # ── 增量 + 定期存檔 ──────────────────────────────────────────────
        # 逐物件算,key = crop 檔名(穩定);已算過的直接重用,每 200 筆 atomic 存檔。
        # → 中途當掉只需補算缺的;新增資料也只算新物件,不會整批重跑。
        cache = crops_dir / f"embeddings_{model}.npz"
        keys = [cp.stem for cp in crop_paths]
        cached: dict = {}
        if cache.exists():
            try:
                with np.load(str(cache), allow_pickle=False) as _d:  # close handle (Windows replace)
                    if "emb" in _d.files and "keys" in _d.files:          # new incremental format
                        _e = _d["emb"]
                        cached = {k: _e[i] for i, k in enumerate(_d["keys"].tolist())}
                    elif "embeddings" in _d.files and "filenames" in _d.files:  # old format → reuse
                        _e = _d["embeddings"]
                        cached = {Path(fn).stem: _e[i]
                                  for i, fn in enumerate(_d["filenames"].tolist())}
            except (OSError, ValueError, KeyError):
                cached = {}

        def _flush_cache() -> None:
            if not cached:
                return
            ks = list(cached.keys())
            arr = np.stack([cached[k] for k in ks]).astype(np.float32)
            tmp = cache.with_name(cache.stem + ".part.npz")  # 須以 .npz 結尾(np.savez 否則自動補)
            np.savez(str(tmp), emb=arr, keys=np.array(ks))
            tmp.replace(cache)  # atomic：當掉也不會留半個壞檔

        total = len(keys)
        done = sum(1 for k in keys if k in cached)
        if progress_cb:
            progress_cb(done, total, "embed")
        since = 0
        for i, k in enumerate(keys):
            if k in cached:
                continue
            cached[k] = np.asarray(_embed(crop_paths[i]), dtype=np.float32)
            done += 1
            since += 1
            if progress_cb:
                progress_cb(done, total, "embed")
            if since >= 200:
                _flush_cache()
                since = 0
        if since:
            _flush_cache()
        emb = np.stack([cached[k] for k in keys]) if keys else np.zeros((0, 1))
    token = uuid.uuid4().hex
    st.session_state[session_key] = {
        "seed": seed, "records": obj_records, "emb": emb, "token": token}
    return obj_records, emb, token


def _render_coverage_view(records: list[dict], emb: np.ndarray, model: str) -> None:
    """嵌入覆蓋圖：在特徵空間找稀疏盲區 → 投影新資料夾 → 排補洞候選 → 送考卷。"""
    # ── 分析單位：整張影像 vs 物件級（偵測資料集才有「物件級」可選）──
    is_det = _is_detection_dataset(records)
    granularity = "整張影像"
    class_names: list[str] | None = None
    # 物件外擴(pad)由系統自動找(共用 objcov_policy)，不再手動輸入
    pad = float((st.session_state.get("objcov_policy") or {}).get("pad", 0.12))
    if is_det:
        granularity = st.radio(
            "分析單位", ["物件級（YOLO）", "整張影像"], horizontal=True,
            key="cov_granularity",
            help="物件級＝讀 labels/ 每個框→裁出物件各自算 embedding（顏色＝類別），"
                 "稀疏對應『某類物件的某種樣態收太少』；整張影像＝每張圖一個點。")
    is_obj = is_det and granularity == "物件級（YOLO）"

    # 切換粒度＝丟掉另一種粒度殘留的候選/選取/快取
    if st.session_state.get("_cov_gran_prev") not in (None, granularity):
        for kk in ("cov_cand_emb", "cov_cand_records", "cov_cand_token",
                   "_cov_proj_cache", "cov_sel"):
            st.session_state.pop(kk, None)
    st.session_state["_cov_gran_prev"] = granularity

    active_token = st.session_state.get("cov_token", "")
    if is_obj:
        class_names = _manual_class_names("cov") or _cov_class_names(records)
        st.session_state["cov_obj_class_names"] = class_names
        obj_records, obj_emb, obj_token = _crop_and_embed_objects(
            records, model, class_names, pad,
            base_token=st.session_state.get("cov_token", ""))
        if not obj_records:
            st.warning("這個資料集的 labels/ 裡找不到任何 bbox；已切回『整張影像』。"
                       + _seg_format_hint([r["path"] for r in records]))
            is_obj = False
        else:
            records, emb, active_token = obj_records, obj_emb, obj_token

    n = len(emb)
    labels = [r.get("label", "") for r in records]
    class_opts = sorted({r.get("label", "") for r in records})

    # 物件類別篩選：對某一類做覆蓋分析（物件級最有意義）。_cov_cls 之後也用來篩候選。
    _cov_cls = None
    if is_obj and len(class_opts) > 1:
        _cls = st.selectbox("物件類別（選一類分析；（全部）＝不篩選）",
                            ["（全部）"] + class_opts, key="cov_class_filter")
        if _cls != "（全部）":
            _cov_cls = _cls
            _keep = [i for i, l in enumerate(labels) if l == _cls]
            records = [records[i] for i in _keep]
            emb = np.asarray(emb)[_keep]
            labels = [labels[i] for i in _keep]
            n = len(emb)
            class_opts = [_cls]
            active_token = f"{active_token}|cls={_cls}"

    # 投影方法來自左側工具列（5 法，含監督）；維度／稀疏度 k 在這裡即時調
    method_lbl = st.session_state.get("cov_proj_method", "監督UMAP")
    if method_lbl not in _METHOD_KEY:
        method_lbl = "PCA"
    method = _METHOD_KEY[method_lbl]
    if method in _SUPERVISED_METHODS:
        st.caption(":gray[ℹ 監督投影：依類別標籤排版的視覺輔助（候選用最近鄰暫定類別），"
                   "分離不代表真實差異。]")
    c2, c3 = st.columns(2)
    dim = 3 if c2.radio("維度", ["2D", "3D"], horizontal=True,
                        key="cov_proj_dim") == "3D" else 2
    k = c3.number_input("稀疏度 k", min_value=1, max_value=max(1, n - 1),
                        value=min(10, max(1, n - 1)), key="cov_sparsity_k",
                        help="到 k 個最近鄰的平均 cosine 距離＝稀疏度（高維算）。")
    if is_obj:
        _obj_filter_controls("cov", records)
        records, emb, active_token = _obj_filter_view(records, emb, active_token)
        labels = [r.get("label", "") for r in records]  # 濾後 labels 要跟 emb 同步！
        n = len(emb)
    k = min(int(k), max(1, n - 1))

    sparsity = _cov_sparsity(emb, int(k), active_token)
    cand_emb = st.session_state.get("cov_cand_emb")
    cand_records = st.session_state.get("cov_cand_records")
    # 候選也篩到主資料同一類別（修正：之前候選沒被類別篩選，會混進別類）
    if _cov_cls and cand_records is not None and cand_emb is not None:
        _ck = [i for i, r in enumerate(cand_records) if r.get("label", "") == _cov_cls]
        cand_emb = np.asarray(cand_emb)[_ck] if _ck else None
        cand_records = [cand_records[i] for i in _ck]
    has_cand = cand_emb is not None and len(cand_emb) > 0
    cand_token = (st.session_state.get("cov_cand_token", "")
                  + (f"|cls={_cov_cls}" if _cov_cls else ""))
    ranked_idx: list[int] = []
    ranked_scores: list[float] = []
    provisional: list[str] = []
    if has_cand:
        ranked_idx, ranked_scores = rank_gap_fillers(cand_emb, emb, k=int(k))
        provisional = nearest_labels(cand_emb, emb, labels)
    coords_d, coords_c = _cov_project5(emb, cand_emb, labels, provisional, method, dim,
                                       active_token, cand_token)

    thr = float(np.percentile(sparsity, _COV_SPARSE_PCT)) if len(sparsity) else 0.0
    n_sparse = int(np.sum(np.asarray(sparsity) >= thr)) if len(sparsity) else 0

    col_map, col_side = st.columns([5, 3], gap="medium")
    with col_map:
        mm1, mm2, mm3 = st.columns(3)
        mm1.metric("物件數" if is_obj else "樣本數", n)
        mm2.metric(f"稀疏點（前 {100 - _COV_SPARSE_PCT}%）", n_sparse,
                   help="稀疏度在自參照百分位門檻以上的點＝覆蓋盲區候選。")
        mm3.metric("候選", len(cand_records) if cand_records else 0)
        st.caption(":orange[⚠ 稀疏＝相對你自己資料的密度（未校正真實分佈）；"
                   "全資料集都缺的類型不會顯示為稀疏。稀疏是『該補的嫌疑』，"
                   "不等於模型一定弱——下方根因會判斷補資料是否真有效。]")
        if len(set(labels)) > 1:
            _cmode = "class" if st.radio(
                "上色依據", ["類別（與縮圖同色）", "稀疏度"], horizontal=True,
                key="cov_color_mode") == "類別（與縮圖同色）" else "sparsity"
        else:
            # 單一類別（已篩選）→ 類別色一致無意義，改以稀疏度上色（亮＝盲區）
            _cmode = "sparsity"
            st.caption(":gray[單一類別 → 以稀疏度上色（亮＝盲區）；點大小／顏色＝稀疏度。]")
        fig = _build_cov_scatter(coords_d, sparsity, records, dim,
                                 coords_c, cand_records, ranked_idx, ranked_scores,
                                 color_mode=_cmode)
        unit = "物件" if is_obj else "張"
        cov_sel = st.session_state.get("cov_sel", {})
        if cov_sel.get("token") != active_token:
            cov_sel = {"token": active_token, "indices": [], "cand": []}
        cov_sel.setdefault("cand", [])

        # 取消框選貼在散點正上方（對齊 Visualize／Compare）；用 placeholder 佔位，
        # 等框選事件處理完、選取定案再填入 → 鈕上顯示正確張數＆啟用狀態。
        _cov_cn = st.session_state.get("_cov_clear_nonce", 0)
        _tb1, _tb2 = st.columns([5, 1])
        _tb1.caption("💡 在散點上拖曳框選／套索／點一群點 → 下方看縮圖＋標籤。")
        _clear_slot = _tb2.empty()

        if dim == 2:
            event = st.plotly_chart(
                fig, use_container_width=True,
                key=f"cov_emb_scatter_{_cov_cn}_{_obj_filter_tag()}",
                on_select="rerun", selection_mode=("points", "box", "lasso"))
            sel_pts: list[dict] = []
            if event is not None:
                so = event.get("selection") if hasattr(event, "get") else None
                if so:
                    sel_pts = list(so.get("points", []))
            _all = selection_points_to_indices(sel_pts)
            picked = [c for c in _all if c < _CMPC_BOFF]                       # 主資料
            picked_cand = sorted({c - _CMPC_BOFF for c in _all if c >= _CMPC_BOFF})  # 候選
            # 單向資料流（對齊 Visualize）：只有「非空」框選事件能改寫選取；
            # 清空一律走上方 ✕ 取消框選（空事件不得把既有選取洗掉）。
            if (picked or picked_cand) and (picked, picked_cand) != (
                    cov_sel.get("indices"), cov_sel.get("cand")):
                cov_sel = {"token": active_token, "indices": picked, "cand": picked_cand}
                _nt = len(picked) + len(picked_cand)
                st.toast(f"已框選 {_nt}（主 {len(picked)}／候選 {len(picked_cand)}）",
                         icon="🖼")
        else:
            st.plotly_chart(fig, use_container_width=True, key="cov_emb_scatter_3d")
            st.caption(":orange[ℹ 3D 不支援框選；要拖框選一群看縮圖請切回 2D。]")
        st.session_state["cov_sel"] = cov_sel
        sel_idx = cov_sel["indices"]
        sel_cand = cov_sel.get("cand", [])
        # 選取定案 → 填入上方佔位的「取消框選」鈕（顯示正確張數＆啟用狀態）。
        _nsel = len(sel_idx) + len(sel_cand)
        _clear_slot.button(
            f"✕ 取消框選（{_nsel}）" if _nsel else "✕ 取消框選",
            key="cov_sel_clear", use_container_width=True, disabled=not _nsel,
            on_click=_cov_clear_selection, args=(f"cov_emb_scatter_{_cov_cn}",))
        st.caption(":gray[ℹ 2-D/3-D 佈局僅供定位、非密度量尺；換投影法或載入候選"
                   "位置會變，但稀疏度數字不變（一律高維算）。]")

        # ── 看影像：左＝主資料、右＝候選資料夾（並排對照）──
        noun = "物件" if is_obj else "影像"
        ghc = st.columns([1, 1, 4])
        gal_n = int(ghc[0].number_input("張數", 3, 60, 12, key="cov_gal_n",
                                        label_visibility="collapsed"))
        full_ctx = is_obj and ghc[1].checkbox(
            "原圖(含框)", key="cov_gal_fullimg",
            help="縮圖改顯示物件在整張圖的位置脈絡（紅框為該圖所有標註）。")
        _cmap = _viz_color_map(records)   # 類別→色，與散點同一套
        gL, gR = st.columns(2, gap="medium")

        with gL:   # 左：主資料
            if sel_idx:
                st.markdown(f"**🖼 主資料 · 你框選的 {len(sel_idx)} {unit}**")
                base_idx = sel_idx
            else:
                st.markdown("**🖼 主資料 · 最稀疏盲區（由稀到密）**")
                base_idx = [int(i) for i in np.argsort(sparsity)[::-1]]
            view_idx = base_idx[:gal_n]
            if not view_idx:
                st.caption("目前沒有可顯示的影像。")
            else:
                st.button(f"🛒 加入購物車（{len(view_idx)} {noun}）",
                          key="cov_add_cart", use_container_width=True,
                          on_click=_batch_add,
                          args=(records, view_idx, "sparse",
                                {i: float(sparsity[i]) for i in view_idx}))
                with st.container(height=520):
                    _gc = st.columns(3)
                    for j, i in enumerate(view_idx):
                        with _gc[j % 3]:
                            rec = records[i]
                            lbl = rec.get("label", "") or "?"
                            col = _cmap.get(lbl, "#9aa0a6")
                            cap = f"{lbl}｜{_rec_fname(rec)} · 稀疏{float(sparsity[i]):.3f}"
                            if full_ctx:
                                ip = Path(rec["image_path"])
                                img = draw_yolo_boxes(ip, yolo_label_path_for(ip),
                                                      class_names)
                                if img is None:  # 壞圖 → 略過，不可把 None 當 Image
                                    st.warning("⚠ 壞檔")
                                else:
                                    img.thumbnail((240, 240))
                                    st.image(img, use_container_width=True)
                            else:
                                thumb = _thumb_or_none(Path(rec["path"]))
                                if thumb:
                                    st.image(thumb, use_container_width=True)
                                else:
                                    st.warning("⚠ 缺檔")
                            st.markdown(
                                f"<div style='background:{col};color:{_text_on(col)};"
                                f"font-size:11px;text-align:center;border-radius:4px;"
                                f"padding:1px 4px;margin-top:2px'>{cap}</div>",
                                unsafe_allow_html=True)

        with gR:   # 右：候選資料夾（與主資料同一類別）
            _clbl = _cov_cls or "全部類別"
            st.markdown(f"**🛒 候選・{_clbl} · 框選 {len(sel_cand)} {unit}**")
            if not has_cand:
                st.caption("先在側欄載入候選資料夾並『🧭 投影候選進此空間』。")
            elif not sel_cand:
                st.caption("在左圖框選黑菱形（候選）→ 這裡顯示你要補進來的物件。")
            else:
                _sc = [i for i in sel_cand if 0 <= i < len(cand_records)]
                st.button(f"加入購物車（{len(_sc)} {noun}）", key="cov_add_cand_cart",
                          use_container_width=True, type="primary",
                          on_click=_batch_add, args=(cand_records, _sc, "gap_filler"))
                _ccn = class_names
                if full_ctx and cand_records and cand_records[0].get("image_path"):
                    _ccn = (_manual_class_names("cov") or _cov_class_names(
                        [{"path": str(cand_records[0]["image_path"])}]) or class_names)
                with st.container(height=520):
                    _gc2 = st.columns(3)
                    for _j, _i in enumerate(_sc[:24]):
                        with _gc2[_j % 3]:
                            _cr = cand_records[_i]
                            if full_ctx and _cr.get("image_path"):
                                _ip = Path(_cr["image_path"])
                                _img = draw_yolo_boxes(_ip, yolo_label_path_for(_ip), _ccn)
                                if _img is None:  # 壞圖 → 略過，不可把 None 當 Image
                                    st.warning("⚠ 壞檔")
                                else:
                                    _img.thumbnail((240, 240))
                                    st.image(_img, use_container_width=True)
                            else:
                                _th = _thumb_or_none(Path(_cr["path"]))
                                if _th:
                                    st.image(_th, use_container_width=True)
                                else:
                                    st.warning("⚠ 缺檔")
                            st.caption(f"{_cr.get('label') or '候選'}｜{_rec_fname(_cr)}")

        # Phase B：H1–H5 根因——這些稀疏區補資料到底有沒有用
        with st.expander("🧭 根因：這些稀疏區補資料有沒有用？（H1–H5 誠實閘）"):
            s1 = st.slider(
                "S1 此概念的人類一致性（來自組考卷 / gauge study）", 0.0, 1.0,
                float(st.session_state.get("quiz_last_consistency", 0.9)), 0.01,
                key="cov_s1",
                help="量『人』：低於門檻＝定義歧義（H2），補資料不會收斂。"
                     "跑過組考卷會自動帶入其自我一致率。")
            radius = _cov_radius(emb, active_token)
            sparse_idx = [int(i) for i in np.argsort(sparsity)[::-1][:min(50, n)]
                          if sparsity[i] >= thr]
            if not sparse_idx:
                st.caption("目前沒有稀疏點。")
            elif n < 30:
                st.info(f"目前只有 {n} 個樣本，根因（H1–H5）在小樣本上很不穩、容易誤判"
                        "『補資料無效』。請先擴量到數十個以上再看根因結論。")
            else:
                _sig_memo: dict[int, str] = {}

                def _sig_for(i: int) -> str:
                    if i not in _sig_memo:
                        _sig_memo[i] = _signal_level_for_record(records[i])[0]
                    return _sig_memo[i]

                diags = diagnose_sparse_points(emb, labels, sparse_idx,
                                               s1_consistency=s1, radius=radius,
                                               signal_level_for=_sig_for)
                from collections import Counter
                cnt = Counter(d["cause"] for d in diags)
                helps = sum(v for c, v in cnt.items() if c in (CAUSE_H1, CAUSE_H5))
                phys = cnt.get(CAUSE_H0, 0)
                st.caption(f"稀疏點 {len(sparse_idx)} 個 · "
                           f":green[補資料有效（H1/H5）：{helps}] · "
                           + (f":red[桶①物理天花板（H0，補資料無效）：{phys}] · "
                              if phys else "")
                           + "其餘為定義/標籤/容量問題（補資料幫助有限）")
                for c, v in cnt.most_common():
                    st.write(f"- {c}：{v} 點")
                if "quiz_last_consistency" not in st.session_state:
                    st.caption(":gray[註：未跑組考卷，S1 用預設 0.9（非實測人類一致性）"
                               "→ H2 定義歧義無法觸發，以上為 S2/S3 兩訊號結論；"
                               "S3 為鄰域標籤熵代理。跑組考卷可補回 S1。]")

    with col_side:
        st.markdown("**① 候選資料夾（選完自動投影）**")
        # 單一資料夾語義:選新的取代舊的(與全域 picker 一致);後端仍吃多行
        # (E2E harness 直寫 session 多資料夾的相容性不變)。
        st.button("📁 選擇候選資料夾", key="cov_cand_pick", use_container_width=True,
                  on_click=_pick_folder_into_text, args=("cov_cand_text", True))
        _ctext = st.session_state.get("cov_cand_text", "").strip()
        for _l in [_x for _x in _ctext.splitlines() if _x.strip()]:
            st.caption(f"• {_l}")
        # 選完即自動投影（資料夾/粒度/模型變了才重算）→ 免按「投影」與「清空」
        _emb_key = f"{_ctext}|{'obj' if is_obj else 'img'}|{model}"
        if _ctext and st.session_state.get("_cov_cand_emb_key") != _emb_key:
            st.session_state["_cov_cand_emb_key"] = _emb_key
            _cov_embed_candidates(model)
            if st.session_state.get("cov_cand_emb") is not None:
                st.rerun()
        if not has_cand:
            st.info("投影 B 後，這裡可選 B 的角色：『採礦池』挑 B 補你的洞，"
                    "或『參照分佈』量你相對 B 缺多少。")
            return

        # ── B 的角色：採礦池(①) vs 參照分佈(②，補上覆蓋圖缺的外部真值)──
        b_role = st.radio(
            "B 的角色", ["採礦池", "參照分佈"], horizontal=True, key="cov_b_role",
            captions=["B＝待挑池：挑 B 裡最能補我稀疏洞的影像",
                      "B＝外部參照/真值：量我相對 B 覆蓋多少、缺哪些區域"])
        is_ref = b_role == "參照分佈"

        if is_ref:
            radius_b = _cov_radius(emb, active_token)
            uncovered, recall, d_b2a = reference_coverage(emb, cand_emb, radius_b)
            rc1, rc2 = st.columns(2)
            rc1.metric("覆蓋參照 B", f"{recall * 100:.0f}%",
                       help="B 的點有多少落在你資料的半徑內＝你覆蓋了參照分佈的多少。"
                            "這是覆蓋圖第一次有的『外部真值』——自我參照稀疏量不到。")
            rc2.metric("未覆蓋點", len(uncovered),
                       help="B 有、你半徑內沒覆蓋到的點＝你相對外部參照缺的區域。")
            st.caption(":gray[參照模式：B 當外部真值，量你相對 B 缺哪裡（補上"
                       "『稀疏＝自我參照、未校正真實分佈』的洞）。]")
            if recall < 0.5:
                st.warning(f"⚠ 你只覆蓋參照 B 的 {recall * 100:.0f}%——代表性嚴重不足，"
                           "據此算的覆蓋率／跨版本比較不可信（你的資料可能太小或偏）。")
            # ── 按候選類別聚合 uncovered → 類級補洞（物件級才有 label；R2 最後一哩）──
            if is_obj and _cov_cls is None and cand_records and uncovered:
                import collections
                _main_cnt = collections.Counter(labels)
                _cand_cnt = collections.Counter(
                    r.get("label", "") for r in cand_records)
                _unc_cnt = collections.Counter(
                    cand_records[i].get("label", "") for i in uncovered)

                def _gap_state(c: str) -> str:
                    if _main_cnt.get(c, 0) == 0:
                        return "🔴 整類缺失"
                    frac = _unc_cnt.get(c, 0) / max(_cand_cnt.get(c, 1), 1)
                    return "🟧 覆蓋稀疏" if frac > 0.6 else "🟩 尚可"
                _rows = [{"類別": c, "valid 有": _main_cnt.get(c, 0),
                          "B 總數": _cand_cnt[c], "未覆蓋": _unc_cnt.get(c, 0),
                          "狀態": _gap_state(c)}
                         for c in sorted(_cand_cnt, key=lambda c: (
                             _main_cnt.get(c, 0) != 0, -_unc_cnt.get(c, 0)))]
                st.markdown("**🎯 按類別補缺口（建議入口：把散亂候選收斂成「補哪一類」）**")
                st.caption("🔴 整類缺失＝你完全沒有（自指稀疏看不到、只有參照量得到）；"
                           "🟧 覆蓋稀疏＝有此類但 B 多數實例你沒覆蓋；🟩 尚可。")
                st.dataframe(pd.DataFrame(_rows), hide_index=True,
                             use_container_width=True)
                _gap_cls = [r["類別"] for r in _rows if r["未覆蓋"] > 0]
                if _gap_cls:
                    _g1, _g2 = st.columns([2, 1])
                    _pc = _g1.selectbox("選一類補（撈 B 的該類『未覆蓋』物件）",
                                        _gap_cls, key="cov_ref_gap_cls")
                    _cls_unc = [i for i in uncovered
                                if cand_records[i].get("label", "") == _pc]
                    _g2.button(f"🛒 加 {len(_cls_unc)} 個「{_pc}」入購物車",
                               key="cov_ref_gap_cart", use_container_width=True,
                               on_click=_batch_add,
                               args=(cand_records, _cls_unc, "reference_class",
                                     {i: float(d_b2a[i]) for i in _cls_unc}))
                st.divider()
            work_idx = uncovered
            work_score = {i: float(d_b2a[i]) for i in uncovered}
            head, csv_name, send_key, cart_src = (
                "你相對參照 B 缺的區域（B 有、你沒覆蓋）",
                "reference_gaps.csv", "cov_to_quiz_ref", "reference")
            if not work_idx:
                st.success("你已覆蓋參照 B 的全部區域（半徑內）。")
                return
        else:
            work_idx = ranked_idx
            work_score = dict(zip(ranked_idx, ranked_scores))
            head, csv_name, send_key, cart_src = (
                "最能補洞的候選", "gap_fillers.csv", "cov_to_quiz", "gap_filler")
            st.caption("依『離既有資料多遠（落在稀疏區）』排序，越前越能補你缺的區域。")

        st.markdown(f"**② {head}**")
        send_n = int(st.number_input(
            "前 N 名", min_value=1, max_value=len(work_idx),
            value=min(12, len(work_idx)), key="cov_send_n"))
        picks = work_idx[:send_n]
        # 組考卷已隱藏 → 移除「送組考卷」handoff，保留「加入購物車」與下方送 Labeling。
        st.button(f"🛒 把前 {len(picks)} 名加入策展購物車", key="cov_cand_cart",
                  use_container_width=True, type="primary", on_click=_batch_add,
                  args=(cand_records, list(picks), cart_src, dict(work_score)))
        # 直接送 Labeling 從頭標註（fresh）；保住「補哪一格／相對外部 B 缺」語境
        _send_to_labeling_ui(
            cand_records, list(picks), source=cart_src, task=LH.TASK_FRESH,
            label="📤 送這批到 Labeling 標註", key=f"{send_key}_lbl",
            original_labels={i: (provisional[i] or "") for i in picks},
            payload={"kind": cart_src,
                     "scores": {str(i): float(work_score[i]) for i in picks}},
            help="把補洞／未覆蓋候選送到 Labeling 從頭標註（fresh，未標新樣本）；"
                 "標完在 Labeling 端「匯出 / 回傳」匯出即完成，不用回 LV。")
        st.caption(":gray[候選無標籤——暫定類別取自最近鄰，到 Labeling 盲標即為新標籤。]")
        with st.container(height=480):
            cols = st.columns(3)
            for j, i in enumerate(picks):
                with cols[j % 3]:
                    p = Path(cand_records[i]["path"])
                    thumb = _thumb_or_none(p)
                    if thumb:
                        st.image(thumb, use_container_width=True,
                                 caption=f"{provisional[i] or '?'}"
                                         f"｜{_rec_fname(cand_records[i])}"
                                         f" · d={work_score[i]:.3f}")
                    else:
                        st.warning("⚠ 缺檔")
        # CSV path 欄給原圖路徑（物件級 path=裁切快取圖，使用者要能找到原始檔）
        csv = _rows_to_csv(
            ["rank", "path", "score", "provisional_label"],
            [[r + 1, cand_records[i].get("image_path") or cand_records[i]["path"],
              f'{work_score[i]:.6f}',
              provisional[i] or ""] for r, i in enumerate(picks)])
        st.download_button("⬇ 匯出 CSV", data=csv,
                           file_name=csv_name, mime="text/csv",
                           key="cov_gap_csv", use_container_width=True)


def _objcov_embed_candidates(folders, model, policy, fallback_cnames) -> None:
    """投影候選資料夾的『物件』進主空間（用主資料集相同的物件政策→可比）。"""
    folders = [Path(f) for f in folders if Path(f).exists()]
    if not folders:
        st.warning("候選資料夾不存在。"); return
    cand_records = _cov_candidate_image_records(folders)
    if not cand_records:
        st.warning("候選資料夾找不到影像。"); return
    # 與主資料同樣地探兩層（root 與其上層）→ 類別 id→名 對齊，避免比到不同類別
    cnames = _manual_class_names("cov") or _cov_class_names(cand_records) or fallback_cnames
    orecs, cemb, _ = _crop_and_embed_objects(
        cand_records, model, cnames, float(policy.get("pad", 0.12)),
        base_token="objcovC|" + repr(sorted(str(f) for f in folders)),
        session_key="_objcov_objC", spinner="裁切候選物件", policy=policy)
    if not orecs:
        st.warning("候選資料夾的 labels/ 找不到 bbox（需 YOLO 偵測資料夾）。"); return
    st.session_state["objcov_cand"] = {
        "token": uuid.uuid4().hex, "recs": orecs, "emb": cemb,
        "labels": [r.get("label", "") for r in orecs]}
    st.session_state.pop("_cov_proj_cache", None)
    st.toast(f"已投影 {len(orecs)} 個候選物件進此空間", icon="🧭")


_OBJCOV_PROJ = {"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap",
                "監督UMAP": "sumap", "LDA(監督)": "lda"}


def _objcov_project(memb, cmemb, mkey, dim, cache_key):
    """投影主(+候選)到同框；監督法(sumap/lda)以「主 vs 候選」分組做監督，凸顯兩邊差異。
    有快取 → 框選等 rerun 不重算（避免畫面當掉）。回傳 (coords_main, coords_cand|None)。"""
    has_c = cmemb is not None and len(cmemb) > 0
    memb = _l2norm(memb)              # 投影用 cosine 幾何（與稀疏度/配對一致）→
    if has_c:                        # 同類候選才會落在主資料上，不被量級差推成獨立一團
        cmemb = _l2norm(cmemb)
    if mkey in _SUPERVISED_METHODS:
        c = st.session_state.get("_objcov_proj_cache")
        if c and c.get("key") == cache_key:
            return c["m"], c["c"]
        combined = np.vstack([memb, cmemb]) if has_c else np.asarray(memb)
        groups = [0] * len(memb) + ([1] * len(cmemb) if has_c else [])
        # 樣本太少(如 3D+合併=4)監督UMAP spectral 會崩 → 退 PCA
        sup = (_supervised_projection(combined, groups, mkey, dim)
               if len(set(groups)) > 1 and len(combined) > dim + 1 else None)
        if sup is None:   # 無候選/只有一群 → 退 PCA
            arr = PCA(n_components=min(dim, max(1, len(combined) - 1)),
                      random_state=42).fit_transform(combined)
        else:
            arr = np.asarray(sup)
            if arr.shape[1] < dim:   # LDA 兩群→1 維,補 PCA 湊 2D/3D
                pad = PCA(n_components=dim - arr.shape[1],
                          random_state=42).fit_transform(combined)
                arr = np.hstack([arr, pad])
        arr = _pad_cols(arr, dim)
        cm, cc = arr[:len(memb)], (arr[len(memb):] if has_c else None)
        st.session_state["_objcov_proj_cache"] = {"key": cache_key, "m": cm, "c": cc}
        return cm, cc

    # 非監督：**fit 在「主」、transform 候選**（固定參考系）→ 候選落在真實相對位置，
    # 不會被 t-SNE/UMAP 的 co-fit 批次效應整批推到另一邊。
    cache = st.session_state.get("_objcov_proj_cache")
    if cache and cache.get("key") == cache_key:
        return cache["m"], cache["c"]
    memb = np.asarray(memb)
    nc = min(dim, max(1, len(memb) - 1))
    if not has_c:
        if mkey == "tsne":
            cm = TSNE(n_components=nc, random_state=42,
                      perplexity=min(30, max(2, len(memb) - 1))).fit_transform(memb)
        elif mkey == "umap":
            cm = _umap().UMAP(n_components=nc, n_neighbors=min(15, max(2, len(memb) - 1)),
                              random_state=42).fit_transform(memb)
        else:
            cm = PCA(n_components=nc, random_state=42).fit_transform(memb)
        cc = None
    else:
        cmemb = np.asarray(cmemb)
        if mkey == "umap":  # UMAP 支援 out-of-sample transform
            red = _umap().UMAP(n_components=nc, n_neighbors=min(15, max(2, len(memb) - 1)),
                               random_state=42).fit(memb)
            cm, cc = red.embedding_, red.transform(cmemb)
        else:  # PCA；t-SNE 無 transform → 疊候選時改用 PCA（co-fit 會把兩組推開、誤導）
            red = PCA(n_components=nc, random_state=42).fit(memb)
            cm, cc = red.transform(memb), red.transform(cmemb)
    cm = _pad_cols(np.asarray(cm), dim)
    cc = _pad_cols(np.asarray(cc), dim) if cc is not None else None
    st.session_state["_objcov_proj_cache"] = {"key": cache_key, "m": cm, "c": cc}
    return cm, cc


def _objcov_sidebar(model: str) -> None:
    """物件補洞模式的左側工具列（對齊 Visualize 概念）：物件 crop 政策（自動／手動）
    + 投影／區塊設定。值寫進 session，由 _render_objcov_view 讀取。"""
    import object_eval as oe
    folders = [Path(f) for f in st.session_state.get("cov_folder_list", [])
               if Path(f).exists()]
    policy = st.session_state.get("objcov_policy") or dict(oe.DEFAULT_POLICY)
    st.markdown("**④ 物件設定（crop）**")
    with st.expander(f"目前 {oe.policy_tag(policy)}（自動／手動）"):
        if st.button("🔬 自動找最佳設定", key="objcov_autotune",
                     use_container_width=True, disabled=not folders):
            recs = discover_images_classifier(folders)
            imgs = [Path(r["path"]) for r in recs]
            cnames = _manual_class_names("cov") or _cov_class_names(recs)
            with st.status("量測中…", expanded=True) as _s:
                _bar = st.progress(0.0)

                def _pp(ci, n, pol, od, ot):
                    _bar.progress(min((ci + od / max(ot, 1)) / max(n, 1), 1.0),
                                  text=f"[{ci+1}/{n}] {oe.policy_tag(pol)} · {od}/{ot}")
                try:
                    _, _best, _ = oe.run_autotune(
                        imgs, cnames, model=model,
                        cache_dir=_dataset_cache_dir(
                            _cov_object_root(recs), "object_crops") / "_autotune",
                        progress=_pp)
                except Exception:
                    _best = None
                if not _best:
                    _best = dict(oe.DEFAULT_POLICY)
                _s.update(label=f"最佳：{oe.policy_tag(_best)}", state="complete",
                          expanded=False)
            st.session_state["objcov_policy"] = _best
            st.session_state.pop("_objcov_main", None)
            st.rerun()
        st.caption("✏️ 手動微調")
        _man = _manual_policy_controls(policy, "objcovman")
        if st.button("套用手動設定", key="objcov_apply_manual", use_container_width=True):
            st.session_state["objcov_policy"] = _man
            st.session_state.pop("_objcov_main", None)
            st.rerun()
    st.markdown("**⑤ 投影方法**")
    # 預設維持 PCA:此處監督UMAP 的監督訊號是「主 vs 候選」分組(刻意推開兩邊),
    # 與補洞工具「候選要落在主資料稀疏區附近」的核心視覺相矛盾,不適合當預設。
    st.selectbox("投影方法", list(_OBJCOV_PROJ), key="objcov_proj",
                 label_visibility="collapsed",
                 help="只用來畫圖；監督UMAP／LDA(監督) 以「主 vs 候選」分組做監督。"
                      "維度／區塊數／稀疏度在右側散點上方即時調。")


def _render_objcov_view(records: list[dict], model: str) -> None:
    """按物件・逐類別補洞：主資料集逐 YOLO 類別找稀疏盲區（標號）→ 去其它資料夾撈
    該（類別,區塊）的物件 → 疊在同一 2D/3D 預覽 → 收進清單（提案，不寫對方資料夾）。"""
    import object_eval as oe
    from completeness import mine_candidates
    if not _is_detection_dataset(records):
        st.info("此模式需 **YOLO 偵測資料夾**（含 images/＋labels/）。請在左側選偵測資料集。")
        return
    class_names = _manual_class_names("cov") or _cov_class_names(records)

    # crop 政策 + 投影方法來自左側工具列（set-once 設定）；這裡只讀。
    policy = st.session_state.get("objcov_policy") or dict(oe.DEFAULT_POLICY)
    _pm = st.session_state.get("objcov_proj", "PCA")
    if _pm not in _OBJCOV_PROJ:
        _pm = "PCA"

    # ── 主資料集物件 embedding（依政策快取）──
    main_token = f"{st.session_state.get('cov_token','')}_{oe.policy_tag(policy)}"
    cache = st.session_state.get("_objcov_main")
    if not cache or cache.get("token") != main_token:
        orecs, oemb, _ = _crop_and_embed_objects(
            records, model, class_names, float(policy.get("pad", 0.12)),
            base_token="objcovM" + main_token, session_key="_objcov_objM", policy=policy)
        if not orecs:
            st.warning("labels/ 找不到任何 bbox。"
                       + _seg_format_hint([r["path"] for r in records])); return
        cache = {"token": main_token, "recs": orecs, "emb": oemb,
                 "labels": [r.get("label", "") for r in orecs]}
        st.session_state["_objcov_main"] = cache
    orecs, oemb, olabels = cache["recs"], cache["emb"], cache["labels"]

    # 即時過濾：信心（資料含 score 才出現）／源短邊 → 同步濾 orecs/oemb/olabels。
    _obj_filter_controls("objcov", orecs)
    _keep = _obj_view_indices(orecs)
    if len(_keep) < len(orecs):
        orecs = [orecs[i] for i in _keep]
        oemb = np.asarray(oemb)[_keep]
        olabels = [olabels[i] for i in _keep]

    # ── 右半部即時調整（不需重跑）：維度 / 區塊數 K / 稀疏度 k ──
    _vc = st.columns([1, 1, 1])
    _dim = 3 if _vc[0].radio("維度", ["2D", "3D"], horizontal=True,
                             key="objcov_dim") == "3D" else 2
    _K = int(_vc[1].slider("區塊數 K", 3, 20, 8, key="objcov_k"))
    _spk = int(_vc[2].number_input("稀疏度 k", 1, 50, 8, key="objcov_spk"))

    # ── 逐類別表（物件數／盲區塊數／最稀疏）──
    classes = sorted(set(olabels))
    # R2-4：整類缺失（classes.txt 有、本資料集 0 個）→ 本模式按「現有類」分塊碰不到 → 指去參照
    if class_names:
        _absent = [c for c in class_names if c not in set(olabels)]
        if _absent:
            st.info(f"ℹ️ classes.txt 有、但本資料集 **0 個** 的類：**{'、'.join(_absent)}**。"
                    "本模式按你『現有的類』分塊，補不到它們——整類缺失請到"
                    "『覆蓋散點・參照分佈』補（摘要卡的「➜ 量缺口」按鈕會帶你過去）。")
    _trk = f"{main_token}|{int(_spk)}"   # 逐類別表快取(框選 rerun 不重算)
    _tc = st.session_state.get("_objcov_table")
    if _tc and _tc.get("key") == _trk:
        rows = _tc["rows"]
    else:
        rows = []
        for c in classes:
            ci0 = [i for i, l in enumerate(olabels) if l == c]
            if len(ci0) < 2:
                rows.append({"類別": c, "物件數": len(ci0), "盲區塊": 0, "最稀疏": 0.0})
                continue
            sp = sparsity_scores(oemb[ci0], k=min(int(_spk), len(ci0) - 1))
            thr = float(np.percentile(sp, 80))
            rows.append({"類別": c, "物件數": len(ci0),
                         "盲區塊": int(np.sum(np.asarray(sp) >= thr)),
                         "最稀疏": round(float(np.max(sp)), 3)})
        rows.sort(key=lambda r: (r["物件數"], -r["最稀疏"]))  # 薄的類別排前面
        st.session_state["_objcov_table"] = {"key": _trk, "rows": rows}
    _maxsp = max([r["最稀疏"] for r in rows] + [0.05])
    _df = pd.DataFrame(rows)
    _df.insert(1, "可分析", ["✅" if r["物件數"] >= 4 else "⚠<4" for r in rows])
    st.dataframe(
        _df, use_container_width=True, hide_index=True,
        column_config={
            "可分析": st.column_config.TextColumn("可分析", help="物件數 <4 無法分塊"),
            "最稀疏": st.column_config.ProgressColumn(
                "最稀疏（越大越薄）", min_value=0.0, max_value=_maxsp, format="%.3f")})
    st.caption(":gray[「⚠<4」的類別物件太少、無法分塊分析；選它只會提示擴量。]")

    _clslist = [r["類別"] for r in rows]
    _analyzable = [r["類別"] for r in rows if r["物件數"] >= 4]
    _didx = _clslist.index(_analyzable[0]) if _analyzable else 0
    pick = st.selectbox("選類別分析（預設＝最薄且可分析）", _clslist, index=_didx)
    ci = [i for i, l in enumerate(olabels) if l == pick]
    if len(ci) < 2:
        st.info(f"「{pick}」只有 {len(ci)} 個物件，太少無法分析；換個類別或整類擴量。")
        return
    if len(ci) < 4:
        st.warning(f"「{pick}」只有 {len(ci)} 個物件，盲區判斷不穩 — 建議整類擴量"
                   "（直接到下面撈這個類別的候選）。")
    memb = oemb[ci]
    _mspk = f"{main_token}|{pick}|{int(_spk)}"
    _mc = st.session_state.get("_objcov_msp")
    if _mc and _mc.get("key") == _mspk:
        msp = _mc["msp"]
    else:
        msp = sparsity_scores(memb, k=min(int(_spk), max(1, len(ci) - 1)))
        st.session_state["_objcov_msp"] = {"key": _mspk, "msp": msp}

    # ── 候選資料夾（同政策、同類別）──
    with st.expander("📁 候選資料夾（去這些資料夾找可補的物件）", expanded=False):
        cand_folders = _folder_picker_list("objcov_cand_list",
                                           add_help="YOLO 偵測資料夾（含 labels/）")
        if st.button("🧭 投影候選進此空間", key="objcov_embed_cand",
                     use_container_width=True, type="primary", disabled=not cand_folders):
            _objcov_embed_candidates(cand_folders, model, policy, class_names)
            st.rerun()
    cand = st.session_state.get("objcov_cand")
    cand_ci = [i for i, l in enumerate(cand["labels"]) if l == pick] if cand else []
    cmemb = cand["emb"][cand_ci] if cand_ci else None

    # ── 投影（主＋候選同框）；區塊只在「主」高維上算並凍結 ──
    cand_token = cand["token"] if cand else ""
    _ckey = f"{main_token}|{pick}|{_pm}|{_dim}|{cand_token}"
    coords_m, coords_c = _objcov_project(memb, cmemb, _OBJCOV_PROJ[_pm], _dim, _ckey)
    if _pm == "t-SNE" and cmemb is not None and len(cmemb):
        st.caption(":orange[ℹ t-SNE 無法投影候選 → 疊候選時自動改用 PCA，候選位置才正確。]")
    if _OBJCOV_PROJ[_pm] in _SUPERVISED_METHODS and cmemb is not None and len(cmemb):
        st.caption(":gray[ℹ 監督投影：主／候選的分離是用分組標籤排版的視覺輔助，"
                   "不代表兩者真的不同。]")
    # 區塊在「主資料的高維 embedding」上分群 → membership 不受候選/投影/渲染順序影響(凍結)
    blocks, _ = _viz_kmeans_blocks(
        np.asarray(memb), list(range(len(ci))), _K,
        f"objcovhi_{main_token}_{pick}_{_K}")
    from collections import Counter as _Counter
    cnt = _Counter(blocks.values())
    _cd = _pad_cols(np.asarray(coords_m), 3)   # 顯示座標（隨投影變）
    nblk = (max(blocks.values()) + 1) if blocks else 0
    centers = np.zeros((nblk, 3))
    block_hi, block_sp = {}, {}
    for b in range(nblk):
        mem = [i for i in range(len(ci)) if blocks.get(i) == b]
        if not mem:
            continue
        centers[b] = _cd[mem].mean(0)          # 數字落在「目前投影」的成員中心
        block_hi[b] = memb[mem].mean(0)        # 配對用的高維形心（凍結）
        block_sp[b] = float(np.mean([msp[i] for i in mem]))
    _present = sorted(block_hi)                 # 有成員的區塊
    block_rank = sorted(block_hi, key=lambda b: (cnt[b], -block_sp[b]))  # 薄＋稀疏優先

    # box-select 狀態（2D 可框選看圖；sentinel：候選 = j + _CMPC_BOFF）
    code_m = [[i] for i in range(len(ci))]
    code_c = [[j + _CMPC_BOFF] for j in range(len(cand_ci))]
    _otok = f"{main_token}|{pick}|{_pm}|{_dim}"
    osel = st.session_state.get("objcov_sel") or {}
    if osel.get("token") != _otok:
        osel = {"token": _otok, "main": [], "cand": []}
    _ocn = st.session_state.get("_objcov_clear_nonce", 0)

    col_map, col_side = st.columns([5, 3], gap="medium")
    with col_map:
        _tb = st.columns([5, 1])
        _tb[0].caption("💡 2D 可框選／套索 → 下方顯示選到的物件；3D 請用右側區塊流程。")
        _clear_slot = _tb[1].empty()   # 等本次選取處理完再畫，否則會一直 disabled
        fig = go.Figure()
        _mk = dict(size=6 if _dim == 2 else 4, color=msp, colorscale="Turbo",
                   showscale=True, colorbar=dict(title="稀疏", thickness=10),
                   opacity=0.85)
        sel_codes = None
        if _dim == 2:
            fig.add_trace(go.Scatter(
                x=coords_m[:, 0], y=coords_m[:, 1], mode="markers", marker=_mk,
                name=f"主・{pick}", customdata=code_m,
                text=[f"稀疏 {s:.2f}" for s in msp],
                hovertemplate="%{text}<extra></extra>"))
            if coords_c is not None and len(coords_c):
                fig.add_trace(go.Scatter(
                    x=coords_c[:, 0], y=coords_c[:, 1], mode="markers",
                    marker=dict(symbol="diamond", color="#111", size=8),
                    name="候選", customdata=code_c,
                    text=[f"候選 #{i}" for i in cand_ci],
                    hovertemplate="%{text}<extra></extra>"))
            if _present:   # 區塊號用 annotations（非 trace）→ 不干擾框選
                fig.update_layout(annotations=[
                    dict(x=float(centers[b, 0]), y=float(centers[b, 1]), text=str(b),
                         showarrow=False, font=dict(size=16, color="#111"),
                         bgcolor="rgba(255,255,255,0.7)") for b in _present])
            fig.update_layout(height=480, dragmode="select",
                              margin=dict(l=0, r=0, t=0, b=0),
                              legend=dict(orientation="h"))
            ev = st.plotly_chart(
                fig, use_container_width=True,
                key=f"objcov2d_{_otok}_{_ocn}_{_obj_filter_tag()}",
                on_select="rerun", selection_mode=("points", "box", "lasso"))
            if ev is not None:
                _so = ev.get("selection") if hasattr(ev, "get") else None
                if _so:
                    sel_codes = selection_points_to_indices(list(_so.get("points", [])))
        else:
            fig.add_trace(go.Scatter3d(
                x=coords_m[:, 0], y=coords_m[:, 1], z=coords_m[:, 2], mode="markers",
                marker=_mk, name=f"主・{pick}", text=[f"稀疏 {s:.2f}" for s in msp],
                hovertemplate="%{text}<extra></extra>"))
            if coords_c is not None and len(coords_c):
                fig.add_trace(go.Scatter3d(
                    x=coords_c[:, 0], y=coords_c[:, 1], z=coords_c[:, 2], mode="markers",
                    marker=dict(symbol="diamond", color="#111", size=4),
                    name="候選", text=[f"候選 #{i}" for i in cand_ci],
                    hovertemplate="%{text}<extra></extra>"))
            if _present:
                fig.update_scenes(annotations=[
                    dict(x=float(centers[b, 0]), y=float(centers[b, 1]),
                         z=float(centers[b, 2]), text=str(b), showarrow=False,
                         font=dict(size=16, color="#111"),
                         bgcolor="rgba(255,255,255,0.7)", borderpad=2)
                    for b in _present])
            fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0),
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True, key=f"objcov3d_{_otok}")
        st.caption("顏色＝高維稀疏度（亮＝越薄）；白底數字＝區塊號（主資料高維上算、"
                   "投影候選後不變）。黑菱形＝候選物件。")
        if sel_codes:
            _ms = sorted({c for c in sel_codes if c < _CMPC_BOFF})
            _cs = sorted({c - _CMPC_BOFF for c in sel_codes if c >= _CMPC_BOFF})
            if (_ms, _cs) != (osel["main"], osel["cand"]):
                osel = {"token": _otok, "main": _ms, "cand": _cs}
                st.toast(f"框選 {len(_ms) + len(_cs)} 個物件", icon="🎯")
        st.session_state["objcov_sel"] = osel
        _nsel = len(osel["main"]) + len(osel["cand"])
        if _clear_slot.button(f"✕ 取消框選（{_nsel}）" if _nsel else "✕ 取消框選",
                              key="objcov_clear_sel", use_container_width=True,
                              disabled=not _nsel):
            st.session_state["_objcov_clear_nonce"] = _ocn + 1
            st.session_state["objcov_sel"] = {"token": _otok, "main": [], "cand": []}
            st.rerun()
        if osel["main"] or osel["cand"]:
            st.markdown(f"**框選：主 {len(osel['main'])} · 候選 {len(osel['cand'])}**")
            _sg = st.columns(8)
            _n = 0
            for i in osel["main"][:24]:
                _mr = orecs[ci[i]]
                _sg[_n % 8].image(str(_mr["path"]), use_container_width=True,
                                  caption=f"{_mr.get('label') or '?'}｜{_rec_fname(_mr)}")
                _n += 1
            for j in osel["cand"][:24]:
                if cand and j < len(cand_ci):
                    _cr2 = cand["recs"][cand_ci[j]]
                    _sg[_n % 8].image(str(_cr2["path"]), use_container_width=True,
                                      caption=f"{_cr2.get('label') or '?'}"
                                              f"｜{_rec_fname(_cr2)}")
                    _n += 1

    with col_side:
        st.markdown(f"**盲區塊（{pick}）**")
        st.caption("選一個區塊 → 看主資料缺的樣態，再去候選找相似的補。")
        if not block_hi:
            st.info("此類別物件太少，無法分塊。"); return
        _sort = st.radio("排序", ["最稀疏", "最薄（物件少）", "區塊號"],
                         horizontal=True, key="objcov_block_sort")
        if _sort == "最稀疏":
            order = sorted(block_hi, key=lambda b: -block_sp[b])
        elif _sort.startswith("最薄"):
            order = sorted(block_hi, key=lambda b: (cnt[b], -block_sp[b]))
        else:
            order = sorted(block_hi)
        opts = [f"#{b}（{cnt[b]} 個·稀{block_sp[b]:.2f}）" for b in order]
        bsel = st.selectbox("進入區塊", opts, key="objcov_block_pick")
        bk = order[opts.index(bsel)]
        bmem = [i for i in range(len(ci)) if blocks.get(i) == bk]
        st.caption(f"主資料・區塊 #{bk}：{len(bmem)} 個物件（你目前的樣態）")
        gcols = st.columns(4)
        for j, i in enumerate(bmem[:8]):
            _br = orecs[ci[i]]
            gcols[j % 4].image(str(_br["path"]), use_container_width=True,
                               caption=f"{_br.get('label') or '?'}｜{_rec_fname(_br)}")

    # ── 撈候選補此（類別,區塊）→ 排名＋加入清單 ──
    st.divider()
    st.markdown(f"**補「{pick} · 區塊 #{bk}」的候選**")
    if not cand:
        st.caption("先在上方「📁 候選資料夾」選資料夾並『投影候選進此空間』。")
        return
    if cmemb is None or not len(cmemb):
        st.warning(f"候選資料夾裡沒有「{pick}」類別的物件。換個資料夾或類別。")
        return
    _radius = _cov_radius(memb, main_token)
    m_idx, m_dist = mine_candidates(cmemb, block_hi[bk], k=24,
                                    max_distance=_radius * 1.5 if _radius else None)
    if not m_idx:
        st.warning(f"候選裡沒有夠接近「#{bk}」這個樣態的物件（都太遠）。換資料夾或選別的區塊。")
        return
    glob = [cand_ci[i] for i in m_idx]  # → cand["recs"] 索引
    # 一列：加入前 N 名 + 加入清單（放在候選縮圖上面）
    ac = st.columns([3, 2])
    nadd = ac[0].slider("加入前 N 名", 1, len(glob), min(12, len(glob)),
                        key="objcov_nadd", label_visibility="collapsed")
    if ac[1].button(f"🛒 加入清單（前 {nadd} 名；提案，不寫對方資料夾）",
                    key="objcov_add", use_container_width=True, type="primary"):
        sc = {gi: float(d) for gi, d in zip(glob, m_dist)}
        _batch_add(cand["recs"], glob[:nadd], source="objcov_backfill", scores=sc)
        _append_curation_entry(cand["recs"], glob[:nadd],
                               reason=f"backfill {pick} #block{bk}")
    st.caption(f"找到 {len(glob)} 個接近 #{bk} 的候選（距離越小越像；標籤為候選自帶／暫定）。")
    pcols = st.columns(6)
    for j, (gi, d) in enumerate(zip(glob[:12], m_dist[:12])):
        with pcols[j % 6]:
            _gr = cand["recs"][gi]
            st.image(str(_gr["path"]), use_container_width=True)
            st.caption(f"{_gr.get('label') or '?'}｜{_rec_fname(_gr)} · d={d:.2f}")


def _completeness_ui() -> None:
    st.markdown("##### 模型收值完整性熱力圖")
    st.caption("把資料切成小棋盤格，看每格「不太多也不太少」。"
               "🟩 健康、🟪 假完整（量夠但都是近重複）、🟥/🟧 缺。")
    st.caption(":gray[👉 這是看**單一資料集「內部」**哪裡缺／假完整；"
               "要比**兩堆資料像不像**（真實 vs 生成、train vs val）請用 "
               "**Compare Distributions**。]")

    with st.sidebar:
        st.markdown("**① 資料夾**")
        _folder_picker_list("cov_folder_list",
                            add_help="結構需為 資料夾／類別／影像。")
        all_models = available_models()
        if not all_models:
            st.error("models/ 內找不到模型檔。")
            return
        st.markdown("**② 模型**")
        model = st.selectbox("模型", all_models, label_visibility="collapsed",
                             index=_default_model_index(all_models),
                             help="算每格內 embedding 多樣性（質量探針）用。")
        _class_source_expander("cov")  # 物件級的 YOLO 類別名(主資料與候選共用)
        st.markdown("**③ 檢視方式**")
        cov_view = st.radio(
            "檢視方式", ["嵌入覆蓋圖", "屬性棋盤"],
            key="cov_view_mode", label_visibility="collapsed",
            captions=["特徵空間：找稀疏盲區＋投影候選補洞＋根因／參照（整圖或物件；可逐類別區塊）",
                      "可解讀屬性軸切格（量化每格夠不夠、抓近重複假完整）"])
        cov_sub = "覆蓋散點"
        if cov_view == "嵌入覆蓋圖":
            cov_sub = st.radio(
                "模式", ["覆蓋散點", "逐類別・區塊補洞"], key="cov_emb_submode",
                captions=["稀疏盲區散點＋候選補洞＋H1–H5 根因＋參照分佈（整圖／物件）",
                          "逐 YOLO 類別把物件切成編號盲區塊 → 對區塊撈候選（物件級）"])
            if cov_sub == "逐類別・區塊補洞":
                _objcov_sidebar(model)
            else:
                st.markdown("**④ 投影方法**")
                st.selectbox("投影方法", list(_METHOD_KEY),
                             index=list(_METHOD_KEY).index("監督UMAP"),
                             key="cov_proj_method", label_visibility="collapsed",
                             help="只用來畫圖；稀疏度一律高維算。LDA／監督UMAP 依類別排版"
                                  "（候選用最近鄰暫定類別）。維度／稀疏度 k 在右側即時調。")
        run = st.button("▶ 開始分析", use_container_width=True, key="run_cov",
                        type="primary")

    if st.session_state.pop("_cov_autorun", False):
        run = True
    if run:
        folders = [Path(f) for f in st.session_state.get("cov_folder_list", [])]
        missing = [str(p) for p in folders if not p.exists()]
        if not folders:
            st.error("請先選擇資料夾。"); return
        if missing:
            st.error(f"資料夾不存在：{', '.join(missing)}"); return
        records = discover_images_classifier(folders)
        if not records:
            st.error("找不到影像（需 資料夾／類別／影像 結構）。"); return
        records = _filter_records_readable(records, "cov")  # 壞檔防呆:剔除無法解碼者
        if not records:
            st.error("找不到影像（需 資料夾／類別／影像 結構）。"); return
        _collapse_sidebar()

        embed_fn = load_model(model)
        with st.status("計算中…", expanded=True) as _status:
            bar = st.progress(0.0, text="特徵擷取…")
            paths = [r["path"] for r in records]
            cache = _dataset_cache_dir(folders[0], f"embeddings_{model}") / "embeddings.npz"

            def _cb(done, total):
                bar.progress(min(done / max(total, 1) * 0.6, 0.6),
                             text=f"特徵擷取 {done}/{total}")
            emb = extract_embeddings(paths, embed_fn, cache_path=cache, progress_cb=_cb)

            stats = []
            for i, p in enumerate(paths):
                try:
                    stats.append(image_stats(p))
                except (OSError, Image.DecompressionBombError):  # 壞圖/超大圖→中性屬性
                    stats.append({a: 0.0 for a in _AUTO_AXES})
                if i % 20 == 0:
                    bar.progress(0.6 + 0.4 * (i + 1) / len(paths),
                                 text=f"影像屬性 {i + 1}/{len(paths)}")
            bar.progress(1.0, text="完成")
            _status.update(label="完成", state="complete", expanded=False)

        st.session_state["cov_records"] = records
        st.session_state["cov_emb"] = emb
        st.session_state["cmp_img_stats"] = stats
        st.session_state["cov_token"] = uuid.uuid4().hex
        st.session_state.pop("cov_active_cell", None)
        st.toast(f"完成：{len(records)} 張影像", icon="✅")

    if "cov_records" not in st.session_state:
        _render_cov_quick_start()
        return

    records = st.session_state["cov_records"]
    emb = st.session_state["cov_emb"]

    _cov_summary_card(records)   # 進門即見的資料集摘要卡（含整類缺失→補洞縫合）

    if cov_view == "嵌入覆蓋圖":
        if cov_sub == "逐類別・區塊補洞":
            _render_objcov_view(records, model)
        else:
            _render_coverage_view(records, emb, model)
        return

    # ── tuning 列（在熱力圖正上方即時調，免重跑）──
    axis_opts = ["label", "split", *_AUTO_AXES]
    _AXIS_LABEL = {"label": "類別", "split": "資料集(split)", "brightness": "亮度",
                   "contrast": "對比", "sharpness": "銳利度", "aspect": "長寬比"}
    tcol = st.columns([1.4, 1.4, 1, 1.3, 1.4])
    ax_x = tcol[0].selectbox("橫看（X）", axis_opts, index=0, key="cov_ax_x",
                             format_func=lambda a: _AXIS_LABEL.get(a, a),
                             help="把資料依哪個特徵切成橫向格子。")
    ax_y = tcol[1].selectbox("直看（Y）", axis_opts, index=2, key="cov_ax_y",
                             format_func=lambda a: _AXIS_LABEL.get(a, a),
                             help="把資料依哪個特徵切成縱向格子。")
    bins = tcol[2].number_input("連續特徵分幾檔", min_value=2, max_value=8, value=3,
                                key="cov_bins",
                                help="亮度這類連續值切成幾段（暗/中/亮＝3）。"
                                     "改這個會清掉已填的真實分佈校正。")
    t_abs = tcol[3].number_input("每格至少幾張", min_value=1, value=10,
                                 key="cov_t_abs",
                                 help="低於此數視為樣本不足。未提供真實分佈時對每格一視同仁。")
    preset = tcol[4].radio("近重複警戒", list(_D_STAR_PRESET), index=1, horizontal=True,
                           key="cov_dstar_preset",
                           help="一格裡的圖太像（疑似近重複充數）就標🟪假完整；越嚴格越容易被判為假完整。")
    d_star = _D_STAR_PRESET[preset]

    bx, lx = _completeness_axis_values(records, ax_x)
    by, ly = _completeness_axis_values(records, ax_y)

    # (a) 真實分佈校正：每格 高/中/低/不適用 先驗（粗分級即可起步）
    grid_key = f"{st.session_state.get('cov_token', '')}|{ax_x}|{ax_y}|{bins}"
    freq_classes = st.session_state.get("cov_freq_classes", {})
    if st.session_state.get("cov_freq_grid_key") != grid_key:
        freq_classes = {}
        st.session_state["cov_freq_grid_key"] = grid_key
        st.session_state["cov_freq_classes"] = freq_classes
    result = build_completeness(records, emb, bx, by, lx, ly,
                                t_abs=int(t_abs), d_star=float(d_star),
                                freq_classes=freq_classes or None)
    cells = result["cells"]
    health = result["health"]

    col_map, col_side = st.columns([5, 3], gap="medium")
    with col_map:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Coverage Health", f"{health['coverage_health']:.0f}%",
                  help="達標格佔比（含假完整 0.7 折）。一眼看資料夠不夠。")
        n_miss = health["counts"].get(STATE_EMPTY, 0) + health["counts"].get(STATE_MISSING, 0)
        m2.metric("缺格數", f"{n_miss}/{len(cells)}")
        m3.metric("不均度 Gini", f"{health['gini']:.2f}",
                  help="0=每格平均；高=量集中在少數格（假完整風險）。")
        m4.metric("假完整格", f"{health['fake_ratio'] * 100:.0f}%",
                  help="量夠但近重複的格子比例。")
        if not result["calibrated"]:
            st.caption(":orange[⚠ 目標數為均勻假設（未校正真實分佈）——稀有格可能被誤判為缺。]")

        # 熱力圖：屬性軸網格、五態離散色
        nx, ny = len(lx), len(ly)
        z = [[None] * nx for _ in range(ny)]
        text = [[""] * nx for _ in range(ny)]
        for c in cells:
            z[c["y"]][c["x"]] = _STATE_Z[c["state"]]
            text[c["y"]][c["x"]] = (f"{c['state']}<br>n={c['n']} / t={c['t']:.0f}"
                                    f"<br>多樣性 d={c['d']:.2f}")
        colorscale = [[_STATE_Z[s] / 5, _STATE_COLOR[s]] for s in
                      (STATE_EMPTY, STATE_MISSING, STATE_LOW, STATE_HEALTHY,
                       STATE_FAKE, STATE_OVER)]
        fig = go.Figure(data=go.Heatmap(
            z=z, x=lx, y=ly, text=text, hoverinfo="text",
            colorscale=colorscale, zmin=0, zmax=5, showscale=False,
            xgap=3, ygap=3,
        ))
        lab_x, lab_y = _AXIS_LABEL.get(ax_x, ax_x), _AXIS_LABEL.get(ax_y, ax_y)
        fig.update_layout(
            height=560, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=lab_x, yaxis_title=lab_y,
            title=f"{lab_x} × {lab_y}　·　🟥缺 🟧偏缺 🟩健康 🟪假完整 🟦過多",
        )
        st.plotly_chart(fig, use_container_width=True, key="cov_heatmap")

        # (a) 真實分佈校正編輯器：每格設 高/中/低/不適用 先驗
        with st.expander("🎚 真實分佈校正（每格頻率先驗：高/中/低/不適用）"):
            st.caption("用粗分級先驗校正『每格該有多少』——稀有格設『低』就不會被誤判為缺，"
                       "現實不存在的組合設『不適用』排除於分母外。空白＝中（用地板值）。")
            df = pd.DataFrame([
                {"格": f"{c['x_label']} × {c['y_label']}",
                 "x": c["x"], "y": c["y"], "n": c["n"],
                 "頻率先驗": freq_classes.get((c["x"], c["y"]), "中")}
                for c in cells
            ])
            edited = st.data_editor(
                df[["格", "n", "頻率先驗"]], key="cov_freq_editor",
                hide_index=True, use_container_width=True, height=240,
                column_config={"頻率先驗": st.column_config.SelectboxColumn(
                    options=["高", "中", "低", "不適用"], required=True)},
                disabled=["格", "n"],
            )
            if st.button("套用校正", key="cov_apply_freq", use_container_width=True):
                new_fc = {}
                for row, c in zip(edited.itertuples(), cells):
                    v = row.頻率先驗
                    if v != "中":
                        new_fc[(c["x"], c["y"])] = v
                st.session_state["cov_freq_classes"] = new_fc
                st.toast("已套用真實分佈校正", icon="🎚")
                st.rerun()

    with col_side:
        st.markdown("**缺格清單（缺口大→小）**")
        gaps = [g for g in health["top_gaps"]
                if g["state"] in (STATE_EMPTY, STATE_MISSING, STATE_LOW)]
        if not gaps:
            st.success("沒有缺格——每格都達標。")
        with st.container(height=200):
            for g in gaps[:30]:
                lab = f"{g['x_label']} × {g['y_label']}"
                if st.button(f"🔴 {lab}　n={g['n']}/t={g['t']:.0f}（缺 {g['shortfall']:.0f}）",
                             key=f"cov_gap_{g['x']}_{g['y']}", use_container_width=True):
                    st.session_state["cov_active_cell"] = (g["x"], g["y"])
                    st.rerun()
        # 點任一格（含健康/假完整）看圖
        st.markdown("**檢視格內影像**")
        active = st.session_state.get("cov_active_cell")
        cell = next((c for c in cells if (c["x"], c["y"]) == active), None)
        if cell is None:
            st.caption("點左方缺格、或下方挑一格，看格內影像。")
            opts = {f"{c['x_label']} × {c['y_label']}（{c['state']} n={c['n']}）":
                    (c["x"], c["y"]) for c in cells if c["n"] > 0}
            pick = st.selectbox("挑一格", ["—"] + list(opts), key="cov_cell_pick")
            if pick != "—":
                st.session_state["cov_active_cell"] = opts[pick]
                st.rerun()
        else:
            st.caption(f"**{cell['x_label']} × {cell['y_label']}** · {cell['state']} · "
                       f"n={cell['n']} / t={cell['t']:.0f} · 多樣性 d={cell['d']:.2f}")
            if cell["state"] == STATE_FAKE:
                st.caption(":violet[假完整：量夠但多樣性低，多為近重複——建議去重而非再補。]")
            with st.container(height=480):
                cols = st.columns(3)
                for j, i in enumerate(cell["indices"][:30]):
                    with cols[j % 3]:
                        p = Path(records[i]["path"])
                        thumb = _thumb_or_none(p)
                        if thumb:
                            st.image(thumb, use_container_width=True)
                        else:
                            st.warning("⚠ 缺檔")

            # (b) 缺格一鍵撈候選：候選池就地設定（popover），免回 sidebar
            with st.popover("🔎 撈候選補此格", use_container_width=True):
                st.caption("候選池資料夾（通常是未標註的影像）")
                # 單一資料夾語義:📁 選新的取代舊的。widget 維持 text_area
                # (E2E 契約 .st-key-cov_pool_text textarea,headless 唯一入口)。
                st.button("📁 選候選池資料夾", key="cov_pool_pick",
                          use_container_width=True,
                          on_click=_pick_folder_into_text, args=("cov_pool_text", True))
                st.text_area("或貼上候選池路徑（取代目前選擇）", key="cov_pool_text",
                             height=80, label_visibility="collapsed",
                             placeholder="例：demo/pool 或 C:/data/unlabeled")
                if st.button("開始撈候選", key="cov_mine_btn",
                             use_container_width=True,
                             disabled=not st.session_state.get("cov_pool_text", "").strip()):
                    _mine_cell_candidates(cell, records, emb, model)
            _render_cov_candidates(cell, records)
            st.button("✕ 關閉", key="cov_cell_close",
                      on_click=lambda: st.session_state.pop("cov_active_cell", None))


_QUIZ_DEMO_DIR = Path(__file__).parent.parent / "demo" / "imagenette" / "train"


def _load_quiz_demo() -> None:
    st.session_state["quiz_folder_list"] = [_demo_classifier_dir()]
    st.session_state["_quiz_autorun"] = True
    _log_usage("quiz_demo_load")


def _quiz_answer(qid: int, label: str) -> None:
    st.session_state.setdefault("quiz_answers", {})[qid] = label
    st.session_state["quiz_pos"] = st.session_state.get("quiz_pos", 0) + 1


def _quiz_reset() -> None:
    for k in ("quiz_spec", "quiz_answers", "quiz_pos"):
        st.session_state.pop(k, None)


def _quiz_generate(records: list[dict], dis, n_q: int) -> None:
    spec = build_quiz(records, dis, n_questions=int(n_q))
    st.session_state["quiz_spec"] = spec
    st.session_state["quiz_answers"] = {}
    st.session_state["quiz_pos"] = 0
    # qid → image (+ bbox for object/box-mode), so multi-rater consensus can be
    # tied back to images/boxes (M2/M3)
    st.session_state["quiz_qid_map"] = {
        q["qid"]: {"path": str(records[q["record_idx"]].get("path", "")),
                   "image_path": str(records[q["record_idx"]].get("image_path", "") or ""),
                   "bbox": records[q["record_idx"]].get("bbox"),
                   "label": records[q["record_idx"]].get("label", "")}
        for q in spec["questions"] if 0 <= q["record_idx"] < len(records)}


def _quiz_skip(pos: int) -> None:
    st.session_state["quiz_pos"] = pos + 1


def _fleiss_from_csvs(answer_maps: list[dict[int, str]]) -> tuple[float, int]:
    """Fleiss kappa across raters' {qid: answer} maps on their common qids."""
    if len(answer_maps) < 2:
        return 0.0, 0
    common = set(answer_maps[0])
    for m in answer_maps[1:]:
        common &= set(m)
    common = sorted(common)
    if not common:
        return 0.0, 0
    cats = sorted({a for m in answer_maps for a in m.values()})
    cat_idx = {c: i for i, c in enumerate(cats)}
    mat = np.zeros((len(common), len(cats)))
    for r, qid in enumerate(common):
        for m in answer_maps:
            mat[r, cat_idx[m[qid]]] += 1
    return fleiss_kappa(mat), len(common)


def _render_quiz_quick_start() -> None:
    st.markdown("##### 快速開始")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, st.container(border=True):
        st.markdown("**① 貼資料夾**")
        st.caption("含類別子資料夾的影像資料夾——考卷會從爭議樣本出題。")
    with c2, st.container(border=True):
        st.markdown("**② 產生考卷**")
        st.caption("自動挑爭議題＋對照題＋換皮重測＋golden 定錨。")
    with c3, st.container(border=True):
        st.markdown("**③ 盲測作答**")
        st.caption("逐題盲答，算自我一致率與 vs golden；多人可算 Fleiss kappa。")
    mid = st.columns([2, 1.6, 2])[1]
    mid.button("✨ 用範例資料試跑（imagenette）", key="quiz_demo_btn",
               type="primary", use_container_width=True, on_click=_load_quiz_demo)


def _open_labeling_tool(tool_id: str = "module_026") -> bool:
    """Best-effort: ask the host engine to open a labeling tool. Uses
    CIM_CONTROL_PORT (injected by ToolProcessManager._make_env). Returns True if
    the start was accepted; False if we can't reach the engine (the caller then
    just tells the user to switch tools via the portal)."""
    import os
    import urllib.request
    port = os.environ.get("CIM_CONTROL_PORT")
    if not port:
        return False
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/tools/{tool_id}/start", method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return 200 <= resp.status < 300
    except Exception:  # noqa: BLE001
        return False


def _quiz_handoff_root() -> Path:
    import os
    base = os.environ.get("CIM_LOG_DIR") or str(Path(__file__).parent.parent / "output")
    return Path(base) / "lv_quiz_handoff"


# ── Unified LV → Labeling hand-over (every feature rides this) ────────────────
def _lv_class_opts() -> list[str]:
    """Label palette for the handoff: the dataset's classes."""
    cn = st.session_state.get("viz_classes")
    if cn:
        return list(cn)
    recs = st.session_state.get("viz_records") or st.session_state.get("quiz_records") or []
    opts = sorted({r.get("label", "") for r in recs if r.get("label")})
    return opts or ["object"]


def _send_to_labeling_ui(records: list[dict], indices, *, source: str, task: str,
                         class_opts: list[str] | None = None, label: str = "📤 送到 Labeling 標註",
                         key: str | None = None, help: str | None = None, **kw) -> None:
    """One shared button every LV feature drops in: export the subset to a
    content-addressed handoff folder and switch to Labeling. The hand-over is
    one-way (LV → Labeling): annotation and feedback complete on the Labeling
    side, with no return to LV. State lives on disk (_pending.json) and is
    consumed by Labeling (module_026 auto-prefills the source; module_014 marks
    the batch done after export)."""
    import labeling_handoff as LH
    idxs = sorted(set(int(i) for i in indices))
    n = len(idxs)
    if st.button(f"{label}（{n}）", key=key or f"send_lbl_{source}",
                 use_container_width=True, disabled=(n == 0),
                 help=help or "把這批影像送到 Labeling 工具標註；標完在 Labeling 端"
                              "「匯出 / 回傳」匯出即完成，不用回 LV。"):
        out = LH.send_to_labeling(records, idxs, source=source, task=task,
                                  class_options=class_opts or _lv_class_opts(),
                                  manifest=st.session_state.get("viz_manifest"), **kw)
        if out is None:
            st.warning("沒有可送出的影像。")
            return
        _log_usage("send_to_labeling", source=source, n=n)
        # 不在這裡叫 engine 啟動 Labeling：單工具架構下 start(module_026) 會先 stop() 把
        # 正在跑的 LV 自己關掉（Streamlit 連線中斷）。改由 _render_send_confirmation 用
        # postMessage 請 portal 自動切到 Labeling——module_026 會自動帶入此批路徑。
        # 單向交棒：標完在 Labeling 端匯出即完成，不回 LV。
        st.session_state["_lv_just_sent"] = (n, source)
        st.rerun()


def _render_send_confirmation() -> None:
    """送出後的確認 + 自動切換到 Labeling。

    LV 對 Labeling 是『單向交棒（送出即忘）』：標註與回饋都在 Labeling 端完成，
    不需回 LV，所以這裡只負責「送出成功 + 切換」，不再有交接箱／待標清單／讀回。
    批次狀態仍寫進磁碟 _pending.json，由 Labeling 端消費（module_026 自動帶入來源、
    module_014 匯出後標記完成）。"""
    sent = st.session_state.pop("_lv_just_sent", None)
    if not sent:
        return
    st.success(
        f"✅ 已送 {sent[0]} 張到 Labeling（{sent[1]}），正在切換到 Labeling 工具…\n\n"
        "在「資料來源」按「執行」載入此批（路徑已自動帶入）即可標註；"
        "標完到「匯出 / 回傳」匯出即完成，**不用回 LV**。\n\n"
        ":gray[（若沒自動切換：請用上方「工作流程」下拉手動切到 Labeling。"
        "批次已存到磁碟，跨重啟不丟。）]")
    # 請 portal（最上層視窗）自動切到 Labeling 整張 sheet（sheet-annotation）——
    # 它才有「資料來源 / 標注工作台 / 審查 / 匯出」四個 tab；指向單一 module_026 會
    # 沒有 tab 列。資料來源 tab 會自動帶入此批路徑。若 portal 不支援 OPEN_TOOL 則
    # 無事發生，使用者照上面提示手動切。
    import streamlit.components.v1 as _components
    _components.html(
        '<script>try{window.top.postMessage({source:"cim-platform",'
        'type:"OPEN_TOOL",payload:{toolId:"sheet-annotation"},'
        'timestamp:new Date().toISOString()},"*");}catch(e){}</script>',
        height=0)


def _labeling_readback_ui() -> None:
    """標註回饋：列出送去 Labeling 的批次，套用已完成的標籤變更回目前 records。

    LV 對此段路徑是無狀態重啟的（單一 active tool，切到 Labeling 時 LV 進程被
    engine 砍掉重開），所以讀回是「使用者主動觸發」的一步，不是自動輪詢——
    契約與 apply_readback() 見 labeling_handoff.py。"""
    import labeling_handoff as LH
    st.caption("列出送去 Labeling 標註的批次；標完、匯出後回來這裡按「套用讀回結果」，"
               "把新標籤套進目前載入的資料。不用回 Labeling，也不會自動輪詢。")
    pending = LH.list_pending()
    if not pending:
        st.info("目前沒有任何送出紀錄。到「Visualize Embeddings」送一批到 Labeling 後，"
                "才會出現在這裡。")
        return

    def _row_label(row: dict) -> str:
        return (f"{row.get('source', '?')} · {row.get('task', '?')} · "
                f"{row.get('created_at', '')} · "
                f"{row.get('n_annotated', 0)}/{row.get('n_total', 0)}")

    idx = st.selectbox("送出批次", list(range(len(pending))),
                       format_func=lambda i: _row_label(pending[i]),
                       key="readback_batch_idx")
    row = pending[idx]
    st.write(f"已標註 {row.get('n_annotated', 0)} / {row.get('n_total', 0)}")

    records = st.session_state.get("viz_records")
    if not records:
        st.warning("請先在「Visualize Embeddings」載入資料集，再回來套用讀回結果。")
        return

    if st.button("📥 套用讀回結果", key="readback_apply_btn", use_container_width=True):
        st.session_state["readback_last_changes"] = LH.apply_readback(row["dir"], records)

    changes = st.session_state.get("readback_last_changes")
    if changes is not None:
        if changes:
            st.success(f"已套用 {len(changes)} 筆變更")
            st.dataframe(
                [{"檔名": c["filename"], "舊標籤": c["old_label"], "新標籤": c["new_label"]}
                 for c in changes],
                use_container_width=True, hide_index=True)
        else:
            st.info("沒有新的標籤變更（可能還沒人標，或標籤沒變）。")


def _quiz_ui() -> None:
    st.markdown("##### 組考卷 · 標註者一致性盲測")
    st.caption("把爭議樣本變成盲測考卷，量「同一人會不會自打嘴巴」與「跨人是否一致」。"
               "只量對既有案例的判定穩定性，量不到庫外新型或 golden 本身對錯。")

    with st.sidebar:
        st.markdown("**① 資料夾**")
        _folder_picker_list("quiz_folder_list",
                            add_help="結構需為 資料夾／類別／影像。")
        all_models = available_models()
        if not all_models:
            st.error("models/ 內找不到模型檔。"); return
        st.markdown("**② 模型**")
        model = st.selectbox("模型", all_models, label_visibility="collapsed",
                             index=_default_model_index(all_models),
                             help="用來找爭議樣本（kNN 標籤分歧）與對照題（以圖搜圖）。")
        run = st.button("▶ 載入資料", use_container_width=True, key="run_quiz",
                        type="primary")

    if st.session_state.pop("_quiz_autorun", False):
        run = True
    if run:
        folders = [Path(f) for f in st.session_state.get("quiz_folder_list", [])]
        missing = [str(p) for p in folders if not p.exists()]
        if not folders:
            st.error("請先選擇資料夾。"); return
        if missing:
            st.error(f"資料夾不存在：{', '.join(missing)}"); return
        records = discover_images_classifier(folders)
        if not records or len({r["label"] for r in records}) < 2:
            st.error("需至少 2 個類別、folder/類別/影像 結構。"); return
        records = _filter_records_readable(records, "quiz")  # 壞檔防呆
        if not records or len({r["label"] for r in records}) < 2:
            st.error("需至少 2 個類別、folder/類別/影像 結構。"); return
        _collapse_sidebar()
        embed_fn = load_model(model)
        with st.status("計算中…", expanded=True):
            paths = [r["path"] for r in records]
            cache = _dataset_cache_dir(folders[0], f"embeddings_{model}") / "embeddings.npz"
            emb = extract_embeddings(paths, embed_fn, cache_path=cache)
            k = min(10, len(records) - 1)
            dis = compute_label_disagreement(emb, [r["label"] for r in records], k=k)
        st.session_state["quiz_records"] = records
        st.session_state["quiz_disagreement"] = dis
        _quiz_reset()
        # 從資料夾載入＝放棄任何「嵌入覆蓋圖」送來的候選 inbound 狀態
        st.session_state.pop("quiz_inbound", None)
        st.session_state.pop("quiz_class_opts", None)
        st.toast(f"已載入 {len(records)} 張影像", icon="✅")

    if "quiz_records" not in st.session_state:
        _render_quiz_quick_start()
        return

    records = st.session_state["quiz_records"]
    dis = st.session_state["quiz_disagreement"]
    # 補洞候選送來時用完整資料集類別當作答選項（暫定標籤只涵蓋部分類別）
    class_opts = (st.session_state.get("quiz_class_opts")
                  or sorted({r["label"] for r in records}))
    if st.session_state.get("quiz_inbound"):
        st.info("ℹ 這批題目是從別的工具送來的（嵌入覆蓋圖補洞候選 / 策展購物車）。"
                "逐題盲標即可，最後匯出作答 CSV——量的是標註者一致性，不改資料集。")

    # ── 出題（尚無考卷）──
    if "quiz_spec" not in st.session_state:
        st.markdown("**產生考卷**")
        c1, c2 = st.columns([1, 3])
        n_q = c1.number_input("題數", min_value=4, max_value=min(40, len(records)),
                              value=min(16, len(records)), key="quiz_n")
        c2.caption("配比：爭議題 / 對照題(distractor) / 換皮重測 / golden 定錨。"
                   "換皮只用幾何變換（裁切/旋轉/翻轉），不動對比亮度。")
        st.button("📝 產生考卷", key="quiz_gen", type="primary",
                  on_click=_quiz_generate, args=(records, dis, int(n_q)))
        _render_quiz_multirater()
        return

    quiz = st.session_state["quiz_spec"]
    questions = quiz["questions"]
    answers = st.session_state.setdefault("quiz_answers", {})
    pos = st.session_state.get("quiz_pos", 0)

    # 組考卷為 LV 內建盲標（量標註者一致性）——不送 Labeling、不讀回，純單向工具
    # ── 作答中 ──
    if pos < len(questions):
        q = questions[pos]
        st.progress((pos) / len(questions), text=f"第 {pos + 1} / {len(questions)} 題")
        col_img, col_ans = st.columns([3, 2], gap="medium")
        with col_img:
            p = Path(records[q["record_idx"]]["path"])
            img = safe_open_image(p)  # 壞檔回 None → 不當 Image 用
            if img is None:
                st.warning(f"無法讀取：{p}")
            else:
                if q["skin"]:
                    img = geometric_skin(img, q["skin"])
                st.image(img, use_container_width=True)
        with col_ans:
            st.markdown("**這張屬於哪一類？**")
            st.caption("（盲測：不顯示原標籤；憑你的判斷選。）")
            for c in class_opts:
                st.button(c, key=f"quiz_ans_{q['qid']}_{c}", use_container_width=True,
                          on_click=_quiz_answer, args=(q["qid"], c))
            st.button("跳過", key=f"quiz_skip_{q['qid']}", use_container_width=True,
                      on_click=_quiz_skip, args=(pos,))
            st.button("✕ 放棄此卷", key="quiz_abandon", on_click=_quiz_reset)
        return

    # ── 評分 ──
    report = score_quiz(answers, quiz)
    # 把自我一致率留給體檢卡當 S1（概念歧義度）預填值
    st.session_state["quiz_last_consistency"] = report["self_consistency"]
    st.markdown("**作答完成 · 成績**")
    m1, m2, m3 = st.columns(3)
    sc, vg = report["self_consistency"], report["vs_golden"]
    m1.metric("自我一致率", f"{sc * 100:.0f}%",
              "✅ 達標(≥90%)" if report["self_pass"] else "⚠ 未達標",
              help="同一題換皮重測你答得一不一致；量你會不會自打嘴巴。")
    m2.metric("vs golden 一致", f"{vg * 100:.0f}%",
              "✅ 達標(≥85%)" if report["golden_pass"] else "⚠ 未達標",
              help="golden / 對照題你和標準答案一致率。")
    m3.metric("作答題數", f"{report['n_answered']}/{report['n_questions']}")
    if report["n_repeat_pairs"] == 0:
        st.caption(":gray[（本卷無換皮重測對，自我一致率以 0 計——增加題數可納入重測。）]")
    csv = _rows_to_csv(["qid", "answer"],
                       [[q, a] for q, a in sorted(answers.items())])
    st.download_button("⬇ 匯出作答 CSV（給多人一致性用）", data=csv,
                       file_name="quiz_answers.csv", mime="text/csv",
                       key="quiz_answers_csv")
    # 下游回推：把考卷量出的爭議樣本收回購物車（策展迴圈 清單→考卷→爭議→清單）
    quiz_recs = st.session_state.get("quiz_records", [])
    q_idx = sorted({int(q["record_idx"]) for q in questions
                    if 0 <= int(q["record_idx"]) < len(quiz_recs)})
    if q_idx:
        st.button(f"🛒 把這 {len(q_idx)} 張爭議影像加入購物車", key="quiz_add_cart",
                  use_container_width=True,
                  help="考卷量出的爭議樣本收進跨工具購物車（標 source=考卷低一致），"
                       "可再一鍵送灰帶覆核裁決。",
                  on_click=_batch_add, args=(quiz_recs, q_idx, "quiz_disputed"))
    st.button("🔁 再出一卷", key="quiz_again", on_click=_quiz_reset)
    st.divider()
    _render_quiz_multirater()


def _render_quiz_multirater() -> None:
    """跨人一致性：上傳 ≥2 份作答 CSV → Fleiss kappa（共同題上計算）。"""
    with st.expander("👥 多人一致性（Fleiss kappa）"):
        st.caption("上傳 2 份以上不同標註者的作答 CSV（qid,answer），"
                   "在共同題上算 Fleiss kappa（≥0.75 為及格）。")
        files = st.file_uploader("作答 CSV（可多選）", type="csv",
                                 accept_multiple_files=True, key="quiz_multi_files")
        if files and len(files) >= 2:
            maps = []
            for f in files:
                m = {}
                for line in f.getvalue().decode("utf-8").splitlines()[1:]:
                    parts = line.split(",")
                    if len(parts) >= 2 and parts[0].strip().isdigit():
                        m[int(parts[0])] = parts[1].strip()
                maps.append(m)
            kappa, n_common = _fleiss_from_csvs(maps)
            c1, c2 = st.columns(2)
            c1.metric("Fleiss kappa", f"{kappa:.3f}",
                      "✅ 達標(≥0.75)" if kappa >= 0.75 else "⚠ 未達標")
            c2.metric("共同題數", n_common)
            if n_common == 0:
                st.warning("這些作答檔沒有共同題（qid 不重疊）。")

            # ── M2：把逐題投票聚合成「共識子集 + 灰帶」——這把尺給『評估』當靶 ──
            import csv as _csv
            import io

            from quiz import consensus_labels
            cons = consensus_labels(maps)
            qid_map = st.session_state.get("quiz_qid_map", {})
            cons_rows, gray_rows = [], []
            for qid, c in cons.items():
                info = qid_map.get(qid, {})
                bbox = info.get("bbox")
                if bbox:  # box-level: key by the full image + the defect box
                    img = info.get("image_path") or info.get("path") or ""
                    fname = Path(img).name if img else f"qid_{qid}"
                else:
                    fname = (Path(info["path"]).name if info.get("path")
                             else f"qid_{qid}")
                row = {"filename": fname, "consensus": c["consensus"],
                       "label": c["label"], "agreement": c["agreement"],
                       "n_votes": c["n_votes"], "qid": qid}
                if bbox:
                    row.update({k: round(float(v), 6) for k, v in
                                zip(("cx", "cy", "w", "h"), bbox)})
                (cons_rows if c["consensus"] else gray_rows).append(row)
            d1, d2 = st.columns(2)
            d1.metric("共識題（可當評估靶）", len(cons_rows))
            d2.metric("灰帶題（送灰帶覆核）", len(gray_rows))
            if cons and not qid_map:
                st.caption(":gray[（本機沒有這份考卷的出題紀錄，匯出以 qid 為鍵；"
                           "在同一 session 產生考卷後再上傳作答，才能對回影像檔名。）]")
            all_rows = cons_rows + gray_rows
            if all_rows:
                box_level = any("cx" in r for r in all_rows)
                cols = (["filename", "consensus", "label", "agreement", "n_votes",
                         "qid"] + (["cx", "cy", "w", "h"] if box_level else []))
                buf = io.StringIO()
                w = _csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(all_rows)
                st.download_button(
                    f"⬇ 匯出共識子集 CSV（{'框層級' if box_level else '圖層級'}，"
                    "給『評估』tab 當 ③ 共識子集）",
                    buf.getvalue(), "consensus_set.csv", "text/csv",
                    key="quiz_consensus_csv", use_container_width=True)
            if gray_rows:
                gbox = any("cx" in r for r in gray_rows)
                gcols = (["filename", "label", "agreement", "n_votes", "qid"]
                         + (["cx", "cy", "w", "h"] if gbox else []))
                gbuf = io.StringIO()
                gw = _csv.DictWriter(gbuf, fieldnames=gcols, extrasaction="ignore")
                gw.writeheader()
                gw.writerows(gray_rows)
                st.download_button(
                    "⬇ 匯出灰帶清單 CSV（送灰帶覆核裁決）", gbuf.getvalue(),
                    "gray_band.csv", "text/csv", key="quiz_gray_csv",
                    use_container_width=True)
            st.caption(":gray[共識子集＝多人一致的題，拿去『評估』tab 當靶，recall 才"
                       "落在穩定的尺上；灰帶題＝票分裂、沒有真值，送灰帶覆核。]")
        elif files:
            st.info("至少需要 2 份作答 CSV。")


_GRAY_DEMO_DIR = Path(__file__).parent.parent / "demo" / "imagenette" / "train"


def _load_gray_demo() -> None:
    st.session_state["gray_folder_list"] = [_demo_classifier_dir()]
    st.session_state["_gray_autorun"] = True
    _log_usage("gray_demo_load")


def _gray_dispose(indices, action: str, soft_map: dict | None = None) -> None:
    """Record a disposition for a batch of gray samples (站內：分級 soft / 排除).
    Relabeling itself goes to Labeling, not here."""
    disp = st.session_state.setdefault("gray_disp", {})
    for i in indices:
        i = int(i)
        if action == "soft" and soft_map and i in soft_map:
            lbl, conf = soft_map[i]
            disp[i] = {"action": "soft", "soft_label": lbl, "confidence": conf}
        else:
            disp[i] = {"action": action}
    st.toast(f"已處置 {len(indices)} 筆 → {action}", icon="✅")


def _gray_enter_focus(pos: int) -> None:
    st.session_state["gray_pos"] = int(pos)
    st.session_state["gray_mode"] = "focus"


def _gray_nav(step: int) -> None:
    st.session_state["gray_pos"] = st.session_state.get("gray_pos", 0) + int(step)


def _gray_set_mode(mode: str) -> None:
    st.session_state["gray_mode"] = mode


def _render_gray_quick_start() -> None:
    st.markdown("##### 快速開始")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, st.container(border=True):
        st.markdown("**① 貼資料夾**")
        st.caption("含類別子資料夾的影像——自動撈出最爭議（灰帶）的樣本。")
    with c2, st.container(border=True):
        st.markdown("**② 看 backlog + 總覽**")
        st.caption("一眼看整個資料集有多少要 audit、各多可疑；總覽一次列出整批。")
    with c3, st.container(border=True):
        st.markdown("**③ 分流三選一**")
        st.caption("可解的送 Labeling 裁決；模稜兩可的給分級標籤或標記排除。")
    mid = st.columns([2, 1.6, 2])[1]
    mid.button("✨ 用範例資料試跑（imagenette）", key="gray_demo_btn",
               type="primary", use_container_width=True, on_click=_load_gray_demo)


def _gray_thumb(records, i):
    t = _thumb_or_none(Path(records[i]["path"]))
    return t


def _gray_focus_view(records, emb, anchors, anchor_indices, disp, view,
                     class_opts, soft_map) -> None:
    """焦點對照（接回舊版的一對一對）：原標類錨例 ｜ 灰帶 ｜ 最近他類錨例 並排 +
    cosine 距離 + 上/下一張 + 單筆三選一動作。"""
    pos = max(0, min(st.session_state.get("gray_pos", 0), len(view) - 1))
    it = view[pos]
    cur_i, orig = it["i"], it["orig"]
    orig_anchor = anchors.get(orig)
    other_anchors = [a for c, a in anchors.items() if c != orig and a is not None]
    orig_idx, orig_dist = (nearest_anchor(emb, cur_i, [orig_anchor])
                           if orig_anchor is not None else (None, float("inf")))
    other_idx, other_dist = nearest_anchor(emb, cur_i, other_anchors)

    n1, n2, n3 = st.columns([1, 2.4, 1])
    n1.button("← 上一張", key="gray_prev", use_container_width=True,
              disabled=pos == 0, on_click=_gray_nav, args=(-1,))
    warn = " ⚠指向他類" if it["points_other"] else ""
    n2.markdown(f"第 **{pos + 1} / {len(view)}** 筆　·　分歧 {it['score']:.2f}　·　"
                f"{orig}→{it['anchor']}{warn}　·　{_rec_fname(records[cur_i])}")
    n3.button("下一張 →", key="gray_next", use_container_width=True,
              disabled=pos >= len(view) - 1, on_click=_gray_nav, args=(1,))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption(f"原標類錨例 · **{orig}**"
                   + (f"　cos {orig_dist:.3f}" if orig_idx is not None else ""))
        t = _gray_thumb(records, orig_idx) if orig_idx is not None else None
        if t:
            st.image(t, use_container_width=True)
        else:
            st.caption("（無）")
    with c2:
        st.caption(f"🌫 灰帶候選 · 原標 **{orig}**")
        t = _gray_thumb(records, cur_i)
        if t:
            st.image(t, use_container_width=True)
        else:
            st.warning("⚠缺檔")
    with c3:
        oc = records[other_idx]["label"] if other_idx is not None else "—"
        st.caption(f"最近他類錨例 · **{oc}**"
                   + (f"　cos {other_dist:.3f}" if other_idx is not None else ""))
        t = _gray_thumb(records, other_idx) if other_idx is not None else None
        if t:
            st.image(t, use_container_width=True)
        else:
            st.caption("（無他類錨例）")

    if orig_idx is not None and other_idx is not None:
        closer = "他類" if other_dist < orig_dist else "原標類"
        st.caption(f":gray[它離 **{closer}** 錨例較近（原標 {orig_dist:.3f} vs 他類 "
                   f"{other_dist:.3f}）。明顯靠他類 → 送 Labeling 改標；兩邊都不近 → 分級/排除。]")

    d = disp.get(cur_i)
    cur_badge = {"soft": "🏷已分級", "exclude": "🚫已排除"}.get(
        (d or {}).get("action"), "未處置")
    st.markdown(f"**這一張的處置**（目前：{cur_badge}）")
    b1, b2, b3 = st.columns(3)
    with b1:
        _send_to_labeling_ui(
            records, [cur_i], source="gray-zone-focus", task=LH.TASK_ADJUDICATE,
            label="📤 送這張裁決", key=f"gray_f_send_{cur_i}", class_opts=class_opts or None,
            original_labels={cur_i: records[cur_i].get("label", "")},
            help="送 Labeling 正式對錨改標。")
    b2.button("🏷 給這張分級", key=f"gray_f_soft_{cur_i}", use_container_width=True,
              on_click=_gray_dispose, args=([cur_i], "soft", soft_map))
    b3.button("🚫 排除這張", key=f"gray_f_excl_{cur_i}", use_container_width=True,
              on_click=_gray_dispose, args=([cur_i], "exclude"))
    st.button("← 回總覽（看 backlog／批次）", key="gray_back",
              on_click=_gray_set_mode, args=("overview",))


def _gray_zone_ui() -> None:
    st.markdown("##### 灰帶覆核 · 分流閘")

    with st.sidebar:
        st.markdown("**① 資料夾**")
        _folder_picker_list("gray_folder_list",
                            add_help="結構需為 資料夾／類別／影像。")
        all_models = available_models()
        if not all_models:
            st.error("models/ 內找不到模型檔。"); return
        st.markdown("**② 模型**")
        model = st.selectbox("模型", all_models, label_visibility="collapsed",
                             index=_default_model_index(all_models))
        n_q = st.number_input("本批張數（最爭議的前 N）", min_value=1, max_value=200,
                              value=20, key="gray_n")
        run = st.button("▶ 建立分流佇列", use_container_width=True, key="run_gray",
                        type="primary")

    if st.session_state.pop("_gray_autorun", False):
        run = True
    if run:
        folders = [Path(f) for f in st.session_state.get("gray_folder_list", [])]
        missing = [str(p) for p in folders if not p.exists()]
        if not folders:
            st.error("請先選擇資料夾。"); return
        if missing:
            st.error(f"資料夾不存在：{', '.join(missing)}"); return
        records = discover_images_classifier(folders)
        if not records or len({r["label"] for r in records}) < 2:
            st.error("需至少 2 個類別、folder/類別/影像 結構。"); return
        records = _filter_records_readable(records, "gray")  # 壞檔防呆
        if not records or len({r["label"] for r in records}) < 2:
            st.error("需至少 2 個類別、folder/類別/影像 結構。"); return
        _collapse_sidebar()
        embed_fn = load_model(model)
        with st.status("計算中…", expanded=True):
            paths = [r["path"] for r in records]
            cache = _dataset_cache_dir(folders[0], f"embeddings_{model}") / "embeddings.npz"
            emb = extract_embeddings(paths, embed_fn, cache_path=cache)
            labels = [r["label"] for r in records]
            dis = compute_label_disagreement(emb, labels, k=min(10, len(records) - 1))
        queue = select_gray_zone(dis, int(n_q))
        anchors = {}  # 每類最不爭議（最明確）的一張當錨例
        for c in sorted(set(labels)):
            cand = [i for i in range(len(records)) if labels[i] == c]
            anchors[c] = min(cand, key=lambda i: dis[i]) if cand else None
        st.session_state["gray_records"] = records
        st.session_state["gray_emb"] = emb
        st.session_state["gray_queue"] = queue
        st.session_state["gray_anchors"] = anchors
        st.session_state["gray_dis"] = dis
        st.session_state["gray_disp"] = {}
        st.session_state.pop("gray_inbound", None)
        st.session_state.pop("gray_mode", None)  # 新批重新自適應 overview/focus
        st.session_state.pop("gray_pos", None)
        st.toast(f"佇列建立：{len(queue)} 筆灰帶候選", icon="🌫")

    if "gray_records" not in st.session_state:
        _render_gray_quick_start()
        return

    if st.session_state.get("gray_inbound"):
        st.info("ℹ 這批佇列是從別的工具送來的（散點框選 / 策展購物車 / 組考卷灰帶清單）。")

    records = st.session_state["gray_records"]
    emb = st.session_state["gray_emb"]
    queue = st.session_state["gray_queue"]
    anchors = st.session_state["gray_anchors"]
    dis = st.session_state.get("gray_dis")
    disp = st.session_state.setdefault("gray_disp", {})
    class_opts = sorted({r["label"] for r in records})
    anchor_indices = [a for a in anchors.values() if a is not None]

    # ── mode 預設（依佇列大小自適應：小批直接逐筆、大批先總覽）──
    if "gray_mode" not in st.session_state:
        st.session_state["gray_mode"] = "focus" if len(queue) <= 8 else "overview"
    st.session_state.setdefault("gray_pos", 0)
    mode = st.session_state["gray_mode"]

    # ── 每筆 triage 資料（保留 anchor 索引與 cosine 距離，不再丟）──
    items = []
    for i in queue:
        a_idx, a_dist = nearest_anchor(emb, i, anchor_indices)
        a_cls = records[a_idx]["label"] if a_idx is not None else None
        orig = records[i]["label"]
        items.append({"i": i, "score": float(dis[i]) if dis is not None else 0.0,
                      "orig": orig, "anchor": a_cls, "anchor_idx": a_idx,
                      "anchor_dist": a_dist,
                      "points_other": a_cls is not None and a_cls != orig})

    # 篩選（讀持久值算 view）+ 排序（高分歧在前，輕重一眼分）
    flt = st.session_state.get("gray_filter", "全部")
    if flt.startswith("⚠"):
        view = [it for it in items if it["points_other"]]
    elif flt.startswith("同類"):
        view = [it for it in items if not it["points_other"]]
    else:
        view = list(items)
    view.sort(key=lambda it: -it["score"])
    view_idx = [it["i"] for it in view]

    def _soft_conf(it) -> float:
        d = it["anchor_dist"]
        return (round(max(0.0, 1.0 - d), 2) if d not in (None, float("inf"))
                else round(1.0 - it["score"], 2))
    soft_map = {it["i"]: (it["anchor"] or it["orig"], _soft_conf(it)) for it in view}

    # ===== 焦點對照（一對一對）=====
    if mode == "focus" and view:
        _gray_focus_view(records, emb, anchors, anchor_indices, disp, view,
                         class_opts, soft_map)
        return

    # ===== 總覽（看 backlog + 批次分流）=====
    # 說明＋backlog 細節收進可折疊區（預設折起，關鍵數字留在標題列）——把版面還給總覽
    if dis is not None:
        from interaction import gray_zone_summary
        s = gray_zone_summary(dis)
        with st.expander(
                f"ℹ️ 灰帶待 audit {s['n_gray']}/{s['n_total']}（{s['pct_gray']}%）　"
                f"🔴{s['high']} 🟡{s['mid']} 🟢{s['low']}　— 點開看說明", expanded=False):
            st.caption("組考卷／散點分歧送來的『灰帶』樣本在這裡 **triage**：一眼看 backlog "
                       "有多少、多可疑，再**分流三選一**——可解的送 Labeling 正式裁決、模稜兩可的"
                       "給分級標籤或標記排除。**改標走 Labeling，不在這裡做**；分級/排除回流評估、"
                       "不寫回資料集。")
            st.caption(f"嚴重度　:red[🔴 高 {s['high']}]　:orange[🟡 中 {s['mid']}]　"
                       f":green[🟢 低 {s['low']}]　·　本批前 {len(queue)} 筆"
                       "（想多看調左側『本批張數』）。分歧度＝鄰域標籤不一致比例，"
                       "探索線索、**非錯標判決**。")

    f_col, ab1, ab2, ab3 = st.columns([3, 1.3, 1.3, 1.3])
    with f_col:
        st.radio("分流篩選", ["全部", "⚠ 指向他類（可解→送 Labeling）",
                            "同類高分歧（模稜兩可→分級/排除）"],
                 horizontal=True, key="gray_filter", label_visibility="collapsed")
    with ab1:
        _send_to_labeling_ui(
            records, view_idx, source="gray-zone", task=LH.TASK_ADJUDICATE,
            label="📤 送裁決", key="gray_to_lbl", class_opts=class_opts or None,
            original_labels={i: records[i].get("label", "") for i in view_idx},
            payload={"anchors": {str(c): {"idx": int(a), "label": records[a].get("label", ""),
                                          "file": Path(records[a]["path"]).name}
                                 for c, a in anchors.items() if a is not None}},
            help="把篩選後的這批送 Labeling 正式對錨裁決；標完在 Labeling 端匯出即完成。")
    ab2.button("🏷 分級", key="gray_soft_btn", use_container_width=True,
               disabled=not view_idx, on_click=_gray_dispose,
               args=(view_idx, "soft", soft_map),
               help="這批給分級 soft label（=最近錨例類別，信賴=1−錨例距離），以群體處理。")
    ab3.button("🚫 排除", key="gray_exclude_btn", use_container_width=True,
               disabled=not view_idx, on_click=_gray_dispose,
               args=(view_idx, "exclude"), help="標為灰帶排除——評估 recall 不計入。")

    _badge = {"soft": "🏷分級", "exclude": "🚫排除"}
    st.markdown(f"**總覽（{len(view)} 筆）**　:gray[點「🔍對照」看大圖三方對照·高分歧在前]")
    with st.container(height=560):
        cols = st.columns(5)
        for j, it in enumerate(view):
            with cols[j % 5], st.container(border=True):
                sc = it["score"]
                col = "red" if sc >= 0.6 else ("orange" if sc >= 0.3 else "green")
                warn = " :red[⚠]" if it["points_other"] else ""
                badge = _badge.get((disp.get(it["i"]) or {}).get("action"), "")
                st.markdown(f':{col}[● {sc:.2f}]{warn} {it["orig"]}→{it["anchor"]} {badge}'
                            f'｜{_rec_fname(records[it["i"]])}')
                t = _gray_thumb(records, it["i"])
                if t:
                    st.image(t, use_container_width=True)
                else:
                    st.warning("⚠缺檔")
                st.button("🔍對照", key=f"gray_focus_{it['i']}", use_container_width=True,
                          on_click=_gray_enter_focus, args=(j,))

    st.caption(f":gray[已站內處置 {len(disp)} 筆（分級/排除）；送 Labeling 的已交棒、不在此計。]")
    if disp:
        import csv as _csv
        import io
        buf = io.StringIO()
        w = _csv.DictWriter(buf, fieldnames=["filename", "score", "orig_label",
                                             "anchor_label", "action", "soft_label",
                                             "confidence"])
        w.writeheader()
        for it in items:
            d = disp.get(it["i"])
            if not d:
                continue
            w.writerow({"filename": Path(records[it["i"]]["path"]).name,
                        "score": round(it["score"], 3), "orig_label": it["orig"],
                        "anchor_label": it["anchor"] or "", "action": d["action"],
                        "soft_label": d.get("soft_label", ""),
                        "confidence": d.get("confidence", "")})
        st.download_button("⬇ 匯出灰帶處置 CSV（分級/排除 → 給評估排除/下游併入）",
                           buf.getvalue(), "gray_disposition.csv", "text/csv",
                           key="gray_disp_csv", use_container_width=True)
    st.button("🛒 佇列加入策展購物車", key="gray_add_cart", use_container_width=True,
              on_click=_batch_add, args=(records, list(queue), "gray"))
    st.caption(":gray[定位：灰帶覆核＝組考卷下游的 triage＋分流。改標走 Labeling、"
               "分級/排除留站內並回流評估；不直接寫回資料集。]")


# ── 評估：在組考卷共識子集上量逐型態 recall ─────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _eval_gt_by_image(folder: Path, names: list[str]) -> dict[str, list[dict]]:
    """Build {image_basename: [GT box dicts]} from a folder's YOLO labels.
    Pure (no streamlit) so the data path is headless-testable."""
    from interaction import parse_yolo_boxes, yolo_label_path_for

    def cname(cid: int) -> str:
        return names[cid] if 0 <= cid < len(names) else f"class_{cid}"

    gt: dict[str, list[dict]] = {}
    images_dir = folder / "images"
    for img in sorted(images_dir.rglob("*")):
        if img.suffix.lower() not in _IMG_EXTS:
            continue
        boxes = parse_yolo_boxes(yolo_label_path_for(img))
        if boxes:
            gt[img.name] = [{"cls": cname(c), "cx": cx, "cy": cy, "w": w, "h": h}
                            for c, cx, cy, w, h in boxes]
    return gt


def _eval_consensus_by_image(csv_text: str, gt_by_image: dict) -> tuple[dict, int, int]:
    """Parse a 組考卷 consensus CSV → per-GT-box consensus flags. Supports both
    image-level (filename,consensus,…) and box-level (…,cx,cy,w,h) CSVs via
    evaluation.consensus_flags. Returns (consensus_by_image, n_consensus_boxes,
    n_gray_boxes)."""
    import csv as _csv
    import io

    from evaluation import consensus_flags
    rows = list(_csv.DictReader(io.StringIO(csv_text)))
    return consensus_flags(rows, gt_by_image)


def _load_eval_demo() -> None:
    """One-click 評估 demo: synthesize predictions + a consensus subset from the
    bundled coco8 GT so the tool's purpose (per-type recall + escape gallery +
    gray-band exclusion) is visible without any uploads."""
    from evaluation import consensus_flags, evaluate_detections
    folder = Path(_demo_detection_dir())
    names = _classes_txt_nested(folder) or []
    gt = _eval_gt_by_image(folder, names)
    if not gt:
        return
    imgs = list(gt)
    miss_img = imgs[-1]                       # all its boxes become escapes (FN)
    gray_img = imgs[-2] if len(imgs) >= 2 else None  # excluded from recall
    preds = {f: ([] if f == miss_img else [{**b, "score": 0.9} for b in bx])
             for f, bx in gt.items()}
    rows = [{"filename": f, "consensus": f != gray_img,
             "cx": b["cx"], "cy": b["cy"], "w": b["w"], "h": b["h"]}
            for f, bx in gt.items() for b in bx]
    cby, n_c, n_g = consensus_flags(rows, gt)
    res = evaluate_detections(gt, preds, consensus_by_image=cby)
    st.session_state["_eval_result"] = {
        "res": res, "folder": str(folder), "used_consensus": True,
        "n_cons_img": n_c, "n_gray_img": n_g, "n_pred_img": len(preds),
        "is_demo": True}
    _log_usage("eval_demo_load")


def _render_eval_quick_start() -> None:
    st.markdown("##### 快速開始")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, st.container(border=True):
        st.markdown("**① 選資料夾**")
        st.caption("含 images/ 與 labels/（YOLO GT）的偵測資料夾。")
    with c2, st.container(border=True):
        st.markdown("**② 上傳模型預測**")
        st.caption("predictions.csv：`filename,class,cx,cy,w,h[,score]`。")
    with c3, st.container(border=True):
        st.markdown("**③（可選）共識子集**")
        st.caption("組考卷匯出的 consensus_set.csv——recall 只在共識上算。")
    mid = st.columns([2, 1.9, 2])[1]
    mid.button("✨ 用範例資料試跑（含刻意漏抓）", key="eval_demo_btn",
               type="primary", use_container_width=True, on_click=_load_eval_demo)
    st.caption(":gray[範例：拿 coco8 的 GT 當靶，預測刻意漏掉一張圖（→ 漏抓畫廊）、"
               "把一張標成灰帶（→ 排除於 recall），一眼看懂這工具在量什麼。]")


def _evaluation_ui() -> None:
    import tempfile

    from evaluation import evaluate_detections
    from interaction import crop_bbox, load_predictions_csv

    st.markdown("##### 評估 · 在共識子集上量逐型態 recall")
    st.caption("匯入模型預測 + GT，做 IoU 配對 → 逐型態 recall、漏抓(FN)畫廊、類別混淆。"
               "搭配『組考卷』的共識子集，recall 才落在穩定的尺上、可簽、可跨版本比"
               "（重定義文件 §5.3）。")
    with st.sidebar:
        st.markdown("**① 資料夾（含 images/ 與 labels/）**")
        # 評估後端只吃第一個資料夾(app 只讀 lines[0]) → 單一資料夾取代語義,避免誤導
        st.button("📁 選擇資料夾", key="eval_pick", use_container_width=True,
                  on_click=_pick_folder_into_text, args=("eval_folder_text", True))
        _picked_paths_display("eval_folder_text")
        st.markdown("**② 模型預測 CSV** `filename,class,cx,cy,w,h[,score]`")
        pred_file = st.file_uploader("predictions.csv", type="csv", key="eval_pred_file")
        st.markdown("**③（可選）組考卷共識子集 CSV**")
        cons_file = st.file_uploader("consensus_set.csv", type="csv", key="eval_cons_file")
        iou = st.slider("IoU 門檻", 0.1, 0.9, 0.5, 0.05, key="eval_iou")
        conf = st.slider("信心門檻", 0.0, 1.0, 0.0, 0.05, key="eval_conf")
        class_aware = st.checkbox("類別需相符（class-aware）", value=True, key="eval_ca")
        run = st.button("▶ 評估", type="primary", use_container_width=True, key="eval_run")

    if run:
        lines = (st.session_state.get("eval_folder_text") or "").strip().splitlines()
        folder = Path(lines[0].strip()) if lines and lines[0].strip() else None
        if not folder or not (folder / "images").exists():
            st.error("資料夾需含 images/（與 labels/）。"); return
        if pred_file is None:
            st.error("請上傳模型預測 CSV。"); return
        # 巢狀佈局也找得到 classes.txt（…/[Small]/test → indoor/classes.txt），否則
        # names=[] → GT 類別退化成 id，class-aware 比對假性全漏(recall=0)
        names = _classes_txt_nested(folder) or []
        gt_by_image = _eval_gt_by_image(folder, names)
        if not gt_by_image:
            st.error("labels/ 裡找不到任何 GT 框。"); return
        _collapse_sidebar()
        with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as tf:
            tf.write(pred_file.getvalue()); pred_path = Path(tf.name)
        pred_by_image = load_predictions_csv(pred_path)
        pred_path.unlink(missing_ok=True)
        cons_by_image = n_c = n_g = None
        if cons_file is not None:
            cons_by_image, n_c, n_g = _eval_consensus_by_image(
                cons_file.getvalue().decode("utf-8"), gt_by_image)
        res = evaluate_detections(gt_by_image, pred_by_image, iou_thresh=iou,
                                  conf_thresh=conf, class_aware=class_aware,
                                  consensus_by_image=cons_by_image)
        # class-aware 但 GT/pred 類別命名空間完全不重疊（常見：CSV 用 id、GT 用名）→
        # 比對會假性全漏 recall≈0；偵測出來給使用者明確提示而非靜默
        _gtc = {str(b["cls"]) for bs in gt_by_image.values() for b in bs}
        _pdc = {str(b["cls"]) for bs in pred_by_image.values() for b in bs}
        st.session_state["_eval_result"] = {
            "res": res, "folder": str(folder), "used_consensus": cons_by_image is not None,
            "n_cons_img": n_c, "n_gray_img": n_g, "n_pred_img": len(pred_by_image),
            "ns_mismatch": bool(class_aware and _gtc and _pdc and not (_gtc & _pdc))}

    data = st.session_state.get("_eval_result")
    if not data:
        _render_eval_quick_start()
        return
    res = data["res"]
    if data.get("is_demo"):
        st.caption(":blue[範例資料（coco8）：預測刻意漏掉一張圖 → 看『漏抓畫廊』；"
                   "一張標為灰帶 → 看它被排除於 recall。換成你的資料夾＋預測 CSV 即真評估。]")
    if data["n_pred_img"] == 0:
        st.warning("預測 CSV 沒對到任何影像（檢查 filename 欄是否為影像檔名）。")
    if data.get("ns_mismatch"):
        st.warning("⚠ 預測 CSV 的 class 與 GT 類別**命名空間完全不重疊**（常見：CSV 用類別 "
                   "id、GT 用類別名）→ class-aware 比對會假性全漏（recall≈0）。請把 CSV 的 "
                   "class 改成類別名，或關閉上方『類別需相符（class-aware）』。")
    if not data["used_consensus"]:
        st.warning("未提供組考卷共識子集——尺未校時 recall 僅供參考（§5.3）。"
                   "建議先在『組考卷』產生共識子集再評估。")
    o = res["overall"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("整體 recall", f"{o['recall'] * 100:.1f}%", help="分母＝(共識)GT，越高漏抓越少")
    c2.metric("整體 precision", f"{o['precision'] * 100:.1f}%")
    c3.metric("漏抓 FN（escape）", o["fn"])
    c4.metric("誤報 FP", o["fp"])
    if data["used_consensus"]:
        st.caption(f":gray[共識 GT {data['n_cons_img']}・灰帶 GT {res['gray']['total']}"
                   "（灰帶不計入 recall，無穩定真值）。]")

    rows = [{"型態": k, "n_GT": v["n_gt"], "recall": f"{v['recall'] * 100:.0f}%",
             "precision": f"{v['precision'] * 100:.0f}%", "TP": v["tp"],
             "FN": v["fn"], "FP": v["fp"]}
            for k, v in sorted(res["per_class"].items(), key=lambda kv: -kv[1]["fn"])]
    if rows:
        st.markdown("**逐型態（FN 多的在前）**")
        st.dataframe(rows, use_container_width=True, hide_index=True)

    fns = res["false_negatives"]
    if fns:
        st.markdown(f"**漏抓畫廊（escape，共 {len(fns)}）** — 每張是被漏掉的缺陷區")
        folder = Path(data["folder"])
        with st.container(height=520):
            cols = st.columns(4)
            for j, fn in enumerate(fns[:40]):
                with cols[j % 4]:
                    im = safe_open_image(folder / "images" / fn["filename"])  # 壞檔 None → 略過
                    if im is None:
                        st.warning(f'⚠ {fn["filename"]}')
                        continue
                    try:
                        b = fn["box"]
                        crop = crop_bbox(im, b["cx"], b["cy"],
                                         b["w"], b["h"], pad=0.4)
                        crop.thumbnail((180, 180))
                        st.image(crop, use_container_width=True,
                                 caption=f'{fn["cls"]}｜{fn["filename"]}')
                    except (OSError, ValueError):
                        st.warning(f'⚠ {fn["filename"]}')
        import csv as _csv
        import io
        buf = io.StringIO()
        w = _csv.writer(buf)
        w.writerow(["filename", "cls", "cx", "cy", "w", "h"])
        for fn in fns:
            b = fn["box"]
            w.writerow([fn["filename"], fn["cls"], b["cx"], b["cy"], b["w"], b["h"]])
        st.download_button("⬇ 匯出漏抓清單 CSV", buf.getvalue(), "escapes.csv",
                           "text/csv", key="eval_fn_csv", use_container_width=True)
        st.caption(":gray[每個漏抓 → 送『體檢卡／桶①佔比』判物理可偵測性與根因（分桶）。]")

    from collections import Counter
    conf_c = Counter((g, p) for g, p in res["confusion"] if g and p and g != p)
    if conf_c:
        st.markdown("**最常見的類別混淆（GT → 預測）**")
        for (g, p), n in conf_c.most_common(8):
            st.write(f"- {g} → {p}：{n}")


def _fewshot_build_bank_execute() -> None:
    """① 建樣本集(主體執行):用凍結模型 embed 樣本 → sample_bank.save 到 .lv_cache。"""
    import sample_bank
    from object_eval import dataset_cache_dir
    st.session_state.pop("fewshot_bank_err", None)
    model = st.session_state.get("anomaly_model") or {}
    meta = model.get("meta") or {}
    samples = list(st.session_state.get("fewshot_sample_folder") or [])
    if not (model.get("_dir") and samples):
        st.session_state["fewshot_bank_err"] = "需先有凍結模型(瑕疵偵測①)、並選樣本資料夾。"
        return
    try:
        with st.spinner("建立樣本集…首次會先載入模型(約 10~30 秒),請稍候"):
            bank = sample_bank.build_sample_bank(
                [Path(s) for s in samples], model=meta.get("model"),
                target_res=meta.get("target_res"),
                object_source=meta.get("object_source", "yolo"))
        bank_dir = dataset_cache_dir(Path(samples[0]), "fewshot_sample_bank")
        sample_bank.save_sample_bank(bank_dir, bank)
        st.session_state["fewshot_sample_bank_dir"] = str(bank_dir)
        _labels = np.asarray(bank["labels"])
        st.session_state["fewshot_sample_bank_summary"] = {
            "n": int(len(_labels)),
            "classes": {c: int(np.sum(_labels == c)) for c in sorted(set(_labels.tolist()))}}
    except Exception as exc:
        st.session_state["fewshot_bank_err"] = f"建立樣本集失敗:{exc}"


def _fewshot_scan_execute() -> None:
    """② 海掃(主體執行 → progress 串流):al_batch retrieve(多樣本比對,建議類別由樣本決定)。"""
    import al_batch
    import sample_bank
    from anomaly_bank_store import load_bank
    from object_eval import classes_for, dataset_cache_dir, list_images
    st.session_state.pop("fewshot_scan_err", None)
    _pending = st.session_state.pop("fewshot_scan_pending", None)
    if _pending is None:
        return
    _k, _min_conf = _pending
    bank_dir = st.session_state.get("fewshot_sample_bank_dir")
    model = st.session_state.get("anomaly_model") or {}
    mdir = model.get("_dir")
    targets = list(st.session_state.get("fewshot_target_folder") or [])
    if not (bank_dir and mdir and targets):
        st.session_state["fewshot_scan_err"] = "需先①建樣本集、有凍結模型、選目標資料夾。"
        return
    try:
        bank = sample_bank.load_sample_bank(bank_dir)
        meta = load_bank(mdir).get("meta", {})
        sample_bank.assert_model_compatible(bank, model=meta.get("model"),
                                            target_res=meta.get("target_res"))
        object_source = meta.get("object_source", "yolo")
        roots = [Path(t) for t in targets]
        image_paths, class_names = [], None
        for r in roots:
            image_paths.extend(list_images(r))
            class_names = class_names or classes_for(r)
        if not image_paths:
            st.session_state["fewshot_scan_err"] = "目標資料夾找不到影像。"
            return
        ck = dataset_cache_dir(roots[0], "fewshot_retrieve")
        _bar = st.progress(0.0, text=f"準備中…首次會先載入模型(約 10~30 秒),再開始海掃 {len(image_paths)} 張圖"
                                     "(之後每 20 張更新)")

        def _cb(d):
            _tot = max(int(d.get("images_total") or 1), 1)
            _proc = int(d.get("images_processed") or 0)
            _bar.progress(min(max(_proc / _tot, 0.0), 1.0),
                          text=f"海掃 {_proc}/{_tot}({_proc / _tot * 100:.0f}%)")

        with st.spinner("海掃中…首次會先載入模型(約 10~30 秒),請稍候"):
            res = al_batch.run_batched(
                image_paths, model_dir=mdir, checkpoint_dir=ck, objective="retrieve",
                ref_vectors=bank["vectors"], ref_labels=list(bank["labels"]), k=int(_k),
                object_source=object_source, class_names=class_names, dataset_dirs=roots,
                progress=_cb, resume=True, on_identity_mismatch="restart",
                min_proposal_conf=float(_min_conf), batch_size=20)
        _bar.empty()
        st.session_state["fewshot_scan_result"] = res
        st.session_state["fewshot_sample_classes"] = sorted(set(np.asarray(bank["labels"]).tolist()))
        st.session_state.pop("fewshot_decisions", None)
    except Exception as exc:
        st.session_state["fewshot_scan_err"] = f"海掃失敗:{exc}"


def _fewshot_export() -> None:
    """③ 匯出(on_click 後主體呼叫):export_retrieval → YOLO(沿用粗框幾何)+ CSV。"""
    import retrieval_export
    st.session_state.pop("fewshot_export_err", None)
    st.session_state.pop("fewshot_export_done", None)
    res = st.session_state.get("fewshot_scan_result") or {}
    records = res.get("topk_records") or []
    out_dir = (st.session_state.get("fewshot_out_dir") or "").strip()
    classes = st.session_state.get("fewshot_sample_classes") or []
    dmap = st.session_state.get("fewshot_decisions") or {}
    theta = float(st.session_state.get("fewshot_theta", 0.3))
    decs = []
    for i, r in enumerate(records):
        if float(r.get("similarity", 0)) < theta:            # 未達門檻 → 不匯(視為 pending)
            continue
        d = dmap.get(r["item_id"])
        if d is None:
            decs.append({"item": i, "decision": "accepted", "final_class": r.get("suggested_class")})
        else:
            decs.append({"item": i, "decision": d["decision"], "final_class": d.get("final_class")})
    source_dirs = [str(t) for t in (st.session_state.get("fewshot_target_folder") or [])]
    try:
        st.session_state["fewshot_export_done"] = retrieval_export.export_retrieval(
            records, decs, out_dir, class_names=classes, source_dirs=source_dirs)
    except Exception as exc:
        st.session_state["fewshot_export_err"] = f"匯出失敗:{exc}"


def _fewshot_render_queue(res, theta) -> None:
    from interaction import crop_bbox
    recs = [r for r in (res.get("topk_records") or []) if float(r.get("similarity", 0)) >= theta]
    done = bool(res.get("done"))
    st.markdown(f"{'✅ 以樣搜樣掃描完成' if done else '⏳ 掃描中'} · 命中 **{len(recs)}** 個"
                f"(相似度 ≥ {theta:.2f};已評分 {res.get('objects_scored')})")
    if not recs:
        st.info("沒有命中 —— 調低相似度門檻,或多加樣本。")
        return
    with st.container(key="fewshot_queue"):
        cols = st.columns(4)
        for j, r in enumerate(recs[:40]):
            with cols[j % 4]:
                _im = safe_open_image(r["image_path"])
                if _im is not None:
                    st.image(crop_bbox(_im, *r["bbox"], pad=0.1), use_container_width=True)
                st.caption(f"建議 **{r.get('suggested_class', '—')}** · 相似 {float(r.get('similarity', 0)):.2f}")


def _fewshot_step_samples() -> None:
    st.markdown("**① 樣本集**:丟一個小樣本資料夾(YOLO;想找的東西,約 4 類、每類 5~10 張)。")
    samples = _folder_picker_list("fewshot_sample_folder", add_help="樣本 YOLO 資料夾:含 images/ 與 labels/")
    st.button("▶ 建立樣本集", key="fewshot_build_bank_btn", type="primary", use_container_width=True,
              disabled=not samples,
              on_click=lambda: st.session_state.update(fewshot_build_pending=True))
    if st.session_state.pop("fewshot_build_pending", False):
        _fewshot_build_bank_execute()
    if st.session_state.get("fewshot_bank_err"):
        st.error(st.session_state["fewshot_bank_err"])
    _sum = st.session_state.get("fewshot_sample_bank_summary")
    with st.container(key="fewshot_bank_info"):
        if _sum:
            st.success(f"✅ 樣本集已建立:共 **{_sum['n']}** 顆樣本 · "
                       + "、".join(f"{c} {n}" for c, n in _sum["classes"].items()))
        else:
            st.caption("尚未建立樣本集。")


def _fewshot_step_scan() -> None:
    if not st.session_state.get("fewshot_sample_bank_dir"):
        st.info("請先到「① 樣本集」建立樣本集。")
        return
    st.markdown("**② 海掃**:選要撈的大資料夾(帶低信心 YOLO 粗框;無框則整張影像)。")
    targets = _folder_picker_list("fewshot_target_folder", add_help="要海掃的大資料夾")
    min_conf = st.slider("粗框信心預篩(低於此的粗框略過;0=不篩)", 0.0, 1.0, 0.0, 0.05, key="fewshot_min_conf")
    theta = st.slider("相似度門檻(佇列只留 ≥ 此的)", 0.0, 1.0, 0.3, 0.05, key="fewshot_theta")
    k = st.slider("取前幾個", 10, 500, 100, 10, key="fewshot_k")
    st.button("▶ 海掃", key="fewshot_scan_btn", type="primary", use_container_width=True,
              disabled=not targets,
              on_click=lambda: st.session_state.update(
                  fewshot_scan_pending=(int(k), float(min_conf))))
    if st.session_state.get("fewshot_scan_pending"):
        _fewshot_scan_execute()
    if st.session_state.get("fewshot_scan_err"):
        st.error(st.session_state["fewshot_scan_err"])
    res = st.session_state.get("fewshot_scan_result")
    if res:
        _fewshot_render_queue(res, float(theta))


def _fewshot_step_confirm() -> None:
    from interaction import crop_bbox
    res = st.session_state.get("fewshot_scan_result") or {}
    theta = float(st.session_state.get("fewshot_theta", 0.3))
    shown = [r for r in (res.get("topk_records") or []) if float(r.get("similarity", 0)) >= theta]
    if not shown:
        st.info("請先「② 海掃」產生命中(或調低相似度門檻)。")
        return
    st.markdown("**③ 確認 / 匯出**:預設全採納建議;可改類 / 略過。匯出 YOLO + CSV 到另選資料夾(不碰來源)。")
    classes = st.session_state.get("fewshot_sample_classes") or []
    st.session_state.setdefault("fewshot_decisions", {})
    cols = st.columns(3)
    for j, r in enumerate(shown[:30]):
        _id = r["item_id"]
        with cols[j % 3]:
            _im = safe_open_image(r["image_path"])
            if _im is not None:
                st.image(crop_bbox(_im, *r["bbox"], pad=0.1), use_container_width=True)
            _cur = st.session_state["fewshot_decisions"].get(_id)
            _tag = _cur["decision"] if _cur else "採納建議"
            st.caption(f"建議 {r.get('suggested_class', '—')} · 相似 {float(r.get('similarity', 0)):.2f} · {_tag}")
            _rc = st.selectbox("改類", ["(採納建議)"] + list(classes),
                               key=f"fewshot_relabel_{j}", label_visibility="collapsed")
            if _rc != "(採納建議)":
                st.session_state["fewshot_decisions"][_id] = {"decision": "relabeled", "final_class": _rc}
            st.button("⏭ 略過", key=f"fewshot_skip_{j}",
                      on_click=lambda _i=_id: st.session_state["fewshot_decisions"].update(
                          {_i: {"decision": "skipped", "final_class": None}}))
    _tf = st.session_state.get("fewshot_target_folder") or []
    _def_out = _anomaly_bank_default_dir(str(_tf[0]), "fewshot_out") if _tf else ""
    if not st.session_state.get("fewshot_out_dir") and _def_out:
        st.session_state["fewshot_out_dir"] = _def_out
    st.text_input("輸出資料夾(另存 YOLO+CSV;來源不動)", key="fewshot_out_dir")
    out_dir = (st.session_state.get("fewshot_out_dir") or "").strip()
    st.button("⬇ 匯出 YOLO + CSV", key="fewshot_export_btn", type="primary", use_container_width=True,
              disabled=not out_dir,
              on_click=lambda: st.session_state.update(fewshot_export_pending=True))
    if st.session_state.pop("fewshot_export_pending", False):
        _fewshot_export()
    if st.session_state.get("fewshot_export_err"):
        st.error(st.session_state["fewshot_export_err"])
    _done = st.session_state.get("fewshot_export_done")
    if _done:
        st.success(f"✅ 以樣搜樣匯出完成 · {_done['objects']} 個標註 + CSV({_done['csv_rows']} 列) → "
                   f"`{_done['out_dir']}`")


def _fewshot_monitor_init(k: int) -> None:
    """🆕 初始化以樣搜樣監看(on_click):用當前樣本集 + 目標夾 → al_service.init_workspace(retrieve)。"""
    import al_service
    st.session_state.pop("fewshot_watch_err", None)
    bank_dir = st.session_state.get("fewshot_sample_bank_dir")
    model = st.session_state.get("anomaly_model") or {}
    target = list(st.session_state.get("fewshot_target_folder") or [])
    ws = (st.session_state.get("fewshot_watch_ws") or "").strip()
    if not (bank_dir and model.get("_dir") and target and ws):
        st.session_state["fewshot_watch_err"] = "需先①建樣本集、②選目標夾、有凍結模型,並填工作區目錄。"
        return
    try:
        al_service.init_workspace(ws, name=Path(target[0]).name,
                                  watch_folders=[str(t) for t in target],
                                  model_dir=str(model["_dir"]), objective="retrieve",
                                  k=int(k), sample_bank_dir=str(bank_dir))
        st.session_state["fewshot_watch_inited"] = ws
    except Exception as exc:
        st.session_state["fewshot_watch_err"] = f"初始化監看失敗:{exc}"


def _fewshot_monitor_scan() -> None:
    """▶ 立即掃描一次(主體執行 → progress 串流):profile 缺 → 先 init;跑 al_service.run_once。"""
    import al_service
    st.session_state.pop("fewshot_watch_err", None)
    bank_dir = st.session_state.get("fewshot_sample_bank_dir")
    model = st.session_state.get("anomaly_model") or {}
    target = list(st.session_state.get("fewshot_target_folder") or [])
    ws = (st.session_state.get("fewshot_watch_ws") or "").strip()
    if not (bank_dir and model.get("_dir") and target and ws):
        st.session_state["fewshot_watch_err"] = "需先①建樣本集、②選目標夾、有凍結模型,並填工作區目錄。"
        return
    if not (Path(ws) / "profile.yaml").exists():
        _fewshot_monitor_init(int(st.session_state.get("fewshot_watch_k", 100)))
        if st.session_state.get("fewshot_watch_err"):
            return
    try:
        _bar = st.progress(0.0, text="準備中…首次會先載入模型(約 10~30 秒),再開始掃描")

        def _cb(d):
            _tot = max(int(d.get("images_total") or 1), 1)
            _bar.progress(min(max(int(d.get("images_processed") or 0) / _tot, 0.0), 1.0),
                          text=f"監看掃描 {d.get('images_processed')}/{_tot}")

        with st.spinner("監看掃描中…首次會先載入模型(約 10~30 秒),請稍候"):
            res = al_service.run_once(ws, progress=_cb)
        _bar.empty()
        st.session_state["fewshot_watch_last"] = res
        if res.get("status") == "error":
            st.session_state["fewshot_watch_err"] = f"監看掃描失敗:{res.get('reason')}"
    except Exception as exc:
        st.session_state["fewshot_watch_err"] = f"監看掃描失敗:{exc}"


def _fewshot_monitor_label(ws: str, item_id: str, decision: str, label: str) -> None:
    """佇列消費 → labels.jsonl。decision 用 al_workspace 合併語彙:'defect'=已確認(下輪移出)、'skip'=留下。"""
    import time as _t

    import al_workspace
    al_workspace.append_label(ws, {"id": item_id, "decision": decision, "label": label,
                                   "decided_at": _t.time()})
    st.toast({"defect": "已採納", "skip": "已略過"}.get(decision, "已標"), icon="✅")


def _fewshot_monitor_render_queue(ws: str, items: list) -> None:
    from interaction import crop_bbox
    st.markdown(f"**監看佇列**(共 {len(items)};✅採納 / ⏭略過,可先用下拉改類 → 下輪自動移出)")
    if not items:
        st.info("佇列空(尚無待標,或都標完了)。")
        return
    classes = st.session_state.get("fewshot_sample_classes") or []
    _limit = int(st.session_state.get("fewshot_watch_limit", 30))
    with st.container(key="fewshot_watch_queue"):
        cols = st.columns(3)
        for j, it in enumerate(items[:_limit]):
            with cols[j % 3]:
                im = safe_open_image(it.get("image_path"))
                if im is None:
                    st.caption("⚠ 缺圖")
                else:
                    st.image(crop_bbox(im, *it.get("bbox", [0.5, 0.5, 1.0, 1.0]), pad=0.1),
                             use_container_width=True)
                _sug = it.get("suggested_class", "—")
                st.caption(f"建議 **{_sug}** · 相似 {float(it.get('similarity', 0)):.2f}")
                _id = str(it.get("id", ""))
                _rc = st.selectbox("改類", ["(採納建議)"] + list(classes),
                                   key=f"fewshot_watch_relabel_{j}", label_visibility="collapsed")
                _final = _sug if _rc == "(採納建議)" else _rc
                _g, _s = st.columns(2)
                _g.button("✅ 採納", key=f"fewshot_watch_accept_{j}",
                          on_click=_fewshot_monitor_label, args=(ws, _id, "defect", _final))
                _s.button("⏭ 略過", key=f"fewshot_watch_skip_{j}",
                          on_click=_fewshot_monitor_label, args=(ws, _id, "skip", _final))
    if len(items) > _limit:
        st.button(f"載入更多(+30,共 {len(items)})", key="fewshot_watch_more",
                  use_container_width=True, on_click=lambda: st.session_state.update(
                      fewshot_watch_limit=min(_limit + 30, len(items))))


def _fewshot_step_monitor() -> None:
    """④ 監看:用當前樣本集背景自動海撈目標夾新圖(復用 al_workspace/al_service,retrieve objective)。"""
    import al_service

    import al_workspace
    if not st.session_state.get("fewshot_sample_bank_dir"):
        st.info("請先到「① 樣本集」建立樣本集。")
        return
    target = list(st.session_state.get("fewshot_target_folder") or [])
    if not target:
        st.info("請先到「② 海掃」選一個目標資料夾(監看會持續掃它、撈進新圖)。")
        return
    st.markdown("**④ 監看**:設一個工作區 → 用**當前樣本集**背景自動海撈目標夾;"
                "可**匯出設定**(profile.yaml)給離線服務排程跑。")
    _def_ws = _anomaly_bank_default_dir(str(target[0]), "fewshot_watch")
    if not st.session_state.get("fewshot_watch_ws") and _def_ws:
        st.session_state["fewshot_watch_ws"] = _def_ws
    st.text_input("工作區目錄(.lv_cache;存 profile / 佇列 / 標註)", key="fewshot_watch_ws",
                  help="設定/佇列/標註都存這;profile.yaml 可匯出給離線服務。")
    k = st.slider("每次要挑前幾個", 10, 500, 100, 10, key="fewshot_watch_k")
    ws = (st.session_state.get("fewshot_watch_ws") or "").strip()
    _c1, _c2, _c3 = st.columns(3)
    _c1.button("🆕 初始化監看", key="fewshot_watch_init_btn", use_container_width=True,
               disabled=not ws, on_click=_fewshot_monitor_init, args=(int(k),))
    _c2.button("▶ 立即掃描一次", key="fewshot_watch_scan_btn", type="primary",
               use_container_width=True, disabled=not ws,
               on_click=lambda: st.session_state.update(fewshot_watch_scan_pending=True))
    _prof = (Path(ws) / "profile.yaml") if ws else None
    with _c3:
        if _prof and _prof.exists():
            st.download_button("📤 匯出設定", key="fewshot_watch_export_btn",
                               data=_prof.read_bytes(),
                               file_name=f"al_profile_{Path(ws).name}.yaml",
                               mime="text/yaml", use_container_width=True)
        else:
            st.button("📤 匯出設定", key="fewshot_watch_export_btn", disabled=True,
                      use_container_width=True, help="先「初始化監看」才有設定可匯出。")
    if st.session_state.pop("fewshot_watch_scan_pending", False):   # 主體執行 → progress 即時串流
        _fewshot_monitor_scan()
    if st.session_state.get("fewshot_watch_err"):
        st.error(st.session_state["fewshot_watch_err"])
    _last = st.session_state.get("fewshot_watch_last")
    if _last and _last.get("status") == "ok":
        st.success(f"✅ 監看掃描完成 · 新增 {_last.get('new')} · "
                   f"評分 {_last.get('objects_scored')} · 佇列 {_last.get('queue_len')}")
    with st.container(key="fewshot_watch_status"):
        if ws and (Path(ws) / "profile.yaml").exists():
            try:
                _stat = al_service.status(ws)
            except Exception:
                _stat = {"last_run": None, "queue_len": 0}
            _lr = _stat.get("last_run")
            if _lr:
                st.caption(f"✅ 上次掃描 {_lr.get('run_id')} · 新增 {_lr.get('new')} · "
                           f"評分 {_lr.get('objects_scored')} · 佇列 {_lr.get('queue_len')}")
            else:
                st.caption("🛰 已初始化,尚未跑過掃描 —— 按「▶ 立即掃描一次」。")
        else:
            st.caption("🛰 尚未初始化監看(填工作區目錄 → 按「🆕 初始化監看」)。")
    if ws and (Path(ws) / "queue.jsonl").exists():
        _fewshot_monitor_render_queue(ws, al_workspace.read_queue(ws))


def _fewshot_search_ui() -> None:
    """🎯 以樣搜樣(第 9 工具):小樣本 → 海掃帶粗框大資料 → 建議類別 → YOLO+CSV 匯出 → 人確認。"""
    st.subheader("🎯 以樣搜樣(小樣本海撈 → YOLO 預標 → 人工確認)")
    model = st.session_state.get("anomaly_model")
    if not model or not model.get("_dir"):
        st.info("以樣搜樣需要一個**凍結模型**當特徵器。請先到『**瑕疵偵測**』① **建立或載入一個模型**"
                "(並存到暫存目錄),再回來這裡。")
        return
    step = st.segmented_control("步驟", ["① 樣本集", "② 海掃", "③ 確認 / 匯出", "④ 監看"],
                                key="fewshot_step", default="① 樣本集") or "① 樣本集"
    st.divider()
    if step == "① 樣本集":
        _fewshot_step_samples()
    elif step == "② 海掃":
        _fewshot_step_scan()
    elif step == "④ 監看":
        _fewshot_step_monitor()
    else:
        _fewshot_step_confirm()


def main() -> None:
    # sidebar 400px：layout 評審 R2 拍板（1.5x 原生支援整數寬度）
    st.set_page_config(page_title="Dataset Analysis", layout="wide",
                       initial_sidebar_state=400)
    # 壓掉 Streamlit 預設頂部留白（block-container ~6rem padding + 預設 header），
    # 把首屏高度還給工作區（嵌在 portal iframe 內時尤其明顯）。
    st.markdown(
        "<style>"
        ".block-container{padding-top:1.2rem!important;padding-bottom:1rem!important}"
        "[data-testid='stHeader']{height:0;min-height:0}"
        "[data-testid='stSidebar']>div:first-child{padding-top:1.2rem}"
        "</style>",
        unsafe_allow_html=True,
    )
    if not st.session_state.get("_usage_session_logged"):
        st.session_state["_usage_session_logged"] = True
        _log_usage("session_start")

    # 單行工具列取代舊的 st.title + sidebar Tool radio——把首屏高度還給工作區
    brand_col, switch_col, help_col = st.columns([2, 3, 1], gap="medium")
    brand_col.markdown("#### Dataset Analysis Tools")
    st.session_state.setdefault("tool_switch", "Visualize Embeddings")
    # 隱藏的工具不進工具列；session 殘留指向它們（或舊名「匯出子集」）時先正規化，
    # 否則 segmented_control 拿到非選項值會報錯。
    if st.session_state.get("tool_switch") == "匯出子集":
        st.session_state["tool_switch"] = "匯出"
    if st.session_state.get("tool_switch") in {"組考卷", "灰帶覆核", "評估"}:
        st.session_state["tool_switch"] = "Visualize Embeddings"
    with switch_col:
        st.caption("🔍 資料探索／覆蓋： Visualize · Compare · 完整度　　🔧 瑕疵偵測　　📦 匯出　　📥 標註回饋")
        tool = st.segmented_control(
            "Tool", ["Visualize Embeddings", "Compare Distributions",
                     "完整度熱力圖", "瑕疵偵測", "🎯 以樣搜樣", "匯出", "📥 標註回饋"],
            key="tool_switch", label_visibility="collapsed",
            on_change=_expand_sidebar,  # 點工具分頁 → 左側設定列自動回來
        ) or "Visualize Embeddings"
    with help_col, st.popover("✨ 功能地圖", use_container_width=True):
        st.markdown(
            "##### 🧭 六個工具的關係（先探索、再行動）\n"
            "- **🔍 資料探索／覆蓋**（看資料夠不夠、像不像；不改資料）：\n"
            "  · **Visualize**＝框選看圖、標籤分歧、離群（看**一堆內部**的點）\n"
            "  · **Compare Distributions**＝**兩堆之間**像不像（A vs B 分布距離）\n"
            "  · **完整度熱力圖**＝這堆**內部**哪裡缺／假完整（單一資料集）\n"
            "- **🏷 標註品質／評估**（量標註與模型好不好）：\n"
            "  · **組考卷**＝量標註者一致性（**只量，不改資料**）\n"
            "  · **灰帶覆核**＝對爭議做**有紀錄的裁決**（提議→雙簽→匯出；⚠ **改標只在這**）\n"
            "  · **評估**＝在共識子集量逐型態 recall（**只讀報表**）\n"
            "- **匯出清單（策展購物車）**＝跨工具收集 → 一鍵分流到組考卷／灰帶覆核／匯出\n"
            "- **怎麼串**：探索看到可疑／缺口 → 框選或「加入清單」→ 一鍵送組考卷／"
            "灰帶覆核 → 匯出。\n"
            "- **最常搞混的兩對**：『**Compare**＝比兩堆之間』vs『**熱力圖**＝看一堆內部』；"
            "『**標籤分歧**＝探索哪些點可疑』vs『**灰帶覆核**＝裁決每一點』。\n"
            "\n---\n"
            "- **框選看圖**：左圖拖曳框選／套索 → 右欄「選取」縮圖牆\n"
            "- **以文搜圖**：Model 選 *chinese-clip* → 右欄「相似」tab 輸入中文查詢\n"
            "- **以圖搜圖**：選取影像後按「🔎 找相似」，↻ 可連鎖跳查\n"
            "- **重複／洩漏掃描**：右欄「重複」tab（phash 嚴格、embedding 語意，"
            "勾「僅跨 split」＝train/val 洩漏）\n"
            "- **離群度・標籤分歧**：Run 完自動計算，右欄排序選單切換\n"
            "- **多樣性選樣／主動學習**：右欄「選樣」tab，farthest-point 挑最該優先標的 N 張\n"
            "- **體檢卡 · 三訊號根因診斷**：選一張圖 → 右欄「體檢卡」tab，"
            "用 S1 人類一致性（組考卷）× S2 覆蓋密度 × S3 模型不確定度 交叉定位 "
            "H1–H5 根因，直接回答『補資料有沒有用』，可匯出 HTML\n"
            "- **匯出清單**：跨視圖累積選取，匯出 CSV（含 sha256）／ZIP\n"
            "- **策展日誌**：選取面板底部 → 記錄『選了哪批＋為什麼』，跨重啟保存、"
            "可一鍵重選、可匯出交接（回到上週的選取）\n"
            "- **固定 UMAP 參考系**：③ 投影方法下的開關——跨 Run 佈局可比較\n"
            "- **比較兩資料夾**：Compare Distributions——逐 YOLO 類別比兩個偵測資料集的"
            "物件分布漂移（漂移表＋選一類看 A vs B 散點，可框選看物件）；需 images/＋labels/\n"
            "- **完整度熱力圖**：把資料依兩屬性軸切格，看每格『不太多不太少』、"
            "整體 Coverage Health、缺格清單（紫＝假完整近重複）。可切「嵌入覆蓋圖」"
            "模式：在原始高維空間找稀疏盲區、投影新資料夾排補洞候選、H1–H5 判斷"
            "補資料有沒有用、一鍵送進組考卷盲標\n"
            "- **組考卷**：把爭議樣本變盲測考卷，量標註者自我一致率／vs golden／"
            "多人 Fleiss kappa\n"
            "- **灰帶覆核**：爭議樣本進覆核佇列，對照錨例 → 提議+品保覆核（雙簽）"
            "→ 匯出決策（不直接寫回資料集）\n"
            "- **資料合約 manifest.jsonl**：每次 Run 自動寫入各資料夾"
            "（sha256／phash／embedding refs），供去重、回溯與下游工具使用"
        )

    # 單向交棒：送出後顯示確認並自動切到 Labeling（不在 LV 端追蹤待標／讀回）
    _render_send_confirmation()

    # 執行後左側設定列收起；點工具分頁(on_change)會展開，這顆是同分頁時的逃生口。
    # 收合 flag 是在本輪 dispatch（Run）時才設好，所以這裡只佔位、dispatch 後再填入，
    # 否則逃生鈕會慢一輪才出現（Run 當下沒鈕可按 → 收合後無法重開）。
    reopen_slot = st.empty()

    if tool == "Visualize Embeddings":
        _visualize_embeddings_ui()
    elif tool == "Compare Distributions":
        _compare_distributions_ui()
    elif tool == "完整度熱力圖":
        _completeness_ui()
    elif tool == "瑕疵偵測":
        _anomaly_ui()
    elif tool == "🎯 以樣搜樣":
        _fewshot_search_ui()
    elif tool == "組考卷":
        _quiz_ui()
    elif tool == "灰帶覆核":
        _gray_zone_ui()
    elif tool == "評估":
        _evaluation_ui()
    elif tool == "📥 標註回饋":
        _labeling_readback_ui()
    else:
        _export_subset_ui()

    # 收合 CSS 與逃生鈕都放在 dispatch 之後：本輪 Run 區塊已設好 flag，這裡即本輪生效
    if st.session_state.get("_sidebar_collapsed"):
        reopen_slot.button("☰ 顯示左側設定列", key="reopen_sidebar",
                           on_click=_expand_sidebar)
        st.markdown(_SIDEBAR_COLLAPSE_CSS, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
