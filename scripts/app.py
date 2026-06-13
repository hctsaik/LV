from __future__ import annotations

import json
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
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent))

from _utils import (
    available_models,
    extract_embeddings,
    load_model,
    load_text_encoder,
    supports_text_query,
)
from interaction import (  # noqa: F401  (parse_folder_paths re-exported for tests)
    attribute_escape,
    build_nn_index,
    compute_label_disagreement,
    compute_outlier_scores,
    draw_yolo_boxes,
    ensure_thumbnails,
    find_duplicate_pairs_embedding,
    find_duplicate_pairs_phash,
    farthest_point_sampling,
    find_similar_indices,
    find_similar_to_vector,
    load_scores_csv,
    make_thumbnail,
    neighbor_hit_density,
    neighbor_label_entropy,
    parse_folder_paths,
    records_to_csv,
    selection_points_to_indices,
    snapshots_to_csv,
    spatial_order,
    thumbnail_path_for,
    yolo_label_path_for,
    zip_selected_images,
)
from manifest import rel_key, set_embedding_refs, update_manifest, write_manifest
from umap_ref import ref_path_for, stable_umap
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
from compare_distributions import (
    build_projection_figure,
    compute_fid,
    compute_inception_score,
    compute_kid,
    compute_lpips_score,
    compute_psnr_score,
    compute_ssim_score,
    get_image_paths,
)
from visualize_embeddings import build_plotly_figure, discover_images, discover_images_classifier

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


def _pick_folder_append(list_key: str) -> None:
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    path = filedialog.askdirectory(title="選擇資料夾")
    root.destroy()
    if path:
        if list_key not in st.session_state:
            st.session_state[list_key] = []
        if path not in st.session_state[list_key]:
            st.session_state[list_key].append(path)


_VIZ_COLORS = ["#e74c3c", "#f39c12", "#2ecc71", "#9b59b6", "#3498db", "#1abc9c", "#95a5a6"]
_VIZ_SYMBOLS = {"train": "circle", "test": "square", "valid": "diamond"}
_METHOD_KEY = {"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap"}

_GRID_BATCH = 60          # cards appended per「載入更多」click
_GRID_CAP = 240           # DOM ceiling agreed in the UX review
_DEFAULT_TOP_OUTLIERS = 50

_USAGE_LOG = Path(__file__).parent.parent / "output" / "usage_log.jsonl"


def _fmt_classes(names: list[str], max_show: int = 10) -> str:
    """Class list for banners — truncated so an 80-class dataset doesn't
    blow up the zero-scroll layout budget."""
    shown = ", ".join(names[:max_show])
    if len(names) > max_show:
        shown += f", …（共 {len(names)} 個）"
    return shown


def read_classes_txt(folder: Path) -> list[str] | None:
    """Return class names from <folder-parent>/classes.txt, or None if absent/empty."""
    classes_file = folder.parent / "classes.txt"
    if not classes_file.exists():
        return None
    lines = [ln.strip() for ln in classes_file.read_text().splitlines() if ln.strip()]
    return lines if lines else None



def _build_viz_figure(
    records: list[dict],
    coords: np.ndarray,
    indices: list[int],
    model_name: str,
    method_label: str,
    dim: int = 2,
    highlight: list[int] | None = None,
) -> go.Figure:
    """Simple scatter for the selected model/method/split combination.

    Each point carries its GLOBAL record index in customdata so box/lasso
    selections map back to records regardless of trace/split filtering.
    """
    labels = sorted({records[i]["label"] for i in indices})
    splits = sorted({records[i]["split"] for i in indices})
    color_map = {lbl: _VIZ_COLORS[j % len(_VIZ_COLORS)] for j, lbl in enumerate(labels)}
    use_3d = dim == 3 and coords.shape[1] >= 3

    traces = []
    for label in labels:
        for split in splits:
            idx = [i for i in indices if records[i]["label"] == label and records[i]["split"] == split]
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
                traces.append(go.Scatter(
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
            traces.append(go.Scatter(
                x=[coords[i, 0] for i in hs], y=[coords[i, 1] for i in hs], **ring))

    fig = go.Figure(data=traces)
    # 620px：layout 評審 R2 拍板的散點高度（填滿左欄、消死白）；
    # plotly 預設邊距很肥，壓到貼齊容器
    layout = dict(title=f"{model_name} · {method_label}", height=620,
                  margin=dict(l=10, r=10, t=40, b=10),
                  legend=dict(title="Class (Split)", groupclick="toggleitem"))
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
                  margin=dict(l=10, r=10, t=20, b=10))
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
    st.session_state["viz_active_image"] = None
    st.session_state["viz_viewer_ctx"] = []
    st.session_state["viz_grid_limit"] = _GRID_BATCH


def _load_more() -> None:
    cur = st.session_state.get("viz_grid_limit", _GRID_BATCH)
    st.session_state["viz_grid_limit"] = min(cur + _GRID_BATCH, _GRID_CAP)


def _export_entry(records: list[dict], i: int) -> dict:
    r = records[i]
    p = Path(r["path"])
    man = st.session_state.get("viz_manifest", {}).get(str(p.resolve()))
    return {"index": i, "filename": p.name, "path": str(p),
            "label": r.get("label", ""), "split": r.get("split", ""),
            "sha256": man.get("sha256") if man else None}


def _add_to_export(records: list[dict], indices: list[int]) -> tuple[int, int]:
    """Add records to the export list (keyed by image path). → (added, skipped)."""
    elist = st.session_state.setdefault("viz_export_list", {})
    added = 0
    for i in indices:
        key = str(records[i]["path"])
        if key not in elist:
            elist[key] = _export_entry(records, i)
            added += 1
    return added, len(indices) - added


def _batch_add(records: list[dict], indices: list[int]) -> None:
    added, skipped = _add_to_export(records, indices)
    msg = f"已加入 {added} 張" + (f"（略過 {skipped} 張重複）" if skipped else "")
    st.toast(msg, icon="⬇")
    _log_usage("export_list_add", n=added)


def _add_one(records: list[dict], idx: int) -> None:
    _add_to_export(records, [idx])


def _remove_from_export(path_key: str) -> None:
    st.session_state.get("viz_export_list", {}).pop(path_key, None)


def _clear_export_list() -> None:
    st.session_state["viz_export_list"] = {}
    st.session_state["viz_clear_list_confirm"] = False


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
    except OSError:
        return None


# ── right-panel renderers（兩欄佈局的右欄）────────────────────────────────

def _render_viewer_slot(records: list[dict], ctx_default: list[int]) -> None:
    """Fixed-height, always-present viewer slot.

    Clicking a card only swaps the slot's content — the grid below never
    moves (zero layout shift), and the YOLO toggle keeps a fixed key so its
    state survives across images.
    """
    with st.container(height=280, border=True, key="viz_image_viewer"):
        idx = st.session_state.get("viz_active_image")
        if idx is None or not (0 <= idx < len(records)):
            st.caption("檢視槽 — 點選下方任一縮圖，在此檢視大圖與標註框，並可逐張加入匯出清單。")
            return
        r = records[idx]
        p = Path(r["path"])
        ctx = st.session_state.get("viz_viewer_ctx") or list(ctx_default) or [idx]
        pos = ctx.index(idx) if idx in ctx else 0
        h1, h2, h3, h4 = st.columns([5, 1, 1, 1])
        h1.markdown(f"**{p.name}** — {r['label']}（{r['split']}）· {pos + 1}/{len(ctx)} · #{idx}")
        h2.button("◀", key="viz_img_prev", disabled=pos <= 0,
                  on_click=_set_active_image, args=(ctx[max(pos - 1, 0)],))
        h3.button("▶", key="viz_img_next", disabled=pos >= len(ctx) - 1,
                  on_click=_set_active_image, args=(ctx[min(pos + 1, len(ctx) - 1)],))
        h4.button("✕", key="viz_img_close", on_click=_set_active_image, args=(None,))

        img_col, ctl_col = st.columns([3, 2])
        with img_col:
            if not p.exists():
                st.warning(f"找不到檔案：{p}")
            else:
                class_names = st.session_state.get("viz_class_names")
                show_boxes = bool(class_names) and st.session_state.get("viz_img_boxes", False)
                try:
                    if show_boxes:
                        st.image(draw_yolo_boxes(p, yolo_label_path_for(p), class_names),
                                 use_container_width=True)
                    else:
                        st.image(str(p), use_container_width=True)
                except OSError as exc:
                    st.warning(f"無法讀取影像：{exc}")
        with ctl_col:
            if st.session_state.get("viz_class_names"):
                st.toggle("顯示標註框", key="viz_img_boxes")
            elist = st.session_state.get("viz_export_list", {})
            if str(p) in elist:
                st.button("✓ 已在清單 — 移除", key="viz_slot_remove", use_container_width=True,
                          on_click=_remove_from_export, args=(str(p),))
            else:
                st.button("⬇ 加入匯出清單", key="viz_slot_add", use_container_width=True,
                          on_click=_add_one, args=(records, idx))
            st.button("🔎 以此找相似", key="viz_slot_similar", use_container_width=True,
                      on_click=_start_query, args=(idx,))
            man = st.session_state.get("viz_manifest", {}).get(str(p.resolve()))
            if man:
                # 資料合約可追溯性：複核時一鍵看到這張圖的 manifest 身分
                with st.popover("📄 Manifest", use_container_width=True):
                    st.caption(f"sha256：`{man.get('sha256', '—')}`")
                    st.caption(f"phash：`{man.get('phash') or '—'}`")
                    st.caption(f"大小：{man.get('size', 0):,} bytes · "
                               f"檔案時間：{man.get('captured_at', '—')}")
                    refs = man.get("embedding_refs", {})
                    if refs:
                        st.caption("embedding refs：" +
                                   "、".join(f"{m} → 列 {r}" for m, r in refs.items()))


def _render_grid(records: list[dict], shown: list[int], show_rank: bool) -> None:
    elist = st.session_state.get("viz_export_list", {})
    with st.container(height=440, key="viz_grid"):
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
                rank = f"｜第{j + 1}" if show_rank else ""
                st.button(f"{mark}#{i}{rank}", key=f"viz_card_{i}",
                          use_container_width=True,
                          on_click=_set_active_image, args=(i, list(shown)))


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

    outlier = st.session_state.get("viz_outlier_scores", {}).get(model_name)
    disagreement = st.session_state.get("viz_label_disagreement", {}).get(model_name)
    show_rank = False
    if sel_indices:
        if selected_split == "All":
            disp = list(sel_indices)
        else:
            disp = [i for i in sel_indices if records[i]["split"] == selected_split]
        if sort == "離群度" and outlier is not None:
            order = sorted(disp, key=lambda i: -float(outlier[i]))
            show_rank = True
        elif sort == "標籤分歧" and disagreement is not None:
            order = sorted(disp, key=lambda i: -float(disagreement[i]))
            show_rank = True
        elif sort == "檔名":
            order = sorted(disp, key=lambda i: records[i]["path"].name)
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
            show_rank = True
            status = f"未選取 · 預設顯示{crit}前 {len(shown)} 張（{note}）"
        else:
            order, shown = [], []
            status = "未選取"
    with st.container(key="viz_status_line"):
        st.caption(
            status,
            help="排序說明：空間順序＝縮圖位置模仿散點圖；離群度＝到鄰居的平均距離，"
                 "越高越「孤立」；標籤分歧＝k 近鄰中標籤不同的比例，越高越值得複查標註"
                 "（後兩者僅供排序參考，非品質判定）。卡片上的「第n」是目前排序的名次，"
                 "#n 是資料點編號。",
        )

    _render_viewer_slot(records, shown)
    _render_grid(records, shown, show_rank)
    if sel_indices and len(order) > len(shown):
        st.button(f"載入更多（+{_GRID_BATCH}）", key="viz_more_btn",
                  use_container_width=True, on_click=_load_more,
                  disabled=len(shown) >= _GRID_CAP)
    with st.expander("詳細表格"):
        df = pd.DataFrame([
            {"index": i, "filename": records[i]["path"].name,
             "label": records[i]["label"], "split": records[i]["split"]}
            for i in shown
        ])
        st.dataframe(df, key="viz_sel_table", hide_index=True,
                     use_container_width=True, height=220)


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
    with st.container(height=380):
        cols = st.columns(3)
        for j, (i, d) in enumerate(zip(idxs, dists)):
            with cols[j % 3]:
                p = Path(records[i]["path"])
                thumb = _thumb_or_none(p)
                if thumb is not None:
                    st.image(thumb, use_container_width=True,
                             caption=f"#{i} · d={d:.4f}")
                else:
                    st.warning(f"缺檔：{p.name}")
                st.button("↻ 以此圖續查", key=f"viz_textpivot_{i}",
                          use_container_width=True,
                          on_click=_pivot_to_image_query, args=(i,))
                st.button("⬇ 加入清單", key=f"viz_textadd_{i}",
                          use_container_width=True,
                          on_click=_add_one, args=(records, i))
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
        st.caption(f"查詢影像：{records[q]['path'].name} — cosine 距離越小越相似。")
        k = st.number_input(
            "k（回傳數量）", min_value=1, max_value=max(1, len(records) - 1),
            value=min(9, len(records) - 1), key="viz_similar_k",
        )
        idxs, dists = find_similar_indices(raw, q, k=int(k),
                                           nn_index=_nn_index_for(model_name))
        if not idxs:
            st.info("沒有其他影像可比對。")
            return
        with st.container(height=380):
            cols = st.columns(3)
            for j, (i, d) in enumerate(zip(idxs, dists)):
                with cols[j % 3]:
                    p = Path(records[i]["path"])
                    thumb = _thumb_or_none(p)
                    if thumb is not None:
                        st.image(thumb, use_container_width=True,
                                 caption=f"#{i} · d={d:.4f}")
                    else:
                        st.warning(f"缺檔：{p.name}")
                    st.button("↻ 以此為查詢", key=f"viz_requery_{i}", use_container_width=True,
                              on_click=_chain_query, args=(i,))
                    st.button("⬇ 加入清單", key=f"viz_simadd_{i}", use_container_width=True,
                              on_click=_add_one, args=(records, i))
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
                  on_click=_batch_add, args=(records, [j for _, j, _ in pairs]))
        with st.container(height=330, key="viz_dup_list"):
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
                        st.button(f"#{idx_}（{records[idx_]['split']}）",
                                  key=f"viz_dup_{row}_{side}", use_container_width=True,
                                  on_click=_set_active_image, args=(idx_, [i, j]))
                with cc[2]:
                    st.caption(f"d={dd}")
                    st.button("⬇ 右側入清單", key=f"viz_dup_addr_{row}",
                              use_container_width=True,
                              on_click=_add_one, args=(records, j))


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
                  on_click=_batch_add, args=(records, picks))
        with st.container(height=320):
            cols = st.columns(3)
            for j, i in enumerate(picks):
                with cols[j % 3]:
                    p = Path(records[i]["path"])
                    thumb = _thumb_or_none(p)
                    if thumb:
                        st.image(thumb, use_container_width=True, caption=f"#{j + 1}")
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

        density = neighbor_hit_density(raw, idx, radius)
        entropy = neighbor_label_entropy(raw, labels, idx, k=20)
        if outlier is not None:
            outlier_pct = float((outlier <= outlier[idx]).mean())
        else:
            outlier_pct = 0.5

        scores = _scores_for(p)
        sc = scores.get(p.name)
        score_v = sc[0] if sc else None
        thr_v = sc[1] if sc else None

        attr = attribute_escape(density, entropy, outlier_pct,
                                score=score_v, threshold=thr_v)

        st.subheader(f"🩺 體檢卡 · {p.name}")
        st.caption(f"{r['label']}（{r['split']}）· #{idx}")
        # 歸因
        st.markdown(f"**歸因：{attr['class']}**　信心 {attr['confidence']:.0%}")
        for reason in attr["reasons"]:
            st.caption(f"· {reason}")
        # 訊號
        c1, c2, c3 = st.columns(3)
        c1.metric("N2 命中密度", density, help=f"半徑 {radius:.3f} 內訓練集相似鄰居數。")
        c2.metric("N3 標籤分歧熵", f"{entropy:.2f}", help="鄰居標籤的歸一化熵；高＝標準漂移。")
        c3.metric("離群度百分位", f"{outlier_pct * 100:.0f}%",
                  help="此圖離群度在資料集中的位置。")
        if score_v is not None:
            st.caption(f"模型分數 {score_v:.3f}" +
                       (f" · 閾值 {thr_v:.3f}（N4 已啟用）" if thr_v is not None
                        else "（無閾值，N4 僅供參考）"))
        else:
            st.caption(":gray[未找到 scores.csv（filename,score[,threshold]）——"
                       "N4 分數閘停用，僅以 embedding 訊號歸因。]")

        # kNN 鄰居縮圖牆
        st.markdown("**最近鄰（它長得像誰）**")
        nbr_idx, nbr_d = find_similar_indices(raw, idx, k=6,
                                              nn_index=_nn_index_for(model_name))
        with st.container(height=170):
            cols = st.columns(3)
            for j, (ni, nd) in enumerate(zip(nbr_idx, nbr_d)):
                with cols[j % 3]:
                    thumb = _thumb_or_none(Path(records[ni]["path"]))
                    if thumb:
                        st.image(thumb, use_container_width=True,
                                 caption=f"{records[ni]['label']} d={nd:.3f}")
        # 匯出
        report = _health_card_report(p, r, idx, attr, density, entropy,
                                     outlier_pct, score_v, thr_v, radius,
                                     nbr_idx, nbr_d, records)
        st.download_button("⬇ 匯出體檢卡 HTML", data=report,
                           file_name=f"healthcard_{p.stem}.html",
                           mime="text/html", key="viz_card_export",
                           use_container_width=True)


def _health_card_report(p, r, idx, attr, density, entropy, outlier_pct,
                        score_v, thr_v, radius, nbr_idx, nbr_d, records) -> str:
    rows = "".join(
        f"<tr><td>#{ni}</td><td>{records[ni]['label']}</td>"
        f"<td>{records[ni]['split']}</td><td>{nd:.4f}</td></tr>"
        for ni, nd in zip(nbr_idx, nbr_d))
    score_line = (f"模型分數 {score_v:.3f}" +
                  (f"，閾值 {thr_v:.3f}（N4 啟用）" if thr_v is not None else "（無閾值）")
                  ) if score_v is not None else "未提供 scores.csv，N4 停用"
    reasons = "".join(f"<li>{x}</li>" for x in attr["reasons"])
    return f"""<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8">
<title>體檢卡 {p.name}</title><style>
body{{font-family:"Noto Sans TC",sans-serif;max-width:720px;margin:24px auto;color:#1a2433}}
h1{{font-size:20px}} .k{{color:#5a6b80}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #e3e8ef;padding:6px 10px;font-size:14px}}
.attr{{background:#eef2f7;border-radius:8px;padding:12px 16px;margin:12px 0}}</style></head><body>
<h1>🩺 Escape 體檢卡 · {p.name}</h1>
<p class="k">{r['label']}（{r['split']}）· #{idx} · {p}</p>
<div class="attr"><b>歸因：{attr['class']}</b>　信心 {attr['confidence']:.0%}
<ul>{reasons}</ul></div>
<p>N2 命中密度 <b>{density}</b>（半徑 {radius:.3f}）　·　N3 標籤分歧熵 <b>{entropy:.2f}</b>
　·　離群度百分位 <b>{outlier_pct*100:.0f}%</b></p>
<p>{score_line}</p>
<h3>最近鄰</h3><table><tr><th>#</th><th>label</th><th>split</th><th>cosine 距離</th></tr>
{rows}</table>
<p class="k">由 LV 產生。歸因僅含 N2/N3（與可選 N4）embedding 訊號；N0 品質、N1 定義仲裁需人工。</p>
</body></html>"""


def _render_export_view() -> None:
    elist = st.session_state.get("viz_export_list", {})
    st.caption(f"匯出清單 — 共 {len(elist)} 張（session 內有效；匯出後可清空，不寫回資料集）")
    if not elist:
        st.info("清單是空的。在「選取」面板批次加入，或在檢視槽逐張加入。")
        return
    snapshots = list(elist.values())
    pseudo_records = [
        {"path": Path(s["path"]), "label": s["label"], "split": s["split"]}
        for s in snapshots
    ]
    with st.container(height=400, key="viz_export_grid"):
        cols = st.columns(4)
        for j, s in enumerate(snapshots):
            with cols[j % 4]:
                thumb = _thumb_or_none(Path(s["path"]))
                if thumb is not None:
                    st.image(thumb, use_container_width=True)
                else:
                    st.warning("⚠ 檔案遺失")
                st.button(f"移除 #{s['index']}", key=f"viz_unlist_{j}",
                          use_container_width=True,
                          on_click=_remove_from_export, args=(s["path"],))
    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇ 匯出 CSV",
        data=snapshots_to_csv(snapshots),
        file_name="selection.csv", mime="text/csv",
        key="viz_export_csv", use_container_width=True,
    )
    d2.download_button(
        "⬇ 匯出 ZIP",
        data=zip_selected_images(pseudo_records, list(range(len(pseudo_records)))),
        file_name="selection_images.zip", mime="application/zip",
        key="viz_export_zip", use_container_width=True,
    )
    c1, c2 = st.columns(2)
    confirm = c1.checkbox("確認清空", key="viz_clear_list_confirm")
    c2.button("🗑 清空清單", key="viz_clear_list_btn", disabled=not confirm,
              on_click=_clear_export_list)


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
    st.session_state.setdefault("viz_panel_view", "選取")
    view = st.segmented_control(
        "面板", ["選取", "相似", "重複", "選樣", "體檢卡", "匯出清單"],
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
    else:
        _render_export_view()


_MODE_CLEAR_KEYS = (
    "viz_records", "viz_embeddings", "viz_raw_embeddings",
    "viz_data_token", "viz_nn_index", "viz_class_names",
    "viz_selection", "viz_active_image", "viz_viewer_ctx",
    "viz_query_chain", "viz_outlier_scores", "viz_grid_limit",
    "viz_export_list", "viz_panel_view", "viz_manifest",
    "viz_phashes", "viz_label_disagreement", "viz_dup_result",
)

_DEMO_DIR = Path(__file__).parent.parent / "demo" / "coco8"


def _load_demo() -> None:
    """快速開始：一鍵載入 coco8 範例並自動執行（detector 模式，零下載）。"""
    st.session_state["viz_mode"] = "Object Detector"
    st.session_state["_viz_mode_prev"] = "Object Detector"
    st.session_state["viz_folder_list"] = [
        str(_DEMO_DIR / "train"), str(_DEMO_DIR / "val")]
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
               use_container_width=True, on_click=_load_demo,
               disabled=not (_DEMO_DIR / "train").exists())


def _visualize_embeddings_ui() -> None:
    with st.sidebar:
        st.markdown("**① 資料**")
        mode = st.radio(
            "模式", ["Object Detector", "Image Classifier"],
            key="viz_mode", horizontal=True,
            captions=["YOLO 格式（images/ + labels/）", "依類別分子資料夾"],
        )
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

        if st.button("📁 新增資料夾", use_container_width=True, key="add_viz_folder"):
            _pick_folder_append("viz_folder_list")
            st.rerun()

        for i, folder in enumerate(st.session_state["viz_folder_list"]):
            c1, c2 = st.columns([5, 1])
            c1.text(Path(folder).name)
            c1.caption(folder)
            if c2.button("✕", key=f"rm_viz_{i}"):
                st.session_state["viz_folder_list"].pop(i)
                st.rerun()

        if not st.session_state["viz_folder_list"]:
            st.caption("尚未選擇任何資料夾")

        st.text_area(
            "或貼上資料夾路徑（每行一個）",
            key="viz_folder_text",
            placeholder="例：C:\\data\\coco8\\train",
            height=68,
            help="與上方清單合併。Detector 模式貼含 images/ 與 labels/ 的資料夾；Classifier 模式貼含類別子資料夾的資料夾。",
        )

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
                    value="apple,banana,orange",
                )
        else:
            class_input = ""

        st.markdown("**② 模型**")
        all_models = available_models()
        if not all_models:
            st.error("models/ 內找不到模型檔，請放入 .pth 模型後重啟。")
            return
        selected_models = st.multiselect(
            "模型", all_models, default=all_models, label_visibility="collapsed",
            help="每個模型各算一份 embedding；chinese-clip 同時解鎖「以文搜圖」。",
        )

        st.markdown("**③ 投影方法**")
        selected_method_labels = st.multiselect(
            "投影方法", list(_METHOD_KEY), default=list(_METHOD_KEY),
            key="viz_methods", label_visibility="collapsed",
            help="只勾選需要的投影可大幅縮短計算時間。",
        )
        if "UMAP" in selected_method_labels:
            uc1, uc2 = st.columns([3, 1])
            uc1.toggle(
                "固定 UMAP 參考系", key="viz_umap_ref",
                help="首跑擬合並凍結 UMAP 空間（存於 embeddings_<model>/umap_ref.pkl）；"
                     "之後新增的影像以 transform 投入同一座標系，舊點完全不動，"
                     "跨 Run 佈局可比較。注意：transform 的擺位是近似值，"
                     "資料大幅改變後請按「↻」重建參考系。",
            )
            if st.session_state.get("viz_umap_ref"):
                uc2.button("↻", key="viz_umap_rebuild_btn", use_container_width=True,
                           help="下次 Run 重新擬合並覆寫參考系",
                           on_click=lambda: st.session_state.__setitem__(
                               "_viz_umap_rebuild", True))
                if st.session_state.get("_viz_umap_rebuild"):
                    st.caption(":orange[↻ 下次 Run 將重建 UMAP 參考系]")

        st.markdown("**④ 執行**")
        n_folders = len(st.session_state.get("viz_folder_list", [])) + len(
            parse_folder_paths(st.session_state.get("viz_folder_text", "")))
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

    if st.session_state.pop("_viz_autorun", False):
        run = True
    if run:
        folders = [Path(f) for f in st.session_state.get("viz_folder_list", [])]
        for p in parse_folder_paths(st.session_state.get("viz_folder_text", "")):
            if p not in folders:
                folders.append(p)
        if not folders:
            st.error("請先選擇至少一個資料夾。")
            return
        missing_dirs = [str(p) for p in folders if not p.exists()]
        if missing_dirs:
            st.error(f"資料夾不存在：{', '.join(missing_dirs)}")
            return

        if not selected_models:
            st.error("Select at least one model.")
            return
        if not selected_method_labels:
            st.error("請至少勾選一種投影方法。")
            return
        method_pairs = [(_METHOD_KEY[lbl], lbl) for lbl in selected_method_labels]

        class_names: list[str] | None = None
        if mode == "Object Detector":
            missing = [str(f) for f in folders if not (f / "images").exists()]
            if missing:
                st.error(f"Folder(s) missing 'images/' subdirectory: {', '.join(missing)}")
                return

            # classes.txt 優先級：手動選擇 > 自動偵測 > 文字輸入
            classes_path = st.session_state.get("viz_classes_file", "")
            if classes_path and Path(classes_path).exists():
                lines = [ln.strip() for ln in Path(classes_path).read_text().splitlines() if ln.strip()]
                class_names = lines
                st.success(f"使用選定的 classes.txt（{len(class_names)} 個類別）：{_fmt_classes(class_names)}")
            else:
                detected = read_classes_txt(folders[0])
                if detected is not None:
                    class_names = detected
                    st.success(f"Auto-detected {len(class_names)} classes: {_fmt_classes(class_names)}")
                else:
                    class_names = [c.strip() for c in class_input.split(",") if c.strip()]
                    if not class_names:
                        st.error("Enter at least one class name.")
                        return
                    st.info(f"Using manually entered classes: {_fmt_classes(class_names)}")

            records = discover_images(folders, class_names)
        else:
            records = discover_images_classifier(folders)
            if records:
                detected_classes = sorted({r["label"] for r in records})
                st.success(f"自動偵測到 {len(detected_classes)} 個類別：{_fmt_classes(detected_classes)}")

        if not records:
            st.error("No images found in the specified folders.")
            return

        empty_folders = [f.name for f in folders if not any(r["split"] == f.name for r in records)]
        if empty_folders:
            st.warning(f"No images found in folder(s): {', '.join(empty_folders)}")

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
                    cache_path = folder / f"embeddings_{model_name}" / "embeddings.npz"
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
                            arr = umap.UMAP(n_components=n_comps, n_neighbors=n_neighbors,
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
        st.session_state["viz_class_names"] = class_names
        st.session_state["viz_selection"] = {
            "token": st.session_state["viz_data_token"], "indices": []
        }
        st.session_state["viz_active_image"] = None
        st.session_state["viz_viewer_ctx"] = []
        st.session_state["viz_query_chain"] = []
        st.session_state["viz_grid_limit"] = _GRID_BATCH
        st.session_state.pop("_viz_mode_snapshot", None)
        st.session_state.pop("_viz_umap_rebuild", None)
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
        st.caption(f"{len(records)} 張影像 · {len(model_names)} 模型 · "
                   f"{len(unique_splits)} 個 split")
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1.4])
        selected_model = c1.selectbox("Model", model_names, key="viz_model_select")
        method_labels = [lbl for lbl, key in _METHOD_KEY.items()
                         if key in embeddings_per_model[selected_model]]
        selected_method = c2.selectbox("Method", method_labels, key="viz_method_select")
        selected_split = c3.selectbox("Split", ["All"] + unique_splits, key="viz_split_select")
        dim = 3 if c4.radio("維度", ["2D", "3D"], horizontal=True, key="viz_dim_radio") == "3D" else 2

        method_key = _METHOD_KEY[selected_method]
        coords = embeddings_per_model[selected_model][method_key]

        if selected_split == "All":
            indices = list(range(len(records)))
        else:
            indices = [i for i, r in enumerate(records) if r["split"] == selected_split]

        # ── 選取生命週期（UX 評審 W2）──
        # 選取只掛 data token：換 model/method/split/dim 一律保留，
        # 只有重新 Run（資料變更）才清空。
        sel_state = st.session_state.get("viz_selection") or {}
        if sel_state.get("token") != data_token:
            sel_state = {"token": data_token, "indices": []}

        # scatter widget key 帶 view 資訊：舊視圖的 widget 事件不可能滲入新視圖
        scatter_key = (f"viz_scatter_{data_token[:8]}_{selected_model}"
                       f"_{method_key}_{selected_split}")

        # NOTE: the 2D interactive chart must keep a STABLE figure spec across
        # reruns — mutating it (e.g. adding a highlight trace) makes Streamlit
        # reset the chart's selection state, silently dropping the user's
        # box/lasso selection. Highlight rings are therefore 3D-only.
        active_idx = st.session_state.get("viz_active_image")
        highlight = [active_idx] if (dim == 3 and active_idx is not None) else []
        fig = _build_viz_figure(records, coords, indices, selected_model, selected_method,
                                dim=dim, highlight=highlight)

        if not sel_state["indices"] and dim == 2:
            st.caption("💡 在圖上拖曳框選或套索圈點，右欄會立即顯示對應縮圖。")
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
                st.plotly_chart(fig, use_container_width=True, key="viz_scatter_3d")
                st.caption("ℹ 3D 模式不支援框選；切回 2D 後選取仍會保留。")
        st.session_state["viz_selection"] = sel_state

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
    with st.container(height=240, border=True, key="cmp_image_viewer"):
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
    with st.container(height=420, key="cmp_grid"):
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
                st.button(f"#{i}（{cmp_groups[i]}）", key=f"cmp_card_{i}",
                          use_container_width=True,
                          on_click=_set_cmp_active, args=(i, list(shown)))


def _compare_distributions_ui() -> None:
    with st.sidebar:
        st.caption("Folder A（直接圖片資料夾）")
        col_a, col_btn_a = st.columns([4, 1])
        if col_btn_a.button("📁", key="browse_a", use_container_width=True):
            _pick_folder("cmp_folder_a")
            st.rerun()
        folder_a = col_a.text_input(
            "Folder A", key="cmp_folder_a",
            placeholder="dataset/train/images", label_visibility="collapsed",
        )

        st.caption("Folder B（直接圖片資料夾）")
        col_b, col_btn_b = st.columns([4, 1])
        if col_btn_b.button("📁", key="browse_b", use_container_width=True):
            _pick_folder("cmp_folder_b")
            st.rerun()
        folder_b = col_b.text_input(
            "Folder B", key="cmp_folder_b",
            placeholder="goal/images", label_visibility="collapsed",
        )
        all_models = available_models()
        if not all_models:
            st.error("No .pth models found in ./models/. Add a model file and restart.")
            return
        selected_model = st.selectbox("Model", all_models)
        name = st.text_input("Output name prefix", value="comparison")
        viz_only = st.toggle("僅視覺化（跳過指標計算）", value=False)
        n_pairs = st.number_input(
            "Pairwise metric samples",
            min_value=1, value=500, step=50,
            help="隨機配對數量，用於 LPIPS 和 SSIM。越大越穩定，但計算越慢。",
            disabled=viz_only,
        )
        run = st.button("▶ Run", use_container_width=True, key="run_cmp")

    if run:
        path_a = Path(folder_a.strip()) if folder_a.strip() else None
        path_b = Path(folder_b.strip()) if folder_b.strip() else None

        if not path_a or not path_b:
            st.error("Enter both Folder A and Folder B paths.")
            return
        if not path_a.exists():
            st.error(f"Folder A not found: {path_a}")
            return
        if not path_b.exists():
            st.error(f"Folder B not found: {path_b}")
            return

        paths_a = get_image_paths(path_a)
        paths_b = get_image_paths(path_b)

        if not paths_a:
            st.error(f"No images found in Folder A: {path_a}")
            return
        if not paths_b:
            st.error(f"No images found in Folder B: {path_b}")
            return

        _CMP_STEPS = 6 if viz_only else 13
        _prog = st.progress(0, text="載入模型…")
        _step = 0

        embed_fn = load_model(selected_model)
        cache_a = path_a.parent / f"embeddings_{selected_model}" / "embeddings.npz"
        cache_b = path_b.parent / f"embeddings_{selected_model}" / "embeddings.npz"

        def _cb_a(done: int, total: int) -> None:
            _prog.progress(min((0 + done / max(total, 1)) / _CMP_STEPS, 1.0),
                           text=f"Folder A 特徵擷取 {done}/{total}")

        emb_a = extract_embeddings(paths_a, embed_fn, cache_path=cache_a, progress_cb=_cb_a)
        _step += 1; _prog.progress(_step / _CMP_STEPS, text="Folder A 特徵向量完成")

        def _cb_b(done: int, total: int) -> None:
            _prog.progress(min((1 + done / max(total, 1)) / _CMP_STEPS, 1.0),
                           text=f"Folder B 特徵擷取 {done}/{total}")

        emb_b = extract_embeddings(paths_b, embed_fn, cache_path=cache_b, progress_cb=_cb_b)
        _step += 1; _prog.progress(_step / _CMP_STEPS, text="Folder B 特徵向量完成")

        from compare_distributions import compute_coverage_gaps
        coverage_gaps = compute_coverage_gaps(emb_a, emb_b)
        _step += 1; _prog.progress(_step / _CMP_STEPS, text="Coverage gap 分析完成")

        combined = np.vstack([emb_a, emb_b])
        n_emb = len(combined)

        n_comps = min(3, max(1, n_emb - 2))
        pca_2d = PCA(n_components=n_comps, random_state=42).fit_transform(combined)
        _step += 1; _prog.progress(_step / _CMP_STEPS, text="PCA 完成")

        perplexity = min(30, max(1, n_emb - 1))
        tsne_2d = TSNE(n_components=n_comps, random_state=42, perplexity=perplexity).fit_transform(combined)
        _step += 1; _prog.progress(_step / _CMP_STEPS, text="t-SNE 完成")

        n_neighbors = min(15, max(2, n_emb - 1))
        umap_2d = umap.UMAP(n_components=n_comps, n_neighbors=n_neighbors, random_state=42).fit_transform(combined)
        _step += 1; _prog.progress(_step / _CMP_STEPS, text="UMAP 完成")

        projections = {"pca": pca_2d, "tsne": tsne_2d, "umap": umap_2d}

        if not viz_only:
            fid_score = compute_fid(str(path_a), str(path_b))
            _step += 1; _prog.progress(_step / _CMP_STEPS, text="FID 完成")

            kid_score = compute_kid(str(path_a), str(path_b))
            _step += 1; _prog.progress(_step / _CMP_STEPS, text="KID 完成")

            lpips_score = compute_lpips_score(paths_a, paths_b, n_pairs=int(n_pairs))
            _step += 1; _prog.progress(_step / _CMP_STEPS, text="LPIPS 完成")

            ssim_score = compute_ssim_score(paths_a, paths_b, n_pairs=int(n_pairs))
            _step += 1; _prog.progress(_step / _CMP_STEPS, text="SSIM 完成")

            psnr_score = compute_psnr_score(paths_a, paths_b, n_pairs=int(n_pairs))
            _step += 1; _prog.progress(_step / _CMP_STEPS, text="PSNR 完成")

            is_a = compute_inception_score(str(path_a))
            _step += 1; _prog.progress(_step / _CMP_STEPS, text="IS(A) 完成")

            is_b = compute_inception_score(str(path_b))
            _step += 1; _prog.progress(_step / _CMP_STEPS, text="IS(B) 完成")
        else:
            fid_score = kid_score = lpips_score = ssim_score = None
            psnr_score = None
            is_a = is_b = None

        _prog.empty()

        st.session_state["cmp_projections"] = projections
        st.session_state["cmp_fid"] = fid_score
        st.session_state["cmp_kid"] = kid_score
        st.session_state["cmp_lpips"] = lpips_score
        st.session_state["cmp_ssim"] = ssim_score
        st.session_state["cmp_psnr"] = psnr_score
        st.session_state["cmp_is_a"] = is_a
        st.session_state["cmp_is_b"] = is_b
        st.session_state["cmp_viz_only"] = viz_only
        st.session_state["cmp_paths_a"] = paths_a
        st.session_state["cmp_paths_b"] = paths_b
        st.session_state["cmp_names"] = (path_a.name, path_b.name)
        st.session_state["cmp_name_prefix"] = name
        st.session_state["cmp_model"] = selected_model
        st.session_state["cmp_coverage_gaps"] = coverage_gaps
        st.session_state["cmp_data_token"] = uuid.uuid4().hex
        st.session_state["cmp_selection"] = None
        st.session_state["cmp_active_image"] = None
        st.session_state["cmp_viewer_ctx"] = []

    if "cmp_projections" not in st.session_state:
        st.info("在左側設定 Folder A／Folder B 與模型後，按 ▶ Run 比較兩個分布。")
        return

    projections = st.session_state["cmp_projections"]
    fid_score = st.session_state["cmp_fid"]
    kid_score = st.session_state["cmp_kid"]
    lpips_score = st.session_state["cmp_lpips"]
    ssim_score = st.session_state["cmp_ssim"]
    psnr_score = st.session_state.get("cmp_psnr")
    is_a = st.session_state.get("cmp_is_a")
    is_b = st.session_state.get("cmp_is_b")
    viz_only = st.session_state.get("cmp_viz_only", False)
    paths_a = st.session_state["cmp_paths_a"]
    paths_b = st.session_state["cmp_paths_b"]
    name_a, name_b = st.session_state["cmp_names"]
    name = st.session_state["cmp_name_prefix"]
    selected_model = st.session_state["cmp_model"]

    if not viz_only:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("FID ↓", f"{fid_score:.4f}")
        col2.metric("KID ↓", f"{kid_score:.6f}")
        col3.metric("LPIPS ↓", f"{lpips_score:.4f}")
        col4.metric("SSIM ↑", f"{ssim_score:.4f}")
        col5, col6, col7 = st.columns(3)
        col5.metric("PSNR ↑ (dB)", f"{psnr_score:.2f}" if psnr_score is not None else "—")
        col6.metric(
            f"IS ↑  ({name_a})",
            f"{is_a[0]:.2f} ± {is_a[1]:.2f}" if is_a is not None else "—",
            help="Inception Score：越高代表影像品質與多樣性越好。基於 ImageNet 分類器，數值供參考。",
        )
        col7.metric(
            f"IS ↑  ({name_b})",
            f"{is_b[0]:.2f} ± {is_b[1]:.2f}" if is_b is not None else "—",
            help="Inception Score：越高代表影像品質與多樣性越好。基於 ImageNet 分類器，數值供參考。",
        )

    cmp_paths = list(paths_a) + list(paths_b)
    cmp_groups = [name_a] * len(paths_a) + [name_b] * len(paths_b)
    cmp_token = st.session_state.get("cmp_data_token", "")

    col_cmp, col_cmpsel = st.columns([5, 3], gap="medium")
    with col_cmp:
        col_m, col_d = st.columns([3, 1])
        selected_method = col_m.selectbox("Method", list(_METHOD_KEY))
        dim = 3 if col_d.radio("維度", ["2D", "3D"], horizontal=True) == "3D" else 2
        method_key = _METHOD_KEY[selected_method]
        proj = projections[method_key]

        fig = _build_cmp_figure(paths_a, paths_b, proj, name_a, name_b, dim=dim)
        sel_state = st.session_state.get("cmp_selection") or {}
        if sel_state.get("token") != cmp_token:
            sel_state = {"token": cmp_token, "indices": []}
        if not sel_state["indices"] and dim == 2:
            st.caption("💡 在圖上拖曳框選或套索圈點，右欄會立即顯示對應影像。")
        with st.container(key="cmp_scatter_wrap"):
            if dim == 2:
                event = st.plotly_chart(
                    fig, use_container_width=True,
                    key=f"cmp_scatter_{cmp_token[:8]}_{method_key}",
                    on_select="rerun", selection_mode=("points", "box", "lasso"),
                )
                sel_points: list[dict] = []
                if event is not None:
                    sel_obj = event.get("selection") if hasattr(event, "get") else None
                    if sel_obj:
                        sel_points = list(sel_obj.get("points", []))
                new_indices = selection_points_to_indices(sel_points)
                if new_indices and new_indices != sel_state["indices"]:
                    sel_state = {"token": cmp_token, "indices": new_indices}
                    st.session_state["cmp_active_image"] = None
                    st.session_state["cmp_viewer_ctx"] = []
                    st.toast(f"已選取 {len(new_indices)} 張", icon="🎯")
            else:
                st.plotly_chart(fig, use_container_width=True, key="cmp_scatter_3d")
                st.caption("ℹ 3D 模式不支援框選；切回 2D 以使用選取。")
        st.session_state["cmp_selection"] = sel_state

        projections_2d = {k: v[:, :2] for k, v in projections.items()}
        dl_fig = build_projection_figure(
            paths_a, paths_b, projections_2d,
            name_a=name_a, name_b=name_b,
            fid_score=fid_score, lpips_score=lpips_score,
            kid_score=kid_score, ssim_score=ssim_score,
        )

        if viz_only:
            st.download_button(
                "⬇ Download HTML (all views)",
                data=dl_fig.to_html(include_plotlyjs="cdn"),
                file_name=f"{name}_projection.html",
                mime="text/html",
            )
        else:
            metrics = {
                "fid": round(fid_score, 4),
                "kid": round(kid_score, 6),
                "lpips": round(lpips_score, 4),
                "ssim": round(ssim_score, 4),
                "psnr": round(psnr_score, 2) if psnr_score is not None else None,
                "is_a_mean": round(is_a[0], 4) if is_a is not None else None,
                "is_a_std":  round(is_a[1], 4) if is_a is not None else None,
                "is_b_mean": round(is_b[0], 4) if is_b is not None else None,
                "is_b_std":  round(is_b[1], 4) if is_b is not None else None,
                "n_a": len(paths_a),
                "n_b": len(paths_b),
                "model": selected_model,
            }
            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "⬇ Download HTML (all views)",
                data=dl_fig.to_html(include_plotlyjs="cdn"),
                file_name=f"{name}_projection.html",
                mime="text/html",
            )
            dl2.download_button(
                "⬇ Download JSON",
                data=json.dumps(metrics, indent=2),
                file_name=f"{name}_metrics.json",
                mime="application/json",
            )

    with col_cmpsel:
        _render_cmp_panel(cmp_paths, cmp_groups)

    # ── Coverage Gap Analysis ─────────────────────────────────────────
    if "cmp_coverage_gaps" in st.session_state:
        st.divider()
        st.subheader("Coverage Gap Analysis")
        st.caption(
            "每個樣本在嵌入空間中到兩群的最近鄰距離（cosine）。"
            "右上角（兩距離皆大）= 兩群皆未覆蓋 → 漏抓風險；"
            "左下角（兩距離皆小）= 兩群邊界重疊 → 誤報風險。"
        )

        d_a_to_a, d_a_to_b, d_b_to_a, d_b_to_b = st.session_state["cmp_coverage_gaps"]

        all_d_a = np.concatenate([d_a_to_a, d_b_to_a])
        all_d_b = np.concatenate([d_a_to_b, d_b_to_b])
        thr_a = float(np.percentile(all_d_a, 50))
        thr_b = float(np.percentile(all_d_b, 50))

        all_x = np.concatenate([d_a_to_a, d_b_to_a])
        all_y = np.concatenate([d_a_to_b, d_b_to_b])
        n_total = len(all_x)
        n_blind   = int(np.sum((all_x >= thr_a) & (all_y >= thr_b)))
        n_overlap = int(np.sum((all_x <  thr_a) & (all_y <  thr_b)))

        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("樣本總數", n_total)
        cm2.metric("盲點 / 異常（漏抓風險）", f"{n_blind}  ({100*n_blind/n_total:.1f}%)")
        cm3.metric("邊界重疊（誤報風險）",     f"{n_overlap} ({100*n_overlap/n_total:.1f}%)")

        from compare_distributions import build_coverage_figure
        cov_fig = build_coverage_figure(
            d_a_to_a, d_a_to_b, d_b_to_a, d_b_to_b,
            paths_a, paths_b, name_a, name_b,
        )
        st.plotly_chart(cov_fig, use_container_width=True)
        st.download_button(
            "⬇ Download Coverage HTML",
            data=cov_fig.to_html(include_plotlyjs="cdn"),
            file_name=f"{name}_coverage_gap.html",
            mime="text/html",
        )


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
        cache = pool_folders[0] / f"embeddings_{model}" / "embeddings.npz"
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
    with st.container(height=240):
        cols = st.columns(3)
        for j, it in enumerate(items):
            with cols[j % 3]:
                thumb = _thumb_or_none(Path(it["path"]))
                if thumb:
                    st.image(thumb, use_container_width=True,
                             caption=f"d={it['d']:.3f}")
                else:
                    st.warning("⚠ 缺檔")
    csv = "path,distance\n" + "\n".join(
        f'"{it["path"]}",{it["d"]:.6f}' for it in items)
    st.download_button("⬇ 匯出候選清單 CSV", data=csv,
                       file_name="cell_candidates.csv", mime="text/csv",
                       key="cov_cand_csv", use_container_width=True)


_COV_DEMO_DIR = Path(__file__).parent.parent / "demo" / "imagenette" / "train"
_D_STAR_PRESET = {"寬鬆": 0.45, "標準": 0.6, "嚴格": 0.75}


def _load_cov_demo() -> None:
    st.session_state["cov_folder_text"] = str(_COV_DEMO_DIR)
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
               on_click=_load_cov_demo,
               disabled=not _COV_DEMO_DIR.exists())


def _completeness_ui() -> None:
    st.markdown("##### 模型收值完整性熱力圖")
    st.caption("把資料切成小棋盤格，看每格「不太多也不太少」。"
               "🟩 健康、🟪 假完整（量夠但都是近重複）、🟥/🟧 缺。")

    with st.sidebar:
        st.markdown("**① 資料夾**")
        st.text_area("含類別子資料夾的影像資料夾（每行一個）", key="cov_folder_text",
                     placeholder="例：demo/imagenette/train", height=68,
                     label_visibility="collapsed",
                     help="結構需為 資料夾／類別／影像。或按主畫面的「✨ 用範例資料試跑」。")
        all_models = available_models()
        if not all_models:
            st.error("models/ 內找不到模型檔。")
            return
        st.markdown("**② 模型**")
        model = st.selectbox("模型", all_models, label_visibility="collapsed",
                             help="算每格內 embedding 多樣性（質量探針）用。")
        run = st.button("▶ 開始分析", use_container_width=True, key="run_cov",
                        type="primary")

    if st.session_state.pop("_cov_autorun", False):
        run = True
    if run:
        folders = parse_folder_paths(st.session_state.get("cov_folder_text", ""))
        missing = [str(p) for p in folders if not p.exists()]
        if not folders:
            st.error("請先輸入至少一個資料夾。"); return
        if missing:
            st.error(f"資料夾不存在：{', '.join(missing)}"); return
        records = discover_images_classifier(folders)
        if not records:
            st.error("找不到影像（需 資料夾／類別／影像 結構）。"); return

        embed_fn = load_model(model)
        with st.status("計算中…", expanded=True) as _status:
            bar = st.progress(0.0, text="特徵擷取…")
            paths = [r["path"] for r in records]
            cache = folders[0] / f"embeddings_{model}" / "embeddings.npz"

            def _cb(done, total):
                bar.progress(min(done / max(total, 1) * 0.6, 0.6),
                             text=f"特徵擷取 {done}/{total}")
            emb = extract_embeddings(paths, embed_fn, cache_path=cache, progress_cb=_cb)

            stats = []
            for i, p in enumerate(paths):
                try:
                    stats.append(image_stats(p))
                except OSError:
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
            with st.container(height=240):
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
                st.text_area("候選池資料夾（每行一個，通常是未標註的影像）",
                             key="cov_pool_text", height=58,
                             placeholder="例：未標註的產線影像資料夾")
                if st.button("開始撈候選", key="cov_mine_btn",
                             use_container_width=True,
                             disabled=not st.session_state.get("cov_pool_text", "").strip()):
                    _mine_cell_candidates(cell, records, emb, model)
            _render_cov_candidates(cell, records)
            st.button("✕ 關閉", key="cov_cell_close",
                      on_click=lambda: st.session_state.pop("cov_active_cell", None))


_QUIZ_DEMO_DIR = Path(__file__).parent.parent / "demo" / "imagenette" / "train"


def _load_quiz_demo() -> None:
    st.session_state["quiz_folder_text"] = str(_QUIZ_DEMO_DIR)
    st.session_state["_quiz_autorun"] = True
    _log_usage("quiz_demo_load")


def _quiz_answer(qid: int, label: str) -> None:
    st.session_state.setdefault("quiz_answers", {})[qid] = label
    st.session_state["quiz_pos"] = st.session_state.get("quiz_pos", 0) + 1


def _quiz_reset() -> None:
    for k in ("quiz_spec", "quiz_answers", "quiz_pos"):
        st.session_state.pop(k, None)


def _quiz_generate(records: list[dict], dis, n_q: int) -> None:
    st.session_state["quiz_spec"] = build_quiz(records, dis, n_questions=int(n_q))
    st.session_state["quiz_answers"] = {}
    st.session_state["quiz_pos"] = 0


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
               type="primary", use_container_width=True, on_click=_load_quiz_demo,
               disabled=not _QUIZ_DEMO_DIR.exists())


def _quiz_ui() -> None:
    st.markdown("##### 組考卷 · 標註者一致性盲測")
    st.caption("把爭議樣本變成盲測考卷，量「同一人會不會自打嘴巴」與「跨人是否一致」。"
               "只量對既有案例的判定穩定性，量不到庫外新型或 golden 本身對錯。")

    with st.sidebar:
        st.markdown("**① 資料夾**")
        st.text_area("含類別子資料夾的影像資料夾（每行一個）", key="quiz_folder_text",
                     placeholder="例：demo/imagenette/train", height=68,
                     label_visibility="collapsed")
        all_models = available_models()
        if not all_models:
            st.error("models/ 內找不到模型檔。"); return
        st.markdown("**② 模型**")
        model = st.selectbox("模型", all_models, label_visibility="collapsed",
                             help="用來找爭議樣本（kNN 標籤分歧）與對照題（以圖搜圖）。")
        run = st.button("▶ 載入資料", use_container_width=True, key="run_quiz",
                        type="primary")

    if st.session_state.pop("_quiz_autorun", False):
        run = True
    if run:
        folders = parse_folder_paths(st.session_state.get("quiz_folder_text", ""))
        missing = [str(p) for p in folders if not p.exists()]
        if not folders:
            st.error("請先輸入資料夾。"); return
        if missing:
            st.error(f"資料夾不存在：{', '.join(missing)}"); return
        records = discover_images_classifier(folders)
        if not records or len({r["label"] for r in records}) < 2:
            st.error("需至少 2 個類別、folder/類別/影像 結構。"); return
        embed_fn = load_model(model)
        with st.status("計算中…", expanded=True):
            paths = [r["path"] for r in records]
            cache = folders[0] / f"embeddings_{model}" / "embeddings.npz"
            emb = extract_embeddings(paths, embed_fn, cache_path=cache)
            k = min(10, len(records) - 1)
            dis = compute_label_disagreement(emb, [r["label"] for r in records], k=k)
        st.session_state["quiz_records"] = records
        st.session_state["quiz_disagreement"] = dis
        _quiz_reset()
        st.toast(f"已載入 {len(records)} 張影像", icon="✅")

    if "quiz_records" not in st.session_state:
        _render_quiz_quick_start()
        return

    records = st.session_state["quiz_records"]
    dis = st.session_state["quiz_disagreement"]
    class_opts = sorted({r["label"] for r in records})

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

    # ── 作答中 ──
    if pos < len(questions):
        q = questions[pos]
        st.progress((pos) / len(questions), text=f"第 {pos + 1} / {len(questions)} 題")
        col_img, col_ans = st.columns([3, 2], gap="medium")
        with col_img:
            p = Path(records[q["record_idx"]]["path"])
            try:
                img = Image.open(p).convert("RGB")
                if q["skin"]:
                    img = geometric_skin(img, q["skin"])
                st.image(img, use_container_width=True)
            except OSError:
                st.warning(f"無法讀取：{p}")
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
    csv = "qid,answer\n" + "\n".join(f"{q},{a}" for q, a in sorted(answers.items()))
    st.download_button("⬇ 匯出作答 CSV（給多人一致性用）", data=csv,
                       file_name="quiz_answers.csv", mime="text/csv",
                       key="quiz_answers_csv")
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
        elif files:
            st.info("至少需要 2 份作答 CSV。")


def main() -> None:
    # sidebar 400px：layout 評審 R2 拍板（1.5x 原生支援整數寬度）
    st.set_page_config(page_title="Dataset Analysis", layout="wide",
                       initial_sidebar_state=400)
    if not st.session_state.get("_usage_session_logged"):
        st.session_state["_usage_session_logged"] = True
        _log_usage("session_start")

    # 單行工具列取代舊的 st.title + sidebar Tool radio——把首屏高度還給工作區
    brand_col, switch_col, help_col = st.columns([2, 3, 1], gap="medium")
    brand_col.markdown("#### Dataset Analysis Tools")
    st.session_state.setdefault("tool_switch", "Visualize Embeddings")
    with switch_col:
        tool = st.segmented_control(
            "Tool", ["Visualize Embeddings", "Compare Distributions",
                     "完整度熱力圖", "組考卷"],
            key="tool_switch", label_visibility="collapsed",
        ) or "Visualize Embeddings"
    with help_col, st.popover("✨ 功能地圖", use_container_width=True):
        st.markdown(
            "- **框選看圖**：左圖拖曳框選／套索 → 右欄「選取」縮圖牆\n"
            "- **以文搜圖**：Model 選 *chinese-clip* → 右欄「相似」tab 輸入中文查詢\n"
            "- **以圖搜圖**：選取影像後按「🔎 找相似」，↻ 可連鎖跳查\n"
            "- **重複／洩漏掃描**：右欄「重複」tab（phash 嚴格、embedding 語意，"
            "勾「僅跨 split」＝train/val 洩漏）\n"
            "- **離群度・標籤分歧**：Run 完自動計算，右欄排序選單切換\n"
            "- **多樣性選樣／主動學習**：右欄「選樣」tab，farthest-point 挑最該優先標的 N 張\n"
            "- **體檢卡**：選一張圖 → 右欄「體檢卡」tab，看 N2 命中密度/N3 標籤熵/"
            "歸因（放 scores.csv 可加 N4 分數閘），可匯出 HTML\n"
            "- **匯出清單**：跨視圖累積選取，匯出 CSV（含 sha256）／ZIP\n"
            "- **固定 UMAP 參考系**：③ 投影方法下的開關——跨 Run 佈局可比較\n"
            "- **比較兩資料夾**：Compare Distributions——FID/KID 等指標＋"
            "點選散點看對應影像\n"
            "- **完整度熱力圖**：把資料依兩屬性軸切格，看每格『不太多不太少』、"
            "整體 Coverage Health、缺格清單（紫＝假完整近重複）\n"
            "- **組考卷**：把爭議樣本變盲測考卷，量標註者自我一致率／vs golden／"
            "多人 Fleiss kappa\n"
            "- **資料合約 manifest.jsonl**：每次 Run 自動寫入各資料夾"
            "（sha256／phash／embedding refs），供去重、回溯與下游工具使用"
        )

    if tool == "Visualize Embeddings":
        _visualize_embeddings_ui()
    elif tool == "Compare Distributions":
        _compare_distributions_ui()
    elif tool == "完整度熱力圖":
        _completeness_ui()
    else:
        _quiz_ui()


if __name__ == "__main__":
    main()
