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
    build_nn_index,
    compute_label_disagreement,
    compute_outlier_scores,
    draw_yolo_boxes,
    ensure_thumbnails,
    find_duplicate_pairs_embedding,
    find_duplicate_pairs_phash,
    find_similar_indices,
    find_similar_to_vector,
    make_thumbnail,
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
    layout = dict(title=f"{model_name} · {method_label}",
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

    def _trace(data, names, color, label):
        common = dict(
            mode="markers", name=label,
            marker=dict(color=color, size=4 if use_3d else 6, opacity=0.7),
            text=names,
            hovertemplate="%{text}<br>Group: " + label + "<extra></extra>",
        )
        if use_3d:
            return go.Scatter3d(x=data[:, 0].tolist(), y=data[:, 1].tolist(), z=data[:, 2].tolist(), **common)
        return go.Scatter(x=data[:, 0].tolist(), y=data[:, 1].tolist(), **common)

    fig = go.Figure(data=[
        _trace(proj[:n_a], [p.name for p in paths_a], "#3498db", name_a),
        _trace(proj[n_a:], [p.name for p in paths_b], "#e74c3c", name_b),
    ])
    layout = dict(legend=dict(title="Group"))
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
        st.caption(status)

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
        c1, c2, c3, c4 = st.columns([1.7, 1.3, 1.1, 1])
        method = c1.selectbox("方法", ["phash（嚴格重複）", "embedding（語意重複）"],
                              key="viz_dup_method", label_visibility="collapsed")
        c2.toggle("僅跨 split", key="viz_dup_cross", help="只列出跨資料夾的重複＝洩漏候選")
        if method.startswith("phash"):
            c3.number_input("漢明 ≤", min_value=0, max_value=16, value=4,
                            key="viz_dup_thr_ph")
        else:
            c3.number_input("cosine ≤", min_value=0.0, max_value=0.5, value=0.05,
                            step=0.01, format="%.2f", key="viz_dup_thr_emb")
        c4.button("🔍 掃描", key="viz_dup_scan", use_container_width=True,
                  on_click=_scan_duplicates, args=(records, model_name))

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
        "面板", ["選取", "相似", "重複", "匯出清單"], key="viz_panel_view",
        label_visibility="collapsed",
    ) or "選取"

    if view == "選取":
        _render_select_view(records, coords, model_name, selected_split, scatter_key)
    elif view == "相似":
        _render_similar_view(records, model_name)
    elif view == "重複":
        _render_dup_view(records, model_name)
    else:
        _render_export_view()


def _visualize_embeddings_ui() -> None:
    st.header("Visualize Embeddings")

    with st.sidebar:
        mode = st.radio(
            "模式", ["Object Detector", "Image Classifier"],
            key="viz_mode", horizontal=True,
        )
        # 切換模式時清除舊結果與資料夾列表
        if st.session_state.get("_viz_mode_prev") != mode:
            st.session_state["_viz_mode_prev"] = mode
            for k in ("viz_records", "viz_embeddings", "viz_raw_embeddings",
                      "viz_data_token", "viz_nn_index", "viz_class_names",
                      "viz_selection", "viz_active_image", "viz_viewer_ctx",
                      "viz_query_chain", "viz_outlier_scores", "viz_grid_limit",
                      "viz_export_list", "viz_panel_view", "viz_manifest",
                      "viz_phashes", "viz_label_disagreement", "viz_dup_result"):
                st.session_state.pop(k, None)
            st.session_state["viz_folder_list"] = []

        st.divider()

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
            placeholder="demo/imagenette/train",
            height=68,
        )
        st.caption("與上方清單合併；適用於無檔案對話框的環境。")

        if mode == "Object Detector":
            st.divider()
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

        all_models = available_models()
        if not all_models:
            st.error("No .pth models found in ./models/. Add a model file and restart.")
            return
        selected_models = st.multiselect("Models", all_models, default=all_models)
        selected_method_labels = st.multiselect(
            "投影方法", list(_METHOD_KEY), default=list(_METHOD_KEY),
            key="viz_methods", help="只勾選需要的投影可大幅縮短計算時間。",
        )
        run = st.button("▶ Run", use_container_width=True, key="run_viz")

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
                proj: dict[str, np.ndarray] = {}
                for mkey, mlabel in method_pairs:
                    if mkey == "pca":
                        arr = PCA(n_components=n_comps, random_state=42).fit_transform(embeddings)
                    elif mkey == "tsne":
                        perplexity = min(30, max(1, n_samples - 1))
                        arr = TSNE(n_components=n_comps, random_state=42,
                                   perplexity=perplexity).fit_transform(embeddings)
                    else:
                        n_neighbors = min(15, max(2, n_samples - 1))
                        arr = umap.UMAP(n_components=n_comps, n_neighbors=n_neighbors,
                                        random_state=42).fit_transform(embeddings)
                    proj[mkey] = arr
                    _step += 1
                    _bar.progress(_step / _n_steps, text=f"[{model_name}] {mlabel} 完成")

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
        # 匯出清單以 image path 為鍵，跨 Run 仍有效 — 刻意不清
        st.toast(f"完成：{len(records)} 張影像 × {len(selected_models)} 模型", icon="✅")

    if "viz_records" not in st.session_state:
        st.info("Configure inputs in the sidebar and click ▶ Run.")
        return

    records = st.session_state["viz_records"]
    embeddings_per_model = st.session_state["viz_embeddings"]
    data_token = st.session_state.get("viz_data_token", "")

    model_names = list(embeddings_per_model.keys())
    unique_splits = sorted({r["split"] for r in records})

    col_plot, col_panel = st.columns([3, 2], gap="medium")

    with col_plot:
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


def _compare_distributions_ui() -> None:
    st.header("Compare Distributions")

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

    if "cmp_projections" not in st.session_state:
        st.info("Configure inputs in the sidebar and click ▶ Run.")
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

    col_m, col_d = st.columns([3, 1])
    selected_method = col_m.selectbox("Method", list(_METHOD_KEY))
    dim = 3 if col_d.radio("維度", ["2D", "3D"], horizontal=True) == "3D" else 2
    method_key = _METHOD_KEY[selected_method]
    proj = projections[method_key]

    fig = _build_cmp_figure(paths_a, paths_b, proj, name_a, name_b, dim=dim)
    st.plotly_chart(fig, use_container_width=True)

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


def main() -> None:
    st.set_page_config(page_title="Dataset Analysis", layout="wide")
    st.title("Dataset Analysis Tools")
    if not st.session_state.get("_usage_session_logged"):
        st.session_state["_usage_session_logged"] = True
        _log_usage("session_start")

    tool = st.sidebar.radio(
        "Tool",
        ["Visualize Embeddings", "Compare Distributions"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    if tool == "Visualize Embeddings":
        _visualize_embeddings_ui()
    else:
        _compare_distributions_ui()


if __name__ == "__main__":
    main()
