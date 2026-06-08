from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent))

from _utils import available_models, extract_embeddings, load_model
from compare_distributions import (
    build_projection_figure,
    compute_fid,
    compute_kid,
    compute_lpips_score,
    compute_ssim_score,
    get_image_paths,
)
from visualize_embeddings import build_plotly_figure, discover_images

def _pick_folder(session_key: str) -> None:
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    path = filedialog.askdirectory(title="選擇資料夾")
    root.destroy()
    if path:
        st.session_state[session_key] = path


_VIZ_COLORS = ["#e74c3c", "#f39c12", "#2ecc71", "#9b59b6", "#3498db", "#1abc9c", "#95a5a6"]
_VIZ_SYMBOLS = {"train": "circle", "test": "square", "valid": "diamond"}
_METHOD_KEY = {"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap"}


def read_classes_txt(folder: Path) -> list[str] | None:
    """Return class names from <folder-parent>/classes.txt, or None if absent/empty."""
    classes_file = folder.parent / "classes.txt"
    if not classes_file.exists():
        return None
    lines = [ln.strip() for ln in classes_file.read_text().splitlines() if ln.strip()]
    return lines if lines else None


def parse_folder_paths(text: str) -> list[Path]:
    """Return a Path for each non-blank line in text."""
    paths = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            paths.append(Path(line))
    return paths


def _build_viz_figure(
    records: list[dict],
    coords: np.ndarray,
    indices: list[int],
    model_name: str,
    method_label: str,
) -> go.Figure:
    """Simple scatter for the selected model/method/split combination."""
    labels = sorted({records[i]["label"] for i in indices})
    splits = sorted({records[i]["split"] for i in indices})
    color_map = {lbl: _VIZ_COLORS[j % len(_VIZ_COLORS)] for j, lbl in enumerate(labels)}

    traces: list[go.Scatter] = []
    for label in labels:
        for split in splits:
            idx = [i for i in indices if records[i]["label"] == label and records[i]["split"] == split]
            if not idx:
                continue
            traces.append(go.Scatter(
                x=[coords[i, 0] for i in idx],
                y=[coords[i, 1] for i in idx],
                mode="markers",
                name=f"{label} ({split})",
                legendgroup=label,
                marker=dict(
                    color=color_map[label],
                    symbol=_VIZ_SYMBOLS.get(split, "circle"),
                    size=7, opacity=0.8,
                ),
                text=[records[i]["path"].name for i in idx],
                hovertemplate="%{text}<br>Label: " + label + "<br>Split: " + split + "<extra></extra>",
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{model_name} · {method_label}",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        legend=dict(title="Class (Split)", groupclick="toggleitem"),
    )
    return fig


def _build_cmp_figure(
    paths_a: list[Path],
    paths_b: list[Path],
    proj_2d: np.ndarray,
    name_a: str,
    name_b: str,
) -> go.Figure:
    """Simple scatter showing two groups in the selected projection."""
    n_a = len(paths_a)
    fig = go.Figure(data=[
        go.Scatter(
            x=proj_2d[:n_a, 0].tolist(), y=proj_2d[:n_a, 1].tolist(),
            mode="markers", name=name_a,
            marker=dict(color="#3498db", size=6, opacity=0.7),
            text=[p.name for p in paths_a],
            hovertemplate="%{text}<br>Group: " + name_a + "<extra></extra>",
        ),
        go.Scatter(
            x=proj_2d[n_a:, 0].tolist(), y=proj_2d[n_a:, 1].tolist(),
            mode="markers", name=name_b,
            marker=dict(color="#e74c3c", size=6, opacity=0.7),
            text=[p.name for p in paths_b],
            hovertemplate="%{text}<br>Group: " + name_b + "<extra></extra>",
        ),
    ])
    fig.update_layout(
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        legend=dict(title="Group"),
    )
    return fig


def _visualize_embeddings_ui() -> None:
    st.header("Visualize Embeddings")

    with st.sidebar:
        folders_text = st.text_area(
            "Folders (one per line)",
            placeholder="dataset/train\ndataset/test\ndataset/valid",
        )
        all_models = available_models()
        if not all_models:
            st.error("No .pth models found in ./models/. Add a model file and restart.")
            return
        selected_models = st.multiselect("Models", all_models, default=all_models)
        class_input = st.text_input(
            "Class names — fallback if classes.txt not found",
            value="apple,banana,orange",
        )
        run = st.button("▶ Run", use_container_width=True, key="run_viz")

    if run:
        folders = parse_folder_paths(folders_text)
        if not folders:
            st.error("Enter at least one folder path.")
            return

        missing = [str(f) for f in folders if not (f / "images").exists()]
        if missing:
            st.error(f"Folder(s) missing 'images/' subdirectory: {', '.join(missing)}")
            return

        if not selected_models:
            st.error("Select at least one model.")
            return

        detected = read_classes_txt(folders[0])
        if detected is not None:
            class_names = detected
            st.success(
                f"Auto-detected {len(class_names)} classes from classes.txt: "
                + ", ".join(class_names)
            )
        else:
            class_names = [c.strip() for c in class_input.split(",") if c.strip()]
            if not class_names:
                st.error("Enter at least one class name.")
                return
            st.info(f"Using manually entered classes: {', '.join(class_names)}")

        _n_steps = len(selected_models) * 4
        _prog = st.progress(0, text="探索圖片中…")
        _step = 0

        records = discover_images(folders, class_names)
        if not records:
            st.error("No images found in the specified folders.")
            _prog.empty()
            return

        empty_folders = [f.name for f in folders if not any(r["split"] == f.name for r in records)]
        if empty_folders:
            st.warning(f"No images found in folder(s): {', '.join(empty_folders)}")

        embeddings_per_model: dict[str, dict[str, np.ndarray]] = {}
        for model_name in selected_models:
            embed_fn = load_model(model_name)
            all_embs = []
            for folder in folders:
                folder_records = [r for r in records if r["split"] == folder.name]
                folder_paths = [r["path"] for r in folder_records]
                cache_path = folder / f"embeddings_{model_name}" / "embeddings.npz"
                if folder_paths:
                    all_embs.append(
                        extract_embeddings(folder_paths, embed_fn, cache_path=cache_path)
                    )
            embeddings = np.vstack(all_embs)
            _step += 1
            _prog.progress(_step / _n_steps, text=f"[{model_name}] 特徵向量提取完成")

            pca_2d = PCA(n_components=2, random_state=42).fit_transform(embeddings)
            _step += 1
            _prog.progress(_step / _n_steps, text=f"[{model_name}] PCA 完成")

            perplexity = min(30, max(5, len(records) - 1))
            tsne_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(embeddings)
            _step += 1
            _prog.progress(_step / _n_steps, text=f"[{model_name}] t-SNE 完成")

            umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
            _step += 1
            _prog.progress(_step / _n_steps, text=f"[{model_name}] UMAP 完成")

            embeddings_per_model[model_name] = {"pca": pca_2d, "tsne": tsne_2d, "umap": umap_2d}

        _prog.empty()

        st.session_state["viz_records"] = records
        st.session_state["viz_embeddings"] = embeddings_per_model

    if "viz_records" not in st.session_state:
        st.info("Configure inputs in the sidebar and click ▶ Run.")
        return

    records = st.session_state["viz_records"]
    embeddings_per_model = st.session_state["viz_embeddings"]

    model_names = list(embeddings_per_model.keys())
    unique_splits = sorted({r["split"] for r in records})

    col1, col2, col3 = st.columns(3)
    selected_model = col1.selectbox("Model", model_names)
    selected_method = col2.selectbox("Method", list(_METHOD_KEY))
    selected_split = col3.selectbox("Split", ["All"] + unique_splits)

    method_key = _METHOD_KEY[selected_method]
    coords = embeddings_per_model[selected_model][method_key]

    if selected_split == "All":
        indices = list(range(len(records)))
    else:
        indices = [i for i, r in enumerate(records) if r["split"] == selected_split]

    fig = _build_viz_figure(records, coords, indices, selected_model, selected_method)
    st.plotly_chart(fig, use_container_width=True)

    dl_fig = build_plotly_figure(records, embeddings_per_model)
    st.download_button(
        "⬇ Download HTML (all views)",
        data=dl_fig.to_html(include_plotlyjs="cdn"),
        file_name="embeddings_visualization.html",
        mime="text/html",
    )


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

        _CMP_STEPS = 5 if viz_only else 9
        _prog = st.progress(0, text="載入模型…")
        _step = 0

        embed_fn = load_model(selected_model)
        cache_a = path_a.parent / f"embeddings_{selected_model}" / "embeddings.npz"
        cache_b = path_b.parent / f"embeddings_{selected_model}" / "embeddings.npz"

        emb_a = extract_embeddings(paths_a, embed_fn, cache_path=cache_a)
        _step += 1; _prog.progress(_step / _CMP_STEPS, text="Folder A 特徵向量完成")

        emb_b = extract_embeddings(paths_b, embed_fn, cache_path=cache_b)
        _step += 1; _prog.progress(_step / _CMP_STEPS, text="Folder B 特徵向量完成")

        combined = np.vstack([emb_a, emb_b])
        n_emb = len(combined)

        pca_2d = PCA(n_components=2, random_state=42).fit_transform(combined)
        _step += 1; _prog.progress(_step / _CMP_STEPS, text="PCA 完成")

        perplexity = min(30, max(5, n_emb - 1))
        tsne_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(combined)
        _step += 1; _prog.progress(_step / _CMP_STEPS, text="t-SNE 完成")

        umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(combined)
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
            _step += 1; _prog.progress(_step / _CMP_STEPS, text="計算完成")
        else:
            fid_score = kid_score = lpips_score = ssim_score = None

        _prog.empty()

        st.session_state["cmp_projections"] = projections
        st.session_state["cmp_fid"] = fid_score
        st.session_state["cmp_kid"] = kid_score
        st.session_state["cmp_lpips"] = lpips_score
        st.session_state["cmp_ssim"] = ssim_score
        st.session_state["cmp_viz_only"] = viz_only
        st.session_state["cmp_paths_a"] = paths_a
        st.session_state["cmp_paths_b"] = paths_b
        st.session_state["cmp_names"] = (path_a.name, path_b.name)
        st.session_state["cmp_name_prefix"] = name
        st.session_state["cmp_model"] = selected_model

    if "cmp_projections" not in st.session_state:
        st.info("Configure inputs in the sidebar and click ▶ Run.")
        return

    projections = st.session_state["cmp_projections"]
    fid_score = st.session_state["cmp_fid"]
    kid_score = st.session_state["cmp_kid"]
    lpips_score = st.session_state["cmp_lpips"]
    ssim_score = st.session_state["cmp_ssim"]
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

    selected_method = st.selectbox("Method", list(_METHOD_KEY))
    method_key = _METHOD_KEY[selected_method]
    proj = projections[method_key]

    fig = _build_cmp_figure(paths_a, paths_b, proj, name_a, name_b)
    st.plotly_chart(fig, use_container_width=True)

    dl_fig = build_projection_figure(
        paths_a, paths_b, projections,
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


def main() -> None:
    st.set_page_config(page_title="Dataset Analysis", layout="wide")
    st.title("Dataset Analysis Tools")

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
