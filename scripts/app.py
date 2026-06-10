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
) -> go.Figure:
    """Simple scatter for the selected model/method/split combination."""
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
                hovertemplate="%{text}<br>Label: " + label + "<br>Split: " + split + "<extra></extra>",
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
            st.session_state.pop("viz_records", None)
            st.session_state.pop("viz_embeddings", None)
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
        run = st.button("▶ Run", use_container_width=True, key="run_viz")

    if run:
        folders = [Path(f) for f in st.session_state.get("viz_folder_list", [])]
        if not folders:
            st.error("請先選擇至少一個資料夾。")
            return

        if not selected_models:
            st.error("Select at least one model.")
            return

        _n_steps = len(selected_models) * 4
        _prog = st.progress(0, text="探索圖片中…")
        _step = 0

        if mode == "Object Detector":
            missing = [str(f) for f in folders if not (f / "images").exists()]
            if missing:
                st.error(f"Folder(s) missing 'images/' subdirectory: {', '.join(missing)}")
                _prog.empty()
                return

            # classes.txt 優先級：手動選擇 > 自動偵測 > 文字輸入
            classes_path = st.session_state.get("viz_classes_file", "")
            if classes_path and Path(classes_path).exists():
                lines = [ln.strip() for ln in Path(classes_path).read_text().splitlines() if ln.strip()]
                class_names = lines
                st.success(f"使用選定的 classes.txt（{len(class_names)} 個類別）：{', '.join(class_names)}")
            else:
                detected = read_classes_txt(folders[0])
                if detected is not None:
                    class_names = detected
                    st.success(f"Auto-detected {len(class_names)} classes: {', '.join(class_names)}")
                else:
                    class_names = [c.strip() for c in class_input.split(",") if c.strip()]
                    if not class_names:
                        st.error("Enter at least one class name.")
                        _prog.empty()
                        return
                    st.info(f"Using manually entered classes: {', '.join(class_names)}")

            records = discover_images(folders, class_names)
        else:
            records = discover_images_classifier(folders)
            if records:
                detected_classes = sorted({r["label"] for r in records})
                st.success(f"自動偵測到 {len(detected_classes)} 個類別：{', '.join(detected_classes)}")

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

            n_samples = len(embeddings)
            n_comps = min(3, max(1, n_samples - 2))
            pca_2d = PCA(n_components=n_comps, random_state=42).fit_transform(embeddings)
            _step += 1
            _prog.progress(_step / _n_steps, text=f"[{model_name}] PCA 完成")

            perplexity = min(30, max(1, n_samples - 1))
            tsne_2d = TSNE(n_components=n_comps, random_state=42, perplexity=perplexity).fit_transform(embeddings)
            _step += 1
            _prog.progress(_step / _n_steps, text=f"[{model_name}] t-SNE 完成")

            n_neighbors = min(15, max(2, n_samples - 1))
            umap_2d = umap.UMAP(n_components=n_comps, n_neighbors=n_neighbors, random_state=42).fit_transform(embeddings)
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

    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    selected_model = col1.selectbox("Model", model_names)
    selected_method = col2.selectbox("Method", list(_METHOD_KEY))
    selected_split = col3.selectbox("Split", ["All"] + unique_splits)
    dim = 3 if col4.radio("維度", ["2D", "3D"], horizontal=True) == "3D" else 2

    method_key = _METHOD_KEY[selected_method]
    coords = embeddings_per_model[selected_model][method_key]

    if selected_split == "All":
        indices = list(range(len(records)))
    else:
        indices = [i for i, r in enumerate(records) if r["split"] == selected_split]

    fig = _build_viz_figure(records, coords, indices, selected_model, selected_method, dim=dim)
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

        _CMP_STEPS = 6 if viz_only else 13
        _prog = st.progress(0, text="載入模型…")
        _step = 0

        embed_fn = load_model(selected_model)
        cache_a = path_a.parent / f"embeddings_{selected_model}" / "embeddings.npz"
        cache_b = path_b.parent / f"embeddings_{selected_model}" / "embeddings.npz"

        emb_a = extract_embeddings(paths_a, embed_fn, cache_path=cache_a)
        _step += 1; _prog.progress(_step / _CMP_STEPS, text="Folder A 特徵向量完成")

        emb_b = extract_embeddings(paths_b, embed_fn, cache_path=cache_b)
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
