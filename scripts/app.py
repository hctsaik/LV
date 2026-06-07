from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent))

from _utils import available_models, extract_embeddings, load_model
from compare_distributions import (
    build_projection_figure,
    compute_fid,
    compute_lpips_score,
    get_image_paths,
)
from visualize_embeddings import build_plotly_figure, discover_images


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

    if not run:
        st.info("Configure inputs in the sidebar and click ▶ Run.")
        return

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

    with st.spinner("Extracting embeddings & reducing dimensions…"):
        records = discover_images(folders, class_names)
        if not records:
            st.error("No images found in the specified folders.")
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
                cache_path = (
                    folder / f"embeddings_{model_name}" / "embeddings.npz"
                )
                all_embs.append(
                    extract_embeddings(folder_paths, embed_fn, cache_path=cache_path)
                )
            embeddings = np.vstack(all_embs)

            pca = PCA(n_components=2, random_state=42)
            pca_2d = pca.fit_transform(embeddings)

            perplexity = min(30, max(5, len(records) - 1))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            tsne_2d = tsne.fit_transform(embeddings)

            embeddings_per_model[model_name] = {"pca": pca_2d, "tsne": tsne_2d}

    fig = build_plotly_figure(records, embeddings_per_model)
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "⬇ Download HTML",
        data=fig.to_html(include_plotlyjs="cdn"),
        file_name="embeddings_visualization.html",
        mime="text/html",
    )


def _compare_distributions_ui() -> None:
    st.header("Compare Distributions")

    with st.sidebar:
        folder_a = st.text_input("Folder A (direct image folder)", placeholder="dataset/train/images")
        folder_b = st.text_input("Folder B (direct image folder)", placeholder="goal/images")
        all_models = available_models()
        if not all_models:
            st.error("No .pth models found in ./models/. Add a model file and restart.")
            return
        selected_model = st.selectbox("Model", all_models)
        name = st.text_input("Output name prefix", value="comparison")
        lpips_pairs = st.number_input("LPIPS pairs", min_value=1, value=500, step=50)
        run = st.button("▶ Run", use_container_width=True, key="run_cmp")

    if not run:
        st.info("Configure inputs in the sidebar and click ▶ Run.")
        return

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

    with st.spinner("Computing embeddings, FID, and LPIPS…"):
        embed_fn = load_model(selected_model)
        cache_a = path_a.parent / f"embeddings_{selected_model}" / "embeddings.npz"
        cache_b = path_b.parent / f"embeddings_{selected_model}" / "embeddings.npz"
        emb_a = extract_embeddings(paths_a, embed_fn, cache_path=cache_a)
        emb_b = extract_embeddings(paths_b, embed_fn, cache_path=cache_b)

        combined = np.vstack([emb_a, emb_b])
        pca = PCA(n_components=2, random_state=42)
        pca_2d = pca.fit_transform(combined)

        fid_score = compute_fid(str(path_a), str(path_b))
        lpips_score = compute_lpips_score(paths_a, paths_b, n_pairs=int(lpips_pairs))

    col1, col2 = st.columns(2)
    col1.metric("FID", f"{fid_score:.4f}")
    col2.metric("LPIPS", f"{lpips_score:.4f}")

    fig = build_projection_figure(
        paths_a, paths_b, pca_2d,
        name_a=path_a.name,
        name_b=path_b.name,
        fid_score=fid_score,
        lpips_score=lpips_score,
    )
    st.plotly_chart(fig, use_container_width=True)

    metrics = {
        "fid": round(fid_score, 4),
        "lpips": round(lpips_score, 4),
        "n_a": len(paths_a),
        "n_b": len(paths_b),
        "folder_a": str(path_a),
        "folder_b": str(path_b),
        "model": selected_model,
    }

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "⬇ Download HTML",
        data=fig.to_html(include_plotlyjs="cdn").encode(),
        file_name=f"{name}_projection.html",
        mime="text/html",
    )
    dl2.download_button(
        "⬇ Download JSON",
        data=json.dumps(metrics, indent=2).encode(),
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
