from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from _utils import extract_embeddings, load_model, save_figure

_COLORS = ["#e74c3c", "#f39c12", "#2ecc71", "#9b59b6", "#3498db", "#1abc9c", "#95a5a6"]
_SYMBOLS = {"train": "circle", "test": "square", "valid": "diamond"}


def parse_label_file(label_path: Path, class_names: list[str]) -> str:
    """YOLO label → class name | 'mix' | 'unknown'."""
    if not label_path.exists():
        return "unknown"
    lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
    if not lines:
        return "unknown"
    class_ids = {int(ln.split()[0]) for ln in lines}
    if len(class_ids) > 1:
        return "mix"
    cid = next(iter(class_ids))
    return class_names[cid] if cid < len(class_names) else f"class_{cid}"


def discover_images(dataset_dir: Path, class_names: list[str]) -> list[dict]:
    """探索 train/test/valid 下的影像，回傳 list of {path, split, label}。"""
    records = []
    for split in ("train", "test", "valid"):
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if not images_dir.exists():
            continue
        for img_path in sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png") for p in images_dir.glob(ext)
        ):
            label_path = labels_dir / f"{img_path.stem}.txt"
            records.append({
                "path": img_path,
                "split": split,
                "label": parse_label_file(label_path, class_names),
            })
    return records


def _label_color_map(labels: list[str]) -> dict[str, str]:
    unique = sorted(set(labels))
    return {lbl: _COLORS[i % len(_COLORS)] for i, lbl in enumerate(unique)}


def build_plotly_figure(
    records: list[dict], pca_2d: np.ndarray, tsne_2d: np.ndarray
) -> go.Figure:
    """互動式圖表：PCA/t-SNE 切換 + split 篩選按鈕 + legend 類別切換。"""
    unique_labels = sorted({r["label"] for r in records})
    unique_splits = sorted({r["split"] for r in records})
    color_map = _label_color_map(unique_labels)

    traces: list[go.Scatter] = []
    trace_meta: list[dict] = []

    for label in unique_labels:
        for split in unique_splits:
            idx = [i for i, r in enumerate(records)
                   if r["label"] == label and r["split"] == split]
            if not idx:
                continue
            traces.append(go.Scatter(
                x=[pca_2d[i, 0] for i in idx],
                y=[pca_2d[i, 1] for i in idx],
                mode="markers",
                name=f"{label} ({split})",
                legendgroup=label,
                marker=dict(
                    color=color_map[label],
                    symbol=_SYMBOLS.get(split, "circle"),
                    size=7, opacity=0.8,
                ),
                text=[records[i]["path"].name for i in idx],
                hovertemplate="%{text}<br>"
                              + f"Label: {label}<br>Split: {split}"
                              + "<extra></extra>",
            ))
            trace_meta.append({
                "label": label, "split": split,
                "pca_x": [pca_2d[i, 0] for i in idx],
                "pca_y": [pca_2d[i, 1] for i in idx],
                "tsne_x": [tsne_2d[i, 0] for i in idx],
                "tsne_y": [tsne_2d[i, 1] for i in idx],
            })

    method_buttons = [
        dict(method="restyle", label="PCA", args=[{
            "x": [m["pca_x"] for m in trace_meta],
            "y": [m["pca_y"] for m in trace_meta],
        }]),
        dict(method="restyle", label="t-SNE", args=[{
            "x": [m["tsne_x"] for m in trace_meta],
            "y": [m["tsne_y"] for m in trace_meta],
        }]),
    ]

    split_buttons = [
        dict(method="restyle", label="All Splits",
             args=[{"visible": [True] * len(traces)}])
    ]
    for s in unique_splits:
        split_buttons.append(dict(
            method="restyle", label=s.capitalize(),
            args=[{"visible": [m["split"] == s for m in trace_meta]}],
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Dataset Embedding Visualization",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        legend=dict(title="Class (Split)", groupclick="toggleitem"),
        updatemenus=[
            dict(type="buttons", direction="right", x=0.0, y=1.12,
                 showactive=True, buttons=method_buttons,
                 bgcolor="#f0f0f0", bordercolor="#ccc"),
            dict(type="buttons", direction="right", x=0.38, y=1.12,
                 showactive=True, buttons=split_buttons,
                 bgcolor="#e8f4fd", bordercolor="#aad4f0"),
        ],
    )
    return fig


def build_matplotlib_figures(
    records: list[dict], pca_2d: np.ndarray, tsne_2d: np.ndarray
) -> tuple:
    """回傳 (pca_fig, tsne_fig)，每個類別一種顏色，所有 split 合併。"""
    unique_labels = sorted({r["label"] for r in records})
    color_map = _label_color_map(unique_labels)

    figs = []
    for coords, method_name in [(pca_2d, "PCA"), (tsne_2d, "t-SNE")]:
        fig, ax = plt.subplots(figsize=(10, 8))
        for label in unique_labels:
            idx = [i for i, r in enumerate(records) if r["label"] == label]
            ax.scatter(
                coords[idx, 0], coords[idx, 1],
                c=color_map[label], label=label, alpha=0.7, s=30,
            )
        ax.set_title(f"Dataset Embeddings — {method_name}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()
        figs.append(fig)
    return figs[0], figs[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize dataset embeddings (PCA/t-SNE)")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="資料集根目錄（含 train/test/valid 子資料夾）")
    parser.add_argument("--model", default="siglip2_base",
                        choices=["dinov2", "siglip2_base", "clip"])
    parser.add_argument("--classes", nargs="+", default=["apple", "banana", "orange"],
                        help="YOLO class ID 順序對應的類別名稱（0-indexed）")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    args = parser.parse_args()

    records = discover_images(args.dataset_dir, args.classes)
    if not records:
        print(f"No images found in {args.dataset_dir}")
        return

    print(f"Found {len(records)} images across {sorted({r['split'] for r in records})} splits")

    embed_fn = load_model(args.model)
    embeddings = extract_embeddings([r["path"] for r in records], embed_fn)

    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(embeddings)

    perplexity = min(30, max(5, len(records) - 1))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_2d = tsne.fit_transform(embeddings)

    plotly_fig = build_plotly_figure(records, pca_2d, tsne_2d)
    pca_fig, tsne_fig = build_matplotlib_figures(records, pca_2d, tsne_2d)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plotly_fig.write_html(str(args.output_dir / "embeddings_visualization.html"))
    pca_fig.savefig(str(args.output_dir / "embeddings_pca.png"), dpi=150, bbox_inches="tight")
    tsne_fig.savefig(str(args.output_dir / "embeddings_tsne.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    print(f"\nSaved to {args.output_dir}/")
    print("  embeddings_visualization.html")
    print("  embeddings_pca.png")
    print("  embeddings_tsne.png")


if __name__ == "__main__":
    main()
