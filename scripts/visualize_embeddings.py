from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from _utils import available_models, extract_embeddings, load_model
from safe_io import safe_read_text

_COLORS = ["#e74c3c", "#f39c12", "#2ecc71", "#9b59b6", "#3498db", "#1abc9c", "#95a5a6"]
_SYMBOLS = {"train": "circle", "test": "square", "valid": "diamond"}


def parse_label_file(label_path: Path, class_names: list[str]) -> str:
    """YOLO label → class name | 'mix' | 'unknown'."""
    if not label_path.exists():
        return "unknown"
    lines = [ln.strip() for ln in safe_read_text(label_path).splitlines() if ln.strip()]
    if not lines:  # 壞 label(讀不到/空)→ unknown,不崩潰
        return "unknown"
    class_ids = {int(ln.split()[0]) for ln in lines}
    if len(class_ids) > 1:
        return "mix"
    cid = next(iter(class_ids))
    return class_names[cid] if cid < len(class_names) else f"class_{cid}"


def discover_images_classifier(folders: list[Path], allow_flat: bool = False) -> list[dict]:
    """從分類資料集探索影像。
    結構：folder/class_name/image.jpg，split 取自 folder 名稱。
    allow_flat=True：某資料夾的類別子資料夾掃不到任何影像時，退回該資料夾根層
    平鋪掃描，label 用資料夾名（無標籤資料夾看 embedding 分佈用）。預設關閉——
    類別導向的工具（cov/quiz/gray）對平鋪資料夾應維持「掃無」而非造出假類別。
    回傳 list of {path, split, label}。
    """
    records = []
    for folder in folders:
        split = folder.name
        n_before = len(records)
        for class_dir in sorted(d for d in folder.iterdir() if d.is_dir()):
            label = class_dir.name
            for img_path in sorted(
                p for ext in ("*.jpg", "*.jpeg", "*.png") for p in class_dir.glob(ext)
            ):
                records.append({"path": img_path, "split": split, "label": label})
        if allow_flat and len(records) == n_before:
            for img_path in sorted(
                p for ext in ("*.jpg", "*.jpeg", "*.png") for p in folder.glob(ext)
            ):
                records.append({"path": img_path, "split": split, "label": folder.name})
    return records


def discover_images(folders: list[Path], class_names: list[str]) -> list[dict]:
    """從指定資料夾列表探索影像（每個資料夾需含 images/（或 VOC 的 JPEGImages/）
    和 labels/；或帶 COCO JSON／VOC XML／LabelMe／NDJSON 標註的平鋪佈局）。
    split 名稱取自資料夾名稱（e.g. train, test）。
    回傳 list of {path, split, label}。
    """
    from label_formats import (annotation_image_label,  # 輕量模組(stdlib+PIL)
                               folder_has_annotations)
    records = []
    for folder in folders:
        images_dir = folder / "images"
        labels_dir = folder / "labels"
        split = folder.name
        if images_dir.exists():
            base = images_dir
        elif (folder / "JPEGImages").exists():
            base = folder / "JPEGImages"  # 經典 VOC 佈局
        elif folder_has_annotations(folder):
            base = folder  # 平鋪佈局(COCO/VOC sidecar/LabelMe/NDJSON):影像在根層
        else:
            print(f"Warning: {images_dir} not found, skipping")
            continue
        for img_path in sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png") for p in base.glob(ext)
        ):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                label = parse_label_file(label_path, class_names)
            else:  # 無 YOLO txt → 試其他標註(單類=類名/多類=mix/無標=unknown)
                label = (annotation_image_label(img_path)
                         or parse_label_file(label_path, class_names))
            records.append({
                "path": img_path,
                "split": split,
                "label": label,
            })
    return records


def _label_color_map(labels: list[str]) -> dict[str, str]:
    unique = sorted(set(labels))
    return {lbl: _COLORS[i % len(_COLORS)] for i, lbl in enumerate(unique)}


def build_plotly_figure(
    records: list[dict],
    embeddings_per_model: dict[str, dict[str, np.ndarray]],
) -> go.Figure:
    """互動式圖表：model × method 切換 + split 篩選 + legend 類別切換。

    embeddings_per_model: {model_name: {"pca": ndarray(N,2), "tsne": ndarray(N,2), "umap": ndarray(N,2)}}
    """
    unique_labels = sorted({r["label"] for r in records})
    unique_splits = sorted({r["split"] for r in records})
    model_names = list(embeddings_per_model.keys())
    color_map = _label_color_map(unique_labels)

    first_model = model_names[0]
    # not every run computes every projection — default to the first one present
    default_coords = next(iter(embeddings_per_model[first_model].values()))

    traces: list[go.Scatter] = []
    trace_meta: list[dict] = []

    for label in unique_labels:
        for split in unique_splits:
            idx = [i for i, r in enumerate(records)
                   if r["label"] == label and r["split"] == split]
            if not idx:
                continue
            traces.append(go.Scatter(
                x=[default_coords[i, 0] for i in idx],
                y=[default_coords[i, 1] for i in idx],
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
            meta: dict = {"label": label, "split": split, "coords": {}}
            for model_name, model_coords in embeddings_per_model.items():
                for method in model_coords:
                    key = f"{model_name}_{method}"
                    meta["coords"][key] = {
                        "x": [model_coords[method][i, 0] for i in idx],
                        "y": [model_coords[method][i, 1] for i in idx],
                    }
            trace_meta.append(meta)

    # 每個 (model, method) 組合一個按鈕
    _method_labels = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}
    available_methods = list(embeddings_per_model[first_model].keys())
    model_method_buttons = []
    for model_name in model_names:
        for method in available_methods:
            method_label = _method_labels.get(method, method.upper())
            key = f"{model_name}_{method}"
            model_method_buttons.append(dict(
                method="restyle",
                label=f"{model_name} · {method_label}",
                args=[{
                    "x": [m["coords"][key]["x"] for m in trace_meta],
                    "y": [m["coords"][key]["y"] for m in trace_meta],
                }],
            ))

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
        title=f"Dataset Embedding Visualization  ({', '.join(model_names)})",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        legend=dict(title="Class (Split)", groupclick="toggleitem"),
        updatemenus=[
            dict(type="buttons", direction="right", x=0.0, y=1.15,
                 showactive=True, buttons=model_method_buttons,
                 bgcolor="#f0f0f0", bordercolor="#ccc"),
            dict(type="buttons", direction="right", x=0.0, y=1.06,
                 showactive=True, buttons=split_buttons,
                 bgcolor="#e8f4fd", bordercolor="#aad4f0"),
        ],
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize dataset embeddings (PCA/t-SNE)")
    parser.add_argument("--folders", nargs="+", type=Path, required=True,
                        help="一或多個資料夾（每個需含 images/ 和 labels/）")
    _all_models = available_models()
    parser.add_argument("--models", nargs="+", default=_all_models, choices=_all_models,
                        help=f"要比較的模型（可多選，預設全部）。可用：{_all_models}")
    parser.add_argument("--classes", nargs="+", default=["apple", "banana", "orange"],
                        help="YOLO class ID 順序對應的類別名稱（0-indexed）")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    args = parser.parse_args()

    records = discover_images(args.folders, args.classes)
    if not records:
        print("No images found in the specified folders.")
        return

    print(f"Found {len(records)} images | Models: {args.models}")

    embeddings_per_model: dict[str, dict[str, np.ndarray]] = {}
    for model_name in args.models:
        print(f"\n[{model_name}]")
        embed_fn = load_model(model_name)
        all_embs = []
        for folder in args.folders:
            folder_records = [r for r in records if r["split"] == folder.name]
            folder_paths = [r["path"] for r in folder_records]
            cache_path = folder / f"embeddings_{model_name}" / "embeddings.npz"
            if folder_paths:
                all_embs.append(extract_embeddings(folder_paths, embed_fn, cache_path=cache_path))
        embeddings = np.vstack(all_embs)

        pca = PCA(n_components=2, random_state=42)
        pca_2d = pca.fit_transform(embeddings)

        perplexity = min(30, max(5, len(records) - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_2d = tsne.fit_transform(embeddings)

        embeddings_per_model[model_name] = {"pca": pca_2d, "tsne": tsne_2d}

    fig = build_plotly_figure(records, embeddings_per_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "embeddings_visualization.html"
    fig.write_html(str(out_path))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
