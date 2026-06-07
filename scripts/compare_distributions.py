from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torchvision.transforms as T
from PIL import Image
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from _utils import available_models, extract_embeddings, load_model


def get_image_paths(folder: Path) -> list[Path]:
    return sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png") for p in folder.glob(ext)
    )


def compute_fid(folder_a: str, folder_b: str) -> float:
    import torch
    from cleanfid import fid as cleanfid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return float(cleanfid.compute_fid(
        folder_a, folder_b, device=device, use_dataparallel=False, num_workers=0
    ))


def compute_lpips_score(
    paths_a: list[Path], paths_b: list[Path], n_pairs: int = 500
) -> float:
    import lpips

    loss_fn = lpips.LPIPS(net="alex")
    loss_fn.eval()

    n = min(n_pairs, len(paths_a), len(paths_b))
    sampled_a = random.sample(paths_a, n)
    sampled_b = random.sample(paths_b, n)

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    total = 0.0
    with torch.no_grad():
        for pa, pb in zip(sampled_a, sampled_b):
            ia = transform(Image.open(pa).convert("RGB")).unsqueeze(0)
            ib = transform(Image.open(pb).convert("RGB")).unsqueeze(0)
            total += loss_fn(ia, ib).item()
    return total / n


_METHOD_LABELS = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}
_CMP_COLORS = ["#e74c3c", "#f39c12", "#2ecc71", "#9b59b6", "#3498db", "#1abc9c", "#95a5a6"]


def build_projection_figure(
    paths_a: list[Path],
    paths_b: list[Path],
    projections: dict[str, np.ndarray],
    name_a: str,
    name_b: str,
    fid_score: float,
    lpips_score: float,
    labels_a: list[str] | None = None,
    labels_b: list[str] | None = None,
) -> go.Figure:
    """projections: {"pca": ndarray(N,2), ...}
    When labels_a/labels_b provided, traces are split by class (color) × group (symbol ●/■).
    """
    n_a = len(paths_a)
    first_proj = next(iter(projections.values()))

    if labels_a is not None and labels_b is not None:
        all_labels = sorted(set(labels_a) | set(labels_b))
        color_map = {lbl: _CMP_COLORS[i % len(_CMP_COLORS)] for i, lbl in enumerate(all_labels)}

        # Each spec: (label, group_name, global_indices_into_combined, file_paths)
        trace_specs: list[tuple] = []
        for label in all_labels:
            idx_a = [i for i, l in enumerate(labels_a) if l == label]
            if idx_a:
                trace_specs.append((label, name_a, idx_a, [paths_a[i] for i in idx_a]))
            idx_b = [n_a + i for i, l in enumerate(labels_b) if l == label]
            if idx_b:
                trace_specs.append((label, name_b, idx_b, [paths_b[i - n_a] for i in idx_b]))

        traces = []
        for label, group, indices, paths in trace_specs:
            traces.append(go.Scatter(
                x=[first_proj[i, 0] for i in indices],
                y=[first_proj[i, 1] for i in indices],
                mode="markers",
                name=f"{label} ({group})",
                legendgroup=label,
                marker=dict(
                    color=color_map[label],
                    symbol="circle" if group == name_a else "square",
                    size=6, opacity=0.75,
                ),
                text=[p.name for p in paths],
                hovertemplate=f"%{{text}}<br>Class: {label}<br>Group: {group}<extra></extra>",
            ))

        method_buttons = []
        for key, proj in projections.items():
            method_buttons.append(dict(
                method="restyle",
                label=_METHOD_LABELS.get(key, key.upper()),
                args=[{
                    "x": [[proj[i, 0] for i in indices] for _, _, indices, _ in trace_specs],
                    "y": [[proj[i, 1] for i in indices] for _, _, indices, _ in trace_specs],
                }],
            ))

        legend_title = f"Class  ● {name_a}  ■ {name_b}"
    else:
        traces = [
            go.Scatter(
                x=first_proj[:n_a, 0].tolist(), y=first_proj[:n_a, 1].tolist(),
                mode="markers", name=name_a,
                marker=dict(color="#3498db", size=6, opacity=0.7),
                text=[p.name for p in paths_a],
                hovertemplate="%{text}<br>Group: " + name_a + "<extra></extra>",
            ),
            go.Scatter(
                x=first_proj[n_a:, 0].tolist(), y=first_proj[n_a:, 1].tolist(),
                mode="markers", name=name_b,
                marker=dict(color="#e74c3c", size=6, opacity=0.7),
                text=[p.name for p in paths_b],
                hovertemplate="%{text}<br>Group: " + name_b + "<extra></extra>",
            ),
        ]
        method_buttons = [
            dict(
                method="restyle",
                label=_METHOD_LABELS.get(key, key.upper()),
                args=[{
                    "x": [proj[:n_a, 0].tolist(), proj[n_a:, 0].tolist()],
                    "y": [proj[:n_a, 1].tolist(), proj[n_a:, 1].tolist()],
                }],
            )
            for key, proj in projections.items()
        ]
        legend_title = "Group"

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=(
            f"Distribution Comparison: {name_a} vs {name_b}<br>"
            f"<sub>FID: {fid_score:.2f} | LPIPS: {lpips_score:.4f}</sub>"
        ),
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        legend=dict(title=legend_title, groupclick="toggleitem"),
        updatemenus=[
            dict(type="buttons", direction="right", x=0.0, y=1.12,
                 showactive=True, buttons=method_buttons,
                 bgcolor="#f0f0f0", bordercolor="#ccc"),
        ] if len(projections) > 1 else [],
    )
    return fig



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare image distribution between two folders (FID/LPIPS)"
    )
    parser.add_argument("--folder-a", type=Path, required=True)
    parser.add_argument("--folder-b", type=Path, required=True)
    _models = available_models()
    parser.add_argument("--model", required=True, choices=_models,
                        help=f"模型名稱，對應 ./models/<model>.pth。可用：{_models}")
    parser.add_argument("--name", default="comparison", help="輸出檔名前綴")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--lpips-pairs", type=int, default=500,
                        help="LPIPS 最大 cross-group pair 數")
    args = parser.parse_args()

    paths_a = get_image_paths(args.folder_a)
    paths_b = get_image_paths(args.folder_b)

    if not paths_a:
        raise ValueError(f"No images found in {args.folder_a}")
    if not paths_b:
        raise ValueError(f"No images found in {args.folder_b}")

    print(f"Group A ({args.folder_a.name}): {len(paths_a)} images")
    print(f"Group B ({args.folder_b.name}): {len(paths_b)} images")

    embed_fn = load_model(args.model)
    cache_a = args.folder_a.parent / f"embeddings_{args.model}" / "embeddings.npz"
    cache_b = args.folder_b.parent / f"embeddings_{args.model}" / "embeddings.npz"
    emb_a = extract_embeddings(paths_a, embed_fn, cache_path=cache_a)
    emb_b = extract_embeddings(paths_b, embed_fn, cache_path=cache_b)

    combined = np.vstack([emb_a, emb_b])
    n = len(combined)

    pca_2d = PCA(n_components=2, random_state=42).fit_transform(combined)

    perplexity = min(30, max(5, n - 1))
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(combined)

    umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(combined)

    print("Computing FID...")
    fid_score = compute_fid(str(args.folder_a), str(args.folder_b))
    print(f"  FID: {fid_score:.4f}")

    print("Computing LPIPS...")
    lpips_score = compute_lpips_score(paths_a, paths_b, n_pairs=args.lpips_pairs)
    print(f"  LPIPS: {lpips_score:.4f}")

    projections = {"pca": pca_2d, "tsne": tsne_2d, "umap": umap_2d}
    name_a = args.folder_a.name
    name_b = args.folder_b.name
    plotly_fig = build_projection_figure(
        paths_a, paths_b, projections, name_a, name_b, fid_score, lpips_score
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plotly_fig.write_html(str(args.output_dir / f"{args.name}_projection.html"))

    metrics = {
        "fid": round(fid_score, 4),
        "lpips": round(lpips_score, 4),
        "n_a": len(paths_a),
        "n_b": len(paths_b),
        "folder_a": str(args.folder_a),
        "folder_b": str(args.folder_b),
        "model": args.model,
    }
    (args.output_dir / f"{args.name}_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print(f"\nSaved to {args.output_dir}/")
    print(f"  {args.name}_projection.html")
    print(f"  {args.name}_metrics.json")


if __name__ == "__main__":
    main()
