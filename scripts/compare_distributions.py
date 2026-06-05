from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA

from _utils import extract_embeddings, load_model, save_figure


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


def build_projection_figure(
    paths_a: list[Path],
    paths_b: list[Path],
    pca_2d: np.ndarray,
    name_a: str,
    name_b: str,
    fid_score: float,
    lpips_score: float,
) -> go.Figure:
    n_a = len(paths_a)
    fig = go.Figure(data=[
        go.Scatter(
            x=pca_2d[:n_a, 0].tolist(), y=pca_2d[:n_a, 1].tolist(),
            mode="markers", name=name_a,
            marker=dict(color="#3498db", size=6, opacity=0.7),
            text=[p.name for p in paths_a],
            hovertemplate="%{text}<br>Group: " + name_a + "<extra></extra>",
        ),
        go.Scatter(
            x=pca_2d[n_a:, 0].tolist(), y=pca_2d[n_a:, 1].tolist(),
            mode="markers", name=name_b,
            marker=dict(color="#e74c3c", size=6, opacity=0.7),
            text=[p.name for p in paths_b],
            hovertemplate="%{text}<br>Group: " + name_b + "<extra></extra>",
        ),
    ])
    fig.update_layout(
        title=(
            f"Distribution Comparison: {name_a} vs {name_b}<br>"
            f"<sub>FID: {fid_score:.2f} | LPIPS: {lpips_score:.4f}</sub>"
        ),
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        legend=dict(title="Group"),
    )
    return fig


def build_matplotlib_figure(
    paths_a: list[Path],
    paths_b: list[Path],
    pca_2d: np.ndarray,
    name_a: str,
    name_b: str,
    fid_score: float,
    lpips_score: float,
) -> plt.Figure:
    n_a = len(paths_a)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pca_2d[:n_a, 0], pca_2d[:n_a, 1],
               c="#3498db", label=name_a, alpha=0.6, s=25)
    ax.scatter(pca_2d[n_a:, 0], pca_2d[n_a:, 1],
               c="#e74c3c", label=name_b, alpha=0.6, s=25)
    ax.set_title(
        f"{name_a} vs {name_b}  |  FID: {fid_score:.2f}, LPIPS: {lpips_score:.4f}"
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare image distribution between two folders (FID/LPIPS)"
    )
    parser.add_argument("--folder-a", type=Path, required=True)
    parser.add_argument("--folder-b", type=Path, required=True)
    parser.add_argument("--model", default="siglip2_base",
                        choices=["dinov2", "siglip2_base", "clip"])
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
    emb_a = extract_embeddings(paths_a, embed_fn)
    emb_b = extract_embeddings(paths_b, embed_fn)

    combined = np.vstack([emb_a, emb_b])
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(combined)

    print("Computing FID...")
    fid_score = compute_fid(str(args.folder_a), str(args.folder_b))
    print(f"  FID: {fid_score:.4f}")

    print("Computing LPIPS...")
    lpips_score = compute_lpips_score(paths_a, paths_b, n_pairs=args.lpips_pairs)
    print(f"  LPIPS: {lpips_score:.4f}")

    name_a = args.folder_a.name
    name_b = args.folder_b.name
    plotly_fig = build_projection_figure(
        paths_a, paths_b, pca_2d, name_a, name_b, fid_score, lpips_score
    )
    mpl_fig = build_matplotlib_figure(
        paths_a, paths_b, pca_2d, name_a, name_b, fid_score, lpips_score
    )

    save_figure(plotly_fig, mpl_fig, args.output_dir, f"{args.name}_projection")
    plt.close("all")

    metrics = {
        "fid": round(fid_score, 4),
        "lpips": round(lpips_score, 4),
        "n_a": len(paths_a),
        "n_b": len(paths_b),
        "folder_a": str(args.folder_a),
        "folder_b": str(args.folder_b),
        "model": args.model,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.name}_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print(f"\nSaved to {args.output_dir}/")
    print(f"  {args.name}_projection.html")
    print(f"  {args.name}_projection.png")
    print(f"  {args.name}_metrics.json")


if __name__ == "__main__":
    main()
