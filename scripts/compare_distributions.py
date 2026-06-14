from __future__ import annotations

import argparse
import json
import os
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

# Inception (FID/KID) weights dir. LV_INCEPTION_DIR lets the host platform point
# at a writable model-house; default resolves next to the package (not cwd, which
# previously broke when launched from another working dir). Unset → local default.
_MODEL_DIR = Path(
    os.environ.get("LV_INCEPTION_DIR") or (Path(__file__).parent.parent / "model")
)


def _load_inception(device: torch.device):
    from cleanfid.inception_torchscript import InceptionV3W
    # Auto-provision: if the weight isn't in the model-house yet, clean-fid fetches
    # it (download=True) into _MODEL_DIR (LV_INCEPTION_DIR / local model/). A fresh
    # clone or the platform model-house then needs no manual file placement; offline
    # machines pre-seed it via `python scripts/setup_models.py --with-compare`.
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = InceptionV3W(str(_MODEL_DIR), download=True, resize_inside=False)
    return model.to(device).eval()


def get_image_paths(folder: Path) -> list[Path]:
    return sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png") for p in folder.glob(ext)
    )


def compute_fid(folder_a: str, folder_b: str) -> float:
    from cleanfid import fid as cleanfid_fid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_model = _load_inception(device)
    return float(cleanfid_fid.compute_fid(
        folder_a, folder_b,
        device=device, use_dataparallel=False, num_workers=0,
        custom_feat_extractor=feat_model,
    ))


def compute_kid(folder_a: str, folder_b: str) -> float:
    """Kernel Inception Distance — MMD-based, more reliable than FID on small datasets. Lower = more similar."""
    from cleanfid.fid import get_folder_features, kernel_distance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_model = _load_inception(device)
    feats1 = get_folder_features(folder_a, feat_model, num_workers=0,
                                  device=device, mode="clean", verbose=False)
    feats2 = get_folder_features(folder_b, feat_model, num_workers=0,
                                  device=device, mode="clean", verbose=False)
    return float(kernel_distance(feats1, feats2))


def compute_lpips_score(
    paths_a: list[Path], paths_b: list[Path], n_pairs: int = 500
) -> float:
    import lpips

    lpips_head = _MODEL_DIR / "lpips" / "v0.1" / "alex.pth"
    if not lpips_head.exists():
        raise FileNotFoundError(
            f"LPIPS weights not found: {lpips_head}\n"
            "Place alex.pth in model/lpips/v0.1/."
        )
    _prev_hub = torch.hub.get_dir()
    torch.hub.set_dir(str(_MODEL_DIR / "hub"))
    try:
        loss_fn = lpips.LPIPS(net="alex", model_path=str(lpips_head), verbose=False)
    finally:
        torch.hub.set_dir(_prev_hub)
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


def compute_ssim_score(
    paths_a: list[Path], paths_b: list[Path], n_pairs: int = 500
) -> float:
    """Structural Similarity Index averaged over random cross-group pairs. Higher = more similar (max 1.0)."""
    import numpy as np
    from skimage.metrics import structural_similarity as ssim

    n = min(n_pairs, len(paths_a), len(paths_b))
    sampled_a = random.sample(paths_a, n)
    sampled_b = random.sample(paths_b, n)

    total = 0.0
    for pa, pb in zip(sampled_a, sampled_b):
        ia = np.array(Image.open(pa).convert("RGB").resize((256, 256)))
        ib = np.array(Image.open(pb).convert("RGB").resize((256, 256)))
        total += ssim(ia, ib, channel_axis=2, data_range=255)
    return total / n


def compute_psnr_score(
    paths_a: list[Path], paths_b: list[Path], n_pairs: int = 500
) -> float:
    """Peak Signal-to-Noise Ratio averaged over random cross-group pairs. Higher = more similar (dB)."""
    n = min(n_pairs, len(paths_a), len(paths_b))
    sampled_a = random.sample(paths_a, n)
    sampled_b = random.sample(paths_b, n)
    total = 0.0
    for pa, pb in zip(sampled_a, sampled_b):
        ia = np.array(Image.open(pa).convert("RGB").resize((256, 256)), dtype=np.float64)
        ib = np.array(Image.open(pb).convert("RGB").resize((256, 256)), dtype=np.float64)
        mse = np.mean((ia - ib) ** 2)
        total += 100.0 if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)
    return total / n


def compute_inception_score(
    folder: str, n_splits: int = 10, batch_size: int = 32
) -> tuple[float, float]:
    """Inception Score for a single folder. Higher = better quality & diversity. Returns (mean, std)."""
    import torch.nn.functional as F
    import torchvision.models as tvm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = _MODEL_DIR / "hub" / "checkpoints" / "inception_v3_google-0cc3c7bd.pth"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"InceptionV3 weights not found: {weights_path}\n"
            "Run once with internet to auto-download, or copy "
            "inception_v3_google-0cc3c7bd.pth to model/hub/checkpoints/."
        )

    _prev_hub = torch.hub.get_dir()
    torch.hub.set_dir(str(_MODEL_DIR / "hub"))
    try:
        from torchvision.models import Inception_V3_Weights
        model = tvm.inception_v3(weights=Inception_V3_Weights.DEFAULT)
    finally:
        torch.hub.set_dir(_prev_hub)
    model = model.to(device).eval()

    transform = T.Compose([
        T.Resize(299),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    paths = get_image_paths(Path(folder))
    if not paths:
        raise ValueError(f"No images found in {folder}")

    preds = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch = torch.stack([
                transform(Image.open(p).convert("RGB")) for p in paths[i: i + batch_size]
            ]).to(device)
            probs = F.softmax(model(batch), dim=1)
            preds.append(probs.cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # (N, 1000)
    n = len(preds)
    n_splits = min(n_splits, n)
    split_size = max(1, n // n_splits)
    scores = []
    for i in range(n_splits):
        part = preds[i * split_size: (i + 1) * split_size]
        if len(part) == 0:
            continue
        py = part.mean(axis=0)
        kl = part * (np.log(part + 1e-10) - np.log(py[np.newaxis] + 1e-10))
        scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))

    return float(np.mean(scores)), float(np.std(scores))


_METHOD_LABELS = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}


def compute_coverage_gaps(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    最近鄰距離（cosine）用於覆蓋缺口分析。
    回傳 (d_a_to_a, d_a_to_b, d_b_to_a, d_b_to_b)：
      d_x_to_y[i] = x[i] 到 y 中最近鄰的距離（同群組時排除自身）。
    """
    from sklearn.neighbors import NearestNeighbors

    def _nn(query: np.ndarray, index: np.ndarray, exclude_self: bool) -> np.ndarray:
        k = 2 if exclude_self and len(index) > 1 else 1
        dists, _ = NearestNeighbors(n_neighbors=k, metric="cosine").fit(index).kneighbors(query)
        return dists[:, k - 1]

    return (
        _nn(emb_a, emb_a, exclude_self=True),
        _nn(emb_a, emb_b, exclude_self=False),
        _nn(emb_b, emb_a, exclude_self=False),
        _nn(emb_b, emb_b, exclude_self=True),
    )


def build_coverage_figure(
    d_a_to_a: np.ndarray,
    d_a_to_b: np.ndarray,
    d_b_to_a: np.ndarray,
    d_b_to_b: np.ndarray,
    paths_a: list[Path],
    paths_b: list[Path],
    name_a: str,
    name_b: str,
) -> go.Figure:
    """d_A vs d_B 散佈圖，以象限標示分布盲點風險。"""
    all_d_a = np.concatenate([d_a_to_a, d_b_to_a])
    all_d_b = np.concatenate([d_a_to_b, d_b_to_b])
    thr_a = float(np.percentile(all_d_a, 50))
    thr_b = float(np.percentile(all_d_b, 50))
    x_max = float(np.max(all_d_a)) * 1.08
    y_max = float(np.max(all_d_b)) * 1.08

    fig = go.Figure()

    # 象限背景色塊：標籤貼到外角，避免與資料點和閾值線重疊
    quads = [
        # (x0, x1, y0, y1, color, label, lx,         ly,         xanchor, yanchor)
        (0,     thr_a, 0,     thr_b, "#f39c12", "邊界重疊（誤報風險）",
         thr_a * 0.02, thr_b * 0.02, "left",  "bottom"),
        (0,     thr_a, thr_b, y_max, "#3498db", f"明確 {name_a} 區",
         thr_a * 0.02, y_max * 0.98, "left",  "top"),
        (thr_a, x_max, 0,     thr_b, "#e74c3c", f"明確 {name_b} 區",
         x_max * 0.98, thr_b * 0.02, "right", "bottom"),
        (thr_a, x_max, thr_b, y_max, "#9b59b6", "盲點 / 異常（漏抓風險）",
         x_max * 0.98, y_max * 0.98, "right", "top"),
    ]
    for x0, x1, y0, y1, color, label, lx, ly, xanc, yanc in quads:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=color, opacity=0.07, line_width=0, layer="below")
        fig.add_annotation(
            x=lx, y=ly, text=label, showarrow=False,
            font=dict(size=9, color=color),
            xanchor=xanc, yanchor=yanc,
            bgcolor="rgba(255,255,255,0.65)", borderpad=3,
        )

    fig.add_trace(go.Scatter(
        x=d_a_to_a.tolist(), y=d_a_to_b.tolist(),
        mode="markers", name=name_a,
        marker=dict(color="#3498db", size=6, opacity=0.75),
        text=[p.name for p in paths_a],
        hovertemplate="%{text}<br>d_A=%{x:.4f}, d_B=%{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d_b_to_a.tolist(), y=d_b_to_b.tolist(),
        mode="markers", name=name_b,
        marker=dict(color="#e74c3c", size=6, opacity=0.75),
        text=[p.name for p in paths_b],
        hovertemplate="%{text}<br>d_A=%{x:.4f}, d_B=%{y:.4f}<extra></extra>",
    ))

    # 閾值線本身不帶 annotation，改在軸上標示數值，避免與象限標籤重疊
    fig.add_vline(x=thr_a, line_dash="dash", line_color="#555", line_width=1.2)
    fig.add_hline(y=thr_b, line_dash="dash", line_color="#555", line_width=1.2)

    fig.update_layout(
        title=f"Coverage Gap Analysis — {name_a} (A) vs {name_b} (B)",
        xaxis_title=f"d_A：到最近 {name_a} 樣本的距離（cosine）",
        yaxis_title=f"d_B：到最近 {name_b} 樣本的距離（cosine）",
        xaxis=dict(range=[0, x_max]),
        yaxis=dict(range=[0, y_max]),
        legend=dict(title="Group"),
    )
    return fig


def build_projection_figure(
    paths_a: list[Path],
    paths_b: list[Path],
    projections: dict[str, np.ndarray],
    name_a: str,
    name_b: str,
    fid_score: float,
    lpips_score: float,
    kid_score: float = 0.0,
    ssim_score: float = 0.0,
) -> go.Figure:
    """projections: {"pca": ndarray(N,2), "tsne": ndarray(N,2), "umap": ndarray(N,2)}"""
    n_a = len(paths_a)
    default = next(iter(projections.values()))
    traces = [
        go.Scatter(
            x=default[:n_a, 0].tolist(), y=default[:n_a, 1].tolist(),
            mode="markers", name=name_a,
            marker=dict(color="#3498db", size=6, opacity=0.7),
            text=[p.name for p in paths_a],
            hovertemplate="%{text}<br>Group: " + name_a + "<extra></extra>",
        ),
        go.Scatter(
            x=default[n_a:, 0].tolist(), y=default[n_a:, 1].tolist(),
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
    fig = go.Figure(data=traces)
    if fid_score is not None:
        subtitle = f"FID: {fid_score:.2f} | KID: {kid_score:.6f} | LPIPS: {lpips_score:.4f} | SSIM: {ssim_score:.4f}"
        title_str = f"Distribution Comparison: {name_a} vs {name_b}<br><sub>{subtitle}</sub>"
    else:
        title_str = f"Distribution Comparison: {name_a} vs {name_b}"
    fig.update_layout(
        title=title_str,
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        legend=dict(title="Group"),
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
    parser.add_argument("--n-pairs", type=int, default=500,
                        help="LPIPS / SSIM 最大 cross-group pair 數")
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

    print("Computing KID...")
    kid_score = compute_kid(str(args.folder_a), str(args.folder_b))
    print(f"  KID: {kid_score:.6f}")

    print("Computing LPIPS...")
    lpips_score = compute_lpips_score(paths_a, paths_b, n_pairs=args.n_pairs)
    print(f"  LPIPS: {lpips_score:.4f}")

    print("Computing SSIM...")
    ssim_score = compute_ssim_score(paths_a, paths_b, n_pairs=args.n_pairs)
    print(f"  SSIM: {ssim_score:.4f}")

    projections = {"pca": pca_2d, "tsne": tsne_2d, "umap": umap_2d}
    name_a = args.folder_a.name
    name_b = args.folder_b.name
    plotly_fig = build_projection_figure(
        paths_a, paths_b, projections, name_a, name_b,
        fid_score=fid_score, lpips_score=lpips_score,
        kid_score=kid_score, ssim_score=ssim_score,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plotly_fig.write_html(str(args.output_dir / f"{args.name}_projection.html"))

    metrics = {
        "fid": round(fid_score, 4),
        "kid": round(kid_score, 6),
        "lpips": round(lpips_score, 4),
        "ssim": round(ssim_score, 4),
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
