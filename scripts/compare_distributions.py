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
from safe_io import safe_open_image, safe_read_text

# Compare-metric weights live under the single models/ root, each in its own
# same-named folder (see MODELS.md). LV_MODELS_DIR relocates the whole root.
from model_manifest import _models_base

_MODELS_DIR = _models_base()
_FID_DIR = _MODELS_DIR / "inception-fid-kid"     # clean-fid: inception-2015-12-05.pt
_IS_DIR = _MODELS_DIR / "inception-score"        # torchvision: checkpoints/inception_v3_*
_LPIPS_DIR = _MODELS_DIR / "lpips"               # checkpoints/alexnet-* + v0.1/alex.pth


def _load_inception(device: torch.device):
    from cleanfid.inception_torchscript import InceptionV3W
    # Auto-provision: if the weight isn't in the model-house yet, clean-fid fetches
    # it (download=True) into models/inception-fid-kid/. A fresh clone or the
    # platform model-house then needs no manual file placement; offline machines
    # pre-seed it via `python scripts/setup_models.py --with-compare`.
    _FID_DIR.mkdir(parents=True, exist_ok=True)
    try:
        model = InceptionV3W(str(_FID_DIR), download=True, resize_inside=False)
    except Exception as e:  # offline / fetch failed → say which feature+model
        from model_manifest import explain
        raise FileNotFoundError(
            explain("inception-2015-12-05", feature="Compare · FID / KID",
                    expected=_FID_DIR / "inception-2015-12-05.pt")
            + f"\n  原始錯誤 : {e}") from e
    return model.to(device).eval()


def get_image_paths(folder: Path) -> list[Path]:
    return sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png") for p in folder.glob(ext)
    )


def _files_features(paths, device, feat_model):
    """Inception features for an explicit image-path list (clean-fid). Using a
    file list — not a folder glob — means we score EXACTLY the resolved images
    (no recursing into object_crops/ caches or extra splits)."""
    from cleanfid.fid import get_files_features
    return get_files_features([str(p) for p in paths], feat_model, num_workers=0,
                              device=device, mode="clean", verbose=False)


def compute_fid(paths_a: list, paths_b: list) -> float:
    from cleanfid.fid import fid_from_feats
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_model = _load_inception(device)
    return float(fid_from_feats(_files_features(paths_a, device, feat_model),
                                _files_features(paths_b, device, feat_model)))


def compute_kid(paths_a: list, paths_b: list) -> float:
    """Kernel Inception Distance — MMD-based, more reliable than FID on small datasets. Lower = more similar."""
    from cleanfid.fid import kernel_distance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_model = _load_inception(device)
    return float(kernel_distance(_files_features(paths_a, device, feat_model),
                                 _files_features(paths_b, device, feat_model)))


def compute_lpips_score(
    paths_a: list[Path], paths_b: list[Path], n_pairs: int = 500
) -> float:
    import lpips

    lpips_head = _LPIPS_DIR / "v0.1" / "alex.pth"
    if not lpips_head.exists():
        from model_manifest import explain
        raise FileNotFoundError(explain(
            "lpips_alex_head", feature="Compare · LPIPS", expected=lpips_head))
    _prev_hub = torch.hub.get_dir()
    torch.hub.set_dir(str(_LPIPS_DIR))  # alexnet backbone → models/lpips/checkpoints/
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
    scored = 0
    with torch.no_grad():
        for pa, pb in zip(sampled_a, sampled_b):
            im_a = safe_open_image(pa)
            im_b = safe_open_image(pb)
            if im_a is None or im_b is None:  # 壞檔 → 整對跳出比較,保持配對對齊
                continue
            ia = transform(im_a).unsqueeze(0)
            ib = transform(im_b).unsqueeze(0)
            total += loss_fn(ia, ib).item()
            scored += 1
    return total / scored if scored else float("nan")


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
    scored = 0
    for pa, pb in zip(sampled_a, sampled_b):
        im_a = safe_open_image(pa)
        im_b = safe_open_image(pb)
        if im_a is None or im_b is None:  # 壞檔 → 整對跳出比較,保持配對對齊
            continue
        ia = np.array(im_a.resize((256, 256)))
        ib = np.array(im_b.resize((256, 256)))
        total += ssim(ia, ib, channel_axis=2, data_range=255)
        scored += 1
    return total / scored if scored else float("nan")


def compute_psnr_score(
    paths_a: list[Path], paths_b: list[Path], n_pairs: int = 500
) -> float:
    """Peak Signal-to-Noise Ratio averaged over random cross-group pairs. Higher = more similar (dB)."""
    n = min(n_pairs, len(paths_a), len(paths_b))
    sampled_a = random.sample(paths_a, n)
    sampled_b = random.sample(paths_b, n)
    total = 0.0
    scored = 0
    for pa, pb in zip(sampled_a, sampled_b):
        im_a = safe_open_image(pa)
        im_b = safe_open_image(pb)
        if im_a is None or im_b is None:  # 壞檔 → 整對跳出比較,保持配對對齊
            continue
        ia = np.array(im_a.resize((256, 256)), dtype=np.float64)
        ib = np.array(im_b.resize((256, 256)), dtype=np.float64)
        mse = np.mean((ia - ib) ** 2)
        total += 100.0 if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)
        scored += 1
    return total / scored if scored else float("nan")


def compute_inception_score(
    folder_or_paths, n_splits: int = 10, batch_size: int = 32
) -> tuple[float, float]:
    """Inception Score for a folder OR an explicit list of image paths.
    Higher = better quality & diversity. Returns (mean, std)."""
    import torch.nn.functional as F
    import torchvision.models as tvm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = _IS_DIR / "checkpoints" / "inception_v3_google-0cc3c7bd.pth"
    if not weights_path.exists():
        from model_manifest import explain
        raise FileNotFoundError(explain(
            "inception_v3_google", feature="Compare · Inception Score",
            expected=weights_path))

    _prev_hub = torch.hub.get_dir()
    torch.hub.set_dir(str(_IS_DIR))  # checkpoints/inception_v3_google-*.pth
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

    paths = (list(folder_or_paths) if isinstance(folder_or_paths, (list, tuple))
             else get_image_paths(Path(folder_or_paths)))
    if not paths:
        raise ValueError(f"No images found: {folder_or_paths}")

    preds = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            tensors = []
            for p in paths[i: i + batch_size]:
                im = safe_open_image(p)
                if im is None:        # 壞檔 → 跳出本批(不佔一列),不崩潰
                    continue
                tensors.append(transform(im))
            if not tensors:           # 整批皆壞 → 此批無可評分影像
                continue
            batch = torch.stack(tensors).to(device)
            probs = F.softmax(model(batch), dim=1)
            preds.append(probs.cpu().numpy())

    if not preds:
        raise ValueError(f"No readable images found: {folder_or_paths}")
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
    fid_score: float | None = None,
    lpips_score: float | None = None,
    kid_score: float | None = None,
    ssim_score: float | None = None,
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
    _m = []
    if fid_score is not None:
        _m.append(f"FID: {fid_score:.2f}")
    if kid_score is not None:
        _m.append(f"KID: {kid_score:.6f}")
    if lpips_score is not None:
        _m.append(f"LPIPS: {lpips_score:.4f}")
    if ssim_score is not None:
        _m.append(f"SSIM: {ssim_score:.4f}")
    title_str = f"Distribution Comparison: {name_a} vs {name_b}"
    if _m:
        title_str += f"<br><sub>{' | '.join(_m)}</sub>"
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
