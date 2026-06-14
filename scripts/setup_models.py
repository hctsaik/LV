"""Provision LV's model weights into the model-house -- idempotent, env-aware.

The clean "clone -> run -> it works" flow so nobody has to hand-place weights in
``models/``. Weights are large binaries NOT committed to git; this script fetches
them once per machine (or per platform model-house).

Usage
-----
    python scripts/setup_models.py                # core models (DINOv2 + Chinese-CLIP)
    python scripts/setup_models.py --with-compare # + Compare Distributions (Inception, LPIPS)

Where they go
-------------
- ``LV_MODELS_DIR``    overrides ``models/``  (DINOv2 .pth, Chinese-CLIP).
- ``LV_INCEPTION_DIR`` overrides ``model/``   (clean-fid Inception, for FID/KID).
Unset -> the package-local defaults. The CIM platform points these at a writable
model-house so the vendored submodule stays thin.

Idempotent: anything already present is skipped (printed as [ok]). Kept dependency-
light for the core models (DINOv2 via urllib, Chinese-CLIP via huggingface_hub);
the torch-heavy Compare extras only import when ``--with-compare`` is given.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

# Official Meta DINOv2 ViT-S/14 pretrained weights (public, no auth). The flat
# scripts/app loads the architecture from the bundled scripts/dinov2_hub/ and
# these weights via torch.load -- we just need the .pth on disk.
_DINOV2_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
_CHINESE_CLIP_REPO = "OFA-Sys/chinese-clip-vit-base-patch16"
_HERE = Path(__file__).resolve().parent


def _models_dir() -> Path:
    return Path(os.environ.get("LV_MODELS_DIR") or (_HERE.parent / "models"))


def _inception_dir() -> Path:
    return Path(os.environ.get("LV_INCEPTION_DIR") or (_HERE.parent / "model"))


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")

    def _hook(block: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            print(f"\r  {dest.name}: {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, _hook)  # noqa: S310 (trusted public URL)
    print()
    tmp.replace(dest)


def ensure_dinov2(models_dir: Path) -> None:
    dest = models_dir / "dinov2_vits14.pth"
    if dest.exists():
        print(f"[ok] dinov2_vits14.pth  ({dest})")
        return
    print(f"[..] dinov2_vits14.pth  <- {_DINOV2_URL}")
    _download(_DINOV2_URL, dest)


def ensure_chinese_clip(models_dir: Path) -> None:
    target = models_dir / "chinese-clip-vit-base-patch16"
    if (target / "config.json").exists():
        print(f"[ok] chinese-clip-vit-base-patch16  ({target})")
        return
    print(f"[..] chinese-clip (~750MB)  <- {_CHINESE_CLIP_REPO}")
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        _CHINESE_CLIP_REPO,
        local_dir=str(target),
        ignore_patterns=["*.msgpack", "*.h5", "*.onnx", "flax*", "tf_*"],
    )


def ensure_inception(inception_dir: Path) -> None:
    dest = inception_dir / "inception-2015-12-05.pt"
    if dest.exists():
        print(f"[ok] inception-2015-12-05.pt  ({dest})")
        return
    print("[..] inception-2015-12-05.pt (clean-fid, FID/KID)...")
    inception_dir.mkdir(parents=True, exist_ok=True)
    from cleanfid.inception_torchscript import InceptionV3W

    InceptionV3W(str(inception_dir), download=True, resize_inside=False)


def ensure_lpips() -> None:
    print("[..] lpips (alex) prewarm...")
    import lpips

    lpips.LPIPS(net="alex")  # downloads its small weights to the torch cache


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    with_compare = "--with-compare" in argv
    models_dir, inception_dir = _models_dir(), _inception_dir()
    print(f"model-house -> models: {models_dir}")
    print(f"            -> inception: {inception_dir}")
    ensure_dinov2(models_dir)
    ensure_chinese_clip(models_dir)
    if with_compare:
        ensure_inception(inception_dir)
        ensure_lpips()
    else:
        print("(skip Compare Distributions extras -- pass --with-compare for "
              "Inception/LPIPS; FID/KID also auto-download on first use.)")
    print("[done] model-house ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
