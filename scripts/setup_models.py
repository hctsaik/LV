"""Provision LV's model weights into the model-house -- idempotent, env-aware.

The clean "clone -> run -> it works" flow so nobody has to hand-place weights.
The list of weights (what / where / from where) lives in ONE place: ../MODELS.md.
This script just reads that manifest (via model_manifest) and fetches whatever is
missing -- edit MODELS.md, this follows.

Usage
-----
    python scripts/setup_models.py                # core models (DINOv2 + Chinese-CLIP)
    python scripts/setup_models.py --with-compare # + Compare extras (Inception, LPIPS)
    python scripts/model_manifest.py              # check what's present / missing

Where they go
-------------
ONE root: ``models/`` -- each model in its own same-named subfolder (see MODELS.md).
``LV_MODELS_DIR`` relocates the whole root (e.g. a host platform's writable
model-house) so the vendored submodule stays thin (weights are NOT committed).

Idempotent: anything already present is skipped (printed as [ok]). Kept dependency-
light (urllib for plain URLs); heavier backends (huggingface_hub, cleanfid, lpips)
import lazily only when a model that needs them is actually fetched.
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

from model_manifest import ModelSpec, load_specs, _models_base


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


def _acquire_url(spec: ModelSpec) -> None:
    if not spec.url:
        raise ValueError(f"{spec.id}: acquire='url' 但 manifest 沒有 url")
    print(f"     <- {spec.url}")
    _download(spec.url, spec.target)


def _acquire_hf(spec: ModelSpec) -> None:
    from huggingface_hub import snapshot_download

    print(f"     <- huggingface: {spec.repo}")
    spec.target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        spec.repo,
        local_dir=str(spec.target),
        ignore_patterns=spec.ignore or None,
    )


def _acquire_cleanfid(spec: ModelSpec) -> None:
    from cleanfid.inception_torchscript import InceptionV3W

    print("     <- clean-fid (download=True)")
    # clean-fid manages <dir>/inception-2015-12-05.pt → point it at the model folder.
    folder = spec.target.parent
    folder.mkdir(parents=True, exist_ok=True)
    InceptionV3W(str(folder), download=True, resize_inside=False)


def _acquire_lpips_pkg(spec: ModelSpec) -> None:
    """The ~6KB LPIPS calibration head ships inside the pip `lpips` package; copy
    it to the exact path Compare reads. Removes the old "place alex.pth manually"
    foot-gun."""
    import lpips

    src = Path(lpips.__file__).parent / "weights" / spec.sentinel.split("lpips/")[-1]
    if not src.exists():  # layout fallback: weights/v0.1/alex.pth
        src = Path(lpips.__file__).parent / "weights" / "v0.1" / "alex.pth"
    if not src.exists():
        raise FileNotFoundError(
            f"lpips 套件內找不到校正頭 ({src});請確認已 pip install lpips")
    print(f"     <- lpips 套件: {src}")
    spec.target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, spec.target)


_ACQUIRERS = {
    "url": _acquire_url,
    "hf": _acquire_hf,
    "cleanfid": _acquire_cleanfid,
    "lpips_pkg": _acquire_lpips_pkg,
}


def _ensure(spec: ModelSpec) -> None:
    if spec.is_present():
        print(f"[ok] {spec.title}  ({spec.sentinel_path})")
        return
    fn = _ACQUIRERS.get(spec.acquire)
    if fn is None:
        print(f"[skip] {spec.title}: 未知 acquire='{spec.acquire}'(請看 MODELS.md)")
        return
    print(f"[..] {spec.title}")
    fn(spec)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    with_compare = "--with-compare" in argv

    specs = load_specs()
    if not specs:
        print("[err] 讀不到 MODELS.md manifest;無法佈署 model。")
        return 1

    print(f"model-house -> {_models_base()}")

    for spec in specs:
        if spec.tier == "compare" and not with_compare:
            continue
        _ensure(spec)

    if not with_compare:
        print("(略過 Compare extras -- 加 --with-compare 取得 Inception/LPIPS;"
              "FID/KID 首次使用也會自動下載。)")
    print("[done] model-house ready. 用 python scripts/model_manifest.py 檢視狀態。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
