"""sklearn LogisticRegression → frozen PyTorch adapter。不取代既有 head artifact。"""
from __future__ import annotations

from typing import Any

import numpy as np

PROVENANCE_KEYS = (
    "model_id",
    "checkpoint_id",
    "embedding_dim",
    "keep_aspect",
    "target_res",
    "pooling",
    "l2norm",
    "scope",
    # ``None`` is a valid value for a whole-image input.  Its *presence* still
    # matters: a crop head must not silently lose its crop geometry.
    "pad",
)

FORBIDDEN_GRADCAM_LABELS = {"正常", "Unknown", "unknown", "normal"}


class HeadAdapterError(ValueError):
    pass


def _l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / max(n, 1e-12)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def sklearn_logits_matrix(head: dict) -> tuple[np.ndarray, np.ndarray, list]:
    """對齊 dino_head._logits：二類補 [0, decision_function]。回 (weight C×D, bias C, classes)."""
    clf = head["model"]
    classes = list(clf.classes_)
    coef = np.asarray(clf.coef_, dtype=np.float32)
    intercept = np.asarray(clf.intercept_, dtype=np.float32).reshape(-1)
    if coef.shape[0] == 1:
        w = np.vstack([np.zeros_like(coef), coef]).astype(np.float32)
        b = np.array([0.0, float(intercept[0])], dtype=np.float32)
    else:
        w = coef
        b = intercept.astype(np.float32)
        if b.shape[0] != w.shape[0]:
            raise HeadAdapterError("coef/intercept 類別數不一致")
    return w, b, classes


class FrozenLogisticHead:
    """純 tensor 線性頭。不繼承 nn.Module，避免測試在無 CUDA 時被意外 .cuda()。"""

    def __init__(self, weight: np.ndarray, bias: np.ndarray, *,
                 l2norm: bool, temperature: float | None, classes: list,
                 dim: int):
        self.weight = np.asarray(weight, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.l2norm = bool(l2norm)
        self.temperature = None if temperature is None else float(temperature)
        self.classes = list(classes)
        self.dim = int(dim)

    def logits_numpy(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != self.dim:
            raise HeadAdapterError(f"特徵維度 {x.shape} 與 head dim={self.dim} 不符")
        if self.l2norm:
            x = _l2n(x)
        logits = x @ self.weight.T + self.bias[None, :]
        if self.temperature is not None:
            logits = logits / max(self.temperature, 1e-6)
        return logits.astype(np.float32)

    def logits_torch(self, x):
        import torch
        import torch.nn.functional as F

        if x.ndim == 1:
            x = x.unsqueeze(0)
        if int(x.shape[-1]) != self.dim:
            raise HeadAdapterError(f"特徵維度 {tuple(x.shape)} 與 head dim={self.dim} 不符")
        if self.l2norm:
            x = F.normalize(x, dim=-1)
        w = torch.as_tensor(self.weight, device=x.device, dtype=x.dtype)
        b = torch.as_tensor(self.bias, device=x.device, dtype=x.dtype)
        logits = F.linear(x, w, b)
        if self.temperature is not None:
            logits = logits / max(self.temperature, 1e-6)
        return logits

    def target_score_torch(self, logits, idx: int):
        """二類 sklearn 的第 0 列是常數 0，不能當 Grad-CAM target；改用 -decision。"""
        import torch
        if (
            logits.shape[-1] == 2
            and self.weight.shape[0] == 2
            and float(np.abs(self.weight[0]).sum()) == 0.0
        ):
            decision = logits[..., 1]
            return decision if idx == 1 else -decision
        return logits[..., idx]

    def contrastive_score_torch(self, logits, idx: int, reference_idx: int | None = None):
        """Return the decision margin for ``idx`` against its strongest rival."""
        import torch
        if logits.shape[-1] < 2:
            raise HeadAdapterError("contrastive target 至少需要兩個類別")
        if reference_idx is None:
            other = torch.cat((logits[..., :idx], logits[..., idx + 1:]), dim=-1)
            rival = other.max(dim=-1).values
        else:
            if int(reference_idx) == int(idx):
                raise HeadAdapterError("target 與 reference 類別不可相同")
            rival = logits[..., int(reference_idx)]
        return logits[..., idx] - rival

    def probabilities_torch(self, logits):
        import torch
        return torch.softmax(logits, dim=-1)

    def predict_numpy(self, x: np.ndarray):
        logits = self.logits_numpy(x)
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        proba = e / e.sum(axis=1, keepdims=True)
        idx = proba.argmax(axis=1)
        labels = np.array([self.classes[i] for i in idx], dtype=object)
        return labels, proba.max(axis=1).astype(np.float32), proba.astype(np.float32)


def from_sklearn_head(head: dict) -> FrozenLogisticHead:
    if not isinstance(head, dict) or "model" not in head:
        raise HeadAdapterError("不是 sklearn head dict")
    w, b, classes = sklearn_logits_matrix(head)
    dim = int(head.get("dim") or w.shape[1])
    T = head.get("temperature")
    return FrozenLogisticHead(
        w, b, l2norm=bool(head.get("l2norm", True)),
        temperature=None if T is None else float(T),
        classes=classes, dim=dim,
    )


def adapter_matches_predict_head(head: dict, x: np.ndarray, *, atol: float = 1e-5) -> bool:
    from dino_head import predict_head

    adapter = from_sklearn_head(head)
    p1, c1, pr1 = predict_head(head, x)
    p2, c2, pr2 = adapter.predict_numpy(x)
    if list(p1) != list(p2):
        return False
    if not np.allclose(pr1, pr2, atol=atol, rtol=1e-4):
        return False
    return bool(np.allclose(c1, c2, atol=atol, rtol=1e-4))


def attach_provenance(head: dict, **fields) -> dict:
    prov = dict(head.get("explain_provenance") or {})
    prov.update(fields)
    out = dict(head)
    out["explain_provenance"] = prov
    return out


def provenance_of(head: dict) -> dict:
    return dict(head.get("explain_provenance") or {})


def missing_provenance_keys(head: dict) -> list[str]:
    p = provenance_of(head)
    return [
        k for k in PROVENANCE_KEYS
        if k not in p or (p[k] in (None, "") and k != "pad")
    ]


def _as_bool(value: Any) -> bool:
    """Accept persisted bool-like values without treating ``\"False\"`` as true."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def head_can_gradcam(head: dict, recipe: dict | None = None) -> tuple[bool, str]:
    missing = missing_provenance_keys(head)
    if missing:
        return False, (
            "legacy head 缺少 provenance（"
            + ", ".join(missing)
            + "），拒開 Grad-CAM，避免對不上原判定"
        )
    p = provenance_of(head)
    head_dim = head.get("dim")
    if head_dim not in (None, "") and int(p["embedding_dim"]) != int(head_dim):
        return False, "head provenance 的 embedding_dim 與 head.dim 不符"
    if _as_bool(p["l2norm"]) != _as_bool(head.get("l2norm", True)):
        return False, "head provenance 的 l2norm 與 head 本身設定不符"
    if recipe:
        for k in (
            "model_id", "keep_aspect", "target_res", "pooling",
            "checkpoint_id", "scope", "l2norm", "pad",
        ):
            # ``pad=None`` is an intentional whole-image recipe value, so it
            # must be compared rather than being treated as unspecified.
            if k not in recipe or (k != "pad" and recipe[k] in (None, "")):
                continue
            same = (_as_bool(p.get(k)) == _as_bool(recipe[k])
                    if k == "l2norm" else str(p.get(k)) == str(recipe[k]))
            if not same:
                return False, f"head provenance 的 {k}={p.get(k)} 與目前配方 {recipe[k]} 不符"
        if recipe.get("embedding_dim") not in (None, ""):
            if int(p.get("embedding_dim", -1)) != int(recipe["embedding_dim"]):
                return False, "embedding_dim 與目前模型不符"
    return True, ""


def is_forbidden_target(label: Any) -> bool:
    return str(label).strip() in FORBIDDEN_GRADCAM_LABELS
