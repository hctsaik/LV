"""單張可解釋性面板（Visualize viewer + anomaly inspector 共用）。

這個模組只在使用者按下「分析此圖」後才做 DINO 推論。它不參與既有的
batch embedding / cache / retrieval / UMAP 管線；anomaly patch map 也以單張、
不落盤的方式重算。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from dino_explain import ExplainError, ExplainRequest, attention_map, checkpoint_identity
from explanation_render import (
    CAPTION_ANOMALY,
    CAPTION_ATTENTION,
    CAPTION_GRADCAM,
    CAPTION_INPUT,
    CAPTION_ORIGINAL,
    export_json_bytes,
    export_png_bytes,
    metadata_dict,
    render_grid,
    render_overlay,
)
from gradcam_explain import GradCAMError, token_gradcam_map
from head_adapter import (
    attach_provenance,
    head_can_gradcam,
    is_forbidden_target,
    provenance_of,
)
from object_eval import DEFAULT_POLICY, _adaptive_pad_px
from safe_io import safe_open_image

_METHODS = {
    "最後一層 · head 平均": "last_layer_head_mean",
    "單一 head": "single_head",
    "rollout（實驗）": "rollout",
}

# Keep the control honest without loading a model merely to render the form.
_DINO_HEAD_COUNTS = {
    "dinov2_vits14": 6,
    "dinov2_vitb14": 12,
    "dinov2_vitl14": 16,
    "dinov2_vitg14": 24,
}

_CAPABILITY_ORDER = ("attention", "gradcam", "anomaly")
_AUTO_TARGET = "（實際 top-1，自動）"
_MIN_ACTIONABLE_MARGIN = 0.05
_MIN_ACTIONABLE_OOF_BAL_ACC = 0.60
_NORMAL_BANK_ERROR = "目前沒有對應的 patch Normal Bank 結果"
_ANOMALY_BUILD_STEP = "① 建模 / 載入模型"
_ANOMALY_APPLY_STEP = "② 套用偵測"


def _gradcam_guidance(result) -> list[dict[str, str]]:
    """Translate evidence checks into conservative model-improvement advice."""
    items: list[dict[str, str]] = []
    if not bool(result.is_predicted_target):
        items.append({
            "level": "warning",
            "title": "這不是模型的實際判斷",
            "message": "目前是反事實 target；可研究該類假說，但不能拿來回答模型為何如此分類。",
        })
    if float(result.probability_margin) < _MIN_ACTIONABLE_MARGIN:
        items.append({
            "level": "warning",
            "title": "top-1 與 runner-up 太接近",
            "message": "先增加這兩類的邊界樣本、檢查標註與類別定義；不要依單張熱點改模型。",
        })
    diag = dict(result.head_diagnostics or {})
    if diag.get("available") and float(diag.get("balanced_accuracy", 0.0)) < _MIN_ACTIONABLE_OOF_BAL_ACC:
        items.append({
            "level": "warning",
            "title": "分類頭的 OOF 泛化偏弱",
            "message": "優先改善資料平衡、去除近重複、加入 acquisition-group holdout，並比較 crop/whole-image embedding。",
        })
    sample_oof = dict(result.sample_oof or {})
    if sample_oof and str(sample_oof.get("predicted")) != str(result.predicted_class):
        items.append({
            "level": "warning",
            "title": "此圖的 OOF 判斷不穩定",
            "message": "拿掉此圖再訓練時，預測會改變；目前 Grad-CAM 不適合作為可靠改善依據。",
        })
    faith = dict(result.faithfulness or {})
    if faith.get("status") == "supported":
        items.append({
            "level": "success",
            "title": "熱區通過局部遮蔽檢查",
            "message": "若紅區不是合理 defect cue，優先處理背景、文字、邊界等捷徑（ROI、資料增強或反例）。",
        })
    else:
        items.append({
            "level": "warning",
            "title": "熱圖尚未通過遮蔽忠實度",
            "message": "遮住最紅 patches 並未比隨機／最藍 patches 更降低分類 margin；不要據此調模型。",
        })
    return items


def _normal_bank_guide_state(session_state) -> dict[str, str]:
    """Return the next honest action for the current anomaly workflow state."""
    model = session_state.get("anomaly_model") or {}
    meta = model.get("meta") or {}
    has_patch_model = (
        meta.get("score_mode") == "patch"
        and model.get("bank_vectors") is not None
    )
    applied = session_state.get("anomaly_apply_result") or {}
    has_patch_result = applied.get("bank") is not None
    if not has_patch_model:
        return {
            "stage": "build",
            "step": _ANOMALY_BUILD_STEP,
            "button": "前往「①建模」建立 patch Normal Bank",
            "status": "目前尚未建立 patch 模式的 Normal Bank。",
        }
    if not has_patch_result:
        return {
            "stage": "apply",
            "step": _ANOMALY_APPLY_STEP,
            "button": "前往「②套用偵測」產生熱圖結果",
            "status": "patch Normal Bank 已建立，但尚未對目標資料夾執行套用偵測。",
        }
    return {
        "stage": "inspect",
        "step": _ANOMALY_APPLY_STEP,
        "button": "前往「②套用偵測」選圖查看熱圖",
        "status": "已有 patch Normal Bank 與套用結果；請在瑕疵偵測 inspector 選擇對應圖片。",
    }


def _go_to_normal_bank_guide(step: str, prefix: str) -> None:
    """Navigate to the anomaly wizard and clear the stale explain request."""
    st.session_state["tool_switch"] = "瑕疵偵測"
    st.session_state["anomaly_step"] = step
    st.session_state["_anomaly_step_shown"] = step
    if step == _ANOMALY_BUILD_STEP:
        st.session_state["anomaly_score_mode"] = "patch"
    # The Visualize viewer is a fragment. Its next fragment rerun must escape
    # to app scope before the top-level tool segmented control can navigate.
    st.session_state["_explain_app_rerun"] = True
    st.session_state.pop(f"{prefix}_pending", None)
    st.session_state.pop(f"{prefix}_error", None)


def _render_normal_bank_guide(prefix: str) -> None:
    guide = _normal_bank_guide_state(st.session_state)
    st.warning(guide["status"])
    with st.container(border=True):
        st.markdown("**如何建立並使用 Anomaly Heatmap**")
        st.markdown(
            "1. 到 **瑕疵偵測 → ①建模**，側欄把「分數依據」設成 "
            "**patch 級**。\n"
            "2. 選一批同產品、同拍攝條件且多數為良品的資料當正常參考；"
            "沒有 YOLO labels 時選 **整張影像**。\n"
            "3. 按 **建立模型**，再到 **②套用偵測** 選目標資料夾並按 "
            "**套用偵測（散點 / 校準）**。\n"
            "4. 在②的結果中選一張圖片，展開可解釋性並選 "
            "**Anomaly Heatmap**。"
        )
        st.caption(
            "Visualize 不會直接借用其他情境的 bank；熱圖要在瑕疵偵測的套用結果中查看，"
            "避免把不相容的 Normal Bank 套到目前圖片。"
        )
        st.button(
            guide["button"], key=f"{prefix}_normal_bank_guide",
            type="primary", use_container_width=True,
            on_click=_go_to_normal_bank_guide, args=(guide["step"], prefix),
        )


def _explainability_capabilities(
    *,
    record: dict,
    model_name: str,
    scope: str,
    classifier_head: dict | None,
    embedding_recipe: dict | None,
    session_head_can_train: bool,
    anomaly_map_available: bool,
) -> dict[str, dict[str, Any]]:
    """Return the capabilities exposed by the shared explanation panel.

    The panel is shared by Visualize and the anomaly inspector, but the
    explanation contracts are not interchangeable.  Keep this decision in a
    small, testable capability layer so callers explicitly provide the
    context-specific anomaly state instead of leaking a global session key
    into the shared UI.
    """
    dino_ok = str(model_name).startswith("dinov2")
    capabilities: dict[str, dict[str, Any]] = {
        "attention": {
            "label": "DINO Attention",
            "state": "ready" if dino_ok else "unavailable",
            "reason": ("可用" if dino_ok else "需要本機 DINOv2 模型"),
        },
        "gradcam": {
            "label": "分類理由（Grad-CAM）",
            "state": "unavailable",
            "reason": "需要可微分分類 head",
        },
        "anomaly": {
            "label": "Anomaly Heatmap",
            "state": "ready" if anomaly_map_available and dino_ok else "unavailable",
            "reason": ("可用" if anomaly_map_available and dino_ok
                       else "需要目前情境的 patch Normal Bank"),
        },
    }
    if not dino_ok:
        for key in ("gradcam", "anomaly"):
            capabilities[key]["reason"] = "需要本機 DINOv2 模型"
        return capabilities

    if classifier_head is None:
        if session_head_can_train:
            capabilities["gradcam"] = {
                "label": "分類理由（Grad-CAM）",
                "state": "setup",
                "reason": "可先用目前資料建立 session head",
            }
    else:
        try:
            ready, reason = _gradcam_preflight(
                record, model_name, scope, classifier_head,
            )
        except (ExplainError, OSError, ValueError) as exc:
            ready, reason = False, str(exc)
        capabilities["gradcam"] = {
            "label": "分類理由（Grad-CAM）",
            "state": "ready" if ready else "unavailable",
            "reason": "可用" if ready else reason,
        }
    return capabilities


def _render_capability_summary(capabilities: dict[str, dict[str, Any]]) -> None:
    """Show one consistent, context-aware status summary in the shared UI."""
    st.caption("共用可解釋性面板；每種方法依目前模型、資料與 Normal Bank 狀態啟用。")
    icons = {"ready": "✅", "setup": "🟡", "unavailable": "—"}
    for key in _CAPABILITY_ORDER:
        item = capabilities[key]
        icon = icons.get(item["state"], "—")
        st.caption(f"{icon} {item['label']}：{item['reason']}")


def _policy() -> dict:
    return st.session_state.get("viz_object_policy", DEFAULT_POLICY)


def _has_object_crop(r: dict) -> bool:
    """Records from either object Visualize or anomaly carry bbox + obj_index.

    Anomaly records deliberately use ``path == image_path`` even for YOLO
    objects, so comparing the two paths would collapse every object in one
    image into the same explanation identity.
    """
    return r.get("bbox") is not None and r.get("obj_index") is not None


def _is_obj_rec(r: dict) -> bool:
    ip = r.get("image_path")
    return _has_object_crop(r) or (bool(ip) and str(ip) != str(r.get("path")))


def _source_id(r: dict) -> str:
    source = str(r.get("image_path") or r.get("path") or "")
    if _has_object_crop(r):
        return f"{source}#{r.get('obj_index')}"
    return source


def _open_full(r: dict) -> Image.Image | None:
    return safe_open_image(r.get("image_path") or r.get("path"))


def _scope_recipe(recipe: dict | None, scope: str) -> dict:
    recipe = dict(recipe or {})
    return recipe if str(recipe.get("scope") or "") == scope else {}


def _prepare_scope(
    r: dict, scope: str, recipe: dict | None = None,
) -> tuple[Image.Image, dict]:
    """Return exactly the image/preprocess recipe used for an explanation.

    A supplied recipe is used only when it is for the selected scope.  This
    lets Attention intentionally inspect a different scope, while Grad-CAM is
    later rejected if that scope does not match the classifier provenance.
    """
    full = _open_full(r)
    if full is None:
        raise ExplainError("無法讀取來源影像")
    matched = _scope_recipe(recipe, scope)

    if scope == "object_crop":
        if not _has_object_crop(r):
            raise ExplainError("這筆資料沒有物件框，無法使用 object_crop")
        pol = matched or _policy()
        pad = float(pol.get("pad", DEFAULT_POLICY["pad"]))
        pad_px = _adaptive_pad_px(r["bbox"], *full.size, pad)
        from interaction import bbox_to_pixels, crop_bbox

        crop = crop_bbox(full, *r["bbox"], pad_px=pad_px)
        crop_px = bbox_to_pixels(*r["bbox"], *full.size, pad_px=pad_px)
        return crop, {
            "keep_aspect": bool(pol.get("keep_aspect", True)),
            "target_res": int(pol.get("target_res", 224)),
            "pooling": str(pol.get("pooling", pol.get("head", "cls"))),
            "pad": pad,
            "crop_bbox_px": crop_px,
            "original_size": full.size,
        }

    pol = matched
    return full, {
        "keep_aspect": bool(pol.get("keep_aspect", False)),
        "target_res": int(pol.get("target_res", 224)),
        "pooling": str(pol.get("pooling", "cls")),
        "pad": None,
        "crop_bbox_px": None,
        "original_size": full.size,
    }


def _ckpt_for(model_id: str):
    from _utils import _DEFAULT_MODELS_DIR

    p = _DEFAULT_MODELS_DIR / model_id / f"{model_id}.pth"
    if p.exists():
        return p
    q = _DEFAULT_MODELS_DIR / f"{model_id}.pth"
    return q if q.exists() else None


def _embedding_dim_for_model(model_id: str) -> int | None:
    if not str(model_id).startswith("dinov2"):
        return None
    from patch_features import model_dim

    return int(model_dim(model_id))


def _recipe_now(model_id: str, prep: dict) -> dict:
    ckpt = _ckpt_for(model_id)
    return {
        "model_id": model_id,
        "checkpoint_id": checkpoint_identity(ckpt),
        "keep_aspect": bool(prep["keep_aspect"]),
        "target_res": int(prep["target_res"]),
        "pooling": prep["pooling"],
        "embedding_dim": _embedding_dim_for_model(model_id),
        "scope": prep.get("scope", "whole_image"),
        "l2norm": True,
        "pad": prep.get("pad"),
    }


def _train_session_head(records, embeddings, model_id: str, prep: dict):
    from dino_head import evaluate_head_oof, train_head

    labels = [str(r.get("label") or "") for r in records]
    if len(set(labels)) < 2:
        raise ExplainError("目前資料少於兩個類別，無法訓練 session head")
    X = np.asarray(embeddings, dtype=np.float32)
    if X.ndim != 2 or len(X) != len(labels):
        raise ExplainError("embedding 與 records 長度不符")
    head = train_head(X, np.array(labels))
    head["explain_diagnostics"] = evaluate_head_oof(
        X, np.array(labels), source_ids=[_source_id(r) for r in records],
    )
    rec = _recipe_now(model_id, prep)
    rec["embedding_dim"] = int(X.shape[1])
    rec["l2norm"] = bool(head.get("l2norm", True))
    return attach_provenance(head, **rec)


def _display_original(r: dict, fallback: Image.Image) -> Image.Image:
    return _open_full(r) or fallback


def _run_attention(r, model_id, scope, method, head_index, embedding_recipe=None):
    img, prep = _prepare_scope(r, scope, embedding_recipe)
    req = ExplainRequest(
        image=img,
        source_id=_source_id(r),
        scope=scope,
        model_id=model_id,
        checkpoint_path=_ckpt_for(model_id),
        keep_aspect=prep["keep_aspect"],
        target_res=prep["target_res"],
        pooling=prep["pooling"],
        method=method,
        head_index=head_index,
        original_size=prep["original_size"],
        crop_bbox_px=prep["crop_bbox_px"],
        pad=prep["pad"],
    )
    return attention_map(req), _display_original(r, img), prep


def _gradcam_preflight(r, model_id, scope, head) -> tuple[bool, str]:
    if not isinstance(head, dict):
        return False, "沒有可微分分類 head"
    try:
        # Grad-CAM must use the classifier's own recipe, not Visualize's
        # currently selected object policy.
        _img, prep = _prepare_scope(r, scope, provenance_of(head))
    except ExplainError as exc:
        return False, str(exc)
    rec = _recipe_now(model_id, prep)
    rec["scope"] = scope
    rec["l2norm"] = bool(head.get("l2norm", True))
    return head_can_gradcam(head, rec)


def _run_gradcam(r, model_id, scope, head, target):
    ok, reason = _gradcam_preflight(r, model_id, scope, head)
    if not ok:
        raise GradCAMError(reason)
    img, prep = _prepare_scope(r, scope, provenance_of(head))
    req = ExplainRequest(
        image=img,
        source_id=_source_id(r),
        scope=scope,
        model_id=model_id,
        checkpoint_path=_ckpt_for(model_id),
        keep_aspect=prep["keep_aspect"],
        target_res=prep["target_res"],
        pooling=prep["pooling"],
        original_size=prep["original_size"],
        crop_bbox_px=prep["crop_bbox_px"],
        pad=prep["pad"],
    )
    return token_gradcam_map(req, head, target_class=target), _display_original(r, img), prep


def _try_anomaly_map(r) -> tuple | None:
    """Recompute one patch grid in memory; do not create/update LV cache files."""
    result = st.session_state.get("anomaly_apply_result") or {}
    bank = result.get("bank")
    records = result.get("records") or []
    if bank is None or not records:
        return None
    rec = next((x for x in records if _source_id(x) == _source_id(r)), None)
    if rec is None:
        return None

    from anomaly_heatmap import render_heatmap
    from anomaly_score import score_object
    from interaction import crop_bbox
    from patch_features import extract_patch_grid

    img = safe_open_image(rec.get("image_path") or rec.get("path"))
    if img is None or rec.get("bbox") is None:
        return None
    crop = crop_bbox(
        img, *rec["bbox"],
        pad_px=_adaptive_pad_px(rec["bbox"], *img.size, DEFAULT_POLICY["pad"]),
    )
    # extract_patch_grid has no cache_dir argument.  This is intentionally a
    # fresh, single-image computation rather than a write-through cache call.
    pf = extract_patch_grid(
        crop,
        model=result.get("_model", "dinov2_vits14"),
        target_res=int(result.get("_target_res", 224)),
    )
    _, pmap = score_object(pf["feats"], pf["grid"], bank)
    return render_heatmap(pmap, crop), pmap, crop


def _recipe_signature(recipe: dict | None) -> tuple:
    fields = (
        "model_id", "checkpoint_id", "embedding_dim", "keep_aspect",
        "target_res", "pooling", "l2norm", "scope", "pad",
    )
    data = dict(recipe or {})
    return tuple((k, str(data.get(k))) for k in fields)


def _head_signature(head: dict | None) -> dict | None:
    if not isinstance(head, dict):
        return None
    return {
        "classes": tuple(map(str, head.get("classes", []))),
        "dim": head.get("dim"),
        "provenance": _recipe_signature(provenance_of(head)),
    }


def _result_signature(
    *, kind: str, r: dict, model_id: str, scope: str, method: str | None = None,
    head_index: int | None = None, target: Any = None, head: dict | None = None,
    embedding_recipe: dict | None = None, context_token: Any = None,
) -> dict:
    return {
        "kind": kind,
        "source_id": _source_id(r),
        "model_id": str(model_id),
        "scope": str(scope),
        "method": method,
        "head_index": head_index,
        "target": None if target is None else str(target),
        "head": _head_signature(head),
        "embedding_recipe": _recipe_signature(embedding_recipe),
        "context_token": None if context_token is None else str(context_token),
    }


def _process_pending(
    prefix: str,
    r: dict,
    model_id: str,
    *,
    classifier_head: dict | None = None,
    embedding_recipe: dict | None = None,
    session_records=None,
    session_embeddings=None,
    session_head_key: str | None = None,
    context_token: Any = None,
) -> None:
    pend = st.session_state.pop(f"{prefix}_pending", None)
    if not pend:
        return
    kind = pend.get("kind")
    try:
        if kind == "attention":
            result, orig, prep = _run_attention(
                r, model_id, pend["scope"], pend["method"], pend.get("head_index"),
                embedding_recipe,
            )
            st.session_state[f"{prefix}_result"] = {
                "kind": "attention", "result": result, "orig": orig, "prep": prep,
                "signature": _result_signature(
                    kind="attention", r=r, model_id=model_id, scope=pend["scope"],
                    method=pend["method"], head_index=pend.get("head_index"),
                    embedding_recipe=embedding_recipe, context_token=context_token,
                ),
            }
        elif kind == "gradcam":
            result, orig, prep = _run_gradcam(
                r, model_id, pend["scope"], classifier_head, pend["target"])
            st.session_state[f"{prefix}_result"] = {
                "kind": "gradcam", "result": result, "orig": orig, "prep": prep,
                "signature": _result_signature(
                    kind="gradcam", r=r, model_id=model_id, scope=pend["scope"],
                    target=pend["target"], head=classifier_head,
                    context_token=context_token,
                ),
            }
        elif kind == "train_head":
            if not session_head_key or session_records is None or session_embeddings is None:
                raise ExplainError("這個檢視沒有可用的目前資料 embedding，不能訓練 session head")
            _img, prep = _prepare_scope(r, pend.get("scope") or "whole_image", embedding_recipe)
            prep["scope"] = pend.get("scope") or "whole_image"
            st.session_state[session_head_key] = _train_session_head(
                session_records, session_embeddings, model_id, prep)
            st.session_state[f"{prefix}_info"] = (
                "已用目前散點資料訓練 session head（不寫檔、不取代既有 joblib）"
            )
        elif kind == "anomaly":
            got = _try_anomaly_map(r)
            if got is None:
                raise ExplainError("目前沒有對應的 patch Normal Bank 結果")
            heat, pmap, crop = got
            st.session_state[f"{prefix}_result"] = {
                "kind": "anomaly", "heat": heat, "pmap": pmap, "orig": crop,
                "signature": _result_signature(
                    kind="anomaly", r=r, model_id=model_id, scope=pend["scope"],
                    embedding_recipe=embedding_recipe, context_token=context_token,
                ),
            }
        else:
            raise ExplainError("未知的可解釋性分析模式")
    except (ExplainError, GradCAMError, OSError, ValueError) as exc:
        st.session_state[f"{prefix}_error"] = str(exc)
        st.session_state.pop(f"{prefix}_result", None)
    except Exception as exc:  # single-image failures must not crash the Streamlit page
        st.session_state[f"{prefix}_error"] = f"分析失敗：{type(exc).__name__}: {exc}"
        st.session_state.pop(f"{prefix}_result", None)


def _add_input_metadata(meta: dict, prep: dict | None) -> dict:
    out = dict(meta)
    if prep:
        out["crop_pad"] = prep.get("pad")
        out["crop_bbox_px"] = prep.get("crop_bbox_px")
    return out


def _downloads(prefix: str, image: Image.Image, meta: dict) -> None:
    d1, d2 = st.columns(2)
    d1.download_button(
        "下載 overlay PNG", data=export_png_bytes(image),
        file_name="explain_overlay.png", mime="image/png", key=f"{prefix}_dl_png",
    )
    d2.download_button(
        "下載 metadata JSON", data=export_json_bytes(meta),
        file_name="explain_meta.json", mime="application/json", key=f"{prefix}_dl_json",
    )


def _show_result(
    prefix: str, opacity: float, show_grid: bool, current_signature: dict,
) -> None:
    packed = st.session_state.get(f"{prefix}_result")
    if not packed:
        return
    if packed.get("signature") != current_signature:
        st.warning("結果與目前圖片或控制項不一致，已過期。請重新按「分析此圖」。")
    kind = packed["kind"]
    if kind == "anomaly":
        st.caption(CAPTION_ANOMALY)
        st.image(packed["heat"], use_container_width=True, caption=CAPTION_ANOMALY)
        st.caption("紅＝相對 Normal Bank 最不像正常。這不是 DINO attention，也不是分類理由。")
        pmap = np.asarray(packed["pmap"], dtype=np.float32)
        meta = {
            "kind": "anomaly", "source_id": packed["signature"]["source_id"],
            "model_id": packed["signature"]["model_id"],
            "grid": list(pmap.shape), "raw_min": float(pmap.min()),
            "raw_max": float(pmap.max()), "raw_sum": float(pmap.sum()),
            "warning": CAPTION_ANOMALY,
        }
        with st.expander("metadata JSON"):
            st.json(meta)
        _downloads(prefix, packed["heat"], meta)
        return

    result = packed["result"]
    scores = result.display_patch_scores
    cmap = "blue_white_red" if kind == "gradcam" else "viridis"
    overlay = render_overlay(
        result.input_image, scores, alpha=opacity, cmap=cmap,
        signed=(kind == "gradcam"),
    )
    if show_grid:
        overlay = render_grid(overlay, result.grid)
        inp_g = render_grid(result.input_image, result.grid)
    else:
        inp_g = result.input_image
    cap = CAPTION_GRADCAM if kind == "gradcam" else CAPTION_ATTENTION
    c1, c2, c3 = st.columns(3)
    c1.image(packed["orig"], use_container_width=True, caption=CAPTION_ORIGINAL)
    c2.image(inp_g, use_container_width=True, caption=CAPTION_INPUT)
    c3.image(overlay, use_container_width=True, caption=cap)
    for warning in result.warnings:
        st.caption(f"⚠ {warning}")
    meta = _add_input_metadata(metadata_dict(result), packed.get("prep"))
    if kind == "gradcam":
        st.info(
            f"實際 top-1 **{result.predicted_class}** `{result.predicted_probability:.1%}` · "
            f"runner-up **{result.runner_up_class}** `{result.runner_up_probability:.1%}` · "
            f"機率差 `{result.probability_margin:.2%}`"
        )
        target_prefix = "實際決策" if result.is_predicted_target else "反事實假說"
        st.caption(
            f"{target_prefix}：**{result.target_class}** 相對 **{result.reference_class}** · "
            f"logit margin `{result.target_score:.4f}`；紅支持前者，藍支持後者。"
        )
        diag = dict(result.head_diagnostics or {})
        if diag.get("available"):
            st.caption(
                f"session head sample-level OOF：balanced accuracy "
                f"`{float(diag.get('balanced_accuracy', 0.0)):.1%}` · macro-F1 "
                f"`{float(diag.get('macro_f1', 0.0)):.1%}` · {diag.get('n_splits')} folds。"
            )
        if result.sample_oof:
            sample = result.sample_oof
            st.caption(
                f"此圖 OOF：actual **{sample.get('actual')}** · predicted "
                f"**{sample.get('predicted')}** · confidence "
                f"`{float(sample.get('confidence', 0.0)):.1%}`。"
            )
        faith = dict(result.faithfulness or {})
        if faith.get("available"):
            drops = dict(faith.get("margin_drops") or {})
            st.caption(
                f"遮蔽檢查：top-red margin drop `{float(drops.get('top', 0.0)):.4f}` · "
                f"random `{float(drops.get('random', 0.0)):.4f}` · "
                f"blue/bottom `{float(drops.get('bottom', 0.0)):.4f}`。"
            )
        st.markdown("##### 模型改善判讀")
        for item in _gradcam_guidance(result):
            message = f"**{item['title']}** — {item['message']}"
            if item["level"] == "success":
                st.success(message)
            else:
                st.warning(message)
    st.caption(
        f"{result.model_id} · input {result.input_size} · grid {result.grid} · "
        f"patch {result.patch_size} · {result.timing_s:.2f}s"
        + (f" · CLS mass {getattr(result, 'cls_mass', 0):.3f}"
           if kind == "attention" else "")
    )
    with st.expander("metadata JSON"):
        st.json(meta)
    _downloads(prefix, overlay, meta)


def render_explainability_panel(
    record: dict | None,
    *,
    model_name: str,
    key_prefix: str = "viz_explain",
    classifier_head: dict | None = None,
    embedding_recipe: dict | None = None,
    session_records=None,
    session_embeddings=None,
    session_head_key: str | None = None,
    anomaly_map_available: bool = False,
    context_token: Any = None,
) -> None:
    """Render the shared panel without inferring until a pending action exists.

    ``classifier_head`` deliberately belongs to the caller: Visualize supplies
    its transient session head, while the anomaly inspector supplies its own
    trained/loaded head.  They are never silently substituted for one another.
    ``anomaly_map_available`` follows the same rule: only the anomaly inspector
    can opt into the Normal Bank explanation, so Visualize cannot accidentally
    present a stale global anomaly result as its own capability.
    """
    if record is None:
        st.caption("選一張圖後可開可解釋性檢視。")
        return
    prefix = key_prefix
    _process_pending(
        prefix, record, model_name,
        classifier_head=classifier_head, embedding_recipe=embedding_recipe,
        session_records=session_records, session_embeddings=session_embeddings,
        session_head_key=session_head_key, context_token=context_token,
    )
    # A train-head pending action completes at the beginning of this rerun.
    # The caller's argument was evaluated before that action, so refresh it
    # here instead of leaving the Grad-CAM button disabled until an unrelated
    # later rerun.
    if classifier_head is None and session_head_key:
        classifier_head = st.session_state.get(session_head_key)
    dino_ok = str(model_name).startswith("dinov2")
    has_crop = _has_object_crop(record)
    scope_opts = ["object_crop", "whole_image"] if has_crop else ["whole_image"]
    requested_default = str((embedding_recipe or {}).get("scope") or "")
    default_scope = (requested_default if requested_default in scope_opts
                     else ("object_crop" if has_crop else "whole_image"))

    with st.expander("可解釋性", expanded=True):
        if not dino_ok:
            _render_capability_summary(_explainability_capabilities(
                record=record,
                model_name=model_name,
                scope=default_scope,
                classifier_head=classifier_head,
                embedding_recipe=embedding_recipe,
                session_head_can_train=False,
                anomaly_map_available=anomaly_map_available,
            ))
            st.warning(f"可解釋性需要本機 DINOv2。目前模型是 {model_name}，已停用。")
            return
        mode = st.radio(
            "模式",
            ["DINO Attention", "分類理由（Grad-CAM）", "Anomaly Heatmap"],
            key=f"{prefix}_mode",
            help="三種圖回答不同問題，不可混稱為同一種 heatmap。",
        )
        scope = st.selectbox(
            "分析單位", scope_opts,
            index=scope_opts.index(default_scope), key=f"{prefix}_scope",
            help="預設跟 embedding 單位走（物件＝crop，整張＝whole_image），"
                 "不是 viewer 的原圖/物件顯示切換。",
        )
        if has_crop and scope == "whole_image":
            st.warning("此 map 不是這筆 object embedding 的輸入；只可作原圖脈絡查看。")
        capabilities = _explainability_capabilities(
            record=record,
            model_name=model_name,
            scope=scope,
            classifier_head=classifier_head,
            embedding_recipe=embedding_recipe,
            session_head_can_train=(
                session_head_key is not None
                and session_records is not None
                and session_embeddings is not None
                and len({str(x.get("label")) for x in session_records}) >= 2
            ),
            anomaly_map_available=anomaly_map_available,
        )
        _render_capability_summary(capabilities)

        method_label = "最後一層 · head 平均"
        method = "last_layer_head_mean"
        head_index = None
        target = None
        classes: list = []
        gradcam_ready = False
        needs_normal_bank_guide = False
        if mode == "DINO Attention":
            method_label = st.selectbox("方法", list(_METHODS), key=f"{prefix}_method")
            method = _METHODS[method_label]
            if method == "single_head":
                max_head = _DINO_HEAD_COUNTS.get(str(model_name), 6) - 1
                head_index = int(st.number_input(
                    "head index", min_value=0, max_value=max_head, value=0,
                    key=f"{prefix}_head_i",
                ))
            if method == "rollout":
                st.caption("rollout 較慢、較耗記憶體，且只是啟發式聚合。")
        elif mode == "分類理由（Grad-CAM）":
            method = "gradcam"
            if classifier_head is None:
                if session_head_key and session_records is not None and session_embeddings is not None:
                    st.caption("沒有可微分分類頭。可用目前散點資料訓練 **session head**（不寫檔）。")
                    n_labels = len({str(x.get("label")) for x in session_records})
                    st.button(
                        "以目前資料訓練 session head", key=f"{prefix}_train_head",
                        disabled=n_labels < 2,
                        on_click=lambda: st.session_state.__setitem__(
                            f"{prefix}_pending", {"kind": "train_head", "scope": scope}),
                    )
                else:
                    st.warning("此 inspector 沒有可微分分類 head；不能把 anomaly map 當 Grad-CAM。")
            else:
                gradcam_ready, reason = _gradcam_preflight(
                    record, model_name, scope, classifier_head)
                if not gradcam_ready:
                    st.warning(reason)
                classes = [
                    c for c in classifier_head.get("classes", [])
                    if not is_forbidden_target(c)
                ]
                target_key = f"{prefix}_target"
                target_options = [_AUTO_TARGET, *classes]
                if st.session_state.get(target_key) not in target_options:
                    st.session_state.pop(target_key, None)
                if classes:
                    selected_target = st.selectbox(
                        "要解釋的決策", target_options, key=target_key,
                        help="預設回答模型為何做出實際 top-1；手選其他類別只是假說分析。",
                    )
                    target = None if selected_target == _AUTO_TARGET else selected_target
                else:
                    st.warning("head 沒有可用的 defect target 類別。")
                st.caption(
                    "預設解釋實際 top-1。手選非 top-1 會明確標成反事實假說；"
                    "正常／Unknown 不是分類頭類別。"
                )
        else:
            method = "anomaly"
            if not anomaly_map_available:
                needs_normal_bank_guide = True

        err = st.session_state.pop(f"{prefix}_error", None)
        if err == _NORMAL_BANK_ERROR:
            needs_normal_bank_guide = True
        elif err:
            st.error(err)
        if needs_normal_bank_guide:
            _render_normal_bank_guide(prefix)
        info = st.session_state.pop(f"{prefix}_info", None)
        if info:
            st.success(info)

        def _queue_analyze():
            if mode == "DINO Attention":
                st.session_state[f"{prefix}_pending"] = {
                    "kind": "attention", "scope": scope, "method": method,
                    "head_index": head_index,
                }
            elif mode == "分類理由（Grad-CAM）":
                st.session_state[f"{prefix}_pending"] = {
                    "kind": "gradcam", "scope": scope, "target": target,
                }
            else:
                st.session_state[f"{prefix}_pending"] = {"kind": "anomaly", "scope": scope}

        disabled = False
        help_txt = "只在按下後才跑模型；切換圖片不會自動推論。"
        if mode == "分類理由（Grad-CAM）":
            if classifier_head is None:
                disabled = True
                help_txt = "先訓練目前 Visualize 資料的 session head，或使用有 provenance 的分類 head。"
            elif not gradcam_ready or not classes:
                disabled = True
                help_txt = "目前輸入與分類 head 的 provenance 不相容。"
        if mode == "Anomaly Heatmap" and not anomaly_map_available:
            disabled = True
            help_txt = "請依上方引導建立 patch Normal Bank，並到瑕疵偵測②查看熱圖。"
        st.button(
            "分析此圖", key=f"{prefix}_analyze", type="primary", disabled=disabled,
            help=help_txt, on_click=_queue_analyze,
        )

        opacity = st.slider("overlay opacity", 0.0, 1.0, 0.45, 0.05,
                             key=f"{prefix}_opacity")
        show_grid = st.checkbox("顯示 patch grid", value=False, key=f"{prefix}_grid")
        kind = ("attention" if mode == "DINO Attention"
                else "gradcam" if mode == "分類理由（Grad-CAM）" else "anomaly")
        _show_result(
            prefix, opacity, show_grid,
            _result_signature(
                kind=kind, r=record, model_id=model_name, scope=scope,
                method=method if kind == "attention" else None,
                head_index=head_index if kind == "attention" else None,
                target=target if kind == "gradcam" else None,
                head=classifier_head if kind == "gradcam" else None,
                embedding_recipe=embedding_recipe, context_token=context_token,
            ),
        )
