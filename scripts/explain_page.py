"""Model Diagnostics page orchestration for the P1 selected-sample workflow.

The module deliberately does not run an explanation engine.  Callers provide
resolved context, audit/prediction evidence, and an optional *render-only*
``display_builder`` for a result that already exists.  This keeps page entry
and widget reruns from starting model inference and makes the product states
testable without launching Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


_HEAD_STATES = {
    "not_ready",
    "incompatible_or_stale",
    "validation_insufficient",
    "sample_level_audit_available",
    "grouped_supported",
}
_FAITHFULNESS_STATES = {
    "unavailable",
    "unsupported",
    "inconclusive",
    "single_sample_smoke_supported",
    "batch_supported",
}
_LOCALIZATION_STATES = {
    "unavailable",
    "incompatible",
    "unverified",
    "mask_agreement_supported",
}

_HEAD_LABELS = {
    "not_ready": "尚未準備分類模型",
    "incompatible_or_stale": "不相容或已過期",
    "validation_insufficient": "分類 head 證據不足",
    "sample_level_audit_available": "已有 sample-level audit",
    "grouped_supported": "grouped validation 支持",
}
_FAITHFULNESS_LABELS = {
    "unavailable": "忠實度未提供",
    "unsupported": "單張忠實度不支持",
    "inconclusive": "單張忠實度無法判定",
    "single_sample_smoke_supported": "單張 smoke test 支持",
    "batch_supported": "批次忠實度支持",
}
_LOCALIZATION_LABELS = {
    "unavailable": "沒有可用 mask",
    "incompatible": "mask mapping 不相容",
    "unverified": "定位尚未驗證",
    "mask_agreement_supported": "mask agreement 支持",
}

_SAMPLE_AUDIT_LIMITATION = "非 grouped holdout；近重複或同批次資料仍可能跨 fold。"
# The signed colour semantics are part of the public page contract.  Keep an
# ASCII fallback alongside the localized copy so browser/E2E clients can
# assert the meaning even when a platform font cannot render CJK text.
_SIGNED_LEGEND = (
    "Blue = lowers the target-reference margin; white = neutral; "
    "red = raises the target-reference margin. Signed Grad-CAM is evidence "
    "for the selected decision, not a segmentation mask or anomaly score; "
    "不是 crack mask."
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(source: Mapping[str, Any], group: str, key: str, *flat_keys: str, default=None):
    nested = _mapping(source.get(group))
    if key in nested:
        return nested[key]
    for candidate in (key, *flat_keys):
        if candidate in source:
            return source[candidate]
    return default


def _short(value: Any, length: int = 10) -> str:
    text = str(value or "").strip()
    return text[:length] + ("…" if len(text) > length else "") if text else "—"


def _format_probability(value: Any) -> str:
    if value in (None, ""):
        return "—"
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return str(value)


def _format_number(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def build_context_header(
    context: Mapping[str, Any],
    *,
    mask_status: str | None = None,
    normal_bank_status: str | None = None,
    result_status: str | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return stable, compact header fields without resolving or copying data."""
    dataset_name = _nested(context, "dataset", "name", "dataset_name")
    n_samples = _nested(context, "dataset", "n_samples")
    n_classes = _nested(context, "dataset", "n_classes")
    fingerprint = _nested(context, "dataset", "fingerprint", "dataset_fingerprint")
    dataset_bits = [str(dataset_name or "未命名資料集")]
    if n_samples not in (None, ""):
        dataset_bits.append(f"{int(n_samples):,} images")
    if n_classes not in (None, ""):
        dataset_bits.append(f"{int(n_classes)} classes")
    dataset_bits.append(f"fp:{_short(fingerprint)}")

    model_id = _nested(context, "model", "id", "model_id")
    checkpoint = _nested(context, "model", "checkpoint_id", "checkpoint_id")
    model_text = str(model_id or "—")
    if checkpoint:
        model_text += f" · ckpt:{_short(checkpoint)}"

    head_provenance = _mapping(context.get("head_provenance"))
    head_id = _nested(context, "head", "id", "head_id", "head_ref")
    if isinstance(head_id, Mapping):
        head_id = head_id.get("digest") or head_id.get("id") or "available"
    scope = _nested(context, "head", "scope", default=context.get("scope"))
    pooling = _nested(context, "head", "pooling", default=head_provenance.get("pooling"))
    head_res = _nested(context, "head", "target_res", default=head_provenance.get("target_res"))
    head_bits = [str(head_id or "unavailable")]
    for value in (scope, pooling, f"{head_res}px" if head_res not in (None, "") else None):
        if value:
            head_bits.append(str(value))

    filename = _nested(context, "image", "filename", "filename", "source_relative_path")
    label = _nested(context, "image", "label", "label")
    split = _nested(context, "image", "split", "split")
    source_sha = _nested(context, "image", "sha256", "source_sha256")
    image_bits = [str(filename or "—")]
    if label not in (None, ""):
        image_bits.append(f"label:{label}")
    if split not in (None, ""):
        image_bits.append(f"split:{split}")
    if source_sha:
        image_bits.append(f"sha:{_short(source_sha)}")

    mask = mask_status or str(context.get("mask_status") or "unavailable")
    bank = normal_bank_status or str(context.get("normal_bank_status") or "unavailable")
    result = result_status or str(context.get("result_status") or "not_run")
    return (
        ("Dataset", " · ".join(dataset_bits)),
        ("Model", model_text),
        ("Head", " · ".join(head_bits)),
        ("Image", " · ".join(image_bits)),
        ("Mask", mask),
        ("Normal Bank", bank),
        ("Result", result),
        ("Context ID", _short(context.get("context_id"), 16)),
    )


def build_prediction_semantics(
    context: Mapping[str, Any],
    prediction: Mapping[str, Any] | None,
    *,
    selected_target: str | None = None,
) -> dict[str, Any]:
    """Keep label, inference, held-out OOF, target, and reference distinct."""
    pred = _mapping(prediction)
    actual = pred.get("actual_label", _nested(context, "image", "label", "label"))
    current_top1 = pred.get("current_top1") or pred.get("predicted_class")
    runner_up = pred.get("runner_up") or pred.get("reference")
    target = selected_target or pred.get("target") or current_top1
    counterfactual = bool(
        target not in (None, "")
        and current_top1 not in (None, "")
        and str(target) != str(current_top1)
    )
    # For the actual decision, compare top-1 with its strongest runner-up.  A
    # manually selected target is a counterfactual and must instead be compared
    # with the model's actual top-1; reusing the old top-1 margin would silently
    # attach the wrong scalar to the new target.
    reference = current_top1 if counterfactual else (pred.get("reference") or runner_up)
    recorded_target = pred.get("target") or current_top1
    recorded_reference = pred.get("reference") or runner_up
    margin_matches = (
        not counterfactual
        or (
            str(recorded_target) == str(target)
            and str(recorded_reference) == str(reference)
        )
    )
    oof = _mapping(pred.get("oof"))
    oof_available = bool(oof.get("available", pred.get("oof_available", False)))
    oof_prediction = oof.get("predicted", pred.get("oof_prediction")) if oof_available else None
    calibrated = bool(pred.get("calibrated", False))
    return {
        "actual_label": str(actual) if actual not in (None, "") else "—",
        "current_top1": str(current_top1) if current_top1 not in (None, "") else "—",
        "current_probability": pred.get("current_probability", pred.get("confidence")),
        "runner_up": str(runner_up) if runner_up not in (None, "") else "—",
        "runner_up_probability": pred.get("runner_up_probability"),
        "margin": (
            pred.get("margin", pred.get("probability_margin"))
            if margin_matches else None
        ),
        "oof_available": oof_available,
        "oof_prediction": str(oof_prediction) if oof_prediction not in (None, "") else None,
        "oof_probability": oof.get("probability", pred.get("oof_probability")),
        "calibration_status": "held-out calibrated" if calibrated else "uncalibrated probability",
        "target": str(target) if target not in (None, "") else "—",
        "reference": str(reference) if reference not in (None, "") else "—",
        "counterfactual": counterfactual,
    }


def build_head_audit(
    audit: Mapping[str, Any] | None,
    *,
    target: str | None,
) -> dict[str, Any]:
    source = _mapping(audit)
    per_class = _mapping(source.get("per_class"))
    target_metrics = _mapping(source.get("target_metrics")) or _mapping(per_class.get(str(target)))
    kind = str(source.get("kind") or "unavailable")
    limitations = source.get("limitations")
    if kind == "sample_stratified_oof" and not limitations:
        limitations = _SAMPLE_AUDIT_LIMITATION
    return {
        "available": bool(source.get("available", False)),
        "kind": kind,
        "n_samples": source.get("n_samples"),
        "n_splits": source.get("n_splits"),
        "balanced_accuracy": source.get("balanced_accuracy"),
        "macro_f1": source.get("macro_f1"),
        "target": target,
        "target_precision": target_metrics.get("precision"),
        "target_recall": target_metrics.get("recall"),
        "target_f1": target_metrics.get("f1"),
        "target_support": target_metrics.get("support"),
        "limitations": str(limitations or "尚未提供 validation split 與 leakage audit。"),
        # Callers may pass the already-normalized audit state as
        # ``explicit_state`` (the app does this for a head that has not been
        # built yet). Preserve it instead of downgrading the state to the
        # generic validation-insufficient fallback.
        "explicit_state": source.get("state") or source.get("explicit_state"),
    }


def classify_head_validation(audit: Mapping[str, Any], *, stale: bool = False) -> str:
    if stale:
        return "incompatible_or_stale"
    explicit = audit.get("explicit_state") or audit.get("state")
    if explicit in _HEAD_STATES:
        return str(explicit)
    if not audit.get("available"):
        return "validation_insufficient"
    try:
        ba = float(audit.get("balanced_accuracy"))
        macro_f1 = float(audit.get("macro_f1"))
    except (TypeError, ValueError):
        return "validation_insufficient"
    recall = audit.get("target_recall")
    if ba < 0.60 or macro_f1 < 0.60:
        return "validation_insufficient"
    if recall not in (None, "") and float(recall) < 0.80:
        return "validation_insufficient"
    kind = str(audit.get("kind") or "")
    return "grouped_supported" if kind in {"grouped_oof", "grouped_holdout"} else "sample_level_audit_available"


def classify_faithfulness(evidence: Mapping[str, Any] | None) -> str:
    source = _mapping(evidence)
    status = str(source.get("status") or "unavailable")
    aliases = {
        "supported": "single_sample_smoke_supported",
        "weak": "unsupported",
        "not_supported": "unsupported",
    }
    status = aliases.get(status, status)
    return status if status in _FAITHFULNESS_STATES else "inconclusive"


def classify_localization(mask_status: str | None) -> str:
    status = str(mask_status or "unavailable").strip().lower()
    if status in {"available", "resolved", "localization_not_evaluated", "not_evaluated"}:
        return "unverified"
    if status in {"incompatible", "mapping_incompatible"}:
        return "incompatible"
    if status in {"supported", "mask_agreement_supported"}:
        return "mask_agreement_supported"
    return "unavailable"


def build_evidence_badges(
    *,
    head_audit: Mapping[str, Any],
    faithfulness: Mapping[str, Any] | None,
    mask_status: str | None,
    stale: bool = False,
) -> dict[str, dict[str, str]]:
    head = classify_head_validation(head_audit, stale=stale)
    attribution = classify_faithfulness(faithfulness)
    localization = classify_localization(mask_status)
    return {
        "head": {"state": head, "label": _HEAD_LABELS[head]},
        "faithfulness": {"state": attribution, "label": _FAITHFULNESS_LABELS[attribution]},
        "localization": {"state": localization, "label": _LOCALIZATION_LABELS[localization]},
    }


def compose_evidence_summary(badges: Mapping[str, Mapping[str, str]]) -> str:
    head = badges["head"]["state"]
    faith = badges["faithfulness"]["state"]
    loc = badges["localization"]["state"]
    if head == "incompatible_or_stale":
        return "目前 context 不相容或已過期，不能把舊結果配上目前設定。"
    if head == "not_ready":
        return "尚未建立分類 head；請先完成模型準備，再產生分類證據熱圖。"
    if head == "validation_insufficient":
        base = "分類 head 證據不足；熱圖只能研究分類器的當次錯誤依據。"
    elif head == "sample_level_audit_available":
        base = "已有 sample-level audit；跨 acquisition／lot／device 泛化尚未驗證。"
    else:
        base = "Grouped classifier evidence 目前受支持。"
    if faith == "single_sample_smoke_supported":
        base += " 單張遮蔽 smoke test 支持當次 margin，但不能推廣到整個模型。"
    elif faith in {"unsupported", "inconclusive", "unavailable"}:
        base += " 單張歸因忠實度尚未獲得支持。"
    if loc != "mask_agreement_supported":
        base += " Localization 尚未驗證，不能宣稱模型找到或分割 crack。"
    return base


@dataclass(frozen=True)
class PageViewModel:
    empty: bool
    header: tuple[tuple[str, str], ...]
    prediction: Mapping[str, Any]
    audit: Mapping[str, Any]
    badges: Mapping[str, Mapping[str, str]]
    summary: str
    head_not_ready: bool
    weak_head: bool
    blocked: bool
    warning: str | None
    counterfactual: bool


def build_page_view(
    context: Mapping[str, Any] | None,
    *,
    head_audit: Mapping[str, Any] | None = None,
    prediction: Mapping[str, Any] | None = None,
    faithfulness: Mapping[str, Any] | None = None,
    mask_status: str | None = None,
    normal_bank_status: str | None = None,
    result_status: str | None = None,
    selected_target: str | None = None,
) -> PageViewModel:
    if not context:
        return PageViewModel(
            empty=True, header=(), prediction={}, audit={}, badges={}, summary="",
            head_not_ready=False, weak_head=False, blocked=False,
            warning=None, counterfactual=False,
        )
    stale = str(result_status or context.get("result_status") or "").lower() in {
        "stale", "historical", "incompatible",
    }
    semantics = build_prediction_semantics(context, prediction, selected_target=selected_target)
    audit_view = build_head_audit(head_audit, target=semantics["target"])
    explicit_mask = mask_status or str(context.get("mask_status") or "unavailable")
    badges = build_evidence_badges(
        head_audit=audit_view,
        faithfulness=faithfulness,
        mask_status=explicit_mask,
        stale=stale,
    )
    head_state = badges["head"]["state"]
    head_not_ready = head_state == "not_ready"
    weak = head_state == "validation_insufficient"
    blocked = head_state == "incompatible_or_stale"
    warning = None
    if head_not_ready:
        warning = "尚未準備分類模型；請先在上方完成模型準備。"
    elif weak:
        warning = "分類 head 不可靠。熱圖只能用來研究分類器為何出錯，不能判斷裂痕位置。"
    elif blocked:
        warning = "Context 不相容或已過期；請以目前資料與 head 重新建立診斷。"
    if semantics["counterfactual"]:
        counterfactual_warning = (
            f"反事實分析：模型目前 top-1 是 {semantics['current_top1']}，"
            f"所選 target 是 {semantics['target']}。這不是模型實際做出該分類的理由。"
        )
        warning = f"{warning} {counterfactual_warning}" if warning else counterfactual_warning
    return PageViewModel(
        empty=False,
        header=build_context_header(
            context,
            mask_status=explicit_mask,
            normal_bank_status=normal_bank_status,
            result_status=result_status,
        ),
        prediction=semantics,
        audit=audit_view,
        badges=badges,
        summary=compose_evidence_summary(badges),
        head_not_ready=head_not_ready,
        weak_head=weak,
        blocked=blocked,
        warning=warning,
        counterfactual=bool(semantics["counterfactual"]),
    )


@dataclass(frozen=True)
class PageActions:
    inspect_head_clicked: bool = False
    attribution_requested: bool = False
    research_confirmed: bool = False
    rendered_attribution: bool = False
    selected_target: str | None = None


def _render_context_header(st: Any, header: Sequence[tuple[str, str]]) -> None:
    """Render a human-first sample summary with technical details collapsed."""
    st.markdown("### 目前分析對象")
    values = dict(header)
    primary = ("Image", "Dataset", "Model", "Head")
    display_names = {
        "Image": "影像",
        "Dataset": "資料集",
        "Model": "模型",
        "Head": "分類模型",
    }
    columns = st.columns(4)
    for column, label in zip(columns, primary):
        column.markdown(
            f"**{display_names[label]}**  \n{values.get(label, '—')}"
        )
    if values.get("Context ID") not in (None, "—"):
        st.caption(f"Context ID：{values['Context ID']}")
    status_bits = [
        f"Mask：{values.get('Mask', '—')}",
        f"Normal Bank：{values.get('Normal Bank', '—')}",
        f"Result：{values.get('Result', '—')}",
    ]
    st.caption(" · ".join(status_bits))
    technical = tuple(
        (label, values.get(label, "—"))
        for label in ("Mask", "Normal Bank", "Result", "Context ID")
        if label in values
    )
    if technical:
        with st.expander("查看資料與模型詳細資訊", expanded=False):
            detail_columns = st.columns(2)
            for index, (label, value) in enumerate(technical):
                detail_columns[index % len(detail_columns)].caption(
                    f"{label}：{value}"
                )


def _render_prediction(st: Any, prediction: Mapping[str, Any]) -> None:
    st.markdown("### 3. 模型判斷")
    columns = st.columns(4)
    values = (
        ("真實標籤", prediction["actual_label"]),
        ("模型目前判斷", f"{prediction['current_top1']} · {_format_probability(prediction['current_probability'])}"),
        ("第二候選", f"{prediction['runner_up']} · {_format_probability(prediction['runner_up_probability'])}"),
        ("Target-reference margin", _format_number(prediction["margin"])),
    )
    for column, (label, value) in zip(columns, values):
        column.metric(label, value)
    if prediction["oof_available"]:
        st.caption(
            f"OOF（留出資料）判斷：{prediction['oof_prediction']}"
            f" · {_format_probability(prediction['oof_probability'])}"
        )
    else:
        st.caption("OOF（留出資料）判斷：unavailable；不可拿目前 inference 冒充 OOF。")
    st.caption(f"機率狀態：{prediction['calibration_status']}")


def _render_audit(st: Any, audit: Mapping[str, Any]) -> None:
    st.markdown("#### 分類可靠度（sample-level OOF）")
    columns = st.columns(4)
    for column, (label, value) in zip(columns, (
        ("Split", audit["kind"]),
        ("Samples / folds", f"{audit['n_samples'] or '—'} / {audit['n_splits'] or '—'}"),
        ("Balanced accuracy", _format_probability(audit["balanced_accuracy"])),
        ("Macro-F1", _format_probability(audit["macro_f1"])),
    )):
        column.metric(label, value)
    st.caption(
        f"Target {audit['target']}：precision {_format_probability(audit['target_precision'])} · "
        f"recall {_format_probability(audit['target_recall'])} · "
        f"F1 {_format_probability(audit['target_f1'])} · support {audit['target_support'] or '—'}"
    )
    st.warning(audit["limitations"])


def _render_badges(st: Any, badges: Mapping[str, Mapping[str, str]], summary: str) -> None:
    st.markdown("#### 證據狀態")
    columns = st.columns(3)
    names = (("分類模型可靠度", "head"), ("熱圖忠實度", "faithfulness"), ("定位狀態", "localization"))
    for column, (title, key) in zip(columns, names):
        column.markdown(f"**{title}**  \n{badges[key]['label']}")
        column.caption(badges[key]["state"])
    st.info(summary)


def _render_primary_summary(st: Any, view: PageViewModel) -> None:
    """Render only the evidence needed to decide what to do next.

    The diagnostics page used to put context, three badges, four metrics,
    audit details, and localization caveats at the same visual level.  Keep
    the primary path to one conclusion, three decision facts, and one next
    action; provenance and fold-level evidence are rendered in the detail
    expander below.
    """
    st.markdown("### 分析結論")
    if view.head_not_ready:
        st.info("尚未建立分類模型；先準備所選解析度的 head，才有分類證據可看。")
    elif view.blocked:
        st.error("目前 context 已過期或不相容；請回到 Visualize 重新選圖。")
    elif view.weak_head:
        st.warning("分類 head 證據不足；熱圖只能研究分類器的當次判斷，不能證明找到 crack。")
    else:
        st.info("分類證據可以檢視，但目前仍不是 grouped validation，也不能宣稱 crack localization。")

    prediction = view.prediction
    columns = st.columns(3)
    facts = (
        ("真實標籤", prediction.get("actual_label") or "—"),
        ("模型目前判斷", f"{prediction.get('current_top1') or '—'} · {_format_probability(prediction.get('current_probability'))}"),
        ("OOF（留出資料）", (
            f"{prediction.get('oof_prediction')} · {_format_probability(prediction.get('oof_probability'))}"
            if prediction.get("oof_available") else "unavailable"
        )),
    )
    for column, (label, value) in zip(columns, facts):
        column.metric(label, value)
    st.caption(
        f"Target-reference margin：{_format_number(prediction.get('margin'))} · "
        f"機率：{prediction.get('calibration_status') or 'unknown'}"
    )
    head = view.badges["head"]
    faithfulness = view.badges["faithfulness"]
    localization = view.badges["localization"]
    st.markdown(f"**分類模型可靠度**：{head['label']}")
    st.caption(
        f"熱圖忠實度：{faithfulness['label']} · "
        f"定位：{localization['label']}（{localization['state']}）"
    )


def _render_attribution(st: Any, display: Mapping[str, Any]) -> None:
    columns = st.columns(3)
    columns[0].image(display.get("source_image"), caption="原圖", use_container_width=True)
    columns[1].image(display.get("model_input"), caption="模型實際輸入", use_container_width=True)
    columns[2].image(display.get("overlay"), caption="Signed Grad-CAM overlay", use_container_width=True)
    st.caption(_SIGNED_LEGEND)


def render_model_diagnostics_page(
    context: Mapping[str, Any] | None,
    *,
    head_audit: Mapping[str, Any] | None = None,
    prediction: Mapping[str, Any] | None = None,
    faithfulness: Mapping[str, Any] | None = None,
    mask_status: str | None = None,
    normal_bank_status: str | None = None,
    result_status: str | None = None,
    selected_target: str | None = None,
    target_options: Sequence[str] = (),
    research_authorized: bool = False,
    display_builder: Callable[[float, bool], Mapping[str, Any]] | None = None,
    primary_action_label: str | None = None,
    primary_action_key: str = "diag_attribution",
    st_module: Any = None,
    show_title: bool = True,
) -> PageActions:
    """Render the P1 workspace and return explicit orchestration actions.

    ``display_builder`` must be render-only: it receives opacity/grid controls
    and returns source/model-input/overlay images for an already-computed raw
    result.  This function never imports or invokes the Grad-CAM engine.
    """
    if st_module is None:
        import streamlit as st_module
    st = st_module
    if show_title:
        st.title("模型診斷")
    if not context:
        st.info("請先到 Visualize 執行資料分析，再選擇一張圖片按「分析這張圖」。")
        return PageActions()

    current_top1 = str(_mapping(prediction).get("current_top1") or _mapping(prediction).get("predicted_class") or "")
    chosen_target = selected_target or current_top1 or None
    options = [str(value) for value in target_options if str(value)]
    if current_top1 and current_top1 not in options:
        options.insert(0, current_top1)
    if options:
        with st.expander("進階：反事實 target", expanded=False):
            default_index = options.index(chosen_target) if chosen_target in options else 0
            chosen_target = st.selectbox(
                "研究 target",
                options,
                index=default_index,
                help="非目前 top-1 的 target 是反事實假說，不是模型實際分類理由。",
            )

    view = build_page_view(
        context,
        head_audit=head_audit,
        prediction=prediction,
        faithfulness=faithfulness,
        mask_status=mask_status,
        normal_bank_status=normal_bank_status,
        result_status=result_status,
        selected_target=chosen_target,
    )
    _render_primary_summary(st, view)

    if view.warning:
        # The concise conclusion above carries the primary state.  Keep the
        # detailed warning only for counterfactual context so the same caveat
        # is not repeated three times in the main column.
        if view.counterfactual:
            st.warning(view.warning)
    if view.counterfactual:
        st.warning("反事實 target 標籤會保留在本次結果；不可把它描述成模型實際 top-1 的理由。")

    with st.expander("查看分析 context、可靠度與 OOF 詳細資料", expanded=False):
        _render_context_header(st, view.header)
        _render_badges(st, view.badges, view.summary)
        _render_audit(st, view.audit)
        st.caption("P1 不顯示 mask overlay，也不以肉眼重疊判定 localization。")

    inspect_clicked = False
    if view.blocked:
        st.caption("目前資料與模型設定不相容；請回到 Visualize 重新選圖。")
    elif view.head_not_ready:
        if primary_action_label:
            st.caption("按一次主操作會準備所選解析度的分類模型，並接著產生熱圖。")
        else:
            st.caption("完成上方模型準備後，這裡才會出現熱圖按鈕。")
    elif view.weak_head:
        # Keep the limitation visible, but do not force a redundant
        # acknowledgement click before the user can inspect the evidence.
        st.caption("分類 head 證據不足；以下熱圖僅供研究解讀，不能證明找到 crack。")

    allowed = (
        not view.blocked and not view.head_not_ready
    )
    attribution_requested = False
    rendered = False
    st.markdown("### 分類證據熱圖")
    if not view.blocked and display_builder is None and primary_action_label:
        attribution_requested = bool(st.button(
            primary_action_label, key=primary_action_key, type="primary"))
    elif allowed and display_builder is None:
        attribution_requested = bool(st.button(
            "產生分類證據熱圖（Signed Grad-CAM）", key=primary_action_key,
            type="primary"))
    elif allowed and display_builder is not None:
        opacity = float(st.slider("overlay opacity", 0.0, 1.0, 0.50, 0.05))
        show_grid = bool(st.checkbox("顯示 patch grid", value=False))
        display = _mapping(display_builder(opacity, show_grid))
        _render_attribution(st, display)
        rendered = True

    return PageActions(
        inspect_head_clicked=inspect_clicked,
        attribution_requested=attribution_requested,
        # Kept in the return contract for callers that still inspect this
        # field; authorization is no longer a separate UI step.
        research_confirmed=False,
        rendered_attribution=rendered,
        selected_target=chosen_target,
    )


# Stable app-level entry point.  Keeping this alias avoids coupling the route
# to the longer test-oriented function name while preserving one renderer.
render_model_diagnostics = render_model_diagnostics_page


__all__ = [
    "PageActions",
    "PageViewModel",
    "build_context_header",
    "build_evidence_badges",
    "build_head_audit",
    "build_page_view",
    "build_prediction_semantics",
    "classify_faithfulness",
    "classify_head_validation",
    "classify_localization",
    "compose_evidence_summary",
    "render_model_diagnostics_page",
    "render_model_diagnostics",
]
