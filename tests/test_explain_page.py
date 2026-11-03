"""Focused tests for the standalone Model Diagnostics page orchestration."""
from __future__ import annotations

from explain_page import (
    build_context_header,
    build_page_view,
    build_prediction_semantics,
    render_model_diagnostics_page,
)


def _context(**updates):
    value = {
        "context_id": "ctx-1",
        "dataset": {
            "name": "MVTec capsule",
            "n_samples": 132,
            "n_classes": 6,
            "fingerprint": "a81c1234567890",
        },
        "model": {"id": "dinov2_vits14", "checkpoint_id": "ckpt-abc"},
        "head": {
            "id": "session-head",
            "scope": "whole_image",
            "pooling": "cls",
            "target_res": 224,
        },
        "image": {
            "filename": "001.png",
            "label": "crack",
            "split": "test",
            "sha256": "deadbeef1234567890",
        },
        "mask_status": "available",
        "normal_bank_status": "unavailable",
        "result_status": "current",
    }
    value.update(updates)
    return value


def _weak_audit():
    return {
        "available": True,
        "kind": "sample_stratified_oof",
        "n_samples": 132,
        "n_splits": 5,
        "balanced_accuracy": 0.501,
        "macro_f1": 0.491,
        "per_class": {
            "crack": {"precision": 0.50, "recall": 0.217, "f1": 0.303, "support": 23},
            "squeeze": {"precision": 0.70, "recall": 0.90, "f1": 0.79, "support": 20},
        },
    }


def _prediction():
    return {
        "actual_label": "crack",
        "current_top1": "squeeze",
        "current_probability": 0.216,
        "runner_up": "crack",
        "runner_up_probability": 0.204,
        "margin": 0.012,
        "calibrated": False,
        "oof": {"available": True, "predicted": "squeeze", "probability": 0.216},
    }


class _FakeStreamlit:
    def __init__(self, *, buttons=None, selected=None):
        self.buttons = dict(buttons or {})
        self.selected = selected
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def _record(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def title(self, *args, **kwargs): self._record("title", *args, **kwargs)
    def info(self, *args, **kwargs): self._record("info", *args, **kwargs)
    def warning(self, *args, **kwargs): self._record("warning", *args, **kwargs)
    def error(self, *args, **kwargs): self._record("error", *args, **kwargs)
    def markdown(self, *args, **kwargs): self._record("markdown", *args, **kwargs)
    def caption(self, *args, **kwargs): self._record("caption", *args, **kwargs)
    def metric(self, *args, **kwargs): self._record("metric", *args, **kwargs)
    def image(self, *args, **kwargs): self._record("image", *args, **kwargs)

    def columns(self, count, **kwargs):
        self._record("columns", count, **kwargs)
        return [self for _ in range(count)]

    def expander(self, *args, **kwargs):
        self._record("expander", *args, **kwargs)
        return self

    def selectbox(self, label, options, *, index=0, **kwargs):
        self._record("selectbox", label, options, index=index, **kwargs)
        return self.selected if self.selected in options else options[index]

    def button(self, label, **kwargs):
        self._record("button", label, **kwargs)
        return bool(self.buttons.get(label, False))

    def slider(self, label, min_value, max_value, value, step, **kwargs):
        self._record("slider", label, min_value, max_value, value, step, **kwargs)
        return value

    def checkbox(self, label, *, value=False, **kwargs):
        self._record("checkbox", label, value=value, **kwargs)
        return value


def _messages(fake, kind):
    return [str(args[0]) for name, args, _ in fake.calls if name == kind and args]


def test_context_header_contains_reference_identity_and_statuses():
    fields = dict(build_context_header(_context()))
    assert fields["Dataset"] == "MVTec capsule · 132 images · 6 classes · fp:a81c123456…"
    assert "dinov2_vits14" in fields["Model"]
    assert "whole_image" in fields["Head"]
    assert "001.png" in fields["Image"]
    assert fields["Mask"] == "available"
    assert fields["Normal Bank"] == "unavailable"
    assert fields["Result"] == "current"


def test_prediction_fields_do_not_mix_label_inference_and_oof():
    semantics = build_prediction_semantics(_context(), _prediction())
    assert semantics["actual_label"] == "crack"
    assert semantics["current_top1"] == "squeeze"
    assert semantics["oof_prediction"] == "squeeze"
    assert semantics["target"] == "squeeze"
    assert semantics["reference"] == "crack"
    assert semantics["calibration_status"] == "uncalibrated probability"
    assert semantics["counterfactual"] is False


def test_missing_oof_is_explicit_and_never_replaced_by_current_prediction():
    prediction = _prediction()
    prediction["oof"] = {"available": False}
    semantics = build_prediction_semantics(_context(), prediction)
    assert semantics["current_top1"] == "squeeze"
    assert semantics["oof_available"] is False
    assert semantics["oof_prediction"] is None


def test_orthogonal_badges_keep_weak_head_smoke_support_and_mask_unverified():
    view = build_page_view(
        _context(),
        head_audit=_weak_audit(),
        prediction=_prediction(),
        faithfulness={"status": "supported"},
        mask_status="available",
    )
    assert view.badges["head"]["state"] == "validation_insufficient"
    assert view.badges["faithfulness"]["state"] == "single_sample_smoke_supported"
    assert view.badges["localization"]["state"] == "unverified"
    assert view.weak_head is True
    assert "不能宣稱模型找到或分割 crack" in view.summary


def test_stale_context_blocks_attribution_instead_of_becoming_weak_head():
    view = build_page_view(
        _context(result_status="stale"),
        head_audit=_weak_audit(),
        prediction=_prediction(),
        result_status="stale",
    )
    assert view.badges["head"]["state"] == "incompatible_or_stale"
    assert view.blocked is True
    assert view.weak_head is False


def test_counterfactual_target_has_persistent_label_and_uses_target_metrics():
    view = build_page_view(
        _context(),
        head_audit=_weak_audit(),
        prediction=_prediction(),
        selected_target="crack",
    )
    assert view.counterfactual is True
    assert view.prediction["target"] == "crack"
    assert view.prediction["reference"] == "squeeze"
    assert view.prediction["margin"] is None
    assert view.audit["target_recall"] == 0.217
    assert "反事實分析" in view.warning


def test_no_context_renders_actionable_empty_state_without_other_sections():
    fake = _FakeStreamlit()
    actions = render_model_diagnostics_page(None, st_module=fake)
    assert actions.rendered_attribution is False
    assert any("Visualize" in message for message in _messages(fake, "info"))
    assert not _messages(fake, "image")


def test_not_ready_head_exposes_preparation_step_without_attribution():
    calls = []

    def display_builder(opacity, show_grid):
        calls.append((opacity, show_grid))
        return {"source_image": "source", "model_input": "input", "overlay": "overlay"}

    fake = _FakeStreamlit()
    actions = render_model_diagnostics_page(
        _context(),
        head_audit={"available": False, "explicit_state": "not_ready", "kind": "unavailable"},
        prediction=_prediction(),
        display_builder=display_builder,
        st_module=fake,
    )
    assert actions.attribution_requested is False
    assert actions.rendered_attribution is False
    assert calls == []
    assert any("尚未建立分類模型" in message for message in _messages(fake, "info"))


def test_not_ready_head_uses_one_combined_prepare_and_analyze_action():
    label = "建立 448×448 模型並分析此圖"
    fake = _FakeStreamlit(buttons={label: True})
    actions = render_model_diagnostics_page(
        _context(),
        head_audit={"available": False, "explicit_state": "not_ready", "kind": "unavailable"},
        prediction=_prediction(),
        primary_action_label=label,
        primary_action_key="diag_analyze",
        st_module=fake,
    )
    assert actions.attribution_requested is True
    button_calls = [
        (args[0], kwargs.get("key"))
        for name, args, kwargs in fake.calls if name == "button"
    ]
    assert button_calls == [(label, "diag_analyze")]


def test_weak_head_can_render_directly_while_warning_stays_visible():
    calls = []

    def display_builder(opacity, show_grid):
        calls.append((opacity, show_grid))
        return {"source_image": "source", "model_input": "input", "overlay": "overlay"}

    fake = _FakeStreamlit()
    actions = render_model_diagnostics_page(
        _context(),
        head_audit=_weak_audit(),
        prediction=_prediction(),
        faithfulness={"status": "inconclusive"},
        mask_status="available",
        display_builder=display_builder,
        st_module=fake,
    )
    assert actions.rendered_attribution is True
    assert actions.research_confirmed is False
    assert calls == [(0.50, False)]
    warnings = _messages(fake, "warning") + _messages(fake, "error")
    assert any("分類 head 證據不足" in message for message in warnings)
    assert not any("我知道這是研究用途" in label
                   for name, args, _ in fake.calls
                   if name == "button" for label in args[:1])


def test_weak_head_direct_action_renders_signed_triptych():
    calls = []

    def display_builder(opacity, show_grid):
        calls.append((opacity, show_grid))
        return {"source_image": "source", "model_input": "input", "overlay": "signed"}

    fake = _FakeStreamlit()
    actions = render_model_diagnostics_page(
        _context(),
        head_audit=_weak_audit(),
        prediction=_prediction(),
        faithfulness={"status": "supported"},
        mask_status="available",
        display_builder=display_builder,
        st_module=fake,
    )
    assert actions.research_confirmed is False
    assert actions.attribution_requested is False
    assert actions.rendered_attribution is True
    assert calls == [(0.50, False)]
    assert len(_messages(fake, "image")) == 3
    warnings = _messages(fake, "warning") + _messages(fake, "error")
    assert any("分類 head 證據不足" in message for message in warnings)
    assert any("不是 crack mask" in message for message in _messages(fake, "caption"))


def test_weak_head_exposes_canonical_button_without_confirmation_step():
    fake = _FakeStreamlit(buttons={"產生分類證據熱圖（Signed Grad-CAM）": True})
    actions = render_model_diagnostics_page(
        _context(),
        head_audit=_weak_audit(),
        prediction=_prediction(),
        faithfulness={"status": "inconclusive"},
        mask_status="available",
        st_module=fake,
    )
    assert actions.research_confirmed is False
    assert actions.attribution_requested is True
    assert actions.rendered_attribution is False
    button_labels = [args[0] for name, args, _ in fake.calls if name == "button"]
    assert button_labels == ["產生分類證據熱圖（Signed Grad-CAM）"]
    assert "我知道這是研究用途，仍要查看熱圖" not in button_labels


def test_counterfactual_targets_live_under_advanced_control_and_remain_warned():
    fake = _FakeStreamlit(selected="crack")
    actions = render_model_diagnostics_page(
        _context(),
        head_audit=_weak_audit(),
        prediction=_prediction(),
        target_options=["squeeze", "crack", "poke"],
        st_module=fake,
    )
    assert actions.selected_target == "crack"
    assert any("進階：反事實 target" in str(args[0]) for name, args, _ in fake.calls if name == "expander")
    warnings = _messages(fake, "warning") + _messages(fake, "error")
    assert any("反事實" in message for message in warnings)


def test_p1_mask_status_does_not_render_mask_image_or_localization_pass():
    fake = _FakeStreamlit()
    render_model_diagnostics_page(
        _context(),
        head_audit=_weak_audit(),
        prediction=_prediction(),
        mask_status="available",
        st_module=fake,
    )
    assert not _messages(fake, "image")
    captions = _messages(fake, "caption")
    assert any("P1 不顯示 mask overlay" in message for message in captions)
    assert all("localization pass" not in message.lower() for message in captions)
