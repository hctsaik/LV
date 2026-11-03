"""Class-similarity panel: charts, threshold preview, evaluation, actions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from class_similarity import AnalysisItem, AnalysisResult, analyze, build_box_summary
from class_similarity_export import evaluation_template, export_report
from class_similarity_records import (
    PreparedInput,
    StaleSelectionError,
    prepare_records,
    selection_original_indices,
    validate_selection,
)
from class_similarity_threshold import (
    ActionRejected,
    PreviewState,
    ThresholdPolicy,
    actions_enabled,
    apply_preview_event,
    axis_display_range,
    candidate_mask,
    current_pr_point,
    evaluate_threshold,
    evaluation_key,
    make_operating_point,
    parse_policy,
    precision_recall_curve,
    preview_key,
    score_protocol_key,
    source_group_id,
    summarize_candidates,
    validate_action,
    validate_evaluation_manifest,
    validate_heldout,
    workload_curve,
)

CS_STATE_KEYS = (
    "cs_status", "cs_prepared", "cs_result", "cs_analysis_key", "cs_fingerprint",
    "cs_local_selected_ids", "cs_inspect_id", "cs_metric", "cs_visible_classes",
    "cs_axis_mode", "cs_draft_policy", "cs_committed_preview", "cs_request_seq",
    "cs_evaluation", "cs_frozen_policy", "cs_table_page", "cs_panel_mode",
    "cs_heldout_used_ids", "cs_purpose", "cs_stale_reason", "cs_last_num",
    "cs_preview_state", "cs_compute_count", "cs_confusion_filter",
    "cs_select_gen", "cs_only_candidates", "cs_only_neg_margin",
)
PANEL_SIM = "類別相似度分布"
PANEL_CONF = "類別混淆"
PANEL_BAD = "疑似標錯"


def sample_scatter_rows(items: Sequence[AnalysisItem], metric: str, analysis_key: str,
                        per_class_cap: int = 200, margin_keep: int = 20) -> list[AnalysisItem]:
    groups: dict[str, list[AnalysisItem]] = {}
    for it in items:
        val = getattr(it, metric, None)
        if val is None or not np.isfinite(val):
            continue
        groups.setdefault(it.label or "未分類／標籤無效", []).append(it)
    chosen: list[AnalysisItem] = []
    for lab in sorted(groups):
        rows = groups[lab]
        if len(rows) <= per_class_cap:
            chosen.extend(rows)
            continue
        with_m = [r for r in rows if r.margin is not None and np.isfinite(r.margin)]
        with_m.sort(key=lambda r: (r.margin, r.record_id))
        keep = list(with_m[:margin_keep])
        if len(keep) < margin_keep:
            rest = sorted(rows, key=lambda r: (getattr(r, metric), r.record_id))
            seen = {r.record_id for r in keep}
            for r in rest:
                if r.record_id not in seen:
                    keep.append(r)
                    seen.add(r.record_id)
                if len(keep) >= margin_keep:
                    break
        seen = {r.record_id for r in keep}
        remaining = [r for r in rows if r.record_id not in seen]
        remaining.sort(key=lambda r: hashlib.sha256(
            f"{analysis_key}{r.record_id}".encode("utf-8")).hexdigest())
        keep.extend(remaining[:max(0, per_class_cap - len(keep))])
        chosen.extend(keep)
    return chosen


def jitter_x(analysis_key: str, record_id: str, scale: float = 0.18) -> float:
    h = hashlib.sha256(f"{analysis_key}{record_id}".encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "big") / float(2 ** 64)
    return (u - 0.5) * 2.0 * scale


def sort_table_rows(items: Sequence[AnalysisItem], *, only_neg=False,
                    only_candidates=False, mask=None) -> list[AnalysisItem]:
    rows = list(items)
    if only_neg:
        rows = [r for r in rows if r.margin is not None and r.margin < 0]
    if only_candidates and mask is not None:
        rows = [r for r, flag in zip(items, mask) if flag]
        if only_neg:
            rows = [r for r in rows if r.margin is not None and r.margin < 0]
    def key(it: AnalysisItem):
        null = it.margin is None or not np.isfinite(it.margin)
        return (1 if null else 0, float(it.margin) if not null else 0.0, it.record_id)
    return sorted(rows, key=key)


def local_selected_valid(ids: Sequence[str], prepared: PreparedInput) -> list[str]:
    ok = {rid for rid, elig in zip(prepared.record_ids, prepared.eligibility)
          if elig and not str(rid).startswith("invalid:")}
    return [i for i in ids if i in ok]


def threshold_slider(value: float, key: str):
    import streamlit.components.v1 as components
    component = components.declare_component(
        "cs_threshold_slider",
        path=str(Path(__file__).with_name("class_similarity_slider")),
    )
    return component(value=float(value), key=key, default=None)


def _st():
    import streamlit as st
    return st


def _status(st) -> str:
    return st.session_state.get("cs_status") or "empty"


def render_panel(records, raw, model, data_token, context: dict) -> dict | None:
    """Streamlit panel. Returns an action dict or None."""
    st = _st()
    gen = int(st.session_state.get("cs_widget_generation") or 0)
    action = None

    k = int(st.session_state.get("cs_k", context.get("k") or 5))
    splits = sorted({str(r.get("split") or "") for r in records if r.get("split")})
    split_opts = ["all"] + [s for s in splits if s]
    analysis_split = st.selectbox("分析 split", split_opts, key=f"cs_split_{gen}")
    k = st.number_input("K（Top-K 平均）", min_value=1, max_value=20, value=int(k),
                        step=1, key=f"cs_k_{gen}")
    st.caption(f"分析母體：split={analysis_split} · 模型={model} · 原始高維特徵 D="
               f"{int(np.asarray(raw).shape[1]) if raw is not None else '?'}")

    ctx = dict(context)
    ctx.update({
        "data_token": data_token, "model": model, "k": int(k),
        "analysis_split": analysis_split, "event_generation": gen,
    })

    status = _status(st)
    if st.button("計算類別相似度", key=f"cs_compute_{gen}", use_container_width=True):
        st.session_state["cs_status"] = "computing"
        st.session_state["cs_compute_count"] = int(st.session_state.get("cs_compute_count") or 0) + 1
        try:
            prepared = prepare_records(records, raw, ctx)
            if prepared.source_status == "source_stale":
                st.session_state["cs_status"] = "stale"
                st.session_state["cs_stale_reason"] = "來源內容與 embedding generation 不一致"
                st.session_state["cs_prepared"] = prepared
                st.rerun()
            result = analyze(
                prepared.embeddings, prepared.labels, prepared.record_ids,
                eligibility=prepared.eligibility,
                invalid_reasons=prepared.invalid_reasons, k=int(k),
            )
            st.session_state["cs_prepared"] = prepared
            st.session_state["cs_result"] = result
            st.session_state["cs_analysis_key"] = prepared.analysis_key
            st.session_state["cs_fingerprint"] = prepared.scope_fp
            st.session_state["cs_status"] = "ready"
            st.session_state["cs_draft_policy"] = ThresholdPolicy()
            st.session_state["cs_committed_preview"] = {
                "policy": ThresholdPolicy().__dict__,
                "preview_key": preview_key(prepared.analysis_key, ThresholdPolicy(), None),
                "seq": 0, "mask": candidate_mask(result.items, ThresholdPolicy()).tolist(),
            }
            st.session_state["cs_preview_state"] = PreviewState(
                draft_policy=ThresholdPolicy(), committed_policy=ThresholdPolicy(),
                request_seq=0, committed_seq=0, analysis_key=prepared.analysis_key,
                preview_key_committed=preview_key(prepared.analysis_key, ThresholdPolicy(), None),
                evaluation_key=None,
            )
            st.session_state["cs_request_seq"] = 0
            st.session_state["cs_local_selected_ids"] = []
            st.session_state["cs_inspect_id"] = None
        except Exception as exc:
            st.session_state["cs_status"] = "error"
            st.session_state["cs_stale_reason"] = str(exc)
        st.rerun()

    prepared: PreparedInput | None = st.session_state.get("cs_prepared")
    result: AnalysisResult | None = st.session_state.get("cs_result")
    status = _status(st)

    if prepared is not None and status == "ready":
        try:
            now = prepare_records(records, raw, ctx)
            if now.scope_fp != prepared.scope_fp or now.analysis_key != prepared.analysis_key:
                st.session_state["cs_status"] = "stale"
                st.session_state["cs_stale_reason"] = "資料／標籤／K／split／embedding 已變更"
                status = "stale"
            if now.source_status == "source_stale":
                st.session_state["cs_status"] = "stale"
                status = "stale"
        except Exception as exc:
            st.session_state["cs_status"] = "error"
            st.session_state["cs_stale_reason"] = str(exc)
            status = "error"

    if status in {"empty", "computing"}:
        st.info("按「計算類別相似度」後，用目前模型的原始高維特徵算 Top-K 平均 cosine。")
        return None
    if status == "error":
        st.error(st.session_state.get("cs_stale_reason") or "計算失敗")
        return None
    if status == "stale":
        st.warning("結果已過期：" + str(st.session_state.get("cs_stale_reason") or "請重新計算"))
        st.caption("過期結果不能選取或匯出。")
        return None
    if prepared is None or result is None:
        return None

    if prepared.source_status == "source_unverified":
        st.caption("與 embedding 的一致性未驗證（缺 generation 來源證明）。數值可探索，但不可用於 PR。")

    metric = st.radio("指標", ["margin", "same", "other"], horizontal=True,
                      key=f"cs_metric_{gen}",
                      format_func=lambda m: {
                          "margin": "margin（same − other）",
                          "same": "同類 Top-K 平均 cosine",
                          "other": "最接近異類 Top-K 平均 cosine",
                      }[m])
    classes = sorted({it.label for it in result.items if it.label})
    visible = st.multiselect("顯示類別（最多 10，不改分析母體）", classes,
                             default=classes[:10], key=f"cs_vis_{gen}",
                             max_selections=10)
    axis_mode = st.radio("顯示範圍", ["完整範圍", "符合資料範圍"], horizontal=True,
                         key=f"cs_axis_{gen}")
    if st.button("重設完整範圍", key=f"cs_axis_reset_{gen}"):
        st.session_state[f"cs_axis_{gen}"] = "完整範圍"
        st.rerun()

    purpose = st.session_state.get("cs_purpose") or "tuning"
    evaluation = st.session_state.get("cs_evaluation")
    eval_key = None
    if evaluation:
        eval_key = evaluation_key(prepared.analysis_key, evaluation.get("truth_digest"),
                                  evaluation.get("purpose") or "tuning",
                                  (st.session_state.get("cs_frozen_policy") or {}).get("policy_id"))

    heldout_locked = purpose == "heldout" and st.session_state.get("cs_frozen_policy")
    _render_threshold_controls(st, prepared, result, gen, eval_key, heldout_locked)
    pstate: PreviewState = st.session_state.get("cs_preview_state")
    draft = pstate.draft_policy if pstate else parse_policy({"mode": "threshold", "threshold": 0.0})
    enabled = bool(pstate and actions_enabled(pstate))
    # Preview must follow the draft policy while the pointer is held (AC18).
    # Actions stay gated on commit via `enabled` / validate_action.
    mask = candidate_mask(result.items, draft)
    summary = summarize_candidates(result.items, mask)
    _render_summary_cards(st, summary, result, prepared, evaluation, mask, enabled)

    shown_items = [it for it in result.items if not visible or it.label in visible]
    box_items = [it for it in result.items if not visible or it.label in visible]
    # box still uses full class population for visible classes, not scatter sample
    fig = _build_figure(result, prepared, metric, visible or classes, box_items,
                        sample_scatter_rows(shown_items, metric, prepared.analysis_key),
                        axis_mode, draft if draft.mode == "threshold" else None,
                        mask)
    event = st.plotly_chart(fig, use_container_width=True, key=f"cs_fig_{gen}_{metric}",
                            on_select="rerun", selection_mode=("points", "box", "lasso"))
    action = _consume_plot_event(st, event, prepared, result, gen) or action

    _render_workload_and_pr(st, result, evaluation, draft, prepared, purpose)
    action = _render_table_and_evidence(st, prepared, result, records, mask, metric,
                                        visible, enabled, gen, context) or action
    action = _render_evaluation_and_ops(st, prepared, result, records, mask, enabled,
                                        gen, context) or action
    action = _render_actions(st, prepared, result, records, mask, enabled, gen, context) or action
    return action


def _render_threshold_controls(st, prepared, result, gen, eval_key, heldout_locked):
    pstate: PreviewState | None = st.session_state.get("cs_preview_state")
    if pstate is None:
        pol = ThresholdPolicy()
        pstate = PreviewState(pol, pol, 0, 0, prepared.analysis_key,
                              preview_key(prepared.analysis_key, pol, eval_key), eval_key)
        st.session_state["cs_preview_state"] = pstate
    canonical = 0.0 if pstate.draft_policy.threshold is None else float(pstate.draft_policy.threshold)
    cols = st.columns([3, 2, 1, 1, 1])
    with cols[0]:
        if heldout_locked:
            st.caption("heldout：門檻已凍結，滑桿停用")
            ev = None
        else:
            ev = threshold_slider(canonical, key=f"cs_slider_{gen}")
        if ev and not heldout_locked:
            kind = ev.get("kind") or "preview"
            nxt = pstate.request_seq + 1
            try:
                st.session_state["cs_preview_state"] = apply_preview_event(
                    pstate,
                    {"kind": kind, "analysis_key": prepared.analysis_key,
                     "request_seq": nxt,
                     "policy": {"mode": "threshold", "threshold": float(ev["value"])}},
                    current_analysis_key=prepared.analysis_key,
                    current_eval_key=eval_key,
                    draft_policy=pstate.draft_policy,
                )
                st.session_state["cs_request_seq"] = st.session_state["cs_preview_state"].request_seq
                if kind == "commit":
                    st.session_state[f"cs_num_{gen}"] = float(ev["value"])
                    st.session_state["cs_last_num"] = float(ev["value"])
            except ActionRejected:
                pass
    with cols[1]:
        num_key = f"cs_num_{gen}"
        if num_key not in st.session_state:
            st.session_state[num_key] = canonical
        if st.session_state.get("cs_last_num") is None:
            st.session_state["cs_last_num"] = float(st.session_state[num_key])
        disabled = bool(heldout_locked)
        num = st.number_input("精確門檻", min_value=-2.0, max_value=2.0, step=1e-6,
                              format="%.6f", key=num_key, disabled=disabled)
        if not disabled and num != st.session_state.get("cs_last_num"):
            st.session_state["cs_last_num"] = num
            pstate = st.session_state["cs_preview_state"]
            nxt = pstate.request_seq + 1
            st.session_state["cs_preview_state"] = apply_preview_event(
                pstate,
                {"kind": "commit", "analysis_key": prepared.analysis_key,
                 "request_seq": nxt,
                 "policy": {"mode": "threshold", "threshold": float(num)}},
                current_analysis_key=prepared.analysis_key,
                current_eval_key=eval_key, draft_policy=pstate.draft_policy,
            )
    with cols[2]:
        if st.button("恢復預設 t=0", key=f"cs_reset_t_{gen}", disabled=heldout_locked):
            pstate = st.session_state["cs_preview_state"]
            nxt = pstate.request_seq + 1
            st.session_state["cs_preview_state"] = apply_preview_event(
                pstate, {"kind": "commit", "analysis_key": prepared.analysis_key,
                         "request_seq": nxt, "policy": {"mode": "threshold", "threshold": 0.0}},
                current_analysis_key=prepared.analysis_key, current_eval_key=eval_key,
                draft_policy=pstate.draft_policy)
            st.session_state[num_key] = 0.0
            st.rerun()
    with cols[3]:
        if st.button("無候選", key=f"cs_none_{gen}", disabled=heldout_locked):
            pstate = st.session_state["cs_preview_state"]
            nxt = pstate.request_seq + 1
            st.session_state["cs_preview_state"] = apply_preview_event(
                pstate, {"kind": "commit", "analysis_key": prepared.analysis_key,
                         "request_seq": nxt, "policy": {"mode": "none"}},
                current_analysis_key=prepared.analysis_key, current_eval_key=eval_key,
                draft_policy=pstate.draft_policy)
            st.rerun()
    with cols[4]:
        if st.button("全部可計分", key=f"cs_all_{gen}", disabled=heldout_locked):
            pstate = st.session_state["cs_preview_state"]
            nxt = pstate.request_seq + 1
            st.session_state["cs_preview_state"] = apply_preview_event(
                pstate, {"kind": "commit", "analysis_key": prepared.analysis_key,
                         "request_seq": nxt, "policy": {"mode": "all_eligible"}},
                current_analysis_key=prepared.analysis_key, current_eval_key=eval_key,
                draft_policy=pstate.draft_policy)
            st.rerun()
    pstate = st.session_state["cs_preview_state"]
    if not actions_enabled(pstate):
        st.caption("門檻更新中：加入候選／保存／下載已暫停，請等提交完成。")


def _render_summary_cards(st, summary, result, prepared, evaluation, mask, enabled):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("候選 / 可計分", f"{summary.candidate_n}/{summary.eligible_n}")
    ratio = "—" if summary.ratio is None else f"{summary.ratio:.1%}"
    c2.metric("候選比例", ratio)
    c3.metric("分析總數", f"{summary.input_count}")
    c4.metric("無法計分", f"{summary.unscorable_n}")
    pstate = st.session_state.get("cs_preview_state")
    draft = pstate.draft_policy if pstate else None
    if draft is not None and draft.mode == "threshold" and draft.threshold is not None:
        pending = "" if enabled else "（預覽未提交）"
        st.caption(f"目前門檻 t={float(draft.threshold):.6f}{pending}")
    st.caption("這個門檻決定哪些資料先請人檢查。Precision／Recall 衡量找出誤標的能力，不是分類模型準確度。")
    if evaluation:
        counts = evaluate_threshold(result.items, mask, evaluation,
                                    source_verified=prepared.source_verified)
        if counts.status == "ready":
            p = "—" if counts.precision is None else f"{counts.precision:.3f}"
            r = "—" if counts.recall is None else f"{counts.recall:.3f}"
            st.markdown(f"**此評估集可計分資料上的 Precision／Recall** = {p} / {r}")
            d1, d2, d3, d4 = st.columns(4)
            if d1.button(f"TP {counts.tp}", key="cs_drill_tp"):
                st.session_state["cs_confusion_filter"] = "tp"
            if d2.button(f"FP {counts.fp}", key="cs_drill_fp"):
                st.session_state["cs_confusion_filter"] = "fp"
            if d3.button(f"FN {counts.fn}", key="cs_drill_fn"):
                st.session_state["cs_confusion_filter"] = "fn"
            if d4.button(f"TN {counts.tn}", key="cs_drill_tn"):
                st.session_state["cs_confusion_filter"] = "tn"
        else:
            st.info(f"PR 尚不可用：{counts.unavailable_reason}；E0={counts.e0_n} E={counts.e_n} "
                    f"coverage={counts.coverage}")
    with st.expander("每類候選摘要", expanded=False):
        st.dataframe(summary.per_class, hide_index=True, use_container_width=True)


def _build_figure(result, prepared, metric, visible, box_items, scatter_rows,
                  axis_mode, policy, mask):
    import plotly.graph_objects as go
    fig = go.Figure()
    summaries = {row["label"]: row for row in build_box_summary(result.items, metric)}
    for lab in visible:
        stats = summaries.get(lab)
        if not stats:
            continue
        fig.add_trace(go.Box(
            x=[lab], q1=[stats.get("q1")], median=[stats.get("median")],
            q3=[stats.get("q3")], lowerfence=[stats.get("whisker_low")],
            upperfence=[stats.get("whisker_high")], name=lab,
            boxpoints=False, hoverinfo="skip",
        ))
    xs, ys, custom, colors = [], [], [], []
    mask_by_id = {result.items[i].record_id: bool(mask[i]) for i in range(len(result.items))}
    for it in scatter_rows:
        if it.label not in visible:
            continue
        xs.append(it.label)
        ys.append(float(getattr(it, metric)))
        orig = prepared.original_index[it.input_index]
        custom.append([prepared.analysis_key, it.record_id, orig])
        colors.append("#d35400" if mask_by_id.get(it.record_id) else "#2980b9")
    jittered = [f"{x}" for x in xs]  # categorical; jitter via dummy numeric overlay
    # Use numeric class index + jitter so points spread
    lab_index = {lab: i for i, lab in enumerate(visible)}
    xj = [lab_index[it.label] + jitter_x(prepared.analysis_key, it.record_id)
          for it in scatter_rows if it.label in visible]
    fig.add_trace(go.Scatter(
        x=xj, y=ys, mode="markers", name="records",
        marker=dict(color=colors, size=8),
        customdata=custom,
        hovertemplate="%{customdata[1]}<br>y=%{y:.4f}<extra></extra>",
    ))
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(visible))), ticktext=visible)
    vals = [getattr(it, metric) for it in result.items
            if it.label in visible and getattr(it, metric) is not None]
    t = None if policy is None or policy.mode != "threshold" else policy.threshold
    rng = axis_display_range(metric, "data" if axis_mode == "符合資料範圍" else "full",
                             [v for v in vals if v is not None], t)
    fig.update_yaxes(range=[rng["ymin"], rng["ymax"]])
    if metric == "margin":
        fig.add_hline(y=0, line_dash="dot", line_color="#888")
        if t is not None and policy and policy.mode == "threshold":
            fig.add_hline(y=t, line_color="#c0392b",
                          annotation_text=f"t={t:.6f}")
            if rng["threshold_outside"]:
                fig.add_annotation(text=f"門檻 {t:.6f} 在視窗外", xref="paper", yref="paper",
                                   x=1, y=1, showarrow=False, font=dict(color="#c0392b"))
    n_scatter = len(ys)
    n_metric = sum(1 for it in result.items if it.label in visible
                   and getattr(it, metric) is not None and np.isfinite(getattr(it, metric)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      title=f"box 使用全量；散點顯示 {n_scatter}/{n_metric}",
                      showlegend=False)
    return fig


def _consume_plot_event(st, event, prepared, result, gen) -> dict | None:
    if event is None:
        return None
    sel = event.get("selection") if hasattr(event, "get") else None
    if not sel:
        return None
    points = list(sel.get("points") or [])
    box = sel.get("box") or sel.get("lasso")
    ids = []
    for p in points:
        cd = p.get("customdata")
        if not cd:
            continue
        try:
            validate_selection(
                {"analysis_key": cd[0], "record_id": cd[1], "original_index": int(cd[2]),
                 "event_generation": gen},
                result, prepared)
        except StaleSelectionError:
            continue
        ids.append(str(cd[1]))
    if not ids:
        return None
    if box:
        cur = list(st.session_state.get("cs_local_selected_ids") or [])
        for i in ids:
            if i not in cur:
                cur.append(i)
        st.session_state["cs_local_selected_ids"] = local_selected_valid(cur, prepared)
    else:
        st.session_state["cs_inspect_id"] = ids[0]
    return None


def _render_workload_and_pr(st, result, evaluation, draft, prepared, purpose):
    import plotly.graph_objects as go
    wl = workload_curve(result.items)
    fig = go.Figure(go.Scatter(x=[i for i, _ in enumerate(wl)],
                               y=[p["candidate_n"] for p in wl],
                               mode="lines+markers",
                               text=[p.get("label") or p.get("mode") for p in wl]))
    fig.update_layout(height=220, title="工作量曲線（threshold → 候選數）",
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True, key="cs_workload")
    if evaluation and purpose != "heldout" and prepared.source_verified:
        pr = precision_recall_curve(result.items, evaluation,
                                    source_verified=prepared.source_verified)
        if pr:
            fig2 = go.Figure(go.Scatter(
                x=[p["recall"] for p in pr], y=[p["precision"] for p in pr],
                mode="lines+markers", name="操作點"))
            cur = current_pr_point(result.items, evaluation, draft,
                                   source_verified=prepared.source_verified)
            if cur["precision"] is not None and cur["recall"] is not None:
                fig2.add_trace(go.Scatter(x=[cur["recall"]], y=[cur["precision"]],
                                          mode="markers", marker=dict(size=12, color="#c0392b"),
                                          name="目前"))
            fig2.update_layout(height=220, title="PR 曲線（找出誤標）",
                               xaxis_title="Recall", yaxis_title="Precision",
                               margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig2, use_container_width=True, key="cs_pr")


def _render_table_and_evidence(st, prepared, result, records, mask, metric, visible,
                               enabled, gen, context) -> dict | None:
    only_cand = st.checkbox("僅候選", key=f"cs_only_cand_{gen}")
    only_neg = st.checkbox("只看 margin<0", key=f"cs_only_neg_{gen}")
    rows = sort_table_rows(result.items, only_neg=only_neg, only_candidates=only_cand, mask=mask)
    drill = st.session_state.get("cs_confusion_filter")
    evaluation = st.session_state.get("cs_evaluation")
    if drill and evaluation:
        counts = evaluate_threshold(result.items, mask, evaluation,
                                    source_verified=prepared.source_verified)
        want = set((counts.confusion_ids or {}).get(drill) or [])
        if want:
            rows = [r for r in rows if r.record_id in want]
    page_size = 50
    n_pages = max(1, (len(rows) + page_size - 1) // page_size)
    page = int(st.number_input("表格頁", min_value=1, max_value=n_pages, value=1,
                               key=f"cs_page_{gen}"))
    page_rows = rows[(page - 1) * page_size: page * page_size]
    local = set(st.session_state.get("cs_local_selected_ids") or [])
    select_gen = int(st.session_state.get("cs_select_gen") or 0)
    for it in page_rows:
        cols = st.columns([0.4, 2, 1, 1, 1, 1, 2])
        key = f"cs_cb_{select_gen}_{it.record_id}"
        if key not in st.session_state:
            st.session_state[key] = it.record_id in local
        checked = cols[0].checkbox("選", key=key, label_visibility="collapsed",
                                   disabled=it.record_id.startswith("invalid:"))
        if checked and it.record_id not in local and not it.record_id.startswith("invalid:"):
            local.add(it.record_id)
        if not checked and it.record_id in local:
            local.discard(it.record_id)
        cols[1].write(it.record_id[:12] + "…")
        cols[2].write("" if it.same is None else f"{it.same:.4f}")
        cols[3].write(it.rival_class or "")
        cols[4].write("" if it.other is None else f"{it.other:.4f}")
        cols[5].write("" if it.margin is None else f"{it.margin:.4f}")
        if cols[6].button("證據", key=f"cs_ev_{it.record_id}"):
            st.session_state["cs_inspect_id"] = it.record_id
    st.session_state["cs_local_selected_ids"] = local_selected_valid(list(local), prepared)
    inspect = st.session_state.get("cs_inspect_id")
    if inspect:
        _render_evidence(st, inspect, prepared, result, records, context)
    return None


def _render_evidence(st, record_id, prepared, result, records, context):
    by = {it.record_id: it for it in result.items}
    it = by.get(record_id)
    if not it:
        return
    rec = prepared.records[it.input_index]
    st.markdown(f"**證據** `{record_id}` · 現標 {it.label} · same="
                f"{None if it.same is None else round(it.same, 4)} · other="
                f"{None if it.other is None else round(it.other, 4)} · margin="
                f"{None if it.margin is None else round(it.margin, 4)}")
    if it.tied_rival_count > 1:
        st.caption(f"多個異類同分（{it.tied_rival_count}），僅列前 10：{it.tied_rival_classes}")
    thumb_fn = context.get("thumb_fn")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Query**")
        _show_image(st, rec, thumb_fn, prepared.source_verified)
    with cols[1]:
        st.markdown("**來源原圖／bbox**")
        src = rec.get("image_path") or rec.get("path")
        _show_image(st, {"path": src, "bbox": rec.get("bbox")}, thumb_fn, prepared.source_verified)
    n1, n2 = st.columns(2)
    with n1:
        st.markdown("**同類 Top-K**")
        for n in it.same_neighbors:
            nb = by.get(n.record_id)
            st.caption(f"{n.record_id[:10]} cos={n.cosine:.4f} {nb.label if nb else ''}")
            if nb:
                _show_image(st, prepared.records[nb.input_index], thumb_fn, prepared.source_verified)
    with n2:
        st.markdown(f"**最接近異類 Top-K（{it.rival_class}）**")
        for n in it.rival_neighbors:
            nb = by.get(n.record_id)
            st.caption(f"{n.record_id[:10]} cos={n.cosine:.4f} {nb.label if nb else ''}")
            if nb:
                _show_image(st, prepared.records[nb.input_index], thumb_fn, prepared.source_verified)


def _show_image(st, rec, thumb_fn, verified):
    path = rec.get("path")
    caption = "" if verified else "與 embedding 的一致性未驗證"
    if path is None:
        st.caption("缺圖")
        return
    p = Path(path)
    if not p.exists():
        st.caption(f"缺檔：{p.name}")
        return
    img = None
    if thumb_fn is not None:
        try:
            img = thumb_fn(p)
        except Exception:
            img = None
    try:
        st.image(str(img or p), caption=caption, use_container_width=True)
    except Exception:
        st.caption(f"缺檔：{p.name}")


def _render_evaluation_and_ops(st, prepared, result, records, mask, enabled, gen, context):
    st.markdown("##### 人工評估")
    scope = st.radio("建立評估模板範圍", ["目前分析母體 U 全部", "本面板勾選"],
                     key=f"cs_tmpl_scope_{gen}")
    if st.button("建立評估模板", key=f"cs_mk_tmpl_{gen}"):
        if scope.endswith("勾選"):
            ids = list(st.session_state.get("cs_local_selected_ids") or [])
            method = "本面板勾選（人工選樣，不代表隨機或全資料）"
        else:
            ids = [rid for rid, elig in zip(prepared.record_ids, prepared.eligibility) if elig]
            method = "分析母體 U 全部"
        blob = evaluation_template(
            frozen_ids=ids, records=prepared.records, record_ids=prepared.record_ids,
            labels=prepared.effective_label, snapshot_key=prepared.source_snapshot_key,
            selection_method=method, evaluation_id=f"eval-{prepared.analysis_key[:8]}",
            level=context.get("level") or "image")
        st.session_state["cs_template_bytes"] = blob
    if st.session_state.get("cs_template_bytes"):
        st.download_button("下載評估模板 JSON", data=st.session_state["cs_template_bytes"],
                           file_name="class_similarity_evaluation.json", mime="application/json",
                           key=f"cs_dl_tmpl_{gen}")
    up = st.file_uploader("匯入評估 JSON", type=["json"], key=f"cs_up_{gen}")
    if up is not None:
        try:
            manifest = json.loads(up.getvalue().decode("utf-8"))
            snap = {"snapshot_key": prepared.source_snapshot_key,
                    "original_labels": {rid: lab for rid, lab in zip(
                        prepared.record_ids, prepared.effective_label)}}
            ev = validate_evaluation_manifest(manifest, snap, result.items)
            st.session_state["cs_evaluation"] = ev
            st.session_state["cs_purpose"] = ev.get("purpose") or "tuning"
            st.success("已匯入評估集")
        except Exception as exc:
            st.error(str(exc))
    if st.session_state.get("cs_evaluation"):
        ev = st.session_state["cs_evaluation"]
        pending = [r for r in ev.get("reviews") or [] if r.get("verdict") == "unreviewed"]
        st.caption(f"待確認 {len(pending)} / {len(ev.get('frozen_ids') or [])}")
        for r in pending[:20]:
            if st.button(f"看 {r['record_id'][:10]}", key=f"cs_pending_{r['record_id']}"):
                st.session_state["cs_inspect_id"] = r["record_id"]
    frozen = st.session_state.get("cs_frozen_policy")
    c1, c2, c3 = st.columns(3)
    pstate = st.session_state.get("cs_preview_state")
    with c1:
        if st.button("保存目前操作點", key=f"cs_save_op_{gen}", disabled=not enabled):
            try:
                validate_action({
                    "analysis_key": prepared.analysis_key,
                    "preview_key": pstate.preview_key_committed,
                    "request_seq": pstate.committed_seq,
                }, pstate, prepared.analysis_key)
            except ActionRejected as exc:
                st.error(str(exc))
            else:
                proto = score_protocol_key(
                    method_version="class_topk_margin_v1", k=prepared.k,
                    model=prepared.model, weights_id="unknown",
                    recipe=prepared.embedding_recipe, analysis_split=prepared.analysis_split)
                ev = st.session_state.get("cs_evaluation") or {}
                recs = prepared.records
                level = context.get("level") or "image"
                ids = list(ev.get("frozen_ids") or [])
                groups = [source_group_id(recs[prepared.record_ids.index(i)], level)
                          for i in ids if i in prepared.record_ids]
                op = make_operating_point(
                    policy=pstate.committed_policy, analysis_key=prepared.analysis_key,
                    score_protocol_key=proto, tuning_evaluation_key=ev.get("truth_digest"),
                    tuning_ids=ids, tuning_group_ids=groups, policy_id=pstate.preview_key_committed[:12],
                    created_at="now")
                st.session_state["cs_frozen_policy"] = op
                st.success(f"已保存 t={op['policy']['threshold']}")
    with c2:
        if st.button("恢復已保存操作點", key=f"cs_restore_op_{gen}", disabled=not frozen):
            op = frozen
            if op.get("analysis_key") != prepared.analysis_key:
                st.error("操作點與目前 analysis_key 不相容")
            else:
                nxt = pstate.request_seq + 1
                st.session_state["cs_preview_state"] = apply_preview_event(
                    pstate, {"kind": "commit", "analysis_key": prepared.analysis_key,
                             "request_seq": nxt, "policy": op["policy"]},
                    current_analysis_key=prepared.analysis_key, current_eval_key=None,
                    draft_policy=pstate.draft_policy)
                st.rerun()
    with c3:
        if st.session_state.get("cs_purpose") == "heldout":
            if st.button("轉為 tuning", key=f"cs_to_tuning_{gen}"):
                used = set(st.session_state.get("cs_heldout_used_ids") or [])
                ev = st.session_state.get("cs_evaluation") or {}
                used |= set(ev.get("frozen_ids") or [])
                st.session_state["cs_heldout_used_ids"] = list(used)
                st.session_state["cs_purpose"] = "tuning"
                st.rerun()
        elif frozen:
            if st.button("以保存點做 heldout", key=f"cs_to_heldout_{gen}"):
                ev = st.session_state.get("cs_evaluation") or {}
                level = context.get("level") or "image"
                ids = list(ev.get("frozen_ids") or [])
                groups = [source_group_id(prepared.records[prepared.record_ids.index(i)], level)
                          for i in ids if i in prepared.record_ids]
                proto = score_protocol_key(
                    method_version="class_topk_margin_v1", k=prepared.k,
                    model=prepared.model, weights_id="unknown",
                    recipe=prepared.embedding_recipe, analysis_split=prepared.analysis_split)
                try:
                    validate_heldout(frozen, analysis_key=prepared.analysis_key,
                                     score_protocol_key=proto, heldout_ids=ids,
                                     heldout_groups=groups, tuning_complete=True)
                    already = set(st.session_state.get("cs_heldout_used_ids") or [])
                    if set(ids) & already:
                        raise Exception("這些 heldout ID 已在本 session 看過")
                    st.session_state["cs_purpose"] = "heldout"
                except Exception as exc:
                    st.error(str(exc))
    return None


def _render_actions(st, prepared, result, records, mask, enabled, gen, context) -> dict | None:
    action = None
    pstate: PreviewState = st.session_state.get("cs_preview_state")
    local = list(st.session_state.get("cs_local_selected_ids") or [])
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("同步至 LV 選取", key=f"cs_sync_{gen}"):
            idxs = selection_original_indices(prepared, local)
            action = {"type": "sync_selection", "ids": idxs,
                      "analysis_key": prepared.analysis_key}
    with c2:
        if st.button("將全部候選加入 LV 選取（含未顯示類別）",
                     key=f"cs_add_cand_{gen}", disabled=not enabled):
            try:
                validate_action({
                    "analysis_key": prepared.analysis_key,
                    "preview_key": pstate.preview_key_committed,
                    "request_seq": pstate.committed_seq,
                }, pstate, prepared.analysis_key)
                ids = [prepared.original_index[i] for i, flag in enumerate(mask)
                       if flag and not result.items[i].record_id.startswith("invalid:")]
                action = {"type": "add_candidates_to_selection", "ids": ids,
                          "analysis_key": prepared.analysis_key,
                          "preview_key": pstate.preview_key_committed,
                          "request_seq": pstate.committed_seq}
            except ActionRejected as exc:
                st.error(str(exc))
    with c3:
        if st.button("加入購物車", key=f"cs_cart_{gen}"):
            idxs = selection_original_indices(prepared, local)
            scores = {}
            for i, it in enumerate(result.items):
                orig = prepared.original_index[i]
                if orig in idxs:
                    scores[orig] = it.margin
            action = {"type": "add_to_cart", "ids": idxs, "scores": scores,
                      "analysis_key": prepared.analysis_key,
                      "reason": f"class_topk_margin_v1 {prepared.analysis_key[:8]}"}
    with c4:
        if st.button("清除本面板勾選", key=f"cs_clear_local_{gen}"):
            st.session_state["cs_local_selected_ids"] = []
            st.session_state["cs_inspect_id"] = None
            st.session_state["cs_select_gen"] = int(st.session_state.get("cs_select_gen") or 0) + 1
            st.rerun()
    scopes = st.selectbox("下載範圍", ["all", "candidates", "selected"], key=f"cs_scope_{gen}")
    can_dl = enabled if scopes != "all" else enabled or scopes == "all"
    # all also requires committed when policy fields are included; disable if not committed
    if not enabled:
        can_dl = False
        st.caption("預覽尚未提交，下載停用。")
    if st.button("下載本面板已勾選" if scopes == "selected" else f"下載 {scopes}",
                 key=f"cs_dl_{gen}", disabled=not can_dl):
        try:
            validate_action({
                "analysis_key": prepared.analysis_key,
                "preview_key": pstate.preview_key_committed,
                "request_seq": pstate.committed_seq,
            }, pstate, prepared.analysis_key)
            ev = st.session_state.get("cs_evaluation")
            counts = evaluate_threshold(result.items, mask, ev,
                                        source_verified=prepared.source_verified) if ev else None
            csv_b, js_b = export_report(
                result, prepared.records, scope=scopes, local_selected_ids=local,
                committed_preview=pstate.committed_policy, evaluation=ev,
                evaluation_counts=counts, prepared=prepared,
                level=context.get("level") or "image",
                analysis_key=prepared.analysis_key,
                preview_key=pstate.preview_key_committed,
                frozen_policy=st.session_state.get("cs_frozen_policy"),
                label_status=prepared.label_status,
            )
            action = {"type": "download", "scope": scopes, "csv": csv_b, "json": js_b,
                      "analysis_key": prepared.analysis_key,
                      "preview_key": pstate.preview_key_committed,
                      "request_seq": pstate.committed_seq}
        except ActionRejected as exc:
            st.error(str(exc))
    if action and action.get("type") == "download":
        st.download_button("⬇ CSV", data=action["csv"], file_name="class_similarity.csv",
                           mime="text/csv", key=f"cs_dl_csv_{gen}")
        st.download_button("⬇ JSON", data=action["json"], file_name="class_similarity.json",
                           mime="application/json", key=f"cs_dl_json_{gen}")
    return action
