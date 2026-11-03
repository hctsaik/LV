"""Controlled observations from saved evidence; never infer semantic root causes."""
from __future__ import annotations

import copy
import math

QUESTION_TYPES = {
    'class_score': '指定類別／框分數',
    'box_localization': '框的位置或範圍',
    'postprocessing': '門檻／NMS 後處理',
    'machine_alarm': '現場機台報警',
}
QUESTION_NEEDS = {
    'box_localization': '需要對齊原圖的定位真值、框座標與定位誤差；目前 CAM 解釋類別／框分數，不解釋框回歸。',
    'postprocessing': '需要解碼前後候選、NMS 與門檻設定及被保留／移除紀錄；CAM 不解釋整段後處理。',
    'machine_alarm': '需要當時原圖、部署版本、機台規則與報警紀錄；CAM 不足以解釋機台最終報警。',
}

FILL_LABELS = {'blur': '模糊填補', 'constant': '固定值填補'}
PARITY_LABELS = {'matched': '相符', 'mismatched': '不符', 'mismatch': '不符', 'unavailable': '尚未驗證'}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def variant_hashes(attribution):
    return sorted({r['modified_sha256'] for r in attribution.get('faithfulness', {}).get('rows', [])
                   if isinstance(r.get('modified_sha256'), str) and r['modified_sha256']})


def record_preservation_review(case, attribution_id, *, verdict, reviewer, reason):
    from fp_cases import _evidence
    attribution = next((a for a in case.get('attributions', []) if a['id'] == attribution_id), None)
    if attribution is None or not variant_hashes(attribution):
        raise ValueError('需要已保存且具有修改圖指紋的歸因結果')
    if verdict not in {'preserved', 'changed', 'uncertain'} or not reviewer.strip() or not reason.strip():
        raise ValueError('請提供目標保留判讀、覆核者與依據')
    return _evidence(case, 'target_preservation_review', '人工核對修改圖是否保留目標外觀',
        attribution_id=attribution_id, variant_sha256=variant_hashes(attribution),
        target_identity=attribution.get('target_identity'), verdict=verdict,
        reviewer=reviewer.strip(), reason=reason.strip())


def preservation_review(case, attribution):
    hashes = variant_hashes(attribution)
    for evidence in reversed(case.get('evidence', [])):
        d = evidence.get('details', {})
        if evidence.get('role') != 'target_preservation_review' or d.get('attribution_id') != attribution['id']:
            continue
        if (hashes and sorted(d.get('variant_sha256', [])) == hashes
                and d.get('target_identity') == attribution.get('target_identity')):
            return {'status': d.get('verdict', 'uncertain'), 'reviewer': d.get('reviewer', ''), 'evidence_id': evidence['id']}
    return {'status': 'unconfirmed', 'reviewer': '', 'evidence_id': None}


def verified_item_review(case, item):
    if not item or item.get('withdrawn'):
        return None
    for evidence in reversed(case.get('evidence', [])):
        d = evidence.get('details', {})
        if evidence.get('role') == 'human_review' and d.get('item_id') == item['id'] and not d.get('candidate'):
            if (d.get('roi') == item.get('roi') and d.get('verdict') in {'normal', 'defect'}
                    and d.get('verdict') == item.get('review', {}).get('verdict')
                    and str(d.get('reviewer', '')).strip() and str(d.get('reason', '')).strip()):
                return evidence
            return None
    return None


def save_cohort_summary(case, job_id, summary):
    from fp_cases import _evidence
    for evidence in case.get('evidence', []):
        if evidence.get('role') == 'cohort_summary' and evidence.get('details', {}).get('job_id') == job_id:
            return evidence['id']
    if not job_id or summary.get('schema_version') != 1:
        raise ValueError('需要已完成的批次核對摘要')
    return _evidence(case, 'cohort_summary', '人工核對群組的歷史批次摘要',
                     job_id=job_id, summary=copy.deepcopy(summary), scope='historical_batch')


def snapshot_cohort(case, item, *, metadata=None):
    """Bind a new human declaration, or validate an existing one at submission."""
    from fp_attribution import target_identity
    current = next((r for r in case.get('items',[]) if r['id']==item['id']),None)
    result = copy.deepcopy(metadata if metadata is not None else (current or item).get('cohort',{}))
    review = verified_item_review(case,current)
    source = next((s for s in case.get('sources',[]) if current and s['id']==current.get('source_id')),{})
    identity = target_identity(case,current) if current else None
    expected = 'defect' if result.get('role')=='true_defect' else 'normal'
    valid = bool(current and not current.get('withdrawn') and review and review['details']['verdict']==expected
                 and result.get('role') in {'normal','true_defect','false_positive'}
                 and all(str(result.get(k,'')).strip() for k in ('group_id','reviewer','reason')))
    if metadata is not None:
        result.update(human_review_evidence_id=review['id'] if review else None,
                      target_identity=identity,source_sha256=source.get('sha256'))
    else:
        valid = valid and result.get('human_review_evidence_id')==(review or {}).get('id') and result.get('target_identity')==identity and result.get('source_sha256')==source.get('sha256')
    if result.get('verified') is True and not valid:
        result['verified']=False
        result['validation_reason']='目前區域、來源或最新人工覆核與群組確認不一致，請重新核對批次資料。'
    elif result.get('verified') is True:
        result.pop('validation_reason',None)
    else:
        result.setdefault('validation_reason','群組資料尚未完成人工確認。')
    return result


def build_explanation_summary(case, evidence_ids=None, attribution_id=None):
    available = {e['id']: e for e in case.get('evidence', [])}
    selected = set(available) if evidence_ids is None else set(evidence_ids)
    if not selected <= set(available):
        raise ValueError('摘要引用證據不存在')
    question = case.get('explanation_question', {}).get('type', 'class_score')
    if question not in QUESTION_TYPES:
        question = 'class_score'
    result = {'schema_version': 1, 'question': {'type': question, 'label': QUESTION_TYPES[question]},
        'status': 'inconclusive', 'observations': [], 'targets': [], 'missing_evidence': [], 'next_actions': [],
        'evidence_ids': [], 'limitations': ['觀察限定指定候選分數與修改方式；不證明反光等語意成因或現場報警原因。']}
    refs, statuses = set(), []
    if question in QUESTION_NEEDS:
        result['missing_evidence'].append(QUESTION_NEEDS[question])
    attributions = [a for a in case.get('attributions', []) if
        (attribution_id is None and a.get('evidence_id') in selected) or
        (a['id'] == attribution_id and a.get('evidence_id') in selected)]
    for attribution in attributions:
        eid = attribution['evidence_id']
        refs.add(eid)
        item = next((i for i in case.get('items', []) if i['id'] == attribution.get('item_id')), None)
        current = bool(item and not item.get('withdrawn'))
        if current and attribution.get('target_identity'):
            from fp_attribution import target_identity
            current = target_identity(case, item) == attribution['target_identity']
        report = attribution.get('faithfulness') or {}
        review = preservation_review(case, attribution)
        if review['evidence_id']:
            refs.add(review['evidence_id'])
        target = {'attribution_id': attribution['id'], 'evidence_id': eid, 'current': current,
            'label': (item or {}).get('label') or attribution.get('target', {}).get('class_name') or '指定框',
            'truth': copy.deepcopy((item or {}).get('review', {'verdict': 'pending'})),
            'scalar': attribution.get('target', {}).get('scalar', 'unknown'),
            'historical_parity': (attribution.get('historical_parity') or {}).get('status', 'unavailable'),
            'correspondence': (attribution.get('correspondence') or {}).get('status', 'unavailable'),
            'preservation': review, 'variant_sha256': variant_hashes(attribution)}
        result['targets'].append(target)
        def observe(text, **values):
            result['observations'].append({'text': text, 'evidence_ids': [eid], **values})
        truth = target['truth'].get('verdict', 'pending')
        human_review = verified_item_review(case, item)
        if not human_review:
            truth = 'pending'
            target['truth'] = {'verdict': 'pending'}
        else:
            refs.add(human_review['id'])
        truth_label = {'normal': '人工確認此區域正常', 'defect': '人工確認此區域有瑕疵'}.get(truth, '尚未有明確人工真值')
        observe(f"{target['label']}：{truth_label}；歷史部署比對{PARITY_LABELS.get(target['historical_parity'],'尚未驗證')}。")
        if human_review:
            result['observations'][-1]['evidence_ids'].append(human_review['id'])
        if truth not in {'normal', 'defect'}:
            result['missing_evidence'].append('需要人工核對原圖真值；未標註不代表正常。')
        if not current:
            observe('原目標已修正、撤回或不存在；以下僅保留歷史觀察。')
        if _number(report.get('baseline_scalar')):
            observe(f"原固定候選分數 {report['baseline_scalar']:.5f}。", baseline_scalar=report['baseline_scalar'])
        for row in report.get('rows', []):
            if row.get('unavailable'):
                observe(f"比例 {row.get('fraction')}、{FILL_LABELS.get(row.get('fill_method'),'舊協定未記錄填補方式')}：公平控制可用 {row.get('available_controls',0)}／要求 {row.get('requested_controls',report.get('controls','未知'))}；有限搜尋未找到足夠同形狀且低重疊區域。")
                result['missing_evidence'].append('控制不足：嘗試較小遮擋比例或其他原圖，再核對相同條件。')
            if not _number(row.get('fixed_scalar')) or not _number(row.get('fixed_score_drop')):
                continue
            delta = row['fixed_score_drop']
            change = f'下降 {delta:.5f}' if delta > 0 else f'上升 {-delta:.5f}' if delta < 0 else '未變'
            role = '熱區修改' if row.get('role') == 'attribution' else '控制修改'
            observe(f"{role}（比例 {row.get('fraction', '未知')}、{FILL_LABELS.get(row.get('fill_method'),'舊協定未記錄填補方式')}）：原候選分數 {change}；修改後 {row['fixed_scalar']:.5f}。",
                kind='variant', modified_sha256=row.get('modified_sha256'), fixed_scalar=row['fixed_scalar'],
                rematched_detection=copy.deepcopy(row.get('rematched_detection')))
        if report.get('protocol_version') == 2:
            from attribution_controls import assess_faithfulness
            assessment = assess_faithfulness(report)
        else:
            assessment = {'status': 'inconclusive', 'stability': {'status': 'unknown'}}
            result['missing_evidence'].append('缺少版本 2 公平區域控制；舊結果不自動視為已驗證。')
        target['assessment'] = copy.deepcopy(assessment)
        status = assessment.get('status', 'inconclusive')
        label = {'supports': '此協定下，熱區修改相對控制使固定分數更低（或上升較少）',
                 'opposes': '此協定下，未觀察到熱區修改相對控制使固定分數更低',
                 'unstable': '不同填補／遮擋比例的方向不一致，觀察不穩定',
                 'inconclusive': '目前控制或可用觀察不足，不能判讀方向'}.get(status, '目前不能判讀方向')
        observe(label+'；這是分數敏感性觀察，不是已確認根因。')
        if review['status'] != 'preserved':
            result['missing_evidence'].append('修改圖尚未確認保留目標外觀，或人工判為已改變／不確定；不能據此判定改善。')
        statuses.append(status if current and human_review and review['status'] == 'preserved' else 'inconclusive')
    if not attributions:
        result['missing_evidence'].append('尚無選取且已保存的模型歸因證據。')
    if any(t['historical_parity'] != 'matched' for t in result['targets']):
        result['missing_evidence'].append('尚未確認現場歷史判定完整重現；配套模型當圖對齊不能取代歷史部署驗證。')
    result['cohorts'] = []
    for eid in sorted(selected):
        evidence = available[eid]
        if evidence.get('role') != 'cohort_summary':
            continue
        cohort = copy.deepcopy(evidence.get('details', {}).get('summary', {}))
        result['cohorts'].append({'evidence_id': eid, 'summary': cohort, 'scope': 'historical_batch'})
        refs.add(eid)
        result['observations'].append({'text': f"歷史批次核對：不同影像 {cohort.get('unique_images',0)} 張、核實影像 {cohort.get('verified_images',0)} 張、人工宣告群組 {cohort.get('unique_groups',0)} 組；不同執行條件分開核對，不能由樣本數推定因果或現場效果。", 'evidence_ids': [eid]})
        contrasts=[(partition.get('identity_key','未知')[:12],row) for partition in cohort.get('partitions',[]) for row in partition.get('contrasts',{}).get('rows',[])]
        for group_key,row in contrasts[:6]:
            role_label={'normal':'正常樣本','true_defect':'確認瑕疵','false_positive':'確認誤逮'}.get(row['role'],'未知角色')
            result['observations'].append({'text': f"執行組別 {group_key}／{role_label}／比例 {row['fraction']}／{FILL_LABELS.get(row['fill_method'],'未記錄填補方式')}：{row['unique_images']} 張影像、{row['unique_groups']} 個人工群組的群組平均相對減分 {row['mean_group_excess_drop']:+.5f}；僅描述此組別的分數差異。", 'evidence_ids':[eid]})
        if len(contrasts)>6:
            result['observations'].append({'text':f'另有 {len(contrasts)-6} 組批次描述性比較保留於工程附錄。','evidence_ids':[eid]})
    if not result['cohorts']:
        result['missing_evidence'].append('單案或同圖多框不構成跨批次獨立支持；需另外核對獨立影像與批次。')
    if question == 'class_score' and statuses and len(set(statuses)) == 1:
        result['status'] = statuses[0]
    elif question == 'class_score' and ('unstable' in statuses or {'supports', 'opposes'} <= set(statuses)):
        result['status'] = 'unstable'
    result['evidence_ids'] = sorted(refs)
    result['missing_evidence'] = list(dict.fromkeys(result['missing_evidence']))
    result['next_actions'] = list(result['missing_evidence'])
    return result


def render_summary_text(summary):
    lines = ['受控證據摘要（自動整理，與人工回覆分開）', '問題：'+summary['question']['label']]
    status_labels={'supports':'相對減分方向有支持（限此觀察協定）','opposes':'相對減分方向未獲支持','unstable':'觀察方向不穩定','inconclusive':'目前證據不足以完成此題型說明'}
    lines.append('整理狀態：'+status_labels.get(summary['status'],'未確認'))
    for row in summary['observations']:
        if row.get('kind') == 'variant':
            continue
        lines.append(row['text']+' 【'+', '.join(row['evidence_ids'])+'】')
    comparisons=[(target,row) for target in summary.get('targets',[]) for row in target.get('assessment',{}).get('comparisons',[])]
    for target,row in comparisons[:6]:
        lines.append(f"比例 {row.get('fraction')}／{FILL_LABELS.get(row.get('fill_method'),'未記錄填補方式')}：熱區減分－控制平均 {row.get('excess_drop',0):+.5f}（控制 {row.get('control_count',0)} 個）；正值代表分數相對更低或上升較少。【{target['evidence_id']}】")
    if len(comparisons)>6:
        lines.append(f'另有 {len(comparisons)-6} 組比較，保留於工程證據附錄。')
    variants=[row for row in summary['observations'] if row.get('kind')=='variant']
    if variants:
        lines.append(f'{len(variants)} 張修改圖的逐圖分數保留於詳細紀錄；重新匹配框不取代原候選固定分數。')
    lines.extend('尚需核對：'+text for text in summary['missing_evidence'])
    lines.extend(summary['limitations'])
    return '\n'.join(lines)
