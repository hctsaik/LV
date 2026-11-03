import copy

from explanation_summary import (build_explanation_summary, record_preservation_review,
    preservation_review, save_cohort_summary, render_summary_text)


def case_fixture():
    rows=[]
    for method in ('constant','blur'):
        for control in (0,1):
            drop=.3 if control==0 else .1
            rows.append(dict(role='attribution' if control==0 else 'random_control',
                fraction=.1,fill_method=method,control_strategy='translated_mask',
                mask_sha256=str(control)*64,modified_sha256=(method+str(control)),
                fixed_score_drop=drop,fixed_scalar=.8-drop))
    return dict(id='case',evidence=[dict(id='cam',role='model_attribution',details={}),
        dict(id='review',role='human_review',details=dict(item_id='item',roi=[0,0,1,1],verdict='normal',reviewer='engineer',reason='spec checked'))],
        items=[dict(id='item',label='scratch',roi=[0,0,1,1],review=dict(verdict='normal'))],
        attributions=[dict(id='attr',item_id='item',evidence_id='cam',target={},
            faithfulness=dict(status='measured',protocol_version=2,controls=1,fractions=[.1],fill_methods=['constant','blur'],baseline_scalar=.8,rows=rows))])


def test_review_required_and_refs_are_resolvable():
    case=case_fixture()
    assert build_explanation_summary(case)['status']=='inconclusive'
    evidence=record_preservation_review(case,'attr',verdict='preserved',reviewer='Alice',reason='all variants checked')
    summary=build_explanation_summary(case)
    assert summary['status']=='supports'
    assert set(summary['evidence_ids'])=={'cam','review',evidence}
    assert 'review' in summary['observations'][0]['evidence_ids']
    rendered=render_summary_text(summary)
    assert '模糊填補' in rendered and '固定值填補' in rendered and '尚未驗證' in rendered
    assert 'unavailable' not in rendered and 'blur' not in rendered
    case['evidence']=[e for e in case['evidence'] if e['id']!='review']
    assert build_explanation_summary(case)['targets'][0]['truth']['verdict']=='pending'


def test_variant_change_invalidates_confirmation_and_old_protocol_is_unknown():
    case=case_fixture()
    record_preservation_review(case,'attr',verdict='preserved',reviewer='Alice',reason='checked')
    case['attributions'][0]['faithfulness']['rows'][0]['modified_sha256']='changed'
    assert preservation_review(case,case['attributions'][0])['status']=='unconfirmed'
    case['attributions'][0]['faithfulness']['protocol_version']=1
    assert build_explanation_summary(case)['targets'][0]['assessment']['status']=='inconclusive'


def test_opposing_fill_and_non_score_question_do_not_claim_cause():
    case=case_fixture()
    record_preservation_review(case,'attr',verdict='preserved',reviewer='Alice',reason='checked')
    rows=case['attributions'][0]['faithfulness']['rows']
    rows[2]['fixed_score_drop']=-.2
    rows[2]['fixed_scalar']=1.
    summary=build_explanation_summary(case)
    assert summary['status']=='unstable'
    assert any('上升 0.20000' in row['text'] for row in summary['observations'])
    assert '上升 0.20000' not in render_summary_text(summary)
    case['explanation_question']={'type':'machine_alarm'}
    summary=build_explanation_summary(case)
    assert summary['status']=='inconclusive'
    assert '機台規則與報警紀錄' in render_summary_text(summary)


def test_missing_controls_and_cohort_are_explicit_frozen_and_selected():
    case=case_fixture()
    case['attributions'][0]['faithfulness']['rows'].append(dict(unavailable=True,fraction=.5,fill_method='blur',available_controls=0,requested_controls=3))
    assert '可用 0／要求 3' in render_summary_text(build_explanation_summary(case))
    cohort=dict(schema_version=1,unique_images=4,verified_images=3,unique_groups=2,partitions=[])
    evidence=save_cohort_summary(case,'job',cohort)
    cohort['unique_images']=99
    assert save_cohort_summary(case,'job',cohort)==evidence
    summary=build_explanation_summary(case,[evidence])
    assert summary['cohorts'][0]['summary']['unique_images']==4
    assert summary['evidence_ids']==[evidence]
    assert not build_explanation_summary(case,['cam'])['cohorts']


def test_withdrawal_only_retains_historical_observations():
    case=case_fixture()
    record_preservation_review(case,'attr',verdict='preserved',reviewer='Alice',reason='checked')
    case['items'][0]['withdrawn']=True
    summary=build_explanation_summary(case)
    assert summary['status']=='inconclusive'
    assert not summary['targets'][0]['current']
