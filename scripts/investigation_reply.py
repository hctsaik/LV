"""Portable human-reviewed replies; technical evidence is a separate download."""
from __future__ import annotations

import base64
import copy
import html
import io
import json
import zipfile
from pathlib import Path

from PIL import Image
from fp_cases import utc_now, digest


def reply_review_warnings(case, evidence_ids=None):
    ids={e['id'] for e in _selected(case,evidence_ids)}
    warnings=[]
    if any(c.get('needs_review') and ids.intersection(c.get('refs',[])) for c in case.get('claims',[])):
        warnings.append('部分引用區域已修正或撤回，先前說明需要重新核對。')
    previous=case.get('reply_revisions',[])
    if previous and any(c.get('needs_review') for c in case.get('claims',[])):
        warnings.append('已保存的舊回覆可能含過期結論，請檢查並保存新的回覆版本。')
    return warnings


def reply_attribution_images(case, store, evidence_ids):
    """Only published, checksummed in-workspace overlays may enter public HTML."""
    selected=_selected(case,evidence_ids)
    images=[]
    rows={r['id']:r for r in case.get('attributions',[])}
    for evidence in selected:
        if evidence.get('role') != 'model_attribution':
            continue
        row=rows.get(evidence.get('details',{}).get('attribution_id'))
        if row is None:
            raise ValueError('選取的熱圖證據不存在，請重新檢查案件')
        relative=row.get('artifact_relative')
        directory=(store.root/relative).resolve() if relative else Path(row.get('artifact_dir','')).resolve()
        path=(directory/'overlay.png').resolve()
        if not directory.is_relative_to(store.root.resolve()) or not path.is_relative_to(directory):
            raise ValueError('熱圖不在目前工作區，請重新附入原始證據')
        expected=row.get('overlay_sha256')
        if expected is None:
            from investigation_jobs import JobStore
            jobs=JobStore(store.root)
            for job in jobs.list(states={'succeeded'}):
                root=jobs.artifact_dir(job['id']).resolve()
                if path.is_relative_to(root):
                    name=path.relative_to(root).as_posix()
                    expected=next((a['sha256'] for a in job['artifacts'] if a['path']==name),None)
                    break
        if not expected:
            raise ValueError('熱圖缺少完整性紀錄，請重新保存分析結果後再加入回覆')
        try:
            data=path.read_bytes()
        except OSError as exc:
            raise ValueError('熱圖資產已遺失，請恢復原始分析證據') from exc
        if digest(data)!=expected:
            raise ValueError('熱圖完整性檢查失敗，請恢復原始分析證據')
        with Image.open(io.BytesIO(data)) as image:
            output=io.BytesIO()
            image.convert('RGB').save(output,format='PNG')
        item=next((i for i in case.get('items',[]) if i['id']==row.get('item_id')),None)
        historical=item is None or bool(item.get('withdrawn'))
        if item is not None and row.get('target_identity'):
            from fp_attribution import target_identity
            historical=historical or target_identity(case,item)!=row['target_identity']
        label=row.get('target',{}).get('class_name') or (item or {}).get('label') or '指定框'
        caption=f"{label}：對應 PyTorch 模型歸因 · {row.get('method','Grad-CAM')}"
        if historical:
            caption+='（歷史區域，需重新核對）'
        images.append({'evidence_id':evidence['id'],'attribution_id':row['id'],'data':output.getvalue(),
                       'sha256':expected,'caption':caption,'legend':row.get('legend',''),
                       'limitation':row.get('limitation','熱圖不是已確認根因或機台最終判定'),'historical':historical})
    return images


def _selected(case, evidence_ids):
    available = {e['id']: e for e in case.get('evidence', [])}
    ids = list(available) if evidence_ids is None else list(evidence_ids)
    if not set(ids) <= set(available):
        raise ValueError('引用證據不存在')
    return [available[i] for i in ids]


def factual_draft(case, evidence_ids=None):
    """Never generates causal claims from heatmaps or arbitrary evidence JSON."""
    selected = _selected(case, evidence_ids)
    ids = {e['id'] for e in selected}
    lines = [str(case.get('description','')).strip(), '', '目前紀錄：']
    for item in case.get('items', []):
        if item.get('withdrawn'):
            continue
        from explanation_summary import verified_item_review
        review_evidence=verified_item_review(case,item)
        if not review_evidence:
            continue
        review = item.get('review', {})
        if review.get('verdict') == 'normal':
            lines.append(f"人工已確認「{item.get('label') or '指定區域'}」正常。【{review_evidence['id']}】")
        elif review.get('verdict') == 'defect':
            lines.append(f"人工已確認「{item.get('label') or '指定區域'}」有瑕疵。【{review_evidence['id']}】")
    for run in case.get('external_runs', []):
        if run.get('evidence_id') not in ids:
            continue
        profile=run.get('profile_snapshot',{})
        name=profile.get('name') or run.get('contract',{}).get('model_version') or '所選模型'
        prediction=run.get('prediction',{})
        observed=f"得到 {len(prediction.get('detections',[]))} 個偵測框" if prediction.get('task')=='detection' else f"輸出類別「{prediction.get('class','未知')}」"
        lines.append(f"本次以「{name}」對提供的影像推論，{observed}；歷史部署與機台最終判定仍需另外核對。")
    for attribution in case.get('attributions', []):
        if attribution.get('evidence_id') not in ids:
            continue
        item=next((r for r in case.get('items',[]) if r['id']==attribution.get('item_id')),None)
        if item is None or item.get('withdrawn'):
            continue
        from fp_attribution import target_identity
        if attribution.get('target_identity') and attribution['target_identity']!=target_identity(case,item):
            continue
        method='正向 Grad-CAM' if attribution.get('method')=='positive' else '正負貢獻診斷'
        observation='已有指定框的歸因結果' if attribution.get('status')=='ok' else '本次尚無有效歸因訊號'
        lines.append(f"已對「{item.get('label') or '指定框'}」執行{method}，{observation}；這不等於已確認誤逮原因。")
    for claim in case.get('claims', []):
        if claim.get('kind') == 'fact' and not claim.get('needs_review') and set(claim.get('refs',[])) <= ids and claim.get('refs'):
            lines.append(str(claim['text']))
    if len(lines) == 3:
        lines.append('目前尚無已確認的判讀結論。')
    lines.extend(['', '尚待確認：誤逮原因與改善效果需依後續證據判斷。'])
    actions = [c['text'] for c in case.get('claims',[]) if c.get('kind') == 'action' and not c.get('needs_review') and set(c.get('refs',[])) <= ids]
    lines.extend(['', '下一步：', *(actions or ['補充原圖、現場輸出或人工覆核結果。'])])
    if any(e.get('role') in {'model_attribution','cohort_summary'} for e in selected) or case.get('explanation_question',{}).get('type','class_score') != 'class_score':
        from explanation_summary import build_explanation_summary, render_summary_text
        lines.extend(['',render_summary_text(build_explanation_summary(case,[e['id'] for e in selected]))])
    return '\n'.join(lines)


def make_reply_revision(case, *, text, evidence_ids, image_ids, relations=None):
    _selected(case, evidence_ids)
    sources = {s['id'] for s in case.get('sources',[])}
    if not set(image_ids) <= sources:
        raise ValueError('回覆圖片不存在')
    relations = dict(relations or {})
    if not set(relations) <= set(evidence_ids) or any(v not in {'support','opposition','pending'} for v in relations.values()):
        raise ValueError('證據關係須為支持、反駁或待查')
    from explanation_summary import build_explanation_summary
    return {'case_id': case['id'], 'case_revision': case['revision'], 'created_at': utc_now(),
            'text': str(text), 'evidence_ids': list(evidence_ids), 'image_ids': list(image_ids),
            'relations': relations, 'authorship': 'engineer',
            'explanation_summary':build_explanation_summary(case,evidence_ids)}


def render_reply_html(case, store, *, text=None, evidence_ids=None, image_ids=None, relations=None):
    evidence = _selected(case, evidence_ids)
    image_ids = [] if image_ids is None else list(image_ids)
    text = factual_draft(case, [e['id'] for e in evidence]) if text is None else text
    revision = make_reply_revision(case, text=text, evidence_ids=[e['id'] for e in evidence], image_ids=image_ids, relations=relations)
    esc = lambda value: html.escape(str(value), quote=True)
    parts = ['<!doctype html><html lang="zh-Hant"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>案件回覆</title><style>body{max-width:880px;margin:32px auto;padding:16px;font:17px/1.7 sans-serif;color:#243044}img{max-width:100%}pre{white-space:pre-wrap;font:inherit}figure{margin:24px 0}small{color:#526078}</style><body>',
             '<h1>' + esc(case.get('title','案件回覆')) + '</h1>',
             '<small>案件版本 ' + esc(case['revision']) + ' · 回覆時間 ' + esc(revision['created_at']) + '</small>',
             '<pre>' + esc(text) + '</pre>']
    if revision['explanation_summary']['targets'] or revision['explanation_summary']['cohorts'] or case.get('explanation_question',{}).get('type','class_score')!='class_score':
        from explanation_summary import render_summary_text
        parts.append('<details open><summary>可核對的自動證據摘要</summary><pre>'+esc(render_summary_text(revision['explanation_summary']))+'</pre></details>')
    for warning in reply_review_warnings(case,[e['id'] for e in evidence]):
        parts.append('<p><strong>'+esc(warning)+'</strong></p>')
    sources = {s['id']:s for s in case.get('sources',[])}
    for source_id in image_ids:
        source = sources[source_id]
        with Image.open(store.image_path(source)) as im:
            output = io.BytesIO()
            im.convert('RGB').save(output, format='PNG')
        encoded = base64.b64encode(output.getvalue()).decode('ascii')
        parts.append('<figure><img alt="案件附圖" src="data:image/png;base64,' + encoded + '"><figcaption>' + esc(source['name']) + ('（截圖）' if source['kind'] == 'screenshot' else '') + '</figcaption></figure>')
    for overlay in reply_attribution_images(case,store,[e['id'] for e in evidence]):
        encoded=base64.b64encode(overlay['data']).decode('ascii')
        parts.append('<figure><img alt="選取的模型歸因熱圖" src="data:image/png;base64,'+encoded+'"><figcaption>'+esc(overlay['caption'])+'<br>'+esc(overlay['legend'])+'<br>'+esc(overlay['limitation'])+'</figcaption></figure>')
    referenced=set(revision['explanation_summary']['evidence_ids'])
    evidence=evidence+[e for e in case.get('evidence',[]) if e['id'] in referenced and e['id'] not in {r['id'] for r in evidence}]
    if evidence:
        labels = {'support':'支持', 'opposition':'反駁', 'pending':'待查'}
        parts.append('<h2>選取的證據</h2><ul>')
        for e in evidence:
            label = labels[revision['relations'].get(e['id'], 'pending')]
            # Public reply intentionally excludes raw details, filesystem paths and hashes.
            parts.append('<li>' + esc(label) + '：' + esc(e['text']) + ' 【'+esc(e['id'])+'】</li>')
        parts.append('</ul>')
    parts.append('</body></html>')
    return ''.join(parts)


def engineering_appendix(case):
    """Separate lossless JSON package. No arbitrary server path is read."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr('case-evidence.json', json.dumps(copy.deepcopy(case), ensure_ascii=False, indent=2, allow_nan=False))
        bundle.writestr('README.txt', '工程附錄含完整案件紀錄及證據來源。圖片與模型資產不包含於此檔；對外回覆請使用獨立圖文 HTML。')
    return output.getvalue()
