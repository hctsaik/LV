"""Real browser material intake, graphical ROI, portable reply and reviewed handoff.

No server paths or JSON are entered in the ordinary UI. The controlled model is
pre-registered solely as an existing-resource fixture; model upload is covered
by the separate model-library browser gate.
"""
from __future__ import annotations

import io
import base64
import json
import re
import zipfile

import pytest
from PIL import Image
from playwright.sync_api import expect

from fp_cases import CaseStore
from investigation_data import DatasetRegistry
from fp_onnx_demo import make_demo_model
from model_profiles import ModelRegistry
from .conftest import load_app, wait_idle
from .test_fp_case_workspace_e2e import fp_server, _click as _base_click

pytestmark = pytest.mark.e2e


def _click(page,name):
    print('[materials] '+name,flush=True)
    _base_click(page,name)


@pytest.fixture(autouse=True)
def material_browser_evidence(page,tmp_path):
    expect.set_options(timeout=30000)
    yield
    if not page.is_closed():
        page.screenshot(path=str(tmp_path/'materials-last-screen.png'),full_page=True)
        (tmp_path/'materials-last-screen.txt').write_text(page.locator('body').inner_text(),encoding='utf-8')


def _select(page,label,choice):
    widget=page.locator('[data-testid="stSelectbox"]:not([data-stale="true"] *)').filter(has=page.get_by_text(label,exact=True))
    expect(widget).to_be_visible()
    widget.locator('[data-baseweb="select"]').click()
    page.get_by_role('option',name=choice,exact=True).click()
    wait_idle(page,timeout=30000)


def _image(color):
    output=io.BytesIO()
    Image.new('RGB',(180,120),color).save(output,'PNG')
    return output.getvalue()


def _upload(page,label,files):
    uploader=page.get_by_test_id('stFileUploader').filter(has=page.get_by_text(label,exact=True))
    uploader.locator('input[type=file]').set_input_files(files)
    wait_idle(page,timeout=60000)


def _multiselect_first(page,label):
    page.get_by_test_id('stMultiSelect').filter(has=page.get_by_text(label,exact=True)).locator('[data-baseweb="select"]').click()
    page.get_by_role('option').first.click()
    page.keyboard.press('Escape')


def _draw(page):
    frame=page.frame_locator('iframe[title*="lv_region_picker"]')
    frame.get_by_role('button',name='圈選爭議區域',exact=True).click()
    canvas=frame.locator('#canvas')
    canvas.scroll_into_view_if_needed()
    box=canvas.bounding_box()
    assert box
    page.mouse.move(box['x']+box['width']*.1,box['y']+box['height']*.15)
    page.mouse.down()
    page.mouse.move(box['x']+box['width']*.7,box['y']+box['height']*.8,steps=8)
    page.mouse.up()
    wait_idle(page,timeout=30000)
    # Streamlit keeps old form DOM briefly during component-triggered reruns.
    # Read only the current input, even while its optional expander is closed.
    coordinate=page.locator('input[aria-label="左 %"]:not([data-stale="true"] *)')
    expect(coordinate).to_have_value(re.compile(r'10(?:\.0+)?'),timeout=30000)


def _new_case(page,image,title,kind='原圖'):
    exp=page.get_by_test_id('stExpander').filter(has=page.get_by_text('建立誤逮案件',exact=True))
    details=exp.locator('details')
    expect(details).to_be_visible()
    if details.get_attribute('open') is None:
        exp.locator('summary').click()
    expect(exp.get_by_role('textbox',name='案件名稱（可留空）',exact=True)).to_be_visible()
    _select(page,'圖片來源',kind)
    _upload(page,'報案圖片或截圖',image)
    expect(exp.get_by_role('textbox',name='案件名稱（可留空）',exact=True)).to_be_visible()
    page.get_by_role('textbox',name='案件名稱（可留空）',exact=True).fill(title)
    page.get_by_role('textbox',name='使用者描述／認為正常的理由',exact=True).fill('使用者回報正常反光被判為瑕疵，請核對')
    _click(page,'建立案件')
    opened=page.get_by_test_id('stSelectbox').filter(has=page.get_by_text('開啟案件',exact=True))
    expect(opened).to_contain_text(title,timeout=30000)


def test_uploaded_materials_graphical_case_reply_and_reference(fp_server,page,tmp_path):
    page.set_default_timeout(30000)
    page.set_viewport_size({'width':1600,'height':1100})
    load_app(page,fp_server.base_url)
    _click(page,'誤逮案件')
    _click(page,'資料庫')
    package=tmp_path/'mixed.zip'
    with zipfile.ZipFile(package,'w') as z:
        z.writestr('a/same.png',_image('gray'))
        z.writestr('a/same.txt','')
        z.writestr('b/same.png',_image('white'))
        z.writestr('broken.png',b'not an image')
        z.writestr('../escape.png',_image('red'))
    _upload(page,'圖片與紀錄',package)
    page.get_by_role('textbox',name='資料集名稱',exact=True).fill('上傳參考資料 E2E')
    _click(page,'預覽匯入')
    expect(page.get_by_text('broken.png',exact=False).first).to_be_visible()
    _click(page,'保存資料集')
    expect(page.get_by_text('已保存：上傳參考資料 E2E',exact=True)).to_be_visible()
    registry=DatasetRegistry(CaseStore(fp_server.case_root))
    dataset=next(d for d in registry.list() if d['name']=='上傳參考資料 E2E')
    assert [r['annotation']['status'] for r in dataset['images']]==['empty','missing']
    assert len(dataset['errors'])==2
    assert not (fp_server.case_root.parent/'escape.png').exists()

    # U01/U10: only a screenshot, no model, actual pointer-drawn ROI and reply.
    _click(page,'案件')
    screenshot=tmp_path/'reported.png'; screenshot.write_bytes(_image('silver'))
    _new_case(page,screenshot,'截圖查案 E2E','截圖')
    _draw(page)
    page.get_by_role('textbox',name='模型預測類別／爭議名稱',exact=True).fill('反光區域')
    _click(page,'加入爭議區域')
    expect(page.get_by_role('textbox',name='覆核者',exact=True)).to_be_visible()
    case=next(c for c in CaseStore(fp_server.case_root).list() if c['title']=='截圖查案 E2E')
    assert case['items'][0]['roi'] != [0,0,1,1]
    assert not case.get('external_runs')
    _click(page,'找對照')
    _select(page,'正常對照資料集','上傳參考資料 E2E')
    expect(page.get_by_role('button',name='找對照案例',exact=True)).to_be_visible()
    # Selection is validated here; expensive actual DINO search has its own gate.
    _click(page,'說明與回覆')
    _multiselect_first(page,'附圖')
    page.get_by_role('textbox',name='回覆內容（可選取複製）',exact=True).fill('目前僅收到截圖，原因尚待原圖與模型輸出確認。')
    _click(page,'保存回覆版本')
    expect(page.get_by_text('回覆版本已保存；不代表根因或改善已確認',exact=True)).to_be_visible()
    with page.expect_download() as download:
        page.get_by_role('button',name='下載含圖片回覆 HTML',exact=True).click()
    rendered=download.value.path().read_text(encoding='utf-8')
    assert 'data:image/png;base64,' in rendered and '原因尚待' in rendered
    assert str(fp_server.case_root) not in rendered
    page.screenshot(path=str(tmp_path/'materials-reply.png'),full_page=True)


def test_uploaded_training_export_without_json(fp_server,page,tmp_path):
    page.set_default_timeout(30000)
    page.set_viewport_size({'width':1600,'height':1100})
    load_app(page,fp_server.base_url)
    _click(page,'誤逮案件')
    # U12: original image reviewed as a real defect; class chosen in a dropdown.
    original=tmp_path/'original.png'; original.write_bytes(_image('orange'))
    _new_case(page,original,'人工訓練交接 E2E')
    # U12 covers the supported precision form; U01 separately covers drawing.
    page.get_by_test_id('stExpander').filter(has=page.get_by_text('精確座標（可選）',exact=True)).locator('summary').click()
    for label,value in [('左 %','10'),('上 %','15'),('右 %','70'),('下 %','80')]:
        page.get_by_role('spinbutton',name=label,exact=True).fill(value)
    page.get_by_role('textbox',name='模型預測類別／爭議名稱',exact=True).fill('crack')
    _click(page,'加入爭議區域')
    expect(page.get_by_role('textbox',name='覆核者',exact=True)).to_be_visible()
    _select(page,'覆核結果','確認有瑕疵')
    page.get_by_role('textbox',name='覆核者',exact=True).fill('Engineer')
    page.get_by_role('textbox',name='判斷依據',exact=True).fill('完整原圖依規格確認一處瑕疵')
    _click(page,'儲存此區域覆核')
    expect(page.get_by_text('此區域覆核已儲存；其他區域維持原狀',exact=True)).to_be_visible()
    _click(page,'訓練交接')
    selector=page.get_by_test_id('stSelectbox').filter(has=page.get_by_text('人工確認類別：',exact=False)).first
    selector.locator('[data-baseweb="select"]').click()
    page.get_by_role('option',name='crack',exact=True).click()
    _multiselect_first(page,'已確認整張標註完整的影像')
    page.get_by_role('textbox',name='完整標註確認者',exact=True).fill('Engineer')
    page.get_by_role('textbox',name='完整檢查與類別確認依據',exact=True).fill('已核對整張原圖全部瑕疵')
    page.get_by_text('我已檢查選取影像的全部內容，真瑕疵框無遺漏，且影像是可用的原始訓練內容',exact=True).click()
    _click(page,'建立訓練交接 ZIP')
    with page.expect_download() as training:
        page.get_by_role('button',name='下載訓練交接 ZIP',exact=True).click()
    with zipfile.ZipFile(training.value.path()) as archive:
        assert archive.read('classes.txt').decode().strip()=='crack'
    page.screenshot(path=str(tmp_path/'materials-training-export.png'),full_page=True)


def test_uploaded_photo_evaluation_without_json(fp_server,page,tmp_path):
    """U12 evaluation: seed existing case/model; upload and confirm data in UI."""
    store=CaseStore(fp_server.case_root)
    case=store.create(store.ingest(_image('orange'),'existing-case.png'),'既有評估案件 E2E')
    page.set_default_timeout(30000)
    page.set_viewport_size({'width':1600,'height':1100})
    load_app(page,fp_server.base_url)
    _click(page,'誤逮案件')
    _select(page,'開啟案件',case['title']+' · '+case['id'])
    # Existing case/model are fixtures; every dataset and truth operation below
    # uses the ordinary upload, preview and photo-review controls.
    _click(page,'資料庫')
    eval_zip=tmp_path/'evaluation.zip'
    with zipfile.ZipFile(eval_zip,'w') as z:
        z.writestr('normal.png',_image('black')); z.writestr('normal.txt','')
    _upload(page,'圖片與紀錄',eval_zip)
    _select(page,'紀錄角色','人工標註')
    page.get_by_role('textbox',name='資料集名稱',exact=True).fill('評估資料 E2E')
    _click(page,'預覽匯入'); _click(page,'保存資料集')
    expect(page.get_by_text('已保存：評估資料 E2E',exact=True)).to_be_visible()
    models=ModelRegistry(fp_server.case_root)
    model=make_demo_model(tmp_path/'eval.onnx')
    asset=models.ingest(model.read_bytes(),model.name)
    config={'output_format':'classification','input_size':[32,32],'class_names':['good','crack'],'normal_classes':['good'],'score_activation':'softmax'}
    models.create('既有評估模型',deployment_asset=asset,runtime_config=config)
    _click(page,'案件'); _click(page,'改善評估')
    _click(page,'使用這兩個模型設定')
    _select(page,'評估資料集','評估資料 E2E')
    _select(page,'整張圖片真值','正常')
    page.get_by_text('已人工檢查整張圖片',exact=True).click()
    page.get_by_role('textbox',name='本張覆核者',exact=True).fill('Engineer')
    page.get_by_text('已確認整張框標註完整',exact=True).click()
    page.get_by_role('textbox',name='本張批次／群組',exact=True).fill('lot-A')
    _click(page,'保存本張確認')
    expect(page.get_by_text('本張確認已保存',exact=True)).to_be_visible(timeout=30000)
    page.screenshot(path=str(tmp_path/'materials-evaluation-confirmations.png'),full_page=True)
    _click(page,'確認並建立評估清單')
    expect(page.get_by_text('評估清單已通過真值與資料重疊檢查',exact=True)).to_be_visible()
    page.get_by_role('spinbutton',name='正常／瑕疵各自最低支持數',exact=True).fill('1')
    _click(page,'執行前後模型評估')
    expect(page.get_by_role('button',name='將完成結果加入案件',exact=True)).to_be_visible(timeout=120000)
    _click(page,'將完成結果加入案件')
    expect(page.get_by_text('僅報案回歸結果',exact=True)).to_be_visible()
    page.screenshot(path=str(tmp_path/'materials-evaluation.png'),full_page=True)


def test_ten_image_mixed_material_preview(fp_server,page,tmp_path):
    """U05: one real upload with relative names and per-record role correction."""
    page.set_default_timeout(30000)
    page.set_viewport_size({'width':1600,'height':1100})
    load_app(page,fp_server.base_url)
    _click(page,'誤逮案件'); _click(page,'資料庫')
    package=tmp_path/'ten-materials.zip'
    names=['a/same.png','b/same.png']+[f'p{i}.png' for i in range(2,8)]+['gt.png','missing.png']
    with zipfile.ZipFile(package,'w') as archive:
        for i,name in enumerate(names):
            archive.writestr(name,_image((i*20,80,120)))
            if name=='missing.png':
                continue
            annotation='' if i==0 else ('0 .5 .5 .2 .2' if name=='gt.png' else '0 .5 .5 .2 .2 .9')
            archive.writestr(name.replace('.png','.txt'),annotation)
        archive.writestr('broken.png',b'not an image')
        archive.writestr('../escape.png',_image('red'))
    _upload(page,'圖片與紀錄',package)
    page.get_by_role('textbox',name='資料集名稱',exact=True).fill('十圖混合 E2E')
    def edit_cell(grid,fraction,text):
        canvas=grid.locator('canvas').first
        canvas.scroll_into_view_if_needed()
        rect=canvas.bounding_box(); assert rect
        page.mouse.click(rect['x']+rect['width']*fraction,rect['y']+54,click_count=2)
        page.wait_for_timeout(300)
        page.keyboard.type(text)
        page.keyboard.press('Enter')
        wait_idle(page,timeout=30000)
        page.wait_for_timeout(700)
    edit_cell(page.get_by_test_id('stDataFrame').first,.7,'crack')
    exp=page.get_by_test_id('stExpander').filter(has=page.get_by_text('自訂圖片與紀錄配對（路徑不同時）',exact=True))
    exp.locator('summary').click()
    pairing=exp.get_by_test_id('stDataFrame')
    edit_cell(pairing,.15,'gt.png')
    edit_cell(pairing,.5,'gt.txt')
    canvas=pairing.locator('canvas').first
    canvas.scroll_into_view_if_needed()
    rect=canvas.bounding_box(); assert rect
    page.mouse.click(rect['x']+rect['width']*.83,rect['y']+54)
    page.keyboard.press('Enter')
    page.keyboard.press('ArrowDown')
    page.keyboard.press('ArrowDown')
    page.keyboard.press('Enter')
    wait_idle(page,timeout=30000)
    _click(page,'預覽匯入')
    page.screenshot(path=str(tmp_path/'ten-material-preview.png'),full_page=True)
    _click(page,'保存資料集')
    expect(page.get_by_text('已保存：十圖混合 E2E',exact=True)).to_be_visible(timeout=30000)
    dataset=next(d for d in DatasetRegistry(CaseStore(fp_server.case_root)).list() if d['name']=='十圖混合 E2E')
    by_name={row['name']:row for row in dataset['images']}
    assert len(by_name)==10 and len(dataset['errors'])==2
    assert by_name['a/same.png']['annotation']['status']=='empty'
    assert by_name['missing.png']['annotation']['status']=='missing'
    assert by_name['gt.png']['annotation']['role']=='ground_truth'
    assert 'score' not in by_name['gt.png']['annotation']['boxes'][0]
    assert by_name['b/same.png']['annotation']['boxes'][0]['class']=='crack'
    assert by_name['b/same.png']['annotation']['boxes'][0]['score']==.9


def test_seeded_cam_evidence_to_portable_reply(fp_server,page,tmp_path):
    """U10 wiring only: published checksummed CAM is seeded, never recomputed."""
    _cam_reply_story(fp_server,page,tmp_path)


def test_controlled_evidence_review_and_reply(fp_server,page,tmp_path):
    """Seed measured outputs; real GUI confirmation/draft/HTML, no model inference."""
    _cam_reply_story(fp_server,page,tmp_path,controlled=True)


def _cam_reply_story(fp_server,page,tmp_path,controlled=False):
    from fp_cases import add_item, digest
    from fp_attribution import target_identity
    store=CaseStore(fp_server.case_root)
    source=store.ingest(_image('gray'),'reported-with-cam.png',source_path='C:/private/customer-image.png')
    case=store.create(source,'CAM 圖文交付 E2E','模型歸因仍需人工判讀')
    item=add_item(case,source['id'],[.1,.15,.7,.8],'crack')
    overlay=_image('red')
    directory=store.root/'reply-cam-fixture'
    directory.mkdir(exist_ok=True)
    (directory/'overlay.png').write_bytes(overlay)
    evidence_id='cam-reply-fixture-evidence'
    attribution_id='cam-reply-fixture-attribution'
    evidence_text='指定框模型歸因（已保存測試證據）'
    case['attributions']=[{'id':attribution_id,'evidence_id':evidence_id,'item_id':item['id'],
        'artifact_relative':'reply-cam-fixture','artifact_dir':str(directory),'overlay_sha256':digest(overlay),
        'target_identity':target_identity(case,item),'status':'ok','method':'positive',
        'target':{'schema_version':2,'class_name':'crack','model_sha256':'fixture-model-hash',
                  'input_sha256':source['sha256'],'box':item['roi'],'scalar':'confidence'},
        'legend':'正向貢獻：亮色表示對指定框分數的正向歸因',
        'limitation':'熱圖不是已確認根因或機台最終判定'}]
    case['evidence'].append({'id':evidence_id,'role':'model_attribution','text':evidence_text,'at':'fixture',
                             'details':{'attribution_id':attribution_id,'item_id':item['id']}})
    case['evidence'].append({'id':'unselected-private-evidence','role':'lv_auxiliary','text':'不應匯出此無關證據',
                             'at':'fixture','details':{'path':'C:/private/unrelated.png'}})
    if controlled:
        from fp_cases import review_item
        review_item(case,item['id'],'normal','fixture reviewer','checked source against specification')
        rows=[]
        for method in ('constant','blur'):
            for control in (0,1):
                pixels=_image(['blue','green','purple','yellow'][len(rows)])
                path=directory/f'variant-{len(rows)}.png'
                path.write_bytes(pixels)
                drop=.3 if control==0 else .1
                rows.append(dict(role='attribution' if control==0 else 'random_control',control=control,
                    fraction=.1,fill_method=method,control_strategy='translated_mask',mask_sha256=str(control)*64,
                    modified_image=str(path),modified_sha256=digest(pixels),fixed_scalar=.8-drop,fixed_score_drop=drop))
        case['attributions'][0]['faithfulness']=dict(status='measured',protocol_version=2,controls=1,
            fractions=[.1],fill_methods=['constant','blur'],baseline_scalar=.8,rows=rows)
    store.save(case)
    page.set_default_timeout(30000)
    page.set_viewport_size({'width':1600,'height':1100})
    load_app(page,fp_server.base_url)
    _click(page,'誤逮案件')
    _select(page,'開啟案件',case['title']+' · '+case['id'])
    _click(page,'說明與回覆')
    page.get_by_test_id('stMultiSelect').filter(has=page.get_by_text('附上哪些證據',exact=True)).locator('[data-baseweb="select"]').click()
    page.get_by_role('option',name=evidence_text,exact=True).click()
    page.keyboard.press('Escape')
    _select(page,evidence_text,'反駁')
    if controlled:
        details=page.locator('details').filter(has=page.locator('summary',has_text='人工確認修改圖是否保留目標外觀')).first
        if not details.evaluate('(el)=>el.open'):details.locator('summary').click()
        _select(page,'修改圖判讀','仍保留目標外觀')
        page.get_by_role('textbox',name='修改圖覆核者',exact=True).fill('browser reviewer')
        page.get_by_role('textbox',name='目標保留判讀依據',exact=True).fill('All four saved variants inspected; target retained')
        page.get_by_text('我已核對本次全部 4 張不同修改圖',exact=True).click()
        _click(page,'保存修改圖人工確認')
        reviews=[e for e in store.load(case['id'])['evidence'] if e['role']=='target_preservation_review']
        assert len(reviews)==1 and len(reviews[0]['details']['variant_sha256'])==4
    reply='此熱圖不足以確認誤逮根因；請核對原圖與現場規則。'
    page.get_by_role('textbox',name='回覆內容（可選取複製）',exact=True).fill(reply)
    _click(page,'保存回覆版本')
    revision=store.load(case['id'])['reply_revisions'][-1]
    assert revision['text']==reply
    preview=page.frame_locator('iframe[srcdoc]').get_by_alt_text('選取的模型歸因熱圖',exact=True)
    expect(preview).to_be_visible()
    expect(preview).to_have_attribute('src','data:image/png;base64,'+base64.b64encode(overlay).decode())
    assert revision['evidence_ids']==[evidence_id]
    assert revision['relations']=={evidence_id:'opposition'}
    assert revision['attribution_images'][0]['attribution_id']==attribution_id
    assert revision['attribution_images'][0]['sha256']==digest(overlay)
    if controlled:
        assert revision['explanation_summary']['status']=='supports'
        assert 'review' in revision['explanation_summary']['targets'][0]['preservation']['reviewer']
    with page.expect_download() as download:
        page.get_by_role('button',name='下載含圖片回覆 HTML',exact=True).click()
    rendered=download.value.path().read_text(encoding='utf-8')
    assert 'alt="選取的模型歸因熱圖"' in rendered
    assert base64.b64encode(overlay).decode() in rendered
    assert reply in rendered and '反駁：'+evidence_text in rendered
    assert '熱圖不是已確認根因或機台最終判定' in rendered
    assert '不應匯出此無關證據' not in rendered
    assert str(store.root) not in rendered and 'C:/private/' not in rendered
    if controlled:
        assert '熱區修改相對控制使固定分數更低' in rendered
        assert reviews[0]['id'] in rendered
        assert '尚未確認現場歷史判定完整重現' in rendered
    with page.expect_download() as appendix:
        page.get_by_role('button',name='下載工程證據附錄',exact=True).click()
    with zipfile.ZipFile(appendix.value.path()) as archive:
        exported=json.loads(archive.read('case-evidence.json'))
        assert exported['reply_revisions'][-1]['evidence_ids']==[evidence_id]
    page.screenshot(path=str(tmp_path/'cam-reply-delivery.png'),full_page=True)


def test_two_browser_tabs_preserve_stale_case_draft(fp_server,page,tmp_path):
    """U14: two real Chromium tabs editing one seeded existing report."""
    store=CaseStore(fp_server.case_root)
    case=store.create(store.ingest(_image('gray'),'concurrent-report.png'),'雙分頁草稿 E2E','original description')
    page.set_default_timeout(30000)
    tab_b=page.context.new_page()
    tab_b.set_default_timeout(30000)
    try:
        for tab in (page,tab_b):
            tab.set_viewport_size({'width':1600,'height':1100})
            load_app(tab,fp_server.base_url)
            _click(tab,'誤逮案件')
            _select(tab,'開啟案件',case['title']+' · '+case['id'])
            expect(tab.get_by_role('textbox',name='報案描述',exact=True)).to_have_value('original description')
        draft_b=tab_b.get_by_role('textbox',name='報案描述',exact=True)
        draft_b.fill('B unsaved draft')
        draft_b.press('Tab')  # Blur sends widget state; it does not submit the form.
        page.get_by_role('textbox',name='報案描述',exact=True).fill('A saved first')
        _click(page,'儲存報案資訊')
        assert store.load(case['id'])['description']=='A saved first'
        revision=store.load(case['id'])['revision']
        _click(tab_b,'儲存報案資訊')
        expect(tab_b.get_by_text('案件已在另一個視窗更新；此頁草稿已保留，請先開啟最新版本並核對後再儲存',exact=False)).to_be_visible()
        expect(draft_b).to_have_value('B unsaved draft')
        assert store.load(case['id'])['revision']==revision
        assert store.load(case['id'])['description']=='A saved first'
        tab_b.screenshot(path=str(tmp_path/'stale-draft-rejected.png'),full_page=True)
        _click(tab_b,'開啟最新版本（保留此頁草稿）')
        expect(tab_b.get_by_text('已開啟最新版本；此頁草稿仍保留，請與目前已儲存內容核對後再儲存。',exact=True)).to_be_visible()
        latest=tab_b.get_by_test_id('stExpander').filter(has=tab_b.get_by_text('目前已儲存的報案資訊',exact=True))
        expect(latest).to_contain_text('A saved first')
        expect(draft_b).to_have_value('B unsaved draft')
        draft_b.fill('A saved first; B reviewed and added')
        _click(tab_b,'儲存報案資訊')
        assert store.load(case['id'])['description']=='A saved first; B reviewed and added'
        assert store.load(case['id'])['revision']==revision+1
        tab_b.screenshot(path=str(tmp_path/'stale-draft-merged.png'),full_page=True)
    finally:
        if not tab_b.is_closed():
            tab_b.screenshot(path=str(tmp_path/'second-tab-final.png'),full_page=True)
            (tmp_path/'second-tab-final.txt').write_text(tab_b.locator('body').inner_text(),encoding='utf-8')
            tab_b.close()


def test_uploaded_references_cam_to_real_dino_handoff(fp_server,page,tmp_path):
    """U11: seeded saved CAM; real upload/search/review/handoffs, no CAM rerun."""
    import numpy as np
    from fp_cases import add_item,digest
    from fp_attribution import attach_result,target_identity
    from investigation_jobs import JobStore
    store=CaseStore(fp_server.case_root)
    source=store.ingest(_image('gray'),'cam-query.png')
    case=store.create(source,'CAM 到上傳參考與交接 E2E')
    item=add_item(case,source['id'],[.1,.15,.7,.8],'crack')
    item['external_target']={'run_id':'seeded-run','output_index':0,'class':'crack'}
    case['external_runs']=[{'id':'seeded-run','source_id':source['id'],
        'profile_snapshot':{'id':'seeded-profile','name':'Saved CAM fixture','revision':1,'gradient_asset':{'format':'pt'}},
        'prediction':{'detections':[{'output_index':0,'class':'crack','score':.8,'box':item['roi']}]}}]
    case=store.save(case)
    identity=target_identity(case,item)
    jobs=JobStore(store.root)
    payload={'case_id':case['id'],'item_id':item['id'],'target_identity':identity,'method':'positive',
             'scalar':'confidence','fractions':[.1,.25,.5],'controls':3,'randomization':False}
    job=jobs.submit('gradcam',payload,'saved-cam-handoff-fixture',context={'case_id':case['id']})
    jobs._update(job['id'],{'state':'interrupted'})
    directory=jobs.artifact_dir(job['id']); directory.mkdir(parents=True)
    for name in ['source.png','model_input.png','overlay.png']:
        (directory/name).write_bytes(_image('gray'))
    np.savez(directory/'maps.npz',source=np.ones((120,180),dtype=np.float32))
    attribution={'id':'saved-cam-handoff','item_id':item['id'],'profile_id':'seeded-profile','profile_revision':1,
        'target_identity':identity,'method':'positive','status':'ok','artifact_dir':str(directory),
        'legend':'受控已保存 CAM；不宣稱根因','limitation':'模型歸因不等於正常或瑕疵真值',
        'target':{'schema_version':2,'model_sha256':'seeded-model','input_sha256':source['sha256'],'class_name':'crack','box':item['roi'],'scalar':'confidence'},
        'correspondence':{'status':'unavailable','scope':'fixture'},'historical_parity':{'status':'unavailable'}}
    case=attach_result(store,case['id'],{'attribution':attribution})
    cam_evidence=case['attributions'][0]['evidence_id']
    portable={**attribution,'artifact_dir':'@artifact/.'}
    (directory/'result.json').write_text(json.dumps({'attribution':portable}),encoding='utf-8')
    jobs._update(job['id'],{'state':'succeeded','artifacts':[{'path':p.name,'size':p.stat().st_size,'sha256':digest(p.read_bytes())} for p in directory.iterdir()]})
    page.set_default_timeout(30000); page.set_viewport_size({'width':1600,'height':1100})
    load_app(page,fp_server.base_url)
    _click(page,'誤逮案件'); _click(page,'資料庫')
    for dataset_name,file_name,color in [('正常上傳參考','normal-ref.png','silver'),('瑕疵上傳參考','defect-ref.png','yellow')]:
        uploader=page.get_by_test_id('stFileUploader').filter(has=page.get_by_text('圖片與紀錄',exact=True))
        remove=uploader.get_by_role('button',name=re.compile('^Remove '))
        while remove.count():
            remove.first.click()
            wait_idle(page,timeout=30000)
        path=tmp_path/file_name; path.write_bytes(_image(color))
        _upload(page,'圖片與紀錄',path)
        expect(uploader.get_by_role('button',name='Remove '+file_name,exact=True)).to_be_visible()
        page.get_by_role('textbox',name='資料集名稱',exact=True).fill(dataset_name)
        page.get_by_role('textbox',name='資料集名稱',exact=True).press('Tab')
        wait_idle(page,timeout=30000)
        _click(page,'預覽匯入'); _click(page,'保存資料集')
        for _ in range(150):
            saved_dataset=next((d for d in DatasetRegistry(store).list() if d['name']==dataset_name),None)
            if saved_dataset is not None:
                break
            page.wait_for_timeout(200)
        assert saved_dataset is not None
        assert [r['name'] for r in saved_dataset['images']]==[file_name]
    # Resume the saved CAM UI and use its real navigation action to establish
    # the exact CAM-to-query context; no server/session-state injection.
    load_app(page,fp_server.base_url+'?diagnostic_case='+case['id']+'&diagnostic_item='+item['id'])
    _click(page,'從此區域找同類案例')
    _select(page,'正常對照資料集','正常上傳參考')
    _select(page,'瑕疵對照資料集','瑕疵上傳參考')
    expect(page.get_by_role('textbox',name='正常對照資料夾（可留空）',exact=True)).to_have_value('')
    expect(page.get_by_role('textbox',name='真瑕疵對照資料夾（可留空）',exact=True)).to_have_value('')
    _click(page,'找對照案例')
    for _ in range(480):
        searched=store.load(case['id'])
        if searched['searches']:
            break
        page.wait_for_timeout(500)
    assert searched['searches'], 'Real DINO comparison was not published'
    search=searched['searches'][-1]
    assert search['recipe']['model']=='dinov2_vits14'
    assert len(search['recipe']['checkpoint_sha256'])==64
    assert search['attribution_evidence_id']==cam_evidence
    assert search['groups']['normal'] and search['groups']['defect']
    print('[materials] real DINO search and CAM lineage verified',flush=True)
    defect=page.get_by_text('真瑕疵對照',exact=True).locator('xpath=ancestor::div[@data-testid="stColumn"][1]')
    defect.get_by_role('button',name='加入待覆核',exact=True).first.click()
    wait_idle(page,timeout=30000)
    _click(page,'覆核批次')
    _select(page,'覆核結果','確認有瑕疵')
    page.get_by_role('textbox',name='覆核者',exact=True).fill('Engineer')
    page.get_by_role('textbox',name='判斷依據',exact=True).fill('人工檢視整張參考圖，確認此測試標註')
    _click(page,'儲存此區域覆核')
    _click(page,'建立／連結 Active Learning 覆核回合')
    _click(page,'準備含圖片的覆核包')
    with page.expect_download() as review_download:
        page.get_by_role('button',name='下載覆核包 ZIP',exact=True).click()
    with zipfile.ZipFile(review_download.value.path()) as archive:
        lineage=json.loads(archive.read('provenance.json'))
        assert cam_evidence in {e['id'] for e in lineage['evidence']}
        assert lineage['searches'][0]['id']==search['id']
        assert lineage['regions'][0]['review']['verdict']=='defect'
    print('[materials] review ZIP lineage verified',flush=True)
    _click(page,'訓練交接')
    selector=page.get_by_test_id('stSelectbox').filter(has=page.get_by_text('人工確認類別：',exact=False)).first
    selector.locator('[data-baseweb="select"]').click(); page.get_by_role('option',name='crack',exact=True).click()
    page.get_by_test_id('stMultiSelect').filter(has=page.get_by_text('已確認整張標註完整的影像',exact=True)).locator('[data-baseweb="select"]').click()
    page.get_by_role('option',name=re.compile('defect-ref.png')).click(); page.keyboard.press('Escape')
    page.get_by_role('textbox',name='完整標註確認者',exact=True).fill('Engineer')
    page.get_by_role('textbox',name='完整檢查與類別確認依據',exact=True).fill('人工確認整張參考圖及crack類別')
    page.get_by_text('我已檢查選取影像的全部內容，真瑕疵框無遺漏，且影像是可用的原始訓練內容',exact=True).click()
    _click(page,'建立訓練交接 ZIP')
    with page.expect_download() as training_download:
        page.get_by_role('button',name='下載訓練交接 ZIP',exact=True).click()
    with zipfile.ZipFile(training_download.value.path()) as archive:
        lineage=json.loads(archive.read('provenance.json'))['lineage']
        assert cam_evidence in {e['id'] for e in lineage['evidence']}
        assert lineage['attributions'][0]['id']=='saved-cam-handoff'
        assert lineage['searches'][0]['id']==search['id']
        assert archive.read('classes.txt').decode().strip()=='crack'
    assert len([j for j in jobs.list(case_id=case['id']) if j['task']=='gradcam'])==1
    page.screenshot(path=str(tmp_path/'cam-dino-upload-handoff.png'),full_page=True)
