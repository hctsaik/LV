"""Uploaded investigation materials. No uploaded archive is extracted to disk.

Names are relative identifiers, never server paths. Import is a preview until
DatasetRegistry.save ingests verified image bytes into the existing CaseStore.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
import stat
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image
from fp_cases import digest, roi_checked, utc_now

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def _name(name):
    name = str(name).replace('\\', '/')
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or ':' in name or '\x00' in name:
        raise ValueError('檔名必須是安全的相對路徑')
    return str(p)


def expand_uploads(files, *, max_files=5000, max_bytes=250 * 1024 * 1024):
    """Return valid byte entries and individual errors, with aggregate limits."""
    entries, errors, total, count = {}, [], 0, 0
    iterable = files.items() if hasattr(files, 'items') else files

    def add(name, data):
        nonlocal total, count
        count += 1
        total += len(data)
        if count > max_files or total > max_bytes:
            raise ValueError('匯入超過檔數或總大小限制')
        name = _name(name)
        if name in entries:
            raise ValueError('重複相對檔名；請分開匯入或重新命名')
        entries[name] = bytes(data)

    for name, data in iterable:
        try:
            name = _name(name)
            if PurePosixPath(name).suffix.lower() != '.zip':
                add(name, data)
                continue
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    try:
                        entry = _name(info.filename)
                        if stat.S_ISLNK(info.external_attr >> 16):
                            raise ValueError('ZIP 不支援符號連結')
                        if info.file_size + total > max_bytes or count >= max_files:
                            raise ValueError('ZIP 超過檔數或解壓大小限制')
                        if info.flag_bits & 1:
                            raise ValueError('ZIP 不支援加密檔案')
                        add(entry, archive.read(info))
                    except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
                        errors.append({'name': info.filename, 'error': str(exc)})
        except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
            errors.append({'name': str(name), 'error': str(exc)})
    return entries, errors


def parse_annotations(data, *, role, format='yolo', class_map=None,
                      coordinates='normalized', box_format='cxcywh', size=None,
                      field_map=None):
    """Parse one paired image's annotation; five columns are GT only.

    CSV/JSON default columns: class,x,y,w,h,score. field_map maps those
    canonical keys to actual columns. xyxy uses x,y,w,h as x1,y1,x2,y2.
    JSON accepts a row list or {detections:[...]} / {boxes:[...]}.
    """
    if role not in {'prediction', 'ground_truth'}:
        raise ValueError('請選預測或人工標註角色')
    if coordinates not in {'normalized', 'pixels'} or box_format not in {'cxcywh', 'xyxy'}:
        raise ValueError('未知座標格式')
    text = data.decode('utf-8-sig')
    mapping = field_map or {}
    if format == 'yolo':
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            values = line.split()
            expected = 6 if role == 'prediction' else 5
            if len(values) != expected:
                raise ValueError(f'{role} 需要 YOLO {expected} 欄；五欄標註不能當預測')
            rows.append(dict(zip(['class', 'x', 'y', 'w', 'h', 'score'], values)))
        mapping = {}
    elif format == 'csv':
        rows = list(csv.DictReader(io.StringIO(text)))
    elif format == 'json':
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = rows.get('detections', rows.get('boxes'))
        if not isinstance(rows, list):
            raise ValueError('JSON 需要每張圖的框列表')
    else:
        raise ValueError('未知標註格式')
    boxes = []
    for row in rows:
        def val(key):
            return row[mapping.get(key, key)]
        try:
            raw_class = str(val('class'))
            if class_map is None or raw_class not in class_map or not str(class_map[raw_class]).strip():
                raise ValueError(f'請確認類別對應：{raw_class}')
            a, b, c, d = [float(val(k)) for k in ['x', 'y', 'w', 'h']]
            if coordinates == 'pixels':
                if not size or min(size) <= 0:
                    raise ValueError('像素座標需要影像尺寸')
                a, c, b, d = a/size[0], c/size[0], b/size[1], d/size[1]
            box = [a-c/2, b-d/2, a+c/2, b+d/2] if box_format == 'cxcywh' else [a,b,c,d]
            result = {'class': str(class_map[raw_class]), 'box': roi_checked(box)}
            if role == 'prediction':
                score = float(val('score'))
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('預測分數須介於 0 與 1')
                result['score'] = score
            boxes.append(result)
        except (KeyError, TypeError) as exc:
            raise ValueError('標註欄位缺漏，請確認欄位對應') from exc
    return {'status': 'empty' if not boxes else 'provided', 'role': role, 'boxes': boxes,
            'raw_text': text, 'format': format, 'coordinates': coordinates, 'class_map': class_map}


def prepare_import(files, *, annotation_role='prediction', annotation_format='yolo',
                   class_map=None, coordinates='normalized', box_format='cxcywh',
                   field_map=None, pairings=None, annotation_roles=None, max_files=5000, max_bytes=250*1024*1024):
    entries, errors = expand_uploads(files, max_files=max_files, max_bytes=max_bytes)
    images, used, hashes = [], set(), {}
    extension = {'yolo': '.txt', 'csv': '.csv', 'json': '.json'}.get(annotation_format)
    if extension is None:
        raise ValueError('未知標註格式')
    # A CSV/JSON master sheet may explicitly identify relative image names.
    # Never fall back to basename matching: a/a.png and b/a.png are distinct.
    master = {}
    image_field = (field_map or {}).get('image')
    if image_field and annotation_format in {'csv', 'json'}:
        for filename, raw in entries.items():
            if PurePosixPath(filename).suffix.lower() != extension:
                continue
            try:
                text = raw.decode('utf-8-sig')
                records = list(csv.DictReader(io.StringIO(text))) if annotation_format == 'csv' else json.loads(text)
                if isinstance(records, dict):
                    records = records.get('detections', records.get('boxes'))
                if not isinstance(records, list):
                    raise ValueError('主清單需要記錄列表')
                groups = {}
                for record in records:
                    name = _name(record[image_field])
                    if name not in entries or PurePosixPath(name).suffix.lower() not in IMAGE_SUFFIXES:
                        raise ValueError(f'主清單影像無法精確配對：{name}')
                    groups.setdefault(name, []).append(record)
                for name, records in groups.items():
                    if name in master:
                        raise ValueError(f'多個主清單對應同一圖片：{name}')
                for name, records in groups.items():
                    master[name] = (filename, records)
                used.add(filename)
            except (ValueError, UnicodeError, KeyError, TypeError) as exc:
                errors.append({'name': filename, 'error': str(exc)})
    for name, data in entries.items():
        if PurePosixPath(name).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                size = list(im.size)
            sha = digest(data)
            annotation_name = (pairings or {}).get(name, str(PurePosixPath(name).with_suffix(extension)))
            master_entry = master.get(name)
            role = (annotation_roles or {}).get(master_entry[0] if master_entry else annotation_name, annotation_role)
            annotation = {'status': 'missing', 'role': role, 'boxes': []}
            if master_entry or annotation_name in entries:
                annotation_name = master_entry[0] if master_entry else annotation_name
                used.add(annotation_name)
                try:
                    payload = json.dumps(master_entry[1]).encode() if master_entry else entries[annotation_name]
                    annotation = parse_annotations(payload, role=role,
                        format='json' if master_entry else annotation_format, class_map=class_map, coordinates=coordinates,
                        box_format=box_format, size=size, field_map=field_map)
                    if master_entry:
                        annotation['raw_text'] = entries[annotation_name].decode('utf-8-sig')
                        annotation['format'] = annotation_format
                except (ValueError, UnicodeError, TypeError) as exc:
                    annotation = {'status': 'error', 'role': role, 'boxes': [], 'error': str(exc),
                                  'raw_text': entries[annotation_name].decode('utf-8-sig', errors='replace')}
                    errors.append({'name': annotation_name, 'error': str(exc)})
            row = {'id': 'image_' + digest(name.encode())[:16], 'name': name, 'data': data,
                   'sha256': sha, 'size': size, 'annotation_name': annotation_name if annotation_name in entries else None,
                   'annotation': annotation, 'duplicate_of': hashes.get(sha)}
            hashes.setdefault(sha, row['id'])
            images.append(row)
        except (ValueError, OSError, Image.DecompressionBombError) as exc:
            errors.append({'name': name, 'error': str(exc)})
    return {'images': images, 'errors': errors,
            'unpaired': [n for n in entries if n not in used and PurePosixPath(n).suffix.lower() not in IMAGE_SUFFIXES]}


class DatasetRegistry:
    def __init__(self, store):
        self.store = store
        self.root = store.root / 'datasets'
        self.root.mkdir(exist_ok=True)

    def save(self, name, preview, *, kind='reference'):
        if not str(name).strip() or not preview['images']:
            raise ValueError('資料集需要名稱與有效圖片')
        result = {'id': 'dataset_' + uuid.uuid4().hex, 'name': str(name).strip(), 'created_at': utc_now(),
                  'images': [], 'errors': copy.deepcopy(preview.get('errors', []))}
        for row in preview['images']:
            copied = {k: copy.deepcopy(v) for k,v in row.items() if k != 'data'}
            copied['source'] = self.store.ingest(row['data'], row['name'], kind=kind)
            result['images'].append(copied)
        target = self.root / (result['id'] + '.json')
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        os.replace(temporary, target)
        return result

    def list(self):
        return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(self.root.glob('dataset_*.json'))]

    def load(self, dataset_id):
        if not str(dataset_id).startswith('dataset_') or not str(dataset_id)[8:].isalnum():
            raise ValueError('無效資料集 ID')
        return json.loads((self.root / (dataset_id + '.json')).read_text(encoding='utf-8'))


def build_evaluation_manifest(dataset, store, confirmations, *, purpose='case_regression',
                              independent_declared=False, split_basis=''):
    if purpose not in {'case_regression', 'independent'}:
        raise ValueError('未知評估用途')
    if purpose == 'independent' and (not independent_declared or not str(split_basis).strip()):
        raise ValueError('請確認獨立性與切分依據')
    rows, seen = [], set()
    for image in dataset['images']:
        confirmation = confirmations.get(image['id'])
        if not confirmation or not confirmation.get('include', True):
            continue
        if confirmation.get('verified') is not True or confirmation.get('truth') not in {'normal','defect'} or not str(confirmation.get('reviewer','')).strip():
            raise ValueError('每張納入影像需要人工確認真值與覆核者')
        if image['sha256'] in seen:
            raise ValueError('重複圖片不能計入評估；請只選一份')
        seen.add(image['sha256'])
        row = {'id': image['id'], 'path': str(store.image_path(image['source'])), 'sha256': image['sha256'],
               'truth': confirmation['truth'], 'verified': True, 'reviewer': confirmation['reviewer'],
               'group_id': str(confirmation.get('group_id',confirmation.get('group','')))}
        if confirmation.get('annotations_complete') is True:
            annotation = image['annotation']
            if annotation['role'] != 'ground_truth' or annotation['status'] not in {'empty','provided'}:
                raise ValueError('完整框評估需要有效人工標註；缺檔或模型預測不算真值')
            row.update(boxes=copy.deepcopy(annotation['boxes']), annotations_complete=True)
        rows.append(row)
    manifest = {'schema_version': 1, 'purpose': purpose, 'dataset_id': dataset['id'], 'items': rows,
                'independent_declared': bool(independent_declared), 'split_basis': split_basis}
    from fp_investigation import validate_evaluation_manifest
    snapshot, _ = validate_evaluation_manifest(manifest, store)
    return snapshot
