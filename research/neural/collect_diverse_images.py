# SPDX-License-Identifier: MIT
"""Prepare and collect a fixed private image extension through the hidden sample.

Preparation fixes the source identities, crops and split before any native
capture. Collection can resume only completed, revalidated individual trials.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

from make_demo_image_scene import write_scene

CATALOGUE = Path(__file__).parent/'data/diverse-extension.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def validate_catalogue(catalogue):
    cases = catalogue['cases']
    assert catalogue['emittance']==.1 and len(cases)==16
    assert sum(r['split']=='train' for r in cases)==12 and sum(r['split']=='validation' for r in cases)==4
    for key in ['name', 'source', 'source_sha1', 'url']:
        assert len({r[key] for r in cases})==len(cases), 'Repeated source identity.'
    for row in cases:
        assert re.fullmatch('[a-z0-9-]+', row['name'])
        assert urlsplit(row['url']).scheme=='https' and urlsplit(row['url']).hostname=='upload.wikimedia.org'
        assert 0<row['source_bytes']<=32*1024*1024
        assert len(row['source_sha1'])==40 and re.fullmatch('[a-f0-9]+', row['source_sha1'])
        assert row['source_size'][0]>=1920 and row['source_size'][1]>=1080
        assert len(row['centering'])==2 and all(0<=v<=1 for v in row['centering'])


def audit_prepared(manifest_path, previous_manifest, *, allow_partial=False):
    record = read(manifest_path)
    assert record['prepared'] or (allow_partial and not record['complete'])
    assert record['selection_sha256']==sha(manifest_path.parent/'selection.json')
    selection = read(manifest_path.parent/'selection.json')
    validate_catalogue(selection)
    assert selection==read(CATALOGUE) and record['previous_manifest_sha256']==sha(previous_manifest)
    old = read(previous_manifest)
    assert old['complete']
    old_ids = {r['source_sha256'] for r in old['cases']}
    assert len(record['cases'])<=len(selection['cases'])
    assert allow_partial or len(record['cases'])==len(selection['cases'])
    identities, texture_ids = set(), set()
    for row, selected in zip(record['cases'], selection['cases']):
        assert all(row[k]==v for k,v in selected.items())
        assert row['label']==record['prefix']+'-'+row['name'] and re.fullmatch('[a-z0-9-]+', row['label'])
        original = manifest_path.parent/(row['name']+'-original.jpg')
        data = original.read_bytes()
        assert len(data)==row['source_bytes'] and hashlib.sha1(data).hexdigest()==row['source_sha1']
        assert hashlib.sha256(data).hexdigest()==row['source_sha256']
        assert row['source_sha256'] not in old_ids|identities
        identities.add(row['source_sha256'])
        texture = manifest_path.parent/(row['name']+'-1080.png')
        assert sha(texture)==row['texture_sha256'] and row['texture_sha256'] not in texture_ids
        texture_ids.add(row['texture_sha256'])
        scene = manifest_path.parent/(row['name']+'-scene')
        assert sha(scene/'manifest.json')==row['scene_manifest_sha256']
        scene_manifest = read(scene/'manifest.json')
        assert scene_manifest['texture_sha256']==row['texture_sha256'] and scene_manifest['emittance']==.1
        for name, digest in scene_manifest['files_sha256'].items():
            assert Path(name).name==name and sha(scene/name)==digest
    return record


def audited_images(manifest_path, base, expected_controls, previous_manifest):
    from evaluate_photo_students import audit_capture
    record = audit_prepared(manifest_path, previous_manifest)
    assert record['complete']
    seen, rows = set(), []
    for row in record['cases']:
        controls, images, hashes, proof = audit_capture(base, row['label'])
        assert controls==expected_controls and hashes==row['capture_hashes']
        assert hashes['color'] not in seen
        seen.add(hashes['color'])
        state = base/(row['label']+'-state')
        scene_manifest = read(state/'result.json')['scene_manifest']
        assert scene_manifest['texture_sha256']==row['texture_sha256'] and scene_manifest['emittance']==.1
        for name, digest in scene_manifest['files_sha256'].items():
            assert Path(name).name==name and sha(state/'assets'/name)==digest
        rows.append({**row, 'proof': proof})
        del images
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'collect'])
    parser.add_argument('--previous-photos', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', default='teacher-diverse')
    parser.add_argument('--resume-preparation', action='store_true', help='Revalidate and continue an incomplete preparation.')
    parser.add_argument('--demo-dir', type=Path)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--capture-dll', type=Path)
    parser.add_argument('--capture-sha256')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(repo) or not re.fullmatch('[a-z0-9-]+', args.prefix):
        raise ValueError('Use a private output directory and simple prefix.')
    manifest_path = args.output/'manifest.json'
    if args.resume_preparation and args.mode!='prepare':
        raise ValueError('Preparation resume is only valid in prepare mode.')
    if args.mode=='prepare':
        if args.output.exists() and not args.resume_preparation:
            raise FileExistsError('Use a fresh private preparation directory.')
        catalogue = read(CATALOGUE)
        validate_catalogue(catalogue)
        old = read(args.previous_photos)
        assert old['complete']
        old_ids = {r['source_sha256'] for r in old['cases']}
        old_urls = {r['source'] for r in old['cases']}
        assert not old_urls & {r['source'] for r in catalogue['cases']}
        if args.resume_preparation:
            record = audit_prepared(manifest_path, args.previous_photos, allow_partial=True)
            assert not record['prepared'] and not record['complete'] and record['prefix']==args.prefix
            old_ids.update(r['source_sha256'] for r in record['cases'])
        else:
            args.output.mkdir(parents=True)
            (args.output/'selection.json').write_bytes(CATALOGUE.read_bytes())
            record = {'schema': 1, 'prepared': False, 'complete': False, 'prefix': args.prefix,
                      'selection_sha256': sha(args.output/'selection.json'),
                      'previous_manifest_sha256': sha(args.previous_photos), 'cases': []}
            dump(manifest_path, record)
        for row in catalogue['cases'][len(record['cases']):]:
            if any((args.output/(row['name']+suffix)).exists() for suffix in ['-original.jpg', '-1080.png', '-scene']):
                raise FileExistsError('Unrecorded preparation artifacts require inspection before resuming.')
            time.sleep(3)  # Respect the source host's request rate, including resumed work.
            request = Request(row['url'], headers={'User-Agent': 'wuwa-mfg-research/1.0 (https://github.com/undeemed/wuwa-mfg)'})
            with urlopen(request, timeout=30) as response:
                data = response.read(32*1024*1024+1)
            assert len(data)==row['source_bytes'] and hashlib.sha1(data).hexdigest()==row['source_sha1']
            digest = hashlib.sha256(data).hexdigest()
            assert digest not in old_ids
            old_ids.add(digest)
            with Image.open(io.BytesIO(data)) as opened:
                assert list(opened.size)==row['source_size']
                image = ImageOps.fit(opened.convert('RGB'), (1920, 1080), method=Image.Resampling.LANCZOS,
                                     centering=tuple(row['centering']))
            (args.output/(row['name']+'-original.jpg')).write_bytes(data)
            texture = args.output/(row['name']+'-1080.png')
            image.save(texture)
            scene = args.output/(row['name']+'-scene')
            write_scene(scene, texture, emittance=.1)
            record['cases'].append({**row, 'label': args.prefix+'-'+row['name'], 'source_sha256': digest,
                                    'texture_sha256': sha(texture), 'scene_manifest_sha256': sha(scene/'manifest.json')})
            dump(manifest_path, record)
            print(json.dumps({'prepared': row['name'], 'split': row['split']}), flush=True)
        record['prepared']=True
        dump(manifest_path, record)
        audit_prepared(manifest_path, args.previous_photos)
        return
    if not all([args.demo_dir, args.base, args.capture_dll, args.capture_sha256]):
        raise ValueError('Collection requires the existing demo, private trial root and checked capture build.')
    record = audit_prepared(manifest_path, args.previous_photos)
    assert record['prefix']==args.prefix
    from evaluate_photo_students import audit_capture
    expected_controls = None
    for row in record['cases']:
        if 'capture_hashes' not in row:
            command = [sys.executable, str(Path(__file__).parent/'collect_demo_image.py'),
                       '--demo-dir', str(args.demo_dir), '--output-dir', str(args.base),
                       '--scene-dir', str(args.output/(row['name']+'-scene')),
                       '--capture-dll', str(args.capture_dll), '--capture-sha256', args.capture_sha256,
                       '--label', row['label']]
            with (args.output/(row['name']+'-collector.log')).open('w') as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
        controls, images, hashes, _ = audit_capture(args.base, row['label'])
        if expected_controls is None:
            expected_controls=controls
        assert controls==expected_controls
        if 'capture_hashes' in row:
            assert row['capture_hashes']==hashes
        row['capture_hashes']=hashes
        dump(manifest_path, record)
        del images
        print(json.dumps({'collected': row['name'], 'split': row['split']}), flush=True)
    record['complete']=True
    dump(manifest_path, record)
    audited_images(manifest_path, args.base, expected_controls, args.previous_photos)
    print('Completed and audited twelve training and four validation image captures.', flush=True)


if __name__=='__main__':
    main()
