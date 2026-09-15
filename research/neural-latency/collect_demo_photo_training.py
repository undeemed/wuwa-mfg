# SPDX-License-Identifier: MIT
"""Collect a small, photo-disjoint training extension using the hidden sample.

Four new photo identities enter training. The three earlier validation photos
remain validation at both the original emission and calibrated emission 0.1.
Images, rendered targets and generated scene assets stay outside the repository.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

from make_demo_image_scene import write_scene
from prepare_demo_photo_cases import CASES as VALIDATION_SOURCES


TRAINING_SOURCES = [
    {'name': 'snow-cat', 'author': 'Von.grzanka', 'license': 'CC BY-SA 3.0',
     'license_url': 'https://creativecommons.org/licenses/by-sa/3.0/',
     'source': 'https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg',
     'url': 'https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg',
     'source_size': [3000, 2000]},
    {'name': 'bridge', 'author': 'Rich Niewiroski Jr.', 'license': 'CC BY 2.5',
     'license_url': 'https://creativecommons.org/licenses/by/2.5/',
     'source': 'https://commons.wikimedia.org/wiki/File:GoldenGateBridge-001.jpg',
     'url': 'https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg',
     'source_size': [3264, 2448], 'source_sha1': '6b3ae6642a74a9ea00baf8db73c3996583a7e50b'},
    {'name': 'lake', 'author': 'Gorgo', 'license': 'Public domain, released by the author',
     'license_url': 'https://commons.wikimedia.org/wiki/File:Moraine_Lake_17092005.jpg#Licensing',
     'source': 'https://commons.wikimedia.org/wiki/File:Moraine_Lake_17092005.jpg',
     'url': 'https://upload.wikimedia.org/wikipedia/commons/c/c5/Moraine_Lake_17092005.jpg',
     'source_size': [2048, 1536]},
    {'name': 'cupola', 'author': 'NASA/Tracy Caldwell Dyson', 'license': 'Public domain in the United States (NASA)',
     'license_url': 'https://www.nasa.gov/nasa-brand-center/images-and-media/',
     'source': 'https://commons.wikimedia.org/wiki/File:Tracy_Caldwell_Dyson_in_Cupola_ISS.jpg',
     'url': 'https://upload.wikimedia.org/wikipedia/commons/9/95/Tracy_Caldwell_Dyson_in_Cupola_ISS.jpg',
     'source_size': [4288, 2848]},
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def audited_photos(manifest_path, base, expected_controls):
    """Recheck capture, source identity and split before training/evaluation."""
    from evaluate_photo_students import audit_capture
    record = read(manifest_path)
    assert record['complete'] and len(record['cases']) == 10
    assert sha(manifest_path.parent / 'selection.json') == record['selection_sha256']
    selection = read(manifest_path.parent / 'selection.json')
    assert selection['train'] == TRAINING_SOURCES and selection['validation'] == VALIDATION_SOURCES
    expected = {(c['name'], .1): ('train', c) for c in TRAINING_SOURCES}
    expected.update({(c['name'], e): ('validation', c) for c in VALIDATION_SOURCES for e in (.1, 1.)})
    seen, rows, identities = set(), [], {'train': set(), 'validation': set()}
    for row in record['cases']:
        key = (row['name'], row['emittance'])
        assert key in expected and key not in seen
        seen.add(key)
        split, source = expected[key]
        assert row['split'] == split and all(row[k] == v for k, v in source.items())
        assert Path(row['label']).name == row['label']
        controls, images, hashes, proof = audit_capture(base, row['label'])
        assert controls == expected_controls and hashes == row['capture_hashes']
        state = read(base / (row['label'] + '-state') / 'result.json')
        for filename, digest in state['scene_manifest']['files_sha256'].items():
            assert Path(filename).name == filename
            assert sha(base / (row['label'] + '-state') / 'assets' / filename) == digest
        assert state['scene_manifest']['texture_sha256'] == row['texture_sha256']
        assert state['scene_manifest'].get('emittance', 1.) == row['emittance']
        identities[split].add(row['source_sha256'])
        rows.append({**row, 'proof': proof})
        del images
    assert len(identities['train']) == 4 and len(identities['validation']) == 3
    assert not identities['train'] & identities['validation']
    train_hashes = {r['capture_hashes']['color'] for r in rows if r['split'] == 'train'}
    val_hashes = {r['capture_hashes']['color'] for r in rows if r['split'] == 'validation'}
    assert len(train_hashes) == 4 and len(val_hashes) == 6 and not train_hashes & val_hashes
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo-dir', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--validation-photos', type=Path, required=True)
    parser.add_argument('--capture-dll', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    repo = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repo):
        raise ValueError('Keep input images and renderer assets private.')
    old = read(args.validation_photos / 'manifest.json')
    assert [r['name'] for r in old['cases']] == [r['name'] for r in VALIDATION_SOURCES]
    output.mkdir(parents=True, exist_ok=False)
    selection = {'schema': 1, 'train': TRAINING_SOURCES, 'validation': VALIDATION_SOURCES,
                 'emittance': .1, 'previous_validation_retained_at_emittance': 1.,
                 'preprocessing': 'RGB, LANCZOS center-fit to 1920x1080; private derived images only.',
                 'selection_scope': 'Chosen before fitting this extension; not a representative corpus or independent final validation.'}
    dump(output / 'selection.json', selection)
    report = {'schema': 1, 'complete': False, 'selection_sha256': sha(output / 'selection.json'), 'cases': []}
    for source, split in [(c, 'train') for c in TRAINING_SOURCES] + [(c, 'validation') for c in VALIDATION_SOURCES]:
        name = source['name']
        if split == 'train':
            request = Request(source['url'], headers={'User-Agent': 'wuwa-mfg-research/1.0 (https://github.com/undeemed/wuwa-mfg)'})
            with urlopen(request, timeout=30) as response:
                data = response.read(32 * 1024 * 1024 + 1)
            if not 1 <= len(data) <= 32 * 1024 * 1024:
                raise ValueError('Unexpected image size.')
            if source.get('source_sha1'):
                assert hashlib.sha1(data).hexdigest() == source['source_sha1']
            with Image.open(io.BytesIO(data)) as opened:
                assert list(opened.size) == source['source_size']
                image = ImageOps.fit(opened.convert('RGB'), (1920, 1080), method=Image.Resampling.LANCZOS)
            (output / (name + '-original.jpg')).write_bytes(data)
            texture = output / (name + '-1080.png')
            image.save(texture)
            identity = {**source, 'source_bytes': len(data), 'source_sha256': hashlib.sha256(data).hexdigest(),
                        'texture_sha256': sha(texture)}
        else:
            original = next(r for r in old['cases'] if r['name'] == name)
            assert all(original[k] == v for k, v in source.items())
            assert sha(args.validation_photos / (name + '-original.jpg')) == original['source_sha256']
            texture = args.validation_photos / (name + '-1080.png')
            assert sha(texture) == original['texture_sha256']
            identity = {**source, **{k: original[k] for k in ('source_bytes', 'source_sha256', 'texture_sha256')}}
            old_label = 'teacher-photo-' + name
            old_state = read(args.base / (old_label + '-state') / 'result.json')
            report['cases'].append({**identity, 'split': split, 'emittance': 1., 'label': old_label,
                                    'capture_hashes': old_state['capture_hashes']})
        scene = output / (name + '-scene')
        write_scene(scene, texture, emittance=.1)
        label = 'teacher-photo-tenth-' + name
        command = [sys.executable, str(Path(__file__).parent / 'collect_demo_image.py'),
                   '--demo-dir', str(args.demo_dir), '--output-dir', str(args.base), '--scene-dir', str(scene),
                   '--capture-dll', str(args.capture_dll), '--capture-sha256', args.capture_sha256, '--label', label]
        subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        state = read(args.base / (label + '-state') / 'result.json')
        assert all(state['restored'].values()) and state['first_reset_capture_complete']
        report['cases'].append({**identity, 'split': split, 'emittance': .1, 'label': label,
                                'capture_hashes': state['capture_hashes']})
        dump(output / 'manifest.json', report)
        print(json.dumps({'completed': label, 'split': split}), flush=True)
    report['complete'] = True
    dump(output / 'manifest.json', report)
    audited_photos(output / 'manifest.json', args.base, state['controls'])
    print('PASS: four training identities, three validation identities at two emissions; captures restored.', flush=True)


if __name__ == '__main__':
    main()
