# SPDX-License-Identifier: Apache-2.0
"""Collect prepared sources through the pinned, inactive-desktop native teacher.

Preparation and capture use separate manifests so collection can safely follow
completed immutable source entries. Every trial restores the sample before the
next source. No final gameplay trial is manufactured by this image collector.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from collect_demo_views import HIDDEN_EXE_SHA256
from compare_output_grade import read_capture
from prepare_broad_sources import read, write


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(base, label, row):
    state = base/(label+'-state')
    record = read(state/'result.json'); run = read(base/'trials'/label/'result.json')
    assert record['capture_archived'] and record['first_reset_capture_complete'] and all(record['restored'].values())
    assert run['model_1920x1080_confirmed'] and run['local_file_sha256']['ngx_dlss_demo.exe'] == HIDDEN_EXE_SHA256
    assert run['local_file_sha256']['dxgi.dll'] == record['capture_dll_sha256']
    assert record['scene_manifest']['texture_sha256'] == row['texture_sha256']
    views = run['isolated_desktop_observations']
    assert len(views) >= 3 and all(v['separate_from_input_desktop'] and v['input_desktop_unchanged'] for v in views)
    assert any(any(w['title'] == 'NVIDIA NGX DLSS Sample (D3D12)' for w in v['windows']) for v in views)
    assert not any(w['visible_on_private_desktop'] for v in views for w in v['windows'])
    storage = record['capture_storage']
    assert storage['complete'] and storage['logical_bytes_unchanged'] and storage['resource_count'] == 16
    resource_hashes = {(r['frame'], r['role']): r['sha256'] for r in storage['resources']}
    for index in range(4):
        frame = read(state/'capture'/f'frame-{index}.json')
        assert frame['complete'] and frame['gpu_completed'] and frame['evaluate_result'] == 1
        for role, resource in frame['resources'].items():
            assert Path(resource['file']).name == resource['file']
            path = state/'capture'/resource['file']
            assert path.stat().st_size == resource['rows']*resource['row_bytes']
            assert sha(path) == resource_hashes[index, role]
    controls, arrays, hashes = read_capture(state/'capture'); del arrays
    assert controls == record['controls'] and hashes == record['capture_hashes']
    return {'id': row['id'], 'label': label, 'split': row['split'], 'bucket': row['bucket'],
            'source_sha256': row['source_sha256'], 'texture_sha256': row['texture_sha256'],
            'capture_hashes': hashes, 'controls': controls,
            'capture_dll_sha256': record['capture_dll_sha256'],
            'runtime_sha256': run['local_file_sha256'], 'restored': record['restored'],
            'inactive_desktop_observations': len(views), 'all_windows_hidden_off_input_desktop': True,
            'four_fenced_frames_rehashed': True, 'allocated_capture_bytes': storage['allocated_after']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'base', 'demo', 'capture-dll', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--limit', type=int, default=240)
    parser.add_argument('--follow-preparation', action='store_true')
    parser.add_argument('--capture-only', action='store_true', help='End each hidden run when all reference frames are saved.')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    assert 1 <= args.limit <= 240 and not args.output.resolve().is_relative_to(repo)
    assert sha(args.capture_dll) == args.capture_sha256
    expected_controls = read(args.sources.parent/'student-brightness-native/result.json')['controls']
    if args.output.exists():
        report = read(args.output)
        assert report['capture_dll_sha256'] == args.capture_sha256
    else:
        report = {'schema': 1, 'complete': False, 'capture_dll_sha256': args.capture_sha256,
                  'controls': expected_controls, 'cases': [], 'final_gameplay_test_count': 0}
        write(args.output, report)
    waiting_since = None
    while len(report['cases']) < args.limit:
        manifest = read(args.sources/'manifest.json')
        for old, row in zip(report['cases'], manifest['cases']):
            assert old['id'] == row['id'] and old['source_sha256'] == row['source_sha256'] and old['split'] == row['split']
        if len(report['cases']) == len(manifest['cases']):
            if manifest['complete'] or not args.follow_preparation:
                break
            if waiting_since is None:
                waiting_since = time.monotonic()
            if time.monotonic() - waiting_since > 300:
                raise RuntimeError('Preparation has supplied no new source for five minutes; safely resume later.')
            time.sleep(10)
            continue
        waiting_since = None
        row = manifest['cases'][len(report['cases'])]; label = 'broad-v1-'+row['id']
        scene = args.sources/row['id']/'scene'
        assert sha(scene/'manifest.json') == row['scene_manifest_sha256']
        state = args.base/(label+'-state')
        if not state.exists():
            if shutil.disk_usage(args.base).free < 1024**3:
                raise RuntimeError('Stopped before launch: fewer than 1 GiB free.')
            assert sha(args.demo/'ngx_dlss_demo.exe') == HIDDEN_EXE_SHA256
            command = [sys.executable, str(Path(__file__).parent/'collect_demo_image.py'),
                       '--demo-dir', str(args.demo), '--output-dir', str(args.base), '--scene-dir', str(scene),
                       '--capture-dll', str(args.capture_dll), '--capture-sha256', args.capture_sha256,
                       '--label', label, '--compact-capture']
            if args.capture_only:
                command.append('--capture-only')
            with (args.sources/row['id']/'collector.log').open('w') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode:
                raise RuntimeError('Capture stopped; inspect the case collector log. Do not relaunch an incomplete trial.')
        checked = audit(args.base, label, row)
        assert checked['controls'] == expected_controls
        assert checked['runtime_sha256']['nvngx_dlssnr.dll'] == '6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927'
        assert checked['capture_hashes']['color'] not in {v['capture_hashes']['color'] for v in report['cases']}
        report['cases'].append(checked); write(args.output, report)
        print(json.dumps({'collected': len(report['cases']), 'id': row['id'], 'split': row['split']}), flush=True)
    manifest = read(args.sources/'manifest.json')
    report['complete'] = manifest['complete'] and len(report['cases']) == len(manifest['cases']) == 240
    report['prepared_manifest_sha256'] = sha(args.sources/'manifest.json') if manifest['complete'] else None
    write(args.output, report)
    print(json.dumps({'complete': report['complete'], 'collected': len(report['cases'])}), flush=True)


if __name__ == '__main__':
    main()
