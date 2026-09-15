# SPDX-License-Identifier: MIT
"""Collect native final-block input features for a private prepared image scene.

Uses the existing bounded capture build and forced inactive-desktop runner.
Restores the original scene and archives only this invocation's staged assets.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from collect_demo_views import HIDDEN_EXE_SHA256, sha256
from decode_post_inputs import load_prefixes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo-dir', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--capture-dll', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[a-z0-9-]+', args.prefix):
        raise ValueError('Use a simple capture prefix.')
    demo, base, private_scene = (p.resolve() for p in (args.demo_dir, args.base, args.scene_dir))
    repo = Path(__file__).resolve().parents[2]
    if base.is_relative_to(repo) or private_scene.is_relative_to(repo):
        raise ValueError('Inputs and captured data must remain outside the repository.')
    if sha256((demo / 'ngx_dlss_demo.exe').read_bytes()) != HIDDEN_EXE_SHA256:
        raise ValueError('The exact hidden sample executable is required.')
    active = subprocess.run(['powershell', '-NoProfile', '-Command',
        '@(Get-Process -Name ngx_dlss_demo,Client-Win64-Shipping -ErrorAction SilentlyContinue).Count'],
        capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if active.stdout.strip() != '0' or list(demo.glob('nr-*.enable')):
        raise ValueError('Close demo/game and finish active captures first.')
    manifest = json.loads((private_scene / 'manifest.json').read_text())
    description = json.loads((private_scene / 'scene.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        if Path(name).name != name or sha256((private_scene / name).read_bytes()) != digest:
            raise ValueError('Private image assets changed.')
    scene = demo.parent.parent / 'media/sponza.json'
    staged = scene.parent / ('NeuralFeatureImage-' + args.prefix)
    state = base / (args.prefix + '-scene-state')
    if staged.exists() or state.exists() or (base/'trials'/(args.prefix+'-original')).exists():
        raise FileExistsError('Use a fresh prefix.')
    for model in description['models']:
        for key in ('file', 'materials'):
            if Path(model[key]).name != model[key] or model[key] not in manifest['files_sha256']:
                raise ValueError('Image model must refer to its checked local assets.')
            model[key] = staged.name + '/' + model[key]
    original = scene.read_bytes()
    original_dll = (demo/'dxgi.dll').read_bytes()
    original_ini = (demo/'OptiScaler.ini').read_bytes()
    state.mkdir(parents=True)
    (state/'original-scene.json').write_bytes(original)
    report = {'schema': 1, 'prefix': args.prefix, 'scene_manifest': manifest, 'feature_capture_complete': False}
    try:
        staged.mkdir()
        for name in manifest['files_sha256']:
            shutil.copyfile(private_scene/name, staged/name)
        scene.write_text(json.dumps(description, indent=2) + '\n')
        command = [sys.executable, str(Path(__file__).parent/'collect_pre_pool.py'),
                   '--demo-dir', str(demo), '--output-dir', str(base), '--capture-dll', str(args.capture_dll),
                   '--capture-sha256', args.capture_sha256, '--prefix', args.prefix, '--post-inputs', '--single-view']
        with (state/'collector.log').open('w') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        _, _, data, _ = load_prefixes(base/'trials'/(args.prefix+'-original'))
        report.update(feature_capture_complete=True, capture_hashes=data['capture_hashes'],
                      feature_hashes=data['input_sha256'], controls=data['controls'])
    finally:
        scene.write_bytes(original)
        if staged.exists():
            if not staged.resolve().is_relative_to(scene.parent.resolve()) or not state.resolve().is_relative_to(base):
                raise RuntimeError('Asset archive paths escaped the expected roots.')
            staged.rename(state/'assets')
        report['restored'] = {'scene': scene.read_bytes() == original,
                              'dll': (demo/'dxgi.dll').read_bytes() == original_dll,
                              'ini': (demo/'OptiScaler.ini').read_bytes() == original_ini}
        (state/'result.json').write_text(json.dumps(report, indent=2) + '\n')
    if not all(report['restored'].values()):
        raise RuntimeError('Restoration does not match the original files.')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
