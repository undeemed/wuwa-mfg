# SPDX-License-Identifier: MIT
"""Collect bounded, private teacher views in the already configured hidden demo.

Uses the existing official-demo runner and fenced texture capture patch. Never
launches or modifies a game. Source scene, DLL and marker are restored finally.
Raw captures stay in the chosen private output directory; do not publish them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


HIDDEN_EXE_SHA256 = 'f262742631d02da935649322f220f0490b114287c98a1d75d60e04557f98a9c6'
# Only camera directions change; the separate north validation view is excluded.
VIEWS = [
    ('northeast', [0, 1.8, 0], [1, 1.8, -1], 'train'),
    ('northwest', [0, 1.8, 0], [-1, 1.8, -1], 'train'),
    ('southeast', [0, 1.8, 0], [1, 1.8, 1], 'train'),
    ('southwest', [0, 1.8, 0], [-1, 1.8, 1], 'train'),
    ('shifted-north', [3, 1.8, 0], [3, 1.8, -1], 'validation'),
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def camera_scene(original, position, target):
    # NVIDIA's scene contains comments, so do not parse it as strict JSON.
    text = original.decode('utf-8')
    camera = {'name': 'Camera0', 'pos': position, 'target': target, 'up': [0, 1, 0]}
    text, count = re.subn(r'"cameras"\s*:\s*\[.*?\]\s*,\s*"active_camera"',
                         '"cameras": [' + json.dumps(camera) + '],\n    "active_camera"',
                         text, count=1, flags=re.S)
    if count != 1:
        raise ValueError('Expected exactly one replaceable cameras block.')
    return text.encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--capture-dll', required=True, type=Path)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--prefix', default='teacher-views')
    args = parser.parse_args()
    if not re.fullmatch('[a-z0-9-]+', args.prefix):
        parser.error('Use a simple lowercase trial prefix.')
    demo = args.demo_dir.resolve()
    output = args.output_dir.resolve()
    scene = demo.parent.parent / 'media' / 'sponza.json'
    runner = Path(__file__).resolve().parents[2] / 'tools' / 'run_neural_demo_trial.py'
    if sha256((demo / 'ngx_dlss_demo.exe').read_bytes()) != HIDDEN_EXE_SHA256:
        raise SystemExit('Refusing a demo executable without the exact hidden-window patch.')
    existing = subprocess.run(['powershell', '-NoProfile', '-Command',
        '@(Get-Process -Name ngx_dlss_demo,Client-Win64-Shipping -ErrorAction SilentlyContinue).Count'],
        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    if existing.stdout.strip() != '0':
        raise SystemExit('Close the game and demo before collecting views.')
    capture_dll = args.capture_dll.read_bytes()
    if sha256(capture_dll) != args.capture_sha256.lower():
        raise SystemExit('Capture DLL hash mismatch.')
    for name in ('nr-kernel-probe.enable', 'nr-kernel-timing.enable',
                 'nr-force-sm89.enable', 'nr-buffer-probe.enable', 'nr-pre-tensor-capture.enable',
                 'nr-pre-pool-capture.enable'):
        if (demo / name).exists():
            raise SystemExit('Disable other demo experiments before collecting views: ' + name)
    capture = demo / 'nr-model-capture'
    if capture.exists():
        raise SystemExit('Archive the existing capture directory first.')
    for name, *_ in VIEWS:
        if (output / 'trials' / (args.prefix + '-' + name)).exists():
            raise SystemExit('Use a fresh prefix; a destination trial already exists.')
    marker = demo / 'nr-model-capture.enable'
    original_marker = marker.read_bytes() if marker.exists() else None
    original_scene = scene.read_bytes()
    original_dll = (demo / 'dxgi.dll').read_bytes()
    # Validate the scene transformation before installing the capture build.
    camera_scene(original_scene, VIEWS[0][1], VIEWS[0][2])
    manifest_path = output / (args.prefix + '-manifest.json')
    if manifest_path.exists():
        raise SystemExit('Refusing to overwrite a collection manifest.')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 1, 'scene_count': 1, 'resolution': [1920, 1080],
        'scene_original_sha256': sha256(original_scene),
        'capture_dll_sha256': sha256(capture_dll), 'views': [],
        'limitation': 'First-reset teacher images in one scene; not temporal or cross-scene validation.'}
    try:
        (demo / 'dxgi.dll').write_bytes(capture_dll)
        marker.write_text('Private fenced teacher capture\n')
        for name, position, target, split in VIEWS:
            label = args.prefix + '-' + name
            changed_scene = camera_scene(original_scene, position, target)
            scene.write_bytes(changed_scene)
            trial = output / 'trials' / label
            print(json.dumps({'starting': label, 'split': split}), flush=True)
            result = subprocess.run([sys.executable, str(runner), label,
                '--demo-dir', str(demo), '--output-dir', str(output),
                '--seconds', '20', '--fps', '60', '--style', '1', '--mask', 'true'],
                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if trial.exists():
                (trial / 'collector-runner.log').write_text(result.stdout + '\n' + result.stderr)
            if result.returncode:
                raise RuntimeError('Demo runner failed: ' + label)
            # Both resolved locations must remain in the explicitly named demo
            # and private output roots before moving this capture directory.
            if not capture.resolve().is_relative_to(demo) or not trial.resolve().is_relative_to(output):
                raise RuntimeError('Capture archive path escaped its intended root.')
            capture.rename(trial / 'capture')
            frame = json.loads((trial / 'capture' / 'frame-0.json').read_text())
            if not (frame['complete'] and frame['gpu_completed'] and frame['evaluate_result'] == 1
                    and frame['controls']['DLSSNR.Reset'] == 1):
                raise RuntimeError('Teacher capture did not complete after its GPU fence.')
            hashes = {}
            for role in ('color', 'output'):
                resource = frame['resources'][role]
                if [resource['width'], resource['height']] != [1920, 1080]:
                    raise RuntimeError('Teacher resolution changed.')
                hashes[role] = sha256((trial / 'capture' / resource['file']).read_bytes())
            item = {'label': label, 'split': split, 'pos': position, 'target': target,
                    'scene_sha256': sha256(changed_scene), 'capture_hashes': hashes}
            manifest['views'].append(item)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(json.dumps({'completed': label, 'capture_hashes': hashes}), flush=True)
    finally:
        scene.write_bytes(original_scene)
        (demo / 'dxgi.dll').write_bytes(original_dll)
        if original_marker is not None:
            marker.write_bytes(original_marker)
        else:
            marker.unlink(missing_ok=True)
        manifest['scene_restored'] = scene.read_bytes() == original_scene
        manifest['dll_restored'] = (demo / 'dxgi.dll').read_bytes() == original_dll
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
