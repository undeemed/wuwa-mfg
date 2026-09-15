# SPDX-License-Identifier: MIT
"""Capture private first-block pooled features in two hidden NVIDIA demo views.

Requires the combined optiscaler-demo-pre-tensor.patch capture build. This tool
never launches a game. Raw tensors and textures must remain outside this repo.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from collect_demo_views import HIDDEN_EXE_SHA256, camera_scene, sha256


MARKERS = ('nr-model-capture.enable', 'nr-kernel-probe.enable',
           'nr-buffer-probe.enable', 'nr-pre-pool-capture.enable')
ARTIFACTS = {'nr-model-capture': 'capture', 'nr-pre-pool-capture': 'pre-pool',
             'nr-buffer-probe.jsonl': 'nr-buffer-probe.jsonl',
             'nr-kernel-probe.csv': 'nr-kernel-probe.csv',
             'nr-launch-contract.jsonl': 'nr-launch-contract.jsonl'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--capture-dll', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--prefix', default='native-pre-pool')
    args = parser.parse_args()
    import re
    if not re.fullmatch('[a-z0-9-]+', args.prefix):
        parser.error('Use a simple lowercase trial prefix.')
    demo, output = args.demo_dir.resolve(), args.output_dir.resolve()
    repo = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repo):
        raise ValueError('Private capture output must be outside this repository.')
    if sha256((demo / 'ngx_dlss_demo.exe').read_bytes()) != HIDDEN_EXE_SHA256:
        raise ValueError('Requires the exact verified hidden-window executable.')
    process_check = subprocess.run(['powershell', '-NoProfile', '-Command',
        '@(Get-Process -Name ngx_dlss_demo,Client-Win64-Shipping -ErrorAction SilentlyContinue).Count'],
        capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if process_check.stdout.strip() != '0':
        raise RuntimeError('Demo and game must be closed before this capture.')
    capture_dll = args.capture_dll.read_bytes()
    if sha256(capture_dll) != args.capture_sha256.lower():
        raise ValueError('Capture DLL hash mismatch.')
    for name in (*MARKERS, *ARTIFACTS, 'nr-kernel-timing.enable',
                 'nr-force-sm89.enable', 'nr-pre-tensor-capture.enable', 'nr-pre-tensor-capture'):
        if (demo / name).exists():
            raise FileExistsError('Archive or disable previous experiment: ' + name)
    views = [('original', None), ('west', [-1, 1.8, 0])]
    for name, _ in views:
        if (output / 'trials' / (args.prefix + '-' + name)).exists():
            raise FileExistsError('Use a fresh trial prefix.')
    scene = demo.parent.parent / 'media' / 'sponza.json'
    original_scene, original_dll = scene.read_bytes(), (demo / 'dxgi.dll').read_bytes()
    camera_scene(original_scene, [0, 1.8, 0], views[1][1])
    manifest_path = output / (args.prefix + '-manifest.json')
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 1, 'capture_dll_sha256': sha256(capture_dll),
                'original_dll_sha256': sha256(original_dll),
                'original_scene_sha256': sha256(original_scene), 'views': []}
    trial = None
    def archive():
        if trial is None or not trial.exists():
            return
        for source_name, target_name in ARTIFACTS.items():
            source, target = demo / source_name, trial / target_name
            if not source.resolve().is_relative_to(demo) or not target.resolve().is_relative_to(output):
                raise ValueError('Archive paths escaped their intended roots.')
            if source.exists():
                if target.exists():
                    raise FileExistsError(target)
                source.rename(target)
    try:
        (demo / 'dxgi.dll').write_bytes(capture_dll)
        for name in MARKERS:
            (demo / name).write_text('One-shot private pooled-tensor capture\n')
        for name, direction in views:
            scene.write_bytes(original_scene if direction is None else
                              camera_scene(original_scene, [0, 1.8, 0], direction))
            label = args.prefix + '-' + name
            trial = output / 'trials' / label
            print(json.dumps({'starting': label}), flush=True)
            run = subprocess.run([sys.executable, str(repo / 'tools/run_neural_demo_trial.py'),
                label, '--demo-dir', str(demo), '--output-dir', str(output),
                '--seconds', '25', '--fps', '60', '--mask', 'true', '--style', '1'],
                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if trial.exists():
                (trial / 'collector-runner.log').write_text(run.stdout + '\n' + run.stderr)
            archive()
            if run.returncode:
                raise RuntimeError('Hidden demo runner failed: ' + label)
            meta = json.loads((trial / 'pre-pool/metadata.json').read_text())
            if not (meta['complete'] and meta['gpu_completed'] and meta['frame'] == 1
                    and meta['noise_counter'] == 0 and meta['tensor_kind'] == 'pooled'
                    and [meta['width'], meta['height']] == [1920, 1080]
                    and [meta['pool_width'], meta['pool_height']] == [960, 576]
                    and meta['bytes'] == 960*576*32 and meta['file'] == 'pre-pool.raw'):
                raise ValueError('Pooled capture contract failed.')
            raw = (trial / 'pre-pool/pre-pool.raw').read_bytes()
            if len(raw) != meta['bytes']:
                raise ValueError('Truncated pooled tensor.')
            frame = json.loads((trial / 'capture/frame-0.json').read_text())
            if not (frame['complete'] and frame['gpu_completed'] and frame['evaluate_result'] == 1
                    and frame['controls']['DLSSNR.Reset'] == 1):
                raise ValueError('Paired first-reset textures are incomplete.')
            record = {'view': name, 'label': label, 'pooled_sha256': sha256(raw), 'capture_hashes': {}}
            for role in ('color', 'output'):
                resource = frame['resources'][role]
                if [resource['width'], resource['height']] != [1920, 1080]:
                    raise ValueError('Paired texture extent changed.')
                filename = Path(resource['file'])
                if filename.name != str(filename):
                    raise ValueError('Texture filename must be a basename.')
                record['capture_hashes'][role] = sha256((trial / 'capture' / filename).read_bytes())
            manifest['views'].append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
            print(json.dumps(record), flush=True)
    finally:
        # Restore configuration before archiving diagnostics, even on failure.
        scene.write_bytes(original_scene)
        (demo / 'dxgi.dll').write_bytes(original_dll)
        for name in MARKERS:
            (demo / name).unlink(missing_ok=True)
        archive()
        if scene.read_bytes() != original_scene or (demo / 'dxgi.dll').read_bytes() != original_dll:
            raise RuntimeError('Demo restoration verification failed.')


if __name__ == '__main__':
    main()
