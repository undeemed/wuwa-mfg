# SPDX-License-Identifier: MIT
"""Collect one private image-plane trial through the exact hidden sample.

Uses the existing fenced capture DLL and published trial runner. Restores the
original sample scene and DLL on success or failure; the runner restores its
INI. No game process or graphics-driver settings are modified.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from collect_demo_views import HIDDEN_EXE_SHA256, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--capture-dll', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--compact-capture', action='store_true', help='Losslessly compress only this new capture after the sample exits and restoration succeeds.')
    parser.add_argument('--capture-only', action='store_true', help='Stop the hidden trial after four fenced frames; do not benchmark its latency.')
    args = parser.parse_args()
    if not re.fullmatch('[a-z0-9-]+', args.label):
        raise ValueError('Use a simple trial label.')
    repo = Path(__file__).resolve().parents[2]
    demo, output, scene_dir = (p.resolve() for p in (args.demo_dir, args.output_dir, args.scene_dir))
    if (output / 'pause-image-collection.enable').exists():
        raise SystemExit('Collection paused before launch by the private pause marker.')
    if output.is_relative_to(repo) or scene_dir.is_relative_to(repo):
        raise ValueError('Keep generated scenes and captures outside the repository.')
    if sha256((demo / 'ngx_dlss_demo.exe').read_bytes()) != HIDDEN_EXE_SHA256:
        raise ValueError('Refusing an executable without the exact hidden-window patch.')
    active = subprocess.run(['powershell', '-NoProfile', '-Command',
        '@(Get-Process -Name ngx_dlss_demo,Client-Win64-Shipping -ErrorAction SilentlyContinue).Count'],
        capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if active.stdout.strip() != '0':
        raise ValueError('Close the game and demo before collecting an image.')
    if list(demo.glob('nr-*.enable')) or (demo / 'nr-model-capture').exists():
        raise ValueError('Other active research markers or unarchived captures are present.')
    template = (scene_dir / 'scene.json').read_bytes()
    scene_description = json.loads(template)
    manifest = json.loads((scene_dir / 'manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        if Path(name).name != name or sha256((scene_dir / name).read_bytes()) != digest:
            raise ValueError('Generated scene content changed.')
    for model in scene_description['models']:
        for key in ('file', 'materials'):
            if not (scene_dir / model[key]).resolve(strict=True).is_relative_to(scene_dir):
                raise ValueError('Image scene references an asset outside its private directory.')
    capture_dll = args.capture_dll.read_bytes()
    if sha256(capture_dll) != args.capture_sha256.lower():
        raise ValueError('Capture DLL hash mismatch.')
    trial = output / 'trials' / args.label
    state = output / (args.label + '-state')
    if trial.exists() or state.exists():
        raise FileExistsError('Use a fresh label for this private trial.')
    scene = demo.parent.parent / 'media/sponza.json'
    staged = scene.parent / ('NeuralResearchImage-' + args.label)
    if staged.exists():
        raise FileExistsError('A staged image directory already exists.')
    for model in scene_description['models']:
        for key in ('file', 'materials'):
            model[key] = staged.name + '/' + Path(model[key]).name
    template = (json.dumps(scene_description, indent=2) + '\n').encode()
    dll = demo / 'dxgi.dll'
    ini = demo / 'OptiScaler.ini'
    original_scene, original_dll, original_ini = (p.read_bytes() for p in (scene, dll, ini))
    state.mkdir(parents=True)
    (state / 'original-scene.json').write_bytes(original_scene)
    marker = demo / 'nr-model-capture.enable'
    capture = demo / 'nr-model-capture'
    report = {'schema': 1, 'label': args.label, 'scene_manifest': manifest,
              'capture_dll_sha256': sha256(capture_dll), 'capture_archived': False,
              'first_reset_capture_complete': False, 'quality_gate_passed': False}
    try:
        staged.mkdir()
        for name in manifest['files_sha256']:
            shutil.copyfile(scene_dir / name, staged / name)
        dll.write_bytes(capture_dll)
        scene.write_bytes(template)
        marker.write_text('Private image-plane teacher capture\n')
        runner = repo / 'tools/run_neural_demo_trial.py'
        command = [sys.executable, str(runner), args.label,
            '--demo-dir', str(demo), '--output-dir', str(output), '--seconds', '20',
            '--fps', '60', '--style', '1', '--mask', 'true', '--isolated-desktop']
        if args.capture_only:
            command.append('--capture-only')
        result = subprocess.run(command, capture_output=True,
            text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        (state / 'runner.log').write_text(result.stdout + '\n' + result.stderr)
        if result.returncode:
            raise RuntimeError('Hidden trial runner failed; inspect the private runner log.')
        frame_path = capture / 'frame-0.json'
        if not frame_path.exists():
            report['failure'] = 'No neural capture; inspect the private trial desktop observations for an error dialog.'
            raise RuntimeError(report['failure'])
        frame = json.loads(frame_path.read_text())
        if not (frame['complete'] and frame['gpu_completed'] and frame['evaluate_result'] == 1
                and frame['controls']['DLSSNR.Reset'] == 1):
            raise RuntimeError('The first-reset GPU capture did not complete.')
        report['first_reset_capture_complete'] = True
        report['controls'] = frame['controls']
        report['capture_hashes'] = {role: sha256((capture / frame['resources'][role]['file']).read_bytes())
                                    for role in ('color', 'output')}
    finally:
        # The trial runner waits for its owned process to exit before returning.
        scene.write_bytes(original_scene)
        dll.write_bytes(original_dll)
        marker.unlink(missing_ok=True)
        if staged.exists():
            if not staged.resolve().is_relative_to(scene.parent.resolve()) or not state.resolve().is_relative_to(output):
                raise RuntimeError('Asset archive escaped its intended private root.')
            staged.rename(state / 'assets')
        if capture.exists():
            if not capture.resolve().is_relative_to(demo) or not state.resolve().is_relative_to(output):
                raise RuntimeError('Capture archive escaped its intended private root.')
            capture.rename(state / 'capture')
            report['capture_archived'] = True
        report['restored'] = {name: p.read_bytes() == data for name, p, data in
            [('scene', scene, original_scene), ('dll', dll, original_dll), ('ini', ini, original_ini)]}
        (state / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    if not all(report['restored'].values()):
        raise RuntimeError('Sample restoration did not match its original bytes.')
    if args.compact_capture:
        from compact_teacher_capture import compact_capture
        report['capture_storage']=compact_capture(state/'capture',output)
        (state/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
