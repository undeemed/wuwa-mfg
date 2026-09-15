# SPDX-License-Identifier: MIT
"""Collect output-kernel scalars and paired textures in one hidden demo run.

Requires a locally built combined demo capture patch. All captured data remains
private. The demo DLL and markers are restored; no game is launched or modified.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from collect_demo_views import HIDDEN_EXE_SHA256, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--capture-dll', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--label', default='native-output-contract')
    args = parser.parse_args()
    if not re.fullmatch('[a-z0-9-]+',args.label):
        parser.error('Use a simple lowercase trial label.')
    demo, output = args.demo_dir.resolve(),args.output_dir.resolve()
    repo = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repo):
        raise ValueError('Private captures must remain outside the repository.')
    if sha256((demo/'ngx_dlss_demo.exe').read_bytes()) != HIDDEN_EXE_SHA256:
        raise ValueError('Requires the exact verified hidden-window executable.')
    running = subprocess.run(['powershell','-NoProfile','-Command',
        '@(Get-Process ngx_dlss_demo,Client-Win64-Shipping -ErrorAction SilentlyContinue).Count'],
        capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    if running.stdout.strip() != '0':
        raise RuntimeError('Game and demo must be closed before a controlled capture.')
    markers = ('nr-kernel-probe.enable','nr-model-capture.enable')
    artifacts = {'nr-model-capture':'capture','nr-kernel-probe.csv':'nr-kernel-probe.csv',
                 'nr-launch-contract.jsonl':'nr-launch-contract.jsonl'}
    if list(demo.glob('nr-*.enable')):
        raise FileExistsError('Disable previous demo experiments before this run.')
    for name in artifacts:
        if (demo/name).exists():
            raise FileExistsError('Archive previous diagnostic output: '+name)
    trial = output/'trials'/args.label
    if trial.exists():
        raise FileExistsError(trial)
    capture_dll = args.capture_dll.read_bytes()
    if sha256(capture_dll) != args.capture_sha256.lower():
        raise ValueError('Capture DLL hash mismatch.')
    original = (demo/'dxgi.dll').read_bytes()
    def archive():
        if not trial.exists():
            return
        for name,target_name in artifacts.items():
            source,target = demo/name,trial/target_name
            if not source.resolve().is_relative_to(demo) or not target.resolve().is_relative_to(output):
                raise ValueError('Archive path escaped its intended root.')
            if source.exists():
                if target.exists():raise FileExistsError(target)
                source.rename(target)
    try:
        (demo/'dxgi.dll').write_bytes(capture_dll)
        for name in markers:(demo/name).write_text('Private output-contract observation\n')
        run = subprocess.run([sys.executable,str(repo/'tools/run_neural_demo_trial.py'),args.label,
            '--demo-dir',str(demo),'--output-dir',str(output),'--seconds','25','--fps','60',
            '--mask','true','--style','1'],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
        if trial.exists():(trial/'collector-runner.log').write_text(run.stdout+'\n'+run.stderr)
        archive()
        if run.returncode:raise RuntimeError('Hidden demo runner failed.')
        frame = json.loads((trial/'capture/frame-0.json').read_text())
        if not (frame['complete'] and frame['gpu_completed'] and frame['evaluate_result']==1
                and frame['controls']['DLSSNR.Reset']==1):
            raise ValueError('First-reset capture did not complete.')
        records = [json.loads(line) for line in (trial/'nr-launch-contract.jsonl').read_text().splitlines()]
        records = [r for r in records if r.get('kind')=='launch' and r.get('frame')==1
                   and 'output_scalar_candidate' in r]
        if len(records)!=2 or any(r['status']!=0 for r in records):
            raise ValueError('Requires both successful first-frame output-kernel contracts.')
        print(json.dumps({'label':args.label,'capture_dll_sha256':sha256(capture_dll),
                          'output_contracts':records},indent=2))
    finally:
        (demo/'dxgi.dll').write_bytes(original)
        for name in markers:(demo/name).unlink(missing_ok=True)
        archive()
        if (demo/'dxgi.dll').read_bytes()!=original:
            raise RuntimeError('Demo DLL restoration verification failed.')


if __name__ == '__main__':
    main()
