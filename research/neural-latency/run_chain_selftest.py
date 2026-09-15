# SPDX-License-Identifier: Apache-2.0
"""Run the original dependency workload inside the exact hidden NVIDIA sample."""
import argparse,json,subprocess,sys
from pathlib import Path
from collect_demo_photo_training import read,sha,dump
from collect_demo_views import HIDDEN_EXE_SHA256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('demo-dir','base','dll','cubin','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--dll-sha256',required=True);p.add_argument('--cubin-sha256',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];demo=a.demo_dir.resolve();base=a.base.resolve()
    assert not base.is_relative_to(repo) and not a.output.resolve().is_relative_to(repo) and not a.output.exists()
    assert sha(demo/'ngx_dlss_demo.exe')==HIDDEN_EXE_SHA256 and sha(a.dll)==a.dll_sha256 and sha(a.cubin)==a.cubin_sha256
    active=subprocess.run(['powershell','-NoProfile','-Command','@(Get-Process ngx_dlss_demo,Client-Win64-Shipping -ErrorAction SilentlyContinue).Count'],capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    assert active.stdout.strip()=='0' and not list(demo.glob('nr-*.enable'))
    artifacts=['nr-chain-selftest.json','nr-chain-selftest.cubin','nr-kernel-probe.csv','nr-launch-contract.jsonl']
    assert not any((demo/n).exists() for n in artifacts)
    normal=(demo/'dxgi.dll').read_bytes();ini=(demo/'OptiScaler.ini').read_bytes()
    assert sha(demo/'dxgi.dll')=='b3b0857a6d94e4745b42f2bb3beb747c527548ceec0abff11afecf89a1f6a590'
    label='native-chain-selftest';a.output.mkdir(parents=True)
    markers=['nr-kernel-probe.enable','nr-chain-selftest.enable']
    try:
        (demo/'dxgi.dll').write_bytes(a.dll.read_bytes());(demo/'nr-chain-selftest.cubin').write_bytes(a.cubin.read_bytes())
        for n in markers:(demo/n).write_text('Original bounded integer dependency workload\n')
        with (a.output/'runner.log').open('x') as log:
            subprocess.run([sys.executable,str(repo/'tools/run_neural_demo_trial.py'),label,'--demo-dir',str(demo),'--output-dir',str(base),
                '--seconds','25','--fps','60','--style','1','--mask','true','--isolated-desktop'],stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    finally:
        (demo/'dxgi.dll').write_bytes(normal);(demo/'OptiScaler.ini').write_bytes(ini)
        for n in markers:(demo/n).unlink(missing_ok=True)
        for n in artifacts:
            if (demo/n).exists():
                assert (demo/n).resolve().is_relative_to(demo)
                (demo/n).rename(a.output/n)
    run=read(base/'trials'/label/'result.json');observations=run['isolated_desktop_observations']
    assert len(observations)>=3 and all(r['separate_from_input_desktop'] and r['input_desktop_unchanged'] for r in observations)
    assert not any(w['visible_on_private_desktop'] for r in observations for w in r['windows'])
    assert run['local_file_sha256']['ngx_dlss_demo.exe']==HIDDEN_EXE_SHA256
    result=read(a.output/'nr-chain-selftest.json')
    report={'complete':True,'test_completed':result['complete'],'test':result,'target_achieved':False,'quality_gate_passed':False,
        'native_runtime_accelerated':False,'new_sample_launches':1,'new_game_launches':0,
        'runtime_sha256':run['local_file_sha256'],'cubin_sha256':a.cubin_sha256,'inactive_desktop_observations':len(observations),
        'all_demo_windows_hidden_and_off_input_desktop':True,'normal_dll_ini_restored':(demo/'dxgi.dll').read_bytes()==normal and (demo/'OptiScaler.ini').read_bytes()==ini,
        'scope':'Original integer kernels on fresh private buffers. No native model kernels are changed; finite tests cannot establish universal NVAPI ordering guarantees.'}
    dump(a.output/'result.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
