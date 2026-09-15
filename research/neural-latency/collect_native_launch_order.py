# SPDX-License-Identifier: Apache-2.0
"""Compare a capture baseline and launch-order observer in the hidden sample.

No game launch. All captures remain private. Restores the normal sample DLL,
INI and scene bytes; only owned research artifacts are archived after each run.
"""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from collect_demo_views import HIDDEN_EXE_SHA256
from collect_demo_photo_training import sha,read,dump
from evaluate_photo_students import audit_capture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('demo-dir','base','baseline-dll','observer-dll','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--baseline-sha256',required=True);p.add_argument('--observer-sha256',required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[2];demo=a.demo_dir.resolve();base=a.base.resolve()
    assert not base.is_relative_to(repo) and not a.output.resolve().is_relative_to(repo) and not a.output.exists()
    files={'dll':demo/'dxgi.dll','ini':demo/'OptiScaler.ini','scene':demo.parent.parent/'media/sponza.json'}
    originals={n:f.read_bytes() for n,f in files.items()}
    assert sha(files['scene'])=='389da6d234662bb88714b4240384ce63739ff06c7e75ff6014fcd2bab574f547'
    assert sha(files['dll'])=='b3b0857a6d94e4745b42f2bb3beb747c527548ceec0abff11afecf89a1f6a590'
    paths=[];proofs=[]
    for suffix,binary,digest in [('control',a.baseline_dll,a.baseline_sha256),('observed',a.observer_dll,a.observer_sha256)]:
        assert sha(demo/'ngx_dlss_demo.exe')==HIDDEN_EXE_SHA256 and sha(binary)==digest
        active=subprocess.run(['powershell','-NoProfile','-Command','@(Get-Process ngx_dlss_demo,Client-Win64-Shipping -ErrorAction SilentlyContinue).Count'],capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        assert active.stdout.strip()=='0' and not list(demo.glob('nr-*.enable'))
        artifacts=['nr-model-capture','nr-buffer-probe.jsonl','nr-launch-contract.jsonl','nr-kernel-probe.csv']
        assert not any((demo/n).exists() for n in artifacts)
        label='native-launch-order-'+suffix;state=base/(label+'-state');state.mkdir(exist_ok=False)
        markers=['nr-model-capture.enable']+(['nr-kernel-probe.enable','nr-buffer-probe.enable'] if suffix=='observed' else [])
        result={'schema':1,'label':label,'capture_dll_sha256':digest,'capture_archived':False,'first_reset_capture_complete':False,'quality_gate_passed':False}
        try:
            files['dll'].write_bytes(binary.read_bytes())
            for n in markers:(demo/n).write_text('Bounded private launch ordering experiment\n')
            with (state/'runner.log').open('x') as log:
                subprocess.run([sys.executable,str(repo/'tools/run_neural_demo_trial.py'),label,'--demo-dir',str(demo),'--output-dir',str(base),
                    '--seconds','25','--fps','60','--style','1','--mask','true','--isolated-desktop'],check=True,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            frame=read(demo/'nr-model-capture/frame-0.json')
            assert frame['complete'] and frame['gpu_completed'] and frame['evaluate_result']==1 and frame['controls']['DLSSNR.Reset']==1
            result.update(first_reset_capture_complete=True,controls=frame['controls'],capture_hashes={n:sha(demo/'nr-model-capture'/frame['resources'][n]['file']) for n in ('color','output')})
        finally:
            for n,f in files.items():f.write_bytes(originals[n])
            for n in markers:(demo/n).unlink(missing_ok=True)
            for n in artifacts:
                current=demo/n
                if current.exists():
                    assert current.resolve().is_relative_to(demo) and state.resolve().is_relative_to(base)
                    current.rename(state/('capture' if n=='nr-model-capture' else n))
            result['capture_archived']=(state/'capture').exists()
            result['restored']={n:f.read_bytes()==originals[n] for n,f in files.items()};dump(state/'result.json',result)
        _,arrays,_,proof=audit_capture(base,label);del arrays;proofs.append(proof);paths.append(state/'capture')
        print('Completed '+label,flush=True)
    comparisons=[]
    for index in range(4):
        metas=[read(d/f'frame-{index}.json') for d in paths];assert metas[0]['controls']==metas[1]['controls']
        row={'frame':index,'reset':metas[0]['controls']['DLSSNR.Reset'],'resources':{}}
        for role in ('color','depth','motion','output'):
            data=[(d/m['resources'][role]['file']).read_bytes() for d,m in zip(paths,metas)]
            assert metas[0]['resources'][role]==metas[1]['resources'][role]
            row['resources'][role]={'byte_equal':data[0]==data[1],'sha256':[__import__('hashlib').sha256(b).hexdigest() for b in data]}
            if role in ('color','output'):
                arrays=[np.frombuffer(b,dtype='<f2').astype(np.float32) for b in data]
                row['resources'][role].update(max_abs=float(np.max(np.abs(arrays[0]-arrays[1]))),mean_abs=float(np.mean(np.abs(arrays[0]-arrays[1]))))
        comparisons.append(row)
    dump(a.output,{'complete':True,'target_achieved':False,'quality_gate_passed':False,'new_sample_launches':2,'new_game_launches':0,
        'capture_proofs':proofs,'paired_frames':comparisons,'normal_state_restored':all(f.read_bytes()==originals[n] for n,f in files.items()),
        'scope':'Four static fenced frames check observer equality, not temporal/game quality. Observer CPU timings include logging and cannot measure batching benefit.'})
    print('Paired launch-order capture completed.',flush=True)


if __name__=='__main__':main()
