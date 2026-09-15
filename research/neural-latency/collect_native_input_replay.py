# SPDX-License-Identifier: Apache-2.0
"""Compare native, traced-original and traced-owned arguments on fixed inputs."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from collect_demo_views import HIDDEN_EXE_SHA256
from collect_demo_photo_training import read,sha,dump
from evaluate_photo_students import audit_capture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('base','demo-dir','source','dll','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--dll-sha256',required=True)
    p.add_argument('--resume',action='store_true',help='Revalidate completed captures and run only missing modes.')
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];demo=a.demo_dir.resolve();base=a.base.resolve();source=a.source.resolve()
    assert not base.is_relative_to(repo) and not a.output.resolve().is_relative_to(repo) and not a.output.exists()
    payload={};source_rows=[]
    for index in range(4):
        meta=read(source/f'frame-{index}.json')
        assert meta['complete'] and meta['gpu_completed'] and meta['evaluate_result']==1 and meta['index']==index
        assert meta['controls']['DLSSNR.Reset']==int(index==0)
        files={}
        for role in ('color','depth','motion'):
            r=meta['resources'][role];name=f'{role}-{index}.raw'
            assert r['file']==name and r['width']==1920 and r['height']==1080
            data=(source/name).read_bytes();assert len(data)==r['row_bytes']*r['rows'] and len(data)<=64*1024*1024
            payload[name]=data;files[role]=sha(source/name)
        payload[f'frame-{index}.json']=(source/f'frame-{index}.json').read_bytes()
        source_rows.append({'index':index,'controls':meta['controls'],'capture_hashes':files})
    assert sum(map(len,payload.values()))<256*1024*1024
    normal={n:(demo/n).read_bytes() for n in ('dxgi.dll','OptiScaler.ini')}
    assert sha(demo/'dxgi.dll')=='b3b0857a6d94e4745b42f2bb3beb747c527548ceec0abff11afecf89a1f6a590'
    proofs=[];states=[]
    def finish(state,label,result):
        assert result['normal_restored'] and result['capture_archived'] and result['source_frames']==source_rows
        run=read(base/'trials'/label/'result.json')
        assert run['local_file_sha256']['dxgi.dll']==a.dll_sha256
        matches=[]
        for index,row in enumerate(source_rows):
            meta=read(state/'capture'/f'frame-{index}.json')
            assert meta['complete'] and meta['gpu_completed'] and meta['evaluate_result']==1 and meta['input_replay']['applied']
            assert meta['controls']==row['controls']
            equal={role:sha(state/'capture'/meta['resources'][role]['file'])==digest for role,digest in row['capture_hashes'].items()}
            matches.append({'frame':index,'inputs_exact':equal})
        result['input_replay_checks']=matches
        first=read(state/'capture/frame-0.json')
        result.update(restored={'dll':result['normal_restored'],'ini':result['normal_restored']},capture_dll_sha256=a.dll_sha256,
            first_reset_capture_complete=first['controls']['DLSSNR.Reset']==1,controls=first['controls'],
            capture_hashes={n:sha(state/'capture'/first['resources'][n]['file']) for n in ('color','output')})
        dump(state/'result.json',result)
        assert all(all(r['inputs_exact'].values()) for r in matches)
        _,arrays,_,proof=audit_capture(base,label);del arrays;proofs.append(proof);states.append(state)
    for mode in ('native','original','owned'):
        assert sha(demo/'ngx_dlss_demo.exe')==HIDDEN_EXE_SHA256 and sha(a.dll)==a.dll_sha256
        active=subprocess.run(['powershell','-NoProfile','-Command','@(Get-Process ngx_dlss_demo,Client-Win64-Shipping -ErrorAction SilentlyContinue).Count'],capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        assert active.stdout.strip()=='0' and not list(demo.glob('nr-*.enable'))
        replay=demo/'nr-input-replay';assert not replay.exists()
        artifacts=['nr-model-capture','nr-buffer-probe.jsonl','nr-kernel-probe.csv','nr-launch-contract.jsonl','nr-command-probe.jsonl']
        assert not any((demo/n).exists() for n in artifacts)
        label='native-input-replay-'+mode;state=base/(label+'-state')
        if state.exists():
            assert a.resume and (state/'result.json').is_file()
            finish(state,label,read(state/'result.json'));print('Revalidated '+label,flush=True);continue
        state.mkdir(exist_ok=False)
        markers=['nr-model-capture.enable','nr-kernel-probe.enable','nr-buffer-probe.enable','nr-input-replay.enable']
        if mode!='native':markers.append('nr-command-probe.enable')
        if mode=='original':markers.append('nr-command-original-arguments.enable')
        result={'schema':1,'mode':mode,'label':label,'capture_archived':False,'quality_gate_passed':False,'source_frames':source_rows}
        try:
            replay.mkdir()
            for name,data in payload.items():(replay/name).write_bytes(data)
            assert all((replay/name).read_bytes()==data for name,data in payload.items())
            (demo/'dxgi.dll').write_bytes(a.dll.read_bytes())
            for n in markers:(demo/n).write_text('Bounded fixed-input native research\n')
            with (state/'runner.log').open('x') as log:
                subprocess.run([sys.executable,str(repo/'tools/run_neural_demo_trial.py'),label,'--demo-dir',str(demo),'--output-dir',str(base),
                    '--seconds','25','--fps','60','--style','1','--mask','true','--isolated-desktop'],check=True,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        finally:
            for n,data in normal.items():(demo/n).write_bytes(data)
            for n in markers:(demo/n).unlink(missing_ok=True)
            for n in artifacts:
                current=demo/n
                if current.exists():
                    assert current.resolve().is_relative_to(demo) and state.resolve().is_relative_to(base)
                    current.rename(state/('capture' if n=='nr-model-capture' else n))
            if replay.exists():
                assert replay.resolve().parent==demo
                for name in payload:
                    file=replay/name;assert file.resolve().parent==replay;file.unlink(missing_ok=True)
                replay.rmdir()
            result['capture_archived']=(state/'capture').exists();result['normal_restored']=all((demo/n).read_bytes()==v for n,v in normal.items())
            dump(state/'result.json',result)
        finish(state,label,result)
        print('Completed '+label,flush=True)
    comparisons=[]
    for index in range(4):
        inputs=[read(s/'capture'/f'frame-{index}.json') for s in states]
        outputs=[(s/'capture'/m['resources']['output']['file']).read_bytes() for s,m in zip(states,inputs)]
        row={'frame':index,'reset':inputs[0]['controls']['DLSSNR.Reset'],'inputs_and_prior_captured_history_exact':True,'modes':{}}
        original=np.frombuffer(outputs[0],dtype='<f2').astype(np.float32)
        for mode,data in zip(('original','owned'),outputs[1:]):
            delta=np.abs(original-np.frombuffer(data,dtype='<f2').astype(np.float32))
            row['modes'][mode]={'output_byte_exact':data==outputs[0],'max_abs':float(delta.max()),'mean_abs':float(delta.mean())}
        comparisons.append(row)
    dump(a.output,{'complete':True,'target_achieved':False,'quality_gate_passed':False,'native_runtime_accelerated':False,
        'new_sample_launches':3,'new_game_launches':0,'replay_source_frames':source_rows,'capture_proofs':proofs,
        'comparisons':comparisons,'normal_state_restored':all((demo/n).read_bytes()==v for n,v in normal.items()),
        'scope':'Fixed captured input/control sequence from reset; compare native calls, method tracing with original arguments, and tracing with copied arguments. No kernel batching or timing-speedup claim.'})
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
