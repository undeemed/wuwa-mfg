# SPDX-License-Identifier: Apache-2.0
"""Compare native, traced-original and traced-owned arguments on fixed inputs."""
import argparse,json,re,subprocess,sys,time
from pathlib import Path
import numpy as np
from collect_demo_views import HIDDEN_EXE_SHA256
from collect_demo_photo_training import read,sha,dump
from evaluate_photo_students import audit_capture


def restore_files(demo,normal):
    # A terminated process can briefly retain its image mapping while the
    # graphics driver unwinds. Retry restoration only; never relaunch a trial.
    for attempt in range(41):
        try:
            for name,data in normal.items():(demo/name).write_bytes(data)
            return
        except PermissionError:
            if attempt==40:raise
            time.sleep(.25)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('base','demo-dir','source','dll','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--dll-sha256',required=True)
    p.add_argument('--resume',action='store_true',help='Revalidate completed captures and run only missing modes.')
    p.add_argument('--batch-pair',action='store_true',help='Compare traced original calls against bounded batches using the same DLL.')
    p.add_argument('--observer-only',action='store_true',help='Capture only traced original calls; compare against a separately verified fixed-input reference.')
    p.add_argument('--label-prefix',help='Fresh private artifact label prefix; required for a batch pair.')
    a=p.parse_args()
    assert not (a.batch_pair and a.observer_only)
    assert not (a.batch_pair or a.observer_only) or a.label_prefix
    prefix=a.label_prefix or 'native-input-replay'
    assert re.fullmatch('[a-z0-9-]+',prefix)
    modes=('original',) if a.observer_only else (('original','batch') if a.batch_pair else ('native','original','owned'))
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
        assert run['exit_code_before_cleanup'] is None
        if result['mode']=='batch':
            batch=read(state/'nr-batch-report.json')
            assert batch['complete_through_last_evaluation'] and batch['last_completed_evaluation']>=256
            assert not any(batch.get(k) for k in ('failure','guard_disabled','extended_api_seen'))
            assert batch['requested_kernels']==batch['submitted_kernels'] and batch['real_api_calls']<batch['requested_kernels']
            assert [r['frame'] for r in batch['sampled_evaluations']]==[1,2,3,4,64,128,256]
            for row in batch['sampled_evaluations']:
                assert row['complete'] and row['requested_kernels']==row['submitted_kernels']==158 and row['fallback_calls']==0
                assert row['real_api_calls']==len(row['flushes'])<158
                assert sum(f['count'] for f in row['flushes'])==158
                assert all(f['status']==0 and 1<=f['count']<=batch['batch_limit'] for f in row['flushes'])
            result['batch_report_sha256']=sha(state/'nr-batch-report.json')
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
    launches=0
    for mode in modes:
        assert sha(demo/'ngx_dlss_demo.exe')==HIDDEN_EXE_SHA256 and sha(a.dll)==a.dll_sha256
        active=subprocess.run(['powershell','-NoProfile','-Command','@(Get-Process ngx_dlss_demo,Client-Win64-Shipping -ErrorAction SilentlyContinue).Count'],capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        assert active.stdout.strip()=='0' and not list(demo.glob('nr-*.enable'))
        replay=demo/'nr-input-replay';assert not replay.exists()
        artifacts=['nr-model-capture','nr-buffer-probe.jsonl','nr-kernel-probe.csv','nr-launch-contract.jsonl','nr-command-probe.jsonl','nr-batch-report.json']
        assert not any((demo/n).exists() for n in artifacts)
        label=prefix+'-'+mode;state=base/(label+'-state')
        if state.exists():
            assert a.resume and (state/'result.json').is_file()
            finish(state,label,read(state/'result.json'));print('Revalidated '+label,flush=True);continue
        state.mkdir(exist_ok=False)
        markers=['nr-model-capture.enable','nr-kernel-probe.enable','nr-buffer-probe.enable','nr-input-replay.enable']
        if mode!='native':markers.append('nr-command-probe.enable')
        if mode in ('original','batch'):markers.append('nr-command-original-arguments.enable')
        if mode=='batch':markers.append('nr-batch-kernels.enable')
        result={'schema':1,'mode':mode,'label':label,'capture_archived':False,'quality_gate_passed':False,'source_frames':source_rows}
        try:
            replay.mkdir()
            for name,data in payload.items():(replay/name).write_bytes(data)
            assert all((replay/name).read_bytes()==data for name,data in payload.items())
            (demo/'dxgi.dll').write_bytes(a.dll.read_bytes())
            for n in markers:(demo/n).write_text('Bounded fixed-input native research\n')
            with (state/'runner.log').open('x') as log:
                launches+=1
                subprocess.run([sys.executable,str(repo/'tools/run_neural_demo_trial.py'),label,'--demo-dir',str(demo),'--output-dir',str(base),
                    '--seconds','25','--fps','60','--style','1','--mask','true','--isolated-desktop'],check=True,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        finally:
            restore_error=None
            try:restore_files(demo,normal)
            except OSError as error:restore_error=error
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
            if restore_error is not None:raise restore_error
        finish(state,label,result)
        print('Completed '+label,flush=True)
    comparisons=[]
    for index in range(4):
        inputs=[read(s/'capture'/f'frame-{index}.json') for s in states]
        outputs=[(s/'capture'/m['resources']['output']['file']).read_bytes() for s,m in zip(states,inputs)]
        row={'frame':index,'reset':inputs[0]['controls']['DLSSNR.Reset'],'inputs_and_prior_captured_history_exact':True,'modes':{}}
        original=np.frombuffer(outputs[0],dtype='<f2').astype(np.float32)
        for mode,data in zip(modes[1:],outputs[1:]):
            delta=np.abs(original-np.frombuffer(data,dtype='<f2').astype(np.float32))
            row['modes'][mode]={'output_byte_exact':data==outputs[0],'max_abs':float(delta.max()),'mean_abs':float(delta.mean())}
        comparisons.append(row)
    dump(a.output,{'complete':True,'target_achieved':False,'quality_gate_passed':False,'native_runtime_accelerated':False,
        'new_sample_launches':launches,'new_game_launches':0,'modes':list(modes),'reference_mode':modes[0],
        'replay_source_frames':source_rows,'capture_proofs':proofs,
        'comparisons':comparisons,'normal_state_restored':all((demo/n).read_bytes()==v for n,v in normal.items()),
        'scope':('Fixed input/control sequence from reset; single pass-through observer capture. No output-equivalence or speedup claim without an external reference.' if a.observer_only else
            'Fixed input/control sequence from reset; traced original calls versus bounded batches. Real batch API status is audited separately from queued acceptance. No speedup claim.' if a.batch_pair else
            'Fixed captured input/control sequence from reset; compare native calls, method tracing with original arguments, and tracing with copied arguments. No kernel batching or timing-speedup claim.')})
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
