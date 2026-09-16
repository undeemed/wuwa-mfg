# SPDX-License-Identifier: Apache-2.0
"""Audit the failed native batch trial, API observer and original ordering tests.

Only numeric evidence is emitted. Captures, vendor code and raw GPU data remain
private. CPU/API success is deliberately separate from GPU completion.
"""
import argparse,collections,hashlib,json,re
from pathlib import Path


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def lines(path):return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','lab','nvapi-interface','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists() and not a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    original=a.base/'native-batch-replay-original-state';failed=a.base/'native-batch-replay-batch-state';api=a.base/'native-api-observer-original-state'
    batch=read(failed/'nr-batch-report.json');recovery=read(a.lab/'native-batch-recovery.json')
    assert recovery['normal_demo_dll_restored'] and recovery['normal_demo_ini_restored'] and recovery['active_markers']==0
    assert not list((failed/'capture').glob('frame-*.json'))
    assert batch['last_completed_evaluation']==4 and batch['requested_kernels']==batch['submitted_kernels']==632 and batch['real_api_calls']==124
    assert not any(batch.get(k) for k in ('failure','guard_disabled','extended_api_seen'))
    boundaries=lines(original/'nr-buffer-probe.jsonl');changed=lines(failed/'nr-buffer-probe.jsonl')
    command=lines(original/'nr-command-probe.jsonl');candidate_commands=lines(failed/'nr-command-probe.jsonl')
    trace=lines(original/'nr-launch-contract.jsonl');candidate=lines(failed/'nr-launch-contract.jsonl')
    for rows in (command,candidate_commands):
        hooks=[r for r in rows if r['kind']=='hook']
        assert len(hooks)==75 and {r['slot'] for r in hooks}==set(range(9,86))-{26,80} and all(r['status']==0 for r in hooks)
        assert not any(r.get('coverage_gap') for r in rows)
    partitions=[]
    for row in batch['sampled_evaluations']:
        frame=row['frame'];assert row['complete'] and row['requested_kernels']==row['submitted_kernels']==158 and row['real_api_calls']==31 and row['fallback_calls']==0
        ends=[];count=0
        for f in row['flushes']:
            assert f['status']==0 and 1<=f['count']<=8;count+=f['count'];ends.append(count)
        assert count==158 and len(ends)==31
        before=[r for r in boundaries if r.get('frame')==frame and r['kind']=='legacy_barrier_call']
        after=[r for r in changed if r.get('frame')==frame and r['kind']=='legacy_barrier_call']
        def signature(r):return {k:r[k] for k in ('before_chain','type','flags','global_uav','inside_launch','inside_nr')}
        assert len(before)==len(after)==20 and list(map(signature,before))==list(map(signature,after))
        required={r['before_chain'] for r in before}
        required|={r['after_chain'] for r in command if r.get('frame')==frame and r['kind']=='command' and not r['inside_launch'] and r['after_chain']>0}
        assert required<=set(ends)
        def contract(rows):return [{k:v for k,v in r.items() if k not in ('chain_cpu_us','status_scope')} for r in rows if r.get('frame')==frame and r['kind']=='launch']
        assert contract(trace)==contract(candidate)
        partitions.append({'frame':frame,'kernels':158,'real_api_calls':31,'batch_end_chain_indices':ends,
            'required_command_and_barrier_boundaries':sorted(required),'all_observed_boundaries_preserved':True,
            'launch_metadata_unchanged':True,'real_api_statuses_all_zero':True,'gpu_completion_established':False})
    events=read(a.lab/'native-batch-driver-events.json');assert len(events)==7 and all(r['Id']==153 and r['provider']=='nvlddmkm' for r in events)
    checks=[]
    for i in range(4):
        m1=read(original/'capture'/f'frame-{i}.json');m2=read(api/'capture'/f'frame-{i}.json')
        assert m1['complete'] and m2['complete'] and m1['gpu_completed'] and m2['gpu_completed']
        assert m1['controls']==m2['controls'] and m1['controls']['DLSSNR.Reset']==int(i==0)
        assert m1['input_replay']['applied'] and m2['input_replay']['applied']
        hashes={}
        for role in ('color','depth','motion','output'):
            first=original/'capture'/m1['resources'][role]['file'];second=api/'capture'/m2['resources'][role]['file']
            assert first.read_bytes()==second.read_bytes();hashes[role]=sha(first)
        checks.append({'frame':i,'inputs_and_prior_history_exact':True,'output_byte_exact':True,'resource_sha256':hashes})
    records=lines(api/'nr-command-probe.jsonl');assert len(records)<32768
    assert [r['event'] for r in records]==list(range(1,len(records)+1))
    assert not any(r.get('coverage_gap') for r in records)
    table={int(i,16):n for n,i in re.findall(r'\{ "([^"]+)", (0x[0-9a-f]+) \}',a.nvapi_interface.read_text())}
    resolutions=[]
    for r in records:
        if r['kind']=='native_api_resolve':resolutions.append({**{k:r[k] for k in ('id','available','wrapped')},'public_name':table.get(r['id'],'not in public interface table')})
    api_rows=[]
    for frame in (1,2,3,4,64,128,256):
        calls=[r for r in records if r['frame']==frame and r['kind']=='native_api_enter'];exits=[r for r in records if r['frame']==frame and r['kind']=='native_api_exit']
        assert len(calls)==len(exits)>0 and all(r['status']==0 for r in exits)
        assert [(r['id'],r['after_chain']) for r in calls]==[(r['id'],r['after_chain']) for r in exits]
        assert all(r['on_evaluation_thread'] and not r['inside_launch'] for r in calls)
        expected=[0,0,1,156,156,157,157] if frame==1 else ([1,1] if frame==2 else [1])
        assert [r['after_chain'] for r in calls]==expected
        api_rows.append({'frame':frame,'calls':[{'after_chain':r['after_chain'],'api':table[r['id']]} for r in calls],'all_statuses_zero':True})
    stress_runs=[]
    for label in ('native-chain-stress','native-chain-overlap'):
        run=read(a.lab/label/'result.json');s=run['test'];assert run['test_completed'] and run['normal_dll_ini_restored'] and s['complete']
        assert s['stress_workload'] and s['distinct_module_handles'] and s['distinct_function_handles'] and len(s['cases'])==48
        assert {(c['elements'],c['kernels'],c['alternating_modules'],c['mixed_block_sizes']) for c in s['cases']}=={(n,k,m,b) for n in (1024,65536,262144) for k in (2,3,8,17) for m in (False,True) for b in (False,True)}
        summaries={}
        for mode in ('serial_barriers','serial_without_barriers','batched'):
            rows=[c[mode] for c in s['cases']];assert all(r['gpu_completed'] and all(v==0 for v in r['launch_statuses']) for r in rows)
            summaries[mode]={'cases':48,'cases_with_mismatches':sum(r['mismatched_elements']>0 for r in rows),'mismatched_elements':sum(r['mismatched_elements'] for r in rows)}
        assert summaries['serial_barriers']['mismatched_elements']==summaries['batched']['mismatched_elements']==0
        stress_runs.append({'label':label,'summary':summaries,'evidence':run})
    overlap=stress_runs[-1]['evidence']['test']['overlap'];assert len(overlap)==30
    overlap_summary={}
    for mode in ('serial_barriers','serial_without_barriers','batched'):
        rows=[r for r in overlap if r['mode']==mode];assert len(rows)==10 and [r['iteration'] for r in rows]==list(range(10))
        assert all(r['gpu_completed'] and r['deadline_cycles']==4000000 and all(v==0 for v in r['launch_statuses']) for r in rows)
        overlap_summary[mode]={'trials':10,'producer_observed_before_deadline':sum(r['producer_observed_before_deadline'] for r in rows)}
    static=read(a.lab/'native-sync-inspection/result.json');assert static['complete'] and not static['native_files_modified'] and len(static['kernels'])==4
    report={'complete':True,'target_achieved':False,'quality_gate_passed':False,'native_runtime_accelerated':False,
        'new_sample_launches':5,'new_game_launches':0,'native_batch_candidate_rejected':True,
        'native_batch':{'partitions':partitions,'runtime_report':batch,'fenced_output_frames':0,'correlated_driver_events':events,
            'recovery':{k:v for k,v in recovery.items() if k!='runtime_report'},'speedup_measured':False},
        'api_observer':{'resolutions':resolutions,'rows':api_rows,'fixed_input_comparisons':checks,'complete_api_coverage_established':False},
        'ordering_tests':stress_runs,'bounded_overlap_summary':overlap_summary,'static_sync_inspection':static,
        'limitations':['API success and CPU recording completion do not prove GPU completion.',
            'Finite original integer tests are not a native-kernel scheduling guarantee.',
            'Native polling loops and the bounded overlap experiment can support a scheduling hypothesis, but do not prove the exact native dependency cycle.',
            'Unwrapped resolved APIs, device/resource operations and internal driver behavior remain outside complete coverage.',
            'No neural model weights, arithmetic, game installation or driver settings were modified.']}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'batch_rejected':True,'api_output_exact_frames':len(checks),'stress_summaries':[r['summary'] for r in stress_runs],'overlap':overlap_summary},indent=2))


if __name__=='__main__':main()
