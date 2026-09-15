# SPDX-License-Identifier: Apache-2.0
"""Summarize bounded launch-scope/barrier metadata without exporting payloads."""
import argparse,collections,json
from pathlib import Path
from collect_demo_photo_training import read,sha,dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('state','paired-capture','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[2]
    assert not a.output.exists() and not a.output.resolve().is_relative_to(repo)
    pair=read(a.paired_capture);assert pair['complete'] and pair['normal_state_restored']
    paths={n:a.state/n for n in ('nr-buffer-probe.jsonl','nr-launch-contract.jsonl')}
    buffer=[json.loads(s) for s in paths['nr-buffer-probe.jsonl'].read_text().splitlines()]
    contract=[json.loads(s) for s in paths['nr-launch-contract.jsonl'].read_text().splitlines()]
    events=[r for r in buffer if 'event' in r];assert [r['event'] for r in events]==list(range(1,len(events)+1))
    assert len(events)<32768,'Trace reached its retention cap.'
    hooks=[r for r in buffer if r['kind']=='barrier_hooks'];assert len(hooks)==1 and hooks[0]['status']==0 and hooks[0]['legacy'] and hooks[0]['enhanced']
    assert not any(r['kind'] in ('barrier_coverage_gap','buffer_omitted') for r in buffer)
    frames=sorted({r['frame'] for r in events if r['kind']=='evaluation_enter'})
    assert frames==[1,2,3,4,64,128,256]
    rows=[]
    for frame in frames:
        ordered=[r for r in events if r['frame']==frame]
        assert ordered[0]['kind']=='evaluation_enter' and ordered[-1]['kind']=='evaluation_exit'
        assert sum(r['kind']=='evaluation_enter' for r in ordered)==sum(r['kind']=='evaluation_exit' for r in ordered)==1
        active=None;entered=[];closed=[];barriers=[]
        for r in ordered:
            if r['kind']=='launch_enter':
                assert active is None and r['inside_launch'];active=r['before_chain'];entered.append(active)
            elif r['kind']=='launch_exit':
                assert active==r['before_chain'] and r['inside_launch'];closed.append(active);active=None
            elif r['kind'] not in ('evaluation_enter','evaluation_exit'):
                assert r['inside_launch']==(active is not None)
                if active is not None:assert r['before_chain']==active
                if r['kind'] in ('legacy_barrier_call','enhanced_barrier_group'):
                    barriers.append({k:r[k] for k in ('kind','before_chain','inside_launch','type') if k in r}|
                        {k:r[k] for k in ('flags','global_uav','count') if k in r})
        assert active is None and entered==closed==list(range(1,159))
        launches=[r for r in contract if r.get('kind')=='launch' and r['frame']==frame]
        assert len(launches)==158 and [r['chain'] for r in launches]==entered
        assert all(r['kernel']==0 and r['status']==0 and r['has_packed_params'] for r in launches)
        rows.append({'frame':frame,'successful_launch_scopes':158,'barriers':barriers,
            'barriers_inside_launch':sum(r['inside_launch'] for r in barriers),
            'barriers_outside_launch':sum(not r['inside_launch'] for r in barriers),
            'barriers_between_launches':sum(not r['inside_launch'] and 0<r['before_chain']<158 for r in barriers),
            'barriers_after_last_launch':sum(not r['inside_launch'] and r['before_chain']==158 for r in barriers),
            'api_counts':dict(collections.Counter(r['api'] for r in launches))})
    report={'complete':True,'target_achieved':False,'quality_gate_passed':False,'native_runtime_accelerated':False,
        'native_calls_reordered':False,'batching_safety_established':False,'events_retained':len(events),'event_cap':32768,'cap_reached':False,
        'input_sha256':{n:sha(f) for n,f in paths.items()},'paired_capture_sha256':sha(a.paired_capture),
        'observer_patch_sha256':sha(Path(__file__).with_name('optiscaler-demo-launch-order.patch')),
        'paired_capture':pair,'matched_input_frames':[r['frame'] for r in pair['paired_frames'] if all(r['resources'][n]['byte_equal'] for n in ('color','depth','motion'))],
        'rows':rows,'limitations':[
            'Observes the intercepted command-list implementations on the calling thread; not proof of all commands, worker threads or implicit driver operations.',
            'Legacy barriers are recorded without restricting resources to the two-buffer registry. Enhanced calls retain group metadata, not resource payloads.',
            'A barrier between single launches cannot simply be moved into or across a combined chain.',
            'Packed-argument lifetime and multi-kernel chain memory semantics remain unverified. No executable batch plan or native speedup is claimed.',
            'Only frames with matching color, depth and motion can test output equality. Later separately launched sample frames can have different temporal inputs.',
            'Observer CPU durations include synchronous logging and are not performance measurements.']}
    dump(a.output,report);print(json.dumps({'events':len(events),'matched_input_frames':report['matched_input_frames'],
        'frames':[{k:v for k,v in r.items() if k!='barriers'} for r in rows]},indent=2))


if __name__=='__main__':main()
