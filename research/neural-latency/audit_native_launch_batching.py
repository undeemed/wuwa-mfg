# SPDX-License-Identifier: Apache-2.0
"""Audit whether an existing metadata trace can justify native launch batching.

This reads private traces only. It neither proposes executable batches nor
changes native calls. Addresses and parameter payloads are never emitted.
"""
import argparse,collections,hashlib,json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):raise ValueError('Use a fresh private output.')
    paths={n:a.trial/n for n in ['nr-buffer-probe.jsonl','nr-launch-contract.jsonl']}
    buffers=[json.loads(s) for s in paths['nr-buffer-probe.jsonl'].read_text().splitlines()]
    contract=[json.loads(s) for s in paths['nr-launch-contract.jsonl'].read_text().splitlines()]
    patch=Path(__file__).with_name('optiscaler-demo-pre-tensor.patch');source=patch.read_text()
    assert 'DemoBufferProbe::BeforeKernel(frame,chain+1);' in source
    assert 'state->frame=frame;state->chain=chain;' in source and 'if(state->barrierEvents>=4096)return;' in source
    assert '(id=ResourceId(b.Transition.pResource))>=0' in source
    hooks=[r for r in buffers if r['kind']=='barrier_hooks'];assert len(hooks)==1
    assert hooks[0]['legacy'] and hooks[0]['enhanced'] and hooks[0]['status']==0
    events=[r for r in buffers if 'event' in r];assert [r['event'] for r in events]==list(range(1,len(events)+1))
    launches=[r for r in contract if r['kind']=='launch'];frames=sorted({r['frame'] for r in launches})
    assert frames==[1,2,3,4,64,128,256]
    cap_reached=len(events)==4096;last_event_frame=events[-1]['frame'];rows=[]
    for frame in frames:
        ls=[r for r in launches if r['frame']==frame]
        assert len(ls)==158 and [r['chain'] for r in ls]==list(range(1,159))
        assert all(r['kernel']==0 and r['status']==0 and r['has_packed_params'] for r in ls)
        bs=[r for r in events if r.get('frame')==frame and r.get('inside_nr') and r['kind']!='barrier_hooks']
        uav=[r for r in bs if r['kind']=='legacy_uav'];assert all(r['global'] for r in uav)
        censored=cap_reached and frame>=last_event_frame
        rows.append({'frame':frame,'successful_single_kernel_calls':len(ls),
            'recorded_barrier_events':len(bs),'global_uav_events':len(uav),
            'associated_chain_markers':[r['before_chain'] for r in uav],
            'event_cap_censors_frame':censored,'sum_chain_cpu_us':sum(r['chain_cpu_us'] for r in ls)})
    complete=[r for r in rows if not r['event_cap_censors_frame']]
    assert len(complete)==6 and all(r['global_uav_events']==16 for r in complete)
    assert all(r['associated_chain_markers']==complete[0]['associated_chain_markers'] for r in complete)
    report={'complete':True,'metadata_only':True,'target_achieved':False,'quality_gate_passed':False,
        'native_calls_modified':False,'batching_safety_established':False,
        'input_sha256':{n:sha(path) for n,path in paths.items()},'observer_source_sha256':sha(patch),
        'registered_buffers':sum(r['kind']=='buffer' for r in buffers),'event_cap':4096,'events_retained':len(events),
        'event_cap_reached':cap_reached,'rows':rows,
        'findings':['Six uncensored sampled frames each contain 158 successful one-kernel calls and 16 logged global UAV barriers with the same associated markers.',
            'The before_chain marker is set immediately before entering each native launch and remains afterward. Events cannot be classified as inside the launch versus between calls from this field.',
            'The observer records transitions only for registered NR-created buffers, plus selected UAV/aliasing events. It does not record the complete D3D12 command stream.',
            'The last sampled frame is censored by the 4096-event cap and cannot establish a complete barrier schedule.',
            'Summed native-call CPU durations are not GPU overhead or a predicted GPU speedup.'],
        'required_evidence_before_batching':['An uncapped bounded evaluation trace distinguishing command-list calls inside native launch APIs from calls between them.',
            'Coverage of all commands and resources relevant to ordering, plus owned copies and verified lifetimes of packed launch arguments.',
            'Documented or directly tested chain ordering/memory semantics, followed by paired output equality and end-to-end GPU timing in the isolated sample.']}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'rows':rows,'batching_safety_established':False},indent=2))


if __name__=='__main__':main()
